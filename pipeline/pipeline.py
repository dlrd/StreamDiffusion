import time
import logging
from typing import List, Optional, Union, Any, Dict, Tuple, Literal
from collections import OrderedDict

import numpy as np
import PIL.Image
import torch
from diffusers import LCMScheduler, StableDiffusionPipeline, TCDScheduler, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

from .image_filter import SimilarImageFilter
from .attention_processors import update_cache_after_unet
from functools import lru_cache


@lru_cache(maxsize=32)
def _get_latent_dimensions(height: int, width: int, scale_factor: int) -> Tuple[int, int]:
    return (int(height // scale_factor), int(width // scale_factor))


class StreamDiffusion:
    def __init__(
        self,
        pipe: StableDiffusionPipeline,
        t_index_list: List[int],
        torch_dtype: torch.dtype = torch.float16,
        width: int = 512,
        height: int = 512,
        do_add_noise: bool = True,
        use_denoising_batch: bool = True,
        frame_buffer_size: int = 1,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    ) -> None:
        self.device = pipe.device
        self.dtype = torch_dtype
        self.generator = None

        self.height = height
        self.width = width

        self.latent_height, self.latent_width = _get_latent_dimensions(
            height, width, pipe.vae_scale_factor
        )

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)

        self.cfg_type = cfg_type

        # Latent feedback: blend current denoised latent with previous for temporal smoothing.
        self.latent_feedback_strength = 0.0  # 0.0 = disabled, 0.1-0.4 recommended
        self._prev_latent = None

        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * frame_buffer_size
            if self.cfg_type == "initialize":
                self.trt_unet_batch_size = (
                    self.denoising_steps_num + 1
                ) * self.frame_bff_size
            elif self.cfg_type == "full":
                self.trt_unet_batch_size = (
                    2 * self.denoising_steps_num * self.frame_bff_size
                )
            else:
                self.trt_unet_batch_size = self.denoising_steps_num * frame_buffer_size
        else:
            self.trt_unet_batch_size = self.frame_bff_size
            self.batch_size = frame_buffer_size

        self.t_list = t_index_list

        self.do_add_noise = do_add_noise
        self.use_denoising_batch = use_denoising_batch

        # SSF (Stochastic Similarity Filter)
        self.similar_image_filter = True
        self.similar_filter = SimilarImageFilter(threshold=0.98, max_skip_frame=3)
        self.prev_image_result = None

        self._ssf_frames_processed = 0
        self._ssf_frames_skipped = 0

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        # Default LCM; configure_scheduler() switches for turbo/hyper.
        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler_type = "LCM"
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        # Do NOT cache vae.config.scaling_factor: TinyVAE swap changes it from 0.18215 to 1.0.

        # Cache normalized ControlNet model list (stable across frames).
        self._cached_controlnet_model = None
        self._cached_controlnet_model_list = None

        # 0-D cn_scale tensors per ControlNet index, refilled per frame.
        self._cn_scale_buf = {}

        self.inference_time_ema = 0

        self.last_internal_timings = {}
        self.enable_profiling = False

        # Scheduler coefficients LRU cache (bounded to prevent unbounded GPU growth).
        self._scheduler_coeffs_cache = OrderedDict()
        self._max_scheduler_cache_size = 16
        self._cache_hits = 0
        self._cache_misses = 0

    def configure_scheduler(self, model_type: str = "default", eta: float = 1.0):
        """Pick the scheduler matching the model type (turbo / hyper / default)."""
        if model_type == "turbo":
            self.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing",
            )
            self.scheduler_type = "Euler"
            logging.info(f"[SD 1.5 Scheduler] Using EulerDiscreteScheduler (timestep_spacing='trailing') for SD Turbo")

        elif model_type == "hyper":
            # Hyper-SD requires timestep_spacing='trailing' for quality.
            self.scheduler = TCDScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing",
            )
            if hasattr(self.scheduler, 'set_eta'):
                self.scheduler.set_eta(eta)
            self.scheduler_type = "TCD"
            logging.info(f"[SD 1.5 Scheduler] Using TCDScheduler (eta={eta}, timestep_spacing='trailing') for Hyper-SD 1.5")

        else:
            self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            self.scheduler_type = "LCM"
            logging.info(f"[SD 1.5 Scheduler] Using default LCMScheduler")

    def _compute_scheduler_coefficients(self, num_inference_steps: int):
        cache_key = (tuple(self.t_list), num_inference_steps)

        if cache_key in self._scheduler_coeffs_cache:
            self._cache_hits += 1
            self._scheduler_coeffs_cache.move_to_end(cache_key)
            return self._scheduler_coeffs_cache[cache_key]

        self._cache_misses += 1

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        sub_timesteps = torch.stack([self.timesteps[t] for t in self.t_list])

        c_skip_list = []
        c_out_list = []

        if self.scheduler_type == "TCD":
            # Simplified TCD coefficients: c_skip=0, c_out=1 (direct model output).
            for timestep in sub_timesteps:
                c_skip_list.append(torch.zeros_like(timestep, dtype=self.dtype))
                c_out_list.append(torch.ones_like(timestep, dtype=self.dtype))
            logging.info(f"[TCD Coefficients] Using simplified approach (c_skip=0, c_out=1)")

        elif self.scheduler_type == "Euler":
            for timestep in sub_timesteps:
                c_skip_list.append(torch.zeros_like(timestep, dtype=self.dtype))
                c_out_list.append(torch.ones_like(timestep, dtype=self.dtype))

        else:
            for timestep in sub_timesteps:
                c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
                c_skip_list.append(c_skip)
                c_out_list.append(c_out)

        c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []

        if hasattr(self.scheduler, 'alphas_cumprod') and self.scheduler.alphas_cumprod is not None:
            for timestep in sub_timesteps:
                t_idx = int(timestep.cpu().item()) if timestep.device.type != 'cpu' else int(timestep.item())
                alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[t_idx].sqrt()
                beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[t_idx]).sqrt()
                alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
                beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)
        else:
            for timestep in sub_timesteps:
                alpha_prod_t_sqrt = torch.ones_like(timestep, dtype=self.dtype, device=self.device)
                beta_prod_t_sqrt = torch.zeros_like(timestep, dtype=self.dtype, device=self.device)
                alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
                beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)

        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        if self.scheduler_type == "TCD":
            logging.info(f"[TCD Alpha/Beta] sub_timesteps: {sub_timesteps.tolist()}")
            logging.info(f"[TCD Alpha/Beta] alpha_prod_t_sqrt: {alpha_prod_t_sqrt.flatten().tolist()}")
            logging.info(f"[TCD Alpha/Beta] beta_prod_t_sqrt: {beta_prod_t_sqrt.flatten().tolist()}")
            logging.info(f"[TCD Alpha/Beta] c_skip: {c_skip.flatten().tolist()}")
            logging.info(f"[TCD Alpha/Beta] c_out: {c_out.flatten().tolist()}")

        coeffs = {
            'sub_timesteps': sub_timesteps,
            'c_skip': c_skip,
            'c_out': c_out,
            'alpha_prod_t_sqrt': alpha_prod_t_sqrt,
            'beta_prod_t_sqrt': beta_prod_t_sqrt,
        }

        if len(self._scheduler_coeffs_cache) >= self._max_scheduler_cache_size:
            oldest_key, oldest_coeffs = self._scheduler_coeffs_cache.popitem(last=False)
            for tensor in oldest_coeffs.values():
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    del tensor

        self._scheduler_coeffs_cache[cache_key] = coeffs
        return coeffs

    def get_ssf_stats(self) -> Dict[str, Any]:
        """SSF performance stats (frames processed/skipped, estimated power savings)."""
        total_frames = self._ssf_frames_processed + self._ssf_frames_skipped
        skip_rate = (self._ssf_frames_skipped / total_frames * 100) if total_frames > 0 else 0
        # Power savings model from the StreamDiffusion paper (2.39x reduction).
        power_savings = skip_rate * 2.39 / 100

        return {
            'frames_processed': self._ssf_frames_processed,
            'frames_skipped': self._ssf_frames_skipped,
            'total_frames': total_frames,
            'skip_rate_percent': round(skip_rate, 2),
            'estimated_power_savings_factor': round(power_savings, 2),
            'ssf_enabled': self.similar_image_filter,
        }

    def get_cache_stats(self) -> Dict[str, int]:
        """Scheduler coefficients cache stats."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate_percent': round(hit_rate, 2),
            'cache_size': len(self._scheduler_coeffs_cache),
        }

    def load_lcm_lora(
        self,
        pretrained_model_name_or_path_or_dict: Union[
            str, Dict[str, torch.Tensor]
        ] = "latent-consistency/lcm-lora-sdv1-5",
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        # The default LCM-LoRA repo dropped pytorch_lora_weights.bin; request the safetensors
        # file explicitly (falling back to diffusers auto-resolution for other repo layouts).
        if "weight_name" not in kwargs and isinstance(pretrained_model_name_or_path_or_dict, str):
            try:
                self.pipe.load_lora_weights(
                    pretrained_model_name_or_path_or_dict, adapter_name,
                    weight_name="pytorch_lora_weights.safetensors", **kwargs
                )
                return
            except Exception:
                pass
        self.pipe.load_lora_weights(
            pretrained_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    def enable_similar_image_filter(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.similar_image_filter = True
        self.similar_filter.set_threshold(threshold)
        self.similar_filter.set_max_skip_frame(max_skip_frame)

    def disable_similar_image_filter(self) -> None:
        self.similar_image_filter = False

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
        generator: Optional[torch.Generator] = torch.Generator(),
        seed: int = 2,
    ) -> None:
        self.generator = generator
        self.generator.manual_seed(seed)
        self._needs_buffer_refill = True

        self._cn_cond_ring: Dict[int, torch.Tensor] = {}
        self._cn_cond_ring_needs_init = True

        if self.denoising_steps_num > 1:
            self.x_t_latent_buffer = torch.zeros(
                (
                    (self.denoising_steps_num - 1) * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None

        if self.cfg_type == "none":
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale
        self.delta = delta

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True

        # Skip the eager text-encoder forward when prompt inputs are unchanged.
        _embed_key = (prompt, negative_prompt, do_classifier_free_guidance,
                      self.cfg_type, self.batch_size)
        if getattr(self, "_embed_cache_key", None) != _embed_key or getattr(self, "prompt_embeds", None) is None:
            encoder_output = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
            )
            self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

            if self.use_denoising_batch and self.cfg_type == "full":
                if encoder_output[1] is not None:
                    uncond_prompt_embeds = encoder_output[1].repeat(self.batch_size, 1, 1)
            elif self.cfg_type == "initialize":
                if encoder_output[1] is not None:
                    uncond_prompt_embeds = encoder_output[1].repeat(self.frame_bff_size, 1, 1)

            if self.guidance_scale > 1.0 and (
                self.cfg_type == "initialize" or self.cfg_type == "full"
            ):
                self.prompt_embeds = torch.cat(
                    [uncond_prompt_embeds, self.prompt_embeds], dim=0
                )
            self._embed_cache_key = _embed_key

        coeffs = self._compute_scheduler_coefficients(num_inference_steps)

        self.sub_timesteps = coeffs['sub_timesteps']
        self.c_skip = coeffs['c_skip']
        self.c_out = coeffs['c_out']
        alpha_prod_t_sqrt = coeffs['alpha_prod_t_sqrt']
        beta_prod_t_sqrt = coeffs['beta_prod_t_sqrt']

        sub_timesteps_tensor = self.sub_timesteps.to(dtype=torch.long, device=self.device)
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=generator,
        ).to(device=self.device, dtype=self.dtype)

        self._init_noise_rolled = torch.cat(
            [self.init_noise[1:], self.init_noise[0:1]], dim=0
        )

        self.stock_noise = torch.zeros_like(self.init_noise)
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        # PERFORMANCE: Pre-compute the constant beta*init_noise tail used by the
        # multi-step buffer update in predict_x0_batch. init_noise is fixed after
        # prepare() in the denoising-batch path, so this product is constant for
        # the session and must not be recomputed every frame.
        if self.denoising_steps_num > 1 and self.do_add_noise:
            self._beta_noise_tail = (
                self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
            ).contiguous()
        else:
            self._beta_noise_tail = None

        # Pre-compute CFG concat tensors that are stable across frames (init_noise isn't).
        if self.use_denoising_batch and (self.cfg_type == "self" or self.cfg_type == "initialize"):
            self.alpha_next = torch.concat(
                [
                    self.alpha_prod_t_sqrt[1:],
                    torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                ],
                dim=0,
            )
            self.beta_next = torch.concat(
                [
                    self.beta_prod_t_sqrt[1:],
                    torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                ],
                dim=0,
            )
        else:
            self.alpha_next = None
            self.beta_next = None

    @torch.no_grad()
    def update_prompt(self, prompt: str) -> None:
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        self._embed_cache_key = None

        # Reset RCFG rolling noise on prompt change: it carries residual from the
        # old prompt across frames, causing a flash on swap (multi-step only).
        if self.stock_noise is not None:
            self.stock_noise.zero_()

        self._needs_buffer_refill = True
        self.prev_image_result = None
        self._cn_cond_ring_needs_init = True

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        return (
            self.alpha_prod_t_sqrt[t_index] * original_samples
            + self.beta_prod_t_sqrt[t_index] * noise
        )

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        if idx is None:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch
            ) / self.alpha_prod_t_sqrt
            denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        else:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch
            ) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = (
                self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
            )

        return denoised_batch

    def unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, list[int]],
        idx: Optional[int] = None,
        controlnet_image: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        controlnet_model: Optional[Union[Any, List[Any]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list_new = torch.concat([t_list[0:1], t_list], dim=0)
            t_list = t_list_new
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list_new = torch.concat([t_list, t_list], dim=0)
            t_list = t_list_new
        else:
            x_t_latent_plus_uc = x_t_latent

        # Multi-ControlNet: accumulate residuals across all controlnets.
        down_block_res_samples = None
        mid_block_res_sample = None

        if controlnet_model is not None and controlnet_image is not None:
            # Cache normalized list since controlnet_model is stable across frames.
            if controlnet_model is not self._cached_controlnet_model:
                if not isinstance(controlnet_model, list):
                    self._cached_controlnet_model_list = [controlnet_model]
                else:
                    self._cached_controlnet_model_list = controlnet_model
                self._cached_controlnet_model = controlnet_model
            controlnet_model = self._cached_controlnet_model_list

            if not isinstance(controlnet_image, list):
                controlnet_image = [controlnet_image]
            if not isinstance(controlnet_conditioning_scale, list):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet_model)

            for i, (cn_model, cn_image, cn_scale) in enumerate(zip(controlnet_model, controlnet_image, controlnet_conditioning_scale)):
                # Granular guidance multiplier (0.0=prompt only, 1.0=balanced, 2.0=max structure).
                residual_multiplier = getattr(self, '_cached_controlnet_guidance_strength', 1.0)

                per_slot_cond = self._build_per_slot_cond(i, cn_image)

                controlnet_cond_input = per_slot_cond
                expected = x_t_latent_plus_uc.shape[0]
                if expected != per_slot_cond.shape[0]:
                    if expected == 2 * per_slot_cond.shape[0]:
                        controlnet_cond_input = torch.cat([per_slot_cond, per_slot_cond])
                    elif expected == per_slot_cond.shape[0] + self.frame_bff_size:
                        controlnet_cond_input = torch.cat(
                            [per_slot_cond[0:self.frame_bff_size], per_slot_cond]
                        )
                    else:
                        controlnet_cond_input = per_slot_cond[0:1].expand(
                            expected, -1, -1, -1
                        ).contiguous()

                # Convert scale to tensor of same dtype as latents — prevents torch.compile
                # recompiles on each new scalar and avoids precision mixing.
                if isinstance(cn_scale, torch.Tensor):
                    cn_scale_tensor = cn_scale
                elif isinstance(cn_scale, (list, tuple)):
                    # Union-style: per-control-type list; 0-D buffer doesn't fit.
                    cn_scale_tensor = torch.tensor(
                        cn_scale, device=x_t_latent.device, dtype=x_t_latent.dtype
                    )
                else:
                    if i not in self._cn_scale_buf:
                        self._cn_scale_buf[i] = torch.zeros(
                            (), device=x_t_latent.device, dtype=x_t_latent.dtype
                        )
                    cn_scale_tensor = self._cn_scale_buf[i]
                    cn_scale_tensor.fill_(float(cn_scale))

                down_samples, mid_sample = cn_model(
                    x_t_latent_plus_uc,
                    t_list,
                    encoder_hidden_states=self.prompt_embeds,
                    controlnet_cond=controlnet_cond_input,
                    conditioning_scale=cn_scale_tensor,
                    return_dict=False,
                )

                if x_t_latent_plus_uc.shape[0] > x_t_latent.shape[0]:
                    del controlnet_cond_input

                self._rotate_cn_ring(i, cn_image)

                if residual_multiplier != 1.0:
                    for r in down_samples:
                        r.mul_(residual_multiplier)
                    mid_sample.mul_(residual_multiplier)

                if not hasattr(self, '_guidance_strength_logged'):
                    logging.info(f"[ControlNet Guidance] Strength: {residual_multiplier:.2f}")
                    self._guidance_strength_logged = True

                # Sum residuals across ControlNets (matches diffusers MultiControlNetModel).
                if down_block_res_samples is None:
                    down_block_res_samples = down_samples
                    mid_block_res_sample = mid_sample
                else:
                    down_block_res_samples = [
                        samples_prev + samples_curr
                        for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                    ]
                    mid_block_res_sample = mid_block_res_sample + mid_sample
                    del down_samples, mid_sample

            if getattr(self, '_cn_cond_ring_needs_init', False):
                self._cn_cond_ring_needs_init = False

        try:
            unet_kwargs = {
                "encoder_hidden_states": self.prompt_embeds,
                "down_block_additional_residuals": down_block_res_samples,
                "mid_block_additional_residual": mid_block_res_sample,
                "return_dict": False,
            }
            # IP-Adapter FaceID: embeddings must match x_t_latent_plus_uc batch (cfg_type+steps dependent).
            if ip_adapter_image_embeds is not None:
                expected_batch = x_t_latent_plus_uc.shape[0]
                expanded_embeds = []
                for embed in ip_adapter_image_embeds:
                    if embed.shape[0] != expected_batch:
                        embed = embed.expand(expected_batch, *embed.shape[1:]).contiguous()
                    expanded_embeds.append(embed)
                unet_kwargs["added_cond_kwargs"] = {"image_embeds": expanded_embeds}

                # FaceID PlusV2: auto-expand clip_embeds on the projection layer.
                if hasattr(self.unet, 'encoder_hid_proj'):
                    proj_layers = getattr(self.unet.encoder_hid_proj, 'image_projection_layers', None)
                    if proj_layers:
                        proj_layer = proj_layers[0]
                        if hasattr(proj_layer, 'clip_embeds') and proj_layer.clip_embeds is not None:
                            ce = proj_layer.clip_embeds
                            needs_unsqueeze = (ce.dim() == 3)
                            if needs_unsqueeze:
                                ce = ce.unsqueeze(1)
                            if ce.shape[0] != expected_batch:
                                proj_layer.clip_embeds = ce.expand(
                                    expected_batch, *ce.shape[1:]
                                ).contiguous()
                            elif needs_unsqueeze:
                                proj_layer.clip_embeds = ce

            model_pred = self.unet(
                x_t_latent_plus_uc,
                t_list,
                **unet_kwargs,
            )[0]

            # StreamV2V cache update runs outside the CUDA graph.
            update_cache_after_unet(self.unet)
        finally:
            if down_block_res_samples is not None:
                del down_block_res_samples
            if mid_block_res_sample is not None:
                del mid_block_res_sample

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            old_stock_noise = self.stock_noise
            self.stock_noise = torch.concat(
                [model_pred[0:1], self.stock_noise[1:]], dim=0
            )
            del old_stock_noise
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        else:
            noise_pred_text = model_pred
        if self.guidance_scale > 1.0 and (
            self.cfg_type == "self" or self.cfg_type == "initialize"
        ):
            noise_pred_uncond = self.stock_noise * self.delta
        if self.guidance_scale > 1.0 and self.cfg_type != "none":
            model_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            model_pred = noise_pred_text

        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                delta_x = self.alpha_next * delta_x
                delta_x = delta_x / self.beta_next
                self.stock_noise = self._init_noise_rolled + delta_x
        else:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        if self.guidance_scale > 1.0:
            if 'x_t_latent_plus_uc' in locals() and x_t_latent_plus_uc is not x_t_latent:
                del x_t_latent_plus_uc
            if 'controlnet_cond_input' in locals() and controlnet_cond_input is not controlnet_image:
                del controlnet_cond_input

        return denoised_batch, model_pred

    def encode_image(self, image_tensors: torch.Tensor) -> torch.Tensor:
        image_tensors = image_tensors.to(
            device=self.device,
            dtype=self.vae.dtype,
        )
        img_latent = retrieve_latents(self.vae.encode(image_tensors), self.generator)
        img_latent = img_latent * self.vae.config.scaling_factor
        x_t_latent = self.add_noise(img_latent, self.init_noise[0], 0)
        return x_t_latent

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(
            x_0_pred_out / self.vae.config.scaling_factor, return_dict=False
        )[0]

    def _ensure_cn_ring(self, cn_index: int, current_cond: torch.Tensor) -> Optional[torch.Tensor]:
        if self.denoising_steps_num <= 1:
            return None
        if current_cond.dim() == 3:
            cur4d = current_cond.unsqueeze(0)
        else:
            cur4d = current_cond[0:1] if current_cond.shape[0] > 1 else current_cond
        ring_batch = (self.denoising_steps_num - 1) * self.frame_bff_size
        expected_shape = (ring_batch, cur4d.shape[1], cur4d.shape[2], cur4d.shape[3])
        ring = self._cn_cond_ring.get(cn_index)
        if ring is None or tuple(ring.shape) != expected_shape:
            ring = torch.zeros(expected_shape, dtype=self.dtype, device=self.device)
            self._cn_cond_ring[cn_index] = ring
            self._cn_cond_ring_needs_init = True
        if self._cn_cond_ring_needs_init:
            ring.copy_(cur4d.expand(ring_batch, -1, -1, -1).to(self.dtype))
        return ring

    def _build_per_slot_cond(self, cn_index: int, current_cond: torch.Tensor) -> torch.Tensor:
        if self.denoising_steps_num <= 1:
            return current_cond.unsqueeze(0) if current_cond.dim() == 3 else current_cond
        ring = self._ensure_cn_ring(cn_index, current_cond)
        if current_cond.dim() == 3:
            cur4d = current_cond.unsqueeze(0)
        else:
            cur4d = current_cond
        fb = self.frame_bff_size
        if cur4d.shape[0] != fb:
            cur4d = cur4d.expand(fb, -1, -1, -1)
        return torch.cat([cur4d, ring], dim=0)

    def _rotate_cn_ring(self, cn_index: int, current_cond: torch.Tensor) -> None:
        if self.denoising_steps_num <= 1:
            return
        ring = self._cn_cond_ring.get(cn_index)
        if ring is None:
            return
        if current_cond.dim() == 3:
            cur4d = current_cond.unsqueeze(0)
        else:
            cur4d = current_cond[0:self.frame_bff_size]
        fb = self.frame_bff_size
        if ring.shape[0] > fb:
            ring[fb:].copy_(ring[:-fb].clone())
        ring[:fb].copy_(cur4d.to(self.dtype))

    def _refill_buffer_from_current(self, x_t_latent: torch.Tensor) -> None:
        if self.x_t_latent_buffer is None:
            return
        fb = self.frame_bff_size
        a0 = self.alpha_prod_t_sqrt[0:fb]
        b0 = self.beta_prod_t_sqrt[0:fb]
        n0 = self.init_noise[0:fb]
        x_0_estimate = (x_t_latent - b0 * n0) / a0
        for k in range(self.denoising_steps_num - 1):
            slot_t = k + 1
            s = slice(slot_t * fb, (slot_t + 1) * fb)
            self.x_t_latent_buffer[k * fb:(k + 1) * fb] = (
                self.alpha_prod_t_sqrt[s] * x_0_estimate
                + self.beta_prod_t_sqrt[s] * self.init_noise[s]
            )

    def predict_x0_batch(
        self,
        x_t_latent: torch.Tensor,
        controlnet_image: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        controlnet_model: Optional[Union[Any, List[Any]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if self.denoising_steps_num > 1:
                if getattr(self, '_needs_buffer_refill', False):
                    self._refill_buffer_from_current(x_t_latent)
                    self._needs_buffer_refill = False
                prev_latent_batch = self.x_t_latent_buffer
                old_x_t_latent = x_t_latent
                x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                if old_x_t_latent is not x_t_latent:
                    del old_x_t_latent

                old_stock_noise = self.stock_noise
                self.stock_noise = torch.cat(
                    (self.init_noise[0:1], self.stock_noise[:-1]), dim=0
                )
                del old_stock_noise
            x_0_pred_batch, model_pred = self.unet_step(
                x_t_latent,
                t_list,
                controlnet_image=controlnet_image,
                controlnet_model=controlnet_model,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
            )

            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                # In-place update of the prepare()-time buffer keeps its pointer
                # stable and avoids a per-frame allocation. beta*init_noise tail
                # is the precomputed constant self._beta_noise_tail.
                if (
                    self.x_t_latent_buffer is not None
                    and self.x_t_latent_buffer.shape[0] == x_0_pred_batch.shape[0] - 1
                ):
                    torch.mul(
                        self.alpha_prod_t_sqrt[1:],
                        x_0_pred_batch[:-1],
                        out=self.x_t_latent_buffer,
                    )
                    if self._beta_noise_tail is not None:
                        self.x_t_latent_buffer.add_(self._beta_noise_tail)
                else:
                    # Fallback for shape quirks (e.g. frame_buffer_size > 1):
                    # preserve the original reassigning behavior exactly.
                    if self.do_add_noise:
                        self.x_t_latent_buffer = (
                            self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                            + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                        )
                    else:
                        self.x_t_latent_buffer = (
                            self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        )
            else:
                x_0_pred_out = x_0_pred_batch
                self.x_t_latent_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(1,).repeat(self.frame_bff_size,)
                x_0_pred, model_pred = self.unet_step(
                    x_t_latent,
                    t,
                    idx,
                    controlnet_image=controlnet_image,
                    controlnet_model=controlnet_model,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    ip_adapter_image_embeds=ip_adapter_image_embeds,
                )
                if idx < len(self.sub_timesteps_tensor) - 1:
                    if self.do_add_noise:
                        x_t_latent = self.alpha_prod_t_sqrt[
                            idx + 1
                        ] * x_0_pred + self.beta_prod_t_sqrt[
                            idx + 1
                        ] * torch.randn_like(
                            x_0_pred, device=self.device, dtype=self.dtype
                        )
                    else:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            x_0_pred_out = x_0_pred

        return x_0_pred_out

    @torch.no_grad()
    def __call__(
        self,
        x: Union[torch.Tensor, PIL.Image.Image, np.ndarray] = None,
        controlnet_image: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        controlnet_model: Optional[Union[Any, List[Any]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        internal_timings = {}

        if x is not None:
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x * 2.0 - 1.0
            if self.similar_image_filter:
                x = self.similar_filter(x)
                if x is None:
                    self._ssf_frames_skipped += 1
                    self._needs_buffer_refill = False
                    self._cn_cond_ring_needs_init = False
                    time.sleep(self.inference_time_ema)
                    return self.prev_image_result
                else:
                    self._ssf_frames_processed += 1

            if self.enable_profiling:
                vae_encode_start = time.time()
                x_t_latent = self.encode_image(x)
                torch.cuda.synchronize()
                internal_timings['vae_encode'] = (time.time() - vae_encode_start) * 1000
            else:
                x_t_latent = self.encode_image(x)
        else:
            x_t_latent = torch.randn((1, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
            if self.enable_profiling:
                internal_timings['vae_encode'] = 0.0

        if self.enable_profiling:
            unet_start = time.time()
            x_0_pred_out = self.predict_x0_batch(
                x_t_latent,
                controlnet_image=controlnet_image,
                controlnet_model=controlnet_model,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
            )
            torch.cuda.synchronize()
            internal_timings['unet_controlnet'] = (time.time() - unet_start) * 1000
        else:
            x_0_pred_out = self.predict_x0_batch(
                x_t_latent,
                controlnet_image=controlnet_image,
                controlnet_model=controlnet_model,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
            )

        if self.latent_feedback_strength > 0.0 and self._prev_latent is not None:
            s = self.latent_feedback_strength
            x_0_pred_out = (1.0 - s) * x_0_pred_out + s * self._prev_latent
        if self.latent_feedback_strength > 0.0:
            self._prev_latent = x_0_pred_out.detach()

        if self.enable_profiling:
            vae_decode_start = time.time()
            x_output = self.decode_image(x_0_pred_out).detach()
            torch.cuda.synchronize()
            internal_timings['vae_decode'] = (time.time() - vae_decode_start) * 1000
        else:
            x_output = self.decode_image(x_0_pred_out).detach()

        self.prev_image_result = x_output

        if self.enable_profiling:
            self.last_internal_timings = internal_timings
        else:
            self.last_internal_timings = {}

        return x_output

    @torch.no_grad()
    def txt2img(self, batch_size: int = 1) -> torch.Tensor:
        # Reuse seeded init_noise -> stable reproducible image (no per-frame flicker).
        x_0_pred_out = self.predict_x0_batch(self.init_noise[:batch_size].clone())
        return self.decode_image(x_0_pred_out).detach()

    def txt2img_sd_turbo(self, batch_size: int = 1) -> torch.Tensor:
        # Reuse seeded init_noise -> stable reproducible image (no per-frame flicker).
        x_t_latent = self.init_noise[:batch_size]
        model_pred = self.unet(
            x_t_latent,
            self.sub_timesteps_tensor,
            encoder_hidden_states=self.prompt_embeds,
            return_dict=False,
        )[0]
        x_0_pred_out = (
            x_t_latent - self.beta_prod_t_sqrt * model_pred
        ) / self.alpha_prod_t_sqrt
        return self.decode_image(x_0_pred_out)
