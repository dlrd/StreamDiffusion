import time
import logging
from typing import List, Optional, Union, Any, Dict, Tuple, Literal
from collections import OrderedDict

import numpy as np
import PIL.Image
import torch
from diffusers import (
    LCMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    TCDScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

from .image_filter import SimilarImageFilter
from .attention_processors import update_cache_after_unet


_SHAPE_CACHE = {}  # (height, width, scale_factor) -> (latent_h, latent_w)


def _get_latent_dimensions(height: int, width: int, scale_factor: int) -> Tuple[int, int]:
    cache_key = (height, width, scale_factor)
    if cache_key not in _SHAPE_CACHE:
        _SHAPE_CACHE[cache_key] = (
            int(height // scale_factor),
            int(width // scale_factor),
        )
    return _SHAPE_CACHE[cache_key]


class StreamDiffusionXL:
    def __init__(
        self,
        pipe: StableDiffusionXLPipeline,
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

        self.latent_feedback_strength = 0.0
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

        self.similar_image_filter = True
        self.similar_filter = SimilarImageFilter(threshold=0.98, max_skip_frame=3)
        self.prev_image_result = None

        self._ssf_frames_processed = 0
        self._ssf_frames_skipped = 0

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        # Scheduler picked by configure_scheduler() per model type.
        self.scheduler = None
        self.scheduler_type = None
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        # Hyper-SDXL 1-step U-Net checkpoint mode: affects scaling + CFG handling.
        self.use_hyper_unet_checkpoint = False

        # Do NOT cache vae.config.scaling_factor: TinyVAE swap changes it.

        self._cached_controlnet_model = None
        self._cached_controlnet_model_list = None

        self._cn_scale_buf = {}

        self.inference_time_ema = 0

        self.last_internal_timings = {}
        self.enable_profiling = False
        self._cuda_events = {}

        self._scheduler_coeffs_cache = OrderedDict()
        self._max_scheduler_cache_size = 16
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype, device):
        """Build SDXL micro-conditioning time_ids tensor."""
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        return torch.tensor([add_time_ids], dtype=dtype, device=device)

    def configure_scheduler(self, model_type: str = "default", eta: float = 1.0, use_checkpoint_unet: bool = False):
        """Pick the scheduler for the SDXL model variant (turbo / hyper / lightning / default)."""
        if model_type == "turbo":
            # SDXL-Turbo uses ADD; EulerAncestral is the official scheduler.
            self.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing",
            )
            self.scheduler_type = "Euler"
            logging.info(f"[Scheduler] Using EulerAncestralDiscreteScheduler (timestep_spacing='trailing') for SDXL-Turbo")

        elif model_type == "hyper":
            # Hyper-SDXL needs timestep_spacing='trailing' for quality.
            # 1-step U-Net checkpoint requires num_train_timesteps=800.
            scheduler_config = self.pipe.scheduler.config.copy()
            if use_checkpoint_unet:
                scheduler_config['num_train_timesteps'] = 800
                logging.info(f"[Scheduler] Using timestep 800 for Hyper-SDXL 1-step U-Net checkpoint")

            self.scheduler = TCDScheduler.from_config(
                scheduler_config,
                timestep_spacing="trailing",
            )
            if hasattr(self.scheduler, 'set_eta'):
                self.scheduler.set_eta(eta)
            self.scheduler_type = "TCD"
            logging.info(f"[Scheduler] Using TCDScheduler (eta={eta}, timestep_spacing='trailing') for Hyper-SDXL")

        elif model_type == "lightning":
            self.scheduler = LCMScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_scaling=1.0,
            )
            self.scheduler_type = "LCM"
            logging.info(f"[Scheduler] Using LCMScheduler (timestep_scaling=1.0) for SDXL Lightning")

        else:
            self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            self.scheduler_type = "LCM"
            logging.info(f"[Scheduler] Using LCMScheduler (default) for SDXL")

    def _compute_scheduler_coefficients(self, num_inference_steps: int):
        cache_key = (tuple(self.t_list), num_inference_steps)

        if cache_key in self._scheduler_coeffs_cache:
            self._cache_hits += 1
            self._scheduler_coeffs_cache.move_to_end(cache_key)
            return self._scheduler_coeffs_cache[cache_key]

        self._cache_misses += 1

        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        sub_timesteps = []
        for t in self.t_list:
            sub_timesteps.append(self.timesteps[t])

        c_skip_list = []
        c_out_list = []
        for timestep in sub_timesteps:
            if self.scheduler_type in ("TCD", "Euler"):
                c_skip_list.append(torch.zeros_like(timestep, dtype=self.dtype))
                c_out_list.append(torch.ones_like(timestep, dtype=self.dtype))
            else:
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
        for timestep in sub_timesteps:
            # alphas_cumprod indexing requires CPU long (Euler timesteps are float).
            t_cpu = timestep.cpu().long() if timestep.is_cuda else timestep.long()
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[t_cpu].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[t_cpu]).sqrt()
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

        coeffs = {
            'sub_timesteps': [int(t) for t in sub_timesteps],
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
        """SSF performance stats."""
        total_frames = self._ssf_frames_processed + self._ssf_frames_skipped
        skip_rate = (self._ssf_frames_skipped / total_frames * 100) if total_frames > 0 else 0
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
            self._x_t_latent_concat_buf = torch.empty(
                (
                    self.denoising_steps_num * self.frame_bff_size,
                    4,
                    self.latent_height,
                    self.latent_width,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.x_t_latent_buffer = None
            self._x_t_latent_concat_buf = None

        self._needs_buffer_refill = True

        self._cn_cond_ring: Dict[Tuple[int, int], torch.Tensor] = {}
        self._cn_cond_ring_needs_init = True

        # Hyper-SDXL U-Net checkpoint needs guidance_scale respected even with cfg_type='none'.
        if self.cfg_type == "none" and not self.use_hyper_unet_checkpoint:
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale

        # Hyper-SDXL CFG sanity warnings (LoRA optimal ~0.8-1.0; 1-step U-Net optimal 0).
        if self.scheduler_type == "TCD":
            if self.use_hyper_unet_checkpoint:
                if self.guidance_scale > 0:
                    if self.cfg_type == "none":
                        logging.info(f"[Hyper-SDXL U-Net] guidance_scale={self.guidance_scale:.2f} (cfg_type='none' -> CFG disabled)")
                    else:
                        logging.warning(f"[Hyper-SDXL U-Net] guidance_scale={self.guidance_scale:.2f} with cfg_type='{self.cfg_type}' (CFG active)")
                        logging.warning(f"[Hyper-SDXL U-Net] Recommended: guidance_scale=0 or cfg_type='none' for 1-step U-Net")
                else:
                    logging.info(f"[Hyper-SDXL U-Net] Using guidance_scale=0 (optimal for 1-step U-Net)")

                if self.guidance_scale == 0 and negative_prompt and len(negative_prompt.strip()) > 0:
                    logging.warning("[Hyper-SDXL U-Net] Negative prompts not supported with CFG=0 - ignoring")
                    negative_prompt = ""
            else:
                if self.guidance_scale > 1.2:
                    logging.warning(f"[Hyper-SDXL LoRA] guidance_scale {self.guidance_scale:.2f} is high (optimal: 0.8-1.0)")
                elif self.guidance_scale < 0.6:
                    logging.warning(f"[Hyper-SDXL LoRA] guidance_scale {self.guidance_scale:.2f} is low (optimal: 0.8-1.0)")
                else:
                    logging.info(f"[Hyper-SDXL LoRA] Using guidance_scale={self.guidance_scale:.2f}")

                if negative_prompt and len(negative_prompt.strip()) > 0:
                    logging.warning("[Hyper-SDXL LoRA] Negative prompts have limited effect with 1-step LoRA")

        self.delta = delta

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True

        # Skip the eager dual-CLIP encode when prompt inputs are unchanged.
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

            # SDXL encode_prompt: (prompt_embeds, neg_prompt_embeds, pooled, neg_pooled).
            if len(encoder_output) > 2:
                pooled_prompt_embeds = encoder_output[2]
                add_time_ids = self._get_add_time_ids(
                    (self.height, self.width),
                    (0, 0),
                    (self.height, self.width),
                    dtype=self.dtype,
                    device=self.device,
                )
                self.added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }
            else:
                self.added_cond_kwargs = None

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

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
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

        # Update SDXL pooled embeds in added_cond_kwargs (time_ids stay).
        if len(encoder_output) > 2:
            pooled_prompt_embeds = encoder_output[2]

            if hasattr(self, 'added_cond_kwargs') and self.added_cond_kwargs is not None:
                self.added_cond_kwargs["text_embeds"] = pooled_prompt_embeds
            else:
                add_time_ids = self._get_add_time_ids(
                    (self.height, self.width),
                    (0, 0),
                    (self.height, self.width),
                    dtype=self.dtype,
                    device=self.device,
                )
                self.added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }

        self._embed_cache_key = None

        # Reset RCFG rolling noise on prompt change (flash fix, multi-step only).
        if hasattr(self, 'stock_noise') and self.stock_noise is not None:
            self.stock_noise.zero_()

        self._needs_buffer_refill = True
        if hasattr(self, 'prev_image_result'):
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

        down_block_res_samples = None
        mid_block_res_sample = None

        if controlnet_model is not None and controlnet_image is not None:
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
                residual_multiplier = getattr(self, '_cached_controlnet_guidance_strength', 1.0)

                # Per-slot cond: align ControlNet conditioning with each multi-step
                # batch slot. cn_image may be a single tensor or List[Tensor] (Union).
                per_slot_cond = self._build_per_slot_cond(i, cn_image)

                cn_image_is_list = isinstance(per_slot_cond, list)
                if cn_image_is_list:
                    if x_t_latent_plus_uc.shape[0] > x_t_latent.shape[0]:
                        controlnet_cond_input = [torch.cat([img, img]) for img in per_slot_cond]
                    else:
                        controlnet_cond_input = per_slot_cond
                else:
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

                if isinstance(cn_scale, torch.Tensor):
                    cn_scale_tensor = cn_scale
                elif isinstance(cn_scale, (list, tuple)):
                    # SDXL Union: per-control-type list, 0-D buffer doesn't fit.
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

                cn_kwargs = dict(
                    encoder_hidden_states=self.prompt_embeds,
                    controlnet_cond=controlnet_cond_input,
                    conditioning_scale=cn_scale_tensor,
                    return_dict=False,
                )
                # SDXL ControlNets need text_embeds + time_ids.
                if hasattr(self, 'added_cond_kwargs') and self.added_cond_kwargs is not None:
                    cn_kwargs["added_cond_kwargs"] = self.added_cond_kwargs
                down_samples, mid_sample = cn_model(
                    x_t_latent_plus_uc,
                    t_list,
                    **cn_kwargs,
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
            if hasattr(self, 'added_cond_kwargs') and self.added_cond_kwargs is not None:
                unet_kwargs["added_cond_kwargs"] = dict(self.added_cond_kwargs)
            # IP-Adapter FaceID: embeddings must match x_t_latent_plus_uc batch.
            if ip_adapter_image_embeds is not None:
                if "added_cond_kwargs" not in unet_kwargs:
                    unet_kwargs["added_cond_kwargs"] = {}
                expected_batch = x_t_latent_plus_uc.shape[0]
                expanded_embeds = []
                for embed in ip_adapter_image_embeds:
                    if embed.shape[0] != expected_batch:
                        embed = embed.expand(expected_batch, *embed.shape[1:]).contiguous()
                    expanded_embeds.append(embed)
                unet_kwargs["added_cond_kwargs"]["image_embeds"] = expanded_embeds

                # FaceID PlusV2: auto-expand clip_embeds on the projection layer.
                if hasattr(self.unet, 'encoder_hid_proj'):
                    proj_layers = getattr(self.unet.encoder_hid_proj, 'image_projection_layers', None)
                    if proj_layers:
                        proj_layer = proj_layers[0]
                        if hasattr(proj_layer, 'clip_embeds') and proj_layer.clip_embeds is not None:
                            ce = proj_layer.clip_embeds
                            if ce.dim() == 3:
                                ce = ce.unsqueeze(1)
                            if ce.shape[0] != expected_batch:
                                proj_layer.clip_embeds = ce.expand(
                                    expected_batch, *ce.shape[1:]
                                ).contiguous()
                            elif ce is not proj_layer.clip_embeds:
                                proj_layer.clip_embeds = ce

            model_pred = self.unet(
                x_t_latent_plus_uc,
                t_list,
                **unet_kwargs,
            )[0]

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
        # U-Net output is channels_last; VAE needs contiguous memory layout.
        x_0_pred_out = x_0_pred_out.contiguous()
        latents = x_0_pred_out / self.vae.config.scaling_factor
        return self.vae.decode(latents, return_dict=False)[0]

    def _ensure_cn_ring(self, ring_key: Tuple[int, int], current_cond: torch.Tensor) -> Optional[torch.Tensor]:
        if self.denoising_steps_num <= 1:
            return None
        if current_cond.dim() == 3:
            cur4d = current_cond.unsqueeze(0)
        else:
            cur4d = current_cond[0:1] if current_cond.shape[0] > 1 else current_cond
        ring_batch = (self.denoising_steps_num - 1) * self.frame_bff_size
        expected_shape = (ring_batch, cur4d.shape[1], cur4d.shape[2], cur4d.shape[3])
        ring = self._cn_cond_ring.get(ring_key)
        if ring is None or tuple(ring.shape) != expected_shape:
            ring = torch.zeros(expected_shape, dtype=self.dtype, device=self.device)
            self._cn_cond_ring[ring_key] = ring
            self._cn_cond_ring_needs_init = True
        if self._cn_cond_ring_needs_init:
            ring.copy_(cur4d.expand(ring_batch, -1, -1, -1).to(self.dtype))
        return ring

    def _build_per_slot_cond_one(self, ring_key: Tuple[int, int], current_cond: torch.Tensor) -> torch.Tensor:
        if self.denoising_steps_num <= 1:
            return current_cond
        ring = self._ensure_cn_ring(ring_key, current_cond)
        if current_cond.dim() == 3:
            cur4d = current_cond.unsqueeze(0)
        else:
            cur4d = current_cond
        fb = self.frame_bff_size
        if cur4d.shape[0] != fb:
            cur4d = cur4d.expand(fb, -1, -1, -1)
        return torch.cat([cur4d, ring], dim=0)

    def _build_per_slot_cond(self, cn_index: int, cn_image):
        if self.denoising_steps_num <= 1:
            if isinstance(cn_image, list):
                return [c.unsqueeze(0) if torch.is_tensor(c) and c.dim() == 3 else c
                        for c in cn_image]
            return cn_image.unsqueeze(0) if cn_image.dim() == 3 else cn_image
        if isinstance(cn_image, list):
            return [self._build_per_slot_cond_one((cn_index, j), c) for j, c in enumerate(cn_image)]
        return self._build_per_slot_cond_one((cn_index, 0), cn_image)

    def _rotate_cn_ring(self, cn_index: int, cn_image) -> None:
        if self.denoising_steps_num <= 1:
            return
        items = cn_image if isinstance(cn_image, list) else [cn_image]
        fb = self.frame_bff_size
        for j, c in enumerate(items):
            ring = self._cn_cond_ring.get((cn_index, j))
            if ring is None:
                continue
            if c.dim() == 3:
                cur4d = c.unsqueeze(0)
            else:
                cur4d = c[0:fb]
            if ring.shape[0] > fb:
                num_chunks = ring.shape[0] // fb
                for _k in range(num_chunks - 1, 0, -1):
                    ring[_k * fb:(_k + 1) * fb].copy_(
                        ring[(_k - 1) * fb:_k * fb]
                    )
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
                fb = self.frame_bff_size
                self._x_t_latent_concat_buf[:fb].copy_(x_t_latent)
                self._x_t_latent_concat_buf[fb:].copy_(prev_latent_batch)
                x_t_latent = self._x_t_latent_concat_buf

                if self.cfg_type in ("self", "initialize"):
                    for _k in range(self.stock_noise.shape[0] // fb - 1, 0, -1):
                        self.stock_noise[_k * fb:(_k + 1) * fb].copy_(
                            self.stock_noise[(_k - 1) * fb:_k * fb]
                        )
                    self.stock_noise[:fb].copy_(self.init_noise[:fb])
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
        # CUDA events for async timing (compatible with torch.compile).
        if self.enable_profiling:
            events = {
                'frame_start': torch.cuda.Event(enable_timing=True),
                'preprocess_end': torch.cuda.Event(enable_timing=True),
                'vae_encode_end': torch.cuda.Event(enable_timing=True),
                'unet_end': torch.cuda.Event(enable_timing=True),
                'vae_decode_end': torch.cuda.Event(enable_timing=True),
                'frame_end': torch.cuda.Event(enable_timing=True),
            }
            events['frame_start'].record()

        if x is not None:
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x * 2.0 - 1.0

            if self.enable_profiling:
                events['preprocess_end'].record()

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

            x_t_latent = self.encode_image(x)

            if self.enable_profiling:
                events['vae_encode_end'].record()
        else:
            x_t_latent = torch.randn((1, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
            if self.enable_profiling:
                events['preprocess_end'].record()
                events['vae_encode_end'].record()

        x_0_pred_out = self.predict_x0_batch(
            x_t_latent,
            controlnet_image=controlnet_image,
            controlnet_model=controlnet_model,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
        )

        if self.enable_profiling:
            events['unet_end'].record()

        if self.latent_feedback_strength > 0.0 and self._prev_latent is not None:
            s = self.latent_feedback_strength
            x_0_pred_out = (1.0 - s) * x_0_pred_out + s * self._prev_latent
        if self.latent_feedback_strength > 0.0:
            self._prev_latent = x_0_pred_out.detach()

        x_output = self.decode_image(x_0_pred_out).detach()

        if self.enable_profiling:
            events['vae_decode_end'].record()

        self.prev_image_result = x_output

        if self.enable_profiling:
            events['frame_end'].record()
            torch.cuda.synchronize()

            preprocess_ms = events['frame_start'].elapsed_time(events['preprocess_end'])
            vae_encode_ms = events['preprocess_end'].elapsed_time(events['vae_encode_end'])
            unet_ms = events['vae_encode_end'].elapsed_time(events['unet_end'])
            vae_decode_ms = events['unet_end'].elapsed_time(events['vae_decode_end'])
            total_ms = events['frame_start'].elapsed_time(events['frame_end'])

            overhead_ms = total_ms - (preprocess_ms + vae_encode_ms + unet_ms + vae_decode_ms)
            fps = 1000.0 / total_ms if total_ms > 0 else 0

            self.last_internal_timings = {
                'preprocess': preprocess_ms,
                'vae_encode': vae_encode_ms,
                'unet_controlnet': unet_ms,
                'vae_decode': vae_decode_ms,
                'overhead': overhead_ms,
                'total_frame': total_ms,
                'fps': fps,
            }

            logging.info(f"[PERF] Total: {total_ms:.1f}ms ({fps:.1f} FPS) | "
                        f"Preprocess: {preprocess_ms:.1f}ms | "
                        f"VAE Encode: {vae_encode_ms:.1f}ms | "
                        f"UNet+ControlNet: {unet_ms:.1f}ms | "
                        f"VAE Decode: {vae_decode_ms:.1f}ms | "
                        f"Overhead: {overhead_ms:.1f}ms")
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
        unet_kwargs = {
            "encoder_hidden_states": self.prompt_embeds,
            "return_dict": False,
        }
        if hasattr(self, 'added_cond_kwargs') and self.added_cond_kwargs is not None:
            unet_kwargs["added_cond_kwargs"] = self.added_cond_kwargs

        model_pred = self.unet(
            x_t_latent,
            self.sub_timesteps_tensor,
            **unet_kwargs,
        )[0]
        x_0_pred_out = (
            x_t_latent - self.beta_prod_t_sqrt * model_pred
        ) / self.alpha_prod_t_sqrt
        return self.decode_image(x_0_pred_out)
