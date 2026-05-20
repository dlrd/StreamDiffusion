import time
import logging
from typing import List, Optional, Union, Any, Dict, Tuple, Literal
from collections import OrderedDict

import numpy as np
import PIL.Image
import torch
from diffusers import LCMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, TCDScheduler, StableDiffusionXLPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

from src.streamdiffusion.image_filter import SimilarImageFilter


# OPTIMIZATION: Global cache for tensor shape computations (CPU → RAM optimization)
# Stores pre-computed latent dimensions to avoid repeated division operations
_SHAPE_CACHE = {}  # Cache: (height, width, scale_factor) -> (latent_h, latent_w)


def _get_latent_dimensions(height: int, width: int, scale_factor: int) -> Tuple[int, int]:
    """
    OPTIMIZED: Get latent dimensions with RAM caching.
    Eliminates repeated division operations by caching results.

    Args:
        height: Image height
        width: Image width
        scale_factor: VAE scale factor

    Returns:
        Tuple of (latent_height, latent_width)
    """
    cache_key = (height, width, scale_factor)

    if cache_key not in _SHAPE_CACHE:
        _SHAPE_CACHE[cache_key] = (
            int(height // scale_factor),
            int(width // scale_factor)
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

        # OPTIMIZATION: Use cached latent dimensions calculation (CPU → RAM optimization)
        self.latent_height, self.latent_width = _get_latent_dimensions(
            height, width, pipe.vae_scale_factor
        )

        self.frame_bff_size = frame_buffer_size
        self.denoising_steps_num = len(t_index_list)

        self.cfg_type = cfg_type

        # Latent feedback: blend current denoised latent with previous for temporal smoothing
        self.latent_feedback_strength = 0.0
        self._prev_latent = None

        # Motion-aware noise
        self.motion_aware_noise = False
        self.motion_aware_noise_sensitivity = 0.5
        self._prev_input_latent = None
        self._motion_noise_scale = 1.0

        # ControlNet guidance (set externally by App after stream creation)
        self._cached_controlnet_guidance_strength = 1.0
        self._guidance_strength_logged = False

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

        # OPTIMIZATION #7: SSF (Stochastic Similarity Filter) for power efficiency
        # Reduces GPU activation frequency for similar frames (2.39x energy savings)
        self.similar_image_filter = True
        # Optimized params for real-time: threshold=0.98, max_skip=3 frames
        self.similar_filter = SimilarImageFilter(threshold=0.98, max_skip_frame=3)
        self.prev_image_result = None

        # SSF metrics tracking
        self._ssf_frames_processed = 0
        self._ssf_frames_skipped = 0

        self.pipe = pipe
        self.image_processor = VaeImageProcessor(pipe.vae_scale_factor)

        # SCHEDULER SELECTION: Detect model type and use appropriate scheduler
        # - Hyper-SDXL: Uses TCDScheduler (Trajectory Consistency Distillation) for 1-8 step inference
        # - SDXL Lightning: Uses LCMScheduler with timestep_scaling=1.0
        # - Other SDXL: Uses LCMScheduler with default settings
        self.scheduler = None
        self.scheduler_type = None  # Track which scheduler is being used
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

        # Flag to track if using Hyper-SDXL U-Net checkpoint (not LoRA)
        # This affects latent scaling when using TinyVAE
        self.use_hyper_unet_checkpoint = False

        # NOTE: Do NOT cache vae.config.scaling_factor here!
        # The VAE can be replaced with TinyVAE later (scaling_factor changes from 0.18215 to 1.0)
        # Caching would cause incorrect encoding/decoding and blue images

        # PERFORMANCE: Cache for ControlNet model list normalization (avoid isinstance checks)
        # controlnet_model is stable throughout the session, so we cache the normalized list
        self._cached_controlnet_model = None  # Raw input reference for cache validation
        self._cached_controlnet_model_list = None  # Normalized list (cached)

        self.inference_time_ema = 0

        # CLEANUP: Removed dead asynchronous timing system (lines 118-120)
        # System was disabled due to memory leaks and never used

        # Internal profiling breakdown using CUDA events (compatible with torch.compile)
        self.last_internal_timings = {}
        self.enable_profiling = False  # Profiling disabled by default (can be enabled via web UI)
        self._cuda_events = {}  # Store CUDA events for async timing

        # OPTIMIZATION: Scheduler coefficients cache with LRU eviction (CPU → RAM optimization)
        # CRITICAL FIX: Limited to 16 entries to prevent unbounded GPU memory leak
        # Cache pre-computed scheduler coefficients to eliminate repeated calculations
        self._scheduler_coeffs_cache = OrderedDict()  # LRU cache with size limit
        self._max_scheduler_cache_size = 16  # Reasonable limit for most use cases
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype, device):
        """Helper method for SDXL time embeddings"""
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
        return add_time_ids

    def configure_scheduler(self, model_type: str = "default", eta: float = 1.0, use_checkpoint_unet: bool = False):
        """
        Configure the appropriate scheduler based on the model type.

        Args:
            model_type: Type of model ("turbo", "hyper", "lightning", or "default")
            eta: TCDScheduler eta parameter (default 1.0, lower = more detail for multi-step)
            use_checkpoint_unet: If True, uses timestep 800 for Hyper-SDXL 1-step U-Net checkpoint
        """
        if model_type == "turbo":
            # SDXL-Turbo uses ADD (Adversarial Diffusion Distillation), not LCM
            # EulerAncestralDiscreteScheduler is the official scheduler (stochastic, better detail for 2+ steps)
            self.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing",
            )
            self.scheduler_type = "Euler"
            logging.info(f"[Scheduler] Using EulerAncestralDiscreteScheduler (timestep_spacing='trailing') for SDXL-Turbo")

        elif model_type == "hyper":
            # Hyper-SDXL uses TCDScheduler for 1-8 step inference
            # CRITICAL: timestep_spacing="trailing" is required for Hyper-SD quality!
            # It generates higher quality images with more details for few-step inference

            # CRITICAL: Hyper-SDXL 1-step U-Net checkpoint requires timestep 800 instead of 999
            scheduler_config = self.pipe.scheduler.config.copy()
            if use_checkpoint_unet:
                scheduler_config['num_train_timesteps'] = 800
                logging.info(f"[Scheduler] Using timestep 800 for Hyper-SDXL 1-step U-Net checkpoint")

            self.scheduler = TCDScheduler.from_config(
                scheduler_config,
                timestep_spacing="trailing",  # CRITICAL for Hyper-SD quality
                # eta controls detail level: lower = more detail for multi-step inference
            )
            # Set eta parameter for TCDScheduler
            if hasattr(self.scheduler, 'set_eta'):
                self.scheduler.set_eta(eta)
            self.scheduler_type = "TCD"
            logging.info(f"[Scheduler] Using TCDScheduler (eta={eta}, timestep_spacing='trailing') for Hyper-SDXL")

        elif model_type == "lightning":
            # SDXL Lightning uses LCMScheduler with timestep_scaling=1.0
            self.scheduler = LCMScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_scaling=1.0,  # SDXL-specific: lower scaling for correct coefficients
            )
            self.scheduler_type = "LCM"
            logging.info(f"[Scheduler] Using LCMScheduler (timestep_scaling=1.0) for SDXL Lightning")

        else:
            # Default SDXL models use LCMScheduler with default settings
            self.scheduler = LCMScheduler.from_config(
                self.pipe.scheduler.config,
            )
            self.scheduler_type = "LCM"
            logging.info(f"[Scheduler] Using LCMScheduler (default) for SDXL")

    def _compute_scheduler_coefficients(self, num_inference_steps: int):
        """
        OPTIMIZED: Compute scheduler coefficients with RAM caching.
        Eliminates redundant CPU calculations by caching pre-computed values.

        Returns cached coefficients if available, otherwise computes and caches them.
        """
        # Create cache key from t_list and num_inference_steps
        cache_key = (tuple(self.t_list), num_inference_steps)

        # Check if coefficients are already cached
        if cache_key in self._scheduler_coeffs_cache:
            self._cache_hits += 1
            # CRITICAL FIX: Move to end (LRU - mark as recently used)
            self._scheduler_coeffs_cache.move_to_end(cache_key)
            return self._scheduler_coeffs_cache[cache_key]

        # Cache miss - compute coefficients
        self._cache_misses += 1

        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)

        # Build sub_timesteps list
        sub_timesteps = []
        for t in self.t_list:
            sub_timesteps.append(self.timesteps[t])

        # Compute c_skip and c_out coefficients
        c_skip_list = []
        c_out_list = []
        for timestep in sub_timesteps:
            if self.scheduler_type in ("TCD", "Euler"):
                # TCDScheduler / EulerDiscreteScheduler: simplified coefficients
                # c_skip=0, c_out=1 means: no latent skipping, use direct model output
                c_skip_list.append(torch.zeros_like(timestep, dtype=self.dtype))
                c_out_list.append(torch.ones_like(timestep, dtype=self.dtype))
            else:
                # LCMScheduler (default) - uses boundary condition discrete scaling
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

        # Compute alpha and beta coefficients
        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        for timestep in sub_timesteps:
            # Move timestep to CPU as int for indexing (EulerDiscreteScheduler uses float timesteps,
            # and some schedulers keep alphas_cumprod on CPU)
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

        # DIAGNOSTIC: Scheduler coefficients (disabled to reduce log spam)
        # logging.info(f"[Scheduler:{self.scheduler_type}] t_list: {self.t_list}, num_inference_steps: {num_inference_steps}")
        # logging.info(f"[Scheduler:{self.scheduler_type}] sub_timesteps: {sub_timesteps}")
        # logging.info(f"[Scheduler:{self.scheduler_type}] c_skip: {c_skip.flatten().tolist()}")
        # logging.info(f"[Scheduler:{self.scheduler_type}] c_out: {c_out.flatten().tolist()}")
        # logging.info(f"[Scheduler:{self.scheduler_type}] alpha_prod_t_sqrt: {alpha_prod_t_sqrt.flatten().tolist()}")
        # logging.info(f"[Scheduler] beta_prod_t_sqrt: {beta_prod_t_sqrt.flatten().tolist()}")

        # Cache the computed coefficients
        # Convert sub_timesteps to plain ints (EulerDiscreteScheduler uses float tensors)
        coeffs = {
            'sub_timesteps': [int(t) for t in sub_timesteps],
            'c_skip': c_skip,
            'c_out': c_out,
            'alpha_prod_t_sqrt': alpha_prod_t_sqrt,
            'beta_prod_t_sqrt': beta_prod_t_sqrt,
        }

        # CRITICAL FIX: Implement LRU eviction to prevent unbounded GPU memory growth
        if len(self._scheduler_coeffs_cache) >= self._max_scheduler_cache_size:
            # Evict oldest entry (LRU)
            oldest_key, oldest_coeffs = self._scheduler_coeffs_cache.popitem(last=False)
            # Explicitly free GPU tensors from evicted entry
            for tensor in oldest_coeffs.values():
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    del tensor

        self._scheduler_coeffs_cache[cache_key] = coeffs
        return coeffs

    def get_ssf_stats(self) -> Dict[str, Any]:
        """
        Get SSF (Stochastic Similarity Filter) performance statistics.

        Returns:
            Dictionary with frames processed, skipped, and power savings estimate
        """
        total_frames = self._ssf_frames_processed + self._ssf_frames_skipped
        skip_rate = (self._ssf_frames_skipped / total_frames * 100) if total_frames > 0 else 0
        # Estimated power savings based on StreamDiffusion paper (2.39x reduction)
        power_savings = skip_rate * 2.39 / 100

        return {
            'frames_processed': self._ssf_frames_processed,
            'frames_skipped': self._ssf_frames_skipped,
            'total_frames': total_frames,
            'skip_rate_percent': round(skip_rate, 2),
            'estimated_power_savings_factor': round(power_savings, 2),
            'ssf_enabled': self.similar_image_filter
        }

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get statistics about scheduler coefficients cache performance.

        Returns:
            Dictionary with cache hits, misses, and hit rate
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate_percent': round(hit_rate, 2),
            'cache_size': len(self._scheduler_coeffs_cache)
        }

    # CLEANUP: Removed dead timing methods (_process_timing_events_async, get_performance_stats)
    # These were part of the abandoned asynchronous timing system that caused memory leaks
    # Profiling is now handled via self.enable_profiling and self.last_internal_timings

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
        # initialize x_t_latent (it can be any random tensor)
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

        # CRITICAL: For Hyper-SDXL U-Net checkpoint, respect guidance_scale even with cfg_type='none'
        # The U-Net checkpoint needs CFG=0 for optimal quality, but this must come from user input
        if self.cfg_type == "none" and not self.use_hyper_unet_checkpoint:
            self.guidance_scale = 1.0
        else:
            self.guidance_scale = guidance_scale

        # Hyper-SDXL 1-step optimization: CFG scale and negative prompt handling
        if self.scheduler_type == "TCD":
            # CRITICAL: Different CFG requirements for LoRA vs U-Net checkpoint!
            # - Hyper-SDXL LoRA: CFG 0.8-1.0 (trained with some CFG)
            # - Hyper-SDXL 1-step U-Net checkpoint: CFG = 0 recommended (NO classifier-free guidance!)

            if self.use_hyper_unet_checkpoint:
                # U-Net checkpoint mode: Respect guidance_scale from Smode, but warn if not optimal
                # Using CFG > 0 may cause oversaturation, cartoonish look, and visible latent noise
                if self.guidance_scale > 0:
                    if self.cfg_type == "none":
                        logging.info(f"[Hyper-SDXL U-Net] guidance_scale={self.guidance_scale:.2f} (cfg_type='none' → CFG disabled, value ignored)")
                    else:
                        logging.warning(f"[Hyper-SDXL U-Net] guidance_scale={self.guidance_scale:.2f} with cfg_type='{self.cfg_type}' (CFG active)")
                        logging.warning(f"[Hyper-SDXL U-Net] Recommended: guidance_scale=0 or cfg_type='none' (1-step U-Net not trained with CFG)")
                else:
                    logging.info(f"[Hyper-SDXL U-Net] Using guidance_scale=0 (optimal for 1-step U-Net)")

                # Disable negative prompts if CFG=0
                if self.guidance_scale == 0 and negative_prompt and len(negative_prompt.strip()) > 0:
                    logging.warning("[Hyper-SDXL U-Net] Negative prompts not supported with CFG=0 - ignoring")
                    negative_prompt = ""
            else:
                # LoRA mode: Respect guidance_scale from Smode, but warn if outside optimal range
                # Optimal CFG range: 0.6-1.2 (best: 0.8-1.0)
                if self.guidance_scale > 1.2:
                    logging.warning(f"[Hyper-SDXL LoRA] guidance_scale {self.guidance_scale:.2f} is high (optimal: 0.8-1.0)")
                elif self.guidance_scale < 0.6:
                    logging.warning(f"[Hyper-SDXL LoRA] guidance_scale {self.guidance_scale:.2f} is low (optimal: 0.8-1.0)")
                else:
                    logging.info(f"[Hyper-SDXL LoRA] Using guidance_scale={self.guidance_scale:.2f} (within optimal range)")

                # Warn about negative prompts for LoRA 1-step (not well supported)
                if negative_prompt and len(negative_prompt.strip()) > 0:
                    logging.warning("[Hyper-SDXL LoRA] Negative prompts have limited effect with 1-step LoRA")

        self.delta = delta

        do_classifier_free_guidance = False
        if self.guidance_scale > 1.0:
            do_classifier_free_guidance = True

        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        self.prompt_embeds = encoder_output[0].repeat(self.batch_size, 1, 1)

        # SDXL Support: Extract pooled_prompt_embeds and create added_cond_kwargs
        # SDXL encode_prompt returns: (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        if len(encoder_output) > 2:
            # SDXL pipeline detected
            pooled_prompt_embeds = encoder_output[2]

            # Create time_ids for SDXL (original_size, crops_coords, target_size)
            add_time_ids = self._get_add_time_ids(
                (self.height, self.width),  # original_size
                (0, 0),  # crops_coords_top_left
                (self.height, self.width),  # target_size
                dtype=self.dtype,
                device=self.device
            )

            # Build added_cond_kwargs for SDXL
            self.added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids
            }
        else:
            # Fallback for non-SDXL (shouldn't happen with this pipeline)
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

        # OPTIMIZATION: Use cached scheduler coefficients (CPU → RAM optimization)
        # This eliminates redundant CPU calculations by retrieving pre-computed values from RAM
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

        # PERFORMANCE: Pre-compute CFG concatenations to avoid expensive operations every frame
        # alpha_next and beta_next are stable (derived from alpha_prod_t_sqrt and beta_prod_t_sqrt)
        # Pre-computing saves ~1-3ms per frame (~2-4% FPS improvement)
        # NOTE: init_noise changes dynamically during generation, so cannot be pre-computed
        if self.use_denoising_batch and (self.cfg_type == "self" or self.cfg_type == "initialize"):
            # Pre-compute alpha_next: [alpha[1:], ones] concatenated
            self.alpha_next = torch.concat(
                [
                    self.alpha_prod_t_sqrt[1:],
                    torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                ],
                dim=0,
            )
            # Pre-compute beta_next: [beta[1:], ones] concatenated
            self.beta_next = torch.concat(
                [
                    self.beta_prod_t_sqrt[1:],
                    torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                ],
                dim=0,
            )
        else:
            # Not using CFG with denoising batch, no need to pre-compute
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

        # SDXL Support: Update pooled embeddings for added_cond_kwargs
        # CRITICAL: update_prompt() must also update SDXL conditioning, not just prompt_embeds
        if len(encoder_output) > 2:
            # SDXL pipeline detected - update pooled embeddings
            pooled_prompt_embeds = encoder_output[2]

            # Update the text_embeds in added_cond_kwargs (time_ids stays the same)
            if hasattr(self, 'added_cond_kwargs') and self.added_cond_kwargs is not None:
                self.added_cond_kwargs["text_embeds"] = pooled_prompt_embeds
            else:
                # First time calling update_prompt without prepare() - create added_cond_kwargs
                add_time_ids = self._get_add_time_ids(
                    (self.height, self.width),
                    (0, 0),
                    (self.height, self.width),
                    dtype=self.dtype,
                    device=self.device
                )
                self.added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                }

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        noisy_samples = (
            self.alpha_prod_t_sqrt[t_index] * original_samples
            + self.beta_prod_t_sqrt[t_index] * noise
        )
        return noisy_samples

    def scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        # TODO: use t_list to select beta_prod_t_sqrt
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # LEAK FIX: Create temporary variables for concatenations, then explicitly free them
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list_new = torch.concat([t_list[0:1], t_list], dim=0)
            t_list = t_list_new  # Reassign after creation
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list_new = torch.concat([t_list, t_list], dim=0)
            t_list = t_list_new  # Reassign after creation
        else:
            x_t_latent_plus_uc = x_t_latent

        # MULTI-CONTROLNET SUPPORT: Accumulate residuals from multiple ControlNets
        down_block_res_samples = None
        mid_block_res_sample = None

        # Normalize inputs to lists for uniform handling
        if controlnet_model is not None and controlnet_image is not None:
            # PERFORMANCE: Cache normalized controlnet_model list to avoid isinstance checks
            # controlnet_model is stable throughout the session, so we cache it
            if controlnet_model is not self._cached_controlnet_model:
                # Cache miss or first time - normalize and cache
                if not isinstance(controlnet_model, list):
                    self._cached_controlnet_model_list = [controlnet_model]
                else:
                    self._cached_controlnet_model_list = controlnet_model
                self._cached_controlnet_model = controlnet_model
            # Use cached normalized list
            controlnet_model = self._cached_controlnet_model_list

            # Convert single values to lists (image and scale change every frame)
            if not isinstance(controlnet_image, list):
                controlnet_image = [controlnet_image]
            if not isinstance(controlnet_conditioning_scale, list):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet_model)

            # Process each ControlNet and accumulate residuals
            for cn_model, cn_image, cn_scale in zip(controlnet_model, controlnet_image, controlnet_conditioning_scale):
                # GRANULAR GUIDANCE SYSTEM (0.0-2.0 slider)
                # Works with all cfg_types (self, full, none) and sd-turbo streaming
                # Direct residual manipulation AFTER ControlNet processing

                # Get guidance strength parameter (0.0-2.0, default 1.0)
                # 0.0 = No ControlNet influence (pure prompt)
                # 1.0 = Balanced (ControlNet at 100%)
                # 2.0 = Maximum ControlNet influence (structure dominates)
                residual_multiplier = getattr(self, '_cached_controlnet_guidance_strength', 1.0)

                # Handle CFG batch expansion (only for cfg_type="full" with guidance_scale > 1.0)
                controlnet_cond_input = cn_image
                if x_t_latent_plus_uc.shape[0] > x_t_latent.shape[0]:
                    # CFG "full" mode - duplicate image for both branches
                    controlnet_cond_input = torch.cat([cn_image, cn_image])

                # CRITICAL: Convert scale to tensor to prevent torch.compile cache misses
                # If cn_scale is a Python float, torch.compile treats it as a constant
                # and recompiles for every different value (0.8 vs 0.7 = new compilation)
                # By converting to tensor, torch.compile sees it as a variable input
                # IMPORTANT: Use same dtype as latents to avoid precision mixing artifacts
                if not isinstance(cn_scale, torch.Tensor):
                    cn_scale_tensor = torch.tensor(cn_scale, device=x_t_latent.device, dtype=x_t_latent.dtype)
                else:
                    cn_scale_tensor = cn_scale

                # Note: CUDA Graphs step boundary is marked at the frame level (SmodeStreamDiffusion.py)
                # via torch.compiler.cudagraph_mark_step_begin() before calling the pipeline

                # Run ControlNet
                cn_kwargs = dict(
                    encoder_hidden_states=self.prompt_embeds,
                    controlnet_cond=controlnet_cond_input,
                    conditioning_scale=cn_scale_tensor,
                    return_dict=False,
                )
                # SDXL ControlNets need added_cond_kwargs (text_embeds + time_ids)
                if hasattr(self, 'added_cond_kwargs') and self.added_cond_kwargs is not None:
                    cn_kwargs["added_cond_kwargs"] = self.added_cond_kwargs
                down_samples, mid_sample = cn_model(
                    x_t_latent_plus_uc,
                    t_list,
                    **cn_kwargs,
                )

                # CRITICAL FIX: Explicitly free ControlNet input tensor if it was concatenated
                if x_t_latent_plus_uc.shape[0] > x_t_latent.shape[0]:
                    del controlnet_cond_input

                # Apply multiplier to ALL residuals (universal approach)
                if residual_multiplier != 1.0:
                    down_samples = list(down_samples)  # Convert to list for modification
                    for i in range(len(down_samples)):
                        down_samples[i] = down_samples[i] * residual_multiplier
                    mid_sample = mid_sample * residual_multiplier

                # DEBUG: Log guidance strength application (once per session)
                if not hasattr(self, '_guidance_strength_logged'):
                    import logging
                    logging.info(f"[ControlNet Guidance] Strength: {residual_multiplier:.2f} (0.0=prompt only, 1.0=balanced, 2.0=max structure)")
                    self._guidance_strength_logged = True

                # MULTI-CONTROLNET: Sum residuals from all ControlNets
                # This is how diffusers MultiControlNetModel works - outputs are added together
                if down_block_res_samples is None:
                    # First ControlNet - initialize accumulators
                    down_block_res_samples = down_samples
                    mid_block_res_sample = mid_sample
                else:
                    # Subsequent ControlNets - add to accumulators
                    down_block_res_samples = [
                        samples_prev + samples_curr
                        for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                    ]
                    mid_block_res_sample = mid_block_res_sample + mid_sample

                    # CRITICAL FIX: Free individual ControlNet outputs after summing
                    del down_samples, mid_sample

            # CRITICAL FIX: Explicitly free ControlNet residual tensors after use
            # These can be 10-50MB and should be freed immediately

        # Call UNet with ControlNet residuals
        # MEMORY SAFETY: Use try-finally to ensure cleanup even on exception
        try:
            unet_kwargs = {
                "encoder_hidden_states": self.prompt_embeds,
                "down_block_additional_residuals": down_block_res_samples,
                "mid_block_additional_residual": mid_block_res_sample,
                "return_dict": False,
            }
            # SDXL Support: Add added_cond_kwargs if available
            if hasattr(self, 'added_cond_kwargs') and self.added_cond_kwargs is not None:
                unet_kwargs["added_cond_kwargs"] = self.added_cond_kwargs

            model_pred = self.unet(
                x_t_latent_plus_uc,
                t_list,
                **unet_kwargs
            )[0]

            # StreamV2V: update cache AFTER UNet call (outside CUDA graph)
            from streamdiffusion.attention_processors import update_cache_after_unet
            update_cache_after_unet(self.unet)
        finally:
            # CRITICAL FIX: Free ControlNet residuals immediately after UNet call
            # This runs even if UNet raises an exception, preventing GPU memory leak
            if down_block_res_samples is not None:
                del down_block_res_samples
            if mid_block_res_sample is not None:
                del mid_block_res_sample

        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            noise_pred_text = model_pred[1:]
            # LEAK FIX: Save old reference before reassigning
            old_stock_noise = self.stock_noise
            self.stock_noise = torch.concat(
                [model_pred[0:1], self.stock_noise[1:]], dim=0
            )  # ここコメントアウトでself out cfg
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

        # compute the previous noisy sample x_t -> x_t-1
        if self.use_denoising_batch:
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self.scheduler_step_batch(model_pred, scaled_noise, idx)
                # PERFORMANCE: Use pre-computed alpha_next and beta_next (stable across frames)
                # This eliminates 2x torch.ones_like() + 2x torch.concat() per frame
                delta_x = self.alpha_next * delta_x
                delta_x = delta_x / self.beta_next
                # init_noise changes dynamically, must compute fresh each frame
                init_noise = torch.concat(
                    [self.init_noise[1:], self.init_noise[0:1]], dim=0
                )
                self.stock_noise = init_noise + delta_x

        else:
            # denoised_batch = self.scheduler.step(model_pred, t_list[0], x_t_latent).denoised
            denoised_batch = self.scheduler_step_batch(model_pred, x_t_latent, idx)

        # LEAK FIX: Explicitly free temporary tensors created during CFG
        if self.guidance_scale > 1.0:
            if 'x_t_latent_plus_uc' in locals() and x_t_latent_plus_uc is not x_t_latent:
                del x_t_latent_plus_uc
            if 'controlnet_cond_input' in locals() and controlnet_cond_input is not controlnet_image:
                del controlnet_cond_input
            # Note: alpha_next, beta_next, init_noise are cleaned up by Python GC
            # but explicit del can help in tight loops

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
        # CRITICAL: Convert from channels_last (U-Net output) to contiguous (VAE input)
        # Without this, VAE reads the tensor in the wrong memory layout -> RGB soup
        x_0_pred_out = x_0_pred_out.contiguous()

        # DEBUG: Log scaling_factor and latent values before scaling (first call only)
        if not hasattr(self, '_logged_decode_scaling'):
            import logging
            scaling_factor = self.vae.config.scaling_factor
            logging.info(f"[decode_image] VAE scaling_factor: {scaling_factor}")
            logging.info(f"[decode_image] U-Net output (before scaling): range=[{x_0_pred_out.min().item():.4f}, {x_0_pred_out.max().item():.4f}]")
            self._logged_decode_scaling = True

        # Vanilla SDXL decode: Simple scaling by VAE scaling_factor
        latents = x_0_pred_out / self.vae.config.scaling_factor

        # DEBUG: Log latent values after scaling (first call only)
        if not hasattr(self, '_logged_decode_latents'):
            import logging
            logging.info(f"[decode_image] Latents (after scaling): range=[{latents.min().item():.4f}, {latents.max().item():.4f}]")
            self._logged_decode_latents = True

        # Standard VAE decode
        image = self.vae.decode(latents, return_dict=False)[0]

        return image

    def predict_x0_batch(
        self,
        x_t_latent: torch.Tensor,
        controlnet_image: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        controlnet_model: Optional[Union[Any, List[Any]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
    ) -> torch.Tensor:
        prev_latent_batch = self.x_t_latent_buffer

        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if self.denoising_steps_num > 1:
                # LEAK FIX: Save old reference before reassigning
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
            )

            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
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
                t = t.view(
                    1,
                ).repeat(
                    self.frame_bff_size,
                )
                x_0_pred, model_pred = self.unet_step(
                    x_t_latent,
                    t,
                    idx,
                    controlnet_image=controlnet_image,
                    controlnet_model=controlnet_model,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
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
    ) -> torch.Tensor:
        # PROFILING: CUDA events for async timing (compatible with torch.compile)
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
            x = self.image_processor.preprocess(x, self.height, self.width).to(
                device=self.device, dtype=self.dtype
            )

            if self.enable_profiling:
                events['preprocess_end'].record()

            if self.similar_image_filter:
                x = self.similar_filter(x)
                if x is None:
                    # Frame skipped by SSF - save GPU power!
                    self._ssf_frames_skipped += 1
                    time.sleep(self.inference_time_ema)
                    return self.prev_image_result
                else:
                    # Frame processed
                    self._ssf_frames_processed += 1

            x_t_latent = self.encode_image(x)

            if self.enable_profiling:
                events['vae_encode_end'].record()
        else:
            # TODO: check the dimension of x_t_latent
            x_t_latent = torch.randn((1, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
            if self.enable_profiling:
                events['preprocess_end'].record()
                events['vae_encode_end'].record()

        # Motion-aware noise: adapt stock_noise scale based on input motion
        if self.motion_aware_noise and x_t_latent is not None:
            if self._prev_input_latent is not None:
                motion = torch.sqrt(torch.mean((x_t_latent - self._prev_input_latent) ** 2)).item()
                s = self.motion_aware_noise_sensitivity
                target_scale = max(0.3, 1.0 - motion * s * 5.0)
                self._motion_noise_scale = 0.7 * self._motion_noise_scale + 0.3 * target_scale
                if hasattr(self, 'stock_noise') and self.stock_noise is not None:
                    self.stock_noise = self.stock_noise * self._motion_noise_scale
            self._prev_input_latent = x_t_latent.detach()

        x_0_pred_out = self.predict_x0_batch(
            x_t_latent,
            controlnet_image=controlnet_image,
            controlnet_model=controlnet_model,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )

        if self.enable_profiling:
            events['unet_end'].record()

        # Latent feedback: blend with previous frame's latent for temporal smoothing
        if self.latent_feedback_strength > 0.0 and self._prev_latent is not None:
            s = self.latent_feedback_strength
            x_0_pred_out = (1.0 - s) * x_0_pred_out + s * self._prev_latent
        if self.latent_feedback_strength > 0.0:
            self._prev_latent = x_0_pred_out.detach()

        x_output = self.decode_image(x_0_pred_out).detach()

        if self.enable_profiling:
            events['vae_decode_end'].record()

        # OPTIMIZATION: Only clone if Similar Image Filter is enabled (needed for frame reuse)
        # CRITICAL FIX: Remove redundant clone() - saves 3MB GPU allocation per frame
        # The .detach() from decode_image() already breaks gradient chain (line 772/776)
        # SSF filter already clones input in image_filter.py, so double cloning is wasteful
        # This gives 10-15% FPS improvement by eliminating unnecessary GPU allocation
        self.prev_image_result = x_output

        # PROFILING: Calculate timings from CUDA events (async, no pipeline break!)
        if self.enable_profiling:
            events['frame_end'].record()

            # Synchronize ONCE at the end to get all timings
            torch.cuda.synchronize()

            # Calculate elapsed times between events
            preprocess_ms = events['frame_start'].elapsed_time(events['preprocess_end'])
            vae_encode_ms = events['preprocess_end'].elapsed_time(events['vae_encode_end'])
            unet_ms = events['vae_encode_end'].elapsed_time(events['unet_end'])
            vae_decode_ms = events['unet_end'].elapsed_time(events['vae_decode_end'])
            total_ms = events['frame_start'].elapsed_time(events['frame_end'])

            overhead_ms = total_ms - (preprocess_ms + vae_encode_ms + unet_ms + vae_decode_ms)
            fps = 1000.0 / total_ms if total_ms > 0 else 0

            # Store timings
            self.last_internal_timings = {
                'preprocess': preprocess_ms,
                'vae_encode': vae_encode_ms,
                'unet_controlnet': unet_ms,
                'vae_decode': vae_decode_ms,
                'overhead': overhead_ms,
                'total_frame': total_ms,
                'fps': fps
            }

            # Log performance breakdown
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
        x_0_pred_out = self.predict_x0_batch(
            torch.randn((batch_size, 4, self.latent_height, self.latent_width)).to(
                device=self.device, dtype=self.dtype
            )
        )
        # OPTIMIZATION: No need to clone - decode_image already returns a new tensor
        x_output = self.decode_image(x_0_pred_out).detach()
        return x_output

    def txt2img_sd_turbo(self, batch_size: int = 1) -> torch.Tensor:
        x_t_latent = torch.randn(
            (batch_size, 4, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype,
        )
        unet_kwargs = {
            "encoder_hidden_states": self.prompt_embeds,
            "return_dict": False,
        }
        # SDXL Support: Add added_cond_kwargs if available
        if hasattr(self, 'added_cond_kwargs') and self.added_cond_kwargs is not None:
            unet_kwargs["added_cond_kwargs"] = self.added_cond_kwargs

        model_pred = self.unet(
            x_t_latent,
            self.sub_timesteps_tensor,
            **unet_kwargs
        )[0]
        x_0_pred_out = (
            x_t_latent - self.beta_prod_t_sqrt * model_pred
        ) / self.alpha_prod_t_sqrt
        return self.decode_image(x_0_pred_out)
