import gc
import os
import sys
from pathlib import Path
import time
import traceback
from typing import List, Literal, Optional, Union, Dict, Any
import logging

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionXLPipeline
from PIL import Image

from src.streamdiffusion.pipeline_xl import StreamDiffusionXL
from src.streamdiffusion.image_utils import postprocess_image

# Package root directory (independent of CWD, works on any machine)
PACKAGE_DIR = Path(__file__).resolve().parent.parent

torch.set_grad_enabled(False)
# Enable TF32 for faster computation on Ampere+ GPUs (RTX 3000/4000/5000+)
# Using new API (PyTorch 2.9+ compatible)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

class StreamDiffusionWrapperXL:
    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        mode: Literal["img2img", "txt2img"] = "img2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = PACKAGE_DIR / "engines/sdxl",
        cache_dir: Optional[Union[str, Path]] = None,
        torch_compile_enabled: bool = False,
        torch_compile_mode: str = "reduce-overhead",
        torch_compile_fullgraph: bool = True,
    ):
        """
        Initializes the StreamDiffusionWrapper.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        mode : Literal["img2img", "txt2img"], optional
            txt2img or img2img, by default "img2img".
        output_type : Literal["pil", "pt", "np", "latent"], optional
            The output type of image, by default "pil".
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
            If None, the default LCM-LoRA
            ("latent-consistency/lcm-lora-sdv1-5") will be used.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
            If None, the default TinyVAE
            ("madebyollin/taesd") will be used.
        device : Literal["cpu", "cuda"], optional
            The device to use for inference, by default "cuda".
        dtype : torch.dtype, optional
            The dtype for inference, by default torch.float16.
        frame_buffer_size : int, optional
            The frame buffer size for denoising batch, by default 1.
        width : int, optional
            The width of the image, by default 512.
        height : int, optional
            The height of the image, by default 512.
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        acceleration : Literal["none", "xformers", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        device_ids : Optional[List[int]], optional
            The device ids to use for DataParallel, by default None.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        enable_similar_image_filter : bool, optional
            Whether to enable similar image filter or not,
            by default False.
        similar_image_filter_threshold : float, optional
            The threshold for similar image filter, by default 0.98.
        similar_image_filter_max_skip_frame : int, optional
            The max skip frame for similar image filter, by default 10.
        use_denoising_batch : bool, optional
            Whether to use denoising batch or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.
        use_safety_checker : bool, optional
            Whether to use safety checker or not, by default False.
        """
        self.sd_turbo = "turbo" in model_id_or_path or "sdxs" in model_id_or_path.lower()

        if mode == "txt2img":
            if cfg_type != "none":
                raise ValueError(
                    f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}"
                )
            if use_denoising_batch and frame_buffer_size > 1:
                if not self.sd_turbo:
                    raise ValueError(
                        "txt2img mode cannot use denoising batch with frame_buffer_size > 1."
                    )

        if mode == "img2img":
            if not use_denoising_batch:
                raise NotImplementedError(
                    "img2img mode must use denoising batch for now."
                )

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.mode = mode
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = (
            len(t_index_list) * frame_buffer_size
            if use_denoising_batch
            else frame_buffer_size
        )

        self.use_denoising_batch = use_denoising_batch
        self.use_safety_checker = use_safety_checker
        self.torch_compile_enabled = torch_compile_enabled
        self.torch_compile_mode = torch_compile_mode
        self.torch_compile_fullgraph = torch_compile_fullgraph

        self.use_tiny_vae = use_tiny_vae  # Track if TinyVAE is used for denormalization control

        self.stream: StreamDiffusionXL = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
            cache_dir=cache_dir,
        )

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(
                self.stream.unet, device_ids=device_ids
            )

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(similar_image_filter_threshold, similar_image_filter_max_skip_frame)

    def recreate_pipe(self):
        if not self.sd_turbo:
            self.stream.load_lcm_lora()
            self.stream.fuse_lora()

        # CRITICAL FIX: Use taesdxl for SDXL models, not taesd (SD1.5)
        self.stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl").to(
            device=self.stream.pipe.device, dtype=self.stream.pipe.dtype
        )

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> None:
        """
        Prepares the model for inference.

        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise,
            by default 1.0.
        """
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
        )

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
        controlnet_image: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        controlnet_model: Optional[Union[Any, List[Any]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Performs img2img or txt2img based on the mode.

        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if self.mode == "img2img":
            return self.img2img(
                image,
                prompt,
                controlnet_image=controlnet_image,
                controlnet_model=controlnet_model,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
            )
        else:
            return self.txt2img(prompt)

    def txt2img(
        self, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs txt2img.

        Parameters
        ----------
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if self.sd_turbo:
            image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
        else:
            image_tensor = self.stream.txt2img(self.frame_buffer_size)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def img2img(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        prompt: Optional[str] = None,
        controlnet_image: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        controlnet_model: Optional[Union[Any, List[Any]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs img2img.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.

        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        image_tensor = self.stream(
            image,
            controlnet_image=controlnet_image,
            controlnet_model=controlnet_model,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocesses the image.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.

        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
        # CRITICAL FIX: Use context manager to avoid file descriptor leaks
        if isinstance(image, str):
            with Image.open(image) as img:
                image = img.convert("RGB").resize((self.width, self.height))
        elif isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return self.stream.image_processor.preprocess(
            image, self.height, self.width
        ).to(device=self.device, dtype=self.dtype)

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Postprocesses the image.

        Both TinyVAE SDXL and standard SDXL VAE output [-1, 1] range.
        Denormalization (x/2 + 0.5) correctly maps to [0, 1].

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The postprocessed image.
        """
        # DIAGNOSTIC: VAE output statistics (disabled to reduce log spam)
        # logging.info(f"[Postprocess] VAE output - min: {image_tensor.min().item():.4f}, "
        #             f"max: {image_tensor.max().item():.4f}, mean: {image_tensor.mean().item():.4f}")

        # Both TinyVAE SDXL and standard SDXL VAE output [-1, 1]
        # Apply denormalization to convert to [0, 1]
        if self.frame_buffer_size > 1:
            result = postprocess_image(image_tensor, output_type=output_type)
        else:
            result = postprocess_image(image_tensor, output_type=output_type)[0]

        return result

    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        warmup: int = 10,
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = PACKAGE_DIR / "engines/sdxl",
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> StreamDiffusionXL:
        """
        Loads the model.

        This method does the following:

        1. Loads the model from the model_id_or_path.
        2. Loads and fuses the LCM-LoRA model from the lcm_lora_id if needed.
        3. Loads the VAE model from the vae_id if needed.
        4. Enables acceleration if needed.
        5. Prepares the model for inference.
        6. Load the safety checker if needed.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
        acceleration : Literal["none", "xfomers", "sfast", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.

        Returns
        -------
        StreamDiffusionXL
            The loaded model.
        """

        # SPECIAL: Detect Hyper-SDXL U-Net mode (format: "base-model/hyper-sdxl-unet")
        # This allows loading the full Hyper-SDXL U-Net checkpoint instead of using LoRA
        use_hyper_unet = "hyper-sdxl-unet" in model_id_or_path.lower()
        base_model_path = model_id_or_path

        if use_hyper_unet:
            # Extract base model path (before /hyper-sdxl-unet)
            base_model_path = model_id_or_path.split("/hyper-sdxl-unet")[0]
            logging.info(f"[Hyper-SDXL U-Net] Detected Hyper U-Net mode - using base model: {base_model_path}")

        try:
            try:
                # Try to load from Hugging Face with auth token
                pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
                    base_model_path, torch_dtype=self.dtype, cache_dir=cache_dir if cache_dir else None
                ).to(device=self.device, dtype=self.dtype)
            except Exception:
                # If it fails, try to load from local files only
                try:
                    pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
                        base_model_path, local_files_only=True, torch_dtype=self.dtype, cache_dir=cache_dir if cache_dir else None
                    ).to(device=self.device, dtype=self.dtype)
                except Exception:
                    pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_single_file(
                        base_model_path, torch_dtype=self.dtype, cache_dir=cache_dir if cache_dir else None
                    ).to(device=self.device)
        except Exception:  # No model found
            traceback.print_exc()
            logging.error("Model load has failed. Doesn't exist.")
            exit()

        stream = StreamDiffusionXL(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
        )

        # Detect model type for scheduler configuration
        # Check both model name AND LoRA names for detection
        is_lightning_model = "lightning" in model_id_or_path.lower()
        is_hyper_model = "hyper" in model_id_or_path.lower()

        # SPECIAL: If model is "hyper-sdxl-unet", load the full Hyper-SDXL U-Net checkpoint
        # This allows using TensorRT with Hyper-SDXL (avoiding LoRA fusion issues)
        if use_hyper_unet:
            from huggingface_hub import hf_hub_download
            from diffusers import UNet2DConditionModel

            logging.info("[Hyper-SDXL U-Net] Loading full Hyper-SDXL U-Net checkpoint (bypassing LoRA)...")

            # CRITICAL: Use the native Diffusers version, NOT the ComfyUI version!
            # ComfyUI version: Hyper-SDXL-1step-Unet-Comfyui.fp16.safetensors (6.9 GB) - causes NaN
            # Diffusers version: Hyper-SDXL-1step-Unet.safetensors (10.3 GB) - native format

            logging.info("[Hyper-SDXL U-Net] Downloading native Diffusers checkpoint (10.3 GB)...")
            unet_path = hf_hub_download(
                repo_id="ByteDance/Hyper-SD",
                filename="Hyper-SDXL-1step-Unet.safetensors",  # <--- Native Diffusers version
                cache_dir=cache_dir if cache_dir else None
            )
            logging.info(f"[Hyper-SDXL U-Net] Downloaded: {unet_path}")

            # Load the native Diffusers U-Net using load_state_dict (cleaner and faster)
            # Native Diffusers format uses standard Diffusers key names - no conversion needed!
            from safetensors.torch import load_file

            logging.info("[Hyper-SDXL U-Net] Loading state dict from native Diffusers checkpoint...")
            state_dict = load_file(unet_path)

            logging.info(f"[Hyper-SDXL U-Net] Loaded {len(state_dict)} keys from checkpoint")

            # Load the state dict into the existing U-Net (preserves config)
            missing_keys, unexpected_keys = stream.pipe.unet.load_state_dict(state_dict, strict=False)

            if missing_keys:
                logging.warning(f"[Hyper-SDXL U-Net] Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                logging.warning(f"[Hyper-SDXL U-Net] Unexpected keys: {len(unexpected_keys)}")

            logging.info(f"[Hyper-SDXL U-Net] ✓ Loaded native Diffusers U-Net via load_state_dict")

            # Update stream reference
            stream.unet = stream.pipe.unet

            # Set flag to enable Hyper U-Net + TinyVAE scaling fix
            stream.use_hyper_unet_checkpoint = True

            logging.info("[Hyper-SDXL U-Net] ✓ Successfully loaded and replaced U-Net")
            logging.info("[Hyper-SDXL U-Net] Will force stabilityai/sdxl-vae after VAE loading")

            is_hyper_model = True  # Ensure scheduler is configured correctly

        # Also check LoRA names for Hyper-SDXL detection
        if lora_dict is not None:
            for lora_name in lora_dict.keys():
                if "hyper" in lora_name.lower():
                    is_hyper_model = True
                    logging.info(f"[Hyper-SDXL] Detected Hyper LoRA: {lora_name}")
                    break

        # Configure scheduler based on model type BEFORE loading LoRAs
        if self.sd_turbo:
            # SDXL-Turbo uses ADD (Adversarial Diffusion Distillation)
            # Requires EulerDiscreteScheduler, NOT LCMScheduler
            stream.configure_scheduler(model_type="turbo")
            logging.info("[SDXL-Turbo] Configured EulerDiscreteScheduler (trailing)")
        elif is_hyper_model:
            # Hyper-SDXL uses TCDScheduler
            # Use timestep 800 for 1-step U-Net checkpoint (native Diffusers version)
            # eta=0.0: Fully deterministic, no stochastic noise (cleanest output)
            stream.configure_scheduler(model_type="hyper", eta=0.0, use_checkpoint_unet=use_hyper_unet)
            timestep_used = "800 (native for 1-step U-Net)" if use_hyper_unet else "999 (LoRA mode)"
            logging.info(f"[Hyper-SDXL] Configured TCDScheduler with timestep {timestep_used}, eta=0.0 (deterministic)")
        elif is_lightning_model:
            # SDXL Lightning uses LCMScheduler with timestep_scaling=1.0
            stream.configure_scheduler(model_type="lightning")
            logging.info("[SDXL Lightning] Configured LCMScheduler with timestep_scaling=1.0")
        else:
            # Default SDXL models use LCMScheduler
            stream.configure_scheduler(model_type="default")
            logging.info("[SDXL] Configured default LCMScheduler")

        # SDXL Lightning models have LCM LoRA baked in - don't load additional LoRA
        if is_lightning_model:
            logging.info("[SDXL Lightning] Model has LCM LoRA baked in, skipping external LoRA loading")

        if not self.sd_turbo:
            if use_lcm_lora and not is_lightning_model and not is_hyper_model:
                if lcm_lora_id is not None:
                    stream.load_lcm_lora(
                        pretrained_model_name_or_path_or_dict=lcm_lora_id
                    )
                else:
                    stream.load_lcm_lora()
                stream.fuse_lora()

        if lora_dict is not None:
            from huggingface_hub import hf_hub_download

            for lora_name, lora_scale in lora_dict.items():
                # Support format "repo_id::weight_name" for multi-file repos
                # Example: "ByteDance/Hyper-SD::Hyper-SDXL-1step-lora.safetensors"
                if "::" in lora_name:
                    repo_id, weight_name = lora_name.split("::", 1)

                    # Special handling for Hyper-SDXL LoRAs (Kohya format)
                    # Use hf_hub_download to get file path, then load directly
                    if "hyper" in lora_name.lower() and "sdxl" in lora_name.lower():
                        logging.info(f"[Hyper-SDXL] Downloading LoRA: {repo_id}/{weight_name}")
                        lora_path = hf_hub_download(repo_id, weight_name)
                        logging.info(f"[Hyper-SDXL] Loading LoRA from: {lora_path}")
                        stream.load_lora(lora_path)
                        logging.info(f"[Hyper-SDXL] LoRA loaded successfully with weight {lora_scale}")
                    else:
                        stream.load_lora(repo_id, weight_name=weight_name)
                        logging.info(f"Use LoRA: {repo_id} (file: {weight_name}) with weight {lora_scale}")
                else:
                    # Local file path or simple repo name
                    if os.path.exists(lora_name):
                        logging.info(f"Use local LoRA: {lora_name} with weight {lora_scale}")
                    else:
                        logging.info(f"Use LoRA: {lora_name} with weight {lora_scale}")
                    stream.load_lora(lora_name)

                stream.fuse_lora(lora_scale=lora_scale)

        # For Hyper U-Net, allow TinyVAE usage
        # if use_hyper_unet:
        #     logging.info(f"[Hyper-SDXL U-Net] Loading madebyollin/sdxl-vae-fp16-fix (exactly like TinyVAE loading)")
        #     from diffusers import AutoencoderKL
        #     stream.vae = AutoencoderKL.from_pretrained(
        #         "madebyollin/sdxl-vae-fp16-fix",
        #         cache_dir=cache_dir if cache_dir else None
        #     ).to(device=pipe.device, dtype=pipe.dtype)
        #     logging.info(f"[Hyper-SDXL U-Net] ✓ Loaded madebyollin/sdxl-vae-fp16-fix (scaling_factor={stream.vae.config.scaling_factor})")
        #     use_tiny_vae = False

        if use_tiny_vae:
            if vae_id is not None:
                # Custom VAE specified
                vae_model_name = vae_id
            else:
                # Auto-select optimized TinyVAE for SDXL
                # Priority: hybrid-sd-tinyvae-xl (best performance) > taesdxl (fallback)
                vae_model_name = "cqyan/hybrid-sd-tinyvae-xl"
                logging.info(f"[TinyVAE] Auto-selected hybrid-sd-tinyvae-xl for optimal SDXL performance")

            try:
                # Try loading the selected/auto-detected VAE
                stream.vae = AutoencoderTiny.from_pretrained(vae_model_name).to(
                    device=pipe.device, dtype=pipe.dtype
                )
                logging.info(f"[TinyVAE] Successfully loaded: {vae_model_name}")
            except Exception as e:
                # Fallback to standard taesdxl if hybrid model fails
                logging.warning(f"[TinyVAE] Failed to load {vae_model_name}: {e}")
                logging.info(f"[TinyVAE] Falling back to madebyollin/taesdxl")
                stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl").to(
                    device=pipe.device, dtype=pipe.dtype
                )
                logging.info(f"[TinyVAE] Loaded taesdxl")

            # Log the native scaling_factor from TinyVAE SDXL
            native_scaling_factor = getattr(stream.vae.config, 'scaling_factor', None)
            logging.info(f"[TinyVAE SDXL] Native scaling_factor: {native_scaling_factor}")

            # CRITICAL: TinyVAE SDXL was trained with scaling_factor=1.0
            # DO NOT override it to 0.13025 (standard SDXL VAE value) - this causes blur!
            # Let TinyVAE use its trained scaling_factor for correct latent normalization
            if native_scaling_factor is not None:
                logging.info(f"[TinyVAE SDXL] Using native scaling_factor: {native_scaling_factor} (trained value)")
            else:
                # Only set if missing (should not happen with taesdxl)
                stream.vae.config.scaling_factor = 1.0
                logging.info(f"[TinyVAE SDXL] Setting scaling_factor to 1.0 (TinyVAE default)")

            # Ensure shift_factor is set
            if not hasattr(stream.vae.config, 'shift_factor') or stream.vae.config.shift_factor is None:
                stream.vae.config.shift_factor = 0.0
                logging.info(f"[TinyVAE SDXL] Setting shift_factor to 0.0")
        else:
            # Log standard VAE config
            vae_scaling = getattr(stream.vae.config, 'scaling_factor', 'not set')
            vae_shift = getattr(stream.vae.config, 'shift_factor', 'not set')
            logging.info(f"[Standard SDXL VAE] scaling_factor={vae_scaling}, shift_factor={vae_shift}")

        try:
            if acceleration == "xformers":
                # Use PyTorch 2.0+ native SDPA (Scaled Dot Product Attention) instead of xformers
                # PyTorch 2.10+ has flash attention built-in, no need for external xformers
                try:
                    stream.pipe.enable_xformers_memory_efficient_attention()
                    logging.info("xformers memory-efficient attention enabled")
                except Exception as e:
                    logging.warning(f"xformers not available ({e}), using PyTorch native SDPA (flash attention)")
                    # PyTorch 2.0+ uses SDPA by default, nothing to do
            if acceleration == "tensorrt":
                self.enable_tensorrt_acceleration(stream, model_id_or_path, use_lcm_lora, use_tiny_vae)
            if acceleration == "sfast":
                from src.streamdiffusion.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )
                stream = accelerate_with_stable_fast(stream)
                logging.info("StableFast acceleration enabled.")
        except Exception:
            traceback.print_exc()
            logging.warning("Acceleration has failed. Falling back to normal mode.")

        # FreeU: Rebalance UNet skip connections for improved detail/quality (DISABLED - needs tuning for TinyVAE)
        # Must be applied BEFORE torch.compile (adds forward hooks to upsampling blocks)
        # Conservative SDXL params: s1=0.7, s2=0.5, b1=1.05, b2=1.1
        # if hasattr(stream.unet, 'enable_freeu'):
        #     stream.unet.enable_freeu(s1=0.7, s2=0.5, b1=1.05, b2=1.1)
        #     logging.info("[FreeU] Enabled UNet skip connection rebalancing (conservative SDXL params)")

        # Apply torch.compile() optimization for additional 15-25% speedup
        # Only beneficial with xformers or sfast (TensorRT already provides maximum optimization)
        # TESTING: Only compile U-Net (VAE decoder causes RGB soup with hybrid TinyVAE)
        # DISABLED TEMPORARILY for testing Hyper-SDXL U-Net loading
        try:
            if self.torch_compile_enabled and hasattr(torch, 'compile') and torch.__version__ >= '2.0' and acceleration != "tensorrt" and sys.platform != 'win32':
                # Configure torch.compile() cache directory - SEPARATE cache per resolution to avoid conflicts
                # This prevents RGB soup when switching between resolutions
                resolution_str = f"{self.width}x{self.height}"
                torch_compile_cache_dir = PACKAGE_DIR / f"torch_compile_cache/sdxl_{resolution_str}"
                torch_compile_cache_dir.mkdir(parents=True, exist_ok=True)

                # DIAGNOSTICS: Check if cache already exists (files are in subdirectories)
                cache_exists = len(list(torch_compile_cache_dir.glob("**/*.py"))) > 0
                cache_file_count = len(list(torch_compile_cache_dir.glob("**/*.py")))
                if cache_exists:
                    logging.info(f"torch.compile() cache FOUND: {torch_compile_cache_dir} ({cache_file_count} .py files)")
                else:
                    logging.info(f"torch.compile() cache EMPTY: {torch_compile_cache_dir} (will compile from scratch)")

                os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(torch_compile_cache_dir)

                # Enable FX graph cache for faster reloads
                os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'

                # Compile U-Net (main bottleneck - 85% of inference time per profiling)
                # Using 'reduce-overhead' mode + fullgraph=True for maximum optimization
                # fullgraph=True eliminates graph breaks for 30-40% speedup (instead of 14%)
                logging.info("Compiling U-Net with torch.compile() (fullgraph=True for max performance)...")
                compile_start = time.time()
                stream.unet = torch.compile(
                    stream.unet,
                    mode=self.torch_compile_mode,
                    fullgraph=self.torch_compile_fullgraph,
                    dynamic=False  # Static shapes for maximum performance
                )
                compile_time = time.time() - compile_start
                logging.info(f"✓ U-Net compiled in {compile_time:.1f}s {'(cache loaded)' if cache_exists and compile_time < 2 else '(fresh compile)'}")

                # Compile VAE encoder (encodes webcam frames to latents)
                if hasattr(stream.vae, 'encoder'):
                    # Set dedicated cache directory for VAE encoder - SEPARATE per resolution
                    vae_encoder_cache = PACKAGE_DIR / f"torch_compile_cache/sdxl_vae_encoder_{resolution_str}"
                    vae_encoder_cache.mkdir(parents=True, exist_ok=True)
                    old_cache_dir = os.environ.get('TORCHINDUCTOR_CACHE_DIR', '')
                    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(vae_encoder_cache)
                    os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'

                    logging.info("Compiling VAE encoder with torch.compile()...")

                    try:
                        stream.vae.encoder = torch.compile(
                            stream.vae.encoder,
                            mode='reduce-overhead',
                            fullgraph=False,
                            dynamic=False
                        )

                        # Warmup: Force compilation immediately with dummy input
                        logging.info("  Triggering compilation with dummy inference...")
                        dummy_input = torch.randn(1, 3, self.height, self.width, dtype=stream.vae.dtype, device=stream.vae.device)
                        with torch.no_grad():
                            _ = stream.vae.encoder(dummy_input)
                        logging.info(f"✓ VAE encoder compiled and cached successfully (cache: {vae_encoder_cache})")

                    finally:
                        # Restore cache directory
                        if old_cache_dir:
                            os.environ['TORCHINDUCTOR_CACHE_DIR'] = old_cache_dir
                        else:
                            os.environ.pop('TORCHINDUCTOR_CACHE_DIR', None)

                # Compile VAE decoder (decodes latents to images)
                if hasattr(stream.vae, 'decoder'):
                    # Set dedicated cache directory for VAE decoder - SEPARATE per resolution
                    vae_decoder_cache = PACKAGE_DIR / f"torch_compile_cache/sdxl_vae_decoder_{resolution_str}"
                    vae_decoder_cache.mkdir(parents=True, exist_ok=True)
                    old_cache_dir = os.environ.get('TORCHINDUCTOR_CACHE_DIR', '')
                    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(vae_decoder_cache)
                    os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'

                    logging.info("Compiling VAE decoder with torch.compile()...")

                    try:
                        stream.vae.decoder = torch.compile(
                            stream.vae.decoder,
                            mode='reduce-overhead',
                            fullgraph=False,
                            dynamic=False
                        )

                        # Warmup: Force compilation immediately with dummy latent input
                        logging.info("  Triggering compilation with dummy inference...")
                        dummy_latent = torch.randn(1, 4, self.height // 8, self.width // 8, dtype=stream.vae.dtype, device=stream.vae.device)
                        with torch.no_grad():
                            _ = stream.vae.decoder(dummy_latent)
                        logging.info(f"✓ VAE decoder compiled and cached successfully (cache: {vae_decoder_cache})")

                    finally:
                        # Restore cache directory
                        if old_cache_dir:
                            os.environ['TORCHINDUCTOR_CACHE_DIR'] = old_cache_dir
                        else:
                            os.environ.pop('TORCHINDUCTOR_CACHE_DIR', None)
            elif not self.torch_compile_enabled:
                logging.info("Skipping torch.compile() - disabled by config (torch_compile_enabled=False)")
            elif acceleration == "tensorrt":
                logging.info("Skipping torch.compile() - TensorRT already provides maximum optimization")
        except Exception as e:
            logging.warning(f"Warning: torch.compile() failed: {e}")
            logging.warning("Continuing without torch.compile() optimization")

        if seed < 0: # Random seed
            seed = np.random.randint(0, 1000000)

        stream.prepare(
            "",
            "",
            num_inference_steps=50,
            guidance_scale=1.2
            if stream.cfg_type in ["full", "self", "initialize"]
            else 1.0,
            generator=torch.manual_seed(seed),
            seed=seed,
        )

        if self.use_safety_checker:
            from transformers import CLIPFeatureExtractor
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(pipe.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.nsfw_fallback_img = Image.new("RGB", (512, 512), (0, 0, 0))

        return stream

    def enable_tensorrt_acceleration(self, stream: StreamDiffusionXL, model_id_or_path: str, use_lcm_lora: bool, use_tiny_vae: bool, engine_dir: Optional[Union[str, Path]] = "engines/sdxl"):
        from polygraphy import cuda
        from src.streamdiffusion.acceleration.tensorrt import (
            TorchVAEEncoder,
            compile_unet,
            compile_vae_decoder,
            compile_vae_encoder,
        )
        from src.streamdiffusion.acceleration.tensorrt.engine import (
            AutoencoderKLEngine,
            UNet2DConditionModelEngine,
        )
        from src.streamdiffusion.acceleration.tensorrt.models import (
            VAE,
            UNet,
            UNetXL,
            UNetXLSimple,
            VAEEncoder,
        )

        def create_prefix(
            model_id_or_path: str,
            max_batch_size: int,
            min_batch_size: int,
        ):
            maybe_path = Path(model_id_or_path)
            if maybe_path.exists():
                return f"{maybe_path.stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-{self.mode}"
            else:
                return f"{model_id_or_path}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-{self.mode}"

        engine_dir = Path(engine_dir)
        unet_path = os.path.join(
            engine_dir,
            create_prefix(
                model_id_or_path=model_id_or_path,
                max_batch_size=stream.trt_unet_batch_size,
                min_batch_size=stream.trt_unet_batch_size,
            ),
            "unet.engine",
        )
        vae_encoder_path = os.path.join(
            engine_dir,
            create_prefix(
                model_id_or_path=model_id_or_path,
                max_batch_size=self.batch_size
                if self.mode == "txt2img"
                else stream.frame_bff_size,
                min_batch_size=self.batch_size
                if self.mode == "txt2img"
                else stream.frame_bff_size,
            ),
            "vae_encoder.engine",
        )
        vae_decoder_path = os.path.join(
            engine_dir,
            create_prefix(
                model_id_or_path=model_id_or_path,
                max_batch_size=self.batch_size
                if self.mode == "txt2img"
                else stream.frame_bff_size,
                min_batch_size=self.batch_size
                if self.mode == "txt2img"
                else stream.frame_bff_size,
            ),
            "vae_decoder.engine",
        )

        if not os.path.exists(unet_path):
            os.makedirs(os.path.dirname(unet_path), exist_ok=True)
            # Use UNetXLSimple for SDXL models WITHOUT ControlNet support
            # This is a simplified version for initial TensorRT support
            # SDXL uses 2 text encoders: CLIP (768) + OpenCLIP (1280) = 2048 total
            unet_model = UNetXLSimple(
                fp16=True,
                device=stream.device,
                max_batch_size=stream.trt_unet_batch_size,
                min_batch_size=stream.trt_unet_batch_size,
                embedding_dim=2048,  # SDXL concatenated text embeddings (768 + 1280)
                unet_dim=stream.unet.config.in_channels,
            )
            compile_unet(
                stream.unet,
                unet_model,
                unet_path + ".onnx",
                unet_path + ".opt.onnx",
                unet_path,
                opt_batch_size=stream.trt_unet_batch_size,
                opt_image_height=self.height,  # Use actual resolution from Smode
                opt_image_width=self.width,
                is_sdxl=True,  # Enable SDXL support (added_cond_kwargs)
                use_simple_wrapper=True,  # Use simplified wrapper without ControlNet
            )

        if not os.path.exists(vae_decoder_path):
            os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
            stream.vae.forward = stream.vae.decode
            vae_decoder_model = VAE(
                device=stream.device,
                max_batch_size=self.batch_size
                if self.mode == "txt2img"
                else stream.frame_bff_size,
                min_batch_size=self.batch_size
                if self.mode == "txt2img"
                else stream.frame_bff_size,
            )
            compile_vae_decoder(
                stream.vae,
                vae_decoder_model,
                vae_decoder_path + ".onnx",
                vae_decoder_path + ".opt.onnx",
                vae_decoder_path,
                opt_batch_size=self.batch_size
                if self.mode == "txt2img"
                else stream.frame_bff_size,
                opt_image_height=self.height,  # Use actual resolution from Smode
                opt_image_width=self.width,
            )
            delattr(stream.vae, "forward")

        if not os.path.exists(vae_encoder_path):
            os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
            vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
            vae_encoder_model = VAEEncoder(
                device=stream.device,
                max_batch_size=self.batch_size
                if self.mode == "txt2img"
                else stream.frame_bff_size,
                min_batch_size=self.batch_size
                if self.mode == "txt2img"
                else stream.frame_bff_size,
            )
            compile_vae_encoder(
                vae_encoder,
                vae_encoder_model,
                vae_encoder_path + ".onnx",
                vae_encoder_path + ".opt.onnx",
                vae_encoder_path,
                opt_batch_size=self.batch_size
                if self.mode == "txt2img"
                else stream.frame_bff_size,
                opt_image_height=self.height,  # Use actual resolution from Smode
                opt_image_width=self.width,
            )

        cuda_stream = cuda.Stream()

        vae_config = stream.vae.config
        vae_dtype = stream.vae.dtype

        stream.unet = UNet2DConditionModelEngine(
            unet_path, cuda_stream, use_cuda_graph=False
        )
        stream.vae = AutoencoderKLEngine(
            vae_encoder_path,
            vae_decoder_path,
            cuda_stream,
            stream.pipe.vae_scale_factor,
            use_cuda_graph=False,
        )
        setattr(stream.vae, "config", vae_config)
        setattr(stream.vae, "dtype", vae_dtype)

        gc.collect()
        torch.cuda.empty_cache()

        logging.info("TensorRT acceleration enabled.")

    def cleanup(self):
        """
        CRITICAL FIX: Cleanup GPU resources when wrapper is destroyed.
        Prevents GPU memory leaks when wrapper instances are recreated.
        """
        try:
            # Free StreamDiffusion pipeline
            if hasattr(self, 'stream') and self.stream is not None:
                # Free large tensor buffers
                if hasattr(self.stream, 'x_t_latent_buffer'):
                    del self.stream.x_t_latent_buffer
                if hasattr(self.stream, 'init_noise'):
                    del self.stream.init_noise
                if hasattr(self.stream, 'stock_noise'):
                    del self.stream.stock_noise
                if hasattr(self.stream, 'prev_image_result'):
                    del self.stream.prev_image_result

                # Free cached scheduler coefficients (GPU tensors!)
                if hasattr(self.stream, '_scheduler_coeffs_cache'):
                    for coeffs in self.stream._scheduler_coeffs_cache.values():
                        for tensor in coeffs.values():
                            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                                del tensor
                    self.stream._scheduler_coeffs_cache.clear()

                # Free similar filter tensor
                if hasattr(self.stream, 'similar_filter') and self.stream.similar_filter is not None:
                    self.stream.similar_filter.prev_tensor = None

                del self.stream
                self.stream = None

            # Free safety checker
            if hasattr(self, 'safety_checker') and self.safety_checker is not None:
                del self.safety_checker
                self.safety_checker = None
            if hasattr(self, 'feature_extractor') and self.feature_extractor is not None:
                del self.feature_extractor
                self.feature_extractor = None

            # Force CUDA cache clear
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            # Don't raise exceptions during cleanup
            logging.warning(f"Warning: Error during cleanup: {e}")

    def __del__(self):
        """Destructor - ensure cleanup is called"""
        try:
            self.cleanup()
        except:
            pass  # Avoid errors during Python shutdown

