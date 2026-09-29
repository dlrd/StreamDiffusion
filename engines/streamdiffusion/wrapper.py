"""SD 1.5 StreamDiffusion wrapper."""
import gc
import os
from pathlib import Path
import traceback
import logging
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
from diffusers import AutoencoderTiny, LCMScheduler, StableDiffusionPipeline
from PIL import Image

from pipeline import StreamDiffusion
from .base_wrapper import BaseStreamDiffusionWrapper, PACKAGE_DIR, lora_signature
from utils.hub import local_first


def _compute_trt_unet_batch_size(t_index_list, frame_buffer_size, cfg_type, use_denoising_batch):
    """Mirror of StreamDiffusion.__init__ logic for trt_unet_batch_size."""
    denoising_steps_num = len(t_index_list)
    if use_denoising_batch:
        if cfg_type == "initialize":
            return (denoising_steps_num + 1) * frame_buffer_size
        elif cfg_type == "full":
            return 2 * denoising_steps_num * frame_buffer_size
        else:
            return denoising_steps_num * frame_buffer_size
    else:
        return frame_buffer_size


def _derive_engine_paths_sd15(
    model_id_or_path, use_lcm_lora, use_tiny_vae, lora_dict, engine_dir,
    trt_unet_batch_size, vae_batch_size, mode, height, width,
    streamv2v_on, streamv2v_maxframes,
):
    """Derive on-disk paths for the three TRT engines (unet, vae enc, vae dec).

    Height/width are folded into the prefix: static-shape TRT engines can't
    serve another resolution, so each resolution needs its own cache path.
    """
    lora_sig = lora_signature(lora_dict)

    def create_prefix(max_batch_size, min_batch_size):
        maybe_path = Path(model_id_or_path)
        stem = maybe_path.stem if maybe_path.exists() else model_id_or_path
        return (
            f"{stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}"
            f"--max_batch-{max_batch_size}--min_batch-{min_batch_size}"
            f"--mode-{mode}--res-{height}x{width}{lora_sig}"
        )

    engine_dir = Path(engine_dir)
    unet_filename = (
        f"unet_v2v_mf{streamv2v_maxframes}.engine" if streamv2v_on else "unet.engine"
    )
    unet_path = os.path.join(
        engine_dir,
        create_prefix(trt_unet_batch_size, trt_unet_batch_size),
        unet_filename,
    )
    vae_encoder_path = os.path.join(
        engine_dir,
        create_prefix(vae_batch_size, vae_batch_size),
        "vae_encoder.engine",
    )
    vae_decoder_path = os.path.join(
        engine_dir,
        create_prefix(vae_batch_size, vae_batch_size),
        "vae_decoder.engine",
    )
    return unet_path, vae_encoder_path, vae_decoder_path


def _build_stub_pipe_sd15(model_id_or_path, cache_dir, device, dtype):
    """Lightweight StableDiffusionPipeline with only the text-side components.

    Skips the UNet/VAE loads — saves ~1.7 GB transient VRAM peak when all TRT
    engines are already on disk.
    """
    from transformers import CLIPTextModel, CLIPTokenizer
    tokenizer = local_first(CLIPTokenizer.from_pretrained,
        model_id_or_path, subfolder="tokenizer",
        cache_dir=cache_dir if cache_dir else None,
    )
    text_encoder = local_first(CLIPTextModel.from_pretrained,
        model_id_or_path, subfolder="text_encoder",
        cache_dir=cache_dir if cache_dir else None,
        torch_dtype=dtype,
    ).to(device)
    scheduler = local_first(LCMScheduler.from_pretrained,
        model_id_or_path, subfolder="scheduler",
        cache_dir=cache_dir if cache_dir else None,
    )
    pipe = StableDiffusionPipeline(
        vae=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=None,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        image_encoder=None,
        requires_safety_checker=False,
    )
    # Skip ``pipe.to(device)`` — text_encoder already on device and iterating
    # would crash on None unet/vae in older diffusers.
    return pipe


class StreamDiffusionWrapper(BaseStreamDiffusionWrapper):
    """SD 1.5 wrapper."""

    def _get_default_engine_dir(self) -> Path:
        return PACKAGE_DIR / "tensorrt_cache" / "sd"

    def recreate_pipe(self):
        if not self.sd_turbo:
            self.stream.load_lcm_lora()
            self.stream.fuse_lora()

        self.stream.vae = local_first(AutoencoderTiny.from_pretrained, "madebyollin/taesd").to(
            device=self.stream.pipe.device, dtype=self.stream.pipe.dtype
        )

    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "tensorrt"] = "none",
        warmup: int = 10,
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> StreamDiffusion:
        if engine_dir is None:
            engine_dir = PACKAGE_DIR / "tensorrt_cache" / "sd"

        # ---- Pre-flight TRT engine cache check (fast path) ------------------
        # If every TRT engine is already on disk, skip the heavy PyTorch
        # pipeline load. Gated off when ANY heavy-path feature is active
        # (FaceID, StreamV2V, custom LoRAs, non-tinyVAE,
        # single-file checkpoint).
        fast_path_taken = False
        if acceleration == "tensorrt":
            faceid_enabled = False
            if self.faceid_config is not None:
                faceid_enabled = (
                    self.faceid_config.enabled if hasattr(self.faceid_config, "enabled")
                    else self.faceid_config.get("faceid_enabled", False)
                )

            v2v_on = bool(getattr(self, "streamv2v_enabled", False))
            v2v_maxframes = int(getattr(self, "streamv2v_cache_maxframes", 1))

            gate_reason = None
            if faceid_enabled:
                gate_reason = "FaceID enabled (forces torch.compile fallback)"
            elif v2v_on:
                gate_reason = "StreamV2V enabled (kvo processor install needs PyTorch UNet)"
            elif lora_dict:
                gate_reason = "custom lora_dict set"
            elif not use_tiny_vae:
                gate_reason = "use_tiny_vae=False (heavy path needs original VAE config)"
            elif Path(model_id_or_path).exists() and Path(model_id_or_path).is_file():
                gate_reason = "single-file checkpoint (no diffusers subfolder layout)"

            if gate_reason is None:
                trt_unet_bs = _compute_trt_unet_batch_size(
                    t_index_list, self.frame_buffer_size, cfg_type, self.use_denoising_batch,
                )
                vae_bs = self.batch_size if self.mode == "txt2img" else self.frame_buffer_size
                unet_path, vae_enc_path, vae_dec_path = _derive_engine_paths_sd15(
                    model_id_or_path=model_id_or_path,
                    use_lcm_lora=use_lcm_lora,
                    use_tiny_vae=use_tiny_vae,
                    lora_dict=lora_dict,
                    engine_dir=engine_dir,
                    trt_unet_batch_size=trt_unet_bs,
                    vae_batch_size=vae_bs,
                    mode=self.mode,
                    height=self.height, width=self.width,
                    streamv2v_on=v2v_on,
                    streamv2v_maxframes=v2v_maxframes,
                )
                if (os.path.exists(unet_path) and os.path.exists(vae_enc_path)
                        and os.path.exists(vae_dec_path)):
                    try:
                        logging.info(
                            "[Cache hit] All TRT engines present (SD 1.5) — using fast load path"
                        )
                        pipe = _build_stub_pipe_sd15(
                            model_id_or_path, cache_dir, self.device, self.dtype,
                        )
                        stream = StreamDiffusion(
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

                        is_turbo_model = "turbo" in model_id_or_path.lower()
                        if is_turbo_model:
                            stream.configure_scheduler(model_type="turbo")
                        else:
                            stream.configure_scheduler(model_type="default")

                        # TinyVAE provides a valid ``vae.config`` / ``.dtype`` for
                        # the TRT engine wrapper. Replaced immediately afterward.
                        stream.vae = local_first(AutoencoderTiny.from_pretrained,
                            vae_id if vae_id is not None else "madebyollin/taesd"
                        ).to(device=self.device, dtype=self.dtype)

                        self.enable_tensorrt_acceleration(
                            stream, model_id_or_path, use_lcm_lora, use_tiny_vae,
                            engine_dir, lora_dict=lora_dict,
                        )

                        if seed < 0:
                            seed = np.random.randint(0, 1000000)
                        stream.prepare(
                            "", "",
                            num_inference_steps=50,
                            guidance_scale=1.2 if stream.cfg_type in ["full", "self", "initialize"] else 1.0,
                            generator=torch.manual_seed(seed),
                            seed=seed,
                        )

                        if self.use_safety_checker:
                            from transformers import CLIPFeatureExtractor
                            from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
                            self.safety_checker = local_first(StableDiffusionSafetyChecker.from_pretrained,
                                "CompVis/stable-diffusion-safety-checker",
                                torch_dtype=self.dtype,
                            ).to(self.device)
                            self.feature_extractor = local_first(CLIPFeatureExtractor.from_pretrained, "openai/clip-vit-base-patch32")
                            self.nsfw_fallback_img = Image.new("RGB", (512, 512), (0, 0, 0))

                        fast_path_taken = True
                        return stream
                    except Exception as e:
                        logging.warning(
                            f"[Cache hit] Fast load path failed ({type(e).__name__}: {e}) — heavy path"
                        )
                        traceback.print_exc()
                        try:
                            del pipe
                        except Exception:
                            pass
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    missing = [
                        p for p in (unet_path, vae_enc_path, vae_dec_path)
                        if not os.path.exists(p)
                    ]
                    logging.info(
                        f"[Cache miss] {len(missing)} of 3 TRT engines missing — heavy path"
                    )
            else:
                logging.info(f"[Cache hit detected but fast path gated off: {gate_reason}]")

        try:
            try:
                pipe = local_first(StableDiffusionPipeline.from_pretrained,
                    model_id_or_path, torch_dtype=self.dtype,
                    cache_dir=cache_dir if cache_dir else None
                ).to(device=self.device, dtype=self.dtype)
            except Exception:
                try:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_id_or_path, local_files_only=True, torch_dtype=self.dtype,
                        cache_dir=cache_dir if cache_dir else None
                    ).to(device=self.device, dtype=self.dtype)
                except Exception:
                    pipe = local_first(StableDiffusionPipeline.from_single_file,
                        model_id_or_path, torch_dtype=self.dtype,
                        cache_dir=cache_dir if cache_dir else None
                    ).to(device=self.device)
        except Exception:
            traceback.print_exc()
            print("Model load has failed. Doesn't exist.")
            exit()

        stream = StreamDiffusion(
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

        is_turbo_model = "turbo" in model_id_or_path.lower()
        is_hyper_model = False
        if lora_dict is not None:
            for lora_name in lora_dict.keys():
                if "hyper" in lora_name.lower():
                    is_hyper_model = True
                    break

        if is_turbo_model:
            stream.configure_scheduler(model_type="turbo")
        elif is_hyper_model:
            stream.configure_scheduler(model_type="hyper", eta=0.0)
        else:
            stream.configure_scheduler(model_type="default")

        # LCM LoRA (skip for Hyper-SD/Turbo)
        if not self.sd_turbo and not is_hyper_model and not is_turbo_model:
            if use_lcm_lora:
                if lcm_lora_id is not None:
                    stream.load_lcm_lora(pretrained_model_name_or_path_or_dict=lcm_lora_id)
                else:
                    stream.load_lcm_lora()
                stream.fuse_lora()

        if lora_dict is not None:
            for lora_name, lora_scale in lora_dict.items():
                if "::" in lora_name:
                    repo_id, weight_name = lora_name.split("::", 1)
                    stream.load_lora(repo_id, weight_name=weight_name)
                    logging.info(f"[LoRA] Loading {repo_id} (file: {weight_name}) with weight {lora_scale}")
                else:
                    stream.load_lora(lora_name)
                    logging.info(f"[LoRA] Loading {lora_name} with weight {lora_scale}")
                stream.fuse_lora(lora_scale=lora_scale)

        if use_tiny_vae:
            if vae_id is not None:
                stream.vae = local_first(AutoencoderTiny.from_pretrained, vae_id).to(
                    device=pipe.device, dtype=pipe.dtype
                )
            else:
                stream.vae = local_first(AutoencoderTiny.from_pretrained, "madebyollin/taesd").to(
                    device=pipe.device, dtype=pipe.dtype
                )
        elif is_hyper_model:
            if vae_id is not None:
                from diffusers import AutoencoderKL
                stream.vae = local_first(AutoencoderKL.from_pretrained, vae_id).to(
                    device=pipe.device, dtype=pipe.dtype
                )
            elif "noVAE" in model_id_or_path or "novae" in model_id_or_path.lower():
                from diffusers import AutoencoderKL
                stream.vae = local_first(AutoencoderKL.from_pretrained,
                    "stabilityai/sd-vae-ft-mse", torch_dtype=pipe.dtype
                ).to(device=pipe.device)

        # IP-Adapter FaceID must load BEFORE torch.compile / TensorRT.
        # Incompatible with TRT (changes UNet architecture) → fall back.
        if self.faceid_config is not None:
            faceid_enabled = (self.faceid_config.enabled if hasattr(self.faceid_config, 'enabled')
                              else self.faceid_config.get('faceid_enabled', False))
            if faceid_enabled and acceleration == "tensorrt":
                logging.warning("[FaceID] Incompatible with TensorRT, falling back to torch.compile")
                acceleration = "none"
            self.load_ip_adapter_faceid(pipe, self.faceid_config, is_sdxl=False)

        try:
            if acceleration == "tensorrt":
                self.enable_tensorrt_acceleration(
                    stream, model_id_or_path, use_lcm_lora, use_tiny_vae,
                    engine_dir, lora_dict=lora_dict,
                )
            if acceleration == "sfast":
                from pipeline.acceleration.sfast import accelerate_with_stable_fast
                stream = accelerate_with_stable_fast(stream)
            if acceleration == "torchcompile":
                self.setup_torch_compile(stream, acceleration, cache_subdir="sd")
        except Exception:
            traceback.print_exc()
            logging.warning("Acceleration failed. Falling back to normal mode.")
            self._emit_warning(False)

        # Free the orphaned original SD 1.5 KL VAE on the PyTorch paths (~160 MB
        # fp16) after the TinyVAE swap — same rationale as the SDXL wrapper. TRT
        # drops it inside enable_tensorrt_acceleration; gate to none/xformers and
        # skip when no swap happened (stream.vae IS pipe.vae, e.g. built-in VAE).
        if acceleration in ("none", "xformers"):
            orphan_vae = getattr(stream.pipe, "vae", None)
            if orphan_vae is not None and orphan_vae is not stream.vae:
                try:
                    orphan_vae.to("cpu")
                    torch.cuda.empty_cache()
                    logging.info("[VAE] Moved orphaned original KL VAE to CPU (TinyVAE active)")
                except Exception:
                    pass

        if seed < 0:
            seed = np.random.randint(0, 1000000)

        stream.prepare(
            "", "",
            num_inference_steps=50,
            guidance_scale=1.2 if stream.cfg_type in ["full", "self", "initialize"] else 1.0,
            generator=torch.manual_seed(seed),
            seed=seed,
        )

        if self.use_safety_checker:
            from transformers import CLIPFeatureExtractor
            from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
            self.safety_checker = local_first(StableDiffusionSafetyChecker.from_pretrained,
                "CompVis/stable-diffusion-safety-checker",
                torch_dtype=self.dtype,
            ).to(pipe.device)
            self.feature_extractor = local_first(CLIPFeatureExtractor.from_pretrained, "openai/clip-vit-base-patch32")
            self.nsfw_fallback_img = Image.new("RGB", (512, 512), (0, 0, 0))

        return stream

    def enable_tensorrt_acceleration(
        self, stream: StreamDiffusion, model_id_or_path: str,
        use_lcm_lora: bool, use_tiny_vae: bool,
        engine_dir: Optional[Union[str, Path]] = None,
        lora_dict: Optional[Dict[str, float]] = None,
    ):
        if engine_dir is None:
            engine_dir = PACKAGE_DIR / "tensorrt_cache" / "sd"

        from polygraphy import cuda
        from pipeline.acceleration.tensorrt import (
            TorchVAEEncoder, compile_unet, compile_vae_decoder, compile_vae_encoder,
        )
        from pipeline.acceleration.tensorrt.engine import (
            AutoencoderKLEngine, UNet2DConditionModelEngine,
        )
        from pipeline.acceleration.tensorrt.models import VAE, UNet, UNetV2V, VAEEncoder

        lora_sig = lora_signature(lora_dict)
        v2v_on = bool(getattr(self, "streamv2v_enabled", False))
        v2v_maxframes = int(getattr(self, "streamv2v_cache_maxframes", 1))

        def create_prefix(model_id_or_path, max_batch_size, min_batch_size):
            maybe_path = Path(model_id_or_path)
            stem = maybe_path.stem if maybe_path.exists() else model_id_or_path
            return (
                f"{stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}"
                f"--max_batch-{max_batch_size}--min_batch-{min_batch_size}"
                f"--mode-{self.mode}--res-{self.height}x{self.width}{lora_sig}"
            )

        engine_dir = Path(engine_dir)
        # Distinct filename for the v2v engine so a stale plain engine is
        # NEVER loaded as v2v (their bindings are incompatible).
        unet_filename = f"unet_v2v_mf{v2v_maxframes}.engine" if v2v_on else "unet.engine"
        unet_path = os.path.join(
            engine_dir,
            create_prefix(model_id_or_path, stream.trt_unet_batch_size, stream.trt_unet_batch_size),
            unet_filename,
        )
        vae_encoder_path = os.path.join(
            engine_dir,
            create_prefix(model_id_or_path,
                          self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                          self.batch_size if self.mode == "txt2img" else stream.frame_bff_size),
            "vae_encoder.engine",
        )
        vae_decoder_path = os.path.join(
            engine_dir,
            create_prefix(model_id_or_path,
                          self.batch_size if self.mode == "txt2img" else stream.frame_bff_size,
                          self.batch_size if self.mode == "txt2img" else stream.frame_bff_size),
            "vae_decoder.engine",
        )

        needs_build = not (
            os.path.exists(unet_path) and os.path.exists(vae_decoder_path)
            and os.path.exists(vae_encoder_path)
        )

        if not os.path.exists(unet_path):
            self._emit_warning(True, "Building TensorRT engine (UNet) - first run can take several minutes")
            os.makedirs(os.path.dirname(unet_path), exist_ok=True)
            if v2v_on:
                # Install kvo passthrough processors BEFORE export so the ONNX
                # trace captures the modified attention behavior.
                from pipeline.attention_processors import (
                    install_kvo_processors, get_kvo_cache_info,
                )
                kvo_procs = install_kvo_processors(
                    stream.unet,
                    max_frames=v2v_maxframes,
                    use_feature_injection=True,
                )
                kvo_shapes, kvo_structure, _ = get_kvo_cache_info(
                    stream.unet, self.height, self.width,
                )
                logging.info(
                    f"[StreamV2V TRT] Installed {len(kvo_procs)} kvo passthrough "
                    f"processors, structure={kvo_structure}, max_cache_frames={v2v_maxframes}"
                )
                unet_model = UNetV2V(
                    kvo_cache_shapes=kvo_shapes,
                    max_cache_frames=v2v_maxframes,
                    fp16=True, device=stream.device,
                    max_batch_size=stream.trt_unet_batch_size,
                    min_batch_size=stream.trt_unet_batch_size,
                    embedding_dim=stream.text_encoder.config.hidden_size,
                    unet_dim=stream.unet.config.in_channels,
                )
                compile_unet(stream.unet, unet_model, unet_path + ".onnx",
                             unet_path + ".opt.onnx", unet_path,
                             opt_batch_size=stream.trt_unet_batch_size,
                             opt_image_height=self.height, opt_image_width=self.width,
                             kvo_processors=kvo_procs)
            else:
                unet_model = UNet(
                    fp16=True, device=stream.device,
                    max_batch_size=stream.trt_unet_batch_size,
                    min_batch_size=stream.trt_unet_batch_size,
                    embedding_dim=stream.text_encoder.config.hidden_size,
                    unet_dim=stream.unet.config.in_channels,
                )
                compile_unet(stream.unet, unet_model, unet_path + ".onnx",
                             unet_path + ".opt.onnx", unet_path,
                             opt_batch_size=stream.trt_unet_batch_size,
                             opt_image_height=self.height, opt_image_width=self.width)
            # Move PyTorch UNet off-GPU as soon as TRT engine is built so the
            # VAE builds don't pay the ~1.7 GB peak.
            try:
                stream.unet.to("cpu")
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

        if not os.path.exists(vae_decoder_path):
            self._emit_warning(True, "Building TensorRT engine (VAE decoder) - first run can take a few minutes")
            os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
            stream.vae.forward = stream.vae.decode
            batch = self.batch_size if self.mode == "txt2img" else stream.frame_bff_size
            vae_decoder_model = VAE(device=stream.device, max_batch_size=batch, min_batch_size=batch)
            compile_vae_decoder(stream.vae, vae_decoder_model, vae_decoder_path + ".onnx",
                                vae_decoder_path + ".opt.onnx", vae_decoder_path, opt_batch_size=batch,
                                opt_image_height=self.height, opt_image_width=self.width)
            delattr(stream.vae, "forward")

        if not os.path.exists(vae_encoder_path):
            self._emit_warning(True, "Building TensorRT engine (VAE encoder) - first run can take a few minutes")
            os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
            vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
            batch = self.batch_size if self.mode == "txt2img" else stream.frame_bff_size
            vae_encoder_model = VAEEncoder(device=stream.device, max_batch_size=batch, min_batch_size=batch)
            compile_vae_encoder(vae_encoder, vae_encoder_model, vae_encoder_path + ".onnx",
                                vae_encoder_path + ".opt.onnx", vae_encoder_path, opt_batch_size=batch,
                                opt_image_height=self.height, opt_image_width=self.width)

        cuda_stream = cuda.Stream()
        vae_scale_factor = stream.pipe.vae_scale_factor
        vae_config = stream.vae.config
        vae_dtype = stream.vae.dtype

        # Free PyTorch UNet + VAE(s) BEFORE loading TRT engines: on a cache
        # hit they'd otherwise stay on GPU, doubling the transient peak.
        pytorch_unet = stream.unet
        pytorch_vae_active = stream.vae
        pipe_vae_original = getattr(stream.pipe, "vae", None)
        if pipe_vae_original is pytorch_vae_active:
            pipe_vae_original = None
        for _ref in (pytorch_unet, pytorch_vae_active, pipe_vae_original):
            if _ref is None:
                continue
            try:
                _ref.to("cpu")
            except Exception:
                pass
        if hasattr(stream.pipe, "unet"):
            del stream.pipe.unet
        if hasattr(stream.pipe, "vae"):
            del stream.pipe.vae
        stream.unet = None
        stream.vae = None
        del pytorch_unet, pytorch_vae_active
        if pipe_vae_original is not None:
            del pipe_vae_original
        gc.collect()
        torch.cuda.empty_cache()

        # CUDA Graph capture saves ~1-2 ms/frame on stable shapes; disabled
        # for StreamV2V (per-frame kvo cache copy_ not yet verified safe).
        unet_use_cuda_graph = not v2v_on
        stream.unet = UNet2DConditionModelEngine(
            unet_path, cuda_stream, use_cuda_graph=unet_use_cuda_graph,
            v2v_cache_maxframes=v2v_maxframes,
        )
        stream.vae = AutoencoderKLEngine(
            vae_encoder_path, vae_decoder_path, cuda_stream,
            vae_scale_factor, use_cuda_graph=True,
        )
        setattr(stream.vae, "config", vae_config)
        setattr(stream.vae, "dtype", vae_dtype)

        gc.collect()
        torch.cuda.empty_cache()
        logging.info("TensorRT acceleration enabled (SD 1.5).")

        try:
            alloc_gb = torch.cuda.memory_allocated() / 1e9
            reserv_gb = torch.cuda.memory_reserved() / 1e9
            logging.info(
                f"[VRAM] after TRT init: allocated={alloc_gb:.2f} GB, "
                f"reserved={reserv_gb:.2f} GB (gap={reserv_gb - alloc_gb:.2f} GB)"
            )
        except Exception:
            pass

        if needs_build:
            self._emit_warning(False)
