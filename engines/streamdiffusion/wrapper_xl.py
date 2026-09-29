"""SDXL StreamDiffusion wrapper."""
import gc
import os
import time
from pathlib import Path
import traceback
from typing import Dict, List, Literal, Optional, Union
import logging

import numpy as np
import torch
from diffusers import AutoencoderTiny, LCMScheduler, StableDiffusionXLPipeline
from PIL import Image

from pipeline.pipeline_xl import StreamDiffusionXL
from .base_wrapper import BaseStreamDiffusionWrapper, PACKAGE_DIR, lora_signature
from utils.hub import local_first


def _compute_trt_unet_batch_size_xl(t_index_list, frame_buffer_size, cfg_type, use_denoising_batch):
    """Mirror of StreamDiffusionXL.__init__ logic for trt_unet_batch_size."""
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


def _derive_engine_paths_sdxl(
    model_id_or_path, use_lcm_lora, use_tiny_vae, lora_dict, engine_dir,
    trt_unet_batch_size, vae_batch_size, mode, height, width,
    streamv2v_on=False, streamv2v_maxframes=1,
):
    """Derive on-disk paths for the SDXL TRT engines.

    UNet filename is ``unet_cn.engine`` for the CN-enabled UNet, or
    ``unet_v2v_xl_mfN.engine`` when StreamV2V is enabled (kvo I/O ports).
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
        f"unet_v2v_xl_mf{streamv2v_maxframes}.engine" if streamv2v_on else "unet_cn.engine"
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


def _build_stub_pipe_sdxl(model_id_or_path, cache_dir, device, dtype):
    """Lightweight SDXL pipeline with only text-side components (two encoders + tokenizers).

    Skips the ~5 GB UNet + 300 MB VAE — the two text encoders (~1.65 GB)
    can't be skipped because ``encode_prompt`` needs them.
    """
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    tokenizer = local_first(CLIPTokenizer.from_pretrained,
        model_id_or_path, subfolder="tokenizer",
        cache_dir=cache_dir if cache_dir else None,
    )
    tokenizer_2 = local_first(CLIPTokenizer.from_pretrained,
        model_id_or_path, subfolder="tokenizer_2",
        cache_dir=cache_dir if cache_dir else None,
    )
    text_encoder = local_first(CLIPTextModel.from_pretrained,
        model_id_or_path, subfolder="text_encoder",
        cache_dir=cache_dir if cache_dir else None,
        torch_dtype=dtype,
    ).to(device)
    text_encoder_2 = local_first(CLIPTextModelWithProjection.from_pretrained,
        model_id_or_path, subfolder="text_encoder_2",
        cache_dir=cache_dir if cache_dir else None,
        torch_dtype=dtype,
    ).to(device)
    scheduler = local_first(LCMScheduler.from_pretrained,
        model_id_or_path, subfolder="scheduler",
        cache_dir=cache_dir if cache_dir else None,
    )
    pipe = StableDiffusionXLPipeline(
        vae=None,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=None,
        scheduler=scheduler,
        image_encoder=None,
        feature_extractor=None,
    )
    return pipe


class StreamDiffusionWrapperXL(BaseStreamDiffusionWrapper):
    """SDXL wrapper."""

    def _get_default_engine_dir(self) -> Path:
        return PACKAGE_DIR / "tensorrt_cache" / "sdxl"

    def recreate_pipe(self):
        if not self.sd_turbo:
            self.stream.load_lcm_lora()
            self.stream.fuse_lora()

        self.stream.vae = local_first(AutoencoderTiny.from_pretrained, "madebyollin/taesdxl").to(
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
    ) -> StreamDiffusionXL:
        if engine_dir is None:
            engine_dir = PACKAGE_DIR / "tensorrt_cache" / "sdxl"

        # Hyper-SDXL U-Net mode (format: "base-model/hyper-sdxl-unet")
        use_hyper_unet = "hyper-sdxl-unet" in model_id_or_path.lower()
        base_model_path = model_id_or_path

        if use_hyper_unet:
            base_model_path = model_id_or_path.split("/hyper-sdxl-unet")[0]
            logging.info(f"[Hyper-SDXL U-Net] Using base model: {base_model_path}")

        # ---- Pre-flight TRT engine cache check (fast path) ------------------
        # Skip the heavy ~7 GB SDXL pipeline load when all TRT engines are
        # cached. Gated off for: FaceID, custom LoRAs, use_tiny_vae=False,
        # Hyper-SDXL U-Net checkpoint mode, single-file, non-tensorrt.
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
            elif use_hyper_unet:
                gate_reason = "Hyper-SDXL U-Net checkpoint mode (overwrites pipe.unet)"
            elif lora_dict:
                gate_reason = "custom lora_dict set"
            elif not use_tiny_vae:
                gate_reason = "use_tiny_vae=False (heavy path needs original VAE config)"
            elif Path(base_model_path).exists() and Path(base_model_path).is_file():
                gate_reason = "single-file checkpoint (no diffusers subfolder layout)"

            if gate_reason is None:
                trt_unet_bs = _compute_trt_unet_batch_size_xl(
                    t_index_list, self.frame_buffer_size, cfg_type, self.use_denoising_batch,
                )
                vae_bs = self.batch_size if self.mode == "txt2img" else self.frame_buffer_size
                unet_path, vae_enc_path, vae_dec_path = _derive_engine_paths_sdxl(
                    model_id_or_path=model_id_or_path,
                    use_lcm_lora=use_lcm_lora,
                    use_tiny_vae=use_tiny_vae,
                    lora_dict=lora_dict,
                    engine_dir=engine_dir,
                    trt_unet_batch_size=trt_unet_bs,
                    vae_batch_size=vae_bs,
                    mode=self.mode,
                    height=self.height, width=self.width,
                )
                if (os.path.exists(unet_path) and os.path.exists(vae_enc_path)
                        and os.path.exists(vae_dec_path)):
                    try:
                        logging.info(
                            "[Cache hit] All TRT engines present (SDXL) — using fast load path"
                        )
                        pipe = _build_stub_pipe_sdxl(
                            base_model_path, cache_dir, self.device, self.dtype,
                        )
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

                        is_lightning_model = "lightning" in model_id_or_path.lower()
                        is_hyper_model = "hyper" in model_id_or_path.lower()
                        if self.sd_turbo:
                            stream.configure_scheduler(model_type="turbo")
                        elif is_hyper_model:
                            stream.configure_scheduler(model_type="hyper", eta=0.0)
                        elif is_lightning_model:
                            stream.configure_scheduler(model_type="lightning")
                        else:
                            stream.configure_scheduler(model_type="default")

                        tiny_vae_id = (
                            vae_id if vae_id is not None else "cqyan/hybrid-sd-tinyvae-xl"
                        )
                        try:
                            stream.vae = local_first(AutoencoderTiny.from_pretrained, tiny_vae_id).to(
                                device=self.device, dtype=self.dtype
                            )
                        except Exception:
                            stream.vae = local_first(AutoencoderTiny.from_pretrained,
                                "madebyollin/taesdxl"
                            ).to(device=self.device, dtype=self.dtype)
                        # Mirror _configure_vae fixes so the TRT engine inherits the right config.
                        if getattr(stream.vae.config, "scaling_factor", None) is None:
                            stream.vae.config.scaling_factor = 1.0
                        if (not hasattr(stream.vae.config, "shift_factor")
                                or stream.vae.config.shift_factor is None):
                            stream.vae.config.shift_factor = 0.0

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
                pipe = local_first(StableDiffusionXLPipeline.from_pretrained,
                    base_model_path, torch_dtype=self.dtype,
                    cache_dir=cache_dir if cache_dir else None
                ).to(device=self.device, dtype=self.dtype)
            except Exception:
                try:
                    pipe = StableDiffusionXLPipeline.from_pretrained(
                        base_model_path, local_files_only=True, torch_dtype=self.dtype,
                        cache_dir=cache_dir if cache_dir else None
                    ).to(device=self.device, dtype=self.dtype)
                except Exception:
                    pipe = local_first(StableDiffusionXLPipeline.from_single_file,
                        base_model_path, torch_dtype=self.dtype,
                        cache_dir=cache_dir if cache_dir else None
                    ).to(device=self.device)
        except Exception:
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

        is_lightning_model = "lightning" in model_id_or_path.lower()
        is_hyper_model = "hyper" in model_id_or_path.lower()

        if use_hyper_unet:
            self._load_hyper_unet(stream, cache_dir)
            is_hyper_model = True

        if lora_dict is not None:
            for lora_name in lora_dict.keys():
                if "hyper" in lora_name.lower():
                    is_hyper_model = True
                    break

        if self.sd_turbo:
            stream.configure_scheduler(model_type="turbo")
        elif is_hyper_model:
            stream.configure_scheduler(model_type="hyper", eta=0.0, use_checkpoint_unet=use_hyper_unet)
        elif is_lightning_model:
            stream.configure_scheduler(model_type="lightning")
        else:
            stream.configure_scheduler(model_type="default")

        if not self.sd_turbo:
            if use_lcm_lora and not is_lightning_model and not is_hyper_model:
                if lcm_lora_id is not None:
                    stream.load_lcm_lora(pretrained_model_name_or_path_or_dict=lcm_lora_id)
                else:
                    stream.load_lcm_lora()
                stream.fuse_lora()

        if lora_dict is not None:
            self._load_loras(stream, lora_dict, cache_dir)

        self._configure_vae(stream, pipe, use_tiny_vae, vae_id, cache_dir)

        # IP-Adapter FaceID must load BEFORE torch.compile / TensorRT.
        if self.faceid_config is not None:
            faceid_enabled = (self.faceid_config.enabled if hasattr(self.faceid_config, 'enabled')
                              else self.faceid_config.get('faceid_enabled', False))
            if faceid_enabled and acceleration == "tensorrt":
                logging.warning("[FaceID] Incompatible with TensorRT, falling back to torch.compile")
                acceleration = "none"
            self.load_ip_adapter_faceid(pipe, self.faceid_config, is_sdxl=True)

        try:
            if acceleration == "tensorrt":
                self.enable_tensorrt_acceleration(
                    stream, model_id_or_path, use_lcm_lora, use_tiny_vae,
                    engine_dir, lora_dict=lora_dict,
                )
            if acceleration == "sfast":
                from pipeline.acceleration.sfast import accelerate_with_stable_fast
                stream = accelerate_with_stable_fast(stream)
        except Exception:
            traceback.print_exc()
            logging.warning("Acceleration failed. Falling back to normal mode.")
            self._emit_warning(False)

        # Free the orphaned original SDXL KL VAE on the PyTorch paths. After
        # _configure_vae swapped stream.vae to TinyVAE, the full KL VAE
        # (~160-320 MB fp16) stays resident via stream.pipe.vae but is never
        # used at runtime. The TRT path already drops it inside
        # enable_tensorrt_acceleration; the sfast path re-binds
        # stream.vae = stream.pipe.vae and still needs it — so gate to
        # none/xformers. Keep .to('cpu') (not del) so pipe.vae_scale_factor /
        # config stay readable; pipe.device stays cuda (vae is last in the
        # sorted signature keys, text_encoder determines device).
        if acceleration in ("none", "xformers"):
            orphan_vae = getattr(stream.pipe, "vae", None)
            if orphan_vae is not None and orphan_vae is not stream.vae:
                try:
                    orphan_vae.to("cpu")
                    torch.cuda.empty_cache()
                    logging.info("[VAE] Moved orphaned original KL VAE to CPU (TinyVAE active)")
                except Exception:
                    pass

        self._setup_torch_compile_sdxl(stream, acceleration)

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

    def _load_hyper_unet(self, stream, cache_dir):
        """Load full Hyper-SDXL U-Net checkpoint (bypasses LoRA)."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        unet_path = local_first(hf_hub_download,
            repo_id="ByteDance/Hyper-SD",
            filename="Hyper-SDXL-1step-Unet.safetensors",
            cache_dir=cache_dir if cache_dir else None
        )

        state_dict = load_file(unet_path)
        missing, unexpected = stream.pipe.unet.load_state_dict(state_dict, strict=False)
        if missing:
            logging.warning(f"[Hyper-SDXL U-Net] Missing keys: {len(missing)}")
        if unexpected:
            logging.warning(f"[Hyper-SDXL U-Net] Unexpected keys: {len(unexpected)}")

        stream.unet = stream.pipe.unet
        stream.use_hyper_unet_checkpoint = True

    def _load_loras(self, stream, lora_dict, cache_dir):
        """Load custom LoRAs (Hyper-SDXL Kohya format handled specially)."""
        from huggingface_hub import hf_hub_download

        for lora_name, lora_scale in lora_dict.items():
            if "::" in lora_name:
                repo_id, weight_name = lora_name.split("::", 1)
                if "hyper" in lora_name.lower() and "sdxl" in lora_name.lower():
                    lora_path = local_first(hf_hub_download, repo_id, weight_name)
                    stream.load_lora(lora_path)
                else:
                    stream.load_lora(repo_id, weight_name=weight_name)
            else:
                stream.load_lora(lora_name)
            stream.fuse_lora(lora_scale=lora_scale)

    def _configure_vae(self, stream, pipe, use_tiny_vae, vae_id, cache_dir):
        """Configure SDXL VAE (hybrid TinyVAE / taesdxl + scaling factors)."""
        if use_tiny_vae:
            if vae_id is not None:
                vae_model_name = vae_id
            else:
                vae_model_name = "cqyan/hybrid-sd-tinyvae-xl"

            try:
                stream.vae = local_first(AutoencoderTiny.from_pretrained, vae_model_name).to(
                    device=pipe.device, dtype=pipe.dtype
                )
            except Exception as e:
                logging.warning(f"[TinyVAE] Failed to load {vae_model_name}: {e}, falling back to taesdxl")
                stream.vae = local_first(AutoencoderTiny.from_pretrained, "madebyollin/taesdxl").to(
                    device=pipe.device, dtype=pipe.dtype
                )

            if getattr(stream.vae.config, 'scaling_factor', None) is None:
                stream.vae.config.scaling_factor = 1.0
            if not hasattr(stream.vae.config, 'shift_factor') or stream.vae.config.shift_factor is None:
                stream.vae.config.shift_factor = 0.0

    def _setup_torch_compile_sdxl(self, stream, acceleration):
        """SDXL-specific torch.compile with resolution-keyed cache directories."""
        if not self.torch_compile_enabled:
            return
        if acceleration != "torchcompile":
            return
        if not hasattr(torch, 'compile') or torch.__version__ < '2.0':
            return

        self._emit_warning(True, "Compiling with torch.compile() - first run can take 1-2 minutes")
        try:
            resolution_str = f"{self.width}x{self.height}"
            cache_dir = PACKAGE_DIR / f"torch_compile_cache/sdxl_{resolution_str}"
            cache_dir.mkdir(parents=True, exist_ok=True)

            os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache_dir)
            os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'

            compile_start = time.time()
            stream.unet = torch.compile(
                stream.unet,
                mode=self.torch_compile_mode,
                fullgraph=getattr(self, 'torch_compile_fullgraph', False),
                dynamic=False
            )
            logging.info(f"U-Net compiled in {time.time() - compile_start:.1f}s")

            if hasattr(stream.vae, 'encoder'):
                self._compile_vae_component(
                    stream.vae, 'encoder',
                    PACKAGE_DIR / f"torch_compile_cache/sdxl_vae_encoder_{resolution_str}",
                    dummy_shape=(1, 3, self.height, self.width),
                )

            if hasattr(stream.vae, 'decoder'):
                self._compile_vae_component(
                    stream.vae, 'decoder',
                    PACKAGE_DIR / f"torch_compile_cache/sdxl_vae_decoder_{resolution_str}",
                    dummy_shape=(1, 4, self.height // 8, self.width // 8),
                )

        except Exception as e:
            logging.warning(f"torch.compile() failed: {e}. Continuing without compilation.")
        finally:
            self._emit_warning(False)

    def enable_tensorrt_acceleration(
        self, stream: StreamDiffusionXL, model_id_or_path: str,
        use_lcm_lora: bool, use_tiny_vae: bool,
        engine_dir: Optional[Union[str, Path]] = None,
        lora_dict: Optional[Dict[str, float]] = None,
    ):
        if engine_dir is None:
            engine_dir = PACKAGE_DIR / "tensorrt_cache" / "sdxl"

        from polygraphy import cuda
        from pipeline.acceleration.tensorrt import (
            TorchVAEEncoder, compile_unet, compile_vae_decoder, compile_vae_encoder,
        )
        from pipeline.acceleration.tensorrt.engine import (
            AutoencoderKLEngine, UNet2DConditionModelEngine,
        )
        from pipeline.acceleration.tensorrt.models import (
            VAE, UNetXL, UNetXLV2V, VAEEncoder,
        )

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
        # Distinct filenames per variant — bindings are incompatible between
        # plain CN and v2v, so a stale engine must never be silently reused.
        unet_filename = (
            f"unet_v2v_xl_mf{v2v_maxframes}.engine" if v2v_on else "unet_cn.engine"
        )
        unet_path = os.path.join(
            engine_dir,
            create_prefix(model_id_or_path, stream.trt_unet_batch_size, stream.trt_unet_batch_size),
            unet_filename,
        )
        batch = self.batch_size if self.mode == "txt2img" else stream.frame_bff_size
        vae_encoder_path = os.path.join(
            engine_dir, create_prefix(model_id_or_path, batch, batch), "vae_encoder.engine",
        )
        vae_decoder_path = os.path.join(
            engine_dir, create_prefix(model_id_or_path, batch, batch), "vae_decoder.engine",
        )

        needs_build = not (
            os.path.exists(unet_path) and os.path.exists(vae_decoder_path)
            and os.path.exists(vae_encoder_path)
        )

        if not os.path.exists(unet_path):
            self._emit_warning(True, "Building TensorRT engine (UNet) - first run can take several minutes")
            os.makedirs(os.path.dirname(unet_path), exist_ok=True)
            if v2v_on:
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
                    f"[StreamV2V TRT XL] Installed {len(kvo_procs)} kvo passthrough processors, "
                    f"structure={kvo_structure}, max_cache_frames={v2v_maxframes}"
                )
                unet_model = UNetXLV2V(
                    kvo_cache_shapes=kvo_shapes,
                    max_cache_frames=v2v_maxframes,
                    fp16=True, device=stream.device,
                    max_batch_size=stream.trt_unet_batch_size,
                    min_batch_size=stream.trt_unet_batch_size,
                    embedding_dim=2048,
                    unet_dim=stream.unet.config.in_channels,
                )
                compile_unet(
                    stream.unet, unet_model, unet_path + ".onnx",
                    unet_path + ".opt.onnx", unet_path,
                    opt_batch_size=stream.trt_unet_batch_size,
                    opt_image_height=self.height, opt_image_width=self.width,
                    is_sdxl=True, kvo_processors=kvo_procs,
                )
            else:
                # UNetXL = SDXL UNet with ControlNet residual ports (9 down + 1 mid).
                # CN ports built in always (~1-3% overhead when no CN active).
                unet_model = UNetXL(
                    fp16=True, device=stream.device,
                    max_batch_size=stream.trt_unet_batch_size,
                    min_batch_size=stream.trt_unet_batch_size,
                    embedding_dim=2048,
                    unet_dim=stream.unet.config.in_channels,
                )
                compile_unet(
                    stream.unet, unet_model, unet_path + ".onnx",
                    unet_path + ".opt.onnx", unet_path,
                    opt_batch_size=stream.trt_unet_batch_size,
                    opt_image_height=self.height, opt_image_width=self.width,
                    is_sdxl=True, use_simple_wrapper=False,
                )
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
            vae_decoder_model = VAE(device=stream.device, max_batch_size=batch, min_batch_size=batch)
            compile_vae_decoder(
                stream.vae, vae_decoder_model, vae_decoder_path + ".onnx",
                vae_decoder_path + ".opt.onnx", vae_decoder_path,
                opt_batch_size=batch, opt_image_height=self.height, opt_image_width=self.width,
            )
            delattr(stream.vae, "forward")

        if not os.path.exists(vae_encoder_path):
            self._emit_warning(True, "Building TensorRT engine (VAE encoder) - first run can take a few minutes")
            os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
            vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
            vae_encoder_model = VAEEncoder(device=stream.device, max_batch_size=batch, min_batch_size=batch)
            compile_vae_encoder(
                vae_encoder, vae_encoder_model, vae_encoder_path + ".onnx",
                vae_encoder_path + ".opt.onnx", vae_encoder_path,
                opt_batch_size=batch, opt_image_height=self.height, opt_image_width=self.width,
            )

        cuda_stream = cuda.Stream()
        vae_scale_factor = stream.pipe.vae_scale_factor
        vae_config = stream.vae.config
        vae_dtype = stream.vae.dtype

        # Free PyTorch UNet + VAE BEFORE TRT engine load: on a cache hit
        # they'd otherwise stay on GPU, spiking VRAM to ~15 GB on SDXL.
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

        # CUDA Graph capture: stable kvo cache pointers (engine.py pre-alloc)
        # make V2V graph-capturable. Engine.infer falls back gracefully on
        # capture failure.
        unet_use_cuda_graph = True
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
        logging.info("TensorRT acceleration enabled (SDXL).")

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
