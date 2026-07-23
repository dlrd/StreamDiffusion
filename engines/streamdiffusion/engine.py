"""StreamDiffusion SD 1.5 / SDXL engine adapter."""
from __future__ import annotations

import gc
import logging
from typing import Any, Dict, Optional, Union

import torch

from ..base_engine import BaseEngine
from .wrapper import StreamDiffusionWrapper
from .wrapper_xl import StreamDiffusionWrapperXL


_SD2_KEYWORDS = ["sd-turbo", "sd_turbo", "2.0", "2.1", "2-1", "stabilityai/sd-turbo"]
_SDXL_KEYWORDS = ["sdxl", "xl", "sd-xl", "sd_xl"]


class StreamDiffusionEngine(BaseEngine):
    """Adapter for the StreamDiffusion SD 1.5 / SDXL pipelines."""

    def __init__(
        self,
        wrapper: Optional[Union[StreamDiffusionWrapper, StreamDiffusionWrapperXL]] = None,
    ) -> None:
        super().__init__()
        self.is_sdxl: bool = False
        self.is_sd2: bool = False
        self.num_inference_steps: int = 50
        if wrapper is not None:
            self.wrapper = wrapper
            self._loaded = True

    @property
    def name(self) -> str:
        return "streamdiffusion"

    def load(self, config: Dict[str, Any], **runtime: Any) -> None:
        model_name = runtime["model_name"]

        self.is_sdxl = any(kw in model_name.lower() for kw in _SDXL_KEYWORDS)
        self.is_sd2 = any(kw in model_name.lower() for kw in _SD2_KEYWORDS)
        is_lightning = "lightning" in model_name.lower()

        WrapperClass = StreamDiffusionWrapperXL if self.is_sdxl else StreamDiffusionWrapper
        model_type = "SDXL" if self.is_sdxl else "SD"
        logging.info(f"[Pipeline] Using {model_type} pipeline for model: {model_name}")

        if self.is_sdxl and is_lightning:
            self.num_inference_steps = 8
            logging.info(
                f"[SDXL Lightning] Using num_inference_steps={self.num_inference_steps}"
            )
        else:
            self.num_inference_steps = 50

        faceid_config: Optional[Dict[str, Any]] = None
        if config.get("faceid_enabled", False):
            faceid_plus_v2 = config.get("faceid_plus_v2", False)
            faceid_config = {
                "faceid_enabled": True,
                "faceid_model": config.get("faceid_model", "h94/IP-Adapter-FaceID"),
                "faceid_scale": config.get("faceid_scale", 0.6),
                "faceid_plus_v2": faceid_plus_v2,
                "faceid_skip_frames": config.get("faceid_skip_frames", 10),
            }
            variant = "PlusV2" if faceid_plus_v2 else "Base"
            logging.info(
                f"[FaceID] Will load IP-Adapter {variant} with scale={faceid_config['faceid_scale']}"
            )

        acc_value = runtime.get("acceleration_str", "none")

        self.wrapper = WrapperClass(
            model_id_or_path=model_name,
            t_index_list=runtime.get("t_index_list"),
            lora_dict=runtime.get("lora_dict"),
            mode="img2img" if runtime.get("mode_is_img2img", True) else "txt2img",
            frame_buffer_size=1,
            width=runtime["width"],
            height=runtime["height"],
            warmup=10,
            acceleration=acc_value,
            device_ids=None,
            use_lcm_lora=True,
            use_tiny_vae=config.get("use_tiny_vae", True),
            enable_similar_image_filter=config.get("similar_image_filter_enabled", True),
            similar_image_filter_threshold=config.get("similar_image_filter_threshold", 0.95),
            similar_image_filter_max_skip_frame=config.get("similar_image_filter_max_skip", 10),
            use_denoising_batch=True,
            cfg_type=runtime["cfg_type"],
            seed=runtime["seed"],
            dtype=runtime["torch_dtype"],
            device=runtime["device"],
            output_type="pt",
            cache_dir=runtime.get("cache_dir"),
            torch_compile_enabled=config.get("torch_compile_enabled", True),
            torch_compile_mode="reduce-overhead",
            torch_compile_fullgraph=not config.get("streamv2v_enabled", False),
            faceid_config=faceid_config,
            streamv2v_enabled=config.get("streamv2v_enabled", False),
            streamv2v_cache_maxframes=config.get("streamv2v_cache_maxframes", 1),
            warning_callback=runtime.get("warning_callback"),
        )

        # Correct is_sd2 from the ACTUAL architecture: name-based detection false-positives on
        # version numbers (e.g. "kohaku-v2.1" is SD 1.5, not SD 2.1) -> wrong ControlNet -> crash.
        _name_sd2 = self.is_sd2
        self.is_sd2 = self._resolve_is_sd2(_name_sd2)
        if self.is_sd2 != _name_sd2:
            logging.info(f"[Pipeline] is_sd2 corrected by architecture: name={_name_sd2} -> arch={self.is_sd2}")
        self._loaded = True
        logging.info("[Engine] StreamDiffusion engine active")

    def _resolve_is_sd2(self, name_based: bool) -> bool:
        """Detect SD 2.x by architecture (text-encoder hidden size / unet cross_attention_dim == 1024),
        not the model name which false-positives on version numbers like 'kohaku-v2.1' (an SD 1.5 model)."""
        stream = getattr(self.wrapper, "stream", None)
        probes = (
            lambda: stream.text_encoder.config.hidden_size == 1024,
            lambda: stream.unet.config.cross_attention_dim == 1024,
            lambda: stream.pipe.text_encoder.config.hidden_size == 1024,
        )
        for probe in probes:
            try:
                return bool(probe())
            except Exception:
                continue
        return name_based

    def run(
        self,
        image: Optional[torch.Tensor] = None,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.wrapper(image=image, prompt=prompt, **kwargs)

    def update_prompt(self, prompt: str) -> None:
        if hasattr(self.wrapper, "stream") and hasattr(self.wrapper.stream, "update_prompt"):
            self.wrapper.stream.update_prompt(prompt)
        else:
            logging.warning("[StreamDiffusionEngine] wrapper has no update_prompt, re-preparing")
            self.wrapper.prepare(prompt)

    def update_params(self, config: Dict[str, Any]) -> None:
        # Hot-reload handled upstream via wrapper recreation when needed.
        pass

    def cleanup(self) -> None:
        if self.wrapper is not None:
            if hasattr(self.wrapper, "cleanup"):
                try:
                    self.wrapper.cleanup()
                except Exception as e:
                    logging.warning(f"[StreamDiffusionEngine] wrapper.cleanup() failed: {e}")
            del self.wrapper
            self.wrapper = None
        self._loaded = False
        gc.collect()
        torch.cuda.empty_cache()
