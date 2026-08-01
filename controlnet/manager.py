"""ControlNet model management.

``ControlNetManager`` owns ControlNet weights on GPU plus pre-cached hot-path
lists (``active_keys`` / ``models_cache`` / ``scales_cache``) consumed by the
pipeline. Lists are rebuilt only on config change via ``update_active_list``.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

import torch
from diffusers import ControlNetModel

from ipc import Acceleration

# xinsir/controlnet-union-sdxl-1.0 control-type indices. Order is fixed by
# the model's training schedule; do not reorder. The control_add_embedding
# linear layer is sized to num_types * addition_time_embed_dim (6 * 256).
UNION_TYPE_OPENPOSE = 0
UNION_TYPE_DEPTH    = 1
UNION_TYPE_HED      = 2
UNION_TYPE_CANNY    = 3
UNION_TYPE_NORMAL   = 4
UNION_TYPE_SEGMENT  = 5
UNION_NUM_TYPES     = 6

PREPROCESSOR_TO_UNION_TYPE = {
    'canny':    UNION_TYPE_CANNY,
    'depth':    UNION_TYPE_DEPTH,
    'openpose': UNION_TYPE_OPENPOSE,
}

if TYPE_CHECKING:
    from SmodeStreamDiffusion import App


PACKAGE_DIR = Path(__file__).resolve().parent.parent


class UnionControlNetWrapper:
    """Polymorphic with ``diffusers.ControlNetModel`` — wraps an SDXL
    ``ControlNetUnionModel`` plus the active preprocessor name list.

    SDXL Union takes all active conditioning images in a single forward pass.
    The pipeline iterates over a list of ControlNet models; we expose one
    entry whose ``__call__`` unpacks a list of images and dispatches them
    to the Union model along with the control_type one-hot tensor +
    control_type_idx mapping.
    """

    def __init__(self, union_model, active_names: List[str], app: "App"):
        self._model = union_model
        self._app = app
        self.set_active(active_names)
        self._control_type_tensor = None
        self._control_type_tensor_key = None

    def set_active(self, active_names: List[str]) -> None:
        """Update which preprocessors are active."""
        self.active_names = list(active_names)
        self.control_type_idx = [
            PREPROCESSOR_TO_UNION_TYPE[name] for name in self.active_names
            if name in PREPROCESSOR_TO_UNION_TYPE
        ]
        unknown = [n for n in active_names if n not in PREPROCESSOR_TO_UNION_TYPE]
        if unknown:
            logging.warning(
                "[Union CN] Skipping unknown preprocessor types: %s (known: %s)",
                unknown, list(PREPROCESSOR_TO_UNION_TYPE.keys()),
            )
        self._control_type_tensor = None
        self._control_type_tensor_key = None

    def _ensure_control_type_tensor(self, sample: torch.Tensor) -> torch.Tensor:
        key = (sample.device, sample.dtype, tuple(self.control_type_idx))
        if self._control_type_tensor_key == key:
            return self._control_type_tensor
        ct = torch.zeros(1, UNION_NUM_TYPES, dtype=sample.dtype, device=sample.device)
        for idx in self.control_type_idx:
            ct[0, idx] = 1
        self._control_type_tensor = ct
        self._control_type_tensor_key = key
        return ct

    @property
    def config(self):
        return self._model.config

    def to(self, *args, **kwargs):
        self._model.to(*args, **kwargs)
        return self

    def eval(self):
        self._model.eval()
        return self

    def __call__(
        self,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond,
        conditioning_scale=1.0,
        added_cond_kwargs=None,
        return_dict: bool = True,
        **kwargs,
    ):
        if not isinstance(controlnet_cond, (list, tuple)):
            controlnet_cond = [controlnet_cond]
        if len(controlnet_cond) != len(self.control_type_idx):
            raise ValueError(
                f"UnionControlNetWrapper: got {len(controlnet_cond)} conditioning "
                f"images but configured for {len(self.control_type_idx)} active "
                f"control types {self.active_names}."
            )

        # ControlNetUnionModel calls torch.mean(condition, dim=(2, 3)) which
        # needs 4D. Preprocessors emit 3D (C, H, W) — expand to batch dim.
        batch_size = sample.shape[0]
        controlnet_cond = [
            c.unsqueeze(0).expand(batch_size, -1, -1, -1) if c.dim() == 3 else c
            for c in controlnet_cond
        ]
        # Per-slot ring feeds 4D conds already at batch_size; if a 4D cond
        # arrives at batch=1, expand to batch_size for shape consistency.
        controlnet_cond = [
            c.expand(batch_size, -1, -1, -1) if (c.dim() == 4 and c.shape[0] == 1) else c
            for c in controlnet_cond
        ]

        ct = self._ensure_control_type_tensor(sample)
        # Union forward reshapes control_embeds by sample batch, so ct must
        # match sample.shape[0] — required for multi-step (Hyper) batches.
        if ct.shape[0] != sample.shape[0]:
            ct = ct.expand(sample.shape[0], -1)

        # Keep conditioning_scale as a 1-D TENSOR (not Python float/list) so
        # torch.compile doesn't specialize on the scale value — a float
        # triggers full recompile on every tweak; a tensor is treated as
        # a dynamic input.
        n = len(controlnet_cond)
        if isinstance(conditioning_scale, torch.Tensor):
            conditioning_scale = conditioning_scale.reshape(-1).to(
                device=sample.device, dtype=sample.dtype
            )
            if conditioning_scale.numel() == 1 and n > 1:
                conditioning_scale = conditioning_scale.expand(n)
        else:
            if not isinstance(conditioning_scale, (list, tuple)):
                conditioning_scale = [conditioning_scale] * n
            conditioning_scale = torch.tensor(
                list(conditioning_scale), device=sample.device, dtype=sample.dtype
            )

        result = self._model(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            control_type=ct,
            control_type_idx=self.control_type_idx,
            conditioning_scale=conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=return_dict,
        )
        return result


def round_to_vit_patch(resolution: int) -> int:
    """Round to nearest multiple of 14 (ViT patch size), clamped to [252, 518].

    Depth-Anything V2 uses a ViT backbone with patch_size=14; H/W must be
    divisible by 14. 252 = 14*18 minimum usable size, 518 = 14*37 native
    training resolution.
    """
    rounded = (resolution + 7) // 14 * 14
    return max(252, min(518, rounded))


class ControlNetManager:
    """Owner of ControlNet models + their pre-cached hot-path lists."""

    def __init__(self, app: "App"):
        self._app = app
        self.models: Dict[str, ControlNetModel] = {}
        self.active_keys: List[str] = []
        self.models_cache: List[ControlNetModel] = []
        self.scales_cache: List[float] = []
        # SDXL Union ProMax — lazy-loaded, shared across canny/depth/openpose.
        self._union_model = None
        self._union_wrapper: "UnionControlNetWrapper" = None

    def _emit_warning(self, active: bool, message: str = "") -> None:
        app = self._app
        send_warning = getattr(app, "_send_warning", None) if app is not None else None
        if send_warning is None:
            return
        try:
            send_warning(active, message)
        except Exception as e:
            logging.warning(f"[ControlNet] warning_callback failed: {e}")

    # ---- Hot-path cache rebuild -----------------------------------------

    def is_union_mode(self) -> bool:
        """True when SDXL Union ControlNet is in use."""
        return getattr(self._app, "is_sdxl", False) and self._union_wrapper is not None

    def update_active_list(self) -> None:
        """Pre-build stable lists of active ControlNets.

        Called on config change, not every frame. In SDXL Union mode,
        ``models_cache`` has at most one entry (the wrapper); ``active_keys``
        still tracks individual preprocessor names. Consumers detect Union
        mode via ``len(models_cache) < len(active_keys)``.
        """
        app = self._app
        self.active_keys = []
        if app._cached_canny_enabled and 'canny' in self.models:
            self.active_keys.append('canny')
        if app._cached_depth_enabled and 'depth' in self.models:
            self.active_keys.append('depth')
        if app._cached_openpose_enabled and 'openpose' in self.models:
            self.active_keys.append('openpose')

        per_name_scales = [
            app._cached_canny_scale if k == 'canny' else
            app._cached_depth_scale if k == 'depth' else
            app._cached_openpose_scale
            for k in self.active_keys
        ]

        if not self.active_keys:
            self.models_cache = []
            self.scales_cache = []
            return

        if getattr(app, "is_sdxl", False) and self._union_wrapper is not None:
            self._union_wrapper.set_active(self.active_keys)
            self.models_cache = [self._union_wrapper]
            # Pre-build the per-name scales as a 1-D tensor ONCE here instead
            # of letting pipeline_xl rebuild it (torch.tensor(...) = fresh
            # alloc + H2D copy) every frame. update_active_list re-runs on
            # every config poll, so live slider changes still propagate within
            # ~2 s. The pipeline's isinstance(cn_scale, torch.Tensor) branch
            # then reuses this with zero per-frame allocation. (Union already
            # received a tensor before — the pipeline converted the list each
            # frame — so the wrapper sees no change.)
            self.scales_cache = [
                torch.tensor(per_name_scales, device=app.device, dtype=app.torch_dtype)
            ]
        else:
            self.models_cache = [self.models[k] for k in self.active_keys]
            self.scales_cache = per_name_scales

    # ---- compile dispatch (torch.compile or TensorRT) ------------------

    def compile(self, model, controlnet_name: str):
        """Accelerate a ControlNet (or depth) model per ``app.acceleration``.

        TENSORRT → build/load engine via ``ControlNetEngine`` or
        ``DepthAnythingEngine``. TORCHCOMPILE → torch.compile (reduce-overhead).
        Other accelerations → unchanged. Returns the (possibly wrapped) model.
        """
        app = self._app
        if not app.controlnet_config.get('torch_compile_enabled', True):
            return model

        if app.acceleration == Acceleration.TENSORRT:
            if controlnet_name == 'depth':
                return self._compile_depth_anything_tensorrt(model)
            if getattr(getattr(model, "config", None),
                       "cross_attention_dim", None) is None:
                return model
            return self._compile_tensorrt(model, controlnet_name)

        if app.acceleration != Acceleration.TORCHCOMPILE:
            return model

        try:
            if not (hasattr(torch, 'compile') and torch.__version__ >= '2.0'):
                return model

            controlnet_cache_dir = PACKAGE_DIR / "torch_compile_cache" / "controlnet" / controlnet_name
            controlnet_cache_dir.mkdir(parents=True, exist_ok=True)

            old_cache_dir = os.environ.get('TORCHINDUCTOR_CACHE_DIR', '')
            os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(controlnet_cache_dir)
            os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'

            logging.info(f"Compiling {controlnet_name} ControlNet with torch.compile()...")
            logging.info(f"  Cache directory: {controlnet_cache_dir}")
            self._emit_warning(
                True,
                f"Compiling {controlnet_name} ControlNet with torch.compile() "
                f"- first run can take 1-2 minutes",
            )

            try:
                compiled_model = torch.compile(
                    model,
                    mode='reduce-overhead',
                    fullgraph=False,
                    dynamic=False,
                )

                logging.info("  Triggering compilation with dummy inference...")

                batch_size = 1
                hidden_dim = getattr(model.config, 'cross_attention_dim', None)
                if hidden_dim is None:
                    # Vision model (Depth-Anything V2): reduce-overhead CUDA
                    # graphs cause hangs/BSOD here. Skip torch.compile.
                    logging.info("  Vision model detected, skipping torch.compile (not a ControlNet)")
                    os.environ['TORCHINDUCTOR_CACHE_DIR'] = old_cache_dir
                    return model

                dummy_sample = torch.randn(
                    batch_size, 4, 64, 64,
                    dtype=app.torch_dtype, device=app.device,
                )
                dummy_timestep = torch.tensor([1], device=app.device)
                dummy_encoder_hidden_states = torch.randn(
                    batch_size, 77, hidden_dim,
                    dtype=app.torch_dtype, device=app.device,
                )
                dummy_controlnet_cond = torch.randn(
                    batch_size, 3, 512, 512,
                    dtype=app.torch_dtype, device=app.device,
                )

                with torch.no_grad():
                    compiled_model(
                        dummy_sample,
                        dummy_timestep,
                        dummy_encoder_hidden_states,
                        dummy_controlnet_cond,
                        conditioning_scale=1.0,
                        return_dict=False,
                    )
                logging.info("  Compilation triggered successfully")

            finally:
                if old_cache_dir:
                    os.environ['TORCHINDUCTOR_CACHE_DIR'] = old_cache_dir
                else:
                    os.environ.pop('TORCHINDUCTOR_CACHE_DIR', None)

            logging.info(f"{controlnet_name} ControlNet compiled and cached successfully")
            return compiled_model

        except Exception as e:
            logging.warning(f"Failed to compile {controlnet_name} ControlNet (non-critical): {e}")
            return model

    # ---- TensorRT compile path ------------------------------------------

    def _trt_engine_path(self, controlnet_name: str):
        """Absolute path to the TRT engine file for this ControlNet.

        Returns None when TRT is not active or the path cannot be derived
        (no UNet TRT engine, SDXL, etc.). Path may not exist.

        Layout: tensorrt_cache/sd/controlnet/<name>/<model_id>--bs-N/controlnet.engine
        """
        app = self._app
        if app.acceleration != Acceleration.TENSORRT:
            return None
        if getattr(app, "is_sdxl", False):
            return None  # SDXL UNet engine has no ControlNet ports
        stream_obj = getattr(app.stream, "stream", None)
        if (stream_obj is None
                or not hasattr(stream_obj, "unet")
                or not hasattr(stream_obj.unet, "stream")):
            return None
        batch = stream_obj.trt_unet_batch_size
        model_id = (app.model_name or "unknown").replace("/", "_").replace("\\", "_")
        # Resolution must be part of the cache key: static-shape TRT engines
        # (build_dynamic_shape=False) bake a min=opt=max profile, so a 512-baked
        # engine cannot serve a 1024 session — without --res it would cache-hit and
        # fail per-frame on the shape mismatch. Match the UNet prefix order (h x w).
        res_h = getattr(app, "height", 512)
        res_w = getattr(app, "width", 512)
        engine_dir = (
            PACKAGE_DIR / "tensorrt_cache" / "sd" / "controlnet"
            / controlnet_name / f"{model_id}--bs-{batch}--res-{res_h}x{res_w}"
        )
        return str(engine_dir / "controlnet.engine")

    def _try_load_cached_trt(self, controlnet_name: str):
        """Load a cached TRT engine for this ControlNet if it exists.

        Used at startup to skip the ~700 MB transient PyTorch ControlNetModel
        load when an engine is already on disk. Returns the engine or None.
        """
        engine_path = self._trt_engine_path(controlnet_name)
        if engine_path is None:
            app = self._app
            stream_obj = getattr(app.stream, "stream", None)
            logging.info(
                f"[ControlNet TRT] No engine path for {controlnet_name} "
                f"(accel={app.acceleration}, is_sdxl={getattr(app, 'is_sdxl', False)}, "
                f"stream={stream_obj is not None}, "
                f"unet_has_stream={hasattr(getattr(stream_obj, 'unet', None), 'stream') if stream_obj else False})"
            )
            return None
        if not os.path.exists(engine_path):
            logging.info(
                f"[ControlNet TRT] No cached engine on disk for {controlnet_name} "
                f"(expected at: {engine_path})"
            )
            return None
        try:
            from pipeline.acceleration.tensorrt.engine import ControlNetEngine
        except Exception as e:
            logging.warning(
                f"[ControlNet TRT] TensorRT module unavailable ({e}); "
                f"will fall back to PyTorch load for {controlnet_name}."
            )
            return None
        try:
            stream_obj = self._app.stream.stream
            cuda_stream = stream_obj.unet.stream
            size_mb = os.path.getsize(engine_path) // (1024 * 1024)
            logging.info(
                f"Loading cached {controlnet_name} TRT engine ({size_mb} MB) "
                f"— skipping PyTorch ControlNetModel load."
            )
            cn_engine = ControlNetEngine(
                engine_path, cuda_stream, use_cuda_graph=True
            )
            logging.info(f"{controlnet_name} ControlNet TRT engine active (cache hit)")
            return cn_engine
        except Exception as e:
            logging.warning(
                f"[ControlNet TRT] Cached engine load failed for {controlnet_name} "
                f"({e}); will fall back to PyTorch + rebuild."
            )
            return None

    def _compile_tensorrt(self, model, controlnet_name: str):
        """Build a TRT engine for this ControlNet from a PyTorch model.

        Shares the cuda.Stream with the U-Net TRT engine. Frees the PyTorch
        model after engine load (~700 MB recovered). SDXL is skipped (UNet
        engine lacks CN ports). On failure returns the PyTorch model.
        """
        app = self._app

        if getattr(app, "is_sdxl", False):
            logging.warning(
                f"[ControlNet TRT] SDXL not supported (UNet engine lacks CN ports). "
                f"Falling back to PyTorch for {controlnet_name} ControlNet."
            )
            return model

        stream_obj = getattr(app.stream, "stream", None)
        if (stream_obj is None
                or not hasattr(stream_obj, "unet")
                or not hasattr(stream_obj.unet, "stream")):
            logging.warning(
                f"[ControlNet TRT] UNet TRT engine not ready, "
                f"falling back to PyTorch for {controlnet_name} ControlNet."
            )
            return model

        try:
            from pipeline.acceleration.tensorrt import compile_controlnet
            from pipeline.acceleration.tensorrt.engine import ControlNetEngine
            from pipeline.acceleration.tensorrt.models import ControlNet as ControlNetONNX
        except Exception as e:
            logging.warning(
                f"[ControlNet TRT] TensorRT module unavailable ({e}), "
                f"falling back to PyTorch for {controlnet_name}."
            )
            return model

        self._emit_warning(
            True,
            f"Building TensorRT engine for {controlnet_name} ControlNet "
            f"- one-time, ~1-3 min",
        )
        try:
            engine_path = self._trt_engine_path(controlnet_name)
            os.makedirs(os.path.dirname(engine_path), exist_ok=True)
            batch = stream_obj.trt_unet_batch_size

            logging.info(
                f"Building TensorRT engine for {controlnet_name} ControlNet "
                f"(one-time, ~1-3 min)..."
            )
            logging.info(f"  Engine path: {engine_path}")
            embedding_dim = stream_obj.text_encoder.config.hidden_size
            unet_in_channels = 4
            if hasattr(stream_obj.unet, "config"):
                unet_in_channels = getattr(
                    stream_obj.unet.config, "in_channels", 4
                )
            onnx_model = ControlNetONNX(
                fp16=True, device=app.device,
                max_batch_size=batch, min_batch_size=batch,
                embedding_dim=embedding_dim,
                unet_dim=unet_in_channels,
            )
            compile_controlnet(
                model, onnx_model,
                engine_path + ".onnx",
                engine_path + ".opt.onnx",
                engine_path,
                opt_batch_size=batch,
                opt_image_height=getattr(app, "height", 512),
                opt_image_width=getattr(app, "width", 512),
            )
            logging.info(f"{controlnet_name} ControlNet TRT engine built")

            cuda_stream = stream_obj.unet.stream
            cn_engine = ControlNetEngine(
                engine_path, cuda_stream, use_cuda_graph=True
            )

            try:
                model.to("cpu")
            except Exception:
                pass
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            logging.info(f"{controlnet_name} ControlNet TRT engine active")
            return cn_engine

        except Exception as e:
            logging.warning(
                f"[ControlNet TRT] Build/load failed for {controlnet_name} ({e}), "
                f"falling back to PyTorch ControlNet."
            )
            import traceback
            logging.debug(traceback.format_exc())
            return model

    # ---- Depth-Anything V2 TensorRT path --------------------------------

    def try_load_depth_anything_cache(self, model_size: str, target_resolution: int):
        """Pre-load probe for the Depth-Anything TRT engine cache.

        Called by ``DepthProcessor.load_model`` before ``from_pretrained``
        so we can skip the ~1.5 GB transient HF model load on cache hit.
        Returns the engine on success, None otherwise.
        """
        image_size = round_to_vit_patch(int(target_resolution))
        engine_path = self._depth_anything_trt_engine_path(model_size, image_size)
        if engine_path is None or not os.path.exists(engine_path):
            return None
        try:
            from pipeline.acceleration.tensorrt.engine import DepthAnythingEngine
        except Exception as e:
            logging.warning(
                f"[Depth-Anything TRT] TensorRT module unavailable ({e}); "
                f"will fall back to PyTorch."
            )
            return None
        try:
            stream_obj = self._app.stream.stream
            cuda_stream = stream_obj.unet.stream
            size_mb = os.path.getsize(engine_path) // (1024 * 1024)
            logging.info(
                f"Loading cached Depth-Anything {model_size.upper()} TRT engine "
                f"@ {image_size}x{image_size} ({size_mb} MB) — skipping PyTorch load."
            )
            engine = DepthAnythingEngine(
                engine_path, cuda_stream,
                image_size=image_size, use_cuda_graph=True,
            )
            logging.info(
                f"Depth-Anything {model_size.upper()} TRT engine active (cache hit)"
            )
            return engine
        except Exception as e:
            logging.warning(
                f"[Depth-Anything TRT] Cached engine load failed ({e}); "
                f"will fall back to PyTorch + rebuild."
            )
            return None

    def _depth_anything_image_size(self) -> int:
        raw = self._app.controlnet_config.get('depth_resolution', 378)
        return round_to_vit_patch(int(raw))

    def _depth_anything_model_size(self, model) -> str:
        """Detect Depth-Anything V2 variant (small/base/large) from HF config."""
        config = getattr(model, "config", None)
        backbone_config = getattr(config, "backbone_config", None) if config else None
        hidden_size = getattr(backbone_config, "hidden_size", None) if backbone_config else None
        if hidden_size is None:
            hidden_size = getattr(config, "hidden_size", None) if config else None
        return {384: "small", 768: "base", 1024: "large"}.get(
            hidden_size, f"hidden{hidden_size}"
        )

    def _depth_anything_trt_engine_path(self, model_size: str, image_size: int):
        app = self._app
        if app.acceleration != Acceleration.TENSORRT:
            return None
        stream_obj = getattr(app.stream, "stream", None)
        if (stream_obj is None
                or not hasattr(stream_obj, "unet")
                or not hasattr(stream_obj.unet, "stream")):
            return None
        engine_dir = (
            PACKAGE_DIR / "tensorrt_cache" / "sd" / "depth_anything"
            / f"{model_size}--res-{image_size}"
        )
        return str(engine_dir / "depth_anything.engine")

    def _try_load_cached_depth_anything_trt(self, model):
        model_size = self._depth_anything_model_size(model)
        image_size = self._depth_anything_image_size()
        engine_path = self._depth_anything_trt_engine_path(model_size, image_size)
        if engine_path is None:
            return None
        if not os.path.exists(engine_path):
            return None
        try:
            from pipeline.acceleration.tensorrt.engine import DepthAnythingEngine
        except Exception as e:
            logging.warning(
                f"[Depth-Anything TRT] TensorRT module unavailable ({e}); "
                f"will fall back to PyTorch."
            )
            return None
        try:
            stream_obj = self._app.stream.stream
            cuda_stream = stream_obj.unet.stream
            size_mb = os.path.getsize(engine_path) // (1024 * 1024)
            logging.info(
                f"Loading cached Depth-Anything {model_size.upper()} TRT engine "
                f"@ {image_size}x{image_size} ({size_mb} MB) — skipping PyTorch load."
            )
            engine = DepthAnythingEngine(
                engine_path, cuda_stream,
                image_size=image_size,
                use_cuda_graph=True,
            )
            logging.info(
                f"Depth-Anything {model_size.upper()} TRT engine active (cache hit)"
            )
            return engine
        except Exception as e:
            logging.warning(
                f"[Depth-Anything TRT] Cached engine load failed ({e}); "
                f"will fall back to PyTorch + rebuild."
            )
            return None

    def _compile_depth_anything_tensorrt(self, model):
        """Build a TRT engine for Depth-Anything V2 from the HF model."""
        app = self._app

        stream_obj = getattr(app.stream, "stream", None)
        if (stream_obj is None
                or not hasattr(stream_obj, "unet")
                or not hasattr(stream_obj.unet, "stream")):
            logging.warning(
                "[Depth-Anything TRT] UNet TRT engine not ready, "
                "falling back to PyTorch."
            )
            return model

        cached = self._try_load_cached_depth_anything_trt(model)
        if cached is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            return cached

        try:
            from pipeline.acceleration.tensorrt import compile_depth_anything
            from pipeline.acceleration.tensorrt.engine import DepthAnythingEngine
            from pipeline.acceleration.tensorrt.models import DepthAnything as DepthAnythingONNX
        except Exception as e:
            logging.warning(
                f"[Depth-Anything TRT] TensorRT module unavailable ({e}), "
                f"falling back to PyTorch."
            )
            return model

        model_size = self._depth_anything_model_size(model)
        image_size = self._depth_anything_image_size()
        self._emit_warning(
            True,
            f"Building TensorRT engine for Depth-Anything V2 {model_size.upper()} "
            f"@ {image_size}x{image_size} - one-time, ~3-5 min",
        )
        try:
            engine_path = self._depth_anything_trt_engine_path(model_size, image_size)
            os.makedirs(os.path.dirname(engine_path), exist_ok=True)

            logging.info(
                f"Building TensorRT engine for Depth-Anything V2 {model_size.upper()} "
                f"@ {image_size}x{image_size} (one-time, ~3-5 min)..."
            )
            logging.info(f"  Engine path: {engine_path}")
            onnx_model = DepthAnythingONNX(
                fp16=True, device=app.device, image_size=image_size,
            )
            compile_depth_anything(
                model, onnx_model,
                engine_path + ".onnx",
                engine_path + ".opt.onnx",
                engine_path,
                image_size=image_size,
            )
            logging.info(f"Depth-Anything {model_size.upper()} TRT engine built")

            cuda_stream = stream_obj.unet.stream
            engine = DepthAnythingEngine(
                engine_path, cuda_stream,
                image_size=image_size, use_cuda_graph=True,
            )

            try:
                model.to("cpu")
            except Exception:
                pass
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            logging.info(f"Depth-Anything {model_size.upper()} TRT engine active")
            return engine

        except Exception as e:
            logging.warning(
                f"[Depth-Anything TRT] Build/load failed ({e}), "
                f"falling back to PyTorch."
            )
            import traceback
            logging.debug(traceback.format_exc())
            return model

    # ---- U-Net+ControlNet warmup ----------------------------------------

    def warmup_integration(self, controlnet_name: str, controlnet_model) -> None:
        """Warmup U-Net+ControlNet integration to trigger torch.compile.

        Skipped when acceleration=TENSORRT (engines are pre-compiled).
        """
        app = self._app
        if not app.stream:
            return

        if app.acceleration == Acceleration.TENSORRT:
            logging.info(
                f"[ControlNet TRT] Skipping warmup for {controlnet_name} "
                f"(engine pre-compiled)"
            )
            return

        dummy_input = None
        self._emit_warning(
            True,
            f"Warming up U-Net+{controlnet_name} integration "
            f"- first run can take 1-2 minutes",
        )
        try:
            logging.info(f"Warming up U-Net+{controlnet_name} integration...")
            warmup_start = time.time()

            dummy_input = torch.randn(
                (1, 3, app.height, app.width),
                dtype=app.torch_dtype,
                device=app.device,
            )

            for _ in range(2):
                # New CUDA Graphs iteration, else iteration 2 overwrites iteration 1's
                # graph-owned outputs (corrupts cudagraph trees; fullgraph=False/V2V).
                torch.compiler.cudagraph_mark_step_begin()
                _ = app.stream(
                    image=dummy_input,
                    controlnet_image=[dummy_input],
                    controlnet_model=[controlnet_model],
                    controlnet_conditioning_scale=[1.0],
                )

            torch.cuda.synchronize()
            warmup_time = time.time() - warmup_start
            logging.info(f"{controlnet_name} integration warmup complete ({warmup_time:.1f}s)")

        except Exception as e:
            logging.warning(f"{controlnet_name} integration warmup failed (non-critical): {e}")
        finally:
            if dummy_input is not None:
                del dummy_input
                torch.cuda.empty_cache()

    # ---- SDXL Union ControlNet (xinsir ProMax) --------------------------

    def _load_union_for(self, preprocessor_name: str) -> None:
        """Ensure SDXL Union model is loaded and registered for this preprocessor."""
        app = self._app
        if self._union_model is None:
            self._emit_warning(
                True,
                "Loading SDXL Union ControlNet (xinsir/controlnet-union-sdxl-1.0, "
                "~2.5 GB) - first run can take a while (downloading + loading weights)...",
            )
            try:
                from diffusers import ControlNetUnionModel
                logging.info(
                    "[Union] Loading SDXL Union ControlNet "
                    "(xinsir/controlnet-union-sdxl-1.0, ~2.5 GB)..."
                )
                # xinsir uses underscore in promax filename + separate
                # config_promax.json, so diffusers can't auto-load variant="promax".
                # Standard variant covers the 8 control types we need.
                self._union_model = ControlNetUnionModel.from_pretrained(
                    "xinsir/controlnet-union-sdxl-1.0",
                    torch_dtype=app.torch_dtype,
                ).to(app.device)
                self._union_model.eval()

                if (app.controlnet_config.get('torch_compile_enabled', True)
                        and hasattr(torch, 'compile')):
                    # Save/restore env so other compile sites don't pollute
                    # the Union FX graph cache.
                    old_cache_dir = os.environ.get('TORCHINDUCTOR_CACHE_DIR', '')
                    old_fx_cache = os.environ.get('TORCHINDUCTOR_FX_GRAPH_CACHE', '')
                    try:
                        cache_dir = PACKAGE_DIR / "torch_compile_cache" / "controlnet" / "union_sdxl"
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache_dir)
                        os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'
                        self._union_model = torch.compile(
                            self._union_model, mode='default',
                            fullgraph=False, dynamic=False,
                        )
                        logging.info(
                            "[Union] torch.compile applied (mode=default). "
                            "First use of each control type compiles (~30-60s), "
                            "then cached on disk."
                        )
                    except Exception as e:
                        logging.warning(
                            f"[Union] torch.compile failed (non-critical, "
                            f"running eager): {e}"
                        )
                    finally:
                        if old_cache_dir:
                            os.environ['TORCHINDUCTOR_CACHE_DIR'] = old_cache_dir
                        else:
                            os.environ.pop('TORCHINDUCTOR_CACHE_DIR', None)
                        if old_fx_cache:
                            os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = old_fx_cache
                        else:
                            os.environ.pop('TORCHINDUCTOR_FX_GRAPH_CACHE', None)

                self._union_wrapper = UnionControlNetWrapper(
                    self._union_model, active_names=[], app=app,
                )
                logging.info(
                    f"[Union] SDXL Union loaded on {app.device} "
                    f"({app.torch_dtype}). Replaces 3 separate ControlNets — "
                    f"single forward pass per frame."
                )
            except Exception as e:
                logging.error(f"[Union] Failed to load SDXL Union ProMax: {e}")
                self._union_model = None
                self._union_wrapper = None
                app.controlnet_config[f'{preprocessor_name}_enabled'] = False
                return
            finally:
                self._emit_warning(False)

        # Multiple preprocessor names point to the same wrapper — intentional;
        # load_models() uses ``name in self.models`` for tracking.
        self.models[preprocessor_name] = self._union_wrapper
        logging.info(f"[Union] {preprocessor_name.capitalize()} control type enabled")

    # ---- Per-ControlNet load helpers ------------------------------------

    def _load_canny(self) -> None:
        app = self._app

        if app.is_sdxl:
            self._load_union_for('canny')
            return

        cached = self._try_load_cached_trt('canny')
        if cached is not None:
            self.models['canny'] = cached
            self.warmup_integration('canny', cached)
            return

        try:
            if app.is_sd2:
                canny_repo = "thibaud/controlnet-sd21"
                canny_filename = "control_v11p_sd21_canny.safetensors"
                canny_config = "thibaud/controlnet-sd21-canny-diffusers"
                cn_label = "SD 2.1"
            else:
                # SD 1.5 ControlNet++ (limingcv, ECCV 2024) FP16 weights by huchenlei.
                canny_repo = "huchenlei/ControlNet_plus_plus_collection_fp16"
                canny_filename = "controlnet++_canny_sd15_fp16.safetensors"
                canny_config = "lllyasviel/sd-controlnet-canny"
                cn_label = "SD 1.5 ControlNet++"
            logging.info(f"Loading Canny ControlNet model ({cn_label})...")
            self._emit_warning(True, f"Loading Canny ControlNet model ({cn_label})...")
            if canny_filename:
                from huggingface_hub import hf_hub_download
                canny_path = hf_hub_download(repo_id=canny_repo, filename=canny_filename)
                kwargs = {"torch_dtype": app.torch_dtype}
                if canny_config:
                    kwargs["config"] = canny_config
                model = ControlNetModel.from_single_file(canny_path, **kwargs).to(app.device)
            else:
                model = ControlNetModel.from_pretrained(
                    canny_repo, torch_dtype=app.torch_dtype
                ).to(app.device)

            model = self.compile(model, 'canny')
            self.models['canny'] = model
            logging.info("Canny ControlNet loaded successfully")

            self.warmup_integration('canny', model)
        except Exception as e:
            logging.error(f"Failed to load Canny ControlNet: {e}")
            if 'canny' in self.models:
                del self.models['canny']
            torch.cuda.empty_cache()
            app.controlnet_config['canny_enabled'] = False
        finally:
            self._emit_warning(False)

    def _load_depth(self) -> None:
        app = self._app

        if app.is_sdxl:
            self._load_union_for('depth')
            return

        # Cache name 'depth_controlnet' is distinct from the depth-anything
        # preprocessor cache (lives under 'depth').
        cached = self._try_load_cached_trt('depth_controlnet')
        if cached is not None:
            self.models['depth'] = cached
            self.warmup_integration('depth', cached)
            return

        try:
            if app.is_sd2:
                depth_repo = "thibaud/controlnet-sd21"
                depth_filename = "control_v11p_sd21_depth.safetensors"
                depth_config = "thibaud/controlnet-sd21-depth-diffusers"
                cn_label = "SD 2.1"
            else:
                depth_repo = "huchenlei/ControlNet_plus_plus_collection_fp16"
                depth_filename = "controlnet++_depth_sd15_fp16.safetensors"
                depth_config = "lllyasviel/sd-controlnet-depth"
                cn_label = "SD 1.5 ControlNet++"
            logging.info(f"Loading Depth ControlNet model ({cn_label})...")
            self._emit_warning(True, f"Loading Depth ControlNet model ({cn_label})...")
            if depth_filename:
                from huggingface_hub import hf_hub_download
                depth_path = hf_hub_download(repo_id=depth_repo, filename=depth_filename)
                kwargs = {"torch_dtype": app.torch_dtype}
                if depth_config:
                    kwargs["config"] = depth_config
                model = ControlNetModel.from_single_file(depth_path, **kwargs).to(app.device)
            else:
                model = ControlNetModel.from_pretrained(
                    depth_repo, torch_dtype=app.torch_dtype
                ).to(app.device)

            model = self.compile(model, 'depth_controlnet')
            self.models['depth'] = model
            logging.info("Depth ControlNet loaded successfully")

            self.warmup_integration('depth', model)
        except Exception as e:
            logging.error(f"Failed to load Depth ControlNet: {e}")
            if 'depth' in self.models:
                del self.models['depth']
            torch.cuda.empty_cache()
            app.controlnet_config['depth_enabled'] = False
        finally:
            self._emit_warning(False)

    def _load_openpose(self) -> None:
        app = self._app

        if app.is_sdxl:
            self._load_union_for('openpose')
            return

        cached = self._try_load_cached_trt('openpose')
        if cached is not None:
            self.models['openpose'] = cached
            self.warmup_integration('openpose', cached)
            return

        try:
            if app.is_sd2:
                # v2 weights: markedly better body fidelity than v1. Both are
                # body-only trained — hand/face keypoints read as extra limbs.
                openpose_repo = "thibaud/controlnet-sd21"
                openpose_filename = "control_v11p_sd21_openposev2.safetensors"
                openpose_config = "thibaud/controlnet-sd21-openpose-diffusers"
                cn_label = "SD 2.1 (openpose v2)"
            else:
                openpose_repo, openpose_filename, openpose_config = "lllyasviel/control_v11p_sd15_openpose", None, None
                cn_label = "SD 1.5"
            logging.info(f"Loading OpenPose ControlNet model ({cn_label})...")
            self._emit_warning(True, f"Loading OpenPose ControlNet model ({cn_label})...")
            if openpose_filename:
                from huggingface_hub import hf_hub_download
                openpose_path = hf_hub_download(repo_id=openpose_repo, filename=openpose_filename)
                kwargs = {"torch_dtype": app.torch_dtype}
                if openpose_config:
                    kwargs["config"] = openpose_config
                model = ControlNetModel.from_single_file(openpose_path, **kwargs).to(app.device)
            else:
                model = ControlNetModel.from_pretrained(
                    openpose_repo, torch_dtype=app.torch_dtype
                ).to(app.device)

            model = self.compile(model, 'openpose')
            self.models['openpose'] = model
            logging.info("OpenPose ControlNet loaded successfully")

            self.warmup_integration('openpose', model)
        except Exception as e:
            logging.error(f"Failed to load OpenPose ControlNet: {e}")
            if 'openpose' in self.models:
                del self.models['openpose']
            torch.cuda.empty_cache()
            app.controlnet_config['openpose_enabled'] = False
        finally:
            self._emit_warning(False)

    # ---- Load / unload models per config --------------------------------

    def load_models(self) -> None:
        """Load (or unload) ControlNet models based on current config.

        Idempotent. Failures disable the matching config flag to avoid
        retry loops.
        """
        app = self._app
        config = app.controlnet_config

        if config.get('canny_enabled', False) and 'canny' not in self.models:
            self._load_canny()

        if config.get('depth_enabled', False) and 'depth' not in self.models:
            self._load_depth()

        if config.get('openpose_enabled', False) and 'openpose' not in self.models:
            self._load_openpose()

        for name in ('canny', 'depth', 'openpose'):
            if not config.get(f'{name}_enabled', False) and name in self.models:
                logging.info(f"Unloading {name.capitalize()} ControlNet...")
                del self.models[name]
                torch.cuda.empty_cache()

        # SDXL Union: deleting per-name entries only drops refs to the shared
        # wrapper; the ~2.5 GB only frees when we drop _union_model itself.
        if (getattr(app, "is_sdxl", False)
                and self._union_model is not None
                and not any(name in self.models for name in ('canny', 'depth', 'openpose'))):
            logging.info("[Union] All controls disabled — unloading Union ProMax (~2.5 GB)")
            try:
                self._union_model.to("cpu")
            except Exception:
                pass
            self._union_model = None
            self._union_wrapper = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    # ---- Cleanup --------------------------------------------------------

    def cleanup(self) -> None:
        """Release all on-GPU ControlNet models. Safe to call multiple times."""
        try:
            for _model_name, model in list(self.models.items()):
                del model
            self.models.clear()
        except Exception as e:
            logging.warning(f"Error cleaning up ControlNet models: {e}")

        if self._union_model is not None:
            try:
                self._union_wrapper.to("cpu") if self._union_wrapper is not None else None
            except Exception:
                pass
            self._union_wrapper = None
            self._union_model = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        self._app = None
