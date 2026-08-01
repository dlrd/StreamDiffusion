"""Preprocessor orchestrator: parallel CUDA streams, unified frame skipping."""
import logging
from typing import Dict, List, Optional, Tuple

import torch

from .base import BasePreprocessor
from ipc import ConfigPacket


class PreprocessorOrchestrator:
    """Coordinates multiple preprocessors with parallel CUDA streams.

    Each registered processor runs on its own CUDA stream. Frame-skip
    counters are unified across all processors so they stay in sync.
    All streams are synchronized at the end of ``preprocess`` even when
    only one processor is active.
    """

    def __init__(self, device: torch.device, torch_dtype: torch.dtype):
        self.device = device
        self.torch_dtype = torch_dtype

        self._processors: Dict[str, BasePreprocessor] = {}
        self._streams: Dict[str, torch.cuda.Stream] = {}
        self._cached_results: Dict[str, torch.Tensor] = {}

        self._frame_counter: int = 0

        # Hot-path cache: list of names enabled at last preprocess() call.
        # Rebuilt only when the (enabled, is_loaded) signature changes.
        self._active_names_cache: List[str] = []
        self._active_signature: Optional[tuple] = None

    def register(self, name: str, processor: BasePreprocessor,
                 cuda_stream: Optional[torch.cuda.Stream] = None) -> None:
        """Register a processor with optional dedicated CUDA stream."""
        self._processors[name] = processor
        if cuda_stream is None and torch.cuda.is_available():
            cuda_stream = torch.cuda.Stream()
        self._streams[name] = cuda_stream
        logging.info(f"[Orchestrator] Registered processor: {name}")

    def unregister(self, name: str) -> None:
        """Unregister a processor and free its resources."""
        if name in self._processors:
            self._processors[name].cleanup()
            del self._processors[name]
        if name in self._streams:
            del self._streams[name]
        if name in self._cached_results:
            del self._cached_results[name]
        logging.info(f"[Orchestrator] Unregistered processor: {name}")

    def update_models(self, config: ConfigPacket) -> None:
        """Load/unload processor models based on config."""
        if hasattr(config, 'canny'):
            processor_configs = {
                'canny': (config.canny.enabled, config.canny),
                'depth': (config.depth.enabled, config.depth),
                'openpose': (config.openpose.enabled, config.openpose),
            }
        else:
            processor_configs = {
                'canny': (config.get('canny_enabled', False), config),
                'depth': (config.get('depth_enabled', False), config),
                'openpose': (config.get('openpose_enabled', False), config),
            }

        for name, (enabled, sub_config) in processor_configs.items():
            processor = self._processors.get(name)
            if name not in self._processors:
                continue

            processor = self._processors[name]
            if enabled:
                # Always call load_model — processors handle the "already
                # loaded with this config" case internally.
                processor.load_model(sub_config)
            elif processor.is_loaded:
                processor.unload_model()
                if name in self._cached_results:
                    self._cached_results.pop(name, None)

    def preprocess(self, image_tensor: torch.Tensor, config,
                   skip_frames: int = 1) -> Optional[Dict[str, torch.Tensor]]:
        """Run all enabled preprocessors with parallel CUDA streams.

        Returns dict {name: tensor} or None if nothing enabled.
        """
        if hasattr(config, 'canny'):
            sig = (
                bool(config.canny.enabled),
                bool(config.depth.enabled),
                bool(config.openpose.enabled),
                self._processors.get('canny') is not None and self._processors['canny'].is_loaded,
                self._processors.get('depth') is not None and self._processors['depth'].is_loaded,
                self._processors.get('openpose') is not None and self._processors['openpose'].is_loaded,
            )
            sub_configs = {
                'canny': config.canny,
                'depth': config.depth,
                'openpose': config.openpose,
            }
        else:
            c_en = bool(config.get('canny_enabled', False))
            d_en = bool(config.get('depth_enabled', False))
            o_en = bool(config.get('openpose_enabled', False))
            sig = (
                c_en, d_en, o_en,
                self._processors.get('canny') is not None and self._processors['canny'].is_loaded,
                self._processors.get('depth') is not None and self._processors['depth'].is_loaded,
                self._processors.get('openpose') is not None and self._processors['openpose'].is_loaded,
            )
            sub_configs = {'canny': config, 'depth': config, 'openpose': config}

        if sig != self._active_signature:
            self._active_names_cache = []
            # GPU-resident processors first: their async work then overlaps
            # canny's blocking D2H copy + CPU chain instead of queuing after it.
            enabled_map = {
                'depth': config.depth.enabled,
                'openpose': config.openpose.enabled,
                'canny': config.canny.enabled,
            }
            self._active_names_cache = [
                name for name, enabled in enabled_map.items()
                if enabled and name in self._processors and self._processors[name].is_loaded
            ]
            self._active_signature = sig
            self._cached_results.clear()

        active_names = self._active_names_cache

        if not active_names:
            return None

        self._frame_counter += 1
        use_cache = skip_frames > 1 and self._frame_counter % skip_frames != 0

        if use_cache:
            cached = {}
            for name in active_names:
                if name in self._cached_results:
                    cached[name] = self._cached_results[name]
            return cached if cached else None

        results: Dict[str, torch.Tensor] = {}

        for name in active_names:
            processor = self._processors[name]
            stream = self._streams.get(name)
            sub_config = sub_configs[name]

            if stream is not None:
                with torch.cuda.stream(stream):
                    result = processor.process(image_tensor, sub_config)
                    if result is not None:
                        results[name] = result
                        self._cached_results[name] = result
            else:
                result = processor.process(image_tensor, sub_config)
                if result is not None:
                    results[name] = result
                    self._cached_results[name] = result

        # Sync ALL streams, even when only 1 active processor: otherwise the
        # UNet may read a partially-written tensor from a non-default stream.
        current_stream = torch.cuda.current_stream()
        for name in active_names:
            stream = self._streams.get(name)
            if stream is not None:
                current_stream.wait_stream(stream)

        return results if results else None

    def get_active_info(self, config, controlnet_models: dict) -> Tuple[
        Optional[List], Optional[List[torch.Tensor]], Optional[List[float]]
    ]:
        """Return (models, images, scales) for active CNs, or (None, None, None)."""
        if hasattr(config, 'canny'):
            cn_info = [
                ('canny', config.canny.enabled, config.canny.scale),
                ('depth', config.depth.enabled, config.depth.scale),
                ('openpose', config.openpose.enabled, config.openpose.scale),
            ]
        else:
            cn_info = [
                ('canny', config.get('canny_enabled', False), config.get('canny_scale', 1.0)),
                ('depth', config.get('depth_enabled', False), config.get('depth_scale', 0.5)),
                ('openpose', config.get('openpose_enabled', False), config.get('openpose_scale', 0.8)),
            ]

        models = []
        images = []
        scales = []

        for name, enabled, scale in cn_info:
            if enabled and name in controlnet_models and name in self._cached_results:
                models.append(controlnet_models[name])
                images.append(self._cached_results[name])
                scales.append(scale)

        if not models:
            return None, None, None

        return models, images, scales

    def cleanup(self) -> None:
        """Release all processors and resources."""
        for name, processor in list(self._processors.items()):
            try:
                processor.cleanup()
            except Exception as e:
                logging.warning(f"[Orchestrator] Error cleaning up {name}: {e}")

        self._processors.clear()
        self._streams.clear()
        self._cached_results.clear()
        self._frame_counter = 0
        logging.info("[Orchestrator] All processors cleaned up")
