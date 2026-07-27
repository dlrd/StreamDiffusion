"""Canny edge detection preprocessor for ControlNet."""
import logging
from typing import Optional

import cv2
import numpy as np
import torch

from ..base import BasePreprocessor


class CannyProcessor(BasePreprocessor):
    """Canny edge detection preprocessor.

    Processes at configurable resolution for speed, then upscales with
    NEAREST to preserve sharp binary edges. Uses OpenCV — no ML model.
    """

    def __init__(self, device: torch.device, torch_dtype: torch.dtype, max_buffer_size: int = 1024,
                 warning_callback=None):
        super().__init__(device, torch_dtype, max_buffer_size, warning_callback)
        self._input_buffer_max: Optional[np.ndarray] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._output_buffer_shape: Optional[tuple] = None
        self._in_ema: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return "canny"

    def load_model(self, config) -> None:
        """Canny uses OpenCV — no model to load."""
        if self._loaded:
            return
        self._emit_warning(True, "Preparing Canny preprocessor...")
        try:
            self._loaded = True
            logging.info("[CannyProcessor] Ready (OpenCV Canny, no model required)")
        finally:
            self._emit_warning(False)

    def unload_model(self) -> None:
        self._input_buffer_max = None
        self._output_buffer = None
        self._output_buffer_shape = None
        self._in_ema = None
        self._loaded = False
        logging.info("[CannyProcessor] Unloaded")

    def process(self, image_tensor: torch.Tensor, config) -> Optional[torch.Tensor]:
        """Run Canny edge detection. Input/output: CHW [0,1] on GPU."""
        if hasattr(config, 'low_threshold'):
            low_threshold = config.low_threshold
            high_threshold = config.high_threshold
            aperture_size = config.aperture_size
            l2_gradient = config.l2_gradient
            canny_resolution = config.resolution
        else:
            low_threshold = config.get('canny_low_threshold', 100)
            high_threshold = config.get('canny_high_threshold', 200)
            aperture_size = config.get('canny_aperture_size', 3)
            l2_gradient = config.get('canny_l2_gradient', False)
            canny_resolution = config.get('canny_resolution', 384)

        # cv2.Canny requires an odd aperture in {3, 5, 7}.
        aperture_size = min(7, max(3, int(aperture_size) | 1))

        original_h, original_w = image_tensor.shape[1], image_tensor.shape[2]

        if canny_resolution < original_h:
            downscaled = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),
                size=(canny_resolution, canny_resolution),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            process_h, process_w = canny_resolution, canny_resolution
        else:
            downscaled = image_tensor
            process_h, process_w = original_h, original_w

        # Temporal IIR denoise (motion-gated per pixel): sensor noise decorrelates
        # across frames while real detail persists -> SNR boost no threshold can give.
        if self._in_ema is None or self._in_ema.shape != downscaled.shape:
            self._in_ema = downscaled.detach().clone()
        else:
            w = (downscaled - self._in_ema).abs().mean(0, keepdim=True).mul_(8.0).clamp_(0.25, 1.0)
            self._in_ema.lerp_(downscaled, w)
        downscaled = self._in_ema

        if self._input_buffer_max is None:
            self._input_buffer_max = np.empty(
                (self.max_buffer_size, self.max_buffer_size, 3), dtype=np.uint8
            )

        input_buffer = self._input_buffer_max[:process_h, :process_w, :]

        # Convert GPU tensor to numpy for OpenCV. Scale+cast on the GPU and do
        # one uint8 D2H into the pre-allocated buffer (blocking: cv2.Canny
        # reads it right after), instead of copying floats down then doing a
        # full-res CPU multiply + astype per frame.
        gpu_u8 = (downscaled.permute(1, 2, 0) * 255).clamp_(0, 255).to(torch.uint8)
        torch.from_numpy(input_buffer).copy_(gpu_u8)
        del gpu_u8

        # Light spatial blur (temporal denoise above does the heavy lifting).
        cv2.GaussianBlur(input_buffer, (3, 3), 0, dst=input_buffer)

        edges = cv2.Canny(
            input_buffer, low_threshold, high_threshold,
            apertureSize=aperture_size, L2gradient=l2_gradient
        )

        # Drop small isolated fragments (sensor-noise specks), keep real contour chains.
        ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        if ncomp > 1:
            small = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < 24) + 1
            if small.size:
                edges[np.isin(labels, small)] = 0

        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges_temp = torch.from_numpy(edges_rgb).float() / 255.0
        edges_temp_permuted = edges_temp.permute(2, 0, 1).to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )

        # NEAREST upscale preserves sharp binary edges (critical for ControlNet).
        if canny_resolution < original_h:
            edges_upscaled = torch.nn.functional.interpolate(
                edges_temp_permuted.unsqueeze(0),
                size=(original_h, original_w),
                mode='nearest'
            ).squeeze(0)
        else:
            edges_upscaled = edges_temp_permuted

        out_shape = (3, original_h, original_w)
        if self._output_buffer is None or self._output_buffer_shape != out_shape:
            self._output_buffer = torch.empty(
                out_shape, device=self.device, dtype=self.torch_dtype
            )
            self._output_buffer_shape = out_shape

        self._output_buffer.copy_(edges_upscaled, non_blocking=True)

        del edges_temp, edges_temp_permuted, edges_upscaled

        self._cached_result = self._output_buffer
        return self._output_buffer
