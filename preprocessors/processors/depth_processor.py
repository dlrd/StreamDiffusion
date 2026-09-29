"""Depth estimation preprocessor using Depth-Anything V2."""
import logging
from typing import Optional

import cv2
import numpy as np
import torch

from ..base import BasePreprocessor
from utils.hub import local_first


class DepthProcessor(BasePreprocessor):
    """Depth estimation preprocessor.

    Method 'grayscale' uses Depth-Anything V2 (small/base/large).
    Method 'sobel' / 'laplacian' / fallback use CPU cv2. AI path is fully
    GPU-resident (downscale, normalize, infer, post-process).
    """

    MODEL_MAP = {
        'small': 'depth-anything/Depth-Anything-V2-Small-hf',
        'base': 'depth-anything/Depth-Anything-V2-Base-hf',
        'large': 'depth-anything/Depth-Anything-V2-Large-hf',
    }

    def __init__(self, device: torch.device, torch_dtype: torch.dtype, max_buffer_size: int = 1024,
                 warning_callback=None):
        super().__init__(device, torch_dtype, max_buffer_size, warning_callback)
        self._model = None
        self._processor = None
        self._current_model_size: Optional[str] = None
        self._depth_mean: Optional[torch.Tensor] = None
        self._depth_std: Optional[torch.Tensor] = None
        self._depth_cache: Optional[np.ndarray] = None
        self._output_buffer_max: Optional[torch.Tensor] = None
        self._temp_cpu_buffer: Optional[np.ndarray] = None
        self._gaussian_kernel: Optional[torch.Tensor] = None
        self._downscale_size: int = 256
        self._compile_fn = None
        self._try_cache_fn = None

    @property
    def name(self) -> str:
        return "depth"

    def set_compile_fn(self, fn):
        """Set the torch.compile / TRT-build wrapper."""
        self._compile_fn = fn

    def set_try_cache_fn(self, fn):
        """Set the TRT cache probe callback.

        Signature: ``fn(model_size: str, target_resolution: int) -> Optional[engine]``.
        On hit, skips the PyTorch HF load entirely (~1.5 GB transient VRAM).
        """
        self._try_cache_fn = fn

    def set_gaussian_kernel(self, kernel: Optional[torch.Tensor]):
        """Set pre-computed Gaussian kernel from ConfigManager."""
        self._gaussian_kernel = kernel

    def load_model(self, config) -> None:
        """Load Depth-Anything V2 model."""
        if hasattr(config, 'model_size'):
            model_size = config.model_size.lower()
            target_resolution = getattr(config, 'resolution', 378)
        else:
            model_size = config.get('depth_model_size', 'small').lower()
            target_resolution = config.get('depth_resolution', 378)

        # Reload triggers: model_size changed, or in TRT mode the ViT-rounded
        # resolution no longer matches the engine's fixed input shape.
        if self._processor is not None and self._current_model_size == model_size:
            engine_image_size = getattr(self._model, 'image_size', None)
            if engine_image_size is None:
                return  # PyTorch mode: no reload on resolution change.
            from controlnet.manager import round_to_vit_patch
            new_engine_image_size = round_to_vit_patch(int(target_resolution))
            if new_engine_image_size == engine_image_size:
                return
            logging.info(
                f"[DepthProcessor] depth_resolution changed "
                f"({engine_image_size} -> {new_engine_image_size}); "
                f"reloading TRT engine."
            )

        self._emit_warning(True, f"Loading Depth-Anything V2 {model_size.upper()} model - first run can take a while (downloading + loading weights)...")
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            if self._model is not None:
                logging.info(f"Unloading Depth-Anything V2 {self._current_model_size.upper()} model...")
                del self._model
                del self._processor
                torch.cuda.empty_cache()

            model_id = self.MODEL_MAP.get(model_size, self.MODEL_MAP['small'])

            # AutoImageProcessor is CPU-only — safe before the cache probe.
            self._processor = local_first(AutoImageProcessor.from_pretrained, model_id)

            self._depth_mean = torch.tensor(
                [0.485, 0.456, 0.406], device=self.device, dtype=self.torch_dtype
            ).view(3, 1, 1)
            self._depth_std = torch.tensor(
                [0.229, 0.224, 0.225], device=self.device, dtype=self.torch_dtype
            ).view(3, 1, 1)

            # Fast path: probe TRT cache before touching PyTorch HF model.
            if self._try_cache_fn is not None:
                cached_engine = self._try_cache_fn(model_size, int(target_resolution))
                if cached_engine is not None:
                    self._model = cached_engine
                    self._current_model_size = model_size
                    self._loaded = True
                    return

            logging.info(f"Loading Depth-Anything V2 {model_size.upper()} model...")
            self._model = local_first(AutoModelForDepthEstimation.from_pretrained,
                model_id, torch_dtype=self.torch_dtype
            ).to(self.device)
            self._model.eval()

            if self._compile_fn is not None:
                self._model = self._compile_fn(self._model, 'depth')

            self._current_model_size = model_size
            self._loaded = True
            logging.info(
                f"Depth-Anything V2 {model_size.upper()} loaded on {self.device} ({self.torch_dtype})"
            )
        except Exception as e:
            logging.error(f"Failed to load Depth-Anything: {e}")
            self._model = None
            self._processor = None
            self._current_model_size = None
            self._loaded = False
            torch.cuda.empty_cache()
        finally:
            self._emit_warning(False)

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._depth_mean = None
        self._depth_std = None
        self._depth_cache = None
        self._output_buffer_max = None
        self._temp_cpu_buffer = None
        self._current_model_size = None
        self._loaded = False
        torch.cuda.empty_cache()
        logging.info("[DepthProcessor] Unloaded")

    def process(self, image_tensor: torch.Tensor, config) -> Optional[torch.Tensor]:
        """Run depth estimation. Input/output: CHW [0,1] on GPU."""
        if hasattr(config, 'method'):
            method = config.method
            blur_kernel = config.blur_kernel
            invert = config.invert
            contrast = config.contrast
            brightness = config.brightness
            near_threshold = config.near_threshold
            far_threshold = config.far_threshold
            depth_resolution = config.resolution
        else:
            method = config.get('depth_method', 'grayscale')
            blur_kernel = config.get('depth_blur_kernel', 5)
            invert = config.get('depth_invert', False)
            contrast = config.get('depth_contrast', 1.0)
            brightness = config.get('depth_brightness', 0)
            near_threshold = config.get('depth_near_threshold', 0)
            far_threshold = config.get('depth_far_threshold', 255)
            depth_resolution = config.get('depth_resolution', self._downscale_size)

        if self._processor is not None and self._model is not None and method == 'grayscale':
            try:
                depth_gpu = self._process_ai_depth(
                    image_tensor, depth_resolution, blur_kernel,
                    invert, contrast, brightness, near_threshold, far_threshold
                )
                return self._depth_gpu_to_output(depth_gpu)
            except Exception as e:
                logging.warning(f"Depth-Anything failed: {e}, falling back to simple method")
                depth = self._process_simple_depth(image_tensor, 'grayscale', blur_kernel,
                                                    invert, contrast, brightness,
                                                    near_threshold, far_threshold)
        else:
            depth = self._process_simple_depth(image_tensor, method, blur_kernel,
                                                invert, contrast, brightness,
                                                near_threshold, far_threshold)

        return self._depth_to_tensor(depth)

    def _process_ai_depth(self, image_tensor, depth_resolution, blur_kernel,
                           invert, contrast, brightness, near_threshold, far_threshold):
        """GPU-resident Depth-Anything V2 inference."""
        original_h, original_w = image_tensor.shape[1], image_tensor.shape[2]

        # TRT engine input shape is baked at build time — override the
        # config-driven resolution. Duck-typed (image_size attr) to avoid
        # importing the TRT module from this preprocessor.
        engine_image_size = getattr(self._model, 'image_size', None)
        if engine_image_size is not None:
            depth_resolution = engine_image_size

        downscaled = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0),
            size=(depth_resolution, depth_resolution),
            mode='bilinear', align_corners=False
        ).squeeze(0)

        # Clamp HDR highlights to [0,1]: the old >1 -> /255 heuristic blacked out
        # HDR frames, yielding a radial-gradient depth.
        downscaled = downscaled.clamp(0.0, 1.0)

        normalized = (downscaled - self._depth_mean) / self._depth_std
        inputs = {"pixel_values": normalized.unsqueeze(0)}

        with torch.no_grad():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(original_h, original_w),
            mode="bilinear", align_corners=False,
        ).squeeze()

        depth_gpu = prediction.detach()
        depth_min = depth_gpu.min()
        depth_max = depth_gpu.max()
        depth_gpu = ((depth_gpu - depth_min) / (depth_max - depth_min) * 255.0)

        if blur_kernel > 1 and self._gaussian_kernel is not None:
            kernel_2d = self._gaussian_kernel.to(dtype=depth_gpu.dtype)
            kernel_size = kernel_2d.shape[-1]
            depth_gpu = torch.nn.functional.conv2d(
                depth_gpu.unsqueeze(0).unsqueeze(0),
                kernel_2d, padding=kernel_size // 2
            ).squeeze()
        elif blur_kernel > 1:
            blur_kernel_odd = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            sigma = 0.3 * ((blur_kernel_odd - 1) * 0.5 - 1) + 0.8
            kernel_1d = torch.exp(
                -torch.arange(-(blur_kernel_odd // 2), blur_kernel_odd // 2 + 1,
                              dtype=depth_gpu.dtype, device=self.device) ** 2
                / (2 * sigma ** 2)
            )
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
            kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
            depth_gpu = torch.nn.functional.conv2d(
                depth_gpu.unsqueeze(0).unsqueeze(0),
                kernel_2d, padding=blur_kernel_odd // 2
            ).squeeze()

        if contrast != 1.0 or brightness != 0:
            depth_gpu = torch.clamp(depth_gpu * contrast + brightness, 0, 255)

        if near_threshold > 0 or far_threshold < 255:
            depth_gpu = torch.clamp(depth_gpu, near_threshold, far_threshold)
            if far_threshold > near_threshold:
                depth_gpu = ((depth_gpu - near_threshold) / (far_threshold - near_threshold) * 255.0)

        if invert:
            depth_gpu = 255.0 - depth_gpu

        del prediction
        return depth_gpu

    def _process_simple_depth(self, image_tensor, method, blur_kernel,
                               invert, contrast, brightness, near_threshold, far_threshold):
        """CPU-based simple depth estimation (fallback)."""
        h, w = image_tensor.shape[1], image_tensor.shape[2]
        if self._temp_cpu_buffer is None or self._temp_cpu_buffer.shape != (h, w, 3):
            self._temp_cpu_buffer = np.empty((h, w, 3), dtype=np.uint8)

        # Scale+cast on the GPU and do one uint8 D2H into the pre-allocated
        # buffer (blocking: the cv2 path reads it right after), instead of
        # copying floats down then doing a full-res CPU multiply + astype.
        gpu_u8 = (image_tensor.permute(1, 2, 0) * 255).clamp_(0, 255).to(torch.uint8)
        torch.from_numpy(self._temp_cpu_buffer).copy_(gpu_u8)
        del gpu_u8

        image_np = self._temp_cpu_buffer

        if method == 'sobel':
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            depth = np.uint8(np.clip(np.sqrt(sobelx ** 2 + sobely ** 2), 0, 255))
        elif method == 'laplacian':
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            depth = np.uint8(np.absolute(cv2.Laplacian(gray, cv2.CV_64F)))
        else:
            depth = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            if blur_kernel > 1:
                bk = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
                depth = cv2.GaussianBlur(depth, (bk, bk), 0)
            if contrast != 1.0 or brightness != 0:
                depth = cv2.convertScaleAbs(depth, alpha=contrast, beta=brightness)
            if near_threshold > 0 or far_threshold < 255:
                depth = np.clip(depth, near_threshold, far_threshold)
                if far_threshold > near_threshold:
                    depth = ((depth - near_threshold) / (far_threshold - near_threshold) * 255).astype(np.uint8)
            if invert:
                depth = 255 - depth

        return depth

    def _depth_gpu_to_output(self, depth_gpu: torch.Tensor) -> torch.Tensor:
        """Convert GPU grayscale depth to CHW RGB output (no CPU roundtrip)."""
        h, w = depth_gpu.shape[-2], depth_gpu.shape[-1]

        if self._output_buffer_max is None:
            self._output_buffer_max = torch.empty(
                (3, self.max_buffer_size, self.max_buffer_size),
                device=self.device, dtype=self.torch_dtype
            )

        # copy_ respects shape, not stride — writing from a broadcasted view
        # at matching shape (3, h, w) into a non-contiguous slice is fine.
        output_view = self._output_buffer_max[:, :h, :w]

        depth_rgb_normalized = (
            depth_gpu.to(self.torch_dtype).unsqueeze(0).expand(3, -1, -1) / 255.0
        )
        output_view.copy_(depth_rgb_normalized)

        self._cached_result = output_view
        return output_view

    def _depth_to_tensor(self, depth: np.ndarray) -> torch.Tensor:
        """Convert grayscale depth numpy array to CHW RGB tensor on GPU."""
        h, w = depth.shape[0], depth.shape[1]

        if self._output_buffer_max is None:
            self._output_buffer_max = torch.empty(
                (3, self.max_buffer_size, self.max_buffer_size),
                device=self.device, dtype=self.torch_dtype
            )

        output_buffer = self._output_buffer_max[:, :h, :w].contiguous()

        depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        depth_temp = torch.from_numpy(depth_rgb).float() / 255.0
        depth_temp_permuted = depth_temp.permute(2, 0, 1).to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        output_buffer.copy_(depth_temp_permuted, non_blocking=True)
        del depth_temp, depth_temp_permuted

        self._cached_result = output_buffer
        return output_buffer
