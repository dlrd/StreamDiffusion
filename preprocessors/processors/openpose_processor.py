"""OpenPose / DWPose preprocessor for ControlNet."""
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from ..base import BasePreprocessor
from utils.hub import local_first

# processors/ → preprocessors/ → package root.
PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent


class OptimizedDWposeDetector:
    """Custom DWPose detector wrapping easy_dwpose ``Wholebody`` with the
    standard DWposeDetector-compatible interface (YOLOX-S + DWPose-M)."""

    def __init__(self, pose_estimation_model):
        self.pose_estimation = pose_estimation_model
        # Temporal keypoint smoothing: EMA vs jitter, hysteresis vs flicker.
        self.ema_alpha = 0.5        # new-frame weight (1.0 = no smoothing)
        self.score_threshold = 0.4  # also filters hallucinated out-of-frame points
        self.hold_frames = 3        # frames a vanished point is kept alive
        self.min_good_frames = 2    # stability required before a point earns hold
        self._prev_kpts = None
        self._prev_scores = None
        self._low_count = None
        self._good_count = None

    def _reset_state(self, kpts, scores):
        self._prev_kpts = kpts.copy()
        self._prev_scores = scores.copy()
        self._low_count = np.zeros(scores.shape, dtype=np.int32)
        # Seed as established so a tracking reset doesn't blank the skeleton.
        self._good_count = (scores > self.score_threshold).astype(np.int32) * self.min_good_frames

    def _match_persons(self, kpts, scores):
        # YOLOX detection order can swap between frames; realign previous state
        # by centroid distance or the EMA blends two people's skeletons.
        n = kpts.shape[0]
        if n == 1:
            return

        def centroids(k, s):
            c = np.full((k.shape[0], 2), 0.5)
            for i in range(k.shape[0]):
                m = s[i] > self.score_threshold
                if m.any():
                    c[i] = k[i][m].mean(axis=0)
            return c

        d = ((centroids(kpts, scores)[:, None, :]
              - centroids(self._prev_kpts, self._prev_scores)[None, :, :]) ** 2).sum(-1)
        order = np.full(n, -1, dtype=int)
        for _ in range(n):
            i, j = np.unravel_index(np.argmin(d), d.shape)
            order[i] = j
            d[i, :] = np.inf
            d[:, j] = np.inf
        self._prev_kpts = self._prev_kpts[order]
        self._prev_scores = self._prev_scores[order]
        self._low_count = self._low_count[order]
        self._good_count = self._good_count[order]

    def _smooth(self, kpts, scores):
        if self._prev_kpts is None or kpts.shape != self._prev_kpts.shape:
            self._reset_state(kpts, scores)
            return kpts, scores
        self._match_persons(kpts, scores)
        good = scores > self.score_threshold
        prev_good = self._prev_scores > self.score_threshold
        both = good & prev_good
        kpts[both] = self.ema_alpha * kpts[both] + (1.0 - self.ema_alpha) * self._prev_kpts[both]
        # Anti-flicker: only stable points earn a hold; one-frame parasites don't.
        established = self._good_count >= self.min_good_frames
        dropped = (~good) & prev_good & established
        self._low_count[good] = 0
        self._low_count[~good] += 1
        hold = dropped & (self._low_count <= self.hold_frames)
        kpts[hold] = self._prev_kpts[hold]
        scores[hold] = self._prev_scores[hold]
        self._good_count[good] += 1
        self._good_count[(~good) & (~hold)] = 0
        self._prev_kpts = kpts.copy()
        self._prev_scores = scores.copy()
        return kpts, scores

    @torch.inference_mode()
    def __call__(self, image, detect_resolution=512, draw_pose=None, output_type="pil", **kwargs):
        from easy_dwpose.body_estimation import resize_image
        from easy_dwpose.draw import draw_openpose

        if draw_pose is None:
            draw_pose = draw_openpose

        if type(image) != np.ndarray:
            image = np.array(image.convert("RGB"))

        image = image.copy()
        original_height, original_width, _ = image.shape

        image = resize_image(image, target_resolution=detect_resolution)
        height, width, _ = image.shape

        candidates, scores = self.pose_estimation(image)

        num_candidates, _, locs = candidates.shape
        candidates[..., 0] /= float(width)
        candidates[..., 1] /= float(height)

        candidates, scores = self._smooth(candidates, scores)

        # Hard cut below threshold for body+hands+face: hallucinated points
        # otherwise reach the draw step and create phantom limbs. A point must
        # also be stable min_good_frames before it becomes drawable, so a
        # one-frame hallucination never draws anything.
        drawable = scores > self.score_threshold
        if self._good_count is not None and self._good_count.shape == scores.shape:
            drawable &= self._good_count >= self.min_good_frames
        scores = np.where(drawable, scores, 0.0)
        # draw_handpose/draw_facepose ignore scores and draw any positive
        # coordinate: filtered points must be masked out via coords.
        candidates[scores <= 0.0] = -1.0

        # A real hand keeps most of its 21 joints; phantom/duplicate hands don't.
        hand_slices = (slice(92, 113), slice(113, 134))
        for i in range(num_candidates):
            means = []
            for sl in hand_slices:
                v = scores[i, sl] > 0.0
                means.append(scores[i, sl][v].mean() if v.any() else 0.0)
                if v.sum() < 7:
                    candidates[i, sl] = -1.0
                    scores[i, sl] = 0.0
            # Left/right collapsed onto the same physical hand: keep the best.
            l_sl, r_sl = hand_slices
            vl = scores[i, l_sl] > 0.0
            vr = scores[i, r_sl] > 0.0
            if vl.any() and vr.any():
                cl = candidates[i, l_sl][vl].mean(axis=0)
                cr = candidates[i, r_sl][vr].mean(axis=0)
                if ((cl - cr) ** 2).sum() < 0.0025:
                    drop = l_sl if means[0] < means[1] else r_sl
                    candidates[i, drop] = -1.0
                    scores[i, drop] = 0.0

        bodies = candidates[:, :18].copy()
        bodies = bodies.reshape(num_candidates * 18, locs)

        body_scores = scores[:, :18]
        for i in range(len(body_scores)):
            for j in range(len(body_scores[i])):
                if body_scores[i][j] > self.score_threshold:
                    body_scores[i][j] = int(18 * i + j)
                else:
                    body_scores[i][j] = -1

        faces = candidates[:, 24:92]
        faces_scores = scores[:, 24:92]

        hands = np.vstack([candidates[:, 92:113], candidates[:, 113:]])
        hands_scores = np.vstack([scores[:, 92:113], scores[:, 113:]])

        pose = dict(
            bodies=bodies, body_scores=body_scores,
            hands=hands, hands_scores=hands_scores,
            faces=faces, faces_scores=faces_scores,
        )

        if not draw_pose:
            return pose

        import PIL.Image
        pose_image = draw_pose(pose, height=height, width=width, **kwargs)
        pose_image = cv2.resize(pose_image, (original_width, original_height), cv2.INTER_LANCZOS4)

        if output_type == "pil":
            pose_image = PIL.Image.fromarray(pose_image)
        elif output_type == "np":
            pass
        else:
            raise ValueError("output_type should be 'pil' or 'np'")

        return pose_image


class OpenPoseProcessor(BasePreprocessor):
    """DWPose human pose detection preprocessor (YOLOX-S + DWPose-M)."""

    def __init__(self, device: torch.device, torch_dtype: torch.dtype, max_buffer_size: int = 1024,
                 warning_callback=None):
        super().__init__(device, torch_dtype, max_buffer_size, warning_callback)
        self._detector: Optional[OptimizedDWposeDetector] = None
        self._input_buffer_max: Optional[np.ndarray] = None
        self._output_buffer_max: Optional[torch.Tensor] = None
        self._pose_cache: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return "openpose"

    def load_model(self, config) -> None:
        """Load DWPose detector with optimized ONNX models."""
        if self._detector is not None:
            return

        self._emit_warning(True, "Loading DWPose preprocessor (YOLOX-S + DWPose-LL-384) - first run can take a while (downloading + loading weights)...")
        try:
            from huggingface_hub import hf_hub_download
            from easy_dwpose.body_estimation import Wholebody

            logging.info("Loading DWPose preprocessor (YOLOX-S + DWPose-LL-384, GPU-accelerated)...")

            model_det_path = local_first(hf_hub_download,
                "hr16/yolox-onnx", "yolox_s.onnx",
                local_dir=str(PACKAGE_DIR / "checkpoints")
            )
            model_pose_path = local_first(hf_hub_download,
                "hr16/UnJIT-DWPose", "dw-ll_ucoco_384_fp16.onnx",
                local_dir=str(PACKAGE_DIR / "checkpoints")
            )

            pose_estimation = Wholebody(
                device=self.device,
                model_det=model_det_path,
                model_pose=model_pose_path
            )

            self._detector = OptimizedDWposeDetector(pose_estimation)
            self._loaded = True
            logging.info("DWPose loaded successfully (YOLOX-S + DWPose-LL-384)")
        except Exception as e:
            logging.error(f"Failed to load DWPose: {e}")
            self._detector = None
            self._loaded = False
            torch.cuda.empty_cache()
        finally:
            self._emit_warning(False)

    def unload_model(self) -> None:
        if self._detector is not None:
            if hasattr(self._detector.pose_estimation, 'cleanup'):
                self._detector.pose_estimation.cleanup()
            del self._detector
            self._detector = None
        self._input_buffer_max = None
        self._output_buffer_max = None
        self._pose_cache = None
        self._loaded = False
        torch.cuda.empty_cache()
        logging.info("[OpenPoseProcessor] Unloaded")

    def process(self, image_tensor: torch.Tensor, config) -> Optional[torch.Tensor]:
        """Run DWPose detection. Input/output: CHW [0,1] on GPU."""
        if self._detector is None:
            logging.warning("DWPose processor not loaded, returning original image")
            return image_tensor

        try:
            h, w = image_tensor.shape[1], image_tensor.shape[2]

            if hasattr(config, 'detect_resolution'):
                detect_resolution = config.detect_resolution
            else:
                detect_resolution = config.get('openpose_detect_resolution', 512)

            if self._input_buffer_max is None:
                self._input_buffer_max = np.empty(
                    (self.max_buffer_size, self.max_buffer_size, 3), dtype=np.uint8
                )

            input_buffer = self._input_buffer_max[:h, :w, :]

            if self._output_buffer_max is None:
                self._output_buffer_max = torch.empty(
                    (3, self.max_buffer_size, self.max_buffer_size),
                    device=self.device, dtype=self.torch_dtype
                )

            output_buffer = self._output_buffer_max[:, :h, :w].contiguous()

            # Convert GPU tensor to numpy for DWPose. Scale+cast on the GPU and
            # do a single uint8 D2H into the pre-allocated buffer, instead of
            # copying fp16/fp32 down then doing two full-res CPU passes
            # (multiply + astype) per frame. Blocking copy: the CPU detector
            # reads input_buffer immediately after.
            gpu_u8 = (image_tensor.permute(1, 2, 0) * 255).clamp_(0, 255).to(torch.uint8)
            torch.from_numpy(input_buffer).copy_(gpu_u8)
            del gpu_u8

            openpose_np = self._detector(
                input_buffer,
                detect_resolution=detect_resolution,
                output_type='np',
                include_hands=True,
                include_face=True,
            )

            openpose_temp = torch.from_numpy(openpose_np).permute(2, 0, 1).to(self.torch_dtype) * (1.0 / 255.0)
            del openpose_np

            output_buffer.copy_(openpose_temp)
            del openpose_temp

            self._cached_result = output_buffer
            return output_buffer

        except Exception as e:
            logging.error(f"DWPose processing failed: {e}")
            return image_tensor
