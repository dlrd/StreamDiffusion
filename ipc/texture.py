"""Shared CUDA texture between Smode (OpenGL) and Python (PyTorch)."""
from __future__ import annotations

import torch
from torch.multiprocessing import reductions


class StreamDiffusionSmodeTexture:
    """Pair of GPU tensors for zero-copy frame exchange with Smode.

    ``smode_tensor`` is float32 HWC and exposed via CUDA IPC.
    ``stream_diffusion_tensor`` is engine-dtype HWC, internal working copy.
    """

    def __init__(
        self, device: int, width: int, height: int, channels: int, dtype: torch.dtype
    ):
        self.device = device
        self.width = width
        self.height = height
        self.channels = channels
        self.dtype = dtype

        self.stream_diffusion_tensor = torch.empty(
            (self.height, self.width, channels), dtype=self.dtype, device=self.device
        )
        self.smode_tensor = torch.empty(
            (self.height, self.width, channels), dtype=torch.float32, device=self.device
        )
        self.smode_tensor_ipc_info = reductions.reduce_tensor(
            self.smode_tensor
        )[1]

        # Pre-allocated CHW buffer for the fused permute+vflip.
        # Smode delivers top-left origin (OpenGL); engines expect bottom-left.
        self._chw_buffer = torch.empty(
            (channels, self.height, self.width), dtype=self.dtype, device=self.device
        )
        self._vflip_row_idx = torch.arange(
            self.height - 1, -1, -1, device=self.device, dtype=torch.long
        )

        # Non-finite input values, counted on the GPU (no per-frame sync) on one frame out of
        # NONFINITE_CHECK_EVERY and read back by pop_nonfinite_stats() every few seconds:
        # [frames with some, values, values in the worst frame].
        self._nonfinite = torch.zeros(3, dtype=torch.int64, device=self.device)
        self._nonfinite_checked = 0
        self._frame_index = 0

    NONFINITE_CHECK_EVERY = 10

    def copy_smode_to_stream_diffusion(self):
        """Copy smode_tensor into stream_diffusion_tensor with implicit dtype cast, clamped to
        [0, 1]: Smode can deliver HDR values above 1, which the VAE encoder and the ControlNet
        preprocessors would otherwise receive out of range.

        Non-finite values are replaced first (NaN -> 0, +Inf -> 1, -Inf -> 0): clamp_ keeps
        NaN, and a single NaN pixel makes the whole generated frame NaN, since the VAE encoder
        and every UNet self-attention mix all positions (then displayed black, and carried to
        the next frames by StreamV2V). Smode colour modifiers can emit them on out-of-range
        input, such as the negative values of a limited-range YUV camera.

        Every frame is cleaned; counting them for the log costs ~0.1 ms, so it runs on one
        frame out of NONFINITE_CHECK_EVERY only."""
        if self._frame_index % self.NONFINITE_CHECK_EVERY == 0:
            bad = self.smode_tensor.isfinite().logical_not_().sum()
            self._nonfinite[0] += bad > 0
            self._nonfinite[1] += bad
            self._nonfinite[2] = torch.maximum(self._nonfinite[2], bad)
            self._nonfinite_checked += 1
        self._frame_index += 1
        self.stream_diffusion_tensor.copy_(self.smode_tensor)
        torch.nan_to_num_(self.stream_diffusion_tensor, nan=0.0, posinf=1.0, neginf=0.0)
        self.stream_diffusion_tensor.clamp_(0.0, 1.0)

    def pop_nonfinite_stats(self) -> tuple:
        """(frames with non-finite input, frames checked, values, values in the worst frame)
        since the last call. One sync: call it every few seconds, not per frame."""
        frames, values, worst = self._nonfinite.tolist()
        checked = self._nonfinite_checked
        self._nonfinite.zero_()
        self._nonfinite_checked = 0
        return frames, checked, values, worst

    def get_permuted_input_tensor(self) -> torch.Tensor:
        """Return CHW + vertically flipped input in a single GPU copy."""
        permuted = self.stream_diffusion_tensor.permute(2, 0, 1)
        torch.index_select(permuted, 1, self._vflip_row_idx, out=self._chw_buffer)
        return self._chw_buffer

    def write_chw_to_smode(self, chw_tensor: torch.Tensor) -> None:
        """Write a CHW engine-output tensor directly into the Smode IPC buffer."""
        self.smode_tensor.copy_(chw_tensor.permute(1, 2, 0))
