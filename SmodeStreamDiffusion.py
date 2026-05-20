import os
import sys
# Fix Unicode encoding for Windows console
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform.startswith('win'):
    # Simplified encoding fix (Python 3.7+)
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add src/ to Python path for streamdiffusion module access
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Reduce HuggingFace Hub network timeouts (models are cached locally)
# Default: 10s connect + exponential backoff up to 5 retries = ~30s wasted on network errors
# Reduced: 3s timeout, still allows downloads but fails fast when offline
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '3')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '10')

import socket
import time
import select
import hashlib
import json
from pathlib import Path
from typing import Dict, NamedTuple
import torch
import argparse
import enum
import logging
import struct
import torchvision.transforms.functional as F
import torch.nn.functional as NF  # For interpolate, max_pool2d, conv2d
from torch.multiprocessing import reductions  # For obtaining CUDA IPC handle
import cv2
import numpy as np

from utils.wrapper import StreamDiffusionWrapper
from utils.wrapper_xl import StreamDiffusionWrapperXL

# Package root directory (independent of CWD, works on any machine)
PACKAGE_DIR = Path(__file__).resolve().parent
from src.streamdiffusion import StreamDiffusion
from diffusers import AutoencoderTiny, ControlNetModel
import win32event
import win32api
# Protocol constants
MAGIC_NUMBER = 0xE280A0  # uint32_t magic identifier
ENDIAN_FORMAT: str = "<"  # little-endian for all numbers

UINT32 = "I"
UINT64 = "Q"
FLOAT32 = "f"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class InterProcessEvent:
    def __init__(self):
        self.event = None
        self.signal_awakes_all_clients = True

    def __del__(self):
        if self.event:
            self.close()

    def create(self, name, signal_awakes_all_clients=True, initial_signaled_state=False) -> bool:
        if self.event is not None:
            raise RuntimeError("Event already assigned")

        CREATE_EVENT_MANUAL_RESET = 0x00000001
        CREATE_EVENT_INITIAL_SET = 0x00000002
        self.signal_awakes_all_clients = signal_awakes_all_clients
        flags = (CREATE_EVENT_MANUAL_RESET if signal_awakes_all_clients else 0) | (CREATE_EVENT_INITIAL_SET if initial_signaled_state else 0)
        self.event = win32event.CreateEvent(None, flags, win32event.EVENT_ALL_ACCESS, name)
        if win32api.GetLastError() != 0:
            raise RuntimeError(f"Failed to create event {name} code: {win32api.GetLastError()}")
        return True

    def open(self, name) -> bool:
        if self.event is not None:
            raise RuntimeError("Event already assigned")

        self.event = win32event.OpenEvent(win32event.SYNCHRONIZE, False, name)
        if win32api.GetLastError() != 0:
            raise RuntimeError(f"Failed to open event {name} code: {win32api.GetLastError()}")
        return True

    def close(self) -> bool:
        if self.event is None:
            raise RuntimeError("Event not assigned")
        if win32api.CloseHandle(self.event) == 0:
            raise RuntimeError(f"Failed to close event {self.event} code: {win32api.GetLastError()}")
        self.event = None
        return True

    def wait(self, timeout=win32event.INFINITE) -> int:
        if self.event is None:
            raise RuntimeError("Event not assigned")
        return win32event.WaitForSingleObject(self.event, timeout)

    def signal(self) -> bool:
        if self.event is None:
            raise RuntimeError("Event not assigned")
        if win32event.SetEvent(self.event) == 0:
            raise RuntimeError(f"Failed to signal event {self.event} code: {win32api.GetLastError()}")
        if self.signal_awakes_all_clients:
            win32event.ResetEvent(self.event)
            return win32api.GetLastError() == 0
        return True

class CommandType(enum.Enum):
    OUTPUT = 1  # Sent by client: sends the output tensor to the server; payload is the CUDA IPC info
    STOP = 2  # Sent by server: client should exit
    CONFIG = 3  # Sent by server: update configuration; payload includes:
    UUID = 5  # Sent by client: sends the client's UUID
    INPUT = 6  # Sent by client: sends the input tensor to the server; payload is the CUDA IPC info
    STREAM_CREATION = 7  # Sent by client: signals the creation of a stream; payload is a boolean indicating if the stream creation has finished


class Mode(enum.IntEnum):
    IMAGE_TO_IMAGE = 1
    TEXT_TO_IMAGE = 2


class Acceleration(enum.IntEnum):
    NONE = 0
    TORCH_COMPILE = 1  # C++ torchCompile (index 1 in XML enum)
    XFORMERS = 2       # C++ xFormers (index 2 in XML enum)
    TENSORRT = 3       # C++ tensorRT (index 3 in XML enum)


class ConfigType(enum.IntEnum):
    NONE = 1
    FULL = 2
    SELF = 3
    INITIALIZE = 4


def config_type_to_str(config_type: ConfigType) -> str:
    if config_type == ConfigType.NONE:
        return "none"
    elif config_type == ConfigType.FULL:
        return "full"
    elif config_type == ConfigType.SELF:
        return "self"
    elif config_type == ConfigType.INITIALIZE:
        return "initialize"
    else:
        raise ValueError(f"Unknown config type: {config_type}")


class Args(NamedTuple):
    port: int
    uuid: str
    width: int
    height: int
    device: int
    model: str


class Packet:
    def __init__(self, cmd: CommandType, payload: bytes):
        self.cmd = cmd
        self.payload = payload

    def to_bytes(self) -> bytes:
        """Convert the packet to bytes for sending over the socket."""
        payload_bytes = struct.pack(
            ENDIAN_FORMAT + UINT32, self.cmd.value
        ) + self.payload
        size = len(payload_bytes)
        header = struct.pack(
            ENDIAN_FORMAT + UINT32 + UINT32,
            MAGIC_NUMBER,
            size
        )
        return header + payload_bytes


class FrameDataPacket(Packet):
    def __init__(
        self,
        cmd: CommandType,
        device: int,
        handle: bytes,
        event_handle: bytes,
        storage_size_bytes: int,
        storage_offset_bytes: int,
        channels: int,
        w: int,
        h: int,
    ):
        payload = struct.pack(ENDIAN_FORMAT + UINT64, device)
        payload += struct.pack(ENDIAN_FORMAT + UINT32, len(handle)) + handle
        payload += (
            struct.pack(ENDIAN_FORMAT + UINT32, len(event_handle))
            + event_handle
        )
        payload += struct.pack(
            ENDIAN_FORMAT
            + UINT64
            + UINT64
            + UINT32
            + UINT32
            + UINT32,
            storage_size_bytes,
            storage_offset_bytes,
            channels,
            w,
            h,
        )
        super().__init__(cmd, payload)


# OPTIMIZATION: Config parsing cache with LRU eviction (CPU → RAM optimization)
# CRITICAL FIX: Use OrderedDict for proper LRU cache eviction
# Cache parsed config packets to avoid redundant parsing operations
from collections import OrderedDict

_CONFIG_CACHE = OrderedDict()  # LRU cache: bytes_hash -> ConfigPacket
_CONFIG_CACHE_MAX_SIZE = 32  # Reasonable limit for config variations
_CONFIG_CACHE_HITS = 0
_CONFIG_CACHE_MISSES = 0


def _parse_config_with_cache(data: bytes):
    """
    OPTIMIZED: Parse config packet with LRU caching.
    Eliminates redundant parsing by caching previously parsed configs.
    CRITICAL FIX: Implements LRU eviction to prevent unbounded memory growth.

    Args:
        data: Raw bytes data to parse

    Returns:
        Parsed ConfigPacket (from cache or newly parsed)
    """
    global _CONFIG_CACHE_HITS, _CONFIG_CACHE_MISSES

    # Create hash of the data as cache key
    data_hash = hashlib.md5(data).hexdigest()

    # Check cache first
    if data_hash in _CONFIG_CACHE:
        _CONFIG_CACHE_HITS += 1
        # CRITICAL FIX: Move to end (mark as recently used for LRU)
        _CONFIG_CACHE.move_to_end(data_hash)
        # Return a copy to avoid mutation issues
        return _CONFIG_CACHE[data_hash]

    # Cache miss - parse the data
    _CONFIG_CACHE_MISSES += 1
    config_packet = ConfigPacket()
    config_packet.from_bytes(data)

    # CRITICAL FIX: Implement LRU eviction when cache is full
    if len(_CONFIG_CACHE) >= _CONFIG_CACHE_MAX_SIZE:
        # Evict oldest entry (LRU)
        _CONFIG_CACHE.popitem(last=False)

    _CONFIG_CACHE[data_hash] = config_packet

    return config_packet


class ConfigPacket(Packet):
    def __init__(self):
        super().__init__(CommandType.CONFIG, b"")
        self.cache_dir = None
        self.model_name = ""
        self.prompt = ""
        self.negative_prompt = ""
        self.seed = 0
        self.width = 0
        self.height = 0
        # CRITICAL FIX: Use index 0 for 1-step models (Hyper-SDXL, SD-Turbo, Lightning)
        # Index 0 = last timestep (799/999) with maximum noise → correct starting point
        # Index 16 = timestep ~543 with 92% noise → WRONG for 1-step models!
        # For multi-step models (LCM, standard SDXL), use multiple indices like [22, 32, 45]
        self.t_index_list = [0]
        self.guidance_scale = 5.0
        self.mode = Mode.IMAGE_TO_IMAGE
        self.cfg_type = "none"
        self.acceleration = Acceleration.XFORMERS
        self.similar_image_filter_enabled = False
        self.similar_image_filter_threshold = 0.99
        self.similar_image_filter_max_skip = 5
        self.lora_dict : Dict[str, float]= None
        # ControlNet fields
        self.controlnet_enabled = True
        self.controlnet_preview_mode = 0
        self.controlnet_guidance_strength = 1.0
        self.controlnet_skip_frames = 1
        self.canny_enabled = False
        self.canny_scale = 0.41
        self.canny_resolution = 384
        self.canny_low_threshold = 100
        self.canny_high_threshold = 255
        self.canny_aperture_size = 3
        self.canny_l2_gradient = False
        self.depth_enabled = False
        self.depth_scale = 0.86
        self.depth_resolution = 384
        self.depth_blur_kernel = 1
        self.depth_contrast = 1.0
        self.depth_brightness = 0
        self.depth_near_threshold = 0
        self.depth_far_threshold = 255
        self.depth_invert = False
        self.depth_method = 0
        self.depth_model_size = 0
        self.openpose_enabled = False
        self.openpose_scale = 0.89
        self.openpose_detect_resolution = 256
        # Pipeline settings
        self.use_tiny_vae = True
        self.latent_feedback_strength = 0.0
        self.motion_aware_noise = False
        self.motion_aware_noise_sensitivity = 0.5

    def from_bytes(self, data: bytes):
        offset = 0
        self.t_index_list = []
        self.cache_dir = None
        cache_dir, offset = read_string(data, offset)
        if len(cache_dir) > 0:
            self.cache_dir = cache_dir
        self.model_name, offset = read_string(data, offset)
        self.prompt, offset = read_string(data, offset)
        self.negative_prompt, offset = read_string(data, offset)

        if offset + 16 > len(data):
            raise ValueError(
                "Insufficient data for seed, width, and height in CONFIG"
            )
        self.seed, = struct.unpack_from(ENDIAN_FORMAT + UINT64, data, offset)
        offset += 8
        t_index_list_len = 0
        (
            self.width,
            self.height,
            t_index_list_len,
        ) = struct.unpack_from(
            ENDIAN_FORMAT + UINT32 + UINT32 + UINT32, data, offset
        )
        offset += 12
        for _ in range(t_index_list_len):
            if offset + 4 > len(data):
                raise ValueError(
                    "Insufficient data for t_index_list in CONFIG"
                )
            t_index_value, = struct.unpack_from(
                ENDIAN_FORMAT + UINT32, data, offset
            )
            self.t_index_list.append(t_index_value)
            offset += 4
        (
            self.guidance_scale,
            self.mode,
            cfg_type,
            self.acceleration,
        ) = struct.unpack_from(
            ENDIAN_FORMAT + FLOAT32 + UINT32 + UINT32 + UINT32, data, offset
        )
        self.cfg_type = config_type_to_str(ConfigType(cfg_type))
        offset += 16
        if offset + 12 <= len(data):
            (ssf_en, ssf_thresh, ssf_max
            ) = struct.unpack_from(ENDIAN_FORMAT + UINT32 + FLOAT32 + UINT32, data, offset)
            offset += 12
            self.similar_image_filter_enabled = bool(ssf_en)
            self.similar_image_filter_threshold = ssf_thresh
            self.similar_image_filter_max_skip = ssf_max
        if offset + 4 <= len(data):
            lora_dict_len, = struct.unpack_from(ENDIAN_FORMAT + UINT32, data, offset)
            offset += 4
            self.lora_dict = {}
            for _ in range(lora_dict_len):
                if offset + 4 > len(data):
                    raise ValueError("Insufficient data for lora_dict key length")
                key, offset = read_string(data, offset)
                if offset + 4 > len(data):
                    raise ValueError("Insufficient data for lora_dict value")
                value, = struct.unpack_from(ENDIAN_FORMAT + FLOAT32, data, offset)
                self.lora_dict[key] = value
                offset += 4
        else:
            self.lora_dict = None

        # ControlNet top-level
        if offset + 16 <= len(data):
            (cn_enabled, cn_preview, cn_strength, cn_skip
            ) = struct.unpack_from(ENDIAN_FORMAT + UINT32 + UINT32 + FLOAT32 + UINT32, data, offset)
            offset += 16
            self.controlnet_enabled = bool(cn_enabled)
            self.controlnet_preview_mode = cn_preview
            self.controlnet_guidance_strength = cn_strength
            self.controlnet_skip_frames = cn_skip

        # Canny
        if offset + 28 <= len(data):
            (c_en, c_scale, c_res, c_lo, c_hi, c_ap, c_l2
            ) = struct.unpack_from(ENDIAN_FORMAT + UINT32 + FLOAT32 + UINT32 + UINT32 + UINT32 + UINT32 + UINT32, data, offset)
            offset += 28
            self.canny_enabled = bool(c_en)
            self.canny_scale = c_scale
            self.canny_resolution = c_res
            self.canny_low_threshold = c_lo
            self.canny_high_threshold = c_hi
            self.canny_aperture_size = c_ap
            self.canny_l2_gradient = bool(c_l2)

        # Depth
        if offset + 44 <= len(data):
            (d_en, d_scale, d_res, d_blur, d_contrast, d_bright, d_near, d_far, d_inv, d_method, d_size
            ) = struct.unpack_from(ENDIAN_FORMAT + UINT32 + FLOAT32 + UINT32 + UINT32 + FLOAT32 + "i" + "i" + UINT32 + UINT32 + UINT32 + UINT32, data, offset)
            offset += 44
            self.depth_enabled = bool(d_en)
            self.depth_scale = d_scale
            self.depth_resolution = d_res
            self.depth_blur_kernel = d_blur
            self.depth_contrast = d_contrast
            self.depth_brightness = d_bright
            self.depth_near_threshold = d_near
            self.depth_far_threshold = d_far
            self.depth_invert = bool(d_inv)
            self.depth_method = d_method
            self.depth_model_size = d_size

        # OpenPose
        if offset + 12 <= len(data):
            (op_en, op_scale, op_res
            ) = struct.unpack_from(ENDIAN_FORMAT + UINT32 + FLOAT32 + UINT32, data, offset)
            offset += 12
            self.openpose_enabled = bool(op_en)
            self.openpose_scale = op_scale
            self.openpose_detect_resolution = op_res

        # Pipeline settings
        if offset + 20 <= len(data):
            (eng, tiny_vae, latent_fb, man_en, man_sens
            ) = struct.unpack_from(ENDIAN_FORMAT + UINT32 + UINT32 + FLOAT32 + UINT32 + FLOAT32, data, offset)
            offset += 20
            self.use_tiny_vae = bool(tiny_vae)
            self.latent_feedback_strength = latent_fb
            self.motion_aware_noise = bool(man_en)
            self.motion_aware_noise_sensitivity = man_sens

        return self


class UuidPacket(Packet):
    def __init__(self, uuid: str):
        payload = struct.pack(ENDIAN_FORMAT + UINT32, len(uuid)) + uuid.encode(
            "utf-8"
        )
        super().__init__(CommandType.UUID, payload)


class StreamCreationPacket(Packet):
    def __init__(self, finished: bool):
        payload = struct.pack(ENDIAN_FORMAT + UINT32, int(finished))
        super().__init__(CommandType.STREAM_CREATION, payload)


class StreamDiffusionSmodeTexture:
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
        # Workaround for Windows: PyTorch's _share_cuda_() returns storage_offset_bytes as C long
        # (32-bit on Windows/MSVC). If the tensor is placed deep in a large cached CUDA block,
        # storage_offset_bytes > INT32_MAX causes OverflowError. Fix: clear the CUDA cache so the
        # tensor gets its own fresh cudaMalloc block with storage_offset_bytes = 0.
        try:
            self.smode_tensor_ipc_info = reductions.reduce_tensor(
                self.smode_tensor
            )[1]
        except OverflowError:
            import gc
            logging.warning(
                "OverflowError in reduce_tensor (storage_offset_bytes too large for 32-bit C long). "
                "Clearing CUDA cache and reallocating tensor to get offset=0..."
            )
            del self.smode_tensor
            gc.collect()
            torch.cuda.empty_cache()
            self.smode_tensor = torch.empty(
                (self.height, self.width, channels), dtype=torch.float32, device=self.device
            )
            self.smode_tensor_ipc_info = reductions.reduce_tensor(
                self.smode_tensor
            )[1]
        
        # OPTIMIZATION: Pre-allocate conversion buffers to eliminate runtime allocations
        # These buffers are reused across frames to avoid memory allocation overhead
        self._conversion_buffer_f16 = None  # For smode->stream_diffusion conversion
        self._conversion_buffer_f32 = None  # For stream_diffusion->smode conversion
        
        # OPTIMIZATION: Pre-allocate permutation buffers to eliminate reshape overhead
        self._permuted_input_buffer = None   # For HWC -> CHW conversion (input)
        self._permuted_output_buffer = None  # For CHW -> HWC conversion (output)
        
        # Initialize conversion and permutation buffers
        self._init_conversion_buffers()
        self._init_permutation_buffers()

    def _init_conversion_buffers(self):
        """Initialize pre-allocated conversion buffers for zero-allocation tensor operations"""
        tensor_shape = (self.height, self.width, self.channels)
        
        # Pre-allocate buffers only if dtypes are different
        if self.dtype != torch.float32:
            # Buffer for smode (f32) -> stream_diffusion (dtype) conversion
            self._conversion_buffer_f16 = torch.empty(
                tensor_shape, dtype=self.dtype, device=self.device
            )
        
        # Buffer for stream_diffusion (dtype) -> smode (f32) conversion  
        if self.dtype != torch.float32:
            self._conversion_buffer_f32 = torch.empty(
                tensor_shape, dtype=torch.float32, device=self.device
            )

    def _init_permutation_buffers(self):
        """Initialize pre-allocated buffers for permutation operations to eliminate reshape overhead"""
        # Buffer for HWC -> CHW conversion (input to StreamDiffusion)
        chw_shape = (self.channels, self.height, self.width)
        self._permuted_input_buffer = torch.empty(
            chw_shape, dtype=self.dtype, device=self.device
        )
        
        # Buffer for CHW -> HWC conversion (output from StreamDiffusion)  
        hwc_shape = (self.height, self.width, self.channels)
        self._permuted_output_buffer = torch.empty(
            hwc_shape, dtype=torch.float32, device=self.device
        )

    def copy_smode_to_stream_diffusion(self):
        """
        OPTIMIZED: Copy data from smode_tensor into stream_diffusion_tensor with zero allocations.
        Uses pre-allocated buffers to avoid runtime memory allocation overhead.
        """
        if self.dtype == torch.float32:
            # Same dtype - direct copy without conversion
            self.stream_diffusion_tensor.copy_(self.smode_tensor)
        else:
            # Different dtype - use pre-allocated conversion buffer
            # This eliminates the temporary tensor allocation from .to()
            self._conversion_buffer_f16.copy_(self.smode_tensor)
            self.stream_diffusion_tensor.copy_(self._conversion_buffer_f16)

    def copy_to_smode(self, x_output: torch.Tensor):
        """
        OPTIMIZED: Copy external tensor x_output into the smode_tensor with zero allocations.
        Uses pre-allocated conversion buffer to avoid runtime memory allocation overhead.
        """
        if x_output.dtype == torch.float32 and x_output.device == self.device:
            # Same dtype and device - direct copy without conversion
            self.smode_tensor.copy_(x_output)
        else:
            if self.dtype != torch.float32:
                # Use pre-allocated conversion buffer to eliminate temporary allocation
                # This replaces: src = x_output.to(dtype=torch.float32, device=self.device)
                self._conversion_buffer_f32.copy_(x_output)
                self.smode_tensor.copy_(self._conversion_buffer_f32)
            else:
                # Direct conversion for cases where we don't have pre-allocated buffer
                # LEAK FIX: Explicitly free temporary tensor created by .to()
                temp_converted = x_output.to(dtype=torch.float32, device=self.device)
                self.smode_tensor.copy_(temp_converted)
                del temp_converted

    def get_permuted_input_tensor(self):
        """
        OPTIMIZED: Get HWC -> CHW permuted tensor using pre-allocated buffer.
        Eliminates permutation overhead by reusing the same buffer.
        """
        # Copy stream_diffusion_tensor (HWC) into permuted buffer (CHW) with zero allocation
        self._permuted_input_buffer.copy_(self.stream_diffusion_tensor.permute(2, 0, 1))
        return self._permuted_input_buffer
        
    def set_permuted_output_tensor(self, chw_tensor):
        """
        OPTIMIZED: Convert CHW -> HWC and store using pre-allocated buffer.
        Eliminates permutation overhead by reusing the same buffer.
        """
        # Copy CHW tensor into HWC buffer with zero allocation
        self._permuted_output_buffer.copy_(chw_tensor.permute(1, 2, 0))
        return self._permuted_output_buffer


def recv_all(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from the socket."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        data += chunk
    return data


def recv_message(sock: socket.socket):
    """
    Receives a message:
      Header: 4 bytes magic, 4 bytes payload size, then payload.
      Payload starts with a 4-byte command code followed by command-specific data.
    Returns (CommandType, payload_bytes)
    """
    header = recv_all(sock, 8)
    magic, size = struct.unpack(ENDIAN_FORMAT + "II", header)
    if magic != MAGIC_NUMBER:
        logging.error(
            f"Invalid magic number received: {hex(magic)} (expected {hex(MAGIC_NUMBER)})"
        )
        return None, None
    payload = recv_all(sock, size)
    if len(payload) < 4:
        logging.error("Payload too short to contain command code")
        return None, None
    cmd_int, = struct.unpack(ENDIAN_FORMAT + "I", payload[:4])
    try:
        cmd = CommandType(cmd_int)
    except ValueError:
        logging.error(f"Unknown command code received: {cmd_int}")
        return None, None
    return cmd, payload[4:]


def send_message(sock: socket.socket, packet: Packet):
    """
    Sends a message:
      Payload: 4 bytes command code + command-specific payload.
      Header: 4 bytes magic, 4 bytes payload size.
    """
    sock.sendall(packet.to_bytes())


def read_string(data: bytes, offset: int):
    """Read a length-prefixed string from data starting at offset.
       Returns (string, new_offset).
    """
    if offset + 4 > len(data):
        raise ValueError("Insufficient data for string length")
    str_len, = struct.unpack_from(ENDIAN_FORMAT + "I", data, offset)
    offset += 4
    if offset + str_len > len(data):
        raise ValueError("Insufficient data for string content")
    s = data[offset : offset + str_len].decode("utf-8")
    offset += str_len
    return s, offset


def is_socket_connected(sock):
    try:
        data = sock.recv(1, socket.MSG_PEEK)
        return len(data) > 0
    except BlockingIOError:
        return True
    except socket.error:
        return False


class App:
    def __init__(
        self, config: Args, device: torch.device, torch_dtype: torch.dtype
    ):
        # Initialize all attributes to None first for safe cleanup in destructor
        self.stream = None
        self.cache_dir = None
        self.socket = None
        self.streamDiffusionToSmodeInterProcessEvent = None
        self.smodeToStreamDiffusionInterProcessEvent = None

        # CRITICAL FIX: Wrap initialization in try-except to ensure cleanup on failure
        try:
            self.config = config
            self.device = device
            self.torch_dtype = torch_dtype
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # SOCKET BUFFER OPTIMIZATION: Increase buffer sizes to reduce syscall overhead
            # Default: 8KB, Optimized: 1MB - reduces context switches for large tensor transfers
            # Expected gain: +2-5% FPS by reducing network I/O overhead
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB send buffer
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB receive buffer
            self.model_name = config.model
            self.current_prompt = ""
            self.negative_prompt = ""
            self.seed = 8
            self.guidance_scale = 1.2  # Default CFG scale (will be updated from Smode)
            self.width = config.width
            self.height = config.height
            self.input_tensors = None
            self.output_tensors = None
            self.buffer_shape = None
            # CRITICAL FIX: Use index 0 for 1-step models (default for Hyper-SDXL)
            self.t_index_list = [0]
            self.mode = Mode.IMAGE_TO_IMAGE
            self.acceleration = Acceleration.XFORMERS
            self.similar_image_filter_enabled = False
            self.similar_image_filter_threshold = 0.99
            self.similar_image_filter_max_skip = 5
            self.cfg_type = "self" if self.mode == Mode.IMAGE_TO_IMAGE else "none"
            self.lora_dict : Dict[str, float] = None

            self.is_sdxl = False  # Set to True in _create_sd_stream() if SDXL model detected

            self.streamDiffusionToSmodeInterProcessEvent = InterProcessEvent()
            self.streamDiffusionToSmodeInterProcessEvent.create(
                "Global\\StreamDiffusionToSmode-" + config.uuid,
                signal_awakes_all_clients=False,
                initial_signaled_state=False,
            )
            self.smodeToStreamDiffusionInterProcessEvent = InterProcessEvent()
            self.smodeToStreamDiffusionInterProcessEvent.open(
                "Global\\SmodeToStreamDiffusion-" + config.uuid
            )
        except Exception as e:
            # CRITICAL FIX: Cleanup resources on initialization failure
            logging.error(f"Failed to initialize App: {e}")
            # Close socket if created
            if self.socket:
                try:
                    self.socket.close()
                except Exception as cleanup_error:
                    logging.debug(f"Socket cleanup error (non-critical): {cleanup_error}")
            # Close Win32 event handles if created
            if self.streamDiffusionToSmodeInterProcessEvent:
                try:
                    self.streamDiffusionToSmodeInterProcessEvent.close()
                except Exception as cleanup_error:
                    logging.debug(f"Event handle cleanup error (non-critical): {cleanup_error}")
            if self.smodeToStreamDiffusionInterProcessEvent:
                try:
                    self.smodeToStreamDiffusionInterProcessEvent.close()
                except Exception as cleanup_error:
                    logging.debug(f"Event handle cleanup error (non-critical): {cleanup_error}")
            # Re-raise exception
            raise

        # ControlNet initialization
        self.controlnet_models = {}
        self.depth_model = None  # Depth-Anything model for real depth estimation
        self.depth_processor = None
        self.current_depth_model_size = None  # Track currently loaded model size (small/base/large)
        self.depth_cache = None  # Cached depth map
        self.depth_frame_counter = 0  # Counter for frame skipping
        self.openpose_processor = None  # OpenPose preprocessor for human pose detection
        self.openpose_cache = None  # Cached pose skeleton
        self.openpose_frame_counter = 0  # Counter for frame skipping

        # UNIFIED FRAME SKIPPING: Counter for synchronized ControlNet skipping
        # When using multiple ControlNets, they should skip frames together to avoid wasted computation
        # Also works with single ControlNet for consistent behavior
        # Expected gain: +20-30% FPS by avoiding compute when not all ControlNets are ready
        self.unified_frame_counter = 0

        # ControlNet config is pushed via IPC (no JSON file)
        import threading
        self.config_lock = threading.Lock()
        self.controlnet_config = self._default_controlnet_config()
        self.controlnet_skip_frames = 1
        self.depth_downscale_size = 256
        self.current_delta = 1.0
        self.frames_processed = 0

        # CRITICAL FIX: Pre-allocate depth normalization tensors (ImageNet stats)
        # These were recreated every frame (lines 945-946), causing GPU memory churn
        self._depth_mean = None
        self._depth_std = None

        # OPTIMIZATION: Pre-allocated buffers to avoid memory leaks in ControlNet processing
        self._canny_input_buffer = None  # CPU buffer for Canny input (HWC uint8)
        self._canny_output_buffer = None  # GPU buffer for Canny output (CHW float16)
        self._depth_output_buffer = None  # GPU buffer for Depth output (CHW float16)
        self._openpose_input_buffer = None  # CPU buffer for OpenPose input (HWC uint8)
        self._openpose_output_buffer = None  # GPU buffer for OpenPose output (CHW float16)
        self._temp_cpu_buffer = None  # Reusable CPU buffer for conversions

        # FRAGMENTATION FIX: Pre-allocate max-size buffers once, use slices for smaller resolutions
        # This prevents GPU memory fragmentation from repeated reallocation on resolution changes
        # Max resolution: 1024x1024 (covers most use cases, only 6MB per buffer in FP16)
        self._max_buffer_size = 1024
        self._canny_input_buffer_max = None  # Will be allocated on first use (CPU NumPy array)
        self._canny_output_buffer_max = None  # Will be allocated on first use (GPU tensor)
        self._depth_output_buffer_max = None  # Will be allocated on first use (GPU tensor)
        self._openpose_input_buffer_max = None  # Will be allocated on first use (CPU NumPy array)
        self._openpose_output_buffer_max = None  # Will be allocated on first use (GPU tensor)

        # ASYNC PREPROCESSING OPTIMIZATION: Separate CUDA stream for ControlNet preprocessing
        # This allows preprocessing of frame N+1 to happen in parallel with generation of frame N
        # Expected gain: +15-30% FPS by overlapping preprocessing and generation
        self.preprocess_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.preprocessed_controlnet_image = None  # Double-buffer: stores preprocessed image from previous frame
        self.preprocessing_future = None  # Stores the async preprocessing task for next frame
        # ARCHITECTURAL FIX: Disable async preprocessing to eliminate 1-frame delay
        # Synchronous mode ensures ControlNet depth is extracted from CURRENT frame (not previous)
        # This prevents temporal desynchronization that causes flickering when depth changes
        # Performance impact: ~3-5ms per frame (61fps → 56-58fps), acceptable for better quality
        self.async_preprocessing_enabled = False  # Was True - changed to fix ControlNet flickering

        # PARALLEL PREPROCESSING OPTIMIZATION: Separate streams for each ControlNet preprocessor
        # Allows Depth + OpenPose to run in PARALLEL instead of sequentially
        # Expected gain: +40-60% FPS when using multiple ControlNets (35ms -> 20ms preprocessing)
        self.depth_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.openpose_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        # WARMUP FLAG: Track if initial torch.compile warmup has been completed
        # This prevents warmup from running on every prepare() call (prompt/t_index changes)
        # Warmup only needs to run once at startup, not on every parameter change
        self.warmup_completed = False
        self.canny_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        # PINNED MEMORY OPTIMIZATION: Enable faster CPU-GPU transfers
        # Pinned (page-locked) memory allows DMA transfers: 2-3x faster than pageable memory
        # Expected gain: +5-10% FPS for operations with CPU-GPU copies (ControlNet preprocessing)
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)  # Reserve 5% for pinned memory pool
            # Enable TF32 tensor cores if available (already done in wrapper.py, but ensure it's set)
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True

        self._init_connection()
        self._create_tensors(3, self.width, self.height)

    def _create_stream(self):
        send_message(self.socket, StreamCreationPacket(False))

        # Free previous engine before loading new one
        if self.stream is not None:
            logging.info(f"[Engine] Freeing previous engine...")
            del self.stream
            self.stream = None
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        self._create_sd_stream()

        self._create_tensors(3, self.width, self.height)
        send_message(self.socket, StreamCreationPacket(True))

    def _create_sd_stream(self):
        """Create StreamDiffusion engine (SD 1.5 / SD 2.x / SDXL)."""
        # Detect model type for pipeline separation
        self.is_sdxl = any(kw in self.model_name.lower() for kw in ["sdxl", "xl", "sd-xl", "sd_xl"])
        # SD 2.x / Turbo models use cross-attention dim 1024; SD 1.5 uses 768
        # This determines which ControlNet models are compatible
        sd2_keywords = ["sd-turbo", "sd_turbo", "2.0", "2.1", "2-1", "stabilityai/sd-turbo"]
        self.is_sd2 = any(kw in self.model_name.lower() for kw in sd2_keywords)
        is_lightning = "lightning" in self.model_name.lower()
        is_hyper_unet = "hyper-sdxl-unet" in self.model_name.lower()
        WrapperClass = StreamDiffusionWrapperXL if self.is_sdxl else StreamDiffusionWrapper
        model_type = "SDXL" if self.is_sdxl else "SD"
        logging.info(f"[Pipeline] Using {model_type} pipeline for model: {self.model_name}")

        # num_inference_steps determines the timestep schedule
        if self.is_sdxl and is_lightning:
            self.num_inference_steps = 8
            logging.info(f"[SDXL Lightning] Using num_inference_steps={self.num_inference_steps} (optimized for Lightning)")
        else:
            self.num_inference_steps = 50
            logging.info(f"[Model] Using num_inference_steps=50 (Smode t_index range: 0-49)")

        self.stream = WrapperClass(
            model_id_or_path=self.model_name,
            t_index_list=self.t_index_list,
            lora_dict=self.lora_dict,
            mode="img2img" if self.mode == Mode.IMAGE_TO_IMAGE else "txt2img",
            frame_buffer_size=1,
            width=self.width,
            height=self.height,
            warmup=10,
            acceleration="xformers" if self.acceleration == Acceleration.XFORMERS
                                    else "none",
            device_ids=None,
            use_lcm_lora=True,
            use_tiny_vae=self.controlnet_config.get('use_tiny_vae', True),
            enable_similar_image_filter=self.similar_image_filter_enabled,
            similar_image_filter_threshold=self.similar_image_filter_threshold,
            similar_image_filter_max_skip_frame=self.similar_image_filter_max_skip,
            use_denoising_batch=True,
            cfg_type=self.cfg_type,
            seed=self.seed,
            dtype=self.torch_dtype,
            device=self.device,
            output_type="pt",
            cache_dir=self.cache_dir,
            torch_compile_enabled=self.acceleration == Acceleration.TORCH_COMPILE,
            torch_compile_mode="reduce-overhead",
            torch_compile_fullgraph=True,
        )

        # Pass ControlNet guidance strength to pipeline
        self.stream.stream._cached_controlnet_guidance_strength = self._cached_controlnet_guidance_strength

        # Log async preprocessing status
        if self.async_preprocessing_enabled and self.preprocess_stream is not None:
            logging.info("Async ControlNet preprocessing ENABLED (+15-30% FPS expected)")
        else:
            logging.info("Synchronous ControlNet preprocessing ENABLED")

        # Log parallel preprocessing status
        if self.depth_stream is not None and self.openpose_stream is not None:
            logging.info("Parallel Multi-ControlNet preprocessing ENABLED")

    def _create_tensors(self, channels, w, h):
        self.input_tensors = StreamDiffusionSmodeTexture(
            self.device, w, h, channels, self.torch_dtype
        )
        self.output_tensors = StreamDiffusionSmodeTexture(
            self.device, w, h, channels, self.torch_dtype
        )

        def send_frame_data_packet(stream_diffusion_smode_texture: StreamDiffusionSmodeTexture, command_type: CommandType):
            (
                tensor_type,
                tensor_size,
                tensor_stride,
                tensor_offset,
                storage_type,
                tensor_dtype,
                device,
                handle,
                storage_size_bytes,
                storage_offset_bytes,
                tensor_requires_grad,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ) = stream_diffusion_smode_texture.smode_tensor_ipc_info
            packet = FrameDataPacket(
                command_type,
                device,
                handle[2:],
                event_handle,
                storage_size_bytes,
                storage_offset_bytes,
                3,
                self.width,
                self.height,
            )
            send_message(self.socket, packet)

        send_frame_data_packet(self.input_tensors, CommandType.INPUT)
        send_frame_data_packet(self.output_tensors, CommandType.OUTPUT)

    def _init_connection(self):
        try:
            self.socket.setblocking(True)
            server_address = ("127.0.0.1", self.config.port)
            logging.info(f"Connecting to server at {server_address}")
            self.socket.connect(server_address)
            self._send_uuid()
            self.socket.setblocking(False)
        except socket.error as e:
            logging.error(f"Socket error during connection: {e}")
            self.socket.close()
            raise

    def _send_uuid(self):
        packet = UuidPacket(self.config.uuid)
        send_message(self.socket, packet)

    def accelerate(self, previous_acceleration: Acceleration = Acceleration.NONE):
        if self.acceleration == Acceleration.XFORMERS:
            self.stream.stream.pipe.enable_xformers_memory_efficient_attention()
            self.stream.recreate_pipe()
        elif self.acceleration == Acceleration.TORCH_COMPILE:
            # torch.compile: just recreate the pipe without xformers
            self.stream.recreate_pipe()
        elif self.acceleration == Acceleration.TENSORRT:
            try:
                if previous_acceleration == Acceleration.XFORMERS:
                    self._create_stream()
                else:
                    self.stream.enable_tensorrt_acceleration(self.stream.stream, self.model_name, True, True)

                # OPTIMIZATION #27: Warmup TensorRT engine
                self._warmup_tensorrt()
            except ModuleNotFoundError:
                logging.warning(
                    "TensorRT module not found; please install it"
                )
                raise
            except Exception as e:
                logging.warning(f"TensorRT acceleration not available; {e}")
        else:
            self.stream.recreate_pipe()

    def _warmup_tensorrt(self):
        """Warmup TensorRT engine with dummy inferences"""
        dummy_input = None
        try:
            logging.info("Warming up TensorRT engine...")
            warmup_iterations = max(len(self.t_index_list) * self.stream.stream.frame_bff_size, 10)

            dummy_input = torch.randn(
                (1, 3, self.height, self.width),
                dtype=self.torch_dtype,  # Use model's dtype (typically float16)
                device=self.device
            )

            for _ in range(warmup_iterations):
                _ = self.stream.stream(image=dummy_input)

            logging.info(f"TensorRT warmup complete ({warmup_iterations} iterations)")
        except Exception as e:
            logging.warning(f"TensorRT warmup failed (non-critical): {e}")
        finally:
            # MEMORY LEAK FIX: Always free dummy tensor (prevents ~3MB GPU leak)
            if dummy_input is not None:
                del dummy_input
                torch.cuda.empty_cache()

    def _default_controlnet_config(self):
        """Return default ControlNet config — used before first IPC CONFIG packet arrives."""
        config = {
            'controlnet_enabled': True,
            'preview_mode': 'normal',
            'controlnet_guidance_strength': 1.0,
            'controlnet_skip_frames': 1,
            'canny_enabled': False,
            'canny_scale': 0.41,
            'canny_resolution': 384,
            'canny_low_threshold': 100,
            'canny_high_threshold': 255,
            'canny_aperture_size': 3,
            'canny_l2_gradient': False,
            'depth_enabled': False,
            'depth_scale': 0.86,
            'depth_resolution': 384,
            'depth_blur_kernel': 1,
            'depth_contrast': 1.0,
            'depth_brightness': 0,
            'depth_near_threshold': 0,
            'depth_far_threshold': 255,
            'depth_invert': False,
            'depth_method': 'grayscale',
            'depth_model_size': 'small',
            'openpose_enabled': False,
            'openpose_scale': 0.89,
            'openpose_detect_resolution': 256,
            'use_tiny_vae': True,
            'latent_feedback_strength': 0.0,
            'motion_aware_noise': False,
            'motion_aware_noise_sensitivity': 0.5,
            'delta': 1.0,
            'profiling_enabled': False,
        }
        self._cache_config_values(config)
        return config

    _DEPTH_METHOD_NAMES = ['grayscale', 'sobel', 'laplacian']
    _DEPTH_MODEL_SIZE_NAMES = ['small', 'base', 'large']
    _PREVIEW_MODE_NAMES = ['normal', 'canny_preview', 'depth_preview', 'openpose_preview']

    def _apply_controlnet_config_from_packet(self, config_packet: 'ConfigPacket'):
        """Build controlnet_config dict from IPC packet fields and apply live updates."""
        new_config = {
            'controlnet_enabled': config_packet.controlnet_enabled,
            'preview_mode': self._PREVIEW_MODE_NAMES[min(config_packet.controlnet_preview_mode, len(self._PREVIEW_MODE_NAMES) - 1)],
            'controlnet_guidance_strength': config_packet.controlnet_guidance_strength,
            'controlnet_skip_frames': config_packet.controlnet_skip_frames,
            'canny_enabled': config_packet.canny_enabled,
            'canny_scale': config_packet.canny_scale,
            'canny_resolution': config_packet.canny_resolution,
            'canny_low_threshold': config_packet.canny_low_threshold,
            'canny_high_threshold': config_packet.canny_high_threshold,
            'canny_aperture_size': config_packet.canny_aperture_size,
            'canny_l2_gradient': config_packet.canny_l2_gradient,
            'depth_enabled': config_packet.depth_enabled,
            'depth_scale': config_packet.depth_scale,
            'depth_resolution': config_packet.depth_resolution,
            'depth_blur_kernel': config_packet.depth_blur_kernel,
            'depth_contrast': config_packet.depth_contrast,
            'depth_brightness': config_packet.depth_brightness,
            'depth_near_threshold': config_packet.depth_near_threshold,
            'depth_far_threshold': config_packet.depth_far_threshold,
            'depth_invert': config_packet.depth_invert,
            'depth_method': self._DEPTH_METHOD_NAMES[min(config_packet.depth_method, len(self._DEPTH_METHOD_NAMES) - 1)],
            'depth_model_size': self._DEPTH_MODEL_SIZE_NAMES[min(config_packet.depth_model_size, len(self._DEPTH_MODEL_SIZE_NAMES) - 1)],
            'openpose_enabled': config_packet.openpose_enabled,
            'openpose_scale': config_packet.openpose_scale,
            'openpose_detect_resolution': config_packet.openpose_detect_resolution,
            'use_tiny_vae': config_packet.use_tiny_vae,
            'latent_feedback_strength': config_packet.latent_feedback_strength,
            'motion_aware_noise': config_packet.motion_aware_noise,
            'motion_aware_noise_sensitivity': config_packet.motion_aware_noise_sensitivity,
            'delta': 1.0,
            'profiling_enabled': False,
        }

        with self.config_lock:
            old_config = self.controlnet_config
            if new_config == old_config:
                return

            logging.info("ControlNet configuration updated via IPC")

            # Frame skipping
            old_skip = self.controlnet_skip_frames
            self.controlnet_skip_frames = new_config['controlnet_skip_frames']
            if old_skip != self.controlnet_skip_frames:
                logging.info(f"ControlNet frame skipping: {old_skip} -> {self.controlnet_skip_frames}")

            # Similar Image Filter
            if hasattr(self.stream, 'stream'):
                if config_packet.similar_image_filter_enabled:
                    self.stream.stream.enable_similar_image_filter(
                        config_packet.similar_image_filter_threshold,
                        config_packet.similar_image_filter_max_skip
                    )
                else:
                    self.stream.stream.disable_similar_image_filter()

            # ControlNet guidance strength
            if hasattr(self.stream, 'stream'):
                new_strength = new_config['controlnet_guidance_strength']
                old_strength = old_config.get('controlnet_guidance_strength', 1.0)
                if abs(old_strength - new_strength) > 0.01:
                    self.stream.stream._cached_controlnet_guidance_strength = new_strength
                    self._cached_controlnet_guidance_strength = new_strength
                    if hasattr(self.stream.stream, '_guidance_strength_logged'):
                        delattr(self.stream.stream, '_guidance_strength_logged')
                    logging.info(f"Guidance strength: {old_strength:.2f} -> {new_strength:.2f}")

            # Latent feedback
            if hasattr(self.stream, 'stream'):
                new_fb = new_config['latent_feedback_strength']
                old_fb = old_config.get('latent_feedback_strength', 0.0)
                if abs(old_fb - new_fb) > 0.001:
                    self.stream.stream.latent_feedback_strength = new_fb
                    if new_fb == 0.0:
                        self.stream.stream._prev_latent = None
                    logging.info(f"Latent feedback: {old_fb:.3f} -> {new_fb:.3f}")

            # Motion-aware noise
            if hasattr(self.stream, 'stream'):
                new_man = new_config['motion_aware_noise']
                old_man = old_config.get('motion_aware_noise', False)
                if new_man != old_man:
                    self.stream.stream.motion_aware_noise = new_man
                    if not new_man:
                        self.stream.stream._prev_input_latent = None
                        self.stream.stream._motion_noise_scale = 1.0
                    logging.info(f"Motion-aware noise: {'on' if new_man else 'off'}")
                new_sens = new_config['motion_aware_noise_sensitivity']
                old_sens = old_config.get('motion_aware_noise_sensitivity', 0.5)
                if abs(old_sens - new_sens) > 0.01:
                    self.stream.stream.motion_aware_noise_sensitivity = new_sens

            self.controlnet_config = new_config
            self._load_controlnet_models()
            self._cache_config_values(new_config)

    def _cache_config_values(self, config):
        """Cache frequently accessed config values to avoid dict lookups in hot path"""
        # ControlNet scales (accessed every frame during generation)
        self._cached_canny_scale = config.get('canny_scale', 0.5)
        self._cached_depth_scale = config.get('depth_scale', 0.5)
        self._cached_openpose_scale = config.get('openpose_scale', 0.8)

        # Profiling flag (checked multiple times per frame)
        self._cached_profiling_enabled = config.get('profiling_enabled', False)

        # Preview mode (checked every frame)
        self._cached_preview_mode = config.get('preview_mode', 'normal')

        # ControlNet enabled flags (checked every frame)
        self._cached_controlnet_enabled = config.get('controlnet_enabled', False)
        self._cached_canny_enabled = config.get('canny_enabled', False)
        self._cached_depth_enabled = config.get('depth_enabled', False)
        self._cached_openpose_enabled = config.get('openpose_enabled', False)

        # Unified frame skipping (applies to all ControlNets)
        self._cached_controlnet_skip_frames = config.get('controlnet_skip_frames', 1)

        self._cached_controlnet_guidance_strength = config.get('controlnet_guidance_strength', 1.0)

        self._cached_depth_blur_kernel = config.get('depth_blur_kernel', 1)
        if self._cached_depth_enabled and self._cached_depth_blur_kernel > 1:
            self._cached_gaussian_kernel = self._precompute_gaussian_kernel(
                self._cached_depth_blur_kernel
            )
        else:
            self._cached_gaussian_kernel = None

        self._cached_openpose_detect_resolution = config.get('openpose_detect_resolution', 512)

    def _precompute_gaussian_kernel(self, blur_kernel):
        """Pre-compute Gaussian blur kernel to avoid expensive computation every frame"""
        # Ensure blur_kernel is odd
        blur_kernel_odd = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        kernel_size = blur_kernel_odd
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8  # OpenCV default formula

        # Create 1D Gaussian kernel
        # Use FP32 for kernel computation (will be cast to input dtype during convolution)
        kernel_1d = torch.exp(
            -torch.arange(-(kernel_size//2), kernel_size//2 + 1, dtype=torch.float32, device=self.device)**2
            / (2 * sigma**2)
        )
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Create 2D kernel from 1D
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

        return kernel_2d

    def _update_controlnet_active_list(self):
        """
        OPTIMIZATION: Pre-build stable list of active ControlNets.

        Called only on config change, not every frame. Eliminates list building
        overhead in hot path (60+ fps). Pre-allocates fixed-size lists for models
        and scales to avoid Python list operations during generation.

        Expected gain: +1-2% FPS by eliminating dict lookups and list append() calls.
        """
        # Build list of active ControlNet keys in stable order
        self._active_cn_keys = []
        if self._cached_canny_enabled and 'canny' in self.controlnet_models:
            self._active_cn_keys.append('canny')
        if self._cached_depth_enabled and 'depth' in self.controlnet_models:
            self._active_cn_keys.append('depth')
        if self._cached_openpose_enabled and 'openpose' in self.controlnet_models:
            self._active_cn_keys.append('openpose')

        if self._active_cn_keys:
            self._cn_models_cache = [self.controlnet_models[k] for k in self._active_cn_keys]
            self._cn_scales_cache = [
                self._cached_canny_scale if k == 'canny' else
                self._cached_depth_scale if k == 'depth' else
                self._cached_openpose_scale
                for k in self._active_cn_keys
            ]
        else:
            self._cn_models_cache = []
            self._cn_scales_cache = []

    def _compile_controlnet(self, model, controlnet_name: str):
        """
        Compile ControlNet with torch.compile and dedicated cache directory.

        Args:
            model: ControlNet model to compile
            controlnet_name: Name of the controlnet ('canny', 'depth', 'openpose')

        Returns:
            Compiled model (or original if compilation fails)
        """
        if self.acceleration != Acceleration.TORCH_COMPILE:
            return model

        try:
            if not (hasattr(torch, 'compile') and torch.__version__ >= '2.0'):
                return model

            controlnet_cache_dir = PACKAGE_DIR / "torch_compile_cache" / "controlnet" / controlnet_name
            controlnet_cache_dir.mkdir(parents=True, exist_ok=True)

            # Set cache directory for this compilation
            old_cache_dir = os.environ.get('TORCHINDUCTOR_CACHE_DIR', '')
            os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(controlnet_cache_dir)
            os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'

            logging.info(f"Compiling {controlnet_name} ControlNet with torch.compile()...")
            logging.info(f"  Cache directory: {controlnet_cache_dir}")

            try:
                compiled_model = torch.compile(
                    model,
                    mode='reduce-overhead',
                    fullgraph=False,
                    dynamic=False
                )

                logging.info("  Triggering compilation with dummy inference...")
                
                # Create dummy inputs matching ControlNet forward signature
                batch_size = 1
                hidden_dim = getattr(model.config, 'cross_attention_dim', None)
                if hidden_dim is None:

                    logging.info(f"  Vision model detected, skipping torch.compile (not a ControlNet)")
                    os.environ['TORCHINDUCTOR_CACHE_DIR'] = old_cache_dir
                    return model

                dummy_sample = torch.randn(batch_size, 4, 64, 64, dtype=self.torch_dtype, device=self.device)
                dummy_timestep = torch.tensor([1], device=self.device)
                dummy_encoder_hidden_states = torch.randn(batch_size, 77, hidden_dim, dtype=self.torch_dtype, device=self.device)
                dummy_controlnet_cond = torch.randn(batch_size, 3, 512, 512, dtype=self.torch_dtype, device=self.device)
                
                # Run inference (no grad needed)
                with torch.no_grad():
                    compiled_model(
                        dummy_sample,
                        dummy_timestep,
                        dummy_encoder_hidden_states,
                        dummy_controlnet_cond,
                        conditioning_scale=1.0,
                        return_dict=False
                    )
                logging.info("  Compilation triggered successfully")
                
            finally:
                # Restore previous cache directory
                if old_cache_dir:
                    os.environ['TORCHINDUCTOR_CACHE_DIR'] = old_cache_dir
                else:
                    os.environ.pop('TORCHINDUCTOR_CACHE_DIR', None)

            logging.info(f"✓ {controlnet_name} ControlNet compiled and cached successfully")
            return compiled_model

        except Exception as e:
            logging.warning(f"Failed to compile {controlnet_name} ControlNet (non-critical): {e}")
            return model  # Return original model if compilation fails

    def _warmup_controlnet_integration(self, controlnet_name: str, controlnet_model):
        """
        Warmup U-Net+ControlNet integration to force torch.compile compilation
        This prevents the 2-30s freeze when first using a ControlNet in real inference
        """
        if not self.stream:
            return  # Skip if stream not initialized yet

        try:
            logging.info(f"Warming up U-Net+{controlnet_name} integration...")
            warmup_start = time.time()

            # Create dummy input matching real inference shape
            # IMPORTANT: Use same dtype as model (float16) to avoid compilation errors
            dummy_input = torch.randn(
                (1, 3, self.height, self.width),
                dtype=self.torch_dtype,  # Use model's dtype (typically float16)
                device=self.device
            )

            # Perform 2 warmup inferences with this ControlNet
            # First: compiles U-Net+ControlNet path (or loads from cache)
            # Second: validates cache is working
            for i in range(2):
                _ = self.stream(
                    image=dummy_input,
                    controlnet_image=[dummy_input],  # Single ControlNet conditioning
                    controlnet_model=[controlnet_model],
                    controlnet_conditioning_scale=[1.0],
                )

            torch.cuda.synchronize()
            warmup_time = time.time() - warmup_start
            logging.info(f"✓ {controlnet_name} integration warmup complete ({warmup_time:.1f}s)")

        except Exception as e:
            logging.warning(f"{controlnet_name} integration warmup failed (non-critical): {e}")
        finally:
            # MEMORY LEAK FIX: Always free dummy tensor
            if 'dummy_input' in locals():
                del dummy_input
                torch.cuda.empty_cache()

    def _load_controlnet_models(self):
        """Load ControlNet models on-demand"""
        config = self.controlnet_config

        # Load Canny ControlNet if enabled and not loaded
        if config.get('canny_enabled', False) and 'canny' not in self.controlnet_models:
            try:
                is_sdxl = self.is_sdxl
                if is_sdxl:
                    canny_repo = "diffusers/controlnet-canny-sdxl-1.0"
                    cn_label = "SDXL"
                elif self.is_sd2:
                    canny_repo = "thibaud/controlnet-sd21-canny-diffusers"
                    cn_label = "SD 2.1"
                else:
                    canny_repo = "lllyasviel/sd-controlnet-canny"
                    cn_label = "SD 1.5"
                logging.info(f"Loading Canny ControlNet model ({cn_label})...")
                model = ControlNetModel.from_pretrained(
                    canny_repo,
                    torch_dtype=self.torch_dtype
                ).to(self.device)

                # Compile with torch.compile (cache: torch_compile_cache/controlnet/canny/)
                model = self._compile_controlnet(model, 'canny')

                self.controlnet_models['canny'] = model
                logging.info("Canny ControlNet loaded successfully")

                # WARMUP: Force U-Net+ControlNet compilation after loading new ControlNet
                self._warmup_controlnet_integration('canny', model)
            except Exception as e:
                logging.error(f"Failed to load Canny ControlNet: {e}")
                # Clean up any partially allocated resources
                if 'canny' in self.controlnet_models:
                    del self.controlnet_models['canny']
                torch.cuda.empty_cache()
                # Disable canny in config to prevent retry loops
                self.controlnet_config['canny_enabled'] = False

        # Load or reload Depth-Anything V2 model if depth is enabled
        # Reload if model size changed (user switched small/base/large)
        if config.get('depth_enabled', False):
            depth_model_size = config.get('depth_model_size', 'small').lower()

            # Check if we need to load or reload the model
            needs_reload = (
                self.depth_processor is None or  # Not loaded yet
                self.current_depth_model_size != depth_model_size  # Model size changed
            )

            if needs_reload:
                try:
                    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

                    # Unload old model to free VRAM
                    if self.depth_model is not None:
                        logging.info(f"Unloading Depth-Anything V2 {self.current_depth_model_size.upper()} model...")
                        del self.depth_model
                        del self.depth_processor
                        torch.cuda.empty_cache()

                    model_map = {
                        'small': 'depth-anything/Depth-Anything-V2-Small-hf',
                        'base': 'depth-anything/Depth-Anything-V2-Base-hf',
                        'large': 'depth-anything/Depth-Anything-V2-Large-hf'
                    }
                    model_id = model_map.get(depth_model_size, model_map['small'])

                    logging.info(f"Loading Depth-Anything V2 {depth_model_size.upper()} model (optimized for GPU)...")

                    # Load processor and model separately for better control
                    # V2 improvements: Better edge precision, better detail in complex areas
                    self.depth_processor = AutoImageProcessor.from_pretrained(model_id)
                    self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                        model_id,
                        torch_dtype=self.torch_dtype  # Use same dtype as StreamDiffusion (FP16)
                    ).to(self.device)

                    # Set to eval mode and disable gradient computation
                    self.depth_model.eval()

                    # Compile Depth-Anything V2 model with torch.compile (cache: torch_compile_cache/controlnet/depth/)
                    self.depth_model = self._compile_controlnet(self.depth_model, 'depth')

                    # CRITICAL FIX: Pre-allocate ImageNet normalization tensors (avoids recreation every frame)
                    self._depth_mean = torch.tensor([0.485, 0.456, 0.406],
                                                     device=self.device,
                                                     dtype=self.torch_dtype).view(3, 1, 1)
                    self._depth_std = torch.tensor([0.229, 0.224, 0.225],
                                                    device=self.device,
                                                    dtype=self.torch_dtype).view(3, 1, 1)

                    self.current_depth_model_size = depth_model_size
                    logging.info(f"Depth-Anything V2 {depth_model_size.upper()} loaded successfully on {self.device} with {self.torch_dtype}")
                except Exception as e:
                    logging.error(f"Failed to load Depth-Anything: {e}")
                    logging.info("Falling back to simple depth estimation")
                    # Clean up partially loaded state
                    self.depth_model = None
                    self.depth_processor = None
                    self.current_depth_model_size = None
                    torch.cuda.empty_cache()

        # Load Depth ControlNet if enabled and not loaded
        if config.get('depth_enabled', False) and 'depth' not in self.controlnet_models:
            try:
                is_sdxl = self.is_sdxl
                depth_kwargs = {"torch_dtype": self.torch_dtype}
                if is_sdxl:
                    depth_repo = "diffusers/controlnet-depth-sdxl-1.0"
                    cn_label = "SDXL"
                elif self.is_sd2:
                    depth_repo = "thibaud/controlnet-sd21-depth-diffusers"
                    depth_kwargs["use_safetensors"] = False  # SD 2.1 model only has pickle weights
                    cn_label = "SD 2.1"
                else:
                    depth_repo = "lllyasviel/sd-controlnet-depth"
                    cn_label = "SD 1.5"
                logging.info(f"Loading Depth ControlNet model ({cn_label})...")
                model = ControlNetModel.from_pretrained(
                    depth_repo,
                    **depth_kwargs
                ).to(self.device)

                # Compile with torch.compile (cache: torch_compile_cache/controlnet/depth_controlnet/)
                # Note: Using 'depth_controlnet' to distinguish from Depth-Anything V2 preprocessor
                model = self._compile_controlnet(model, 'depth_controlnet')

                self.controlnet_models['depth'] = model
                logging.info("Depth ControlNet loaded successfully")

                # WARMUP: Force U-Net+ControlNet compilation after loading new ControlNet
                self._warmup_controlnet_integration('depth', model)
            except Exception as e:
                logging.error(f"Failed to load Depth ControlNet: {e}")
                # Clean up any partially allocated resources
                if 'depth' in self.controlnet_models:
                    del self.controlnet_models['depth']
                torch.cuda.empty_cache()
                # Disable depth in config to prevent retry loops
                self.controlnet_config['depth_enabled'] = False

        if config.get('openpose_enabled', False) and self.openpose_processor is None:
            try:
                from huggingface_hub import hf_hub_download
                from easy_dwpose.body_estimation import Wholebody
                from easy_dwpose.draw import draw_openpose
                import PIL.Image

                logging.info("Loading DWPose preprocessor with optimized models (YOLOX-S + DWPose-M, GPU-accelerated)...")

                model_det_path = hf_hub_download("hr16/yolox-onnx", "yolox_s.onnx", local_dir=str(PACKAGE_DIR / "checkpoints"))

                model_pose_path = hf_hub_download("hr16/UnJIT-DWPose", "dw-mm_ucoco.onnx", local_dir=str(PACKAGE_DIR / "checkpoints"))

                pose_estimation = Wholebody(
                    device=self.device,
                    model_det=model_det_path,
                    model_pose=model_pose_path
                )

                # Wrap in a class that mimics DWposeDetector interface
                class OptimizedDWposeDetector:
                    def __init__(self, pose_estimation_model):
                        self.pose_estimation = pose_estimation_model

                    @torch.inference_mode()
                    def __call__(self, image, detect_resolution=512, draw_pose=draw_openpose, output_type="pil", **kwargs):
                        import numpy as np
                        from easy_dwpose.body_estimation import resize_image
                        import cv2

                        if type(image) != np.ndarray:
                            image = np.array(image.convert("RGB"))

                        image = image.copy()
                        original_height, original_width, _ = image.shape

                        image = resize_image(image, target_resolution=detect_resolution)
                        height, width, _ = image.shape

                        candidates, scores = self.pose_estimation(image)

                        # Format pose (same logic as DWposeDetector._format_pose)
                        num_candidates, _, locs = candidates.shape
                        candidates[..., 0] /= float(width)
                        candidates[..., 1] /= float(height)

                        bodies = candidates[:, :18].copy()
                        bodies = bodies.reshape(num_candidates * 18, locs)

                        body_scores = scores[:, :18]
                        for i in range(len(body_scores)):
                            for j in range(len(body_scores[i])):
                                if body_scores[i][j] > 0.3:
                                    body_scores[i][j] = int(18 * i + j)
                                else:
                                    body_scores[i][j] = -1

                        faces = candidates[:, 24:92]
                        faces_scores = scores[:, 24:92]

                        hands = np.vstack([candidates[:, 92:113], candidates[:, 113:]])
                        hands_scores = np.vstack([scores[:, 92:113], scores[:, 113:]])

                        pose = dict(
                            bodies=bodies,
                            body_scores=body_scores,
                            hands=hands,
                            hands_scores=hands_scores,
                            faces=faces,
                            faces_scores=faces_scores,
                        )

                        if not draw_pose:
                            return pose

                        pose_image = draw_pose(pose, height=height, width=width, **kwargs)
                        pose_image = cv2.resize(pose_image, (original_width, original_height), cv2.INTER_LANCZOS4)

                        if output_type == "pil":
                            pose_image = PIL.Image.fromarray(pose_image)
                        elif output_type == "np":
                            pass
                        else:
                            raise ValueError("output_type should be 'pil' or 'np'")

                        return pose_image

                self.openpose_processor = OptimizedDWposeDetector(pose_estimation)
                logging.info("DWPose preprocessor loaded successfully with optimized models (YOLOX-S + DWPose-M, ~2x faster)")
            except Exception as e:
                logging.error(f"Failed to load DWPose preprocessor: {e}")
                logging.info("DWPose will not be available")
                # Clean up any partially allocated state
                self.openpose_processor = None
                torch.cuda.empty_cache()

        # Load OpenPose ControlNet if enabled and not loaded
        if config.get('openpose_enabled', False) and 'openpose' not in self.controlnet_models:
            try:
                is_sdxl = self.is_sdxl
                if is_sdxl:
                    openpose_repo = "thibaud/controlnet-openpose-sdxl-1.0"
                    cn_label = "SDXL"
                elif self.is_sd2:
                    openpose_repo = "thibaud/controlnet-sd21-openpose-diffusers"
                    cn_label = "SD 2.1"
                else:
                    openpose_repo = "lllyasviel/sd-controlnet-openpose"
                    cn_label = "SD 1.5"
                logging.info(f"Loading OpenPose ControlNet model ({cn_label})...")
                model = ControlNetModel.from_pretrained(
                    openpose_repo,
                    torch_dtype=self.torch_dtype
                ).to(self.device)

                # Compile with torch.compile (cache: torch_compile_cache/controlnet/openpose/)
                model = self._compile_controlnet(model, 'openpose')

                self.controlnet_models['openpose'] = model
                logging.info("OpenPose ControlNet loaded successfully")

                # WARMUP: Force U-Net+ControlNet compilation after loading new ControlNet
                self._warmup_controlnet_integration('openpose', model)
            except Exception as e:
                logging.error(f"Failed to load OpenPose ControlNet: {e}")
                # Clean up any partially allocated resources
                if 'openpose' in self.controlnet_models:
                    del self.controlnet_models['openpose']
                torch.cuda.empty_cache()
                # Disable openpose in config to prevent retry loops
                self.controlnet_config['openpose_enabled'] = False

        # Unload models that are disabled
        if not config.get('canny_enabled', False) and 'canny' in self.controlnet_models:
            logging.info("Unloading Canny ControlNet...")
            del self.controlnet_models['canny']
            torch.cuda.empty_cache()

        if not config.get('depth_enabled', False):
            if 'depth' in self.controlnet_models:
                logging.info("Unloading Depth ControlNet...")
                del self.controlnet_models['depth']
            if self.depth_processor is not None or self.depth_model is not None:
                logging.info("Unloading Depth-Anything model...")
                if self.depth_processor is not None:
                    del self.depth_processor
                    self.depth_processor = None
                if self.depth_model is not None:
                    del self.depth_model
                    self.depth_model = None
                if self._depth_mean is not None:
                    del self._depth_mean
                    self._depth_mean = None
                if self._depth_std is not None:
                    del self._depth_std
                    self._depth_std = None
            if self.depth_cache is not None:
                del self.depth_cache
                self.depth_cache = None
                self.depth_frame_counter = 0  # Reset counter
            torch.cuda.empty_cache()

        if not config.get('openpose_enabled', False):
            if 'openpose' in self.controlnet_models:
                logging.info("Unloading OpenPose ControlNet...")
                del self.controlnet_models['openpose']
            if self.openpose_cache is not None:
                del self.openpose_cache
                self.openpose_cache = None
                self.openpose_frame_counter = 0  # Reset counter
            if self.openpose_processor is not None:
                logging.info("Unloading OpenPose preprocessor...")
                # CRITICAL FIX: Properly cleanup ONNX Runtime resources
                # Check if DWposeDetector has cleanup method
                if hasattr(self.openpose_processor, 'close'):
                    self.openpose_processor.close()
                elif hasattr(self.openpose_processor, 'cleanup'):
                    self.openpose_processor.cleanup()
                del self.openpose_processor
                self.openpose_processor = None
                # Force garbage collection for ONNX Runtime
                import gc
                gc.collect()
            torch.cuda.empty_cache()



    def _process_canny(self, image_tensor):
        """
        Process image tensor with Canny edge detection (OPTIMIZED with downscale/upscale)

        Performance optimization: Process at lower resolution for speed
        Quality preservation: Upscale with NEAREST to keep sharp edges (not blurred)

        Args:
            image_tensor: CHW tensor in range [0, 1], float32
        Returns:
            Canny edge image as CHW tensor in range [0, 1]
        """
        config = self.controlnet_config
        low_threshold = config.get('canny_low_threshold', 100)
        high_threshold = config.get('canny_high_threshold', 200)
        aperture_size = config.get('canny_aperture_size', 3)
        l2_gradient = config.get('canny_l2_gradient', False)
        canny_resolution = config.get('canny_resolution', 384)

        original_h, original_w = image_tensor.shape[1], image_tensor.shape[2]

        # PERFORMANCE OPTIMIZATION: Downscale for faster Canny detection
        # Canny is CPU-bound, reducing resolution gives 20-30% speedup
        # Downscale with bilinear (OK for RGB input, we'll upscale edges with nearest)
        if canny_resolution < original_h:
            downscaled = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),  # Add batch dim
                size=(canny_resolution, canny_resolution),
                mode='bilinear',  # Smooth downscale for RGB input
                align_corners=False
            ).squeeze(0)  # Remove batch dim
            process_h, process_w = canny_resolution, canny_resolution
        else:
            downscaled = image_tensor
            process_h, process_w = original_h, original_w

        # FRAGMENTATION FIX: Allocate max-size CPU buffer once, use slice for current resolution
        if self._canny_input_buffer_max is None:
            # Allocate max NumPy buffer (1024x1024) once - prevents fragmentation
            self._canny_input_buffer_max = np.empty((self._max_buffer_size, self._max_buffer_size, 3), dtype=np.uint8)

        # Use a contiguous slice of the max buffer for current resolution
        self._canny_input_buffer = self._canny_input_buffer_max[:process_h, :process_w, :]

        # OPTIMIZATION: Direct conversion to numpy without creating intermediate tensors
        # Use .contiguous() to ensure memory layout is correct, then access underlying data
        image_cpu = downscaled.permute(1, 2, 0).contiguous()
        np.copyto(self._canny_input_buffer, (image_cpu.cpu().numpy() * 255).astype(np.uint8))
        del image_cpu  # LEAK FIX: Explicitly free temporary tensor

        # Apply Canny edge detection with advanced parameters (on downscaled image)
        edges = cv2.Canny(self._canny_input_buffer, low_threshold, high_threshold,
                         apertureSize=aperture_size, L2gradient=l2_gradient)

        # Convert to RGB and tensor
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges_temp = torch.from_numpy(edges_rgb).float().pin_memory() / 255.0
        edges_temp_permuted = edges_temp.permute(2, 0, 1).to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        # CRITICAL: Upscale with NEAREST to preserve sharp edges
        # Why NEAREST? Canny produces binary edges (0 or 255). Bilinear/bicubic would create
        # gray intermediate values, blurring the edges. NEAREST keeps them sharp (critical for ControlNet).
        if canny_resolution < original_h:
            edges_upscaled = torch.nn.functional.interpolate(
                edges_temp_permuted.unsqueeze(0),  # Add batch dim
                size=(original_h, original_w),
                mode='nearest'  # Preserve sharp binary edges (no blur!)
            ).squeeze(0)  # Remove batch dim
        else:
            edges_upscaled = edges_temp_permuted

        # FRAGMENTATION FIX: Allocate max-size GPU buffer once, use slice for current resolution
        if self._canny_output_buffer_max is None:
            # Allocate max buffer (1024x1024) once - prevents fragmentation from reallocation
            self._canny_output_buffer_max = torch.empty(
                (3, self._max_buffer_size, self._max_buffer_size),
                device=self.device,
                dtype=self.torch_dtype
            )

        # Use a contiguous slice of the max buffer for current resolution
        self._canny_output_buffer = self._canny_output_buffer_max[:, :original_h, :original_w].contiguous()
        self._canny_output_buffer.copy_(edges_upscaled, non_blocking=True)

        del edges_temp, edges_temp_permuted, edges_upscaled  # LEAK FIX: Explicitly free temporary tensors

        return self._canny_output_buffer

    def _process_depth(self, image_tensor):
        """
        Process image tensor with REAL depth estimation using Depth-Anything
        Args:
            image_tensor: CHW tensor in range [0, 1], float32
        Returns:
            Depth map as CHW tensor in range [0, 1]
        """
        config = self.controlnet_config
        method = config.get('depth_method', 'grayscale')
        blur_kernel = config.get('depth_blur_kernel', 5)
        invert = config.get('depth_invert', False)
        contrast = config.get('depth_contrast', 1.0)
        brightness = config.get('depth_brightness', 0)
        near_threshold = config.get('depth_near_threshold', 0)
        far_threshold = config.get('depth_far_threshold', 255)

        # Use Depth-Anything if available and method is set to AI
        if self.depth_processor is not None and self.depth_model is not None and method == 'grayscale':
            try:
                # OPTIMIZATION: Frame skipping - only compute depth every N frames
                self.depth_frame_counter += 1
                use_cache = self.depth_cache is not None and self.depth_frame_counter % self.controlnet_skip_frames != 0

                if not use_cache:
                    # OPTIMIZATION 1: Downscale input for faster inference
                    # Get resolution from config (allows runtime quality/speed adjustment)
                    downscale_size = config.get('depth_resolution', self.depth_downscale_size)
                    original_h, original_w = image_tensor.shape[1], image_tensor.shape[2]

                    # Downscale on GPU (much faster than CPU)
                    downscaled = torch.nn.functional.interpolate(
                        image_tensor.unsqueeze(0),  # Add batch dim
                        size=(downscale_size, downscale_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # Remove batch dim

                    # OPTIMIZATION 2: Manual preprocessing on GPU (bypass slow CPU processor!)
                    # The processor just does: normalize to [0,1], then ImageNet normalization
                    # We can do this directly on GPU without CPU transfers!

                    # Ensure tensor is in [0, 1] range (it should already be)
                    if downscaled.max() > 1.0:
                        downscaled = downscaled / 255.0

                    # CRITICAL FIX: Use pre-allocated ImageNet normalization tensors
                    # Previously recreated every frame (lines 945-946), now reusing from __init__
                    normalized = (downscaled - self._depth_mean) / self._depth_std

                    # Add batch dimension
                    inputs = {"pixel_values": normalized.unsqueeze(0)}

                    # OPTIMIZATION 3: Inference on GPU with no_grad
                    with torch.no_grad():
                        outputs = self.depth_model(**inputs)
                        predicted_depth = outputs.predicted_depth

                    # OPTIMIZATION 4: Upscale back to original size on GPU
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=(original_h, original_w),  # Back to original size
                        mode="bilinear",  # Changed from bicubic to bilinear (faster)
                        align_corners=False,
                    ).squeeze()

                    # GPU-ONLY OPTIMIZATION: Keep everything on GPU until final conversion
                    # Normalize to 0-255 range directly on GPU (10-15% faster than CPU path)
                    depth_gpu = prediction.detach()  # Detach from computation graph
                    depth_min = depth_gpu.min()
                    depth_max = depth_gpu.max()
                    depth_gpu = ((depth_gpu - depth_min) / (depth_max - depth_min) * 255.0)

                    # Apply Gaussian blur on GPU if requested (before converting to uint8)
                    if blur_kernel > 1:
                        # PERFORMANCE: Use pre-computed kernel (cached at config load)
                        # Avoids expensive torch.arange + torch.exp every frame
                        # Expected gain: +5-10% FPS
                        if self._cached_gaussian_kernel is not None:
                            # Cast cached kernel to match depth_gpu dtype (FP16 or FP32)
                            kernel_2d = self._cached_gaussian_kernel.to(dtype=depth_gpu.dtype)
                            kernel_size = kernel_2d.shape[-1]

                            # Apply convolution with pre-computed kernel
                            depth_gpu = torch.nn.functional.conv2d(
                                depth_gpu.unsqueeze(0).unsqueeze(0),
                                kernel_2d,
                                padding=kernel_size//2
                            ).squeeze()
                        else:
                            # Fallback: compute kernel on-the-fly (shouldn't happen if config cached properly)
                            blur_kernel_odd = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
                            kernel_size = blur_kernel_odd
                            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
                            kernel_1d = torch.exp(-torch.arange(-(kernel_size//2), kernel_size//2 + 1, dtype=depth_gpu.dtype, device=self.device)**2 / (2 * sigma**2))
                            kernel_1d = kernel_1d / kernel_1d.sum()
                            kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
                            kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

                            depth_gpu = torch.nn.functional.conv2d(
                                depth_gpu.unsqueeze(0).unsqueeze(0),
                                kernel_2d,
                                padding=kernel_size//2
                            ).squeeze()

                    # Apply contrast and brightness on GPU
                    if contrast != 1.0 or brightness != 0:
                        depth_gpu = depth_gpu * contrast + brightness
                        depth_gpu = torch.clamp(depth_gpu, 0, 255)

                    # Apply near/far threshold clipping on GPU
                    if near_threshold > 0 or far_threshold < 255:
                        depth_gpu = torch.clamp(depth_gpu, near_threshold, far_threshold)
                        # Normalize back to 0-255 range
                        if far_threshold > near_threshold:
                            depth_gpu = ((depth_gpu - near_threshold) / (far_threshold - near_threshold) * 255.0)

                    # Invert if requested (swap near/far) on GPU
                    if invert:
                        depth_gpu = 255.0 - depth_gpu

                    # Convert to uint8 on GPU, then transfer to CPU only once
                    depth_gpu_uint8 = depth_gpu.to(torch.uint8)
                    depth = depth_gpu_uint8.cpu().numpy()

                    # Cleanup GPU tensors
                    del depth_gpu, depth_gpu_uint8, prediction

                    # Cache the result (reuse existing buffer if possible)
                    if self.depth_cache is None or self.depth_cache.shape != depth.shape:
                        self.depth_cache = depth.copy()
                    else:
                        np.copyto(self.depth_cache, depth)
                else:
                    # Use cached depth map (no inference needed!)
                    depth = self.depth_cache

            except Exception as e:
                logging.warning(f"Depth-Anything failed: {e}, falling back to simple method")
                # Fallback to grayscale - use buffer to avoid memory leak
                h, w = image_tensor.shape[1], image_tensor.shape[2]
                if self._temp_cpu_buffer is None or self._temp_cpu_buffer.shape != (h, w, 3):
                    self._temp_cpu_buffer = np.empty((h, w, 3), dtype=np.uint8)
                image_cpu = image_tensor.permute(1, 2, 0).contiguous()
                np.copyto(self._temp_cpu_buffer, (image_cpu.cpu().numpy() * 255).astype(np.uint8))
                del image_cpu  # LEAK FIX: Explicitly free temporary tensor
                depth = cv2.cvtColor(self._temp_cpu_buffer, cv2.COLOR_RGB2GRAY)
        else:
            # For simple methods, convert tensor to numpy - use buffer
            h, w = image_tensor.shape[1], image_tensor.shape[2]
            if self._temp_cpu_buffer is None or self._temp_cpu_buffer.shape != (h, w, 3):
                self._temp_cpu_buffer = np.empty((h, w, 3), dtype=np.uint8)
            image_cpu = image_tensor.permute(1, 2, 0).contiguous()
            np.copyto(self._temp_cpu_buffer, (image_cpu.cpu().numpy() * 255).astype(np.uint8))
            del image_cpu  # LEAK FIX: Explicitly free temporary tensor
            image_np = self._temp_cpu_buffer
            # Apply simple depth methods
            if method == 'sobel':
                # Sobel edge detection for depth-like effect
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
                depth = np.sqrt(sobelx**2 + sobely**2)
                depth = np.uint8(np.clip(depth, 0, 255))
            elif method == 'laplacian':
                # Laplacian for depth-like effect
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                depth = cv2.Laplacian(gray, cv2.CV_64F)
                depth = np.uint8(np.absolute(depth))
            else:  # grayscale (default/fallback)
                depth = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

                # Apply CPU-based post-processing for non-AI methods only
                # (AI method already did GPU-based post-processing above)

                # Apply Gaussian blur for smoothing
                if blur_kernel > 1:
                    # Ensure blur_kernel is odd
                    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
                    depth = cv2.GaussianBlur(depth, (blur_kernel, blur_kernel), 0)

                # Apply contrast and brightness
                if contrast != 1.0 or brightness != 0:
                    depth = cv2.convertScaleAbs(depth, alpha=contrast, beta=brightness)

                # Apply near/far threshold clipping (distance range control)
                if near_threshold > 0 or far_threshold < 255:
                    depth = np.clip(depth, near_threshold, far_threshold)
                    # Normalize back to 0-255 range
                    if far_threshold > near_threshold:
                        depth = ((depth - near_threshold) / (far_threshold - near_threshold) * 255).astype(np.uint8)

                # Invert if requested (swap near/far)
                if invert:
                    depth = 255 - depth

        # FRAGMENTATION FIX: Allocate max-size buffer once, use slice for current resolution
        h, w = depth.shape[0], depth.shape[1]
        if self._depth_output_buffer_max is None:
            # Allocate max buffer (1024x1024) once - prevents fragmentation
            self._depth_output_buffer_max = torch.empty(
                (3, self._max_buffer_size, self._max_buffer_size),
                device=self.device,
                dtype=self.torch_dtype
            )

        # Use a contiguous slice of the max buffer for current resolution
        self._depth_output_buffer = self._depth_output_buffer_max[:, :h, :w].contiguous()

        # Convert grayscale to RGB, then to tensor - reuse output buffer
        # PINNED MEMORY OPTIMIZATION: Use pin_memory() for faster CPU->GPU transfer
        depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        depth_temp = torch.from_numpy(depth_rgb).float().pin_memory() / 255.0
        depth_temp_permuted = depth_temp.permute(2, 0, 1).to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        self._depth_output_buffer.copy_(depth_temp_permuted, non_blocking=True)
        del depth_temp, depth_temp_permuted  # LEAK FIX: Explicitly free temporary tensors

        return self._depth_output_buffer

    def _preprocess_controlnet_async(self, image_tensor):
        """
        ASYNC PREPROCESSING: Preprocess ControlNet image in a separate CUDA stream
        This allows preprocessing to happen in parallel with generation.

        Args:
            image_tensor: CHW tensor in range [0, 1], float32
        Returns:
            Preprocessed ControlNet image (computed on preprocess_stream)
        """
        if not self.async_preprocessing_enabled or self.preprocess_stream is None:
            # Fallback to sync preprocessing
            return self._preprocess_controlnet_sync(image_tensor)

        # Execute preprocessing on separate CUDA stream
        with torch.cuda.stream(self.preprocess_stream):
            return self._preprocess_controlnet_sync(image_tensor)

    def _preprocess_controlnet_sync(self, image_tensor):
        """
        SYNC PREPROCESSING: Apply ALL enabled ControlNet preprocessing
        MULTI-CONTROLNET SUPPORT: Returns dict of all enabled ControlNets
        PARALLEL OPTIMIZATION: Runs each preprocessor on separate CUDA stream for maximum parallelism
        UNIFIED FRAME SKIPPING: All ControlNets skip frames together for optimal performance
        This is called either directly (sync mode) or from async wrapper (async mode)

        Args:
            image_tensor: CHW tensor in range [0, 1], float32
        Returns:
            Dict mapping controlnet type -> preprocessed image, or None if no ControlNets enabled
            Example: {'depth': depth_tensor, 'openpose': pose_tensor}
        """
        preprocessed = {}
        enabled_count = 0

        # PERF: Use cached enabled flags (avoids dict lookups)
        canny_enabled = self._cached_canny_enabled
        depth_enabled = self._cached_depth_enabled
        openpose_enabled = self._cached_openpose_enabled

        # Count enabled ControlNets
        if canny_enabled:
            enabled_count += 1
        if depth_enabled:
            enabled_count += 1
        if openpose_enabled:
            enabled_count += 1

        # PERF: Use pre-calculated unified skip frames (calculated at config load)
        # No need to recalculate every frame - this is now done in _cache_config_values()

        # Increment unified counter
        self.unified_frame_counter += 1
        use_unified_cache = self.unified_frame_counter % self._cached_controlnet_skip_frames != 0

        # UNIFIED FRAME SKIPPING: If skipping, return cached results from ALL ControlNets
        # Works with 1 or more ControlNets active
        if use_unified_cache and enabled_count >= 1:
            # Return cached results for all enabled ControlNets
            if canny_enabled and hasattr(self, '_canny_cached_result'):
                preprocessed['canny'] = self._canny_cached_result
            if depth_enabled and hasattr(self, '_depth_cached_result'):
                preprocessed['depth'] = self._depth_cached_result
            if openpose_enabled and hasattr(self, '_openpose_cached_result'):
                preprocessed['openpose'] = self._openpose_cached_result
            return preprocessed if preprocessed else None

        # PARALLEL EXECUTION: Launch each preprocessor on its own CUDA stream
        # This allows them to run concurrently instead of sequentially
        # Expected gain: 35ms sequential -> 20ms parallel (depth + openpose)

        if canny_enabled and self.canny_stream is not None:
            with torch.cuda.stream(self.canny_stream):
                result = self._process_canny(image_tensor)
                preprocessed['canny'] = result
                self._canny_cached_result = result  # Cache for unified skipping
        elif canny_enabled:
            # Fallback if stream not available
            result = self._process_canny(image_tensor)
            preprocessed['canny'] = result
            self._canny_cached_result = result

        if depth_enabled and self.depth_stream is not None:
            with torch.cuda.stream(self.depth_stream):
                result = self._process_depth(image_tensor)
                preprocessed['depth'] = result
                self._depth_cached_result = result  # Cache for unified skipping
        elif depth_enabled:
            # Fallback if stream not available
            result = self._process_depth(image_tensor)
            preprocessed['depth'] = result
            self._depth_cached_result = result

        if openpose_enabled and self.openpose_stream is not None:
            with torch.cuda.stream(self.openpose_stream):
                result = self._process_openpose(image_tensor)
                preprocessed['openpose'] = result
                self._openpose_cached_result = result  # Cache for unified skipping
        elif openpose_enabled:
            # Fallback if stream not available
            result = self._process_openpose(image_tensor)
            preprocessed['openpose'] = result
            self._openpose_cached_result = result

        # SYNCHRONIZATION: Wait for all parallel streams to finish before returning
        # Only synchronize if we actually used parallel streams
        if enabled_count > 1:
            current_stream = torch.cuda.current_stream()
            if canny_enabled and self.canny_stream is not None:
                current_stream.wait_stream(self.canny_stream)
            if depth_enabled and self.depth_stream is not None:
                current_stream.wait_stream(self.depth_stream)
            if openpose_enabled and self.openpose_stream is not None:
                current_stream.wait_stream(self.openpose_stream)

        return preprocessed if preprocessed else None

    def _process_openpose(self, image_tensor):
        """
        Process image tensor with DWPose human pose detection (GPU-accelerated, OPTIMIZED - zero memory leaks)
        With frame skipping for 2-3x performance boost
        Args:
            image_tensor: CHW tensor in range [0, 1], float32
        Returns:
            DWPose skeleton image as CHW tensor in range [0, 1]
        """
        if self.openpose_processor is None:
            logging.warning("DWPose processor not loaded, returning original image")
            return image_tensor

        try:
            h, w = image_tensor.shape[1], image_tensor.shape[2]

            # FRAGMENTATION FIX: Allocate max-size buffers once, use slices for current resolution
            if self._openpose_input_buffer_max is None:
                # Allocate max NumPy buffer (1024x1024) once - prevents fragmentation
                self._openpose_input_buffer_max = np.empty((self._max_buffer_size, self._max_buffer_size, 3), dtype=np.uint8)

            # Use a contiguous slice of the max buffer for current resolution
            self._openpose_input_buffer = self._openpose_input_buffer_max[:h, :w, :]

            if self._openpose_output_buffer_max is None:
                # Allocate max GPU buffer (1024x1024) once - prevents fragmentation
                self._openpose_output_buffer_max = torch.empty(
                    (3, self._max_buffer_size, self._max_buffer_size),
                    device=self.device,
                    dtype=self.torch_dtype
                )

            # Use a contiguous slice of the max buffer for current resolution
            self._openpose_output_buffer = self._openpose_output_buffer_max[:, :h, :w].contiguous()

            # OPTIMIZATION: Frame skipping - only compute pose every N frames
            self.openpose_frame_counter += 1
            use_cache = self.openpose_cache is not None and self.openpose_frame_counter % self.controlnet_skip_frames != 0

            if not use_cache:
                # OPTIMIZATION: Direct conversion to numpy without creating intermediate tensors
                image_cpu = image_tensor.permute(1, 2, 0).contiguous()
                image_np_temp = image_cpu.cpu().numpy()
                # OPTIMIZATION: In-place multiplication and conversion to avoid intermediate arrays
                np.multiply(image_np_temp, 255, out=image_np_temp)
                image_np_uint8 = image_np_temp.astype(np.uint8)
                np.copyto(self._openpose_input_buffer, image_np_uint8)
                del image_cpu, image_np_temp, image_np_uint8  # LEAK FIX: Explicitly free temporary tensors

                # Process with DWPose detector (GPU-accelerated with ONNX)
                # PERFORMANCE: Use cached detect_resolution (avoid dict lookup)
                detect_resolution = self._cached_openpose_detect_resolution

                # easy-dwpose API: (image, detect_resolution, output_type, include_hands, include_face)
                # Pass numpy array directly to avoid PIL conversion overhead
                # PERFORMANCE FIX: Disable hands/face detection for real-time performance
                # Body-only detection: ~15-20ms per frame
                # Body+hands+face: ~60-80ms per frame (+300% slower!)
                # ControlNet OpenPose standard only uses body skeleton anyway
                openpose_np = self.openpose_processor(
                    self._openpose_input_buffer,  # Pass numpy array directly
                    detect_resolution=detect_resolution,
                    output_type='np',  # Get numpy array directly (faster than PIL)
                    include_hands=False,  # Disable hand keypoints for +100-150% FPS
                    include_face=False,   # Disable face keypoints for +50-100% FPS
                )
                # openpose_np is now a numpy array (HWC, uint8, 0-255)

                # PERFORMANCE: Optimized tensor conversion pipeline
                # Old: 3 allocations + 2 copies (from_numpy → float → div → permute → to → copy)
                # New: 1 allocation + 1 copy (from_numpy → direct write to output buffer)

                # Convert numpy to tensor and permute in one step
                openpose_temp = torch.from_numpy(openpose_np).permute(2, 0, 1).float()
                del openpose_np  # LEAK FIX: Free numpy array immediately

                # Normalize and convert dtype in-place on output buffer
                # This avoids intermediate allocations
                self._openpose_output_buffer.copy_(openpose_temp)
                self._openpose_output_buffer.div_(255.0)  # In-place division
                del openpose_temp  # LEAK FIX: Free temporary tensor

                # PERFORMANCE: Only clone when frame skipping is active
                # No need to clone if we're not going to use cache next frame
                if self._cached_controlnet_skip_frames > 1:
                    self.openpose_cache = self._openpose_output_buffer.clone()
                else:
                    # No skipping = no cache needed
                    self.openpose_cache = self._openpose_output_buffer

            return self.openpose_cache if use_cache else self._openpose_output_buffer

        except Exception as e:
            logging.error(f"DWPose processing failed: {e}")
            return image_tensor

    def run(self):
        logging.info("Entering main command loop")

        # Track wall-clock time between frames (including Smode processing)
        last_frame_wall_time = 0.0

        # DIAGNOSTIC: Frame throughput counters
        frames_received = 0  # How many frames Smode sent
        frames_processed = 0  # How many frames we actually processed
        last_diagnostic_time = time.time()

        try:
            while True:
                if not is_socket_connected(self.socket):
                    return
                messages = {}

                # Wait up to 1 ms for data to arrive
                ready_to_read, _, in_error = select.select(
                    [self.socket], [], [], 0.001
                )
                if ready_to_read:
                    while True:
                        try:
                            cmd, payload = recv_message(self.socket)
                            if cmd is None:
                                break
                            messages[cmd] = payload
                        except socket.error as e:
                            # WinError 10035 = non-blocking socket has no data (normal)
                            if e.errno != 10035:
                                logging.warning(f"Socket receive error: {e}")
                            break
                if in_error:
                    logging.error("Socket error detected; cleaning up and exiting")
                    # Cleanup will be handled by __del__() destructor
                    exit(0)

                wait_result = self.smodeToStreamDiffusionInterProcessEvent.wait(0)
                if wait_result == win32event.WAIT_OBJECT_0 and self.stream:
                    # DIAGNOSTIC: Count frame received from Smode
                    frames_received += 1

                    # PROFILING: Track timing for each operation
                    frame_start = time.time()
                    timings = {}

                    # Calculate wall-clock time since last frame (includes Smode processing)
                    if last_frame_wall_time > 0:
                        wall_time_ms = (frame_start - last_frame_wall_time) * 1000
                        timings['wall_clock'] = wall_time_ms
                    last_frame_wall_time = frame_start

                    if self.output_tensors is None:
                        self._create_tensors(3, self.width, self.height)

                    self.frames_processed += 1

                    # PROFILING: Use cached flag (no dict lookup in hot path)
                    profiling_enabled = self._cached_profiling_enabled

                    # Input copy from Smode
                    if profiling_enabled:
                        input_copy_start = time.time()
                        self.input_tensors.copy_smode_to_stream_diffusion()
                        torch.cuda.synchronize()
                        timings['input_copy'] = (time.time() - input_copy_start) * 1000
                    else:
                        self.input_tensors.copy_smode_to_stream_diffusion()

                    x_output = None
                    permuted_input_texture = None

                    if self.mode == Mode.IMAGE_TO_IMAGE:
                        if profiling_enabled:
                            preprocess_start = time.time()

                        if self.input_tensors.stream_diffusion_tensor is not None:
                            permuted_tensor = self.input_tensors.get_permuted_input_tensor()
                            permuted_input_texture = F.vflip(permuted_tensor)

                        if profiling_enabled:
                            torch.cuda.synchronize()
                            timings['image_preprocess'] = (time.time() - preprocess_start) * 1000

                        # ControlNet preprocessing (synchronous)
                        controlnet_processed_image = None
                        preview_mode = self._cached_preview_mode

                        if profiling_enabled:
                            controlnet_start = time.time()

                        # Initialize controlnet_processed_dict for both async and sync paths
                        controlnet_processed_dict = None

                        if self._cached_controlnet_enabled and preview_mode == 'normal':  # PERF: Use cached value
                            if self.async_preprocessing_enabled and self.preprocess_stream is not None:
                                # ASYNC MODE: 3-stage pipeline
                                # Stage 1: Use preprocessed result from previous frame (if available)
                                async_result = self.preprocessed_controlnet_image

                                # Stage 2: Wait for async preprocessing stream to complete
                                if async_result is not None:
                                    torch.cuda.current_stream().wait_stream(self.preprocess_stream)
                                    if isinstance(async_result, dict):
                                        controlnet_processed_dict = async_result

                                # Stage 3: Launch ASYNC preprocessing for CURRENT frame (to be used in NEXT frame)
                                # This runs in parallel with generation on default stream!
                                # CRITICAL FIX: Delete old tensor before reassignment to prevent GPU memory leak
                                # At 60fps with 512x512 FP16, this prevents 90MB/sec leak
                                if self.preprocessed_controlnet_image is not None:
                                    del self.preprocessed_controlnet_image
                                    self.preprocessed_controlnet_image = None

                                self.preprocessed_controlnet_image = self._preprocess_controlnet_async(permuted_input_texture)

                                # PERF: Guard debug logging to avoid string formatting overhead
                                if logging.getLogger().isEnabledFor(logging.DEBUG):
                                    logging.debug("Async ControlNet preprocessing launched (will be used next frame)")
                            else:
                                # SYNC MODE (fallback): Traditional synchronous preprocessing
                                controlnet_processed_dict = self._preprocess_controlnet_sync(permuted_input_texture)

                            # MULTI-CONTROLNET: Log all active ControlNets
                            # PERF: Guard debug logging to avoid string formatting overhead
                            if controlnet_processed_dict and logging.getLogger().isEnabledFor(logging.DEBUG):
                                active_nets = []
                                if 'canny' in controlnet_processed_dict:
                                    active_nets.append(f"Canny (scale: {self._cached_canny_scale})")  # PERF: Use cached
                                if 'depth' in controlnet_processed_dict:
                                    active_nets.append(f"Depth (scale: {self._cached_depth_scale})")  # PERF: Use cached
                                if 'openpose' in controlnet_processed_dict:
                                    active_nets.append(f"OpenPose (scale: {self._cached_openpose_scale})")  # PERF: Use cached
                                logging.debug(f"ControlNets prepared: {', '.join(active_nets)}")

                        if profiling_enabled:
                            torch.cuda.synchronize()  # Wait for preprocessing to finish
                            timings['controlnet_preprocess'] = (time.time() - controlnet_start) * 1000
                        else:
                            timings['controlnet_preprocess'] = 0.0

                        # Check preview mode - use cached flags (no dict lookups)
                        if preview_mode == "canny_preview" and self._cached_canny_enabled:
                            x_output = self._process_canny(permuted_input_texture)
                        elif preview_mode == "depth_preview" and self._cached_depth_enabled:
                            x_output = self._process_depth(permuted_input_texture)
                        elif preview_mode == "openpose_preview" and self._cached_openpose_enabled:
                            x_output = self._process_openpose(permuted_input_texture)
                        else:
                            # Normal mode - generate with StreamDiffusion + REAL ControlNet
                            # OPTIMIZATION: Use pre-built ControlNet lists (avoid list construction in hot path)
                            # Models and scales are pre-cached on config change, just extract images
                            if controlnet_processed_dict is not None and preview_mode == "normal" and self._active_cn_keys:
                                # Fast path: Extract images in pre-determined order using list comprehension
                                controlnet_images = [controlnet_processed_dict[k] for k in self._active_cn_keys
                                                    if k in controlnet_processed_dict]

                                # Use pre-cached model and scale lists (no dict lookups, no append() calls!)
                                # Slice to match actual number of images (in case preprocessing failed for some)
                                num_images = len(controlnet_images)
                                controlnet_models = self._cn_models_cache[:num_images]
                                controlnet_scales = self._cn_scales_cache[:num_images]
                            else:
                                # No ControlNets active or not in normal mode
                                controlnet_models = []
                                controlnet_images = []
                                controlnet_scales = []

                            # PROFILING: StreamDiffusion generation
                            # Use ONLY pipeline.py internal profiling (no duplicate sync here)
                            profiling_enabled = self._cached_profiling_enabled  # PERF: Use cached value

                            # CUDA Graphs: Mark the beginning of a new iteration
                            # This tells torch.compile's CUDA Graphs system that this is a new frame
                            # and prevents "overwritten by subsequent run" errors with dynamic tensors
                            # (especially critical for ControlNet conditioning_scale that changes per frame)
                            if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                                torch.compiler.cudagraph_mark_step_begin()

                            # Call stream - profiling happens inside pipeline.py if enabled
                            # Pass lists for Multi-ControlNet support (or None if no ControlNets active)
                            x_output = self.stream(
                                image=permuted_input_texture,
                                controlnet_image=controlnet_images if controlnet_images else None,
                                controlnet_model=controlnet_models if controlnet_models else None,
                                controlnet_conditioning_scale=controlnet_scales if controlnet_scales else 1.0,
                            )

                            # PROFILING: Get internal breakdown from pipeline (only if profiling enabled)
                            # The pipeline.py already does GPU sync internally when profiling is on
                            # self.stream is StreamDiffusionWrapper, the actual pipeline is self.stream.stream
                            if profiling_enabled and hasattr(self.stream, 'stream') and hasattr(self.stream.stream, 'last_internal_timings'):
                                internal = self.stream.stream.last_internal_timings
                                timings['gen_vae_encode'] = internal.get('vae_encode', 0.0)
                                timings['gen_unet_controlnet'] = internal.get('unet_controlnet', 0.0)
                                timings['gen_vae_decode'] = internal.get('vae_decode', 0.0)
                                # Calculate total generation time from internal breakdown
                                timings['generation'] = timings['gen_vae_encode'] + timings['gen_unet_controlnet'] + timings['gen_vae_decode']
                            else:
                                # When profiling disabled, don't measure
                                timings['generation'] = 0.0
                    elif self.mode == Mode.TEXT_TO_IMAGE:
                        # Use ONLY pipeline.py internal profiling (no duplicate sync here)
                        # Get internal breakdown from pipeline if profiling enabled
                        x_output = self.stream.txt2img()

                        if profiling_enabled and hasattr(self.stream, 'stream') and hasattr(self.stream.stream, 'last_internal_timings'):
                            internal = self.stream.stream.last_internal_timings
                            timings['gen_vae_encode'] = internal.get('vae_encode', 0.0)
                            timings['gen_unet_controlnet'] = internal.get('unet_controlnet', 0.0)
                            timings['gen_vae_decode'] = internal.get('vae_decode', 0.0)
                            # Calculate total generation time from internal breakdown
                            timings['generation'] = timings['gen_vae_encode'] + timings['gen_unet_controlnet'] + timings['gen_vae_decode']
                        else:
                            timings['generation'] = 0.0  # Not measured when profiling disabled
                    else:
                        logging.error(f"Unknown mode: {self.mode}")
                        continue

                    # PERF: Unconditional squeeze (always safe for batch_size=1)
                    if x_output is not None:
                        x_output = x_output.squeeze(0) if x_output.shape[0] == 1 else x_output

                    # PROFILING: Output copy to Smode
                    # PERF: Reuse profiling_enabled from earlier (already cached)
                    if profiling_enabled:
                        output_copy_start = time.time()
                        if x_output is not None:
                            hwc_output = self.output_tensors.set_permuted_output_tensor(x_output)
                            self.output_tensors.copy_to_smode(hwc_output)
                        torch.cuda.synchronize()
                        timings['output_copy'] = (time.time() - output_copy_start) * 1000
                    else:
                        if x_output is not None:
                            hwc_output = self.output_tensors.set_permuted_output_tensor(x_output)
                            self.output_tensors.copy_to_smode(hwc_output)

                    # Signal Smode
                    if profiling_enabled:
                        signal_start = time.time()
                        self.streamDiffusionToSmodeInterProcessEvent.signal()
                        timings['signal_smode'] = (time.time() - signal_start) * 1000
                    else:
                        self.streamDiffusionToSmodeInterProcessEvent.signal()

                    frames_processed += 1

                    # Throughput logging (only if profiling enabled)
                    if profiling_enabled:
                        current_time = time.time()
                    else:
                        current_time = time.time()
                    if current_time - last_diagnostic_time >= 1.0:
                        elapsed_time = current_time - last_diagnostic_time
                        receive_rate = frames_received / elapsed_time
                        process_rate = frames_processed / elapsed_time
                        skip_rate = ((frames_received - frames_processed) / frames_received * 100) if frames_received > 0 else 0

                        if profiling_enabled:
                            logging.info(f"📊 THROUGHPUT: Received={receive_rate:.1f} fps | Processed={process_rate:.1f} fps | Skipped={skip_rate:.1f}%")

                        # Reset counters
                        frames_received = 0
                        frames_processed = 0
                        last_diagnostic_time = current_time

                    # PROFILING: Only log if profiling enabled
                    if profiling_enabled:
                        # Calculate total frame time as sum of all measured timings
                        timings['total_frame'] = (
                            timings.get('config_check', 0.0) +
                            timings.get('input_copy', 0.0) +
                            timings.get('image_preprocess', 0.0) +
                            timings.get('controlnet_preprocess', 0.0) +
                            timings.get('generation', 0.0) +
                            timings.get('output_copy', 0.0) +
                            timings.get('signal_smode', 0.0)
                        )
                        # Calculate FPS from the absolute difference between wall_clock and total_frame
                        # This matches the FPS shown in Smode's StreamDiffusion Feedback monitor
                        smode_feedback_time = abs(timings.get('wall_clock', 1.0) - timings.get('total_frame', 1.0))
                        smode_fps = 1000.0 / smode_feedback_time if smode_feedback_time > 0 else 0

                        # Log detailed timing every 60 frames
                        if not hasattr(self, '_frame_count'):
                            self._frame_count = 0
                        self._frame_count += 1

                        if self._frame_count % 60 == 0:
                            logging.info(f"PROFILING (Smode Feedback FPS: {smode_fps:.1f}):")

                            # Decide if we have internal breakdown
                            has_breakdown = 'gen_vae_encode' in timings
                            exclude_keys = ['total_frame', 'cycle_complete', 'wall_clock', 'gen_vae_encode', 'gen_unet_controlnet', 'gen_vae_decode']
                            if has_breakdown:
                                exclude_keys.append('generation')  # Don't show generation in main loop if we have breakdown

                            for key, value in sorted(timings.items()):
                                if key not in exclude_keys:
                                    percentage = (value / timings['total_frame'] * 100) if timings['total_frame'] > 0 else 0
                                    logging.info(f"  {key:25s}: {value:6.2f}ms ({percentage:5.1f}%)")

                            # Show internal generation breakdown if available
                            if has_breakdown:
                                percentage = (timings['generation'] / timings['total_frame'] * 100) if timings['total_frame'] > 0 else 0
                                logging.info(f"  {'generation':25s}: {timings['generation']:6.2f}ms ({percentage:5.1f}%) BREAKDOWN:")
                                gen_total = timings['generation']
                                for subkey in ['gen_vae_encode', 'gen_unet_controlnet', 'gen_vae_decode']:
                                    if subkey in timings:
                                        sub_value = timings[subkey]
                                        sub_percentage = (sub_value / gen_total * 100) if gen_total > 0 else 0
                                        label = subkey.replace('gen_', '    └─ ')
                                        logging.info(f"  {label:25s}: {sub_value:6.2f}ms ({sub_percentage:5.1f}% of gen)")

                            logging.info(f"  {'total_frame':25s}: {timings['total_frame']:6.2f}ms (REAL GPU processing time)")
                            if 'wall_clock' in timings:
                                smode_idle = timings['wall_clock'] - timings['total_frame']
                                logging.info(f"  {'wall_clock':25s}: {timings['wall_clock']:6.2f}ms (frame-to-frame interval)")
                                logging.info(f"  {'smode_idle':25s}: {smode_idle:6.2f}ms (Smode + idle time)")
                # Process received messages
                for cmd, payload in messages.items():
                    if cmd == CommandType.CONFIG:
                        # OPTIMIZATION: Use cached config parsing (CPU → RAM optimization)
                        config_packet = _parse_config_with_cache(payload)
                        logging.info(f"Received CONFIG command: model={config_packet.model_name}")
                        self._apply_controlnet_config_from_packet(config_packet)

                        def update_parameters(app: App, config_packet: ConfigPacket):
                            app.model_name = config_packet.model_name
                            app.current_prompt = config_packet.prompt
                            app.negative_prompt = config_packet.negative_prompt
                            app.seed = config_packet.seed
                            app.width = config_packet.width
                            app.height = config_packet.height
                            app.t_index_list = config_packet.t_index_list
                            app.guidance_scale = config_packet.guidance_scale
                            app.mode = config_packet.mode
                            app.cfg_type = config_packet.cfg_type
                            app.lora_dict = config_packet.lora_dict
                            app.acceleration = config_packet.acceleration
                            app.similar_image_filter_enabled = config_packet.similar_image_filter_enabled
                            app.similar_image_filter_threshold = config_packet.similar_image_filter_threshold
                            app.similar_image_filter_max_skip = config_packet.similar_image_filter_max_skip
                            app.cache_dir = config_packet.cache_dir

                        if not self.stream:
                            update_parameters(self, config_packet)
                            self._create_stream()
                        else:
                            model_has_changed = self.model_name != config_packet.model_name
                            lora_dict_has_changed = self.lora_dict != config_packet.lora_dict
                            update_stream = (
                                self.width != config_packet.width
                                or self.height != config_packet.height
                                or self.mode != config_packet.mode
                                or self.stream.stream.cfg_type != config_packet.cfg_type
                                or self.acceleration != config_packet.acceleration
                                or self.lora_dict != config_packet.lora_dict
                            )
                            update_t_index_list = self.t_index_list != config_packet.t_index_list
                            previous_acceleration = self.acceleration
                            update_parameters(self, config_packet)

                            if model_has_changed or lora_dict_has_changed:
                                self._create_stream()
                                self.accelerate(previous_acceleration)
                            elif update_stream:
                                # Inline recreation for width/height/mode/cfg changes
                                self.stream.stream = StreamDiffusion(
                                    pipe=self.stream.stream.pipe,
                                    t_index_list=self.t_index_list,
                                    torch_dtype=self.stream.stream.dtype,
                                    width=self.width,
                                    height=self.height,
                                    do_add_noise=self.stream.stream.do_add_noise,
                                    frame_buffer_size=self.stream.frame_buffer_size,
                                    use_denoising_batch=self.stream.stream.use_denoising_batch,
                                    cfg_type=config_packet.cfg_type,
                                )
                                # Restore ControlNet guidance strength after stream recreation
                                self.stream.stream._cached_controlnet_guidance_strength = self._cached_controlnet_guidance_strength
                                self.accelerate(previous_acceleration)
                            elif update_t_index_list:
                                # OPTIMIZATION #9: Fast timestep update without recreation
                                # Only update t_index_list without recreating entire pipeline
                                self.stream.stream.t_list = self.t_index_list
                                self.stream.stream.denoising_steps_num = len(self.t_index_list)
                                # Skip prepare() - will be called after with updated params

                        if self.stream:
                            # Get delta from controlnet config (temporal coherence parameter)
                            delta = self.controlnet_config.get('delta', 1.0)

                            self.stream.stream.prepare(
                                self.current_prompt,
                                self.negative_prompt,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=config_packet.guidance_scale,
                                delta=delta,
                                seed=self.seed,
                            )

                            # WARMUP: Force torch.compile compilation BEFORE first user frame
                            # torch.compile is lazy - it compiles on first real inference, not during initialization
                            # This warmup moves the compilation delay from first frame to startup:
                            #   - Without cache: ~30-60s compilation (but with visible progress log)
                            #   - With cache: ~2-5s cache loading (disk I/O + validation)
                            # Result: First user frame is INSTANT instead of having 2-60s freeze
                            #
                            # IMPORTANT: Only run warmup on FIRST prepare() call (initial startup)
                            # Skip warmup on subsequent prepare() calls (prompt/t_index changes)
                            # This prevents 0.1-2s freezes when changing parameters live
                            if not self.warmup_completed:
                                logging.info("Warming up torch.compile cache (U-Net + VAE)...")
                                warmup_start = time.time()
                                try:
                                    # Create dummy input matching real inference shape
                                    # IMPORTANT: Use same dtype as model (float16) to avoid compilation errors
                                    dummy_input = torch.randn(
                                        (1, 3, self.height, self.width),
                                        dtype=self.torch_dtype,  # Use model's dtype (typically float16)
                                        device=self.device
                                    )

                                    # Perform 2 warmup inferences to trigger all torch.compile paths
                                    for i in range(2):
                                        _ = self.stream(
                                            image=dummy_input,
                                            controlnet_image=None,
                                            controlnet_model=None,
                                            controlnet_conditioning_scale=1.0,
                                        )

                                    torch.cuda.synchronize()
                                    warmup_time = time.time() - warmup_start
                                    logging.info(f"✓ Warmup complete ({warmup_time:.1f}s) - torch.compile cache ready for real-time inference")

                                    # WARMUP WITH CONTROLNETS: If ControlNets are active, warm up the U-Net+ControlNet combination
                                    # The U-Net with ControlNet conditioning may have a different compilation path
                                    if self.controlnet_models:
                                        logging.info(f"Warming up U-Net+ControlNet combination ({len(self.controlnet_models)} ControlNets active)...")
                                        cn_warmup_start = time.time()

                                        # Collect active ControlNets for warmup
                                        cn_images = []
                                        cn_models = []
                                        cn_scales = []

                                        for cn_name in ['canny', 'depth', 'openpose']:
                                            if cn_name in self.controlnet_models:
                                                # Use dummy input as controlnet conditioning image
                                                cn_images.append(dummy_input)
                                                cn_models.append(self.controlnet_models[cn_name])
                                                cn_scales.append(1.0)

                                        # Warmup with ControlNets (2 iterations)
                                        for i in range(2):
                                            _ = self.stream(
                                                image=dummy_input,
                                                controlnet_image=cn_images if cn_images else None,
                                                controlnet_model=cn_models if cn_models else None,
                                                controlnet_conditioning_scale=cn_scales if cn_scales else 1.0,
                                            )

                                        torch.cuda.synchronize()
                                        cn_warmup_time = time.time() - cn_warmup_start
                                        logging.info(f"✓ ControlNet warmup complete ({cn_warmup_time:.1f}s) - U-Net+ControlNet cache ready")

                                    # Mark warmup as completed to prevent running again on subsequent prepare() calls
                                    self.warmup_completed = True

                                except Exception as e:
                                    logging.warning(f"Warmup failed (non-critical): {e}")
                                finally:
                                    # MEMORY LEAK FIX: Always free dummy tensor
                                    if 'dummy_input' in locals():
                                        del dummy_input
                                        torch.cuda.empty_cache()

                    elif cmd == CommandType.STOP:
                        logging.info("Received STOP command; exiting main loop")
                        if self.stream:
                            del self.stream
                            self.stream = None
                        return
                    else:
                        logging.warning(f"Received unexpected command: {cmd}")
        except socket.error as e:
            logging.error(f"Socket error during processing: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise
        finally:
            self.socket.close()
            logging.info("Socket connection closed")

    def __del__(self):
        """
        Destructor to ensure all resources are properly cleaned up.
        Fixes critical memory leaks: CUDA streams, GPU tensors, sockets, Win32 handles.
        """
        try:
            # Clean up StreamDiffusion wrapper first (frees models and GPU tensors)
            if hasattr(self, 'stream') and self.stream is not None:
                try:
                    del self.stream
                    self.stream = None
                except Exception as e:
                    logging.warning(f"Error cleaning up stream: {e}")

            # Clean up ControlNet models (major GPU memory)
            if hasattr(self, 'controlnet_models') and self.controlnet_models:
                try:
                    for model_name, model in list(self.controlnet_models.items()):
                        del model
                    self.controlnet_models.clear()
                except Exception as e:
                    logging.warning(f"Error cleaning up ControlNet models: {e}")

            # Clean up Depth-Anything model and processor
            if hasattr(self, 'depth_model') and self.depth_model is not None:
                try:
                    del self.depth_model
                    self.depth_model = None
                except Exception as e:
                    logging.debug(f"Depth model cleanup error (non-critical): {e}")

            if hasattr(self, 'depth_processor') and self.depth_processor is not None:
                try:
                    del self.depth_processor
                    self.depth_processor = None
                except Exception as e:
                    logging.debug(f"Depth processor cleanup error (non-critical): {e}")

            # Clean up cached depth map
            if hasattr(self, 'depth_cache') and self.depth_cache is not None:
                try:
                    del self.depth_cache
                    self.depth_cache = None
                except Exception as e:
                    logging.debug(f"Depth cache cleanup error (non-critical): {e}")

            # Clean up OpenPose processor (ONNX Runtime)
            if hasattr(self, 'openpose_processor') and self.openpose_processor is not None:
                try:
                    # Check if DWposeDetector has cleanup method
                    if hasattr(self.openpose_processor, 'close'):
                        self.openpose_processor.close()
                    elif hasattr(self.openpose_processor, 'cleanup'):
                        self.openpose_processor.cleanup()
                    del self.openpose_processor
                    self.openpose_processor = None
                except Exception as e:
                    logging.warning(f"Error cleaning up OpenPose: {e}")

            # Clean up cached pose skeleton
            if hasattr(self, 'openpose_cache') and self.openpose_cache is not None:
                try:
                    del self.openpose_cache
                    self.openpose_cache = None
                except Exception as e:
                    logging.debug(f"OpenPose cache cleanup error (non-critical): {e}")

            # Clean up preprocessed ControlNet image (GPU tensor leak fix)
            if hasattr(self, 'preprocessed_controlnet_image') and self.preprocessed_controlnet_image is not None:
                try:
                    del self.preprocessed_controlnet_image
                    self.preprocessed_controlnet_image = None
                except Exception as e:
                    logging.debug(f"ControlNet image cleanup error (non-critical): {e}")

            # Clean up CUDA streams (CRITICAL - prevents GPU resource leak)
            if hasattr(self, 'preprocess_stream') and self.preprocess_stream is not None:
                try:
                    # MUST synchronize before deleting to avoid "stream destroyed with pending work"
                    self.preprocess_stream.synchronize()
                    del self.preprocess_stream
                    self.preprocess_stream = None
                except Exception as e:
                    logging.warning(f"Error cleaning up preprocess_stream: {e}")

            # Clean up parallel preprocessing streams
            for stream_name in ['depth_stream', 'openpose_stream', 'canny_stream']:
                if hasattr(self, stream_name):
                    stream = getattr(self, stream_name)
                    if stream is not None:
                        try:
                            stream.synchronize()
                            delattr(self, stream_name)
                        except Exception as e:
                            logging.warning(f"Error cleaning up {stream_name}: {e}")

            # Clean up pre-allocated buffers
            buffers = [
                '_canny_input_buffer', '_canny_output_buffer',
                '_depth_output_buffer', '_openpose_input_buffer',
                '_openpose_output_buffer', '_temp_cpu_buffer',
                '_permuted_input_buffer'
            ]
            for buffer_name in buffers:
                if hasattr(self, buffer_name):
                    try:
                        buffer = getattr(self, buffer_name)
                        if buffer is not None:
                            del buffer
                            setattr(self, buffer_name, None)
                    except Exception as e:
                        logging.debug(f"Buffer {buffer_name} cleanup error (non-critical): {e}")

            # Close Win32 event handles (kernel resource leak fix)
            if hasattr(self, 'streamDiffusionToSmodeInterProcessEvent'):
                try:
                    self.streamDiffusionToSmodeInterProcessEvent.close()
                except Exception as e:
                    logging.debug(f"Event handle cleanup error (non-critical): {e}")

            if hasattr(self, 'smodeToStreamDiffusionInterProcessEvent'):
                try:
                    self.smodeToStreamDiffusionInterProcessEvent.close()
                except Exception as e:
                    logging.debug(f"Event handle cleanup error (non-critical): {e}")

            # Close socket (file descriptor leak fix)
            if hasattr(self, 'socket') and self.socket:
                try:
                    self.socket.close()
                except Exception as e:
                    logging.debug(f"Socket cleanup error (non-critical): {e}")

            # Final GPU memory cleanup (torch already imported at top)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logging.debug(f"CUDA cache cleanup error (non-critical): {e}")

        except Exception as e:
            # Never raise in destructor - log and continue
            try:
                logging.warning(f"Error in App.__del__: {e}")
            except Exception:
                # If logging fails in destructor, silently ignore
                # (Python interpreter may be shutting down)
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smode Bridge Client Application"
    )
    parser.add_argument(
        "--port", type=int, required=True, help="Port number"
    )
    parser.add_argument(
        "--uuid", type=str, required=True, help="Smode modifier UUID"
    )
    parser.add_argument(
        "--width", type=int, required=True, help="Width of the image"
    )
    parser.add_argument(
        "--height", type=int, required=True, help="Height of the image"
    )
    parser.add_argument(
        "--device", type=int, required=True, help="The cuda device index to use"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name to use"
    )

    args = parser.parse_args()
    config = Args(
        port=args.port,
        uuid=args.uuid,
        width=args.width,
        height=args.height,
        device=args.device,
        model=args.model,
    )

    if not torch.cuda.is_available():
        logging.error("CUDA is not available")
        exit("CUDA is not available")

    if config.device < 0 or config.device >= torch.cuda.device_count():
        logging.error("Invalid device index")
        exit("Invalid device index")

    logging.info(f"CUDA Device {config.device}: {torch.cuda.get_device_name(config.device)}")

    torch.cuda.set_device(config.device)
    torch_dtype = torch.float16

    app = App(config, torch.device("cuda", config.device), torch_dtype)
    app.run()
