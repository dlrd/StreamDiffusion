"""Wire packets for the Smode <-> Python IPC protocol."""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from typing import Dict, Optional

from .protocol import (
    ENDIAN_FORMAT,
    MAGIC_NUMBER,
    UINT32,
    INT32,
    UINT64,
    FLOAT32,
    CommandType,
    Mode,
    ConfigType,
    Acceleration,
    config_type_to_str,
)
from .socket_helpers import read_string

from config.schema import (
    ControlNetConfig,
    CNConfig,
    CannyConfig,
    DepthConfig,
    OpenPoseConfig,
    SimilarImageFilterConfig,
)


# Enum index -> string, mirroring the plugin C++ enums.
PREVIEW_MODE_MAP = {
    0: "normal",
    1: "canny_preview",
    2: "depth_preview",
    3: "openpose_preview",
}
DEPTH_MODEL_SIZE_MAP = {0: "small", 1: "base", 2: "large"}


class Packet:
    def __init__(self, cmd: CommandType, payload: bytes):
        self.cmd = cmd
        self.payload = payload

    def to_bytes(self) -> bytes:
        payload_bytes = (
            struct.pack(ENDIAN_FORMAT + UINT32, self.cmd.value) + self.payload
        )
        size = len(payload_bytes)
        header = struct.pack(ENDIAN_FORMAT + UINT32 + UINT32, MAGIC_NUMBER, size)
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
        payload += struct.pack(ENDIAN_FORMAT + UINT32, len(event_handle)) + event_handle
        payload += struct.pack(
            ENDIAN_FORMAT + UINT32 + UINT32 + UINT32 + UINT32 + UINT32,
            storage_size_bytes,
            storage_offset_bytes,
            channels,
            w,
            h,
        )
        super().__init__(cmd, payload)


# LRU cache for parsed CONFIG payloads: identical bytes are common because
# Smode resends on every parameter change.
_CONFIG_CACHE: "OrderedDict[str, ConfigPacket]" = OrderedDict()
_CONFIG_CACHE_MAX_SIZE = 28


def _parse_config_with_cache(data: bytes):
    """Parse a config packet with LRU caching keyed on payload hash."""
    data_hash = hashlib.md5(data).hexdigest()

    if data_hash in _CONFIG_CACHE:
        _CONFIG_CACHE.move_to_end(data_hash)
        return _CONFIG_CACHE[data_hash]

    config_packet = ConfigPacket()
    config_packet.from_bytes(data)

    if len(_CONFIG_CACHE) >= _CONFIG_CACHE_MAX_SIZE:
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
        # Index 0 = last timestep with maximum noise: correct starting point
        # for 1-step distilled models (Hyper-SDXL, SD-Turbo, Lightning).
        self.t_index_list = [0]
        self.guidance_scale = 5.0
        self.mode = Mode.IMAGE_TO_IMAGE
        self.cfg_type = "none"
        self.acceleration = Acceleration.NONE
        self.lora_dict: Optional[Dict[str, float]] = None
        self.similar_image_filter_config: Optional[SimilarImageFilterConfig] = None
        self.controlnet_config = ControlNetConfig(
            controlnet=CNConfig(),
            canny=CannyConfig(),
            depth=DepthConfig(),
            openpose=OpenPoseConfig(),
        )
        
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
            raise ValueError("Insufficient data for seed, width, and height in CONFIG")
        (self.seed,) = struct.unpack_from(ENDIAN_FORMAT + UINT64, data, offset)
        offset += 8
        t_index_list_len = 0
        (
            self.width,
            self.height,
            t_index_list_len,
        ) = struct.unpack_from(ENDIAN_FORMAT + UINT32 + UINT32 + UINT32, data, offset)
        offset += 12
        for _ in range(t_index_list_len):
            if offset + 4 > len(data):
                raise ValueError("Insufficient data for t_index_list in CONFIG")
            (t_index_value,) = struct.unpack_from(ENDIAN_FORMAT + UINT32, data, offset)
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
        if offset < len(data):
            self.similar_image_filter_config, offset = self.parse_similar_image_filter(
                data, offset
            )

        if offset + 4 <= len(data):
            (lora_dict_len,) = struct.unpack_from(ENDIAN_FORMAT + UINT32, data, offset)
            offset += 4
            self.lora_dict = {}
            for _ in range(lora_dict_len):
                if offset + 4 > len(data):
                    raise ValueError("Insufficient data for lora_dict key length")
                key, offset = read_string(data, offset)
                if offset + 4 > len(data):
                    raise ValueError("Insufficient data for lora_dict value")
                (value,) = struct.unpack_from(ENDIAN_FORMAT + FLOAT32, data, offset)
                self.lora_dict[key] = value
                offset += 4
        else:
            self.lora_dict = None

        if offset < len(data):
            self.controlnet_config.controlnet, offset = self.parse_controlnet_config(
                data, offset
            )

        if offset < len(data):
            self.controlnet_config.canny, offset = self.parse_canny_config(data, offset)

        if offset < len(data):
            self.controlnet_config.depth, offset = self.parse_depth_config(data, offset)

        if offset < len(data):
            self.controlnet_config.openpose, offset = self.parse_openpose_config(data, offset)

        if offset < len(data):
            offset = self.parse_streamv2v_settings(data, offset)

        return self

    def parse_similar_image_filter(self, data: bytes, offset: int):
        SIZE = 12

        if offset + SIZE > len(data):
            raise ValueError("Insufficient data for SimilarImageFilterConfig")

        enabled, threshold, max_skip = struct.unpack_from(
            ENDIAN_FORMAT + UINT32 + FLOAT32 + UINT32,
            data,
            offset,
        )

        return (
            SimilarImageFilterConfig(
                enabled=bool(enabled),
                threshold=threshold,
                max_skip=max_skip,
            ),
            offset + SIZE,
        )

    def parse_controlnet_config(self, data: bytes, offset: int):
        SIZE = 16

        if offset + SIZE > len(data):
            raise ValueError("Insufficient data for controlnet_config length")
        (
            enabled,
            preview_mode,
            guidance_strength,
            skip_frames,
        ) = struct.unpack_from(
            ENDIAN_FORMAT +
            UINT32 +
            UINT32 +
            FLOAT32 +
            UINT32,
            data,
            offset,
        )
        return (
            CNConfig(
                enabled=bool(enabled),
                preview_mode=PREVIEW_MODE_MAP.get(preview_mode, "normal"),
                guidance_strength=guidance_strength,
                skip_frames=skip_frames,
            ),
            offset + SIZE,
        )

    def parse_canny_config(self, data: bytes, offset: int):
        SIZE = 24

        if offset + SIZE > len(data):
            raise ValueError("Insufficient data for CannyConfig")

        (
            enabled,
            scale,
            resolution,
            low_threshold,
            high_threshold,
            l2_gradient,
        ) = struct.unpack_from(
            ENDIAN_FORMAT
            + UINT32
            + FLOAT32
            + UINT32
            + UINT32
            + UINT32
            + UINT32,
            data,
            offset,
        )

        return (
            CannyConfig(
                enabled=bool(enabled),
                scale=scale,
                resolution=resolution,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                l2_gradient=bool(l2_gradient),
            ),
            offset + SIZE,
        )

    def parse_depth_config(self, data: bytes, offset: int):
        SIZE = 40

        if offset + SIZE > len(data):
            raise ValueError("Insufficient data for DepthConfig")
        (
            enabled,
            scale,
            resolution,
            blur_kernel,
            contrast,
            brightness,
            near_threshold,
            far_threshold,
            invert,
            model_size,
        ) = struct.unpack_from(
            ENDIAN_FORMAT
            + UINT32
            + FLOAT32
            + UINT32
            + UINT32
            + FLOAT32
            + INT32
            + INT32
            + UINT32
            + UINT32
            + UINT32,
            data,
            offset,
        )

        return (
            DepthConfig(
                enabled=bool(enabled),
                scale=scale,
                method="grayscale",
                model_size=DEPTH_MODEL_SIZE_MAP.get(model_size, "small"),
                resolution=resolution,
                blur_kernel=blur_kernel,
                contrast=contrast,
                brightness=brightness,
                near_threshold=near_threshold,
                far_threshold=far_threshold,
                invert=bool(invert),
            ),
            offset + SIZE,
        )

    def parse_openpose_config(self, data: bytes, offset: int):
        SIZE = 12

        if offset + SIZE > len(data):
            raise ValueError("Insufficient data for OpenPoseConfig")
        (
            enabled,
            scale,
            detect_resolution,
        ) = struct.unpack_from(
            ENDIAN_FORMAT + UINT32 + FLOAT32 + UINT32,
            data,
            offset,
        )

        return (
            OpenPoseConfig(
                enabled=bool(enabled),
                scale=scale,
                detect_resolution=detect_resolution,
            ),
            offset + SIZE,
        )

    def parse_streamv2v_settings(self, data: bytes, offset: int):
        SIZE = 12
        if offset + SIZE > len(data):
            raise ValueError("Insufficient data for StreamV2V settings")
        (enabled, cache_maxframes, cache_interval) = struct.unpack_from(
            ENDIAN_FORMAT + UINT32 + UINT32 + UINT32,
            data,
            offset,
        )
        self.controlnet_config.streamv2v.enabled = bool(enabled)
        self.controlnet_config.streamv2v.cache_maxframes = cache_maxframes
        self.controlnet_config.streamv2v.cache_interval = cache_interval
        return offset + SIZE


class UuidPacket(Packet):
    def __init__(self, uuid: str):
        payload = struct.pack(ENDIAN_FORMAT + UINT32, len(uuid)) + uuid.encode("utf-8")
        super().__init__(CommandType.UUID, payload)


class StreamCreationPacket(Packet):
    def __init__(self, finished: bool):
        payload = struct.pack(ENDIAN_FORMAT + UINT32, int(finished))
        super().__init__(CommandType.STREAM_CREATION, payload)


class WarningPacket(Packet):
    def __init__(self, active: bool, message: str = ""):
        message_bytes = message.encode("utf-8")
        payload = struct.pack(ENDIAN_FORMAT + UINT32, int(active))
        payload += struct.pack(ENDIAN_FORMAT + UINT32, len(message_bytes)) + message_bytes
        super().__init__(CommandType.WARNING, payload)
