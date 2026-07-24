"""Smode <-> StreamDiffusion orchestrator (SD 1.5 / SDXL).

Responsibilities:
  - IPC server (CUDA shared memory + Win32 events + named pipes)
  - Main real-time inference loop
  - Config file hot-reload (ControlNet toggles, scales, FaceID, ...)
  - StreamDiffusion engine lifecycle
  - Preview modes (canny / depth / openpose / mask / masked_image)
"""
import os
import sys

from config.schema import (ControlNetConfig, CNConfig, CannyConfig, OpenPoseConfig, DepthConfig, SimilarImageFilterConfig)
from engines.streamdiffusion.base_wrapper import BaseStreamDiffusionWrapper
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Fail fast when offline (models are normally cached locally).
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '3')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '10')

# PyTorch CUDA allocator: expandable segments + capped split size eliminate
# the fragmentation that drifts VRAM peaks up over long sessions. Must be
# set before any torch import. ~1-2 GB peak VRAM saved on heavy SDXL stacks.
os.environ.setdefault(
    'PYTORCH_CUDA_ALLOC_CONF',
    'expandable_segments:True,max_split_size_mb:512',
)

import socket
import time
import select
from pathlib import Path
from typing import Dict, Optional
import torch
import argparse
import logging

from engines import StreamDiffusionEngine

from utils.low_latency import LowLatencyController

from preprocessors.processors.ip_adapter_processor import IPAdapterFaceIDProcessor
from preprocessors.processors.canny_processor import CannyProcessor
from preprocessors.processors.depth_processor import DepthProcessor
from preprocessors.processors.openpose_processor import OpenPoseProcessor
from preprocessors.orchestrator import PreprocessorOrchestrator

PACKAGE_DIR = Path(__file__).resolve().parent
from pipeline import StreamDiffusion
from diffusers import ControlNetModel
from controlnet import ControlNetManager
import win32event

import struct

from ipc import (
    InterProcessEvent,
    CommandType, Mode, Acceleration, ConfigType, config_type_to_str, Args,
    Packet, FrameDataPacket, ConfigPacket, UuidPacket, StreamCreationPacket, WarningPacket,
    _parse_config_with_cache,
    StreamDiffusionSmodeTexture,
    recv_all, recv_message, send_message, read_string, is_socket_connected,
    MAGIC_NUMBER, ENDIAN_FORMAT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class App:
    def __init__(
        self, config: Args, device: torch.device, torch_dtype: torch.dtype
    ):
        self.stream = None
        self.cache_dir = None
        self.socket = None
        self._rx_buffer = b""
        self.streamDiffusionToSmodeInterProcessEvent = None
        self.smodeToStreamDiffusionInterProcessEvent = None

        try:
            self.config = config
            self.device = device
            self.torch_dtype = torch_dtype
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 1MB socket buffers reduce syscall overhead for tensor transfers.
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            self.model_name = config.model
            self.current_prompt = ""
            self.negative_prompt = ""
            self.seed = 8
            self.guidance_scale = 1.2
            self.width = config.width
            self.height = config.height
            self.input_tensors = None
            self.output_tensors = None
            # Index 0 = last timestep with max noise (correct for 1-step distilled models).
            self.t_index_list = [0]
            self.mode = Mode.IMAGE_TO_IMAGE
            self.acceleration = Acceleration.NONE
            self.cfg_type = "self" if self.mode == Mode.IMAGE_TO_IMAGE else "none"
            self.lora_dict: Dict[str, float] = None
            self.engine = None
            self.is_sdxl = False


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
            logging.error(f"Failed to initialize App: {e}")
            if self.socket:
                try:
                    self.socket.close()
                except Exception as cleanup_error:
                    logging.debug(f"Socket cleanup error (non-critical): {cleanup_error}")
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
            raise

        self.controlnet_manager = ControlNetManager(self)

        self.faceid_processor: Optional[IPAdapterFaceIDProcessor] = None

        self.preprocessor_orchestrator: Optional[PreprocessorOrchestrator] = None

        self.low_latency = LowLatencyController()

        self.similar_image_filter = SimilarImageFilterConfig()

        self.controlnet_config = ControlNetConfig(
                    controlnet = CNConfig(),
                    canny=CannyConfig(),
                    depth=DepthConfig(),
                    openpose=OpenPoseConfig(),
                )
        
        self._cache_config_values(self.controlnet_config)
        
        self.controlnet_skip_frames = self.controlnet_config.controlnet.skip_frames
        self.current_delta = 1.0

        # torch.compile warmup runs once at startup, not on every prepare() call.
        self.warmup_completed = False
        # Frame trig consumed (auto-reset event) while the stream was still loading.
        self._pending_frame = False
        self.canny_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        # NOTE: torch.cuda.set_per_process_memory_fraction(0.95) was REMOVED.
        # It hard-capped the caching allocator at ~15.2 GB on the 16 GB card and
        # raised a PyTorch OOM BEFORE WDDM could oversubscribe/page — turning
        # survivable pressure (SDXL+V2V at 1024) into a hard crash. Its comment
        # ("reserve 5% for pinned memory") was also wrong (pinned memory is host
        # RAM). Allocator caps stay OFF under WDDM; the driver handles fragmentation.
        if torch.cuda.is_available():
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True

        self._init_connection()
        self._create_tensors(3, self.width, self.height)

    def _send_warning(self, active: bool, message: str = "") -> None:
        try:
            send_message(self.socket, WarningPacket(active, message))
        except Exception as e:
            logging.warning(f"[Warning] Failed to send warning packet: {e}")

    def _create_stream(self):
        send_message(self.socket, StreamCreationPacket(False))

        # New stream = new UNet: warmup (and StreamV2V install) must rerun.
        self.warmup_completed = False
        self._streamv2v_active = False

        if self.stream is not None:
            logging.info(f"[Engine] Freeing previous engine...")
            if self.engine is not None:
                try:
                    self.engine.cleanup()
                except Exception as e:
                    logging.warning(f"[Engine] cleanup() raised: {e}")
                self.engine = None
            self.stream = None
            # gc.collect() BEFORE empty_cache(): the diffusers pipe<->components
            # <->attn-processor graph (and _faceid_pipe_ref back-refs) holds
            # reference cycles that only gc breaks; flushing the cache first
            # leaves cycle-held tensors alive so their blocks aren't released
            # before the next engine's (TRT) allocations. Matches the rest of
            # the codebase's teardown order.
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        if self.acceleration == Acceleration.TENSORRT:
            acceleration_str = "tensorrt"
        elif self.acceleration == Acceleration.TORCHCOMPILE:
            acceleration_str = "torchcompile"
        else:
            acceleration_str = "none"

        self.engine = StreamDiffusionEngine()
        self.engine.load(
            self.controlnet_config,
            model_name=self.model_name,
            t_index_list=self.t_index_list,
            lora_dict=self.lora_dict,
            mode_is_img2img=(self.mode == Mode.IMAGE_TO_IMAGE),
            width=self.width,
            height=self.height,
            seed=self.seed,
            cfg_type=self.cfg_type,
            torch_dtype=self.torch_dtype,
            device=self.device,
            cache_dir=self.cache_dir,
            acceleration_str=acceleration_str,
            warning_callback=self._send_warning,
        )

        self.stream = self.engine.wrapper

        self.num_inference_steps = self.engine.num_inference_steps
        self.is_sdxl = getattr(self.engine, "is_sdxl", False)
        self.is_sd2 = getattr(self.engine, "is_sd2", False)

        # pipeline.last_internal_timings only present when profiling instrumentation is on.
        self._stream_has_internal_timings = (
            hasattr(self.stream, "stream")
            and hasattr(self.stream.stream, "last_internal_timings")
        )
        _inner = getattr(self.stream, "stream", None)
        self._ssf_enabled_cached = (
            _inner is not None
            and getattr(_inner, "similar_image_filter", False)
            and getattr(_inner, "similar_filter", None) is not None
        )

        self._post_load_streamdiffusion()

        if hasattr(self.stream, "stream"):
            self.stream.stream._cached_controlnet_guidance_strength = self._cached_controlnet_guidance_strength

        self._create_tensors(3, self.width, self.height)
        send_message(self.socket, StreamCreationPacket(True))

    def _post_load_streamdiffusion(self):
        """Post-load setup: FaceID processor + modular preprocessor orchestrator."""
        if self.faceid_processor is not None:
            try:
                self.faceid_processor.cleanup()
            except Exception:
                pass
            self.faceid_processor = None

        faceid_enabled = self.controlnet_config.get('faceid_enabled', False)
        faceid_wrapper_loaded = getattr(self.stream, '_faceid_loaded', False)

        if faceid_enabled and faceid_wrapper_loaded:
            try:
                self.faceid_processor = IPAdapterFaceIDProcessor(self.device, self.torch_dtype)
                self.faceid_processor.load_model(self.controlnet_config)
                plus_v2 = self.controlnet_config.get('faceid_plus_v2', False)
                if hasattr(self.stream, '_faceid_pipe_ref'):
                    self.faceid_processor.attach_pipe(self.stream._faceid_pipe_ref, plus_v2=plus_v2)
                variant = "PlusV2" if plus_v2 else "Base"
                logging.info(f"[FaceID] Processor initialized (variant={variant})")
            except Exception as e:
                logging.error(f"[FaceID] Failed to initialize processor: {e}")
                import traceback
                traceback.print_exc()
                self.faceid_processor = None
        elif faceid_enabled and not faceid_wrapper_loaded:
            logging.warning("[FaceID] Enabled in config but IP-Adapter failed to load in wrapper")

        # Initialize modular preprocessor orchestrator (only preprocessing path).
        # Free the PREVIOUS orchestrator first (mirrors __del__ and the FaceID
        # cleanup above): on a recreation it owns Depth-Anything (~0.4-1.5 GB),
        # DWPose, YOLO/SAM2 on GPU, and was previously left to deferred GC —
        # double-resident with the freshly loaded models at the peak-VRAM moment
        # of _create_stream.
        if self.preprocessor_orchestrator is not None:
            try:
                self.preprocessor_orchestrator.cleanup()
            except Exception as e:
                logging.debug(f"Orchestrator cleanup error (non-critical): {e}")
            self.preprocessor_orchestrator = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logging.info("[Orchestrator] Initializing modular preprocessors...")
        self.preprocessor_orchestrator = PreprocessorOrchestrator(self.device, self.torch_dtype)

        self.preprocessor_orchestrator.register('canny', CannyProcessor(
            self.device, self.torch_dtype, warning_callback=self._send_warning))
        self.preprocessor_orchestrator.register('depth', DepthProcessor(
            self.device, self.torch_dtype, warning_callback=self._send_warning))
        self.preprocessor_orchestrator.register('openpose', OpenPoseProcessor(
            self.device, self.torch_dtype, warning_callback=self._send_warning))

        depth_proc = self.preprocessor_orchestrator._processors.get('depth')
        if depth_proc:
            depth_proc.set_compile_fn(self.controlnet_manager.compile)
            # Skip the ~1.5 GB HF model load when a Depth-Anything TRT engine is already cached.
            depth_proc.set_try_cache_fn(
                self.controlnet_manager.try_load_depth_anything_cache
            )

        self.preprocessor_orchestrator.update_models(self.controlnet_config)

        logging.info("[Orchestrator] Modular preprocessors ready")

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
            self.socket.settimeout(30)  # fail fast if host is slow to bind
            server_address = ("127.0.0.1", self.config.port)
            logging.info(f"Connecting to server at {server_address}")
            self.socket.connect(server_address)
            self._send_uuid()
            self.socket.settimeout(None)
            self.socket.setblocking(False)
        except socket.error as e:
            logging.error(f"Socket error during connection: {e}")
            self.socket.close()
            raise

    def _send_uuid(self):
        packet = UuidPacket(self.config.uuid)
        send_message(self.socket, packet)

    def accelerate(self, previous_acceleration: Acceleration = Acceleration.NONE):
        if self.acceleration == Acceleration.TORCHCOMPILE:
            # torch.compile is activated during stream creation via setup_torch_compile()
            # in the wrapper (base_wrapper.py). Ensure it's properly initialized.
            if previous_acceleration == Acceleration.TENSORRT:
                # Switching from TensorRT requires full stream recreation
                self._create_stream()
            else:
                # Already in PyTorch mode; just recreate pipe with torch.compile active
                self.stream.recreate_pipe()
            logging.info("Torch.compile acceleration enabled")
        elif self.acceleration == Acceleration.TENSORRT:
            try:
                if previous_acceleration == Acceleration.TORCHCOMPILE:
                    # Switching from torch.compile to TensorRT requires full stream recreation
                    self._create_stream()
                else:
                    self.stream.enable_tensorrt_acceleration(self.stream.stream, self.model_name, True, True)

                self._warmup_tensorrt()
                logging.info("TensorRT acceleration enabled")
            except ModuleNotFoundError:
                logging.warning("TensorRT module not found; please install it")
                raise
            except Exception as e:
                logging.warning(f"TensorRT acceleration not available; {e}")
        else:
            # No acceleration (NONE) - use default PyTorch
            self.stream.recreate_pipe()
            logging.info("Acceleration disabled (PyTorch default)")

    def _warmup_tensorrt(self):
        """Warm up TensorRT engine with dummy inferences."""
        dummy_input = None
        try:
            logging.info("Warming up TensorRT engine...")
            warmup_iterations = max(len(self.t_index_list) * self.stream.stream.frame_bff_size, 10)

            dummy_input = torch.randn(
                (1, 3, self.height, self.width),
                dtype=self.torch_dtype,
                device=self.device
            )

            for _ in range(warmup_iterations):
                _ = self.stream.stream(image=dummy_input)

            logging.info(f"TensorRT warmup complete ({warmup_iterations} iterations)")
        except Exception as e:
            logging.warning(f"TensorRT warmup failed (non-critical): {e}")
        finally:
            if dummy_input is not None:
                del dummy_input
                torch.cuda.empty_cache()

    def _cache_config_values(self, config: ControlNetConfig):
        self._cached_canny_scale = config.canny.scale
        self._cached_depth_scale = config.depth.scale
        self._cached_openpose_scale = config.openpose.scale

        self._cached_profiling_enabled = config.profiling_enabled
        self._cached_preview_mode = config.controlnet.preview_mode

        self._cached_controlnet_enabled = config.controlnet.enabled
        self._cached_canny_enabled = config.canny.enabled
        self._cached_depth_enabled = config.depth.enabled
        self._cached_openpose_enabled = config.openpose.enabled

        self._cached_controlnet_skip_frames = config.controlnet.skip_frames
        self._cached_controlnet_guidance_strength = config.controlnet.guidance_strength

        self._cached_openpose_detect_resolution = (
            config.openpose.detect_resolution
        )

        if self.stream is not None:
            self.controlnet_manager.load_models()
        self.controlnet_manager.update_active_list()

    def _apply_live_config(self):
        if not (self.stream and hasattr(self.stream, 'stream')):
            return
        inner = self.stream.stream

        sif = getattr(self, 'similar_image_filter', None)
        if sif is not None:
            if sif.enabled:
                inner.enable_similar_image_filter(sif.threshold, sif.max_skip)
            else:
                inner.disable_similar_image_filter()
            self._ssf_enabled_cached = (
                getattr(inner, "similar_image_filter", False)
                and getattr(inner, "similar_filter", None) is not None
            )

        if self.preprocessor_orchestrator is not None:
            self.preprocessor_orchestrator.update_models(self.controlnet_config)

        new_strength = self._cached_controlnet_guidance_strength
        if getattr(inner, '_cached_controlnet_guidance_strength', None) != new_strength:
            inner._cached_controlnet_guidance_strength = new_strength
            if hasattr(inner, '_guidance_strength_logged'):
                delattr(inner, '_guidance_strength_logged')

    def _receive_pending_messages(self) -> dict:
        """Drain pending control messages; never blocks, partial messages stay buffered."""
        messages = {}
        while True:
            ready_to_read, _, in_error = select.select([self.socket], [], [], 0)
            if in_error:
                logging.error("Socket reported an exceptional condition")
            if not ready_to_read:
                break
            try:
                chunk = self.socket.recv(65536)
            except socket.error as e:
                # WinError 10035 = non-blocking socket has no data (normal).
                if e.errno != 10035:
                    logging.warning(f"Socket receive error: {e}")
                break
            if not chunk:
                break
            self._rx_buffer += chunk

        buf = self._rx_buffer
        offset = 0
        while len(buf) - offset >= 8:
            magic, size = struct.unpack_from(ENDIAN_FORMAT + "II", buf, offset)
            if magic != MAGIC_NUMBER:
                logging.error(f"Invalid magic number received: {hex(magic)}")
                offset = len(buf)  # drop garbage to resync
                break
            if len(buf) - offset < 8 + size:
                break  # incomplete message: keep buffered, frame loop goes on
            payload = buf[offset + 8: offset + 8 + size]
            offset += 8 + size
            if len(payload) < 4:
                continue
            cmd_int, = struct.unpack(ENDIAN_FORMAT + "I", payload[:4])
            try:
                cmd = CommandType(cmd_int)
            except ValueError:
                logging.error(f"Unknown command code received: {cmd_int}")
                continue
            messages[cmd] = payload[4:]
        self._rx_buffer = buf[offset:]
        return messages

    def _handle_pending_commands(self, messages: dict) -> bool:
        """Process incoming CONFIG / STOP messages. Returns True on STOP."""
        for cmd, payload in messages.items():
            if cmd == CommandType.CONFIG:
                config_packet = _parse_config_with_cache(payload)    

                def update_parameters(app: App, config_packet: ConfigPacket):
                    app.model_name = config_packet.model_name
                    app.current_prompt = config_packet.prompt
                    app.negative_prompt = config_packet.negative_prompt
                    app.seed = config_packet.seed
                    app.width = config_packet.width
                    app.height = config_packet.height
                    # Guard against an empty / out-of-range timestep list (host bug) so the scheduler can't crash.
                    app.t_index_list = [t for t in config_packet.t_index_list if 0 <= t < 50] or [1]
                    app.guidance_scale = config_packet.guidance_scale
                    app.mode = config_packet.mode
                    app.cfg_type = config_packet.cfg_type
                    app.lora_dict = config_packet.lora_dict
                    app.acceleration = config_packet.acceleration
                    app.cache_dir = config_packet.cache_dir
                    app.similar_image_filter = config_packet.similar_image_filter_config
                    app.controlnet_config = config_packet.controlnet_config

                if not self.stream:
                    prepare_needed = True
                    prompt_only_change = False
                    guidance_live_change = False
                    update_parameters(self, config_packet)
                    # self.apply_controlnet_config(config_packet.controlnet_config)
                    self._cache_config_values(self.controlnet_config)
                    self._create_stream()
                    self.controlnet_manager.load_models()
                    self.controlnet_manager.update_active_list()
                else:
                    model_has_changed = self.model_name != config_packet.model_name
                    lora_dict_has_changed = self.lora_dict != config_packet.lora_dict
                    inner_stream = getattr(self.stream, "stream", None)
                    current_cfg_type = (
                        inner_stream.cfg_type if inner_stream is not None else None
                    )
                    update_stream = (
                        self.width != config_packet.width
                        or self.height != config_packet.height
                        or self.mode != config_packet.mode
                        or current_cfg_type != config_packet.cfg_type
                        or self.acceleration != config_packet.acceleration
                        or self.lora_dict != config_packet.lora_dict
                    )
                    # Compare filtered vs filtered (raw out-of-range values = spurious change).
                    _new_t_index_list = [t for t in config_packet.t_index_list if 0 <= t < 50] or [1]
                    update_t_index_list = self.t_index_list != _new_t_index_list
                    prompt_changed = self.current_prompt != config_packet.prompt
                    negative_changed = self.negative_prompt != config_packet.negative_prompt
                    guidance_changed = self.guidance_scale != config_packet.guidance_scale
                    seed_changed = self.seed != config_packet.seed
                    # Crossing 1.0 changes the embed layout -> prepare(); same-side = live scalar.
                    guidance_threshold_crossed = (
                        (self.guidance_scale > 1.0) != (config_packet.guidance_scale > 1.0)
                    )
                    structural_change = (
                        model_has_changed
                        or lora_dict_has_changed
                        or update_stream
                        or update_t_index_list
                        or seed_changed
                        or (guidance_changed and guidance_threshold_crossed)
                    )
                    # Prompt-only change -> lightweight update_prompt(); prepare() if CFG concat or negative changed.
                    cfg_uses_concat = config_packet.cfg_type in ("full", "initialize")
                    prepare_needed = (
                        structural_change
                        or negative_changed
                        or (prompt_changed and cfg_uses_concat)
                    )
                    prompt_only_change = prompt_changed and not prepare_needed
                    # guidance_scale is read live in unet_step -> applied in place.
                    guidance_live_change = (
                        guidance_changed and not guidance_threshold_crossed and not prepare_needed
                    )
                    previous_acceleration = self.acceleration
                    update_parameters(self, config_packet)
                    self._cache_config_values(self.controlnet_config)

                    if model_has_changed or lora_dict_has_changed:
                        self._create_stream()
                        self.accelerate(previous_acceleration)
                    elif update_stream:
                        # Inline recreation for width/height/mode/cfg changes.
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
                        self.stream.stream._cached_controlnet_guidance_strength = self._cached_controlnet_guidance_strength
                        self.accelerate(previous_acceleration)
                    elif update_t_index_list:
                        new_len = len(self.t_index_list)
                        old_len = self.stream.stream.denoising_steps_num
                        # TRT bakes batch = denoising_steps x frame_buffer statically;
                        # rebuild when step count changes. PyTorch handles dynamic batch.
                        if (new_len != old_len
                                and self.acceleration == Acceleration.TENSORRT):
                            logging.info(
                                f"[Engine] Denoising steps {old_len}->{new_len} "
                                f"changes TRT batch; recreating stream."
                            )
                            self._create_stream()
                        else:
                            self.stream.stream.t_list = self.t_index_list
                            self.stream.stream.denoising_steps_num = new_len

                if self.stream and hasattr(self.stream, 'stream'):

                    if prepare_needed:
                        self.stream.stream.prepare(
                            self.current_prompt,
                            self.negative_prompt,
                            num_inference_steps=self.num_inference_steps,
                            guidance_scale=config_packet.guidance_scale,
                            delta=self.current_delta,
                            seed=self.seed,
                        )
                    elif prompt_only_change:
                        self.stream.stream.update_prompt(self.current_prompt)
                    if guidance_live_change:
                        self.stream.stream.guidance_scale = self.guidance_scale

                    self._apply_live_config()

                    # StreamV2V: reset attention cache only on prompt change.
                    if getattr(self, '_streamv2v_active', False):
                        prev_prompt = getattr(self, '_streamv2v_last_prompt', None)
                        if prev_prompt != self.current_prompt:
                            from pipeline.attention_processors import reset_attention_cache
                            reset_attention_cache(self.stream.stream.unet)
                            self._streamv2v_last_prompt = self.current_prompt
                            logging.info("[StreamV2V] Attention cache reset (prompt changed)")

                    # Warmup: trigger torch.compile compilation before first user frame.
                    # Only runs on the very first prepare() call.
                    if not self.warmup_completed:
                        # StreamV2V install: only on PyTorch UNets. TRT v2v engines
                        # carry the kvo cache as engine I/O internally; non-v2v TRT
                        # engines (e.g. SDXL) can't host StreamV2V and would crash
                        # the named_modules() walk in enable_cached_attention.
                        if self.controlnet_config.get('streamv2v_enabled', False):
                            unet_obj = self.stream.stream.unet
                            trt_v2v_engine = getattr(unet_obj, '_is_v2v', False)
                            is_pytorch_unet = hasattr(unet_obj, 'named_modules')
                            if trt_v2v_engine:
                                self._streamv2v_active = True
                                logging.info(
                                    "[StreamV2V] TRT engine has kvo cache as engine I/O; "
                                    "skipping PyTorch attention processor install."
                                )
                            elif not is_pytorch_unet:
                                self._streamv2v_active = False
                                logging.warning(
                                    "[StreamV2V] Active TRT engine does not support v2v. "
                                    "Set streamv2v_enabled=false or switch to SD 1.5."
                                )
                            else:
                                from pipeline.attention_processors import enable_cached_attention
                                maxframes = self.controlnet_config.streamv2v.cache_maxframes
                                interval = self.controlnet_config.streamv2v.cache_interval
                                count = enable_cached_attention(
                                    self.stream.stream.unet,
                                    cache_maxframes=maxframes,
                                    cache_interval=interval,
                                    height=self.height,
                                    width=self.width,
                                    batch_size=self.stream.stream.batch_size,
                                    device=self.device,
                                    dtype=self.torch_dtype,
                                )
                                self._streamv2v_active = True
                                logging.info(f"[StreamV2V] Pre-installed on {count} layers before warmup")

                        is_torch_compile_warmup = self.acceleration == Acceleration.TORCHCOMPILE
                        if is_torch_compile_warmup:
                            self._send_warning(True, "Warming up torch.compile - first frame(s) can take 1-2 minutes")
                        logging.info("Warming up torch.compile cache (U-Net + VAE)...")
                        warmup_start = time.time()
                        try:
                            dummy_input = torch.randn(
                                (1, 3, self.height, self.width),
                                dtype=self.torch_dtype,
                                device=self.device
                            )

                            # When IP-Adapter is loaded the UNet requires image_embeds.
                            dummy_faceid_embeds = None
                            faceid_wrapper_loaded = getattr(self.stream, '_faceid_loaded', False)
                            if faceid_wrapper_loaded:
                                dummy_faceid_embeds = [torch.zeros(
                                    1, 1, 512, dtype=self.torch_dtype, device=self.device
                                )]

                            for i in range(2):
                                torch.compiler.cudagraph_mark_step_begin()
                                _ = self.stream(
                                    image=dummy_input,
                                    controlnet_image=None,
                                    controlnet_model=None,
                                    controlnet_conditioning_scale=1.0,
                                    ip_adapter_image_embeds=dummy_faceid_embeds,
                                )

                            torch.cuda.synchronize()
                            warmup_time = time.time() - warmup_start
                            logging.info(f"Warmup complete ({warmup_time:.1f}s)")

                            try:
                                _alloc = torch.cuda.memory_allocated() / 1e9
                                _reserv = torch.cuda.memory_reserved() / 1e9
                                logging.info(
                                    f"[VRAM] after warmup: allocated={_alloc:.2f} GB, "
                                    f"reserved={_reserv:.2f} GB (gap={_reserv - _alloc:.2f} GB)"
                                )
                            except Exception:
                                pass

                            # Warm up U-Net+ControlNet combination separately.
                            cn_models_dict = self.controlnet_manager.models
                            if cn_models_dict:
                                logging.info(f"Warming up U-Net+ControlNet ({len(cn_models_dict)} active)...")
                                cn_warmup_start = time.time()

                                cn_images = []
                                cn_models = []
                                cn_scales = []

                                for cn_name in ['canny', 'depth', 'openpose']:
                                    if cn_name in cn_models_dict:
                                        cn_images.append(dummy_input)
                                        cn_models.append(cn_models_dict[cn_name])
                                        cn_scales.append(1.0)

                                for i in range(2):
                                    torch.compiler.cudagraph_mark_step_begin()
                                    _ = self.stream(
                                        image=dummy_input,
                                        controlnet_image=cn_images if cn_images else None,
                                        controlnet_model=cn_models if cn_models else None,
                                        controlnet_conditioning_scale=cn_scales if cn_scales else 1.0,
                                        ip_adapter_image_embeds=dummy_faceid_embeds,
                                    )

                                torch.cuda.synchronize()
                                cn_warmup_time = time.time() - cn_warmup_start
                                logging.info(f"ControlNet warmup complete ({cn_warmup_time:.1f}s)")

                            self.warmup_completed = True

                        except Exception:
                            self.warmup_completed = True
                            import traceback
                            logging.error(f"Warmup failed:\n{traceback.format_exc()}")
                        finally:
                            if 'dummy_input' in locals():
                                del dummy_input
                            if is_torch_compile_warmup:
                                self._send_warning(False)
                                torch.cuda.empty_cache()

            elif cmd == CommandType.STOP:
                logging.info("Received STOP command; exiting main loop")
                if self.stream:
                    del self.stream
                    self.stream = None
                return True
            else:
                logging.warning(f"Received unexpected command: {cmd}")
        return False

    def _recover_from_failed_command(self):
        # Survive a failed CONFIG (parse or load error): log it, release the host
        # spinner, reset to a clean state so the next CONFIG can retry.
        import traceback
        logging.error(f"Command handling failed:\n{traceback.format_exc()}")
        engine = getattr(self, 'engine', None)
        if engine is not None:
            try:
                engine.cleanup()
            except Exception:
                pass
            self.engine = None
        self.stream = None
        self.warmup_completed = False
        self._streamv2v_active = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            send_message(self.socket, StreamCreationPacket(True))
        except Exception:
            pass

    def _process_frame(self, timings: dict) -> None:
        """Per-frame compute: input -> preprocess -> inference -> output -> signal."""
        profiling_enabled = self._cached_profiling_enabled

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
                permuted_input_texture = self.input_tensors.get_permuted_input_tensor()

            if profiling_enabled:
                torch.cuda.synchronize()
                timings['image_preprocess'] = (time.time() - preprocess_start) * 1000

            preview_mode = self._cached_preview_mode

            # SSF upfront gate: skip pipeline + preprocessors when the filter decides to.
            if preview_mode == 'normal' and getattr(self, '_ssf_enabled_cached', False):
                inner_stream = self.stream.stream
                # Match the pipeline's normalization: (C,H,W)[0,1] -> (1,C,H,W)[-1,1].
                # Reuse a persistent buffer (realloc only on resolution
                # change) to avoid a fresh full-frame fp16 alloc + two
                # elementwise kernels every gated frame. decide_skip copies
                # into its own prev_tensor, so reusing this buffer is safe.
                ssf_buf = getattr(self, '_ssf_probe_buf', None)
                if ssf_buf is None or ssf_buf.shape[1:] != permuted_input_texture.shape:
                    ssf_buf = torch.empty(
                        (1,) + tuple(permuted_input_texture.shape),
                        dtype=permuted_input_texture.dtype,
                        device=permuted_input_texture.device,
                    )
                    self._ssf_probe_buf = ssf_buf
                torch.mul(permuted_input_texture, 2.0, out=ssf_buf[0])
                ssf_buf.sub_(1.0)
                if inner_stream.similar_filter.decide_skip(ssf_buf):
                    self.streamDiffusionToSmodeInterProcessEvent.signal()
                    return

            if profiling_enabled:
                controlnet_start = time.time()

            controlnet_processed_dict = None

            if (self._cached_controlnet_enabled and preview_mode == 'normal'
                    and self.preprocessor_orchestrator is not None):
                controlnet_processed_dict = self.preprocessor_orchestrator.preprocess(
                    permuted_input_texture,
                    self.controlnet_config,
                    skip_frames=self._cached_controlnet_skip_frames,
                )

                if controlnet_processed_dict and logging.getLogger().isEnabledFor(logging.DEBUG):
                    active_nets = []
                    if 'canny' in controlnet_processed_dict:
                        active_nets.append(f"Canny (scale: {self._cached_canny_scale})")
                    if 'depth' in controlnet_processed_dict:
                        active_nets.append(f"Depth (scale: {self._cached_depth_scale})")
                    if 'openpose' in controlnet_processed_dict:
                        active_nets.append(f"OpenPose (scale: {self._cached_openpose_scale})")
                    logging.debug(f"ControlNets prepared: {', '.join(active_nets)}")

            if profiling_enabled:
                torch.cuda.synchronize()
                timings['controlnet_preprocess'] = (time.time() - controlnet_start) * 1000
            else:
                timings['controlnet_preprocess'] = 0.0

            # Preview modes: delegate to the matching orchestrator processor.
            if preview_mode == "canny_preview" and self._cached_canny_enabled:
                x_output = self.preprocessor_orchestrator._processors['canny'].process(
                    permuted_input_texture, self.controlnet_config)
            elif preview_mode == "depth_preview" and self._cached_depth_enabled:
                x_output = self.preprocessor_orchestrator._processors['depth'].process(
                    permuted_input_texture, self.controlnet_config)
            elif preview_mode == "openpose_preview" and self._cached_openpose_enabled:
                x_output = self.preprocessor_orchestrator._processors['openpose'].process(
                    permuted_input_texture, self.controlnet_config)
            else:
                # Normal generation path with optional ControlNet conditioning.
                cn_mgr = self.controlnet_manager
                if controlnet_processed_dict is not None and preview_mode == "normal" and cn_mgr.active_keys:
                    controlnet_images = [controlnet_processed_dict[k] for k in cn_mgr.active_keys
                                        if k in controlnet_processed_dict]

                    if cn_mgr.is_union_mode():
                        # SDXL Union: one wrapper consumes the full list of conditioning
                        # images in a single forward pass.
                        controlnet_images = [controlnet_images]
                        controlnet_models = cn_mgr.models_cache
                        controlnet_scales = cn_mgr.scales_cache
                    else:
                        num_images = len(controlnet_images)
                        controlnet_models = cn_mgr.models_cache[:num_images]
                        controlnet_scales = cn_mgr.scales_cache[:num_images]
                else:
                    controlnet_models = []
                    controlnet_images = []
                    controlnet_scales = []

                # Marks a new CUDA Graphs iteration: prevents "overwritten by subsequent run"
                # errors when per-frame tensors (e.g. ControlNet conditioning_scale) change.
                torch.compiler.cudagraph_mark_step_begin()

                ip_adapter_embeds = None
                if self.faceid_processor is not None and self.faceid_processor.is_loaded:
                    ip_adapter_embeds = self.faceid_processor.process(
                        permuted_input_texture, self.controlnet_config
                    )

                x_output = self.stream(
                    image=permuted_input_texture,
                    controlnet_image=controlnet_images if controlnet_images else None,
                    controlnet_model=controlnet_models if controlnet_models else None,
                    controlnet_conditioning_scale=controlnet_scales if controlnet_scales else 1.0,
                    ip_adapter_image_embeds=ip_adapter_embeds,
                )

                if profiling_enabled and self._stream_has_internal_timings:
                    internal = self.stream.stream.last_internal_timings
                    timings['gen_vae_encode'] = internal.get('vae_encode', 0.0)
                    timings['gen_unet_controlnet'] = internal.get('unet_controlnet', 0.0)
                    timings['gen_vae_decode'] = internal.get('vae_decode', 0.0)
                    timings['generation'] = timings['gen_vae_encode'] + timings['gen_unet_controlnet'] + timings['gen_vae_decode']
                else:
                    timings['generation'] = 0.0
        elif self.mode == Mode.TEXT_TO_IMAGE:
            x_output = self.stream.txt2img()

            if profiling_enabled and self._stream_has_internal_timings:
                internal = self.stream.stream.last_internal_timings
                timings['gen_vae_encode'] = internal.get('vae_encode', 0.0)
                timings['gen_unet_controlnet'] = internal.get('unet_controlnet', 0.0)
                timings['gen_vae_decode'] = internal.get('vae_decode', 0.0)
                timings['generation'] = timings['gen_vae_encode'] + timings['gen_unet_controlnet'] + timings['gen_vae_decode']
            else:
                timings['generation'] = 0.0
        else:
            logging.error(f"Unknown mode: {self.mode}")
            # Signal anyway so the host never waits forever on this frame.
            self.streamDiffusionToSmodeInterProcessEvent.signal()
            return

        if x_output is not None:
            x_output = x_output.squeeze(0) if x_output.shape[0] == 1 else x_output

        if profiling_enabled:
            output_copy_start = time.time()
            if x_output is not None:
                self.output_tensors.write_chw_to_smode(x_output)
            torch.cuda.synchronize()
            timings['output_copy'] = (time.time() - output_copy_start) * 1000
        elif x_output is not None:
            self.output_tensors.write_chw_to_smode(x_output)
            # Sync before "done": bounds the async GPU queue (deep queue = ~700ms
            # stall on param edits) and guarantees the frame is fully written.
            torch.cuda.synchronize()


        if profiling_enabled:
            signal_start = time.time()
            self.streamDiffusionToSmodeInterProcessEvent.signal()
            timings['signal_smode'] = (time.time() - signal_start) * 1000
        else:
            self.streamDiffusionToSmodeInterProcessEvent.signal()

    def run(self):
        logging.info("Entering main command loop")

        self.low_latency.apply(
            self.controlnet_config.low_latency_mode
        )

        last_frame_wall_time = 0.0

        frames_received = 0
        frames_processed = 0
        last_diagnostic_time = time.time()

        # The frame trigger is the Win32 event; the socket carries rare control
        # messages. is_socket_connected does a MSG_PEEK syscall - throttle it.
        EVENT_WAIT_MS = 2
        SOCKET_HEALTH_CHECK_EVERY = 32
        loop_iter = 0

        try:
            while True:
                loop_iter += 1
                if loop_iter % SOCKET_HEALTH_CHECK_EVERY == 0 and not is_socket_connected(self.socket):
                    return
                messages = self._receive_pending_messages()

                wait_result = self.smodeToStreamDiffusionInterProcessEvent.wait(EVENT_WAIT_MS)
                frame_signaled = wait_result == win32event.WAIT_OBJECT_0
                if frame_signaled and not self.stream:
                    # Auto-reset event: the trig is consumed here. Remember it and
                    # replay once the stream exists (manual/atPreload trig only once).
                    self._pending_frame = True
                if (frame_signaled or self._pending_frame) and self.stream:
                    self._pending_frame = False
                    frames_received += 1

                    frame_start = time.time()
                    timings = {}

                    if last_frame_wall_time > 0:
                        wall_time_ms = (frame_start - last_frame_wall_time) * 1000
                        timings['wall_clock'] = wall_time_ms
                    last_frame_wall_time = frame_start

                    if self.output_tensors is None:
                        self._create_tensors(3, self.width, self.height)

                    profiling_enabled = self._cached_profiling_enabled

                    self._process_frame(timings)

                    frames_processed += 1
                    self.low_latency.tick()

                    current_time = time.time()
                    if current_time - last_diagnostic_time >= 1.0:
                        elapsed_time = current_time - last_diagnostic_time
                        receive_rate = frames_received / elapsed_time
                        process_rate = frames_processed / elapsed_time
                        skip_rate = ((frames_received - frames_processed) / frames_received * 100) if frames_received > 0 else 0

                        if profiling_enabled:
                            logging.info(f"THROUGHPUT: Received={receive_rate:.1f} fps | Processed={process_rate:.1f} fps | Skipped={skip_rate:.1f}%")

                        frames_received = 0
                        frames_processed = 0
                        last_diagnostic_time = current_time

                    if profiling_enabled:
                        timings['total_frame'] = (
                            timings.get('config_check', 0.0) +
                            timings.get('input_copy', 0.0) +
                            timings.get('image_preprocess', 0.0) +
                            timings.get('controlnet_preprocess', 0.0) +
                            timings.get('generation', 0.0) +
                            timings.get('output_copy', 0.0) +
                            timings.get('signal_smode', 0.0)
                        )
                        smode_feedback_time = abs(timings.get('wall_clock', 1.0) - timings.get('total_frame', 1.0))
                        smode_fps = 1000.0 / smode_feedback_time if smode_feedback_time > 0 else 0

                        if not hasattr(self, '_frame_count'):
                            self._frame_count = 0
                        self._frame_count += 1

                        if self._frame_count % 60 == 0:
                            logging.info(f"PROFILING (Smode Feedback FPS: {smode_fps:.1f}):")

                            has_breakdown = 'gen_vae_encode' in timings
                            exclude_keys = ['total_frame', 'cycle_complete', 'wall_clock', 'gen_vae_encode', 'gen_unet_controlnet', 'gen_vae_decode']
                            if has_breakdown:
                                exclude_keys.append('generation')

                            for key, value in sorted(timings.items()):
                                if key not in exclude_keys:
                                    percentage = (value / timings['total_frame'] * 100) if timings['total_frame'] > 0 else 0
                                    logging.info(f"  {key:25s}: {value:6.2f}ms ({percentage:5.1f}%)")

                            if has_breakdown:
                                percentage = (timings['generation'] / timings['total_frame'] * 100) if timings['total_frame'] > 0 else 0
                                logging.info(f"  {'generation':25s}: {timings['generation']:6.2f}ms ({percentage:5.1f}%) BREAKDOWN:")
                                gen_total = timings['generation']
                                for subkey in ['gen_vae_encode', 'gen_unet_controlnet', 'gen_vae_decode']:
                                    if subkey in timings:
                                        sub_value = timings[subkey]
                                        sub_percentage = (sub_value / gen_total * 100) if gen_total > 0 else 0
                                        label = subkey.replace('gen_', '    +- ')
                                        logging.info(f"  {label:25s}: {sub_value:6.2f}ms ({sub_percentage:5.1f}% of gen)")

                            logging.info(f"  {'total_frame':25s}: {timings['total_frame']:6.2f}ms (REAL GPU processing time)")
                            if 'wall_clock' in timings:
                                smode_idle = timings['wall_clock'] - timings['total_frame']
                                logging.info(f"  {'wall_clock':25s}: {timings['wall_clock']:6.2f}ms (frame-to-frame interval)")
                                logging.info(f"  {'smode_idle':25s}: {smode_idle:6.2f}ms (Smode + idle time)")
                try:
                    if self._handle_pending_commands(messages):
                        return
                except Exception:
                    self._recover_from_failed_command()
        except socket.error as e:
            logging.error(f"Socket error during processing: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise
        finally:
            self.socket.close()
            logging.info("Socket connection closed")

    def __del__(self):
        try:
            if hasattr(self, 'low_latency'):
                try:
                    self.low_latency.apply(False)
                except Exception:
                    pass

            if hasattr(self, 'stream') and self.stream is not None:
                try:
                    del self.stream
                    self.stream = None
                except Exception as e:
                    logging.warning(f"Error cleaning up stream: {e}")

            if hasattr(self, 'controlnet_manager'):
                self.controlnet_manager.cleanup()

            if hasattr(self, 'faceid_processor') and self.faceid_processor is not None:
                try:
                    self.faceid_processor.cleanup()
                    self.faceid_processor = None
                except Exception as e:
                    logging.debug(f"FaceID processor cleanup error (non-critical): {e}")

            if hasattr(self, 'preprocessor_orchestrator') and self.preprocessor_orchestrator is not None:
                try:
                    self.preprocessor_orchestrator.cleanup()
                    self.preprocessor_orchestrator = None
                except Exception as e:
                    logging.warning(f"Error cleaning up preprocessor orchestrator: {e}")

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

            if hasattr(self, 'socket') and self.socket:
                try:
                    self.socket.close()
                except Exception as e:
                    logging.debug(f"Socket cleanup error (non-critical): {e}")

            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logging.debug(f"CUDA cache cleanup error (non-critical): {e}")

        except Exception as e:
            try:
                logging.warning(f"Error in App.__del__: {e}")
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smode Bridge Client Application"
    )
    parser.add_argument(
        "--port", type=int, default=4190, help="Port number"
    )
    parser.add_argument(
        "--uuid", type=str, default="9598a3b7-c13c-4ae3-9b64-2ed55275f03b", help="Smode modifier UUID"
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Width of the image"
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Height of the image")
    parser.add_argument(
        "--device", type=int, default=0, help="The CUDA device index to use"
    )
    parser.add_argument(
        "--model", type=str, default="stabilityai/sd-turbo", help="Model name to use"
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
