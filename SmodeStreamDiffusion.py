import socket
import time
import select
from typing import NamedTuple
import torch
import argparse
import enum
import logging
import struct
import torchvision.transforms.functional as F
from torch.multiprocessing import reductions  # For obtaining CUDA IPC handle

from utils.wrapper import StreamDiffusionWrapper
from src.streamdiffusion import StreamDiffusion
from diffusers import AutoencoderTiny
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
        flags = CREATE_EVENT_MANUAL_RESET if signal_awakes_all_clients else 0 | CREATE_EVENT_INITIAL_SET if initial_signaled_state else 0
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
    XFORMERS = 1
    TENSORRT = 2


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
            + UINT32
            + UINT32
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


class ConfigPacket(Packet):
    def __init__(self):
        super().__init__(CommandType.CONFIG, b"")
        self.model_name = ""
        self.prompt = ""
        self.negative_prompt = ""
        self.seed = 0
        self.width = 0
        self.height = 0
        self.t_index_list = [16]
        self.guidance_scale = 5.0
        self.mode = Mode.IMAGE_TO_IMAGE
        self.cfg_type = "none"
        self.acceleration = Acceleration.XFORMERS

    def from_bytes(self, data: bytes):
        offset = 0
        self.t_index_list = []
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
        self.smode_tensor_ipc_info = reductions.reduce_tensor(
            self.smode_tensor
        )[1]

    def copy_smode_to_stream_diffusion(self):
        """
        Copy data from smode_tensor into stream_diffusion_tensor, performing necessary
        type conversion and ensuring device alignment.
        """
        converted = self.smode_tensor.to(dtype=self.dtype, device=self.device)
        self.stream_diffusion_tensor.copy_(converted)

    def copy_to_smode(self, x_output: torch.Tensor):
        """
        Copy external tensor x_output into the smode_tensor, updating its IPC info.
        """
        src = x_output.to(dtype=torch.float32, device=self.device)
        self.smode_tensor.copy_(src)


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
        self.config = config
        self.device = device
        self.torch_dtype = torch_dtype
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.model_name = config.model
        self.current_prompt = ""
        self.negative_prompt = ""
        self.seed = 8
        self.width = config.width
        self.height = config.height
        self.input_tensors = None
        self.output_tensors = None
        self.buffer_shape = None
        self.t_index_list = [16]
        self.mode = Mode.IMAGE_TO_IMAGE
        self.acceleration = Acceleration.XFORMERS

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
        self._init_connection()
        self._create_stream()

    def _create_stream(self):
        send_message(self.socket, StreamCreationPacket(False))
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=self.model_name,
            t_index_list=self.t_index_list,
            lora_dict=None,
            mode="img2img" if self.mode == Mode.IMAGE_TO_IMAGE else "txt2img",
            frame_buffer_size=1,
            width=self.width,
            height=self.height,
            warmup=10,
            acceleration="xformers" if self.acceleration == Acceleration.XFORMERS
                                    else "tensorrt"
                                    if self.acceleration == Acceleration.TENSORRT
                                    else "none",
            device_ids=None,
            use_lcm_lora=True,
            use_tiny_vae=True,
            enable_similar_image_filter=False,
            similar_image_filter_threshold=0.98,
            use_denoising_batch=True,
            cfg_type="self" if self.mode == Mode.IMAGE_TO_IMAGE else "none",
            seed=self.seed,
            dtype=self.torch_dtype,
            device=self.device,
            output_type="pt",
        )

        self._create_tensors(3, self.width, self.height)

        send_message(self.socket, StreamCreationPacket(True))

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

    def run(self):
        logging.info("Entering main command loop")
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
                        except socket.error:
                            break
                if in_error:
                    logging.error("Socket error detected; exiting main loop")
                    exit(0)

                wait_result = self.smodeToStreamDiffusionInterProcessEvent.wait(0)
                if wait_result == win32event.WAIT_OBJECT_0:
                    compute_time = time.time()
                    if self.output_tensors is None:
                        self._create_tensors(3, self.width, self.height)

                    x_output = None
                    permuted_input_texture = None

                    self.input_tensors.copy_smode_to_stream_diffusion()

                    if self.mode == Mode.IMAGE_TO_IMAGE:
                        if self.input_tensors.stream_diffusion_tensor is not None:
                            permuted_input_texture = F.vflip(
                                self.input_tensors.stream_diffusion_tensor.permute(
                                    2, 0, 1
                                )
                            )
                        x_output = self.stream.img2img(
                            image=permuted_input_texture
                        )
                    elif self.mode == Mode.TEXT_TO_IMAGE:
                        x_output = self.stream.txt2img()
                    else:
                        logging.error(f"Unknown mode: {self.mode}")
                        continue

                    if x_output is not None and x_output.shape[0] == 1:
                        x_output = x_output.squeeze(0)

                    if x_output is not None:
                        x_output = x_output.permute(1, 2, 0)

                    if x_output is not None:
                        self.output_tensors.copy_to_smode(x_output)
                    logging.info(f"{(time.time() - compute_time) * 1000} ms")
                    self.streamDiffusionToSmodeInterProcessEvent.signal()
                # Process received messages
                for cmd, payload in messages.items():
                    if cmd == CommandType.CONFIG:
                        config_packet = ConfigPacket()
                        config_packet.from_bytes(payload)
                        logging.info(
                            f"Received CONFIG command: {vars(config_packet)}"
                        )
                        update_stream = (
                            self.model_name != config_packet.model_name
                            or self.width != config_packet.width
                            or self.height != config_packet.height
                            or self.mode != config_packet.mode
                            or self.stream.stream.cfg_type != config_packet.cfg_type
                            or self.acceleration != config_packet.acceleration
                        )
                        update_t_index_list = (
                            self.t_index_list != config_packet.t_index_list
                        )
                        self.model_name = config_packet.model_name
                        self.current_prompt = config_packet.prompt
                        self.negative_prompt = config_packet.negative_prompt
                        self.seed = config_packet.seed
                        self.width = config_packet.width
                        self.height = config_packet.height
                        self.t_index_list = config_packet.t_index_list
                        self.mode = config_packet.mode
                        self.stream.mode = (
                            "img2img"
                            if self.mode == Mode.IMAGE_TO_IMAGE
                            else "txt2img"
                        )
                        self.acceleration = config_packet.acceleration

                        if update_stream or update_t_index_list:
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
                            if self.acceleration == Acceleration.XFORMERS:
                                self.stream.stream.pipe.enable_xformers_memory_efficient_attention()
                                self.stream.recreate_pipe()
                            elif self.acceleration == Acceleration.TENSORRT:
                                try:
                                    from src.streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
                                    self.stream.stream = accelerate_with_tensorrt(
                                        self.stream.stream,
                                        "engines",
                                        max_batch_size=2,
                                    )
                                except ImportError:
                                    logging.warning(
                                        "TensorRT acceleration not available; continuing with xformers"
                                    )
                                    self.stream.stream.pipe.enable_xformers_memory_efficient_attention()
                                    self.stream.recreate_pipe()

                        if update_stream:
                            self._create_stream()
                            self.ipc_info = None
                            self.buffer_shape = None

                        if self.stream:
                            self.stream.stream.prepare(
                                self.current_prompt,
                                self.negative_prompt,
                                num_inference_steps=50,
                                guidance_scale=config_packet.guidance_scale,
                                seed=self.seed,
                            )

                    elif cmd == CommandType.STOP:
                        logging.info("Received STOP command; exiting main loop")
                        return
                    else:
                        logging.warning(f"Received unexpected command: {cmd}")

                # No fixed sleep here; select timeout manages CPU usage

        except socket.error as e:
            logging.error(f"Socket error during processing: {e}")
        finally:
            self.socket.close()
            logging.info("Socket connection closed")


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
