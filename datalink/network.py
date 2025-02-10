import struct
import math
import socket

from multiprocessing import Process
from threading import Thread, Event
from typing import Tuple

from datalink.ipc import messaging
from datalink.utils import jpg_encode  # TODO: could be handled by the original process                                                             
from datalink.data import JPGImageData, RawImageData, RemoteControlData, SensorData, SimData

MAX_DGRAM_SIZE = 2**16


def get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


# --------------------


class ImageDataSender(Thread):
    DGRAM_HEADER_SIZE = 64
    DATA_HEADER_SIZE = struct.calcsize("=BQ")
    MAX_DATA_SIZE = MAX_DGRAM_SIZE - DGRAM_HEADER_SIZE - DATA_HEADER_SIZE

    def __init__(self, addr: Tuple[str, int]):
        super().__init__()
        self._addr = addr

    def run(self) -> None:
        q = messaging.q_image.get_consumer()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock = sock
            while True:
                raw_image_data: RawImageData = q.get()
                jpg = jpg_encode(raw_image_data.image_array, 80)
                self._send(jpg, raw_image_data.timestamp)

    def _send(self, image, timestamp: int):
        size = len(image)  # TODO: optimise (could be estimated and done without len4each)
        segment_count = math.ceil(size / self.__class__.MAX_DATA_SIZE)
        header = struct.pack("=BQ", segment_count, timestamp)
        image_mv = memoryview(image)
        start_pos = 0
        while segment_count:
            end_pos = min(size, start_pos + self.__class__.MAX_DATA_SIZE)
            self._sock.sendto(header + image_mv[start_pos:end_pos], self._addr)
            start_pos = end_pos
            segment_count -= 1


class ImageDataReceiver(Thread):
    def __init__(self, addr: Tuple[str, int]):
        super().__init__()
        self._addr = addr

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(self._addr)
            self._sock = s
            q = messaging.q_image.get_producer()
            while True:
                jpg_image_data = self._recv()
                q.put(jpg_image_data)

    def _recv(self) -> JPGImageData:
        jpg_data = b""
        segment = 255
        while segment != 1:
            dgram, _ = self._sock.recvfrom(MAX_DGRAM_SIZE)
            segment, timestamp = struct.unpack("=BQ", dgram[0:9])
            jpg_data += dgram[9:]
        return JPGImageData(timestamp=timestamp, jpg=jpg_data)


# --------------------


class SensorDataSender(Thread):
    def __init__(self, addr: Tuple[str, int]):
        super().__init__()
        self._addr = addr

    def run(self):
        q = messaging.q_sensor.get_consumer()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            while True:
                dataframe = q.get()
                sock.sendto(dataframe.to_bytes(), self._addr)


class SensorDataReceiver(Thread):
    def __init__(self, addr: Tuple[str, int]):
        super().__init__()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(addr)

    def run(self):
        try:
            self._run()
        finally:
            self._sock.close()

    def get_client_ip(self):
        _, addr = self._sock.recvfrom(MAX_DGRAM_SIZE)
        return addr[0]

    def _run(self):
        q = messaging.q_sensor.get_producer()
        while True:
            data, _ = self._sock.recvfrom(MAX_DGRAM_SIZE)
            decoded_data = SensorData.from_bytes(data)
            q.put(decoded_data)


# --------------------


class RemoteDataSender(Thread):
    def __init__(self, addr: Tuple[str, int]):
        super().__init__()
        self._addr = addr

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            q = messaging.q_control.get_consumer()
            while True:
                data = q.get()
                s.sendto(data.to_bytes(), self._addr)


class RemoteDataReceiver(Thread):
    def __init__(self, addr: Tuple[str, int]):
        super().__init__()
        self._addr = addr

    def run(self):
        q = messaging.q_control.get_producer()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(self._addr)
            while True:
                data = sock.recv(MAX_DGRAM_SIZE)
                q.put(data)


# --------------------


class TcpServer(Process):
    def __init__(self, server_ip: str):
        super().__init__()
        self._sock = self._bind_listen((server_ip, 8888))

    def _log(self, msg: str):
        print(f"[{self.__class__.__name__}] {msg}")

    def _bind_listen(self, addr: tuple):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(addr)
        sock.listen()
        self._log(f"Active on: {addr}")
        return sock

    def run(self):
        while True:
            self._log(f"Waiting for connection...")
            sock, addr = self._sock.accept()
            self._log(f"Connected: {addr}")
            connection = TcpConnection(sock)
            connection.receive_loop()
            del connection
            self._log(f"Disconnected: {addr}")


class TcpConnection:
    def __init__(self, socket: socket):
        self._socket = socket

    def __del__(self):
        self._close()

    def _log(self, msg: str, who: str = "[Connection]"):
        print(f"{who} {msg}")

    def _close(self):
        if self._socket is None:
            return
        try:
            self._log(f"Closing cocket...")
            self._socket.shutdown(socket.SHUT_RDWR)
            self._socket.close()
            self._log(f"Socket closed")
        except OSError as e:
            self._log(f"Error while closing socket: {e}")
            pass

    def _recv_data(self, size):
        data = b""
        while len(data) < size:
            more = self._socket.recv(size - len(data))
            if not more:
                raise IOError("Socket closed before all data received")
            data += more
        return data

    def receive_loop(self):
        def send(exit_event: Event):
            q = messaging.q_control.get_consumer()
            self._socket.settimeout(1.0)  # Set a timeout of 1 second
            while not exit_event.is_set():
                try:
                    data: RemoteControlData = q.get(timeout=1)
                    if data is not None:
                        self._socket.sendall(data.to_bytes())
                except socket.timeout:
                    if exit_event.is_set():
                        break
                except OSError:
                    exit_event.set()
                    self._log("Send failed - connection closed by client")
                    break

        def recv(exit_event: Event):
            q = messaging.q_simulation.get_producer()
            while not exit_event.is_set():
                try:
                    data = self._recv_data(SimData.SIZE)
                    q.put(data)
                except OSError:
                    self._log("Recv failed - connection closed by client")
                    exit_event.set()
                    break

        exit_event = Event()
        t_recv = Thread(target=recv, args=[exit_event], daemon=True)
        t_send = Thread(target=send, args=[exit_event], daemon=True)
        ts = [t_recv, t_send]
        [t.start() for t in ts]
        exit_event.wait()
        print("[Connection] Shutting down send and recv threads..")
        [t.join() for t in ts]
        print("[Connection] send and recv threads shut down")


# --------------------


class NetworkClient(Process):
    def __init__(self, server_ip: str):
        super().__init__()
        self._server_ip = server_ip

    def run(self):
        t_image = ImageDataSender(addr=(self._server_ip, 5500))
        t_sensor = SensorDataSender(addr=(self._server_ip, 5510))
        t_remote = RemoteDataReceiver(addr=(get_local_ip(), 5520))
        ts = [t_image, t_sensor, t_remote]
        [t.start() for t in ts]
        [t.join() for t in ts]


class NetworkServer(Process):
    def __init__(self, server_ip: str = get_local_ip()):
        super().__init__()
        self._server_ip = server_ip

    def run(self):
        t_image = ImageDataReceiver(addr=(self._server_ip, 5500))
        t_sensor = SensorDataReceiver(addr=(self._server_ip, 5510))
        t_remote = RemoteDataSender(addr=(t_sensor.get_client_ip(), 5520))
        threads = [t_image, t_sensor, t_remote]
        [t.start() for t in threads]
        [t.join() for t in threads]