from time import sleep

import struct
import socket

from multiprocessing import Process
from threading import Thread, Event

# TODO: remove from here, or abstract out form data to prevent loading numpy
from datalink.data import ControlData, Serializable, SimData
from datalink.ipc import SPMCQueue

# Helpers
# -------------------------------------------------------------------------------------------------


def get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


class ClassLogger:
    def log(self, msg: str,  append=False, end="\n"):
        name = f" - {self._id}" if hasattr(self, "_id") else ""
        start = "" if append else f"[{self.__class__.__name__}{name}] "
        print(f"{start}{msg}", end=end)


# Classes
# -------------------------------------------------------------------------------------------------


class TcpClient(Process, ClassLogger):
    def __init__(self, addr: tuple, q_recv: SPMCQueue = None, q_send: SPMCQueue = None):
        super().__init__()
        self.addr = addr
        self.q_recv = q_recv
        self.q_send = q_send
        self.sock = None
        self.time_to_reconnect = 2

    def run(self):
        while True:
            try:
                if not self._connect():
                    continue
                self._communicate()
            finally:
                self._close()

    def _connect(self) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.25)
        try:
            self.log(f"Connecting to {self.addr[0]}:{self.addr[1]} ... ", end="")
            sock.connect(self.addr)
            sock.settimeout(None)
            self.sock = sock
            self.log(f"success", append=True)
            return True
        except (TimeoutError, BlockingIOError, ConnectionError) as e:
            self.log(f"{e.strerror} - retry in {self.time_to_reconnect}s", append=True)
            sleep(self.time_to_reconnect)
        return False

    def _close(self):
        if self.sock:
            try:
                self.log(f"Closing socket ... ", end="")
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()
                self.sock = None
                self.log(f"success", append=True)
            except Exception as e:
                # self.log(f"error: {e}", append=True)
                self.log(f"already closed", append=True)

    def _communicate(self):
        def recv_size(size):
            data = b""
            while len(data) < size:
                more = self.sock.recv(size - len(data))
                if not more:
                    raise IOError("Socket closed before all data received")
                data += more
            return data

        def send(exit_event: Event):
            q = self.q_send.get_consumer()
            while not exit_event.is_set():
                try:
                    data: Serializable = q.get(timeout=1000)
                    if data is None:
                        self.log("Timeout on data send")
                        continue
                    data_bytes = data.to_bytes()
                    data_bytes_size = struct.pack("I", len(data_bytes))
                    self.sock.sendall(data_bytes_size)
                    self.sock.sendall(data_bytes)
                except socket.timeout:
                    if exit_event.is_set():
                        break
                except OSError:
                    exit_event.set()
                    break

        def recv(exit_event: Event):
            q = self.q_recv.get_producer()
            while not exit_event.is_set():
                try:
                    data = recv_size(ControlData.SIZE)
                    q.put(data)
                except OSError:
                    exit_event.set()
                    break

        exit_event = Event()
        
        ts = []
        if self.q_recv:
            ts.append(Thread(target=recv, args=[exit_event], daemon=True))
        if self.q_send:   
            ts.append(Thread(target=send, args=[exit_event], daemon=True))
        [t.start() for t in ts]
        [t.join() for t in ts]
            


class TcpServer(Process, ClassLogger):
    def __init__(self, addr: tuple, q_recv: SPMCQueue, q_send: SPMCQueue, id: str):
        super().__init__()
        self.addr = addr
        self.q_recv = q_recv
        self.q_send = q_send
        self._id = id
        self.sock = None
        self._bind_listen()

    def _bind_listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(self.addr)
        sock.listen()
        self.log(f"Listening on {self.addr[0]}:{self.addr[1]}")
        self.sock = sock

    def run(self):
        while True:
            self.log(f"Waiting for connection... ", end="")
            sock, addr = self.sock.accept()
            self.log(f"connected from {addr[0]}:{addr[1]}", append=True)

            connection = TcpConnection(sock=sock, q_recv=self.q_recv, q_send=self.q_send)
            connection.run()
            self.log(f"Client {addr[0]}:{addr[1]} disconnected")


class TcpConnection(ClassLogger):
    def __init__(self, sock: socket, q_recv: SPMCQueue, q_send: SPMCQueue):
        self.sock = sock
        self.q_recv = q_recv
        self.q_send = q_send
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # TODO: handle timeout differently
        self.sock.settimeout(5.0)

    def __del__(self):
        self._close()

    def _close(self):
        if self.sock is None:
            return
        try:
            self.log(f"Closing socket ... ", end="")
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()
            self.log(f"success", append=True)
        except OSError as e:
            self.log(f"error: {e}", append=True)
            pass

    def _recv_data(self, size):
        data = b""
        while len(data) < size:
            more = self.sock.recv(size - len(data))
            if not more:
                raise IOError("Socket closed before all data received")
            data += more
        return data

    def run(self):
        def send(exit_event: Event):
            q = self.q_send.get_consumer()
            while not exit_event.is_set():
                try:
                    data: Serializable = q.get(timeout=100)
                    if data:
                        self.sock.sendall(data.to_bytes())
                except socket.timeout:
                    if exit_event.is_set():
                        break
                except OSError:
                    exit_event.set()
                    # self.log("Send failed - client disconnected")
                    break

        def recv(exit_event: Event):
            q = self.q_recv.get_producer()
            while not exit_event.is_set():
                try:
                    size_bytes = self.sock.recv(4)
                    if not size_bytes:
                        exit_event.set()
                        break
                    size = struct.unpack("I", size_bytes)[0]
                    data = self._recv_data(size)
                    q.put(data)
                except OSError:
                    # self.log("Recv failed - client disconnected")
                    exit_event.set()
                    break

        exit_event = Event()
        t_recv = Thread(target=recv, args=[exit_event], daemon=True)
        t_send = Thread(target=send, args=[exit_event], daemon=True)
        ts = [t_recv, t_send]
        [t.start() for t in ts]
        exit_event.wait()
        [t.join() for t in ts]
