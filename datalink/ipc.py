import zmq

from typing import Any, Iterable, Tuple
from time import time_ns
from enum import Enum
from multiprocessing import Process


class AddrType(Enum):
    TCP = "tcp"
    IPC = "ipc"


class AbstractQueue:
    def get_producer(self):
        raise NotImplementedError()
    
    def get_consumer(self):
        raise NotImplementedError()


class SPMCQueue(AbstractQueue):
    """Single Producer Multiple Consumers Queue using ZMQ IPC sockets"""

    def __init__(self, name: str, type: AddrType, port: int | None = None, q_size=1):
        assert type == AddrType.IPC or port, f"[{self.__class__.__name__}] Port required for TCP"

        self.name = name
        self.type = type
        self.port = port
        self.q_size = q_size

        self.producer = None

    def get_producer(self) -> "Producer":
        if self.producer is None:
            self.producer = SPMCQueue.Producer(
                port=self.port, name=self.name, type=self.type, q_size=self.q_size
            )
        return self.producer

    def get_consumer(self) -> "Consumer":
        return SPMCQueue.Consumer(
            port=self.port, name=self.name, type=self.type, q_size=self.q_size
        )

    class Producer:
        def __init__(self, name, type, port, q_size):

            self._socket = zmq.Context.instance().socket(zmq.PUB)
            if q_size == 1:
                self._socket.setsockopt(zmq.CONFLATE, 1)
            self._socket.set_hwm(q_size)

            if type == AddrType.TCP:
                self._socket.bind(f"tcp://*:{port}")
            elif type == AddrType.IPC:
                self._socket.bind(f"ipc://{name}")
            else:
                raise NotImplementedError(f"[SPMCQueue {self._name} Prod] unknown type: '{type}'")

        def __del__(self):
            self.close()

        def close(self):
            if self._socket and not self._socket.closed:
                self._socket.close()

        def put(self, data: Any):
            self._socket.send_pyobj((time_ns(), data))

    class Consumer:
        def __init__(self, name, type, port, q_size):
            self.last_put_timestamp = None
            self.last_get_timestamp = None

            self._socket = zmq.Context.instance().socket(zmq.SUB)
            if q_size == 1:
                self._socket.setsockopt(zmq.CONFLATE, 1)
            self._socket.set_hwm(q_size)

            if type == AddrType.TCP:
                self._socket.connect(f"tcp://localhost:{port}")
            elif type == AddrType.IPC:
                self._socket.connect(f"ipc://{name}")
            else:
                raise NotImplementedError(f"[SPMCQueue {self._name} Cons] unknown type: '{type}'")

            self._socket.subscribe("")

        def __del__(self):
            self.close()

        def close(self):
            if self._socket and not self._socket.closed:
                self._socket.close()

        def poll(self, timeout: int = None) -> int:
            """Returns 0 if no data, else > 0"""
            return self._socket.poll(timeout)

        def get(self, timeout: int = None) -> Any:
            if self._socket.poll(timeout) == 0:
                return None
            self.last_put_timestamp, data = self._socket.recv_pyobj()
            self.last_get_timestamp = time_ns()
            return data


class MPMCProxy(Process):
    def __init__(self, name: str, addr_type: AddrType, ports: Tuple[int, int] = None):
        assert addr_type == AddrType.IPC or (
            isinstance(ports, Iterable) and isinstance(ports[0], int) and isinstance(ports[1], int)
        ), "[Proxy] ports are required for TCP addr type"

        super().__init__(daemon=True)

        self.name_prefix = name
        self.addr_type = addr_type

        if addr_type == AddrType.TCP:
            self.xpub_addr = f"tcp://localhost:{ports[0]}"
            self.xsub_addr = f"tcp://localhost:{ports[1]}"
        elif addr_type == AddrType.IPC:
            self.xpub_addr = f"ipc://{name}_xpub"
            self.xsub_addr = f"ipc://{name}_xsub"
        else:
            raise ValueError(f"Unsupported address type: {addr_type}")

    def run(self):
        frontend = zmq.Context().instance().socket(zmq.XSUB)
        backend = zmq.Context().instance().socket(zmq.XPUB)
        try:
            frontend.bind(self.xsub_addr)
            backend.bind(self.xpub_addr)
        except zmq.ZMQError as e:
            print(f"[MPMCQueue Proxy {self.name_prefix}] Proxy not started - bind error: {e}")
            frontend.close()
            backend.close()
            return

        print(f"[MPMC Proxy] Starting {self.name_prefix} - frontend on {self.xsub_addr}, backend on {self.xpub_addr}")
        zmq.proxy(frontend, backend)


class MPMCQueue(AbstractQueue):
    def __init__(
        self,
        name: str,
        addr_type: AddrType,
        ports: Tuple[int, int] = None,
        q_size: int = 1,
    ):
        assert addr_type == AddrType.IPC or (
            isinstance(ports, Iterable) and isinstance(ports[0], int) and isinstance(ports[1], int)
        ), "Proxy ports are required for TCP addr type"

        self.name = name
        self.addr_type = addr_type
        self.ports = ports
        self.q_size = q_size

        if addr_type == AddrType.TCP:
            self.xpub_addr = f"tcp://localhost:{ports[0]}"
            self.xsub_addr = f"tcp://localhost:{ports[1]}"
        elif addr_type == AddrType.IPC:
            self.xpub_addr = f"ipc://{name}_xpub"
            self.xsub_addr = f"ipc://{name}_xsub"
        else:
            raise ValueError(f"Unsupported address type: {addr_type}")

    def get_proxy(self):
        return MPMCProxy(name=self.name, addr_type=self.addr_type, ports=self.ports)

    def get_producer(self) -> "Producer":
        return MPMCQueue.Producer(
            proxy_addr=self.xsub_addr,
            q_size=self.q_size,
            q_name=self.name,
        )

    def get_consumer(self, topic:bytes=b"") -> "Consumer":
        return MPMCQueue.Consumer(
            proxy_addr=self.xpub_addr,
            q_size=self.q_size,
            q_name=self.name,
            topic=topic
        )

    class Producer:
        def __init__(self, proxy_addr: str, q_size: bool, q_name: str):
            self._proxy_addr = proxy_addr
            self._name = q_name

            self._context = zmq.Context.instance()
            self._socket = self._context.socket(zmq.PUB)

            if q_size == 1:
                self._socket.setsockopt(zmq.CONFLATE, 1)
            self._socket.set_hwm(q_size)

            try:
                self._socket.connect(self._proxy_addr)
                print(f"[I] [MPMCQueue {self._name} Prod] Connected to: {self._proxy_addr}")
            except zmq.ZMQError as e:
                print(f"[E] [MPMCQueue {self._name} Prod] Connecting {self._proxy_addr}: {e}")
                raise

        def __del__(self):
            self.close()

        def close(self):
            if self._socket and not self._socket.closed:
                self._socket.close()

        def put(self, data: Any, topic: bytes = b""):
            payload = (time_ns(), data)
            try:
                self._socket.send(topic, flags=zmq.SNDMORE | zmq.NOBLOCK)
                self._socket.send_pyobj(payload, flags=zmq.NOBLOCK)
            except zmq.Again:
                print(f"[W] [MPMCQueue {self._name} Prod] Msg skipped (would block)")
            except Exception as e:
                print(f"[E] [MPMCQueue {self._name} Prod] Error sending message: {e}")


    class Consumer:
        def __init__(self, proxy_addr: str, q_size: int, q_name: str, topic=b""):
            self._proxy_addr = proxy_addr
            self._name = q_name

            self.last_put_timestamp = None
            self.last_get_timestamp = None

            self._context = zmq.Context.instance()
            self._socket = self._context.socket(zmq.SUB)

            if q_size == 1:
                self._socket.setsockopt(zmq.CONFLATE, 1)
            self._socket.set_hwm(q_size)

            try:
                self._socket.connect(self._proxy_addr)
                print(f"[I] [MPMCQueue {self._name} Cons] Connected to {self._proxy_addr}")
            except zmq.ZMQError as e:
                print(f"[E] [MPMCQueue {self._name} Cons] failed connect {self._proxy_addr}: {e}")
                raise

            self._socket.subscribe(topic)
        
        def __del__(self):
            self.close()

        def close(self):
            if self._socket and not self._socket.closed:
                self._socket.close()

        def poll(self, timeout: int = None) -> int:
            """Returns 0 if no data, else > 0"""
            return self._socket.poll(timeout)

        def get(self, timeout: int = None) -> tuple[bytes, Any] | None:
            try:
                if not self._socket.poll(timeout):
                    return None

                timestamped_data: tuple = self._socket.recv_pyobj(flags=zmq.NOBLOCK)
                self.last_get_timestamp = time_ns()
                self.last_put_timestamp = timestamped_data[0]
                data = timestamped_data[1]
                return data
            except Exception as e:
                print(f"[MPMCQueue {self._name} Cons] Error receiving message: {e}")
                return None

