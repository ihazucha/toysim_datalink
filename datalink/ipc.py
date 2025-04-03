from multiprocessing.sharedctypes import Synchronized
import zmq

from typing import Any
from multiprocessing import Value
from time import time_ns
from enum import Enum


# TODO:
# 1. tidy up the impl
# 2. ensure name is unique - introduce some common messaging system


class SPMCQueueType(Enum):
    TCP = "tcp"
    IPC = "ipc"


class SPMCQueue:
    """Single Producer Multiple Consumers Queue using ZMQ IPC sockets"""

    def __init__(self, name: str, type: SPMCQueueType, port: int | None = None, no_queue=True):
        assert (
            type == SPMCQueueType.IPC or port
        ), f"[{self.__class__.__name__}] Port required for TCP"
        self.name = name
        self.type = type
        self.port = port
        self.no_queue = no_queue
        self.producer = None

    def get_producer(self):
        if self.producer is None:
            self.producer = SPMCQueue.Producer(
                port=self.port, name=self.name, type=self.type, no_queue=self.no_queue
            )
        return self.producer

    def get_consumer(self):
        return SPMCQueue.Consumer(
            port=self.port, name=self.name, type=self.type, no_queue=self.no_queue
        )

    class Producer:
        def __init__(
            self,
            port: int,
            name: str,
            type: SPMCQueueType,
            no_queue=True,
        ):
            self._port = port
            self._name = name
            self._type = type
            self._socket = zmq.Context.instance().socket(zmq.PUB)

            if no_queue:
                # High water mark - limits queue size
                self._socket.set_hwm(1)
                # Always deliver the most recent message
                self._socket.setsockopt(zmq.CONFLATE, 1)

            if self._type == SPMCQueueType.TCP:
                self._socket.bind(f"tcp://*:{self._port}")
            elif self._type == SPMCQueueType.IPC:
                self._socket.bind(f"ipc:///tmp/{self._name}")
            else:
                raise NotImplementedError(f"[SPMCQueue {self._name}] unknown type: '{self._type}'")

        def __del__(self):
            if self._socket and not self._socket.closed:
                self._socket.close()

        def put(self, data: Any):
            self._socket.send_pyobj((time_ns(), data))

    class Consumer:
        def __init__(self, port: int, name: str, type: SPMCQueueType, no_queue=True):
            self._port = port
            self._name = name
            self._type = type
            self.last_put_timestamp = None
            self.last_get_timestamp = None
            self._socket = zmq.Context.instance().socket(zmq.SUB)
            if no_queue:
                # High water mark - limits queue size
                self._socket.set_hwm(1)
                # Always deliver the most recent message
                self._socket.setsockopt(zmq.CONFLATE, 1)
            if self._type == SPMCQueueType.TCP:
                self._socket.connect(f"tcp://localhost:{self._port}")
            elif self._type == SPMCQueueType.IPC:
                self._socket.connect(f"ipc:///tmp/{self._name}")
            else:
                raise NotImplementedError(f"[SPMCQueue {self._name}] unknown type: '{self._type}'")
            self._socket.subscribe("")

        def __del__(self):
            if self._socket and not self._socket.closed:
                self._socket.close()

        def get(self, timeout: int = None) -> Any:
            e = self._socket.poll(timeout)
            if e == 0:
                return None
            self.last_put_timestamp, data = self._socket.recv_pyobj()
            self.last_get_timestamp = time_ns()
            return data
