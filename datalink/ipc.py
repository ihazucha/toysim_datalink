from multiprocessing.sharedctypes import Synchronized
import zmq

from typing import Any
from multiprocessing import Value
from time import time_ns



class SPMCQueue:
    """Single Producer Multiple Consumers Queue using ZMQ IPC sockets"""

    ZMQ_CONTEXT = zmq.Context()

    def __init__(self, port: int):
        self._port = port
        # TODO: doesn't work for Messaging singleton - on Windows, every process creates new instance
        self._has_producer = Value("b", False)

    def get_producer(self):
        assert not self._has_producer.value, f"[{self.__class__.__name__}] Producer already exists!"
        self._has_producer.value = True
        return SPMCQueue.Producer(self._port, self._has_producer)

    def get_consumer(self):
        return SPMCQueue.Consumer(self._port)

    class Producer:
        def __init__(self, port: int, has_producer: Synchronized, real_time=True):
            self._port = port
            self._has_producer = has_producer
            self._socket = SPMCQueue.ZMQ_CONTEXT.socket(zmq.PUB)
            if real_time:
                # High water mark - limits queue size
                self._socket.set_hwm(1)
                # Always deliver the most recent message
                self._socket.setsockopt(zmq.CONFLATE, 1)
            self._socket.bind(f"tcp://*:{self._port}")

        def __del__(self):
            if self._socket and not self._socket.closed:
                self._socket.close()
            self._has_producer.value = False

        def put(self, data: Any):
            self._socket.send_pyobj((time_ns(), data))

    class Consumer:
        def __init__(self, port: int, real_time=True):
            self._port = port
            self._socket = SPMCQueue.ZMQ_CONTEXT.socket(zmq.SUB)
            if real_time:
                # High water mark - limits queue size
                self._socket.set_hwm(1)
                # Always deliver the most recent message
                self._socket.setsockopt(zmq.CONFLATE, 1)
            self._socket.connect(f"tcp://localhost:{self._port}")
            self._socket.subscribe("")

            self.last_put_timestamp = None
            self.last_get_timestamp = None

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


class Messaging:
    def __init__(self):
        self.q_image = SPMCQueue(port=10001)
        self.q_sensor = SPMCQueue(port=10002)
        self.q_control = SPMCQueue(port=10003)
        self.q_simulation = SPMCQueue(port=10004)
        self.q_processing = SPMCQueue(port=10005)
        self.q_ui = SPMCQueue(port=10006)

messaging = Messaging()
