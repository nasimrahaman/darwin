import time
import zmq
import Antipasti.legacy.netdatautils as ndu


class WorkerBase(object):
    def __init__(self, worker_id=None, pull_from_address=None, push_to_address=None):
        """
        Abstract class for worker to handle network comms. No computation is to be done here.

        :param worker_id:
        :param pull_from_address:
        :param push_to_address:
        """
        # Private
        self._zmq_context = zmq.Context()
        self._push_socket = None
        self._pull_socket = None
        self._pulled = None
        self._pushed = None
        self._last_pulled = None
        self._last_pushed = None
        # Public
        self.worker_id = worker_id
        self.pull_address = pull_from_address
        self.push_address = push_to_address
        # TODO
        pass

    @property
    def push_socket(self):
        if self._push_socket is None:
            self._push_socket = self._zmq_context.socket(zmq.PUSH)
            self._push_socket.connect(self.push_address)
        return self._push_socket

    @property
    def pull_socket(self):
        if self._pull_socket is None:
            self._pull_socket = self._zmq_context.socket(zmq.PULL)
            self._pull_socket.connect(self.pull_address)
        return self._pull_socket

    def get_from_pulled(self, key, default=None):
        return self._pulled.get(key, default)

    def add_to_push(self, key, value):
        return self._pushed.update({key: value})

    def pull(self):
        """Pull from server. Cache is destroyed."""
        self._pulled = self.pull_socket.recv_json()
        self._last_pulled = time.time()

    def push(self):
        """Push to server. Cache is destroyed."""
        self.push_socket.send_json(self._pushed)
        # Flush self._pushed
        self._pushed = {}
        self._last_pushed = time.time()

    def work(self):
        raise NotImplementedError

    def from_config(self, config_file):
        config = ndu.yaml2dict(config_file)

        self.worker_id = config.get('worker_id', self.worker_id)
        self.pull_address = config.get('pull_from_address', self.pull_address)
        self.push_address = config.get('push_to_address', self.push_address)


class Worker(WorkerBase):
    pass
