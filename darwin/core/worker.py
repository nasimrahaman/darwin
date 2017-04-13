import time
import zmq
import threading
import Antipasti.legacy.netdatautils as ndu
import Antipasti.utilities.pyutils2 as py2


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
        self._listener_latency = 1
        self._successfully_pulled_from_server = threading.Event()
        self._stop_listening_event = threading.Event()
        self._listener_thread = None
        self._debug_logger = py2.DebugLogger('WorkerBase')

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

    def _make_listener_thread(self):

        def _listener_thread():
            while True:
                # Die if required to
                if self._stop_listening_event.is_set():
                    break
                # Pull only when the pulled from server event is cleared
                if not self._successfully_pulled_from_server.is_set():
                    # This blocks until the pull was successful
                    # TODO Set stop processing event
                    self.pull()
                else:
                    time.sleep(self._listener_latency)
                    continue
                # Set event to say the pull was successful
                self._successfully_pulled_from_server.set()

        return threading.Thread(target=_listener_thread)

    def listen(self):
        self._stop_listening_event.clear()
        self._listener_thread = self._make_listener_thread()
        self._listener_thread.start()

    def stop_listening(self):
        self._stop_listening_event.set()

    def work(self):
        raise NotImplementedError

    def update(self):
        # Computations go here
        # ...
        # When parameter update is done in the subclass, clear event such that listner is allowed
        # to pull again.
        self._successfully_pulled_from_server.clear()

    def from_config(self, config_file):
        config = ndu.yaml2dict(config_file)
        self.worker_id = config.get('worker_id', self.worker_id)
        self.pull_address = config.get('pull_from_address', self.pull_address)
        self.push_address = config.get('push_to_address', self.push_address)
        return config


class Worker(WorkerBase):
    def __init__(self, model, feeder, **base_kwargs):
        super(Worker, self).__init__(**base_kwargs)
        # Public
        self.feeder = feeder
        self.model = model
        # TODO

    def from_config(self, config_file):
        config = super(Worker, self).from_config(config_file)
        # TODO

    pass
