import zmq
import Antipasti.legacy.netdatautils as ndu


class WorkerBase(object):
    def __init__(self, worker_id=None, pull_from_port=None, push_to_port=None):
        """
        Abstract class for worker to handle network comms. No computation is to be done here.

        :param worker_id:
        :param pull_from_port:
        :param push_to_port:
        """
        self.worker_id = worker_id
        self.pull_port = pull_from_port
        self.push_port = push_to_port
        # TODO
        pass

    def connect_to_ports(self):
        # TODO
        pass

    def pull(self):
        # TODO
        pass

    def push(self):
        # TODO
        pass

    def work(self):
        raise NotImplementedError

    def from_config(self, config_file):
        config = ndu.yaml2dict(config_file)

        self.worker_id = config.get('worker_id', self.worker_id)
        self.pull_port = config.get('pull_from_port', self.pull_port)
        self.push_port = config.get('push_to_port', self.push_port)

