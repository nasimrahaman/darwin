import Antipasti.legacy.netdatautils as ndu
import Antipasti.legacy.pykit as py
import Antipasti.backend as A


class Model(object):
    """Encapsulate all the model building business."""
    def __init__(self, model_constructor, devices=None):
        # Private
        self._model = None
        # Public
        self.devices = devices
        self.model_constructor = model_constructor
        self.construct_kwargs = None

        self.inputs = {}
        self.targets = {}
        self.outputs = {}

    @property
    def model(self):
        if self._model is None:
            self.make_model()
        return self._model

    def make_model(self, construct_kwargs=None):
        construct_kwargs = self.construct_kwargs if construct_kwargs is None else construct_kwargs
        self._model = self.model_constructor(**construct_kwargs)

    def build(self, inputs, targets, construct_kwargs=None):
        # Gather kwargs to construct network
        construct_kwargs = self.construct_kwargs if construct_kwargs is None else construct_kwargs
        devices = self.devices

        # Construct model
        if construct_kwargs is not None:
            self.make_model(construct_kwargs)
        model = self.model

        def _validate_device_tensor_mappings(mapping):
            if not isinstance(mapping, dict):
                mapping = {device: py.delist(py.obj2list(mapping)) for device in devices}
            else:
                assert set(mapping.keys()) == set(devices), \
                    "Must define one input for all devices or none at all."
                mapping = {device: py.delist(py.obj2list(mapping[device])) for device in devices}
            return mapping

        inputs = self.inputs = _validate_device_tensor_mappings(inputs)
        targets = self.targets = _validate_device_tensor_mappings(targets)
        outputs = self.outputs = {}

        # Make forward pass with all devices
        for device in devices:
            input_on_device = inputs.get(device)
            targets_on_device = outputs.get(device)
            with A.ContextSupermanager(device=device).manage():
                # Get output
                output_on_device = outputs[device] = model(inputs)
                # Clear cache (the good old keras bug)
                model._output_tensor_cache = {}
                # TODO
                pass

    def save(self):
        pass

    def load(self):
        pass

    def from_config(self, config_file):
        config = ndu.yaml2dict(config_file)
        self.devices = config.get('devices', self.devices)
    pass
