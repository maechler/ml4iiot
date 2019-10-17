from abc import ABC, abstractmethod
from ml4iiot.utility import get_recursive_config


class AbstractOutput(ABC):

    def __init__(self, config):
        super().__init__()

        self.config = config

    def open(self):
        pass

    def close(self):
        pass

    @abstractmethod
    def emit(self, input_frame, output_frame):
        pass

    def get_config(self, *args, **kwargs):
        return get_recursive_config(self.config, *args, **kwargs)
