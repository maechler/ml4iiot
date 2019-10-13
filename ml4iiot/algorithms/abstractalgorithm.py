from abc import ABC, abstractmethod
from ml4iiot.utility import get_recursive_config


class AbstractAlgorithm(ABC):

    def __init__(self, config):
        super().__init__()

        self.config = config

    @abstractmethod
    def compute(self, input_frame):
        pass

    def get_config(self, *args):
        return get_recursive_config(self.config, *args)
