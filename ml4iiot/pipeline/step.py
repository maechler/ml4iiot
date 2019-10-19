from abc import ABC, abstractmethod
from ml4iiot.utility import get_recursive_config


class AbstractStep(ABC):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def init(self):
        pass

    def destroy(self):
        pass

    @abstractmethod
    def process(self, data_frame):
        pass

    def get_config(self, *args, **kwargs):
        return get_recursive_config(self.config, *args, **kwargs)