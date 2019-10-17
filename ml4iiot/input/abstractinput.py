from abc import ABC, abstractmethod
from ml4iiot.utility import get_recursive_config, instance_from_config


class AbstractInput(ABC):

    def __init__(self, config):
        super().__init__()

        self.config = config
        windowing_strategy = self.get_config('windowing_strategy')
        windowing_strategy['config']['input'] = self
        self.windowing_strategy = instance_from_config(self.get_config('windowing_strategy'))

    def open(self):
        pass

    def close(self):
        pass

    @abstractmethod
    def next_data_frame(self, batch_size=1):
        pass

    def __next__(self):
        return self.next_window()

    def __iter__(self):
        return self

    def get_config(self, *args, **kwargs):
        return get_recursive_config(self.config, *args, **kwargs)

    def next_window(self):
        return self.windowing_strategy.next_window()

