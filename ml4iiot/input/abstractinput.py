from abc import abstractmethod
from ml4iiot.pipeline.step import AbstractStep
from ml4iiot.utility import instance_from_config


class AbstractInput(AbstractStep):

    def __init__(self, config):
        super().__init__(config)

        windowing_strategy = self.get_config('windowing_strategy')
        windowing_strategy['config']['input'] = self
        self.windowing_strategy = instance_from_config(self.get_config('windowing_strategy'))

    @abstractmethod
    def next_data_frame(self, batch_size=1):
        pass

    def __next__(self):
        return self.process(None)

    def __iter__(self):
        return self

    def process(self, data_frame):
        return self.windowing_strategy.next_window()
