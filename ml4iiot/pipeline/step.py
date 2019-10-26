from abc import ABC, abstractmethod
from pandas import DataFrame
from ml4iiot.utility import get_recursive_config


class AbstractStep(ABC):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def init(self) -> None:
        pass

    def destroy(self) -> None:
        pass

    @abstractmethod
    def process(self, data_frame: DataFrame) -> None:
        pass

    def get_config(self, *args, **kwargs):
        return get_recursive_config(self.config, *args, **kwargs)