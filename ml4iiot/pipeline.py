import os
from abc import abstractmethod, ABC
from pandas import DataFrame
from ml4iiot.utility import instance_from_config, get_recursive_config


class Pipeline:
    def __init__(self, config):
        self.config = config
        keras_backend = self.get_config('settings', 'keras_backend', default='')

        if keras_backend:
            os.environ['KERAS_BACKEND'] = keras_backend

        self.input = instance_from_config(self.get_config('input'))
        self.algorithm = instance_from_config(self.get_config('algorithm'))
        self.output = instance_from_config(self.get_config('output'))
        self.steps = []

        self.steps.append(self.input)

        for preprocessing_step in self.get_config('preprocessing', default=[]):
            self.steps.append(instance_from_config(preprocessing_step))

        self.steps.append(self.algorithm)

        for postprocessing_step in self.get_config('postprocessing', default=[]):
            self.steps.append(instance_from_config(postprocessing_step))

        self.steps.append(self.output)

    def run(self) -> None:
        for step in self.steps:
            step.init()

        try:
            steps = self.steps[1:]  # Skip input step

            for data_frame in iter(self.input):
                try:
                    for step in steps:
                        step.process(data_frame)
                except SkipDataFrameException:
                    pass
        except Exception as e:
            print(e)

        finally:
            for step in self.steps:
                step.destroy()

    def get_config(self, *args, **kwargs):
        return get_recursive_config(self.config, *args, **kwargs)


class SkipDataFrameException(Exception):
    pass


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
