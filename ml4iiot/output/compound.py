from pandas import DataFrame
from ml4iiot.output.abstractoutput import AbstractOutput
from ml4iiot.utility import instance_from_config


class CompoundOutput(AbstractOutput):

    def __init__(self, config):
        super().__init__(config)

        self.output_adapters = []

        for output_config in self.get_config('output_adapters'):
            self.output_adapters.append(instance_from_config(output_config))

    def init(self) -> None:
        super().init()

        for output_adapter in self.output_adapters:
            output_adapter.init()

    def process(self, data_frame: DataFrame) -> None:
        for output_adapter in self.output_adapters:
            output_adapter.process(data_frame)

    def destroy(self) -> None:
        super().destroy()

        for output_adapter in self.output_adapters:
            output_adapter.destroy()
