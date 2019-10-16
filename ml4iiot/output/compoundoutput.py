from ml4iiot.output.abstractoutput import AbstractOutput
from ml4iiot.utility import instance_from_config


class CompoundOutput(AbstractOutput):

    def __init__(self, config):
        super().__init__(config)

        self.output_adapters = []

        for output_config in self.get_config('output_adapters'):
            self.output_adapters.append(instance_from_config(output_config))

    def open(self):
        super().open()

        for output_adapter in self.output_adapters:
            output_adapter.open()

    def emit(self, input_frame, output_frame):
        for output_adapter in self.output_adapters:
            output_adapter.emit(input_frame, output_frame)

    def close(self):
        super().close()

        for output_adapter in self.output_adapters:
            output_adapter.close()
