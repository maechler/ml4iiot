from ml4iiot.utility import instance_from_config


class Pipeline:

    def __init__(self, config):
        self.config = config
        self.input = instance_from_config(config['input'])
        self.output = instance_from_config(config['output'])
        self.algorithm = instance_from_config(config['algorithm'])

    def run(self):
        self.input.open()
        self.output.open()

        try:
            for input_frame in iter(self.input):
                output_frame = self.algorithm.compute(input_frame)

                self.output.emit(input_frame, output_frame)
        finally:
            self.input.close()
            self.output.close()
