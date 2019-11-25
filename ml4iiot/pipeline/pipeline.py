from ml4iiot.utility import instance_from_config, get_recursive_config


class Pipeline:

    def __init__(self, config):
        self.config = config
        self.input = instance_from_config(self.get_config('input'))
        self.algorithm = instance_from_config(config['algorithm'])
        self.output = instance_from_config(config['output'])
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

        finally:
            for step in self.steps:
                step.destroy()

    def get_config(self, *args, **kwargs):
        return get_recursive_config(self.config, *args, **kwargs)


class SkipDataFrameException(Exception):
    pass
