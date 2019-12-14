from pandas import DataFrame
from ml4iiot.pipeline import SkipDataFrameException
from ml4iiot.pipeline import AbstractStep
from ml4iiot.utility import instance_from_config


class SkipDataFrame(AbstractStep):
    def __init__(self, config):
        super().__init__(config)

        self.do_skip_condition = instance_from_config(self.get_config('do_skip', default={'class': 'ml4iiot.conditions.FalseCondition'}))

    def do_skip(self, data_frame: DataFrame):
        return self.do_skip_condition.evaluate(data_frame)

    def process(self, data_frame: DataFrame) -> None:
        if self.do_skip(data_frame):
            raise SkipDataFrameException
