from pandas import DataFrame
from ml4iiot.pipeline.step import AbstractStep
import pandas as pd
from ml4iiot.utility import instance_from_config


class AbstractAlgorithm(AbstractStep):
    def __init__(self, config):
        super().__init__(config)

        self.do_fit_condition = instance_from_config(self.get_config('do_fit_condition', default={'class': 'ml4iiot.conditions.TrueCondition'}))
        self.do_predict_condition = instance_from_config(self.get_config('do_predict_condition', default={'class': 'ml4iiot.conditions.TrueCondition'}))

        pd.options.mode.chained_assignment = None

    def do_fit(self, data_frame: DataFrame):
        return self.do_fit_condition.evaluate(data_frame)

    def do_predict(self, data_frame: DataFrame):
        return self.do_predict_condition.evaluate(data_frame)
