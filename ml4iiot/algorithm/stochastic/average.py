from pandas import DataFrame
from ml4iiot.algorithm.abstractalgorithm import AbstractAlgorithm


class Average(AbstractAlgorithm):

    def __init__(self, config):
        super().__init__(config)

        self.column_mapping = self.get_config('column_mapping')

    def process(self, data_frame: DataFrame) -> None:
        for source_column, target_column in self.column_mapping.items():
            data_frame[target_column] = float('nan')

            data_frame[target_column].iloc[-1] = data_frame[source_column].mean()


class ExponentialWeightedMovingAverage(AbstractAlgorithm):

    def __init__(self, config):
        super().__init__(config)

        self.column_mapping = self.get_config('column_mapping')
        self.smoothing_factor = self.get_config('smoothing_factor', default=0.1)
        self.moving_average = None

    def process(self, data_frame: DataFrame) -> None:
        for source_column, target_column in self.column_mapping.items():
            current_average = data_frame[source_column].mean()

            if self.moving_average is None:
                self.moving_average = current_average
            else:
                self.moving_average = self.smoothing_factor * current_average + (1 - self.smoothing_factor) * self.moving_average

            data_frame[target_column] = float('nan')
            data_frame[target_column].iloc[-1] = self.moving_average
