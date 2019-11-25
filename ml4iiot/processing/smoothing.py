from pandas import DataFrame
from ml4iiot.pipeline.step import AbstractStep


class MovingExponentialSmoothing(AbstractStep):
    def __init__(self, config: dict):
        super().__init__(config)

        self.source_column = self.get_config('source_column')
        self.target_column = self.get_config('target_column')
        self.smoothing_factor = self.get_config('smoothing_factor', default=0.1)
        self.moving_value = None

    def process(self, data_frame: DataFrame) -> None:
        if self.source_column not in data_frame:
            return

        current_value = data_frame[self.source_column].values[-1]

        if self.moving_value is None:
            self.moving_value = current_value
        else:
            self.moving_value = self.smoothing_factor * current_value + (1 - self.smoothing_factor) * self.moving_value

        data_frame[self.target_column] = float('nan')
        data_frame[self.target_column].iloc[-1] = self.moving_value
