from pandas import DataFrame
from ml4iiot.pipeline import AbstractStep


class MovingExponentialSmoothing(AbstractStep):
    def __init__(self, config: dict):
        super().__init__(config)

        self.column_mapping = self.get_config('column_mapping')
        self.smoothing_factor = self.get_config('smoothing_factor', default=0.1)
        self.moving_values = {}

    def get_column_mapping_key(self, source_column: str, target_column: str) -> str:
        return source_column + '_' + target_column

    def process(self, data_frame: DataFrame) -> None:
        for source_column, target_column in self.column_mapping.items():
            if source_column not in data_frame:
                continue

            key = self.get_column_mapping_key(source_column, target_column)
            current_value = data_frame[source_column].values[-1]

            if key not in self.moving_values:
                self.moving_values[key] = current_value
            else:
                self.moving_values[key] = self.smoothing_factor * current_value + (1 - self.smoothing_factor) * self.moving_values[key]

            data_frame[target_column] = float('nan')
            data_frame[target_column].iloc[-1] = self.moving_values[key]
