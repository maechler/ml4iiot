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
        self.moving_averages = {}

    def process(self, data_frame: DataFrame) -> None:
        for source_column, target_column in self.column_mapping.items():
            current_average = data_frame[source_column].mean()

            if target_column not in self.moving_averages:
                self.moving_averages[target_column] = current_average
            else:
                self.moving_averages[target_column] = self.smoothing_factor * current_average + (1 - self.smoothing_factor) * self.moving_averages[target_column]

            data_frame[target_column] = float('nan')
            data_frame[target_column].iloc[-1] = self.moving_averages[target_column]


class ExponentialWeightedMovingMinMaxAverage(AbstractAlgorithm):

    def __init__(self, config):
        super().__init__(config)

        self.column_mapping = self.get_config('column_mapping')
        self.short_smoothing_factor = self.get_config('short_smoothing_factor', default=0.1)
        self.long_smoothing_factor = self.get_config('long_smoothing_factor', default=0.01)

        self.short_moving_averages = {}
        self.long_moving_averages = {}

        self.short_moving_mins = {}
        self.long_moving_mins = {}

        self.short_moving_maxs = {}
        self.long_moving_maxs = {}

    def process(self, data_frame: DataFrame) -> None:
        for source_column, target_column in self.column_mapping.items():
            current_average = data_frame[source_column].mean()
            current_min = data_frame[source_column].min()
            current_max = data_frame[source_column].max()

            if target_column not in self.short_moving_averages:
                self.short_moving_averages[target_column] = current_average
                self.long_moving_averages[target_column] = current_average

                self.short_moving_mins[target_column] = current_min
                self.long_moving_mins[target_column] = current_min

                self.short_moving_maxs[target_column] = current_max
                self.long_moving_maxs[target_column] = current_max
            else:
                self.short_moving_averages[target_column] = self.short_smoothing_factor * current_average + (1 - self.short_smoothing_factor) * self.short_moving_averages[target_column]
                self.long_moving_averages[target_column] = self.long_smoothing_factor * current_average + (1 - self.long_smoothing_factor) * self.long_moving_averages[target_column]

                self.short_moving_mins[target_column] = self.short_smoothing_factor * current_min + (1 - self.short_smoothing_factor) * self.short_moving_mins[target_column]
                self.long_moving_mins[target_column] = self.long_smoothing_factor * current_min + (1 - self.long_smoothing_factor) * self.long_moving_mins[target_column]

                self.short_moving_maxs[target_column] = self.short_smoothing_factor * current_max + (1 - self.short_smoothing_factor) * self.short_moving_maxs[target_column]
                self.long_moving_maxs[target_column] = self.long_smoothing_factor * current_max + (1 - self.long_smoothing_factor) * self.long_moving_maxs[target_column]

                data_frame[target_column + '_short_average'] = float('nan')
                data_frame[target_column + '_short_average'].iloc[-1] = self.short_moving_averages[target_column]
                data_frame[target_column + '_long_average'] = float('nan')
                data_frame[target_column + '_long_average'].iloc[-1] = self.long_moving_averages[target_column]

                data_frame[target_column + '_short_min'] = float('nan')
                data_frame[target_column + '_short_min'].iloc[-1] = self.short_moving_mins[target_column]
                data_frame[target_column + '_long_min'] = float('nan')
                data_frame[target_column + '_long_min'].iloc[-1] = self.long_moving_mins[target_column]

                data_frame[target_column + '_short_max'] = float('nan')
                data_frame[target_column + '_short_max'].iloc[-1] = self.short_moving_maxs[target_column]
                data_frame[target_column + '_long_max'] = float('nan')
                data_frame[target_column + '_long_max'].iloc[-1] = self.long_moving_maxs[target_column]

                data_frame[target_column + '_diff_average'] = float('nan')
                data_frame[target_column + '_diff_average'].iloc[-1] = abs(self.short_moving_averages[target_column] - self.long_moving_averages[target_column])

                data_frame[target_column + '_diff_min'] = float('nan')
                data_frame[target_column + '_diff_min'].iloc[-1] = abs(self.short_moving_mins[target_column] - self.long_moving_mins[target_column])

                data_frame[target_column + '_diff_max'] = float('nan')
                data_frame[target_column + '_diff_max'].iloc[-1] = abs(self.short_moving_maxs[target_column] - self.long_moving_maxs[target_column])
