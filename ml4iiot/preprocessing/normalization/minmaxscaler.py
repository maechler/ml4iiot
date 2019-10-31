from pandas import DataFrame, Series
from sklearn.preprocessing import MinMaxScaler as ScikitLearnMinMaxScaler
from ml4iiot.pipeline.step import AbstractStep


class MinMaxScaler(AbstractStep):
    def __init__(self, config: dict):
        super().__init__(config)

        self.mode = 'normalize'

        self.source_range_min = self.get_config('source_range_min', default='window')
        self.source_range_max = self.get_config('source_range_max', default='window')
        self.source_column = self.get_config('source_column')

        self.target_range_min = self.get_config('target_range_min', default=0)
        self.target_range_max = self.get_config('target_range_min', default=1)
        self.target_column = self.get_config('target_column')

        self.smoothing_factor = self.get_config('smoothing_factor', default=0.1)
        self.moving_range_min = None
        self.moving_range_max = None

        self.fit_column = self.get_config('fit_column', default=None)
        self.first_run = True

        self.scaler = ScikitLearnMinMaxScaler(feature_range=(self.target_range_min, self.target_range_max))

    def process(self, data_frame: DataFrame) -> None:
        if self.mode == 'normalize':
            self.normalize(data_frame)
        else:
            self.denormalize(data_frame)

        self.first_run = False

    def normalize(self, data_frame: DataFrame) -> None:
        self.fit_scaler(data_frame[self.source_column])

        normalized = self.scaler.transform(data_frame[[self.source_column]])
        data_frame[self.target_column] = normalized[:, 0]

    def denormalize(self, data_frame: DataFrame) -> None:
        fit_series = data_frame[self.fit_column] if self.fit_column in data_frame else None

        self.fit_scaler(fit_series)

        denormalized = self.scaler.inverse_transform(data_frame[[self.source_column]])
        data_frame[self.target_column] = denormalized[:, 0]

    def fit_scaler(self, series: Series) -> None:
        min_value = self.get_source_range_min(series)
        max_value = self.get_source_range_max(series)

        if (self.first_run or
                self.source_range_min == 'window' or self.source_range_min == 'moving_window' or
                self.source_range_min == 'window' or self.source_range_min == 'moving_window'):
            self.scaler.fit([[min_value], [max_value]])

    def get_source_range_min(self, source):
        if self.source_range_min == 'window':
            return min(source)
        elif self.source_range_min == 'moving_window':
            if self.moving_range_min is None:
                self.moving_range_min = min(source)
            else:
                self.moving_range_min = min(source) * self.smoothing_factor + self.moving_range_min * (1 - self.smoothing_factor)

            return self.moving_range_min
        else:
            return self.source_range_min

    def get_source_range_max(self, source):
        if self.source_range_max == 'window':
            return max(source)
        elif self.source_range_max == 'moving_window':
            if self.moving_range_max is None:
                self.moving_range_max = max(source)
            else:
                self.moving_range_max = max(source) * self.smoothing_factor + self.moving_range_max * (1 - self.smoothing_factor)

            return self.moving_range_max
        else:
            return self.source_range_max
