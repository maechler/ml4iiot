from abc import abstractmethod
from pandas import DataFrame, Series
from scipy.signal import detrend
from ml4iiot.pipeline import AbstractStep
import numpy as np


class AbstractAggregateTransform(AbstractStep):
    def __init__(self, config):
        super().__init__(config)

        self.column_mapping = self.get_config('column_mapping')

    def process(self, data_frame: DataFrame) -> None:
        for source_column, target_column in self.column_mapping.items():
            data_frame[target_column] = float('nan')

            data_frame[target_column].iloc[-1] = self.transform(data_frame[source_column])

    @abstractmethod
    def transform(self, series: Series):
        pass


class Average(AbstractAggregateTransform):
    def transform(self, series: Series):
        return series.mean()


class StandardDeviation(AbstractAggregateTransform):
    def transform(self, series: Series):
        return series.std()


class Minimum(AbstractAggregateTransform):
    def transform(self, series: Series):
        return series.min()


class Maximum(AbstractAggregateTransform):
    def transform(self, series: Series):
        return series.max()


class FastFourierTransform(AbstractStep):
    def __init__(self, config):
        super().__init__(config)

        self.column_mapping = self.get_config('column_mapping')
        self.detrend = self.get_config('detrend', default=False)

    def process(self, data_frame: DataFrame) -> None:
        for source_column, target_column in self.column_mapping.items():
            if self.detrend:
                fft = np.fft.fft(detrend(data_frame[source_column].values))
            else:
                fft = np.fft.fft(data_frame[source_column].values)

            data_frame[target_column] = np.absolute(fft)
