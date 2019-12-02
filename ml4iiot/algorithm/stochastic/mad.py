from pandas import DataFrame
from ml4iiot.algorithm.abstractalgorithm import AbstractAlgorithm
import numpy as np


class Mad(AbstractAlgorithm):

    def __init__(self, config):
        super().__init__(config)

        self.column_mapping = self.get_config('column_mapping')

    def process(self, data_frame: DataFrame) -> None:
        for source_column, target_column in self.column_mapping.items():
            mad = compute_mad(data_frame[source_column].values)

            data_frame[target_column] = float('nan')
            data_frame[target_column].iloc[-1] = mad


def compute_mad(data: list):
    data = np.array(data)
    median = np.median(data)
    deviations = data - median
    absolute_deviations = np.abs(deviations)
    mad = np.median(absolute_deviations)

    return mad
