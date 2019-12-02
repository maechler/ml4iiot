import sys
from pandas import DataFrame
from ml4iiot.algorithm.abstractalgorithm import AbstractAlgorithm
import numpy as np
from ml4iiot.algorithm.stochastic.mad import compute_mad


class ModifiedZScore(AbstractAlgorithm):

    def __init__(self, config):
        super().__init__(config)

        self.column_mapping = self.get_config('column_mapping')
        self.anomaly_threshold = self.get_config('anomaly_threshold', default=3.5)
        self.zscore_constant = self.get_config('zscore_constant', default=0.6745)

    def process(self, data_frame: DataFrame) -> None:
        for source_column, target_column in self.column_mapping.items():
            data_frame[target_column] = float('nan')
            data_frame[target_column + '_is_anomaly'] = False

            data = np.array(data_frame[source_column].values)
            median = np.median(data)
            mad = compute_mad(data_frame[source_column].values)

            if mad == 0:
                mad = sys.float_info.min

            for index, row in data_frame.iterrows():
                zscore = (self.zscore_constant * (row[source_column] - median)) / mad
                data_frame[target_column].loc[index] = zscore
                data_frame[target_column + '_is_anomaly'].loc[index] = abs(zscore) > self.anomaly_threshold
