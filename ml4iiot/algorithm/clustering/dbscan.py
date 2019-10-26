from pandas import DataFrame
from ml4iiot.algorithm.abstractalgorithm import AbstractAlgorithm
from sklearn.cluster import DBSCAN as SkDBSCAN
import numpy as np


class DBSCAN(AbstractAlgorithm):

    def __init__(self, config: dict):
        super().__init__(config)

        self.epsilon = self.get_config('epsilon')
        self.min_samples = self.get_config('min_samples')
        self.source_column = self.get_config('source_column')
        self.db = SkDBSCAN(eps=self.epsilon, min_samples=self.min_samples)

    def process(self, data_frame: DataFrame) -> None:
        reshaped = np.array(data_frame[self.source_column].values).reshape(-1, 1)

        self.db.fit(reshaped)

        number_of_clusters = len(set(self.db.labels_)) - (1 if -1 in self.db.labels_ else 0)
        number_of_noise = list(self.db.labels_).count(-1)

        data_frame['number_of_clusters'] = float('nan')
        data_frame['number_of_noise'] = float('nan')

        data_frame['number_of_clusters'].iloc[-1] = number_of_clusters
        data_frame['number_of_noise'].iloc[-1] = number_of_noise
