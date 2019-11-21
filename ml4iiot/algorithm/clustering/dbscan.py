from collections import deque

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
        if not self.do_fit(data_frame) or not self.do_predict(data_frame):
            return

        reshaped = np.array(data_frame[self.source_column].values).reshape(-1, 1)

        self.db.fit(reshaped)

        data_frame['number_of_clusters'] = float('nan')
        data_frame['number_of_noise'] = float('nan')

        data_frame['number_of_clusters'].iloc[-1] = self.get_number_of_clusters()
        data_frame['number_of_noise'].iloc[-1] = self.get_number_of_noise()

    def get_number_of_clusters(self) -> int:
        return len(set(self.db.labels_)) - (1 if -1 in self.db.labels_ else 0)

    def get_number_of_noise(self) -> int:
        return list(self.db.labels_).count(-1)


class BatchDBSCAN(DBSCAN):

    def __init__(self, config: dict):
        super().__init__(config)

        self.queue_size = self.get_config('queue_size', default=100)
        self.queue = deque(maxlen=self.queue_size)

    def process(self, data_frame: DataFrame) -> None:
        if not self.do_fit(data_frame) or not self.do_predict(data_frame):
            return

        self.queue.append(data_frame[self.source_column].values)

        if len(self.queue) == self.queue_size:
            self.db.fit(list(self.queue))

            data_frame['number_of_clusters'] = float('nan')
            data_frame['number_of_noise'] = float('nan')

            data_frame['number_of_clusters'].iloc[-1] = self.get_number_of_clusters()
            data_frame['number_of_noise'].iloc[-1] = self.get_number_of_noise()
