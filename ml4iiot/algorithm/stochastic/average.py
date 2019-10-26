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
