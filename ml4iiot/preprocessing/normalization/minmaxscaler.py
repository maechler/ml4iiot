from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler as ScikitLearnMinMaxScaler
from ml4iiot.pipeline.step import AbstractStep


class MinMaxScaler(AbstractStep):
    def __init__(self, config: dict):
        super().__init__(config)

        self.scaler = ScikitLearnMinMaxScaler(feature_range=(0, 1))
        self.column_mapping = self.get_config('column_mapping')
        self.source_column = self.get_config('source_column', default=None)

    def process(self, data_frame: DataFrame) -> None:
        if self.source_column is None:
            self.normalize(data_frame)
        else:
            self.denormalize(data_frame)

    def normalize(self, data_frame: DataFrame) -> None:
        for source_column, target_column in self.column_mapping.items():
            if source_column not in data_frame:
                continue

            source = data_frame[[source_column]]
            normalized = self.scaler.fit_transform(source)
            data_frame[target_column] = normalized[:, 0]

    def denormalize(self, data_frame: DataFrame) -> None:
        for source_column, target_column in self.column_mapping.items():
            if source_column not in data_frame:
                continue

            source = data_frame[[self.source_column]]
            source_n = data_frame[[source_column]]

            self.scaler.fit(source)

            reconstruction = self.scaler.inverse_transform(source_n)
            data_frame[target_column] = reconstruction[:, 0]
