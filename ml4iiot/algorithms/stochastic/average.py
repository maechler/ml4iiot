from ml4iiot.algorithms.abstractalgorithm import AbstractAlgorithm
import pandas as pd


class Average(AbstractAlgorithm):

    def __init__(self, config):
        super().__init__(config)

        self.columns = self.get_config('columns')

    def compute(self, input_frame):
        data_frame_data = {
            'index': [input_frame.index.tolist()[-1]]
        }

        for column in self.columns:
            data_frame_data[self.columns[column]] = [input_frame[column].mean()]

        data_frame = pd.DataFrame.from_dict(data_frame_data)
        data_frame.set_index('index', inplace=True)

        return data_frame
