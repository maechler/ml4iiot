from ml4iiot.algorithms.abstractalgorithm import AbstractAlgorithm
import pandas as pd


class Average(AbstractAlgorithm):

    def compute(self, input_frame):
        data_frame_data = {
            'index': [input_frame.index.tolist()[-1]]
        }

        for column in self.get_config('columns'):
            data_frame_data[column + '_average'] = [input_frame[column].mean()]

        data_frame = pd.DataFrame.from_dict(data_frame_data)
        data_frame.set_index('index', inplace=True)

        return data_frame
