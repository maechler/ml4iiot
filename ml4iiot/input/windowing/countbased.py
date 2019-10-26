from pandas import DataFrame
from ml4iiot.input.windowing.abstractwindowing import AbstractWindowingStrategy


class CountBasedWindowingStrategy(AbstractWindowingStrategy):

    def __init__(self, config):
        super().__init__(config)

        self.window_size = int(self.get_config('window_size'))
        self.stride_size = int(self.get_config('stride_size'))
        self.batch_size = int(self.get_config('batch_size'))
        self.input = self.get_config('input')
        self.running_data_frame = None
        self.previous_start_index = None

    def next_window(self) -> DataFrame:
        if self.running_data_frame is None:
            self.running_data_frame = self.input.next_data_frame(batch_size=1)
            previous_start_iloc = None
            start_iloc = 0
            end_iloc = self.window_size
        else:
            previous_start_iloc = self.running_data_frame.index.get_loc(self.previous_start_index)
            start_iloc = previous_start_iloc + self.stride_size  # Shift start position by stride
            end_iloc = start_iloc + self.window_size

        while len(self.running_data_frame.index) < end_iloc:
            next_data_frame = self.input.next_data_frame(batch_size=self.batch_size)
            self.running_data_frame = self.running_data_frame.append(next_data_frame)

        selected_window = self.running_data_frame.iloc[start_iloc:end_iloc]
        self.previous_start_index = selected_window.index[0]

        if previous_start_iloc is not None:
            # Drop data frame parts that are in the past
            self.running_data_frame = self.running_data_frame.iloc[previous_start_iloc:]

        return selected_window
