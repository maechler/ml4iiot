import datetime
from pandas import DataFrame
from ml4iiot.input.windowing.abstractwindowing import AbstractWindowingStrategy


class TimeBasedWindowingStrategy(AbstractWindowingStrategy):

    def __init__(self, config):
        super().__init__(config)

        self.running_data_frame = None
        self.window_next_start_timestamp = None
        self.resample_config = self.get_config('resample')
        self.input = self.get_config('input')
        self.batch_size = int(self.get_config('batch_size'))
        self.window_size_time_delta = self.size_to_timedelta(self.get_config('window_size'))
        self.stride_size_time_delta = self.size_to_timedelta(self.get_config('stride_size'))

    def next_window(self) -> DataFrame:
        if self.running_data_frame is None:
            self.running_data_frame = self.input.next_data_frame(batch_size=1)
            window_start_timestamp = self.running_data_frame.index[0]
        else:
            window_start_timestamp = self.window_next_start_timestamp

        window_end_timestamp = window_start_timestamp + self.window_size_time_delta

        while self.running_data_frame.index[-1] < window_end_timestamp:
            self.running_data_frame = self.running_data_frame.append(self.input.next_data_frame(batch_size=self.batch_size))

        self.running_data_frame = self.resample_batch(self.running_data_frame, self.resample_config)
        selected_window = self.running_data_frame[window_start_timestamp:window_end_timestamp]

        self.window_next_start_timestamp = window_start_timestamp + self.stride_size_time_delta
        # Drop data frame parts that are in the past
        self.running_data_frame = self.running_data_frame[self.running_data_frame.index > window_start_timestamp]

        return selected_window

    def size_to_timedelta(self, size: str) -> datetime.timedelta:
        int_size = int(size.rstrip('ms'))
        timedelta_milliseconds = int_size if 'ms' in size else 1000 * int_size

        return datetime.timedelta(milliseconds=timedelta_milliseconds)
