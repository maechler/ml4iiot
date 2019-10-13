import datetime
import pandas as pd
from abc import ABC, abstractmethod
from collections import deque
from ml4iiot.utility import get_recursive_config, str2bool


class AbstractInput(ABC):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.iteration_started = False
        self.window_size = self.sanitize_size(self.get_config('window_size'))
        self.stride_size = self.sanitize_size(self.get_config('stride_size'))
        self.resample_enabled = str2bool(self.get_config('resample', 'enabled'))
        self.data_frame_queue = deque(maxlen=self.window_size)

        if self.resample_enabled:
            self.target_sampling_rate = self.get_config('resample', 'target_sampling_rate')
            self.window_size_time_delta = self.size_to_timedelta(self.get_config('window_size'))
            self.stride_size_time_delta = self.size_to_timedelta(self.get_config('stride_size'))
            self.running_data_frame = None
            self.window_next_start_timestamp = None

    def open(self):
        pass

    def close(self):
        pass

    @abstractmethod
    def next_data_frame(self, recommended_frame_size=None, recommended_end_timestamp=None):
        pass

    def __next__(self):
        return self.next_batch()

    def __iter__(self):
        self.iteration_started = False
        self.data_frame_queue.clear()

        return self

    def get_config(self, *args):
        return get_recursive_config(self.config, *args)

    def sanitize_size(self, size):
        return int(size.rstrip('ms')) if type(size) is str else size

    def size_to_timedelta(self, size):
        int_size = int(size.rstrip('ms'))
        timedelta_milliseconds = int_size if 'ms' in size else 1000 * int_size

        return datetime.timedelta(milliseconds=timedelta_milliseconds)

    def next_batch(self):
        return self.next_batch_with_resample() if self.resample_enabled else self.next_batch_without_resample()

    def next_batch_without_resample(self):
        if self.iteration_started:
            for _ in range(self.stride_size):
                self.data_frame_queue.append(self.next_data_frame())
        else:
            self.iteration_started = True

        while len(self.data_frame_queue) < self.window_size:
            self.data_frame_queue.append(self.next_data_frame())

        return pd.concat(list(self.data_frame_queue))

    def next_batch_with_resample(self):
        if not self.iteration_started:
            self.running_data_frame = self.next_data_frame(recommended_frame_size=1)
            self.iteration_started = True
            window_start_timestamp = self.running_data_frame.index[-1]
        else:
            window_start_timestamp = self.window_next_start_timestamp

        window_end_timestamp = window_start_timestamp + self.window_size_time_delta

        while self.running_data_frame.index[-1] < window_end_timestamp:
            self.running_data_frame = self.running_data_frame.append(self.next_data_frame(recommended_frame_size=200))

        self.running_data_frame = self.resample_batch(self.running_data_frame)
        selected_window = self.running_data_frame[window_start_timestamp:window_end_timestamp]

        self.window_next_start_timestamp = window_start_timestamp + self.stride_size_time_delta
        self.running_data_frame = self.running_data_frame[self.running_data_frame.index > window_start_timestamp]

        return selected_window

    def resample_batch(self, batch):
        if not str2bool(self.get_config('resample', 'enabled')):
            return batch

        method = self.get_config('resample', 'method')
        resampled_batch = batch.resample(self.target_sampling_rate)

        if method == 'ffill':
            return resampled_batch.ffill()
        elif method == 'bfill':
            return resampled_batch.bfill()
        elif method == 'fill_value':
            return resampled_batch.asfreq(fill_value=self.get_config('resample', 'fill_value'))
        elif method == 'interpolate':
            return resampled_batch.interpolate(method=self.get_config('resample', 'interpolation_method'))
        else:
            return resampled_batch
