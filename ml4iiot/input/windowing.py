import datetime
from abc import ABC, abstractmethod
from pandas import DataFrame
from ml4iiot.utility import get_recursive_config, str2bool


class AbstractWindowingStrategy(ABC):

    def __init__(self, config: dict):
        super().__init__()

        self.config = config

    @abstractmethod
    def next_window(self) -> DataFrame:
        pass

    def get_config(self, *args, **kwargs):
        return get_recursive_config(self.config, *args, **kwargs)

    def resample_batch(self, batch: DataFrame, resample_config: dict) -> DataFrame:
        if not str2bool(get_recursive_config(resample_config, 'enabled')):
            return batch

        target_sampling_rate = get_recursive_config(resample_config, 'target_sampling_rate')
        method = get_recursive_config(resample_config, 'method')
        resampled_batch = batch.resample(target_sampling_rate)

        if method == 'ffill':
            return resampled_batch.ffill()
        elif method == 'bfill':
            return resampled_batch.bfill()
        elif method == 'mean':
            interpolation_method = get_recursive_config(resample_config, 'interpolation_method')

            return resampled_batch.mean().interpolate(method=interpolation_method)
        elif method == 'fill_value':
            fill_value = get_recursive_config(resample_config, 'fill_value')

            return resampled_batch.asfreq(fill_value=fill_value)
        elif method == 'interpolate':
            interpolation_method = get_recursive_config(resample_config, 'interpolation_method')

            return resampled_batch.interpolate(method=interpolation_method)
        else:
            return resampled_batch


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