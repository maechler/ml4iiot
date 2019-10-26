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
