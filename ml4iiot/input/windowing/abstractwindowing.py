from abc import ABC, abstractmethod
from ml4iiot.utility import get_recursive_config, str2bool


class AbstractWindowingStrategy(ABC):
    def __init__(self, config):
        super().__init__()

        self.config = config

    @abstractmethod
    def next_window(self):
        pass

    def get_config(self, *args):
        return get_recursive_config(self.config, *args)

    def resample_batch(self, batch, resample_config):
        if not str2bool(get_recursive_config(resample_config, 'enabled')):
            return batch

        target_sampling_rate = get_recursive_config(resample_config, 'target_sampling_rate')
        method = get_recursive_config(resample_config, 'method')
        resampled_batch = batch.resample(target_sampling_rate)

        if method == 'ffill':
            return resampled_batch.ffill()
        elif method == 'bfill':
            return resampled_batch.bfill()
        elif method == 'fill_value':
            fill_value = get_recursive_config(resample_config, 'fill_value')

            return resampled_batch.asfreq(fill_value=fill_value)
        elif method == 'interpolate':
            interpolation_method = get_recursive_config(resample_config, 'interpolation_method')

            return resampled_batch.interpolate(method=interpolation_method)
        else:
            return resampled_batch
