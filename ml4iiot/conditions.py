from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from pandas import DataFrame
from ml4iiot.utility import get_recursive_config, instance_from_config, str2bool, datetime_string_to_object


class AbstractCondition(ABC):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.inverted = str2bool(self.get_config('inverted', default=False))

    def get_config(self, *args, **kwargs):
        return get_recursive_config(self.config, *args, **kwargs)

    def evaluate(self, data_frame: DataFrame) -> bool:
        return not self.evaluate_logic(data_frame) if self.inverted else self.evaluate_logic(data_frame)

    @abstractmethod
    def evaluate_logic(self, data_frame: DataFrame) -> bool:
        pass


class CompositeCondition(AbstractCondition):
    def __init__(self, config):
        super().__init__(config)

        self.conditions = []
        self.operator = self.get_config('operator', default='and')

        for condition_config in self.get_config('conditions', default=[]):
            self.conditions.append(instance_from_config(condition_config))

    def evaluate_logic(self, data_frame: DataFrame) -> bool:
        if self.operator == 'and':
            result = True

            for condition in self.conditions:
                result = result and condition.evaluate(data_frame)

            return result
        elif self.operator == 'or':
            result = False

            for condition in self.conditions:
                result = result or condition.evaluate(data_frame)

            return result
        else:
            raise ValueError('Condition operator "' + str(self.operator) + '" is not supported.')


class TrueCondition(AbstractCondition):
    def evaluate_logic(self, data_frame: DataFrame) -> bool:
        return True


class FalseCondition(AbstractCondition):
    def evaluate_logic(self, data_frame: DataFrame) -> bool:
        return False


class WeekdayCondition(AbstractCondition):
    def __init__(self, config):
        super().__init__(config)

        self.weekdays = self.get_config('weekdays')

    def evaluate_logic(self, data_frame: DataFrame) -> bool:
        days = data_frame.resample('D').pad()

        if len(days) > 7:
            raise ValueError('Dataframe is larger than 7 days, can not apply weekday condition.')

        for index, row in days.iterrows():
            if index.day_name() in self.weekdays:
                return True

        return False


class DaytimeCondition(AbstractCondition):
    def __init__(self, config):
        super().__init__(config)

        self.start_time = datetime.strptime(self.get_config('start_time'), '%H:%M:%S').time()
        self.end_time = datetime.strptime(self.get_config('end_time'), '%H:%M:%S').time()

    def evaluate_logic(self, data_frame: DataFrame) -> bool:
        data_frame_start_timestamp = data_frame.index[0]
        data_frame_end_timestamp = data_frame.index[-1]
        time_delta = data_frame_end_timestamp - data_frame_start_timestamp

        if time_delta.days > 0:
            raise ValueError('Dataframe is larger than 1 day, can not apply daytime condition.')

        if data_frame_start_timestamp.time() < self.start_time or data_frame_end_timestamp.time() < self.start_time:
            return False

        if data_frame_start_timestamp.time() > self.end_time or data_frame_end_timestamp.time() > self.end_time:
            return False

        return True


class DatetimeCondition(AbstractCondition):
    def __init__(self, config):
        super().__init__(config)

        self.start_datetime = None
        self.end_datetime = None
        start_datetime_config = self.get_config('start_datetime', default=None)
        end_datetime_config = self.get_config('end_datetime', default=None)

        if start_datetime_config is not None:
            self.start_datetime = datetime_string_to_object(start_datetime_config, 'iso')

        if end_datetime_config is not None:
            self.end_datetime = datetime_string_to_object(end_datetime_config, 'iso')

    def evaluate_logic(self, data_frame: DataFrame) -> bool:
        data_frame_start_timestamp = data_frame.index[0]
        data_frame_end_timestamp = data_frame.index[-1]

        if self.start_datetime is not None:
            if data_frame_start_timestamp < self.start_datetime or data_frame_end_timestamp < self.start_datetime:
                return False

        if self.end_datetime is not None:
            if data_frame_start_timestamp > self.end_datetime or data_frame_end_timestamp > self.end_datetime:
                return False

        return True


class TimeDeltaCondition(AbstractCondition):
    def __init__(self, config):
        super().__init__(config)

        self.min_time_delta = None
        self.max_time_delta = None

        if self.get_config('min_time_delta', default=None) is not None:
            self.min_time_delta = timedelta(
                microseconds=self.get_config('min_time_delta', 'microseconds', default=0),
                milliseconds=self.get_config('min_time_delta', 'milliseconds', default=0),
                seconds=self.get_config('min_time_delta', 'seconds', default=0),
                minutes=self.get_config('min_time_delta', 'minutes', default=0),
                hours=self.get_config('min_time_delta', 'hours', default=0),
                days=self.get_config('min_time_delta', 'days', default=0),
                weeks=self.get_config('min_time_delta', 'weeks', default=0),
            )

        if self.get_config('max_time_delta', default=None) is not None:
            self.max_time_delta = timedelta(
                microseconds=self.get_config('max_time_delta', 'microseconds', default=0),
                milliseconds=self.get_config('max_time_delta', 'milliseconds', default=0),
                seconds=self.get_config('max_time_delta', 'seconds', default=0),
                minutes=self.get_config('max_time_delta', 'minutes', default=0),
                hours=self.get_config('max_time_delta', 'hours', default=0),
                days=self.get_config('max_time_delta', 'days', default=0),
                weeks=self.get_config('max_time_delta', 'weeks', default=0),
            )

    def evaluate_logic(self, data_frame: DataFrame) -> bool:
        previous_index = None

        for index, _ in data_frame.iterrows():
            if previous_index is not None:
                if self.min_time_delta and index - previous_index < self.min_time_delta:
                    return False
                elif self.max_time_delta and index - previous_index > self.max_time_delta:
                    return False

            previous_index = index

        return True


class CounterCondition(AbstractCondition):
    def __init__(self, config):
        super().__init__(config)

        self.counter = 0
        self.counter_target = self.get_config('counter_target')

    def evaluate_logic(self, data_frame: DataFrame) -> bool:
        self.counter = self.counter + 1

        return self.counter % self.counter_target == 0


class StanardDeviationCondition(AbstractCondition):
    def __init__(self, config):
        super().__init__(config)

        self.column = self.get_config('column')
        self.std_min = self.get_config('std_min', default=None)
        self.std_max = self.get_config('std_max', default=None)

    def evaluate_logic(self, data_frame: DataFrame) -> bool:
        std = data_frame[self.column].std()
        in_range = True

        if self.std_min is not None:
            in_range = in_range and std > self.std_min

        if self.std_max is not None:
            in_range = in_range and std < self.std_max

        return in_range
