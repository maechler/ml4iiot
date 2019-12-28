from abc import ABC, abstractmethod
import json
import pandas as pd
from kafka.consumer.fetcher import ConsumerRecord
from pandas import DataFrame
from ml4iiot.input.abstractinput import AbstractInput
from kafka import KafkaConsumer
from ml4iiot.utility import append_value_to_dict_list, instance_from_config, get_recursive_config


class KafkaInput(AbstractInput):

    def __init__(self, config):
        super().__init__(config)

        self.consumer = None
        self.topics_mapping = self.get_config('kafka_topics_mapping')
        self.input_mapper = instance_from_config(self.get_config('kafka_input_mapper', default={'class': 'ml4iiot.input.kafka.JsonInputMapper'}))

    def init(self):
        super().init()

        self.consumer = KafkaConsumer(
            *self.topics_mapping.keys(),
            bootstrap_servers=[self.get_config('kafka_server', default='localhost:9092')],
            enable_auto_commit=True
        )

    def next_data_frame(self, batch_size: int = 1) -> DataFrame:
        pandas_dict = {}
        record_count = 0

        while record_count < batch_size:
            record = self.consumer.__next__()
            record_dict = self.input_mapper.from_kafka_record_to_dict(record)

            for key, value in record_dict.items():
                if key == self.index_column:
                    append_value_to_dict_list(pandas_dict, self.index_column, value)
                else:
                    append_value_to_dict_list(pandas_dict, self.topics_mapping[record.topic], value)

            record_count = record_count + 1

        data_frame = pd.DataFrame.from_dict(pandas_dict)
        data_frame.set_index(self.index_column, inplace=True)

        return data_frame

    def destroy(self) -> None:
        super().destroy()

        self.consumer.close()


class AbstractInputMapper(ABC):
    def __init__(self, config):
        self.config = config

    def get_config(self, *args, **kwargs):
        return get_recursive_config(self.config, *args, **kwargs)

    @abstractmethod
    def from_kafka_record_to_dict(self, record: ConsumerRecord) -> dict:
        pass


class JsonInputMapper(AbstractInputMapper):
    def from_kafka_record_to_dict(self, record: ConsumerRecord) -> dict:
        return json.loads(record.value)
