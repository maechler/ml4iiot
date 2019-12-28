from abc import ABC, abstractmethod
from pandas import Series
from ml4iiot.output.abstractoutput import AbstractOutput
from kafka import KafkaProducer
import json
from ml4iiot.utility import instance_from_config, get_recursive_config


class KafkaOutput(AbstractOutput):

    def __init__(self, config):
        super().__init__(config)

        self.producer = None
        self.topics_mapping = self.get_config('kafka_topics_mapping')
        self.output_mapper = instance_from_config(self.get_config('kafka_output_mapper', default={'class': 'ml4iiot.output.kafka.JsonOutputMapper'}))

    def init(self):
        super().init()

        self.producer = KafkaProducer(bootstrap_servers=[self.get_config('kafka_server', default='localhost:9092')])

    def process(self, data_frame) -> None:
        for data_frame_column, topic in self.topics_mapping.items():
            mapped_value = self.output_mapper.series_to_bytes(data_frame[data_frame_column])

            self.producer.send(topic, value=mapped_value)

    def destroy(self) -> None:
        super().destroy()

        self.producer.close()


class AbstractOutputMapper(ABC):
    def __init__(self, config):
        self.config = config

    def get_config(self, *args, **kwargs):
        return get_recursive_config(self.config, *args, **kwargs)

    @abstractmethod
    def series_to_bytes(self, series: Series) -> bytes:
        pass


class JsonOutputMapper(AbstractOutputMapper):
    def series_to_bytes(self, series: Series) -> bytes:
        data_frame_dict = {
            'timestamp': int(series.index[-1]),
            'value': float(series.iloc[-1])
        }

        return json.dumps(data_frame_dict).encode('utf-8')
