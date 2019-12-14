from pandas import DataFrame
from ml4iiot.output.abstractoutput import AbstractOutput
from ml4iiot.utility import instance_from_config, get_absolute_path


class CsvOutput(AbstractOutput):

    def __init__(self, config):
        super().__init__(config)

        self.do_output_condition = instance_from_config(self.get_config('do_output_condition', default={'class': 'ml4iiot.conditions.TrueCondition'}))
        self.output_file_path = get_absolute_path(self.get_config('output_file_path', default='./out/csv_output.csv'))
        self.date_format = self.get_config('date_format', default='%s')
        self.columns = self.get_config('columns', default=None)
        self.output_file = None
        self.write_header = True

    def init(self) -> None:
        super().init()

        self.output_file = open(str(self.output_file_path), 'w')

    def process(self, data_frame: DataFrame) -> None:
        if self.do_output_condition.evaluate(data_frame):
            data_frame.to_csv(self.output_file, header=self.write_header, date_format=self.date_format, columns=self.columns)

            self.write_header = False

    def destroy(self) -> None:
        super().destroy()

        self.output_file.close()
