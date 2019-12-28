from pandas import DataFrame
from ml4iiot.input.abstractinput import AbstractInput
import pandas as pd
import csv
from ml4iiot.utility import datetime_string_to_object, append_value_to_dict_list


class CsvInput(AbstractInput):

    def __init__(self, config):
        super().__init__(config)

        self.csv_file = None
        self.reader = None
        self.stop_iteration_raised = False
        self.columns = self.get_config('columns')

    def init(self) -> None:
        super().init()

        self.stop_iteration_raised = False
        self.csv_file = open(self.get_config('csv_file'))
        self.reader = csv.DictReader(self.csv_file, delimiter=self.get_config('delimiter'))

    def next_data_frame(self, batch_size: int = 1) -> DataFrame:
        if self.stop_iteration_raised:
            # We ran out of values in the previous call, raise exception
            raise StopIteration

        pandas_dict = {}

        for i in range(0, batch_size):
            try:
                row = next(self.reader)
                self.add_row_to_pandas_dict(pandas_dict, row)
            except StopIteration:
                # Catch exception to at least return all rows possible
                self.stop_iteration_raised = True

        data_frame = pd.DataFrame.from_dict(pandas_dict)
        data_frame.set_index(self.index_column, inplace=True)

        return data_frame

    def add_row_to_pandas_dict(self, pandas_dict: dict, row: dict) -> None:
        for key, value in row.items():
            if key not in self.columns:
                continue

            if self.get_column_type(key) == 'datetime':
                append_value_to_dict_list(
                    pandas_dict,
                    key,
                    datetime_string_to_object(value, self.get_config('columns', key, 'datetime_format'))
                )
            elif self.get_column_type(key) == 'int':
                append_value_to_dict_list(pandas_dict, key, int(value))
            elif self.get_column_type(key) == 'float':
                append_value_to_dict_list(pandas_dict, key, float(value))
            elif self.get_column_type(key) == 'str':
                append_value_to_dict_list(pandas_dict, key, str(value))
            else:
                append_value_to_dict_list(pandas_dict, key, value)

    def get_column_type(self, column_name: str) -> str:
        column_config = self.columns[column_name]
        column_type = column_config if type(column_config) is str else column_config['type']

        if column_type == 'integer' or column_type == 'int':
            return 'int'
        elif column_type == 'string' or column_type == 'str':
            return 'str'
        else:
            return column_type

    def destroy(self) -> None:
        super().destroy()

        self.csv_file.close()
