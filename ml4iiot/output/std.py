from ml4iiot.output.abstractoutput import AbstractOutput
import sys
from ml4iiot.utility import str2bool


class StdOutput(AbstractOutput):

    def __init__(self, config):
        super().__init__(config)

        self.show_columns_progress = self.get_config('show_columns_progress', default=[])
        self.show_data_frame = str2bool(self.get_config('show_data_frame', default=False))

    def process(self, data_frame):
        output = ''
        progress_output = []

        if len(self.show_columns_progress) > 0:
            output += '\r'

        for progress_config in self.show_columns_progress:
            column_name = progress_config['column']
            value = data_frame.index[-1] if column_name == 'index' else data_frame[column_name][-1]

            progress_output.append('{0}={1}'.format(column_name, value))

        output += ', '.join(progress_output)

        if self.show_data_frame:
            output += str(data_frame) + '\n'

        sys.stdout.write(output)
