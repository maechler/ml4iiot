from ml4iiot.output.abstractoutput import AbstractOutput
import sys
from ml4iiot.utility import str2bool


class StdOutput(AbstractOutput):

    def __init__(self, config):
        super().__init__(config)

        self.show_columns_progress = self.get_config('show_columns_progress', default=[])
        self.show_input = str2bool(self.get_config('show_input', default=False))
        self.show_output = str2bool(self.get_config('show_output', default=False))

    def emit(self, input_frame, output_frame):
        output = ''
        progress_output = []

        if len(self.show_columns_progress) > 0:
            output += '\r'

        for progress_config in self.show_columns_progress:
            source = input_frame if progress_config['source'] == 'input' else output_frame
            column_name = progress_config['column']
            value = source.index[-1] if column_name == 'index' else source[column_name][-1]

            progress_output.append('{0}={1}'.format(column_name, value))

        output += ', '.join(progress_output)

        if self.show_input:
            output += str(input_frame) + '\n'

        if self.show_output:
            output += str(output_frame) + '\n'

        sys.stdout.write(output)
