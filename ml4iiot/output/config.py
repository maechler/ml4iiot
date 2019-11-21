import os
from datetime import datetime
from pandas import DataFrame
from ml4iiot.output.abstractoutput import AbstractOutput
from ml4iiot.utility import get_cli_arguments, get_absolute_path


class ConfigOutput(AbstractOutput):
    def __init__(self, config):
        super().__init__(config)

        self.cli_arguments = get_cli_arguments()
        self.save_path = self.get_config('save_path', default='./out/')
        self.config_content = None

    def init(self) -> None:
        with open(self.cli_arguments.config_path, 'r') as config_file_handle:
            self.config_content = config_file_handle.read()

    def process(self, data_frame: DataFrame) -> None:
        pass

    def destroy(self) -> None:
        target_file_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_conf.' + self.cli_arguments.format
        save_path = os.path.join(str(get_absolute_path(self.save_path)), datetime.now().strftime('%Y_%m_%d'))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        target_path = os.path.join(save_path, target_file_name)

        with open(target_path, 'w') as target_file_handle:
            target_file_handle.write(self.config_content)
