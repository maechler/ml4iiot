from pandas import DataFrame
from ml4iiot.output.abstractoutput import AbstractOutput
from ml4iiot.utility import get_cli_arguments, get_current_out_path


class ConfigOutput(AbstractOutput):
    def __init__(self, config):
        super().__init__(config)

        self.cli_arguments = get_cli_arguments()
        self.config_content = None

    def init(self) -> None:
        with open(self.cli_arguments.config_path, 'r') as config_file_handle:
            self.config_content = config_file_handle.read()

    def process(self, data_frame: DataFrame) -> None:
        pass

    def destroy(self) -> None:
        target_file_name = 'conf.' + self.cli_arguments.format
        save_path = get_current_out_path(target_file_name)

        with open(save_path, 'w') as target_file_handle:
            target_file_handle.write(self.config_content)
