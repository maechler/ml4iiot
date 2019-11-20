from datetime import datetime
from shutil import copyfile
from pandas import DataFrame
from ml4iiot.output.abstractoutput import AbstractOutput
from ml4iiot.utility import get_cli_arguments, get_absolute_path


class ConfigOutput(AbstractOutput):
    def __init__(self, config):
        super().__init__(config)

        self.save_path = self.get_config('save_path', default='./out/')

    def process(self, data_frame: DataFrame) -> None:
        pass

    def destroy(self) -> None:
        args = get_cli_arguments()
        config_path = args.config_path
        config_format = args.format
        target_file_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_conf.' + config_format
        target_path = get_absolute_path(self.save_path + target_file_name)

        copyfile(config_path, target_path)
