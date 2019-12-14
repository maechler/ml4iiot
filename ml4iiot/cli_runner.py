import yaml
import json
import os
import sys
from ml4iiot.pipeline import Pipeline
from ml4iiot.utility import get_cli_arguments


def load_config(config_path: str, config_format: str) -> dict:
    with open(config_path) as config_file:
        if config_format == 'yaml':
            return yaml.load(config_file, Loader=yaml.FullLoader)
        elif config_format == 'json':
            return json.load(config_file)
        else:
            raise ValueError('Config format "{0}" is not supported.'.format(config_format))


def run(config_path: str, config_format: str) -> None:
    if os.path.isfile(config_path):
        config = load_config(config_path, config_format)
        pipeline = Pipeline(config['pipeline'])

        pipeline.run()
    elif os.path.isdir(config_path):
        config_folder = config_path

        for config_file in os.listdir(config_path):
            try:
                config_file_path = os.path.join(config_folder, config_file)
                config = load_config(config_file_path, config_format)

                sys.argv.append('-c')
                sys.argv.append(config_file_path)

                pipeline = Pipeline(config['pipeline'])

                pipeline.run()
            except Exception as e:
                print(e)
                pass
    else:
        print('Invalid config path "' + config_path + '" provided.')


if __name__ == '__main__':
    args = get_cli_arguments()

    run(args.config_path, args.format)
