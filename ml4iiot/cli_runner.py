import yaml
import json
import os
import sys
from ml4iiot.pipeline import Pipeline
from ml4iiot.utility import get_cli_arguments


def run(config_path: str, config_format: str) -> None:
    with open(config_path) as config_file:
        if config_format == 'yaml':
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        elif config_format == 'json':
            config = json.load(config_file)
        else:
            raise ValueError('Config format "{0}" is not supported.'.format(config_format))

    pipeline = Pipeline(config['pipeline'])

    pipeline.run()


if __name__ == '__main__':
    args = get_cli_arguments()

    run(args.config_path, args.format)
