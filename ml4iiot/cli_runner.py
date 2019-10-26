import argparse
import yaml
import json
from ml4iiot.pipeline.pipeline import Pipeline


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
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_path', help='Path to a config file.', type=str, required=True)
    parser.add_argument('-f', '--format', help='Format of the config file.', type=str, default='yaml')

    args = parser.parse_args()

    run(args.config_path, args.format)
