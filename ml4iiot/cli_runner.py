import argparse
import yaml
import json
from ml4iiot.pipeline.pipeline import Pipeline


def run(config_path, format):
    if format == 'yaml':
        config_file = open(config_path)
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        config_file.close()
    elif format == 'json':
        config_file = open(config_path)
        config = json.load(config_file)
        config_file.close()
    else:
        raise ValueError('Config format "{0}" is not supported.'.format(format))

    pipeline = Pipeline(config['pipeline'])

    pipeline.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_path', help='Path to a config file.', type=str, required=True)
    parser.add_argument('-f', '--format', help='Format of the config file.', type=str, required=True, default='yaml')

    args = parser.parse_args()

    run(args.config_path, args.format)
