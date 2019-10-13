import argparse
import yaml
from ml4iiot.pipeline.pipeline import Pipeline


def run(config_path):
    config_file = open(config_path)
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    pipeline = Pipeline(config['pipeline'])

    config_file.close()
    pipeline.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_path', help='Path to a yaml config file.', type=str, required=True)

    args = parser.parse_args()

    run(args.config_path)
