import argparse
import importlib
import os
from argparse import Namespace
from datetime import datetime, timezone
import dateutil
from pathlib import Path
from typing import Union


def str2bool(v) -> bool:
    return str(v).lower() in ('yes', 'true', '1')


def instance_from_config(config: dict) -> any:
    module_name, class_name = config['class'].rsplit('.', 1)
    class_config = config['config'] if 'config' in config else {}

    return getattr(importlib.import_module(module_name), class_name)(class_config)


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_absolute_path(path: Union[Path, str]) -> Path:
    path_obj = Path(path)

    return path_obj if path_obj.is_absolute() else get_project_root().joinpath(path_obj)


def get_current_out_path(file_name=None) -> str:
    config_name = get_file_name_from_path((get_cli_arguments()).config_path)
    date_time = datetime.now().strftime('%Y_%m_%d')
    i = 1

    out_path = os.path.join(str(get_absolute_path('./out')), date_time, config_name)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if file_name is not None:
        out_path_raw = out_path
        out_path = os.path.join(out_path_raw, datetime.now().strftime('%H_%M_%S') + '_' + file_name)

        while os.path.exists(out_path):
            out_path = os.path.join(out_path_raw, datetime.now().strftime('%H_%M_%S') + '_' + str(i) + '_' + file_name)
            i = i + 1

    return out_path


def get_recursive_config(config: dict, *args, **kwargs):
    for arg in args:
        if arg in config:
            config = config[arg]
        else:
            if 'default' not in kwargs:
                raise ValueError('Config with name "' + str(arg) + '" is not set.')
            else:
                return kwargs['default']

    return config


def datetime_string_to_object(datetime_string: str, datetime_format: str) -> datetime:
    if datetime_format == 'timestamp':
        return datetime.fromtimestamp(float(datetime_string), tz=timezone.utc)
    elif datetime_format == 'iso':
        return dateutil.parser.isoparse(datetime_string)
    else:
        return datetime.strptime(datetime_string, datetime_format)


def get_cli_arguments() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_path', help='Path to a config file or folder containing multiple config files.', type=str, required=True)
    parser.add_argument('-f', '--format', help='Format of the config file.', type=str, default='yaml')

    return parser.parse_args()


def get_file_name_from_path(path: str) -> str:
    return str(os.path.basename(path).split('.')[0])


def append_value_to_dict_list(my_dict: dict, key: str, value) -> None:
    if key not in my_dict:
        my_dict[key] = [value]
    else:
        my_dict[key].append(value)
