import importlib
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
