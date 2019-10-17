import importlib
from pathlib import Path


def str2bool(v):
    return str(v).lower() in ('yes', 'true', '1')


def instance_from_config(config):
    module_name, class_name = config['class'].rsplit('.', 1)
    class_config = config['config'] if 'config' in config else {}

    return getattr(importlib.import_module(module_name), class_name)(class_config)


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_absolute_path(path) -> Path:
    path_obj = Path(path)

    return path_obj if path_obj.is_absolute() else get_project_root().joinpath(path_obj)


def get_recursive_config(config, *args, **kwargs):
    for arg in args:
        if arg in config:
            config = config[arg]
        else:
            if 'default' not in kwargs:
                raise ValueError('Config with name "' + str(arg) + '" is not set.')
            else:
                return kwargs['default']

    return config
