import json
import os
import munch
import random
import numpy as np


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return config_dict


def process_config(jsonfile):
    config_dict = get_config_from_json(jsonfile)
    config_dict = expand_shell_vars(config_dict)
    config = munch.Munch(config_dict)
    if ('test' in config):
        config.test = munch.Munch(config.test)
    return config


# Function to recursively expand user path data (~) and env variables in a nested data structure such as a decoded dict from JSON
def expand_shell_vars(obj):
    if isinstance(obj, str):
        return os.path.expandvars(os.path.expanduser(obj))
    elif isinstance(obj, dict):
        return {key: expand_shell_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [expand_shell_vars(item) for item in obj]
    else:
        return obj

