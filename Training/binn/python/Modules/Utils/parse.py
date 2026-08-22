"""Small parsing and serialization helpers for the BINN pipeline.

Contents
--------
- parse_function: inspect a callable signature and defaults.
- dictToPath: serialize a parameter dictionary into a path-like string.
- clone_empty_instance: create an uninitialized instance of an object's class.
- to_torch_grad: convert an array to a gradient-enabled Torch tensor.
- unravel_one_level: flatten a one-level nested dictionary into a single parameter dictionary.
- checker: report whether a filesystem path already exists on disk.
"""

import inspect
import os
import torch


def parse_function(func):
    """
    Parses a function and returns its name and keyword arguments with default values.

    Parameters:
        func (function): The function object to parse.

    Returns:
        dict: {"function_name": {arg_name: default_value or None}}
    """
    sig = inspect.signature(func)
    arg_info = {}
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            arg_info[name] = None  # No default = required positional or keyword arg
        else:
            arg_info[name] = param.default

    return {func.__name__: arg_info}


def dictToPath(arg_dict, sep='/', kv_delim='_'):
    """
    Converts a dictionary into a path string like arg1_val1/arg2_val2.

    Parameters:
        arg_dict (dict): Dictionary of key-value pairs.
        sep (str): Separator between key-value pairs (default: '/').
        kv_delim (str): Delimiter between key and value (default: '_').

    Returns:
        str: Generated path string.
    """
    parts = [f"{key}{kv_delim}{value}" for key, value in arg_dict.items()]
    return os.path.join(*parts) if sep == '/' else sep.join(parts)

    
def clone_empty_instance(obj):
    """
    Create a new instance of the same class as `obj`,
    but with no attributes (clean __dict__).
    """
    cls = obj.__class__
    new_obj = cls.__new__(cls)  # bypass __init__
    return new_obj

def to_torch_grad(ndarray, device):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr


def unravel_one_level(d):
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result.update(value)  # flatten one level
        else:
            result[key] = value
    return result

def checker(path):
    split = path.split('/')
    combined = ''
    for component in split:
        combined += component + '/'
        try:
            os.listdir(combined)
        except:
            print(f'No directory found: {combined}')
            print("Last good directory:", combined.rsplit('/',2)[0])
            print(os.listdir(combined.rsplit('/',2)[0]))
            return combined.rsplit('/',2)[0]
