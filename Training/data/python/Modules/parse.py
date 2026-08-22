"""Small parsing and serialization helpers for the data pipeline.

Contents
--------
- parse_function: inspect a callable signature and defaults.
- dictToPath: serialize a parameter dictionary into a path-like string.
- pathToDict: parse a path-like string back into a parameter dictionary.
- clone_empty_instance: create an uninitialized instance of an object's class.
- to_torch_grad: convert an array to a gradient-enabled Torch tensor.
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

def pathToDict(path_str, sep='/', kv_delim='_'):
    """
    Converts a path string like 'arg1_val1/arg2_val2' back into a dictionary.

    Parameters:
        path_str (str): Path string containing key-value pairs.
        sep (str): Separator between key-value pairs (default: '/').
        kv_delim (str): Delimiter between key and value (default: '_').

    Returns:
        dict: Dictionary of key-value pairs.
    """
    result = {}
    parts = path_str.split(sep)
    for part in parts:
        if kv_delim not in part:
            raise ValueError(f"Missing key-value delimiter '{kv_delim}' in part: {part}")
        key, value = part.split(kv_delim, 1)
        result[key] = value
    return result
    
def clone_empty_instance(obj):
    """
    Create a new instance of the same class as `obj`,
    but with no attributes (clean __dict__).
    """
    cls = obj.__class__
    new_obj = cls.__new__(cls)  # bypass __init__
    return new_obj

def to_torch_grad(ndarray, device):
    """Convert an array to a float Torch tensor on `device` with gradients enabled."""
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr
