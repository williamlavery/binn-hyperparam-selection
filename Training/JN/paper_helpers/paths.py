"""Path-string helpers shared by the paper notebooks.

Contents
--------
- dictToPath"""

import os


def dictToPath(arg_dict, sep="/", kv_delim="_"):
    """Convert a dictionary into a path string like ``arg1_val1/arg2_val2``."""
    parts = [f"{key}{kv_delim}{value}" for key, value in arg_dict.items()]
    return os.path.join(*parts) if sep == "/" else sep.join(parts)
