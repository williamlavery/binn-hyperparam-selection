"""Filesystem and dataframe helpers for locating notebook result artifacts.

Contents
--------
- find_data_obj_files
- paths_to_df
- _is_path_scalar
- _flatten_pipeline_filters
- condense_df
- print_path_components"""

import os
from numbers import Number

import pandas as pd


PIPELINE_PARAM_GROUPS = {
    "RDEq_params",
    "add_noise_params",
    "binnTV_params",
    "binnGenerateIndicesArgs",
    "binn_construction_params",
    "BNdata_loss_params",
    "pde_loss_params",
    "binn_fit_params",
}

IGNORED_PIPELINE_GROUPS = {
    "RDEq_params_store",
    "additional_params",
    "RDEq_extra_params",
    "binn_fit_params_additionals",
    "runtime",
    "grid",
    "data",
    "binn",
}


def find_data_obj_files(start_dir, target_filename="data_obj.npy"):
    """
    Recursively finds all files named `data_obj.npy` in directories with structure X_Y.

    Parameters:
        start_dir (str): Root directory to start the search.
        target_filename (str): The name of the file to search for.

    Returns:
        List[str]: List of full paths to matching files.
    """
    matches = []
    for root, dirs, files in os.walk(start_dir):
        if target_filename in files:
            matches.append(os.path.join(root, target_filename))
    return matches


def paths_to_df(paths, base_dir=None, legacy=False):
    """
    Converts a list of paths into a pandas DataFrame based on X_Y directory names.

    Parameters:
        paths (List[str]): List of paths to data_obj.npy files.
        base_dir (str | None): Directory to treat as the root when extracting
            path keys. When omitted, the deepest shared root across ``paths`` is
            used unless ``legacy=True``.
        legacy (bool): Revert to the old implementation style, which parses the
            full path directly instead of using a relative path rooted at
            ``base_dir`` or the shared common path.

    Returns:
        pd.DataFrame: DataFrame where each row represents a file and columns are variables X.
    """
    if not paths:
        return pd.DataFrame()

    if legacy:
        root_dir = None
    elif base_dir is not None:
        root_dir = os.path.abspath(base_dir)
    else:
        root_dir = os.path.commonpath(paths)

    records = []
    for path in paths:
        row = {"full_path": path}
        path_dict = {}
        parse_path = path if legacy else os.path.relpath(path, root_dir)
        parts = os.path.normpath(parse_path).split(os.sep)
        for part in parts:
            if "_" in part:
                try:
                    x, y = part.rsplit("_", 1)
                    row[x] = y
                    path_dict[x] = y
                except ValueError:
                    continue  # Skip if the format doesn't match
        row["path_dict"] = path_dict
        records.append(row)
    return pd.DataFrame(records)


def _is_path_scalar(value):
    return value is None or isinstance(value, (str, bool, Number, tuple))


def _flatten_pipeline_filters(filters):
    """
    Convert nested pipeline parameter dictionaries to the same flat dictionary
    used by ``dictToPath(all_func_info)`` in ``binn/python/pipeline``.

    Accepted inputs include:
    - the final flat ``all_func_info`` dictionary,
    - a single nested block such as ``data_obj_params`` or ``model_params``,
    - a wrapper containing ``data_obj_params``, ``TV_params``, ``model_params``,
      and/or ``fit_params``.

    Non-path runtime/storage values such as arrays, tensors, inputs, and
    ``RDEq_params_store`` are intentionally ignored because the pipeline does
    not include them in the saved model directory path.
    """
    flat = {}

    def visit(obj, parent_key=None):
        if not isinstance(obj, dict):
            return

        for key, value in obj.items():
            if key in IGNORED_PIPELINE_GROUPS:
                continue

            if isinstance(value, dict):
                if key in PIPELINE_PARAM_GROUPS:
                    for inner_key, inner_value in value.items():
                        if isinstance(inner_value, dict):
                            visit(inner_value, key)
                        elif _is_path_scalar(inner_value):
                            flat[inner_key] = inner_value
                    continue

                visit(value, key)
                continue

            if _is_path_scalar(value):
                flat[key] = value

    visit(filters)
    constraint_samples = flat.pop("constraintSamples", None)
    if constraint_samples is not None and int(constraint_samples) != 100:
        flat["np"] = int(constraint_samples)
    return flat



def condense_df(df, filters):
    """
    Filters a path-derived DataFrame by matching path parameter values.

    ``filters`` may be either the flat dictionary used in the saved path or one
    of the nested dictionaries produced by the BINN pipeline. Nested pipeline
    dictionaries are flattened to the same logical dictionary used by
    ``BN_model_finder`` before it calls ``dictToPath(all_func_info)``.

    Parameters:
        df (pd.DataFrame): Original DataFrame.
        filters (dict): Flat or nested dictionary of desired parameter values.

    Returns:
        pd.DataFrame: Filtered/condensed DataFrame.
    """
    filters = _flatten_pipeline_filters(filters)
    filtered_df = df.copy()

    def _matches_row(row):
        row_dict = row.get("path_dict", {})
        for key, value in filters.items():
            if key in row_dict:
                if str(row_dict[key]) != str(value):
                    return False
                continue
            if key in row.index:
                if str(row[key]) != str(value):
                    return False
                continue
            return False
        return True

    if "path_dict" in filtered_df.columns:
        mask = filtered_df.apply(_matches_row, axis=1)
        return filtered_df.loc[mask].reset_index(drop=True)

    for key, value in filters.items():
        if key not in filtered_df.columns:
            raise KeyError(
                f"condense_df could not find a column for {key!r}. Available columns: {list(filtered_df.columns)}"
            )
        filtered_df = filtered_df[filtered_df[key].astype(str) == str(value)]
    return filtered_df.reset_index(drop=True)

def print_path_components(path):
    components = os.path.normpath(path).split(os.sep)
    for part in components:
        print(part)
