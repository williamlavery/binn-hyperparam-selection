"""Notebook helpers for locating data objects and true generating functions.

Contents
--------
- build_catalog: build a dataframe catalog of saved data artifacts.
- load_original_data: load the original experimental data object.
- load_data_object: load a generated data object matching a filter set.
- build_true_diffusion_callable: build the true diffusion function from labels.
- build_true_growth_callable: build the true growth function from labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from Training.JN.paper_helpers.file_finder import condense_df, find_data_obj_files, paths_to_df
from Training.JN.paper_helpers.paths import dictToPath
from Training.data.python.pipeline.components.data__initialise import (
    DIFFUSION_PARAMETERS,
    GROWTH_PARAMETERS,
)
from Training.data.python.pipeline.config.store import (
    diffusion_func1,
    diffusion_func2,
    diffusion_func3,
    diffusion_func4,
    growth_func1,
    growth_func2,
    growth_func3,
    growth_func4,
)


def build_catalog(root: str | Path, target_filename: str) -> Any:
    root_path = Path(root)
    return paths_to_df(
        find_data_obj_files(start_dir=str(root_path), target_filename=target_filename),
        base_dir=str(root_path),
    )


def load_original_data(
    data_root: str | Path,
    relative_path: str = "Lagergren_et_al_2020/originalDataObj.npy",
):
    original_data_path = Path(data_root) / relative_path
    return np.load(original_data_path, allow_pickle=True).item(0)


def load_data_object(filters: dict[str, Any], data_root: str | Path, data_df=None):
    root_path = Path(data_root)
    catalog = data_df if data_df is not None else build_catalog(root_path, "data_obj.npy")
    filtered_catalog = condense_df(catalog, filters)
    if filtered_catalog.empty:
        expected_path = root_path / dictToPath(filters) / "data_obj.npy"
        raise FileNotFoundError(
            "No data object was found on disk for the requested filters. "
            f"Expected: {expected_path}"
        )

    data_file_path = Path(filtered_catalog["full_path"].iloc[0])
    data_obj = np.load(data_file_path, allow_pickle=True).item()
    return data_obj, filtered_catalog


def build_true_diffusion_callable(diffusion_label: str, theta_D=None):
    theta, func_name = DIFFUSION_PARAMETERS[diffusion_label]
    theta = np.asarray(theta_D if theta_D is not None else theta, dtype=float)
    diffusion_template = {
        "diffusion_func1": diffusion_func1,
        "diffusion_func2": diffusion_func2,
        "diffusion_func3": diffusion_func3,
        "diffusion_func4": diffusion_func4,
    }[func_name]

    def diffusion_func(u):
        return diffusion_template(np.asarray(u, dtype=float), theta)

    return diffusion_func


def build_true_growth_callable(growth_label: str, theta_G=None):
    theta, func_name = GROWTH_PARAMETERS[growth_label]
    theta = np.asarray(theta_G if theta_G is not None else theta, dtype=float)
    growth_template = {
        "growth_func1": growth_func1,
        "growth_func2": growth_func2,
        "growth_func3": growth_func3,
        "growth_func4": growth_func4,
    }[func_name]

    def growth_func(u):
        return growth_template(np.asarray(u, dtype=float), theta)

    return growth_func
