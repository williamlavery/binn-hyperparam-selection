"""Notebook helpers for locating, filtering, and loading trained BINN models.

Contents
--------
- build_model_catalog: build a dataframe catalog of saved model files.
- infer_path_keys: infer the path-key suffix used in saved model directories.
- active_constraint_settings: expand a constraint tuple into active settings.
- build_model_path: construct a model path from fixed and dynamic filters.
- resolve_model_path_from_catalog: resolve the exact saved model path from a catalog.
- build_requested_models: enumerate requested model combinations on a grid.
- summarize_requested_models: report how many requested models exist on disk.
- load_requested_models: load requested BINN models into nested dictionaries.
- filter_model_catalog: filter a model catalog and fail clearly if empty.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import torch

from Training.JN.paper_helpers.file_finder import condense_df, find_data_obj_files, paths_to_df
from Training.JN.paper_helpers.paths import dictToPath

DEFAULT_OPTIONAL_MODEL_VALUES = {
    "binnActivation": "silu",
    "binnSurfaceHiddenLayers": 3,
    "binnDGHiddenLayers": 3,
}


def build_model_catalog(root: str | Path, target_filename: str = "binnModel0.pth"):
    root_path = Path(root)
    return paths_to_df(
        find_data_obj_files(start_dir=str(root_path), target_filename=target_filename),
        base_dir=str(root_path),
    )


def infer_path_keys(model_df, start_key: str = "dataX1num") -> list[str]:
    if model_df.empty:
        raise ValueError("Cannot infer path keys from an empty model catalog.")

    path_dict = model_df.iloc[0].get("path_dict", {})
    path_keys = list(path_dict.keys())
    if start_key not in path_keys:
        raise KeyError(
            f"Could not find start key {start_key!r} in the inferred path keys: {path_keys}"
        )
    return path_keys[path_keys.index(start_key) :]


def active_constraint_settings(
    constraint_tuple,
    constraint_weights: dict[str, Any],
    constraint_bounds: dict[str, Any],
) -> dict[str, Any]:
    settings = {"constraintTuple": constraint_tuple}
    for enabled, key in zip(
        constraint_tuple,
        ("D_bound_weight", "D_mono_weight", "G_bound_weight", "G_mono_weight"),
    ):
        if enabled:
            settings[key] = constraint_weights[key]

    for enabled, key in zip(
        constraint_tuple,
        ("D_bound", None, "G_bound", None),
    ):
        if enabled and key is not None:
            settings[key] = constraint_bounds[key]

    return settings


def build_model_path(
    root: str | Path,
    fixed_filters: dict[str, Any],
    dynamic_filters: dict[str, Any],
    *,
    path_keys: list[str] | None = None,
    model_df=None,
    start_key: str = "dataX1num",
    model_label: int = 0,
) -> Path:
    model_info = fixed_filters | dynamic_filters | {"binnModelLabel": model_label}

    if path_keys is None:
        if model_df is None:
            raise ValueError("Either path_keys or model_df must be provided to build_model_path.")
        inferred_path_keys = infer_path_keys(model_df, start_key=start_key)
        path_keys = [key for key in inferred_path_keys if key in model_info]

    missing_path_keys = [key for key in path_keys if key not in model_info]
    if missing_path_keys:
        raise KeyError(
            "The model path dictionary is missing explicit values for: "
            f"{missing_path_keys}"
        )

    path_dict = {key: model_info[key] for key in path_keys}
    return Path(root) / dictToPath(path_dict) / f"binnModel{model_label}.pth"


def _normalized_catalog_value(key: str, value):
    if str(value) == "nan":
        return DEFAULT_OPTIONAL_MODEL_VALUES.get(key, value)
    return value


def _values_match(key: str, left, right) -> bool:
    return str(_normalized_catalog_value(key, left)) == str(
        _normalized_catalog_value(key, right)
    )


def _row_matches_filters_strict(row, filters: dict[str, Any]) -> bool:
    row_path_dict = row.get("path_dict", {})
    for key, value in filters.items():
        if key in row_path_dict:
            if not _values_match(key, row_path_dict[key], value):
                return False
            continue
        if key in row.index:
            if not _values_match(key, row[key], value):
                return False
            continue
        return False
    return True


def resolve_unique_catalog_row(model_df, filters: dict[str, Any]):
    subset = condense_df(model_df, filters)
    if subset.empty:
        raise FileNotFoundError(
            "No BINN model matched the requested filters: "
            f"{filters}"
        )

    if len(subset) > 1:
        strict_mask = subset.apply(
            lambda row: _row_matches_filters_strict(row, filters),
            axis=1,
        )
        strict_subset = subset.loc[strict_mask].reset_index(drop=True)
        if not strict_subset.empty:
            subset = strict_subset

    effective_filters = dict(filters)
    for key, default in DEFAULT_OPTIONAL_MODEL_VALUES.items():
        effective_filters.setdefault(key, default)

    for key in DEFAULT_OPTIONAL_MODEL_VALUES:
        if len(subset) <= 1 or key not in subset.columns:
            continue
        narrowed = subset[
            subset[key].apply(
                lambda value: _values_match(key, value, effective_filters[key])
            )
        ].reset_index(drop=True)
        if not narrowed.empty:
            subset = narrowed

    sort_columns = [
        column
        for column in ("binnTVsplitSeed", "full_path")
        if column in subset.columns
    ]
    if sort_columns:
        subset = subset.sort_values(sort_columns).reset_index(drop=True)

    if len(subset) != 1:
        preview_columns = [
            column
            for column in (
                "binnES",
                "binnDsize",
                "binnUsize",
                "binnTVsplitSeed",
                "binnActivation",
                "binnSurfaceHiddenLayers",
                "binnDGHiddenLayers",
                "full_path",
            )
            if column in subset.columns
        ]
        raise ValueError(
            "Expected exactly one BINN match, but found "
            f"{len(subset)}. Requested filters: {filters}. Matching rows:\n"
            f"{subset[preview_columns]}"
        )

    return subset.iloc[0]


def resolve_model_path_from_catalog(
    model_df,
    fixed_filters: dict[str, Any],
    dynamic_filters: dict[str, Any],
    *,
    model_label: int = 0,
) -> Path:
    model_filters = fixed_filters | dynamic_filters | {"binnModelLabel": model_label}
    resolved_row = resolve_unique_catalog_row(model_df, model_filters)
    return Path(resolved_row["full_path"])


def build_requested_models(
    root: str | Path,
    path_keys: list[str],
    fixed_filters: dict[str, Any],
    grid: dict[str, list[Any]],
    *,
    model_label: int = 0,
    extra_filters: dict[str, Any] | None = None,
):
    grid_keys = list(grid.keys())
    requested_models = []
    for values in product(*(grid[key] for key in grid_keys)):
        dynamic_filters = dict(zip(grid_keys, values))
        if extra_filters:
            dynamic_filters |= extra_filters
        requested_models.append(
            {
                **dynamic_filters,
                "path": build_model_path(
                    root,
                    path_keys,
                    fixed_filters,
                    dynamic_filters,
                    model_label=model_label,
                ),
            }
        )
    return requested_models


def summarize_requested_models(requested_models, label_keys: list[str]):
    existing_paths = [entry for entry in requested_models if entry["path"].exists()]
    missing_paths = [entry for entry in requested_models if not entry["path"].exists()]

    print(f"Requested model combinations: {len(requested_models)}")
    print(f"Already trained model files found: {len(existing_paths)}")
    print(f"Missing model files: {len(missing_paths)}")
    if missing_paths:
        first_missing = missing_paths[0]
        print({key: first_missing[key] for key in label_keys} | {"path": first_missing["path"]})

    return {
        "requested_count": len(requested_models),
        "trained_count": len(existing_paths),
        "missing_count": len(missing_paths),
        "missing_paths": [entry["path"] for entry in missing_paths],
    }


def load_requested_models(
    requested_models,
    *,
    device: str = "cpu",
    group_keys: tuple[str, ...],
    missing: str = "error",
):
    models = {}
    missing_model_paths = []

    for request in requested_models:
        file_path = request["path"]
        if not file_path.exists():
            missing_model_paths.append(file_path)
            continue

        binn_loaded = torch.load(
            file_path,
            map_location=device,
            weights_only=False,
        )
        if hasattr(binn_loaded, "load_best_val"):
            binn_loaded.load_best_val(device=device)

        current_level = models
        for key in group_keys[:-1]:
            current_level = current_level.setdefault(request[key], {})
        current_level[request[group_keys[-1]]] = binn_loaded

    if missing_model_paths and missing == "error":
        raise FileNotFoundError(
            "Missing one or more BINN models. First missing path: "
            f"{missing_model_paths[0]}. Missing {len(missing_model_paths)} total model(s)."
        )

    return models


def filter_model_catalog(model_df, filters: dict[str, Any], *, empty_message: str):
    filtered_catalog = condense_df(model_df, filters)
    if filtered_catalog.empty:
        raise FileNotFoundError(empty_message)
    return filtered_catalog
