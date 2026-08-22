"""Grid orchestration helpers for the synthetic data-generation pipeline.

Contents
--------
- build_base_data_params: assemble shared parameters from a runtime config.
- iter_data_grid: iterate over every configured synthetic-data combination.
- print_run_summary: print a compact summary for one pipeline run.
- run_data_pipeline: execute the full grid of configured data-generation runs.
"""

from __future__ import annotations

from itertools import product
from math import prod

from Training.data.python.pipeline.components.data__simulate import DATA_sim
from Training.pipeline_runtime import print_nested


def build_base_data_params(config: dict) -> dict:
    grid = config["grid"]
    runtime = config["runtime"]
    project_root = config["project_root"]

    x2 = grid.get("x2")
    x2_num = 1 if x2 is None else len(x2)

    return {
        "RDEq_params_store": {
            "dim": grid.get("dim", 1 if x2 is None else 2),
            "x1": grid["x1"],
            "x2": x2,
            "t": grid["t"],
            "inputs": grid["inputs"],
            "K": grid["K"],
        },
        "RDEq_params": {
            "dataX1num": len(grid["x1"]),
            "dataX2num": x2_num,
            "dataTnum": len(grid["t"]),
            "dataK": grid["K"],
        },
        "additional_params": {
            "inital_path": str(project_root / "Training" / "data" / "dataObj"),
            "binn_path": str(project_root / "Training" / "binn"),
            "denoise_path": str(project_root / "Training" / "dn"),
            "plot_bool": runtime["plot_bool"],
            "overwrite_bool": runtime["overwrite_bool"],
        },
        "add_noise_params": {},
    }


def iter_data_grid(config: dict):
    sweep = config["sweep"]
    param_lists = [
        sweep["dataGammas"],
        sweep["dataNoisePercents"],
        sweep["dataNoiseSeeds"],
        sweep["dataICLabels"],
        sweep["dataDiffLabels"],
        sweep["dataGrowLabels"],
    ]
    return product(*param_lists), prod(len(values) for values in param_lists)


def print_run_summary(run_number, total, data_obj_params, omit_keys=None):
    omit_keys = set(omit_keys or [])
    grid = data_obj_params["RDEq_params_store"]
    print("=" * 70)
    print(f"RUN {run_number}/{total}: data-generation settings")
    print("=" * 70)
    print(f"  dataX1num                  = {len(grid['x1'])}")
    if grid.get("x2") is not None:
        print(f"  dataX2num                  = {len(grid['x2'])}")
    print(f"  dataTnum                   = {len(grid['t'])}")
    print("-" * 70)
    print_nested(data_obj_params, omit_keys=omit_keys)
    print("-" * 70)


def run_data_pipeline(config: dict):
    data_obj_params = build_base_data_params(config)
    param_grid, total = iter_data_grid(config)

    for run_number, params in enumerate(param_grid, start=1):
        (
            data_gamma,
            data_noise_percent,
            data_noise_seed,
            data_ic_label,
            data_diff_label,
            data_grow_label,
        ) = params

        data_obj_params["add_noise_params"] = {
            "dataGamma": data_gamma,
            "dataNoisePercent": data_noise_percent,
            "dataNoiseSeed": data_noise_seed,
        }
        data_obj_params["RDEq_params"].update(
            {
                "dataICLabel": data_ic_label,
                "dataDiffLabel": data_diff_label,
                "dataGrowLabel": data_grow_label,
            }
        )

        print_run_summary(
            run_number,
            total,
            data_obj_params,
            omit_keys={"x1", "x2", "t", "inputs"},
        )
        DATA_sim(data_obj_params)
