"""Grid orchestration helpers for BINN training sweeps.

Contents
--------
- _as_sweep_values: normalize scalar or list sweep inputs into a non-empty list.
- constraint_weight_sweep: iterate over active constraint-weight combinations for one constraint tuple.
- constraint_weight_sweep_count: count how many weight combinations a constraint tuple contributes.
- constraint_tuple_to_params: convert constraint flags into model parameters and optional bounds.
- loss_weight_params: encode non-default surface and PDE loss weights in the run params.
- build_base_params: assemble shared data, TV, model, and fit parameter dictionaries from config.
- iter_run_grid: yield every configured BINN run across data, architecture, and loss sweeps.
- print_run_summary: print a compact human-readable summary of one queued BINN run.
- run_binn_pipeline: execute the full configured BINN sweep from the generated run grid.
"""

from __future__ import annotations

from itertools import product
from math import prod


from Training.binn.python.pipeline.components.binn__simulate import BN_sim_smart
from Training.data.python.Modules.dataClass import generate_inputs
from Training.pipeline_runtime import print_nested


CONSTRAINT_FLAG_NAMES = ("D_bound", "D_mono", "G_bound", "G_mono")
CONSTRAINT_WEIGHT_NAMES = {
    "D_bound": "D_bound_weight",
    "D_mono": "D_mono_weight",
    "G_bound": "G_bound_weight",
    "G_mono": "G_mono_weight",
}
CONSTRAINT_BOUND_NAMES = {
    "D_bound": "D_bound",
    "G_bound": "G_bound",
}
DEFAULT_CONSTRAINT_BOUNDS = {
    "D_bound": (0, 0.1),
    "G_bound": (-0.02 / (1 / 24), 0.1 / (1 / 24)),
}
CONSTRAINT_PARAM_NAMES = (
    "constraintTuple",
    *CONSTRAINT_WEIGHT_NAMES.values(),
    *CONSTRAINT_BOUND_NAMES.values(),
)
LOSS_WEIGHT_PARAM_NAMES = ("surface_weight", "pde_weight")


def _as_sweep_values(value):
    if isinstance(value, list):
        if not value:
            raise ValueError("constraint weight sweep lists cannot be empty")
        return value
    return [value]


def constraint_weight_sweep(all_constraints, constraint_tuple, constraint_weights):
    if not all_constraints:
        yield constraint_weights
        return

    enabled_weight_names = [
        CONSTRAINT_WEIGHT_NAMES[flag_name]
        for enabled, flag_name in zip(constraint_tuple, CONSTRAINT_FLAG_NAMES)
        if enabled
    ]

    if not enabled_weight_names:
        yield constraint_weights
        return

    value_lists = [
        _as_sweep_values(constraint_weights[weight_name])
        for weight_name in enabled_weight_names
    ]
    for selected_values in product(*value_lists):
        selected_weights = dict(constraint_weights)
        for weight_name, selected_value in zip(enabled_weight_names, selected_values):
            selected_weights[weight_name] = selected_value
        yield selected_weights


def constraint_weight_sweep_count(all_constraints, constraint_tuple, constraint_weights):
    if not all_constraints:
        return 1

    total = 1
    for enabled, flag_name in zip(constraint_tuple, CONSTRAINT_FLAG_NAMES):
        if enabled:
            total *= len(_as_sweep_values(constraint_weights[CONSTRAINT_WEIGHT_NAMES[flag_name]]))
    return total


def constraint_tuple_to_params(
    all_constraints,
    constraint_tuple,
    constraint_weights,
    constraint_bounds,
):
    if not all_constraints:
        return {}

    constraint_tuple = tuple(constraint_tuple)
    if len(constraint_tuple) != len(CONSTRAINT_FLAG_NAMES):
        raise ValueError(
            "constraint tuple must be (D_bound, D_mono, G_bound, G_mono)"
        )
    params = {"constraintTuple": constraint_tuple}
    for enabled, flag_name in zip(constraint_tuple, CONSTRAINT_FLAG_NAMES):
        if enabled:
            weight_name = CONSTRAINT_WEIGHT_NAMES[flag_name]
            params[weight_name] = constraint_weights[weight_name]
            bound_name = CONSTRAINT_BOUND_NAMES.get(flag_name)
            if bound_name is not None:
                params[bound_name] = tuple(
                    constraint_bounds.get(
                        bound_name,
                        DEFAULT_CONSTRAINT_BOUNDS[bound_name],
                    )
                )
    return params


def loss_weight_params(surface_weight, pde_weight):
    """Only encode loss weights in params/path when one deviates from 1."""
    if surface_weight == 1e0 and pde_weight == 1e0:
        return {}
    return {"surface_weight": surface_weight, "pde_weight": pde_weight}


def build_base_params(config: dict) -> tuple[dict, dict, dict, dict]:
    project_root = config["project_root"]
    grid = config["grid"]
    binn = config["binn"]
    runtime = config["runtime"]
    t = grid["ts"][0]

    x2 = grid.get("x2")
    x2_num = 1 if x2 is None else len(x2)

    data_obj_params = {
        "RDEq_params_store": {
            "dim": grid.get("dim", 1 if x2 is None else 2),
            "x1": grid["x1"],
            "x2": x2,
            "t": t,
            "inputs": generate_inputs(grid["x1"], x2, t, dim=grid.get("dim", 1)),
            "K": grid["K"],
        },
        "RDEq_params": {
            "dataX1num": len(grid["x1"]),
            "dataX2num": x2_num,
            "dataTnum": len(t),
            "dataK": grid["K"],
        },
        "additional_params": {
            "inital_path": str(project_root / "Training" / "data" / "dataObj"),
            "binn_path": str(project_root / "Training" / "binn"),
            "plot_bool": runtime["plot_bool"],
            "overwrite_bool": runtime["overwrite_bool"],
        },
        "add_noise_params": {},
    }

    TV_params = {
        "binnTV_params": {
            "binnVF": binn["binnVFs"][0],
            "binnGenerateIndicesLabel": binn["binnGenerateIndicesLabel"],
            "binnGenerateIndicesArgs": {"binnTVsplitSeed": binn["binnTVsplitSeeds"][0]},
        },
    }

    model_params = {
        "binn_model_params": {
            "binn_construction_params": {
                "binnUsize": binn["binnUsizes"][0],
                "binnDsize": binn["binnDsizes"][0],
                "binnGsize": binn["binnGsizes"][0],
                "DoneParamBool": binn["DoneParamBool"],
                "perfectPDE": binn.get("perfectPDE", False),
                "binnActivation": binn.get("binnActivation", "silu"),
                "binnSurfaceHiddenLayers": binn.get("binnSurfaceHiddenLayers", [3])[0],
                "binnDGHiddenLayers": binn.get("binnDGHiddenLayers", [3])[0],
                "binnDevice": binn["binnDevice"],
                "constraintSamples": binn["constraintSamples"][0],
                "allConstraints": binn["allConstraints"][0],
                **constraint_tuple_to_params(
                    binn["allConstraints"][0],
                    binn["constraintTuples"][0],
                    next(
                        constraint_weight_sweep(
                            binn["allConstraints"][0],
                            binn["constraintTuples"][0],
                            binn["constraintWeights"],
                        )
                    ),
                    binn.get("constraintBounds", DEFAULT_CONSTRAINT_BOUNDS),
                ),
                **loss_weight_params(
                    binn["surfaceWeights"][0],
                    binn["pdeWeights"][0],
                ),
            },
            "BNdata_loss_params": {
                "BNdataLossFuncLabel": binn["BNdataLossFuncLabels"][0],
            },
            "pde_loss_params": {
                "BCbool": binn["BCbool"],
                "numPDEsamples": binn["numPDEsamples"][0],
            },
        },
    }

    fit_params = {
        "binn_fit_params": {
            "binnLR": binn["binnLR"],
            "binnBatchSize": len(grid["x1"]) * (x2_num),
            "binnRelUpdateThresh": binn["binnRelUpdateThresh"],
            "binnRelSaveThresh": binn["binnRelSaveThresh"],
            "binnES": binn["binnESs"][0],
            "binnModelLabel": 0,
        },
        "binn_fit_params_additionals": {
            "binnEpochs": binn["binnEpochs"],
            "binnES_check": binn["binnES_check"],
            "printFreq": binn["printFreq"],
            "storeConstraintLosses": binn.get("storeConstraintLosses", True),
            "storeAdamDiagnostics": binn.get("storeAdamDiagnostics", False),
            "storeDataLossDiagnostics": binn.get("storeDataLossDiagnostics", False),
            "storeFixedGridPDEDiagnostics": binn.get(
                "storeFixedGridPDEDiagnostics", False
            ),
            "fixedGridPDEFrequency": binn.get("fixedGridPDEFrequency", 10),
            "fixedGridPDEShape": binn.get("fixedGridPDEShape", None),
        },
    }
    return data_obj_params, TV_params, model_params, fit_params


def iter_run_grid(config: dict):
    data = config["data"]
    binn = config["binn"]
    grid = config["grid"]
    perfect_pde = bool(binn.get("perfectPDE", False))
    binn_dsizes = binn["binnDsizes"] if not perfect_pde else [binn["binnDsizes"][0]]
    binn_gsizes = binn["binnGsizes"] if not perfect_pde else [binn["binnGsizes"][0]]
    base_param_lists = [
        data["gammas"],
        data["noisePercents"],
        data["noiseSeeds"],
        data["ICLabels"],
        data["diffLabels"],
        data["growLabels"],
        binn["binnUsizes"],
        binn_dsizes,
        binn_gsizes,
        binn.get("binnSurfaceHiddenLayers", [3]),
        binn.get("binnDGHiddenLayers", [3]),
        binn["binnESs"],
        binn["binnTVsplitSeeds"],
        binn["binnVFs"],
        binn["surfaceWeights"],
        binn["pdeWeights"],
        binn["binnModelLabels"],
        binn["numPDEsamples"],
        binn["constraintSamples"],
        binn["BNdataLossFuncLabels"],
    ]
    base_total = prod(len(values) for values in base_param_lists)
    constraint_total = 0
    if perfect_pde:
        constraint_total = 1
    else:
        for all_constraint in binn["allConstraints"]:
            constraint_tuples = (
                binn["constraintTuples"]
                if all_constraint
                else [(False, False, False, False)]
            )
            for constraint_tuple in constraint_tuples:
                constraint_total += constraint_weight_sweep_count(
                    all_constraint,
                    constraint_tuple,
                    binn["constraintWeights"],
                )
    total = base_total * constraint_total * len(grid["ts"])

    def run_grid():
        for base_params in product(*base_param_lists):
            if perfect_pde:
                constraint_iter = [
                    (False, (False, False, False, False), binn["constraintWeights"])
                ]
            else:
                constraint_iter = []
                for all_constraint in binn["allConstraints"]:
                    constraint_tuples = (
                        binn["constraintTuples"]
                        if all_constraint
                        else [(False, False, False, False)]
                    )
                    for constraint_tuple in constraint_tuples:
                        for selected_constraint_weights in constraint_weight_sweep(
                            all_constraint,
                            constraint_tuple,
                            binn["constraintWeights"],
                        ):
                            constraint_iter.append(
                                (all_constraint, constraint_tuple, selected_constraint_weights)
                            )

            for all_constraint, constraint_tuple, selected_constraint_weights in constraint_iter:
                for t in grid["ts"]:
                    yield (
                        *base_params,
                        all_constraint,
                        constraint_tuple,
                        selected_constraint_weights,
                        t,
                    )

    return run_grid(), total


def print_run_summary(run_number, total, data_obj_params, TV_params, model_params, fit_params):
    grid = data_obj_params["RDEq_params_store"]
    print("=" * 70)
    print(f"RUN {run_number}/{total}: BINN settings")
    print("=" * 70)
    print(f"  dataX1num                  = {len(grid['x1'])}")
    if grid.get("x2") is not None:
        print(f"  dataX2num                  = {len(grid['x2'])}")
    print(f"  dataTnum                   = {len(grid['t'])}")
    print("-" * 70)
    print("data_obj_params:")
    print_nested(data_obj_params, omit_keys={"x1", "x2", "t", "inputs", "u_clean"})
    print("-" * 70)
    print("TV_params:")
    print_nested(TV_params)
    print("-" * 70)
    print("model_params:")
    print_nested(model_params)
    print("-" * 70)
    print("fit_params:")
    print_nested(fit_params)
    print("=" * 70)


def run_binn_pipeline(config: dict) -> None:
    binn = config["binn"]
    data_obj_params, TV_params, model_params, fit_params = build_base_params(config)
    param_grid, total = iter_run_grid(config)

    for run_number, params in enumerate(param_grid, start=1):
        (
            data_gamma,
            data_noise_percent,
            data_noise_seed,
            data_ic_label,
            data_diff_label,
            data_grow_label,
            binn_usize,
            binn_dsize,
            binn_gsize,
            binn_surface_hidden_layers,
            binn_dg_hidden_layers,
            binn_es,
            binn_tv_split_seed,
            binn_vf,
            surface_weight,
            pde_weight,
            binn_model_label,
            num_pde_sample,
            constraint_sample,
            bn_data_loss_label,
            all_constraint,
            constraint_tuple,
            selected_constraint_weights,
            t,
        ) = params

        data_obj_params["RDEq_params_store"]["t"] = t
        data_obj_params["RDEq_params_store"]["inputs"] = generate_inputs(
            data_obj_params["RDEq_params_store"]["x1"],
            data_obj_params["RDEq_params_store"].get("x2"),
            t,
            dim=data_obj_params["RDEq_params_store"].get("dim", 1),
        )
        data_obj_params["RDEq_params"]["dataTnum"] = len(t)
        binn_construction_params = model_params["binn_model_params"][
            "binn_construction_params"
        ]
        for key in CONSTRAINT_PARAM_NAMES:
            binn_construction_params.pop(key, None)
        for key in LOSS_WEIGHT_PARAM_NAMES:
            binn_construction_params.pop(key, None)

        model_params["binn_model_params"]["binn_construction_params"].update(
            {
                "binnUsize": binn_usize,
                "binnDsize": binn_dsize,
                "binnGsize": binn_gsize,
                "perfectPDE": binn.get("perfectPDE", False),
                "binnActivation": binn.get("binnActivation", "silu"),
                "binnSurfaceHiddenLayers": binn_surface_hidden_layers,
                "binnDGHiddenLayers": binn_dg_hidden_layers,
                "constraintSamples": constraint_sample,
                "allConstraints": all_constraint,
                **constraint_tuple_to_params(
                    all_constraint,
                    constraint_tuple,
                    selected_constraint_weights,
                    binn.get("constraintBounds", DEFAULT_CONSTRAINT_BOUNDS),
                ),
                **loss_weight_params(surface_weight, pde_weight),
            }
        )
        model_params["binn_model_params"]["pde_loss_params"][
            "numPDEsamples"
        ] = num_pde_sample
        BNdata_loss_params = model_params["binn_model_params"]["BNdata_loss_params"]
        BNdata_loss_params.pop("GLSpow", None)
        BNdata_loss_params["BNdataLossFuncLabel"] = bn_data_loss_label
        fit_params["binn_fit_params"].update(
            {
                "binnES": binn_es,
                "binnModelLabel": binn_model_label,
            }
        )
        data_obj_params["RDEq_params"].update(
            {
                "dataICLabel": data_ic_label,
                "dataDiffLabel": data_diff_label,
                "dataGrowLabel": data_grow_label,
            }
        )
        data_obj_params["add_noise_params"] = {
            "dataGamma": data_gamma,
            "dataNoisePercent": data_noise_percent,
            "dataNoiseSeed": data_noise_seed,
        }
        TV_params["binnTV_params"].update(
            {
                "binnVF": binn_vf,
                "binnGenerateIndicesArgs": {"binnTVsplitSeed": binn_tv_split_seed},
            }
        )

        print_run_summary(
            run_number,
            total,
            data_obj_params,
            TV_params,
            model_params,
            fit_params,
        )
        BN_sim_smart(data_obj_params, TV_params, model_params, fit_params)
