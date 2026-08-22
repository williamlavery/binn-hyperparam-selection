"""Initialise PDE ingredients and solver choices for data generation.

Contents
--------
- DATA_RDEq2_func_info: return the PDE-parameter subset used for bookkeeping.
- _initial_condition_from_label: resolve the configured 1D or 2D initial-condition generator.
- _reaction_functions: resolve the diffusion and growth callables together with their parameters.
- DATA_RDEq2: build the PDE specification, parameters, and solver outputs.
"""

import re

from Training.data.python.Modules.PDESolver_1D import PDE_RHS_1D, PDE_sim_1d
from Training.data.python.Modules.PDESolver_2D import PDE_RHS_2D, PDE_sim_2d
from Training.data.python.pipeline.config.store import (
    diffusion_func1,
    diffusion_func2,
    diffusion_func3,
    diffusion_func4,
    growth_func1,
    growth_func2,
    growth_func3,
    growth_func4,
    ic1,
    ic1_flipped,
    scratch,
)


DIFFUSION_PARAMETERS = {
    "const": ([0.02472], "diffusion_func1"),
    "linear": ([0.015, 0.06], "diffusion_func2"),
    "quadratic": ([0.01, 0.044], "diffusion_func3"),
    "exp": ([0.003, 0.095, 2.5], "diffusion_func4"),
}

GROWTH_PARAMETERS = {
    "zero": ([0], "growth_func1"),
    "const": ([1.3], "growth_func1"),
    "linear": ([2.4, -3], "growth_func2"),
    "quadratic": ([2.1, -0.29], "growth_func3"),
    "exp": ([0.7, -1.3, 4], "growth_func4"),
}


def DATA_RDEq2_func_info(data_obj_params):
    """Return the deterministic PDE parameters used in the path signature."""
    return data_obj_params["RDEq_params"]


def _initial_condition_from_label(label, dim):
    if label == "cos":
        return ic1

    if label == "cosFlipped":
        return ic1_flipped

    if label == "cosFlippedArtifical":
        return ic1

    if label == "scratch":
        return scratch

    match = re.fullmatch(r"cosFlat(\d*\.?\d+)", label)
    if match:
        amplitude = float(match.group(1))
        if dim == 1:
            return lambda x: ic1(x, amplitude=amplitude)
        return lambda x1, x2: ic1(x1, x2, amplitude=amplitude)

    raise ValueError(f"Unknown initial-condition label: {label!r}")


def _reaction_functions(diffusion_label, growth_label):
    try:
        theta_D, diffusion_func_name = DIFFUSION_PARAMETERS[diffusion_label]
    except KeyError as exc:
        raise ValueError(f"Unknown diffusion label: {diffusion_label!r}") from exc

    try:
        theta_G, growth_func_name = GROWTH_PARAMETERS[growth_label]
    except KeyError as exc:
        raise ValueError(f"Unknown growth label: {growth_label!r}") from exc

    diffusion_template = {
        "diffusion_func1": diffusion_func1,
        "diffusion_func2": diffusion_func2,
        "diffusion_func3": diffusion_func3,
        "diffusion_func4": diffusion_func4,
    }[diffusion_func_name]
    growth_template = {
        "growth_func1": growth_func1,
        "growth_func2": growth_func2,
        "growth_func3": growth_func3,
        "growth_func4": growth_func4,
    }[growth_func_name]

    def diffusion_func(u):
        return diffusion_template(u, theta_D)

    def growth_func(u):
        return growth_template(u, theta_G)

    return diffusion_func, growth_func, theta_D, theta_G


def DATA_RDEq2(data_obj_params):
    """Solve the reaction-diffusion equation for one configured parameter set."""
    print("===============================================================")
    print("Forward solving reaction-diffusion data")
    print("===============================================================")

    RDEq_params_store = data_obj_params["RDEq_params_store"]
    RDEq_params = data_obj_params["RDEq_params"]

    x1 = RDEq_params_store["x1"]
    x2 = RDEq_params_store.get("x2")
    t = RDEq_params_store["t"]
    dim = int(RDEq_params_store.get("dim", 1 if x2 is None else 2))

    initial_condition = _initial_condition_from_label(RDEq_params["dataICLabel"], dim)
    diffusion_func, growth_func, theta_D, theta_G = _reaction_functions(
        RDEq_params["dataDiffLabel"],
        RDEq_params["dataGrowLabel"],
    )

    if dim == 1:
        u_clean = PDE_sim_1d(
            PDE_RHS_1D,
            initial_condition,
            x1,
            t,
            diffusion_func,
            growth_func,
            clear=False,
        )
    else:
        u_clean = PDE_sim_2d(
            PDE_RHS_2D,
            initial_condition,
            x1,
            x2,
            t,
            diffusion_func,
            growth_func,
            clear=False,
        )

    return DATA_RDEq2_func_info(data_obj_params), u_clean, (theta_D, theta_G)
