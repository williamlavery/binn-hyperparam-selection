"""Construct BINN models and resolve their callable loss components.

Contents
--------
- BN_model_construction_func_info: return model-construction metadata.
- BN_params_binn_builder: assemble constructor parameters from data settings.
- BN_model_construction: build a BINN instance for a given dataset and config.
"""

from functools import partial

from Training.binn.python.Modules.Models.BuildBINNs import BINN
from Training.binn.python.Modules.Utils.parse import unravel_one_level
from Training.binn.python.pipeline.components.binn__modelDataLoss import (
    BN_model_data_loss_func,
)
from Training.binn.python.pipeline.components.binn__modelPDEloss import (
    BN_model_pde_loss_func,
)
from Training.data.python.pipeline.config.store import (
    diffusion_func1,
    diffusion_func1_du,
    diffusion_func2,
    diffusion_func2_du,
    diffusion_func3,
    diffusion_func3_du,
    diffusion_func4,
    diffusion_func4_du,
    growth_func1,
    growth_func2,
    growth_func3,
    growth_func4,
)


def BN_model_construction_func_info(model_params):
    binn_model_params = model_params["binn_model_params"]
    func_info = unravel_one_level(binn_model_params)
    perfect_pde = bool(func_info.get("perfectPDE", False))
    if not perfect_pde:
        func_info.pop("perfectPDE", None)
    if str(func_info.get("binnActivation", "silu")).lower() == "silu":
        func_info.pop("binnActivation", None)
    surface_hidden_layers = int(func_info.get("binnSurfaceHiddenLayers", 3))
    dg_hidden_layers = int(func_info.get("binnDGHiddenLayers", 3))
    if surface_hidden_layers == 3 and dg_hidden_layers == 3:
        func_info.pop("binnSurfaceHiddenLayers", None)
        func_info.pop("binnDGHiddenLayers", None)
    constraint_samples = int(func_info.pop("constraintSamples", 100))
    if constraint_samples != 100:
        func_info["np"] = constraint_samples
    func_info.pop("GLSpow", None)
    if perfect_pde:
        for key in (
            "binnDsize",
            "binnGsize",
            "constraintTuple",
            "D_bound_weight",
            "D_mono_weight",
            "G_bound_weight",
            "G_mono_weight",
            "D_bound",
            "G_bound",
            "allConstraints",
        ):
            func_info.pop(key, None)
    return func_info


def BN_params_binn_builder(data_obj, data_obj_params):
    diffLabel = data_obj_params["RDEq_params"]["dataDiffLabel"]
    growLabel = data_obj_params["RDEq_params"]["dataGrowLabel"]

    if diffLabel == "const":
        diffusion_func = partial(diffusion_func1, theta=data_obj.theta_D)
        diffusion_derivative_func = partial(diffusion_func1_du, theta=data_obj.theta_D)
    elif diffLabel == "linear":
        diffusion_func = partial(diffusion_func2, theta=data_obj.theta_D)
        diffusion_derivative_func = partial(diffusion_func2_du, theta=data_obj.theta_D)
    elif diffLabel == "quadratic":
        diffusion_func = partial(diffusion_func3, theta=data_obj.theta_D)
        diffusion_derivative_func = partial(diffusion_func3_du, theta=data_obj.theta_D)
    elif diffLabel == "exp":
        diffusion_func = partial(diffusion_func4, theta=data_obj.theta_D)
        diffusion_derivative_func = partial(diffusion_func4_du, theta=data_obj.theta_D)
    else:
        raise ValueError(f"Unknown diffusion label: {diffLabel!r}")

    if growLabel == "const":
        growth_func = partial(growth_func1, theta=data_obj.theta_G)
    elif growLabel == "linear":
        growth_func = partial(growth_func2, theta=data_obj.theta_G)
    elif growLabel == "quadratic":
        growth_func = partial(growth_func3, theta=data_obj.theta_G)
    elif growLabel == "exp":
        growth_func = partial(growth_func4, theta=data_obj.theta_G)
    elif growLabel == "zero":
        growth_func = partial(growth_func1, theta=data_obj.theta_G)
    else:
        raise ValueError(f"Unknown growth label: {growLabel!r}")

    RDEq_extra_params = {
        "thetaD": data_obj.theta_D,
        "thetaG": data_obj.theta_G,
        "diffusionTrueFunc": diffusion_func,
        "diffusionTrueDerivFunc": diffusion_derivative_func,
        "growthTrueFunc": growth_func,
        "max_u_clean": data_obj.u_clean.max(),
        "min_u_clean": data_obj.u_clean.min(),
    }
    data_obj_params["RDEq_params_store"]["u_clean"] = data_obj.u_clean
    data_obj_params["RDEq_extra_params"] = RDEq_extra_params

    return data_obj_params


def BN_model_construction(data_obj_orig, data_obj_params, model_params):
    data_obj_params = BN_params_binn_builder(
        data_obj=data_obj_orig,
        data_obj_params=data_obj_params,
    )

    _, data_loss_func = BN_model_data_loss_func(model_params=model_params)
    _, pde_loss_func = BN_model_pde_loss_func(model_params=model_params)

    NN_binn = BINN(
        data_obj_params=data_obj_params,
        model_params=model_params,
        data_loss_func=data_loss_func,
        pde_loss_func=pde_loss_func,
    )

    func_info = BN_model_construction_func_info(model_params=model_params)
    return func_info, NN_binn
