"""Resolve PDE-loss callables for BINN training runs.

Contents
--------
- BN_model_pde_loss_func_info: return PDE-loss metadata for bookkeeping.
- BN_model_pde_loss_func: choose the configured PDE-loss callable.
"""

from Training.binn.python.Modules.Models.BuildBINNs import (
    pde_loss_with_bc,
    pde_loss_without_bc,
)


def BN_model_pde_loss_func_info(model_params):
    binn_model_params = model_params["binn_model_params"]
    return binn_model_params["pde_loss_params"]


def BN_model_pde_loss_func(model_params):
    binn_model_params = model_params["binn_model_params"]
    pde_loss_params = binn_model_params["pde_loss_params"]
    BCbool = pde_loss_params["BCbool"]

    pde_loss_func = pde_loss_with_bc if BCbool else pde_loss_without_bc

    func_info = BN_model_pde_loss_func_info(model_params)
    return func_info, pde_loss_func
