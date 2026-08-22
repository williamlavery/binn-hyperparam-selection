"""Resolve data-loss callables for BINN training runs.

Contents
--------
- _parse_gls_power_label: extract the GLS power from labels such as `GLS0.5`.
- BN_model_data_loss_func_info: return the chosen data-loss settings for bookkeeping.
- BN_model_data_loss_func: resolve the configured data-loss callable for training.
"""

from Training.binn.python.Modules.Models.BuildBINNs import (
    data_loss_GLS,
    data_loss_GLSpow,
    data_loss_MSE,
    data_loss_MSEmodified10,
)


def _parse_gls_power_label(label):
    if label == "GLSpow":
        return 0.5
    if not label.startswith("GLS") or label == "GLS":
        return None
    try:
        return float(label[len("GLS") :])
    except ValueError:
        return None


def BN_model_data_loss_func_info(model_params):
    binn_model_params = model_params["binn_model_params"]
    return binn_model_params["BNdata_loss_params"]


def BN_model_data_loss_func(model_params):
    binn_model_params = model_params["binn_model_params"]
    BNdata_loss_params = binn_model_params["BNdata_loss_params"]
    BNdataLossFuncLabel = BNdata_loss_params["BNdataLossFuncLabel"]
    gls_power = _parse_gls_power_label(BNdataLossFuncLabel)

    if BNdataLossFuncLabel == "MSE":
        BN_model_data_loss_func = data_loss_MSE
    elif BNdataLossFuncLabel == "MSEmodified10":
        BN_model_data_loss_func = data_loss_MSEmodified10
    elif BNdataLossFuncLabel == "GLS":
        BN_model_data_loss_func = data_loss_GLS
    elif gls_power is not None:
        BNdata_loss_params["GLSpow"] = gls_power
        BN_model_data_loss_func = data_loss_GLSpow
    else:
        raise ValueError(f"Unknown data loss label: {BNdataLossFuncLabel!r}")

    func_info = BN_model_data_loss_func_info(model_params)
    return func_info, BN_model_data_loss_func
