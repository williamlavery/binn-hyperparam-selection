"""Train-validation splitting helpers for BINN data objects.

Contents
--------
- BN_random_val_indices: sample validation indices for a dataset.
- BN_TVsplit_func_info: return split metadata for bookkeeping.
- BN_TVsplit: build train and validation tensors for one dataset.
"""

import numpy as np

from Training.binn.python.Modules.Utils.parse import unravel_one_level


def BN_random_val_indices(inputs, VF, binnGenerateIndicesArgs):
    seed = binnGenerateIndicesArgs["binnTVsplitSeed"]
    N = len(inputs)
    num_val = int(VF * N)
    np.random.seed(seed)
    p = np.random.permutation(N)
    return p[-num_val:]


def BN_TVsplit_func_info(TV_params):
    binnTV_params = TV_params["binnTV_params"]
    binnVF = binnTV_params["binnVF"]
    if binnVF:
        return unravel_one_level(binnTV_params)
    return {"binnVF": binnVF}


def BN_TVsplit(data_obj, TV_params, model_params):
    inputs = data_obj.inputs
    u = data_obj.u.copy()
    outputs = u.reshape(-1)[:, None]

    binnTV_params = TV_params["binnTV_params"]
    binnVF = binnTV_params["binnVF"]
    binnGenerateIndicesLabel = binnTV_params["binnGenerateIndicesLabel"]
    binnGenerateIndicesArgs = binnTV_params["binnGenerateIndicesArgs"]

    if binnVF:
        if binnGenerateIndicesLabel == "random":
            val_indices = BN_random_val_indices(inputs, binnVF, binnGenerateIndicesArgs)

        N = len(inputs)
        all_indices = np.arange(N)
        train_indices = np.setdiff1d(all_indices, val_indices)

        x_train_np = inputs[train_indices]
        y_train_np = outputs[train_indices]
        x_val_np = inputs[val_indices]
        y_val_np = outputs[val_indices]

        val_dic = {
            "x_val_np": x_val_np,
            "y_val_np": y_val_np,
            "validation_data": [x_val_np, y_val_np],
        }
    else:
        x_train_np = inputs
        y_train_np = outputs
        val_dic = {
            "x_val_np": [],
            "y_val_np": [],
            "validation_data": None,
        }

    train_dic = {"x_train_np": x_train_np, "y_train_np": y_train_np}
    TV_dic = {"train_dic": train_dic, "val_dic": val_dic}
    func_info = BN_TVsplit_func_info(TV_params=TV_params)

    return func_info, TV_dic
