"""Load saved data objects and construct their identifying metadata.

Contents
--------
- BN_load_func_info: return the data-object signature used in BINN run paths.
- BN_load_raw_func_info: assemble the raw data-object metadata from data settings.
- BN_load_raw_data: load the saved data object for the current parameter setting.
"""

import __main__
import os

import numpy as np

from Training.data.python.Modules.dataClass import Data
from Training.binn.python.Modules.Utils.parse import dictToPath

__main__.Data = Data


def BN_load_func_info(data_obj_params, TV_params, model_params, fit_params):
    return BN_load_raw_func_info(data_obj_params=data_obj_params)


def BN_load_raw_func_info(data_obj_params):
    RDEq_params_store = data_obj_params["RDEq_params_store"]
    RDEq_params = data_obj_params["RDEq_params"]
    add_noise_params = data_obj_params["add_noise_params"]

    x1 = RDEq_params_store["x1"]
    x2 = RDEq_params_store.get("x2")
    t = RDEq_params_store["t"]
    K = RDEq_params_store["K"]

    func_info = {
        "dataX1num": len(x1),
        "dataX2num": 1 if x2 is None else len(x2),
        "dataTnum": len(t),
        "dataK": K,
    }

    func_info.update(RDEq_params)
    func_info.update(add_noise_params)

    return func_info


def BN_load_raw_data(data_obj_params):
    additional_params = data_obj_params["additional_params"]
    dataObj_path = additional_params["inital_path"]

    func_info = BN_load_raw_func_info(data_obj_params=data_obj_params)
    info_path = dictToPath(func_info)

    load_path = os.path.join(os.path.join(dataObj_path, info_path, "data_obj.npy"))
    data_obj = np.load(load_path, allow_pickle=True).item(0)

    return func_info, data_obj
