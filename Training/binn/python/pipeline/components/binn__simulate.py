"""Run one BINN training or retraining simulation from configured settings.

Contents
--------
- BN_sim_smart: orchestrate data loading, training, retraining, and saving.
"""

import os

from Training.binn.python.pipeline.components.binn__retrain import (
    BN_model_check_ES,
    BN_model_finder,
    BN_retrain,
)
from Training.binn.python.pipeline.components.binn__firstTrain import (
    BN_first_train_smart,
)
from Training.binn.python.pipeline.components.binn__loadData import BN_load_raw_data
from Training.binn.python.pipeline.components.binn__saveModel import BN_save_model


def BN_sim_smart(data_obj_params, TV_params, model_params, fit_params):
    binn_model_params = model_params["binn_model_params"]
    binn_fit_params = fit_params["binn_fit_params"]
    binn_construction_params = binn_model_params["binn_construction_params"]
    additional_params = data_obj_params["additional_params"]
    binnTV_params = TV_params["binnTV_params"]

    binnModelLabel = binn_fit_params["binnModelLabel"]
    binn_path = additional_params["binn_path"]

    model_save_dir, exist_bool = BN_model_finder(
        path_intro=binn_path,
        data_obj_params=data_obj_params,
        TV_params=TV_params,
        model_params=model_params,
        fit_params=fit_params,
    )

    model_save_path = os.path.join(model_save_dir, f"binnModel{binnModelLabel}.pth")

    utilize_pretrained = BN_model_check_ES(
        path_intro=binn_path,
        data_obj_params=data_obj_params,
        TV_params=TV_params,
        model_params=model_params,
        fit_params=fit_params,
    )

    load_ES_bool = True if utilize_pretrained else False

    if os.path.isfile(model_save_path):
        modelW = BN_retrain(
            model_dir_path=model_save_dir,
            TV_params=TV_params,
            fit_params=fit_params,
            load_ES_bool=load_ES_bool,
        )
        _, data_obj = BN_load_raw_data(data_obj_params)
    else:
        data_obj, modelW = BN_first_train_smart(
            model_save_dir=model_save_dir,
            data_obj_params=data_obj_params,
            TV_params=TV_params,
            model_params=model_params,
            fit_params=fit_params,
        )

    modelW.u_clean = data_obj.u_clean
    modelW.u_nosiy = data_obj.u

    BN_save_model(
        modelW=modelW,
        model_save_dir=model_save_dir,
        data_obj=data_obj,
        TV_params=TV_params,
        model_params=model_params,
        fit_params=fit_params,
    )
