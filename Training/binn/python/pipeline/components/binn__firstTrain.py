"""Initial training helpers for BINN model runs.

Contents
--------
- BN_model_fit_smart_func_info: return fit-parameter metadata for bookkeeping.
- BN_model_fit_first_smart: train a fresh BINN model and manage checkpoints.
- exc_data_pipeline: regenerate data if required before training.
- BN_first_train_smart: orchestrate initial training for one parameter setting.
"""

import os

import torch

from Training.binn.python.Modules.Utils.ModelWrapper import ModelWrapper
from Training.binn.python.Modules.Utils.parse import to_torch_grad
from Training.binn.python.pipeline.components.binn__modelConstructor import (
    BN_model_construction,
)
from Training.binn.python.pipeline.components.binn__loadData import (
    BN_load_raw_data,
)
from Training.binn.python.pipeline.components.binn__splitTV import BN_TVsplit
from Training.data.python.pipeline.components.data__simulate import DATA_sim


def BN_model_fit_smart_func_info(fit_params):
    return fit_params["binn_fit_params"]


def BN_model_fit_first_smart(model_save_dir, NN_binn, model_params, fit_params, TV_dic):
    train_dic = TV_dic["train_dic"]
    val_dic = TV_dic["val_dic"]
    validation_data = val_dic["validation_data"]
    x_train_np = train_dic["x_train_np"]
    y_train_np = train_dic["y_train_np"]
    x_val_np = val_dic["x_val_np"]
    y_val_np = val_dic["y_val_np"]

    binn_model_params = model_params["binn_model_params"]
    binn_construction_params = binn_model_params["binn_construction_params"]
    device = binn_construction_params["binnDevice"]

    x_train = to_torch_grad(x_train_np, device)
    y_train = to_torch_grad(y_train_np, device)

    if validation_data:
        x_val = to_torch_grad(x_val_np, device)
        y_val = to_torch_grad(y_val_np, device)
        validation_data = [x_val, y_val]

    binn_fit_params = fit_params["binn_fit_params"]
    binn_fit_params_additionals = fit_params["binn_fit_params_additionals"]
    binnEpochs = binn_fit_params_additionals["binnEpochs"]
    printFreq = binn_fit_params_additionals["printFreq"]
    storeConstraintLosses = binn_fit_params_additionals.get(
        "storeConstraintLosses", True
    )
    storeAdamDiagnostics = binn_fit_params_additionals.get(
        "storeAdamDiagnostics", False
    )
    storeDataLossDiagnostics = binn_fit_params_additionals.get(
        "storeDataLossDiagnostics", False
    )
    storeFixedGridPDEDiagnostics = binn_fit_params_additionals.get(
        "storeFixedGridPDEDiagnostics", False
    )
    fixedGridPDEFrequency = binn_fit_params_additionals.get(
        "fixedGridPDEFrequency", 10
    )
    fixedGridPDEShape = binn_fit_params_additionals.get("fixedGridPDEShape", None)

    binnLR = binn_fit_params["binnLR"]
    binnBatchSize = binn_fit_params["binnBatchSize"]
    binnRelUpdateThresh = binn_fit_params["binnRelUpdateThresh"]
    binnRelSaveThresh = binn_fit_params["binnRelSaveThresh"]
    binnES = binn_fit_params["binnES"]

    binnModelLabel = binn_fit_params["binnModelLabel"]

    parameters = NN_binn.parameters()
    opt = torch.optim.Adam(parameters, lr=binnLR)
    weights_path_relative = os.path.join(model_save_dir, f"Weights_binn_num{binnModelLabel}")
    os.makedirs(weights_path_relative, exist_ok=True)

    modelW = ModelWrapper(
        model=NN_binn,
        optimizer=opt,
        loss=NN_binn.loss,
        save_name=f"{weights_path_relative}/test",
    )

    modelW.model_save_dir = model_save_dir
    modelW.binnModelLabel = binnModelLabel

    modelW.x_train = x_train_np
    modelW.y_train = y_train_np
    modelW.x_val = x_val_np
    modelW.y_val = y_val_np

    modelW.x_train_torch = x_train
    modelW.y_train_torch = y_train
    modelW.validation_data = validation_data

    modelW.batch_size = binnBatchSize
    modelW.early_stopping = binnES
    modelW.rel_update_thresh = binnRelUpdateThresh
    modelW.rel_save_thresh = binnRelSaveThresh

    modelW.verbose = 1

    modelW.fit(
        x_tr_input=modelW.x_train_torch,
        y_tr_input=modelW.y_train_torch,
        batch_size=modelW.batch_size,
        epochs=int(binnEpochs),
        verbose=modelW.verbose,
        validation_data=modelW.validation_data,
        early_stopping=modelW.early_stopping,
        rel_update_thresh=modelW.rel_update_thresh,
        rel_save_thresh=modelW.rel_save_thresh,
        print_freq=printFreq,
        store_constraint_losses=storeConstraintLosses,
        store_adam_diagnostics=storeAdamDiagnostics,
        store_data_loss_diagnostics=storeDataLossDiagnostics,
        store_fixed_grid_pde_diagnostics=storeFixedGridPDEDiagnostics,
        fixed_grid_pde_frequency=fixedGridPDEFrequency,
        fixed_grid_pde_shape=fixedGridPDEShape,
    )

    func_info = BN_model_fit_smart_func_info(fit_params=fit_params)
    return func_info, modelW


def exc_data_pipeline(data_obj_params):
    DATA_sim(data_obj_params=data_obj_params)


def BN_first_train_smart(model_save_dir, data_obj_params, TV_params, model_params, fit_params):
    exc_data_pipeline(data_obj_params=data_obj_params)
    _, data_obj_orig = BN_load_raw_data(data_obj_params=data_obj_params)
    data_obj = data_obj_orig

    _, TV_dic = BN_TVsplit(data_obj=data_obj, model_params=model_params, TV_params=TV_params)

    _, NN_binn = BN_model_construction(
        data_obj_orig=data_obj_orig,
        data_obj_params=data_obj_params,
        model_params=model_params,
    )

    binn_fit_params = fit_params["binn_fit_params"]
    binnModelLabel = binn_fit_params["binnModelLabel"]
    binnTV_params = TV_params["binnTV_params"]
    binn_TV_additionals = binnTV_params["binnGenerateIndicesArgs"]

    print("--------------------------------------------------------------------------------------")
    print(
        f"-------------------------- BINN FIRST TRAINING (model_num ={binnModelLabel}, "
        f"binn_TV_additionals = {binn_TV_additionals}) ---------------------------"
    )
    print("--------------------------------------------------------------------------------------")

    func_info, modelW = BN_model_fit_first_smart(
        model_save_dir=model_save_dir,
        NN_binn=NN_binn,
        model_params=model_params,
        fit_params=fit_params,
        TV_dic=TV_dic,
    )

    return data_obj, modelW
