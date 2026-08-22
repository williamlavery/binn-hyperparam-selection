"""Retraining helpers for existing BINN model directories.

Contents
--------
- _sync_retrain_metadata: bind a loaded wrapper to the current run directory and ES settings.
- _run_metadata: build the standard run metadata block for copied checkpoints.
- _sync_saved_wrapper: rewrite a copied checkpoint with current run metadata if it exists.
- _sync_run_checkpoints: sync top-level and sidecar wrapper checkpoints for the current run.
- BN_model_finder: build the save directory for a BINN run and report whether it exists.
- BN_model_check_ES: inspect a saved run to decide whether retraining should continue.
- BN_model_fit_again: resume fitting from an existing wrapper checkpoint.
- BN_retrain: orchestrate checkpoint loading, metadata sync, and resumed training.
"""

import os
from copy import deepcopy

import numpy as np
import torch

from Training.binn.python.Modules.Models import register_legacy_pickle_globals
from Training.binn.python.Modules.Utils.parse import dictToPath
from Training.binn.python.pipeline.components.binn__modelConstructor import (
    BN_model_construction_func_info,
)
from Training.binn.python.pipeline.components.binn__loadData import BN_load_func_info
from Training.binn.python.pipeline.components.binn__splitTV import BN_TVsplit_func_info
from Training.binn.python.pipeline.components.binn__firstTrain import (
    BN_model_fit_smart_func_info,
)



def _sync_retrain_metadata(modelW, model_dir_path, fit_params):
    """Bind a loaded wrapper to the current run directory and ES settings."""
    binn_fit_params = fit_params["binn_fit_params"]
    binn_fit_params_additionals = fit_params["binn_fit_params_additionals"]
    binnModelLabel = binn_fit_params["binnModelLabel"]

    weights_dir = os.path.join(model_dir_path, f"Weights_binn_num{binnModelLabel}")
    os.makedirs(weights_dir, exist_ok=True)

    modelW.model_save_dir = model_dir_path
    modelW.binnModelLabel = binnModelLabel
    modelW.save_name = os.path.join(weights_dir, "test")
    modelW.batch_size = binn_fit_params["binnBatchSize"]
    modelW.early_stopping = binn_fit_params["binnES"]
    modelW.rel_update_thresh = binn_fit_params["binnRelUpdateThresh"]
    modelW.rel_save_thresh = binn_fit_params["binnRelSaveThresh"]
    modelW.print_freq = binn_fit_params_additionals["printFreq"]
    modelW.store_constraint_losses = binn_fit_params_additionals.get(
        "storeConstraintLosses", True
    )
    modelW.store_adam_diagnostics = binn_fit_params_additionals.get(
        "storeAdamDiagnostics", False
    )
    modelW.store_data_loss_diagnostics = binn_fit_params_additionals.get(
        "storeDataLossDiagnostics", False
    )
    modelW.store_fixed_grid_pde_diagnostics = binn_fit_params_additionals.get(
        "storeFixedGridPDEDiagnostics", False
    )
    modelW.fixed_grid_pde_frequency = binn_fit_params_additionals.get(
        "fixedGridPDEFrequency", 10
    )
    modelW.fixed_grid_pde_shape = binn_fit_params_additionals.get(
        "fixedGridPDEShape", None
    )
    return modelW



def _run_metadata(model_dir_path, fit_params):
    binn_fit_params = fit_params["binn_fit_params"]
    binn_fit_params_additionals = fit_params["binn_fit_params_additionals"]
    binnModelLabel = binn_fit_params["binnModelLabel"]
    weights_dir = os.path.join(model_dir_path, f"Weights_binn_num{binnModelLabel}")
    return {
        "model_save_dir": model_dir_path,
        "binnModelLabel": binnModelLabel,
        "save_name": os.path.join(weights_dir, "test"),
        "batch_size": binn_fit_params["binnBatchSize"],
        "early_stopping": binn_fit_params["binnES"],
        "rel_update_thresh": binn_fit_params["binnRelUpdateThresh"],
        "rel_save_thresh": binn_fit_params["binnRelSaveThresh"],
        "print_freq": binn_fit_params_additionals["printFreq"],
        "store_constraint_losses": binn_fit_params_additionals.get(
            "storeConstraintLosses", True
        ),
        "store_adam_diagnostics": binn_fit_params_additionals.get(
            "storeAdamDiagnostics", False
        ),
        "store_data_loss_diagnostics": binn_fit_params_additionals.get(
            "storeDataLossDiagnostics", False
        ),
        "store_fixed_grid_pde_diagnostics": binn_fit_params_additionals.get(
            "storeFixedGridPDEDiagnostics", False
        ),
        "fixed_grid_pde_frequency": binn_fit_params_additionals.get(
            "fixedGridPDEFrequency", 10
        ),
        "fixed_grid_pde_shape": binn_fit_params_additionals.get(
            "fixedGridPDEShape", None
        ),
    }


def _sync_saved_wrapper(checkpoint_path, model_dir_path, fit_params):
    """Rewrite a copied checkpoint with current run metadata if it exists."""
    if not os.path.exists(checkpoint_path):
        return

    register_legacy_pickle_globals()
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if isinstance(checkpoint, dict) and checkpoint.get("format") == "ModelWrapperCheckpoint":
        checkpoint["run_metadata"] = _run_metadata(model_dir_path, fit_params)
        torch.save(checkpoint, checkpoint_path)
        return

    modelW = _sync_retrain_metadata(checkpoint, model_dir_path, fit_params)
    torch.save(modelW, checkpoint_path)


def _sync_run_checkpoints(model_dir_path, fit_params):
    """Sync top-level and sidecar wrapper checkpoints for the current run."""
    binnModelLabel = fit_params["binn_fit_params"]["binnModelLabel"]
    weights_prefix = os.path.join(
        model_dir_path,
        f"Weights_binn_num{binnModelLabel}",
        "test",
    )
    for checkpoint_path in (
        os.path.join(model_dir_path, f"binnModel{binnModelLabel}.pth"),
        f"{weights_prefix}_best_val.pth",
        f"{weights_prefix}_ES.pth",
        f"{weights_prefix}_expired.pth",
    ):
        _sync_saved_wrapper(checkpoint_path, model_dir_path, fit_params)


def BN_model_finder(path_intro, data_obj_params, TV_params, model_params, fit_params):
    all_func_info = {}
    func_info_inital = BN_load_func_info(
        data_obj_params=data_obj_params,
        TV_params=TV_params,
        model_params=model_params,
        fit_params=fit_params,
    )
    all_func_info.update(func_info_inital)

    func_info_TV = BN_TVsplit_func_info(TV_params=TV_params)
    all_func_info.update(func_info_TV)

    func_info_build = BN_model_construction_func_info(model_params=model_params)
    all_func_info.update(func_info_build)

    func_info_fit = BN_model_fit_smart_func_info(fit_params=fit_params)
    all_func_info.update(func_info_fit)

    full_path = os.path.join(path_intro, dictToPath(all_func_info))

    if os.path.exists(full_path) and os.listdir(full_path):
        return full_path, 1
    os.makedirs(full_path, exist_ok=True)
    return full_path, 0


def BN_model_check_ES(path_intro, data_obj_params, TV_params, model_params, fit_params):
    import shutil

    def copy_dir_contents(src, dest):
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source directory does not exist: {src}")

        os.makedirs(dest, exist_ok=True)

        for item in os.listdir(src):
            src_path = os.path.join(src, item)
            dest_path = os.path.join(dest, item)

            if os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dest_path)

    model_save_dir_current, exist_bool = BN_model_finder(
        path_intro=path_intro,
        data_obj_params=data_obj_params,
        TV_params=TV_params,
        model_params=model_params,
        fit_params=fit_params,
    )

    if exist_bool:
        return None

    fit_params_to_update = deepcopy(fit_params)
    ES_current = fit_params["binn_fit_params"]["binnES"]

    sorted_ES = np.array(
        sorted(fit_params["binn_fit_params_additionals"]["binnES_check"], reverse=True)
    )

    for ES in sorted_ES[sorted_ES < ES_current]:
        fit_params_to_update["binn_fit_params"]["binnES"] = ES

        model_save_dir_old, exist_bool = BN_model_finder(
            path_intro=path_intro,
            data_obj_params=data_obj_params,
            TV_params=TV_params,
            model_params=model_params,
            fit_params=fit_params_to_update,
        )

        if exist_bool:
            binnModelLabel = fit_params["binn_fit_params"]["binnModelLabel"]
            copy_dir_contents(model_save_dir_old, model_save_dir_current)

            model_path = os.path.join(model_save_dir_current, f"binnModel{binnModelLabel}.pth")

            register_legacy_pickle_globals()
            modelW = torch.load(model_path, weights_only=False)
            modelW = _sync_retrain_metadata(modelW, model_save_dir_current, fit_params)
            torch.save(modelW, model_path)
            _sync_run_checkpoints(model_save_dir_current, fit_params)

            print("-----------------------------------------------")
            print(f"Utilised model with ES={ES}")
            print("-----------------------------------------------")

            return 1

    return None


def BN_model_fit_again(model_dir_path, TV_params, fit_params, load_ES_bool=True):
    binnModelLabel = fit_params["binn_fit_params"]["binnModelLabel"]
    model_path_full = os.path.join(model_dir_path, f"binnModel{binnModelLabel}.pth")
    binn_fit_params_additionals = fit_params["binn_fit_params_additionals"]
    binnEpochs = binn_fit_params_additionals["binnEpochs"]
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

    register_legacy_pickle_globals()
    modelW = torch.load(model_path_full, weights_only=False)
    modelW = _sync_retrain_metadata(modelW, model_dir_path, fit_params)
    _sync_run_checkpoints(model_dir_path, fit_params)
    total_train_losses = len(modelW.train_loss_list)
    print(f"Currently have trained {total_train_losses} epochs")

    if (
        modelW.early_stopping is not None
        and total_train_losses - modelW.last_improved >= modelW.early_stopping
    ):
        print("Stopped training. Early stopping.")
        return modelW

    if load_ES_bool:
        modelW.load_ES()
    else:
        modelW.load_expired()
    modelW = _sync_retrain_metadata(modelW, model_dir_path, fit_params)

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
        print_freq=modelW.print_freq,
        store_constraint_losses=storeConstraintLosses,
        store_adam_diagnostics=storeAdamDiagnostics,
        store_data_loss_diagnostics=storeDataLossDiagnostics,
        store_fixed_grid_pde_diagnostics=storeFixedGridPDEDiagnostics,
        fixed_grid_pde_frequency=fixedGridPDEFrequency,
        fixed_grid_pde_shape=fixedGridPDEShape,
    )

    return modelW


def BN_retrain(model_dir_path, TV_params, fit_params, load_ES_bool=True):
    binnModelLabel = fit_params["binn_fit_params"]["binnModelLabel"]
    binnSplitSeed = TV_params["binnTV_params"]["binnGenerateIndicesArgs"]["binnTVsplitSeed"]

    print("--------------------------------------------------------------------------------------")
    print(
        f"-------------------------- BINN RETRAINING (model_num ={binnModelLabel}, "
        f"binnSplitSeed = {binnSplitSeed}) ---------------------------"
    )
    print("--------------------------------------------------------------------------------------")

    model = BN_model_fit_again(
        model_dir_path=model_dir_path,
        TV_params=TV_params,
        fit_params=fit_params,
        load_ES_bool=load_ES_bool,
    )

    return model
