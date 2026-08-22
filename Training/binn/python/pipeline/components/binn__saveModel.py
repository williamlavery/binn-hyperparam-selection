"""Save trained BINN checkpoints and generated prediction artifacts.

Contents
--------
- BN_save_model: persist the trained wrapper, checkpoint, and generated outputs.
"""

import os

import torch

from Training.binn.python.pipeline.components.binn__generateDataObj import (
    BN_save_model_generated_dataObj,
)


def BN_save_model(modelW, model_save_dir, data_obj, TV_params, model_params, fit_params):
    binn_model_params = model_params["binn_model_params"]
    binn_construction_params = binn_model_params["binn_construction_params"]

    binn_fit_params = fit_params["binn_fit_params"]
    binnModelLabel = binn_fit_params["binnModelLabel"]
    device = binn_construction_params["binnDevice"]

    binnTV_params = TV_params["binnTV_params"]
    binnGenerateIndicesArgs = binnTV_params["binnGenerateIndicesArgs"]

    model_save_path_full = os.path.join(model_save_dir, f"binnModel{binnModelLabel}.pth")
    torch.save(modelW, model_save_path_full)

    print("******************************")
    print(f"Model number = {binnModelLabel}")
    print(f"binnGenerateIndicesArgs= {binnGenerateIndicesArgs}")
    print(f"Number of trained epochs = {len(modelW.train_loss_list)}")
    print("******************************")

    BN_save_model_generated_dataObj(
        model_loaded=modelW,
        dataobj=data_obj,
        model_label=binnModelLabel,
        path_to_dataObj_dir=model_save_dir,
        device=device,
    )
