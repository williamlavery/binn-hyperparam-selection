"""Generate model-predicted data objects from trained BINN checkpoints.

Contents
--------
- BN_save_model_generated_dataObj: save a data object populated with BINN predictions.
"""

import os

import numpy as np
import torch

from Training.data.python.Modules.dataClass import Data
from Training.binn.python.Modules.Utils.parse import to_torch_grad


def BN_save_model_generated_dataObj(model_loaded, dataobj, model_label, path_to_dataObj_dir, device):
    dim = getattr(dataobj, "dim", 1)
    Nt = len(dataobj.t)

    model_loaded.load_best_val()

    with torch.no_grad():
        u_pred_flat_torch = model_loaded.model(
            to_torch_grad(dataobj.inputs, device)
        )
        if dim == 1:
            Nx = len(dataobj.x1)
            u = u_pred_flat_torch.reshape(Nx, Nt).detach().cpu().numpy()
            data_obj_binn = Data(
                x1=dataobj.x1,
                t=dataobj.t,
                u_clean=[],
                u=u,
                gamma=dataobj.gamma,
                theta_D=dataobj.theta_D,
                theta_G=dataobj.theta_G,
            )
        else:
            Nx1 = len(dataobj.x1)
            Nx2 = len(dataobj.x2)
            u = u_pred_flat_torch.reshape(Nx1, Nx2, Nt).detach().cpu().numpy()
            data_obj_binn = Data(
                x1=dataobj.x1,
                x2=dataobj.x2,
                t=dataobj.t,
                u_clean=[],
                u=u,
                gamma=dataobj.gamma,
                theta_D=dataobj.theta_D,
                theta_G=dataobj.theta_G,
            )

    dataObj_name = f"data_binn_num{model_label}"
    save_path = os.path.join(path_to_dataObj_dir, dataObj_name)
    np.save(save_path, data_obj_binn)
    print(f"Saved data_obj:'{dataObj_name}'")
