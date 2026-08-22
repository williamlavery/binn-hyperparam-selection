"""Run simulation, add noise, save artifacts, and optionally preview outputs.

Contents
--------
- _apply_artificial_flip: optionally invert simulated densities for the flipped-cosine ablation case.
- DATA_finder: build the on-disk output directory for a parameter set.
- DATA_sim: run the full synthetic data-generation pipeline for one setting.
"""

import os

import numpy as np

from Training.data.python.Modules.dataClass import plot_1d_data, plot_2d_data
from Training.data.python.Modules.parse import dictToPath
from Training.data.python.pipeline.components.data__addNoise import (
    DATA_add_noise,
    DATA_add_noise_func_info,
)
from Training.data.python.pipeline.components.data__constructor import DATA_data_construct
from Training.data.python.pipeline.components.data__initialise import (
    DATA_RDEq2,
    DATA_RDEq2_func_info,
)

def _apply_artificial_flip(data_obj_params, u, u_clean):
    if data_obj_params["RDEq_params"].get("dataICLabel") != "cosFlippedArtifical":
        return u, u_clean

    carrying_capacity = data_obj_params["RDEq_params_store"]["K"]
    return carrying_capacity - u, carrying_capacity - u_clean


def DATA_finder(data_obj_params):
    """Return the output directory for a data object and whether it exists."""
    func_info = {}
    func_info.update(DATA_RDEq2_func_info(data_obj_params))
    func_info.update(DATA_add_noise_func_info(data_obj_params))

    data_obj_path = data_obj_params["additional_params"]["inital_path"]
    data_path_dir = os.path.join(data_obj_path, dictToPath(func_info))

    file_present = os.path.isfile(os.path.join(data_path_dir, "data_obj.npy"))
    os.makedirs(data_path_dir, exist_ok=True)
    return data_path_dir, file_present


def DATA_sim(data_obj_params):
    """Generate, noise, and save one reaction-diffusion data object."""
    data_path_dir, file_present = DATA_finder(data_obj_params)
    data_path_file = os.path.join(data_path_dir, "data_obj.npy")
    additional_params = data_obj_params["additional_params"]

    if file_present and not additional_params["overwrite_bool"]:
        print(f"Using existing data object: {data_path_file}")
    else:
        _, u_clean, (theta_D, theta_G) = DATA_RDEq2(data_obj_params)
        _, u, additional_info = DATA_add_noise(u_clean, data_obj_params)
        u, u_clean = _apply_artificial_flip(data_obj_params, u, u_clean)

        data_obj = DATA_data_construct(data_obj_params, u, u_clean, theta_D, theta_G)
        data_obj.additional_info = additional_info
        np.save(data_path_file, arr=data_obj, allow_pickle=True)
        print(f"Saved data object: {data_path_file}")

    if additional_params["plot_bool"]:
        data_obj = np.load(data_path_file, allow_pickle=True).item()
        if getattr(data_obj, "dim", 1) == 1:
            plot_1d_data(data_obj.x1, data_obj.t, data_obj.u, data_obj.u_clean)
        else:
            plot_2d_data(data_obj.x1, data_obj.x2, data_obj.t, data_obj.u, data_obj.u_clean)
