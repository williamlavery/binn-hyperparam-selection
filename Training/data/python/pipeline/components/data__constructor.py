"""Construct serializable data objects from simulated arrays and metadata.

Contents
--------
- DATA_data_construct: build the `Data` object stored by the pipeline.
"""

from Training.data.python.Modules.dataClass import Data


def DATA_data_construct(data_obj_params, u, u_clean, theta_D, theta_G):
    """Build the serializable data object used by the BINN pipeline."""
    RDEq_params_store = data_obj_params["RDEq_params_store"]
    add_noise_params = data_obj_params["add_noise_params"]

    return Data(
        x1=RDEq_params_store["x1"],
        x2=RDEq_params_store.get("x2"),
        t=RDEq_params_store["t"],
        u_clean=u_clean,
        u=u,
        gamma=add_noise_params["dataGamma"],
        theta_D=theta_D,
        theta_G=theta_G,
        K=RDEq_params_store["K"],
    )
