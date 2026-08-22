"""Build configuration dictionaries for BINN training sweeps.

Contents
--------
- BinnConfig: frozen dataclass of default BINN training-sweep settings.
- _load_original_x_time_axis: load the original experimental spatial and temporal axes.
- _build_spatial_axes: build spatial axes, keeping 2D on the same range as the 1D data.
- build_config: materialize the runtime, grid, and parameter-store config dictionary.
"""

from __future__ import annotations

import __main__
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from Training.data.python.Modules.dataClass import OriginalData, generate_inputs
from Training.pipeline_runtime import find_project_root


@dataclass(frozen=True)
class BinnConfig:
    original_data_relative_path: str = (
        "Training/data/dataObj/Lagergren_et_al_2020/originalDataObj.npy"
    )
    data_x1_num: int = 38
    data_x2_num: int = 1
    data_t_nums: list[int] = field(default_factory=lambda: [5])
    data_k: float = 1
    gammas: list[float] = field(default_factory=lambda: [0])
    noise_percents: list[float] = field(default_factory=lambda: [0])
    noise_seeds: list[int] = field(default_factory=lambda: [0])
    ic_labels: list[str] = field(default_factory=lambda: ["cos"])
    diff_labels: list[str] = field(default_factory=lambda: ["const"])
    grow_labels: list[str] = field(default_factory=lambda: ["const", "linear", "quadratic", "exp"]) 
    binn_usizes: list[int] = field(default_factory=lambda: [64])
    binn_dsizes: list[int] = field(default_factory=lambda: [8])
    binn_gsizes: list[int] = field(default_factory=lambda: [8])
    binn_model_labels: list[int] = field(default_factory=lambda: [0])
    binn_es_values: list[int] = field(default_factory=lambda: [2000, 3000, 5000])
    binn_tv_split_seeds: list[int] = field(default_factory=lambda: [0,1,2])
    binn_vfs: list[float] = field(default_factory=lambda: [0.2])
    surface_weights: list[float] = field(default_factory=lambda: [1e0])
    pde_weights: list[float] = field(default_factory=lambda: [1e0])
    all_constraints: list[bool] = field(default_factory=lambda: [True])
    constraint_tuples: list[tuple[bool, bool, bool, bool]] = field(
        default_factory=lambda: [ (True, True, True, True) ]
    )
    constraint_weights: dict = field(
        default_factory=lambda: {
            "D_bound_weight": [0],
            "D_mono_weight": [1e1],
            "G_bound_weight": [0],
            "G_mono_weight": [1e1], 
        }
    )
    # Diffusion bound in mm^2 day^-1; growth bound in day^-1 (= (-0.48, 2.4)).
    # Bounding is disabled by default (D_bound_weight = G_bound_weight = 0 above);
    # only monotonicity is enforced.
    constraint_bounds: dict = field(
        default_factory=lambda: {
            "D_bound": (0, 0.1),
            "G_bound": (-0.02 / (1 / 24), 0.1 / (1 / 24)),
        }
    )
    done_param_bool: bool = False
    perfect_pde: bool = False
    binn_activation: str = "silu"
    binn_surface_hidden_layers: list[int] = field(default_factory=lambda: [3])
    binn_dg_hidden_layers: list[int] = field(default_factory=lambda: [3])
    binn_generate_indices_label: str = "random"
    binn_device: str = "cpu"
    binn_lr: float = 1e-3
    binn_rel_update_thresh: float = 0.05
    binn_rel_save_thresh: float = 0.05
    binn_epochs: int = int(1e6)
    binn_es_check: list[int] = field(default_factory=lambda: [2000, 3000])
    print_freq: int = 100
    bc_bool: int = 0
    num_pde_samples: list[int] = field(default_factory=lambda: [100])
    bn_data_loss_labels: list[str] = field(default_factory=lambda: [f"MSE"])
    constraint_samples: list[int] = field(default_factory=lambda: [100])
    store_constraint_losses: bool = True
    store_adam_diagnostics: bool = True
    store_data_loss_diagnostics: bool = True
    plot_bool: bool = False
    overwrite_bool: bool = False


def _load_original_x_time_axis(config: BinnConfig, project_root: Path) -> np.ndarray:
    __main__.OriginalData = OriginalData
    original_path = project_root / config.original_data_relative_path
    original_data = np.load(original_path, allow_pickle=True).item(0)
    return original_data.x, original_data.t


def _build_spatial_axes(
    original_x: np.ndarray, data_x1_num: int, data_x2_num: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build spatial axes, keeping 2D on the same range as the 1D data."""
    def _build_x2_axis(x1: np.ndarray, data_x2_num: int) -> np.ndarray | None:
        """Return a real x2 axis only when the config requests a 2D grid."""
        if data_x2_num <= 1:
            return None
        return np.linspace(np.min(x1), np.max(x1), data_x2_num)

    x1 = np.linspace(np.min(original_x), np.max(original_x), data_x1_num)
    x2 = _build_x2_axis(x1, data_x2_num)
    return x1, x2


def build_config() -> dict:
    config = BinnConfig()
    project_root = find_project_root(Path(__file__))
    original_x, original_t = _load_original_x_time_axis(config, project_root)

    x1, x2 = _build_spatial_axes(original_x, config.data_x1_num, config.data_x2_num)
    ts = [np.linspace(np.min(original_t), np.max(original_t), n) for n in config.data_t_nums]
    dim = 1 if x2 is None else 2

    return {
        "project_root": project_root,
        "grid": {
            "dim": dim,
            "x1": x1,
            "x2": x2,
            "ts": ts,
            "K": config.data_k,
        },
        "data": {
            "gammas": config.gammas,
            "noisePercents": config.noise_percents,
            "noiseSeeds": config.noise_seeds,
            "ICLabels": config.ic_labels,
            "diffLabels": config.diff_labels,
            "growLabels": config.grow_labels,
        },
        "binn": {
            "binnUsizes": config.binn_usizes,
            "binnDsizes": config.binn_dsizes,
            "binnGsizes": config.binn_gsizes,
            "binnModelLabels": config.binn_model_labels,
            "binnESs": config.binn_es_values,
            "binnTVsplitSeeds": config.binn_tv_split_seeds,
            "binnVFs": config.binn_vfs,
            "surfaceWeights": config.surface_weights,
            "pdeWeights": config.pde_weights,
            "allConstraints": config.all_constraints,
            "constraintTuples": config.constraint_tuples,
            "constraintWeights": config.constraint_weights,
            "constraintBounds": config.constraint_bounds,
            "DoneParamBool": config.done_param_bool,
            "perfectPDE": config.perfect_pde,
            "binnActivation": config.binn_activation,
            "binnSurfaceHiddenLayers": config.binn_surface_hidden_layers,
            "binnDGHiddenLayers": config.binn_dg_hidden_layers,
            "binnGenerateIndicesLabel": config.binn_generate_indices_label,
            "binnDevice": config.binn_device,
            "binnLR": config.binn_lr,
            "binnRelUpdateThresh": config.binn_rel_update_thresh,
            "binnRelSaveThresh": config.binn_rel_save_thresh,
            "binnEpochs": config.binn_epochs,
            "binnES_check": config.binn_es_check,
            "printFreq": config.print_freq,
            "BCbool": config.bc_bool,
            "numPDEsamples": config.num_pde_samples,
            "BNdataLossFuncLabels": config.bn_data_loss_labels,
            "constraintSamples": config.constraint_samples,
            "storeConstraintLosses": config.store_constraint_losses,
            "storeAdamDiagnostics": config.store_adam_diagnostics,
            "storeDataLossDiagnostics": config.store_data_loss_diagnostics,
        },
        "runtime": {
            "plot_bool": config.plot_bool,
            "overwrite_bool": config.overwrite_bool,
        },
    }
