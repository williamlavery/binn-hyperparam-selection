"""Build configuration dictionaries for synthetic data-generation sweeps.

Contents
--------
- _load_original_x_time_axis: load the original experimental spatial and temporal axes.
- _build_spatial_axes: build 1D or 2D spatial grids from the requested axis sizes.
- build_config: materialize the runtime, grid, and parameter-store config dict.
"""

from __future__ import annotations

import __main__
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from Training.data.python.Modules.dataClass import OriginalData, generate_inputs
from Training.pipeline_runtime import find_project_root


@dataclass(frozen=True)
class DataConfig:
	original_data_relative_path: str = (
		"Training/data/dataObj/Lagergren_et_al_2020/originalDataObj.npy"
	)
	data_x1_num: int = 38
	data_x2_num: int = 1
	data_t_num: int = 5
	data_k: float = 1
	data_gammas: list[float] = field(default_factory=lambda: [0])
	data_noise_percents: list[float] = field(default_factory=lambda: [0])
	data_noise_seeds: list[int] = field(default_factory=lambda: [0])
	data_ic_labels: list[str] = field(default_factory=lambda: ["cos"])
	data_diff_labels: list[str] = field(default_factory=lambda: ["const"])
	data_grow_labels: list[str] = field(default_factory=lambda: ["const", "linear", "quadratic", "exp"])
	plot_bool: bool = False
	overwrite_bool: bool = True


def _load_original_x_time_axis(config: DataConfig, project_root: Path) -> np.ndarray:
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

	# Newer 2D-only unit-square version:
	# if data_x2_num <= 1:
	# 	x1 = np.linspace(np.min(original_x), np.max(original_x), data_x1_num)
	# 	return x1, None
	#
	# x1 = np.linspace(0.0, 1.0, data_x1_num)
	# x2 = np.linspace(0.0, 1.0, data_x2_num)
	# return x1, x2


def build_config() -> dict:
	"""Build the complete data-generation configuration."""
	config = DataConfig()
	project_root = find_project_root(Path(__file__))
	original_x, original_t = _load_original_x_time_axis(config, project_root)

	x1, x2 = _build_spatial_axes(original_x, config.data_x1_num, config.data_x2_num)
	t = np.linspace(np.min(original_t), np.max(original_t), config.data_t_num)
	dim = 1 if x2 is None else 2

	return {
		"project_root": project_root,
		"grid": {
			"dim": dim,
			"x1": x1,
			"x2": x2,
			"t": t,
			"inputs": generate_inputs(x1, x2, t, dim=dim),
			"K": config.data_k,
		},
		"sweep": {
			"dataGammas": config.data_gammas,
			"dataNoisePercents": config.data_noise_percents,
			"dataNoiseSeeds": config.data_noise_seeds,
			"dataICLabels": config.data_ic_labels,
			"dataDiffLabels": config.data_diff_labels,
			"dataGrowLabels": config.data_grow_labels,
		},
		"runtime": {
			"plot_bool": config.plot_bool,
			"overwrite_bool": config.overwrite_bool,
		},
	}
