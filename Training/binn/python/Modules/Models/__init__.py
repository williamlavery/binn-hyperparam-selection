"""Model modules used by the BINN pipeline.

Contents
--------
- register_legacy_pickle_globals: expose legacy notebook globals needed by older saved Torch objects.
"""

import sys

from . import BuildBINNs as _BuildBINNs
from . import BuildMLP as _BuildMLP

_BuildMLP.BuildMLP2 = _BuildMLP.BuildMLP

_this_package = sys.modules[__name__]
for _alias, _target in (
	("binn", "Training.binn"),
	("binn.python", "Training.binn.python"),
	("binn.python.Modules", "Training.binn.python.Modules"),
):
	if _target in sys.modules:
		sys.modules.setdefault(_alias, sys.modules[_target])
sys.modules.setdefault("binn.python.Modules.Models", _this_package)

for _prefix in ("Training.binn.python.Modules.Models", "binn.python.Modules.Models"):
	sys.modules[f"{_prefix}.BuildBINNs_1D"] = _BuildBINNs
	sys.modules[f"{_prefix}.BuildMLP2"] = _BuildMLP


def register_legacy_pickle_globals() -> None:
	"""Expose old notebook globals needed by previously saved torch objects."""
	import __main__

	from Training.binn.python.Modules.Utils.ModelWrapper import ModelWrapper

	legacy_symbols = {
		"ModelWrapper": ModelWrapper,
		"BINN": _BuildBINNs.BINN,
		"u_MLP": _BuildBINNs.u_MLP,
		"D_MLP": _BuildBINNs.D_MLP,
		"G_MLP": _BuildBINNs.G_MLP,
		"pde_loss_without_bc": _BuildBINNs.pde_loss_without_bc,
		"pde_loss_with_bc": _BuildBINNs.pde_loss_with_bc,
		"data_loss_MSE": _BuildBINNs.data_loss_MSE,
		"data_loss_MSEmodified10": _BuildBINNs.data_loss_MSEmodified10,
		"data_loss_GLS": _BuildBINNs.data_loss_GLS,
		"data_loss_GLSpow": _BuildBINNs.data_loss_GLSpow,
		"generate_random_inputs": _BuildBINNs.generate_random_inputs,
	}
	for name, value in legacy_symbols.items():
		setattr(__main__, name, value)
