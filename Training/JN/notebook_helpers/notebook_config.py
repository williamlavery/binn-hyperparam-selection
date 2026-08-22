"""Shared notebook configuration values for loading trained BINN models.

Contents
--------
- _first: unwrap scalar-like config values stored as one-element sequences.
- exported notebook defaults for batch size, thresholds, learning rate,
  architecture flags, loss labels, boundary settings, and model labels.
"""

from __future__ import annotations

from Training.binn.python.pipeline.config.binn__config import BinnConfig


_BINN_CONFIG = BinnConfig()


def _first(values):
    return values[0] if isinstance(values, (list, tuple)) else values


binn_batch_size = _BINN_CONFIG.data_x1_num * _BINN_CONFIG.data_x2_num
binn_rel_update_thresh = _BINN_CONFIG.binn_rel_update_thresh
binn_rel_save_thresh = _BINN_CONFIG.binn_rel_save_thresh
binn_lr = _BINN_CONFIG.binn_lr
binnVF = _first(_BINN_CONFIG.binn_vfs)
binnDevice = _BINN_CONFIG.binn_device
D_one_param_bool = _BINN_CONFIG.done_param_bool
BNdataLossFuncLabel = _first(_BINN_CONFIG.bn_data_loss_labels)
BCbool = _BINN_CONFIG.bc_bool
binnGenerateIndicesLabel = _BINN_CONFIG.binn_generate_indices_label
binn_model_num = _first(_BINN_CONFIG.binn_model_labels)
