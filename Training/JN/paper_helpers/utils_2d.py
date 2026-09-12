"""Small shared utilities for two-dimensional paper helper plotting.

Contents
--------
- sort_outer_dict
- hist_properties"""

import numpy as np
import torch

from .utils import metric_u_grid, scale_function_by_percent_error


def sort_outer_dict(d, descending=False, drop_keys=None):
    """Return a dictionary sorted by outer keys, optionally dropping keys first."""
    drop_keys = set(drop_keys or [])
    filtered = {key: value for key, value in d.items() if key not in drop_keys}
    return {key: filtered[key] for key in sorted(filtered, reverse=descending)}


def hist_properties(dataobj, num_bins_data_plot=100, low=5, high=95):
    """Compute histogram and percentile summaries for a data object's u-field."""
    u = dataobj.u
    u_flat = u.flatten()

    hist, bin_edges = torch.histogram(torch.tensor(u_flat), bins=num_bins_data_plot)
    bin_indices = torch.bucketize(torch.tensor(u_flat), bin_edges[1:-1])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {
        "hist": hist,
        "bin_edges": bin_edges,
        "bin_indices": bin_indices,
        "bin_centers": bin_centers,
        "low_count_thresh": np.percentile(hist.numpy(), low),
        "low_count": np.percentile(u, low),
        "high_count_thresh": np.percentile(hist.numpy(), high),
        "high_count": np.percentile(u, high),
    }


__all__ = [
    "hist_properties",
    "metric_u_grid",
    "scale_function_by_percent_error",
    "sort_outer_dict",
]
