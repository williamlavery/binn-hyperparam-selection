"""Small shared utilities for two-dimensional paper helper plotting.

Contents
--------
- sort_outer_dict
- scale_function_by_percent_error
- hist_properties"""

import numpy as np
import torch


def sort_outer_dict(d, descending=False, drop_keys=None):
    """Return a dictionary sorted by outer keys, optionally dropping keys first."""
    drop_keys = set(drop_keys or [])
    filtered = {key: value for key, value in d.items() if key not in drop_keys}
    return {key: filtered[key] for key in sorted(filtered, reverse=descending)}


def scale_function_by_percent_error(func, beta, n_points=1001):
    """Return a scaled callable with target percentage offset ``beta``.

    The helper also returns the mean squared error and mean absolute percentage
    error over an evenly spaced grid on ``[0, 1]``.
    """
    scale = 1.0 + beta / 100.0

    def scaled_func(u):
        return scale * func(u)

    u = np.linspace(0.0, 1.0, n_points)
    func_vals = np.array([func(ui) for ui in u])
    scaled_vals = scale * func_vals
    mse = np.mean((scaled_vals - func_vals) ** 2)

    mask = func_vals != 0
    ape = (
        np.mean(np.abs((scaled_vals[mask] - func_vals[mask]) / func_vals[mask])) * 100.0
        if np.any(mask)
        else np.nan
    )

    return scaled_func, mse, ape


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


__all__ = ["hist_properties", "scale_function_by_percent_error", "sort_outer_dict"]
