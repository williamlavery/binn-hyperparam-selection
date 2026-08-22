"""Core utility helpers shared across the paper notebooks.

Contents
--------
- to_torch
- hist_properties
- symbolic_from_function
- scale_function_by_percent_error"""

import numpy as np
import torch
import sympy as sp
import ast
import inspect


def to_torch(ndarray, device):
    """
    Convert a NumPy array to a Torch tensor on the given device.

    Parameters
    ----------
    ndarray : numpy.ndarray
        Input array to convert.
    device : str or torch.device
        Target device (e.g. "cpu" or "cuda").

    Returns
    -------
    torch.Tensor
        Float tensor on the requested device with `requires_grad=True`.
    """
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr


def hist_properties(dataobj, num_bins_data_plot=100, low=5, high=95):
    """
    Compute histogram-based properties of the u-field in a data object.

    The function flattens `dataobj.u`, computes a histogram, and derives
    statistics such as bin centers, percentiles, and threshold counts.

    Parameters
    ----------
    dataobj : object
        Object with at least a `u` attribute (NumPy or Torch array-like).
    num_bins_data_plot : int, optional
        Number of histogram bins, by default 100.
    low : float, optional
        Lower percentile (0–100) to use for count/field thresholds, by default 5.
    high : float, optional
        Upper percentile (0–100), by default 95.

    Returns
    -------
    dict
        Dictionary with keys:
        - "hist": histogram counts (Torch tensor)
        - "bin_edges": histogram bin edges (Torch tensor)
        - "bin_indices": bin index for each flattened u value (Torch tensor)
        - "bin_centers": bin centers (Torch tensor)
        - "low_count_thresh": low-count threshold on histogram counts (float)
        - "low_count": low percentile of u values (float)
        - "high_count_thresh": high-count threshold on histogram counts (float)
        - "high_count": high percentile of u values (float)
    """
    u = dataobj.u
    u_flat = u.flatten()

    hist, bin_edges = torch.histogram(torch.tensor(u_flat), bins=num_bins_data_plot)

    low_count_thresh = np.percentile(hist.numpy(), low)
    high_count_thresh = np.percentile(hist.numpy(), high)
    bin_indices = torch.bucketize(torch.tensor(u_flat), bin_edges[1:-1])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {
        "hist": hist,
        "bin_edges": bin_edges,
        "bin_indices": bin_indices,
        "bin_centers": bin_centers,
        "low_count_thresh": low_count_thresh,
        "low_count": np.percentile(u, low),
        "high_count_thresh": high_count_thresh,
        "high_count": np.percentile(u, high),
    }


def symbolic_from_function(func, var_name="u"):
    """
    Build a SymPy expression from a Python function's return statement.

    The function introspects the source of `func`, locates the return
    expression, and re-evaluates it in a symbolic context where `var_name`
    is a SymPy symbol. This is useful for extracting a symbolic D(u)
    given a Python implementation.

    Parameters
    ----------
    func : callable
        Python function whose return expression should be made symbolic.
    var_name : str, optional
        Name of the symbolic variable to use, by default 'u'.

    Returns
    -------
    sympy.Expr
        SymPy expression representing the function's return value.
    """
    # Get source code
    source = inspect.getsource(func).strip()

    # Parse the function's AST
    tree = ast.parse(source)

    # Get the return statement's expression
    return_node = next(
        node for node in ast.walk(tree) if isinstance(node, ast.Return)
    )

    # Create a mapping for allowed names (e.g., math, np, etc.)
    allowed_names = func.__globals__.copy()

    # Create symbolic variable
    u = sp.Symbol(var_name)

    # Evaluate the return expression in symbolic context
    expr = eval(
        compile(ast.Expression(return_node.value), "<ast>", "eval"),
        {**allowed_names, var_name: u},
    )

    return expr


def scale_function_by_percent_error(func, beta, n_points=1001):
    """Return a percent-scaled callable plus MSE and mean absolute percentage error."""
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
