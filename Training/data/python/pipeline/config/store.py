"""Stored functional forms for synthetic diffusion, growth, and initial data.

Contents
--------
- diffusion_func1: constant diffusion law used for the Case I synthetic PDE.
- diffusion_func1_du: derivative of the constant diffusion law with respect to density.
- diffusion_func2: linear diffusion law used for Case II synthetic data.
- diffusion_func2_du: derivative of the linear diffusion law with respect to density.
- diffusion_func3: quadratic diffusion law used for Case II synthetic data.
- diffusion_func3_du: derivative of the quadratic diffusion law with respect to density.
- diffusion_func4: exponential-saturation diffusion law used for Case II synthetic data.
- diffusion_func4_du: derivative of the exponential diffusion law with respect to density.
- growth_func1: constant growth law, also reused for the zero-growth configuration.
- growth_func2: linear growth law for the synthetic PDE right-hand side.
- growth_func3: quadratic growth law for the synthetic PDE right-hand side.
- growth_func4: exponential growth law for the synthetic PDE right-hand side.
- ic1: cosine initial-condition profile on a one-dimensional grid.
- ic1_flipped: phase-flipped cosine initial-condition profile on a one-dimensional grid.
- scratch: separable cosine initial-condition surface on a two-dimensional grid.
- resample_xy: interpolate a two-dimensional field onto a new grid.

Units and conventions
---------------------
Diffusion is evaluated on a spatial grid in mm and a time grid in days, so the
diffusion laws return values in mm^2 day^-1 and the growth laws return per-capita
rates in day^-1. The density argument ``u`` is the normalised (dimensionless)
density produced by the initial-condition constructors (amplitude O(1)).

IMPORTANT -- the growth functions carry a factor of 1/2 that the diffusion
functions do NOT. Every ``growth_func*`` below divides the raw parameters
``theta`` by 2, so the parameters stored in ``GROWTH_PARAMETERS`` (in
``pipeline/components/data__initialise.py``) are TWICE the effective
coefficients. The effective (post-1/2) growth laws are the ones reported in the
paper's ground-truth table (Table 1, ``tab:expressions``); the raw ``theta`` in
the code are double those table values. Concretely, with the stored ``theta``:

    G1 (const)      theta=[1.3]           -> theta0/2                = 0.65
    G2 (linear)     theta=[2.4, -3]       -> (theta0 + theta1 u)/2   = 1.2 - 1.5 u
    G3 (quadratic)  theta=[2.1, -0.29]    -> (theta0 + theta1 u^2)/2 = 1.05 - 0.145 u^2
    G4 (exp)        theta=[0.7, -1.3, 4]  -> (theta0 + theta1(1 - e^{-theta2 u}))/2
                                          = 0.35 - 0.65(1 - e^{-4 u})

The exponential rate ``theta[2]`` (e.g. 4 for G4) sits inside the exponent and is
NOT halved, matching g_{4,3}=4 in the table. The diffusion functions have no such
factor: their stored parameters equal the table values directly.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def diffusion_func1(u, theta):
    return np.full_like(u, theta[0])


def diffusion_func1_du(u, theta):
    return np.zeros_like(u)


def diffusion_func2(u, theta):
    return theta[0] + theta[1] * u


def diffusion_func2_du(u, theta):
    return np.full_like(u, theta[1])


def diffusion_func3(u, theta):
    return theta[0] + theta[1] * u**2


def diffusion_func3_du(u, theta):
    return 2 * theta[1] * u


def diffusion_func4(u, theta):
    return theta[0] + theta[1] * (1 - np.exp(-theta[2] * u))


def diffusion_func4_du(u, theta):
    return theta[1] * theta[2] * np.exp(-theta[2] * u)


# NOTE: every growth_func* divides by 2 (see module docstring). The `theta`
# passed in from GROWTH_PARAMETERS are therefore 2x the effective coefficients
# reported in the paper's Table 1. Diffusion functions do NOT have this factor.


def growth_func1(u, theta):
    # G1 = theta0 / 2  (e.g. theta0=1.3 -> effective 0.65 day^-1)
    return np.full_like(u, theta[0]) / 2


def growth_func2(u, theta):
    # G2 = (theta0 + theta1 u) / 2  (e.g. [2.4, -3] -> 1.2 - 1.5 u)
    return (theta[0] + theta[1] * u) / 2


def growth_func3(u, theta):
    # G3 = (theta0 + theta1 u^2) / 2  (e.g. [2.1, -0.29] -> 1.05 - 0.145 u^2)
    return (theta[0] + theta[1] * u**2) / 2


def growth_func4(u, theta):
    # G4 = (theta0 + theta1 (1 - e^{-theta2 u})) / 2
    #      (e.g. [0.7, -1.3, 4] -> 0.35 - 0.65 (1 - e^{-4 u}); note theta2 is
    #       inside the exponent and is NOT halved)
    return (theta[0] + theta[1] * (1 - np.exp(-theta[2] * u))) / 2


def ic1(x, y=None, amplitude=1.0):
    """Return a cosine-squared bump for 1D or 2D grids."""
    x = np.asarray(x)
    if y is None:
        mean_x = 0.5 * (x.min() + x.max())
        length = x.max() - x.min()
        if length == 0:
            return np.zeros_like(x)
        return amplitude / 2 * (1 - np.cos(2 * np.pi * (x - mean_x) / length))

    y = np.asarray(y)
    xc = 0.5 * (x.min() + x.max())
    yc = 0.5 * (y.min() + y.max())
    lx = x.max() - x.min()
    ly = y.max() - y.min()
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    cos_x = np.cos(np.pi * (x_grid - xc) / lx)
    cos_y = np.cos(np.pi * (y_grid - yc) / ly)
    return amplitude/2 * (cos_x**2) * (cos_y**2)


def ic1_flipped(x, y=None, amplitude=1.0):
    """Return the complement of the standard cosine initial condition."""
    return amplitude - ic1(x, y, amplitude=amplitude)


def scratch(x, y=None):
    """Simple scratch initial condition for 1D or 2D."""
    x = np.asarray(x)
    if y is None:
        mid = 0.5 * (x.min() + x.max())
        return (x <= mid).astype(float)

    y = np.asarray(y)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    mid = 0.5 * (x.min() + x.max())
    return (x_grid <= mid).astype(float)


def resample_xy(u, x, y):
    nx_orig, ny_orig = u.shape[:2]
    x_orig = np.linspace(0, 1, nx_orig)
    y_orig = np.linspace(0, 1, ny_orig)
    x_new = np.linspace(0, 1, len(x))
    y_new = np.linspace(0, 1, len(y))
    interpolator = RegularGridInterpolator(
        (x_orig, y_orig),
        u,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    x_grid, y_grid = np.meshgrid(x_new, y_new, indexing="ij")
    points = np.stack([x_grid, y_grid], axis=-1)
    return interpolator(points)
