"""One-dimensional PDE solver utilities for synthetic data generation.

Contents
--------
- Du_1D: finite-difference diffusion operator assembly in one dimension.
- PDE_RHS_1D: right-hand side for the one-dimensional reaction-diffusion PDE.
- PDE_sim_1d: integrate on a dense 1D grid and interpolate the solution back to the requested grid.
"""

import numpy as np

from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from IPython.display import clear_output


import scipy.io as sio
import scipy.optimize

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset




def Du_1D(D_flat, dx, nx):
    """
    Function that generates the matrix finite difference expression for a 1D problem
    
    Parameters:
        D_flat: np.ndarray
            Flattened diffusion coefficient matrix of size nx.
        dx: float
            Step size in the x-direction.
        nx: int
            Number of grid points in the x-direction.
    
    Returns:
        sparse.coo_matrix:
            The finite difference matrix for the diffusion operator.
    """

    # Reshape D_flat into a 1D array (since ny = 1)
    D = D_flat.reshape((nx, 1))  # Reshape to maintain compatibility with 2D operations

    # Precompute constants
    inv_dx2 = 1 / (2 * dx ** 2)

    # X-direction neighbors (left and right)
    D_left = np.roll(D, shift=1, axis=0)  # Left neighbor (i-1)
    D_right = np.roll(D, shift=-1, axis=0)  # Right neighbor (i+1)
    
    # Handle boundary conditions in x-direction
    D_left[0, :] = D[1, :]  # Enforce BC at the left edge
    D_right[-1, :] = D[-2, :]  # Enforce BC at the right edge

    # X-direction contributions
    data_x_left = (D_left + D) * inv_dx2
    data_x_center = -(2 * D + D_left + D_right) * inv_dx2
    data_x_right = (D + D_right) * inv_dx2

    # Combine contributions
    data = np.hstack((data_x_left.flatten(), data_x_center.flatten(), data_x_right.flatten()))

    # Construct row and column indices
    indices = np.arange(nx)  # Only along x-direction
    row = np.tile(indices, 3)  # Replicate for left, center, right

    # Construct columns
    col_x_left = np.where(indices > 0, indices - 1, indices + 1)
    col_x_right = np.where(indices < nx - 1, indices + 1, indices - 1)

    col = np.hstack((col_x_left, indices, col_x_right))

    # Construct the sparse matrix
    size = nx  # Since ny=1, the total size is just nx
    
    return sparse.coo_matrix((data, (row, col)), shape=(size, size))


        

def PDE_RHS_1D(t,y,x,D,f):  
    
    dx = x[1] - x[0]
    nx = len(x) 


    try:
        # Case 4: D(y) and f(y) (least variable case)
        Du_mat = Du_1D(D(y), dx, nx)
        return Du_mat.dot(y) + y * f(y)
    except Exception as e4:
        raise RuntimeError(
            "WL - Ensure D and f are correctly defined "
            "for the expected input combinations (y, t)."
                    )



def PDE_sim_1d(
    RHS, IC_func, x, t, D, f,
    numtsim=1000, numxsim=200,
    clear=True
):
    """
    1D PDE simulator (legacy style) updated to mirror PDE_sim_2d:

      - Initial condition is given as a function IC_func(x).
      - Integration is done on a dense spatial grid x_sim, then
        interpolated back to the coarse grid x at requested times t.
    """

    # ─────────────────────────────────────────────────────────────
    # 1. Initialize dense grids and IC from IC_func(x_sim)
    # ─────────────────────────────────────────────────────────────
    def initialize_simulation(t, x, IC_func, numtsim, numxsim):
        t_sim = np.linspace(np.min(t), np.max(t), numtsim)
        x_sim = np.linspace(np.min(x), np.max(x), numxsim)

        # IC_func expects a 1D array x_sim
        IC_dense = IC_func(x_sim)         # shape (numxsim,)
        y0 = IC_dense.flatten()           # (numxsim,)

        return t_sim, x_sim, y0

    # ─────────────────────────────────────────────────────────────
    # 2. Find write indices
    # ─────────────────────────────────────────────────────────────
    def find_write_indices(t, t_sim):
        return np.array([np.abs(tp - t_sim).argmin() for tp in t])

    # ─────────────────────────────────────────────────────────────
    # 3. Integration routine (same progress printing style)
    # ─────────────────────────────────────────────────────────────
    def integrate(t_sim, y0, RHS_func, write_indices,
                  query_grid, solver_shape, IC_coarse, clear):

        y = np.zeros(solver_shape)
        y[..., 0] = IC_coarse   # IC on coarse grid (len(x),)

        r = scipy.integrate.ode(RHS_func)
        r.set_integrator("dopri5").set_initial_value(y0, t[0])

        write_count = 0

        for i in range(1, len(t_sim)):

            if i in write_indices:
                write_count += 1

                # Dense solution at this t
                sol = r.integrate(t_sim[i]).reshape(query_grid[1])  # (numxsim,)

                # Interpolate back to coarse x
                f_interpolate = RegularGridInterpolator(query_grid[0], sol)
                y[..., write_count] = f_interpolate(query_grid[2]).reshape(IC_coarse.shape)

                if clear:
                    clear_output()  # clear_output used for removing error messages
            else:
                r.integrate(t_sim[i])

            if not r.successful():
                print("Integration failed")
                return 1e6 * np.ones_like(y)

            # Progress printing (with flush)
            progress = (i + 1) / len(t_sim) * 100
            print(
                "\rProgress: {}% complete".format(round(progress, 2)),
                end="",
                flush=True,
            )

        print("\rProgress: 100.00% complete")
        return y

    # ─────────────────────────────────────────────────────────────
    # 4. Main driver logic
    # ─────────────────────────────────────────────────────────────
    t_sim, x_sim, y0 = initialize_simulation(
        t, x, IC_func, numtsim, numxsim
    )

    write_indices = find_write_indices(t, t_sim)

    def RHS_1D(t_curr, y_flat):
        # RHS must accept flattened y and dense spatial grid x_sim
        return RHS(t_curr, y_flat.flatten(), x_sim, D, f)

    # IC on coarse grid
    IC_coarse = IC_func(x)   # shape (len(x),)

    solver_shape = (len(x), len(t))

    # Dense → coarse interpolation grid (1D)
    query_grid = (
        (x_sim,),          # axes of dense solution
        (numxsim,),        # shape of dense solution
        x[:, None],        # coarse query points, shape (len(x),1)
    )

    return integrate(
        t_sim, y0, RHS_1D, write_indices,
        query_grid, solver_shape, IC_coarse, clear
    )


