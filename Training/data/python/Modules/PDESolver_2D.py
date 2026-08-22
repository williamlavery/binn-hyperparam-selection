"""Two-dimensional PDE solver utilities for synthetic data generation.

Contents
--------
- Du_2d: finite-difference diffusion operator assembly in two dimensions.
- PDE_RHS_2D: right-hand side for the two-dimensional reaction-diffusion PDE.
- PDE_sim_2d: integrate on a dense 2D grid and interpolate the solution back to the requested grid.
"""

import numpy as np

from scipy import integrate
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from IPython.display import clear_output

import itertools
import scipy


def Du_2d(D_flat, dx, dy, nx, ny):
    """Assemble the sparse two-dimensional diffusion operator with no-flux edges."""
    size = nx * ny
    D = D_flat.reshape((nx, ny))

    inv_dx2 = 1 / (2 * dx ** 2)
    inv_dy2 = 1 / (2 * dy ** 2)

    D_left = np.roll(D, shift=1, axis=0)
    D_right = np.roll(D, shift=-1, axis=0)
    D_left[0, :] = D[1, :]
    D_right[-1, :] = D[-2, :]

    D_up = np.roll(D, shift=1, axis=1)
    D_down = np.roll(D, shift=-1, axis=1)
    D_up[:, 0] = D[:, 1]
    D_down[:, -1] = D[:, -2]

    data_x_left = (D_left + D) * inv_dx2
    data_x_center = -(2 * D + D_left + D_right) * inv_dx2
    data_x_right = (D + D_right) * inv_dx2

    data_y_up = (D_up + D) * inv_dy2
    data_y_center = -(2 * D + D_up + D_down) * inv_dy2
    data_y_down = (D + D_down) * inv_dy2

    data = np.dstack((data_x_left, data_x_center, data_x_right, data_y_up, data_y_center, data_y_down))
    data = np.swapaxes(data, 0, 1).flatten()

    indices = np.arange(nx * ny).reshape(nx, ny)

    row_x = np.repeat(indices, 3)
    row_y = np.repeat(indices, 3)
    row = np.dstack((row_x, row_y)).flatten()

    col_x_left = np.where(indices % nx > 0, indices - 1, indices + 1)
    col_x_right = np.where(indices % nx < nx - 1, indices + 1, indices - 1)
    col_y_upper = np.where(indices // nx > 0, indices - nx, indices + nx)
    col_y_lower = np.where(indices // nx < ny - 1, indices + nx, indices - nx)

    col = np.dstack((col_x_left, indices, col_x_right, col_y_upper, indices, col_y_lower)).flatten()

    return sparse.coo_matrix((data, (row, col)), shape=(size, size))


def PDE_RHS_2D(t, y, x1, x2, D, f):
    """Evaluate the semi-discrete 2D reaction-diffusion right-hand side."""
    dx1 = x1[1] - x1[0]
    dx2 = x2[1] - x2[0]
    nx1, nx2 = len(x1), len(x2)
    try:
        Du_mat = Du_2d(D(y), dx1, dx2, nx1, nx2)
        return Du_mat.dot(y) + y * f(y)
    except Exception as exc:
        raise RuntimeError(
            "Ensure D and f are correctly defined for the expected input combinations (y, t)."
        ) from exc


def PDE_sim_2d(
    RHS,
    IC_func,
    x1,
    x2,
    t,
    D,
    f,
    numtsim=1000,
    numxsim1=200,
    numxsim2=200,
    clear=True,
):
    """Simulate a 2D PDE on a dense grid and interpolate snapshots to the coarse grid."""
    def initialize_simulation(t, x1, x2, IC_func, numtsim, numxsim1, numxsim2):
        t_sim = np.linspace(np.min(t), np.max(t), numtsim)
        x1_sim = np.linspace(np.min(x1), np.max(x1), numxsim1)
        x2_sim = np.linspace(np.min(x2), np.max(x2), numxsim2)
        IC_dense = IC_func(x1_sim, x2_sim)
        y0 = IC_dense.flatten()
        return t_sim, x1_sim, x2_sim, y0

    def find_write_indices(t, t_sim):
        return np.array([np.abs(tp - t_sim).argmin() for tp in t])

    def integrate(t_sim, y0, RHS_func, write_indices, query_grid, solver_shape, IC_coarse, clear):
        y = np.zeros(solver_shape)
        y[..., 0] = IC_coarse

        r = scipy.integrate.ode(RHS_func)
        r.set_integrator("dopri5").set_initial_value(y0, t[0])

        write_count = 0
        for i in range(1, len(t_sim)):
            if i in write_indices:
                write_count += 1
                sol = r.integrate(t_sim[i]).reshape(query_grid[1])
                f_interpolate = RegularGridInterpolator(query_grid[0], sol)
                y[..., write_count] = f_interpolate(query_grid[2]).reshape(IC_coarse.shape)
                if clear:
                    clear_output()
            else:
                r.integrate(t_sim[i])
            if not r.successful():
                print("Integration failed")
                return 1e6 * np.ones_like(y)
        return y

    t_sim, x1_sim, x2_sim, y0 = initialize_simulation(t, x1, x2, IC_func, numtsim, numxsim1, numxsim2)
    write_indices = find_write_indices(t, t_sim)

    def RHS_2D(t_loc, y):
        return RHS(t_loc, y.flatten(), x1_sim, x2_sim, D, f)

    solver_shape = (len(x1), len(x2), len(t))
    query_grid = ((x1_sim, x2_sim), (numxsim1, numxsim2), np.array(list(itertools.product(x1, x2))))

    return integrate(t_sim, y0, RHS_2D, write_indices, query_grid, solver_shape, IC_func(x1, x2), clear)
