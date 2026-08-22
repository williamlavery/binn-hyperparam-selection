"""Dimension-agnostic data containers for 1D/2D BINN pipelines.

Contents
--------
- OriginalData: wrapper around the original experimental 1D dataset plus derived metadata.
- generate_inputs: build flattened model inputs from spatial and time grids.
- Data: unified serializable container for generated 1D or 2D datasets.
- plot_1d_data: plot 1D noisy and clean time slices for quick inspection.
- plot_2d_data: plot 2D heatmaps of `u(x1,x2,t)` at selected time points.
- reconstruct_u: rebuild an output grid from flattened inputs using the first and last coordinates.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


class OriginalData:
    """Original 1D dataset wrapper used for time-axis bootstrapping."""

    def __init__(self, data, plot: bool = False):
        self.plot = plot
        self.data = data

        self.inputs = self.data["inputs"]
        self.u = self.data["outputs"]
        self.u_clean = self.data["clean"]
        self.u0 = self.u[..., 0]

        self.x = np.unique(self.inputs[:, 0]).copy()
        self.x_num = len(self.x)
        self.t = np.unique(self.inputs[:, 1]).copy()
        self.t_num = len(self.t)

        self.xmin = np.min(self.x)
        self.xmax = np.max(self.x)
        self.L = self.xmax - self.xmin
        self.nx = len(self.x)

        self.D = self.data["D"]
        self.r = self.data["r"]
        self.gamma = 0.2
        self.K = self.data["K"]

        self.class_info = {
            "original?": 1,
            "DValue": self.D,
            "rValue": self.r,
            "gamma": self.gamma,
            "K": self.K,
            "xNum": self.x_num,
            "tNum": self.t_num,
        }

        if self.plot:
            plot_1d_data(self.x, self.t, self.u, self.u_clean)


def generate_inputs(x1, x2, t, dim: int | None = None):
    """Generate flattened input coordinates for 1D or 2D data."""
    if dim is None:
        dim = 1 if x2 is None else 2

    if dim == 1:
        nx = len(x1)
        nt = len(t)
        return np.array([np.repeat(x1, nt), np.tile(t, nx)]).T

    if x2 is None:
        raise ValueError("x2 is required when dim=2")

    x1_grid, x2_grid, t_grid = np.meshgrid(x1, x2, t, indexing="ij")
    return np.stack([x1_grid, x2_grid, t_grid], axis=-1).reshape(-1, 3)


class Data:
    """Unified Data container for 1D and 2D pipelines."""

    def __init__(
        self,
        x1,
        t,
        u_clean,
        u,
        theta_D,
        theta_G,
        *,
        x2=None,
        K=1,
        gamma=0.2,
        plot=False,
    ):
        self.plot = plot
        self.u = u
        self.u_clean = u_clean
        self.u0 = self.u[..., 0]

        self.x1 = np.asarray(x1)
        self.x2 = None if x2 is None else np.asarray(x2)
        self.t = np.asarray(t)

        self.dim = 1 if self.x2 is None else 2
        self.inputs = generate_inputs(self.x1, self.x2, self.t, dim=self.dim)

        if self.dim == 1:
            self.x = self.x1
            self.xmin = np.min(self.x1)
            self.xmax = np.max(self.x1)
            self.L = self.xmax - self.xmin
            self.nx = len(self.x1)
        else:
            self.x = None
            self.xmin = np.min(self.x1)
            self.xmax = np.max(self.x1)
            self.ymin = np.min(self.x2)
            self.ymax = np.max(self.x2)
            self.Lx = self.xmax - self.xmin
            self.Ly = self.ymax - self.ymin
            self.nx = len(self.x1)
            self.ny = len(self.x2)

        self.theta_D = theta_D
        self.theta_G = theta_G
        self.gamma = gamma
        self.K = K

        self.ClassInfo = {
            "dim": self.dim,
            "xNum": len(self.x1),
            "tNum": len(self.t),
            "gamma": gamma,
            "K": K,
        }
        if self.dim == 2:
            self.ClassInfo["yNum"] = len(self.x2)

        if self.plot:
            if self.dim == 1:
                plot_1d_data(self.x1, self.t, self.u, self.u_clean)
            else:
                plot_2d_data(self.x1, self.x2, self.t, self.u, self.u_clean)


def plot_1d_data(x, t, u, u_clean):
    """Plot 1D time slices of noisy and clean data."""
    plt.figure(figsize=(8, 4))
    for tidx in range(len(t)):
        plt.plot(x, u[:, tidx], label="Noisy")
        plt.xlabel("x [mm]")
        plt.ylabel("u [cells/mm^2]")
        plt.title(f"Cell Density at t={t[tidx]:.2f} s")
    for tidx in range(len(t)):
        plt.plot(
            x,
            u_clean[:, tidx],
            label=f"Clean t={t[tidx]:.2f} s",
            linestyle="--",
        )
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_2d_data(x1, x2, t, u, u_clean, times_to_plot=None):
    """Plot 2D heatmaps of u(x1,x2,t) at selected time points."""
    nx, ny, nt = u.shape
    if times_to_plot is None:
        times_to_plot = [0, nt // 2, nt - 1] if nt >= 3 else list(range(nt))

    n_plots = len(times_to_plot)
    fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 8), squeeze=False)
    for col, tidx in enumerate(times_to_plot):
        tt = t[tidx]

        ax = axes[0, col]
        im = ax.imshow(
            u[:, :, tidx].T,
            origin="lower",
            extent=[x1[0], x1[-1], x2[0], x2[-1]],
            aspect="auto",
        )
        ax.set_title(f"Noisy u(x1,x2,t) at t={tt:.2f}")
        ax.set_xlabel("x1 [mm]")
        ax.set_ylabel("x2 [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, col]
        im = ax.imshow(
            u_clean[:, :, tidx].T,
            origin="lower",
            extent=[x1[0], x1[-1], x2[0], x2[-1]],
            aspect="auto",
        )
        ax.set_title(f"Clean u(x1,x2,t) at t={tt:.2f}")
        ax.set_xlabel("x1 [mm]")
        ax.set_ylabel("x2 [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def reconstruct_u(inputs, outputs):
    """Reconstruct a 1D u(x,t) grid from flattened inputs/outputs."""
    inputs = np.asarray(inputs)
    outputs = np.asarray(outputs).flatten()

    unique_x = np.unique(inputs[:, 0])
    unique_t = np.unique(inputs[:, -1])
    u_recon = np.empty((len(unique_x), len(unique_t)))

    x_index = {x: i for i, x in enumerate(unique_x)}
    t_index = {t_val: j for j, t_val in enumerate(unique_t)}

    for (x_val, t_val), u_val in zip(inputs[:, [0, -1]], outputs):
        i = x_index[x_val]
        j = t_index[t_val]
        u_recon[i, j] = u_val

    return u_recon, unique_x, unique_t

    x_vals = np.unique(inputs[:, 0])
    t_vals = np.unique(inputs[:, 1])

    x_to_idx = {x: i for i, x in enumerate(x_vals)}
    t_to_idx = {t: i for i, t in enumerate(t_vals)}

    u_recon = np.empty((len(x_vals), len(t_vals)))
    u_recon[:] = np.nan  # optional: fill with NaN in case some combinations are missing

    for (x, t), u in zip(inputs, outputs):
        i = x_to_idx[x]
        j = t_to_idx[t]
        u_recon[i, j] = u

    return u_recon
#===================================================================================================
#===================================================================================================   
#===================================================================================================            
