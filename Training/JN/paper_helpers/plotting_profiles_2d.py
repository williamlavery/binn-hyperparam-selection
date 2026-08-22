"""Two-dimensional profile plotting helpers.

Contents
--------
- plot_initial_condition_2d
- _print_initial_condition_error"""

import numpy as np
import matplotlib.pyplot as plt


def plot_initial_condition_2d(
    dataobj,
    tidx=0,
    x1_idx=None,
    x2_idx=None,
    filename=None,
    K_orig=1.7e3,
    tol=1e-6,
):
    """Plot a 2D field and two central slices at a selected time index."""
    u = dataobj.u
    x1 = dataobj.x1
    x2 = dataobj.x2

    nx1, nx2, nt = u.shape
    if not 0 <= tidx < nt:
        raise ValueError(f"tidx={tidx} out of range for Nt={nt}")

    if x1_idx is None:
        x1_idx = nx1 // 2
    if x2_idx is None:
        x2_idx = nx2 // 2

    u_slice_2d = u[:, :, tidx] * K_orig
    u_slice_x1 = u[x1_idx, :, tidx] * K_orig
    u_slice_x2 = u[:, x2_idx, tidx] * K_orig

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    extent = [x1.min(), x1.max(), x2.min(), x2.max()]
    image = axes[0].imshow(
        u_slice_2d.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        cmap="viridis",
    )
    axes[0].set_title(f"Heat map at t = {tidx}/{nt - 1} T")
    axes[0].set_xlabel("x1 [mm]")
    axes[0].set_ylabel("x2 [mm]")
    fig.colorbar(image, ax=axes[0], label=r"cell density [cells mm$^{-2}$]")

    axes[1].plot(x2, u_slice_x1, color="black")
    axes[1].set_title(f"Slice at x1 = {x1[x1_idx]:.3f}")
    axes[1].set_xlabel("x2 [mm]")
    axes[1].set_ylabel(r"cell density [cells mm$^{-2}$]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x1, u_slice_x2, color="black")
    axes[2].set_title(f"Slice at x2 = {x2[x2_idx]:.3f}")
    axes[2].set_xlabel("x1 [mm]")
    axes[2].set_ylabel(r"cell density [cells mm$^{-2}$]")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(f"{filename}.png", dpi=120, bbox_inches="tight", facecolor="None")

    if hasattr(dataobj, "u_clean"):
        _print_initial_condition_error(u, dataobj.u_clean, K_orig=K_orig, tol=tol)

    plt.show()


def _print_initial_condition_error(u, u_clean, K_orig, tol):
    """Print absolute and relative errors between noisy and clean fields."""
    err = np.abs(u - u_clean)
    mse_val = np.mean(err**2)
    abs_val = np.mean(err)

    print("MSE between u and u_clean:", K_orig * mse_val)
    print("ABS between u and u_clean [cells]:", K_orig * abs_val)

    u_ref_abs = np.abs(u_clean)
    u_max = float(np.max(u_ref_abs)) if u_ref_abs.size else 0.0
    u_min = max(tol, 1e-6 * max(1.0, u_max))
    mask = u_ref_abs >= u_min

    if np.any(mask):
        rel_err_mean = np.mean(err[mask] / u_ref_abs[mask])
        print(
            f"ABS (%) between u and u_clean (on |u_clean| >= {u_min:.3e}):",
            100 * rel_err_mean,
        )
    else:
        print(
            "ABS (%) between u and u_clean: not defined; all |u_clean| below threshold"
        )


__all__ = ["plot_initial_condition_2d"]
