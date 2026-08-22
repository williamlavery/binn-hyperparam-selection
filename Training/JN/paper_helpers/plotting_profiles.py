"""Profile and density-error plotting helpers for one-dimensional notebooks.

Contents
--------
- plot_initial_condition_1d
- plot_eval2
- plot_repeats_u
- plot_repeats_D
- plot_eval_D_multi
- plot_repeats_times_u
- plot_repeats_times_u_control
- plot_eval2
- plot_binned_density_error_agg_groups_multi_data
- plot_density_histograms
- plot_density_histograms
- plot_binned_clean_vs_noisy_density_error
- plot_binned_density_diffusion_error_agg_groups_multi_data
- plot_error_space_time_agg_groups_multi_data
- plot_binned_density_error_agg_groups_multi_data_sum
- plot_binned_density_error_agg_groups_multi_data_median
- plot_binned_density_error_agg_groups_multi_data_mean"""

import numpy as np
import torch, sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import itertools

from .utils import to_torch, hist_properties


def plot_initial_condition_1d(dataobj, filename=None, K_orig=1.7e3):
    """
    Plot 1D initial-condition profiles for u(x, t).

    The function plots u(x, t) at all stored time steps using a grayscale
    marker scheme, and optionally saves the figure as a transparent PNG.
    It also reports several error metrics vs. a clean reference.

    Parameters
    ----------
    dataobj : object
        Data container with attributes:
        - u : noisy u(x, t)
        - u_clean : clean reference u(x, t)
        - x : spatial grid
        - t : time grid
    filename : str, optional
        Base filename for saving (without extension). If None, no file is saved.
    K_orig : float, optional
        Scaling factor applied to u-values before plotting and error reporting.

    Returns
    -------
    None
    """
    u = dataobj.u
    x = dataobj.x
    t = dataobj.t

    markers = ["o", "s", "D", "^", "p", "<", ">", "p", "*", "h", "x", "+"]
    grayscale_colors = ["#111111", "#444444", "#888888", "#BBBBBB", "#EEEEEE"]

    t_labels = ["0"] + [
        f"{int(i)}/{u.shape[-1]-1} T" for i in range(1, u.shape[-1] - 1)
    ] + ["T"]
    fig, ax = plt.subplots(figsize=(7, 5))

    for tidx in range(u.shape[-1]):
        color = grayscale_colors[tidx % len(grayscale_colors)]
        marker = markers[tidx % len(markers)]
        ax.plot(
            x,
            u[:, tidx] * K_orig,
            label=f"t = {t_labels[tidx]}",
            markersize=4,
            color=color,
            marker=marker,
            lw=0,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

    ax.set_xlabel("x [mm]")
    ax.set_ylabel(r"cell density [cells mm$^{-2}]$")
    ax.set_facecolor("white")
    ax.legend(fontsize=9)
    plt.tight_layout()

    if filename:
        plt.savefig(f"{filename}.png", dpi=100, bbox_inches="tight", facecolor="None")

    err = np.abs(u - dataobj.u_clean)

    print("MSE between u and u_clean:", K_orig * np.mean(err**2))
    print("ABS between u and u_clean [cells]:", K_orig * np.mean(err))
    print("ABS (%) between u and u_clean:", 100 * np.mean(err / dataobj.u_clean))
    plt.show()


def plot_eval2(
    modelWrappers_dic,
    dataobj,
    IC=0,
    device="cpu",
    num_bins=10,
    label="",
    clean=True,
    noisy=True,
    error=True,
    K=1700,
    save_name=None,
    colors=None,
):
    """
    Plot histogram of u-values (train vs. validation) and prediction vs. truth.

    First, a stacked bar histogram of train and validation u values is shown.
    Then, for each model wrapper, the predicted u(x, t) at t=0 and t=T is
    compared against both clean and noisy data, with optional error curves.

    Parameters
    ----------
    modelWrappers_dic : dict
        Dictionary {seed: modelWrapper} (or similar) for a single IC.
    dataobj : object
        Data container with attributes x, u, u_clean, t, inputs.
    IC : int, optional
        Index of the initial condition (for labeling only), by default 0.
    device : str, optional
        Device used when running the model, by default "cpu".
    num_bins : int, optional
        Number of bins in the u-histogram, by default 10.
    label : str, optional
        Base label for this dataset, used in titles, by default ''.
    clean : bool, optional
        If True, plot clean curves, by default True.
    noisy : bool, optional
        If True, plot noisy curves, by default True.
    error : bool, optional
        If True, plot absolute error on a second axis, by default True.
    K : float, optional
        Scaling factor applied to u-values, by default 1700.
    save_name : str, optional
        Filename for saving the prediction plot (PNG). If None, no file is saved.
    colors : list, optional
        Custom color list. If None, a default purple palette is used.

    Returns
    -------
    None
    """
    if not colors:
        colors = ["#9013FE", "#5E239D", "#B580FF", "#E5D5FF"]
    markers = ["o", "s", "D", "^", "p", "<", ">", "p", "*", "h", "x", "+"]
    grayscale_colors = ["#111111", "#444444", "#888888", "#BBBBBB", "#EEEEEE"]

    # All models for same IC will have same data distribution -> use first model
    modelWrapper = list(modelWrappers_dic.values())[0]
    u_val = np.array([k for (_, _), k in zip(modelWrapper.x_val, modelWrapper.y_val)])
    u_train = np.array(
        [k for (_, _), k in zip(modelWrapper.x_train, modelWrapper.y_train)]
    )

    # Get data
    x = dataobj.x
    u = dataobj.u * K
    u_clean = dataobj.u_clean * K
    t = dataobj.t
    Nt, Nx = len(dataobj.t), len(dataobj.x)

    # Histogram properties
    model = modelWrapper.model
    model.eval()
    h_properties = hist_properties(dataobj, num_bins)

    hist = h_properties["hist"]
    bin_edges = h_properties["bin_edges"] * K
    bin_centers = h_properties["bin_centers"] * K

    # Compute separate histograms for train/val
    hist_train, _ = np.histogram(u_train * K, bins=bin_edges)
    hist_val, _ = np.histogram(u_val * K, bins=bin_edges)

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_width = bin_edges[1] - bin_edges[0]

    ax.bar(
        bin_centers,
        hist_val,
        width=bar_width,
        alpha=0.7,
        label="Validation",
        color=colors[1],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        bin_centers,
        hist_train,
        width=bar_width,
        alpha=0.7,
        bottom=hist_val,
        label="Train",
        color=colors[0],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel(r"cell density [cells mm$^{-2}]$", fontsize=11)
    ax.set_ylabel("frequency", fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Prediction vs. truth at t=0 and t=T
    # ------------------------------------------------------------------
    for modelWrapper_indx in modelWrappers_dic.keys():
        modelWrapper = modelWrappers_dic[modelWrapper_indx]

        x_val = modelWrapper.x_val
        y_val = modelWrapper.y_val
        x_train = modelWrapper.x_train
        y_train = modelWrapper.y_train

        model = modelWrapper.model

        with torch.no_grad():
            u_pred_flat = model(to_torch(dataobj.inputs, device)) * K
            u_pred = u_pred_flat.reshape(Nx, Nt).cpu().numpy()

        fig, ax1 = plt.subplots(figsize=(7, 5))

        ax1.plot(
            x,
            u_pred[:, 0],
            "-",
            lw=2,
            alpha=0.7,
            color=colors[0],
            label=r"$\hat{u}_{dn}(0,x)$",
        )
        ax1.plot(
            x,
            u_pred[:, -1],
            "-",
            lw=2,
            alpha=0.7,
            color=colors[1],
            label=r"$\hat{u}_{dn}(T,x)$",
        )

        if error:
            ax2 = ax1.twinx()
            ax2.set_ylabel(r"abs. error [cells mm$^{-2}]$", fontsize=11)
            ax2.set_yscale("log")

        abs_e = np.abs(u_pred - u_clean)
        mse = np.mean(abs_e)

        # Clean reference
        if clean:
            ax1.plot(
                x,
                u_clean[:, 0],
                label="t=0",
                markersize=4,
                color=grayscale_colors[0],
                marker=markers[0],
                lw=0,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
            ax1.plot(
                x,
                u_clean[:, -1],
                label="t=T",
                markersize=4,
                color=grayscale_colors[4],
                marker=markers[4],
                lw=0,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )

        if noisy:
            ax1.plot(x, u[:, 0], label="t=0", color=grayscale_colors[0], lw=1)
            ax1.plot(x, u[:, -1], label="t=T", color=grayscale_colors[3], lw=1)

        if error:
            ax2.plot(
                x,
                abs_e[:, 0],
                label="err(x,0)",
                markersize=4,
                color="#8b0000",
                marker=markers[0],
                markeredgecolor="black",
                markeredgewidth=0.5,
                alpha=0.7,
                lw=1,
            )
            ax2.plot(
                x,
                abs_e[:, -1],
                label="err(x,T)",
                markersize=4,
                color="#dd0505",
                marker=markers[4],
                markeredgecolor="black",
                markeredgewidth=0.5,
                alpha=0.7,
                lw=1,
            )

        ax1.set_xlabel("x [mm]", fontsize=11)
        ax1.set_ylabel(r"cell density [cells mm$^{-2}]$", fontsize=11)

        lines1, labels1 = ax1.get_legend_handles_labels()
        if error:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="lower right",
                fontsize=8,
                ncol=1,
            )
        else:
            ax1.legend(lines1, labels1, loc="lower right", fontsize=8, ncol=1)

        ax1.set_facecolor("white")
        plt.tight_layout()

        if save_name:
            plt.savefig(save_name, dpi=100, bbox_inches="tight", facecolor="None")


def plot_repeats_u(
    binn_models_dics,
    base_colors,
    filename=None,
    bbox_to_anchor=(1.02, 1.0),
    legend_fontsize=10,
):
    """
    Plot best validation loss for all repeats across widths and depths (u-models).

    Bars are grouped by width and hatched by depth; each repeat is overlaid
    on top of the same x-position with a tiny label (R1, R2, ...).

    Parameters
    ----------
    binn_models_dics : dict
        Nested dict {depth: {width: {seed: modelWrapper}}}.
    base_colors : list
        Base color for each width.
    filename : str, optional
        Output filename (PNG). If None, the figure is not saved.
    bbox_to_anchor : tuple, optional
        Legend anchor position, by default (1.02, 1.0).
    legend_fontsize : int, optional
        Fontsize for legend labels, by default 10.

    Returns
    -------
    None
    """
    hatch_styles = ["", "//", ".."]

    depths = list(binn_models_dics.keys())
    widths = list(binn_models_dics[depths[0]].keys())

    def collect_metric(binn_models, metric_attr):
        """
        Return a dict {(width, depth): [metric_per_repeat, ...]}.

        Parameters
        ----------
        binn_models : dict
            Nested dict {depth: {width: {seed: modelWrapper}}}.
        metric_attr : str
            Name of attribute to read from each modelWrapper.

        Returns
        -------
        dict
            Mapping (width, depth) to list of metric values over repeats.
        """
        out = {}
        for d in depths:
            for w in widths:
                repeats = binn_models[d][w]
                out[(w, d)] = [getattr(m, metric_attr) for m in repeats.values()]
        return out

    data = collect_metric(binn_models_dics, "best_val_loss")

    bar_width = 0.15
    intra_gap, inter_gap = 0.02, 0.30
    current_x = 0.0

    max_repeats = max(len(v) for v in data.values())
    label_cmap = cm.get_cmap("tab10", max_repeats)

    xtick_positions = []
    xtick_labels = []

    plt.figure(figsize=(7, 5))

    for (w, base_c) in zip(widths, base_colors):
        group_x_positions = []

        for i_d, d in enumerate(depths):
            vals = data[(w, d)]

            for i_r, v in enumerate(vals):
                alpha = 0.9 - 0.15 * i_r
                plt.bar(
                    current_x,
                    v,
                    width=bar_width,
                    color=base_c,
                    alpha=max(alpha, 0.25),
                    hatch=hatch_styles[i_d],
                    edgecolor="k",
                )

                plt.text(
                    current_x,
                    v,
                    f"R{i_r + 1}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="black",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        pad=0.3,
                        alpha=0.9,
                    ),
                )

            group_x_positions.append(current_x)
            current_x += bar_width + (
                intra_gap if i_d < len(depths) - 1 else inter_gap
            )

        center_x = np.mean(group_x_positions)
        xtick_positions.append(center_x)
        xtick_labels.append(f"{w}")

    plt.xticks(xtick_positions, xtick_labels)
    plt.yscale("log")
    plt.ylabel(r"Validation loss [a.u.]")

    depth_patches = [
        Patch(
            facecolor="white",
            edgecolor="k",
            hatch=hatch_styles[i],
            label=rf"NN$_\mathrm{{D}}$ = {d}",
        )
        for i, d in enumerate(depths)
    ]
    plt.legend(
        handles=depth_patches,
        loc="upper left",
        bbox_to_anchor=bbox_to_anchor,
        fontsize=legend_fontsize,
    )

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if filename:
        plt.savefig(f"{filename}", dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()


def plot_repeats_D(
    binn_models_dics,
    base_colors,
    filename=None,
    bbox_to_anchor=(1.02, 1.0),
    legend_fontsize=10,
):
    """
    Plot best diffusion error for all repeats across widths and depths (D-models).

    Bars are grouped by width and hatched by depth; each repeat is shown as a bar.

    Parameters
    ----------
    binn_models_dics : dict
        Nested dict {depth: {width: {seed: modelWrapper}}}.
    base_colors : list
        Base color for each width.
    filename : str, optional
        Output filename (PNG). If None, the figure is not saved.
    bbox_to_anchor : tuple, optional
        Legend position, by default (1.02, 1.0).
    legend_fontsize : int, optional
        Font size for legend text.

    Returns
    -------
    None
    """
    depths = list(binn_models_dics.keys())
    widths = list(binn_models_dics[depths[0]].keys())

    hatch_styles = ["", "//", ".."]

    def collect_metric(binn_models, metric_attr):
        """
        Return a dict {(width, depth): [metric_per_repeat, ...]}.

        See `plot_repeats_u` for structure.
        """
        out = {}
        for d in depths:
            for w in widths:
                repeats = binn_models[d][w]
                out[(w, d)] = [getattr(m, metric_attr) for m in repeats.values()]
        return out

    data = collect_metric(binn_models_dics, "best_diffusion_error")

    bar_width = 0.15
    intra_gap, inter_gap = 0.02, 0.30
    current_x = 0.0

    xtick_positions = []
    xtick_labels = []

    plt.figure(figsize=(7, 5))

    for (w, base_c) in zip(widths, base_colors):
        group_x_positions = []

        for i_d, d in enumerate(depths):
            vals = data[(w, d)]

            for i_r, v in enumerate(vals):
                alpha = 0.9 - 0.15 * i_r
                plt.bar(
                    current_x,
                    v,
                    width=bar_width,
                    color=base_c,
                    alpha=max(alpha, 0.25),
                    hatch=hatch_styles[i_d],
                    edgecolor="k",
                )

                plt.text(
                    current_x,
                    v,
                    f"R{i_r + 1}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="black",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        pad=0.3,
                        alpha=0.9,
                    ),
                )

            group_x_positions.append(current_x)
            current_x += bar_width + (
                intra_gap if i_d < len(depths) - 1 else inter_gap
            )

        center_x = np.mean(group_x_positions)
        xtick_positions.append(center_x)
        xtick_labels.append(f"{w}")

    plt.xticks(xtick_positions, xtick_labels)
    plt.yscale("log")
    plt.ylabel(r"Diffusion MSE [days$^{-2}$ mm$^{4}$]")

    depth_patches = [
        Patch(
            facecolor="white",
            edgecolor="k",
            hatch=hatch_styles[i],
            label=rf"NN$_\mathrm{{D}}$ = {d}",
        )
        for i, d in enumerate(depths)
    ]

    plt.legend(
        handles=depth_patches,
        loc="upper left",
        bbox_to_anchor=bbox_to_anchor,
        fontsize=legend_fontsize,
    )

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if filename:
        plt.savefig(f"{filename}.png", dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()


def plot_eval_D_multi(
    modelWrapper_dics,
    dataobj,
    D_sym_true_lst,
    colors,
    labels,
    Dnum=1,
    device="cpu",
    num_bins=50,
    K=1700,
    name=None,
    fill=True,
    legend_pos=(0.5, 0.5),
    linestyles=itertools.cycle(
        ["-", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]
    ),
    errs=None,
):
    """
    Plot multiple learned diffusion profiles D(u) vs. their true counterparts.

    For each experiment group, the ensemble of predicted D(u) over repeats
    is averaged and plotted, with optional min/max shading. True diffusion
    curves and percentile lines of u are also shown.

    Parameters
    ----------
    modelWrapper_dics : dict
        Dictionary of {group_key: {seed: modelWrapper}}.
    dataobj : object
        Data container with attribute `u` for computing u percentiles.
    D_sym_true_lst : list
        List of symbolic or numerical true D(u) functions/arrays
        corresponding to each D_true in the first wrapper of each group.
    colors : list
        Colors to use for each group (truncated to len(labels)).
    labels : list
        Labels for each group (same length as `modelWrapper_dics`).
    Dnum : int, optional
        Index number for D in the label (D_1, D_2, ...), by default 1.
    device : str, optional
        Device for potential model evaluation (not heavily used here), by default "cpu".
    num_bins : int, optional
        Number of bins for u histogram for percentile marking, by default 50.
    K : float, optional
        Scaling factor applied to u-values, by default 1700.
    name : str, optional
        Output filename (PNG). If None, figure is not saved.
    fill : bool, optional
        If True, show min/max shading for ensemble, by default True.
    legend_pos : tuple, optional
        Legend anchor point in axes coordinates, by default (0.5, 0.5).
    linestyles : iterator, optional
        Cycle of linestyles for each group.
    errs : list, optional
        If provided as [err_up, err_low], plot ±percent bands around true D.

    Returns
    -------
    None
    """
    if errs:
        [err_up, err_low] = errs

    colors = colors[: len(labels)]
    h_properties = hist_properties(dataobj, num_bins)
    low_u = h_properties["low_count"]
    high_u = h_properties["high_count"]

    sample_model = list(list(modelWrapper_dics.values())[0].values())[0].model
    u_vals_torch = sample_model.u_vals_torch
    u_vals_np = sample_model.u_vals.flatten()

    results = []
    D_true_lst = []

    # Collect ensemble statistics per group
    for wrapper_dic, color, label in zip(
        modelWrapper_dics.values(), colors, labels
    ):
        diffusion_errors = []
        D_ensemble = []

        for i, wrapper in enumerate(wrapper_dic.values()):
            D_true_check = list(wrapper.model.D_true)
            if i == 0 and D_true_check not in D_true_lst:
                D_true_lst.append(D_true_check)

            model = wrapper.model
            model.eval()

            with torch.no_grad():
                diff_pred = (
                    model.D_scale * model.diffusion(model.u_vals_torch).flatten()
                )
                diffusion_error = (model.D_true_torch - diff_pred) ** 2
                diffusion_errors.append(diffusion_error.unsqueeze(0))
                D_ensemble.append(diff_pred.unsqueeze(0))

        D_ensemble = torch.cat(D_ensemble, dim=0)
        D_mean = torch.mean(D_ensemble, dim=0)
        D_min = torch.min(D_ensemble, dim=0).values
        D_max = torch.max(D_ensemble, dim=0).values

        diffusion_errors = torch.cat(diffusion_errors, dim=0)
        mse = torch.mean(diffusion_errors).item()

        results.append(
            (mse, label, color, D_mean.numpy(), D_min.numpy(), D_max.numpy())
        )

    fig, ax = plt.subplots(figsize=(7, 5))

    for j, (mse, label, color, D_mean, D_min, D_max) in enumerate(results):
        ls = next(linestyles)
        ax.plot(
            u_vals_np * K,
            D_mean,
            lw=2,
            color=color,
            linestyle=ls,
            label=label,
        )

        if fill:
            ax.fill_between(
                u_vals_np * K, D_min, D_max, alpha=0.4, color=color
            )

    # Plot true D(u) and optional error bands
    for i, (D_true, D_sym) in enumerate(zip(D_true_lst, D_sym_true_lst)):
        if i == 0:
            ax.plot(
                u_vals_np * K,
                D_true,
                "--",
                lw=2,
                color="k",
                label=f"$D_{Dnum}(u)$",
                zorder=3,
            )

            if errs is not None:
                ax.plot(
                    u_vals_np * K,
                    np.array(D_true) * (1 + err_up / 100),
                    "--",
                    lw=1,
                    color="r",
                    label=f"{err_up}%",
                    zorder=3,
                )
                ax.plot(
                    u_vals_np * K,
                    np.array(D_true) * (1 - err_low / 100),
                    "-.",
                    lw=1,
                    color="r",
                    label=f"{err_low}%",
                    zorder=3,
                )
        else:
            ax.plot(u_vals_np * K, D_true, "--", lw=2, color="k", zorder=3)

    ax.axvline(
        low_u * K, color="#666666", ls="-.", lw=1, label="5% $u$-perc."
    )
    ax.axvline(
        high_u * K, color="#BBBBBB", ls="--", lw=1, label="95% $u$-perc."
    )
    ax.set_xlabel(r"cell density [cells mm$^{-2}]$", fontsize=11)
    ax.set_ylabel(r"cell diffusion [mm$^2$ days$^{-1}$]", fontsize=11)
    ax.set_facecolor("white")
    ax.legend(
        loc="center",
        bbox_to_anchor=legend_pos,
        ncols=2,
        fontsize=12,
    )

    plt.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()


def plot_repeats_times_u(binn_models_dics, base_colors, filename=None):
    """
    Plot total training time for u-models across widths and depths.

    For each (width, depth) pair, the mean and standard deviation of total
    training time (after clipping epoch times at the 95th percentile) are
    shown as a bar with error bar on a log y-axis.

    Parameters
    ----------
    binn_models_dics : dict
        Nested dict {depth: {width: {seed: modelWrapper}}}.
    base_colors : list
        Colors for each width.
    filename : str, optional
        Output filename (PNG). If None, figure is not saved.

    Returns
    -------
    None
    """
    hatch_styles = ["", "//", ".."]

    depths = list(binn_models_dics.keys())
    widths = list(binn_models_dics[depths[0]].keys())

    def mean_95(epoch_times):
        arr = np.array(epoch_times, dtype=np.float64)
        p95 = np.percentile(arr, 95)
        return np.mean(arr[arr <= p95])

    def collect_metric(binn_models, metric_attr1, metric_attr2):
        out = {}
        for d in depths:
            for w in widths:
                repeats = binn_models[d][w]
                out[(w, d)] = [
                    mean_95(getattr(m, metric_attr1))
                    * len(getattr(m, metric_attr2))
                    for m in repeats.values()
                ]
        return out

    data = collect_metric(
        binn_models_dics,
        metric_attr1="epoch_times",
        metric_attr2="train_loss_list",
    )

    bar_width = 0.15
    intra_gap, inter_gap = 0.02, 0.10
    current_x = 0.0

    xtick_positions = []
    xtick_labels = []

    plt.figure(figsize=(7, 5))

    for (w, base_c) in zip(widths, base_colors):
        group_x_positions = []

        for i_d, d in enumerate(depths):
            vals = data[(w, d)]
            mean_val = np.mean(vals)
            std_val = np.std(vals)

            plt.bar(
                current_x,
                mean_val,
                yerr=std_val,
                width=bar_width,
                color=base_c,
                hatch=hatch_styles[i_d],
                edgecolor="k",
                error_kw=dict(ecolor="gray", lw=2),
                capsize=4,
            )

            group_x_positions.append(current_x)
            current_x += bar_width + (
                intra_gap if i_d < len(depths) - 1 else inter_gap
            )

        center_x = np.mean(group_x_positions)
        xtick_positions.append(center_x)
        xtick_labels.append(f"{w}")

    plt.xticks([], [])
    plt.ylabel(r"Total training time [s]", fontsize=14)

    depth_patches = [
        Patch(
            facecolor="white",
            edgecolor="k",
            hatch=hatch_styles[i],
            label=rf"NN$_\mathrm{{D}}$ = {d}",
        )
        for i, d in enumerate(depths)
    ]
    plt.legend(handles=depth_patches, loc="upper left", fontsize=16)

    plt.yscale("log")

    ax = plt.gca()
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10, subs="auto", numticks=100)
    )
    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.tick_params(axis="y", which="minor", labelsize=11)
    ax.tick_params(axis="y", which="major", labelsize=12)

    plt.tight_layout()

    if filename:
        plt.savefig(f"{filename}.png", dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()


def plot_repeats_times_u_control(
    binn_models_dics_control, binn_models_dics, base_colors, filename=None
):
    """
    Plot total training time using control models for epoch-time measurements.

    Uses epoch times from `binn_models_dics_control` but epoch counts from
    `binn_models_dics`. Otherwise identical to `plot_repeats_times_u`.

    Parameters
    ----------
    binn_models_dics_control : dict
        Nested dict of control models {depth: {width: {seed: modelWrapper}}}.
    binn_models_dics : dict
        Nested dict of main models {depth: {width: {seed: modelWrapper}}}.
    base_colors : list
        Colors per width.
    filename : str, optional
        Output filename (PNG). If None, figure is not saved.

    Returns
    -------
    None
    """
    hatch_styles = ["", "//", ".."]

    depths = list(binn_models_dics.keys())
    widths = list(binn_models_dics[depths[0]].keys())

    def mean_95(epoch_times):
        arr = np.array(epoch_times, dtype=np.float64)
        p95 = np.percentile(arr, 95)
        return np.mean(arr[arr <= p95])

    def collect_metric(binn_models_control, binn_models, metric_attr1, metric_attr2):
        out = {}
        for d in depths:
            for w in widths:
                repeats_control = binn_models_control[d][w]
                repeats = binn_models[d][w]
                out[(w, d)] = [
                    mean_95(getattr(m_control, metric_attr1))
                    * len(getattr(m, metric_attr2))
                    for m_control, m in zip(
                        repeats_control.values(), repeats.values()
                    )
                ]
        return out

    data = collect_metric(
        binn_models_dics_control,
        binn_models_dics,
        metric_attr1="epoch_times",
        metric_attr2="train_loss_list",
    )

    bar_width = 0.15
    intra_gap, inter_gap = 0.02, 0.10
    current_x = 0.0

    xtick_positions = []
    xtick_labels = []

    plt.figure(figsize=(7, 5))

    for (w, base_c) in zip(widths, base_colors):
        group_x_positions = []

        for i_d, d in enumerate(depths):
            vals = data[(w, d)]
            mean_val = np.mean(vals)
            std_val = np.std(vals)

            plt.bar(
                current_x,
                mean_val,
                yerr=std_val,
                width=bar_width,
                color=base_c,
                hatch=hatch_styles[i_d],
                edgecolor="k",
                error_kw=dict(ecolor="gray", lw=2),
                capsize=4,
            )

            group_x_positions.append(current_x)
            current_x += bar_width + (
                intra_gap if i_d < len(depths) - 1 else inter_gap
            )

        center_x = np.mean(group_x_positions)
        xtick_positions.append(center_x)
        xtick_labels.append(f"{w}")

    plt.xticks([], [])
    plt.ylabel(r"Total training time [s]", fontsize=14)

    depth_patches = [
        Patch(
            facecolor="white",
            edgecolor="k",
            hatch=hatch_styles[i],
            label=rf"NN$_\mathrm{{D}}$ = {d}",
        )
        for i, d in enumerate(depths)
    ]
    plt.legend(handles=depth_patches, loc="upper left", fontsize=16)

    plt.yscale("log")

    ax = plt.gca()
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10, subs="auto", numticks=100)
    )
    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.tick_params(axis="y", which="minor", labelsize=11)
    ax.tick_params(axis="y", which="major", labelsize=12)

    plt.tight_layout()

    if filename:
        plt.savefig(f"{filename}.png", dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()



def plot_eval2(
    modelWrappers_dic,
    dataobj,
    IC=0,
    device="cpu",
    num_bins=10,
    label="",
    clean=True,
    noisy=True,
    error=True,
    K=1700,
    histogram_save_name = None,
    save_name=None,
    colors=None,
    plot_raw_error=True,
    raw_error_save_name=None,
    plot_binned_error=True,          # NEW
    binned_error_save_name=None,     # NEW
    binned_error_logy=False,         # NEW: log scale for mean abs error
    bin_reference="clean",           # NEW: "clean" or "noisy" for binning variable
    fig_size = (7, 5),
    legend_title = "noise (%)",
):
    if not colors:
        colors = ["#9013FE", "#5E239D", "#B580FF", "#E5D5FF"]
    markers = ["o", "s", "D", "^", "p", "<", ">", "p", "*", "h", "x", "+"]
    grayscale_colors = ["#111111", "#444444", "#888888", "#BBBBBB", "#EEEEEE"]

    # All models for same IC will have same data distribution -> use first model
    modelWrapper0 = list(modelWrappers_dic.values())[0]
    u_val = np.array([k for (_, _), k in zip(modelWrapper0.x_val, modelWrapper0.y_val)])
    u_train = np.array([k for (_, _), k in zip(modelWrapper0.x_train, modelWrapper0.y_train)])

    # Get data
    x = dataobj.x
    u = dataobj.u * K
    u_clean = dataobj.u_clean * K
    t = dataobj.t
    Nt, Nx = len(dataobj.t), len(dataobj.x)

    # Histogram properties
    model = modelWrapper0.model
    model.eval()
    h_properties = hist_properties(dataobj, num_bins)

    bin_edges = h_properties["bin_edges"] * K
    bin_centers = h_properties["bin_centers"] * K

    # Compute separate histograms for train/val
    hist_train, _ = np.histogram(u_train * K, bins=bin_edges)
    hist_val, _ = np.histogram(u_val * K, bins=bin_edges)

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=fig_size)
    bar_width = bin_edges[1] - bin_edges[0]

    ax.bar(
        bin_centers, hist_val, width=bar_width, alpha=0.7, label="Validation",
        color=colors[1], edgecolor="black", linewidth=0.5
    )
    ax.bar(
        bin_centers, hist_train, width=bar_width, alpha=0.7, bottom=hist_val,
        label="Train", color=colors[0], edgecolor="black", linewidth=0.5
    )

    ax.set_xlabel("cell density [cells mm$^{-2}]$", fontsize=11)
    ax.set_ylabel("frequency", fontsize=11)
    ax.legend()
    plt.tight_layout()

    if histogram_save_name:
        plt.savefig(histogram_save_name, dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    # Helper: binned stats
    def binned_error_stats(u_ref_flat, err_signed_flat, edges):
        """
        Bin by u_ref_flat using edges; aggregate signed and abs errors per bin.
        Returns:
          centers, mean_abs, mean_signed, counts
        """
        u_ref_flat = np.asarray(u_ref_flat)
        err_signed_flat = np.asarray(err_signed_flat)

        abs_err = np.abs(err_signed_flat)

        # Assign each point to a bin index in [0, nbins-1]
        bin_idx = np.digitize(u_ref_flat, edges) - 1
        nb = len(edges) - 1

        mean_abs = np.full(nb, np.nan, dtype=float)
        mean_signed = np.full(nb, np.nan, dtype=float)
        counts = np.zeros(nb, dtype=int)

        for b in range(nb):
            m = bin_idx == b
            c = int(np.sum(m))
            counts[b] = c
            if c > 0:
                mean_abs[b] = float(np.mean(abs_err[m]))
                mean_signed[b] = float(np.mean(err_signed_flat[m]))

        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, mean_abs, mean_signed, counts

    # ------------------------------------------------------------------
    # Prediction vs. truth at t=0 and t=T (+ abs error)
    # ------------------------------------------------------------------
    for modelWrapper_indx, modelWrapper in modelWrappers_dic.items():
        model = modelWrapper.model

        with torch.no_grad():
            u_pred_flat = model(to_torch(dataobj.inputs, device)) * K
            u_pred = u_pred_flat.reshape(Nx, Nt).cpu().numpy()

        signed_e = (u_pred - u_clean)
        abs_e = np.abs(signed_e)

        # Your “abs error curve” plot (as before)
        fig, ax1 = plt.subplots(figsize=fig_size)
        ax1.plot(x, u_pred[:, 0], "-", lw=2, alpha=0.7, color=colors[0], label=r"$\hat{u}_{dn}(0,x)$")
        ax1.plot(x, u_pred[:, -1], "-", lw=2, alpha=0.7, color=colors[1], label=r"$\hat{u}_{dn}(T,x)$")

        if error:
            ax2 = ax1.twinx()
            ax2.set_ylabel("abs. error [cells mm$^{-2}]$", fontsize=11)
            ax2.set_yscale("log")

        if clean:
            ax1.plot(
                x, u_clean[:, 0], label="t=0", markersize=4, color=grayscale_colors[0],
                marker=markers[0], lw=0, markeredgecolor="black", markeredgewidth=0.5
            )
            ax1.plot(
                x, u_clean[:, -1], label="t=T", markersize=4, color=grayscale_colors[4],
                marker=markers[4], lw=0, markeredgecolor="black", markeredgewidth=0.5
            )

        if noisy:
            ax1.plot(x, u[:, 0], label="t=0", color=grayscale_colors[0], lw=1)
            ax1.plot(x, u[:, -1], label="t=T", color=grayscale_colors[3], lw=1)

        if error:
            ax2.plot(
                x, abs_e[:, 0], label="|err(x,0)|", markersize=4, color="#8b0000",
                marker=markers[0], markeredgecolor="black", markeredgewidth=0.5, alpha=0.7, lw=1
            )
            ax2.plot(
                x, abs_e[:, -1], label="|err(x,T)|", markersize=4, color="#dd0505",
                marker=markers[4], markeredgecolor="black", markeredgewidth=0.5, alpha=0.7, lw=1
            )

        ax1.set_xlabel("x [mm]", fontsize=11)
        ax1.set_ylabel("cell density [cells mm$^{-2}]$", fontsize=11)

        lines1, labels1 = ax1.get_legend_handles_labels()
        if error:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8, ncol=1, title=legend_title)
        else:
            ax1.legend(lines1, labels1, loc="lower right", fontsize=8, ncol=1, title=legend_title)

        ax1.set_facecolor("white")
        plt.tight_layout()
        if save_name:
            plt.savefig(save_name, dpi=100, bbox_inches="tight", facecolor="None")
        plt.show()

        # ------------------------------------------------------------------
        # Signed/raw error vs x (separate figure)
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Signed/raw error vs x for ALL time (separate figure)
        # ------------------------------------------------------------------
        if plot_raw_error:
            fig_e, ax_e = plt.subplots(figsize=fig_size)
            ax_e.axhline(0.0, lw=1, color="k")

            # Plot signed error for every time slice with same styling
            for j in range(Nt):
                ax_e.plot(
                    x,
                    signed_e[:, j],
                    label=f"err(x,t_{j})=pred-clean",
                    markersize=4,
                    marker=markers[j % len(markers)],
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    alpha=0.8,
                    lw=1,
                )

            ax_e.set_xlabel("x [mm]", fontsize=11)
            ax_e.set_ylabel("signed error [cells mm$^{-2}]$", fontsize=11)
            ax_e.legend(loc="best", fontsize=8, title=legend_title)
            ax_e.set_facecolor("white")
            plt.tight_layout()

            if raw_error_save_name:
                plt.savefig(raw_error_save_name, dpi=100, bbox_inches="tight", facecolor="None")
            plt.show()
            # ------------------------------------------------------------------
            # Net signed error across ALL time (sum over t)
            # ------------------------------------------------------------------
            net_signed_e = np.sum(signed_e, axis=1)  # shape (Nx,)

            fig_net, ax_net = plt.subplots(figsize=fig_size)
            ax_net.axhline(0.0, lw=1, color="k")

            ax_net.plot(
                x,
                net_signed_e,
                markersize=5,
                marker=markers[0],
                markeredgecolor="black",
                markeredgewidth=0.5,
                alpha=0.9,
                lw=1.5,
                label=r"$\sum_t (\hat u(x,t)-u(x,t))$",
            )

            ax_net.set_xlabel("x [mm]", fontsize=11)
            ax_net.set_ylabel("net signed error [cells mm$^{-2}]$", fontsize=11)
            ax_net.legend(loc="best", fontsize=8, title=legend_title)
            ax_net.set_facecolor("white")
            plt.tight_layout()

            # optional save
            if raw_error_save_name:
                base, ext = raw_error_save_name.rsplit(".", 1)
                plt.savefig(f"{base}_net.{ext}", dpi=100, bbox_inches="tight", facecolor="None")

            plt.show()


        # ------------------------------------------------------------------
    # NEW: Error vs binned density (separate figure) -- PERCENT SCALED
    #      AGGREGATED over models: mean + min-max shaded bands
    # ------------------------------------------------------------------
    if plot_binned_error:
        # choose what we bin by (same for all models)
        if bin_reference == "noisy":
            u_ref = u
        else:
            u_ref = u_clean  # default / recommended

        u_ref_flat = u_ref.reshape(-1)

        # Fixed bin count/centers (guaranteed consistent across models)
        nb = len(bin_edges) - 1
        dens_centers_ref = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        eps = 1e-12  # avoid divide-by-zero if a bin center is ~0
        denom = np.maximum(np.abs(dens_centers_ref), eps)

        # Collect per-model curves: (n_models, n_bins)
        per_model_abs_pct = []
        per_model_signed_pct = []

        # Loop over models ONLY to compute curves; plotting happens once
        for _, mw in modelWrappers_dic.items():
            model_i = mw.model
            model_i.eval()

            with torch.no_grad():
                u_pred_flat_i = model_i(to_torch(dataobj.inputs, device)) * K
                u_pred_i = u_pred_flat_i.reshape(Nx, Nt).cpu().numpy()

            signed_e_i = (u_pred_i - u_clean)
            signed_flat_i = signed_e_i.reshape(-1)

            dens_centers, mean_abs, mean_signed, counts = binned_error_stats(
                u_ref_flat=u_ref_flat,
                err_signed_flat=signed_flat_i,
                edges=bin_edges,
            )

            # Defensive: force fixed length nb (pad/trim with NaNs)
            mean_abs = np.asarray(mean_abs, dtype=float)
            mean_signed = np.asarray(mean_signed, dtype=float)

            if mean_abs.shape[0] != nb:
                tmp = np.full(nb, np.nan, dtype=float)
                m = min(nb, mean_abs.shape[0])
                tmp[:m] = mean_abs[:m]
                mean_abs = tmp

            if mean_signed.shape[0] != nb:
                tmp = np.full(nb, np.nan, dtype=float)
                m = min(nb, mean_signed.shape[0])
                tmp[:m] = mean_signed[:m]
                mean_signed = tmp

            mean_abs_pct = 100.0 * (mean_abs / denom)
            mean_signed_pct = 100.0 * (mean_signed / denom)

            per_model_abs_pct.append(mean_abs_pct)
            per_model_signed_pct.append(mean_signed_pct)

        # Stack to (M, B)
        per_model_abs_pct = np.vstack(per_model_abs_pct)
        per_model_signed_pct = np.vstack(per_model_signed_pct)

        # Aggregate across models, ignoring NaNs (bins with no samples in a model)
        abs_mean = np.nanmean(per_model_abs_pct, axis=0)
        abs_min  = np.nanmin(per_model_abs_pct, axis=0)
        abs_max  = np.nanmax(per_model_abs_pct, axis=0)

        signed_mean = np.nanmean(per_model_signed_pct, axis=0)
        signed_min  = np.nanmin(per_model_signed_pct, axis=0)
        signed_max  = np.nanmax(per_model_signed_pct, axis=0)

        fig_b, ax_b1 = plt.subplots(figsize=fig_size)

        # Mean abs % error (solid) + min-max band
        ax_b1.plot(
            dens_centers_ref,
            abs_mean,
            lw=1.8,
            linestyle="-",
            marker="o",
            label="mean |err| / density (%)",
        )
        # ax_b1.fill_between(
        #     dens_centers_ref,
        #     abs_min,
        #     abs_max,
        #     alpha=0.2,
        #     linewidth=0,
        #    # label="min–max |err| / density (%)",
        # )

        ax_b1.set_xlabel("Binned density [cells mm$^{-2}]$", fontsize=11)
        ax_b1.set_ylabel("Mean abs. % error [%]", fontsize=11)
        if binned_error_logy:
            ax_b1.set_yscale("log")

        # Signed % error on secondary axis (gray dotted) + min-max band
        ax_b2 = ax_b1.twinx()
        ax_b2.axhline(0.0, lw=1, color="k", linestyle="--", label="zero")

        ax_b2.plot(
            dens_centers_ref,
            signed_mean,
            lw=1.8,
            linestyle=":",
            marker="s",
            color="0.5",
            label="Mean signed err / density (%)",
        )
        # ax_b2.fill_between(
        #     dens_centers_ref,
        #     signed_min,
        #     signed_max,
        #     color="0.5",
        #     alpha=0.15,
        #     linewidth=0,
        #     #label="min–max signed err / density (%)",
        # )

        ax_b2.set_ylabel("Mean signed % error [%]", fontsize=11)

        # Combine legends (both axes)
        l1, lab1 = ax_b1.get_legend_handles_labels()
        l2, lab2 = ax_b2.get_legend_handles_labels()
        ax_b1.legend(l1 + l2, lab1 + lab2, loc="best", fontsize=8, title=legend_title)

        ax_b1.set_facecolor("white")
        plt.tight_layout()

        if binned_error_save_name:
            plt.savefig(
                binned_error_save_name,
                dpi=100,
                bbox_inches="tight",
                facecolor="None",
            )
        plt.show()
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_binned_density_error_agg_groups_multi_data(
    modelWrappers_by_group,
    dataobjs_by_group,
    *,
    device="cpu",
    K=1700,
    # --- binning: specify spacing (width) instead of number of bins ---
    bin_width=None,            # e.g. 50.0 (cells mm^-2). If None, falls back to num_bins.
    num_bins=10,               # fallback for legacy behaviour
    bin_reference="clean",     # "clean" or "noisy"
    bin_min=None,              # optional override for global lower edge (after scaling by K)
    bin_max=None,              # optional override for global upper edge (after scaling by K)
    error_units="percent",     # "percent" (default) or "absolute"
    square_error=False,
    binned_error_logy=False,
    fig_size=(7, 5),
    group_colors=None,
    abs_linestyle="-",
    signed_linestyle=":",
    baseline_linestyle="--",
    abs_marker="o",
    signed_marker="s",
    abs_band_alpha=0.20,
    signed_band_alpha=0.15,
    legend_fontsize=9,
    legend_loc="upper right",
    legend_bbox_to_anchor=None,
    legend_framealpha=0.9,
    save_names=None,  # None or dict with keys: "both", "abs", "signed", "both_mean_only"
    legend_title="noise (%)",
    legend_title_fontsize=None,
    axis_fontsizes=None,
    show_dual_axis=False,
    signed_legend_labels=None,
):
    """
    Produces single-axis absolute and signed error plots by default.
    If show_dual_axis=True, also produces:
      1) both abs (left axis) + signed (right axis), WITH min-max shading
      4) both abs+signed, mean lines only (NO shading)

    Binning
    ------
    If `bin_width` is provided, bins are *equal-width* in density space (fairer across groups
    with different density ranges). Bin edges are computed *globally across all groups* so
    the x-axis is directly comparable; groups with no samples in a bin produce NaNs there.

    Parameters
    ----------
    error_units : {"percent","absolute"}
        - "percent": plot mean abs/signed percentage error relative to |dens_centers|
        - "absolute": plot mean abs/signed error in absolute units of u (after scaling by K)
    show_dual_axis : bool
        If True, also create the dual-y-axis absolute/signed summary plots.
    legend_title_fontsize : int or None
        Fontsize for legend titles. Defaults to legend_fontsize.
    axis_fontsizes : dict or None
        Optional axis font-size controls with keys "xaxis", "xtick_labels",
        "yaxis", and "ytick_labels".
    signed_legend_labels : list or dict or None
        Optional labels for the signed-only plot. By default the group names are used.
    """
    legend_title_fontsize = legend_fontsize if legend_title_fontsize is None else legend_title_fontsize
    axis_fontsizes = axis_fontsizes or {}
    xaxis_fontsize = axis_fontsizes.get("xaxis", None)
    xtick_fontsize = axis_fontsizes.get("xtick_labels", xaxis_fontsize)
    yaxis_fontsize = axis_fontsizes.get("yaxis", None)
    ytick_fontsize = axis_fontsizes.get("ytick_labels", yaxis_fontsize)

    def format_axis(ax, xlabel=None, ylabel=None):
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=xaxis_fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=yaxis_fontsize)
        if xtick_fontsize is not None:
            ax.tick_params(axis="x", labelsize=xtick_fontsize)
        if ytick_fontsize is not None:
            ax.tick_params(axis="y", labelsize=ytick_fontsize)

    def legend_kwargs():
        kwargs = {
            "loc": legend_loc,
            "fontsize": legend_fontsize,
            "title": legend_title,
            "title_fontsize": legend_title_fontsize,
            "framealpha": legend_framealpha,
        }
        if legend_bbox_to_anchor is not None:
            kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        return kwargs

    group_names = list(modelWrappers_by_group.keys())

    # Map dataobjs into a dict keyed by group name
    if isinstance(dataobjs_by_group, dict):
        dataobj_map = dataobjs_by_group
    else:
        if len(dataobjs_by_group) != len(group_names):
            raise ValueError(
                f"dataobjs_by_group must have same length as groups. "
                f"Got {len(dataobjs_by_group)} vs {len(group_names)}."
            )
        dataobj_map = {g: dataobjs_by_group[i] for i, g in enumerate(group_names)}

    # Colors
    if group_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        group_colors = {g: cycle[i % len(cycle)] for i, g in enumerate(group_names)}
    elif isinstance(group_colors, list):
        group_colors = {g: group_colors[i % len(group_colors)] for i, g in enumerate(group_names)}
    else:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for i, g in enumerate(group_names):
            if g not in group_colors:
                group_colors[g] = cycle[i % len(cycle)]

    plot_keys = ("both", "abs", "signed", "both_mean_only", "hist")
    if save_names is None:
        save_names = {}
        enabled_plots = set(plot_keys)
    else:
        enabled_plots = {
            key for key in plot_keys
            if save_names.get(key)
        }
        if not enabled_plots:
            enabled_plots = set(plot_keys)

    # ---- error unit handling ----
    eu = str(error_units).lower().strip()
    if eu in {"%", "pct", "percent", "percentage"}:
        error_units = "percent"
    elif eu in {"abs", "absolute"}:
        error_units = "absolute"
    else:
        raise ValueError(f"error_units must be 'percent' or 'absolute' (got {error_units!r})")

    # Axis labels based on error_units
    if error_units == "percent":
        ylab_abs = "Mean abs. % error [%]"
        ylab_signed = "Mean signed % error [%]"
    else:
        ylab_abs = "Mean abs. error [cells mm$^{-2}]$"
        ylab_signed = "Mean signed error [cells mm$^{-2}]$"

    density_xlabel = "Binned density [cells mm$^{-2}]$"

    # ---------- GLOBAL bin edges (fair comparison across different density ranges) ----------
    # Collect reference densities from all groups to define a common set of edges
    all_u_ref = []
    for g in group_names:
        dataobj = dataobj_map[g]
        u = dataobj.u * K
        u_clean = dataobj.u_clean * K
        u_ref = u if bin_reference == "noisy" else u_clean
        all_u_ref.append(u_ref.reshape(-1))
    all_u_ref = np.concatenate(all_u_ref) if len(all_u_ref) else np.array([0.0])

    lo = float(np.nanmin(all_u_ref)) if bin_min is None else float(bin_min)
    hi = float(np.nanmax(all_u_ref)) if bin_max is None else float(bin_max)
    if np.isclose(lo, hi):
        lo, hi = lo - 0.5, hi + 0.5

    if bin_width is not None:
        bw = float(bin_width)
        if bw <= 0:
            raise ValueError(f"bin_width must be > 0 (got {bin_width})")
        # ensure the right edge reaches/exceeds hi
        bin_edges = np.arange(lo, hi + bw, bw)
        if bin_edges[-1] < hi:
            bin_edges = np.append(bin_edges, bin_edges[-1] + bw)
    else:
        # legacy: fixed number of bins over the global range
        bin_edges = np.linspace(lo, hi, num_bins + 1)

    nb = len(bin_edges) - 1
    dens_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # denom used only for percent mode
    eps = 1e-12
    denom = np.maximum(np.abs(dens_centers), eps)

    # ---------- Precompute aggregated curves per group ----------
    agg = {}  # g -> dict with dens_centers, abs_mean/min/max, signed_mean/min/max

    def binned_stats(u_ref_flat_local, signed_err_flat, edges):
        abs_err = np.abs(signed_err_flat)
        idx = np.digitize(u_ref_flat_local, edges) - 1

        mean_abs = np.full(nb, np.nan)
        mean_signed = np.full(nb, np.nan)

        for b in range(nb):
            m = idx == b
            if np.any(m):
                mean_abs[b] = np.mean(abs_err[m])
                mean_signed[b] = np.mean(signed_err_flat[m])
        return mean_abs, mean_signed

    for g in group_names:
        wrappers = modelWrappers_by_group[g]
        dataobj = dataobj_map[g]

        u = dataobj.u * K
        u_clean = dataobj.u_clean * K
        Nt, Nx = len(dataobj.t), len(dataobj.x)

        u_ref = u if bin_reference == "noisy" else u_clean
        u_ref_flat = u_ref.reshape(-1)

        inputs = torch.from_numpy(dataobj.inputs).float().to(device)

        abs_list, signed_list = [], []
        for mw in wrappers.values():
            model = mw.model.to(device).eval()
            with torch.no_grad():
                u_pred = model(inputs) * K
                u_pred = u_pred.reshape(Nx, Nt).cpu().numpy()

            signed_flat = (u_pred - u_clean).reshape(-1)
            mean_abs, mean_signed = binned_stats(u_ref_flat, signed_flat, bin_edges)

            if error_units == "percent":
                abs_vals = 100.0 * mean_abs / denom
                signed_vals = 100.0 * mean_signed / denom
            else:
                abs_vals = mean_abs
                signed_vals = mean_signed

            abs_list.append(abs_vals)
            signed_list.append(signed_vals)

        abs_arr = np.vstack(abs_list)
        signed_arr = np.vstack(signed_list)

        agg[g] = dict(
            dens_centers=dens_centers,
            abs_mean=np.nanmean(abs_arr, axis=0),
            abs_min=np.nanmin(abs_arr, axis=0),
            abs_max=np.nanmax(abs_arr, axis=0),
            signed_mean=np.nanmean(signed_arr, axis=0),
            signed_min=np.nanmin(signed_arr, axis=0),
            signed_max=np.nanmax(signed_arr, axis=0),
        )

    dual_axis_with_band = None
    if show_dual_axis:
        # ---------- Plot 1: both axes + shading ----------
        fig1, ax1 = plt.subplots(figsize=fig_size)
        ax1b = ax1.twinx()
        ax1b.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

        for g in group_names:
            color = group_colors[g]
            d = agg[g]

            ax1.plot(
                d["dens_centers"], d["abs_mean"],
                color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}"
            )
            ax1.fill_between(
                d["dens_centers"], d["abs_min"], d["abs_max"], color=color, alpha=abs_band_alpha
            )

            ax1b.plot(
                d["dens_centers"], d["signed_mean"],
                color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8,
                mfc=color, mec="black", mew=1.0
            )
            ax1b.fill_between(
                d["dens_centers"], d["signed_min"], d["signed_max"], color=color, alpha=signed_band_alpha
            )

        format_axis(ax1, "binned density [cells mm$^{-2}]$", ylab_abs)
        format_axis(ax1b, ylabel=ylab_signed)
        if binned_error_logy:
            ax1.set_yscale("log")
        ax1.legend(**legend_kwargs())
        ax1.set_facecolor("white")
        plt.tight_layout()
        if "both" in save_names and save_names["both"]:
            plt.savefig(save_names["both"], dpi=100, bbox_inches="tight", facecolor="None")
        plt.show()
        dual_axis_with_band = (fig1, ax1, ax1b)

    # ---------- Plot 2: abs only + shading ----------
    fig2, ax_abs = plt.subplots(figsize=fig_size)
    for g in group_names:
        color = group_colors[g]
        d = agg[g]
        ax_abs.plot(
            d["dens_centers"], d["abs_mean"],
            color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}",
            mfc=color, mec="black", mew=1.0
        )
        ax_abs.fill_between(
            d["dens_centers"], d["abs_min"], d["abs_max"], color=color, alpha=abs_band_alpha
        )

    format_axis(ax_abs, "binned density [cells mm$^{-2}]$", ylab_abs)
    if binned_error_logy:
        ax_abs.set_yscale("log")
    ax_abs.legend(**legend_kwargs())
    ax_abs.set_facecolor("white")
    plt.tight_layout()
    if "abs" in save_names and save_names["abs"]:
        plt.savefig(save_names["abs"], dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    # ---------- Plot 3: signed only + shading ----------
    fig3, ax_signed = plt.subplots(figsize=fig_size)
    ax_signed.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)
    signed_markers = ['o', 's', '^', 'p']
    for i, g in enumerate(group_names):
        color = group_colors[g]
        d = agg[g]
        if isinstance(signed_legend_labels, dict):
            signed_label = signed_legend_labels.get(g, f"{g}")
        elif signed_legend_labels is not None and i < len(signed_legend_labels):
            signed_label = signed_legend_labels[i]
        else:
            signed_label = f"{g}"
        ax_signed.plot(
            d["dens_centers"], d["signed_mean"],
            color=color, linestyle=signed_linestyle, marker=signed_markers[i % len(signed_markers)],
            lw=1.8, label=signed_label,
            mfc=color, mec="black", mew=1.0
        )
        ax_signed.fill_between(
            d["dens_centers"], d["signed_min"], d["signed_max"], color=color, alpha=signed_band_alpha
        )

    format_axis(ax_signed, "binned density [cells mm$^{-2}]$", ylab_signed)
    ax_signed.legend(**legend_kwargs())
    ax_signed.set_facecolor("white")
    plt.tight_layout()
    if "signed" in save_names and save_names["signed"]:
        plt.savefig(save_names["signed"], dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    dual_axis_mean_only = None
    if show_dual_axis:
        # ---------- Plot 4: both axes, mean lines only (no shading) ----------
        fig4, ax4 = plt.subplots(figsize=fig_size)
        ax4b = ax4.twinx()
        ax4b.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

        for g in group_names:
            color = group_colors[g]
            d = agg[g]
            ax4.plot(
                d["dens_centers"], d["abs_mean"],
                color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}"
            )
            ax4b.plot(
                d["dens_centers"], d["signed_mean"],
                color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8
            )

        format_axis(ax4, "binned density [cells mm$^{-2}]$", ylab_abs)
        format_axis(ax4b, ylabel=ylab_signed)
        if binned_error_logy:
            ax4.set_yscale("log")
        ax4.legend(**legend_kwargs())
        ax4.set_facecolor("white")
        plt.tight_layout()
        if "both_mean_only" in save_names and save_names["both_mean_only"]:
            plt.savefig(save_names["both_mean_only"], dpi=100, bbox_inches="tight", facecolor="None")
        plt.show()
        dual_axis_mean_only = (fig4, ax4, ax4b)

    return dual_axis_with_band, (fig2, ax_abs), (fig3, ax_signed), dual_axis_mean_only


def plot_density_histograms(
    modelWrappers_dic,
    # --- binning: specify spacing (width) instead of number of bins ---
    bin_width=None,     # e.g. 50.0; if None, fall back to num_bins
    num_bins=10,        # fallback for legacy behaviour
    K=1700,
    colors=None,
    fig_size=(7, 5),
    save_name=None,
    show=True,
    ax=None,
):
    """
    Plot stacked Train/Validation density histograms on the same axis, one stacked pair per
    *highest-level key* of `modelWrappers_dic`.

    Binning is defined globally for a fair comparison:
      - if `bin_width` is provided: equal-width bins over the global min/max (all groups)
      - else: `num_bins` bins over the global min/max (legacy)
    """

    if colors is None:
        colors = ["#9013FE", "#5E239D", "#B580FF", "#E5D5FF", "#1f77b4", "#ff7f0e"]

    def _extract_u_from_wrapper(mw, split="train"):
        if split == "train":
            xs, ys = mw.x_train, mw.y_train
        else:
            xs, ys = mw.x_val, mw.y_val
        return np.array([k for (_, _), k in zip(xs, ys)], dtype=float)

    def _get_global_edges():
        # gather ALL u across ALL wrappers to define consistent edges
        u_all_list = []
        for inner_dic in modelWrappers_dic.values():
            for mw in inner_dic.values():
                u_all_list.append(_extract_u_from_wrapper(mw, "train"))
                u_all_list.append(_extract_u_from_wrapper(mw, "val"))
        u_all = np.concatenate(u_all_list) * K if len(u_all_list) else np.array([0.0])

        lo, hi = float(np.min(u_all)), float(np.max(u_all))
        if np.isclose(lo, hi):
            lo, hi = lo - 0.5, hi + 0.5

        if bin_width is not None:
            bw = float(bin_width)
            if bw <= 0:
                raise ValueError(f"bin_width must be > 0 (got {bin_width})")
            edges = np.arange(lo, hi + bw, bw)
            if edges[-1] < hi:
                edges = np.append(edges, edges[-1] + bw)
        else:
            edges = np.linspace(lo, hi, num_bins + 1)

        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        return edges, centers, width

    edges, centers, bar_width = _get_global_edges()

    # ---- figure / axis ----
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    # ---- compute + plot per outer key ----
    for i, (outer_key, inner_dic) in enumerate(modelWrappers_dic.items()):
        h_train_list, h_val_list = [], []

        for _, mw in inner_dic.items():
            u_train = _extract_u_from_wrapper(mw, "train") * K
            u_val = _extract_u_from_wrapper(mw, "val") * K

            h_train, _ = np.histogram(u_train, bins=edges)
            h_val, _ = np.histogram(u_val, bins=edges)

            h_train_list.append(h_train)
            h_val_list.append(h_val)

        h_train_mean = np.mean(np.vstack(h_train_list), axis=0)
        h_val_mean = np.mean(np.vstack(h_val_list), axis=0)

        c = colors[i % len(colors)]

        ax.bar(
            centers,
            h_val_mean,
            width=bar_width,
            alpha=0.35,
            color=c,
            edgecolor="black",
            linewidth=0.5,
            label=f"{outer_key} (val)",
        )
        ax.bar(
            centers,
            h_train_mean,
            width=bar_width,
            alpha=0.75,
            bottom=h_val_mean,
            color=c,
            edgecolor="black",
            linewidth=0.5,
            label=f"{outer_key} (train)",
        )

    ax.set_xlabel("cell density [cells mm$^{-2}$]")
    ax.set_ylabel("frequency")
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save_name:
        fig.savefig(save_name, dpi=100, bbox_inches="tight", facecolor="None")
    if show:
        plt.show()

    return ax




import numpy as np
import matplotlib.pyplot as plt


def plot_density_histograms(
    modelWrappers_dic,
    num_bins=10,
    K=1700,
    colors=None,
    fig_size=(7, 5),
    save_name=None,
    show=True,
    ax=None,
):
    """
    Plot stacked Train/Validation density histograms on the same axis, one stacked pair per
    *highest-level key* of `modelWrappers_dic`.

    Expected structure (same as before):
        modelWrappers_dic[outer_key][inner_key] -> modelWrapper
    where each modelWrapper has:
        - x_train, y_train  (iterables of pairs like before)
        - x_val,   y_val

    Procedure:
      - For each outer_key: average histograms over inner models (inner_key level)
      - Plot on a shared axis with separation by outer_key (different colors)

    Minimal, self-contained: only needs modelWrappers_dic and basic plotting params.
    """

    if colors is None:
        colors = ["#9013FE", "#5E239D", "#B580FF", "#E5D5FF", "#1f77b4", "#ff7f0e"]

    # ---- helpers ----
    def _extract_u_from_wrapper(mw, split="train"):
        # Matches your earlier pattern:
        # u = np.array([k for (_, _), k in zip(mw.x_train, mw.y_train)])
        if split == "train":
            xs, ys = mw.x_train, mw.y_train
        else:
            xs, ys = mw.x_val, mw.y_val
        return np.array([k for (_, _), k in zip(xs, ys)], dtype=float)

    def _get_global_edges():
        # Use first available modelWrapper to define bin edges consistently
        first_outer = next(iter(modelWrappers_dic.values()))
        mw0 = next(iter(first_outer.values()))
        u_all = np.concatenate(
            [
                _extract_u_from_wrapper(mw0, "train"),
                _extract_u_from_wrapper(mw0, "val"),
            ]
        ) * K
        # Guard against degenerate range
        lo, hi = float(np.min(u_all)), float(np.max(u_all))
        if np.isclose(lo, hi):
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, num_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        return edges, centers, width

    edges, centers, bar_width = _get_global_edges()

    # ---- figure / axis ----
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    # ---- compute + plot per outer key ----
    for i, (outer_key, inner_dic) in enumerate(modelWrappers_dic.items()):
        # average over inner models
        h_train_list, h_val_list = [], []

        for _, mw in inner_dic.items():
            u_train = _extract_u_from_wrapper(mw, "train") * K
            u_val = _extract_u_from_wrapper(mw, "val") * K

            h_train, _ = np.histogram(u_train, bins=edges)
            h_val, _ = np.histogram(u_val, bins=edges)

            h_train_list.append(h_train)
            h_val_list.append(h_val)

        h_train_mean = np.mean(np.vstack(h_train_list), axis=0)
        h_val_mean = np.mean(np.vstack(h_val_list), axis=0)

        c = colors[i % len(colors)]

        # stacked bars: val at bottom, train on top (matching your style)
        ax.bar(
            centers,
            h_val_mean,
            width=bar_width,
            alpha=0.35,
            color=c,
            edgecolor="black",
            linewidth=0.5,
            label=f"{outer_key} (val)",
        )
        ax.bar(
            centers,
            h_train_mean,
            width=bar_width,
            alpha=0.75,
            bottom=h_val_mean,
            color=c,
            edgecolor="black",
            linewidth=0.5,
            label=f"{outer_key} (train)",
        )

    ax.set_xlabel("cell density [cells mm$^{-2}$]")
    ax.set_ylabel("frequency")
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save_name:
        fig.savefig(save_name, dpi=100, bbox_inches="tight", facecolor="None")
    if show:
        plt.show()

    return ax



def plot_binned_clean_vs_noisy_density_error(
    dataobjs_by_group,
    *,
    K=1700,
    num_bins=10,
    bin_reference="clean",  # "clean" or "noisy" (which density defines the bins)
    error_units="percent",  # "percent" (default) or "absolute"
    binned_error_logy=False,
    fig_size=(7, 5),
    group_colors=None,      # None, list, or dict {group: color}
    abs_linestyle="-",
    signed_linestyle=":",
    baseline_linestyle="--",
    abs_marker="o",
    signed_marker="s",
    abs_band_alpha=0.20,
    signed_band_alpha=0.15,
    legend_fontsize=9,
    legend_loc="upper right",
    legend_bbox_to_anchor=None,
    legend_framealpha=0.9,
    save_names=None,  # None or dict with keys: "both","abs","signed","both_mean_only"
    legend_title="noise (%)",
    legend_title_fontsize=None,
    axis_fontsizes=None,
    show_dual_axis=False,
):
    """
    Plots clean-vs-noisy density errors, with single-axis absolute and signed
    plots by default. If show_dual_axis=True, also creates the dual-y-axis
    absolute/signed summary plots.

      - uses only dataobj.u_clean (clean) vs dataobj.u (noisy)
      - error is defined as: signed = (u_noisy - u_clean), abs = |signed|
      - binned by either clean or noisy densities (bin_reference)
      - aggregates across each group's dataobjs (mean + min/max bands across replicates)

    error_units : {"percent","absolute"}
        - "percent": plot mean abs/signed percentage error relative to |bin center| (default)
        - "absolute": plot mean abs/signed error in absolute units of u (after scaling by K)

    Expected per dataobj:
      - dataobj.u_clean : array
      - dataobj.u       : array
    Shapes can be (Nt,Nx), (Nx,Nt), or any shape; we flatten for binning.

    show_dual_axis : bool
        If True, also create the dual-y-axis absolute/signed summary plots.
    legend_title_fontsize : int or None
        Fontsize for legend titles. Defaults to legend_fontsize.
    axis_fontsizes : dict or None
        Optional axis font-size controls with keys "xaxis", "xtick_labels",
        "yaxis", and "ytick_labels".
    """
    legend_title_fontsize = legend_fontsize if legend_title_fontsize is None else legend_title_fontsize
    axis_fontsizes = axis_fontsizes or {}
    xaxis_fontsize = axis_fontsizes.get("xaxis", None)
    xtick_fontsize = axis_fontsizes.get("xtick_labels", xaxis_fontsize)
    yaxis_fontsize = axis_fontsizes.get("yaxis", None)
    ytick_fontsize = axis_fontsizes.get("ytick_labels", yaxis_fontsize)

    def format_axis(ax, xlabel=None, ylabel=None):
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=xaxis_fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=yaxis_fontsize)
        if xtick_fontsize is not None:
            ax.tick_params(axis="x", labelsize=xtick_fontsize)
        if ytick_fontsize is not None:
            ax.tick_params(axis="y", labelsize=ytick_fontsize)

    def legend_kwargs():
        kwargs = {
            "loc": legend_loc,
            "fontsize": legend_fontsize,
            "title": legend_title,
            "title_fontsize": legend_title_fontsize,
            "framealpha": legend_framealpha,
        }
        if legend_bbox_to_anchor is not None:
            kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        return kwargs

    # -----------------------------
    # Normalize inputs to dict-of-lists
    # -----------------------------
    if isinstance(dataobjs_by_group, dict):
        group_names = list(dataobjs_by_group.keys())
        group_to_dataobjs = {
            g: (v if isinstance(v, (list, tuple)) else [v]) for g, v in dataobjs_by_group.items()
        }
    else:
        raise ValueError("dataobjs_by_group must be a dict: {group: dataobj or [dataobj,...]}")

    # -----------------------------
    # Colors
    # -----------------------------
    if group_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        group_colors = {g: cycle[i % len(cycle)] for i, g in enumerate(group_names)}
    elif isinstance(group_colors, list):
        group_colors = {g: group_colors[i % len(group_colors)] for i, g in enumerate(group_names)}
    else:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for i, g in enumerate(group_names):
            if g not in group_colors:
                group_colors[g] = cycle[i % len(cycle)]

    if save_names is None:
        save_names = {}

    # -----------------------------
    # error unit handling + labels
    # -----------------------------
    eu = str(error_units).lower().strip()
    if eu in {"%", "pct", "percent", "percentage"}:
        error_units = "percent"
    elif eu in {"abs", "absolute"}:
        error_units = "absolute"
    else:
        raise ValueError(f"error_units must be 'percent' or 'absolute' (got {error_units!r})")

    if error_units == "percent":
        ylab_abs = "Mean abs. % error [%]"
        ylab_signed = "Mean signed % error [%]"
    else:
        ylab_abs = "Mean abs. error [cells mm$^{-2}$]"
        ylab_signed = "Mean signed error [cells mm$^{-2}$]"

    # -----------------------------
    # Helper: binned mean of abs and signed errors
    # -----------------------------
    def binned_means(u_ref_flat, signed_flat, bin_edges):
        nb = len(bin_edges) - 1
        dens_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        idx = np.digitize(u_ref_flat, bin_edges) - 1

        mean_abs = np.full(nb, np.nan)
        mean_signed = np.full(nb, np.nan)

        abs_flat = np.abs(signed_flat)
        for b in range(nb):
            m = idx == b
            if np.any(m):
                mean_abs[b] = np.mean(abs_flat[m])
                mean_signed[b] = np.mean(signed_flat[m])

        if error_units == "percent":
            eps = 1e-12
            denom = np.maximum(np.abs(dens_centers), eps)
            abs_vals = 100.0 * mean_abs / denom
            signed_vals = 100.0 * mean_signed / denom
        else:  # "absolute"
            abs_vals = mean_abs
            signed_vals = mean_signed

        return dens_centers, abs_vals, signed_vals

    # -----------------------------
    # Aggregate per group: mean + min/max across replicates
    # -----------------------------
    agg = {}
    for g in group_names:
        reps = group_to_dataobjs[g]

        # Build group-wide bin edges (based on chosen reference)
        u_refs = []
        for d in reps:
            u_clean = np.asarray(d.u_clean) * K
            u_noisy = np.asarray(d.u) * K
            u_ref = u_clean if bin_reference == "clean" else u_noisy
            u_refs.append(u_ref.reshape(-1))

        u_ref_all = np.concatenate(u_refs, axis=0)
        bin_edges = np.histogram_bin_edges(u_ref_all, bins=num_bins)

        abs_list, signed_list = [], []
        dens_centers_final = None

        for d in reps:
            u_clean = np.asarray(d.u_clean) * K
            u_noisy = np.asarray(d.u) * K

            u_ref = u_clean if bin_reference == "clean" else u_noisy
            u_ref_flat = u_ref.reshape(-1)

            signed_flat = (u_noisy - u_clean).reshape(-1)

            dens_centers, abs_vals, signed_vals = binned_means(u_ref_flat, signed_flat, bin_edges)
            dens_centers_final = dens_centers  # same for all reps in group
            abs_list.append(abs_vals)
            signed_list.append(signed_vals)

        abs_arr = np.vstack(abs_list)     # [nreps, nbins]
        signed_arr = np.vstack(signed_list)

        agg[g] = dict(
            dens_centers=dens_centers_final,
            abs_mean=np.nanmean(abs_arr, axis=0),
            abs_min=np.nanmin(abs_arr, axis=0),
            abs_max=np.nanmax(abs_arr, axis=0),
            signed_mean=np.nanmean(signed_arr, axis=0),
            signed_min=np.nanmin(signed_arr, axis=0),
            signed_max=np.nanmax(signed_arr, axis=0),
        )

    dual_axis_with_band = None
    if show_dual_axis:
        # -----------------------------
        # Plot 1: both axes + shading
        # -----------------------------
        fig1, ax1 = plt.subplots(figsize=fig_size)
        ax1b = ax1.twinx()
        ax1b.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

        for g in group_names:
            color = group_colors[g]
            d = agg[g]

            ax1.plot(
                d["dens_centers"], d["abs_mean"],
                color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}"
            )
            #ax1.fill_between(d["dens_centers"], d["abs_min"], d["abs_max"], color=color, alpha=abs_band_alpha)

            ax1b.plot(
                d["dens_centers"], d["signed_mean"],
                color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8,
                mfc=color, mec="black", mew=1.0
            )
            #ax1b.fill_between(d["dens_centers"], d["signed_min"], d["signed_max"], color=color, alpha=signed_band_alpha)

        format_axis(ax1, "Binned density [cells mm$^{-2}]$", ylab_abs)
        format_axis(ax1b, ylabel=ylab_signed)
        if binned_error_logy:
            ax1.set_yscale("log")
        ax1.legend(**legend_kwargs())
        ax1.set_facecolor("white")
        plt.tight_layout()
        if save_names.get("both"):
            plt.savefig(save_names["both"], dpi=100, bbox_inches="tight", facecolor="None")
            print("saved plot:", save_names["both"])
        plt.show()
        dual_axis_with_band = (fig1, ax1, ax1b)

    # -----------------------------
    # Plot 2: abs only + shading
    # -----------------------------
    fig2, ax_abs = plt.subplots(figsize=fig_size)
    for g in group_names:
        color = group_colors[g]
        d = agg[g]
        ax_abs.plot(
            d["dens_centers"], d["abs_mean"],
            color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}",
            mfc=color, mec="black", mew=1.0
        )
        #ax_abs.fill_between(d["dens_centers"], d["abs_min"], d["abs_max"], color=color, alpha=abs_band_alpha)

    format_axis(ax_abs, "Binned density [cells mm$^{-2}]$", ylab_abs)
    if binned_error_logy:
        ax_abs.set_yscale("log")
    ax_abs.legend(**legend_kwargs())
    ax_abs.set_facecolor("white")
    plt.tight_layout()
    if save_names.get("abs"):
        plt.savefig(save_names["abs"], dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    # -----------------------------
    # Plot 3: signed only + shading
    # -----------------------------
    fig3, ax_signed = plt.subplots(figsize=fig_size)
    ax_signed.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

    for g in group_names:
        color = group_colors[g]
        d = agg[g]
        ax_signed.plot(
            d["dens_centers"], d["signed_mean"],
            color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8, label=f"{g}",
            mfc=color, mec="black", mew=1.0
        )
        #ax_signed.fill_between(d["dens_centers"], d["signed_min"], d["signed_max"], color=color, alpha=signed_band_alpha)

    format_axis(ax_signed, "Binned density [cells mm$^{-2}]$", ylab_signed)
    ax_signed.legend(**legend_kwargs())
    ax_signed.set_facecolor("white")
    plt.tight_layout()
    if save_names.get("signed"):
        plt.savefig(save_names["signed"], dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    dual_axis_mean_only = None
    if show_dual_axis:
        # -----------------------------
        # Plot 4: both axes, mean lines only
        # -----------------------------
        fig4, ax4 = plt.subplots(figsize=fig_size)
        ax4b = ax4.twinx()
        ax4b.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

        for g in group_names:
            color = group_colors[g]
            d = agg[g]
            ax4.plot(
                d["dens_centers"], d["abs_mean"],
                color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}"
            )
            ax4b.plot(
                d["dens_centers"], d["signed_mean"],
                color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8
            )

        format_axis(ax4, "Binned density [cells mm$^{-2}]$", ylab_abs)
        format_axis(ax4b, ylabel=ylab_signed)
        if binned_error_logy:
            ax4.set_yscale("log")
        ax4.legend(**legend_kwargs())
        ax4.set_facecolor("white")
        plt.tight_layout()
        if save_names.get("both_mean_only"):
            plt.savefig(save_names["both_mean_only"], dpi=100, bbox_inches="tight", facecolor="None")
        plt.show()
        dual_axis_mean_only = (fig4, ax4, ax4b)

    return dual_axis_with_band, (fig2, ax_abs), (fig3, ax_signed), dual_axis_mean_only


def plot_binned_density_diffusion_error_agg_groups_multi_data(
    modelWrappers_by_group,
    dataobjs_by_group,
    *,
    true_diffusion_fn,         # true_diffusion_fn(u_np) -> D_true(u_np)  (u only)
    K=1700,
    num_bins=10,
    error_units="percent",     # "percent" or "absolute"
    binned_error_logy=False,
    fig_size=(7, 5),
    group_colors=None,         # None, list, or dict {group: color}
    abs_linestyle="-",
    signed_linestyle=":",
    baseline_linestyle="--",
    abs_marker="o",
    signed_marker="s",
    abs_band_alpha=0.20,
    signed_band_alpha=0.15,
    legend_fontsize=9,
    legend_loc="upper left",
    legend_bbox_to_anchor=None,
    legend_framealpha=0.9,
    save_names=None,           # keys: "both","abs","signed","both_mean_only"
    legend_title="noise (%)",
    legend_title_fontsize=None,
    axis_fontsizes=None,
    device="cpu",
    pct_eps=1e-12,             # stability epsilon for percent mode
    show_dual_axis=False,
):
    """
    Same plotting structure/style as `plot_binned_clean_vs_noisy_density_error`, but for diffusion errors
    from an ensemble of models, binned by density.

    - Groups: keys of modelWrappers_by_group (e.g. noise levels)
    - Within each group: multiple runs (wrappers.values())
    - Binning x-axis: density u (from ONE u-ray: model.u_vals_torch)
    - True diffusion: provided by user: true_diffusion_fn(u)  (ONLY argument is u)
    - Pred diffusion: diff_pred = model.D_scale * model.diffusion(model.u_vals_torch).flatten()
    - Error:
        * absolute mode: signed = (D_pred - D_true), abs = |signed|
        * percent mode : signed% = 100*(D_pred - D_true)/D_true, abs% = |signed%|
      then binned by u.

    X-axis scaling requirement:
      - x is scaled just before plotting to 0..K: x_plot = dens_centers * K, and xlim is (0, K).
        (This assumes u_vals_torch is in [0,1]. If it's already in physical units, set K=1.)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    legend_title_fontsize = legend_fontsize if legend_title_fontsize is None else legend_title_fontsize
    axis_fontsizes = axis_fontsizes or {}
    xaxis_fontsize = axis_fontsizes.get("xaxis", None)
    xtick_fontsize = axis_fontsizes.get("xtick_labels", xaxis_fontsize)
    yaxis_fontsize = axis_fontsizes.get("yaxis", None)
    ytick_fontsize = axis_fontsizes.get("ytick_labels", yaxis_fontsize)

    def format_axis(ax, xlabel=None, ylabel=None):
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=xaxis_fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=yaxis_fontsize)
        if xtick_fontsize is not None:
            ax.tick_params(axis="x", labelsize=xtick_fontsize)
        if ytick_fontsize is not None:
            ax.tick_params(axis="y", labelsize=ytick_fontsize)

    def legend_kwargs():
        kwargs = {
            "loc": legend_loc,
            "fontsize": legend_fontsize,
            "title": legend_title,
            "title_fontsize": legend_title_fontsize,
            "framealpha": legend_framealpha,
        }
        if legend_bbox_to_anchor is not None:
            kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        return kwargs

    if true_diffusion_fn is None or not callable(true_diffusion_fn):
        raise ValueError("true_diffusion_fn must be a callable with signature true_diffusion_fn(u).")

    # -----------------------------
    # Normalize inputs to dicts
    # -----------------------------
    if not isinstance(modelWrappers_by_group, dict):
        raise ValueError("modelWrappers_by_group must be a dict: {group: {run: modelWrapper, ...}, ...}")

    # dataobjs_by_group kept for compatibility; normalize similarly but not required for computations here
    if isinstance(dataobjs_by_group, dict):
        group_names = list(modelWrappers_by_group.keys())
    else:
        raise ValueError("dataobjs_by_group must be a dict (kept for API consistency with your codebase).")

    # -----------------------------
    # Colors (same logic as your base function)
    # -----------------------------
    if group_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        group_colors = {g: cycle[i % len(cycle)] for i, g in enumerate(group_names)}
    elif isinstance(group_colors, list):
        group_colors = {g: group_colors[i % len(group_colors)] for i, g in enumerate(group_names)}
    else:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for i, g in enumerate(group_names):
            if g not in group_colors:
                group_colors[g] = cycle[i % len(cycle)]

    if save_names is None:
        save_names = {}

    # -----------------------------
    # error unit handling + labels (same style as base)
    # -----------------------------
    eu = str(error_units).lower().strip()
    if eu in {"%", "pct", "percent", "percentage"}:
        error_units = "percent"
    elif eu in {"abs", "absolute"}:
        error_units = "absolute"
    else:
        raise ValueError(f"error_units must be 'percent' or 'absolute' (got {error_units!r})")

    if error_units == "percent":
        ylab_abs = "Mean abs. % error [%]"
        ylab_signed = "Mean signed % error [%]"
    else:
        # keep diffusion units generic (you can swap to your preferred unit label)
        ylab_abs = "Mean abs. error [days$^{-2}$ mm$^{4}$]"
        ylab_signed = "Mean signed error [days$^{-2}$ mm$^{4}$]"

    # -----------------------------
    # Helper: binned mean of abs and signed errors (mirrors your base function)
    # -----------------------------
    def binned_means(u_ref_flat, signed_flat, bin_edges):
        nb = len(bin_edges) - 1
        dens_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        idx = np.digitize(u_ref_flat, bin_edges) - 1

        mean_abs = np.full(nb, np.nan)
        mean_signed = np.full(nb, np.nan)

        abs_flat = np.abs(signed_flat)
        for b in range(nb):
            m = idx == b
            if np.any(m):
                mean_abs[b] = np.mean(abs_flat[m])
                mean_signed[b] = np.mean(signed_flat[m])

        if error_units == "percent":
            # For diffusion percent error, base it on D_true(u) pointwise BEFORE binning.
            # So here we assume signed_flat is already in percent units.
            abs_vals = mean_abs
            signed_vals = mean_signed
        else:
            abs_vals = mean_abs
            signed_vals = mean_signed

        return dens_centers, abs_vals, signed_vals

    # -----------------------------
    # Aggregate per group: mean + min/max across runs (wrappers)
    # -----------------------------
    agg = {}
    for g in group_names:
        wrappers = modelWrappers_by_group[g]

        # Build group-wide bin edges from ALL runs' u-values (one u-ray per run; concatenate)
        u_refs = []
        for mw in wrappers.values():
            model = mw.model.to(device).eval()
            with torch.no_grad():
                u_for_D = model.u_vals_torch.detach().to("cpu").flatten().numpy()
            u_refs.append(u_for_D.reshape(-1))

        u_ref_all = np.concatenate(u_refs, axis=0)
        bin_edges = np.histogram_bin_edges(u_ref_all, bins=num_bins)

        abs_list, signed_list = [], []
        dens_centers_final = None

        for mw in wrappers.values():
            model = mw.model.to(device).eval()

            with torch.no_grad():
                u_for_D = model.u_vals_torch.detach().to("cpu").flatten().numpy()
                D_pred = (
                    model.D_scale * model.diffusion(model.u_vals_torch).flatten()
                ).detach().to("cpu").flatten().numpy()

            D_true = np.asarray(true_diffusion_fn(u_for_D)).reshape(-1)

            if D_pred.size != D_true.size:
                raise ValueError(
                    f"Size mismatch in group {g!r}: pred diffusion has {D_pred.size} values, "
                    f"but true_diffusion_fn returned {D_true.size} values. "
                    "They must match model.u_vals_torch.flatten()."
                )

            if error_units == "percent":
                denom = np.maximum(np.abs(D_true), pct_eps)
                signed_flat = 100.0 * (D_pred - D_true) / denom
            else:
                signed_flat = (D_pred - D_true)

            dens_centers, abs_vals, signed_vals = binned_means(u_for_D.reshape(-1), signed_flat.reshape(-1), bin_edges)
            dens_centers_final = dens_centers
            abs_list.append(abs_vals)
            signed_list.append(signed_vals)

        abs_arr = np.vstack(abs_list)     # [nruns, nbins]
        signed_arr = np.vstack(signed_list)

        agg[g] = dict(
            dens_centers=dens_centers_final,
            abs_mean=np.nanmean(abs_arr, axis=0),
            abs_min=np.nanmin(abs_arr, axis=0),
            abs_max=np.nanmax(abs_arr, axis=0),
            signed_mean=np.nanmean(signed_arr, axis=0),
            signed_min=np.nanmin(signed_arr, axis=0),
            signed_max=np.nanmax(signed_arr, axis=0),
        )

    # -----------------------------
    # X-axis scaling: enforce 0..K by scaling just before plotting
    # -----------------------------
    def x_plot(x):
        return x * K  # assumes u in [0,1]

    xlim_lo, xlim_hi = 0.0, float(K)

    dual_axis_with_band = None
    if show_dual_axis:
        # -----------------------------
        # Plot 1: both axes + shading
        # -----------------------------
        fig1, ax1 = plt.subplots(figsize=fig_size)
        ax1b = ax1.twinx()
        ax1b.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

        for g in group_names:
            color = group_colors[g]
            d = agg[g]
            x = x_plot(d["dens_centers"])

            ax1.plot(
                x, d["abs_mean"],
                color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}"
            )
            ax1.fill_between(x, d["abs_min"], d["abs_max"], color=color, alpha=abs_band_alpha)

            ax1b.plot(
                x, d["signed_mean"],
                color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8,
                mfc=color, mec="black", mew=1.0
            )
            ax1b.fill_between(x, d["signed_min"], d["signed_max"], color=color, alpha=signed_band_alpha)

        format_axis(ax1, "Binned density [cells mm$^{-2}]$", ylab_abs)
        format_axis(ax1b, ylabel=ylab_signed)
        ax1.set_xlim(xlim_lo, xlim_hi)
        ax1b.set_xlim(xlim_lo, xlim_hi)
        if binned_error_logy:
            ax1.set_yscale("log")
        ax1.legend(**legend_kwargs())
        ax1.set_facecolor("white")
        plt.tight_layout()
        if save_names.get("both"):
            plt.savefig(save_names["both"], dpi=100, bbox_inches="tight", facecolor="None")
        plt.show()
        dual_axis_with_band = (fig1, ax1, ax1b)

    # -----------------------------
    # Plot 2: abs only + shading
    # -----------------------------
    fig2, ax_abs = plt.subplots(figsize=fig_size)
    for g in group_names:
        color = group_colors[g]
        d = agg[g]
        x = x_plot(d["dens_centers"])

        ax_abs.plot(
            x, d["abs_mean"],
            color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}",
            mfc=color, mec="black", mew=1.0
        )
        ax_abs.fill_between(x, d["abs_min"], d["abs_max"], color=color, alpha=abs_band_alpha)

    format_axis(ax_abs, "Binned density [cells mm$^{-2}]$", ylab_abs)
    ax_abs.set_xlim(xlim_lo, xlim_hi)
    if binned_error_logy:
        ax_abs.set_yscale("log")
    ax_abs.legend(**legend_kwargs())
    ax_abs.set_facecolor("white")
    plt.tight_layout()
    if save_names.get("abs"):
        plt.savefig(save_names["abs"], dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    # -----------------------------
    # Plot 3: signed only + shading
    # -----------------------------
    fig3, ax_signed = plt.subplots(figsize=fig_size)
    ax_signed.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

    for g in group_names:
        color = group_colors[g]
        d = agg[g]
        x = x_plot(d["dens_centers"])

        ax_signed.plot(
            x, d["signed_mean"],
            color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8, label=f"{g}",
            mfc=color, mec="black", mew=1.0
        )
        ax_signed.fill_between(x, d["signed_min"], d["signed_max"], color=color, alpha=signed_band_alpha)

    format_axis(ax_signed, "Binned density [cells mm$^{-2}]$", ylab_signed)
    ax_signed.set_xlim(xlim_lo, xlim_hi)
    ax_signed.legend(**legend_kwargs())
    ax_signed.set_facecolor("white")
    plt.tight_layout()
    if save_names.get("signed"):
        plt.savefig(save_names["signed"], dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    dual_axis_mean_only = None
    if show_dual_axis:
        # -----------------------------
        # Plot 4: both axes, mean lines only
        # -----------------------------
        fig4, ax4 = plt.subplots(figsize=fig_size)
        ax4b = ax4.twinx()
        ax4b.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

        for g in group_names:
            color = group_colors[g]
            d = agg[g]
            x = x_plot(d["dens_centers"])

            ax4.plot(
                x, d["abs_mean"],
                color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}"
            )
            ax4b.plot(
                x, d["signed_mean"],
                color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8
            )

        format_axis(ax4, "Binned density [cells mm$^{-2}]$", ylab_abs)
        format_axis(ax4b, ylabel=ylab_signed)
        ax4.set_xlim(xlim_lo, xlim_hi)
        ax4b.set_xlim(xlim_lo, xlim_hi)
        if binned_error_logy:
            ax4.set_yscale("log")
        ax4.legend(**legend_kwargs())
        ax4.set_facecolor("white")
        plt.tight_layout()
        if save_names.get("both_mean_only"):
            plt.savefig(save_names["both_mean_only"], dpi=100, bbox_inches="tight", facecolor="None")
        plt.show()
        dual_axis_mean_only = (fig4, ax4, ax4b)

    return dual_axis_with_band, (fig2, ax_abs), (fig3, ax_signed), dual_axis_mean_only


import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_error_space_time_agg_groups_multi_data(
    modelWrappers_by_group,
    dataobjs_by_group,
    *,
    device="cpu",
    K=1700,
    error_units="percent",          # "percent" or "absolute"
    error_kind="abs",               # "abs" or "signed"
    percent_reference="clean",      # "clean" or "pred_denom"
    eps=1e-12,
    aggregate_over="models",        # "models" or "models_and_data"
    show_band=True,
    band_mode="minmax",             # "minmax" or "std"
    band_alpha=0.20,
    cmap="viridis",
    fig_size=(7, 5),
    vmin=None,
    vmax=None,
    save_names=None,                # None or dict with keys like {"g1":"file.png", ...} or {"all":"file.png"}
    suptitle=None,
):
    """
    Analogous to plot_binned_density_error_agg_groups_multi_data, but for error as a function of space and time.

    Assumptions about each dataobj:
      - dataobj.inputs : array of (x,t) pairs, shape (N,2). (User note: e.g. (190,2))
      - dataobj.u_clean: clean target values at those points, shape (N,) or (N,1)
      - optionally dataobj.u (noisy), not required here

    What it does:
      - For each group:
          * runs each model wrapper on dataobj.inputs -> u_pred
          * computes signed error (u_pred - u_clean) and abs error
          * aggregates across models (and optionally across multiple dataobjs per group)
          * plots a space-time heatmap by reshaping onto a regular (x,t) grid

    Returns:
      figs_by_group: dict[group] -> (fig, ax, im)
    """

    group_names = list(modelWrappers_by_group.keys())

    # ---- map dataobjs ----
    if isinstance(dataobjs_by_group, dict):
        dataobj_map = dataobjs_by_group
    else:
        if len(dataobjs_by_group) != len(group_names):
            raise ValueError(
                f"dataobjs_by_group must have same length as groups. "
                f"Got {len(dataobjs_by_group)} vs {len(group_names)}."
            )
        dataobj_map = {g: dataobjs_by_group[i] for i, g in enumerate(group_names)}

    # allow a single dataobj per group OR a list/tuple of dataobjs per group
    def _as_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    # ---- unit handling ----
    eu = str(error_units).lower().strip()
    if eu in {"%", "pct", "percent", "percentage"}:
        error_units = "percent"
    elif eu in {"abs", "absolute"}:
        error_units = "absolute"
    else:
        raise ValueError(f"error_units must be 'percent' or 'absolute' (got {error_units!r})")

    ek = str(error_kind).lower().strip()
    if ek not in {"abs", "signed"}:
        raise ValueError(f"error_kind must be 'abs' or 'signed' (got {error_kind!r})")

    agg_over = str(aggregate_over).lower().strip()
    if agg_over not in {"models", "models_and_data"}:
        raise ValueError("aggregate_over must be 'models' or 'models_and_data'")

    band_mode = str(band_mode).lower().strip()
    if band_mode not in {"minmax", "std"}:
        raise ValueError("band_mode must be 'minmax' or 'std'")

    if save_names is None:
        save_names = {}

    figs_by_group = {}

    for g in group_names:
        wrappers = modelWrappers_by_group[g]
        data_list = _as_list(dataobj_map[g])

        # We'll aggregate in a dict keyed by (x,t) grid location. In practice we reshape to a grid.
        # We assume the (x,t) pairs form a regular grid (possibly repeated ordering).
        per_sample_stack = []  # each entry is (M, Nx, Nt) for one dataobj, where M = #models

        grid_meta = None  # (x_unique, t_unique, Xgrid, Tgrid)
        for dataobj in data_list:
            # inputs: (N,2) with columns [x, t]
            inputs_np = np.asarray(dataobj.inputs)
            if inputs_np.ndim != 2 or inputs_np.shape[1] != 2:
                raise ValueError(f"{g}: dataobj.inputs must have shape (N,2), got {inputs_np.shape}")

            x = inputs_np[:, 0]
            t = inputs_np[:, 1]

            x_unique = np.unique(x)
            t_unique = np.unique(t)
            Nx = len(x_unique)
            Nt = len(t_unique)

            # map each (x,t) to indices in the grid
            x_to_i = {val: i for i, val in enumerate(x_unique)}
            t_to_j = {val: j for j, val in enumerate(t_unique)}
            xi = np.array([x_to_i[v] for v in x], dtype=int)
            tj = np.array([t_to_j[v] for v in t], dtype=int)

            # target clean values at those points
            u_clean = np.asarray(dataobj.u_clean).reshape(-1)
            if u_clean.shape[0] != inputs_np.shape[0]:
                raise ValueError(
                    f"{g}: u_clean length ({u_clean.shape[0]}) must match inputs N ({inputs_np.shape[0]})"
                )

            # torch inputs
            inputs = torch.from_numpy(inputs_np).float().to(device)

            # predict for each model in group
            model_err_grids = []
            for mw in wrappers.values():
                model = mw.model.to(device).eval()
                with torch.no_grad():
                    u_pred = model(inputs).reshape(-1).detach().cpu().numpy()

                # scale to physical units (same idea as your original)
                u_pred = u_pred * K
                u_clean_scaled = u_clean * K

                signed = (u_pred - u_clean_scaled)
                if ek == "abs":
                    vals = np.abs(signed)
                else:
                    vals = signed

                if error_units == "percent":
                    # default: percent relative to |u_clean| (like your density version)
                    if str(percent_reference).lower().strip() == "clean":
                        denom = np.maximum(np.abs(u_clean_scaled), eps)
                    elif str(percent_reference).lower().strip() == "pred_denom":
                        denom = np.maximum(np.abs(u_pred), eps)
                    else:
                        raise ValueError("percent_reference must be 'clean' or 'pred_denom'")
                    vals = 100.0 * vals / denom

                # place into a (Nx, Nt) grid
                grid = np.full((Nx, Nt), np.nan, dtype=float)
                grid[xi, tj] = vals
                model_err_grids.append(grid)

            model_err_grids = np.stack(model_err_grids, axis=0)  # (M, Nx, Nt)
            per_sample_stack.append(model_err_grids)

            # store grid meta (use first dataobj)
            if grid_meta is None:
                Xg, Tg = np.meshgrid(t_unique, x_unique)  # careful: pcolormesh wants X=columns, Y=rows
                grid_meta = (x_unique, t_unique, Xg, Tg)
            else:
                # sanity check same grid if aggregating across data
                x_u0, t_u0, _, _ = grid_meta
                if not (np.array_equal(x_u0, x_unique) and np.array_equal(t_u0, t_unique)):
                    raise ValueError(
                        f"{g}: multiple dataobjs have different (x,t) grids; "
                        f"need identical grids to aggregate. "
                    )

        # aggregate
        # per_sample_stack: list of (M,Nx,Nt). If multiple dataobjs, stack them too.
        if len(per_sample_stack) == 1:
            all_grids = per_sample_stack[0]  # (M,Nx,Nt)
        else:
            # shape -> (S,M,Nx,Nt)
            all_grids = np.stack(per_sample_stack, axis=0)

            if agg_over == "models":
                # aggregate within each sample over models, but keep samples separate for band
                # mean over models -> (S,Nx,Nt)
                all_grids = np.nanmean(all_grids, axis=1)
            else:
                # flatten samples and models into one ensemble axis
                # (S,M,Nx,Nt) -> (S*M,Nx,Nt)
                S, M, Nx, Nt = all_grids.shape
                all_grids = all_grids.reshape(S * M, Nx, Nt)

        # now all_grids is either:
        #   - (M,Nx,Nt) if single dataobj
        #   - (S,Nx,Nt) if aggregate_over="models" with multiple dataobjs
        #   - (E,Nx,Nt) ensemble if aggregate_over="models_and_data"
        if all_grids.ndim == 3:
            ensemble = all_grids
        else:
            raise RuntimeError("Unexpected aggregation shape.")

        mean_grid = np.nanmean(ensemble, axis=0)  # (Nx,Nt)

        if show_band:
            if band_mode == "minmax":
                lo_grid = np.nanmin(ensemble, axis=0)
                hi_grid = np.nanmax(ensemble, axis=0)
            else:  # std
                std_grid = np.nanstd(ensemble, axis=0)
                lo_grid = mean_grid - std_grid
                hi_grid = mean_grid + std_grid
        else:
            lo_grid = hi_grid = None

        x_unique, t_unique, Xg, Tg = grid_meta

        # ---- plot heatmap of mean error ----
        fig, ax = plt.subplots(figsize=fig_size)

        # pcolormesh expects "X" as t and "Y" as x for our choice above
        # We give edges if possible; otherwise use centers (matplotlib will still render)
        im = ax.pcolormesh(
            t_unique, x_unique, mean_grid,
            shading="auto",
            vmin=vmin, vmax=vmax,
            cmap=cmap,
        )
        cb = fig.colorbar(im, ax=ax)

        if error_units == "percent":
            cb.set_label(f"mean {ek} % error [%]")
        else:
            cb.set_label(f"mean {ek} error [cells mm$^{{-2}}$]")

        ax.set_xlabel("t")
        ax.set_ylabel("x")
        title = f"{g}: mean {ek} error over space-time"
        ax.set_title(title if suptitle is None else suptitle)

        plt.tight_layout()

        # save
        if g in save_names and save_names[g]:
            plt.savefig(save_names[g], dpi=120, bbox_inches="tight", facecolor="None")
        elif "all" in save_names and save_names["all"]:
            # if user passes one filename, append group name
            base = save_names["all"]
            if base.lower().endswith((".png", ".pdf", ".svg", ".jpg", ".jpeg")):
                stem, ext = base.rsplit(".", 1)
                out = f"{stem}_{g}.{ext}"
            else:
                out = f"{base}_{g}.png"
            plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="None")

        plt.show()

        figs_by_group[g] = (fig, ax, im)

        # ---- optional: band visualization (two extra heatmaps) ----
        if show_band:
            # Plot band width as another heatmap (hi-lo), often the most interpretable “uncertainty”
            band = hi_grid - lo_grid
            figb, axb = plt.subplots(figsize=fig_size)
            imb = axb.pcolormesh(
                t_unique, x_unique, band,
                shading="auto",
                cmap=cmap,
            )
            cbb = figb.colorbar(imb, ax=axb)
            cbb.set_label(f"{band_mode} band width ({ek}, {error_units})")
            axb.set_xlabel("t")
            axb.set_ylabel("x")
            axb.set_title(f"{g}: band width over space-time")
            plt.tight_layout()
            plt.show()

    return figs_by_group


import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_binned_density_error_agg_groups_multi_data_sum(
    modelWrappers_by_group,
    dataobjs_by_group,
    *,
    device="cpu",
    K=1700,
    # --- binning: specify spacing (width) instead of number of bins ---
    bin_width=None,            # e.g. 50.0 (cells mm^-2). If None, falls back to num_bins.
    num_bins=10,               # fallback for legacy behaviour
    bin_reference="clean",     # "clean" or "noisy"
    bin_min=None,              # optional override for global lower edge (after scaling by K)
    bin_max=None,              # optional override for global upper edge (after scaling by K)
    # --- plotting ---
    binned_error_logy=False,
    fig_size=(7, 5),
    group_colors=None,
    abs_linestyle="-",
    signed_linestyle=":",
    baseline_linestyle="--",
    abs_marker="o",
    signed_marker="s",
    abs_band_alpha=0.20,
    signed_band_alpha=0.15,
    legend_fontsize=9,
    save_names=None,  # None or dict with keys: "both", "abs", "signed", "both_sum_only"
    legend_title="noise (%)",
    # --- labels ---
    xlab="binned density [cells mm$^{-2}]$",
    ylab_abs="sum abs. error [cells mm$^{-2}]$",
    ylab_signed="sum signed error [cells mm$^{-2}]$",
):
    """
    Same idea as your mean-aggregated function, but uses SUM within each density bin
    (across all space-time points), not MEAN.

    Notes
    -----
    - This is in absolute units only (no percentage mode).
    - Bin edges are GLOBAL across all groups for fair comparison.
    - For each group: we compute per-model binned sums, then aggregate across TV splits
      via mean/min/max (to retain your shading style).
    """

    group_names = list(modelWrappers_by_group.keys())

    # Map dataobjs into a dict keyed by group name
    if isinstance(dataobjs_by_group, dict):
        dataobj_map = dataobjs_by_group
    else:
        if len(dataobjs_by_group) != len(group_names):
            raise ValueError(
                f"dataobjs_by_group must have same length as groups. "
                f"Got {len(dataobjs_by_group)} vs {len(group_names)}."
            )
        dataobj_map = {g: dataobjs_by_group[i] for i, g in enumerate(group_names)}

    # Colors
    if group_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        group_colors = {g: cycle[i % len(cycle)] for i, g in enumerate(group_names)}
    elif isinstance(group_colors, list):
        group_colors = {g: group_colors[i % len(group_colors)] for i, g in enumerate(group_names)}
    else:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for i, g in enumerate(group_names):
            if g not in group_colors:
                group_colors[g] = cycle[i % len(cycle)]

    if save_names is None:
        save_names = {}

    # ---------- GLOBAL bin edges ----------
    all_u_ref = []
    for g in group_names:
        dataobj = dataobj_map[g]
        u = dataobj.u * K
        u_clean = dataobj.u_clean * K
        u_ref = u if bin_reference == "noisy" else u_clean
        all_u_ref.append(u_ref.reshape(-1))
    all_u_ref = np.concatenate(all_u_ref) if len(all_u_ref) else np.array([0.0])

    lo = float(np.nanmin(all_u_ref)) if bin_min is None else float(bin_min)
    hi = float(np.nanmax(all_u_ref)) if bin_max is None else float(bin_max)
    if np.isclose(lo, hi):
        lo, hi = lo - 0.5, hi + 0.5

    if bin_width is not None:
        bw = float(bin_width)
        if bw <= 0:
            raise ValueError(f"bin_width must be > 0 (got {bin_width})")
        bin_edges = np.arange(lo, hi + bw, bw)
        if bin_edges[-1] < hi:
            bin_edges = np.append(bin_edges, bin_edges[-1] + bw)
    else:
        bin_edges = np.linspace(lo, hi, num_bins + 1)

    nb = len(bin_edges) - 1
    dens_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def binned_sums(u_ref_flat_local, signed_err_flat, edges):
        abs_err = np.abs(signed_err_flat)
        idx = np.digitize(u_ref_flat_local, edges) - 1

        sum_abs = np.full(nb, np.nan)
        sum_signed = np.full(nb, np.nan)

        for b in range(nb):
            m = idx == b
            if np.any(m):
                sum_abs[b] = np.sum(abs_err[m])
                sum_signed[b] = np.sum(signed_err_flat[m])
        return sum_abs, sum_signed

    # ---------- Precompute aggregated curves per group ----------
    agg = {}
    for g in group_names:
        wrappers = modelWrappers_by_group[g]
        dataobj = dataobj_map[g]

        u = dataobj.u * K
        u_clean = dataobj.u_clean * K
        Nt, Nx = len(dataobj.t), len(dataobj.x)

        u_ref = u if bin_reference == "noisy" else u_clean
        u_ref_flat = u_ref.reshape(-1)

        inputs = torch.from_numpy(dataobj.inputs).float().to(device)

        abs_list, signed_list = [], []
        for mw in wrappers.values():
            model = mw.model.to(device).eval()
            with torch.no_grad():
                u_pred = model(inputs) * K
                u_pred = u_pred.reshape(Nx, Nt).cpu().numpy()

            signed_flat = (u_pred - u_clean).reshape(-1)
            sum_abs, sum_signed = binned_sums(u_ref_flat, signed_flat, bin_edges)

            abs_list.append(sum_abs)
            signed_list.append(sum_signed)

        abs_arr = np.vstack(abs_list)     # [n_models, nb]
        signed_arr = np.vstack(signed_list)

        agg[g] = dict(
            dens_centers=dens_centers,
            abs_mean=np.nanmean(abs_arr, axis=0),
            abs_min=np.nanmin(abs_arr, axis=0),
            abs_max=np.nanmax(abs_arr, axis=0),
            signed_mean=np.nanmean(signed_arr, axis=0),
            signed_min=np.nanmin(signed_arr, axis=0),
            signed_max=np.nanmax(signed_arr, axis=0),
        )

    # ---------- Plot 1: both axes + shading ----------
    fig1, ax1 = plt.subplots(figsize=fig_size)
    ax1b = ax1.twinx()
    ax1b.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

    for g in group_names:
        color = group_colors[g]
        d = agg[g]

        ax1.plot(
            d["dens_centers"], d["abs_mean"],
            color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}"
        )
        #ax1.fill_between(d["dens_centers"], d["abs_min"], d["abs_max"], color=color, alpha=abs_band_alpha)

        ax1b.plot(
            d["dens_centers"], d["signed_mean"],
            color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8,
            mfc=color, mec="black", mew=1.0
        )
        #ax1b.fill_between(d["dens_centers"], d["signed_min"], d["signed_max"], color=color, alpha=signed_band_alpha)

    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab_abs)
    ax1b.set_ylabel(ylab_signed)
    if binned_error_logy:
        ax1.set_yscale("log")
    ax1.legend(loc="best", fontsize=legend_fontsize, title=legend_title)
    ax1.set_facecolor("white")
    plt.tight_layout()
    if save_names.get("both"):
        plt.savefig(save_names["both"], dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    # ---------- Plot 2: abs only + shading ----------
    fig2, ax_abs = plt.subplots(figsize=fig_size)
    for g in group_names:
        color = group_colors[g]
        d = agg[g]
        ax_abs.plot(
            d["dens_centers"], d["abs_mean"],
            color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}",
            mfc=color, mec="black", mew=1.0
        )
        #ax_abs.fill_between(d["dens_centers"], d["abs_min"], d["abs_max"], color=color, alpha=abs_band_alpha)

    ax_abs.set_xlabel(xlab)
    ax_abs.set_ylabel(ylab_abs)
    if binned_error_logy:
        ax_abs.set_yscale("log")
    ax_abs.legend(loc="best", fontsize=legend_fontsize, title=legend_title)
    ax_abs.set_facecolor("white")
    plt.tight_layout()
    if save_names.get("abs"):
        plt.savefig(save_names["abs"], dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    # ---------- Plot 3: signed only + shading ----------
    fig3, ax_signed = plt.subplots(figsize=fig_size)
    ax_signed.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

    for g in group_names:
        color = group_colors[g]
        d = agg[g]
        ax_signed.plot(
            d["dens_centers"], d["signed_mean"],
            color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8, label=f"{g}",
            mfc=color, mec="black", mew=1.0
        )
        #ax_signed.fill_between(d["dens_centers"], d["signed_min"], d["signed_max"], color=color, alpha=signed_band_alpha)

    ax_signed.set_xlabel(xlab)
    ax_signed.set_ylabel(ylab_signed)
    ax_signed.legend(loc="best", fontsize=legend_fontsize, title=legend_title)
    ax_signed.set_facecolor("white")
    plt.tight_layout()
    if save_names.get("signed"):
        plt.savefig(save_names["signed"], dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    # ---------- Plot 4: both axes, mean lines only (no shading) ----------
    fig4, ax4 = plt.subplots(figsize=fig_size)
    ax4b = ax4.twinx()
    ax4b.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

    for g in group_names:
        color = group_colors[g]
        d = agg[g]
        ax4.plot(
            d["dens_centers"], d["abs_mean"],
            color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}"
        )
        ax4b.plot(
            d["dens_centers"], d["signed_mean"],
            color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8
        )

    ax4.set_xlabel(xlab)
    ax4.set_ylabel(ylab_abs)
    ax4b.set_ylabel(ylab_signed)
    if binned_error_logy:
        ax4.set_yscale("log")
    ax4.legend(loc="best", fontsize=legend_fontsize, title=legend_title)
    ax4.set_facecolor("white")
    plt.tight_layout()
    if save_names.get("both_sum_only"):
        plt.savefig(save_names["both_sum_only"], dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    return (fig1, ax1, ax1b), (fig2, ax_abs), (fig3, ax_signed), (fig4, ax4, ax4b)
import numpy as np
import torch
import matplotlib.pyplot as plt












def plot_binned_density_error_agg_groups_multi_data_median(
    modelWrappers_by_group,
    dataobjs_by_group,
    *,
    device="cpu",
    K=1700,
    # --- binning: prefer spacing for fair comparison across different density ranges ---
    bin_width=None,            # e.g. 50.0 (cells mm^-2). If None, falls back to num_bins.
    num_bins=10,               # fallback for legacy behaviour
    bin_reference="clean",     # "clean" or "noisy"
    bin_min=None,              # optional override for global lower edge (after scaling by K)
    bin_max=None,              # optional override for global upper edge (after scaling by K)
    error_units="percent",     # "percent" (default) or "absolute"
    square_error=False,
    binned_error_logy=False,
    fig_size=(7, 5),
    group_colors=None,
    abs_linestyle="-",
    signed_linestyle=":",
    baseline_linestyle="--",
    abs_marker="o",
    signed_marker="s",
    abs_band_alpha=0.20,
    signed_band_alpha=0.15,
    legend_fontsize=9,
    save_names=None,  # dict keys: "both", "abs", "signed", "both_median_only", "hist"
    legend_title="noise (%)",
    # --- histogram controls ---
    plot_histograms=True,
    histogram_split="trainval",   # "train", "val", or "trainval"
    hist_alpha=0.20,              # shading alpha for min--max band
    hist_linewidth=1.8,
):
    """
    Median analogue of plot_binned_density_error_agg_groups_multi_data.

    Produces 4 plots:
      1) both abs (left axis) + signed (right axis), WITH min-max shading
      2) abs only (single axis), WITH min-max shading
      3) signed only (single axis), WITH min-max shading + 0 baseline
      4) both abs+signed, median lines only (NO shading)

    Optional 5th plot:
      5) per-group density histogram (counts per bin) computed from each TV split's
         data (train/val/train+val), with min--max shading across TV splits and
         the *average* number of data points per bin reported in the legend.

    Notes
    -----
    - Within-bin aggregation uses MEDIAN (abs + signed).
    - Across TV splits, central curve is nanmedian across splits; shading is min--max.
    - Histogram shading (min--max) is across TV splits (wrappers) based on their
      train/val data distributions (mw.y_train / mw.y_val).
    """

    group_names = list(modelWrappers_by_group.keys())

    # Map dataobjs into a dict keyed by group name
    if isinstance(dataobjs_by_group, dict):
        dataobj_map = dataobjs_by_group
    else:
        if len(dataobjs_by_group) != len(group_names):
            raise ValueError(
                f"dataobjs_by_group must have same length as groups. "
                f"Got {len(dataobjs_by_group)} vs {len(group_names)}."
            )
        dataobj_map = {g: dataobjs_by_group[i] for i, g in enumerate(group_names)}

    # Colors
    if group_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        group_colors = {g: cycle[i % len(cycle)] for i, g in enumerate(group_names)}
    elif isinstance(group_colors, list):
        group_colors = {g: group_colors[i % len(group_colors)] for i, g in enumerate(group_names)}
    else:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for i, g in enumerate(group_names):
            if g not in group_colors:
                group_colors[g] = cycle[i % len(cycle)]

    if save_names is None:
        save_names = {}

    # ---- error unit handling ----
    eu = str(error_units).lower().strip()
    if eu in {"%", "pct", "percent", "percentage"}:
        error_units = "percent"
    elif eu in {"abs", "absolute"}:
        error_units = "absolute"
    else:
        raise ValueError(f"error_units must be 'percent' or 'absolute' (got {error_units!r})")

    # Axis labels based on error_units
    if error_units == "percent":
        ylab_abs = "median abs. % error [%]"
        ylab_signed = "median signed % error [%]"
    else:
        ylab_abs = "median abs. error [cells mm$^{-2}]$"
        ylab_signed = "median signed error [cells mm$^{-2}]$"

    # ---------- GLOBAL bin edges (fair comparison across different density ranges) ----------
    all_u_ref = []
    for g in group_names:
        dataobj = dataobj_map[g]
        u = dataobj.u * K
        u_clean = dataobj.u_clean * K
        u_ref = u if bin_reference == "noisy" else u_clean
        all_u_ref.append(u_ref.reshape(-1))
    all_u_ref = np.concatenate(all_u_ref) if len(all_u_ref) else np.array([0.0])

    lo = float(np.nanmin(all_u_ref)) if bin_min is None else float(bin_min)
    hi = float(np.nanmax(all_u_ref)) if bin_max is None else float(bin_max)
    if np.isclose(lo, hi):
        lo, hi = lo - 0.5, hi + 0.5

    if bin_width is not None:
        bw = float(bin_width)
        if bw <= 0:
            raise ValueError(f"bin_width must be > 0 (got {bin_width})")
        bin_edges = np.arange(lo, hi + bw, bw)
        if bin_edges[-1] < hi:
            bin_edges = np.append(bin_edges, bin_edges[-1] + bw)
    else:
        bin_edges = np.linspace(lo, hi, num_bins + 1)

    nb = len(bin_edges) - 1
    dens_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_bar_width = bin_edges[1] - bin_edges[0]

    # denom used only for percent mode
    eps = 1e-12
    denom = np.maximum(np.abs(dens_centers), eps)

    # ---------- helpers ----------
    def binned_medians(u_ref_flat_local, signed_err_flat, edges):
        abs_err = np.abs(signed_err_flat)
        idx = np.digitize(u_ref_flat_local, edges) - 1

        med_abs = np.full(nb, np.nan)
        med_signed = np.full(nb, np.nan)

        for b in range(nb):
            m = idx == b
            if np.any(m):
                med_abs[b] = np.median(abs_err[m])
                med_signed[b] = np.median(signed_err_flat[m])
        return med_abs, med_signed

    def _extract_u_from_wrapper(mw, split):
        # matches your histogram function's extraction pattern
        if split == "train":
            xs, ys = mw.x_train, mw.y_train
        elif split == "val":
            xs, ys = mw.x_val, mw.y_val
        else:
            raise ValueError("split must be 'train' or 'val'")
        return np.array([k for (_, _), k in zip(xs, ys)], dtype=float)

    # ---------- Precompute aggregated curves per group ----------
    agg = {}      # errors
    hist_agg = {} # histogram counts (optional)

    for g in group_names:
        wrappers = modelWrappers_by_group[g]
        dataobj = dataobj_map[g]

        u = dataobj.u * K
        u_clean = dataobj.u_clean * K
        Nt, Nx = len(dataobj.t), len(dataobj.x)

        u_ref = u if bin_reference == "noisy" else u_clean
        u_ref_flat = u_ref.reshape(-1)

        inputs = torch.from_numpy(dataobj.inputs).float().to(device)

        abs_list, signed_list = [], []
        for mw in wrappers.values():
            model = mw.model.to(device).eval()
            with torch.no_grad():
                u_pred = model(inputs) * K
                u_pred = u_pred.reshape(Nx, Nt).cpu().numpy()

            signed_flat = (u_pred - u_clean).reshape(-1)
            med_abs, med_signed = binned_medians(u_ref_flat, signed_flat, bin_edges)

            if error_units == "percent":
                abs_vals = 100.0 * med_abs / denom
                signed_vals = 100.0 * med_signed / denom
            else:
                abs_vals = med_abs
                signed_vals = med_signed

            abs_list.append(abs_vals)
            signed_list.append(signed_vals)

        abs_arr = np.vstack(abs_list)
        signed_arr = np.vstack(signed_list)

        agg[g] = dict(
            dens_centers=dens_centers,
            abs_median=np.nanmedian(abs_arr, axis=0),
            abs_min=np.nanmin(abs_arr, axis=0),
            abs_max=np.nanmax(abs_arr, axis=0),
            signed_median=np.nanmedian(signed_arr, axis=0),
            signed_min=np.nanmin(signed_arr, axis=0),
            signed_max=np.nanmax(signed_arr, axis=0),
        )

        # ---------- histogram aggregation across TV splits ----------
        if plot_histograms:
            h_list = []
            for mw in wrappers.values():
                if histogram_split == "train":
                    u_hist = _extract_u_from_wrapper(mw, "train")
                elif histogram_split == "val":
                    u_hist = _extract_u_from_wrapper(mw, "val")
                elif histogram_split == "trainval":
                    u_hist = np.concatenate(
                        [_extract_u_from_wrapper(mw, "train"), _extract_u_from_wrapper(mw, "val")]
                    )
                else:
                    raise ValueError("histogram_split must be 'train', 'val', or 'trainval'")

                u_hist = u_hist * K
                h, _ = np.histogram(u_hist, bins=bin_edges)
                h_list.append(h.astype(float))

            h_arr = np.vstack(h_list)  # [n_splits, nb]
            hist_agg[g] = dict(
                centers=dens_centers,
                count_median=np.nanmedian(h_arr, axis=0),
                count_min=np.nanmin(h_arr, axis=0),
                count_max=np.nanmax(h_arr, axis=0),
                # average number of points per bin (across bins) for legend
                avg_points_per_bin=float(np.nanmean(np.nanmedian(h_arr, axis=0))),
            )

    # ---------- Plot 1: both axes + shading ----------
    fig1, ax1 = plt.subplots(figsize=fig_size)
    ax1b = ax1.twinx()
    ax1b.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

    for g in group_names:
        color = group_colors[g]
        d = agg[g]

        ax1.plot(
            d["dens_centers"], d["abs_median"],
            color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}"
        )
        #ax1.fill_between(d["dens_centers"], d["abs_min"], d["abs_max"], color=color, alpha=abs_band_alpha)

        ax1b.plot(
            d["dens_centers"], d["signed_median"],
            color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8,
            mfc=color, mec="black", mew=1.0
        )
        #ax1b.fill_between(d["dens_centers"], d["signed_min"], d["signed_max"], color=color, alpha=signed_band_alpha)

    ax1.set_xlabel(density_xlabel)
    ax1.set_ylabel(ylab_abs)
    ax1b.set_ylabel(ylab_signed)
    if binned_error_logy:
        ax1.set_yscale("log")
    ax1.legend(loc="best", fontsize=legend_fontsize, title=legend_title)
    ax1.set_facecolor("white")
    plt.tight_layout()
    if save_names.get("both"):
        plt.savefig(save_names["both"], dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    # ---------- Plot 2: abs only + shading ----------
    fig2, ax_abs = plt.subplots(figsize=fig_size)
    for g in group_names:
        color = group_colors[g]
        d = agg[g]
        ax_abs.plot(
            d["dens_centers"], d["abs_median"],
            color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}",
            mfc=color, mec="black", mew=1.0
        )
        #ax_abs.fill_between(d["dens_centers"], d["abs_min"], d["abs_max"], color=color, alpha=abs_band_alpha)

    ax_abs.set_xlabel(density_xlabel)
    ax_abs.set_ylabel(ylab_abs)
    if binned_error_logy:
        ax_abs.set_yscale("log")
    ax_abs.legend(loc="best", fontsize=legend_fontsize, title=legend_title)
    ax_abs.set_facecolor("white")
    plt.tight_layout()
    if save_names.get("abs"):
        plt.savefig(save_names["abs"], dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    # ---------- Plot 3: signed only + shading ----------
    fig3, ax_signed = plt.subplots(figsize=fig_size)
    ax_signed.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)
    signed_markers = ["^", "s", "p", "o"]
    for i, g in enumerate(group_names):
        color = group_colors[g]
        d = agg[g]
        ax_signed.plot(
            d["dens_centers"], d["signed_median"],
            color=color, linestyle=signed_linestyle, marker=signed_markers[i], lw=1.25, label=f"{g}",
            mfc=color, mec="black", mew=1.0
        )
        #ax_signed.fill_between(d["dens_centers"], d["signed_min"], d["signed_max"], color=color, alpha=signed_band_alpha)

    ax_signed.set_xlabel("Binned density [cells mm$^{-2}]$")
    ax_signed.set_ylabel(ylab_signed)
    ax_signed.legend(loc="best", fontsize=legend_fontsize, title=legend_title)
    ax_signed.set_facecolor("white")
    plt.tight_layout()
    if save_names.get("signed"):
        plt.savefig(save_names["signed"], dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    # ---------- Plot 4: both axes, median lines only (no shading) ----------
    fig4, ax4 = plt.subplots(figsize=fig_size)
    ax4b = ax4.twinx()
    ax4b.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

    for g in group_names:
        color = group_colors[g]
        d = agg[g]
        ax4.plot(
            d["dens_centers"], d["abs_median"],
            color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}"
        )
        ax4b.plot(
            d["dens_centers"], d["signed_median"],
            color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8
        )

    ax4.set_xlabel("Binned density [cells mm$^{-2}]$")
    ax4.set_ylabel(ylab_abs)
    ax4b.set_ylabel(ylab_signed)
    if binned_error_logy:
        ax4.set_yscale("log")
    ax4.legend(loc="best", fontsize=legend_fontsize, title=legend_title)
    ax4.set_facecolor("white")
    plt.tight_layout()
    if save_names.get("both_median_only"):
        plt.savefig(save_names["both_median_only"], dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()

    # ---------- Plot 5: histograms with min--max shading ----------
    fig5 = ax_hist = None
    if plot_histograms:
        fig5, ax_hist = plt.subplots(figsize=fig_size)

        for g in group_names:
            color = group_colors[g]
            h = hist_agg[g]

            # median counts as bars
            ax_hist.bar(
                h["centers"],
                h["count_median"],
                width=bin_bar_width,
                alpha=0.35,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=f"{g} (avg/bin={h['avg_points_per_bin']:.1f})",
            )
            # min--max band as fill between lines
            ax_hist.plot(h["centers"], h["count_median"], color=color, lw=hist_linewidth)
            ax_hist.fill_between(
                h["centers"], h["count_min"], h["count_max"],
                color=color, alpha=hist_alpha
            )

        ax_hist.set_xlabel("Binned density [cells mm$^{-2}]$")
        ax_hist.set_ylabel("data points per density bin")
        ax_hist.legend(loc="best", fontsize=legend_fontsize, title=legend_title)
        ax_hist.set_facecolor("white")
        plt.tight_layout()
        if save_names.get("hist"):
            plt.savefig(save_names["hist"], dpi=100, bbox_inches="tight", facecolor="None")
        plt.show()

    return (fig1, ax1, ax1b), (fig2, ax_abs), (fig3, ax_signed), (fig4, ax4, ax4b), (fig5, ax_hist)


def plot_binned_density_error_agg_groups_multi_data_mean(
    modelWrappers_by_group,
    dataobjs_by_group,
    *,
    device="cpu",
    K=1700,
    # --- binning: prefer spacing for fair comparison across different density ranges ---
    bin_width=None,            # e.g. 50.0 (cells mm^-2). If None, falls back to num_bins.
    num_bins=10,               # fallback for legacy behaviour
    bin_reference="clean",     # "clean" or "noisy"
    bin_min=None,              # optional override for global lower edge (after scaling by K)
    bin_max=None,              # optional override for global upper edge (after scaling by K)
    error_units="percent",     # "percent" (default) or "absolute"
    binned_error_logy=False,
    fig_size=(7, 5),
    group_colors=None,
    abs_linestyle="-",
    signed_linestyle=":",
    baseline_linestyle="--",
    abs_marker="o",
    signed_marker="s",
    abs_band_alpha=0.20,
    signed_band_alpha=0.15,
    legend_fontsize=9,
    axis_fontsizes=None,
    seed_index=None,
    fill=True,
    grid=True,
    save_names=None,  # dict keys: "both", "abs", "signed", "both_mean_only", "hist"
    legend_title="noise (%)",
    # --- histogram controls ---
    plot_histograms=True,
    histogram_split="trainval",   # "train", "val", or "trainval"
    hist_alpha=0.20,              # shading alpha for min--max band
    hist_linewidth=1.8,
    square_error=False
):
    """
    mean analogue of plot_binned_density_error_agg_groups_multi_data.

    Produces 4 plots:
      1) both abs (left axis) + signed (right axis), WITH min-max shading
      2) abs only (single axis), WITH min-max shading
      3) signed only (single axis), WITH min-max shading + 0 baseline
      4) both abs+signed, mean lines only (NO shading)

    Optional 5th plot:
      5) per-group density histogram (counts per bin) computed from each TV split's
         data (train/val/train+val), with min--max shading across TV splits and
         the *average* number of data points per bin reported in the legend.

    Notes
    -----
    - Within-bin aggregation uses mean (abs + signed).
    - Across TV splits, central curve is nanmean across splits; shading is min--max.
    - Histogram shading (min--max) is across TV splits (wrappers) based on their
      train/val data distributions (mw.y_train / mw.y_val).
    - If `seed_index` is not None, only that TV split is used for each group and
      fill is disabled automatically.
    - If `square_error` is True, squared residual loss is plotted instead of raw
      differences, so the "signed" and "abs" quantities coincide.
    """

    group_names = list(modelWrappers_by_group.keys())

    # Map dataobjs into a dict keyed by group name
    if isinstance(dataobjs_by_group, dict):
        dataobj_map = dataobjs_by_group
    else:
        if len(dataobjs_by_group) != len(group_names):
            raise ValueError(
                f"dataobjs_by_group must have same length as groups. "
                f"Got {len(dataobjs_by_group)} vs {len(group_names)}."
            )
        dataobj_map = {g: dataobjs_by_group[i] for i, g in enumerate(group_names)}

    # Colors
    if group_colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        group_colors = {g: cycle[i % len(cycle)] for i, g in enumerate(group_names)}
    elif isinstance(group_colors, list):
        group_colors = {g: group_colors[i % len(group_colors)] for i, g in enumerate(group_names)}
    else:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for i, g in enumerate(group_names):
            if g not in group_colors:
                group_colors[g] = cycle[i % len(cycle)]

    plot_keys = ("both", "abs", "signed", "both_mean_only", "hist")
    if save_names is None:
        save_names = {}
        enabled_plots = set(plot_keys)
    else:
        enabled_plots = {key for key in plot_keys if save_names.get(key)}
        if not enabled_plots:
            enabled_plots = set(plot_keys)

    axis_fontsizes = {
        "xaxis": 12,
        "xtick_labels": 10,
        "yaxis": 12,
        "ytick_labels": 10,
        **(axis_fontsizes or {}),
    }

    if seed_index is not None:
        fill = False

    # ---- error unit handling ----
    eu = str(error_units).lower().strip()
    if eu in {"%", "pct", "percent", "percentage"}:
        error_units = "percent"
    elif eu in {"abs", "absolute"}:
        error_units = "absolute"
    else:
        raise ValueError(f"error_units must be 'percent' or 'absolute' (got {error_units!r})")

    # Axis labels based on error_units
    if square_error:
        if error_units == "percent":
            ylab_abs = "Data loss [a.u.]"
            ylab_signed = "Data loss [a.u.]"
        else:
            ylab_abs = "Data loss [cells$^2$ mm$^{-4}$]"
            ylab_signed = "Data loss [cells$^2$ mm$^{-4}$]"
    elif error_units == "percent":
        ylab_abs = "Mean abs. % difference [%]"
        ylab_signed = "Mean % difference [%]"
    else:
        ylab_abs = "Mean abs. difference [cells mm$^{-2}]$"
        ylab_signed = r"$Mean \{\hat{u} - u_{data}\}$ [cells mm$^{-2}]$"

    density_xlabel = "Binned density [cells mm$^{-2}]$"

    def _apply_grid(ax):
        if not grid:
            return
        ax.set_axisbelow(True)
        ax.grid(True, which="major", linestyle="-", alpha=0.4)

    def _select_wrappers_for_seed(wrapper_dict, group_name):
        wrapper_values = list(wrapper_dict.values())
        if seed_index is None:
            return wrapper_values
        if seed_index < 0 or seed_index >= len(wrapper_values):
            raise IndexError(
                f"seed_index={seed_index} out of range for group {group_name!r} "
                f"with {len(wrapper_values)} TV splits."
            )
        return [wrapper_values[seed_index]]

    # ---------- GLOBAL bin edges (fair comparison across different density ranges) ----------
    all_u_ref = []
    for g in group_names:
        dataobj = dataobj_map[g]
        u = dataobj.u * K
        u_clean = dataobj.u_clean * K
        u_ref = u if bin_reference == "noisy" else u_clean
        all_u_ref.append(u_ref.reshape(-1))
    all_u_ref = np.concatenate(all_u_ref) if len(all_u_ref) else np.array([0.0])

    lo = float(np.nanmin(all_u_ref)) if bin_min is None else float(bin_min)
    hi = float(np.nanmax(all_u_ref)) if bin_max is None else float(bin_max)
    if np.isclose(lo, hi):
        lo, hi = lo - 0.5, hi + 0.5

    if bin_width is not None:
        bw = float(bin_width)
        if bw <= 0:
            raise ValueError(f"bin_width must be > 0 (got {bin_width})")
        bin_edges = np.arange(lo, hi + bw, bw)
        if bin_edges[-1] < hi:
            bin_edges = np.append(bin_edges, bin_edges[-1] + bw)
    else:
        bin_edges = np.linspace(lo, hi, num_bins + 1)

    nb = len(bin_edges) - 1
    dens_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_bar_width = bin_edges[1] - bin_edges[0]

    # denom used only for percent mode
    eps = 1e-12
    denom = np.maximum(np.abs(dens_centers), eps)

    # ---------- helpers ----------
    def binned_means(u_ref_flat_local, signed_err_flat, edges):
        abs_err = np.abs(signed_err_flat)
        sq_err = np.square(signed_err_flat)
        idx = np.digitize(u_ref_flat_local, edges) - 1

        med_abs = np.full(nb, np.nan)
        med_signed = np.full(nb, np.nan)

        for b in range(nb):
            m = idx == b
            if np.any(m):
                if square_error:
                    med_abs[b] = np.mean(sq_err[m])
                    med_signed[b] = np.mean(sq_err[m])
                else:
                    med_abs[b] = np.mean(abs_err[m])
                    med_signed[b] = np.mean(signed_err_flat[m])
        return med_abs, med_signed

    def _extract_u_from_wrapper(mw, split):
        # matches your histogram function's extraction pattern
        if split == "train":
            xs, ys = mw.x_train, mw.y_train
        elif split == "val":
            xs, ys = mw.x_val, mw.y_val
        else:
            raise ValueError("split must be 'train' or 'val'")
        return np.array([k for (_, _), k in zip(xs, ys)], dtype=float)

    # ---------- Precompute aggregated curves per group ----------
    agg = {}      # errors
    hist_agg = {} # histogram counts (optional)

    for g in group_names:
        wrappers = _select_wrappers_for_seed(modelWrappers_by_group[g], g)
        dataobj = dataobj_map[g]

        u = dataobj.u * K
        u_clean = dataobj.u_clean * K
        Nt, Nx = len(dataobj.t), len(dataobj.x)

        u_ref = u if bin_reference == "noisy" else u_clean
        u_ref_flat = u_ref.reshape(-1)

        inputs = torch.from_numpy(dataobj.inputs).float().to(device)

        abs_list, signed_list = [], []
        for mw in wrappers:
            model = mw.model.to(device).eval()
            with torch.no_grad():
                u_pred = model(inputs) * K
                u_pred = u_pred.reshape(Nx, Nt).cpu().numpy()

            signed_flat = (u_pred - u_clean).reshape(-1)
            med_abs, med_signed = binned_means(u_ref_flat, signed_flat, bin_edges)

            if error_units == "percent":
                if square_error:
                    abs_vals = np.square(100.0 * np.sqrt(med_abs) / denom)
                    signed_vals = np.square(100.0 * np.sqrt(med_signed) / denom)
                else:
                    abs_vals = 100.0 * med_abs / denom
                    signed_vals = 100.0 * med_signed / denom
            else:
                abs_vals = med_abs
                signed_vals = med_signed

            abs_list.append(abs_vals)
            signed_list.append(signed_vals)

        abs_arr = np.vstack(abs_list)
        signed_arr = np.vstack(signed_list)

        agg[g] = dict(
            dens_centers=dens_centers,
            abs_mean=np.nanmean(abs_arr, axis=0),
            abs_min=np.nanmin(abs_arr, axis=0),
            abs_max=np.nanmax(abs_arr, axis=0),
            signed_mean=np.nanmean(signed_arr, axis=0),
            signed_min=np.nanmin(signed_arr, axis=0),
            signed_max=np.nanmax(signed_arr, axis=0),
        )

        # ---------- histogram aggregation across TV splits ----------
        if plot_histograms:
            h_list = []
            for mw in wrappers:
                if histogram_split == "train":
                    u_hist = _extract_u_from_wrapper(mw, "train")
                elif histogram_split == "val":
                    u_hist = _extract_u_from_wrapper(mw, "val")
                elif histogram_split == "trainval":
                    u_hist = np.concatenate(
                        [_extract_u_from_wrapper(mw, "train"), _extract_u_from_wrapper(mw, "val")]
                    )
                else:
                    raise ValueError("histogram_split must be 'train', 'val', or 'trainval'")

                u_hist = u_hist * K
                h, _ = np.histogram(u_hist, bins=bin_edges)
                h_list.append(h.astype(float))

            h_arr = np.vstack(h_list)  # [n_splits, nb]
            hist_agg[g] = dict(
                centers=dens_centers,
                count_mean=np.nanmean(h_arr, axis=0),
                count_min=np.nanmin(h_arr, axis=0),
                count_max=np.nanmax(h_arr, axis=0),
                # average number of points per bin (across bins) for legend
                avg_points_per_bin=float(np.nanmean(np.nanmean(h_arr, axis=0))),
            )

    fig1 = ax1 = ax1b = None
    if "both" in enabled_plots:
        # ---------- Plot 1: both axes + shading ----------
        fig1, ax1 = plt.subplots(figsize=fig_size)
        ax1b = ax1.twinx()
        ax1b.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

        for g in group_names:
            color = group_colors[g]
            d = agg[g]

            ax1.plot(
                d["dens_centers"], d["abs_mean"],
                color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}"
            )
            if fill:
                ax1.fill_between(
                    d["dens_centers"], d["abs_min"], d["abs_max"], color=color, alpha=abs_band_alpha
                )

            ax1b.plot(
                d["dens_centers"], d["signed_mean"],
                color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8,
                mfc=color, mec="black", mew=1.0
            )
            if fill:
                ax1b.fill_between(
                    d["dens_centers"], d["signed_min"], d["signed_max"], color=color, alpha=signed_band_alpha
                )

        ax1.set_xlabel(density_xlabel, fontsize=axis_fontsizes["xaxis"])
        ax1.set_ylabel(ylab_abs, fontsize=axis_fontsizes["yaxis"])
        ax1b.set_ylabel(ylab_signed, fontsize=axis_fontsizes["yaxis"])
        ax1.tick_params(axis="x", labelsize=axis_fontsizes["xtick_labels"])
        ax1.tick_params(axis="y", labelsize=axis_fontsizes["ytick_labels"])
        ax1b.tick_params(axis="y", labelsize=axis_fontsizes["ytick_labels"])
        if binned_error_logy:
            ax1.set_yscale("log")
        _apply_grid(ax1)
        ax1.legend(
            loc="best",
            fontsize=legend_fontsize,
            title=legend_title,
            title_fontsize=legend_fontsize,
        )
        ax1.set_facecolor("white")
        plt.tight_layout()
        if save_names.get("both"):
            plt.savefig(save_names["both"], dpi=100, bbox_inches="tight", facecolor="None")
        plt.show()

    fig2 = ax_abs = None
    if "abs" in enabled_plots:
        # ---------- Plot 2: abs only + shading ----------
        fig2, ax_abs = plt.subplots(figsize=fig_size)
        for g in group_names:
            color = group_colors[g]
            d = agg[g]
            ax_abs.plot(
                d["dens_centers"], d["abs_mean"],
                color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}",
                mfc=color, mec="black", mew=1.0
            )
            if fill:
                ax_abs.fill_between(
                    d["dens_centers"], d["abs_min"], d["abs_max"], color=color, alpha=abs_band_alpha
                )

        ax_abs.set_xlabel(density_xlabel, fontsize=axis_fontsizes["xaxis"])
        ax_abs.set_ylabel(ylab_abs, fontsize=axis_fontsizes["yaxis"])
        ax_abs.tick_params(axis="x", labelsize=axis_fontsizes["xtick_labels"])
        ax_abs.tick_params(axis="y", labelsize=axis_fontsizes["ytick_labels"])
        if binned_error_logy:
            ax_abs.set_yscale("log")
        _apply_grid(ax_abs)
        ax_abs.legend(
            loc="best",
            fontsize=legend_fontsize,
            title=legend_title,
            title_fontsize=legend_fontsize,
        )
        ax_abs.set_facecolor("white")
        plt.tight_layout()
        if save_names.get("abs"):
            plt.savefig(save_names["abs"], dpi=100, bbox_inches="tight", facecolor="None")
            print("saved plot:", save_names["abs"])
        plt.show()

    # Single-axis plots should use solid lines for cleaner within-panel comparison.
    single_axis_linestyle = "-"

    fig3 = ax_signed = None
    if "signed" in enabled_plots:
        # ---------- Plot 3: signed only + shading ----------
        fig3, ax_signed = plt.subplots(figsize=fig_size)
        ax_signed.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)
        signed_markers = ["^", "s", "p", "o"]
        for i, g in enumerate(group_names):
            color = group_colors[g]
            d = agg[g]
            ax_signed.plot(
                d["dens_centers"], d["signed_mean"],
                color=color, linestyle=single_axis_linestyle, marker=signed_markers[i], lw=1.25, label=f"{g}",
                mfc=color, mec="black", mew=1.0
            )
            if fill:
                ax_signed.fill_between(
                    d["dens_centers"], d["signed_min"], d["signed_max"], color=color, alpha=signed_band_alpha
                )

        ax_signed.set_xlabel(density_xlabel, fontsize=axis_fontsizes["xaxis"])
        ax_signed.set_ylabel(ylab_signed, fontsize=axis_fontsizes["yaxis"])
        ax_signed.tick_params(axis="x", labelsize=axis_fontsizes["xtick_labels"])
        ax_signed.tick_params(axis="y", labelsize=axis_fontsizes["ytick_labels"])
        _apply_grid(ax_signed)
        ax_signed.legend(
            loc="best",
            fontsize=legend_fontsize,
            title=legend_title,
            title_fontsize=legend_fontsize,
        )
        ax_signed.set_facecolor("white")
        plt.tight_layout()
        if save_names.get("signed"):
            plt.savefig(save_names["signed"], dpi=100, bbox_inches="tight", facecolor="None")
            print("saved plot:", save_names["signed"])
        plt.show()

    fig4 = ax4 = ax4b = None
    if "both_mean_only" in enabled_plots:
        # ---------- Plot 4: both axes, mean lines only (no shading) ----------
        fig4, ax4 = plt.subplots(figsize=fig_size)
        ax4b = ax4.twinx()
        ax4b.axhline(0, color="k", linestyle=baseline_linestyle, lw=1)

        for g in group_names:
            color = group_colors[g]
            d = agg[g]
            ax4.plot(
                d["dens_centers"], d["abs_mean"],
                color=color, linestyle=abs_linestyle, marker=abs_marker, lw=1.8, label=f"{g}"
            )
            ax4b.plot(
                d["dens_centers"], d["signed_mean"],
                color=color, linestyle=signed_linestyle, marker=signed_marker, lw=1.8
            )

        ax4.set_xlabel(density_xlabel, fontsize=axis_fontsizes["xaxis"])
        ax4.set_ylabel(ylab_abs, fontsize=axis_fontsizes["yaxis"])
        ax4b.set_ylabel(ylab_signed, fontsize=axis_fontsizes["yaxis"])
        ax4.tick_params(axis="x", labelsize=axis_fontsizes["xtick_labels"])
        ax4.tick_params(axis="y", labelsize=axis_fontsizes["ytick_labels"])
        ax4b.tick_params(axis="y", labelsize=axis_fontsizes["ytick_labels"])
        if binned_error_logy:
            ax4.set_yscale("log")
        _apply_grid(ax4)
        ax4.legend(
            loc="best",
            fontsize=legend_fontsize,
            title=legend_title,
            title_fontsize=legend_fontsize,
        )
        ax4.set_facecolor("white")
        plt.tight_layout()
        if save_names.get("both_mean_only"):
            plt.savefig(save_names["both_mean_only"], dpi=100, bbox_inches="tight", facecolor="None")
            print("saved plot:", save_names["both_mean_only"])
        plt.show()

    # ---------- Plot 5: histograms with min--max shading ----------
    fig5 = ax_hist = None
    if plot_histograms and "hist" in enabled_plots:
        fig5, ax_hist = plt.subplots(figsize=fig_size)

        for g in group_names:
            color = group_colors[g]
            h = hist_agg[g]

            # mean counts as bars
            ax_hist.bar(
                h["centers"],
                h["count_mean"],
                width=bin_bar_width,
                alpha=0.35,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=f"{g} (avg/bin={h['avg_points_per_bin']:.1f})",
            )
            # min--max band as fill between lines
            ax_hist.plot(h["centers"], h["count_mean"], color=color, lw=hist_linewidth)
            if fill:
                ax_hist.fill_between(
                    h["centers"], h["count_min"], h["count_max"],
                    color=color, alpha=hist_alpha
                )

        ax_hist.set_xlabel(density_xlabel, fontsize=axis_fontsizes["xaxis"])
        ax_hist.set_ylabel("Data points per density bin", fontsize=axis_fontsizes["yaxis"])
        ax_hist.tick_params(axis="x", labelsize=axis_fontsizes["xtick_labels"])
        ax_hist.tick_params(axis="y", labelsize=axis_fontsizes["ytick_labels"])
        _apply_grid(ax_hist)
        ax_hist.legend(
            loc="best",
            fontsize=legend_fontsize,
            title=legend_title,
            title_fontsize=legend_fontsize,
        )
        ax_hist.set_facecolor("white")
        plt.tight_layout()
        if save_names.get("hist"):
            plt.savefig(save_names["hist"], dpi=100, bbox_inches="tight", facecolor="None")
            print("saved plot:", save_names["hist"])
        plt.show()

    return (fig1, ax1, ax1b), (fig2, ax_abs), (fig3, ax_signed), (fig4, ax4, ax4b), (fig5, ax_hist)
