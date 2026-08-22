"""Real-space plotting helpers for one-dimensional learned functions.

Contents
--------
- symbolic_from_function
- plot_eval_D_multi_fixedD
- plot_eval_G_multi
- plot_eval_D_multi_gray
- plot_eval_D_multi_gray_gridLayout
- plot_eval_G_multi_gray
- plot_eval_D_multi
- plot_eval_D_multi_fixedD
- plot_eval_G_multi
- plot_eval_G_multi_fixedG"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import sympy as sp
import ast
import inspect
import itertools


from .utils import to_torch, hist_properties


def symbolic_from_function(func, var_name='u'):
    # Get source code
    source = inspect.getsource(func).strip()

    # Parse the function's AST
    tree = ast.parse(source)

    # Get the return statement's expression
    return_node = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Return)
    )

    # Create a mapping for allowed names (e.g., math, np, etc.)
    allowed_names = func.__globals__.copy()

    # Create symbolic variable
    u = sp.Symbol(var_name)

    # Evaluate the return expression in symbolic context
    expr = eval(
        compile(ast.Expression(return_node.value), "<ast>", "eval"),
        {**allowed_names, var_name: u}
    )

    return expr




def plot_eval_D_multi_fixedD(
    modelWrapper_dics,
    dataobjs,
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
    legend_title = None,
    legend_ncols=2,
    legend_fontsize=12,
    linestyles=itertools.cycle(["-", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]),
    errs=None,
):
    """
    Plot diffusion D(u) ensembles + true curves for multiple configs / datasets.

    modelWrapper_dics : dict
        {config_key: {seed: wrapper, ...}, ...}
    dataobjs : list or single
        One data object per configuration (used for 5–95% u-percentiles).
    D_sym_true_lst : list
        Symbolic / reference true D(u) for each config (same ordering as dataobjs).
    colors, labels : list
        Colours and labels per configuration.
    errs : (err_up, err_low) or None
        Percentage error bands around the true D(u).
    """
    if errs:
        err_up, err_low = errs
    else:
        err_up = err_low = None

    # Allow passing a single dataobj
    if not isinstance(dataobjs, (list, tuple)):
        dataobjs = [dataobjs]

    colors = colors[: len(labels)]

    # Compute 5–95% u-percentiles for each data object
    hist_props_list = [hist_properties(d, num_bins) for d in dataobjs]
    low_us = [hp["low_count"] for hp in hist_props_list]
    high_us = [hp["high_count"] for hp in hist_props_list]

    results = []
    D_true_lst = []
    u_grids = []
    u_grids_K = []

    # ------------------------------------------------
    # Loop over configurations
    # ------------------------------------------------
    for wrapper_dic, color, label in zip(modelWrapper_dics.values(), colors, labels):
        # Sample model for THIS configuration
        sample_model = list(wrapper_dic.values())[0].model
        u_vals_torch = sample_model.u_vals_torch
        u_vals_np = sample_model.u_vals.flatten()

        u_grids.append(u_vals_np)
        u_grids_K.append(u_vals_np * K)

        diffusion_errors = []
        D_ensemble = []

        for i, wrapper in enumerate(wrapper_dic.values()):
            D_true_check = list(wrapper.model.D_true)
            if i == 0:# and D_true_check not in D_true_lst:
                D_true_lst.append(D_true_check)

            model = wrapper.model
            model.eval()

            with torch.no_grad():
                diff_pred = model.D_scale * model.diffusion(model.u_vals_torch).flatten()
                diffusion_error = (model.D_true_torch - diff_pred) ** 2

                diffusion_errors.append(diffusion_error.unsqueeze(0))  # [1, N]
                D_ensemble.append(diff_pred.unsqueeze(0))              # [1, N]

        D_ensemble = torch.cat(D_ensemble, dim=0)  # [runs, N]
        D_mean = D_ensemble.mean(0).cpu().numpy()
        D_min = D_ensemble.min(0).values.cpu().numpy()
        D_max = D_ensemble.max(0).values.cpu().numpy()

        diffusion_errors = torch.cat(diffusion_errors, dim=0)
        mse = diffusion_errors.mean().item()

        results.append((mse, label, color, D_mean, D_min, D_max))

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    # Ensemble predictions for each configuration
    for j, (mse, label, color, D_mean, D_min, D_max) in enumerate(results):
        # handle linestyles whether it's a list or cycle
        try:
            ls = linestyles[j]
        except TypeError:
            ls = next(linestyles)

        u_grid_K = u_grids_K[j]

        ax.plot(u_grid_K, D_mean, lw=3, color=color, linestyle=ls, label=label)
        if fill:
            ax.fill_between(u_grid_K, D_min, D_max, alpha=0.4, color=color)

    # ------------------------------------------------
    # True curves + error bands + color-coded percentile lines
    # ------------------------------------------------
    for i, (D_true, D_sym, low_u, high_u, u_grid, u_grid_K) in enumerate(
        zip(D_true_lst, D_sym_true_lst, low_us, high_us, u_grids, u_grids_K)
    ):
        D_true = np.array(D_true)
        col = colors[i % len(colors)]

        # True curve
        if i == 0:
            ax.plot(
                u_grid_K,
                D_true,
                "--",
                lw=1,
                color="k",
                label=f"$D_{Dnum}$",
                zorder=4,
            )

        # Error bands around true curve (all i, but only first with legend labels)
        if errs is not None:
            if i == 0:
                ax.plot(
                    u_grid_K,
                    D_true * (1 + err_up / 100),
                    "--",
                    lw=1,
                    color="r",
                    label=f"{err_up}%",
                    zorder=3,
                )
                ax.plot(
                    u_grid_K,
                    D_true * (1 - err_low / 100),
                    "-.",
                    lw=1,
                    color="r",
                    label=f"{err_low}%",
                    zorder=3,
                )
            else:
                ax.plot(
                    u_grid_K,
                    D_true * (1 + err_up / 100),
                    "--",
                    lw=1,
                    color="r",
                    zorder=3,
                )
                ax.plot(
                    u_grid_K,
                    D_true * (1 - err_low / 100),
                    "-.",
                    lw=1,
                    color="r",
                    zorder=3,
                )


    ax.set_xlabel(r"Cell density [cells mm$^{-2}]$", fontsize=11)
    ax.set_ylabel(r"Diffusion [mm$^2$ days$^{-1}$]", fontsize=11)
    ax.set_facecolor("white")
    ax.legend(
        loc="center",
        bbox_to_anchor=legend_pos,
        ncols=legend_ncols,
        fontsize=legend_fontsize,
        title = legend_title,
    )

    plt.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()


def plot_eval_G_multi(
    modelWrapper_dics,
    dataobjs,
    G_sym_true_lst,
    colors,
    labels,
    Gnum=1,
    device="cpu",
    num_bins=50,
    K=1700,
    name=None,
    fill=True,
    legend_pos=(0.5, 0.5),
    legend_ncols=2,
    legend_fontsize=12,
    legend_title = None,
    linestyles=itertools.cycle(["-", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]),
    errs=None,
    xlim = None,
):
    """
    Plot growth G(u) ensembles + true curves for multiple configs / datasets.

    modelWrapper_dics : dict
        {config_key: {seed: wrapper, ...}, ...}
    dataobjs : list or single
        One data object per configuration (used for 5–95% u-percentiles).
    G_sym_true_lst : list
        Symbolic / reference true G(u) for each config (same ordering as dataobjs).
    errs : (err_up, err_low) or None
        Percentage error bands around the true G(u).
    """
    if errs:
        err_up, err_low = errs
    else:
        err_up = err_low = None

    if not isinstance(dataobjs, (list, tuple)):
        dataobjs = [dataobjs]

    colors = colors[: len(labels)]

    # Percentiles for each dataset
    hist_props_list = [hist_properties(d, num_bins) for d in dataobjs]
    low_us = [hp["low_count"] for hp in hist_props_list]
    high_us = [hp["high_count"] for hp in hist_props_list]

    results = []
    G_true_lst = []
    u_grids = []
    u_grids_K = []

    # ------------------------------------------------
    # Loop over configurations
    # ------------------------------------------------
    for wrapper_dic, color, label in zip(modelWrapper_dics.values(), colors, labels):
        # Sample model for THIS configuration
        sample_model = list(wrapper_dic.values())[0].model
        u_vals_torch = sample_model.u_vals_torch
        u_vals_np = sample_model.u_vals.flatten()

        u_grids.append(u_vals_np)
        u_grids_K.append(u_vals_np * K)

        G_ensemble = []
        growth_errors = []

        for i, wrapper in enumerate(wrapper_dic.values()):
            G_true_check = list(wrapper.model.G_true)
            if i == 0:# and G_true_check not in G_true_lst:
                G_true_lst.append(G_true_check)

            model = wrapper.model
            model.eval()

            with torch.no_grad():
                grow_pred = model.G_scale * model.growth(model.u_vals_torch).flatten()
                growth_error = (model.G_true_torch - grow_pred) ** 2

                growth_errors.append(growth_error.unsqueeze(0))  # [1, N]
                G_ensemble.append(grow_pred.unsqueeze(0))        # [1, N]

        G_ensemble = torch.cat(G_ensemble, dim=0)
        G_mean = G_ensemble.mean(0).cpu().numpy()
        G_min = G_ensemble.min(0).values.cpu().numpy()
        G_max = G_ensemble.max(0).values.cpu().numpy()

        growth_errors = torch.cat(growth_errors, dim=0)
        mse = growth_errors.mean().item()

        results.append((mse, label, color, G_mean, G_min, G_max))

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    # Ensemble curves
    for j, (mse, label, color, G_mean, G_min, G_max) in enumerate(results):
        try:
            ls = linestyles[j]
        except TypeError:
            ls = next(linestyles)

        u_grid_K = u_grids_K[j]
        ax.plot(u_grid_K, G_mean, lw=3, color=color, linestyle=ls, label=label)
        if fill:
            ax.fill_between(u_grid_K, G_min, G_max, alpha=0.4, color=color)

    # ------------------------------------------------
    # True curves + error bands + color-coded percentile lines
    # ------------------------------------------------
    for i, (G_true, G_sym, low_u, high_u, u_grid, u_grid_K) in enumerate(
        zip(G_true_lst, G_sym_true_lst, low_us, high_us, u_grids, u_grids_K)
    ):
        G_true = np.array(G_true)
        col = colors[i % len(colors)]

        # True curve
        if i == 0:
            ax.plot(
                u_grid_K,
                G_true,
                "--",
                lw=1,
                color="k",
                label=f"$G_{Gnum}$",
                zorder=4,
            )
        else:
            ax.plot(
                u_grid_K,
                G_true,
                "--",
                lw=1,
                color="k",
                zorder=4,
            )

        # Error bands, for all i but only first with labels
        if errs is not None:
            if i == 0:
                ax.plot(
                    u_grid_K,
                    G_true * (1 + err_up / 100),
                    "--",
                    lw=1,
                    color="r",
                    label=f"{err_up}%",
                    zorder=3,
                )
                ax.plot(
                    u_grid_K,
                    G_true * (1 - err_low / 100),
                    "-.",
                    lw=1,
                    color="r",
                    label=f"{err_low}%",
                    zorder=3,
                )
            else:
                ax.plot(
                    u_grid_K,
                    G_true * (1 + err_up / 100),
                    "--",
                    lw=1,
                    color="r",
                    zorder=3,
                )
                ax.plot(
                    u_grid_K,
                    G_true * (1 - err_low / 100),
                    "-.",
                    lw=1,
                    color="r",
                    zorder=3,
                )


    ax.set_xlabel(r"Cell density [cells mm$^{-2}]$", fontsize=11)
    ax.set_ylabel(r"Growth [mm$^2$ days$^{-1}$]", fontsize=11)
    ax.set_facecolor("white")
    ax.legend(
        loc="center",
        bbox_to_anchor=legend_pos,
        ncols=legend_ncols,
        fontsize=legend_fontsize,
    )

    plt.tight_layout()

    if xlim:
        ax.set_xlim(xlim)
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()


def plot_eval_D_multi_gray(
    modelWrapper_dics,
    dataobjs,
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
    legend_ncols=2,
    legend_fontsize=12,
    legend_title = None,
    linestyles=itertools.cycle(["-", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]),
    errs=None,
    figsize=(7, 5),
    axis_fontsizes=None,
    ylim=None,
    y_lim=None,
):
    """
    Plot diffusion D(u) ensembles + true curves for multiple configs / datasets.

    modelWrapper_dics : dict
        {config_key: {seed: wrapper, ...}, ...}
    dataobjs : list or single
        One data object per configuration (used for 5–95% u-percentiles).
    D_sym_true_lst : list
        Symbolic / reference true D(u) for each config (same ordering as dataobjs).
    colors, labels : list
        Colours and labels per configuration.
    errs : (err_up, err_low) or None
        Percentage error bands around the true D(u).
    axis_fontsizes : dict or None
        Optional axis font-size controls with keys "xaxis", "xtick_labels",
        "yaxis", and "ytick_labels".
    ylim, y_lim : tuple or None
        Optional y-axis limits. `y_lim` is accepted as a backward-compatible alias.
    """
    if ylim is None:
        ylim = y_lim

    axis_fontsizes = axis_fontsizes or {}
    xaxis_fontsize = axis_fontsizes.get("xaxis", 11)
    xtick_fontsize = axis_fontsizes.get("xtick_labels", xaxis_fontsize)
    yaxis_fontsize = axis_fontsizes.get("yaxis", 11)
    ytick_fontsize = axis_fontsizes.get("ytick_labels", yaxis_fontsize)

    if errs:
        err_up, err_low = errs
    else:
        err_up = err_low = None

    # Allow passing a single dataobj
    if not isinstance(dataobjs, (list, tuple)):
        dataobjs = [dataobjs]

    colors = colors[: len(labels)]

    # Compute 5–95% u-percentiles for each data object
    hist_props_list = [hist_properties(d, num_bins) for d in dataobjs]
    low_us = [hp["low_count"] for hp in hist_props_list]
    high_us = [hp["high_count"] for hp in hist_props_list]

    results = []
    D_true_lst = []
    u_grids = []
    u_grids_K = []

    # ------------------------------------------------
    # Loop over configurations
    # ------------------------------------------------
    for wrapper_dic, color, label in zip(modelWrapper_dics.values(), colors, labels):
        # Sample model for THIS configuration
        sample_model = list(wrapper_dic.values())[0].model
        u_vals_torch = sample_model.u_vals_torch
        u_vals_np = sample_model.u_vals.flatten()

        u_grids.append(u_vals_np)
        u_grids_K.append(u_vals_np * K)

        diffusion_errors = []
        D_ensemble = []

        for i, wrapper in enumerate(wrapper_dic.values()):
            D_true_check = list(wrapper.model.D_true)
            if i == 0:  # and D_true_check not in D_true_lst:
                D_true_lst.append(D_true_check)

            model = wrapper.model
            model.eval()

            with torch.no_grad():
                diff_pred = model.D_scale * model.diffusion(model.u_vals_torch).flatten()
                diffusion_error = (model.D_true_torch - diff_pred) ** 2

                diffusion_errors.append(diffusion_error.unsqueeze(0))  # [1, N]
                D_ensemble.append(diff_pred.unsqueeze(0))              # [1, N]

        D_ensemble = torch.cat(D_ensemble, dim=0)  # [runs, N]
        D_mean = D_ensemble.mean(0).cpu().numpy()
        D_min = D_ensemble.min(0).values.cpu().numpy()
        D_max = D_ensemble.max(0).values.cpu().numpy()

        diffusion_errors = torch.cat(diffusion_errors, dim=0)
        mse = diffusion_errors.mean().item()

        results.append((mse, label, color, D_mean, D_min, D_max))

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Ensemble predictions for each configuration
    for j, (mse, label, color, D_mean, D_min, D_max) in enumerate(results):
        # handle linestyles whether it's a list or cycle
        try:
            ls = linestyles[j]
        except TypeError:
            ls = next(linestyles)

        u_grid_K = u_grids_K[j]

        ax.plot(u_grid_K, D_mean, lw=3, color=color, linestyle=ls, label=label)
        if fill:
            ax.fill_between(u_grid_K, D_min, D_max, alpha=0.4, color=color)

    # ------------------------------------------------
    # True curves + error bands + color-coded percentile lines
    # ------------------------------------------------
    for i, (D_true, D_sym, low_u, high_u, u_grid, u_grid_K) in enumerate(
        zip(D_true_lst, D_sym_true_lst, low_us, high_us, u_grids, u_grids_K)
    ):
        D_true = np.array(D_true)
        col = colors[i % len(colors)]

        # True curve
        if i == 0:
            ax.plot(
                u_grid_K,
                D_true,
                "--",
                lw=1,
                color="k",
                label=f"$D_{Dnum}$",
                zorder=4,
            )
        else:
            ax.plot(
                u_grid_K,
                D_true,
                "--",
                lw=1,
                color="k",
                zorder=4,
            )

        # Error bands around true curve (all i, but only first with legend labels)
        if errs is not None:
            if i == 0:
                ax.plot(
                    u_grid_K,
                    D_true * (1 + err_up / 100),
                    "--",
                    lw=1,
                    color="r",
                    label=f"{err_up}%",
                    zorder=3,
                )
                ax.plot(
                    u_grid_K,
                    D_true * (1 - err_low / 100),
                    "--",
                    lw=1,
                    color="r",
                    #label=f"{err_low}%",
                    zorder=3,
                )
            else:
                ax.plot(
                    u_grid_K,
                    D_true * (1 + err_up / 100),
                    "--",
                    lw=1,
                    color="r",
                    zorder=3,
                )
                ax.plot(
                    u_grid_K,
                    D_true * (1 - err_low / 100),
                    "-.",
                    lw=1,
                    color="r",
                    zorder=3,
                )


        if i == 0:
            # ---  gray shading outside the 5–95% interval ---
            x_min = u_grid_K[0]
            x_max = u_grid_K[-1]
            ax.axvspan(x_min, low_u * K, facecolor="0.9", alpha=0.5, zorder=0)
            ax.axvspan(high_u * K, x_max, facecolor="0.9", alpha=0.5, zorder=0)
            # ---------------------------------------------------

    ax.set_xlabel(r"Cell density [cells mm$^{-2}]$", fontsize=xaxis_fontsize)
    ax.set_ylabel(r"Diffusion [mm$^2$ days$^{-1}$]", fontsize=yaxis_fontsize)
    ax.tick_params(axis="x", labelsize=xtick_fontsize)
    ax.tick_params(axis="y", labelsize=ytick_fontsize)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_facecolor("white")
    ax.legend(
        loc="center",
        bbox_to_anchor=legend_pos,
        ncols=legend_ncols,
        fontsize=legend_fontsize,
        title = legend_title,
        title_fontsize = legend_fontsize,
    )

    plt.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()


def plot_eval_D_multi_gray_gridLayout(
    modelWrapper_dics,
    dataobjs,
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
    legend_ncols=2,
    legend_fontsize=12,
    legend_title=None,
    linestyles=itertools.cycle(["-", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]),
    errs=None,
    figsize=(7, 5),
    axis_fontsizes=None,
    ylim=None,
    y_lim=None,
    show_xlabel=True,
    show_ylabel=True,
    show_xtick_labels=True,
    show_ytick_labels=True,
):
    """
    Grid-layout variant of plot_eval_D_multi_gray.

    The plotting logic is the same, but axis labels and tick labels can be
    selectively suppressed so separately saved panels can later be assembled
    into a shared-axis grid layout. Unlike the standard helper, this version
    preserves a fixed canvas and fixed subplot margins when exporting so all
    panels align in a downstream grid.
    """
    if ylim is None:
        ylim = y_lim

    axis_fontsizes = axis_fontsizes or {}
    xaxis_fontsize = axis_fontsizes.get("xaxis", 11)
    xtick_fontsize = axis_fontsizes.get("xtick_labels", xaxis_fontsize)
    yaxis_fontsize = axis_fontsizes.get("yaxis", 11)
    ytick_fontsize = axis_fontsizes.get("ytick_labels", yaxis_fontsize)

    if errs:
        err_up, err_low = errs
    else:
        err_up = err_low = None

    if not isinstance(dataobjs, (list, tuple)):
        dataobjs = [dataobjs]

    colors = colors[: len(labels)]

    hist_props_list = [hist_properties(d, num_bins) for d in dataobjs]
    low_us = [hp["low_count"] for hp in hist_props_list]
    high_us = [hp["high_count"] for hp in hist_props_list]

    results = []
    D_true_lst = []
    u_grids = []
    u_grids_K = []

    for wrapper_dic, color, label in zip(modelWrapper_dics.values(), colors, labels):
        sample_model = list(wrapper_dic.values())[0].model
        u_vals_np = sample_model.u_vals.flatten()

        u_grids.append(u_vals_np)
        u_grids_K.append(u_vals_np * K)

        diffusion_errors = []
        D_ensemble = []

        for i, wrapper in enumerate(wrapper_dic.values()):
            D_true_check = list(wrapper.model.D_true)
            if i == 0:
                D_true_lst.append(D_true_check)

            model = wrapper.model
            model.eval()

            with torch.no_grad():
                diff_pred = model.D_scale * model.diffusion(model.u_vals_torch).flatten()
                diffusion_error = (model.D_true_torch - diff_pred) ** 2

                diffusion_errors.append(diffusion_error.unsqueeze(0))
                D_ensemble.append(diff_pred.unsqueeze(0))

        D_ensemble = torch.cat(D_ensemble, dim=0)
        D_mean = D_ensemble.mean(0).cpu().numpy()
        D_min = D_ensemble.min(0).values.cpu().numpy()
        D_max = D_ensemble.max(0).values.cpu().numpy()

        diffusion_errors = torch.cat(diffusion_errors, dim=0)
        mse = diffusion_errors.mean().item()

        results.append((mse, label, color, D_mean, D_min, D_max))

    fig, ax = plt.subplots(figsize=figsize)

    for j, (mse, label, color, D_mean, D_min, D_max) in enumerate(results):
        try:
            ls = linestyles[j]
        except TypeError:
            ls = next(linestyles)

        u_grid_K = u_grids_K[j]
        ax.plot(u_grid_K, D_mean, lw=3, color=color, linestyle=ls, label=label)
        if fill:
            ax.fill_between(u_grid_K, D_min, D_max, alpha=0.4, color=color)

    for i, (D_true, D_sym, low_u, high_u, u_grid, u_grid_K) in enumerate(
        zip(D_true_lst, D_sym_true_lst, low_us, high_us, u_grids, u_grids_K)
    ):
        D_true = np.array(D_true)

        if i == 0:
            ax.plot(
                u_grid_K,
                D_true,
                "--",
                lw=1,
                color="k",
                label=f"$D_{Dnum}$",
                zorder=4,
            )
        else:
            ax.plot(
                u_grid_K,
                D_true,
                "--",
                lw=1,
                color="k",
                zorder=4,
            )

        if errs is not None:
            if i == 0:
                ax.plot(
                    u_grid_K,
                    D_true * (1 + err_up / 100),
                    "--",
                    lw=1,
                    color="r",
                    label=f"{err_up}%",
                    zorder=3,
                )
                ax.plot(
                    u_grid_K,
                    D_true * (1 - err_low / 100),
                    "--",
                    lw=1,
                    color="r",
                    zorder=3,
                )
            else:
                ax.plot(
                    u_grid_K,
                    D_true * (1 + err_up / 100),
                    "--",
                    lw=1,
                    color="r",
                    zorder=3,
                )
                ax.plot(
                    u_grid_K,
                    D_true * (1 - err_low / 100),
                    "-.",
                    lw=1,
                    color="r",
                    zorder=3,
                )

        if i == 0:
            x_min = u_grid_K[0]
            x_max = u_grid_K[-1]
            ax.axvspan(x_min, low_u * K, facecolor="0.9", alpha=0.5, zorder=0)
            ax.axvspan(high_u * K, x_max, facecolor="0.9", alpha=0.5, zorder=0)

    if show_xlabel:
        ax.set_xlabel(r"Cell density [cells mm$^{-2}]$", fontsize=xaxis_fontsize)
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(r"Diffusion [mm$^2$ days$^{-1}$]", fontsize=yaxis_fontsize)
    else:
        ax.set_ylabel("")

    ax.tick_params(axis="x", labelsize=xtick_fontsize, labelbottom=show_xtick_labels)
    ax.tick_params(axis="y", labelsize=ytick_fontsize, labelleft=show_ytick_labels)
    if not show_xtick_labels:
        ax.set_xticklabels([])
    if not show_ytick_labels:
        ax.set_yticklabels([])

    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_facecolor("white")
    ax.legend(
        loc="center",
        bbox_to_anchor=legend_pos,
        ncols=legend_ncols,
        fontsize=legend_fontsize,
        title=legend_title,
        title_fontsize=legend_fontsize,
    )

    # Keep fixed subplot geometry across panels so exported PNGs align when
    # assembled into a shared-axis grid, regardless of which labels are shown.
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.98)
    if name:
        plt.savefig(name, dpi=100, facecolor="None")
        print("saved plot:", name)
    plt.show()

def plot_eval_G_multi_gray(
    modelWrapper_dics,
    dataobjs,
    G_sym_true_lst,
    colors,
    labels,
    Gnum=1,
    device="cpu",
    num_bins=50,
    K=1700,
    name=None,
    fill=True,
    legend_pos=(0.5, 0.5),
    legend_ncols=2,
    legend_fontsize=12,
    linestyles=itertools.cycle(["-", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]),
    errs=None,
    xlim = None,
    legend_title = "Noise (%)",
    figsize = (7, 5),
    axis_fontsizes=None,
):
    """
    Plot growth G(u) ensembles + true curves for multiple configs / datasets.

    modelWrapper_dics : dict
        {config_key: {seed: wrapper, ...}, ...}
    dataobjs : list or single
        One data object per configuration (used for 5–95% u-percentiles).
    G_sym_true_lst : list
        Symbolic / reference true G(u) for each config (same ordering as dataobjs).
    errs : (err_up, err_low) or None
        Percentage error bands around the true G(u).
    """
    if errs:
        err_up, err_low = errs
    else:
        err_up = err_low = None

    if axis_fontsizes is None:
        axis_fontsizes = {
            "xaxis": 11,
            "xtick_labels": 11,
            "yaxis": 11,
            "ytick_labels": 11,
        }

    if not isinstance(dataobjs, (list, tuple)):
        dataobjs = [dataobjs]

    colors = colors[: len(labels)]

    # Percentiles for each dataset
    hist_props_list = [hist_properties(d, num_bins) for d in dataobjs]
    low_us = [hp["low_count"] for hp in hist_props_list]
    high_us = [hp["high_count"] for hp in hist_props_list]

    results = []
    G_true_lst = []
    u_grids = []
    u_grids_K = []

    # ------------------------------------------------
    # Loop over configurations
    # ------------------------------------------------
    for wrapper_dic, color, label in zip(modelWrapper_dics.values(), colors, labels):
        # Sample model for THIS configuration
        sample_model = list(wrapper_dic.values())[0].model
        u_vals_torch = sample_model.u_vals_torch
        u_vals_np = sample_model.u_vals.flatten()

        u_grids.append(u_vals_np)
        u_grids_K.append(u_vals_np * K)

        G_ensemble = []
        growth_errors = []

        for i, wrapper in enumerate(wrapper_dic.values()):
            G_true_check = list(wrapper.model.G_true)
            if i == 0:  # and G_true_check not in G_true_lst:
                G_true_lst.append(G_true_check)

            model = wrapper.model
            model.eval()

            with torch.no_grad():
                grow_pred = model.G_scale * model.growth(model.u_vals_torch).flatten()
                growth_error = (model.G_true_torch - grow_pred) ** 2

                growth_errors.append(growth_error.unsqueeze(0))  # [1, N]
                G_ensemble.append(grow_pred.unsqueeze(0))        # [1, N]

        G_ensemble = torch.cat(G_ensemble, dim=0)
        G_mean = G_ensemble.mean(0).cpu().numpy()
        G_min = G_ensemble.min(0).values.cpu().numpy()
        G_max = G_ensemble.max(0).values.cpu().numpy()

        growth_errors = torch.cat(growth_errors, dim=0)
        mse = growth_errors.mean().item()

        results.append((mse, label, color, G_mean, G_min, G_max))

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Ensemble curves
    for j, (mse, label, color, G_mean, G_min, G_max) in enumerate(results):
        try:
            ls = linestyles[j]
        except TypeError:
            ls = next(linestyles)

        u_grid_K = u_grids_K[j]
        ax.plot(u_grid_K, G_mean, lw=3, color=color, linestyle=ls, label=label)
        if fill:
            ax.fill_between(u_grid_K, G_min, G_max, alpha=0.4, color=color)

    # ------------------------------------------------
    # True curves + error bands + color-coded percentile lines
    # ------------------------------------------------
    for i, (G_true, G_sym, low_u, high_u, u_grid, u_grid_K) in enumerate(
        zip(G_true_lst, G_sym_true_lst, low_us, high_us, u_grids, u_grids_K)
    ):
        G_true = np.array(G_true)
        col = colors[i % len(colors)]

        # True curve
        if i == 0:
            ax.plot(
                u_grid_K,
                G_true,
                "--",
                lw=1,
                color="k",
                label=f"$G_{Gnum}$",
                zorder=4,
            )
        else:
            ax.plot(
                u_grid_K,
                G_true,
                "--",
                lw=1,
                color="k",
                zorder=4,
            )

        # Error bands, for all i but only first with labels
        if errs is not None:
            if i == 0:
                ax.plot(
                    u_grid_K,
                    G_true * (1 + err_up / 100),
                    "--",
                    lw=1,
                    color="r",
                    label=f"{err_up}%",
                    zorder=3,
                )
                ax.plot(
                    u_grid_K,
                    G_true * (1 - err_low / 100),
                    "--",
                    lw=1,
                    color="r",
                    #label=f"{err_low}%",
                    zorder=3,
                )
            else:
                ax.plot(
                    u_grid_K,
                    G_true * (1 + err_up / 100),
                    "--",
                    lw=1,
                    color="r",
                    zorder=3,
                )
                ax.plot(
                    u_grid_K,
                    G_true * (1 - err_low / 100),
                    "-.",
                    lw=1,
                    color="r",
                    zorder=3,
                )

            # --- gray shading outside the 5–95% interval ---
            x_min = u_grid_K[0]
            x_max = u_grid_K[-1]
            ax.axvspan(x_min, low_u * K, facecolor="0.9", alpha=0.5, zorder=0)
            ax.axvspan(high_u * K, x_max, facecolor="0.9", alpha=0.5, zorder=0)
            # ---------------------------------------------------


    ax.set_xlabel(
        r"Cell density [cells mm$^{-2}]$",
        fontsize=axis_fontsizes["xaxis"],
    )
    ax.set_ylabel(
        r"Growth [days$^{-1}$]",
        fontsize=axis_fontsizes["yaxis"],
    )
    ax.tick_params(axis="x", labelsize=axis_fontsizes["xtick_labels"])
    ax.tick_params(axis="y", labelsize=axis_fontsizes["ytick_labels"])
    ax.set_facecolor("white")
    ax.legend(
        loc="center",
        bbox_to_anchor=legend_pos,
        ncols=legend_ncols,
        fontsize=legend_fontsize,
        title= legend_title,
        title_fontsize = legend_fontsize,
    )

    plt.tight_layout()

    if xlim:
        ax.set_xlim(xlim)
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()


def plot_eval_D_multi(
    modelWrapper_dics,
    dataobjs,
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
    legend_ncols=2,
    legend_fontsize=12,
    legend_title=None,
    linestyles=itertools.cycle(["-", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]),
    errs=None,
    figsize=(7, 5)
):
    """
    Plot diffusion D(u) ensembles + true curves for multiple configs / datasets,
    but only over the central 90% u-band (5–95%), with no gray shading.
    """
    if errs:
        err_up, err_low = errs
    else:
        err_up = err_low = None

    # Allow passing a single dataobj
    if not isinstance(dataobjs, (list, tuple)):
        dataobjs = [dataobjs]

    n_confs = len(modelWrapper_dics)
    colors = colors[: len(labels)]

    # If only one dataobj is given but multiple configs, broadcast it
    if len(dataobjs) == 1 and n_confs > 1:
        dataobjs = dataobjs * n_confs

    # Safety check: one dataobj per configuration
    if len(dataobjs) != n_confs:
        raise ValueError(
            f"Number of dataobjs ({len(dataobjs)}) must match number of configs ({n_confs})."
        )

    # Compute 5–95% u-percentiles for each data object
    hist_props_list = [hist_properties(d, num_bins) for d in dataobjs]
    low_us = [hp["low_count"] for hp in hist_props_list]
    high_us = [hp["high_count"] for hp in hist_props_list]
    low_us_K = [u * K for u in low_us]
    high_us_K = [u * K for u in high_us]

    results = []
    D_true_lst = []
    u_grids = []
    u_grids_K = []

    # ------------------------------------------------
    # Loop over configurations
    # ------------------------------------------------
    for wrapper_dic, color, label in zip(modelWrapper_dics.values(), colors, labels):
        # Sample model for THIS configuration
        sample_model = list(wrapper_dic.values())[0].model
        u_vals_np = sample_model.u_vals.flatten()

        u_grids.append(u_vals_np)
        u_grids_K.append(u_vals_np * K)

        diffusion_errors = []
        D_ensemble = []

        for i, wrapper in enumerate(wrapper_dic.values()):
            D_true_check = list(wrapper.model.D_true)
            if i == 0:
                D_true_lst.append(D_true_check)

            model = wrapper.model
            model.eval()

            with torch.no_grad():
                diff_pred = model.D_scale * model.diffusion(model.u_vals_torch).flatten()
                diffusion_error = (model.D_true_torch - diff_pred) ** 2

                diffusion_errors.append(diffusion_error.unsqueeze(0))  # [1, N]
                D_ensemble.append(diff_pred.unsqueeze(0))              # [1, N]

        D_ensemble = torch.cat(D_ensemble, dim=0)  # [runs, N]
        D_mean = D_ensemble.mean(0).cpu().numpy()
        D_min = D_ensemble.min(0).values.cpu().numpy()
        D_max = D_ensemble.max(0).values.cpu().numpy()

        diffusion_errors = torch.cat(diffusion_errors, dim=0)
        mse = diffusion_errors.mean().item()

        results.append((mse, label, color, D_mean, D_min, D_max))

    # Sanity check: all arrays same length as number of configs
    assert len(results) == len(u_grids_K) == len(low_us_K) == len(high_us_K)

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Ensemble predictions for each configuration, cropped to [low_us_K, high_us_K]
    for j, (mse, label, color, D_mean, D_min, D_max) in enumerate(results):
        try:
            ls = linestyles[j]
        except TypeError:
            ls = next(linestyles)

        u_grid_K = u_grids_K[j]
        x_low = low_us_K[j]
        x_high = high_us_K[j]
        mask = (u_grid_K >= x_low) & (u_grid_K <= x_high)

        u_plot = u_grid_K[mask]
        D_mean_plot = D_mean[mask]
        D_min_plot = D_min[mask]
        D_max_plot = D_max[mask]

        ax.plot(u_plot, D_mean_plot, lw=3, color=color, linestyle=ls, label=label)
        if fill:
            ax.fill_between(u_plot, D_min_plot, D_max_plot, alpha=0.4, color=color)

    # ------------------------------------------------
    # True curves + error bands, also cropped
    # ------------------------------------------------
    for i, (D_true, D_sym, low_u, high_u, u_grid, u_grid_K) in enumerate(
        zip(D_true_lst, D_sym_true_lst, low_us, high_us, u_grids, u_grids_K)
    ):
        D_true = np.array(D_true)

        x_low = low_us_K[i]
        x_high = high_us_K[i]
        mask = (u_grid_K >= x_low) & (u_grid_K <= x_high)

        u_plot = u_grid_K[mask]
        D_true_plot = D_true[mask]

        # True curve
        if i == 0:
            ax.plot(
                u_plot,
                D_true_plot,
                "--",
                lw=1,
                color="k",
                label=f"$D_{Dnum}$",
                zorder=4,
            )
        else:
            ax.plot(
                u_plot,
                D_true_plot,
                "--",
                lw=1,
                color="k",
                zorder=4,
            )

        # Error bands around true curve, cropped
        if errs is not None:
            D_up = (D_true * (1 + err_up / 100))[mask]
            D_low = (D_true * (1 - err_low / 100))[mask]

            if i == 0:
                ax.plot(
                    u_plot,
                    D_up,
                    "--",
                    lw=1,
                    color="r",
                    label=f"{err_up}%",
                    zorder=3,
                )
                ax.plot(
                    u_plot,
                    D_low,
                    "--",
                    lw=1,
                    color="r",
                    zorder=3,
                )
            else:
                ax.plot(
                    u_plot,
                    D_up,
                    "--",
                    lw=1,
                    color="r",
                    zorder=3,
                )
                ax.plot(
                    u_plot,
                    D_low,
                    "-.",
                    lw=1,
                    color="r",
                    zorder=3,
                )

    ax.set_xlabel(r"Cell density [cells mm$^{-2}]$", fontsize=11)
    ax.set_ylabel(r"Diffusion [mm$^2$ days$^{-1}$]", fontsize=11)
    ax.set_facecolor("white")
    ax.legend(
        loc="center",
        bbox_to_anchor=legend_pos,
        ncols=legend_ncols,
        fontsize=legend_fontsize,
        title=legend_title,
        title_fontsize=legend_fontsize,
    )

    plt.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()





def plot_eval_D_multi_fixedD(
    modelWrapper_dics,
    dataobjs,
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
    legend_ncols=2,
    legend_fontsize=12,
    legend_title=None,
    linestyles=itertools.cycle(["-", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]),
    errs=None,
):
    """
    Plot diffusion D(u) ensembles + true curves for multiple configs / datasets,
    but only over the central 90% u-band (5–95%), with no gray shading.
    """
    if errs:
        err_up, err_low = errs
    else:
        err_up = err_low = None

    # Allow passing a single dataobj
    if not isinstance(dataobjs, (list, tuple)):
        dataobjs = [dataobjs]

    n_confs = len(modelWrapper_dics)
    colors = colors[: len(labels)]

    # If only one dataobj is given but multiple configs, broadcast it
    if len(dataobjs) == 1 and n_confs > 1:
        dataobjs = dataobjs * n_confs

    # Safety check: one dataobj per configuration
    if len(dataobjs) != n_confs:
        raise ValueError(
            f"Number of dataobjs ({len(dataobjs)}) must match number of configs ({n_confs})."
        )

    # Compute 5–95% u-percentiles for each data object
    hist_props_list = [hist_properties(d, num_bins) for d in dataobjs]
    low_us = [hp["low_count"] for hp in hist_props_list]
    high_us = [hp["high_count"] for hp in hist_props_list]
    low_us_K = [u * K for u in low_us]
    high_us_K = [u * K for u in high_us]

    results = []
    D_true_lst = []
    u_grids = []
    u_grids_K = []

    # ------------------------------------------------
    # Loop over configurations
    # ------------------------------------------------
    for wrapper_dic, color, label in zip(modelWrapper_dics.values(), colors, labels):
        # Sample model for THIS configuration
        sample_model = list(wrapper_dic.values())[0].model
        u_vals_np = sample_model.u_vals.flatten()

        u_grids.append(u_vals_np)
        u_grids_K.append(u_vals_np * K)

        diffusion_errors = []
        D_ensemble = []

        for i, wrapper in enumerate(wrapper_dic.values()):
            D_true_check = list(wrapper.model.D_true)
            if i == 0:
                D_true_lst.append(D_true_check)

            model = wrapper.model
            model.eval()

            with torch.no_grad():
                diff_pred = model.D_scale * model.diffusion(model.u_vals_torch).flatten()
                diffusion_error = np.abs((model.D_true_torch - diff_pred)) ** 2
                diffusion_errors.append(diffusion_error.unsqueeze(0))  # [1, N]
                D_ensemble.append(diff_pred.unsqueeze(0))              # [1, N]

        D_ensemble = torch.cat(D_ensemble, dim=0)  # [runs, N]
        D_mean = D_ensemble.mean(0).cpu().numpy()
        D_min = D_ensemble.min(0).values.cpu().numpy()
        D_max = D_ensemble.max(0).values.cpu().numpy()

        diffusion_errors = torch.cat(diffusion_errors, dim=0)
        mse = diffusion_errors.mean().item()

        results.append((mse, label, color, D_mean, D_min, D_max))

    # Sanity check: all arrays same length as number of configs
    assert len(results) == len(u_grids_K) == len(low_us_K) == len(high_us_K)

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    # Ensemble predictions for each configuration, cropped to [low_us_K, high_us_K]
    for j, (mse, label, color, D_mean, D_min, D_max) in enumerate(results):
        try:
            ls = linestyles[j]
        except TypeError:
            ls = next(linestyles)

        u_grid_K = u_grids_K[j]
        x_low = low_us_K[j]
        x_high = high_us_K[j]
        mask = (u_grid_K >= x_low) & (u_grid_K <= x_high)

        u_plot = u_grid_K[mask]
        D_mean_plot = D_mean[mask]
        D_min_plot = D_min[mask]
        D_max_plot = D_max[mask]

        ax.plot(u_plot, D_mean_plot, lw=3, color=color, linestyle=ls, label=label)
        if fill:
            ax.fill_between(u_plot, D_min_plot, D_max_plot, alpha=0.4, color=color)

    # ------------------------------------------------
    # True curves + error bands, also cropped
    # ------------------------------------------------
    for i, (D_true, D_sym, low_u, high_u, u_grid, u_grid_K) in enumerate(
        zip(D_true_lst, D_sym_true_lst, low_us, high_us, u_grids, u_grids_K)
    ):
        D_true = np.array(D_true)

        x_low = low_us_K[i]
        x_high = high_us_K[i]
        mask = (u_grid_K >= x_low) & (u_grid_K <= x_high)

        u_plot = u_grid_K[mask]
        D_true_plot = D_true[mask]

        # True curve
        if i == 0:
            ax.plot(
                u_plot,
                D_true_plot,
                "--",
                lw=1,
                color="k",
                label=f"$D_{Dnum}$",
                zorder=4,
            )
        # else:
        #     ax.plot(
        #         u_plot,
        #         D_true_plot,
        #         "--",
        #         lw=1,
        #         color="k",
        #         zorder=4,
        #     )

        # Error bands around true curve, cropped
        if errs is not None:
            D_up = (D_true * (1 + err_up / 100))[mask]
            D_low = (D_true * (1 - err_low / 100))[mask]

            if i == 0:
                ax.plot(
                    u_plot,
                    D_up,
                    "--",
                    lw=1,
                    color="r",
                    label=f"{err_up}%",
                    zorder=3,
                )
                ax.plot(
                    u_plot,
                    D_low,
                    "--",
                    lw=1,
                    color="r",
                    zorder=3,
                )
            # else:
            #     ax.plot(
            #         u_plot,
            #         D_up,
            #         "--",
            #         lw=1,
            #         color="r",
            #         zorder=3,
            #     )
            #     ax.plot(
            #         u_plot,
            #         D_low,
            #         "-.",
            #         lw=1,
            #         color="r",
            #         zorder=3,
            #     )

    ax.set_xlabel(r"Cell density [cells mm$^{-2}]$", fontsize=11)
    ax.set_ylabel(r"Diffusion [mm$^2$ days$^{-1}$]", fontsize=11)
    ax.set_facecolor("white")
    ax.legend(
        loc="center",
        bbox_to_anchor=legend_pos,
        ncols=legend_ncols,
        fontsize=legend_fontsize,
        title=legend_title,
        title_fontsize=legend_fontsize,
    )

    plt.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()

    return results



def plot_eval_G_multi(
    modelWrapper_dics,
    dataobjs,
    G_sym_true_lst,
    colors,
    labels,
    Gnum=1,
    device="cpu",
    num_bins=50,
    K=1700,
    name=None,
    fill=True,
    legend_pos=(0.5, 0.5),
    legend_ncols=2,
    legend_fontsize=12,
    legend_title=None,
    linestyles=itertools.cycle(["-", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]),
    errs=None,
    xlim=None,
    y_label = r"cell growth [mm$^2$ days$^{-1}$]",
):
    """
    Plot growth G(u) ensembles + true curves for multiple configs / datasets,
    but only over the central 90% u-band (5–95%).
    """
    if errs:
        err_up, err_low = errs
    else:
        err_up = err_low = None

    if not isinstance(dataobjs, (list, tuple)):
        dataobjs = [dataobjs]

    colors = colors[: len(labels)]

    # Percentiles for each dataset
    hist_props_list = [hist_properties(d, num_bins) for d in dataobjs]
    low_us = [hp["low_count"] for hp in hist_props_list]
    high_us = [hp["high_count"] for hp in hist_props_list]
    low_us_K = [u * K for u in low_us]
    high_us_K = [u * K for u in high_us]

    results = []
    G_true_lst = []
    u_grids = []
    u_grids_K = []

    # ------------------------------------------------
    # Loop over configurations
    # ------------------------------------------------
    for wrapper_dic, color, label in zip(modelWrapper_dics.values(), colors, labels):
        sample_model = list(wrapper_dic.values())[0].model
        u_vals_np = sample_model.u_vals.flatten()

        u_grids.append(u_vals_np)
        u_grids_K.append(u_vals_np * K)

        G_ensemble = []
        growth_errors = []

        for i, wrapper in enumerate(wrapper_dic.values()):
            G_true_check = list(wrapper.model.G_true)
            if i == 0:
                G_true_lst.append(G_true_check)

            model = wrapper.model
            model.eval()

            with torch.no_grad():
                grow_pred = model.G_scale * model.growth(model.u_vals_torch).flatten()
                growth_error = (model.G_true_torch - grow_pred) ** 2

                growth_errors.append(growth_error.unsqueeze(0))  # [1, N]
                G_ensemble.append(grow_pred.unsqueeze(0))        # [1, N]

        G_ensemble = torch.cat(G_ensemble, dim=0)
        G_mean = G_ensemble.mean(0).cpu().numpy()
        G_min = G_ensemble.min(0).values.cpu().numpy()
        G_max = G_ensemble.max(0).values.cpu().numpy()

        growth_errors = torch.cat(growth_errors, dim=0)
        mse = growth_errors.mean().item()

        results.append((mse, label, color, G_mean, G_min, G_max))

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    # Ensemble curves, cropped to [low_us_K, high_us_K]
    for j, (mse, label, color, G_mean, G_min, G_max) in enumerate(results):
        try:
            ls = linestyles[j]
        except TypeError:
            ls = next(linestyles)

        u_grid_K = u_grids_K[j]
        x_low = low_us_K[j]
        x_high = high_us_K[j]
        mask = (u_grid_K >= x_low) & (u_grid_K <= x_high)

        u_plot = u_grid_K[mask]
        G_mean_plot = G_mean[mask]
        G_min_plot = G_min[mask]
        G_max_plot = G_max[mask]

        ax.plot(u_plot, G_mean_plot, lw=3, color=color, linestyle=ls, label=label)
        if fill:
            ax.fill_between(u_plot, G_min_plot, G_max_plot, alpha=0.4, color=color)

    # ------------------------------------------------
    # True curves + error bands, cropped
    # ------------------------------------------------
    for i, (G_true, G_sym, low_u, high_u, u_grid, u_grid_K) in enumerate(
        zip(G_true_lst, G_sym_true_lst, low_us, high_us, u_grids, u_grids_K)
    ):
        G_true = np.array(G_true)

        x_low = low_us_K[i]
        x_high = high_us_K[i]
        mask = (u_grid_K >= x_low) & (u_grid_K <= x_high)

        u_plot = u_grid_K[mask]
        G_true_plot = G_true[mask]

        # True curve
        if i == 0:
            ax.plot(
                u_plot,
                G_true_plot,
                "--",
                lw=1,
                color="k",
                label=f"$G_{Gnum}$",
                zorder=4,
            )
        else:
            ax.plot(
                u_plot,
                G_true_plot,
                "--",
                lw=1,
                color="k",
                zorder=4,
            )

        # Error bands if requested
        if errs is not None:
            G_up = (G_true * (1 + err_up / 100))[mask]
            G_low = (G_true * (1 - err_low / 100))[mask]

            if i == 0:
                ax.plot(
                    u_plot,
                    G_up,
                    "--",
                    lw=1,
                    color="r",
                    label=f"{err_up}%",
                    zorder=3,
                )
                ax.plot(
                    u_plot,
                    G_low,
                    "--",
                    lw=1,
                    color="r",
                    zorder=3,
                )
            else:
                ax.plot(
                    u_plot,
                    G_up,
                    "--",
                    lw=1,
                    color="r",
                    zorder=3,
                )
                ax.plot(
                    u_plot,
                    G_low,
                    "-.",
                    lw=1,
                    color="r",
                    zorder=3,
                )

    ax.set_xlabel(r"Cell density [cells mm$^{-2}]$", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_facecolor("white")
    ax.legend(
        loc="center",
        bbox_to_anchor=legend_pos,
        ncols=legend_ncols,
        fontsize=legend_fontsize,
        title=legend_title,
        title_fontsize=legend_fontsize,
    )
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()



def plot_eval_G_multi_fixedG(
    modelWrapper_dics,
    dataobjs,
    G_sym_true_lst,
    colors,
    labels,
    Gnum=1,
    device="cpu",
    num_bins=50,
    K=1700,
    name=None,
    fill=True,
    legend_pos=(0.5, 0.5),
    legend_ncols=2,
    legend_fontsize=12,
    legend_title=None,
    linestyles=itertools.cycle(["-", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]),
    errs=None,
):
    """
    Plot growth G(u) ensembles + true curves for multiple configs / datasets,
    but only over the central 90% u-band (5–95%).
    """
    if errs:
        err_up, err_low = errs
    else:
        err_up = err_low = None

    if not isinstance(dataobjs, (list, tuple)):
        dataobjs = [dataobjs]

    colors = colors[: len(labels)]

    # Percentiles for each dataset
    hist_props_list = [hist_properties(d, num_bins) for d in dataobjs]
    low_us = [hp["low_count"] for hp in hist_props_list]
    high_us = [hp["high_count"] for hp in hist_props_list]
    low_us_K = [u * K for u in low_us]
    high_us_K = [u * K for u in high_us]

    results = []
    G_true_lst = []
    u_grids = []
    u_grids_K = []

    # ------------------------------------------------
    # Loop over configurations
    # ------------------------------------------------
    for wrapper_dic, color, label in zip(modelWrapper_dics.values(), colors, labels):
        sample_model = list(wrapper_dic.values())[0].model
        u_vals_np = sample_model.u_vals.flatten()

        u_grids.append(u_vals_np)
        u_grids_K.append(u_vals_np * K)

        G_ensemble = []
        growth_errors = []

        for i, wrapper in enumerate(wrapper_dic.values()):
            G_true_check = list(wrapper.model.G_true)
            if i == 0:
                G_true_lst.append(G_true_check)

            model = wrapper.model
            model.eval()

            with torch.no_grad():
                grow_pred = model.G_scale * model.growth(model.u_vals_torch).flatten()
                growth_error = (model.G_true_torch - grow_pred) ** 2

                growth_errors.append(growth_error.unsqueeze(0))  # [1, N]
                G_ensemble.append(grow_pred.unsqueeze(0))        # [1, N]

        G_ensemble = torch.cat(G_ensemble, dim=0)
        G_mean = G_ensemble.mean(0).cpu().numpy()
        G_min = G_ensemble.min(0).values.cpu().numpy()
        G_max = G_ensemble.max(0).values.cpu().numpy()

        growth_errors = torch.cat(growth_errors, dim=0)
        mse = growth_errors.mean().item()

        results.append((mse, label, color, G_mean, G_min, G_max))

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    # Ensemble curves, cropped to [low_us_K, high_us_K]
    for j, (mse, label, color, G_mean, G_min, G_max) in enumerate(results):
        try:
            ls = linestyles[j]
        except TypeError:
            ls = next(linestyles)

        u_grid_K = u_grids_K[j]
        x_low = low_us_K[j]
        x_high = high_us_K[j]
        mask = (u_grid_K >= x_low) & (u_grid_K <= x_high)

        u_plot = u_grid_K[mask]
        G_mean_plot = G_mean[mask]
        G_min_plot = G_min[mask]
        G_max_plot = G_max[mask]

        ax.plot(u_plot, G_mean_plot, lw=3, color=color, linestyle=ls, label=label)
        if fill:
            ax.fill_between(u_plot, G_min_plot, G_max_plot, alpha=0.4, color=color)

    # ------------------------------------------------
    # True curves + error bands, cropped
    # ------------------------------------------------
    for i, (G_true, G_sym, low_u, high_u, u_grid, u_grid_K) in enumerate(
        zip(G_true_lst, G_sym_true_lst, low_us, high_us, u_grids, u_grids_K)
    ):
        G_true = np.array(G_true)

        x_low = low_us_K[i]
        x_high = high_us_K[i]
        mask = (u_grid_K >= x_low) & (u_grid_K <= x_high)

        u_plot = u_grid_K[mask]
        G_true_plot = G_true[mask]

        # True curve
        if i == 0:
            ax.plot(
                u_plot,
                G_true_plot,
                "--",
                lw=1,
                color="k",
                label=f"$G_{Gnum}$",
                zorder=4,
            )
        # else:
        #     ax.plot(
        #         u_plot,
        #         G_true_plot,
        #         "--",
        #         lw=1,
        #         color="k",
        #         zorder=4,
        #     )

        # Error bands if requested
        if errs is not None:
            G_up = (G_true * (1 + err_up / 100))[mask]
            G_low = (G_true * (1 - err_low / 100))[mask]

            if i == 0:
                ax.plot(
                    u_plot,
                    G_up,
                    "--",
                    lw=1,
                    color="r",
                    label=f"{err_up}%",
                    zorder=3,
                )
                ax.plot(
                    u_plot,
                    G_low,
                    "--",
                    lw=1,
                    color="r",
                    zorder=3,
                )
            else:
                ax.plot(
                    u_plot,
                    G_up,
                    "--",
                    lw=1,
                    color="r",
                    zorder=3,
                )
                ax.plot(
                    u_plot,
                    G_low,
                    "-.",
                    lw=1,
                    color="r",
                    zorder=3,
                )

    ax.set_xlabel(r"Cell density [cells mm$^{-2}]$", fontsize=11)
    ax.set_ylabel(r"Growth [mm$^2$ days$^{-1}$]", fontsize=11)
    ax.set_facecolor("white")
    ax.legend(
        loc="center",
        bbox_to_anchor=legend_pos,
        ncols=legend_ncols,
        fontsize=legend_fontsize,
        title=legend_title,
        title_fontsize=legend_fontsize,
    )

    plt.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()
