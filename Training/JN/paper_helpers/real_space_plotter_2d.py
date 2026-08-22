"""Real-space plotting helpers for two-dimensional learned functions.

Contents
--------
- symbolic_from_function
- true_diffusion_expression
- true_growth_expression
- plot_eval_D_multi_gray
- plot_eval_G_multi_gray"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import sympy as sp
import ast
import inspect
import itertools
from .utils_2d import hist_properties
from .prepare_model_loss_2d import _extract_model_wrappers


def symbolic_from_function(func, var_name="u"):
    source = inspect.getsource(func).strip()
    tree = ast.parse(source)
    return_node = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.Return))
    )
    allowed_names = func.__globals__.copy()
    u = sp.Symbol(var_name)
    expr = eval(
        compile(ast.Expression(return_node.value), "<ast>", "eval"),
        {**allowed_names, var_name: u},
    )
    return expr


def true_diffusion_expression(label, var_name="u"):
    """Return the symbolic diffusion law used to generate the data."""
    u = sp.Symbol(var_name)
    expressions = {
        "const": sp.Float("0.02472"),
        "linear": sp.Float("0.015") + sp.Float("0.06") * u,
        "quadratic": sp.Float("0.01") + sp.Float("0.044") * u**2,
        "exp": sp.Float("0.003")
        + sp.Float("0.095") * (1 - sp.exp(-sp.Float("2.5") * u)),
    }
    return expressions[label]


def true_growth_expression(label, var_name="u"):
    """Return the symbolic growth law used to generate the data."""
    u = sp.Symbol(var_name)
    expressions = {
        "const": sp.Float("1.3") / 2,
        "linear": (sp.Float("2.4") - sp.Float("3.0") * u) / 2,
        "quadratic": (sp.Float("2.1") - sp.Float("0.29") * u**2) / 2,
        "exp": (sp.Float("0.7") + sp.Float("1.3") * (1 - sp.exp(sp.Float("4.0") * u)))
        / 2,
        "zero": sp.Integer(0),
    }
    return expressions[label]


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
    legend_title=None,
    linestyles=itertools.cycle(["-", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]),
    errs=None,
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
    ylim, y_lim : tuple or None
        Optional y-axis limits. `y_lim` is accepted as a backward-compatible alias.
    """
    if ylim is None:
        ylim = y_lim

    if errs:
        (err_up, err_low) = errs
    else:
        err_up = err_low = None
    axis_fontsizes = {
        "xaxis": 15,
        "xtick_labels": 15,
        "yaxis": 15,
        "ytick_labels": 15,
        **(axis_fontsizes or {}),
    }
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
        model_wrappers = _extract_model_wrappers(wrapper_dic)
        sample_model = model_wrappers[0].model
        u_vals_np = sample_model.u_vals.flatten()
        u_grids.append(u_vals_np)
        u_grids_K.append(u_vals_np * K)
        diffusion_errors = []
        D_ensemble = []
        for i, wrapper in enumerate(model_wrappers):
            D_true_check = list(wrapper.model.D_true)
            if i == 0:
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
        D_mean = D_ensemble.mean(0).cpu().numpy()
        D_min = D_ensemble.min(0).values.cpu().numpy()
        D_max = D_ensemble.max(0).values.cpu().numpy()
        diffusion_errors = torch.cat(diffusion_errors, dim=0)
        mse = diffusion_errors.mean().item()
        results.append((mse, label, color, D_mean, D_min, D_max))
    (fig, ax) = plt.subplots(figsize=(7, 5))
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
                u_grid_K, D_true, "--", lw=1, color="k", label=f"$D_{Dnum}$", zorder=4
            )
        else:
            ax.plot(u_grid_K, D_true, "--", lw=1, color="k", zorder=4)
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
    ax.set_xlabel(
        "Cell density [cells mm$^{-2}]$",
        fontsize=axis_fontsizes["xaxis"],
    )
    ax.set_ylabel(
        "Diffusion [mm$^2$ days$^{-1}$]",
        fontsize=axis_fontsizes["yaxis"],
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis="x", labelsize=axis_fontsizes["xtick_labels"])
    ax.tick_params(axis="y", labelsize=axis_fontsizes["ytick_labels"])
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
    legend_title=None,
    linestyles=itertools.cycle(["-", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]),
    errs=None,
    axis_fontsizes=None,
    ylim=None,
    y_lim=None,
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
    ylim, y_lim : tuple or None
        Optional y-axis limits. `y_lim` is accepted as a backward-compatible alias.
    """
    if ylim is None:
        ylim = y_lim

    if errs:
        (err_up, err_low) = errs
    else:
        err_up = err_low = None
    axis_fontsizes = {
        "xaxis": 15,
        "xtick_labels": 15,
        "yaxis": 15,
        "ytick_labels": 15,
        **(axis_fontsizes or {}),
    }
    if not isinstance(dataobjs, (list, tuple)):
        dataobjs = [dataobjs]
    colors = colors[: len(labels)]
    hist_props_list = [hist_properties(d, num_bins) for d in dataobjs]
    low_us = [hp["low_count"] for hp in hist_props_list]
    high_us = [hp["high_count"] for hp in hist_props_list]
    results = []
    G_true_lst = []
    u_grids = []
    u_grids_K = []
    for wrapper_dic, color, label in zip(modelWrapper_dics.values(), colors, labels):
        model_wrappers = _extract_model_wrappers(wrapper_dic)
        sample_model = model_wrappers[0].model
        u_vals_np = sample_model.u_vals.flatten()
        u_grids.append(u_vals_np)
        u_grids_K.append(u_vals_np * K)
        G_ensemble = []
        growth_errors = []
        for i, wrapper in enumerate(model_wrappers):
            G_true_check = list(wrapper.model.G_true)
            if i == 0:
                G_true_lst.append(G_true_check)
            model = wrapper.model
            model.eval()
            with torch.no_grad():
                grow_pred = model.G_scale * model.growth(model.u_vals_torch).flatten()
                growth_error = (model.G_true_torch - grow_pred) ** 2
                growth_errors.append(growth_error.unsqueeze(0))
                G_ensemble.append(grow_pred.unsqueeze(0))
        G_ensemble = torch.cat(G_ensemble, dim=0)
        G_mean = G_ensemble.mean(0).cpu().numpy()
        G_min = G_ensemble.min(0).values.cpu().numpy()
        G_max = G_ensemble.max(0).values.cpu().numpy()
        growth_errors = torch.cat(growth_errors, dim=0)
        mse = growth_errors.mean().item()
        results.append((mse, label, color, G_mean, G_min, G_max))
    (fig, ax) = plt.subplots(figsize=(7, 5))
    for j, (mse, label, color, G_mean, G_min, G_max) in enumerate(results):
        try:
            ls = linestyles[j]
        except TypeError:
            ls = next(linestyles)
        u_grid_K = u_grids_K[j]
        ax.plot(u_grid_K, G_mean, lw=3, color=color, linestyle=ls, label=label)
        if fill:
            ax.fill_between(u_grid_K, G_min, G_max, alpha=0.4, color=color)
    for i, (G_true, G_sym, low_u, high_u, u_grid, u_grid_K) in enumerate(
        zip(G_true_lst, G_sym_true_lst, low_us, high_us, u_grids, u_grids_K)
    ):
        G_true = np.array(G_true)
        if i == 0:
            ax.plot(
                u_grid_K, G_true, "--", lw=1, color="k", label=f"$G_{Gnum}$", zorder=4
            )
        else:
            ax.plot(u_grid_K, G_true, "--", lw=1, color="k", zorder=4)
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
            x_min = u_grid_K[0]
            x_max = u_grid_K[-1]
            ax.axvspan(x_min, low_u * K, facecolor="0.9", alpha=0.5, zorder=0)
            ax.axvspan(high_u * K, x_max, facecolor="0.9", alpha=0.5, zorder=0)
    ax.set_xlabel(
        "Cell density [cells mm$^{-2}]$",
        fontsize=axis_fontsizes["xaxis"],
    )
    ax.set_ylabel(
        "Growth [days$^{-1}$]",
        fontsize=axis_fontsizes["yaxis"],
    )
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis="x", labelsize=axis_fontsizes["xtick_labels"])
    ax.tick_params(axis="y", labelsize=axis_fontsizes["ytick_labels"])
    ax.set_facecolor("white")
    ax.legend(
        loc="center",
        bbox_to_anchor=legend_pos,
        ncols=legend_ncols,
        frameon=True,
        fontsize=legend_fontsize,
        title=legend_title,
        title_fontsize=legend_fontsize,
    )
    plt.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()
