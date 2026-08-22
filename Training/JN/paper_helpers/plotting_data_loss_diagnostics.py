"""Diagnostics for data-loss component behaviour across training.

Contents
--------
- _is_model_wrapper
- _run_groups_from_models_dics
- _normalise_models_dics_list
- _merge_settings
- _index_from_settings
- _line_style_for_group
- _marker_for_group
- _color_for_group
- _add_es_style_legend
- _create_broken_x_axes
- _split_mask
- _plot_line
- _fill_between
- _scatter_last
- _format_broken_axes
- _add_broken_axis_diagonals
- _format_single_axis
- _legend_settings
- _group_zorder
- _diagnostics_for_model
- _component_config
- _series_from_diagnostics
- _model_data_loss_func_name
- _component_for_model
- _should_plot_component_for_model
- _group_component_stats
- _best_epoch_idx
- _output_name
- plot_data_loss_diagnostics"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DATA_LOSS_DIAGNOSTIC_COMPONENTS = {
    "mse_mean": {
        "label": "MSE",
        "ylabel": "Mean squared residual",
        "yscale": "log",
    },
    "gls_pred_mean": {
        "label": "GLS(pred), unnormalised",
        "ylabel": "Mean weighted residual",
        "yscale": "log",
    },
    "gls_pred_normalised": {
        "label": "GLS(pred), normalised",
        "ylabel": "Weighted mean residual",
        "yscale": "log",
    },
    "gls_true_normalised": {
        "label": "GLS(true), normalised",
        "ylabel": "Weighted mean residual",
        "yscale": "log",
    },
    "gls_pred_to_mse": {
        "label": "GLS(pred) / MSE",
        "ylabel": "Loss scale ratio [a.u.]",
        "yscale": "log",
    },
    "weight_pred_q50": {
        "label": "Pred weight q50",
        "ylabel": r"$|\hat u|^{-2\gamma}$",
        "yscale": "log",
    },
    "weight_pred_q90": {
        "label": "Pred weight q90",
        "ylabel": r"$|\hat u|^{-2\gamma}$",
        "yscale": "log",
    },
    "weight_pred_q99": {
        "label": "Pred weight q99",
        "ylabel": r"$|\hat u|^{-2\gamma}$ [a.u.]",
        "yscale": "log",
    },
    "weight_pred_max": {
        "label": "Pred weight max",
        "ylabel": r"$|\hat u|^{-2\gamma}$",
        "yscale": "log",
    },
    "weight_pred_neff_frac": {
        "label": "Pred weight effective sample fraction",
        "ylabel": r"$N_{\rm eff}/N$ [a.u.]",
        "yscale": "linear",
    },
    "pred_q01": {
        "label": "Prediction q01",
        "ylabel": r"$\hat u$",
        "yscale": "linear",
    },
    "pred_min": {
        "label": "Prediction min",
        "ylabel": r"$\hat u$",
        "yscale": "linear",
    },
    "pred_lt_1e-3_frac": {
        "label": r"Fraction $\hat u < 10^{-3}$",
        "ylabel": "Fraction",
        "yscale": "linear",
    },
    "gls_pred_top1_frac": {
        "label": "Top 1% GLS(pred) contribution",
        "ylabel": "Contribution fraction",
        "yscale": "linear",
    },
    "gls_pred_top5_frac": {
        "label": "Top 5% GLS(pred) contribution",
        "ylabel": "Contribution fraction",
        "yscale": "linear",
    },
    "mse_top1_frac": {
        "label": "Top 1% MSE contribution",
        "ylabel": "Contribution fraction",
        "yscale": "linear",
    },
    "mse_top5_frac": {
        "label": "Top 5% MSE contribution",
        "ylabel": "Contribution fraction",
        "yscale": "linear",
    },
    "actual_data_loss_top1_frac": {
        "label": "Top 1% actual data-loss contribution",
        "ylabel": "Data loss top 1% [a.u.]",
        "yscale": "linear",
    },
    "actual_data_loss_top5_frac": {
        "label": "Top 5% actual data-loss contribution",
        "ylabel": "Data loss top 5% [a.u.]",
        "yscale": "linear",
    },
}


DEFAULT_SETTINGS = {
    "name": None,
    "split": "val",
    "components": [
        "gls_pred_to_mse",
        "weight_pred_q99",
        "weight_pred_neff_frac",
        "pred_lt_1e-3_frac",
        "actual_data_loss_top1_frac",
        "gls_pred_normalised",
        "gls_true_normalised",
    ],
    "figsize": (7, 5),
    "fill": True,
    "fill_alpha": 0.18,
    "line_width": 2.0,
    "colors": ["#A855F7", "#EC4899", "#F43F5E", "#7E22CE", "#BE185D"],
    "color_indices": None,
    "line_styles": ["-", "--", "-.", ":"],
    "line_style_indices": None,
    "marker_styles": ["o", "s", "D", "^", "v", "P", "*", "X"],
    "marker_indices": None,
    "best_model_markers": True,
    "best_marker_size": 36,
    "best_epoch_termination": True,
    "es_entries": [],
    "group_labels": None,
    "legend": {
        "panel": 2,
        "loc": "best",
        "fontsize": 10,
        "title": None,
        "framealpha": 0.85,
        "ncols": 1,
    },
    "fontsizes": {
        "xaxis": 12,
        "xtick_labels": 10,
        "yaxis": 12,
        "ytick_labels": 10,
    },
    "xscale": "linear",
    "xaxis": None,
    "x_min": None,
    "x_max": None,
    "y_floor": 1e-16,
    "grid": True,
    "show": True,
}


def _is_model_wrapper(value):
    return hasattr(value, "val_loss_list")


def _run_groups_from_models_dics(models_dics, group_label):
    if _is_model_wrapper(models_dics):
        return [(group_label, [models_dics])]

    if not models_dics:
        raise ValueError("models_dics is empty.")

    values = list(models_dics.values())
    if all(_is_model_wrapper(value) for value in values):
        return [(group_label, list(values))]

    if not all(isinstance(value, dict) for value in values):
        raise ValueError(
            "Each group must be a ModelWrapper, {seed: modelWrapper, ...}, "
            "or {config_key: {seed: modelWrapper, ...}}."
        )

    run_groups = []
    for key, model_dic in models_dics.items():
        model_wrappers = list(model_dic.values())
        if not all(_is_model_wrapper(value) for value in model_wrappers):
            raise ValueError(
                "Nested input must be {config_key: {seed: modelWrapper, ...}}."
            )
        label = group_label if len(values) == 1 else str(key)
        run_groups.append((label, model_wrappers))
    return run_groups


def _normalise_models_dics_list(models_dics_list, group_labels=None):
    if _is_model_wrapper(models_dics_list) or isinstance(models_dics_list, dict):
        models_dics_list = [models_dics_list]

    if not isinstance(models_dics_list, list) or not models_dics_list:
        raise ValueError(
            "models_dics_list must be a non-empty ModelWrapper, dict, or list."
        )

    if group_labels is None:
        group_labels = [f"Group {i + 1}" for i in range(len(models_dics_list))]
    elif len(group_labels) != len(models_dics_list):
        raise ValueError("plot_settings['group_labels'] must match models_dics_list.")

    run_groups = []
    for models_dics, group_label in zip(models_dics_list, group_labels):
        run_groups.extend(_run_groups_from_models_dics(models_dics, group_label))
    return run_groups


def _merge_settings(plot_settings):
    plot_settings = plot_settings or {}
    settings = {**DEFAULT_SETTINGS, **plot_settings}
    settings["legend"] = {
        **DEFAULT_SETTINGS["legend"],
        **settings.get("legend", {}),
    }
    settings["fontsizes"] = {
        **DEFAULT_SETTINGS["fontsizes"],
        **settings.get("fontsizes", {}),
    }
    if settings.get("xaxis") is not None:
        settings["xaxis"] = {
            "min": 1,
            "max": 1e5,
            "break": 1e3,
            **settings["xaxis"],
        }
    return settings


def _index_from_settings(settings, key, group_idx):
    indices = settings.get(key)
    if indices is not None and group_idx < len(indices):
        return int(indices[group_idx])
    return group_idx


def _line_style_for_group(settings, group_idx):
    line_styles = settings.get("line_styles") or ["-"]
    style_idx = _index_from_settings(settings, "line_style_indices", group_idx)
    return line_styles[style_idx % len(line_styles)]


def _marker_for_group(settings, group_idx):
    marker_styles = settings.get("marker_styles") or ["o"]
    marker_idx = _index_from_settings(settings, "marker_indices", group_idx)
    return marker_styles[marker_idx % len(marker_styles)]


def _color_for_group(settings, group_idx):
    colors = settings["colors"]
    color_idx = _index_from_settings(settings, "color_indices", group_idx)
    return colors[color_idx % len(colors)]


def _add_es_style_legend(ax, settings):
    es_entries = settings.get("es_entries") or []
    if not es_entries:
        return

    line_styles = settings.get("line_styles") or ["-"]
    marker_styles = settings.get("marker_styles") or ["o"]
    handles = [
        Line2D(
            [0],
            [0],
            color="black",
            lw=settings["line_width"],
            ls=line_styles[i % len(line_styles)],
            marker=marker,
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
        )
        for i, (_, marker) in enumerate(es_entries)
    ]
    labels = [str(label) for label, _ in es_entries]
    legend_cfg = settings["legend"]
    style_legend = ax.legend(
        handles=handles,
        labels=labels,
        title=legend_cfg.get("marker_title", "ES"),
        title_fontsize=legend_cfg.get("title_fontsize", legend_cfg["fontsize"]),
        loc=legend_cfg.get("loc_upd", "upper right"),
        fontsize=legend_cfg["fontsize"],
        framealpha=legend_cfg["framealpha"],
        ncols=legend_cfg.get("marker_ncols", 1),
    )
    ax.add_artist(style_legend)


def _create_broken_x_axes(figsize):
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 5], "wspace": 0.05},
    )
    return fig, ax1, ax2


def _split_mask(x, break_x):
    return x <= break_x, x > break_x


def _plot_line(ax1, ax2, x, y, *, break_x, color, line_width, line_style, zorder):
    mask1, mask2 = _split_mask(x, break_x)
    ax1.plot(
        x[mask1],
        y[mask1],
        color=color,
        lw=line_width,
        ls=line_style,
        zorder=zorder,
    )
    ax2.plot(
        x[mask2],
        y[mask2],
        color=color,
        lw=line_width,
        ls=line_style,
        zorder=zorder,
    )


def _fill_between(ax1, ax2, x, ymin, ymax, *, break_x, color, alpha, zorder):
    mask1, mask2 = _split_mask(x, break_x)
    ax1.fill_between(
        x[mask1],
        ymin[mask1],
        ymax[mask1],
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=zorder,
    )
    ax2.fill_between(
        x[mask2],
        ymin[mask2],
        ymax[mask2],
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=zorder,
    )


def _scatter_last(ax1, ax2, x, y, *, break_x, color, marker, size, zorder):
    if len(x) == 0:
        return
    target_ax = ax1 if x[-1] <= break_x else ax2
    target_ax.scatter(
        x[-1],
        y[-1],
        color=color,
        edgecolors="black",
        linewidths=0.6,
        s=size,
        marker=marker,
        zorder=zorder,
    )


def _format_broken_axes(ax1, ax2, settings, component_cfg):
    fontsizes = settings["fontsizes"]
    xaxis = settings["xaxis"]
    yscale = component_cfg.get("yscale", "log")

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.tick_params(axis="x", labelsize=fontsizes["xtick_labels"])
        ax.tick_params(axis="y", labelsize=fontsizes["ytick_labels"])
        ax.set_facecolor("white")
        if settings["grid"]:
            ax.grid(True, which="both", alpha=0.25)

    ax2.set_xlabel("Epoch", fontsize=fontsizes["xaxis"])
    ax1.set_ylabel(component_cfg["ylabel"], fontsize=fontsizes["yaxis"])
    ax1.set_xlim(left=xaxis["min"], right=xaxis["break"])
    ax2.set_xlim(left=xaxis["break"], right=xaxis["max"])

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)


def _add_broken_axis_diagonals(ax1, ax2):
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)


def _format_single_axis(ax, settings, component_cfg):
    fontsizes = settings["fontsizes"]
    ax.set_xlabel("Epoch", fontsize=fontsizes["xaxis"])
    ax.set_ylabel(component_cfg["ylabel"], fontsize=fontsizes["yaxis"])
    ax.tick_params(axis="x", labelsize=fontsizes["xtick_labels"])
    ax.tick_params(axis="y", labelsize=fontsizes["ytick_labels"])
    ax.set_xscale(settings["xscale"])
    ax.set_yscale(component_cfg.get("yscale", "log"))
    if settings["x_min"] is not None or settings["x_max"] is not None:
        ax.set_xlim(settings["x_min"], settings["x_max"])
    if settings["grid"]:
        ax.grid(True, which="both", alpha=0.25)


def _legend_settings(settings):
    legend_cfg = dict(settings["legend"])
    panel = legend_cfg.pop("panel", None)
    loc_upd = legend_cfg.pop("loc_upd", None)
    marker_title = legend_cfg.pop("marker_title", None)
    marker_ncols = legend_cfg.pop("marker_ncols", None)
    _ = (panel, loc_upd, marker_title, marker_ncols)
    return legend_cfg


def _group_zorder(component_name, group_label):
    if component_name == "actual_data_loss_top1_frac":
        if str(group_label) == "MSE":
            return 3
        if str(group_label) == "GLS":
            return 2
    return 2


def _diagnostics_for_model(model_wrapper, split):
    attr = f"{split}_data_loss_diagnostics"
    diagnostics = getattr(model_wrapper, attr, None)
    if isinstance(diagnostics, dict):
        return diagnostics
    return {}


def _component_config(component):
    if isinstance(component, str):
        cfg = dict(DATA_LOSS_DIAGNOSTIC_COMPONENTS.get(component, {}))
        cfg.setdefault("label", component.replace("_", " "))
        cfg.setdefault("ylabel", component.replace("_", " "))
        cfg.setdefault("yscale", "log")
        return component, cfg

    cfg = dict(component)
    name = cfg.pop("name")
    base = dict(DATA_LOSS_DIAGNOSTIC_COMPONENTS.get(name, {}))
    base.update(cfg)
    base.setdefault("label", name.replace("_", " "))
    base.setdefault("ylabel", name.replace("_", " "))
    base.setdefault("yscale", "log")
    return name, base


def _series_from_diagnostics(diagnostics, component, y_floor):
    y = np.asarray(diagnostics.get(component, []), dtype=float)
    if y.size == 0:
        return None, None
    x = np.asarray(diagnostics.get("epoch", np.arange(y.size)), dtype=float)
    if x.size != y.size:
        x = np.arange(y.size, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return None, None
    x = x[finite]
    y = y[finite]
    y = np.maximum(y, y_floor)
    return x, y


def _model_data_loss_func_name(model_wrapper):
    model = getattr(model_wrapper, "model", None)
    data_loss_func = getattr(model, "data_loss_func", None)
    return getattr(data_loss_func, "__name__", "")


def _component_for_model(model_wrapper, component):
    if component not in {"actual_data_loss_top1_frac", "actual_data_loss_top5_frac"}:
        return component

    suffix = "top1_frac" if component.endswith("top1_frac") else "top5_frac"
    func_name = _model_data_loss_func_name(model_wrapper)
    if func_name == "data_loss_GLS":
        return f"gls_pred_{suffix}"
    if func_name == "data_loss_MSE":
        return f"mse_{suffix}"
    return component


def _should_plot_component_for_model(model_wrapper, component):
    if component != "gls_pred_to_mse":
        return True
    return _model_data_loss_func_name(model_wrapper) == "data_loss_GLS"


def _group_component_stats(model_wrappers, component, split, y_floor):
    series = []
    for model_wrapper in model_wrappers:
        if not _should_plot_component_for_model(model_wrapper, component):
            continue
        diagnostics = _diagnostics_for_model(model_wrapper, split)
        model_component = _component_for_model(model_wrapper, component)
        x, y = _series_from_diagnostics(diagnostics, model_component, y_floor)
        if x is not None:
            series.append((x, y))
    if not series:
        return None

    lengths = [len(y) for _, y in series]
    max_len = max(lengths)
    if max_len == 0:
        return None

    longest_idx = int(np.argmax(lengths))
    x = np.asarray(series[longest_idx][0][:max_len], dtype=float)
    values = np.full((len(series), max_len), np.nan, dtype=np.float64)
    for idx, (_, y) in enumerate(series):
        values[idx, : len(y)] = y

    return {
        "epochs": x,
        "mean": np.nanmean(values, axis=0),
        "min": np.nanmin(values, axis=0),
        "max": np.nanmax(values, axis=0),
        "max_len": max_len,
    }


def _best_epoch_idx(model_wrappers, epochs):
    if len(epochs) == 0:
        return 0

    best_losses = np.asarray(
        [getattr(model, "best_val_loss", np.nan) for model in model_wrappers],
        dtype=np.float64,
    )
    if np.all(np.isnan(best_losses)):
        return len(epochs) - 1

    best_seed_idx = int(np.nanargmin(best_losses))
    best_model = model_wrappers[best_seed_idx]
    best_epoch = getattr(best_model, "last_improved", None)
    if best_epoch is None:
        return len(epochs) - 1

    best_epoch = float(best_epoch)
    idx = int(np.searchsorted(epochs, best_epoch, side="right") - 1)
    return max(0, min(idx, len(epochs) - 1))


def _output_name(base_name, component):
    if base_name is None:
        return None
    path = Path(base_name)
    safe_component = component.replace("/", "_").replace(" ", "_")
    return str(path.with_name(f"{path.stem}_{safe_component}{path.suffix}"))


def plot_data_loss_diagnostics(models_dics_list=None, plot_settings=None, models_dics=None):
    """Plot stored MSE/GLS weighting diagnostics across model groups.

    Diagnostics are produced only for models trained with
    ``storeDataLossDiagnostics=True`` in ``binn__config.py``.
    """

    if models_dics_list is None:
        if models_dics is None:
            raise ValueError("Provide models_dics_list or models_dics.")
        models_dics_list = models_dics

    settings = _merge_settings(plot_settings)
    run_groups = _normalise_models_dics_list(
        models_dics_list,
        settings.get("group_labels"),
    )

    figures = {}
    for component in settings["components"]:
        component_name, component_cfg = _component_config(component)
        use_broken_x = settings.get("xaxis") is not None
        if use_broken_x:
            fig, ax1, ax2 = _create_broken_x_axes(settings["figsize"])
            legend_ax = ax1 if settings["legend"].get("panel", 2) == 1 else ax2
        else:
            fig, ax = plt.subplots(figsize=settings["figsize"])
            legend_ax = ax

        plotted = False
        seen_labels = set()
        line_handles = []
        line_labels = []
        for group_idx, (group_label, model_wrappers) in enumerate(run_groups):
            color = _color_for_group(settings, group_idx)
            line_style = _line_style_for_group(settings, group_idx)
            marker = _marker_for_group(settings, group_idx)
            zorder = _group_zorder(component_name, group_label)
            stats = _group_component_stats(
                model_wrappers,
                component_name,
                settings["split"],
                settings["y_floor"],
            )
            if stats is None:
                continue

            if settings["best_epoch_termination"]:
                final_idx = _best_epoch_idx(model_wrappers, stats["epochs"])
            else:
                final_idx = len(stats["epochs"]) - 1

            end_idx = min(final_idx + 1, len(stats["epochs"]))
            x = stats["epochs"][:end_idx]
            mean = stats["mean"][:end_idx]
            ymin = stats["min"][:end_idx]
            ymax = stats["max"][:end_idx]

            if use_broken_x:
                break_x = settings["xaxis"]["break"]
                _plot_line(
                    ax1,
                    ax2,
                    x,
                    mean,
                    break_x=break_x,
                    color=color,
                    line_width=settings["line_width"],
                    line_style=line_style,
                    zorder=zorder,
                )
                if settings["best_model_markers"]:
                    _scatter_last(
                        ax1,
                        ax2,
                        x,
                        mean,
                        break_x=break_x,
                        color=color,
                        marker=marker,
                        size=settings["best_marker_size"],
                        zorder=zorder + 0.5,
                    )
            else:
                ax.plot(
                    x,
                    mean,
                    color=color,
                    lw=settings["line_width"],
                    ls=line_style,
                    zorder=zorder,
                )
                if settings["best_model_markers"] and len(x):
                    ax.scatter(
                        x[-1],
                        mean[-1],
                        color=color,
                        edgecolors="black",
                        linewidths=0.6,
                        s=settings["best_marker_size"],
                        marker=marker,
                        zorder=zorder + 0.5,
                    )

            if group_label not in seen_labels:
                line_handles.append(
                    Line2D([0], [0], color=color, lw=settings["line_width"])
                )
                line_labels.append(group_label)
                seen_labels.add(group_label)
            if settings["fill"] and len(model_wrappers) > 1:
                if use_broken_x:
                    _fill_between(
                        ax1,
                        ax2,
                        x,
                        ymin,
                        ymax,
                        break_x=settings["xaxis"]["break"],
                        color=color,
                        alpha=settings["fill_alpha"],
                        zorder=zorder - 0.2,
                    )
                else:
                    ax.fill_between(
                        x,
                        ymin,
                        ymax,
                        color=color,
                        alpha=settings["fill_alpha"],
                        linewidth=0,
                        zorder=zorder - 0.2,
                    )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        if use_broken_x:
            _format_broken_axes(ax1, ax2, settings, component_cfg)
            _add_broken_axis_diagonals(ax1, ax2)
        else:
            _format_single_axis(ax, settings, component_cfg)

        legend_cfg = _legend_settings(settings)
        line_legend = legend_ax.legend(
            handles=line_handles,
            labels=line_labels,
            **legend_cfg,
        )
        legend_ax.add_artist(line_legend)
        _add_es_style_legend(ax2 if use_broken_x else ax, settings)
        fig.tight_layout()

        output_name = _output_name(settings["name"], component_name)
        if output_name:
            Path(output_name).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_name, bbox_inches="tight", dpi=300)

        if settings["show"]:
            plt.show()
        else:
            plt.close(fig)

        figures[component_name] = fig

    return figures
