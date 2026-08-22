"""Running-min validation-loss plotting helpers with broken x-axes.

Contents
--------
- _merge_plot_settings
- _prepare_run_data_with_styles
- _create_broken_x_axes
- _set_log_safe_xlim
- _marker_for_run
- _positive_for_log
- _plot_runs_on_broken_axes
- _format_axes
- _add_broken_axis_diagonals
- _apply_grid
- _add_line_legend
- _add_es_marker_legend
- plot_running_min_loss_component_broken_x_log_lst"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .prepare_model_loss import prepare_model_run_data


DEFAULT_SETTINGS = {
    "xaxis": {"min": 1, "max": 1e5, "break": 1e3},
    "legend": {
        "panel": 1,          # 1 → ax1, 2 → ax2
        "loc": (0.05, 0.95),
        "loc_upd": (0.35, 0.95),
        "fontsize": 10,
        "ncols": 1,
        "framealpha": 0.8,
        "title": "$W_u$",   # legend title
        "marker_title": "$ES$",
    },
    "name": "running_min_val_loss_broken_xaxis_loglog.png",
    "fill": True,
    "line_lengths": {"hlength": 10000, "vlength_factor": 2.0},
    "line_widths_on_axis": {"hwidth": 1, "vwidth": 1},
    "line_width": 1.5,
    "fontsizes": {
        "xaxis": 12,
        "xtick_labels": 10,
        "yaxis": 12,
        "ytick_labels": 10,
    },
    "figsize": (7, 5),
    "ylabel": "Minimum validation loss [a.u.]",
    "y_floor": 1e-16,
    "y_lim": None,
    "grid": True,
    "es_entries": [
        ("1000", "o"),
        ("2000", "s"),
        ("3000", "D"),
    ],
}


def _merge_plot_settings(plot_settings):
    """Merge user-supplied plot_settings with DEFAULT_SETTINGS."""
    plot_settings = plot_settings or {}
    if "y_lim" not in plot_settings and "ylim" in plot_settings:
        plot_settings = {**plot_settings, "y_lim": plot_settings["ylim"]}
    settings = {**DEFAULT_SETTINGS, **plot_settings}

    xaxis = {**DEFAULT_SETTINGS["xaxis"], **settings.get("xaxis", {})}
    legend = {**DEFAULT_SETTINGS["legend"], **settings.get("legend", {})}
    name = settings.get("name", DEFAULT_SETTINGS["name"])
    fill = settings.get("fill", DEFAULT_SETTINGS["fill"])
    line_width = settings.get("line_width", DEFAULT_SETTINGS["line_width"])
    figsize = settings.get("figsize", DEFAULT_SETTINGS["figsize"])

    line_width_on_axis = {
        **DEFAULT_SETTINGS["line_widths_on_axis"],
        **settings.get("line_widths_on_axis", {}),
    }
    line_lengths = {
        **DEFAULT_SETTINGS["line_lengths"],
        **settings.get("line_lengths", {}),
    }
    fontsizes = {
        **DEFAULT_SETTINGS["fontsizes"],
        **settings.get("fontsizes", {}),
    }
    es_entries = settings.get("es_entries", DEFAULT_SETTINGS.get("es_entries", []))
    ylabel = settings.get("ylabel", DEFAULT_SETTINGS["ylabel"])
    y_floor = settings.get("y_floor", DEFAULT_SETTINGS["y_floor"])
    y_lim = settings.get("y_lim", DEFAULT_SETTINGS["y_lim"])
    grid = settings.get("grid", DEFAULT_SETTINGS["grid"])

    return {
        "settings": settings,
        "xaxis": xaxis,
        "legend": legend,
        "name": name,
        "fill": fill,
        "line_width": line_width,
        "figsize": figsize,
        "line_width_on_axis": line_width_on_axis,
        "line_lengths": line_lengths,
        "fontsizes": fontsizes,
        "es_entries": es_entries,
        "ylabel": ylabel,
        "y_floor": y_floor,
        "y_lim": y_lim,
        "grid": grid,
    }


def _prepare_run_data_with_styles(
    models_dics_list,
    plot_params,
    marker_styles,
    line_styles,
    loss_attr="val_loss_list",
):
    """
    Prepare run data from all model dictionaries and attach marker/linestyle.
    Also tag each run with an 'es_index' so we can map to es_entries.

    Returns:
        run_data_all (list[dict])
        label_to_color (dict[str, color])
    """
    run_data_all = []
    label_to_color = {}

    for i, models_dics in enumerate(models_dics_list):
        marker = marker_styles[i % len(marker_styles)]
        linestyle = line_styles[i % len(line_styles)]

        run_data, _ = prepare_model_run_data(
            models_dics,
            plot_params=plot_params,
            loss_attr=loss_attr,
        )

        for run in run_data:
            run["markerstyle"] = marker
            run["linestyle"] = linestyle
            # ES index per group: models_dics_list[i] ↔ es_entries[i]
            run["es_index"] = i

            label = run["label"]
            if label not in label_to_color:
                label_to_color[label] = run["color"]

        run_data_all.extend(run_data)

    # If you ever want sorting by performance, uncomment:
    # run_data_all.sort(key=lambda x: x["hticks"])

    return run_data_all, label_to_color


def _create_broken_x_axes(figsize):
    """Create a figure with two subplots sharing the y-axis (broken x-axis layout)."""
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 5], "wspace": 0.05},
    )
    return fig, ax1, ax2


def _set_log_safe_xlim(ax, left, right):
    if left <= 0:
        left = 1.0
    ax.set_xlim(left=left, right=right)


def _marker_for_run(run, es_entries):
    """
    Choose marker for the best-epoch scatter based on es_entries and group index.

    models_dics_list[i] is associated with es_entries[i], so each run carries
    an 'es_index' telling us which ES setting it belongs to.
    """
    es_idx = run.get("es_index", None)
    if es_idx is not None and 0 <= es_idx < len(es_entries):
        # es_entries[es_idx] = (label_es, marker)
        return es_entries[es_idx][1]
    # Fallback to the group marker if out of range
    return run.get("markerstyle", "o")


def _positive_for_log(values, y_floor):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0, y_floor, values)


def _plot_runs_on_broken_axes(
    ax1,
    ax2,
    run_data_all,
    break_x,
    fill,
    line_width,
    es_entries,
    y_floor,
):
    """Plot all runs on the two axes, with optional fill between min and max."""
    for run in run_data_all:
        x = np.where(run["epochs"] == 0, 1e-1, run["epochs"])
        y_min = _positive_for_log(run["running_min_min"], y_floor)
        color = run["color"]
        linestyle = run.get("linestyle", "-")

        mask1 = x <= break_x
        mask2 = x > break_x

        # No labels here → legend is built manually
        ax1.plot(
            x[mask1],
            y_min[mask1],
            color=color,
            linestyle=linestyle,
            lw=line_width,
        )
        ax2.plot(
            x[mask2],
            y_min[mask2],
            color=color,
            linestyle=linestyle,
            lw=line_width,
        )

        # Best-model marker (black outline), using ES-based marker
        best_idx = run["best_epoch_seed_idx"][0]
        marker = _marker_for_run(run, es_entries)
        ax2.scatter(
            x[best_idx],
            y_min[best_idx],
            facecolors=color,
            edgecolors="black",
            linewidths=0.8,
            s=40,
            zorder=5,
            marker=marker,
        )

        if fill:
            y_max = _positive_for_log(run["running_min_max"], y_floor)
            ax1.fill_between(
                x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color
            )
            ax2.fill_between(
                x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color
            )


def _format_axes(ax1, ax2, xaxis, fontsizes, ylabel, y_lim=None):
    """Set axis scales, limits, labels, and tick fonts."""
    x_min = xaxis["min"]
    x_max = xaxis["max"]
    break_x = xaxis["break"]

    xfont = fontsizes["xaxis"]
    xtick_font = fontsizes["xtick_labels"]
    yfont = fontsizes["yaxis"]
    ytick_font = fontsizes["ytick_labels"]

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_facecolor("white")

    ax2.set_xlabel("Epoch", fontsize=xfont)
    _set_log_safe_xlim(ax1, x_min, break_x)
    _set_log_safe_xlim(ax2, break_x, x_max)
    ax1.set_ylabel(ylabel, fontsize=yfont)

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    ax1.yaxis.set_tick_params(labelsize=ytick_font)
    ax1.xaxis.set_tick_params(labelsize=xtick_font)
    ax2.xaxis.set_tick_params(labelsize=xtick_font)
    if y_lim is not None:
        ax1.set_ylim(y_lim)
        ax2.set_ylim(y_lim)


def _add_broken_axis_diagonals(ax1, ax2):
    """Draw diagonal lines to indicate the broken x-axis."""
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)


def _apply_grid(ax1, ax2, grid):
    if grid:
        for ax in (ax1, ax2):
            ax.grid(True, which="both", ls="-", lw=0.5, alpha=0.7)


def _add_line_legend(ax1, ax2, label_to_color, line_width, legend_cfg):
    """Add legend for colored lines (one entry per label)."""
    legend_title = legend_cfg.get("title", "W_u")
    def _sort_key(label):
        try:
            return (0, float(label))
        except (TypeError, ValueError):
            return (1, str(label))

    sorted_labels = sorted(label_to_color.keys(), key=_sort_key)

    print("SORTED", sorted_labels)

    line_handles = [
        Line2D(
            [0],
            [0],
            color=label_to_color[lbl],
            lw=line_width,
        )
        for lbl in sorted_labels
    ]

    line_legend_kwargs = dict(
        handles=line_handles,
        labels=sorted_labels,
        title=legend_title,
        fontsize=legend_cfg["fontsize"],
        title_fontsize=legend_cfg["fontsize"],
        ncols=legend_cfg["ncols"],
        framealpha=legend_cfg["framealpha"],
    )

    if legend_cfg["panel"] == 1:
        line_legend = ax1.legend(
            bbox_to_anchor=legend_cfg["loc"],
            **line_legend_kwargs,
        )
        ax1.add_artist(line_legend)
    else:
        line_legend = ax2.legend(
            bbox_to_anchor=legend_cfg["loc"],
            **line_legend_kwargs,
        )
        ax2.add_artist(line_legend)


def _add_es_marker_legend(ax2, es_entries, legend_cfg):
    """Add separate legend for ES line styles and marker shapes."""
    line_styles = ["-", "--", "-.", ":"]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=mk,
            linestyle=line_styles[i % len(line_styles)],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=8,
        )
        for i, (label_es, mk) in enumerate(es_entries)
    ]
    marker_labels = [lab for (lab, _) in es_entries]

    marker_legend = ax2.legend(
        handles=marker_handles,
        labels=marker_labels,
        bbox_to_anchor=legend_cfg["loc_upd"],
        fontsize=legend_cfg["fontsize"],
        title=legend_cfg.get("marker_title", "ES"),
        title_fontsize=legend_cfg["fontsize"],
        framealpha=legend_cfg["framealpha"],
    )
    ax2.add_artist(marker_legend)


def plot_running_min_loss_component_broken_x_log_lst(
    models_dics_list,
    plot_params=None,
    plot_settings=None,
    loss_attr="val_constraint_loss_list",
):
    """
    Plot a running-min curve for any recorded loss-component list.

    Examples
    --------
    loss_attr="val_constraint_loss_list"
    loss_attr="val_D_bound_loss_list"
    loss_attr="val_D_mono_loss_list"
    loss_attr="val_G_bound_loss_list"
    loss_attr="val_G_mono_loss_list"
    loss_attr="train_D_mono_loss_list"
    """
    plot_settings = {
        "name": f"running_min_{loss_attr}_broken_xaxis_loglog.png",
        "ylabel": f"Running min {loss_attr.replace('_', ' ')} [a.u]",
        **(plot_settings or {}),
    }

    cfg = _merge_plot_settings(plot_settings)
    xaxis = cfg["xaxis"]
    legend = cfg["legend"]
    name = cfg["name"]
    fill = cfg["fill"]
    line_width = cfg["line_width"]
    figsize = cfg["figsize"]
    line_width_on_axis = cfg["line_width_on_axis"]
    line_lengths = cfg["line_lengths"]
    fontsizes = cfg["fontsizes"]
    es_entries = cfg["es_entries"]
    ylabel = cfg["ylabel"]
    y_floor = cfg["y_floor"]
    y_lim = cfg["y_lim"]
    grid = cfg["grid"]

    hlength = line_lengths["hlength"]
    vlength_factor = line_lengths["vlength_factor"]
    hwidth = line_width_on_axis["hwidth"]
    vwidth = line_width_on_axis["vwidth"]
    _ = (hlength, vlength_factor, hwidth, vwidth)

    break_x = xaxis["break"]
    x_min = xaxis["min"]
    x_max = xaxis["max"]
    _ = (x_min, x_max)

    marker_styles = ["o", "s", "D", "^", "v", "P", "*", "X"]
    line_styles = ["-", "--", "-.", ":"]

    run_data_all, label_to_color = _prepare_run_data_with_styles(
        models_dics_list,
        plot_params,
        marker_styles,
        line_styles,
        loss_attr=loss_attr,
    )

    fig, ax1, ax2 = _create_broken_x_axes(figsize)
    _plot_runs_on_broken_axes(
        ax1,
        ax2,
        run_data_all,
        break_x=break_x,
        fill=fill,
        line_width=line_width,
        es_entries=es_entries,
        y_floor=y_floor,
    )
    _format_axes(ax1, ax2, xaxis, fontsizes, ylabel, y_lim)
    _add_broken_axis_diagonals(ax1, ax2)
    _apply_grid(ax1, ax2, grid)
    _add_line_legend(ax1, ax2, label_to_color, line_width, legend)
    _add_es_marker_legend(ax2, es_entries, legend)

    fig.tight_layout()

    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()

    return {
        "name": name,
        "legend": legend,
        "fill": fill,
        "xaxis": xaxis,
        "grid": grid,
        "loss_attr": loss_attr,
        "num_runs": len(run_data_all),
    }
