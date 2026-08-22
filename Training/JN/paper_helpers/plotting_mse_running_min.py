"""Running-min diffusion MSE plotting helpers with broken x-axes.

Contents
--------
- _default_diff_plot_settings
- _merge_diff_plot_settings
- _gather_all_run_data
- _create_broken_x_axes
- _set_log_safe_xlim
- _marker_for_run
- _plot_diffusion_runs
- _format_broken_x_axes
- _add_broken_axis_diagonals
- _apply_grid
- _add_line_legend
- _add_marker_legend
- plot_running_min_MSE_diff_loss_broken_x_log_lst"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from .prepare_diff_mse import prepare_diff_run_data


def _default_diff_plot_settings():
    """Return default settings for the diffusion MSE plot."""
    return {
        "xaxis": {"min": 1, "max": 1e5, "break": 1e3},
        "legend": {
            "panel": 1,          # 1 → ax1, 2 → ax2
            "loc": (0.05, 0.95),
            "loc_upd": (0.35, 0.95),
            "fontsize": 10,
            "ncols": 1,
            "marker_ncols": 1,
            "framealpha": 0.8,
            "allow_overlap": True,
            "title": "$W_u$",   # legend title
            "marker_title": "$ES$",
        },
        "name": "running_min_diff_loss_broken_xaxis_loglog.png",
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
        "ylabel": "Minimum diffusion MSE",
        "figsize": (7, 5),
        "grid": True,
        "es_entries": [
            ("1000", "o"),
            ("2000", "s"),
            ("3000", "D"),
        ],
    }


def _merge_diff_plot_settings(plot_settings):
    """Merge user-provided settings with defaults and normalize structure."""
    defaults = _default_diff_plot_settings()
    settings = {**defaults, **(plot_settings or {})}

    xaxis = {**defaults["xaxis"], **settings.get("xaxis", {})}
    legend = {**defaults["legend"], **settings.get("legend", {})}
    line_width_on_axis = {
        **defaults["line_widths_on_axis"],
        **settings.get("line_widths_on_axis", {}),
    }
    line_lengths = {
        **defaults["line_lengths"],
        **settings.get("line_lengths", {}),
    }
    fontsizes = {**defaults["fontsizes"], **settings.get("fontsizes", {})}
    es_entries = settings.get("es_entries", defaults.get("es_entries", []))
    ylabel = settings.get("ylabel", defaults["ylabel"])
    ylim = settings.get("ylim", defaults.get("ylim", None))
    grid = settings.get("grid", defaults["grid"])

    merged = {
        "xaxis": xaxis,
        "legend": legend,
        "name": settings.get("name", defaults["name"]),
        "fill": settings.get("fill", defaults["fill"]),
        "line_width": settings.get("line_width", defaults["line_width"]),
        "figsize": settings.get("figsize", defaults["figsize"]),
        "line_width_on_axis": line_width_on_axis,
        "line_lengths": line_lengths,
        "fontsizes": fontsizes,
        "es_entries": es_entries,
        "ylabel": ylabel,
        "ylim": ylim,
        "grid": grid,
    }
    return merged


def _gather_all_run_data(
    models_dics_list,
    plot_params,
    marker_styles,
    line_styles,
    restrict_to_central_90=False,
):
    """
    Prepare all run data across groups, assigning markers and linestyles per group.

    Returns:
        all_run_data: list of run dicts
        label_to_color: mapping from label → color for legend
    """
    all_run_data = []
    label_to_color = {}

    for i, models_dics in enumerate(models_dics_list):
        group_runs = prepare_diff_run_data(
            models_dics,
            plot_params=plot_params,
            restrict_to_central_90=restrict_to_central_90,
        )

        marker = marker_styles[i % len(marker_styles)]
        linestyle = line_styles[i % len(line_styles)]

        for run in group_runs:
            # Style per group
            run["markerstyle"] = marker
            run["linestyle"] = linestyle
            # ES index per group: models_dics_list[i] ↔ es_entries[i]
            run["es_index"] = i

            label = run["label"]
            if label not in label_to_color:
                label_to_color[label] = run["color"]

        all_run_data.extend(group_runs)

    # If you want to restore the original sort by hticks, uncomment:
    # all_run_data.sort(key=lambda x: x["hticks"])

    return all_run_data, label_to_color


def _create_broken_x_axes(figsize):
    """Create and return a figure with two subplots sharing y, for broken x-axis."""
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
    # Fallback to the line/group marker if out of range
    return run.get("markerstyle", "o")


def _plot_diffusion_runs(
    ax1,
    ax2,
    all_run_data,
    break_x,
    fill,
    line_width,
    es_entries,
):
    """Plot each run on the broken x-axis, with optional fill and best-epoch marker."""
    for run in all_run_data:
        sf = run.get("prediction_frequency", 1)
        # Epoch indices are scaled by inferred saved-prediction cadence.
        x = np.where(run["epochs"] * sf == 0, 1e-1, run["epochs"] * sf)

        # Running-min stats for diffusion MSE
        y_mean = run["mean_vals"]
        y_min = run["min_vals"]
        y_max = run["max_vals"]

        color = run["color"]
        linestyle = run.get("linestyle", "-")

        mask1 = x <= break_x
        mask2 = x > break_x

        # central line = mean; shaded min–max band
        ax1.plot(
            x[mask1],
            y_mean[mask1],
            color=color,
            linestyle=linestyle,
            lw=line_width,
        )
        ax2.plot(
            x[mask2],
            y_mean[mask2],
            color=color,
            linestyle=linestyle,
            lw=line_width,
        )

        if fill:
            ax1.fill_between(
                x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color
            )
            ax2.fill_between(
                x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color
            )

        # Best-model marker: use best_epoch_idx if available, otherwise argmin
        best_idx, best_seed = run.get("best_epoch_seed_idx", [None, None])
        if best_idx is None and len(y_mean) > 0:
            best_idx = int(np.argmin(y_mean))

        if best_idx is not None and 0 <= best_idx < len(x):
            marker = _marker_for_run(run, es_entries)
        
            ax2.scatter(
                run["x_orig"][run["best_epoch_idx_orig"]],
                run["final_mses"][best_seed],
                facecolors=color,
                edgecolors="black",
                linewidths=0.8,
                s=40,
                zorder=5,
                marker=marker,
            )


def _format_broken_x_axes(ax1, ax2, xaxis, fontsizes):
    """Apply scales, labels, limits, and tick formatting to the axes."""
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
    ax1.set_ylabel(r"Diffusion MSE [mm$^4$ days$^{-2}$]", fontsize=yfont)

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    ax1.yaxis.set_tick_params(labelsize=ytick_font)
    ax1.xaxis.set_tick_params(labelsize=xtick_font)
    ax2.xaxis.set_tick_params(labelsize=xtick_font)


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


def _add_line_legend(ax1, ax2, label_to_color, legend, line_width):
    """Add legend for line colors (one entry per label)."""
    legend_title = legend.get("title", "$W_u$")

    def _sort_key(label):
        try:
            return (0, float(label))
        except (TypeError, ValueError):
            return (1, str(label))

    sorted_labels = sorted(label_to_color.keys(), key=_sort_key)
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
        fontsize=legend["fontsize"],
        title_fontsize=legend["fontsize"],
        ncols=legend["ncols"],
        framealpha=legend["framealpha"],
    )

    if legend["panel"] == 1:
        line_legend = ax1.legend(
            loc="upper left",
            bbox_to_anchor=legend["loc"],
            **line_legend_kwargs,
        )
        line_legend.set_in_layout(False)
        ax1.add_artist(line_legend)
    else:
        line_legend = ax2.legend(
            loc="upper left",
            bbox_to_anchor=legend["loc"],
            **line_legend_kwargs,
        )
        line_legend.set_in_layout(False)
        ax2.add_artist(line_legend)


def _add_marker_legend(ax2, es_entries, legend):
    """Add separate legend for ES line styles and markers."""
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
        loc="upper left",
        bbox_to_anchor=legend["loc_upd"],
        fontsize=legend["fontsize"],
        title=legend.get("marker_title", "ES"),
        title_fontsize=legend["fontsize"],
        ncols=legend.get("marker_ncols", 1),
        framealpha=legend["framealpha"],
    )
    marker_legend.set_in_layout(False)
    ax2.add_artist(marker_legend)


def plot_running_min_MSE_diff_loss_broken_x_log_lst(
    models_dics_list,
    plot_params=None,
    plot_settings=None,
    restrict_to_central_90=False,
):
    """
    Plot running-min diffusion MSE loss for multiple experiment groups,
    using a broken log–log x-axis. Each group gets its own marker/linestyle.
    """
    # ---------- Settings ----------
    merged = _merge_diff_plot_settings(plot_settings)
    xaxis = merged["xaxis"]
    legend = merged["legend"]
    name = merged["name"]
    fill = merged["fill"]
    line_width = merged["line_width"]
    figsize = merged["figsize"]
    line_lengths = merged["line_lengths"]
    line_width_on_axis = merged["line_width_on_axis"]
    fontsizes = merged["fontsizes"]
    es_entries = merged["es_entries"]
    ylim = merged["ylim"]
    grid = merged["grid"]

    # Unused, but kept for backward-compatibility (if you log these elsewhere)
    _ = (
        line_lengths["hlength"],
        line_lengths["vlength_factor"],
        line_width_on_axis["hwidth"],
        line_width_on_axis["vwidth"],
    )

    # ---------- Marker / linestyle cycles ----------
    marker_styles = ["o", "s", "D", "^", "v", "P", "*", "X"]
    line_styles = ["-", "--", "-.", ":"]

    # ---------- Collect all runs ----------
    all_run_data, label_to_color = _gather_all_run_data(
        models_dics_list,
        plot_params,
        marker_styles,
        line_styles,
        restrict_to_central_90=restrict_to_central_90,
    )

    # ---------- Figure + axes ----------
    fig, ax1, ax2 = _create_broken_x_axes(figsize)

    # ---------- Plot runs ----------
    _plot_diffusion_runs(
        ax1,
        ax2,
        all_run_data,
        xaxis["break"],
        fill, 
        line_width,
        es_entries,
    )

    # ---------- Formatting & decorations ----------
    _format_broken_x_axes(ax1, ax2, xaxis, fontsizes)
    _add_broken_axis_diagonals(ax1, ax2)
    _apply_grid(ax1, ax2, grid)
    _add_line_legend(ax1, ax2, label_to_color, legend, line_width)
    _add_marker_legend(ax2, es_entries, legend)

    fig.tight_layout()
    if ylim:
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)

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
        "num_runs": len(all_run_data),
    }
