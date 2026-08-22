"""Final-loss summary plotting helpers for grouped training runs.

Contents
--------
- _collect_grouped_run_data
- plot_final_loss_lst
- plot_final_loss_lst_colLayout"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import sys

from .prepare_model_loss import prepare_model_run_data


def _collect_grouped_run_data(models_dics_list, plot_params=None):
    grouped_data = []
    all_labels = []

    for models_dics in models_dics_list:
        run_data, raw_timing_data = prepare_model_run_data(models_dics, plot_params=plot_params)
        timing_by_label = {d["label"]: d for d in raw_timing_data}
        combined = {}
        for d in run_data:
            timing = timing_by_label.get(d["label"], {})
            combined[d["label"]] = {**d, **timing}
        grouped_data.append(combined)

        for lbl in combined.keys():
            if lbl not in all_labels:
                all_labels.append(lbl)

    return grouped_data, all_labels

def plot_final_loss_lst(
    models_dics_list,
    label_list=None,
    plot_params=None,
    name=None,
    bbox_to_anchor=(0.5, 0.5),
    legend_size=16,
    legend_ncols=1,
    y_label="Validation loss [a.u.]",
    hatch_patterns=(
        "", "////", "\\\\\\\\", "xxxx", "++++", "....", "ooo", "***"
    ),
    figsize=(7, 5),
    colors=None,
    hatch_linewidth=1,
    legend_title="ES",
    legend_bool=True,
    x_label=r"$N_u$",
    axis_fontsizes=None,
    ylim=None,
    grid=False,
    grid_alpha=0
):
    """
    Bars use:
      - consistent fill colors across groups (same label → same color)
      - black hatch patterns per group (different texture per group)
      - log y-axis
      - spacing between bars of same label

    Error bars span min–max across repeats (using min_best_model_loss / max_best_model_loss
    if available, otherwise falling back to mean ± std_best_model_loss).
    """

    import matplotlib as mpl
    from matplotlib.patches import Patch

    if axis_fontsizes is None:
        axis_fontsizes = {
            "xaxis": 15,
            "xtick_labels": 15,
            "yaxis": 15,
            "ytick_labels": 15,
        }

    # ---------- Blue palette for *labels* ----------
    if colors is None:
        colors = [
            "#0033A0",
            "#1E90FF",
            "#6699CC",
            "#A4C8E1",
            "#D6EAF8",
        ]

    # Label list is group names, not x-axis labels
    if label_list is None:
        label_list = [f"Group {i+1}" for i in range(len(models_dics_list))]

    # ---------- Gather Data ----------
    grouped_data = []
    all_labels = []

    for models_dics in models_dics_list:
        run_data, _ = prepare_model_run_data(models_dics, plot_params=plot_params)
        label_to_data = {d["label"]: d for d in run_data}
        grouped_data.append(label_to_data)

        for lbl in label_to_data.keys():
            if lbl not in all_labels:
                all_labels.append(lbl)

    print("All x-axis labels found (ordered):", all_labels)


    # x-axis labels sorted numerically
    num_labels = len(all_labels)
    num_groups = len(models_dics_list)

    # ---------- Assign consistent colors per label ----------
    label_color_map = {
        label: colors[i % len(colors)]
        for i, label in enumerate(all_labels)
    }

    # ---------- Bar Geometry ----------
    group_width = 0.8
    slot_width = group_width / num_groups
    bar_width = slot_width * 0.8
    group_offsets = (
        np.arange(num_groups) * slot_width - group_width / 2 + slot_width / 2
    )

    fig, ax1 = plt.subplots(figsize=figsize)

    # ---------- Hatch settings ----------
    old_hatch_lw = mpl.rcParams["hatch.linewidth"]
    old_hatch_color = mpl.rcParams["hatch.color"]

    mpl.rcParams["hatch.linewidth"] = hatch_linewidth
    mpl.rcParams["hatch.color"] = "black"

    try:
        # ---------- Plot bars ----------
        for i, label in enumerate(all_labels):
            label_color = label_color_map[label]

            for j, group_dict in enumerate(grouped_data):
                if label not in group_dict:
                    continue

                d = group_dict[label]

                mean_val = d["mean_best_model_loss"]

                # Prefer explicit min/max if present; otherwise fall back to std
                if "min_best_model_loss" in d and "max_best_model_loss" in d:
                    min_val = d["min_best_model_loss"]
                    max_val = d["max_best_model_loss"]
                else:
                    min_val = mean_val - d["std_best_model_loss"]
                    max_val = mean_val + d["std_best_model_loss"]

                # Asymmetric error bars: [lower, upper] relative to mean
                lower_err = mean_val - min_val
                upper_err = max_val - mean_val
                yerr = np.array([[lower_err], [upper_err]])

                # hatch distinguishes the *group*
                hatch = hatch_patterns[j % len(hatch_patterns)]
                x_pos = i + group_offsets[j]

                ax1.bar(
                    x_pos,
                    mean_val,
                    yerr=yerr,
                    width=bar_width,
                    capsize=5,
                    facecolor=label_color,    # consistent across groups
                    alpha=0.7,
                    edgecolor="black",
                    hatch=hatch,              # group distinction
                    linewidth=1.5,
                    error_kw=dict(
                        ecolor="k",
                        linewidth=3,
                    ),
                )

        # ---------- Legend ----------
        # Shows only groups (hatch patterns)
        custom_handles = [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=hatch_patterns[i % len(hatch_patterns)],
                linewidth=1.5,
                label=label_list[i],
            )
            for i in range(num_groups)
        ]

        if legend_bool:
            ax1.legend(
                custom_handles,
                label_list,
                loc="upper left",
                bbox_to_anchor=bbox_to_anchor,
                fontsize=legend_size,
                ncols=legend_ncols,
                title_fontsize=legend_size,
                title=legend_title,
            )

        # ---------- Axes ----------
        ax1.set_ylabel(y_label, fontsize=axis_fontsizes["yaxis"])
        ax1.set_yscale("log")
        if ylim:
            ax1.set_ylim(ylim)
        ax1.grid(bool(grid), axis="y", which="both", linestyle="-", linewidth=0.5, alpha=grid_alpha)

        ax1.tick_params(axis="y", which="major", length=10)
        ax1.tick_params(axis="y", which="minor", length=5)
        ax1.tick_params(axis="y", labelsize=axis_fontsizes["ytick_labels"])

        ax1.set_xticks(np.arange(num_labels))
        ax1.set_xticklabels(
            all_labels,
            rotation=0,
            ha="center",
            fontsize=axis_fontsizes["xtick_labels"],
        )

        ax1.set_xlabel(x_label, fontsize=axis_fontsizes["xaxis"])

        fig.tight_layout()

        if name:
            plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
            print("saved plot:", name)

        plt.show()

    finally:
        mpl.rcParams["hatch.linewidth"] = old_hatch_lw
        mpl.rcParams["hatch.color"] = old_hatch_color


def plot_final_loss_lst_colLayout(
    models_dics_list,
    label_list=None,
    plot_params=None,
    name=None,
    bbox_to_anchor=(0.5, 0.5),
    legend_size=16,
    legend_ncols=1,
    y_label="Validation loss [a.u.]",
    hatch_patterns=(
        "", "////", "\\\\\\\\", "xxxx", "++++", "....", "ooo", "***"
    ),
    figsize=(7, 5),
    colors=None,
    hatch_linewidth=1,
    legend_title="ES",
    x_label=r"$N_u$",
    axis_fontsizes=None,
    ylim=None,
    show_xlabel=True,
    show_xtick_labels=True,
    grid=False,
    grid_alpha=0
):
    """
    Column-layout variant of plot_final_loss_lst.

    Keeps a fixed canvas geometry on export and allows the x-axis label/tick
    labels to be suppressed for panels that will share a column in a composite
    figure.
    """

    import matplotlib as mpl
    from matplotlib.patches import Patch

    if axis_fontsizes is None:
        axis_fontsizes = {
            "xaxis": 15,
            "xtick_labels": 15,
            "yaxis": 15,
            "ytick_labels": 15,
        }

    if colors is None:
        colors = [
            "#0033A0",
            "#1E90FF",
            "#6699CC",
            "#A4C8E1",
            "#D6EAF8",
        ]

    if label_list is None:
        label_list = [f"Group {i+1}" for i in range(len(models_dics_list))]

    grouped_data = []
    all_labels = []

    for models_dics in models_dics_list:
        run_data, _ = prepare_model_run_data(models_dics, plot_params=plot_params)
        label_to_data = {d["label"]: d for d in run_data}
        grouped_data.append(label_to_data)

        for lbl in label_to_data.keys():
            if lbl not in all_labels:
                all_labels.append(lbl)

    print("All x-axis labels found (ordered):", all_labels)

    num_labels = len(all_labels)
    num_groups = len(models_dics_list)

    label_color_map = {
        label: colors[i % len(colors)]
        for i, label in enumerate(all_labels)
    }

    group_width = 0.8
    slot_width = group_width / num_groups
    bar_width = slot_width * 0.8
    group_offsets = (
        np.arange(num_groups) * slot_width - group_width / 2 + slot_width / 2
    )

    fig, ax1 = plt.subplots(figsize=figsize)

    old_hatch_lw = mpl.rcParams["hatch.linewidth"]
    old_hatch_color = mpl.rcParams["hatch.color"]

    mpl.rcParams["hatch.linewidth"] = hatch_linewidth
    mpl.rcParams["hatch.color"] = "black"

    try:
        for i, label in enumerate(all_labels):
            label_color = label_color_map[label]

            for j, group_dict in enumerate(grouped_data):
                if label not in group_dict:
                    continue

                d = group_dict[label]

                mean_val = d["mean_best_model_loss"]

                if "min_best_model_loss" in d and "max_best_model_loss" in d:
                    min_val = d["min_best_model_loss"]
                    max_val = d["max_best_model_loss"]
                else:
                    min_val = mean_val - d["std_best_model_loss"]
                    max_val = mean_val + d["std_best_model_loss"]

                lower_err = mean_val - min_val
                upper_err = max_val - mean_val
                yerr = np.array([[lower_err], [upper_err]])

                hatch = hatch_patterns[j % len(hatch_patterns)]
                x_pos = i + group_offsets[j]

                ax1.bar(
                    x_pos,
                    mean_val,
                    yerr=yerr,
                    width=bar_width,
                    capsize=5,
                    facecolor=label_color,
                    alpha=0.7,
                    edgecolor="black",
                    hatch=hatch,
                    linewidth=1.5,
                    error_kw=dict(
                        ecolor="k",
                        linewidth=3,
                    ),
                )

        custom_handles = [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=hatch_patterns[i % len(hatch_patterns)],
                linewidth=1.5,
                label=label_list[i],
            )
            for i in range(num_groups)
        ]

        ax1.legend(
            custom_handles,
            label_list,
            loc="upper left",
            bbox_to_anchor=bbox_to_anchor,
            fontsize=legend_size,
            ncols=legend_ncols,
            title_fontsize=legend_size,
            title=legend_title,
        )

        ax1.set_ylabel(y_label, fontsize=axis_fontsizes["yaxis"])

        ax1.set_yscale("log")
        if ylim:
            ax1.set_ylim(ylim)
        ax1.grid(bool(grid), axis="y", which="both", linestyle="-", linewidth=0.5, alpha=grid_alpha)

        ax1.tick_params(axis="y", which="major", length=10)
        ax1.tick_params(axis="y", which="minor", length=5)
        ax1.tick_params(axis="y", labelsize=axis_fontsizes["ytick_labels"])

        ax1.set_xticks(np.arange(num_labels))
        ax1.set_xticklabels(
            all_labels,
            rotation=0,
            ha="center",
            fontsize=axis_fontsizes["xtick_labels"],
        )

        if show_xlabel:
            ax1.set_xlabel(x_label, fontsize=axis_fontsizes["xaxis"])
        else:
            ax1.set_xlabel("")
        ax1.tick_params(
            axis="x",
            labelsize=axis_fontsizes["xtick_labels"],
            labelbottom=show_xtick_labels,
        )
        if not show_xtick_labels:
            ax1.set_xticklabels([])

        fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.98)

        if name:
            plt.savefig(name, dpi=100, facecolor="None")
            print("saved plot:", name)

        plt.show()

    finally:
        mpl.rcParams["hatch.linewidth"] = old_hatch_lw
        mpl.rcParams["hatch.color"] = old_hatch_color
