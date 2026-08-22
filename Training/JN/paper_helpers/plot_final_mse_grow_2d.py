"""Two-dimensional growth MSE summary plotting helpers.

Contents
--------
- plot_grow_mse_lst"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings

from .prepare_grow_mse_2d import prepare_grow_run_data


def plot_grow_mse_lst(
    models_dics_list,
    label_list=None,
    plot_params=None,
    name=None,
    bbox_to_anchor=(0.5, 0.5),
    legend_size=16,
    legend_ncols=1,
    y_label=r"Growth MSE [mm$^4$ days$^{-2}$]",
    hatch_patterns=("", "////", "\\\\\\\\", "xxxx", "++++", "....", "ooo", "***"),
    figsize=(7, 5),
    colors=None,
    hatch_linewidth=1,
    legend_title="ES",
    legend_bool=True,
    x_label=r"$N_u$",
    x_label_fontsize=14,
    axis_fontsizes=None,
    # --- PE ARGS ---
    pe_values=None,          # tuple/list/array of y-values for horizontal lines
    pe_label_list=None,      # labels for PE legend
    pe_legend_title="MPE",   # title for PE legend
    pe_legend_ncols=1,       #   number of columns in PE legend
    pe_legend_bool=True,
    pe_bbox_to_anchor_offset=(
        0.0,
        -0.15,
    ),  # offset relative to bbox_to_anchor of ES legend
    pe_linestyles=("--", "-.", ":"),  # linestyles for PE lines
    restrict_to_central_90=False,
    dataobj=None,
):
    """
    Growth MSE bar plot with:
      - consistent fill colors per x-label (same label → same color)
      - black hatch patterns per group (different texture per group)
      - log y-axis
      - spacing between bars of same label

    Additional:
      - horizontal red lines at `pe_values`
      - separate PE legend with labels from `pe_label_list`

    Error bars span min–max across repeats (using min_mse / max_mse if available,
    otherwise falling back to avg_mse ± std_mse).
    """

    import matplotlib as mpl

    if axis_fontsizes is None:
        axis_fontsizes = {
            "xaxis": x_label_fontsize,
            "xtick_labels": 14,
            "yaxis": 14,
            "ytick_labels": 14,
        }

    # ---------- Green-ish palette for *labels* ----------
    if colors is None:
        colors = [
            "#00441B",  # Deep Forest Green – dark, bold
            "#1A9850",  # Vivid Green – saturated, classic "plot green"
            "#66C2A4",  # Soft Mint Green – pleasant midtone
            "#B2E2E2",  # Light Aqua – soft, readable
            "#E5F5F9",  # Very Pale Greenish-White – background-compatible
        ][
            ::-1
        ]  # Reverse for better contrast from bottom to top

    # Label list is group names, not x-axis labels
    if label_list is None:
        label_list = [f"Group {i+1}" for i in range(len(models_dics_list))]

    # ---------- Gather Data ----------
    grouped_data = []
    all_labels = []  # ordered list, no duplicates

    for models_dics in models_dics_list:
        mse_data = prepare_grow_run_data(
            models_dics,
            plot_params=plot_params,
            restrict_to_central_90=restrict_to_central_90,
            dataobj=dataobj,
        )
        label_to_data = {d["label"]: d for d in mse_data}
        grouped_data.append(label_to_data)

        # preserve first-appearance order
        for lbl in label_to_data.keys():
            if lbl not in all_labels:
                all_labels.append(lbl)

    print("All x-axis labels found (ordered):", all_labels)

    num_labels = len(all_labels)
    num_groups = len(models_dics_list)

    # ---------- Assign consistent colors per label ----------
    label_color_map = {
        label: colors[i % len(colors)] for i, label in enumerate(all_labels)
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
    mpl.rcParams["hatch.color"] = "gray"

    try:
        # ---------- Plot bars ----------
        for i, label in enumerate(all_labels):
            label_color = label_color_map[label]

            for j, group_dict in enumerate(grouped_data):
                if label not in group_dict:
                    continue

                d = group_dict[label]
                hatch = hatch_patterns[j % len(hatch_patterns)]
                x_pos = i + group_offsets[j]

                center = d["avg_mse"]

                if "min_mse" in d and "max_mse" in d:
                    min_val = d["min_mse"]
                    max_val = d["max_mse"]
                else:
                    min_val = center - d["std_mse"]
                    max_val = center + d["std_mse"]

                lower_err = center - min_val
                upper_err = max_val - center
                yerr = np.array([[lower_err], [upper_err]])

                ax1.bar(
                    x_pos,
                    center,
                    yerr=yerr,
                    width=bar_width,
                    capsize=5,
                    facecolor=label_color,  # consistent per label
                    alpha=0.7,
                    edgecolor="black",
                    hatch=hatch,  # distinguishes group
                    linewidth=1.5,
                    error_kw=dict(
                        ecolor="k",
                        linewidth=3,
                    ),
                )

        # ---------- ES Legend (groups) ----------
        es_handles = [
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
            es_legend = ax1.legend(
                es_handles,
                label_list,
                bbox_to_anchor=bbox_to_anchor,
                fontsize=legend_size,
                ncols=legend_ncols,
                title_fontsize=legend_size,
                title=legend_title,
            )
            ax1.add_artist(es_legend)  # keep this legend when adding the second one

        # ---------- Horizontal PE lines + PE Legend ----------
        pe_handles = []
        if pe_values is not None:
            # Ensure iterable
            if not isinstance(pe_values, (list, tuple, np.ndarray)):
                pe_values = [pe_values]

            # Default labels if none provided
            if pe_label_list is None:
                pe_label_list = [f"PE {i+1}" for i in range(len(pe_values))]

            for y_val, lbl, pe_ls in zip(pe_values, pe_label_list, pe_linestyles):
                line = ax1.axhline(
                    y=y_val,
                    color="red",
                    linestyle=pe_ls,
                    linewidth=1.5,
                    label=lbl,
                )
                pe_handles.append(line)

            # Second legend, positioned relative to the first
            pe_bbox = (
                bbox_to_anchor[0] + pe_bbox_to_anchor_offset[0],
                bbox_to_anchor[1] + pe_bbox_to_anchor_offset[1],
            )
            if pe_legend_bool:
                ax1.legend(
                    pe_handles,
                    pe_label_list,
                    bbox_to_anchor=pe_bbox,
                    fontsize=legend_size,
                    title_fontsize=legend_size,
                    title=pe_legend_title,
                    ncols=pe_legend_ncols,
                )

        # ---------- Axes ----------
        ax1.set_ylabel(y_label, fontsize=axis_fontsizes["yaxis"])
        ax1.set_yscale("log")

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
    if restrict_to_central_90 and dataobj is None:
        warnings.warn(
            "restrict_to_central_90=True but no dataobj was passed to plot_grow_mse_lst. "
            "The central 90% support should be derived from dataobj; falling back to the model u-grid.",
            RuntimeWarning,
            stacklevel=2,
        )
