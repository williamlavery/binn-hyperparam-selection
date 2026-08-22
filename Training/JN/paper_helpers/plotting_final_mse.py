"""One-dimensional diffusion MSE summary plotting helpers.

Contents
--------
- plot_diff_mse_lst"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import sys
import warnings

from .prepare_diff_mse import prepare_diff_run_data

def plot_diff_mse_lst(
    models_dics_list,
    label_list=None,
    plot_params=None,
    name=None,
    bbox_to_anchor=(0.5, 0.5),
    legend_size=16,
    legend_ncols=1,
    legend_loc="upper left",
    legend_on_axes=False,
    legend_framealpha=0.9,
    y_label=r"Diffusion MSE [mm$^4$ days$^{-2}$]",
    hatch_patterns=(
        "", "////", "\\\\\\\\", "xxxx", "++++", "....", "ooo", "***"
    ),
    figsize=(7, 5),
    colors=None,
    hatch_linewidth=1,
    legend_title="ES",
    x_label=r"$N_u$",
    axis_fontsizes=None,
    group_width=0.8,
    bar_width_fraction=0.8,
    # --- PE ARGS ---
    pe_values=None,              # tuple/list/array of y-values for horizontal lines
    pe_label_list=None,          # labels for PE legend
    pe_legend_title="MPE (%)",       # title for PE legend
    pe_legend_ncols=1,          # number of columns in PE legend
    pe_legend_loc=None,
    pe_legend_on_axes=False,
    pe_legend_framealpha=0.9,
    pe_bbox_to_anchor_offset=(0.0, -0.15),  # offset relative to bbox_to_anchor of ES legend
    pe_linestyles=("--", "-.", ":"),        # linestyles for PE lines
    ylim =None,
    restrict_to_central_90=False,
    dataobj=None,
):
    """
    Diffusion MSE bar plot with:
      - consistent fill colors per x-label (same label → same color)
      - black hatch patterns per group (different texture per group)
      - log y-axis
      - adjustable spacing between bars of same label via `group_width`
        and `bar_width_fraction`

    Additional:
      - horizontal red lines at `pe_values`
      - separate PE legend with labels from `pe_label_list`

    Error bars span min–max across repeats (using min_mse / max_mse if available,
    otherwise falling back to avg_mse ± std_mse).
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    from matplotlib.patches import Patch

    if axis_fontsizes is None:
        axis_fontsizes = {
            "xaxis": 14,
            "xtick_labels": 14,
            "yaxis": 14,
            "ytick_labels": 14,
        }

    # ---------- Orange-ish palette for *labels* ----------
    if colors is None:
        colors = [
            "#8B2500",  # Burnt Orange – deep and bold
            "#FF7F0E",  # Vivid Orange – saturated, readable
            "#FDB863",  # Soft Tangerine – warm midtone
            "#FDD9B5",  # Peach – light orange-beige
            "#FFF1E0",  # Cream – very light accent
        ][::-1]

    # Label list is group names, not x-axis labels
    if label_list is None:
        label_list = [f"Group {i+1}" for i in range(len(models_dics_list))]

    # ---------- Gather Data ----------
    grouped_data = []
    all_labels = []   # ordered, no duplicates

    for models_dics in models_dics_list:
        mse_data = prepare_diff_run_data(
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

    if not (0 < group_width <= 1):
        raise ValueError("group_width must satisfy 0 < group_width <= 1.")
    if not (0 < bar_width_fraction <= 1):
        raise ValueError("bar_width_fraction must satisfy 0 < bar_width_fraction <= 1.")

    # ---------- Assign consistent colors per label ----------
    label_color_map = {
        label: colors[i % len(colors)]
        for i, label in enumerate(all_labels)
    }

    # ---------- Bar Geometry ----------
    slot_width = group_width / num_groups
    bar_width = slot_width * bar_width_fraction
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
                    facecolor=label_color,   # consistent per label
                    alpha=0.7,
                    edgecolor="black",
                    hatch=hatch,             # distinguishes group
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

        es_legend_kwargs = {
            "loc": legend_loc,
            "fontsize": legend_size,
            "ncols": legend_ncols,
            "title_fontsize": legend_size,
            "title": legend_title,
            "framealpha": legend_framealpha,
        }
        if not legend_on_axes and bbox_to_anchor is not None:
            es_legend_kwargs["bbox_to_anchor"] = bbox_to_anchor

        es_legend = ax1.legend(
            es_handles,
            label_list,
            **es_legend_kwargs,
        )
        es_legend.set_zorder(1000)
        es_legend.get_frame().set_alpha(legend_framealpha)
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
                    zorder=1,
                )
                pe_handles.append(line)

            # Second legend, positioned relative to the first
            pe_legend_kwargs = {
                "fontsize": legend_size,
                "title_fontsize": legend_size,
                "title": pe_legend_title,
                "ncols": pe_legend_ncols,
                "framealpha": pe_legend_framealpha,
            }
            if pe_legend_on_axes:
                pe_legend_kwargs["loc"] = pe_legend_loc or "best"
            elif pe_legend_loc is not None:
                pe_legend_kwargs["loc"] = pe_legend_loc
            else:
                pe_bbox = (
                    bbox_to_anchor[0] + pe_bbox_to_anchor_offset[0],
                    bbox_to_anchor[1] + pe_bbox_to_anchor_offset[1],
                )
                pe_legend_kwargs["loc"] = legend_loc
                pe_legend_kwargs["bbox_to_anchor"] = pe_bbox

            pe_legend = ax1.legend(
                pe_handles,
                pe_label_list,
                **pe_legend_kwargs,
            )
            pe_legend.set_zorder(1000)
            pe_legend.get_frame().set_alpha(pe_legend_framealpha)
            pe_legend.get_frame().set_facecolor("white")

        # ---------- Axes ----------
        ax1.set_ylabel(y_label, fontsize=axis_fontsizes["yaxis"])
        ax1.set_yscale("log")

        if ylim:
            ax1.set_ylim(ylim)

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
            "restrict_to_central_90=True but no dataobj was passed to plot_diff_mse_lst. "
            "The central 90% support should be derived from dataobj; falling back to the model u-grid.",
            RuntimeWarning,
            stacklevel=2,
        )
