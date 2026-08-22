"""One-dimensional growth MSE summary plotting helpers.

Contents
--------
- plot_grow_mse_lst_extended"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import sys
import warnings

from .prepare_grow_mse import prepare_grow_run_data



def plot_grow_mse_lst_extended(
    models_dics_list,
    label_list=None,
    plot_params=None,
    name=None,
    bbox_to_anchor=(0.5, 0.5),
    legend_size=16,
    y_label=r"Growth MSE [mm$^4$ days$^{-2}$]",
    hatch_patterns=("", "////", "\\\\\\\\", "xxxx", "++++", "....", "ooo", "***"),
    figsize=(7, 5),
    colors=None,
    hatch_linewidth=1,
    legend_title="ES",
    x_label=r"$N_u$",
    x_label_fontsize=14,
    axis_fontsizes=None,
    group_width=0.8,
    bar_width_fraction=0.8,
    # --- PE / MPE ARGS (backward compatible) ---
    pe_values=None,               # flat: list of y-values (global)
    pe_label_list=None,           # flat: list of labels (global)  <-- now used as annotations
    pe_legend_title="MPE (%)",
    pe_legend_ncols=1,
    pe_bbox_to_anchor_offset=(0.0, -0.15),
    pe_linestyles=("--", "-.", ":"),  # kept for backward compatibility (ignored now)
    # --- NEW: nested per x-tick ---
    pe_values_nested=None,        # list (len=num_labels) of lists of y-values
    pe_label_list_nested=None,    # list (len=num_labels) of lists of labels  <-- used as annotations
    pe_linestyles_nested=None,    # kept for backward compatibility (ignored now)
    y_lim=None,
    ylim=None,
    restrict_to_central_90=False,
    dataobj=None,
):
    """
    Growth MSE bar plot with:
      - consistent fill colors per x-label (same label → same color)
      - hatch patterns per group (texture per group)
      - log y-axis
      - adjustable spacing via `group_width` and `bar_width_fraction`

    Additional:
      - MPE lines can be provided globally (pe_values) OR per x-label via nested lists:
          pe_values_nested[i] / pe_label_list_nested[i]
        where i corresponds to the i-th x-tick label.

      - When nested is used, red horizontal line *segments* are drawn only over that x-tick.

    UPDATED:
      - All MPE lines are SOLID (no different linestyles).
      - Each MPE line is annotated with its provided label (e.g. 0.1, 0.2, 4, 5, ...).
        If a label is None, it falls back to numbering (1,2,3,...).
      - MPE legend is a single solid red line with title "MPE" (no entries),
        positioned using existing pe_bbox_to_anchor_offset etc.
      - Optional y-axis limits can be set with `y_lim=(ymin, ymax)` or `ylim=(ymin, ymax)`.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

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
            "#00441B",  # Deep Forest Green
            "#1A9850",  # Vivid Green
            "#66C2A4",  # Soft Mint Green
            "#B2E2E2",  # Light Aqua
            "#E5F5F9",  # Very Pale
        ][::-1]

    if label_list is None:
        label_list = [f"Group {i+1}" for i in range(len(models_dics_list))]

    # ---------- Gather Data ----------
    grouped_data = []
    all_labels = []

    for models_dics in models_dics_list:
        mse_data = prepare_grow_run_data(
            models_dics,
            plot_params=plot_params,
            restrict_to_central_90=restrict_to_central_90,
            dataobj=dataobj,
        )
        label_to_data = {d["label"]: d for d in mse_data}
        grouped_data.append(label_to_data)

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
    label_color_map = {label: colors[i % len(colors)] for i, label in enumerate(all_labels)}

    # ---------- Bar Geometry ----------
    slot_width = group_width / num_groups
    bar_width = slot_width * bar_width_fraction
    group_offsets = np.arange(num_groups) * slot_width - group_width / 2 + slot_width / 2

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
                    facecolor=label_color,
                    alpha=0.7,
                    edgecolor="black",
                    hatch=hatch,
                    linewidth=1.5,
                    error_kw=dict(ecolor="k", linewidth=3),
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

        es_legend = ax1.legend(
            es_handles,
            label_list,
            bbox_to_anchor=bbox_to_anchor,
            fontsize=legend_size,
            title_fontsize=legend_size,
            title=legend_title,
        )
        ax1.add_artist(es_legend)

        # ---------- MPE lines (flat OR nested) + inline annotations ----------
        # All MPE lines SOLID.

        def _as_list(x):
            if x is None:
                return None
            return x if isinstance(x, (list, tuple, np.ndarray)) else [x]

        # Used only as fallback when lbl is None
        mpe_key_to_num = {}

        def _get_mpe_key(lbl, y_val):
            return lbl if lbl is not None else float(y_val)

        def _fallback_num(lbl, y_val):
            key = _get_mpe_key(lbl, y_val)
            if key not in mpe_key_to_num:
                mpe_key_to_num[key] = len(mpe_key_to_num) + 1
            return mpe_key_to_num[key]

        def _fmt_lbl(lbl):
            if isinstance(lbl, (int, np.integer)):
                return str(int(lbl))
            if isinstance(lbl, (float, np.floating)):
                return f"{float(lbl):g}"  # compact float
            return str(lbl)

        ann_dx = 0.02
        ann_kwargs = dict(
            color="red",
            fontsize=max(8, int(0.75 * legend_size)),
            va="center",
            ha="left",
            clip_on=True,
        )

        drew_any_mpe = False

        if pe_values_nested is not None:
            if len(pe_values_nested) != num_labels:
                raise ValueError(
                    f"pe_values_nested must have length {num_labels} (one list per x-label). "
                    f"Got {len(pe_values_nested)}."
                )

            if pe_label_list_nested is None:
                pe_label_list_nested = [
                    [f"PE {k+1}" for k in range(len(_as_list(pe_values_nested[i]) or []))]
                    for i in range(num_labels)
                ]

            for i in range(num_labels):
                vals_i = _as_list(pe_values_nested[i]) or []
                labs_i = _as_list(pe_label_list_nested[i]) or []

                if len(vals_i) != len(labs_i):
                    raise ValueError(
                        f"For x-index {i} ('{all_labels[i]}'): pe_values_nested and "
                        f"pe_label_list_nested must have same inner length. "
                        f"Got {len(vals_i)} and {len(labs_i)}."
                    )

                xmin = i - group_width / 2
                xmax = i + group_width / 2

                for y_val, lbl in zip(vals_i, labs_i):
                    ax1.hlines(
                        y=y_val,
                        xmin=xmin,
                        xmax=xmax,
                        colors="red",
                        linestyles="-",
                        linewidth=1.5,
                    )
                    drew_any_mpe = True

                    txt = _fmt_lbl(lbl) if lbl is not None else str(_fallback_num(lbl, y_val))
                    ax1.text(xmax + ann_dx, y_val, txt, **ann_kwargs)

        elif pe_values is not None:
            vals = _as_list(pe_values) or []
            labs = _as_list(pe_label_list) or [f"PE {i+1}" for i in range(len(vals))]

            for y_val, lbl in zip(vals, labs):
                ax1.axhline(y=y_val, color="red", linestyle="-", linewidth=1.5)
                drew_any_mpe = True

                txt = _fmt_lbl(lbl) if lbl is not None else str(_fallback_num(lbl, y_val))
                ax1.text(1.005, y_val, txt, transform=ax1.get_yaxis_transform(), **ann_kwargs)

        # ---------- MPE Legend: single solid line, title only ----------
        if drew_any_mpe:
            pe_bbox = (
                bbox_to_anchor[0] + pe_bbox_to_anchor_offset[0],
                bbox_to_anchor[1] + pe_bbox_to_anchor_offset[1],
            )

            # One handle; blank label so only the title shows up.
            mpe_handle = Line2D([0], [0], color="red", linestyle="-", linewidth=1.5)
            ax1.legend(
                [mpe_handle],
                [""],
                bbox_to_anchor=pe_bbox,
                fontsize=legend_size,
                title_fontsize=legend_size,
                title=pe_legend_title,
                ncols=pe_legend_ncols,
                handlelength=2.5,
                handletextpad=0.0,
                borderaxespad=0.0,
                frameon=True,
            )

        # ---------- Axes ----------
        ax1.set_ylabel(y_label, fontsize=axis_fontsizes["yaxis"])
        ax1.set_yscale("log")
        applied_ylim = y_lim if y_lim is not None else ylim
        if applied_ylim is not None:
            ax1.set_ylim(applied_ylim)

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

    if restrict_to_central_90 and dataobj is None:
        warnings.warn(
            "restrict_to_central_90=True but no dataobj was passed to plot_grow_mse_lst_extended. "
            "The central 90% support should be derived from dataobj; falling back to the model u-grid.",
            RuntimeWarning,
            stacklevel=2,
        )
