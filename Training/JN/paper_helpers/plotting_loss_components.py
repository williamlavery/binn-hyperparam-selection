"""Loss-component plotting helpers for grouped training runs.

Contents
--------
- _component_from_name
- _resolve_components
- _component_weight
- _merge_plot_settings
- _is_model_wrapper
- _run_groups_from_models_dics
- _normalise_models_dics_list
- _run_groups_from_models_dics_for_seed
- _normalise_models_dics_list_for_seed
- _train_attr_for_val_attr
- _collect_component_losses
- _pad_loss_arrays
- _apply_running_min
- _apply_smoothing
- _component_stats
- _component_member_series
- _interpolate_hex_color
- _component_group_color
- _component_member_color
- _best_epoch_idx
- _best_epoch_idx_for_model
- _positive_for_log
- _first_pde_data_flip_epoch
- _legend_with_group_label
- _group_plot_name
- _saved_loss_at_best_epoch
- _print_best_epoch_component_losses
- _create_broken_x_axes
- _format_axes
- _add_broken_axis_diagonals
- _apply_grid
- _add_line_legend
- _marker_legend_entries_for_groups
- _add_marker_legend
- _color_index_for_group
- _num_color_groups
- _marker_for_group
- _line_style_for_group
- _line_style_for_es_group
- _add_best_marker
- _add_termination_marker
- _plot_mean_loss_components_broken_x_log_lst
- plot_running_min_loss_components_broken_x_log_lst
- plot_running_min_loss_components_seed_broken_x_log_lst
- plot_smoothed_loss_components_seed_broken_x_log_lst"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


DEFAULT_COMPONENTS = {
    "constraint": {
        "attr": "val_constraint_loss_list",
        "label": "Constraint",
        "color": "#6B7280",
    },
    "pde": {
        "attr": "val_pde_loss_list",
        "label": "PDE",
        "color": "#7E22CE",
    },
    "data": {
        "attr": "val_data_loss_list",
        "label": "Data",
        "color": "#DB2777",
    },
}


COMPONENT_ALIASES = {
    "constraint": {
        "attr": "val_constraint_loss_list",
        "label": "Constraint",
        "color": "#6B7280",
    },
    "pde": {
        "attr": "val_pde_loss_list",
        "label": "PDE",
        "color": "#7E22CE",
        "weight_attr": "pde_weight",
    },
    "data": {
        "attr": "val_data_loss_list",
        "label": "Data",
        "color": "#DB2777",
        "weight_attr": "surface_weight",
    },
    "D_bound": {
        "attr": "val_D_bound_loss_list",
        "label": "D bound",
        "color": "C4",
    },
    "D_mono": {
        "attr": "val_D_mono_loss_list",
        "label": "D mono",
        "color": "C1",
    },
    "G_bound": {
        "attr": "val_G_bound_loss_list",
        "label": "G bound",
        "color": "C5",
    },
    "G_mono": {
        "attr": "val_G_mono_loss_list",
        "label": "G mono",
        "color": "C6",
    },
}


COMPONENT_PALETTES = {
    "constraint": ("#E5E7EB", "#6B7280", "#111827"),
    "pde": ("#E9D5FF", "#A855F7", "#581C87"),
    "data": ("#FBCFE8", "#EC4899", "#831843"),
    "D_bound": ("#FCE7F3", "#F43F5E", "#881337"),
    "D_mono": ("#EDE9FE", "#8B5CF6", "#4C1D95"),
    "G_bound": ("#E0F2FE", "#0284C7", "#0C4A6E"),
    "G_mono": ("#DCFCE7", "#16A34A", "#14532D"),
}


DEFAULT_SETTINGS = {
    "xaxis": {"min": 1, "max": 1e5, "break": 1e3},
    "legend": {
        "panel": 1,
        "loc": (0.05, 0.95),
        "loc_upd": (0.35, 0.95),
        "fontsize": 10,
        "ncols": 1,
        "title": "Loss component",
        "marker_title": "ES",
        "framealpha": 0.6,
        "facecolor": "white",
        "edgecolor": "white",
    },
    "name": "mean_loss_components_broken_xaxis_loglog.png",
    "fill": True,
    "line_width": 1.5,
    "fill_alpha": 0.2,
    "fontsizes": {
        "xaxis": 12,
        "xtick_labels": 10,
        "yaxis": 12,
        "ytick_labels": 10,
    },
    "figsize": (7, 5),
    "ylabel": "Loss component [a.u]",
    "y_floor": 1e-16,
    "y_lim": None,
    "running_min": False,
    "best_epoch_termination": True,
    "group_labels": None,
    "color_indices": None,
    "es_entries": [],
    "marker_indices": None,
    "line_styles": ["-", "--", "-.", ":"],
    "line_style_indices": None,
    "best_model_markers": True,
    "best_marker_size": 40,
    "member_labels": None,
    "grid": True,
}


def _component_from_name(name):
    if name in COMPONENT_ALIASES:
        return dict(COMPONENT_ALIASES[name])

    attr = name if name.endswith("_loss_list") else f"val_{name}_loss_list"
    label = (
        name.removeprefix("val_")
        .removeprefix("train_")
        .removesuffix("_loss_list")
        .replace("_", " ")
        .title()
    )
    return {
        "attr": attr,
        "label": label,
        "color": None,
    }


def _resolve_components(components):
    if components is None:
        return {name: dict(component) for name, component in DEFAULT_COMPONENTS.items()}

    if isinstance(components, dict):
        resolved = {}
        for i, (name, component) in enumerate(components.items()):
            if isinstance(component, str):
                component = _component_from_name(component)
            else:
                component = dict(component)
            component.setdefault("attr", _component_from_name(name)["attr"])
            component.setdefault("label", _component_from_name(name)["label"])
            component.setdefault("color", None)
            component.setdefault("weight_attr", _component_from_name(name).get("weight_attr"))
            resolved[name] = component
    else:
        resolved = {
            name: _component_from_name(name)
            for name in components
        }

    for i, component in enumerate(resolved.values()):
        if component["color"] is None:
            component["color"] = f"C{i}"
    return resolved


def _component_weight(model_wrapper, component, normalize_by_loss_weights):
    if not normalize_by_loss_weights:
        return 1.0

    weight_attr = component.get("weight_attr")
    if not weight_attr:
        return 1.0

    model = getattr(model_wrapper, "model", None)
    if model is None:
        return 1.0

    weight = getattr(model, weight_attr, 1.0)
    if weight is None:
        return 1.0

    try:
        weight = float(weight)
    except (TypeError, ValueError):
        return 1.0

    return 1.0 if weight == 0 else weight


def _merge_plot_settings(plot_settings):
    plot_settings = plot_settings or {}
    settings = {**DEFAULT_SETTINGS, **plot_settings}
    return {
        "xaxis": {**DEFAULT_SETTINGS["xaxis"], **settings.get("xaxis", {})},
        "legend": {**DEFAULT_SETTINGS["legend"], **settings.get("legend", {})},
        "fontsizes": {
            **DEFAULT_SETTINGS["fontsizes"],
            **settings.get("fontsizes", {}),
        },
        "name": settings.get("name", DEFAULT_SETTINGS["name"]),
        "fill": settings.get("fill", DEFAULT_SETTINGS["fill"]),
        "line_width": settings.get("line_width", DEFAULT_SETTINGS["line_width"]),
        "fill_alpha": settings.get("fill_alpha", DEFAULT_SETTINGS["fill_alpha"]),
        "figsize": settings.get("figsize", DEFAULT_SETTINGS["figsize"]),
        "ylabel": settings.get("ylabel", DEFAULT_SETTINGS["ylabel"]),
        "y_floor": settings.get("y_floor", DEFAULT_SETTINGS["y_floor"]),
        "y_lim": settings.get("y_lim", DEFAULT_SETTINGS["y_lim"]),
        "running_min": settings.get("running_min", DEFAULT_SETTINGS["running_min"]),
        "best_epoch_termination": settings.get(
            "best_epoch_termination",
            DEFAULT_SETTINGS["best_epoch_termination"],
        ),
        "group_labels": settings.get("group_labels", DEFAULT_SETTINGS["group_labels"]),
        "color_indices": settings.get("color_indices", DEFAULT_SETTINGS["color_indices"]),
        "es_entries": settings.get("es_entries", DEFAULT_SETTINGS["es_entries"]),
        "marker_indices": settings.get("marker_indices", DEFAULT_SETTINGS["marker_indices"]),
        "line_styles": settings.get("line_styles", DEFAULT_SETTINGS["line_styles"]),
        "line_style_indices": settings.get(
            "line_style_indices",
            DEFAULT_SETTINGS["line_style_indices"],
        ),
        "best_model_markers": settings.get(
            "best_model_markers",
            DEFAULT_SETTINGS["best_model_markers"],
        ),
        "best_marker_size": settings.get(
            "best_marker_size",
            DEFAULT_SETTINGS["best_marker_size"],
        ),
        "member_labels": settings.get("member_labels", DEFAULT_SETTINGS["member_labels"]),
        "grid": settings.get("grid", DEFAULT_SETTINGS["grid"]),
    }


def _is_model_wrapper(value):
    return hasattr(value, "val_loss_list")


def _run_groups_from_models_dics(models_dics, group_label):
    if not models_dics:
        raise ValueError("models_dics is empty.")

    values = list(models_dics.values())
    if all(_is_model_wrapper(value) for value in values):
        return [(group_label, list(values))]

    if not all(isinstance(value, dict) for value in values):
        raise ValueError(
            "Each group must be either {seed: modelWrapper, ...} or "
            "{config_key: {seed: modelWrapper, ...}}."
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
    if isinstance(models_dics_list, dict):
        models_dics_list = [models_dics_list]

    if not isinstance(models_dics_list, list) or not models_dics_list:
        raise ValueError("models_dics_list must be a non-empty dict or list of dicts.")

    if group_labels is None:
        group_labels = [f"Group {i + 1}" for i in range(len(models_dics_list))]
    elif len(group_labels) != len(models_dics_list):
        raise ValueError("plot_settings['group_labels'] must match models_dics_list length.")

    run_groups = []
    for models_dics, group_label in zip(models_dics_list, group_labels):
        run_groups.extend(_run_groups_from_models_dics(models_dics, group_label))
    return run_groups


def _run_groups_from_models_dics_for_seed(models_dics, group_label, seed):
    if not models_dics:
        raise ValueError("models_dics is empty.")

    values = list(models_dics.values())
    if all(_is_model_wrapper(value) for value in values):
        if seed not in models_dics:
            raise KeyError(f"Seed {seed!r} not found in model group {group_label!r}.")
        return [(group_label, models_dics[seed])]

    if not all(isinstance(value, dict) for value in values):
        raise ValueError(
            "Each group must be either {seed: modelWrapper, ...} or "
            "{config_key: {seed: modelWrapper, ...}}."
        )

    run_groups = []
    for key, model_dic in models_dics.items():
        if seed not in model_dic:
            raise KeyError(f"Seed {seed!r} not found for group {key!r}.")
        model_wrapper = model_dic[seed]
        if not _is_model_wrapper(model_wrapper):
            raise ValueError(
                "Nested input must be {config_key: {seed: modelWrapper, ...}}."
            )
        label = group_label if len(values) == 1 else str(key)
        run_groups.append((label, model_wrapper))
    return run_groups


def _normalise_models_dics_list_for_seed(models_dics_list, group_labels=None, seed=0):
    if isinstance(models_dics_list, dict):
        models_dics_list = [models_dics_list]

    if not isinstance(models_dics_list, list) or not models_dics_list:
        raise ValueError("models_dics_list must be a non-empty dict or list of dicts.")

    if group_labels is None:
        group_labels = [f"Group {i + 1}" for i in range(len(models_dics_list))]
    elif len(group_labels) != len(models_dics_list):
        raise ValueError("plot_settings['group_labels'] must match models_dics_list length.")

    run_groups = []
    for models_dics, group_label in zip(models_dics_list, group_labels):
        run_groups.extend(
            _run_groups_from_models_dics_for_seed(
                models_dics,
                group_label=group_label,
                seed=seed,
            )
        )
    return run_groups


def _train_attr_for_val_attr(loss_attr):
    return "train_" + loss_attr[4:] if loss_attr.startswith("val_") else None


def _collect_component_losses(model_wrappers, loss_attr):
    values = []
    fallback_attr = _train_attr_for_val_attr(loss_attr)

    for model in model_wrappers:
        raw = getattr(model, loss_attr, [])
        if len(raw):
            values.append(np.asarray(raw, dtype=np.float64))
            continue

        if fallback_attr is not None and len(getattr(model, fallback_attr, [])):
            values.append(np.asarray(getattr(model, fallback_attr), dtype=np.float64))
            continue

        raise AttributeError(
            f"Model wrapper has no values for {loss_attr!r}"
            + (f" or fallback {fallback_attr!r}." if fallback_attr else ".")
        )

    return values


def _pad_loss_arrays(loss_arrays):
    lengths = [len(values) for values in loss_arrays]
    max_len = max(lengths)
    padded = np.full((len(loss_arrays), max_len), np.nan, dtype=np.float64)

    for i, values in enumerate(loss_arrays):
        padded[i, : len(values)] = values

    return padded, lengths


def _apply_running_min(values, lengths):
    running = np.full_like(values, np.nan)
    for i, length in enumerate(lengths):
        if length:
            running[i, :length] = np.minimum.accumulate(values[i, :length])
    return running


def _apply_smoothing(values, lengths, window):
    if window is None:
        window = 1

    window = int(window)
    if window <= 1:
        return values.copy()

    smoothed = np.full_like(values, np.nan)
    for i, length in enumerate(lengths):
        if length:
            cumsum = np.cumsum(values[i, :length], dtype=np.float64)
            for end_idx in range(length):
                start_idx = max(0, end_idx - window + 1)
                window_sum = cumsum[end_idx]
                if start_idx > 0:
                    window_sum -= cumsum[start_idx - 1]
                smoothed[i, end_idx] = window_sum / (end_idx - start_idx + 1)
    return smoothed


def _component_stats(model_wrappers, loss_attr, running_min):
    loss_arrays = _collect_component_losses(model_wrappers, loss_attr)
    values, lengths = _pad_loss_arrays(loss_arrays)

    if running_min:
        values = _apply_running_min(values, lengths)

    return {
        "epochs": np.arange(values.shape[1]),
        "mean": np.nanmean(values, axis=0),
        "min": np.nanmin(values, axis=0),
        "max": np.nanmax(values, axis=0),
    }


def _component_member_series(model_wrappers, loss_attr, running_min):
    loss_arrays = _collect_component_losses(model_wrappers, loss_attr)
    values, lengths = _pad_loss_arrays(loss_arrays)

    if running_min:
        values = _apply_running_min(values, lengths)

    return values, lengths


def _interpolate_hex_color(hex_colors, t):
    hex_colors = [color.lstrip("#") for color in hex_colors]
    rgb = np.asarray(
        [
            [int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)]
            for color in hex_colors
        ],
        dtype=np.float64,
    )
    t = float(np.clip(t, 0, 1))
    if len(rgb) == 1:
        out = rgb[0]
    else:
        scaled = t * (len(rgb) - 1)
        left = int(np.floor(scaled))
        right = min(left + 1, len(rgb) - 1)
        frac = scaled - left
        out = (1 - frac) * rgb[left] + frac * rgb[right]
    return "#" + "".join(f"{int(round(channel * 255)):02X}" for channel in out)


def _component_group_color(name, component, group_idx, n_groups):
    palette = component.get("palette", COMPONENT_PALETTES.get(name))
    if palette is None:
        base = component.get("color", f"C{group_idx}")
        return base

    if n_groups <= 1:
        return component.get("color") or _interpolate_hex_color(palette, 0.5)

    return _interpolate_hex_color(palette, group_idx / (n_groups - 1))


def _component_member_color(name, component, member_idx, n_members):
    palette = component.get("palette", COMPONENT_PALETTES.get(name))
    if palette is None:
        return component.get("color", f"C{member_idx}")

    if n_members <= 1:
        return component.get("color") or _interpolate_hex_color(palette, 0.5)

    return _interpolate_hex_color(palette, member_idx / (n_members - 1))


def _best_epoch_idx(model_wrappers, max_len):
    best_losses = np.asarray(
        [getattr(model, "best_val_loss", np.nan) for model in model_wrappers],
        dtype=np.float64,
    )
    if np.all(np.isnan(best_losses)):
        return max_len - 1

    best_seed_idx = int(np.nanargmin(best_losses))
    best_model = model_wrappers[best_seed_idx]
    best_epoch = getattr(best_model, "last_improved", None)
    if best_epoch is None:
        return max_len - 1

    best_epoch = int(best_epoch)
    if best_epoch >= max_len and (best_epoch - 1) < max_len:
        best_epoch -= 1
    return max(0, min(best_epoch, max_len - 1))


def _best_epoch_idx_for_model(model_wrapper, max_len):
    if max_len <= 0:
        return 0

    best_epoch = getattr(model_wrapper, "last_improved", None)
    if best_epoch is None:
        best_epoch = getattr(getattr(model_wrapper, "model", None), "epochs", None)
        if best_epoch is None:
            return max_len - 1

    best_epoch = int(best_epoch)
    if best_epoch >= max_len and (best_epoch - 1) < max_len:
        best_epoch -= 1
    return max(0, min(best_epoch, max_len - 1))


def _positive_for_log(values, y_floor):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0, y_floor, values)


def _first_pde_data_flip_epoch(component_epochs):
    pde_epochs = component_epochs.get("pde")
    data_epochs = component_epochs.get("data")
    if pde_epochs is None or data_epochs is None:
        return None

    n_shared = min(len(pde_epochs["epochs"]), len(data_epochs["epochs"]))
    if n_shared <= 0:
        return None

    epochs = np.asarray(pde_epochs["epochs"][:n_shared], dtype=np.int64)
    pde_values = np.asarray(pde_epochs["values"][:n_shared], dtype=np.float64)
    data_values = np.asarray(data_epochs["values"][:n_shared], dtype=np.float64)
    valid = epochs >= 1
    if not np.any(valid):
        return None

    epochs = epochs[valid]
    delta = data_values[valid] - pde_values[valid]
    seen_data_above_pde = False
    for idx, delta_value in enumerate(delta):
        if delta_value > 0:
            seen_data_above_pde = True
            continue
        if seen_data_above_pde:
            return int(epochs[idx])
    return None


def _legend_with_group_label(legend, group_label=None):
    legend = dict(legend)
    title = legend.get("title")
    if group_label is not None:
        title = group_label if not title else f"{title}\n{group_label}"
    legend["title"] = title
    return legend


def _group_plot_name(base_name, group_label):
    if not base_name:
        return base_name

    path = Path(base_name)
    safe_group = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in str(group_label)
    ).strip("_")
    safe_group = safe_group or "group"
    return str(path.with_name(f"{path.stem}_{safe_group}{path.suffix}"))


def _saved_loss_at_best_epoch(model_wrapper, loss_attr):
    loss_arrays = _collect_component_losses([model_wrapper], loss_attr)
    values = np.asarray(loss_arrays[0], dtype=np.float64)
    if len(values) == 0:
        return None, None, None

    best_idx = _best_epoch_idx_for_model(model_wrapper, len(values))
    return 0, best_idx, float(values[0]), float(values[best_idx])


def _print_best_epoch_component_losses(group_label, model_wrapper, components):
    pde_component = components.get("pde")
    data_component = components.get("data")
    if pde_component is None or data_component is None:
        return

    pde_initial_epoch, pde_best_epoch, pde_initial_loss, pde_best_loss = _saved_loss_at_best_epoch(
        model_wrapper,
        pde_component["attr"],
    )
    data_initial_epoch, data_best_epoch, data_initial_loss, data_best_loss = _saved_loss_at_best_epoch(
        model_wrapper,
        data_component["attr"],
    )
    if pde_best_epoch is None or data_best_epoch is None:
        return

    print(
        f"{group_label}: "
        f"initial saved epoch={pde_initial_epoch}, "
        f"{pde_component['label']}={pde_initial_loss:.6e}, "
        f"{data_component['label']}={data_initial_loss:.6e}; "
        f"best saved epoch={pde_best_epoch}, "
        f"{pde_component['label']}={pde_best_loss:.6e}, "
        f"{data_component['label']}={data_best_loss:.6e}"
    )


def _create_broken_x_axes(figsize):
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 5], "wspace": 0.05},
    )
    return fig, ax1, ax2


def _format_axes(ax1, ax2, xaxis, fontsizes, ylabel, y_lim=None):
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_facecolor("white")

    ax2.set_xlabel("Epoch", fontsize=fontsizes["xaxis"])
    ax1.set_ylabel(ylabel, fontsize=fontsizes["yaxis"])
    ax1.set_xlim(left=xaxis["min"], right=xaxis["break"])
    ax2.set_xlim(left=xaxis["break"], right=xaxis["max"])

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    ax1.yaxis.set_tick_params(labelsize=fontsizes["ytick_labels"])
    ax1.xaxis.set_tick_params(labelsize=fontsizes["xtick_labels"])
    ax2.xaxis.set_tick_params(labelsize=fontsizes["xtick_labels"])
    if y_lim is not None:
        ax1.set_ylim(y_lim)
        ax2.set_ylim(y_lim)


def _add_broken_axis_diagonals(ax1, ax2):
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)


def _apply_grid(ax1, ax2, grid):
    for ax in (ax1, ax2):
        ax.grid(grid, which="both", ls="-", lw=0.5, alpha=0.7)


def _add_line_legend(ax1, ax2, line_entries, line_width, legend):
    unique_entries = []
    seen_labels = set()
    for entry in line_entries:
        if entry["label"] is None:
            continue
        if entry["label"] in seen_labels:
            continue
        seen_labels.add(entry["label"])
        unique_entries.append(entry)

    handles = [
        Line2D([0], [0], color=entry["color"], lw=line_width)
        for entry in unique_entries
    ]
    labels = [entry["label"] for entry in unique_entries]
    legend_kwargs = dict(
        handles=handles,
        labels=labels,
        title=legend["title"],
        fontsize=legend["fontsize"],
        title_fontsize=legend["fontsize"],
        ncols=legend["ncols"],
        framealpha=legend.get("framealpha", 0.6),
        facecolor=legend.get("facecolor", "white"),
        edgecolor=legend.get("edgecolor", "white"),
    )

    target_ax = ax1 if legend["panel"] == 1 else ax2
    legend_artist = target_ax.legend(bbox_to_anchor=legend["loc"], **legend_kwargs)
    target_ax.add_artist(legend_artist)


def _marker_legend_entries_for_groups(group_indices, cfg, flip_epochs_by_group=None):
    es_entries = cfg.get("es_entries") or []
    if not es_entries or not group_indices:
        return []

    marker_entries = []
    seen_marker_idx = set()
    line_styles = cfg.get("line_styles") or ["-"]

    for group_idx in group_indices:
        marker_indices = cfg.get("marker_indices")
        if marker_indices is not None and group_idx < len(marker_indices):
            marker_idx = int(marker_indices[group_idx])
        else:
            marker_idx = group_idx

        if marker_idx in seen_marker_idx or not (0 <= marker_idx < len(es_entries)):
            continue
        seen_marker_idx.add(marker_idx)

        label = es_entries[marker_idx][0]
        if flip_epochs_by_group is not None:
            flip_epoch = flip_epochs_by_group.get(group_idx)
            flip_text = "none" if flip_epoch is None else str(flip_epoch)
            label = f"{label} (flip {flip_text})"

        marker_entries.append(
            {
                "label": label,
                "marker": es_entries[marker_idx][1],
                "line_style": line_styles[marker_idx % len(line_styles)],
            }
        )

    return marker_entries


def _add_marker_legend(ax2, cfg, marker_entries=None):
    if marker_entries is None:
        group_count = 0
        marker_indices = cfg.get("marker_indices")
        if marker_indices is not None:
            group_count = len(marker_indices)
        else:
            group_labels = cfg.get("group_labels")
            if group_labels is not None:
                group_count = len(group_labels)
            else:
                group_count = len(cfg.get("es_entries") or [])
        marker_entries = _marker_legend_entries_for_groups(
            list(range(group_count)),
            cfg,
        )
    if not marker_entries:
        return

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=entry["marker"],
            linestyle=entry["line_style"],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=8,
        )
        for entry in marker_entries
    ]
    marker_labels = [entry["label"] for entry in marker_entries]
    marker_legend = ax2.legend(
        handles=marker_handles,
        labels=marker_labels,
        bbox_to_anchor=cfg["legend"].get("loc_upd", (1, 1)),
        fontsize=cfg["legend"]["fontsize"],
        title=cfg["legend"].get("marker_title", "ES"),
        title_fontsize=cfg["legend"]["fontsize"],
        framealpha=cfg["legend"].get("framealpha", 0.6),
        facecolor=cfg["legend"].get("facecolor", "white"),
        edgecolor=cfg["legend"].get("edgecolor", "white"),
    )
    ax2.add_artist(marker_legend)


def _color_index_for_group(group_idx, cfg):
    color_indices = cfg.get("color_indices")
    if color_indices is not None and group_idx < len(color_indices):
        return int(color_indices[group_idx])
    return group_idx


def _num_color_groups(cfg, run_groups):
    color_indices = cfg.get("color_indices")
    if color_indices:
        return max(int(index) for index in color_indices) + 1
    return len(run_groups)


def _marker_for_group(group_idx, cfg):
    es_entries = cfg.get("es_entries") or []
    if not es_entries:
        return "o"

    marker_indices = cfg.get("marker_indices")
    if marker_indices is not None and group_idx < len(marker_indices):
        marker_idx = int(marker_indices[group_idx])
    else:
        marker_idx = group_idx

    if 0 <= marker_idx < len(es_entries):
        return es_entries[marker_idx][1]
    return "o"


def _line_style_for_group(group_idx, cfg):
    line_styles = cfg.get("line_styles") or ["-"]
    line_style_indices = cfg.get("line_style_indices")
    if line_style_indices is not None and group_idx < len(line_style_indices):
        line_style_idx = int(line_style_indices[group_idx])
    else:
        line_style_idx = group_idx
    return line_styles[line_style_idx % len(line_styles)]


def _line_style_for_es_group(group_idx, cfg):
    line_styles = cfg.get("line_styles") or ["-"]
    marker_indices = cfg.get("marker_indices")
    if marker_indices is not None and group_idx < len(marker_indices):
        line_style_idx = int(marker_indices[group_idx])
    else:
        line_style_idx = group_idx
    return line_styles[line_style_idx % len(line_styles)]


def _add_best_marker(ax1, ax2, x, y, break_x, color, marker, cfg):
    if len(x) == 0:
        return
    target_ax = ax1 if x[-1] <= break_x else ax2
    target_ax.scatter(
        x[-1],
        y[-1],
        facecolors=color,
        edgecolors="black",
        linewidths=0.8,
        s=cfg["best_marker_size"],
        zorder=5,
        marker=marker,
    )


def _add_termination_marker(ax1, ax2, x, y, break_x, color, cfg, marker="o"):
    if len(x) == 0:
        return
    target_ax = ax1 if x[-1] <= break_x else ax2
    target_ax.scatter(
        x[-1],
        y[-1],
        facecolors=color,
        edgecolors="black",
        linewidths=0.8,
        s=cfg["best_marker_size"],
        zorder=5,
        marker=marker,
    )


def _plot_mean_loss_components_broken_x_log_lst(
    models_dics_list=None,
    plot_settings=None,
    components=None,
    models_dics=None,
    legend_labels=None,
    legend_label_title=None,
):
    """
    Plot constraint, PDE, and data loss components across experiment groups.

    The central curve is the mean across seeds. The shaded band spans the
    seed-wise min to max. Set plot_settings={"running_min": True} to plot the
    running minimum of each seed before aggregating.

    Parameters
    ----------
    models_dics_list : dict or list[dict]
        Same grouped structure used by the diffusion MSE helper. Each element
        may be {seed: modelWrapper, ...} or
        {config_key: {seed: modelWrapper, ...}}.
    plot_settings : dict, optional
        Supports the same broad settings style as the existing broken-axis
        loss plots, including xaxis, legend, name, fill, line_width, figsize,
        ylabel, y_floor, running_min, and best_epoch_termination.
    components : list[str] or dict, optional
        Components to display. A list can contain aliases such as
        "constraint", "pde", "data", "D_bound", "D_mono", "G_bound", or
        "G_mono". A dict can provide custom attr, label, and color entries.
    """
    plot_settings = dict(plot_settings or {})
    if legend_labels is not None:
        plot_settings["group_labels"] = list(legend_labels)
    if legend_label_title is not None:
        legend_settings = dict(plot_settings.get("legend", {}))
        legend_settings["title"] = legend_label_title
        plot_settings["legend"] = legend_settings

    cfg = _merge_plot_settings(plot_settings)
    components = _resolve_components(components)

    if models_dics_list is None:
        models_dics_list = models_dics
    elif models_dics is not None:
        raise ValueError("Pass only one of models_dics_list or models_dics.")

    run_groups = _normalise_models_dics_list(
        models_dics_list,
        group_labels=cfg["group_labels"],
    )
    if not run_groups:
        raise ValueError("No model groups were found.")

    fig, ax1, ax2 = _create_broken_x_axes(cfg["figsize"])
    break_x = cfg["xaxis"]["break"]
    line_entries = []
    n_color_groups = _num_color_groups(cfg, run_groups)

    for group_idx, (group_label, model_wrappers) in enumerate(run_groups):
        if not model_wrappers:
            continue

        component_data = {}
        max_len = 0
        for name, component in components.items():
            stats = _component_stats(
                model_wrappers,
                component["attr"],
                running_min=cfg["running_min"],
            )
            component_data[name] = stats
            max_len = max(max_len, len(stats["epochs"]))

        if cfg["best_epoch_termination"]:
            final_idx = _best_epoch_idx(model_wrappers, max_len)
        else:
            final_idx = max_len - 1

        for name, stats in component_data.items():
            component = components[name]
            color_idx = _color_index_for_group(group_idx, cfg)
            color = _component_group_color(name, component, color_idx, n_color_groups)
            line_style = _line_style_for_group(group_idx, cfg)
            label = (
                component["label"]
                if len(run_groups) == 1
                else f"{component['label']} | {group_label}"
            )
            line_entries.append({"label": label, "color": color})

            end_idx = min(final_idx + 1, len(stats["epochs"]))
            epochs = stats["epochs"][:end_idx]
            x = np.where(epochs == 0, 1e-1, epochs)
            y_mean = _positive_for_log(stats["mean"][:end_idx], cfg["y_floor"])
            y_min = _positive_for_log(stats["min"][:end_idx], cfg["y_floor"])
            y_max = _positive_for_log(stats["max"][:end_idx], cfg["y_floor"])

            mask1 = x <= break_x
            mask2 = x > break_x

            ax1.plot(
                x[mask1],
                y_mean[mask1],
                color=color,
                lw=cfg["line_width"],
                ls=line_style,
            )
            ax2.plot(
                x[mask2],
                y_mean[mask2],
                color=color,
                lw=cfg["line_width"],
                ls=line_style,
            )

            if cfg["fill"]:
                ax1.fill_between(
                    x[mask1],
                    y_min[mask1],
                    y_max[mask1],
                    color=color,
                    alpha=cfg["fill_alpha"],
                )
                ax2.fill_between(
                    x[mask2],
                    y_min[mask2],
                    y_max[mask2],
                    color=color,
                    alpha=cfg["fill_alpha"],
                )

            if cfg["best_model_markers"]:
                _add_best_marker(
                    ax1,
                    ax2,
                    x,
                    y_mean,
                    break_x,
                    color,
                    _marker_for_group(group_idx, cfg),
                    cfg,
                )

    _format_axes(ax1, ax2, cfg["xaxis"], cfg["fontsizes"], cfg["ylabel"], cfg["y_lim"])
    _add_broken_axis_diagonals(ax1, ax2)
    _apply_grid(ax1, ax2, cfg["grid"])
    _add_line_legend(
        ax1,
        ax2,
        line_entries,
        cfg["line_width"],
        cfg["legend"],
    )
    _add_marker_legend(ax2, cfg)

    fig.tight_layout()

    if cfg["name"]:
        plt.savefig(cfg["name"], dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", cfg["name"])

    plt.show()

    return {
        "name": cfg["name"],
        "xaxis": cfg["xaxis"],
        "components": {
            name: component["attr"] for name, component in components.items()
        },
        "running_min": cfg["running_min"],
        "best_epoch_termination": cfg["best_epoch_termination"],
        "grid": cfg["grid"],
        "num_groups": len(run_groups),
        "num_runs": sum(len(model_wrappers) for _, model_wrappers in run_groups),
    }


def plot_running_min_loss_components_broken_x_log_lst(
    models_dics_list=None,
    plot_settings=None,
    components=None,
    models_dics=None,
    legend_labels=None,
    legend_label_title=None,
):
    """
    Backward-compatible wrapper that plots running-min component curves.

    Prefer `plot_mean_loss_components_broken_x_log_lst` for the raw mean and
    min-max component trajectories.
    """
    plot_settings = {
        "name": "running_min_loss_components_broken_xaxis_loglog.png",
        "ylabel": "Running min loss component [a.u]",
        **(plot_settings or {}),
        "running_min": True,
    }
    return _plot_mean_loss_components_broken_x_log_lst(
        models_dics_list=models_dics_list,
        plot_settings=plot_settings,
        components=components,
        models_dics=models_dics,
        legend_labels=legend_labels,
        legend_label_title=legend_label_title,
    )


def plot_running_min_loss_components_seed_broken_x_log_lst(
    models_dics_list=None,
    plot_settings=None,
    components=None,
    models_dics=None,
    legend_labels=None,
    legend_label_title=None,
    seed=0,
):
    """
    Plot running-min loss components for one split from each model group.

    This mirrors `plot_running_min_loss_components_broken_x_log_lst`, but it
    does not aggregate across splits. Instead, it selects the same split key
    from every group, controlled by `seed` and defaulting to 0.
    """
    plot_settings = {
        "name": "running_min_loss_components_seed_broken_xaxis_loglog.png",
        "ylabel": "Running min loss component [a.u]",
        "fill": False,
        **(plot_settings or {}),
        "running_min": True,
    }

    if legend_labels is not None:
        plot_settings["group_labels"] = list(legend_labels)
    if legend_label_title is not None:
        legend_settings = dict(plot_settings.get("legend", {}))
        legend_settings["title"] = legend_label_title
        plot_settings["legend"] = legend_settings

    cfg = _merge_plot_settings(plot_settings)
    components = _resolve_components(components)

    if models_dics_list is None:
        models_dics_list = models_dics
    elif models_dics is not None:
        raise ValueError("Pass only one of models_dics_list or models_dics.")

    run_groups = _normalise_models_dics_list_for_seed(
        models_dics_list,
        group_labels=cfg["group_labels"],
        seed=seed,
    )
    if not run_groups:
        raise ValueError("No model groups were found.")

    fig, ax1, ax2 = _create_broken_x_axes(cfg["figsize"])
    break_x = cfg["xaxis"]["break"]
    line_entries = []
    n_color_groups = _num_color_groups(cfg, run_groups)

    for group_idx, (group_label, model_wrapper) in enumerate(run_groups):
        for name, component in components.items():
            values, lengths = _component_member_series(
                [model_wrapper],
                component["attr"],
                running_min=cfg["running_min"],
            )
            series_len = int(lengths[0])
            if series_len <= 0:
                continue

            end_idx = series_len
            if cfg["best_epoch_termination"]:
                end_idx = min(
                    series_len,
                    _best_epoch_idx_for_model(model_wrapper, series_len) + 1,
                )

            epochs = np.arange(end_idx)
            x = np.where(epochs == 0, 1e-1, epochs)
            y = _positive_for_log(values[0, :end_idx], cfg["y_floor"])
            mask1 = x <= break_x
            mask2 = x > break_x

            color_idx = _color_index_for_group(group_idx, cfg)
            color = _component_group_color(name, component, color_idx, n_color_groups)
            line_style = _line_style_for_es_group(group_idx, cfg)
            label = (
                component["label"]
                if len(run_groups) == 1
                else f"{component['label']} | {group_label}"
            )
            line_entries.append({"label": label, "color": color})

            ax1.plot(
                x[mask1],
                y[mask1],
                color=color,
                lw=cfg["line_width"],
                ls=line_style,
            )
            ax2.plot(
                x[mask2],
                y[mask2],
                color=color,
                lw=cfg["line_width"],
                ls=line_style,
            )

            if cfg["best_model_markers"]:
                _add_termination_marker(
                    ax1,
                    ax2,
                    x,
                    y,
                    break_x,
                    color,
                    cfg,
                    marker=_marker_for_group(group_idx, cfg),
                )

    _format_axes(ax1, ax2, cfg["xaxis"], cfg["fontsizes"], cfg["ylabel"], cfg["y_lim"])
    _add_broken_axis_diagonals(ax1, ax2)
    _apply_grid(ax1, ax2, cfg["grid"])
    _add_line_legend(
        ax1,
        ax2,
        line_entries,
        cfg["line_width"],
        cfg["legend"],
    )
    _add_marker_legend(ax2, cfg)

    fig.tight_layout()

    if cfg["name"]:
        plt.savefig(cfg["name"], dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", cfg["name"])

    plt.show()

    return {
        "name": cfg["name"],
        "xaxis": cfg["xaxis"],
        "components": {
            name: component["attr"] for name, component in components.items()
        },
        "running_min": cfg["running_min"],
        "best_epoch_termination": cfg["best_epoch_termination"],
        "grid": cfg["grid"],
        "seed": seed,
        "num_groups": len(run_groups),
        "num_runs": len(run_groups),
    }


def plot_smoothed_loss_components_seed_broken_x_log_lst(
    models_dics_list=None,
    plot_settings=None,
    components=None,
    models_dics=None,
    legend_labels=None,
    legend_label_title=None,
    seed=0,
    smoothing_window=25,
    separate_plots_by_group=False,
    include_flip_epoch_in_legend=False,
    print_best_epoch_losses=False,
    normalize_by_loss_weights=False,
):
    """
    Plot smoothed raw loss components for one split from each model group.

    This mirrors `plot_running_min_loss_components_seed_broken_x_log_lst`, but
    it uses the selected seed's raw loss trajectories with a moving-average
    smoothing window instead of a running minimum. Set
    `separate_plots_by_group=True` to emit one figure per color-group. Set
    `include_flip_epoch_in_legend=True` to annotate the ES legend with the
    first smoothed PDE/data flip epoch. Set `print_best_epoch_losses=True` to
    print the saved PDE and data losses at the model's best epoch. Set
    `normalize_by_loss_weights=True` to divide stored weighted loss terms by
    their associated model weights (currently `surface_weight` for data and
    `pde_weight` for PDE), so plotted curves reflect the underlying residuals.
    """
    plot_settings = {
        "name": "smoothed_loss_components_seed_broken_xaxis_loglog.png",
        "ylabel": "Smoothed loss component [a.u]",
        "fill": False,
        **(plot_settings or {}),
        "running_min": False,
    }

    if legend_labels is not None:
        plot_settings["group_labels"] = list(legend_labels)
    if legend_label_title is not None:
        legend_settings = dict(plot_settings.get("legend", {}))
        legend_settings["title"] = legend_label_title
        plot_settings["legend"] = legend_settings

    cfg = _merge_plot_settings(plot_settings)
    components = _resolve_components(components)

    if models_dics_list is None:
        models_dics_list = models_dics
    elif models_dics is not None:
        raise ValueError("Pass only one of models_dics_list or models_dics.")

    run_groups = _normalise_models_dics_list_for_seed(
        models_dics_list,
        group_labels=cfg["group_labels"],
        seed=seed,
    )
    if not run_groups:
        raise ValueError("No model groups were found.")

    n_color_groups = _num_color_groups(cfg, run_groups)
    plotted_names = []

    if separate_plots_by_group:
        grouped_runs = {}
        group_order = []
        for group_idx, (group_label, model_wrapper) in enumerate(run_groups):
            color_idx = _color_index_for_group(group_idx, cfg)
            if color_idx not in grouped_runs:
                grouped_runs[color_idx] = []
                group_order.append(color_idx)
            grouped_runs[color_idx].append((group_idx, group_label, model_wrapper))
        plot_groups = [grouped_runs[color_idx] for color_idx in group_order]
    else:
        plot_groups = [[
            (group_idx, group_label, model_wrapper)
            for group_idx, (group_label, model_wrapper) in enumerate(run_groups)
        ]]

    for active_groups in plot_groups:
        plot_name = cfg["name"]
        if separate_plots_by_group:
            plot_name = _group_plot_name(cfg["name"], active_groups[0][1])

        fig, ax1, ax2 = _create_broken_x_axes(cfg["figsize"])
        break_x = cfg["xaxis"]["break"]
        line_entries = []
        flip_epochs_by_group = {}

        for group_idx, group_label, model_wrapper in active_groups:
            if print_best_epoch_losses:
                _print_best_epoch_component_losses(group_label, model_wrapper, components)
            component_epochs = {}
            for name, component in components.items():
                values, lengths = _component_member_series(
                    [model_wrapper],
                    component["attr"],
                    running_min=False,
                )
                weight = _component_weight(
                    model_wrapper,
                    component,
                    normalize_by_loss_weights,
                )
                if weight != 1.0:
                    values = values / weight
                values = _apply_smoothing(values, lengths, smoothing_window)
                series_len = int(lengths[0])
                if series_len <= 0:
                    continue

                end_idx = series_len
                if cfg["best_epoch_termination"]:
                    end_idx = min(
                        series_len,
                        _best_epoch_idx_for_model(model_wrapper, series_len) + 1,
                    )

                epochs = np.arange(end_idx)
                x = np.where(epochs == 0, 1e-1, epochs)
                y = _positive_for_log(values[0, :end_idx], cfg["y_floor"])
                mask1 = x <= break_x
                mask2 = x > break_x

                if separate_plots_by_group:
                    component_epochs[name] = {
                        "epochs": epochs,
                        "values": values[0, :end_idx].copy(),
                    }

                color_idx = _color_index_for_group(group_idx, cfg)
                color = _component_group_color(name, component, color_idx, n_color_groups)
                line_style = _line_style_for_es_group(group_idx, cfg)
                label = (
                    component["label"]
                    if separate_plots_by_group or len(run_groups) == 1
                    else f"{component['label']} | {group_label}"
                )
                line_entries.append({"label": label, "color": color})

                ax1.plot(
                    x[mask1],
                    y[mask1],
                    color=color,
                    lw=cfg["line_width"],
                    ls=line_style,
                )
                ax2.plot(
                    x[mask2],
                    y[mask2],
                    color=color,
                    lw=cfg["line_width"],
                    ls=line_style,
                )

                if cfg["best_model_markers"]:
                    _add_termination_marker(
                        ax1,
                        ax2,
                        x,
                        y,
                        break_x,
                        color,
                        cfg,
                        marker=_marker_for_group(group_idx, cfg),
                    )

            if separate_plots_by_group and include_flip_epoch_in_legend:
                flip_epochs_by_group[group_idx] = _first_pde_data_flip_epoch(component_epochs)

        legend_cfg = cfg["legend"]
        if separate_plots_by_group:
            legend_cfg = _legend_with_group_label(
                {**cfg["legend"], "title": None},
                group_label=None,
            )

        _format_axes(ax1, ax2, cfg["xaxis"], cfg["fontsizes"], cfg["ylabel"], cfg["y_lim"])
        _add_broken_axis_diagonals(ax1, ax2)
        _apply_grid(ax1, ax2, cfg["grid"])
        _add_line_legend(
            ax1,
            ax2,
            line_entries,
            cfg["line_width"],
            legend_cfg,
        )
        marker_entries = None
        if separate_plots_by_group:
            marker_entries = _marker_legend_entries_for_groups(
                [group_idx for group_idx, _, _ in active_groups],
                cfg,
                flip_epochs_by_group=(
                    flip_epochs_by_group if include_flip_epoch_in_legend else None
                ),
            )
        _add_marker_legend(ax2, cfg, marker_entries=marker_entries)

        fig.tight_layout()

        if plot_name:
            plt.savefig(plot_name, dpi=100, bbox_inches="tight", facecolor="None")
            print("saved plot:", plot_name)
            plotted_names.append(plot_name)

        plt.show()

    return {
        "name": cfg["name"],
        "names": plotted_names or ([cfg["name"]] if cfg["name"] else []),
        "xaxis": cfg["xaxis"],
        "components": {
            name: component["attr"] for name, component in components.items()
        },
        "running_min": cfg["running_min"],
        "best_epoch_termination": cfg["best_epoch_termination"],
        "grid": cfg["grid"],
        "seed": seed,
        "smoothing_window": int(smoothing_window),
        "separate_plots_by_group": bool(separate_plots_by_group),
        "include_flip_epoch_in_legend": bool(include_flip_epoch_in_legend),
        "print_best_epoch_losses": bool(print_best_epoch_losses),
        "normalize_by_loss_weights": bool(normalize_by_loss_weights),
        "num_groups": len(run_groups),
        "num_runs": len(run_groups),
    }
