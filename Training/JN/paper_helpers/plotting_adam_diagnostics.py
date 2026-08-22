"""Diagnostics for Adam-era diffusion constraint behaviour.

Contents
--------
- _is_model_wrapper
- _run_groups_from_models_dics
- _normalise_models_dics_list
- _merge_settings
- _positive
- _raw_array
- _pad_stack
- _collect_attr_series
- _stored_series_should_fallback
- _diff_prediction_series
- _d_bounds
- _prediction_bound_violation_series
- _prediction_mono_violation_series
- _prediction_violation_series
- _infer_prediction_frequency
- _collect_prediction_violation_stats
- _best_model_and_epoch_for_group
- _best_epoch_for_group
- _evaluated_best_d_mono_loss
- _legend_label_for_group
- _collect_stats
- _component_config
- _interpolate_hex_color
- _diagnostic_group_color
- _component_output_name
- _create_broken_x_axes
- _add_broken_axis_diagonals
- _set_log_safe_xlim
- _set_log_safe_ylim
- _format_single_axis
- _format_broken_axes
- _plot_series
- _fill_series
- _plot_on_single_axis
- _plot_on_broken_axes
- _marker_for_group
- _color_index_for_group
- _num_color_groups
- _add_best_model_marker
- _add_best_model_marker_to_axes
- _add_group_legend
- _add_marker_legend
- plot_adam_D_diagnostics"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D


DIAGNOSTIC_ATTRS = {
    "data": {
        "attr": "val_data_loss_list",
        "label": "Val data loss",
        "color": "#DB2777",
        "panel": "loss",
    },
    "pde": {
        "attr": "val_pde_loss_list",
        "label": "Val PDE loss",
        "color": "#7E22CE",
        "panel": "loss",
    },
    "D_bound": {
        "attr": "val_D_bound_loss_list",
        "label": "D bound violation",
        "color": "#E11D48",
        "panel": "loss",
        "prediction_violation": "D_bound",
    },
    "D_mono": {
        "attr": "val_D_mono_loss_list",
        "label": "D mono violation",
        "color": "#BE185D",
        "panel": "loss",
        "prediction_violation": "D_mono",
    },
    "D_exp_avg_sq": {
        "attr": "adam_D_exp_avg_sq_mean_list",
        "label": "Adam D exp_avg_sq mean",
        "color": "#7E22CE",
        "panel": "adam",
    },
    "D_exp_avg_sq_max": {
        "attr": "adam_D_exp_avg_sq_max_list",
        "label": "Adam D exp_avg_sq max",
        "color": "#7E22CE",
        "panel": "adam",
    },
    "D_effective_lr": {
        "attr": "adam_D_effective_lr_mean_list",
        "label": "Adam D effective lr mean",
        "color": "#DB2777",
        "panel": "lr",
    },
    "D_effective_lr_min": {
        "attr": "adam_D_effective_lr_min_list",
        "label": "Adam D effective lr min",
        "color": "#DB2777",
        "panel": "lr",
    },
    "D_grad_norm": {
        "attr": "adam_D_grad_norm_list",
        "label": "D grad norm",
        "color": "#E11D48",
        "panel": "grad",
    },
}


DIAGNOSTIC_PALETTES = {
    "data": ("#FBCFE8", "#EC4899", "#831843"),
    "pde": ("#E9D5FF", "#A855F7", "#581C87"),
    "D_bound": ("#FFE4E6", "#F43F5E", "#881337"),
    "D_mono": ("#FCE7F3", "#DB2777", "#831843"),
    "D_exp_avg_sq": ("#E9D5FF", "#A855F7", "#581C87"),
    "D_exp_avg_sq_max": ("#E9D5FF", "#A855F7", "#581C87"),
    "D_effective_lr": ("#FBCFE8", "#EC4899", "#831843"),
    "D_effective_lr_min": ("#FBCFE8", "#EC4899", "#831843"),
    "D_grad_norm": ("#FFE4E6", "#F43F5E", "#881337"),
}


DEFAULT_SETTINGS = {
    "figsize": (7, 5),
    "name": None,
    "fill": True,
    "fill_alpha": 0.18,
    "line_width": 1.5,
    "yscale": "log",
    "xscale": "linear",
    "x_min": None,
    "x_max": None,
    "y_min": None,
    "y_max": None,
    "ylabel": None,
    "xaxis": None,
    "fontsizes": {
        "xaxis": 12,
        "xtick_labels": 10,
        "yaxis": 12,
        "ytick_labels": 10,
    },
    "y_floor": 1e-16,
    "group_labels": None,
    "legend": {
        "panel": 1,
        "loc": (0.05, 0.95),
        "loc_upd": (0.35, 0.95),
        "fontsize": 9,
        "title": "$W_u$",
        "marker_title": "ES",
        "ncols": 1,
        "allow_overlap": True,
    },
    "es_entries": [],
    "marker_indices": None,
    "color_indices": None,
    "show": True,
    "zero_loss_tol": 1e-30,
    "best_model_markers": True,
    "best_marker_size": 40,
    "components": [
        "data",
        "pde",
        "D_bound",
        "D_mono",
        "D_exp_avg_sq",
        "D_effective_lr",
        "D_grad_norm",
    ],
}


PANEL_LABELS = {
    "loss": "Loss [a.u.]",
    "adam": "Adam second moment [a.u.]",
    "lr": "Effective LR [a.u.]",
    "grad": "Gradient norm [a.u.]",
    "extra": "Diagnostic [a.u.]",
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
        raise ValueError("plot_settings['group_labels'] must match models_dics_list length.")

    run_groups = []
    for models_dics, group_label in zip(models_dics_list, group_labels):
        run_groups.extend(_run_groups_from_models_dics(models_dics, group_label))
    return run_groups


def _merge_settings(plot_settings):
    settings = {**DEFAULT_SETTINGS, **(plot_settings or {})}
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


def _positive(values, y_floor):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= 0, y_floor, values)


def _raw_array(values):
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _pad_stack(series_list):
    lengths = [len(series) for series in series_list]
    max_len = max(lengths)
    values = np.full((len(series_list), max_len), np.nan, dtype=np.float64)
    for i, series in enumerate(series_list):
        values[i, : len(series)] = series
    return values


def _collect_attr_series(model_wrappers, attr):
    series_list = []
    missing = []
    for i, model_wrapper in enumerate(model_wrappers):
        values = getattr(model_wrapper, attr, None)
        if values is None:
            missing.append(i)
            continue
        if len(values) == 0:
            missing.append(i)
            continue
        series_list.append(np.asarray(values, dtype=np.float64))
    return series_list, missing


def _stored_series_should_fallback(series_list, zero_loss_tol):
    if not series_list:
        return True
    finite_values = np.concatenate(
        [np.ravel(series[np.isfinite(series)]) for series in series_list]
    )
    if finite_values.size == 0:
        return True
    return np.nanmax(np.abs(finite_values)) <= zero_loss_tol


def _diff_prediction_series(model_wrapper):
    preds = getattr(model_wrapper, "diffusion_preds", None)
    if preds is None or len(preds) == 0:
        return None
    return [_raw_array(pred).reshape(-1) for pred in preds]


def _d_bounds(model_wrapper):
    model = getattr(model_wrapper, "model", None)
    lower = getattr(model, "D_min", 0.0)
    upper = getattr(model, "D_max", 0.1)
    return float(lower), float(upper)


def _prediction_bound_violation_series(model_wrapper):
    preds = _diff_prediction_series(model_wrapper)
    if preds is None:
        return None

    lower, upper = _d_bounds(model_wrapper)
    return np.asarray(
        [
            np.mean(np.maximum(lower - pred, 0.0) ** 2 + np.maximum(pred - upper, 0.0) ** 2)
            for pred in preds
        ],
        dtype=np.float64,
    )


def _prediction_mono_violation_series(model_wrapper, direction="increasing"):
    preds = _diff_prediction_series(model_wrapper)
    if preds is None:
        return None

    series = []
    for pred in preds:
        diffs = np.diff(pred)
        if direction == "increasing":
            violation = np.maximum(-diffs, 0.0)
        elif direction == "decreasing":
            violation = np.maximum(diffs, 0.0)
        else:
            raise ValueError(f"Unknown D monotonicity direction: {direction!r}")
        series.append(np.mean(violation ** 2) if violation.size else 0.0)
    return np.asarray(series, dtype=np.float64)


def _prediction_violation_series(model_wrapper, violation_type):
    if violation_type == "D_bound":
        return _prediction_bound_violation_series(model_wrapper)
    if violation_type == "D_mono":
        return _prediction_mono_violation_series(model_wrapper)
    raise ValueError(f"Unknown prediction violation type: {violation_type!r}")


def _infer_prediction_frequency(model_wrapper, prediction_count):
    model = getattr(model_wrapper, "model", None)
    epoch_count = getattr(model, "epochs", None)
    if epoch_count is None or epoch_count <= 0:
        epoch_count = len(getattr(model_wrapper, "train_loss_list", []))
    if not prediction_count or not epoch_count:
        return 1
    return max(1, int(round(epoch_count / prediction_count)))


def _collect_prediction_violation_stats(model_wrappers, violation_type, y_floor):
    series_list = []
    epochs_list = []
    missing = []

    for i, model_wrapper in enumerate(model_wrappers):
        series = _prediction_violation_series(model_wrapper, violation_type)
        if series is None or len(series) == 0:
            missing.append(i)
            continue
        series_list.append(series)
        frequency = _infer_prediction_frequency(model_wrapper, len(series))
        epochs_list.append(np.arange(len(series)) * frequency)

    if not series_list:
        raise AttributeError(
            f"No wrappers have non-empty diffusion_preds for {violation_type!r}."
        )

    values = _pad_stack(series_list)
    max_len = values.shape[1]
    if epochs_list:
        epoch_values = np.full((len(epochs_list), max_len), np.nan, dtype=np.float64)
        for i, epochs in enumerate(epochs_list):
            epoch_values[i, : len(epochs)] = epochs
        epochs = np.nanmedian(epoch_values, axis=0)
    else:
        epochs = np.arange(max_len)

    return {
        "epochs": epochs,
        "mean": _positive(np.nanmean(values, axis=0), y_floor),
        "min": _positive(np.nanmin(values, axis=0), y_floor),
        "max": _positive(np.nanmax(values, axis=0), y_floor),
        "missing": missing,
        "source": "diffusion_preds",
    }


def _best_model_and_epoch_for_group(model_wrappers):
    best_losses = np.asarray(
        [getattr(model, "best_val_loss", np.nan) for model in model_wrappers],
        dtype=np.float64,
    )
    if np.all(np.isnan(best_losses)):
        return None, None

    best_seed_idx = int(np.nanargmin(best_losses))
    best_model = model_wrappers[best_seed_idx]
    best_epoch = getattr(best_model, "last_improved", None)
    if best_epoch is not None:
        return best_model, int(best_epoch)

    val_losses = getattr(best_model, "val_loss_list", None)
    if val_losses is not None and len(val_losses):
        return best_model, int(np.nanargmin(np.asarray(val_losses, dtype=np.float64)))

    return best_model, None


def _best_epoch_for_group(model_wrappers):
    _, best_epoch = _best_model_and_epoch_for_group(model_wrappers)
    return best_epoch


def _evaluated_best_d_mono_loss(model_wrappers):
    best_model, best_epoch = _best_model_and_epoch_for_group(model_wrappers)
    if best_model is None:
        return None

    series = _prediction_mono_violation_series(best_model)
    if series is None or len(series) == 0:
        return None

    if best_epoch is None:
        idx = len(series) - 1
    else:
        frequency = _infer_prediction_frequency(best_model, len(series))
        epochs = np.arange(len(series)) * frequency
        idx = int(np.nanargmin(np.abs(epochs - best_epoch)))

    return float(series[idx])


def _legend_label_for_group(name, group_label, model_wrappers):
    label = str(group_label)
    if name == "D_mono":
        loss = _evaluated_best_d_mono_loss(model_wrappers)
        if loss is not None and np.isfinite(loss):
            label = f"{label}"# (best loss={loss:.2g})"
    return label


def _collect_stats(model_wrappers, cfg, y_floor, zero_loss_tol):
    attr = cfg["attr"]
    series_list, missing = _collect_attr_series(model_wrappers, attr)
    violation_type = cfg.get("prediction_violation")
    if violation_type and _stored_series_should_fallback(series_list, zero_loss_tol):
        return _collect_prediction_violation_stats(model_wrappers, violation_type, y_floor)

    if not series_list:
        if violation_type:
            return _collect_prediction_violation_stats(model_wrappers, violation_type, y_floor)
        raise AttributeError(
            f"No wrappers have non-empty {attr!r}. "
            "For Adam diagnostics, reload/retrain models saved after the "
            "Adam tracking code was added."
        )

    values = _pad_stack(series_list)
    return {
        "epochs": np.arange(values.shape[1]),
        "mean": _positive(np.nanmean(values, axis=0), y_floor),
        "min": _positive(np.nanmin(values, axis=0), y_floor),
        "max": _positive(np.nanmax(values, axis=0), y_floor),
        "missing": missing,
        "source": attr,
    }


def _component_config(component):
    if isinstance(component, str):
        if component in DIAGNOSTIC_ATTRS:
            return component, dict(DIAGNOSTIC_ATTRS[component])

        attr = component
        if not attr.endswith("_list"):
            attr = f"{attr}_list"
        return component, {
            "attr": attr,
            "label": component.replace("_", " "),
            "color": None,
            "panel": "extra",
        }

    name = component.get("name", component.get("attr"))
    if name is None:
        raise KeyError("Custom diagnostic components must define 'name' or 'attr'.")

    if name in DIAGNOSTIC_ATTRS:
        cfg = {**DIAGNOSTIC_ATTRS[name], **component}
    else:
        cfg = dict(component)
        cfg.setdefault("attr", name if name.endswith("_list") else f"{name}_list")

    cfg.setdefault("label", name.replace("_", " "))
    cfg.setdefault("color", None)
    cfg.setdefault("panel", "extra")
    return name, cfg


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


def _diagnostic_group_color(name, cfg, group_idx, n_groups):
    palette = cfg.get("palette", DIAGNOSTIC_PALETTES.get(name))
    if palette is None:
        return cfg.get("color", f"C{group_idx}")

    if n_groups <= 1:
        return cfg.get("color") or _interpolate_hex_color(palette, 0.5)

    return _interpolate_hex_color(palette, group_idx / (n_groups - 1))


def _component_output_name(base_name, component_name):
    if not base_name:
        return None

    path = Path(base_name)
    safe_component = component_name.replace("/", "_").replace(" ", "_")
    if path.suffix:
        return str(path.with_name(f"{path.stem}_{safe_component}{path.suffix}"))

    return str(path / f"{safe_component}.png")


def _create_broken_x_axes(figsize):
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 5], "wspace": 0.05},
    )
    return fig, ax1, ax2


def _add_broken_axis_diagonals(ax1, ax2):
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)


def _set_log_safe_xlim(ax, left, right, xscale):
    if xscale == "log" and left <= 0:
        left = 1.0
    ax.set_xlim(left=left, right=right)


def _set_log_safe_ylim(ax, bottom, top, yscale):
    if yscale == "log" and bottom is not None and bottom <= 0:
        bottom = 1e-16
    ax.set_ylim(bottom=bottom, top=top)


def _format_single_axis(ax, cfg, settings):
    fontsizes = settings["fontsizes"]
    ax.set_xlabel("Epoch", fontsize=fontsizes["xaxis"])
    ylabel = settings.get("ylabel") or PANEL_LABELS.get(cfg["panel"], cfg["panel"])
    ax.set_ylabel(
        ylabel,
        fontsize=fontsizes["yaxis"],
    )
    ax.tick_params(axis="x", labelsize=fontsizes["xtick_labels"])
    ax.tick_params(axis="y", labelsize=fontsizes["ytick_labels"])
    ax.set_xscale(settings["xscale"])
    ax.set_yscale(settings["yscale"])
    ax.set_facecolor("white")
    ax.grid(True, which="both", ls="-", lw=0.5, alpha=0.7)
    if settings["x_min"] is not None or settings["x_max"] is not None:
        left = settings["x_min"] if settings["x_min"] is not None else ax.get_xlim()[0]
        right = settings["x_max"] if settings["x_max"] is not None else ax.get_xlim()[1]
        _set_log_safe_xlim(ax, left, right, settings["xscale"])
    if settings["y_min"] is not None or settings["y_max"] is not None:
        bottom = settings["y_min"] if settings["y_min"] is not None else ax.get_ylim()[0]
        top = settings["y_max"] if settings["y_max"] is not None else ax.get_ylim()[1]
        _set_log_safe_ylim(ax, bottom, top, settings["yscale"])


def _format_broken_axes(ax1, ax2, cfg, settings):
    xaxis = settings["xaxis"]
    x_min = xaxis["min"]
    x_max = settings["x_max"] if settings["x_max"] is not None else xaxis["max"]
    break_x = xaxis["break"]
    fontsizes = settings["fontsizes"]
    ylabel = settings.get("ylabel") or PANEL_LABELS.get(cfg["panel"], cfg["panel"])

    for ax in (ax1, ax2):
        ax.set_xscale(settings["xscale"])
        ax.set_yscale(settings["yscale"])
        ax.set_facecolor("white")
        ax.grid(True, which="both", ls="-", lw=0.5, alpha=0.7)
        ax.tick_params(axis="x", labelsize=fontsizes["xtick_labels"])
        ax.tick_params(axis="y", labelsize=fontsizes["ytick_labels"])

    ax2.set_xlabel("Epoch", fontsize=fontsizes["xaxis"])
    ax1.set_ylabel(
        ylabel,
        fontsize=fontsizes["yaxis"],
    )
    _set_log_safe_xlim(ax1, x_min, break_x, settings["xscale"])
    _set_log_safe_xlim(ax2, break_x, x_max, settings["xscale"])

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)
    if settings["y_min"] is not None or settings["y_max"] is not None:
        bottom = settings["y_min"] if settings["y_min"] is not None else ax1.get_ylim()[0]
        top = settings["y_max"] if settings["y_max"] is not None else ax1.get_ylim()[1]
        _set_log_safe_ylim(ax1, bottom, top, settings["yscale"])
        _set_log_safe_ylim(ax2, bottom, top, settings["yscale"])
    _add_broken_axis_diagonals(ax1, ax2)


def _plot_series(ax, x, y, color, settings):
    ax.plot(x, y, color=color, lw=settings["line_width"])


def _fill_series(ax, x, y_min, y_max, color, settings):
    ax.fill_between(
        x,
        y_min,
        y_max,
        color=color,
        alpha=settings["fill_alpha"],
    )


def _plot_on_single_axis(ax, x, stats, color, settings):
    _plot_series(ax, x, stats["mean"], color, settings)
    if settings["fill"]:
        _fill_series(ax, x, stats["min"], stats["max"], color, settings)


def _plot_on_broken_axes(ax1, ax2, x, stats, color, settings):
    break_x = settings["xaxis"]["break"]
    mask1 = x <= break_x
    mask2 = x > break_x

    _plot_series(ax1, x[mask1], stats["mean"][mask1], color, settings)
    _plot_series(ax2, x[mask2], stats["mean"][mask2], color, settings)

    if settings["fill"]:
        _fill_series(
            ax1,
            x[mask1],
            stats["min"][mask1],
            stats["max"][mask1],
            color,
            settings,
        )
        _fill_series(
            ax2,
            x[mask2],
            stats["min"][mask2],
            stats["max"][mask2],
            color,
            settings,
        )


def _marker_for_group(group_idx, settings):
    es_entries = settings.get("es_entries") or []
    if not es_entries:
        return "o"

    marker_indices = settings.get("marker_indices")
    if marker_indices is not None and group_idx < len(marker_indices):
        marker_idx = marker_indices[group_idx]
    else:
        marker_idx = group_idx

    if marker_idx is None:
        return "o"
    marker_idx = int(marker_idx)
    if 0 <= marker_idx < len(es_entries):
        return es_entries[marker_idx][1]
    return "o"


def _color_index_for_group(group_idx, settings):
    color_indices = settings.get("color_indices")
    if color_indices is not None and group_idx < len(color_indices):
        return int(color_indices[group_idx])
    return group_idx


def _num_color_groups(settings, run_groups):
    color_indices = settings.get("color_indices")
    if color_indices:
        return max(int(index) for index in color_indices) + 1
    return len(run_groups)


def _add_best_model_marker(ax, x, y, best_epoch, color, settings, marker="o"):
    if best_epoch is None or len(x) == 0:
        return None

    idx = int(np.nanargmin(np.abs(np.asarray(x, dtype=np.float64) - best_epoch)))
    ax.scatter(
        x[idx],
        y[idx],
        facecolors=color,
        edgecolors="black",
        linewidths=0.8,
        s=settings["best_marker_size"],
        zorder=5,
        marker=marker,
    )
    return {"epoch": float(x[idx]), "value": float(y[idx])}


def _add_best_model_marker_to_axes(
    ax1,
    ax2,
    x,
    y,
    best_epoch,
    color,
    settings,
    marker="o",
):
    if settings.get("xaxis") is None:
        return _add_best_model_marker(ax1, x, y, best_epoch, color, settings, marker)

    if best_epoch is None or len(x) == 0:
        return None

    idx = int(np.nanargmin(np.abs(np.asarray(x, dtype=np.float64) - best_epoch)))
    target_ax = ax1 if x[idx] <= settings["xaxis"]["break"] else ax2
    target_ax.scatter(
        x[idx],
        y[idx],
        facecolors=color,
        edgecolors="black",
        linewidths=0.8,
        s=settings["best_marker_size"],
        zorder=5,
        marker=marker,
    )
    return {"epoch": float(x[idx]), "value": float(y[idx])}


def _add_group_legend(ax_or_axes, line_entries, settings):
    if isinstance(ax_or_axes, tuple):
        ax1, ax2 = ax_or_axes
        target_ax = ax1 if settings["legend"].get("panel", 1) == 1 else ax2
    else:
        target_ax = ax_or_axes

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
        Line2D([0], [0], color=entry["color"], lw=settings["line_width"])
        for entry in unique_entries
    ]
    labels = [entry["label"] for entry in unique_entries]
    legend_kwargs = dict(
        handles=handles,
        labels=labels,
        fontsize=settings["legend"]["fontsize"],
        title=settings["legend"]["title"],
        title_fontsize=settings["legend"]["fontsize"],
        ncols=settings["legend"]["ncols"],
    )
    loc = settings["legend"].get("loc", "best")
    allow_overlap = settings["legend"].get("allow_overlap", True)
    if allow_overlap and loc == "best":
        loc = "upper left"
    if isinstance(loc, tuple):
        legend = target_ax.legend(
            loc="upper left",
            bbox_to_anchor=loc,
            **legend_kwargs,
        )
    else:
        legend = target_ax.legend(loc=loc, **legend_kwargs)
    legend.set_in_layout(False)
    target_ax.add_artist(legend)


def _add_marker_legend(ax_or_axes, settings):
    es_entries = settings.get("es_entries") or []
    if not es_entries:
        return

    if isinstance(ax_or_axes, tuple):
        _, ax2 = ax_or_axes
        target_ax = ax2
    else:
        target_ax = ax_or_axes

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=8,
        )
        for _, marker in es_entries
    ]
    marker_labels = [label for label, _ in es_entries]

    legend = target_ax.legend(
        handles=marker_handles,
        labels=marker_labels,
        loc="upper left",
        bbox_to_anchor=settings["legend"].get("loc_upd", (1, 1)),
        fontsize=settings["legend"]["fontsize"],
        title=settings["legend"].get("marker_title", "ES"),
        title_fontsize=settings["legend"]["fontsize"],
    )
    legend.set_in_layout(False)
    target_ax.add_artist(legend)


def plot_adam_D_diagnostics(
    models_dics_list=None,
    plot_settings=None,
    models_dics=None,
):
    """
    Plot Adam conditioning diagnostics for the diffusion head.

    Parameters
    ----------
    models_dics_list : ModelWrapper, dict, or list
        Same grouped input style as the loss-component plotting helpers. Each
        group may be one ModelWrapper, {seed: modelWrapper, ...}, or
        {config_key: {seed: modelWrapper, ...}}.
    plot_settings : dict, optional
        Useful keys:
        - components: list of aliases/attrs to display
        - name: base path for saved figures, or None. If a suffix is supplied,
          each component is saved as stem_component.suffix.
        - x_min, x_max, y_min, y_max, ylabel, xscale, yscale
    """
    settings = _merge_settings(plot_settings)

    if models_dics_list is None:
        models_dics_list = models_dics
    elif models_dics is not None:
        raise ValueError("Pass only one of models_dics_list or models_dics.")

    run_groups = _normalise_models_dics_list(
        models_dics_list,
        group_labels=settings["group_labels"],
    )

    component_configs = []
    for i, component in enumerate(settings["components"]):
        name, cfg = _component_config(component)
        if cfg["color"] is None:
            cfg["color"] = f"C{i}"
        component_configs.append((name, cfg))

    output = {
        "num_groups": len(run_groups),
        "num_runs": sum(len(model_wrappers) for _, model_wrappers in run_groups),
        "components": {},
        "sources": {},
        "saved": {},
        "best_markers": {},
    }
    for name, cfg in component_configs:
        output["components"][name] = cfg["attr"]
        if settings.get("xaxis") is None:
            fig, ax = plt.subplots(1, 1, figsize=settings["figsize"])
            axes_for_plot = (ax, None)
        else:
            fig, ax1, ax2 = _create_broken_x_axes(settings["figsize"])
            axes_for_plot = (ax1, ax2)
        line_entries = []
        sources = []
        best_markers = []
        n_color_groups = _num_color_groups(settings, run_groups)

        for group_idx, (group_label, model_wrappers) in enumerate(run_groups):
            stats = _collect_stats(
                model_wrappers,
                cfg,
                settings["y_floor"],
                settings["zero_loss_tol"],
            )
            color_idx = _color_index_for_group(group_idx, settings)
            color = _diagnostic_group_color(name, cfg, color_idx, n_color_groups)
            label = _legend_label_for_group(name, group_label, model_wrappers)
            line_entries.append({"label": label, "color": color})
            sources.append(stats["source"])

            x = np.asarray(stats["epochs"], dtype=np.float64)
            if settings["xscale"] == "log":
                x = np.where(x <= 0, 1e-1, x)

            if settings.get("xaxis") is None:
                ax, _ = axes_for_plot
                _plot_on_single_axis(ax, x, stats, color, settings)
            else:
                ax1, ax2 = axes_for_plot
                _plot_on_broken_axes(ax1, ax2, x, stats, color, settings)

            if settings["best_model_markers"]:
                marker_style = _marker_for_group(group_idx, settings)
                marker = _add_best_model_marker_to_axes(
                    axes_for_plot[0],
                    axes_for_plot[1],
                    x,
                    stats["mean"],
                    _best_epoch_for_group(model_wrappers),
                    color,
                    settings,
                    marker_style,
                )
                best_markers.append(marker)

        if settings.get("xaxis") is None:
            ax, _ = axes_for_plot
            _format_single_axis(ax, cfg, settings)
        else:
            ax1, ax2 = axes_for_plot
            _format_broken_axes(ax1, ax2, cfg, settings)
        # title = cfg["label"]
        # if "diffusion_preds" in sources:
        #     title = f"{title} (from D predictions)"
        #ax.set_title(title)

        legend_axes = axes_for_plot if settings.get("xaxis") is not None else axes_for_plot[0]
        _add_group_legend(legend_axes, line_entries, settings)
        _add_marker_legend(legend_axes, settings)
        fig.tight_layout()

        output_name = _component_output_name(settings["name"], name)
        if output_name:
            plt.savefig(output_name, dpi=100, bbox_inches="tight", facecolor="None")
            print("saved plot:", output_name)
            output["saved"][name] = output_name
        output["sources"][name] = sources
        output["best_markers"][name] = best_markers

        if settings["show"]:
            plt.show()
        else:
            plt.close(fig)

    return output
