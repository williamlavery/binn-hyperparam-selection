"""Plots learned diffusion and growth functions across saved epochs.

Contents
--------
- _as_numpy
- _blend_with_white
- _blend_colors
- _epoch_color
- _epoch_color_map
- _get_density_scale
- _select_one_wrapper_per_group
- _select_wrapper
- _last_best_training_epoch
- _build_epochs_to_plot
- _infer_prediction_epoch_stride
- _resolve_epochs_sf
- _get_prediction
- _global_y_limits
- _restrict_to_central_range
- _density_values_from_wrapper
- _central_u_bounds
- _build_legend_handles
- _style_axes
- _save_secondary_copy
- _build_save_base
- _quantity_defaults
- _default_y_lims
- _normalize_y_lims
- _format_limit_value
- _y_lim_suffix
- _truth_attr_for_quantity
- _truth_label_for_quantity
- _get_truth_values
- _best_pred_attr_for_quantity
- _get_best_prediction_values
- _normalize_errs
- _merge_epoch_plot_settings
- _legend_for_plot
- _groups_from_models_dics_list
- _color_from_plot_params
- _plot_quantity_across_epochs
- plot_diffusion_across_epochs
- plot_growth_across_epochs"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from .paths import dictToPath
from .utils import hist_properties


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().flatten().numpy()
    return np.asarray(x).flatten()


def _blend_with_white(base_color: str, t: float) -> Tuple[float, float, float, float]:
    br, bg, bb, ba = to_rgba(base_color)
    r = (1 - t) * 1.0 + t * br
    g = (1 - t) * 1.0 + t * bg
    b = (1 - t) * 1.0 + t * bb
    return r, g, b, ba


def _blend_colors(
    start_color: str,
    end_color: str,
    t: float,
) -> Tuple[float, float, float, float]:
    sr, sg, sb, sa = to_rgba(start_color)
    er, eg, eb, ea = to_rgba(end_color)
    t = min(1.0, max(0.0, float(t)))
    return (
        (1 - t) * sr + t * er,
        (1 - t) * sg + t * eg,
        (1 - t) * sb + t * eb,
        (1 - t) * sa + t * ea,
    )


def _epoch_color(
    ep: int,
    plotted_epochs: Sequence[int],
    base_color: str,
    best_color: str,
    best_saved_epoch: int,
) -> Tuple[float, float, float, float]:
    plotted_sorted = sorted(plotted_epochs)

    if len(plotted_sorted) == 1:
        return to_rgba(best_color)

    if ep == best_saved_epoch:
        return to_rgba(best_color)

    before_best = [epoch for epoch in plotted_sorted if epoch < best_saved_epoch]
    after_best = [epoch for epoch in plotted_sorted if epoch > best_saved_epoch]

    if ep < best_saved_epoch:
        if not before_best:
            return to_rgba(base_color)
        idx = before_best.index(ep)
        t_raw = idx / max(1, len(before_best) - 1)
        return _blend_with_white(base_color, 0.2 + 0.8 * t_raw)

    if not after_best:
        return to_rgba(best_color)

    idx = after_best.index(ep)
    t_raw = (idx + 1) / max(1, len(after_best))
    # Keep post-best curves visibly purple instead of fading to near-white.
    return _blend_with_white(best_color, 0.8 - 0.5 * t_raw)


def _epoch_color_map(
    plotted_epochs: Sequence[int],
    base_color: str,
    best_color: str,
    best_saved_epoch: int,
) -> Dict[int, Tuple[float, float, float, float]]:
    plotted_sorted = sorted(plotted_epochs)
    n_colors = len(plotted_sorted)

    if n_colors == 1:
        return {plotted_sorted[0]: to_rgba(best_color)}

    return {
        ep: _epoch_color(
            ep=ep,
            plotted_epochs=plotted_sorted,
            base_color=base_color,
            best_color=best_color,
            best_saved_epoch=best_saved_epoch,
        )
        for ep in plotted_sorted
    }


def _get_density_scale(wrapper: Any, species_label: str) -> float:
    model = wrapper.model
    val = None

    if species_label.lower() == "red" and hasattr(model, "u_red_max"):
        val = float(model.u_red_max) * 1e6
    elif species_label.lower() == "green" and hasattr(model, "u_green_max"):
        val = float(model.u_green_max) * 1e6
    elif hasattr(model, "u_max"):
        val = float(model.u_max) * 1e6

    if val is None or not np.isfinite(val) or val <= 0:
        return 1.0

    return val


def _select_one_wrapper_per_group(
    model_wrapper_groups: Dict[Any, Dict[Any, Any]],
    model_index: int,
    seed_index: int = 0,
) -> Dict[Any, Any]:
    selected = {}

    for group_key, group_dict in model_wrapper_groups.items():
        if not group_dict:
            raise ValueError(f"Group '{group_key}' has no models.")

        if hasattr(group_dict, "model"):
            selected[group_key] = group_dict
            continue

        if isinstance(group_dict, (list, tuple)):
            if model_index < 0 or model_index >= len(group_dict):
                raise IndexError(
                    f"model_index={model_index} out of range for group '{group_key}' "
                    f"(has {len(group_dict)} models)."
                )
            selected[group_key] = _select_wrapper(group_dict[model_index], seed_index)
            continue

        if not isinstance(group_dict, dict):
            raise TypeError(
                "Expected each group to be a model wrapper, list, or dictionary; "
                f"group '{group_key}' is {type(group_dict).__name__}."
            )

        if all(hasattr(value, "model") for value in group_dict.values()):
            selected[group_key] = _select_wrapper(group_dict, seed_index)
            continue

        inner_keys = sorted(group_dict.keys())

        if model_index < 0 or model_index >= len(inner_keys):
            raise IndexError(
                f"model_index={model_index} out of range for group '{group_key}' "
                f"(has {len(inner_keys)} models)."
            )

        selected[group_key] = _select_wrapper(group_dict[inner_keys[model_index]], seed_index)

    return selected


def _select_wrapper(candidate: Any, seed_index: int = 0) -> Any:
    if hasattr(candidate, "model"):
        return candidate

    if isinstance(candidate, dict):
        inner_keys = sorted(candidate.keys())

        if not inner_keys:
            raise ValueError("Cannot select a model wrapper from an empty dictionary.")

        if seed_index < 0 or seed_index >= len(inner_keys):
            raise IndexError(
                f"seed_index={seed_index} out of range for selected model "
                f"(has {len(inner_keys)} seeds)."
            )

        return _select_wrapper(candidate[inner_keys[seed_index]], seed_index=0)

    raise TypeError(
        "Expected a model wrapper or nested dictionary containing model wrappers; "
        f"got {type(candidate).__name__}."
    )


def _last_best_training_epoch(wrapper: Any, fallback_epoch: int) -> int:
    if hasattr(wrapper, "last_improved"):
        return int(wrapper.last_improved)
    return fallback_epoch


def _build_epochs_to_plot(
    wrapper: Any,
    pred_attr: str,
    epoch_step: int,
    epochs_sf: int,
    continue_after_best: bool = False,
    post_best_only: bool = False,
) -> Tuple[Sequence[int], int]:
    saved_count = len(getattr(wrapper, pred_attr))

    if saved_count <= 0:
        raise ValueError(f"No saved predictions found for '{pred_attr}'.")

    fallback_best_raw = (saved_count - 1) * epochs_sf
    best_raw_epoch = _last_best_training_epoch(wrapper, fallback_best_raw)

    best_saved_epoch = min(saved_count - 1, int(np.floor(best_raw_epoch / epochs_sf)))
    start_saved_epoch = best_saved_epoch if post_best_only else 0
    end_saved_epoch = saved_count - 1 if continue_after_best else best_saved_epoch

    epochs = list(
        range(
            start_saved_epoch,
            end_saved_epoch + 1,
            max(1, int(epoch_step)),
        )
    )

    if best_saved_epoch not in epochs:
        epochs.append(best_saved_epoch)
    if epochs[-1] != end_saved_epoch:
        epochs.append(end_saved_epoch)

    return sorted(set(epochs)), best_saved_epoch


def _infer_prediction_epoch_stride(wrapper: Any, pred_attr: str) -> Optional[int]:
    saved_count = len(getattr(wrapper, pred_attr, []))
    if saved_count <= 1:
        return None

    save_index = np.asarray(getattr(wrapper, "save_index", []), dtype=float)
    if save_index.size >= 2:
        diffs = np.diff(save_index)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return max(1, int(round(float(np.median(diffs)))))

    model_epochs = getattr(getattr(wrapper, "model", None), "epochs", None)
    if model_epochs is not None and model_epochs > 0:
        return max(1, int(round(float(model_epochs) / float(saved_count - 1))))

    return None


def _resolve_epochs_sf(
    selected_wrappers: Dict[Any, Any],
    pred_attr: str,
    epochs_sf: Optional[int],
) -> int:
    if epochs_sf is not None:
        return max(1, int(epochs_sf))

    inferred = [
        stride
        for stride in (
            _infer_prediction_epoch_stride(wrapper, pred_attr)
            for wrapper in selected_wrappers.values()
        )
        if stride is not None
    ]
    if inferred:
        return max(1, int(round(float(np.median(inferred)))))

    return 1


def _get_prediction(
    wrapper: Any,
    pred_attr: str,
    ep: int,
    target_len: int,
) -> np.ndarray:
    seq = getattr(wrapper, pred_attr)

    if ep >= len(seq):
        return np.zeros(target_len, dtype=float)

    arr = _as_numpy(seq[ep])

    if arr.size < target_len:
        out = np.zeros(target_len, dtype=float)
        out[:arr.size] = arr
        return out

    if arr.size > target_len:
        return arr[:target_len]

    return arr


def _global_y_limits(
    selected_wrappers: Dict[Any, Any],
    x_vals_by_group: Dict[Any, np.ndarray],
    pred_attr: str,
    epoch_step: int,
    epochs_sf: int,
    continue_after_best: bool = False,
    post_best_only: bool = False,
) -> Tuple[float, float]:
    y_min, y_max = np.inf, -np.inf

    for group_key, wrapper in selected_wrappers.items():
        xg = x_vals_by_group[group_key]
        epochs, _ = _build_epochs_to_plot(
            wrapper=wrapper,
            pred_attr=pred_attr,
            epoch_step=epoch_step,
            epochs_sf=epochs_sf,
            continue_after_best=continue_after_best,
            post_best_only=post_best_only,
        )

        for ep in epochs:
            arr = _get_prediction(wrapper, pred_attr, ep, target_len=len(xg))
            y_min = min(y_min, float(arr.min()))
            y_max = max(y_max, float(arr.max()))

    return (
        y_min if np.isfinite(y_min) else 0.0,
        y_max if np.isfinite(y_max) else 1.0,
    )


def _restrict_to_central_range(
    x: np.ndarray,
    y: np.ndarray,
    low_x: float,
    high_x: float,
    restrict: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if not restrict:
        return x, y

    mask = (x >= low_x) & (x <= high_x)
    return x[mask], y[mask]


def _density_values_from_wrapper(wrapper: Any) -> Optional[np.ndarray]:
    candidates = []

    for attr in ("u_clean", "u_nosiy", "u_noisy"):
        if hasattr(wrapper, attr):
            candidates.append(np.asarray(getattr(wrapper, attr), dtype=float).reshape(-1))

    for attr in ("y_train", "y_val"):
        if hasattr(wrapper, attr):
            candidates.append(np.asarray(getattr(wrapper, attr), dtype=float).reshape(-1))

    if not candidates:
        return None

    values = np.concatenate(candidates)
    values = values[np.isfinite(values)]
    return values if values.size else None


def _central_u_bounds(
    dataobj: Any,
    selected_wrappers: Dict[Any, Any],
    num_bins: int,
) -> Optional[Tuple[float, float]]:
    if dataobj is not None:
        h_props = hist_properties(dataobj, num_bins_data_plot=num_bins)
        return float(h_props["low_count"]), float(h_props["high_count"])

    wrapper_values = []
    for wrapper in selected_wrappers.values():
        values = _density_values_from_wrapper(wrapper)
        if values is not None:
            wrapper_values.append(values)

    if not wrapper_values:
        return None

    values = np.concatenate(wrapper_values)
    return float(np.percentile(values, 5)), float(np.percentile(values, 95))


def _build_legend_handles(
    plotted_epochs: Sequence[int],
    epochs_sf: int,
    color_fn,
    best_model_handle: Optional[Line2D],
) -> Sequence[Line2D]:
    handles = []

    plotted_sorted = sorted(plotted_epochs)
    n_epochs = len(plotted_sorted)

    num_demo = min(3, n_epochs)
    demo_positions = sorted(
        {int(round(x)) for x in np.linspace(0, n_epochs - 1, num_demo)}
    )

    for pos in demo_positions:
        ep_demo = plotted_sorted[pos]
        train_ep_demo = ep_demo * epochs_sf
        handles.append(
            Line2D(
                [0],
                [0],
                color=color_fn(ep_demo),
                lw=3,
                label=f"{train_ep_demo}",
            )
        )

    if best_model_handle is not None:
        handles.append(best_model_handle)

    return handles


def _style_axes(
    ax: Any,
    *,
    xtick_labelsize: float,
    ytick_labelsize: float,
    major_tick_length: float,
    major_tick_width: float,
    minor_tick_length: float,
    minor_tick_width: float,
) -> None:
    ax.tick_params(
        axis="x",
        which="major",
        labelsize=xtick_labelsize,
        length=major_tick_length,
        width=major_tick_width,
    )
    ax.tick_params(
        axis="y",
        which="major",
        labelsize=ytick_labelsize,
        length=major_tick_length,
        width=major_tick_width,
    )
    ax.tick_params(
        axis="x",
        which="minor",
        length=minor_tick_length,
        width=minor_tick_width,
    )
    ax.tick_params(
        axis="y",
        which="minor",
        length=minor_tick_length,
        width=minor_tick_width,
    )


def _save_secondary_copy(
    image_path: str,
    save_dir2: str,
    figsize: Tuple[float, float],
    dpi: int,
) -> None:
    import matplotlib.image as mpimg

    os.makedirs(save_dir2, exist_ok=True)
    out_path = os.path.join(save_dir2, os.path.basename(image_path))

    img = mpimg.imread(image_path)
    fig_copy, ax_copy = plt.subplots(figsize=figsize)
    ax_copy.imshow(img)
    ax_copy.axis("off")
    fig_copy.tight_layout()
    fig_copy.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_copy)


def _build_save_base(save_dic: Dict[str, Any], save_name: str, base_dir: str) -> str:
    if save_dic:
        return os.path.join(base_dir, dictToPath(save_dic), save_name)
    return os.path.join(base_dir, save_name)


def _quantity_defaults(quantity_name: str) -> Dict[str, Any]:
    defaults = {
        "diffusion": {
            "pred_attr": "diffusion_preds",
            "model_eval_attr": "diffusion",
            "model_scale_attr": "D_scale",
            "ylabel": r"Diffusion [mm$^2$ days$^{-1}$]",
            "base_color": "#d35400",
            "best_color": "#d35400",
        },
        "growth": {
            "pred_attr": "growth_preds",
            "model_eval_attr": "growth",
            "model_scale_attr": "G_scale",
            "ylabel": r"Growth [days$^{-1}$]",
            "base_color": "#2e7d32",
            "best_color": "#2e7d32",
        },
    }

    return defaults.get(quantity_name, defaults["diffusion"])


def _default_y_lims() -> Sequence[Tuple[float, float]]:
    return [(0.0, 0.1), (0.01, 0.03)]


def _normalize_y_lims(
    y_lim: Optional[Tuple[float, float]],
    y_lims: Optional[Sequence[Tuple[float, float]]],
) -> Sequence[Tuple[float, float]]:
    if y_lims is not None:
        return [tuple(lim) for lim in y_lims]

    if y_lim is not None:
        return [tuple(y_lim)]

    return list(_default_y_lims())


def _format_limit_value(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _y_lim_suffix(y_lim: Tuple[float, float]) -> str:
    return f"ylim_{_format_limit_value(y_lim[0])}_{_format_limit_value(y_lim[1])}"


def _truth_attr_for_quantity(quantity_name: str) -> str:
    return "G_true" if quantity_name == "growth" else "D_true"


def _truth_label_for_quantity(quantity_name: str) -> str:
    return r"$G_{\mathrm{true}}$" if quantity_name == "growth" else r"$D_{\mathrm{true}}$"


def _get_truth_values(wrapper: Any, quantity_name: str, target_len: int) -> Optional[np.ndarray]:
    model = wrapper.model
    truth_attr = _truth_attr_for_quantity(quantity_name)

    if hasattr(model, truth_attr):
        truth = _as_numpy(getattr(model, truth_attr))
    elif quantity_name == "growth" and hasattr(model, "G_true_torch"):
        truth = _as_numpy(model.G_true_torch)
    elif quantity_name != "growth" and hasattr(model, "D_true_torch"):
        truth = _as_numpy(model.D_true_torch)
    else:
        return None

    if truth.size < target_len:
        out = np.full(target_len, np.nan, dtype=float)
        out[: truth.size] = truth
        return out

    if truth.size > target_len:
        return truth[:target_len]

    return truth


def _best_pred_attr_for_quantity(quantity_name: str) -> str:
    return "best_growth_pred" if quantity_name == "growth" else "best_diffusion_pred"


def _get_best_prediction_values(
    wrapper: Any,
    quantity_name: str,
    target_len: int,
) -> Optional[np.ndarray]:
    attr_name = _best_pred_attr_for_quantity(quantity_name)
    if not hasattr(wrapper, attr_name):
        return None

    best_pred = _as_numpy(getattr(wrapper, attr_name))
    if best_pred.size < target_len:
        out = np.full(target_len, np.nan, dtype=float)
        out[: best_pred.size] = best_pred
        return out
    if best_pred.size > target_len:
        return best_pred[:target_len]
    return best_pred


def _normalize_errs(errs: Any) -> Optional[Tuple[float, float]]:
    if errs is None:
        return None

    if isinstance(errs, (int, float)):
        return float(errs), float(errs)

    if len(errs) == 1:
        err = float(errs[0])
        return err, err

    return float(errs[0]), float(errs[1])


def _merge_epoch_plot_settings(
    plot_settings: Optional[Dict[str, Any]],
    quantity_name: str,
) -> Dict[str, Any]:
    quantity = _quantity_defaults(quantity_name)
    defaults = {
        "quantity_name": quantity_name,
        "pred_attr": quantity["pred_attr"],
        "model_eval_attr": quantity["model_eval_attr"],
        "model_scale_attr": quantity["model_scale_attr"],
        "ylabel": quantity["ylabel"],
        "base_color": quantity["base_color"],
        "best_color": quantity["best_color"],
        "dataobj": None,
        "species_label": "red",
        "save_dic": {},
        "save_name": f"{quantity_name}_across_epochs.png",
        "base_dir": "plots",
        "name": None,
        "labels": None,
        "num_bins": 50,
        "K": 1.0,
        "overwrite": None,
        "legend": {
            "loc": (0.5, 0.5),
            "fontsize": 10,
            "ncols": 1,
            "framealpha": 0.8,
        },
        "legends": None,
        "figsize": (7, 5),
        "x_lim": None,
        "y_lim": None,
        "y_lims": None,
        "ylim": None,
        "ylims": None,
        "plot_truth": True,
        "truth_label": None,
        "truth_linewidth": 1,
        "errs": None,
        "mpe": None,
        "mpe_linewidth": 1,
        "shade_outside_central_90": True,
        "shade_central_90": None,
        "shade_color": "0.9",
        "shade_alpha": 0.5,
        "crop_to_central_90": False,
        "axis_labels": True,
        "fontsizes": {
            "xaxis": 15.0,
            "xtick_labels": 15.0,
            "yaxis": 15.0,
            "ytick_labels": 15.0,
        },
        "axis_fontsizes": None,
        "tick_params": {
            "major_tick_length": 4.0,
            "major_tick_width": 1.0,
            "minor_tick_length": 2.5,
            "minor_tick_width": 0.8,
        },
        "epoch_step": 1,
        "continue_after_best": False,
        "post_best_only": False,
        "dpi": 100,
        "epochs_sf": None,
        "epoch_sf": None,
        "save_dir2": None,
        "model_index": 0,
        "seed_index": 0,
        "restrict_to_central_90": True,
    }

    settings = {**defaults, **(plot_settings or {})}
    settings["legend"] = {**defaults["legend"], **settings.get("legend", {})}
    settings["legends"] = settings.get(
        "legends",
        settings.get("legend_by_plot", None),
    )
    if settings.get("axis_fontsizes") is not None:
        settings["fontsizes"] = {
            **defaults["fontsizes"],
            **settings["axis_fontsizes"],
        }
    settings["fontsizes"] = {**defaults["fontsizes"], **settings.get("fontsizes", {})}
    settings["tick_params"] = {
        **defaults["tick_params"],
        **settings.get("tick_params", {}),
    }
    settings["dataobj"] = settings.get("dataobj", settings.get("data_obj"))

    if settings.get("ylim") is not None and settings.get("y_lim") is None:
        settings["y_lim"] = settings["ylim"]

    if settings.get("ylims") is not None and settings.get("y_lims") is None:
        settings["y_lims"] = settings["ylims"]

    if settings.get("mpe") is not None and settings.get("errs") is None:
        settings["errs"] = settings["mpe"]

    if (
        settings.get("epoch_sf") is not None
        and settings.get("epochs_sf") is None
    ):
        settings["epochs_sf"] = settings["epoch_sf"]

    if (
        settings.get("shade_central_90") is not None
        and "shade_outside_central_90" not in (plot_settings or {})
    ):
        settings["shade_outside_central_90"] = settings["shade_central_90"]

    # Allow quantity-specific aliases for the truth/reference curve label.
    if settings.get("truth_label") is None:
        if quantity_name == "growth" and settings.get("G_true_label") is not None:
            settings["truth_label"] = settings["G_true_label"]
        elif quantity_name != "growth" and settings.get("D_true_label") is not None:
            settings["truth_label"] = settings["D_true_label"]

    return settings


def _legend_for_plot(
    base_legend: Dict[str, Any],
    legends: Any,
    plot_index: int,
    y_lim: Optional[Tuple[float, float]],
) -> Dict[str, Any]:
    if legends is None:
        return dict(base_legend)

    override = None
    if isinstance(legends, (list, tuple)):
        if plot_index < len(legends):
            override = legends[plot_index]
    elif isinstance(legends, dict):
        y_lim_tuple = tuple(y_lim) if y_lim is not None else None
        y_suffix = _y_lim_suffix(y_lim_tuple) if y_lim_tuple is not None else None
        for key in (plot_index, str(plot_index), y_lim_tuple, y_suffix):
            if key in legends:
                override = legends[key]
                break

    if override is None:
        return dict(base_legend)

    return {**base_legend, **override}


def _groups_from_models_dics_list(
    models_dics_list: Optional[Sequence[Dict[Any, Any]]],
    labels: Optional[Sequence[str]],
    plot_settings: Optional[Dict[str, Any]],
) -> Optional[Dict[Any, Dict[Any, Any]]]:
    if models_dics_list is None:
        return None

    if labels is None:
        es_entries = (plot_settings or {}).get("es_entries", [])
        labels = [entry[0] for entry in es_entries[: len(models_dics_list)]]

    if labels is None or len(labels) < len(models_dics_list):
        labels = [f"group_{i + 1}" for i in range(len(models_dics_list))]

    return {
        labels[i]: models_dics
        for i, models_dics in enumerate(models_dics_list)
    }


def _color_from_plot_params(
    model_wrapper_groups: Dict[Any, Dict[Any, Any]],
    plot_params: Optional[Dict[Any, Sequence[Any]]],
    model_index: int,
) -> Optional[str]:
    if not plot_params or not model_wrapper_groups:
        return None

    first_group = next(iter(model_wrapper_groups.values()))
    model_keys = sorted(first_group.keys())

    if model_index < 0 or model_index >= len(model_keys):
        return None

    style = plot_params.get(model_keys[model_index])
    if not style:
        return None

    return style[0]


def _plot_quantity_across_epochs(
    models_dics_list: Optional[Sequence[Dict[Any, Any]]] = None,
    plot_params: Optional[Dict[Any, Sequence[Any]]] = None,
    plot_settings: Optional[Dict[str, Any]] = None,
    *,
    quantity_name: str = "diffusion",
    pred_attr: Optional[str] = None,
    model_eval_attr: Optional[str] = None,
    model_scale_attr: Optional[str] = None,
    ylabel: Optional[str] = None,
    base_color: Optional[str] = None,
    best_color: Optional[str] = None,
    model_wrapper_groups: Optional[Dict[Any, Dict[Any, Any]]] = None,
    dataobj: Any = None,
    species_label: str = "red",
    save_dic: Optional[Dict[str, Any]] = None,
    save_name: str = "across_epochs.png",
    name: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    num_bins: int = 50,
    K: float = 1.0,
    base_dir: str = "plots",
    overwrite: Optional[bool] = None,
    legend_pos: Any = (0.5, 0.5),
    legend_ncols: int = 1,
    legend_fontsize: int = 10,
    legend_framealpha: float = 0.8,
    figsize: Tuple[float, float] = (7, 5),
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    y_lims: Optional[Sequence[Tuple[float, float]]] = None,
    plot_truth: bool = True,
    truth_label: Optional[str] = None,
    truth_linewidth: float = 1.0,
    errs: Any = None,
    mpe_linewidth: float = 1.0,
    shade_outside_central_90: bool = True,
    shade_color: str = "0.9",
    shade_alpha: float = 0.5,
    crop_to_central_90: bool = False,
    axis_labels: bool = True,
    xlabel_fontsize: float = 11.0,
    ylabel_fontsize: float = 11.0,
    xtick_labelsize: float = 10.0,
    ytick_labelsize: float = 10.0,
    major_tick_length: float = 4.0,
    major_tick_width: float = 1.0,
    minor_tick_length: float = 2.5,
    minor_tick_width: float = 0.8,
    epoch_step: int = 1,
    continue_after_best: bool = False,
    post_best_only: bool = False,
    dpi: int = 100,
    epochs_sf: Optional[int] = None,
    save_dir2: Optional[str] = None,
    model_index: int = 0,
    seed_index: int = 0,
    restrict_to_central_90: bool = True,
) -> None:
    legends = None
    if plot_settings is not None:
        settings = _merge_epoch_plot_settings(plot_settings, quantity_name)
        quantity_name = settings["quantity_name"]
        pred_attr = pred_attr or settings["pred_attr"]
        model_eval_attr = model_eval_attr or settings["model_eval_attr"]
        model_scale_attr = model_scale_attr or settings["model_scale_attr"]
        ylabel = ylabel or settings["ylabel"]
        base_color = base_color or settings["base_color"]
        best_color = best_color or settings["best_color"]
        dataobj = dataobj if dataobj is not None else settings["dataobj"]
        species_label = settings["species_label"]
        save_dic = settings["save_dic"]
        save_name = settings["save_name"]
        name = settings["name"]
        labels = labels or settings["labels"]
        num_bins = settings["num_bins"]
        K = settings["K"]
        base_dir = settings["base_dir"]
        overwrite = settings["overwrite"] if overwrite is None else overwrite
        legend_pos = settings["legend"]["loc"]
        legend_ncols = settings["legend"]["ncols"]
        legend_fontsize = settings["legend"]["fontsize"]
        legend_framealpha = settings["legend"]["framealpha"]
        legends = settings["legends"]
        figsize = settings["figsize"]
        x_lim = settings["x_lim"]
        y_lim = settings["y_lim"]
        y_lims = settings["y_lims"]
        plot_truth = settings["plot_truth"]
        truth_label = settings["truth_label"]
        truth_linewidth = settings["truth_linewidth"]
        errs = settings["errs"]
        mpe_linewidth = settings["mpe_linewidth"]
        shade_outside_central_90 = settings["shade_outside_central_90"]
        shade_color = settings["shade_color"]
        shade_alpha = settings["shade_alpha"]
        crop_to_central_90 = settings["crop_to_central_90"]
        axis_labels = settings["axis_labels"]
        xlabel_fontsize = settings["fontsizes"]["xaxis"]
        ylabel_fontsize = settings["fontsizes"]["yaxis"]
        xtick_labelsize = settings["fontsizes"]["xtick_labels"]
        ytick_labelsize = settings["fontsizes"]["ytick_labels"]
        major_tick_length = settings["tick_params"]["major_tick_length"]
        major_tick_width = settings["tick_params"]["major_tick_width"]
        minor_tick_length = settings["tick_params"]["minor_tick_length"]
        minor_tick_width = settings["tick_params"]["minor_tick_width"]
        epoch_step = settings["epoch_step"]
        continue_after_best = settings["continue_after_best"]
        post_best_only = settings["post_best_only"]
        dpi = settings["dpi"]
        epochs_sf = settings["epochs_sf"]
        save_dir2 = settings["save_dir2"]
        if "model_index" in plot_settings:
            model_index = settings["model_index"]
        if "seed_index" in plot_settings:
            seed_index = settings["seed_index"]
        restrict_to_central_90 = settings["restrict_to_central_90"]

    quantity = _quantity_defaults(quantity_name)
    pred_attr = pred_attr or quantity["pred_attr"]
    model_eval_attr = model_eval_attr or quantity["model_eval_attr"]
    model_scale_attr = model_scale_attr or quantity["model_scale_attr"]
    ylabel = ylabel or quantity["ylabel"]
    base_color = base_color or quantity["base_color"]
    best_color = best_color or quantity["best_color"]
    save_dic = save_dic or {}
    overwrite = False if overwrite is None else overwrite

    if model_wrapper_groups is None:
        model_wrapper_groups = _groups_from_models_dics_list(
            models_dics_list=models_dics_list,
            labels=labels,
            plot_settings=plot_settings,
        )

    if model_wrapper_groups is None:
        raise ValueError("Pass either model_wrapper_groups or models_dics_list.")

    plot_param_color = _color_from_plot_params(
        model_wrapper_groups=model_wrapper_groups,
        plot_params=plot_params,
        model_index=model_index,
    )
    if plot_param_color is not None and not (plot_settings or {}).get("base_color"):
        base_color = plot_param_color

    out_base = name or _build_save_base(save_dic, save_name, base_dir=base_dir)
    out_root, _ = os.path.splitext(out_base)

    outer_keys = list(model_wrapper_groups.keys())

    if not outer_keys:
        raise ValueError("No groups in model_wrapper_groups.")

    if labels is None:
        labels = [str(k) for k in outer_keys]

    selected_wrappers = _select_one_wrapper_per_group(
        model_wrapper_groups=model_wrapper_groups,
        model_index=model_index,
        seed_index=seed_index,
    )
    epochs_sf = _resolve_epochs_sf(
        selected_wrappers=selected_wrappers,
        pred_attr=pred_attr,
        epochs_sf=epochs_sf,
    )

    central_bounds_needed = (
        restrict_to_central_90 or shade_outside_central_90 or crop_to_central_90
    )
    central_bounds = None
    if central_bounds_needed:
        central_bounds = _central_u_bounds(
            dataobj=dataobj,
            selected_wrappers=selected_wrappers,
            num_bins=num_bins,
        )

    if central_bounds_needed and central_bounds is None:
        restrict_to_central_90 = False
        shade_outside_central_90 = False
        crop_to_central_90 = False
        low_u = -np.inf
        high_u = np.inf
    elif central_bounds is not None:
        low_u, high_u = central_bounds
    else:
        low_u = -np.inf
        high_u = np.inf

    x_vals_by_group = {}
    sf_by_group = {}

    for group_key, wrapper in selected_wrappers.items():
        u_vals_np = np.asarray(wrapper.model.u_vals).flatten()
        sf = _get_density_scale(wrapper, species_label)
        sf_by_group[group_key] = sf
        x_vals_by_group[group_key] = u_vals_np * sf * K

    global_ylim = _global_y_limits(
        selected_wrappers=selected_wrappers,
        x_vals_by_group=x_vals_by_group,
        pred_attr=pred_attr,
        epoch_step=epoch_step,
        epochs_sf=epochs_sf,
        continue_after_best=continue_after_best,
        post_best_only=post_best_only,
    )
    resolved_y_lims = _normalize_y_lims(y_lim=y_lim, y_lims=y_lims)
    errs = _normalize_errs(errs)
    truth_label = truth_label or _truth_label_for_quantity(quantity_name)

    for group_key, label in zip(outer_keys, labels):
        wrapper = selected_wrappers[group_key]
        xg = x_vals_by_group[group_key]
        sf = sf_by_group[group_key]

        plotted_epochs, best_saved_epoch = _build_epochs_to_plot(
            wrapper=wrapper,
            pred_attr=pred_attr,
            epoch_step=epoch_step,
            epochs_sf=epochs_sf,
            continue_after_best=continue_after_best,
            post_best_only=post_best_only,
        )

        best_raw_epoch = best_saved_epoch * epochs_sf

        base_legend = {
            "loc": legend_pos,
            "ncols": legend_ncols,
            "fontsize": legend_fontsize,
            "framealpha": legend_framealpha,
        }

        for y_lim_index, current_y_lim in enumerate(resolved_y_lims):
            legend_cfg = _legend_for_plot(
                base_legend=base_legend,
                legends=legends,
                plot_index=y_lim_index,
                y_lim=current_y_lim,
            )
            y_suffix = (
                f"_{_y_lim_suffix(current_y_lim)}"
                if len(resolved_y_lims) > 1
                else ""
            )
            out_png = (
                f"{out_root}_{quantity_name}_{group_key}_model{model_index}"
                f"{y_suffix}.png"
            )

            if (not overwrite) and os.path.isfile(out_png):
                print(
                    f"{quantity_name.capitalize()} file exists and overwrite=False; "
                    f"skipping: {out_png}"
                )
                continue
            if overwrite and os.path.isfile(out_png):
                print(f"Overwriting existing {quantity_name} plot: {out_png}")

            fig, ax = plt.subplots(figsize=figsize)

            epoch_colors = _epoch_color_map(
                plotted_epochs=plotted_epochs,
                base_color=base_color,
                best_color=best_color,
                best_saved_epoch=best_saved_epoch,
            )
            color_fn = lambda ep: epoch_colors[ep]

            low_x = low_u * sf * K
            high_x = high_u * sf * K
            x_min = float(np.nanmin(xg))
            x_max = float(np.nanmax(xg))

            if shade_outside_central_90:
                ax.axvspan(
                    x_min,
                    low_x,
                    facecolor=shade_color,
                    alpha=shade_alpha,
                    zorder=0,
                )
                ax.axvspan(
                    high_x,
                    x_max,
                    facecolor=shade_color,
                    alpha=shade_alpha,
                    zorder=0,
                )

            for ep in plotted_epochs:
                arr = _get_prediction(
                    wrapper=wrapper,
                    pred_attr=pred_attr,
                    ep=ep,
                    target_len=len(xg),
                )

                x_plot, y_plot = _restrict_to_central_range(
                    x=xg,
                    y=arr,
                    low_x=low_x,
                    high_x=high_x,
                    restrict=crop_to_central_90,
                )

                ax.plot(
                    x_plot,
                    y_plot,
                    lw=3,
                    color=color_fn(ep),
                    zorder=2,
                )

            best_model_handle = None
            truth_handle = None
            mpe_handle = None
            sample_model = wrapper.model
            u_vals_torch = getattr(sample_model, "u_vals_torch", None)

            best_eval_np = _get_best_prediction_values(
                wrapper=wrapper,
                quantity_name=quantity_name,
                target_len=len(xg),
            )
            if best_eval_np is None and u_vals_torch is not None and hasattr(sample_model, model_eval_attr):
                sample_model.eval()

                with torch.no_grad():
                    raw_eval = getattr(
                        sample_model, model_eval_attr
                    )(u_vals_torch).flatten()

                    if hasattr(sample_model, model_scale_attr):
                        raw_eval = getattr(sample_model, model_scale_attr) * raw_eval

                best_eval_np = _as_numpy(raw_eval)

            if best_eval_np is not None:
                x_best, y_best = _restrict_to_central_range(
                    x=xg,
                    y=best_eval_np,
                    low_x=low_x,
                    high_x=high_x,
                    restrict=crop_to_central_90,
                )

                ax.plot(
                    x_best,
                    y_best,
                    lw=3,
                    ls="-",
                    color="k",
                    zorder=5,
                    label="_nolegend_",
                )

                best_model_handle = Line2D(
                    [0],
                    [0],
                    lw=3,
                    ls="-",
                    color="k",
                    label=f"best model ({best_raw_epoch})",
                )

            truth_values = _get_truth_values(
                wrapper=wrapper,
                quantity_name=quantity_name,
                target_len=len(xg),
            )
            if plot_truth and truth_values is not None:
                x_truth, y_truth = _restrict_to_central_range(
                    x=xg,
                    y=truth_values,
                    low_x=low_x,
                    high_x=high_x,
                    restrict=crop_to_central_90,
                )
                ax.plot(
                    x_truth,
                    y_truth,
                    "--",
                    lw=truth_linewidth,
                    color="k",
                    zorder=6,
                    label="_nolegend_",
                )
                truth_handle = Line2D(
                    [0],
                    [0],
                    lw=truth_linewidth,
                    ls="--",
                    color="k",
                    label=truth_label,
                )

                if errs is not None:
                    err_up, err_low = errs
                    ax.plot(
                        x_truth,
                        y_truth * (1 + err_up / 100),
                        "--",
                        lw=mpe_linewidth,
                        color="r",
                        zorder=4,
                        label="_nolegend_",
                    )
                    ax.plot(
                        x_truth,
                        y_truth * (1 - err_low / 100),
                        "--",
                        lw=mpe_linewidth,
                        color="r",
                        zorder=4,
                        label="_nolegend_",
                    )
                    mpe_handle = Line2D(
                        [0],
                        [0],
                        lw=mpe_linewidth,
                        ls="--",
                        color="r",
                        label=f"{err_up:g}% MPE",
                    )

            ax.set_facecolor("white")

            if x_lim is not None:
                ax.set_xlim(x_lim)
            elif crop_to_central_90:
                ax.set_xlim((low_x, high_x))

            ax.set_ylim(current_y_lim if current_y_lim is not None else global_ylim)

            if axis_labels:
                ax.set_xlabel(
                    r"Cell density [cells mm$^{-2}$]",
                    fontsize=xlabel_fontsize,
                )
                ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

            _style_axes(
                ax,
                xtick_labelsize=xtick_labelsize,
                ytick_labelsize=ytick_labelsize,
                major_tick_length=major_tick_length,
                major_tick_width=major_tick_width,
                minor_tick_length=minor_tick_length,
                minor_tick_width=minor_tick_width,
            )

            legend_handles = _build_legend_handles(
                plotted_epochs=plotted_epochs,
                epochs_sf=epochs_sf,
                color_fn=color_fn,
                best_model_handle=best_model_handle,
            )
            if truth_handle is not None:
                legend_handles.append(truth_handle)
            if mpe_handle is not None:
                legend_handles.append(mpe_handle)

            if legend_handles:
                legend = ax.legend(
                    handles=legend_handles,
                    fontsize=legend_cfg["fontsize"],
                    ncol=legend_cfg["ncols"],
                    frameon=True,
                    facecolor="white",
                    framealpha=legend_cfg["framealpha"],
                    loc=legend_cfg["loc"],
                    title_fontsize=legend_cfg["fontsize"] + 1,
                    title="Epoch ",
                )
                legend.set_zorder(1e6)

            ax.grid(False)
            fig.tight_layout()

            out_dir = os.path.dirname(out_png)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
            plt.show()
            plt.close(fig)

            print(
                f"Saved {quantity_name} plot for {group_key}, "
                f"model_index={model_index}, y_lim={current_y_lim}: {out_png}"
            )

            if save_dir2:
                _save_secondary_copy(out_png, save_dir2, figsize, dpi)
                print(
                    f"Also saved second copy of {quantity_name} plot for {group_key}"
                )


def plot_diffusion_across_epochs(
    model_wrapper_groups: Optional[Dict[Any, Dict[Any, Any]]] = None,
    dataobj: Any = None,
    species_label: str = "red",
    save_dic: Optional[Dict[str, Any]] = None,
    save_name: str = "diffusion_across_epochs.png",
    labels: Optional[Sequence[str]] = None,
    models_dics_list: Optional[Sequence[Dict[Any, Any]]] = None,
    plot_params: Optional[Dict[Any, Sequence[Any]]] = None,
    plot_settings: Optional[Dict[str, Any]] = None,
    num_bins: int = 50,
    K: float = 1.0,
    base_dir: str = "plots",
    overwrite: Optional[bool] = None,
    legend_pos: Any = (0.5, 0.5),
    legend_ncols: int = 1,
    legend_fontsize: int = 10,
    figsize: Tuple[float, float] = (7, 5),
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    y_lims: Optional[Sequence[Tuple[float, float]]] = None,
    plot_truth: bool = True,
    truth_label: Optional[str] = None,
    truth_linewidth: float = 1.0,
    errs: Any = None,
    mpe_linewidth: float = 1.0,
    shade_outside_central_90: bool = True,
    shade_color: str = "0.9",
    shade_alpha: float = 0.5,
    crop_to_central_90: bool = False,
    axis_labels: bool = True,
    xlabel_fontsize: float = 11.0,
    ylabel_fontsize: float = 11.0,
    xtick_labelsize: float = 10.0,
    ytick_labelsize: float = 10.0,
    major_tick_length: float = 4.0,
    major_tick_width: float = 1.0,
    minor_tick_length: float = 2.5,
    minor_tick_width: float = 0.8,
    epoch_step: int = 1,
    continue_after_best: bool = False,
    post_best_only: bool = False,
    dpi: int = 100,
    epochs_sf: Optional[int] = None,
    save_dir2: Optional[str] = None,
    model_index: int = 0,
    seed_index: int = 0,
    restrict_to_central_90: bool = True,
    diffusion_base_color: Optional[str] = None,
    diffusion_best_color: str = "#d35400",
) -> None:
    """
    Plot diffusion(u) across saved epochs.

    The best epoch is inferred from wrapper.last_improved when available.
    By default, predictions beyond that epoch are not plotted. When
    continue_after_best=True, later saved predictions are included and fade
    lighter as epochs move away from the best model. When
    post_best_only=True, only the best epoch and later saved predictions are
    shown.
    """
    _plot_quantity_across_epochs(
        models_dics_list=models_dics_list,
        plot_params=plot_params,
        plot_settings=plot_settings,
        quantity_name="diffusion",
        pred_attr="diffusion_preds",
        model_eval_attr="diffusion",
        model_scale_attr="D_scale",
        ylabel=r"Diffusion [mm$^2$ days$^{-1}$]",
        base_color=diffusion_base_color,
        best_color=diffusion_best_color,
        model_wrapper_groups=model_wrapper_groups,
        dataobj=dataobj,
        species_label=species_label,
        save_dic=save_dic or {},
        save_name=save_name,
        labels=labels,
        num_bins=num_bins,
        K=K,
        base_dir=base_dir,
        overwrite=overwrite,
        legend_pos=legend_pos,
        legend_ncols=legend_ncols,
        legend_fontsize=legend_fontsize,
        figsize=figsize,
        x_lim=x_lim,
        y_lim=y_lim,
        y_lims=y_lims,
        plot_truth=plot_truth,
        truth_label=truth_label,
        truth_linewidth=truth_linewidth,
        errs=errs,
        mpe_linewidth=mpe_linewidth,
        shade_outside_central_90=shade_outside_central_90,
        shade_color=shade_color,
        shade_alpha=shade_alpha,
        crop_to_central_90=crop_to_central_90,
        axis_labels=axis_labels,
        xlabel_fontsize=xlabel_fontsize,
        ylabel_fontsize=ylabel_fontsize,
        xtick_labelsize=xtick_labelsize,
        ytick_labelsize=ytick_labelsize,
        major_tick_length=major_tick_length,
        major_tick_width=major_tick_width,
        minor_tick_length=minor_tick_length,
        minor_tick_width=minor_tick_width,
        epoch_step=epoch_step,
        continue_after_best=continue_after_best,
        post_best_only=post_best_only,
        dpi=dpi,
        epochs_sf=epochs_sf,
        save_dir2=save_dir2,
        model_index=model_index,
        seed_index=seed_index,
        restrict_to_central_90=restrict_to_central_90,
    )


def plot_growth_across_epochs(
    model_wrapper_groups: Optional[Dict[Any, Dict[Any, Any]]] = None,
    dataobj: Any = None,
    species_label: str = "red",
    save_dic: Optional[Dict[str, Any]] = None,
    save_name: str = "growth_across_epochs.png",
    labels: Optional[Sequence[str]] = None,
    models_dics_list: Optional[Sequence[Dict[Any, Any]]] = None,
    plot_params: Optional[Dict[Any, Sequence[Any]]] = None,
    plot_settings: Optional[Dict[str, Any]] = None,
    num_bins: int = 50,
    K: float = 1.0,
    base_dir: str = "plots",
    overwrite: Optional[bool] = None,
    legend_pos: Any = (0.5, 0.5),
    legend_ncols: int = 1,
    legend_fontsize: int = 10,
    figsize: Tuple[float, float] = (7, 5),
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    y_lims: Optional[Sequence[Tuple[float, float]]] = None,
    plot_truth: bool = True,
    truth_label: Optional[str] = None,
    truth_linewidth: float = 1.0,
    errs: Any = None,
    mpe_linewidth: float = 1.0,
    shade_outside_central_90: bool = True,
    shade_color: str = "0.9",
    shade_alpha: float = 0.5,
    crop_to_central_90: bool = False,
    axis_labels: bool = True,
    xlabel_fontsize: float = 11.0,
    ylabel_fontsize: float = 11.0,
    xtick_labelsize: float = 10.0,
    ytick_labelsize: float = 10.0,
    major_tick_length: float = 4.0,
    major_tick_width: float = 1.0,
    minor_tick_length: float = 2.5,
    minor_tick_width: float = 0.8,
    epoch_step: int = 1,
    continue_after_best: bool = False,
    post_best_only: bool = False,
    dpi: int = 100,
    epochs_sf: Optional[int] = None,
    save_dir2: Optional[str] = None,
    model_index: int = 0,
    seed_index: int = 0,
    restrict_to_central_90: bool = True,
    growth_base_color: Optional[str] = None,
    growth_best_color: str = "#2e7d32",
) -> None:
    """
    Plot growth(u) across saved epochs.

    The best epoch is inferred from wrapper.last_improved when available.
    By default, predictions beyond that epoch are not plotted. When
    continue_after_best=True, later saved predictions are included and fade
    lighter as epochs move away from the best model. When
    post_best_only=True, only the best epoch and later saved predictions are
    shown.
    """
    _plot_quantity_across_epochs(
        models_dics_list=models_dics_list,
        plot_params=plot_params,
        plot_settings=plot_settings,
        quantity_name="growth",
        pred_attr="growth_preds",
        model_eval_attr="growth",
        model_scale_attr="G_scale",
        ylabel=r"Growth [days$^{-1}$]",
        base_color=growth_base_color,
        best_color=growth_best_color,
        model_wrapper_groups=model_wrapper_groups,
        dataobj=dataobj,
        species_label=species_label,
        save_dic=save_dic or {},
        save_name=save_name,
        labels=labels,
        num_bins=num_bins,
        K=K,
        base_dir=base_dir,
        overwrite=overwrite,
        legend_pos=legend_pos,
        legend_ncols=legend_ncols,
        legend_fontsize=legend_fontsize,
        figsize=figsize,
        x_lim=x_lim,
        y_lim=y_lim,
        y_lims=y_lims,
        plot_truth=plot_truth,
        truth_label=truth_label,
        truth_linewidth=truth_linewidth,
        errs=errs,
        mpe_linewidth=mpe_linewidth,
        shade_outside_central_90=shade_outside_central_90,
        shade_color=shade_color,
        shade_alpha=shade_alpha,
        crop_to_central_90=crop_to_central_90,
        axis_labels=axis_labels,
        xlabel_fontsize=xlabel_fontsize,
        ylabel_fontsize=ylabel_fontsize,
        xtick_labelsize=xtick_labelsize,
        ytick_labelsize=ytick_labelsize,
        major_tick_length=major_tick_length,
        major_tick_width=major_tick_width,
        minor_tick_length=minor_tick_length,
        minor_tick_width=minor_tick_width,
        epoch_step=epoch_step,
        continue_after_best=continue_after_best,
        post_best_only=post_best_only,
        dpi=dpi,
        epochs_sf=epochs_sf,
        save_dir2=save_dir2,
        model_index=model_index,
        seed_index=seed_index,
        restrict_to_central_90=restrict_to_central_90,
    )
