"""Prepare growth MSE summaries from trained model wrappers.

Contents
--------
- _trained_epoch_count
- _grow_prediction_count
- infer_grow_prediction_frequency
- _central_90_mask
- _masked_mse_from_pred
- prepare_grow_run_data"""

import numpy as np
import torch

from .prepare_model_loss import prepare_model_run_data
from .utils import hist_properties


def _trained_epoch_count(model_wrapper):
    model = getattr(model_wrapper, "model", None)
    epochs = getattr(model, "epochs", None)
    if epochs is not None and epochs > 0:
        return int(epochs)

    for attr in ("train_loss_list", "val_loss_list"):
        values = getattr(model_wrapper, attr, None)
        if values is not None and len(values) > 0:
            return len(values)

    return None


def _grow_prediction_count(model_wrapper):
    preds = getattr(model_wrapper, "growth_preds", None)
    if preds is not None and len(preds) > 0:
        return len(preds)

    errors = getattr(model_wrapper, "growth_errors", None)
    if errors is not None and len(errors) > 0:
        return len(errors)

    return None


def infer_grow_prediction_frequency(model_wrappers):
    """Infer saved-growth-prediction cadence from stored arrays and epochs."""
    freqs = []
    for model_wrapper in model_wrappers:
        prediction_count = _grow_prediction_count(model_wrapper)
        epoch_count = _trained_epoch_count(model_wrapper)
        if prediction_count and epoch_count:
            freqs.append(max(1, int(round(epoch_count / prediction_count))))

    if not freqs:
        return 1

    return int(np.median(freqs))


def _central_90_mask(model_wrapper, dataobj=None):
    u_vals = getattr(getattr(model_wrapper, "model", None), "u_vals", None)
    if u_vals is None:
        u_vals = getattr(getattr(model_wrapper, "model", None), "u_vals_torch", None)
    if u_vals is None:
        return None

    u_vals = np.asarray(u_vals, dtype=float).reshape(-1)
    if u_vals.size == 0:
        return None

    if dataobj is not None:
        h_props = hist_properties(dataobj)
        low_u = float(h_props["low_count"])
        high_u = float(h_props["high_count"])
    else:
        low_u, high_u = np.percentile(u_vals, [5, 95])
    mask = (u_vals >= low_u) & (u_vals <= high_u)
    return mask if np.any(mask) else None


def _masked_mse_from_pred(true_vals, pred_vals, mask):
    true_arr = np.asarray(true_vals, dtype=float).reshape(-1)
    pred_arr = np.asarray(pred_vals, dtype=float).reshape(-1)
    if true_arr.shape != pred_arr.shape:
        raise ValueError(
            f"True/pred shapes do not match for central-90 MSE: {true_arr.shape} vs {pred_arr.shape}."
        )
    if mask is None or mask.shape[0] != true_arr.shape[0]:
        return float(np.mean((true_arr - pred_arr) ** 2))
    return float(np.mean((true_arr[mask] - pred_arr[mask]) ** 2))


def prepare_grow_run_data(
    models_dics,
    plot_params=None,
    best_epoch_termination=True,
    val_loss_best=True,
    restrict_to_central_90=False,
    dataobj=None,
):
    import numpy as np
    try:
        import torch
    except ImportError:
        torch = None

    if plot_params is None:
        plot_params = {}

    run_data = []
    model_keys = list(models_dics.keys())
    dic_values = list(models_dics.values())

    if val_loss_best:
        total_stats,_ = prepare_model_run_data(models_dics, plot_params=None, best_epoch_termination=True)

    for i, (key, model_dic) in enumerate(zip(model_keys, dic_values)):
        model_wrappers = list(model_dic.values())
        sf = infer_grow_prediction_frequency(model_wrappers)

        # Use user-defined or default styling
        color, linestyle, markerstyle, label = plot_params.get(
            key, [f"C{i}", "-", ".", f"Run {i+1}"]
        )

        best_epochs_lst = []
        grow_errs = []

        # ---------- First pass: compute best epochs and max length ----------
        max_len = 0
        for m in model_wrappers:
            # Epoch index where validation loss is best
            val_losses = np.array(m.val_loss_list)
            best_val_epoch = np.argmin(np.abs(val_losses - m.best_val_loss))
            best_epochs_lst.append(best_val_epoch)

            # Use full length of recorded growth errors for this model
            series_len = len(m.growth_errors)
            max_len = max(max_len, series_len)

        # ---------- Second pass: build padded growth error arrays ----------
        for m in model_wrappers:
            series_vals = []
            mask = _central_90_mask(m, dataobj=dataobj) if restrict_to_central_90 else None

            if restrict_to_central_90 and getattr(m, "growth_preds", None):
                true_vals = getattr(m.model, "G_true_torch", None)
                if true_vals is None:
                    raise AttributeError(
                        "restrict_to_central_90=True requires model.G_true_torch to be available."
                    )
                true_vals = true_vals.detach().cpu().numpy() if torch is not None and isinstance(true_vals, torch.Tensor) else true_vals
                for grow_pred in m.growth_preds:
                    pred_vals = grow_pred.detach().cpu().numpy() if torch is not None and isinstance(grow_pred, torch.Tensor) else grow_pred
                    series_vals.append(_masked_mse_from_pred(true_vals, pred_vals, mask))
            else:
                for grow_err in m.growth_errors:
                    try:
                        if torch is not None and isinstance(grow_err, torch.Tensor):
                            val = torch.mean(grow_err[:, 0]).detach().cpu().numpy()
                        else:
                            arr = np.asarray(grow_err)
                            if arr.ndim > 1:
                                val = np.mean(arr[:, 0])
                            else:
                                val = np.mean(arr)
                    except Exception:
                        arr = np.asarray(grow_err)
                        if arr.ndim > 1:
                            val = np.mean(arr[:, 0])
                        else:
                            val = np.mean(arr)

                    series_vals.append(val)

            padded = np.full(max_len, np.nan, dtype=float)
            series_len = len(series_vals)
            padded[:series_len] = series_vals
            grow_errs.append(padded)

        grow_errs_arr = np.vstack(grow_errs)

        # ---------- Running mins across time ----------
        running_mins = np.minimum.accumulate(
            np.nan_to_num(grow_errs_arr, nan=np.inf),
            axis=1,
        )
        running_min_min = np.nanmin(running_mins, axis=0)
        running_min_max = np.nanmax(running_mins, axis=0)

        if val_loss_best:
            x_orig = total_stats[i]['epochs']
            best_epoch_idx_orig = total_stats[i]['best_epoch_seed_idx'][0]
            best_epoch_idx = min(max_len - 1, best_epoch_idx_orig//sf + 1)
            best_seed_idx = total_stats[i]['best_epoch_seed_idx'][1]
        else:
            best_epoch_idx = np.nanargmin(running_min_min)
            best_seed_idx = np.nanargmin(grow_errs_arr[:, best_epoch_idx]) # CHECK this is correct

        epochs = np.arange(max_len)  # full recorded training length

        mean_vals = np.nanmean(grow_errs_arr, axis=0)
        std_vals = np.nanstd(grow_errs_arr, axis=0)

        # ---------- Final growth MSE stats (matching prepare_growth_mse_data) ----------
        # Each best_growth_error can be an array/tensor; take its mean per run
        if restrict_to_central_90:
            final_mses = []
            for m in model_wrappers:
                best_pred = getattr(m, "best_growth_pred", None)
                true_vals = getattr(m.model, "G_true_torch", None)
                mask = _central_90_mask(m, dataobj=dataobj)
                if best_pred is None or true_vals is None:
                    raise AttributeError(
                        "restrict_to_central_90=True requires best_growth_pred and model.G_true_torch."
                    )
                pred_vals = best_pred.detach().cpu().numpy() if torch is not None and isinstance(best_pred, torch.Tensor) else best_pred
                true_arr = true_vals.detach().cpu().numpy() if torch is not None and isinstance(true_vals, torch.Tensor) else true_vals
                final_mses.append(_masked_mse_from_pred(true_arr, pred_vals, mask))
            final_mses = np.array(final_mses, dtype=np.float64)
        else:
            final_mses = np.array([m.best_growth_error for m in model_wrappers], dtype=np.float64)

        avg_mse = float(np.mean(final_mses))
        std_mse = float(np.std(final_mses))
        min_mse = float(np.min(final_mses))
        max_mse = float(np.max(final_mses))
        
        # For backward-compatibility, keep mean_final/std_final aliases
        mean_final = avg_mse
        std_final = std_mse

        label_with_stats = f"{label}"  # keep label as-is; you can append stats if desired

        min_vals = np.nanmin(grow_errs_arr, axis=0)
        max_vals = np.nanmax(grow_errs_arr, axis=0)

        if best_epoch_termination:
            mean_vals = mean_vals[: best_epoch_idx + 1]
            min_vals = min_vals[: best_epoch_idx + 1]
            max_vals = max_vals[: best_epoch_idx + 1]
            epochs = epochs[: best_epoch_idx + 1]

        run_data.append(
            {
                # --- scalar summary (from prepare_growth_mse_data) ---
                "label": label_with_stats,
                "avg_mse": avg_mse,
                "std_mse": std_mse,
                "min_mse": min_mse,
                "max_mse": max_mse,
                "final_mses": final_mses,
                "color": color,

                # --- aliases/backwards-compatible fields ---
                "mean_final": mean_final,
                "std_final": std_final,

                # --- time-series and running-min info ---
                "epochs": epochs,
                "mean_vals": mean_vals,
                "std_vals": std_vals,
                "min_vals": min_vals,
                "max_vals": max_vals,
                "running_min_min": running_min_min,
                "running_min_max": running_min_max,
                "best_epoch_seed_idx": [best_epoch_idx, best_seed_idx],
                "best_epochs_lst": np.array(best_epochs_lst),
                "best_epoch_idx_orig": best_epoch_idx_orig,
                "x_orig": x_orig,
                "prediction_frequency": sf,

                # --- styling ---
                "linestyle": linestyle,
                "markerstyle": markerstyle,
            }
        )

    return run_data
