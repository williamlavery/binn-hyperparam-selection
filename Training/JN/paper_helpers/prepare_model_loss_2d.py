"""Prepare grouped loss summaries from two-dimensional model wrappers.

Contents
--------
- _unwrap_wrapper_dict
- _extract_model_wrappers
- _collect_val_losses
- _collect_train_losses
- _pad_and_stack_losses
- _compute_running_min_stats
- _compute_best_model_loss_stats
- _collect_best_epochs_and_hticks
- _collect_best_epochs_and_hticks_train
- _select_best_model_train
- _compute_best_epoch_idx
- _compute_timing_stats
- _format_label_with_stats
- prepare_model_run_data"""

import numpy as np

from .prepare_model_loss import (
    _collect_epoch_times,
    _ensure_plot_params,
    _get_style_for_key,
    _select_best_model,
)


def _unwrap_wrapper_dict(model_dic):
    """
    Accept either {seed: wrapper} or {width: {seed: wrapper}} and return the
    innermost wrapper mapping expected by the plotting helpers.
    """
    current = model_dic
    while isinstance(current, dict) and current:
        first_value = next(iter(current.values()))
        if hasattr(first_value, "model") or hasattr(first_value, "train_loss_list"):
            return current
        if len(current) == 1 and isinstance(first_value, dict):
            current = first_value
            continue
        break
    return current


def _extract_model_wrappers(model_dic):
    """Return a flat list of model wrappers from a supported nested mapping."""
    wrapper_dict = _unwrap_wrapper_dict(model_dic)
    if not isinstance(wrapper_dict, dict):
        raise TypeError(
            "Expected a dict of model wrappers or a single nested wrapper dict."
        )

    model_wrappers = list(wrapper_dict.values())
    if not model_wrappers:
        raise ValueError("Model wrapper dictionary is empty.")

    first_wrapper = model_wrappers[0]
    if isinstance(first_wrapper, dict):
        raise TypeError(
            "Model wrapper dictionary is nested deeper than expected."
        )
    return model_wrappers


def _collect_val_losses(model_wrappers):
    """
    Collect validation (or train) loss lists into an array per run.

    Returns
    -------
    list[np.ndarray]
        One array per run (variable length).
    """
    val_losses_raw = []
    for m in model_wrappers:
        if len(m.val_loss_list):
            losses = np.array(m.val_loss_list, dtype=np.float64)
        else:
            losses = np.array(m.train_loss_list, dtype=np.float64)
        val_losses_raw.append(losses)
    return val_losses_raw


def _collect_train_losses(model_wrappers):
    """
    Collect training loss lists into an array per run (no fallback).

    Returns
    -------
    list[np.ndarray]
        One array per run (variable length).
    """
    train_losses_raw = []
    for m in model_wrappers:
        losses = np.array(m.train_loss_list, dtype=np.float64)
        train_losses_raw.append(losses)
    return train_losses_raw


def _pad_and_stack_losses(val_losses_raw):
    """
    Pad each run's loss array to the maximum length and stack into 2D array.

    Padding is done with NaNs, and we return the original lengths so that
    downstream computations can ignore padded regions.

    Returns
    -------
    tuple
        (val_losses, max_len_val, lengths)
        val_losses : np.ndarray, shape (n_runs, max_len_val)
        lengths    : list[int] original length per run
    """
    lengths = [len(v) for v in val_losses_raw]
    max_len_val = max(lengths) if lengths else 0

    if max_len_val == 0:
        # No data at all; return empty arrays
        return (
            np.full((len(val_losses_raw), 0), np.nan, dtype=np.float64),
            0,
            lengths,
        )

    val_losses = np.full((len(val_losses_raw), max_len_val), np.nan, dtype=np.float64)
    for i, (v, L) in enumerate(zip(val_losses_raw, lengths)):
        val_losses[i, :L] = v

    return val_losses, max_len_val, lengths


def _compute_running_min_stats(val_losses, lengths):
    """
    Compute per-run running minima and aggregate per-epoch min/max,
    respecting the true length of each run.

    For epochs beyond a run's length, that run contributes NaN and is ignored
    in the aggregation. Thus, when only one run has data at a given epoch,
    running_min_min == running_min_max → no shaded band there.
    """
    if val_losses.size == 0:
        return np.array([]), np.array([])

    n_runs, max_len_val = val_losses.shape
    running_mins = np.full_like(val_losses, np.nan)

    for j in range(n_runs):
        L = lengths[j]
        if L == 0:
            continue
        v = val_losses[j, :L]
        # running min over valid portion
        rm = np.minimum.accumulate(v)
        running_mins[j, :L] = rm
        # entries beyond L remain NaN

    running_min_min = np.nanmin(running_mins, axis=0)
    running_min_max = np.nanmax(running_mins, axis=0)

    return running_min_min, running_min_max


def _compute_best_model_loss_stats(val_losses):
    """
    Compute mean and std of the final loss for each run
    (last non-NaN value in each row of val_losses).

    Returns
    -------
    tuple
        (mean_final, std_final, last_valid_vals)
    """
    if val_losses.size == 0:
        return np.nan, np.nan, np.array([])

    last_valid_vals = np.array(
        [
            row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan
            for row in val_losses
        ]
    )
    mean_final = np.nanmean(last_valid_vals)
    std_final = np.nanstd(last_valid_vals)
    return mean_final, std_final, last_valid_vals


def _collect_best_epochs_and_hticks(model_wrappers, val_losses_raw):
    """
    For each run, find the epoch index where the best_val_loss occurs,
    and collect best_val_loss values.

    Returns
    -------
    tuple
        (best_epochs_lst, hticks_vals, hticks)
    """
    hticks_vals = []
    best_epochs_lst = []

    for idx, m in enumerate(model_wrappers):
        if len(val_losses_raw[idx]) == 0:
            best_val_epoch = 0
        else:
            best_val_epoch = int(
                np.argmin(np.abs(val_losses_raw[idx] - m.best_val_loss))
            )
        best_epochs_lst.append(best_val_epoch)
        hticks_vals.append(m.best_val_loss)

    hticks = float(np.mean(hticks_vals)) if len(hticks_vals) else np.nan
    return best_epochs_lst, hticks_vals, hticks


def _collect_best_epochs_and_hticks_train(train_losses_raw):
    """
    Training-loss analogue of _collect_best_epochs_and_hticks.

    For each run, find the epoch index where the training loss is minimal,
    and collect those minimal values.

    Returns
    -------
    tuple
        (best_epochs_lst_train, hticks_vals_train, hticks_train)
    """
    hticks_vals = []
    best_epochs_lst = []

    for losses in train_losses_raw:
        if len(losses) == 0:
            best_epoch = 0
            best_val = np.nan
        else:
            best_epoch = int(np.argmin(losses))
            best_val = float(losses[best_epoch])
        best_epochs_lst.append(best_epoch)
        hticks_vals.append(best_val)

    hticks = float(np.nanmean(hticks_vals)) if len(hticks_vals) else np.nan
    return best_epochs_lst, hticks_vals, hticks


def _select_best_model_train(model_wrappers, train_losses_raw):
    """
    Training-loss analogue of _select_best_model.

    Uses per-run minimal training loss to select the best seed.

    Returns
    -------
    tuple
        (best_model_train, best_seed_idx_train, best_model_loss_train)
    """
    best_model_loss = [
        float(np.min(losses)) if len(losses) else np.nan for losses in train_losses_raw
    ]
    best_seed_idx = int(np.nanargmin(best_model_loss))
    best_model = model_wrappers[best_seed_idx]
    return best_model, best_seed_idx, best_model_loss


def _compute_best_epoch_idx(
    best_model, best_seed_idx, best_epochs_lst, max_len_val, attr_name="last_improved"
):
    """
    Determine the best epoch index for the configuration.

    Preference:
      1. best_model.<attr_name> (if available and attr_name is not None)
      2. epoch index of best value for that seed (from best_epochs_lst)

    Includes the original heuristic for 1-based indexing.
    """
    best_epoch_idx = None
    if attr_name is not None:
        best_epoch_idx = getattr(best_model, attr_name, None)

    if best_epoch_idx is None:
        best_epoch_idx = best_epochs_lst[best_seed_idx]

    best_epoch_idx = int(best_epoch_idx)

    # Heuristic for 1-based indexing
    if (
        max_len_val > 0
        and best_epoch_idx >= max_len_val
        and (best_epoch_idx - 1) < max_len_val
    ):
        best_epoch_idx = best_epoch_idx - 1

    # Clip to valid range
    if max_len_val > 0:
        best_epoch_idx = max(0, min(best_epoch_idx, max_len_val - 1))
    else:
        best_epoch_idx = 0
    return best_epoch_idx


def _compute_timing_stats(epoch_times_arr, model_wrappers, sf=1):
    """
    Compute timing statistics:
      - per-run mean epoch time
      - global mean/std of epoch times
      - mean/std of total run time (using sampling factor sf)
    """
    if not epoch_times_arr:
        return np.nan, np.nan, np.nan, np.nan

    mean_epoch_times = [np.mean(times) for times in epoch_times_arr]
    avg_epoch_time = np.mean(mean_epoch_times)
    std_epoch_time = np.std(mean_epoch_times)

    total_num_epochs = [len(m.epoch_times) for m in model_wrappers]
    total_time_arr = [
        np.mean(times) * epoch_num * sf if len(times) else np.nan
        for times, epoch_num in zip(epoch_times_arr, total_num_epochs)
    ]
    run_time_mean = np.nanmean(total_time_arr)
    run_time_std = np.nanstd(total_time_arr)

    return avg_epoch_time, std_epoch_time, run_time_mean, run_time_std


def _format_label_with_stats(label, mean_final, std_final):
    """Create label string with mean ± std in scientific notation."""
    mean_str = f"{mean_final:.2e}" if not np.isnan(mean_final) else "nan"
    std_str = f"{std_final:.1e}" if not np.isnan(std_final) else "nan"
    return f"{label} ({mean_str} ± {std_str})"


def prepare_model_run_data(models_dics, plot_params=None, best_epoch_termination=True):
    """
    Aggregate validation-loss and training-loss data across runs
    for each model configuration.

    Parameters
    ----------
    models_dics : dict
        Nested dictionary of model wrappers, typically
        {config_key: {seed: modelWrapper, ...}, ...}.
        Each modelWrapper is expected to have attributes:
        - val_loss_list
        - train_loss_list
        - epoch_times
        - best_val_loss
        - last_improved  (epoch index of best_val_loss)
    plot_params : dict, optional
        Plot styling overrides, mapping `config_key` to
        [color, linestyle, markerstyle, label].

    Returns
    -------
    tuple
        (run_data, raw_timing_data)

        run_data : list of dict
            One entry per configuration with keys like:
            - 'label', 'mean_best_model_loss', 'std_best_model_loss'
            - 'min_best_model_loss', 'max_best_model_loss'
            - 'mean_final', 'hticks'
            - 'epochs', 'running_min_min', 'running_min_max'
            - 'best_epoch_seed_idx', 'best_epochs_lst'
            - 'label_with_stats', 'color', 'linestyle', 'markerstyle'
            and training-loss analogues (prefixed with 'train_'):
            - 'train_mean_best_model_loss', 'train_std_best_model_loss'
            - 'train_min_best_model_loss', 'train_max_best_model_loss'
            - 'train_mean_final', 'train_hticks'
            - 'train_epochs', 'train_running_min_min', 'train_running_min_max'
            - 'train_best_epoch_seed_idx', 'train_best_epochs_lst'
        raw_timing_data : list of dict
            Per-configuration timing statistics:
            - 'avg_epoch_time', 'std_epoch_time'
            - 'run_time_mean', 'run_time_std', 'color', 'label'
    """
    plot_params = _ensure_plot_params(plot_params)

    run_data = []
    raw_timing_data = []

    model_keys = list(models_dics.keys())
    dic_values = list(models_dics.values())

    for i, (key, model_dic) in enumerate(zip(model_keys, dic_values)):
        model_wrappers = _extract_model_wrappers(model_dic)
        sample_model = model_wrappers[0]  # kept for future use / compatibility
        sf = 1  # sampling factor; left at 1 (can be adapted if needed)
        _ = sample_model  # avoid unused variable warnings

        # Styling
        color, linestyle, markerstyle, label = _get_style_for_key(key, i, plot_params)

        # -----------------------------------------------------------------
        # Validation loss and timing collections
        # -----------------------------------------------------------------
        val_losses_raw = _collect_val_losses(model_wrappers)
        epoch_times_arr = _collect_epoch_times(model_wrappers, percentile=80)

        # Stack and running minima (respecting true lengths) for validation
        val_losses, max_len_val, lengths_val = _pad_and_stack_losses(val_losses_raw)
        running_min_min, running_min_max = _compute_running_min_stats(
            val_losses, lengths_val
        )
        epochs = np.arange(max_len_val) if max_len_val > 0 else np.array([])

        # Final-loss stats (per time-series trajectory) for validation
        mean_final, std_final, _ = _compute_best_model_loss_stats(val_losses)

        # Best epochs and hticks (validation)
        best_epochs_lst, hticks_vals, hticks = _collect_best_epochs_and_hticks(
            model_wrappers, val_losses_raw
        )

        # Best model across seeds (based on best_val_loss)
        best_model, best_seed_idx, best_model_loss = _select_best_model(model_wrappers)
        best_epoch_idx = _compute_best_epoch_idx(
            best_model,
            best_seed_idx,
            best_epochs_lst,
            max_len_val,
            attr_name="last_improved",  # validation-specific attribute
        )

        # -----------------------------------------------------------------
        # Training loss collections (analogous to validation)
        # -----------------------------------------------------------------
        train_losses_raw = _collect_train_losses(model_wrappers)

        train_losses, max_len_train, lengths_train = _pad_and_stack_losses(
            train_losses_raw
        )
        train_running_min_min, train_running_min_max = _compute_running_min_stats(
            train_losses, lengths_train
        )
        train_epochs = np.arange(max_len_train) if max_len_train > 0 else np.array([])

        # Final-loss stats for training trajectories
        train_mean_final, train_std_final, _ = _compute_best_model_loss_stats(
            train_losses
        )

        # Best epochs and hticks based on training loss
        (
            train_best_epochs_lst,
            train_hticks_vals,
            train_hticks,
        ) = _collect_best_epochs_and_hticks_train(train_losses_raw)

        # Best model across seeds based on minimal training loss
        (
            best_model_train,
            best_seed_idx_train,
            best_model_loss_train,
        ) = _select_best_model_train(model_wrappers, train_losses_raw)

        # For training, we *always* use the argmin-based epoch index
        train_best_epoch_idx = _compute_best_epoch_idx(
            best_model_train,
            best_seed_idx_train,
            train_best_epochs_lst,
            max_len_train,
            attr_name=None,  # do not use any attribute, rely on argmin index
        )

        # -----------------------------------------------------------------
        # Timing stats
        # -----------------------------------------------------------------
        (
            avg_epoch_time,
            std_epoch_time,
            run_time_mean,
            run_time_std,
        ) = _compute_timing_stats(epoch_times_arr, model_wrappers, sf=sf)

        # Label with validation stats (unchanged behavior)
        label_with_stats = _format_label_with_stats(label, mean_final, std_final)

        # Optionally terminate validation curves at best epoch
        if best_epoch_termination and len(epochs) > 0:
            running_min_min = running_min_min[: best_epoch_idx + 1]
            running_min_max = running_min_max[: best_epoch_idx + 1]
            epochs = epochs[: best_epoch_idx + 1]

        # Aggregate run data (existing keys kept, training keys added)
        run_data.append(
            {
                # ------------------ validation-loss based ------------------
                "label": label,
                "mean_best_model_loss": float(np.mean(best_model_loss)),
                "std_best_model_loss": float(np.std(best_model_loss)),
                "min_best_model_loss": float(np.min(best_model_loss)),
                "max_best_model_loss": float(np.max(best_model_loss)),
                "mean_final": float(mean_final),
                "hticks": float(hticks),
                "best_epochs_lst": best_epochs_lst,
                "epochs": epochs,
                "running_min_min": running_min_min,
                "running_min_max": running_min_max,
                "best_epoch_seed_idx": [
                    best_epoch_idx,
                    best_seed_idx,
                ],  # from best seed's .last_improved if available
                "label_with_stats": label_with_stats,
                "color": color,
                "linestyle": linestyle,
                "markerstyle": markerstyle,
                # ------------------- training-loss based -------------------
                "train_mean_best_model_loss": float(np.nanmean(best_model_loss_train)),
                "train_std_best_model_loss": float(np.nanstd(best_model_loss_train)),
                "train_min_best_model_loss": float(np.nanmin(best_model_loss_train)),
                "train_max_best_model_loss": float(np.nanmax(best_model_loss_train)),
                "train_mean_final": float(train_mean_final),
                "train_hticks": float(train_hticks),
                "train_best_epochs_lst": train_best_epochs_lst,
                "train_epochs": train_epochs,
                "train_running_min_min": train_running_min_min,
                "train_running_min_max": train_running_min_max,
                "train_best_epoch_seed_idx": [
                    train_best_epoch_idx,
                    best_seed_idx_train,
                ],
            }
        )

        # Timing summary
        raw_timing_data.append(
            {
                "label": label,
                "avg_epoch_time": float(avg_epoch_time),
                "std_epoch_time": float(std_epoch_time),
                "run_time_mean": float(run_time_mean),
                "run_time_std": float(run_time_std),
                "color": color,
            }
        )

    return run_data, raw_timing_data
