"""Prepare grouped loss summaries from one-dimensional model wrappers.

Contents
--------
- _ensure_plot_params
- _get_style_for_key
- _train_attr_for_val_attr
- _collect_val_losses
- _collect_epoch_times
- _pad_and_stack_losses
- _compute_running_min_stats
- _compute_best_model_loss_stats
- _collect_best_epochs_and_hticks
- _select_best_model
- _compute_best_epoch_idx
- _compute_timing_stats
- _format_label_with_stats
- prepare_model_run_data"""

import numpy as np


def _ensure_plot_params(plot_params):
    """Return a non-None plot_params dict."""
    return plot_params or {}


def _get_style_for_key(key, index, plot_params):
    """
    Extract individual plot parameters (color, linestyle, markerstyle, label)
    for a given configuration key, falling back to defaults.
    """
    default_style = [f"C{index}", "-", ".", f"Run {index + 1}"]
    color, linestyle, markerstyle, label = plot_params.get(key, default_style)
    return color, linestyle, markerstyle, label


def _train_attr_for_val_attr(loss_attr):
    return "train_" + loss_attr[4:] if loss_attr.startswith("val_") else None


def _collect_val_losses(model_wrappers, loss_attr="val_loss_list"):
    """
    Collect validation (or train) loss lists into an array per run.

    Returns
    -------
    list[np.ndarray]
        One array per run (variable length).
    """
    val_losses_raw = []
    for m in model_wrappers:
        values = getattr(m, loss_attr, [])
        fallback_attr = _train_attr_for_val_attr(loss_attr)

        if len(values):
            losses = np.array(values, dtype=np.float64)
        elif fallback_attr is not None and len(getattr(m, fallback_attr, [])):
            losses = np.array(getattr(m, fallback_attr), dtype=np.float64)
        elif loss_attr != "val_loss_list":
            raise AttributeError(
                f"Model wrapper has no values for {loss_attr!r} "
                f"or fallback {fallback_attr!r}."
            )
        else:
            losses = np.array(m.train_loss_list, dtype=np.float64)
        val_losses_raw.append(losses)
    return val_losses_raw


def _collect_epoch_times(model_wrappers, percentile=80):
    """
    Collect epoch times per run and clip each run at the given percentile.

    Returns
    -------
    list[np.ndarray]
        Clipped epoch time arrays per run.
    """
    epoch_times_arr = []
    for m in model_wrappers:
        arr = np.array(m.epoch_times, dtype=np.float64)
        p = np.percentile(arr, percentile)
        clipped = arr[arr <= p]
        epoch_times_arr.append(clipped)
    return epoch_times_arr


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
    max_len_val = max(lengths)

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
    last_valid_vals = np.array(
        [
            row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan
            for row in val_losses
        ]
    )
    mean_final = np.nanmean(last_valid_vals)
    std_final = np.nanstd(last_valid_vals)
    return mean_final, std_final, last_valid_vals


def _collect_best_epochs_and_hticks(
    model_wrappers,
    val_losses_raw,
    loss_attr="val_loss_list",
):
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
        if loss_attr == "val_loss_list":
            best_val_epoch = np.argmin(
                np.abs(val_losses_raw[idx] - m.best_val_loss)
            )
            hticks_vals.append(m.best_val_loss)
        else:
            best_val_epoch = int(np.nanargmin(val_losses_raw[idx]))
            hticks_vals.append(np.nanmin(val_losses_raw[idx]))
        best_epochs_lst.append(best_val_epoch)

    hticks = np.mean(hticks_vals)
    return best_epochs_lst, hticks_vals, hticks


def _select_best_model(model_wrappers):
    """
    Select the best model (lowest best_val_loss) across seeds.

    Returns
    -------
    tuple
        (best_model, best_seed_idx, best_model_loss_list)
    """
    best_model_loss = [m.best_val_loss for m in model_wrappers]
    best_seed_idx = int(np.nanargmin(best_model_loss))
    best_model = model_wrappers[best_seed_idx]
    return best_model, best_seed_idx, best_model_loss


def _compute_best_epoch_idx(best_model, best_seed_idx, best_epochs_lst, max_len_val):
    """
    Determine the best epoch index for the configuration.

    Preference:
      1. best_model.last_improved (if available)
      2. epoch index of best_val_loss for that seed

    Includes the original heuristic for 1-based indexing.
    """
    best_epoch_idx = getattr(best_model, "last_improved", None)
    if best_epoch_idx is None:
        best_epoch_idx = best_epochs_lst[best_seed_idx]

    best_epoch_idx = int(best_epoch_idx)

    # Heuristic for 1-based indexing
    if best_epoch_idx >= max_len_val and (best_epoch_idx - 1) < max_len_val:
        best_epoch_idx = best_epoch_idx - 1

    # Clip to valid range
    best_epoch_idx = max(0, min(best_epoch_idx, max_len_val - 1))
    return best_epoch_idx


def _compute_timing_stats(epoch_times_arr, model_wrappers, sf=1):
    """
    Compute timing statistics:
      - per-run mean epoch time
      - global mean/std of epoch times
      - mean/std of total run time (using sampling factor sf)
    """
    mean_epoch_times = [np.mean(times) for times in epoch_times_arr]
    avg_epoch_time = np.mean(mean_epoch_times)
    std_epoch_time = np.std(mean_epoch_times)

    total_num_epochs = [len(m.epoch_times) for m in model_wrappers]
    total_time_arr = [
        np.mean(times) * epoch_num * sf
        for times, epoch_num in zip(epoch_times_arr, total_num_epochs)
    ]
    run_time_mean = np.mean(total_time_arr)
    run_time_std = np.std(total_time_arr)

    return avg_epoch_time, std_epoch_time, run_time_mean, run_time_std


def _format_label_with_stats(label, mean_final, std_final):
    """Create label string with mean ± std in scientific notation."""
    mean_str = f"{mean_final:.2e}"
    std_str = f"{std_final:.1e}"
    return f"{label} ({mean_str} ± {std_str})"


def prepare_model_run_data(
    models_dics,
    plot_params=None,
    best_epoch_termination=True,
    loss_attr="val_loss_list",
):
    """
    Aggregate validation-loss data across runs for each model configuration.

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
            - 'epochs', 'running_min_min', 'running_min_max'
            - 'best_epoch_idx', 'best_epochs_lst'
            - 'hticks', 'color', 'linestyle', 'markerstyle'
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
        model_wrappers = list(model_dic.values())
        sample_model = model_wrappers[0]  # kept for future use / compatibility
        sf = 1  # sampling factor; left at 1 (can be adapted if needed)
        _ = sample_model  # avoid unused variable warnings

        # Styling
        color, linestyle, markerstyle, label = _get_style_for_key(
            key, i, plot_params
        )

        # Loss and timing collections
        val_losses_raw = _collect_val_losses(model_wrappers, loss_attr=loss_attr)
        epoch_times_arr = _collect_epoch_times(model_wrappers, percentile=80)

        # Stack and running minima (respecting true lengths)
        val_losses, max_len_val, lengths = _pad_and_stack_losses(val_losses_raw)
        running_min_min, running_min_max = _compute_running_min_stats(
            val_losses, lengths
        )
        epochs = np.arange(max_len_val)

        # Final-loss stats (per time-series trajectory)
        mean_final, std_final, _ = _compute_best_model_loss_stats(val_losses)

        # Best epochs and hticks
        best_epochs_lst, hticks_vals, hticks = _collect_best_epochs_and_hticks(
            model_wrappers,
            val_losses_raw,
            loss_attr=loss_attr,
        )

        # Best model across seeds (based on best_val_loss)
        best_model, best_seed_idx, best_model_loss = _select_best_model(model_wrappers)
        best_epoch_idx = _compute_best_epoch_idx(
            best_model, best_seed_idx, best_epochs_lst, max_len_val
        )


        # Timing stats
        (
            avg_epoch_time,
            std_epoch_time,
            run_time_mean,
            run_time_std,
        ) = _compute_timing_stats(epoch_times_arr, model_wrappers, sf=sf)

        # Label with stats
        label_with_stats = _format_label_with_stats(label, mean_final, std_final)

        if best_epoch_termination:
            running_min_min = running_min_min[: best_epoch_idx + 1]
            running_min_max = running_min_max[: best_epoch_idx + 1]
            epochs = epochs[: best_epoch_idx + 1]

        # Aggregate run data
        run_data.append(
            {
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
                "best_epoch_seed_idx": [best_epoch_idx,best_seed_idx],  # from best seed's .last_improved if available
                "label_with_stats": label_with_stats,
                "color": color,
                "linestyle": linestyle,
                "markerstyle": markerstyle,
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
