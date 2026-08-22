"""Observation-noise helpers for generated synthetic datasets.

Contents
--------
- DATA_add_noise_func_info: return the noise parameters for bookkeeping.
- DATA_add_noise: add reproducible non-negative Gaussian noise to clean data.
"""

import numpy as np


def DATA_add_noise_func_info(data_obj_params):
    """Return the parameters that identify the stochastic observation model."""
    return data_obj_params["add_noise_params"]


def DATA_add_noise(u_clean, data_obj_params, epsilon=1e-12):
    """Add reproducible, non-negative Gaussian observation noise."""
    add_noise_params = data_obj_params["add_noise_params"]
    gamma = float(add_noise_params["dataGamma"])
    noise_percent = float(add_noise_params["dataNoisePercent"])
    seed = int(add_noise_params["dataNoiseSeed"])

    u_noisy = u_clean.copy()
    signal = u_noisy

    if noise_percent == 0:
        additional_info = {
            "achieved_percent": 0.0,
            "sigma_effective": 0.0,
        }
        return DATA_add_noise_func_info(data_obj_params), u_noisy, additional_info

    target_fraction = noise_percent / 100.0
    tolerance = 1e-6
    max_iter = 25

    rng = np.random.default_rng(seed)
    base_noise = rng.standard_normal(size=signal.shape) * np.abs(signal) ** gamma

    signal_floor = max(float(epsilon), 1e-3)
    mask = np.abs(signal) >= signal_floor
    if not np.any(mask):
        raise ValueError(
            "All signal entries are below the threshold used for relative-noise "
            "calibration."
        )

    denominator = np.abs(signal[mask])
    baseline_fraction = np.mean(np.abs(base_noise[mask]) / denominator)
    if baseline_fraction == 0:
        raise ValueError("Baseline noise is zero; check gamma and input signal.")

    scale = target_fraction / baseline_fraction
    err_fraction = np.inf
    proposal = signal.copy()

    for _ in range(max_iter):
        noise = base_noise * scale
        proposal = np.clip(signal + noise, 0.0, np.inf)
        err_fraction = np.mean(np.abs(proposal[mask] - signal[mask]) / denominator)

        if abs(err_fraction - target_fraction) <= tolerance:
            break
        if err_fraction == 0:
            raise RuntimeError("Noise calibration collapsed to zero after clipping.")
        scale *= target_fraction / err_fraction
    else:
        raise RuntimeError(
            "Noise scaling did not converge after "
            f"{max_iter} iterations: achieved={err_fraction * 100:.5f}%, "
            f"target={noise_percent:.5f}%."
        )

    u_noisy[...] = proposal
    additional_info = {
        "achieved_percent": float(err_fraction * 100.0),
        "sigma_effective": float(np.std(base_noise * scale)),
    }

    return DATA_add_noise_func_info(data_obj_params), u_noisy, additional_info

