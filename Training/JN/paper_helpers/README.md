# Notebook Helper Modules

This directory contains the reusable helper code that turns saved data objects and BINN checkpoints into the paper-style plots used throughout the notebooks.

## What lives here

The helper modules fall into a few broad groups.

### Cataloguing and path parsing

- `file_finder.py`: discover saved artifacts on disk and convert path metadata into dataframe form.
- `paths.py`: serialize parameter dictionaries into the folder-signature format used throughout the repository.

### Plotting learned functions and diagnostics

- `plotting_final_mse.py`, `plotting_final_mse_2d.py`
- `plot_final_mse_grow.py`, `plot_final_mse_grow_2d.py`
- `plotting_final_loss.py`
- `plotting_loss_components.py`
- `plotting_loss_running_min.py`
- `plotting_mse_running_min.py`
- `plotting_mse_running_min_growth.py`
- `plotting_adam_diagnostics.py`
- `plotting_data_loss_diagnostics.py`
- `plotting_timings.py`

These modules assemble the summary panels used to compare hyperparameter settings across validation loss, constitutive-function error, optimization diagnostics, and runtime.

### Real-space and profile visualization

- `plotting_profiles.py`, `plotting_profiles_2d.py`
- `real_space_plotter.py`, `real_space_plotter_2d.py`

These helpers compare learned and true diffusion/growth behavior directly in physical or state space.

### Precomputation helpers

- `prepare_diff_mse.py`, `prepare_diff_mse_2d.py`
- `prepare_grow_mse.py`, `prepare_grow_mse_2d.py`
- `prepare_model_loss.py`, `prepare_model_loss_2d.py`

These modules extract reusable diagnostics from loaded models so the notebooks can focus on plotting logic.

### Shared utilities

- `utils.py`, `utils_2d.py`
- `__init__.py`

These expose a notebook-friendly import surface and small shared utilities.

## Relationship to the notebooks

The notebooks usually do not reimplement plotting logic directly. Instead, they:

1. build catalogues with `notebook_helpers/`
2. select the required trained runs
3. pass those runs into the functions defined here

This keeps the notebooks more compact and makes it easier to keep figure styling consistent across cases and supplementary analyses.
