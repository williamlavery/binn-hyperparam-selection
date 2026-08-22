# Synthetic Data Pipeline

This directory contains the synthetic data-generation pipeline used to create the `data_obj.npy` artifacts consumed by the BINN training and analysis notebooks.

## What the pipeline does

For each configured parameter combination, the data pipeline:

1. Loads the original time axis and spatial extent from `Lagergren_et_al_2020/originalDataObj.npy`.
2. Builds a 1D or 2D spatial grid and a reduced time grid.
3. Selects an initial condition, diffusion law, and growth law from the configured label set.
4. Solves the forward reaction-diffusion PDE.
5. Optionally injects observational noise.
6. Saves a serialized `Data` object under `Training/data/dataObj/`.

The main runtime entry point is:

```bash
python Training/data/python/pipeline/exec/data__sim.py
```

## Key code locations

- `Modules/dataClass.py`: defines the `OriginalData` and `Data` containers and the input-grid construction helpers.
- `Modules/PDESolver_1D.py`: one-dimensional PDE solver utilities.
- `Modules/PDESolver_2D.py`: two-dimensional PDE solver utilities.
- `pipeline/config/data__config.py`: top-level sweep settings.
- `pipeline/config/store.py`: stored diffusion, growth, and initial-condition functions.
- `pipeline/components/data__initialise.py`: maps labels such as `const`, `linear`, `quadratic`, `exp`, and `cos` to callable PDE ingredients.
- `pipeline/components/data__simulate.py`: serializes the generated data object to disk.
- `pipeline/components/runner.py`: expands the sweep grid and runs all configured combinations.

## Label families used in the synthetic experiments

The current code supports:

- diffusion labels: `const`, `linear`, `quadratic`, `exp`
- growth labels: `zero`, `const`, `linear`, `quadratic`, `exp`
- initial-condition labels: `cos`, `cosFlipped`, `scratch`, and amplitude-modified forms such as `cosFlat0.5`

These are implemented through `Training/data/python/pipeline/config/store.py` and selected in `data__initialise.py`.

## Configuration points

Edit `pipeline/config/data__config.py` to change:

- output dimensionality through `data_x2_num`
- spatial and temporal resolution
- family sweeps over diffusion and growth laws
- noise percentages and seeds
- plotting and overwrite behavior

The default configuration in the repository is oriented toward the paper-aligned 1D studies, with optional support for 2D generation.

## Output convention

Saved data objects live under `Training/data/dataObj/` in path signatures such as:

```text
dataX1num_38/dataX2num_1/dataTnum_5/dataK_1/dataICLabel_cos/...
```

This naming scheme is important: notebook catalogues later parse these directory names back into dataframe metadata.
