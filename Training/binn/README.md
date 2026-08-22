# BINN Training Pipeline

This directory stores both the trained BINN checkpoints and the Python code used to produce them.

## What the BINN pipeline does

For each configured training combination, the BINN pipeline:

1. Locates a matching saved `data_obj.npy`.
2. Builds a BINN with separate subnetworks for the surface fit and the constitutive diffusion/growth terms.
3. Splits available data into training and validation subsets.
4. Optimizes a combined objective containing data loss, PDE loss, and optional constraint penalties.
5. Saves the trained checkpoint and any requested diagnostics into a parameter-encoded directory under `Training/binn/`.

The main runtime entry point is:

```bash
python Training/binn/python/pipeline/exec/binn__sim.py
```

## Key code locations

- `python/pipeline/config/binn__config.py`: top-level sweep settings.
- `python/pipeline/components/runner.py`: expands the full hyperparameter grid.
- `python/pipeline/components/binn__modelConstructor.py`: builds the BINN and wires in the appropriate true-function metadata.
- `python/pipeline/components/binn__loadData.py`: loads the matching saved data object.
- `python/pipeline/components/binn__splitTV.py`: creates train/validation splits.
- `python/pipeline/components/binn__firstTrain.py`: initial training pass.
- `python/pipeline/components/binn__retrain.py`: resumed or repeated fitting stage.
- `python/pipeline/components/binn__saveModel.py`: checkpoint serialization.
- `python/pipeline/components/binn__simulate.py`: orchestration wrapper for one run.

## Hyperparameters explored in this repository

The default sweep configuration includes variations over:

- surface-network width `binnUsize`
- diffusion and growth widths `binnDsize`, `binnGsize`
- hidden-layer depth for the surface and constitutive subnetworks
- activation function
- validation fraction and split seed
- early-stopping setting
- PDE collocation count
- data-loss label
- PDE and surface loss weights
- monotonicity and bound constraints for learned constitutive laws

These settings are defined in `python/pipeline/config/binn__config.py`.

## Saved model layout

Trained models are stored inside nested path signatures that mirror both the data settings and the BINN settings. The notebooks rely heavily on this convention, using folder names as structured metadata.


## Storage and version-control helpers

The repository root includes two scripts intended for this directory:

- [commit_push_everything_but_trained_models.sh](../../commit_push_everything_but_trained_models.sh): commit all non-model changes.
- [batch_commit_push_data_dirs.sh](../../batch_commit_push_data_dirs.sh): progressively add and push large trained-model batches, with Git LFS tracking when needed.

## Related documentation

- [Training/binn/python/Modules/README.md](python/Modules/README.md)
- [Training/JN/README.md](../JN/README.md)
