# Training Pipeline Guide

This directory contains the full workflow used in the repository: project-root discovery, synthetic data generation, BINN training sweeps, and notebook-based figure reproduction.

## Directory overview

- `pipeline_runtime.py`: shared helpers that locate the repository root and register import paths for scripts and notebooks.
- `data/`: synthetic data-generation code and saved `data_obj.npy` artifacts.
- `binn/`: trained BINN checkpoints plus the training pipeline code.
- `JN/`: notebooks and helper modules for reproducing paper and supplementary figures.

## End-to-end workflow

The pipeline runs in three stages.

### 1. Generate data objects

The data pipeline builds reaction-diffusion datasets on prescribed spatial and temporal grids, solves the forward PDE, optionally adds noise, and serializes the result as a `Data` object under `Training/data/dataObj/`.

Entry point:

```bash
python Training/data/python/pipeline/exec/data__sim.py
```

Main configuration file:

- `Training/data/python/pipeline/config/data__config.py`

The data config controls:

- original-data path from [[1]](##References)`original_data_relative_path`
- spatial grid sizes `data_x1_num`, `data_x2_num`
- time grid size `data_t_num`
- carrying capacity `data_k`
- gamma sweep `data_gammas`
- noise percentages `data_noise_percents`
- noise seeds `data_noise_seeds`
- initial-condition labels `data_ic_labels`
- diffusion-family labels `data_diff_labels`
- growth-family labels `data_grow_labels`
- plotting toggle `plot_bool`
- overwrite toggle `overwrite_bool`

### 2. Train BINN sweeps

The BINN pipeline loads saved data objects, constructs the BINN architecture, trains the BINN model, and stores trained checkpoints inside parameter-encoded folder names under `Training/binn/`.

Entry point:

```bash
python Training/binn/python/pipeline/exec/binn__sim.py
```

Main configuration file:

- `Training/binn/python/pipeline/config/binn__config.py`

The BINN config controls:

- original-data path `original_data_relative_path`
- spatial grid sizes `data_x1_num`, `data_x2_num`
- time-grid sweep `data_t_nums`
- carrying capacity `data_k`
- gamma sweep `gammas`
- noise percentages `noise_percents`
- noise seeds `noise_seeds`
- initial-condition labels `ic_labels`
- diffusion-family labels `diff_labels`
- growth-family labels `grow_labels`
- surface-network widths `binn_usizes`
- diffusion-network widths `binn_dsizes`
- growth-network widths `binn_gsizes`
- saved model labels `binn_model_labels`
- early-stopping sweep values `binn_es_values`
- train-validation split seeds `binn_tv_split_seeds`
- validation fractions `binn_vfs`
- surface-loss weights `surface_weights`
- PDE-loss weights `pde_weights`
- all-constraints toggle sweep `all_constraints`
- explicit constraint tuples `constraint_tuples`
- constraint-weight dictionary `constraint_weights`
- constraint-bound dictionary `constraint_bounds`
- done-parameter toggle `done_param_bool`
- Use ground-truth for PDE residual toggle `perfect_pde`
- BINN activation choice `binn_activation`
- density network hidden-layer counts `binn_surface_hidden_layers`
- diffusion/growth hidden-layer counts `binn_dg_hidden_layers`
- train-validation index-generation label `binn_generate_indices_label`
- device selection `binn_device`
- learning rate `binn_lr`
- relative update threshold `binn_rel_update_thresh`
- relative save threshold `binn_rel_save_thresh`
- maximum epochs for training `binn_epochs`
- early-stopping check points `binn_es_check`
- print frequency `print_freq`
- boundary-condition toggle `bc_bool`
- PDE collocation counts `num_pde_samples`
- data-loss labels `bn_data_loss_labels`
- constraint-sample counts `constraint_samples`
- constraint-loss storage toggle `store_constraint_losses`
- Adam-diagnostics storage toggle `store_adam_diagnostics`
- data-loss-diagnostics storage toggle `store_data_loss_diagnostics`
- plotting toggle `plot_bool`
- overwrite toggle `overwrite_bool`

### 3. Reproduce figures in notebooks

The notebooks in `Training/JN/` rebuild catalogues of saved artifacts, load selected data/model combinations, and generate the figures saved under `Training/JN/pngs/`.

See [Training/JN/README.md](JN/README.md) for the notebook-by-notebook guide.

## Artifact layout

Two directory trees matter most, together with the pipeline code that creates and consumes them:

- `Training/data/python/pipeline/` and `Training/data/dataObj/`: the executable synthetic-data pipeline together with the saved data objects it writes, keyed by data settings such as grid size, initial condition, diffusion family, growth family, and noise.
- `Training/binn/python/pipeline/` and `Training/binn/`: the executable BINN training pipeline together with the saved trained models it writes, keyed by both the data settings above and the BINN hyperparameters used during training.

Those path signatures are intentionally verbose because the notebooks later reconstruct catalogues directly from the folder names.

## Suggested usage pattern

### Recreating results

1. Edit `Training/data/python/pipeline/config/data__config.py` for the data families, grids, and noise settings you want to generate.
2. Run `python Training/data/python/pipeline/exec/data__sim.py` to create the required `data_obj.npy` files.
3. Edit `Training/binn/python/pipeline/config/binn__config.py` for the BINN sweep you want to reproduce.
4. Run `python Training/binn/python/pipeline/exec/binn__sim.py` to train and save the corresponding checkpoints.
5. Open the relevant notebook in `Training/JN/` and regenerate the figure panels.

### Loading results

1. Leave the saved `Training/data/dataObj/` and `Training/binn/` trees in place.
2. Open the relevant notebook in `Training/JN/`.
3. Let the notebook rebuild its catalogues of data objects and trained checkpoints from the folder names on disk.
4. Run only the analysis and plotting cells needed for the figures or diagnostics you want to inspect.
5. Use the repository-root commit helper scripts if you want to version-control code and notebook changes separately from large model directories.

## References

1. J. H. Lagergren et al. "Biologically-informed neural networks guide mechanistic modeling from sparse experimental data". In: *PLoS Comput Biol* 16.12 (2020), e1008462. DOI: [10.1371/journal.pcbi.1008462](https://doi.org/10.1371/journal.pcbi.1008462).

## Related documentation

- [Training/data/python/README.md](data/python/README.md)
- [Training/binn/README.md](binn/README.md)
- [Training/JN/README.md](JN/README.md)
