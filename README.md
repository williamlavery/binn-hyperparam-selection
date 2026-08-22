# Hyperparameter Selection for Biologically-Informed Neural Networks in Reaction-Diffusion Systems

Biologically-informed neural networks (BINNs) have emerged as a flexible subclass of physics-informed neural networks (PINNs) for learning terms in partial differential equations from data. BINNs are especially well suited to biological systems, where governing equations are often highly nonlinear and only partially known a priori, and where observations are sparse, noisy, and incomplete.

Applying BINNs effectively in practice depends critically on hyperparameter selection, yet those choices are often made heuristically and only lightly documented. This repository provides the synthetic-data generation, BINN training code, hyperparameter sweeps, and notebook analyses used to study that problem systematically. In particular, it supports a diagnostic workflow for selecting BINN hyperparameters from training-time signals, with experiments spanning diffusion and growth right-hand-side terms and data settings ranging from `1D+t` to `2D+t`.

Across the repository, the workflow is organized around three linked stages:

1. Generate standardized `data_obj.npy` datasets from prescribed diffusion, growth, and initial-condition families.
2. Train BINN models over hyperparameter grids spanning network widths, depths, PDE/data-loss weights, collocation counts, activation functions, early stopping, and constraint settings.
3. Reproduce paper and supplementary figures through notebooks that catalogue trained checkpoints, reload saved runs, and compare learned constitutive functions against the known ground truth.

The notebook suite covers four main cases together with supplementary analyses:

- `case1.ipynb`: Case I width sweeps and baseline 1D diagnostics.
- `Note_depth.ipynb`: Case I depth sweep.
- `case2.ipynb`: Case II diffusion-family comparisons.
- `case3.ipynb`: Case III growth-family comparisons and collocation sensitivity.
- `case4.ipynb`: Case IV noisy 2D experiments and early-stopping comparisons.
- `S1_1.ipynb`, `S1_2.ipynb`, `S1_3.ipynb`, `S1_4.ipynb`, `S2.ipynb`, `Note_depth.ipynb`: supplementary sensitivity analyses for PDE weight, constraint weight, collocation count, activation choice, and data loss.
- `colocation_figure.ipynb` and `TVsplit_figure.ipynb`: schematic notebooks illustrating PDE collocation sampling and train/validation splitting.

## Pipeline

At a high level, the repository workflow is:

```text
original / stored data
    -> synthetic PDE solve + noise injection
    -> saved data objects under Training/data/dataObj/
    -> BINN training sweeps under Training/binn/
    -> notebook catalogues + figure generation under Training/JN/
```

The runtime entry points are:

- `python Training/data/python/pipeline/exec/data__sim.py`
- `python Training/binn/python/pipeline/exec/binn__sim.py`

Both entry points build their sweep settings from Python config files.

## Repository structure

Colour key:

- <span style="color:#0f766e;">README / documentation</span>
- <span style="color:#2563eb;">Python source</span>
- <span style="color:#7c3aed;">Notebook</span>
- <span style="color:#b45309;">Shell script</span>
- <span style="color:#047857;">Environment / config</span>
- <span style="color:#be123c;">Spreadsheet / inventory</span>

<pre><code>hyperparam-selection/
├── <span style="color:#0f766e;">README.md</span>
├── <span style="color:#047857;">environment.yml</span>
├── DataStore/
│   ├── FKPP/
│   │   └── 1x38/
│   │       └── paper_IC/
│   │           └── ...
│   ├── paper/
│   │   └── ...
│   └── PFKPP/
│       └── ...
├── Training/
│   ├── <span style="color:#0f766e;">README.md</span>
│   ├── <span style="color:#2563eb;">pipeline_runtime.py</span>
│   ├── data/
│   │   ├── dataObj/
│   │   │   ├── dataX1num_11/
│   │   │   │   └── ...
│   │   │   ├── dataX1num_38/
│   │   │   │   └── ...
│   │   │   └── Lagergren_et_al_2020/
│   │   │       └── ...
│   │   └── python/
│   │       ├── <span style="color:#0f766e;">README.md</span>
│   │       ├── Modules/
│   │       │   └── ...
│   │       └── pipeline/
│   │           ├── components/
│   │           │   └── ...
│   │           ├── config/
│   │           │   └── ...
│   │           └── exec/
│   │               └── ...
│   ├── binn/
│   │   ├── <span style="color:#0f766e;">README.md</span>
│   │   ├── dataX1num_11/
│   │   │   └── ...
│   │   ├── dataX1num_38/
│   │   │   └── ...
│   │   └── python/
│   │       ├── Modules/
│   │       │   ├── <span style="color:#0f766e;">README.md</span>
│   │       │   └── ...
│   │       └── pipeline/
│   │           └── ...
│   └── JN/
│       ├── <span style="color:#0f766e;">README.md</span>
│       ├── notebook_helpers/
│       │   └── ...
│       ├── paper_helpers/
│       │   ├── <span style="color:#0f766e;">README.md</span>
│       │   └── ...
│       ├── pngs/
│       │   ├── 1D/
│       │   │   └── ...
│       │   ├── 2D/
│       │   │   └── ...
│       │   ├── colocation/
│       │   │   └── ...
│       │   └── TVsplit/
│       │       └── ...
│       ├── <span style="color:#7c3aed;">case1.ipynb</span>
│       ├── <span style="color:#7c3aed;">case2.ipynb</span>
│       ├── <span style="color:#7c3aed;">case3.ipynb</span>
│       ├── <span style="color:#7c3aed;">case4.ipynb</span>
│       ├── <span style="color:#7c3aed;">Note_depth.ipynb</span>
│       ├── <span style="color:#7c3aed;">S1_1.ipynb</span>
│       ├── <span style="color:#7c3aed;">S1_2.ipynb</span>
│       ├── <span style="color:#7c3aed;">S1_3.ipynb</span>
│       ├── <span style="color:#7c3aed;">S1_4.ipynb</span>
│       ├── <span style="color:#7c3aed;">S2.ipynb</span>
│       ├── <span style="color:#7c3aed;">colocation_figure.ipynb</span>
│       └── <span style="color:#7c3aed;">TVsplit_figure.ipynb</span>
├── <span style="color:#be123c;">notebook_data_obj_inventory.xlsx</span>
├── <span style="color:#be123c;">notebook_model_inventory.xlsx</span>
├── <span style="color:#b45309;">commit_push_everything_but_trained_models.sh</span>
└── <span style="color:#b45309;">batch_commit_push_data_dirs.sh</span>
</code></pre>

## Environment setup

Create the project environment first:

```bash
conda env create -f environment.yml
conda activate hyperparameter-selection
```

Then open the notebooks and select that same environment as the kernel.

## Documentation guide

Use the README files below for focused guidance on each stage of the repository:

- [README.md](README.md): top-level overview, setup, and workflow.
- [Training/README.md](Training/README.md): full project workflow from data generation through notebook analysis.
- [Training/data/python/README.md](Training/data/python/README.md): synthetic data-generation pipeline, PDE families, and configuration points.
- [Training/binn/README.md](Training/binn/README.md): BINN training sweeps, saved model layout, and model-storage helpers.
- [Training/binn/python/Modules/README.md](Training/binn/python/Modules/README.md): code-level overview of BINN architectures, MLP builders, and utility modules.
- [Training/JN/README.md](Training/JN/README.md): notebook guide describing what each notebook reproduces.
- [Training/JN/paper_helpers/README.md](Training/JN/paper_helpers/README.md): plotting, cataloguing, figure-building, and analysis helpers used by the notebooks.

## Practical workflow

Typical usage is:

1. Generate or regenerate synthetic datasets by editing `Training/data/python/pipeline/config/data__config.py` and running `data__sim.py`.
2. Configure BINN sweeps in `Training/binn/python/pipeline/config/binn__config.py` and run `binn__sim.py`.
3. Open the notebooks in `Training/JN/` to catalogue saved artifacts, reproduce figures, and compare learned diffusion/growth functions to the ground truth.

The Excel inventories `notebook_data_obj_inventory.xlsx` and `notebook_model_inventory.xlsx` are auxiliary summaries of saved artifacts for notebook-facing inspection.

## Model-storage helpers

Two shell helpers support version-control workflows around large trained-model directories:

- `commit_push_everything_but_trained_models.sh`: stages and pushes all changes except the main trained-model trees under `Training/binn/dataX1num_11` and `Training/binn/dataX1num_38`.
- `batch_commit_push_data_dirs.sh`: stages untracked model files in sequential batches, with Git LFS tracking for files above the configured size threshold.

## Primary Reference

J. H. Lagergren et al. "Biologically-informed neural networks guide mechanistic modeling from sparse experimental data". In: *PLoS Comput Biol* 16.12 (2020), e1008462. DOI: [10.1371/journal.pcbi.1008462](https://doi.org/10.1371/journal.pcbi.1008462).
