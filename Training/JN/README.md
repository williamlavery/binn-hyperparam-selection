# Notebook Guide

This directory contains the analysis notebooks used to reproduce the main and supplementary figures for the hyperparameter-selection study.

## Notebook workflow

The notebooks follow a common pattern:

1. Bootstrap repository imports with `notebook_helpers/notebook_setup.py`.
2. Build a catalogue of available saved data objects and BINN checkpoints.
3. Filter those catalogues to the case-specific settings.
4. Reload saved models and diagnostics.
5. Generate preprint-ready figures under `Training/JN/pngs/`.

The helper modules in `notebook_helpers/` and `paper_helpers/` are shared across nearly all notebooks.

## Main case notebooks

- `case1.ipynb`: Case I 1D baseline experiments, centered on width sweeps and the main learned-function diagnostics.
- `Note_depth.ipynb`: Case I depth comparisons for the constitutive subnetworks.
- `case2.ipynb`: Case II diffusion-family comparisons across constant, linear, quadratic, and exponential diffusion settings.
- `case3.ipynb`: Case III growth-family comparisons and related width/diagnostic studies.
- `case4.ipynb`: Case IV noisy 2D experiments, including early-stopping and width comparisons for the two-dimensional setting.

## Supplementary notebooks
- `S1_1.ipynb`: PDE-weight comparison.
- `S1_2.ipynb`: constraint-weight sensitivity.
- `S1_3.ipynb`: collocation-sample analysis for Case IIIA.
- `S1_4.ipynb`: activation-function analysis.
- `S2.ipynb`: MSE-versus-GLS data-loss comparison.

## Figure-construction notebooks

- `colocation_figure.ipynb`: visual explanation of PDE collocation points versus observed data-grid locations.
- `TVsplit_figure.ipynb`: visual explanation of train/validation splitting.

## Support code

- `notebook_helpers/`: path bootstrapping plus data/model loading utilities.
- `paper_helpers/`: plotting, file-finding, profile-comparison, timing, loss-summary, and real-space figure helpers.
- `pngs/`: saved notebook outputs grouped by case and figure number.

## Recommended usage

Run the data and BINN pipelines first, then open the notebook matching the figure family you want to regenerate. The notebooks expect the corresponding `data_obj.npy` files and trained `.pth` checkpoints to already exist on disk.

## Related documentation

- [Training/JN/paper_helpers/README.md](paper_helpers/README.md)
- [Training/README.md](../README.md)
