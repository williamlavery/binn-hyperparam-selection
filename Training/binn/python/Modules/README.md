# BINN Model Modules

This directory contains the reusable model-level code behind the BINN training pipeline.

## Main modules

- `Models/BuildBINNs.py`: the core BINN implementation used throughout the repository.
- `Models/BuildMLP.py`: configurable MLP builder used by the BINN surface, diffusion, and growth subnetworks.
- `Utils/Gradient.py`: automatic-differentiation helpers for PDE residuals and constitutive derivatives.
- `Utils/ModelWrapper.py`: wrapper utilities for training and checkpoint handling.
- `Utils/parse.py`: path and parameter-parsing helpers used when serializing runs.

## BINN structure

The `BINN` class combines:

- a surface-fitting network for the observed state `u`
- a diffusion network for `D(u)` or `D(u,t)`
- a growth network for `G(u)` or `G(u,t)`

It supports both 1D and 2D spatial input and can enforce:

- diffusion bounds
- growth bounds
- diffusion monotonicity
- growth monotonicity

The PDE loss is formed through autograd-based derivatives, while optional boundary-condition losses and diagnostic bookkeeping are handled inside the same model stack.

## Important customization points

If you want to change the learning behavior, the most important places to inspect are:

- activation selection in `BuildBINNs.py`
- hidden-layer widths/depths in `BuildMLP.py` and the BINN config
- bound and monotonicity penalties in `BuildBINNs.py`
- true-function injection in `Training/binn/python/pipeline/components/binn__modelConstructor.py`

## Relationship to the rest of the codebase

The pipeline layer chooses *which* model to build for a given run; this `Modules/` layer defines *how* that model behaves once instantiated.
