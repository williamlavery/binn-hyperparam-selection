"""Dimension-agnostic BINN components for 1D/2D pipelines.

Contents
--------
- _activation_from_name: map a configured activation label to the matching Torch module.
- _hidden_layer_sizes: expand a width and hidden-layer count into an MLP layer list.
- _bound_tuple: normalize configured lower/upper bounds for coefficient constraints.
- generate_random_inputs: sample PDE collocation points for 1D or 2D inputs.
- _zero_constraint: return a scalar zero penalty on the correct device/dtype.
- _call_u_func: evaluate a surface network on 1D or 2D inputs with a unified interface.
- _fixed_constraint_inputs: build the fixed density grid used for D/G constraint penalties.
- apply_constraints: accumulate D/G range and monotonicity penalties on a fixed u grid.
- generate_bc_inputs: sample points on the no-flux boundary for 1D or 2D domains.
- apply_BC: evaluate the boundary-condition penalty for sampled no-flux points.
- pde_loss_without_bc: return pointwise PDE residual loss without boundary-condition terms.
- perfect_pde_loss_without_bc: return pointwise PDE residual loss using the exact analytic D/G.
- bc_no_flux_loss: return unweighted no-flux boundary residuals.
- pde_loss_with_bc: return pointwise PDE residual loss with no-flux boundary penalties.
- data_loss_MSE: compute the standard mean-squared data-fit loss.
- _initial_low_density_weight: weight t=t_min, low-density data points for modified MSE losses.
- data_loss_MSEmodified10: compute the weighted MSE variant that emphasizes low-density starts.
- data_loss_GLS: compute the generalized least-squares data loss with inverse-variance weighting.
- data_loss_GLSpow: compute the GLS-style loss with a configurable power on the response scale.
- u_MLP: surface network for cell density u(x,t) or u(x1,x2,t).
- D_MLP: diffusion head for a non-negative normalized diffusivity D(u).
- G_MLP: growth head for normalized proliferation G(u).
- BINN: unified BINN wrapper combining surface, diffusion, and growth networks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from Training.binn.python.Modules.Models.BuildMLP import BuildMLP
from Training.binn.python.Modules.Utils.Gradient import Gradient


# Physical bounds on the coefficient functions.
# Diffusion in mm^2 day^-1; growth (per-capita rate) in day^-1.
# G_BOUND evaluates to (-0.48, 2.4) day^-1 (hourly rates -0.02, 0.1 scaled by 24).
D_BOUND = (0, 0.1)
G_BOUND = (-0.02 / (1 / 24), 0.1 / (1 / 24))
D_MONO = "increasing"
G_MONO = "decreasing"
D_BOUND_WEIGHT = 1e4
D_MONO_WEIGHT = 1e4
G_BOUND_WEIGHT = 1e4 
G_MONO_WEIGHT = 1e4


def _activation_from_name(name):
    activation_name = str(name).lower()
    if activation_name == "silu":
        return nn.SiLU()
    if activation_name == "sigmoid":
        return nn.Sigmoid()
    if activation_name == "tanh":
        return nn.Tanh()
    raise ValueError(
        f"Unsupported BINN activation {name!r}. Expected one of: silu, sigmoid, tanh."
    )


def _hidden_layer_sizes(width, hidden_layers):
    hidden_layers = int(hidden_layers)
    if hidden_layers <= 0:
        raise ValueError(
            f"BINN hidden layer count must be a positive integer, got {hidden_layers!r}."
        )
    return [width] * hidden_layers + [1]


def _bound_tuple(params, key, default):
    values = params.get(key, default)
    if len(values) != 2:
        raise ValueError(f"{key} must be a two-value tuple: (min, max)")
    lower, upper = float(values[0]), float(values[1])
    if upper == 0:
        raise ValueError(f"{key} upper bound must be non-zero for normalization")
    return lower, upper


def generate_random_inputs(self, inputs):
    """Sample PDE collocation points for 1D or 2D inputs."""
    torch.manual_seed(self.loss_count)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(self.loss_count)

    if self.dim == 1:
        unit = torch.rand(self.num_samples, 2, requires_grad=True, device=inputs.device)
        x = unit[:, 0:1] * (self.x1_max - self.x1_min) + self.x1_min
        t = unit[:, 1:2] * (self.t_max - self.t_min) + self.t_min
        return torch.cat([x, t], dim=1).float()

    unit = torch.rand(self.num_samples, 3, requires_grad=True, device=inputs.device)
    x1 = unit[:, 0:1] * (self.x1_max - self.x1_min) + self.x1_min
    x2 = unit[:, 1:2] * (self.x2_max - self.x2_min) + self.x2_min
    t = unit[:, 2:3] * (self.t_max - self.t_min) + self.t_min
    return torch.cat([x1, x2, t], dim=1).float()


def _zero_constraint(reference):
    return torch.zeros((), device=reference.device, dtype=reference.dtype)


def _call_u_func(func, u):
    values = func(u.detach().cpu().numpy())
    return torch.as_tensor(values, device=u.device, dtype=u.dtype)


def _fixed_constraint_inputs(self, reference):
    u = self.constraint_u_values.to(device=reference.device, dtype=reference.dtype)
    u = u.detach().clone().requires_grad_(True)

    if getattr(self.diffusion, "inputs", 1) == 1 and (
        self.growth is None or getattr(self.growth, "inputs", 1) == 1
    ):
        return u, None

    t_mid = 0.5 * (self.t_min + self.t_max)
    t = torch.full_like(u, t_mid, requires_grad=True)
    return u, t


def apply_constraints(self, D, G, u):
    """Accumulate D/G range and monotonicity penalties on a fixed u grid."""
    reference = D if D is not None else G
    self.D_bound_loss = _zero_constraint(reference)
    self.D_mono_loss = _zero_constraint(reference)
    self.G_bound_loss = _zero_constraint(reference)
    self.G_mono_loss = _zero_constraint(reference)

    if not (self.D_bound or self.D_mono or self.G_bound or self.G_mono):
        self.D_loss = self.D_bound_loss + self.D_mono_loss
        self.G_loss = self.G_bound_loss + self.G_mono_loss
        self.constraint_loss = self.D_loss + self.G_loss
        return

    u_constraint, t_constraint = _fixed_constraint_inputs(self, reference)
    D_constraint = (
        self.diffusion(u_constraint)
        if self.diffusion.inputs == 1
        else self.diffusion(u_constraint, t_constraint)
    )
    G_constraint = None
    if self.growth is not None:
        G_constraint = (
            self.growth(u_constraint)
            if self.growth.inputs == 1
            else self.growth(u_constraint, t_constraint)
        )

    if self.D_bound:
        lower_violation = torch.relu(self.alpha_D_min - D_constraint).pow(2)
        upper_violation = torch.relu(D_constraint - self.alpha_D_max).pow(2)
        self.D_bound_loss = self.D_weight * torch.sum(
            lower_violation + upper_violation
        )

    if self.G_bound and G_constraint is not None:
        lower_violation = torch.relu(self.alpha_G_min - G_constraint).pow(2)
        upper_violation = torch.relu(G_constraint - self.alpha_G_max).pow(2)
        self.G_bound_loss = self.G_weight * torch.sum(
            lower_violation + upper_violation
        )

    if self.D_mono:
        try:
            dDdu = Gradient(D_constraint, u_constraint, order=1)
            if D_MONO == "increasing":
                invalid_D_mono = dDdu < 0.0
            elif D_MONO == "decreasing":
                invalid_D_mono = dDdu > 0.0
            else:
                raise ValueError(f"Unknown D_mono constraint: {D_MONO!r}")
            self.D_mono_loss = self.dDdu_weight * torch.sum(torch.where(
                invalid_D_mono, dDdu ** 2, torch.zeros_like(dDdu)
            ))
        except RuntimeError:
            pass

    if self.G_mono and G_constraint is not None:
        try:
            dGdu = Gradient(G_constraint, u_constraint, order=1)
            if G_MONO == "increasing":
                invalid_G_mono = dGdu < 0.0
            elif G_MONO == "decreasing":
                invalid_G_mono = dGdu > 0.0
            else:
                raise ValueError(f"Unknown G_mono constraint: {G_MONO!r}")
            self.G_mono_loss = self.dGdu_weight * torch.sum(torch.where(
                invalid_G_mono, dGdu ** 2, torch.zeros_like(dGdu)
            ))
        except RuntimeError:
            pass

    self.D_loss = self.D_bound_loss + self.D_mono_loss
    self.G_loss = self.G_bound_loss + self.G_mono_loss
    self.constraint_loss = self.D_loss + self.G_loss


def generate_bc_inputs(self, inputs):
    """Sample points on the no-flux boundary for 1D or 2D."""
    torch.manual_seed(self.loss_count)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(self.loss_count)

    t = (
        torch.rand(self.num_bcs, 1, requires_grad=True, device=inputs.device)
        * (self.t_max - self.t_min)
        + self.t_min
    )

    if self.dim == 1:
        selector = torch.randint(0, 2, (self.num_bcs, 1), device=inputs.device, dtype=torch.bool)
        x_left = torch.full_like(t, self.x1_min)
        x_right = torch.full_like(t, self.x1_max)
        x = torch.where(selector, x_left, x_right)
        return torch.cat([x, t], dim=1)

    face = torch.randint(0, 4, (self.num_bcs, 1), device=inputs.device)
    x1_free = (
        torch.rand(self.num_bcs, 1, device=inputs.device)
        * (self.x1_max - self.x1_min)
        + self.x1_min
    )
    x2_free = (
        torch.rand(self.num_bcs, 1, device=inputs.device)
        * (self.x2_max - self.x2_min)
        + self.x2_min
    )

    x1_min = torch.full_like(t, self.x1_min)
    x1_max = torch.full_like(t, self.x1_max)
    x2_min = torch.full_like(t, self.x2_min)
    x2_max = torch.full_like(t, self.x2_max)

    x1 = torch.where(face == 0, x1_min, torch.where(face == 1, x1_max, x1_free))
    x2 = torch.where(face == 2, x2_min, torch.where(face == 3, x2_max, x2_free))

    return torch.cat([x1, x2, t], dim=1)


def apply_BC(self, inputs):
    self.bc_loss = 0
    inputs_bc = generate_bc_inputs(self, inputs)
    u_bc = self.surface_fitter(inputs_bc)
    self.bc_loss += self.bc_weight * torch.sum(bc_no_flux_loss(self, inputs_bc, u_bc))


def pde_loss_without_bc(self, inputs, outputs):
    """Return pointwise PDE residual loss without boundary-condition terms."""
    if getattr(self, "perfect_pde", False):
        return perfect_pde_loss_without_bc(self, inputs, outputs)

    if self.dim == 1:
        t = inputs[:, 1:2]
        u = outputs.clone()
        d1 = Gradient(u, inputs, order=1)
        ux = d1[:, 0:1]
        ut = d1[:, 1:2]

        D = self.diffusion(u) if self.diffusion.inputs == 1 else self.diffusion(u, t)
        div = Gradient(D * ux, inputs)[:, 0:1]

        if self.growth is not None:
            G = self.growth(u) if self.growth.inputs == 1 else self.growth(u, t)
            rhs = self.D_max * div + self.G_max * G * u
        else:
            rhs = self.D_max * div
            G = None

        pde = (ut - rhs).pow(2)
        apply_constraints(self, D, G, u)
        return pde

    t = inputs[:, 2:3]
    u = outputs.clone()
    d1 = Gradient(u, inputs, order=1)
    ux1, ux2, ut = d1[:, 0:1], d1[:, 1:2], d1[:, 2:3]

    D = self.diffusion(u) if self.diffusion.inputs == 1 else self.diffusion(u, t)
    div = Gradient(D * ux1, inputs)[:, 0:1] + Gradient(D * ux2, inputs)[:, 1:2]

    if self.growth is not None:
        G = self.growth(u) if self.growth.inputs == 1 else self.growth(u, t)
        rhs = self.D_max * div + self.G_max * G * u
    else:
        rhs = self.D_max * div
        G = None

    pde = (ut - rhs).pow(2)
    apply_constraints(self, D, G, u)
    return pde


def perfect_pde_loss_without_bc(self, inputs, outputs):
    """Return pointwise PDE residual loss using the exact analytic D/G."""
    u = outputs.clone()
    d1 = Gradient(u, inputs, order=1)

    if self.dim == 1:
        ux = d1[:, 0:1]
        ut = d1[:, 1:2]
        uxx = Gradient(ux, inputs, order=1)[:, 0:1]
        lap_u = uxx
        grad_u_sq = ux.pow(2)
    else:
        ux1, ux2, ut = d1[:, 0:1], d1[:, 1:2], d1[:, 2:3]
        uxx1 = Gradient(ux1, inputs, order=1)[:, 0:1]
        uxx2 = Gradient(ux2, inputs, order=1)[:, 1:2]
        lap_u = uxx1 + uxx2
        grad_u_sq = ux1.pow(2) + ux2.pow(2)

    D_true = _call_u_func(self.diffusion_true_func, u)
    dDdu_true = _call_u_func(self.diffusion_true_deriv_func, u)
    G_true = _call_u_func(self.growth_true_func, u) if self.growth_true_func is not None else None

    rhs = D_true * lap_u + dDdu_true * grad_u_sq
    if G_true is not None:
        rhs = rhs + G_true * u

    self.D_bound_loss = _zero_constraint(u)
    self.D_mono_loss = _zero_constraint(u)
    self.G_bound_loss = _zero_constraint(u)
    self.G_mono_loss = _zero_constraint(u)
    self.constraint_loss = _zero_constraint(u)
    return (ut - rhs).pow(2)


def bc_no_flux_loss(self, inputs_bc, u_bc):
    """Return unweighted no-flux boundary residuals."""
    grads = Gradient(u_bc, inputs_bc, order=1)

    if self.dim == 1:
        dudx = grads[:, 0:1]
        return dudx.pow(2)

    du_dx1 = grads[:, 0:1]
    du_dx2 = grads[:, 1:2]

    on_x1_min = inputs_bc[:, 0:1] == self.x1_min
    on_x1_max = inputs_bc[:, 0:1] == self.x1_max
    on_x2_min = inputs_bc[:, 1:2] == self.x2_min
    on_x2_max = inputs_bc[:, 1:2] == self.x2_max

    dudn = torch.zeros_like(du_dx1)
    dudn = torch.where(on_x1_min | on_x1_max, du_dx1, dudn)
    dudn = torch.where(on_x2_min | on_x2_max, du_dx2, dudn)

    return dudn.pow(2)


def pde_loss_with_bc(self, inputs, outputs):
    """Return pointwise PDE residual loss with no-flux boundary penalties."""
    self.pde_loss_val = 0
    self.bc_loss_val_total = 0

    pde_loss = pde_loss_without_bc(self, inputs, outputs)
    apply_BC(self, inputs)
    return pde_loss


def data_loss_MSE(self, pred, true):
    return (pred - true).pow(2)


def _initial_low_density_weight(self, true, factor, threshold=0.012):
    """Weight t=t_min, low-density data points for modified MSE losses."""
    weights = torch.ones_like(true)
    inputs = getattr(self, "inputs", None)
    if inputs is None or inputs.ndim < 2 or inputs.shape[0] != true.shape[0]:
        return weights

    t_col = int(getattr(self, "dim", 1))
    if t_col >= inputs.shape[1]:
        return weights

    t_values = inputs[:, t_col : t_col + 1]
    t_min = torch.as_tensor(
        getattr(self, "t_min", 0.0),
        dtype=t_values.dtype,
        device=t_values.device,
    )
    is_initial_time = torch.isclose(
        t_values,
        t_min,
        rtol=1e-5,
        atol=1e-8,
    )
    is_low_density = true < threshold
    return torch.where(is_initial_time & is_low_density, weights * factor, weights)


def data_loss_MSEmodified10(self, pred, true):
    residual = (pred - true).pow(2)
    weights = _initial_low_density_weight(self, true, factor=10.0, threshold=0.1)
    return weights * residual


def data_loss_GLS(self, pred, true):
    residual = (pred - true).pow(2)
    residual *= pred.abs().clamp(min=1e-10).pow(-2 * self.gamma)
    return residual


def data_loss_GLSpow(self, pred, true):
    residual = (pred - true).pow(2)
    gls_power = float(getattr(self, "GLS_power", 0.5))
    residual *= pred.abs().clamp(min=1e-10).pow(-gls_power)
    return residual


class u_MLP(nn.Module):
    """Surface network for cell density u(x,t) or u(x1,x2,t)."""

    def __init__(
        self,
        input_features=2,
        u_size=64,
        activation_name="silu",
        hidden_layers=3,
    ):
        super().__init__()
        self.size = u_size
        self.mlp = BuildMLP(
            input_features=input_features,
            layers=_hidden_layer_sizes(self.size, hidden_layers),
            activation=_activation_from_name(activation_name),
            linear_output=False,
            output_activation=nn.Softplus(),
            seed=0,
        )

    def forward(self, inputs):
        return self.mlp(inputs)


class D_MLP(nn.Module):
    """Diffusion head for a non-negative normalised diffusivity D(u)."""

    def __init__(
        self,
        input_features=1,
        D_size=4,
        use_single_bias=False,
        activation_name="silu",
        hidden_layers=3,
    ):
        super().__init__()
        self.inputs = input_features
        self.min = 0
        self.max = 0.1

        self.size = D_size
        self.mlp = BuildMLP(
            input_features=input_features,
            layers=_hidden_layer_sizes(self.size, hidden_layers),
            activation=_activation_from_name(activation_name),
            linear_output=False,
            output_activation=nn.Softplus(),
            seed=1,
            use_single_bias=use_single_bias,
        )

    def forward(self, u):
        return self.mlp(u)


class G_MLP(nn.Module):
    """Growth head for normalised proliferation G(u)."""

    def __init__(
        self,
        input_features=1,
        G_size=4,
        activation_name="silu",
        hidden_layers=3,
    ):
        super().__init__()
        self.inputs = input_features
        # Growth bounds in day^-1: (-0.48, 2.4).
        self.min = -0.02 / (1 / 24)
        self.max = 0.1 / (1 / 24)

        self.size = G_size
        self.mlp = BuildMLP(
            input_features=input_features,
            layers=_hidden_layer_sizes(self.size, hidden_layers),
            activation=_activation_from_name(activation_name),
            linear_output=True,
            seed=2,
        )

    def forward(self, u, t=None):
        return self.mlp(u)


class BINN(nn.Module):
    """Unified BINN wrapper combining surface, diffusion, and growth networks."""

    def __init__(self, data_obj_params, model_params, data_loss_func, pde_loss_func):
        RDEq_params_store = data_obj_params["RDEq_params_store"]

        x1 = RDEq_params_store["x1"]
        x2 = RDEq_params_store.get("x2")
        t = RDEq_params_store["t"]
        K = RDEq_params_store["K"]
        gamma = data_obj_params["add_noise_params"]["dataGamma"]

        diffusion_true_func = data_obj_params["RDEq_extra_params"]["diffusionTrueFunc"]
        diffusion_true_deriv_func = data_obj_params["RDEq_extra_params"]["diffusionTrueDerivFunc"]
        growth_true_func = data_obj_params["RDEq_extra_params"]["growthTrueFunc"]

        u_max = data_obj_params["RDEq_extra_params"]["max_u_clean"]
        u_min = data_obj_params["RDEq_extra_params"]["min_u_clean"]

        binn_model_params = model_params["binn_model_params"]
        binn_construction_params = binn_model_params["binn_construction_params"]
        BNdata_loss_params = binn_model_params.get("BNdata_loss_params", {})

        binnUsize = binn_construction_params["binnUsize"]
        binnDsize = binn_construction_params["binnDsize"]
        binnGsize = binn_construction_params["binnGsize"]
        done_param_bool = binn_construction_params["DoneParamBool"]
        perfect_pde = bool(binn_construction_params.get("perfectPDE", False))
        binn_activation = binn_construction_params.get("binnActivation", "silu")
        surface_hidden_layers = binn_construction_params.get(
            "binnSurfaceHiddenLayers",
            3,
        )
        dg_hidden_layers = binn_construction_params.get("binnDGHiddenLayers", 3)
        allConstraints = binn_construction_params.get("allConstraints", False)
        constraint_tuple = binn_construction_params.get("constraintTuple")
        if constraint_tuple is None:
            constraint_tuple = (allConstraints, allConstraints, allConstraints, allConstraints)
        constraint_tuple = tuple(constraint_tuple)
        if len(constraint_tuple) != 4:
            raise ValueError(
                "constraintTuple must be (D_bound, D_mono, G_bound, G_mono)"
            )

        device = binn_construction_params["binnDevice"]
        numPDEsamples = binn_model_params["pde_loss_params"]["numPDEsamples"]

        super().__init__()

        self.dim = 1 if x2 is None else 2
        self.surface_fitter = u_MLP(
            input_features=self.dim + 1,
            u_size=binnUsize,
            activation_name=binn_activation,
            hidden_layers=surface_hidden_layers,
        ).to(device)
        self.perfect_pde = perfect_pde
        self.diffusion = None
        self.allConstraints = allConstraints
        self.constraintTuple = constraint_tuple
        self.D_bound = bool(self.constraintTuple[0])
        self.D_mono = bool(self.constraintTuple[1])
        self.G_bound = bool(self.constraintTuple[2])
        self.G_mono = bool(self.constraintTuple[3])
        self.D_bound_weight = binn_construction_params.get(
            "D_bound_weight", D_BOUND_WEIGHT
        )
        self.D_mono_weight = binn_construction_params.get(
            "D_mono_weight", D_MONO_WEIGHT
        )
        self.G_bound_weight = binn_construction_params.get(
            "G_bound_weight", G_BOUND_WEIGHT
        )
        self.G_mono_weight = binn_construction_params.get(
            "G_mono_weight", G_MONO_WEIGHT
        )
        self.growth = None
        if not self.perfect_pde:
            self.diffusion = D_MLP(
                D_size=binnDsize,
                use_single_bias=done_param_bool,
                activation_name=binn_activation,
                hidden_layers=dg_hidden_layers,
            ).to(device)
            self.growth = (
                G_MLP(
                    G_size=binnGsize,
                    activation_name=binn_activation,
                    hidden_layers=dg_hidden_layers,
                ).to(device)
                if binnGsize
                else None
            )

        self.D_min, self.D_max = _bound_tuple(
            binn_construction_params,
            "D_bound",
            D_BOUND,
        )
        self.D_scale = self.D_max
        self.alpha_D_min = self.D_min / self.D_scale
        self.alpha_D_max = self.D_max / self.D_scale
        if self.growth is not None:
            self.G_min, self.G_max = _bound_tuple(
                binn_construction_params,
                "G_bound",
                G_BOUND,
            )
            self.G_scale = self.G_max
            self.alpha_G_min = self.G_min / self.G_scale
            self.alpha_G_max = self.G_max / self.G_scale
        self.K = K

        self.x1_arr = x1
        self.x2_arr = x2
        self.t_arr = t

        self.x1_min, self.x1_max = float(np.min(x1)), float(np.max(x1))
        self.t_min, self.t_max = float(np.min(t)), float(np.max(t))
        if self.dim == 2:
            self.x2_min, self.x2_max = float(np.min(x2)), float(np.max(x2))

        self.IC_weight = 1e0
        self.surface_weight = binn_construction_params.get("surface_weight", 1e0)
        self.pde_weight = binn_construction_params.get("pde_weight", 1e0)
        self.D_weight = self.D_bound_weight
        self.dDdu_weight = self.D_mono_weight
        self.gamma = gamma
        self.GLS_power = float(BNdata_loss_params.get("GLSpow", 0.5))
        self.num_samples = numPDEsamples
        self.diffusion_samples = 20
        self.constraint_samples = int(
            binn_construction_params.get("constraintSamples", self.diffusion_samples)
        )

        if self.growth is not None:
            self.G_weight = self.G_bound_weight
            self.dGdu_weight = self.G_mono_weight

        self.pde_loss_func = pde_loss_func
        self.data_loss_func = data_loss_func
        self.inputs_gen_func = generate_random_inputs

        self.bc_weight = 1e0
        self.num_bcs = 100

        self.epochs = 0
        self.val_batch_it = 0
        self.tr_batch_it = 0
        self.loss_count = 0
        self.pde_losses_all = {}
        self.inputs_all = {}

        self.u_vals = np.linspace(u_min, u_max, self.diffusion_samples)
        self.u_vals_torch = torch.tensor(
            self.u_vals,
            device=device,
            dtype=torch.float32,
        ).reshape(-1, 1)
        self.register_buffer(
            "constraint_u_values",
            torch.linspace(
                float(u_min),
                float(u_max),
                self.constraint_samples,
                device=device,
                dtype=torch.float32,
            ).reshape(-1, 1),
        )

        self.D_true = diffusion_true_func(self.u_vals)
        self.D_true_torch = torch.tensor(self.D_true, device=device, dtype=torch.float32)
        self.G_true = growth_true_func(self.u_vals)
        self.G_true_torch = torch.tensor(self.G_true, device=device, dtype=torch.float32)
        self.diffusion_true_func = diffusion_true_func
        self.diffusion_true_deriv_func = diffusion_true_deriv_func
        self.growth_true_func = growth_true_func

    def forward(self, inputs):
        self.inputs = inputs
        return self.surface_fitter(self.inputs)

    def loss(self, pred, true):
        self.data_loss_val = 0
        self.pde_loss_val = 0
        self.data_loss_val_total = 0
        self.pde_loss_val_total = 0
        self.D_bound_loss = 0
        self.D_mono_loss = 0
        self.G_bound_loss = 0
        self.G_mono_loss = 0
        self.constraint_loss = 0
        self.bc_loss = 0

        inputs = self.inputs
        inputs_rand = self.inputs_gen_func(self, inputs)
        outputs_rand = self.surface_fitter(inputs_rand)

        self.data_loss_val_total = self.data_loss_func(self, pred, true)
        self.pde_loss_val_total += self.pde_loss_func(self, inputs_rand, outputs_rand)

        self.data_loss_val = self.surface_weight * torch.mean(self.data_loss_val_total)
        self.pde_loss_val += self.pde_weight * torch.mean(self.pde_loss_val_total)
        auxiliary_loss = self.constraint_loss + self.bc_loss

        self.loss_count += 1

        return (
            self.data_loss_val + self.pde_loss_val + auxiliary_loss,
            self.data_loss_val,
            self.pde_loss_val,
        )

    def dg_parameters(self):
        if self.diffusion is None:
            return []
        params = list(self.diffusion.parameters())
        if self.growth is not None:
            params += list(self.growth.parameters())
        return params
