"""Training wrapper used by the BINN pipeline.

Contents
--------
- get_nvidia_smi_output: capture raw `nvidia-smi` output lines for GPU memory inspection.
- parse_gpu_usages: extract per-GPU memory usage values from `nvidia-smi` output.
- pick_lowest_usage_gpu: choose the lowest-memory GPU from an allowed device subset.
- GetLowestGPU: select the preferred compute device across CUDA, MPS, or CPU modes.
- synchronize_if_needed: synchronize CUDA execution before timing-sensitive operations.
- TimeRemaining: estimate remaining runtime from elapsed time and epoch progress.
- ModelWrapper: lightweight training, checkpointing, and diagnostic wrapper around a BINN model.
"""

import math
import os
import random
import subprocess
import time
from datetime import timedelta

import numpy as np
import torch

from Training.binn.python.Modules.Utils.Gradient import Gradient


def get_nvidia_smi_output():
    nvidia_smi = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE)
    nvidia_smi_output = nvidia_smi.communicate()[0].decode("utf8")
    return nvidia_smi_output.split("\n")


def parse_gpu_usages(nvidia_smi_lines):
    usages = []
    for line in nvidia_smi_lines:
        str_idx = line.find("MiB / ")
        if str_idx != -1:
            usages.append(int(line[str_idx - 7 : str_idx]))
    return usages


def pick_lowest_usage_gpu(usages, pick_from):
    gpus_sorted = np.argsort(usages)
    for idx in gpus_sorted:
        if idx in pick_from:
            return "cuda:" + str(idx)
    return "cpu"


def GetLowestGPU(
    pick_from=[0, 1, 2, 3],
    verbose=True,
    return_usages=False,
    mps=False,
    cpu=False,
):
    if cpu:
        if verbose:
            print("Device set to cpu")
        return "cpu"
    if not torch.cuda.is_available() or not pick_from:
        if mps:
            print("Device set to mps")
            return "mps"
        if verbose:
            print("Device set to cpu")
        return "cpu"
    nvidia_smi_lines = get_nvidia_smi_output()
    usages = parse_gpu_usages(nvidia_smi_lines)
    device = pick_lowest_usage_gpu(usages, pick_from)
    if verbose:
        print(" ======================= GPU USAGES ================")
        print("Device set to " + device)
        print("=====================================================")
    if return_usages:
        return device, usages
    return device


def synchronize_if_needed(x):
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    elif x.device.type == "mps":
        torch.mps.synchronize()


def TimeRemaining(
    current_iter,
    total_iter,
    start_time,
    previous_time=None,
    ops_per_iter=1.0,
):
    current_time = time.time()
    elapsed = current_time - start_time
    remaining = total_iter * elapsed / current_iter - elapsed
    ms_per_op = None
    if previous_time is not None:
        ms_per_op = (current_time - previous_time) / ops_per_iter
    elapsed = str(timedelta(seconds=int(elapsed)))
    remaining = str(timedelta(seconds=int(remaining)))
    return elapsed, remaining, ms_per_op


CONSTRAINT_LOSS_ATTRS = (
    ("D_bound_loss", "D_bound_loss"),
    ("D_mono_loss", "D_mono_loss"),
    ("G_bound_loss", "G_bound_loss"),
    ("G_mono_loss", "G_mono_loss"),
    ("constraint_loss", "constraint_loss"),
)


class ModelWrapper:
    """Lightweight helper around a PyTorch model."""

    _CHECKPOINT_STATE_KEYS = (
        "train_loss_list",
        "val_loss_list",
        "train_pde_loss_list",
        "val_pde_loss_list",
        "train_data_loss_list",
        "val_data_loss_list",
        "train_D_bound_loss_list",
        "val_D_bound_loss_list",
        "train_D_mono_loss_list",
        "val_D_mono_loss_list",
        "train_G_bound_loss_list",
        "val_G_bound_loss_list",
        "train_G_mono_loss_list",
        "val_G_mono_loss_list",
        "train_constraint_loss_list",
        "val_constraint_loss_list",
        "adam_D_exp_avg_sq_mean_list",
        "adam_D_exp_avg_sq_max_list",
        "adam_D_effective_lr_mean_list",
        "adam_D_effective_lr_min_list",
        "adam_D_grad_norm_list",
        "train_data_loss_diagnostics",
        "val_data_loss_diagnostics",
        "fixed_grid_pde_loss_list",
        "fixed_grid_pde_epoch_list",
        "best_fixed_grid_pde_loss",
        "fixed_grid_pde_settings",
        "fixed_grid_pde_frequency",
        "fixed_grid_pde_shape",
        "epoch_times",
        "diffusion_errors",
        "growth_errors",
        "diffusion_preds",
        "growth_preds",
        "save_index",
        "best_train_loss",
        "best_val_loss",
        "best_diffusion_pred",
        "best_diffusion_error",
        "best_growth_pred",
        "best_growth_error",
        "last_improved",
        "max_trigger",
        "trigger_list",
        "print_freq",
        "store_constraint_losses",
        "store_adam_diagnostics",
        "store_data_loss_diagnostics",
        "store_fixed_grid_pde_diagnostics",
        "avg_epoch_time",
    )

    _MODEL_RUNTIME_KEYS = ("epochs", "loss_count")

    def __init__(
        self,
        model,
        optimizer,
        loss,
        regularizer=None,
        save_name=None,
        save_best_train=False,
        save_best_val=True,
        save_opt=False,
        save_reg=False,
        seed=0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.regularizer = regularizer
        self.augmentation = None
        self.save_name = save_name
        self.save_best_train = bool(save_name and save_best_train)
        self.save_best_val = bool(save_name and save_best_val)
        self.save_opt = bool(save_name and save_opt)
        self.save_reg = bool(save_name and save_reg)
        self.seed = seed

        self.train_loss_list = []
        self.val_loss_list = []
        self.train_pde_loss_list = []
        self.val_pde_loss_list = []
        self.train_data_loss_list = []
        self.val_data_loss_list = []
        self.train_D_bound_loss_list = []
        self.val_D_bound_loss_list = []
        self.train_D_mono_loss_list = []
        self.val_D_mono_loss_list = []
        self.train_G_bound_loss_list = []
        self.val_G_bound_loss_list = []
        self.train_G_mono_loss_list = []
        self.val_G_mono_loss_list = []
        self.train_constraint_loss_list = []
        self.val_constraint_loss_list = []
        self.adam_D_exp_avg_sq_mean_list = []
        self.adam_D_exp_avg_sq_max_list = []
        self.adam_D_effective_lr_mean_list = []
        self.adam_D_effective_lr_min_list = []
        self.adam_D_grad_norm_list = []
        self.train_data_loss_diagnostics = self._empty_data_loss_diagnostics()
        self.val_data_loss_diagnostics = self._empty_data_loss_diagnostics()
        self.fixed_grid_pde_loss_list = []
        self.fixed_grid_pde_epoch_list = []
        self.best_fixed_grid_pde_loss = None
        self.fixed_grid_pde_frequency = 10
        self.fixed_grid_pde_shape = None
        self.fixed_grid_pde_settings = {}
        self.epoch_times = []

        self.diffusion_errors, self.growth_errors = [], []
        self.diffusion_preds, self.growth_preds = [], []

        self.train_pde_losses_list_spatial = []
        self.train_data_losses_list_spatial = []
        self.x_train_list = []
        self.loss_count_list = []
        self.save_index = []

        self.train = False
        self.val = False
        self.store_constraint_losses = True
        self.store_adam_diagnostics = False
        self.store_data_loss_diagnostics = False
        self.store_fixed_grid_pde_diagnostics = False

        if self.seed is not None:
            self.set_seed(self.seed)

    def fit(
        self,
        x_tr_input,
        y_tr_input,
        *,
        batch_size=None,
        epochs=1,
        verbose=1,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_freq=1,
        early_stopping=None,
        include_val_aug=False,
        include_val_reg=False,
        lr_dec_epoch=None,
        lr_dec_prop=1.0,
        rel_update_thresh=0.01,
        rel_save_thresh=0.01,
        print_freq=100,
        store_constraint_losses=True,
        store_adam_diagnostics=False,
        store_data_loss_diagnostics=False,
        store_fixed_grid_pde_diagnostics=False,
        fixed_grid_pde_frequency=10,
        fixed_grid_pde_shape=None,
    ):
        if self.seed is not None:
            self.set_seed(self.seed)

        self.early_stopping = early_stopping

        if batch_size is None:
            batch_size = len(x_tr_input)
        train_batches_per_epoch = max(1, math.ceil(len(x_tr_input) / batch_size))

        if validation_data is not None:
            x_val, y_val = validation_data
            val_batch_size = batch_size
            val_batches_per_epoch = max(1, math.ceil(len(x_val) / val_batch_size))

        self.best_train_loss = getattr(self, "best_train_loss", float("inf"))
        self.best_val_loss = getattr(self, "best_val_loss", float("inf"))
        self.last_improved = getattr(self, "last_improved", 0)
        self.max_trigger = getattr(self, "max_trigger", 0)
        self.trigger_list = getattr(self, "trigger_list", [])
        global_start_time = time.time()
        self.print_freq = print_freq
        self.store_constraint_losses = bool(store_constraint_losses)
        self.store_adam_diagnostics = bool(store_adam_diagnostics)
        self.store_data_loss_diagnostics = bool(store_data_loss_diagnostics)
        self.store_fixed_grid_pde_diagnostics = bool(store_fixed_grid_pde_diagnostics)
        self.fixed_grid_pde_frequency = max(1, int(fixed_grid_pde_frequency))
        self.fixed_grid_pde_shape = fixed_grid_pde_shape
        self.fixed_grid_pde_settings = {
            "frequency": self.fixed_grid_pde_frequency,
            "shape": self.fixed_grid_pde_shape,
        }
        if not isinstance(getattr(self, "train_data_loss_diagnostics", None), dict):
            self.train_data_loss_diagnostics = self._empty_data_loss_diagnostics()
        if not isinstance(getattr(self, "val_data_loss_diagnostics", None), dict):
            self.val_data_loss_diagnostics = self._empty_data_loss_diagnostics()
        if not isinstance(getattr(self, "fixed_grid_pde_loss_list", None), list):
            self.fixed_grid_pde_loss_list = []
        if not isinstance(getattr(self, "fixed_grid_pde_epoch_list", None), list):
            self.fixed_grid_pde_epoch_list = []
        self.avg_epoch_time = None
        trigger = self.model.epochs - self.last_improved

        for epoch in range(initial_epoch, initial_epoch + epochs):
            trigger = self.model.epochs - self.last_improved
            if trigger > self.max_trigger:
                self.max_trigger = trigger

            if early_stopping is not None and trigger >= early_stopping:
                print("\n\nEarly stopping: no improvement.")
                self.save(f"{self.save_name}_ES")
                print(f"Saved model with early stopping at epoch {self.model.epochs}")
                break

            self.train, self.val = True, False
            epoch_start_time = time.time()
            self.model.train()

            if shuffle:
                torch.manual_seed(self.model.epochs)
                perm = torch.randperm(len(x_tr_input), device=x_tr_input.device)
            else:
                perm = torch.arange(len(x_tr_input), device=x_tr_input.device)
            x_tr = x_tr_input[perm].detach()
            y_tr = y_tr_input[perm].detach()

            epoch_train_losses = []
            epoch_train_pde = []
            epoch_train_data = []
            epoch_train_constraints = (
                {name: [] for _, name in CONSTRAINT_LOSS_ATTRS}
                if self.store_constraint_losses
                else None
            )
            epoch_D_grad_norms = [] if self.store_adam_diagnostics else None
            for batch_idx in range(train_batches_per_epoch):
                if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                    break

                start = batch_idx * batch_size
                stop = (
                    (batch_idx + 1) * batch_size
                    if batch_idx + 1 < train_batches_per_epoch
                    else len(x_tr)
                )

                x_true = x_tr[start:stop]
                y_true = y_tr[start:stop]

                x_true.requires_grad_(True)
                self.optimizer.zero_grad(set_to_none=True)

                y_pred = self.model(x_true)

                losses = self.loss(y_pred, y_true)
                task_loss, data_loss, pde_loss = losses[:3]

                reg_loss = (
                    self.regularizer(self.model, x_true, y_true, y_pred)
                    if self.regularizer is not None
                    else 0.0
                )
                total_loss = task_loss + reg_loss
                total_loss.backward()
                if self.store_adam_diagnostics:
                    epoch_D_grad_norms.append(
                        self._parameter_grad_norm(self._diffusion_parameters())
                    )
                self.optimizer.step()

                epoch_train_losses.append(total_loss.detach())
                epoch_train_pde.append(pde_loss.detach())
                epoch_train_data.append(data_loss.detach())
                if self.store_constraint_losses:
                    for model_attr, loss_name in CONSTRAINT_LOSS_ATTRS:
                        epoch_train_constraints[loss_name].append(
                            self._mean_model_loss_attr(model_attr, total_loss).detach()
                        )

                if hasattr(self.model, "epochs") and self.model.epochs % 10 == 0:
                    self.save_index.append(self.model.loss_count - 1)

            train_loss_epoch = torch.mean(torch.stack(epoch_train_losses)).item()
            train_pde_epoch = torch.mean(torch.stack(epoch_train_pde)).item()
            train_data_epoch = torch.mean(torch.stack(epoch_train_data)).item()

            self.train_loss_list.append(train_loss_epoch)
            self.train_pde_loss_list.append(train_pde_epoch)
            self.train_data_loss_list.append(train_data_epoch)
            if self.store_data_loss_diagnostics:
                self._append_data_loss_diagnostics(
                    self.train_data_loss_diagnostics,
                    x_tr_input,
                    y_tr_input,
                )
            
            if self.store_constraint_losses:
                train_constraint_epochs = {
                    loss_name: torch.mean(torch.stack(values)).item()
                    for loss_name, values in epoch_train_constraints.items()
                }
                self.train_D_bound_loss_list.append(train_constraint_epochs["D_bound_loss"])
                self.train_D_mono_loss_list.append(train_constraint_epochs["D_mono_loss"])
                self.train_G_bound_loss_list.append(train_constraint_epochs["G_bound_loss"])
                self.train_G_mono_loss_list.append(train_constraint_epochs["G_mono_loss"])
                self.train_constraint_loss_list.append(train_constraint_epochs["constraint_loss"])
            if self.store_adam_diagnostics:
                adam_D_stats = self._adam_conditioning_stats(self._diffusion_parameters())
                self.adam_D_exp_avg_sq_mean_list.append(adam_D_stats["exp_avg_sq_mean"])
                self.adam_D_exp_avg_sq_max_list.append(adam_D_stats["exp_avg_sq_max"])
                self.adam_D_effective_lr_mean_list.append(adam_D_stats["effective_lr_mean"])
                self.adam_D_effective_lr_min_list.append(adam_D_stats["effective_lr_min"])
                self.adam_D_grad_norm_list.append(float(np.mean(epoch_D_grad_norms)))
            save_best_val_checkpoint = False

            if validation_data is not None and (epoch % validation_freq == 0):
                self._validate(
                    x_val,
                    y_val,
                    val_batches_per_epoch,
                    val_batch_size,
                    validation_steps,
                    include_val_aug,
                    include_val_reg,
                )
                if self.store_data_loss_diagnostics:
                    self._append_data_loss_diagnostics(
                        self.val_data_loss_diagnostics,
                        x_val,
                        y_val,
                    )

                val_loss_epoch = self.val_loss_list[-1]
                previous_best = self.best_val_loss
                if val_loss_epoch < previous_best * (1 - rel_update_thresh):
                    self.best_val_loss = val_loss_epoch
                    self.last_improved = self.model.epochs
                    self._record_best_function_estimates()

                    if (
                        self.save_best_val
                        and self.save_name
                        and val_loss_epoch < previous_best * (1 - rel_save_thresh)
                    ):
                        save_best_val_checkpoint = True

            if validation_data is None:
                train_loss_epoch = self.train_loss_list[-1]
                previous_best = self.best_val_loss
                if train_loss_epoch < previous_best * (1 - rel_update_thresh):
                    self.best_val_loss = train_loss_epoch
                    self.last_improved = self.model.epochs
                    self._record_best_function_estimates()

                    if (
                        self.save_best_val
                        and self.save_name
                        and train_loss_epoch < previous_best * (1 - rel_save_thresh)
                    ):
                        save_best_val_checkpoint = True

            if self.model.epochs % 10 == 0:
                self._record_diagnostics()
            if (
                self.store_fixed_grid_pde_diagnostics
                and self.model.epochs % self.fixed_grid_pde_frequency == 0
            ):
                self._record_fixed_grid_pde_diagnostic()

            if verbose == 1 and epoch % self.print_freq == 0:
                synchronize_if_needed(x_tr)

            if verbose == 1 and epoch % self.print_freq == 0:
                elapsed, remaining, _ = TimeRemaining(
                    current_iter=self.model.epochs + 1,
                    total_iter=initial_epoch + epochs,
                    start_time=global_start_time,
                    previous_time=epoch_start_time,
                    ops_per_iter=batch_size,
                )
                msg = (
                    f"\rEpoch {self.model.epochs + 1}/{initial_epoch + epochs} | "
                    f"Train loss: {train_loss_epoch:1.4e}"
                )
                if validation_data is not None:
                    msg += f" | Val loss: {self.val_loss_list[-1]:1.4e}"
                msg += f" | Remaining: {remaining}        "
                msg += f" | Trigger = {trigger}"
                msg += f" | Elapsed = {epoch_start_time-global_start_time:.1f} s"
                msg += f" | Max trigger = {self.max_trigger}"
                if self.diffusion_errors:
                    msg += f" | D error ={self.diffusion_errors[-1]:.3e}"
                if self.model.growth and self.growth_errors:
                    msg += f" | G error ={self.growth_errors[-1]:.3e}"

                print(msg, end="\r", flush=True)

            self.epoch_times.append(time.time() - epoch_start_time)
            if save_best_val_checkpoint:
                self.save(f"{self.save_name}_best_val")

            if hasattr(self.model, "epochs"):
                self.model.epochs += 1

        if early_stopping is None or trigger < early_stopping:
            print("\nNumber of epochs to train finished rather than early stopping.")
            self.save(f"{self.save_name}_expired")
            print(f"Saved model at total trained epochs {self.model.epochs}")

        if verbose == 1:
            print("\nTraining finished.")
            print(f"\nTotal epochs trained = {self.model.epochs}")
            if hasattr(self, "best_diffusion_error"):
                print(f"\nBest D error ={self.best_diffusion_error:.3e}")
            if self.model.growth and hasattr(self, "best_growth_error"):
                print(f"\nBest G error ={self.best_growth_error:.3e}")
            print(f"\nBest val loss = {self.best_val_loss:.3e}")

    def _validate(
        self,
        x_val,
        y_val,
        val_batches_per_epoch,
        val_batch_size,
        validation_steps,
        include_val_aug,
        include_val_reg,
    ):
        self.model.eval()

        val_loss_acc = 0.0
        val_pde_acc = 0.0
        val_data_acc = 0.0
        val_reg_acc = 0.0
        val_constraint_acc = (
            {name: 0.0 for _, name in CONSTRAINT_LOSS_ATTRS}
            if self.store_constraint_losses
            else None
        )

        for idx in range(val_batches_per_epoch):
            if validation_steps is not None and idx >= validation_steps:
                break

            start = idx * val_batch_size
            stop = (
                (idx + 1) * val_batch_size
                if idx + 1 < val_batches_per_epoch
                else len(x_val)
            )
            x_true = x_val[start:stop].clone()
            y_true = y_val[start:stop].clone()

            if include_val_aug and self.augmentation is not None:
                x_true, y_true = self.augmentation(x_true, y_true)

            x_true.requires_grad_(True)
            y_pred = self.model(x_true)

            losses = self.loss(y_pred, y_true)
            val_loss_acc += losses[0]
            val_data_acc += losses[1]
            val_pde_acc += losses[2]
            if self.store_constraint_losses:
                for model_attr, loss_name in CONSTRAINT_LOSS_ATTRS:
                    val_constraint_acc[loss_name] += self._mean_model_loss_attr(
                        model_attr,
                        losses[0],
                    )

            if include_val_reg and self.regularizer is not None:
                val_reg_acc += self.regularizer(self.model, x_true, y_true, y_pred)

        val_loss = val_loss_acc / val_batches_per_epoch
        val_data = val_data_acc / val_batches_per_epoch
        val_pde = val_pde_acc / val_batches_per_epoch

        self.val_loss_list.append(val_loss.item() if torch.is_tensor(val_loss) else val_loss)
        self.val_data_loss_list.append(val_data.item() if torch.is_tensor(val_data) else val_data)
        self.val_pde_loss_list.append(val_pde.item() if torch.is_tensor(val_pde) else val_pde)
        if self.store_constraint_losses:
            val_constraints = {
                loss_name: value / val_batches_per_epoch
                for loss_name, value in val_constraint_acc.items()
            }
            self.val_D_bound_loss_list.append(
                self._to_item(val_constraints["D_bound_loss"])
            )
            self.val_D_mono_loss_list.append(self._to_item(val_constraints["D_mono_loss"]))
            self.val_G_bound_loss_list.append(
                self._to_item(val_constraints["G_bound_loss"])
            )
            self.val_G_mono_loss_list.append(self._to_item(val_constraints["G_mono_loss"]))
            self.val_constraint_loss_list.append(
                self._to_item(val_constraints["constraint_loss"])
            )

    def _mean_model_loss_attr(self, attr_name, like):
        value = getattr(self.model, attr_name, None)
        if value is None:
            if torch.is_tensor(like):
                return torch.zeros((), device=like.device, dtype=like.dtype)
            return torch.tensor(0.0)
        if not torch.is_tensor(value):
            if torch.is_tensor(like):
                return torch.as_tensor(value, device=like.device, dtype=like.dtype)
            return torch.as_tensor(value)
        return torch.mean(value)

    @staticmethod
    def _to_item(value):
        return value.item() if torch.is_tensor(value) else value

    @staticmethod
    def _empty_data_loss_diagnostics():
        return {
            "epoch": [],
            "mse_mean": [],
            "gls_pred_mean": [],
            "gls_pred_normalised": [],
            "gls_true_mean": [],
            "gls_true_normalised": [],
            "gls_pred_to_mse": [],
            "gls_true_to_mse": [],
            "weight_pred_min": [],
            "weight_pred_q50": [],
            "weight_pred_q90": [],
            "weight_pred_q99": [],
            "weight_pred_max": [],
            "weight_true_min": [],
            "weight_true_q50": [],
            "weight_true_q90": [],
            "weight_true_q99": [],
            "weight_true_max": [],
            "weight_pred_neff": [],
            "weight_pred_neff_frac": [],
            "weight_true_neff": [],
            "weight_true_neff_frac": [],
            "pred_min": [],
            "pred_q01": [],
            "pred_q50": [],
            "pred_max": [],
            "true_min": [],
            "true_q01": [],
            "true_q50": [],
            "true_max": [],
            "pred_lt_1e-2_frac": [],
            "pred_lt_1e-3_frac": [],
            "pred_lt_1e-4_frac": [],
            "mse_top1_frac": [],
            "mse_top5_frac": [],
            "gls_pred_top1_frac": [],
            "gls_pred_top5_frac": [],
            "gls_true_top1_frac": [],
            "gls_true_top5_frac": [],
            "actual_data_loss_top1_frac": [],
            "actual_data_loss_top5_frac": [],
        }

    @staticmethod
    def _append_scalar_diagnostic(diagnostics, key, value):
        diagnostics.setdefault(key, []).append(float(value))

    @staticmethod
    def _quantile(values, q):
        if values.numel() == 0:
            return float("nan")
        return torch.quantile(values, q).item()

    @staticmethod
    def _effective_sample_size(weights):
        weights = weights.reshape(-1)
        denom = torch.sum(weights.pow(2))
        if weights.numel() == 0 or denom <= 0:
            return float("nan")
        return (torch.sum(weights).pow(2) / denom).item()

    @staticmethod
    def _top_fraction(values, percent):
        values = values.reshape(-1).abs()
        total = torch.sum(values)
        if values.numel() == 0 or total <= 0:
            return 0.0
        k = max(1, int(math.ceil(values.numel() * percent / 100.0)))
        return (torch.topk(values, k).values.sum() / total).item()

    def _append_data_loss_diagnostics(self, diagnostics, x_values, y_values):
        if diagnostics is None:
            return

        if not isinstance(diagnostics, dict):
            diagnostics = self._empty_data_loss_diagnostics()

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            x_eval = x_values.detach()
            y_true = y_values.detach()
            y_pred = self.model(x_eval).detach()

            pred_abs = y_pred.abs().clamp(min=1e-10)
            true_abs = y_true.abs().clamp(min=1e-10)
            residual = (y_pred - y_true).pow(2)
            gamma = float(getattr(self.model, "gamma", 0.0))

            weight_pred = pred_abs.pow(-2.0 * gamma)
            weight_true = true_abs.pow(-2.0 * gamma)
            gls_pred = weight_pred * residual
            gls_true = weight_true * residual
            data_loss_func = getattr(self.model, "data_loss_func", None)
            actual_data_loss = (
                data_loss_func(self.model, y_pred, y_true).detach()
                if callable(data_loss_func)
                else residual
            )

            n_values = max(1, y_true.numel())
            mse_mean = torch.mean(residual).item()
            gls_pred_mean = torch.mean(gls_pred).item()
            gls_true_mean = torch.mean(gls_true).item()
            gls_pred_weight_sum = torch.sum(weight_pred)
            gls_true_weight_sum = torch.sum(weight_true)
            gls_pred_normalised = (
                torch.sum(gls_pred) / gls_pred_weight_sum
                if gls_pred_weight_sum > 0
                else torch.tensor(float("nan"), device=y_true.device)
            )
            gls_true_normalised = (
                torch.sum(gls_true) / gls_true_weight_sum
                if gls_true_weight_sum > 0
                else torch.tensor(float("nan"), device=y_true.device)
            )

            diagnostics.setdefault("epoch", []).append(int(getattr(self.model, "epochs", 0)))
            for key, value in (
                ("mse_mean", mse_mean),
                ("gls_pred_mean", gls_pred_mean),
                ("gls_pred_normalised", gls_pred_normalised.item()),
                ("gls_true_mean", gls_true_mean),
                ("gls_true_normalised", gls_true_normalised.item()),
                ("gls_pred_to_mse", gls_pred_mean / max(mse_mean, 1e-300)),
                ("gls_true_to_mse", gls_true_mean / max(mse_mean, 1e-300)),
                ("weight_pred_min", torch.min(weight_pred).item()),
                ("weight_pred_q50", self._quantile(weight_pred, 0.50)),
                ("weight_pred_q90", self._quantile(weight_pred, 0.90)),
                ("weight_pred_q99", self._quantile(weight_pred, 0.99)),
                ("weight_pred_max", torch.max(weight_pred).item()),
                ("weight_true_min", torch.min(weight_true).item()),
                ("weight_true_q50", self._quantile(weight_true, 0.50)),
                ("weight_true_q90", self._quantile(weight_true, 0.90)),
                ("weight_true_q99", self._quantile(weight_true, 0.99)),
                ("weight_true_max", torch.max(weight_true).item()),
                ("weight_pred_neff", self._effective_sample_size(weight_pred)),
                (
                    "weight_pred_neff_frac",
                    self._effective_sample_size(weight_pred) / n_values,
                ),
                ("weight_true_neff", self._effective_sample_size(weight_true)),
                (
                    "weight_true_neff_frac",
                    self._effective_sample_size(weight_true) / n_values,
                ),
                ("pred_min", torch.min(y_pred).item()),
                ("pred_q01", self._quantile(y_pred.reshape(-1), 0.01)),
                ("pred_q50", self._quantile(y_pred.reshape(-1), 0.50)),
                ("pred_max", torch.max(y_pred).item()),
                ("true_min", torch.min(y_true).item()),
                ("true_q01", self._quantile(y_true.reshape(-1), 0.01)),
                ("true_q50", self._quantile(y_true.reshape(-1), 0.50)),
                ("true_max", torch.max(y_true).item()),
                ("pred_lt_1e-2_frac", torch.mean((y_pred < 1e-2).float()).item()),
                ("pred_lt_1e-3_frac", torch.mean((y_pred < 1e-3).float()).item()),
                ("pred_lt_1e-4_frac", torch.mean((y_pred < 1e-4).float()).item()),
                ("mse_top1_frac", self._top_fraction(residual, 1.0)),
                ("mse_top5_frac", self._top_fraction(residual, 5.0)),
                ("gls_pred_top1_frac", self._top_fraction(gls_pred, 1.0)),
                ("gls_pred_top5_frac", self._top_fraction(gls_pred, 5.0)),
                ("gls_true_top1_frac", self._top_fraction(gls_true, 1.0)),
                ("gls_true_top5_frac", self._top_fraction(gls_true, 5.0)),
                ("actual_data_loss_top1_frac", self._top_fraction(actual_data_loss, 1.0)),
                ("actual_data_loss_top5_frac", self._top_fraction(actual_data_loss, 5.0)),
            ):
                self._append_scalar_diagnostic(diagnostics, key, value)

        if was_training:
            self.model.train()

    def _fixed_grid_shape(self):
        shape = self.fixed_grid_pde_shape
        if shape is None:
            return (80, 80) if getattr(self.model, "dim", 1) == 1 else (25, 25, 25)
        if isinstance(shape, int):
            return (shape, shape) if getattr(self.model, "dim", 1) == 1 else (shape, shape, shape)
        return tuple(int(value) for value in shape)

    def _fixed_grid_pde_inputs(self):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        shape = self._fixed_grid_shape()

        if getattr(self.model, "dim", 1) == 1:
            if len(shape) != 2:
                raise ValueError("1D fixed_grid_pde_shape must be (x_points, t_points).")
            x = torch.linspace(
                self.model.x1_min,
                self.model.x1_max,
                shape[0],
                device=device,
                dtype=dtype,
            )
            t = torch.linspace(
                self.model.t_min,
                self.model.t_max,
                shape[1],
                device=device,
                dtype=dtype,
            )
            xx, tt = torch.meshgrid(x, t, indexing="ij")
            inputs = torch.stack((xx.reshape(-1), tt.reshape(-1)), dim=1)
        else:
            if len(shape) != 3:
                raise ValueError(
                    "2D fixed_grid_pde_shape must be (x1_points, x2_points, t_points)."
                )
            x1 = torch.linspace(
                self.model.x1_min,
                self.model.x1_max,
                shape[0],
                device=device,
                dtype=dtype,
            )
            x2 = torch.linspace(
                self.model.x2_min,
                self.model.x2_max,
                shape[1],
                device=device,
                dtype=dtype,
            )
            t = torch.linspace(
                self.model.t_min,
                self.model.t_max,
                shape[2],
                device=device,
                dtype=dtype,
            )
            xx1, xx2, tt = torch.meshgrid(x1, x2, t, indexing="ij")
            inputs = torch.stack(
                (xx1.reshape(-1), xx2.reshape(-1), tt.reshape(-1)),
                dim=1,
            )

        return inputs.detach().clone().requires_grad_(True)

    def _fixed_grid_pde_residual_loss(self):
        was_training = self.model.training
        self.model.eval()

        inputs = self._fixed_grid_pde_inputs()
        outputs = self.model.surface_fitter(inputs)

        if getattr(self.model, "perfect_pde", False):
            if getattr(self.model, "dim", 1) == 1:
                u = outputs.clone()
                d1 = Gradient(u, inputs, order=1)
                ux = d1[:, 0:1]
                ut = d1[:, 1:2]
                uxx = Gradient(ux, inputs, order=1)[:, 0:1]
                lap_u = uxx
                grad_u_sq = ux.pow(2)
            else:
                u = outputs.clone()
                d1 = Gradient(u, inputs, order=1)
                ux1, ux2, ut = d1[:, 0:1], d1[:, 1:2], d1[:, 2:3]
                uxx1 = Gradient(ux1, inputs, order=1)[:, 0:1]
                uxx2 = Gradient(ux2, inputs, order=1)[:, 1:2]
                lap_u = uxx1 + uxx2
                grad_u_sq = ux1.pow(2) + ux2.pow(2)

            D = torch.as_tensor(
                self.model.diffusion_true_func(u.detach().cpu().numpy()),
                device=u.device,
                dtype=u.dtype,
            )
            dDdu = torch.as_tensor(
                self.model.diffusion_true_deriv_func(u.detach().cpu().numpy()),
                device=u.device,
                dtype=u.dtype,
            )
            rhs = D * lap_u + dDdu * grad_u_sq
            if getattr(self.model, "growth_true_func", None) is not None:
                G = torch.as_tensor(
                    self.model.growth_true_func(u.detach().cpu().numpy()),
                    device=u.device,
                    dtype=u.dtype,
                )
                rhs = rhs + G * u

            loss = torch.mean((ut - rhs).pow(2))

            if was_training:
                self.model.train()

            return float(loss.detach().cpu())

        if getattr(self.model, "dim", 1) == 1:
            t = inputs[:, 1:2]
            u = outputs.clone()
            d1 = Gradient(u, inputs, order=1)
            ux = d1[:, 0:1]
            ut = d1[:, 1:2]

            D = (
                self.model.diffusion(u)
                if self.model.diffusion.inputs == 1
                else self.model.diffusion(u, t)
            )
            div = Gradient(D * ux, inputs)[:, 0:1]
        else:
            t = inputs[:, 2:3]
            u = outputs.clone()
            d1 = Gradient(u, inputs, order=1)
            ux1, ux2, ut = d1[:, 0:1], d1[:, 1:2], d1[:, 2:3]

            D = (
                self.model.diffusion(u)
                if self.model.diffusion.inputs == 1
                else self.model.diffusion(u, t)
            )
            div = Gradient(D * ux1, inputs)[:, 0:1] + Gradient(D * ux2, inputs)[:, 1:2]

        if self.model.growth is not None:
            G = (
                self.model.growth(u)
                if self.model.growth.inputs == 1
                else self.model.growth(u, t)
            )
            rhs = self.model.D_max * div + self.model.G_max * G * u
        else:
            rhs = self.model.D_max * div

        loss = torch.mean((ut - rhs).pow(2))

        if was_training:
            self.model.train()

        return float(loss.detach().cpu())

    def _record_fixed_grid_pde_diagnostic(self):
        self.fixed_grid_pde_epoch_list.append(int(getattr(self.model, "epochs", 0)))
        self.fixed_grid_pde_loss_list.append(self._fixed_grid_pde_residual_loss())

    def _diffusion_parameters(self):
        diffusion = getattr(self.model, "diffusion", None)
        if diffusion is None:
            return []
        return list(diffusion.parameters())

    @staticmethod
    def _parameter_grad_norm(parameters):
        squared_norm = 0.0
        for param in parameters:
            if param.grad is None:
                continue
            squared_norm += float(torch.sum(param.grad.detach().pow(2)).cpu())
        return math.sqrt(squared_norm)

    def _adam_conditioning_stats(self, parameters):
        exp_avg_sq_values = []
        effective_lr_values = []

        if self.optimizer is None:
            return {
                "exp_avg_sq_mean": np.nan,
                "exp_avg_sq_max": np.nan,
                "effective_lr_mean": np.nan,
                "effective_lr_min": np.nan,
            }

        param_group_by_param = {}
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                param_group_by_param[param] = group

        for param in parameters:
            state = self.optimizer.state.get(param, {})
            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg_sq is None:
                continue

            group = param_group_by_param.get(param, {})
            lr = group.get("lr", np.nan)
            eps = group.get("eps", 1e-8)
            beta2 = group.get("betas", (0.9, 0.999))[1]
            step = state.get("step", 0)
            if torch.is_tensor(step):
                step = int(step.item())
            step = max(int(step), 1)

            exp_avg_sq_detached = exp_avg_sq.detach()
            exp_avg_sq_values.append(exp_avg_sq_detached.reshape(-1))

            bias_correction2 = 1.0 - beta2 ** step
            exp_avg_sq_hat = exp_avg_sq_detached / bias_correction2
            effective_lr = lr / (torch.sqrt(exp_avg_sq_hat) + eps)
            effective_lr_values.append(effective_lr.reshape(-1))

        if not exp_avg_sq_values:
            return {
                "exp_avg_sq_mean": np.nan,
                "exp_avg_sq_max": np.nan,
                "effective_lr_mean": np.nan,
                "effective_lr_min": np.nan,
            }

        exp_avg_sq_all = torch.cat(exp_avg_sq_values).detach().cpu()
        effective_lr_all = torch.cat(effective_lr_values).detach().cpu()
        return {
            "exp_avg_sq_mean": float(torch.mean(exp_avg_sq_all)),
            "exp_avg_sq_max": float(torch.max(exp_avg_sq_all)),
            "effective_lr_mean": float(torch.mean(effective_lr_all)),
            "effective_lr_min": float(torch.min(effective_lr_all)),
        }

    def _function_estimates(self):
        if getattr(self.model, "diffusion", None) is None:
            return None, 0.0, None, 0.0

        with torch.no_grad():
            diff_pred = self.model.D_scale * self.model.diffusion(self.model.u_vals_torch).flatten()
            diffusion_error = torch.mean((self.model.D_true_torch - diff_pred) ** 2).item()

            growth_pred = None
            growth_error = None
            if self.model.growth:
                growth_pred = self.model.G_scale * self.model.growth(self.model.u_vals_torch).flatten()
                growth_error = torch.mean((self.model.G_true_torch - growth_pred) ** 2).item()

        return diff_pred, diffusion_error, growth_pred, growth_error

    def _record_best_function_estimates(self):
        diff_pred, diffusion_error, growth_pred, growth_error = self._function_estimates()
        if diff_pred is not None:
            self.best_diffusion_pred = diff_pred.detach().cpu()
            self.best_diffusion_error = diffusion_error
        if self.model.growth and growth_pred is not None:
            self.best_growth_pred = growth_pred.detach().cpu()
            self.best_growth_error = growth_error
        if getattr(self, "store_fixed_grid_pde_diagnostics", False):
            self.best_fixed_grid_pde_loss = self._fixed_grid_pde_residual_loss()

    def _record_diagnostics(self):
        diff_pred, diffusion_error, growth_pred, growth_error = self._function_estimates()
        if diff_pred is not None:
            self.diffusion_errors.append(diffusion_error)
            self.diffusion_preds.append(diff_pred.detach().cpu())

        if self.model.growth and growth_pred is not None:
            self.growth_errors.append(growth_error)
            self.growth_preds.append(growth_pred.detach().cpu())

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def save(self, save_name):
        if not save_name:
            return
        torch.save(self._checkpoint(), save_name + ".pth")

    def load_best_val(self, device=None):
        if self.save_name:
            best_path = f"{self.save_name}_best_val.pth"
            if os.path.exists(best_path):
                self._load_checkpoint(best_path, device=device)

    def load_ES(self):
        if self.save_name:
            es_path = f"{self.save_name}_ES.pth"
            if os.path.exists(es_path):
                self._load_checkpoint(es_path)

    def load_expired(self):
        if self.save_name:
            expired_path = f"{self.save_name}_expired.pth"
            if os.path.exists(expired_path):
                self._load_checkpoint(expired_path)

    def _checkpoint(self):
        optimizer_state = None
        if self.optimizer is not None:
            optimizer_state = self.optimizer.state_dict()

        return {
            "format": "ModelWrapperCheckpoint",
            "version": 1,
            "model_state_dict": self._checkpoint_value(self.model.state_dict()),
            "optimizer_state_dict": self._checkpoint_value(optimizer_state),
            "wrapper_state": {
                key: self._checkpoint_value(getattr(self, key))
                for key in self._CHECKPOINT_STATE_KEYS
                if hasattr(self, key)
            },
            "model_runtime_state": {
                key: getattr(self.model, key)
                for key in self._MODEL_RUNTIME_KEYS
                if hasattr(self.model, key)
            },
        }

    def _load_checkpoint(self, checkpoint_path, device=None):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if isinstance(checkpoint, dict) and checkpoint.get("format") == "ModelWrapperCheckpoint":
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is not None and self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state)
                self._move_optimizer_state_to_model_device()
            for key, value in checkpoint.get("wrapper_state", {}).items():
                setattr(self, key, value)
            for key, value in checkpoint.get("model_runtime_state", {}).items():
                setattr(self.model, key, value)
            return

        self._load_legacy_wrapper(checkpoint)

    def _move_optimizer_state_to_model_device(self):
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            return

        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def _load_legacy_wrapper(self, loaded):
        preserve = {
            key: getattr(self, key)
            for key in (
                "save_name",
                "model_save_dir",
                "binnModelLabel",
                "batch_size",
                "early_stopping",
                "rel_update_thresh",
                "rel_save_thresh",
                "print_freq",
                "x_train",
                "y_train",
                "x_val",
                "y_val",
                "x_train_torch",
                "y_train_torch",
                "validation_data",
                "verbose",
            )
            if hasattr(self, key)
        }
        self.__dict__.update(loaded.__dict__)
        for key, value in preserve.items():
            setattr(self, key, value)

    @staticmethod
    def _checkpoint_value(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, list):
            return [ModelWrapper._checkpoint_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(ModelWrapper._checkpoint_value(item) for item in value)
        if isinstance(value, dict):
            return {
                key: ModelWrapper._checkpoint_value(item)
                for key, item in value.items()
            }
        return value
