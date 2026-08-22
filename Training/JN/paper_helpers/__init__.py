"""Publication-facing helper surface used by the paper notebooks.

Contents
--------
- no top-level functions; re-export module symbols for notebook imports."""

from .file_finder import condense_df, print_path_components
from .paths import dictToPath
from .plot_final_mse_grow import plot_grow_mse_lst_extended
from .plot_final_mse_grow_2d import plot_grow_mse_lst as plot_grow_mse_lst_2d
from .plotting_final_loss import plot_final_loss_lst, plot_final_loss_lst_colLayout
from .plotting_final_mse import plot_diff_mse_lst
from .plotting_final_mse_2d import plot_diff_mse_lst as plot_diff_mse_lst_2d
from .plotting_loss_running_min import plot_running_min_loss_component_broken_x_log_lst
from .plotting_loss_components import (
    plot_running_min_loss_components_broken_x_log_lst,
    plot_running_min_loss_components_seed_broken_x_log_lst,
    plot_smoothed_loss_components_seed_broken_x_log_lst,
)
from .plotting_adam_diagnostics import plot_adam_D_diagnostics
from .plotting_data_loss_diagnostics import plot_data_loss_diagnostics
from .plotting_mse_running_min import plot_running_min_MSE_diff_loss_broken_x_log_lst
from .plotting_mse_running_min_growth import plot_running_min_MSE_grow_loss_broken_x_log_lst
from .plotting_functions_over_epochs import (
    plot_diffusion_across_epochs,
    plot_growth_across_epochs,
)
from .plotting_profiles import (
    plot_binned_density_error_agg_groups_multi_data_mean,
    plot_initial_condition_1d,
)
from .plotting_profiles_2d import plot_initial_condition_2d
from .plotting_timings import plot_total_run_times_lst, plot_total_run_times_lst_colLayout
from .real_space_plotter import (
    plot_eval_D_multi_gray,
    plot_eval_D_multi_gray_gridLayout,
    plot_eval_G_multi_gray,
    symbolic_from_function,
)
from .utils import scale_function_by_percent_error
from .real_space_plotter_2d import (
    plot_eval_D_multi_gray as plot_eval_D_multi_gray_2d,
    plot_eval_G_multi_gray as plot_eval_G_multi_gray_2d,
    true_diffusion_expression,
    true_growth_expression,
)

__all__ = [
    "condense_df",
    "dictToPath",
    "scale_function_by_percent_error",
    "print_path_components",
    "plot_adam_D_diagnostics",
    "plot_binned_density_error_agg_groups_multi_data_mean",
    "plot_data_loss_diagnostics",
    "plot_diff_mse_lst",
    "plot_diff_mse_lst_2d",
    "plot_diffusion_across_epochs",
    "plot_eval_D_multi_gray",
    "plot_eval_D_multi_gray_2d",
    "plot_eval_D_multi_gray_gridLayout",
    "plot_eval_G_multi_gray",
    "plot_eval_G_multi_gray_2d",
    "plot_final_loss_lst",
    "plot_final_loss_lst_colLayout",
    "plot_grow_mse_lst_2d",
    "plot_grow_mse_lst_extended",
    "plot_growth_across_epochs",
    "plot_initial_condition_1d",
    "plot_initial_condition_2d",
    "plot_running_min_MSE_diff_loss_broken_x_log_lst",
    "plot_running_min_MSE_grow_loss_broken_x_log_lst",
    "plot_running_min_loss_component_broken_x_log_lst",
    "plot_running_min_loss_components_broken_x_log_lst",
    "plot_running_min_loss_components_seed_broken_x_log_lst",
    "plot_smoothed_loss_components_seed_broken_x_log_lst",
    "plot_total_run_times_lst",
    "plot_total_run_times_lst_colLayout",
    "symbolic_from_function",
    "true_diffusion_expression",
    "true_growth_expression",
]
