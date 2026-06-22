# Utility Module Reference: `utils/medallion/silver/silver_eda_addons.py`

## Module Purpose

This module adds Silver EDA helper behavior for derived views or supplemental analysis outputs.

## Pipeline Role

- Stage support: Silver
- Primary responsibility: This module adds Silver EDA helper behavior for derived views or supplemental analysis outputs.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_safe_numeric_series` | Coerce a series to numeric values with invalid entries set to NaN. | short |
| `_choose_order_column` | Select the first available ordering column for sequence-based summaries. | deep |
| `_iqr_bounds` | Compute lower bound, upper bound, and IQR for numeric outlier checks. | short |
| `_mad` | Compute median absolute deviation for a numeric series. | short |
| `scalar_to_int` | Convert a non-missing scalar value to ``int``. | deep |
| `_plot_heatmap_from_pivot` | Save a heatmap image from a pivoted dataframe and return the image path. | deep |
| `build_missingness_by_group_table` | Build a long missingness table by group and feature. | deep |
| `build_missingness_group_artifacts` | Build missingness-by-state and missingness-by-episode artifacts. | deep |
| `build_state_transition_artifacts` | Build row-collapsed state transition and dwell artifacts. | deep |
| `build_robust_feature_comparison_artifacts` | Build robust state-comparison statistics and plots for numeric features. | deep |
| `build_pca_diagnostics_artifacts` | Build PCA explained-variance and loading diagnostics. | deep |
| `build_outlier_audit_artifacts` | Build overall and optional state-level outlier summary artifacts. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_safe_numeric_series` | `series` | Coerce a series to numeric values with invalid entries set to NaN. |
| `_choose_order_column` | `dataframe` | Select the first available ordering column for sequence-based summaries. |
| `_iqr_bounds` | `series, iqr_multiplier` | Compute lower bound, upper bound, and IQR for numeric outlier checks. |
| `_mad` | `series` | Compute median absolute deviation for a numeric series. |
| `scalar_to_int` | `value, name` | Convert a non-missing scalar value to ``int``. |
| `_plot_heatmap_from_pivot` | `plot_matrix, *, title, out_path, x_label, y_label, figsize` | Save a heatmap image from a pivoted dataframe and return the image path. |
| `build_missingness_by_group_table` | `dataframe, *, group_column, feature_columns, include_only_features_with_missingness` | Build a long missingness table by group and feature. |
| `build_missingness_group_artifacts` | `dataframe, *, feature_columns, artifacts_dir, state_column, episode_column, top_feature_count_for_heatmap, top_episode_count_for_heatmap` | Build missingness-by-state and missingness-by-episode artifacts. |
| `build_state_transition_artifacts` | `dataframe, *, state_column, artifacts_dir, episode_column, order_column, state_order` | Build row-collapsed state transition and dwell artifacts. |
| `build_robust_feature_comparison_artifacts` | `dataframe, *, feature_columns, state_column, artifacts_dir, baseline_state, comparison_states, plot_features, state_plot_order, max_plot_features` | Build robust state-comparison statistics and plots for numeric features. |
| `build_pca_diagnostics_artifacts` | `dataframe, *, feature_columns, artifacts_dir, sample_row_count, use_robust_scaler, top_loading_count` | Build PCA explained-variance and loading diagnostics. |
| `build_outlier_audit_artifacts` | `dataframe, *, feature_columns, artifacts_dir, state_column, iqr_multiplier, robust_z_threshold, max_plot_features` | Build overall and optional state-level outlier summary artifacts. |

## Side Effects

- Source includes file-write calls; helpers can write configured files or artifacts when those paths are passed by the caller.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories and file-write operations.

## Failure Behavior

- Source raises `KeyError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because Silver notebooks depend on repeatable profiling and EDA helpers before the modeling-ready Gold layer is built.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
