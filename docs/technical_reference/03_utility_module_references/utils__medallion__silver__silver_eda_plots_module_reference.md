# Utility Module Reference: `utils/medallion/silver/silver_eda_plots.py`

## Module Purpose

This module creates Silver EDA plotting outputs used for data inspection and capstone explanation.

## Pipeline Role

- Stage support: Silver
- Primary responsibility: This module creates Silver EDA plotting outputs used for data inspection and capstone explanation.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `plot_correlation_heatmap` | Plot a correlation heatmap and optionally save it to disk. | deep |
| `plot_state_distribution_histograms` | Plot feature distribution histograms grouped by state. | deep |
| `build_flag_spans` | Build contiguous x-axis spans where a binary flag equals one. | deep |
| `resolve_time_axis_series` | Resolve the preferred plotting time axis from available Silver columns. | short |
| `plot_top_feature_overlay` | Plot z-scored feature overlays across a shared time axis. | deep |
| `plot_feature_timeseries_with_flag_spans` | Plot one feature at a time with anomaly spans overlaid. | deep |
| `plot_aligned_onset_series` | Plot the aligned onset mean series for one feature. | deep |

## Configuration Dependencies

- No explicit configuration dependency was determined from available source.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `plot_correlation_heatmap` | `correlation_matrix, *, output_path` | Plot a correlation heatmap and optionally save it to disk. |
| `plot_state_distribution_histograms` | `dataframe, *, features, state_column, state_values, output_dir` | Plot feature distribution histograms grouped by state. |
| `build_flag_spans` | `dataframe, *, flag_column, x_column` | Build contiguous x-axis spans where a binary flag equals one. |
| `resolve_time_axis_series` | `dataframe` | Resolve the preferred plotting time axis from available Silver columns. |
| `plot_top_feature_overlay` | `dataframe, *, features, output_path` | Plot z-scored feature overlays across a shared time axis. |
| `plot_feature_timeseries_with_flag_spans` | `dataframe, *, features, output_dir` | Plot one feature at a time with anomaly spans overlaid. |
| `plot_aligned_onset_series` | `onset_summary_df, *, feature_name, output_path` | Plot the aligned onset mean series for one feature. |

## Side Effects

- Source includes file-write calls; helpers can write configured files or artifacts when those paths are passed by the caller.
- Source includes directory creation; helpers can create configured output directories.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories and file-write operations.

## Failure Behavior

- Not determined from available source

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because Silver notebooks depend on repeatable profiling and EDA helpers before the modeling-ready Gold layer is built.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
