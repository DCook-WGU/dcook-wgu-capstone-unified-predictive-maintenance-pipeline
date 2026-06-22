# Utility Module Reference: `utils/medallion/silver/silver_eda_profiles.py`

## Module Purpose

This module builds Silver EDA profile summaries for sensors, status fields, and analytical subsets.

## Pipeline Role

- Stage support: Silver
- Primary responsibility: This module builds Silver EDA profile summaries for sensors, status fields, and analytical subsets.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `z_score` | Z-score normalize a numeric pandas Series. | medium |
| `build_silver_overview_summary` | Build a quick structural overview of the Silver dataframe. | deep |
| `build_missingness_audit_table` | Build a column-level missingness summary table. | deep |
| `build_duplicate_summary` | Summarize duplicate rows and duplicate key identifiers. | medium |
| `build_numeric_describe_table` | Build a transposed ``describe`` table for numeric columns. | medium |
| `build_categorical_cardinality_table` | Build non-numeric, non-datetime column cardinality statistics. | medium |
| `profile_sensor_state_table` | Build per-sensor descriptive statistics by state value. | deep |
| `build_state_sensor_profile_table` | Build state-by-sensor profile rows using sequence-friendly inputs. | medium |
| `build_feature_behavior_effect_size_table` | Build effect-size-style summary table using standardized mean shift vs baseline state. | deep |
| `build_plot_feature_list` | Build an ordered list of top features from an effect-size table. | medium |

## Configuration Dependencies

- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `z_score` | `series` | Z-score normalize a numeric pandas Series. |
| `build_silver_overview_summary` | `dataframe` | Build a quick structural overview of the Silver dataframe. |
| `build_missingness_audit_table` | `dataframe, *, include_only_nonzero` | Build a column-level missingness summary table. |
| `build_duplicate_summary` | `dataframe` | Summarize duplicate rows and duplicate key identifiers. |
| `build_numeric_describe_table` | `dataframe, *, include_columns` | Build a transposed ``describe`` table for numeric columns. |
| `build_categorical_cardinality_table` | `dataframe, *, exclude_columns` | Build non-numeric, non-datetime column cardinality statistics. |
| `profile_sensor_state_table` | `df, *, sensors, state_values, state_column` | Build per-sensor descriptive statistics by state value. |
| `build_state_sensor_profile_table` | `dataframe, *, sensors, state_column, state_values` | Build state-by-sensor profile rows using sequence-friendly inputs. |
| `build_feature_behavior_effect_size_table` | `dataframe, *, sensors, state_column, baseline_state, comparison_states` | Build effect-size-style summary table using standardized mean shift vs baseline state. |
| `build_plot_feature_list` | `effect_size_df, *, max_features` | Build an ordered list of top features from an effect-size table. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Not determined from available source

## Failure Behavior

- Source raises `KeyError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because Silver notebooks depend on repeatable profiling and EDA helpers before the modeling-ready Gold layer is built.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
