# Utility Module Reference: `utils/medallion/silver/silver_eda_dropped.py`

## Module Purpose

This module tracks sensors or columns removed from Silver EDA and documents the reason for each exclusion.

## Pipeline Role

- Stage support: Silver
- Primary responsibility: This module tracks sensors or columns removed from Silver EDA and documents the reason for each exclusion.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `load_dropped_sensor_dataframe` | Load the dropped-feature dataframe artifact from disk. | deep |
| `attach_state_column_to_dropped_dataframe` | Join state columns from the Silver dataframe onto dropped-feature rows. | deep |
| `build_dropped_sensor_profiles_from_silver_preeda_truth` | Build profile and effect-size tables for quarantined Silver features. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `load_dropped_sensor_dataframe` | `*, dropped_path` | Load the dropped-feature dataframe artifact from disk. |
| `attach_state_column_to_dropped_dataframe` | `dropped_dataframe, *, silver_dataframe, state_column, synthetic_state_column, join_key` | Join state columns from the Silver dataframe onto dropped-feature rows. |
| `build_dropped_sensor_profiles_from_silver_preeda_truth` | `dropped_dataframe, *, dropped_feature_columns, state_column, state_values, baseline_state, comparison_states` | Build profile and effect-size tables for quarantined Silver features. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `FileNotFoundError` for invalid input, missing context, or failed validation paths.
- Source raises `KeyError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because Silver notebooks depend on repeatable profiling and EDA helpers before the modeling-ready Gold layer is built.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
