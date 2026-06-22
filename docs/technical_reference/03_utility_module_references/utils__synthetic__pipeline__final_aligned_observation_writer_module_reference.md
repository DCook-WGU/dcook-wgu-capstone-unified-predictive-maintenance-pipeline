# Utility Module Reference: `utils/synthetic/pipeline/final_aligned_observation_writer.py`

## Module Purpose

This module writes final aligned observation records for the synthetic pipeline.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module writes final aligned observation records for the synthetic pipeline.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_get_existing_columns` | Return existing Postgres columns for the final-aligned table. | deep |
| `_infer_alter_column_type` | Infer a conservative Postgres type for an added final-aligned column. | deep |
| `_add_missing_columns` | Add dataframe columns that are missing from the final-aligned target. | deep |
| `_build_sensor_columns` | Return expected wide sensor columns for final aligned observations. | short |
| `_coalesce_left_then_right` | Prefer left values while filling missing values from the right series. | deep |
| `_require_columns` | Raise a clear error when a dataframe is missing required columns. | deep |
| `_validate_premelt_columns` | Validate premelt columns needed to restore original row context. | deep |
| `_validate_rebuilt_columns` | Validate rebuilt columns needed for final alignment. | deep |
| `load_premelt_for_final_alignment` | Load premelt observation rows that provide original ordering metadata. | deep |
| `load_rebuilt_for_final_alignment` | Load rebuilt observations for final alignment, optionally complete rows only. | deep |
| `build_final_aligned_observations_dataframe` | Merge premelt row context with rebuilt sensor values into final wide rows. | deep |
| `ensure_final_aligned_table_exists` | Create the final-aligned observation table and lookup indexes. | deep |
| `write_final_aligned_observations` | Write final-aligned observations to Postgres with optional table replace. | deep |
| `build_final_aligned_observations_stage` | Build the final-aligned table from premelt and rebuilt stages in windows. | deep |

## Configuration Dependencies

- Dataset, run, stage, or recipe identifiers used for traceability.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_get_existing_columns` | `engine, *, schema, table` | Return existing Postgres columns for the final-aligned table. |
| `_infer_alter_column_type` | `series` | Infer a conservative Postgres type for an added final-aligned column. |
| `_add_missing_columns` | `engine, *, schema, table, dataframe` | Add dataframe columns that are missing from the final-aligned target. |
| `_build_sensor_columns` | `n_sensors` | Return expected wide sensor columns for final aligned observations. |
| `_coalesce_left_then_right` | `left, right` | Prefer left values while filling missing values from the right series. |
| `_require_columns` | `dataframe, required_columns, frame_name` | Raise a clear error when a dataframe is missing required columns. |
| `_validate_premelt_columns` | `dataframe, n_sensors` | Validate premelt columns needed to restore original row context. |
| `_validate_rebuilt_columns` | `dataframe, n_sensors` | Validate rebuilt columns needed for final alignment. |
| `load_premelt_for_final_alignment` | `engine, *, schema, table_name, dataset_id, run_id` | Load premelt observation rows that provide original ordering metadata. |
| `load_rebuilt_for_final_alignment` | `engine, *, schema, table_name, dataset_id, run_id, complete_only` | Load rebuilt observations for final alignment, optionally complete rows only. |
| `build_final_aligned_observations_dataframe` | `*, premelt_dataframe, rebuilt_dataframe, n_sensors, prefer_rebuilt_sensor_values` | Merge premelt row context with rebuilt sensor values into final wide rows. |
| `ensure_final_aligned_table_exists` | `engine, *, schema, table_name` | Create the final-aligned observation table and lookup indexes. |
| `write_final_aligned_observations` | `engine, dataframe, *, schema, table_name, if_exists` | Write final-aligned observations to Postgres with optional table replace. |
| `build_final_aligned_observations_stage` | `engine, *, schema, premelt_table, rebuilt_table, target_table, dataset_id, run_id, n_sensors, complete_only, prefer_rebuilt_sensor_values, if_exists, observation_window_size` | Build the final-aligned table from premelt and rebuilt stages in windows. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.

## Artifact / SQL / File-System Interactions

- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.

## Failure Behavior

- Source raises `TypeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because the synthetic subsystem depends on repeatable staging, truth handling, streaming handoff, or final alignment behavior to make controlled pump telemetry tests explainable.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
