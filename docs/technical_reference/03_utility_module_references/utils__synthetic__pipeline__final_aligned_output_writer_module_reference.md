# Utility Module Reference: `utils/synthetic/pipeline/final_aligned_output_writer.py`

## Module Purpose

This module exports final aligned synthetic pipeline outputs for downstream use.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module exports final aligned synthetic pipeline outputs for downstream use.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_build_sensor_columns` | Return expected wide sensor columns for final synthetic output. | short |
| `_get_existing_columns` | Return existing Postgres columns for the final synthetic output table. | deep |
| `_infer_alter_column_type` | Infer a conservative Postgres type for an added final-output column. | deep |
| `_add_missing_columns` | Add dataframe columns that are missing from the final-output target. | deep |
| `_resolve_dataset_run_from_table` | Resolve a single dataset/run pair from a rebuilt source table. | deep |
| `_resolve_first_existing_column` | Return the first priority column that exists in a column list. | short |
| `_normalize_machine_status_value` | Map synthetic status labels into the final output machine-status values. | deep |
| `_validate_rebuilt_columns` | Validate rebuilt source columns and resolve timestamp/status sources. | deep |
| `build_final_aligned_synthetic_output_dataframe` | Build the compact final synthetic output from rebuilt observations. | deep |
| `load_rebuilt_for_final_output` | Load rebuilt observations for final synthetic output generation. | deep |
| `_get_rebuilt_observation_bounds` | Return row count and observation-index bounds for rebuilt rows. | deep |
| `ensure_final_aligned_synthetic_output_table_exists` | Create the compact final synthetic output table and indexes. | deep |
| `write_final_aligned_synthetic_output` | Write compact final synthetic output rows to Postgres. | deep |
| `build_synthetic_final_aligned_output_stage` | Build compact final synthetic output from rebuilt rows in windows. | deep |

## Configuration Dependencies

- Dataset, run, stage, or recipe identifiers used for traceability.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_build_sensor_columns` | `n_sensors` | Return expected wide sensor columns for final synthetic output. |
| `_get_existing_columns` | `engine, *, schema, table` | Return existing Postgres columns for the final synthetic output table. |
| `_infer_alter_column_type` | `series` | Infer a conservative Postgres type for an added final-output column. |
| `_add_missing_columns` | `engine, *, schema, table, dataframe` | Add dataframe columns that are missing from the final-output target. |
| `_resolve_dataset_run_from_table` | `engine, *, schema, table_name, dataset_id, run_id` | Resolve a single dataset/run pair from a rebuilt source table. |
| `_resolve_first_existing_column` | `columns, priority_columns` | Return the first priority column that exists in a column list. |
| `_normalize_machine_status_value` | `value, *, status_mapping` | Map synthetic status labels into the final output machine-status values. |
| `_validate_rebuilt_columns` | `dataframe, *, n_sensors, timestamp_source_priority, status_source_priority` | Validate rebuilt source columns and resolve timestamp/status sources. |
| `build_final_aligned_synthetic_output_dataframe` | `rebuilt_dataframe, *, n_sensors, timestamp_source_priority, status_source_priority, status_mapping, timestamp_output_column, machine_status_output_column, sort_output` | Build the compact final synthetic output from rebuilt observations. |
| `load_rebuilt_for_final_output` | `engine, *, schema, table_name, dataset_id, run_id, complete_only, observation_index_min, observation_index_max` | Load rebuilt observations for final synthetic output generation. |
| `_get_rebuilt_observation_bounds` | `engine, *, schema, table_name, dataset_id, run_id, complete_only` | Return row count and observation-index bounds for rebuilt rows. |
| `ensure_final_aligned_synthetic_output_table_exists` | `engine, *, schema, table_name, n_sensors, timestamp_output_column, machine_status_output_column` | Create the compact final synthetic output table and indexes. |
| `write_final_aligned_synthetic_output` | `engine, dataframe, *, schema, table_name, n_sensors, timestamp_output_column, machine_status_output_column, if_exists` | Write compact final synthetic output rows to Postgres. |
| `build_synthetic_final_aligned_output_stage` | `engine, *, schema, rebuilt_table, target_table, dataset_id, run_id, n_sensors, complete_only, if_exists, observation_window_size, timestamp_source_priority, status_source_priority, status_mapping, timestamp_output_column, machine_status_output_column` | Build compact final synthetic output from rebuilt rows in windows. |

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
