# Utility Module Reference: `utils/synthetic/pipeline/rebuild_comparison.py`

## Module Purpose

This module compares rebuilt synthetic observations against source/final aligned outputs.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module compares rebuilt synthetic observations against source/final aligned outputs.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_get_existing_columns` | Return existing Postgres columns for a comparison-stage table. | deep |
| `_infer_alter_column_type` | Infer a conservative Postgres type for an added comparison column. | deep |
| `_add_missing_columns` | Add dataframe columns that are missing from the comparison target table. | deep |
| `_build_sensor_columns` | Return sensor column names expected in premelt and rebuilt frames. | short |
| `_normalize_missing_scalar` | Normalize pandas missing values before scalar comparison. | deep |
| `_compare_scalar` | Compare scalar values with tolerance for numeric-looking values. | deep |
| `_validate_premelt_columns` | Validate original premelt columns needed for rebuild comparison. | deep |
| `_validate_rebuilt_columns` | Validate rebuilt columns needed for field-by-field comparison. | deep |
| `load_premelt_for_comparison` | Load original premelt observations for rebuild comparison. | deep |
| `load_rebuilt_for_comparison` | Load rebuilt wide observations for comparison against premelt rows. | deep |
| `build_rebuild_comparison_dataframe` | Compare original premelt observations against rebuilt observations. | deep |
| `ensure_rebuild_comparison_table_exists` | Create the rebuild comparison table and mismatch lookup indexes. | deep |
| `_remove_existing_comparison_rows` | Drop comparison rows whose observation keys already exist in the target. | deep |
| `write_rebuild_comparison_batch` | Append comparison rows after adding completion time and missing columns. | deep |
| `build_rebuild_comparison_stage` | Build comparison rows in observation windows and write them to Postgres. | deep |

## Configuration Dependencies

- Dataset, run, stage, or recipe identifiers used for traceability.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_get_existing_columns` | `engine, *, schema, table` | Return existing Postgres columns for a comparison-stage table. |
| `_infer_alter_column_type` | `series` | Infer a conservative Postgres type for an added comparison column. |
| `_add_missing_columns` | `engine, *, schema, table, dataframe` | Add dataframe columns that are missing from the comparison target table. |
| `_build_sensor_columns` | `n_sensors` | Return sensor column names expected in premelt and rebuilt frames. |
| `_normalize_missing_scalar` | `value` | Normalize pandas missing values before scalar comparison. |
| `_compare_scalar` | `left, right, *, float_tolerance` | Compare scalar values with tolerance for numeric-looking values. |
| `_validate_premelt_columns` | `dataframe, n_sensors` | Validate original premelt columns needed for rebuild comparison. |
| `_validate_rebuilt_columns` | `dataframe, n_sensors` | Validate rebuilt columns needed for field-by-field comparison. |
| `load_premelt_for_comparison` | `engine, *, schema, table_name, dataset_id, run_id` | Load original premelt observations for rebuild comparison. |
| `load_rebuilt_for_comparison` | `engine, *, schema, table_name, dataset_id, run_id` | Load rebuilt wide observations for comparison against premelt rows. |
| `build_rebuild_comparison_dataframe` | `premelt_dataframe, rebuilt_dataframe, *, n_sensors, float_tolerance` | Compare original premelt observations against rebuilt observations. |
| `ensure_rebuild_comparison_table_exists` | `engine, *, schema, table_name` | Create the rebuild comparison table and mismatch lookup indexes. |
| `_remove_existing_comparison_rows` | `engine, *, comparison_dataframe, schema, target_table` | Drop comparison rows whose observation keys already exist in the target. |
| `write_rebuild_comparison_batch` | `engine, dataframe, *, schema, table_name` | Append comparison rows after adding completion time and missing columns. |

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
