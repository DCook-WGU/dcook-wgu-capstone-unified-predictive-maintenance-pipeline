# Synthetic Utility Reference: final_aligned_output_writer.py

Source path:

`utils/synthetic/pipeline/final_aligned_output_writer.py`

## Purpose

Builds compact final synthetic output tables.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `_build_sensor_columns` | `n_sensors` | Return expected wide sensor columns for final synthetic output. |
| Function | `_get_existing_columns` | `engine` | Return existing Postgres columns for the final synthetic output table. |
| Function | `_infer_alter_column_type` | `series` | Infer a conservative Postgres type for an added final-output column. |
| Function | `_add_missing_columns` | `engine` | Add dataframe columns that are missing from the final-output target. |
| Function | `_resolve_dataset_run_from_table` | `engine` | Resolve a single dataset/run pair from a rebuilt source table. |
| Function | `_resolve_first_existing_column` | `columns, priority_columns` | Return the first priority column that exists in a column list. |
| Function | `_normalize_machine_status_value` | `value` | Map synthetic status labels into the final output machine-status values. |
| Function | `_validate_rebuilt_columns` | `dataframe` | Validate rebuilt source columns and resolve timestamp/status sources. |
| Function | `build_final_aligned_synthetic_output_dataframe` | `rebuilt_dataframe` | Build the compact final synthetic output from rebuilt observations. The output keeps dataset/run/asset identity, timestamp, sensor columns, and normalized machine status for downstream inspection or Bronze-style use. |
| Function | `load_rebuilt_for_final_output` | `engine` | Load rebuilt observations for final synthetic output generation. |
| Function | `_get_rebuilt_observation_bounds` | `engine` | Return row count and observation-index bounds for rebuilt rows. |
| Function | `ensure_final_aligned_synthetic_output_table_exists` | `engine` | Create the compact final synthetic output table and indexes. |
| Function | `write_final_aligned_synthetic_output` | `engine, dataframe` | Write compact final synthetic output rows to Postgres. |
| Function | `build_synthetic_final_aligned_output_stage` | `engine` | Build compact final synthetic output from rebuilt rows in windows. |

## Configuration Dependencies

- Uses SQL schema or table name settings.
- Uses dataset identity values.
- Uses run or recipe identity values.

## Inputs and Outputs

Key inputs:
- Configuration values, dataset identity, run identity, or recipe identity
- Database engine, schema, table, or SQL runtime context
- Pandas dataframes or dataframe-like stage inputs

Key outputs:
- Dataframes or transformed stage outputs
- SQL table rows, status updates, or database-stage records

## Logging, Ledger, and Artifact Behavior

### Logging

- No direct logger calls detected in this module.

### Ledger

- No direct ledger behavior detected in this module.

### SQL/database

- Uses SQL, PostgreSQL, engine, table, or database write/read behavior.

### Artifacts

- No direct artifact write pattern detected in this module.

## Downstream Usage

- `notebooks/synthetic/synthetic_11_build_final_aligned_observations_stage.ipynb`
- `notebooks/synthetic/synthetic_11a_build_final_aligned_observations_stage.ipynb`
- `notebooks/synthetic/synthetic_11b_build_final_aligned_observations_stage.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`
- `notebooks/synthetic/synthetic_pipeline_condensed-09_11.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
