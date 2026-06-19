# Synthetic Utility Reference: final_aligned_observation_writer.py

Source path:

`utils/synthetic/pipeline/final_aligned_observation_writer.py`

## Purpose

Builds the lineage-rich final-aligned observation stage.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `_get_existing_columns` | `engine` | Return existing Postgres columns for the final-aligned table. |
| Function | `_infer_alter_column_type` | `series` | Infer a conservative Postgres type for an added final-aligned column. |
| Function | `_add_missing_columns` | `engine` | Add dataframe columns that are missing from the final-aligned target. |
| Function | `_build_sensor_columns` | `n_sensors` | Return expected wide sensor columns for final aligned observations. |
| Function | `_coalesce_left_then_right` | `left, right` | Prefer left values while filling missing values from the right series. |
| Function | `_require_columns` | `dataframe, required_columns, frame_name` | Raise a clear error when a dataframe is missing required columns. |
| Function | `_validate_premelt_columns` | `dataframe, n_sensors` | Validate premelt columns needed to restore original row context. |
| Function | `_validate_rebuilt_columns` | `dataframe, n_sensors` | Validate rebuilt columns needed for final alignment. |
| Function | `load_premelt_for_final_alignment` | `engine` | Load premelt observation rows that provide original ordering metadata. |
| Function | `load_rebuilt_for_final_alignment` | `engine` | Load rebuilt observations for final alignment, optionally complete rows only. |
| Function | `build_final_aligned_observations_dataframe` | `` | Merge premelt row context with rebuilt sensor values into final wide rows. Premelt contributes batch/order context, while rebuilt rows contribute the reconstructed sensor values and rebuild status fields used downstream. |
| Function | `ensure_final_aligned_table_exists` | `engine` | Create the final-aligned observation table and lookup indexes. |
| Function | `write_final_aligned_observations` | `engine, dataframe` | Write final-aligned observations to Postgres with optional table replace. |
| Function | `build_final_aligned_observations_stage` | `engine` | Build the final-aligned table from premelt and rebuilt stages in windows. |

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

- Writes or prepares files/artifacts such as CSV, Parquet, JSON, or metadata outputs.

## Downstream Usage

- `notebooks/synthetic/synthetic_11_build_final_aligned_observations_stage.ipynb`
- `notebooks/synthetic/synthetic_11a_build_final_aligned_observations_stage.ipynb`
- `notebooks/synthetic/synthetic_11b_build_final_aligned_observations_stage.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
