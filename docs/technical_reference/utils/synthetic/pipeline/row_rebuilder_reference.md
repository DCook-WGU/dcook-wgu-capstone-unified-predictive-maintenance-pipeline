# Synthetic Utility Reference: row_rebuilder.py

Source path:

`utils/synthetic/pipeline/row_rebuilder.py`

## Purpose

Rebuilds wide synthetic observations from consumed sensor messages.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `_get_existing_columns` | `engine` | Return existing Postgres columns for a rebuild-stage table. |
| Function | `_infer_alter_column_type` | `series` | Infer a conservative Postgres type for an added rebuild column. |
| Function | `_add_missing_columns` | `engine` | Add dataframe columns that are not present in the rebuild target table. |
| Function | `_validate_consumed_columns` | `dataframe` | Validate consumed long-message columns required for observation rebuild. |
| Function | `_build_sensor_columns` | `n_sensors` | Return the expected wide sensor column names for rebuilt observations. |
| Function | `ensure_rebuilt_stage_table_exists` | `engine` | Create the rebuilt wide-observation table and core rebuild indexes. |
| Function | `load_consumed_messages_for_rebuild` | `engine` | Load consumed long sensor messages that are eligible for rebuild. |
| Function | `deduplicate_consumed_messages` | `dataframe` | Deduplicate at the logical sensor-message level. Canonical identity: dataset_id + run_id + asset_id + observation_index + sensor_index Keep the latest received row for each logical identity. |
| Function | `build_rebuilt_observations_dataframe` | `dataframe` | Rebuild wide observations from long consumed messages. Returns ------- rebuilt_dataframe Wide rebuilt observations. rebuilt_keys Observation identity keys used for optional rebuild-status updates. |
| Function | `_remove_already_rebuilt_observations` | `engine` | Drop rebuilt rows whose observation keys already exist in the target table. |
| Function | `write_rebuilt_observations_batch` | `engine, dataframe` | Append rebuilt wide observations after removing already-written keys. |
| Function | `mark_consumed_messages_rebuilt` | `engine, observation_keys` | Mark consumed long rows for rebuilt observations as rebuilt. The input dataframe contains one row per rebuilt observation key, while the source table may contain many sensor-message rows for that observation. |
| Function | `rebuild_consumed_messages_to_observations` | `engine` | Rebuild consumed long sensor messages into wide observation rows. The function processes bounded observation windows, writes completed wide rows to the rebuilt stage, and optionally marks source consumed rows rebuilt. |

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

- `notebooks/synthetic/synthetic_09_row_rebuilder.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`
- `notebooks/synthetic/synthetic_pipeline_condensed-09_11.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
