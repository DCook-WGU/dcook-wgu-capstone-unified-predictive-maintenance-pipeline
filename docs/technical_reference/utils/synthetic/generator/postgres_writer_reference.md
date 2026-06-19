# Synthetic Utility Reference: postgres_writer.py

Source path:

`utils/synthetic/generator/postgres_writer.py`

## Purpose

Writes generated synthetic outputs to PostgreSQL when that path is enabled.

## Pipeline Role

Generator-side utility used before the staged PostgreSQL/Kafka synthetic pipeline. It helps create, shape, or export synthetic pump telemetry.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `scalar_to_int` | `value, name` | Convert a SQL scalar value to int and reject missing sequence values. |
| Function | `ensure_sequence` | `engine` | Create the Postgres sequence used for synthetic batch or cycle IDs. |
| Function | `reserve_next_batch_id` | `engine` | Reserve and return the next synthetic batch identifier. |
| Function | `reserve_cycle_range` | `engine` | Reserve a contiguous global-cycle range and return its first value. |
| Function | `reset_sequence` | `engine` | Reset a Postgres sequence so the next value starts at start_at. |
| Function | `reset_synthetic_sequences` | `engine` | Reset the synthetic batch and cycle sequences for one dataset. |
| Function | `_ensure_stream_table_exists` | `engine` | Create the base stream table if missing. Sensor columns are added dynamically later. |
| Function | `_get_existing_columns` | `engine` | Read the current column names for a synthetic stream table. |
| Function | `_infer_alter_column_type` | `series` | Infer a Postgres column type for dynamic synthetic stream columns. |
| Function | `_add_missing_columns` | `engine` | Add dataframe columns that are not yet present in the stream table. |
| Function | `_prepare_dataframe_for_copy` | `dataframe` | Sanitize columns and serialize nested values before COPY loading. |
| Function | `_copy_dataframe_to_table` | `engine, dataframe` | Bulk-load a prepared dataframe into a Postgres table with COPY. |
| Function | `write_stream_batch` | `engine, dataframe` | Write a synthetic stream batch to the dataset stream table. The helper writes to: synthetic_<dataset_name>_<artifact_name> Behavior: - ensures the base stream table exists - auto-adds any missing columns for this dataframe - uses COPY bulk load by default for faster inserts - falls back to the generic layer writer if COPY fails |

## Configuration Dependencies

- Uses SQL schema or table name settings.
- Uses dataset identity values.

## Inputs and Outputs

Key inputs:
- Configuration values, dataset identity, run identity, or recipe identity
- Database engine, schema, table, or SQL runtime context
- Filesystem paths and artifact files
- Pandas dataframes or dataframe-like stage inputs

Key outputs:
- Dataframes or transformed stage outputs
- File-based artifacts or metadata outputs
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

- `notebooks/synthetic/synthetic_01_generate_synethic_data.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
