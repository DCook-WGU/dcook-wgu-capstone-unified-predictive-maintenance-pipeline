# Synthetic Utility Reference: postgres_to_bronze.py

Source path:

`utils/synthetic/pipeline/postgres_to_bronze.py`

## Purpose

Utilities for converting the wide synthetic stream table in Postgres into a single bronze-ready table that looks like the original pump sensor dataset. Main goals: 1. Read one or more synthetic batches from Postgres. 2. Sort them into one stable sequence across batches. 3. Create unified row numbering and unified episode numbering. 4. Add a fresh time index and timestamp series. 5. Derive the original-style machine status label. 6. Cut the dataframe down to the columns needed for Bronze handoff.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `build_engine_from_project_env` | `` | Return a SQLAlchemy engine using the same project utility pattern as the rest of the capstone. |
| Function | `get_table_columns` | `engine` | Return table columns in ordinal order. |
| Function | `get_sensor_columns` | `columns` | Return sensor columns in numeric order: sensor_00, sensor_01, ..., sensor_51. |
| Function | `_int_or_default` | `value, default` | Convert a nullable SQL aggregate value to int with a default. |
| Function | `read_synthetic_stream_dataframe` | `engine` | Read the wide synthetic stream table from Postgres. Notes: - batch_ids is optional - selected_columns is optional |
| Function | `get_distinct_batch_ids` | `engine` | Return distinct batch ids from a table. |
| Function | `get_unloaded_source_batch_ids` | `engine` | Compare source and target tables and return which source batches have not yet been loaded into the target append table. |
| Function | `ensure_handoff_control_table` | `engine` | Create the append-control table used by the Bronze handoff builder. |
| Function | `get_effective_handoff_offsets` | `engine` | Resolve next append offsets from control table or target table scan. |
| Function | `upsert_handoff_control_record` | `engine` | Persist append-control offsets after a Bronze handoff append. |
| Function | `get_handoff_control_record` | `engine` | Read the active append-control record for a Bronze target table. |
| Function | `get_handoff_append_offsets` | `engine` | Read the current append target table and return the next starting offsets for ids and time fields. |
| Function | `validate_synthetic_stream_dataframe` | `dataframe` | Basic schema validation for the wide synthetic stream table. |
| Function | `choose_sort_columns` | `dataframe` | Choose the best available sort columns for one unified sequence. Preference: 1. global_cycle_id 2. batch_id + row_in_batch 3. fallback ordered columns |
| Function | `sort_synthetic_stream_dataframe` | `dataframe` | Stable sort for one unified ordered stream. |
| Function | `add_unified_row_id` | `dataframe` | Add a single unified row number across the selected rows. |
| Function | `add_unified_episode_id` | `dataframe` | Create a batch-safe unified episode id. This matters because each batch can restart meta__episode_id at 0. |
| Function | `derive_machine_status` | `dataframe` | Convert synthetic labels to the original pump-style machine_status values. Resolution order: 1. stream_state when it maps cleanly 2. phase when stream_state did not resolve 3. default_value |
| Function | `add_synthetic_anomaly_flag` | `dataframe` | Optional helper flag for quick sanity checks or downstream testing. NORMAL => 0 anything else => 1 |
| Function | `trim_unified_dataframe` | `dataframe` | Optional row trimming after the unified table is built. This happens BEFORE observation_time_index/timestamp are assigned so the final output stays contiguous. |
| Function | `add_observation_time_fields` | `dataframe` | Add final contiguous time index + timestamp values to the final row set. |
| Function | `select_bronze_handoff_columns` | `dataframe` | Select the final handoff columns. Default behavior: timestamp + sensor_* + machine_status Optional: - anomaly_flag__synthetic - lineage columns - any other remaining columns |
| Function | `prepare_synthetic_postgres_for_bronze_handoff` | `raw_dataframe` | Full in-memory direct wide-table -> bronze handoff preparation. Supports both: - fresh rebuilds - append-aware continuation when offsets are provided |
| Function | `build_bronze_handoff_from_postgres` | `engine` | Read the raw wide stream from Postgres and return the final bronze handoff dataframe. |
| Function | `build_append_aware_bronze_handoff_from_postgres` | `engine` | Append-aware builder. Behavior: 1. Detect source batches not yet loaded into the target append table 2. Read target offsets from the append table 3. Build only the new rows with continued ids and time fields |
| Function | `summarize_bronze_handoff_dataframe` | `dataframe` | Small summary payload for notebook logging / truth payloads. |
| Function | `write_bronze_handoff_to_postgres` | `engine, dataframe` | Write the final bronze handoff dataframe using the same generic Postgres writer pattern used elsewhere in the project. |

## Configuration Dependencies

- Uses SQL schema or table name settings.
- Uses dataset identity values.
- Uses run or recipe identity values.

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

- `notebooks/synthetic/synthetic_00_postgres_to_bronze_no_kafka.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
