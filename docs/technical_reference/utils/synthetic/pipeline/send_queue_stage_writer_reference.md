# Synthetic Utility Reference: send_queue_stage_writer.py

Source path:

`utils/synthetic/pipeline/send_queue_stage_writer.py`

## Purpose

Builds the staged send queue for synthetic producer workflows.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `log_step_timing` | `step_name, start_time` | Print elapsed time for a send-queue build step and return a fresh timer. |
| Function | `_validate_source_columns` | `dataframe, required_columns` | Raise a clear error when the staged sensor table is missing queue inputs. |
| Function | `_ensure_send_queue_runtime_columns` | `engine` | Add runtime queue columns used by producer claiming and acknowledgements. |
| Function | `_ensure_send_queue_indexes` | `engine` | Create indexes used by status checks, claim ordering, and message lookup. |
| Function | `_apply_send_queue_owner_and_grants` | `engine` | Grant the producer role ownership and DML access to the send queue table. |
| Function | `build_sensor_messages_send_queue` | `engine` | Build the send queue from staged sensor messages using chunked pandas writes. Each output row represents one sensor message ready to be claimed by the Kafka producer. The function adds queue status fields, message keys, and producer delivery placeholders while preserving the staged sensor columns. |
| Function | `build_sensor_messages_send_queue_sql_native` | `engine` | Build the synthetic sensor message send queue directly inside Postgres. This avoids reading 11M+ rows into pandas just to add queue metadata. The output table keeps the same queue/runtime fields expected by the producer queue manager and Kafka producer stages. |
| Function | `validate_sensor_messages_send_queue` | `engine` | Return row, timestamp, sensor, and pending-message checks for the send queue. |

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

- `notebooks/synthetic/synthetic_05_build_send_queue_stage.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
