# Synthetic Utility Reference: kafka_consumer_adapter.py

Source path:

`utils/synthetic/pipeline/kafka_consumer_adapter.py`

## Purpose

Consumes synthetic telemetry messages from Kafka and stages them for rebuild.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `_is_missing` | `value` | Return True for pandas/numpy missing values without failing on objects. |
| Function | `_normalize_scalar` | `value` | Convert pandas, datetime, Decimal, and numpy scalars into Postgres-safe values. |
| Function | `_parse_message_value` | `raw_value` | Decode a Kafka message value into the producer payload dictionary. |
| Function | `_get_nested` | `payload` | Read nested payload fields and return None when any level is missing. |
| Function | `build_consumed_message_record` | `` | Flatten the producer payload and Kafka metadata into one landed table row. |
| Function | `_get_existing_columns` | `engine` | Return existing columns for a consumed-stage table. |
| Function | `_infer_alter_column_type` | `series` | Infer a conservative Postgres column type for an added dataframe column. |
| Function | `_add_missing_columns` | `engine` | Add dataframe columns that are missing from the consumed-stage table. |
| Function | `ensure_consumed_stage_table_exists` | `engine` | Create the Kafka consumed-message landing table and core indexes. |
| Function | `ensure_consumed_stage_runtime_schema` | `engine` | One-time setup for the consumed Kafka message landing table. This should run once before the consumer loop starts. It avoids repeated table/schema alignment checks inside every consumer batch. |
| Function | `_yield_record_batches` | `records, batch_size` | Yield records in bounded batches for database insertion. |
| Function | `_insert_records_on_conflict_do_nothing` | `engine` | Insert consumed records while ignoring already-landed Kafka offsets. |
| Function | `write_consumed_messages_batch` | `engine, dataframe` | Append a consumer batch into the landed message table. Deduping is handled by the Postgres primary key with ON CONFLICT DO NOTHING instead of re-reading the full topic history. |
| Function | `build_confluent_consumer_config` | `` | Build the Confluent consumer configuration for landed-message ingestion. |
| Function | `create_confluent_consumer` | `` | Create a Confluent Kafka consumer from explicit or environment settings. |
| Function | `consume_kafka_messages_once` | `` | Pull a finite batch of Kafka messages from an already-subscribed consumer. |
| Function | `land_consumed_messages_to_postgres` | `engine` | Normalize a consumed Kafka batch and append it into the landed stage table. |
| Function | `run_kafka_consumer_to_postgres_once` | `engine` | Consume a finite Kafka batch and land it to Postgres. Commits offsets only after successful Postgres landing if commit_on_success=True. |
| Function | `run_kafka_consumer_to_postgres_loop` | `engine` | Run the Kafka-to-Postgres consumer until empty, capped, or interrupted. |

## Configuration Dependencies

- Uses SQL schema or table name settings.
- Uses configuration dictionaries or resolved stage configuration.
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

- `notebooks/synthetic/synthetic_08_kafka_consumer_adapter.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
