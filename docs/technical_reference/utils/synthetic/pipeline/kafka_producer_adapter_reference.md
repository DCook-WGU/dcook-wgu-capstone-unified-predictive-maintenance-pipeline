# Synthetic Utility Reference: kafka_producer_adapter.py

Source path:

`utils/synthetic/pipeline/kafka_producer_adapter.py`

## Purpose

Sends queued synthetic telemetry messages to Kafka when streaming services are available.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `_get_first_env_value` | `names` | Return the first non-empty environment value from a list of aliases. |
| Function | `get_kafka_bootstrap_servers_from_env` | `env_names` | Read Kafka bootstrap servers from the supported environment variables. |
| Function | `_is_missing` | `value` | Return True for pandas/numpy missing values without failing on objects. |
| Function | `_normalize_scalar` | `value` | Convert pandas, datetime, Decimal, and numpy scalars into JSON-safe values. |
| Function | `json_dumps_safe` | `payload` | Serialize a Kafka payload after normalizing dataframe scalar types. |
| Function | `build_confluent_producer_config` | `` | Build the Confluent producer configuration used by the synthetic sender. |
| Function | `create_confluent_producer` | `` | Create a Confluent Kafka producer from explicit or environment settings. |
| Function | `build_sensor_message_payload` | `row` | Convert one claimed send-queue row into the nested Kafka message payload. The payload groups observation, sensor, metadata, telemetry, and producer fields so the consumer can rebuild the landed Postgres row deterministically. |
| Function | `publish_claimed_batch_to_kafka` | `` | Publish an already-claimed dataframe to Kafka. This function does not touch Postgres queue state. It only publishes. Queue success/failure updates should be handled by the caller. |
| Function | `run_send_queue_producer_once` | `engine` | Claim the next pending queue batch, publish it to Kafka, then mark it sent or failed as a single claim group. |
| Function | `run_send_queue_producer_loop` | `engine` | Repeatedly publish queue batches until: - queue is empty - control row disables the run - max_batches is reached - a failure occurs and stop_on_failure=True |

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

- `notebooks/synthetic/synthetic_07_kafka_producer_adapter.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
