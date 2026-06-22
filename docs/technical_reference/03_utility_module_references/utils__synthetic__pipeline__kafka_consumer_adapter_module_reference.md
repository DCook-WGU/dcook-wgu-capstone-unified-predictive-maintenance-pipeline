# Utility Module Reference: `utils/synthetic/pipeline/kafka_consumer_adapter.py`

## Module Purpose

This module adapts consumed Kafka telemetry messages into project staging structures.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module adapts consumed Kafka telemetry messages into project staging structures.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_is_missing` | Return True for pandas/numpy missing values without failing on objects. | deep |
| `_normalize_scalar` | Convert pandas, datetime, Decimal, and numpy scalars into Postgres-safe values. | deep |
| `_parse_message_value` | Decode a Kafka message value into the producer payload dictionary. | deep |
| `_get_nested` | Read nested payload fields and return None when any level is missing. | deep |
| `build_consumed_message_record` | Flatten the producer payload and Kafka metadata into one landed table row. | deep |
| `_get_existing_columns` | Return existing columns for a consumed-stage table. | deep |
| `_infer_alter_column_type` | Infer a conservative Postgres column type for an added dataframe column. | deep |
| `_add_missing_columns` | Add dataframe columns that are missing from the consumed-stage table. | deep |
| `ensure_consumed_stage_table_exists` | Create the Kafka consumed-message landing table and core indexes. | deep |
| `ensure_consumed_stage_runtime_schema` | One-time setup for the consumed Kafka message landing table. | deep |
| `_yield_record_batches` | Yield records in bounded batches for database insertion. | deep |
| `_insert_records_on_conflict_do_nothing` | Insert consumed records while ignoring already-landed Kafka offsets. | deep |
| `write_consumed_messages_batch` | Append a consumer batch into the landed message table. | deep |
| `build_confluent_consumer_config` | Build the Confluent consumer configuration for landed-message ingestion. | deep |
| `create_confluent_consumer` | Create a Confluent Kafka consumer from explicit or environment settings. | deep |
| `consume_kafka_messages_once` | Pull a finite batch of Kafka messages from an already-subscribed consumer. | deep |
| `land_consumed_messages_to_postgres` | Normalize a consumed Kafka batch and append it into the landed stage table. | deep |
| `run_kafka_consumer_to_postgres_once` | Consume a finite Kafka batch and land it to Postgres. | deep |
| `run_kafka_consumer_to_postgres_loop` | Run the Kafka-to-Postgres consumer until empty, capped, or interrupted. | deep |

## Configuration Dependencies

- Resolved pipeline configuration dictionaries and YAML-derived stage settings.
- Environment variables where runtime mode or optional integration behavior is configured.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.
- Kafka topic and producer/consumer settings for synthetic streaming handoff.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_is_missing` | `value` | Return True for pandas/numpy missing values without failing on objects. |
| `_normalize_scalar` | `value` | Convert pandas, datetime, Decimal, and numpy scalars into Postgres-safe values. |
| `_parse_message_value` | `raw_value` | Decode a Kafka message value into the producer payload dictionary. |
| `_get_nested` | `payload, *keys` | Read nested payload fields and return None when any level is missing. |
| `build_consumed_message_record` | `*, payload, kafka_topic, kafka_partition, kafka_offset, consumer_group_id, consumer_worker_id, raw_value_text` | Flatten the producer payload and Kafka metadata into one landed table row. |
| `_get_existing_columns` | `engine, *, schema, table` | Return existing columns for a consumed-stage table. |
| `_infer_alter_column_type` | `series` | Infer a conservative Postgres column type for an added dataframe column. |
| `_add_missing_columns` | `engine, *, schema, table, dataframe` | Add dataframe columns that are missing from the consumed-stage table. |
| `ensure_consumed_stage_table_exists` | `engine, *, schema, table_name` | Create the Kafka consumed-message landing table and core indexes. |
| `ensure_consumed_stage_runtime_schema` | `engine, *, schema, table_name` | One-time setup for the consumed Kafka message landing table. |
| `_yield_record_batches` | `records, batch_size` | Yield records in bounded batches for database insertion. |
| `_insert_records_on_conflict_do_nothing` | `engine, *, schema, table_name, records, chunk_size` | Insert consumed records while ignoring already-landed Kafka offsets. |
| `write_consumed_messages_batch` | `engine, dataframe, *, schema, table_name, ensure_table, align_schema` | Append a consumer batch into the landed message table. |
| `build_confluent_consumer_config` | `*, bootstrap_servers, consumer_group_id, auto_offset_reset, enable_auto_commit, extra_config` | Build the Confluent consumer configuration for landed-message ingestion. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.
- Source includes Kafka producer/consumer terminology or calls; helpers participate in synthetic streaming handoff when used by the synthetic pipeline.

## Artifact / SQL / File-System Interactions

- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.
- Kafka/PostgreSQL handoff: Source references producer, consumer, topic, or streaming-stage behavior.

## Failure Behavior

- Source raises `ImportError` for invalid input, missing context, or failed validation paths.
- Source raises `RuntimeError` for invalid input, missing context, or failed validation paths.
- Source raises `TypeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because the synthetic subsystem depends on repeatable staging, truth handling, streaming handoff, or final alignment behavior to make controlled pump telemetry tests explainable.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
