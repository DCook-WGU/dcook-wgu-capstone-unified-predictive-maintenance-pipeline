# Utility Module Reference: `utils/synthetic/pipeline/kafka_producer_adapter.py`

## Module Purpose

This module adapts staged synthetic telemetry rows into Kafka-producer messages.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module adapts staged synthetic telemetry rows into Kafka-producer messages.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_get_first_env_value` | Return the first non-empty environment value from a list of aliases. | short |
| `get_kafka_bootstrap_servers_from_env` | Read Kafka bootstrap servers from the supported environment variables. | deep |
| `_is_missing` | Return True for pandas/numpy missing values without failing on objects. | deep |
| `_normalize_scalar` | Convert pandas, datetime, Decimal, and numpy scalars into JSON-safe values. | deep |
| `json_dumps_safe` | Serialize a Kafka payload after normalizing dataframe scalar types. | short |
| `build_confluent_producer_config` | Build the Confluent producer configuration used by the synthetic sender. | deep |
| `create_confluent_producer` | Create a Confluent Kafka producer from explicit or environment settings. | deep |
| `build_sensor_message_payload` | Convert one claimed send-queue row into the nested Kafka message payload. | deep |
| `publish_claimed_batch_to_kafka` | Publish an already-claimed dataframe to Kafka. | deep |
| `run_send_queue_producer_once` | Claim the next pending queue batch, publish it to Kafka, then mark it sent or failed as a single claim group. | deep |
| `run_send_queue_producer_loop` | Repeatedly publish queue batches until: - queue is empty - control row disables the run - max_batches is reached - a failure occurs and stop_on_failure=True | deep |

## Configuration Dependencies

- Resolved pipeline configuration dictionaries and YAML-derived stage settings.
- Environment variables where runtime mode or optional integration behavior is configured.
- Kafka topic and producer/consumer settings for synthetic streaming handoff.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_get_first_env_value` | `names` | Return the first non-empty environment value from a list of aliases. |
| `get_kafka_bootstrap_servers_from_env` | `env_names` | Read Kafka bootstrap servers from the supported environment variables. |
| `_is_missing` | `value` | Return True for pandas/numpy missing values without failing on objects. |
| `_normalize_scalar` | `value` | Convert pandas, datetime, Decimal, and numpy scalars into JSON-safe values. |
| `json_dumps_safe` | `payload` | Serialize a Kafka payload after normalizing dataframe scalar types. |
| `build_confluent_producer_config` | `*, bootstrap_servers, client_id, acks, linger_ms, compression_type, extra_config` | Build the Confluent producer configuration used by the synthetic sender. |
| `create_confluent_producer` | `*, bootstrap_servers, client_id, acks, linger_ms, compression_type, extra_config` | Create a Confluent Kafka producer from explicit or environment settings. |
| `build_sensor_message_payload` | `row` | Convert one claimed send-queue row into the nested Kafka message payload. |
| `publish_claimed_batch_to_kafka` | `*, producer, claimed_dataframe, topic, flush_timeout_seconds` | Publish an already-claimed dataframe to Kafka. |
| `run_send_queue_producer_once` | `engine, *, dataset_id, run_id, schema, queue_table, control_table, batch_size, topic, producer, bootstrap_servers, producer_worker_id, client_id, flush_timeout_seconds` | Claim the next pending queue batch, publish it to Kafka, then mark it sent or failed as a single claim group. |
| `run_send_queue_producer_loop` | `engine, *, dataset_id, run_id, schema, queue_table, control_table, bootstrap_servers, producer_worker_id, client_id, max_batches, stop_on_failure, flush_timeout_seconds, enable_progress_logging, progress_every_batches` | Repeatedly publish queue batches until: - queue is empty - control row disables the run - max_batches is reached - a failure occurs and stop_on_failure=True |

## Side Effects

- Source includes Kafka producer/consumer terminology or calls; helpers participate in synthetic streaming handoff when used by the synthetic pipeline.

## Artifact / SQL / File-System Interactions

- Kafka/PostgreSQL handoff: Source references producer, consumer, topic, or streaming-stage behavior.

## Failure Behavior

- Source raises `ImportError` for invalid input, missing context, or failed validation paths.
- Source raises `RuntimeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because the synthetic subsystem depends on repeatable staging, truth handling, streaming handoff, or final alignment behavior to make controlled pump telemetry tests explainable.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
