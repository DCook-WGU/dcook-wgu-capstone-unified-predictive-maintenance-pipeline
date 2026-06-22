# utils/synthetic/pipeline/send_queue_stage_writer.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `send_queue_stage_writer.py` that need deeper explanation than the 071d module-level reference. The selected function builds the PostgreSQL send queue consumed by the synthetic Kafka producer path.

## Source Grounding

Sources used:

- `utils/synthetic/pipeline/send_queue_stage_writer.py`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__synthetic__pipeline__send_queue_stage_writer_module_reference.md`
- `function_inventory.json`
- `technical_reference/01_notebook_workflow_references/`
- `technical_reference/00_project_manual/`
- Documentation standards under `docs_agent_pack/00_STANDARDS/`

The active utility source file is the source of truth.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
| -------- | -------------- | ------------------------ |
| `build_sensor_messages_send_queue` | Builds the sensor-message send queue from staged long sensor messages | Synthetic Kafka producer preparation |

## Module-Level Technical Context

The send queue stage converts staged long-form sensor messages into a producer-ready PostgreSQL table. It adds deterministic message keys, queue status fields, claim fields, producer delivery placeholders, and lookup indexes needed by the Kafka producer adapter.

## Deep Function References

### `build_sensor_messages_send_queue`

#### Functional Purpose

`build_sensor_messages_send_queue` reads a staged synthetic sensor-message table and writes a queue table whose rows can be claimed and published by the producer. Each output row represents one sensor message with queue status, message key, claim fields, and delivery placeholders.

#### Pipeline Context

Active notebook source confirms use in the synthetic send-queue notebook, with a SQL-native variant also present in that stage. This function is the chunked pandas write path for building `synthetic_sensor_messages_send_queue` from `synthetic_sensor_messages_stage`.

#### Inputs and Assumptions

Important inputs include the SQLAlchemy engine, schema, source table, target table, write mode, default queue status, chunk size, optional owner role, and optional owner/grant behavior.

The source table must contain required queue inputs including dataset/run identity, asset identity, generated row id, observation index/timestamp, message sequence index, batch fields, stream/phase labels, fault metadata, sensor name/index/value, telemetry event fields, and producer send attempt.

#### Outputs and Return Contract

The function returns the sanitized target table name. The operational output is a PostgreSQL queue table containing ordered staged sensor rows plus queue metadata.

#### Side Effects

Confirmed side effects are:

- Creates the schema if needed.
- Reads source table columns and source row count.
- Writes transformed chunks to the target queue table through `write_layer_dataframe`.
- Uses the requested write mode for the first chunk and appends later chunks.
- Adds runtime columns such as `claim_token`, `claimed_at`, producer topic/worker fields, acknowledgement fields, and delivery status/error fields.
- Creates indexes for queue status, producer order, message key, sent time, claim token, claim order, and claimed time.
- Optionally applies owner and grants to the configured producer role.

#### Failure Behavior and Guardrails

The function raises `ValueError` if required source columns are missing or the source table is empty. Supporting SQL helpers may raise database errors when schemas, tables, permissions, or connections are invalid.

#### Lineage, Idempotency, and Reproducibility Role

The generated `message_key` combines `dataset_id`, `run_id`, `asset_id`, `observation_index`, and `sensor_index`. Queue order is stable by `observation_index`, `message_sequence_index`, and `sensor_index`. The default queue status, queued timestamp, claim fields, and delivery placeholders create the state contract used by later producer lifecycle steps.

Idempotency depends on the selected `if_exists` mode. With the default `replace`, the queue table is rebuilt. With append mode, new chunks are appended.

#### Why This Function Matters

The Kafka producer should not infer message identity or claim state from raw staged rows. This function creates the durable queue contract that lets the producer claim, publish, mark sent, and mark failed records in a controlled way.

#### Verification Method

- Confirm the target queue table row count matches the source staged message count.
- Confirm required queue fields and runtime fields exist.
- Confirm `message_key` is populated and stable for representative rows.
- Confirm `queue_status` defaults to the configured value.
- Confirm queue indexes exist for status, claim, message key, and producer ordering.
- Confirm empty source tables raise `ValueError`.

## Cross-Function Relationships

`build_sensor_messages_send_queue` prepares the PostgreSQL queue consumed by `run_send_queue_producer_once`. Its output fields, especially `message_key`, `queue_status`, `claim_token`, and producer delivery columns, are the fields the producer path uses to claim rows and record Kafka delivery outcomes.

## Source-Limited Items

- Whether the chunked builder or SQL-native builder is preferred for every run is Not determined from available source.
- Kafka publishing is not performed by this function from available source.
