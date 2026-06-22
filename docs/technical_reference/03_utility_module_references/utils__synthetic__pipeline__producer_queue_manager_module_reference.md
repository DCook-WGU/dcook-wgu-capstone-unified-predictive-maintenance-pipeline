# Utility Module Reference: `utils/synthetic/pipeline/producer_queue_manager.py`

## Module Purpose

This module manages queued synthetic telemetry records for Kafka producer handoff.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module manages queued synthetic telemetry records for Kafka producer handoff.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `scalar_to_int` | Convert a required scalar result into an int and reject missing values. | deep |
| `_grant_schema_usage_create` | Grant schema access needed by a queue runtime role. | deep |
| `_apply_table_owner_and_grants` | Assign ownership and DML grants for a runtime queue table. | deep |
| `ensure_send_queue_runtime_columns` | Ensure the send queue table has the runtime columns needed by the producer. | deep |
| `ensure_simulation_state_control_table` | Ensure the simulation-state control table exists. | deep |
| `upsert_simulation_state_control` | Insert or update the control row that drives producer loop behavior. | deep |
| `read_simulation_state_control` | Read the producer control row for one dataset/run pair. | deep |
| `get_send_queue_status_counts` | Return row counts by queue status for producer monitoring. | deep |
| `claim_pending_send_queue_batch` | Compatibility wrapper around claim_pending_sensor_messages_batch. | deep |
| `claim_pending_sensor_messages_batch` | Atomically claim one producer batch from the send queue. | deep |
| `mark_claimed_batch_sent` | Mark all rows for a claim token as delivered to Kafka. | deep |
| `mark_claimed_batch_failed` | Mark all rows for a claim token as failed and store the delivery error. | deep |
| `mark_claimed_batch_sent_count` | Mark a claim as sent and return the number of updated queue rows. | deep |
| `mark_claimed_batch_failed_count` | Mark a claim as failed and return the number of updated queue rows. | deep |
| `requeue_failed_messages` | Move failed rows back to pending for retry, but only if they have not yet reached max_send_attempts. | deep |
| `release_stale_claims` | Return stale claimed rows to pending so the producer can retry them. | deep |

## Configuration Dependencies

- Dataset, run, stage, or recipe identifiers used for traceability.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `scalar_to_int` | `value, name` | Convert a required scalar result into an int and reject missing values. |
| `_grant_schema_usage_create` | `engine, *, schema, role_name` | Grant schema access needed by a queue runtime role. |
| `_apply_table_owner_and_grants` | `engine, *, schema, table_name, owner_role` | Assign ownership and DML grants for a runtime queue table. |
| `ensure_send_queue_runtime_columns` | `engine, *, schema, table_name` | Ensure the send queue table has the runtime columns needed by the producer. |
| `ensure_simulation_state_control_table` | `engine, *, schema, table_name, owner_role, apply_owner_and_grants` | Ensure the simulation-state control table exists. |
| `upsert_simulation_state_control` | `engine, *, dataset_id, run_id, is_enabled, producer_topic, producer_batch_size, producer_poll_seconds, max_send_attempts, schema, table_name` | Insert or update the control row that drives producer loop behavior. |
| `read_simulation_state_control` | `engine, *, dataset_id, run_id, schema, table_name` | Read the producer control row for one dataset/run pair. |
| `get_send_queue_status_counts` | `engine, *, schema, table_name` | Return row counts by queue status for producer monitoring. |
| `claim_pending_send_queue_batch` | `engine, *, dataset_id, run_id, batch_size, schema, table_name, producer_topic, producer_worker_id, claim_token` | Compatibility wrapper around claim_pending_sensor_messages_batch. |
| `claim_pending_sensor_messages_batch` | `engine, *, dataset_id, run_id, schema, queue_table, batch_size, producer_worker_id, producer_topic, claim_token, ensure_runtime_columns` | Atomically claim one producer batch from the send queue. |
| `mark_claimed_batch_sent` | `engine, *, claim_token, schema, table_name` | Mark all rows for a claim token as delivered to Kafka. |
| `mark_claimed_batch_failed` | `engine, *, claim_token, error_message, schema, table_name` | Mark all rows for a claim token as failed and store the delivery error. |
| `mark_claimed_batch_sent_count` | `engine, *, claim_token, schema, table_name` | Mark a claim as sent and return the number of updated queue rows. |
| `mark_claimed_batch_failed_count` | `engine, *, claim_token, error_message, schema, table_name` | Mark a claim as failed and return the number of updated queue rows. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.
- Source includes Kafka producer/consumer terminology or calls; helpers participate in synthetic streaming handoff when used by the synthetic pipeline.

## Artifact / SQL / File-System Interactions

- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.
- Kafka/PostgreSQL handoff: Source references producer, consumer, topic, or streaming-stage behavior.

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because the synthetic subsystem depends on repeatable staging, truth handling, streaming handoff, or final alignment behavior to make controlled pump telemetry tests explainable.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
