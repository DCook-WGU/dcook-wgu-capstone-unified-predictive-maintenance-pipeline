# Synthetic Utility Reference: producer_queue_manager.py

Source path:

`utils/synthetic/pipeline/producer_queue_manager.py`

## Purpose

Manages controlled claiming and send status updates for queued synthetic messages.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `scalar_to_int` | `value, name` | Convert a required scalar result into an int and reject missing values. |
| Function | `_grant_schema_usage_create` | `engine` | Grant schema access needed by a queue runtime role. |
| Function | `_apply_table_owner_and_grants` | `engine` | Assign ownership and DML grants for a runtime queue table. |
| Function | `ensure_send_queue_runtime_columns` | `engine` | Ensure the send queue table has the runtime columns needed by the producer. This is safe to run repeatedly. Important: This function assumes the queue table was already created with the correct owner/grants by the queue-stage builder or bootstrap logic. |
| Function | `ensure_simulation_state_control_table` | `engine` | Ensure the simulation-state control table exists. If apply_owner_and_grants=True, this function will also assign ownership and grants. That should only be used from an admin/bootstrap step. |
| Function | `upsert_simulation_state_control` | `engine` | Insert or update the control row that drives producer loop behavior. The control row stores whether a synthetic run is active, which topic to publish to, the producer batch size, polling delay, and retry ceiling. |
| Function | `read_simulation_state_control` | `engine` | Read the producer control row for one dataset/run pair. |
| Function | `get_send_queue_status_counts` | `engine` | Return row counts by queue status for producer monitoring. |
| Function | `claim_pending_send_queue_batch` | `engine` | Compatibility wrapper around claim_pending_sensor_messages_batch. Prefer claim_pending_sensor_messages_batch for new producer code because it returns both claim_token and dataframe. |
| Function | `claim_pending_sensor_messages_batch` | `engine` | Atomically claim one producer batch from the send queue. This is the preferred queue-claim function for the synthetic sensor-message producer path. It filters by dataset_id and run_id, claims rows in deterministic send order, and uses FOR UPDATE SKIP LOCKED so multiple workers cannot claim the same rows. |
| Function | `mark_claimed_batch_sent` | `engine` | Mark all rows for a claim token as delivered to Kafka. |
| Function | `mark_claimed_batch_failed` | `engine` | Mark all rows for a claim token as failed and store the delivery error. |
| Function | `mark_claimed_batch_sent_count` | `engine` | Mark a claim as sent and return the number of updated queue rows. |
| Function | `mark_claimed_batch_failed_count` | `engine` | Mark a claim as failed and return the number of updated queue rows. |
| Function | `requeue_failed_messages` | `engine` | Move failed rows back to pending for retry, but only if they have not yet reached max_send_attempts. Since the queue starts with producer_send_attempt = 1, this increments attempts when a failed row is requeued. |
| Function | `release_stale_claims` | `engine` | Return stale claimed rows to pending so the producer can retry them. |

## Configuration Dependencies

- Uses SQL schema or table name settings.
- Uses dataset identity values.
- Uses filesystem paths or resolved artifact locations.
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

- No direct artifact write pattern detected in this module.

## Downstream Usage

- `notebooks/synthetic/synthetic_06.5_testing_producer_queue_manager.ipynb`
- `notebooks/synthetic/synthetic_06_producer_queue_manager.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
