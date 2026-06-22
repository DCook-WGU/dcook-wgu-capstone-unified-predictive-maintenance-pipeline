# utils/synthetic/pipeline/kafka_producer_adapter.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `kafka_producer_adapter.py` that need deeper explanation than the 071d module-level reference. The selected functions publish claimed send-queue rows to Kafka and coordinate one queue claim/publish/status-update cycle.

## Source Grounding

Sources used:

- `utils/synthetic/pipeline/kafka_producer_adapter.py`
- `utils/synthetic/pipeline/producer_queue_manager.py`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__synthetic__pipeline__kafka_producer_adapter_module_reference.md`
- `function_inventory.json`
- `technical_reference/01_notebook_workflow_references/`
- `technical_reference/00_project_manual/`
- Documentation standards under `docs_agent_pack/00_STANDARDS/`

The active utility source file is the source of truth.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
| -------- | -------------- | ------------------------ |
| `publish_claimed_batch_to_kafka` | Publishes an already-claimed queue dataframe to Kafka | Synthetic producer publish step |
| `run_send_queue_producer_once` | Claims one pending queue batch, publishes it, and marks it sent or failed | Synthetic producer lifecycle |

## Module-Level Technical Context

`kafka_producer_adapter.py` bridges the PostgreSQL send queue and the Kafka producer. It builds JSON-safe payloads from queue rows, publishes them with Confluent Kafka producer semantics when available, and coordinates durable queue status updates through queue-manager helpers.

## Deep Function References

### `publish_claimed_batch_to_kafka`

#### Functional Purpose

`publish_claimed_batch_to_kafka` publishes an already-claimed dataframe of queue rows to a Kafka topic. It serializes each row into the project sensor-message payload shape, uses `message_key` as the Kafka key, polls periodically for delivery callbacks, flushes the producer, and returns a delivery summary.

#### Pipeline Context

This function supports the publish portion of the synthetic Kafka producer path. It is called by `run_send_queue_producer_once` after queue rows have already been claimed from PostgreSQL.

#### Inputs and Assumptions

Important inputs include a Kafka producer object, a claimed queue dataframe, a topic, and flush timeout seconds.

The dataframe is expected to contain send-queue fields such as `message_key`, dataset/run/asset identity, observation fields, sensor fields, metadata fields, telemetry fields, and producer attempt/queued fields. The function sorts rows by `observation_index`, `message_sequence_index`, and `sensor_index` before publishing.

#### Outputs and Return Contract

The function returns a dictionary with:

- `claimed_rows`
- `topic`
- `delivered_count`
- `error_count`
- `errors`

For an empty claimed dataframe, the returned counts are zero and no publish calls are made.

#### Side Effects

Confirmed side effects are Kafka producer interactions: `produce`, periodic `poll`, delivery callbacks, and `flush`. The function does not update PostgreSQL queue state.

#### Failure Behavior and Guardrails

The function retries the same row after `BufferError` by polling the producer buffer. Delivery callback errors are collected. If `flush` leaves messages in the producer queue after the timeout, an error message is added. Exceptions outside the handled `BufferError` path propagate to the caller.

#### Lineage, Idempotency, and Reproducibility Role

Message identity is carried through `message_key`, while dataset/run/asset and observation/sensor metadata are serialized into the Kafka payload. The function does not enforce idempotency by itself; idempotent queue state is handled by the caller's claim and status-update cycle.

#### Why This Function Matters

The project needs a controlled boundary between durable queue state and Kafka publication. This function keeps publishing focused on payload serialization and broker acknowledgement rather than mixing in database status changes.

#### Verification Method

- Confirm empty claimed dataframes return zero-count summaries.
- Confirm published keys match queue `message_key` values.
- Confirm `delivered_count` equals claimed rows when all broker acknowledgements succeed.
- Confirm callback errors increase `error_count`.
- Confirm PostgreSQL queue status does not change when this function is called alone.

### `run_send_queue_producer_once`

#### Functional Purpose

`run_send_queue_producer_once` executes one producer cycle: read the run control row, resolve topic and batch size, claim pending queue rows, publish the claimed rows to Kafka, and mark the claim group as sent or failed.

#### Pipeline Context

Active notebook source confirms use in the synthetic Kafka producer adapter notebook as an optional one-batch smoke test, and in an all-in-one synthetic workflow notebook. This function is the one-shot controlled producer path for testing or loop orchestration.

#### Inputs and Assumptions

Important inputs include the SQLAlchemy engine, `dataset_id`, `run_id`, schema, queue table, control table, optional batch size, optional topic, optional producer object, bootstrap servers, worker id, client id, and flush timeout.

The function assumes a `simulation_state_control` row exists for the dataset/run. It uses that control row for `is_enabled`, default producer topic, and producer batch size. If no producer object is supplied, it creates one from explicit or environment bootstrap settings.

#### Outputs and Return Contract

The function returns a status dictionary. Source-confirmed statuses include:

- `disabled`
- `empty`
- `sent`
- `failed`

Returned fields include claimed, sent, failed, topic, claim token when applicable, errors, delivered count, and expected count depending on the status.

#### Side Effects

Confirmed side effects are:

- Reads the producer control row from PostgreSQL.
- Claims pending queue rows by updating `queue_status` to `claimed`, assigning `claim_token`, `claimed_at`, producer topic, worker id, and delivery status.
- Publishes claimed rows to Kafka through `publish_claimed_batch_to_kafka`.
- Marks claimed rows `sent` with sent/ack timestamps when all deliveries are acknowledged.
- Marks claimed rows `failed` and stores delivery error text when delivery mismatches or exceptions occur.
- Flushes a producer created inside the function in the `finally` block.

#### Failure Behavior and Guardrails

The function returns `disabled` when the control row has `is_enabled=False`. It raises `ValueError` if the resolved topic is empty. If no pending rows are claimed, it returns `empty`. Delivery mismatches are treated as failed claims. Exceptions during publishing are caught, written to the queue failure status, and returned as a failed result.

The underlying claim helper validates non-empty dataset/run identifiers and positive batch size, and uses `FOR UPDATE SKIP LOCKED` to prevent concurrent producers from claiming the same rows.

#### Lineage, Idempotency, and Reproducibility Role

The lifecycle is keyed by `dataset_id`, `run_id`, `message_key`, and `claim_token`. Queue state transitions are durable in PostgreSQL: pending to claimed, then claimed to sent or failed. The function only marks rows sent after all claimed messages are acknowledged, which protects the queue from reporting sent rows when the broker did not confirm delivery.

#### Why This Function Matters

This function is the controlled Kafka handoff unit. It keeps producer testing bounded to one batch and makes success/failure auditable through queue status fields.

#### Verification Method

- Confirm disabled control rows return `status="disabled"` and claim no rows.
- Confirm an empty queue returns `status="empty"`.
- Confirm successful publish updates claimed rows to `sent`.
- Confirm delivery mismatch or publish exception updates claimed rows to `failed`.
- Confirm returned `delivered_count` equals `expected_count` for successful runs.
- Confirm `claim_token` links returned status to the queue rows updated in PostgreSQL.

## Cross-Function Relationships

`run_send_queue_producer_once` calls `publish_claimed_batch_to_kafka` after claiming queue rows. The publish function reports broker acknowledgement results; the one-shot producer function interprets those results and updates PostgreSQL queue status as sent or failed.

## Source-Limited Items

- Kafka partition assignment behavior is Not determined from available source; the selected functions do not explicitly set partitions.
- Retry behavior after a failed claim is not handled by these selected functions beyond marking rows failed.
