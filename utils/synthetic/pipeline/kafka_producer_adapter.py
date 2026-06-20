from __future__ import annotations

import json
import os
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence

from time import perf_counter

import pandas as pd

from utils.synthetic.pipeline.producer_queue_manager import (
    claim_pending_sensor_messages_batch,
    mark_claimed_batch_failed_count,
    mark_claimed_batch_sent_count,
    read_simulation_state_control,
)

try:
    from confluent_kafka import Producer
except ImportError:  # pragma: no cover
    Producer = None


# -----------------------------------------------------------------------------
# Env helpers
# -----------------------------------------------------------------------------

def _get_first_env_value(names: Sequence[str]) -> Optional[str]:
    """Return the first non-empty environment value from a list of aliases."""
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def get_kafka_bootstrap_servers_from_env(
    env_names: Sequence[str] = (
        "KAFKA_BOOTSTRAP_SERVERS",
        "BOOTSTRAP_SERVERS",
        "KAFKA_BROKERS",
    ),
) -> str:
    """Read Kafka bootstrap servers from the supported environment variables."""
    value = _get_first_env_value(env_names)
    if value is None:
        raise RuntimeError(
            "Missing Kafka bootstrap servers. Checked: "
            + ", ".join(env_names)
        )
    return value


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------

def _is_missing(value: Any) -> bool:
    """Return True for pandas/numpy missing values without failing on objects."""
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _normalize_scalar(value: Any) -> Any:
    """Convert pandas, datetime, Decimal, and numpy scalars into JSON-safe values."""
    if _is_missing(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def json_dumps_safe(payload: Mapping[str, Any]) -> str:
    """Serialize a Kafka payload after normalizing dataframe scalar types."""
    def _default(value: Any) -> Any:
        normalized = _normalize_scalar(value)
        if isinstance(normalized, (str, int, float, bool)) or normalized is None:
            return normalized
        return str(normalized)

    return json.dumps(payload, default=_default, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Producer creation
# -----------------------------------------------------------------------------

def build_confluent_producer_config(
    *,
    bootstrap_servers: str,
    client_id: str = "synthetic-telemetry-producer",
    acks: str = "all",
    linger_ms: int = 0,
    compression_type: Optional[str] = None,
    extra_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the Confluent producer configuration used by the synthetic sender."""
    config: dict[str, Any] = {
        "bootstrap.servers": str(bootstrap_servers).strip(),
        "client.id": str(client_id).strip(),
        "acks": str(acks).strip(),
        "linger.ms": int(linger_ms),
    }

    if compression_type:
        config["compression.type"] = str(compression_type).strip()

    if extra_config:
        config.update(extra_config)

    return config


def create_confluent_producer(
    *,
    bootstrap_servers: Optional[str] = None,
    client_id: str = "synthetic-telemetry-producer",
    acks: str = "all",
    linger_ms: int = 0,
    compression_type: Optional[str] = None,
    extra_config: Optional[dict[str, Any]] = None,
):
    """Create a Confluent Kafka producer from explicit or environment settings."""
    if Producer is None:
        raise ImportError(
            "confluent_kafka is not installed. Install 'confluent-kafka' in your environment."
        )

    resolved_bootstrap_servers = bootstrap_servers or get_kafka_bootstrap_servers_from_env()

    config = build_confluent_producer_config(
        bootstrap_servers=resolved_bootstrap_servers,
        client_id=client_id,
        acks=acks,
        linger_ms=linger_ms,
        compression_type=compression_type,
        extra_config=extra_config,
    )
    return Producer(config)


# -----------------------------------------------------------------------------
# Payload builder
# -----------------------------------------------------------------------------

def build_sensor_message_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """
    Convert one claimed send-queue row into the nested Kafka message payload.

    The payload groups observation, sensor, metadata, telemetry, and producer
    fields so the consumer can rebuild the landed Postgres row deterministically.
    """
    return {
        "message_key": _normalize_scalar(row.get("message_key")),
        "dataset_id": _normalize_scalar(row.get("dataset_id")),
        "run_id": _normalize_scalar(row.get("run_id")),
        "asset_id": _normalize_scalar(row.get("asset_id")),
        "generated_row_id": _normalize_scalar(row.get("generated_row_id")),
        "observation": {
            "observation_index": _normalize_scalar(row.get("observation_index")),
            "observation_timestamp": _normalize_scalar(row.get("observation_timestamp")),
            "batch_id": _normalize_scalar(row.get("batch_id")),
            "row_in_batch": _normalize_scalar(row.get("row_in_batch")),
            "global_cycle_id": _normalize_scalar(row.get("global_cycle_id")),
            "stream_state": _normalize_scalar(row.get("stream_state")),
            "phase": _normalize_scalar(row.get("phase")),
        },
        "sensor": {
            "sensor_name": _normalize_scalar(row.get("sensor_name")),
            "sensor_index": _normalize_scalar(row.get("sensor_index")),
            "sensor_value": _normalize_scalar(row.get("sensor_value")),
            "message_sequence_index": _normalize_scalar(row.get("message_sequence_index")),
        },
        "metadata": {
            "meta_episode_id": _normalize_scalar(row.get("meta_episode_id")),
            "meta_primary_fault_type": _normalize_scalar(row.get("meta_primary_fault_type")),
            "meta_magnitude": _normalize_scalar(row.get("meta_magnitude")),
            "created_at": _normalize_scalar(row.get("created_at")),
        },
        "telemetry": {
            "is_telemetry_event": _normalize_scalar(row.get("is_telemetry_event")),
            "telemetry_event_type": _normalize_scalar(row.get("telemetry_event_type")),
        },
        "producer": {
            "producer_send_attempt": _normalize_scalar(row.get("producer_send_attempt")),
            "queued_at": _normalize_scalar(row.get("queued_at")),
        },
    }


# -----------------------------------------------------------------------------
# Publish helpers
# -----------------------------------------------------------------------------

def publish_claimed_batch_to_kafka(
    *,
    producer,
    claimed_dataframe: pd.DataFrame,
    topic: str,
    flush_timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """
    Publish an already-claimed dataframe to Kafka.

    This function does not touch Postgres queue state. It only publishes.
    Queue success/failure updates should be handled by the caller.
    """
    if claimed_dataframe.empty:
        return {
            "claimed_rows": 0,
            "topic": str(topic).strip(),
            "delivered_count": 0,
            "error_count": 0,
            "errors": [],
        }

    working = claimed_dataframe.sort_values(
        by=["observation_index", "message_sequence_index", "sensor_index"],
        kind="stable",
    ).reset_index(drop=True)

    errors: list[str] = []
    delivered_count = 0

    def _delivery_callback(err, msg) -> None:
        """Track broker acknowledgement results for the current claimed batch."""
        nonlocal delivered_count
        if err is not None:
            errors.append(str(err))
        else:
            delivered_count += 1

    POLL_EVERY_MESSAGES = 1000  # poll periodically to drain broker delivery callbacks

    for row_number, row in enumerate(working.to_dict("records"), start=1):
        payload = build_sensor_message_payload(row)
        key = str(row.get("message_key"))
        value = json_dumps_safe(payload)

        # Back off when the local producer buffer is full, then retry the same row.
        produced = False
        while not produced:
            try:
                producer.produce(
                    topic=str(topic).strip(),
                    key=key,
                    value=value,
                    on_delivery=_delivery_callback,
                )
                produced = True
            except BufferError:
                producer.poll(0.25)

        if row_number % POLL_EVERY_MESSAGES == 0:
            producer.poll(0)

    remaining = producer.flush(timeout=float(flush_timeout_seconds))

    if remaining > 0:
        errors.append(f"{remaining} message(s) remained in producer queue after flush timeout.")

    return {
        "claimed_rows": int(len(working)),
        "topic": str(topic).strip(),
        "delivered_count": int(delivered_count),
        "error_count": int(len(errors)),
        "errors": errors,
    }


def run_send_queue_producer_once(
    engine,
    *,
    dataset_id: str,
    run_id: str,
    schema: str = "capstone",
    queue_table: str = "synthetic_sensor_messages_send_queue",
    control_table: str = "simulation_state_control",
    batch_size: Optional[int] = None,
    topic: Optional[str] = None,
    producer=None,
    bootstrap_servers: Optional[str] = None,
    producer_worker_id: str = "producer_worker_001",
    client_id: str = "synthetic-telemetry-producer",
    flush_timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """
    Claim the next pending queue batch, publish it to Kafka, then mark it sent
    or failed as a single claim group.
    """
    # Runtime control keeps topic and batch settings in Postgres for the run.
    control_row = read_simulation_state_control(
        engine=engine,
        dataset_id=dataset_id,
        run_id=run_id,
        schema=schema,
        table_name=control_table,
    )

    if not bool(control_row.get("is_enabled", False)):
        return {
            "status": "disabled",
            "claimed_rows": 0,
            "sent_rows": 0,
            "failed_rows": 0,
            "topic": control_row.get("producer_topic"),
        }

    resolved_batch_size = int(batch_size or control_row.get("producer_batch_size") or 500)
    resolved_topic = str(topic or control_row.get("producer_topic") or "").strip()

    if not resolved_topic:
        raise ValueError("Producer topic is empty. Set it in simulation_state_control or pass topic explicitly.")

    created_producer = False
    if producer is None:
        producer = create_confluent_producer(
            bootstrap_servers=bootstrap_servers,
            client_id=client_id,
        )
        created_producer = True

    claim_token, claimed_dataframe = claim_pending_sensor_messages_batch(
        engine=engine,
        dataset_id=dataset_id,
        run_id=run_id,
        batch_size=resolved_batch_size,
        schema=schema,
        queue_table=queue_table,
        producer_topic=resolved_topic,
        producer_worker_id=producer_worker_id,
        ensure_runtime_columns=False,
    )

    if claimed_dataframe.empty:
        return {
            "status": "empty",
            "claimed_rows": 0,
            "sent_rows": 0,
            "failed_rows": 0,
            "topic": resolved_topic,
        }

    try:
        # Only mark rows as sent after every produced message is acknowledged.
        publish_result = publish_claimed_batch_to_kafka(
            producer=producer,
            claimed_dataframe=claimed_dataframe,
            topic=resolved_topic,
            flush_timeout_seconds=flush_timeout_seconds,
        )

        expected_count = int(len(claimed_dataframe))
        delivered_count = int(publish_result.get("delivered_count", 0))
        error_count = int(publish_result.get("error_count", 0))

        if error_count > 0 or delivered_count != expected_count:
            mismatch_message = (
                f"Producer delivery mismatch for claim_token={claim_token}. "
                f"expected={expected_count}, delivered={delivered_count}, error_count={error_count}"
            )

            all_errors = list(publish_result.get("errors", []))
            all_errors.append(mismatch_message)

            error_message = " | ".join(all_errors)[:4000]

            failed_count = mark_claimed_batch_failed_count(
                engine=engine,
                claim_token=claim_token,
                error_message=error_message,
                schema=schema,
                table_name=queue_table,
            )

            return {
                "status": "failed",
                "claimed_rows": int(len(claimed_dataframe)),
                "sent_rows": 0,
                "failed_rows": int(failed_count),
                "topic": resolved_topic,
                "claim_token": claim_token,
                "errors": all_errors,
                "delivered_count": delivered_count,
                "expected_count": expected_count,
            }

        sent_count = mark_claimed_batch_sent_count(
            engine=engine,
            claim_token=claim_token,
            schema=schema,
            table_name=queue_table,
        )

        return {
            "status": "sent",
            "claimed_rows": int(len(claimed_dataframe)),
            "sent_rows": int(sent_count),
            "failed_rows": 0,
            "topic": resolved_topic,
            "claim_token": claim_token,
            "errors": [],
            "delivered_count": delivered_count,
            "expected_count": expected_count,
        }

    except Exception as exc:
        error_message = str(exc)[:4000]

        failed_count = mark_claimed_batch_failed_count(
            engine=engine,
            claim_token=claim_token,
            error_message=error_message,
            schema=schema,
            table_name=queue_table,
        )

        return {
            "status": "failed",
            "claimed_rows": int(len(claimed_dataframe)),
            "sent_rows": 0,
            "failed_rows": int(failed_count),
            "topic": resolved_topic,
            "claim_token": claim_token,
            "errors": [error_message],
            "delivered_count": 0,
            "expected_count": int(len(claimed_dataframe)),
        }
    
    finally:
        if created_producer:
            try:
                producer.flush(timeout=float(flush_timeout_seconds))
            except Exception:
                pass


def run_send_queue_producer_loop(
    engine,
    *,
    dataset_id: str,
    run_id: str,
    schema: str = "capstone",
    queue_table: str = "synthetic_sensor_messages_send_queue",
    control_table: str = "simulation_state_control",
    bootstrap_servers: Optional[str] = None,
    producer_worker_id: str = "producer_worker_001",
    client_id: str = "synthetic-telemetry-producer",
    max_batches: Optional[int] = None,
    stop_on_failure: bool = True,
    flush_timeout_seconds: float = 30.0,
    enable_progress_logging: bool = True,
    progress_every_batches: int = 1,
) -> list[dict[str, Any]]:
    """
    Repeatedly publish queue batches until:
    - queue is empty
    - control row disables the run
    - max_batches is reached
    - a failure occurs and stop_on_failure=True
    """
    results: list[dict[str, Any]] = []
    producer = create_confluent_producer(
        bootstrap_servers=bootstrap_servers,
        client_id=client_id,
    )

    batch_counter = 0
    cumulative_claimed_rows = 0
    cumulative_sent_rows = 0
    cumulative_failed_rows = 0

    try:
        while True:
            # Re-read control each batch so the run can be paused externally.
            if max_batches is not None and batch_counter >= int(max_batches):
                if enable_progress_logging:
                    print(
                        f"[producer-loop] max_batches reached: {max_batches}",
                        flush=True,
                    )
                break

            control_row = read_simulation_state_control(
                engine=engine,
                dataset_id=dataset_id,
                run_id=run_id,
                schema=schema,
                table_name=control_table,
            )

            if not bool(control_row.get("is_enabled", False)):
                disabled_result = {
                    "status": "disabled",
                    "claimed_rows": 0,
                    "sent_rows": 0,
                    "failed_rows": 0,
                    "topic": control_row.get("producer_topic"),
                }
                results.append(disabled_result)

                if enable_progress_logging:
                    print(
                        "[producer-loop] disabled by simulation_state_control.",
                        flush=True,
                    )

                break

            batch_started_at = time.perf_counter()

            result = run_send_queue_producer_once(
                engine=engine,
                dataset_id=dataset_id,
                run_id=run_id,
                schema=schema,
                queue_table=queue_table,
                control_table=control_table,
                producer=producer,
                producer_worker_id=producer_worker_id,
                client_id=client_id,
                flush_timeout_seconds=flush_timeout_seconds,
            )

            batch_elapsed_seconds = time.perf_counter() - batch_started_at

            results.append(result)
            batch_counter += 1

            claimed_rows = int(result.get("claimed_rows", 0) or 0)
            sent_rows = int(result.get("sent_rows", 0) or 0)
            failed_rows = int(result.get("failed_rows", 0) or 0)

            messages_per_second = (
                sent_rows / batch_elapsed_seconds
                if batch_elapsed_seconds > 0
                else 0.0
            )

            cumulative_claimed_rows += claimed_rows
            cumulative_sent_rows += sent_rows
            cumulative_failed_rows += failed_rows

            should_print_progress = (
                enable_progress_logging
                and progress_every_batches > 0
                and batch_counter % int(progress_every_batches) == 0
            )

            if should_print_progress:
                print(
                    "[producer-loop] "
                    f"batch={batch_counter:,} | "
                    f"status={result.get('status')} | "
                    f"seconds={batch_elapsed_seconds:,.2f} | "
                    f"msg_per_sec={messages_per_second:,.0f} | "
                    f"claimed={claimed_rows:,} | "
                    f"sent={sent_rows:,} | "
                    f"failed={failed_rows:,} | "
                    f"delivered={int(result.get('delivered_count', 0) or 0):,} | "
                    f"cumulative_sent={cumulative_sent_rows:,} | "
                    f"cumulative_failed={cumulative_failed_rows:,}",
                    flush=True,
                )

            if result["status"] == "empty":
                if enable_progress_logging:
                    print("[producer-loop] queue is empty.", flush=True)
                break

            if result["status"] == "failed" and stop_on_failure:
                if enable_progress_logging:
                    print(
                        "[producer-loop] stopping on failure. "
                        f"errors={result.get('errors', [])}",
                        flush=True,
                    )
                break

            poll_seconds = float(control_row.get("producer_poll_seconds") or 0.0)
            if poll_seconds > 0:
                time.sleep(poll_seconds)

    finally:
        try:
            producer.flush(timeout=float(flush_timeout_seconds))
        except Exception:
            pass

        if enable_progress_logging:
            print(
                "[producer-loop] finished | "
                f"batches={batch_counter:,} | "
                f"cumulative_claimed={cumulative_claimed_rows:,} | "
                f"cumulative_sent={cumulative_sent_rows:,} | "
                f"cumulative_failed={cumulative_failed_rows:,}",
                flush=True,
            )

    return results



__all__ = [
    "get_kafka_bootstrap_servers_from_env",
    "build_confluent_producer_config",
    "create_confluent_producer",
    "build_sensor_message_payload",
    "publish_claimed_batch_to_kafka",
    "run_send_queue_producer_once",
    "run_send_queue_producer_loop",
]
