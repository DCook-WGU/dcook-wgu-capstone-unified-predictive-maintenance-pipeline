from __future__ import annotations

import json
import os
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence

from confluent_kafka import Consumer, KafkaException


import pandas as pd

from utils.postgres_util import (
    sanitize_sql_identifier,
    create_schema_if_not_exists,
    execute_sql,
    read_sql_dataframe,
)
from utils.layer_postgres_writer import write_layer_dataframe

try:
    from confluent_kafka import Consumer
except ImportError:  # pragma: no cover
    Consumer = None


# -----------------------------------------------------------------------------
# Env helpers
# -----------------------------------------------------------------------------

def _get_first_env_value(names: Sequence[str]) -> Optional[str]:
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
    value = _get_first_env_value(env_names)
    if value is None:
        raise RuntimeError(
            "Missing Kafka bootstrap servers. Checked: "
            + ", ".join(env_names)
        )
    return value


def get_kafka_consumer_group_from_env(
    env_names: Sequence[str] = (
        "KAFKA_CONSUMER_GROUP_ID",
        "CONSUMER_GROUP_ID",
    ),
    default: str = "synthetic-telemetry-consumer-group",
) -> str:
    return _get_first_env_value(env_names) or str(default).strip()


# -----------------------------------------------------------------------------
# Serialization / normalization helpers
# -----------------------------------------------------------------------------

def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _normalize_scalar(value: Any) -> Any:
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


def _parse_message_value(raw_value: Any) -> dict[str, Any]:
    if raw_value is None:
        return {}

    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8")

    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return {}
        return json.loads(raw_value)

    if isinstance(raw_value, dict):
        return raw_value

    raise TypeError(f"Unsupported Kafka message value type: {type(raw_value)}")


def _get_nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def build_consumed_message_record(
    *,
    payload: Mapping[str, Any],
    kafka_topic: str,
    kafka_partition: int,
    kafka_offset: int,
    consumer_group_id: str,
    consumer_worker_id: str,
    raw_value_text: Optional[str] = None,
) -> dict[str, Any]:
    """
    Flatten the producer payload into a landed Postgres row.
    """
    return {
        "dataset_id": _normalize_scalar(payload.get("dataset_id")),
        "run_id": _normalize_scalar(payload.get("run_id")),
        "asset_id": _normalize_scalar(payload.get("asset_id")),
        "message_key": _normalize_scalar(payload.get("message_key")),
        "generated_row_id": _normalize_scalar(payload.get("generated_row_id")),

        "observation_index": _normalize_scalar(_get_nested(payload, "observation", "observation_index")),
        "observation_timestamp": _normalize_scalar(_get_nested(payload, "observation", "observation_timestamp")),
        "batch_id": _normalize_scalar(_get_nested(payload, "observation", "batch_id")),
        "row_in_batch": _normalize_scalar(_get_nested(payload, "observation", "row_in_batch")),
        "global_cycle_id": _normalize_scalar(_get_nested(payload, "observation", "global_cycle_id")),
        "stream_state": _normalize_scalar(_get_nested(payload, "observation", "stream_state")),
        "phase": _normalize_scalar(_get_nested(payload, "observation", "phase")),

        "sensor_name": _normalize_scalar(_get_nested(payload, "sensor", "sensor_name")),
        "sensor_index": _normalize_scalar(_get_nested(payload, "sensor", "sensor_index")),
        "sensor_value": _normalize_scalar(_get_nested(payload, "sensor", "sensor_value")),
        "message_sequence_index": _normalize_scalar(_get_nested(payload, "sensor", "message_sequence_index")),

        "meta_episode_id": _normalize_scalar(_get_nested(payload, "metadata", "meta_episode_id")),
        "meta_primary_fault_type": _normalize_scalar(_get_nested(payload, "metadata", "meta_primary_fault_type")),
        "meta_magnitude": _normalize_scalar(_get_nested(payload, "metadata", "meta_magnitude")),
        "created_at": _normalize_scalar(_get_nested(payload, "metadata", "created_at")),

        "is_telemetry_event": _normalize_scalar(_get_nested(payload, "telemetry", "is_telemetry_event")),
        "telemetry_event_type": _normalize_scalar(_get_nested(payload, "telemetry", "telemetry_event_type")),

        "producer_send_attempt": _normalize_scalar(_get_nested(payload, "producer", "producer_send_attempt")),
        "queued_at": _normalize_scalar(_get_nested(payload, "producer", "queued_at")),

        "consumer_received_at": pd.Timestamp.utcnow(),
        "kafka_topic": str(kafka_topic).strip(),
        "kafka_partition": int(kafka_partition),
        "kafka_offset": int(kafka_offset),
        "consumer_group_id": str(consumer_group_id).strip(),
        "consumer_worker_id": str(consumer_worker_id).strip(),

        "payload_json": raw_value_text,
        "is_duplicate": False,
        "rebuild_status": "pending",
    }


# -----------------------------------------------------------------------------
# Landed table helpers
# -----------------------------------------------------------------------------

def _get_existing_columns(engine, *, schema: str, table: str) -> set[str]:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table)

    columns_dataframe = read_sql_dataframe(
        engine,
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema_name
          AND table_name = :table_name
        """,
        params={"schema_name": safe_schema, "table_name": safe_table},
    )
    return set(columns_dataframe["column_name"].astype(str).tolist())


def _infer_alter_column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE PRECISION"
    if pd.api.types.is_datetime64tz_dtype(series):
        return "TIMESTAMPTZ"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    return "TEXT"


def _add_missing_columns(engine, *, schema: str, table: str, dataframe: pd.DataFrame) -> None:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table)

    existing = _get_existing_columns(engine, schema=safe_schema, table=safe_table)
    desired = [sanitize_sql_identifier(column) for column in dataframe.columns]

    missing = [column for column in desired if column not in existing]
    if not missing:
        return

    working = dataframe.copy()
    working.columns = [sanitize_sql_identifier(column) for column in working.columns]

    for column in missing:
        column_type = _infer_alter_column_type(working[column])
        execute_sql(
            engine,
            f'ALTER TABLE "{safe_schema}"."{safe_table}" ADD COLUMN "{column}" {column_type};',
        )

    print(f"[consumer] Added {len(missing)} new columns to {safe_schema}.{safe_table}")


def ensure_consumed_stage_table_exists(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_consumed_stage",
) -> str:
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    CREATE TABLE IF NOT EXISTS "{safe_schema}"."{safe_table}" (
        message_key TEXT,
        kafka_topic TEXT NOT NULL,
        kafka_partition INTEGER NOT NULL,
        kafka_offset BIGINT NOT NULL,
        consumer_received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        payload_json TEXT,
        is_duplicate BOOLEAN NOT NULL DEFAULT FALSE,
        rebuild_status TEXT,
        observation_index BIGINT,
        sensor_index BIGINT,
        PRIMARY KEY (kafka_topic, kafka_partition, kafka_offset)
    );
    """
    execute_sql(engine, sql)

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_obs_sensor"
        ON "{safe_schema}"."{safe_table}" (observation_index, sensor_index);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_rebuild_status"
        ON "{safe_schema}"."{safe_table}" (rebuild_status);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_message_key"
        ON "{safe_schema}"."{safe_table}" (message_key);
        '''
    )

    return safe_table


def write_consumed_messages_batch(
    engine,
    dataframe: pd.DataFrame,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_consumed_stage",
) -> str:
    """
    Append a consumer batch into the landed message table.

    Deduping is handled before write by kafka topic/partition/offset.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if dataframe.empty:
        return sanitize_sql_identifier(table_name)

    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = ensure_consumed_stage_table_exists(
        engine,
        schema=schema,
        table_name=table_name,
    )

    working = dataframe.copy()
    working.columns = [sanitize_sql_identifier(column) for column in working.columns]

    _add_missing_columns(
        engine,
        schema=safe_schema,
        table=safe_table,
        dataframe=working,
    )

    existing_offsets = read_sql_dataframe(
        engine,
        f"""
        SELECT kafka_topic, kafka_partition, kafka_offset
        FROM "{safe_schema}"."{safe_table}"
        WHERE kafka_topic IN :topic_list
        """,
        params={"topic_list": tuple(working["kafka_topic"].dropna().astype(str).unique().tolist())},
    )

    if not existing_offsets.empty:
        existing_keys = set(
            zip(
                existing_offsets["kafka_topic"].astype(str),
                existing_offsets["kafka_partition"].astype(int),
                existing_offsets["kafka_offset"].astype(int),
            )
        )

        incoming_keys = list(
            zip(
                working["kafka_topic"].astype(str),
                working["kafka_partition"].astype(int),
                working["kafka_offset"].astype(int),
            )
        )

        keep_mask = [key not in existing_keys for key in incoming_keys]
        working = working.loc[keep_mask].reset_index(drop=True)

    if working.empty:
        return safe_table

    return write_layer_dataframe(
        engine=engine,
        dataframe=working,
        schema=safe_schema,
        table_name=safe_table,
        if_exists="append",
        index=False,
    )


# -----------------------------------------------------------------------------
# Consumer creation
# -----------------------------------------------------------------------------

def build_confluent_consumer_config(
    *,
    bootstrap_servers: str,
    consumer_group_id: str,
    auto_offset_reset: str = "earliest",
    enable_auto_commit: bool = False,
    extra_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "bootstrap.servers": str(bootstrap_servers).strip(),
        "group.id": str(consumer_group_id).strip(),
        "auto.offset.reset": str(auto_offset_reset).strip(),
        "enable.auto.commit": bool(enable_auto_commit),
    }

    if extra_config:
        config.update(extra_config)

    return config


def create_confluent_consumer(
    *,
    bootstrap_servers: Optional[str] = None,
    consumer_group_id: Optional[str] = None,
    auto_offset_reset: str = "earliest",
    enable_auto_commit: bool = False,
    extra_config: Optional[dict[str, Any]] = None,
):
    if Consumer is None:
        raise ImportError(
            "confluent_kafka is not installed. Install 'confluent-kafka' in your environment."
        )

    resolved_bootstrap_servers = bootstrap_servers or get_kafka_bootstrap_servers_from_env()
    resolved_group_id = consumer_group_id or get_kafka_consumer_group_from_env()

    config = build_confluent_consumer_config(
        bootstrap_servers=resolved_bootstrap_servers,
        consumer_group_id=resolved_group_id,
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=enable_auto_commit,
        extra_config=extra_config,
    )
    return Consumer(config)


# -----------------------------------------------------------------------------
# Consume / land helpers
# -----------------------------------------------------------------------------

def consume_kafka_messages_once(
    *,
    consumer,
    topic: str,
    max_messages: int = 500,
    poll_timeout_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Poll a finite batch of Kafka messages from a subscribed topic.
    """
    if max_messages <= 0:
        raise ValueError("max_messages must be > 0")

    #consumer.subscribe([str(topic).strip()])

    messages: list[dict[str, Any]] = []

    while len(messages) < max_messages:
        msg = consumer.poll(timeout=float(poll_timeout_seconds))
        if msg is None:
            break

        if msg.error():
            raise RuntimeError(str(msg.error()))

        raw_value = msg.value()
        raw_value_text = raw_value.decode("utf-8") if isinstance(raw_value, bytes) else str(raw_value)
        payload = _parse_message_value(raw_value)

        messages.append(
            {
                "payload": payload,
                "raw_value_text": raw_value_text,
                "kafka_topic": msg.topic(),
                "kafka_partition": msg.partition(),
                "kafka_offset": msg.offset(),
            }
        )

    return messages


def land_consumed_messages_to_postgres(
    engine,
    *,
    consumed_messages: list[dict[str, Any]],
    consumer_group_id: str,
    consumer_worker_id: str,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_consumed_stage",
) -> dict[str, Any]:
    """
    Normalize a consumed Kafka batch and append it into the landed stage table.
    """
    if not consumed_messages:
        return {
            "received_count": 0,
            "written_count": 0,
            "table_name": sanitize_sql_identifier(table_name),
        }

    records: list[dict[str, Any]] = []

    for item in consumed_messages:
        payload = item["payload"]
        record = build_consumed_message_record(
            payload=payload,
            kafka_topic=item["kafka_topic"],
            kafka_partition=item["kafka_partition"],
            kafka_offset=item["kafka_offset"],
            consumer_group_id=consumer_group_id,
            consumer_worker_id=consumer_worker_id,
            raw_value_text=item.get("raw_value_text"),
        )
        records.append(record)

    dataframe = pd.DataFrame(records)

    table_written = write_consumed_messages_batch(
        engine=engine,
        dataframe=dataframe,
        schema=schema,
        table_name=table_name,
    )

    return {
        "received_count": int(len(consumed_messages)),
        "written_count": int(len(dataframe)),
        "table_name": table_written,
    }


def run_kafka_consumer_to_postgres_once(
    engine,
    *,
    topic: str,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_consumed_stage",
    max_messages: int = 500,
    poll_timeout_seconds: float = 1.0,
    consumer_group_id: Optional[str] = None,
    consumer_worker_id: str = "consumer_worker_001",
    bootstrap_servers: Optional[str] = None,
    consumer=None,
    commit_on_success: bool = True,
    auto_offset_reset: str = "earliest",
) -> dict[str, Any]:
    """
    Consume a finite Kafka batch and land it to Postgres.

    Commits offsets only after successful Postgres landing if commit_on_success=True.
    """
    resolved_group_id = consumer_group_id or get_kafka_consumer_group_from_env()

    created_consumer = False
    if consumer is None:
        consumer = create_confluent_consumer(
            bootstrap_servers=bootstrap_servers,
            consumer_group_id=resolved_group_id,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=False,
            extra_config={
                "session.timeout.ms": 120000,
                "max.poll.interval.ms": 600000,
                "heartbeat.interval.ms": 3000,
            },
        )
        consumer.subscribe([str(topic).strip()])
        created_consumer = True

    try:
        consumed_messages = consume_kafka_messages_once(
            consumer=consumer,
            topic=topic,
            max_messages=max_messages,
            poll_timeout_seconds=poll_timeout_seconds,
        )

        if not consumed_messages:
            return {
                "status": "empty",
                "received_count": 0,
                "written_count": 0,
                "table_name": sanitize_sql_identifier(table_name),
                "topic": str(topic).strip(),
            }

        result = land_consumed_messages_to_postgres(
            engine=engine,
            consumed_messages=consumed_messages,
            consumer_group_id=resolved_group_id,
            consumer_worker_id=consumer_worker_id,
            schema=schema,
            table_name=table_name,
        )

        if commit_on_success:
            try:
                consumer.commit(asynchronous=False)
            except KafkaException as exc:
                return {
                    "status": "commit_failed_assignment_lost",
                    "received_count": int(result["received_count"]),
                    "written_count": int(result["written_count"]),
                    "table_name": result["table_name"],
                    "topic": str(topic).strip(),
                    "error": str(exc),
                }

        return {
            "status": "landed",
            "received_count": int(result["received_count"]),
            "written_count": int(result["written_count"]),
            "table_name": result["table_name"],
            "topic": str(topic).strip(),
        }

    finally:
        if created_consumer:
            try:
                consumer.close()
            except Exception:
                pass

            

def run_kafka_consumer_to_postgres_loop(
    engine,
    *,
    topic: str,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_consumed_stage",
    max_messages: int = 500,
    poll_timeout_seconds: float = 1.0,
    consumer_group_id: Optional[str] = None,
    consumer_worker_id: str = "consumer_worker_001",
    bootstrap_servers: Optional[str] = None,
    auto_offset_reset: str = "earliest",
    max_batches: Optional[int] = None,
    idle_sleep_seconds: float = 0.0,
    stop_on_empty: bool = True,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    resolved_group_id = consumer_group_id or get_kafka_consumer_group_from_env()
    '''
    consumer = create_confluent_consumer(
        bootstrap_servers=bootstrap_servers,
        consumer_group_id=resolved_group_id,
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=False,
    )
    '''

    consumer = create_confluent_consumer(
        bootstrap_servers=bootstrap_servers,
        consumer_group_id=resolved_group_id,
        auto_offset_reset=auto_offset_reset,
        enable_auto_commit=False,
        extra_config={
            "session.timeout.ms": 120000,
            "max.poll.interval.ms": 600000,
            "heartbeat.interval.ms": 3000,
        },
    )

    consumer.subscribe([str(topic).strip()])

    batch_counter = 0

    try:
        while True:
            if max_batches is not None and batch_counter >= int(max_batches):
                break

            result = run_kafka_consumer_to_postgres_once(
                engine=engine,
                topic=topic,
                schema=schema,
                table_name=table_name,
                max_messages=max_messages,
                poll_timeout_seconds=poll_timeout_seconds,
                consumer_group_id=resolved_group_id,
                consumer_worker_id=consumer_worker_id,
                consumer=consumer,
                commit_on_success=True,
                auto_offset_reset=auto_offset_reset,
            )
            results.append(result)
            batch_counter += 1

            if result["status"] == "empty" and stop_on_empty:
                break

            if idle_sleep_seconds > 0:
                time.sleep(float(idle_sleep_seconds))

    finally:
        try:
            consumer.close()
        except Exception:
            pass

    return results


__all__ = [
    "get_kafka_bootstrap_servers_from_env",
    "get_kafka_consumer_group_from_env",
    "build_confluent_consumer_config",
    "create_confluent_consumer",
    "build_consumed_message_record",
    "ensure_consumed_stage_table_exists",
    "write_consumed_messages_batch",
    "consume_kafka_messages_once",
    "land_consumed_messages_to_postgres",
    "run_kafka_consumer_to_postgres_once",
    "run_kafka_consumer_to_postgres_loop",
]