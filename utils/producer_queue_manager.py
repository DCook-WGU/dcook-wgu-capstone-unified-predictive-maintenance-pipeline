from __future__ import annotations

import uuid
from typing import Optional

import pandas as pd

from utils.postgres_util import (
    sanitize_sql_identifier,
    create_schema_if_not_exists,
    execute_sql,
    read_sql_dataframe,
)


# -----------------------------------------------------------------------------
# Queue table runtime column helpers
# -----------------------------------------------------------------------------

def ensure_send_queue_runtime_columns(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_send_queue",
) -> None:
    """
    Ensure the send queue table has the runtime columns needed by the producer.

    This is safe to run repeatedly.
    """
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    statements = [
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS claim_token TEXT;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS claimed_at TIMESTAMPTZ;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS producer_topic TEXT;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS producer_worker_id TEXT;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS producer_ack_at TIMESTAMPTZ;
        ''',
    ]

    for sql in statements:
        execute_sql(engine, sql)

    index_statements = [
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_claim_token"
        ON "{safe_schema}"."{safe_table}" (claim_token);
        ''',
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_queue_claim_order"
        ON "{safe_schema}"."{safe_table}" (queue_status, observation_index, message_sequence_index, sensor_index);
        ''',
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_claimed_at"
        ON "{safe_schema}"."{safe_table}" (claimed_at);
        ''',
    ]

    for sql in index_statements:
        execute_sql(engine, sql)


# -----------------------------------------------------------------------------
# Simulation state control table
# -----------------------------------------------------------------------------

def ensure_simulation_state_control_table(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "simulation_state_control",
) -> str:
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    CREATE TABLE IF NOT EXISTS "{safe_schema}"."{safe_table}" (
        control_id BIGSERIAL PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
        producer_topic TEXT,
        producer_batch_size INTEGER NOT NULL DEFAULT 500,
        producer_poll_seconds DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        max_send_attempts INTEGER NOT NULL DEFAULT 3,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (dataset_id, run_id)
    );
    """
    execute_sql(engine, sql)

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_dataset_run"
        ON "{safe_schema}"."{safe_table}" (dataset_id, run_id);
        '''
    )

    return safe_table


def upsert_simulation_state_control(
    engine,
    *,
    dataset_id: str,
    run_id: str,
    is_enabled: bool = True,
    producer_topic: Optional[str] = None,
    producer_batch_size: int = 500,
    producer_poll_seconds: float = 0.0,
    max_send_attempts: int = 3,
    schema: str = "capstone",
    table_name: str = "simulation_state_control",
) -> None:
    if producer_batch_size <= 0:
        raise ValueError("producer_batch_size must be > 0")
    if producer_poll_seconds < 0:
        raise ValueError("producer_poll_seconds must be >= 0")
    if max_send_attempts <= 0:
        raise ValueError("max_send_attempts must be > 0")

    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = ensure_simulation_state_control_table(
        engine,
        schema=schema,
        table_name=table_name,
    )

    sql = f"""
    INSERT INTO "{safe_schema}"."{safe_table}" (
        dataset_id,
        run_id,
        is_enabled,
        producer_topic,
        producer_batch_size,
        producer_poll_seconds,
        max_send_attempts,
        updated_at
    )
    VALUES (
        :dataset_id,
        :run_id,
        :is_enabled,
        :producer_topic,
        :producer_batch_size,
        :producer_poll_seconds,
        :max_send_attempts,
        now()
    )
    ON CONFLICT (dataset_id, run_id)
    DO UPDATE SET
        is_enabled = EXCLUDED.is_enabled,
        producer_topic = EXCLUDED.producer_topic,
        producer_batch_size = EXCLUDED.producer_batch_size,
        producer_poll_seconds = EXCLUDED.producer_poll_seconds,
        max_send_attempts = EXCLUDED.max_send_attempts,
        updated_at = now()
    """
    execute_sql(
        engine,
        sql,
        params={
            "dataset_id": str(dataset_id).strip(),
            "run_id": str(run_id).strip(),
            "is_enabled": bool(is_enabled),
            "producer_topic": None if producer_topic is None else str(producer_topic).strip(),
            "producer_batch_size": int(producer_batch_size),
            "producer_poll_seconds": float(producer_poll_seconds),
            "max_send_attempts": int(max_send_attempts),
        },
    )


def read_simulation_state_control(
    engine,
    *,
    dataset_id: str,
    run_id: str,
    schema: str = "capstone",
    table_name: str = "simulation_state_control",
) -> dict:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    SELECT
        dataset_id,
        run_id,
        is_enabled,
        producer_topic,
        producer_batch_size,
        producer_poll_seconds,
        max_send_attempts,
        updated_at,
        created_at
    FROM "{safe_schema}"."{safe_table}"
    WHERE dataset_id = :dataset_id
      AND run_id = :run_id
    LIMIT 1
    """
    dataframe = read_sql_dataframe(
        engine,
        sql,
        params={
            "dataset_id": str(dataset_id).strip(),
            "run_id": str(run_id).strip(),
        },
    )

    if dataframe.empty:
        raise ValueError(
            f"No simulation state control row found for dataset_id={dataset_id!r}, run_id={run_id!r}"
        )

    return dataframe.iloc[0].to_dict()


# -----------------------------------------------------------------------------
# Queue status helpers
# -----------------------------------------------------------------------------

def get_send_queue_status_counts(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_send_queue",
) -> pd.DataFrame:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    SELECT
        queue_status,
        COUNT(*) AS row_count
    FROM "{safe_schema}"."{safe_table}"
    GROUP BY queue_status
    ORDER BY queue_status
    """
    return read_sql_dataframe(engine, sql)


def claim_pending_send_queue_batch(
    engine,
    *,
    batch_size: int,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_send_queue",
    producer_topic: Optional[str] = None,
    producer_worker_id: Optional[str] = None,
    claim_token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Atomically claim the next pending queue rows in deterministic send order.

    Uses FOR UPDATE SKIP LOCKED so multiple workers can safely operate without
    claiming the same rows.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    ensure_send_queue_runtime_columns(
        engine,
        schema=schema,
        table_name=table_name,
    )

    resolved_claim_token = str(claim_token).strip() if claim_token else str(uuid.uuid4())
    resolved_worker_id = (
        str(producer_worker_id).strip()
        if producer_worker_id
        else "producer_worker_default"
    )
    resolved_topic = None if producer_topic is None else str(producer_topic).strip()

    sql = f"""
    WITH next_rows AS (
        SELECT ctid
        FROM "{safe_schema}"."{safe_table}"
        WHERE queue_status = 'pending'
          AND producer_sent_at IS NULL
        ORDER BY observation_index, message_sequence_index, sensor_index
        LIMIT :batch_size
        FOR UPDATE SKIP LOCKED
    )
    UPDATE "{safe_schema}"."{safe_table}" AS q
    SET
        queue_status = 'claimed',
        claim_token = :claim_token,
        claimed_at = now(),
        producer_worker_id = :producer_worker_id,
        producer_topic = COALESCE(:producer_topic, q.producer_topic),
        producer_delivery_status = 'claimed',
        producer_delivery_error = NULL
    FROM next_rows
    WHERE q.ctid = next_rows.ctid
    RETURNING q.*
    """
    return read_sql_dataframe(
        engine,
        sql,
        params={
            "batch_size": int(batch_size),
            "claim_token": resolved_claim_token,
            "producer_worker_id": resolved_worker_id,
            "producer_topic": resolved_topic,
        },
    )


def mark_claimed_batch_sent(
    engine,
    *,
    claim_token: str,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_send_queue",
) -> pd.DataFrame:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    UPDATE "{safe_schema}"."{safe_table}"
    SET
        queue_status = 'sent',
        producer_sent_at = now(),
        producer_ack_at = now(),
        producer_delivery_status = 'sent',
        producer_delivery_error = NULL
    WHERE claim_token = :claim_token
      AND queue_status = 'claimed'
    RETURNING *
    """
    return read_sql_dataframe(
        engine,
        sql,
        params={"claim_token": str(claim_token).strip()},
    )


def mark_claimed_batch_failed(
    engine,
    *,
    claim_token: str,
    error_message: str,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_send_queue",
) -> pd.DataFrame:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    UPDATE "{safe_schema}"."{safe_table}"
    SET
        queue_status = 'failed',
        producer_delivery_status = 'failed',
        producer_delivery_error = :error_message
    WHERE claim_token = :claim_token
      AND queue_status = 'claimed'
    RETURNING *
    """
    return read_sql_dataframe(
        engine,
        sql,
        params={
            "claim_token": str(claim_token).strip(),
            "error_message": str(error_message).strip(),
        },
    )


def requeue_failed_messages(
    engine,
    *,
    max_send_attempts: int = 3,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_send_queue",
) -> pd.DataFrame:
    """
    Move failed rows back to pending for retry, but only if they have not yet
    reached max_send_attempts.

    Since the queue starts with producer_send_attempt = 1, this increments
    attempts when a failed row is requeued.
    """
    if max_send_attempts <= 0:
        raise ValueError("max_send_attempts must be > 0")

    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    UPDATE "{safe_schema}"."{safe_table}"
    SET
        queue_status = 'pending',
        claim_token = NULL,
        claimed_at = NULL,
        producer_worker_id = NULL,
        producer_delivery_status = 'retry_pending',
        producer_send_attempt = COALESCE(producer_send_attempt, 1) + 1
    WHERE queue_status = 'failed'
      AND COALESCE(producer_send_attempt, 1) < :max_send_attempts
    RETURNING *
    """
    return read_sql_dataframe(
        engine,
        sql,
        params={"max_send_attempts": int(max_send_attempts)},
    )


def release_stale_claims(
    engine,
    *,
    stale_after_minutes: int = 15,
    max_send_attempts: int = 3,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_send_queue",
) -> pd.DataFrame:
    """
    Return stale claimed rows to pending so the producer can retry them.
    """
    if stale_after_minutes <= 0:
        raise ValueError("stale_after_minutes must be > 0")
    if max_send_attempts <= 0:
        raise ValueError("max_send_attempts must be > 0")

    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    UPDATE "{safe_schema}"."{safe_table}"
    SET
        queue_status = 'pending',
        claim_token = NULL,
        claimed_at = NULL,
        producer_worker_id = NULL,
        producer_delivery_status = 'stale_claim_requeued',
        producer_send_attempt = COALESCE(producer_send_attempt, 1) + 1
    WHERE queue_status = 'claimed'
      AND claimed_at IS NOT NULL
      AND claimed_at < (now() - (:stale_after_minutes || ' minutes')::interval)
      AND COALESCE(producer_send_attempt, 1) < :max_send_attempts
    RETURNING *
    """
    return read_sql_dataframe(
        engine,
        sql,
        params={
            "stale_after_minutes": int(stale_after_minutes),
            "max_send_attempts": int(max_send_attempts),
        },
    )


__all__ = [
    "ensure_send_queue_runtime_columns",
    "ensure_simulation_state_control_table",
    "upsert_simulation_state_control",
    "read_simulation_state_control",
    "get_send_queue_status_counts",
    "claim_pending_send_queue_batch",
    "mark_claimed_batch_sent",
    "mark_claimed_batch_failed",
    "requeue_failed_messages",
    "release_stale_claims",
]