from __future__ import annotations

import uuid
from typing import Optional, Any, cast
import math

import pandas as pd
from sqlalchemy import text

from utils.database.postgres import (
    sanitize_sql_identifier,
    create_schema_if_not_exists,
    execute_sql,
    read_sql_dataframe,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def scalar_to_int(value: object, name: str = "value") -> int:
    """Convert a required scalar result into an int and reject missing values."""
    if value is None:
        raise ValueError(f"{name} cannot be missing.")

    if value is pd.NA:
        raise ValueError(f"{name} cannot be missing.")

    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"{name} cannot be missing.")

    return int(cast(Any, value))

# -----------------------------------------------------------------------------
# Internal permission/bootstrap helpers
# -----------------------------------------------------------------------------

def _grant_schema_usage_create(
    engine,
    *,
    schema: str,
    role_name: str,
) -> None:
    """Grant schema access needed by a queue runtime role."""
    safe_schema = sanitize_sql_identifier(schema)
    safe_role = sanitize_sql_identifier(role_name)

    execute_sql(
        engine,
        f'''
        GRANT USAGE, CREATE ON SCHEMA "{safe_schema}" TO {safe_role};
        '''
    )


def _apply_table_owner_and_grants(
    engine,
    *,
    schema: str,
    table_name: str,
    owner_role: str,
) -> None:
    """Assign ownership and DML grants for a runtime queue table."""
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)
    safe_owner = sanitize_sql_identifier(owner_role)

    _grant_schema_usage_create(
        engine,
        schema=safe_schema,
        role_name=safe_owner,
    )

    execute_sql(
        engine,
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}" OWNER TO {safe_owner};
        '''
    )

    execute_sql(
        engine,
        f'''
        GRANT SELECT, INSERT, UPDATE, DELETE
        ON TABLE "{safe_schema}"."{safe_table}"
        TO {safe_owner};
        '''
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

    Important:
    This function assumes the queue table was already created with the correct
    owner/grants by the queue-stage builder or bootstrap logic.
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
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_dataset_run_claim_order"
        ON "{safe_schema}"."{safe_table}" (
            queue_status,
            dataset_id,
            run_id,
            observation_index,
            message_sequence_index,
            sensor_index
        );
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
    owner_role: Optional[str] = None,
    apply_owner_and_grants: bool = False,
) -> str:
    """
    Ensure the simulation-state control table exists.

    If apply_owner_and_grants=True, this function will also assign ownership
    and grants. That should only be used from an admin/bootstrap step.
    """
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

    if apply_owner_and_grants:
        if owner_role is None or str(owner_role).strip() == "":
            raise ValueError("owner_role must be provided when apply_owner_and_grants=True")

        _apply_table_owner_and_grants(
            engine,
            schema=safe_schema,
            table_name=safe_table,
            owner_role=owner_role,
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
    """
    Insert or update the control row that drives producer loop behavior.

    The control row stores whether a synthetic run is active, which topic to
    publish to, the producer batch size, polling delay, and retry ceiling.
    """
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
    """Read the producer control row for one dataset/run pair."""
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
    """Return row counts by queue status for producer monitoring."""
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
    dataset_id: str,
    run_id: str,
    batch_size: int,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_send_queue",
    producer_topic: Optional[str] = None,
    producer_worker_id: Optional[str] = None,
    claim_token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compatibility wrapper around claim_pending_sensor_messages_batch.

    Prefer claim_pending_sensor_messages_batch for new producer code because it
    returns both claim_token and dataframe.
    """
    _, dataframe = claim_pending_sensor_messages_batch(
        engine=engine,
        dataset_id=dataset_id,
        run_id=run_id,
        schema=schema,
        queue_table=table_name,
        batch_size=batch_size,
        producer_worker_id=producer_worker_id or "producer_worker_default",
        producer_topic=producer_topic or "pump.telemetry.synthetic",
        claim_token=claim_token,
        ensure_runtime_columns=False,
    )

    return dataframe

def claim_pending_sensor_messages_batch(
    engine,
    *,
    dataset_id: str,
    run_id: str,
    schema: str = "capstone",
    queue_table: str = "synthetic_sensor_messages_send_queue",
    batch_size: int = 26000,
    producer_worker_id: str = "producer_worker_001",
    producer_topic: str = "pump.telemetry.synthetic",
    claim_token: Optional[str] = None,
    ensure_runtime_columns: bool = False,
) -> tuple[str, pd.DataFrame]:
    """
    Atomically claim one producer batch from the send queue.

    This is the preferred queue-claim function for the synthetic sensor-message
    producer path.

    It filters by dataset_id and run_id, claims rows in deterministic send order,
    and uses FOR UPDATE SKIP LOCKED so multiple workers cannot claim the same rows.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    resolved_dataset_id = str(dataset_id).strip()
    resolved_run_id = str(run_id).strip()

    if not resolved_dataset_id:
        raise ValueError("dataset_id cannot be empty")
    if not resolved_run_id:
        raise ValueError("run_id cannot be empty")

    safe_schema = sanitize_sql_identifier(schema)
    safe_queue_table = sanitize_sql_identifier(queue_table)

    if ensure_runtime_columns:
        ensure_send_queue_runtime_columns(
            engine,
            schema=safe_schema,
            table_name=safe_queue_table,
        )

    resolved_claim_token = str(claim_token).strip() if claim_token else str(uuid.uuid4())

    # Claim rows and return the exact claimed payload in one transaction.
    sql = f"""
    WITH rows_to_claim AS (
        SELECT message_key
        FROM "{safe_schema}"."{safe_queue_table}"
        WHERE queue_status = 'pending'
          AND producer_sent_at IS NULL
          AND dataset_id = :dataset_id
          AND run_id = :run_id
        ORDER BY observation_index, message_sequence_index, sensor_index
        LIMIT :batch_size
        FOR UPDATE SKIP LOCKED
    ),
    claimed_rows AS (
        UPDATE "{safe_schema}"."{safe_queue_table}" AS queue
        SET
            queue_status = 'claimed',
            claim_token = :claim_token,
            claimed_at = CURRENT_TIMESTAMP,
            producer_worker_id = :producer_worker_id,
            producer_topic = :producer_topic,
            producer_delivery_status = 'claimed',
            producer_delivery_error = NULL
        FROM rows_to_claim
        WHERE queue.message_key = rows_to_claim.message_key
        RETURNING
            queue.dataset_id,
            queue.run_id,
            queue.asset_id,
            queue.message_key,
            queue.generated_row_id,
            queue.observation_index,
            queue.observation_timestamp,
            queue.message_sequence_index,
            queue.batch_id,
            queue.row_in_batch,
            queue.global_cycle_id,
            queue.stream_state,
            queue.phase,
            queue.created_at,
            queue.meta_episode_id,
            queue.meta_primary_fault_type,
            queue.meta_magnitude,
            queue.sensor_name,
            queue.sensor_index,
            queue.sensor_value,
            queue.is_telemetry_event,
            queue.telemetry_event_type,
            queue.producer_send_attempt,
            queue.queue_status,
            queue.queued_at,
            queue.claim_token,
            queue.claimed_at,
            queue.producer_topic,
            queue.producer_worker_id,
            queue.producer_delivery_status,
            queue.producer_delivery_error
    )
    SELECT *
    FROM claimed_rows
    ORDER BY observation_index, message_sequence_index, sensor_index
    """

    with engine.begin() as connection:
        dataframe = pd.read_sql(
            text(sql),
            connection,
            params={
                "dataset_id": resolved_dataset_id,
                "run_id": resolved_run_id,
                "batch_size": int(batch_size),
                "claim_token": resolved_claim_token,
                "producer_worker_id": str(producer_worker_id).strip(),
                "producer_topic": str(producer_topic).strip(),
            },
        )

    return resolved_claim_token, dataframe



def mark_claimed_batch_sent(
    engine,
    *,
    claim_token: str,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_send_queue",
) -> pd.DataFrame:
    """Mark all rows for a claim token as delivered to Kafka."""
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
    """Mark all rows for a claim token as failed and store the delivery error."""
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

def mark_claimed_batch_sent_count(
    engine,
    *,
    claim_token: str,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_send_queue",
) -> int:
    """Mark a claim as sent and return the number of updated queue rows."""
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    WITH updated AS (
        UPDATE "{safe_schema}"."{safe_table}"
        SET
            queue_status = 'sent',
            producer_sent_at = now(),
            producer_ack_at = now(),
            producer_delivery_status = 'sent',
            producer_delivery_error = NULL
        WHERE claim_token = :claim_token
          AND queue_status = 'claimed'
        RETURNING 1
    )
    SELECT COUNT(*) AS updated_count
    FROM updated
    """

    dataframe = read_sql_dataframe(
        engine,
        sql,
        params={"claim_token": str(claim_token).strip()},
    )

    return scalar_to_int(
        dataframe.at[0, "updated_count"],
        "updated_count",
    )

def mark_claimed_batch_failed_count(
    engine,
    *,
    claim_token: str,
    error_message: str,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_send_queue",
) -> int:
    """Mark a claim as failed and return the number of updated queue rows."""
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    WITH updated AS (
        UPDATE "{safe_schema}"."{safe_table}"
        SET
            queue_status = 'failed',
            producer_delivery_status = 'failed',
            producer_delivery_error = :error_message
        WHERE claim_token = :claim_token
          AND queue_status = 'claimed'
        RETURNING 1
    )
    SELECT COUNT(*) AS updated_count
    FROM updated
    """

    dataframe = read_sql_dataframe(
        engine,
        sql,
        params={
            "claim_token": str(claim_token).strip(),
            "error_message": str(error_message).strip(),
        },
    )

    return scalar_to_int(
        dataframe.at[0, "updated_count"],
        "updated_count",
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

    # Reset claim fields while preserving prior delivery error text for review.
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

    # Stale claims are treated like retries because the original worker may be gone.
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
    "claim_pending_sensor_messages_batch",
    "mark_claimed_batch_sent",
    "mark_claimed_batch_failed",
    "mark_claimed_batch_sent_count",
    "mark_claimed_batch_failed_count",
    "requeue_failed_messages",
    "release_stale_claims",
]
