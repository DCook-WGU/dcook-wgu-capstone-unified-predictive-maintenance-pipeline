from __future__ import annotations

from typing import Sequence

import pandas as pd

from utils.postgres_util import (
    sanitize_sql_identifier,
    create_schema_if_not_exists,
    execute_sql,
    read_sql_dataframe,
)
from utils.layer_postgres_writer import write_layer_dataframe

from utils.chunk_stage_util import (
    get_table_columns,
    get_table_row_count,
    process_postgres_table_in_chunks,
)

# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _validate_source_columns(dataframe: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Send-queue source table is missing required columns: "
            + ", ".join(missing)
        )


# -----------------------------------------------------------------------------
# Queue runtime bootstrap helpers
# -----------------------------------------------------------------------------

def _ensure_send_queue_runtime_columns(
    engine,
    *,
    schema: str,
    table_name: str,
) -> None:
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


def _ensure_send_queue_indexes(
    engine,
    *,
    schema: str,
    table_name: str,
) -> None:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    index_statements = [
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_queue_status"
        ON "{safe_schema}"."{safe_table}" (queue_status);
        ''',
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_producer_order"
        ON "{safe_schema}"."{safe_table}" (observation_index, message_sequence_index, sensor_index);
        ''',
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_message_key"
        ON "{safe_schema}"."{safe_table}" (message_key);
        ''',
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_producer_sent_at"
        ON "{safe_schema}"."{safe_table}" (producer_sent_at);
        ''',
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


def _apply_send_queue_owner_and_grants(
    engine,
    *,
    schema: str,
    table_name: str,
    owner_role: str,
) -> None:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)
    safe_owner_role = sanitize_sql_identifier(owner_role)

    execute_sql(
        engine,
        f'''
        GRANT USAGE, CREATE ON SCHEMA "{safe_schema}" TO {safe_owner_role};
        '''
    )

    execute_sql(
        engine,
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}" OWNER TO {safe_owner_role};
        '''
    )

    execute_sql(
        engine,
        f'''
        GRANT SELECT, INSERT, UPDATE, DELETE
        ON TABLE "{safe_schema}"."{safe_table}"
        TO {safe_owner_role};
        '''
    )


# -----------------------------------------------------------------------------
# Stage builder
# -----------------------------------------------------------------------------

def build_sensor_messages_send_queue(
    engine,
    *,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_messages_timestamped_stage",
    target_table: str = "synthetic_sensor_messages_send_queue",
    if_exists: str = "replace",
    queue_status_default: str = "pending",
    chunk_size: int = 10000,
    queue_owner_role: str = "kafka_producer",
    apply_owner_and_grants: bool = True,
) -> str:
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_source_table = sanitize_sql_identifier(source_table)
    safe_target_table = sanitize_sql_identifier(target_table)

    source_columns = get_table_columns(
        engine,
        schema_name=safe_schema,
        table_name=safe_source_table,
    )
    preview_dataframe = pd.DataFrame(columns=source_columns)

    required_columns = [
        "dataset_id",
        "run_id",
        "asset_id",
        "generated_row_id",
        "observation_index",
        "observation_timestamp",
        "message_sequence_index",
        "batch_id",
        "row_in_batch",
        "global_cycle_id",
        "stream_state",
        "phase",
        "created_at",
        "meta_episode_id",
        "meta_primary_fault_type",
        "meta_magnitude",
        "sensor_name",
        "sensor_index",
        "sensor_value",
        "is_telemetry_event",
        "telemetry_event_type",
        "producer_send_attempt",
    ]
    _validate_source_columns(preview_dataframe, required_columns)

    source_row_count = get_table_row_count(
        engine,
        schema_name=safe_schema,
        table_name=safe_source_table,
    )
    if source_row_count == 0:
        raise ValueError(
            f"Source table '{safe_schema}.{safe_source_table}' is empty."
        )

    ordered_columns = [
        "dataset_id",
        "run_id",
        "asset_id",
        "message_key",
        "generated_row_id",
        "observation_index",
        "observation_timestamp",
        "message_sequence_index",
        "batch_id",
        "row_in_batch",
        "global_cycle_id",
        "stream_state",
        "phase",
        "created_at",
        "meta_episode_id",
        "meta_primary_fault_type",
        "meta_magnitude",
        "sensor_name",
        "sensor_index",
        "sensor_value",
        "is_telemetry_event",
        "telemetry_event_type",
        "producer_send_attempt",
        "queue_status",
        "queued_at",
        "claim_token",
        "claimed_at",
        "producer_topic",
        "producer_worker_id",
        "producer_sent_at",
        "producer_ack_at",
        "producer_delivery_status",
        "producer_delivery_error",
    ]

    has_written_first_chunk = False

    def transform_chunk_func(
        df_chunk: pd.DataFrame,
        chunk_number: int,
        start_row: int,
        end_row: int,
    ) -> pd.DataFrame:
        dataframe = df_chunk.copy()

        dataframe = dataframe.sort_values(
            by=["observation_index", "message_sequence_index", "sensor_index"],
            kind="stable",
        ).reset_index(drop=True)

        queue_built_at = pd.Timestamp.utcnow()

        dataframe["queue_status"] = str(queue_status_default).strip()
        dataframe["queued_at"] = queue_built_at

        dataframe["claim_token"] = None
        dataframe["claimed_at"] = pd.NaT
        dataframe["producer_topic"] = None
        dataframe["producer_worker_id"] = None

        dataframe["producer_sent_at"] = pd.NaT
        dataframe["producer_ack_at"] = pd.NaT
        dataframe["producer_delivery_status"] = None
        dataframe["producer_delivery_error"] = None

        dataframe["message_key"] = (
            dataframe["asset_id"].astype(str)
            + "|"
            + dataframe["observation_index"].astype(int).astype(str)
            + "|"
            + dataframe["sensor_index"].astype(int).astype(str)
        )

        remaining_columns = [
            column for column in dataframe.columns
            if column not in ordered_columns
        ]
        dataframe = dataframe[ordered_columns + remaining_columns]
        return dataframe

    def write_chunk_func(
        df_out: pd.DataFrame,
        chunk_number: int,
        start_row: int,
        end_row: int,
    ) -> None:
        nonlocal has_written_first_chunk

        write_layer_dataframe(
            engine=engine,
            dataframe=df_out,
            schema=safe_schema,
            table_name=safe_target_table,
            if_exists=if_exists if not has_written_first_chunk else "append",
            index=False,
        )

        has_written_first_chunk = True
        print(f"[send-queue] wrote chunk {chunk_number} with {len(df_out):,} rows")

    process_postgres_table_in_chunks(
        engine,
        schema_name=safe_schema,
        table_name=safe_source_table,
        select_columns=source_columns,
        order_by_sql="observation_index, message_sequence_index, sensor_index",
        transform_chunk_func=transform_chunk_func,
        write_chunk_func=write_chunk_func,
        chunk_size=chunk_size,
    )

    _ensure_send_queue_runtime_columns(
        engine,
        schema=safe_schema,
        table_name=safe_target_table,
    )

    _ensure_send_queue_indexes(
        engine,
        schema=safe_schema,
        table_name=safe_target_table,
    )

    if apply_owner_and_grants:
        _apply_send_queue_owner_and_grants(
            engine,
            schema=safe_schema,
            table_name=safe_target_table,
            owner_role=queue_owner_role,
        )

    return safe_target_table


# -----------------------------------------------------------------------------
# Validation helper
# -----------------------------------------------------------------------------

def validate_sensor_messages_send_queue(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_send_queue",
) -> pd.DataFrame:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    SELECT
        COUNT(*) AS row_count,
        COUNT(DISTINCT observation_index) AS distinct_observation_count,
        COUNT(DISTINCT sensor_name) AS distinct_sensor_name_count,
        MIN(observation_timestamp) AS min_observation_timestamp,
        MAX(observation_timestamp) AS max_observation_timestamp,
        COUNT(*) FILTER (WHERE queue_status = 'pending') AS pending_count,
        COUNT(*) FILTER (WHERE producer_sent_at IS NULL) AS unsent_count
    FROM "{safe_schema}"."{safe_table}"
    """
    return read_sql_dataframe(engine, sql)


__all__ = [
    "build_sensor_messages_send_queue",
    "validate_sensor_messages_send_queue",
]