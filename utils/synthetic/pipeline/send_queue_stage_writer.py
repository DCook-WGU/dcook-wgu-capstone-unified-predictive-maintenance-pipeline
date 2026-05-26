from __future__ import annotations

from typing import Sequence
from time import perf_counter

import pandas as pd

from utils.database.postgres import (
    sanitize_sql_identifier,
    create_schema_if_not_exists,
    execute_sql,
    read_sql_dataframe,
    table_exists,
)

from utils.database.layer_postgres import write_layer_dataframe

from utils.database.chunk_stage_util import (
    get_table_columns,
    get_table_row_count,
    process_postgres_table_in_chunks,
)

# -----------------------------------------------------------------------------
# Time Logging helpers
# -----------------------------------------------------------------------------

def log_step_timing(step_name: str, start_time: float) -> float:
    elapsed_seconds = perf_counter() - start_time
    print(f"[timing] {step_name}: {elapsed_seconds:,.2f} seconds")
    return perf_counter()

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
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_dataset_run_obs"
        ON "{safe_schema}"."{safe_table}" (dataset_id, run_id, observation_index);
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
# Stage builder - Old Method
# -----------------------------------------------------------------------------

def build_sensor_messages_send_queue(
    engine,
    *,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_messages_stage",
    target_table: str = "synthetic_sensor_messages_send_queue",
    if_exists: str = "replace",
    queue_status_default: str = "pending",
    chunk_size: int = 10000,
    queue_owner_role: str = "kafka_producer",
    apply_owner_and_grants: bool = False,
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
            dataframe["dataset_id"].astype(str)
            + "|"
            + dataframe["run_id"].astype(str)
            + "|"
            + dataframe["asset_id"].astype(str)
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
# Stage builder - SQL Native
# -----------------------------------------------------------------------------

def build_sensor_messages_send_queue_sql_native(
    engine,
    *,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_messages_stage",
    target_table: str = "synthetic_sensor_messages_send_queue",
    if_exists: str = "replace",
    queue_status_default: str = "pending",
    queue_owner_role: str = "kafka_producer",
    apply_owner_and_grants: bool = False,
    enable_timing_logging: bool = True,
) -> str:
    """
    Build the synthetic sensor message send queue directly inside Postgres.

    This avoids reading 11M+ rows into pandas just to add queue metadata.
    The output table keeps the same queue/runtime fields expected by the
    producer queue manager and Kafka producer stages.
    """
    timer = perf_counter()

    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_source_table = sanitize_sql_identifier(source_table)
    safe_target_table = sanitize_sql_identifier(target_table)

    source_columns = get_table_columns(
        engine,
        schema_name=safe_schema,
        table_name=safe_source_table,
    )

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

    missing_columns = [
        column
        for column in required_columns
        if column not in source_columns
    ]

    if missing_columns:
        raise ValueError(
            "Send-queue source table is missing required columns: "
            + ", ".join(missing_columns)
        )

    source_row_count = get_table_row_count(
        engine,
        schema_name=safe_schema,
        table_name=safe_source_table,
    )

    if source_row_count == 0:
        raise ValueError(
            f"Source table '{safe_schema}.{safe_source_table}' is empty."
        )

    if enable_timing_logging:
        timer = log_step_timing("source validation complete", timer)

    write_mode = str(if_exists).strip().lower()
    target_exists = table_exists(
        engine,
        schema=safe_schema,
        table_name=safe_target_table,
    )

    create_select_sql = f'''
    SELECT
        dataset_id,
        run_id,
        asset_id,
        CONCAT_WS(
            '|',
            dataset_id::TEXT,
            run_id::TEXT,
            asset_id::TEXT,
            observation_index::TEXT,
            sensor_index::TEXT
        ) AS message_key,
        generated_row_id,
        observation_index,
        observation_timestamp,
        message_sequence_index,
        batch_id,
        row_in_batch,
        global_cycle_id,
        stream_state,
        phase,
        created_at,
        meta_episode_id,
        meta_primary_fault_type,
        meta_magnitude,
        sensor_name,
        sensor_index,
        sensor_value,
        is_telemetry_event,
        telemetry_event_type,
        producer_send_attempt,
        CAST(:queue_status_default AS TEXT) AS queue_status,
        CURRENT_TIMESTAMP AS queued_at,
        NULL::TEXT AS claim_token,
        NULL::TIMESTAMPTZ AS claimed_at,
        NULL::TEXT AS producer_topic,
        NULL::TEXT AS producer_worker_id,
        NULL::TIMESTAMPTZ AS producer_sent_at,
        NULL::TIMESTAMPTZ AS producer_ack_at,
        NULL::TEXT AS producer_delivery_status,
        NULL::TEXT AS producer_delivery_error
    FROM "{safe_schema}"."{safe_source_table}"
    ORDER BY observation_index, message_sequence_index, sensor_index
    '''

    params = {
        "queue_status_default": str(queue_status_default).strip(),
    }

    if write_mode == "replace":
        execute_sql(
            engine,
            f'DROP TABLE IF EXISTS "{safe_schema}"."{safe_target_table}" CASCADE;',
        )

        execute_sql(
            engine,
            f'''
            CREATE TABLE "{safe_schema}"."{safe_target_table}" AS
            {create_select_sql}
            ''',
            params=params,
        )

    elif write_mode == "fail":
        if target_exists:
            raise ValueError(
                f"Target table already exists: {safe_schema}.{safe_target_table}"
            )

        execute_sql(
            engine,
            f'''
            CREATE TABLE "{safe_schema}"."{safe_target_table}" AS
            {create_select_sql}
            ''',
            params=params,
        )

    elif write_mode == "append":
        if not target_exists:
            execute_sql(
                engine,
                f'''
                CREATE TABLE "{safe_schema}"."{safe_target_table}" AS
                {create_select_sql}
                ''',
                params=params,
            )
        else:
            execute_sql(
                engine,
                f'''
                INSERT INTO "{safe_schema}"."{safe_target_table}"
                {create_select_sql}
                ''',
                params=params,
            )

    else:
        raise ValueError("if_exists must be one of: 'replace', 'append', 'fail'.")

    if enable_timing_logging:
        timer = log_step_timing("send queue table build complete", timer)

    _ensure_send_queue_runtime_columns(
        engine,
        schema=safe_schema,
        table_name=safe_target_table,
    )

    if enable_timing_logging:
        timer = log_step_timing("runtime columns verified", timer)

    _ensure_send_queue_indexes(
        engine,
        schema=safe_schema,
        table_name=safe_target_table,
    )

    execute_sql(
        engine,
        f'ANALYZE "{safe_schema}"."{safe_target_table}";',
    )

    if enable_timing_logging:
        timer = log_step_timing("indexes/analyze complete", timer)

    if apply_owner_and_grants:
        _apply_send_queue_owner_and_grants(
            engine,
            schema=safe_schema,
            table_name=safe_target_table,
            owner_role=queue_owner_role,
        )

        if enable_timing_logging:
            timer = log_step_timing("owner/grants applied", timer)

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
    "build_sensor_messages_send_queue_sql_native",
    "validate_sensor_messages_send_queue",
]