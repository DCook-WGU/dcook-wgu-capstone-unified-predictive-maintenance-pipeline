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
) -> str:
    """
    Build the final Kafka-ready send queue from the timestamped sensor messages stage.

    Behavior:
    - reads fully prepared long-format timestamped messages
    - preserves producer send order
    - adds queue tracking fields
    - does not publish to Kafka
    - leaves producer_sent_at null until the producer actually sends
    """
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_source_table = sanitize_sql_identifier(source_table)
    safe_target_table = sanitize_sql_identifier(target_table)

    source_sql = f"""
    SELECT *
    FROM "{safe_schema}"."{safe_source_table}"
    ORDER BY observation_index, message_sequence_index, sensor_index
    """
    dataframe = read_sql_dataframe(engine, source_sql)

    if dataframe.empty:
        raise ValueError(
            f"Source table '{safe_schema}.{safe_source_table}' is empty."
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
    _validate_source_columns(dataframe, required_columns)

    # -------------------------------------------------------------------------
    # Preserve deterministic producer-facing order
    # -------------------------------------------------------------------------
    dataframe = dataframe.sort_values(
        by=["observation_index", "message_sequence_index", "sensor_index"],
        kind="stable",
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Add queue metadata
    # -------------------------------------------------------------------------
    queue_built_at = pd.Timestamp.utcnow()

    dataframe["queue_status"] = str(queue_status_default).strip()
    dataframe["queued_at"] = queue_built_at
    dataframe["producer_sent_at"] = pd.NaT
    dataframe["producer_delivery_status"] = None
    dataframe["producer_delivery_error"] = None

    # Optional convenience key for producer use
    dataframe["message_key"] = (
        dataframe["asset_id"].astype(str)
        + "|"
        + dataframe["observation_index"].astype(int).astype(str)
        + "|"
        + dataframe["sensor_index"].astype(int).astype(str)
    )

    # -------------------------------------------------------------------------
    # Final column order
    # -------------------------------------------------------------------------
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
        "producer_sent_at",
        "producer_delivery_status",
        "producer_delivery_error",
    ]

    remaining_columns = [
        column for column in dataframe.columns
        if column not in ordered_columns
    ]
    dataframe = dataframe[ordered_columns + remaining_columns]

    # -------------------------------------------------------------------------
    # Write send queue table
    # -------------------------------------------------------------------------
    table_name = write_layer_dataframe(
        engine=engine,
        dataframe=dataframe,
        schema=safe_schema,
        table_name=safe_target_table,
        if_exists=if_exists,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Helpful indexes for producer consumption
    # -------------------------------------------------------------------------
    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_queue_status"
        ON "{safe_schema}"."{safe_target_table}" (queue_status);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_producer_order"
        ON "{safe_schema}"."{safe_target_table}" (observation_index, message_sequence_index, sensor_index);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_message_key"
        ON "{safe_schema}"."{safe_target_table}" (message_key);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_producer_sent_at"
        ON "{safe_schema}"."{safe_target_table}" (producer_sent_at);
        '''
    )

    return table_name


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