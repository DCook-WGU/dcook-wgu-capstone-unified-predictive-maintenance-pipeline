from __future__ import annotations

from typing import Optional

import pandas as pd

from utils.postgres_util import (
    sanitize_sql_identifier,
    create_schema_if_not_exists,
    execute_sql,
    read_sql_dataframe,
)
from utils.layer_postgres_writer import write_layer_dataframe


# -----------------------------------------------------------------------------
# Timing config helpers
# -----------------------------------------------------------------------------

def ensure_simulation_timing_config_table(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "simulation_timing_config",
) -> str:
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    CREATE TABLE IF NOT EXISTS "{safe_schema}"."{safe_table}" (
        config_id BIGSERIAL PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        simulation_start_datetime TIMESTAMPTZ NOT NULL,
        sampling_interval_seconds DOUBLE PRECISION NOT NULL,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
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

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_active_created"
        ON "{safe_schema}"."{safe_table}" (is_active, created_at DESC);
        '''
    )

    return safe_table


def insert_simulation_timing_config(
    engine,
    *,
    dataset_id: str,
    run_id: str,
    simulation_start_datetime: str,
    sampling_interval_seconds: float,
    schema: str = "capstone",
    table_name: str = "simulation_timing_config",
    set_active: bool = True,
    deactivate_existing_for_run: bool = True,
) -> None:
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = ensure_simulation_timing_config_table(
        engine,
        schema=schema,
        table_name=table_name,
    )

    if deactivate_existing_for_run and set_active:
        deactivate_sql = f"""
        UPDATE "{safe_schema}"."{safe_table}"
        SET is_active = FALSE
        WHERE dataset_id = :dataset_id
          AND run_id = :run_id
        """
        execute_sql(
            engine,
            deactivate_sql,
            params={
                "dataset_id": str(dataset_id).strip(),
                "run_id": str(run_id).strip(),
            },
        )

    insert_sql = f"""
    INSERT INTO "{safe_schema}"."{safe_table}" (
        dataset_id,
        run_id,
        simulation_start_datetime,
        sampling_interval_seconds,
        is_active
    )
    VALUES (
        :dataset_id,
        :run_id,
        :simulation_start_datetime,
        :sampling_interval_seconds,
        :is_active
    )
    """
    execute_sql(
        engine,
        insert_sql,
        params={
            "dataset_id": str(dataset_id).strip(),
            "run_id": str(run_id).strip(),
            "simulation_start_datetime": str(simulation_start_datetime).strip(),
            "sampling_interval_seconds": float(sampling_interval_seconds),
            "is_active": bool(set_active),
        },
    )


def load_simulation_timing_config(
    engine,
    *,
    dataset_id: str,
    run_id: str,
    schema: str = "capstone",
    table_name: str = "simulation_timing_config",
) -> dict:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    SELECT
        dataset_id,
        run_id,
        simulation_start_datetime,
        sampling_interval_seconds,
        is_active,
        created_at
    FROM "{safe_schema}"."{safe_table}"
    WHERE dataset_id = :dataset_id
      AND run_id = :run_id
    ORDER BY is_active DESC, created_at DESC
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
            f"No timing config found for dataset_id={dataset_id!r}, run_id={run_id!r} "
            f"in {safe_schema}.{safe_table}"
        )

    record = dataframe.iloc[0].to_dict()
    return record


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _validate_source_columns(dataframe: pd.DataFrame) -> None:
    required_columns = [
        "dataset_id",
        "run_id",
        "asset_id",
        "generated_row_id",
        "observation_index",
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

    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Timestamp-stage source table is missing required columns: "
            + ", ".join(missing)
        )


def _resolve_dataset_and_run(
    dataframe: pd.DataFrame,
    dataset_id: Optional[str],
    run_id: Optional[str],
) -> tuple[str, str]:
    unique_pairs = dataframe[["dataset_id", "run_id"]].drop_duplicates().reset_index(drop=True)

    if dataset_id is None and run_id is None:
        if len(unique_pairs) != 1:
            raise ValueError(
                "Source table contains multiple dataset_id/run_id pairs. "
                "Pass dataset_id and run_id explicitly."
            )
        return (
            str(unique_pairs.loc[0, "dataset_id"]).strip(),
            str(unique_pairs.loc[0, "run_id"]).strip(),
        )

    if dataset_id is None or run_id is None:
        raise ValueError("dataset_id and run_id must both be provided together, or both omitted.")

    return str(dataset_id).strip(), str(run_id).strip()


# -----------------------------------------------------------------------------
# Stage builder
# -----------------------------------------------------------------------------

def build_sensor_messages_timestamped_stage(
    engine,
    *,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_messages_stage",
    target_table: str = "synthetic_sensor_messages_timestamped_stage",
    timing_config_table: str = "simulation_timing_config",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    if_exists: str = "replace",
) -> str:
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

    _validate_source_columns(dataframe)

    resolved_dataset_id, resolved_run_id = _resolve_dataset_and_run(
        dataframe=dataframe,
        dataset_id=dataset_id,
        run_id=run_id,
    )

    timing_config = load_simulation_timing_config(
        engine,
        dataset_id=resolved_dataset_id,
        run_id=resolved_run_id,
        schema=schema,
        table_name=timing_config_table,
    )

    simulation_start_datetime = pd.Timestamp(timing_config["simulation_start_datetime"])
    sampling_interval_seconds = float(timing_config["sampling_interval_seconds"])

    if sampling_interval_seconds <= 0:
        raise ValueError("sampling_interval_seconds must be greater than 0.")

    dataframe = dataframe.sort_values(
        by=["observation_index", "message_sequence_index", "sensor_index"],
        kind="stable",
    ).reset_index(drop=True)

    dataframe["observation_timestamp"] = (
        simulation_start_datetime
        + pd.to_timedelta(
            (dataframe["observation_index"].astype(int) - 1) * sampling_interval_seconds,
            unit="s",
        )
    )

    ordered_columns = [
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

    remaining_columns = [
        column for column in dataframe.columns
        if column not in ordered_columns
    ]
    dataframe = dataframe[ordered_columns + remaining_columns]

    table_name = write_layer_dataframe(
        engine=engine,
        dataframe=dataframe,
        schema=safe_schema,
        table_name=safe_target_table,
        if_exists=if_exists,
        index=False,
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_observation_index"
        ON "{safe_schema}"."{safe_target_table}" (observation_index);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_observation_timestamp"
        ON "{safe_schema}"."{safe_target_table}" (observation_timestamp);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_obs_msg_seq"
        ON "{safe_schema}"."{safe_target_table}" (observation_index, message_sequence_index);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_obs_sensor"
        ON "{safe_schema}"."{safe_target_table}" (observation_index, sensor_index);
        '''
    )

    return table_name


# -----------------------------------------------------------------------------
# Validation helper
# -----------------------------------------------------------------------------

def validate_sensor_messages_timestamped_stage(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_timestamped_stage",
) -> pd.DataFrame:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    SELECT
        COUNT(*) AS row_count,
        COUNT(DISTINCT observation_index) AS distinct_observation_count,
        MIN(observation_timestamp) AS min_observation_timestamp,
        MAX(observation_timestamp) AS max_observation_timestamp,
        COUNT(DISTINCT observation_timestamp) AS distinct_observation_timestamp_count
    FROM "{safe_schema}"."{safe_table}"
    """
    return read_sql_dataframe(engine, sql)


__all__ = [
    "ensure_simulation_timing_config_table",
    "insert_simulation_timing_config",
    "load_simulation_timing_config",
    "build_sensor_messages_timestamped_stage",
    "validate_sensor_messages_timestamped_stage",
]