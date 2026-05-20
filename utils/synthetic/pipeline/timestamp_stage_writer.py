from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from utils.database.chunk_stage_util import resolve_dataset_run_from_table
from utils.database.postgres import (
    create_schema_if_not_exists,
    execute_sql,
    read_sql_dataframe,
    sanitize_sql_identifier,
    table_exists,
)


"""SQL-native builder for the synthetic observations timestamped stage.

This module keeps the existing notebook-facing API, but computes the timestamped
stage entirely in Postgres rather than chunking the full table through pandas.
"""


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
        ''',
    )
    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_active_created"
        ON "{safe_schema}"."{safe_table}" (is_active, created_at DESC);
        ''',
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

    return dataframe.iloc[0].to_dict()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _get_table_columns(engine, *, schema: str, table_name: str) -> list[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = :schema_name
      AND table_name = :table_name
    ORDER BY ordinal_position
    """
    dataframe = read_sql_dataframe(
        engine,
        sql,
        params={
            "schema_name": str(schema).strip(),
            "table_name": str(table_name).strip(),
        },
    )
    return dataframe["column_name"].astype(str).tolist()



def _validate_source_columns(columns: Sequence[str]) -> None:
    required_columns = [
        "dataset_id",
        "run_id",
        "asset_id",
        "generated_row_id",
        "observation_index",
        "batch_id",
        "row_in_batch",
        "global_cycle_id",
        "stream_state",
        "phase",
        "created_at",
        "meta_episode_id",
        "meta_primary_fault_type",
        "meta_magnitude",
        "is_telemetry_event",
        "telemetry_event_type",
        "producer_send_attempt",
    ]
    missing = [column for column in required_columns if column not in columns]
    if missing:
        raise ValueError(
            "Observation timestamp-stage source table is missing required columns: "
            + ", ".join(missing)
        )



def _build_select_sql(*, safe_schema: str, safe_source_table: str, remaining_source_columns: Sequence[str]) -> str:
    remaining_sql = ",\n        ".join([f'"{column}"' for column in remaining_source_columns])
    remaining_clause = f",\n        {remaining_sql}" if remaining_sql else ""

    return f"""
    SELECT
        dataset_id,
        run_id,
        asset_id,
        generated_row_id,
        observation_index,
        CAST(:simulation_start_datetime AS TIMESTAMPTZ)
            + (((observation_index - 1)::DOUBLE PRECISION * :sampling_interval_seconds)
            * INTERVAL '1 second') AS observation_timestamp,
        batch_id,
        row_in_batch,
        global_cycle_id,
        stream_state,
        phase,
        created_at,
        meta_episode_id,
        meta_primary_fault_type,
        meta_magnitude,
        is_telemetry_event,
        telemetry_event_type,
        producer_send_attempt{remaining_clause}
    FROM "{safe_schema}"."{safe_source_table}"
    WHERE dataset_id = :dataset_id
      AND run_id = :run_id
    ORDER BY observation_index
    """



def _write_stage_sql_native(
    engine,
    *,
    schema: str,
    target_table: str,
    select_sql: str,
    params: dict,
    if_exists: str,
) -> str:
    safe_schema = sanitize_sql_identifier(schema)
    safe_target_table = sanitize_sql_identifier(target_table)
    write_mode = str(if_exists).strip().lower()
    target_exists = table_exists(engine, schema=safe_schema, table_name=safe_target_table)

    if write_mode == "replace":
        execute_sql(engine, f'DROP TABLE IF EXISTS "{safe_schema}"."{safe_target_table}"')
        execute_sql(
            engine,
            f'CREATE TABLE "{safe_schema}"."{safe_target_table}" AS\n{select_sql}',
            params=params,
        )
        return safe_target_table

    if write_mode == "fail":
        if target_exists:
            raise ValueError(f"Target table already exists: {safe_schema}.{safe_target_table}")
        execute_sql(
            engine,
            f'CREATE TABLE "{safe_schema}"."{safe_target_table}" AS\n{select_sql}',
            params=params,
        )
        return safe_target_table

    if write_mode == "append":
        if not target_exists:
            execute_sql(
                engine,
                f'CREATE TABLE "{safe_schema}"."{safe_target_table}" AS\n{select_sql}',
                params=params,
            )
            return safe_target_table

        execute_sql(
            engine,
            f'INSERT INTO "{safe_schema}"."{safe_target_table}"\n{select_sql}',
            params=params,
        )
        return safe_target_table

    raise ValueError("if_exists must be one of: 'replace', 'append', 'fail'.")


# -----------------------------------------------------------------------------
# Stage builder
# -----------------------------------------------------------------------------


def build_observations_timestamped_stage(
    engine,
    *,
    schema: str = "capstone",
    source_table: str = "synthetic_observations_premelt_stage",
    target_table: str = "synthetic_observations_timestamped_stage",
    timing_config_table: str = "simulation_timing_config",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    if_exists: str = "replace",
    chunk_size: int = 10000,
) -> str:
    """Build the timestamped stage directly inside Postgres.

    `chunk_size` is kept only for backward-compatible notebook calls. It is not
    used in the SQL-native implementation.
    """
    _ = chunk_size

    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_source_table = sanitize_sql_identifier(source_table)
    safe_target_table = sanitize_sql_identifier(target_table)

    source_columns = _get_table_columns(engine, schema=safe_schema, table_name=safe_source_table)
    if not source_columns:
        raise ValueError(f"Source table does not exist or has no columns: {safe_schema}.{safe_source_table}")
    _validate_source_columns(source_columns)

    source_row_count_sql = (
        f'SELECT COUNT(*) AS row_count FROM "{safe_schema}"."{safe_source_table}"'
    )
    source_row_count = int(read_sql_dataframe(engine, source_row_count_sql).loc[0, "row_count"])
    if source_row_count == 0:
        raise ValueError(f"Source table '{safe_schema}.{safe_source_table}' is empty.")

    resolved_dataset_id, resolved_run_id = resolve_dataset_run_from_table(
        engine,
        schema_name=safe_schema,
        table_name=safe_source_table,
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
    sampling_interval_seconds = float(timing_config["sampling_interval_seconds"])
    if sampling_interval_seconds <= 0:
        raise ValueError("sampling_interval_seconds must be greater than 0.")

    ordered_source_front_columns = [
        "dataset_id",
        "run_id",
        "asset_id",
        "generated_row_id",
        "observation_index",
        "batch_id",
        "row_in_batch",
        "global_cycle_id",
        "stream_state",
        "phase",
        "created_at",
        "meta_episode_id",
        "meta_primary_fault_type",
        "meta_magnitude",
        "is_telemetry_event",
        "telemetry_event_type",
        "producer_send_attempt",
    ]
    remaining_source_columns = [
        column for column in source_columns if column not in ordered_source_front_columns
    ]

    select_sql = _build_select_sql(
        safe_schema=safe_schema,
        safe_source_table=safe_source_table,
        remaining_source_columns=remaining_source_columns,
    )

    table_name = _write_stage_sql_native(
        engine,
        schema=safe_schema,
        target_table=safe_target_table,
        select_sql=select_sql,
        params={
            "dataset_id": str(resolved_dataset_id).strip(),
            "run_id": str(resolved_run_id).strip(),
            "simulation_start_datetime": str(timing_config["simulation_start_datetime"]),
            "sampling_interval_seconds": sampling_interval_seconds,
        },
        if_exists=if_exists,
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_observation_index"
        ON "{safe_schema}"."{safe_target_table}" (observation_index);
        ''',
    )
    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_observation_timestamp"
        ON "{safe_schema}"."{safe_target_table}" (observation_timestamp);
        ''',
    )
    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_dataset_run_obs"
        ON "{safe_schema}"."{safe_target_table}" (dataset_id, run_id, observation_index);
        ''',
    )

    return safe_target_table


# -----------------------------------------------------------------------------
# Validation helper
# -----------------------------------------------------------------------------


def validate_observations_timestamped_stage(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_observations_timestamped_stage",
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


# -----------------------------------------------------------------------------
# Backward-compatible alias names
# -----------------------------------------------------------------------------


def build_sensor_messages_timestamped_stage(*args, **kwargs) -> str:
    return build_observations_timestamped_stage(*args, **kwargs)



def validate_sensor_messages_timestamped_stage(*args, **kwargs) -> pd.DataFrame:
    return validate_observations_timestamped_stage(*args, **kwargs)


__all__ = [
    "ensure_simulation_timing_config_table",
    "insert_simulation_timing_config",
    "load_simulation_timing_config",
    "build_observations_timestamped_stage",
    "validate_observations_timestamped_stage",
    "build_sensor_messages_timestamped_stage",
    "validate_sensor_messages_timestamped_stage",
]
