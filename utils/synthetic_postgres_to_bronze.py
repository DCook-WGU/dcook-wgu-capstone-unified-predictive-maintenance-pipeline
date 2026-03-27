"""
Utilities for converting the wide synthetic stream table in Postgres into a
single bronze-ready table that looks like the original pump sensor dataset.

Main goals:
1. Read one or more synthetic batches from Postgres.
2. Sort them into one stable sequence across batches.
3. Create unified row numbering and unified episode numbering.
4. Add a fresh time index and timestamp series.
5. Derive the original-style machine status label.
6. Cut the dataframe down to the columns needed for Bronze handoff.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import text

import json

from utils.layer_postgres_writer import write_layer_dataframe
from utils.postgres_util import (
    get_engine_from_env,
    read_sql_dataframe,
    sanitize_sql_identifier,
    table_exists,
)

from pandas.tseries.frequencies import to_offset


DEFAULT_STREAM_STATE_TO_MACHINE_STATUS = {
    "normal": "NORMAL",
    "buildup": "NORMAL",
    "abnormal": "BROKEN",
    "recovery": "RECOVERING",
}

DEFAULT_PHASE_TO_MACHINE_STATUS = {
    "normal": "NORMAL",
    "normal_before": "NORMAL",
    "buildup": "NORMAL",
    "failure": "BROKEN",
    "abnormal": "BROKEN",
    "recovery": "RECOVERING",
    "normal_after": "NORMAL",
}


def build_engine_from_project_env(*, driver: str = "psycopg2", echo: bool = False) -> Engine:
    """
    Return a SQLAlchemy engine using the same project utility pattern as the
    rest of the capstone.
    """
    return get_engine_from_env(driver=driver, echo=echo)


def get_table_columns(engine: Engine, *, schema: str, table_name: str) -> list[str]:
    """
    Return table columns in ordinal order.
    """
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

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
        params={"schema_name": safe_schema, "table_name": safe_table},
    )
    return dataframe["column_name"].astype(str).tolist()


def get_sensor_columns(columns: Iterable[str]) -> list[str]:
    """
    Return sensor columns in numeric order:
    sensor_00, sensor_01, ..., sensor_51.
    """
    sensor_columns = []

    for column in columns:
        column = str(column)
        if column.startswith("sensor_"):
            suffix = column.replace("sensor_", "", 1)
            if suffix.isdigit():
                sensor_columns.append((int(suffix), column))

    sensor_columns = sorted(sensor_columns, key=lambda item: item[0])
    return [column for _, column in sensor_columns]

def _int_or_default(value: Any, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return int(default)
    return int(value)

def read_synthetic_stream_dataframe(
    engine: Engine,
    *,
    schema: str,
    table_name: str,
    batch_ids: Optional[Iterable[int]] = None,
    selected_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Read the wide synthetic stream table from Postgres.

    Notes:
    - batch_ids is optional
    - selected_columns is optional
    """
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    if selected_columns:
        cleaned_columns = [f'"{sanitize_sql_identifier(column)}"' for column in selected_columns]
        select_sql = ", ".join(cleaned_columns)
    else:
        select_sql = "*"

    sql = f'SELECT {select_sql} FROM "{safe_schema}"."{safe_table}"'

    if batch_ids is not None:
        clean_batch_ids = [int(value) for value in batch_ids]
        if clean_batch_ids:
            batch_sql = ", ".join(str(value) for value in clean_batch_ids)
            sql += f" WHERE batch_id IN ({batch_sql})"

    return read_sql_dataframe(engine, sql)

def get_distinct_batch_ids(
    engine: Engine,
    *,
    schema: str,
    table_name: str,
    batch_column: str = "batch_id",
) -> list[int]:
    """
    Return distinct batch ids from a table.
    """
    if not table_exists(engine, schema=schema, table_name=table_name):
        return []

    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)
    safe_batch = sanitize_sql_identifier(batch_column)

    sql = f'''
    SELECT DISTINCT "{safe_batch}" AS batch_id
    FROM "{safe_schema}"."{safe_table}"
    WHERE "{safe_batch}" IS NOT NULL
    ORDER BY 1
    '''

    dataframe = read_sql_dataframe(engine, sql)

    if dataframe.empty:
        return []

    return [int(value) for value in dataframe["batch_id"].tolist()]

def get_unloaded_source_batch_ids(
    engine: Engine,
    *,
    source_schema: str,
    source_table: str,
    target_schema: str,
    target_table: str,
    requested_batch_ids: Optional[Iterable[int]] = None,
    batch_column: str = "batch_id",
) -> dict[str, Any]:
    """
    Compare source and target tables and return which source batches have not
    yet been loaded into the target append table.
    """
    source_batch_ids = set(
        get_distinct_batch_ids(
            engine,
            schema=source_schema,
            table_name=source_table,
            batch_column=batch_column,
        )
    )

    if requested_batch_ids is None:
        candidate_batch_ids = sorted(source_batch_ids)
    else:
        requested_batch_ids = {int(value) for value in requested_batch_ids}
        candidate_batch_ids = sorted(source_batch_ids.intersection(requested_batch_ids))

    target_batch_ids = set(
        get_distinct_batch_ids(
            engine,
            schema=target_schema,
            table_name=target_table,
            batch_column=batch_column,
        )
    )

    already_loaded_batch_ids = sorted(set(candidate_batch_ids).intersection(target_batch_ids))
    new_batch_ids = sorted(set(candidate_batch_ids) - target_batch_ids)

    return {
        "candidate_batch_ids": candidate_batch_ids,
        "already_loaded_batch_ids": already_loaded_batch_ids,
        "new_batch_ids": new_batch_ids,
        "candidate_batch_count": len(candidate_batch_ids),
        "already_loaded_batch_count": len(already_loaded_batch_ids),
        "new_batch_count": len(new_batch_ids),
    }

def get_unloaded_source_batch_ids(
    engine: Engine,
    *,
    source_schema: str,
    source_table: str,
    target_schema: str,
    target_table: str,
    requested_batch_ids: Optional[Iterable[int]] = None,
    batch_column: str = "batch_id",
) -> dict[str, Any]:
    """
    Compare source and target tables and return which source batches have not
    yet been loaded into the target append table.
    """
    source_batch_ids = set(
        get_distinct_batch_ids(
            engine,
            schema=source_schema,
            table_name=source_table,
            batch_column=batch_column,
        )
    )

    if requested_batch_ids is None:
        candidate_batch_ids = sorted(source_batch_ids)
    else:
        requested_batch_ids = {int(value) for value in requested_batch_ids}
        candidate_batch_ids = sorted(source_batch_ids.intersection(requested_batch_ids))

    target_batch_ids = set(
        get_distinct_batch_ids(
            engine,
            schema=target_schema,
            table_name=target_table,
            batch_column=batch_column,
        )
    )

    already_loaded_batch_ids = sorted(set(candidate_batch_ids).intersection(target_batch_ids))
    new_batch_ids = sorted(set(candidate_batch_ids) - target_batch_ids)

    return {
        "candidate_batch_ids": candidate_batch_ids,
        "already_loaded_batch_ids": already_loaded_batch_ids,
        "new_batch_ids": new_batch_ids,
        "candidate_batch_count": len(candidate_batch_ids),
        "already_loaded_batch_count": len(already_loaded_batch_ids),
        "new_batch_count": len(new_batch_ids),
    }

def ensure_handoff_control_table(
    engine: Engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_bronze_handoff_control",
) -> None:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f'''
    CREATE TABLE IF NOT EXISTS "{safe_schema}"."{safe_table}" (
        dataset_name TEXT NOT NULL,
        target_schema TEXT NOT NULL,
        target_table TEXT NOT NULL,
        last_loaded_batch_id BIGINT,
        loaded_batch_count BIGINT NOT NULL DEFAULT 0,
        next_unified_row_id BIGINT NOT NULL DEFAULT 1,
        next_unified_episode_id BIGINT NOT NULL DEFAULT 0,
        next_observation_time_index BIGINT NOT NULL DEFAULT 0,
        next_timestamp TIMESTAMP NOT NULL,
        last_append_row_count BIGINT NOT NULL DEFAULT 0,
        last_loaded_batch_ids_json JSONB,
        last_truth_hash TEXT,
        last_process_run_id TEXT,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        notes TEXT,
        created_at_utc TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at_utc TIMESTAMP NOT NULL DEFAULT NOW(),
        PRIMARY KEY (dataset_name, target_schema, target_table)
    )
    '''

    with engine.begin() as conn:
        conn.exec_driver_sql(sql)

def ensure_handoff_control_table(
    engine: Engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_bronze_handoff_control",
) -> None:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f'''
    CREATE TABLE IF NOT EXISTS "{safe_schema}"."{safe_table}" (
        dataset_name TEXT NOT NULL,
        target_schema TEXT NOT NULL,
        target_table TEXT NOT NULL,
        last_loaded_batch_id BIGINT,
        loaded_batch_count BIGINT NOT NULL DEFAULT 0,
        next_unified_row_id BIGINT NOT NULL DEFAULT 1,
        next_unified_episode_id BIGINT NOT NULL DEFAULT 0,
        next_observation_time_index BIGINT NOT NULL DEFAULT 0,
        next_timestamp TIMESTAMP NOT NULL,
        last_append_row_count BIGINT NOT NULL DEFAULT 0,
        last_loaded_batch_ids_json JSONB,
        last_truth_hash TEXT,
        last_process_run_id TEXT,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        notes TEXT,
        created_at_utc TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at_utc TIMESTAMP NOT NULL DEFAULT NOW(),
        PRIMARY KEY (dataset_name, target_schema, target_table)
    )
    '''

    with engine.begin() as conn:
        conn.exec_driver_sql(sql)

def get_effective_handoff_offsets(
    engine: Engine,
    *,
    dataset_name: str,
    target_schema: str,
    target_table: str,
    initial_start_timestamp: str = "2018-04-01 00:00:00",
    frequency: str = "1min",
    control_schema: str = "capstone",
    control_table: str = "synthetic_bronze_handoff_control",
) -> dict[str, Any]:
    control_record = get_handoff_control_record(
        engine,
        dataset_name=dataset_name,
        target_schema=target_schema,
        target_table=target_table,
        control_schema=control_schema,
        control_table=control_table,
    )

    if control_record is not None:
        return {
            "offset_source": "control_table",
            "target_exists": True,
            "existing_row_count": None,
            "next_unified_row_id": int(control_record["next_unified_row_id"]),
            "next_unified_episode_id": int(control_record["next_unified_episode_id"]),
            "next_observation_time_index": int(control_record["next_observation_time_index"]),
            "next_timestamp": pd.to_datetime(control_record["next_timestamp"]),
            "last_loaded_batch_id": control_record.get("last_loaded_batch_id"),
            "loaded_batch_count": int(control_record.get("loaded_batch_count", 0)),
        }

    append_offsets = get_handoff_append_offsets(
        engine,
        schema=target_schema,
        table_name=target_table,
        initial_start_timestamp=initial_start_timestamp,
        frequency=frequency,
    )

    append_offsets["offset_source"] = "append_table_scan"
    return append_offsets



def upsert_handoff_control_record(
    engine: Engine,
    *,
    dataset_name: str,
    target_schema: str,
    target_table: str,
    last_loaded_batch_id: Optional[int],
    loaded_batch_count: int,
    next_unified_row_id: int,
    next_unified_episode_id: int,
    next_observation_time_index: int,
    next_timestamp: Any,
    last_append_row_count: int,
    last_loaded_batch_ids: Optional[list[int]] = None,
    last_truth_hash: Optional[str] = None,
    last_process_run_id: Optional[str] = None,
    notes: Optional[str] = None,
    control_schema: str = "capstone",
    control_table: str = "synthetic_bronze_handoff_control",
) -> None:
    ensure_handoff_control_table(
        engine,
        schema=control_schema,
        table_name=control_table,
    )

    safe_schema = sanitize_sql_identifier(control_schema)
    safe_table = sanitize_sql_identifier(control_table)

    sql = f'''
    INSERT INTO "{safe_schema}"."{safe_table}" (
        dataset_name,
        target_schema,
        target_table,
        last_loaded_batch_id,
        loaded_batch_count,
        next_unified_row_id,
        next_unified_episode_id,
        next_observation_time_index,
        next_timestamp,
        last_append_row_count,
        last_loaded_batch_ids_json,
        last_truth_hash,
        last_process_run_id,
        notes,
        is_active,
        updated_at_utc
    )
    VALUES (
        :dataset_name,
        :target_schema,
        :target_table,
        :last_loaded_batch_id,
        :loaded_batch_count,
        :next_unified_row_id,
        :next_unified_episode_id,
        :next_observation_time_index,
        :next_timestamp,
        :last_append_row_count,
        CAST(:last_loaded_batch_ids_json AS JSONB),
        :last_truth_hash,
        :last_process_run_id,
        :notes,
        TRUE,
        NOW()
    )
    ON CONFLICT (dataset_name, target_schema, target_table)
    DO UPDATE SET
        last_loaded_batch_id = EXCLUDED.last_loaded_batch_id,
        loaded_batch_count = EXCLUDED.loaded_batch_count,
        next_unified_row_id = EXCLUDED.next_unified_row_id,
        next_unified_episode_id = EXCLUDED.next_unified_episode_id,
        next_observation_time_index = EXCLUDED.next_observation_time_index,
        next_timestamp = EXCLUDED.next_timestamp,
        last_append_row_count = EXCLUDED.last_append_row_count,
        last_loaded_batch_ids_json = EXCLUDED.last_loaded_batch_ids_json,
        last_truth_hash = EXCLUDED.last_truth_hash,
        last_process_run_id = EXCLUDED.last_process_run_id,
        notes = EXCLUDED.notes,
        is_active = TRUE,
        updated_at_utc = NOW()
    '''

    params = {
        "dataset_name": dataset_name,
        "target_schema": target_schema,
        "target_table": target_table,
        "last_loaded_batch_id": last_loaded_batch_id,
        "loaded_batch_count": int(loaded_batch_count),
        "next_unified_row_id": int(next_unified_row_id),
        "next_unified_episode_id": int(next_unified_episode_id),
        "next_observation_time_index": int(next_observation_time_index),
        "next_timestamp": pd.to_datetime(next_timestamp).to_pydatetime(),
        "last_append_row_count": int(last_append_row_count),
        "last_loaded_batch_ids_json": json.dumps(last_loaded_batch_ids or []),
        "last_truth_hash": last_truth_hash,
        "last_process_run_id": last_process_run_id,
        "notes": notes,
    }

    with engine.begin() as conn:
        conn.execute(text(sql), params)




def ensure_handoff_control_table(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_bronze_handoff_control",
) -> None:
    create_sql = f'''
    CREATE TABLE IF NOT EXISTS "{schema}"."{table_name}" (
        dataset_name TEXT NOT NULL,
        target_schema TEXT NOT NULL,
        target_table TEXT NOT NULL,
        last_loaded_batch_id BIGINT,
        loaded_batch_count BIGINT NOT NULL DEFAULT 0,
        next_unified_row_id BIGINT NOT NULL DEFAULT 1,
        next_unified_episode_id BIGINT NOT NULL DEFAULT 0,
        next_observation_time_index BIGINT NOT NULL DEFAULT 0,
        next_timestamp TIMESTAMP NOT NULL,
        last_append_row_count BIGINT NOT NULL DEFAULT 0,
        last_loaded_batch_ids_json JSONB,
        last_truth_hash TEXT,
        last_process_run_id TEXT,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        notes TEXT,
        created_at_utc TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at_utc TIMESTAMP NOT NULL DEFAULT NOW(),
        PRIMARY KEY (dataset_name, target_schema, target_table)
    )
    '''

    with engine.begin() as conn:
        conn.execute(text(create_sql))

def get_handoff_control_record(
    engine: Engine,
    *,
    dataset_name: str,
    target_schema: str,
    target_table: str,
    control_schema: str = "capstone",
    control_table: str = "synthetic_bronze_handoff_control",
) -> Optional[dict[str, Any]]:
    ensure_handoff_control_table(
        engine,
        schema=control_schema,
        table_name=control_table,
    )

    safe_schema = sanitize_sql_identifier(control_schema)
    safe_table = sanitize_sql_identifier(control_table)

    sql = f'''
    SELECT *
    FROM "{safe_schema}"."{safe_table}"
    WHERE dataset_name = :dataset_name
      AND target_schema = :target_schema
      AND target_table = :target_table
      AND is_active = TRUE
    '''

    dataframe = read_sql_dataframe(
        engine,
        sql,
        params={
            "dataset_name": dataset_name,
            "target_schema": target_schema,
            "target_table": target_table,
        },
    )

    if dataframe.empty:
        return None

    record = dataframe.iloc[0].to_dict()

    if record.get("next_timestamp") is not None:
        record["next_timestamp"] = pd.to_datetime(record["next_timestamp"])

    return record



def get_handoff_append_offsets(
    engine: Engine,
    *,
    schema: str,
    table_name: str,
    initial_start_timestamp: str = "2018-04-01 00:00:00",
    frequency: str = "1min",
    unified_row_id_column: str = "unified_row_id",
    unified_episode_id_column: str = "meta__episode_id_unified",
    time_index_column: str = "observation_time_index",
    timestamp_column: str = "timestamp",
) -> dict[str, Any]:
    """
    Read the current append target table and return the next starting offsets
    for ids and time fields.
    """
    if not table_exists(engine, schema=schema, table_name=table_name):
        return {
            "target_exists": False,
            "existing_row_count": 0,
            "next_unified_row_id": 1,
            "next_unified_episode_id": 0,
            "next_observation_time_index": 0,
            "next_timestamp": pd.Timestamp(initial_start_timestamp),
        }

    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)
    safe_row_id = sanitize_sql_identifier(unified_row_id_column)
    safe_episode_id = sanitize_sql_identifier(unified_episode_id_column)
    safe_time_index = sanitize_sql_identifier(time_index_column)
    safe_timestamp = sanitize_sql_identifier(timestamp_column)

    sql = f'''
    SELECT
        COUNT(*) AS existing_row_count,
        MAX("{safe_row_id}") AS max_unified_row_id,
        MAX("{safe_episode_id}") AS max_unified_episode_id,
        MAX("{safe_time_index}") AS max_observation_time_index,
        MAX("{safe_timestamp}") AS max_timestamp
    FROM "{safe_schema}"."{safe_table}"
    '''

    dataframe = read_sql_dataframe(engine, sql)
    record = dataframe.iloc[0].to_dict()

    max_timestamp = record.get("max_timestamp")
    if max_timestamp is None or pd.isna(max_timestamp):
        next_timestamp = pd.Timestamp(initial_start_timestamp)
    else:
        next_timestamp = pd.to_datetime(max_timestamp) + to_offset(frequency)

    max_unified_row_id = _int_or_default(record.get("max_unified_row_id"), default=0)
    max_unified_episode_id = _int_or_default(record.get("max_unified_episode_id"), default=-1)
    max_observation_time_index = _int_or_default(record.get("max_observation_time_index"), default=-1)
    existing_row_count = _int_or_default(record.get("existing_row_count"), default=0)

    return {
        "target_exists": True,
        "existing_row_count": existing_row_count,
        "next_unified_row_id": max_unified_row_id + 1,
        "next_unified_episode_id": max_unified_episode_id + 1,
        "next_observation_time_index": max_observation_time_index + 1,
        "next_timestamp": next_timestamp,
    }

def validate_synthetic_stream_dataframe(
    dataframe: pd.DataFrame,
    *,
    expected_min_sensor_columns: int = 1,
    require_batch_row_order_columns: bool = True,
    require_label_source: bool = True,
) -> dict[str, Any]:
    """
    Basic schema validation for the wide synthetic stream table.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    sensor_columns = get_sensor_columns(dataframe.columns)

    has_stream_state = "stream_state" in dataframe.columns
    has_phase = "phase" in dataframe.columns
    has_batch_id = "batch_id" in dataframe.columns
    has_row_in_batch = "row_in_batch" in dataframe.columns
    has_global_cycle_id = "global_cycle_id" in dataframe.columns

    missing_requirements = []

    if len(sensor_columns) < int(expected_min_sensor_columns):
        missing_requirements.append("sensor columns")

    if require_batch_row_order_columns and not (has_global_cycle_id or (has_batch_id and has_row_in_batch)):
        missing_requirements.append("ordering columns (global_cycle_id or batch_id + row_in_batch)")

    if require_label_source and not (has_stream_state or has_phase):
        missing_requirements.append("stream_state or phase")

    report = {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "sensor_column_count": int(len(sensor_columns)),
        "has_stream_state": bool(has_stream_state),
        "has_phase": bool(has_phase),
        "has_batch_id": bool(has_batch_id),
        "has_row_in_batch": bool(has_row_in_batch),
        "has_global_cycle_id": bool(has_global_cycle_id),
        "missing_requirements": missing_requirements,
    }

    if missing_requirements:
        raise ValueError(f"Synthetic stream dataframe failed validation: {missing_requirements}")

    return report


def choose_sort_columns(dataframe: pd.DataFrame) -> list[str]:
    """
    Choose the best available sort columns for one unified sequence.

    Preference:
    1. global_cycle_id
    2. batch_id + row_in_batch
    3. fallback ordered columns
    """
    columns = set(dataframe.columns)

    if "global_cycle_id" in columns:
        return ["global_cycle_id"]

    if "batch_id" in columns and "row_in_batch" in columns:
        return ["batch_id", "row_in_batch"]

    preferred = ["batch_id", "cycle_id", "global_row_id", "row_in_batch", "created_at"]
    return [column for column in preferred if column in columns]


def sort_synthetic_stream_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Stable sort for one unified ordered stream.
    """
    out = dataframe.copy()
    sort_columns = choose_sort_columns(out)

    if sort_columns:
        out = out.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    return out


def add_unified_row_id(
    dataframe: pd.DataFrame,
    *,
    output_column: str = "unified_row_id",
    start_at: int = 1,
) -> pd.DataFrame:
    """
    Add a single unified row number across the selected rows.
    """
    out = dataframe.copy()
    out[output_column] = range(int(start_at), int(start_at) + len(out))
    return out


def add_unified_episode_id(
    dataframe: pd.DataFrame,
    *,
    source_episode_column: str = "meta__episode_id",
    batch_column: str = "batch_id",
    output_column: str = "meta__episode_id_unified",
    start_at: int = 0,
) -> pd.DataFrame:
    """
    Create a batch-safe unified episode id.

    This matters because each batch can restart meta__episode_id at 0.
    """
    out = dataframe.copy()

    if source_episode_column not in out.columns:
        return out

    key_columns = [source_episode_column]
    if batch_column in out.columns:
        key_columns = [batch_column, source_episode_column]

    distinct_keys = out[key_columns].drop_duplicates().reset_index(drop=True)
    distinct_keys[output_column] = range(int(start_at), int(start_at) + len(distinct_keys))

    out = out.merge(distinct_keys, how="left", on=key_columns, sort=False)
    return out


def derive_machine_status(
    dataframe: pd.DataFrame,
    *,
    stream_state_column: str = "stream_state",
    phase_column: str = "phase",
    output_column: str = "machine_status",
    stream_state_map: Optional[dict[str, str]] = None,
    phase_map: Optional[dict[str, str]] = None,
    default_value: str = "NORMAL",
) -> pd.DataFrame:
    """
    Convert synthetic labels to the original pump-style machine_status values.

    Resolution order:
    1. stream_state when it maps cleanly
    2. phase when stream_state did not resolve
    3. default_value
    """
    out = dataframe.copy()

    stream_state_map = stream_state_map or DEFAULT_STREAM_STATE_TO_MACHINE_STATUS
    phase_map = phase_map or DEFAULT_PHASE_TO_MACHINE_STATUS

    machine_status = pd.Series([default_value] * len(out), index=out.index, dtype="object")
    resolved_mask = pd.Series(False, index=out.index)

    if stream_state_column in out.columns:
        mapped_stream_state = (
            out[stream_state_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(stream_state_map)
        )
        valid_stream_state = mapped_stream_state.notna()
        machine_status.loc[valid_stream_state] = mapped_stream_state.loc[valid_stream_state]
        resolved_mask = resolved_mask | valid_stream_state

    if phase_column in out.columns:
        mapped_phase = (
            out[phase_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(phase_map)
        )
        valid_phase = mapped_phase.notna() & (~resolved_mask)
        machine_status.loc[valid_phase] = mapped_phase.loc[valid_phase]

    out[output_column] = machine_status.fillna(default_value)
    return out


def add_synthetic_anomaly_flag(
    dataframe: pd.DataFrame,
    *,
    machine_status_column: str = "machine_status",
    output_column: str = "anomaly_flag__synthetic",
    normal_statuses: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Optional helper flag for quick sanity checks or downstream testing.

    NORMAL => 0
    anything else => 1
    """
    out = dataframe.copy()

    if machine_status_column not in out.columns:
        raise ValueError(f"Missing machine status column: {machine_status_column}")

    normal_statuses = normal_statuses or ["NORMAL"]
    normal_statuses = {str(value).strip().upper() for value in normal_statuses}

    out[output_column] = (
        ~out[machine_status_column].astype(str).str.upper().isin(normal_statuses)
    ).astype("int8")

    return out


def trim_unified_dataframe(
    dataframe: pd.DataFrame,
    *,
    target_total_rows: Optional[int] = None,
    trim_mode: str = "head",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Optional row trimming after the unified table is built.

    This happens BEFORE observation_time_index/timestamp are assigned so the
    final output stays contiguous.
    """
    out = dataframe.copy()

    if target_total_rows is None:
        return out

    target_total_rows = int(target_total_rows)

    if target_total_rows <= 0:
        raise ValueError("target_total_rows must be greater than 0.")

    if len(out) <= target_total_rows:
        return out

    trim_mode = str(trim_mode).strip().lower()

    if trim_mode == "head":
        return out.head(target_total_rows).copy()

    if trim_mode == "tail":
        return out.tail(target_total_rows).copy()

    if trim_mode == "random":
        sampled = out.sample(n=target_total_rows, random_state=random_state)
        return sampled.sort_index().copy()

    raise ValueError("trim_mode must be one of: head, tail, random")


def add_observation_time_fields(
    dataframe: pd.DataFrame,
    *,
    start_timestamp: str = "2018-04-01 00:00:00",
    frequency: str = "1min",
    time_index_column: str = "observation_time_index",
    timestamp_column: str = "timestamp",
    start_index_at: int = 0,
) -> pd.DataFrame:
    """
    Add final contiguous time index + timestamp values to the final row set.
    """
    out = dataframe.copy().reset_index(drop=True)

    if len(out) == 0:
        out[time_index_column] = pd.Series(dtype="int64")
        out[timestamp_column] = pd.Series(dtype="datetime64[ns]")
        return out

    out[time_index_column] = range(int(start_index_at), int(start_index_at) + len(out))
    out[timestamp_column] = pd.date_range(
        start=start_timestamp,
        periods=len(out),
        freq=frequency,
    )
    return out


def select_bronze_handoff_columns(
    dataframe: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    machine_status_column: str = "machine_status",
    include_anomaly_flag: bool = False,
    anomaly_flag_column: str = "anomaly_flag__synthetic",
    keep_lineage_columns: bool = False,
    extra_lineage_columns: Optional[Sequence[str]] = None,
    keep_other_columns: bool = False,
) -> pd.DataFrame:
    """
    Select the final handoff columns.

    Default behavior:
    timestamp + sensor_* + machine_status

    Optional:
    - anomaly_flag__synthetic
    - lineage columns
    - any other remaining columns
    """
    out = dataframe.copy()
    sensor_columns = get_sensor_columns(out.columns)

    selected_columns: list[str] = []

    if timestamp_column in out.columns:
        selected_columns.append(timestamp_column)

    selected_columns.extend(sensor_columns)

    if machine_status_column in out.columns:
        selected_columns.append(machine_status_column)

    if include_anomaly_flag and anomaly_flag_column in out.columns:
        selected_columns.append(anomaly_flag_column)

    lineage_columns = [
        "unified_row_id",
        "observation_time_index",
        "batch_id",
        "row_in_batch",
        "global_cycle_id",
        "cycle_id",
        "global_row_id",
        "created_at",
        "stream_state",
        "phase",
        "meta__episode_id",
        "meta__episode_id_unified",
        "meta__primary_sensor",
        "meta__primary_fault_type",
        "meta__magnitude",
    ]

    if extra_lineage_columns:
        lineage_columns.extend([str(column) for column in extra_lineage_columns])

    if keep_lineage_columns:
        for column in lineage_columns:
            if column in out.columns and column not in selected_columns:
                selected_columns.append(column)

    if keep_other_columns:
        for column in out.columns:
            if column not in selected_columns:
                selected_columns.append(column)

    return out.loc[:, selected_columns].copy()


def prepare_synthetic_postgres_for_bronze_handoff(
    raw_dataframe: pd.DataFrame,
    *,
    start_timestamp: str = "2018-04-01 00:00:00",
    frequency: str = "1min",
    start_unified_row_id: int = 1,
    start_unified_episode_id: int = 0,
    start_observation_time_index: int = 0,
    target_total_rows: Optional[int] = None,
    trim_mode: str = "head",
    keep_lineage_columns: bool = False,
    include_anomaly_flag: bool = False,
    keep_other_columns: bool = False,
) -> pd.DataFrame:
    """
    Full in-memory direct wide-table -> bronze handoff preparation.

    Supports both:
    - fresh rebuilds
    - append-aware continuation when offsets are provided
    """
    validate_synthetic_stream_dataframe(raw_dataframe)

    dataframe = raw_dataframe.copy()
    dataframe = sort_synthetic_stream_dataframe(dataframe)
    dataframe = add_unified_row_id(
        dataframe,
        start_at=start_unified_row_id,
    )
    dataframe = add_unified_episode_id(
        dataframe,
        start_at=start_unified_episode_id,
    )
    dataframe = derive_machine_status(dataframe)

    if include_anomaly_flag:
        dataframe = add_synthetic_anomaly_flag(dataframe)

    dataframe = trim_unified_dataframe(
        dataframe,
        target_total_rows=target_total_rows,
        trim_mode=trim_mode,
    )

    dataframe = add_observation_time_fields(
        dataframe,
        start_timestamp=start_timestamp,
        frequency=frequency,
        start_index_at=start_observation_time_index,
    )

    dataframe = select_bronze_handoff_columns(
        dataframe,
        include_anomaly_flag=include_anomaly_flag,
        keep_lineage_columns=keep_lineage_columns,
        keep_other_columns=keep_other_columns,
    )

    return dataframe


def build_bronze_handoff_from_postgres(
    engine: Engine,
    *,
    source_schema: str,
    source_table: str,
    batch_ids: Optional[Iterable[int]] = None,
    start_timestamp: str = "2018-04-01 00:00:00",
    frequency: str = "1min",
    target_total_rows: Optional[int] = None,
    trim_mode: str = "head",
    keep_lineage_columns: bool = False,
    include_anomaly_flag: bool = False,
    keep_other_columns: bool = False,
) -> pd.DataFrame:
    """
    Read the raw wide stream from Postgres and return the final bronze handoff dataframe.
    """
    raw_dataframe = read_synthetic_stream_dataframe(
        engine,
        schema=source_schema,
        table_name=source_table,
        batch_ids=batch_ids,
    )

    return prepare_synthetic_postgres_for_bronze_handoff(
        raw_dataframe,
        start_timestamp=start_timestamp,
        frequency=frequency,
        target_total_rows=target_total_rows,
        trim_mode=trim_mode,
        keep_lineage_columns=keep_lineage_columns,
        include_anomaly_flag=include_anomaly_flag,
        keep_other_columns=keep_other_columns,
    )

def build_append_aware_bronze_handoff_from_postgres(
    engine: Engine,
    *,
    source_schema: str,
    source_table: str,
    target_schema: str,
    target_table: str,
    batch_ids: Optional[Iterable[int]] = None,
    initial_start_timestamp: str = "2018-04-01 00:00:00",
    frequency: str = "1min",
    target_total_rows: Optional[int] = None,
    trim_mode: str = "head",
    keep_lineage_columns: bool = True,
    include_anomaly_flag: bool = False,
    keep_other_columns: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Append-aware builder.

    Behavior:
    1. Detect source batches not yet loaded into the target append table
    2. Read target offsets from the append table
    3. Build only the new rows with continued ids and time fields
    """
    batch_plan = get_unloaded_source_batch_ids(
        engine,
        source_schema=source_schema,
        source_table=source_table,
        target_schema=target_schema,
        target_table=target_table,
        requested_batch_ids=batch_ids,
    )

    offsets = get_effective_handoff_offsets(
        engine,
        dataset_name=source_table.replace("synthetic_", "").replace("_stream", ""),
        target_schema=target_schema,
        target_table=target_table,
        initial_start_timestamp=initial_start_timestamp,
        frequency=frequency,
    )

    new_batch_ids = batch_plan["new_batch_ids"]

    if not new_batch_ids:
        empty_dataframe = pd.DataFrame()
        append_plan = {
            "load_mode": "append",
            **batch_plan,
            **offsets,
            "appended_row_count": 0,
        }
        return empty_dataframe, append_plan

    raw_dataframe = read_synthetic_stream_dataframe(
        engine,
        schema=source_schema,
        table_name=source_table,
        batch_ids=new_batch_ids,
    )

    append_dataframe = prepare_synthetic_postgres_for_bronze_handoff(
        raw_dataframe,
        start_timestamp=str(offsets["next_timestamp"]),
        frequency=frequency,
        start_unified_row_id=offsets["next_unified_row_id"],
        start_unified_episode_id=offsets["next_unified_episode_id"],
        start_observation_time_index=offsets["next_observation_time_index"],
        target_total_rows=target_total_rows,
        trim_mode=trim_mode,
        keep_lineage_columns=keep_lineage_columns,
        include_anomaly_flag=include_anomaly_flag,
        keep_other_columns=keep_other_columns,
    )

    append_plan = {
        "load_mode": "append",
        **batch_plan,
        **offsets,
        "appended_row_count": int(len(append_dataframe)),
    }

    return append_dataframe, append_plan

def summarize_bronze_handoff_dataframe(dataframe: pd.DataFrame) -> dict[str, Any]:
    """
    Small summary payload for notebook logging / truth payloads.
    """
    summary = {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "sensor_column_count": int(len(get_sensor_columns(dataframe.columns))),
        "timestamp_min": None,
        "timestamp_max": None,
        "machine_status_counts": {},
        "batch_count": None,
        "episode_count_unified": None,
    }

    if len(dataframe) > 0 and "timestamp" in dataframe.columns:
        summary["timestamp_min"] = str(pd.to_datetime(dataframe["timestamp"]).min())
        summary["timestamp_max"] = str(pd.to_datetime(dataframe["timestamp"]).max())

    if "machine_status" in dataframe.columns:
        summary["machine_status_counts"] = {
            str(key): int(value)
            for key, value in dataframe["machine_status"].value_counts(dropna=False).to_dict().items()
        }

    if "batch_id" in dataframe.columns:
        summary["batch_count"] = int(dataframe["batch_id"].nunique())

    if "meta__episode_id_unified" in dataframe.columns:
        summary["episode_count_unified"] = int(dataframe["meta__episode_id_unified"].nunique())

    return summary


def write_bronze_handoff_to_postgres(
    engine: Engine,
    dataframe: pd.DataFrame,
    *,
    schema: str,
    table_name: str,
    if_exists: str = "replace",
    logger: Optional[Any] = None,
) -> str:
    """
    Write the final bronze handoff dataframe using the same generic Postgres
    writer pattern used elsewhere in the project.
    """
    return write_layer_dataframe(
        engine=engine,
        dataframe=dataframe,
        schema=schema,
        table_name=table_name,
        if_exists=if_exists,
        index=False,
        logger=logger,
        allow_empty=False,
    )


__all__ = [
    "DEFAULT_STREAM_STATE_TO_MACHINE_STATUS",
    "DEFAULT_PHASE_TO_MACHINE_STATUS",
    "build_engine_from_project_env",
    "get_table_columns",
    "get_sensor_columns",
    "read_synthetic_stream_dataframe",
    "get_distinct_batch_ids",
    "get_unloaded_source_batch_ids",
    "get_handoff_control_record"
    "get_handoff_control_record",
    "get_effective_handoff_offsets",
    "upsert_handoff_control_record",
    "ensure_handoff_control_table",
    "get_handoff_append_offsets",
    "validate_synthetic_stream_dataframe",
    "choose_sort_columns",
    "sort_synthetic_stream_dataframe",
    "add_unified_row_id",
    "add_unified_episode_id",
    "derive_machine_status",
    "add_synthetic_anomaly_flag",
    "trim_unified_dataframe",
    "add_observation_time_fields",
    "select_bronze_handoff_columns",
    "prepare_synthetic_postgres_for_bronze_handoff",
    "build_bronze_handoff_from_postgres",
    "build_append_aware_bronze_handoff_from_postgres",
    "summarize_bronze_handoff_dataframe",
    "write_bronze_handoff_to_postgres",
    "table_exists",
]