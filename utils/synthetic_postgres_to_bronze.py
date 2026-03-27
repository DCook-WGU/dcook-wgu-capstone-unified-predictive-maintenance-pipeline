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

from utils.layer_postgres_writer import write_layer_dataframe
from utils.postgres_util import (
    get_engine_from_env,
    read_sql_dataframe,
    sanitize_sql_identifier,
    table_exists,
)


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
    target_total_rows: Optional[int] = None,
    trim_mode: str = "head",
    keep_lineage_columns: bool = False,
    include_anomaly_flag: bool = False,
    keep_other_columns: bool = False,
) -> pd.DataFrame:
    """
    Full in-memory direct wide-table -> bronze handoff preparation.
    """
    validate_synthetic_stream_dataframe(raw_dataframe)

    dataframe = raw_dataframe.copy()
    dataframe = sort_synthetic_stream_dataframe(dataframe)
    dataframe = add_unified_row_id(dataframe)
    dataframe = add_unified_episode_id(dataframe)
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
    "summarize_bronze_handoff_dataframe",
    "write_bronze_handoff_to_postgres",
    "table_exists",
]