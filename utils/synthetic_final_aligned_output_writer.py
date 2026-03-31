
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd

from utils.postgres_util import (
    sanitize_sql_identifier,
    create_schema_if_not_exists,
    execute_sql,
    read_sql_dataframe,
)
from utils.layer_postgres_writer import write_layer_dataframe


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def _build_sensor_columns(n_sensors: int = 52) -> list[str]:
    return [f"sensor_{i:02d}" for i in range(n_sensors)]


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
        ORDER BY ordinal_position
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

    existing_columns = _get_existing_columns(engine, schema=safe_schema, table=safe_table)
    desired_columns = [sanitize_sql_identifier(column) for column in dataframe.columns]

    missing_columns = [column for column in desired_columns if column not in existing_columns]
    if not missing_columns:
        return

    working = dataframe.copy()
    working.columns = [sanitize_sql_identifier(column) for column in working.columns]

    for column in missing_columns:
        column_type = _infer_alter_column_type(working[column])
        execute_sql(
            engine,
            f'ALTER TABLE "{safe_schema}"."{safe_table}" ADD COLUMN "{column}" {column_type};',
        )


def _resolve_dataset_run_from_table(
    engine,
    *,
    schema: str,
    table_name: str,
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> tuple[str, str]:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    where_clauses = []
    params: dict[str, str] = {}

    if dataset_id is not None:
        where_clauses.append("dataset_id = :dataset_id")
        params["dataset_id"] = str(dataset_id).strip()

    if run_id is not None:
        where_clauses.append("run_id = :run_id")
        params["run_id"] = str(run_id).strip()

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    dataframe = read_sql_dataframe(
        engine,
        f"""
        SELECT DISTINCT dataset_id, run_id
        FROM "{safe_schema}"."{safe_table}"
        {where_sql}
        ORDER BY dataset_id, run_id
        """,
        params=params,
    )

    if dataframe.empty:
        raise ValueError(
            f"No dataset_id/run_id rows found in {safe_schema}.{safe_table} "
            f"for dataset_id={dataset_id!r}, run_id={run_id!r}."
        )

    if dataset_id is not None and run_id is not None:
        return str(dataset_id).strip(), str(run_id).strip()

    if len(dataframe) > 1:
        pairs = dataframe.astype(str).agg(" / ".join, axis=1).tolist()
        raise ValueError(
            "Multiple dataset_id/run_id pairs were found. "
            "Pass dataset_id and run_id explicitly. "
            f"Found: {pairs}"
        )

    return (
        str(dataframe.iloc[0]["dataset_id"]).strip(),
        str(dataframe.iloc[0]["run_id"]).strip(),
    )


def _resolve_first_existing_column(
    columns: Iterable[str],
    priority_columns: Sequence[str],
) -> Optional[str]:
    column_set = set(columns)
    for column in priority_columns:
        if column in column_set:
            return column
    return None


def _normalize_machine_status_value(
    value,
    *,
    status_mapping: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    if pd.isna(value):
        return None

    normalized = str(value).strip()
    if normalized == "":
        return None

    lookup = normalized.lower()

    mapping = {
        "normal": "NORMAL",
        "broken": "BROKEN",
        "abnormal": "BROKEN",
        "failure": "BROKEN",
        "failed": "BROKEN",
        "fault": "BROKEN",
        "recovering": "RECOVERING",
        "recovery": "RECOVERING",
    }

    if status_mapping:
        mapping.update(
            {
                str(source_value).strip().lower(): str(target_value).strip()
                for source_value, target_value in status_mapping.items()
            }
        )

    return mapping.get(lookup, normalized.upper())


def _validate_rebuilt_columns(
    dataframe: pd.DataFrame,
    *,
    n_sensors: int,
    timestamp_source_priority: Sequence[str],
    status_source_priority: Sequence[str],
) -> tuple[str, str]:
    sensor_columns = _build_sensor_columns(n_sensors)

    required_base_columns = [
        "dataset_id",
        "run_id",
        "asset_id",
    ] + sensor_columns

    missing_base_columns = [
        column for column in required_base_columns if column not in dataframe.columns
    ]
    if missing_base_columns:
        raise ValueError(
            "Rebuilt table is missing required columns: "
            + ", ".join(missing_base_columns)
        )

    timestamp_source_column = _resolve_first_existing_column(
        dataframe.columns,
        timestamp_source_priority,
    )
    if timestamp_source_column is None:
        raise ValueError(
            "Rebuilt table is missing a timestamp source column. "
            f"Checked: {list(timestamp_source_priority)}"
        )

    status_source_column = _resolve_first_existing_column(
        dataframe.columns,
        status_source_priority,
    )
    if status_source_column is None:
        raise ValueError(
            "Rebuilt table is missing a machine status source column. "
            f"Checked: {list(status_source_priority)}"
        )

    return timestamp_source_column, status_source_column


# -----------------------------------------------------------------------------
# Dataframe builder
# -----------------------------------------------------------------------------

def build_final_aligned_synthetic_output_dataframe(
    rebuilt_dataframe: pd.DataFrame,
    *,
    n_sensors: int = 52,
    timestamp_source_priority: Sequence[str] = (
        "observation_timestamp",
        "timestamp",
        "created_at",
    ),
    status_source_priority: Sequence[str] = (
        "machine_status",
        "stream_state",
        "phase",
    ),
    status_mapping: Optional[Mapping[str, str]] = None,
    timestamp_output_column: str = "timestamp",
    machine_status_output_column: str = "machine_status",
    sort_output: bool = True,
) -> pd.DataFrame:
    if not isinstance(rebuilt_dataframe, pd.DataFrame):
        raise TypeError("rebuilt_dataframe must be a pandas DataFrame.")

    if rebuilt_dataframe.empty:
        output_columns = [
            "dataset_id",
            "run_id",
            "asset_id",
            timestamp_output_column,
            *_build_sensor_columns(n_sensors),
            machine_status_output_column,
        ]
        return pd.DataFrame(columns=output_columns)

    timestamp_source_column, status_source_column = _validate_rebuilt_columns(
        rebuilt_dataframe,
        n_sensors=n_sensors,
        timestamp_source_priority=timestamp_source_priority,
        status_source_priority=status_source_priority,
    )

    sensor_columns = _build_sensor_columns(n_sensors)

    working = rebuilt_dataframe.copy()

    working[timestamp_output_column] = pd.to_datetime(
        working[timestamp_source_column],
        errors="coerce",
        utc=True,
    )

    working[machine_status_output_column] = working[status_source_column].map(
        lambda value: _normalize_machine_status_value(
            value,
            status_mapping=status_mapping,
        )
    )

    final_columns = [
        "dataset_id",
        "run_id",
        "asset_id",
        timestamp_output_column,
        *sensor_columns,
        machine_status_output_column,
    ]

    output_dataframe = working.loc[:, final_columns].copy()

    if sort_output:
        sort_columns = [timestamp_output_column, "dataset_id", "run_id", "asset_id"]
        output_dataframe = output_dataframe.sort_values(
            by=sort_columns,
            kind="stable",
        ).reset_index(drop=True)

    return output_dataframe


# -----------------------------------------------------------------------------
# SQL read helpers
# -----------------------------------------------------------------------------

def load_rebuilt_for_final_output(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_observations_rebuilt_stage",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    complete_only: bool = True,
    observation_index_min: Optional[int] = None,
    observation_index_max: Optional[int] = None,
) -> pd.DataFrame:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    where_clauses = []
    params: dict[str, object] = {}

    if dataset_id is not None:
        where_clauses.append("dataset_id = :dataset_id")
        params["dataset_id"] = str(dataset_id).strip()

    if run_id is not None:
        where_clauses.append("run_id = :run_id")
        params["run_id"] = str(run_id).strip()

    if complete_only:
        where_clauses.append("rebuild_is_complete = TRUE")

    if observation_index_min is not None:
        where_clauses.append("observation_index >= :observation_index_min")
        params["observation_index_min"] = int(observation_index_min)

    if observation_index_max is not None:
        where_clauses.append("observation_index <= :observation_index_max")
        params["observation_index_max"] = int(observation_index_max)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    sql = f"""
    SELECT *
    FROM "{safe_schema}"."{safe_table}"
    {where_sql}
    ORDER BY observation_index
    """
    return read_sql_dataframe(engine, sql, params=params)


def _get_rebuilt_observation_bounds(
    engine,
    *,
    schema: str,
    table_name: str,
    dataset_id: str,
    run_id: str,
    complete_only: bool,
) -> dict:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    where_clauses = [
        "dataset_id = :dataset_id",
        "run_id = :run_id",
    ]
    params: dict[str, object] = {
        "dataset_id": str(dataset_id).strip(),
        "run_id": str(run_id).strip(),
    }

    if complete_only:
        where_clauses.append("rebuild_is_complete = TRUE")

    where_sql = "WHERE " + " AND ".join(where_clauses)

    bounds_dataframe = read_sql_dataframe(
        engine,
        f"""
        SELECT
            COUNT(*) AS row_count,
            MIN(observation_index) AS min_observation_index,
            MAX(observation_index) AS max_observation_index
        FROM "{safe_schema}"."{safe_table}"
        {where_sql}
        """,
        params=params,
    )

    row = bounds_dataframe.iloc[0]
    row_count = int(row["row_count"] or 0)

    return {
        "row_count": row_count,
        "min_observation_index": None if pd.isna(row["min_observation_index"]) else int(row["min_observation_index"]),
        "max_observation_index": None if pd.isna(row["max_observation_index"]) else int(row["max_observation_index"]),
    }


# -----------------------------------------------------------------------------
# Target table helpers
# -----------------------------------------------------------------------------

def ensure_final_aligned_synthetic_output_table_exists(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_observations_final_output",
    n_sensors: int = 52,
    timestamp_output_column: str = "timestamp",
    machine_status_output_column: str = "machine_status",
) -> str:
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = sanitize_sql_identifier(table_name)
    safe_timestamp_output_column = sanitize_sql_identifier(timestamp_output_column)
    safe_machine_status_output_column = sanitize_sql_identifier(machine_status_output_column)

    sensor_columns_sql = ",\n        ".join(
        f'"{sanitize_sql_identifier(column)}" DOUBLE PRECISION'
        for column in _build_sensor_columns(n_sensors)
    )

    sql = f"""
    CREATE TABLE IF NOT EXISTS "{safe_schema}"."{safe_table}" (
        dataset_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        "{safe_timestamp_output_column}" TIMESTAMPTZ,
        {sensor_columns_sql},
        "{safe_machine_status_output_column}" TEXT
    );
    """
    execute_sql(engine, sql)

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_dataset_run_asset"
        ON "{safe_schema}"."{safe_table}" (dataset_id, run_id, asset_id);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_{safe_timestamp_output_column}"
        ON "{safe_schema}"."{safe_table}" ("{safe_timestamp_output_column}");
        '''
    )

    return safe_table


def write_final_aligned_synthetic_output(
    engine,
    dataframe: pd.DataFrame,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_observations_final_output",
    n_sensors: int = 52,
    timestamp_output_column: str = "timestamp",
    machine_status_output_column: str = "machine_status",
    if_exists: str = "replace",
) -> str:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if dataframe.empty:
        return sanitize_sql_identifier(table_name)

    if if_exists not in {"replace", "append"}:
        raise ValueError("if_exists must be either 'replace' or 'append'.")

    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = sanitize_sql_identifier(table_name)

    if if_exists == "replace":
        execute_sql(engine, f'DROP TABLE IF EXISTS "{safe_schema}"."{safe_table}";')

    safe_table = ensure_final_aligned_synthetic_output_table_exists(
        engine,
        schema=safe_schema,
        table_name=safe_table,
        n_sensors=n_sensors,
        timestamp_output_column=timestamp_output_column,
        machine_status_output_column=machine_status_output_column,
    )

    working = dataframe.copy()
    working.columns = [sanitize_sql_identifier(column) for column in working.columns]

    _add_missing_columns(
        engine,
        schema=safe_schema,
        table=safe_table,
        dataframe=working,
    )

    write_layer_dataframe(
        engine=engine,
        dataframe=working,
        schema=safe_schema,
        table_name=safe_table,
        if_exists="append",
        index=False,
    )

    return safe_table


# -----------------------------------------------------------------------------
# Orchestration helper
# -----------------------------------------------------------------------------

def build_synthetic_final_aligned_output_stage(
    engine,
    *,
    schema: str = "capstone",
    rebuilt_table: str = "synthetic_sensor_observations_rebuilt_stage",
    target_table: str = "synthetic_sensor_observations_final_output",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    n_sensors: int = 52,
    complete_only: bool = True,
    if_exists: str = "replace",
    observation_window_size: int = 2500,
    timestamp_source_priority: Sequence[str] = (
        "observation_timestamp",
        "timestamp",
        "created_at",
    ),
    status_source_priority: Sequence[str] = (
        "machine_status",
        "stream_state",
        "phase",
    ),
    status_mapping: Optional[Mapping[str, str]] = None,
    timestamp_output_column: str = "timestamp",
    machine_status_output_column: str = "machine_status",
) -> dict:
    safe_schema = sanitize_sql_identifier(schema)
    safe_rebuilt_table = sanitize_sql_identifier(rebuilt_table)
    safe_target_table = sanitize_sql_identifier(target_table)

    resolved_dataset_id, resolved_run_id = _resolve_dataset_run_from_table(
        engine,
        schema=safe_schema,
        table_name=safe_rebuilt_table,
        dataset_id=dataset_id,
        run_id=run_id,
    )

    bounds = _get_rebuilt_observation_bounds(
        engine,
        schema=safe_schema,
        table_name=safe_rebuilt_table,
        dataset_id=resolved_dataset_id,
        run_id=resolved_run_id,
        complete_only=complete_only,
    )

    stats = {
        "status": "empty",
        "dataset_id": resolved_dataset_id,
        "run_id": resolved_run_id,
        "rebuilt_rows_available": int(bounds["row_count"]),
        "rebuilt_rows_read": 0,
        "final_rows_written": 0,
        "windows_processed": 0,
        "target_table": safe_target_table,
        "timestamp_output_column": timestamp_output_column,
        "machine_status_output_column": machine_status_output_column,
    }

    if bounds["row_count"] == 0:
        return stats

    has_written_first_chunk = False
    min_observation_index = int(bounds["min_observation_index"])
    max_observation_index = int(bounds["max_observation_index"])

    for observation_index_min in range(
        min_observation_index,
        max_observation_index + 1,
        int(observation_window_size),
    ):
        observation_index_max = min(
            observation_index_min + int(observation_window_size) - 1,
            max_observation_index,
        )

        rebuilt_window = load_rebuilt_for_final_output(
            engine,
            schema=safe_schema,
            table_name=safe_rebuilt_table,
            dataset_id=resolved_dataset_id,
            run_id=resolved_run_id,
            complete_only=complete_only,
            observation_index_min=observation_index_min,
            observation_index_max=observation_index_max,
        )

        if rebuilt_window.empty:
            continue

        output_window = build_final_aligned_synthetic_output_dataframe(
            rebuilt_window,
            n_sensors=n_sensors,
            timestamp_source_priority=timestamp_source_priority,
            status_source_priority=status_source_priority,
            status_mapping=status_mapping,
            timestamp_output_column=timestamp_output_column,
            machine_status_output_column=machine_status_output_column,
            sort_output=False,
        )

        write_final_aligned_synthetic_output(
            engine,
            output_window,
            schema=safe_schema,
            table_name=safe_target_table,
            n_sensors=n_sensors,
            timestamp_output_column=timestamp_output_column,
            machine_status_output_column=machine_status_output_column,
            if_exists=if_exists if not has_written_first_chunk else "append",
        )

        has_written_first_chunk = True
        stats["status"] = "built"
        stats["rebuilt_rows_read"] += int(len(rebuilt_window))
        stats["final_rows_written"] += int(len(output_window))
        stats["windows_processed"] += 1

    return stats


__all__ = [
    "load_rebuilt_for_final_output",
    "build_final_aligned_synthetic_output_dataframe",
    "ensure_final_aligned_synthetic_output_table_exists",
    "write_final_aligned_synthetic_output",
    "build_synthetic_final_aligned_output_stage",
]
