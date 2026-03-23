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

import os
import re
from typing import Any, Iterable, Optional

import pandas as pd


DEFAULT_STREAM_STATE_MAP = {
    "normal": "NORMAL",
    "abnormal": "BROKEN",
    "recovery": "RECOVERING",
}

DEFAULT_PHASE_MAP = {
    "normal_before": "NORMAL",
    "buildup": "NORMAL",
    "failure": "BROKEN",
    "recovery": "RECOVERING",
    "normal_after": "NORMAL",
    "normal": "NORMAL",
    "abnormal": "BROKEN",
}


def _postgres_credentials_from_env() -> dict[str, str]:
    return {
        "host": os.getenv("DB_HOST") or os.getenv("POSTGRES_HOST") or "localhost",
        "port": os.getenv("DB_PORT") or os.getenv("POSTGRES_PORT") or "5432",
        "dbname": os.getenv("DB_NAME") or os.getenv("POSTGRES_DB") or "postgres",
        "user": os.getenv("DB_USER") or os.getenv("POSTGRES_USER") or "postgres",
        "password": os.getenv("DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD") or "postgres",
    }


def build_postgres_engine_from_env() -> Any:
    """
    Build a Postgres connection object from common environment variable names.

    Return order:
    1. SQLAlchemy engine if SQLAlchemy + psycopg are installed
    2. psycopg connection
    3. psycopg2 connection

    This keeps the notebook flexible across slightly different environments.
    """
    creds = _postgres_credentials_from_env()

    # First choice: SQLAlchemy engine
    try:
        from sqlalchemy import create_engine

        try:
            import psycopg  # noqa: F401

            driver = "postgresql+psycopg"
        except Exception:
            driver = "postgresql+psycopg2"

        url = (
            f"{driver}://{creds['user']}:{creds['password']}"
            f"@{creds['host']}:{creds['port']}/{creds['dbname']}"
        )
        return create_engine(url)
    except Exception:
        pass

    # Second choice: psycopg
    try:
        import psycopg

        return psycopg.connect(**creds)
    except Exception:
        pass

    # Third choice: psycopg2
    try:
        import psycopg2

        return psycopg2.connect(**creds)
    except Exception as exc:
        raise ImportError(
            "No supported Postgres client is installed. Install one of: "
            "sqlalchemy + psycopg, psycopg, or psycopg2."
        ) from exc


def _is_sqlalchemy_engine(conn: Any) -> bool:
    return hasattr(conn, "connect") and hasattr(conn, "dialect")


def _read_sql_dataframe(conn: Any, sql: str) -> pd.DataFrame:
    if _is_sqlalchemy_engine(conn):
        return pd.read_sql(sql, conn)

    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    finally:
        cursor.close()


def _execute_sql(conn: Any, sql: str) -> None:
    if _is_sqlalchemy_engine(conn):
        with conn.begin() as active_conn:
            active_conn.exec_driver_sql(sql)
        return

    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        conn.commit()
    finally:
        cursor.close()


def table_exists(conn: Any, *, schema: str, table_name: str) -> bool:
    sql = f"""
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = '{schema}'
          AND table_name = '{table_name}'
    ) AS exists_flag
    """
    df = _read_sql_dataframe(conn, sql)
    return bool(df.loc[0, "exists_flag"])


def get_table_columns(conn: Any, *, schema: str, table_name: str) -> list[str]:
    sql = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = '{schema}'
      AND table_name = '{table_name}'
    ORDER BY ordinal_position
    """
    df = _read_sql_dataframe(conn, sql)
    return df["column_name"].astype(str).tolist()


def get_sensor_columns(columns: Iterable[str]) -> list[str]:
    """
    Return sensor columns in numeric order: sensor_00, sensor_01, ..., sensor_51.
    """
    sensor_cols = []

    for col in columns:
        col = str(col)
        match = re.fullmatch(r"sensor_(\d+)", col)
        if match:
            sensor_cols.append((int(match.group(1)), col))

    sensor_cols = sorted(sensor_cols, key=lambda item: item[0])
    return [col for _, col in sensor_cols]


def quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def build_batch_filter_sql(batch_ids: Optional[Iterable[int]]) -> str:
    if batch_ids is None:
        return ""

    cleaned = []
    for value in batch_ids:
        cleaned.append(str(int(value)))

    if not cleaned:
        return ""

    joined = ", ".join(cleaned)
    return f" WHERE batch_id IN ({joined}) "


def read_synthetic_stream_table(
    conn: Any,
    *,
    schema: str,
    table_name: str,
    batch_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Read the synthetic stream table from Postgres.
    """
    fq_table = f"{quote_identifier(schema)}.{quote_identifier(table_name)}"
    where_sql = build_batch_filter_sql(batch_ids)
    sql = f"SELECT * FROM {fq_table}{where_sql}"
    return _read_sql_dataframe(conn, sql)


def choose_sort_columns(df: pd.DataFrame) -> list[str]:
    """
    Choose the best available sort columns for creating one unified sequence.
    """
    preferred = [
        "batch_id",
        "global_cycle_id",
        "cycle_id",
        "global_row_id",
        "row_in_batch",
        "created_at",
    ]
    return [col for col in preferred if col in df.columns]


def sort_synthetic_stream(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a stable sort so multiple batches become one ordered stream.
    """
    df = df.copy()
    sort_cols = choose_sort_columns(df)

    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def add_unified_row_id(
    df: pd.DataFrame,
    *,
    row_id_col: str = "unified_row_id",
    start_at: int = 1,
) -> pd.DataFrame:
    df = df.copy()
    df[row_id_col] = range(start_at, start_at + len(df))
    return df


def add_unified_episode_id(
    df: pd.DataFrame,
    *,
    original_episode_col: str = "meta__episode_id",
    batch_col: str = "batch_id",
    unified_episode_col: str = "meta__episode_id_unified",
    start_at: int = 0,
) -> pd.DataFrame:
    """
    Batches often restart episode numbering at 0. This creates a batch-safe
    episode id across the combined table.
    """
    df = df.copy()

    if original_episode_col not in df.columns:
        return df

    key_cols = [original_episode_col]
    if batch_col in df.columns:
        key_cols = [batch_col, original_episode_col]

    distinct_keys = df[key_cols].drop_duplicates().reset_index(drop=True)
    distinct_keys[unified_episode_col] = range(start_at, start_at + len(distinct_keys))

    df = df.merge(distinct_keys, how="left", on=key_cols)
    return df


def add_time_index_and_timestamps(
    df: pd.DataFrame,
    *,
    start_timestamp: str = "2018-04-01 00:00:00",
    frequency: str = "1min",
    time_index_col: str = "observation_time_index",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Add a clean sequential time index and timestamp field.
    """
    df = df.copy()

    if len(df) == 0:
        df[time_index_col] = pd.Series(dtype="int64")
        df[timestamp_col] = pd.Series(dtype="datetime64[ns]")
        return df

    df[time_index_col] = range(len(df))
    df[timestamp_col] = pd.date_range(start=start_timestamp, periods=len(df), freq=frequency)
    return df


def derive_machine_status(
    df: pd.DataFrame,
    *,
    stream_state_col: str = "stream_state",
    phase_col: str = "phase",
    output_col: str = "machine_status",
    stream_state_map: Optional[dict[str, str]] = None,
    phase_map: Optional[dict[str, str]] = None,
    default_value: str = "NORMAL",
) -> pd.DataFrame:
    """
    Convert generator labels into the original dataset label style.
    """
    df = df.copy()

    stream_state_map = stream_state_map or DEFAULT_STREAM_STATE_MAP
    phase_map = phase_map or DEFAULT_PHASE_MAP

    label_series = pd.Series([default_value] * len(df), index=df.index, dtype="object")

    if stream_state_col in df.columns:
        mapped_state = df[stream_state_col].astype(str).str.strip().str.lower().map(stream_state_map)
        label_series = mapped_state.fillna(label_series)

    if phase_col in df.columns:
        mapped_phase = df[phase_col].astype(str).str.strip().str.lower().map(phase_map)
        label_series = label_series.where(label_series.ne(default_value), mapped_phase.fillna(default_value))

    df[output_col] = label_series.fillna(default_value)
    return df


def trim_dataframe(
    df: pd.DataFrame,
    *,
    target_total_rows: Optional[int] = None,
    trim_mode: str = "head",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Optional row trimming after the unified table is built.

    trim_mode:
    - head
    - tail
    - random
    """
    df = df.copy()

    if target_total_rows is None:
        return df

    target_total_rows = int(target_total_rows)

    if target_total_rows <= 0:
        raise ValueError("target_total_rows must be greater than 0")

    if len(df) <= target_total_rows:
        return df

    trim_mode = str(trim_mode).strip().lower()

    if trim_mode == "head":
        return df.head(target_total_rows).copy()

    if trim_mode == "tail":
        return df.tail(target_total_rows).copy()

    if trim_mode == "random":
        return df.sample(n=target_total_rows, random_state=random_state).sort_index().copy()

    raise ValueError("trim_mode must be one of: head, tail, random")


def reorder_for_original_dataset(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    status_col: str = "machine_status",
    keep_lineage_columns: bool = True,
) -> pd.DataFrame:
    """
    Reorder columns to look like the original pump dataset, with optional
    lineage columns kept at the end for Bronze handoff auditing.
    """
    df = df.copy()
    sensor_cols = get_sensor_columns(df.columns)

    base_cols = []
    if timestamp_col in df.columns:
        base_cols.append(timestamp_col)

    base_cols.extend(sensor_cols)

    if status_col in df.columns:
        base_cols.append(status_col)

    lineage_preferred = [
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

    lineage_cols = [col for col in lineage_preferred if col in df.columns]

    if keep_lineage_columns:
        ordered = base_cols + [col for col in lineage_cols if col not in base_cols]
    else:
        ordered = base_cols

    remaining = [col for col in df.columns if col not in ordered]
    ordered = ordered + remaining

    return df.loc[:, ordered].copy()


def prepare_bronze_ready_dataframe(
    raw_df: pd.DataFrame,
    *,
    start_timestamp: str = "2018-04-01 00:00:00",
    frequency: str = "1min",
    keep_lineage_columns: bool = True,
    target_total_rows: Optional[int] = None,
    trim_mode: str = "head",
) -> pd.DataFrame:
    """
    Full in-memory pipeline for converting the raw synthetic stream dataframe
    into a bronze-ready dataframe.
    """
    df = raw_df.copy()

    df = sort_synthetic_stream(df)
    df = add_unified_row_id(df)
    df = add_unified_episode_id(df)
    df = add_time_index_and_timestamps(
        df,
        start_timestamp=start_timestamp,
        frequency=frequency,
    )
    df = derive_machine_status(df)
    df = trim_dataframe(
        df,
        target_total_rows=target_total_rows,
        trim_mode=trim_mode,
    )
    df = reorder_for_original_dataset(
        df,
        keep_lineage_columns=keep_lineage_columns,
    )

    return df


def summarize_bronze_ready_dataframe(df: pd.DataFrame) -> dict:
    sensor_cols = get_sensor_columns(df.columns)

    summary = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "sensor_column_count": int(len(sensor_cols)),
        "first_timestamp": None,
        "last_timestamp": None,
        "status_counts": {},
        "batch_count": None,
        "episode_count_unified": None,
    }

    if "timestamp" in df.columns and len(df) > 0:
        summary["first_timestamp"] = str(pd.to_datetime(df["timestamp"]).min())
        summary["last_timestamp"] = str(pd.to_datetime(df["timestamp"]).max())

    if "machine_status" in df.columns:
        summary["status_counts"] = {
            str(k): int(v) for k, v in df["machine_status"].value_counts(dropna=False).to_dict().items()
        }

    if "batch_id" in df.columns:
        summary["batch_count"] = int(df["batch_id"].nunique())

    if "meta__episode_id_unified" in df.columns:
        summary["episode_count_unified"] = int(df["meta__episode_id_unified"].nunique())

    return summary


def create_schema_if_missing(conn: Any, *, schema: str) -> None:
    sql = f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(schema)}"
    _execute_sql(conn, sql)


def _postgres_type_for_series(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE PRECISION"
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    return "TEXT"


def _write_dataframe_with_dbapi(
    df: pd.DataFrame,
    conn: Any,
    *,
    schema: str,
    table_name: str,
    if_exists: str = "replace",
    index: bool = False,
) -> None:
    if index:
        df = df.reset_index()

    create_schema_if_missing(conn, schema=schema)

    fq_table = f"{quote_identifier(schema)}.{quote_identifier(table_name)}"

    if if_exists == "replace":
        _execute_sql(conn, f"DROP TABLE IF EXISTS {fq_table}")
    elif if_exists not in {"append", "fail"}:
        raise ValueError("if_exists must be replace, append, or fail")

    if if_exists == "fail" and table_exists(conn, schema=schema, table_name=table_name):
        raise ValueError(f"Target table already exists: {schema}.{table_name}")

    if not table_exists(conn, schema=schema, table_name=table_name):
        column_sql_parts = []
        for col in df.columns:
            pg_type = _postgres_type_for_series(df[col])
            column_sql_parts.append(f"{quote_identifier(col)} {pg_type}")

        create_sql = f"CREATE TABLE {fq_table} ({', '.join(column_sql_parts)})"
        _execute_sql(conn, create_sql)

    if len(df) == 0:
        return

    placeholders = ", ".join(["%s"] * len(df.columns))
    insert_cols = ", ".join(quote_identifier(col) for col in df.columns)
    insert_sql = f"INSERT INTO {fq_table} ({insert_cols}) VALUES ({placeholders})"

    values = []
    for row in df.itertuples(index=False, name=None):
        clean_row = []
        for value in row:
            if pd.isna(value):
                clean_row.append(None)
            elif isinstance(value, pd.Timestamp):
                clean_row.append(value.to_pydatetime())
            else:
                clean_row.append(value)
        values.append(tuple(clean_row))

    cursor = conn.cursor()
    try:
        cursor.executemany(insert_sql, values)
        conn.commit()
    finally:
        cursor.close()


def write_dataframe_to_postgres(
    df: pd.DataFrame,
    conn: Any,
    *,
    schema: str,
    table_name: str,
    if_exists: str = "replace",
    index: bool = False,
) -> None:
    if _is_sqlalchemy_engine(conn):
        create_schema_if_missing(conn, schema=schema)
        df.to_sql(
            name=table_name,
            con=conn,
            schema=schema,
            if_exists=if_exists,
            index=index,
            method="multi",
            chunksize=5000,
        )
        return

    _write_dataframe_with_dbapi(
        df,
        conn,
        schema=schema,
        table_name=table_name,
        if_exists=if_exists,
        index=index,
    )


def build_bronze_ready_from_postgres(
    conn: Any,
    *,
    source_schema: str,
    source_table: str,
    batch_ids: Optional[Iterable[int]] = None,
    start_timestamp: str = "2018-04-01 00:00:00",
    frequency: str = "1min",
    keep_lineage_columns: bool = True,
    target_total_rows: Optional[int] = None,
    trim_mode: str = "head",
) -> pd.DataFrame:
    raw_df = read_synthetic_stream_table(
        conn,
        schema=source_schema,
        table_name=source_table,
        batch_ids=batch_ids,
    )

    bronze_df = prepare_bronze_ready_dataframe(
        raw_df,
        start_timestamp=start_timestamp,
        frequency=frequency,
        keep_lineage_columns=keep_lineage_columns,
        target_total_rows=target_total_rows,
        trim_mode=trim_mode,
    )

    return bronze_df