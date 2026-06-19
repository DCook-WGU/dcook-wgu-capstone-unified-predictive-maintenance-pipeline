from __future__ import annotations

import os
import gc
import psutil

from numbers import Integral
from typing import Any, Callable, Mapping, Optional

import pandas as pd

from sqlalchemy import text

from utils.database.postgres import sanitize_sql_identifier


# -----------------------------------------------------------------------------
# Memory Logger Helpers
# -----------------------------------------------------------------------------


def memory_gb() -> float:
    """
    Return the current process resident memory usage in gigabytes.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def log_memory(label: str) -> None:
    """
    Print a labeled memory snapshot for long-running chunked notebook steps.
    """
    print(f"[memory] {label}: {memory_gb():.2f} GB")


def copy_sql_params(params: Mapping[Any, Any] | None = None) -> dict[str, Any]:
    """
    Copy SQL query parameters into a mutable string-keyed dictionary.

    This avoids Pylance incorrectly inferring byte-keyed dictionaries from
    dict(params or {}) while still preserving runtime behavior for pandas /
    SQLAlchemy named parameters.
    """
    if params is None:
        return {}

    return {
        str(key): value
        for key, value in params.items()
    }


# -----------------------------------------------------------------------------
# Get Table Columns
# -----------------------------------------------------------------------------


def get_table_columns(
    engine,
    *,
    schema_name: str,
    table_name: str,
) -> list[str]:
    """
    Return column names for a Postgres table without reading data rows.
    """
    safe_schema = sanitize_sql_identifier(schema_name)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f'''
    SELECT *
    FROM "{safe_schema}"."{safe_table}"
    LIMIT 0
    '''

    with engine.begin() as connection:
        dataframe = pd.read_sql(text(sql), connection)

    return list(dataframe.columns)

# -----------------------------------------------------------------------------
# Resolve Dataset Run From Table
# -----------------------------------------------------------------------------

def resolve_dataset_run_from_table(
    engine,
    *,
    schema_name: str,
    table_name: str,
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    where_sql: str = "",
    params: Optional[Mapping[str, Any]] = None,
) -> tuple[str, str]:
    """
    Resolve a single dataset_id/run_id pair from parameters or table contents.

    Raises when only one identifier is provided, no matching rows exist, or the
    filtered table contains multiple dataset/run pairs.
    """
    safe_schema = sanitize_sql_identifier(schema_name)
    safe_table = sanitize_sql_identifier(table_name)
    query_params: dict[str, Any] = copy_sql_params(params)

    if dataset_id is not None and run_id is not None:
        return str(dataset_id).strip(), str(run_id).strip()

    if dataset_id is None and run_id is None:
        sql = f'''
        SELECT DISTINCT dataset_id, run_id
        FROM "{safe_schema}"."{safe_table}"
        {where_sql}
        ORDER BY dataset_id, run_id
        '''
        with engine.begin() as connection:
            dataframe = pd.read_sql(text(sql), connection, params=query_params)

        if dataframe.empty:
            raise ValueError(
                f"No dataset_id/run_id rows found in {safe_schema}.{safe_table}."
            )

        if len(dataframe) != 1:
            raise ValueError(
                f"{safe_schema}.{safe_table} contains multiple dataset_id/run_id pairs. "
                "Pass dataset_id and run_id explicitly for chunked processing."
            )

        return (
            str(dataframe.loc[0, "dataset_id"]).strip(),
            str(dataframe.loc[0, "run_id"]).strip(),
        )

    raise ValueError("dataset_id and run_id must both be provided together, or both omitted.")

# -----------------------------------------------------------------------------
# Get Table Row Count
# -----------------------------------------------------------------------------


def get_table_row_count(
    engine,
    *,
    schema_name: str,
    table_name: str,
    where_sql: str = "",
    params: Optional[Mapping[str, Any]] = None,
) -> int:
    """
    Count rows in a Postgres table, optionally using a caller-supplied WHERE clause.
    """
    safe_schema = sanitize_sql_identifier(schema_name)
    safe_table = sanitize_sql_identifier(table_name)
    query_params: dict[str, Any] = copy_sql_params(params)

    sql = f'''
    SELECT COUNT(*) AS row_count
    FROM "{safe_schema}"."{safe_table}"
    {where_sql}
    '''

    with engine.begin() as connection:
        dataframe = pd.read_sql(text(sql), connection, params=query_params)

    raw_row_count = dataframe.at[0, "row_count"]

    if not isinstance(raw_row_count, Integral):
        raise TypeError(
            f"Expected row_count to be an integer-compatible value, "
            f"got {type(raw_row_count).__name__}: {raw_row_count!r}"
        )

    return int(raw_row_count)

# -----------------------------------------------------------------------------
# Read Table Chunk By Row Number
# -----------------------------------------------------------------------------


def read_table_chunk_by_row_number(
    engine,
    *,
    schema_name: str,
    table_name: str,
    select_columns: list[str],
    order_by_sql: str,
    start_row: int,
    chunk_size: int,
    where_sql: str = "",
    params: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """
    Read one deterministic row-number window from a Postgres table.

    The caller supplies the selected columns and ORDER BY expression used to
    create the row numbers for chunked processing.
    """
    safe_schema = sanitize_sql_identifier(schema_name)
    safe_table = sanitize_sql_identifier(table_name)
    query_params: dict[str, Any] = copy_sql_params(params)

    quoted_columns = ", ".join([f'"{column}"' for column in select_columns])
    end_row = int(start_row) + int(chunk_size)

    sql = f'''
    WITH ordered_source AS (
        SELECT
            {quoted_columns},
            ROW_NUMBER() OVER (ORDER BY {order_by_sql}) AS __row_num
        FROM "{safe_schema}"."{safe_table}"
        {where_sql}
    )
    SELECT {quoted_columns}
    FROM ordered_source
    WHERE __row_num > :start_row
      AND __row_num <= :end_row
    ORDER BY __row_num
    '''

    query_params["start_row"] = int(start_row)
    query_params["end_row"] = int(end_row)

    with engine.begin() as connection:
        dataframe = pd.read_sql(text(sql), connection, params=query_params)

    return dataframe

# -----------------------------------------------------------------------------
# Process Postgres Table In Chunks
# -----------------------------------------------------------------------------

def process_postgres_table_in_chunks(
    engine,
    *,
    schema_name: str,
    table_name: str,
    select_columns: list[str],
    order_by_sql: str,
    transform_chunk_func: Callable[[pd.DataFrame, int, int, int], Any],
    write_chunk_func: Callable[[Any, int, int, int], None],
    chunk_size: int = 10000,
    where_sql: str = "",
    params: Optional[Mapping[str, Any]] = None,
    enable_memory_logging: bool = False,
) -> None:
    """
    Stream a Postgres table through transform and write callbacks in row chunks.

    Prints chunk progress and optionally memory snapshots; the callbacks perform
    the caller-specific transformation and persistence.
    """
    total_rows = get_table_row_count(
        engine,
        schema_name=schema_name,
        table_name=table_name,
        where_sql=where_sql,
        params=params,
    )

    if total_rows == 0:
        return

    for chunk_number, start_row in enumerate(range(0, total_rows, chunk_size), start=1):
        if enable_memory_logging:
            log_memory(f"chunk {chunk_number} - before read")

        end_row = min(start_row + chunk_size, total_rows)

        print(
            f"[chunk] {chunk_number} | "
            f"source rows {start_row:,} to {end_row - 1:,}"
        )

        dataframe_chunk = read_table_chunk_by_row_number(
            engine,
            schema_name=schema_name,
            table_name=table_name,
            select_columns=select_columns,
            order_by_sql=order_by_sql,
            start_row=start_row,
            chunk_size=chunk_size,
            where_sql=where_sql,
            params=params,
        )

        if dataframe_chunk.empty:
            del dataframe_chunk
            gc.collect()

            if enable_memory_logging:
                log_memory(f"chunk {chunk_number} - empty chunk after gc")

            continue

        if enable_memory_logging:
            log_memory(f"chunk {chunk_number} - after read")

        transformed = transform_chunk_func(
            dataframe_chunk,
            chunk_number,
            start_row,
            end_row,
        )

        if enable_memory_logging:
            log_memory(f"chunk {chunk_number} - after transform")

        write_chunk_func(
            transformed,
            chunk_number,
            start_row,
            end_row,
        )

        if enable_memory_logging:
            log_memory(f"chunk {chunk_number} - after write")

        del dataframe_chunk
        del transformed
        gc.collect()

        if enable_memory_logging:
            log_memory(f"chunk {chunk_number} - after gc")

# -----------------------------------------------------------------------------
# Get Observation Index Bounds
# -----------------------------------------------------------------------------

def get_observation_index_bounds(
    engine,
    *,
    schema_name: str,
    table_name: str,
    dataset_id: str,
    run_id: str,
    extra_where_sql: str = "",
    params: Optional[Mapping[str, Any]] = None,
) -> tuple[Optional[int], Optional[int]]:
    """
    Return the min/max observation_index for one dataset/run filter.
    """
    safe_schema = sanitize_sql_identifier(schema_name)
    safe_table = sanitize_sql_identifier(table_name)
    params = dict(params or {})

    sql = f'''
    SELECT
        MIN(observation_index) AS min_observation_index,
        MAX(observation_index) AS max_observation_index
    FROM "{safe_schema}"."{safe_table}"
    WHERE dataset_id = :dataset_id
      AND run_id = :run_id
      {extra_where_sql}
    '''

    params["dataset_id"] = str(dataset_id).strip()
    params["run_id"] = str(run_id).strip()

    with engine.begin() as connection:
        dataframe = pd.read_sql(text(sql), connection, params=params)

    min_value = dataframe.at[0, "min_observation_index"]
    max_value = dataframe.at[0, "max_observation_index"]

    if pd.isna(min_value) or pd.isna(max_value):
        return None, None

    if not isinstance(min_value, Integral):
        raise TypeError(
            f"Expected min_observation_index to be integer-compatible, "
            f"got {type(min_value).__name__}: {min_value!r}"
        )

    if not isinstance(max_value, Integral):
        raise TypeError(
            f"Expected max_observation_index to be integer-compatible, "
            f"got {type(max_value).__name__}: {max_value!r}"
        )

    return int(min_value), int(max_value)

# -----------------------------------------------------------------------------
# Read Table For Observation Window
# -----------------------------------------------------------------------------

def read_table_for_observation_window(
    engine,
    *,
    schema_name: str,
    table_name: str,
    select_columns: list[str],
    dataset_id: str,
    run_id: str,
    observation_index_min: int,
    observation_index_max: int,
    extra_where_sql: str = "",
    params: Optional[Mapping[str, Any]] = None,
    order_by_sql: str = "observation_index",
) -> pd.DataFrame:
    """
    Read rows for one inclusive observation_index window and dataset/run pair.
    """
    safe_schema = sanitize_sql_identifier(schema_name)
    safe_table = sanitize_sql_identifier(table_name)
    params = dict(params or {})

    quoted_columns = ", ".join([f'"{column}"' for column in select_columns])

    sql = f'''
    SELECT {quoted_columns}
    FROM "{safe_schema}"."{safe_table}"
    WHERE dataset_id = :dataset_id
      AND run_id = :run_id
      AND observation_index >= :observation_index_min
      AND observation_index <= :observation_index_max
      {extra_where_sql}
    ORDER BY {order_by_sql}
    '''

    params["dataset_id"] = str(dataset_id).strip()
    params["run_id"] = str(run_id).strip()
    params["observation_index_min"] = int(observation_index_min)
    params["observation_index_max"] = int(observation_index_max)

    with engine.begin() as connection:
        dataframe = pd.read_sql(text(sql), connection, params=params)

    return dataframe

# -----------------------------------------------------------------------------
# Process Observation Index Windows
# -----------------------------------------------------------------------------

def process_observation_index_windows(
    engine,
    *,
    schema_name: str,
    table_name: str,
    select_columns: list[str],
    dataset_id: str,
    run_id: str,
    transform_chunk_func: Callable[[pd.DataFrame, int, int, int], Any],
    write_chunk_func: Callable[[Any, int, int, int], None],
    window_size: int = 5000,
    extra_where_sql: str = "",
    params: Optional[Mapping[str, Any]] = None,
    order_by_sql: str = "observation_index",
) -> None:
    """
    Process a dataset/run table in observation_index windows.

    Prints window progress, reads each non-empty window, and delegates transform
    and write behavior to caller-supplied callbacks.
    """
    min_index, max_index = get_observation_index_bounds(
        engine,
        schema_name=schema_name,
        table_name=table_name,
        dataset_id=dataset_id,
        run_id=run_id,
        extra_where_sql=extra_where_sql,
        params=params,
    )

    if min_index is None or max_index is None:
        return

    window_number = 0

    for observation_index_min in range(min_index, max_index + 1, window_size):
        observation_index_max = min(observation_index_min + window_size - 1, max_index)
        window_number += 1

        print(
            f"[obs-window] {window_number} | "
            f"observation_index {observation_index_min:,} to {observation_index_max:,}"
        )

        dataframe_window = read_table_for_observation_window(
            engine,
            schema_name=schema_name,
            table_name=table_name,
            select_columns=select_columns,
            dataset_id=dataset_id,
            run_id=run_id,
            observation_index_min=observation_index_min,
            observation_index_max=observation_index_max,
            extra_where_sql=extra_where_sql,
            params=params,
            order_by_sql=order_by_sql,
        )

        if dataframe_window.empty:
            continue

        transformed = transform_chunk_func(
            dataframe_window,
            window_number,
            observation_index_min,
            observation_index_max,
        )
        write_chunk_func(
            transformed,
            window_number,
            observation_index_min,
            observation_index_max,
        )

        del dataframe_window
        del transformed
        gc.collect()
