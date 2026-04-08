from __future__ import annotations

import gc
from typing import Any, Callable, Mapping, Optional

import pandas as pd
from sqlalchemy import text

from utils.postgres_util import sanitize_sql_identifier


def get_table_columns(
    engine,
    *,
    schema_name: str,
    table_name: str,
) -> list[str]:
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
    safe_schema = sanitize_sql_identifier(schema_name)
    safe_table = sanitize_sql_identifier(table_name)
    params = dict(params or {})

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
            dataframe = pd.read_sql(text(sql), connection, params=params)

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


def get_table_row_count(
    engine,
    *,
    schema_name: str,
    table_name: str,
    where_sql: str = "",
    params: Optional[Mapping[str, Any]] = None,
) -> int:
    safe_schema = sanitize_sql_identifier(schema_name)
    safe_table = sanitize_sql_identifier(table_name)
    params = dict(params or {})

    sql = f'''
    SELECT COUNT(*) AS row_count
    FROM "{safe_schema}"."{safe_table}"
    {where_sql}
    '''

    with engine.begin() as connection:
        dataframe = pd.read_sql(text(sql), connection, params=params)

    return int(dataframe.loc[0, "row_count"])


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
    safe_schema = sanitize_sql_identifier(schema_name)
    safe_table = sanitize_sql_identifier(table_name)
    params = dict(params or {})

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

    params["start_row"] = int(start_row)
    params["end_row"] = int(end_row)

    with engine.begin() as connection:
        dataframe = pd.read_sql(text(sql), connection, params=params)

    return dataframe


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
) -> None:
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
        end_row = min(start_row + chunk_size, total_rows)

        print(
            f"[chunk] {chunk_number} | "
            f"source rows {start_row:,} to {end_row - 1:,}"
        )

        df_chunk = read_table_chunk_by_row_number(
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

        if df_chunk.empty:
            continue

        transformed = transform_chunk_func(df_chunk, chunk_number, start_row, end_row)
        write_chunk_func(transformed, chunk_number, start_row, end_row)

        del df_chunk
        del transformed
        gc.collect()


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

    min_value = dataframe.loc[0, "min_observation_index"]
    max_value = dataframe.loc[0, "max_observation_index"]

    if pd.isna(min_value) or pd.isna(max_value):
        return None, None

    return int(min_value), int(max_value)


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

        df_window = read_table_for_observation_window(
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

        if df_window.empty:
            continue

        transformed = transform_chunk_func(
            df_window,
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

        del df_window
        del transformed
        gc.collect()