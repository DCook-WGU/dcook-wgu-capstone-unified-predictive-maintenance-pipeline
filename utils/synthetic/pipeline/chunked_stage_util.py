from __future__ import annotations

import gc
from typing import Callable, Optional, Mapping, Any

import pandas as pd
from sqlalchemy import text


def get_table_row_count(
    engine,
    *,
    schema_name: str,
    table_name: str,
    where_sql: str = "",
    params: Optional[Mapping[str, Any]] = None,
) -> int:
    params = dict(params or {})

    sql = f'''
    SELECT COUNT(*) AS row_count
    FROM "{schema_name}"."{table_name}"
    {where_sql}
    '''

    with engine.begin() as connection:
        df_count = pd.read_sql(text(sql), connection, params=params)

    return int(df_count.loc[0, "row_count"])


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
    Generic Postgres chunk reader using ROW_NUMBER() windowing.

    start_row is 0-based.
    """
    params = dict(params or {})
    end_row = int(start_row) + int(chunk_size)

    quoted_columns = ", ".join([f'"{col}"' for col in select_columns])

    sql = f'''
    WITH ordered_source AS (
        SELECT
            {quoted_columns},
            ROW_NUMBER() OVER (ORDER BY {order_by_sql}) AS __row_num
        FROM "{schema_name}"."{table_name}"
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
        df_chunk = pd.read_sql(text(sql), connection, params=params)

    return df_chunk


def process_postgres_table_in_chunks(
    engine,
    *,
    schema_name: str,
    table_name: str,
    select_columns: list[str],
    order_by_sql: str,
    transform_chunk_func: Callable[[pd.DataFrame, int, int, int], pd.DataFrame],
    write_chunk_func: Callable[[pd.DataFrame, int, int, int], None],
    chunk_size: int = 10000,
    where_sql: str = "",
    params: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Generic read -> transform -> write chunk pipeline.

    transform_chunk_func signature:
        (df_chunk, chunk_number, start_row, end_row) -> pd.DataFrame

    write_chunk_func signature:
        (df_out, chunk_number, start_row, end_row) -> None
    """
    total_rows = get_table_row_count(
        engine,
        schema_name=schema_name,
        table_name=table_name,
        where_sql=where_sql,
        params=params,
    )

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
            print(f"[chunk] {chunk_number} returned 0 rows")
            continue

        df_out = transform_chunk_func(df_chunk, chunk_number, start_row, end_row)
        write_chunk_func(df_out, chunk_number, start_row, end_row)

        del df_chunk
        del df_out
        gc.collect()