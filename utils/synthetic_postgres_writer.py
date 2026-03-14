from __future__ import annotations

from typing import Optional, List
import pandas as pd

from utils.postgres_util import (
    get_engine,
    create_schema_if_not_exists,
    execute_sql,
    write_layer_dataframe,
    read_sql_dataframe,
)

def ensure_sequence(engine, *, schema: str, sequence_name: str) -> None:
    create_schema_if_not_exists(engine, schema)
    sql = f'CREATE SEQUENCE IF NOT EXISTS "{schema}"."{sequence_name}"'
    execute_sql(engine, sql)


def reserve_next_batch_id(engine, *, schema: str, sequence_name: str) -> int:
    sql = f'SELECT nextval(\'"{schema}"."{sequence_name}"\') AS batch_id'
    dataframe = read_sql_dataframe(engine, sql)
    return int(dataframe.loc[0, "batch_id"])


def reserve_cycle_range(engine, *, schema: str, sequence_name: str, n_rows: int) -> int:
    if n_rows <= 0:
        raise ValueError("n_rows must be > 0")

    start_dataframe = read_sql_dataframe(engine, f'SELECT nextval(\'"{schema}"."{sequence_name}"\') AS v')
    start = int(start_dataframe.loc[0, "v"])

    if n_rows > 1:
        execute_sql(engine, f'SELECT setval(\'"{schema}"."{sequence_name}"\', {start + (n_rows - 1)})')

    return start


def _ensure_stream_table_exists(engine, *, schema: str, table: str) -> None:
    """
    Creates the base stream table if missing. Sensor columns are added dynamically later.
    """
    create_schema_if_not_exists(engine, schema)

    sql = f"""
    CREATE TABLE IF NOT EXISTS "{schema}"."{table}" (
        batch_id BIGINT NOT NULL,
        row_in_batch INTEGER NOT NULL,
        global_cycle_id BIGINT,
        stream_state TEXT,
        phase TEXT,
        created_at TIMESTAMPTZ DEFAULT now(),
        PRIMARY KEY (batch_id, row_in_batch)
    );
    """
    execute_sql(engine, sql)

    execute_sql(engine, f'CREATE INDEX IF NOT EXISTS "idx_{table}_batch" ON "{schema}"."{table}" (batch_id, row_in_batch);')
    execute_sql(engine, f'CREATE INDEX IF NOT EXISTS "idx_{table}_cycle" ON "{schema}"."{table}" (global_cycle_id);')


def _get_existing_columns(engine, *, schema: str, table: str) -> set[str]:
    columns_dataframe = read_sql_dataframe(
        engine,
        f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = '{table}'
        """
    )
    return set(columns_dataframe["column_name"].astype(str).tolist())


def _add_missing_columns(engine, *, schema: str, table: str, dataframe: pd.DataFrame) -> None:
    existing = _get_existing_columns(engine, schema=schema, table=table)
    desired: List[str] = list(map(str, dataframe.columns))

    missing = [column for column in desired if column not in existing]
    if not missing:
        return

    for column in missing:
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            column_type = "DOUBLE PRECISION"
        else:
            column_type = "TEXT"

        execute_sql(engine, f'ALTER TABLE "{schema}"."{table}" ADD COLUMN "{column}" {column_type};')

    # Optional: print a short debug line
    print(f"[synthetic] Added {len(missing)} new columns to {schema}.{table}")


def write_stream_batch(
    engine,
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    schema: str = "public",
    artifact_name: str = "stream",
    batch_id: int,
    cycle_start: Optional[int] = None,
) -> str:
    """
    Writes a batch to the table name:
      synthetic_<dataset_name>_<artifact_name>

    Behavior:
      - ensures base table exists
      - auto-adds any missing columns for this dataframe (sensor columns etc.)
      - appends rows
    """
    out = dataframe.copy()

    # Required ordering columns first
    out.insert(0, "batch_id", int(batch_id))
    out.insert(1, "row_in_batch", range(len(out)))

    if cycle_start is not None:
        out.insert(2, "global_cycle_id", [int(cycle_start) + i for i in range(len(out))])

    table = f"synthetic_{dataset_name}_{artifact_name}"

    # 1) Ensure base table exists
    _ensure_stream_table_exists(engine, schema=schema, table=table)

    # 2) Ensure columns exist (this is the missing piece)
    _add_missing_columns(engine, schema=schema, table=table, dataframe=out)

    # 3) Append
    table_name = write_layer_dataframe(
        engine=engine,
        dataframe=out,
        layer="synthetic",
        dataset_name=dataset_name,
        artifact_name=artifact_name,
        schema=schema,
        if_exists="append",
        index=False,
    )
    return table_name