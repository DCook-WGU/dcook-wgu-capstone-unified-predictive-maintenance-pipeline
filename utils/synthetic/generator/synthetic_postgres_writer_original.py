from __future__ import annotations

from typing import List, Optional

import pandas as pd

from utils.postgres_util import (
    sanitize_sql_identifier,
    create_schema_if_not_exists,
    execute_sql,
    read_sql_dataframe,
)
from utils.layer_postgres_writer import write_layer_dataframe


# -----------------------------------------------------------------------------
# Sequence helpers
# -----------------------------------------------------------------------------

def ensure_sequence(engine, *, schema: str, sequence_name: str) -> None:
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_sequence = sanitize_sql_identifier(sequence_name)
    sql = f'CREATE SEQUENCE IF NOT EXISTS "{safe_schema}"."{safe_sequence}"'
    execute_sql(engine, sql)



def reserve_next_batch_id(engine, *, schema: str, sequence_name: str) -> int:
    safe_schema = sanitize_sql_identifier(schema)
    safe_sequence = sanitize_sql_identifier(sequence_name)
    sql = f'SELECT nextval(\'"{safe_schema}"."{safe_sequence}"\') AS batch_id'
    dataframe = read_sql_dataframe(engine, sql)
    return int(dataframe.loc[0, "batch_id"])



def reserve_cycle_range(engine, *, schema: str, sequence_name: str, n_rows: int) -> int:
    if n_rows <= 0:
        raise ValueError("n_rows must be > 0")

    safe_schema = sanitize_sql_identifier(schema)
    safe_sequence = sanitize_sql_identifier(sequence_name)

    start_dataframe = read_sql_dataframe(
        engine,
        f'SELECT nextval(\'"{safe_schema}"."{safe_sequence}"\') AS v',
    )
    start = int(start_dataframe.loc[0, "v"])

    if n_rows > 1:
        execute_sql(
            engine,
            f'SELECT setval(\'"{safe_schema}"."{safe_sequence}"\', {start + (n_rows - 1)})',
        )

    return start

def reset_sequence(engine, *, schema: str, sequence_name: str, start_at: int = 1) -> None:
    # nextval returns start_at when is_called=false
    sql = f"SELECT setval('\"{schema}\".\"{sequence_name}\"', {int(start_at)}, false)"
    execute_sql(engine, sql)


def reset_synthetic_sequences(engine, *, schema: str, dataset_name: str) -> None:
    ds = str(dataset_name).strip().lower()
    reset_sequence(engine, schema=schema, sequence_name=f"seq_synthetic_{ds}_batch_id", start_at=1)
    reset_sequence(engine, schema=schema, sequence_name=f"seq_synthetic_{ds}_cycle_id", start_at=1)


# -----------------------------------------------------------------------------
# Stream table helpers
# -----------------------------------------------------------------------------

def _ensure_stream_table_exists(engine, *, schema: str, table: str) -> None:
    """
    Create the base stream table if missing.
    Sensor columns are added dynamically later.
    """
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = sanitize_sql_identifier(table)

    sql = f"""
    CREATE TABLE IF NOT EXISTS "{safe_schema}"."{safe_table}" (
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

    execute_sql(
        engine,
        f'CREATE INDEX IF NOT EXISTS "idx_{safe_table}_batch" ON "{safe_schema}"."{safe_table}" (batch_id, row_in_batch);',
    )
    execute_sql(
        engine,
        f'CREATE INDEX IF NOT EXISTS "idx_{safe_table}_cycle" ON "{safe_schema}"."{safe_table}" (global_cycle_id);',
    )



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

    existing = _get_existing_columns(engine, schema=safe_schema, table=safe_table)
    desired: List[str] = [sanitize_sql_identifier(column) for column in dataframe.columns]

    missing = [column for column in desired if column not in existing]
    if not missing:
        return

    working = dataframe.copy()
    working.columns = [sanitize_sql_identifier(column) for column in working.columns]

    for column in missing:
        column_type = _infer_alter_column_type(working[column])
        execute_sql(
            engine,
            f'ALTER TABLE "{safe_schema}"."{safe_table}" ADD COLUMN "{column}" {column_type};',
        )

    print(f"[synthetic] Added {len(missing)} new columns to {safe_schema}.{safe_table}")


# -----------------------------------------------------------------------------
# Batch writer
# -----------------------------------------------------------------------------

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
    Write a synthetic stream batch to the table:
      synthetic_<dataset_name>_<artifact_name>

    Behavior:
      - ensures the base stream table exists
      - auto-adds any missing columns for this dataframe
      - appends rows through the generic layer writer
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    out = dataframe.copy()

    out.insert(0, "batch_id", int(batch_id))
    out.insert(1, "row_in_batch", range(len(out)))

    if cycle_start is not None:
        out.insert(2, "global_cycle_id", [int(cycle_start) + i for i in range(len(out))])

    table = f"synthetic_{dataset_name}_{artifact_name}"

    _ensure_stream_table_exists(engine, schema=schema, table=table)
    _add_missing_columns(engine, schema=schema, table=table, dataframe=out)

    table_name = write_layer_dataframe(
        engine=engine,
        dataframe=out,
        schema=schema,
        table_name=table,
        if_exists="append",
        index=False,
    )
    return table_name


__all__ = [
    "ensure_sequence",
    "reserve_next_batch_id",
    "reserve_cycle_range",
    "write_stream_batch",
]
