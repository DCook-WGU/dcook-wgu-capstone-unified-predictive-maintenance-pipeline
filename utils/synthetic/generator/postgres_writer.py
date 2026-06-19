from __future__ import annotations

import csv
import io
import json
from typing import List, Optional, Any, cast

import math
import numpy as np
import pandas as pd

from utils.database.postgres import (
    sanitize_sql_identifier,
    create_schema_if_not_exists,
    execute_sql,
    read_sql_dataframe,
)
from utils.database.layer_postgres import write_layer_dataframe


# -----------------------------------------------------------------------------
# Sequence helpers
# -----------------------------------------------------------------------------

def scalar_to_int(value: object, name: str = "value") -> int:
    """Convert a SQL scalar value to int and reject missing sequence values."""
    if value is None:
        raise ValueError(f"{name} cannot be missing.")

    if value is pd.NA:
        raise ValueError(f"{name} cannot be missing.")

    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"{name} cannot be missing.")

    return int(cast(Any, value))

def ensure_sequence(engine, *, schema: str, sequence_name: str) -> None:
    """Create the Postgres sequence used for synthetic batch or cycle IDs."""
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_sequence = sanitize_sql_identifier(sequence_name)
    sql = f'CREATE SEQUENCE IF NOT EXISTS "{safe_schema}"."{safe_sequence}"'
    execute_sql(engine, sql)



def reserve_next_batch_id(engine, *, schema: str, sequence_name: str) -> int:
    """Reserve and return the next synthetic batch identifier."""
    safe_schema = sanitize_sql_identifier(schema)
    safe_sequence = sanitize_sql_identifier(sequence_name)

    sql = f'SELECT nextval(\'"{safe_schema}"."{safe_sequence}"\') AS batch_id'

    dataframe = read_sql_dataframe(engine, sql)

    return scalar_to_int(
        dataframe.at[0, "batch_id"],
        "batch_id",
    )


def reserve_cycle_range(engine, *, schema: str, sequence_name: str, n_rows: int) -> int:
    """Reserve a contiguous global-cycle range and return its first value."""
    if n_rows <= 0:
        raise ValueError("n_rows must be > 0")

    safe_schema = sanitize_sql_identifier(schema)
    safe_sequence = sanitize_sql_identifier(sequence_name)

    start_dataframe = read_sql_dataframe(
        engine,
        f'SELECT nextval(\'"{safe_schema}"."{safe_sequence}"\') AS v',
    )

    start = scalar_to_int(
        start_dataframe.at[0, "v"],
        "sequence_start_value",
    )

    if n_rows > 1:
        execute_sql(
            engine,
            f'SELECT setval(\'"{safe_schema}"."{safe_sequence}"\', {start + (n_rows - 1)})',
        )

    return start


def reset_sequence(engine, *, schema: str, sequence_name: str, start_at: int = 1) -> None:
    """Reset a Postgres sequence so the next value starts at start_at."""
    # nextval returns start_at when is_called=false
    sql = f"SELECT setval('\"{schema}\".\"{sequence_name}\"', {int(start_at)}, false)"
    execute_sql(engine, sql)


def reset_synthetic_sequences(engine, *, schema: str, dataset_name: str) -> None:
    """Reset the synthetic batch and cycle sequences for one dataset."""
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
    """Read the current column names for a synthetic stream table."""
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
    """Infer a Postgres column type for dynamic synthetic stream columns."""
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE PRECISION"
    #if pd.api.types.is_datetime64tz_dtype(series):
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return "TIMESTAMPTZ"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    return "TEXT"



def _add_missing_columns(engine, *, schema: str, table: str, dataframe: pd.DataFrame) -> None:
    """Add dataframe columns that are not yet present in the stream table."""
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
# COPY helpers
# -----------------------------------------------------------------------------

def _prepare_dataframe_for_copy(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Sanitize columns and serialize nested values before COPY loading."""
    out = dataframe.copy()
    out.columns = [sanitize_sql_identifier(column) for column in out.columns]

    for column in out.columns:
        series = out[column]

        if isinstance(series.dtype, pd.CategoricalDtype):
            out[column] = series.astype(object)
            series = out[column]

        if pd.api.types.is_object_dtype(series):
            out[column] = series.map(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
            )

    return out



def _copy_dataframe_to_table(
    engine,
    dataframe: pd.DataFrame,
    *,
    schema: str,
    table: str,
) -> None:
    """Bulk-load a prepared dataframe into a Postgres table with COPY."""
    if dataframe.empty:
        return

    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table)
    out = _prepare_dataframe_for_copy(dataframe)

    column_list_sql = ", ".join(f'"{column}"' for column in out.columns)
    copy_sql = (
        f'COPY "{safe_schema}"."{safe_table}" ({column_list_sql}) '
        "FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
    )

    buffer = io.StringIO()
    out.to_csv(
        buffer,
        index=False,
        header=False,
        na_rep="\\N",
        quoting=csv.QUOTE_MINIMAL,
    )
    buffer.seek(0)

    raw_connection = engine.raw_connection()
    try:
        cursor = raw_connection.cursor()
        cursor.copy_expert(copy_sql, buffer)
        raw_connection.commit()
    except Exception:
        raw_connection.rollback()
        raise
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        raw_connection.close()


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
    use_copy: bool = True,
) -> str:
    """Write a synthetic stream batch to the dataset stream table.

    The helper writes to:
      synthetic_<dataset_name>_<artifact_name>

    Behavior:
      - ensures the base stream table exists
      - auto-adds any missing columns for this dataframe
      - uses COPY bulk load by default for faster inserts
      - falls back to the generic layer writer if COPY fails
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    out = dataframe.copy()
    row_count = len(out)

    out.insert(0, "batch_id", int(batch_id))
    out.insert(1, "row_in_batch", np.arange(row_count, dtype="int64"))

    if cycle_start is not None:
        out.insert(
            2,
            "global_cycle_id",
            np.arange(int(cycle_start), int(cycle_start) + row_count, dtype="int64"),
        )

    table = f"synthetic_{dataset_name}_{artifact_name}"

    _ensure_stream_table_exists(engine, schema=schema, table=table)
    _add_missing_columns(engine, schema=schema, table=table, dataframe=out)

    if use_copy:
        try:
            _copy_dataframe_to_table(
                engine,
                out,
                schema=schema,
                table=table,
            )
            print(
                f"[synthetic] COPY loaded {row_count:,} rows into {sanitize_sql_identifier(schema)}.{sanitize_sql_identifier(table)}"
            )
            return sanitize_sql_identifier(table)
        except Exception as exc:
            print(
                "[synthetic] COPY bulk load failed; falling back to pandas.to_sql. "
                f"Reason: {exc}"
            )

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
