from __future__ import annotations

from typing import Optional
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
    df = read_sql_dataframe(engine, sql)
    return int(df.loc[0, "batch_id"])

def reserve_cycle_range(engine, *, schema: str, sequence_name: str, n_rows: int) -> int:
    """
    Reserve a contiguous cycle range by advancing a sequence by n_rows.
    Returns the starting cycle id for this batch.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be > 0")

    # nextval gives the start
    start_df = read_sql_dataframe(engine, f'SELECT nextval(\'"{schema}"."{sequence_name}"\') AS v')
    start = int(start_df.loc[0, "v"])

    # Advance the sequence by (n_rows-1) so the reserved block is contiguous
    if n_rows > 1:
        execute_sql(engine, f'SELECT setval(\'"{schema}"."{sequence_name}"\', {start + (n_rows - 1)})')

    return start

def write_stream_batch(
    engine,
    df: pd.DataFrame,
    *,
    dataset_name: str,
    schema: str = "public",
    artifact_name: str = "stream",
    batch_id: int,
    cycle_start: Optional[int] = None,
) -> str:
    """
    Writes a batch to the medallion-style table name:
      synthetic_<dataset_name>_<artifact_name>
    using your write_layer_dataframe wrapper. :contentReference[oaicite:2]{index=2}
    """
    out = df.copy()
    out.insert(0, "batch_id", int(batch_id))
    out.insert(1, "row_in_batch", range(len(out)))

    if cycle_start is not None:
        out.insert(2, "global_cycle_id", [int(cycle_start) + i for i in range(len(out))])

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