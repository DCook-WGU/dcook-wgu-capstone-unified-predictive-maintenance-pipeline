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

from utils.chunk_stage_util import (
    get_table_columns,
    process_observation_index_windows,
    resolve_dataset_run_from_table,
    read_table_for_observation_window,
)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

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

    print(f"[final-align] Added {len(missing)} new columns to {safe_schema}.{safe_table}")


def _build_sensor_columns(n_sensors: int = 52) -> list[str]:
    return [f"sensor_{i:02d}" for i in range(n_sensors)]


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _validate_premelt_columns(dataframe: pd.DataFrame, n_sensors: int) -> None:
    required_columns = [
        "dataset_id",
        "run_id",
        "asset_id",
        "generated_row_id",
        "observation_index",
        "batch_id",
        "row_in_batch",
        "global_cycle_id",
        "stream_state",
        "phase",
        "created_at",
        "meta_episode_id",
        "meta_primary_fault_type",
        "meta_magnitude",
    ] + _build_sensor_columns(n_sensors)

    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Premelt table is missing required columns: "
            + ", ".join(missing)
        )


def _validate_rebuilt_columns(dataframe: pd.DataFrame, n_sensors: int) -> None:
    required_columns = [
        "dataset_id",
        "run_id",
        "asset_id",
        "generated_row_id",
        "observation_index",
        "observation_timestamp",
        "stream_state",
        "phase",
        "meta_episode_id",
        "meta_primary_fault_type",
        "meta_magnitude",
        "rebuild_sensor_count",
        "rebuild_is_complete",
        "rebuild_completed_at",
        "rebuild_notes",
    ] + _build_sensor_columns(n_sensors)

    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Rebuilt table is missing required columns: "
            + ", ".join(missing)
        )


# -----------------------------------------------------------------------------
# Read helpers
# -----------------------------------------------------------------------------

def load_premelt_for_final_alignment(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_observations_premelt_stage",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    where_clauses = []
    params = {}

    if dataset_id is not None:
        where_clauses.append("dataset_id = :dataset_id")
        params["dataset_id"] = str(dataset_id).strip()

    if run_id is not None:
        where_clauses.append("run_id = :run_id")
        params["run_id"] = str(run_id).strip()

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


def load_rebuilt_for_final_alignment(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_observations_rebuilt_stage",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    complete_only: bool = True,
) -> pd.DataFrame:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    where_clauses = []
    params = {}

    if dataset_id is not None:
        where_clauses.append("dataset_id = :dataset_id")
        params["dataset_id"] = str(dataset_id).strip()

    if run_id is not None:
        where_clauses.append("run_id = :run_id")
        params["run_id"] = str(run_id).strip()

    if complete_only:
        where_clauses.append("rebuild_is_complete = TRUE")

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


# -----------------------------------------------------------------------------
# Final alignment builder
# -----------------------------------------------------------------------------

def build_final_aligned_observations_stage(
    engine,
    *,
    schema: str = "capstone",
    premelt_table: str = "synthetic_observations_premelt_stage",
    rebuilt_table: str = "synthetic_sensor_observations_rebuilt_stage",
    target_table: str = "synthetic_sensor_observations_final_aligned_stage",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    n_sensors: int = 52,
    complete_only: bool = True,
    prefer_rebuilt_sensor_values: bool = True,
    if_exists: str = "replace",
    observation_window_size: int = 2500,
) -> dict:
    safe_schema = sanitize_sql_identifier(schema)
    safe_premelt_table = sanitize_sql_identifier(premelt_table)
    safe_rebuilt_table = sanitize_sql_identifier(rebuilt_table)
    safe_target_table = sanitize_sql_identifier(target_table)

    resolved_dataset_id, resolved_run_id = resolve_dataset_run_from_table(
        engine,
        schema_name=safe_schema,
        table_name=safe_premelt_table,
        dataset_id=dataset_id,
        run_id=run_id,
    )

    premelt_columns = get_table_columns(
        engine,
        schema_name=safe_schema,
        table_name=safe_premelt_table,
    )
    rebuilt_columns = get_table_columns(
        engine,
        schema_name=safe_schema,
        table_name=safe_rebuilt_table,
    )

    has_written_first_chunk = False

    stats = {
        "status": "empty",
        "premelt_rows": 0,
        "rebuilt_rows": 0,
        "final_rows": 0,
        "target_table": safe_target_table,
    }

    rebuilt_extra_where_sql = " AND rebuild_is_complete = TRUE" if complete_only else ""

    def transform_chunk_func(df_premelt_window: pd.DataFrame, window_number: int, obs_min: int, obs_max: int) -> pd.DataFrame:
        rebuilt_window = read_table_for_observation_window(
            engine,
            schema_name=safe_schema,
            table_name=safe_rebuilt_table,
            select_columns=rebuilt_columns,
            dataset_id=resolved_dataset_id,
            run_id=resolved_run_id,
            observation_index_min=obs_min,
            observation_index_max=obs_max,
            extra_where_sql=rebuilt_extra_where_sql,
            order_by_sql="observation_index",
        )

        stats["premelt_rows"] += int(len(df_premelt_window))
        stats["rebuilt_rows"] += int(len(rebuilt_window))

        return build_final_aligned_observations_stage(
            premelt_dataframe=df_premelt_window,
            rebuilt_dataframe=rebuilt_window,
            n_sensors=n_sensors,
            prefer_rebuilt_sensor_values=prefer_rebuilt_sensor_values,
        )

    def write_chunk_func(df_out: pd.DataFrame, window_number: int, obs_min: int, obs_max: int) -> None:
        nonlocal has_written_first_chunk

        if df_out.empty:
            return

        written_table = write_final_aligned_observations(
            engine=engine,
            dataframe=df_out,
            schema=schema,
            table_name=target_table,
            if_exists=if_exists if not has_written_first_chunk else "append",
        )

        has_written_first_chunk = True
        stats["status"] = "built"
        stats["final_rows"] += int(len(df_out))
        stats["target_table"] = written_table

    process_observation_index_windows(
        engine,
        schema_name=safe_schema,
        table_name=safe_premelt_table,
        select_columns=premelt_columns,
        dataset_id=resolved_dataset_id,
        run_id=resolved_run_id,
        transform_chunk_func=transform_chunk_func,
        write_chunk_func=write_chunk_func,
        window_size=observation_window_size,
        order_by_sql="observation_index",
    )

    return stats


# -----------------------------------------------------------------------------
# Target table helpers
# -----------------------------------------------------------------------------

def ensure_final_aligned_table_exists(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_observations_final_aligned_stage",
) -> str:
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    CREATE TABLE IF NOT EXISTS "{safe_schema}"."{safe_table}" (
        dataset_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        observation_index BIGINT NOT NULL,
        batch_id BIGINT,
        row_in_batch INTEGER,
        global_cycle_id BIGINT,
        stream_state TEXT,
        phase TEXT,
        created_at TIMESTAMPTZ,
        meta_episode_id TEXT,
        meta_primary_fault_type TEXT,
        meta_magnitude DOUBLE PRECISION,
        generated_row_id TEXT,
        observation_timestamp TIMESTAMPTZ,
        rebuild_sensor_count INTEGER,
        rebuild_is_complete BOOLEAN,
        rebuild_completed_at TIMESTAMPTZ,
        rebuild_notes TEXT,
        PRIMARY KEY (dataset_id, run_id, asset_id, observation_index)
    );
    """
    execute_sql(engine, sql)

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_batch_row"
        ON "{safe_schema}"."{safe_table}" (batch_id, row_in_batch);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_observation_timestamp"
        ON "{safe_schema}"."{safe_table}" (observation_timestamp);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_rebuild_complete"
        ON "{safe_schema}"."{safe_table}" (rebuild_is_complete);
        '''
    )

    return safe_table


def write_final_aligned_observations(
    engine,
    dataframe: pd.DataFrame,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_observations_final_aligned_stage",
    if_exists: str = "replace",
) -> str:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if dataframe.empty:
        return sanitize_sql_identifier(table_name)

    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = ensure_final_aligned_table_exists(
        engine,
        schema=schema,
        table_name=table_name,
    )

    working = dataframe.copy()
    working.columns = [sanitize_sql_identifier(column) for column in working.columns]

    _add_missing_columns(
        engine,
        schema=safe_schema,
        table=safe_table,
        dataframe=working,
    )

    return write_layer_dataframe(
        engine=engine,
        dataframe=working,
        schema=safe_schema,
        table_name=safe_table,
        if_exists=if_exists,
        index=False,
    )


# -----------------------------------------------------------------------------
# Orchestration helper
# -----------------------------------------------------------------------------

def build_final_aligned_observations_stage(
    engine,
    *,
    schema: str = "capstone",
    premelt_table: str = "synthetic_observations_premelt_stage",
    rebuilt_table: str = "synthetic_sensor_observations_rebuilt_stage",
    target_table: str = "synthetic_sensor_observations_final_aligned_stage",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    n_sensors: int = 52,
    complete_only: bool = True,
    prefer_rebuilt_sensor_values: bool = True,
    if_exists: str = "replace",
    observation_window_size: int = 2500,
) -> dict:
    safe_schema = sanitize_sql_identifier(schema)
    safe_premelt_table = sanitize_sql_identifier(premelt_table)
    safe_rebuilt_table = sanitize_sql_identifier(rebuilt_table)
    safe_target_table = sanitize_sql_identifier(target_table)

    resolved_dataset_id, resolved_run_id = resolve_dataset_run_from_table(
        engine,
        schema_name=safe_schema,
        table_name=safe_premelt_table,
        dataset_id=dataset_id,
        run_id=run_id,
    )

    premelt_columns = get_table_columns(
        engine,
        schema_name=safe_schema,
        table_name=safe_premelt_table,
    )
    rebuilt_columns = get_table_columns(
        engine,
        schema_name=safe_schema,
        table_name=safe_rebuilt_table,
    )

    has_written_first_chunk = False

    stats = {
        "status": "empty",
        "premelt_rows": 0,
        "rebuilt_rows": 0,
        "final_rows": 0,
        "target_table": safe_target_table,
    }

    rebuilt_extra_where_sql = " AND rebuild_is_complete = TRUE" if complete_only else ""

    def transform_chunk_func(df_premelt_window: pd.DataFrame, window_number: int, obs_min: int, obs_max: int) -> pd.DataFrame:
        rebuilt_window = read_table_for_observation_window(
            engine,
            schema_name=safe_schema,
            table_name=safe_rebuilt_table,
            select_columns=rebuilt_columns,
            dataset_id=resolved_dataset_id,
            run_id=resolved_run_id,
            observation_index_min=obs_min,
            observation_index_max=obs_max,
            extra_where_sql=rebuilt_extra_where_sql,
            order_by_sql="observation_index",
        )

        stats["premelt_rows"] += int(len(df_premelt_window))
        stats["rebuilt_rows"] += int(len(rebuilt_window))

        return build_final_aligned_observations_stage(
            engine, 
            premelt_table=df_premelt_window,
            rebuilt_table=rebuilt_window,
            n_sensors=n_sensors,
            prefer_rebuilt_sensor_values=prefer_rebuilt_sensor_values,
        )

    def write_chunk_func(df_out: pd.DataFrame, window_number: int, obs_min: int, obs_max: int) -> None:
        nonlocal has_written_first_chunk

        if df_out.empty:
            return

        written_table = write_final_aligned_observations(
            engine=engine,
            dataframe=df_out,
            schema=schema,
            table_name=target_table,
            if_exists=if_exists if not has_written_first_chunk else "append",
        )

        has_written_first_chunk = True
        stats["status"] = "built"
        stats["final_rows"] += int(len(df_out))
        stats["target_table"] = written_table

    process_observation_index_windows(
        engine,
        schema_name=safe_schema,
        table_name=safe_premelt_table,
        select_columns=premelt_columns,
        dataset_id=resolved_dataset_id,
        run_id=resolved_run_id,
        transform_chunk_func=transform_chunk_func,
        write_chunk_func=write_chunk_func,
        window_size=observation_window_size,
        order_by_sql="observation_index",
    )

    return stats


__all__ = [
    "load_premelt_for_final_alignment",
    "load_rebuilt_for_final_alignment",
    "build_final_aligned_observations_dataframe",
    "ensure_final_aligned_table_exists",
    "write_final_aligned_observations",
    "build_final_aligned_observations_stage",
]