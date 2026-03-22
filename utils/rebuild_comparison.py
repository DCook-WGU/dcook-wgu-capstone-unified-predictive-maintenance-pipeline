from __future__ import annotations

from typing import List, Optional, Sequence

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
    desired = [sanitize_sql_identifier(column) for column in dataframe.columns]

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

    print(f"[compare] Added {len(missing)} new columns to {safe_schema}.{safe_table}")


def _build_sensor_columns(n_sensors: int = 52) -> list[str]:
    return [f"sensor_{i:02d}" for i in range(n_sensors)]


def _normalize_missing_scalar(value):
    if pd.isna(value):
        return None
    return value


def _compare_scalar(left, right, *, float_tolerance: float = 1e-9) -> bool:
    left = _normalize_missing_scalar(left)
    right = _normalize_missing_scalar(right)

    if left is None and right is None:
        return True
    if left is None or right is None:
        return False

    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= float(float_tolerance)

    return left == right


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
        "stream_state",
        "phase",
        "meta_episode_id",
        "meta_primary_fault_type",
        "meta_magnitude",
    ] + _build_sensor_columns(n_sensors)

    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Premelt comparison source is missing required columns: "
            + ", ".join(missing)
        )


def _validate_rebuilt_columns(dataframe: pd.DataFrame, n_sensors: int) -> None:
    required_columns = [
        "dataset_id",
        "run_id",
        "asset_id",
        "generated_row_id",
        "observation_index",
        "stream_state",
        "phase",
        "meta_episode_id",
        "meta_primary_fault_type",
        "meta_magnitude",
        "rebuild_sensor_count",
        "rebuild_is_complete",
    ] + _build_sensor_columns(n_sensors)

    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Rebuilt comparison source is missing required columns: "
            + ", ".join(missing)
        )


# -----------------------------------------------------------------------------
# Read helpers
# -----------------------------------------------------------------------------

def load_premelt_for_comparison(
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


def load_rebuilt_for_comparison(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_observations_rebuilt_stage",
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


# -----------------------------------------------------------------------------
# Comparison builder
# -----------------------------------------------------------------------------

def build_rebuild_comparison_dataframe(
    premelt_dataframe: pd.DataFrame,
    rebuilt_dataframe: pd.DataFrame,
    *,
    n_sensors: int = 52,
    float_tolerance: float = 1e-9,
) -> pd.DataFrame:
    if premelt_dataframe.empty and rebuilt_dataframe.empty:
        return pd.DataFrame()

    if not premelt_dataframe.empty:
        _validate_premelt_columns(premelt_dataframe, n_sensors=n_sensors)
    if not rebuilt_dataframe.empty:
        _validate_rebuilt_columns(rebuilt_dataframe, n_sensors=n_sensors)

    key_columns = ["dataset_id", "run_id", "asset_id", "observation_index"]

    original = premelt_dataframe.copy()
    rebuilt = rebuilt_dataframe.copy()

    original = original.rename(columns={column: f"original__{column}" for column in original.columns if column not in key_columns})
    rebuilt = rebuilt.rename(columns={column: f"rebuilt__{column}" for column in rebuilt.columns if column not in key_columns})

    comparison = original.merge(
        rebuilt,
        on=key_columns,
        how="outer",
        indicator=True,
    )

    comparison["exists_in_original"] = comparison["_merge"].isin(["both", "left_only"])
    comparison["exists_in_rebuilt"] = comparison["_merge"].isin(["both", "right_only"])

    metadata_columns = [
        "generated_row_id",
        "stream_state",
        "phase",
        "meta_episode_id",
        "meta_primary_fault_type",
        "meta_magnitude",
    ]

    for column in metadata_columns:
        comparison[f"match__{column}"] = comparison.apply(
            lambda row: _compare_scalar(
                row.get(f"original__{column}"),
                row.get(f"rebuilt__{column}"),
                float_tolerance=float_tolerance,
            ),
            axis=1,
        )

    sensor_columns = _build_sensor_columns(n_sensors=n_sensors)

    for column in sensor_columns:
        comparison[f"match__{column}"] = comparison.apply(
            lambda row: _compare_scalar(
                row.get(f"original__{column}"),
                row.get(f"rebuilt__{column}"),
                float_tolerance=float_tolerance,
            ),
            axis=1,
        )

    match_columns = [f"match__{column}" for column in metadata_columns + sensor_columns]

    comparison["comparison_mismatch_count"] = comparison[match_columns].apply(
        lambda row: int((~row.fillna(False)).sum()),
        axis=1,
    )

    comparison["comparison_all_fields_match"] = (
        comparison["exists_in_original"]
        & comparison["exists_in_rebuilt"]
        & (comparison["comparison_mismatch_count"] == 0)
    )

    comparison["comparison_notes"] = comparison.apply(
        lambda row: (
            "Missing from rebuilt table."
            if bool(row["exists_in_original"]) and not bool(row["exists_in_rebuilt"])
            else "Missing from original premelt table."
            if bool(row["exists_in_rebuilt"]) and not bool(row["exists_in_original"])
            else None
            if bool(row["comparison_all_fields_match"])
            else f"{int(row['comparison_mismatch_count'])} field mismatches found."
        ),
        axis=1,
    )

    ordered_columns = key_columns + [
        "exists_in_original",
        "exists_in_rebuilt",
        "comparison_mismatch_count",
        "comparison_all_fields_match",
        "comparison_notes",
        "rebuilt__rebuild_sensor_count",
        "rebuilt__rebuild_is_complete",
    ]

    remaining_columns = [column for column in comparison.columns if column not in ordered_columns and column != "_merge"]
    comparison = comparison[ordered_columns + remaining_columns]

    comparison = comparison.sort_values(
        by=["dataset_id", "run_id", "asset_id", "observation_index"],
        kind="stable",
    ).reset_index(drop=True)

    return comparison


# -----------------------------------------------------------------------------
# Target table helpers
# -----------------------------------------------------------------------------

def ensure_rebuild_comparison_table_exists(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_rebuild_comparison_stage",
) -> str:
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    CREATE TABLE IF NOT EXISTS "{safe_schema}"."{safe_table}" (
        dataset_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        observation_index BIGINT NOT NULL,
        exists_in_original BOOLEAN,
        exists_in_rebuilt BOOLEAN,
        comparison_mismatch_count INTEGER,
        comparison_all_fields_match BOOLEAN,
        comparison_notes TEXT,
        comparison_completed_at TIMESTAMPTZ DEFAULT now(),
        PRIMARY KEY (dataset_id, run_id, asset_id, observation_index)
    );
    """
    execute_sql(engine, sql)

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_all_match"
        ON "{safe_schema}"."{safe_table}" (comparison_all_fields_match);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_mismatch_count"
        ON "{safe_schema}"."{safe_table}" (comparison_mismatch_count);
        '''
    )

    return safe_table


def _remove_existing_comparison_rows(
    engine,
    *,
    comparison_dataframe: pd.DataFrame,
    schema: str,
    target_table: str,
) -> pd.DataFrame:
    if comparison_dataframe.empty:
        return comparison_dataframe.copy()

    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(target_table)

    if _get_existing_columns(engine, schema=safe_schema, table=safe_table) == set():
        return comparison_dataframe.copy()

    existing = read_sql_dataframe(
        engine,
        f"""
        SELECT dataset_id, run_id, asset_id, observation_index
        FROM "{safe_schema}"."{safe_table}"
        """
    )

    if existing.empty:
        return comparison_dataframe.copy()

    existing_keys = set(
        zip(
            existing["dataset_id"].astype(str),
            existing["run_id"].astype(str),
            existing["asset_id"].astype(str),
            existing["observation_index"].astype(int),
        )
    )

    incoming_keys = list(
        zip(
            comparison_dataframe["dataset_id"].astype(str),
            comparison_dataframe["run_id"].astype(str),
            comparison_dataframe["asset_id"].astype(str),
            comparison_dataframe["observation_index"].astype(int),
        )
    )

    keep_mask = [key not in existing_keys for key in incoming_keys]
    return comparison_dataframe.loc[keep_mask].reset_index(drop=True)


def write_rebuild_comparison_batch(
    engine,
    dataframe: pd.DataFrame,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_rebuild_comparison_stage",
) -> str:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if dataframe.empty:
        return sanitize_sql_identifier(table_name)

    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = ensure_rebuild_comparison_table_exists(
        engine,
        schema=schema,
        table_name=table_name,
    )

    working = dataframe.copy()
    working["comparison_completed_at"] = pd.Timestamp.utcnow()
    working.columns = [sanitize_sql_identifier(column) for column in working.columns]

    working = _remove_existing_comparison_rows(
        engine,
        comparison_dataframe=working,
        schema=safe_schema,
        target_table=safe_table,
    )

    if working.empty:
        return safe_table

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
        if_exists="append",
        index=False,
    )


# -----------------------------------------------------------------------------
# Orchestration helper
# -----------------------------------------------------------------------------

def build_rebuild_comparison_stage(
    engine,
    *,
    schema: str = "capstone",
    premelt_table: str = "synthetic_observations_premelt_stage",
    rebuilt_table: str = "synthetic_sensor_observations_rebuilt_stage",
    target_table: str = "synthetic_sensor_rebuild_comparison_stage",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    n_sensors: int = 52,
    float_tolerance: float = 1e-9,
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

    stats = {
        "status": "empty",
        "comparison_rows": 0,
        "target_table": safe_target_table,
    }

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
            order_by_sql="observation_index",
        )

        return build_rebuild_comparison_dataframe(
            premelt_dataframe=df_premelt_window,
            rebuilt_dataframe=rebuilt_window,
            n_sensors=n_sensors,
            float_tolerance=float_tolerance,
        )

    def write_chunk_func(df_out: pd.DataFrame, window_number: int, obs_min: int, obs_max: int) -> None:
        if df_out.empty:
            return

        write_rebuild_comparison_batch(
            engine=engine,
            dataframe=df_out,
            schema=schema,
            table_name=target_table,
        )

        stats["status"] = "built"
        stats["comparison_rows"] += int(len(df_out))

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
    "load_premelt_for_comparison",
    "load_rebuilt_for_comparison",
    "build_rebuild_comparison_dataframe",
    "ensure_rebuild_comparison_table_exists",
    "write_rebuild_comparison_batch",
    "compare_premelt_to_rebuilt_observations",
]