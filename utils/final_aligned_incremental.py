from __future__ import annotations

import uuid
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
    read_table_for_observation_window,
)
from utils.final_aligned_observation_writer import (
    build_final_aligned_observations_dataframe,
    ensure_final_aligned_table_exists,
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

    print(f"[final-align-incremental] Added {len(missing)} new columns to {safe_schema}.{safe_table}")


def _remove_existing_target_rows(
    engine,
    *,
    dataframe: pd.DataFrame,
    schema: str,
    target_table: str,
) -> tuple[pd.DataFrame, int]:
    if dataframe.empty:
        return dataframe.copy(), 0

    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(target_table)

    existing_columns = _get_existing_columns(engine, schema=safe_schema, table=safe_table)
    if not existing_columns:
        return dataframe.copy(), 0

    existing = read_sql_dataframe(
        engine,
        f"""
        SELECT dataset_id, run_id, asset_id, observation_index
        FROM "{safe_schema}"."{safe_table}"
        """
    )

    if existing.empty:
        return dataframe.copy(), 0

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
            dataframe["dataset_id"].astype(str),
            dataframe["run_id"].astype(str),
            dataframe["asset_id"].astype(str),
            dataframe["observation_index"].astype(int),
        )
    )

    keep_mask = [key not in existing_keys for key in incoming_keys]
    skipped_existing_count = int(len(incoming_keys) - sum(keep_mask))

    return dataframe.loc[keep_mask].reset_index(drop=True), skipped_existing_count


# -----------------------------------------------------------------------------
# Runtime columns on rebuilt source
# -----------------------------------------------------------------------------

def ensure_rebuilt_final_align_runtime_columns(
    engine,
    *,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_observations_rebuilt_stage",
) -> None:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(source_table)

    statements = [
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS final_align_status TEXT;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS final_align_token TEXT;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS final_align_started_at TIMESTAMPTZ;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS final_align_completed_at TIMESTAMPTZ;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS final_align_error TEXT;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS final_align_attempt BIGINT;
        ''',
    ]

    for sql in statements:
        execute_sql(engine, sql)

    execute_sql(
        engine,
        f"""
        UPDATE "{safe_schema}"."{safe_table}"
        SET final_align_status = COALESCE(final_align_status, 'pending'),
            final_align_attempt = COALESCE(final_align_attempt, 0)
        WHERE final_align_status IS NULL
           OR final_align_attempt IS NULL
        """
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_final_align_status"
        ON "{safe_schema}"."{safe_table}" (final_align_status);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_final_align_obs"
        ON "{safe_schema}"."{safe_table}" (observation_index);
        '''
    )


# -----------------------------------------------------------------------------
# Claim / read / build
# -----------------------------------------------------------------------------

def claim_rebuilt_rows_for_final_align(
    engine,
    *,
    batch_size: int = 1000,
    schema: str = "capstone",
    rebuilt_table: str = "synthetic_sensor_observations_rebuilt_stage",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    complete_only: bool = True,
    final_align_token: Optional[str] = None,
) -> pd.DataFrame:
    ensure_rebuilt_final_align_runtime_columns(
        engine,
        schema=schema,
        source_table=rebuilt_table,
    )

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0")

    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(rebuilt_table)
    resolved_token = str(final_align_token).strip() if final_align_token else str(uuid.uuid4())

    filters = ["final_align_status = 'pending'"]
    params = {
        "row_limit": int(batch_size),
        "final_align_token": resolved_token,
    }

    if complete_only:
        filters.append("rebuild_is_complete = TRUE")
    if dataset_id is not None:
        filters.append("dataset_id = :dataset_id")
        params["dataset_id"] = str(dataset_id).strip()
    if run_id is not None:
        filters.append("run_id = :run_id")
        params["run_id"] = str(run_id).strip()

    where_sql = " AND ".join(filters)

    sql = f"""
    WITH next_rows AS (
        SELECT ctid
        FROM "{safe_schema}"."{safe_table}"
        WHERE {where_sql}
        ORDER BY observation_index
        LIMIT :row_limit
        FOR UPDATE SKIP LOCKED
    )
    UPDATE "{safe_schema}"."{safe_table}" AS src
    SET
        final_align_status = 'claimed',
        final_align_token = :final_align_token,
        final_align_started_at = now(),
        final_align_completed_at = NULL,
        final_align_error = NULL,
        final_align_attempt = COALESCE(final_align_attempt, 0) + 1
    FROM next_rows
    WHERE src.ctid = next_rows.ctid
    RETURNING src.*;
    """
    return read_sql_dataframe(engine, sql, params=params)


def load_premelt_for_claimed_final_align(
    engine,
    claimed_rows: pd.DataFrame,
    *,
    schema: str = "capstone",
    premelt_table: str = "synthetic_observations_premelt_stage",
) -> pd.DataFrame:
    if claimed_rows.empty:
        return pd.DataFrame()

    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(premelt_table)

    select_columns = get_table_columns(
        engine,
        schema_name=safe_schema,
        table_name=safe_table,
    )

    resolved_dataset_id = str(claimed_rows["dataset_id"].iloc[0]).strip()
    resolved_run_id = str(claimed_rows["run_id"].iloc[0]).strip()
    obs_min = int(claimed_rows["observation_index"].min())
    obs_max = int(claimed_rows["observation_index"].max())

    premelt_window = read_table_for_observation_window(
        engine,
        schema_name=safe_schema,
        table_name=safe_table,
        select_columns=select_columns,
        dataset_id=resolved_dataset_id,
        run_id=resolved_run_id,
        observation_index_min=obs_min,
        observation_index_max=obs_max,
        order_by_sql="observation_index",
    )

    if premelt_window.empty:
        return premelt_window

    claimed_keys = claimed_rows[
        ["dataset_id", "run_id", "asset_id", "observation_index"]
    ].drop_duplicates()

    return premelt_window.merge(
        claimed_keys.assign(_keep_flag=True),
        on=["dataset_id", "run_id", "asset_id", "observation_index"],
        how="inner",
    ).drop(columns=["_keep_flag"])


def build_claimed_final_aligned_rows(
    engine,
    claimed_rows: pd.DataFrame,
    *,
    schema: str = "capstone",
    premelt_table: str = "synthetic_observations_premelt_stage",
    n_sensors: int = 52,
    prefer_rebuilt_sensor_values: bool = True,
) -> pd.DataFrame:
    if claimed_rows.empty:
        return pd.DataFrame()

    premelt_dataframe = load_premelt_for_claimed_final_align(
        engine,
        claimed_rows,
        schema=schema,
        premelt_table=premelt_table,
    )

    if premelt_dataframe.empty:
        return pd.DataFrame()

    final_dataframe = build_final_aligned_observations_dataframe(
        premelt_dataframe=premelt_dataframe,
        rebuilt_dataframe=claimed_rows,
        n_sensors=n_sensors,
        prefer_rebuilt_sensor_values=prefer_rebuilt_sensor_values,
    )

    if final_dataframe.empty:
        return final_dataframe

    final_dataframe = final_dataframe.copy()
    final_dataframe["final_align_token"] = str(claimed_rows["final_align_token"].iloc[0])
    final_dataframe["final_align_received_at"] = pd.Timestamp.utcnow()

    return final_dataframe


def write_claimed_final_aligned_rows(
    engine,
    dataframe: pd.DataFrame,
    *,
    schema: str = "capstone",
    target_table: str = "synthetic_sensor_observations_final_aligned_stage",
) -> dict:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if dataframe.empty:
        return {
            "target_table": sanitize_sql_identifier(target_table),
            "written_count": 0,
            "skipped_existing_count": 0,
        }

    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_target = ensure_final_aligned_table_exists(
        engine,
        schema=schema,
        table_name=target_table,
    )

    working = dataframe.copy()
    working.columns = [sanitize_sql_identifier(column) for column in working.columns]

    working, skipped_existing_count = _remove_existing_target_rows(
        engine,
        dataframe=working,
        schema=safe_schema,
        target_table=safe_target,
    )

    if working.empty:
        return {
            "target_table": safe_target,
            "written_count": 0,
            "skipped_existing_count": skipped_existing_count,
        }

    _add_missing_columns(
        engine,
        schema=safe_schema,
        table=safe_target,
        dataframe=working,
    )

    write_layer_dataframe(
        engine=engine,
        dataframe=working,
        schema=safe_schema,
        table_name=safe_target,
        if_exists="append",
        index=False,
    )

    return {
        "target_table": safe_target,
        "written_count": int(len(working)),
        "skipped_existing_count": int(skipped_existing_count),
    }


# -----------------------------------------------------------------------------
# Status updates
# -----------------------------------------------------------------------------

def mark_claimed_final_align_completed(
    engine,
    *,
    final_align_token: str,
    schema: str = "capstone",
    rebuilt_table: str = "synthetic_sensor_observations_rebuilt_stage",
) -> None:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(rebuilt_table)

    sql = f"""
    UPDATE "{safe_schema}"."{safe_table}"
    SET
        final_align_status = 'completed',
        final_align_completed_at = now(),
        final_align_error = NULL
    WHERE final_align_token = :final_align_token
      AND final_align_status = 'claimed'
    """
    execute_sql(
        engine,
        sql,
        params={"final_align_token": str(final_align_token).strip()},
    )


def mark_claimed_final_align_failed(
    engine,
    *,
    final_align_token: str,
    error_message: str,
    schema: str = "capstone",
    rebuilt_table: str = "synthetic_sensor_observations_rebuilt_stage",
) -> None:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(rebuilt_table)

    sql = f"""
    UPDATE "{safe_schema}"."{safe_table}"
    SET
        final_align_status = 'failed',
        final_align_error = :error_message
    WHERE final_align_token = :final_align_token
      AND final_align_status = 'claimed'
    """
    execute_sql(
        engine,
        sql,
        params={
            "final_align_token": str(final_align_token).strip(),
            "error_message": str(error_message).strip()[:4000],
        },
    )


def requeue_failed_final_aligns(
    engine,
    *,
    schema: str = "capstone",
    rebuilt_table: str = "synthetic_sensor_observations_rebuilt_stage",
) -> int:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(rebuilt_table)

    sql = f"""
    UPDATE "{safe_schema}"."{safe_table}"
    SET
        final_align_status = 'pending',
        final_align_token = NULL,
        final_align_started_at = NULL,
        final_align_completed_at = NULL,
        final_align_error = NULL
    WHERE final_align_status = 'failed'
    """
    execute_sql(engine, sql)

    result = read_sql_dataframe(
        engine,
        f"""
        SELECT COUNT(*) AS row_count
        FROM "{safe_schema}"."{safe_table}"
        WHERE final_align_status = 'pending'
        """
    )
    return int(result.loc[0, "row_count"])


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def final_align_rebuilt_observations_to_stage(
    engine,
    *,
    batch_size: int = 1000,
    schema: str = "capstone",
    premelt_table: str = "synthetic_observations_premelt_stage",
    rebuilt_table: str = "synthetic_sensor_observations_rebuilt_stage",
    target_table: str = "synthetic_sensor_observations_final_aligned_stage",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    n_sensors: int = 52,
    complete_only: bool = True,
    prefer_rebuilt_sensor_values: bool = True,
) -> dict:
    claimed = claim_rebuilt_rows_for_final_align(
        engine=engine,
        batch_size=batch_size,
        schema=schema,
        rebuilt_table=rebuilt_table,
        dataset_id=dataset_id,
        run_id=run_id,
        complete_only=complete_only,
    )

    if claimed.empty:
        return {
            "status": "empty",
            "claimed_count": 0,
            "written_count": 0,
            "skipped_existing_count": 0,
            "target_table": sanitize_sql_identifier(target_table),
        }

    final_align_token = str(claimed["final_align_token"].iloc[0])

    try:
        final_dataframe = build_claimed_final_aligned_rows(
            engine=engine,
            claimed_rows=claimed,
            schema=schema,
            premelt_table=premelt_table,
            n_sensors=n_sensors,
            prefer_rebuilt_sensor_values=prefer_rebuilt_sensor_values,
        )

        write_result = write_claimed_final_aligned_rows(
            engine=engine,
            dataframe=final_dataframe,
            schema=schema,
            target_table=target_table,
        )

        mark_claimed_final_align_completed(
            engine=engine,
            final_align_token=final_align_token,
            schema=schema,
            rebuilt_table=rebuilt_table,
        )

        return {
            "status": "completed",
            "final_align_token": final_align_token,
            "claimed_count": int(len(claimed)),
            "written_count": int(write_result["written_count"]),
            "skipped_existing_count": int(write_result["skipped_existing_count"]),
            "target_table": write_result["target_table"],
        }

    except Exception as exc:
        mark_claimed_final_align_failed(
            engine=engine,
            final_align_token=final_align_token,
            error_message=str(exc),
            schema=schema,
            rebuilt_table=rebuilt_table,
        )
        return {
            "status": "failed",
            "final_align_token": final_align_token,
            "claimed_count": int(len(claimed)),
            "written_count": 0,
            "skipped_existing_count": 0,
            "target_table": sanitize_sql_identifier(target_table),
            "error": str(exc),
        }


def run_final_align_loop(
    engine,
    *,
    batch_size: int = 1000,
    schema: str = "capstone",
    premelt_table: str = "synthetic_observations_premelt_stage",
    rebuilt_table: str = "synthetic_sensor_observations_rebuilt_stage",
    target_table: str = "synthetic_sensor_observations_final_aligned_stage",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    n_sensors: int = 52,
    complete_only: bool = True,
    prefer_rebuilt_sensor_values: bool = True,
    max_iterations: Optional[int] = None,
    stop_on_failure: bool = True,
) -> list[dict]:
    results: list[dict] = []
    iteration = 0

    while True:
        if max_iterations is not None and iteration >= int(max_iterations):
            break

        result = final_align_rebuilt_observations_to_stage(
            engine=engine,
            batch_size=batch_size,
            schema=schema,
            premelt_table=premelt_table,
            rebuilt_table=rebuilt_table,
            target_table=target_table,
            dataset_id=dataset_id,
            run_id=run_id,
            n_sensors=n_sensors,
            complete_only=complete_only,
            prefer_rebuilt_sensor_values=prefer_rebuilt_sensor_values,
        )
        results.append(result)
        iteration += 1

        if result["status"] == "empty":
            break

        if result["status"] == "failed" and stop_on_failure:
            break

    return results


__all__ = [
    "ensure_rebuilt_final_align_runtime_columns",
    "claim_rebuilt_rows_for_final_align",
    "load_premelt_for_claimed_final_align",
    "build_claimed_final_aligned_rows",
    "write_claimed_final_aligned_rows",
    "mark_claimed_final_align_completed",
    "mark_claimed_final_align_failed",
    "requeue_failed_final_aligns",
    "final_align_rebuilt_observations_to_stage",
    "run_final_align_loop",
]