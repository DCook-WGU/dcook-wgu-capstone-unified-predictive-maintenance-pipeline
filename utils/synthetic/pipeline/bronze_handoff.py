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

    print(f"[bronze-handoff] Added {len(missing)} new columns to {safe_schema}.{safe_table}")


def _validate_handoff_mode(mode: str) -> str:
    resolved = str(mode).strip().lower()
    allowed = {"row", "row_batch", "full_batch"}
    if resolved not in allowed:
        raise ValueError(f"mode must be one of {sorted(allowed)}")
    return resolved


def _resolve_effective_batch_size(
    engine,
    *,
    mode: str,
    batch_size: int,
    schema: str,
    source_table: str,
    dataset_id: Optional[str],
    run_id: Optional[str],
    complete_only: bool,
) -> int:
    resolved_mode = _validate_handoff_mode(mode)

    if resolved_mode == "row":
        return 1

    if resolved_mode == "row_batch":
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0 for row_batch mode.")
        return int(batch_size)

    # full_batch
    count_sql = f"""
    SELECT COUNT(*) AS row_count
    FROM "{sanitize_sql_identifier(schema)}"."{sanitize_sql_identifier(source_table)}"
    WHERE bronze_handoff_status = 'pending'
      {"AND rebuild_is_complete = TRUE" if complete_only else ""}
      {"AND dataset_id = :dataset_id" if dataset_id is not None else ""}
      {"AND run_id = :run_id" if run_id is not None else ""}
    """
    params = {}
    if dataset_id is not None:
        params["dataset_id"] = str(dataset_id).strip()
    if run_id is not None:
        params["run_id"] = str(run_id).strip()

    count_df = read_sql_dataframe(engine, count_sql, params=params)
    if count_df.empty:
        return 0
    return int(count_df.loc[0, "row_count"])


# -----------------------------------------------------------------------------
# Source runtime columns
# -----------------------------------------------------------------------------

def ensure_final_aligned_runtime_columns(
    engine,
    *,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_observations_final_aligned_stage",
) -> None:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(source_table)

    statements = [
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS bronze_handoff_status TEXT;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS bronze_handoff_token TEXT;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS bronze_handoff_started_at TIMESTAMPTZ;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS bronze_handoff_completed_at TIMESTAMPTZ;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS bronze_handoff_error TEXT;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS bronze_handoff_mode TEXT;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS bronze_handoff_target_table TEXT;
        ''',
        f'''
        ALTER TABLE "{safe_schema}"."{safe_table}"
        ADD COLUMN IF NOT EXISTS bronze_handoff_attempt BIGINT;
        ''',
    ]

    for sql in statements:
        execute_sql(engine, sql)

    execute_sql(
        engine,
        f"""
        UPDATE "{safe_schema}"."{safe_table}"
        SET bronze_handoff_status = COALESCE(bronze_handoff_status, 'pending'),
            bronze_handoff_attempt = COALESCE(bronze_handoff_attempt, 0)
        WHERE bronze_handoff_status IS NULL
           OR bronze_handoff_attempt IS NULL
        """
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_bronze_status"
        ON "{safe_schema}"."{safe_table}" (bronze_handoff_status);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_bronze_order"
        ON "{safe_schema}"."{safe_table}" (batch_id, row_in_batch);
        '''
    )


# -----------------------------------------------------------------------------
# Target table helpers
# -----------------------------------------------------------------------------

def ensure_bronze_handoff_target_table_exists(
    engine,
    *,
    schema: str = "capstone",
    target_table: str = "bronze_observations_input_stage",
) -> str:
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = sanitize_sql_identifier(target_table)

    sql = f"""
    CREATE TABLE IF NOT EXISTS "{safe_schema}"."{safe_table}" (
        dataset_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        observation_index BIGINT NOT NULL,
        bronze_handoff_token TEXT,
        bronze_handoff_mode TEXT,
        bronze_handoff_received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (dataset_id, run_id, asset_id, observation_index)
    );
    """
    execute_sql(engine, sql)

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_bronze_received_at"
        ON "{safe_schema}"."{safe_table}" (bronze_handoff_received_at);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_bronze_token"
        ON "{safe_schema}"."{safe_table}" (bronze_handoff_token);
        '''
    )

    return safe_table


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
# Claim / write / update
# -----------------------------------------------------------------------------

def claim_final_aligned_rows_for_bronze(
    engine,
    *,
    mode: str = "row_batch",
    batch_size: int = 500,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_observations_final_aligned_stage",
    target_table: str = "bronze_observations_input_stage",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    complete_only: bool = True,
    handoff_token: Optional[str] = None,
) -> pd.DataFrame:
    ensure_final_aligned_runtime_columns(
        engine,
        schema=schema,
        source_table=source_table,
    )

    resolved_mode = _validate_handoff_mode(mode)
    resolved_batch_size = _resolve_effective_batch_size(
        engine,
        mode=resolved_mode,
        batch_size=batch_size,
        schema=schema,
        source_table=source_table,
        dataset_id=dataset_id,
        run_id=run_id,
        complete_only=complete_only,
    )

    if resolved_batch_size <= 0:
        return pd.DataFrame()

    safe_schema = sanitize_sql_identifier(schema)
    safe_source = sanitize_sql_identifier(source_table)
    resolved_target = sanitize_sql_identifier(target_table)
    resolved_token = str(handoff_token).strip() if handoff_token else str(uuid.uuid4())

    filters = ["bronze_handoff_status = 'pending'"]
    params = {
        "row_limit": int(resolved_batch_size),
        "handoff_token": resolved_token,
        "bronze_handoff_mode": resolved_mode,
        "bronze_handoff_target_table": resolved_target,
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
        FROM "{safe_schema}"."{safe_source}"
        WHERE {where_sql}
        ORDER BY batch_id, row_in_batch
        LIMIT :row_limit
        FOR UPDATE SKIP LOCKED
    )
    UPDATE "{safe_schema}"."{safe_source}" AS src
    SET
        bronze_handoff_status = 'claimed',
        bronze_handoff_token = :handoff_token,
        bronze_handoff_started_at = now(),
        bronze_handoff_mode = :bronze_handoff_mode,
        bronze_handoff_target_table = :bronze_handoff_target_table,
        bronze_handoff_error = NULL,
        bronze_handoff_attempt = COALESCE(bronze_handoff_attempt, 0) + 1
    FROM next_rows
    WHERE src.ctid = next_rows.ctid
    RETURNING src.*
    """
    return read_sql_dataframe(engine, sql, params=params)


def write_claimed_rows_to_bronze_target(
    engine,
    dataframe: pd.DataFrame,
    *,
    schema: str = "capstone",
    target_table: str = "bronze_observations_input_stage",
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
    safe_target = ensure_bronze_handoff_target_table_exists(
        engine,
        schema=schema,
        target_table=target_table,
    )

    working = dataframe.copy()
    working["bronze_handoff_received_at"] = pd.Timestamp.utcnow()
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


def mark_claimed_handoff_completed(
    engine,
    *,
    handoff_token: str,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_observations_final_aligned_stage",
) -> None:
    safe_schema = sanitize_sql_identifier(schema)
    safe_source = sanitize_sql_identifier(source_table)

    sql = f"""
    UPDATE "{safe_schema}"."{safe_source}"
    SET
        bronze_handoff_status = 'completed',
        bronze_handoff_completed_at = now(),
        bronze_handoff_error = NULL
    WHERE bronze_handoff_token = :handoff_token
      AND bronze_handoff_status = 'claimed'
    """
    execute_sql(
        engine,
        sql,
        params={"handoff_token": str(handoff_token).strip()},
    )


def mark_claimed_handoff_failed(
    engine,
    *,
    handoff_token: str,
    error_message: str,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_observations_final_aligned_stage",
) -> None:
    safe_schema = sanitize_sql_identifier(schema)
    safe_source = sanitize_sql_identifier(source_table)

    sql = f"""
    UPDATE "{safe_schema}"."{safe_source}"
    SET
        bronze_handoff_status = 'failed',
        bronze_handoff_error = :error_message
    WHERE bronze_handoff_token = :handoff_token
      AND bronze_handoff_status = 'claimed'
    """
    execute_sql(
        engine,
        sql,
        params={
            "handoff_token": str(handoff_token).strip(),
            "error_message": str(error_message).strip()[:4000],
        },
    )


def requeue_failed_bronze_handoffs(
    engine,
    *,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_observations_final_aligned_stage",
) -> int:
    safe_schema = sanitize_sql_identifier(schema)
    safe_source = sanitize_sql_identifier(source_table)

    sql = f"""
    UPDATE "{safe_schema}"."{safe_source}"
    SET
        bronze_handoff_status = 'pending',
        bronze_handoff_token = NULL,
        bronze_handoff_started_at = NULL,
        bronze_handoff_completed_at = NULL,
        bronze_handoff_error = NULL
    WHERE bronze_handoff_status = 'failed'
    """
    execute_sql(engine, sql)
    result = read_sql_dataframe(
        engine,
        f"""
        SELECT COUNT(*) AS row_count
        FROM "{safe_schema}"."{safe_source}"
        WHERE bronze_handoff_status = 'pending'
        """
    )
    return int(result.loc[0, "row_count"])


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def handoff_final_aligned_observations_to_bronze(
    engine,
    *,
    mode: str = "row_batch",
    batch_size: int = 500,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_observations_final_aligned_stage",
    target_table: str = "bronze_observations_input_stage",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    complete_only: bool = True,
) -> dict:
    claimed = claim_final_aligned_rows_for_bronze(
        engine=engine,
        mode=mode,
        batch_size=batch_size,
        schema=schema,
        source_table=source_table,
        target_table=target_table,
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

    handoff_token = str(claimed["bronze_handoff_token"].iloc[0])

    try:
        write_result = write_claimed_rows_to_bronze_target(
            engine=engine,
            dataframe=claimed,
            schema=schema,
            target_table=target_table,
        )

        mark_claimed_handoff_completed(
            engine=engine,
            handoff_token=handoff_token,
            schema=schema,
            source_table=source_table,
        )

        return {
            "status": "completed",
            "handoff_token": handoff_token,
            "claimed_count": int(len(claimed)),
            "written_count": int(write_result["written_count"]),
            "skipped_existing_count": int(write_result["skipped_existing_count"]),
            "target_table": write_result["target_table"],
        }

    except Exception as exc:
        mark_claimed_handoff_failed(
            engine=engine,
            handoff_token=handoff_token,
            error_message=str(exc),
            schema=schema,
            source_table=source_table,
        )
        return {
            "status": "failed",
            "handoff_token": handoff_token,
            "claimed_count": int(len(claimed)),
            "written_count": 0,
            "skipped_existing_count": 0,
            "target_table": sanitize_sql_identifier(target_table),
            "error": str(exc),
        }


def run_bronze_handoff_loop(
    engine,
    *,
    mode: str = "row_batch",
    batch_size: int = 500,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_observations_final_aligned_stage",
    target_table: str = "bronze_observations_input_stage",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    complete_only: bool = True,
    max_iterations: Optional[int] = None,
    stop_on_failure: bool = True,
) -> list[dict]:
    results: list[dict] = []
    iteration = 0

    while True:
        if max_iterations is not None and iteration >= int(max_iterations):
            break

        result = handoff_final_aligned_observations_to_bronze(
            engine=engine,
            mode=mode,
            batch_size=batch_size,
            schema=schema,
            source_table=source_table,
            target_table=target_table,
            dataset_id=dataset_id,
            run_id=run_id,
            complete_only=complete_only,
        )
        results.append(result)
        iteration += 1

        if result["status"] == "empty":
            break

        if result["status"] == "failed" and stop_on_failure:
            break

        if _validate_handoff_mode(mode) == "full_batch":
            break

    return results


__all__ = [
    "ensure_final_aligned_runtime_columns",
    "ensure_bronze_handoff_target_table_exists",
    "claim_final_aligned_rows_for_bronze",
    "write_claimed_rows_to_bronze_target",
    "mark_claimed_handoff_completed",
    "mark_claimed_handoff_failed",
    "requeue_failed_bronze_handoffs",
    "handoff_final_aligned_observations_to_bronze",
    "run_bronze_handoff_loop",
]