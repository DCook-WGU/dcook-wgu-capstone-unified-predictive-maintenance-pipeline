# =============================================================================
# SQL Setup Cell
# Purpose:
#   Provide shared Postgres helpers for writing notebook outputs to the
#   capstone database without replacing the notebook artifact pipeline.
# =============================================================================

import os
import re
import json
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from utils.database.postgres import get_engine_from_env, read_sql_dataframe


# -----------------------------------------------------------------------------
# Connection and environment context
# -----------------------------------------------------------------------------

engine = get_engine_from_env()

CAPSTONE_SCHEMA = os.getenv("CAPSTONE_SCHEMA", "capstone")
DATASET_ID = os.getenv(
    "SYNTHETIC_DATASET_ID",
    globals().get("DATASET_NAME", "pump_synthetic_v1"),
)
RUN_ID = os.getenv(
    "SYNTHETIC_RUN_ID",
    globals().get("RUN_ID", "synthetic_run_001"),
)

print(f"SQL schema: {CAPSTONE_SCHEMA}")
print(f"Dataset ID: {DATASET_ID}")
print(f"Run ID: {RUN_ID}")


# -----------------------------------------------------------------------------
# Identifier safety
# -----------------------------------------------------------------------------

def safe_sql_identifier(value: str) -> str:
    """
    Validate a SQL identifier before using it in a schema/table reference.

    This prevents accidental unsafe SQL when table/schema names are built from
    notebook variables or environment variables.
    """
    value = str(value).strip()

    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value):
        raise ValueError(f"Unsafe SQL identifier: {value}")

    return value


def sql_table_ref(schema: str, table: str) -> str:
    """
    Return a safely quoted schema.table reference.
    """
    safe_schema = safe_sql_identifier(schema)
    safe_table = safe_sql_identifier(table)

    return f'"{safe_schema}"."{safe_table}"'


# -----------------------------------------------------------------------------
# Value normalization helpers
# -----------------------------------------------------------------------------

def to_builtin(value: Any) -> Any:
    """
    Convert pandas/numpy values into JSON-safe Python values.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}

    if isinstance(value, list):
        return [to_builtin(v) for v in value]

    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]

    if isinstance(value, set):
        return [to_builtin(v) for v in sorted(value)]

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def to_scalar(value: Any) -> Any:
    """
    Convert values for normal SQL scalar columns.
    """
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime()

    if isinstance(value, datetime):
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def to_json_string(value: Any) -> str:
    """
    Convert Python/pandas/numpy objects to a JSON string for JSONB columns.
    """
    return json.dumps(to_builtin(value), ensure_ascii=False, default=str)


def row_to_payload(row: pd.Series, exclude_columns: Optional[set[str]] = None) -> dict:
    """
    Convert a dataframe row into a JSON payload dictionary.
    """
    exclude_columns = exclude_columns or set()

    return {
        str(column): to_builtin(row[column])
        for column in row.index
        if column not in exclude_columns
    }


def get_row_value(row: pd.Series, candidate_columns: list[str], default: Any = None) -> Any:
    """
    Return the first available non-null row value from a list of candidate columns.
    """
    for column in candidate_columns:
        if column in row.index:
            value = row[column]
            if to_scalar(value) is not None:
                return to_scalar(value)

    return default


def get_existing_dataframe(candidate_names: list[str]) -> pd.DataFrame:
    """
    Find the first dataframe in the current notebook globals using candidate names.
    """
    for name in candidate_names:
        value = globals().get(name)

        if isinstance(value, pd.DataFrame):
            print(f"Using dataframe: {name} -> {value.shape[0]:,} rows x {value.shape[1]:,} columns")
            return value.copy()

    raise NameError(
        "No dataframe found. Checked: "
        + ", ".join(candidate_names)
        + ". Update the candidate_names list for this notebook."
    )


# -----------------------------------------------------------------------------
# SQL execution helpers
# -----------------------------------------------------------------------------

def execute_many(sql: str, rows: list[dict], *, chunk_size: int = 5_000) -> int:
    """
    Execute a parameterized SQL statement for many rows in chunks.
    """
    if not rows:
        print("No rows to write.")
        return 0

    total_rows = len(rows)

    with engine.begin() as connection:
        for start_index in range(0, total_rows, chunk_size):
            chunk = rows[start_index : start_index + chunk_size]
            connection.execute(text(sql), chunk)

    print(f"Wrote {total_rows:,} rows.")
    return total_rows


def delete_dataset_run_rows(schema: str, table: str, *, dataset_id: str, run_id: str) -> int:
    """
    Delete existing rows for one dataset/run before writing notebook outputs.

    This keeps notebook SQL writes idempotent when rerunning a notebook.
    """
    table_reference = sql_table_ref(schema, table)

    sql = f"""
    DELETE FROM {table_reference}
    WHERE dataset_id = :dataset_id
      AND run_id = :run_id
    """

    with engine.begin() as connection:
        result = connection.execute(
            text(sql),
            {
                "dataset_id": dataset_id,
                "run_id": run_id,
            },
        )

    deleted_count = int(result.rowcount or 0)
    print(f"Deleted {deleted_count:,} existing rows from {schema}.{table}.")
    return deleted_count


def upsert_pipeline_run(
    *,
    pipeline_stage: str,
    run_status: str = "completed",
    pipeline_mode: str = "notebook",
    dataset_name: Optional[str] = None,
    source_system: str = "notebook",
    notes: Optional[str] = None,
    runtime_facts: Optional[dict] = None,
) -> None:
    """
    Upsert a capstone.pipeline_runs record for the current notebook/run.
    """
    table_reference = sql_table_ref(CAPSTONE_SCHEMA, "pipeline_runs")

    sql = f"""
    INSERT INTO {table_reference} (
        run_id,
        dataset_id,
        dataset_name,
        pipeline_stage,
        pipeline_mode,
        run_status,
        started_at_utc,
        completed_at_utc,
        source_system,
        notes,
        runtime_facts
    )
    VALUES (
        :run_id,
        :dataset_id,
        :dataset_name,
        :pipeline_stage,
        :pipeline_mode,
        :run_status,
        now(),
        now(),
        :source_system,
        :notes,
        CAST(:runtime_facts AS jsonb)
    )
    ON CONFLICT (run_id)
    DO UPDATE SET
        dataset_id = EXCLUDED.dataset_id,
        dataset_name = EXCLUDED.dataset_name,
        pipeline_stage = EXCLUDED.pipeline_stage,
        pipeline_mode = EXCLUDED.pipeline_mode,
        run_status = EXCLUDED.run_status,
        completed_at_utc = now(),
        source_system = EXCLUDED.source_system,
        notes = EXCLUDED.notes,
        runtime_facts = EXCLUDED.runtime_facts
    """

    with engine.begin() as connection:
        connection.execute(
            text(sql),
            {
                "run_id": RUN_ID,
                "dataset_id": DATASET_ID,
                "dataset_name": dataset_name,
                "pipeline_stage": pipeline_stage,
                "pipeline_mode": pipeline_mode,
                "run_status": run_status,
                "source_system": source_system,
                "notes": notes,
                "runtime_facts": to_json_string(runtime_facts or {}),
            },
        )

    print(f"Upserted pipeline run for stage: {pipeline_stage}")


def log_data_quality_event(
    *,
    layer_name: str,
    table_name: str,
    check_name: str,
    check_status: str,
    severity: str = "info",
    row_count: Optional[int] = None,
    details_json: Optional[dict] = None,
) -> None:
    """
    Insert a data quality event into capstone.data_quality_events.
    """
    table_reference = sql_table_ref(CAPSTONE_SCHEMA, "data_quality_events")

    sql = f"""
    INSERT INTO {table_reference} (
        run_id,
        dataset_id,
        layer_name,
        table_name,
        severity,
        check_name,
        check_status,
        row_count,
        details_json,
        created_at_utc
    )
    VALUES (
        :run_id,
        :dataset_id,
        :layer_name,
        :table_name,
        :severity,
        :check_name,
        :check_status,
        :row_count,
        CAST(:details_json AS jsonb),
        now()
    )
    """

    with engine.begin() as connection:
        connection.execute(
            text(sql),
            {
                "run_id": RUN_ID,
                "dataset_id": DATASET_ID,
                "layer_name": layer_name,
                "table_name": table_name,
                "severity": severity,
                "check_name": check_name,
                "check_status": check_status,
                "row_count": row_count,
                "details_json": to_json_string(details_json or {}),
            },
        )

    print(f"Logged DQ event: {layer_name}.{table_name} | {check_name} | {check_status}")


def log_pipeline_artifact(
    *,
    layer_name: str,
    stage_name: str,
    artifact_name: str,
    artifact_type: str,
    artifact_path: Optional[str] = None,
    truth_hash: Optional[str] = None,
    parent_truth_hash: Optional[str] = None,
    metadata_json: Optional[dict] = None,
) -> None:
    """
    Insert an artifact record into capstone.pipeline_artifacts.
    """
    table_reference = sql_table_ref(CAPSTONE_SCHEMA, "pipeline_artifacts")

    sql = f"""
    INSERT INTO {table_reference} (
        run_id,
        dataset_id,
        layer_name,
        stage_name,
        artifact_name,
        artifact_type,
        artifact_path,
        truth_hash,
        parent_truth_hash,
        created_at_utc,
        metadata_json
    )
    VALUES (
        :run_id,
        :dataset_id,
        :layer_name,
        :stage_name,
        :artifact_name,
        :artifact_type,
        :artifact_path,
        :truth_hash,
        :parent_truth_hash,
        now(),
        CAST(:metadata_json AS jsonb)
    )
    """

    with engine.begin() as connection:
        connection.execute(
            text(sql),
            {
                "run_id": RUN_ID,
                "dataset_id": DATASET_ID,
                "layer_name": layer_name,
                "stage_name": stage_name,
                "artifact_name": artifact_name,
                "artifact_type": artifact_type,
                "artifact_path": artifact_path,
                "truth_hash": truth_hash,
                "parent_truth_hash": parent_truth_hash,
                "metadata_json": to_json_string(metadata_json or {}),
            },
        )

    print(f"Logged artifact: {layer_name}.{stage_name} -> {artifact_name}")


def preview_sql_table(schema: str, table: str, limit: int = 5) -> pd.DataFrame:
    """
    Preview rows from a SQL table.
    """
    table_reference = sql_table_ref(schema, table)

    return read_sql_dataframe(
        engine,
        f"""
        SELECT *
        FROM {table_reference}
        LIMIT :limit
        """,
        params={"limit": int(limit)},
    )