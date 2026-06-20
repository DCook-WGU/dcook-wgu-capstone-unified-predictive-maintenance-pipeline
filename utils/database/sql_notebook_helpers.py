"""Notebook-facing SQL helpers for capstone PostgreSQL metadata.

This module supports notebooks that need to write lightweight SQL metadata or
preview database tables without replacing the normal artifact outputs. It builds
one SQLAlchemy engine from the current environment at import time and resolves
``CAPSTONE_SCHEMA``, ``DATASET_ID``, and ``RUN_ID`` from environment variables
with notebook-global fallbacks.

The helpers write records to metadata tables such as ``pipeline_runs``,
``data_quality_events``, and ``pipeline_artifacts`` in the configured capstone
schema. They use print-style notebook status messages and do not write project
ledger entries directly.
"""

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

# Engine is created at import time; Postgres environment variables (DB_HOST, DB_USER, etc.)
# must be set before this module is imported, otherwise the import will raise RuntimeError.
engine = get_engine_from_env()

CAPSTONE_SCHEMA = os.getenv("CAPSTONE_SCHEMA", "capstone")

#DATASET_ID = os.getenv(
#    "SYNTHETIC_DATASET_ID",
#    globals().get("DATASET_NAME", "pump_synthetic_v1"),
#)
#RUN_ID = os.getenv(
#    "SYNTHETIC_RUN_ID",
#    globals().get("RUN_ID", "synthetic_run_001"),
#)

# globals().get("DATASET_NAME") reads the notebook-level DATASET_NAME variable when
# the DATASET_ID env var is not set, letting notebooks configure context before import.
DATASET_ID = os.getenv(
    "DATASET_ID",
    globals().get("DATASET_NAME", "pump"),
)
RUN_ID = os.getenv(
    "RUN_ID",
    globals().get("RUN_ID", "run_001"),
)


#print(f"SQL schema: {CAPSTONE_SCHEMA}")
#print(f"Dataset ID: {DATASET_ID}")
#print(f"Run ID: {RUN_ID}")


# -----------------------------------------------------------------------------
# Identifier safety
# -----------------------------------------------------------------------------

def safe_sql_identifier(value: str) -> str:
    """
    Validate a SQL identifier before using it in a schema/table reference.

    This prevents accidental unsafe SQL when table/schema names are built from
    notebook variables or environment variables.

    Parameters
    ----------
    value:
        Schema or table identifier to validate.

    Returns
    -------
    str
        Stripped identifier that is safe to interpolate into a quoted SQL table
        reference.

    Raises
    ------
    ValueError
        If the identifier contains characters outside the project-safe pattern.
    """
    value = str(value).strip()

    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value):
        raise ValueError(f"Unsafe SQL identifier: {value}")

    return value


def sql_table_ref(schema: str, table: str) -> str:
    """
    Return a safely quoted schema.table reference.

    Parameters
    ----------
    schema, table:
        SQL schema and table names to validate and quote.

    Returns
    -------
    str
        Double-quoted ``"schema"."table"`` reference.

    Raises
    ------
    ValueError
        If either identifier fails validation.
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

    Parameters
    ----------
    value:
        Python, pandas, or numpy value to normalize.

    Returns
    -------
    Any
        JSON-safe Python value, with missing scalar values converted to
        ``None`` and datetime-like values converted to ISO strings.
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

    # pd.isna() raises TypeError for arrays and some custom types; the bare except
    # is intentional to keep to_builtin safe for any value type found in a row.
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def to_scalar(value: Any) -> Any:
    """
    Convert values for normal SQL scalar columns.

    Parameters
    ----------
    value:
        Python, pandas, or numpy value to normalize for SQL binding.

    Returns
    -------
    Any
        Scalar value suitable for SQLAlchemy parameter binding, with missing
        values converted to ``None`` and pandas timestamps converted to Python
        datetimes.
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

    # Same bare except as to_builtin: pd.isna() may raise for non-scalar types.
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def to_json_string(value: Any) -> str:
    """
    Convert Python/pandas/numpy objects to a JSON string for JSONB columns.

    Parameters
    ----------
    value:
        Object to normalize and serialize for PostgreSQL JSONB columns.

    Returns
    -------
    str
        JSON string using project-safe builtin conversions.
    """
    return json.dumps(to_builtin(value), ensure_ascii=False, default=str)


def row_to_payload(row: pd.Series, exclude_columns: Optional[set[str]] = None) -> dict:
    """
    Convert a dataframe row into a JSON payload dictionary.

    Parameters
    ----------
    row:
        Dataframe row to serialize.
    exclude_columns:
        Optional set of columns to leave out of the payload.

    Returns
    -------
    dict
        Mapping from column name to JSON-safe value.
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

    Parameters
    ----------
    row:
        Dataframe row to inspect.
    candidate_columns:
        Ordered column names to check.
    default:
        Value returned when no candidate column contains a non-null value.

    Returns
    -------
    Any
        First non-null SQL scalar value found, or ``default``.
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

    Parameters
    ----------
    candidate_names:
        Ordered notebook variable names to inspect.

    Returns
    -------
    pandas.DataFrame
        Copy of the first matching dataframe.

    Raises
    ------
    NameError
        If none of the candidate names exist as pandas dataframes in the current
        notebook global scope.
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

    Parameters
    ----------
    sql:
        Parameterized SQL statement using named parameters.
    rows:
        Row dictionaries to bind into the SQL statement.
    chunk_size:
        Maximum number of rows sent per database execute call.

    Returns
    -------
    int
        Number of rows submitted to PostgreSQL.

    Side Effects
    ------------
    Executes SQL against the module-level engine. Prints notebook status
    messages and does not write ledger entries.
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

    Parameters
    ----------
    schema, table:
        Target table location to delete from.
    dataset_id, run_id:
        Dataset/run key used to identify rows from the current notebook run.

    Returns
    -------
    int
        Number of rows reported deleted by PostgreSQL.

    Side Effects
    ------------
    Deletes rows from the target SQL table through the module-level engine.

    Raises
    ------
    ValueError
        If schema or table identifiers are unsafe.
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

    Parameters
    ----------
    pipeline_stage:
        Notebook or pipeline stage name to store in metadata.
    run_status:
        Status value stored for the run. Defaults to ``completed``.
    pipeline_mode:
        Execution mode stored in metadata. Defaults to ``notebook``.
    dataset_name:
        Optional human-readable dataset name.
    source_system:
        Source system label stored in metadata.
    notes:
        Optional free-text notes for the run.
    runtime_facts:
        Optional structured metadata serialized to JSONB.

    Side Effects
    ------------
    Inserts or updates one row in ``CAPSTONE_SCHEMA.pipeline_runs`` using the
    module-level ``DATASET_ID`` and ``RUN_ID`` resolved from environment or
    notebook context. Prints notebook status messages and does not write ledger
    entries.
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

    Parameters
    ----------
    layer_name, table_name:
        Layer/table context for the data quality event.
    check_name, check_status:
        Name and status of the check being recorded.
    severity:
        Event severity label. Defaults to ``info``.
    row_count:
        Optional row count connected to the event.
    details_json:
        Optional structured details serialized to JSONB.

    Side Effects
    ------------
    Inserts one row into ``CAPSTONE_SCHEMA.data_quality_events`` using the
    module-level ``DATASET_ID`` and ``RUN_ID``. Prints notebook status messages
    and does not write ledger entries.
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

    Parameters
    ----------
    layer_name, stage_name:
        Layer and stage connected to the artifact.
    artifact_name, artifact_type:
        Artifact identity values stored in metadata.
    artifact_path:
        Optional path to an artifact created elsewhere by the notebook.
    truth_hash, parent_truth_hash:
        Optional lineage hashes associated with the artifact.
    metadata_json:
        Optional structured artifact metadata serialized to JSONB.

    Side Effects
    ------------
    Inserts one metadata row into ``CAPSTONE_SCHEMA.pipeline_artifacts`` using
    the module-level ``DATASET_ID`` and ``RUN_ID``. This function does not create
    or move artifact files and does not write ledger entries.
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

    Parameters
    ----------
    schema, table:
        SQL table to preview.
    limit:
        Maximum number of rows returned.

    Returns
    -------
    pandas.DataFrame
        Query result from the requested table.

    Raises
    ------
    ValueError
        If schema or table identifiers are unsafe.
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
