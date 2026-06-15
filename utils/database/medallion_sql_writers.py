# =============================================================================
# Medallion SQL Writers
# File:
#   utils/database/medallion_sql_writers.py
#
# Purpose:
#   Provide reusable SQL writer functions for Bronze, Silver, and Gold notebook
#   outputs. These functions keep the notebooks clean while still using the
#   project database helpers and Postgres schema established by the bootstrap.
# =============================================================================

from __future__ import annotations

from typing import Any, Optional, Mapping

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from utils.database.postgres import read_sql_dataframe

from utils.database.layer_postgres import sanitize_sql_identifier

from utils.database.sql_notebook_helpers import (
    get_row_value,
    row_to_payload,
    sql_table_ref,
    to_builtin,
    to_json_string,
    to_scalar,
)


# =============================================================================
# Internal helpers
# =============================================================================

def _resolve_dataframe(
    *,
    dataframe: Optional[pd.DataFrame],
    candidate_names: list[str],
    notebook_globals: Optional[dict[str, Any]],
) -> pd.DataFrame:
    """
    Resolve the dataframe to write.

    Prefer an explicitly supplied dataframe. If none is supplied, search the
    notebook globals for the first matching dataframe name.
    """
    if dataframe is not None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        print(f"Using supplied dataframe -> {dataframe.shape[0]:,} rows x {dataframe.shape[1]:,} columns")
        return dataframe.copy()

    if notebook_globals is None:
        raise ValueError(
            "Either dataframe or notebook_globals must be provided."
        )

    for name in candidate_names:
        value = notebook_globals.get(name)

        if isinstance(value, pd.DataFrame):
            print(f"Using dataframe: {name} -> {value.shape[0]:,} rows x {value.shape[1]:,} columns")
            return value.copy()

    raise NameError(
        "No dataframe found. Checked: "
        + ", ".join(candidate_names)
        + ". Pass dataframe=... directly or update candidate_names."
    )


def _execute_many(
    engine: Engine,
    sql: str,
    rows: list[dict[str, Any]],
    *,
    chunk_size: int = 5_000,
) -> int:
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


def _delete_dataset_run_rows(
    engine: Engine,
    *,
    schema: str,
    table: str,
    dataset_id: str,
    run_id: str,
) -> int:
    """
    Delete existing rows for one dataset/run before writing notebook outputs.
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


def _delete_model_score_rows(
    engine: Engine,
    *,
    schema: str,
    table: str,
    dataset_id: str,
    run_id: str,
    model_name: str,
    model_stage: str,
) -> int:
    """
    Delete existing score rows for one dataset/run/model/stage.
    """
    table_reference = sql_table_ref(schema, table)

    sql = f"""
    DELETE FROM {table_reference}
    WHERE dataset_id = :dataset_id
      AND run_id = :run_id
      AND model_name = :model_name
      AND model_stage = :model_stage
    """

    with engine.begin() as connection:
        result = connection.execute(
            text(sql),
            {
                "dataset_id": dataset_id,
                "run_id": run_id,
                "model_name": model_name,
                "model_stage": model_stage,
            },
        )

    deleted_count = int(result.rowcount or 0)
    print(f"Deleted {deleted_count:,} existing rows from {schema}.{table} for {model_name}/{model_stage}.")
    return deleted_count


def _stage_pipeline_run_id(run_id: str, pipeline_stage: str) -> str:
    """
    Build a stage-specific pipeline_runs.run_id.

    The bootstrap table currently has run_id as the primary key. If every
    notebook used the same RUN_ID, later notebooks would overwrite earlier
    pipeline_runs records. This keeps the row-level run_id unchanged in layer
    tables while allowing one metadata row per notebook/stage.
    """
    clean_stage = str(pipeline_stage).strip().replace(" ", "_")
    return f"{run_id}__{clean_stage}"


def _upsert_pipeline_run(
    engine: Engine,
    *,
    capstone_schema: str,
    dataset_id: str,
    run_id: str,
    pipeline_stage: str,
    dataset_name: Optional[str] = None,
    pipeline_mode: str = "notebook",
    run_status: str = "completed",
    source_system: str = "notebook",
    notes: Optional[str] = None,
    runtime_facts: Optional[dict[str, Any]] = None,
) -> None:
    """
    Upsert one capstone.pipeline_runs row for a notebook/stage.
    """
    table_reference = sql_table_ref(capstone_schema, "pipeline_runs")
    pipeline_run_id = _stage_pipeline_run_id(run_id, pipeline_stage)

    runtime_payload = dict(runtime_facts or {})
    runtime_payload["source_run_id"] = run_id
    runtime_payload["pipeline_run_id"] = pipeline_run_id

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
        :pipeline_run_id,
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
                "pipeline_run_id": pipeline_run_id,
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "pipeline_stage": pipeline_stage,
                "pipeline_mode": pipeline_mode,
                "run_status": run_status,
                "source_system": source_system,
                "notes": notes,
                "runtime_facts": to_json_string(runtime_payload),
            },
        )

    print(f"Upserted pipeline run metadata: {pipeline_run_id}")


def _log_data_quality_event(
    engine: Engine,
    *,
    capstone_schema: str,
    dataset_id: str,
    run_id: str,
    layer_name: str,
    table_name: str,
    check_name: str,
    check_status: str,
    severity: str = "info",
    row_count: Optional[int] = None,
    details_json: Optional[dict[str, Any]] = None,
) -> None:
    """
    Insert one capstone.data_quality_events row.
    """
    table_reference = sql_table_ref(capstone_schema, "data_quality_events")

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
                "run_id": run_id,
                "dataset_id": dataset_id,
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


def _log_pipeline_artifact(
    engine: Engine,
    *,
    capstone_schema: str,
    dataset_id: str,
    run_id: str,
    layer_name: str,
    stage_name: str,
    artifact_name: str,
    artifact_type: str,
    artifact_path: Optional[str] = None,
    truth_hash: Optional[str] = None,
    parent_truth_hash: Optional[str] = None,
    metadata_json: Optional[dict[str, Any]] = None,
) -> None:
    """
    Insert one capstone.pipeline_artifacts row.
    """
    table_reference = sql_table_ref(capstone_schema, "pipeline_artifacts")

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
                "run_id": run_id,
                "dataset_id": dataset_id,
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


# =============================================================================
# Bronze writer
# =============================================================================

def write_bronze_sensor_observations_sql(
    engine,
    *,
    capstone_schema: str,
    layer_schema: str = "bronze",
    dataset_id: str,
    run_id: str,
    notebook_globals: Optional[dict[str, Any]] = None,
    dataframe: Optional[pd.DataFrame] = None,
    dataset_name: Optional[str] = None,
    candidate_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Write final Bronze dataframe rows to bronze.sensor_observations.
    """
    schema = sanitize_sql_identifier(layer_schema)
    metadata_schema = sanitize_sql_identifier(capstone_schema)

    table = "sensor_observations"

    candidates = candidate_names or [
        "bronze_dataframe",
        "bronze_df",
        "bronze_processed_dataframe",
        "bronze_output_dataframe",
        "final_bronze_dataframe",
        "output_dataframe",
    ]

    source_dataframe = _resolve_dataframe(
        dataframe=dataframe,
        candidate_names=candidates,
        notebook_globals=notebook_globals,
    ).reset_index(drop=False)

    _delete_dataset_run_rows(
        engine,
        schema=schema,
        table=table,
        dataset_id=dataset_id,
        run_id=run_id,
    )

    rows: list[dict[str, Any]] = []

    for _, row in source_dataframe.iterrows():
        source_row_id = get_row_value(
            row,
            [
                "meta__record_id",
                "meta__source_row_id",
                "source_row_id",
                "index",
            ],
        )

        rows.append(
            {
                "dataset_id": dataset_id,
                "run_id": run_id,
                "asset_id": get_row_value(row, ["meta__asset_id", "asset_id"], default=None),
                "event_time": get_row_value(row, ["event_time", "timestamp", "datetime", "date_time"]),
                "event_step": get_row_value(row, ["event_step", "step", "cycle", "index"]),
                "time_index": get_row_value(row, ["time_index", "event_step", "step", "cycle", "index"]),
                "source_table": get_row_value(row, ["meta__source_file", "source_table"], default=None),
                "source_row_id": None if source_row_id is None else str(source_row_id),
                "raw_payload": to_json_string(row_to_payload(row)),
                "meta_truth_hash": get_row_value(row, ["meta__truth_hash"], default=None),
                "meta_parent_truth_hash": get_row_value(row, ["meta__parent_truth_hash"], default=None),
            }
        )

    insert_sql = f"""
    INSERT INTO {sql_table_ref(schema, table)} (
        dataset_id,
        run_id,
        asset_id,
        event_time,
        event_step,
        time_index,
        source_table,
        source_row_id,
        raw_payload,
        meta_truth_hash,
        meta_parent_truth_hash,
        meta_ingested_at_utc
    )
    VALUES (
        :dataset_id,
        :run_id,
        :asset_id,
        :event_time,
        :event_step,
        :time_index,
        :source_table,
        :source_row_id,
        CAST(:raw_payload AS jsonb),
        :meta_truth_hash,
        :meta_parent_truth_hash,
        now()
    )
    """

    _execute_many(engine, insert_sql, rows)

    _upsert_pipeline_run(
        engine,
        capstone_schema=metadata_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        pipeline_stage="bronze_preprocessing",
        dataset_name=dataset_name or dataset_id,
        runtime_facts={
            "bronze_sql_table": f"{schema}.{table}",
            "row_count": len(source_dataframe),
            "column_count": len(source_dataframe.columns),
        },
    )

    _log_data_quality_event(
        engine,
        capstone_schema=metadata_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        layer_name=schema,
        table_name=table,
        check_name="bronze_sql_write",
        check_status="pass",
        row_count=len(source_dataframe),
        details_json={
            "sql_schema": schema,
            "sql_table": table,
        },
    )

    return read_sql_dataframe(
        engine,
        f"""
        SELECT dataset_id, run_id, COUNT(*) AS row_count
        FROM {sql_table_ref(schema, table)}
        WHERE dataset_id = :dataset_id
          AND run_id = :run_id
        GROUP BY dataset_id, run_id
        """,
        params={
            "dataset_id": dataset_id,
            "run_id": run_id,
        },
    )


# =============================================================================
# Silver writer
# =============================================================================

def write_silver_sensor_observation_features_sql(
    *,
    engine: Engine,
    capstone_schema: str,
    dataset_id: str,
    run_id: str,
    notebook_globals: Optional[dict[str, Any]] = None,
    dataframe: Optional[pd.DataFrame] = None,
    dataset_name: Optional[str] = None,
    candidate_names: Optional[list[str]] = None,
    feature_set_id: str = "silver_feature_set_v1",
) -> pd.DataFrame:
    """
    Write final Silver dataframe rows to silver.sensor_observation_features.
    """
    schema = "silver"
    table = "sensor_observation_features"

    candidates = candidate_names or [
        "silver_dataframe",
        "silver_df",
        "silver_processed_dataframe",
        "silver_output_dataframe",
        "cleaned_dataframe",
        "cleaned_df",
        "analysis_dataframe",
    ]

    source_dataframe = _resolve_dataframe(
        dataframe=dataframe,
        candidate_names=candidates,
        notebook_globals=notebook_globals,
    ).reset_index(drop=False)

    _delete_dataset_run_rows(
        engine,
        schema=schema,
        table=table,
        dataset_id=dataset_id,
        run_id=run_id,
    )

    canonical_columns = {
        "dataset_id",
        "run_id",
        "asset_id",
        "event_time",
        "event_step",
        "time_index",
        "index",
        "meta__asset_id",
        "meta__truth_hash",
        "meta__parent_truth_hash",
        "meta__feature_set_id",
    }

    feature_columns = [
        column
        for column in source_dataframe.columns
        if column not in canonical_columns
        and not str(column).startswith("meta__")
    ]

    quality_columns = [
        column
        for column in source_dataframe.columns
        if str(column).startswith("meta__")
        or "quality" in str(column).lower()
        or "missing" in str(column).lower()
        or "null" in str(column).lower()
        or "outlier" in str(column).lower()
    ]

    rows: list[dict[str, Any]] = []

    for _, row in source_dataframe.iterrows():
        features_json = {
            column: to_builtin(row[column])
            for column in feature_columns
        }

        quality_json = {
            column: to_builtin(row[column])
            for column in quality_columns
            if column in row.index
        }

        rows.append(
            {
                "dataset_id": dataset_id,
                "run_id": run_id,
                "asset_id": get_row_value(row, ["meta__asset_id", "asset_id"], default=None),
                "event_time": get_row_value(row, ["event_time", "timestamp", "datetime", "date_time"]),
                "event_step": get_row_value(row, ["event_step", "step", "cycle", "index"]),
                "time_index": get_row_value(row, ["time_index", "event_step", "step", "cycle", "index"]),
                "feature_set_id": get_row_value(
                    row,
                    ["meta__feature_set_id", "feature_set_id"],
                    default=feature_set_id,
                ),
                "features_json": to_json_string(features_json),
                "quality_json": to_json_string(quality_json),
                "meta_truth_hash": get_row_value(row, ["meta__truth_hash"], default=None),
                "meta_parent_truth_hash": get_row_value(row, ["meta__parent_truth_hash"], default=None),
            }
        )

    insert_sql = f"""
    INSERT INTO {sql_table_ref(schema, table)} (
        dataset_id,
        run_id,
        asset_id,
        event_time,
        event_step,
        time_index,
        feature_set_id,
        features_json,
        quality_json,
        meta_truth_hash,
        meta_parent_truth_hash,
        meta_processed_at_utc
    )
    VALUES (
        :dataset_id,
        :run_id,
        :asset_id,
        :event_time,
        :event_step,
        :time_index,
        :feature_set_id,
        CAST(:features_json AS jsonb),
        CAST(:quality_json AS jsonb),
        :meta_truth_hash,
        :meta_parent_truth_hash,
        now()
    )
    """

    _execute_many(engine, insert_sql, rows)

    _upsert_pipeline_run(
        engine,
        capstone_schema=capstone_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        pipeline_stage="silver_preprocessing",
        dataset_name=dataset_name or dataset_id,
        runtime_facts={
            "silver_sql_table": f"{schema}.{table}",
            "row_count": len(source_dataframe),
            "column_count": len(source_dataframe.columns),
            "feature_column_count": len(feature_columns),
            "quality_column_count": len(quality_columns),
        },
    )

    _log_data_quality_event(
        engine,
        capstone_schema=capstone_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        layer_name=schema,
        table_name=table,
        check_name="silver_sql_write",
        check_status="pass",
        row_count=len(source_dataframe),
        details_json={
            "feature_column_count": len(feature_columns),
            "quality_column_count": len(quality_columns),
        },
    )

    return read_sql_dataframe(
        engine,
        f"""
        SELECT dataset_id, run_id, feature_set_id, COUNT(*) AS row_count
        FROM {sql_table_ref(schema, table)}
        WHERE dataset_id = :dataset_id
          AND run_id = :run_id
        GROUP BY dataset_id, run_id, feature_set_id
        ORDER BY feature_set_id
        """,
        params={
            "dataset_id": dataset_id,
            "run_id": run_id,
        },
    )


def log_silver_eda_sql(
    *,
    engine: Engine,
    capstone_schema: str,
    dataset_id: str,
    run_id: str,
    notebook_globals: Optional[dict[str, Any]] = None,
    dataframe: Optional[pd.DataFrame] = None,
    dataset_name: Optional[str] = None,
    candidate_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Log Silver EDA profile metadata to capstone metadata tables.
    """
    candidates = candidate_names or [
        "silver_eda_dataframe",
        "silver_dataframe",
        "silver_df",
        "analysis_dataframe",
        "eda_dataframe",
        "cleaned_dataframe",
        "dataframe",
        "silver_subset_df",
    ]

    source_dataframe = _resolve_dataframe(
        dataframe=dataframe,
        candidate_names=candidates,
        notebook_globals=notebook_globals,
    )

    profile = {
        "row_count": int(len(source_dataframe)),
        "column_count": int(len(source_dataframe.columns)),
        "missing_cell_count": int(source_dataframe.isna().sum().sum()),
        "duplicate_row_count": int(source_dataframe.duplicated().sum()),
        "numeric_column_count": int(len(source_dataframe.select_dtypes(include="number").columns)),
        "non_numeric_column_count": int(len(source_dataframe.select_dtypes(exclude="number").columns)),
        "columns": [str(column) for column in source_dataframe.columns],
    }

    _upsert_pipeline_run(
        engine,
        capstone_schema=capstone_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        pipeline_stage="silver_eda",
        dataset_name=dataset_name or dataset_id,
        runtime_facts=profile,
        notes="Silver EDA profile logged from notebook.",
    )

    _log_data_quality_event(
        engine,
        capstone_schema=capstone_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        layer_name="silver",
        table_name="silver_eda_notebook",
        check_name="silver_eda_profile",
        check_status="pass",
        row_count=len(source_dataframe),
        details_json=profile,
    )

    artifact_variables = [
        "silver_eda_summary_path",
        "silver_profile_path",
        "silver_correlation_path",
        "silver_missingness_path",
        "silver_eda_ledger_path",
        "eda_summary_path",
        "eda_profile_path",
        "correlation_output_path",
        "missingness_output_path",
        "ledger_output_path",
    ]

    if notebook_globals is not None:
        for variable_name in artifact_variables:
            artifact_path = notebook_globals.get(variable_name)

            if artifact_path is not None:
                _log_pipeline_artifact(
                    engine,
                    capstone_schema=capstone_schema,
                    dataset_id=dataset_id,
                    run_id=run_id,
                    layer_name="silver",
                    stage_name="silver_eda",
                    artifact_name=variable_name,
                    artifact_type="eda_artifact",
                    artifact_path=str(artifact_path),
                    metadata_json={"source_variable": variable_name},
                )

    return read_sql_dataframe(
        engine,
        f"""
        SELECT
            layer_name,
            table_name,
            check_name,
            check_status,
            row_count,
            created_at_utc
        FROM {sql_table_ref(capstone_schema, "data_quality_events")}
        WHERE dataset_id = :dataset_id
          AND run_id = :run_id
          AND layer_name = 'silver'
        ORDER BY created_at_utc DESC
        LIMIT 10
        """,
        params={
            "dataset_id": dataset_id,
            "run_id": run_id,
        },
    )




SILVER_EDA_TABLES = {
    "profile_df": "eda_dataset_profile",
    "feature_statistics_df": "eda_feature_statistics",
    "missingness_summary_df": "eda_missingness_summary",
    "correlation_pairs_df": "eda_correlation_pairs",
    "outlier_summary_df": "eda_outlier_summary",
    "categorical_distribution_df": "eda_categorical_distribution",
    "artifact_index_df": "eda_artifact_index",
}


def write_silver_eda_sql_outputs(
    engine: Engine,
    dataset_id: str,
    run_id: str,
    notebook_name: str,
    profile_df: pd.DataFrame | None = None,
    feature_statistics_df: pd.DataFrame | None = None,
    missingness_summary_df: pd.DataFrame | None = None,
    correlation_pairs_df: pd.DataFrame | None = None,
    outlier_summary_df: pd.DataFrame | None = None,
    categorical_distribution_df: pd.DataFrame | None = None,
    artifact_index_df: pd.DataFrame | None = None,
    schema: str = "silver",
) -> dict[str, int]:
    """
    Write Silver 02b EDA summary outputs to PostgreSQL.

    This function writes durable EDA summary tables created by the Silver 02b
    notebook. It is intended for reproducibility and SQL-based inspection of
    the exploratory analysis layer. It does not write every temporary notebook
    dataframe. Instead, it writes profile, feature-statistic, missingness,
    correlation, outlier, categorical-distribution, and artifact-index summaries.

    Existing rows for the same dataset/run/notebook are deleted before the new
    rows are inserted so the notebook can be rerun without accumulating duplicate
    summary records.

    Parameters
    ----------
    engine:
        SQLAlchemy engine connected to the capstone PostgreSQL database.
    dataset_id:
        Dataset identifier for the current run.
    run_id:
        Run identifier for the current pipeline execution.
    notebook_name:
        Name of the notebook producing the EDA summaries.
    profile_df:
        Dataset/profile-level summary dataframe.
    feature_statistics_df:
        Numeric feature statistics dataframe.
    missingness_summary_df:
        Missingness summary dataframe.
    correlation_pairs_df:
        Long-form correlation pair dataframe.
    outlier_summary_df:
        Outlier summary dataframe.
    categorical_distribution_df:
        Categorical/status distribution dataframe.
    artifact_index_df:
        EDA artifact index dataframe.
    schema:
        Target database schema. Defaults to ``silver``.

    Returns
    -------
    dict[str, int]
        Mapping from SQL table name to number of rows written.

    Side Effects
    ------------
    Deletes existing rows for the same dataset/run/notebook from the target
    Silver EDA tables and inserts the provided summary records.
    """
    frames: Mapping[str, pd.DataFrame | None] = {
        "eda_dataset_profile": profile_df,
        "eda_feature_statistics": feature_statistics_df,
        "eda_missingness_summary": missingness_summary_df,
        "eda_correlation_pairs": correlation_pairs_df,
        "eda_outlier_summary": outlier_summary_df,
        "eda_categorical_distribution": categorical_distribution_df,
        "eda_artifact_index": artifact_index_df,
    }

    rows_written: dict[str, int] = {}

    with engine.begin() as conn:
        for table_name, dataframe in frames.items():
            if dataframe is None:
                rows_written[table_name] = 0
                continue

            if dataframe.empty:
                rows_written[table_name] = 0
                continue

            conn.execute(
                text(
                    f"""
                    DELETE FROM {schema}.{table_name}
                    WHERE dataset_id = :dataset_id
                      AND run_id = :run_id
                      AND notebook_name = :notebook_name
                    """
                ),
                {
                    "dataset_id": dataset_id,
                    "run_id": run_id,
                    "notebook_name": notebook_name,
                },
            )

            dataframe.to_sql(
                table_name,
                conn,
                schema=schema,
                if_exists="append",
                index=False,
            )

            rows_written[table_name] = int(len(dataframe))

    return rows_written

# =============================================================================
# Gold writers
# =============================================================================

def _ensure_gold_preprocessed_features_table(engine: Engine) -> None:
    """
    Create or migrate gold.preprocessed_features.

    CREATE TABLE IF NOT EXISTS only handles brand-new databases. The ALTER TABLE
    statements below protect reruns against schema drift when an older bootstrap
    already created gold.preprocessed_features without newer columns.
    """
    schema = "gold"
    table = "preprocessed_features"
    table_reference = sql_table_ref(schema, table)

    with engine.begin() as connection:
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {table_reference} (
                    preprocessed_id BIGSERIAL PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    asset_id TEXT,
                    event_time TIMESTAMPTZ,
                    event_step BIGINT,
                    time_index BIGINT,
                    feature_set_id TEXT,
                    split_name TEXT,
                    is_train BOOLEAN,
                    features_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    meta_truth_hash TEXT,
                    meta_parent_truth_hash TEXT,
                    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        )

        connection.execute(
            text(
                f"""
                ALTER TABLE {table_reference}
                ADD COLUMN IF NOT EXISTS asset_id TEXT;

                ALTER TABLE {table_reference}
                ADD COLUMN IF NOT EXISTS event_time TIMESTAMPTZ;

                ALTER TABLE {table_reference}
                ADD COLUMN IF NOT EXISTS event_step BIGINT;

                ALTER TABLE {table_reference}
                ADD COLUMN IF NOT EXISTS time_index BIGINT;

                ALTER TABLE {table_reference}
                ADD COLUMN IF NOT EXISTS feature_set_id TEXT;

                ALTER TABLE {table_reference}
                ADD COLUMN IF NOT EXISTS split_name TEXT;

                ALTER TABLE {table_reference}
                ADD COLUMN IF NOT EXISTS is_train BOOLEAN;

                ALTER TABLE {table_reference}
                ADD COLUMN IF NOT EXISTS features_json JSONB NOT NULL DEFAULT '{{}}'::jsonb;

                ALTER TABLE {table_reference}
                ADD COLUMN IF NOT EXISTS meta_truth_hash TEXT;

                ALTER TABLE {table_reference}
                ADD COLUMN IF NOT EXISTS meta_parent_truth_hash TEXT;

                ALTER TABLE {table_reference}
                ADD COLUMN IF NOT EXISTS created_at_utc TIMESTAMPTZ NOT NULL DEFAULT now();
                """
            )
        )

        connection.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS "idx_gold_preprocessed_dataset_run"
                ON {table_reference} (dataset_id, run_id)
                """
            )
        )

        connection.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS "idx_gold_preprocessed_split"
                ON {table_reference} (dataset_id, run_id, split_name)
                """
            )
        )


def write_gold_preprocessed_features_sql(
    *,
    engine: Engine,
    capstone_schema: str,
    dataset_id: str,
    run_id: str,
    notebook_globals: Optional[dict[str, Any]] = None,
    dataframe: Optional[pd.DataFrame] = None,
    dataset_name: Optional[str] = None,
    candidate_names: Optional[list[str]] = None,
    feature_set_id: str = "gold_preprocessed_v1",
) -> pd.DataFrame:
    """
    Write Gold preprocessed features to gold.preprocessed_features.
    """
    schema = "gold"
    table = "preprocessed_features"

    _ensure_gold_preprocessed_features_table(engine)

    candidates = candidate_names or [
        "gold_preprocessed_prescaled_dataframe",
        "gold_preprocessed_scaled_dataframe",
        "gold_preprocessed_dataframe",
        "gold_preprocessed_df",
        "preprocessed_dataframe",
        "preprocessed_df",
        "gold_preprocessed_scaled",
        "gold_dataframe",
        "gold_df",
        "gold_train_dataframe",
        "gold_test_dataframe",
        "gold_fit_dataframe",
    ]

    source_dataframe = _resolve_dataframe(
        dataframe=dataframe,
        candidate_names=candidates,
        notebook_globals=notebook_globals,
    ).reset_index(drop=False)

    _delete_dataset_run_rows(
        engine,
        schema=schema,
        table=table,
        dataset_id=dataset_id,
        run_id=run_id,
    )

    exclude_columns = {
        "dataset_id",
        "run_id",
        "asset_id",
        "event_time",
        "event_step",
        "time_index",
        "index",
        "meta__asset_id",
        "meta__truth_hash",
        "meta__parent_truth_hash",
        "meta__feature_set_id",
        "meta__split",
        "meta__is_train",
        "meta__is_train_flag",
    }

    feature_columns = [
        column
        for column in source_dataframe.columns
        if column not in exclude_columns
        and not str(column).startswith("meta__")
    ]

    rows: list[dict[str, Any]] = []

    for _, row in source_dataframe.iterrows():
        features_json = {
            column: to_builtin(row[column])
            for column in feature_columns
        }

        is_train_value = get_row_value(
            row,
            ["meta__is_train_flag", "meta__is_train", "is_train"],
            default=None,
        )

        if isinstance(is_train_value, str):
            is_train_value = is_train_value.strip().lower() in {"true", "1", "yes", "train"}

        rows.append(
            {
                "dataset_id": dataset_id,
                "run_id": run_id,
                "asset_id": get_row_value(row, ["meta__asset_id", "asset_id"], default=None),
                "event_time": get_row_value(row, ["event_time", "timestamp", "datetime", "date_time"]),
                "event_step": get_row_value(row, ["event_step", "step", "cycle", "index"]),
                "time_index": get_row_value(row, ["time_index", "event_step", "step", "cycle", "index"]),
                "feature_set_id": get_row_value(
                    row,
                    ["meta__feature_set_id", "feature_set_id"],
                    default=feature_set_id,
                ),
                "split_name": get_row_value(row, ["meta__split", "split_name"], default=None),
                "is_train": is_train_value,
                "features_json": to_json_string(features_json),
                "meta_truth_hash": get_row_value(row, ["meta__truth_hash"], default=None),
                "meta_parent_truth_hash": get_row_value(row, ["meta__parent_truth_hash"], default=None),
            }
        )

    insert_sql = f"""
    INSERT INTO {sql_table_ref(schema, table)} (
        dataset_id,
        run_id,
        asset_id,
        event_time,
        event_step,
        time_index,
        feature_set_id,
        split_name,
        is_train,
        features_json,
        meta_truth_hash,
        meta_parent_truth_hash,
        created_at_utc
    )
    VALUES (
        :dataset_id,
        :run_id,
        :asset_id,
        :event_time,
        :event_step,
        :time_index,
        :feature_set_id,
        :split_name,
        :is_train,
        CAST(:features_json AS jsonb),
        :meta_truth_hash,
        :meta_parent_truth_hash,
        now()
    )
    """

    _execute_many(engine, insert_sql, rows)

    _upsert_pipeline_run(
        engine,
        capstone_schema=capstone_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        pipeline_stage="gold_preprocessing",
        dataset_name=dataset_name or dataset_id,
        runtime_facts={
            "gold_preprocessed_sql_table": f"{schema}.{table}",
            "row_count": len(source_dataframe),
            "feature_column_count": len(feature_columns),
        },
    )

    _log_data_quality_event(
        engine,
        capstone_schema=capstone_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        layer_name=schema,
        table_name=table,
        check_name="gold_preprocessing_sql_write",
        check_status="pass",
        row_count=len(source_dataframe),
        details_json={"feature_column_count": len(feature_columns)},
    )

    return read_sql_dataframe(
        engine,
        f"""
        SELECT dataset_id, run_id, split_name, COUNT(*) AS row_count
        FROM {sql_table_ref(schema, table)}
        WHERE dataset_id = :dataset_id
          AND run_id = :run_id
        GROUP BY dataset_id, run_id, split_name
        ORDER BY split_name
        """,
        params={
            "dataset_id": dataset_id,
            "run_id": run_id,
        },
    )


def write_gold_anomaly_scores_sql(
    *,
    engine: Engine,
    capstone_schema: str,
    dataset_id: str,
    run_id: str,
    model_name: str,
    model_stage: str,
    score_column_candidates: list[str],
    flag_column_candidates: list[str],
    notebook_globals: Optional[dict[str, Any]] = None,
    dataframe: Optional[pd.DataFrame] = None,
    dataset_name: Optional[str] = None,
    candidate_names: Optional[list[str]] = None,
    evidence_column_mode: str = "basic",
) -> pd.DataFrame:
    """
    Generic writer for gold.anomaly_detection_scores.
    """
    schema = "gold"
    table = "anomaly_detection_scores"

    candidates = candidate_names or [
        "scored_dataframe",
        "scored_df",
        "results_dataframe",
        "results_df",
    ]

    source_dataframe = _resolve_dataframe(
        dataframe=dataframe,
        candidate_names=candidates,
        notebook_globals=notebook_globals,
    ).reset_index(drop=False)

    score_column = next(
        (column for column in score_column_candidates if column in source_dataframe.columns),
        None,
    )

    flag_column = next(
        (column for column in flag_column_candidates if column in source_dataframe.columns),
        None,
    )

    if score_column is None:
        raise KeyError(f"No score column found. Checked: {score_column_candidates}")

    if flag_column is None:
        raise KeyError(f"No flag column found. Checked: {flag_column_candidates}")

    _delete_model_score_rows(
        engine,
        schema=schema,
        table=table,
        dataset_id=dataset_id,
        run_id=run_id,
        model_name=model_name,
        model_stage=model_stage,
    )

    if evidence_column_mode == "cascade":
        evidence_columns = [
            column
            for column in source_dataframe.columns
            if str(column).startswith("stage")
            or str(column).startswith("cascade")
            or "breach" in str(column).lower()
            or "drift" in str(column).lower()
            or "persistence" in str(column).lower()
            or "corroboration" in str(column).lower()
            or "evidence" in str(column).lower()
        ]
    else:
        evidence_columns = [score_column, flag_column]

    rows: list[dict[str, Any]] = []

    for _, row in source_dataframe.iterrows():
        anomaly_flag = bool(get_row_value(row, [flag_column], default=False))

        evidence_json = {
            column: to_builtin(row[column])
            for column in evidence_columns
            if column in row.index
        }

        evidence_json["score_column"] = score_column
        evidence_json["flag_column"] = flag_column

        rows.append(
            {
                "dataset_id": dataset_id,
                "run_id": run_id,
                "asset_id": get_row_value(row, ["meta__asset_id", "asset_id"], default=None),
                "event_time": get_row_value(row, ["event_time", "timestamp", "datetime", "date_time"]),
                "event_step": get_row_value(row, ["event_step", "step", "cycle", "index"]),
                "time_index": get_row_value(row, ["time_index", "event_step", "step", "cycle", "index"]),
                "model_name": model_name,
                "model_stage": model_stage,
                "anomaly_score": to_scalar(row[score_column]),
                "anomaly_flag": anomaly_flag,
                "alert_severity": "alert" if anomaly_flag else "normal",
                "evidence_json": to_json_string(evidence_json),
                "meta_truth_hash": get_row_value(row, ["meta__truth_hash"], default=None),
                "meta_parent_truth_hash": get_row_value(row, ["meta__parent_truth_hash"], default=None),
            }
        )

    insert_sql = f"""
    INSERT INTO {sql_table_ref(schema, table)} (
        dataset_id,
        run_id,
        asset_id,
        event_time,
        event_step,
        time_index,
        model_name,
        model_stage,
        anomaly_score,
        anomaly_flag,
        alert_severity,
        evidence_json,
        meta_truth_hash,
        meta_parent_truth_hash,
        meta_scored_at_utc
    )
    VALUES (
        :dataset_id,
        :run_id,
        :asset_id,
        :event_time,
        :event_step,
        :time_index,
        :model_name,
        :model_stage,
        :anomaly_score,
        :anomaly_flag,
        :alert_severity,
        CAST(:evidence_json AS jsonb),
        :meta_truth_hash,
        :meta_parent_truth_hash,
        now()
    )
    """

    _execute_many(engine, insert_sql, rows)

    alert_count = int(source_dataframe[flag_column].sum())

    _upsert_pipeline_run(
        engine,
        capstone_schema=capstone_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        pipeline_stage=f"gold_{model_stage}_modeling",
        dataset_name=dataset_name or dataset_id,
        runtime_facts={
            "model_name": model_name,
            "model_stage": model_stage,
            "score_column": score_column,
            "flag_column": flag_column,
            "row_count": len(source_dataframe),
            "alert_count": alert_count,
            "evidence_column_count": len(evidence_columns),
        },
    )

    _log_data_quality_event(
        engine,
        capstone_schema=capstone_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        layer_name=schema,
        table_name=table,
        check_name=f"{model_stage}_sql_write",
        check_status="pass",
        row_count=len(source_dataframe),
        details_json={
            "model_name": model_name,
            "model_stage": model_stage,
            "alert_count": alert_count,
            "evidence_column_count": len(evidence_columns),
        },
    )

    return read_sql_dataframe(
        engine,
        f"""
        SELECT
            model_name,
            model_stage,
            COUNT(*) AS row_count,
            SUM(CASE WHEN anomaly_flag THEN 1 ELSE 0 END) AS alert_count
        FROM {sql_table_ref(schema, table)}
        WHERE dataset_id = :dataset_id
          AND run_id = :run_id
          AND model_name = :model_name
          AND model_stage = :model_stage
        GROUP BY model_name, model_stage
        """,
        params={
            "dataset_id": dataset_id,
            "run_id": run_id,
            "model_name": model_name,
            "model_stage": model_stage,
        },
    )


def write_gold_baseline_scores_sql(
    *,
    engine: Engine,
    capstone_schema: str,
    dataset_id: str,
    run_id: str,
    notebook_globals: Optional[dict[str, Any]] = None,
    dataframe: Optional[pd.DataFrame] = None,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper for baseline Isolation Forest scores.
    """
    return write_gold_anomaly_scores_sql(
        engine=engine,
        capstone_schema=capstone_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        model_name="baseline_isolation_forest",
        model_stage="baseline",
        score_column_candidates=["baseline_score", "anomaly_score", "score"],
        flag_column_candidates=["baseline_flag", "anomaly_flag", "is_anomaly"],
        notebook_globals=notebook_globals,
        dataframe=dataframe,
        dataset_name=dataset_name,
        candidate_names=[
            "baseline_scored_dataframe",
            "baseline_scored_df",
            "baseline_results_dataframe",
            "baseline_results_df",
            "scored_dataframe",
            "scored_df",
            "baseline_results",
        ],
        evidence_column_mode="basic",
    )


def write_gold_cascade_scores_sql(
    *,
    engine: Engine,
    capstone_schema: str,
    dataset_id: str,
    run_id: str,
    notebook_globals: Optional[dict[str, Any]] = None,
    dataframe: Optional[pd.DataFrame] = None,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper for cascade anomaly scores.
    """
    return write_gold_anomaly_scores_sql(
        engine=engine,
        capstone_schema=capstone_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        model_name="cascade_isolation_forest_rule_confirmation",
        model_stage="cascade_final",
        score_column_candidates=[
            "cascade_final_score",
            "stage3_score",
            "stage2_score",
            "stage1_score",
            "anomaly_score",
        ],
        flag_column_candidates=[
            "cascade_final_flag",
            "stage3_final_flag",
            "stage3_flag",
            "anomaly_flag",
            "is_anomaly",
        ],
        notebook_globals=notebook_globals,
        dataframe=dataframe,
        dataset_name=dataset_name,
        candidate_names=[
            "cascade_scored_dataframe",
            "cascade_scored_df",
            "cascade_results_dataframe",
            "cascade_results_df",
            "scored_dataframe",
            "scored_df",
            "cascade_results",
            
        ],
        evidence_column_mode="cascade",
    )


def write_gold_model_comparison_results_sql(
    *,
    engine: Engine,
    capstone_schema: str,
    dataset_id: str,
    run_id: str,
    notebook_globals: Optional[dict[str, Any]] = None,
    dataframe: Optional[pd.DataFrame] = None,
    dataset_name: Optional[str] = None,
    candidate_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Write baseline-vs-cascade comparison summary to gold.model_comparison_results.
    """
    schema = "gold"
    table = "model_comparison_results"

    candidates = candidate_names or [
        "baseline_vs_cascade",
        "baseline_vs_cascade_df",
        "baseline_vs_cascade_dataframe",
        "comparison_dataframe",
        "comparison_results_dataframe",
        "comparison_df",
    ]

    source_dataframe = _resolve_dataframe(
        dataframe=dataframe,
        candidate_names=candidates,
        notebook_globals=notebook_globals,
    ).reset_index(drop=True)

    _delete_dataset_run_rows(
        engine,
        schema=schema,
        table=table,
        dataset_id=dataset_id,
        run_id=run_id,
    )

    lower_columns = {
        str(column).lower(): column
        for column in source_dataframe.columns
    }

    model_column = lower_columns.get("model")

    if model_column is None:
        raise KeyError("Comparison dataframe needs a 'model' column.")

    baseline_rows = source_dataframe[
        source_dataframe[model_column].astype(str).str.lower().str.contains("baseline")
    ]

    cascade_rows = source_dataframe[
        source_dataframe[model_column].astype(str).str.lower().str.contains("cascade")
    ]

    if baseline_rows.empty:
        raise ValueError("Could not find baseline row in comparison dataframe.")

    if cascade_rows.empty:
        raise ValueError("Could not find cascade row in comparison dataframe.")

    baseline_row = baseline_rows.iloc[0]
    cascade_row = cascade_rows.iloc[0]

    def comparison_value(row: pd.Series, candidates: list[str], default: Any = None) -> Any:
        for candidate in candidates:
            if candidate in row.index:
                return to_scalar(row[candidate])
        return default

    record = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "baseline_model": str(baseline_row[model_column]),
        "comparison_model": str(cascade_row[model_column]),
        "alert_count_baseline": comparison_value(
            baseline_row,
            ["alert_count_all_rows", "alert_count", "alerts", "final_alert_count"],
        ),
        "alert_count_comparison": comparison_value(
            cascade_row,
            ["alert_count_all_rows", "alert_count", "alerts", "final_alert_count"],
        ),
        "precision_baseline": comparison_value(baseline_row, ["precision"]),
        "precision_comparison": comparison_value(cascade_row, ["precision"]),
        "recall_baseline": comparison_value(baseline_row, ["recall"]),
        "recall_comparison": comparison_value(cascade_row, ["recall"]),
        "f1_baseline": comparison_value(baseline_row, ["f1", "f1_score"]),
        "f1_comparison": comparison_value(cascade_row, ["f1", "f1_score"]),
        "comparison_json": to_json_string(
            {
                "comparison_rows": source_dataframe.to_dict(orient="records"),
                "source_columns": [str(column) for column in source_dataframe.columns],
            }
        ),
    }

    insert_sql = f"""
    INSERT INTO {sql_table_ref(schema, table)} (
        dataset_id,
        run_id,
        baseline_model,
        comparison_model,
        alert_count_baseline,
        alert_count_comparison,
        precision_baseline,
        precision_comparison,
        recall_baseline,
        recall_comparison,
        f1_baseline,
        f1_comparison,
        comparison_json,
        created_at_utc
    )
    VALUES (
        :dataset_id,
        :run_id,
        :baseline_model,
        :comparison_model,
        :alert_count_baseline,
        :alert_count_comparison,
        :precision_baseline,
        :precision_comparison,
        :recall_baseline,
        :recall_comparison,
        :f1_baseline,
        :f1_comparison,
        CAST(:comparison_json AS jsonb),
        now()
    )
    """

    _execute_many(engine, insert_sql, [record])

    _upsert_pipeline_run(
        engine,
        capstone_schema=capstone_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        pipeline_stage="gold_model_comparison",
        dataset_name=dataset_name or dataset_id,
        runtime_facts={
            "comparison_sql_table": f"{schema}.{table}",
            "baseline_model": record["baseline_model"],
            "comparison_model": record["comparison_model"],
            "alert_count_baseline": record["alert_count_baseline"],
            "alert_count_comparison": record["alert_count_comparison"],
            "precision_baseline": record["precision_baseline"],
            "precision_comparison": record["precision_comparison"],
            "recall_baseline": record["recall_baseline"],
            "recall_comparison": record["recall_comparison"],
            "f1_baseline": record["f1_baseline"],
            "f1_comparison": record["f1_comparison"],
        },
    )

    _log_data_quality_event(
        engine,
        capstone_schema=capstone_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        layer_name=schema,
        table_name=table,
        check_name="gold_comparison_sql_write",
        check_status="pass",
        row_count=1,
        details_json=record,
    )

    return read_sql_dataframe(
        engine,
        f"""
        SELECT
            dataset_id,
            run_id,
            baseline_model,
            comparison_model,
            alert_count_baseline,
            alert_count_comparison,
            precision_baseline,
            precision_comparison,
            recall_baseline,
            recall_comparison,
            f1_baseline,
            f1_comparison,
            created_at_utc
        FROM {sql_table_ref(schema, table)}
        WHERE dataset_id = :dataset_id
          AND run_id = :run_id
        ORDER BY created_at_utc DESC
        LIMIT 5
        """,
        params={
            "dataset_id": dataset_id,
            "run_id": run_id,
        },
    )


def log_gold_05_anomaly_detection_summary_sql(
    *,
    engine: Engine,
    capstone_schema: str,
    dataset_id: str,
    run_id: str,
    notebook_globals: Optional[dict[str, Any]] = None,
    dataframe: Optional[pd.DataFrame] = None,
    dataset_name: Optional[str] = None,
    candidate_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Log Gold 05 anomaly-detection summary metadata.
    """
    candidates = candidate_names or [
        "baseline_vs_cascade",
        "baseline_vs_cascade_df",
        "comparison_dataframe",
        "comparison_results_dataframe",
        "final_results_dataframe",
        "results_dataframe",
        "lead_time_comparison_df",
        "detected_rows_review_df",
        "comparison_summary_df", 
        "failure_lead_time_df",
        
    ]

    source_dataframe = _resolve_dataframe(
        dataframe=dataframe,
        candidate_names=candidates,
        notebook_globals=notebook_globals,
    )

    selected_run_key = None
    if notebook_globals is not None:
        selected_run_key = notebook_globals.get("SELECTED_RUN_KEY", None)

    summary = {
        "row_count": int(len(source_dataframe)),
        "column_count": int(len(source_dataframe.columns)),
        "columns": [str(column) for column in source_dataframe.columns],
        "selected_run_key": selected_run_key,
    }

    if "model" in source_dataframe.columns:
        summary["models"] = [
            str(value)
            for value in source_dataframe["model"].dropna().unique().tolist()
        ]

    if "alert_count_all_rows" in source_dataframe.columns:
        if "model" in source_dataframe.columns:
            summary["alert_count_all_rows_by_model"] = (
                source_dataframe[["model", "alert_count_all_rows"]]
                .dropna()
                .to_dict(orient="records")
            )
        else:
            summary["alert_count_all_rows"] = (
                source_dataframe["alert_count_all_rows"].dropna().tolist()
            )

    _upsert_pipeline_run(
        engine,
        capstone_schema=capstone_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        pipeline_stage="gold_anomaly_detection_summary",
        dataset_name=dataset_name or dataset_id,
        runtime_facts=summary,
        notes="Gold 05 anomaly-detection summary logged from notebook.",
    )

    _log_data_quality_event(
        engine,
        capstone_schema=capstone_schema,
        dataset_id=dataset_id,
        run_id=run_id,
        layer_name="gold",
        table_name="gold_05_anomaly_detection_notebook",
        check_name="gold_05_summary_sql_log",
        check_status="pass",
        row_count=len(source_dataframe),
        details_json=summary,
    )

    artifact_variables = [
        "baseline_summary_path",
        "cascade_summary_path",
        "comparison_summary_path",
        "baseline_vs_cascade_path",
        "gold_anomaly_detection_ledger_path",
        "selected_run_artifact_path",
        "model_comparison_plot_path",
        "final_report_table_path",
    ]

    if notebook_globals is not None:
        for variable_name in artifact_variables:
            artifact_path = notebook_globals.get(variable_name)

            if artifact_path is not None:
                _log_pipeline_artifact(
                    engine,
                    capstone_schema=capstone_schema,
                    dataset_id=dataset_id,
                    run_id=run_id,
                    layer_name="gold",
                    stage_name="gold_05_anomaly_detection",
                    artifact_name=variable_name,
                    artifact_type="gold_anomaly_detection_artifact",
                    artifact_path=str(artifact_path),
                    metadata_json={
                        "source_variable": variable_name,
                        "selected_run_key": selected_run_key,
                    },
                )

    return read_sql_dataframe(
        engine,
        f"""
        SELECT
            pipeline_stage,
            run_status,
            completed_at_utc,
            runtime_facts
        FROM {sql_table_ref(capstone_schema, "pipeline_runs")}
        WHERE dataset_id = :dataset_id
          AND runtime_facts ->> 'source_run_id' = :run_id
        ORDER BY completed_at_utc DESC
        LIMIT 10
        """,
        params={
            "dataset_id": dataset_id,
            "run_id": run_id,
        },
    )