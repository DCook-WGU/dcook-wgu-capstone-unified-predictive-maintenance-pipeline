from __future__ import annotations

from typing import Sequence

import pandas as pd

from utils.postgres_util import (
    sanitize_sql_identifier,
    create_schema_if_not_exists,
    execute_sql,
    read_sql_dataframe,
    get_engine_from_env,
)
from utils.layer_postgres_writer import write_layer_dataframe


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _validate_source_columns(dataframe: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Premelt source table is missing required columns: "
            + ", ".join(missing)
        )


def _build_generated_row_id(run_id: str, observation_index: pd.Series) -> pd.Series:
    return (
        str(run_id).strip()
        + "_obs_"
        + observation_index.astype(int).astype(str).str.zfill(12)
    )


# -----------------------------------------------------------------------------
# Stage builder
# -----------------------------------------------------------------------------

def build_observations_premelt_stage(
    engine,
    *,
    schema: str = "capstone",
    source_table: str = "synthetic_pump_stream",
    target_table: str = "synthetic_observations_premelt_stage",
    dataset_id: str = "pump_synthetic_v1",
    run_id: str = "premelt_run_001",
    asset_id: str = "pump_asset_001",
    if_exists: str = "replace",
) -> str:
    """
    Build the premelt stage from the raw synthetic stream table.

    Rules:
    - reads from the existing wide source table
    - preserves order using batch_id -> row_in_batch
    - creates observation_index after ordering
    - adds forward-only metadata fields
    - does not create timestamps
    - does not melt to long format
    - does not introduce Kafka behavior yet
    """
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_source_table = sanitize_sql_identifier(source_table)
    safe_target_table = sanitize_sql_identifier(target_table)

    source_sql = f"""
    SELECT *
    FROM "{safe_schema}"."{safe_source_table}"
    ORDER BY batch_id, row_in_batch
    """
    dataframe = read_sql_dataframe(engine, source_sql)

    if dataframe.empty:
        raise ValueError(
            f"Source table '{safe_schema}.{safe_source_table}' is empty."
        )

    required_columns = [
        "batch_id",
        "row_in_batch",
        "global_cycle_id",
        "stream_state",
        "phase",
        "created_at",
        "meta_episode_id",
        "meta_primary_fault_type",
        "meta_magnitude",
    ]
    required_columns.extend([f"sensor_{i:02d}" for i in range(52)])

    _validate_source_columns(dataframe, required_columns)

    # -------------------------------------------------------------------------
    # Enforce deterministic order before indexing
    # -------------------------------------------------------------------------
    dataframe = dataframe.sort_values(
        by=["batch_id", "row_in_batch"],
        kind="stable",
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Add required stage fields
    # -------------------------------------------------------------------------
    dataframe["dataset_id"] = str(dataset_id).strip()
    dataframe["run_id"] = str(run_id).strip()
    dataframe["asset_id"] = str(asset_id).strip()

    dataframe["observation_index"] = range(1, len(dataframe) + 1)
    dataframe["generated_row_id"] = _build_generated_row_id(
        run_id=run_id,
        observation_index=dataframe["observation_index"],
    )

    # telemetry framework stub fields
    dataframe["is_telemetry_event"] = False
    dataframe["telemetry_event_type"] = None
    dataframe["producer_send_attempt"] = 1

    # -------------------------------------------------------------------------
    # Reorder columns for readability
    # -------------------------------------------------------------------------
    preferred_front_columns = [
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
        "is_telemetry_event",
        "telemetry_event_type",
        "producer_send_attempt",
    ]

    remaining_columns = [
        column for column in dataframe.columns
        if column not in preferred_front_columns
    ]
    dataframe = dataframe[preferred_front_columns + remaining_columns]

    # -------------------------------------------------------------------------
    # Write stage table
    # -------------------------------------------------------------------------
    table_name = write_layer_dataframe(
        engine=engine,
        dataframe=dataframe,
        schema=safe_schema,
        table_name=safe_target_table,
        if_exists=if_exists,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Helpful indexes for downstream stages
    # -------------------------------------------------------------------------
    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_observation_index"
        ON "{safe_schema}"."{safe_target_table}" (observation_index);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_batch_row"
        ON "{safe_schema}"."{safe_target_table}" (batch_id, row_in_batch);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_generated_row_id"
        ON "{safe_schema}"."{safe_target_table}" (generated_row_id);
        '''
    )

    return table_name


# -----------------------------------------------------------------------------
# Validation helper
# -----------------------------------------------------------------------------

def validate_observations_premelt_stage(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_observations_premelt_stage",
) -> pd.DataFrame:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    SELECT
        COUNT(*) AS row_count,
        MIN(observation_index) AS min_observation_index,
        MAX(observation_index) AS max_observation_index,
        COUNT(DISTINCT generated_row_id) AS distinct_generated_row_id_count,
        MIN(batch_id) AS min_batch_id,
        MAX(batch_id) AS max_batch_id
    FROM "{safe_schema}"."{safe_table}"
    """
    return read_sql_dataframe(engine, sql)


__all__ = [
    "build_observations_premelt_stage",
    "validate_observations_premelt_stage",
]