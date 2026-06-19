"""Legacy/support melt-stage writer for synthetic sensor message staging.

This module is retained as an original support implementation for building the
long-format synthetic sensor message stage from the premelt observation table.
It remains useful as a reference for the chunked melt approach, but current
copy-back decisions should treat it as legacy/support material rather than the
primary maintained pipeline path.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from utils.database.postgres import (
    sanitize_sql_identifier,
    create_schema_if_not_exists,
    execute_sql,
    read_sql_dataframe,
)
from utils.database.layer_postgres import write_layer_dataframe
from utils.database.chunk_stage_util import (
    get_table_row_count,
    process_postgres_table_in_chunks,
)


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _build_sensor_columns(n_sensors: int = 52) -> list[str]:
    """
    Build the expected wide sensor column names for the premelt source table.

    Parameters
    ----------
    n_sensors:
        Number of sensor columns expected in the source dataframe.

    Returns
    -------
    list[str]
        Sensor column names in ``sensor_00`` through ``sensor_nn`` format.
    """
    return [f"sensor_{i:02d}" for i in range(n_sensors)]


def _validate_source_columns(dataframe: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """
    Confirm that the premelt source table exposes all required columns.

    Parameters
    ----------
    dataframe:
        Zero-row preview dataframe read from the source table.
    required_columns:
        Columns needed for the legacy/support melt-stage write.

    Raises
    ------
    ValueError
        If one or more required source columns are missing.
    """
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Melt-stage source table is missing required columns: "
            + ", ".join(missing)
        )


def _extract_sensor_index(sensor_name_series: pd.Series) -> pd.Series:
    """
    Extract numeric sensor indexes from names such as ``sensor_00``.

    Parameters
    ----------
    sensor_name_series:
        Series containing long-form sensor names.

    Returns
    -------
    pandas.Series
        Integer sensor indexes parsed from the trailing digits.
    """
    return (
        sensor_name_series.astype(str)
        .str.extract(r"(\d+)$", expand=False)
        .astype(int)
    )


def _build_message_sequence_index_with_rng(
    *,
    observation_count: int,
    sensors_per_observation: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build a randomized 0..(n_sensors-1) sequence for each observation
    using one shared RNG so chunking stays deterministic across the full run.
    """
    out = np.empty(observation_count * sensors_per_observation, dtype=int)

    for start in range(0, len(out), sensors_per_observation):
        out[start:start + sensors_per_observation] = rng.permutation(sensors_per_observation)

    return out


# -----------------------------------------------------------------------------
# Stage builder
# -----------------------------------------------------------------------------

def build_sensor_messages_stage(
    engine,
    *,
    schema: str = "capstone",
    source_table: str = "synthetic_observations_premelt_stage",
    target_table: str = "synthetic_sensor_messages_stage",
    if_exists: str = "replace",
    random_seed: int = 42,
    n_sensors: int = 52,
    chunk_size: int = 10000,
) -> str:
    """
    Build the long-format sensor message stage from the premelt observation stage
    in chunks instead of loading/melting the full table at once.
    """
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_source_table = sanitize_sql_identifier(source_table)
    safe_target_table = sanitize_sql_identifier(target_table)

    sensor_columns = _build_sensor_columns(n_sensors=n_sensors)

    id_columns = [
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

    required_columns = list(id_columns) + list(sensor_columns)

    # -------------------------------------------------------------------------
    # Validate source table and emptiness before chunk loop
    # -------------------------------------------------------------------------
    preview_sql = f'''
    SELECT *
    FROM "{safe_schema}"."{safe_source_table}"
    LIMIT 0
    '''
    preview_dataframe = read_sql_dataframe(engine, preview_sql)
    _validate_source_columns(preview_dataframe, required_columns)

    source_row_count = get_table_row_count(
        engine,
        schema_name=safe_schema,
        table_name=safe_source_table,
    )

    if source_row_count == 0:
        raise ValueError(
            f"Source table '{safe_schema}.{safe_source_table}' is empty."
        )

    rng = np.random.default_rng(random_seed)
    has_written_first_chunk = False

    ordered_columns = [
        "dataset_id",
        "run_id",
        "asset_id",
        "generated_row_id",
        "observation_index",
        "message_sequence_index",
        "batch_id",
        "row_in_batch",
        "global_cycle_id",
        "stream_state",
        "phase",
        "created_at",
        "meta_episode_id",
        "meta_primary_fault_type",
        "meta_magnitude",
        "sensor_name",
        "sensor_index",
        "sensor_value",
        "is_telemetry_event",
        "telemetry_event_type",
        "producer_send_attempt",
    ]

    def transform_chunk_func(
        df_chunk: pd.DataFrame,
        chunk_number: int,
        start_row: int,
        end_row: int,
    ) -> pd.DataFrame:
        df_work = df_chunk.copy()

        # Optional downcast to reduce memory pressure before melt
        for col in sensor_columns:
            if col in df_work.columns:
                df_work[col] = pd.to_numeric(df_work[col], errors="coerce").astype("float32")

        df_work = df_work.sort_values(
            by=["observation_index"],
            kind="stable",
        ).reset_index(drop=True)

        df_long = pd.melt(
            df_work,
            id_vars=id_columns,
            value_vars=sensor_columns,
            var_name="sensor_name",
            value_name="sensor_value",
        )

        df_long["sensor_index"] = _extract_sensor_index(df_long["sensor_name"])

        df_long = df_long.sort_values(
            by=["observation_index", "sensor_index"],
            kind="stable",
        ).reset_index(drop=True)

        observation_count = len(df_work)
        df_long["message_sequence_index"] = _build_message_sequence_index_with_rng(
            observation_count=observation_count,
            sensors_per_observation=n_sensors,
            rng=rng,
        )

        remaining_columns = [
            column for column in df_long.columns
            if column not in ordered_columns
        ]
        df_long = df_long[ordered_columns + remaining_columns]

        print(
            f"[chunk] {chunk_number} melted "
            f"{len(df_work):,} observations -> {len(df_long):,} sensor rows"
        )

        return df_long

    def write_chunk_func(
        df_out: pd.DataFrame,
        chunk_number: int,
        start_row: int,
        end_row: int,
    ) -> None:
        nonlocal has_written_first_chunk

        chunk_if_exists = if_exists if not has_written_first_chunk else "append"

        write_layer_dataframe(
            engine=engine,
            dataframe=df_out,
            schema=safe_schema,
            table_name=safe_target_table,
            if_exists=chunk_if_exists,
            index=False,
        )

        has_written_first_chunk = True

        print(
            f"[chunk] {chunk_number} wrote {len(df_out):,} rows "
            f"to {safe_schema}.{safe_target_table}"
        )

    process_postgres_table_in_chunks(
        engine,
        schema_name=safe_schema,
        table_name=safe_source_table,
        select_columns=required_columns,
        order_by_sql="observation_index",
        transform_chunk_func=transform_chunk_func,
        write_chunk_func=write_chunk_func,
        chunk_size=chunk_size,
    )

    # -------------------------------------------------------------------------
    # Helpful indexes for downstream timestamp/send stages
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
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_obs_sensor"
        ON "{safe_schema}"."{safe_target_table}" (observation_index, sensor_index);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_obs_message_seq"
        ON "{safe_schema}"."{safe_target_table}" (observation_index, message_sequence_index);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_generated_row_id"
        ON "{safe_schema}"."{safe_target_table}" (generated_row_id);
        '''
    )

    return safe_target_table


# -----------------------------------------------------------------------------
# Validation helper
# -----------------------------------------------------------------------------

def validate_sensor_messages_stage(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_stage",
) -> pd.DataFrame:
    """
    Summarize the legacy/support sensor message stage for quick validation.

    Parameters
    ----------
    engine:
        SQLAlchemy engine connected to the capstone PostgreSQL database.
    schema:
        Schema containing the sensor message stage table.
    table_name:
        Long-format sensor message stage table to summarize.

    Returns
    -------
    pandas.DataFrame
        One-row summary with row count, observation count, sensor count, and
        min/max sensor and message-sequence indexes.

    Side Effects
    ------------
    Reads from PostgreSQL only. No source tables, artifact files, logger state,
    or ledger entries are modified.
    """
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    SELECT
        COUNT(*) AS row_count,
        COUNT(DISTINCT observation_index) AS distinct_observation_count,
        COUNT(DISTINCT sensor_name) AS distinct_sensor_name_count,
        MIN(sensor_index) AS min_sensor_index,
        MAX(sensor_index) AS max_sensor_index,
        MIN(message_sequence_index) AS min_message_sequence_index,
        MAX(message_sequence_index) AS max_message_sequence_index
    FROM "{safe_schema}"."{safe_table}"
    """
    return read_sql_dataframe(engine, sql)


__all__ = [
    "build_sensor_messages_stage",
    "validate_sensor_messages_stage",
]
