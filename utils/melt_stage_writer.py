from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from utils.postgres_util import (
    sanitize_sql_identifier,
    create_schema_if_not_exists,
    execute_sql,
    read_sql_dataframe,
)
from utils.layer_postgres_writer import write_layer_dataframe


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _build_sensor_columns(n_sensors: int = 52) -> list[str]:
    return [f"sensor_{i:02d}" for i in range(n_sensors)]


def _validate_source_columns(dataframe: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Melt-stage source table is missing required columns: "
            + ", ".join(missing)
        )


def _extract_sensor_index(sensor_name_series: pd.Series) -> pd.Series:
    return (
        sensor_name_series.astype(str)
        .str.extract(r"(\d+)$", expand=False)
        .astype(int)
    )


def _build_message_sequence_index(
    observation_count: int,
    sensors_per_observation: int,
    random_seed: int,
) -> np.ndarray:
    """
    Build a randomized 0..(n_sensors-1) sequence for each observation.

    Output length:
        observation_count * sensors_per_observation

    Assumes the long dataframe is sorted by:
        observation_index, sensor_index
    before applying this output.
    """
    rng = np.random.default_rng(random_seed)
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
) -> str:
    """
    Build the long-format sensor message stage from the premelt observation stage.

    Behavior:
    - reads ordered observation rows from the premelt table
    - melts sensor_00..sensor_51 into long format
    - adds sensor_name, sensor_index, sensor_value
    - adds randomized message_sequence_index per observation
    - preserves observation reconstruction through observation_index + sensor_index
    - does not create timestamps yet
    - does not introduce Kafka yet
    """
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_source_table = sanitize_sql_identifier(source_table)
    safe_target_table = sanitize_sql_identifier(target_table)

    sensor_columns = _build_sensor_columns(n_sensors=n_sensors)

    source_sql = f"""
    SELECT *
    FROM "{safe_schema}"."{safe_source_table}"
    ORDER BY observation_index
    """
    dataframe = read_sql_dataframe(engine, source_sql)

    if dataframe.empty:
        raise ValueError(
            f"Source table '{safe_schema}.{safe_source_table}' is empty."
        )

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
        "is_telemetry_event",
        "telemetry_event_type",
        "producer_send_attempt",
    ]
    required_columns.extend(sensor_columns)

    _validate_source_columns(dataframe, required_columns)

    # -------------------------------------------------------------------------
    # Enforce deterministic observation order before melt
    # -------------------------------------------------------------------------
    dataframe = dataframe.sort_values(
        by=["observation_index"],
        kind="stable",
    ).reset_index(drop=True)

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

    # -------------------------------------------------------------------------
    # Melt wide sensors to long messages
    # -------------------------------------------------------------------------
    long_dataframe = pd.melt(
        dataframe,
        id_vars=id_columns,
        value_vars=sensor_columns,
        var_name="sensor_name",
        value_name="sensor_value",
    )

    long_dataframe["sensor_index"] = _extract_sensor_index(long_dataframe["sensor_name"])

    # -------------------------------------------------------------------------
    # Re-sort into clean observation/message order
    # -------------------------------------------------------------------------
    long_dataframe = long_dataframe.sort_values(
        by=["observation_index", "sensor_index"],
        kind="stable",
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Add randomized send-order index within each observation
    # -------------------------------------------------------------------------
    observation_count = len(dataframe)
    long_dataframe["message_sequence_index"] = _build_message_sequence_index(
        observation_count=observation_count,
        sensors_per_observation=n_sensors,
        random_seed=random_seed,
    )

    # -------------------------------------------------------------------------
    # Final column order
    # -------------------------------------------------------------------------
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

    remaining_columns = [
        column for column in long_dataframe.columns
        if column not in ordered_columns
    ]
    long_dataframe = long_dataframe[ordered_columns + remaining_columns]

    # -------------------------------------------------------------------------
    # Write long stage table
    # -------------------------------------------------------------------------
    table_name = write_layer_dataframe(
        engine=engine,
        dataframe=long_dataframe,
        schema=safe_schema,
        table_name=safe_target_table,
        if_exists=if_exists,
        index=False,
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

    return table_name


# -----------------------------------------------------------------------------
# Validation helper
# -----------------------------------------------------------------------------

def validate_sensor_messages_stage(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_messages_stage",
) -> pd.DataFrame:
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