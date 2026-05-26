from __future__ import annotations

from typing import Sequence

import gc


from time import perf_counter

import numpy as np
import pandas as pd

from sqlalchemy import text

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
    log_memory,
)




# -----------------------------------------------------------------------------
# Time Logging helpers
# -----------------------------------------------------------------------------

def log_step_timing(step_name: str, start_time: float) -> float:
    elapsed_seconds = perf_counter() - start_time
    print(f"[timing] {step_name}: {elapsed_seconds:,.2f} seconds")
    return perf_counter()


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
# Stage Helpers
# -----------------------------------------------------------------------------

def quote_ident(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def fq_table(schema: str, table_name: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(table_name)}"

# -----------------------------------------------------------------------------
# Get Table Columns
# -----------------------------------------------------------------------------

def get_table_columns(engine, *, schema: str, table_name: str) -> list[str]:
    columns_df = read_sql_dataframe(
        engine,
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table_name
        ORDER BY ordinal_position
        """,
        params={
            "schema": schema,
            "table_name": table_name,
        },
    )

    return columns_df["column_name"].astype(str).tolist()

# -----------------------------------------------------------------------------
# Verify Columns Exist
# -----------------------------------------------------------------------------

def ensure_sensor_columns_exist(
    engine,
    *,
    schema: str,
    table_name: str,
    sensor_columns: list[str],
) -> None:
    existing_columns = set(
        get_table_columns(
            engine,
            schema=schema,
            table_name=table_name,
        )
    )

    missing_sensor_columns = [
        sensor_column
        for sensor_column in sensor_columns
        if sensor_column not in existing_columns
    ]

    if not missing_sensor_columns:
        print("No missing sensor columns found.")
        return

    with engine.begin() as conn:
        for sensor_column in missing_sensor_columns:
            conn.execute(
                text(
                    f"""
                    ALTER TABLE {fq_table(schema, table_name)}
                    ADD COLUMN IF NOT EXISTS {quote_ident(sensor_column)} DOUBLE PRECISION
                    """
                )
            )

    print("Added missing sensor columns:", missing_sensor_columns)


# -----------------------------------------------------------------------------
# Stage builder - Old Method
# -----------------------------------------------------------------------------

def build_sensor_messages_stage(
    engine,
    *,
    schema: str = "capstone",
    source_table: str = "synthetic_observations_timestamped_stage",
    target_table: str = "synthetic_sensor_messages_stage",
    if_exists: str = "replace",
    random_seed: int = 42,
    n_sensors: int = 52,
    chunk_size: int = 10000,
    enable_memory_logging: bool = False,
) -> str:
    """
    Build the long-format sensor message stage from the timestamped premelt
    observation stage in chunks instead of loading/melting the full table at once.
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
        "observation_timestamp",
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
        "observation_timestamp",
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
        dataframe_chunk: pd.DataFrame,
        chunk_number: int,
        start_row: int,
        end_row: int,
    ) -> pd.DataFrame:
        dataframe_work = dataframe_chunk.copy()

        # Optional downcast to reduce memory pressure before melt
        for column in sensor_columns:
            if column in dataframe_work.columns:
                dataframe_work[column] = pd.to_numeric(dataframe_work[column], errors="coerce").astype("float32")

        dataframe_work = dataframe_work.sort_values(
            by=["observation_index"],
            kind="stable",
        ).reset_index(drop=True)

        dataframe_long = pd.melt(
            dataframe_work,
            id_vars=id_columns,
            value_vars=sensor_columns,
            var_name="sensor_name",
            value_name="sensor_value",
        )

        dataframe_long["sensor_index"] = _extract_sensor_index(dataframe_long["sensor_name"])

        dataframe_long = dataframe_long.sort_values(
            by=["observation_index", "sensor_index"],
            kind="stable",
        ).reset_index(drop=True)

        observation_count = len(dataframe_work)
        dataframe_long["message_sequence_index"] = _build_message_sequence_index_with_rng(
            observation_count=observation_count,
            sensors_per_observation=n_sensors,
            rng=rng,
        )

        remaining_columns = [
            column for column in dataframe_long.columns
            if column not in ordered_columns
        ]
        dataframe_long = dataframe_long[ordered_columns + remaining_columns]

        print(
            f"[chunk] {chunk_number} melted "
            f"{len(dataframe_work):,} observations -> {len(dataframe_long):,} sensor rows"
        )

        return dataframe_long

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
        enable_memory_logging=enable_memory_logging,
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
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_observation_timestamp"
        ON "{safe_schema}"."{safe_target_table}" (observation_timestamp);
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
# Stage builder - SQL Native 
# -----------------------------------------------------------------------------

def build_sensor_messages_stage_sql_native(
    engine,
    *,
    schema: str,
    source_table: str,
    target_table: str,
    n_sensors: int = 52,
    enable_memory_logging: bool = False,
) -> str:
    sensor_columns = [f"sensor_{sensor_index:02d}" for sensor_index in range(n_sensors)]

    ensure_sensor_columns_exist(
        engine,
        schema=schema,
        table_name=source_table,
        sensor_columns=sensor_columns,
    )

    timer = perf_counter()

    if enable_memory_logging:
        log_memory("stage 04 sql-native - before metadata read")

    source_columns = get_table_columns(
        engine,
        schema=schema,
        table_name=source_table,
    )

    timer = log_step_timing("metadata read complete", timer)
    if enable_memory_logging:
        log_memory("stage 04 sql-native - after metadata read")

    passthrough_columns = [
        column
        for column in source_columns
        if column not in sensor_columns
    ]

    passthrough_select_sql = ",\n        ".join(
        f"t.{quote_ident(column)}"
        for column in passthrough_columns
    )

    sensor_values_sql = ",\n            ".join(
        (
            f"({sensor_index}::integer, "
            f"'{sensor_column}'::text, "
            f"t.{quote_ident(sensor_column)}::double precision)"
        )
        for sensor_index, sensor_column in enumerate(sensor_columns)
    )

    source_fq = fq_table(schema, source_table)
    target_fq = fq_table(schema, target_table)

    create_sql = f"""
    CREATE TABLE {target_fq} AS
    SELECT
        {passthrough_select_sql},
        v.sensor_name,
        v.sensor_index,
        v.sensor_value,
        v.sensor_index AS message_sequence_index
    FROM {source_fq} AS t
    CROSS JOIN LATERAL (
        VALUES
            {sensor_values_sql}
    ) AS v(sensor_index, sensor_name, sensor_value)
    """

    index_sql_statements = [
        f"""
        CREATE INDEX IF NOT EXISTS ix_{target_table}_obs_sensor
        ON {target_fq} (observation_index, sensor_index)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS ix_{target_table}_run_obs
        ON {target_fq} (dataset_id, run_id, observation_index)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS ix_{target_table}_sensor_name
        ON {target_fq} (sensor_name)
        """,
    ]

    
    if enable_memory_logging:
        log_memory("stage 04 sql-native - before SQL melt/create table")

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {target_fq} CASCADE"))
        conn.execute(text(create_sql))

        timer = log_step_timing("SQL melt/create table complete", timer)

        if enable_memory_logging:
            log_memory("stage 04 sql-native - after SQL melt/create table")

        if enable_memory_logging:
            log_memory("stage 04 sql-native - before indexes/analyze")

        for index_sql in index_sql_statements:
            conn.execute(text(index_sql))

        conn.execute(text(f"ANALYZE {target_fq}"))

        timer = log_step_timing("indexes/analyze complete", timer)

        if enable_memory_logging:
            log_memory("stage 04 sql-native - after indexes/analyze")

    gc.collect()
    if enable_memory_logging:
        log_memory("stage 04 sql-native - after gc")

    return target_table


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
        COUNT(DISTINCT observation_timestamp) AS distinct_observation_timestamp_count,
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
    "build_sensor_messages_stage_sql_native",
    "validate_sensor_messages_stage",
]
