from __future__ import annotations

from typing import Sequence, Any, cast
import math 
import pandas as pd

from utils.database.postgres import (
    create_schema_if_not_exists,
    execute_sql,
    read_sql_dataframe,
    sanitize_sql_identifier,
    table_exists,
)


"""SQL-native builder for the synthetic observations premelt stage.

This module intentionally keeps the same public function names used by the
notebook, but moves the heavy stage construction into Postgres with
CREATE TABLE AS / INSERT INTO ... SELECT.
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _get_table_columns(engine, *, schema: str, table_name: str) -> list[str]:
    """Return source table columns in database ordinal order."""
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = :schema_name
      AND table_name = :table_name
    ORDER BY ordinal_position
    """
    dataframe = read_sql_dataframe(
        engine,
        sql,
        params={
            "schema_name": str(schema).strip(),
            "table_name": str(table_name).strip(),
        },
    )
    return dataframe["column_name"].astype(str).tolist()



def _validate_source_columns(columns: Sequence[str], required_columns: Sequence[str]) -> None:
    """Validate that the synthetic stream table has required premelt inputs."""
    missing = [column for column in required_columns if column not in columns]
    if missing:
        raise ValueError(
            "Premelt source table is missing required columns: " + ", ".join(missing)
        )



def _build_select_sql(*, safe_schema: str, safe_source_table: str, remaining_source_columns: Sequence[str]) -> str:
    """Build the SQL SELECT that assigns observation identity and metadata."""
    remaining_sql = ",\n        ".join([f'"{column}"' for column in remaining_source_columns])
    remaining_clause = f",\n        {remaining_sql}" if remaining_sql else ""

    return f"""
    WITH ordered_source AS (
        SELECT
            *,
            ROW_NUMBER() OVER (ORDER BY batch_id, row_in_batch) AS observation_index
        FROM "{safe_schema}"."{safe_source_table}"
    )
    SELECT
        CAST(:dataset_id AS TEXT) AS dataset_id,
        CAST(:run_id AS TEXT) AS run_id,
        CAST(:asset_id AS TEXT) AS asset_id,
        CAST(:run_id AS TEXT) || '_obs_' || LPAD(observation_index::TEXT, 12, '0') AS generated_row_id,
        observation_index,
        batch_id,
        row_in_batch,
        global_cycle_id,
        stream_state,
        phase,
        created_at,
        meta_episode_id,
        meta_primary_fault_type,
        meta_magnitude,
        FALSE AS is_telemetry_event,
        NULL::TEXT AS telemetry_event_type,
        1::INTEGER AS producer_send_attempt{remaining_clause}
    FROM ordered_source
    ORDER BY observation_index
    """



def _write_stage_sql_native(
    engine,
    *,
    schema: str,
    target_table: str,
    select_sql: str,
    params: dict,
    if_exists: str,
) -> str:
    """Create, append to, or fail on the premelt target table in Postgres."""
    safe_schema = sanitize_sql_identifier(schema)
    safe_target_table = sanitize_sql_identifier(target_table)
    write_mode = str(if_exists).strip().lower()
    target_exists = table_exists(engine, schema=safe_schema, table_name=safe_target_table)

    if write_mode == "replace":
        execute_sql(engine, f'DROP TABLE IF EXISTS "{safe_schema}"."{safe_target_table}"')
        execute_sql(
            engine,
            f'CREATE TABLE "{safe_schema}"."{safe_target_table}" AS\n{select_sql}',
            params=params,
        )
        return safe_target_table

    if write_mode == "fail":
        if target_exists:
            raise ValueError(f"Target table already exists: {safe_schema}.{safe_target_table}")
        execute_sql(
            engine,
            f'CREATE TABLE "{safe_schema}"."{safe_target_table}" AS\n{select_sql}',
            params=params,
        )
        return safe_target_table

    if write_mode == "append":
        if not target_exists:
            execute_sql(
                engine,
                f'CREATE TABLE "{safe_schema}"."{safe_target_table}" AS\n{select_sql}',
                params=params,
            )
            return safe_target_table

        execute_sql(
            engine,
            f'INSERT INTO "{safe_schema}"."{safe_target_table}"\n{select_sql}',
            params=params,
        )
        return safe_target_table

    raise ValueError("if_exists must be one of: 'replace', 'append', 'fail'.")


# -----------------------------------------------------------------------------
# Stage builder
# -----------------------------------------------------------------------------

def scalar_to_int(value: object, name: str = "value") -> int:
    """Convert a scalar SQL result to int and reject missing values."""
    if value is None:
        raise ValueError(f"{name} cannot be missing.")

    if value is pd.NA:
        raise ValueError(f"{name} cannot be missing.")

    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"{name} cannot be missing.")

    return int(cast(Any, value))


def dataframe_row_count_to_int(
    dataframe: pd.DataFrame,
    *,
    column: str = "row_count",
) -> int:
    """Return a count value from a one-row dataframe as a plain int."""
    if dataframe.empty:
        return 0

    return scalar_to_int(
        dataframe.at[0, column],
        column,
    )

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
    """Build the premelt stage directly inside Postgres.

    This preserves the same notebook call signature as the pandas version, but
    avoids a full wide-table round-trip through pandas. The stage assigns
    dataset/run/asset identity, generated row IDs, observation indexes, and
    producer metadata while keeping sensor columns wide for the next stage.
    """
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_source_table = sanitize_sql_identifier(source_table)
    safe_target_table = sanitize_sql_identifier(target_table)

    source_columns = _get_table_columns(engine, schema=safe_schema, table_name=safe_source_table)
    if not source_columns:
        raise ValueError(f"Source table does not exist or has no columns: {safe_schema}.{safe_source_table}")

    row_count_sql = (
    f'SELECT COUNT(*) AS row_count FROM "{safe_schema}"."{safe_source_table}"'
)

    row_count_dataframe = read_sql_dataframe(
        engine,
        row_count_sql,
    )

    row_count = dataframe_row_count_to_int(
        row_count_dataframe,
        column="row_count",
    )

    if row_count == 0:
        raise ValueError(f"Source table '{safe_schema}.{safe_source_table}' is empty.")


    # Validate the wide generator stream before creating derived observation IDs.
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
    _validate_source_columns(source_columns, required_columns)

    ordered_source_front_columns = [
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
    remaining_source_columns = [
        column for column in source_columns if column not in ordered_source_front_columns
    ]

    select_sql = _build_select_sql(
        safe_schema=safe_schema,
        safe_source_table=safe_source_table,
        remaining_source_columns=remaining_source_columns,
    )

    table_name = _write_stage_sql_native(
        engine,
        schema=safe_schema,
        target_table=safe_target_table,
        select_sql=select_sql,
        params={
            "dataset_id": str(dataset_id).strip(),
            "run_id": str(run_id).strip(),
            "asset_id": str(asset_id).strip(),
        },
        if_exists=if_exists,
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_observation_index"
        ON "{safe_schema}"."{safe_target_table}" (observation_index);
        ''',
    )
    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_batch_row"
        ON "{safe_schema}"."{safe_target_table}" (batch_id, row_in_batch);
        ''',
    )
    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_generated_row_id"
        ON "{safe_schema}"."{safe_target_table}" (generated_row_id);
        ''',
    )
    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_target_table}_dataset_run_obs"
        ON "{safe_schema}"."{safe_target_table}" (dataset_id, run_id, observation_index);
        ''',
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
    """Return row-count and identity checks for the premelt stage table."""
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
