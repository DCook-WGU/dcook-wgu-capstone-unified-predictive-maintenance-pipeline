from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

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
    process_observation_index_windows,
    resolve_dataset_run_from_table,
)

# -----------------------------------------------------------------------------
# Shared column helpers
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

    print(f"[rebuild] Added {len(missing)} new columns to {safe_schema}.{safe_table}")


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _validate_consumed_columns(dataframe: pd.DataFrame) -> None:
    required_columns = [
        "dataset_id",
        "run_id",
        "asset_id",
        "message_key",
        "generated_row_id",
        "observation_index",
        "observation_timestamp",
        "stream_state",
        "phase",
        "sensor_name",
        "sensor_index",
        "sensor_value",
        "meta_episode_id",
        "meta_primary_fault_type",
        "meta_magnitude",
        "consumer_received_at",
        "kafka_topic",
        "kafka_partition",
        "kafka_offset",
        "rebuild_status",
    ]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            "Consumed-stage source table is missing required columns: "
            + ", ".join(missing)
        )


def _build_sensor_columns(n_sensors: int = 52) -> list[str]:
    return [f"sensor_{i:02d}" for i in range(n_sensors)]


# -----------------------------------------------------------------------------
# Target table helpers
# -----------------------------------------------------------------------------

def ensure_rebuilt_stage_table_exists(
    engine,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_observations_rebuilt_stage",
) -> str:
    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = f"""
    CREATE TABLE IF NOT EXISTS "{safe_schema}"."{safe_table}" (
        dataset_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        observation_index BIGINT NOT NULL,
        observation_timestamp TIMESTAMPTZ,
        generated_row_id TEXT,
        stream_state TEXT,
        phase TEXT,
        meta_episode_id TEXT,
        meta_primary_fault_type TEXT,
        meta_magnitude DOUBLE PRECISION,
        rebuild_sensor_count INTEGER,
        rebuild_is_complete BOOLEAN,
        rebuild_completed_at TIMESTAMPTZ,
        rebuild_notes TEXT,
        PRIMARY KEY (dataset_id, run_id, asset_id, observation_index)
    );
    """
    execute_sql(engine, sql)

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_obs_ts"
        ON "{safe_schema}"."{safe_table}" (observation_timestamp);
        '''
    )

    execute_sql(
        engine,
        f'''
        CREATE INDEX IF NOT EXISTS "idx_{safe_table}_rebuild_complete"
        ON "{safe_schema}"."{safe_table}" (rebuild_is_complete);
        '''
    )

    return safe_table


# -----------------------------------------------------------------------------
# Read / dedupe helpers
# -----------------------------------------------------------------------------

def load_consumed_messages_for_rebuild(
    engine,
    *,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_messages_consumed_stage",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    rebuild_status: Optional[str] = "pending",
) -> pd.DataFrame:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(source_table)

    where_clauses = []
    params = {}

    if dataset_id is not None:
        where_clauses.append("dataset_id = :dataset_id")
        params["dataset_id"] = str(dataset_id).strip()

    if run_id is not None:
        where_clauses.append("run_id = :run_id")
        params["run_id"] = str(run_id).strip()

    if rebuild_status is not None:
        where_clauses.append("rebuild_status = :rebuild_status")
        params["rebuild_status"] = str(rebuild_status).strip()

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    sql = f"""
    SELECT *
    FROM "{safe_schema}"."{safe_table}"
    {where_sql}
    ORDER BY observation_index, sensor_index, consumer_received_at, kafka_offset
    """
    dataframe = read_sql_dataframe(engine, sql, params=params)

    if dataframe.empty:
        return dataframe

    _validate_consumed_columns(dataframe)
    return dataframe


def deduplicate_consumed_messages(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Deduplicate at the logical sensor-message level.

    Canonical identity:
    dataset_id + run_id + asset_id + observation_index + sensor_index

    Keep the latest received row for each logical identity.
    """
    if dataframe.empty:
        return dataframe.copy()

    working = dataframe.copy()

    sort_columns = [
        "dataset_id",
        "run_id",
        "asset_id",
        "observation_index",
        "sensor_index",
        "consumer_received_at",
        "kafka_partition",
        "kafka_offset",
    ]

    present_sort_columns = [column for column in sort_columns if column in working.columns]
    working = working.sort_values(by=present_sort_columns, kind="stable").reset_index(drop=True)

    logical_key = [
        "dataset_id",
        "run_id",
        "asset_id",
        "observation_index",
        "sensor_index",
    ]

    deduped = working.drop_duplicates(
        subset=logical_key,
        keep="last",
    ).reset_index(drop=True)

    return deduped


# -----------------------------------------------------------------------------
# Rebuild helpers
# -----------------------------------------------------------------------------

def build_rebuilt_observations_dataframe(
    dataframe: pd.DataFrame,
    *,
    n_sensors: int = 52,
    complete_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rebuild wide observations from long consumed messages.

    Returns
    -------
    rebuilt_dataframe
        Wide rebuilt observations.
    rebuilt_keys
        Observation identity keys used for optional rebuild-status updates.
    """
    if dataframe.empty:
        empty_keys = pd.DataFrame(
            columns=["dataset_id", "run_id", "asset_id", "observation_index"]
        )
        return pd.DataFrame(), empty_keys

    working = dataframe.copy()

    # Base observation metadata from deduped long messages
    group_columns = ["dataset_id", "run_id", "asset_id", "observation_index"]

    observation_summary = (
        working.groupby(group_columns, dropna=False)
        .agg(
            generated_row_id=("generated_row_id", "first"),
            observation_timestamp=("observation_timestamp", "first"),
            stream_state=("stream_state", "first"),
            phase=("phase", "first"),
            meta_episode_id=("meta_episode_id", "first"),
            meta_primary_fault_type=("meta_primary_fault_type", "first"),
            meta_magnitude=("meta_magnitude", "first"),
            rebuild_sensor_count=("sensor_index", "nunique"),
        )
        .reset_index()
    )

    observation_summary["rebuild_is_complete"] = (
        observation_summary["rebuild_sensor_count"].astype(int) == int(n_sensors)
    )
    observation_summary["rebuild_completed_at"] = pd.Timestamp.utcnow()
    observation_summary["rebuild_notes"] = observation_summary["rebuild_is_complete"].map(
        lambda is_complete: None if bool(is_complete) else f"Observation missing one or more of {n_sensors} sensors."
    )

    if complete_only:
        observation_summary = observation_summary.loc[
            observation_summary["rebuild_is_complete"]
        ].reset_index(drop=True)

    if observation_summary.empty:
        empty_keys = pd.DataFrame(columns=group_columns)
        return pd.DataFrame(), empty_keys

    valid_observation_keys = observation_summary[group_columns].copy()

    # Keep only rows that belong to observations being rebuilt
    keyed = working.merge(
        valid_observation_keys.assign(_keep_flag=True),
        on=group_columns,
        how="inner",
    )

    # Pivot long sensor rows to wide sensor columns
    wide_sensors = (
        keyed.pivot_table(
            index=group_columns,
            columns="sensor_index",
            values="sensor_value",
            aggfunc="first",
        )
        .reset_index()
    )

    # Rename numeric sensor columns to sensor_00 ... sensor_51
    renamed_columns = {}
    for column in wide_sensors.columns:
        if isinstance(column, int):
            renamed_columns[column] = f"sensor_{int(column):02d}"
    wide_sensors = wide_sensors.rename(columns=renamed_columns)

    # Guarantee all sensor columns exist
    sensor_columns = _build_sensor_columns(n_sensors=n_sensors)
    for column in sensor_columns:
        if column not in wide_sensors.columns:
            wide_sensors[column] = None

    rebuilt = observation_summary.merge(
        wide_sensors,
        on=group_columns,
        how="left",
    )

    ordered_columns = [
        "dataset_id",
        "run_id",
        "asset_id",
        "generated_row_id",
        "observation_index",
        "observation_timestamp",
        "stream_state",
        "phase",
        "meta_episode_id",
        "meta_primary_fault_type",
        "meta_magnitude",
    ] + sensor_columns + [
        "rebuild_sensor_count",
        "rebuild_is_complete",
        "rebuild_completed_at",
        "rebuild_notes",
    ]

    remaining_columns = [column for column in rebuilt.columns if column not in ordered_columns]
    rebuilt = rebuilt[ordered_columns + remaining_columns]

    rebuilt = rebuilt.sort_values(
        by=["dataset_id", "run_id", "asset_id", "observation_index"],
        kind="stable",
    ).reset_index(drop=True)

    return rebuilt, valid_observation_keys


def _remove_already_rebuilt_observations(
    engine,
    *,
    rebuilt_dataframe: pd.DataFrame,
    schema: str,
    target_table: str,
) -> pd.DataFrame:
    if rebuilt_dataframe.empty:
        return rebuilt_dataframe.copy()

    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(target_table)

    if _get_existing_columns(engine, schema=safe_schema, table=safe_table) == set():
        return rebuilt_dataframe.copy()

    existing = read_sql_dataframe(
        engine,
        f"""
        SELECT dataset_id, run_id, asset_id, observation_index
        FROM "{safe_schema}"."{safe_table}"
        """
    )

    if existing.empty:
        return rebuilt_dataframe.copy()

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
            rebuilt_dataframe["dataset_id"].astype(str),
            rebuilt_dataframe["run_id"].astype(str),
            rebuilt_dataframe["asset_id"].astype(str),
            rebuilt_dataframe["observation_index"].astype(int),
        )
    )

    keep_mask = [key not in existing_keys for key in incoming_keys]
    return rebuilt_dataframe.loc[keep_mask].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Write helpers
# -----------------------------------------------------------------------------

def write_rebuilt_observations_batch(
    engine,
    dataframe: pd.DataFrame,
    *,
    schema: str = "capstone",
    table_name: str = "synthetic_sensor_observations_rebuilt_stage",
) -> str:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if dataframe.empty:
        return sanitize_sql_identifier(table_name)

    safe_schema = create_schema_if_not_exists(engine, schema)
    safe_table = ensure_rebuilt_stage_table_exists(
        engine,
        schema=schema,
        table_name=table_name,
    )

    working = dataframe.copy()
    working.columns = [sanitize_sql_identifier(column) for column in working.columns]

    working = _remove_already_rebuilt_observations(
        engine,
        rebuilt_dataframe=working,
        schema=safe_schema,
        target_table=safe_table,
    )

    if working.empty:
        return safe_table

    _add_missing_columns(
        engine,
        schema=safe_schema,
        table=safe_table,
        dataframe=working,
    )

    return write_layer_dataframe(
        engine=engine,
        dataframe=working,
        schema=safe_schema,
        table_name=safe_table,
        if_exists="append",
        index=False,
    )


# -----------------------------------------------------------------------------
# Status update helpers
# -----------------------------------------------------------------------------

def mark_consumed_messages_rebuilt(
    engine,
    observation_keys: pd.DataFrame,
    *,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_messages_consumed_stage",
) -> int:
    """
    Mark all consumed long rows for rebuilt observations as rebuilt.
    """
    if observation_keys.empty:
        return 0

    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(source_table)

    updated_count = 0

    for _, row in observation_keys.iterrows():
        sql = f"""
        UPDATE "{safe_schema}"."{safe_table}"
        SET rebuild_status = 'rebuilt'
        WHERE dataset_id = :dataset_id
          AND run_id = :run_id
          AND asset_id = :asset_id
          AND observation_index = :observation_index
          AND rebuild_status = 'pending'
        """
        execute_sql(
            engine,
            sql,
            params={
                "dataset_id": str(row["dataset_id"]).strip(),
                "run_id": str(row["run_id"]).strip(),
                "asset_id": str(row["asset_id"]).strip(),
                "observation_index": int(row["observation_index"]),
            },
        )
        updated_count += 1

    return updated_count


# -----------------------------------------------------------------------------
# Orchestration helper
# -----------------------------------------------------------------------------

def rebuild_consumed_messages_to_observations(
    engine,
    *,
    schema: str = "capstone",
    source_table: str = "synthetic_sensor_messages_consumed_stage",
    target_table: str = "synthetic_sensor_observations_rebuilt_stage",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    rebuild_status: Optional[str] = "pending",
    n_sensors: int = 52,
    complete_only: bool = True,
    mark_source_rebuilt: bool = True,
    observation_window_size: int = 2500,
) -> dict:
    safe_schema = sanitize_sql_identifier(schema)
    safe_source_table = sanitize_sql_identifier(source_table)

    where_sql = ""
    params = {}
    extra_where_sql = ""

    if rebuild_status is not None:
        extra_where_sql += " AND rebuild_status = :rebuild_status"
        params["rebuild_status"] = str(rebuild_status).strip()
        where_sql = "WHERE rebuild_status = :rebuild_status"

    resolved_dataset_id, resolved_run_id = resolve_dataset_run_from_table(
        engine,
        schema_name=safe_schema,
        table_name=safe_source_table,
        dataset_id=dataset_id,
        run_id=run_id,
        where_sql=where_sql,
        params=params,
    )

    source_columns = get_table_columns(
        engine,
        schema_name=safe_schema,
        table_name=safe_source_table,
    )

    stats = {
        "status": "empty",
        "consumed_rows": 0,
        "deduped_rows": 0,
        "rebuilt_rows": 0,
        "rebuilt_observations": 0,
        "updated_source_observation_groups": 0,
        "target_table": sanitize_sql_identifier(target_table),
    }

    def transform_chunk_func(df_window: pd.DataFrame, window_number: int, obs_min: int, obs_max: int) -> dict:
        deduped = deduplicate_consumed_messages(df_window)

        rebuilt_dataframe, rebuilt_keys = build_rebuilt_observations_dataframe(
            deduped,
            n_sensors=n_sensors,
            complete_only=complete_only,
        )

        return {
            "consumed_rows": int(len(df_window)),
            "deduped_rows": int(len(deduped)),
            "rebuilt_dataframe": rebuilt_dataframe,
            "rebuilt_keys": rebuilt_keys,
        }

    def write_chunk_func(payload: dict, window_number: int, obs_min: int, obs_max: int) -> None:
        stats["consumed_rows"] += int(payload["consumed_rows"])
        stats["deduped_rows"] += int(payload["deduped_rows"])

        rebuilt_dataframe = payload["rebuilt_dataframe"]
        rebuilt_keys = payload["rebuilt_keys"]

        if rebuilt_dataframe.empty:
            return

        written_table = write_rebuilt_observations_batch(
            engine=engine,
            dataframe=rebuilt_dataframe,
            schema=schema,
            table_name=target_table,
        )

        stats["status"] = "rebuilt"
        stats["target_table"] = written_table
        stats["rebuilt_rows"] += int(len(rebuilt_dataframe))
        stats["rebuilt_observations"] += int(len(rebuilt_keys))

        if mark_source_rebuilt and not rebuilt_keys.empty:
            updated_count = mark_consumed_messages_rebuilt(
                engine=engine,
                observation_keys=rebuilt_keys,
                schema=schema,
                source_table=source_table,
            )
            stats["updated_source_observation_groups"] += int(updated_count)

    process_observation_index_windows(
        engine,
        schema_name=safe_schema,
        table_name=safe_source_table,
        select_columns=source_columns,
        dataset_id=resolved_dataset_id,
        run_id=resolved_run_id,
        transform_chunk_func=transform_chunk_func,
        write_chunk_func=write_chunk_func,
        window_size=observation_window_size,
        extra_where_sql=extra_where_sql,
        params=params,
        order_by_sql="observation_index, sensor_index, consumer_received_at, kafka_offset",
    )

    if stats["rebuilt_rows"] == 0 and stats["consumed_rows"] > 0:
        stats["status"] = "no_complete_observations"

    return stats


__all__ = [
    "ensure_rebuilt_stage_table_exists",
    "load_consumed_messages_for_rebuild",
    "deduplicate_consumed_messages",
    "build_rebuilt_observations_dataframe",
    "write_rebuilt_observations_batch",
    "mark_consumed_messages_rebuilt",
    "rebuild_consumed_messages_to_observations",
]