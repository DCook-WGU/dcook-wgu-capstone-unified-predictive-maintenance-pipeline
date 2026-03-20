from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy.types import BIGINT, BOOLEAN, FLOAT, JSON, TEXT, TIMESTAMP

from utils.postgres_util import (
    sanitize_sql_identifier,
    create_schema_if_not_exists,
    read_sql_dataframe,
    table_exists,
)


# -----------------------------------------------------------------------------
# Naming helpers
# -----------------------------------------------------------------------------

def build_layer_table_name(
    *,
    dataset_name: str,
    layer: Optional[str] = None,
    artifact_name: Optional[str] = None,
    include_layer_prefix: bool = False,
) -> str:
    """
    Build a standard layer table name.
    """
    dataset_part = sanitize_sql_identifier(dataset_name)
    parts = []

    if include_layer_prefix:
        if not layer:
            raise ValueError("layer is required when include_layer_prefix=True")
        parts.append(sanitize_sql_identifier(layer))

    parts.append(dataset_part)

    if artifact_name is not None and str(artifact_name).strip() != "":
        parts.append(sanitize_sql_identifier(artifact_name))

    return "_".join(parts)


# -----------------------------------------------------------------------------
# Dtype inference helpers
# -----------------------------------------------------------------------------

def _series_looks_like_json(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False

    sample = non_null.iloc[0]
    return isinstance(sample, (dict, list))


def _infer_sqlalchemy_dtype_for_series(series: pd.Series):
    """
    Conservative dtype inference for Postgres writes.
    """
    if pd.api.types.is_bool_dtype(series):
        return BOOLEAN()

    if pd.api.types.is_integer_dtype(series):
        return BIGINT()

    if pd.api.types.is_float_dtype(series):
        return FLOAT(precision=53)

    if pd.api.types.is_datetime64tz_dtype(series):
        return TIMESTAMP(timezone=True)

    if pd.api.types.is_datetime64_any_dtype(series):
        return TIMESTAMP(timezone=False)

    if pd.api.types.is_object_dtype(series):
        if _series_looks_like_json(series):
            return JSON()
        return TEXT()

    if pd.api.types.is_string_dtype(series):
        return TEXT()

    if isinstance(series.dtype, pd.CategoricalDtype):
        return TEXT()

    return TEXT()


def infer_sqlalchemy_dtypes(
    dataframe: pd.DataFrame,
    dtype_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a dtype mapping for pandas.to_sql().
    """
    dtype_map: Dict[str, Any] = {}
    overrides = dtype_overrides or {}

    for column in dataframe.columns:
        if column in overrides:
            dtype_map[str(column)] = overrides[column]
        else:
            dtype_map[str(column)] = _infer_sqlalchemy_dtype_for_series(dataframe[column])

    return dtype_map


# -----------------------------------------------------------------------------
# Dataframe preparation helpers
# -----------------------------------------------------------------------------

def prepare_layer_dataframe(
    dataframe: pd.DataFrame,
    *,
    truth_hash: Optional[str] = None,
    parent_truth_hash: Optional[str] = None,
    pipeline_mode: Optional[str] = None,
    process_run_id: Optional[str] = None,
    add_loaded_at_column: bool = False,
    loaded_at_column: str = "meta__loaded_at",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Return a copy of the dataframe with optional capstone-style meta columns.

    This does not overwrite an existing column of the same name.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    out = dataframe.copy()

    meta_values = {
        "meta__truth_hash": truth_hash,
        "meta__parent_truth_hash": parent_truth_hash,
        "meta__pipeline_mode": pipeline_mode,
        "meta__process_run_id": process_run_id,
    }

    if extra_meta:
        meta_values.update(extra_meta)

    for column_name, value in meta_values.items():
        if value is not None and column_name not in out.columns:
            out[column_name] = value

    if add_loaded_at_column and loaded_at_column not in out.columns:
        out[loaded_at_column] = pd.Timestamp.now(tz="UTC")

    return out


# -----------------------------------------------------------------------------
# Layer IO
# -----------------------------------------------------------------------------

def write_layer_dataframe(
    engine: Engine,
    dataframe: pd.DataFrame,
    *,
    schema: str,
    dataset_name: Optional[str] = None,
    layer: Optional[str] = None,
    artifact_name: Optional[str] = None,
    table_name: Optional[str] = None,
    include_layer_prefix_in_table_name: bool = False,
    if_exists: str = "append",
    index: bool = False,
    chunksize: int = 5000,
    method: str = "multi",
    allow_empty: bool = False,
    dtype_overrides: Optional[Dict[str, Any]] = None,
    logger: Optional[Any] = None,
) -> str:
    """
    Generic dataframe writer for Bronze / Silver / Gold / synthetic layers.

    Returns the final table name used.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    if dataframe.empty and not allow_empty:
        raise ValueError("Refusing to write an empty dataframe. Set allow_empty=True to permit this.")

    if if_exists not in {"fail", "replace", "append"}:
        raise ValueError("if_exists must be one of: fail, replace, append")

    safe_schema = create_schema_if_not_exists(engine, schema)

    if table_name is None:
        if dataset_name is None:
            raise ValueError("dataset_name is required when table_name is not provided.")

        table_name = build_layer_table_name(
            dataset_name=dataset_name,
            layer=layer,
            artifact_name=artifact_name,
            include_layer_prefix=include_layer_prefix_in_table_name,
        )
    else:
        table_name = sanitize_sql_identifier(table_name)

    out = dataframe.copy()
    out.columns = [sanitize_sql_identifier(column) for column in out.columns]

    dtype_map = infer_sqlalchemy_dtypes(out, dtype_overrides=dtype_overrides)

    if logger is not None:
        logger.info(
            "Writing dataframe to Postgres | schema=%s | table=%s | rows=%s | columns=%s | if_exists=%s",
            safe_schema,
            table_name,
            len(out),
            len(out.columns),
            if_exists,
        )

    out.to_sql(
        name=table_name,
        con=engine,
        schema=safe_schema,
        if_exists=if_exists,
        index=index,
        chunksize=chunksize,
        method=method,
        dtype=dtype_map,
    )

    return table_name


def read_layer_dataframe(
    engine: Engine,
    *,
    schema: str,
    table_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    layer: Optional[str] = None,
    artifact_name: Optional[str] = None,
    include_layer_prefix_in_table_name: bool = False,
    columns: Optional[Sequence[str]] = None,
    where_clause: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
    require_exists: bool = False,
) -> pd.DataFrame:
    """
    Read a layer table back into pandas.
    """
    safe_schema = sanitize_sql_identifier(schema)

    if table_name is None:
        if dataset_name is None:
            raise ValueError("dataset_name is required when table_name is not provided.")

        table_name = build_layer_table_name(
            dataset_name=dataset_name,
            layer=layer,
            artifact_name=artifact_name,
            include_layer_prefix=include_layer_prefix_in_table_name,
        )
    else:
        table_name = sanitize_sql_identifier(table_name)

    if require_exists and not table_exists(engine, schema=safe_schema, table_name=table_name):
        raise FileNotFoundError(f"Table does not exist: {safe_schema}.{table_name}")

    if columns:
        safe_columns = ", ".join(f'"{sanitize_sql_identifier(column)}"' for column in columns)
    else:
        safe_columns = "*"

    sql_parts = [f'SELECT {safe_columns} FROM "{safe_schema}"."{table_name}"']

    if where_clause:
        sql_parts.append(f"WHERE {where_clause}")

    if order_by:
        sql_parts.append(f"ORDER BY {order_by}")

    if limit is not None:
        sql_parts.append(f"LIMIT {int(limit)}")

    sql = "\n".join(sql_parts)
    return read_sql_dataframe(engine, sql, params=params or {})


__all__ = [
    "build_layer_table_name",
    "infer_sqlalchemy_dtypes",
    "prepare_layer_dataframe",
    "write_layer_dataframe",
    "read_layer_dataframe",
    "table_exists",
]