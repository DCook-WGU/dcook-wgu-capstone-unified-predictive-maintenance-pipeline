from __future__ import annotations

import os
import re
import uuid
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

import psycopg2
from urllib.parse import quote_plus



VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PostgresUtilsError(Exception):
    """Base exception for postgres utility errors."""


class InvalidIdentifierError(PostgresUtilsError):
    """Raised when a schema, table, or column name is not SQL-safe."""


class EmptyKeyColumnsError(PostgresUtilsError):
    """Raised when an upsert operation is missing key columns."""


# ============================================================================
# Identifier helpers
# ============================================================================


def validate_identifier(identifier: str, label: str = "identifier") -> str:
    """
    Validate a SQL identifier so it can be safely interpolated into raw SQL.

    PostgreSQL does not allow parameter binding for schema/table/column names,
    so anything we place into the SQL string must be validated first.
    """
    if identifier is None:
        raise InvalidIdentifierError(f"{label} cannot be None.")

    cleaned = str(identifier).strip()
    if cleaned == "":
        raise InvalidIdentifierError(f"{label} cannot be empty.")

    if not VALID_IDENTIFIER.match(cleaned):
        raise InvalidIdentifierError(
            f"Invalid {label}: {identifier!r}. Only letters, numbers, and underscores are allowed, "
            "and the name must not start with a number."
        )

    return cleaned



def validate_identifiers(identifiers: Sequence[str], label: str = "identifier") -> list[str]:
    return [validate_identifier(value, label=label) for value in identifiers]



def q(identifier: str) -> str:
    """Quote a validated identifier for PostgreSQL."""
    safe_identifier = validate_identifier(identifier)
    return f'"{safe_identifier}"'


# ============================================================================
# Connection helpers
# ============================================================================


def build_postgres_url(
    *,
    host: str,
    port: int | str,
    database: str,
    username: str,
    password: str,
    driver: str = "psycopg2",
) -> str:
    """Build a SQLAlchemy PostgreSQL connection URL."""
    user_q = quote_plus(username)
    pwd_q = quote_plus(password)
    
    #url = f"postgresql+psycopg2://{user_q}:{pwd_q}@{host}:{port}/{database}"    

    return f"postgresql+{driver}://{user_q}:{pwd_q}@{host}:{port}/{database}"



def build_postgres_url_from_env(prefix: str = "POSTGRES", driver: str = "psycopg2") -> str:
    """
    Build a PostgreSQL URL from environment variables.

    Supported variables, in lookup order:
    - {prefix}_HOST, PGHOST
    - {prefix}_PORT, PGPORT
    - {prefix}_DB, {prefix}_DATABASE, PGDATABASE
    - {prefix}_USER, PGUSER
    - {prefix}_PASSWORD, PGPASSWORD
    """
    host = os.getenv(f"{prefix}_HOST") or os.getenv("PGHOST") or "localhost"
    port = os.getenv(f"{prefix}_PORT") or os.getenv("PGPORT") or "5432"
    database = (
        os.getenv(f"{prefix}_DB")
        or os.getenv(f"{prefix}_DATABASE")
        or os.getenv(f"{prefix}_NAME")
        or os.getenv("PGDATABASE")
    )
    username = os.getenv(f"{prefix}_USER") or os.getenv("PGUSER")
    password = os.getenv(f"{prefix}_PASSWORD") or os.getenv("PGPASSWORD")

    missing: list[str] = []
    if not database:
        missing.append(f"{prefix}_DB/{prefix}_DATABASE/{prefix}_NAME or PGDATABASE ")
    if not username:
        missing.append(f"{prefix}_USER or PGUSER")
    if not password:
        missing.append(f"{prefix}_PASSWORD or PGPASSWORD")

    if missing:
        raise PostgresUtilsError(
            "Missing PostgreSQL environment variables: " + ", ".join(missing)
        )

    return build_postgres_url(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        driver=driver,
    )



def get_engine(
    postgres_url: Optional[str] = None,
    *,
    echo: bool = False,
    pool_pre_ping: bool = True,
    future: bool = True,
    connect_args: Optional[dict[str, Any]] = None,
    env_prefix: str = "POSTGRES",
) -> Engine:
    """
    Create and return a SQLAlchemy engine.

    If postgres_url is not supplied, environment variables are used.
    """
    if not postgres_url:
        postgres_url = build_postgres_url_from_env(prefix=env_prefix)

    return create_engine(
        postgres_url,
        echo=echo,
        pool_pre_ping=pool_pre_ping,
        future=future,
        connect_args=connect_args or {},
    )



def test_connection(engine: Engine) -> bool:
    """Return True if the database connection succeeds."""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return True


# ============================================================================
# Schema and table helpers
# ============================================================================


def create_schema_if_not_exists(engine: Engine, schema: str) -> None:
    safe_schema = validate_identifier(schema, label="schema")
    sql = f"CREATE SCHEMA IF NOT EXISTS {q(safe_schema)}"
    with engine.begin() as connection:
        connection.execute(text(sql))



def table_exists(engine: Engine, table_name: str, schema: str = "public") -> bool:
    safe_schema = validate_identifier(schema, label="schema")
    safe_table = validate_identifier(table_name, label="table_name")
    inspector = inspect(engine)
    return inspector.has_table(safe_table, schema=safe_schema)



def get_table_columns(engine: Engine, table_name: str, schema: str = "public") -> list[str]:
    safe_schema = validate_identifier(schema, label="schema")
    safe_table = validate_identifier(table_name, label="table_name")
    inspector = inspect(engine)
    columns = inspector.get_columns(safe_table, schema=safe_schema)
    return [str(column["name"]) for column in columns]



def execute_sql(
    engine: Engine,
    sql: str,
    params: Optional[Mapping[str, Any]] = None,
) -> None:
    """Execute a SQL statement inside a transaction."""
    with engine.begin() as connection:
        connection.execute(text(sql), params or {})



def read_sql_dataframe(
    engine: Engine,
    sql: str,
    params: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Read a SQL query into a pandas DataFrame."""
    with engine.connect() as connection:
        return pd.read_sql_query(text(sql), connection, params=params or {})



def read_table_dataframe(
    engine: Engine,
    table_name: str,
    *,
    schema: str = "public",
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read a database table into a DataFrame with optional projection and filtering.

    Notes:
    - columns are identifier-validated here.
    - where and order_by are treated as raw SQL fragments and should come from
      trusted code, not untrusted user input.
    """
    safe_schema = validate_identifier(schema, label="schema")
    safe_table = validate_identifier(table_name, label="table_name")

    if columns:
        safe_columns = validate_identifiers(columns, label="column_name")
        select_clause = ", ".join(q(column_name) for column_name in safe_columns)
    else:
        select_clause = "*"

    sql_parts = [f"SELECT {select_clause} FROM {q(safe_schema)}.{q(safe_table)}"]

    if where:
        sql_parts.append(f"WHERE {where}")

    if order_by:
        sql_parts.append(f"ORDER BY {order_by}")

    if limit is not None:
        if int(limit) < 0:
            raise ValueError("limit must be >= 0")
        sql_parts.append("LIMIT :_limit")
        params = dict(params or {})
        params["_limit"] = int(limit)

    sql = "\n".join(sql_parts)
    return read_sql_dataframe(engine, sql, params=params)



def delete_where(
    engine: Engine,
    table_name: str,
    *,
    schema: str = "public",
    where: str,
    params: Optional[Mapping[str, Any]] = None,
) -> None:
    """Delete rows from a table using a trusted WHERE clause."""
    if not where or str(where).strip() == "":
        raise ValueError("delete_where requires a non-empty WHERE clause.")

    safe_schema = validate_identifier(schema, label="schema")
    safe_table = validate_identifier(table_name, label="table_name")
    sql = f"DELETE FROM {q(safe_schema)}.{q(safe_table)} WHERE {where}"
    execute_sql(engine, sql, params=params)


# ============================================================================
# DataFrame write helpers
# ============================================================================


def write_dataframe(
    engine: Engine,
    dataframe: pd.DataFrame,
    table_name: str,
    *,
    schema: str = "public",
    if_exists: str = "append",
    chunksize: int = 10_000,
    method: str | None = "multi",
    index: bool = False,
    dtype: Optional[dict[str, Any]] = None,
    create_schema: bool = True,
) -> None:
    """
    Write a DataFrame to PostgreSQL using pandas.to_sql.

    if_exists: 'fail', 'replace', or 'append'
    """
    safe_schema = validate_identifier(schema, label="schema")
    safe_table = validate_identifier(table_name, label="table_name")

    if if_exists not in {"fail", "replace", "append"}:
        raise ValueError("if_exists must be one of: 'fail', 'replace', 'append'")

    if create_schema:
        create_schema_if_not_exists(engine, safe_schema)

    dataframe.to_sql(
        name=safe_table,
        con=engine,
        schema=safe_schema,
        if_exists=if_exists,
        index=index,
        chunksize=chunksize,
        method=method,
        dtype=dtype,
    )



def append_dataframe(
    engine: Engine,
    dataframe: pd.DataFrame,
    table_name: str,
    *,
    schema: str = "public",
    chunksize: int = 10_000,
    method: str | None = "multi",
    index: bool = False,
    dtype: Optional[dict[str, Any]] = None,
    create_schema: bool = True,
) -> None:
    write_dataframe(
        engine=engine,
        dataframe=dataframe,
        table_name=table_name,
        schema=schema,
        if_exists="append",
        chunksize=chunksize,
        method=method,
        index=index,
        dtype=dtype,
        create_schema=create_schema,
    )



def replace_dataframe(
    engine: Engine,
    dataframe: pd.DataFrame,
    table_name: str,
    *,
    schema: str = "public",
    chunksize: int = 10_000,
    method: str | None = "multi",
    index: bool = False,
    dtype: Optional[dict[str, Any]] = None,
    create_schema: bool = True,
) -> None:
    write_dataframe(
        engine=engine,
        dataframe=dataframe,
        table_name=table_name,
        schema=schema,
        if_exists="replace",
        chunksize=chunksize,
        method=method,
        index=index,
        dtype=dtype,
        create_schema=create_schema,
    )


# ============================================================================
# Upsert helpers
# ============================================================================


def ensure_unique_index(
    engine: Engine,
    table_name: str,
    *,
    key_columns: Sequence[str],
    schema: str = "public",
    index_name: Optional[str] = None,
) -> str:
    """
    Ensure a unique index exists for the supplied key columns.

    This is useful for ON CONFLICT support during upserts.
    """
    safe_schema = validate_identifier(schema, label="schema")
    safe_table = validate_identifier(table_name, label="table_name")
    safe_key_columns = validate_identifiers(key_columns, label="key_column")

    if not safe_key_columns:
        raise EmptyKeyColumnsError("key_columns must contain at least one column.")

    if index_name is None:
        joined = "_".join(safe_key_columns)
        index_name = f"uq_{safe_table}_{joined}"

    safe_index_name = validate_identifier(index_name, label="index_name")
    key_list = ", ".join(q(column_name) for column_name in safe_key_columns)

    sql = f"""
    CREATE UNIQUE INDEX IF NOT EXISTS {q(safe_index_name)}
    ON {q(safe_schema)}.{q(safe_table)} ({key_list})
    """
    execute_sql(engine, sql)
    return safe_index_name



def upsert_dataframe(
    engine: Engine,
    dataframe: pd.DataFrame,
    table_name: str,
    *,
    key_columns: Sequence[str],
    schema: str = "public",
    update_columns: Optional[Sequence[str]] = None,
    chunksize: int = 10_000,
    method: str | None = "multi",
    index: bool = False,
    dtype: Optional[dict[str, Any]] = None,
    create_schema: bool = True,
    keep_stage_table: bool = False,
    ensure_conflict_index: bool = True,
) -> None:
    """
    Upsert a DataFrame into PostgreSQL using a temporary staging table.

    Requirements:
    - key_columns must identify a unique row.
    - the target table must support ON CONFLICT on those columns. This helper
      can create a unique index automatically when ensure_conflict_index=True.
    """
    safe_schema = validate_identifier(schema, label="schema")
    safe_table = validate_identifier(table_name, label="table_name")
    safe_key_columns = validate_identifiers(key_columns, label="key_column")

    if not safe_key_columns:
        raise EmptyKeyColumnsError("key_columns must contain at least one column.")

    if dataframe.empty:
        return

    if create_schema:
        create_schema_if_not_exists(engine, safe_schema)

    if not table_exists(engine, safe_table, schema=safe_schema):
        write_dataframe(
            engine=engine,
            dataframe=dataframe,
            table_name=safe_table,
            schema=safe_schema,
            if_exists="fail",
            chunksize=chunksize,
            method=method,
            index=index,
            dtype=dtype,
            create_schema=False,
        )
        if ensure_conflict_index:
            ensure_unique_index(
                engine=engine,
                table_name=safe_table,
                schema=safe_schema,
                key_columns=safe_key_columns,
            )
        return

    target_columns = [str(column_name) for column_name in dataframe.columns]
    safe_target_columns = validate_identifiers(target_columns, label="column_name")

    missing_key_columns = [
        key_column for key_column in safe_key_columns if key_column not in safe_target_columns
    ]
    if missing_key_columns:
        raise PostgresUtilsError(
            f"DataFrame is missing required key column(s): {missing_key_columns}"
        )

    if update_columns is None:
        safe_update_columns = [
            column_name for column_name in safe_target_columns if column_name not in safe_key_columns
        ]
    else:
        safe_update_columns = validate_identifiers(update_columns, label="update_column")

    existing_columns = set(get_table_columns(engine, safe_table, schema=safe_schema))
    missing_target_columns = [
        column_name for column_name in safe_target_columns if column_name not in existing_columns
    ]
    if missing_target_columns:
        raise PostgresUtilsError(
            f"Target table {safe_schema}.{safe_table} is missing DataFrame column(s): {missing_target_columns}"
        )

    if ensure_conflict_index:
        ensure_unique_index(
            engine=engine,
            table_name=safe_table,
            schema=safe_schema,
            key_columns=safe_key_columns,
        )

    stage_table = f"_stage_{safe_table}_{uuid.uuid4().hex[:10]}"

    write_dataframe(
        engine=engine,
        dataframe=dataframe,
        table_name=stage_table,
        schema=safe_schema,
        if_exists="fail",
        chunksize=chunksize,
        method=method,
        index=index,
        dtype=dtype,
        create_schema=False,
    )

    insert_columns_sql = ", ".join(q(column_name) for column_name in safe_target_columns)
    select_columns_sql = ", ".join(q(column_name) for column_name in safe_target_columns)
    conflict_columns_sql = ", ".join(q(column_name) for column_name in safe_key_columns)

    if safe_update_columns:
        update_assignments_sql = ", ".join(
            f"{q(column_name)} = EXCLUDED.{q(column_name)}"
            for column_name in safe_update_columns
        )
        conflict_action_sql = f"DO UPDATE SET {update_assignments_sql}"
    else:
        conflict_action_sql = "DO NOTHING"

    merge_sql = f"""
    INSERT INTO {q(safe_schema)}.{q(safe_table)} ({insert_columns_sql})
    SELECT {select_columns_sql}
    FROM {q(safe_schema)}.{q(stage_table)}
    ON CONFLICT ({conflict_columns_sql})
    {conflict_action_sql}
    """

    drop_stage_sql = f"DROP TABLE IF EXISTS {q(safe_schema)}.{q(stage_table)}"

    try:
        execute_sql(engine, merge_sql)
    finally:
        if not keep_stage_table:
            execute_sql(engine, drop_stage_sql)


# ============================================================================
# Medallion-friendly wrappers
# ============================================================================


def build_medallion_table_name(
    layer: str,
    dataset_name: str,
    artifact_name: str = "dataframe",
) -> str:
    """
    Build a predictable table name for medallion-layer assets.

    Example:
        build_medallion_table_name("silver", "pump", "readings")
        -> silver_pump_readings
    """
    safe_layer = validate_identifier(layer.lower(), label="layer")
    safe_dataset = validate_identifier(dataset_name.lower(), label="dataset_name")
    safe_artifact = validate_identifier(artifact_name.lower(), label="artifact_name")
    return f"{safe_layer}_{safe_dataset}_{safe_artifact}"



def write_layer_dataframe(
    engine: Engine,
    dataframe: pd.DataFrame,
    *,
    layer: str,
    dataset_name: str,
    artifact_name: str = "dataframe",
    schema: str = "public",
    if_exists: str = "append",
    chunksize: int = 10_000,
    method: str | None = "multi",
    index: bool = False,
    dtype: Optional[dict[str, Any]] = None,
    create_schema: bool = True,
) -> str:
    """Write a DataFrame using a standardized medallion-layer table name."""
    table_name = build_medallion_table_name(
        layer=layer,
        dataset_name=dataset_name,
        artifact_name=artifact_name,
    )

    write_dataframe(
        engine=engine,
        dataframe=dataframe,
        table_name=table_name,
        schema=schema,
        if_exists=if_exists,
        chunksize=chunksize,
        method=method,
        index=index,
        dtype=dtype,
        create_schema=create_schema,
    )
    return table_name



def read_layer_dataframe(
    engine: Engine,
    *,
    layer: str,
    dataset_name: str,
    artifact_name: str = "dataframe",
    schema: str = "public",
    columns: Optional[Sequence[str]] = None,
    where: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Read a DataFrame using a standardized medallion-layer table name."""
    table_name = build_medallion_table_name(
        layer=layer,
        dataset_name=dataset_name,
        artifact_name=artifact_name,
    )

    return read_table_dataframe(
        engine=engine,
        table_name=table_name,
        schema=schema,
        columns=columns,
        where=where,
        params=params,
        order_by=order_by,
        limit=limit,
    )


# ============================================================================
# Convenience helpers for notebooks and pipelines
# ============================================================================


def dataframe_row_count(engine: Engine, table_name: str, schema: str = "public") -> int:
    safe_schema = validate_identifier(schema, label="schema")
    safe_table = validate_identifier(table_name, label="table_name")
    sql = f"SELECT COUNT(*) AS row_count FROM {q(safe_schema)}.{q(safe_table)}"
    result = read_sql_dataframe(engine, sql)
    return int(result.loc[0, "row_count"])



def list_tables(engine: Engine, schema: str = "public") -> list[str]:
    safe_schema = validate_identifier(schema, label="schema")
    inspector = inspect(engine)
    return sorted(inspector.get_table_names(schema=safe_schema))



def dispose_engine(engine: Engine) -> None:
    """Explicitly dispose an engine when you are done with it."""
    engine.dispose()
