from __future__ import annotations

import os
import re
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# -----------------------------------------------------------------------------
# Identifier helpers
# -----------------------------------------------------------------------------

def sanitize_sql_identifier(name: str) -> str:
    """
    Convert an arbitrary string into a conservative SQL identifier.

    Rules:
    - lowercase
    - non alphanumeric chars become underscores
    - repeated underscores collapse
    - leading/trailing underscores are removed
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()

    if not cleaned:
        raise ValueError("Identifier resolved to an empty string after sanitization.")

    return cleaned


# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------

def _get_first_env_value(names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


# -----------------------------------------------------------------------------
# Engine / connection helpers
# -----------------------------------------------------------------------------

def build_postgres_url(
    *,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
    driver: str = "psycopg2",
) -> str:
    """
    Build a SQLAlchemy Postgres URL.

    IMPORTANT:
    User/password must be URL-encoded because they may contain characters like '@', '!', ':'.
    """
    from urllib.parse import quote_plus

    if not all([host, port, database, user, password]):
        raise ValueError("host, port, database, user, and password are required.")

    user_q = quote_plus(str(user))
    pass_q = quote_plus(str(password))
    db_q = quote_plus(str(database))

    return f"postgresql+{driver}://{user_q}:{pass_q}@{host}:{int(port)}/{db_q}"



def get_engine(
    *,
    postgres_url: Optional[str] = None,
    host: Optional[str] = None,
    port: int = 5432,
    database: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    driver: str = "psycopg2",
    echo: bool = False,
) -> Engine:
    """
    Create a SQLAlchemy engine.

    Either pass postgres_url directly, or pass host / port / database / user /
    password and let this helper construct the URL.
    """
    if postgres_url is None:
        postgres_url = build_postgres_url(
            host=str(host),
            port=int(port),
            database=str(database),
            user=str(user),
            password=str(password),
            driver=driver,
        )

    return create_engine(postgres_url, future=True, echo=echo, pool_pre_ping=True)



def get_engine_from_env(
    *,
    host_env_names: Sequence[str] = ("DB_HOST", "POSTGRES_HOST"),
    port_env_names: Sequence[str] = ("DB_PORT", "POSTGRES_PORT"),
    database_env_names: Sequence[str] = ("DB_NAME", "POSTGRES_DB", "POSTGRES_DATABASE"),
    user_env_names: Sequence[str] = ("DB_USER", "POSTGRES_USER"),
    password_env_names: Sequence[str] = ("DB_PASSWORD", "POSTGRES_PASSWORD"),
    driver: str = "psycopg2",
    echo: bool = False,
) -> Engine:
    """
    Create a SQLAlchemy engine from environment variables.

    Default resolution order supports both your Docker-style DB_* variables and
    a more standard POSTGRES_* naming convention.
    """
    host = _get_first_env_value(host_env_names)
    port_raw = _get_first_env_value(port_env_names) or "5432"
    database = _get_first_env_value(database_env_names)
    user = _get_first_env_value(user_env_names)
    password = _get_first_env_value(password_env_names)

    missing = []
    if host is None:
        missing.append("host")
    if database is None:
        missing.append("database")
    if user is None:
        missing.append("user")
    if password is None:
        missing.append("password")

    if missing:
        raise RuntimeError(
            "Missing required Postgres environment values for: "
            + ", ".join(missing)
            + ". Checked DB_* and POSTGRES_* names by default."
        )

    try:
        port = int(port_raw)
    except Exception as exc:
        raise RuntimeError(f"Invalid Postgres port value: {port_raw!r}") from exc

    return get_engine(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        driver=driver,
        echo=echo,
    )


# -----------------------------------------------------------------------------
# Low-level SQL helpers
# -----------------------------------------------------------------------------

def create_schema_if_not_exists(engine: Engine, schema: str) -> str:
    safe_schema = sanitize_sql_identifier(schema)
    with engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{safe_schema}"'))
    return safe_schema



def execute_sql(engine: Engine, sql: str, params: Optional[Mapping[str, Any]] = None) -> None:
    with engine.begin() as connection:
        connection.execute(text(sql), params or {})



def read_sql_dataframe(
    engine: Engine,
    sql: str,
    params: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    with engine.begin() as connection:
        return pd.read_sql(text(sql), connection, params=params or {})



def table_exists(engine: Engine, *, schema: str, table_name: str) -> bool:
    safe_schema = sanitize_sql_identifier(schema)
    safe_table = sanitize_sql_identifier(table_name)

    sql = """
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = :schema_name
          AND table_name = :table_name
    ) AS table_exists
    """
    result = read_sql_dataframe(
        engine,
        sql,
        params={"schema_name": safe_schema, "table_name": safe_table},
    )
    return bool(result.loc[0, "table_exists"])


__all__ = [
    "sanitize_sql_identifier",
    "build_postgres_url",
    "get_engine",
    "get_engine_from_env",
    "create_schema_if_not_exists",
    "execute_sql",
    "read_sql_dataframe",
    "table_exists",
]
