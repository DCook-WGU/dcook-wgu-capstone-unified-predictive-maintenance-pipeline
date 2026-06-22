# Utility Module Reference: `utils/database/postgres.py`

## Module Purpose

This module contains PostgreSQL connection and execution helpers used by notebooks and SQL-facing utilities.

## Pipeline Role

- Stage support: Database / SQL persistence
- Primary responsibility: This module contains PostgreSQL connection and execution helpers used by notebooks and SQL-facing utilities.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`, `EDA_Notebook_Pump_Silver_01_PreEDA`, `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `sanitize_sql_identifier` | Convert an arbitrary string into a conservative SQL identifier. | deep |
| `_get_first_env_value` | Return the first non-empty environment variable value from an ordered list. | short |
| `build_postgres_url` | Build a SQLAlchemy Postgres URL. | deep |
| `get_engine` | Create a SQLAlchemy engine. | deep |
| `get_engine_from_env` | Create a SQLAlchemy engine from environment variables. | deep |
| `create_schema_if_not_exists` | Create a sanitized Postgres schema when absent and return its safe name. | deep |
| `execute_sql` | Execute one SQL statement inside a managed transaction. | deep |
| `read_sql_dataframe` | Execute a SQL query and return the result as a pandas DataFrame. | deep |
| `table_exists` | Return whether a sanitized schema/table exists in Postgres. | deep |

## Configuration Dependencies

- Environment variables where runtime mode or optional integration behavior is configured.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `sanitize_sql_identifier` | `name` | Convert an arbitrary string into a conservative SQL identifier. |
| `_get_first_env_value` | `names` | Return the first non-empty environment variable value from an ordered list. |
| `build_postgres_url` | `*, host, port, database, user, password, driver` | Build a SQLAlchemy Postgres URL. |
| `get_engine` | `*, postgres_url, host, port, database, user, password, driver, echo` | Create a SQLAlchemy engine. |
| `get_engine_from_env` | `*, host_env_names, port_env_names, database_env_names, user_env_names, password_env_names, driver, echo` | Create a SQLAlchemy engine from environment variables. |
| `create_schema_if_not_exists` | `engine, schema` | Create a sanitized Postgres schema when absent and return its safe name. |
| `execute_sql` | `engine, sql, params` | Execute one SQL statement inside a managed transaction. |
| `read_sql_dataframe` | `engine, sql, params` | Execute a SQL query and return the result as a pandas DataFrame. |
| `table_exists` | `engine, *, schema, table_name` | Return whether a sanitized schema/table exists in Postgres. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.

## Artifact / SQL / File-System Interactions

- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.

## Failure Behavior

- Source raises `RuntimeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`, `EDA_Notebook_Pump_Silver_01_PreEDA`, `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`

## Module Importance

This module matters because SQL persistence and metadata logging must stay consistent across notebook reruns and Medallion handoffs.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
