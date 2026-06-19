# Database Utility Reference: sql_notebook_helpers.py

Source path:

`utils/database/sql_notebook_helpers.py`

## Purpose

Provides small notebook-facing helpers for PostgreSQL identifiers, JSON-safe value conversion, SQL execution, metadata logging, artifact metadata logging, and table previews.

This module supports notebooks that need lightweight SQL persistence or inspection without replacing the normal file-based artifact pipeline.

## Pipeline Role

This module is used by notebooks and SQL writer utilities that need common PostgreSQL helper behavior:

- Build safe schema/table references.
- Convert pandas and numpy values for SQL scalar or JSONB columns.
- Find notebook dataframes by expected variable names.
- Execute parameterized SQL statements in chunks.
- Delete dataset/run rows before reruns.
- Record pipeline run, data quality, and artifact metadata.
- Preview SQL tables for notebook inspection.

## Configuration and Environment Behavior

The module creates a SQLAlchemy engine at import time with `get_engine_from_env()`.

The metadata context is resolved as follows:

| Value | Source |
|---|---|
| `CAPSTONE_SCHEMA` | `CAPSTONE_SCHEMA` environment variable, defaulting to `capstone` |
| `DATASET_ID` | `DATASET_ID` environment variable, then notebook `DATASET_NAME`, then `pump` |
| `RUN_ID` | `RUN_ID` environment variable, then notebook `RUN_ID`, then `run_001` |

These defaults make the helpers convenient in notebooks, but final notebook runs should still set clear dataset and run identifiers through the project configuration flow.

## Main Functions

| Function | Main Inputs | Purpose |
|---|---|---|
| `safe_sql_identifier` | schema or table name | Validate a SQL identifier before quoting |
| `sql_table_ref` | schema, table | Return a quoted `"schema"."table"` reference |
| `to_builtin` | Python, pandas, or numpy value | Convert values for JSON serialization |
| `to_scalar` | Python, pandas, or numpy value | Convert values for SQL scalar binding |
| `to_json_string` | object | Serialize a value for PostgreSQL JSONB |
| `row_to_payload` | pandas row | Convert a dataframe row to a JSON-safe payload |
| `get_row_value` | pandas row, candidate columns | Read the first available non-null value |
| `get_existing_dataframe` | notebook variable names | Find a dataframe in notebook globals |
| `execute_many` | SQL string, row dictionaries | Execute parameterized SQL in chunks |
| `delete_dataset_run_rows` | schema, table, dataset/run ids | Delete rerun rows for idempotent writes |
| `upsert_pipeline_run` | stage metadata | Upsert one run record |
| `log_data_quality_event` | check metadata | Insert one data quality event |
| `log_pipeline_artifact` | artifact metadata | Insert one artifact metadata record |
| `preview_sql_table` | schema, table, row limit | Return a preview dataframe |

## Database and Table Assumptions

The metadata helpers assume the configured capstone schema contains:

- `pipeline_runs`
- `data_quality_events`
- `pipeline_artifacts`

The preview and delete helpers can target other schemas and tables when the caller provides safe identifiers.

## SQL Side Effects

- `execute_many` executes the caller-provided parameterized SQL through the module-level engine.
- `delete_dataset_run_rows` deletes rows matching a dataset/run pair from the requested table.
- `upsert_pipeline_run` inserts or updates one row in `CAPSTONE_SCHEMA.pipeline_runs`.
- `log_data_quality_event` inserts one row in `CAPSTONE_SCHEMA.data_quality_events`.
- `log_pipeline_artifact` inserts one row in `CAPSTONE_SCHEMA.pipeline_artifacts`.
- `preview_sql_table` reads rows only.

The module does not reset databases, drop tables, rebuild infrastructure, or create artifact files.

## Logging, Ledger, and Artifact Behavior

### Logging

The module uses print-style notebook messages for dataframe lookup, row writes, deletes, and metadata logging.

### Ledger

The module does not write project ledger entries directly. It records SQL-facing metadata in PostgreSQL tables.

### Artifacts

`log_pipeline_artifact` records metadata about artifacts created elsewhere. It does not create, copy, move, or validate artifact files.

## Notebook Usage Context

The helpers are designed for notebook cells that need a small SQL action after a dataframe or artifact path already exists.

The module-level engine and identity values are convenient for notebook usage, but shared utility writers can also use the lower-level value conversion and identifier helpers directly.

## Return Values

- Identifier and conversion helpers return strings or normalized Python values.
- `row_to_payload` returns a JSON-safe dictionary.
- `get_existing_dataframe` returns a dataframe copy.
- `execute_many` and `delete_dataset_run_rows` return row counts.
- Metadata loggers return `None`.
- `preview_sql_table` returns a pandas dataframe.

## Common Failure Points

- Unsafe schema or table names raise `ValueError`.
- Missing notebook dataframe variables raise `NameError`.
- PostgreSQL connection settings must be available to build the module-level engine.
- Metadata tables must exist before metadata logging helpers are called.

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted database tests that use this module.
