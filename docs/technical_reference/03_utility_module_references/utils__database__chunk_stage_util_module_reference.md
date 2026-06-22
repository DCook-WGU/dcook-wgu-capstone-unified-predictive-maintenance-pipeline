# Utility Module Reference: `utils/database/chunk_stage_util.py`

## Module Purpose

This module supports chunk/window-oriented database staging work for larger notebook or pipeline transfers.

## Pipeline Role

- Stage support: Database / SQL persistence
- Primary responsibility: This module supports chunk/window-oriented database staging work for larger notebook or pipeline transfers.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `memory_gb` | Return the current process resident memory usage in gigabytes. | short |
| `log_memory` | Print a labeled memory snapshot for long-running chunked notebook steps. | deep |
| `copy_sql_params` | Copy SQL query parameters into a mutable string-keyed dictionary. | deep |
| `get_table_columns` | Return column names for a Postgres table without reading data rows. | deep |
| `resolve_dataset_run_from_table` | Resolve a single dataset_id/run_id pair from parameters or table contents. | deep |
| `get_table_row_count` | Count rows in a Postgres table, optionally using a caller-supplied WHERE clause. | deep |
| `read_table_chunk_by_row_number` | Read one deterministic row-number window from a Postgres table. | deep |
| `process_postgres_table_in_chunks` | Stream a Postgres table through transform and write callbacks in row chunks. | deep |
| `get_observation_index_bounds` | Return the min/max observation_index for one dataset/run filter. | deep |
| `read_table_for_observation_window` | Read rows for one inclusive observation_index window and dataset/run pair. | deep |
| `process_observation_index_windows` | Process a dataset/run table in observation_index windows. | deep |

## Configuration Dependencies

- Dataset, run, stage, or recipe identifiers used for traceability.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `memory_gb` | `No explicit parameters` | Return the current process resident memory usage in gigabytes. |
| `log_memory` | `label` | Print a labeled memory snapshot for long-running chunked notebook steps. |
| `copy_sql_params` | `params` | Copy SQL query parameters into a mutable string-keyed dictionary. |
| `get_table_columns` | `engine, *, schema_name, table_name` | Return column names for a Postgres table without reading data rows. |
| `resolve_dataset_run_from_table` | `engine, *, schema_name, table_name, dataset_id, run_id, where_sql, params` | Resolve a single dataset_id/run_id pair from parameters or table contents. |
| `get_table_row_count` | `engine, *, schema_name, table_name, where_sql, params` | Count rows in a Postgres table, optionally using a caller-supplied WHERE clause. |
| `read_table_chunk_by_row_number` | `engine, *, schema_name, table_name, select_columns, order_by_sql, start_row, chunk_size, where_sql, params` | Read one deterministic row-number window from a Postgres table. |
| `process_postgres_table_in_chunks` | `engine, *, schema_name, table_name, select_columns, order_by_sql, transform_chunk_func, write_chunk_func, chunk_size, where_sql, params, enable_memory_logging` | Stream a Postgres table through transform and write callbacks in row chunks. |
| `get_observation_index_bounds` | `engine, *, schema_name, table_name, dataset_id, run_id, extra_where_sql, params` | Return the min/max observation_index for one dataset/run filter. |
| `read_table_for_observation_window` | `engine, *, schema_name, table_name, select_columns, dataset_id, run_id, observation_index_min, observation_index_max, extra_where_sql, params, order_by_sql` | Read rows for one inclusive observation_index window and dataset/run pair. |
| `process_observation_index_windows` | `engine, *, schema_name, table_name, select_columns, dataset_id, run_id, transform_chunk_func, write_chunk_func, window_size, extra_where_sql, params, order_by_sql` | Process a dataset/run table in observation_index windows. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.

## Artifact / SQL / File-System Interactions

- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.

## Failure Behavior

- Source raises `TypeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because SQL persistence and metadata logging must stay consistent across notebook reruns and Medallion handoffs.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
