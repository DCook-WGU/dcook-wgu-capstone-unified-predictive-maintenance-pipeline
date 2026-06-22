# Utility Module Reference: `utils/database/sql_notebook_helpers.py`

## Module Purpose

This module bridges notebook dataframes, SQL metadata tables, artifact registration, and data-quality event logging.

## Pipeline Role

- Stage support: Database / SQL persistence
- Primary responsibility: This module bridges notebook dataframes, SQL metadata tables, artifact registration, and data-quality event logging.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `safe_sql_identifier` | Validate a SQL identifier before using it in a schema/table reference. | deep |
| `sql_table_ref` | Return a safely quoted schema.table reference. | deep |
| `to_builtin` | Convert pandas/numpy values into JSON-safe Python values. | deep |
| `to_scalar` | Convert values for normal SQL scalar columns. | deep |
| `to_json_string` | Convert Python/pandas/numpy objects to a JSON string for JSONB columns. | deep |
| `row_to_payload` | Convert a dataframe row into a JSON payload dictionary. | short |
| `get_row_value` | Return the first available non-null row value from a list of candidate columns. | short |
| `get_existing_dataframe` | Find the first dataframe in the current notebook globals using candidate names. | deep |
| `execute_many` | Execute a parameterized SQL statement for many rows in chunks. | deep |
| `delete_dataset_run_rows` | Delete existing rows for one dataset/run before writing notebook outputs. | deep |
| `upsert_pipeline_run` | Upsert a capstone.pipeline_runs record for the current notebook/run. | deep |
| `log_data_quality_event` | Insert a data quality event into capstone.data_quality_events. | deep |
| `log_pipeline_artifact` | Insert an artifact record into capstone.pipeline_artifacts. | deep |
| `preview_sql_table` | Preview rows from a SQL table. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.
- Dataset, run, stage, or recipe identifiers used for traceability.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `safe_sql_identifier` | `value` | Validate a SQL identifier before using it in a schema/table reference. |
| `sql_table_ref` | `schema, table` | Return a safely quoted schema.table reference. |
| `to_builtin` | `value` | Convert pandas/numpy values into JSON-safe Python values. |
| `to_scalar` | `value` | Convert values for normal SQL scalar columns. |
| `to_json_string` | `value` | Convert Python/pandas/numpy objects to a JSON string for JSONB columns. |
| `row_to_payload` | `row, exclude_columns` | Convert a dataframe row into a JSON payload dictionary. |
| `get_row_value` | `row, candidate_columns, default` | Return the first available non-null row value from a list of candidate columns. |
| `get_existing_dataframe` | `candidate_names` | Find the first dataframe in the current notebook globals using candidate names. |
| `execute_many` | `sql, rows, *, chunk_size` | Execute a parameterized SQL statement for many rows in chunks. |
| `delete_dataset_run_rows` | `schema, table, *, dataset_id, run_id` | Delete existing rows for one dataset/run before writing notebook outputs. |
| `upsert_pipeline_run` | `*, pipeline_stage, run_status, pipeline_mode, dataset_name, source_system, notes, runtime_facts` | Upsert a capstone.pipeline_runs record for the current notebook/run. |
| `log_data_quality_event` | `*, layer_name, table_name, check_name, check_status, severity, row_count, details_json` | Insert a data quality event into capstone.data_quality_events. |
| `log_pipeline_artifact` | `*, layer_name, stage_name, artifact_name, artifact_type, artifact_path, truth_hash, parent_truth_hash, metadata_json` | Insert an artifact record into capstone.pipeline_artifacts. |
| `preview_sql_table` | `schema, table, limit` | Preview rows from a SQL table. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.
- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `NameError` for invalid input, missing context, or failed validation paths.
- Source raises `RuntimeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.
- Source raises `for` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because SQL persistence and metadata logging must stay consistent across notebook reruns and Medallion handoffs.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
