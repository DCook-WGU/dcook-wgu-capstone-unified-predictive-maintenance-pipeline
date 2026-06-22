# Utility Module Reference: `utils/synthetic/generator/postgres_writer.py`

## Module Purpose

This module writes synthetic generator outputs to PostgreSQL staging tables.

## Pipeline Role

- Stage support: Synthetic generator
- Primary responsibility: This module writes synthetic generator outputs to PostgreSQL staging tables.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `scalar_to_int` | Convert a SQL scalar value to int and reject missing sequence values. | deep |
| `ensure_sequence` | Create the Postgres sequence used for synthetic batch or cycle IDs. | deep |
| `reserve_next_batch_id` | Reserve and return the next synthetic batch identifier. | deep |
| `reserve_cycle_range` | Reserve a contiguous global-cycle range and return its first value. | deep |
| `reset_sequence` | Reset a Postgres sequence so the next value starts at start_at. | deep |
| `reset_synthetic_sequences` | Reset the synthetic batch and cycle sequences for one dataset. | deep |
| `_ensure_stream_table_exists` | Create the base stream table if missing. | deep |
| `_get_existing_columns` | Read the current column names for a synthetic stream table. | deep |
| `_infer_alter_column_type` | Infer a Postgres column type for dynamic synthetic stream columns. | deep |
| `_add_missing_columns` | Add dataframe columns that are not yet present in the stream table. | deep |
| `_prepare_dataframe_for_copy` | Sanitize columns and serialize nested values before COPY loading. | short |
| `_copy_dataframe_to_table` | Bulk-load a prepared dataframe into a Postgres table with COPY. | deep |
| `write_stream_batch` | Write a synthetic stream batch to the dataset stream table. | deep |

## Configuration Dependencies

- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `scalar_to_int` | `value, name` | Convert a SQL scalar value to int and reject missing sequence values. |
| `ensure_sequence` | `engine, *, schema, sequence_name` | Create the Postgres sequence used for synthetic batch or cycle IDs. |
| `reserve_next_batch_id` | `engine, *, schema, sequence_name` | Reserve and return the next synthetic batch identifier. |
| `reserve_cycle_range` | `engine, *, schema, sequence_name, n_rows` | Reserve a contiguous global-cycle range and return its first value. |
| `reset_sequence` | `engine, *, schema, sequence_name, start_at` | Reset a Postgres sequence so the next value starts at start_at. |
| `reset_synthetic_sequences` | `engine, *, schema, dataset_name` | Reset the synthetic batch and cycle sequences for one dataset. |
| `_ensure_stream_table_exists` | `engine, *, schema, table` | Create the base stream table if missing. |
| `_get_existing_columns` | `engine, *, schema, table` | Read the current column names for a synthetic stream table. |
| `_infer_alter_column_type` | `series` | Infer a Postgres column type for dynamic synthetic stream columns. |
| `_add_missing_columns` | `engine, *, schema, table, dataframe` | Add dataframe columns that are not yet present in the stream table. |
| `_prepare_dataframe_for_copy` | `dataframe` | Sanitize columns and serialize nested values before COPY loading. |
| `_copy_dataframe_to_table` | `engine, dataframe, *, schema, table` | Bulk-load a prepared dataframe into a Postgres table with COPY. |
| `write_stream_batch` | `engine, dataframe, *, dataset_name, schema, artifact_name, batch_id, cycle_start, use_copy` | Write a synthetic stream batch to the dataset stream table. |

## Side Effects

- Source includes file-write calls; helpers can write configured files or artifacts when those paths are passed by the caller.
- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories and file-write operations.
- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.

## Failure Behavior

- Source raises `TypeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.
- Source raises `finally` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because the synthetic subsystem depends on repeatable staging, truth handling, streaming handoff, or final alignment behavior to make controlled pump telemetry tests explainable.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
