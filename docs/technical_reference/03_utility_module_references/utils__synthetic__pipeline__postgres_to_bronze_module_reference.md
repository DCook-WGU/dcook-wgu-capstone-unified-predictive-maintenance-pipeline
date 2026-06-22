# Utility Module Reference: `utils/synthetic/pipeline/postgres_to_bronze.py`

## Module Purpose

This module extracts synthetic records from PostgreSQL and prepares them for Bronze ingestion.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module extracts synthetic records from PostgreSQL and prepares them for Bronze ingestion.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `build_engine_from_project_env` | Return a SQLAlchemy engine using the same project utility pattern as the rest of the capstone. | medium |
| `get_table_columns` | Return table columns in ordinal order. | deep |
| `get_sensor_columns` | Return sensor columns in numeric order: sensor_00, sensor_01, ..., sensor_51. | short |
| `_int_or_default` | Convert a nullable SQL aggregate value to int with a default. | deep |
| `read_synthetic_stream_dataframe` | Read the wide synthetic stream table from Postgres. | deep |
| `get_distinct_batch_ids` | Return distinct batch ids from a table. | deep |
| `get_unloaded_source_batch_ids` | Compare source and target tables and return which source batches have not yet been loaded into the target append table. | deep |
| `ensure_handoff_control_table` | Create the append-control table used by the Bronze handoff builder. | deep |
| `get_effective_handoff_offsets` | Resolve next append offsets from control table or target table scan. | deep |
| `upsert_handoff_control_record` | Persist append-control offsets after a Bronze handoff append. | deep |
| `get_handoff_control_record` | Read the active append-control record for a Bronze target table. | deep |
| `get_handoff_append_offsets` | Read the current append target table and return the next starting offsets for ids and time fields. | deep |
| `validate_synthetic_stream_dataframe` | Basic schema validation for the wide synthetic stream table. | deep |
| `choose_sort_columns` | Choose the best available sort columns for one unified sequence. | short |
| `sort_synthetic_stream_dataframe` | Stable sort for one unified ordered stream. | short |
| `add_unified_row_id` | Add a single unified row number across the selected rows. | short |
| `add_unified_episode_id` | Create a batch-safe unified episode id. | short |
| `derive_machine_status` | Convert synthetic labels to the original pump-style machine_status values. | deep |
| `add_synthetic_anomaly_flag` | Optional helper flag for quick sanity checks or downstream testing. | deep |
| `trim_unified_dataframe` | Optional row trimming after the unified table is built. | deep |

## Configuration Dependencies

- Environment variables where runtime mode or optional integration behavior is configured.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `build_engine_from_project_env` | `*, driver, echo` | Return a SQLAlchemy engine using the same project utility pattern as the rest of the capstone. |
| `get_table_columns` | `engine, *, schema, table_name` | Return table columns in ordinal order. |
| `get_sensor_columns` | `columns` | Return sensor columns in numeric order: sensor_00, sensor_01, ..., sensor_51. |
| `_int_or_default` | `value, default` | Convert a nullable SQL aggregate value to int with a default. |
| `read_synthetic_stream_dataframe` | `engine, *, schema, table_name, batch_ids, selected_columns` | Read the wide synthetic stream table from Postgres. |
| `get_distinct_batch_ids` | `engine, *, schema, table_name, batch_column` | Return distinct batch ids from a table. |
| `get_unloaded_source_batch_ids` | `engine, *, source_schema, source_table, target_schema, target_table, requested_batch_ids, batch_column` | Compare source and target tables and return which source batches have not yet been loaded into the target append table. |
| `ensure_handoff_control_table` | `engine, *, schema, table_name` | Create the append-control table used by the Bronze handoff builder. |
| `get_effective_handoff_offsets` | `engine, *, dataset_name, target_schema, target_table, initial_start_timestamp, frequency, control_schema, control_table` | Resolve next append offsets from control table or target table scan. |
| `upsert_handoff_control_record` | `engine, *, dataset_name, target_schema, target_table, last_loaded_batch_id, loaded_batch_count, next_unified_row_id, next_unified_episode_id, next_observation_time_index, next_timestamp, last_append_row_count, last_loaded_batch_ids, last_truth_hash, last_process_run_id, notes, control_schema, control_table` | Persist append-control offsets after a Bronze handoff append. |
| `get_handoff_control_record` | `engine, *, dataset_name, target_schema, target_table, control_schema, control_table` | Read the active append-control record for a Bronze target table. |
| `get_handoff_append_offsets` | `engine, *, schema, table_name, initial_start_timestamp, frequency, unified_row_id_column, unified_episode_id_column, time_index_column, timestamp_column` | Read the current append target table and return the next starting offsets for ids and time fields. |
| `validate_synthetic_stream_dataframe` | `dataframe, *, expected_min_sensor_columns, require_batch_row_order_columns, require_label_source` | Basic schema validation for the wide synthetic stream table. |
| `choose_sort_columns` | `dataframe` | Choose the best available sort columns for one unified sequence. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.

## Artifact / SQL / File-System Interactions

- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.
- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `TypeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`

## Module Importance

This module matters because the synthetic subsystem depends on repeatable staging, truth handling, streaming handoff, or final alignment behavior to make controlled pump telemetry tests explainable.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
