# Utility Module Reference: `utils/synthetic/pipeline/row_rebuilder.py`

## Module Purpose

This module rebuilds wide observation rows from melted or consumed synthetic sensor messages.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module rebuilds wide observation rows from melted or consumed synthetic sensor messages.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_get_existing_columns` | Return existing Postgres columns for a rebuild-stage table. | deep |
| `_infer_alter_column_type` | Infer a conservative Postgres type for an added rebuild column. | deep |
| `_add_missing_columns` | Add dataframe columns that are not present in the rebuild target table. | deep |
| `_validate_consumed_columns` | Validate consumed long-message columns required for observation rebuild. | deep |
| `_build_sensor_columns` | Return the expected wide sensor column names for rebuilt observations. | short |
| `ensure_rebuilt_stage_table_exists` | Create the rebuilt wide-observation table and core rebuild indexes. | deep |
| `load_consumed_messages_for_rebuild` | Load consumed long sensor messages that are eligible for rebuild. | deep |
| `deduplicate_consumed_messages` | Deduplicate at the logical sensor-message level. | short |
| `build_rebuilt_observations_dataframe` | Rebuild wide observations from long consumed messages. | deep |
| `_remove_already_rebuilt_observations` | Drop rebuilt rows whose observation keys already exist in the target table. | deep |
| `write_rebuilt_observations_batch` | Append rebuilt wide observations after removing already-written keys. | deep |
| `mark_consumed_messages_rebuilt` | Mark consumed long rows for rebuilt observations as rebuilt. | deep |
| `rebuild_consumed_messages_to_observations` | Rebuild consumed long sensor messages into wide observation rows. | deep |

## Configuration Dependencies

- Dataset, run, stage, or recipe identifiers used for traceability.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_get_existing_columns` | `engine, *, schema, table` | Return existing Postgres columns for a rebuild-stage table. |
| `_infer_alter_column_type` | `series` | Infer a conservative Postgres type for an added rebuild column. |
| `_add_missing_columns` | `engine, *, schema, table, dataframe` | Add dataframe columns that are not present in the rebuild target table. |
| `_validate_consumed_columns` | `dataframe` | Validate consumed long-message columns required for observation rebuild. |
| `_build_sensor_columns` | `n_sensors` | Return the expected wide sensor column names for rebuilt observations. |
| `ensure_rebuilt_stage_table_exists` | `engine, *, schema, table_name` | Create the rebuilt wide-observation table and core rebuild indexes. |
| `load_consumed_messages_for_rebuild` | `engine, *, schema, source_table, dataset_id, run_id, rebuild_status` | Load consumed long sensor messages that are eligible for rebuild. |
| `deduplicate_consumed_messages` | `dataframe` | Deduplicate at the logical sensor-message level. |
| `build_rebuilt_observations_dataframe` | `dataframe, *, n_sensors, complete_only` | Rebuild wide observations from long consumed messages. |
| `_remove_already_rebuilt_observations` | `engine, *, rebuilt_dataframe, schema, target_table` | Drop rebuilt rows whose observation keys already exist in the target table. |
| `write_rebuilt_observations_batch` | `engine, dataframe, *, schema, table_name` | Append rebuilt wide observations after removing already-written keys. |
| `mark_consumed_messages_rebuilt` | `engine, observation_keys, *, schema, source_table` | Mark consumed long rows for rebuilt observations as rebuilt. |
| `rebuild_consumed_messages_to_observations` | `engine, *, schema, source_table, target_table, dataset_id, run_id, rebuild_status, n_sensors, complete_only, mark_source_rebuilt, observation_window_size` | Rebuild consumed long sensor messages into wide observation rows. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.
- Source includes Kafka producer/consumer terminology or calls; helpers participate in synthetic streaming handoff when used by the synthetic pipeline.

## Artifact / SQL / File-System Interactions

- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.
- Kafka/PostgreSQL handoff: Source references producer, consumer, topic, or streaming-stage behavior.

## Failure Behavior

- Source raises `TypeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because the synthetic subsystem depends on repeatable staging, truth handling, streaming handoff, or final alignment behavior to make controlled pump telemetry tests explainable.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
