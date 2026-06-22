# Utility Module Reference: `utils/synthetic/pipeline/timestamp_stage_writer.py`

## Module Purpose

This module adds and persists timestamped synthetic observations before queueing or melting.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module adds and persists timestamped synthetic observations before queueing or melting.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `ensure_simulation_timing_config_table` | Create the timing configuration table used to timestamp observations. | deep |
| `insert_simulation_timing_config` | Insert or update the simulation timing configuration for a dataset/run. | deep |
| `load_simulation_timing_config` | Load the active timing configuration for one dataset/run pair. | deep |
| `_get_table_columns` | Return source table columns in database ordinal order. | deep |
| `_validate_source_columns` | Validate that the premelt table has timestamp-stage inputs. | deep |
| `_build_select_sql` | Build SQL that derives observation timestamps from timing config. | deep |
| `_write_stage_sql_native` | Create, append to, or fail on the timestamp target table in Postgres. | deep |
| `scalar_to_int` | Convert a scalar SQL result to int and reject missing values. | deep |
| `dataframe_row_count_to_int` | Return a count value from a one-row dataframe as a plain int. | short |
| `build_observations_timestamped_stage` | Build the timestamped stage directly inside Postgres. | deep |
| `validate_observations_timestamped_stage` | Return row-count and timestamp-range checks for the timestamped stage. | deep |
| `build_sensor_messages_timestamped_stage` | Backward-compatible alias for `build_observations_timestamped_stage`. | medium |
| `validate_sensor_messages_timestamped_stage` | Backward-compatible alias for `validate_observations_timestamped_stage`. | deep |

## Configuration Dependencies

- Resolved pipeline configuration dictionaries and YAML-derived stage settings.
- Dataset, run, stage, or recipe identifiers used for traceability.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `ensure_simulation_timing_config_table` | `engine, *, schema, table_name` | Create the timing configuration table used to timestamp observations. |
| `insert_simulation_timing_config` | `*, engine, dataset_id, run_id, simulation_start_datetime, sampling_interval_seconds, schema, table_name, set_active, deactivate_existing_for_run` | Insert or update the simulation timing configuration for a dataset/run. |
| `load_simulation_timing_config` | `engine, *, dataset_id, run_id, schema, table_name` | Load the active timing configuration for one dataset/run pair. |
| `_get_table_columns` | `engine, *, schema, table_name` | Return source table columns in database ordinal order. |
| `_validate_source_columns` | `columns` | Validate that the premelt table has timestamp-stage inputs. |
| `_build_select_sql` | `*, safe_schema, safe_source_table, remaining_source_columns` | Build SQL that derives observation timestamps from timing config. |
| `_write_stage_sql_native` | `engine, *, schema, target_table, select_sql, params, if_exists` | Create, append to, or fail on the timestamp target table in Postgres. |
| `scalar_to_int` | `value, name` | Convert a scalar SQL result to int and reject missing values. |
| `dataframe_row_count_to_int` | `dataframe, *, column` | Return a count value from a one-row dataframe as a plain int. |
| `build_observations_timestamped_stage` | `engine, *, schema, source_table, target_table, timing_config_table, dataset_id, run_id, if_exists, chunk_size` | Build the timestamped stage directly inside Postgres. |
| `validate_observations_timestamped_stage` | `engine, *, schema, table_name` | Return row-count and timestamp-range checks for the timestamped stage. |
| `build_sensor_messages_timestamped_stage` | `*args, **kwargs` | Backward-compatible alias for `build_observations_timestamped_stage`. |
| `validate_sensor_messages_timestamped_stage` | `*args, **kwargs` | Backward-compatible alias for `validate_observations_timestamped_stage`. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.
- Source includes Kafka producer/consumer terminology or calls; helpers participate in synthetic streaming handoff when used by the synthetic pipeline.

## Artifact / SQL / File-System Interactions

- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.
- Kafka/PostgreSQL handoff: Source references producer, consumer, topic, or streaming-stage behavior.

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because the synthetic subsystem depends on repeatable staging, truth handling, streaming handoff, or final alignment behavior to make controlled pump telemetry tests explainable.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
