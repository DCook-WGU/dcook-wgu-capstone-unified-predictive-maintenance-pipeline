# Utility Module Reference: `utils/synthetic/pipeline/melt_stage_writer.py`

## Module Purpose

This module validates and writes melted synthetic sensor messages into the pipeline staging layer.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module validates and writes melted synthetic sensor messages into the pipeline staging layer.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `log_step_timing` | Print elapsed time for a melt-stage step and return a new timer. | short |
| `_build_sensor_columns` | Build the expected wide sensor column names for a synthetic observation. | short |
| `_validate_source_columns` | Validate that a timestamped observation frame has melt-stage inputs. | deep |
| `_extract_sensor_index` | Extract the numeric sensor index from names like `sensor_00`. | short |
| `_build_message_sequence_index_with_rng` | Build a randomized 0..(n_sensors-1) sequence for each observation using one shared RNG so chunking stays deterministic across the full run. | short |
| `quote_ident` | Quote a SQL identifier for direct SQLAlchemy text statements. | short |
| `fq_table` | Return a quoted fully qualified table name. | deep |
| `get_table_columns` | Return source table columns in database ordinal order. | deep |
| `ensure_sensor_columns_exist` | Add missing wide sensor columns before SQL-native melting. | deep |
| `build_sensor_messages_stage` | Build the long-format sensor message stage from the timestamped premelt observation stage in chunks instead of loading/melting the full table at once. | deep |
| `build_sensor_messages_stage_sql_native` | Build the long sensor-message stage using a SQL CROSS JOIN LATERAL melt. | deep |
| `validate_sensor_messages_stage` | Return row-count and sensor coverage checks for the melt stage table. | deep |

## Configuration Dependencies

- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `log_step_timing` | `step_name, start_time` | Print elapsed time for a melt-stage step and return a new timer. |
| `_build_sensor_columns` | `n_sensors` | Build the expected wide sensor column names for a synthetic observation. |
| `_validate_source_columns` | `dataframe, required_columns` | Validate that a timestamped observation frame has melt-stage inputs. |
| `_extract_sensor_index` | `sensor_name_series` | Extract the numeric sensor index from names like `sensor_00`. |
| `_build_message_sequence_index_with_rng` | `*, observation_count, sensors_per_observation, rng` | Build a randomized 0..(n_sensors-1) sequence for each observation using one shared RNG so chunking stays deterministic across the full run. |
| `quote_ident` | `identifier` | Quote a SQL identifier for direct SQLAlchemy text statements. |
| `fq_table` | `schema, table_name` | Return a quoted fully qualified table name. |
| `get_table_columns` | `engine, *, schema, table_name` | Return source table columns in database ordinal order. |
| `ensure_sensor_columns_exist` | `engine, *, schema, table_name, sensor_columns` | Add missing wide sensor columns before SQL-native melting. |
| `build_sensor_messages_stage` | `engine, *, schema, source_table, target_table, if_exists, random_seed, n_sensors, chunk_size, enable_memory_logging` | Build the long-format sensor message stage from the timestamped premelt observation stage in chunks instead of loading/melting the full table at once. |
| `build_sensor_messages_stage_sql_native` | `engine, *, schema, source_table, target_table, n_sensors, enable_memory_logging` | Build the long sensor-message stage using a SQL CROSS JOIN LATERAL melt. |
| `validate_sensor_messages_stage` | `engine, *, schema, table_name` | Return row-count and sensor coverage checks for the melt stage table. |

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
