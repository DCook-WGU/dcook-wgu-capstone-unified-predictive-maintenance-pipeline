# Utility Module Reference: `utils/synthetic/pipeline/melt_stage_writer_original.py`

## Module Purpose

This module retains the original melt-stage validation/writer implementation as a legacy support reference.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module retains the original melt-stage validation/writer implementation as a legacy support reference.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_build_sensor_columns` | Build the expected wide sensor column names for the premelt source table. | short |
| `_validate_source_columns` | Confirm that the premelt source table exposes all required columns. | deep |
| `_extract_sensor_index` | Extract numeric sensor indexes from names such as ``sensor_00``. | short |
| `_build_message_sequence_index_with_rng` | Build a randomized 0..(n_sensors-1) sequence for each observation using one shared RNG so chunking stays deterministic across the full run. | short |
| `build_sensor_messages_stage` | Build the long-format sensor message stage from the premelt observation stage in chunks instead of loading/melting the full table at once. | deep |
| `validate_sensor_messages_stage` | Summarize the legacy/support sensor message stage for quick validation. | deep |

## Configuration Dependencies

- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_build_sensor_columns` | `n_sensors` | Build the expected wide sensor column names for the premelt source table. |
| `_validate_source_columns` | `dataframe, required_columns` | Confirm that the premelt source table exposes all required columns. |
| `_extract_sensor_index` | `sensor_name_series` | Extract numeric sensor indexes from names such as ``sensor_00``. |
| `_build_message_sequence_index_with_rng` | `*, observation_count, sensors_per_observation, rng` | Build a randomized 0..(n_sensors-1) sequence for each observation using one shared RNG so chunking stays deterministic across the full run. |
| `build_sensor_messages_stage` | `engine, *, schema, source_table, target_table, if_exists, random_seed, n_sensors, chunk_size` | Build the long-format sensor message stage from the premelt observation stage in chunks instead of loading/melting the full table at once. |
| `validate_sensor_messages_stage` | `engine, *, schema, table_name` | Summarize the legacy/support sensor message stage for quick validation. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.
- Source includes Kafka producer/consumer terminology or calls; helpers participate in synthetic streaming handoff when used by the synthetic pipeline.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
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
