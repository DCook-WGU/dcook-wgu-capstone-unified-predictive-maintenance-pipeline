# Utility Module Reference: `utils/synthetic/pipeline/send_queue_stage_writer.py`

## Module Purpose

This module writes send-queue records that feed the synthetic Kafka producer path.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module writes send-queue records that feed the synthetic Kafka producer path.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `log_step_timing` | Print elapsed time for a send-queue build step and return a fresh timer. | short |
| `_validate_source_columns` | Raise a clear error when the staged sensor table is missing queue inputs. | deep |
| `_ensure_send_queue_runtime_columns` | Add runtime queue columns used by producer claiming and acknowledgements. | deep |
| `_ensure_send_queue_indexes` | Create indexes used by status checks, claim ordering, and message lookup. | deep |
| `_apply_send_queue_owner_and_grants` | Grant the producer role ownership and DML access to the send queue table. | deep |
| `build_sensor_messages_send_queue` | Build the send queue from staged sensor messages using chunked pandas writes. | deep |
| `build_sensor_messages_send_queue_sql_native` | Build the synthetic sensor message send queue directly inside Postgres. | deep |
| `validate_sensor_messages_send_queue` | Return row, timestamp, sensor, and pending-message checks for the send queue. | deep |

## Configuration Dependencies

- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `log_step_timing` | `step_name, start_time` | Print elapsed time for a send-queue build step and return a fresh timer. |
| `_validate_source_columns` | `dataframe, required_columns` | Raise a clear error when the staged sensor table is missing queue inputs. |
| `_ensure_send_queue_runtime_columns` | `engine, *, schema, table_name` | Add runtime queue columns used by producer claiming and acknowledgements. |
| `_ensure_send_queue_indexes` | `engine, *, schema, table_name` | Create indexes used by status checks, claim ordering, and message lookup. |
| `_apply_send_queue_owner_and_grants` | `engine, *, schema, table_name, owner_role` | Grant the producer role ownership and DML access to the send queue table. |
| `build_sensor_messages_send_queue` | `engine, *, schema, source_table, target_table, if_exists, queue_status_default, chunk_size, queue_owner_role, apply_owner_and_grants` | Build the send queue from staged sensor messages using chunked pandas writes. |
| `build_sensor_messages_send_queue_sql_native` | `engine, *, schema, source_table, target_table, if_exists, queue_status_default, queue_owner_role, apply_owner_and_grants, enable_timing_logging` | Build the synthetic sensor message send queue directly inside Postgres. |
| `validate_sensor_messages_send_queue` | `engine, *, schema, table_name` | Return row, timestamp, sensor, and pending-message checks for the send queue. |

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
