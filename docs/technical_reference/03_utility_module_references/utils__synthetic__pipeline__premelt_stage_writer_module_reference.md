# Utility Module Reference: `utils/synthetic/pipeline/premelt_stage_writer.py`

## Module Purpose

This module writes wide synthetic observations to the premelt staging layer before timestamping and melting.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module writes wide synthetic observations to the premelt staging layer before timestamping and melting.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_get_table_columns` | Return source table columns in database ordinal order. | deep |
| `_validate_source_columns` | Validate that the synthetic stream table has required premelt inputs. | deep |
| `_build_select_sql` | Build the SQL SELECT that assigns observation identity and metadata. | deep |
| `_write_stage_sql_native` | Create, append to, or fail on the premelt target table in Postgres. | deep |
| `scalar_to_int` | Convert a scalar SQL result to int and reject missing values. | deep |
| `dataframe_row_count_to_int` | Return a count value from a one-row dataframe as a plain int. | short |
| `build_observations_premelt_stage` | Build the premelt stage directly inside Postgres. | deep |
| `validate_observations_premelt_stage` | Return row-count and identity checks for the premelt stage table. | deep |

## Configuration Dependencies

- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_get_table_columns` | `engine, *, schema, table_name` | Return source table columns in database ordinal order. |
| `_validate_source_columns` | `columns, required_columns` | Validate that the synthetic stream table has required premelt inputs. |
| `_build_select_sql` | `*, safe_schema, safe_source_table, remaining_source_columns` | Build the SQL SELECT that assigns observation identity and metadata. |
| `_write_stage_sql_native` | `engine, *, schema, target_table, select_sql, params, if_exists` | Create, append to, or fail on the premelt target table in Postgres. |
| `scalar_to_int` | `value, name` | Convert a scalar SQL result to int and reject missing values. |
| `dataframe_row_count_to_int` | `dataframe, *, column` | Return a count value from a one-row dataframe as a plain int. |
| `build_observations_premelt_stage` | `engine, *, schema, source_table, target_table, dataset_id, run_id, asset_id, if_exists` | Build the premelt stage directly inside Postgres. |
| `validate_observations_premelt_stage` | `engine, *, schema, table_name` | Return row-count and identity checks for the premelt stage table. |

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
