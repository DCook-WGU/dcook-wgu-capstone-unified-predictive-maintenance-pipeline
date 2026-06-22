# utils/synthetic/pipeline/row_rebuilder.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `row_rebuilder.py` that need deeper explanation than the 071d module-level reference. The selected function rebuilds wide observation rows from consumed long-form Kafka sensor messages and manages the rebuilt-stage handoff state.

## Source Grounding

Sources used:

- `utils/synthetic/pipeline/row_rebuilder.py`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__synthetic__pipeline__row_rebuilder_module_reference.md`
- `function_inventory.json`
- `technical_reference/01_notebook_workflow_references/`
- `technical_reference/00_project_manual/`
- Documentation standards under `docs_agent_pack/00_STANDARDS/`

The active utility source file is the source of truth.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
| -------- | -------------- | ------------------------ |
| `rebuild_consumed_messages_to_observations` | Rebuilds long consumed sensor messages into wide observation rows | Synthetic consumed-message rebuild stage |

## Module-Level Technical Context

`row_rebuilder.py` turns consumed long-form Kafka messages back into wide pump observations. It validates consumed-message columns, deduplicates by logical sensor-message identity, pivots sensor values into `sensor_00` through `sensor_51`, writes rebuilt observations, and optionally marks source consumed messages as rebuilt.

## Deep Function References

### `rebuild_consumed_messages_to_observations`

#### Functional Purpose

`rebuild_consumed_messages_to_observations` orchestrates rebuilding consumed sensor messages into wide observation rows. It processes bounded observation-index windows, deduplicates long messages, rebuilds complete observations, writes them to a rebuilt-stage table, and optionally marks source consumed messages as rebuilt.

#### Pipeline Context

Active notebook source confirms use in the synthetic row rebuilder notebook and in a condensed synthetic pipeline notebook. It supports the handoff from consumed Kafka message rows to wide observation rows that can later be aligned and handed to Bronze.

#### Inputs and Assumptions

Important inputs include the SQLAlchemy engine, schema, consumed source table, rebuilt target table, optional `dataset_id`, optional `run_id`, rebuild status filter, expected sensor count, `complete_only`, `mark_source_rebuilt`, and observation window size.

The source table must contain consumed long-message columns including dataset/run/asset identity, `message_key`, generated row id, observation index/timestamp, stream state, phase, sensor name/index/value, fault metadata, consumer receive time, Kafka topic/partition/offset, and `rebuild_status`.

#### Outputs and Return Contract

The function returns a stats dictionary with source-confirmed fields:

- `status`
- `consumed_rows`
- `deduped_rows`
- `rebuilt_rows`
- `rebuilt_observations`
- `updated_source_observation_groups`
- `target_table`

Status starts as `empty`, becomes `rebuilt` when rows are written, and becomes `no_complete_observations` when source rows were consumed but no rebuilt rows were written.

#### Side Effects

Confirmed side effects are:

- Reads consumed message windows from PostgreSQL.
- Creates or verifies the rebuilt target table through supporting write helpers.
- Adds missing target columns when rebuilt data contains columns not yet present.
- Appends rebuilt wide observations through `write_layer_dataframe`.
- Skips observations already present in the target by observation key.
- Optionally updates source consumed-message rows from `pending` to `rebuilt` for rebuilt observation keys.

#### Failure Behavior and Guardrails

The function resolves dataset/run from the source table when not provided. Supporting helpers raise `ValueError` if consumed source columns are missing. Empty windows are skipped. Incomplete observations are excluded when `complete_only=True`. Database failures propagate from SQL and write helpers.

#### Lineage, Idempotency, and Reproducibility Role

Logical observation identity is `dataset_id`, `run_id`, `asset_id`, and `observation_index`. Sensor-message deduplication keeps the latest received row for each dataset/run/asset/observation/sensor identity based on receive and Kafka ordering fields. The target write avoids duplicate rebuilt observation keys, and optional source marking records rebuild completion at the observation group level.

#### Why This Function Matters

The streaming path sends one sensor message at a time, but Bronze-style processing expects wide observation rows. This function protects the shape transition by checking completeness, preserving identity keys, and making rebuild state visible in PostgreSQL.

#### Verification Method

- Confirm consumed rows with all expected sensor indexes rebuild into one wide observation per observation key.
- Confirm `rebuild_sensor_count` equals the expected sensor count for complete observations.
- Confirm incomplete observations are excluded when `complete_only=True`.
- Confirm reruns do not duplicate observations already present in the rebuilt target.
- Confirm source rows are marked `rebuilt` when `mark_source_rebuilt=True`.

## Cross-Function Relationships

This function consumes rows produced by Kafka consumer landing stages and prepares the wide observation shape needed by later final-alignment and Bronze handoff utilities. It is downstream of queue publication and upstream of final aligned observation handoff.

## Source-Limited Items

- The exact consumer script or notebook that writes the consumed source table is Not determined from available source in this function reference.
- This function does not publish Kafka messages or write directly to Bronze from available source.
