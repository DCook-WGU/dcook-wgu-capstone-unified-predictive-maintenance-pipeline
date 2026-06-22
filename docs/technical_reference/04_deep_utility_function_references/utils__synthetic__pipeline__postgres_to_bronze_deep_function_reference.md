# utils/synthetic/pipeline/postgres_to_bronze.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `postgres_to_bronze.py` that need deeper explanation than the 071d module-level reference. The selected function reads synthetic PostgreSQL output and prepares an in-memory Bronze-ready dataframe.

## Source Grounding

Sources used:

- `utils/synthetic/pipeline/postgres_to_bronze.py`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__synthetic__pipeline__postgres_to_bronze_module_reference.md`
- `function_inventory.json`
- `technical_reference/01_notebook_workflow_references/`
- `technical_reference/00_project_manual/`
- Documentation standards under `docs_agent_pack/00_STANDARDS/`

The active utility source file is the source of truth.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
| -------- | -------------- | ------------------------ |
| `build_bronze_handoff_from_postgres` | Reads wide synthetic stream rows and returns a Bronze-ready dataframe | Synthetic PostgreSQL to Bronze handoff preparation |

## Module-Level Technical Context

`postgres_to_bronze.py` prepares synthetic PostgreSQL output so it can resemble the original pump sensor dataset expected by the Bronze preprocessing path. It validates the wide stream table, chooses stable sort order, creates unified row and episode identifiers, derives `machine_status`, assigns contiguous time fields, and selects the final handoff columns.

## Deep Function References

### `build_bronze_handoff_from_postgres`

#### Functional Purpose

`build_bronze_handoff_from_postgres` reads a wide synthetic stream table from PostgreSQL and returns a Bronze-ready dataframe. It is an in-memory preparation function, not a database write function.

#### Pipeline Context

The module reference and project manual context identify this module as the bridge from synthetic PostgreSQL output into Bronze ingestion. Active notebook or script usage of this specific function is Not determined from available source; the source confirms the function contract and the module's Bronze handoff role.

#### Inputs and Assumptions

Important inputs include the SQLAlchemy engine, source schema, source table, optional batch ids, start timestamp, frequency, target row trim settings, lineage-column retention flag, anomaly flag option, and whether to keep other columns.

The source table must be a wide synthetic stream table with sensor columns and enough ordering and label context for validation. Validation requires at least one `sensor_XX` column, ordering columns such as `global_cycle_id` or `batch_id` plus `row_in_batch`, and either `stream_state` or `phase`.

#### Outputs and Return Contract

The function returns a pandas dataframe prepared for Bronze handoff. Source-confirmed output preparation includes:

- Stable sorting.
- `unified_row_id`.
- Unified episode id when `meta__episode_id` is present.
- `machine_status` derived from `stream_state` or `phase`.
- Optional `anomaly_flag__synthetic`.
- Optional trimming.
- Contiguous `observation_time_index` and `timestamp`.
- Final column selection with timestamp, sensor columns, machine status, and optional lineage/other columns.

#### Side Effects

Confirmed side effects are PostgreSQL reads through `read_synthetic_stream_dataframe`. No SQL writes, Kafka publishes, file writes, artifacts, or ledger updates are performed by this function from available source.

#### Failure Behavior and Guardrails

Supporting validation raises `TypeError` when the raw dataframe is not a pandas dataframe and `ValueError` when required sensor, ordering, or label inputs are missing. `trim_unified_dataframe` raises `ValueError` for non-positive target row counts or unsupported trim modes. SQL read failures propagate from the database helper.

#### Lineage, Idempotency, and Reproducibility Role

The function can preserve lineage columns when `keep_lineage_columns=True`. It creates a deterministic unified sequence from source ordering columns, assigns contiguous time fields from the configured start timestamp and frequency, and maps synthetic states to original-style `machine_status` values. It does not track loaded batch state; append-aware behavior is handled by separate functions in the module.

#### Why This Function Matters

Bronze preprocessing expects a pump-like dataframe shape. This function makes synthetic PostgreSQL output conform to that shape while preserving optional synthetic lineage fields for review.

#### Verification Method

- Confirm returned dataframe contains timestamp, sensor columns, and `machine_status`.
- Confirm `machine_status` maps normal/buildup to `NORMAL`, abnormal/failure to `BROKEN`, and recovery to `RECOVERING`.
- Confirm `observation_time_index` and `timestamp` are contiguous after trimming.
- Confirm selected batch ids restrict source rows when provided.
- Confirm `keep_lineage_columns=True` retains configured lineage columns.

## Cross-Function Relationships

`build_bronze_handoff_from_postgres` reads from a PostgreSQL wide synthetic stream output and prepares the dataframe shape that Bronze ingestion expects. It is conceptually downstream of rebuilt/final-aligned synthetic observations and upstream of Bronze preprocessing.

## Source-Limited Items

- Direct active notebook use of `build_bronze_handoff_from_postgres` is Not determined from available source.
- SQL writes and append-control table updates are not performed by this selected function from available source.
