# utils/synthetic/pipeline/bronze_handoff.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `bronze_handoff.py` that need deeper explanation than the 071d module-level reference. The selected function claims final-aligned synthetic observations, writes them to the Bronze input table, and marks the source handoff state.

## Source Grounding

Sources used:

- `utils/synthetic/pipeline/bronze_handoff.py`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__synthetic__pipeline__bronze_handoff_module_reference.md`
- `function_inventory.json`
- `technical_reference/01_notebook_workflow_references/`
- `technical_reference/00_project_manual/`
- Documentation standards under `docs_agent_pack/00_STANDARDS/`

The active utility source file is the source of truth.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
| -------- | -------------- | ------------------------ |
| `handoff_final_aligned_observations_to_bronze` | Claims, writes, and marks one final-aligned-to-Bronze handoff batch | Synthetic final-aligned output to Bronze input stage |

## Module-Level Technical Context

`bronze_handoff.py` handles the durable transfer of final-aligned synthetic observations into a Bronze input staging table. It uses explicit source-side handoff status fields and target-side primary keys so repeated runs can claim batches safely, avoid duplicate target rows, and record completion or failure.

## Deep Function References

### `handoff_final_aligned_observations_to_bronze`

#### Functional Purpose

`handoff_final_aligned_observations_to_bronze` performs one Bronze handoff batch. It claims pending rows from a final-aligned source table, writes claimed rows into the Bronze handoff target, then marks the claimed source rows completed. If the write path fails, it marks the claimed source rows failed and returns the error.

#### Pipeline Context

Active notebook source confirms import in the synthetic Bronze handoff notebook and calls in an all-in-one synthetic workflow notebook. This function supports the final step that makes synthetic observations available to the Bronze input path.

#### Inputs and Assumptions

Important inputs include the SQLAlchemy engine, handoff mode, batch size, schema, source table, target table, optional `dataset_id`, optional `run_id`, and `complete_only`.

Supported handoff modes are `row`, `row_batch`, and `full_batch`. The source table must exist and must receive runtime columns such as `bronze_handoff_status`, token, started/completed timestamps, error, mode, target table, and attempt count. When `complete_only=True`, claimed rows must have `rebuild_is_complete=TRUE`.

#### Outputs and Return Contract

The function returns a dictionary. Source-confirmed statuses include:

- `empty`
- `completed`
- `failed`

Returned fields include claimed count, written count, skipped existing count, target table, and handoff token or error when applicable.

#### Side Effects

Confirmed side effects are:

- Adds or verifies source runtime handoff columns and indexes through the claim path.
- Claims pending source rows by setting `bronze_handoff_status='claimed'`, token, started time, handoff mode, target table, and attempt count.
- Creates the Bronze target table when needed.
- Adds missing target columns based on the claimed dataframe.
- Removes rows whose dataset/run/asset/observation key already exists in the target before writing.
- Appends new claimed rows to the Bronze target table.
- Marks claimed source rows `completed` after the write succeeds.
- Marks claimed source rows `failed` with error text when the write path raises.

#### Failure Behavior and Guardrails

The function returns `empty` when no rows are claimed. Supporting validation raises `ValueError` for unsupported handoff modes, missing source tables, or invalid row-batch sizes. A write failure is caught, recorded on the source claim with `bronze_handoff_status='failed'`, and returned as a failed result.

#### Lineage, Idempotency, and Reproducibility Role

The handoff is keyed by `dataset_id`, `run_id`, `asset_id`, and `observation_index`. The source-side `bronze_handoff_token` ties a claim to completion or failure updates. The target table uses the same identity fields as a primary key, and the write path filters out existing target keys before appending new rows.

#### Why This Function Matters

The Bronze handoff is the point where synthetic observations become Bronze input records. This function gives that transition a durable status lifecycle, duplicate protection, and a clear failure state instead of relying on one-off dataframe writes.

#### Verification Method

- Confirm pending source rows move to `claimed` and then `completed` after a successful handoff.
- Confirm returned `claimed_count` matches the number of rows claimed.
- Confirm target row count increases by `written_count`.
- Confirm duplicate target keys are counted in `skipped_existing_count`.
- Confirm a forced write failure marks claimed source rows `failed` and records an error.
- Confirm `complete_only=True` excludes incomplete rebuilt observations.

## Cross-Function Relationships

`handoff_final_aligned_observations_to_bronze` is downstream of final-aligned synthetic observation preparation and upstream of Bronze preprocessing. It relies on source runtime columns, target table creation, duplicate filtering, and status updates implemented by supporting helpers in the same module.

## Source-Limited Items

- The exact Bronze notebook read path from the handoff target table is Not determined from available source in this function reference.
- This function does not publish Kafka messages or rebuild consumed long messages from available source.
