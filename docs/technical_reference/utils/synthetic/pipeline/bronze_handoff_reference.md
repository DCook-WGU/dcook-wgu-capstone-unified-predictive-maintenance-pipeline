# Synthetic Utility Reference: bronze_handoff.py

Source path:

`utils/synthetic/pipeline/bronze_handoff.py`

## Purpose

Manages synthetic final-output handoff into the Bronze-facing path.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `_get_existing_columns` | `engine` | Return existing columns for a Bronze handoff table. |
| Function | `_infer_alter_column_type` | `series` | Infer a conservative Postgres type for an added Bronze handoff column. |
| Function | `_add_missing_columns` | `engine` | Add dataframe columns that are missing from the Bronze handoff target. |
| Function | `_validate_handoff_mode` | `mode` | Validate and normalize the Bronze handoff batching mode. |
| Function | `dataframe_row_count_to_int` | `dataframe` | Read a one-row count dataframe into a Python int. |
| Function | `_resolve_effective_batch_size` | `engine` | Resolve row, row-batch, or full-batch size for a Bronze handoff claim. |
| Function | `ensure_final_aligned_runtime_columns` | `engine` | Add Bronze handoff claim/status columns to final-aligned observations. |
| Function | `ensure_bronze_handoff_target_table_exists` | `engine` | Create the Bronze handoff target table and handoff lookup indexes. |
| Function | `_remove_existing_target_rows` | `engine` | Remove claimed rows that already exist in the Bronze handoff target. |
| Function | `claim_final_aligned_rows_for_bronze` | `engine` | Claim final-aligned observations for transfer into the Bronze input table. |
| Function | `write_claimed_rows_to_bronze_target` | `engine, dataframe` | Append claimed final-aligned rows to the Bronze handoff target table. |
| Function | `mark_claimed_handoff_completed` | `engine` | Mark claimed final-aligned rows as completed after Bronze target write. |
| Function | `mark_claimed_handoff_failed` | `engine` | Mark claimed final-aligned rows as failed and store the handoff error. |
| Function | `requeue_failed_bronze_handoffs` | `engine` | Return failed Bronze handoff claims to pending status. |
| Function | `handoff_final_aligned_observations_to_bronze` | `engine` | Claim, write, and mark one final-aligned-to-Bronze handoff batch. |
| Function | `run_bronze_handoff_loop` | `engine` | Run Bronze handoff batches until empty, capped, full-batch, or failed. |

## Configuration Dependencies

- Uses SQL schema or table name settings.
- Uses dataset identity values.
- Uses run or recipe identity values.

## Inputs and Outputs

Key inputs:
- Configuration values, dataset identity, run identity, or recipe identity
- Database engine, schema, table, or SQL runtime context
- Pandas dataframes or dataframe-like stage inputs

Key outputs:
- Dataframes or transformed stage outputs
- SQL table rows, status updates, or database-stage records

## Logging, Ledger, and Artifact Behavior

### Logging

- No direct logger calls detected in this module.

### Ledger

- No direct ledger behavior detected in this module.

### SQL/database

- Uses SQL, PostgreSQL, engine, table, or database write/read behavior.

### Artifacts

- No direct artifact write pattern detected in this module.

## Downstream Usage

- `notebooks/synthetic/synthetic_00_postgres_to_bronze_no_kafka.ipynb`
- `notebooks/synthetic/synthetic_11_build_final_aligned_observations_stage.ipynb`
- `notebooks/synthetic/synthetic_12_bronze_handoff.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
