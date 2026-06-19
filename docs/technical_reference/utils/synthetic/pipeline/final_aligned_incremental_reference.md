# Synthetic Utility Reference: final_aligned_incremental.py

Source path:

`utils/synthetic/pipeline/final_aligned_incremental.py`

## Purpose

Supports incremental final alignment with claim/status handling.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `_get_existing_columns` | `engine` | Return existing Postgres columns for incremental final alignment. |
| Function | `_infer_alter_column_type` | `series` | Infer a conservative Postgres type for an added aligned-output column. |
| Function | `_add_missing_columns` | `engine` | Add dataframe columns that are missing from the final-aligned target. |
| Function | `_remove_existing_target_rows` | `engine` | Remove rows already present in the final-aligned target table. |
| Function | `ensure_rebuilt_final_align_runtime_columns` | `engine` | Add final-alignment claim/status columns to the rebuilt source table. |
| Function | `claim_rebuilt_rows_for_final_align` | `engine` | Claim a bounded batch of rebuilt observations for final alignment. |
| Function | `load_premelt_for_claimed_final_align` | `engine, claimed_rows` | Load premelt rows matching a claimed final-alignment batch. |
| Function | `build_claimed_final_aligned_rows` | `engine, claimed_rows` | Build final-aligned rows for one claimed rebuilt batch. |
| Function | `write_claimed_final_aligned_rows` | `engine, dataframe` | Append claimed final-aligned rows and report duplicate-key skips. |
| Function | `mark_claimed_final_align_completed` | `engine` | Mark rebuilt rows for a final-alignment token as completed. |
| Function | `mark_claimed_final_align_failed` | `engine` | Mark rebuilt rows for a final-alignment token as failed. |
| Function | `requeue_failed_final_aligns` | `engine` | Return failed final-alignment claims to pending status. |
| Function | `final_align_rebuilt_observations_to_stage` | `engine` | Claim, build, write, and mark one final-alignment batch. |
| Function | `run_final_align_loop` | `engine` | Run final alignment batches until empty, capped, or failed. |

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

- `notebooks/synthetic/synthetic_11a_build_final_aligned_observations_stage.ipynb`
- `notebooks/synthetic/synthetic_11b_build_final_aligned_observations_stage.ipynb`
- `notebooks/synthetic/synthetic_pipeline_condensed-09_11.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
