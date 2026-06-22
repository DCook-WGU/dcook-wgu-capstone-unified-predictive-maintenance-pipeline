# Utility Module Reference: `utils/synthetic/pipeline/bronze_handoff.py`

## Module Purpose

This module prepares final synthetic outputs for handoff into the Bronze preprocessing path.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module prepares final synthetic outputs for handoff into the Bronze preprocessing path.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_get_existing_columns` | Return existing columns for a Bronze handoff table. | deep |
| `_infer_alter_column_type` | Infer a conservative Postgres type for an added Bronze handoff column. | deep |
| `_add_missing_columns` | Add dataframe columns that are missing from the Bronze handoff target. | deep |
| `_validate_handoff_mode` | Validate and normalize the Bronze handoff batching mode. | deep |
| `dataframe_row_count_to_int` | Read a one-row count dataframe into a Python int. | short |
| `_resolve_effective_batch_size` | Resolve row, row-batch, or full-batch size for a Bronze handoff claim. | deep |
| `ensure_final_aligned_runtime_columns` | Add Bronze handoff claim/status columns to final-aligned observations. | deep |
| `ensure_bronze_handoff_target_table_exists` | Create the Bronze handoff target table and handoff lookup indexes. | deep |
| `_remove_existing_target_rows` | Remove claimed rows that already exist in the Bronze handoff target. | deep |
| `claim_final_aligned_rows_for_bronze` | Claim final-aligned observations for transfer into the Bronze input table. | deep |
| `write_claimed_rows_to_bronze_target` | Append claimed final-aligned rows to the Bronze handoff target table. | deep |
| `mark_claimed_handoff_completed` | Mark claimed final-aligned rows as completed after Bronze target write. | deep |
| `mark_claimed_handoff_failed` | Mark claimed final-aligned rows as failed and store the handoff error. | deep |
| `requeue_failed_bronze_handoffs` | Return failed Bronze handoff claims to pending status. | deep |
| `handoff_final_aligned_observations_to_bronze` | Claim, write, and mark one final-aligned-to-Bronze handoff batch. | deep |
| `run_bronze_handoff_loop` | Run Bronze handoff batches until empty, capped, full-batch, or failed. | deep |

## Configuration Dependencies

- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_get_existing_columns` | `engine, *, schema, table` | Return existing columns for a Bronze handoff table. |
| `_infer_alter_column_type` | `series` | Infer a conservative Postgres type for an added Bronze handoff column. |
| `_add_missing_columns` | `engine, *, schema, table, dataframe` | Add dataframe columns that are missing from the Bronze handoff target. |
| `_validate_handoff_mode` | `mode` | Validate and normalize the Bronze handoff batching mode. |
| `dataframe_row_count_to_int` | `dataframe, *, column` | Read a one-row count dataframe into a Python int. |
| `_resolve_effective_batch_size` | `engine, *, mode, batch_size, schema, source_table, dataset_id, run_id, complete_only` | Resolve row, row-batch, or full-batch size for a Bronze handoff claim. |
| `ensure_final_aligned_runtime_columns` | `engine, *, schema, source_table` | Add Bronze handoff claim/status columns to final-aligned observations. |
| `ensure_bronze_handoff_target_table_exists` | `engine, *, schema, target_table` | Create the Bronze handoff target table and handoff lookup indexes. |
| `_remove_existing_target_rows` | `engine, *, dataframe, schema, target_table` | Remove claimed rows that already exist in the Bronze handoff target. |
| `claim_final_aligned_rows_for_bronze` | `engine, *, mode, batch_size, schema, source_table, target_table, dataset_id, run_id, complete_only, handoff_token` | Claim final-aligned observations for transfer into the Bronze input table. |
| `write_claimed_rows_to_bronze_target` | `engine, dataframe, *, schema, target_table` | Append claimed final-aligned rows to the Bronze handoff target table. |
| `mark_claimed_handoff_completed` | `engine, *, handoff_token, schema, source_table` | Mark claimed final-aligned rows as completed after Bronze target write. |
| `mark_claimed_handoff_failed` | `engine, *, handoff_token, error_message, schema, source_table` | Mark claimed final-aligned rows as failed and store the handoff error. |
| `requeue_failed_bronze_handoffs` | `engine, *, schema, source_table` | Return failed Bronze handoff claims to pending status. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.

## Artifact / SQL / File-System Interactions

- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.

## Failure Behavior

- Source raises `TypeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because the synthetic subsystem depends on repeatable staging, truth handling, streaming handoff, or final alignment behavior to make controlled pump telemetry tests explainable.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
