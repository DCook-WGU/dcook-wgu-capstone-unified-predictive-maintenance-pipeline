# Utility Module Reference: `utils/synthetic/pipeline/final_aligned_incremental.py`

## Module Purpose

This module builds final aligned synthetic observations incrementally from staged/rebuilt data.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module builds final aligned synthetic observations incrementally from staged/rebuilt data.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_get_existing_columns` | Return existing Postgres columns for incremental final alignment. | deep |
| `_infer_alter_column_type` | Infer a conservative Postgres type for an added aligned-output column. | deep |
| `_add_missing_columns` | Add dataframe columns that are missing from the final-aligned target. | deep |
| `_remove_existing_target_rows` | Remove rows already present in the final-aligned target table. | deep |
| `ensure_rebuilt_final_align_runtime_columns` | Add final-alignment claim/status columns to the rebuilt source table. | deep |
| `claim_rebuilt_rows_for_final_align` | Claim a bounded batch of rebuilt observations for final alignment. | deep |
| `load_premelt_for_claimed_final_align` | Load premelt rows matching a claimed final-alignment batch. | deep |
| `build_claimed_final_aligned_rows` | Build final-aligned rows for one claimed rebuilt batch. | deep |
| `write_claimed_final_aligned_rows` | Append claimed final-aligned rows and report duplicate-key skips. | deep |
| `mark_claimed_final_align_completed` | Mark rebuilt rows for a final-alignment token as completed. | deep |
| `mark_claimed_final_align_failed` | Mark rebuilt rows for a final-alignment token as failed. | deep |
| `requeue_failed_final_aligns` | Return failed final-alignment claims to pending status. | deep |
| `final_align_rebuilt_observations_to_stage` | Claim, build, write, and mark one final-alignment batch. | deep |
| `run_final_align_loop` | Run final alignment batches until empty, capped, or failed. | deep |

## Configuration Dependencies

- Dataset, run, stage, or recipe identifiers used for traceability.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_get_existing_columns` | `engine, *, schema, table` | Return existing Postgres columns for incremental final alignment. |
| `_infer_alter_column_type` | `series` | Infer a conservative Postgres type for an added aligned-output column. |
| `_add_missing_columns` | `engine, *, schema, table, dataframe` | Add dataframe columns that are missing from the final-aligned target. |
| `_remove_existing_target_rows` | `engine, *, dataframe, schema, target_table` | Remove rows already present in the final-aligned target table. |
| `ensure_rebuilt_final_align_runtime_columns` | `engine, *, schema, source_table` | Add final-alignment claim/status columns to the rebuilt source table. |
| `claim_rebuilt_rows_for_final_align` | `engine, *, batch_size, schema, rebuilt_table, dataset_id, run_id, complete_only, final_align_token` | Claim a bounded batch of rebuilt observations for final alignment. |
| `load_premelt_for_claimed_final_align` | `engine, claimed_rows, *, schema, premelt_table` | Load premelt rows matching a claimed final-alignment batch. |
| `build_claimed_final_aligned_rows` | `engine, claimed_rows, *, schema, premelt_table, n_sensors, prefer_rebuilt_sensor_values` | Build final-aligned rows for one claimed rebuilt batch. |
| `write_claimed_final_aligned_rows` | `engine, dataframe, *, schema, target_table` | Append claimed final-aligned rows and report duplicate-key skips. |
| `mark_claimed_final_align_completed` | `engine, *, final_align_token, schema, rebuilt_table` | Mark rebuilt rows for a final-alignment token as completed. |
| `mark_claimed_final_align_failed` | `engine, *, final_align_token, error_message, schema, rebuilt_table` | Mark rebuilt rows for a final-alignment token as failed. |
| `requeue_failed_final_aligns` | `engine, *, schema, rebuilt_table` | Return failed final-alignment claims to pending status. |
| `final_align_rebuilt_observations_to_stage` | `engine, *, batch_size, schema, premelt_table, rebuilt_table, target_table, dataset_id, run_id, n_sensors, complete_only, prefer_rebuilt_sensor_values` | Claim, build, write, and mark one final-alignment batch. |
| `run_final_align_loop` | `engine, *, batch_size, schema, premelt_table, rebuilt_table, target_table, dataset_id, run_id, n_sensors, complete_only, prefer_rebuilt_sensor_values, max_iterations, stop_on_failure` | Run final alignment batches until empty, capped, or failed. |

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
