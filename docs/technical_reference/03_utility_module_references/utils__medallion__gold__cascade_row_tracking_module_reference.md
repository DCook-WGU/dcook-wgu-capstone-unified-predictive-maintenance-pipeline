# Utility Module Reference: `utils/medallion/gold/cascade_row_tracking.py`

## Module Purpose

This module tracks row membership and handoff metadata across Gold cascade modeling stages.

## Pipeline Role

- Stage support: Gold
- Primary responsibility: This module tracks row membership and handoff metadata across Gold cascade modeling stages.

## Primary Consumers

`EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `ensure_stable_row_id` | Ensure the dataframe has a stable unique row id for downstream stage scoring and merge-back operations. | deep |
| `get_identity_columns` | Return stable identity and ordering columns that should be carried through all stage scoring outputs. | short |
| `build_stage_scoring_frame` | Build the exact dataframe that will be scored by a cascade stage while preserving row identity and ordering columns. | deep |
| `score_isolation_forest_stage` | Score one Isolation Forest stage and return row-level stage results. | deep |
| `merge_stage_results_back` | Merge row-level stage results back onto the full master dataframe. | deep |
| `finalize_stage_flag_columns` | Fill missing stage flag columns after merge-back so non-candidate rows are represented as 0 rather than NaN. | deep |
| `get_detected_rows_dataframe` | Return a dataframe containing only rows where the requested flag column is 1. | deep |
| `get_stage_detected_rows_dataframe` | Return detected rows for standard baseline/cascade stage naming patterns. | short |

## Configuration Dependencies

- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `ensure_stable_row_id` | `dataframe, row_id_column` | Ensure the dataframe has a stable unique row id for downstream stage scoring and merge-back operations. |
| `get_identity_columns` | `dataframe, row_id_column` | Return stable identity and ordering columns that should be carried through all stage scoring outputs. |
| `build_stage_scoring_frame` | `dataframe, feature_columns, mask, row_id_column` | Build the exact dataframe that will be scored by a cascade stage while preserving row identity and ordering columns. |
| `score_isolation_forest_stage` | `stage_dataframe, model, feature_columns, stage_name, row_id_column` | Score one Isolation Forest stage and return row-level stage results. |
| `merge_stage_results_back` | `master_dataframe, stage_results_dataframe, stage_name, row_id_column` | Merge row-level stage results back onto the full master dataframe. |
| `finalize_stage_flag_columns` | `dataframe, stage_names` | Fill missing stage flag columns after merge-back so non-candidate rows are represented as 0 rather than NaN. |
| `get_detected_rows_dataframe` | `dataframe, *, target_flag_column, row_id_column, score_column, decision_column, pred_column, include_columns, preferred_identity_columns, sort_by, ascending, require_flag_column` | Return a dataframe containing only rows where the requested flag column is 1. |
| `get_stage_detected_rows_dataframe` | `dataframe, *, stage_name, row_id_column, include_columns, sort_by, ascending` | Return detected rows for standard baseline/cascade stage naming patterns. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Not determined from available source

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.
- Source raises `an` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`

## Module Importance

This module matters because Gold notebooks depend on stable shared helpers for model input preparation, cascade modeling, evaluation, validation contracts, and artifact traceability.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
