# utils/medallion/gold/cascade_row_tracking.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `cascade_row_tracking.py` that need deeper explanation than the module-level utility reference. The focus is row identity, stage-specific scoring inputs, row-level Isolation Forest scoring outputs, sparse stage merge-back behavior, and final flag-column normalization.

## Source Grounding

Sources used:

- `utils/medallion/gold/cascade_row_tracking.py`
- `function_inventory.json`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__medallion__gold__cascade_row_tracking_module_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_01_PreProcessing_code_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_02_Baseline_Modeling_code_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling_code_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling_code_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling_code_reference.md`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling_deep_technical_reference.md`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling_deep_technical_reference.md`
- `technical_reference/00_project_manual/notebook_relationship_map.md`
- `technical_reference/00_project_manual/notebook_dependency_matrix.md`
- `technical_reference/00_project_manual/artifact_and_table_handoff_map.md`
- `technical_reference/00_project_manual/medallion_handoff_map.md`

The active utility source file is the source of truth for function behavior. Workflow, deep technical, and manual references provide consumer and handoff context only.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
|---|---|---|
| `ensure_stable_row_id` | Creates or validates a stable unique row identifier on a dataframe copy. | Gold preprocessing and all row-tracked baseline/cascade scoring stages. |
| `build_stage_scoring_frame` | Builds a stage-specific scoring dataframe with identity columns and feature columns. | Baseline, Stage 1, and Stage 2 model scoring preparation. |
| `score_isolation_forest_stage` | Produces stage-prefixed Isolation Forest score, decision, prediction, and flag columns. | Row-level baseline/cascade scoring output construction. |
| `merge_stage_results_back` | Left-merges sparse stage results back onto the full master dataframe by stable row id. | Full-population cascade result assembly after candidate-stage scoring. |
| `finalize_stage_flag_columns` | Fills missing stage flags as integer zeros without filling sparse score columns. | Final result cleanup before truth, comparison, and validation handoff. |

## Module-Level Technical Context

`cascade_row_tracking.py` protects row-level alignment across Gold model stages. Gold cascade notebooks score different populations at different stages: Stage 1 can score the full population, while Stage 2 can score only candidate rows. These helpers keep a stable row key in the scoring frame, attach stage-prefixed scoring columns, and merge sparse outputs back to the master frame without losing full-row context. The selected functions do not write files, update SQL tables, update ledgers, register W&B artifacts, or create truth records directly.

## Deep Function References

### `ensure_stable_row_id`

#### Functional Purpose

`ensure_stable_row_id` ensures that a dataframe has a stable row identifier column. If the configured row-id column is missing, the function adds a zero-based integer sequence to a dataframe copy. If the column already exists, the function preserves it and validates that it contains no null values and no duplicates.

#### Pipeline Context

Workflow references confirm use in Gold_01 to stamp `meta__row_id` before Gold model-input handoff, and in Gold_03a, Gold_03b, and Gold_03c before cascade scoring. The same row identity is used later by stage scoring and merge-back helpers.

#### Inputs and Assumptions

- `dataframe` is the source frame whose rows need stable identity.
- `row_id_column` defaults to `meta__row_id`.
- Existing row IDs are treated as authoritative and are not regenerated.
- A missing row-id column is filled with `np.arange(len(out), dtype=np.int64)`.
- The row-id column must be non-null and unique after creation or validation.

#### Outputs and Return Contract

The function returns a dataframe copy. The returned frame contains `row_id_column`, either carried forward from the source frame or created as a deterministic integer sequence based on current row order.

#### Side Effects

The function copies the input dataframe and adds or validates the row-id column on the copy. No mutation of the caller's dataframe and no external side effects are confirmed from available source.

#### Failure Behavior and Guardrails

- Raises `ValueError` if `row_id_column` contains null values.
- Raises `ValueError` if `row_id_column` is not unique.
- Does not validate whether an existing row-id sequence is semantically tied to the original source order; it validates null and uniqueness only.

#### Lineage and Reproducibility Role

`meta__row_id` is the join key that lets stage-specific scoring outputs return to the full population. Without a stable row id, Stage 2 candidate-only outputs could not be safely merged back without relying on fragile dataframe index position.

#### Why This Function Matters

The Gold cascade produces sparse stage outputs. Stable row identity is the mechanism that keeps row-level model evidence attached to the correct observation after filtering, scoring, and merge-back.

#### Verification Method

- Confirm the returned dataframe contains `meta__row_id`.
- Confirm `returned["meta__row_id"].is_unique` is true.
- Confirm `returned["meta__row_id"].isna().any()` is false.
- Call the function on a copy with duplicate IDs and confirm `ValueError`.
- Confirm the source dataframe is unchanged when the row-id column was missing.

### `build_stage_scoring_frame`

#### Functional Purpose

`build_stage_scoring_frame` builds the exact dataframe that will be passed into a scoring stage. It first guarantees stable row identity, carries preferred identity/order columns when present, appends the requested feature columns, applies an optional candidate mask, and validates that the resulting scoring frame is non-empty with unique non-null row IDs.

#### Pipeline Context

Workflow references confirm use in Gold_02 baseline scoring and in Gold_03a, Gold_03b, and Gold_03c cascade stages. Stage 1 can score all rows with no mask. Stage 2 can use a candidate mask so only Stage 1 positives are scored while row identity remains available for merge-back.

#### Inputs and Assumptions

- `dataframe` is the full or intermediate model frame.
- `feature_columns` identifies columns to score. Columns already present as identity columns are not duplicated.
- `mask` is optional. When supplied, it is used with `working.loc[mask, selected_columns]`.
- `row_id_column` defaults to `meta__row_id`.
- Preferred identity columns include `meta__row_id`, `meta__record_id`, `event_id`, `time_index`, `event_step`, `event_time`, `meta__asset_id`, `meta__run_id`, and `machine_status` when those columns exist.

#### Outputs and Return Contract

The function returns a dataframe copy containing identity/order columns plus the requested feature columns. If a mask is supplied, the returned frame contains only rows selected by that mask.

#### Side Effects

The function creates dataframe copies only. No external file, SQL, ledger, W&B, truth, or artifact side effects are confirmed from available source.

#### Failure Behavior and Guardrails

- `ensure_stable_row_id` raises if row IDs are null or duplicated.
- Raises `ValueError` if the scoring frame is empty after optional masking.
- Raises `ValueError` if row IDs become null or non-unique in the scoring frame.
- The active source does not explicitly raise for missing feature columns; pandas selection will fail if requested feature columns are not present in `working`.

#### Lineage and Reproducibility Role

The returned scoring frame preserves row identity and ordering fields next to the model features. This makes the stage output auditable and mergeable even when only a candidate subset is scored.

#### Why This Function Matters

This helper is the boundary between full-population Gold outputs and stage-specific model inputs. It prevents candidate filtering from dropping the key needed to return sparse model evidence to the original population.

#### Verification Method

- Confirm the output includes `meta__row_id` and all requested feature columns.
- Confirm a candidate mask reduces row count as expected.
- Confirm the output row IDs are unique and non-null.
- Confirm an all-false mask raises `ValueError`.
- Confirm the output preserves expected ordering columns when present.

### `score_isolation_forest_stage`

#### Functional Purpose

`score_isolation_forest_stage` applies a fitted Isolation Forest-compatible model to a stage scoring frame and returns row-level outputs with stage-prefixed columns. It creates score, decision, prediction, and binary flag columns for the supplied `stage_name`.

#### Pipeline Context

Workflow references confirm use in Gold_02 for baseline row-tracked scoring and in Gold_03a, Gold_03b, and Gold_03c for Stage 1 and Stage 2 cascade scoring. The helper supports row-level traceability by returning model outputs next to the row identifier.

#### Inputs and Assumptions

- `stage_dataframe` must contain the stable row-id column and all requested `feature_columns`.
- `model` must support `score_samples`, `decision_function`, and `predict`.
- `feature_columns` define the matrix passed to the model.
- `stage_name` controls output column names: `{stage_name}_score`, `{stage_name}_decision`, `{stage_name}_pred`, and `{stage_name}_flag`.
- The model's prediction convention is assumed to use `-1` for anomaly rows; the flag is computed as `pred == -1`.
- During scoring, `row_id_column` is set as the DataFrame index (`set_index(row_id_column, drop=False)`) and reset before returning (`reset_index(drop=True)`). The `row_id_column` column is preserved as a regular column in the output.

#### Outputs and Return Contract

The function returns a dataframe copy with:

- `{stage_name}_score` — raw `model.score_samples(X)` output. **Lower values indicate more anomalous rows** (sklearn IsolationForest convention). Some inline notebook helpers (e.g., `compute_anomaly_scores_isolation_forest`) return the negated form (`-score_samples()`), where higher values indicate more anomalous rows. These are distinct values; readers must account for this direction difference when reading `{stage_name}_score` columns produced by this utility.
- `{stage_name}_decision` — `model.decision_function(X)`: signed offset from the decision boundary.
- `{stage_name}_pred` — `model.predict(X)`: `1` (inlier) or `-1` (outlier).
- `{stage_name}_flag` — `(pred == -1).astype(int)`: `1` for outlier/anomaly candidate; `0` for inlier.

The row-id column remains present after the final `reset_index(drop=True)`.

#### Side Effects

The function copies and scores the stage dataframe in memory. No source dataframe mutation, model persistence, artifact write, SQL write, ledger update, W&B operation, or truth-record write is confirmed from available source.

#### Failure Behavior and Guardrails

- Raises `ValueError` if the feature matrix is empty.
- Raises `ValueError` if the feature matrix index does not match the stage dataframe index after row-id indexing.
- Missing feature columns, unfitted models, incompatible feature values, or unsupported model interfaces fail through pandas or the model object.

#### Lineage and Reproducibility Role

The stage-prefixed columns provide row-level evidence for each stage. Because the frame is indexed by `meta__row_id` during scoring, the model outputs remain aligned to stable row identity rather than to transient dataframe position.

#### Why This Function Matters

The cascade interpretation depends on knowing which stage scored each row and how that stage classified it. Stage-prefixed score, decision, prediction, and flag columns make later comparison, diagnostics, and validation possible.

#### Verification Method

- Confirm the returned dataframe contains the four stage-prefixed columns.
- Confirm `{stage_name}_flag` equals `1` where `{stage_name}_pred == -1`.
- Confirm the row-id column is still present.
- Confirm an empty scoring frame raises `ValueError`.
- Confirm returned row count equals the input stage scoring frame row count.

### `merge_stage_results_back`

#### Functional Purpose

`merge_stage_results_back` attaches stage scoring outputs back onto the full master dataframe using the stable row-id column. It performs a left merge so all master rows are preserved, including rows that were not scored by a masked candidate stage.

#### Pipeline Context

Workflow references confirm this merge-back pattern in Gold_03a, Gold_03b, and Gold_03c. It is central to Stage 2 candidate scoring because Stage 2 can produce a sparse result frame while the final cascade output still needs one row per original observation.

#### Inputs and Assumptions

- `master_dataframe` is the full frame receiving stage outputs.
- `stage_results_dataframe` contains `row_id_column` and any available stage result columns for the stage.
- `stage_name` determines expected stage columns: `{stage_name}_score`, `{stage_name}_decision`, `{stage_name}_pred`, and `{stage_name}_flag`.
- Both master and stage result row IDs must be unique.

#### Outputs and Return Contract

The function returns a dataframe copy produced by left-merging available stage result columns onto the master dataframe. Non-scored rows remain present and receive missing values for stage result columns.

#### Side Effects

The function copies and merges dataframes in memory. No external side effects are confirmed from available source.

#### Failure Behavior and Guardrails

- Raises `ValueError` if the stage results do not include the row-id column.
- Raises `ValueError` if master row IDs are not unique.
- Raises `ValueError` if stage result row IDs are not unique.
- Stage result columns other than the row-id column are optional; only available stage columns are merged.

#### Lineage and Reproducibility Role

The left merge preserves full-population row count and makes non-candidate score sparsity meaningful. Deep Gold references confirm that non-candidate Stage 2 rows retain `NaN` score values, which distinguishes "not evaluated by Stage 2" from a numeric score.

#### Why This Function Matters

Cascade stages filter rows, but downstream comparison and validation need a full-population output. This helper keeps sparse stage results aligned without collapsing or reordering the original population.

#### Verification Method

- Confirm the merged row count equals the master row count.
- Confirm all master row IDs remain present.
- Confirm non-candidate rows have missing stage score values after a candidate-only merge.
- Confirm duplicate row IDs in either input raise `ValueError`.
- Confirm missing row-id column in stage results raises `ValueError`.

### `finalize_stage_flag_columns`

#### Functional Purpose

`finalize_stage_flag_columns` normalizes available stage flag columns after merge-back. For each requested stage name, it fills missing values in `{stage_name}_flag` with `0` and casts the column to integer.

#### Pipeline Context

Workflow and deep technical references confirm use in Gold_03a, Gold_03b, and Gold_03c after stage result assembly. The helper is used after sparse candidate-stage merges so flag columns are reliable integer-like indicators even when score columns intentionally remain sparse.

#### Inputs and Assumptions

- `dataframe` is the assembled model output frame.
- `stage_names` contains stage prefixes such as `stage1`, `stage2`, and `stage3`.
- Only columns that already exist are modified. Missing stage flag columns are skipped.

#### Outputs and Return Contract

The function returns a dataframe copy. Available `{stage_name}_flag` columns have missing values filled with `0` and are cast to `int`.

#### Side Effects

The function copies the dataframe and updates flag columns on the copy. No external side effects are confirmed from available source.

#### Failure Behavior and Guardrails

- No custom exception is raised for missing flag columns.
- Casting can fail if a present flag column contains values that cannot be converted to integers after `fillna(0)`.
- Score columns are not filled or cast by this helper.

#### Lineage and Reproducibility Role

The function preserves the distinction between sparse stage scores and finalized flags. Non-candidate score `NaN` values can continue to mean "not evaluated," while missing flags are normalized to `0` for downstream filtering and comparison.

#### Why This Function Matters

Downstream notebooks and reports need flag columns that behave like binary indicators. This helper makes sparse candidate-stage outputs usable without erasing the score-column signal that a row did not enter that stage.

#### Verification Method

- Confirm each existing requested flag column contains no null values after finalization.
- Confirm finalized flag columns have integer dtype.
- Confirm score columns such as `stage2_score` retain `NaN` values for non-candidate rows.
- Confirm missing stage names are skipped without adding new columns.
- Confirm the source dataframe is unchanged.

## Cross-Function Relationships

- `ensure_stable_row_id` establishes the key used by all downstream stage scoring and merge-back operations.
- `build_stage_scoring_frame` calls `ensure_stable_row_id`, carries identity columns, and optionally filters rows by candidate mask.
- `score_isolation_forest_stage` adds stage-prefixed scoring columns to the stage frame.
- `merge_stage_results_back` attaches those sparse or full stage outputs back to the master frame on `meta__row_id`.
- `finalize_stage_flag_columns` cleans flag columns after merge-back while leaving sparse score columns available to indicate rows that were not evaluated by a candidate-only stage.
- Workflow references confirm this pattern in Gold_03a, Gold_03b, and Gold_03c; Gold_02 uses the scoring-frame and scoring helper for baseline row-tracked output.

## Source-Limited Items

- The selected functions do not write row-tracking artifacts directly; row-tracking artifact paths in notebooks are outside this utility source.
- Direct SQL, W&B, ledger, truth-record, and model persistence behavior is Not determined from available source for these selected functions.
- The active source does not attach dataset IDs, run IDs, parent truth hashes, or pipeline mode fields inside these functions.
- `stage3_flag` alias creation from `stage3_confirmed_flag` appears in notebook references, not in the selected utility source.
