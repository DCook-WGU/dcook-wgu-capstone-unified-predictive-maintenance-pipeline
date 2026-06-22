# utils/medallion/gold/gold_cascade_modeling.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `gold_cascade_modeling.py` that need deeper explanation than the module-level utility reference. The focus is the staged cascade modeling contract: Stage 1 and Stage 2 model orchestration, threshold selection, Stage 3 confirmation handoff, metric assembly, and comparison-ready summary fields.

## Source Grounding

Sources used:

- `utils/medallion/gold/gold_cascade_modeling.py`
- `utils/medallion/gold/gold_modeling_common.py`
- `function_inventory.json`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__medallion__gold__gold_cascade_modeling_module_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling_code_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling_code_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling_code_reference.md`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling_deep_technical_reference.md`
- `technical_reference/00_project_manual/notebook_relationship_map.md`
- `technical_reference/00_project_manual/notebook_dependency_matrix.md`
- `technical_reference/00_project_manual/artifact_and_table_handoff_map.md`
- `technical_reference/00_project_manual/medallion_handoff_map.md`

The active utility source file is the source of truth for function behavior. Workflow and manual references provide consumer and handoff context only.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
|---|---|---|
| `run_cascade_pipeline` | Coordinates Stage 1 fit/scoring, Stage 2 fit/scoring, threshold selection, Stage 3 rule confirmation, metrics, and scored dataframe outputs. | Gold cascade modeling and model comparison preparation. |
| `run_stage2_selection` | Selects a Stage 2 threshold from a percentile grid using recall filtering and metric-based ranking. | Tuned Stage 2 cascade threshold selection. |
| `build_cascade_summary` | Adds compact variant and feature-count fields to a cascade summary without mutating the input summary. | Comparison-ready cascade summary preparation. |

## Module-Level Technical Context

`gold_cascade_modeling.py` implements an in-memory three-stage cascade workflow. Stage 1 is a broad Isolation Forest screen, Stage 2 is a narrower Isolation Forest confirmation on a reduced feature set, and Stage 3 applies rule-based confirmation helpers from the Stage 3 rules module. The selected functions return fitted models, scored dataframe copies, threshold metadata, metrics, and summary dictionaries. They do not write files, persist models, update SQL, update ledgers, register W&B artifacts, or create truth records directly.

## Deep Function References

### "run_cascade_pipeline"

#### Functional Purpose

`run_cascade_pipeline` coordinates the shared three-stage cascade workflow. It fits Stage 1 and Stage 2 Isolation Forest models, selects thresholds, scores fit/train/test/all frames, creates a Stage 2 candidate gate, applies Stage 3 rule confirmation helpers, computes final cascade predictions, evaluates each scored partition, and returns fitted models, scored frames, and a summary payload.

#### Pipeline Context

This function supports Gold cascade modeling concepts used across Gold_03a, Gold_03b, and Gold_03c. Module references confirm Gold_03b and Gold_03c as consumers of `gold_cascade_modeling.py`, while manual references confirm that Gold_03a through Gold_03c provide cascade outputs to Gold_04 and Gold_06A. Direct notebook invocation of `run_cascade_pipeline` is Not determined from available source.

#### Inputs and Assumptions

- `fit_dataframe` is used to fit both Stage 1 and Stage 2 models.
- `train_dataframe` is used for Stage 1 threshold selection and either fixed Stage 2 threshold selection or Stage 2 grid search.
- `test_dataframe` and `all_dataframe` are scored and evaluated after threshold selection.
- `stage1_feature_columns` and `stage2_feature_columns` must contain usable columns in the fit/scoring frames.
- `reference_profile` and `stage3_sensor_groups` are passed to Stage 3 rule helpers for breach and corroboration logic.
- `label_column` must exist in scored frames before evaluation.
- `stage1_model_params` and `stage2_model_params` are passed into Isolation Forest constructors by the Stage 1 and Stage 2 fit helpers.
- `variant` must be supported by `_variant_defaults`: `default`, `tuned`, or `improved`.

#### Outputs and Return Contract

The function returns a dictionary with:

- `stage1_model`: fitted Stage 1 Isolation Forest.
- `stage2_model`: fitted Stage 2 Isolation Forest.
- `summary`: variant config, fit info, thresholds, threshold-selection metadata, metrics, `cascade_metrics`, and Stage 3 debug metadata from the test frame.
- `scored_fit`, `scored_train`, `scored_test`, `scored_all`: dataframe copies with stage score columns, prediction flags, Stage 2 candidate flag, Stage 3 evidence columns, and `cascade_predicted_anomaly`.

`cascade_metrics` includes alert counts for all rows and test rows plus test-partition precision, recall, F1, ROC AUC, and average precision.

#### Side Effects

The function fits two model objects and creates scored dataframe copies. No external file writes, model persistence, SQL writes, ledger updates, W&B operations, path creation, or truth-record writes are confirmed in the active utility source.

#### Failure Behavior and Guardrails

- Unsupported cascade variant raises `ValueError` through `_variant_defaults`.
- Missing usable Stage 1 or Stage 2 features raise `ValueError` through the fit helpers.
- Unsupported Stage 2 optimization metric raises `ValueError` through `run_stage2_selection`.
- Empty score arrays raise `ValueError` through percentile threshold selection.
- Missing label, prediction, or score columns can raise through evaluation helpers.
- Stage 3 helper failures depend on the imported Stage 3 rule functions and their input contracts.

#### Lineage and Reproducibility Role

The summary records variant defaults, Stage 1 and Stage 2 model parameters, feature columns, fit row counts, threshold values, threshold-selection metadata, full Stage 2 threshold-table records, metrics for each partition, and Stage 3 debug information for the test frame. The function does not attach dataset IDs, run IDs, truth hashes, parent truth hashes, artifact paths, SQL model stages, or pipeline mode directly.

#### Why This Function Matters

The cascade is the capstone's main alternative to the single-model baseline. This function captures the technical contract for staged filtering: broad detection, narrower confirmation, rule-based evidence, and comparison-ready metrics. Changes to this function can alter model comparison results even if notebook artifact code remains unchanged.

#### Verification Method

- Confirm returned `summary["variant_config"]` matches the requested variant.
- Confirm `scored_all` contains `stage1_anomaly_score`, `stage1_predicted_anomaly`, `stage2_anomaly_score`, `stage2_predicted_anomaly`, `stage2_candidate_flag`, `stage3_confirmed_flag`, and `cascade_predicted_anomaly`.
- Confirm every `cascade_predicted_anomaly == 1` row also has `stage2_candidate_flag == 1` and `stage3_confirmed_flag == 1`.
- Confirm `summary["stage2_threshold_table"]` is populated for variants using Stage 2 search and empty for fixed-percentile variants.
- Confirm `cascade_metrics["final_alert_count_test_rows"]` equals the sum of `scored_test["cascade_predicted_anomaly"]`.
- Confirm input dataframes are not mutated by comparing columns before and after the call.

### "run_stage2_selection"

#### Functional Purpose

`run_stage2_selection` selects a Stage 2 threshold from a grid of threshold percentiles. It evaluates each candidate threshold, filters candidates by a minimum recall floor when possible, ranks the remaining candidates by the requested optimization metric and tie-breakers, and returns the selected threshold with selection metadata and the full threshold table.

#### Pipeline Context

This function supports tuned cascade behavior. The Gold_03b workflow reference confirms config-driven Stage 2 selection with threshold-grid and parameter-search modes, and manual references confirm that Gold_03b's selected Stage 2 settings are handed to Gold_03c through saved artifacts. Direct notebook invocation of this exact shared utility function is Not determined from available source.

#### Inputs and Assumptions

- `fitted_model` must be a fitted Stage 2 Isolation Forest or compatible estimator.
- `validation_dataframe` must contain the Stage 2 feature columns and the configured label column.
- `feature_columns` identify the Stage 2 scoring feature set.
- `label_column` is used for precision, recall, F1, ROC AUC, and average precision calculation.
- `threshold_percentiles` supplies the candidate percentile grid.
- `min_recall` defines the recall floor. If no candidates meet it, the function falls back to the full threshold table.
- `optimization_metric` must be a column in the filtered threshold table.

#### Outputs and Return Contract

The function returns:

- `selected_threshold`: threshold value from the best candidate row.
- `selection_info`: `selection_method`, `optimization_metric`, `min_recall`, selected threshold percentile, selected threshold, and selected precision/recall/F1/predicted-positive count.
- `threshold_table`: dataframe with one row per evaluated percentile and columns for threshold percentile, threshold, precision, recall, F1, predicted-positive count, ROC AUC, and average precision.

#### Side Effects

No external side effects are confirmed from available source. The function scores and evaluates in memory through `evaluate_stage2_model_with_thresholds`.

#### Failure Behavior and Guardrails

- If `optimization_metric` is not a column in the filtered threshold table, the function raises `ValueError`.
- If no rows meet `min_recall`, the function uses the full threshold table rather than failing.
- Empty or invalid threshold percentile inputs can fail through threshold evaluation or downstream `.iloc[0]` access if no threshold rows exist.
- Missing label or feature inputs can fail through the scoring and metric helpers.

#### Lineage and Reproducibility Role

The returned metadata records the threshold grid result, recall floor, optimization metric, selected percentile, selected threshold, and selected candidate metrics. This supports reproducible tuning because a reviewer can compare the selected row against the full threshold table.

#### Why This Function Matters

Stage 2 tuning controls how many Stage 1 candidates survive into rule confirmation. A stable threshold-selection contract is necessary for comparing default, tuned, and improved cascade variants without relying on notebook-local variables alone.

#### Verification Method

- Confirm `threshold_table` has one row per requested percentile.
- Confirm `selected_threshold` equals the threshold in the best row after recall filtering and sorting.
- Confirm the selected row meets `min_recall` when at least one candidate meets the recall floor.
- Pass an unsupported `optimization_metric` and confirm `ValueError`.
- Confirm `selection_info["selected_metrics"]` matches the selected row's precision, recall, F1, and predicted-positive count.

### "build_cascade_summary"

#### Functional Purpose

`build_cascade_summary` adds compact comparison-oriented fields to a cascade summary dictionary. It records the cascade variant and Stage 1/Stage 2 feature counts while preserving the input summary dictionary unchanged.

#### Pipeline Context

This helper supports summary preparation for downstream comparison. Manual references confirm that cascade summaries from Gold_03a through Gold_03c are consumed by Gold_04 and Gold_06A. Direct notebook invocation of this exact shared utility function is Not determined from available source.

#### Inputs and Assumptions

- `variant` identifies the cascade variant represented by the summary.
- `summary` is a dictionary containing previously computed cascade metrics and metadata.
- `stage1_feature_columns` and `stage2_feature_columns` are the feature lists used by the cascade.

#### Outputs and Return Contract

The function returns a shallow copy of `summary` with:

- `variant`
- `stage1_feature_count`
- `stage2_feature_count`

No nested summary data is deep-copied.

#### Side Effects

No side effects are confirmed from available source. The input `summary` dictionary is not modified at the top level.

#### Failure Behavior and Guardrails

No custom exceptions are defined. Values must be compatible with `dict(summary)` and `len(list(...))` operations.

#### Lineage and Reproducibility Role

The helper adds compact fields that make summary payloads easier to compare across cascade variants. It does not add truth hashes, dataset IDs, run IDs, artifact paths, SQL model stages, or pipeline mode directly.

#### Why This Function Matters

Gold comparison needs compact, variant-labeled model summaries. This helper gives later comparison code a stable place to find variant and feature-count metadata without recalculating those values from notebook-local variables.

#### Verification Method

- Confirm the returned dictionary contains `variant`, `stage1_feature_count`, and `stage2_feature_count`.
- Confirm the original `summary` does not gain those keys after the call.
- Confirm feature counts match the lengths of the provided feature lists.

## Cross-Function Relationships

- `run_cascade_pipeline` fits Stage 1 and Stage 2 models through lower-level fit helpers.
- It calibrates Stage 1 using train-frame scores and `choose_threshold_by_percentile`.
- For tuned or improved variants, it calls `run_stage2_selection`; for fixed variants, it selects Stage 2 threshold by a fixed percentile.
- It scores each partition, builds `stage2_candidate_flag`, applies Stage 3 rule helpers, and computes `cascade_predicted_anomaly`.
- It packages threshold metadata, Stage 2 selection metadata, metrics, and Stage 3 debug details into `summary`.
- `build_cascade_summary` can add variant and feature-count fields to such summaries for comparison readiness.
- Manual and workflow references confirm that cascade notebook outputs feed Gold_04 comparison and Gold_06A replay validation, but persistence and validation-contract writes are outside this utility module.

## Source-Limited Items

- Direct notebook invocation of `run_cascade_pipeline`, `run_stage2_selection`, and `build_cascade_summary` is Not determined from available source.
- File persistence for cascade results, joblib models, thresholds JSON, summary JSON, metadata JSON, validation contracts, truth records, ledger output, SQL rows, and W&B artifacts is confirmed for cascade notebooks, not for these selected utility functions.
- Dataset ID, run ID, parent truth hash, pipeline mode, SQL model stage, and artifact path handling inside these selected functions is Not determined from available source.
- Gold_03c's weighted Stage 3 operating-mode search is documented in workflow references but is not implemented by these selected shared utility functions.
