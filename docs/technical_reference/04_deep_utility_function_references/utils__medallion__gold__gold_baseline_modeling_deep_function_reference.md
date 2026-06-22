# utils/medallion/gold/gold_baseline_modeling.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `gold_baseline_modeling.py` that need deeper explanation than the module-level utility reference. The focus is the baseline Isolation Forest modeling contract: fit input, threshold calibration, score-column behavior, evaluation payloads, and comparison-ready summary structure.

## Source Grounding

Sources used:

- `utils/medallion/gold/gold_baseline_modeling.py`
- `utils/medallion/gold/gold_modeling_common.py`
- `function_inventory.json`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__medallion__gold__gold_baseline_modeling_module_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_02_Baseline_Modeling_code_reference.md`
- `technical_reference/00_project_manual/notebook_relationship_map.md`
- `technical_reference/00_project_manual/notebook_dependency_matrix.md`
- `technical_reference/00_project_manual/artifact_and_table_handoff_map.md`
- `technical_reference/00_project_manual/medallion_handoff_map.md`

The active utility source file is the source of truth for function behavior. Workflow and manual references provide consumer and handoff context only.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
|---|---|---|
| `run_baseline_pipeline` | Coordinates baseline fit, train-score threshold calibration, scoring of fit/train/test/all frames, evaluation, and summary assembly. | Gold baseline modeling and comparison preparation. |
| `fit_baseline_isolation_forest` | Fits the baseline Isolation Forest on the supplied fit dataframe and returns model fit metadata. | Baseline model training on Gold fit rows. |
| `score_baseline_model` | Produces anomaly scores and optional binary prediction flags on a dataframe copy. | Baseline scoring for fit, train, test, and all-row frames. |
| `evaluate_baseline_model` | Evaluates scored baseline rows against labels using shared Gold metric helpers. | Baseline metric payload generation for model comparison. |

## Module-Level Technical Context

`gold_baseline_modeling.py` implements an in-memory baseline modeling workflow. It fits a single Isolation Forest, converts scikit-learn score direction into the project anomaly-score convention, calibrates a threshold from training scores, scores multiple dataframe partitions, and returns metrics that can be saved or compared by notebook callers. The selected functions do not write artifacts, update SQL, update ledgers, register W&B artifacts, or create truth records directly.

## Deep Function References

### "run_baseline_pipeline"

#### Functional Purpose

`run_baseline_pipeline` orchestrates the shared baseline Isolation Forest workflow. It fits the baseline model, scores training rows without a threshold to derive the calibration scores, selects a threshold by percentile, scores fit/train/test/all frames with that threshold, evaluates each scored frame against labels, and packages the model, scored frames, threshold, and summary payloads into one return dictionary.

#### Pipeline Context

This function supports the Gold baseline modeling stage represented by Gold_02. Project manual and workflow references confirm that Gold_02 provides baseline results, model artifacts, summary/threshold outputs, a validation contract, and `gold_baseline` lineage outputs to Gold_04 and Gold_06A. Direct invocation of this shared utility function by the Gold_02 notebook is Not determined from available source.

#### Inputs and Assumptions

- `fit_dataframe` is expected to contain the fit population for model training. The Gold workflow context describes this as Gold_01's normal-only fit data.
- `train_dataframe` is used for threshold calibration through train-only baseline scores.
- `test_dataframe` and `all_dataframe` are scored and evaluated after threshold selection.
- `feature_columns` should identify the modeling feature set. Missing columns are filtered by lower-level fit and score helpers.
- `threshold_percentile` controls the percentile cutoff used by `choose_threshold_by_percentile`.
- `model_params` is passed into `fit_baseline_isolation_forest` and may override Isolation Forest parameters such as `n_estimators`, `contamination`, `max_samples`, `max_features`, `bootstrap`, `random_state`, and `n_jobs`.
- `label_column`, `score_column_name`, and `prediction_column_name` define the evaluation and output column contract.

#### Outputs and Return Contract

The function returns a dictionary with:

- `model`: fitted Isolation Forest.
- `threshold`: selected anomaly-score threshold.
- `summary`: baseline summary dictionary.
- `scored_fit`, `scored_train`, `scored_test`, `scored_all`: scored dataframe copies with score and prediction columns.
- `train_scores_for_threshold`: one-column dataframe containing the train scores used for threshold selection.
- `train_scoring_info_no_threshold`: metadata from the pre-threshold train scoring pass.

The `summary` includes fit metadata, threshold metadata, metrics for fit/train/test/all frames, scoring metadata for each frame, and `baseline_metrics`, a compact model metric summary built from test metrics and feature metadata.

#### Side Effects

The function fits a model object and creates dataframe copies through supporting helpers. No external file writes, SQL writes, model persistence, ledger updates, W&B operations, path creation, or truth-record writes are confirmed in the active utility source.

#### Failure Behavior and Guardrails

- No usable feature columns or zero fit rows raise `ValueError` through `fit_baseline_isolation_forest`.
- Empty training scores raise `ValueError` through `choose_threshold_by_percentile`.
- Missing label, score, or prediction columns raise `ValueError` through `evaluate_baseline_model`.
- Invalid model parameters can fail through scikit-learn `IsolationForest`.
- The function does not add notebook-level guards such as zero-alert checks, truth-hash checks, SQL write checks, or artifact path checks.

#### Lineage and Reproducibility Role

The returned summary records model parameters, selected features, fit row count, threshold percentile and score distribution, scoring column names, row counts, and evaluation metrics. `random_state` is captured in the fit metadata when the default or caller-provided value is passed into the model. The function does not carry dataset IDs, run IDs, truth hashes, parent truth hashes, artifact paths, or pipeline mode directly.

#### Why This Function Matters

The baseline model is the reference point for the capstone's model comparison narrative. Centralizing fit, scoring, thresholding, and evaluation in one utility makes the baseline result easier to reproduce and compare against staged cascade variants.

#### Verification Method

- Confirm the returned `model` has been fitted by checking that it can score the selected feature frame.
- Confirm `summary["threshold_info"]["percentile"]` matches `threshold_percentile`.
- Confirm all scored frames contain `score_column_name` and `prediction_column_name`.
- Confirm `baseline_metrics["evaluation_metrics"]` matches `summary["test_metrics"]`.
- Confirm `train_scores_for_threshold` contains only the score column used for threshold calibration.
- Confirm no input dataframe was mutated by comparing input columns before and after the call.

### "fit_baseline_isolation_forest"

#### Functional Purpose

`fit_baseline_isolation_forest` fits the baseline Isolation Forest estimator on the supplied fit dataframe and selected feature columns. It returns both the fitted model and metadata describing the training matrix and model parameters.

#### Pipeline Context

This function supports the model-fit step for Gold baseline modeling. The Gold_02 workflow reference confirms that the baseline model is trained on Gold_01's normal-only fit rows and then saved by notebook code outside this utility. Direct invocation of this shared utility function by Gold_02 is Not determined from available source.

#### Inputs and Assumptions

- `fit_dataframe` must contain at least one row and at least one feature column after filtering.
- `feature_columns` may include columns not present in the dataframe; missing columns are ignored.
- Isolation Forest parameters are supplied through explicit keyword arguments with defaults: `n_estimators=200`, `contamination="auto"`, `max_samples="auto"`, `max_features=1.0`, `bootstrap=False`, `random_state=42`, and `n_jobs=-1`.
- The selected feature columns must be numeric or otherwise acceptable to scikit-learn.

#### Outputs and Return Contract

The function returns:

- A fitted `IsolationForest`.
- `fit_info`, containing `model_type`, `fit_row_count`, `feature_count`, `feature_columns`, and a `params` block with the model settings used for the fit.

#### Side Effects

The returned model is fitted in memory. No file persistence, artifact registration, SQL write, ledger update, or dataframe mutation is confirmed in the active utility source.

#### Failure Behavior and Guardrails

- If no usable feature columns remain after filtering, the function raises `ValueError`.
- If the fit matrix has zero rows, the function raises `ValueError`.
- Invalid parameter values or incompatible feature values can raise through scikit-learn during model construction or `fit`.

#### Lineage and Reproducibility Role

The fit metadata records the feature list, feature count, row count, and model parameters, including `random_state`. This makes it possible to audit whether two baseline runs used the same model setup. The function does not attach truth hashes, dataset IDs, run IDs, or artifact paths.

#### Why This Function Matters

The fit contract defines what the baseline model learns as normal behavior. If the fit dataframe or feature list changes, all downstream baseline scores, thresholds, and comparisons can change.

#### Verification Method

- Confirm `fit_info["feature_columns"]` contains only columns present in `fit_dataframe`.
- Confirm `fit_info["fit_row_count"]` equals the fit matrix row count.
- Confirm `fit_info["params"]["random_state"]` is stable for reproducible runs.
- Pass an empty fit dataframe and confirm `ValueError`.
- Pass only missing feature columns and confirm `ValueError`.

### "score_baseline_model"

#### Functional Purpose

`score_baseline_model` scores a dataframe with a fitted baseline Isolation Forest. It writes anomaly scores into a dataframe copy and, when a threshold is supplied, adds binary anomaly prediction flags.

#### Pipeline Context

This function supports baseline scoring across fit, train, test, and all-row frames. The Gold_02 workflow confirms that baseline scores feed result artifacts, model comparison, SQL persistence, and replay validation, but those side effects occur in notebook code or other helpers outside this utility.

#### Inputs and Assumptions

- `fitted_model` must be a fitted estimator compatible with the selected feature matrix.
- `dataframe` must contain usable feature columns after filtering.
- `feature_columns` are filtered to columns present in the dataframe.
- `score_column_name` defines the anomaly-score output column.
- `threshold` is optional. If supplied, it is used to create the binary prediction column.
- `prediction_column_name` is used only when `threshold` is not `None`.

#### Outputs and Return Contract

The function returns:

- A scored dataframe copy.
- `scoring_info`, containing row count, feature count, score column name, score minimum, score maximum, score mean, and, when thresholded, prediction column name and predicted-positive count.

Scores use the project convention from `compute_anomaly_scores_isolation_forest`: larger values mean more anomalous rows.

#### Side Effects

The function creates a dataframe copy and adds score/prediction columns to that copy. No source dataframe mutation or external side effects are confirmed from available source.

#### Failure Behavior and Guardrails

- Missing feature columns are filtered out, but the function does not add a custom check for an empty feature list before scoring.
- If the resulting feature frame is invalid, model scoring can fail through the fitted estimator or shared scoring helper.
- If `threshold` is `None`, no prediction column is created and no predicted-positive count is recorded.

#### Lineage and Reproducibility Role

The score metadata records the score column name, row count, feature count, and score distribution. When thresholding is used, it also records the prediction column and alert count. The function does not write truth metadata or artifact paths.

#### Why This Function Matters

The score column is the numerical basis for thresholding, evaluation, model comparison, and downstream anomaly analysis. Using a shared helper keeps the score direction and prediction rule consistent across baseline partitions.

#### Verification Method

- Confirm the returned dataframe contains `score_column_name`.
- Confirm the input dataframe does not gain the score column.
- Confirm `prediction_column_name` is present only when a threshold is supplied.
- Confirm `scoring_info["predicted_positive_count"]` equals the sum of the prediction column.
- Confirm higher anomaly scores are flagged when they are greater than or equal to the threshold.

### "evaluate_baseline_model"

#### Functional Purpose

`evaluate_baseline_model` computes baseline evaluation metrics from a scored dataframe. It compares the configured label column against the configured prediction column and passes the score column into the shared metric helper for ranking metrics.

#### Pipeline Context

This function supports Gold baseline metric generation for fit, train, test, and all-row frames. The returned metrics feed the baseline summary built by `run_baseline_pipeline` and can support downstream comparison artifacts produced by notebook code.

#### Inputs and Assumptions

- `scored_dataframe` must contain `label_column`, `score_column`, and `prediction_column`.
- Labels and predictions must be coercible to binary integers by the shared metric helper.
- Scores must be numeric or coercible to float for ROC AUC and average precision.

#### Outputs and Return Contract

The function returns the dictionary from `evaluate_against_labels`, including:

- `row_count`
- `positive_label_count`
- `predicted_positive_count`
- `precision`
- `recall`
- `f1`
- `roc_auc`
- `average_precision`

Ranking metrics are `None` when the shared metric helper cannot compute them.

#### Side Effects

No side effects are confirmed from available source. The function reads columns and returns a metrics dictionary.

#### Failure Behavior and Guardrails

- Missing label column raises `ValueError`.
- Missing score column raises `ValueError`.
- Missing prediction column raises `ValueError`.
- The shared metric helper uses `zero_division=0` for precision, recall, and F1 behavior and catches ranking metric failures by returning `None`.

#### Lineage and Reproducibility Role

The function does not handle artifact paths, truth hashes, or dataset/run identifiers. Its reproducibility role is metric-contract stability: the same label, prediction, and score columns produce the same metric payload shape for downstream summaries.

#### Why This Function Matters

Baseline evaluation is the comparison anchor for all cascade variants. A stable metric payload makes the baseline result comparable to cascade summaries and replay validation checks.

#### Verification Method

- Confirm required columns exist before evaluation.
- Confirm `row_count` matches the scored dataframe row count.
- Confirm `predicted_positive_count` matches the prediction column sum after coercion.
- Confirm missing required columns raise `ValueError`.
- Confirm `roc_auc` and `average_precision` are either floats or `None`.

## Cross-Function Relationships

- `run_baseline_pipeline` calls `fit_baseline_isolation_forest` to build the fitted model and fit metadata.
- It calls `score_baseline_model` once on training rows without a threshold to derive threshold-selection scores.
- It uses `choose_threshold_by_percentile` from the shared modeling helper to select the baseline threshold.
- It calls `score_baseline_model` again for fit, train, test, and all-row frames with the selected threshold.
- It calls `evaluate_baseline_model` on each scored frame and packages those metrics into `summary`.
- Manual and workflow references confirm that Gold_02 notebook outputs based on baseline modeling feed Gold_04 comparison and Gold_06A replay validation, but artifact persistence and SQL writes are outside this utility module.

## Source-Limited Items

- Direct notebook invocation of these shared baseline utility functions is Not determined from available source.
- File persistence for baseline results, baseline model joblib, thresholds JSON, summary JSON, metadata JSON, truth records, ledger output, SQL rows, and W&B artifacts is confirmed for the Gold_02 notebook workflow, not for these selected utility functions.
- Dataset ID, run ID, parent truth hash, pipeline mode, and artifact path handling inside these selected functions is Not determined from available source.
