# Utility Module Reference: `utils/medallion/gold/gold_baseline_modeling.py`

## Module Purpose

This module builds baseline Gold anomaly-detection model inputs, model outputs, and evaluation summaries.

## Pipeline Role

- Stage support: Gold
- Primary responsibility: This module builds baseline Gold anomaly-detection model inputs, model outputs, and evaluation summaries.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_array_to_float_list` | Convert numpy/pandas/scalar score values into a plain list[float]. | short |
| `_series_to_float_list` | Convert a pandas Series into list[float] for typed metric helpers. | short |
| `_series_to_object_list` | Convert a pandas Series into list[Any] for typed label helpers. | deep |
| `fit_baseline_isolation_forest` | Fit the baseline Isolation Forest on normal-only Gold fit rows. | deep |
| `score_baseline_model` | Score a dataframe with the fitted baseline model. | deep |
| `evaluate_baseline_model` | Evaluate a scored baseline dataframe against anomaly labels. | deep |
| `run_baseline_pipeline` | Run the end-to-end baseline Isolation Forest workflow. | deep |

## Configuration Dependencies

- No explicit configuration dependency was determined from available source.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_array_to_float_list` | `values` | Convert numpy/pandas/scalar score values into a plain list[float]. |
| `_series_to_float_list` | `series` | Convert a pandas Series into list[float] for typed metric helpers. |
| `_series_to_object_list` | `series` | Convert a pandas Series into list[Any] for typed label helpers. |
| `fit_baseline_isolation_forest` | `fit_dataframe, *, feature_columns, n_estimators, contamination, max_samples, max_features, bootstrap, random_state, n_jobs` | Fit the baseline Isolation Forest on normal-only Gold fit rows. |
| `score_baseline_model` | `fitted_model, dataframe, *, feature_columns, score_column_name, threshold, prediction_column_name` | Score a dataframe with the fitted baseline model. |
| `evaluate_baseline_model` | `scored_dataframe, *, label_column, score_column, prediction_column` | Evaluate a scored baseline dataframe against anomaly labels. |
| `run_baseline_pipeline` | `*, fit_dataframe, train_dataframe, test_dataframe, all_dataframe, feature_columns, threshold_percentile, model_params, score_column_name, prediction_column_name, label_column` | Run the end-to-end baseline Isolation Forest workflow. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Not determined from available source

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because Gold notebooks depend on stable shared helpers for model input preparation, cascade modeling, evaluation, validation contracts, and artifact traceability.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
