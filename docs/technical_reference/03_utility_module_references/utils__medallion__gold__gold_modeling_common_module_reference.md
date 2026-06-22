# Utility Module Reference: `utils/medallion/gold/gold_modeling_common.py`

## Module Purpose

This module provides shared Gold modeling utilities for feature preparation, metrics, model artifacts, and evaluation support.

## Pipeline Role

- Stage support: Gold
- Primary responsibility: This module provides shared Gold modeling utilities for feature preparation, metrics, model artifacts, and evaluation support.

## Primary Consumers

`EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `compute_anomaly_scores_isolation_forest` | Compute anomaly scores from a fitted Isolation Forest. | deep |
| `choose_threshold_by_percentile` | Choose an anomaly threshold from a score percentile. | deep |
| `build_prediction_flags_from_scores` | Convert anomaly scores to binary prediction flags. | medium |
| `evaluate_against_labels` | Evaluate binary predictions against labels. | deep |
| `build_model_metric_summary` | Build a compact model metric summary payload. | deep |

## Configuration Dependencies

- No explicit configuration dependency was determined from available source.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `compute_anomaly_scores_isolation_forest` | `fitted_model, feature_frame` | Compute anomaly scores from a fitted Isolation Forest. |
| `choose_threshold_by_percentile` | `scores, *, percentile` | Choose an anomaly threshold from a score percentile. |
| `build_prediction_flags_from_scores` | `scores, *, threshold` | Convert anomaly scores to binary prediction flags. |
| `evaluate_against_labels` | `true_labels, predicted_labels, *, scores` | Evaluate binary predictions against labels. |
| `build_model_metric_summary` | `*, model_name, threshold, threshold_info, evaluation_metrics, feature_columns` | Build a compact model metric summary payload. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`

## Module Importance

This module matters because Gold notebooks depend on stable shared helpers for model input preparation, cascade modeling, evaluation, validation contracts, and artifact traceability.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
