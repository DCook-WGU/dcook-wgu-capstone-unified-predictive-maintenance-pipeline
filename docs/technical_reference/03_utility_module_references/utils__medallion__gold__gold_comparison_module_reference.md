# Utility Module Reference: `utils/medallion/gold/gold_comparison.py`

## Module Purpose

This module compares Gold model outputs and summarizes baseline-versus-cascade evaluation results.

## Pipeline Role

- Stage support: Gold
- Primary responsibility: This module compares Gold model outputs and summarizes baseline-versus-cascade evaluation results.

## Primary Consumers

`EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `load_model_result_artifacts` | Normalize baseline and cascade summaries into one comparison payload. | deep |
| `validate_comparison_inputs` | Validate that baseline and cascade summaries can be compared together. | deep |
| `build_model_comparison_dataframe` | Build one comparison dataframe containing baseline plus cascade variants. | deep |
| `build_alert_count_comparison` | Build a focused alert-count comparison table. | deep |
| `build_metric_comparison` | Build a focused metric comparison table. | deep |
| `build_comparison_summary` | Build a compact summary for reporting and downstream display. | deep |
| `build_baseline_vs_best_cascade_delta` | Compare the baseline row against the best cascade row by F1. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.
- Dataset, run, stage, or recipe identifiers used for traceability.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `load_model_result_artifacts` | `*, baseline_summary, cascade_summaries` | Normalize baseline and cascade summaries into one comparison payload. |
| `validate_comparison_inputs` | `comparison_payload` | Validate that baseline and cascade summaries can be compared together. |
| `build_model_comparison_dataframe` | `comparison_payload` | Build one comparison dataframe containing baseline plus cascade variants. |
| `build_alert_count_comparison` | `comparison_payload` | Build a focused alert-count comparison table. |
| `build_metric_comparison` | `comparison_payload` | Build a focused metric comparison table. |
| `build_comparison_summary` | `comparison_df` | Build a compact summary for reporting and downstream display. |
| `build_baseline_vs_best_cascade_delta` | `comparison_df` | Compare the baseline row against the best cascade row by F1. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.

## Failure Behavior

- Source raises `for` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Module Importance

This module matters because Gold notebooks depend on stable shared helpers for model input preparation, cascade modeling, evaluation, validation contracts, and artifact traceability.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
