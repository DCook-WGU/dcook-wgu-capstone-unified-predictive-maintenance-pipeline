# Utility Module Reference: `utils/medallion/gold/gold_cascade_stage3_rules.py`

## Module Purpose

This module applies Stage 3 cascade decision rules and threshold logic to Gold model outputs.

## Pipeline Role

- Stage support: Gold
- Primary responsibility: This module applies Stage 3 cascade decision rules and threshold logic to Gold model outputs.

## Primary Consumers

`EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `compute_primary_breach_count` | Count strong feature breaches against the reference profile. | medium |
| `compute_secondary_breach_count` | Count corroborating breaches across sensor groups. | medium |
| `compute_persistence_flag` | Flag rows that belong to persistent candidate-anomaly runs. | deep |
| `compute_drift_flag` | Flag local score drift from rolling mean shifts. | deep |
| `compose_stage3_decision` | Compose the final Stage 3 confirmation flag from rule evidence. | medium |

## Configuration Dependencies

- No explicit configuration dependency was determined from available source.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `compute_primary_breach_count` | `dataframe, *, feature_columns, reference_profile, z_threshold, output_column` | Count strong feature breaches against the reference profile. |
| `compute_secondary_breach_count` | `dataframe, *, sensor_groups, reference_profile, z_threshold, output_column` | Count corroborating breaches across sensor groups. |
| `compute_persistence_flag` | `dataframe, *, candidate_column, group_columns, min_consecutive_rows, output_column` | Flag rows that belong to persistent candidate-anomaly runs. |
| `compute_drift_flag` | `dataframe, *, score_column, group_columns, rolling_window, min_drift_delta, output_column` | Flag local score drift from rolling mean shifts. |
| `compose_stage3_decision` | `dataframe, *, primary_breach_column, secondary_breach_column, persistence_column, drift_column, min_primary_breaches, min_secondary_breaches, require_persistence_or_drift, output_column` | Compose the final Stage 3 confirmation flag from rule evidence. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Not determined from available source

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation`

## Module Importance

This module matters because Gold notebooks depend on stable shared helpers for model input preparation, cascade modeling, evaluation, validation contracts, and artifact traceability.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
