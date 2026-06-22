# Utility Module Reference: `utils/medallion/gold/gold_cascade_modeling.py`

## Module Purpose

This module supports staged Gold cascade modeling, including model fitting, scoring, and output assembly.

## Pipeline Role

- Stage support: Gold
- Primary responsibility: This module supports staged Gold cascade modeling, including model fitting, scoring, and output assembly.

## Primary Consumers

`EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_array_to_float_list` | Convert NumPy/pandas score outputs to a plain Python list of floats. | short |
| `_array_to_object_list` | Convert NumPy/pandas label outputs to a plain Python list for metrics calls. | deep |
| `_series_to_float_list` | Convert a pandas Series to Sequence[float] for shared metric utilities. | short |
| `_series_to_object_list` | Convert a pandas Series to Sequence[Any] for shared metric utilities. | short |
| `_evaluate_scored_frame` | Evaluate a scored cascade frame with Pylance-safe sequence conversions. | deep |
| `fit_stage1_model` | Fit the broad Stage 1 Isolation Forest. | deep |
| `fit_stage2_model` | Fit the narrower Stage 2 Isolation Forest. | deep |
| `_score_stage_dataframe` | Score a dataframe with a fitted Isolation Forest stage model. | medium |
| `evaluate_stage2_model_with_thresholds` | Evaluate multiple Stage 2 threshold percentiles against labels. | deep |
| `run_stage2_selection` | Select the Stage 2 threshold from a threshold grid. | deep |
| `_variant_defaults` | Return variant-specific defaults for cascade behavior. | deep |
| `run_cascade_pipeline` | Run the end-to-end three-stage cascade modeling workflow. | deep |
| `build_cascade_summary` | Add compact comparison-friendly fields to a cascade summary. | deep |

## Configuration Dependencies

- Resolved pipeline configuration dictionaries and YAML-derived stage settings.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_array_to_float_list` | `values` | Convert NumPy/pandas score outputs to a plain Python list of floats. |
| `_array_to_object_list` | `values` | Convert NumPy/pandas label outputs to a plain Python list for metrics calls. |
| `_series_to_float_list` | `series` | Convert a pandas Series to Sequence[float] for shared metric utilities. |
| `_series_to_object_list` | `series` | Convert a pandas Series to Sequence[Any] for shared metric utilities. |
| `_evaluate_scored_frame` | `scored_frame, *, label_column, prediction_column, score_column` | Evaluate a scored cascade frame with Pylance-safe sequence conversions. |
| `fit_stage1_model` | `fit_dataframe, *, feature_columns, model_params` | Fit the broad Stage 1 Isolation Forest. |
| `fit_stage2_model` | `fit_dataframe, *, feature_columns, model_params` | Fit the narrower Stage 2 Isolation Forest. |
| `_score_stage_dataframe` | `fitted_model, dataframe, *, feature_columns, score_column, threshold, prediction_column` | Score a dataframe with a fitted Isolation Forest stage model. |
| `evaluate_stage2_model_with_thresholds` | `fitted_model, dataframe, *, feature_columns, label_column, threshold_percentiles, score_column, prediction_column` | Evaluate multiple Stage 2 threshold percentiles against labels. |
| `run_stage2_selection` | `fitted_model, validation_dataframe, *, feature_columns, label_column, threshold_percentiles, min_recall, optimization_metric` | Select the Stage 2 threshold from a threshold grid. |
| `_variant_defaults` | `variant` | Return variant-specific defaults for cascade behavior. |
| `run_cascade_pipeline` | `*, fit_dataframe, train_dataframe, test_dataframe, all_dataframe, stage1_feature_columns, stage2_feature_columns, reference_profile, stage3_sensor_groups, label_column, stage1_model_params, stage2_model_params, variant` | Run the end-to-end three-stage cascade modeling workflow. |
| `build_cascade_summary` | `*, variant, summary, stage1_feature_columns, stage2_feature_columns` | Add compact comparison-friendly fields to a cascade summary. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.

## Artifact / SQL / File-System Interactions

- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`

## Module Importance

This module matters because Gold notebooks depend on stable shared helpers for model input preparation, cascade modeling, evaluation, validation contracts, and artifact traceability.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
