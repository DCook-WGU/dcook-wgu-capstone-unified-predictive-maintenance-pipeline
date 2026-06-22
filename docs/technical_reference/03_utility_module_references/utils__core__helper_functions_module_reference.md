# Utility Module Reference: `utils/core/helper_functions.py`

## Module Purpose

This module contains shared dataframe and notebook convenience helpers that do not belong to a specific Medallion layer.

## Pipeline Role

- Stage support: Core / cross-stage
- Primary responsibility: This module contains shared dataframe and notebook convenience helpers that do not belong to a specific Medallion layer.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`, `EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation`, `EDA_Notebook_Pump_Silver_01_PreEDA`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `cfg_require_mapping` | Validate that a config value is a mapping and return it with mapping type hints. | deep |
| `cfg_optional_mapping` | Return an optional config mapping, using an empty mapping when omitted. | short |
| `require_dict` | Return a dictionary value, using an empty dict for None. | deep |
| `require_list` | Return a list value, using an empty list for None. | deep |
| `scalar_to_float` | Convert a present scalar value to float, rejecting None, pandas NA, and NaN. | deep |
| `normalize_feature_columns` | Normalize feature-column results into list[str]. | deep |
| `require_mapping` | Validate that a loaded JSON/config object is a dictionary. | deep |
| `require_str_list` | Validate that a loaded JSON/config object is a list of strings. | deep |
| `require_float` | Convert a scalar or threshold-return tuple into a float. | deep |
| `as_bool_array` | Convert a Pandas/NumPy boolean mask into a NumPy bool array. | short |
| `as_int_array` | Convert labels/flags into a NumPy int array. | deep |
| `as_float_array` | Convert scores into a flat NumPy float array. | deep |
| `choose_threshold_by_percentile` | Choose anomaly threshold using a score percentile. | deep |
| `choose_threshold_value` | Normalize score input and threshold helper output for Pylance. | short |
| `normalize_text_value` | Convert any scalar value into lowercase stripped text. | short |
| `get_nested_mapping` | Safely extract a nested dictionary from a truth/config record. | deep |
| `require_truth_record` | Validate a loaded truth record. | deep |
| `require_int_value` | Validate that a nullable integer value is present before converting it. | deep |

## Configuration Dependencies

- No explicit configuration dependency was determined from available source.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `cfg_require_mapping` | `value, name` | Validate that a config value is a mapping and return it with mapping type hints. |
| `cfg_optional_mapping` | `value, name` | Return an optional config mapping, using an empty mapping when omitted. |
| `require_dict` | `value, name` | Return a dictionary value, using an empty dict for None. |
| `require_list` | `value, name` | Return a list value, using an empty list for None. |
| `scalar_to_float` | `value, name` | Convert a present scalar value to float, rejecting None, pandas NA, and NaN. |
| `normalize_feature_columns` | `value, name` | Normalize feature-column results into list[str]. |
| `require_mapping` | `value, name` | Validate that a loaded JSON/config object is a dictionary. |
| `require_str_list` | `value, name` | Validate that a loaded JSON/config object is a list of strings. |
| `require_float` | `value, name` | Convert a scalar or threshold-return tuple into a float. |
| `as_bool_array` | `value, name` | Convert a Pandas/NumPy boolean mask into a NumPy bool array. |
| `as_int_array` | `value, name` | Convert labels/flags into a NumPy int array. |
| `as_float_array` | `value, name` | Convert scores into a flat NumPy float array. |
| `choose_threshold_by_percentile` | `scores, percentile, *, return_info` | Choose anomaly threshold using a score percentile. |
| `choose_threshold_value` | `scores, percentile` | Normalize score input and threshold helper output for Pylance. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `TypeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`, `EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation`, `EDA_Notebook_Pump_Silver_01_PreEDA`

## Module Importance

This module matters because shared configuration, paths, logging, ledger, truth, artifact, and optional W&B behavior reduce duplicated notebook setup.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
