# Synthetic Utility Reference: missingness.py

Source path:

`utils/synthetic/generator/missingness.py`

## Purpose

Applies controlled missingness patterns to synthetic pump telemetry.

## Pipeline Role

Generator-side utility used before the staged PostgreSQL/Kafka synthetic pipeline. It helps create, shape, or export synthetic pump telemetry.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Class | `MissingnessSpec` | `` | Missingness audit settings reconstructed from a parent truth record. missingness_pct_all: dict[sensor] -> percent missing (0..100) missingness_pct_by_state: dict[state] -> dict[sensor] -> percent missing (0..100) missingness_state_dependent_flag: dict[sensor] -> bool |
| Function | `_payload_mapping` | `payload, key` | Return a mapping payload value or raise a typed shape error. |
| Function | `_payload_str_float_dict` | `payload, key` | Normalize a flat payload mapping to dict[str, float]. |
| Function | `_payload_str_bool_dict` | `payload, key` | Normalize a flat payload mapping to dict[str, bool]. |
| Function | `_payload_nested_str_float_dict` | `payload, key` | Normalize a nested payload mapping to dict[str, dict[str, float]]. |
| Function | `_payload_string_list` | `payload, key` | Normalize a payload scalar or iterable to a cleaned string list. |
| Function | `build_missingness_spec_from_truth_payload` | `payload` | Build a MissingnessSpec from truth.runtime_facts.missingness_quarantine. Permanent behavior: - use missingness_pct_all / missingness_pct_by_state / dependency flags when present - merge dropped-sensor global missingness from `dropped_missing_pct` - optionally merge dropped-sensor state missingness from `dropped_missing_pct_by_state` if that key is added to the payload later - ensure dropped sensors get a default dependency flag - normalize the historical typo in state_col_synth |
| Function | `_pct_to_present_count` | `n, missing_pct` | Convert a missing percentage into a clamped present-row count. |
| Function | `apply_exact_missingness_mask` | `df` | Mask eligible rows so each sensor keeps exactly its target count. Rows outside eligible_row_idx are not touched. This gives the generator a strict missingness option when clustered gaps are not needed. |
| Function | `apply_clustered_missingness_mask` | `df` | Apply clustered NaN runs while matching each sensor's present count. The mask creates short and occasional longer gaps inside eligible rows so synthetic missingness resembles sensor dropout instead of uniform removal. |
| Function | `build_present_counts_for_block` | `` | Build per-sensor present counts for one phase or state block. |

## Configuration Dependencies

- No direct configuration dependency was detected by static review.

## Inputs and Outputs

Key inputs:
- Pandas dataframes or dataframe-like stage inputs

Key outputs:
- Dataframes or transformed stage outputs

## Logging, Ledger, and Artifact Behavior

### Logging

- No direct logger calls detected in this module.

### Ledger

- No direct ledger behavior detected in this module.

### SQL/database

- No direct SQL/database behavior detected in this module.

### Artifacts

- No direct artifact write pattern detected in this module.

## Downstream Usage

- `notebooks/eda/EDA_Notebook_Pump_Silver_01_PreEDA.ipynb`
- `notebooks/eda/EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3.ipynb`
- `notebooks/eda/EDA_Notebook_Pump_Silver_02b_EDA_v2.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_01_PreProcessing.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_02_Baseline_Modeling.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_04_Comparision.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_05_Anomaly_Detection.ipynb`
- `notebooks/orchestrator_v1.ipynb`
- `notebooks/preprocessing/EDA_Notebook_Pump_Bronze_01_Preprocessing.ipynb`
- `notebooks/synthetic/synthetic_01_generate_synethic_data.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`
- `notebooks/synthetic/synthetic_pipeline_testing_export_csv.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
