# Utility Module Reference: `utils/synthetic/generator/missingness.py`

## Module Purpose

This module replays or applies missingness patterns to synthetic telemetry.

## Pipeline Role

- Stage support: Synthetic generator
- Primary responsibility: This module replays or applies missingness patterns to synthetic telemetry.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Silver_01_PreEDA`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `MissingnessSpec` | Missingness audit settings reconstructed from a parent truth record. | deep |
| `_payload_mapping` | Return a mapping payload value or raise a typed shape error. | deep |
| `_payload_str_float_dict` | Normalize a flat payload mapping to dict[str, float]. | short |
| `_payload_str_bool_dict` | Normalize a flat payload mapping to dict[str, bool]. | short |
| `_payload_nested_str_float_dict` | Normalize a nested payload mapping to dict[str, dict[str, float]]. | deep |
| `_payload_string_list` | Normalize a payload scalar or iterable to a cleaned string list. | deep |
| `build_missingness_spec_from_truth_payload` | Build a MissingnessSpec from truth.runtime_facts.missingness_quarantine. | deep |
| `_pct_to_present_count` | Convert a missing percentage into a clamped present-row count. | deep |
| `apply_exact_missingness_mask` | Mask eligible rows so each sensor keeps exactly its target count. | deep |
| `apply_clustered_missingness_mask` | Apply clustered NaN runs while matching each sensor's present count. | deep |
| `build_present_counts_for_block` | Build per-sensor present counts for one phase or state block. | deep |

## Configuration Dependencies

- No explicit configuration dependency was determined from available source.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_payload_mapping` | `payload, key` | Return a mapping payload value or raise a typed shape error. |
| `_payload_str_float_dict` | `payload, key` | Normalize a flat payload mapping to dict[str, float]. |
| `_payload_str_bool_dict` | `payload, key` | Normalize a flat payload mapping to dict[str, bool]. |
| `_payload_nested_str_float_dict` | `payload, key` | Normalize a nested payload mapping to dict[str, dict[str, float]]. |
| `_payload_string_list` | `payload, key` | Normalize a payload scalar or iterable to a cleaned string list. |
| `build_missingness_spec_from_truth_payload` | `payload` | Build a MissingnessSpec from truth.runtime_facts.missingness_quarantine. |
| `_pct_to_present_count` | `n, missing_pct` | Convert a missing percentage into a clamped present-row count. |
| `apply_exact_missingness_mask` | `df, *, sensor_cols, rng, present_counts, eligible_row_idx` | Mask eligible rows so each sensor keeps exactly its target count. |
| `apply_clustered_missingness_mask` | `df, *, sensor_cols, rng, present_counts, eligible_row_idx, mean_gap_len, long_gap_probability` | Apply clustered NaN runs while matching each sensor's present count. |
| `build_present_counts_for_block` | `*, sensors, n_rows, pct_all, pct_by_state, use_by_state` | Build per-sensor present counts for one phase or state block. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `TypeError` for invalid input, missing context, or failed validation paths.
- Source raises `a` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Silver_01_PreEDA`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Module Importance

This module matters because the synthetic subsystem depends on repeatable staging, truth handling, streaming handoff, or final alignment behavior to make controlled pump telemetry tests explainable.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
