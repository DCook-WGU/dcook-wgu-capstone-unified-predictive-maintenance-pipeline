# Utility Module Reference: `utils/synthetic/generator/profiles.py`

## Module Purpose

This module loads and represents synthetic sensor/state profiles used by the generator.

## Pipeline Role

- Stage support: Synthetic generator
- Primary responsibility: This module loads and represents synthetic sensor/state profiles used by the generator.

## Primary Consumers

`EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation`, `EDA_Notebook_Pump_Gold_06B_Test_Early_Warning_Validation`, `EDA_Notebook_Pump_Silver_01_PreEDA`, `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `SensorRichProfile` | Sensor distribution summary used by the generator for one state scope. | short |
| `_require_columns` | Raise a clear error when an input profile artifact is missing columns. | deep |
| `load_rich_profile_csv` | Load a Silver EDA rich-profile CSV into generator profile objects. | deep |
| `load_correlation_pairs_csv` | Load and normalize pairwise sensor correlations for generator use. | deep |
| `load_group_map_csv` | Load the sensor-to-group map used for correlated group movement. | medium |
| `load_fault_pairings_csv` | Load primary-to-secondary fault propagation settings. | deep |
| `merge_profile_dicts` | Merge profile dictionaries, with extra profiles replacing base keys. | short |
| `load_and_merge_rich_profiles` | Load base profiles and optional dropped-sensor profiles for one state. | medium |

## Configuration Dependencies

- No explicit configuration dependency was determined from available source.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_require_columns` | `dataframe, required, name` | Raise a clear error when an input profile artifact is missing columns. |
| `load_rich_profile_csv` | `path, state_scope` | Load a Silver EDA rich-profile CSV into generator profile objects. |
| `load_correlation_pairs_csv` | `path` | Load and normalize pairwise sensor correlations for generator use. |
| `load_group_map_csv` | `path` | Load the sensor-to-group map used for correlated group movement. |
| `load_fault_pairings_csv` | `path` | Load primary-to-secondary fault propagation settings. |
| `merge_profile_dicts` | `base, extra` | Merge profile dictionaries, with extra profiles replacing base keys. |
| `load_and_merge_rich_profiles` | `*, base_profile_csv_path, state_scope, dropped_profile_csv_path` | Load base profiles and optional dropped-sensor profiles for one state. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation`, `EDA_Notebook_Pump_Gold_06B_Test_Early_Warning_Validation`, `EDA_Notebook_Pump_Silver_01_PreEDA`, `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Module Importance

This module matters because the synthetic subsystem depends on repeatable staging, truth handling, streaming handoff, or final alignment behavior to make controlled pump telemetry tests explainable.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
