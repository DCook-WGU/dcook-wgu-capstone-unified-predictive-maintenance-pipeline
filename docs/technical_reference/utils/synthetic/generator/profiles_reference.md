# Synthetic Utility Reference: profiles.py

Source path:

`utils/synthetic/generator/profiles.py`

## Purpose

Loads and structures sensor profile statistics used by synthetic telemetry generation.

## Pipeline Role

Generator-side utility used before the staged PostgreSQL/Kafka synthetic pipeline. It helps create, shape, or export synthetic pump telemetry.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Class | `SensorRichProfile` | `` | Sensor distribution summary used by the generator for one state scope. The values come from Silver EDA profile exports and give the synthetic generator enough information to sample bounded, state-specific telemetry. |
| Function | `_require_columns` | `dataframe, required, name` | Raise a clear error when an input profile artifact is missing columns. |
| Function | `load_rich_profile_csv` | `path, state_scope` | Load a Silver EDA rich-profile CSV into generator profile objects. Each row becomes one SensorRichProfile keyed by sensor name. The function validates the expected export shape before coercing numeric profile fields. |
| Function | `load_correlation_pairs_csv` | `path` | Load and normalize pairwise sensor correlations for generator use. The loader accepts both the newer pearson/spearman export shape and the older signed correlation export shape so the generator can reuse either Silver EDA artifact version without changing downstream code. |
| Function | `load_group_map_csv` | `path` | Load the sensor-to-group map used for correlated group movement. |
| Function | `load_fault_pairings_csv` | `path` | Load primary-to-secondary fault propagation settings. |
| Function | `merge_profile_dicts` | `base, extra` | Merge profile dictionaries, with extra profiles replacing base keys. |
| Function | `load_and_merge_rich_profiles` | `` | Load base profiles and optional dropped-sensor profiles for one state. dropped_profile_csv_path should point to one of: pump__silver_eda__dropped_feature_profiles__normal.csv pump__silver_eda__dropped_feature_profiles__abnormal.csv pump__silver_eda__dropped_feature_profiles__recovery.csv |

## Configuration Dependencies

- Uses filesystem paths or resolved artifact locations.

## Inputs and Outputs

Key inputs:
- Database engine, schema, table, or SQL runtime context
- Filesystem paths and artifact files
- Pandas dataframes or dataframe-like stage inputs

Key outputs:
- Dataframes or transformed stage outputs
- File-based artifacts or metadata outputs
- SQL table rows, status updates, or database-stage records

## Logging, Ledger, and Artifact Behavior

### Logging

- No direct logger calls detected in this module.

### Ledger

- No direct ledger behavior detected in this module.

### SQL/database

- Uses SQL, PostgreSQL, engine, table, or database write/read behavior.

### Artifacts

- Writes or prepares files/artifacts such as CSV, Parquet, JSON, or metadata outputs.

## Downstream Usage

- `notebooks/eda/EDA_Notebook_Pump_Silver_01_PreEDA.ipynb`
- `notebooks/eda/EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3.ipynb`
- `notebooks/eda/EDA_Notebook_Pump_Silver_02b_EDA_v2.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_01_PreProcessing.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation.ipynb`
- `notebooks/orchestrator_v1.ipynb`
- `notebooks/preprocessing/EDA_Notebook_Pump_Bronze_01_Preprocessing.ipynb`
- `notebooks/synthetic/synthetic_01_generate_synethic_data.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
