# Synthetic Utility Reference: run.py

Source path:

`utils/synthetic/generator/run.py`

## Purpose

Orchestrates the synthetic generator workflow from configuration through export.

## Pipeline Role

Generator-side utility used before the staged PostgreSQL/Kafka synthetic pipeline. It helps create, shape, or export synthetic pump telemetry.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `parse_args` | `` | Parse CLI options for the standalone synthetic generation stage. |
| Function | `main` | `` | Run the generator stage from config, parent truth, and CLI overrides. The script loads Silver EDA artifacts from a parent truth record, builds a SyntheticGenerator, writes the generated episode artifact, and records the synthetic truth metadata needed by downstream stages. |

## Configuration Dependencies

- Uses configuration dictionaries or resolved stage configuration.
- Uses dataset identity values.
- Uses filesystem paths or resolved artifact locations.
- Uses run or recipe identity values.

## Inputs and Outputs

Key inputs:
- Configuration values, dataset identity, run identity, or recipe identity
- Filesystem paths and artifact files
- Pandas dataframes or dataframe-like stage inputs

Key outputs:
- Dataframes or transformed stage outputs
- File-based artifacts or metadata outputs

## Logging, Ledger, and Artifact Behavior

### Logging

- No direct logger calls detected in this module.

### Ledger

- No direct ledger behavior detected in this module.

### SQL/database

- No direct SQL/database behavior detected in this module.

### Artifacts

- Writes or prepares files/artifacts such as CSV, Parquet, JSON, or metadata outputs.

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
- `notebooks/experiments/EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_06B_Test_Early_Warning_Validation.ipynb`
- `notebooks/orchestrator_v1.ipynb`
- `notebooks/preprocessing/EDA_Notebook_Pump_Bronze_01_Preprocessing.ipynb`
- `notebooks/synthetic/synthetic_00_postgres_to_bronze_no_kafka.ipynb`
- `notebooks/synthetic/synthetic_01_generate_synethic_data.ipynb`
- `notebooks/synthetic/synthetic_02_build_premilt_observations_stage.ipynb`
- `notebooks/synthetic/synthetic_03_sythetic_observations_timestamped_stage.ipynb`
- `notebooks/synthetic/synthetic_04_build_sensor_messages_stage.ipynb`
- `notebooks/synthetic/synthetic_05_build_send_queue_stage.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
