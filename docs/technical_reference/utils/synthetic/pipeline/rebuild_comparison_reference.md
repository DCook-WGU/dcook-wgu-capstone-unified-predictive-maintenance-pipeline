# Synthetic Utility Reference: rebuild_comparison.py

Source path:

`utils/synthetic/pipeline/rebuild_comparison.py`

## Purpose

Compares rebuilt observations against source-stage expectations.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `_get_existing_columns` | `engine` | Return existing Postgres columns for a comparison-stage table. |
| Function | `_infer_alter_column_type` | `series` | Infer a conservative Postgres type for an added comparison column. |
| Function | `_add_missing_columns` | `engine` | Add dataframe columns that are missing from the comparison target table. |
| Function | `_build_sensor_columns` | `n_sensors` | Return sensor column names expected in premelt and rebuilt frames. |
| Function | `_normalize_missing_scalar` | `value` | Normalize pandas missing values before scalar comparison. |
| Function | `_compare_scalar` | `left, right` | Compare scalar values with tolerance for numeric-looking values. |
| Function | `_validate_premelt_columns` | `dataframe, n_sensors` | Validate original premelt columns needed for rebuild comparison. |
| Function | `_validate_rebuilt_columns` | `dataframe, n_sensors` | Validate rebuilt columns needed for field-by-field comparison. |
| Function | `load_premelt_for_comparison` | `engine` | Load original premelt observations for rebuild comparison. |
| Function | `load_rebuilt_for_comparison` | `engine` | Load rebuilt wide observations for comparison against premelt rows. |
| Function | `build_rebuild_comparison_dataframe` | `premelt_dataframe, rebuilt_dataframe` | Compare original premelt observations against rebuilt observations. The output flags row presence, per-field matches, total mismatch count, and notes that explain whether a row is missing or which fields differ. |
| Function | `ensure_rebuild_comparison_table_exists` | `engine` | Create the rebuild comparison table and mismatch lookup indexes. |
| Function | `_remove_existing_comparison_rows` | `engine` | Drop comparison rows whose observation keys already exist in the target. |
| Function | `write_rebuild_comparison_batch` | `engine, dataframe` | Append comparison rows after adding completion time and missing columns. |
| Function | `build_rebuild_comparison_stage` | `engine` | Build comparison rows in observation windows and write them to Postgres. |

## Configuration Dependencies

- Uses SQL schema or table name settings.
- Uses dataset identity values.
- Uses run or recipe identity values.

## Inputs and Outputs

Key inputs:
- Configuration values, dataset identity, run identity, or recipe identity
- Database engine, schema, table, or SQL runtime context
- Pandas dataframes or dataframe-like stage inputs

Key outputs:
- Dataframes or transformed stage outputs
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
- `notebooks/experiments/EDA_Notebook_Pump_Gold_02_Baseline_Modeling.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_04_Comparision.ipynb`
- `notebooks/experiments/EDA_Notebook_Pump_Gold_05_Anomaly_Detection.ipynb`
- `notebooks/preprocessing/EDA_Notebook_Pump_Bronze_01_Preprocessing.ipynb`
- `notebooks/synthetic/synthetic_10_rebuild_comparison.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
