# Synthetic Utility Reference: generator.py

Source path:

`utils/synthetic/generator/generator.py`

## Purpose

Generates synthetic pump telemetry from profile statistics and configured state behavior.

## Pipeline Role

Generator-side utility used before the staged PostgreSQL/Kafka synthetic pipeline. It helps create, shape, or export synthetic pump telemetry.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `_as_float` | `value, default` | Coerce optional tuning values to float with a default fallback. |
| Function | `_as_int` | `value, default` | Coerce optional tuning values to int with a default fallback. |
| Function | `_as_object_dict` | `value` | Return a string-keyed dict when a tuning value is mapping-like. |
| Function | `_as_object_list` | `value` | Return a list for list/tuple tuning values and [] otherwise. |
| Function | `_pair_key` | `left, right` | Build a stable unordered key for a sensor pair. |
| Class | `EpisodeSpec` | `` | Row-count and fault settings for one generated synthetic episode. |
| Class | `SyntheticGenerator` | `` | Generate synthetic pump telemetry from Silver EDA profile artifacts. The generator samples bounded sensor values, adds correlation structure, injects configured fault behavior, stamps row-level truth metadata, and can replay missingness patterns from a parent truth record. |

## Configuration Dependencies

- Uses configuration dictionaries or resolved stage configuration.
- Uses filesystem paths or resolved artifact locations.

## Inputs and Outputs

Key inputs:
- Configuration values, dataset identity, run identity, or recipe identity
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
- `notebooks/orchestrator_v1.ipynb`
- `notebooks/preprocessing/EDA_Notebook_Pump_Bronze_01_Preprocessing.ipynb`
- `notebooks/synthetic/synthetic_01_generate_synethic_data.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
