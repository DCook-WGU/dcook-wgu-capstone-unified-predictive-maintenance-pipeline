# Synthetic Utility Reference: melt_stage_writer.py

Source path:

`utils/synthetic/pipeline/melt_stage_writer.py`

## Purpose

Transforms staged synthetic observations into sensor-message records.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `log_step_timing` | `step_name, start_time` | Print elapsed time for a melt-stage step and return a new timer. |
| Function | `_build_sensor_columns` | `n_sensors` | Build the expected wide sensor column names for a synthetic observation. |
| Function | `_validate_source_columns` | `dataframe, required_columns` | Validate that a timestamped observation frame has melt-stage inputs. |
| Function | `_extract_sensor_index` | `sensor_name_series` | Extract the numeric sensor index from names like `sensor_00`. |
| Function | `_build_message_sequence_index_with_rng` | `` | Build a randomized 0..(n_sensors-1) sequence for each observation using one shared RNG so chunking stays deterministic across the full run. |
| Function | `quote_ident` | `identifier` | Quote a SQL identifier for direct SQLAlchemy text statements. |
| Function | `fq_table` | `schema, table_name` | Return a quoted fully qualified table name. |
| Function | `get_table_columns` | `engine` | Return source table columns in database ordinal order. |
| Function | `ensure_sensor_columns_exist` | `engine` | Add missing wide sensor columns before SQL-native melting. |
| Function | `build_sensor_messages_stage` | `engine` | Build the long-format sensor message stage from the timestamped premelt observation stage in chunks instead of loading/melting the full table at once. The output has one row per observation/sensor pair and is the input shape used by later send-queue construction. |
| Function | `build_sensor_messages_stage_sql_native` | `engine` | Build the long sensor-message stage using a SQL CROSS JOIN LATERAL melt. |
| Function | `validate_sensor_messages_stage` | `engine` | Return row-count and sensor coverage checks for the melt stage table. |

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

- `notebooks/synthetic/synthetic_02_build_premilt_observations_stage.ipynb`
- `notebooks/synthetic/synthetic_04_build_sensor_messages_stage.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`
- `notebooks/synthetic/synthetic_pipeline_condensed-02_03.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
