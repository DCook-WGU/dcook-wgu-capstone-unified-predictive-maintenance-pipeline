# Synthetic Utility Reference: timestamp_stage_writer.py

Source path:

`utils/synthetic/pipeline/timestamp_stage_writer.py`

## Purpose

Assigns timestamps to staged synthetic observations.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `ensure_simulation_timing_config_table` | `engine` | Create the timing configuration table used to timestamp observations. |
| Function | `insert_simulation_timing_config` | `` | Insert or update the simulation timing configuration for a dataset/run. This function is intentionally idempotent. The database bootstrap may seed an initial timing row for the default synthetic dataset/run, and notebooks may rerun the same timing setup cell multiple times during development. Instead of failing on duplicate ``dataset_id`` / ``run_id`` values, this function updates the existing row. Parameters ---------- engine: SQLAlchemy engine connected to the project Postgres database. dataset_id: Logical dataset identifier, such as ``pump_synthetic_v1``. run_id: Logical synthetic run identifier, such as ``synthetic_run_001``. simulation_start_datetime: Timestamp used as the beginning of the generated synthetic timeline. sampling_interval_seconds: Number of seconds between generated observation timestamps. schema: Database schema containing the timing configuration table. table_name: Timing configuration table name. set_active: Whether this timing row should be marked active. deactivate_existing_for_run: When true, deactivate other active timing rows for the same dataset before activating this run. The current ``dataset_id`` / ``run_id`` row is then inserted or updated. |
| Function | `load_simulation_timing_config` | `engine` | Load the active timing configuration for one dataset/run pair. |
| Function | `_get_table_columns` | `engine` | Return source table columns in database ordinal order. |
| Function | `_validate_source_columns` | `columns` | Validate that the premelt table has timestamp-stage inputs. |
| Function | `_build_select_sql` | `` | Build SQL that derives observation timestamps from timing config. |
| Function | `_write_stage_sql_native` | `engine` | Create, append to, or fail on the timestamp target table in Postgres. |
| Function | `scalar_to_int` | `value, name` | Convert a scalar SQL result to int and reject missing values. |
| Function | `dataframe_row_count_to_int` | `dataframe` | Return a count value from a one-row dataframe as a plain int. |
| Function | `build_observations_timestamped_stage` | `engine` | Build the timestamped stage directly inside Postgres. `chunk_size` is kept only for backward-compatible notebook calls. It is not used in the SQL-native implementation. The stage keeps observation rows wide and adds `observation_timestamp` from the configured start time and sampling interval. |
| Function | `validate_observations_timestamped_stage` | `engine` | Return row-count and timestamp-range checks for the timestamped stage. |
| Function | `build_sensor_messages_timestamped_stage` | `` | Backward-compatible alias for `build_observations_timestamped_stage`. |
| Function | `validate_sensor_messages_timestamped_stage` | `` | Backward-compatible alias for `validate_observations_timestamped_stage`. |

## Configuration Dependencies

- Uses SQL schema or table name settings.
- Uses configuration dictionaries or resolved stage configuration.
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

- No direct artifact write pattern detected in this module.

## Downstream Usage

- `notebooks/synthetic/synthetic_03_sythetic_observations_timestamped_stage.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`
- `notebooks/synthetic/synthetic_pipeline_condensed-02_03.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
