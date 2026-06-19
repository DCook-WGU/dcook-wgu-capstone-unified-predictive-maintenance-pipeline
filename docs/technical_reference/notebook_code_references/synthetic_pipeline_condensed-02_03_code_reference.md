# Notebook Code Reference: synthetic_pipeline_condensed-02_03

Notebook path:

`notebooks/synthetic/synthetic_pipeline_condensed-02_03.ipynb`

## Notebook Purpose

This notebook documents and runs a focused support step in the synthetic pump telemetry workflow.

Notebook stage:

`Synthetic support`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08, Code Cell 09, Code Cell 10, Code Cell 11, Code Cell 12, Code Cell 13, Code Cell 14, Code Cell 15, Code Cell 16, Code Cell 17, Code Cell 18, Code Cell 19, Code Cell 20, Code Cell 21, Code Cell 22, Code Cell 23, Code Cell 24, Code Cell 25 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_observations_premelt_stage`
- `build_observations_timestamped_stage`
- `database`
- `datetime`
- `ensure_simulation_timing_config_table`
- `get_engine_from_env`
- `insert_simulation_timing_config`
- `os`
- `pipeline`
- `postgres`
- `premelt_stage_writer`
- `read_sql_dataframe`
- `synthetic`
- `timedelta`
- `timestamp_stage_writer`
- `utils`
- `validate_observations_premelt_stage`
- `validate_observations_timestamped_stage`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `from datetime import datetime, timedelta`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.premelt_stage_writer import ( build_observations_premelt_stage, validate_observations_premelt_stage,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.timestamp_stage_writer import ( ensure_simulation_timing_config_table, insert_simulation_timing_config, build_observations_timestamped_stage, validate`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `from datetime import datetime, timedelta` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.premelt_stage_writer import ( build_observations_premelt_stage, validate_observations_premelt_stage,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.timestamp_stage_writer import ( ensure_simulation_timing_config_table, insert_simulation_timing_config, build_observations_timestamped_stage, validate` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `CAPSTONE_SCHEMA`
- `getenv`
- `os`
- `pump_asset_001`
- `pump_synthetic_v1`
- `replace`
- `simulation_timing_config`
- `SYNTHETIC_ASSET_ID`
- `SYNTHETIC_DATASET_ID`
- `synthetic_observations_premelt_stage`
- `synthetic_observations_timestamped_stage`
- `synthetic_pump_stream`
- `synthetic_run_001`
- `SYNTHETIC_RUN_ID`

### Outputs

- `ASSET_ID`
- `CHUNK_SIZE`
- `DATASET_ID`
- `IF_EXISTS_FLAG`
- `NUMBER_OF_SENSORS`
- `PREMELT_DESTINATION_TABLE_NAME`
- `PREMELT_SOURCE_TABLE_NAME`
- `RANDOM_SEED`
- `RUN_ID`
- `SCHEMA`
- `SIMULATION_SAMPLING_INTERVAL_SECONDS`
- `SIMULATION_START_DATETIME`
- `SIMULATION_TIME_CONFIG_TABLE_NAME`
- `TIMESTAMPED_DESTINATION_TABLE_NAME`
- `TIMESTAMPED_SOURCE_TABLE_NAME`

### Key Operations

- `SCHEMA = os.getenv("CAPSTONE_SCHEMA", "synthetic_run_001")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_ID = os.getenv("SYNTHETIC_DATASET_ID", "pump_synthetic_v1")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RUN_ID = os.getenv("SYNTHETIC_RUN_ID", "synthetic_run_001")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ASSET_ID = os.getenv("SYNTHETIC_ASSET_ID", "pump_asset_001")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `IF_EXISTS_FLAG = str("replace")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RANDOM_SEED = int(42)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `NUMBER_OF_SENSORS = int(52)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CHUNK_SIZE = int(50000)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PREMELT_SOURCE_TABLE_NAME = str("synthetic_pump_stream")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PREMELT_DESTINATION_TABLE_NAME = str("synthetic_observations_premelt_stage")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SIMULATION_TIME_CONFIG_TABLE_NAME = str("simulation_timing_config")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SIMULATION_START_DATETIME = str("2026-04-16 00:00:00+00:00")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `getenv`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `SCHEMA = os.getenv("CAPSTONE_SCHEMA", "synthetic_run_001")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_ID = os.getenv("SYNTHETIC_DATASET_ID", "pump_synthetic_v1")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_ID = os.getenv("SYNTHETIC_RUN_ID", "synthetic_run_001")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ASSET_ID = os.getenv("SYNTHETIC_ASSET_ID", "pump_asset_001")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `IF_EXISTS_FLAG = str("replace")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RANDOM_SEED = int(42)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `NUMBER_OF_SENSORS = int(52)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CHUNK_SIZE = int(50000)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PREMELT_SOURCE_TABLE_NAME = str("synthetic_pump_stream")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PREMELT_DESTINATION_TABLE_NAME = str("synthetic_observations_premelt_stage")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SIMULATION_TIME_CONFIG_TABLE_NAME = str("simulation_timing_config")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SIMULATION_START_DATETIME = str("2026-04-16 00:00:00+00:00")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SIMULATION_SAMPLING_INTERVAL_SECONDS = float(60.0)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TIMESTAMPED_SOURCE_TABLE_NAME = str("synthetic_observations_premelt_stage")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TIMESTAMPED_DESTINATION_TABLE_NAME = str("synthetic_observations_timestamped_stage")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 03 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `get_engine_from_env`

### Outputs

- `engine`

### Key Operations

- `e`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `get_engine_from_env`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `e` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 04 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `d`
- `datetime`
- `H`
- `hours`
- `m`
- `M`
- `now`
- `strftime`
- `timedelta`
- `Y_`

### Outputs

- `adjusted_time`
- `current_datetime`
- `formatted_datetime`

### Key Operations

- `current_datetime = datetime.now()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `adjusted_time = current_datetime - timedelta(hours=4)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `formatted_datetime = adjusted_time.strftime("%m%d%Y_%H%M")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `now`
- `strftime`
- `timedelta`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `current_datetime = datetime.now()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `adjusted_time = current_datetime - timedelta(hours=4)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `formatted_datetime = adjusted_time.strftime("%m%d%Y_%H%M")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 05 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `at`
- `f`
- `formatted_datetime`
- `Observation`
- `Starting`
- `Step`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `p`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `p` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 06 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `ADD`
- `Added`
- `ALTER`
- `begin`
- `COLUMN`
- `columns`
- `conn`
- `DOUBLE`
- `dropped`
- `Emergency`
- `engine`
- `execute`
- `EXISTS`
- `expects`
- `f`
- `fix`
- `full`
- `generated`
- `missing`

### Outputs

- `missing_sensor_columns_for_premelt`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Emergency fix: add missing dropped sensor columns to source table`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Premelt expects the full 52-sensor schema.`: Documents the purpose or boundary of the surrounding notebook step.
- `# These dropped sensors were not modeled/generated, so we restore them as NULLs.`: Documents the purpose or boundary of the surrounding notebook step.
- `from sqlalchemy import text`: Imports a dependency or project helper used by later cells.
- `missing_sensor_columns_for_premelt = ["sensor_15", "sensor_50"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `with engine.begin() as conn: for sensor_col in missing_sensor_columns_for_premelt: conn.execute( text( f""" ALTER TABLE {SCHEMA}.{PREMELT_SOURCE_TABLE_NAME} ADD COLUMN IF NOT EXIST`: Controls validation, iteration, file handling, or error handling for this step.
- `print( "Added missing premelt source columns if needed:", missing_sensor_columns_for_premelt,`: Displays a notebook-facing result for inspection.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `begin`
- `execute`
- `text`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Emergency fix: add missing dropped sensor columns to source table` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Premelt expects the full 52-sensor schema.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# These dropped sensors were not modeled/generated, so we restore them as NULLs.` | Documents the purpose or boundary of the surrounding notebook step. |
| `from sqlalchemy import text` | Imports a dependency or project helper used by later cells. |
| `missing_sensor_columns_for_premelt = ["sensor_15", "sensor_50"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `with engine.begin() as conn: for sensor_col in missing_sensor_columns_for_premelt: conn.execute( text( f""" ALTER TABLE {SCHEMA}.{PREMELT_SOURCE_TABLE_NAME} ADD COLUMN IF NOT EXIST` | Controls validation, iteration, file handling, or error handling for this step. |
| `print( "Added missing premelt source columns if needed:", missing_sensor_columns_for_premelt,` | Displays a notebook-facing result for inspection. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 07 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_observations_premelt_stage`
- `Built`
- `IF_EXISTS_FLAG`
- `PREMELT_DESTINATION_TABLE_NAME`
- `PREMELT_SOURCE_TABLE_NAME`
- `table`

### Outputs

- `asset_id`
- `dataset_id`
- `engine`
- `if_exists`
- `premelt_table_name`
- `run_id`
- `schema`
- `source_table`
- `target_table`

### Key Operations

- `premelt_table_name = build_observations_premelt_stage( engine=engine, schema=SCHEMA, source_table=PREMELT_SOURCE_TABLE_NAME, target_table=PREMELT_DESTINATION_TABLE_NAME, dataset_id`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Built table:", premelt_table_name)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_observations_premelt_stage`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `premelt_table_name = build_observations_premelt_stage( engine=engine, schema=SCHEMA, source_table=PREMELT_SOURCE_TABLE_NAME, target_table=PREMELT_DESTINATION_TABLE_NAME, dataset_id` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Built table:", premelt_table_name)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 08 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `PREMELT_DESTINATION_TABLE_NAME`
- `validate_observations_premelt_stage`

### Outputs

- `engine`
- `schema`
- `table_name`
- `validation_dataframe`

### Key Operations

- `validation_dataframe = validate_observations_premelt_stage( engine=engine, schema=SCHEMA, table_name=PREMELT_DESTINATION_TABLE_NAME,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `validate_observations_premelt_stage`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `validation_dataframe = validate_observations_premelt_stage( engine=engine, schema=SCHEMA, table_name=PREMELT_DESTINATION_TABLE_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 09 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `asset_id`
- `batch_id`
- `BY`
- `capstone`
- `dataset_id`
- `engine`
- `generated_row_id`
- `global_cycle_id`
- `Inspection`
- `is_telemetry_event`
- `LIMIT`
- `meta_episode_id`
- `observation_index`
- `ORDER`
- `phase`
- `producer_send_attempt`
- `read_sql_dataframe`
- `row_in_batch`
- `run_id`
- `SELECT`

### Outputs

- `inspection_dataframe`

### Key Operations

- `# Inspection`: Documents the purpose or boundary of the surrounding notebook step.
- `inspection_dataframe = read_sql_dataframe( engine, """ SELECT dataset_id, run_id, asset_id, generated_row_id, observation_index, batch_id, row_in_batch, global_cycle_id, stream_sta`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(inspection_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Inspection` | Documents the purpose or boundary of the surrounding notebook step. |
| `inspection_dataframe = read_sql_dataframe( engine, """ SELECT dataset_id, run_id, asset_id, generated_row_id, observation_index, batch_id, row_in_batch, global_cycle_id, stream_sta` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(inspection_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 10 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dispose`
- `Dispose`
- `Engine`
- `engine`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `# Dispose Engine`: Documents the purpose or boundary of the surrounding notebook step.
- `engine.dispose()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `dispose`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Dispose Engine` | Documents the purpose or boundary of the surrounding notebook step. |
| `engine.dispose()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 11 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `d`
- `datetime`
- `H`
- `hours`
- `m`
- `M`
- `now`
- `strftime`
- `timedelta`
- `Y_`

### Outputs

- `adjusted_time`
- `current_datetime`
- `formatted_datetime`

### Key Operations

- `current_datetime = datetime.now()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `adjusted_time = current_datetime - timedelta(hours=4)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `formatted_datetime = adjusted_time.strftime("%m%d%Y_%H%M")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `now`
- `strftime`
- `timedelta`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `current_datetime = datetime.now()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `adjusted_time = current_datetime - timedelta(hours=4)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `formatted_datetime = adjusted_time.strftime("%m%d%Y_%H%M")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 12 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `at`
- `End`
- `f`
- `formatted_datetime`
- `Observation`
- `of`
- `Step`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `p`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `p` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 13 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `d`
- `datetime`
- `H`
- `hours`
- `m`
- `M`
- `now`
- `strftime`
- `timedelta`
- `Y_`

### Outputs

- `adjusted_time`
- `current_datetime`
- `formatted_datetime`

### Key Operations

- `current_datetime = datetime.now()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `adjusted_time = current_datetime - timedelta(hours=4)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `formatted_datetime = adjusted_time.strftime("%m%d%Y_%H%M")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `now`
- `strftime`
- `timedelta`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `current_datetime = datetime.now()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `adjusted_time = current_datetime - timedelta(hours=4)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `formatted_datetime = adjusted_time.strftime("%m%d%Y_%H%M")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 14 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `at`
- `f`
- `formatted_datetime`
- `of`
- `Start`
- `Step`
- `Timestamping`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `p`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `p` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 15 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `get_engine_from_env`

### Outputs

- `engine`

### Key Operations

- `e`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `get_engine_from_env`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `e` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 16 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `ensure_simulation_timing_config_table`
- `SIMULATION_TIME_CONFIG_TABLE_NAME`

### Outputs

- `engine`
- `schema`
- `table_name`

### Key Operations

- `ensure_simulation_timing_config_table( engine=engine, schema=SCHEMA, table_name=SIMULATION_TIME_CONFIG_TABLE_NAME,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `ensure_simulation_timing_config_table`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `ensure_simulation_timing_config_table( engine=engine, schema=SCHEMA, table_name=SIMULATION_TIME_CONFIG_TABLE_NAME,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 17 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `config`
- `insert_simulation_timing_config`
- `ready`
- `SIMULATION_SAMPLING_INTERVAL_SECONDS`
- `SIMULATION_TIME_CONFIG_TABLE_NAME`
- `Timing`

### Outputs

- `dataset_id`
- `deactivate_existing_for_run`
- `engine`
- `run_id`
- `sampling_interval_seconds`
- `schema`
- `set_active`
- `simulation_start_datetime`
- `table_name`

### Key Operations

- `insert_simulation_timing_config( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, simulation_start_datetime=SIMULATION_START_DATETIME, sampling_interval_seconds=SIMULATION_SAMP`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Timing config ready.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `insert_simulation_timing_config`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `insert_simulation_timing_config( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, simulation_start_datetime=SIMULATION_START_DATETIME, sampling_interval_seconds=SIMULATION_SAMP` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Timing config ready.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 18 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_observations_timestamped_stage`
- `Built`
- `IF_EXISTS_FLAG`
- `SIMULATION_TIME_CONFIG_TABLE_NAME`
- `table`
- `TIMESTAMPED_DESTINATION_TABLE_NAME`
- `TIMESTAMPED_SOURCE_TABLE_NAME`

### Outputs

- `chunk_size`
- `dataset_id`
- `engine`
- `if_exists`
- `run_id`
- `schema`
- `source_table`
- `target_table`
- `timestamped_table_name`
- `timing_config_table`

### Key Operations

- `timestamped_table_name = build_observations_timestamped_stage( engine=engine, schema=SCHEMA, source_table=TIMESTAMPED_SOURCE_TABLE_NAME, target_table=TIMESTAMPED_DESTINATION_TABLE_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Built table:", timestamped_table_name)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_observations_timestamped_stage`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `timestamped_table_name = build_observations_timestamped_stage( engine=engine, schema=SCHEMA, source_table=TIMESTAMPED_SOURCE_TABLE_NAME, target_table=TIMESTAMPED_DESTINATION_TABLE_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Built table:", timestamped_table_name)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 19 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `TIMESTAMPED_DESTINATION_TABLE_NAME`
- `validate_observations_timestamped_stage`

### Outputs

- `engine`
- `schema`
- `table_name`
- `validation_dataframe`

### Key Operations

- `validation_dataframe = validate_observations_timestamped_stage( engine=engine, schema=SCHEMA, table_name=TIMESTAMPED_DESTINATION_TABLE_NAME,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `validate_observations_timestamped_stage`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `validation_dataframe = validate_observations_timestamped_stage( engine=engine, schema=SCHEMA, table_name=TIMESTAMPED_DESTINATION_TABLE_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 20 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `capstone`
- `engine`
- `generated_row_id`
- `LIMIT`
- `meta_episode_id`
- `observation_index`
- `observation_timestamp`
- `ORDER`
- `phase`
- `read_sql_dataframe`
- `SELECT`
- `sensor_00`
- `sensor_01`
- `stream_state`
- `synthetic_observations_timestamped_stage`

### Outputs

- `sample_dataframe`

### Key Operations

- `sample_dataframe = read_sql_dataframe( engine, """ SELECT observation_index, observation_timestamp, generated_row_id, stream_state, phase, meta_episode_id, sensor_00, sensor_01 FRO`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(sample_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `sample_dataframe = read_sql_dataframe( engine, """ SELECT observation_index, observation_timestamp, generated_row_id, stream_state, phase, meta_episode_id, sensor_00, sensor_01 FRO` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(sample_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 21 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `capstone`
- `engine`
- `LIMIT`
- `meta_episode_id`
- `observation_index`
- `observation_timestamp`
- `ORDER`
- `phase`
- `read_sql_dataframe`
- `SELECT`
- `stream_state`
- `synthetic_observations_timestamped_stage`

### Outputs

- `timestamp_check_dataframe`

### Key Operations

- `timestamp_check_dataframe = read_sql_dataframe( engine, """ SELECT observation_index, observation_timestamp, stream_state, phase, meta_episode_id FROM capstone.synthetic_observatio`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(timestamp_check_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `timestamp_check_dataframe = read_sql_dataframe( engine, """ SELECT observation_index, observation_timestamp, stream_state, phase, meta_episode_id FROM capstone.synthetic_observatio` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(timestamp_check_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 22 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dispose`
- `Dispose`
- `Engine`
- `engine`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `# Dispose Engine`: Documents the purpose or boundary of the surrounding notebook step.
- `engine.dispose()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `dispose`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Dispose Engine` | Documents the purpose or boundary of the surrounding notebook step. |
| `engine.dispose()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 23 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `d`
- `datetime`
- `H`
- `hours`
- `m`
- `M`
- `now`
- `strftime`
- `timedelta`
- `Y_`

### Outputs

- `adjusted_time`
- `current_datetime`
- `formatted_datetime`

### Key Operations

- `current_datetime = datetime.now()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `adjusted_time = current_datetime - timedelta(hours=4)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `formatted_datetime = adjusted_time.strftime("%m%d%Y_%H%M")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `now`
- `strftime`
- `timedelta`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `current_datetime = datetime.now()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `adjusted_time = current_datetime - timedelta(hours=4)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `formatted_datetime = adjusted_time.strftime("%m%d%Y_%H%M")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 24 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `at`
- `End`
- `f`
- `formatted_datetime`
- `of`
- `Step`
- `Timestamping`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `p`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `p` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 25 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `another`
- `com`
- `https`
- `ipynb`
- `jupyter`
- `notebook`
- `questions`
- `Reference`
- `run`
- `running`
- `stackoverflow`
- `synthetic_pipeline_testing_export_csv`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `# Reference: https://stackoverflow.com/questions/49817409/running-a-jupyter-notebook-from-another-notebook`: Documents the purpose or boundary of the surrounding notebook step.
- `%run ./synthetic_pipeline_testing_export_csv.ipynb`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Reference: https://stackoverflow.com/questions/49817409/running-a-jupyter-notebook-from-another-notebook` | Documents the purpose or boundary of the surrounding notebook step. |
| `%run ./synthetic_pipeline_testing_export_csv.ipynb` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

