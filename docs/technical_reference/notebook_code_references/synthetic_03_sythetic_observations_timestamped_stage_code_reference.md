# Notebook Code Reference: synthetic_03_sythetic_observations_timestamped_stage

Notebook path:

`notebooks/synthetic/synthetic_03_sythetic_observations_timestamped_stage.ipynb`

## Notebook Purpose

This notebook documents and runs a focused support step in the synthetic pump telemetry workflow.

Notebook stage:

`Synthetic support`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08, Code Cell 09, Code Cell 10 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_observations_timestamped_stage`
- `core`
- `database`
- `ensure_simulation_timing_config_table`
- `env_float`
- `env_helpers`
- `env_int`
- `env_str`
- `get_engine_from_env`
- `insert_simulation_timing_config`
- `os`
- `pipeline`
- `postgres`
- `read_sql_dataframe`
- `synthetic`
- `timestamp_stage_writer`
- `utils`
- `validate_observations_timestamped_stage`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `from utils.core.env_helpers import ( env_float, env_int, env_str,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.timestamp_stage_writer import ( ensure_simulation_timing_config_table, insert_simulation_timing_config, build_observations_timestamped_stage, validate`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `from utils.core.env_helpers import ( env_float, env_int, env_str,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
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

- `capstone`
- `CAPSTONE_SCHEMA`
- `config`
- `dataset`
- `env_float`
- `env_int`
- `env_str`
- `id`
- `interval`
- `pump_synthetic_v1`
- `replace`
- `run`
- `sampling`
- `seconds`
- `simulation`
- `simulation_timing_config`
- `source`
- `start`
- `SYNTHETIC_DATASET_ID`
- `synthetic_observations_premelt_stage`

### Outputs

- `aliases`
- `CHUNK_SIZE`
- `DATASET_ID`
- `IF_EXISTS_FLAG`
- `RUN_ID`
- `SCHEMA`
- `SIMULATION_SAMPLING_INTERVAL_SECONDS`
- `SIMULATION_START_DATETIME`
- `SIMULATION_TIME_CONFIG_TABLE_NAME`
- `TIMESTAMPED_DESTINATION_TABLE_NAME`
- `TIMESTAMPED_SOURCE_TABLE_NAME`

### Key Operations

- `SCHEMA = env_str("CAPSTONE_SCHEMA", "capstone")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_ID = env_str( "SYNTHETIC_DATASET_ID", "pump_synthetic_v1", aliases=("DATASET_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `RUN_ID = env_str( "SYNTHETIC_RUN_ID", "synthetic_run_001", aliases=("RUN_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `IF_EXISTS_FLAG = "replace"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CHUNK_SIZE = env_int("SYNTHETIC_TIMESTAMP_SOURCE_ROW_CHUNK_SIZE", 250000)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SIMULATION_TIME_CONFIG_TABLE_NAME = env_str( "SYNTHETIC_SIMULATION_TIMING_CONFIG_TABLE", "simulation_timing_config",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `SIMULATION_START_DATETIME = env_str( "SYNTHETIC_SIMULATION_START_DATETIME", "2026-04-06 12:30:00+00:00",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `SIMULATION_SAMPLING_INTERVAL_SECONDS = env_float( "SYNTHETIC_SIMULATION_SAMPLING_INTERVAL_SECONDS", 60.0,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `env_float`
- `env_int`
- `env_str`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `SCHEMA = env_str("CAPSTONE_SCHEMA", "capstone")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_ID = env_str( "SYNTHETIC_DATASET_ID", "pump_synthetic_v1", aliases=("DATASET_ID",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `RUN_ID = env_str( "SYNTHETIC_RUN_ID", "synthetic_run_001", aliases=("RUN_ID",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `IF_EXISTS_FLAG = "replace"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CHUNK_SIZE = env_int("SYNTHETIC_TIMESTAMP_SOURCE_ROW_CHUNK_SIZE", 250000)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SIMULATION_TIME_CONFIG_TABLE_NAME = env_str( "SYNTHETIC_SIMULATION_TIMING_CONFIG_TABLE", "simulation_timing_config",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SIMULATION_START_DATETIME = env_str( "SYNTHETIC_SIMULATION_START_DATETIME", "2026-04-06 12:30:00+00:00",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SIMULATION_SAMPLING_INTERVAL_SECONDS = env_float( "SYNTHETIC_SIMULATION_SAMPLING_INTERVAL_SECONDS", 60.0,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `TIMESTAMPED_SOURCE_TABLE_NAME = env_str( "SYNTHETIC_PREMELT_OBSERVATIONS_TABLE", "synthetic_observations_premelt_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `TIMESTAMPED_DESTINATION_TABLE_NAME = env_str( "SYNTHETIC_TIMESTAMPED_OBSERVATIONS_TABLE", "synthetic_observations_timestamped_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Timestamp config")` | Displays a notebook-facing result for inspection. |
| `print("schema:", SCHEMA)` | Displays a notebook-facing result for inspection. |
| `print("dataset id:", DATASET_ID)` | Displays a notebook-facing result for inspection. |
| `print("run id:", RUN_ID)` | Displays a notebook-facing result for inspection. |
| `print("source table:", TIMESTAMPED_SOURCE_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("target table:", TIMESTAMPED_DESTINATION_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("timing config table:", SIMULATION_TIME_CONFIG_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("simulation start:", SIMULATION_START_DATETIME)` | Displays a notebook-facing result for inspection. |
| `print("sampling interval seconds:", SIMULATION_SAMPLING_INTERVAL_SECONDS)` | Displays a notebook-facing result for inspection. |

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

## Code Cell 05 — Code Reference

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

## Code Cell 06 — Code Reference

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

## Code Cell 07 — Code Reference

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

## Code Cell 08 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `engine`
- `f`
- `generated_row_id`
- `LIMIT`
- `meta_episode_id`
- `observation_index`
- `observation_timestamp`
- `ORDER`
- `phase`
- `read_sql_dataframe`
- `SCHEMA`
- `SELECT`
- `sensor_00`
- `sensor_01`
- `stream_state`
- `TIMESTAMPED_DESTINATION_TABLE_NAME`

### Outputs

- `sample_dataframe`

### Key Operations

- `sample_dataframe = read_sql_dataframe( engine, f""" SELECT observation_index, observation_timestamp, generated_row_id, stream_state, phase, meta_episode_id, sensor_00, sensor_01 FR`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(sample_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `sample_dataframe = read_sql_dataframe( engine, f""" SELECT observation_index, observation_timestamp, generated_row_id, stream_state, phase, meta_episode_id, sensor_00, sensor_01 FR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(sample_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 09 — Code Reference

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

