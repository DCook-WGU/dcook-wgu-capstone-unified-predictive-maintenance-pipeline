# Notebook Code Reference: synthetic_02_build_premilt_observations_stage

Notebook path:

`notebooks/synthetic/synthetic_02_build_premilt_observations_stage.ipynb`

## Notebook Purpose

This notebook documents and runs a focused support step in the synthetic pump telemetry workflow.

Notebook stage:

`Synthetic support`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_observations_premelt_stage`
- `core`
- `database`
- `env_helpers`
- `env_int`
- `env_str`
- `get_engine_from_env`
- `os`
- `pipeline`
- `postgres`
- `premelt_stage_writer`
- `read_sql_dataframe`
- `synthetic`
- `utils`
- `validate_observations_premelt_stage`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.premelt_stage_writer import ( build_observations_premelt_stage, validate_observations_premelt_stage,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.core.env_helpers import ( env_int, env_str,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.premelt_stage_writer import ( build_observations_premelt_stage, validate_observations_premelt_stage,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.env_helpers import ( env_int, env_str,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `asset`
- `capstone`
- `CAPSTONE_SCHEMA`
- `config`
- `dataset`
- `env_int`
- `env_str`
- `id`
- `Premelt`
- `pump_asset_001`
- `pump_synthetic_v1`
- `replace`
- `run`
- `source`
- `SYNTHETIC_ASSET_ID`
- `SYNTHETIC_DATASET_ID`
- `synthetic_observations_premelt_stage`
- `SYNTHETIC_PREMELT_OBSERVATIONS_TABLE`
- `SYNTHETIC_PREMELT_SOURCE_ROW_CHUNK_SIZE`
- `synthetic_pump_stream`

### Outputs

- `aliases`
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

### Key Operations

- `SCHEMA = env_str("CAPSTONE_SCHEMA", "capstone")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_ID = env_str( "SYNTHETIC_DATASET_ID", "pump_synthetic_v1", aliases=("DATASET_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `RUN_ID = env_str( "SYNTHETIC_RUN_ID", "synthetic_run_001", aliases=("RUN_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ASSET_ID = env_str( "SYNTHETIC_ASSET_ID", "pump_asset_001",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `IF_EXISTS_FLAG = "replace"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RANDOM_SEED = env_int("SYNTHETIC_RANDOM_SEED", 42)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `NUMBER_OF_SENSORS = env_int("SYNTHETIC_SENSOR_COUNT", 52)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CHUNK_SIZE = env_int("SYNTHETIC_PREMELT_SOURCE_ROW_CHUNK_SIZE", 10000)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PREMELT_SOURCE_TABLE_NAME = env_str( "SYNTHETIC_SOURCE_STREAM_TABLE", "synthetic_pump_stream",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
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
| `ASSET_ID = env_str( "SYNTHETIC_ASSET_ID", "pump_asset_001",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `IF_EXISTS_FLAG = "replace"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RANDOM_SEED = env_int("SYNTHETIC_RANDOM_SEED", 42)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `NUMBER_OF_SENSORS = env_int("SYNTHETIC_SENSOR_COUNT", 52)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CHUNK_SIZE = env_int("SYNTHETIC_PREMELT_SOURCE_ROW_CHUNK_SIZE", 10000)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PREMELT_SOURCE_TABLE_NAME = env_str( "SYNTHETIC_SOURCE_STREAM_TABLE", "synthetic_pump_stream",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PREMELT_DESTINATION_TABLE_NAME = env_str( "SYNTHETIC_PREMELT_OBSERVATIONS_TABLE", "synthetic_observations_premelt_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Premelt config")` | Displays a notebook-facing result for inspection. |
| `print("schema:", SCHEMA)` | Displays a notebook-facing result for inspection. |
| `print("dataset id:", DATASET_ID)` | Displays a notebook-facing result for inspection. |
| `print("run id:", RUN_ID)` | Displays a notebook-facing result for inspection. |
| `print("asset id:", ASSET_ID)` | Displays a notebook-facing result for inspection. |
| `print("source table:", PREMELT_SOURCE_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("target table:", PREMELT_DESTINATION_TABLE_NAME)` | Displays a notebook-facing result for inspection. |

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

## Code Cell 05 — Code Reference

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

## Code Cell 06 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `asset_id`
- `batch_id`
- `BY`
- `dataset_id`
- `engine`
- `f`
- `generated_row_id`
- `global_cycle_id`
- `is_telemetry_event`
- `LIMIT`
- `meta_episode_id`
- `observation_index`
- `ORDER`
- `phase`
- `PREMELT_DESTINATION_TABLE_NAME`
- `producer_send_attempt`
- `read_sql_dataframe`
- `row_in_batch`
- `run_id`
- `SCHEMA`

### Outputs

- `inspection_dataframe`

### Key Operations

- `inspection_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, generated_row_id, observation_index, batch_id, row_in_batch, global_cycle_id, stream_st`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(inspection_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `inspection_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, generated_row_id, observation_index, batch_id, row_in_batch, global_cycle_id, stream_st` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(inspection_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 07 — Code Reference

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

