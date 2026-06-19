# Notebook Code Reference: synthetic_04_build_sensor_messages_stage

Notebook path:

`notebooks/synthetic/synthetic_04_build_sensor_messages_stage.ipynb`

## Notebook Purpose

This notebook documents and runs a focused support step in the synthetic pump telemetry workflow.

Notebook stage:

`Synthetic support`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08, Code Cell 09, Code Cell 10, Code Cell 11, Code Cell 12, Code Cell 13 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_sensor_messages_stage`
- `build_sensor_messages_stage_sql_native`
- `chunk_stage_util`
- `core`
- `database`
- `env_helpers`
- `env_int`
- `env_str`
- `get_engine_from_env`
- `Imports`
- `imports`
- `inspect`
- `melt_stage_writer`
- `nbuild_sensor_messages_stage`
- `nbuild_sensor_messages_stage_sql_native`
- `os`
- `passed`
- `pipeline`
- `postgres`
- `process_postgres_table_in_chunks`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.database.chunk_stage_util import process_postgres_table_in_chunks`: Imports a dependency or project helper used by later cells.
- `from utils.synthetic.pipeline.melt_stage_writer import ( build_sensor_messages_stage, validate_sensor_messages_stage, build_sensor_messages_stage_sql_native,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.core.env_helpers import ( env_int, env_str,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Stage 4 imports passed.")`: Displays a notebook-facing result for inspection.
- `import inspect`: Imports a dependency or project helper used by later cells.
- `print("Imports passed.")`: Displays a notebook-facing result for inspection.
- `print("process_postgres_table_in_chunks:")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `import`
- `signature`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.chunk_stage_util import process_postgres_table_in_chunks` | Imports a dependency or project helper used by later cells. |
| `from utils.synthetic.pipeline.melt_stage_writer import ( build_sensor_messages_stage, validate_sensor_messages_stage, build_sensor_messages_stage_sql_native,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.env_helpers import ( env_int, env_str,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Stage 4 imports passed.")` | Displays a notebook-facing result for inspection. |
| `import inspect` | Imports a dependency or project helper used by later cells. |
| `print("Imports passed.")` | Displays a notebook-facing result for inspection. |
| `print("process_postgres_table_in_chunks:")` | Displays a notebook-facing result for inspection. |
| `print(inspect.signature(process_postgres_table_in_chunks))` | Displays a notebook-facing result for inspection. |
| `print("\nbuild_sensor_messages_stage:")` | Displays a notebook-facing result for inspection. |
| `print(inspect.signature(build_sensor_messages_stage))` | Displays a notebook-facing result for inspection. |
| `print("\nbuild_sensor_messages_stage_sql_native:")` | Displays a notebook-facing result for inspection. |
| `print(inspect.signature(build_sensor_messages_stage_sql_native))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `batch`
- `capstone`
- `CAPSTONE_SCHEMA`
- `chunk`
- `config`
- `env_int`
- `env_str`
- `melt`
- `message`
- `pump_synthetic_v1`
- `replace`
- `row`
- `size`
- `source`
- `Synthetic`
- `SYNTHETIC_DATASET_ID`
- `SYNTHETIC_MELT_SOURCE_ROW_CHUNK_SIZE`
- `synthetic_observations_timestamped_stage`
- `SYNTHETIC_RANDOM_SEED`
- `synthetic_run_001`

### Outputs

- `aliases`
- `CHUNK_SIZE`
- `DATASET_ID`
- `IF_EXISTS_FLAG`
- `MELT_DESTINATION_TABLE_NAME`
- `MELT_SOURCE_TABLE_NAME`
- `MESSAGE_BATCH_SIZE`
- `NUMBER_OF_SENSORS`
- `OBSERVATION_BATCH_SIZE`
- `RANDOM_SEED`
- `RUN_ID`
- `SCHEMA`

### Key Operations

- `SCHEMA = env_str("CAPSTONE_SCHEMA", "capstone")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_ID = env_str( "SYNTHETIC_DATASET_ID", "pump_synthetic_v1", aliases=("DATASET_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `RUN_ID = env_str( "SYNTHETIC_RUN_ID", "synthetic_run_001", aliases=("RUN_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `IF_EXISTS_FLAG = "replace"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RANDOM_SEED = env_int("SYNTHETIC_RANDOM_SEED", 42)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `NUMBER_OF_SENSORS = env_int("SYNTHETIC_SENSOR_COUNT", 52)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CHUNK_SIZE = env_int("SYNTHETIC_MELT_SOURCE_ROW_CHUNK_SIZE", 25000)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `MELT_SOURCE_TABLE_NAME = env_str( "SYNTHETIC_TIMESTAMPED_OBSERVATIONS_TABLE", "synthetic_observations_timestamped_stage",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `MELT_DESTINATION_TABLE_NAME = env_str( "SYNTHETIC_SENSOR_MESSAGES_TABLE", "synthetic_sensor_messages_stage",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

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
| `IF_EXISTS_FLAG = "replace"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RANDOM_SEED = env_int("SYNTHETIC_RANDOM_SEED", 42)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `NUMBER_OF_SENSORS = env_int("SYNTHETIC_SENSOR_COUNT", 52)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CHUNK_SIZE = env_int("SYNTHETIC_MELT_SOURCE_ROW_CHUNK_SIZE", 25000)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MELT_SOURCE_TABLE_NAME = env_str( "SYNTHETIC_TIMESTAMPED_OBSERVATIONS_TABLE", "synthetic_observations_timestamped_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `MELT_DESTINATION_TABLE_NAME = env_str( "SYNTHETIC_SENSOR_MESSAGES_TABLE", "synthetic_sensor_messages_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `OBSERVATION_BATCH_SIZE = env_int("OBSERVATION_BATCH_SIZE", 500)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MESSAGE_BATCH_SIZE = OBSERVATION_BATCH_SIZE * NUMBER_OF_SENSORS` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Synthetic melt config")` | Displays a notebook-facing result for inspection. |
| `print("schema:", SCHEMA)` | Displays a notebook-facing result for inspection. |
| `print("source table:", MELT_SOURCE_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("target table:", MELT_DESTINATION_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("source row chunk size:", CHUNK_SIZE)` | Displays a notebook-facing result for inspection. |
| `print("message batch size:", MESSAGE_BATCH_SIZE)` | Displays a notebook-facing result for inspection. |

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

- `build_sensor_messages_stage_sql_native`
- `Built`
- `MELT_DESTINATION_TABLE_NAME`
- `MELT_SOURCE_TABLE_NAME`
- `melted`
- `native`
- `NUMBER_OF_SENSORS`
- `SQL`
- `table`

### Outputs

- `enable_memory_logging`
- `engine`
- `melt_table_name`
- `n_sensors`
- `schema`
- `source_table`
- `target_table`

### Key Operations

- `melt_table_name = build_sensor_messages_stage_sql_native( engine=engine, schema=SCHEMA, source_table=MELT_SOURCE_TABLE_NAME, target_table=MELT_DESTINATION_TABLE_NAME, n_sensors=NUM`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Built SQL-native melted table:", melt_table_name)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_sensor_messages_stage_sql_native`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `melt_table_name = build_sensor_messages_stage_sql_native( engine=engine, schema=SCHEMA, source_table=MELT_SOURCE_TABLE_NAME, target_table=MELT_DESTINATION_TABLE_NAME, n_sensors=NUM` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Built SQL-native melted table:", melt_table_name)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 05 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_sensor_messages_stage`
- `IF_EXISTS_FLAG`
- `MELT_DESTINATION_TABLE_NAME`
- `MELT_SOURCE_TABLE_NAME`
- `NUMBER_OF_SENSORS`

### Outputs

- `chunk_size`
- `enable_memory_logging`
- `engine`
- `if_exists`
- `melt_table_name`
- `n_sensors`
- `random_seed`
- `schema`
- `source_table`
- `target_table`

### Key Operations

- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `melt_table_name = build_sensor_messages_stage( engine=engine, schema=SCHEMA, source_table=MELT_SOURCE_TABLE_NAME, target_table=MELT_DESTINATION_TABLE_NAME, if_exists=IF_EXISTS_FLAG`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `build_sensor_messages_stage`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `melt_table_name = build_sensor_messages_stage( engine=engine, schema=SCHEMA, source_table=MELT_SOURCE_TABLE_NAME, target_table=MELT_DESTINATION_TABLE_NAME, if_exists=IF_EXISTS_FLAG` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 06 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Built`
- `melt_table_name`
- `table`

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

## Code Cell 07 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `MELT_DESTINATION_TABLE_NAME`
- `validate_sensor_messages_stage`

### Outputs

- `engine`
- `schema`
- `table_name`
- `validation_dataframe`

### Key Operations

- `validation_dataframe = validate_sensor_messages_stage( engine=engine, schema=SCHEMA, table_name=MELT_DESTINATION_TABLE_NAME,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `validate_sensor_messages_stage`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `validation_dataframe = validate_sensor_messages_stage( engine=engine, schema=SCHEMA, table_name=MELT_DESTINATION_TABLE_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 08 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `COUNT`
- `CROSS`
- `DISTINCT`
- `distinct_observation_count`
- `distinct_sensor_index_count`
- `distinct_sensor_name_count`
- `engine`
- `expected_message_row_count`
- `f`
- `JOIN`
- `MAX`
- `max_message_sequence_index`
- `max_sensor_index`
- `MELT_DESTINATION_TABLE_NAME`
- `MELT_SOURCE_TABLE_NAME`
- `message_counts`
- `message_row_count`
- `message_row_delta`
- `message_sequence_index`
- `MIN`

### Outputs

- `stage_04_validation_dataframe`

### Key Operations

- `stage_04_validation_dataframe = read_sql_dataframe( engine, f""" WITH source_counts AS ( SELECT COUNT(*) AS source_observation_count FROM "{SCHEMA}"."{MELT_SOURCE_TABLE_NAME}" ), m`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_04_validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `AS`
- `COUNT`
- `display`
- `MAX`
- `MIN`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_04_validation_dataframe = read_sql_dataframe( engine, f""" WITH source_counts AS ( SELECT COUNT(*) AS source_observation_count FROM "{SCHEMA}"."{MELT_SOURCE_TABLE_NAME}" ), m` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_04_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 09 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `COUNT`
- `DISTINCT`
- `distinct_sensor_index_count`
- `distinct_sensor_name_count`
- `engine`
- `f`
- `GROUP`
- `HAVING`
- `LIMIT`
- `MELT_DESTINATION_TABLE_NAME`
- `message_count`
- `NUMBER_OF_SENSORS`
- `observation_index`
- `ORDER`
- `read_sql_dataframe`
- `SCHEMA`
- `SELECT`
- `sensor_index`
- `sensor_name`

### Outputs

- `incomplete_observations_dataframe`

### Key Operations

- `incomplete_observations_dataframe = read_sql_dataframe( engine, f""" SELECT observation_index, COUNT(*) AS message_count, COUNT(DISTINCT sensor_index) AS distinct_sensor_index_coun`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(incomplete_observations_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `incomplete_observations_dataframe = read_sql_dataframe( engine, f""" SELECT observation_index, COUNT(*) AS message_count, COUNT(DISTINCT sensor_index) AS distinct_sensor_index_coun` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(incomplete_observations_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 10 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `engine`
- `f`
- `generated_row_id`
- `LIMIT`
- `MELT_DESTINATION_TABLE_NAME`
- `message_sequence_index`
- `meta_episode_id`
- `observation_index`
- `observation_timestamp`
- `ORDER`
- `phase`
- `read_sql_dataframe`
- `SCHEMA`
- `SELECT`
- `sensor_index`
- `sensor_name`
- `sensor_value`
- `stream_state`

### Outputs

- `sample_sensor_messages_dataframe`

### Key Operations

- `sample_sensor_messages_dataframe = read_sql_dataframe( engine, f""" SELECT observation_index, generated_row_id, observation_timestamp, sensor_name, sensor_index, sensor_value, mess`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(sample_sensor_messages_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `sample_sensor_messages_dataframe = read_sql_dataframe( engine, f""" SELECT observation_index, generated_row_id, observation_timestamp, sensor_name, sensor_index, sensor_value, mess` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(sample_sensor_messages_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 11 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `engine`
- `f`
- `generated_row_id`
- `LIMIT`
- `MELT_DESTINATION_TABLE_NAME`
- `message_sequence_index`
- `meta_episode_id`
- `observation_index`
- `observation_timestamp`
- `ORDER`
- `phase`
- `read_sql_dataframe`
- `SCHEMA`
- `SELECT`
- `sensor_index`
- `sensor_name`
- `sensor_value`
- `stream_state`

### Outputs

- `sample_dataframe`

### Key Operations

- `sample_dataframe = read_sql_dataframe( engine, f""" SELECT observation_index, observation_timestamp, generated_row_id, sensor_name, sensor_index, sensor_value, message_sequence_ind`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(sample_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `sample_dataframe = read_sql_dataframe( engine, f""" SELECT observation_index, observation_timestamp, generated_row_id, sensor_name, sensor_index, sensor_value, message_sequence_ind` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(sample_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 12 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `COUNT`
- `DISTINCT`
- `distinct_msg_seq_count`
- `distinct_observation_timestamp_count`
- `distinct_sensor_count`
- `engine`
- `f`
- `GROUP`
- `LIMIT`
- `MAX`
- `max_msg_seq`
- `MELT_DESTINATION_TABLE_NAME`
- `message_count`
- `message_sequence_index`
- `MIN`
- `min_msg_seq`
- `observation_index`
- `observation_timestamp`
- `ORDER`

### Outputs

- `sequence_check_dataframe`

### Key Operations

- `sequence_check_dataframe = read_sql_dataframe( engine, f""" SELECT observation_index, COUNT(*) AS message_count, COUNT(DISTINCT sensor_index) AS distinct_sensor_count, COUNT(DISTIN`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(sequence_check_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `MAX`
- `MIN`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `sequence_check_dataframe = read_sql_dataframe( engine, f""" SELECT observation_index, COUNT(*) AS message_count, COUNT(DISTINCT sensor_index) AS distinct_sensor_count, COUNT(DISTIN` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(sequence_check_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 13 — Code Reference

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

