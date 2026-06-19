# Notebook Code Reference: synthetic_11a_build_final_aligned_observations_stage

Notebook path:

`notebooks/synthetic/synthetic_11a_build_final_aligned_observations_stage.ipynb`

## Notebook Purpose

This notebook documents and runs a focused support step in the synthetic pump telemetry workflow.

Notebook stage:

`Synthetic support`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08 |
| Sample Output | Code Cell 09, Code Cell 10, Code Cell 11, Code Cell 12 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Aligned`
- `aligned`
- `Build`
- `final`
- `Final`
- `full`
- `lineage`
- `observation`
- `Observations`
- `Purpose`
- `rich`
- `Source`
- `Stage`
- `synthetic_observations_premelt_stage`
- `synthetic_sensor_observations_final_aligned_stage`
- `synthetic_sensor_observations_rebuilt_stage`
- `table`
- `Target`
- `the`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Stage 11A — Final Aligned Observations Stage`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Purpose:`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build the full lineage-rich final aligned observation table.`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# Source:`: Documents the purpose or boundary of the surrounding notebook step.
- `# - synthetic_observations_premelt_stage`: Documents the purpose or boundary of the surrounding notebook step.
- `# - synthetic_sensor_observations_rebuilt_stage`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# Target:`: Documents the purpose or boundary of the surrounding notebook step.
- `# - synthetic_sensor_observations_final_aligned_stage`: Documents the purpose or boundary of the surrounding notebook step.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Stage 11A — Final Aligned Observations Stage` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Purpose:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build the full lineage-rich final aligned observation table.` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Source:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - synthetic_observations_premelt_stage` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - synthetic_sensor_observations_rebuilt_stage` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Target:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - synthetic_sensor_observations_final_aligned_stage` | Documents the purpose or boundary of the surrounding notebook step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 02 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_final_aligned_observations_stage`
- `build_synthetic_final_aligned_output_stage`
- `core`
- `database`
- `env_bool`
- `env_helpers`
- `env_int`
- `env_optional_int`
- `env_str`
- `execute_sql`
- `final_aligned_incremental`
- `final_aligned_observation_writer`
- `final_aligned_output_writer`
- `get_engine_from_env`
- `max_colwidth`
- `os`
- `pandas`
- `pipeline`
- `postgres`
- `read_sql_dataframe`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe, execute_sql,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import execute_sql, read_sql_dataframe`: Imports a dependency or project helper used by later cells.
- `from utils.synthetic.pipeline.final_aligned_observation_writer import ( build_final_aligned_observations_stage,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.final_aligned_output_writer import build_synthetic_final_aligned_output_stage`: Imports a dependency or project helper used by later cells.
- `from utils.synthetic.pipeline.final_aligned_incremental import run_final_align_loop`: Imports a dependency or project helper used by later cells.
- `from utils.core.env_helpers import ( env_bool, env_int, env_optional_int, env_str,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `import`
- `set_option`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe, execute_sql,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import execute_sql, read_sql_dataframe` | Imports a dependency or project helper used by later cells. |
| `from utils.synthetic.pipeline.final_aligned_observation_writer import ( build_final_aligned_observations_stage,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.final_aligned_output_writer import build_synthetic_final_aligned_output_stage` | Imports a dependency or project helper used by later cells. |
| `from utils.synthetic.pipeline.final_aligned_incremental import run_final_align_loop` | Imports a dependency or project helper used by later cells. |
| `from utils.core.env_helpers import ( env_bool, env_int, env_optional_int, env_str,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `pd.set_option("display.max_colwidth", None)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 03 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `aligned`
- `batch`
- `broken`
- `BROKEN`
- `capstone`
- `CAPSTONE_SCHEMA`
- `config`
- `created_at`
- `dataset`
- `env_bool`
- `env_int`
- `env_optional_int`
- `env_str`
- `event_time`
- `failed`
- `failure`
- `fault`
- `Final`
- `FINAL_ALIGNMENT_SOURCE_TABLE_NAME`

### Outputs

- `aliases`
- `BATCH_SIZE`
- `COMPLETE_ONLY_FLAG`
- `DATASET_ID`
- `default`
- `IF_EXISTS_FLAG`
- `MAX_ITERATIONS`
- `NUMBER_OF_SENSORS`
- `OBSERVATION_WINDOW_SIZE`
- `PREMELT_TABLE_NAME`
- `REBUILT_TABLE_NAME`
- `RUN_ID`
- `SCHEMA`
- `STATUS_MAPPING`
- `STATUS_SOURCE_PRIORITY`
- `STOP_ON_FAILURE`
- `TARGET_TABLE_NAME`
- `TIMESTAMP_SOURCE_PRIORITY`

### Key Operations

- `SCHEMA = env_str("CAPSTONE_SCHEMA", "capstone")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_ID = env_str( "SYNTHETIC_DATASET_ID", "pump_synthetic_v1", aliases=("DATASET_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `RUN_ID = env_str( "SYNTHETIC_RUN_ID", "synthetic_run_001", aliases=("RUN_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `NUMBER_OF_SENSORS = env_int("SYNTHETIC_SENSOR_COUNT", 52)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `IF_EXISTS_FLAG = "replace"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `COMPLETE_ONLY_FLAG = env_bool( "SYNTHETIC_FINAL_ALIGN_COMPLETE_ONLY", True,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `OBSERVATION_WINDOW_SIZE = env_int( "SYNTHETIC_REBUILD_OBSERVATION_WINDOW_SIZE", 2500,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `BATCH_SIZE = env_int( "SYNTHETIC_FINAL_ALIGN_BATCH_SIZE", 1000,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `env_bool`
- `env_int`
- `env_optional_int`
- `env_str`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `SCHEMA = env_str("CAPSTONE_SCHEMA", "capstone")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_ID = env_str( "SYNTHETIC_DATASET_ID", "pump_synthetic_v1", aliases=("DATASET_ID",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `RUN_ID = env_str( "SYNTHETIC_RUN_ID", "synthetic_run_001", aliases=("RUN_ID",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `NUMBER_OF_SENSORS = env_int("SYNTHETIC_SENSOR_COUNT", 52)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `IF_EXISTS_FLAG = "replace"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `COMPLETE_ONLY_FLAG = env_bool( "SYNTHETIC_FINAL_ALIGN_COMPLETE_ONLY", True,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `OBSERVATION_WINDOW_SIZE = env_int( "SYNTHETIC_REBUILD_OBSERVATION_WINDOW_SIZE", 2500,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BATCH_SIZE = env_int( "SYNTHETIC_FINAL_ALIGN_BATCH_SIZE", 1000,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `MAX_ITERATIONS = env_optional_int( "SYNTHETIC_FINAL_ALIGN_MAX_ITERATIONS", default=None,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STOP_ON_FAILURE = env_bool( "SYNTHETIC_FINAL_ALIGN_STOP_ON_FAILURE", True,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PREMELT_TABLE_NAME = env_str( "SYNTHETIC_PREMELT_OBSERVATIONS_TABLE", "synthetic_observations_premelt_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `REBUILT_TABLE_NAME = env_str( "SYNTHETIC_REBUILT_OBSERVATIONS_TABLE", "synthetic_sensor_observations_rebuilt_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `TARGET_TABLE_NAME = env_str( "SYNTHETIC_FINAL_ALIGNED_OBSERVATIONS_TABLE", "synthetic_sensor_observations_final_aligned_stage", aliases=("FINAL_ALIGNMENT_SOURCE_TABLE_NAME",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `TIMESTAMP_SOURCE_PRIORITY = ( "observation_timestamp", "timestamp", "created_at", "event_time",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STATUS_SOURCE_PRIORITY = ( "machine_status", "stream_state", "phase",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STATUS_MAPPING = { "normal": "NORMAL", "broken": "BROKEN", "abnormal": "BROKEN", "failure": "BROKEN", "failed": "BROKEN", "fault": "BROKEN", "recovering": "RECOVERING", "recovery":` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Final aligned config")` | Displays a notebook-facing result for inspection. |
| `print("schema:", SCHEMA)` | Displays a notebook-facing result for inspection. |
| `print("dataset id:", DATASET_ID)` | Displays a notebook-facing result for inspection. |
| `print("run id:", RUN_ID)` | Displays a notebook-facing result for inspection. |
| `print("premelt table:", PREMELT_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("rebuilt table:", REBUILT_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("target table:", TARGET_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("batch size:", BATCH_SIZE)` | Displays a notebook-facing result for inspection. |
| `print("max iterations:", MAX_ITERATIONS)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 04 — Code Reference

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

## Code Cell 05 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `column_name`
- `columns`
- `data_type`
- `engine`
- `information_schema`
- `ORDER`
- `ordinal_position`
- `read_sql_dataframe`
- `SCHEMA`
- `schema_name`
- `SELECT`
- `table_name`
- `table_schema`
- `TARGET_TABLE_NAME`
- `WHERE`

### Outputs

- `params`
- `stage_11a_final_aligned_columns_dataframe`

### Key Operations

- `stage_11a_final_aligned_columns_dataframe = read_sql_dataframe( engine, """ SELECT ordinal_position, column_name, data_type FROM information_schema.columns WHERE table_schema = :sc`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_11a_final_aligned_columns_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_11a_final_aligned_columns_dataframe = read_sql_dataframe( engine, """ SELECT ordinal_position, column_name, data_type FROM information_schema.columns WHERE table_schema = :sc` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_11a_final_aligned_columns_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 06 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `CASCADE`
- `DROP`
- `Dropped`
- `engine`
- `execute_sql`
- `EXISTS`
- `f`
- `SCHEMA`
- `Stage`
- `stale`
- `table`
- `TABLE`
- `target`
- `TARGET_TABLE_NAME`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `execute_sql( engine, f""" DROP TABLE IF EXISTS "{SCHEMA}"."{TARGET_TABLE_NAME}" CASCADE """`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print(f"Dropped stale Stage 11 target table: {SCHEMA}.{TARGET_TABLE_NAME}")`: Displays a notebook-facing result for inspection.
- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `execute_sql`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `execute_sql( engine, f""" DROP TABLE IF EXISTS "{SCHEMA}"."{TARGET_TABLE_NAME}" CASCADE """` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"Dropped stale Stage 11 target table: {SCHEMA}.{TARGET_TABLE_NAME}")` | Displays a notebook-facing result for inspection. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 07 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `aligned`
- `Build`
- `build_final_aligned_observations_stage`
- `DataFrame`
- `final`
- `full`
- `NUMBER_OF_SENSORS`
- `observation`
- `PREMELT_TABLE_NAME`
- `REBUILT_TABLE_NAME`
- `replace`
- `stage`
- `TARGET_TABLE_NAME`

### Outputs

- `complete_only`
- `dataset_id`
- `engine`
- `if_exists`
- `n_sensors`
- `observation_window_size`
- `prefer_rebuilt_sensor_values`
- `premelt_table`
- `rebuilt_table`
- `run_id`
- `schema`
- `stage_11a_result`
- `target_table`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build full final aligned observation stage`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `stage_11a_result = build_final_aligned_observations_stage( engine=engine, schema=SCHEMA, premelt_table=PREMELT_TABLE_NAME, rebuilt_table=REBUILT_TABLE_NAME, target_table=TARGET_TAB`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(pd.DataFrame([stage_11a_result]))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_final_aligned_observations_stage`
- `DataFrame`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build full final aligned observation stage` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `stage_11a_result = build_final_aligned_observations_stage( engine=engine, schema=SCHEMA, premelt_table=PREMELT_TABLE_NAME, rebuilt_table=REBUILT_TABLE_NAME, target_table=TARGET_TAB` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(pd.DataFrame([stage_11a_result]))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 08 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `COUNT`
- `dataset_id`
- `DATASET_ID`
- `DISTINCT`
- `distinct_generated_row_id_count`
- `distinct_observation_count`
- `engine`
- `f`
- `FILTER`
- `final_row_count`
- `generated_row_id`
- `MAX`
- `max_observation_index`
- `max_observation_timestamp`
- `MIN`
- `min_observation_index`
- `min_observation_timestamp`
- `NULL`
- `null_generated_row_id_count`
- `null_observation_index_count`

### Outputs

- `params`
- `stage_11a_validation_dataframe`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Validate Stage 11A`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `stage_11a_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS final_row_count, COUNT(DISTINCT observation_index) AS distinct_observation_count, COUNT(DISTINC`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_11a_validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `MAX`
- `MIN`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Validate Stage 11A` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `stage_11a_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS final_row_count, COUNT(DISTINCT observation_index) AS distinct_observation_count, COUNT(DISTINC` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_11a_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 09 — Sample Output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `asset_id`
- `BY`
- `dataset_id`
- `DATASET_ID`
- `engine`
- `f`
- `generated_row_id`
- `LIMIT`
- `observation_index`
- `observation_timestamp`
- `ORDER`
- `phase`
- `read_sql_dataframe`
- `rebuild_is_complete`
- `rebuild_sensor_count`
- `run_id`
- `RUN_ID`
- `SCHEMA`
- `SELECT`
- `sensor_00`

### Outputs

- `params`
- `stage_11a_sample_dataframe`

### Key Operations

- `stage_11a_sample_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, generated_row_id, observation_index, observation_timestamp, stream_state, phase, `: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_11a_sample_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_11a_sample_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, generated_row_id, observation_index, observation_timestamp, stream_state, phase, ` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_11a_sample_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 10 — Sample Output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `COUNT`
- `dataset_id`
- `DATASET_ID`
- `engine`
- `f`
- `GROUP`
- `ORDER`
- `phase`
- `read_sql_dataframe`
- `row_count`
- `run_id`
- `RUN_ID`
- `SCHEMA`
- `SELECT`
- `stream_state`
- `TARGET_TABLE_NAME`
- `WHERE`

### Outputs

- `params`
- `stage_11a_status_distribution_dataframe`

### Key Operations

- `stage_11a_status_distribution_dataframe = read_sql_dataframe( engine, f""" SELECT stream_state, phase, COUNT(*) AS row_count FROM "{SCHEMA}"."{TARGET_TABLE_NAME}" WHERE dataset_id `: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_11a_status_distribution_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_11a_status_distribution_dataframe = read_sql_dataframe( engine, f""" SELECT stream_state, phase, COUNT(*) AS row_count FROM "{SCHEMA}"."{TARGET_TABLE_NAME}" WHERE dataset_id ` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_11a_status_distribution_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 11 — Sample Output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `COUNT`
- `dataset_id`
- `DATASET_ID`
- `DISTINCT`
- `distinct_generated_row_id_count`
- `distinct_observation_count`
- `engine`
- `f`
- `FILTER`
- `final_row_count`
- `generated_row_id`
- `MAX`
- `max_observation_index`
- `max_observation_timestamp`
- `MIN`
- `min_observation_index`
- `min_observation_timestamp`
- `NULL`
- `null_generated_row_id_count`
- `null_observation_index_count`

### Outputs

- `params`
- `stage_11a_validation_dataframe`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Validate Stage 11A`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `stage_11a_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS final_row_count, COUNT(DISTINCT observation_index) AS distinct_observation_count, COUNT(DISTINC`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_11a_validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `MAX`
- `MIN`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Validate Stage 11A` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `stage_11a_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS final_row_count, COUNT(DISTINCT observation_index) AS distinct_observation_count, COUNT(DISTINC` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_11a_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 12 — Sample Output

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

