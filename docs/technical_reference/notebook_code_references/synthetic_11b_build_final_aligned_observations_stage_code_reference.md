# Notebook Code Reference: synthetic_11b_build_final_aligned_observations_stage

Notebook path:

`notebooks/synthetic/synthetic_11b_build_final_aligned_observations_stage.ipynb`

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

- `a`
- `Build`
- `compact`
- `Compact`
- `final`
- `Final`
- `Kaggle`
- `like`
- `output`
- `Output`
- `Purpose`
- `Source`
- `Stage`
- `Synthetic`
- `synthetic_sensor_observations_final_output`
- `synthetic_sensor_observations_rebuilt_stage`
- `table`
- `Target`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Stage 11B — Compact Synthetic Final Output`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Purpose:`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build a Kaggle-like compact final output table.`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# Source:`: Documents the purpose or boundary of the surrounding notebook step.
- `# - synthetic_sensor_observations_rebuilt_stage`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# Target:`: Documents the purpose or boundary of the surrounding notebook step.
- `# - synthetic_sensor_observations_final_output`: Documents the purpose or boundary of the surrounding notebook step.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Stage 11B — Compact Synthetic Final Output` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Purpose:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build a Kaggle-like compact final output table.` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Source:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - synthetic_sensor_observations_rebuilt_stage` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Target:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - synthetic_sensor_observations_final_output` | Documents the purpose or boundary of the surrounding notebook step. |

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
- `BROKEN`
- `broken`
- `capstone`
- `CAPSTONE_SCHEMA`
- `config`
- `dataset`
- `env_bool`
- `env_int`
- `env_str`
- `failed`
- `failure`
- `fault`
- `final`
- `Final`
- `id`
- `machine_status`
- `NORMAL`

### Outputs

- `aliases`
- `COMPLETE_ONLY_FLAG`
- `DATASET_ID`
- `FINAL_OUTPUT_TABLE_NAME`
- `IF_EXISTS_FLAG`
- `MACHINE_STATUS_OUTPUT_COLUMN`
- `N_SENSORS`
- `NUMBER_OF_SENSORS`
- `OBSERVATION_WINDOW_SIZE`
- `REBUILT_TABLE_NAME`
- `RUN_ID`
- `SCHEMA`
- `STATUS_MAPPING`
- `TIMESTAMP_OUTPUT_COLUMN`

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
- `REBUILT_TABLE_NAME = env_str( "SYNTHETIC_REBUILT_OBSERVATIONS_TABLE", "synthetic_sensor_observations_rebuilt_stage",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `env_bool`
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
| `NUMBER_OF_SENSORS = env_int("SYNTHETIC_SENSOR_COUNT", 52)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `IF_EXISTS_FLAG = "replace"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `COMPLETE_ONLY_FLAG = env_bool( "SYNTHETIC_FINAL_ALIGN_COMPLETE_ONLY", True,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `OBSERVATION_WINDOW_SIZE = env_int( "SYNTHETIC_REBUILD_OBSERVATION_WINDOW_SIZE", 2500,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `REBUILT_TABLE_NAME = env_str( "SYNTHETIC_REBUILT_OBSERVATIONS_TABLE", "synthetic_sensor_observations_rebuilt_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FINAL_OUTPUT_TABLE_NAME = env_str( "SYNTHETIC_FINAL_OUTPUT_TABLE", "synthetic_sensor_observations_final_output",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `N_SENSORS = NUMBER_OF_SENSORS` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TIMESTAMP_OUTPUT_COLUMN = "timestamp"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MACHINE_STATUS_OUTPUT_COLUMN = "machine_status"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STATUS_MAPPING = { "normal": "NORMAL", "broken": "BROKEN", "abnormal": "BROKEN", "failure": "BROKEN", "failed": "BROKEN", "fault": "BROKEN", "recovering": "RECOVERING", "recovery":` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Final aligned config")` | Displays a notebook-facing result for inspection. |
| `print("schema:", SCHEMA)` | Displays a notebook-facing result for inspection. |
| `print("dataset id:", DATASET_ID)` | Displays a notebook-facing result for inspection. |
| `print("run id:", RUN_ID)` | Displays a notebook-facing result for inspection. |
| `print("rebuilt table:", REBUILT_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("final output table:", FINAL_OUTPUT_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("batch size:", OBSERVATION_WINDOW_SIZE)` | Displays a notebook-facing result for inspection. |

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
- `FINAL_OUTPUT_TABLE_NAME`
- `information_schema`
- `ORDER`
- `ordinal_position`
- `read_sql_dataframe`
- `SCHEMA`
- `schema_name`
- `SELECT`
- `table_name`
- `table_schema`
- `WHERE`

### Outputs

- `params`
- `stage_11b_final_output_columns_dataframe`

### Key Operations

- `stage_11b_final_output_columns_dataframe = read_sql_dataframe( engine, """ SELECT ordinal_position, column_name, data_type FROM information_schema.columns WHERE table_schema = :sch`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_11b_final_output_columns_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_11b_final_output_columns_dataframe = read_sql_dataframe( engine, """ SELECT ordinal_position, column_name, data_type FROM information_schema.columns WHERE table_schema = :sch` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_11b_final_output_columns_dataframe)` | Displays a notebook-facing result for inspection. |

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

- `Build`
- `build_synthetic_final_aligned_output_stage`
- `compact`
- `DataFrame`
- `FINAL_OUTPUT_TABLE_NAME`
- `Kaggle`
- `like`
- `output`
- `REBUILT_TABLE_NAME`
- `replace`
- `synthetic`

### Outputs

- `complete_only`
- `dataset_id`
- `engine`
- `if_exists`
- `machine_status_output_column`
- `n_sensors`
- `observation_window_size`
- `rebuilt_table`
- `run_id`
- `schema`
- `stage_11b_result`
- `target_table`
- `timestamp_output_column`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build compact Kaggle-like synthetic output`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `stage_11b_result = build_synthetic_final_aligned_output_stage( engine=engine, schema=SCHEMA, rebuilt_table=REBUILT_TABLE_NAME, target_table=FINAL_OUTPUT_TABLE_NAME, dataset_id=DATA`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(pd.DataFrame([stage_11b_result]))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_synthetic_final_aligned_output_stage`
- `DataFrame`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build compact Kaggle-like synthetic output` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `stage_11b_result = build_synthetic_final_aligned_output_stage( engine=engine, schema=SCHEMA, rebuilt_table=REBUILT_TABLE_NAME, target_table=FINAL_OUTPUT_TABLE_NAME, dataset_id=DATA` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(pd.DataFrame([stage_11b_result]))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 08 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `asset_id`
- `asset_id_count`
- `BROKEN`
- `broken_rows`
- `compact`
- `COUNT`
- `DATASET_ID`
- `dataset_id`
- `dataset_id_count`
- `DISTINCT`
- `engine`
- `f`
- `FILTER`
- `final`
- `FINAL_OUTPUT_TABLE_NAME`
- `MACHINE_STATUS_OUTPUT_COLUMN`
- `MAX`
- `max_timestamp`
- `MIN`
- `min_timestamp`

### Outputs

- `params`
- `stage_11b_validation_dataframe`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Validate compact final output`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `stage_11b_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS row_count, COUNT(DISTINCT dataset_id) AS dataset_id_count, COUNT(DISTINCT run_id) AS run_id_cou`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_11b_validation_dataframe)`: Displays a notebook-facing result for inspection.

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
| `# Validate compact final output` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `stage_11b_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS row_count, COUNT(DISTINCT dataset_id) AS dataset_id_count, COUNT(DISTINCT run_id) AS run_id_cou` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_11b_validation_dataframe)` | Displays a notebook-facing result for inspection. |

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
- `FINAL_OUTPUT_TABLE_NAME`
- `LIMIT`
- `MACHINE_STATUS_OUTPUT_COLUMN`
- `ORDER`
- `read_sql_dataframe`
- `RUN_ID`
- `run_id`
- `SCHEMA`
- `SELECT`
- `sensor_00`
- `sensor_01`
- `sensor_02`
- `sensor_03`
- `sensor_04`

### Outputs

- `params`
- `stage_11b_sample_dataframe`

### Key Operations

- `stage_11b_sample_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, "{TIMESTAMP_OUTPUT_COLUMN}", sensor_00, sensor_01, sensor_02, sensor_03, sensor_0`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_11b_sample_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_11b_sample_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, "{TIMESTAMP_OUTPUT_COLUMN}", sensor_00, sensor_01, sensor_02, sensor_03, sensor_0` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_11b_sample_dataframe)` | Displays a notebook-facing result for inspection. |

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
- `FINAL_OUTPUT_TABLE_NAME`
- `GROUP`
- `machine_status`
- `MACHINE_STATUS_OUTPUT_COLUMN`
- `ORDER`
- `read_sql_dataframe`
- `row_count`
- `RUN_ID`
- `run_id`
- `SCHEMA`
- `SELECT`
- `WHERE`

### Outputs

- `params`
- `stage_11b_status_distribution_dataframe`

### Key Operations

- `stage_11b_status_distribution_dataframe = read_sql_dataframe( engine, f""" SELECT "{MACHINE_STATUS_OUTPUT_COLUMN}" AS machine_status, COUNT(*) AS row_count FROM "{SCHEMA}"."{FINAL_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_11b_status_distribution_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_11b_status_distribution_dataframe = read_sql_dataframe( engine, f""" SELECT "{MACHINE_STATUS_OUTPUT_COLUMN}" AS machine_status, COUNT(*) AS row_count FROM "{SCHEMA}"."{FINAL_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_11b_status_distribution_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 11 — Sample Output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `compact`
- `DATASET_ID`
- `dataset_id`
- `engine`
- `f`
- `final`
- `FINAL_OUTPUT_TABLE_NAME`
- `LIMIT`
- `ORDER`
- `output`
- `Preview`
- `read_sql_dataframe`
- `RUN_ID`
- `run_id`
- `SCHEMA`
- `SELECT`
- `TIMESTAMP_OUTPUT_COLUMN`
- `WHERE`

### Outputs

- `params`
- `preview_final_output_dataframe`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Preview compact final output`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `preview_final_output_dataframe = read_sql_dataframe( engine, f""" SELECT * FROM "{SCHEMA}"."{FINAL_OUTPUT_TABLE_NAME}" WHERE dataset_id = :dataset_id AND run_id = :run_id ORDER BY `: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(preview_final_output_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Preview compact final output` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `preview_final_output_dataframe = read_sql_dataframe( engine, f""" SELECT * FROM "{SCHEMA}"."{FINAL_OUTPUT_TABLE_NAME}" WHERE dataset_id = :dataset_id AND run_id = :run_id ORDER BY ` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(preview_final_output_dataframe)` | Displays a notebook-facing result for inspection. |

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

