# Notebook Code Reference: synthetic_10_rebuild_comparison

Notebook path:

`notebooks/synthetic/synthetic_10_rebuild_comparison.ipynb`

## Notebook Purpose

This notebook documents and runs a focused support step in the synthetic pump telemetry workflow.

Notebook stage:

`Synthetic support`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08, Code Cell 09, Code Cell 10, Code Cell 11, Code Cell 12, Code Cell 13, Code Cell 14 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `build_rebuild_comparison_stage`
- `cell`
- `column`
- `columns`
- `content`
- `core`
- `count`
- `cutting`
- `database`
- `env_float`
- `env_helpers`
- `env_int`
- `env_str`
- `every`
- `execute_sql`
- `get_engine_from_env`
- `inside`
- `line`
- `max_columns`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe, execute_sql,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.rebuild_comparison import build_rebuild_comparison_stage`: Imports a dependency or project helper used by later cells.
- `from utils.core.env_helpers import ( env_float, env_int, env_str`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Show every column regardless of count`: Documents the purpose or boundary of the surrounding notebook step.
- `pd.set_option('display.max_columns', None)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Prevent columns from wrapping to a new line`: Documents the purpose or boundary of the surrounding notebook step.
- `pd.set_option('display.width', 1000)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Optional: Prevent cell content inside columns from cutting off`: Documents the purpose or boundary of the surrounding notebook step.

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
| `from utils.synthetic.pipeline.rebuild_comparison import build_rebuild_comparison_stage` | Imports a dependency or project helper used by later cells. |
| `from utils.core.env_helpers import ( env_float, env_int, env_str` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Show every column regardless of count` | Documents the purpose or boundary of the surrounding notebook step. |
| `pd.set_option('display.max_columns', None)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Prevent columns from wrapping to a new line` | Documents the purpose or boundary of the surrounding notebook step. |
| `pd.set_option('display.width', 1000)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Optional: Prevent cell content inside columns from cutting off` | Documents the purpose or boundary of the surrounding notebook step. |
| `pd.set_option('display.max_colwidth', None)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `capstone`
- `CAPSTONE_SCHEMA`
- `comparison`
- `config`
- `dataset`
- `env_float`
- `env_int`
- `env_str`
- `id`
- `premelt`
- `pump_synthetic_v1`
- `Rebuild`
- `rebuilt`
- `run`
- `SYNTHETIC_DATASET_ID`
- `synthetic_observations_premelt_stage`
- `SYNTHETIC_PREMELT_OBSERVATIONS_TABLE`
- `SYNTHETIC_REBUILD_COMPARISON_FLOAT_TOLERANCE`
- `SYNTHETIC_REBUILD_COMPARISON_TABLE`
- `SYNTHETIC_REBUILD_OBSERVATION_WINDOW_SIZE`

### Outputs

- `aliases`
- `DATASET_ID`
- `FLOAT_TOLERANCE`
- `NUMBER_OF_SENSORS`
- `OBSERVATION_WINDOW_SIZE`
- `PREMELT_SOURCE_TABLE_NAME`
- `REBUILT_DESTINATION_TABLE_NAME`
- `RUN_ID`
- `SCHEMA`
- `TARGET_TABLE_NAME`

### Key Operations

- `SCHEMA = env_str("CAPSTONE_SCHEMA", "capstone")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_ID = env_str( "SYNTHETIC_DATASET_ID", "pump_synthetic_v1", aliases=("DATASET_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `RUN_ID = env_str( "SYNTHETIC_RUN_ID", "synthetic_run_001", aliases=("RUN_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `NUMBER_OF_SENSORS = env_int("SYNTHETIC_SENSOR_COUNT", 52)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `FLOAT_TOLERANCE = env_float( "SYNTHETIC_REBUILD_COMPARISON_FLOAT_TOLERANCE", 1e-9,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `OBSERVATION_WINDOW_SIZE = env_int( "SYNTHETIC_REBUILD_OBSERVATION_WINDOW_SIZE", 2500,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `PREMELT_SOURCE_TABLE_NAME = env_str( "SYNTHETIC_PREMELT_OBSERVATIONS_TABLE", "synthetic_observations_premelt_stage",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

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
| `NUMBER_OF_SENSORS = env_int("SYNTHETIC_SENSOR_COUNT", 52)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FLOAT_TOLERANCE = env_float( "SYNTHETIC_REBUILD_COMPARISON_FLOAT_TOLERANCE", 1e-9,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `OBSERVATION_WINDOW_SIZE = env_int( "SYNTHETIC_REBUILD_OBSERVATION_WINDOW_SIZE", 2500,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PREMELT_SOURCE_TABLE_NAME = env_str( "SYNTHETIC_PREMELT_OBSERVATIONS_TABLE", "synthetic_observations_premelt_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `REBUILT_DESTINATION_TABLE_NAME = env_str( "SYNTHETIC_REBUILT_OBSERVATIONS_TABLE", "synthetic_sensor_observations_rebuilt_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `TARGET_TABLE_NAME = env_str( "SYNTHETIC_REBUILD_COMPARISON_TABLE", "synthetic_sensor_rebuild_comparison_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Rebuild comparison config")` | Displays a notebook-facing result for inspection. |
| `print("schema:", SCHEMA)` | Displays a notebook-facing result for inspection. |
| `print("dataset id:", DATASET_ID)` | Displays a notebook-facing result for inspection. |
| `print("run id:", RUN_ID)` | Displays a notebook-facing result for inspection. |
| `print("premelt table:", PREMELT_SOURCE_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("rebuilt table:", REBUILT_DESTINATION_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("target table:", TARGET_TABLE_NAME)` | Displays a notebook-facing result for inspection. |

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

- `build_rebuild_comparison_stage`
- `NUMBER_OF_SENSORS`
- `PREMELT_SOURCE_TABLE_NAME`
- `REBUILT_DESTINATION_TABLE_NAME`
- `TARGET_TABLE_NAME`

### Outputs

- `comparison_result`
- `dataset_id`
- `engine`
- `float_tolerance`
- `n_sensors`
- `observation_window_size`
- `premelt_table`
- `rebuilt_table`
- `run_id`
- `schema`
- `target_table`

### Key Operations

- `comparison_result = build_rebuild_comparison_stage( engine=engine, schema=SCHEMA, premelt_table=PREMELT_SOURCE_TABLE_NAME, rebuilt_table=REBUILT_DESTINATION_TABLE_NAME, target_tabl`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(comparison_result)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_rebuild_comparison_stage`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `comparison_result = build_rebuild_comparison_stage( engine=engine, schema=SCHEMA, premelt_table=PREMELT_SOURCE_TABLE_NAME, rebuilt_table=REBUILT_DESTINATION_TABLE_NAME, target_tabl` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(comparison_result)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 05 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `all_match_count`
- `comparison_all_fields_match`
- `comparison_mismatch_count`
- `comparison_row_count`
- `COUNT`
- `engine`
- `f`
- `FILTER`
- `MAX`
- `max_mismatch_count`
- `mismatch_count`
- `read_sql_dataframe`
- `SCHEMA`
- `SELECT`
- `TARGET_TABLE_NAME`
- `WHERE`

### Outputs

- `summary_dataframe`

### Key Operations

- `summary_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS comparison_row_count, COUNT(*) FILTER (WHERE comparison_all_fields_match = TRUE) AS all_match_count, COUNT(*`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(summary_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `MAX`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `summary_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS comparison_row_count, COUNT(*) FILTER (WHERE comparison_all_fields_match = TRUE) AS all_match_count, COUNT(*` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(summary_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 06 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `asset_id`
- `BY`
- `comparison_all_fields_match`
- `comparison_mismatch_count`
- `comparison_notes`
- `DATASET_ID`
- `dataset_id`
- `engine`
- `exists_in_original`
- `exists_in_rebuilt`
- `f`
- `LIMIT`
- `observation_index`
- `ORDER`
- `read_sql_dataframe`
- `rebuilt_rebuild_is_complete`
- `rebuilt_rebuild_sensor_count`
- `run_id`
- `RUN_ID`
- `SCHEMA`

### Outputs

- `mismatch_dataframe`
- `params`

### Key Operations

- `mismatch_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, observation_index, comparison_mismatch_count, comparison_notes, exists_in_original, exist`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(mismatch_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `mismatch_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, observation_index, comparison_mismatch_count, comparison_notes, exists_in_original, exist` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(mismatch_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 07 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `column`
- `column_name`
- `columns`
- `DataFrame`
- `engine`
- `found`
- `head`
- `information_schema`
- `match`
- `Match`
- `match_column`
- `ORDER`
- `ordinal_position`
- `read_sql_dataframe`
- `SCHEMA`
- `schema_name`
- `SELECT`
- `startswith`
- `table_name`

### Outputs

- `comparison_columns_dataframe`
- `match_columns`
- `params`

### Key Operations

- `comparison_columns_dataframe = read_sql_dataframe( engine, """ SELECT column_name FROM information_schema.columns WHERE table_schema = :schema_name AND table_name = :table_name ORD`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `match_columns = [ column for column in comparison_columns_dataframe["column_name"].tolist() if column.startswith("match")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Match columns found:", len(match_columns))`: Displays a notebook-facing result for inspection.
- `display(pd.DataFrame({"match_column": match_columns}).head(100))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `DataFrame`
- `display`
- `head`
- `read_sql_dataframe`
- `startswith`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `comparison_columns_dataframe = read_sql_dataframe( engine, """ SELECT column_name FROM information_schema.columns WHERE table_schema = :schema_name AND table_name = :table_name ORD` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `match_columns = [ column for column in comparison_columns_dataframe["column_name"].tolist() if column.startswith("match")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Match columns found:", len(match_columns))` | Displays a notebook-facing result for inspection. |
| `display(pd.DataFrame({"match_column": match_columns}).head(100))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 08 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `ALL`
- `BY`
- `column`
- `COUNT`
- `dataset_id`
- `DATASET_ID`
- `DESC`
- `engine`
- `f`
- `failed_count`
- `FILTER`
- `join`
- `match_column`
- `match_columns`
- `n`
- `nORDER`
- `nUNION`
- `passed_count`
- `read_sql_dataframe`
- `RUN_ID`

### Outputs

- `match_failure_sql_parts`
- `match_failure_summary_dataframe`
- `params`

### Key Operations

- `match_failure_sql_parts = [ f""" SELECT '{column}' AS match_column, COUNT(*) FILTER (WHERE "{column}" = FALSE) AS failed_count, COUNT(*) FILTER (WHERE "{column}" = TRUE) AS passed_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `match_failure_summary_dataframe = read_sql_dataframe( engine, "\nUNION ALL\n".join(match_failure_sql_parts) + "\nORDER BY failed_count DESC, match_column", params={ "dataset_id": D`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(match_failure_summary_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `join`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `match_failure_sql_parts = [ f""" SELECT '{column}' AS match_column, COUNT(*) FILTER (WHERE "{column}" = FALSE) AS failed_count, COUNT(*) FILTER (WHERE "{column}" = TRUE) AS passed_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `match_failure_summary_dataframe = read_sql_dataframe( engine, "\nUNION ALL\n".join(match_failure_sql_parts) + "\nORDER BY failed_count DESC, match_column", params={ "dataset_id": D` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(match_failure_summary_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 09 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `asset_id`
- `BY`
- `dataset_id`
- `DATASET_ID`
- `engine`
- `f`
- `LIMIT`
- `match_meta_episode_id`
- `observation_index`
- `ORDER`
- `original_meta_episode_id`
- `read_sql_dataframe`
- `rebuilt_meta_episode_id`
- `RUN_ID`
- `run_id`
- `SCHEMA`
- `SELECT`
- `TARGET_TABLE_NAME`
- `WHERE`

### Outputs

- `episode_id_mismatch_sample_dataframe`
- `params`

### Key Operations

- `episode_id_mismatch_sample_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, observation_index, original_meta_episode_id, rebuilt_meta_episode_id, m`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(episode_id_mismatch_sample_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `episode_id_mismatch_sample_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, observation_index, original_meta_episode_id, rebuilt_meta_episode_id, m` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(episode_id_mismatch_sample_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 10 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `engine`
- `f`
- `observation_index`
- `read_sql_dataframe`
- `SCHEMA`
- `SELECT`
- `TARGET_TABLE_NAME`
- `WHERE`

### Outputs

- `detail_dataframe`
- `observation_to_check`

### Key Operations

- `observation_to_check = 1`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `detail_dataframe = read_sql_dataframe( engine, f""" SELECT * FROM "{SCHEMA}"."{TARGET_TABLE_NAME}" WHERE observation_index = {int(observation_to_check)} """,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(detail_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `observation_to_check = 1` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `detail_dataframe = read_sql_dataframe( engine, f""" SELECT * FROM "{SCHEMA}"."{TARGET_TABLE_NAME}" WHERE observation_index = {int(observation_to_check)} """,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(detail_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 11 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `all_fields_match_count`
- `comparison_all_fields_match`
- `comparison_mismatch_count`
- `comparison_row_count`
- `COUNT`
- `dataset_id`
- `DATASET_ID`
- `engine`
- `exists_in_original`
- `exists_in_rebuilt`
- `f`
- `FILTER`
- `MAX`
- `max_mismatch_count`
- `mismatch_row_count`
- `missing_from_original_count`
- `missing_from_rebuilt_count`
- `read_sql_dataframe`
- `ready_for_stage_11`
- `run_id`

### Outputs

- `params`
- `stage_10_final_validation_dataframe`

### Key Operations

- `stage_10_final_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS comparison_row_count, COUNT(*) FILTER (WHERE comparison_all_fields_match = TRUE) AS all_fi`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_10_final_validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `MAX`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_10_final_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS comparison_row_count, COUNT(*) FILTER (WHERE comparison_all_fields_match = TRUE) AS all_fi` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_10_final_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 12 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__file__`
- `_compare_scalar`
- `pipeline`
- `rebuild_comparison`
- `rebuild_comparison_module`
- `synthetic`
- `utils`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import utils.synthetic.pipeline.rebuild_comparison as rebuild_comparison_module`: Imports a dependency or project helper used by later cells.
- `print(rebuild_comparison_module.__file__)`: Displays a notebook-facing result for inspection.
- `print(rebuild_comparison_module._compare_scalar(0, "0"))`: Displays a notebook-facing result for inspection.
- `print(rebuild_comparison_module._compare_scalar(0.0, "0"))`: Displays a notebook-facing result for inspection.
- `print(rebuild_comparison_module._compare_scalar("0", "0"))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `_compare_scalar`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import utils.synthetic.pipeline.rebuild_comparison as rebuild_comparison_module` | Imports a dependency or project helper used by later cells. |
| `print(rebuild_comparison_module.__file__)` | Displays a notebook-facing result for inspection. |
| `print(rebuild_comparison_module._compare_scalar(0, "0"))` | Displays a notebook-facing result for inspection. |
| `print(rebuild_comparison_module._compare_scalar(0.0, "0"))` | Displays a notebook-facing result for inspection. |
| `print(rebuild_comparison_module._compare_scalar("0", "0"))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 13 — Code Reference

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
- `TABLE`
- `TARGET_TABLE_NAME`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `execute_sql( engine, f""" DROP TABLE IF EXISTS "{SCHEMA}"."{TARGET_TABLE_NAME}" CASCADE """`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print(f"Dropped {SCHEMA}.{TARGET_TABLE_NAME}")`: Displays a notebook-facing result for inspection.
- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `execute_sql`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `execute_sql( engine, f""" DROP TABLE IF EXISTS "{SCHEMA}"."{TARGET_TABLE_NAME}" CASCADE """` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"Dropped {SCHEMA}.{TARGET_TABLE_NAME}")` | Displays a notebook-facing result for inspection. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 14 — Code Reference

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

