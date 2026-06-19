# Notebook Code Reference: synthetic_12_bronze_handoff

Notebook path:

`notebooks/synthetic/synthetic_12_bronze_handoff.ipynb`

## Notebook Purpose

This notebook prepares the Bronze layer by ingesting, validating, and standardizing the raw pump dataset for downstream Silver processing.

Notebook stage:

`Bronze`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08, Code Cell 09, Code Cell 10, Code Cell 11 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `bronze_handoff`
- `cast`
- `core`
- `database`
- `env_bool`
- `env_helpers`
- `env_int`
- `env_optional_int`
- `env_str`
- `get_engine_from_env`
- `handoff_final_aligned_observations_to_bronze`
- `os`
- `pandas`
- `pipeline`
- `postgres`
- `read_sql_dataframe`
- `run_bronze_handoff_loop`
- `synthetic`
- `typing`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `from typing import Any, cast`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.bronze_handoff import ( handoff_final_aligned_observations_to_bronze, run_bronze_handoff_loop,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.core.env_helpers import ( env_bool, env_int, env_optional_int, env_str,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `from typing import Any, cast` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.bronze_handoff import ( handoff_final_aligned_observations_to_bronze, run_bronze_handoff_loop,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.env_helpers import ( env_bool, env_int, env_optional_int, env_str,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `aggregate`
- `an`
- `Any`
- `at`
- `avoids`
- `by`
- `Cannot`
- `cast`
- `caused`
- `cells`
- `code`
- `column`
- `Column`
- `column_name`
- `columns`
- `containing`
- `contains`
- `dataframe`
- `Dataframe`

### Outputs

- `dataframe_int_value`
- `value`

### Key Operations

- `def dataframe_int_value( dataframe: pd.DataFrame, column_name: str, row_index: int = 0,`: Defines notebook-local logic used later in the notebook.
- `) -> int: """ Return a dataframe scalar value as an integer. This helper is used in notebook validation cells where SQL aggregate queries return one-row dataframes. It keeps valida`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `cast`
- `dataframe_int_value`
- `isna`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def dataframe_int_value( dataframe: pd.DataFrame, column_name: str, row_index: int = 0,` | Defines notebook-local logic used later in the notebook. |
| `) -> int: """ Return a dataframe scalar value as an integer. This helper is used in notebook validation cells where SQL aggregate queries return one-row dataframes. It keeps valida` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 03 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `batch`
- `Bronze`
- `BRONZE_HANDOFF_OBSERVATIONS_TABLE`
- `bronze_observations_input_stage`
- `capstone`
- `CAPSTONE_SCHEMA`
- `config`
- `dataset`
- `env_bool`
- `env_int`
- `env_optional_int`
- `env_str`
- `handoff`
- `id`
- `iterations`
- `max`
- `pump_synthetic_v1`
- `replace`
- `run`
- `size`

### Outputs

- `aliases`
- `BATCH_SIZE`
- `BRONZE_HANDOFF_TARGET_TABLE_NAME`
- `COMPLETE_ONLY_FLAG`
- `DATASET_ID`
- `default`
- `FINAL_ALIGNMENT_SOURCE_TABLE_NAME`
- `IF_EXISTS_FLAG`
- `MAX_ITERATIONS`
- `RUN_ID`
- `SCHEMA`
- `STOP_ON_FAILURE_FLAG`

### Key Operations

- `SCHEMA = env_str("CAPSTONE_SCHEMA", "capstone")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_ID = env_str( "SYNTHETIC_DATASET_ID", "pump_synthetic_v1", aliases=("DATASET_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `RUN_ID = env_str( "SYNTHETIC_RUN_ID", "synthetic_run_001", aliases=("RUN_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `BATCH_SIZE = env_int( "SYNTHETIC_BRONZE_HANDOFF_BATCH_SIZE", 500,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `IF_EXISTS_FLAG = "replace"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `COMPLETE_ONLY_FLAG = env_bool( "SYNTHETIC_BRONZE_HANDOFF_COMPLETE_ONLY", True,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `STOP_ON_FAILURE_FLAG = env_bool( "SYNTHETIC_BRONZE_HANDOFF_STOP_ON_FAILURE", True,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

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
| `BATCH_SIZE = env_int( "SYNTHETIC_BRONZE_HANDOFF_BATCH_SIZE", 500,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `IF_EXISTS_FLAG = "replace"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `COMPLETE_ONLY_FLAG = env_bool( "SYNTHETIC_BRONZE_HANDOFF_COMPLETE_ONLY", True,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STOP_ON_FAILURE_FLAG = env_bool( "SYNTHETIC_BRONZE_HANDOFF_STOP_ON_FAILURE", True,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `MAX_ITERATIONS = env_optional_int( "SYNTHETIC_BRONZE_HANDOFF_MAX_ITERATIONS", default=None,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FINAL_ALIGNMENT_SOURCE_TABLE_NAME = env_str( "SYNTHETIC_FINAL_ALIGNED_OBSERVATIONS_TABLE", "synthetic_sensor_observations_final_aligned_stage", aliases=("FINAL_ALIGNMENT_SOURCE_TAB` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BRONZE_HANDOFF_TARGET_TABLE_NAME = env_str( "BRONZE_HANDOFF_OBSERVATIONS_TABLE", "bronze_observations_input_stage", aliases=("BRONZE_HANDOFF_TARGET_TABLE_NAME",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Bronze handoff config")` | Displays a notebook-facing result for inspection. |
| `print("schema:", SCHEMA)` | Displays a notebook-facing result for inspection. |
| `print("dataset id:", DATASET_ID)` | Displays a notebook-facing result for inspection. |
| `print("run id:", RUN_ID)` | Displays a notebook-facing result for inspection. |
| `print("source table:", FINAL_ALIGNMENT_SOURCE_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("target table:", BRONZE_HANDOFF_TARGET_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("batch size:", BATCH_SIZE)` | Displays a notebook-facing result for inspection. |
| `print("max iterations:", MAX_ITERATIONS)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 04 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `be`
- `BRONZE_HANDOFF_RUN_MODE`
- `env_str`
- `f`
- `full_batch`
- `must`
- `of`
- `one`
- `r`
- `raise`
- `Received`
- `row`
- `row_batch`
- `SYNTHETIC_BRONZE_HANDOFF_RUN_MODE`
- `ValueError`

### Outputs

- `aliases`
- `RUN_HANDOFF`
- `RUN_MODE`

### Key Operations

- `RUN_HANDOFF = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RUN_MODE = env_str( "SYNTHETIC_BRONZE_HANDOFF_RUN_MODE", "full_batch", aliases=("BRONZE_HANDOFF_RUN_MODE",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if RUN_MODE not in {"row", "row_batch", "full_batch"}: raise ValueError( "RUN_MODE must be one of: 'row', 'row_batch', 'full_batch'. " f"Received: {RUN_MODE!r}" )`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `env_str`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `RUN_HANDOFF = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_MODE = env_str( "SYNTHETIC_BRONZE_HANDOFF_RUN_MODE", "full_batch", aliases=("BRONZE_HANDOFF_RUN_MODE",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if RUN_MODE not in {"row", "row_batch", "full_batch"}: raise ValueError( "RUN_MODE must be one of: 'row', 'row_batch', 'full_batch'. " f"Received: {RUN_MODE!r}" )` | Controls validation, iteration, file handling, or error handling for this step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 05 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `full_batch`
- `Local`
- `Mode`
- `override`
- `row`
- `row_batch`
- `RUN_MODE`
- `Script`
- `style`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `# Local override`: Documents the purpose or boundary of the surrounding notebook step.
- `# Script style`: Documents the purpose or boundary of the surrounding notebook step.
- `# Mode`: Documents the purpose or boundary of the surrounding notebook step.
- `# "row" \| "row_batch" \| "full_batch"`: Documents the purpose or boundary of the surrounding notebook step.
- `#RUN_MODE = "row_batch"`: Documents the purpose or boundary of the surrounding notebook step.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Local override` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Script style` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Mode` | Documents the purpose or boundary of the surrounding notebook step. |
| `# "row" \| "row_batch" \| "full_batch"` | Documents the purpose or boundary of the surrounding notebook step. |
| `#RUN_MODE = "row_batch"` | Documents the purpose or boundary of the surrounding notebook step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 06 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Batch`
- `BATCH_SIZE`
- `Bronze`
- `BRONZE_HANDOFF_TARGET_TABLE_NAME`
- `Complete`
- `COMPLETE_ONLY_FLAG`
- `f`
- `FINAL_ALIGNMENT_SOURCE_TABLE_NAME`
- `handoff`
- `mode`
- `only`
- `run`
- `RUN_MODE`
- `SCHEMA`
- `size`
- `Source`
- `table`
- `Target`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `print(f"Bronze handoff run mode: {RUN_MODE}")`: Displays a notebook-facing result for inspection.
- `print(f"Batch size: {BATCH_SIZE}")`: Displays a notebook-facing result for inspection.
- `print(f"Complete only: {COMPLETE_ONLY_FLAG}")`: Displays a notebook-facing result for inspection.
- `print(f"Source table: {SCHEMA}.{FINAL_ALIGNMENT_SOURCE_TABLE_NAME}")`: Displays a notebook-facing result for inspection.
- `print(f"Target table: {SCHEMA}.{BRONZE_HANDOFF_TARGET_TABLE_NAME}")`: Displays a notebook-facing result for inspection.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print(f"Bronze handoff run mode: {RUN_MODE}")` | Displays a notebook-facing result for inspection. |
| `print(f"Batch size: {BATCH_SIZE}")` | Displays a notebook-facing result for inspection. |
| `print(f"Complete only: {COMPLETE_ONLY_FLAG}")` | Displays a notebook-facing result for inspection. |
| `print(f"Source table: {SCHEMA}.{FINAL_ALIGNMENT_SOURCE_TABLE_NAME}")` | Displays a notebook-facing result for inspection. |
| `print(f"Target table: {SCHEMA}.{BRONZE_HANDOFF_TARGET_TABLE_NAME}")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 07 — Code Reference

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

## Code Cell 08 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `active`
- `alignment`
- `before`
- `bronze_handoff_status`
- `but`
- `Check`
- `complete_only`
- `COMPLETE_ONLY_FLAG`
- `completed`
- `completed_count`
- `configured`
- `COUNT`
- `dataframe_int_value`
- `dataset`
- `dataset_id`
- `DATASET_ID`
- `engine`
- `f`
- `failed`
- `failed_count`

### Outputs

- `complete_row_count`
- `params`
- `source_preflight_df`
- `source_row_count`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Preflight validation for Stage 12 source table`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `source_preflight_df = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS source_row_count, COUNT(*) FILTER (WHERE rebuild_is_complete = TRUE) AS complete_row_count, COUNT(*) FILTE`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(source_preflight_df)`: Displays a notebook-facing result for inspection.
- `source_row_count = dataframe_int_value(source_preflight_df, "source_row_count")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `complete_row_count = dataframe_int_value(source_preflight_df, "complete_row_count")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if source_row_count == 0: raise ValueError( "Stage 12 source table has zero rows for the active dataset/run. " "Run Stage 11 final alignment before Stage 12." )`: Controls validation, iteration, file handling, or error handling for this step.
- `if COMPLETE_ONLY_FLAG and complete_row_count == 0: raise ValueError( "Stage 12 is configured for complete_only=True, but no rows have " "rebuild_is_complete=True. Check Stage 11 ou`: Controls validation, iteration, file handling, or error handling for this step.
- `print("Stage 12 source preflight passed.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `dataframe_int_value`
- `display`
- `FILTER`
- `read_sql_dataframe`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Preflight validation for Stage 12 source table` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `source_preflight_df = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS source_row_count, COUNT(*) FILTER (WHERE rebuild_is_complete = TRUE) AS complete_row_count, COUNT(*) FILTE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(source_preflight_df)` | Displays a notebook-facing result for inspection. |
| `source_row_count = dataframe_int_value(source_preflight_df, "source_row_count")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `complete_row_count = dataframe_int_value(source_preflight_df, "complete_row_count")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if source_row_count == 0: raise ValueError( "Stage 12 source table has zero rows for the active dataset/run. " "Run Stage 11 final alignment before Stage 12." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `if COMPLETE_ONLY_FLAG and complete_row_count == 0: raise ValueError( "Stage 12 is configured for complete_only=True, but no rows have " "rebuild_is_complete=True. Check Stage 11 ou` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Stage 12 source preflight passed.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 09 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Bronze`
- `BRONZE_HANDOFF_TARGET_TABLE_NAME`
- `COMPLETE_ONLY_FLAG`
- `DataFrame`
- `else`
- `FINAL_ALIGNMENT_SOURCE_TABLE_NAME`
- `handoff`
- `latest`
- `not_run`
- `result`
- `run`
- `Run`
- `run_bronze_handoff_loop`
- `RUN_HANDOFF`
- `RUN_MODE`
- `status`
- `STOP_ON_FAILURE_FLAG`

### Outputs

- `batch_size`
- `complete_only`
- `dataset_id`
- `engine`
- `latest_result`
- `max_iterations`
- `mode`
- `results`
- `run_id`
- `schema`
- `source_table`
- `stop_on_failure`
- `target_table`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Run Bronze handoff`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `if RUN_HANDOFF: results = run_bronze_handoff_loop( engine=engine, mode=RUN_MODE, batch_size=BATCH_SIZE, schema=SCHEMA, source_table=FINAL_ALIGNMENT_SOURCE_TABLE_NAME, target_table=`: Controls validation, iteration, file handling, or error handling for this step.
- `else: results = []`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `latest_result = results[-1] if results else {"status": "not_run"}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("Bronze handoff run mode:", RUN_MODE)`: Displays a notebook-facing result for inspection.
- `print("Bronze handoff latest result:")`: Displays a notebook-facing result for inspection.
- `print(latest_result)`: Displays a notebook-facing result for inspection.
- `display(pd.DataFrame(results))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `DataFrame`
- `display`
- `run_bronze_handoff_loop`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Run Bronze handoff` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `if RUN_HANDOFF: results = run_bronze_handoff_loop( engine=engine, mode=RUN_MODE, batch_size=BATCH_SIZE, schema=SCHEMA, source_table=FINAL_ALIGNMENT_SOURCE_TABLE_NAME, target_table=` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: results = []` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `latest_result = results[-1] if results else {"status": "not_run"}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Bronze handoff run mode:", RUN_MODE)` | Displays a notebook-facing result for inspection. |
| `print("Bronze handoff latest result:")` | Displays a notebook-facing result for inspection. |
| `print(latest_result)` | Displays a notebook-facing result for inspection. |
| `display(pd.DataFrame(results))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 10 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `after`
- `Bronze`
- `bronze_handoff_status`
- `BRONZE_HANDOFF_TARGET_TABLE_NAME`
- `BY`
- `COUNT`
- `dataframe_int_value`
- `DATASET_ID`
- `dataset_id`
- `DISTINCT`
- `distinct_observation_index_count`
- `engine`
- `f`
- `FILTER`
- `FINAL_ALIGNMENT_SOURCE_TABLE_NAME`
- `GROUP`
- `handoff`
- `has`
- `machine_status`
- `MAX`

### Outputs

- `null_machine_status_count`
- `null_timestamp_count`
- `params`
- `source_status_df`
- `target_row_count`
- `target_validation_df`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Post-handoff validation`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `target_validation_df = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS target_row_count, COUNT(DISTINCT observation_index) AS distinct_observation_index_count, MIN(observation_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `source_status_df = read_sql_dataframe( engine, f""" SELECT bronze_handoff_status, COUNT(*) AS row_count FROM "{SCHEMA}"."{FINAL_ALIGNMENT_SOURCE_TABLE_NAME}" WHERE dataset_id = :da`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(target_validation_df)`: Displays a notebook-facing result for inspection.
- `display(source_status_df)`: Displays a notebook-facing result for inspection.
- `target_row_count = dataframe_int_value(target_validation_df, "target_row_count")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `null_timestamp_count = dataframe_int_value(target_validation_df, "null_timestamp_count")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `null_machine_status_count = dataframe_int_value(target_validation_df, "null_machine_status_count")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `COUNT`
- `dataframe_int_value`
- `display`
- `FILTER`
- `MAX`
- `MIN`
- `read_sql_dataframe`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Post-handoff validation` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `target_validation_df = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS target_row_count, COUNT(DISTINCT observation_index) AS distinct_observation_index_count, MIN(observation_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `source_status_df = read_sql_dataframe( engine, f""" SELECT bronze_handoff_status, COUNT(*) AS row_count FROM "{SCHEMA}"."{FINAL_ALIGNMENT_SOURCE_TABLE_NAME}" WHERE dataset_id = :da` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(target_validation_df)` | Displays a notebook-facing result for inspection. |
| `display(source_status_df)` | Displays a notebook-facing result for inspection. |
| `target_row_count = dataframe_int_value(target_validation_df, "target_row_count")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `null_timestamp_count = dataframe_int_value(target_validation_df, "null_timestamp_count")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `null_machine_status_count = dataframe_int_value(target_validation_df, "null_machine_status_count")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if target_row_count == 0: raise ValueError("Bronze handoff target table has zero rows after Stage 12.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if null_timestamp_count > 0: raise ValueError( f"Bronze handoff target has {null_timestamp_count:,} null observation timestamps." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `if null_machine_status_count > 0: raise ValueError( f"Bronze handoff target has {null_machine_status_count:,} null machine_status values." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Stage 12 Bronze handoff validation passed.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 11 — Code Reference

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

