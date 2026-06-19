# Notebook Code Reference: synthetic_11_build_final_aligned_observations_stage

Notebook path:

`notebooks/synthetic/synthetic_11_build_final_aligned_observations_stage.ipynb`

## Notebook Purpose

This notebook documents and runs a focused support step in the synthetic pump telemetry workflow.

Notebook stage:

`Synthetic support`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Imports | Code Cell 01 |
| Notebook Configurables | Code Cell 02 |
| SQL Runtime Context | Code Cell 03 |
| Source Table Smoke Check | Code Cell 04 |
| Expected Row Count | Code Cell 05 |
| Build Full Final-Aligned Observation Stage | Code Cell 06 |
| Validate Full Final-Aligned Observation Stage | Code Cell 07 |
| Build Compact Final Synthetic Output | Code Cell 08 |
| Validate Compact Final Synthetic Output | Code Cell 09 |
| Status Distribution | Code Cell 10 |
| Sample Output | Code Cell 11 |
| Final Stage 11 Summary | Code Cell 12 |
| Cleanup | Code Cell 13 |

## Code Cell 01 — Imports

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
- `env_str`
- `final_aligned_observation_writer`
- `final_aligned_output_writer`
- `get_engine_from_env`
- `max_columns`
- `max_colwidth`
- `os`
- `pandas`
- `pipeline`
- `postgres`
- `read_sql_dataframe`
- `set_option`
- `synthetic`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.final_aligned_observation_writer import ( build_final_aligned_observations_stage,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.final_aligned_output_writer import ( build_synthetic_final_aligned_output_stage,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.core.env_helpers import ( env_bool, env_int, env_str,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `pd.set_option("display.max_colwidth", None)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `pd.set_option("display.max_columns", 120)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `import`
- `set_option`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.final_aligned_observation_writer import ( build_final_aligned_observations_stage,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.final_aligned_output_writer import ( build_synthetic_final_aligned_output_stage,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.env_helpers import ( env_bool, env_int, env_str,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `pd.set_option("display.max_colwidth", None)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `pd.set_option("display.max_columns", 120)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Notebook Configurables

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `aligned`
- `be`
- `both`
- `capstone`
- `CAPSTONE_SCHEMA`
- `compact`
- `complete`
- `config`
- `dataset`
- `env_bool`
- `env_int`
- `env_str`
- `exists`
- `f`
- `final`
- `FINAL_ALIGNMENT_SOURCE_TABLE_NAME`
- `Got`
- `id`
- `lower`
- `machine_status`

### Outputs

- `aliases`
- `COMPLETE_ONLY_FLAG`
- `DATASET_ID`
- `FINAL_ALIGN_OUTPUT_MODE`
- `FINAL_ALIGNED_STAGE_TABLE_NAME`
- `FINAL_OUTPUT_TABLE_NAME`
- `IF_EXISTS_FLAG`
- `MACHINE_STATUS_OUTPUT_COLUMN`
- `NUMBER_OF_SENSORS`
- `OBSERVATION_WINDOW_SIZE`
- `PREMELT_TABLE_NAME`
- `REBUILT_TABLE_NAME`
- `RUN_ID`
- `SCHEMA`
- `TIMESTAMP_OUTPUT_COLUMN`
- `VALID_OUTPUT_MODES`

### Key Operations

- `SCHEMA = env_str("CAPSTONE_SCHEMA", "capstone")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_ID = env_str( "SYNTHETIC_DATASET_ID", "pump_synthetic_v1", aliases=("DATASET_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `RUN_ID = env_str( "SYNTHETIC_RUN_ID", "synthetic_run_001", aliases=("RUN_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `NUMBER_OF_SENSORS = env_int("SYNTHETIC_SENSOR_COUNT", 52)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `FINAL_ALIGN_OUTPUT_MODE = env_str( "SYNTHETIC_STAGE_11_OUTPUT_MODE", "both",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `).strip().lower()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `VALID_OUTPUT_MODES = {"stage", "compact", "both"}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if FINAL_ALIGN_OUTPUT_MODE not in VALID_OUTPUT_MODES: raise ValueError( "SYNTHETIC_STAGE_11_OUTPUT_MODE must be one of " f"{sorted(VALID_OUTPUT_MODES)}. Got: {FINAL_ALIGN_OUTPUT_MO`: Controls validation, iteration, file handling, or error handling for this step.
- `IF_EXISTS_FLAG = env_str( "SYNTHETIC_FINAL_ALIGN_IF_EXISTS", "replace",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `).strip().lower()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `env_bool`
- `env_int`
- `env_str`
- `lower`
- `sorted`
- `strip`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `SCHEMA = env_str("CAPSTONE_SCHEMA", "capstone")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_ID = env_str( "SYNTHETIC_DATASET_ID", "pump_synthetic_v1", aliases=("DATASET_ID",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `RUN_ID = env_str( "SYNTHETIC_RUN_ID", "synthetic_run_001", aliases=("RUN_ID",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `NUMBER_OF_SENSORS = env_int("SYNTHETIC_SENSOR_COUNT", 52)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FINAL_ALIGN_OUTPUT_MODE = env_str( "SYNTHETIC_STAGE_11_OUTPUT_MODE", "both",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).strip().lower()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `VALID_OUTPUT_MODES = {"stage", "compact", "both"}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if FINAL_ALIGN_OUTPUT_MODE not in VALID_OUTPUT_MODES: raise ValueError( "SYNTHETIC_STAGE_11_OUTPUT_MODE must be one of " f"{sorted(VALID_OUTPUT_MODES)}. Got: {FINAL_ALIGN_OUTPUT_MO` | Controls validation, iteration, file handling, or error handling for this step. |
| `IF_EXISTS_FLAG = env_str( "SYNTHETIC_FINAL_ALIGN_IF_EXISTS", "replace",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).strip().lower()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `COMPLETE_ONLY_FLAG = env_bool( "SYNTHETIC_FINAL_ALIGN_COMPLETE_ONLY", True,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `OBSERVATION_WINDOW_SIZE = env_int( "SYNTHETIC_REBUILD_OBSERVATION_WINDOW_SIZE", 2500,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PREMELT_TABLE_NAME = env_str( "SYNTHETIC_PREMELT_OBSERVATIONS_TABLE", "synthetic_observations_premelt_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `REBUILT_TABLE_NAME = env_str( "SYNTHETIC_REBUILT_OBSERVATIONS_TABLE", "synthetic_sensor_observations_rebuilt_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FINAL_ALIGNED_STAGE_TABLE_NAME = env_str( "SYNTHETIC_FINAL_ALIGNED_OBSERVATIONS_TABLE", "synthetic_sensor_observations_final_aligned_stage", aliases=("FINAL_ALIGNED_STAGE_TABLE_NAM` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FINAL_OUTPUT_TABLE_NAME = env_str( "SYNTHETIC_FINAL_OUTPUT_TABLE", "synthetic_sensor_observations_final_output",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `TIMESTAMP_OUTPUT_COLUMN = env_str( "SYNTHETIC_FINAL_OUTPUT_TIMESTAMP_COLUMN", "timestamp",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `MACHINE_STATUS_OUTPUT_COLUMN = env_str( "SYNTHETIC_FINAL_OUTPUT_STATUS_COLUMN", "machine_status",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Synthetic Stage 11 config")` | Displays a notebook-facing result for inspection. |
| `print("schema:", SCHEMA)` | Displays a notebook-facing result for inspection. |
| `print("dataset id:", DATASET_ID)` | Displays a notebook-facing result for inspection. |
| `print("run id:", RUN_ID)` | Displays a notebook-facing result for inspection. |
| `print("output mode:", FINAL_ALIGN_OUTPUT_MODE)` | Displays a notebook-facing result for inspection. |
| `print("premelt table:", PREMELT_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("rebuilt table:", REBUILT_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("final-aligned stage table:", FINAL_ALIGNED_STAGE_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("compact final output table:", FINAL_OUTPUT_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("if exists:", IF_EXISTS_FLAG)` | Displays a notebook-facing result for inspection. |
| `print("complete only:", COMPLETE_ONLY_FLAG)` | Displays a notebook-facing result for inspection. |
| `print("observation window size:", OBSERVATION_WINDOW_SIZE)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 03 — SQL Runtime Context

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

## Code Cell 04 — Source Table Smoke Check

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `engine`
- `f`
- `information_schema`
- `Missing`
- `ORDER`
- `premelt_table`
- `PREMELT_TABLE_NAME`
- `raise`
- `read_sql_dataframe`
- `rebuilt_table`
- `REBUILT_TABLE_NAME`
- `RuntimeError`
- `s`
- `SCHEMA`
- `schema_name`
- `SELECT`
- `sorted`
- `source`
- `Stage`

### Outputs

- `found_source_tables`
- `missing_source_tables`
- `params`
- `required_source_tables`
- `source_table_check_dataframe`

### Key Operations

- `source_table_check_dataframe = read_sql_dataframe( engine, """ SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema = :schema_name AND table_name IN (:`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(source_table_check_dataframe)`: Displays a notebook-facing result for inspection.
- `found_source_tables = set(source_table_check_dataframe["table_name"].tolist())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `required_source_tables = {PREMELT_TABLE_NAME, REBUILT_TABLE_NAME}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `missing_source_tables = sorted(required_source_tables - found_source_tables)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if missing_source_tables: raise RuntimeError(f"Missing Stage 11 source table(s): {missing_source_tables}")`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `display`
- `IN`
- `read_sql_dataframe`
- `RuntimeError`
- `sorted`
- `table`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `source_table_check_dataframe = read_sql_dataframe( engine, """ SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema = :schema_name AND table_name IN (:` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(source_table_check_dataframe)` | Displays a notebook-facing result for inspection. |
| `found_source_tables = set(source_table_check_dataframe["table_name"].tolist())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `required_source_tables = {PREMELT_TABLE_NAME, REBUILT_TABLE_NAME}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `missing_source_tables = sorted(required_source_tables - found_source_tables)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if missing_source_tables: raise RuntimeError(f"Missing Stage 11 source table(s): {missing_source_tables}")` | Controls validation, iteration, file handling, or error handling for this step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 05 — Expected Row Count

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `active`
- `COMPLETE_ONLY_FLAG`
- `Confirm`
- `count`
- `COUNT`
- `DATASET_ID`
- `dataset_id`
- `DISTINCT`
- `else`
- `engine`
- `exist`
- `Expected`
- `f`
- `FILTER`
- `iloc`
- `MAX`
- `max_observation_index`
- `MIN`
- `min_observation_index`
- `NULL`

### Outputs

- `expected_counts`
- `expected_counts_dataframe`
- `EXPECTED_ROW_COUNT`
- `params`

### Key Operations

- `expected_counts_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS rebuilt_row_count, COUNT(*) FILTER (WHERE rebuild_is_complete = TRUE) AS rebuilt_complete_count, COU`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(expected_counts_dataframe)`: Displays a notebook-facing result for inspection.
- `expected_counts = expected_counts_dataframe.iloc[0].to_dict()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `EXPECTED_ROW_COUNT = int( expected_counts["rebuilt_complete_count"] if COMPLETE_ONLY_FLAG else expected_counts["rebuilt_row_count"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if EXPECTED_ROW_COUNT <= 0: raise RuntimeError( "Expected row count is zero. Confirm Stage 09/10 rebuilt observations " "exist for the active dataset_id and run_id." )`: Controls validation, iteration, file handling, or error handling for this step.
- `print("Expected Stage 11 row count:", EXPECTED_ROW_COUNT)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `MAX`
- `MIN`
- `read_sql_dataframe`
- `RuntimeError`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `expected_counts_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS rebuilt_row_count, COUNT(*) FILTER (WHERE rebuild_is_complete = TRUE) AS rebuilt_complete_count, COU` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(expected_counts_dataframe)` | Displays a notebook-facing result for inspection. |
| `expected_counts = expected_counts_dataframe.iloc[0].to_dict()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `EXPECTED_ROW_COUNT = int( expected_counts["rebuilt_complete_count"] if COMPLETE_ONLY_FLAG else expected_counts["rebuilt_row_count"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if EXPECTED_ROW_COUNT <= 0: raise RuntimeError( "Expected row count is zero. Confirm Stage 09/10 rebuilt observations " "exist for the active dataset_id and run_id." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Expected Stage 11 row count:", EXPECTED_ROW_COUNT)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 06 — Build Full Final-Aligned Observation Stage

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `aligned`
- `both`
- `build`
- `build_final_aligned_observations_stage`
- `COMPLETE_ONLY_FLAG`
- `DataFrame`
- `else`
- `final`
- `FINAL_ALIGN_OUTPUT_MODE`
- `FINAL_ALIGNED_STAGE_TABLE_NAME`
- `full`
- `IF_EXISTS_FLAG`
- `NUMBER_OF_SENSORS`
- `observation`
- `PREMELT_TABLE_NAME`
- `REBUILT_TABLE_NAME`
- `Skipped`
- `stage`

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

- `stage_11a_result = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if FINAL_ALIGN_OUTPUT_MODE in {"stage", "both"}: stage_11a_result = build_final_aligned_observations_stage( engine=engine, schema=SCHEMA, premelt_table=PREMELT_TABLE_NAME, rebuilt_`: Displays a notebook-facing result for inspection.
- `else: print("Skipped full final-aligned observation stage build.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_final_aligned_observations_stage`
- `DataFrame`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_11a_result = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if FINAL_ALIGN_OUTPUT_MODE in {"stage", "both"}: stage_11a_result = build_final_aligned_observations_stage( engine=engine, schema=SCHEMA, premelt_table=PREMELT_TABLE_NAME, rebuilt_` | Displays a notebook-facing result for inspection. |
| `else: print("Skipped full final-aligned observation stage build.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 07 — Validate Full Final-Aligned Observation Stage

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `aligned`
- `both`
- `COUNT`
- `DATASET_ID`
- `dataset_id`
- `DISTINCT`
- `distinct_generated_row_id_count`
- `distinct_observation_count`
- `else`
- `engine`
- `EXPECTED_ROW_COUNT`
- `f`
- `FILTER`
- `final`
- `FINAL_ALIGN_OUTPUT_MODE`
- `FINAL_ALIGNED_STAGE_TABLE_NAME`
- `final_row_count`
- `full`
- `generated_row_id`
- `MAX`

### Outputs

- `params`
- `stage_11a_validation_dataframe`

### Key Operations

- `stage_11a_validation_dataframe = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if FINAL_ALIGN_OUTPUT_MODE in {"stage", "both"}: stage_11a_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS final_row_count, COUNT(DISTINCT observation_in`: Loads input data, configuration, or artifacts required by the current stage.
- `else: print("Skipped full final-aligned observation stage validation.")`: Displays a notebook-facing result for inspection.

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
| `stage_11a_validation_dataframe = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if FINAL_ALIGN_OUTPUT_MODE in {"stage", "both"}: stage_11a_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS final_row_count, COUNT(DISTINCT observation_in` | Loads input data, configuration, or artifacts required by the current stage. |
| `else: print("Skipped full final-aligned observation stage validation.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 08 — Build Compact Final Synthetic Output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `both`
- `build`
- `build_synthetic_final_aligned_output_stage`
- `compact`
- `COMPLETE_ONLY_FLAG`
- `DataFrame`
- `else`
- `final`
- `FINAL_ALIGN_OUTPUT_MODE`
- `FINAL_OUTPUT_TABLE_NAME`
- `IF_EXISTS_FLAG`
- `NUMBER_OF_SENSORS`
- `output`
- `REBUILT_TABLE_NAME`
- `Skipped`
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

- `stage_11b_result = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if FINAL_ALIGN_OUTPUT_MODE in {"compact", "both"}: stage_11b_result = build_synthetic_final_aligned_output_stage( engine=engine, schema=SCHEMA, rebuilt_table=REBUILT_TABLE_NAME, ta`: Displays a notebook-facing result for inspection.
- `else: print("Skipped compact final synthetic output build.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_synthetic_final_aligned_output_stage`
- `DataFrame`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_11b_result = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if FINAL_ALIGN_OUTPUT_MODE in {"compact", "both"}: stage_11b_result = build_synthetic_final_aligned_output_stage( engine=engine, schema=SCHEMA, rebuilt_table=REBUILT_TABLE_NAME, ta` | Displays a notebook-facing result for inspection. |
| `else: print("Skipped compact final synthetic output build.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 09 — Validate Compact Final Synthetic Output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `asset_id`
- `asset_id_count`
- `both`
- `BROKEN`
- `broken_rows`
- `compact`
- `COUNT`
- `dataset_id`
- `DATASET_ID`
- `dataset_id_count`
- `DISTINCT`
- `else`
- `engine`
- `EXPECTED_ROW_COUNT`
- `f`
- `FILTER`
- `final`
- `FINAL_ALIGN_OUTPUT_MODE`
- `FINAL_OUTPUT_TABLE_NAME`
- `MACHINE_STATUS_OUTPUT_COLUMN`

### Outputs

- `params`
- `stage_11b_validation_dataframe`

### Key Operations

- `stage_11b_validation_dataframe = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if FINAL_ALIGN_OUTPUT_MODE in {"compact", "both"}: stage_11b_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS row_count, COUNT(DISTINCT dataset_id) AS dat`: Loads input data, configuration, or artifacts required by the current stage.
- `else: print("Skipped compact final synthetic output validation.")`: Displays a notebook-facing result for inspection.

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
| `stage_11b_validation_dataframe = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if FINAL_ALIGN_OUTPUT_MODE in {"compact", "both"}: stage_11b_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS row_count, COUNT(DISTINCT dataset_id) AS dat` | Loads input data, configuration, or artifacts required by the current stage. |
| `else: print("Skipped compact final synthetic output validation.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 10 — Status Distribution

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `aligned`
- `both`
- `BY`
- `compact`
- `Compact`
- `COUNT`
- `dataset_id`
- `DATASET_ID`
- `distribution`
- `engine`
- `f`
- `final`
- `FINAL_ALIGN_OUTPUT_MODE`
- `FINAL_ALIGNED_STAGE_TABLE_NAME`
- `FINAL_OUTPUT_TABLE_NAME`
- `Full`
- `GROUP`
- `machine_status`
- `MACHINE_STATUS_OUTPUT_COLUMN`
- `ORDER`

### Outputs

- `params`
- `stage_11a_status_distribution_dataframe`
- `stage_11b_status_distribution_dataframe`

### Key Operations

- `if FINAL_ALIGN_OUTPUT_MODE in {"stage", "both"}: stage_11a_status_distribution_dataframe = read_sql_dataframe( engine, f""" SELECT stream_state, phase, COUNT(*) AS row_count FROM "`: Loads input data, configuration, or artifacts required by the current stage.
- `if FINAL_ALIGN_OUTPUT_MODE in {"compact", "both"}: stage_11b_status_distribution_dataframe = read_sql_dataframe( engine, f""" SELECT "{MACHINE_STATUS_OUTPUT_COLUMN}" AS machine_sta`: Loads input data, configuration, or artifacts required by the current stage.

Important functions or methods detected:
- `COUNT`
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `if FINAL_ALIGN_OUTPUT_MODE in {"stage", "both"}: stage_11a_status_distribution_dataframe = read_sql_dataframe( engine, f""" SELECT stream_state, phase, COUNT(*) AS row_count FROM "` | Loads input data, configuration, or artifacts required by the current stage. |
| `if FINAL_ALIGN_OUTPUT_MODE in {"compact", "both"}: stage_11b_status_distribution_dataframe = read_sql_dataframe( engine, f""" SELECT "{MACHINE_STATUS_OUTPUT_COLUMN}" AS machine_sta` | Loads input data, configuration, or artifacts required by the current stage. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 11 — Sample Output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `aligned`
- `asset_id`
- `both`
- `BY`
- `compact`
- `Compact`
- `dataset_id`
- `DATASET_ID`
- `engine`
- `f`
- `final`
- `FINAL_ALIGN_OUTPUT_MODE`
- `FINAL_ALIGNED_STAGE_TABLE_NAME`
- `FINAL_OUTPUT_TABLE_NAME`
- `Full`
- `generated_row_id`
- `LIMIT`
- `MACHINE_STATUS_OUTPUT_COLUMN`
- `observation_index`
- `observation_timestamp`

### Outputs

- `params`
- `stage_11a_sample_dataframe`
- `stage_11b_sample_dataframe`

### Key Operations

- `if FINAL_ALIGN_OUTPUT_MODE in {"stage", "both"}: stage_11a_sample_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, generated_row_id, observation_in`: Loads input data, configuration, or artifacts required by the current stage.
- `if FINAL_ALIGN_OUTPUT_MODE in {"compact", "both"}: stage_11b_sample_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, "{TIMESTAMP_OUTPUT_COLUMN}", s`: Loads input data, configuration, or artifacts required by the current stage.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `if FINAL_ALIGN_OUTPUT_MODE in {"stage", "both"}: stage_11a_sample_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, generated_row_id, observation_in` | Loads input data, configuration, or artifacts required by the current stage. |
| `if FINAL_ALIGN_OUTPUT_MODE in {"compact", "both"}: stage_11b_sample_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, "{TIMESTAMP_OUTPUT_COLUMN}", s` | Loads input data, configuration, or artifacts required by the current stage. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 12 — Final Stage 11 Summary

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `actual_rows`
- `all`
- `append`
- `bool`
- `but`
- `compact_final_output`
- `complete`
- `completed`
- `DataFrame`
- `empty`
- `EXPECTED_ROW_COUNT`
- `expected_rows`
- `failed`
- `FINAL_ALIGNED_STAGE_TABLE_NAME`
- `FINAL_OUTPUT_TABLE_NAME`
- `final_row_count`
- `full_final_aligned_stage`
- `iloc`
- `more`
- `one`

### Outputs

- `compact_ready`
- `stage_11_summary_dataframe`
- `stage_ready`
- `summary_rows`

### Key Operations

- `summary_rows = []`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if stage_11a_result is not None: stage_ready = bool(stage_11a_validation_dataframe["ready_for_bronze_handoff_check"].iloc[0]) summary_rows.append( { "output": "full_final_aligned_s`: Controls validation, iteration, file handling, or error handling for this step.
- `if stage_11b_result is not None: compact_ready = bool(stage_11b_validation_dataframe["ready_for_export"].iloc[0]) summary_rows.append( { "output": "compact_final_output", "table_na`: Controls validation, iteration, file handling, or error handling for this step.
- `stage_11_summary_dataframe = pd.DataFrame(summary_rows)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `display(stage_11_summary_dataframe)`: Displays a notebook-facing result for inspection.
- `if not stage_11_summary_dataframe.empty and not stage_11_summary_dataframe["ready"].all(): raise RuntimeError("Stage 11 completed, but one or more outputs failed validation.")`: Controls validation, iteration, file handling, or error handling for this step.
- `print("Stage 11 complete.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `all`
- `append`
- `bool`
- `DataFrame`
- `display`
- `RuntimeError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `summary_rows = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if stage_11a_result is not None: stage_ready = bool(stage_11a_validation_dataframe["ready_for_bronze_handoff_check"].iloc[0]) summary_rows.append( { "output": "full_final_aligned_s` | Controls validation, iteration, file handling, or error handling for this step. |
| `if stage_11b_result is not None: compact_ready = bool(stage_11b_validation_dataframe["ready_for_export"].iloc[0]) summary_rows.append( { "output": "compact_final_output", "table_na` | Controls validation, iteration, file handling, or error handling for this step. |
| `stage_11_summary_dataframe = pd.DataFrame(summary_rows)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `display(stage_11_summary_dataframe)` | Displays a notebook-facing result for inspection. |
| `if not stage_11_summary_dataframe.empty and not stage_11_summary_dataframe["ready"].all(): raise RuntimeError("Stage 11 completed, but one or more outputs failed validation.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Stage 11 complete.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 13 — Cleanup

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dispose`
- `engine`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `e`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `dispose`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `e` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

