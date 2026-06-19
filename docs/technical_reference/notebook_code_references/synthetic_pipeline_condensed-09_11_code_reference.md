# Notebook Code Reference: synthetic_pipeline_condensed-09_11

Notebook path:

`notebooks/synthetic/synthetic_pipeline_condensed-09_11.ipynb`

## Notebook Purpose

This notebook documents and runs a focused support step in the synthetic pump telemetry workflow.

Notebook stage:

`Synthetic support`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08, Code Cell 09, Code Cell 10, Code Cell 11, Code Cell 12, Code Cell 13, Code Cell 14, Code Cell 15 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_synthetic_final_aligned_output_stage`
- `database`
- `final_aligned_incremental`
- `final_aligned_output_writer`
- `get_engine_from_env`
- `os`
- `pandas`
- `pipeline`
- `postgres`
- `read_sql_dataframe`
- `rebuild_consumed_messages_to_observations`
- `row_rebuilder`
- `run_final_align_loop`
- `synthetic`
- `utils`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.row_rebuilder import ( rebuild_consumed_messages_to_observations,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.final_aligned_output_writer import build_synthetic_final_aligned_output_stage`: Imports a dependency or project helper used by later cells.
- `from utils.synthetic.pipeline.final_aligned_incremental import run_final_align_loop`: Imports a dependency or project helper used by later cells.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.row_rebuilder import ( rebuild_consumed_messages_to_observations,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.final_aligned_output_writer import build_synthetic_final_aligned_output_stage` | Imports a dependency or project helper used by later cells. |
| `from utils.synthetic.pipeline.final_aligned_incremental import run_final_align_loop` | Imports a dependency or project helper used by later cells. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `broken`
- `BROKEN`
- `CAPSTONE_SCHEMA`
- `created_at`
- `currently`
- `drain`
- `everything`
- `failed`
- `failure`
- `fault`
- `getenv`
- `machine_status`
- `NORMAL`
- `normal`
- `observation_timestamp`
- `os`
- `pending`
- `phase`
- `pump_asset_001`

### Outputs

- `ASSET_ID`
- `BATCH_SIZE`
- `COMPLETE_ONLY_FLAG`
- `DATASET_ID`
- `IF_EXISTS_FLAG`
- `MARK_SOURCE_REBUILT_FLAG`
- `MAX_ITERATIONS`
- `NUMBER_OF_SENSORS`
- `OBSERVATION_WINDOW_SIZE`
- `PREMELT_TABLE_NAME`
- `REBUILD_CONSUMED_MESSAGES_SOURCE_TABLE_NAME`
- `REBUILD_STATUS`
- `REBUILT_CONSUMED_MESSAGES_DESTINATION_TABLE_NAME`
- `REBUILT_TABLE_NAME`
- `RUN_ID`
- `SCHEMA`
- `STATUS_MAPPING`
- `STATUS_SOURCE_PRIORITY`
- `STOP_ON_FAILURE`
- `TARGET_TABLE_NAME`

### Key Operations

- `SCHEMA = os.getenv("CAPSTONE_SCHEMA", "synthetic_run_001")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_ID = os.getenv("SYNTHETIC_DATASET_ID", "pump_synthetic_v1")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RUN_ID = os.getenv("SYNTHETIC_RUN_ID", "synthetic_run_001")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ASSET_ID = os.getenv("SYNTHETIC_ASSET_ID", "pump_asset_001")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `NUMBER_OF_SENSORS = int(52)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `IF_EXISTS_FLAG = str("replace")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `COMPLETE_ONLY_FLAG = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `BATCH_SIZE = 1000`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `MAX_ITERATIONS = None # None = drain everything currently pending`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `STOP_ON_FAILURE = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `OBSERVATION_WINDOW_SIZE = int(2500)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `REBUILD_STATUS = str("pending")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `getenv`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `SCHEMA = os.getenv("CAPSTONE_SCHEMA", "synthetic_run_001")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_ID = os.getenv("SYNTHETIC_DATASET_ID", "pump_synthetic_v1")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_ID = os.getenv("SYNTHETIC_RUN_ID", "synthetic_run_001")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ASSET_ID = os.getenv("SYNTHETIC_ASSET_ID", "pump_asset_001")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `NUMBER_OF_SENSORS = int(52)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `IF_EXISTS_FLAG = str("replace")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `COMPLETE_ONLY_FLAG = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BATCH_SIZE = 1000` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MAX_ITERATIONS = None # None = drain everything currently pending` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STOP_ON_FAILURE = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `OBSERVATION_WINDOW_SIZE = int(2500)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `REBUILD_STATUS = str("pending")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MARK_SOURCE_REBUILT_FLAG = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `REBUILD_CONSUMED_MESSAGES_SOURCE_TABLE_NAME = str("synthetic_sensor_messages_consumed_stage")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `REBUILT_CONSUMED_MESSAGES_DESTINATION_TABLE_NAME = str("synthetic_sensor_observations_rebuilt_stage")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PREMELT_TABLE_NAME = "synthetic_observations_premelt_stage"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `REBUILT_TABLE_NAME = "synthetic_sensor_observations_rebuilt_stage"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TARGET_TABLE_NAME = "synthetic_sensor_observations_final_aligned_stage"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TIMESTAMP_SOURCE_PRIORITY = ( "observation_timestamp", "timestamp", "created_at",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STATUS_SOURCE_PRIORITY = ( "machine_status", "stream_state", "phase",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STATUS_MAPPING = { "normal": "NORMAL", "broken": "BROKEN", "abnormal": "BROKEN", "failure": "BROKEN", "failed": "BROKEN", "fault": "BROKEN", "recovering": "RECOVERING", "recovery":` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

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

- `COMPLETE_ONLY_FLAG`
- `MARK_SOURCE_REBUILT_FLAG`
- `NUMBER_OF_SENSORS`
- `REBUILD_CONSUMED_MESSAGES_SOURCE_TABLE_NAME`
- `rebuild_consumed_messages_to_observations`
- `REBUILT_CONSUMED_MESSAGES_DESTINATION_TABLE_NAME`

### Outputs

- `complete_only`
- `dataset_id`
- `engine`
- `mark_source_rebuilt`
- `n_sensors`
- `observation_window_size`
- `rebuild_result`
- `rebuild_status`
- `run_id`
- `schema`
- `source_table`
- `target_table`

### Key Operations

- `rebuild_result = rebuild_consumed_messages_to_observations( engine=engine, schema=SCHEMA, source_table=REBUILD_CONSUMED_MESSAGES_SOURCE_TABLE_NAME, target_table=REBUILT_CONSUMED_ME`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(rebuild_result)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `rebuild_consumed_messages_to_observations`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `rebuild_result = rebuild_consumed_messages_to_observations( engine=engine, schema=SCHEMA, source_table=REBUILD_CONSUMED_MESSAGES_SOURCE_TABLE_NAME, target_table=REBUILT_CONSUMED_ME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(rebuild_result)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 05 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `capstone`
- `complete_row_count`
- `COUNT`
- `engine`
- `FILTER`
- `MAX`
- `max_observation_index`
- `max_observation_timestamp`
- `MIN`
- `min_observation_index`
- `min_observation_timestamp`
- `observation_index`
- `observation_timestamp`
- `read_sql_dataframe`
- `rebuild_is_complete`
- `rebuilt_row_count`
- `SELECT`
- `synthetic_sensor_observations_rebuilt_stage`
- `WHERE`

### Outputs

- `validation_dataframe`

### Key Operations

- `validation_dataframe = read_sql_dataframe( engine, """ SELECT COUNT(*) AS rebuilt_row_count, COUNT(*) FILTER (WHERE rebuild_is_complete = TRUE) AS complete_row_count, MIN(observati`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(validation_dataframe)`: Displays a notebook-facing result for inspection.

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
| `validation_dataframe = read_sql_dataframe( engine, """ SELECT COUNT(*) AS rebuilt_row_count, COUNT(*) FILTER (WHERE rebuild_is_complete = TRUE) AS complete_row_count, MIN(observati` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 06 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `asset_id`
- `BY`
- `capstone`
- `dataset_id`
- `engine`
- `LIMIT`
- `meta_episode_id`
- `meta_primary_fault_type`
- `observation_index`
- `observation_timestamp`
- `ORDER`
- `phase`
- `read_sql_dataframe`
- `rebuild_is_complete`
- `rebuild_sensor_count`
- `run_id`
- `SELECT`
- `sensor_00`
- `sensor_01`
- `sensor_02`

### Outputs

- `sample_dataframe`

### Key Operations

- `sample_dataframe = read_sql_dataframe( engine, """ SELECT dataset_id, run_id, asset_id, observation_index, observation_timestamp, stream_state, phase, meta_episode_id, meta_primary`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(sample_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `sample_dataframe = read_sql_dataframe( engine, """ SELECT dataset_id, run_id, asset_id, observation_index, observation_timestamp, stream_state, phase, meta_episode_id, meta_primary` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(sample_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 07 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `asset_id`
- `BY`
- `capstone`
- `dataset_id`
- `engine`
- `LIMIT`
- `observation_index`
- `ORDER`
- `read_sql_dataframe`
- `rebuild_is_complete`
- `rebuild_notes`
- `rebuild_sensor_count`
- `run_id`
- `SELECT`
- `synthetic_sensor_observations_rebuilt_stage`
- `WHERE`

### Outputs

- `incomplete_dataframe`

### Key Operations

- `incomplete_dataframe = read_sql_dataframe( engine, """ SELECT dataset_id, run_id, asset_id, observation_index, rebuild_sensor_count, rebuild_is_complete, rebuild_notes FROM capston`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(incomplete_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `incomplete_dataframe = read_sql_dataframe( engine, """ SELECT dataset_id, run_id, asset_id, observation_index, rebuild_sensor_count, rebuild_is_complete, rebuild_notes FROM capston` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(incomplete_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 08 — Code Reference

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

## Code Cell 09 — Code Reference

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

## Code Cell 10 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_synthetic_final_aligned_output_stage`
- `COMPLETE_ONLY_FLAG`
- `Full`
- `IF_EXISTS_FLAG`
- `machine_status`
- `NUMBER_OF_SENSORS`
- `REBUILT_TABLE_NAME`
- `Run`
- `TARGET_TABLE_NAME`
- `timestamp`

### Outputs

- `complete_only`
- `dataset_id`
- `engine`
- `final_output_result`
- `if_exists`
- `machine_status_output_column`
- `n_sensors`
- `observation_window_size`
- `rebuilt_table`
- `run_id`
- `schema`
- `status_mapping`
- `status_source_priority`
- `target_table`
- `timestamp_output_column`
- `timestamp_source_priority`

### Key Operations

- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Full Run`: Documents the purpose or boundary of the surrounding notebook step.
- `final_output_result = build_synthetic_final_aligned_output_stage( engine=engine, schema=SCHEMA, rebuilt_table=REBUILT_TABLE_NAME, target_table=TARGET_TABLE_NAME, dataset_id=DATASET`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(final_output_result)`: Displays a notebook-facing result for inspection.
- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `build_synthetic_final_aligned_output_stage`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Full Run` | Documents the purpose or boundary of the surrounding notebook step. |
| `final_output_result = build_synthetic_final_aligned_output_stage( engine=engine, schema=SCHEMA, rebuilt_table=REBUILT_TABLE_NAME, target_table=TARGET_TABLE_NAME, dataset_id=DATASET` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(final_output_result)` | Displays a notebook-facing result for inspection. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 11 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Batched`
- `COMPLETE_ONLY_FLAG`
- `DataFrame`
- `Incremental`
- `NUMBER_OF_SENSORS`
- `PREMELT_TABLE_NAME`
- `REBUILT_TABLE_NAME`
- `Run`
- `run_final_align_loop`
- `TARGET_TABLE_NAME`

### Outputs

- `batch_size`
- `complete_only`
- `dataset_id`
- `engine`
- `final_align_results`
- `max_iterations`
- `n_sensors`
- `prefer_rebuilt_sensor_values`
- `premelt_table`
- `rebuilt_table`
- `run_id`
- `schema`
- `stop_on_failure`
- `target_table`

### Key Operations

- `# Batched Incremental Run`: Documents the purpose or boundary of the surrounding notebook step.
- `final_align_results = run_final_align_loop( engine=engine, batch_size=BATCH_SIZE, schema=SCHEMA, premelt_table=PREMELT_TABLE_NAME, rebuilt_table=REBUILT_TABLE_NAME, target_table=TA`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(pd.DataFrame(final_align_results))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `DataFrame`
- `display`
- `run_final_align_loop`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Batched Incremental Run` | Documents the purpose or boundary of the surrounding notebook step. |
| `final_align_results = run_final_align_loop( engine=engine, batch_size=BATCH_SIZE, schema=SCHEMA, premelt_table=PREMELT_TABLE_NAME, rebuilt_table=REBUILT_TABLE_NAME, target_table=TA` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(pd.DataFrame(final_align_results))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 12 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `asset_id`
- `asset_id_count`
- `BROKEN`
- `broken_rows`
- `COUNT`
- `dataset_id`
- `dataset_id_count`
- `DISTINCT`
- `engine`
- `f`
- `FILTER`
- `machine_status`
- `MAX`
- `max_timestamp`
- `MIN`
- `min_timestamp`
- `NORMAL`
- `normal_rows`
- `read_sql_dataframe`
- `RECOVERING`

### Outputs

- `validation_dataframe`

### Key Operations

- `validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS row_count, COUNT(DISTINCT dataset_id) AS dataset_id_count, COUNT(DISTINCT run_id) AS run_id_count, COUNT(`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(validation_dataframe)`: Displays a notebook-facing result for inspection.

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
| `validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS row_count, COUNT(DISTINCT dataset_id) AS dataset_id_count, COUNT(DISTINCT run_id) AS run_id_count, COUNT(` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 13 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `asset_id`
- `BY`
- `dataset_id`
- `engine`
- `f`
- `LIMIT`
- `machine_status`
- `ORDER`
- `read_sql_dataframe`
- `run_id`
- `SCHEMA`
- `SELECT`
- `sensor_00`
- `sensor_01`
- `sensor_02`
- `sensor_03`
- `sensor_04`
- `TARGET_TABLE_NAME`
- `timestamp`

### Outputs

- `sample_dataframe`

### Key Operations

- `sample_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, timestamp, sensor_00, sensor_01, sensor_02, sensor_03, sensor_04, machine_status FROM {SCHE`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(sample_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `sample_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, timestamp, sensor_00, sensor_01, sensor_02, sensor_03, sensor_04, machine_status FROM {SCHE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(sample_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 14 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `COUNT`
- `engine`
- `f`
- `GROUP`
- `machine_status`
- `ORDER`
- `read_sql_dataframe`
- `row_count`
- `SCHEMA`
- `SELECT`
- `TARGET_TABLE_NAME`

### Outputs

- `status_distribution_dataframe`

### Key Operations

- `status_distribution_dataframe = read_sql_dataframe( engine, f""" SELECT machine_status, COUNT(*) AS row_count FROM {SCHEMA}.{TARGET_TABLE_NAME} GROUP BY machine_status ORDER BY mac`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(status_distribution_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `status_distribution_dataframe = read_sql_dataframe( engine, f""" SELECT machine_status, COUNT(*) AS row_count FROM {SCHEMA}.{TARGET_TABLE_NAME} GROUP BY machine_status ORDER BY mac` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(status_distribution_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 15 — Code Reference

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

