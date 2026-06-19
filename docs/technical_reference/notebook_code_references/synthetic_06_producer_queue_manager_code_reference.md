# Notebook Code Reference: synthetic_06_producer_queue_manager

Notebook path:

`notebooks/synthetic/synthetic_06_producer_queue_manager.ipynb`

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

- `core`
- `database`
- `ensure_send_queue_runtime_columns`
- `ensure_simulation_state_control_table`
- `env_float`
- `env_helpers`
- `env_int`
- `env_str`
- `get_engine_from_env`
- `get_send_queue_status_counts`
- `os`
- `pipeline`
- `postgres`
- `producer_queue_manager`
- `read_simulation_state_control`
- `read_sql_dataframe`
- `synthetic`
- `upsert_simulation_state_control`
- `utils`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.producer_queue_manager import ( ensure_send_queue_runtime_columns, ensure_simulation_state_control_table, upsert_simulation_state_control, read_simula`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.core.env_helpers import ( env_float, env_int, env_str,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.producer_queue_manager import ( ensure_send_queue_runtime_columns, ensure_simulation_state_control_table, upsert_simulation_state_control, read_simula` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.env_helpers import ( env_float, env_int, env_str,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `apply`
- `batch`
- `capstone`
- `CAPSTONE_SCHEMA`
- `config`
- `dataset`
- `env_float`
- `env_int`
- `env_str`
- `grants`
- `id`
- `kafka_producer`
- `KAFKA_TOPIC`
- `manager`
- `observation`
- `owner`
- `Producer`
- `producer`
- `PRODUCER_CONTROL_TABLE`
- `PRODUCER_QUEUE_TABLE`

### Outputs

- `aliases`
- `APPLY_OWNER_AND_GRANTS_FLAG`
- `CONTROL_OWNER_ROLE`
- `DATASET_ID`
- `OBSERVATION_BATCH_SIZE`
- `PRODUCER_MAX_SEND_ATTEMPTS`
- `PRODUCER_POLL_SECONDS`
- `PRODUCER_TOPIC`
- `PRODUCER_WORKER_ID`
- `RUN_ID`
- `SCHEMA`
- `SEND_QUEUE_TABLE_NAME`
- `SIMULATION_TABLE_NAME`

### Key Operations

- `SCHEMA = env_str("CAPSTONE_SCHEMA", "capstone")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_ID = env_str( "SYNTHETIC_DATASET_ID", "pump_synthetic_v1", aliases=("DATASET_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `RUN_ID = env_str( "SYNTHETIC_RUN_ID", "synthetic_run_001", aliases=("RUN_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `SIMULATION_TABLE_NAME = env_str( "SYNTHETIC_CONTROL_TABLE", "simulation_state_control", aliases=("PRODUCER_CONTROL_TABLE",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `SEND_QUEUE_TABLE_NAME = env_str( "SYNTHETIC_SEND_QUEUE_TABLE", "synthetic_sensor_messages_send_queue", aliases=("PRODUCER_QUEUE_TABLE",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `PRODUCER_TOPIC = env_str( "SYNTHETIC_KAFKA_TOPIC", "pump.telemetry.synthetic", aliases=("KAFKA_TOPIC",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `PRODUCER_WORKER_ID = env_str( "SYNTHETIC_PRODUCER_WORKER_ID", "producer_worker_test_001",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

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
| `SIMULATION_TABLE_NAME = env_str( "SYNTHETIC_CONTROL_TABLE", "simulation_state_control", aliases=("PRODUCER_CONTROL_TABLE",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SEND_QUEUE_TABLE_NAME = env_str( "SYNTHETIC_SEND_QUEUE_TABLE", "synthetic_sensor_messages_send_queue", aliases=("PRODUCER_QUEUE_TABLE",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PRODUCER_TOPIC = env_str( "SYNTHETIC_KAFKA_TOPIC", "pump.telemetry.synthetic", aliases=("KAFKA_TOPIC",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PRODUCER_WORKER_ID = env_str( "SYNTHETIC_PRODUCER_WORKER_ID", "producer_worker_test_001",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `OBSERVATION_BATCH_SIZE = env_int("OBSERVATION_BATCH_SIZE", 500)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PRODUCER_POLL_SECONDS = env_float("PRODUCER_POLL_SECONDS", 0.0)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PRODUCER_MAX_SEND_ATTEMPTS = env_int("PRODUCER_MAX_SEND_ATTEMPTS", 3)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTROL_OWNER_ROLE = "kafka_producer"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `APPLY_OWNER_AND_GRANTS_FLAG = False` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Producer queue manager config")` | Displays a notebook-facing result for inspection. |
| `print("schema:", SCHEMA)` | Displays a notebook-facing result for inspection. |
| `print("dataset id:", DATASET_ID)` | Displays a notebook-facing result for inspection. |
| `print("run id:", RUN_ID)` | Displays a notebook-facing result for inspection. |
| `print("send queue table:", SEND_QUEUE_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("producer topic:", PRODUCER_TOPIC)` | Displays a notebook-facing result for inspection. |
| `print("observation batch size:", OBSERVATION_BATCH_SIZE)` | Displays a notebook-facing result for inspection. |
| `print("apply owner/grants:", APPLY_OWNER_AND_GRANTS_FLAG)` | Displays a notebook-facing result for inspection. |

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

- `batch`
- `COUNT`
- `DATASET_ID`
- `dataset_id`
- `Derived`
- `DISTINCT`
- `engine`
- `f`
- `loc`
- `number`
- `observation`
- `OBSERVATION_BATCH_SIZE`
- `of`
- `producer`
- `read_sql_dataframe`
- `run_id`
- `RUN_ID`
- `SCHEMA`
- `SELECT`
- `SEND_QUEUE_TABLE_NAME`

### Outputs

- `NUMBER_OF_SENSORS`
- `params`
- `PRODUCER_BATCH_SIZE`

### Key Operations

- `NUMBER_OF_SENSORS = int( read_sql_dataframe( engine, f""" SELECT COUNT(DISTINCT sensor_index) AS sensor_count FROM "{SCHEMA}"."{SEND_QUEUE_TABLE_NAME}" WHERE dataset_id = :dataset_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `PRODUCER_BATCH_SIZE = OBSERVATION_BATCH_SIZE * NUMBER_OF_SENSORS`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("Derived producer batch sizing")`: Displays a notebook-facing result for inspection.
- `print("number of sensors:", NUMBER_OF_SENSORS)`: Displays a notebook-facing result for inspection.
- `print("observation batch size:", OBSERVATION_BATCH_SIZE)`: Displays a notebook-facing result for inspection.
- `print("producer batch size:", PRODUCER_BATCH_SIZE)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `NUMBER_OF_SENSORS = int( read_sql_dataframe( engine, f""" SELECT COUNT(DISTINCT sensor_index) AS sensor_count FROM "{SCHEMA}"."{SEND_QUEUE_TABLE_NAME}" WHERE dataset_id = :dataset_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PRODUCER_BATCH_SIZE = OBSERVATION_BATCH_SIZE * NUMBER_OF_SENSORS` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Derived producer batch sizing")` | Displays a notebook-facing result for inspection. |
| `print("number of sensors:", NUMBER_OF_SENSORS)` | Displays a notebook-facing result for inspection. |
| `print("observation batch size:", OBSERVATION_BATCH_SIZE)` | Displays a notebook-facing result for inspection. |
| `print("producer batch size:", PRODUCER_BATCH_SIZE)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 05 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `APPLY_OWNER_AND_GRANTS_FLAG`
- `control`
- `CONTROL_OWNER_ROLE`
- `ensure_simulation_state_control_table`
- `Ensured`
- `SIMULATION_TABLE_NAME`
- `table`

### Outputs

- `apply_owner_and_grants`
- `control_table_name`
- `engine`
- `owner_role`
- `schema`
- `table_name`

### Key Operations

- `control_table_name = ensure_simulation_state_control_table( engine=engine, schema=SCHEMA, table_name=SIMULATION_TABLE_NAME, owner_role=CONTROL_OWNER_ROLE, apply_owner_and_grants=AP`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Ensured control table:", control_table_name)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `ensure_simulation_state_control_table`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `control_table_name = ensure_simulation_state_control_table( engine=engine, schema=SCHEMA, table_name=SIMULATION_TABLE_NAME, owner_role=CONTROL_OWNER_ROLE, apply_owner_and_grants=AP` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Ensured control table:", control_table_name)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 06 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `control`
- `PRODUCER_MAX_SEND_ATTEMPTS`
- `row`
- `SIMULATION_TABLE_NAME`
- `upsert_simulation_state_control`
- `Upserted`

### Outputs

- `dataset_id`
- `engine`
- `is_enabled`
- `max_send_attempts`
- `producer_batch_size`
- `producer_poll_seconds`
- `producer_topic`
- `run_id`
- `schema`
- `table_name`

### Key Operations

- `upsert_simulation_state_control( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, is_enabled=True, producer_topic=PRODUCER_TOPIC, producer_batch_size=PRODUCER_BATCH_SIZE, produ`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Upserted control row.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `upsert_simulation_state_control`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `upsert_simulation_state_control( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, is_enabled=True, producer_topic=PRODUCER_TOPIC, producer_batch_size=PRODUCER_BATCH_SIZE, produ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Upserted control row.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 07 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `read_simulation_state_control`
- `SIMULATION_TABLE_NAME`

### Outputs

- `control_row`
- `dataset_id`
- `engine`
- `run_id`
- `schema`
- `table_name`

### Key Operations

- `control_row = read_simulation_state_control( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, schema=SCHEMA, table_name=SIMULATION_TABLE_NAME,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(control_row)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_simulation_state_control`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `control_row = read_simulation_state_control( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, schema=SCHEMA, table_name=SIMULATION_TABLE_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(control_row)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 08 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `ensure_send_queue_runtime_columns`
- `Ensured`
- `indexes`
- `queue`
- `runtime`
- `send`
- `SEND_QUEUE_TABLE_NAME`

### Outputs

- `engine`
- `schema`
- `table_name`

### Key Operations

- `ensure_send_queue_runtime_columns( engine=engine, schema=SCHEMA, table_name=SEND_QUEUE_TABLE_NAME,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Ensured send queue runtime columns and indexes.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `ensure_send_queue_runtime_columns`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `ensure_send_queue_runtime_columns( engine=engine, schema=SCHEMA, table_name=SEND_QUEUE_TABLE_NAME,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Ensured send queue runtime columns and indexes.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 09 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `claim_token`
- `claimed_count`
- `COUNT`
- `dataset_id`
- `DATASET_ID`
- `DESC`
- `engine`
- `f`
- `FILTER`
- `GROUP`
- `NULL`
- `ORDER`
- `producer_sent_at`
- `queue_status`
- `read_sql_dataframe`
- `row_count`
- `run_id`
- `RUN_ID`
- `SCHEMA`

### Outputs

- `params`
- `queue_health_dataframe`

### Key Operations

- `queue_health_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count, COUNT(*) FILTER (WHERE claim_token IS NULL) AS unclaimed_count, COUNT(*) FILTE`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(queue_health_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `queue_health_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count, COUNT(*) FILTER (WHERE claim_token IS NULL) AS unclaimed_count, COUNT(*) FILTE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(queue_health_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 10 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `ANALYZE`
- `BY`
- `dataset_id`
- `DATASET_ID`
- `engine`
- `EXPLAIN`
- `f`
- `LIMIT`
- `message_key`
- `message_sequence_index`
- `NULL`
- `observation_index`
- `ORDER`
- `pending`
- `producer_batch_size`
- `PRODUCER_BATCH_SIZE`
- `producer_sent_at`
- `queue_status`
- `read_sql_dataframe`
- `RUN_ID`

### Outputs

- `params`
- `producer_pickup_explain_dataframe`

### Key Operations

- `producer_pickup_explain_dataframe = read_sql_dataframe( engine, f""" EXPLAIN ANALYZE SELECT message_key, observation_index, message_sequence_index, sensor_index FROM "{SCHEMA}"."{S`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(producer_pickup_explain_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `producer_pickup_explain_dataframe = read_sql_dataframe( engine, f""" EXPLAIN ANALYZE SELECT message_key, observation_index, message_sequence_index, sensor_index FROM "{SCHEMA}"."{S` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(producer_pickup_explain_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 11 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `get_send_queue_status_counts`
- `SEND_QUEUE_TABLE_NAME`

### Outputs

- `engine`
- `queue_status_dataframe`
- `schema`
- `table_name`

### Key Operations

- `queue_status_dataframe = get_send_queue_status_counts( engine=engine, schema=SCHEMA, table_name=SEND_QUEUE_TABLE_NAME,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(queue_status_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `get_send_queue_status_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `queue_status_dataframe = get_send_queue_status_counts( engine=engine, schema=SCHEMA, table_name=SEND_QUEUE_TABLE_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(queue_status_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 12 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `before`
- `claim_token`
- `claimed`
- `claimed_count`
- `clean`
- `COUNT`
- `dataset_id`
- `DATASET_ID`
- `engine`
- `f`
- `failed`
- `failed_count`
- `FILTER`
- `iloc`
- `NULL`
- `pending`
- `pending_count`
- `populated_claim_token_count`
- `populated_error_count`
- `populated_sent_at_count`

### Outputs

- `params`
- `ready_for_stage_7`
- `row`
- `stage_7_readiness_dataframe`

### Key Operations

- `stage_7_readiness_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS total_queue_rows, COUNT(*) FILTER (WHERE queue_status = 'pending') AS pending_count, COUNT(*) FILT`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_7_readiness_dataframe)`: Displays a notebook-facing result for inspection.
- `row = stage_7_readiness_dataframe.iloc[0]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ready_for_stage_7 = ( int(row["total_queue_rows"]) > 0 and int(row["pending_count"]) == int(row["total_queue_rows"]) and int(row["claimed_count"]) == 0 and int(row["sent_count"]) =`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Ready for Stage 7:", ready_for_stage_7)`: Displays a notebook-facing result for inspection.
- `if not ready_for_stage_7: print("Queue is not clean. Use 06.5 repair/reset tools before running Stage 7.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_7_readiness_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS total_queue_rows, COUNT(*) FILTER (WHERE queue_status = 'pending') AS pending_count, COUNT(*) FILT` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_7_readiness_dataframe)` | Displays a notebook-facing result for inspection. |
| `row = stage_7_readiness_dataframe.iloc[0]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ready_for_stage_7 = ( int(row["total_queue_rows"]) > 0 and int(row["pending_count"]) == int(row["total_queue_rows"]) and int(row["claimed_count"]) == 0 and int(row["sent_count"]) =` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Ready for Stage 7:", ready_for_stage_7)` | Displays a notebook-facing result for inspection. |
| `if not ready_for_stage_7: print("Queue is not clean. Use 06.5 repair/reset tools before running Stage 7.")` | Displays a notebook-facing result for inspection. |

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

## Code Cell 14 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `claim_token`
- `COUNT`
- `dataset_id`
- `DATASET_ID`
- `DESC`
- `engine`
- `error_count`
- `f`
- `FILTER`
- `GROUP`
- `NULL`
- `ORDER`
- `populated_claim_token_count`
- `producer_delivery_error`
- `producer_sent_at`
- `queue_status`
- `read_sql_dataframe`
- `row_count`
- `RUN_ID`

### Outputs

- `params`
- `progress_dataframe`

### Key Operations

- `progress_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count, COUNT(*) FILTER (WHERE producer_sent_at IS NOT NULL) AS sent_timestamp_count, COUN`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(progress_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `progress_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count, COUNT(*) FILTER (WHERE producer_sent_at IS NOT NULL) AS sent_timestamp_count, COUN` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(progress_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 15 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `consumed_row_count`
- `consumer_received_at`
- `COUNT`
- `dataset_id`
- `DATASET_ID`
- `DISTINCT`
- `distinct_message_key_count`
- `distinct_observation_count`
- `engine`
- `f`
- `FILTER`
- `first_consumed_at`
- `last_consumed_at`
- `MAX`
- `message_key`
- `MIN`
- `observation_index`
- `pending`
- `pending_rebuild_count`
- `read_sql_dataframe`

### Outputs

- `consumer_progress_dataframe`
- `params`

### Key Operations

- `consumer_progress_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS consumed_row_count, COUNT(DISTINCT message_key) AS distinct_message_key_count, COUNT(DISTINCT obse`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(consumer_progress_dataframe)`: Displays a notebook-facing result for inspection.

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
| `consumer_progress_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS consumed_row_count, COUNT(DISTINCT message_key) AS distinct_message_key_count, COUNT(DISTINCT obse` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(consumer_progress_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

