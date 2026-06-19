# Notebook Code Reference: synthetic_07_kafka_producer_adapter

Notebook path:

`notebooks/synthetic/synthetic_07_kafka_producer_adapter.ipynb`

## Notebook Purpose

This notebook documents and runs a focused support step in the synthetic pump telemetry workflow.

Notebook stage:

`Synthetic support`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06 |
| One Batch | Code Cell 07, Code Cell 08 |
| Loop | Code Cell 09, Code Cell 10, Code Cell 11, Code Cell 12, Code Cell 13, Code Cell 14, Code Cell 15, Code Cell 16 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_sensor_message_payload`
- `core`
- `database`
- `env_float`
- `env_helpers`
- `env_int`
- `env_optional_int`
- `env_str`
- `get_engine_from_env`
- `json_dumps_safe`
- `kafka_producer_adapter`
- `os`
- `pipeline`
- `postgres`
- `read_sql_dataframe`
- `run_send_queue_producer_loop`
- `run_send_queue_producer_once`
- `synthetic`
- `utils`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.kafka_producer_adapter import ( run_send_queue_producer_once, run_send_queue_producer_loop, build_sensor_message_payload, json_dumps_safe,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.core.env_helpers import ( env_float, env_int, env_optional_int, env_str,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.kafka_producer_adapter import ( run_send_queue_producer_once, run_send_queue_producer_loop, build_sensor_message_payload, json_dumps_safe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.env_helpers import ( env_float, env_int, env_optional_int, env_str,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Adapter`
- `batch`
- `batches`
- `capstone`
- `CAPSTONE_SCHEMA`
- `client`
- `config`
- `dataset`
- `env_float`
- `env_int`
- `env_optional_int`
- `env_str`
- `failure`
- `flush`
- `id`
- `KAFKA_CLIENT_ID`
- `KAFKA_TOPIC`
- `loop`
- `max`
- `observation`

### Outputs

- `aliases`
- `CLIENT_ID`
- `DATASET_ID`
- `default`
- `FLUSH_TIMEOUT_SECONDS`
- `MAX_BATCHES`
- `OBSERVATION_BATCH_SIZE`
- `PRODUCER_MAX_SEND_ATTEMPTS`
- `PRODUCER_POLL_SECONDS`
- `PRODUCER_TOPIC`
- `PRODUCER_WORKER_ID`
- `RUN_ID`
- `RUN_ONE_BATCH_PRODUCER_SMOKE_TEST_FLAG`
- `RUN_PRODUCER_LOOP_FLAG`
- `SCHEMA`
- `SEND_QUEUE_TABLE_NAME`
- `SIMULATION_TABLE_NAME`
- `STOP_ON_FAILURE_FLAG`

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
| `SIMULATION_TABLE_NAME = env_str( "SYNTHETIC_CONTROL_TABLE", "simulation_state_control", aliases=("PRODUCER_CONTROL_TABLE",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SEND_QUEUE_TABLE_NAME = env_str( "SYNTHETIC_SEND_QUEUE_TABLE", "synthetic_sensor_messages_send_queue", aliases=("PRODUCER_QUEUE_TABLE",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PRODUCER_TOPIC = env_str( "SYNTHETIC_KAFKA_TOPIC", "pump.telemetry.synthetic", aliases=("KAFKA_TOPIC",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PRODUCER_WORKER_ID = env_str( "SYNTHETIC_PRODUCER_WORKER_ID", "producer_worker_test_001",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CLIENT_ID = env_str( "SYNTHETIC_PRODUCER_CLIENT_ID", "synthetic-telemetry-producer", aliases=("KAFKA_CLIENT_ID", "SYNTHETIC_PRODUCER_GROUP_ID"),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `OBSERVATION_BATCH_SIZE = env_int("OBSERVATION_BATCH_SIZE", 500)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PRODUCER_POLL_SECONDS = env_float("PRODUCER_POLL_SECONDS", 0.0)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PRODUCER_MAX_SEND_ATTEMPTS = env_int("PRODUCER_MAX_SEND_ATTEMPTS", 3)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FLUSH_TIMEOUT_SECONDS = env_float("PRODUCER_FLUSH_TIMEOUT_SECONDS", 30.0)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_PRODUCER_LOOP_FLAG = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_ONE_BATCH_PRODUCER_SMOKE_TEST_FLAG = False` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MAX_BATCHES = env_optional_int( "PRODUCER_MAX_BATCHES_LIMIT", default=1, aliases=("SYNTHETIC_PRODUCER_MAX_BATCHES_LIMIT",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STOP_ON_FAILURE_FLAG = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Producer Adapter config")` | Displays a notebook-facing result for inspection. |
| `print("schema:", SCHEMA)` | Displays a notebook-facing result for inspection. |
| `print("dataset id:", DATASET_ID)` | Displays a notebook-facing result for inspection. |
| `print("run id:", RUN_ID)` | Displays a notebook-facing result for inspection. |
| `print("send queue table:", SEND_QUEUE_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("producer topic:", PRODUCER_TOPIC)` | Displays a notebook-facing result for inspection. |
| `print("observation batch size:", OBSERVATION_BATCH_SIZE)` | Displays a notebook-facing result for inspection. |
| `print("client id:", CLIENT_ID)` | Displays a notebook-facing result for inspection. |
| `print("flush timeout seconds:", FLUSH_TIMEOUT_SECONDS)` | Displays a notebook-facing result for inspection. |
| `print("run producer loop:", RUN_PRODUCER_LOOP_FLAG)` | Displays a notebook-facing result for inspection. |
| `print("producer max batches:", MAX_BATCHES)` | Displays a notebook-facing result for inspection. |
| `print("stop on failure:", STOP_ON_FAILURE_FLAG)` | Displays a notebook-facing result for inspection. |

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

- `BY`
- `claim_token`
- `COUNT`
- `dataset_id`
- `DATASET_ID`
- `DESC`
- `engine`
- `f`
- `FILTER`
- `GROUP`
- `NULL`
- `null_claim_token_count`
- `ORDER`
- `populated_claim_token_count`
- `producer_sent_at`
- `queue_status`
- `read_sql_dataframe`
- `row_count`
- `run_id`
- `RUN_ID`

### Outputs

- `params`
- `pre_send_queue_status_dataframe`

### Key Operations

- `pre_send_queue_status_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count, COUNT(*) FILTER (WHERE claim_token IS NULL) AS null_claim_token_count`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(pre_send_queue_status_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `pre_send_queue_status_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count, COUNT(*) FILTER (WHERE claim_token IS NULL) AS null_claim_token_count` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(pre_send_queue_status_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 06 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `available`
- `build_sensor_message_payload`
- `BY`
- `DATASET_ID`
- `dataset_id`
- `else`
- `empty`
- `engine`
- `f`
- `iloc`
- `json_dumps_safe`
- `LIMIT`
- `message_sequence_index`
- `No`
- `observation_index`
- `ORDER`
- `pending`
- `preview`
- `queue_status`
- `read_sql_dataframe`

### Outputs

- `params`
- `payload`
- `preview_dataframe`

### Key Operations

- `preview_dataframe = read_sql_dataframe( engine, f""" SELECT * FROM "{SCHEMA}"."{SEND_QUEUE_TABLE_NAME}" WHERE dataset_id = :dataset_id AND run_id = :run_id AND queue_status = 'pend`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if preview_dataframe.empty: print("No pending rows available for payload preview.")`: Displays a notebook-facing result for inspection.
- `else: payload = build_sensor_message_payload(preview_dataframe.iloc[0].to_dict()) print(json_dumps_safe(payload))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_sensor_message_payload`
- `json_dumps_safe`
- `read_sql_dataframe`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `preview_dataframe = read_sql_dataframe( engine, f""" SELECT * FROM "{SCHEMA}"."{SEND_QUEUE_TABLE_NAME}" WHERE dataset_id = :dataset_id AND run_id = :run_id AND queue_status = 'pend` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if preview_dataframe.empty: print("No pending rows available for payload preview.")` | Displays a notebook-facing result for inspection. |
| `else: payload = build_sensor_message_payload(preview_dataframe.iloc[0].to_dict()) print(json_dumps_safe(payload))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 07 — One Batch

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `batch`
- `before`
- `else`
- `exactly`
- `execution`
- `loop`
- `normal`
- `official`
- `one`
- `only`
- `Optional`
- `producer`
- `run`
- `RUN_ONE_BATCH_PRODUCER_SMOKE_TEST_FLAG`
- `run_send_queue_producer_once`
- `running`
- `SEND_QUEUE_TABLE_NAME`
- `SIMULATION_TABLE_NAME`
- `Skipping`
- `smoke`

### Outputs

- `client_id`
- `control_table`
- `dataset_id`
- `engine`
- `flush_timeout_seconds`
- `one_batch_result`
- `producer_worker_id`
- `queue_table`
- `run_id`
- `schema`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Optional smoke test: run exactly one producer batch`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# This is not the normal official Stage 7 execution path.`: Documents the purpose or boundary of the surrounding notebook step.
- `# Use only when you want to test one batch before running the loop.`: Documents the purpose or boundary of the surrounding notebook step.
- `#RUN_ONE_BATCH_PRODUCER_SMOKE_TEST_FLAG = False`: Documents the purpose or boundary of the surrounding notebook step.
- `if RUN_ONE_BATCH_PRODUCER_SMOKE_TEST_FLAG: one_batch_result = run_send_queue_producer_once( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, schema=SCHEMA, queue_table=SEND_QUE`: Displays a notebook-facing result for inspection.
- `else: print("RUN_ONE_BATCH_PRODUCER_SMOKE_TEST_FLAG is False. Skipping one-batch smoke test.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `run_send_queue_producer_once`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Optional smoke test: run exactly one producer batch` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# This is not the normal official Stage 7 execution path.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Use only when you want to test one batch before running the loop.` | Documents the purpose or boundary of the surrounding notebook step. |
| `#RUN_ONE_BATCH_PRODUCER_SMOKE_TEST_FLAG = False` | Documents the purpose or boundary of the surrounding notebook step. |
| `if RUN_ONE_BATCH_PRODUCER_SMOKE_TEST_FLAG: one_batch_result = run_send_queue_producer_once( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, schema=SCHEMA, queue_table=SEND_QUE` | Displays a notebook-facing result for inspection. |
| `else: print("RUN_ONE_BATCH_PRODUCER_SMOKE_TEST_FLAG is False. Skipping one-batch smoke test.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 08 — One Batch

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
- `null_claim_token_count`
- `ORDER`
- `populated_claim_token_count`
- `producer_delivery_error`
- `producer_sent_at`
- `queue_status`
- `read_sql_dataframe`
- `row_count`

### Outputs

- `params`
- `post_send_queue_status_dataframe`

### Key Operations

- `post_send_queue_status_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count, COUNT(*) FILTER (WHERE claim_token IS NULL) AS null_claim_token_coun`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(post_send_queue_status_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `post_send_queue_status_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count, COUNT(*) FILTER (WHERE claim_token IS NULL) AS null_claim_token_coun` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(post_send_queue_status_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 09 — Loop

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `control`
- `disables`
- `drain`
- `else`
- `empty`
- `execution`
- `failure`
- `first`
- `full`
- `Kafka`
- `loop`
- `occurs`
- `Official`
- `producer`
- `queue`
- `reached`
- `row`
- `run`
- `Run`

### Outputs

- `client_id`
- `control_table`
- `dataset_id`
- `enable_progress_logging`
- `engine`
- `flush_timeout_seconds`
- `loop_results`
- `max_batches`
- `producer_worker_id`
- `progress_every_batches`
- `queue_table`
- `run_id`
- `schema`
- `stop_on_failure`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Run Kafka producer loop`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Official Stage 7 execution path.`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# For first test:`: Documents the purpose or boundary of the surrounding notebook step.
- `# RUN_PRODUCER_LOOP_FLAG = True`: Documents the purpose or boundary of the surrounding notebook step.
- `# MAX_BATCHES = 1`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# For full queue drain:`: Documents the purpose or boundary of the surrounding notebook step.
- `# RUN_PRODUCER_LOOP_FLAG = True`: Documents the purpose or boundary of the surrounding notebook step.
- `# MAX_BATCHES = None`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `display`
- `run_send_queue_producer_loop`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Run Kafka producer loop` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Official Stage 7 execution path.` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# For first test:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# RUN_PRODUCER_LOOP_FLAG = True` | Documents the purpose or boundary of the surrounding notebook step. |
| `# MAX_BATCHES = 1` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# For full queue drain:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# RUN_PRODUCER_LOOP_FLAG = True` | Documents the purpose or boundary of the surrounding notebook step. |
| `# MAX_BATCHES = None` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The loop stops when:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - queue is empty` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - control row disables the run` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - max_batches is reached` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - a failure occurs and stop_on_failure=True` | Documents the purpose or boundary of the surrounding notebook step. |
| `#RUN_PRODUCER_LOOP_FLAG = False` | Documents the purpose or boundary of the surrounding notebook step. |
| `if RUN_PRODUCER_LOOP_FLAG: loop_results = run_send_queue_producer_loop( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, schema=SCHEMA, queue_table=SEND_QUEUE_TABLE_NAME, contr` | Displays a notebook-facing result for inspection. |
| `else: print("RUN_PRODUCER_LOOP_FLAG is False. Skipping producer loop.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 10 — Loop

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `claim_token`
- `claimed`
- `claimed_count`
- `COUNT`
- `DATASET_ID`
- `dataset_id`
- `DISTINCT`
- `distinct_message_key_count`
- `distinct_observation_count`
- `engine`
- `error_count`
- `f`
- `failed`
- `failed_count`
- `FILTER`
- `message_key`
- `NULL`
- `observation_index`
- `pending`
- `pending_count`

### Outputs

- `params`
- `stage_7_final_validation_dataframe`

### Key Operations

- `stage_7_final_validation_dataframe = read_sql_dataframe( engine, f""" WITH queue_summary AS ( SELECT COUNT(*) AS total_queue_rows, COUNT(*) FILTER (WHERE queue_status = 'sent') AS `: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_7_final_validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `AS`
- `COUNT`
- `display`
- `FILTER`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_7_final_validation_dataframe = read_sql_dataframe( engine, f""" WITH queue_summary AS ( SELECT COUNT(*) AS total_queue_rows, COUNT(*) FILTER (WHERE queue_status = 'sent') AS ` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_7_final_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 11 — Loop

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

## Code Cell 12 — Loop

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

## Code Cell 13 — Loop

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `consumer_received_at`
- `COUNT`
- `dataset_id`
- `DISTINCT`
- `distinct_dataset_count`
- `distinct_message_key_count`
- `distinct_observation_count`
- `distinct_run_count`
- `engine`
- `f`
- `first_consumed_at`
- `last_consumed_at`
- `MAX`
- `message_key`
- `MIN`
- `observation_index`
- `read_sql_dataframe`
- `run_id`
- `SCHEMA`
- `SELECT`

### Outputs

- `consumed_global_dataframe`

### Key Operations

- `consumed_global_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS total_consumed_rows, COUNT(DISTINCT dataset_id) AS distinct_dataset_count, COUNT(DISTINCT run_id) AS`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(consumed_global_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `MAX`
- `MIN`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `consumed_global_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS total_consumed_rows, COUNT(DISTINCT dataset_id) AS distinct_dataset_count, COUNT(DISTINCT run_id) AS` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(consumed_global_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 14 — Loop

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `consumer_group_id`
- `consumer_received_at`
- `consumer_worker_id`
- `COUNT`
- `dataset_id`
- `DESC`
- `DISTINCT`
- `distinct_message_key_count`
- `distinct_observation_count`
- `engine`
- `f`
- `first_consumed_at`
- `GROUP`
- `last_consumed_at`
- `LIMIT`
- `MAX`
- `message_key`
- `MIN`
- `observation_index`

### Outputs

- `consumed_dataset_run_breakdown_dataframe`

### Key Operations

- `consumed_dataset_run_breakdown_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, consumer_group_id, consumer_worker_id, COUNT(*) AS row_count, COUNT(DISTINCT `: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(consumed_dataset_run_breakdown_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `MAX`
- `MIN`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `consumed_dataset_run_breakdown_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, consumer_group_id, consumer_worker_id, COUNT(*) AS row_count, COUNT(DISTINCT ` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(consumed_dataset_run_breakdown_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 15 — Loop

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `consumer_received_at`
- `COUNT`
- `dataset_id`
- `DISTINCT`
- `distinct_dataset_count`
- `distinct_message_key_count`
- `distinct_observation_count`
- `distinct_run_count`
- `engine`
- `f`
- `FILTER`
- `first_consumed_at`
- `last_consumed_at`
- `MAX`
- `message_key`
- `MIN`
- `NULL`
- `null_message_key_count`
- `null_sensor_value_count`
- `observation_index`

### Outputs

- `stage_8_global_validation_dataframe`

### Key Operations

- `stage_8_global_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS total_consumed_rows, COUNT(DISTINCT dataset_id) AS distinct_dataset_count, COUNT(DISTINCT `: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_8_global_validation_dataframe)`: Displays a notebook-facing result for inspection.

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
| `stage_8_global_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS total_consumed_rows, COUNT(DISTINCT dataset_id) AS distinct_dataset_count, COUNT(DISTINCT ` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_8_global_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 16 — Loop

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `consumed_row_count`
- `consumer_received_at`
- `COUNT`
- `DATASET_ID`
- `dataset_id`
- `DISTINCT`
- `distinct_message_key_count`
- `distinct_observation_count`
- `distinct_sensor_index_count`
- `distinct_sensor_name_count`
- `duplicate_offset_count`
- `engine`
- `f`
- `FILTER`
- `first_consumed_at`
- `is_duplicate`
- `last_consumed_at`
- `MAX`
- `max_observation_index`
- `message_key`

### Outputs

- `params`
- `stage_8_final_validation_dataframe`

### Key Operations

- `stage_8_final_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS consumed_row_count, COUNT(DISTINCT message_key) AS distinct_message_key_count, COUNT(DISTIN`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_8_final_validation_dataframe)`: Displays a notebook-facing result for inspection.

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
| `stage_8_final_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS consumed_row_count, COUNT(DISTINCT message_key) AS distinct_message_key_count, COUNT(DISTIN` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_8_final_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

