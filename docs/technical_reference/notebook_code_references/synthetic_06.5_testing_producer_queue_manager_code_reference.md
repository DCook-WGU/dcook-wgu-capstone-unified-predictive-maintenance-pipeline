# Notebook Code Reference: synthetic_06.5_testing_producer_queue_manager

Notebook path:

`notebooks/synthetic/synthetic_06.5_testing_producer_queue_manager.ipynb`

## Status

Testing / support reference.

This notebook is documented because it remains in the repository and supports focused validation of the producer queue manager. It should not be treated as the preferred end-to-end synthetic workflow unless it is explicitly selected for testing or troubleshooting.

## Notebook Purpose

This notebook documents and runs a focused support step in the synthetic pump telemetry workflow.

Notebook stage:

`Synthetic support`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08, Code Cell 09, Code Cell 10, Code Cell 11, Code Cell 12, Code Cell 13, Code Cell 14, Code Cell 15, Code Cell 16, Code Cell 17, Code Cell 18, Code Cell 19, Code Cell 20, Code Cell 21, Code Cell 22, Code Cell 23, Code Cell 24 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `claim_pending_sensor_messages_batch`
- `core`
- `database`
- `ensure_send_queue_runtime_columns`
- `env_float`
- `env_helpers`
- `env_int`
- `env_str`
- `execute_sql`
- `get_engine_from_env`
- `get_send_queue_status_counts`
- `mark_claimed_batch_failed`
- `mark_claimed_batch_sent`
- `os`
- `pipeline`
- `postgres`
- `producer_queue_manager`
- `read_simulation_state_control`
- `read_sql_dataframe`
- `release_stale_claims`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe, execute_sql,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.producer_queue_manager import ( ensure_send_queue_runtime_columns, read_simulation_state_control, get_send_queue_status_counts, claim_pending_sensor_m`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.core.env_helpers import ( env_float, env_int, env_str,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe, execute_sql,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.producer_queue_manager import ( ensure_send_queue_runtime_columns, read_simulation_state_control, get_send_queue_status_counts, claim_pending_sensor_m` | Imports a dependency or project helper used by later cells. |
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

- `all`
- `almost`
- `alter`
- `always`
- `are`
- `back`
- `behavior`
- `but`
- `can`
- `claim`
- `claimed`
- `claims`
- `default`
- `Emergency`
- `emergency`
- `failed`
- `Flags`
- `flags`
- `intentionally`
- `It`

### Outputs

- `EMERGENCY_RESET_ALL_UNSENT_CLAIMS_FLAG`
- `MARK_FAILED_TEST_FLAG`
- `MARK_SENT_TEST_FLAG`
- `RELEASE_STALE_CLAIMS_TEST_FLAG`
- `REQUEUE_FAILED_TEST_FLAG`
- `RESET_TEST_CLAIMS_FLAG`
- `RUN_CLAIM_TEST_FLAG`
- `TEST_ERROR_MESSAGE`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# 6.5 Testing / Sandbox Safety Flags`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# This notebook is NOT part of the official pipeline.`: Documents the purpose or boundary of the surrounding notebook step.
- `# These flags intentionally default to safe/no-op behavior.`: Documents the purpose or boundary of the surrounding notebook step.
- `RUN_CLAIM_TEST_FLAG = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Reset claimed-but-unsent rows back to pending.`: Documents the purpose or boundary of the surrounding notebook step.
- `# This is usually safe for test claims.`: Documents the purpose or boundary of the surrounding notebook step.
- `RESET_TEST_CLAIMS_FLAG = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Optional lifecycle tests.`: Documents the purpose or boundary of the surrounding notebook step.
- `# Leave these False unless you are intentionally testing that transition.`: Documents the purpose or boundary of the surrounding notebook step.
- `MARK_SENT_TEST_FLAG = False`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 6.5 Testing / Sandbox Safety Flags` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# This notebook is NOT part of the official pipeline.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# These flags intentionally default to safe/no-op behavior.` | Documents the purpose or boundary of the surrounding notebook step. |
| `RUN_CLAIM_TEST_FLAG = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Reset claimed-but-unsent rows back to pending.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# This is usually safe for test claims.` | Documents the purpose or boundary of the surrounding notebook step. |
| `RESET_TEST_CLAIMS_FLAG = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Optional lifecycle tests.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Leave these False unless you are intentionally testing that transition.` | Documents the purpose or boundary of the surrounding notebook step. |
| `MARK_SENT_TEST_FLAG = False` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MARK_FAILED_TEST_FLAG = False` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `REQUEUE_FAILED_TEST_FLAG = False` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RELEASE_STALE_CLAIMS_TEST_FLAG = False` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Emergency reset should almost always stay False.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# It can alter larger portions of the queue.` | Documents the purpose or boundary of the surrounding notebook step. |
| `EMERGENCY_RESET_ALL_UNSENT_CLAIMS_FLAG = False` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TEST_ERROR_MESSAGE = "Manual Stage 6.5 failed-message test"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Stage 6.5 sandbox flags")` | Displays a notebook-facing result for inspection. |
| `print("run claim test:", RUN_CLAIM_TEST_FLAG)` | Displays a notebook-facing result for inspection. |
| `print("reset test claims:", RESET_TEST_CLAIMS_FLAG)` | Displays a notebook-facing result for inspection. |
| `print("mark sent test:", MARK_SENT_TEST_FLAG)` | Displays a notebook-facing result for inspection. |
| `print("mark failed test:", MARK_FAILED_TEST_FLAG)` | Displays a notebook-facing result for inspection. |
| `print("requeue failed test:", REQUEUE_FAILED_TEST_FLAG)` | Displays a notebook-facing result for inspection. |
| `print("release stale claims test:", RELEASE_STALE_CLAIMS_TEST_FLAG)` | Displays a notebook-facing result for inspection. |
| `print("emergency reset all unsent claims:", EMERGENCY_RESET_ALL_UNSENT_CLAIMS_FLAG)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 03 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `attempts`
- `batch`
- `capstone`
- `CAPSTONE_SCHEMA`
- `claim`
- `config`
- `dataset`
- `env_float`
- `env_int`
- `env_str`
- `id`
- `kafka_producer`
- `KAFKA_TOPIC`
- `manager`
- `max`
- `observation`
- `producer`
- `Producer`
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
| `print("Producer queue manager claim/reset test config")` | Displays a notebook-facing result for inspection. |
| `print("schema:", SCHEMA)` | Displays a notebook-facing result for inspection. |
| `print("dataset id:", DATASET_ID)` | Displays a notebook-facing result for inspection. |
| `print("run id:", RUN_ID)` | Displays a notebook-facing result for inspection. |
| `print("send queue table:", SEND_QUEUE_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("producer topic:", PRODUCER_TOPIC)` | Displays a notebook-facing result for inspection. |
| `print("producer worker id:", PRODUCER_WORKER_ID)` | Displays a notebook-facing result for inspection. |
| `print("producer max send attempts:", PRODUCER_MAX_SEND_ATTEMPTS)` | Displays a notebook-facing result for inspection. |
| `print("observation batch size:", OBSERVATION_BATCH_SIZE)` | Displays a notebook-facing result for inspection. |

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

- `batch`
- `claim`
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
- `RUN_ID`
- `run_id`
- `SCHEMA`
- `SELECT`

### Outputs

- `NUMBER_OF_SENSORS`
- `params`
- `PRODUCER_BATCH_SIZE`

### Key Operations

- `NUMBER_OF_SENSORS = int( read_sql_dataframe( engine, f""" SELECT COUNT(DISTINCT sensor_index) AS sensor_count FROM "{SCHEMA}"."{SEND_QUEUE_TABLE_NAME}" WHERE dataset_id = :dataset_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `PRODUCER_BATCH_SIZE = OBSERVATION_BATCH_SIZE * NUMBER_OF_SENSORS`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("Derived claim batch sizing")`: Displays a notebook-facing result for inspection.
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
| `print("Derived claim batch sizing")` | Displays a notebook-facing result for inspection. |
| `print("number of sensors:", NUMBER_OF_SENSORS)` | Displays a notebook-facing result for inspection. |
| `print("observation batch size:", OBSERVATION_BATCH_SIZE)` | Displays a notebook-facing result for inspection. |
| `print("producer batch size:", PRODUCER_BATCH_SIZE)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 06 — Code Reference

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

## Code Cell 07 — Code Reference

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

## Code Cell 08 — Code Reference

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

## Code Cell 09 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `bool`
- `claim`
- `claim_token`
- `claimed`
- `claimed_count`
- `COUNT`
- `dataset_id`
- `def`
- `diagnostic`
- `does`
- `engine`
- `f`
- `failed`
- `failed_count`
- `FILTER`
- `helper`
- `iloc`
- `mark`
- `mutate`
- `NULL`

### Outputs

- `check_stage_7_readiness`
- `params`
- `ready_for_stage_7`
- `row`
- `stage_7_readiness_dataframe`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Stage 7 readiness helper`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Safe diagnostic only. This does not claim, reset, mark sent, or mutate rows.`: Documents the purpose or boundary of the surrounding notebook step.
- `def check_stage_7_readiness( engine, *, schema: str, send_queue_table_name: str, dataset_id: str, run_id: str,`: Defines notebook-local logic used later in the notebook.
- `) -> tuple[bool, object]: stage_7_readiness_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS total_queue_rows, COUNT(*) FILTER (WHERE queue_status = 'pending') AS pe`: Loads input data, configuration, or artifacts required by the current stage.

Important functions or methods detected:
- `check_stage_7_readiness`
- `COUNT`
- `FILTER`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Stage 7 readiness helper` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Safe diagnostic only. This does not claim, reset, mark sent, or mutate rows.` | Documents the purpose or boundary of the surrounding notebook step. |
| `def check_stage_7_readiness( engine, *, schema: str, send_queue_table_name: str, dataset_id: str, run_id: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> tuple[bool, object]: stage_7_readiness_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS total_queue_rows, COUNT(*) FILTER (WHERE queue_status = 'pending') AS pe` | Loads input data, configuration, or artifacts required by the current stage. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 10 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `cells`
- `check`
- `Check`
- `check_stage_7_readiness`
- `clean`
- `enough`
- `Queue`
- `queue`
- `readiness`
- `Ready`
- `ready_for_stage_7`
- `repair`
- `rerun`
- `reset`
- `Stage`
- `stage_7_readiness_dataframe`
- `the`
- `then`
- `this`
- `Use`

### Outputs

- `dataset_id`
- `engine`
- `run_id`
- `schema`
- `send_queue_table_name`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Check whether queue is clean enough for Stage 7`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `ready_for_stage_7, stage_7_readiness_dataframe = check_stage_7_readiness( engine=engine, schema=SCHEMA, send_queue_table_name=SEND_QUEUE_TABLE_NAME, dataset_id=DATASET_ID, run_id=R`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_7_readiness_dataframe)`: Displays a notebook-facing result for inspection.
- `print("Ready for Stage 7:", ready_for_stage_7)`: Displays a notebook-facing result for inspection.
- `if not ready_for_stage_7: print( "Queue is not clean for Stage 7. Use the 6.5 reset/repair cells, " "then rerun this readiness check." )`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `check_stage_7_readiness`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Check whether queue is clean enough for Stage 7` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `ready_for_stage_7, stage_7_readiness_dataframe = check_stage_7_readiness( engine=engine, schema=SCHEMA, send_queue_table_name=SEND_QUEUE_TABLE_NAME, dataset_id=DATASET_ID, run_id=R` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_7_readiness_dataframe)` | Displays a notebook-facing result for inspection. |
| `print("Ready for Stage 7:", ready_for_stage_7)` | Displays a notebook-facing result for inspection. |
| `if not ready_for_stage_7: print( "Queue is not clean for Stage 7. Use the 6.5 reset/repair cells, " "then rerun this readiness check." )` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 11 — Code Reference

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

## Code Cell 12 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `batch`
- `Claim`
- `claim_pending_sensor_messages_batch`
- `Claimed`
- `claimed`
- `count`
- `else`
- `Expected`
- `head`
- `No`
- `one`
- `producer`
- `PRODUCER_BATCH_SIZE`
- `row`
- `rows`
- `RUN_CLAIM_TEST_FLAG`
- `SEND_QUEUE_TABLE_NAME`
- `sized`
- `token`

### Outputs

- `batch_size`
- `claim_token`
- `claimed_dataframe`
- `dataset_id`
- `engine`
- `producer_topic`
- `producer_worker_id`
- `queue_table`
- `run_id`
- `schema`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Claim one producer-sized batch`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `claim_token = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `claimed_dataframe = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if RUN_CLAIM_TEST_FLAG: claim_token, claimed_dataframe = claim_pending_sensor_messages_batch( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, schema=SCHEMA, queue_table=SEND_Q`: Displays a notebook-facing result for inspection.
- `else: print("RUN_CLAIM_TEST_FLAG is False. No rows claimed.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `claim_pending_sensor_messages_batch`
- `display`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Claim one producer-sized batch` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `claim_token = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `claimed_dataframe = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if RUN_CLAIM_TEST_FLAG: claim_token, claimed_dataframe = claim_pending_sensor_messages_batch( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, schema=SCHEMA, queue_table=SEND_Q` | Displays a notebook-facing result for inspection. |
| `else: print("RUN_CLAIM_TEST_FLAG is False. No rows claimed.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 13 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `after`
- `BY`
- `claim`
- `claim_token`
- `claimed_at`
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
- `null_claimed_at_count`
- `ORDER`
- `populated_claim_token_count`
- `populated_claimed_at_count`
- `populated_topic_count`

### Outputs

- `claim_status_validation_dataframe`
- `params`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Validate claim status after claim test`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `claim_status_validation_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count, COUNT(*) FILTER (WHERE claim_token IS NULL) AS null_claim_token_cou`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(claim_status_validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Validate claim status after claim test` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `claim_status_validation_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count, COUNT(*) FILTER (WHERE claim_token IS NULL) AS null_claim_token_cou` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(claim_status_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 14 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `claim`
- `claim_token`
- `claimed`
- `claimed_dataframe`
- `elif`
- `else`
- `empty`
- `head`
- `message_sequence_index`
- `No`
- `observation_index`
- `producer_topic`
- `producer_worker_id`
- `queue_status`
- `rows`
- `run`
- `sensor_index`
- `sensor_name`
- `test`
- `was`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `if claimed_dataframe is None: print("No claim test was run.")`: Displays a notebook-facing result for inspection.
- `elif claimed_dataframe.empty: print("No rows were claimed.")`: Displays a notebook-facing result for inspection.
- `else: display( claimed_dataframe[ [ "claim_token", "observation_index", "message_sequence_index", "sensor_name", "sensor_index", "queue_status", "producer_topic", "producer_worker_`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `if claimed_dataframe is None: print("No claim test was run.")` | Displays a notebook-facing result for inspection. |
| `elif claimed_dataframe.empty: print("No rows were claimed.")` | Displays a notebook-facing result for inspection. |
| `else: display( claimed_dataframe[ [ "claim_token", "observation_index", "message_sequence_index", "sensor_name", "sensor_index", "queue_status", "producer_topic", "producer_worker_` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 15 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `available`
- `batch`
- `cell`
- `changes`
- `claim`
- `claimed`
- `count`
- `else`
- `first`
- `head`
- `intentionally`
- `mark`
- `mark_claimed_batch_sent`
- `MARK_SENT_TEST_FLAG`
- `Marked`
- `No`
- `normal`
- `Only`
- `Optional`
- `producer_ack_at`

### Outputs

- `claim_token`
- `engine`
- `schema`
- `sent_dataframe`
- `table_name`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Optional: mark the claimed batch as sent`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# WARNING:`: Documents the purpose or boundary of the surrounding notebook step.
- `# This changes queue_status to 'sent' and stamps producer_sent_at / producer_ack_at.`: Documents the purpose or boundary of the surrounding notebook step.
- `# The normal reset-test cell will NOT reset sent rows.`: Documents the purpose or boundary of the surrounding notebook step.
- `# Only run this when intentionally testing the sent transition.`: Documents the purpose or boundary of the surrounding notebook step.
- `sent_dataframe = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if MARK_SENT_TEST_FLAG: if not claim_token: raise ValueError("No claim_token available. Run the claim test first.") sent_dataframe = mark_claimed_batch_sent( engine=engine, claim_t`: Displays a notebook-facing result for inspection.
- `else: print("MARK_SENT_TEST_FLAG is False. Skipping mark-sent test.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `head`
- `mark_claimed_batch_sent`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Optional: mark the claimed batch as sent` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# WARNING:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# This changes queue_status to 'sent' and stamps producer_sent_at / producer_ack_at.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The normal reset-test cell will NOT reset sent rows.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Only run this when intentionally testing the sent transition.` | Documents the purpose or boundary of the surrounding notebook step. |
| `sent_dataframe = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if MARK_SENT_TEST_FLAG: if not claim_token: raise ValueError("No claim_token available. Run the claim test first.") sent_dataframe = mark_claimed_batch_sent( engine=engine, claim_t` | Displays a notebook-facing result for inspection. |
| `else: print("MARK_SENT_TEST_FLAG is False. Skipping mark-sent test.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 16 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `after`
- `available`
- `batch`
- `be`
- `claim`
- `claimed`
- `count`
- `else`
- `failed`
- `first`
- `head`
- `It`
- `mark`
- `mark_claimed_batch_failed`
- `MARK_FAILED_TEST_FLAG`
- `Marked`
- `No`
- `on`
- `only`
- `Optional`

### Outputs

- `claim_token`
- `engine`
- `error_message`
- `failed_dataframe`
- `schema`
- `table_name`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Optional: mark the claimed batch as failed`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# WARNING:`: Documents the purpose or boundary of the surrounding notebook step.
- `# This should not be run after mark-sent on the same claim token.`: Documents the purpose or boundary of the surrounding notebook step.
- `# It only updates rows still in queue_status = 'claimed'.`: Documents the purpose or boundary of the surrounding notebook step.
- `failed_dataframe = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if MARK_FAILED_TEST_FLAG: if not claim_token: raise ValueError("No claim_token available. Run the claim test first.") failed_dataframe = mark_claimed_batch_failed( engine=engine, c`: Displays a notebook-facing result for inspection.
- `else: print("MARK_FAILED_TEST_FLAG is False. Skipping mark-failed test.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `head`
- `mark_claimed_batch_failed`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Optional: mark the claimed batch as failed` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# WARNING:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# This should not be run after mark-sent on the same claim token.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# It only updates rows still in queue_status = 'claimed'.` | Documents the purpose or boundary of the surrounding notebook step. |
| `failed_dataframe = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if MARK_FAILED_TEST_FLAG: if not claim_token: raise ValueError("No claim_token available. Run the claim test first.") failed_dataframe = mark_claimed_batch_failed( engine=engine, c` | Displays a notebook-facing result for inspection. |
| `else: print("MARK_FAILED_TEST_FLAG is False. Skipping mark-failed test.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 17 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `attempts`
- `by`
- `COUNT`
- `count`
- `created`
- `current`
- `DATASET_ID`
- `dataset_id`
- `else`
- `f`
- `failed`
- `failed_candidate_count`
- `flag`
- `head`
- `intentionally`
- `Keep`
- `max`
- `message`
- `messages`
- `Optional`

### Outputs

- `engine`
- `failed_before_requeue_dataframe`
- `max_send_attempts`
- `params`
- `requeued_failed_dataframe`
- `schema`
- `table_name`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Optional: requeue failed messages`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# WARNING:`: Documents the purpose or boundary of the surrounding notebook step.
- `# The current utility requeues failed rows by table/status and max attempts.`: Documents the purpose or boundary of the surrounding notebook step.
- `# Keep this flag False unless you intentionally created failed rows in this run.`: Documents the purpose or boundary of the surrounding notebook step.
- `requeued_failed_dataframe = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if REQUEUE_FAILED_TEST_FLAG: failed_before_requeue_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS failed_candidate_count FROM "{SCHEMA}"."{SEND_QUEUE_TABLE_NAME}" `: Loads input data, configuration, or artifacts required by the current stage.
- `else: print("REQUEUE_FAILED_TEST_FLAG is False. Skipping failed-message requeue test.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `head`
- `read_sql_dataframe`
- `requeue_failed_messages`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Optional: requeue failed messages` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# WARNING:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The current utility requeues failed rows by table/status and max attempts.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Keep this flag False unless you intentionally created failed rows in this run.` | Documents the purpose or boundary of the surrounding notebook step. |
| `requeued_failed_dataframe = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if REQUEUE_FAILED_TEST_FLAG: failed_before_requeue_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS failed_candidate_count FROM "{SCHEMA}"."{SEND_QUEUE_TABLE_NAME}" ` | Loads input data, configuration, or artifacts required by the current stage. |
| `else: print("REQUEUE_FAILED_TEST_FLAG is False. Skipping failed-message requeue test.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 18 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `are`
- `by`
- `claim`
- `claimed`
- `claimed_at`
- `claims`
- `COUNT`
- `count`
- `current`
- `DATASET_ID`
- `dataset_id`
- `else`
- `f`
- `head`
- `intentionally`
- `interval`
- `Keep`
- `minutes`
- `now`
- `NULL`

### Outputs

- `engine`
- `max_send_attempts`
- `params`
- `released_stale_dataframe`
- `schema`
- `stale_after_minutes`
- `stale_before_release_dataframe`
- `table_name`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Optional: release stale claims`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# WARNING:`: Documents the purpose or boundary of the surrounding notebook step.
- `# The current utility releases stale claimed rows by table/status/time.`: Documents the purpose or boundary of the surrounding notebook step.
- `# Keep this False unless you are intentionally testing stale claim recovery.`: Documents the purpose or boundary of the surrounding notebook step.
- `released_stale_dataframe = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if RELEASE_STALE_CLAIMS_TEST_FLAG: stale_before_release_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS stale_candidate_count FROM "{SCHEMA}"."{SEND_QUEUE_TABLE_NAM`: Loads input data, configuration, or artifacts required by the current stage.
- `else: print("RELEASE_STALE_CLAIMS_TEST_FLAG is False. Skipping stale-claim release test.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `head`
- `now`
- `read_sql_dataframe`
- `release_stale_claims`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Optional: release stale claims` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# WARNING:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The current utility releases stale claimed rows by table/status/time.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Keep this False unless you are intentionally testing stale claim recovery.` | Documents the purpose or boundary of the surrounding notebook step. |
| `released_stale_dataframe = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if RELEASE_STALE_CLAIMS_TEST_FLAG: stale_before_release_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS stale_candidate_count FROM "{SCHEMA}"."{SEND_QUEUE_TABLE_NAM` | Loads input data, configuration, or artifacts required by the current stage. |
| `else: print("RELEASE_STALE_CLAIMS_TEST_FLAG is False. Skipping stale-claim release test.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 19 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `get_send_queue_status_counts`
- `SEND_QUEUE_TABLE_NAME`

### Outputs

- `engine`
- `schema`
- `status_dataframe`
- `table_name`

### Key Operations

- `status_dataframe = get_send_queue_status_counts( engine=engine, schema=SCHEMA, table_name=SEND_QUEUE_TABLE_NAME,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(status_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `get_send_queue_status_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `status_dataframe = get_send_queue_status_counts( engine=engine, schema=SCHEMA, table_name=SEND_QUEUE_TABLE_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(status_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 20 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `already`
- `are`
- `but`
- `claimed`
- `COUNT`
- `dataset`
- `dataset_id`
- `DATASET_ID`
- `does`
- `else`
- `engine`
- `execute_sql`
- `f`
- `It`
- `marked`
- `No`
- `NULL`
- `only`
- `pending`
- `producer_sent_at`

### Outputs

- `before_reset_dataframe`
- `claim_token`
- `claimed_at`
- `params`
- `producer_delivery_error`
- `producer_delivery_status`
- `producer_topic`
- `producer_worker_id`
- `queue_status`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Reset claimed-but-unsent test rows`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# This resets only rows for this dataset/run that are still claimed and unsent.`: Documents the purpose or boundary of the surrounding notebook step.
- `# It does not reset rows already marked as sent.`: Documents the purpose or boundary of the surrounding notebook step.
- `if RESET_TEST_CLAIMS_FLAG: before_reset_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS reset_candidate_count FROM "{SCHEMA}"."{SEND_QUEUE_TABLE_NAME}" WHERE queue_`: Loads input data, configuration, or artifacts required by the current stage.
- `else: print("RESET_TEST_CLAIMS_FLAG is False. No claimed rows reset.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `execute_sql`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Reset claimed-but-unsent test rows` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# This resets only rows for this dataset/run that are still claimed and unsent.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# It does not reset rows already marked as sent.` | Documents the purpose or boundary of the surrounding notebook step. |
| `if RESET_TEST_CLAIMS_FLAG: before_reset_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS reset_candidate_count FROM "{SCHEMA}"."{SEND_QUEUE_TABLE_NAME}" WHERE queue_` | Loads input data, configuration, or artifacts required by the current stage. |
| `else: print("RESET_TEST_CLAIMS_FLAG is False. No claimed rows reset.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 21 — Code Reference

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
- `Post`
- `producer_sent_at`
- `queue`
- `queue_status`
- `read_sql_dataframe`
- `reset`

### Outputs

- `params`
- `post_reset_status_dataframe`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Post-reset queue validation`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `post_reset_status_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count, COUNT(*) FILTER (WHERE claim_token IS NULL) AS null_claim_token_count, CO`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(post_reset_status_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Post-reset queue validation` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `post_reset_status_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count, COUNT(*) FILTER (WHERE claim_token IS NULL) AS null_claim_token_count, CO` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(post_reset_status_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 22 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `after`
- `all`
- `before`
- `broader`
- `BY`
- `claimed`
- `claiming`
- `completed`
- `COUNT`
- `dataset`
- `DATASET_ID`
- `dataset_id`
- `else`
- `Emergency`
- `EMERGENCY_RESET_ALL_UNSENT_CLAIMS_FLAG`
- `engine`
- `execute_sql`
- `f`
- `GROUP`

### Outputs

- `claim_token`
- `claimed_at`
- `emergency_candidate_dataframe`
- `params`
- `producer_delivery_error`
- `producer_delivery_status`
- `producer_topic`
- `producer_worker_id`
- `queue_status`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Emergency reset: all unsent claimed rows for this dataset/run`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# WARNING:`: Documents the purpose or boundary of the surrounding notebook step.
- `# This is broader than the normal test reset.`: Documents the purpose or boundary of the surrounding notebook step.
- `# Use only if a test stopped after claiming rows and before resetting them.`: Documents the purpose or boundary of the surrounding notebook step.
- `if EMERGENCY_RESET_ALL_UNSENT_CLAIMS_FLAG: emergency_candidate_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count FROM "{SCHEMA}"."{SEND_QUEUE_`: Loads input data, configuration, or artifacts required by the current stage.
- `else: print("EMERGENCY_RESET_ALL_UNSENT_CLAIMS_FLAG is False. Emergency reset skipped.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `execute_sql`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Emergency reset: all unsent claimed rows for this dataset/run` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# WARNING:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# This is broader than the normal test reset.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Use only if a test stopped after claiming rows and before resetting them.` | Documents the purpose or boundary of the surrounding notebook step. |
| `if EMERGENCY_RESET_ALL_UNSENT_CLAIMS_FLAG: emergency_candidate_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count FROM "{SCHEMA}"."{SEND_QUEUE_` | Loads input data, configuration, or artifacts required by the current stage. |
| `else: print("EMERGENCY_RESET_ALL_UNSENT_CLAIMS_FLAG is False. Emergency reset skipped.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 23 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `cells`
- `check`
- `Check`
- `check_stage_7_readiness`
- `clean`
- `enough`
- `Queue`
- `queue`
- `readiness`
- `Ready`
- `ready_for_stage_7`
- `repair`
- `rerun`
- `reset`
- `Stage`
- `stage_7_readiness_dataframe`
- `the`
- `then`
- `this`
- `Use`

### Outputs

- `dataset_id`
- `engine`
- `run_id`
- `schema`
- `send_queue_table_name`

### Key Operations

- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Check whether queue is clean enough for Stage 7`: Documents the purpose or boundary of the surrounding notebook step.
- `# -----------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `ready_for_stage_7, stage_7_readiness_dataframe = check_stage_7_readiness( engine=engine, schema=SCHEMA, send_queue_table_name=SEND_QUEUE_TABLE_NAME, dataset_id=DATASET_ID, run_id=R`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_7_readiness_dataframe)`: Displays a notebook-facing result for inspection.
- `print("Ready for Stage 7:", ready_for_stage_7)`: Displays a notebook-facing result for inspection.
- `if not ready_for_stage_7: print( "Queue is not clean for Stage 7. Use the 6.5 reset/repair cells, " "then rerun this readiness check." )`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `check_stage_7_readiness`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Check whether queue is clean enough for Stage 7` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `ready_for_stage_7, stage_7_readiness_dataframe = check_stage_7_readiness( engine=engine, schema=SCHEMA, send_queue_table_name=SEND_QUEUE_TABLE_NAME, dataset_id=DATASET_ID, run_id=R` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_7_readiness_dataframe)` | Displays a notebook-facing result for inspection. |
| `print("Ready for Stage 7:", ready_for_stage_7)` | Displays a notebook-facing result for inspection. |
| `if not ready_for_stage_7: print( "Queue is not clean for Stage 7. Use the 6.5 reset/repair cells, " "then rerun this readiness check." )` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 24 — Code Reference

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
