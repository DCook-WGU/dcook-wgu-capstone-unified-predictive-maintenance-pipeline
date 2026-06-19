# Notebook Code Reference: synthetic_08_kafka_consumer_adapter

Notebook path:

`notebooks/synthetic/synthetic_08_kafka_consumer_adapter.ipynb`

## Notebook Purpose

This notebook documents and runs a focused support step in the synthetic pump telemetry workflow.

Notebook stage:

`Synthetic support`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01, Code Cell 02, Code Cell 03 |
| Single Batch | Code Cell 04 |
| Loop Batches | Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08, Code Cell 09, Code Cell 10, Code Cell 11, Code Cell 12, Code Cell 13 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `core`
- `database`
- `env_bool`
- `env_float`
- `env_helpers`
- `env_int`
- `env_optional_int`
- `env_str`
- `get_engine_from_env`
- `kafka_consumer_adapter`
- `os`
- `pipeline`
- `postgres`
- `read_sql_dataframe`
- `run_kafka_consumer_to_postgres_loop`
- `run_kafka_consumer_to_postgres_once`
- `synthetic`
- `utils`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.kafka_consumer_adapter import ( run_kafka_consumer_to_postgres_once, run_kafka_consumer_to_postgres_loop,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.core.env_helpers import ( env_bool, env_float, env_int, env_optional_int, env_str,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.kafka_consumer_adapter import ( run_kafka_consumer_to_postgres_once, run_kafka_consumer_to_postgres_loop,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.env_helpers import ( env_bool, env_float, env_int, env_optional_int, env_str,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `batch`
- `batches`
- `capstone`
- `CAPSTONE_SCHEMA`
- `config`
- `consumer`
- `CONSUMER_AUTO_OFFSET_RESET`
- `CONSUMER_IDLE_SLEEP_SECONDS`
- `CONSUMER_MAX_BATCHES_LIMIT`
- `CONSUMER_SCHEMA`
- `CONSUMER_STOP_ON_EMPTY`
- `CONSUMER_TARGET_TABLE`
- `consumer_worker_001`
- `dataset`
- `earliest`
- `empty`
- `env_bool`
- `env_float`
- `env_int`
- `env_optional_int`

### Outputs

- `aliases`
- `AUTO_OFFSET_RESET`
- `CONSUMER_BATCH_SIZE`
- `CONSUMER_DESTINATION_TABLE_NAME`
- `CONSUMER_GROUP_ID`
- `CONSUMER_POLL_TIMEOUT_SECONDS`
- `CONSUMER_WORKER_ID`
- `DATASET_ID`
- `default`
- `DEFAULT_CONSUMER_BATCH_SIZE`
- `IDLE_SLEEP_SECONDS`
- `MAX_BATCHES`
- `NUMBER_OF_SENSORS`
- `OBSERVATION_BATCH_SIZE`
- `RUN_ID`
- `SCHEMA`
- `STOP_ON_EMPTY_FLAG`
- `TOPIC`

### Key Operations

- `SCHEMA = env_str( "CAPSTONE_SCHEMA", "capstone", aliases=("CONSUMER_SCHEMA",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `DATASET_ID = env_str( "SYNTHETIC_DATASET_ID", "pump_synthetic_v1", aliases=("DATASET_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `RUN_ID = env_str( "SYNTHETIC_RUN_ID", "synthetic_run_001", aliases=("RUN_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `NUMBER_OF_SENSORS = env_int("SYNTHETIC_SENSOR_COUNT", 52)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `OBSERVATION_BATCH_SIZE = env_int( "OBSERVATION_BATCH_SIZE", 500, aliases=("SYNTHETIC_OBSERVATION_BATCH_SIZE",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `TOPIC = env_str( "SYNTHETIC_KAFKA_TOPIC", "pump.telemetry.synthetic", aliases=("KAFKA_TOPIC",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `CONSUMER_DESTINATION_TABLE_NAME = env_str( "SYNTHETIC_CONSUMED_MESSAGES_TABLE", "synthetic_sensor_messages_consumed_stage", aliases=("CONSUMER_TARGET_TABLE",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `env_bool`
- `env_float`
- `env_int`
- `env_optional_int`
- `env_str`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `SCHEMA = env_str( "CAPSTONE_SCHEMA", "capstone", aliases=("CONSUMER_SCHEMA",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DATASET_ID = env_str( "SYNTHETIC_DATASET_ID", "pump_synthetic_v1", aliases=("DATASET_ID",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `RUN_ID = env_str( "SYNTHETIC_RUN_ID", "synthetic_run_001", aliases=("RUN_ID",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `NUMBER_OF_SENSORS = env_int("SYNTHETIC_SENSOR_COUNT", 52)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `OBSERVATION_BATCH_SIZE = env_int( "OBSERVATION_BATCH_SIZE", 500, aliases=("SYNTHETIC_OBSERVATION_BATCH_SIZE",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `TOPIC = env_str( "SYNTHETIC_KAFKA_TOPIC", "pump.telemetry.synthetic", aliases=("KAFKA_TOPIC",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CONSUMER_DESTINATION_TABLE_NAME = env_str( "SYNTHETIC_CONSUMED_MESSAGES_TABLE", "synthetic_sensor_messages_consumed_stage", aliases=("CONSUMER_TARGET_TABLE",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CONSUMER_GROUP_ID = env_str( "SYNTHETIC_CONSUMER_GROUP_ID", "synthetic-telemetry-consumer-group", aliases=("KAFKA_CONSUMER_GROUP_ID", "CONSUMER_GROUP_ID"),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CONSUMER_WORKER_ID = env_str( "SYNTHETIC_CONSUMER_WORKER_ID", "consumer_worker_001", aliases=("CONSUMER_WORKER_ID",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DEFAULT_CONSUMER_BATCH_SIZE = OBSERVATION_BATCH_SIZE * NUMBER_OF_SENSORS` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONSUMER_BATCH_SIZE = env_int( "CONSUMER_BATCH_SIZE", DEFAULT_CONSUMER_BATCH_SIZE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CONSUMER_POLL_TIMEOUT_SECONDS = env_float( "CONSUMER_POLL_TIMEOUT_SECONDS", 1.0,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `AUTO_OFFSET_RESET = env_str( "CONSUMER_AUTO_OFFSET_RESET", "earliest", aliases=("SYNTHETIC_AUTO_OFFSET_RESET",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `MAX_BATCHES = env_optional_int( "CONSUMER_MAX_BATCHES_LIMIT", default=100000,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `IDLE_SLEEP_SECONDS = env_float( "CONSUMER_IDLE_SLEEP_SECONDS", 0.0,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STOP_ON_EMPTY_FLAG = env_bool( "CONSUMER_STOP_ON_EMPTY", True,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Kafka consumer config")` | Displays a notebook-facing result for inspection. |
| `print("schema:", SCHEMA)` | Displays a notebook-facing result for inspection. |
| `print("dataset id:", DATASET_ID)` | Displays a notebook-facing result for inspection. |
| `print("run id:", RUN_ID)` | Displays a notebook-facing result for inspection. |
| `print("topic:", TOPIC)` | Displays a notebook-facing result for inspection. |
| `print("target table:", CONSUMER_DESTINATION_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("consumer group id:", CONSUMER_GROUP_ID)` | Displays a notebook-facing result for inspection. |
| `print("consumer worker id:", CONSUMER_WORKER_ID)` | Displays a notebook-facing result for inspection. |
| `print("observation batch size:", OBSERVATION_BATCH_SIZE)` | Displays a notebook-facing result for inspection. |
| `print("message batch size:", CONSUMER_BATCH_SIZE)` | Displays a notebook-facing result for inspection. |
| `print("max batches:", MAX_BATCHES)` | Displays a notebook-facing result for inspection. |
| `print("stop on empty:", STOP_ON_EMPTY_FLAG)` | Displays a notebook-facing result for inspection. |

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

## Code Cell 04 — Single Batch

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `CONSUMER_BATCH_SIZE`
- `CONSUMER_DESTINATION_TABLE_NAME`
- `CONSUMER_POLL_TIMEOUT_SECONDS`
- `run_kafka_consumer_to_postgres_once`

### Outputs

- `auto_offset_reset`
- `consumer_group_id`
- `consumer_worker_id`
- `engine`
- `max_messages`
- `poll_timeout_seconds`
- `result`
- `schema`
- `table_name`
- `topic`

### Key Operations

- `result = run_kafka_consumer_to_postgres_once( engine=engine, topic=TOPIC, schema=SCHEMA, table_name=CONSUMER_DESTINATION_TABLE_NAME, max_messages=CONSUMER_BATCH_SIZE, poll_timeout_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(result)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `run_kafka_consumer_to_postgres_once`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `result = run_kafka_consumer_to_postgres_once( engine=engine, topic=TOPIC, schema=SCHEMA, table_name=CONSUMER_DESTINATION_TABLE_NAME, max_messages=CONSUMER_BATCH_SIZE, poll_timeout_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(result)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 05 — Loop Batches

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `CONSUMER_BATCH_SIZE`
- `CONSUMER_DESTINATION_TABLE_NAME`
- `CONSUMER_POLL_TIMEOUT_SECONDS`
- `run_kafka_consumer_to_postgres_loop`
- `STOP_ON_EMPTY_FLAG`

### Outputs

- `auto_offset_reset`
- `consumer_group_id`
- `consumer_worker_id`
- `engine`
- `idle_sleep_seconds`
- `max_batches`
- `max_messages`
- `poll_timeout_seconds`
- `results`
- `schema`
- `stop_on_empty`
- `table_name`
- `topic`

### Key Operations

- `results = run_kafka_consumer_to_postgres_loop( engine=engine, topic=TOPIC, schema=SCHEMA, table_name=CONSUMER_DESTINATION_TABLE_NAME, max_messages=CONSUMER_BATCH_SIZE, poll_timeout`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(results)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `run_kafka_consumer_to_postgres_loop`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `results = run_kafka_consumer_to_postgres_loop( engine=engine, topic=TOPIC, schema=SCHEMA, table_name=CONSUMER_DESTINATION_TABLE_NAME, max_messages=CONSUMER_BATCH_SIZE, poll_timeout` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(results)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 06 — Loop Batches

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `CONSUMER_DESTINATION_TABLE_NAME`
- `consumer_received_at`
- `COUNT`
- `DISTINCT`
- `distinct_kafka_messages`
- `distinct_observation_count`
- `engine`
- `f`
- `kafka_offset`
- `kafka_partition`
- `kafka_topic`
- `MAX`
- `max_consumer_received_at`
- `MIN`
- `min_consumer_received_at`
- `observation_index`
- `read_sql_dataframe`
- `row_count`
- `SCHEMA`
- `SELECT`

### Outputs

- `validation_dataframe`

### Key Operations

- `validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS row_count, COUNT(DISTINCT observation_index) AS distinct_observation_count, COUNT(DISTINCT kafka_topic \|\|`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `MAX`
- `MIN`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS row_count, COUNT(DISTINCT observation_index) AS distinct_observation_count, COUNT(DISTINCT kafka_topic \|\|` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 07 — Loop Batches

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
- `NULL`
- `null_message_key_count`
- `null_sensor_value_count`
- `numeric`

### Outputs

- `params`
- `stage_8_progress_dataframe`

### Key Operations

- `stage_8_progress_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS consumed_row_count, 11700000 - COUNT(*) AS remaining_message_count, ROUND((COUNT(*)::numeric / 1170`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_8_progress_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `MAX`
- `MIN`
- `read_sql_dataframe`
- `ROUND`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_8_progress_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS consumed_row_count, 11700000 - COUNT(*) AS remaining_message_count, ROUND((COUNT(*)::numeric / 1170` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_8_progress_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 08 — Loop Batches

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

## Code Cell 09 — Loop Batches

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

## Code Cell 10 — Loop Batches

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

## Code Cell 11 — Loop Batches

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

## Code Cell 12 — Loop Batches

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

## Code Cell 13 — Loop Batches

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

