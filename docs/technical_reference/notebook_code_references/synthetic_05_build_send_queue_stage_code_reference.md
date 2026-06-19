# Notebook Code Reference: synthetic_05_build_send_queue_stage

Notebook path:

`notebooks/synthetic/synthetic_05_build_send_queue_stage.ipynb`

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

- `build_sensor_messages_send_queue`
- `build_sensor_messages_send_queue_sql_native`
- `core`
- `database`
- `env_helpers`
- `env_int`
- `env_str`
- `get_engine_from_env`
- `os`
- `pipeline`
- `postgres`
- `read_sql_dataframe`
- `send_queue_stage_writer`
- `synthetic`
- `utils`
- `validate_sensor_messages_send_queue`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.synthetic.pipeline.send_queue_stage_writer import ( build_sensor_messages_send_queue, build_sensor_messages_send_queue_sql_native, validate_sensor_messages_send_queue,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.core.env_helpers import ( env_int, env_str,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.send_queue_stage_writer import ( build_sensor_messages_send_queue, build_sensor_messages_send_queue_sql_native, validate_sensor_messages_send_queue,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.env_helpers import ( env_int, env_str,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `capstone`
- `CAPSTONE_SCHEMA`
- `config`
- `default`
- `env_int`
- `env_str`
- `kafka_producer`
- `pending`
- `PRODUCER_QUEUE_TABLE`
- `pump_synthetic_v1`
- `queue`
- `replace`
- `send`
- `source`
- `status`
- `Synthetic`
- `SYNTHETIC_DATASET_ID`
- `SYNTHETIC_QUEUE_OWNER_ROLE`
- `synthetic_run_001`
- `SYNTHETIC_RUN_ID`

### Outputs

- `aliases`
- `APPLY_OWNER_AND_GRANTS_FLAG`
- `CHUNK_SIZE`
- `DATASET_ID`
- `IF_EXISTS_FLAG`
- `OBSERVATION_BATCH_SIZE`
- `QUEUE_OWNER_ROLE`
- `QUEUE_STATUS_DEFAULT`
- `RUN_ID`
- `SCHEMA`
- `SEND_QUEUE_DESTINATION_TABLE_NAME`
- `SEND_QUEUE_SOURCE_TABLE_NAME`

### Key Operations

- `SCHEMA = env_str("CAPSTONE_SCHEMA", "capstone")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_ID = env_str( "SYNTHETIC_DATASET_ID", "pump_synthetic_v1", aliases=("DATASET_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `RUN_ID = env_str( "SYNTHETIC_RUN_ID", "synthetic_run_001", aliases=("RUN_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `SEND_QUEUE_SOURCE_TABLE_NAME = env_str( "SYNTHETIC_SENSOR_MESSAGES_TABLE", "synthetic_sensor_messages_stage",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `SEND_QUEUE_DESTINATION_TABLE_NAME = env_str( "SYNTHETIC_SEND_QUEUE_TABLE", "synthetic_sensor_messages_send_queue", aliases=("PRODUCER_QUEUE_TABLE",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `IF_EXISTS_FLAG = "replace"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `QUEUE_STATUS_DEFAULT = "pending"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `QUEUE_OWNER_ROLE = env_str("SYNTHETIC_QUEUE_OWNER_ROLE", "kafka_producer")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
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
| `SEND_QUEUE_SOURCE_TABLE_NAME = env_str( "SYNTHETIC_SENSOR_MESSAGES_TABLE", "synthetic_sensor_messages_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SEND_QUEUE_DESTINATION_TABLE_NAME = env_str( "SYNTHETIC_SEND_QUEUE_TABLE", "synthetic_sensor_messages_send_queue", aliases=("PRODUCER_QUEUE_TABLE",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `IF_EXISTS_FLAG = "replace"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `QUEUE_STATUS_DEFAULT = "pending"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `QUEUE_OWNER_ROLE = env_str("SYNTHETIC_QUEUE_OWNER_ROLE", "kafka_producer")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `APPLY_OWNER_AND_GRANTS_FLAG = False` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CHUNK_SIZE = env_int("SYNTHETIC_SEND_QUEUE_CHUNK_SIZE", 50000)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `OBSERVATION_BATCH_SIZE = env_int("OBSERVATION_BATCH_SIZE", 500)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Synthetic send queue config")` | Displays a notebook-facing result for inspection. |
| `print("schema:", SCHEMA)` | Displays a notebook-facing result for inspection. |
| `print("source table:", SEND_QUEUE_SOURCE_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("target table:", SEND_QUEUE_DESTINATION_TABLE_NAME)` | Displays a notebook-facing result for inspection. |
| `print("queue status default:", QUEUE_STATUS_DEFAULT)` | Displays a notebook-facing result for inspection. |

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

- `NUMBER_OF_SENSORS = int( read_sql_dataframe( engine, f""" SELECT COUNT(DISTINCT sensor_index) AS sensor_count FROM "{SCHEMA}"."{SEND_QUEUE_SOURCE_TABLE_NAME}" WHERE dataset_id = :d`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `NUMBER_OF_SENSORS = int( read_sql_dataframe( engine, f""" SELECT COUNT(DISTINCT sensor_index) AS sensor_count FROM "{SCHEMA}"."{SEND_QUEUE_SOURCE_TABLE_NAME}" WHERE dataset_id = :d` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PRODUCER_BATCH_SIZE = OBSERVATION_BATCH_SIZE * NUMBER_OF_SENSORS` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Derived claim batch sizing")` | Displays a notebook-facing result for inspection. |
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
- `build_sensor_messages_send_queue_sql_native`
- `Built`
- `IF_EXISTS_FLAG`
- `native`
- `queue`
- `send`
- `SEND_QUEUE_DESTINATION_TABLE_NAME`
- `SEND_QUEUE_SOURCE_TABLE_NAME`
- `SQL`
- `table`

### Outputs

- `apply_owner_and_grants`
- `enable_timing_logging`
- `engine`
- `if_exists`
- `queue_owner_role`
- `queue_status_default`
- `schema`
- `send_queue_table_name`
- `source_table`
- `target_table`

### Key Operations

- `send_queue_table_name = build_sensor_messages_send_queue_sql_native( engine=engine, schema=SCHEMA, source_table=SEND_QUEUE_SOURCE_TABLE_NAME, target_table=SEND_QUEUE_DESTINATION_TA`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Built SQL-native send queue table:", send_queue_table_name)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_sensor_messages_send_queue_sql_native`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `send_queue_table_name = build_sensor_messages_send_queue_sql_native( engine=engine, schema=SCHEMA, source_table=SEND_QUEUE_SOURCE_TABLE_NAME, target_table=SEND_QUEUE_DESTINATION_TA` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Built SQL-native send queue table:", send_queue_table_name)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 06 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `APPLY_OWNER_AND_GRANTS_FLAG`
- `build_sensor_messages_send_queue`
- `Built`
- `IF_EXISTS_FLAG`
- `queue`
- `SEND_QUEUE_DESTINATION_TABLE_NAME`
- `SEND_QUEUE_SOURCE_TABLE_NAME`
- `table`

### Outputs

- `apply_owner_and_grants`
- `chunk_size`
- `engine`
- `if_exists`
- `queue_owner_role`
- `queue_status_default`
- `schema`
- `send_queue_table_name`
- `source_table`
- `target_table`

### Key Operations

- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `send_queue_table_name = build_sensor_messages_send_queue( engine=engine, schema=SCHEMA, source_table=SEND_QUEUE_SOURCE_TABLE_NAME, target_table=SEND_QUEUE_DESTINATION_TABLE_NAME, i`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Built queue table:", send_queue_table_name)`: Displays a notebook-facing result for inspection.
- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `build_sensor_messages_send_queue`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `send_queue_table_name = build_sensor_messages_send_queue( engine=engine, schema=SCHEMA, source_table=SEND_QUEUE_SOURCE_TABLE_NAME, target_table=SEND_QUEUE_DESTINATION_TABLE_NAME, i` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Built queue table:", send_queue_table_name)` | Displays a notebook-facing result for inspection. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 07 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `SEND_QUEUE_DESTINATION_TABLE_NAME`
- `validate_sensor_messages_send_queue`

### Outputs

- `engine`
- `schema`
- `table_name`
- `validation_dataframe`

### Key Operations

- `validation_dataframe = validate_sensor_messages_send_queue( engine=engine, schema=SCHEMA, table_name=SEND_QUEUE_DESTINATION_TABLE_NAME,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `validate_sensor_messages_send_queue`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `validation_dataframe = validate_sensor_messages_send_queue( engine=engine, schema=SCHEMA, table_name=SEND_QUEUE_DESTINATION_TABLE_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 08 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `claim_token`
- `COUNT`
- `CROSS`
- `DISTINCT`
- `distinct_message_key_count`
- `distinct_observation_count`
- `distinct_sensor_name_count`
- `duplicate_message_key_count`
- `engine`
- `f`
- `FILTER`
- `JOIN`
- `message_key`
- `NULL`
- `observation_index`
- `pending`
- `pending_count`
- `producer_sent_at`
- `queue_counts`
- `queue_row_count`

### Outputs

- `stage_05_validation_dataframe`

### Key Operations

- `stage_05_validation_dataframe = read_sql_dataframe( engine, f""" WITH source_counts AS ( SELECT COUNT(*) AS source_message_row_count FROM "{SCHEMA}"."{SEND_QUEUE_SOURCE_TABLE_NAME}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_05_validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `AS`
- `COUNT`
- `display`
- `FILTER`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_05_validation_dataframe = read_sql_dataframe( engine, f""" WITH source_counts AS ( SELECT COUNT(*) AS source_message_row_count FROM "{SCHEMA}"."{SEND_QUEUE_SOURCE_TABLE_NAME}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_05_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 09 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `claim_token`
- `claimed_at`
- `engine`
- `f`
- `LIMIT`
- `message_key`
- `message_sequence_index`
- `observation_index`
- `observation_timestamp`
- `ORDER`
- `producer_ack_at`
- `producer_delivery_error`
- `producer_delivery_status`
- `producer_sent_at`
- `producer_topic`
- `producer_worker_id`
- `queue_status`
- `queued_at`
- `read_sql_dataframe`

### Outputs

- `sample_dataframe`

### Key Operations

- `sample_dataframe = read_sql_dataframe( engine, f""" SELECT message_key, observation_index, observation_timestamp, message_sequence_index, sensor_name, sensor_index, sensor_value, q`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(sample_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `sample_dataframe = read_sql_dataframe( engine, f""" SELECT message_key, observation_index, observation_timestamp, message_sequence_index, sensor_name, sensor_index, sensor_value, q` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(sample_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 10 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `capstone`
- `engine`
- `head`
- `LIMIT`
- `message_sequence_index`
- `NULL`
- `observation_index`
- `ORDER`
- `pending`
- `producer_sent_at`
- `queue_status`
- `read_sql_dataframe`
- `SELECT`
- `sensor_index`
- `synthetic_sensor_messages_send_queue`
- `WHERE`

### Outputs

- `pending_dataframe`

### Key Operations

- `pending_dataframe = read_sql_dataframe( engine, """ SELECT * FROM capstone.synthetic_sensor_messages_send_queue WHERE queue_status = 'pending' AND producer_sent_at IS NULL ORDER BY`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(pending_dataframe.head(100))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `head`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `pending_dataframe = read_sql_dataframe( engine, """ SELECT * FROM capstone.synthetic_sensor_messages_send_queue WHERE queue_status = 'pending' AND producer_sent_at IS NULL ORDER BY` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(pending_dataframe.head(100))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 11 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `COUNT`
- `DESC`
- `engine`
- `f`
- `GROUP`
- `ORDER`
- `queue_status`
- `read_sql_dataframe`
- `row_count`
- `SCHEMA`
- `SELECT`
- `SEND_QUEUE_DESTINATION_TABLE_NAME`

### Outputs

- `queue_status_distribution_dataframe`

### Key Operations

- `queue_status_distribution_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count FROM "{SCHEMA}"."{SEND_QUEUE_DESTINATION_TABLE_NAME}" GROUP BY que`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(queue_status_distribution_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `queue_status_distribution_dataframe = read_sql_dataframe( engine, f""" SELECT queue_status, COUNT(*) AS row_count FROM "{SCHEMA}"."{SEND_QUEUE_DESTINATION_TABLE_NAME}" GROUP BY que` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(queue_status_distribution_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 12 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `claim_token`
- `count`
- `engine`
- `f`
- `head`
- `LIMIT`
- `MESSAGE_BATCH_SIZE`
- `message_key`
- `message_sequence_index`
- `observation_index`
- `ORDER`
- `pending`
- `Preview`
- `producer_sent_at`
- `queue_status`
- `read_sql_dataframe`
- `row`
- `SCHEMA`
- `SELECT`

### Outputs

- `producer_pickup_preview_dataframe`

### Key Operations

- `producer_pickup_preview_dataframe = read_sql_dataframe( engine, f""" SELECT message_key, observation_index, message_sequence_index, sensor_index, sensor_name, sensor_value, queue_s`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(producer_pickup_preview_dataframe.head(104))`: Displays a notebook-facing result for inspection.
- `print("Preview row count:", len(producer_pickup_preview_dataframe))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `head`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `producer_pickup_preview_dataframe = read_sql_dataframe( engine, f""" SELECT message_key, observation_index, message_sequence_index, sensor_index, sensor_name, sensor_value, queue_s` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(producer_pickup_preview_dataframe.head(104))` | Displays a notebook-facing result for inspection. |
| `print("Preview row count:", len(producer_pickup_preview_dataframe))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 13 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `engine`
- `f`
- `pg_tables`
- `read_sql_dataframe`
- `SCHEMA`
- `schema_name`
- `schemaname`
- `SELECT`
- `SEND_QUEUE_DESTINATION_TABLE_NAME`
- `table_name`
- `tablename`
- `tableowner`
- `WHERE`

### Outputs

- `ownership_dataframe`
- `params`

### Key Operations

- `ownership_dataframe = read_sql_dataframe( engine, f""" SELECT schemaname, tablename, tableowner FROM pg_tables WHERE schemaname = :schema_name AND tablename = :table_name """, para`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(ownership_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `ownership_dataframe = read_sql_dataframe( engine, f""" SELECT schemaname, tablename, tableowner FROM pg_tables WHERE schemaname = :schema_name AND tablename = :table_name """, para` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(ownership_dataframe)` | Displays a notebook-facing result for inspection. |

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

