# Notebook Code Reference: synthetic_all_in_one_wip_v1

Notebook path:

`notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`

## Status

Testing / support reference.

This notebook is documented because it remains in the repository as a work-in-progress synthetic workflow reference. It should not be treated as the preferred current implementation unless it is explicitly selected for testing, troubleshooting, or historical comparison.

## Notebook Purpose

This notebook documents and runs a focused support step in the synthetic pump telemetry workflow.

Notebook stage:

`Synthetic support`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08, Code Cell 09, Code Cell 10, Code Cell 11, Code Cell 12, Code Cell 13, Code Cell 14, Code Cell 15, Code Cell 16, Code Cell 17, Code Cell 18, Code Cell 19, Code Cell 20, Code Cell 21, Code Cell 22, Code Cell 23, Code Cell 24, Code Cell 25, Code Cell 26, Code Cell 27, Code Cell 28, Code Cell 29, Code Cell 30, Code Cell 31, Code Cell 32, Code Cell 33, Code Cell 34, Code Cell 35, Code Cell 36, Code Cell 37, Code Cell 38, Code Cell 39, Code Cell 40, Code Cell 41, Code Cell 42, Code Cell 43, Code Cell 44, Code Cell 45, Code Cell 46, Code Cell 47, Code Cell 48, Code Cell 49, Code Cell 50, Code Cell 51, Code Cell 52, Code Cell 53, Code Cell 54, Code Cell 55, Code Cell 56, Code Cell 57, Code Cell 58, Code Cell 59, Code Cell 60, Code Cell 61, Code Cell 62, Code Cell 63 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `append_truth_index`
- `bronze_handoff`
- `build_missingness_spec_from_truth_payload`
- `build_observations_premelt_stage`
- `build_postgres_url`
- `build_rebuild_comparison_stage`
- `build_sensor_message_payload`
- `build_sensor_messages_send_queue`
- `build_sensor_messages_stage`
- `build_sensor_messages_timestamped_stage`
- `build_synthetic_final_aligned_output_stage`
- `build_truth_config_block`
- `build_truth_record`
- `claim_pending_send_queue_batch`
- `config_loader`
- `configure_logging`
- `core`
- `database`
- `ensure_send_queue_runtime_columns`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `import gc`: Imports a dependency or project helper used by later cells.
- `import psutil`: Imports a dependency or project helper used by later cells.
- `import logging`: Imports a dependency or project helper used by later cells.
- `import json`: Imports a dependency or project helper used by later cells.
- `import numpy as np`: Imports a dependency or project helper used by later cells.
- `import inspect`: Imports a dependency or project helper used by later cells.
- `import numpy as np`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `import random`: Imports a dependency or project helper used by later cells.
- `from typing import Dict, List, Optional, Tuple, Any`: Imports a dependency or project helper used by later cells.
- `from pathlib import Path`: Imports a dependency or project helper used by later cells.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `import gc` | Imports a dependency or project helper used by later cells. |
| `import psutil` | Imports a dependency or project helper used by later cells. |
| `import logging` | Imports a dependency or project helper used by later cells. |
| `import json` | Imports a dependency or project helper used by later cells. |
| `import numpy as np` | Imports a dependency or project helper used by later cells. |
| `import inspect` | Imports a dependency or project helper used by later cells. |
| `import numpy as np` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `import random` | Imports a dependency or project helper used by later cells. |
| `from typing import Dict, List, Optional, Tuple, Any` | Imports a dependency or project helper used by later cells. |
| `from pathlib import Path` | Imports a dependency or project helper used by later cells. |
| `from typing import Optional, Dict, Any` | Imports a dependency or project helper used by later cells. |
| `from utils.core.paths import get_paths` | Imports a dependency or project helper used by later cells. |
| `from utils.core.file_io import save_data` | Imports a dependency or project helper used by later cells. |
| `from utils.core.logging import ( configure_logging, log_layer_paths,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.config_loader import ( load_pipeline_config, set_wandb_dir_from_config, export_config_snapshot, build_truth_config_block,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.truths import ( make_process_run_id, initialize_layer_truth, update_truth_section, build_truth_record, save_truth_record, append_truth_index, stamp_truth_columns, l` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe, build_postgres_url, execute_sql,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.layer_postgres import ( write_layer_dataframe, prepare_layer_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.generator.profiles import ( load_rich_profile_csv, load_and_merge_rich_profiles, load_correlation_pairs_csv, load_group_map_csv, load_fault_pairings_csv,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.generator.missingness import build_missingness_spec_from_truth_payload` | Imports a dependency or project helper used by later cells. |
| `from utils.synthetic.generator.export import export_synthetic_batch_to_parquet` | Imports a dependency or project helper used by later cells. |
| `from utils.synthetic.generator.generator import ( SyntheticGenerator, EpisodeSpec,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.generator.postgres_writer import ( ensure_sequence, reserve_next_batch_id, reserve_cycle_range, reset_sequence, reset_synthetic_sequences, write_stream_batch,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.premelt_stage_writer import ( build_observations_premelt_stage, validate_observations_premelt_stage,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.melt_stage_writer import ( build_sensor_messages_stage, validate_sensor_messages_stage,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.timestamp_stage_writer import ( ensure_simulation_timing_config_table, insert_simulation_timing_config, build_sensor_messages_timestamped_stage, valid` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.send_queue_stage_writer import ( build_sensor_messages_send_queue, validate_sensor_messages_send_queue,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.producer_queue_manager import ( ensure_send_queue_runtime_columns, ensure_simulation_state_control_table, upsert_simulation_state_control, read_simula` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.kafka_producer_adapter import ( run_send_queue_producer_once, run_send_queue_producer_loop, build_sensor_message_payload, json_dumps_safe` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.kafka_consumer_adapter import ( run_kafka_consumer_to_postgres_once, run_kafka_consumer_to_postgres_loop,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.row_rebuilder import ( rebuild_consumed_messages_to_observations` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.rebuild_comparison import ( build_rebuild_comparison_stage` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.final_aligned_output_writer import ( build_synthetic_final_aligned_output_stage` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.pipeline.bronze_handoff import ( handoff_final_aligned_observations_to_bronze, run_bronze_handoff_loop,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: Parquet output, SQL or medallion table write, truth record.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `def`
- `f`
- `GB`
- `getpid`
- `label`
- `memory`
- `memory_info`
- `os`
- `psutil`
- `rss`

### Outputs

- `log_memory`
- `memory_gb`
- `process`

### Key Operations

- `def memory_gb() -> float: process = psutil.Process(os.getpid()) return process.memory_info().rss / (1024 ** 3)`: Defines notebook-local logic used later in the notebook.
- `def log_memory(label: str) -> None: print(f"[memory] {label}: {memory_gb():.2f} GB")`: Defines notebook-local logic used later in the notebook.

Important functions or methods detected:
- `getpid`
- `log_memory`
- `memory_gb`
- `memory_info`
- `Process`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def memory_gb() -> float: process = psutil.Process(os.getpid()) return process.memory_info().rss / (1024 ** 3)` | Defines notebook-local logic used later in the notebook. |
| `def log_memory(label: str) -> None: print(f"[memory] {label}: {memory_gb():.2f} GB")` | Defines notebook-local logic used later in the notebook. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 03 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `abnormal`
- `Adapter`
- `Aligned`
- `batch`
- `BROKEN`
- `broken`
- `Bronze`
- `Build`
- `can`
- `cap`
- `capstone`
- `Comparison`
- `Consumer`
- `consumer`
- `consumer_worker_001`
- `control`
- `core`
- `created_at`
- `default`

### Outputs

- `ALLOW_OVERSHOOT`
- `ASSET_ID`
- `AUTO_OFFSET_RESET`
- `CHUNK_SIZE`
- `CLAIM_BATCH_SIZE`
- `CLIENT_ID`
- `COMPLETE_ONLY_FLAG`
- `CONSUMER_BATCH_MODE`
- `CONSUMER_DESTINATION_TABLE_NAME`
- `CONSUMER_GROUP_ID`
- `CONSUMER_MAX_MESSAGES`
- `CONSUMER_POLL_TIMEOUT_SECONDS`
- `CONSUMER_WORKER_ID`
- `DATASET`
- `DATASET_ID`
- `EPISODE_MAX_ROWS`
- `FLOAT_TOLERANCE`
- `FLUSH_TIMEOUT_SECONDS`
- `IDLE_SLEEP_SECONDS`
- `IF_EXISTS_FLAG`

### Key Operations

- `SCHEMA = str("capstone")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_ID = str("pump_synthetic_v1")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RUN_ID = str("premelt_run_001")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ASSET_ID = str("pump_asset_001")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `IF_EXISTS_FLAG = str("replace")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RANDOM_SEED = int(42)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `NUMBER_OF_SENSORS = int(52)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CHUNK_SIZE = int(10000)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SEND_QUEUE_MODE = "LOOP"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONSUMER_BATCH_MODE = "LOOP"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Generator`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `control`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `SCHEMA = str("capstone")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_ID = str("pump_synthetic_v1")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_ID = str("premelt_run_001")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ASSET_ID = str("pump_asset_001")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `IF_EXISTS_FLAG = str("replace")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RANDOM_SEED = int(42)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `NUMBER_OF_SENSORS = int(52)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CHUNK_SIZE = int(10000)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SEND_QUEUE_MODE = "LOOP"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONSUMER_BATCH_MODE = "LOOP"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Generator` | Documents the purpose or boundary of the surrounding notebook step. |
| `# --- Notebook params ---` | Documents the purpose or boundary of the surrounding notebook step. |
| `STAGE = "synthetic"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MODE = "train"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---- Mode switch ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `MODE = "batch" # "single" \| "batch"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TARGET_ROWS = 72_000` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MAX_EPISODES = 1_000_000 # safety cap` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---- policy knobs ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `EPISODE_MAX_ROWS = 3_000 # prevents monster episodes; forces multiple episodes in a 10k batch` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ALLOW_OVERSHOOT = False # if True, can overshoot when remaining can't fit minimum core` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---- failure rarity control (match real dataset frequency) ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Real dataset: ~7 failures per 250,000 rows => ~1 failure per 35,714 rows` | Documents the purpose or boundary of the surrounding notebook step. |
| `ROWS_PER_FAILURE = 250_000 / 7 # ~35714.2857` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Premelt - Stamping` | Documents the purpose or boundary of the surrounding notebook step. |
| `PREMELT_SOURCE_TABLE_NAME = str("synthetic_pump_stream")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PREMELT_DESTINATION_TABLE_NAME = str("synthetic_observations_premelt_stage")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Melting` | Documents the purpose or boundary of the surrounding notebook step. |
| `MELT_SOURCE_TABLE_NAME = str("synthetic_observations_premelt_stage")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MELT_DESTINATION_TABLE_NAME = str("synthetic_sensor_messages_stage")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Timestamping` | Documents the purpose or boundary of the surrounding notebook step. |
| `TIMESTAMP_CHUNK_SIZE = int(250000)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SIMULATION_TIME_CONFIG_TABLE_NAME = str("simulation_timing_config")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SIMULATION_START_DATETIME = str("2026-03-19 08:00:00+00:00")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SIMULATION_SAMPLING_INTERVAL_SECONDS = float(60.0)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TIMESTAMPED_SOURCE_TABLE_NAME = str("synthetic_sensor_messages_stage")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TIMESTAMPED_DESTINATION_TABLE_NAME = str("synthetic_sensor_messages_timestamped_stage")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Build Send Queue` | Documents the purpose or boundary of the surrounding notebook step. |
| `QUEUE_STATUS_DEFAULT = str("pending")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SEND_QUEUE_CHUNK_SIZE = int(250000)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SEND_QUEUE_SOURCE_TABLE_NAME = str("synthetic_sensor_messages_timestamped_stage")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SEND_QUEUE_DESTINATION_TABLE_NAME = str("synthetic_sensor_messages_send_queue")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Producer Queue Manager` | Documents the purpose or boundary of the surrounding notebook step. |
| `IS_ENABLED_FLAG = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PRODUCER_TOPIC = str("pump.telemetry.synthetic")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PRODUCER_BATCH_SIZE = int(5200)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PRODUCER_POLL_SECONDS = float(0.0)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PRODUCER_MAX_SEND_ATTEMPTS = int(3)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PRODUCER_WORKER_ID = str("producer_worker_001")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SIMULATION_TABLE_NAME = str("simulation_state_control")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SEND_QUEUE_TABLE_NAME = str("synthetic_sensor_messages_send_queue")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CLAIM_BATCH_SIZE = int(500)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STALE_CLAIM_RELEASE_MINUTES = int(15)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Producer Adapter` | Documents the purpose or boundary of the surrounding notebook step. |
| `CLIENT_ID = str("pump-telemetry-producer")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FLUSH_TIMEOUT_SECONDS = float(30.0)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `39 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 04 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Create`
- `get_engine_from_env`

### Outputs

- `engine`

### Key Operations

- `# Create Engine`: Documents the purpose or boundary of the surrounding notebook step.
- `engine = get_engine_from_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get_engine_from_env`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Create Engine` | Documents the purpose or boundary of the surrounding notebook step. |
| `engine = get_engine_from_env()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 05 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_observations_premelt_stage`
- `Built`
- `IF_EXISTS_FLAG`
- `PREMELT_DESTINATION_TABLE_NAME`
- `PREMELT_SOURCE_TABLE_NAME`
- `table`

### Outputs

- `asset_id`
- `dataset_id`
- `engine`
- `if_exists`
- `premelt_table_name`
- `run_id`
- `schema`
- `source_table`
- `target_table`

### Key Operations

- `premelt_table_name = build_observations_premelt_stage( engine=engine, schema=SCHEMA, source_table=PREMELT_SOURCE_TABLE_NAME, target_table=PREMELT_DESTINATION_TABLE_NAME, dataset_id`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Built table:", premelt_table_name)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_observations_premelt_stage`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `premelt_table_name = build_observations_premelt_stage( engine=engine, schema=SCHEMA, source_table=PREMELT_SOURCE_TABLE_NAME, target_table=PREMELT_DESTINATION_TABLE_NAME, dataset_id` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Built table:", premelt_table_name)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 06 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `PREMELT_DESTINATION_TABLE_NAME`
- `validate_observations_premelt_stage`

### Outputs

- `engine`
- `observation_validation_dataframe`
- `schema`
- `table_name`

### Key Operations

- `observation_validation_dataframe = validate_observations_premelt_stage( engine=engine, schema=SCHEMA, table_name=PREMELT_DESTINATION_TABLE_NAME,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(observation_validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `validate_observations_premelt_stage`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `observation_validation_dataframe = validate_observations_premelt_stage( engine=engine, schema=SCHEMA, table_name=PREMELT_DESTINATION_TABLE_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(observation_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 07 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `asset_id`
- `batch_id`
- `BY`
- `capstone`
- `dataset_id`
- `engine`
- `generated_row_id`
- `global_cycle_id`
- `Inspection`
- `is_telemetry_event`
- `LIMIT`
- `meta_episode_id`
- `observation_index`
- `ORDER`
- `phase`
- `producer_send_attempt`
- `read_sql_dataframe`
- `row_in_batch`
- `run_id`
- `SELECT`

### Outputs

- `inspection_dataframe`

### Key Operations

- `# Inspection`: Documents the purpose or boundary of the surrounding notebook step.
- `inspection_dataframe = read_sql_dataframe( engine, """ SELECT dataset_id, run_id, asset_id, generated_row_id, observation_index, batch_id, row_in_batch, global_cycle_id, stream_sta`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(inspection_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Inspection` | Documents the purpose or boundary of the surrounding notebook step. |
| `inspection_dataframe = read_sql_dataframe( engine, """ SELECT dataset_id, run_id, asset_id, generated_row_id, observation_index, batch_id, row_in_batch, global_cycle_id, stream_sta` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(inspection_dataframe)` | Displays a notebook-facing result for inspection. |

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

- `Create`
- `get_engine_from_env`

### Outputs

- `engine`

### Key Operations

- `# Create Engine`: Documents the purpose or boundary of the surrounding notebook step.
- `engine = get_engine_from_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get_engine_from_env`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Create Engine` | Documents the purpose or boundary of the surrounding notebook step. |
| `engine = get_engine_from_env()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 10 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `ensure_simulation_timing_config_table`
- `SIMULATION_TIME_CONFIG_TABLE_NAME`

### Outputs

- `engine`
- `schema`
- `table_name`

### Key Operations

- `ensure_simulation_timing_config_table( engine=engine, schema=SCHEMA, table_name=SIMULATION_TIME_CONFIG_TABLE_NAME,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `ensure_simulation_timing_config_table`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `ensure_simulation_timing_config_table( engine=engine, schema=SCHEMA, table_name=SIMULATION_TIME_CONFIG_TABLE_NAME,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 11 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `config`
- `insert_simulation_timing_config`
- `ready`
- `SIMULATION_SAMPLING_INTERVAL_SECONDS`
- `SIMULATION_TIME_CONFIG_TABLE_NAME`
- `Timing`

### Outputs

- `dataset_id`
- `deactivate_existing_for_run`
- `engine`
- `run_id`
- `sampling_interval_seconds`
- `schema`
- `set_active`
- `simulation_start_datetime`
- `table_name`

### Key Operations

- `insert_simulation_timing_config( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, simulation_start_datetime=SIMULATION_START_DATETIME, sampling_interval_seconds=SIMULATION_SAMP`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Timing config ready.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `insert_simulation_timing_config`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `insert_simulation_timing_config( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, simulation_start_datetime=SIMULATION_START_DATETIME, sampling_interval_seconds=SIMULATION_SAMP` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Timing config ready.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 12 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_sensor_messages_timestamped_stage`
- `Built`
- `IF_EXISTS_FLAG`
- `SIMULATION_TIME_CONFIG_TABLE_NAME`
- `table`
- `TIMESTAMP_CHUNK_SIZE`
- `TIMESTAMPED_DESTINATION_TABLE_NAME`
- `TIMESTAMPED_SOURCE_TABLE_NAME`

### Outputs

- `chunk_size`
- `dataset_id`
- `engine`
- `if_exists`
- `run_id`
- `schema`
- `source_table`
- `target_table`
- `timestamped_table_name`
- `timing_config_table`

### Key Operations

- `timestamped_table_name = build_sensor_messages_timestamped_stage( engine=engine, schema=SCHEMA, source_table=TIMESTAMPED_SOURCE_TABLE_NAME, target_table=TIMESTAMPED_DESTINATION_TAB`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Built table:", timestamped_table_name)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_sensor_messages_timestamped_stage`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `timestamped_table_name = build_sensor_messages_timestamped_stage( engine=engine, schema=SCHEMA, source_table=TIMESTAMPED_SOURCE_TABLE_NAME, target_table=TIMESTAMPED_DESTINATION_TAB` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Built table:", timestamped_table_name)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 13 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `TIMESTAMPED_DESTINATION_TABLE_NAME`
- `validate_sensor_messages_timestamped_stage`

### Outputs

- `engine`
- `schema`
- `table_name`
- `timestamp_validation_dataframe`

### Key Operations

- `timestamp_validation_dataframe = validate_sensor_messages_timestamped_stage( engine=engine, schema=SCHEMA, table_name=TIMESTAMPED_DESTINATION_TABLE_NAME,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(timestamp_validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `validate_sensor_messages_timestamped_stage`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `timestamp_validation_dataframe = validate_sensor_messages_timestamped_stage( engine=engine, schema=SCHEMA, table_name=TIMESTAMPED_DESTINATION_TABLE_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(timestamp_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 14 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `capstone`
- `engine`
- `LIMIT`
- `message_sequence_index`
- `meta_episode_id`
- `observation_index`
- `observation_timestamp`
- `ORDER`
- `phase`
- `read_sql_dataframe`
- `SELECT`
- `sensor_index`
- `sensor_name`
- `sensor_value`
- `stream_state`
- `synthetic_sensor_messages_timestamped_stage`

### Outputs

- `timetstamp_sample_dataframe`

### Key Operations

- `timetstamp_sample_dataframe = read_sql_dataframe( engine, """ SELECT observation_index, observation_timestamp, message_sequence_index, sensor_name, sensor_index, sensor_value, stre`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(timetstamp_sample_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `timetstamp_sample_dataframe = read_sql_dataframe( engine, """ SELECT observation_index, observation_timestamp, message_sequence_index, sensor_name, sensor_index, sensor_value, stre` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(timetstamp_sample_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 15 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `capstone`
- `COUNT`
- `DISTINCT`
- `distinct_timestamps_in_observation`
- `engine`
- `GROUP`
- `LIMIT`
- `MAX`
- `message_count`
- `MIN`
- `observation_index`
- `observation_timestamp`
- `observation_timestamp_max`
- `observation_timestamp_min`
- `ORDER`
- `read_sql_dataframe`
- `SELECT`
- `synthetic_sensor_messages_timestamped_stage`

### Outputs

- `timestamp_check_dataframe`

### Key Operations

- `timestamp_check_dataframe = read_sql_dataframe( engine, """ SELECT observation_index, COUNT(*) AS message_count, COUNT(DISTINCT observation_timestamp) AS distinct_timestamps_in_obs`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(timestamp_check_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `MAX`
- `MIN`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `timestamp_check_dataframe = read_sql_dataframe( engine, """ SELECT observation_index, COUNT(*) AS message_count, COUNT(DISTINCT observation_timestamp) AS distinct_timestamps_in_obs` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(timestamp_check_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 16 — Code Reference

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

## Code Cell 17 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Create`
- `get_engine_from_env`

### Outputs

- `engine`

### Key Operations

- `# Create Engine`: Documents the purpose or boundary of the surrounding notebook step.
- `engine = get_engine_from_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get_engine_from_env`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Create Engine` | Documents the purpose or boundary of the surrounding notebook step. |
| `engine = get_engine_from_env()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 18 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `after`
- `before`
- `chunk`
- `collect`
- `gc`
- `log_memory`
- `melt`
- `read`
- `write`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `log_memory("before read")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# read chunk`: Documents the purpose or boundary of the surrounding notebook step.
- `log_memory("after read")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# melt chunk`: Documents the purpose or boundary of the surrounding notebook step.
- `log_memory("after melt")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# write chunk`: Documents the purpose or boundary of the surrounding notebook step.
- `log_memory("after write")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gc.collect()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `log_memory("after gc")`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `collect`
- `log_memory`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `log_memory("before read")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# read chunk` | Documents the purpose or boundary of the surrounding notebook step. |
| `log_memory("after read")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# melt chunk` | Documents the purpose or boundary of the surrounding notebook step. |
| `log_memory("after melt")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# write chunk` | Documents the purpose or boundary of the surrounding notebook step. |
| `log_memory("after write")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gc.collect()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `log_memory("after gc")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 19 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_sensor_messages_stage`
- `IF_EXISTS_FLAG`
- `MELT_DESTINATION_TABLE_NAME`
- `MELT_SOURCE_TABLE_NAME`
- `NUMBER_OF_SENSORS`

### Outputs

- `chunk_size`
- `engine`
- `if_exists`
- `melt_table_name`
- `n_sensors`
- `random_seed`
- `schema`
- `source_table`
- `target_table`

### Key Operations

- `melt_table_name = build_sensor_messages_stage( engine=engine, schema=SCHEMA, source_table=MELT_SOURCE_TABLE_NAME, target_table=MELT_DESTINATION_TABLE_NAME, if_exists=IF_EXISTS_FLAG`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `build_sensor_messages_stage`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `melt_table_name = build_sensor_messages_stage( engine=engine, schema=SCHEMA, source_table=MELT_SOURCE_TABLE_NAME, target_table=MELT_DESTINATION_TABLE_NAME, if_exists=IF_EXISTS_FLAG` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 20 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `after`
- `before`
- `chunk`
- `collect`
- `gc`
- `log_memory`
- `melt`
- `read`
- `write`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `log_memory("before read")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# read chunk`: Documents the purpose or boundary of the surrounding notebook step.
- `log_memory("after read")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# melt chunk`: Documents the purpose or boundary of the surrounding notebook step.
- `log_memory("after melt")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# write chunk`: Documents the purpose or boundary of the surrounding notebook step.
- `log_memory("after write")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gc.collect()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `log_memory("after gc")`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `collect`
- `log_memory`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `log_memory("before read")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# read chunk` | Documents the purpose or boundary of the surrounding notebook step. |
| `log_memory("after read")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# melt chunk` | Documents the purpose or boundary of the surrounding notebook step. |
| `log_memory("after melt")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# write chunk` | Documents the purpose or boundary of the surrounding notebook step. |
| `log_memory("after write")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gc.collect()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `log_memory("after gc")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 21 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Built`
- `melt_table_name`
- `table`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `p`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `p` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 22 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `MELT_DESTINATION_TABLE_NAME`
- `validate_sensor_messages_stage`

### Outputs

- `engine`
- `melt_validation_dataframe`
- `schema`
- `table_name`

### Key Operations

- `melt_validation_dataframe = validate_sensor_messages_stage( engine=engine, schema=SCHEMA, table_name=MELT_DESTINATION_TABLE_NAME,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(melt_validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `validate_sensor_messages_stage`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `melt_validation_dataframe = validate_sensor_messages_stage( engine=engine, schema=SCHEMA, table_name=MELT_DESTINATION_TABLE_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(melt_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 23 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `capstone`
- `engine`
- `generated_row_id`
- `LIMIT`
- `message_sequence_index`
- `meta_episode_id`
- `observation_index`
- `ORDER`
- `phase`
- `read_sql_dataframe`
- `SELECT`
- `sensor_index`
- `sensor_name`
- `sensor_value`
- `stream_state`
- `synthetic_sensor_messages_stage`

### Outputs

- `melt_sample_dataframe`

### Key Operations

- `melt_sample_dataframe = read_sql_dataframe( engine, """ SELECT observation_index, generated_row_id, sensor_name, sensor_index, sensor_value, message_sequence_index, stream_state, p`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(melt_sample_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `melt_sample_dataframe = read_sql_dataframe( engine, """ SELECT observation_index, generated_row_id, sensor_name, sensor_index, sensor_value, message_sequence_index, stream_state, p` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(melt_sample_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 24 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `capstone`
- `COUNT`
- `DISTINCT`
- `distinct_msg_seq_count`
- `distinct_sensor_count`
- `engine`
- `GROUP`
- `LIMIT`
- `MAX`
- `max_msg_seq`
- `message_count`
- `message_sequence_index`
- `MIN`
- `min_msg_seq`
- `observation_index`
- `ORDER`
- `read_sql_dataframe`
- `SELECT`
- `sensor_index`

### Outputs

- `sequence_check_dataframe`

### Key Operations

- `sequence_check_dataframe = read_sql_dataframe( engine, """ SELECT observation_index, COUNT(*) AS message_count, COUNT(DISTINCT sensor_index) AS distinct_sensor_count, MIN(message_s`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(sequence_check_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `MAX`
- `MIN`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `sequence_check_dataframe = read_sql_dataframe( engine, """ SELECT observation_index, COUNT(*) AS message_count, COUNT(DISTINCT sensor_index) AS distinct_sensor_count, MIN(message_s` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(sequence_check_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 25 — Code Reference

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

## Code Cell 26 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Create`
- `get_engine_from_env`

### Outputs

- `engine`

### Key Operations

- `# Create Engine`: Documents the purpose or boundary of the surrounding notebook step.
- `engine = get_engine_from_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get_engine_from_env`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Create Engine` | Documents the purpose or boundary of the surrounding notebook step. |
| `engine = get_engine_from_env()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 27 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_sensor_messages_send_queue`
- `Built`
- `IF_EXISTS_FLAG`
- `SEND_QUEUE_CHUNK_SIZE`
- `SEND_QUEUE_DESTINATION_TABLE_NAME`
- `SEND_QUEUE_SOURCE_TABLE_NAME`
- `table`

### Outputs

- `chunk_size`
- `engine`
- `if_exists`
- `queue_status_default`
- `schema`
- `send_queue_table_name`
- `source_table`
- `target_table`

### Key Operations

- `send_queue_table_name = build_sensor_messages_send_queue( engine=engine, schema=SCHEMA, source_table=SEND_QUEUE_SOURCE_TABLE_NAME, target_table=SEND_QUEUE_DESTINATION_TABLE_NAME, i`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Built table:", send_queue_table_name)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_sensor_messages_send_queue`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `send_queue_table_name = build_sensor_messages_send_queue( engine=engine, schema=SCHEMA, source_table=SEND_QUEUE_SOURCE_TABLE_NAME, target_table=SEND_QUEUE_DESTINATION_TABLE_NAME, i` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Built table:", send_queue_table_name)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 28 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `SEND_QUEUE_DESTINATION_TABLE_NAME`
- `validate_sensor_messages_send_queue`

### Outputs

- `engine`
- `schema`
- `send_queue_validation_dataframe`
- `table_name`

### Key Operations

- `send_queue_validation_dataframe = validate_sensor_messages_send_queue( engine=engine, schema=SCHEMA, table_name=SEND_QUEUE_DESTINATION_TABLE_NAME,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(send_queue_validation_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `validate_sensor_messages_send_queue`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `send_queue_validation_dataframe = validate_sensor_messages_send_queue( engine=engine, schema=SCHEMA, table_name=SEND_QUEUE_DESTINATION_TABLE_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(send_queue_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 29 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `capstone`
- `engine`
- `LIMIT`
- `message_key`
- `message_sequence_index`
- `observation_index`
- `observation_timestamp`
- `ORDER`
- `producer_sent_at`
- `queue_status`
- `queued_at`
- `read_sql_dataframe`
- `SELECT`
- `sensor_index`
- `sensor_name`
- `sensor_value`
- `synthetic_sensor_messages_send_queue`

### Outputs

- `send_queue_sample_dataframe`

### Key Operations

- `send_queue_sample_dataframe = read_sql_dataframe( engine, """ SELECT message_key, observation_index, observation_timestamp, message_sequence_index, sensor_name, sensor_index, senso`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(send_queue_sample_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `send_queue_sample_dataframe = read_sql_dataframe( engine, """ SELECT message_key, observation_index, observation_timestamp, message_sequence_index, sensor_name, sensor_index, senso` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(send_queue_sample_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 30 — Code Reference

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

## Code Cell 31 — Code Reference

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

## Code Cell 32 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Create`
- `get_engine_from_env`

### Outputs

- `engine`

### Key Operations

- `# Create Engine`: Documents the purpose or boundary of the surrounding notebook step.
- `engine = get_engine_from_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get_engine_from_env`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Create Engine` | Documents the purpose or boundary of the surrounding notebook step. |
| `engine = get_engine_from_env()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 33 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `ensure_send_queue_runtime_columns`
- `SEND_QUEUE_TABLE_NAME`

### Outputs

- `engine`
- `schema`
- `table_name`

### Key Operations

- `ensure_send_queue_runtime_columns( engine=engine, schema = SCHEMA, table_name=SEND_QUEUE_TABLE_NAME,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `ensure_send_queue_runtime_columns`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `ensure_send_queue_runtime_columns( engine=engine, schema = SCHEMA, table_name=SEND_QUEUE_TABLE_NAME,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 34 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `ensure_simulation_state_control_table`
- `SIMULATION_TABLE_NAME`

### Outputs

- `engine`
- `schema`
- `table_name`

### Key Operations

- `ensure_simulation_state_control_table( engine=engine, schema=SCHEMA, table_name=SIMULATION_TABLE_NAME,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `ensure_simulation_state_control_table`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `ensure_simulation_state_control_table( engine=engine, schema=SCHEMA, table_name=SIMULATION_TABLE_NAME,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 35 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `IS_ENABLED_FLAG`
- `PRODUCER_MAX_SEND_ATTEMPTS`
- `SIMULATION_TABLE_NAME`
- `upsert_simulation_state_control`

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

- `upsert_simulation_state_control( engine=engine, dataset_id = DATASET_ID, run_id = RUN_ID, is_enabled = IS_ENABLED_FLAG, producer_topic = PRODUCER_TOPIC, producer_batch_size = PRODU`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `upsert_simulation_state_control`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `upsert_simulation_state_control( engine=engine, dataset_id = DATASET_ID, run_id = RUN_ID, is_enabled = IS_ENABLED_FLAG, producer_topic = PRODUCER_TOPIC, producer_batch_size = PRODU` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 36 — Code Reference

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

## Code Cell 37 — Code Reference

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

## Code Cell 38 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Create`
- `get_engine_from_env`

### Outputs

- `engine`

### Key Operations

- `# Create Engine`: Documents the purpose or boundary of the surrounding notebook step.
- `engine = get_engine_from_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get_engine_from_env`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Create Engine` | Documents the purpose or boundary of the surrounding notebook step. |
| `engine = get_engine_from_env()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 39 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `an`
- `available`
- `elif`
- `else`
- `f`
- `LOOP`
- `option`
- `please`
- `raise`
- `run_send_queue_producer_loop`
- `run_send_queue_producer_once`
- `select`
- `selected`
- `SEND_QUEUE_MODE`
- `SEND_QUEUE_TABLE_NAME`
- `SIMULATION_TABLE_NAME`
- `SINGLE`
- `STOP_ON_FAILURE_FLAG`
- `ValueError`

### Outputs

- `client_id`
- `control_table`
- `dataset_id`
- `engine`
- `flush_timeout_seconds`
- `max_batches`
- `producer_worker_id`
- `queue_table`
- `result`
- `results`
- `run_id`
- `schema`
- `stop_on_failure`

### Key Operations

- `if SEND_QUEUE_MODE == "SINGLE": result = run_send_queue_producer_once( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, schema=SCHEMA, queue_table=SEND_QUEUE_TABLE_NAME, contro`: Displays a notebook-facing result for inspection.
- `elif SEND_QUEUE_MODE == "LOOP": results = run_send_queue_producer_loop( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, schema=SCHEMA, queue_table=SEND_QUEUE_TABLE_NAME, contr`: Displays a notebook-facing result for inspection.
- `else: raise ValueError(f"SEND_QUEUE_MODE: {SEND_QUEUE_MODE} selected is not an available option, please select SINGLE or LOOP")`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`
- `run_send_queue_producer_loop`
- `run_send_queue_producer_once`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `if SEND_QUEUE_MODE == "SINGLE": result = run_send_queue_producer_once( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, schema=SCHEMA, queue_table=SEND_QUEUE_TABLE_NAME, contro` | Displays a notebook-facing result for inspection. |
| `elif SEND_QUEUE_MODE == "LOOP": results = run_send_queue_producer_loop( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, schema=SCHEMA, queue_table=SEND_QUEUE_TABLE_NAME, contr` | Displays a notebook-facing result for inspection. |
| `else: raise ValueError(f"SEND_QUEUE_MODE: {SEND_QUEUE_MODE} selected is not an available option, please select SINGLE or LOOP")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 40 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_sensor_message_payload`
- `BY`
- `capstone`
- `engine`
- `iloc`
- `json_dumps_safe`
- `LIMIT`
- `message_sequence_index`
- `observation_index`
- `ORDER`
- `read_sql_dataframe`
- `SELECT`
- `sensor_index`
- `synthetic_sensor_messages_send_queue`
- `to_dict`

### Outputs

- `payload`
- `preview_dataframe`

### Key Operations

- `preview_dataframe = read_sql_dataframe( engine, """ SELECT * FROM capstone.synthetic_sensor_messages_send_queue ORDER BY observation_index, message_sequence_index, sensor_index LIM`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `payload = build_sensor_message_payload(preview_dataframe.iloc[0].to_dict())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print(json_dumps_safe(payload))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_sensor_message_payload`
- `json_dumps_safe`
- `read_sql_dataframe`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `preview_dataframe = read_sql_dataframe( engine, """ SELECT * FROM capstone.synthetic_sensor_messages_send_queue ORDER BY observation_index, message_sequence_index, sensor_index LIM` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `payload = build_sensor_message_payload(preview_dataframe.iloc[0].to_dict())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print(json_dumps_safe(payload))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 41 — Code Reference

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

## Code Cell 42 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Create`
- `get_engine_from_env`

### Outputs

- `engine`

### Key Operations

- `# Create Engine`: Documents the purpose or boundary of the surrounding notebook step.
- `engine = get_engine_from_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get_engine_from_env`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Create Engine` | Documents the purpose or boundary of the surrounding notebook step. |
| `engine = get_engine_from_env()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 43 — Code Reference

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

## Code Cell 44 — Code Reference

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

- `rebuild_validation_dataframe`

### Key Operations

- `rebuild_validation_dataframe = read_sql_dataframe( engine, """ SELECT COUNT(*) AS rebuilt_row_count, COUNT(*) FILTER (WHERE rebuild_is_complete = TRUE) AS complete_row_count, MIN(o`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(rebuild_validation_dataframe)`: Displays a notebook-facing result for inspection.

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
| `rebuild_validation_dataframe = read_sql_dataframe( engine, """ SELECT COUNT(*) AS rebuilt_row_count, COUNT(*) FILTER (WHERE rebuild_is_complete = TRUE) AS complete_row_count, MIN(o` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(rebuild_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 45 — Code Reference

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

- `rebuild_sample_dataframe`

### Key Operations

- `rebuild_sample_dataframe = read_sql_dataframe( engine, """ SELECT dataset_id, run_id, asset_id, observation_index, observation_timestamp, stream_state, phase, meta_episode_id, meta`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(rebuild_sample_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `rebuild_sample_dataframe = read_sql_dataframe( engine, """ SELECT dataset_id, run_id, asset_id, observation_index, observation_timestamp, stream_state, phase, meta_episode_id, meta` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(rebuild_sample_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 46 — Code Reference

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

- `rebuild_incomplete_dataframe`

### Key Operations

- `rebuild_incomplete_dataframe = read_sql_dataframe( engine, """ SELECT dataset_id, run_id, asset_id, observation_index, rebuild_sensor_count, rebuild_is_complete, rebuild_notes FROM`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(rebuild_incomplete_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `rebuild_incomplete_dataframe = read_sql_dataframe( engine, """ SELECT dataset_id, run_id, asset_id, observation_index, rebuild_sensor_count, rebuild_is_complete, rebuild_notes FROM` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(rebuild_incomplete_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 47 — Code Reference

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

## Code Cell 48 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Create`
- `get_engine_from_env`

### Outputs

- `engine`

### Key Operations

- `# Create Engine`: Documents the purpose or boundary of the surrounding notebook step.
- `engine = get_engine_from_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get_engine_from_env`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Create Engine` | Documents the purpose or boundary of the surrounding notebook step. |
| `engine = get_engine_from_env()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 49 — Code Reference

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

## Code Cell 50 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `all_match_count`
- `capsone`
- `comparison_all_fields_match`
- `comparison_mismatch_count`
- `comparison_row_count`
- `COUNT`
- `engine`
- `FILTER`
- `MAX`
- `max_mismatch_count`
- `mismatch_count`
- `read_sql_dataframe`
- `SELECT`
- `synthetic_sensor_rebuild_comparison_stage`
- `WHERE`

### Outputs

- `comparison_summary_dataframe`

### Key Operations

- `comparison_summary_dataframe = read_sql_dataframe( engine, """ SELECT COUNT(*) AS comparison_row_count, COUNT(*) FILTER (WHERE comparison_all_fields_match = TRUE) AS all_match_coun`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(comparison_summary_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `FILTER`
- `MAX`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `comparison_summary_dataframe = read_sql_dataframe( engine, """ SELECT COUNT(*) AS comparison_row_count, COUNT(*) FILTER (WHERE comparison_all_fields_match = TRUE) AS all_match_coun` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(comparison_summary_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 51 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `asset_id`
- `BY`
- `capstone`
- `comparison_all_fields_match`
- `comparison_mismatch_count`
- `comparison_notes`
- `dataset_id`
- `engine`
- `exists_in_original`
- `exists_in_rebuilt`
- `LIMIT`
- `observation_index`
- `ORDER`
- `read_sql_dataframe`
- `rebuilt__rebuild_is_complete`
- `rebuilt__rebuild_sensor_count`
- `run_id`
- `SELECT`
- `synthetic_sensor_rebuild_comparison_stage`
- `WHERE`

### Outputs

- `comparison_mismatch_dataframe`

### Key Operations

- `comparison_mismatch_dataframe = read_sql_dataframe( engine, """ SELECT dataset_id, run_id, asset_id, observation_index, comparison_mismatch_count, comparison_notes, exists_in_origi`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(comparison_mismatch_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `comparison_mismatch_dataframe = read_sql_dataframe( engine, """ SELECT dataset_id, run_id, asset_id, observation_index, comparison_mismatch_count, comparison_notes, exists_in_origi` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(comparison_mismatch_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 52 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `capstone`
- `engine`
- `f`
- `observation_index`
- `read_sql_dataframe`
- `rebuild_observation_to_check`
- `SELECT`
- `synthetic_sensor_rebuild_comparison_stage`
- `WHERE`

### Outputs

- `rebuild_detail_dataframe`

### Key Operations

- `rebuild_detail_dataframe = read_sql_dataframe( engine, f""" SELECT * FROM capstone.synthetic_sensor_rebuild_comparison_stage WHERE observation_index = {int(rebuild_observation_to_c`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(rebuild_detail_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `rebuild_detail_dataframe = read_sql_dataframe( engine, f""" SELECT * FROM capstone.synthetic_sensor_rebuild_comparison_stage WHERE observation_index = {int(rebuild_observation_to_c` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(rebuild_detail_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 53 — Code Reference

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

## Code Cell 54 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Create`
- `get_engine_from_env`

### Outputs

- `engine`

### Key Operations

- `# Create Engine`: Documents the purpose or boundary of the surrounding notebook step.
- `engine = get_engine_from_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get_engine_from_env`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Create Engine` | Documents the purpose or boundary of the surrounding notebook step. |
| `engine = get_engine_from_env()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 55 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_synthetic_final_aligned_output_stage`
- `COMPLETE_ONLY_FLAG`
- `IF_EXISTS_FLAG`
- `machine_status`
- `NUMBER_OF_SENSORS`
- `REBUILT_SOURCE_TABLE_NAME`
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

- `final_output_result = build_synthetic_final_aligned_output_stage( engine=engine, schema=SCHEMA, rebuilt_table=REBUILT_SOURCE_TABLE_NAME, target_table=TARGET_TABLE_NAME, dataset_id=`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(final_output_result)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_synthetic_final_aligned_output_stage`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `final_output_result = build_synthetic_final_aligned_output_stage( engine=engine, schema=SCHEMA, rebuilt_table=REBUILT_SOURCE_TABLE_NAME, target_table=TARGET_TABLE_NAME, dataset_id=` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(final_output_result)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 56 — Code Reference

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

- `final_output_validation_dataframe`

### Key Operations

- `final_output_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS row_count, COUNT(DISTINCT dataset_id) AS dataset_id_count, COUNT(DISTINCT run_id) AS run_id_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(final_output_validation_dataframe)`: Displays a notebook-facing result for inspection.

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
| `final_output_validation_dataframe = read_sql_dataframe( engine, f""" SELECT COUNT(*) AS row_count, COUNT(DISTINCT dataset_id) AS dataset_id_count, COUNT(DISTINCT run_id) AS run_id_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(final_output_validation_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 57 — Code Reference

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

- `final_output_sample_dataframe`

### Key Operations

- `final_output_sample_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, timestamp, sensor_00, sensor_01, sensor_02, sensor_03, sensor_04, machine_stat`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(final_output_sample_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `final_output_sample_dataframe = read_sql_dataframe( engine, f""" SELECT dataset_id, run_id, asset_id, timestamp, sensor_00, sensor_01, sensor_02, sensor_03, sensor_04, machine_stat` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(final_output_sample_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 58 — Code Reference

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

- `final_output_status_distribution_dataframe`

### Key Operations

- `final_output_status_distribution_dataframe = read_sql_dataframe( engine, f""" SELECT machine_status, COUNT(*) AS row_count FROM {SCHEMA}.{TARGET_TABLE_NAME} GROUP BY machine_status`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(final_output_status_distribution_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `COUNT`
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `final_output_status_distribution_dataframe = read_sql_dataframe( engine, f""" SELECT machine_status, COUNT(*) AS row_count FROM {SCHEMA}.{TARGET_TABLE_NAME} GROUP BY machine_status` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(final_output_status_distribution_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 59 — Code Reference

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

## Code Cell 60 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Create`
- `get_engine_from_env`

### Outputs

- `engine`

### Key Operations

- `# Create Engine`: Documents the purpose or boundary of the surrounding notebook step.
- `engine = get_engine_from_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get_engine_from_env`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Create Engine` | Documents the purpose or boundary of the surrounding notebook step. |
| `engine = get_engine_from_env()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 61 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `bronze_observations_input_stage`
- `capstone`
- `else`
- `full_batch`
- `no_run`
- `premelt_run_001`
- `pump_synthetic_v1`
- `row`
- `row_batch`
- `run_bronze_handoff_loop`
- `run_mode`
- `status`
- `synthetic_sensor_observations_final_aligned_stage`

### Outputs

- `batch_size`
- `bronze_handoff_results`
- `complete_only`
- `dataset_id`
- `engine`
- `max_iterations`
- `mode`
- `run_id`
- `schema`
- `source_table`
- `stop_on_failure`
- `target_table`

### Key Operations

- `bronze_handoff_results = run_bronze_handoff_loop( engine=engine, mode=run_mode, # "row" \| "row_batch" \| "full_batch" batch_size=500, schema="capstone", source_table="synthetic_sens`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print(bronze_handoff_results[-1] if bronze_handoff_results else {"status": "no_run"})`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `run_bronze_handoff_loop`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `bronze_handoff_results = run_bronze_handoff_loop( engine=engine, mode=run_mode, # "row" \| "row_batch" \| "full_batch" batch_size=500, schema="capstone", source_table="synthetic_sens` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(bronze_handoff_results[-1] if bronze_handoff_results else {"status": "no_run"})` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 62 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `bronze_observations_input_stage`
- `capstone`
- `full_batch`
- `handoff_final_aligned_observations_to_bronze`
- `ignored`
- `premelt_run_001`
- `pump_synthetic_v1`
- `synthetic_sensor_observations_final_aligned_stage`

### Outputs

- `batch_size`
- `complete_only`
- `dataset_id`
- `engine`
- `handoff_final_aligned_observation_result`
- `mode`
- `run_id`
- `schema`
- `source_table`
- `target_table`

### Key Operations

- `handoff_final_aligned_observation_result = handoff_final_aligned_observations_to_bronze( engine=engine, mode="full_batch", batch_size=500, # ignored in full_batch schema="capstone"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(handoff_final_aligned_observation_result)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `handoff_final_aligned_observations_to_bronze`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `handoff_final_aligned_observation_result = handoff_final_aligned_observations_to_bronze( engine=engine, mode="full_batch", batch_size=500, # ignored in full_batch schema="capstone"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(handoff_final_aligned_observation_result)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 63 — Code Reference

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
