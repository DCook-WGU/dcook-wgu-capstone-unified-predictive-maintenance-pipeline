# Notebook Code Reference: synthetic_00_postgres_to_bronze_no_kafka

Notebook path:

`notebooks/synthetic/synthetic_00_postgres_to_bronze_no_kafka.ipynb`

## Notebook Purpose

This notebook prepares the Bronze layer by ingesting, validating, and standardizing the raw pump dataset for downstream Silver processing.

Notebook stage:

`Bronze`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Notes before you run this | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06 |
| Connect to Postgres and inspect the source table | Code Cell 07, Code Cell 08 |
| Preview the raw source rows | Code Cell 09, Code Cell 10 |
| Build the bronze handoff dataframe | Code Cell 11 |
| Status distribution and batch coverage checks | Code Cell 12 |
| Write to Postgres | Code Cell 13 |
| Optional: export parquet / csv | Code Cell 14, Code Cell 15, Code Cell 16 |

## Code Cell 01 — Notes before you run this

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append_truth_index`
- `build_append_aware_bronze_handoff_from_postgres`
- `build_engine_from_project_env`
- `build_truth_record`
- `config_loader`
- `configure_logging`
- `core`
- `database`
- `ensure_handoff_control_table`
- `file_io`
- `frequencies`
- `get_handoff_control_record`
- `get_paths`
- `get_sensor_columns`
- `get_table_columns`
- `initialize_layer_truth`
- `json`
- `load_pipeline_config`
- `log_layer_paths`
- `logging`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import logging`: Imports a dependency or project helper used by later cells.
- `import json`: Imports a dependency or project helper used by later cells.
- `from pathlib import Path`: Imports a dependency or project helper used by later cells.
- `from typing import Optional`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `from pandas.tseries.frequencies import to_offset`: Imports a dependency or project helper used by later cells.
- `from utils.core.paths import get_paths`: Imports a dependency or project helper used by later cells.
- `from utils.core.file_io import save_data`: Imports a dependency or project helper used by later cells.
- `from utils.core.logging_setup import ( configure_logging, log_layer_paths,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.core.config_loader import load_pipeline_config`: Imports a dependency or project helper used by later cells.
- `from utils.core.truths import ( make_process_run_id, initialize_layer_truth, update_truth_section, build_truth_record, save_truth_record, append_truth_index,`: Imports a dependency or project helper used by later cells.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import logging` | Imports a dependency or project helper used by later cells. |
| `import json` | Imports a dependency or project helper used by later cells. |
| `from pathlib import Path` | Imports a dependency or project helper used by later cells. |
| `from typing import Optional` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `from pandas.tseries.frequencies import to_offset` | Imports a dependency or project helper used by later cells. |
| `from utils.core.paths import get_paths` | Imports a dependency or project helper used by later cells. |
| `from utils.core.file_io import save_data` | Imports a dependency or project helper used by later cells. |
| `from utils.core.logging_setup import ( configure_logging, log_layer_paths,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.config_loader import load_pipeline_config` | Imports a dependency or project helper used by later cells. |
| `from utils.core.truths import ( make_process_run_id, initialize_layer_truth, update_truth_section, build_truth_record, save_truth_record, append_truth_index,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.postgres import table_exists` | Imports a dependency or project helper used by later cells. |
| `from utils.synthetic.pipeline.postgres_to_bronze import ( build_engine_from_project_env, get_table_columns, get_sensor_columns, read_synthetic_stream_dataframe, validate_synthetic_` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: truth record.

## Code Cell 02 — Notes before you run this

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `default`
- `Direct`
- `handoff`
- `layer`
- `make_process_run_id`
- `naming`
- `Notebook`
- `params`
- `pump`
- `synthetic`
- `synthetic_bronze_handoff`
- `synthetic_to_bronze`
- `train`

### Outputs

- `DATASET`
- `DIRECT_LAYER_NAME`
- `MODE`
- `PROCESS_RUN_ID`
- `PROFILE`
- `STAGE`

### Key Operations

- `# --- Notebook params ---`: Documents the purpose or boundary of the surrounding notebook step.
- `STAGE = "synthetic"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET = "pump"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `MODE = "train"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PROFILE = "default"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# --- Direct handoff layer naming ---`: Documents the purpose or boundary of the surrounding notebook step.
- `DIRECT_LAYER_NAME = "synthetic_bronze_handoff"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PROCESS_RUN_ID = make_process_run_id("synthetic_to_bronze")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `make_process_run_id`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# --- Notebook params ---` | Documents the purpose or boundary of the surrounding notebook step. |
| `STAGE = "synthetic"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MODE = "train"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# --- Direct handoff layer naming ---` | Documents the purpose or boundary of the surrounding notebook step. |
| `DIRECT_LAYER_NAME = "synthetic_bronze_handoff"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PROCESS_RUN_ID = make_process_run_id("synthetic_to_bronze")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 03 — Notes before you run this

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `batch`
- `configs`
- `data`
- `execution_mode`
- `exist_ok`
- `get`
- `get_paths`
- `load_pipeline_config`
- `logs_root`
- `lower`
- `mkdir`
- `name`
- `notebook`
- `orchestration_mode`
- `parents`
- `resolved_paths`
- `root`
- `strip`
- `truth`
- `truths_dir`

### Outputs

- `ARTIFACTS_ROOT`
- `CONFIG`
- `config_obj`
- `config_root`
- `dataset`
- `DATASET_NAME`
- `LOGS_PATH`
- `mode`
- `paths`
- `PATHS`
- `PIPELINE`
- `PIPELINE_MODE`
- `profile`
- `project_root`
- `stage`
- `TRUTH_INDEX_PATH`
- `TRUTH_VERSION`
- `TRUTHS_PATH`

### Key Operations

- `paths = get_paths()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `config_obj = load_pipeline_config( config_root=paths.configs, stage=STAGE, dataset=DATASET, mode=MODE, profile=PROFILE, project_root=paths.root,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `CONFIG = config_obj.data`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PATHS = CONFIG["resolved_paths"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PIPELINE = CONFIG.get("pipeline", {"execution_mode": "batch", "orchestration_mode": "notebook"})`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PIPELINE_MODE = str(PIPELINE.get("execution_mode", "batch"))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_NAME = str(CONFIG["dataset"]["name"]).strip().lower()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_VERSION = str(CONFIG["versions"]["truth"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTHS_PATH = Path(PATHS["truths_dir"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_INDEX_PATH = Path(PATHS["truth_index_path"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ARTIFACTS_ROOT = Path(PATHS["artifacts_root"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get`
- `get_paths`
- `load_pipeline_config`
- `lower`
- `mkdir`
- `Path`
- `strip`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `paths = get_paths()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `config_obj = load_pipeline_config( config_root=paths.configs, stage=STAGE, dataset=DATASET, mode=MODE, profile=PROFILE, project_root=paths.root,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CONFIG = config_obj.data` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PATHS = CONFIG["resolved_paths"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PIPELINE = CONFIG.get("pipeline", {"execution_mode": "batch", "orchestration_mode": "notebook"})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PIPELINE_MODE = str(PIPELINE.get("execution_mode", "batch"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_NAME = str(CONFIG["dataset"]["name"]).strip().lower()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_VERSION = str(CONFIG["versions"]["truth"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTHS_PATH = Path(PATHS["truths_dir"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_INDEX_PATH = Path(PATHS["truth_index_path"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ARTIFACTS_ROOT = Path(PATHS["artifacts_root"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LOGS_PATH = Path(PATHS["logs_root"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTHS_PATH.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `LOGS_PATH.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("DATASET_NAME:", DATASET_NAME)` | Displays a notebook-facing result for inspection. |
| `print("PIPELINE_MODE:", PIPELINE_MODE)` | Displays a notebook-facing result for inspection. |
| `print("TRUTHS_PATH:", TRUTHS_PATH)` | Displays a notebook-facing result for inspection. |
| `print("ARTIFACTS_ROOT:", ARTIFACTS_ROOT)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 04 — Notes before you run this

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Bronze`
- `capstone`
- `configure_logging`
- `current_layer`
- `direct`
- `getLogger`
- `handoff`
- `INFO`
- `info`
- `log`
- `log_layer_paths`
- `logging`
- `logs`
- `notebook`
- `paths`
- `Postgres`
- `starting`
- `synthetic`
- `Synthetic`
- `synthetic_postgres_to_bronze_no_kafka`

### Outputs

- `direct_log_path`
- `level`
- `logger`
- `overwrite_handlers`

### Key Operations

- `direct_log_path = paths.logs / "synthetic_postgres_to_bronze_no_kafka.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `configure_logging( "capstone", direct_log_path, level=logging.INFO, overwrite_handlers=True,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger = logging.getLogger("capstone.synthetic_to_bronze")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `logger.info("Synthetic direct Postgres to Bronze handoff notebook starting")`: Writes a logger message for traceability during notebook execution.
- `log_layer_paths(paths, current_layer="synthetic", logger=logger)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `configure_logging`
- `getLogger`
- `info`
- `log_layer_paths`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `direct_log_path = paths.logs / "synthetic_postgres_to_bronze_no_kafka.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `configure_logging( "capstone", direct_log_path, level=logging.INFO, overwrite_handlers=True,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger = logging.getLogger("capstone.synthetic_to_bronze")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Synthetic direct Postgres to Bronze handoff notebook starting")` | Writes a logger message for traceability during notebook execution. |
| `log_layer_paths(paths, current_layer="synthetic", logger=logger)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 05 — Notes before you run this

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `continue`
- `dataset`
- `dataset_name`
- `def`
- `encoding`
- `exists`
- `file`
- `get`
- `given`
- `hash`
- `json`
- `JSONDecodeError`
- `jsonl`
- `latest`
- `layer`
- `layer_name`
- `loads`
- `lower`
- `open`

### Outputs

- `dataset_name_norm`
- `get_latest_truth_hash`
- `latest_record`
- `layer_name_norm`
- `line`
- `record`
- `record_dataset`
- `record_layer`
- `truth_hash`

### Key Operations

- `def get_latest_truth_hash( *, truth_index_path: Path, layer_name: str, dataset_name: str,`: Defines notebook-local logic used later in the notebook.
- `) -> Optional[str]: """ Return the latest truth hash for a given layer + dataset from truth_index.jsonl. """ if not truth_index_path.exists(): return None dataset_name_norm = str(d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `exists`
- `get`
- `get_latest_truth_hash`
- `loads`
- `lower`
- `open`
- `strip`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def get_latest_truth_hash( *, truth_index_path: Path, layer_name: str, dataset_name: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> Optional[str]: """ Return the latest truth hash for a given layer + dataset from truth_index.jsonl. """ if not truth_index_path.exists(): return None dataset_name_norm = str(d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 06 — Notes before you run this

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__synthetic__bronze_handoff_append_slice`
- `_bronze_handoff_append`
- `_stream`
- `all`
- `append`
- `APPEND`
- `Append`
- `are`
- `ARTIFACTS_ROOT`
- `available`
- `aware`
- `back`
- `batches`
- `capstone`
- `continuity`
- `CONTROL`
- `Control`
- `csv`
- `export`
- `f`

### Outputs

- `BATCH_IDS`
- `CONTROL_SCHEMA`
- `CONTROL_TABLE`
- `CSV_OUTPUT_PATH`
- `dataset_name`
- `EXPORT_CSV`
- `EXPORT_PARQUET`
- `INCLUDE_SYNTHETIC_ANOMALY_FLAG`
- `KEEP_LINEAGE_COLUMNS`
- `KEEP_OTHER_COLUMNS`
- `layer_name`
- `PARENT_TRUTH_HASH`
- `PARQUET_OUTPUT_PATH`
- `SAMPLING_FREQUENCY`
- `SOURCE_SCHEMA`
- `SOURCE_TABLE`
- `START_TIMESTAMP`
- `TARGET_IF_EXISTS`
- `TARGET_SCHEMA`
- `TARGET_TABLE`

### Key Operations

- `# ---------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Source Postgres settings`: Documents the purpose or boundary of the surrounding notebook step.
- `# ---------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `SOURCE_SCHEMA = "capstone"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SOURCE_TABLE = f"synthetic_{DATASET_NAME}_stream"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Use None for all available source batches not yet loaded`: Documents the purpose or boundary of the surrounding notebook step.
- `BATCH_IDS = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# ---------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Final output shaping`: Documents the purpose or boundary of the surrounding notebook step.
- `# ---------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `START_TIMESTAMP = "2018-04-01 00:00:00"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SAMPLING_FREQUENCY = "1min"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get_latest_truth_hash`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Source Postgres settings` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `SOURCE_SCHEMA = "capstone"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SOURCE_TABLE = f"synthetic_{DATASET_NAME}_stream"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Use None for all available source batches not yet loaded` | Documents the purpose or boundary of the surrounding notebook step. |
| `BATCH_IDS = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Final output shaping` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `START_TIMESTAMP = "2018-04-01 00:00:00"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SAMPLING_FREQUENCY = "1min"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TARGET_TOTAL_ROWS = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRIM_MODE = "head" # head \| tail \| random` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Keep True in append mode so continuity fields are preserved` | Documents the purpose or boundary of the surrounding notebook step. |
| `KEEP_LINEAGE_COLUMNS = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `INCLUDE_SYNTHETIC_ANOMALY_FLAG = False` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `KEEP_OTHER_COLUMNS = False` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Append-aware Postgres write-back` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `WRITE_TO_POSTGRES = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TARGET_SCHEMA = "capstone"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TARGET_TABLE = f"synthetic_{DATASET_NAME}_bronze_handoff_append"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TARGET_IF_EXISTS = "append"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Control table` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `CONTROL_SCHEMA = "capstone"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTROL_TABLE = "synthetic_bronze_handoff_control"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# File export` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `EXPORT_PARQUET = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `EXPORT_CSV = False` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PARQUET_OUTPUT_PATH = ( ARTIFACTS_ROOT / "synthetic" / DATASET_NAME / f"{DATASET_NAME}__synthetic__bronze_handoff_append_slice.parquet"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CSV_OUTPUT_PATH = ( ARTIFACTS_ROOT / "synthetic" / DATASET_NAME / f"{DATASET_NAME}__synthetic__bronze_handoff_append_slice.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Truth lineage` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `PARENT_TRUTH_HASH = get_latest_truth_hash( truth_index_path=TRUTH_INDEX_PATH, layer_name="synthetic", dataset_name=DATASET_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("SOURCE:", f"{SOURCE_SCHEMA}.{SOURCE_TABLE}")` | Displays a notebook-facing result for inspection. |
| `print("APPEND TARGET:", f"{TARGET_SCHEMA}.{TARGET_TABLE}")` | Displays a notebook-facing result for inspection. |
| `print("CONTROL TABLE:", f"{CONTROL_SCHEMA}.{CONTROL_TABLE}")` | Displays a notebook-facing result for inspection. |
| `print("PARENT_TRUTH_HASH:", PARENT_TRUTH_HASH)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 07 — Connect to Postgres and inspect the source table

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_engine_from_project_env`
- `column`
- `column_count`
- `columns`
- `count`
- `exists`
- `f`
- `First`
- `found`
- `get_sensor_columns`
- `get_table_columns`
- `info`
- `inspected`
- `logger`
- `raise`
- `s`
- `Sensor`
- `sensor_count`
- `Source`
- `SOURCE_SCHEMA`

### Outputs

- `engine`
- `schema`
- `source_columns`
- `source_exists`
- `table_name`

### Key Operations

- `engine = build_engine_from_project_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `source_exists = table_exists( engine, schema=SOURCE_SCHEMA, table_name=SOURCE_TABLE,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Source table exists:", source_exists)`: Displays a notebook-facing result for inspection.
- `if not source_exists: raise ValueError(f"Source table not found: {SOURCE_SCHEMA}.{SOURCE_TABLE}")`: Controls validation, iteration, file handling, or error handling for this step.
- `source_columns = get_table_columns( engine, schema=SOURCE_SCHEMA, table_name=SOURCE_TABLE,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Source column count:", len(source_columns))`: Displays a notebook-facing result for inspection.
- `print("Sensor column count:", len(get_sensor_columns(source_columns)))`: Displays a notebook-facing result for inspection.
- `print("First 25 columns:", source_columns[:25])`: Displays a notebook-facing result for inspection.
- `logger.info( "Source table inspected \| schema=%s \| table=%s \| column_count=%s \| sensor_count=%s", SOURCE_SCHEMA, SOURCE_TABLE, len(source_columns), len(get_sensor_columns(source_co`: Writes a logger message for traceability during notebook execution.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `build_engine_from_project_env`
- `get_sensor_columns`
- `get_table_columns`
- `info`
- `table_exists`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `engine = build_engine_from_project_env()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `source_exists = table_exists( engine, schema=SOURCE_SCHEMA, table_name=SOURCE_TABLE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Source table exists:", source_exists)` | Displays a notebook-facing result for inspection. |
| `if not source_exists: raise ValueError(f"Source table not found: {SOURCE_SCHEMA}.{SOURCE_TABLE}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `source_columns = get_table_columns( engine, schema=SOURCE_SCHEMA, table_name=SOURCE_TABLE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Source column count:", len(source_columns))` | Displays a notebook-facing result for inspection. |
| `print("Sensor column count:", len(get_sensor_columns(source_columns)))` | Displays a notebook-facing result for inspection. |
| `print("First 25 columns:", source_columns[:25])` | Displays a notebook-facing result for inspection. |
| `logger.info( "Source table inspected \| schema=%s \| table=%s \| column_count=%s \| sensor_count=%s", SOURCE_SCHEMA, SOURCE_TABLE, len(source_columns), len(get_sensor_columns(source_co` | Writes a logger message for traceability during notebook execution. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 08 — Connect to Postgres and inspect the source table

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_engine_from_project_env`
- `Control`
- `CONTROL_SCHEMA`
- `CONTROL_TABLE`
- `ensure_handoff_control_table`
- `f`
- `ready`
- `table`

### Outputs

- `engine`
- `schema`
- `table_name`

### Key Operations

- `engine = build_engine_from_project_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ensure_handoff_control_table( engine, schema=CONTROL_SCHEMA, table_name=CONTROL_TABLE,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print(f"Control table ready: {CONTROL_SCHEMA}.{CONTROL_TABLE}")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_engine_from_project_env`
- `ensure_handoff_control_table`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `engine = build_engine_from_project_env()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ensure_handoff_control_table( engine, schema=CONTROL_SCHEMA, table_name=CONTROL_TABLE,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"Control table ready: {CONTROL_SCHEMA}.{CONTROL_TABLE}")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 09 — Preview the raw source rows

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `dumps`
- `engine`
- `head`
- `indent`
- `info`
- `json`
- `loaded`
- `logger`
- `Raw`
- `read_synthetic_stream_dataframe`
- `report`
- `rows`
- `s`
- `shape`
- `SOURCE_SCHEMA`
- `SOURCE_TABLE`
- `stream`
- `synthetic`
- `validate_synthetic_stream_dataframe`

### Outputs

- `batch_ids`
- `raw_stream_df`
- `schema`
- `table_name`
- `validation_report`

### Key Operations

- `raw_stream_df = read_synthetic_stream_dataframe( engine, schema=SOURCE_SCHEMA, table_name=SOURCE_TABLE, batch_ids=BATCH_IDS,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `validation_report = validate_synthetic_stream_dataframe(raw_stream_df)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("Raw shape:", raw_stream_df.shape)`: Displays a notebook-facing result for inspection.
- `print("Validation report:")`: Displays a notebook-facing result for inspection.
- `print(json.dumps(validation_report, indent=2))`: Displays a notebook-facing result for inspection.
- `display(raw_stream_df.head())`: Displays a notebook-facing result for inspection.
- `logger.info( "Raw synthetic stream loaded \| rows=%s \| columns=%s", len(raw_stream_df), len(raw_stream_df.columns),`: Writes a logger message for traceability during notebook execution.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`
- `dumps`
- `head`
- `info`
- `read_synthetic_stream_dataframe`
- `validate_synthetic_stream_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `raw_stream_df = read_synthetic_stream_dataframe( engine, schema=SOURCE_SCHEMA, table_name=SOURCE_TABLE, batch_ids=BATCH_IDS,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `validation_report = validate_synthetic_stream_dataframe(raw_stream_df)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Raw shape:", raw_stream_df.shape)` | Displays a notebook-facing result for inspection. |
| `print("Validation report:")` | Displays a notebook-facing result for inspection. |
| `print(json.dumps(validation_report, indent=2))` | Displays a notebook-facing result for inspection. |
| `display(raw_stream_df.head())` | Displays a notebook-facing result for inspection. |
| `logger.info( "Raw synthetic stream loaded \| rows=%s \| columns=%s", len(raw_stream_df), len(raw_stream_df.columns),` | Writes a logger message for traceability during notebook execution. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 10 — Preview the raw source rows

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Actual`
- `columns`
- `count`
- `Expected`
- `f`
- `found`
- `get_sensor_columns`
- `raise`
- `raw_stream_df`
- `sensor`
- `ValueError`

### Outputs

- `actual_sensor_count`
- `expected_sensor_count`

### Key Operations

- `expected_sensor_count = 52`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `actual_sensor_count = len(get_sensor_columns(raw_stream_df.columns))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("Expected sensor count:", expected_sensor_count)`: Displays a notebook-facing result for inspection.
- `print("Actual sensor count:", actual_sensor_count)`: Displays a notebook-facing result for inspection.
- `if actual_sensor_count != expected_sensor_count: raise ValueError( f"Expected {expected_sensor_count} sensor columns, found {actual_sensor_count}." )`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `get_sensor_columns`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `expected_sensor_count = 52` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `actual_sensor_count = len(get_sensor_columns(raw_stream_df.columns))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Expected sensor count:", expected_sensor_count)` | Displays a notebook-facing result for inspection. |
| `print("Actual sensor count:", actual_sensor_count)` | Displays a notebook-facing result for inspection. |
| `if actual_sensor_count != expected_sensor_count: raise ValueError( f"Expected {expected_sensor_count} sensor columns, found {actual_sensor_count}." )` | Controls validation, iteration, file handling, or error handling for this step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 11 — Build the bronze handoff dataframe

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Append`
- `append_handoff_df`
- `append_plan`
- `build_append_aware_bronze_handoff_from_postgres`
- `default`
- `dumps`
- `engine`
- `INCLUDE_SYNTHETIC_ANOMALY_FLAG`
- `indent`
- `info`
- `json`
- `logger`
- `nAppend`
- `plan`
- `s`
- `SAMPLING_FREQUENCY`
- `START_TIMESTAMP`
- `summarize_bronze_handoff_dataframe`
- `summary`

### Outputs

- `append_summary`
- `batch_ids`
- `frequency`
- `include_anomaly_flag`
- `initial_start_timestamp`
- `keep_lineage_columns`
- `keep_other_columns`
- `source_schema`
- `source_table`
- `target_schema`
- `target_table`
- `target_total_rows`
- `trim_mode`

### Key Operations

- `append_handoff_df, append_plan = build_append_aware_bronze_handoff_from_postgres( engine, source_schema=SOURCE_SCHEMA, source_table=SOURCE_TABLE, target_schema=TARGET_SCHEMA, targe`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `append_summary = summarize_bronze_handoff_dataframe(append_handoff_df)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("Append plan:")`: Displays a notebook-facing result for inspection.
- `print(json.dumps(append_plan, indent=2, default=str))`: Displays a notebook-facing result for inspection.
- `print("\nAppend summary:")`: Displays a notebook-facing result for inspection.
- `print(json.dumps(append_summary, indent=2))`: Displays a notebook-facing result for inspection.
- `logger.info("Append plan: %s", append_plan)`: Writes a logger message for traceability during notebook execution.
- `logger.info("Append summary: %s", append_summary)`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `build_append_aware_bronze_handoff_from_postgres`
- `dumps`
- `info`
- `summarize_bronze_handoff_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `append_handoff_df, append_plan = build_append_aware_bronze_handoff_from_postgres( engine, source_schema=SOURCE_SCHEMA, source_table=SOURCE_TABLE, target_schema=TARGET_SCHEMA, targe` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `append_summary = summarize_bronze_handoff_dataframe(append_handoff_df)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Append plan:")` | Displays a notebook-facing result for inspection. |
| `print(json.dumps(append_plan, indent=2, default=str))` | Displays a notebook-facing result for inspection. |
| `print("\nAppend summary:")` | Displays a notebook-facing result for inspection. |
| `print(json.dumps(append_summary, indent=2))` | Displays a notebook-facing result for inspection. |
| `logger.info("Append plan: %s", append_plan)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Append summary: %s", append_summary)` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 12 — Status distribution and batch coverage checks

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append`
- `Append`
- `append_handoff_df`
- `batch_id`
- `batches`
- `by`
- `columns`
- `count`
- `counts`
- `dataframe`
- `dropna`
- `else`
- `episode`
- `found`
- `head`
- `machine_status`
- `meta__episode_id_unified`
- `new`
- `nmachine_status`
- `No`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `print("Append dataframe shape:", append_handoff_df.shape)`: Displays a notebook-facing result for inspection.
- `print("Append dataframe columns:")`: Displays a notebook-facing result for inspection.
- `print(append_handoff_df.columns.tolist())`: Displays a notebook-facing result for inspection.
- `if len(append_handoff_df) > 0: display(append_handoff_df.head())`: Displays a notebook-facing result for inspection.
- `else: print("No new batches were found to append.")`: Displays a notebook-facing result for inspection.
- `if "machine_status" in append_handoff_df.columns and len(append_handoff_df) > 0: print("\nmachine_status counts:") print(append_handoff_df["machine_status"].value_counts(dropna=Fal`: Displays a notebook-facing result for inspection.
- `if "batch_id" in append_handoff_df.columns and len(append_handoff_df) > 0: print("\nrows by batch_id:") print(append_handoff_df["batch_id"].value_counts().sort_index())`: Displays a notebook-facing result for inspection.
- `if "meta__episode_id_unified" in append_handoff_df.columns and len(append_handoff_df) > 0: print("\nunified episode count:", append_handoff_df["meta__episode_id_unified"].nunique()`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `head`
- `nunique`
- `sort_index`
- `tolist`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print("Append dataframe shape:", append_handoff_df.shape)` | Displays a notebook-facing result for inspection. |
| `print("Append dataframe columns:")` | Displays a notebook-facing result for inspection. |
| `print(append_handoff_df.columns.tolist())` | Displays a notebook-facing result for inspection. |
| `if len(append_handoff_df) > 0: display(append_handoff_df.head())` | Displays a notebook-facing result for inspection. |
| `else: print("No new batches were found to append.")` | Displays a notebook-facing result for inspection. |
| `if "machine_status" in append_handoff_df.columns and len(append_handoff_df) > 0: print("\nmachine_status counts:") print(append_handoff_df["machine_status"].value_counts(dropna=Fal` | Displays a notebook-facing result for inspection. |
| `if "batch_id" in append_handoff_df.columns and len(append_handoff_df) > 0: print("\nrows by batch_id:") print(append_handoff_df["batch_id"].value_counts().sort_index())` | Displays a notebook-facing result for inspection. |
| `if "meta__episode_id_unified" in append_handoff_df.columns and len(append_handoff_df) > 0: print("\nunified episode count:", append_handoff_df["meta__episode_id_unified"].nunique()` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 13 — Write to Postgres

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append`
- `append_handoff_df`
- `Appended`
- `changed`
- `else`
- `engine`
- `f`
- `new`
- `No`
- `Postgres`
- `rows`
- `table`
- `Target`
- `TARGET_IF_EXISTS`
- `TARGET_SCHEMA`
- `TARGET_TABLE`
- `to`
- `write_bronze_handoff_to_postgres`
- `WRITE_TO_POSTGRES`

### Outputs

- `if_exists`
- `logger`
- `schema`
- `table_name`
- `written_table_name`

### Key Operations

- `written_table_name = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if WRITE_TO_POSTGRES: if len(append_handoff_df) == 0: print("No new rows to append. Target table not changed.") else: written_table_name = write_bronze_handoff_to_postgres( engine,`: Displays a notebook-facing result for inspection.
- `else: print("WRITE_TO_POSTGRES is False")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `write_bronze_handoff_to_postgres`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `written_table_name = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if WRITE_TO_POSTGRES: if len(append_handoff_df) == 0: print("No new rows to append. Target table not changed.") else: written_table_name = write_bronze_handoff_to_postgres( engine,` | Displays a notebook-facing result for inspection. |
| `else: print("WRITE_TO_POSTGRES is False")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 14 — Optional: export parquet / csv

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append`
- `append_handoff_df`
- `csv`
- `CSV_OUTPUT_PATH`
- `else`
- `export`
- `EXPORT_CSV`
- `EXPORT_PARQUET`
- `file`
- `new`
- `No`
- `parquet`
- `PARQUET_OUTPUT_PATH`
- `save_data`
- `Saved`
- `selected`
- `slice`
- `to`

### Outputs

- `index`
- `saved_csv_path`
- `saved_parquet_path`

### Key Operations

- `saved_parquet_path = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `saved_csv_path = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if len(append_handoff_df) == 0: print("No new append slice to export.")`: Displays a notebook-facing result for inspection.
- `else: if EXPORT_PARQUET: saved_parquet_path = save_data( append_handoff_df, PARQUET_OUTPUT_PATH, index=False, ) print("Saved parquet:", saved_parquet_path) if EXPORT_CSV: saved_csv`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `save_data`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `saved_parquet_path = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `saved_csv_path = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(append_handoff_df) == 0: print("No new append slice to export.")` | Displays a notebook-facing result for inspection. |
| `else: if EXPORT_PARQUET: saved_parquet_path = save_data( append_handoff_df, PARQUET_OUTPUT_PATH, index=False, ) print("Saved parquet:", saved_parquet_path) if EXPORT_CSV: saved_csv` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 15 — Optional: export parquet / csv

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `already_loaded_batch_count`
- `append_handoff_df`
- `append_plan`
- `append_summary`
- `append_truth_index`
- `appended_row_count`
- `artifact_paths`
- `batch_count`
- `batch_id`
- `batch_ids`
- `BATCH_IDS`
- `bool`
- `build_truth_record`
- `candidate_batch_count`
- `column`
- `columns`
- `common`
- `config_snapshot`
- `Convert`
- `created_at`

### Outputs

- `append_plan_safe`
- `column_count`
- `dataset_name`
- `feature_columns`
- `layer_name`
- `make_json_safe`
- `meta_columns`
- `parent_truth_hash`
- `pipeline_mode`
- `process_run_id`
- `row_count`
- `truth_base`
- `truth_dir`
- `truth_index_path`
- `truth_path`
- `truth_record`
- `truth_version`

### Key Operations

- `def make_json_safe(value): """ Convert common pandas / Python values to JSON-safe plain types. """ if isinstance(value, pd.Timestamp): return str(value) if isinstance(value, Path):`: Defines notebook-local logic used later in the notebook.
- `append_plan_safe = make_json_safe(append_plan)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `truth_base = initialize_layer_truth( truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name=DIRECT_LAYER_NAME, process_run_id=PROCESS_RUN_ID, pipeline_mode=PIPELINE_MOD`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `truth_base = update_truth_section( truth_base, "append_plan", append_plan_safe,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `truth_base = update_truth_section( truth_base, "config_snapshot", { "source_schema": SOURCE_SCHEMA, "source_table": SOURCE_TABLE, "batch_ids": BATCH_IDS, "start_timestamp": START_T`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `truth_base = update_truth_section( truth_base, "runtime_facts", { "source_row_count": int(len(raw_stream_df)), "appended_row_count": int(len(append_handoff_df)), "final_column_coun`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `truth_base = update_truth_section( truth_base, "artifact_paths", { "postgres_table": f"{TARGET_SCHEMA}.{TARGET_TABLE}", "postgres_table_written_name": written_table_name, "parquet_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append_truth_index`
- `bool`
- `build_truth_record`
- `get`
- `get_sensor_columns`
- `info`
- `initialize_layer_truth`
- `isinstance`
- `items`
- `make_json_safe`
- `save_truth_record`
- `sorted`
- `startswith`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def make_json_safe(value): """ Convert common pandas / Python values to JSON-safe plain types. """ if isinstance(value, pd.Timestamp): return str(value) if isinstance(value, Path):` | Defines notebook-local logic used later in the notebook. |
| `append_plan_safe = make_json_safe(append_plan)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `truth_base = initialize_layer_truth( truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name=DIRECT_LAYER_NAME, process_run_id=PROCESS_RUN_ID, pipeline_mode=PIPELINE_MOD` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `truth_base = update_truth_section( truth_base, "append_plan", append_plan_safe,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `truth_base = update_truth_section( truth_base, "config_snapshot", { "source_schema": SOURCE_SCHEMA, "source_table": SOURCE_TABLE, "batch_ids": BATCH_IDS, "start_timestamp": START_T` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `truth_base = update_truth_section( truth_base, "runtime_facts", { "source_row_count": int(len(raw_stream_df)), "appended_row_count": int(len(append_handoff_df)), "final_column_coun` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `truth_base = update_truth_section( truth_base, "artifact_paths", { "postgres_table": f"{TARGET_SCHEMA}.{TARGET_TABLE}", "postgres_table_written_name": written_table_name, "parquet_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `meta_columns = [column for column in append_handoff_df.columns if str(column).startswith("meta__")]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `meta_columns += [column for column in append_handoff_df.columns if column in { "unified_row_id", "observation_time_index", "batch_id", "row_in_batch", "global_cycle_id", "cycle_id"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `}]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `meta_columns = sorted(set(meta_columns))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `feature_columns = [column for column in append_handoff_df.columns if column not in meta_columns]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `truth_record = build_truth_record( truth_base=truth_base, row_count=len(append_handoff_df), column_count=len(append_handoff_df.columns), meta_columns=meta_columns, feature_columns=` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `truth_path = save_truth_record( truth_record, truth_dir=TRUTHS_PATH, dataset_name=DATASET_NAME, layer_name=DIRECT_LAYER_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `append_truth_index( truth_record, truth_index_path=TRUTH_INDEX_PATH,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Truth saved to:", truth_path)` | Displays a notebook-facing result for inspection. |
| `print("Truth hash:", truth_record["truth_hash"])` | Displays a notebook-facing result for inspection. |
| `logger.info("Truth saved \| path=%s \| truth_hash=%s", truth_path, truth_record["truth_hash"])` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- Artifact or state outputs detected: truth record.

## Code Cell 16 — Optional: export parquet / csv

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `after`
- `append`
- `append_handoff_df`
- `appended`
- `astype`
- `batch_id`
- `columns`
- `Control`
- `dropna`
- `else`
- `engine`
- `f`
- `get`
- `get_handoff_control_record`
- `max`
- `meta__episode_id_unified`
- `new`
- `No`
- `observation_time_index`
- `PROCESS_RUN_ID`

### Outputs

- `control_before`
- `control_schema`
- `control_table`
- `dataset_name`
- `last_append_row_count`
- `last_loaded_batch_id`
- `last_loaded_batch_ids`
- `last_process_run_id`
- `last_truth_hash`
- `loaded_batch_count`
- `new_loaded_batch_count`
- `next_observation_time_index`
- `next_timestamp`
- `next_unified_episode_id`
- `next_unified_row_id`
- `notes`
- `previous_loaded_batch_count`
- `target_schema`
- `target_table`

### Key Operations

- `control_before = get_handoff_control_record( engine, dataset_name=DATASET_NAME, target_schema=TARGET_SCHEMA, target_table=TARGET_TABLE, control_schema=CONTROL_SCHEMA, control_table`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if len(append_handoff_df) == 0: print("No new rows appended. Control table not updated.")`: Displays a notebook-facing result for inspection.
- `else: last_loaded_batch_ids = ( sorted(append_handoff_df["batch_id"].dropna().astype(int).unique().tolist()) if "batch_id" in append_handoff_df.columns else [] ) last_loaded_batch_`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `astype`
- `dropna`
- `get`
- `get_handoff_control_record`
- `max`
- `sorted`
- `to_datetime`
- `to_offset`
- `tolist`
- `unique`
- `upsert_handoff_control_record`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `control_before = get_handoff_control_record( engine, dataset_name=DATASET_NAME, target_schema=TARGET_SCHEMA, target_table=TARGET_TABLE, control_schema=CONTROL_SCHEMA, control_table` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if len(append_handoff_df) == 0: print("No new rows appended. Control table not updated.")` | Displays a notebook-facing result for inspection. |
| `else: last_loaded_batch_ids = ( sorted(append_handoff_df["batch_id"].dropna().astype(int).unique().tolist()) if "batch_id" in append_handoff_df.columns else [] ) last_loaded_batch_` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

