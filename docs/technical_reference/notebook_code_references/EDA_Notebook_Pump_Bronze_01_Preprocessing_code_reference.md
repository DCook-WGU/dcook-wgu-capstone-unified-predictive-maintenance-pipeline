# Notebook Code Reference: EDA_Notebook_Pump_Bronze_01_Preprocessing

Notebook path:

`notebooks/preprocessing/EDA_Notebook_Pump_Bronze_01_Preprocessing.ipynb`

## Notebook Purpose

This notebook prepares the Bronze layer by ingesting, validating, and standardizing the raw pump dataset for downstream Silver processing.

Notebook stage:

`Bronze`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Environment Setup and Utility Imports | Code Cell 01 |
| Define configuration mapping guards | Code Cell 02 |
| Load Project Paths, Configuration, and Runtime Settings | Code Cell 03 |
| Notebook Level Configuration | Code Cell 04 |
| Defer Bronze Artifact Folder Creation Until the Dataset Name Is Resolved | Code Cell 05 |
| SQL Runtime Context | Code Cell 06, Code Cell 07, Code Cell 08 |
| SQL Smoke Check | Code Cell 09 |
| Start Logging for the Bronze Stage | Code Cell 10, Code Cell 11 |
| Define Dataset Name Resolution Logic | Code Cell 12 |
| Resolve the Provisional Dataset Identity | Code Cell 13 |
| Prepare Dataset Resolution Metadata | Code Cell 14 |
| Defer the Config Snapshot Until the Final Dataset Name Exists | Code Cell 15 |
| Initialize the Experiment Tracking Run | Code Cell 16 |
| Start experiment tracking for this stage | Code Cell 17 |
| Ingest the Raw Dataset into the Bronze Layer | Code Cell 18 |
| Confirm the Final Dataset Identity After Ingestion | Code Cell 19 |
| Create the Bronze Artifact Folders with the Artifact Util | Code Cell 20 |
| Update Tracking Metadata with the Final Dataset Name | Code Cell 21 |
| Log Any Dataset Name Changes | Code Cell 22 |
| Build the Bronze Truth Record Foundation | Code Cell 23 |
| Bronze Data Review | Code Cell 24 |
| Finalize Lineage Metadata and Save the Bronze Truth Record | Code Cell 25 |
| Define a Helper to Reorder Bronze Columns | Code Cell 26 |
| Reorder the Bronze Dataframe Columns | Code Cell 27 |
| Save the Bronze Dataset | Code Cell 28 |
| Create Bronze Profiling Outputs | Code Cell 29 |
| Finalize Bronze Run Tracking | Code Cell 30 |
| Finish the Tracking Run | Code Cell 31 |
| Define Validation Helpers for Bronze Checks | Code Cell 32 |
| Define integer validation helper | Code Cell 33 |
| Validate Bronze Lineage and Truth Consistency | Code Cell 34 |
| Final Bronze Structural Check | Code Cell 35 |
| Bronze SQL Write Cell | Code Cell 36 |
| Preview the Bronze SQL Output | Code Cell 37 |
| Preview the SQL-facing layer output | Code Cell 38 |
| Close the database engine cleanly | Code Cell 39 |

## Code Cell 01 — Environment Setup and Utility Imports

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `append_truth_index`
- `artifact_file_path`
- `artifacts`
- `build_artifact_dirs`
- `build_artifact_dirs_from_config`
- `build_file_fingerprint`
- `build_truth_config_block`
- `build_truth_record`
- `cast`
- `columns`
- `config_loader`
- `configure_logging`
- `core`
- `Custom`
- `database`
- `datetime`
- `delete_dataset_run_rows`
- `execute_many`
- `export_config_snapshot`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `import glob`: Imports a dependency or project helper used by later cells.
- `from pathlib import Path`: Imports a dependency or project helper used by later cells.
- `import yaml`: Imports a dependency or project helper used by later cells.
- `import sys`: Imports a dependency or project helper used by later cells.
- `import json`: Imports a dependency or project helper used by later cells.
- `import logging`: Imports a dependency or project helper used by later cells.
- `import wandb`: Imports a dependency or project helper used by later cells.
- `from datetime import datetime, timezone`: Imports a dependency or project helper used by later cells.
- `from typing import Optional, Tuple, List, Any, Dict, Mapping, cast`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `import numpy as np`: Imports a dependency or project helper used by later cells.

Important functions or methods detected:
- `import`
- `set_option`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `import glob` | Imports a dependency or project helper used by later cells. |
| `from pathlib import Path` | Imports a dependency or project helper used by later cells. |
| `import yaml` | Imports a dependency or project helper used by later cells. |
| `import sys` | Imports a dependency or project helper used by later cells. |
| `import json` | Imports a dependency or project helper used by later cells. |
| `import logging` | Imports a dependency or project helper used by later cells. |
| `import wandb` | Imports a dependency or project helper used by later cells. |
| `from datetime import datetime, timezone` | Imports a dependency or project helper used by later cells. |
| `from typing import Optional, Tuple, List, Any, Dict, Mapping, cast` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `import numpy as np` | Imports a dependency or project helper used by later cells. |
| `import hashlib` | Imports a dependency or project helper used by later cells. |
| `# Custom Utilities Module` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.core.paths import get_paths` | Imports a dependency or project helper used by later cells. |
| `from utils.core.file_io import ingest_data, save_data, load_json` | Imports a dependency or project helper used by later cells. |
| `from utils.core.logging_setup import ( configure_logging, log_layer_paths` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.logging_profiler import profile_dataframe` | Imports a dependency or project helper used by later cells. |
| `from utils.core.truths import ( make_process_run_id, build_file_fingerprint, extract_truth_hash, identify_meta_columns, identify_feature_columns, initialize_layer_truth, update_tru` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.wandb_utils import ( log_metrics, log_dataframe_head, log_dir_as_artifact, log_files_as_artifact,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.wandb_utils import finalize_wandb_stage` | Imports a dependency or project helper used by later cells. |
| `from utils.core.config_loader import ( load_pipeline_config, build_truth_config_block, set_wandb_dir_from_config, export_config_snapshot,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.layer_postgres import ( read_layer_dataframe, write_layer_dataframe, prepare_layer_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.sql_notebook_helpers import ( delete_dataset_run_rows, execute_many, get_existing_dataframe, get_row_value, log_data_quality_event, log_pipeline_artifact, previ` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.medallion_sql_writers import ( log_gold_05_anomaly_detection_summary_sql, log_silver_eda_sql, write_bronze_sensor_observations_sql, write_gold_baseline_scores_s` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.artifacts import ( build_artifact_dirs, build_artifact_dirs_from_config, artifact_file_path,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.notebook_context import load_notebook_context` | Imports a dependency or project helper used by later cells. |
| `# Show more columns` | Documents the purpose or boundary of the surrounding notebook step. |
| `pd.set_option("display.max_columns", 100)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `pd.set_option("display.width", 200)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: SQL or medallion table write, truth record.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Define configuration mapping guards

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__name__`
- `a`
- `Any`
- `be`
- `cast`
- `def`
- `f`
- `got`
- `isinstance`
- `mapping`
- `Mapping`
- `must`
- `name`
- `object`
- `r`
- `raise`
- `type`
- `TypeError`
- `value`

### Outputs

- `cfg_optional_mapping`
- `cfg_require_mapping`

### Key Operations

- `def cfg_require_mapping(value: object, name: str) -> Mapping[str, Any]: if not isinstance(value, Mapping): raise TypeError( f"{name} must be a mapping, got {type(value).__name__}: `: Defines notebook-local logic used later in the notebook.
- `def cfg_optional_mapping(value: object \| None, name: str) -> Mapping[str, Any]: if value is None: return {} return cfg_require_mapping(value, name)`: Defines notebook-local logic used later in the notebook.

Important functions or methods detected:
- `cast`
- `cfg_optional_mapping`
- `cfg_require_mapping`
- `isinstance`
- `type`
- `TypeError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def cfg_require_mapping(value: object, name: str) -> Mapping[str, Any]: if not isinstance(value, Mapping): raise TypeError( f"{name} must be a mapping, got {type(value).__name__}: ` | Defines notebook-local logic used later in the notebook. |
| `def cfg_optional_mapping(value: object \| None, name: str) -> Mapping[str, Any]: if value is None: return {} return cfg_require_mapping(value, name)` | Defines notebook-local logic used later in the notebook. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 03 — Load Project Paths, Configuration, and Runtime Settings

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `aliases`
- `bronze`
- `capstone`
- `context`
- `context_loaded`
- `dataset_config`
- `default`
- `execution`
- `info`
- `load_notebook_context`
- `Loaded`
- `loaded`
- `log`
- `LOG_PATH`
- `log_path`
- `logger`
- `logger_child_name`
- `message`
- `mode`

### Outputs

- `BRONZE_CFG`
- `CONFIG`
- `CONFIG_MAP`
- `CONFIG_PROFILE`
- `CONFIG_RUN_MODE`
- `CONTEXT_DATASET`
- `CONTEXT_LAYER`
- `CONTEXT_LOG_FILE`
- `CONTEXT_RECIPE_ID`
- `CONTEXT_STAGE`
- `CTX`
- `data`
- `dataset`
- `DATASET_CFG`
- `EXECUTION_CFG`
- `extra`
- `FILENAMES`
- `kind`
- `ledger`
- `log_filename`

### Key Operations

- `# Shared notebook context`: Documents the purpose or boundary of the surrounding notebook step.
- `CONTEXT_STAGE = "bronze"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_DATASET = "pump"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LAYER = "bronze"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_RUN_MODE = "train"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_PROFILE = "default"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LOG_FILE = "bronze.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.bronze", log_filename=CONTEXT_L`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Shared aliases used throughout the notebook`: Documents the purpose or boundary of the surrounding notebook step.
- `paths = CTX.paths`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG = CTX.config`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `info`
- `load_notebook_context`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Shared notebook context` | Documents the purpose or boundary of the surrounding notebook step. |
| `CONTEXT_STAGE = "bronze"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LAYER = "bronze"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_RUN_MODE = "train"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LOG_FILE = "bronze.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.bronze", log_filename=CONTEXT_L` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Shared aliases used throughout the notebook` | Documents the purpose or boundary of the surrounding notebook step. |
| `paths = CTX.paths` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_MAP = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE_CFG = CTX.stage_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BRONZE_CFG = CTX.stage_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RESOLVED_PATHS = CTX.resolved_paths` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FILENAMES = CTX.filenames` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `VERSIONS_CFG = CTX.versions` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUNTIME_CFG = CTX.runtime` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_CFG = CTX.dataset_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WANDB_CFG = CTX.wandb` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `EXECUTION_CFG = CTX.execution` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PIPELINE = CTX.pipeline` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger = CTX.logger` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger = CTX.ledger` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LOG_PATH = CTX.log_path` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_RECIPE_ID = CTX.recipe_id` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info( "Notebook context loaded", extra={ "stage": CONTEXT_STAGE, "recipe_id": CONTEXT_RECIPE_ID, "dataset": CONTEXT_DATASET, "mode": CONFIG_RUN_MODE, "profile": CONFIG_PROFI` | Writes a logger message for traceability during notebook execution. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="context_loaded", message="Loaded shared notebook context.", data={ "stage": CONTEXT_STAGE, "recipe_id": CONTEXT_RECIPE_ID, "dataset": CONTEXT_DATASET` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 04 — Notebook Level Configuration

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `Artifacts`
- `B`
- `Base`
- `bool`
- `Bronze`
- `bronze`
- `BRONZE_CFG`
- `bronze_observations_input_stage`
- `bronze_train_file_name`
- `build_truth_config_block`
- `capstone`
- `cast`
- `CONFIG`
- `CONFIG_RUN_MODE`
- `Configured`
- `data`
- `data_bronze_train_dir`
- `data_raw_dir`
- `DataFrame`

### Outputs

- `ADD_RECORD_ID`
- `ARTIFACTS_ROOT`
- `ASSET_ID`
- `BRONZE_DATA_PATH`
- `BRONZE_SOURCE_MODE`
- `BRONZE_TRAIN_DATA_FILE_NAME`
- `BRONZE_VERSION`
- `CONFIG_PROFILE`
- `DATASET_CANDIDATES`
- `DATASET_NAME`
- `DATASET_NAME_ARGUMENT`
- `DATASET_NAME_CONFIG`
- `DATASET_NAME_POSTGRES`
- `LABEL_SOURCE`
- `LABEL_SOURCE_DF`
- `LABEL_TYPE`
- `LABEL_TYPE_DF`
- `LAYER_NAME`
- `LAYER_SCHEMA`
- `LOGS_PATH`

### Key Operations

- `TRUTH_CONFIG = cast(Dict[str, Any], build_truth_config_block(CONFIG))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_CONFIG["pipeline"] = PIPELINE`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `# ---- Stage details ----`: Documents the purpose or boundary of the surrounding notebook step.
- `STAGE = "bronze"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `LAYER_NAME = str(BRONZE_CFG["layer_name"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `BRONZE_VERSION = str(VERSIONS_CFG["bronze"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_VERSION = str(VERSIONS_CFG["truth"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `PIPELINE_MODE = str(PIPELINE["execution_mode"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RUN_MODE = str(RUNTIME_CFG.get("mode", CONFIG_RUN_MODE))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_PROFILE = str(RUNTIME_CFG.get("profile", CONFIG_PROFILE))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `bool`
- `build_truth_config_block`
- `cast`
- `get`
- `lower`
- `make_process_run_id`
- `mkdir`
- `Path`
- `set_wandb_dir_from_config`
- `strip`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `TRUTH_CONFIG = cast(Dict[str, Any], build_truth_config_block(CONFIG))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_CONFIG["pipeline"] = PIPELINE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Stage details ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `STAGE = "bronze"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LAYER_NAME = str(BRONZE_CFG["layer_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BRONZE_VERSION = str(VERSIONS_CFG["bronze"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_VERSION = str(VERSIONS_CFG["truth"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `PIPELINE_MODE = str(PIPELINE["execution_mode"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_MODE = str(RUNTIME_CFG.get("mode", CONFIG_RUN_MODE))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = str(RUNTIME_CFG.get("profile", CONFIG_PROFILE))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Dataset details ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `DATASET_NAME_ARGUMENT = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_NAME_CONFIG = str(DATASET_CFG.get("name", "pump"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_NAME = DATASET_NAME_CONFIG.strip().lower()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_CANDIDATES = list(BRONZE_CFG["dataset_candidates"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SPLIT_STATUS = str(DATASET_CFG["split_status"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LABEL_TYPE = DATASET_CFG.get("label_type")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LABEL_SOURCE = DATASET_CFG.get("label_source")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_ID = str(DATASET_CFG["run_id"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ASSET_ID = str(DATASET_CFG["asset_id"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# DataFrame-friendly values.` | Documents the purpose or boundary of the surrounding notebook step. |
| `LABEL_TYPE_DF = pd.NA if LABEL_TYPE is None else LABEL_TYPE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LABEL_SOURCE_DF = pd.NA if LABEL_SOURCE is None else LABEL_SOURCE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Processing lineage ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `PROCESS_RUN_ID = make_process_run_id( str(BRONZE_CFG["process_run_id_prefix"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ADD_RECORD_ID = bool(BRONZE_CFG["add_record_id"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RECORD_ID_INPUTS = list(BRONZE_CFG["record_id_inputs"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RECORD_ID_METHOD = str(BRONZE_CFG["record_id_method"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- W&B ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `WANDB_PROJECT = str(WANDB_CFG.get("project", "capstone"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WANDB_ENTITY = str(WANDB_CFG.get("entity", ""))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WANDB_RUN_NAME = f"{BRONZE_VERSION}"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- File names ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `BRONZE_TRAIN_DATA_FILE_NAME = str(FILENAMES["bronze_train_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RAW_FILE_NAME = str(DATASET_CFG["raw_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Base paths only ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `ARTIFACTS_ROOT = Path(str(RESOLVED_PATHS["artifacts_root"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RAW_DATA_PATH = Path(str(RESOLVED_PATHS["data_raw_dir"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RAW_FILE_PATH = Path(str(RESOLVED_PATHS["raw_file_path"])).parent` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BRONZE_DATA_PATH = Path(str(RESOLVED_PATHS["data_bronze_train_dir"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTHS_PATH = Path(str(RESOLVED_PATHS["truths_dir"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_INDEX_PATH = Path(str(RESOLVED_PATHS["truth_index_path"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LOGS_PATH = Path(str(RESOLVED_PATHS["logs_root"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Source mode ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `BRONZE_SOURCE_MODE = "file" # "file" \| "postgres_handoff"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Postgres handoff defaults ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `DATASET_NAME_POSTGRES = DATASET_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `POSTGRES_SOURCE_TABLE_NAME = "bronze_observations_input_stage"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `POSTGRES_SOURCE_TABLE_DATASET_MAP = { POSTGRES_SOURCE_TABLE_NAME: str(DATASET_NAME_POSTGRES).strip(),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `17 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 05 — Defer Bronze Artifact Folder Creation Until the Dataset Name Is Resolved

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `after`
- `artifact`
- `Bronze`
- `creation`
- `dataset`
- `deferred`
- `directory`
- `resolution`
- `until`

### Outputs

- `BRONZE_ARTIFACT_DIRS`

### Key Operations

- `BRONZE_ARTIFACT_DIRS = {}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("Bronze artifact directory creation deferred until after dataset resolution.")`: Displays a notebook-facing result for inspection.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `BRONZE_ARTIFACT_DIRS = {}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Bronze artifact directory creation deferred until after dataset resolution.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 06 — SQL Runtime Context

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `accepting`
- `accidentally`
- `an`
- `Any`
- `Asset`
- `asset__001`
- `ASSET_ID`
- `asset_id`
- `ASSET_ID_DEFAULT_FALLBACK`
- `be`
- `candidates`
- `capstone`
- `CAPSTONE_ASSET_ID`
- `CAPSTONE_DATASET_ID`
- `CAPSTONE_RUN_ID`
- `CAPSTONE_SCHEMA`
- `cast`
- `CONFIG`
- `config`

### Outputs

- `dataset_config`
- `dataset_config_asset_id`
- `dataset_config_id`
- `dataset_config_name`
- `dataset_config_run_id`
- `engine`
- `first_non_empty_string`
- `is_synthetic_run`
- `raw_asset_id`
- `raw_dataset_id`
- `raw_run_id`
- `text_value`

### Key Operations

- `engine = get_engine_from_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CAPSTONE_SCHEMA: str = str(os.getenv("CAPSTONE_SCHEMA", "capstone"))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def first_non_empty_string(*values: object) -> str \| None: """ Return the first non-empty string-like value from a list of candidates. This helper skips None, empty strings, whites`: Defines notebook-local logic used later in the notebook.
- `dataset_config = ( cast(Dict[str, Any], CONFIG.get("dataset", {})) if isinstance(CONFIG.get("dataset", {}), dict) else {}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `dataset_config_name = dataset_config.get("name")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `dataset_config_id = dataset_config.get("dataset_id", dataset_config.get("id"))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `dataset_config_run_id = dataset_config.get("run_id")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `dataset_config_asset_id = dataset_config.get("asset_id")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `is_synthetic_run = str(RUN_MODE).lower() in { "synthetic", "synthetic_train", "synthetic_run", "simulation",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if is_synthetic_run: raw_dataset_id = first_non_empty_string( os.getenv("CAPSTONE_DATASET_ID"), os.getenv("SYNTHETIC_DATASET_ID"), globals().get("DATASET_ID"), dataset_config_id, g`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `cast`
- `first_non_empty_string`
- `get`
- `get_engine_from_env`
- `getenv`
- `globals`
- `isinstance`
- `lower`
- `strip`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `engine = get_engine_from_env()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CAPSTONE_SCHEMA: str = str(os.getenv("CAPSTONE_SCHEMA", "capstone"))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def first_non_empty_string(*values: object) -> str \| None: """ Return the first non-empty string-like value from a list of candidates. This helper skips None, empty strings, whites` | Defines notebook-local logic used later in the notebook. |
| `dataset_config = ( cast(Dict[str, Any], CONFIG.get("dataset", {})) if isinstance(CONFIG.get("dataset", {}), dict) else {}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `dataset_config_name = dataset_config.get("name")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `dataset_config_id = dataset_config.get("dataset_id", dataset_config.get("id"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `dataset_config_run_id = dataset_config.get("run_id")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `dataset_config_asset_id = dataset_config.get("asset_id")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `is_synthetic_run = str(RUN_MODE).lower() in { "synthetic", "synthetic_train", "synthetic_run", "simulation",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if is_synthetic_run: raw_dataset_id = first_non_empty_string( os.getenv("CAPSTONE_DATASET_ID"), os.getenv("SYNTHETIC_DATASET_ID"), globals().get("DATASET_ID"), dataset_config_id, g` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: raw_dataset_id = first_non_empty_string( os.getenv("CAPSTONE_DATASET_ID"), globals().get("DATASET_ID"), dataset_config_id, globals().get("DATASET_NAME_CONFIG"), dataset_confi` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if raw_dataset_id is None: raise ValueError( "DATASET_ID could not be resolved. " "Set CAPSTONE_DATASET_ID or configure CONFIG['dataset']['name'] / " "CONFIG['dataset']['dataset_id` | Controls validation, iteration, file handling, or error handling for this step. |
| `if raw_run_id is None: raise ValueError( "RUN_ID could not be resolved. " "Set CAPSTONE_RUN_ID, CONFIG['dataset']['run_id'], or default_fallbacks.run_id." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `if raw_asset_id is None: raise ValueError( "ASSET_ID could not be resolved. " "Set CAPSTONE_ASSET_ID, CONFIG['dataset']['asset_id'], or default_fallbacks.asset_id." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `DATASET_ID: str = raw_dataset_id` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `RUN_ID: str = raw_run_id` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ASSET_ID: str = raw_asset_id` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"SQL schema: {CAPSTONE_SCHEMA}")` | Displays a notebook-facing result for inspection. |
| `print(f"Dataset ID: {DATASET_ID}")` | Displays a notebook-facing result for inspection. |
| `print(f"Run ID: {RUN_ID}")` | Displays a notebook-facing result for inspection. |
| `print(f"Asset ID: {ASSET_ID}")` | Displays a notebook-facing result for inspection. |
| `print(f"Synthetic run: {is_synthetic_run}")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 07 — SQL Runtime Context

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `are`
- `available`
- `check`
- `CONFIG`
- `CONFIG_MAP`
- `Context`
- `context`
- `context_sanity_check`
- `CTX`
- `DataFrame`
- `dataset`
- `DATASET_CFG`
- `EXECUTION_CFG`
- `f`
- `FILENAMES`
- `globals`
- `info`
- `ledger`
- `LOG_PATH`

### Outputs

- `data`
- `extra`
- `kind`
- `logger`
- `message`
- `missing_context_vars`
- `required_context_vars`
- `step`

### Key Operations

- `required_context_vars = [ "CTX", "paths", "CONFIG", "CONFIG_MAP", "STAGE_CFG", "RESOLVED_PATHS", "FILENAMES", "VERSIONS_CFG", "RUNTIME_CFG", "DATASET_CFG", "WANDB_CFG", "EXECUTION_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_context_vars = [name for name in required_context_vars if name not in globals()]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if missing_context_vars: raise NameError(f"Missing required shared context variables: {missing_context_vars}")`: Controls validation, iteration, file handling, or error handling for this step.
- `logger.info( "Context sanity check passed", extra={ "stage": CTX.stage, "recipe_id": CTX.recipe_id, "dataset": CTX.dataset, "mode": CTX.mode, "profile": CTX.profile, "log_path": st`: Writes a logger message for traceability during notebook execution.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="check", step="context_sanity_check", message="Verified shared notebook context variables are available.", data={ "stage": CTX.stage, "recipe_id": CTX.recipe_id, "`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `pd.DataFrame( [ {"name": name, "status": "present"} for name in required_context_vars ]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `DataFrame`
- `globals`
- `info`
- `NameError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `required_context_vars = [ "CTX", "paths", "CONFIG", "CONFIG_MAP", "STAGE_CFG", "RESOLVED_PATHS", "FILENAMES", "VERSIONS_CFG", "RUNTIME_CFG", "DATASET_CFG", "WANDB_CFG", "EXECUTION_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_context_vars = [name for name in required_context_vars if name not in globals()]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if missing_context_vars: raise NameError(f"Missing required shared context variables: {missing_context_vars}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `logger.info( "Context sanity check passed", extra={ "stage": CTX.stage, "recipe_id": CTX.recipe_id, "dataset": CTX.dataset, "mode": CTX.mode, "profile": CTX.profile, "log_path": st` | Writes a logger message for traceability during notebook execution. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="check", step="context_sanity_check", message="Verified shared notebook context variables are available.", data={ "stage": CTX.stage, "recipe_id": CTX.recipe_id, "` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `pd.DataFrame( [ {"name": name, "status": "present"} for name in required_context_vars ]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 08 — SQL Runtime Context

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Bronze`
- `BRONZE_CFG`
- `check`
- `context`
- `DEFAULT_FALLBACKS`
- `f`
- `globals`
- `info`
- `logger`
- `Missing`
- `name`
- `NameError`
- `passed`
- `raise`
- `sanity`
- `variables`

### Outputs

- `bronze_required_context_vars`
- `missing_bronze_context_vars`

### Key Operations

- `bronze_required_context_vars = [ "BRONZE_CFG", "DEFAULT_FALLBACKS",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_bronze_context_vars = [ name for name in bronze_required_context_vars if name not in globals()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if missing_bronze_context_vars: raise NameError(f"Missing Bronze context variables: {missing_bronze_context_vars}")`: Controls validation, iteration, file handling, or error handling for this step.
- `logger.info("Bronze context sanity check passed")`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `globals`
- `info`
- `NameError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `bronze_required_context_vars = [ "BRONZE_CFG", "DEFAULT_FALLBACKS",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_bronze_context_vars = [ name for name in bronze_required_context_vars if name not in globals()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_bronze_context_vars: raise NameError(f"Missing Bronze context variables: {missing_bronze_context_vars}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `logger.info("Bronze context sanity check passed")` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 09 — SQL Smoke Check

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `bronze`
- `BY`
- `capstone_schema`
- `CAPSTONE_SCHEMA`
- `engine`
- `gold`
- `information_schema`
- `metadata`
- `ORDER`
- `read_sql_dataframe`
- `SELECT`
- `silver`
- `table_name`
- `table_schema`
- `tables`
- `WHERE`

### Outputs

- `params`
- `sql_smoke_check_dataframe`

### Key Operations

- `sql_smoke_check_dataframe = read_sql_dataframe( engine, """ SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema IN (:capstone_schema, 'bronze', 'silve`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(sql_smoke_check_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `IN`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `sql_smoke_check_dataframe = read_sql_dataframe( engine, """ SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema IN (:capstone_schema, 'bronze', 'silve` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(sql_smoke_check_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 10 — Start Logging for the Bronze Stage

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `bronze`
- `Bronze`
- `capstone`
- `configure_logging`
- `Create`
- `current_layer`
- `DEBUG`
- `directory`
- `exist_ok`
- `exists`
- `file`
- `getLogger`
- `info`
- `Initial`
- `Initiate`
- `initiation`
- `Insure`
- `load`
- `log`
- `Log`

### Outputs

- `bronze_log_path`
- `capstone_logger`
- `level`
- `logger`
- `overwrite_handlers`

### Key Operations

- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Create bronze log path`: Documents the purpose or boundary of the surrounding notebook step.
- `bronze_log_path = paths.logs / "bronze.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Insure directory exists`: Documents the purpose or boundary of the surrounding notebook step.
- `bronze_log_path.parent.mkdir(parents=True, exist_ok=True)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Initial Logger`: Documents the purpose or boundary of the surrounding notebook step.
- `capstone_logger = configure_logging( "capstone", bronze_log_path, level=logging.DEBUG, overwrite_handlers=True,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Initiate Logger and log file`: Documents the purpose or boundary of the surrounding notebook step.
- `logger = logging.getLogger("capstone.bronze")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Log load and initiation`: Documents the purpose or boundary of the surrounding notebook step.
- `logger.info("Bronze stage starting")`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `configure_logging`
- `getLogger`
- `info`
- `log_layer_paths`
- `mkdir`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Create bronze log path` | Documents the purpose or boundary of the surrounding notebook step. |
| `bronze_log_path = paths.logs / "bronze.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Insure directory exists` | Documents the purpose or boundary of the surrounding notebook step. |
| `bronze_log_path.parent.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Initial Logger` | Documents the purpose or boundary of the surrounding notebook step. |
| `capstone_logger = configure_logging( "capstone", bronze_log_path, level=logging.DEBUG, overwrite_handlers=True,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Initiate Logger and log file` | Documents the purpose or boundary of the surrounding notebook step. |
| `logger = logging.getLogger("capstone.bronze")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Log load and initiation` | Documents the purpose or boundary of the surrounding notebook step. |
| `logger.info("Bronze stage starting")` | Writes a logger message for traceability during notebook execution. |
| `log_layer_paths(paths, current_layer="bronze", logger=logger)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 11 — Start Logging for the Bronze Stage

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_paths_logged`
- `add`
- `CONTEXT_LAYER`
- `CONTEXT_STAGE`
- `current_layer`
- `f`
- `info`
- `layer`
- `ledger`
- `log_layer_paths`
- `log_path`
- `LOG_PATH`
- `Logged`
- `logged`
- `paths`
- `project`
- `Project`
- `s`
- `stage`

### Outputs

- `data`
- `extra`
- `kind`
- `logger`
- `message`
- `step`

### Key Operations

- `log_layer_paths(paths, current_layer=CONTEXT_LAYER, logger=logger)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger.info( "Project paths logged for %s layer", CONTEXT_LAYER, extra={"stage": CONTEXT_STAGE, "layer": CONTEXT_LAYER},`: Writes a logger message for traceability during notebook execution.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step=f"{CONTEXT_LAYER}_paths_logged", message="Logged project layer paths.", data={ "stage": CONTEXT_STAGE, "layer": CONTEXT_LAYER, "log_path": str(LOG_PAT`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `info`
- `log_layer_paths`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `log_layer_paths(paths, current_layer=CONTEXT_LAYER, logger=logger)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info( "Project paths logged for %s layer", CONTEXT_LAYER, extra={"stage": CONTEXT_STAGE, "layer": CONTEXT_LAYER},` | Writes a logger message for traceability during notebook execution. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step=f"{CONTEXT_LAYER}_paths_logged", message="Logged project layer paths.", data={ "stage": CONTEXT_STAGE, "layer": CONTEXT_LAYER, "log_path": str(LOG_PAT` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 12 — Define Dataset Name Resolution Logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `__`
- `an`
- `append`
- `argument`
- `Argument`
- `argument_value`
- `based`
- `before`
- `Bronze`
- `character`
- `CLI`
- `Config`
- `config`
- `config_value`
- `dataset`
- `Dataset`
- `dataset_name`
- `def`
- `details`

### Outputs

- `_generate_deterministic_dataset_name_from_file_details`
- `_normalize_dataset_name`
- `cleaned_characters`
- `content_fingerprint`
- `fallback_value_text`
- `file_size_bytes`
- `file_stem_normalized`
- `file_stem_raw`
- `first_chunk`
- `generated_dataset_name`
- `last_chunk`
- `mapped_dataset_name`
- `modified_timestamp`
- `normalized_value`
- `path_object`
- `resolve_dataset_name_for_bronze_pre_ingest`
- `sample_hasher`
- `seek_position`
- `source_table_dataset_map`
- `stat_result`

### Key Operations

- `def resolve_dataset_name_for_bronze_pre_ingest( *, argument_value: Optional[str] = None, config_value: Optional[str] = None, handoff_dataset_name: Optional[str] = None, source_tabl`: Defines notebook-local logic used later in the notebook.
- `) -> Tuple[str, str, str]: """ Resolve dataset name before Bronze ingestion. Priority order: 1. CLI / Argument 2. Config File 3. Explicit handoff dataset name 4. Source table -> da`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `_generate_deterministic_dataset_name_from_file_details`
- `_normalize_dataset_name`
- `append`
- `encode`
- `exists`
- `get`
- `hexdigest`
- `is_file`
- `isalnum`
- `join`
- `lower`
- `max`
- `open`
- `Path`
- `read`
- `replace`
- `resolve_dataset_name_for_bronze_pre_ingest`
- `return`
- `seek`
- `sha1`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def resolve_dataset_name_for_bronze_pre_ingest( *, argument_value: Optional[str] = None, config_value: Optional[str] = None, handoff_dataset_name: Optional[str] = None, source_tabl` | Defines notebook-local logic used later in the notebook. |
| `) -> Tuple[str, str, str]: """ Resolve dataset name before Bronze ingestion. Priority order: 1. CLI / Argument 2. Config File 3. Explicit handoff dataset name 4. Source table -> da` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 13 — Resolve the Provisional Dataset Identity

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BRONZE_SOURCE_MODE`
- `DATASET_NAME_ARGUMENT`
- `DATASET_NAME_CONFIG`
- `DATASET_NAME_POSTGRES`
- `else`
- `file`
- `postgres_handoff`
- `POSTGRES_SOURCE_TABLE_DATASET_MAP`
- `POSTGRES_SOURCE_TABLE_NAME`
- `PROVISIONAL_DATASET_METHOD`
- `PROVISIONAL_DATASET_NAME`
- `PROVISIONAL_DATASET_SOURCE_COLUMN`
- `RAW_FILE_NAME`
- `RAW_FILE_PATH`
- `resolve_dataset_name_for_bronze_pre_ingest`
- `unknown_dataset`

### Outputs

- `argument_value`
- `config_value`
- `fallback_value`
- `handoff_dataset_name`
- `source_path`
- `source_table_dataset_map`
- `source_table_name`

### Key Operations

- `PROVISIONAL_DATASET_NAME, PROVISIONAL_DATASET_SOURCE_COLUMN, PROVISIONAL_DATASET_METHOD = ( resolve_dataset_name_for_bronze_pre_ingest( argument_value=DATASET_NAME_ARGUMENT, config`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `resolve_dataset_name_for_bronze_pre_ingest`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `PROVISIONAL_DATASET_NAME, PROVISIONAL_DATASET_SOURCE_COLUMN, PROVISIONAL_DATASET_METHOD = ( resolve_dataset_name_for_bronze_pre_ingest( argument_value=DATASET_NAME_ARGUMENT, config` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 14 — Prepare Dataset Resolution Metadata

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `astype`
- `attrs`
- `Bronze`
- `column`
- `columns`
- `contains`
- `Could`
- `dataframe`
- `dataset`
- `dataset_column`
- `dataset_name`
- `dataset_resolution`
- `def`
- `dropna`
- `elif`
- `exactly`
- `exists`
- `f`
- `fallback_dataset_name`
- `fallback_method`

### Outputs

- `dataset_method`
- `dataset_source_column`
- `dataset_values`
- `resolved_dataset_name`
- `unique_dataset_values`
- `write_dataset_resolution_attrs`

### Key Operations

- `def write_dataset_resolution_attrs( dataframe, *, dataset_column: str = "meta__dataset", fallback_dataset_name: str \| None = None, fallback_method: str = "fallback_dataset_name",`: Defines notebook-local logic used later in the notebook.
- `): """ Write Bronze dataset resolution metadata into dataframe.attrs. Resolution order: 1. Use dataset_column if it exists and contains exactly one non-null value 2. Otherwise use `: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `dropna`
- `sorted`
- `strip`
- `unique`
- `ValueError`
- `write_dataset_resolution_attrs`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def write_dataset_resolution_attrs( dataframe, *, dataset_column: str = "meta__dataset", fallback_dataset_name: str \| None = None, fallback_method: str = "fallback_dataset_name",` | Defines notebook-local logic used later in the notebook. |
| `): """ Write Bronze dataset resolution metadata into dataframe.attrs. Resolution order: 1. Use dataset_column if it exists and contains exactly one non-null value 2. Otherwise use ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 15 — Defer the Config Snapshot Until the Final Dataset Name Exists

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `after`
- `be`
- `Config`
- `dataset`
- `final`
- `resolution`
- `saved`
- `snapshot`
- `will`

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

## Code Cell 16 — Initialize the Experiment Tracking Run

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `B`
- `BRONZE_SOURCE_MODE`
- `elif`
- `else`
- `f`
- `file`
- `postgres`
- `postgres_handoff`
- `POSTGRES_SOURCE_TABLE_NAME`
- `public`
- `raise`
- `RAW_FILE_NAME`
- `RAW_FILE_PATH`
- `setup`
- `Unsupported`
- `ValueError`
- `W`

### Outputs

- `WANDB_RAW_DATA_FILE`
- `WANDB_RAW_PATH`
- `WANDB_SOURCE_KIND`
- `WANDB_SOURCE_REFERENCE`
- `WANDB_SOURCE_TABLE_NAME`

### Key Operations

- `if BRONZE_SOURCE_MODE == "file": WANDB_SOURCE_KIND = "file" WANDB_SOURCE_REFERENCE = str(RAW_FILE_PATH / RAW_FILE_NAME) WANDB_SOURCE_TABLE_NAME = None WANDB_RAW_DATA_FILE = RAW_FIL`: Controls validation, iteration, file handling, or error handling for this step.
- `elif BRONZE_SOURCE_MODE == "postgres_handoff": WANDB_SOURCE_KIND = "postgres_handoff" WANDB_SOURCE_REFERENCE = f"postgres:public.{POSTGRES_SOURCE_TABLE_NAME}" WANDB_SOURCE_TABLE_NA`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `else: raise ValueError(f"Unsupported BRONZE_SOURCE_MODE for W&B setup: {BRONZE_SOURCE_MODE}")`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `if BRONZE_SOURCE_MODE == "file": WANDB_SOURCE_KIND = "file" WANDB_SOURCE_REFERENCE = str(RAW_FILE_PATH / RAW_FILE_NAME) WANDB_SOURCE_TABLE_NAME = None WANDB_RAW_DATA_FILE = RAW_FIL` | Controls validation, iteration, file handling, or error handling for this step. |
| `elif BRONZE_SOURCE_MODE == "postgres_handoff": WANDB_SOURCE_KIND = "postgres_handoff" WANDB_SOURCE_REFERENCE = f"postgres:public.{POSTGRES_SOURCE_TABLE_NAME}" WANDB_SOURCE_TABLE_NA` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `else: raise ValueError(f"Unsupported BRONZE_SOURCE_MODE for W&B setup: {BRONZE_SOURCE_MODE}")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 17 — Start experiment tracking for this stage

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add_record_id`
- `ADD_RECORD_ID`
- `ASSET_ID`
- `asset_id`
- `B`
- `bronze`
- `BRONZE_DATA_PATH`
- `bronze_out_path`
- `BRONZE_SOURCE_MODE`
- `BRONZE_VERSION`
- `bronze_version`
- `dataset_name_provisional`
- `dataset_resolution_stage`
- `else`
- `info`
- `init`
- `initialized`
- `LABEL_SOURCE`
- `label_source`
- `LABEL_TYPE`

### Outputs

- `config`
- `entity`
- `job_type`
- `name`
- `project`
- `wandb_run`

### Key Operations

- `wandb_run = wandb.init( project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="bronze", config={ "dataset_name_provisional": PROVISIONAL_DATASET_NAME, "dataset_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger.info("W&B initialized: %s", wandb_run.name)`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `info`
- `init`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `wandb_run = wandb.init( project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="bronze", config={ "dataset_name_provisional": PROVISIONAL_DATASET_NAME, "dataset_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("W&B initialized: %s", wandb_run.name)` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- Artifact or state outputs detected: optional experiment tracking call.

## Code Cell 18 — Ingest the Raw Dataset into the Bronze Layer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `batch_id`
- `bronze_observations_input_stage`
- `BRONZE_SOURCE_MODE`
- `dataset_id`
- `DATASET_NAME_ARGUMENT`
- `DATASET_NAME_POSTGRES`
- `elif`
- `else`
- `f`
- `file`
- `get_engine_from_env`
- `ingest_data`
- `meta__dataset`
- `postgres_handoff`
- `public`
- `raise`
- `RAW_FILE_NAME`
- `RAW_FILE_PATH`
- `read_layer_dataframe`
- `row_in_batch`

### Outputs

- `add_record_id`
- `asset_id`
- `dataframe`
- `dataset_candidates`
- `dataset_column`
- `dataset_name`
- `dataset_name_config`
- `engine`
- `fallback_dataset_name`
- `fallback_method`
- `file_name`
- `order_by`
- `params`
- `require_exists`
- `run_id`
- `schema`
- `split`
- `table_name`
- `validate`
- `where_clause`

### Key Operations

- `if BRONZE_SOURCE_MODE == "file": dataframe = ingest_data( RAW_FILE_PATH, file_name=RAW_FILE_NAME, dataset_name=DATASET_NAME_ARGUMENT, dataset_name_config=DATASET_NAME_CONFIG, datas`: Controls validation, iteration, file handling, or error handling for this step.
- `elif BRONZE_SOURCE_MODE == "postgres_handoff": engine = get_engine_from_env() dataframe = read_layer_dataframe( engine=engine, schema="public", table_name="bronze_observations_inpu`: Loads input data, configuration, or artifacts required by the current stage.
- `else: raise ValueError(f"Unsupported BRONZE_SOURCE_MODE: {BRONZE_SOURCE_MODE}")`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `get_engine_from_env`
- `ingest_data`
- `read_layer_dataframe`
- `ValueError`
- `write_dataset_resolution_attrs`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `if BRONZE_SOURCE_MODE == "file": dataframe = ingest_data( RAW_FILE_PATH, file_name=RAW_FILE_NAME, dataset_name=DATASET_NAME_ARGUMENT, dataset_name_config=DATASET_NAME_CONFIG, datas` | Controls validation, iteration, file handling, or error handling for this step. |
| `elif BRONZE_SOURCE_MODE == "postgres_handoff": engine = get_engine_from_env() dataframe = read_layer_dataframe( engine=engine, schema="public", table_name="bronze_observations_inpu` | Loads input data, configuration, or artifacts required by the current stage. |
| `else: raise ValueError(f"Unsupported BRONZE_SOURCE_MODE: {BRONZE_SOURCE_MODE}")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 19 — Confirm the Final Dataset Identity After Ingestion

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `attrs`
- `Bronze`
- `dataframe`
- `dataset_name`
- `did`
- `get`
- `ingest`
- `metadata`
- `raise`
- `to`
- `ValueError`
- `write`

### Outputs

- `DATASET_METHOD`
- `dataset_resolution`
- `DATASET_SOURCE_COLUMN`
- `RESOLVED_DATASET_NAME`

### Key Operations

- `dataset_resolution = dataframe.attrs.get("dataset_resolution", {})`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if not dataset_resolution: raise ValueError( "Bronze ingest did not write dataset_resolution metadata to dataframe.attrs." )`: Controls validation, iteration, file handling, or error handling for this step.
- `RESOLVED_DATASET_NAME = dataset_resolution.get("dataset_name")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_SOURCE_COLUMN = dataset_resolution.get("dataset_source_column")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_METHOD = dataset_resolution.get("dataset_method")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if RESOLVED_DATASET_NAME is None: raise ValueError("Bronze ingest did not return dataset_resolution metadata in dataframe.attrs.")`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `get`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `dataset_resolution = dataframe.attrs.get("dataset_resolution", {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not dataset_resolution: raise ValueError( "Bronze ingest did not write dataset_resolution metadata to dataframe.attrs." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `RESOLVED_DATASET_NAME = dataset_resolution.get("dataset_name")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_SOURCE_COLUMN = dataset_resolution.get("dataset_source_column")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_METHOD = dataset_resolution.get("dataset_method")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if RESOLVED_DATASET_NAME is None: raise ValueError("Bronze ingest did not return dataset_resolution metadata in dataframe.attrs.")` | Controls validation, iteration, file handling, or error handling for this step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 20 — Create the Bronze Artifact Folders with the Artifact Util

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__bronze__artifact_summary`
- `__bronze__resolved_config`
- `artifact`
- `Bronze`
- `bronze`
- `bronze_ledger_file_name`
- `build_artifact_dirs`
- `Config`
- `CONFIG`
- `config`
- `execution`
- `export_config_snapshot`
- `f`
- `FILENAMES`
- `get`
- `json`
- `lineage`
- `metadata`
- `profiles`
- `quality`

### Outputs

- `artifacts_root`
- `BRONZE_ARTIFACT_DIRS`
- `BRONZE_ARTIFACT_SUMMARY_PATH`
- `BRONZE_ARTIFACTS_PATH`
- `BRONZE_CONFIG_DIR`
- `bronze_ledger_path`
- `BRONZE_LINEAGE_DIR`
- `BRONZE_METADATA_DIR`
- `BRONZE_PROFILE_DIR`
- `BRONZE_QUALITY_DIR`
- `BRONZE_SCHEMA_DIR`
- `BRONZE_SUMMARY_DIR`
- `CONFIG_SNAPSHOT_PATH`
- `dataset_name`
- `family`
- `stage`
- `subdirs`

### Key Operations

- `BRONZE_ARTIFACT_DIRS = build_artifact_dirs( artifacts_root=ARTIFACTS_ROOT, stage="bronze", dataset_name=RESOLVED_DATASET_NAME, family=None, subdirs=[ "profiles", "quality", "schema`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `BRONZE_ARTIFACTS_PATH = BRONZE_ARTIFACT_DIRS["root"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `BRONZE_PROFILE_DIR = BRONZE_ARTIFACT_DIRS["profiles"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `BRONZE_QUALITY_DIR = BRONZE_ARTIFACT_DIRS["quality"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `BRONZE_SCHEMA_DIR = BRONZE_ARTIFACT_DIRS["schema"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `BRONZE_SUMMARY_DIR = BRONZE_ARTIFACT_DIRS["summaries"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `BRONZE_METADATA_DIR = BRONZE_ARTIFACT_DIRS["metadata"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `BRONZE_CONFIG_DIR = BRONZE_ARTIFACT_DIRS["config"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `BRONZE_LINEAGE_DIR = BRONZE_ARTIFACT_DIRS["lineage"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_SNAPSHOT_PATH = ( BRONZE_CONFIG_DIR / f"{RESOLVED_DATASET_NAME}__bronze__resolved_config.yaml"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `build_artifact_dirs`
- `export_config_snapshot`
- `get`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `BRONZE_ARTIFACT_DIRS = build_artifact_dirs( artifacts_root=ARTIFACTS_ROOT, stage="bronze", dataset_name=RESOLVED_DATASET_NAME, family=None, subdirs=[ "profiles", "quality", "schema` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BRONZE_ARTIFACTS_PATH = BRONZE_ARTIFACT_DIRS["root"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BRONZE_PROFILE_DIR = BRONZE_ARTIFACT_DIRS["profiles"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BRONZE_QUALITY_DIR = BRONZE_ARTIFACT_DIRS["quality"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BRONZE_SCHEMA_DIR = BRONZE_ARTIFACT_DIRS["schema"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BRONZE_SUMMARY_DIR = BRONZE_ARTIFACT_DIRS["summaries"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BRONZE_METADATA_DIR = BRONZE_ARTIFACT_DIRS["metadata"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BRONZE_CONFIG_DIR = BRONZE_ARTIFACT_DIRS["config"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BRONZE_LINEAGE_DIR = BRONZE_ARTIFACT_DIRS["lineage"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_SNAPSHOT_PATH = ( BRONZE_CONFIG_DIR / f"{RESOLVED_DATASET_NAME}__bronze__resolved_config.yaml"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BRONZE_ARTIFACT_SUMMARY_PATH = ( BRONZE_SUMMARY_DIR / f"{RESOLVED_DATASET_NAME}__bronze__artifact_summary.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `bronze_ledger_path = ( BRONZE_LINEAGE_DIR / FILENAMES["bronze_ledger_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if CONFIG["execution"].get("save_config_snapshot", True): export_config_snapshot(CONFIG, CONFIG_SNAPSHOT_PATH)` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Bronze artifact root:", BRONZE_ARTIFACTS_PATH)` | Displays a notebook-facing result for inspection. |
| `print("Config snapshot path:", CONFIG_SNAPSHOT_PATH)` | Displays a notebook-facing result for inspection. |
| `BRONZE_ARTIFACT_DIRS` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 21 — Update Tracking Metadata with the Final Dataset Name

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `bronze_ingest_final`
- `BRONZE_SOURCE_MODE`
- `config`
- `dataset_method`
- `DATASET_METHOD`
- `dataset_name_final`
- `dataset_resolution_stage`
- `dataset_source_column`
- `DATASET_SOURCE_COLUMN`
- `RESOLVED_DATASET_NAME`
- `source_kind`
- `source_mode`
- `source_reference`
- `source_table_name`
- `update`
- `wandb_run`
- `WANDB_SOURCE_KIND`
- `WANDB_SOURCE_REFERENCE`
- `WANDB_SOURCE_TABLE_NAME`

### Outputs

- `allow_val_change`

### Key Operations

- `wandb_run.config.update( { "dataset_name_final": RESOLVED_DATASET_NAME, "dataset_source_column": DATASET_SOURCE_COLUMN, "dataset_method": DATASET_METHOD, "dataset_resolution_stage"`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `update`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `wandb_run.config.update( { "dataset_name_final": RESOLVED_DATASET_NAME, "dataset_source_column": DATASET_SOURCE_COLUMN, "dataset_method": DATASET_METHOD, "dataset_resolution_stage"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 22 — Log Any Dataset Name Changes

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `after`
- `available`
- `became`
- `Bronze`
- `changed`
- `Dataset`
- `dataset`
- `evidence`
- `info`
- `ingest`
- `inside`
- `logger`
- `name`
- `provisional`
- `PROVISIONAL_DATASET_NAME`
- `resolved`
- `RESOLVED_DATASET_NAME`
- `s`
- `when`

### Outputs

- `DATASET_NAME`

### Key Operations

- `if PROVISIONAL_DATASET_NAME != RESOLVED_DATASET_NAME: logger.info( "Dataset name changed after Bronze ingest when inside-dataset evidence became available \| provisional=%s \| resolv`: Writes a logger message for traceability during notebook execution.
- `DATASET_NAME = RESOLVED_DATASET_NAME`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `info`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `if PROVISIONAL_DATASET_NAME != RESOLVED_DATASET_NAME: logger.info( "Dataset name changed after Bronze ingest when inside-dataset evidence became available \| provisional=%s \| resolv` | Writes a logger message for traceability during notebook execution. |
| `DATASET_NAME = RESOLVED_DATASET_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 23 — Build the Bronze Truth Record Foundation

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add_record_id`
- `ADD_RECORD_ID`
- `artifact_paths`
- `asset_id`
- `ASSET_ID`
- `Bronze`
- `BRONZE_DATA_PATH`
- `bronze_output_dir`
- `bronze_output_file_name`
- `bronze_version`
- `BRONZE_VERSION`
- `config_snapshot`
- `dataset_method`
- `DATASET_METHOD`
- `dataset_name_final`
- `dataset_name_provisional`
- `DATASET_SOURCE_COLUMN`
- `dataset_source_column`
- `else`
- `ingestion`

### Outputs

- `bronze_truth`
- `dataset_name`
- `layer_name`
- `parent_truth_hash`
- `pipeline_mode`
- `process_run_id`
- `truth_version`

### Key Operations

- `bronze_truth = initialize_layer_truth( truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name=LAYER_NAME, process_run_id=PROCESS_RUN_ID, pipeline_mode=PIPELINE_MODE, pa`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `bronze_truth = update_truth_section( bronze_truth, "config_snapshot", { "bronze_version": BRONZE_VERSION, "split_status": SPLIT_STATUS, "label_type": LABEL_TYPE, "label_source": LA`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `bronze_truth = update_truth_section( bronze_truth, "runtime_facts", { "source_run_id": RUN_ID, "raw_file_path": str(RAW_FILE_PATH / RAW_FILE_NAME), "raw_data_dir": str(RAW_DATA_PAT`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `bronze_truth = update_truth_section( bronze_truth, "artifact_paths", { "bronze_output_dir": str(BRONZE_DATA_PATH), "bronze_output_file_name": "pump__bronze__train.parquet", },`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `bronze_truth = update_truth_section( bronze_truth, "notes", { "purpose": "Bronze ingestion truth record", },`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `initialize_layer_truth`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `bronze_truth = initialize_layer_truth( truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name=LAYER_NAME, process_run_id=PROCESS_RUN_ID, pipeline_mode=PIPELINE_MODE, pa` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `bronze_truth = update_truth_section( bronze_truth, "config_snapshot", { "bronze_version": BRONZE_VERSION, "split_status": SPLIT_STATUS, "label_type": LABEL_TYPE, "label_source": LA` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `bronze_truth = update_truth_section( bronze_truth, "runtime_facts", { "source_run_id": RUN_ID, "raw_file_path": str(RAW_FILE_PATH / RAW_FILE_NAME), "raw_data_dir": str(RAW_DATA_PAT` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `bronze_truth = update_truth_section( bronze_truth, "artifact_paths", { "bronze_output_dir": str(BRONZE_DATA_PATH), "bronze_output_file_name": "pump__bronze__train.parquet", },` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `bronze_truth = update_truth_section( bronze_truth, "notes", { "purpose": "Bronze ingestion truth record", },` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 24 — Bronze Data Review

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `All`
- `ascending`
- `Basic`
- `block`
- `categorical`
- `category`
- `column`
- `Columns`
- `columns`
- `compact`
- `d`
- `dataframe`
- `Dataframe`
- `deep`
- `Describe`
- `describe`
- `Dtypes`
- `dtypes`
- `else`

### Outputs

- `desc_num`
- `desc_obj`
- `mem_mb`
- `meta_columns`
- `missing`
- `object_category_columns`
- `other_columns`

### Key Operations

- `# Basic Dataframe Information/Summary`: Documents the purpose or boundary of the surrounding notebook step.
- `# Shape`: Documents the purpose or boundary of the surrounding notebook step.
- `print("Shape:", dataframe.shape)`: Displays a notebook-facing result for inspection.
- `logger.info("Shape: %s", dataframe.shape)`: Writes a logger message for traceability during notebook execution.
- `# Dtypes as a compact block`: Documents the purpose or boundary of the surrounding notebook step.
- `print("\nData types:")`: Displays a notebook-facing result for inspection.
- `print(dataframe.dtypes)`: Displays a notebook-facing result for inspection.
- `logger.info("Dtypes:\n%s", dataframe.dtypes.to_string())`: Writes a logger message for traceability during notebook execution.
- `# Memory Usages`: Documents the purpose or boundary of the surrounding notebook step.
- `print("\nMemory usage (MB):")`: Displays a notebook-facing result for inspection.
- `print(dataframe.memory_usage(deep=True).sum() / (1024 ** 2))`: Displays a notebook-facing result for inspection.
- `mem_mb = dataframe.memory_usage(deep=True).sum() / (1024 ** 2)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `describe`
- `display`
- `Head`
- `head`
- `info`
- `isna`
- `join`
- `mean`
- `memory_usage`
- `missingness`
- `select_dtypes`
- `sort_values`
- `startswith`
- `sum`
- `text`
- `to_frame`
- `to_string`
- `usage`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Basic Dataframe Information/Summary` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Shape` | Documents the purpose or boundary of the surrounding notebook step. |
| `print("Shape:", dataframe.shape)` | Displays a notebook-facing result for inspection. |
| `logger.info("Shape: %s", dataframe.shape)` | Writes a logger message for traceability during notebook execution. |
| `# Dtypes as a compact block` | Documents the purpose or boundary of the surrounding notebook step. |
| `print("\nData types:")` | Displays a notebook-facing result for inspection. |
| `print(dataframe.dtypes)` | Displays a notebook-facing result for inspection. |
| `logger.info("Dtypes:\n%s", dataframe.dtypes.to_string())` | Writes a logger message for traceability during notebook execution. |
| `# Memory Usages` | Documents the purpose or boundary of the surrounding notebook step. |
| `print("\nMemory usage (MB):")` | Displays a notebook-facing result for inspection. |
| `print(dataframe.memory_usage(deep=True).sum() / (1024 ** 2))` | Displays a notebook-facing result for inspection. |
| `mem_mb = dataframe.memory_usage(deep=True).sum() / (1024 ** 2)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Memory usage (MB): %.2f", mem_mb)` | Writes a logger message for traceability during notebook execution. |
| `# Head(15) as text (truncate columns for readability)` | Documents the purpose or boundary of the surrounding notebook step. |
| `print("\nFirst 15 rows:")` | Displays a notebook-facing result for inspection. |
| `display(dataframe.head(15))` | Displays a notebook-facing result for inspection. |
| `logger.info("Head(15):\n%s", dataframe.head(15).to_string(max_cols=40, max_rows=15))` | Writes a logger message for traceability during notebook execution. |
| `# Describe Numbers` | Documents the purpose or boundary of the surrounding notebook step. |
| `print("\nBasic numeric summary:")` | Displays a notebook-facing result for inspection. |
| `display(dataframe.describe(include=[np.number]).T)` | Displays a notebook-facing result for inspection. |
| `desc_num = dataframe.describe(include=[np.number]).T` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Numeric describe (truncated):\n%s", desc_num.to_string(max_rows=60))` | Writes a logger message for traceability during notebook execution. |
| `# Describe Objects and categorical` | Documents the purpose or boundary of the surrounding notebook step. |
| `print("\nBasic object / categorical summary:")` | Displays a notebook-facing result for inspection. |
| `object_category_columns = dataframe.select_dtypes(include=["object", "category", "string"]).columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(object_category_columns): desc_obj = dataframe[object_category_columns].describe().T display(desc_obj) logger.info("Object/categorical describe (truncated):\n%s", desc_obj.t` | Writes a logger message for traceability during notebook execution. |
| `else: logger.info("No object/category/string columns to describe.")` | Writes a logger message for traceability during notebook execution. |
| `# Meta Columns` | Documents the purpose or boundary of the surrounding notebook step. |
| `print("\nMeta Columns Summary:")` | Displays a notebook-facing result for inspection. |
| `meta_columns = [column for column in dataframe.columns if column.startswith("meta__")]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `dataframe[meta_columns].head(3)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("Meta Columns: (%d): %s", len(meta_columns), "\n".join(meta_columns))` | Writes a logger message for traceability during notebook execution. |
| `# All Other Columns` | Documents the purpose or boundary of the surrounding notebook step. |
| `print("\nAll Other Columns Summary:")` | Displays a notebook-facing result for inspection. |
| `other_columns = [column for column in dataframe.columns if not column.startswith("meta__")]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `dataframe[other_columns].head(3)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("All Other Columns: (%d): %s", len(other_columns), "\n".join(other_columns))` | Writes a logger message for traceability during notebook execution. |
| `# Missing` | Documents the purpose or boundary of the surrounding notebook step. |
| `missing = (dataframe[other_columns].isna().mean() * 100).sort_values(ascending=False).head(20)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `display(missing.to_frame("pct_missing"))` | Displays a notebook-facing result for inspection. |
| `logger.info("Top missingness (%%):\n%s", missing.to_string())` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 25 — Finalize Lineage Metadata and Save the Bronze Truth Record

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append_truth_index`
- `Bronze`
- `build_file_fingerprint`
- `build_truth_record`
- `hash`
- `identify_feature_columns`
- `identify_meta_columns`
- `info`
- `logger`
- `meta__parent_truth_hash`
- `meta__pipeline_mode`
- `meta__truth_hash`
- `process_run_id`
- `PROCESS_RUN_ID`
- `RAW_FILE_NAME`
- `RAW_FILE_PATH`
- `s`
- `save_truth_record`
- `shape`
- `sorted`

### Outputs

- `bronze_feature_columns`
- `bronze_meta_columns`
- `bronze_source_fingerprint`
- `bronze_truth`
- `BRONZE_TRUTH_HASH`
- `bronze_truth_path`
- `bronze_truth_record`
- `column_count`
- `dataframe`
- `dataset_name`
- `feature_columns`
- `layer_name`
- `meta_columns`
- `parent_truth_hash`
- `pipeline_mode`
- `row_count`
- `truth_base`
- `truth_dir`
- `truth_hash`
- `truth_index_path`

### Key Operations

- `bronze_source_fingerprint = build_file_fingerprint(RAW_FILE_PATH / RAW_FILE_NAME)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `bronze_truth = update_truth_section( bronze_truth, "source_fingerprint", bronze_source_fingerprint,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `bronze_meta_columns = sorted( set( identify_meta_columns(dataframe) + [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode", ] )`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `bronze_feature_columns = identify_feature_columns(dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `bronze_truth_record = build_truth_record( truth_base=bronze_truth, row_count=len(dataframe), column_count=dataframe.shape[1] + 3, meta_columns=bronze_meta_columns, feature_columns=`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `BRONZE_TRUTH_HASH = bronze_truth_record["truth_hash"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `dataframe = stamp_truth_columns( dataframe, truth_hash=BRONZE_TRUTH_HASH, parent_truth_hash=None, pipeline_mode=PIPELINE_MODE,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `bronze_truth_path = save_truth_record( bronze_truth_record, truth_dir=TRUTHS_PATH, dataset_name=DATASET_NAME, layer_name=LAYER_NAME,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `append_truth_index`
- `build_file_fingerprint`
- `build_truth_record`
- `identify_feature_columns`
- `identify_meta_columns`
- `info`
- `save_truth_record`
- `sorted`
- `stamp_truth_columns`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `bronze_source_fingerprint = build_file_fingerprint(RAW_FILE_PATH / RAW_FILE_NAME)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `bronze_truth = update_truth_section( bronze_truth, "source_fingerprint", bronze_source_fingerprint,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `bronze_meta_columns = sorted( set( identify_meta_columns(dataframe) + [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode", ] )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `bronze_feature_columns = identify_feature_columns(dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `bronze_truth_record = build_truth_record( truth_base=bronze_truth, row_count=len(dataframe), column_count=dataframe.shape[1] + 3, meta_columns=bronze_meta_columns, feature_columns=` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BRONZE_TRUTH_HASH = bronze_truth_record["truth_hash"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `dataframe = stamp_truth_columns( dataframe, truth_hash=BRONZE_TRUTH_HASH, parent_truth_hash=None, pipeline_mode=PIPELINE_MODE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `bronze_truth_path = save_truth_record( bronze_truth_record, truth_dir=TRUTHS_PATH, dataset_name=DATASET_NAME, layer_name=LAYER_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `append_truth_index( bronze_truth_record, truth_index_path=TRUTH_INDEX_PATH,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("Bronze truth hash: %s", BRONZE_TRUTH_HASH)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Bronze truth path: %s", bronze_truth_path)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Bronze process_run_id: %s", PROCESS_RUN_ID)` | Writes a logger message for traceability during notebook execution. |
| `print("Bronze truth hash:", BRONZE_TRUTH_HASH)` | Displays a notebook-facing result for inspection. |
| `print("Bronze truth path:", bronze_truth_path)` | Displays a notebook-facing result for inspection. |
| `print("Bronze process_run_id:", PROCESS_RUN_ID)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- Artifact or state outputs detected: truth record.

## Code Cell 26 — Define a Helper to Reorder Bronze Columns

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append`
- `bronze_columns`
- `column`
- `columns`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `extend`
- `final_order`
- `meta__`
- `startswith`

### Outputs

- `collect_meta_columns`
- `existing_columns`
- `meta_columns`
- `reorder_bronze_columns`

### Key Operations

- `def collect_meta_columns(existing_columns: List[str]) -> List[str]: meta_columns: List[str] = [] for column in existing_columns: if column.startswith("meta__"): meta_columns.append`: Defines notebook-local logic used later in the notebook.
- `def reorder_bronze_columns(dataframe: pd.DataFrame) -> pd.DataFrame: existing_columns = list(dataframe.columns) meta_columns = collect_meta_columns(existing_columns) bronze_columns`: Defines notebook-local logic used later in the notebook.

Important functions or methods detected:
- `append`
- `collect_meta_columns`
- `copy`
- `extend`
- `reorder_bronze_columns`
- `startswith`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def collect_meta_columns(existing_columns: List[str]) -> List[str]: meta_columns: List[str] = [] for column in existing_columns: if column.startswith("meta__"): meta_columns.append` | Defines notebook-local logic used later in the notebook. |
| `def reorder_bronze_columns(dataframe: pd.DataFrame) -> pd.DataFrame: existing_columns = list(dataframe.columns) meta_columns = collect_meta_columns(existing_columns) bronze_columns` | Defines notebook-local logic used later in the notebook. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 27 — Reorder the Bronze Dataframe Columns

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Bronze`
- `collect_meta_columns`
- `column_name`
- `columns`
- `d`
- `front`
- `group`
- `info`
- `logger`
- `Meta`
- `meta`
- `meta__`
- `meta_column_count`
- `moved`
- `non_meta_column_count`
- `order`
- `original`
- `preserving`
- `put`
- `Reorder`

### Outputs

- `dataframe`
- `meta_columns_after_reorder`
- `non_meta_columns_after_reorder`

### Key Operations

- `# Reorder dataframe and put meta columns in the front.`: Documents the purpose or boundary of the surrounding notebook step.
- `dataframe = reorder_bronze_columns(dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `meta_columns_after_reorder = collect_meta_columns(list(dataframe.columns))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `non_meta_columns_after_reorder = [ column_name for column_name in dataframe.columns if not column_name.startswith("meta__")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger.info( "Bronze columns reordered successfully. " "Meta columns moved to the front while preserving original within-group order. " "meta_column_count=%d \| non_meta_column_coun`: Writes a logger message for traceability during notebook execution.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `collect_meta_columns`
- `info`
- `reorder_bronze_columns`
- `startswith`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Reorder dataframe and put meta columns in the front.` | Documents the purpose or boundary of the surrounding notebook step. |
| `dataframe = reorder_bronze_columns(dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `meta_columns_after_reorder = collect_meta_columns(list(dataframe.columns))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `non_meta_columns_after_reorder = [ column_name for column_name in dataframe.columns if not column_name.startswith("meta__")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info( "Bronze columns reordered successfully. " "Meta columns moved to the front while preserving original within-group order. " "meta_column_count=%d \| non_meta_column_coun` | Writes a logger message for traceability during notebook execution. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 28 — Save the Bronze Dataset

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BRONZE_TRAIN_DATA_FILE_NAME`
- `Data`
- `data_bronze_train`
- `dataframe`
- `Parquet`
- `paths`
- `Save`
- `save_data`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `# Save Data as Parquet`: Documents the purpose or boundary of the surrounding notebook step.
- `save_data(dataframe, paths.data_bronze_train, BRONZE_TRAIN_DATA_FILE_NAME)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `save_data`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Save Data as Parquet` | Documents the purpose or boundary of the surrounding notebook step. |
| `save_data(dataframe, paths.data_bronze_train, BRONZE_TRAIN_DATA_FILE_NAME)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 29 — Create Bronze Profiling Outputs

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BRONZE_PROFILE_DIR`
- `dataframe`
- `logger`
- `metrics`
- `profile_dataframe`
- `saved`

### Outputs

- `artifacts_dir`

### Key Operations

- `metrics, saved = profile_dataframe( dataframe, logger, artifacts_dir=BRONZE_PROFILE_DIR,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `profile_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `metrics, saved = profile_dataframe( dataframe, logger, artifacts_dir=BRONZE_PROFILE_DIR,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 30 — Finalize Bronze Run Tracking

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `bronze`
- `BRONZE_TRAIN_DATA_FILE_NAME`
- `data_bronze_train`
- `EDA_Notebook_Pump_Bronze_01_Preprocessing`
- `finalize_wandb_stage`
- `ipynb`
- `logs`
- `notebooks`
- `paths`
- `Preprocessing`
- `root`
- `wandb_run`

### Outputs

- `dataframe`
- `dataset_artifact_name`
- `dataset_dirs`
- `logs_dir`
- `notebook_path`
- `profile`
- `project_root`
- `stage`

### Key Operations

- `finalize_wandb_stage( wandb_run, stage="bronze", dataframe=dataframe, project_root=paths.root, logs_dir=paths.logs, dataset_dirs=[paths.data_bronze_train], dataset_artifact_name=BR`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `finalize_wandb_stage`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `finalize_wandb_stage( wandb_run, stage="bronze", dataframe=dataframe, project_root=paths.root, logs_dir=paths.logs, dataset_dirs=[paths.data_bronze_train], dataset_artifact_name=BR` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 31 — Finish the Tracking Run

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `finish`
- `wandb_run`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `w`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `finish`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `w` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 32 — Define Validation Helpers for Bronze Checks

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__name__`
- `a`
- `Any`
- `be`
- `cannot`
- `cast`
- `def`
- `dictionary`
- `f`
- `got`
- `isinstance`
- `must`
- `name`
- `r`
- `raise`
- `type`
- `TypeError`
- `value`
- `ValueError`

### Outputs

- `require_dict`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `cast`
- `isinstance`
- `require_dict`
- `type`
- `TypeError`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 33 — Define integer validation helper

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__name__`
- `Any`
- `be`
- `cannot`
- `cast`
- `convertible`
- `def`
- `exc`
- `f`
- `got`
- `must`
- `name`
- `object`
- `r`
- `raise`
- `to`
- `type`
- `TypeError`
- `value`
- `ValueError`

### Outputs

- `require_int_value`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `cast`
- `except`
- `require_int_value`
- `type`
- `TypeError`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 34 — Validate Bronze Lineage and Truth Consistency

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `astype`
- `be`
- `Bronze`
- `BRONZE_TRUTH_HASH`
- `bronze_truth_path`
- `but`
- `check`
- `column`
- `column_count`
- `column_name`
- `columns`
- `contain`
- `count`
- `created`
- `dataframe`
- `does`
- `dropna`
- `exists`
- `extract_truth_hash`

### Outputs

- `bronze_dataframe_truth_hash`
- `bronze_parent_values`
- `bronze_truth_column_count`
- `bronze_truth_row_count`
- `dataframe_column_count`
- `dataframe_row_count`
- `loaded_bronze_truth`
- `loaded_bronze_truth_hash`
- `loaded_bronze_truth_raw`
- `missing_bronze_meta_columns`
- `parent_truth_hash`
- `required_bronze_meta_columns`

### Key Operations

- `required_bronze_meta_columns = [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_bronze_meta_columns = [ column_name for column_name in required_bronze_meta_columns if column_name not in dataframe.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if missing_bronze_meta_columns: raise ValueError( f"Bronze dataframe is missing required lineage columns: {missing_bronze_meta_columns}" )`: Controls validation, iteration, file handling, or error handling for this step.
- `bronze_dataframe_truth_hash = extract_truth_hash(dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if bronze_dataframe_truth_hash is None: raise ValueError("Bronze dataframe does not contain a readable meta__truth_hash value.")`: Controls validation, iteration, file handling, or error handling for this step.
- `if bronze_dataframe_truth_hash != BRONZE_TRUTH_HASH: raise ValueError( "Bronze dataframe truth hash does not match BRONZE_TRUTH_HASH:\n" f"dataframe={bronze_dataframe_truth_hash}\n`: Controls validation, iteration, file handling, or error handling for this step.
- `bronze_parent_values = ( dataframe["meta__parent_truth_hash"] .dropna() .astype(str) .unique() .tolist()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if bronze_parent_values: raise ValueError( "Bronze should not have a populated parent truth hash, but found values:\n" f"{bronze_parent_values}" )`: Controls validation, iteration, file handling, or error handling for this step.
- `if not Path(bronze_truth_path).exists(): raise FileNotFoundError(f"Bronze truth file was not created: {bronze_truth_path}")`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `astype`
- `dropna`
- `exists`
- `extract_truth_hash`
- `FileNotFoundError`
- `get`
- `KeyError`
- `load_json`
- `Path`
- `require_dict`
- `require_int_value`
- `strip`
- `tolist`
- `unique`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `required_bronze_meta_columns = [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_bronze_meta_columns = [ column_name for column_name in required_bronze_meta_columns if column_name not in dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_bronze_meta_columns: raise ValueError( f"Bronze dataframe is missing required lineage columns: {missing_bronze_meta_columns}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `bronze_dataframe_truth_hash = extract_truth_hash(dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if bronze_dataframe_truth_hash is None: raise ValueError("Bronze dataframe does not contain a readable meta__truth_hash value.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if bronze_dataframe_truth_hash != BRONZE_TRUTH_HASH: raise ValueError( "Bronze dataframe truth hash does not match BRONZE_TRUTH_HASH:\n" f"dataframe={bronze_dataframe_truth_hash}\n` | Controls validation, iteration, file handling, or error handling for this step. |
| `bronze_parent_values = ( dataframe["meta__parent_truth_hash"] .dropna() .astype(str) .unique() .tolist()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if bronze_parent_values: raise ValueError( "Bronze should not have a populated parent truth hash, but found values:\n" f"{bronze_parent_values}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `if not Path(bronze_truth_path).exists(): raise FileNotFoundError(f"Bronze truth file was not created: {bronze_truth_path}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `loaded_bronze_truth_raw = load_json(bronze_truth_path)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `loaded_bronze_truth = require_dict( loaded_bronze_truth_raw, "loaded_bronze_truth",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if "truth_hash" not in loaded_bronze_truth: raise KeyError("Saved Bronze truth file is missing required key: truth_hash")` | Controls validation, iteration, file handling, or error handling for this step. |
| `loaded_bronze_truth_hash = str(loaded_bronze_truth["truth_hash"]).strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if loaded_bronze_truth_hash != BRONZE_TRUTH_HASH: raise ValueError( "Saved Bronze truth file hash does not match BRONZE_TRUTH_HASH:\n" f"file={loaded_bronze_truth_hash}\n" f"record` | Controls validation, iteration, file handling, or error handling for this step. |
| `parent_truth_hash = loaded_bronze_truth.get("parent_truth_hash")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if parent_truth_hash is not None: raise ValueError( "Bronze truth file parent_truth_hash should be None.\n" f"Found: {parent_truth_hash}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `if "row_count" not in loaded_bronze_truth: raise KeyError("Saved Bronze truth file is missing required key: row_count")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if "column_count" not in loaded_bronze_truth: raise KeyError("Saved Bronze truth file is missing required key: column_count")` | Controls validation, iteration, file handling, or error handling for this step. |
| `bronze_truth_row_count = require_int_value( loaded_bronze_truth["row_count"], "loaded_bronze_truth['row_count']",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `bronze_truth_column_count = require_int_value( loaded_bronze_truth["column_count"], "loaded_bronze_truth['column_count']",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `dataframe_row_count = int(len(dataframe))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `dataframe_column_count = int(dataframe.shape[1])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if bronze_truth_row_count != dataframe_row_count: raise ValueError( "Bronze truth row_count does not match dataframe row count:\n" f"truth={bronze_truth_row_count}\n" f"dataframe={` | Controls validation, iteration, file handling, or error handling for this step. |
| `if bronze_truth_column_count != dataframe_column_count: raise ValueError( "Bronze truth column_count does not match stamped dataframe column count:\n" f"truth={bronze_truth_column_` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Bronze lineage sanity check passed.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 35 — Final Bronze Structural Check

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Basic`
- `Bronze`
- `Column`
- `columns`
- `Data`
- `dataframe`
- `describe`
- `Descriptive`
- `df`
- `dtype`
- `dtypes`
- `f`
- `head`
- `info`
- `n`
- `numeric`
- `only`
- `Overview`
- `rows`
- `Rows`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `print("=== Bronze: Data Overview ===")`: Displays a notebook-facing result for inspection.
- `print(f"Shape: {dataframe.shape[0]:,} rows × {dataframe.shape[1]} columns\n")`: Displays a notebook-facing result for inspection.
- `print("=== Column Types ===")`: Displays a notebook-facing result for inspection.
- `display(dataframe.dtypes.to_frame("dtype").head(20))`: Displays a notebook-facing result for inspection.
- `print("\n=== df.info() ===")`: Displays a notebook-facing result for inspection.
- `display(dataframe.info())`: Displays a notebook-facing result for inspection.
- `print("\n=== Basic Descriptive Statistics (numeric only) ===")`: Displays a notebook-facing result for inspection.
- `display(dataframe.describe().T)`: Displays a notebook-facing result for inspection.
- `print("\n=== Top Sample Rows ===")`: Displays a notebook-facing result for inspection.
- `display(dataframe.head())`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `describe`
- `display`
- `head`
- `info`
- `Statistics`
- `to_frame`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print("=== Bronze: Data Overview ===")` | Displays a notebook-facing result for inspection. |
| `print(f"Shape: {dataframe.shape[0]:,} rows × {dataframe.shape[1]} columns\n")` | Displays a notebook-facing result for inspection. |
| `print("=== Column Types ===")` | Displays a notebook-facing result for inspection. |
| `display(dataframe.dtypes.to_frame("dtype").head(20))` | Displays a notebook-facing result for inspection. |
| `print("\n=== df.info() ===")` | Displays a notebook-facing result for inspection. |
| `display(dataframe.info())` | Displays a notebook-facing result for inspection. |
| `print("\n=== Basic Descriptive Statistics (numeric only) ===")` | Displays a notebook-facing result for inspection. |
| `display(dataframe.describe().T)` | Displays a notebook-facing result for inspection. |
| `print("\n=== Top Sample Rows ===")` | Displays a notebook-facing result for inspection. |
| `display(dataframe.head())` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 36 — Bronze SQL Write Cell

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `else`
- `get`
- `globals`
- `Postgres`
- `skipped`
- `write`
- `write_bronze_sensor_observations_sql`

### Outputs

- `bronze_sql_summary_dataframe`
- `capstone_schema`
- `dataframe`
- `dataset_id`
- `dataset_name`
- `engine`
- `layer_schema`
- `notebook_globals`
- `run_id`
- `WRITE_TO_POSTGRES`

### Key Operations

- `WRITE_TO_POSTGRES = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if WRITE_TO_POSTGRES: bronze_sql_summary_dataframe = write_bronze_sensor_observations_sql( engine=engine, capstone_schema=CAPSTONE_SCHEMA, layer_schema=LAYER_SCHEMA, dataset_id=DAT`: Displays a notebook-facing result for inspection.
- `else: print("Postgres write skipped.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `get`
- `globals`
- `write_bronze_sensor_observations_sql`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `WRITE_TO_POSTGRES = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if WRITE_TO_POSTGRES: bronze_sql_summary_dataframe = write_bronze_sensor_observations_sql( engine=engine, capstone_schema=CAPSTONE_SCHEMA, layer_schema=LAYER_SCHEMA, dataset_id=DAT` | Displays a notebook-facing result for inspection. |
| `else: print("Postgres write skipped.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 37 — Preview the Bronze SQL Output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `engine`
- `information_schema`
- `LAYER_SCHEMA`
- `read_sql_dataframe`
- `schema`
- `SELECT`
- `sensor_observations`
- `table_name`
- `table_schema`
- `tables`
- `WHERE`

### Outputs

- `params`

### Key Operations

- `read_sql_dataframe( engine, """ SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema = :schema AND table_name = 'sensor_observations' """, params={"sch`: Loads input data, configuration, or artifacts required by the current stage.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `read_sql_dataframe( engine, """ SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema = :schema AND table_name = 'sensor_observations' """, params={"sch` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 38 — Preview the SQL-facing layer output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `LAYER_SCHEMA`
- `preview_sql_table`
- `sensor_observations`

### Outputs

- `limit`
- `schema`
- `table`

### Key Operations

- `preview_sql_table( schema=LAYER_SCHEMA, table="sensor_observations", limit=5,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `preview_sql_table`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `preview_sql_table( schema=LAYER_SCHEMA, table="sensor_observations", limit=5,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 39 — Close the database engine cleanly

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

