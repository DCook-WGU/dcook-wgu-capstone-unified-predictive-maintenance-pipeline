# Notebook Code Reference: EDA_Notebook_Pump_Silver_01_PreEDA

Notebook path:

`notebooks/eda/EDA_Notebook_Pump_Silver_01_PreEDA.ipynb`

## Notebook Purpose

This notebook performs Silver Pre-EDA checks and prepares cleaned metadata and profiling inputs from the Bronze output.

Notebook stage:

`Silver`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01 |
| Helper Functions | Code Cell 02 |
| Environment Setup, Paths, and Runtime Configuration | Code Cell 03, Code Cell 04 |
| Context Santity Check | Code Cell 05, Code Cell 06 |
| Defer Silver artifact folder creation | Code Cell 07 |
| SQL Runtime Context | Code Cell 08 |
| SQL Smoke Check | Code Cell 09 |
| Logging Setup | Code Cell 10, Code Cell 11 |
| Initialize Experiment Tracking | Code Cell 12 |
| Initialize the Silver Ledger | Code Cell 13, Code Cell 14 |
| Load the Bronze Dataset into the Silver Workflow | Code Cell 15 |
| Resolve the Parent Truth Record and Confirm Dataset Identity | Code Cell 16 |
| Create the Silver Pre-EDA Artifact Folders After Dataset Resolution | Code Cell 17 |
| Initial Silver Input Review | Code Cell 18 |
| Remove Junk Import Columns | Code Cell 19 |
| Remove import-generated junk columns | Code Cell 20 |
| Remove Duplicate Column Names | Code Cell 21 |
| Remove duplicate columns before canonicalization | Code Cell 22 |
| Validate the Dataset Name for Silver | Code Cell 23 |
| Validate dataset identity for the Silver layer | Code Cell 24 |
| Add and Confirm Required Silver Metadata Columns | Code Cell 25 |
| Resolve the Label or Status Source | Code Cell 26 |
| Review current column layout | Code Cell 27, Code Cell 35, Code Cell 40, Code Cell 46, Code Cell 74 |
| Resolve the label source used for anomaly flags | Code Cell 28 |
| Answer | Code Cell 29 |
| Preview current dataframe rows | Code Cell 30, Code Cell 82 |
| Protect Canonical Output Column Names | Code Cell 31 |
| Protect canonical metadata column names | Code Cell 32 |
| Build the Canonical Identity and Ordering Fields | Code Cell 33 |
| Build canonical row identity and ordering fields | Code Cell 34 |
| Build the Binary Anomaly Flag | Code Cell 36 |
| Define label-to-binary normalization | Code Cell 37, Code Cell 39 |
| Define status-to-anomaly conversion | Code Cell 38 |
| Build Episode IDs from the Anomaly Signal | Code Cell 41 |
| Build anomaly episode identifiers | Code Cell 42 |
| Identify the Candidate Feature Set | Code Cell 43 |
| Classify column types for feature selection | Code Cell 44, Code Cell 48 |
| Define prefix-based feature exclusions | Code Cell 45 |
| Define identifier-column exclusion heuristic | Code Cell 47 |
| Select model-ready candidate features | Code Cell 49 |
| Identify Columns That May Need One-Hot Encoding Later | Code Cell 50 |
| Apply feature exclusion rules | Code Cell 51, Code Cell 70 |
| Mid-Workflow Structural Review | Code Cell 52, Code Cell 55, Code Cell 57 |
| Review dataframe structure and dtypes | Code Cell 53, Code Cell 56, Code Cell 58, Code Cell 61 |
| Review intermediate output | Code Cell 54, Code Cell 62, Code Cell 63, Code Cell 64, Code Cell 67 |
| Evaluate Missingness and Quarantine High-Missing Features | Code Cell 59 |
| Quarantine features with excessive missingness | Code Cell 60 |
| Summarize global missingness after quarantine | Code Cell 65 |
| Define configuration mapping guards | Code Cell 66 |
| Finalize the Feature Lists After Quarantine | Code Cell 68 |
| Create a Stable Feature Set Identifier | Code Cell 69 |
| Reorder the Silver Columns into a Cleaner Final Layout | Code Cell 71 |
| Define metadata column ordering helper | Code Cell 72, Code Cell 73 |
| Define final Silver column ordering | Code Cell 75, Code Cell 76 |
| Run Final Quick Quality Checks | Code Cell 77 |
| Run final Silver quality checks | Code Cell 78 |
| Build the Silver Feature Registry | Code Cell 79, Code Cell 80, Code Cell 81 |
| Build the Silver Truth Record and Save the Final Outputs | Code Cell 83 |
| Assign a reproducible feature set identifier | Code Cell 84 |
| Save the Ledger Artifact | Code Cell 85 |
| Finalize Experiment Tracking | Code Cell 86 |
| Run a Final Lineage and Consistency Check | Code Cell 87 |
| Define integer validation helper | Code Cell 88 |
| Silver SQL Write Cell | Code Cell 89 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__future__`
- `annotations`
- `Any`
- `append_truth_index`
- `Artifact`
- `artifact_file_path`
- `artifacts`
- `Biases`
- `build_artifact_dirs`
- `build_artifact_dirs_from_config`
- `build_file_fingerprint`
- `build_truth_config_block`
- `build_truth_record`
- `cast`
- `Charts`
- `classification_report`
- `cluster`
- `columns`
- `Config`
- `config_loader`

### Outputs

- `DropKeep`

### Key Operations

- `from __future__ import annotations`: Imports a dependency or project helper used by later cells.
- `from dataclasses import dataclass, field`: Imports a dependency or project helper used by later cells.
- `from datetime import datetime, timezone`: Imports a dependency or project helper used by later cells.
- `from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Pattern, Literal, Mapping, cast`: Imports a dependency or project helper used by later cells.
- `from pathlib import Path`: Imports a dependency or project helper used by later cells.
- `import yaml`: Imports a dependency or project helper used by later cells.
- `import re`: Imports a dependency or project helper used by later cells.
- `import os`: Imports a dependency or project helper used by later cells.
- `import logging`: Imports a dependency or project helper used by later cells.
- `import wandb`: Imports a dependency or project helper used by later cells.
- `# Pands and Numpy`: Documents the purpose or boundary of the surrounding notebook step.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.

Important functions or methods detected:
- `import`
- `set_option`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `from __future__ import annotations` | Imports a dependency or project helper used by later cells. |
| `from dataclasses import dataclass, field` | Imports a dependency or project helper used by later cells. |
| `from datetime import datetime, timezone` | Imports a dependency or project helper used by later cells. |
| `from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Pattern, Literal, Mapping, cast` | Imports a dependency or project helper used by later cells. |
| `from pathlib import Path` | Imports a dependency or project helper used by later cells. |
| `import yaml` | Imports a dependency or project helper used by later cells. |
| `import re` | Imports a dependency or project helper used by later cells. |
| `import os` | Imports a dependency or project helper used by later cells. |
| `import logging` | Imports a dependency or project helper used by later cells. |
| `import wandb` | Imports a dependency or project helper used by later cells. |
| `# Pands and Numpy` | Documents the purpose or boundary of the surrounding notebook step. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `import numpy as np` | Imports a dependency or project helper used by later cells. |
| `# Plotting/Charts` | Documents the purpose or boundary of the surrounding notebook step. |
| `import matplotlib.pyplot as plt` | Imports a dependency or project helper used by later cells. |
| `import seaborn as sns` | Imports a dependency or project helper used by later cells. |
| `# SKLearn Modules` | Documents the purpose or boundary of the surrounding notebook step. |
| `from sklearn.model_selection import train_test_split, KFold` | Imports a dependency or project helper used by later cells. |
| `from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder` | Imports a dependency or project helper used by later cells. |
| `from sklearn.decomposition import PCA` | Imports a dependency or project helper used by later cells. |
| `from sklearn.cluster import KMeans` | Imports a dependency or project helper used by later cells. |
| `from sklearn.ensemble import RandomForestClassifier, IsolationForest` | Imports a dependency or project helper used by later cells. |
| `from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score` | Imports a dependency or project helper used by later cells. |
| `import pyarrow.parquet as pq` | Imports a dependency or project helper used by later cells. |
| `import pyarrow as pa` | Imports a dependency or project helper used by later cells. |
| `import hashlib` | Imports a dependency or project helper used by later cells. |
| `# Custom Utilities Module` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Path Module` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.core.paths import get_paths` | Imports a dependency or project helper used by later cells. |
| `# File IO Helper` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.core.file_io import ( load_data, save_data, save_json, load_json` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Logging Profiler` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.core.logging_profiler import profile_dataframe` | Imports a dependency or project helper used by later cells. |
| `# Logging Module` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.core.logging_setup import ( configure_logging, log_layer_paths,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Weight and Biases` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.core.wandb_utils import finalize_wandb_stage` | Imports a dependency or project helper used by later cells. |
| `# Truth Store` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.core.truths import ( make_process_run_id, build_file_fingerprint, extract_truth_hash, identify_meta_columns, identify_feature_columns, initialize_layer_truth, update_tru` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Database Helpers` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Postgre Layer Helpers` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.database.layer_postgres import ( read_layer_dataframe, write_layer_dataframe, prepare_layer_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# SQL Helpers` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.database.sql_notebook_helpers import ( delete_dataset_run_rows, execute_many, get_existing_dataframe, get_row_value, log_data_quality_event, log_pipeline_artifact, previ` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Layer Specific SQL Helper` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.database.medallion_sql_writers import ( log_gold_05_anomaly_detection_summary_sql, log_silver_eda_sql, write_bronze_sensor_observations_sql, write_gold_baseline_scores_s` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Config Loader` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.core.config_loader import ( load_pipeline_config, build_truth_config_block, set_wandb_dir_from_config, export_config_snapshot,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Artifact Directory Creation` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.core.artifacts import ( build_artifact_dirs, build_artifact_dirs_from_config, artifact_file_path,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `10 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: SQL or medallion table write, truth record.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Helper Functions

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

## Code Cell 03 — Environment Setup, Paths, and Runtime Configuration

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `aliases`
- `capstone`
- `context`
- `context_loaded`
- `dataset_config`
- `default`
- `default_fallbacks`
- `execution`
- `info`
- `load_notebook_context`
- `Loaded`
- `loaded`
- `log`
- `log_filename`
- `log_path`
- `LOG_PATH`
- `logger`
- `logger_child_name`
- `message`

### Outputs

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
- `DEFAULT_FALLBACKS_CFG`
- `DEFAULTS_FALLBACKS_CFG`
- `EXECUTION_CFG`
- `extra`
- `FILENAMES`
- `kind`
- `ledger`

### Key Operations

- `# Shared notebook context`: Documents the purpose or boundary of the surrounding notebook step.
- `CONTEXT_STAGE = "silver_preeda"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_DATASET = "pump"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LAYER = "silver"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_RUN_MODE = "train"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_PROFILE = "default"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LOG_FILE = "silver.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.silver.preeda", log_filename=CO`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `CONTEXT_STAGE = "silver_preeda"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LAYER = "silver"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_RUN_MODE = "train"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LOG_FILE = "silver.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.silver.preeda", log_filename=CO` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Shared aliases used throughout the notebook` | Documents the purpose or boundary of the surrounding notebook step. |
| `paths = CTX.paths` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_MAP = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE_CFG = CTX.stage_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_CFG = CTX.stage_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
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
| `DEFAULTS_FALLBACKS_CFG = CTX.default_fallbacks` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DEFAULT_FALLBACKS_CFG = CTX.default_fallbacks` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info( "Notebook context loaded", extra={ "stage": CONTEXT_STAGE, "recipe_id": CONTEXT_RECIPE_ID, "dataset": CONTEXT_DATASET, "mode": CONFIG_RUN_MODE, "profile": CONFIG_PROFI` | Writes a logger message for traceability during notebook execution. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="context_loaded", message="Loaded shared notebook context.", data={ "stage": CONTEXT_STAGE, "recipe_id": CONTEXT_RECIPE_ID, "dataset": CONTEXT_DATASET` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 04 — Environment Setup, Paths, and Runtime Configuration

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `after`
- `Any`
- `are`
- `Artifact`
- `Artifacts`
- `artifacts`
- `asset_id`
- `B`
- `Base`
- `Bronze`
- `bronze_train_file_name`
- `build_truth_config_block`
- `Candidate`
- `Canonical`
- `capstone`
- `cast`
- `compile`
- `CONFIG`
- `CONFIG_RUN_MODE`
- `created`

### Outputs

- `ASSET_ID_DEFAULT_FALLBACK`
- `BASELINE_DAYS`
- `BASELINE_GAP_HOURS`
- `BRONZE_TRAIN_DATA_FILE_NAME`
- `BRONZE_TRAIN_DATA_PATH`
- `CANONICAL_EXCLUDE_COLUMNS`
- `CANONICAL_NON_META_ORDER`
- `CANONICAL_OUTPUT_COLUMNS`
- `CLEANING_RECIPE_ID`
- `CORRELATION_THRESHOLD`
- `DATASET_NAME`
- `DATASET_NAME_CONFIG`
- `DATASET_RUN_ID`
- `DROPPED_SENSORS_FILE_NAME`
- `FEATURE_REGISTRY_FILE_NAME`
- `flags`
- `JUNK_COLUMN_CANDIDATES`
- `LABEL_COLUMN_CANDIDATES`
- `LABEL_COLUMNS_ORDER`
- `LABEL_EXCLUDE_COLUMNS`

### Key Operations

- `TRUTH_CONFIG = cast(Dict[str, Any], build_truth_config_block(CONFIG))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_CONFIG["pipeline"] = PIPELINE`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `# ---- Stage details ----`: Documents the purpose or boundary of the surrounding notebook step.
- `STAGE = "silver"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `LAYER_NAME = str(SILVER_CFG["layer_name"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CLEANING_RECIPE_ID = str(SILVER_CFG["cleaning_recipe_id"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_VERSION = str(VERSIONS_CFG["silver"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_VERSION = str(VERSIONS_CFG["truth"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `PIPELINE_MODE = str(PIPELINE["execution_mode"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RUN_MODE = str(RUNTIME_CFG.get("mode", CONFIG_RUN_MODE))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `build_truth_config_block`
- `cast`
- `compile`
- `get`
- `getenv`
- `make_process_run_id`
- `mkdir`
- `Path`
- `set_wandb_dir_from_config`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `TRUTH_CONFIG = cast(Dict[str, Any], build_truth_config_block(CONFIG))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_CONFIG["pipeline"] = PIPELINE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Stage details ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `STAGE = "silver"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LAYER_NAME = str(SILVER_CFG["layer_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CLEANING_RECIPE_ID = str(SILVER_CFG["cleaning_recipe_id"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_VERSION = str(VERSIONS_CFG["silver"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_VERSION = str(VERSIONS_CFG["truth"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `PIPELINE_MODE = str(PIPELINE["execution_mode"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_MODE = str(RUNTIME_CFG.get("mode", CONFIG_RUN_MODE))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `DATASET_NAME_CONFIG = str(DATASET_CFG.get("name", "pump"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_NAME = DATASET_NAME_CONFIG` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_PROCESS_RUN_ID = make_process_run_id( str(SILVER_CFG["process_run_id_prefix"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- W&B ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `WANDB_PROJECT = str(WANDB_CFG.get("project", "capstone"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WANDB_ENTITY = str(WANDB_CFG.get("entity", ""))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WANDB_RUN_NAME = f"{SILVER_VERSION}"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Canonical outputs ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `CANONICAL_OUTPUT_COLUMNS = list(SILVER_CFG["canonical_output_columns"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CANONICAL_NON_META_ORDER = list(SILVER_CFG["canonical_non_meta_order"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `META_REQUIRED_COLUMNS = list(SILVER_CFG["meta_required_columns"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CANONICAL_EXCLUDE_COLUMNS = list(SILVER_CFG["canonical_exclude_columns"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LABEL_EXCLUDE_COLUMNS = list(SILVER_CFG["label_exclude_columns"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LABEL_COLUMNS_ORDER = list(SILVER_CFG["label_columns_order"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Defaults / fallbacks ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Defaults / fallbacks ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `ASSET_ID_DEFAULT_FALLBACK = str( os.getenv( "SYNTHETIC_ASSET_ID", DATASET_CFG.get( "asset_id", DEFAULTS_FALLBACKS_CFG.get("asset_id", "pump_asset_001"), ), )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `RUN_ID_DEFAULT_FALLBACK = str( os.getenv( "SYNTHETIC_RUN_ID", DATASET_CFG.get( "run_id", DEFAULTS_FALLBACKS_CFG.get("run_id", "synthetic_run_001"), ), )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DATASET_RUN_ID = str(DATASET_CFG.get("run_id", RUN_ID_DEFAULT_FALLBACK))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RAW_PREFIX = str(SILVER_CFG["raw_prefix"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Candidate lists ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `TIME_COLUMN_CANDIDATES = list(SILVER_CFG["time_column_candidates"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STEP_COLUMN_CANDIDATES = list(SILVER_CFG["step_column_candidates"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TIE_BREAKER_CANDIDATES = list(SILVER_CFG["tie_breaker_candidates"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STATUS_COLUMN_CANDIDATES = list(SILVER_CFG["status_column_candidates"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LABEL_COLUMN_CANDIDATES = list(SILVER_CFG["label_column_candidates"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `NORMAL_STATUS_VALUES = { str(value) for value in DATASET_CFG.get("normal_status_values", ["normal"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `RegexLike = Union[str, Pattern[str]]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `UNNAMED_COLUMN_REGEX = re.compile( r"^unnamed:\s*\d+(\.\d+)?$", flags=re.IGNORECASE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `JUNK_COLUMN_CANDIDATES = [ "Unnamed:", "Unnamed", "unnamed", "...", "level_0", "", " ", "\t", "\ufeff", "\ufeff<name>",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- QA / EDA thresholds ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `MIN_TIME_PARSE_SUCCESS_PERCENT = float(SILVER_CFG["min_time_parse_success_percent"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MIN_STEP_PARSE_SUCCESS_PERCENT = float(SILVER_CFG["min_step_parse_success_percent"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MISSING_STATE_DIFF_GATE_PCT = float(SILVER_CFG["minimum_state_difference_gate_percentage"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MIN_ROWS_PER_STATE_FOR_GATE = int(SILVER_CFG["minimum_rows_per_state_for_gate"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `41 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 05 — Context Santity Check

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

## Code Cell 06 — Context Santity Check

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `check`
- `context`
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
- `Silver`
- `SILVER_CFG`
- `variables`

### Outputs

- `missing_silver_context_vars`
- `silver_required_context_vars`

### Key Operations

- `silver_required_context_vars = [ "SILVER_CFG",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_silver_context_vars = [ name for name in silver_required_context_vars if name not in globals()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if missing_silver_context_vars: raise NameError(f"Missing Silver context variables: {missing_silver_context_vars}")`: Controls validation, iteration, file handling, or error handling for this step.
- `logger.info("Silver context sanity check passed")`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `globals`
- `info`
- `NameError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `silver_required_context_vars = [ "SILVER_CFG",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_silver_context_vars = [ name for name in silver_required_context_vars if name not in globals()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_silver_context_vars: raise NameError(f"Missing Silver context variables: {missing_silver_context_vars}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `logger.info("Silver context sanity check passed")` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 07 — Defer Silver artifact folder creation

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `artifact`
- `creation`
- `DATASET_NAME`
- `deferred`
- `folder`
- `resolved`
- `Silver`
- `until`

### Outputs

- `CONFIG_SNAPSHOT_PATH`
- `DROPPED_SENSORS_DATA_PATH`
- `FEATURE_REGISTRY_PATH`
- `SILVER_ARTIFACTS_PATH`
- `SILVER_CONFIG_DIR`
- `SILVER_LINEAGE_DIR`
- `SILVER_METADATA_DIR`
- `SILVER_PREEDA_ARTIFACT_DIRS`
- `SILVER_PROFILE_DIR`
- `SILVER_QUALITY_DIR`
- `SILVER_REGISTRY_DIR`
- `SILVER_SUMMARY_DIR`

### Key Operations

- `SILVER_PREEDA_ARTIFACT_DIRS = {}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_ARTIFACTS_PATH = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_REGISTRY_DIR = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_PROFILE_DIR = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_QUALITY_DIR = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_SUMMARY_DIR = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_METADATA_DIR = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_CONFIG_DIR = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_LINEAGE_DIR = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_SNAPSHOT_PATH = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DROPPED_SENSORS_DATA_PATH = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `FEATURE_REGISTRY_PATH = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `SILVER_PREEDA_ARTIFACT_DIRS = {}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_ARTIFACTS_PATH = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_REGISTRY_DIR = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_PROFILE_DIR = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_QUALITY_DIR = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_SUMMARY_DIR = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_METADATA_DIR = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_CONFIG_DIR = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_LINEAGE_DIR = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_SNAPSHOT_PATH = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DROPPED_SENSORS_DATA_PATH = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FEATURE_REGISTRY_PATH = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Silver artifact folder creation deferred until DATASET_NAME is resolved.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 08 — SQL Runtime Context

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
- `by`
- `candidates`
- `capstone`
- `CAPSTONE_ASSET_ID`
- `CAPSTONE_DATASET_ID`
- `CAPSTONE_RUN_ID`
- `CAPSTONE_SCHEMA`
- `cast`
- `CONFIG`

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

- `# =============================================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# SQL Runtime Context`: Documents the purpose or boundary of the surrounding notebook step.
- `# Purpose:`: Documents the purpose or boundary of the surrounding notebook step.
- `# Establish the Postgres connection and resolve the dataset/run identifiers`: Documents the purpose or boundary of the surrounding notebook step.
- `# used by SQL logging and medallion table writes.`: Documents the purpose or boundary of the surrounding notebook step.
- `# =============================================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `engine = get_engine_from_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CAPSTONE_SCHEMA: str = str(os.getenv("CAPSTONE_SCHEMA", "capstone"))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def first_non_empty_string(*values: object) -> str \| None: """ Return the first non-empty string-like value from a list of candidates. This helper skips None, empty strings, whites`: Defines notebook-local logic used later in the notebook.
- `dataset_config = ( cast(Dict[str, Any], CONFIG.get("dataset", {})) if isinstance(CONFIG.get("dataset", {}), dict) else {}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `dataset_config_name = dataset_config.get("name")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

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
| `# =============================================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# SQL Runtime Context` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Purpose:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Establish the Postgres connection and resolve the dataset/run identifiers` | Documents the purpose or boundary of the surrounding notebook step. |
| `# used by SQL logging and medallion table writes.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =============================================================================` | Documents the purpose or boundary of the surrounding notebook step. |
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

## Code Cell 10 — Logging Setup

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

## Code Cell 11 — Logging Setup

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `capstone`
- `configure_logging`
- `Create`
- `current_layer`
- `DEBUG`
- `file`
- `getLogger`
- `info`
- `Initial`
- `Initiate`
- `initiation`
- `load`
- `loads`
- `Log`
- `log`
- `log_layer_paths`
- `Logging`
- `logging`
- `logs`
- `Original`

### Outputs

- `level`
- `logger`
- `overwrite_handlers`
- `silver_log_path`

### Key Operations

- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Original Logging Setup`: Documents the purpose or boundary of the surrounding notebook step.
- `# Create silver log path`: Documents the purpose or boundary of the surrounding notebook step.
- `silver_log_path = paths.logs / "silver.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Initial Logger`: Documents the purpose or boundary of the surrounding notebook step.
- `configure_logging( "capstone", silver_log_path, level=logging.DEBUG, overwrite_handlers=True,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Initiate Logger and log file`: Documents the purpose or boundary of the surrounding notebook step.
- `logger = logging.getLogger("capstone.silver")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Log load and initiation`: Documents the purpose or boundary of the surrounding notebook step.
- `logger.info("Silver stage starting")`: Writes a logger message for traceability during notebook execution.
- `# Log paths loads`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `configure_logging`
- `getLogger`
- `info`
- `log_layer_paths`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Original Logging Setup` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Create silver log path` | Documents the purpose or boundary of the surrounding notebook step. |
| `silver_log_path = paths.logs / "silver.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Initial Logger` | Documents the purpose or boundary of the surrounding notebook step. |
| `configure_logging( "capstone", silver_log_path, level=logging.DEBUG, overwrite_handlers=True,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Initiate Logger and log file` | Documents the purpose or boundary of the surrounding notebook step. |
| `logger = logging.getLogger("capstone.silver")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Log load and initiation` | Documents the purpose or boundary of the surrounding notebook step. |
| `logger.info("Silver stage starting")` | Writes a logger message for traceability during notebook execution. |
| `# Log paths loads` | Documents the purpose or boundary of the surrounding notebook step. |
| `log_layer_paths(paths, current_layer="silver", logger=logger)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 12 — Initialize Experiment Tracking

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `B`
- `bronze_path`
- `BRONZE_TRAIN_DATA_FILE_NAME`
- `BRONZE_TRAIN_DATA_PATH`
- `CLEANING_RECIPE_ID`
- `cleaning_recipe_id`
- `info`
- `init`
- `initialized`
- `logger`
- `min_time_parse_success_percent`
- `MIN_TIME_PARSE_SUCCESS_PERCENT`
- `QUARANTINE_MISSING_PCT`
- `quarantine_missing_pct`
- `ROLLING_MINUTES`
- `rolling_window`
- `s`
- `silver`
- `silver_out_dir`
- `SILVER_TRAIN_DATA_PATH`

### Outputs

- `config`
- `entity`
- `job_type`
- `name`
- `project`
- `wandb_run`

### Key Operations

- `# W&B`: Documents the purpose or boundary of the surrounding notebook step.
- `wandb_run = wandb.init( project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="silver", config={ "silver_version": SILVER_VERSION, "cleaning_recipe_id": CLEANIN`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger.info("W&B initialized: %s", wandb_run.name)`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `info`
- `init`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# W&B` | Documents the purpose or boundary of the surrounding notebook step. |
| `wandb_run = wandb.init( project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="silver", config={ "silver_version": SILVER_VERSION, "cleaning_recipe_id": CLEANIN` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("W&B initialized: %s", wandb_run.name)` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- Artifact or state outputs detected: optional experiment tracking call.

## Code Cell 13 — Initialize the Silver Ledger

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `add`
- `available`
- `by`
- `cell`
- `checkpoint`
- `context`
- `CONTEXT_RECIPE_ID`
- `CONTEXT_STAGE`
- `creating`
- `info`
- `initialized`
- `instead`
- `Keep`
- `ledger`
- `Ledger`
- `ledger_context_ready`
- `load_notebook_context`
- `LOG_PATH`
- `log_path`

### Outputs

- `data`
- `extra`
- `kind`
- `logger`
- `message`
- `step`

### Key Operations

- `# Ledger was initialized by load_notebook_context().`: Documents the purpose or boundary of the surrounding notebook step.
- `# Keep this cell as a visible notebook checkpoint instead of re-creating Ledger.`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger.add( kind="step", step="ledger_context_ready", message="Ledger is available from shared notebook context.", data={ "stage": CONTEXT_STAGE, "recipe_id": CONTEXT_RECIPE_ID, "l`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger.info( "Ledger ready from notebook context", extra={"stage": CONTEXT_STAGE, "recipe_id": CONTEXT_RECIPE_ID},`: Writes a logger message for traceability during notebook execution.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `info`
- `load_notebook_context`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Ledger was initialized by load_notebook_context().` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Keep this cell as a visible notebook checkpoint instead of re-creating Ledger.` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger.add( kind="step", step="ledger_context_ready", message="Ledger is available from shared notebook context.", data={ "stage": CONTEXT_STAGE, "recipe_id": CONTEXT_RECIPE_ID, "l` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info( "Ledger ready from notebook context", extra={"stage": CONTEXT_STAGE, "recipe_id": CONTEXT_RECIPE_ID},` | Writes a logger message for traceability during notebook execution. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 14 — Initialize the Silver Ledger

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `CLEANING_RECIPE_ID`
- `init`
- `Initialized`
- `Original`
- `recipe_id`
- `Setup`
- `STAGE`
- `stage`

### Outputs

- `kind`
- `ledger`
- `logger`
- `message`
- `step`

### Key Operations

- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Original Ledger Setup`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger = Ledger(stage=STAGE, recipe_id=CLEANING_RECIPE_ID)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ledger.add( kind="step", step="init", message="Initialized ledger", logger=logger`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `Ledger`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Original Ledger Setup` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger = Ledger(stage=STAGE, recipe_id=CLEANING_RECIPE_ID)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger.add( kind="step", step="init", message="Initialized ledger", logger=logger` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 15 — Load the Bronze Dataset into the Silver Workflow

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `All`
- `Artifact`
- `back`
- `be`
- `Bronze`
- `bronze_cols`
- `bronze_path`
- `bronze_rows`
- `BRONZE_TRAIN_DATA_FILE_NAME`
- `BRONZE_TRAIN_DATA_PATH`
- `cols`
- `columns`
- `derived`
- `else`
- `exists`
- `f`
- `file`
- `FileNotFoundError`
- `files`

### Outputs

- `bronze_data_path`
- `consequence`
- `data`
- `dataframe`
- `kind`
- `logger`
- `message`
- `parquet_files`
- `preferred_bronze`
- `step`
- `why`

### Key Operations

- `# Load Data`: Documents the purpose or boundary of the surrounding notebook step.
- `preferred_bronze = BRONZE_TRAIN_DATA_PATH / BRONZE_TRAIN_DATA_FILE_NAME`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if preferred_bronze.exists(): bronze_data_path = preferred_bronze`: Controls validation, iteration, file handling, or error handling for this step.
- `else: parquet_files = sorted(BRONZE_TRAIN_DATA_PATH.glob("*.parquet")) if len(parquet_files) == 0: raise FileNotFoundError(f"No parquet files found in {BRONZE_TRAIN_DATA_PATH}") if`: Writes a logger message for traceability during notebook execution.
- `if not bronze_data_path.exists(): raise FileNotFoundError(f"Bronze parquet not found: {bronze_data_path}")`: Controls validation, iteration, file handling, or error handling for this step.
- `dataframe = load_data(bronze_data_path.parent, bronze_data_path.name)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `logger.info("Loaded Bronze: %s \| shape=%s", bronze_data_path, dataframe.shape)`: Writes a logger message for traceability during notebook execution.
- `wandb_run.log({"bronze_rows": int(dataframe.shape[0]), "bronze_cols": int(dataframe.shape[1])})`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="load_bronze", message="Loaded Bronze Parquet", why="Silver must be derived from reprodicible Bronze Artifact", consequence="All silver outputs trace `: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `add`
- `display`
- `exists`
- `FileNotFoundError`
- `glob`
- `head`
- `info`
- `load_data`
- `log`
- `sorted`
- `warning`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Load Data` | Documents the purpose or boundary of the surrounding notebook step. |
| `preferred_bronze = BRONZE_TRAIN_DATA_PATH / BRONZE_TRAIN_DATA_FILE_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if preferred_bronze.exists(): bronze_data_path = preferred_bronze` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: parquet_files = sorted(BRONZE_TRAIN_DATA_PATH.glob("*.parquet")) if len(parquet_files) == 0: raise FileNotFoundError(f"No parquet files found in {BRONZE_TRAIN_DATA_PATH}") if` | Writes a logger message for traceability during notebook execution. |
| `if not bronze_data_path.exists(): raise FileNotFoundError(f"Bronze parquet not found: {bronze_data_path}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `dataframe = load_data(bronze_data_path.parent, bronze_data_path.name)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `logger.info("Loaded Bronze: %s \| shape=%s", bronze_data_path, dataframe.shape)` | Writes a logger message for traceability during notebook execution. |
| `wandb_run.log({"bronze_rows": int(dataframe.shape[0]), "bronze_cols": int(dataframe.shape[1])})` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="load_bronze", message="Loaded Bronze Parquet", why="Silver must be derived from reprodicible Bronze Artifact", consequence="All silver outputs trace ` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `display(dataframe.head(3))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 16 — Resolve the Parent Truth Record and Confirm Dataset Identity

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `astype`
- `Bronze`
- `bronze`
- `CLEANING_RECIPE_ID`
- `cleaning_recipe_id`
- `columns`
- `config_snapshot`
- `contain`
- `dataset`
- `dataset_name_config`
- `DATASET_NAME_CONFIG`
- `dataset_name_from_parent_truth`
- `dataset_name_parent_truth`
- `does`
- `dropna`
- `else`
- `extract_truth_hash`
- `fillna`
- `get_dataset_name_from_truth`

### Outputs

- `BRONZE_DATASET_NAME`
- `column_name`
- `dataframe`
- `dataset_name`
- `DATASET_NAME`
- `layer_name`
- `parent_layer_name`
- `PARENT_PIPELINE_MODE`
- `parent_truth`
- `parent_truth_hash`
- `pipeline_mode`
- `PIPELINE_MODE`
- `process_run_id`
- `SILVER_PARENT_TRUTH_HASH`
- `silver_truth`
- `truth_dir`
- `truth_version`

### Key Operations

- `SILVER_PARENT_TRUTH_HASH = extract_truth_hash(dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if SILVER_PARENT_TRUTH_HASH is None: raise ValueError("Silver input dataframe does not contain a readable meta__truth_hash value.")`: Controls validation, iteration, file handling, or error handling for this step.
- `BRONZE_DATASET_NAME = ( dataframe["meta__dataset"] .dropna() .astype("string") .str.strip()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `BRONZE_DATASET_NAME = BRONZE_DATASET_NAME[BRONZE_DATASET_NAME != ""]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if len(BRONZE_DATASET_NAME) == 0: raise ValueError("Silver input dataframe is missing usable meta__dataset values.")`: Controls validation, iteration, file handling, or error handling for this step.
- `BRONZE_DATASET_NAME = str(BRONZE_DATASET_NAME.iloc[0]).strip()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `parent_truth = load_parent_truth_record_from_dataframe( dataframe=dataframe, truth_dir=TRUTHS_PATH, parent_layer_name="bronze", dataset_name=BRONZE_DATASET_NAME, column_name="meta_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `DATASET_NAME = get_dataset_name_from_truth(parent_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_PARENT_TRUTH_HASH = get_truth_hash(parent_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PARENT_PIPELINE_MODE = get_pipeline_mode_from_truth(parent_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `astype`
- `dropna`
- `extract_truth_hash`
- `fillna`
- `get_dataset_name_from_truth`
- `get_pipeline_mode_from_truth`
- `get_truth_hash`
- `info`
- `initialize_layer_truth`
- `load_parent_truth_record_from_dataframe`
- `strip`
- `update_truth_section`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `SILVER_PARENT_TRUTH_HASH = extract_truth_hash(dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if SILVER_PARENT_TRUTH_HASH is None: raise ValueError("Silver input dataframe does not contain a readable meta__truth_hash value.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `BRONZE_DATASET_NAME = ( dataframe["meta__dataset"] .dropna() .astype("string") .str.strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BRONZE_DATASET_NAME = BRONZE_DATASET_NAME[BRONZE_DATASET_NAME != ""]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(BRONZE_DATASET_NAME) == 0: raise ValueError("Silver input dataframe is missing usable meta__dataset values.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `BRONZE_DATASET_NAME = str(BRONZE_DATASET_NAME.iloc[0]).strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `parent_truth = load_parent_truth_record_from_dataframe( dataframe=dataframe, truth_dir=TRUTHS_PATH, parent_layer_name="bronze", dataset_name=BRONZE_DATASET_NAME, column_name="meta_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DATASET_NAME = get_dataset_name_from_truth(parent_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_PARENT_TRUTH_HASH = get_truth_hash(parent_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PARENT_PIPELINE_MODE = get_pipeline_mode_from_truth(parent_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if PARENT_PIPELINE_MODE is not None: PIPELINE_MODE = PARENT_PIPELINE_MODE` | Controls validation, iteration, file handling, or error handling for this step. |
| `if "meta__pipeline_mode" not in dataframe.columns: dataframe["meta__pipeline_mode"] = PIPELINE_MODE` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: dataframe["meta__pipeline_mode"] = dataframe["meta__pipeline_mode"].fillna(PIPELINE_MODE)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth = initialize_layer_truth( truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name=LAYER_NAME, process_run_id=SILVER_PROCESS_RUN_ID, pipeline_mode=PIPELINE_M` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth = update_truth_section( silver_truth, "config_snapshot", { "silver_version": SILVER_VERSION, "cleaning_recipe_id": CLEANING_RECIPE_ID, "dataset_name_config": DATASET_N` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth = update_truth_section( silver_truth, "runtime_facts", { "parent_layer_name": "bronze", "parent_truth_hash": SILVER_PARENT_TRUTH_HASH, "dataset_name_from_parent_truth"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("Resolved Bronze parent truth hash: %s", SILVER_PARENT_TRUTH_HASH)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Resolved Silver dataset name from Bronze truth: %s", DATASET_NAME)` | Writes a logger message for traceability during notebook execution. |
| `print("Silver parent truth hash:", SILVER_PARENT_TRUTH_HASH)` | Displays a notebook-facing result for inspection. |
| `print("Silver dataset name from parent truth:", DATASET_NAME)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 17 — Create the Silver Pre-EDA Artifact Folders After Dataset Resolution

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__silver_preeda__resolved_config`
- `aligned`
- `an`
- `artifact`
- `artifacts`
- `be`
- `before`
- `bool`
- `Bronze`
- `build_artifact_dirs`
- `CONFIG`
- `Config`
- `config`
- `creating`
- `dataset`
- `Dropped`
- `DROPPED_SENSORS_FILE_NAME`
- `EXECUTION_CFG`
- `explicit`
- `export_config_snapshot`

### Outputs

- `artifacts_root`
- `CONFIG_SNAPSHOT_PATH`
- `DATASET_ID`
- `dataset_name`
- `destination`
- `DROPPED_SENSORS_DATA_PATH`
- `family`
- `FEATURE_REGISTRY_PATH`
- `SILVER_ARTIFACTS_PATH`
- `SILVER_CONFIG_DIR`
- `SILVER_LINEAGE_DIR`
- `SILVER_METADATA_DIR`
- `SILVER_PREEDA_ARTIFACT_DIRS`
- `SILVER_PROFILE_DIR`
- `SILVER_QUALITY_DIR`
- `SILVER_REGISTRY_DIR`
- `SILVER_SUMMARY_DIR`
- `stage`
- `subdirs`

### Key Operations

- `if DATASET_NAME is None or str(DATASET_NAME).strip() == "": raise ValueError("DATASET_NAME must be resolved from the Bronze parent truth before creating Silver artifacts.")`: Controls validation, iteration, file handling, or error handling for this step.
- `# Keep SQL dataset id aligned with the resolved dataset unless an explicit`: Documents the purpose or boundary of the surrounding notebook step.
- `# external DATASET_ID was provided.`: Documents the purpose or boundary of the surrounding notebook step.
- `DATASET_ID = os.getenv("DATASET_ID", DATASET_NAME)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_PREEDA_ARTIFACT_DIRS = build_artifact_dirs( artifacts_root=paths.artifacts, stage="silver", dataset_name=DATASET_NAME, family="preeda", subdirs=[ "registry", "profiles", "qu`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `SILVER_ARTIFACTS_PATH = SILVER_PREEDA_ARTIFACT_DIRS["root"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_REGISTRY_DIR = SILVER_PREEDA_ARTIFACT_DIRS["registry"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_PROFILE_DIR = SILVER_PREEDA_ARTIFACT_DIRS["profiles"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_QUALITY_DIR = SILVER_PREEDA_ARTIFACT_DIRS["quality"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_SUMMARY_DIR = SILVER_PREEDA_ARTIFACT_DIRS["summaries"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_METADATA_DIR = SILVER_PREEDA_ARTIFACT_DIRS["metadata"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `bool`
- `build_artifact_dirs`
- `export_config_snapshot`
- `get`
- `getenv`
- `strip`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `if DATASET_NAME is None or str(DATASET_NAME).strip() == "": raise ValueError("DATASET_NAME must be resolved from the Bronze parent truth before creating Silver artifacts.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `# Keep SQL dataset id aligned with the resolved dataset unless an explicit` | Documents the purpose or boundary of the surrounding notebook step. |
| `# external DATASET_ID was provided.` | Documents the purpose or boundary of the surrounding notebook step. |
| `DATASET_ID = os.getenv("DATASET_ID", DATASET_NAME)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_PREEDA_ARTIFACT_DIRS = build_artifact_dirs( artifacts_root=paths.artifacts, stage="silver", dataset_name=DATASET_NAME, family="preeda", subdirs=[ "registry", "profiles", "qu` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_ARTIFACTS_PATH = SILVER_PREEDA_ARTIFACT_DIRS["root"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_REGISTRY_DIR = SILVER_PREEDA_ARTIFACT_DIRS["registry"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_PROFILE_DIR = SILVER_PREEDA_ARTIFACT_DIRS["profiles"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_QUALITY_DIR = SILVER_PREEDA_ARTIFACT_DIRS["quality"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_SUMMARY_DIR = SILVER_PREEDA_ARTIFACT_DIRS["summaries"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_METADATA_DIR = SILVER_PREEDA_ARTIFACT_DIRS["metadata"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_CONFIG_DIR = SILVER_PREEDA_ARTIFACT_DIRS["config"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_LINEAGE_DIR = SILVER_PREEDA_ARTIFACT_DIRS["lineage"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_SNAPSHOT_PATH = ( SILVER_CONFIG_DIR / f"{DATASET_NAME}__silver_preeda__resolved_config.yaml"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DROPPED_SENSORS_DATA_PATH = ( SILVER_PROFILE_DIR / DROPPED_SENSORS_FILE_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FEATURE_REGISTRY_PATH = ( SILVER_REGISTRY_DIR / FEATURE_REGISTRY_FILE_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if bool(EXECUTION_CFG.get("save_config_snapshot", True)): export_config_snapshot( CONFIG, destination=CONFIG_SNAPSHOT_PATH, )` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Silver artifact root:", SILVER_ARTIFACTS_PATH)` | Displays a notebook-facing result for inspection. |
| `print("Config snapshot:", CONFIG_SNAPSHOT_PATH)` | Displays a notebook-facing result for inspection. |
| `print("Dropped sensors path:", DROPPED_SENSORS_DATA_PATH)` | Displays a notebook-facing result for inspection. |
| `print("Feature registry path:", FEATURE_REGISTRY_PATH)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 18 — Initial Silver Input Review

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Basic`
- `Column`
- `columns`
- `Data`
- `dataframe`
- `describe`
- `Descriptive`
- `df`
- `dtype`
- `dtypes`
- `EDA`
- `f`
- `head`
- `info`
- `n`
- `numeric`
- `only`
- `Overview`
- `Pre`
- `rows`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `print("=== Silver Pre-EDA: Data Overview ===")`: Displays a notebook-facing result for inspection.
- `print(f"Shape: {dataframe.shape[0]:,} rows × {dataframe.shape[1]} columns\n")`: Displays a notebook-facing result for inspection.
- `print("=== Column Types ===")`: Displays a notebook-facing result for inspection.
- `display(dataframe.dtypes.to_frame("dtype").head(20))`: Displays a notebook-facing result for inspection.
- `print("\n=== df.info() ===")`: Displays a notebook-facing result for inspection.
- `display(dataframe.info())`: Displays a notebook-facing result for inspection.
- `print("\n=== Basic Descriptive Statistics (numeric only) ===")`: Displays a notebook-facing result for inspection.
- `display(dataframe.describe().T)`: Displays a notebook-facing result for inspection.

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
| `print("=== Silver Pre-EDA: Data Overview ===")` | Displays a notebook-facing result for inspection. |
| `print(f"Shape: {dataframe.shape[0]:,} rows × {dataframe.shape[1]} columns\n")` | Displays a notebook-facing result for inspection. |
| `print("=== Column Types ===")` | Displays a notebook-facing result for inspection. |
| `display(dataframe.dtypes.to_frame("dtype").head(20))` | Displays a notebook-facing result for inspection. |
| `print("\n=== df.info() ===")` | Displays a notebook-facing result for inspection. |
| `display(dataframe.info())` | Displays a notebook-facing result for inspection. |
| `print("\n=== Basic Descriptive Statistics (numeric only) ===")` | Displays a notebook-facing result for inspection. |
| `display(dataframe.describe().T)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 19 — Remove Junk Import Columns

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `append`
- `be`
- `bool`
- `candidate`
- `column`
- `columns`
- `compile`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `default`
- `default_pattern`
- `drop`
- `else`
- `errors`
- `f`
- `flags`
- `got`

### Outputs

- `_ensure_regex_pattern`
- `column_normalized`
- `column_string`
- `dataframe_copy`
- `is_junk_prefix`
- `is_regex_match`
- `junk_prefixes`
- `normalize_candidates`
- `pattern`
- `regex`
- `remove_junk_import_columns`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `_ensure_regex_pattern`
- `append`
- `bool`
- `compile`
- `copy`
- `drop`
- `isinstance`
- `lower`
- `remove_junk_import_columns`
- `search`
- `startswith`
- `strip`
- `tuple`
- `type`
- `TypeError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 20 — Remove import-generated junk columns

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `candidates`
- `columns`
- `dataframe`
- `Dropped`
- `dropped`
- `else`
- `found`
- `info`
- `junk`
- `junk_columns_found`
- `ledger`
- `matched`
- `No`
- `pattern`
- `pattern_used`
- `regex`
- `remove_junk_import_columns`
- `s`
- `UNNAMED_COLUMN_REGEX`

### Outputs

- `data`
- `default_pattern`
- `junk_column_candidates`
- `kind`
- `logger`
- `message`
- `regex_pattern`
- `step`

### Key Operations

- `dataframe, junk_columns_found, pattern_used = remove_junk_import_columns( dataframe, junk_column_candidates=JUNK_COLUMN_CANDIDATES, regex_pattern=UNNAMED_COLUMN_REGEX, default_patt`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `if junk_columns_found: logger.info("Dropped junk columns: %s", junk_columns_found) ledger.add( kind="step", step="remove_junk_import_columns", message="Dropped junk columns via can`: Writes a logger message for traceability during notebook execution.
- `else: logger.info("No junk columns found.") ledger.add( kind="step", step="remove_junk_import_columns", message="No junk columns matched candidates/regex pattern", data={"dropped":`: Writes a logger message for traceability during notebook execution.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `add`
- `info`
- `remove_junk_import_columns`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `dataframe, junk_columns_found, pattern_used = remove_junk_import_columns( dataframe, junk_column_candidates=JUNK_COLUMN_CANDIDATES, regex_pattern=UNNAMED_COLUMN_REGEX, default_patt` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `if junk_columns_found: logger.info("Dropped junk columns: %s", junk_columns_found) ledger.add( kind="step", step="remove_junk_import_columns", message="Dropped junk columns via can` | Writes a logger message for traceability during notebook execution. |
| `else: logger.info("No junk columns found.") ledger.add( kind="step", step="remove_junk_import_columns", message="No junk columns matched candidates/regex pattern", data={"dropped":` | Writes a logger message for traceability during notebook execution. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 21 — Remove Duplicate Column Names

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `copy`
- `def`
- `DropKeep`
- `duplicated`
- `first`
- `is_unique`
- `keep`
- `loc`
- `tolist`
- `Tuple`

### Outputs

- `dataframe`
- `deduplicate_columns`
- `duplicates_found`

### Key Operations

- `def deduplicate_columns( dataframe: pd.DataFrame, *, keep: DropKeep = "first",`: Defines notebook-local logic used later in the notebook.
- `) -> Tuple[pd.DataFrame, List[str]]: dataframe = dataframe.copy() if dataframe.columns.is_unique: return dataframe, [] duplicates_found = dataframe.columns[dataframe.columns.duplic`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `copy`
- `deduplicate_columns`
- `duplicated`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def deduplicate_columns( dataframe: pd.DataFrame, *, keep: DropKeep = "first",` | Defines notebook-local logic used later in the notebook. |
| `) -> Tuple[pd.DataFrame, List[str]]: dataframe = dataframe.copy() if dataframe.columns.is_unique: return dataframe, [] duplicates_found = dataframe.columns[dataframe.columns.duplic` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 22 — Remove duplicate columns before canonicalization

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `column`
- `count`
- `dataframe`
- `deduplicate_columns`
- `detected`
- `duplicate`
- `Duplicate`
- `duplicates`
- `duplicates_columns_found`
- `else`
- `first`
- `info`
- `keep`
- `kept`
- `ledger`
- `names`
- `No`
- `occurrence`
- `Removed`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `step`

### Key Operations

- `dataframe, duplicates_columns_found = deduplicate_columns(dataframe, keep="first")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if duplicates_columns_found: logger.warning("Duplicate column names removed (kept first): %s", duplicates_columns_found)`: Writes a logger message for traceability during notebook execution.
- `else: logger.info("No duplicate column names detected.")`: Writes a logger message for traceability during notebook execution.
- `ledger.add( kind="step", step="deduplicate_columns", message="Removed duplicate column names (kept first occurrence).", data={"duplicates": duplicates_columns_found, "count": len(d`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `deduplicate_columns`
- `info`
- `names`
- `removed`
- `warning`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `dataframe, duplicates_columns_found = deduplicate_columns(dataframe, keep="first")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if duplicates_columns_found: logger.warning("Duplicate column names removed (kept first): %s", duplicates_columns_found)` | Writes a logger message for traceability during notebook execution. |
| `else: logger.info("No duplicate column names detected.")` | Writes a logger message for traceability during notebook execution. |
| `ledger.add( kind="step", step="deduplicate_columns", message="Removed duplicate column names (kept first occurrence).", data={"duplicates": duplicates_columns_found, "count": len(d` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 23 — Validate the Dataset Name for Silver

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `__`
- `an`
- `append`
- `assign`
- `astype`
- `because`
- `bool`
- `Bronze`
- `bronze_source_column`
- `but`
- `character`
- `column`
- `columns`
- `config`
- `config_value`
- `configured`
- `consistent`
- `contains`
- `dataframe`

### Outputs

- `_clean_values`
- `_normalize_dataset_name`
- `cleaned_characters`
- `config_dataset_name_normalized`
- `dataset_method`
- `dataset_name`
- `normalized_value`
- `top_value`
- `truth_dataset_name_normalized`
- `unique_values`
- `validate_dataset_name_for_silver`
- `values`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `_clean_values`
- `_normalize_dataset_name`
- `append`
- `astype`
- `dropna`
- `isalnum`
- `join`
- `lower`
- `replace`
- `strip`
- `unique`
- `validate_dataset_name_for_silver`
- `value_counts`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 24 — Validate dataset identity for the Silver layer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `against`
- `Bronze`
- `dataset`
- `dataset_method`
- `DATASET_METHOD`
- `dataset_source_col`
- `dataset_source_column`
- `DATASET_SOURCE_COLUMN`
- `dataset_validation`
- `decision`
- `ledger`
- `lower`
- `meta__dataset`
- `name`
- `parent`
- `recorded`
- `runtime_facts`
- `Silver`
- `stamped`

### Outputs

- `bronze_source_column`
- `config_value`
- `data`
- `dataframe`
- `DATASET_NAME`
- `fail_on_multiple_in_bronze`
- `kind`
- `logger`
- `message`
- `silver_truth`
- `step`
- `truth_dataset_name`

### Key Operations

- `VALIDATED_DATASET_NAME, DATASET_SOURCE_COLUMN, DATASET_METHOD = validate_dataset_name_for_silver( dataframe=dataframe, truth_dataset_name=DATASET_NAME, config_value=None, bronze_so`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `DATASET_NAME = str(VALIDATED_DATASET_NAME).strip().lower()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `silver_truth["dataset_name"] = DATASET_NAME`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_truth = update_truth_section( silver_truth, "runtime_facts", { "dataset_validation": { "dataset_name": DATASET_NAME, "dataset_source_column": DATASET_SOURCE_COLUMN, "dataset`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="decision", step="validate_dataset_name", message="Validated Bronze-stamped dataset name for Silver against parent truth and recorded the validation in Truth Store`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `lower`
- `strip`
- `update_truth_section`
- `validate_dataset_name_for_silver`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `VALIDATED_DATASET_NAME, DATASET_SOURCE_COLUMN, DATASET_METHOD = validate_dataset_name_for_silver( dataframe=dataframe, truth_dataset_name=DATASET_NAME, config_value=None, bronze_so` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DATASET_NAME = str(VALIDATED_DATASET_NAME).strip().lower()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `silver_truth["dataset_name"] = DATASET_NAME` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth = update_truth_section( silver_truth, "runtime_facts", { "dataset_validation": { "dataset_name": DATASET_NAME, "dataset_source_column": DATASET_SOURCE_COLUMN, "dataset` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="decision", step="validate_dataset_name", message="Validated Bronze-stamped dataset name for Silver against parent truth and recorded the validation in Truth Store` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 25 — Add and Confirm Required Silver Metadata Columns

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `ASSET_ID_DEFAULT_FALLBACK`
- `being`
- `cleaning_recipe_id`
- `CLEANING_RECIPE_ID`
- `cols`
- `column_name`
- `columns`
- `context`
- `copy`
- `elif`
- `else`
- `Ensure`
- `Ensured`
- `exist`
- `exists`
- `goes`
- `isoformat`
- `ledger`
- `level`

### Outputs

- `data`
- `dataframe`
- `kind`
- `logger`
- `message`
- `SILVER_PROCESSED_AT_UTC`
- `silver_truth`
- `step`

### Key Operations

- `dataframe = dataframe.copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Silver runtime context goes to truth, not dataframe`: Documents the purpose or boundary of the surrounding notebook step.
- `SILVER_PROCESSED_AT_UTC = pd.Timestamp.now(tz="UTC")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Ensure required row-level/source columns exist`: Documents the purpose or boundary of the surrounding notebook step.
- `for column_name in META_REQUIRED_COLUMNS: if column_name not in dataframe.columns: if column_name == "meta__dataset": dataframe[column_name] = pd.NA elif column_name == "meta__spli`: Controls validation, iteration, file handling, or error handling for this step.
- `silver_truth = update_truth_section( silver_truth, "runtime_facts", { "processed_at_utc": SILVER_PROCESSED_AT_UTC, "silver_version": SILVER_VERSION, "cleaning_recipe_id": CLEANING_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="silver_meta_contract", message="Ensured required row-level/source meta exists. Silver runtime metadata is being stored in Truth Store, not row-stampe`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `copy`
- `isoformat`
- `now`
- `RangeIndex`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `dataframe = dataframe.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Silver runtime context goes to truth, not dataframe` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_PROCESSED_AT_UTC = pd.Timestamp.now(tz="UTC")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Ensure required row-level/source columns exist` | Documents the purpose or boundary of the surrounding notebook step. |
| `for column_name in META_REQUIRED_COLUMNS: if column_name not in dataframe.columns: if column_name == "meta__dataset": dataframe[column_name] = pd.NA elif column_name == "meta__spli` | Controls validation, iteration, file handling, or error handling for this step. |
| `silver_truth = update_truth_section( silver_truth, "runtime_facts", { "processed_at_utc": SILVER_PROCESSED_AT_UTC, "silver_version": SILVER_VERSION, "cleaning_recipe_id": CLEANING_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="silver_meta_contract", message="Ensured required row-level/source meta exists. Silver runtime metadata is being stored in Truth Store, not row-stampe` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 26 — Resolve the Label or Status Source

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `amount`
- `Any`
- `both`
- `Candidates`
- `Choose`
- `column`
- `columns`
- `context`
- `convert`
- `count`
- `counts`
- `coverage_pct`
- `DataFrame`
- `dataframe`
- `def`
- `dropna`
- `elif`
- `else`
- `existed`
- `format`

### Outputs

- `_column_info`
- `_top_values`
- `chosen_column`
- `chosen_from`
- `chosen_info`
- `chosen_type`
- `info`
- `non_null`
- `resolve_label_or_status_source`
- `total_rows`
- `unique`
- `value_counts`

### Key Operations

- `def resolve_label_or_status_source( dataframe: pd.DataFrame, *, label_candidates: list[str], status_candidates: list[str], top_n: int = 10,`: Defines notebook-local logic used later in the notebook.
- `) -> tuple[Optional[str], str, Dict[str, Any]]: # Helper function to get value counts from dataframe column # holds the top n amount and convert the the value count information int`: Records or exports ledger information for stage-level traceability.

Important functions or methods detected:
- `_column_info`
- `_top_values`
- `context`
- `head`
- `items`
- `iter`
- `keys`
- `next`
- `notna`
- `nunique`
- `resolve_label_or_status_source`
- `source`
- `sum`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def resolve_label_or_status_source( dataframe: pd.DataFrame, *, label_candidates: list[str], status_candidates: list[str], top_n: int = 10,` | Defines notebook-local logic used later in the notebook. |
| `) -> tuple[Optional[str], str, Dict[str, Any]]: # Helper function to get value counts from dataframe column # holds the top n amount and convert the the value count information int` | Records or exports ledger information for stage-level traceability. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 27 — Review current column layout

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `dataframe`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 28 — Resolve the label source used for anomaly flags

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `coverage_pct`
- `dataframe`
- `decision`
- `found_labels`
- `found_status`
- `get`
- `keys`
- `label`
- `LABEL_COLUMN_CANDIDATES`
- `label_resolution`
- `label_source_col`
- `label_source_column`
- `LABEL_SOURCE_COLUMN`
- `label_source_info`
- `LABEL_SOURCE_INFO`
- `label_source_type`
- `LABEL_SOURCE_TYPE`
- `ledger`
- `non_null_count`

### Outputs

- `chosen_summary`
- `data`
- `found_label_columns`
- `found_status_columns`
- `HAS_LABEL_CANDIDATES`
- `HAS_STATUS_CANDIDATES`
- `kind`
- `label_candidates`
- `logger`
- `message`
- `silver_truth`
- `status_candidates`
- `step`
- `top_n`

### Key Operations

- `LABEL_SOURCE_COLUMN, LABEL_SOURCE_TYPE, LABEL_SOURCE_INFO = resolve_label_or_status_source( dataframe, label_candidates=LABEL_COLUMN_CANDIDATES, status_candidates=STATUS_COLUMN_CAN`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `found_label_columns = list(LABEL_SOURCE_INFO.get("found_labels", {}).keys())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `found_status_columns = list(LABEL_SOURCE_INFO.get("found_status", {}).keys())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `HAS_LABEL_CANDIDATES = int(len(found_label_columns) > 0)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `HAS_STATUS_CANDIDATES = int(len(found_status_columns) > 0)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `chosen_summary = { "top_values": LABEL_SOURCE_INFO.get("top_values", {}), "unique_count": LABEL_SOURCE_INFO.get("unique_count", 0), "non_null_count": LABEL_SOURCE_INFO.get("non_nul`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_truth = update_truth_section( silver_truth, "runtime_facts", { "label_resolution": { "label_source_column": LABEL_SOURCE_COLUMN, "label_source_type": LABEL_SOURCE_TYPE, "lab`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="decision", step="resolve_label_or_status_source", message="Resolved label/status source and wrote the resolution to Truth Store.", data={ "label_source_col": LABE`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `get`
- `keys`
- `resolve_label_or_status_source`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `LABEL_SOURCE_COLUMN, LABEL_SOURCE_TYPE, LABEL_SOURCE_INFO = resolve_label_or_status_source( dataframe, label_candidates=LABEL_COLUMN_CANDIDATES, status_candidates=STATUS_COLUMN_CAN` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `found_label_columns = list(LABEL_SOURCE_INFO.get("found_labels", {}).keys())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `found_status_columns = list(LABEL_SOURCE_INFO.get("found_status", {}).keys())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `HAS_LABEL_CANDIDATES = int(len(found_label_columns) > 0)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `HAS_STATUS_CANDIDATES = int(len(found_status_columns) > 0)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `chosen_summary = { "top_values": LABEL_SOURCE_INFO.get("top_values", {}), "unique_count": LABEL_SOURCE_INFO.get("unique_count", 0), "non_null_count": LABEL_SOURCE_INFO.get("non_nul` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth = update_truth_section( silver_truth, "runtime_facts", { "label_resolution": { "label_source_column": LABEL_SOURCE_COLUMN, "label_source_type": LABEL_SOURCE_TYPE, "lab` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="decision", step="resolve_label_or_status_source", message="Resolved label/status source and wrote the resolution to Truth Store.", data={ "label_source_col": LABE` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 29 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `dataframe`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 30 — Preview current dataframe rows

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dataframe`
- `head`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 31 — Protect Canonical Output Column Names

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__`
- `a`
- `already`
- `canonical`
- `canonical_name`
- `canonical_output_columns`
- `column`
- `columns`
- `contains`
- `continue`
- `copy`
- `dataset`
- `def`
- `don`
- `ensure`
- `event_step`
- `event_time`
- `existing`
- `f`
- `input`

### Outputs

- `base_new_name`
- `counter`
- `dataframe`
- `new_name`
- `protect_canonical_output_names`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `copy`
- `name`
- `protect_canonical_output_names`
- `rename`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 32 — Protect canonical metadata column names

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `A`
- `add`
- `any`
- `be`
- `can`
- `canonical`
- `Canonical`
- `canonical_name_collision_protection`
- `canonical_outputs`
- `collide`
- `collisions`
- `cols`
- `columns`
- `Confirm`
- `conflict`
- `contain`
- `created`
- `creating`
- `dataframe`
- `does`

### Outputs

- `canonical_output_columns`
- `consequence`
- `data`
- `kind`
- `logger`
- `message`
- `raw_prefix`
- `rename_map_json`
- `step`
- `why`

### Key Operations

- `# Run protection`: Documents the purpose or boundary of the surrounding notebook step.
- `dataframe, rename_map = protect_canonical_output_names( dataframe, canonical_output_columns=CANONICAL_OUTPUT_COLUMNS, raw_prefix = "raw__"`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# json my rename map`: Documents the purpose or boundary of the surrounding notebook step.
- `rename_map_json = {str(key): str(value) for key, value in rename_map.items()}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `if rename_map_json: ledger.add( kind="step", step="canonical_name_collision_protection", message="Renamed input columns that collide with canonical outputs (Policy A).", why="Preve`: Records or exports ledger information for stage-level traceability.
- `else: ledger.add( kind="step", step="canonical_name_collision_protection", message="No canonical-name collisions found (Policy A).", why="Confirm input does not contain columns tha`: Records or exports ledger information for stage-level traceability.
- `display(rename_map_json)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `display`
- `found`
- `items`
- `outputs`
- `protect_canonical_output_names`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Run protection` | Documents the purpose or boundary of the surrounding notebook step. |
| `dataframe, rename_map = protect_canonical_output_names( dataframe, canonical_output_columns=CANONICAL_OUTPUT_COLUMNS, raw_prefix = "raw__"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# json my rename map` | Documents the purpose or boundary of the surrounding notebook step. |
| `rename_map_json = {str(key): str(value) for key, value in rename_map.items()}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `if rename_map_json: ledger.add( kind="step", step="canonical_name_collision_protection", message="Renamed input columns that collide with canonical outputs (Policy A).", why="Preve` | Records or exports ledger information for stage-level traceability. |
| `else: ledger.add( kind="step", step="canonical_name_collision_protection", message="No canonical-name collisions found (Policy A).", why="Confirm input does not contain columns tha` | Records or exports ledger information for stage-level traceability. |
| `display(rename_map_json)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 33 — Build the Canonical Identity and Ordering Fields

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `api`
- `append`
- `asset__single`
- `asset_column`
- `astype`
- `bool`
- `break`
- `candidate`
- `coerce`
- `columns`
- `continue`
- `copy`
- `cumcount`
- `D`
- `dataframe`
- `DataFrame`
- `dataset_name`
- `def`
- `default_asset_id`

### Outputs

- `_ensure_grouping_columns_exists`
- `_pick_first_existing_candidate_column`
- `build_canonical_identity_and_order_master`
- `candidates`
- `chosen_parse_success_percent`
- `chosen_parse_time_series`
- `chosen_step_column`
- `chosen_time_column`
- `dataframe_copy`
- `default_asset_value`
- `default_run_value`
- `evaluate_step_candidates`
- `evaluate_time_candidates`
- `group_columns`
- `group_count`
- `info`
- `minimum_parse_success_percent`
- `numeric_series`
- `ordering_mode`
- `parse_success_percent`

### Key Operations

- `def _pick_first_existing_candidate_column(dataframe: pd.DataFrame, candidates: List[str]) -> Optional[str]: for candidate in candidates: if candidate in dataframe.columns: return c`: Defines notebook-local logic used later in the notebook.
- `def _ensure_grouping_columns_exists( dataframe: pd.DataFrame, *, asset_column: str = "meta__asset_id", run_column: str = "meta__run_id", default_asset_value: str = "asset__single",`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: if asset_column not in dataframe.columns: dataframe[asset_column] = default_asset_value if run_column not in dataframe.columns: dataframe[run_column] = default_r`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def evaluate_time_candidates( dataframe: pd.DataFrame, *, candidates: List[str], minimum_parse_success_percent: float,`: Defines notebook-local logic used later in the notebook.
- `) -> Dict[str, Any]: chosen_time_column = None chosen_parse_time_series = None chosen_parse_success_percent = None for candidate in candidates: if candidate not in dataframe.column`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def evaluate_step_candidates( dataframe: pd.DataFrame, *, candidates: List[str], minimum_parse_success_percent: float,`: Defines notebook-local logic used later in the notebook.
- `) -> Dict[str, Any]: chosen_step_column = None chosen_parse_success_percent = None for candidate in candidates: if candidate not in dataframe.columns: continue numeric_series = pd.`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def build_canonical_identity_and_order_master( dataframe: pd.DataFrame, *, dataset_name: str, time_candidates: List[str], step_candidates: List[str], tie_breaker_candidates: List[s`: Defines notebook-local logic used later in the notebook.
- `) -> Tuple[pd.DataFrame, Dict[str, Any]]: dataframe_copy = dataframe.copy() dataframe_copy = _ensure_grouping_columns_exists( dataframe_copy, default_asset_value=default_asset_id, `: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `_ensure_grouping_columns_exists`
- `_pick_first_existing_candidate_column`
- `append`
- `astype`
- `bool`
- `build_canonical_identity_and_order_master`
- `copy`
- `cumcount`
- `drop_duplicates`
- `evaluate_step_candidates`
- `evaluate_time_candidates`
- `extend`
- `floor`
- `groupby`
- `is_datetime64_any_dtype`
- `mean`
- `notna`
- `reset_index`
- `sort_values`
- `to_datetime`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def _pick_first_existing_candidate_column(dataframe: pd.DataFrame, candidates: List[str]) -> Optional[str]: for candidate in candidates: if candidate in dataframe.columns: return c` | Defines notebook-local logic used later in the notebook. |
| `def _ensure_grouping_columns_exists( dataframe: pd.DataFrame, *, asset_column: str = "meta__asset_id", run_column: str = "meta__run_id", default_asset_value: str = "asset__single",` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: if asset_column not in dataframe.columns: dataframe[asset_column] = default_asset_value if run_column not in dataframe.columns: dataframe[run_column] = default_r` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def evaluate_time_candidates( dataframe: pd.DataFrame, *, candidates: List[str], minimum_parse_success_percent: float,` | Defines notebook-local logic used later in the notebook. |
| `) -> Dict[str, Any]: chosen_time_column = None chosen_parse_time_series = None chosen_parse_success_percent = None for candidate in candidates: if candidate not in dataframe.column` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def evaluate_step_candidates( dataframe: pd.DataFrame, *, candidates: List[str], minimum_parse_success_percent: float,` | Defines notebook-local logic used later in the notebook. |
| `) -> Dict[str, Any]: chosen_step_column = None chosen_parse_success_percent = None for candidate in candidates: if candidate not in dataframe.columns: continue numeric_series = pd.` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def build_canonical_identity_and_order_master( dataframe: pd.DataFrame, *, dataset_name: str, time_candidates: List[str], step_candidates: List[str], tie_breaker_candidates: List[s` | Defines notebook-local logic used later in the notebook. |
| `) -> Tuple[pd.DataFrame, Dict[str, Any]]: dataframe_copy = dataframe.copy() dataframe_copy = _ensure_grouping_columns_exists( dataframe_copy, default_asset_value=default_asset_id, ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 34 — Build canonical row identity and ordering fields

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `ASSET_ID_DEFAULT_FALLBACK`
- `build_canonical_identity_and_order_master`
- `Built`
- `canonical`
- `canonical_info`
- `cols`
- `column`
- `columns`
- `dataframe`
- `decision`
- `event_date`
- `event_step`
- `event_time`
- `head`
- `identity`
- `ledger`
- `master`
- `meta__asset_id`
- `meta__event_id`

### Outputs

- `data`
- `dataset_name`
- `default_asset_id`
- `default_run_id`
- `kind`
- `logger`
- `message`
- `minimum_step_parse_success_percent`
- `minimum_time_parse_success_percent`
- `preview_columns`
- `silver_truth`
- `step`
- `step_candidates`
- `tie_breaker_candidates`
- `time_candidates`

### Key Operations

- `dataframe, canonical_info = build_canonical_identity_and_order_master( dataframe, dataset_name=DATASET_NAME, time_candidates=TIME_COLUMN_CANDIDATES, step_candidates=STEP_COLUMN_CAN`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_truth = update_truth_section( silver_truth, "runtime_facts", { "canonical_info": canonical_info, },`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="decision", step="build_canonical_identity_and_order_master", message="Built canonical identity + ordering master and wrote the resolution to Truth Store.", data={`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `preview_columns = [ "meta__asset_id", "meta__run_id", "event_time", "event_step", "time_index", "meta__event_id", "event_date",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `preview_columns = [column for column in preview_columns if column in dataframe.columns]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `display(dataframe[preview_columns].head(10), canonical_info)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `build_canonical_identity_and_order_master`
- `display`
- `head`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `dataframe, canonical_info = build_canonical_identity_and_order_master( dataframe, dataset_name=DATASET_NAME, time_candidates=TIME_COLUMN_CANDIDATES, step_candidates=STEP_COLUMN_CAN` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth = update_truth_section( silver_truth, "runtime_facts", { "canonical_info": canonical_info, },` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="decision", step="build_canonical_identity_and_order_master", message="Built canonical identity + ordering master and wrote the resolution to Truth Store.", data={` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `preview_columns = [ "meta__asset_id", "meta__run_id", "event_time", "event_step", "time_index", "meta__event_id", "event_date",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `preview_columns = [column for column in preview_columns if column in dataframe.columns]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `display(dataframe[preview_columns].head(10), canonical_info)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 35 — Review current column layout

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `dataframe`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 36 — Build the Binary Anomaly Flag

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_flag`

### Outputs

- `ANOMALY_FLAG_COLUMN`

### Key Operations

- `A`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `A` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 37 — Define label-to-binary normalization

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `already`
- `anomaly`
- `Any`
- `anything`
- `astype`
- `back`
- `become`
- `booleans`
- `broken`
- `coerce`
- `common`
- `Conservative`
- `conservative`
- `Convert`
- `copy`
- `coverage`
- `def`
- `else`
- `enough`

### Outputs

- `binary_series`
- `info`
- `mapped`
- `mapped_non_null_percent`
- `mapping`
- `normalize_label_to_binary`
- `numeric_non_null_percent`
- `numeric_series`
- `raw_series`
- `string_series`

### Key Operations

- `def normalize_label_to_binary( series: pd.Series,`: Defines notebook-local logic used later in the notebook.
- `) -> Tuple[pd.Series, Dict[str, Any]]: """ Convert a label-like series to 0/1. Handles common patterns: - already numeric 0/1 - booleans - strings like "normal"/"anomaly", "ok"/"fa`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `copy`
- `fillna`
- `lower`
- `map`
- `mean`
- `normalize_label_to_binary`
- `notna`
- `strip`
- `to_numeric`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def normalize_label_to_binary( series: pd.Series,` | Defines notebook-local logic used later in the notebook. |
| `) -> Tuple[pd.Series, Dict[str, Any]]: """ Convert a label-like series to 0/1. Handles common patterns: - already numeric 0/1 - booleans - strings like "normal"/"anomaly", "ok"/"fa` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 38 — Define status-to-anomaly conversion

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `all`
- `Any`
- `astype`
- `choose`
- `Convert`
- `def`
- `default`
- `Determine`
- `dropna`
- `dtype`
- `else`
- `empty`
- `fillna`
- `frequent`
- `index`
- `int64`
- `into`
- `like`
- `method`

### Outputs

- `anomaly_flag`
- `build_anomaly_flag_from_status`
- `chosen_normal_value`
- `cleaned`
- `info`
- `non_null_values`
- `raw_series`

### Key Operations

- `def build_anomaly_flag_from_status( series: pd.Series, *, normal_value: Optional[str] = None,`: Defines notebook-local logic used later in the notebook.
- `) -> Tuple[pd.Series, Dict[str, Any]]: """ Convert a status-like series into anomaly_flag: anomaly_flag = 1 if status != normal_value else 0 If normal_value is None, choose the mod`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `build_anomaly_flag_from_status`
- `dropna`
- `fillna`
- `mode`
- `normal`
- `notna`
- `nunique`
- `Series`
- `strip`
- `sum`
- `value_counts`
- `where`
- `zeros`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_anomaly_flag_from_status( series: pd.Series, *, normal_value: Optional[str] = None,` | Defines notebook-local logic used later in the notebook. |
| `) -> Tuple[pd.Series, Dict[str, Any]]: """ Convert a status-like series into anomaly_flag: anomaly_flag = 1 if status != normal_value else 0 If normal_value is None, choose the mod` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 39 — Define label-to-binary normalization

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `across`
- `add`
- `all`
- `anomaly`
- `anomaly_build_info`
- `anomaly_flag`
- `ANOMALY_FLAG_COLUMN`
- `anomaly_flag_counts`
- `anomaly_rate_percent`
- `anomaly_series`
- `Any`
- `astype`
- `available`
- `Basic`
- `being`
- `binary`
- `build_anomaly_flag`
- `build_anomaly_flag_from_status`
- `Built`

### Outputs

- `anomaly_counts`
- `anomaly_counts_json`
- `cleaned_status`
- `consequence`
- `data`
- `kind`
- `logger`
- `message`
- `normal_value_text`
- `preview_columns`
- `step`
- `why`

### Key Operations

- `anomaly_build_info: Dict[str, Any] = { "source_type": LABEL_SOURCE_TYPE, "source_column": LABEL_SOURCE_COLUMN }`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if LABEL_SOURCE_TYPE == "label" and LABEL_SOURCE_COLUMN: anomaly_series, method_info = normalize_label_to_binary(dataframe[LABEL_SOURCE_COLUMN]) dataframe[ANOMALY_FLAG_COLUMN] = an`: Controls validation, iteration, file handling, or error handling for this step.
- `elif LABEL_SOURCE_TYPE == "status" and LABEL_SOURCE_COLUMN: anomaly_series, method_info = build_anomaly_flag_from_status(dataframe[LABEL_SOURCE_COLUMN], normal_value=None) datafram`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `else: # No label/status available: default to all normal dataframe[ANOMALY_FLAG_COLUMN] = np.zeros(len(dataframe), dtype=np.int64) anomaly_build_info.update({"method": "no_label_or`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Basic summary for ledger`: Documents the purpose or boundary of the surrounding notebook step.
- `anomaly_counts = dataframe[ANOMALY_FLAG_COLUMN].value_counts(dropna=False).to_dict()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `anomaly_counts_json = {str(key): int(value) for key, value in anomaly_counts.items()}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `anomaly_build_info["anomaly_flag_counts"] = anomaly_counts_json`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `anomaly_build_info["anomaly_rate_percent"] = float((dataframe[ANOMALY_FLAG_COLUMN].mean() * 100.0)) if len(dataframe) else 0.0`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger.add( kind="step", step="build_anomaly_flag", message="Built anomaly_flag from resolved label/status source.", why="Silver requires a consistent binary anomaly flag for segme`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `anomaly_flag`
- `astype`
- `build_anomaly_flag_from_status`
- `display`
- `fillna`
- `flags`
- `get`
- `head`
- `items`
- `mean`
- `normalize_label_to_binary`
- `strip`
- `to_dict`
- `update`
- `value_counts`
- `zeros`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `anomaly_build_info: Dict[str, Any] = { "source_type": LABEL_SOURCE_TYPE, "source_column": LABEL_SOURCE_COLUMN }` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if LABEL_SOURCE_TYPE == "label" and LABEL_SOURCE_COLUMN: anomaly_series, method_info = normalize_label_to_binary(dataframe[LABEL_SOURCE_COLUMN]) dataframe[ANOMALY_FLAG_COLUMN] = an` | Controls validation, iteration, file handling, or error handling for this step. |
| `elif LABEL_SOURCE_TYPE == "status" and LABEL_SOURCE_COLUMN: anomaly_series, method_info = build_anomaly_flag_from_status(dataframe[LABEL_SOURCE_COLUMN], normal_value=None) datafram` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `else: # No label/status available: default to all normal dataframe[ANOMALY_FLAG_COLUMN] = np.zeros(len(dataframe), dtype=np.int64) anomaly_build_info.update({"method": "no_label_or` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Basic summary for ledger` | Documents the purpose or boundary of the surrounding notebook step. |
| `anomaly_counts = dataframe[ANOMALY_FLAG_COLUMN].value_counts(dropna=False).to_dict()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `anomaly_counts_json = {str(key): int(value) for key, value in anomaly_counts.items()}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `anomaly_build_info["anomaly_flag_counts"] = anomaly_counts_json` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `anomaly_build_info["anomaly_rate_percent"] = float((dataframe[ANOMALY_FLAG_COLUMN].mean() * 100.0)) if len(dataframe) else 0.0` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger.add( kind="step", step="build_anomaly_flag", message="Built anomaly_flag from resolved label/status source.", why="Silver requires a consistent binary anomaly flag for segme` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# # Quick preview` | Documents the purpose or boundary of the surrounding notebook step. |
| `preview_columns = [ column for column in [ LABEL_SOURCE_COLUMN, "status_normal_value", "is_normal", "is_anomaly", "anomaly_flag" ] if column and column in dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(dataframe[preview_columns].head(12), anomaly_build_info)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 40 — Review current column layout

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `dataframe`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 41 — Build Episode IDs from the Anomaly Signal

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `_tmp_group`
- `a`
- `an`
- `An`
- `anomaly`
- `anomaly_flag`
- `anomaly_flag_column`
- `append`
- `are`
- `asset`
- `astype`
- `available`
- `back`
- `Build`
- `by`
- `column`
- `columns`
- `computed`
- `Container`

### Outputs

- `anomaly_series`
- `build_episode_ids_from_anomaly_flag`
- `current_episode`
- `drop_tmp_group`
- `episode_ids`
- `flag`
- `group_columns`
- `group_df`
- `grouped`
- `idx`
- `in_anomaly_window`
- `ordering_column`
- `working`

### Key Operations

- `def build_episode_ids_from_anomaly_flag( dataframe: pd.DataFrame, *, anomaly_flag_column: str = "anomaly_flag", group_columns: list[str] \| None = None,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.Series: """ Build an episode id for each row in a time series dataset. An 'episode' is: normal -> anomaly_flag==1 (failure+recovery window) -> back to normal Each time we t`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `anomaly`
- `append`
- `astype`
- `build_episode_ids_from_anomaly_flag`
- `columns`
- `copy`
- `drop`
- `fillna`
- `group`
- `groupby`
- `normal`
- `Series`
- `sort_values`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_episode_ids_from_anomaly_flag( dataframe: pd.DataFrame, *, anomaly_flag_column: str = "anomaly_flag", group_columns: list[str] \| None = None,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.Series: """ Build an episode id for each row in a time series dataset. An 'episode' is: normal -> anomaly_flag==1 (failure+recovery window) -> back to normal Each time we t` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 42 — Build anomaly episode identifiers

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `anomaly`
- `anomaly_flag`
- `build_episode_ids_from_anomaly_flag`
- `Building`
- `Built`
- `dataframe`
- `Episode`
- `episode`
- `episode_column`
- `id`
- `ids`
- `info`
- `ledger`
- `meta__episode_id`
- `nunique`
- `s`
- `Silver`
- `silver_build_episode_ids`
- `sort_index`

### Outputs

- `anomaly_flag_column`
- `data`
- `episode_counts`
- `kind`
- `logger`
- `message`
- `step`

### Key Operations

- `logger.info("Building episode ids from anomaly_flag for Silver dataframe.")`: Writes a logger message for traceability during notebook execution.
- `dataframe["meta__episode_id"] = build_episode_ids_from_anomaly_flag( dataframe, anomaly_flag_column="anomaly_flag",`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `episode_counts = dataframe["meta__episode_id"].value_counts().sort_index()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `logger.info("Episode id summary: %s", episode_counts.to_dict())`: Writes a logger message for traceability during notebook execution.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger.add( kind="step", step="silver_build_episode_ids", message="Built meta__episode_id for Silver dataframe using anomaly windows.", data={ "episode_column": "meta__episode_id",`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `add`
- `build_episode_ids_from_anomaly_flag`
- `info`
- `nunique`
- `sort_index`
- `to_dict`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `logger.info("Building episode ids from anomaly_flag for Silver dataframe.")` | Writes a logger message for traceability during notebook execution. |
| `dataframe["meta__episode_id"] = build_episode_ids_from_anomaly_flag( dataframe, anomaly_flag_column="anomaly_flag",` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `episode_counts = dataframe["meta__episode_id"].value_counts().sort_index()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Episode id summary: %s", episode_counts.to_dict())` | Writes a logger message for traceability during notebook execution. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger.add( kind="step", step="silver_build_episode_ids", message="Built meta__episode_id for Silver dataframe using anomaly windows.", data={ "episode_column": "meta__episode_id",` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 43 — Identify the Candidate Feature Set

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `meta__`
- `raw__`

### Outputs

- `DEFAULT_EXCLUDE_PREFIXES`

### Key Operations

- `D`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `D` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 44 — Classify column types for feature selection

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `api`
- `boolean`
- `categorical`
- `CategoricalDtype`
- `datetime`
- `def`
- `dtype`
- `is_bool_dtype`
- `is_categorical_dtype`
- `is_datetime64_any_dtype`
- `is_numeric_dtype`
- `is_object_dtype`
- `is_string_dtype`
- `isinstance`
- `numeric`
- `other`
- `Series`
- `series`
- `text`
- `types`

### Outputs

- `classify_column_type`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `classify_column_type`
- `is_bool_dtype`
- `is_categorical_dtype`
- `is_datetime64_any_dtype`
- `is_numeric_dtype`
- `is_object_dtype`
- `is_string_dtype`
- `isinstance`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 45 — Define prefix-based feature exclusions

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `bool`
- `column_name`
- `def`
- `exclude_prefixes`
- `prefix`
- `startswith`

### Outputs

- `should_exclude_by_prefix`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `should_exclude_by_prefix`
- `startswith`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 46 — Review current column layout

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `dataframe`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 47 — Define identifier-column exclusion heuristic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `an`
- `api`
- `bool`
- `break`
- `cardinality`
- `column_name`
- `columns`
- `dataframe`
- `DataFrame`
- `def`
- `dropna`
- `etc`
- `Examples`
- `exclude`
- `extremely`
- `feature`
- `features`
- `guid`
- `has`
- `Heuristic`

### Outputs

- `identifier_keywords`
- `keyword_hit`
- `looks_like_identifier_column`
- `lower_name`
- `series`
- `total_rows`
- `unique_count`
- `unique_ratio`

### Key Operations

- `def looks_like_identifier_column( dataframe: pd.DataFrame, *, column_name: str, unique_ratio_threshold: float = 0.50,`: Defines notebook-local logic used later in the notebook.
- `) -> bool: """ Heuristic: exclude obvious identifier-like columns from features. Examples: *id*, *uuid*, *serial*, etc. or extremely high-cardinality columns. """ lower_name = colu`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `is_object_dtype`
- `is_string_dtype`
- `looks_like_identifier_column`
- `lower`
- `nunique`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def looks_like_identifier_column( dataframe: pd.DataFrame, *, column_name: str, unique_ratio_threshold: float = 0.50,` | Defines notebook-local logic used later in the notebook. |
| `) -> bool: """ Heuristic: exclude obvious identifier-like columns from features. Examples: *id*, *uuid*, *serial*, etc. or extremely high-cardinality columns. """ lower_name = colu` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 48 — Classify column types for feature selection

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `across`
- `Any`
- `append`
- `bool`
- `boolean`
- `candidate_column_count`
- `candidate_columns`
- `categorical`
- `classify_column_type`
- `column_name`
- `columns`
- `continue`
- `counts`
- `DataFrame`
- `dataframe`
- `datasets`
- `datetime`
- `decisions`
- `def`
- `Default`

### Outputs

- `column_type`
- `identify_feature_set`
- `selected_feature_columns`

### Key Operations

- `def identify_feature_set( dataframe: pd.DataFrame, *, exclude_prefixes: List[str], exclude_columns: List[str], label_source_column: Optional[str], include_categorical_features: boo`: Defines notebook-local logic used later in the notebook.
- `) -> Tuple[List[str], Dict[str, List[str]], Dict[str, Any]]: """ Returns: - selected_feature_columns: List[str] - feature_groups: Dict[group_name, List[str]] - info: Dict[str, Any]`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append`
- `bool`
- `boolean`
- `classify_column_type`
- `extend`
- `identify_feature_set`
- `looks_like_identifier_column`
- `should_exclude_by_prefix`
- `sorted`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def identify_feature_set( dataframe: pd.DataFrame, *, exclude_prefixes: List[str], exclude_columns: List[str], label_source_column: Optional[str], include_categorical_features: boo` | Defines notebook-local logic used later in the notebook. |
| `) -> Tuple[List[str], Dict[str, List[str]], Dict[str, Any]]: """ Returns: - selected_feature_columns: List[str] - feature_groups: Dict[group_name, List[str]] - info: Dict[str, Any]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 49 — Select model-ready candidate features

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `CANONICAL_EXCLUDE_COLUMNS`
- `dataframe`
- `DEFAULT_EXCLUDE_PREFIXES`
- `extend`
- `FEATURE_COLUMNS`
- `FEATURE_GROUPS`
- `FEATURE_INFO`
- `identify_feature_set`
- `LABEL_EXCLUDE_COLUMNS`

### Outputs

- `exclude_columns`
- `exclude_columns_combined`
- `exclude_prefixes`
- `include_categorical_features`
- `include_datetime_features`
- `include_text_features`
- `label_source_column`

### Key Operations

- `exclude_columns_combined = []`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `exclude_columns_combined.extend(CANONICAL_EXCLUDE_COLUMNS)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `exclude_columns_combined.extend(LABEL_EXCLUDE_COLUMNS)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `FEATURE_COLUMNS, FEATURE_GROUPS, FEATURE_INFO = identify_feature_set( dataframe, exclude_prefixes=DEFAULT_EXCLUDE_PREFIXES, exclude_columns=exclude_columns_combined, label_source_c`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `extend`
- `identify_feature_set`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `exclude_columns_combined = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `exclude_columns_combined.extend(CANONICAL_EXCLUDE_COLUMNS)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `exclude_columns_combined.extend(LABEL_EXCLUDE_COLUMNS)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FEATURE_COLUMNS, FEATURE_GROUPS, FEATURE_INFO = identify_feature_set( dataframe, exclude_prefixes=DEFAULT_EXCLUDE_PREFIXES, exclude_columns=exclude_columns_combined, label_source_c` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 50 — Identify Columns That May Need One-Hot Encoding Later

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `api`
- `be`
- `bool`
- `categorical`
- `CategoricalDtype`
- `column_name`
- `columns`
- `DataFrame`
- `def`
- `describing`
- `dtype`
- `encode`
- `encoded`
- `encoding`
- `excluded_columns`
- `fields`
- `file`
- `Gold`
- `hot`
- `Identify`

### Outputs

- `categorical_columns`
- `identify_one_hot_encoding_columns`
- `one_hot_encoding_truths`
- `working_excluded_columns`

### Key Operations

- `def identify_one_hot_encoding_columns( silver_dataframe: pd.DataFrame, *, excluded_columns: list[str] \| None = None,`: Defines notebook-local logic used later in the notebook.
- `) -> tuple[list[str], dict]: """ Identify categorical columns that should be one-hot encoded later in Gold. Returns ------- one_hot_encoding_columns : list[str] Ordered list of cat`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `and`
- `bool`
- `identify_one_hot_encoding_columns`
- `is_bool_dtype`
- `is_categorical_dtype`
- `is_object_dtype`
- `isinstance`
- `sorted`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def identify_one_hot_encoding_columns( silver_dataframe: pd.DataFrame, *, excluded_columns: list[str] \| None = None,` | Defines notebook-local logic used later in the notebook. |
| `) -> tuple[list[str], dict]: """ Identify categorical columns that should be one-hot encoded later in Gold. Returns ------- one_hot_encoding_columns : list[str] Ordered list of cat` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 51 — Apply feature exclusion rules

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dataframe`
- `exclude_columns_combined`
- `identify_one_hot_encoding_columns`
- `needs_one_hot_encoding`
- `one_hot_encoding_columns`
- `silver_one_hot_encoding_columns`
- `silver_one_hot_encoding_truths`
- `silver_truth`

### Outputs

- `excluded_columns`
- `silver_dataframe`

### Key Operations

- `silver_one_hot_encoding_columns, silver_one_hot_encoding_truths = identify_one_hot_encoding_columns( silver_dataframe=dataframe, excluded_columns=exclude_columns_combined`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_truth["needs_one_hot_encoding"] = silver_one_hot_encoding_truths["needs_one_hot_encoding"]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_truth["one_hot_encoding_columns"] = silver_one_hot_encoding_truths["one_hot_encoding_columns"]`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `identify_one_hot_encoding_columns`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `silver_one_hot_encoding_columns, silver_one_hot_encoding_truths = identify_one_hot_encoding_columns( silver_dataframe=dataframe, excluded_columns=exclude_columns_combined` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth["needs_one_hot_encoding"] = silver_one_hot_encoding_truths["needs_one_hot_encoding"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth["one_hot_encoding_columns"] = silver_one_hot_encoding_truths["one_hot_encoding_columns"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 52 — Mid-Workflow Structural Review

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `dataframe`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 53 — Review dataframe structure and dtypes

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dataframe`
- `info`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`
- `info`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 54 — Review intermediate output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `FEATURE_COLUMNS`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 55 — Mid-Workflow Structural Review

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `copy`
- `dataframe`
- `FEATURE_COLUMNS`

### Outputs

- `dataframe_backup`
- `FEATURE_COLUMNS_backup`

### Key Operations

- `dataframe_backup = dataframe.copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `FEATURE_COLUMNS_backup = FEATURE_COLUMNS`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `copy`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `dataframe_backup = dataframe.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FEATURE_COLUMNS_backup = FEATURE_COLUMNS` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 56 — Review dataframe structure and dtypes

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dataframe_backup`
- `info`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `#`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `display`
- `info`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `#` | Documents the purpose or boundary of the surrounding notebook step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 57 — Mid-Workflow Structural Review

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `copy`
- `dataframe_backup`

### Outputs

- `dataframe`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `copy`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 58 — Review dataframe structure and dtypes

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dataframe`
- `info`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`
- `info`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 59 — Evaluate Missingness and Quarantine High-Missing Features

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `all`
- `api`
- `ascending`
- `audit_dict`
- `bool`
- `by`
- `clean_df`
- `column`
- `copy`
- `def`
- `drop`
- `drop_all_null`
- `drop_reasons`
- `dropped_dataframe_out_path`
- `dropped_df_or_none`
- `dropped_features`
- `dropped_missing_percentage`
- `Drops`
- `dtype`

### Outputs

- `all_null`
- `audit`
- `columns`
- `compute_missingness_percentage`
- `dataframe`
- `dropped`
- `dropped_dataframe`
- `kept`
- `mask`
- `missing_percentage`
- `over_thresh`
- `percentage`
- `present`
- `quarantine_features_by_missingness`

### Key Operations

- `def compute_missingness_percentage( dataframe: pd.DataFrame, *, columns: List[str], sort_desc: bool = True,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.Series: """ Returns a Series indexed by column name with percent missing (0..100). Only includes columns that exist in the dataframe. """ columns = [column for column in co`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def quarantine_features_by_missingness( dataframe: pd.DataFrame, *, feature_columns: List[str], threshold_percentage: float, drop_all_null: bool = True, numeric_only: bool = True, `: Defines notebook-local logic used later in the notebook.
- `) -> Tuple[pd.DataFrame, List[str], List[str], pd.Series, Dict[str, object], Optional[pd.DataFrame]]: """ Drops features whose missingness >= threshold_percentage (and optionally a`: Writes an artifact or output used for review or downstream notebooks.

Important functions or methods detected:
- `compute_missingness_percentage`
- `copy`
- `drop`
- `is_numeric_dtype`
- `isna`
- `mean`
- `missing`
- `mkdir`
- `mul`
- `quarantine_features_by_missingness`
- `Series`
- `sort_values`
- `threshold_percentage`
- `to_parquet`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def compute_missingness_percentage( dataframe: pd.DataFrame, *, columns: List[str], sort_desc: bool = True,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.Series: """ Returns a Series indexed by column name with percent missing (0..100). Only includes columns that exist in the dataframe. """ columns = [column for column in co` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def quarantine_features_by_missingness( dataframe: pd.DataFrame, *, feature_columns: List[str], threshold_percentage: float, drop_all_null: bool = True, numeric_only: bool = True, ` | Defines notebook-local logic used later in the notebook. |
| `) -> Tuple[pd.DataFrame, List[str], List[str], pd.Series, Dict[str, object], Optional[pd.DataFrame]]: """ Drops features whose missingness >= threshold_percentage (and optionally a` | Writes an artifact or output used for review or downstream notebooks. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: Parquet output.

## Code Cell 60 — Quarantine features with excessive missingness

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dataframe`
- `drop_reasons`
- `dropped_dataframe`
- `dropped_features`
- `dropped_missing_pct`
- `dropped_missing_percentage`
- `DROPPED_SENSORS_DATA_PATH`
- `kept_features`
- `missing_audit`
- `missing_pct`
- `quarantine_features_by_missingness`
- `QUARANTINE_MISSING_PCT`
- `runtime_facts`
- `sensor_drop_audit`
- `threshold_pct`
- `update_truth_section`

### Outputs

- `drop_all_null`
- `dropped_dataframe_out_path`
- `feature_columns`
- `numeric_only`
- `save_dropped_dataframe`
- `silver_truth`
- `threshold_percentage`

### Key Operations

- `dataframe, FEATURE_COLUMNS, dropped_features, missing_pct, missing_audit, dropped_dataframe = quarantine_features_by_missingness( dataframe, feature_columns=FEATURE_COLUMNS, thresh`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_truth= update_truth_section( silver_truth, "runtime_facts", { "sensor_drop_audit": { "threshold_pct": float(QUARANTINE_MISSING_PCT), "kept_features": FEATURE_COLUMNS, "dropp`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `quarantine_features_by_missingness`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `dataframe, FEATURE_COLUMNS, dropped_features, missing_pct, missing_audit, dropped_dataframe = quarantine_features_by_missingness( dataframe, feature_columns=FEATURE_COLUMNS, thresh` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth= update_truth_section( silver_truth, "runtime_facts", { "sensor_drop_audit": { "threshold_pct": float(QUARANTINE_MISSING_PCT), "kept_features": FEATURE_COLUMNS, "dropp` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 61 — Review dataframe structure and dtypes

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dropped_dataframe`
- `info`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `#`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `display`
- `info`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `#` | Documents the purpose or boundary of the surrounding notebook step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 62 — Review intermediate output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dropped_features`
- `missing_audit`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 63 — Review intermediate output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `missing_audit`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 64 — Review intermediate output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `Any`
- `any`
- `append`
- `astype`
- `bool`
- `broken`
- `cast`
- `column`
- `columns`
- `continue`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `dependent`
- `diff_gate_pct`
- `else`
- `eq`
- `f`

### Outputs

- `any_small_state`
- `compute_global_missingness`
- `dataframe_copy`
- `dataframe_state`
- `features`
- `label_resolution`
- `missing_global`
- `missing_global_percent`
- `missing_state`
- `runtime_facts`
- `spread`
- `state_column`
- `state_column_synthetic`
- `state_list`
- `state_map`
- `truth_state_column`
- `unmapped_dataframe`
- `value`

### Key Operations

- `def compute_global_missingness( dataframe: pd.DataFrame, *, feature_columns: List[str], silver_truth: Dict[str, Any], state_list: Optional[List[str]] = None, state_column_fallback:`: Defines notebook-local logic used later in the notebook.
- `) -> Dict[str, object]: if state_list is None: state_list = ["normal", "abnormal", "recovery"] if state_map is None: state_map = { "normal": "normal", "broken": "abnormal", "recove`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `any`
- `append`
- `astype`
- `bool`
- `cast`
- `compute_global_missingness`
- `copy`
- `eq`
- `get`
- `in`
- `isfinite`
- `isna`
- `items`
- `KeyError`
- `lower`
- `map`
- `max`
- `mean`
- `min`
- `mul`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def compute_global_missingness( dataframe: pd.DataFrame, *, feature_columns: List[str], silver_truth: Dict[str, Any], state_list: Optional[List[str]] = None, state_column_fallback:` | Defines notebook-local logic used later in the notebook. |
| `) -> Dict[str, object]: if state_list is None: state_list = ["normal", "abnormal", "recovery"] if state_map is None: state_map = { "normal": "normal", "broken": "abnormal", "recove` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 65 — Summarize global missingness after quarantine

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `add`
- `all`
- `Any`
- `artifact_paths`
- `broken`
- `Building`
- `cast`
- `compute_global_missingness`
- `dataframe`
- `decision`
- `diff_gate_pct`
- `drop_reasons`
- `Dropped`
- `dropped`
- `dropped_count`
- `dropped_features`
- `dropped_missing_pct`
- `dropped_missing_percentage`
- `DROPPED_SENSORS_DATA_PATH`

### Outputs

- `consequence`
- `data`
- `drop_reasons_map`
- `dropped_missing_pct_map`
- `feature_columns`
- `gate`
- `kind`
- `logger`
- `message`
- `minimum_rows_per_state_for_gate`
- `missing_audit_map`
- `missing_state_difference_gate_percentage`
- `missingness_audit`
- `missingness_audit_map`
- `payload`
- `rows_by_state_map`
- `silver_truth`
- `state_column_fallback`
- `state_list`
- `state_list_raw`

### Key Operations

- `missingness_audit = compute_global_missingness( dataframe, feature_columns=FEATURE_COLUMNS, silver_truth=silver_truth, state_list=["normal", "abnormal", "recovery"], state_column_f`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_audit_map = cast(Dict[str, Any], missingness_audit)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `gate = cast( Dict[str, Any], missingness_audit_map.get("missingness_state_gate_params", {}) or {},`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_audit_map = cast(Dict[str, Any], missing_audit)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `drop_reasons_map = cast( Dict[str, Any], missing_audit_map.get("drop_reasons", {}) or {},`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `dropped_missing_pct_map = cast( Dict[str, Any], missing_audit_map.get("dropped_missing_percentage", {}) or {},`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `rows_by_state_map = cast( Dict[str, Any], gate.get("rows_by_state", {}) or {},`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `cast`
- `compute_global_missingness`
- `get`
- `head`
- `isinstance`
- `items`
- `threshold`
- `to_dict`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `missingness_audit = compute_global_missingness( dataframe, feature_columns=FEATURE_COLUMNS, silver_truth=silver_truth, state_list=["normal", "abnormal", "recovery"], state_column_f` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_audit_map = cast(Dict[str, Any], missingness_audit)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gate = cast( Dict[str, Any], missingness_audit_map.get("missingness_state_gate_params", {}) or {},` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_audit_map = cast(Dict[str, Any], missing_audit)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `drop_reasons_map = cast( Dict[str, Any], missing_audit_map.get("drop_reasons", {}) or {},` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `dropped_missing_pct_map = cast( Dict[str, Any], missing_audit_map.get("dropped_missing_percentage", {}) or {},` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `rows_by_state_map = cast( Dict[str, Any], gate.get("rows_by_state", {}) or {},` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `state_map_from_gate = cast( Dict[str, Any], gate.get("state_map", {}) or {},` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `state_list_raw = gate.get("state_list", [])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if isinstance(state_list_raw, list): state_list_values = [str(value) for value in state_list_raw]` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: state_list_values = []` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Building one payload and reuse it for the Truth store and Ledger file` | Documents the purpose or boundary of the surrounding notebook step. |
| `payload = { # --- quarantine decision --- "threshold_pct": float(QUARANTINE_MISSING_PCT), "dropped_count": int(len(dropped_features)), "dropped_features": list(dropped_features), "` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# ---- Truth: runtime_facts ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `silver_truth = update_truth_section( silver_truth, "runtime_facts", {"missingness_quarantine": payload},` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# ---- Truth: artifact_paths dropped-sensors parquet ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `if DROPPED_SENSORS_DATA_PATH is not None: silver_truth = update_truth_section( silver_truth, "artifact_paths", {"silver_preeda_dropped_sensors_parquet": str(DROPPED_SENSORS_DATA_PA` | Controls validation, iteration, file handling, or error handling for this step. |
| `# ---- Ledger: same payload ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger.add( kind="decision", step="missingness_quarantine", message="Dropped features exceeding missingness threshold (or all-null).", why="High-missingness features add noise/inst` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Saved missingness_quarantine payload to Truth + Ledger.")` | Records or exports ledger information for stage-level traceability. |
| `print("Dropped:", payload["dropped_features"])` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 66 — Define configuration mapping guards

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `cfg_require_mapping`
- `get`
- `global`
- `missingness_audit`
- `normal`
- `recovery`
- `selected_sensor`
- `sensor_48`

### Outputs

- `abnormal_missingness`
- `missingness_audit_map`
- `missingness_pct_all`
- `missingness_pct_by_state`
- `normal_missingness`
- `recovery_missingness`
- `sensor`

### Key Operations

- `sensor = "sensor_48"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `missingness_audit_map = cfg_require_mapping( missingness_audit, "missingness_audit",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_pct_all = cfg_require_mapping( missingness_audit_map.get("missingness_pct_all", {}), "missingness_audit['missingness_pct_all']",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_pct_by_state = cfg_require_mapping( missingness_audit_map.get("missingness_pct_by_state", {}), "missingness_audit['missingness_pct_by_state']",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `normal_missingness = cfg_require_mapping( missingness_pct_by_state.get("normal", {}), "missingness_pct_by_state['normal']",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `abnormal_missingness = cfg_require_mapping( missingness_pct_by_state.get("abnormal", {}), "missingness_pct_by_state['abnormal']",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `recovery_missingness = cfg_require_mapping( missingness_pct_by_state.get("recovery", {}), "missingness_pct_by_state['recovery']",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `cfg_require_mapping`
- `get`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `sensor = "sensor_48"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `missingness_audit_map = cfg_require_mapping( missingness_audit, "missingness_audit",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_pct_all = cfg_require_mapping( missingness_audit_map.get("missingness_pct_all", {}), "missingness_audit['missingness_pct_all']",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_pct_by_state = cfg_require_mapping( missingness_audit_map.get("missingness_pct_by_state", {}), "missingness_audit['missingness_pct_by_state']",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `normal_missingness = cfg_require_mapping( missingness_pct_by_state.get("normal", {}), "missingness_pct_by_state['normal']",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `abnormal_missingness = cfg_require_mapping( missingness_pct_by_state.get("abnormal", {}), "missingness_pct_by_state['abnormal']",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `recovery_missingness = cfg_require_mapping( missingness_pct_by_state.get("recovery", {}), "missingness_pct_by_state['recovery']",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("selected_sensor:", sensor)` | Displays a notebook-facing result for inspection. |
| `print("global:", missingness_pct_all.get(sensor))` | Displays a notebook-facing result for inspection. |
| `print("normal:", normal_missingness.get(sensor))` | Displays a notebook-facing result for inspection. |
| `print("abnormal:", abnormal_missingness.get(sensor))` | Displays a notebook-facing result for inspection. |
| `print("recovery:", recovery_missingness.get(sensor))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 67 — Review intermediate output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `silver_truth`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 68 — Finalize the Feature Lists After Quarantine

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `already`
- `Attach`
- `audit`
- `by`
- `case`
- `check`
- `column`
- `columns`
- `counts`
- `drop_reasons`
- `Dropped`
- `dropped`
- `dropped_count`
- `dropped_missing_pct`
- `dropped_missing_percentage`
- `drops`
- `else`
- `empty`
- `Ensure`
- `features`

### Outputs

- `dropped_features`
- `dropped_set`
- `FEATURE_COLUMNS`
- `FEATURE_GROUPS`
- `FEATURE_INFO`

### Key Operations

- `# Normalize the dropped features just in case.`: Documents the purpose or boundary of the surrounding notebook step.
- `dropped_features = dropped_features or []`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `dropped_set = set(dropped_features)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# 1) Ensure FEATURE_COLUMNS is "final schema" (already updated by quarantine return)`: Documents the purpose or boundary of the surrounding notebook step.
- `FEATURE_COLUMNS = [column for column in FEATURE_COLUMNS if column not in dropped_set]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# 2) Rebuild FEATURE_GROUPS to only include final schema columns`: Documents the purpose or boundary of the surrounding notebook step.
- `# - keeps original group names`: Documents the purpose or boundary of the surrounding notebook step.
- `# - drops columns that were quarantined`: Documents the purpose or boundary of the surrounding notebook step.
- `# - drops columns not in FEATURE_COLUMNS`: Documents the purpose or boundary of the surrounding notebook step.
- `if "FEATURE_GROUPS" in globals() and isinstance(FEATURE_GROUPS, dict): FEATURE_GROUPS = { group: [column for column in columns if column in FEATURE_COLUMNS and column not in droppe`: Controls validation, iteration, file handling, or error handling for this step.
- `else: FEATURE_GROUPS = {}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# 3) Update FEATURE_INFO counts and quarantine summary`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `get`
- `globals`
- `isinstance`
- `items`
- `update`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Normalize the dropped features just in case.` | Documents the purpose or boundary of the surrounding notebook step. |
| `dropped_features = dropped_features or []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `dropped_set = set(dropped_features)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# 1) Ensure FEATURE_COLUMNS is "final schema" (already updated by quarantine return)` | Documents the purpose or boundary of the surrounding notebook step. |
| `FEATURE_COLUMNS = [column for column in FEATURE_COLUMNS if column not in dropped_set]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# 2) Rebuild FEATURE_GROUPS to only include final schema columns` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - keeps original group names` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - drops columns that were quarantined` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - drops columns not in FEATURE_COLUMNS` | Documents the purpose or boundary of the surrounding notebook step. |
| `if "FEATURE_GROUPS" in globals() and isinstance(FEATURE_GROUPS, dict): FEATURE_GROUPS = { group: [column for column in columns if column in FEATURE_COLUMNS and column not in droppe` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: FEATURE_GROUPS = {}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# 3) Update FEATURE_INFO counts and quarantine summary` | Documents the purpose or boundary of the surrounding notebook step. |
| `if "FEATURE_INFO" not in globals() or not isinstance(FEATURE_INFO, dict): FEATURE_INFO = {}` | Controls validation, iteration, file handling, or error handling for this step. |
| `FEATURE_INFO["selected_feature_count"] = int(len(FEATURE_COLUMNS))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FEATURE_INFO["group_counts"] = {group: int(len(columns)) for group, columns in FEATURE_GROUPS.items()}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FEATURE_INFO["missingness_quarantine"] = { "threshold_pct": float(QUARANTINE_MISSING_PCT), "dropped_count": int(len(dropped_features)), "dropped_features": list(dropped_features),` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Attach my audit here` | Documents the purpose or boundary of the surrounding notebook step. |
| `if "missing_audit" in globals() and isinstance(missing_audit, dict): FEATURE_INFO["missingness_quarantine"].update( { "dropped_missing_pct": missing_audit.get("dropped_missing_perc` | Controls validation, iteration, file handling, or error handling for this step. |
| `# 4) Quick sanity check` | Documents the purpose or boundary of the surrounding notebook step. |
| `print("Final FEATURE_COLUMNS:", len(FEATURE_COLUMNS))` | Displays a notebook-facing result for inspection. |
| `print("Final FEATURE_GROUPS:", len(FEATURE_GROUPS), "groups")` | Displays a notebook-facing result for inspection. |
| `print("Dropped:", FEATURE_INFO["missingness_quarantine"]["dropped_count"], FEATURE_INFO["missingness_quarantine"]["dropped_features"])` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 69 — Create a Stable Feature Set Identifier

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `based`
- `column`
- `def`
- `Deterministic`
- `encode`
- `f`
- `feature`
- `feature_columns`
- `feature_set__`
- `hashlib`
- `hexdigest`
- `identifier`
- `join`
- `md5`
- `name`
- `names`
- `on`
- `sorted`
- `utf`

### Outputs

- `build_feature_set_identifier`
- `digest`
- `normalized`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `build_feature_set_identifier`
- `encode`
- `hexdigest`
- `join`
- `md5`
- `sorted`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 70 — Apply feature exclusion rules

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `build_feature_set_identifier`
- `cols`
- `columns`
- `dataframe`
- `decision`
- `DEFAULT_EXCLUDE_PREFIXES`
- `exclude_columns`
- `exclude_columns_combined`
- `exclude_prefixes`
- `feature`
- `FEATURE_COLUMNS`
- `FEATURE_GROUPS`
- `feature_groups`
- `feature_set`
- `feature_set_id`
- `finalize_feature_set`
- `Finalized`
- `g`
- `items`

### Outputs

- `data`
- `FEATURE_COUNT`
- `FEATURE_SET_IDENTIFIER`
- `kind`
- `logger`
- `message`
- `silver_truth`
- `step`

### Key Operations

- `FEATURE_SET_IDENTIFIER = build_feature_set_identifier(FEATURE_COLUMNS)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `FEATURE_COUNT = int(len(FEATURE_COLUMNS))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `silver_truth = update_truth_section( silver_truth, "runtime_facts", { "feature_set": { "feature_set_id": FEATURE_SET_IDENTIFIER, "feature_count": FEATURE_COUNT, } },`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="decision", step="finalize_feature_set", message="Finalized feature set and wrote feature set metadata to Truth Store.", data={ "feature_set_id": FEATURE_SET_IDENT`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `build_feature_set_identifier`
- `items`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `FEATURE_SET_IDENTIFIER = build_feature_set_identifier(FEATURE_COLUMNS)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FEATURE_COUNT = int(len(FEATURE_COLUMNS))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `silver_truth = update_truth_section( silver_truth, "runtime_facts", { "feature_set": { "feature_set_id": FEATURE_SET_IDENTIFIER, "feature_count": FEATURE_COUNT, } },` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="decision", step="finalize_feature_set", message="Finalized feature set and wrote feature set metadata to Truth Store.", data={ "feature_set_id": FEATURE_SET_IDENT` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 71 — Reorder the Silver Columns into a Cleaner Final Layout

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append`
- `column`
- `columns`
- `def`
- `existing_columns`
- `kept`

### Outputs

- `safe_list_columns`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append`
- `safe_list_columns`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 72 — Define metadata column ordering helper

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append`
- `column`
- `def`
- `existing_columns`
- `meta__`
- `meta_columns`
- `startswith`

### Outputs

- `collect_meta_columns`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append`
- `collect_meta_columns`
- `startswith`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 73 — Define metadata column ordering helper

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anything`
- `append`
- `canonical_non_meta_order`
- `collect_meta_columns`
- `column`
- `columns`
- `copy`
- `dataframe`
- `DataFrame`
- `def`
- `extend`
- `feature_columns`
- `final_order`
- `groups`
- `label_columns_order`
- `order`
- `original`
- `preserve`
- `primary`
- `Remainder`

### Outputs

- `canonical_columns`
- `existing_columns`
- `feature_columns_present`
- `label_columns`
- `meta_columns`
- `ordered_set`
- `reorder_silver_columns`

### Key Operations

- `def reorder_silver_columns( dataframe: pd.DataFrame, *, canonical_non_meta_order: List[str], label_columns_order: List[str], feature_columns: List[str],`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: existing_columns = list(dataframe.columns) meta_columns = collect_meta_columns(existing_columns) meta_columns = sorted(meta_columns) canonical_columns = safe_lis`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append`
- `collect_meta_columns`
- `columns`
- `copy`
- `extend`
- `reorder_silver_columns`
- `safe_list_columns`
- `sorted`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def reorder_silver_columns( dataframe: pd.DataFrame, *, canonical_non_meta_order: List[str], label_columns_order: List[str], feature_columns: List[str],` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: existing_columns = list(dataframe.columns) meta_columns = collect_meta_columns(existing_columns) meta_columns = sorted(meta_columns) canonical_columns = safe_lis` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 74 — Review current column layout

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `dataframe`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 75 — Define final Silver column ordering

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `reorder_silver_columns`

### Outputs

- `canonical_non_meta_order`
- `dataframe`
- `feature_columns`
- `label_columns_order`

### Key Operations

- `dataframe = reorder_silver_columns( dataframe, canonical_non_meta_order=CANONICAL_NON_META_ORDER, label_columns_order=LABEL_COLUMNS_ORDER, feature_columns=FEATURE_COLUMNS,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `reorder_silver_columns`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `dataframe = reorder_silver_columns( dataframe, canonical_non_meta_order=CANONICAL_NON_META_ORDER, label_columns_order=LABEL_COLUMNS_ORDER, feature_columns=FEATURE_COLUMNS,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 76 — Define final Silver column ordering

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `dataframe`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 77 — Run Final Quick Quality Checks

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Anomaly`
- `anomaly_flag`
- `anomaly_flag_column`
- `Any`
- `api`
- `avoid`
- `checks`
- `column`
- `columns`
- `continue`
- `DataFrame`
- `dataframe`
- `def`
- `dropna`
- `Duplicate`
- `duplicated`
- `else`
- `entries`
- `event_id_column`
- `feature`

### Outputs

- `anomaly_counts`
- `anomaly_rate_percent`
- `compute_quick_quality_checks`
- `duplicate_event_id_count`
- `duplicate_row_count`
- `missing_percent`
- `numeric_feature_count`
- `sorted_missingness`
- `top_missingness`
- `total_rows`
- `value_counts`

### Key Operations

- `def compute_quick_quality_checks( dataframe: pd.DataFrame, *, feature_columns: List[str], event_id_column: str = "meta__event_id", anomaly_flag_column: str = "anomaly_flag",`: Defines notebook-local logic used later in the notebook.
- `) -> Dict[str, Any]: total_rows = int(len(dataframe)) # Duplicate checks duplicate_row_count = int(dataframe.duplicated().sum()) duplicate_event_id_count = None if event_id_column `: Records or exports ledger information for stage-level traceability.

Important functions or methods detected:
- `columns`
- `compute_quick_quality_checks`
- `duplicated`
- `entries`
- `is_numeric_dtype`
- `isna`
- `items`
- `mean`
- `sorted`
- `sum`
- `to_dict`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def compute_quick_quality_checks( dataframe: pd.DataFrame, *, feature_columns: List[str], event_id_column: str = "meta__event_id", anomaly_flag_column: str = "anomaly_flag",` | Defines notebook-local logic used later in the notebook. |
| `) -> Dict[str, Any]: total_rows = int(len(dataframe)) # Duplicate checks duplicate_row_count = int(dataframe.duplicated().sum()) duplicate_event_id_count = None if event_id_column ` | Records or exports ledger information for stage-level traceability. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 78 — Run final Silver quality checks

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `compute_quick_quality_checks`
- `dataframe`

### Outputs

- `feature_columns`
- `quality_info`

### Key Operations

- `quality_info = compute_quick_quality_checks( dataframe, feature_columns=FEATURE_COLUMNS,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `compute_quick_quality_checks`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `quality_info = compute_quick_quality_checks( dataframe, feature_columns=FEATURE_COLUMNS,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 79 — Build the Silver Feature Registry

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `column_count`
- `columns`
- `dataframe`
- `dataset_name`
- `DATASET_NAME`
- `DEFAULT_EXCLUDE_PREFIXES`
- `exclude_columns`
- `exclude_columns_combined`
- `exclude_prefixes`
- `FEATURE_COLUMNS`
- `feature_columns`
- `feature_count`
- `FEATURE_GROUPS`
- `feature_groups`
- `feature_info`
- `FEATURE_INFO`
- `feature_registry`
- `feature_set_id`
- `FEATURE_SET_IDENTIFIER`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `feature_registry: Dict[str, Any] = { "dataset_name": DATASET_NAME, "row_count": int(len(dataframe)), "column_count": int(len(dataframe.columns)), "feature_set_id": FEATURE_SET_IDEN`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `items`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `feature_registry: Dict[str, Any] = { "dataset_name": DATASET_NAME, "row_count": int(len(dataframe)), "column_count": int(len(dataframe.columns)), "feature_set_id": FEATURE_SET_IDEN` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 80 — Build the Silver Feature Registry

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `before`
- `columns`
- `dataframe`
- `detected`
- `Duplicate`
- `duplicated`
- `f`
- `is_unique`
- `raise`
- `save`
- `tolist`
- `ValueError`

### Outputs

- `duplicates_found`

### Key Operations

- `i`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `duplicated`
- `tolist`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `i` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 81 — Build the Silver Feature Registry

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `dataframe`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 82 — Preview current dataframe rows

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dataframe`
- `head`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 83 — Build the Silver Truth Record and Save the Final Outputs

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__silver__train`
- `add`
- `append_truth_index`
- `artifact_paths`
- `artifacts`
- `astype`
- `bronze_data_path`
- `bronze_source_path`
- `build_file_fingerprint`
- `build_silver_truth_record`
- `build_truth_record`
- `Built`
- `canonicalization`
- `columns`
- `dropna`
- `DROPPED_SENSORS_DATA_PATH`
- `else`
- `exist_ok`
- `f`
- `feature_registry`

### Outputs

- `column_count`
- `data`
- `dataframe`
- `dataset_name`
- `feature_columns`
- `kind`
- `layer_name`
- `logger`
- `message`
- `meta_columns`
- `parent_truth_hash`
- `pipeline_mode`
- `row_count`
- `silver_truth`
- `SILVER_TRUTH_HASH`
- `silver_truth_path`
- `silver_truth_record`
- `step`
- `truth_base`
- `truth_dir`

### Key Operations

- `#SILVER_ARTIFACTS_PATH = paths.artifacts / "silver" / DATASET_NAME`: Documents the purpose or boundary of the surrounding notebook step.
- `#SILVER_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)`: Documents the purpose or boundary of the surrounding notebook step.
- `#SILVER_TRAIN_DATA_FILE_NAME = f"{DATASET_NAME}__silver__train.parquet"`: Documents the purpose or boundary of the surrounding notebook step.
- `silver_truth = update_truth_section( silver_truth, "source_fingerprint", build_file_fingerprint(bronze_data_path),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_truth = update_truth_section( silver_truth, "runtime_facts", { "source_run_ids": ( dataframe["meta__run_id"].dropna().astype(str).unique().tolist() if "meta__run_id" in data`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_truth = update_truth_section( silver_truth, "artifact_paths", { "bronze_source_path": str(bronze_data_path), "silver_output_dir": str(SILVER_TRAIN_DATA_PATH), "silver_output`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_truth = update_truth_section( silver_truth, "notes", { "purpose": "Silver preprocessing / canonicalization truth record", },`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_truth_record = build_truth_record( truth_base=silver_truth, row_count=len(dataframe), column_count=dataframe.shape[1], meta_columns=identify_meta_columns(dataframe), feature`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `append_truth_index`
- `astype`
- `build_file_fingerprint`
- `build_truth_record`
- `dropna`
- `identify_feature_columns`
- `identify_meta_columns`
- `locals`
- `mkdir`
- `save_truth_record`
- `stamp_truth_columns`
- `tolist`
- `unique`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `#SILVER_ARTIFACTS_PATH = paths.artifacts / "silver" / DATASET_NAME` | Documents the purpose or boundary of the surrounding notebook step. |
| `#SILVER_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)` | Documents the purpose or boundary of the surrounding notebook step. |
| `#SILVER_TRAIN_DATA_FILE_NAME = f"{DATASET_NAME}__silver__train.parquet"` | Documents the purpose or boundary of the surrounding notebook step. |
| `silver_truth = update_truth_section( silver_truth, "source_fingerprint", build_file_fingerprint(bronze_data_path),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth = update_truth_section( silver_truth, "runtime_facts", { "source_run_ids": ( dataframe["meta__run_id"].dropna().astype(str).unique().tolist() if "meta__run_id" in data` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth = update_truth_section( silver_truth, "artifact_paths", { "bronze_source_path": str(bronze_data_path), "silver_output_dir": str(SILVER_TRAIN_DATA_PATH), "silver_output` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth = update_truth_section( silver_truth, "notes", { "purpose": "Silver preprocessing / canonicalization truth record", },` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth_record = build_truth_record( truth_base=silver_truth, row_count=len(dataframe), column_count=dataframe.shape[1], meta_columns=identify_meta_columns(dataframe), feature` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_TRUTH_HASH = silver_truth_record["truth_hash"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `dataframe = stamp_truth_columns( dataframe, truth_hash=SILVER_TRUTH_HASH, parent_truth_hash=SILVER_PARENT_TRUTH_HASH, pipeline_mode=PIPELINE_MODE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_registry["truth_hash"] = SILVER_TRUTH_HASH` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_registry["parent_truth_hash"] = SILVER_PARENT_TRUTH_HASH` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth_path = save_truth_record( silver_truth_record, truth_dir=TRUTHS_PATH, dataset_name=DATASET_NAME, layer_name=LAYER_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `append_truth_index( silver_truth_record, truth_index_path=TRUTH_INDEX_PATH,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="build_silver_truth_record", message="Built and saved Silver truth record and stamped only truth lineage columns to dataframe.", data={ "silver_truth_` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: truth record.

## Code Cell 84 — Assign a reproducible feature set identifier

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__silver__train`
- `add`
- `cols`
- `columns`
- `contract`
- `dataframe`
- `DATASET_NAME`
- `dataset_name`
- `export`
- `f`
- `feature_count`
- `FEATURE_COUNT`
- `feature_registry`
- `FEATURE_REGISTRY_FILE_NAME`
- `feature_registry_path`
- `feature_set_id`
- `FEATURE_SET_IDENTIFIER`
- `Finalized`
- `ledger`
- `meta`

### Outputs

- `create_dirs`
- `data`
- `file_name`
- `file_path`
- `indent`
- `index`
- `kind`
- `logger`
- `message`
- `saved_parquet_path`
- `saved_registry_path`
- `step`

### Key Operations

- `#SILVER_TRAIN_DATA_FILE_NAME = f"{DATASET_NAME}__silver__train.parquet"`: Documents the purpose or boundary of the surrounding notebook step.
- `saved_parquet_path = save_data( dataframe, file_path=SILVER_TRAIN_DATA_PATH, file_name=SILVER_TRAIN_DATA_FILE_NAME, create_dirs=True, index=False,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `saved_registry_path = save_json( feature_registry, file_path=SILVER_REGISTRY_DIR, file_name=FEATURE_REGISTRY_FILE_NAME, create_dirs=True, indent=2,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="silver_finalize_export", message="Finalized Silver export under the strict truth-store meta contract.", data={ "dataset_name": DATASET_NAME, "silver_`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `save_data`
- `save_json`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `#SILVER_TRAIN_DATA_FILE_NAME = f"{DATASET_NAME}__silver__train.parquet"` | Documents the purpose or boundary of the surrounding notebook step. |
| `saved_parquet_path = save_data( dataframe, file_path=SILVER_TRAIN_DATA_PATH, file_name=SILVER_TRAIN_DATA_FILE_NAME, create_dirs=True, index=False,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `saved_registry_path = save_json( feature_registry, file_path=SILVER_REGISTRY_DIR, file_name=FEATURE_REGISTRY_FILE_NAME, create_dirs=True, indent=2,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="silver_finalize_export", message="Finalized Silver export under the strict truth-store meta contract.", data={ "dataset_name": DATASET_NAME, "silver_` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 85 — Save the Ledger Artifact

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__ledger`
- `DATASET_NAME`
- `f`
- `json`
- `ledger`
- `Save`
- `silver__`
- `SILVER_LINEAGE_DIR`
- `the`
- `write_json`

### Outputs

- `saved_ledger_path`

### Key Operations

- `# Save the ledger`: Documents the purpose or boundary of the surrounding notebook step.
- `saved_ledger_path = ledger.write_json( SILVER_LINEAGE_DIR / f"silver__{DATASET_NAME}__ledger.json"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `write_json`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Save the ledger` | Documents the purpose or boundary of the surrounding notebook step. |
| `saved_ledger_path = ledger.write_json( SILVER_LINEAGE_DIR / f"silver__{DATASET_NAME}__ledger.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 86 — Finalize Experiment Tracking

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `B`
- `Close`
- `data_silver_train`
- `dataset`
- `DATASET_NAME`
- `f`
- `finalize_wandb_stage`
- `finish`
- `latest`
- `logs`
- `paths`
- `root`
- `the`
- `W`
- `wandb_run`

### Outputs

- `aliases`
- `dataframe`
- `dataset_artifact_name`
- `dataset_dirs`
- `finalize_info`
- `logger`
- `logs_dir`
- `notebook_path`
- `profile`
- `project_root`
- `run`
- `stage`
- `table_key`
- `table_n`

### Key Operations

- `finalize_info = finalize_wandb_stage( run=wandb_run, stage=STAGE, dataframe=dataframe, project_root=paths.root, logs_dir=paths.logs, dataset_dirs=[paths.data_silver_train], dataset`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Close the W&B run`: Documents the purpose or boundary of the surrounding notebook step.
- `wandb_run.finish()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `finalize_wandb_stage`
- `finish`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `finalize_info = finalize_wandb_stage( run=wandb_run, stage=STAGE, dataframe=dataframe, project_root=paths.root, logs_dir=paths.logs, dataset_dirs=[paths.data_silver_train], dataset` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Close the W&B run` | Documents the purpose or boundary of the surrounding notebook step. |
| `wandb_run.finish()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 87 — Run a Final Lineage and Consistency Check

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

## Code Cell 88 — Define integer validation helper

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__name__`
- `a`
- `Any`
- `astype`
- `be`
- `cast`
- `check`
- `column`
- `column_count`
- `column_name`
- `columns`
- `contain`
- `count`
- `created`
- `dataframe`
- `dataframe_parent`
- `dictionary`
- `does`
- `dropna`
- `exists`

### Outputs

- `loaded_silver_parent_truth_hash`
- `loaded_silver_truth`
- `loaded_silver_truth_hash`
- `loaded_silver_truth_raw`
- `missing_silver_meta_columns`
- `required_silver_meta_columns`
- `silver_dataframe_truth_hash`
- `silver_parent_values`
- `silver_truth_column_count`
- `silver_truth_row_count`

### Key Operations

- `required_silver_meta_columns = [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_silver_meta_columns = [ column_name for column_name in required_silver_meta_columns if column_name not in dataframe.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if missing_silver_meta_columns: raise ValueError( f"Silver dataframe is missing required lineage columns: {missing_silver_meta_columns}" )`: Controls validation, iteration, file handling, or error handling for this step.
- `silver_dataframe_truth_hash = extract_truth_hash(dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if silver_dataframe_truth_hash is None: raise ValueError("Silver dataframe does not contain a readable meta__truth_hash value.")`: Controls validation, iteration, file handling, or error handling for this step.
- `if silver_dataframe_truth_hash != SILVER_TRUTH_HASH: raise ValueError( "Silver dataframe truth hash does not match SILVER_TRUTH_HASH:\n" f"dataframe={silver_dataframe_truth_hash}\n`: Controls validation, iteration, file handling, or error handling for this step.
- `silver_parent_values = dataframe["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if not silver_parent_values: raise ValueError("Silver dataframe is missing populated meta__parent_truth_hash values.")`: Controls validation, iteration, file handling, or error handling for this step.
- `if len(silver_parent_values) != 1: raise ValueError( "Silver dataframe has multiple parent truth hashes:\n" f"{silver_parent_values}" )`: Controls validation, iteration, file handling, or error handling for this step.
- `if silver_parent_values[0] != SILVER_PARENT_TRUTH_HASH: raise ValueError( "Silver dataframe parent truth hash does not match SILVER_PARENT_TRUTH_HASH:\n" f"dataframe_parent={silver`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `astype`
- `cast`
- `dropna`
- `exists`
- `extract_truth_hash`
- `FileNotFoundError`
- `get`
- `isinstance`
- `load_json`
- `Path`
- `require_int_value`
- `strip`
- `tolist`
- `type`
- `TypeError`
- `unique`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `required_silver_meta_columns = [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_silver_meta_columns = [ column_name for column_name in required_silver_meta_columns if column_name not in dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_silver_meta_columns: raise ValueError( f"Silver dataframe is missing required lineage columns: {missing_silver_meta_columns}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `silver_dataframe_truth_hash = extract_truth_hash(dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if silver_dataframe_truth_hash is None: raise ValueError("Silver dataframe does not contain a readable meta__truth_hash value.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if silver_dataframe_truth_hash != SILVER_TRUTH_HASH: raise ValueError( "Silver dataframe truth hash does not match SILVER_TRUTH_HASH:\n" f"dataframe={silver_dataframe_truth_hash}\n` | Controls validation, iteration, file handling, or error handling for this step. |
| `silver_parent_values = dataframe["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not silver_parent_values: raise ValueError("Silver dataframe is missing populated meta__parent_truth_hash values.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if len(silver_parent_values) != 1: raise ValueError( "Silver dataframe has multiple parent truth hashes:\n" f"{silver_parent_values}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `if silver_parent_values[0] != SILVER_PARENT_TRUTH_HASH: raise ValueError( "Silver dataframe parent truth hash does not match SILVER_PARENT_TRUTH_HASH:\n" f"dataframe_parent={silver` | Controls validation, iteration, file handling, or error handling for this step. |
| `if not Path(silver_truth_path).exists(): raise FileNotFoundError(f"Silver truth file was not created: {silver_truth_path}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `loaded_silver_truth_raw = load_json(silver_truth_path)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not isinstance(loaded_silver_truth_raw, dict): raise TypeError( "loaded_silver_truth must be a dictionary, " f"got {type(loaded_silver_truth_raw).__name__}: {loaded_silver_truth` | Controls validation, iteration, file handling, or error handling for this step. |
| `loaded_silver_truth = cast(Dict[str, Any], loaded_silver_truth_raw)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `loaded_silver_truth_hash = str(loaded_silver_truth.get("truth_hash", "")).strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if loaded_silver_truth_hash != SILVER_TRUTH_HASH: raise ValueError( "Saved Silver truth file hash does not match SILVER_TRUTH_HASH:\n" f"file={loaded_silver_truth_hash}\n" f"record` | Controls validation, iteration, file handling, or error handling for this step. |
| `loaded_silver_parent_truth_hash = str( loaded_silver_truth.get("parent_truth_hash", "")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).strip()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if loaded_silver_parent_truth_hash != SILVER_PARENT_TRUTH_HASH: raise ValueError( "Saved Silver truth file parent hash does not match SILVER_PARENT_TRUTH_HASH:\n" f"truth={loaded_s` | Controls validation, iteration, file handling, or error handling for this step. |
| `silver_truth_row_count = require_int_value( loaded_silver_truth.get("row_count"), "loaded_silver_truth['row_count']",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_truth_column_count = require_int_value( loaded_silver_truth.get("column_count"), "loaded_silver_truth['column_count']",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if silver_truth_row_count != len(dataframe): raise ValueError( "Silver truth row_count does not match dataframe row count:\n" f"truth={silver_truth_row_count}\n" f"dataframe={len(d` | Controls validation, iteration, file handling, or error handling for this step. |
| `if silver_truth_column_count != dataframe.shape[1]: raise ValueError( "Silver truth column_count does not match stamped dataframe column count:\n" f"truth={silver_truth_column_coun` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Silver PreEDA lineage sanity check passed.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 89 — Silver SQL Write Cell

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `else`
- `get`
- `globals`
- `log_silver_eda_sql`
- `Postgres`
- `skipped`
- `write`

### Outputs

- `capstone_schema`
- `dataset_id`
- `dataset_name`
- `engine`
- `notebook_globals`
- `run_id`
- `silver_eda_sql_summary_dataframe`
- `WRITE_TO_POSTGRES`

### Key Operations

- `WRITE_TO_POSTGRES = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if WRITE_TO_POSTGRES: silver_eda_sql_summary_dataframe = log_silver_eda_sql( engine=engine, capstone_schema=CAPSTONE_SCHEMA, dataset_id=DATASET_ID, run_id=RUN_ID, notebook_globals=`: Displays a notebook-facing result for inspection.
- `else: print("Postgres write skipped.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `get`
- `globals`
- `log_silver_eda_sql`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `WRITE_TO_POSTGRES = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if WRITE_TO_POSTGRES: silver_eda_sql_summary_dataframe = log_silver_eda_sql( engine=engine, capstone_schema=CAPSTONE_SCHEMA, dataset_id=DATASET_ID, run_id=RUN_ID, notebook_globals=` | Displays a notebook-facing result for inspection. |
| `else: print("Postgres write skipped.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

