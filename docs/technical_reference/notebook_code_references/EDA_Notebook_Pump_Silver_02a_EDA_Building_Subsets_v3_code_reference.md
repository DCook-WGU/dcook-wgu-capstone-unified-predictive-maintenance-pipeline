# Notebook Code Reference: EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3

Notebook path:

`notebooks/eda/EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3.ipynb`

## Notebook Purpose

This notebook builds Silver analytical subsets and clean-normal construction outputs for deeper exploratory analysis.

Notebook stage:

`Silver`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Answer | Code Cell 01, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 10, Code Cell 11, Code Cell 12, Code Cell 13, Code Cell 14, Code Cell 15, Code Cell 16, Code Cell 17, Code Cell 18, Code Cell 19, Code Cell 20, Code Cell 21, Code Cell 22, Code Cell 23, Code Cell 24, Code Cell 25, Code Cell 26, Code Cell 27, Code Cell 28, Code Cell 32, Code Cell 33, Code Cell 35, Code Cell 36, Code Cell 38, Code Cell 42, Code Cell 43, Code Cell 46, Code Cell 48, Code Cell 49, Code Cell 50 |
| Define configuration mapping guards | Code Cell 02 |
| SQL Runtime Context | Code Cell 08 |
| Review intermediate output | Code Cell 09, Code Cell 40, Code Cell 41 |
| Review window quality distribution | Code Cell 29 |
| Preview the clean-normal sensor baseline | Code Cell 30 |
| Confirm baseline profile shape | Code Cell 31 |
| Inspect an individual sensor profile | Code Cell 34, Code Cell 44, Code Cell 45 |
| Define clean-normal training quality classes | Code Cell 37 |
| Build clean-normal quality summary | Code Cell 39 |
| Plot sensor profile against the learned baseline | Code Cell 47 |
| Review final Silver subset structure | Code Cell 51 |
| Silver EDA SQL Logging Cell | Code Cell 52 |

## Code Cell 01 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__future__`
- `annotations`
- `Any`
- `append_truth_index`
- `artifact_file_path`
- `artifacts`
- `build_artifact_dirs_from_config`
- `build_truth_config_block`
- `build_truth_record`
- `cast`
- `config_loader`
- `configure_logging`
- `core`
- `database`
- `dataclass`
- `dataclasses`
- `datetime`
- `delete_dataset_run_rows`
- `execute_many`
- `export_config_snapshot`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `from __future__ import annotations`: Imports a dependency or project helper used by later cells.
- `from dataclasses import dataclass`: Imports a dependency or project helper used by later cells.
- `from datetime import datetime, timezone`: Imports a dependency or project helper used by later cells.
- `from pathlib import Path`: Imports a dependency or project helper used by later cells.
- `from typing import Any, Dict, Mapping, Union, List, Optional, Sequence, Tuple, cast`: Imports a dependency or project helper used by later cells.
- `import json`: Imports a dependency or project helper used by later cells.
- `import logging`: Imports a dependency or project helper used by later cells.
- `import math`: Imports a dependency or project helper used by later cells.
- `import re`: Imports a dependency or project helper used by later cells.
- `import wandb`: Imports a dependency or project helper used by later cells.
- `import os`: Imports a dependency or project helper used by later cells.
- `import numpy as np`: Imports a dependency or project helper used by later cells.

Important functions or methods detected:
- `import`
- `set_option`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `from __future__ import annotations` | Imports a dependency or project helper used by later cells. |
| `from dataclasses import dataclass` | Imports a dependency or project helper used by later cells. |
| `from datetime import datetime, timezone` | Imports a dependency or project helper used by later cells. |
| `from pathlib import Path` | Imports a dependency or project helper used by later cells. |
| `from typing import Any, Dict, Mapping, Union, List, Optional, Sequence, Tuple, cast` | Imports a dependency or project helper used by later cells. |
| `import json` | Imports a dependency or project helper used by later cells. |
| `import logging` | Imports a dependency or project helper used by later cells. |
| `import math` | Imports a dependency or project helper used by later cells. |
| `import re` | Imports a dependency or project helper used by later cells. |
| `import wandb` | Imports a dependency or project helper used by later cells. |
| `import os` | Imports a dependency or project helper used by later cells. |
| `import numpy as np` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `import matplotlib.pyplot as plt` | Imports a dependency or project helper used by later cells. |
| `import seaborn as sns` | Imports a dependency or project helper used by later cells. |
| `from IPython.display import display` | Imports a dependency or project helper used by later cells. |
| `from utils.core.paths import get_paths` | Imports a dependency or project helper used by later cells. |
| `from utils.core.file_io import ( load_data, save_json, load_json,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.logging_setup import ( configure_logging, log_layer_paths,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.ledger import Ledger` | Imports a dependency or project helper used by later cells. |
| `from utils.core.truths import ( make_process_run_id, identify_meta_columns, load_parent_truth_record_from_dataframe, get_dataset_name_from_truth, get_truth_hash, get_parent_truth_h` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.config_loader import ( load_pipeline_config, build_truth_config_block, set_wandb_dir_from_config, export_config_snapshot,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.artifacts import ( build_artifact_dirs_from_config, artifact_file_path,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.layer_postgres import ( read_layer_dataframe, write_layer_dataframe, prepare_layer_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.sql_notebook_helpers import ( delete_dataset_run_rows, execute_many, get_existing_dataframe, get_row_value, log_data_quality_event, log_pipeline_artifact, previ` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.medallion_sql_writers import ( log_gold_05_anomaly_detection_summary_sql, log_silver_eda_sql, write_bronze_sensor_observations_sql, write_gold_baseline_scores_s` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.notebook_context import load_notebook_context` | Imports a dependency or project helper used by later cells. |
| `pd.set_option("display.max_columns", 200)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `pd.set_option("display.width", 240)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `pd.set_option("display.max_rows", 200)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
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

## Code Cell 03 — Answer

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
- `eda_subsets`
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
- `EXECUTION_CFG`
- `extra`
- `FILENAMES`
- `kind`
- `ledger`
- `log_filename`

### Key Operations

- `# Shared notebook context`: Documents the purpose or boundary of the surrounding notebook step.
- `CONTEXT_STAGE = "silver_eda"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_DATASET = "pump"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LAYER = "silver"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_RUN_MODE = "train"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_PROFILE = "default"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LOG_FILE = "silver_eda_subsets.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.silver.eda_subsets", log_filena`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `CONTEXT_STAGE = "silver_eda"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LAYER = "silver"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_RUN_MODE = "train"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LOG_FILE = "silver_eda_subsets.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.silver.eda_subsets", log_filena` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Shared aliases used throughout the notebook` | Documents the purpose or boundary of the surrounding notebook step. |
| `paths = CTX.paths` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_MAP = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE_CFG = CTX.stage_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_CFG = CTX.stage_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
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
| `DEFAULT_FALLBACKS_CFG = CTX.default_fallbacks` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info( "Notebook context loaded", extra={ "stage": CONTEXT_STAGE, "recipe_id": CONTEXT_RECIPE_ID, "dataset": CONTEXT_DATASET, "mode": CONFIG_RUN_MODE, "profile": CONFIG_PROFI` | Writes a logger message for traceability during notebook execution. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="context_loaded", message="Loaded shared notebook context.", data={ "stage": CONTEXT_STAGE, "recipe_id": CONTEXT_RECIPE_ID, "dataset": CONTEXT_DATASET` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 04 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__silver__eda_subsets__truth`
- `a`
- `Any`
- `appends`
- `are`
- `artifact`
- `Artifacts`
- `asset__001`
- `asset_id`
- `B`
- `Base`
- `below`
- `build_truth_config_block`
- `Canonical`
- `capstone`
- `CAPSTONE_ASSET_ID`
- `CAPSTONE_RUN_ID`
- `cast`
- `cell`
- `cleaning_recipe_id`

### Outputs

- `ARTIFACTS_ROOT`
- `ASSET_ID_DEFAULT_FALLBACK`
- `BRONZE_TRAIN_DATA_PATH`
- `DATASET_NAME`
- `DATASET_NAME_CONFIG`
- `DATASET_RUN_ID`
- `FEATURE_REGISTRY_FILE_NAME`
- `LAYER_NAME`
- `LEANING_RECIPE_ID`
- `LOG_FILE_NAME`
- `LOGGER_NAME`
- `LOGS_PATH`
- `PIPELINE_MODE`
- `RUN_ID_DEFAULT_FALLBACK`
- `RUN_MODE`
- `SILVER_EDA_ARTIFACTS_ROOT`
- `SILVER_EDA_TRUTH_DIR`
- `SILVER_EDA_TRUTH_STAGE`
- `SILVER_EDA_TRUTH_SUMMARY_PATH`
- `SILVER_PROCESS_RUN_ID`

### Key Operations

- `DATASET_NAME_CONFIG = str(DATASET_CFG.get("name", "pump"))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_NAME = DATASET_NAME_CONFIG`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `TRUTH_CONFIG = cast(Dict[str, Any], build_truth_config_block(CONFIG))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_CONFIG["pipeline"] = PIPELINE`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `# ---- Stage details ----`: Documents the purpose or boundary of the surrounding notebook step.
- `STAGE = "silver_eda"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `LAYER_NAME = str(SILVER_EDA_CFG.get("layer_name", "silver_eda"))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `LEANING_RECIPE_ID = str(SILVER_EDA_CFG["cleaning_recipe_id"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_VERSION = str(VERSIONS_CFG["silver_eda"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_VERSION = str(VERSIONS_CFG["truth"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `build_truth_config_block`
- `cast`
- `get`
- `getenv`
- `make_process_run_id`
- `mkdir`
- `Path`
- `save_truth_record`
- `set_wandb_dir_from_config`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `DATASET_NAME_CONFIG = str(DATASET_CFG.get("name", "pump"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_NAME = DATASET_NAME_CONFIG` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `TRUTH_CONFIG = cast(Dict[str, Any], build_truth_config_block(CONFIG))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_CONFIG["pipeline"] = PIPELINE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Stage details ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `STAGE = "silver_eda"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LAYER_NAME = str(SILVER_EDA_CFG.get("layer_name", "silver_eda"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LEANING_RECIPE_ID = str(SILVER_EDA_CFG["cleaning_recipe_id"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_VERSION = str(VERSIONS_CFG["silver_eda"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_VERSION = str(VERSIONS_CFG["truth"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `PIPELINE_MODE = str(PIPELINE["execution_mode"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_MODE = str(RUNTIME_CFG.get("mode", CONFIG_RUN_MODE))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_PROCESS_RUN_ID = make_process_run_id( str(SILVER_EDA_CFG["process_run_id_prefix"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_SUBSET_PROCESS_RUN_ID = make_process_run_id( str(SILVER_EDA_CFG.get("subset_process_run_id_prefix", "silver_subset_process"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- W&B ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `WANDB_PROJECT = str(WANDB_CFG.get("project", "capstone"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WANDB_ENTITY = str(WANDB_CFG.get("entity", ""))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WANDB_RUN_NAME = f"{SILVER_VERSION}"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Defaults / fallbacks ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `ASSET_ID_DEFAULT_FALLBACK = str( os.getenv( "CAPSTONE_ASSET_ID", DATASET_CFG.get( "asset_id", DEFAULT_FALLBACKS_CFG.get("asset_id", "asset__001"), ), )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `RUN_ID_DEFAULT_FALLBACK = str( os.getenv( "CAPSTONE_RUN_ID", DATASET_CFG.get( "run_id", DEFAULT_FALLBACKS_CFG.get("run_id", "run__001"), ), )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DATASET_RUN_ID = str(DATASET_CFG.get("run_id", RUN_ID_DEFAULT_FALLBACK))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- File names ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_TRAIN_DATA_FILE_NAME = str(FILENAMES["silver_train_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FEATURE_REGISTRY_FILE_NAME = str( FILENAMES.get("feature_registry_file_name", "feature_registry.json")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Base paths only ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `BRONZE_TRAIN_DATA_PATH = Path(str(RESOLVED_PATHS["data_bronze_train_dir"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_TRAIN_DATA_PATH = Path(str(RESOLVED_PATHS["data_silver_train_dir"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ARTIFACTS_ROOT = Path(str(RESOLVED_PATHS["artifacts_root"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTHS_PATH = Path(str(RESOLVED_PATHS["truths_dir"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_INDEX_PATH = Path(str(RESOLVED_PATHS["truth_index_path"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LOGS_PATH = Path(str(RESOLVED_PATHS["logs_root"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Canonical Silver EDA artifact roots ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Final subdirectories are resolved in the artifact-dir cell below.` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_EDA_ARTIFACTS_ROOT = ( ARTIFACTS_ROOT / "silver" / str(DATASET_NAME) / "eda"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_SUBSET_ARTIFACTS_ROOT = ( SILVER_EDA_ARTIFACTS_ROOT / "subsets"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `TRUTH_LAYER_NAME = "silver"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_STAGE_NAME = "eda_subsets"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_NOTEBOOK_NAME = "silver_02a_eda_subsets"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# =============================================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Silver EDA Subsets Truth Directory` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Purpose:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Use the existing truth utility contract:` | Documents the purpose or boundary of the surrounding notebook step. |
| `31 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: truth record.

## Code Cell 05 — Answer

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

## Code Cell 06 — Answer

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
- `SILVER_EDA_CFG`
- `variables`

### Outputs

- `missing_silver_context_vars`
- `silver_required_context_vars`

### Key Operations

- `silver_required_context_vars = [ "SILVER_EDA_CFG",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `silver_required_context_vars = [ "SILVER_EDA_CFG",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_silver_context_vars = [ name for name in silver_required_context_vars if name not in globals()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_silver_context_vars: raise NameError(f"Missing Silver context variables: {missing_silver_context_vars}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `logger.info("Silver context sanity check passed")` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 07 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__silver__eda__resolved_config`
- `aliases`
- `Aligned`
- `aligned_onset_plots`
- `all`
- `artifact`
- `Artifact`
- `artifact_dir`
- `artifacts`
- `Backward`
- `become`
- `Canonical`
- `cells`
- `compatible`
- `CONFIG`
- `config`
- `correlation_analysis`
- `dataset`
- `DATASET_NAME_CONFIG`
- `dir`

### Outputs

- `ALIGNED_ONSET_PLOT_DIR`
- `CONFIG_SNAPSHOT_PATH`
- `CORRELATION_ARTIFACT_DIR`
- `destination`
- `DISTRIBUTION_PLOT_DIR`
- `GENERATOR_INPUT_DIR`
- `PCA_ARTIFACT_DIR`
- `SENSOR_PROFILE_DIR`
- `SILVER_EDA_ALIGNED_ONSET_PLOTS_DIR`
- `SILVER_EDA_ARTIFACT_DIR`
- `SILVER_EDA_CONFIG_DIR`
- `SILVER_EDA_CORRELATION_DIR`
- `SILVER_EDA_DISTRIBUTION_DIR`
- `SILVER_EDA_GENERATOR_INPUTS_DIR`
- `SILVER_EDA_LINEAGE_DIR`
- `SILVER_EDA_METADATA_DIR`
- `SILVER_EDA_OUTPUT_DIR`
- `SILVER_EDA_PCA_DIR`
- `SILVER_EDA_SENSOR_PROFILES_DIR`
- `SILVER_EDA_STAGE_NAME`

### Key Operations

- `# =============================================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Canonical Silver EDA Artifact Directories`: Documents the purpose or boundary of the surrounding notebook step.
- `# Purpose:`: Documents the purpose or boundary of the surrounding notebook step.
- `# Force all Silver EDA outputs into:`: Documents the purpose or boundary of the surrounding notebook step.
- `# artifacts/silver/<dataset>/eda/`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# The "subsets" folder is only for subset output files. It must not become the`: Documents the purpose or boundary of the surrounding notebook step.
- `# parent folder for the rest of the EDA artifact tree.`: Documents the purpose or boundary of the surrounding notebook step.
- `# =============================================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `SILVER_LAYER_NAME = "silver"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_EDA_STAGE_NAME = "eda"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_EDA_ARTIFACT_DIR = ( paths.artifacts / SILVER_LAYER_NAME / DATASET_NAME_CONFIG / SILVER_EDA_STAGE_NAME`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `export_config_snapshot`
- `get`
- `mkdir`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =============================================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Canonical Silver EDA Artifact Directories` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Purpose:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Force all Silver EDA outputs into:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# artifacts/silver/<dataset>/eda/` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The "subsets" folder is only for subset output files. It must not become the` | Documents the purpose or boundary of the surrounding notebook step. |
| `# parent folder for the rest of the EDA artifact tree.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =============================================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_LAYER_NAME = "silver"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_STAGE_NAME = "eda"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_ARTIFACT_DIR = ( paths.artifacts / SILVER_LAYER_NAME / DATASET_NAME_CONFIG / SILVER_EDA_STAGE_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_EDA_ALIGNED_ONSET_PLOTS_DIR = SILVER_EDA_ARTIFACT_DIR / "aligned_onset_plots"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_CONFIG_DIR = SILVER_EDA_ARTIFACT_DIR / "config"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_CORRELATION_DIR = SILVER_EDA_ARTIFACT_DIR / "correlation_analysis"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_DISTRIBUTION_DIR = SILVER_EDA_ARTIFACT_DIR / "distribution_plots"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_GENERATOR_INPUTS_DIR = SILVER_EDA_ARTIFACT_DIR / "generator_inputs"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_LINEAGE_DIR = SILVER_EDA_ARTIFACT_DIR / "lineage"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_METADATA_DIR = SILVER_EDA_ARTIFACT_DIR / "metadata"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_PCA_DIR = SILVER_EDA_ARTIFACT_DIR / "pca"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_SENSOR_PROFILES_DIR = SILVER_EDA_ARTIFACT_DIR / "sensor_profiles"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_SUBSETS_DIR = SILVER_EDA_ARTIFACT_DIR / "subsets"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_SUMMARIES_DIR = SILVER_EDA_ARTIFACT_DIR / "summaries"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_TIMELINE_OVERLAYS_DIR = SILVER_EDA_ARTIFACT_DIR / "timeline_overlays"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for artifact_dir in [ SILVER_EDA_ARTIFACT_DIR, SILVER_EDA_ALIGNED_ONSET_PLOTS_DIR, SILVER_EDA_CONFIG_DIR, SILVER_EDA_CORRELATION_DIR, SILVER_EDA_DISTRIBUTION_DIR, SILVER_EDA_GENERA` | Controls validation, iteration, file handling, or error handling for this step. |
| `]: artifact_dir.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Backward-compatible aliases` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# These keep older downstream cells working while preventing them from using` | Documents the purpose or boundary of the surrounding notebook step. |
| `# /eda/subsets as the root artifact directory.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_SUBSET_ARTIFACT_DIR = SILVER_EDA_ARTIFACT_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_SUBSET_OUTPUT_DIR = SILVER_EDA_SUBSETS_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_SUBSET_SUMMARY_DIR = SILVER_EDA_SUMMARIES_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_SUBSET_METADATA_DIR = SILVER_EDA_METADATA_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_SUBSET_CONFIG_DIR = SILVER_EDA_CONFIG_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_SUBSET_LINEAGE_DIR = SILVER_EDA_LINEAGE_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SENSOR_PROFILE_DIR = SILVER_EDA_SENSOR_PROFILES_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CORRELATION_ARTIFACT_DIR = SILVER_EDA_CORRELATION_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PCA_ARTIFACT_DIR = SILVER_EDA_PCA_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DISTRIBUTION_PLOT_DIR = SILVER_EDA_DISTRIBUTION_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GENERATOR_INPUT_DIR = SILVER_EDA_GENERATOR_INPUTS_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ALIGNED_ONSET_PLOT_DIR = SILVER_EDA_ALIGNED_ONSET_PLOTS_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TIMELINE_OVERLAY_DIR = SILVER_EDA_TIMELINE_OVERLAYS_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_SUMMARY_DIR = SILVER_EDA_SUMMARIES_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_OUTPUT_DIR = SILVER_EDA_SUBSETS_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_SNAPSHOT_PATH = ( SILVER_EDA_CONFIG_DIR / f"{DATASET_NAME_CONFIG}__silver__eda__resolved_config.yaml"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if CONFIG["execution"].get("save_config_snapshot", True): export_config_snapshot( CONFIG, destination=CONFIG_SNAPSHOT_PATH, )` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Silver EDA artifact root:", SILVER_EDA_ARTIFACT_DIR)` | Displays a notebook-facing result for inspection. |
| `print("Silver EDA subset output dir:", SILVER_EDA_SUBSETS_DIR)` | Displays a notebook-facing result for inspection. |
| `print("Aligned onset plot dir:", SILVER_EDA_ALIGNED_ONSET_PLOTS_DIR)` | Displays a notebook-facing result for inspection. |
| `print("Timeline overlay dir:", SILVER_EDA_TIMELINE_OVERLAYS_DIR)` | Displays a notebook-facing result for inspection. |

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

## Code Cell 09 — Review intermediate output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `bronze`
- `BY`
- `capstone`
- `CAPSTONE_SCHEMA`
- `capstone_schema`
- `Check`
- `Confirm`
- `connection`
- `database`
- `engine`
- `exist`
- `expected`
- `gold`
- `information_schema`
- `metadata`
- `ORDER`
- `Purpose`
- `read_sql_dataframe`
- `schemas`
- `SELECT`

### Outputs

- `params`
- `sql_smoke_check_dataframe`

### Key Operations

- `# =============================================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# SQL Smoke Check`: Documents the purpose or boundary of the surrounding notebook step.
- `# Purpose:`: Documents the purpose or boundary of the surrounding notebook step.
- `# Confirm the database connection and expected capstone schemas/tables exist.`: Documents the purpose or boundary of the surrounding notebook step.
- `# =============================================================================`: Documents the purpose or boundary of the surrounding notebook step.
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
| `# =============================================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# SQL Smoke Check` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Purpose:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Confirm the database connection and expected capstone schemas/tables exist.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =============================================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `sql_smoke_check_dataframe = read_sql_dataframe( engine, """ SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema IN (:capstone_schema, 'bronze', 'silve` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(sql_smoke_check_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 10 — Answer

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

## Code Cell 11 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `builder`
- `capstone`
- `configure_logging`
- `current_layer`
- `getLogger`
- `INFO`
- `info`
- `LOG_FILE_NAME`
- `log_layer_paths`
- `Logging`
- `logging`
- `LOGS_PATH`
- `Original`
- `paths`
- `Setup`
- `Silver`
- `silver`
- `silver_eda`
- `starting`
- `subset`

### Outputs

- `level`
- `logger`
- `overwrite_handlers`
- `subset_log_path`

### Key Operations

- `"""`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Original Logging Setup`: Documents the purpose or boundary of the surrounding notebook step.
- `subset_log_path = LOGS_PATH / LOG_FILE_NAME`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `configure_logging( "capstone", subset_log_path, level=logging.INFO, overwrite_handlers=True,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger = logging.getLogger("capstone.silver_eda")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `logger.info("Silver subset builder starting")`: Writes a logger message for traceability during notebook execution.
- `log_layer_paths(paths, current_layer="silver", logger=logger)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Logging to:", subset_log_path) """`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `configure_logging`
- `getLogger`
- `info`
- `log_layer_paths`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `"""` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Original Logging Setup` | Documents the purpose or boundary of the surrounding notebook step. |
| `subset_log_path = LOGS_PATH / LOG_FILE_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `configure_logging( "capstone", subset_log_path, level=logging.INFO, overwrite_handlers=True,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger = logging.getLogger("capstone.silver_eda")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Silver subset builder starting")` | Writes a logger message for traceability during notebook execution. |
| `log_layer_paths(paths, current_layer="silver", logger=logger)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Logging to:", subset_log_path) """` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 12 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `B`
- `CLEANING_RECIPE_ID`
- `cleaning_recipe_id`
- `info`
- `init`
- `initialized`
- `logger`
- `run`
- `s`
- `silver_eda`
- `silver_out_dir`
- `silver_path`
- `SILVER_TRAIN_DATA_FILE_NAME`
- `SILVER_TRAIN_DATA_PATH`
- `silver_version`
- `SILVER_VERSION`
- `W`
- `wandb`
- `WANDB_ENTITY`
- `WANDB_PROJECT`

### Outputs

- `config`
- `entity`
- `job_type`
- `name`
- `project`
- `wandb_run`

### Key Operations

- `# W&B`: Documents the purpose or boundary of the surrounding notebook step.
- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `wandb_run = wandb.init( project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="silver_eda", config={ "silver_version": SILVER_VERSION, "cleaning_recipe_id": CLE`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger.info("W&B initialized: %s", wandb.run.name)`: Writes a logger message for traceability during notebook execution.
- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `info`
- `init`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# W&B` | Documents the purpose or boundary of the surrounding notebook step. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb_run = wandb.init( project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="silver_eda", config={ "silver_version": SILVER_VERSION, "cleaning_recipe_id": CLE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("W&B initialized: %s", wandb.run.name)` | Writes a logger message for traceability during notebook execution. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- Artifact or state outputs detected: optional experiment tracking call.

## Code Cell 13 — Answer

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

## Code Cell 14 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `builder`
- `init`
- `initialized`
- `Initialized`
- `Original`
- `recipe_id`
- `Setup`
- `Silver`
- `silver_eda__v001`
- `STAGE`
- `stage`
- `subset`

### Outputs

- `kind`
- `ledger`
- `logger`
- `message`
- `step`

### Key Operations

- `"""`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Original Ledger Setup`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger = Ledger(stage=STAGE, recipe_id="silver_eda__v001")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ledger.add( kind="step", step="init", message="Initialized Silver subset-builder ledger.", logger=logger,`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Ledger initialized.") """`: Records or exports ledger information for stage-level traceability.

Important functions or methods detected:
- `add`
- `Ledger`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `"""` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Original Ledger Setup` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger = Ledger(stage=STAGE, recipe_id="silver_eda__v001")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger.add( kind="step", step="init", message="Initialized Silver subset-builder ledger.", logger=logger,` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Ledger initialized.") """` | Records or exports ledger information for stage-level traceability. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 15 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `column_count`
- `columns`
- `dataframe`
- `else`
- `exists`
- `f`
- `FileNotFoundError`
- `files`
- `first`
- `found`
- `glob`
- `head`
- `info`
- `ledger`
- `Load`
- `load_data`
- `load_silver`
- `Loaded`
- `Multiple`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `parquet_candidates`
- `preferred_silver_path`
- `silver_data_path`
- `silver_eda_df`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Load Silver dataframe`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `preferred_silver_path = SILVER_TRAIN_DATA_PATH / SILVER_TRAIN_DATA_FILE_NAME`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if preferred_silver_path.exists(): silver_data_path = preferred_silver_path`: Controls validation, iteration, file handling, or error handling for this step.
- `else: parquet_candidates = sorted(SILVER_TRAIN_DATA_PATH.glob("*.parquet")) if not parquet_candidates: raise FileNotFoundError(f"No Silver parquet files found in {SILVER_TRAIN_DATA`: Writes a logger message for traceability during notebook execution.
- `silver_eda_df = load_data(silver_data_path.parent, silver_data_path.name)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `logger.info("Loaded Silver dataframe: %s \| shape=%s", silver_data_path, silver_eda_df.shape)`: Writes a logger message for traceability during notebook execution.
- `ledger.add( kind="step", step="load_silver", message="Loaded Silver dataframe.", data={ "silver_path": str(silver_data_path), "shape": list(silver_eda_df.shape), "column_count": in`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Loaded:", silver_data_path)`: Displays a notebook-facing result for inspection.
- `print("Shape:", silver_eda_df.shape)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `display`
- `exists`
- `FileNotFoundError`
- `glob`
- `head`
- `info`
- `load_data`
- `sorted`
- `warning`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Load Silver dataframe` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `preferred_silver_path = SILVER_TRAIN_DATA_PATH / SILVER_TRAIN_DATA_FILE_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if preferred_silver_path.exists(): silver_data_path = preferred_silver_path` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: parquet_candidates = sorted(SILVER_TRAIN_DATA_PATH.glob("*.parquet")) if not parquet_candidates: raise FileNotFoundError(f"No Silver parquet files found in {SILVER_TRAIN_DATA` | Writes a logger message for traceability during notebook execution. |
| `silver_eda_df = load_data(silver_data_path.parent, silver_data_path.name)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Loaded Silver dataframe: %s \| shape=%s", silver_data_path, silver_eda_df.shape)` | Writes a logger message for traceability during notebook execution. |
| `ledger.add( kind="step", step="load_silver", message="Loaded Silver dataframe.", data={ "silver_path": str(silver_data_path), "shape": list(silver_eda_df.shape), "column_count": in` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Loaded:", silver_data_path)` | Displays a notebook-facing result for inspection. |
| `print("Shape:", silver_eda_df.shape)` | Displays a notebook-facing result for inspection. |
| `display(silver_eda_df.head(3))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 16 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__silver__feature_registry`
- `__silver__truth__`
- `add`
- `artifact`
- `astype`
- `bool`
- `column`
- `Could`
- `dataset`
- `dir`
- `directories`
- `dirs`
- `dropna`
- `exist_ok`
- `extract_truth_hash`
- `f`
- `Feature`
- `feature_set`
- `get`
- `get_artifact_path_from_truth`

### Outputs

- `CANONICAL_INFO`
- `column_name`
- `data`
- `dataframe`
- `dataset_name`
- `DATASET_NAME`
- `feature_registry_dir`
- `FEATURE_REGISTRY_FILE_NAME`
- `FEATURE_REGISTRY_PATH`
- `FEATURE_SET_INFO`
- `kind`
- `LABEL_SOURCE_COLUMN`
- `LABEL_SOURCE_TYPE`
- `logger`
- `message`
- `NEEDS_ONE_HOT_ENCODING`
- `ONE_HOT_ENCODING_COLUMNS`
- `parent_layer_name`
- `PIPELINE_MODE`
- `PIPELINE_MODE_FROM_TRUTH`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Resolve truth / dataset identity / artifact dirs`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `SILVER_TRUTH_HASH = extract_truth_hash(silver_eda_df)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if SILVER_TRUTH_HASH is None: raise ValueError("Could not resolve meta__truth_hash from Silver dataframe.")`: Controls validation, iteration, file handling, or error handling for this step.
- `SILVER_DATASET_NAME = ( silver_eda_df["meta__dataset"] .dropna() .astype("string") .str.strip()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `SILVER_DATASET_NAME = SILVER_DATASET_NAME[SILVER_DATASET_NAME != ""]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if len(SILVER_DATASET_NAME) == 0: raise ValueError("Silver dataframe is missing usable meta__dataset values.")`: Controls validation, iteration, file handling, or error handling for this step.
- `SILVER_DATASET_NAME = str(SILVER_DATASET_NAME.iloc[0]).strip()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `silver_truth = load_parent_truth_record_from_dataframe( dataframe=silver_eda_df, truth_dir=TRUTHS_PATH, parent_layer_name="silver", dataset_name=SILVER_DATASET_NAME, column_name="m`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `astype`
- `bool`
- `dropna`
- `extract_truth_hash`
- `get`
- `get_artifact_path_from_truth`
- `get_dataset_name_from_truth`
- `get_parent_truth_hash`
- `get_pipeline_mode_from_truth`
- `get_truth_hash`
- `get_truth_value`
- `info`
- `load_parent_truth_record_from_dataframe`
- `mkdir`
- `Path`
- `strip`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Resolve truth / dataset identity / artifact dirs` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_TRUTH_HASH = extract_truth_hash(silver_eda_df)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if SILVER_TRUTH_HASH is None: raise ValueError("Could not resolve meta__truth_hash from Silver dataframe.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `SILVER_DATASET_NAME = ( silver_eda_df["meta__dataset"] .dropna() .astype("string") .str.strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_DATASET_NAME = SILVER_DATASET_NAME[SILVER_DATASET_NAME != ""]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(SILVER_DATASET_NAME) == 0: raise ValueError("Silver dataframe is missing usable meta__dataset values.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `SILVER_DATASET_NAME = str(SILVER_DATASET_NAME.iloc[0]).strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `silver_truth = load_parent_truth_record_from_dataframe( dataframe=silver_eda_df, truth_dir=TRUTHS_PATH, parent_layer_name="silver", dataset_name=SILVER_DATASET_NAME, column_name="m` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DATASET_NAME = get_dataset_name_from_truth(silver_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_TRUTH_HASH = get_truth_hash(silver_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_PARENT_TRUTH_HASH = get_parent_truth_hash(silver_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PIPELINE_MODE_FROM_TRUTH = get_pipeline_mode_from_truth(silver_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if PIPELINE_MODE_FROM_TRUTH is not None: PIPELINE_MODE = PIPELINE_MODE_FROM_TRUTH` | Controls validation, iteration, file handling, or error handling for this step. |
| `SILVER_TRUTH_PATH = ( TRUTHS_PATH / "silver" / f"{DATASET_NAME}__silver__truth__{SILVER_TRUTH_HASH}.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `LABEL_SOURCE_COLUMN = get_truth_value( silver_truth, "runtime_facts", "label_resolution", "label_source_column", required=False,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `LABEL_SOURCE_TYPE = get_truth_value( silver_truth, "runtime_facts", "label_resolution", "label_source_type", required=False,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CANONICAL_INFO = silver_truth.get("runtime_facts", {}).get("canonical_info", {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FEATURE_SET_INFO = silver_truth.get("runtime_facts", {}).get("feature_set", {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `QUALITY_INFO = silver_truth.get("runtime_facts", {}).get("quality_info", {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `NEEDS_ONE_HOT_ENCODING = bool( silver_truth.get("needs_one_hot_encoding", False)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ONE_HOT_ENCODING_COLUMNS = list( silver_truth.get("one_hot_encoding_columns", [])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_registry_dir = get_artifact_path_from_truth( silver_truth, "feature_registry_dir",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FEATURE_REGISTRY_FILE_NAME = f"{DATASET_NAME}__silver__feature_registry.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FEATURE_REGISTRY_PATH = Path(feature_registry_dir) / "registry" / FEATURE_REGISTRY_FILE_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#SILVER_EDA_ARTIFACT_DIR = SILVER_SUBSET_ARTIFACTS_ROOT / DATASET_NAME` | Documents the purpose or boundary of the surrounding notebook step. |
| `#SILVER_EDA_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_SUBSET_TRUTH_DIR = TRUTHS_PATH / "silver"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_SUBSET_TRUTH_DIR.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("Loaded Silver truth: %s", SILVER_TRUTH_PATH)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Resolved dataset name from Silver truth: %s", DATASET_NAME)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Resolved label source column from Silver truth: %s", LABEL_SOURCE_COLUMN)` | Writes a logger message for traceability during notebook execution. |
| `ledger.add( kind="step", step="resolve_truth", message="Resolved Silver truth, dataset identity, and subset artifact directories.", data={ "dataset_name": DATASET_NAME, "silver_tru` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Loaded Silver truth:", SILVER_TRUTH_PATH)` | Displays a notebook-facing result for inspection. |
| `print("Silver truth hash:", SILVER_TRUTH_HASH)` | Displays a notebook-facing result for inspection. |
| `print("Resolved dataset name:", DATASET_NAME)` | Displays a notebook-facing result for inspection. |
| `print("Resolved label source column:", LABEL_SOURCE_COLUMN)` | Displays a notebook-facing result for inspection. |
| `print("Feature registry path:", FEATURE_REGISTRY_PATH)` | Displays a notebook-facing result for inspection. |
| `print("Subset artifact dir:", SILVER_EDA_ARTIFACT_DIR)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 17 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__name__`
- `a`
- `actually`
- `add`
- `After`
- `Any`
- `be`
- `before`
- `building`
- `but`
- `cast`
- `col`
- `column`
- `columns`
- `config`
- `count`
- `dataframe`
- `dictionary`
- `empty`
- `exist`

### Outputs

- `CONFIG_SENSOR_COLUMNS`
- `consequence`
- `data`
- `FEATURE_COLUMNS`
- `feature_columns_raw`
- `feature_registry`
- `feature_registry_raw`
- `kind`
- `logger`
- `message`
- `meta_column_set`
- `meta_columns`
- `missing_feature_columns`
- `step`
- `why`

### Key Operations

- `if FEATURE_REGISTRY_PATH is None: raise ValueError( "FEATURE_REGISTRY_PATH was not resolved from Silver truth before loading the feature registry." )`: Controls validation, iteration, file handling, or error handling for this step.
- `feature_registry_raw = load_json( FEATURE_REGISTRY_PATH.parent, FEATURE_REGISTRY_PATH.name,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if feature_registry_raw is None: raise ValueError( f"Feature registry JSON loaded as None: {FEATURE_REGISTRY_PATH}" )`: Controls validation, iteration, file handling, or error handling for this step.
- `if not isinstance(feature_registry_raw, dict): raise TypeError( "Feature registry must be a dictionary, " f"got {type(feature_registry_raw).__name__}: {FEATURE_REGISTRY_PATH}" )`: Controls validation, iteration, file handling, or error handling for this step.
- `feature_registry = cast(Dict[str, Any], feature_registry_raw)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `feature_columns_raw = feature_registry.get("feature_columns", [])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if feature_columns_raw is None: feature_columns_raw = []`: Controls validation, iteration, file handling, or error handling for this step.
- `if not isinstance(feature_columns_raw, list): raise TypeError( "Feature registry key 'feature_columns' must be a list, " f"got {type(feature_columns_raw).__name__}: {FEATURE_REGIST`: Controls validation, iteration, file handling, or error handling for this step.
- `FEATURE_COLUMNS = [ str(col).strip() for col in feature_columns_raw if str(col).strip() != ""`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if len(FEATURE_COLUMNS) == 0: raise ValueError( f"Feature registry was loaded but feature_columns is empty: {FEATURE_REGISTRY_PATH}" )`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `add`
- `cast`
- `get`
- `identify_meta_columns`
- `info`
- `isinstance`
- `load_json`
- `log`
- `strip`
- `type`
- `TypeError`
- `ValueError`
- `warning`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `if FEATURE_REGISTRY_PATH is None: raise ValueError( "FEATURE_REGISTRY_PATH was not resolved from Silver truth before loading the feature registry." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `feature_registry_raw = load_json( FEATURE_REGISTRY_PATH.parent, FEATURE_REGISTRY_PATH.name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if feature_registry_raw is None: raise ValueError( f"Feature registry JSON loaded as None: {FEATURE_REGISTRY_PATH}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `if not isinstance(feature_registry_raw, dict): raise TypeError( "Feature registry must be a dictionary, " f"got {type(feature_registry_raw).__name__}: {FEATURE_REGISTRY_PATH}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `feature_registry = cast(Dict[str, Any], feature_registry_raw)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `feature_columns_raw = feature_registry.get("feature_columns", [])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if feature_columns_raw is None: feature_columns_raw = []` | Controls validation, iteration, file handling, or error handling for this step. |
| `if not isinstance(feature_columns_raw, list): raise TypeError( "Feature registry key 'feature_columns' must be a list, " f"got {type(feature_columns_raw).__name__}: {FEATURE_REGIST` | Controls validation, iteration, file handling, or error handling for this step. |
| `FEATURE_COLUMNS = [ str(col).strip() for col in feature_columns_raw if str(col).strip() != ""` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if len(FEATURE_COLUMNS) == 0: raise ValueError( f"Feature registry was loaded but feature_columns is empty: {FEATURE_REGISTRY_PATH}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `# Keep only features that actually exist in the dataframe` | Documents the purpose or boundary of the surrounding notebook step. |
| `missing_feature_columns = [ col for col in FEATURE_COLUMNS if col not in silver_eda_df.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_feature_columns: logger.warning( "Some feature registry columns were not found in the dataframe and will be skipped: %s", missing_feature_columns[:20], )` | Writes a logger message for traceability during notebook execution. |
| `FEATURE_COLUMNS = [ col for col in FEATURE_COLUMNS if col in silver_eda_df.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if len(FEATURE_COLUMNS) == 0: raise ValueError( "After intersecting the feature registry with dataframe columns, no usable feature columns remained." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `CONFIG_SENSOR_COLUMNS = FEATURE_COLUMNS` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `meta_columns = identify_meta_columns(silver_eda_df)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `meta_column_set = set(meta_columns)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Loaded Silver Feature Registry: %s", FEATURE_REGISTRY_PATH)` | Writes a logger message for traceability during notebook execution. |
| `ledger.add( kind="step", step="load_silver_feature_registry", message="Loaded Silver Feature Registry JSON file using the path resolved from Silver truth.", why="Silver subset buil` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# wandb_run.log({"feature_registry_keys": int(len(feature_registry))})` | Documents the purpose or boundary of the surrounding notebook step. |
| `print("Loaded Silver Feature Registry:", FEATURE_REGISTRY_PATH)` | Displays a notebook-facing result for inspection. |
| `print("Feature registry keys:", int(len(feature_registry)))` | Displays a notebook-facing result for inspection. |
| `print("Feature column count:", int(len(FEATURE_COLUMNS)))` | Displays a notebook-facing result for inspection. |
| `print("First 10 feature columns:", FEATURE_COLUMNS[:10])` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 18 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `abnormal`
- `add`
- `Any`
- `astype`
- `break`
- `broken`
- `Built`
- `c`
- `candidate_col`
- `column`
- `columns`
- `cooldown`
- `cooling`
- `Could`
- `def`
- `else`
- `Expected`
- `failed`
- `failure`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `normalize_machine_state`
- `STATE_COL_SOURCE`
- `STATE_COL_SOURCE_RAW`
- `step`
- `text`

### Key Operations

- `STATE_COL_SOURCE_RAW = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `STATE_COL_SOURCE = "machine_status__synthetic"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `for candidate_col in [ "machine_status__synthetic", "machine_status", "status", "state",`: Controls validation, iteration, file handling, or error handling for this step.
- `]: if candidate_col in silver_eda_df.columns: STATE_COL_SOURCE_RAW = candidate_col break`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if STATE_COL_SOURCE_RAW is None: raise KeyError( "Could not resolve a source state column. Expected one of: " "machine_status__synthetic, machine_status, status, state" )`: Controls validation, iteration, file handling, or error handling for this step.
- `def normalize_machine_state(value: Any) -> str: text = str(value).strip().lower() if text in {"normal", "norm", "ok"}: return "normal" if text in {"broken", "abnormal", "fault", "f`: Defines notebook-local logic used later in the notebook.
- `if STATE_COL_SOURCE_RAW == STATE_COL_SOURCE: silver_eda_df[STATE_COL_SOURCE] = ( silver_eda_df[STATE_COL_SOURCE] .astype("string") .str.strip() .str.lower() )`: Controls validation, iteration, file handling, or error handling for this step.
- `else: silver_eda_df[STATE_COL_SOURCE] = silver_eda_df[STATE_COL_SOURCE_RAW].map(normalize_machine_state)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger.info( "Resolved source-state column: raw=%s -> normalized=%s", STATE_COL_SOURCE_RAW, STATE_COL_SOURCE,`: Writes a logger message for traceability during notebook execution.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="normalize_source_state", message="Built normalized source-state column.", data={ "source_state_raw_column": STATE_COL_SOURCE_RAW, "source_state_norma`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `astype`
- `display`
- `head`
- `info`
- `KeyError`
- `lower`
- `map`
- `normalize_machine_state`
- `strip`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `STATE_COL_SOURCE_RAW = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STATE_COL_SOURCE = "machine_status__synthetic"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for candidate_col in [ "machine_status__synthetic", "machine_status", "status", "state",` | Controls validation, iteration, file handling, or error handling for this step. |
| `]: if candidate_col in silver_eda_df.columns: STATE_COL_SOURCE_RAW = candidate_col break` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if STATE_COL_SOURCE_RAW is None: raise KeyError( "Could not resolve a source state column. Expected one of: " "machine_status__synthetic, machine_status, status, state" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `def normalize_machine_state(value: Any) -> str: text = str(value).strip().lower() if text in {"normal", "norm", "ok"}: return "normal" if text in {"broken", "abnormal", "fault", "f` | Defines notebook-local logic used later in the notebook. |
| `if STATE_COL_SOURCE_RAW == STATE_COL_SOURCE: silver_eda_df[STATE_COL_SOURCE] = ( silver_eda_df[STATE_COL_SOURCE] .astype("string") .str.strip() .str.lower() )` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: silver_eda_df[STATE_COL_SOURCE] = silver_eda_df[STATE_COL_SOURCE_RAW].map(normalize_machine_state)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info( "Resolved source-state column: raw=%s -> normalized=%s", STATE_COL_SOURCE_RAW, STATE_COL_SOURCE,` | Writes a logger message for traceability during notebook execution. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="normalize_source_state", message="Built normalized source-state column.", data={ "source_state_raw_column": STATE_COL_SOURCE_RAW, "source_state_norma` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display( silver_eda_df[ [c for c in [STATE_COL_SOURCE_RAW, STATE_COL_SOURCE] if c in silver_eda_df.columns] ].head(10)` | Displays a notebook-facing result for inspection. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 19 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `Built`
- `dropna`
- `ledger`
- `name`
- `orient`
- `quick`
- `Quick`
- `records`
- `rename_axis`
- `reset_index`
- `row_count`
- `row_pct`
- `silver_eda_df`
- `source`
- `source_state_counts`
- `source_state_summary`
- `state`
- `STATE_COL_SOURCE`
- `summary`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `source_state_counts_df`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Quick source-state summary`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `source_state_counts_df = ( silver_eda_df[STATE_COL_SOURCE] .value_counts(dropna=False) .rename_axis(STATE_COL_SOURCE) .reset_index(name="row_count")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `source_state_counts_df["row_pct"] = ( source_state_counts_df["row_count"] / len(silver_eda_df)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="source_state_summary", message="Built quick source-state summary.", data={ "source_state_counts": source_state_counts_df.to_dict(orient="records"), }`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(source_state_counts_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `display`
- `rename_axis`
- `reset_index`
- `to_dict`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Quick source-state summary` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `source_state_counts_df = ( silver_eda_df[STATE_COL_SOURCE] .value_counts(dropna=False) .rename_axis(STATE_COL_SOURCE) .reset_index(name="row_count")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `source_state_counts_df["row_pct"] = ( source_state_counts_df["row_count"] / len(silver_eda_df)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="source_state_summary", message="Built quick source-state summary.", data={ "source_state_counts": source_state_counts_df.to_dict(orient="records"), }` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(source_state_counts_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 20 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `above`
- `aligned`
- `all`
- `artifacts`
- `available`
- `avoids`
- `back`
- `baseline`
- `builder`
- `casing`
- `column`
- `columns`
- `configuration`
- `created`
- `DataFrame`
- `dataframe`
- `downstream`
- `elif`
- `else`
- `fall`

### Outputs

- `config_sensor_columns`
- `config_summary_df`
- `DELTA_DEVIATION_THRESHOLD`
- `EXCLUDE_SENSOR_COUNT`
- `FINAL_METHOD`
- `KEEP_WINDOW_FRAC`
- `LABEL_COLUMN`
- `MIN_WINDOW_ROWS`
- `NORMAL_VALUES`
- `raw_config_sensor_columns`
- `SENSOR_COLUMNS`
- `SUSPECT_SENSOR_COUNT`
- `TRIM_FRAC`
- `VALUE_DEVIATION_THRESHOLD`
- `WINDOWS_PER_EPISODE`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Subset-builder configuration`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Use the normalized source-state column created above. This avoids mixed`: Documents the purpose or boundary of the surrounding notebook step.
- `# casing such as NORMAL / normal and keeps all downstream profiling logic`: Documents the purpose or boundary of the surrounding notebook step.
- `# using the same state values.`: Documents the purpose or boundary of the surrounding notebook step.
- `LABEL_COLUMN = STATE_COL_SOURCE`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `NORMAL_VALUES = ["normal"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Prefer the feature registry columns when available, then fall back to`: Documents the purpose or boundary of the surrounding notebook step.
- `# dataframe sensor_* columns. This keeps the subset builder aligned with`: Documents the purpose or boundary of the surrounding notebook step.
- `# the Silver feature lineage.`: Documents the purpose or boundary of the surrounding notebook step.
- `raw_config_sensor_columns = globals().get("CONFIG_SENSOR_COLUMNS", [])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `DataFrame`
- `display`
- `get`
- `globals`
- `isinstance`
- `startswith`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Subset-builder configuration` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Use the normalized source-state column created above. This avoids mixed` | Documents the purpose or boundary of the surrounding notebook step. |
| `# casing such as NORMAL / normal and keeps all downstream profiling logic` | Documents the purpose or boundary of the surrounding notebook step. |
| `# using the same state values.` | Documents the purpose or boundary of the surrounding notebook step. |
| `LABEL_COLUMN = STATE_COL_SOURCE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `NORMAL_VALUES = ["normal"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Prefer the feature registry columns when available, then fall back to` | Documents the purpose or boundary of the surrounding notebook step. |
| `# dataframe sensor_* columns. This keeps the subset builder aligned with` | Documents the purpose or boundary of the surrounding notebook step. |
| `# the Silver feature lineage.` | Documents the purpose or boundary of the surrounding notebook step. |
| `raw_config_sensor_columns = globals().get("CONFIG_SENSOR_COLUMNS", [])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if isinstance(raw_config_sensor_columns, list): config_sensor_columns = [ str(column) for column in raw_config_sensor_columns ]` | Controls validation, iteration, file handling, or error handling for this step. |
| `elif isinstance(raw_config_sensor_columns, tuple): config_sensor_columns = [ str(column) for column in raw_config_sensor_columns ]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `else: config_sensor_columns = []` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if len(config_sensor_columns) > 0: SENSOR_COLUMNS = [ column for column in config_sensor_columns if column.startswith("sensor_") and column in silver_eda_df.columns ]` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: SENSOR_COLUMNS = [ str(column) for column in silver_eda_df.columns if str(column).startswith("sensor_") ]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if len(SENSOR_COLUMNS) == 0: raise ValueError("No sensor_* columns were resolved for profiling.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `# Window-baseline settings` | Documents the purpose or boundary of the surrounding notebook step. |
| `TRIM_FRAC = 0.10` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WINDOWS_PER_EPISODE = 5` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `KEEP_WINDOW_FRAC = 0.80` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MIN_WINDOW_ROWS = 500` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Row scoring settings` | Documents the purpose or boundary of the surrounding notebook step. |
| `VALUE_DEVIATION_THRESHOLD = 8.0` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DELTA_DEVIATION_THRESHOLD = 5.0` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SUSPECT_SENSOR_COUNT = 8` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `EXCLUDE_SENSOR_COUNT = 21` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Metadata label used in summary/truth artifacts` | Documents the purpose or boundary of the surrounding notebook step. |
| `FINAL_METHOD = "trimmed_normal_episode_sensor_profile"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `config_summary_df = pd.DataFrame( [ {"setting": "label_column", "value": LABEL_COLUMN}, {"setting": "normal_values", "value": str(NORMAL_VALUES)}, {"setting": "sensor_count", "valu` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(config_summary_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 21 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__is_normal_candidate`
- `__normal_episode_id`
- `__status_group`
- `A`
- `assign`
- `changes`
- `copy`
- `cumsum`
- `dataframe`
- `def`
- `episode`
- `fill_value`
- `group`
- `IDs`
- `isin`
- `label_column`
- `nan`
- `ne`
- `new`
- `non`

### Outputs

- `assign_normal_episodes`
- `normal_mask`
- `status_change`
- `working_df`

### Key Operations

- `def assign_normal_episodes( dataframe, label_column, normal_values,`: Defines notebook-local logic used later in the notebook.
- `): working_df = dataframe.copy() normal_mask = working_df[label_column].isin(normal_values) # A new group starts whenever normal/non-normal status changes status_change = normal_ma`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `assign_normal_episodes`
- `copy`
- `cumsum`
- `isin`
- `ne`
- `shift`
- `where`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def assign_normal_episodes( dataframe, label_column, normal_values,` | Defines notebook-local logic used later in the notebook. |
| `): working_df = dataframe.copy() normal_mask = working_df[label_column].isin(normal_values) # A new group starts whenever normal/non-normal status changes status_change = normal_ma` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 22 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__keep_after_episode_trim`
- `__normal_episode_id`
- `continue`
- `copy`
- `dataframe`
- `def`
- `dropna`
- `episode_id`
- `eq`
- `index`
- `loc`
- `unique`

### Outputs

- `episode_index`
- `episode_mask`
- `episode_row_count`
- `kept_index`
- `min_rows_after_trim`
- `normal_episode_ids`
- `trim_count`
- `trim_frac`
- `trim_normal_episode_edges`
- `working_df`

### Key Operations

- `def trim_normal_episode_edges( dataframe, trim_frac=0.10, min_rows_after_trim=1000,`: Defines notebook-local logic used later in the notebook.
- `): working_df = dataframe.copy() working_df["__keep_after_episode_trim"] = False normal_episode_ids = ( working_df["__normal_episode_id"] .dropna() .unique() ) for episode_id in no`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `copy`
- `dropna`
- `eq`
- `trim_normal_episode_edges`
- `unique`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def trim_normal_episode_edges( dataframe, trim_frac=0.10, min_rows_after_trim=1000,` | Defines notebook-local logic used later in the notebook. |
| `): working_df = dataframe.copy() working_df["__keep_after_episode_trim"] = False normal_episode_ids = ( working_df["__normal_episode_id"] .dropna() .unique() ) for episode_id in no` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 23 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__keep_after_episode_trim`
- `__normal_episode_id`
- `__window_`
- `__window_id`
- `array_split`
- `continue`
- `copy`
- `dataframe`
- `def`
- `enumerate`
- `episode_`
- `episode_df`
- `episode_id`
- `f`
- `groupby`
- `index`
- `loc`
- `local_window_id`
- `nan`
- `start`

### Outputs

- `create_episode_windows`
- `episode_indices`
- `kept_df`
- `min_window_rows`
- `window_counter`
- `window_label`
- `window_splits`
- `windows_per_episode`
- `working_df`

### Key Operations

- `def create_episode_windows( dataframe, windows_per_episode=5, min_window_rows=500,`: Defines notebook-local logic used later in the notebook.
- `): working_df = dataframe.copy() working_df["__window_id"] = np.nan kept_df = working_df[working_df["__keep_after_episode_trim"]].copy() window_counter = 0 for episode_id, episode_`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `array_split`
- `copy`
- `create_episode_windows`
- `enumerate`
- `groupby`
- `to_numpy`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def create_episode_windows( dataframe, windows_per_episode=5, min_window_rows=500,` | Defines notebook-local logic used later in the notebook. |
| `): working_df = dataframe.copy() working_df["__window_id"] = np.nan kept_df = working_df[working_df["__keep_after_episode_trim"]].copy() window_counter = 0 for episode_id, episode_` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 24 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__window_id`
- `abs`
- `append`
- `baseline`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `delta`
- `delta_abs_median`
- `delta_abs_q95`
- `delta_iqr`
- `delta_mean`
- `delta_median`
- `delta_std`
- `diff`
- `dropna`
- `group_df`
- `groupby`
- `iqr`

### Outputs

- `calculate_window_sensor_stats`
- `delta_q25`
- `delta_q75`
- `deltas`
- `q25`
- `q75`
- `sensor_deltas`
- `sensor_values`
- `values`
- `window_df`
- `window_stats`

### Key Operations

- `def calculate_window_sensor_stats( dataframe, sensor_columns,`: Defines notebook-local logic used later in the notebook.
- `): window_stats = [] window_df = dataframe[dataframe["__window_id"].notna()].copy() for window_id, group_df in window_df.groupby("__window_id"): sensor_values = group_df[sensor_col`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `abs`
- `append`
- `calculate_window_sensor_stats`
- `copy`
- `DataFrame`
- `diff`
- `dropna`
- `groupby`
- `isna`
- `mean`
- `median`
- `notna`
- `quantile`
- `std`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def calculate_window_sensor_stats( dataframe, sensor_columns,` | Defines notebook-local logic used later in the notebook. |
| `): window_stats = [] window_df = dataframe[dataframe["__window_id"].notna()].copy() for window_id, group_df in window_df.groupby("__window_id"): sensor_values = group_df[sensor_col` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 25 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `agg`
- `as_index`
- `count`
- `def`
- `delta_abs_q95`
- `delta_rank_pct`
- `exclude`
- `groupby`
- `iqr`
- `iqr_rank_pct`
- `keep`
- `mean`
- `median`
- `missing_pct`
- `missing_rank_pct`
- `pct`
- `quantile`
- `rank`
- `row_count`
- `sensor`

### Outputs

- `cutoff_score`
- `keep_window_frac`
- `score_and_filter_windows`
- `window_median_delta_abs_q95`
- `window_median_iqr`
- `window_missing_pct`
- `window_quality_df`
- `window_row_count`
- `window_sensor_count`

### Key Operations

- `def score_and_filter_windows( window_sensor_stats_df, keep_window_frac=0.80,`: Defines notebook-local logic used later in the notebook.
- `): window_quality_df = ( window_sensor_stats_df .groupby("window_id", as_index=False) .agg( window_row_count=("row_count", "median"), window_sensor_count=("sensor", "count"), windo`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `agg`
- `groupby`
- `quantile`
- `rank`
- `score_and_filter_windows`
- `where`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def score_and_filter_windows( window_sensor_stats_df, keep_window_frac=0.80,` | Defines notebook-local logic used later in the notebook. |
| `): window_quality_df = ( window_sensor_stats_df .groupby("window_id", as_index=False) .agg( window_row_count=("row_count", "median"), window_sensor_count=("sensor", "count"), windo` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 26 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `agg`
- `as_index`
- `copy`
- `count`
- `def`
- `delta_abs_median`
- `delta_abs_q95`
- `delta_iqr`
- `delta_median`
- `eq`
- `groupby`
- `iqr`
- `isin`
- `keep`
- `loc`
- `mean`
- `median`
- `missing_pct`
- `q05`
- `q25`

### Outputs

- `baseline_delta_abs_median`
- `baseline_delta_abs_q95`
- `baseline_delta_iqr`
- `baseline_delta_median`
- `baseline_iqr`
- `baseline_mean`
- `baseline_median`
- `baseline_missing_pct`
- `baseline_q05`
- `baseline_q25`
- `baseline_q75`
- `baseline_q95`
- `baseline_row_count`
- `baseline_std`
- `baseline_window_count`
- `build_final_sensor_baseline`
- `final_baseline_df`
- `kept_window_stats_df`
- `kept_windows`

### Key Operations

- `def build_final_sensor_baseline( window_sensor_stats_df, window_quality_df,`: Defines notebook-local logic used later in the notebook.
- `): kept_windows = window_quality_df.loc[ window_quality_df["window_quality_class"].eq("keep"), "window_id" ] kept_window_stats_df = window_sensor_stats_df[ window_sensor_stats_df["`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `agg`
- `build_final_sensor_baseline`
- `copy`
- `eq`
- `groupby`
- `isin`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_final_sensor_baseline( window_sensor_stats_df, window_quality_df,` | Defines notebook-local logic used later in the notebook. |
| `): kept_windows = window_quality_df.loc[ window_quality_df["window_quality_class"].eq("keep"), "window_id" ] kept_window_stats_df = window_sensor_stats_df[ window_sensor_stats_df["` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 27 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `assign_normal_episodes`
- `build_final_sensor_baseline`
- `calculate_window_sensor_stats`
- `create_episode_windows`
- `final_sensor_baseline_df`
- `kept_window_stats_df`
- `score_and_filter_windows`
- `silver_eda_df`
- `trim_normal_episode_edges`

### Outputs

- `keep_window_frac`
- `label_column`
- `min_rows_after_trim`
- `min_window_rows`
- `normal_profile_df`
- `normal_values`
- `sensor_columns`
- `trim_frac`
- `window_quality_df`
- `window_sensor_stats_df`
- `windows_per_episode`

### Key Operations

- `normal_profile_df = assign_normal_episodes( silver_eda_df, label_column=LABEL_COLUMN, normal_values=NORMAL_VALUES,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `normal_profile_df = trim_normal_episode_edges( normal_profile_df, trim_frac=TRIM_FRAC, min_rows_after_trim=1000,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `normal_profile_df = create_episode_windows( normal_profile_df, windows_per_episode=WINDOWS_PER_EPISODE, min_window_rows=MIN_WINDOW_ROWS,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `window_sensor_stats_df = calculate_window_sensor_stats( normal_profile_df, sensor_columns=SENSOR_COLUMNS,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `window_quality_df = score_and_filter_windows( window_sensor_stats_df, keep_window_frac=KEEP_WINDOW_FRAC,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `final_sensor_baseline_df, kept_window_stats_df = build_final_sensor_baseline( window_sensor_stats_df, window_quality_df,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `assign_normal_episodes`
- `build_final_sensor_baseline`
- `calculate_window_sensor_stats`
- `create_episode_windows`
- `score_and_filter_windows`
- `trim_normal_episode_edges`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `normal_profile_df = assign_normal_episodes( silver_eda_df, label_column=LABEL_COLUMN, normal_values=NORMAL_VALUES,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `normal_profile_df = trim_normal_episode_edges( normal_profile_df, trim_frac=TRIM_FRAC, min_rows_after_trim=1000,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `normal_profile_df = create_episode_windows( normal_profile_df, windows_per_episode=WINDOWS_PER_EPISODE, min_window_rows=MIN_WINDOW_ROWS,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `window_sensor_stats_df = calculate_window_sensor_stats( normal_profile_df, sensor_columns=SENSOR_COLUMNS,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `window_quality_df = score_and_filter_windows( window_sensor_stats_df, keep_window_frac=KEEP_WINDOW_FRAC,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `final_sensor_baseline_df, kept_window_stats_df = build_final_sensor_baseline( window_sensor_stats_df, window_quality_df,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 28 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `head`
- `sort_values`
- `window_quality_df`
- `window_quality_score`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`
- `head`
- `sort_values`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 29 — Review window quality distribution

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `value_counts`
- `window_quality_class`
- `window_quality_df`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 30 — Preview the clean-normal sensor baseline

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `final_sensor_baseline_df`
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

## Code Cell 31 — Confirm baseline profile shape

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `final_sensor_baseline_df`
- `shape`

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

## Code Cell 32 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `artifact`
- `artifact_type`
- `artifacts`
- `baseline`
- `baseline_config`
- `baseline_row_count`
- `baseline_rows_per_sensor_median`
- `baseline_window_count`
- `baseline_windows_per_sensor_median`
- `created_at_utc`
- `csv`
- `CSV`
- `datetime`
- `default`
- `delta_deviation_threshold`
- `DELTA_DEVIATION_THRESHOLD`
- `dump`
- `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`
- `encoding`

### Outputs

- `BASELINE_EXPORT_DIR`
- `baseline_json_path`
- `baseline_table_path`
- `data`
- `json_ready_baseline_df`
- `kind`
- `logger`
- `message`
- `sensor_baseline_json`
- `sensor_profiles_dict`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Save sensor baseline profiles to JSON / CSV`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `BASELINE_EXPORT_DIR = SENSOR_PROFILE_DIR`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `BASELINE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `baseline_table_path = BASELINE_EXPORT_DIR / "silver_sensor_baseline_profiles.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `final_sensor_baseline_df.to_csv(baseline_table_path, index=False)`: Writes an artifact or output used for review or downstream notebooks.
- `json_ready_baseline_df = ( final_sensor_baseline_df .replace([np.inf, -np.inf], np.nan) .where(pd.notna(final_sensor_baseline_df), None)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `sensor_profiles_dict = ( json_ready_baseline_df .set_index("sensor") .to_dict(orient="index")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `sensor_baseline_json = { "artifact_type": "sensor_baseline_profile", "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_notebook": "EDA_Notebook_Pump_Silver_02a_EDA_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `display`
- `dump`
- `head`
- `isoformat`
- `median`
- `mkdir`
- `notna`
- `now`
- `nunique`
- `open`
- `replace`
- `set_index`
- `to_csv`
- `to_dict`
- `where`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Save sensor baseline profiles to JSON / CSV` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `BASELINE_EXPORT_DIR = SENSOR_PROFILE_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BASELINE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_table_path = BASELINE_EXPORT_DIR / "silver_sensor_baseline_profiles.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `final_sensor_baseline_df.to_csv(baseline_table_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `json_ready_baseline_df = ( final_sensor_baseline_df .replace([np.inf, -np.inf], np.nan) .where(pd.notna(final_sensor_baseline_df), None)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `sensor_profiles_dict = ( json_ready_baseline_df .set_index("sensor") .to_dict(orient="index")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `sensor_baseline_json = { "artifact_type": "sensor_baseline_profile", "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_notebook": "EDA_Notebook_Pump_Silver_02a_EDA_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_json_path = BASELINE_EXPORT_DIR / "silver_sensor_baseline_profiles.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `with open(baseline_json_path, "w", encoding="utf-8") as file: json.dump(sensor_baseline_json, file, indent=2, default=str)` | Controls validation, iteration, file handling, or error handling for this step. |
| `ledger.add( kind="artifact", step="save_sensor_baseline_profiles", message="Saved per-sensor baseline profile JSON and CSV artifacts.", data={ "baseline_json_path": str(baseline_js` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Saved sensor profile JSON:", baseline_json_path)` | Displays a notebook-facing result for inspection. |
| `print("Saved sensor profile table:", baseline_table_path)` | Displays a notebook-facing result for inspection. |
| `display(final_sensor_baseline_df.head())` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 33 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__delta_abnormal_flag`
- `__profile_plot`
- `__value_abnormal_flag`
- `a`
- `abnormal`
- `abs`
- `actual`
- `against`
- `any`
- `astype`
- `ax1`
- `axhline`
- `band`
- `bbox_inches`
- `bool`
- `close`
- `column`
- `columns`
- `copy`
- `def`

### Outputs

- `alpha`
- `ax2`
- `axes`
- `baseline_delta_abs_q95`
- `baseline_df`
- `baseline_iqr`
- `baseline_median`
- `baseline_q05`
- `baseline_q95`
- `dataframe`
- `delta_flag`
- `delta_flag_column`
- `display_sensor_profile`
- `facecolors`
- `figsize`
- `fontsize`
- `index_column`
- `label`
- `linestyle`
- `linewidth`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Sensor profile display and plotting helpers`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `def display_sensor_profile( baseline_df, sensor_name,`: Defines notebook-local logic used later in the notebook.
- `): """ Display one sensor's learned normal profile as a readable metric table. The profile is one row per sensor in final_sensor_baseline_df. This function pivots that row vertical`: Displays a notebook-facing result for inspection.
- `def plot_sensor_profile_with_baseline( dataframe, baseline_df, sensor_name, *, index_column=None, quality_column="final_row_quality_class", value_flag_column=None, delta_flag_colum`: Defines notebook-local logic used later in the notebook.
- `): """ Plot one sensor against its learned normal profile. What it shows ------------- - The sensor's actual values over time / row order. - The learned normal median. - The learne`: Writes an artifact or output used for review or downstream notebooks.
- `def plot_all_sensor_profiles( dataframe, baseline_df, sensor_columns, *, index_column=None, quality_column="final_row_quality_class", max_points=20000, save_dir=None, show_plots=Tr`: Defines notebook-local logic used later in the notebook.
- `): """ Loop through every sensor and display/save each learned normal profile plot. """ plot_results = {} for sensor_name in sensor_columns: display_sensor_profile( baseline_df=bas`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `abs`
- `any`
- `astype`
- `axhline`
- `close`
- `copy`
- `diff`
- `display`
- `display_sensor_profile`
- `eq`
- `get_legend_handles_labels`
- `gt`
- `join`
- `legend`
- `lt`
- `mkdir`
- `Path`
- `plot`
- `plot_all_sensor_profiles`
- `plot_sensor_profile_with_baseline`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Sensor profile display and plotting helpers` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `def display_sensor_profile( baseline_df, sensor_name,` | Defines notebook-local logic used later in the notebook. |
| `): """ Display one sensor's learned normal profile as a readable metric table. The profile is one row per sensor in final_sensor_baseline_df. This function pivots that row vertical` | Displays a notebook-facing result for inspection. |
| `def plot_sensor_profile_with_baseline( dataframe, baseline_df, sensor_name, *, index_column=None, quality_column="final_row_quality_class", value_flag_column=None, delta_flag_colum` | Defines notebook-local logic used later in the notebook. |
| `): """ Plot one sensor against its learned normal profile. What it shows ------------- - The sensor's actual values over time / row order. - The learned normal median. - The learne` | Writes an artifact or output used for review or downstream notebooks. |
| `def plot_all_sensor_profiles( dataframe, baseline_df, sensor_columns, *, index_column=None, quality_column="final_row_quality_class", max_points=20000, save_dir=None, show_plots=Tr` | Defines notebook-local logic used later in the notebook. |
| `): """ Loop through every sensor and display/save each learned normal profile plot. """ plot_results = {} for sensor_name in sensor_columns: display_sensor_profile( baseline_df=bas` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: plot/image artifact.

## Code Cell 34 — Inspect an individual sensor profile

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `display_sensor_profile`
- `final_sensor_baseline_df`
- `sensor_04`

### Outputs

- `baseline_df`
- `sensor_name`

### Key Operations

- `display_sensor_profile( baseline_df=final_sensor_baseline_df, sensor_name="sensor_04",`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display_sensor_profile`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `display_sensor_profile( baseline_df=final_sensor_baseline_df, sensor_name="sensor_04",` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 35 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Active`
- `DataFrame`
- `DELTA_DEVIATION_THRESHOLD`
- `EXCLUDE_SENSOR_COUNT`
- `row`
- `scoring`
- `setting`
- `SUSPECT_SENSOR_COUNT`
- `thresholds`
- `value`
- `VALUE_DEVIATION_THRESHOLD`

### Outputs

- `scoring_config_df`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Active row-scoring thresholds`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `scoring_config_df = pd.DataFrame( [ {"setting": "VALUE_DEVIATION_THRESHOLD", "value": VALUE_DEVIATION_THRESHOLD}, {"setting": "DELTA_DEVIATION_THRESHOLD", "value": DELTA_DEVIATION_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(scoring_config_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `DataFrame`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Active row-scoring thresholds` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `scoring_config_df = pd.DataFrame( [ {"setting": "VALUE_DEVIATION_THRESHOLD", "value": VALUE_DEVIATION_THRESHOLD}, {"setting": "DELTA_DEVIATION_THRESHOLD", "value": DELTA_DEVIATION_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(scoring_config_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 36 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__any_abnormal_flag`
- `__delta_abnormal_flag`
- `__delta_deviation`
- `__normal_episode_id`
- `__value_abnormal_flag`
- `__value_deviation`
- `abs`
- `astype`
- `baseline_df`
- `concat`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `diff`
- `eq`
- `f`
- `groupby`
- `gt`
- `iloc`

### Outputs

- `axis`
- `baseline_delta_abs_q95`
- `baseline_iqr`
- `baseline_median`
- `delta_count`
- `delta_deviation`
- `delta_deviation_col`
- `delta_flag`
- `delta_flag_col`
- `delta_threshold`
- `derived_columns`
- `derived_df`
- `normal_mask`
- `safe_delta`
- `safe_delta_floor`
- `safe_iqr`
- `safe_iqr_floor`
- `score_rows_against_sensor_baseline`
- `scored_df`
- `sensor_baseline`

### Key Operations

- `def score_rows_against_sensor_baseline( dataframe, baseline_df, sensor_columns, label_column, normal_values, value_threshold=3.0, delta_threshold=2.0, safe_iqr_floor=1e-6, safe_del`: Defines notebook-local logic used later in the notebook.
- `): scored_df = dataframe.copy() normal_mask = scored_df[label_column].isin(normal_values) value_count = pd.Series(0, index=scored_df.index) delta_count = pd.Series(0, index=scored_`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `abs`
- `astype`
- `concat`
- `copy`
- `DataFrame`
- `diff`
- `eq`
- `groupby`
- `gt`
- `isin`
- `max`
- `score_rows_against_sensor_baseline`
- `Series`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def score_rows_against_sensor_baseline( dataframe, baseline_df, sensor_columns, label_column, normal_values, value_threshold=3.0, delta_threshold=2.0, safe_iqr_floor=1e-6, safe_del` | Defines notebook-local logic used later in the notebook. |
| `): scored_df = dataframe.copy() normal_mask = scored_df[label_column].isin(normal_values) value_count = pd.Series(0, index=scored_df.index) delta_count = pd.Series(0, index=scored_` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 37 — Define clean-normal training quality classes

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `clean`
- `copy`
- `def`
- `eq`
- `exclude`
- `ge`
- `is_clean_normal_for_training`
- `isin`
- `label_column`
- `loc`
- `normal_total_abnormal_sensor_count`
- `normal_training_quality_class`
- `normal_values`
- `not_normal`
- `scored_df`
- `suspect`

### Outputs

- `classified_df`
- `classify_normal_training_quality`
- `exclude_sensor_count`
- `normal_mask`
- `suspect_sensor_count`

### Key Operations

- `def classify_normal_training_quality( scored_df, label_column, normal_values, suspect_sensor_count=3, exclude_sensor_count=6,`: Defines notebook-local logic used later in the notebook.
- `): classified_df = scored_df.copy() normal_mask = classified_df[label_column].isin(normal_values) classified_df["normal_training_quality_class"] = "not_normal" classified_df.loc[ n`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `classify_normal_training_quality`
- `copy`
- `eq`
- `ge`
- `isin`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def classify_normal_training_quality( scored_df, label_column, normal_values, suspect_sensor_count=3, exclude_sensor_count=6,` | Defines notebook-local logic used later in the notebook. |
| `): classified_df = scored_df.copy() normal_mask = classified_df[label_column].isin(normal_values) classified_df["normal_training_quality_class"] = "not_normal" classified_df.loc[ n` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 38 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `classify_normal_training_quality`
- `DELTA_DEVIATION_THRESHOLD`
- `final_sensor_baseline_df`
- `normal_profile_df`
- `score_rows_against_sensor_baseline`
- `VALUE_DEVIATION_THRESHOLD`

### Outputs

- `baseline_df`
- `dataframe`
- `delta_threshold`
- `exclude_sensor_count`
- `label_column`
- `normal_values`
- `scored_df`
- `scored_normal_quality_df`
- `sensor_columns`
- `suspect_sensor_count`
- `value_threshold`

### Key Operations

- `scored_normal_quality_df = score_rows_against_sensor_baseline( dataframe=normal_profile_df, baseline_df=final_sensor_baseline_df, sensor_columns=SENSOR_COLUMNS, label_column=LABEL_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `scored_normal_quality_df = classify_normal_training_quality( scored_df=scored_normal_quality_df, label_column=LABEL_COLUMN, normal_values=NORMAL_VALUES, suspect_sensor_count=SUSPEC`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `classify_normal_training_quality`
- `score_rows_against_sensor_baseline`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `scored_normal_quality_df = score_rows_against_sensor_baseline( dataframe=normal_profile_df, baseline_df=final_sensor_baseline_df, sensor_columns=SENSOR_COLUMNS, label_column=LABEL_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `scored_normal_quality_df = classify_normal_training_quality( scored_df=scored_normal_quality_df, label_column=LABEL_COLUMN, normal_values=NORMAL_VALUES, suspect_sensor_count=SUSPEC` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 39 — Build clean-normal quality summary

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dropna`
- `name`
- `normal_training_quality_class`
- `rename_axis`
- `reset_index`
- `row_count`
- `row_pct`
- `scored_normal_quality_df`
- `sum`
- `value_counts`

### Outputs

- `quality_summary_df`

### Key Operations

- `quality_summary_df = ( scored_normal_quality_df["normal_training_quality_class"] .value_counts(dropna=False) .rename_axis("normal_training_quality_class") .reset_index(name="row_co`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `quality_summary_df["row_pct"] = ( quality_summary_df["row_count"] / quality_summary_df["row_count"].sum()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(quality_summary_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `rename_axis`
- `reset_index`
- `sum`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `quality_summary_df = ( scored_normal_quality_df["normal_training_quality_class"] .value_counts(dropna=False) .rename_axis("normal_training_quality_class") .reset_index(name="row_co` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `quality_summary_df["row_pct"] = ( quality_summary_df["row_count"] / quality_summary_df["row_count"].sum()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(quality_summary_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 40 — Review intermediate output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `describe`
- `normal_delta_abnormal_sensor_count`
- `normal_total_abnormal_sensor_count`
- `normal_value_abnormal_sensor_count`
- `scored_normal_quality_df`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `display( scored_normal_quality_df[ [ "normal_value_abnormal_sensor_count", "normal_delta_abnormal_sensor_count", "normal_total_abnormal_sensor_count", ] ].describe()`: Displays a notebook-facing result for inspection.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `describe`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `display( scored_normal_quality_df[ [ "normal_value_abnormal_sensor_count", "normal_delta_abnormal_sensor_count", "normal_total_abnormal_sensor_count", ] ].describe()` | Displays a notebook-facing result for inspection. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 41 — Review intermediate output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `ne`
- `normal_delta_abnormal_sensor_count`
- `normal_total_abnormal_sensor_count`
- `normal_training_quality_class`
- `normal_value_abnormal_sensor_count`
- `not_normal`
- `quantile`
- `scored_normal_quality_df`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `display( scored_normal_quality_df[ scored_normal_quality_df["normal_training_quality_class"].ne("not_normal") ][ [ "normal_value_abnormal_sensor_count", "normal_delta_abnormal_sens`: Displays a notebook-facing result for inspection.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`
- `ne`
- `quantile`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `display( scored_normal_quality_df[ scored_normal_quality_df["normal_training_quality_class"].ne("not_normal") ][ [ "normal_value_abnormal_sensor_count", "normal_delta_abnormal_sens` | Displays a notebook-facing result for inspection. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 42 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `clean`
- `copy`
- `eq`
- `exclude`
- `final_row_quality_class`
- `isin`
- `normal_training_quality_class`
- `row_is_clean_normal`
- `row_is_exclude_from_normal_training`
- `row_is_suspect_normal`
- `scored_normal_quality_df`
- `suspect`

### Outputs

- `silver_subset_df`

### Key Operations

- `silver_subset_df = scored_normal_quality_df.copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `silver_subset_df["final_row_quality_class"] = ( silver_subset_df["normal_training_quality_class"]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_subset_df["row_is_clean_normal"] = ( silver_subset_df["final_row_quality_class"].eq("clean")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_subset_df["row_is_suspect_normal"] = ( silver_subset_df["final_row_quality_class"].eq("suspect")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_subset_df["row_is_exclude_from_normal_training"] = ( silver_subset_df["final_row_quality_class"].isin(["suspect", "exclude"])`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `copy`
- `eq`
- `isin`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `silver_subset_df = scored_normal_quality_df.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `silver_subset_df["final_row_quality_class"] = ( silver_subset_df["normal_training_quality_class"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_subset_df["row_is_clean_normal"] = ( silver_subset_df["final_row_quality_class"].eq("clean")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_subset_df["row_is_suspect_normal"] = ( silver_subset_df["final_row_quality_class"].eq("suspect")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_subset_df["row_is_exclude_from_normal_training"] = ( silver_subset_df["final_row_quality_class"].isin(["suspect", "exclude"])` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 43 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `astype`
- `copy`
- `eq`
- `final_sensor_baseline_df`
- `lower`
- `normal`
- `plot_sensor_profile_with_baseline`
- `sensor_04`
- `silver_subset_df`
- `STATE_COL_SOURCE`
- `string`
- `time_index`

### Outputs

- `baseline_df`
- `dataframe`
- `index_column`
- `max_points`
- `normal_only_df`
- `sensor_name`
- `show_delta`

### Key Operations

- `normal_only_df = silver_subset_df[ silver_subset_df[STATE_COL_SOURCE].astype("string").str.lower().eq("normal")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `].copy()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `plot_sensor_profile_with_baseline( dataframe=normal_only_df, baseline_df=final_sensor_baseline_df, sensor_name="sensor_04", index_column="time_index", show_delta=True, max_points=2`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `copy`
- `eq`
- `lower`
- `plot_sensor_profile_with_baseline`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `normal_only_df = silver_subset_df[ silver_subset_df[STATE_COL_SOURCE].astype("string").str.lower().eq("normal")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plot_sensor_profile_with_baseline( dataframe=normal_only_df, baseline_df=final_sensor_baseline_df, sensor_name="sensor_04", index_column="time_index", show_delta=True, max_points=2` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 44 — Inspect an individual sensor profile

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `all`
- `BASELINE_EXPORT_DIR`
- `every`
- `f`
- `files`
- `final_sensor_baseline_df`
- `head`
- `inline`
- `it`
- `Keeping`
- `normal_only_df`
- `notebook`
- `Optional`
- `plot`
- `plot_all_sensor_profiles`
- `plots`
- `PNG`
- `profile`
- `render`
- `result`

### Outputs

- `all_sensor_profile_plot_results`
- `baseline_df`
- `dataframe`
- `index_column`
- `max_points`
- `save_dir`
- `saved_plot_paths`
- `sensor_columns`
- `SENSOR_PROFILE_PLOT_DIR`
- `SHOW_ALL_SENSOR_PROFILE_PLOTS`
- `show_plots`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Optional: display/save every sensor profile plot`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Set SHOW_ALL_SENSOR_PROFILE_PLOTS = True if you want the notebook to`: Documents the purpose or boundary of the surrounding notebook step.
- `# render all sensor plots inline. Keeping it False still saves PNG files.`: Documents the purpose or boundary of the surrounding notebook step.
- `SHOW_ALL_SENSOR_PROFILE_PLOTS = False`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SENSOR_PROFILE_PLOT_DIR = BASELINE_EXPORT_DIR / "plots"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `all_sensor_profile_plot_results = plot_all_sensor_profiles( dataframe=normal_only_df, baseline_df=final_sensor_baseline_df, sensor_columns=SENSOR_COLUMNS, index_column="time_index"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `saved_plot_paths = [ str(result["saved_path"]) for result in all_sensor_profile_plot_results.values() if result["saved_path"] is not None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print(f"Saved {len(saved_plot_paths)} sensor profile plots to: {SENSOR_PROFILE_PLOT_DIR}")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `DataFrame`
- `display`
- `head`
- `plot_all_sensor_profiles`
- `values`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Optional: display/save every sensor profile plot` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Set SHOW_ALL_SENSOR_PROFILE_PLOTS = True if you want the notebook to` | Documents the purpose or boundary of the surrounding notebook step. |
| `# render all sensor plots inline. Keeping it False still saves PNG files.` | Documents the purpose or boundary of the surrounding notebook step. |
| `SHOW_ALL_SENSOR_PROFILE_PLOTS = False` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SENSOR_PROFILE_PLOT_DIR = BASELINE_EXPORT_DIR / "plots"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `all_sensor_profile_plot_results = plot_all_sensor_profiles( dataframe=normal_only_df, baseline_df=final_sensor_baseline_df, sensor_columns=SENSOR_COLUMNS, index_column="time_index"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `saved_plot_paths = [ str(result["saved_path"]) for result in all_sensor_profile_plot_results.values() if result["saved_path"] is not None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"Saved {len(saved_plot_paths)} sensor profile plots to: {SENSOR_PROFILE_PLOT_DIR}")` | Displays a notebook-facing result for inspection. |
| `display(pd.DataFrame({"saved_plot_path": saved_plot_paths}).head(10))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 45 — Inspect an individual sensor profile

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `check`
- `display_sensor_profile`
- `final_sensor_baseline_df`
- `normal_only_df`
- `plot_sensor_profile_with_baseline`
- `Quick`
- `selected`
- `sensor_00`
- `sensor_04`
- `sensor_14`
- `sensor_20`
- `SENSOR_COLUMNS`
- `sensors`
- `time_index`
- `visual`

### Outputs

- `baseline_df`
- `dataframe`
- `index_column`
- `max_points`
- `sensor_name`
- `show_delta`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Quick visual check for selected sensors`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `for sensor_name in ["sensor_00", "sensor_04", "sensor_14", "sensor_20"]: if sensor_name in SENSOR_COLUMNS: display_sensor_profile( baseline_df=final_sensor_baseline_df, sensor_name`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `display_sensor_profile`
- `plot_sensor_profile_with_baseline`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Quick visual check for selected sensors` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `for sensor_name in ["sensor_00", "sensor_04", "sensor_14", "sensor_20"]: if sensor_name in SENSOR_COLUMNS: display_sensor_profile( baseline_df=final_sensor_baseline_df, sensor_name` | Controls validation, iteration, file handling, or error handling for this step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 46 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `ax1`
- `axhline`
- `band`
- `baseline`
- `baseline_df`
- `Check`
- `columns`
- `copy`
- `dataframe`
- `def`
- `delta`
- `Delta`
- `diff`
- `else`
- `eq`
- `f`
- `fig`
- `final_row_quality_class`
- `get_legend_handles_labels`
- `gt`

### Outputs

- `alpha`
- `ax2`
- `baseline_delta_abs_q95`
- `baseline_iqr`
- `baseline_median`
- `baseline_q05`
- `baseline_q95`
- `figsize`
- `index_column`
- `label`
- `linestyle`
- `linewidth`
- `loc`
- `lower_normal_band`
- `marker`
- `max_points`
- `plot_df`
- `plot_sensor_profile_with_baseline`
- `quality_column`
- `s`

### Key Operations

- `def plot_sensor_profile_with_baseline( dataframe, baseline_df, sensor_name, *, index_column=None, quality_column="final_row_quality_class", figsize=(16, 6), show_delta=True, max_po`: Defines notebook-local logic used later in the notebook.
- `): plot_df = dataframe.copy() if max_points is not None and len(plot_df) > max_points: plot_df = plot_df.iloc[-max_points:].copy() sensor_profile = ( baseline_df .loc[baseline_df["`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `axhline`
- `copy`
- `diff`
- `eq`
- `get_legend_handles_labels`
- `gt`
- `legend`
- `lt`
- `plot`
- `plot_sensor_profile_with_baseline`
- `scatter`
- `set_title`
- `set_xlabel`
- `set_ylabel`
- `show`
- `subplots`
- `tight_layout`
- `twinx`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def plot_sensor_profile_with_baseline( dataframe, baseline_df, sensor_name, *, index_column=None, quality_column="final_row_quality_class", figsize=(16, 6), show_delta=True, max_po` | Defines notebook-local logic used later in the notebook. |
| `): plot_df = dataframe.copy() if max_points is not None and len(plot_df) > max_points: plot_df = plot_df.iloc[-max_points:].copy() sensor_profile = ( baseline_df .loc[baseline_df["` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 47 — Plot sensor profile against the learned baseline

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `final_sensor_baseline_df`
- `plot_sensor_profile_with_baseline`
- `sensor_04`
- `silver_subset_df`
- `time_index`

### Outputs

- `baseline_df`
- `dataframe`
- `index_column`
- `max_points`
- `sensor_name`
- `show_delta`

### Key Operations

- `plot_sensor_profile_with_baseline( dataframe=silver_subset_df, baseline_df=final_sensor_baseline_df, sensor_name="sensor_04", index_column="time_index", show_delta=True, max_points`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `plot_sensor_profile_with_baseline`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `plot_sensor_profile_with_baseline( dataframe=silver_subset_df, baseline_df=final_sensor_baseline_df, sensor_name="sensor_04", index_column="time_index", show_delta=True, max_points` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 48 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `astype`
- `Build`
- `clean`
- `column`
- `copy`
- `dropna`
- `eq`
- `exclude`
- `final_row_quality_class`
- `isin`
- `loc`
- `lower`
- `machine_status__profiled`
- `machine_status__synthetic`
- `name`
- `normal`
- `normal_clean`
- `normal_contaminated`
- `normal_suspect`

### Outputs

- `final_quality_counts_df`
- `mask_normal_clean`
- `mask_normal_exclude`
- `mask_normal_suspect`
- `mask_source_abnormal`
- `mask_source_normal`
- `mask_source_recovery`
- `PROFILED_ABNORMAL_VALUE`
- `PROFILED_NORMAL_CLEAN_VALUE`
- `PROFILED_NORMAL_CONTAMINATED_VALUE`
- `PROFILED_NORMAL_SUSPECT_VALUE`
- `PROFILED_RECOVERY_VALUE`
- `profiled_state_counts_df`
- `quality_class`
- `silver_subset_df`
- `SOURCE_ABNORMAL_VALUE`
- `SOURCE_NORMAL_VALUE`
- `SOURCE_RECOVERY_VALUE`
- `STATE_COL_PROFILED`
- `STATE_COL_SOURCE`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build profiled state column`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `STATE_COL_SOURCE = "machine_status__synthetic"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `STATE_COL_PROFILED = "machine_status__profiled"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SOURCE_NORMAL_VALUE = "normal"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SOURCE_ABNORMAL_VALUE = "abnormal"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SOURCE_RECOVERY_VALUE = "recovery"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PROFILED_NORMAL_CLEAN_VALUE = "normal_clean"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PROFILED_NORMAL_SUSPECT_VALUE = "normal_suspect"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PROFILED_NORMAL_CONTAMINATED_VALUE = "normal_contaminated"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PROFILED_ABNORMAL_VALUE = "abnormal"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `astype`
- `copy`
- `display`
- `eq`
- `isin`
- `lower`
- `rename_axis`
- `reset_index`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build profiled state column` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `STATE_COL_SOURCE = "machine_status__synthetic"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STATE_COL_PROFILED = "machine_status__profiled"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SOURCE_NORMAL_VALUE = "normal"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SOURCE_ABNORMAL_VALUE = "abnormal"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SOURCE_RECOVERY_VALUE = "recovery"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PROFILED_NORMAL_CLEAN_VALUE = "normal_clean"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PROFILED_NORMAL_SUSPECT_VALUE = "normal_suspect"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PROFILED_NORMAL_CONTAMINATED_VALUE = "normal_contaminated"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PROFILED_ABNORMAL_VALUE = "abnormal"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PROFILED_RECOVERY_VALUE = "recovery"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `silver_subset_df = scored_normal_quality_df.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `silver_subset_df["final_row_quality_class"] = ( silver_subset_df["normal_training_quality_class"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_subset_df["row_is_clean_normal"] = ( silver_subset_df["final_row_quality_class"].astype("string").eq("clean")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_subset_df["row_is_suspect_normal"] = ( silver_subset_df["final_row_quality_class"].astype("string").eq("suspect")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_subset_df["row_is_exclude_from_normal_training"] = ( silver_subset_df["final_row_quality_class"] .astype("string") .isin(["suspect", "exclude"])` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_subset_df[STATE_COL_PROFILED] = ( silver_subset_df[STATE_COL_SOURCE] .astype("string") .str.lower()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `mask_source_normal = silver_subset_df[STATE_COL_SOURCE].astype("string").str.lower().eq(SOURCE_NORMAL_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `mask_source_abnormal = silver_subset_df[STATE_COL_SOURCE].astype("string").str.lower().eq(SOURCE_ABNORMAL_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `mask_source_recovery = silver_subset_df[STATE_COL_SOURCE].astype("string").str.lower().eq(SOURCE_RECOVERY_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `quality_class = silver_subset_df["final_row_quality_class"].astype("string").str.lower()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `mask_normal_clean = mask_source_normal & quality_class.eq("clean")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `mask_normal_suspect = mask_source_normal & quality_class.eq("suspect")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `mask_normal_exclude = mask_source_normal & quality_class.eq("exclude")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `silver_subset_df.loc[mask_normal_clean, STATE_COL_PROFILED] = PROFILED_NORMAL_CLEAN_VALUE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_subset_df.loc[mask_normal_suspect, STATE_COL_PROFILED] = PROFILED_NORMAL_SUSPECT_VALUE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_subset_df.loc[mask_normal_exclude, STATE_COL_PROFILED] = PROFILED_NORMAL_CONTAMINATED_VALUE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_subset_df.loc[mask_source_abnormal, STATE_COL_PROFILED] = PROFILED_ABNORMAL_VALUE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_subset_df.loc[mask_source_recovery, STATE_COL_PROFILED] = PROFILED_RECOVERY_VALUE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `profiled_state_counts_df = ( silver_subset_df[STATE_COL_PROFILED] .value_counts(dropna=False) .rename_axis(STATE_COL_PROFILED) .reset_index(name="row_count")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `profiled_state_counts_df["row_pct"] = ( profiled_state_counts_df["row_count"] / len(silver_subset_df)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `final_quality_counts_df = ( silver_subset_df["final_row_quality_class"] .value_counts(dropna=False) .rename_axis("final_row_quality_class") .reset_index(name="row_count")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `final_quality_counts_df["row_pct"] = ( final_quality_counts_df["row_count"] / len(silver_subset_df)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(profiled_state_counts_df)` | Displays a notebook-facing result for inspection. |
| `display(final_quality_counts_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 49 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold_train__normal_clean`
- `__silver_subsets__normal_clean`
- `__silver_subsets__normal_contaminated`
- `__silver_subsets__profiled_dataframe`
- `__silver_subsets__summary`
- `add`
- `alias`
- `Artifact`
- `artifact`
- `baseline_json_path`
- `baseline_table_path`
- `Build`
- `clean`
- `consumption`
- `DATASET_NAME`
- `dataset_name`
- `diagnostics`
- `directory`
- `downstream`
- `dropna`

### Outputs

- `data`
- `gold_train_normal_clean_path`
- `kind`
- `logger`
- `message`
- `normal_clean_path`
- `normal_contaminated_path`
- `profiled_df_path`
- `SILVER_SUBSET_DATA_DIR`
- `step`
- `subset_artifact_dir`
- `subset_summary`
- `subset_summary_path`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Save subset outputs`: Documents the purpose or boundary of the surrounding notebook step.
- `# - parquet files -> data directory for downstream Gold use`: Documents the purpose or boundary of the surrounding notebook step.
- `# - summary json -> artifact directory for diagnostics/traceability`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Data output directory for downstream pipeline consumption`: Documents the purpose or boundary of the surrounding notebook step.
- `#SILVER_SUBSET_DATA_DIR = SILVER_TRAIN_DATA_PATH / "subset_outputs" / DATASET_NAME`: Documents the purpose or boundary of the surrounding notebook step.
- `SILVER_SUBSET_DATA_DIR = SILVER_TRAIN_DATA_PATH`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_SUBSET_DATA_DIR.mkdir(parents=True, exist_ok=True)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Artifact output directory for notebook diagnostics / summary metadata`: Documents the purpose or boundary of the surrounding notebook step.
- `subset_artifact_dir = SILVER_EDA_ARTIFACT_DIR`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `subset_artifact_dir.mkdir(parents=True, exist_ok=True)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `dump`
- `eq`
- `globals`
- `mkdir`
- `open`
- `to_dict`
- `to_parquet`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Save subset outputs` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - parquet files -> data directory for downstream Gold use` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - summary json -> artifact directory for diagnostics/traceability` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Data output directory for downstream pipeline consumption` | Documents the purpose or boundary of the surrounding notebook step. |
| `#SILVER_SUBSET_DATA_DIR = SILVER_TRAIN_DATA_PATH / "subset_outputs" / DATASET_NAME` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_SUBSET_DATA_DIR = SILVER_TRAIN_DATA_PATH` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_SUBSET_DATA_DIR.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Artifact output directory for notebook diagnostics / summary metadata` | Documents the purpose or boundary of the surrounding notebook step. |
| `subset_artifact_dir = SILVER_EDA_ARTIFACT_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `subset_artifact_dir.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Data files Gold should read` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `profiled_df_path = ( SILVER_SUBSET_DATA_DIR / f"{DATASET_NAME}__silver_subsets__profiled_dataframe.parquet"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `normal_clean_path = ( SILVER_SUBSET_DATA_DIR / f"{DATASET_NAME}__silver_subsets__normal_clean.parquet"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `normal_contaminated_path = ( SILVER_SUBSET_DATA_DIR / f"{DATASET_NAME}__silver_subsets__normal_contaminated.parquet"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Optional: clean training-only alias for Gold` | Documents the purpose or boundary of the surrounding notebook step. |
| `gold_train_normal_clean_path = ( SILVER_SUBSET_DATA_DIR / f"{DATASET_NAME}__gold_train__normal_clean.parquet"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Artifact summary file` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `subset_summary_path = ( SILVER_EDA_SUMMARY_DIR / f"{DATASET_NAME}__silver_subsets__summary.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Save parquet data outputs` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `silver_subset_df.to_parquet(profiled_df_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `silver_subset_df.loc[ silver_subset_df[STATE_COL_PROFILED].eq(PROFILED_NORMAL_CLEAN_VALUE)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `].to_parquet(normal_clean_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `silver_subset_df.loc[ silver_subset_df[STATE_COL_PROFILED].eq(PROFILED_NORMAL_CONTAMINATED_VALUE)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `].to_parquet(normal_contaminated_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Optional alias file specifically named for Gold training input` | Documents the purpose or boundary of the surrounding notebook step. |
| `silver_subset_df.loc[ silver_subset_df[STATE_COL_PROFILED].eq(PROFILED_NORMAL_CLEAN_VALUE)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `].to_parquet(gold_train_normal_clean_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build and save summary json` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `subset_summary = { "dataset_name": DATASET_NAME, "stage": STAGE, "final_method": FINAL_METHOD, "state_col_source": STATE_COL_SOURCE, "state_col_profiled": STATE_COL_PROFILED, "sour` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `with open(subset_summary_path, "w", encoding="utf-8") as f: json.dump(subset_summary, f, indent=2)` | Controls validation, iteration, file handling, or error handling for this step. |
| `ledger.add( kind="step", step="save_subset_outputs", message="Saved profiled subset parquet outputs to the data directory and summary metadata to the artifact directory.", data={ "` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Saved data:", profiled_df_path)` | Displays a notebook-facing result for inspection. |
| `print("Saved data:", normal_clean_path)` | Displays a notebook-facing result for inspection. |
| `print("Saved data:", normal_contaminated_path)` | Displays a notebook-facing result for inspection. |
| `#print("Saved data:", gold_train_normal_clean_path)` | Documents the purpose or boundary of the surrounding notebook step. |
| `print("Saved artifact summary:", subset_summary_path)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: Parquet output.

## Code Cell 50 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__silver__truth__`
- `add`
- `Add`
- `append_truth_index`
- `artifact_paths`
- `before`
- `build_silver_eda_subsets_truth_record`
- `build_truth_record`
- `building`
- `Built`
- `col`
- `columns`
- `config_snapshot`
- `created`
- `dataframe`
- `documents`
- `does`
- `downstream`
- `EDA`
- `eda_subsets`

### Outputs

- `column_count`
- `data`
- `dataset_name`
- `expected_silver_eda_subsets_truth_path`
- `feature_columns`
- `kind`
- `layer_name`
- `logger`
- `message`
- `meta_columns`
- `parent_truth_hash`
- `pipeline_mode`
- `process_run_id`
- `row_count`
- `silver_eda_subsets_truth`
- `SILVER_EDA_SUBSETS_TRUTH_HASH`
- `silver_eda_subsets_truth_path`
- `silver_eda_subsets_truth_record`
- `step`
- `truth_base`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Finalize Silver EDA Subsets truth record`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `silver_eda_subsets_truth = initialize_layer_truth( truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name="silver", process_run_id=SILVER_SUBSET_PROCESS_RUN_ID, pipelin`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_eda_subsets_truth = update_truth_section( silver_eda_subsets_truth, "config_snapshot", { "source_config_stage": "silver_eda", "effective_stage": "eda_subsets", "effective_la`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_eda_subsets_truth = update_truth_section( silver_eda_subsets_truth, "runtime_facts", { "parent_layer_name": "silver", "parent_truth_hash": SILVER_TRUTH_HASH, "state_col_sour`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_eda_subsets_truth = update_truth_section( silver_eda_subsets_truth, "artifact_paths", { "profiled_df_path": str(profiled_df_path), "normal_clean_path": str(normal_clean_path`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_eda_subsets_truth = update_truth_section( silver_eda_subsets_truth, "notes", { "purpose": ( "Silver EDA subset-building truth record. This record documents the " "profiled d`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `append_truth_index`
- `build_truth_record`
- `get`
- `identify_meta_columns`
- `initialize_layer_truth`
- `save_truth_record`
- `startswith`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Finalize Silver EDA Subsets truth record` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `silver_eda_subsets_truth = initialize_layer_truth( truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name="silver", process_run_id=SILVER_SUBSET_PROCESS_RUN_ID, pipelin` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_subsets_truth = update_truth_section( silver_eda_subsets_truth, "config_snapshot", { "source_config_stage": "silver_eda", "effective_stage": "eda_subsets", "effective_la` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_subsets_truth = update_truth_section( silver_eda_subsets_truth, "runtime_facts", { "parent_layer_name": "silver", "parent_truth_hash": SILVER_TRUTH_HASH, "state_col_sour` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_subsets_truth = update_truth_section( silver_eda_subsets_truth, "artifact_paths", { "profiled_df_path": str(profiled_df_path), "normal_clean_path": str(normal_clean_path` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_subsets_truth = update_truth_section( silver_eda_subsets_truth, "notes", { "purpose": ( "Silver EDA subset-building truth record. This record documents the " "profiled d` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_subsets_truth_record = build_truth_record( truth_base=silver_eda_subsets_truth, row_count=len(silver_subset_df), column_count=silver_subset_df.shape[1], meta_columns=ide` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_EDA_SUBSETS_TRUTH_HASH = silver_eda_subsets_truth_record["truth_hash"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Add stage metadata before saving and indexing.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The existing truth utility does not add stage-specific filenames, so stage` | Documents the purpose or boundary of the surrounding notebook step. |
| `# identity is stored in the truth record and truth_index.jsonl.` | Documents the purpose or boundary of the surrounding notebook step. |
| `silver_eda_subsets_truth_record["truth_stage"] = "eda_subsets"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_subsets_truth_record["notebook_name"] = "silver_02a_eda_subsets"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `expected_silver_eda_subsets_truth_path = ( TRUTHS_PATH / "silver" / f"{DATASET_NAME}__silver__truth__{SILVER_EDA_SUBSETS_TRUTH_HASH}.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_subsets_truth_record["truth_path"] = str( expected_silver_eda_subsets_truth_path` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_subsets_truth_path = save_truth_record( silver_eda_subsets_truth_record, truth_dir=TRUTHS_PATH, dataset_name=DATASET_NAME, layer_name="silver",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `append_truth_index( silver_eda_subsets_truth_record, truth_index_path=TRUTH_INDEX_PATH,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="build_silver_eda_subsets_truth_record", message="Built and indexed Silver EDA subsets truth record.", data={ "silver_eda_subsets_truth_hash": SILVER_` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Saved Silver EDA subsets truth:", silver_eda_subsets_truth_path)` | Displays a notebook-facing result for inspection. |
| `print("SILVER_EDA_SUBSETS_TRUTH_HASH:", SILVER_EDA_SUBSETS_TRUTH_HASH)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: truth record.

## Code Cell 51 — Review final Silver subset structure

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `info`
- `silver_subset_df`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `s`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `info`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `s` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 52 — Silver EDA SQL Logging Cell

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

