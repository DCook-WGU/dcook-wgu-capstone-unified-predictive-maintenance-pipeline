# Notebook Code Reference: EDA_Notebook_Pump_Silver_02b_EDA_v2

Notebook path:

`notebooks/eda/EDA_Notebook_Pump_Silver_02b_EDA_v2.ipynb`

## Notebook Purpose

This notebook reviews profiled Silver states, sensor groups, distributions, and anomaly context before Gold modeling.

Notebook stage:

`Silver`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Answer | Code Cell 01, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 10, Code Cell 11, Code Cell 12, Code Cell 13, Code Cell 14, Code Cell 15, Code Cell 16, Code Cell 17, Code Cell 18, Code Cell 19, Code Cell 20, Code Cell 21, Code Cell 22, Code Cell 23, Code Cell 24, Code Cell 25, Code Cell 26, Code Cell 27, Code Cell 28, Code Cell 29, Code Cell 30, Code Cell 31, Code Cell 32, Code Cell 33, Code Cell 35, Code Cell 37, Code Cell 41, Code Cell 42, Code Cell 43, Code Cell 51, Code Cell 52, Code Cell 53, Code Cell 54, Code Cell 55, Code Cell 56 |
| Define configuration mapping guards | Code Cell 02 |
| Canonical Silver EDA Artifact Directories | Code Cell 07, Code Cell 08 |
| Review intermediate output | Code Cell 09 |
| Compute the correlation matrix | Code Cell 34, Code Cell 36, Code Cell 49, Code Cell 50, Code Cell 61 |
| Load required stage inputs | Code Cell 38 |
| Record traceability output | Code Cell 39, Code Cell 40, Code Cell 44, Code Cell 45, Code Cell 46 |
| Run validation guardrails | Code Cell 47, Code Cell 48 |
| QA | Code Cell 57 |
| Build the Silver EDA profile table | Code Cell 58 |
| Select numeric columns for EDA statistics | Code Cell 59 |
| Build the missingness summary table | Code Cell 60, Code Cell 64 |
| Build sensor outlier summary records | Code Cell 62 |
| Build categorical summary records | Code Cell 63 |

## Code Cell 01 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__future__`
- `AgglomerativeClustering`
- `annotations`
- `Any`
- `append_truth_index`
- `artifact_file_path`
- `artifacts`
- `build_artifact_dirs`
- `build_artifact_dirs_from_config`
- `build_truth_config_block`
- `build_truth_record`
- `cast`
- `cluster`
- `config_loader`
- `configure_logging`
- `core`
- `database`
- `dataclass`
- `dataclasses`
- `datetime`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `from __future__ import annotations`: Imports a dependency or project helper used by later cells.
- `from dataclasses import dataclass`: Imports a dependency or project helper used by later cells.
- `from datetime import datetime, timezone`: Imports a dependency or project helper used by later cells.
- `from pathlib import Path`: Imports a dependency or project helper used by later cells.
- `from typing import Any, Dict, List, Optional, Sequence, Tuple, Mapping, cast`: Imports a dependency or project helper used by later cells.
- `import json`: Imports a dependency or project helper used by later cells.
- `import logging`: Imports a dependency or project helper used by later cells.
- `import math`: Imports a dependency or project helper used by later cells.
- `import re`: Imports a dependency or project helper used by later cells.
- `import os`: Imports a dependency or project helper used by later cells.
- `import numpy as np`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.

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
| `from typing import Any, Dict, List, Optional, Sequence, Tuple, Mapping, cast` | Imports a dependency or project helper used by later cells. |
| `import json` | Imports a dependency or project helper used by later cells. |
| `import logging` | Imports a dependency or project helper used by later cells. |
| `import math` | Imports a dependency or project helper used by later cells. |
| `import re` | Imports a dependency or project helper used by later cells. |
| `import os` | Imports a dependency or project helper used by later cells. |
| `import numpy as np` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `import matplotlib.pyplot as plt` | Imports a dependency or project helper used by later cells. |
| `import seaborn as sns` | Imports a dependency or project helper used by later cells. |
| `from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis` | Imports a dependency or project helper used by later cells. |
| `from sklearn.decomposition import PCA` | Imports a dependency or project helper used by later cells. |
| `from sklearn.preprocessing import StandardScaler, RobustScaler` | Imports a dependency or project helper used by later cells. |
| `from sklearn.cluster import AgglomerativeClustering, KMeans` | Imports a dependency or project helper used by later cells. |
| `from sklearn.ensemble import IsolationForest` | Imports a dependency or project helper used by later cells. |
| `from utils.core.paths import get_paths` | Imports a dependency or project helper used by later cells. |
| `from utils.core.file_io import ( load_json, save_json,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.logging_setup import ( configure_logging, log_layer_paths,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.ledger import Ledger` | Imports a dependency or project helper used by later cells. |
| `from utils.core.truths import ( make_process_run_id, identify_meta_columns, load_parent_truth_record_from_dataframe, get_dataset_name_from_truth, get_truth_hash, get_parent_truth_h` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.config_loader import ( load_pipeline_config, build_truth_config_block, set_wandb_dir_from_config, export_config_snapshot,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.artifacts import ( build_artifact_dirs, build_artifact_dirs_from_config, artifact_file_path,` | Imports a dependency or project helper used by later cells. |
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
- `cannot`
- `cast`
- `def`
- `dictionary`
- `f`
- `got`
- `isinstance`
- `isnan`
- `Mapping`
- `mapping`
- `math`
- `must`
- `NA`
- `name`
- `NaN`
- `object`

### Outputs

- `cfg_optional_mapping`
- `cfg_require_mapping`
- `require_dict`
- `require_list`
- `scalar_to_float`

### Key Operations

- `def require_dict(value: Any \| None, name: str) -> Dict[str, Any]: if value is None: return {} if not isinstance(value, dict): raise TypeError( f"{name} must be a dictionary, got {t`: Defines notebook-local logic used later in the notebook.
- `def require_list(value: Any \| None, name: str) -> List[Any]: if value is None: return [] if not isinstance(value, list): raise TypeError( f"{name} must be a list, got {type(value).`: Defines notebook-local logic used later in the notebook.
- `def scalar_to_float(value: object, name: str = "value") -> float: if value is None: raise ValueError(f"{name} cannot be None.") if value is pd.NA: raise ValueError(f"{name} cannot `: Defines notebook-local logic used later in the notebook.
- `def cfg_require_mapping(value: object, name: str) -> Mapping[str, Any]: if not isinstance(value, Mapping): raise TypeError( f"{name} must be a mapping, got {type(value).__name__}: `: Defines notebook-local logic used later in the notebook.
- `def cfg_optional_mapping(value: object \| None, name: str) -> Mapping[str, Any]: if value is None: return {} return cfg_require_mapping(value, name)`: Defines notebook-local logic used later in the notebook.

Important functions or methods detected:
- `cast`
- `cfg_optional_mapping`
- `cfg_require_mapping`
- `isinstance`
- `isnan`
- `require_dict`
- `require_list`
- `scalar_to_float`
- `type`
- `TypeError`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def require_dict(value: Any \| None, name: str) -> Dict[str, Any]: if value is None: return {} if not isinstance(value, dict): raise TypeError( f"{name} must be a dictionary, got {t` | Defines notebook-local logic used later in the notebook. |
| `def require_list(value: Any \| None, name: str) -> List[Any]: if value is None: return [] if not isinstance(value, list): raise TypeError( f"{name} must be a list, got {type(value).` | Defines notebook-local logic used later in the notebook. |
| `def scalar_to_float(value: object, name: str = "value") -> float: if value is None: raise ValueError(f"{name} cannot be None.") if value is pd.NA: raise ValueError(f"{name} cannot ` | Defines notebook-local logic used later in the notebook. |
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
- `eda_profile`
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
- `CONTEXT_LOG_FILE = "silver_eda_profile.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.silver.eda_profile", log_filena`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `CONTEXT_LOG_FILE = "silver_eda_profile.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.silver.eda_profile", log_filena` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
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

- `__silver__eda_profile__truth`
- `a`
- `Any`
- `append`
- `appends`
- `artifact`
- `Artifacts`
- `artifacts`
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

### Outputs

- `ARTIFACTS_ROOT`
- `ASSET_ID_DEFAULT_FALLBACK`
- `BRONZE_TRAIN_DATA_PATH`
- `CLEANING_RECIPE_ID`
- `DATASET_NAME`
- `DATASET_NAME_CONFIG`
- `DATASET_RUN_ID`
- `EDA_DATAFRAME_NAME`
- `EDA_NOTEBOOK_NAME`
- `FEATURE_REGISTRY_FILE_NAME`
- `LAYER_NAME`
- `LOG_FILE_NAME`
- `LOGGER_NAME`
- `LOGS_PATH`
- `PIPELINE_MODE`
- `RUN_ID_DEFAULT_FALLBACK`
- `RUN_MODE`
- `SILVER_ARTIFACTS_ROOT`
- `SILVER_EDA_ARTIFACTS_ROOT`
- `SILVER_EDA_TRUTH_DIR`

### Key Operations

- `TRUTH_CONFIG = cast(Dict[str, Any], build_truth_config_block(CONFIG))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_CONFIG["pipeline"] = PIPELINE`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `# ---- Stage details ----`: Documents the purpose or boundary of the surrounding notebook step.
- `STAGE = "silver_eda"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `LAYER_NAME = str(SILVER_EDA_CFG.get("layer_name", "silver_eda"))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CLEANING_RECIPE_ID = str(SILVER_EDA_CFG["cleaning_recipe_id"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_VERSION = str(VERSIONS_CFG["silver_eda"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_VERSION = str(VERSIONS_CFG["truth"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `PIPELINE_MODE = str(PIPELINE["execution_mode"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RUN_MODE = str(RUNTIME_CFG.get("mode", CONFIG_RUN_MODE))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `build_truth_config_block`
- `cast`
- `get`
- `getenv`
- `lower`
- `make_process_run_id`
- `mkdir`
- `Path`
- `save_truth_record`
- `set_wandb_dir_from_config`
- `strip`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `TRUTH_CONFIG = cast(Dict[str, Any], build_truth_config_block(CONFIG))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_CONFIG["pipeline"] = PIPELINE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Stage details ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `STAGE = "silver_eda"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LAYER_NAME = str(SILVER_EDA_CFG.get("layer_name", "silver_eda"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CLEANING_RECIPE_ID = str(SILVER_EDA_CFG["cleaning_recipe_id"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_VERSION = str(VERSIONS_CFG["silver_eda"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_VERSION = str(VERSIONS_CFG["truth"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `PIPELINE_MODE = str(PIPELINE["execution_mode"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_MODE = str(RUNTIME_CFG.get("mode", CONFIG_RUN_MODE))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `DATASET_NAME_CONFIG = str(DATASET_CFG.get("name", "pump"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_NAME = DATASET_NAME_CONFIG.strip().lower()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Keep process-run compatibility with existing downstream cells.` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_PROCESS_RUN_ID = make_process_run_id( str(SILVER_EDA_CFG["process_run_id_prefix"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_SUBSETS_PROCESS_RUN_ID = make_process_run_id( str(SILVER_EDA_CFG.get("subsets_process_run_id_prefix", "silver_subsets"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Notebook identifiers ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `EDA_NOTEBOOK_NAME = "EDA_Notebook_Pump_Silver_02b_EDA_v2"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `EDA_DATAFRAME_NAME = "silver_eda_dataframe"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
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
| `# The artifact-dir cell below owns the detailed subfolders.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The root must remain artifacts/silver/<dataset>/eda.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Do not append "subsets" here. The subsets directory is only one child` | Documents the purpose or boundary of the surrounding notebook step. |
| `# directory under the EDA root.` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_ARTIFACTS_ROOT = ( ARTIFACTS_ROOT / "silver" / str(DATASET_NAME)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_EDA_ARTIFACTS_ROOT = ( SILVER_ARTIFACTS_ROOT / "eda"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `39 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

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

## Code Cell 07 — Canonical Silver EDA Artifact Directories

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__silver__eda__resolved_config`
- `aliases`
- `Aligned`
- `aligned_onset_plots`
- `artifact`
- `artifact_dir`
- `artifacts`
- `Backward`
- `cells`
- `compatible`
- `config`
- `CONFIG`
- `correlation_analysis`
- `DATASET_NAME_CONFIG`
- `dir`
- `directory`
- `distribution_plots`
- `downstream`
- `EDA`
- `eda`

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

- `SILVER_LAYER_NAME = "silver"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_EDA_STAGE_NAME = "eda"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_EDA_ARTIFACT_DIR = ( paths.artifacts / SILVER_LAYER_NAME / DATASET_NAME_CONFIG / SILVER_EDA_STAGE_NAME`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `SILVER_EDA_ALIGNED_ONSET_PLOTS_DIR = SILVER_EDA_ARTIFACT_DIR / "aligned_onset_plots"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_EDA_CONFIG_DIR = SILVER_EDA_ARTIFACT_DIR / "config"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_EDA_CORRELATION_DIR = SILVER_EDA_ARTIFACT_DIR / "correlation_analysis"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_EDA_DISTRIBUTION_DIR = SILVER_EDA_ARTIFACT_DIR / "distribution_plots"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_EDA_GENERATOR_INPUTS_DIR = SILVER_EDA_ARTIFACT_DIR / "generator_inputs"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_EDA_LINEAGE_DIR = SILVER_EDA_ARTIFACT_DIR / "lineage"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_EDA_METADATA_DIR = SILVER_EDA_ARTIFACT_DIR / "metadata"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_EDA_PCA_DIR = SILVER_EDA_ARTIFACT_DIR / "pca"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `export_config_snapshot`
- `get`
- `mkdir`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
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

## Code Cell 08 — Canonical Silver EDA Artifact Directories

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

- `capstone`
- `configure_logging`
- `current_layer`
- `EDA`
- `getLogger`
- `INFO`
- `info`
- `LOG_FILE_NAME`
- `log_layer_paths`
- `logging`
- `Logging`
- `LOGS_PATH`
- `notebook`
- `Original`
- `paths`
- `profiled`
- `Setup`
- `silver`
- `Silver`
- `silver_eda_profiled`

### Outputs

- `level`
- `logger`
- `overwrite_handlers`
- `silver_eda_log_path`

### Key Operations

- `"""`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Original Logging Setup`: Documents the purpose or boundary of the surrounding notebook step.
- `silver_eda_log_path = LOGS_PATH / LOG_FILE_NAME`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `configure_logging( "capstone", silver_eda_log_path, level=logging.INFO, overwrite_handlers=True,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger = logging.getLogger("capstone.silver_eda_profiled")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `logger.info("Silver profiled EDA notebook starting")`: Writes a logger message for traceability during notebook execution.
- `log_layer_paths(paths, current_layer="silver", logger=logger)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Logging to:", silver_eda_log_path) """`: Displays a notebook-facing result for inspection.

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
| `silver_eda_log_path = LOGS_PATH / LOG_FILE_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `configure_logging( "capstone", silver_eda_log_path, level=logging.INFO, overwrite_handlers=True,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger = logging.getLogger("capstone.silver_eda_profiled")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Silver profiled EDA notebook starting")` | Writes a logger message for traceability during notebook execution. |
| `log_layer_paths(paths, current_layer="silver", logger=logger)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Logging to:", silver_eda_log_path) """` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 12 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Add`
- `B`
- `by`
- `default`
- `disabled`
- `else`
- `enabled`
- `experiment`
- `Experiment`
- `Keeping`
- `later`
- `light`
- `needed`
- `notebook`
- `Optional`
- `other`
- `run`
- `this`
- `tracking`
- `W`

### Outputs

- `USE_EXPERIMENT_TRACKING`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Optional experiment tracking`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Keeping this notebook light by default.`: Documents the purpose or boundary of the surrounding notebook step.
- `# Add W&B or other tracking later if needed.`: Documents the purpose or boundary of the surrounding notebook step.
- `USE_EXPERIMENT_TRACKING = False`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if USE_EXPERIMENT_TRACKING: print("Experiment tracking enabled.")`: Displays a notebook-facing result for inspection.
- `else: print("Experiment tracking disabled for this notebook run.")`: Displays a notebook-facing result for inspection.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Optional experiment tracking` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Keeping this notebook light by default.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Add W&B or other tracking later if needed.` | Documents the purpose or boundary of the surrounding notebook step. |
| `USE_EXPERIMENT_TRACKING = False` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if USE_EXPERIMENT_TRACKING: print("Experiment tracking enabled.")` | Displays a notebook-facing result for inspection. |
| `else: print("Experiment tracking disabled for this notebook run.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

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
- `CLEANING_RECIPE_ID`
- `EDA`
- `init`
- `Initialized`
- `initialized`
- `Original`
- `profiled`
- `recipe_id`
- `Setup`
- `Silver`
- `stage`
- `STAGE`

### Outputs

- `kind`
- `ledger`
- `logger`
- `message`
- `step`

### Key Operations

- `"""`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Original Ledger Setup`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger = Ledger(stage=STAGE, recipe_id=CLEANING_RECIPE_ID)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ledger.add( kind="step", step="init", message="Initialized Silver profiled EDA ledger.", logger=logger,`: Records or exports ledger information for stage-level traceability.
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
| `ledger = Ledger(stage=STAGE, recipe_id=CLEANING_RECIPE_ID)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger.add( kind="step", step="init", message="Initialized Silver profiled EDA ledger.", logger=logger,` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Ledger initialized.") """` | Records or exports ledger information for stage-level traceability. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 15 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__silver_subsets__profiled_dataframe`
- `add`
- `at`
- `auto`
- `by`
- `candidate`
- `Checked`
- `column_count`
- `columns`
- `Could`
- `created`
- `dataframe`
- `DATASET_NAME_CONFIG`
- `directories`
- `else`
- `exists`
- `Expected`
- `extend`
- `f`
- `fallback`

### Outputs

- `data`
- `engine`
- `expected_profiled_df_path`
- `fallback_profiled_dirs`
- `kind`
- `logger`
- `message`
- `profiled_candidates`
- `PROFILED_DF_PATH`
- `silver_eda_dataframe`
- `step`

### Key Operations

- `expected_profiled_df_path = ( SILVER_TRAIN_DATA_PATH / f"{DATASET_NAME_CONFIG}__silver_subsets__profiled_dataframe.parquet"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `fallback_profiled_dirs = [ SILVER_EDA_OUTPUT_DIR, SILVER_EDA_ARTIFACT_DIR,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if expected_profiled_df_path.exists(): PROFILED_DF_PATH = expected_profiled_df_path`: Controls validation, iteration, file handling, or error handling for this step.
- `else: profiled_candidates: list[Path] = [] for fallback_profiled_dir in fallback_profiled_dirs: if fallback_profiled_dir.exists(): profiled_candidates.extend( fallback_profiled_dir`: Writes a logger message for traceability during notebook execution.
- `silver_eda_dataframe = pd.read_parquet( PROFILED_DF_PATH, engine="auto",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="load_profiled_dataframe", message="Loaded profiled Silver dataframe from 02a output.", data={ "profiled_df_path": str(PROFILED_DF_PATH), "shape": lis`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Loaded:", PROFILED_DF_PATH)`: Displays a notebook-facing result for inspection.
- `print("Shape:", silver_eda_dataframe.shape)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `display`
- `exists`
- `extend`
- `FileNotFoundError`
- `head`
- `is_file`
- `read_parquet`
- `resolve`
- `rglob`
- `sorted`
- `warning`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `expected_profiled_df_path = ( SILVER_TRAIN_DATA_PATH / f"{DATASET_NAME_CONFIG}__silver_subsets__profiled_dataframe.parquet"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `fallback_profiled_dirs = [ SILVER_EDA_OUTPUT_DIR, SILVER_EDA_ARTIFACT_DIR,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if expected_profiled_df_path.exists(): PROFILED_DF_PATH = expected_profiled_df_path` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: profiled_candidates: list[Path] = [] for fallback_profiled_dir in fallback_profiled_dirs: if fallback_profiled_dir.exists(): profiled_candidates.extend( fallback_profiled_dir` | Writes a logger message for traceability during notebook execution. |
| `silver_eda_dataframe = pd.read_parquet( PROFILED_DF_PATH, engine="auto",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="load_profiled_dataframe", message="Loaded profiled Silver dataframe from 02a output.", data={ "profiled_df_path": str(PROFILED_DF_PATH), "shape": lis` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Loaded:", PROFILED_DF_PATH)` | Displays a notebook-facing result for inspection. |
| `print("Shape:", silver_eda_dataframe.shape)` | Displays a notebook-facing result for inspection. |
| `display(silver_eda_dataframe.head(3))` | Displays a notebook-facing result for inspection. |

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
- `Any`
- `are`
- `artifact`
- `artifacts`
- `astype`
- `at`
- `bool`
- `canonical`
- `cast`
- `column`
- `compatibility`
- `Could`
- `dataset`
- `dir`
- `directories`
- `dropna`
- `EDA`

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

- `SILVER_TRUTH_HASH = extract_truth_hash(silver_eda_dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if SILVER_TRUTH_HASH is None: raise ValueError("Could not resolve meta__truth_hash from Silver dataframe.")`: Controls validation, iteration, file handling, or error handling for this step.
- `SILVER_DATASET_NAME = ( silver_eda_dataframe["meta__dataset"] .dropna() .astype("string") .str.strip()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `SILVER_DATASET_NAME = SILVER_DATASET_NAME[SILVER_DATASET_NAME != ""]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if len(SILVER_DATASET_NAME) == 0: raise ValueError("Silver dataframe is missing usable meta__dataset values.")`: Controls validation, iteration, file handling, or error handling for this step.
- `SILVER_DATASET_NAME = str(SILVER_DATASET_NAME.iloc[0]).strip()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `silver_truth = load_parent_truth_record_from_dataframe( dataframe=silver_eda_dataframe, truth_dir=TRUTHS_PATH, parent_layer_name="silver", dataset_name=SILVER_DATASET_NAME, column_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `DATASET_NAME = get_dataset_name_from_truth(silver_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_TRUTH_HASH = get_truth_hash(silver_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_PARENT_TRUTH_HASH = get_parent_truth_hash(silver_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `astype`
- `bool`
- `cast`
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
| `SILVER_TRUTH_HASH = extract_truth_hash(silver_eda_dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if SILVER_TRUTH_HASH is None: raise ValueError("Could not resolve meta__truth_hash from Silver dataframe.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `SILVER_DATASET_NAME = ( silver_eda_dataframe["meta__dataset"] .dropna() .astype("string") .str.strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_DATASET_NAME = SILVER_DATASET_NAME[SILVER_DATASET_NAME != ""]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(SILVER_DATASET_NAME) == 0: raise ValueError("Silver dataframe is missing usable meta__dataset values.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `SILVER_DATASET_NAME = str(SILVER_DATASET_NAME.iloc[0]).strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `silver_truth = load_parent_truth_record_from_dataframe( dataframe=silver_eda_dataframe, truth_dir=TRUTHS_PATH, parent_layer_name="silver", dataset_name=SILVER_DATASET_NAME, column_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
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
| `runtime_facts = cast( Dict[str, Any], silver_truth.get("runtime_facts", {}) or {},` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CANONICAL_INFO = cast( Dict[str, Any], runtime_facts.get("canonical_info", {}) or {},` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FEATURE_SET_INFO = cast( Dict[str, Any], runtime_facts.get("feature_set", {}) or {},` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `QUALITY_INFO = cast( Dict[str, Any], runtime_facts.get("quality_info", {}) or {},` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `NEEDS_ONE_HOT_ENCODING = bool( silver_truth.get("needs_one_hot_encoding", False)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ONE_HOT_ENCODING_COLUMNS = list( silver_truth.get("one_hot_encoding_columns", [])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_registry_dir = get_artifact_path_from_truth( silver_truth, "feature_registry_dir",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FEATURE_REGISTRY_FILE_NAME = f"{DATASET_NAME}__silver__feature_registry.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FEATURE_REGISTRY_PATH = Path(feature_registry_dir) / "registry" / FEATURE_REGISTRY_FILE_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Keep compatibility variables pointed at the canonical Silver EDA artifact tree.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# These variables must not point to /eda/subsets as the root.` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_EDA_ARTIFACT_DIR = SILVER_EDA_ARTIFACT_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_OUTPUT_DIR = SILVER_EDA_SUBSETS_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Truth records are stored at the layer level:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# artifacts/truths/silver/` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The EDA profile identity is stored in truth_stage and truth_index metadata,` | Documents the purpose or boundary of the surrounding notebook step. |
| `# not as nested folders such as silver/eda/eda_profile/silver.` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_EDA_TRUTH_STAGE = "eda_profile"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_TRUTH_DIR = TRUTHS_PATH / "silver"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_EDA_TRUTH_DIR.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_EDA_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("Loaded Silver truth: %s", SILVER_TRUTH_PATH)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Resolved dataset name from Silver truth: %s", DATASET_NAME)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Resolved label source column from Silver truth: %s", LABEL_SOURCE_COLUMN)` | Writes a logger message for traceability during notebook execution. |
| `ledger.add( kind="step", step="resolve_truth", message="Resolved Silver truth, dataset identity, and subset artifact directories.", data={ "dataset_name": DATASET_NAME, "silver_tru` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Loaded Silver truth:", SILVER_TRUTH_PATH)` | Displays a notebook-facing result for inspection. |
| `print("Silver truth hash:", SILVER_TRUTH_HASH)` | Displays a notebook-facing result for inspection. |
| `print("Resolved dataset name:", DATASET_NAME)` | Displays a notebook-facing result for inspection. |
| `4 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 17 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `After`
- `be`
- `before`
- `but`
- `column`
- `column_name`
- `columns`
- `config`
- `count`
- `dataframe`
- `EDA`
- `empty`
- `exports`
- `f`
- `feature`
- `Feature`
- `feature_count`
- `feature_registry_path`
- `FEATURE_REGISTRY_PATH`

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

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Load feature registry from Silver truth and resolve feature columns`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `if FEATURE_REGISTRY_PATH is None: raise ValueError( "FEATURE_REGISTRY_PATH was not resolved from Silver truth before loading the feature registry." )`: Controls validation, iteration, file handling, or error handling for this step.
- `feature_registry_raw = load_json( FEATURE_REGISTRY_PATH.parent, FEATURE_REGISTRY_PATH.name,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `feature_registry = require_dict( feature_registry_raw, "feature_registry",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `feature_columns_raw = require_list( feature_registry.get("feature_columns"), "feature_registry['feature_columns']",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `FEATURE_COLUMNS = [ str(column_name).strip() for column_name in feature_columns_raw if str(column_name).strip() != ""`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `get`
- `identify_meta_columns`
- `info`
- `load_json`
- `require_dict`
- `require_list`
- `strip`
- `ValueError`
- `warning`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Load feature registry from Silver truth and resolve feature columns` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `if FEATURE_REGISTRY_PATH is None: raise ValueError( "FEATURE_REGISTRY_PATH was not resolved from Silver truth before loading the feature registry." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `feature_registry_raw = load_json( FEATURE_REGISTRY_PATH.parent, FEATURE_REGISTRY_PATH.name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_registry = require_dict( feature_registry_raw, "feature_registry",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_columns_raw = require_list( feature_registry.get("feature_columns"), "feature_registry['feature_columns']",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FEATURE_COLUMNS = [ str(column_name).strip() for column_name in feature_columns_raw if str(column_name).strip() != ""` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if len(FEATURE_COLUMNS) == 0: raise ValueError( f"Feature registry was loaded but feature_columns is empty: {FEATURE_REGISTRY_PATH}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `missing_feature_columns = [ column_name for column_name in FEATURE_COLUMNS if column_name not in silver_eda_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_feature_columns: logger.warning( "Some feature registry columns were not found in the dataframe and will be skipped: %s", missing_feature_columns[:20], )` | Writes a logger message for traceability during notebook execution. |
| `FEATURE_COLUMNS = [ column_name for column_name in FEATURE_COLUMNS if column_name in silver_eda_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if len(FEATURE_COLUMNS) == 0: raise ValueError( "After intersecting the feature registry with dataframe columns, no usable feature columns remained." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `CONFIG_SENSOR_COLUMNS = FEATURE_COLUMNS` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `meta_columns = identify_meta_columns(silver_eda_dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `meta_column_set = set(meta_columns)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Loaded Silver Feature Registry: %s", FEATURE_REGISTRY_PATH)` | Writes a logger message for traceability during notebook execution. |
| `ledger.add( kind="step", step="load_silver_feature_registry", message="Loaded Silver Feature Registry JSON file using the path resolved from Silver truth.", why="Silver EDA should ` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
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

- `abs`
- `Absolute`
- `back`
- `coerce`
- `ddof`
- `def`
- `dropna`
- `dtype`
- `else`
- `empty`
- `eps`
- `errors`
- `Falls`
- `float64`
- `functions`
- `helper`
- `index`
- `isnan`
- `median`
- `nan`

### Outputs

- `center`
- `mad`
- `robust_abs_z`
- `robust_center_scale`
- `s`
- `scale`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Robust helper functions`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `def robust_center_scale(series: pd.Series, eps: float = 1e-9) -> tuple[float, float]: """ Returns robust center and robust scale using: - median - MAD * 1.4826 Falls back to std if`: Defines notebook-local logic used later in the notebook.
- `def robust_abs_z(series: pd.Series, center: float, scale: float) -> pd.Series: """ Absolute robust z-score. """ s = pd.to_numeric(series, errors="coerce") if scale <= 0 or np.isnan`: Defines notebook-local logic used later in the notebook.

Important functions or methods detected:
- `abs`
- `dropna`
- `isnan`
- `median`
- `return`
- `robust_abs_z`
- `robust_center_scale`
- `scalar_to_float`
- `Series`
- `std`
- `to_numeric`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Robust helper functions` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `def robust_center_scale(series: pd.Series, eps: float = 1e-9) -> tuple[float, float]: """ Returns robust center and robust scale using: - median - MAD * 1.4826 Falls back to std if` | Defines notebook-local logic used later in the notebook. |
| `def robust_abs_z(series: pd.Series, center: float, scale: float) -> pd.Series: """ Absolute robust z-score. """ s = pd.to_numeric(series, errors="coerce") if scale <= 0 or np.isnan` | Defines notebook-local logic used later in the notebook. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 19 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `all_rows`
- `astype`
- `column_name`
- `columns`
- `count`
- `DataFrame`
- `dataframe`
- `eq`
- `f`
- `KeyError`
- `lower`
- `machine_status__profiled`
- `machine_status__synthetic`
- `masks`
- `Missing`
- `normal`
- `normal_clean`
- `normal_contaminated`
- `profiled`

### Outputs

- `mask_profiled_abnormal`
- `mask_profiled_normal_clean`
- `mask_profiled_normal_contaminated`
- `mask_profiled_recovery`
- `mask_source_normal`
- `missing_state_cols`
- `PROFILED_ABNORMAL_VALUE`
- `PROFILED_NORMAL_CLEAN_VALUE`
- `PROFILED_NORMAL_CONTAMINATED_VALUE`
- `PROFILED_RECOVERY_VALUE`
- `profiled_state_mask_summary_df`
- `required_state_cols`
- `SOURCE_NORMAL_VALUE`
- `STATE_COL_PROFILED`
- `STATE_COL_SOURCE`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Source / profiled state columns and masks`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `STATE_COL_SOURCE = "machine_status__synthetic"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `STATE_COL_PROFILED = "machine_status__profiled"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SOURCE_NORMAL_VALUE = "normal"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PROFILED_NORMAL_CLEAN_VALUE = "normal_clean"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PROFILED_NORMAL_CONTAMINATED_VALUE = "normal_contaminated"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PROFILED_ABNORMAL_VALUE = "abnormal"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PROFILED_RECOVERY_VALUE = "recovery"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `required_state_cols = [STATE_COL_SOURCE, STATE_COL_PROFILED]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `missing_state_cols = [ column_name for column_name in required_state_cols if column_name not in silver_eda_dataframe.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `astype`
- `DataFrame`
- `display`
- `eq`
- `KeyError`
- `lower`
- `strip`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Source / profiled state columns and masks` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `STATE_COL_SOURCE = "machine_status__synthetic"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STATE_COL_PROFILED = "machine_status__profiled"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SOURCE_NORMAL_VALUE = "normal"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PROFILED_NORMAL_CLEAN_VALUE = "normal_clean"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PROFILED_NORMAL_CONTAMINATED_VALUE = "normal_contaminated"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PROFILED_ABNORMAL_VALUE = "abnormal"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PROFILED_RECOVERY_VALUE = "recovery"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `required_state_cols = [STATE_COL_SOURCE, STATE_COL_PROFILED]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `missing_state_cols = [ column_name for column_name in required_state_cols if column_name not in silver_eda_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_state_cols: raise KeyError( f"Missing required state columns in profiled dataframe: {missing_state_cols}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `mask_source_normal = ( silver_eda_dataframe[STATE_COL_SOURCE] .astype(str) .str.lower() .str.strip() .eq(SOURCE_NORMAL_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `mask_profiled_normal_clean = ( silver_eda_dataframe[STATE_COL_PROFILED] .astype(str) .str.lower() .str.strip() .eq(PROFILED_NORMAL_CLEAN_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `mask_profiled_normal_contaminated = ( silver_eda_dataframe[STATE_COL_PROFILED] .astype(str) .str.lower() .str.strip() .eq(PROFILED_NORMAL_CONTAMINATED_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `mask_profiled_abnormal = ( silver_eda_dataframe[STATE_COL_PROFILED] .astype(str) .str.lower() .str.strip() .eq(PROFILED_ABNORMAL_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `mask_profiled_recovery = ( silver_eda_dataframe[STATE_COL_PROFILED] .astype(str) .str.lower() .str.strip() .eq(PROFILED_RECOVERY_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `profiled_state_mask_summary_df = pd.DataFrame( { "state_scope": [ "all_rows", "source_normal", "profiled_normal_clean", "profiled_normal_contaminated", "profiled_abnormal", "profil` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(profiled_state_mask_summary_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 20 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `Built`
- `column_count`
- `dataframe`
- `DataFrame`
- `dataset_name`
- `DATASET_NAME`
- `FEATURE_COLUMNS`
- `feature_count`
- `ledger`
- `mask_profiled_abnormal`
- `mask_profiled_normal_clean`
- `mask_profiled_normal_contaminated`
- `mask_profiled_recovery`
- `meta_columns`
- `meta_count`
- `overview`
- `profiled`
- `profiled_abnormal_rows`
- `profiled_normal_clean_rows`

### Outputs

- `data`
- `index`
- `kind`
- `logger`
- `message`
- `overview_summary_df`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Quick overview summary`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `overview_summary_df = pd.DataFrame( { "value": [ DATASET_NAME, silver_eda_dataframe.shape[0], silver_eda_dataframe.shape[1], len(FEATURE_COLUMNS), len(meta_columns), int(mask_profi`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="quick_overview", message="Built quick overview summary for profiled Silver dataframe.", data={"overview": overview_summary_df["value"].to_dict()}, lo`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(overview_summary_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `DataFrame`
- `display`
- `sum`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Quick overview summary` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `overview_summary_df = pd.DataFrame( { "value": [ DATASET_NAME, silver_eda_dataframe.shape[0], silver_eda_dataframe.shape[1], len(FEATURE_COLUMNS), len(meta_columns), int(mask_profi` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="quick_overview", message="Built quick overview summary for profiled Silver dataframe.", data={"overview": overview_summary_df["value"].to_dict()}, lo` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(overview_summary_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 21 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `Built`
- `column_name`
- `columns`
- `columns_with_any_missing`
- `DataFrame`
- `dataset`
- `drop`
- `else`
- `empty`
- `Full`
- `full`
- `full_dataset_missingness`
- `head`
- `isna`
- `ledger`
- `max`
- `max_missing_pct`
- `mean`
- `mean_missing_pct_across_columns`

### Outputs

- `ascending`
- `data`
- `index`
- `kind`
- `logger`
- `message`
- `missingness_df`
- `missingness_summary_df`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Full-dataset missingness summary`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `missingness_df = pd.DataFrame( { "column_name": silver_eda_dataframe.columns, "missing_count": silver_eda_dataframe.isna().sum().values, "row_count": len(silver_eda_dataframe), }`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_df["missing_pct"] = ( missingness_df["missing_count"] / missingness_df["row_count"]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_df = missingness_df.sort_values( ["missing_pct", "missing_count", "column_name"], ascending=[False, False, True],`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `).reset_index(drop=True)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="full_dataset_missingness", message="Built full-dataset missingness summary.", data={ "top_missing_columns": missingness_df.head(20).to_dict(orient="r`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(missingness_df.head(25))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `DataFrame`
- `display`
- `head`
- `isna`
- `max`
- `mean`
- `reset_index`
- `sort_values`
- `sum`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Full-dataset missingness summary` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `missingness_df = pd.DataFrame( { "column_name": silver_eda_dataframe.columns, "missing_count": silver_eda_dataframe.isna().sum().values, "row_count": len(silver_eda_dataframe), }` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_df["missing_pct"] = ( missingness_df["missing_count"] / missingness_df["row_count"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_df = missingness_df.sort_values( ["missing_pct", "missing_count", "column_name"], ascending=[False, False, True],` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).reset_index(drop=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="full_dataset_missingness", message="Built full-dataset missingness summary.", data={ "top_missing_columns": missingness_df.head(20).to_dict(orient="r` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(missingness_df.head(25))` | Displays a notebook-facing result for inspection. |
| `missingness_summary_df = pd.DataFrame( { "value": [ int((missingness_df["missing_count"] > 0).sum()), float(missingness_df["missing_pct"].max()) if not missingness_df.empty else np` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(missingness_summary_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 22 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `append`
- `Built`
- `c`
- `check_name`
- `checks`
- `columns`
- `continue`
- `DataFrame`
- `duplicate`
- `Duplicate`
- `duplicate_checks`
- `duplicate_key_check`
- `duplicate_rate`
- `duplicate_summary`
- `duplicated`
- `else`
- `event_step`
- `event_time`
- `full_row_duplicates`

### Outputs

- `data`
- `duplicate_check_rows`
- `duplicate_count`
- `DUPLICATE_KEY_CANDIDATES`
- `duplicate_summary_df`
- `existing_cols`
- `full_row_duplicate_count`
- `kind`
- `logger`
- `message`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Duplicate row / duplicate key checks`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `DUPLICATE_KEY_CANDIDATES = [ ["meta__asset_id", "time_index"], ["meta__asset_id", "event_step"], ["meta__asset_id", "event_time"], ["meta__record_id"],`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `duplicate_check_rows = []`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `full_row_duplicate_count = int(silver_eda_dataframe.duplicated().sum())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `duplicate_check_rows.append( { "check_name": "full_row_duplicates", "key_columns": None, "duplicate_count": full_row_duplicate_count, "row_count": int(len(silver_eda_dataframe)), "`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `for key_cols in DUPLICATE_KEY_CANDIDATES: existing_cols = [c for c in key_cols if c in silver_eda_dataframe.columns] if not existing_cols: continue duplicate_count = int(silver_eda`: Controls validation, iteration, file handling, or error handling for this step.
- `duplicate_summary_df = pd.DataFrame(duplicate_check_rows)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ledger.add( kind="step", step="duplicate_checks", message="Built duplicate row / duplicate key summary.", data={"duplicate_summary": duplicate_summary_df.to_dict(orient="records") `: Records or exports ledger information for stage-level traceability.

Important functions or methods detected:
- `add`
- `append`
- `DataFrame`
- `display`
- `duplicated`
- `join`
- `sum`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Duplicate row / duplicate key checks` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `DUPLICATE_KEY_CANDIDATES = [ ["meta__asset_id", "time_index"], ["meta__asset_id", "event_step"], ["meta__asset_id", "event_time"], ["meta__record_id"],` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `duplicate_check_rows = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `full_row_duplicate_count = int(silver_eda_dataframe.duplicated().sum())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `duplicate_check_rows.append( { "check_name": "full_row_duplicates", "key_columns": None, "duplicate_count": full_row_duplicate_count, "row_count": int(len(silver_eda_dataframe)), "` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for key_cols in DUPLICATE_KEY_CANDIDATES: existing_cols = [c for c in key_cols if c in silver_eda_dataframe.columns] if not existing_cols: continue duplicate_count = int(silver_eda` | Controls validation, iteration, file handling, or error handling for this step. |
| `duplicate_summary_df = pd.DataFrame(duplicate_check_rows)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger.add( kind="step", step="duplicate_checks", message="Built duplicate row / duplicate key summary.", data={"duplicate_summary": duplicate_summary_df.to_dict(orient="records") ` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(duplicate_summary_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 23 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `api`
- `append`
- `astype`
- `bias`
- `Built`
- `categorical`
- `categorical_column_count`
- `coerce`
- `column`
- `column_name`
- `columns`
- `continue`
- `DataFrame`
- `ddof`
- `drop`
- `dropna`
- `else`
- `empty`
- `errors`

### Outputs

- `categorical_columns`
- `categorical_profile_df`
- `categorical_profile_rows`
- `data`
- `kind`
- `logger`
- `message`
- `non_null`
- `numeric_columns`
- `numeric_profile_df`
- `numeric_profile_rows`
- `s`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Numeric / categorical profiling summary`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `numeric_columns = [ column for column in silver_eda_dataframe.columns if pd.api.types.is_numeric_dtype(silver_eda_dataframe[column])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `categorical_columns = [ column for column in silver_eda_dataframe.columns if column not in numeric_columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `numeric_profile_rows = []`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `for column in FEATURE_COLUMNS: if column not in silver_eda_dataframe.columns: continue s = pd.to_numeric(silver_eda_dataframe[column], errors="coerce") non_null = s.dropna() numeri`: Controls validation, iteration, file handling, or error handling for this step.
- `numeric_profile_df = pd.DataFrame(numeric_profile_rows).sort_values("feature").reset_index(drop=True)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `categorical_profile_rows = []`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `for column in categorical_columns: s = silver_eda_dataframe[column].astype("string") categorical_profile_rows.append( { "column_name": column, "row_count": int(len(s)), "non_null_c`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `add`
- `append`
- `astype`
- `DataFrame`
- `display`
- `dropna`
- `head`
- `is_numeric_dtype`
- `isna`
- `max`
- `mean`
- `median`
- `min`
- `mode`
- `notna`
- `nunique`
- `quantile`
- `reset_index`
- `scipy_kurtosis`
- `scipy_skew`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Numeric / categorical profiling summary` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `numeric_columns = [ column for column in silver_eda_dataframe.columns if pd.api.types.is_numeric_dtype(silver_eda_dataframe[column])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `categorical_columns = [ column for column in silver_eda_dataframe.columns if column not in numeric_columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `numeric_profile_rows = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for column in FEATURE_COLUMNS: if column not in silver_eda_dataframe.columns: continue s = pd.to_numeric(silver_eda_dataframe[column], errors="coerce") non_null = s.dropna() numeri` | Controls validation, iteration, file handling, or error handling for this step. |
| `numeric_profile_df = pd.DataFrame(numeric_profile_rows).sort_values("feature").reset_index(drop=True)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `categorical_profile_rows = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for column in categorical_columns: s = silver_eda_dataframe[column].astype("string") categorical_profile_rows.append( { "column_name": column, "row_count": int(len(s)), "non_null_c` | Controls validation, iteration, file handling, or error handling for this step. |
| `categorical_profile_df = pd.DataFrame(categorical_profile_rows).sort_values("column_name").reset_index(drop=True)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger.add( kind="step", step="numeric_categorical_profile", message="Built numeric and categorical profiling summaries.", data={ "numeric_feature_count": int(len(numeric_profile_d` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(numeric_profile_df.head(20))` | Displays a notebook-facing result for inspection. |
| `display(categorical_profile_df.head(20))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 24 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dropna`
- `keep`
- `machine_status__synthetic`
- `name`
- `on`
- `rename_axis`
- `reporting`
- `reset_index`
- `row_count`
- `row_pct`
- `silver_eda_dataframe`
- `Source`
- `state`
- `STATE_COL_SOURCE`
- `value_counts`

### Outputs

- `source_state_counts_df`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Source-state reporting (keep on machine_status__synthetic)`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `source_state_counts_df = ( silver_eda_dataframe[STATE_COL_SOURCE] .value_counts(dropna=False) .rename_axis(STATE_COL_SOURCE) .reset_index(name="row_count")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `source_state_counts_df["row_pct"] = source_state_counts_df["row_count"] / len(silver_eda_dataframe)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(source_state_counts_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `rename_axis`
- `reporting`
- `reset_index`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Source-state reporting (keep on machine_status__synthetic)` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `source_state_counts_df = ( silver_eda_dataframe[STATE_COL_SOURCE] .value_counts(dropna=False) .rename_axis(STATE_COL_SOURCE) .reset_index(name="row_count")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `source_state_counts_df["row_pct"] = source_state_counts_df["row_count"] / len(silver_eda_dataframe)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(source_state_counts_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 25 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `append`
- `Build`
- `coerce`
- `columns`
- `continue`
- `DataFrame`
- `ddof`
- `def`
- `df`
- `dropna`
- `else`
- `empty`
- `errors`
- `groupby`
- `grouped`
- `helper`
- `isna`
- `max`
- `mean`

### Outputs

- `build_state_sensor_profile_table`
- `non_null`
- `s`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Profile / grouped summary helper`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `def build_state_sensor_profile_table( df: pd.DataFrame, *, sensor_cols: Sequence[str], state_col: str,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """ Build per-state per-sensor summary stats. """ rows: list[dict[str, Any]] = [] for state_value, state_df in df.groupby(state_col, dropna=False): for sensor_co`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append`
- `build_state_sensor_profile_table`
- `DataFrame`
- `dropna`
- `groupby`
- `isna`
- `max`
- `mean`
- `median`
- `min`
- `quantile`
- `std`
- `sum`
- `to_numeric`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Profile / grouped summary helper` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `def build_state_sensor_profile_table( df: pd.DataFrame, *, sensor_cols: Sequence[str], state_col: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Build per-state per-sensor summary stats. """ rows: list[dict[str, Any]] = [] for state_value, state_df in df.groupby(state_col, dropna=False): for sensor_co` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 26 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `add`
- `Built`
- `copy`
- `DataFrame`
- `dictionary`
- `keys`
- `ledger`
- `loc`
- `mask_profiled_abnormal`
- `mask_profiled_normal_clean`
- `mask_profiled_normal_contaminated`
- `mask_profiled_recovery`
- `normal_clean`
- `normal_contaminated`
- `orient`
- `Profiled`
- `profiled`
- `profiled_subset_summary`
- `records`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `PROFILED_STATE_SUBSETS`
- `profiled_subset_summary_df`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Profiled-state subsets`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `PROFILED_STATE_SUBSETS = { "normal_clean": silver_eda_dataframe.loc[mask_profiled_normal_clean].copy(), "normal_contaminated": silver_eda_dataframe.loc[mask_profiled_normal_contami`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `profiled_subset_summary_df = pd.DataFrame( { "state_name": list(PROFILED_STATE_SUBSETS.keys()), "row_count": [len(v) for v in PROFILED_STATE_SUBSETS.values()], }`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `profiled_subset_summary_df["row_pct"] = profiled_subset_summary_df["row_count"] / len(silver_eda_dataframe)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="profiled_state_subsets", message="Built profiled-state subset dictionary.", data={"profiled_subset_summary": profiled_subset_summary_df.to_dict(orien`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(profiled_subset_summary_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `copy`
- `DataFrame`
- `display`
- `keys`
- `to_dict`
- `values`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Profiled-state subsets` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `PROFILED_STATE_SUBSETS = { "normal_clean": silver_eda_dataframe.loc[mask_profiled_normal_clean].copy(), "normal_contaminated": silver_eda_dataframe.loc[mask_profiled_normal_contami` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `profiled_subset_summary_df = pd.DataFrame( { "state_name": list(PROFILED_STATE_SUBSETS.keys()), "row_count": [len(v) for v in PROFILED_STATE_SUBSETS.values()], }` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `profiled_subset_summary_df["row_pct"] = profiled_subset_summary_df["row_count"] / len(silver_eda_dataframe)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="profiled_state_subsets", message="Built profiled-state subset dictionary.", data={"profiled_subset_summary": profiled_subset_summary_df.to_dict(orien` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(profiled_subset_summary_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 27 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abs`
- `add`
- `Built`
- `copy`
- `drop`
- `Dropped`
- `dropped`
- `dropped_sensor_review`
- `eq`
- `feature`
- `fillna`
- `head`
- `is_all_missing`
- `is_constant_or_near_constant`
- `ledger`
- `missing`
- `missing_count`
- `missing_pct`
- `non_null_count`
- `numeric_profile_df`

### Outputs

- `ascending`
- `data`
- `dropped_sensor_review_df`
- `kind`
- `logger`
- `message`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Dropped / missing sensor review`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `dropped_sensor_review_df = numeric_profile_df.copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `dropped_sensor_review_df["missing_pct"] = ( dropped_sensor_review_df["missing_count"] / dropped_sensor_review_df["row_count"]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `dropped_sensor_review_df["is_all_missing"] = dropped_sensor_review_df["non_null_count"].eq(0)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `dropped_sensor_review_df["is_constant_or_near_constant"] = ( dropped_sensor_review_df["std"].fillna(0).abs() <= 1e-12`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `dropped_sensor_review_df = dropped_sensor_review_df.sort_values( ["missing_pct", "is_constant_or_near_constant", "feature"], ascending=[False, False, True],`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `).reset_index(drop=True)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="dropped_sensor_review", message="Built dropped/missing sensor review table.", data={"top_dropped_sensor_review": dropped_sensor_review_df.head(20).to`: Records or exports ledger information for stage-level traceability.

Important functions or methods detected:
- `abs`
- `add`
- `copy`
- `display`
- `eq`
- `fillna`
- `head`
- `reset_index`
- `sort_values`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Dropped / missing sensor review` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `dropped_sensor_review_df = numeric_profile_df.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `dropped_sensor_review_df["missing_pct"] = ( dropped_sensor_review_df["missing_count"] / dropped_sensor_review_df["row_count"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `dropped_sensor_review_df["is_all_missing"] = dropped_sensor_review_df["non_null_count"].eq(0)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `dropped_sensor_review_df["is_constant_or_near_constant"] = ( dropped_sensor_review_df["std"].fillna(0).abs() <= 1e-12` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `dropped_sensor_review_df = dropped_sensor_review_df.sort_values( ["missing_pct", "is_constant_or_near_constant", "feature"], ascending=[False, False, True],` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).reset_index(drop=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="dropped_sensor_review", message="Built dropped/missing sensor review table.", data={"top_dropped_sensor_review": dropped_sensor_review_df.head(20).to` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(dropped_sensor_review_df.head(25))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 28 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `Built`
- `by`
- `column`
- `columns`
- `dropna`
- `episode`
- `Episode`
- `episode_by_source_state`
- `episode_source_count_rows`
- `episode_total_rows`
- `Expected`
- `found`
- `groupby`
- `head`
- `KeyError`
- `ledger`
- `left`
- `merge`
- `meta__episode_id`

### Outputs

- `data`
- `episode_source_counts_df`
- `episode_source_totals_df`
- `how`
- `kind`
- `logger`
- `message`
- `on`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Episode-by-source-state summary`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `if "meta__episode_id" not in silver_eda_dataframe.columns: raise KeyError("Expected episode column not found: meta__episode_id")`: Controls validation, iteration, file handling, or error handling for this step.
- `episode_source_counts_df = ( silver_eda_dataframe.groupby(["meta__episode_id", STATE_COL_SOURCE], dropna=False) .size() .rename("row_count") .reset_index()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `episode_source_totals_df = ( silver_eda_dataframe.groupby("meta__episode_id") .size() .rename("episode_total_rows") .reset_index()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `episode_source_counts_df = episode_source_counts_df.merge( episode_source_totals_df, on="meta__episode_id", how="left",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `episode_source_counts_df["row_pct"] = ( episode_source_counts_df["row_count"] / episode_source_counts_df["episode_total_rows"]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `display`
- `groupby`
- `head`
- `KeyError`
- `merge`
- `rename`
- `reset_index`
- `size`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Episode-by-source-state summary` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `if "meta__episode_id" not in silver_eda_dataframe.columns: raise KeyError("Expected episode column not found: meta__episode_id")` | Controls validation, iteration, file handling, or error handling for this step. |
| `episode_source_counts_df = ( silver_eda_dataframe.groupby(["meta__episode_id", STATE_COL_SOURCE], dropna=False) .size() .rename("row_count") .reset_index()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `episode_source_totals_df = ( silver_eda_dataframe.groupby("meta__episode_id") .size() .rename("episode_total_rows") .reset_index()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `episode_source_counts_df = episode_source_counts_df.merge( episode_source_totals_df, on="meta__episode_id", how="left",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `episode_source_counts_df["row_pct"] = ( episode_source_counts_df["row_count"] / episode_source_counts_df["episode_total_rows"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="episode_by_source_state", message="Built episode-by-source-state summary.", data={"episode_source_count_rows": int(len(episode_source_counts_df))}, l` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(episode_source_counts_df.head(20))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 29 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `Built`
- `by`
- `dropna`
- `Episode`
- `episode`
- `episode_by_profiled_state`
- `episode_profiled_count_rows`
- `episode_total_rows`
- `groupby`
- `head`
- `ledger`
- `left`
- `merge`
- `meta__episode_id`
- `profiled`
- `rename`
- `reset_index`
- `row_count`
- `row_pct`

### Outputs

- `data`
- `episode_profiled_counts_df`
- `episode_profiled_totals_df`
- `how`
- `kind`
- `logger`
- `message`
- `on`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Episode-by-profiled-state summary`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `episode_profiled_counts_df = ( silver_eda_dataframe.groupby(["meta__episode_id", STATE_COL_PROFILED], dropna=False) .size() .rename("row_count") .reset_index()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `episode_profiled_totals_df = ( silver_eda_dataframe.groupby("meta__episode_id") .size() .rename("episode_total_rows") .reset_index()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `episode_profiled_counts_df = episode_profiled_counts_df.merge( episode_profiled_totals_df, on="meta__episode_id", how="left",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `episode_profiled_counts_df["row_pct"] = ( episode_profiled_counts_df["row_count"] / episode_profiled_counts_df["episode_total_rows"]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="episode_by_profiled_state", message="Built episode-by-profiled-state summary.", data={"episode_profiled_count_rows": int(len(episode_profiled_counts_`: Records or exports ledger information for stage-level traceability.

Important functions or methods detected:
- `add`
- `display`
- `groupby`
- `head`
- `merge`
- `rename`
- `reset_index`
- `size`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Episode-by-profiled-state summary` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `episode_profiled_counts_df = ( silver_eda_dataframe.groupby(["meta__episode_id", STATE_COL_PROFILED], dropna=False) .size() .rename("row_count") .reset_index()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `episode_profiled_totals_df = ( silver_eda_dataframe.groupby("meta__episode_id") .size() .rename("episode_total_rows") .reset_index()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `episode_profiled_counts_df = episode_profiled_counts_df.merge( episode_profiled_totals_df, on="meta__episode_id", how="left",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `episode_profiled_counts_df["row_pct"] = ( episode_profiled_counts_df["row_count"] / episode_profiled_counts_df["episode_total_rows"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="episode_by_profiled_state", message="Built episode-by-profiled-state summary.", data={"episode_profiled_count_rows": int(len(episode_profiled_counts_` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(episode_profiled_counts_df.head(20))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 30 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `Any`
- `append`
- `Built`
- `by`
- `DataFrame`
- `dropna`
- `else`
- `feature`
- `feature_col`
- `FEATURE_COLUMNS`
- `group_column`
- `group_value`
- `groupby`
- `grouped`
- `head`
- `isna`
- `ledger`
- `missingness`
- `Missingness`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `missing_count`
- `missing_pct`
- `missingness_by_source_state_df`
- `row_count`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Missingness by source state`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `source_missingness_rows: list[dict[str, Any]] = []`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `for state_value, state_df in silver_eda_dataframe.groupby(STATE_COL_SOURCE, dropna=False): for feature_col in FEATURE_COLUMNS: row_count = int(len(state_df)) missing_count = int(st`: Controls validation, iteration, file handling, or error handling for this step.
- `missingness_by_source_state_df = pd.DataFrame(source_missingness_rows)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ledger.add( kind="step", step="missingness_by_source_state", message="Built missingness summary grouped by source state.", data={"source_missingness_rows": int(len(missingness_by_s`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(missingness_by_source_state_df.head(20))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `append`
- `DataFrame`
- `display`
- `groupby`
- `head`
- `isna`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Missingness by source state` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `source_missingness_rows: list[dict[str, Any]] = []` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for state_value, state_df in silver_eda_dataframe.groupby(STATE_COL_SOURCE, dropna=False): for feature_col in FEATURE_COLUMNS: row_count = int(len(state_df)) missing_count = int(st` | Controls validation, iteration, file handling, or error handling for this step. |
| `missingness_by_source_state_df = pd.DataFrame(source_missingness_rows)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger.add( kind="step", step="missingness_by_source_state", message="Built missingness summary grouped by source state.", data={"source_missingness_rows": int(len(missingness_by_s` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(missingness_by_source_state_df.head(20))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 31 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `Any`
- `append`
- `Built`
- `by`
- `DataFrame`
- `dropna`
- `else`
- `feature`
- `feature_col`
- `FEATURE_COLUMNS`
- `group_column`
- `group_value`
- `groupby`
- `grouped`
- `head`
- `isna`
- `ledger`
- `missingness`
- `Missingness`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `missing_count`
- `missing_pct`
- `missingness_by_profiled_state_df`
- `row_count`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Missingness by profiled state`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `profiled_missingness_rows: list[dict[str, Any]] = []`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `for state_value, state_df in silver_eda_dataframe.groupby(STATE_COL_PROFILED, dropna=False): for feature_col in FEATURE_COLUMNS: row_count = int(len(state_df)) missing_count = int(`: Controls validation, iteration, file handling, or error handling for this step.
- `missingness_by_profiled_state_df = pd.DataFrame(profiled_missingness_rows)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ledger.add( kind="step", step="missingness_by_profiled_state", message="Built missingness summary grouped by profiled state.", data={"profiled_missingness_rows": int(len(missingnes`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(missingness_by_profiled_state_df.head(20))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `append`
- `DataFrame`
- `display`
- `groupby`
- `head`
- `isna`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Missingness by profiled state` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `profiled_missingness_rows: list[dict[str, Any]] = []` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for state_value, state_df in silver_eda_dataframe.groupby(STATE_COL_PROFILED, dropna=False): for feature_col in FEATURE_COLUMNS: row_count = int(len(state_df)) missing_count = int(` | Controls validation, iteration, file handling, or error handling for this step. |
| `missingness_by_profiled_state_df = pd.DataFrame(profiled_missingness_rows)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger.add( kind="step", step="missingness_by_profiled_state", message="Built missingness summary grouped by profiled state.", data={"profiled_missingness_rows": int(len(missingnes` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(missingness_by_profiled_state_df.head(20))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 32 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `agg`
- `also`
- `Any`
- `append`
- `ascending`
- `astype`
- `Build`
- `Built`
- `by`
- `c`
- `column`
- `consecutive`
- `continue`
- `copy`
- `could`
- `count`
- `counts`
- `DataFrame`
- `def`

### Outputs

- `build_state_transition_tables`
- `columns`
- `data`
- `dwell_df`
- `dwell_summary_df`
- `episode_col`
- `fallback_order_cols`
- `kind`
- `logger`
- `max_run_length`
- `mean_run_length`
- `median_run_length`
- `message`
- `min_run_length`
- `order_col`
- `run_count`
- `run_len`
- `run_state`
- `state_col`
- `states`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# State transitions / dwell lengths`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `def build_state_transition_tables( df: pd.DataFrame, *, state_col: str, episode_col: str = "meta__episode_id", order_col: str = "time_index",`: Defines notebook-local logic used later in the notebook.
- `) -> tuple[pd.DataFrame, pd.DataFrame]: """ Build: - transition table: from_state -> to_state counts - dwell table: consecutive run lengths by state """ if episode_col not in df.co`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `source_transition_counts_df, source_dwell_summary_df = build_state_transition_tables( silver_eda_dataframe, state_col=STATE_COL_SOURCE, episode_col="meta__episode_id", order_col="t`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `profiled_transition_counts_df, profiled_dwell_summary_df = build_state_transition_tables( silver_eda_dataframe, state_col=STATE_COL_PROFILED, episode_col="meta__episode_id", order_`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="state_transitions_and_dwell", message="Built state transition and dwell summaries for source and profiled states.", data={ "source_transition_rows": `: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Source transitions")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `agg`
- `append`
- `astype`
- `build_state_transition_tables`
- `copy`
- `DataFrame`
- `display`
- `groupby`
- `head`
- `KeyError`
- `ne`
- `notna`
- `rename`
- `reset_index`
- `shift`
- `sort_values`
- `tolist`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# State transitions / dwell lengths` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `def build_state_transition_tables( df: pd.DataFrame, *, state_col: str, episode_col: str = "meta__episode_id", order_col: str = "time_index",` | Defines notebook-local logic used later in the notebook. |
| `) -> tuple[pd.DataFrame, pd.DataFrame]: """ Build: - transition table: from_state -> to_state counts - dwell table: consecutive run lengths by state """ if episode_col not in df.co` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `source_transition_counts_df, source_dwell_summary_df = build_state_transition_tables( silver_eda_dataframe, state_col=STATE_COL_SOURCE, episode_col="meta__episode_id", order_col="t` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `profiled_transition_counts_df, profiled_dwell_summary_df = build_state_transition_tables( silver_eda_dataframe, state_col=STATE_COL_PROFILED, episode_col="meta__episode_id", order_` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="state_transitions_and_dwell", message="Built state transition and dwell summaries for source and profiled states.", data={ "source_transition_rows": ` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Source transitions")` | Displays a notebook-facing result for inspection. |
| `display(source_transition_counts_df.head(20))` | Displays a notebook-facing result for inspection. |
| `print("Source dwell summary")` | Displays a notebook-facing result for inspection. |
| `display(source_dwell_summary_df)` | Displays a notebook-facing result for inspection. |
| `print("Profiled transitions")` | Displays a notebook-facing result for inspection. |
| `display(profiled_transition_counts_df.head(20))` | Displays a notebook-facing result for inspection. |
| `print("Profiled dwell summary")` | Displays a notebook-facing result for inspection. |
| `display(profiled_dwell_summary_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 33 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abs`
- `abs_correlation`
- `add`
- `apply`
- `astype`
- `baseline`
- `bool`
- `Built`
- `clean`
- `coerce`
- `columns`
- `copy`
- `corr`
- `correlation`
- `correlation_analysis`
- `csv`
- `drop`
- `errors`
- `exist_ok`
- `FEATURE_COLUMNS`

### Outputs

- `ascending`
- `by`
- `clean_corr_pairs`
- `CORRELATION_ARTIFACT_DIR`
- `correlation_matrix_normal_clean`
- `correlation_matrix_normal_clean_path`
- `data`
- `kind`
- `logger`
- `message`
- `normal_clean_df`
- `sensor_correlation_pairs_normal_clean_path`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Normal-clean correlation baseline`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `CORRELATION_ARTIFACT_DIR = SILVER_EDA_ARTIFACT_DIR / "correlation_analysis"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CORRELATION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `normal_clean_df = silver_eda_dataframe.loc[mask_profiled_normal_clean, FEATURE_COLUMNS].copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `normal_clean_df = normal_clean_df.apply(pd.to_numeric, errors="coerce")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `correlation_matrix_normal_clean = normal_clean_df.corr()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `correlation_matrix_normal_clean_path = ( CORRELATION_ARTIFACT_DIR / "correlation_matrix_normal_clean.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `correlation_matrix_normal_clean.to_csv(correlation_matrix_normal_clean_path)`: Writes an artifact or output used for review or downstream notebooks.
- `clean_corr_pairs = ( correlation_matrix_normal_clean.where( np.triu(np.ones(correlation_matrix_normal_clean.shape), k=1).astype(bool) ) .stack() .reset_index()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `abs`
- `add`
- `apply`
- `astype`
- `copy`
- `corr`
- `display`
- `head`
- `mkdir`
- `ones`
- `reset_index`
- `sort_values`
- `stack`
- `to_csv`
- `to_dict`
- `triu`
- `where`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Normal-clean correlation baseline` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `CORRELATION_ARTIFACT_DIR = SILVER_EDA_ARTIFACT_DIR / "correlation_analysis"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CORRELATION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `normal_clean_df = silver_eda_dataframe.loc[mask_profiled_normal_clean, FEATURE_COLUMNS].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `normal_clean_df = normal_clean_df.apply(pd.to_numeric, errors="coerce")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `correlation_matrix_normal_clean = normal_clean_df.corr()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `correlation_matrix_normal_clean_path = ( CORRELATION_ARTIFACT_DIR / "correlation_matrix_normal_clean.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `correlation_matrix_normal_clean.to_csv(correlation_matrix_normal_clean_path)` | Writes an artifact or output used for review or downstream notebooks. |
| `clean_corr_pairs = ( correlation_matrix_normal_clean.where( np.triu(np.ones(correlation_matrix_normal_clean.shape), k=1).astype(bool) ) .stack() .reset_index()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `clean_corr_pairs.columns = ["sensor_a", "sensor_b", "correlation"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `clean_corr_pairs["abs_correlation"] = clean_corr_pairs["correlation"].abs()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `clean_corr_pairs = clean_corr_pairs.sort_values( by=["abs_correlation", "sensor_a", "sensor_b"], ascending=[False, True, True],` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).reset_index(drop=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `sensor_correlation_pairs_normal_clean_path = ( CORRELATION_ARTIFACT_DIR / "sensor_correlation_pairs_normal_clean.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `clean_corr_pairs.to_csv(sensor_correlation_pairs_normal_clean_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `ledger.add( kind="step", step="normal_clean_correlation", message="Built normal-clean correlation baseline and pair table.", data={ "normal_clean_rows": int(len(normal_clean_df)), ` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Saved:", correlation_matrix_normal_clean_path)` | Displays a notebook-facing result for inspection. |
| `print("Saved:", sensor_correlation_pairs_normal_clean_path)` | Displays a notebook-facing result for inspection. |
| `display(correlation_matrix_normal_clean.iloc[:10, :10])` | Displays a notebook-facing result for inspection. |
| `display(clean_corr_pairs.head(20))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 34 — Compute the correlation matrix

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abs`
- `abs_correlation`
- `abs_correlation_clean`
- `abs_correlation_contaminated`
- `abs_correlation_delta`
- `add`
- `apply`
- `artifacts`
- `astype`
- `bool`
- `build`
- `Built`
- `clean_corr_pairs`
- `coerce`
- `comparison`
- `contaminated`
- `copy`
- `corr`
- `correlation`
- `CORRELATION_ARTIFACT_DIR`

### Outputs

- `ascending`
- `by`
- `columns`
- `contaminated_corr_pairs`
- `correlation_delta_pairs_df`
- `correlation_delta_pairs_path`
- `correlation_matrix_normal_contaminated`
- `correlation_matrix_normal_contaminated_path`
- `data`
- `how`
- `kind`
- `logger`
- `message`
- `na_position`
- `normal_contaminated_df`
- `on`
- `step`

### Key Operations

- `normal_contaminated_df = silver_eda_dataframe.loc[ mask_profiled_normal_contaminated, FEATURE_COLUMNS,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `].copy()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `normal_contaminated_df = normal_contaminated_df.apply(pd.to_numeric, errors="coerce")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if len(normal_contaminated_df) > 1: correlation_matrix_normal_contaminated = normal_contaminated_df.corr() correlation_matrix_normal_contaminated_path = ( CORRELATION_ARTIFACT_DIR `: Records or exports ledger information for stage-level traceability.
- `else: correlation_matrix_normal_contaminated = pd.DataFrame() correlation_delta_pairs_df = pd.DataFrame() print("Not enough normal_contaminated rows to build correlation comparison`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `abs`
- `add`
- `apply`
- `astype`
- `copy`
- `corr`
- `DataFrame`
- `display`
- `drop`
- `head`
- `merge`
- `ones`
- `reset_index`
- `sort_values`
- `stack`
- `to_csv`
- `to_dict`
- `triu`
- `where`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `normal_contaminated_df = silver_eda_dataframe.loc[ mask_profiled_normal_contaminated, FEATURE_COLUMNS,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `normal_contaminated_df = normal_contaminated_df.apply(pd.to_numeric, errors="coerce")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(normal_contaminated_df) > 1: correlation_matrix_normal_contaminated = normal_contaminated_df.corr() correlation_matrix_normal_contaminated_path = ( CORRELATION_ARTIFACT_DIR ` | Records or exports ledger information for stage-level traceability. |
| `else: correlation_matrix_normal_contaminated = pd.DataFrame() correlation_delta_pairs_df = pd.DataFrame() print("Not enough normal_contaminated rows to build correlation comparison` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 35 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abs`
- `add`
- `adjacency_map`
- `append`
- `at`
- `baseline`
- `Build`
- `by`
- `clean`
- `columns`
- `component`
- `components`
- `continue`
- `correlated`
- `CORRELATION_ARTIFACT_DIR`
- `correlation_matrix_normal_clean`
- `correlations`
- `csv`
- `DataFrame`
- `drop`

### Outputs

- `abs_corr_clean`
- `correlation_value`
- `current`
- `data`
- `group_name`
- `kind`
- `logger`
- `message`
- `sensor_group_map_df`
- `sensor_group_map_path`
- `stack`
- `start`
- `step`
- `SUBSYSTEM_CORR_THRESHOLD`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build correlated baseline subsystems from normal-clean`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `SUBSYSTEM_CORR_THRESHOLD = 0.80`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `abs_corr_clean = correlation_matrix_normal_clean.abs().fillna(0.0)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `adjacency_map: dict[str, set[str]] = { sensor: set() for sensor in FEATURE_COLUMNS`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `for sensor_a in FEATURE_COLUMNS: for sensor_b in FEATURE_COLUMNS: if sensor_a == sensor_b: continue if sensor_a not in abs_corr_clean.index or sensor_b not in abs_corr_clean.column`: Controls validation, iteration, file handling, or error handling for this step.
- `visited: set[str] = set()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `components: list[list[str]] = []`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `for sensor in FEATURE_COLUMNS: if sensor in visited: continue stack = [sensor] component: list[str] = [] while stack: current = stack.pop() if current in visited: continue visited.`: Controls validation, iteration, file handling, or error handling for this step.
- `sensor_group_rows: list[dict[str, object]] = []`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `abs`
- `add`
- `append`
- `DataFrame`
- `display`
- `enumerate`
- `fillna`
- `get`
- `head`
- `nunique`
- `pop`
- `reset_index`
- `scalar_to_float`
- `sort_values`
- `sorted`
- `to_csv`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build correlated baseline subsystems from normal-clean` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `SUBSYSTEM_CORR_THRESHOLD = 0.80` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `abs_corr_clean = correlation_matrix_normal_clean.abs().fillna(0.0)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `adjacency_map: dict[str, set[str]] = { sensor: set() for sensor in FEATURE_COLUMNS` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for sensor_a in FEATURE_COLUMNS: for sensor_b in FEATURE_COLUMNS: if sensor_a == sensor_b: continue if sensor_a not in abs_corr_clean.index or sensor_b not in abs_corr_clean.column` | Controls validation, iteration, file handling, or error handling for this step. |
| `visited: set[str] = set()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `components: list[list[str]] = []` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for sensor in FEATURE_COLUMNS: if sensor in visited: continue stack = [sensor] component: list[str] = [] while stack: current = stack.pop() if current in visited: continue visited.` | Controls validation, iteration, file handling, or error handling for this step. |
| `sensor_group_rows: list[dict[str, object]] = []` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for group_num, sensors in enumerate( sorted(components, key=lambda values: (-len(values), values[0])), start=1,` | Controls validation, iteration, file handling, or error handling for this step. |
| `): group_name = f"subsystem_{group_num:02d}" for sensor in sensors: sensor_group_rows.append( { "sensor": sensor, "subsystem_group": group_name, "group_size": int(len(sensors)), } ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `sensor_group_map_df = ( pd.DataFrame(sensor_group_rows) .sort_values(by=["subsystem_group", "sensor"]) .reset_index(drop=True)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `sensor_group_map_path = CORRELATION_ARTIFACT_DIR / "sensor_group_map_normal_clean.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `sensor_group_map_df.to_csv(sensor_group_map_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `ledger.add( kind="step", step="sensor_subsystem_grouping", message="Grouped sensors into baseline subsystems using normal-clean correlations.", data={ "subsystem_corr_threshold": f` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Saved:", sensor_group_map_path)` | Displays a notebook-facing result for inspection. |
| `display(sensor_group_map_df.head(30))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 36 — Compute the correlation matrix

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `abs`
- `abs_correlation`
- `abs_correlation_abnormal`
- `abs_correlation_clean`
- `abs_correlation_delta_abnormal_vs_clean`
- `add`
- `apply`
- `astype`
- `bool`
- `build`
- `Built`
- `changes`
- `clean`
- `clean_corr_pairs`
- `coerce`
- `comparison`
- `copy`
- `corr`
- `correlation`

### Outputs

- `abnormal_corr_pairs`
- `abnormal_df`
- `ascending`
- `by`
- `columns`
- `correlation_matrix_abnormal`
- `data`
- `fault_pairings_df`
- `fault_pairings_path`
- `how`
- `kind`
- `logger`
- `message`
- `na_position`
- `on`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Fault-propagation pair changes`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `abnormal_df = silver_eda_dataframe.loc[ mask_profiled_abnormal, FEATURE_COLUMNS,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `].copy()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `abnormal_df = abnormal_df.apply(pd.to_numeric, errors="coerce")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if len(abnormal_df) > 1: correlation_matrix_abnormal = abnormal_df.corr() abnormal_corr_pairs = ( correlation_matrix_abnormal.where( np.triu(np.ones(correlation_matrix_abnormal.sha`: Records or exports ledger information for stage-level traceability.
- `else: fault_pairings_df = pd.DataFrame() print("Not enough abnormal rows to build fault-pairing comparison.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `abs`
- `add`
- `apply`
- `astype`
- `copy`
- `corr`
- `DataFrame`
- `display`
- `drop`
- `head`
- `merge`
- `ones`
- `reset_index`
- `sort_values`
- `stack`
- `to_csv`
- `to_dict`
- `triu`
- `where`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Fault-propagation pair changes` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `abnormal_df = silver_eda_dataframe.loc[ mask_profiled_abnormal, FEATURE_COLUMNS,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `abnormal_df = abnormal_df.apply(pd.to_numeric, errors="coerce")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(abnormal_df) > 1: correlation_matrix_abnormal = abnormal_df.corr() abnormal_corr_pairs = ( correlation_matrix_abnormal.where( np.triu(np.ones(correlation_matrix_abnormal.sha` | Records or exports ledger information for stage-level traceability. |
| `else: fault_pairings_df = pd.DataFrame() print("Not enough abnormal rows to build fault-pairing comparison.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 37 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__name__`
- `a`
- `abnormal`
- `abs_correlation_abnormal`
- `abs_correlation_delta_abnormal_vs_clean`
- `add`
- `all`
- `append`
- `are`
- `artifact`
- `artifacts`
- `ascending`
- `astype`
- `be`
- `bounded_near_constant`
- `bounded_normal`
- `Build`
- `build`
- `built`
- `but`

### Outputs

- `abnormal_mask`
- `abnormal_profile_source_df`
- `build_rich_feature_profile`
- `data`
- `distribution_family`
- `EPISODE_STATUS_EXPORT_PATH`
- `errors`
- `fault_pairings_df`
- `fault_pairings_generator_df`
- `fault_pairings_obj`
- `fault_source_df`
- `feature_columns`
- `feature_profile_abnormal_df`
- `feature_profile_abnormal_path`
- `feature_profile_normal_clean_df`
- `feature_profile_normal_clean_path`
- `feature_profile_recovery_df`
- `feature_profile_recovery_path`
- `GENERATOR_INPUT_DIR`
- `generator_input_manifest`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Export generator-ready Silver subset artifacts`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# These files are the direct contract used by the synthetic generator.`: Documents the purpose or boundary of the surrounding notebook step.
- `# They are built from profiled states, especially normal_clean.`: Documents the purpose or boundary of the surrounding notebook step.
- `GENERATOR_INPUT_DIR = SILVER_EDA_ARTIFACT_DIR / "generator_inputs"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GENERATOR_INPUT_DIR.mkdir(parents=True, exist_ok=True)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Episode status counts are written later, but define the path here`: Documents the purpose or boundary of the surrounding notebook step.
- `# so the generator manifest and final truth cell can safely reference it.`: Documents the purpose or boundary of the surrounding notebook step.
- `EPISODE_STATUS_EXPORT_PATH = GENERATOR_INPUT_DIR / "episode_status_counts.json"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `def infer_distribution_family( *, std: float, iqr: float, skewness: float, kurtosis: float,`: Defines notebook-local logic used later in the notebook.
- `) -> str: """ Small, generator-friendly distribution family classifier. """ if not np.isfinite(std) or std <= 1e-12: return "near_constant" if not np.isfinite(iqr) or iqr <= 1e-12:`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `all`
- `append`
- `astype`
- `build_rich_feature_profile`
- `cast`
- `clip`
- `copy`
- `DataFrame`
- `display`
- `drop_duplicates`
- `dropna`
- `dump`
- `eq`
- `fillna`
- `get`
- `globals`
- `head`
- `infer_distribution_family`
- `isfinite`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Export generator-ready Silver subset artifacts` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# These files are the direct contract used by the synthetic generator.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# They are built from profiled states, especially normal_clean.` | Documents the purpose or boundary of the surrounding notebook step. |
| `GENERATOR_INPUT_DIR = SILVER_EDA_ARTIFACT_DIR / "generator_inputs"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GENERATOR_INPUT_DIR.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Episode status counts are written later, but define the path here` | Documents the purpose or boundary of the surrounding notebook step. |
| `# so the generator manifest and final truth cell can safely reference it.` | Documents the purpose or boundary of the surrounding notebook step. |
| `EPISODE_STATUS_EXPORT_PATH = GENERATOR_INPUT_DIR / "episode_status_counts.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `def infer_distribution_family( *, std: float, iqr: float, skewness: float, kurtosis: float,` | Defines notebook-local logic used later in the notebook. |
| `) -> str: """ Small, generator-friendly distribution family classifier. """ if not np.isfinite(std) or std <= 1e-12: return "near_constant" if not np.isfinite(iqr) or iqr <= 1e-12:` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def build_rich_feature_profile( dataframe: pd.DataFrame, *, feature_columns: Sequence[str], state_scope: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Build the rich profile format expected by utils.synthetic.generator.synthetic_profiles. Required columns: sensor, mean, std, min, max, median, iqr, p01, p05,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build state-specific dataframes` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `normal_clean_mask = ( silver_eda_dataframe[STATE_COL_PROFILED] .astype(str) .eq(PROFILED_NORMAL_CLEAN_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `abnormal_mask = ( silver_eda_dataframe[STATE_COL_PROFILED] .astype(str) .eq(PROFILED_ABNORMAL_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `recovery_mask = ( silver_eda_dataframe[STATE_COL_PROFILED] .astype(str) .eq(PROFILED_RECOVERY_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `normal_clean_profile_source_df = cast( pd.DataFrame, silver_eda_dataframe.loc[normal_clean_mask, FEATURE_COLUMNS].copy(),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `abnormal_profile_source_df = cast( pd.DataFrame, silver_eda_dataframe.loc[abnormal_mask, FEATURE_COLUMNS].copy(),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `recovery_profile_source_df = cast( pd.DataFrame, silver_eda_dataframe.loc[recovery_mask, FEATURE_COLUMNS].copy(),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if normal_clean_profile_source_df.empty: raise ValueError("Cannot build generator profile: normal_clean source dataframe is empty.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if abnormal_profile_source_df.empty: raise ValueError("Cannot build generator profile: abnormal source dataframe is empty.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if recovery_profile_source_df.empty: raise ValueError("Cannot build generator profile: recovery source dataframe is empty.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Generator-ready rich profiles` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `feature_profile_normal_clean_df = build_rich_feature_profile( normal_clean_profile_source_df, feature_columns=FEATURE_COLUMNS, state_scope="normal_clean",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_profile_abnormal_df = build_rich_feature_profile( abnormal_profile_source_df, feature_columns=FEATURE_COLUMNS, state_scope="abnormal",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_profile_recovery_df = build_rich_feature_profile( recovery_profile_source_df, feature_columns=FEATURE_COLUMNS, state_scope="recovery",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_profile_normal_clean_path = ( GENERATOR_INPUT_DIR / "feature_profile_normal_clean.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_profile_abnormal_path = ( GENERATOR_INPUT_DIR / "feature_profile_abnormal.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_profile_recovery_path = ( GENERATOR_INPUT_DIR / "feature_profile_recovery.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `EPISODE_STATUS_EXPORT_PATH = ( GENERATOR_INPUT_DIR / "episode_status_counts.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_profile_normal_clean_df.to_csv(feature_profile_normal_clean_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `feature_profile_abnormal_df.to_csv(feature_profile_abnormal_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `feature_profile_recovery_df.to_csv(feature_profile_recovery_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Generator-ready group map` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `sensor_group_map_obj = globals().get("sensor_group_map_df")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if sensor_group_map_obj is None: raise ValueError( "sensor_group_map_df is missing. Run the normal-clean group-map cell first." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `if not isinstance(sensor_group_map_obj, pd.DataFrame): raise TypeError( f"sensor_group_map_df must be a DataFrame, got {type(sensor_group_map_obj).__name__}." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `sensor_group_map_df = sensor_group_map_obj` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if sensor_group_map_df.empty: raise ValueError( "sensor_group_map_df is empty. Run the normal-clean group-map cell first." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `49 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 38 — Load required stage inputs

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__silver__dropped_sensors`
- `__silver_preeda__dropped_sensors`
- `abnormal`
- `actual`
- `Add`
- `add`
- `Align`
- `align`
- `all`
- `all_null`
- `Alternate`
- `an`
- `apply`
- `are`
- `artifact`
- `ARTIFACTS_ROOT`
- `astype`
- `auto`
- `be`
- `before`

### Outputs

- `abnormal_mask`
- `data`
- `drop_reasons`
- `dropped_feature_profile_abnormal_path`
- `dropped_feature_profile_normal_clean_path`
- `dropped_feature_profile_recovery_path`
- `dropped_features_from_registry`
- `dropped_missing_pct`
- `dropped_physical_sensors`
- `dropped_profile_abnormal_df`
- `dropped_profile_normal_clean_df`
- `dropped_profile_recovery_df`
- `dropped_profile_source_df`
- `DROPPED_SENSOR_REGISTRY_PATH`
- `dropped_sensor_registry_payload`
- `dropped_sensors_df`
- `DROPPED_SENSORS_PARQUET_PATH`
- `engine`
- `feature_columns`
- `feature_registry_payload`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Export dropped-sensor rich profiles for synthetic generator`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Correct behavior:`: Documents the purpose or boundary of the surrounding notebook step.
- `# - Use PreEDA feature registry to identify true dropped physical sensors.`: Documents the purpose or boundary of the surrounding notebook step.
- `# - Use the PreEDA dropped_sensors parquet to get the actual dropped sensor values.`: Documents the purpose or boundary of the surrounding notebook step.
- `# - Use silver_eda_dataframe only for the profiled state masks.`: Documents the purpose or boundary of the surrounding notebook step.
- `# - sensor_15 is all-null and should not be generated.`: Documents the purpose or boundary of the surrounding notebook step.
- `# - sensor_50 should be generated, then missingness-masked.`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `import re`: Imports a dependency or project helper used by later cells.
- `DROPPED_SENSOR_REGISTRY_PATH = ( GENERATOR_INPUT_DIR / "dropped_sensor_registry.json"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `astype`
- `build_rich_feature_profile`
- `compile`
- `copy`
- `DataFrame`
- `display`
- `dump`
- `eq`
- `exists`
- `FileNotFoundError`
- `get`
- `globals`
- `in`
- `items`
- `join`
- `load`
- `match`
- `NameError`
- `open`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Export dropped-sensor rich profiles for synthetic generator` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Correct behavior:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - Use PreEDA feature registry to identify true dropped physical sensors.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - Use the PreEDA dropped_sensors parquet to get the actual dropped sensor values.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - Use silver_eda_dataframe only for the profiled state masks.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - sensor_15 is all-null and should not be generated.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - sensor_50 should be generated, then missingness-masked.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `import re` | Imports a dependency or project helper used by later cells. |
| `DROPPED_SENSOR_REGISTRY_PATH = ( GENERATOR_INPUT_DIR / "dropped_sensor_registry.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Load dropped sensor metadata from PreEDA feature registry` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `if "FEATURE_REGISTRY_PATH" not in globals(): raise NameError( "FEATURE_REGISTRY_PATH is not defined. " "Run the feature registry / parent Silver setup cells before this cell." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `with open(FEATURE_REGISTRY_PATH, "r", encoding="utf-8") as f: feature_registry_payload = json.load(f)` | Controls validation, iteration, file handling, or error handling for this step. |
| `missingness_quarantine = ( feature_registry_payload .get("feature_info", {}) .get("missingness_quarantine", {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `dropped_features_from_registry = [ str(sensor).strip() for sensor in missingness_quarantine.get("dropped_features", []) if str(sensor).strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `dropped_missing_pct = { str(sensor): float(value) for sensor, value in (missingness_quarantine.get("dropped_missing_pct", {}) or {}).items()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `drop_reasons = { str(sensor): str(reason) for sensor, reason in (missingness_quarantine.get("drop_reasons", {}) or {}).items()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `RAW_SENSOR_PATTERN = re.compile(r"^sensor_\d{2}$")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `dropped_physical_sensors = [ sensor for sensor in dropped_features_from_registry if RAW_SENSOR_PATTERN.match(sensor)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Generate only dropped sensors that still contain real values.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# sensor_15 is all-null, so it should be restored later as NaN, not generated.` | Documents the purpose or boundary of the surrounding notebook step. |
| `profile_eligible_dropped_sensors = [ sensor for sensor in dropped_physical_sensors if drop_reasons.get(sensor) != "all_null"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Dropped physical sensors from registry:", dropped_physical_sensors)` | Displays a notebook-facing result for inspection. |
| `print("Profile-eligible dropped sensors:", profile_eligible_dropped_sensors)` | Displays a notebook-facing result for inspection. |
| `print("Dropped missing pct:", dropped_missing_pct)` | Displays a notebook-facing result for inspection. |
| `print("Drop reasons:", drop_reasons)` | Displays a notebook-facing result for inspection. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Resolve PreEDA dropped-sensors parquet` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `def resolve_existing_path_from_candidates( candidates: list[Path], *, label: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> Path: for path in candidates: if path.exists(): return path raise FileNotFoundError( f"Could not resolve {label}. Tried:\n" + "\n".join(str(path) for path in candidates) )` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DROPPED_SENSORS_PARQUET_PATH = resolve_existing_path_from_candidates( [ # Most likely locations ARTIFACTS_ROOT / "silver" / DATASET_NAME / "pump__silver_preeda__dropped_sensors.par` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `dropped_sensors_df = pd.read_parquet( DROPPED_SENSORS_PARQUET_PATH, engine="auto",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("DROPPED_SENSORS_PARQUET_PATH:", DROPPED_SENSORS_PARQUET_PATH)` | Displays a notebook-facing result for inspection. |
| `print("dropped_sensors_df shape:", dropped_sensors_df.shape)` | Displays a notebook-facing result for inspection. |
| `print("dropped_sensors_df columns:", dropped_sensors_df.columns.tolist())` | Displays a notebook-facing result for inspection. |
| `missing_source_columns = [ sensor for sensor in profile_eligible_dropped_sensors if sensor not in dropped_sensors_df.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_source_columns: raise ValueError( "These profile-eligible dropped sensors are missing from dropped_sensors_df: " f"{missing_source_columns}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Align dropped sensor values to profiled dataframe rows` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `if len(dropped_sensors_df) != len(silver_eda_dataframe): raise ValueError( "Cannot safely align dropped_sensors_df to silver_eda_dataframe by row order. " f"dropped_sensors_df rows` | Controls validation, iteration, file handling, or error handling for this step. |
| `dropped_profile_source_df = dropped_sensors_df[profile_eligible_dropped_sensors].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `dropped_profile_source_df.index = silver_eda_dataframe.index` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Dropped profile source columns:", dropped_profile_source_df.columns.tolist())` | Displays a notebook-facing result for inspection. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `33 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 39 — Record traceability output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abs`
- `add`
- `append`
- `apply`
- `axis`
- `Built`
- `clean_vs_abnormal_shift`
- `clean_vs_abnormal_shift_zish`
- `clean_vs_contaminated_shift`
- `clean_vs_contaminated_shift_zish`
- `clean_vs_recovery_shift`
- `clean_vs_recovery_shift_zish`
- `coerce`
- `DataFrame`
- `def`
- `df`
- `drop`
- `dropna`
- `else`
- `errors`

### Outputs

- `abnormal_medians`
- `ascending`
- `build_state_median_lookup`
- `clean_medians`
- `clean_std_df`
- `columns`
- `contaminated_medians`
- `data`
- `hotspot_sensor_df`
- `kind`
- `logger`
- `medians`
- `message`
- `recovery_medians`
- `rows`
- `scale`
- `sensor_cols`
- `state_col`
- `state_median_df`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Hotspot sensor summary`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `def build_state_median_lookup( df: pd.DataFrame, *, sensor_cols: Sequence[str], state_col: str,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: rows = [] for state_value, state_df in df.groupby(state_col, dropna=False): medians = state_df[list(sensor_cols)].apply(pd.to_numeric, errors="coerce").median(ax`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `state_median_df = build_state_median_lookup( silver_eda_dataframe, sensor_cols=FEATURE_COLUMNS, state_col=STATE_COL_PROFILED,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `clean_medians = ( state_median_df[state_median_df["state_value"] == PROFILED_NORMAL_CLEAN_VALUE] .rename(columns={"median_value": "median_clean"}) [["sensor", "median_clean"]]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `contaminated_medians = ( state_median_df[state_median_df["state_value"] == PROFILED_NORMAL_CONTAMINATED_VALUE] .rename(columns={"median_value": "median_contaminated"}) [["sensor", `: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `abnormal_medians = ( state_median_df[state_median_df["state_value"] == PROFILED_ABNORMAL_VALUE] .rename(columns={"median_value": "median_abnormal"}) [["sensor", "median_abnormal"]]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `abs`
- `add`
- `append`
- `apply`
- `build_state_median_lookup`
- `DataFrame`
- `display`
- `groupby`
- `head`
- `items`
- `max`
- `median`
- `merge`
- `notna`
- `rename`
- `replace`
- `reset_index`
- `sort_values`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Hotspot sensor summary` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `def build_state_median_lookup( df: pd.DataFrame, *, sensor_cols: Sequence[str], state_col: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: rows = [] for state_value, state_df in df.groupby(state_col, dropna=False): medians = state_df[list(sensor_cols)].apply(pd.to_numeric, errors="coerce").median(ax` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `state_median_df = build_state_median_lookup( silver_eda_dataframe, sensor_cols=FEATURE_COLUMNS, state_col=STATE_COL_PROFILED,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `clean_medians = ( state_median_df[state_median_df["state_value"] == PROFILED_NORMAL_CLEAN_VALUE] .rename(columns={"median_value": "median_clean"}) [["sensor", "median_clean"]]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `contaminated_medians = ( state_median_df[state_median_df["state_value"] == PROFILED_NORMAL_CONTAMINATED_VALUE] .rename(columns={"median_value": "median_contaminated"}) [["sensor", ` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `abnormal_medians = ( state_median_df[state_median_df["state_value"] == PROFILED_ABNORMAL_VALUE] .rename(columns={"median_value": "median_abnormal"}) [["sensor", "median_abnormal"]]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `recovery_medians = ( state_median_df[state_median_df["state_value"] == PROFILED_RECOVERY_VALUE] .rename(columns={"median_value": "median_recovery"}) [["sensor", "median_recovery"]]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `clean_std_df = numeric_profile_df[["feature", "std", "missing_count", "row_count"]].rename( columns={ "feature": "sensor", "std": "std_full_dataset", "missing_count": "missing_coun` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `hotspot_sensor_df = clean_std_df.merge(clean_medians, on="sensor", how="left")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `hotspot_sensor_df = hotspot_sensor_df.merge(contaminated_medians, on="sensor", how="left")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `hotspot_sensor_df = hotspot_sensor_df.merge(abnormal_medians, on="sensor", how="left")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `hotspot_sensor_df = hotspot_sensor_df.merge(recovery_medians, on="sensor", how="left")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `hotspot_sensor_df["clean_vs_contaminated_shift"] = ( hotspot_sensor_df["median_contaminated"] - hotspot_sensor_df["median_clean"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `hotspot_sensor_df["clean_vs_abnormal_shift"] = ( hotspot_sensor_df["median_abnormal"] - hotspot_sensor_df["median_clean"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `hotspot_sensor_df["clean_vs_recovery_shift"] = ( hotspot_sensor_df["median_recovery"] - hotspot_sensor_df["median_clean"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `scale = hotspot_sensor_df["std_full_dataset"].replace(0, np.nan)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `hotspot_sensor_df["clean_vs_contaminated_shift_zish"] = ( hotspot_sensor_df["clean_vs_contaminated_shift"].abs() / scale` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `hotspot_sensor_df["clean_vs_abnormal_shift_zish"] = ( hotspot_sensor_df["clean_vs_abnormal_shift"].abs() / scale` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `hotspot_sensor_df["clean_vs_recovery_shift_zish"] = ( hotspot_sensor_df["clean_vs_recovery_shift"].abs() / scale` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `hotspot_sensor_df["hotspot_score"] = hotspot_sensor_df[ [ "clean_vs_contaminated_shift_zish", "clean_vs_abnormal_shift_zish", "clean_vs_recovery_shift_zish", ]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `].max(axis=1, skipna=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `hotspot_sensor_df = hotspot_sensor_df.sort_values( ["hotspot_score", "sensor"], ascending=[False, True],` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).reset_index(drop=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="hotspot_sensor_summary", message="Built hotspot sensor summary from profiled-state median shifts.", data={ "top_hotspot_sensors": hotspot_sensor_df.h` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(hotspot_sensor_df.head(25))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 40 — Record traceability output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `artifacts`
- `CORRELATION_ARTIFACT_DIR`
- `csv`
- `dwell`
- `fault`
- `hotspot`
- `hotspot_sensor_df`
- `index`
- `ledger`
- `profiled_dwell_summary_df`
- `profiled_state_dwell_summary`
- `profiled_state_transition_counts`
- `profiled_transition_counts_df`
- `Save`
- `save_midstage_artifacts`
- `Saved`
- `sensor_hotspot_summary_profiled_states`
- `source_dwell_summary_df`
- `source_state_dwell_summary`

### Outputs

- `data`
- `hotspot_sensor_path`
- `kind`
- `logger`
- `message`
- `profiled_dwell_summary_path`
- `profiled_transition_counts_path`
- `source_dwell_summary_path`
- `source_transition_counts_path`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Save subsystem / fault / hotspot artifacts`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `hotspot_sensor_path = CORRELATION_ARTIFACT_DIR / "sensor_hotspot_summary_profiled_states.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `hotspot_sensor_df.to_csv(hotspot_sensor_path, index=False)`: Writes an artifact or output used for review or downstream notebooks.
- `source_transition_counts_path = CORRELATION_ARTIFACT_DIR / "source_state_transition_counts.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `profiled_transition_counts_path = CORRELATION_ARTIFACT_DIR / "profiled_state_transition_counts.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `source_transition_counts_df.to_csv(source_transition_counts_path, index=False)`: Writes an artifact or output used for review or downstream notebooks.
- `profiled_transition_counts_df.to_csv(profiled_transition_counts_path, index=False)`: Writes an artifact or output used for review or downstream notebooks.
- `source_dwell_summary_path = CORRELATION_ARTIFACT_DIR / "source_state_dwell_summary.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `profiled_dwell_summary_path = CORRELATION_ARTIFACT_DIR / "profiled_state_dwell_summary.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `source_dwell_summary_df.to_csv(source_dwell_summary_path, index=False)`: Writes an artifact or output used for review or downstream notebooks.

Important functions or methods detected:
- `add`
- `to_csv`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Save subsystem / fault / hotspot artifacts` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `hotspot_sensor_path = CORRELATION_ARTIFACT_DIR / "sensor_hotspot_summary_profiled_states.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `hotspot_sensor_df.to_csv(hotspot_sensor_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `source_transition_counts_path = CORRELATION_ARTIFACT_DIR / "source_state_transition_counts.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `profiled_transition_counts_path = CORRELATION_ARTIFACT_DIR / "profiled_state_transition_counts.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `source_transition_counts_df.to_csv(source_transition_counts_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `profiled_transition_counts_df.to_csv(profiled_transition_counts_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `source_dwell_summary_path = CORRELATION_ARTIFACT_DIR / "source_state_dwell_summary.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `profiled_dwell_summary_path = CORRELATION_ARTIFACT_DIR / "profiled_state_dwell_summary.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `source_dwell_summary_df.to_csv(source_dwell_summary_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `profiled_dwell_summary_df.to_csv(profiled_dwell_summary_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `ledger.add( kind="step", step="save_midstage_artifacts", message="Saved subsystem, transition, dwell, and hotspot artifacts.", data={ "hotspot_sensor_path": str(hotspot_sensor_path` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Saved:", hotspot_sensor_path)` | Displays a notebook-facing result for inspection. |
| `print("Saved:", source_transition_counts_path)` | Displays a notebook-facing result for inspection. |
| `print("Saved:", profiled_transition_counts_path)` | Displays a notebook-facing result for inspection. |
| `print("Saved:", source_dwell_summary_path)` | Displays a notebook-facing result for inspection. |
| `print("Saved:", profiled_dwell_summary_path)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 41 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `Any`
- `apply`
- `axis`
- `Built`
- `coerce`
- `comparison`
- `copy`
- `DataFrame`
- `ddof`
- `def`
- `df`
- `else`
- `empty`
- `errors`
- `feature_cols`
- `FEATURE_COLUMNS`
- `high`
- `isna`
- `ledger`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `numeric_df`
- `per_sensor_missing_pct`
- `per_sensor_std`
- `profiled_compare_summary_df`
- `state_df`
- `step`
- `summarize_profiled_state`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# State comparison summary`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `def summarize_profiled_state( df: pd.DataFrame, *, feature_cols: Sequence[str], state_value: str,`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, Any]: state_df = df[df[STATE_COL_PROFILED] == state_value].copy() numeric_df = state_df[list(feature_cols)].apply(pd.to_numeric, errors="coerce") per_sensor_missing_`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `profiled_compare_summary_df = pd.DataFrame( [ summarize_profiled_state(silver_eda_dataframe, feature_cols=FEATURE_COLUMNS, state_value=PROFILED_NORMAL_CLEAN_VALUE), summarize_profi`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="profiled_state_compare_summary", message="Built high-level profiled-state comparison summary.", data={"profiled_state_compare_summary": profiled_comp`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(profiled_compare_summary_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `apply`
- `copy`
- `DataFrame`
- `display`
- `isna`
- `max`
- `mean`
- `median`
- `std`
- `summarize_profiled_state`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# State comparison summary` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `def summarize_profiled_state( df: pd.DataFrame, *, feature_cols: Sequence[str], state_value: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, Any]: state_df = df[df[STATE_COL_PROFILED] == state_value].copy() numeric_df = state_df[list(feature_cols)].apply(pd.to_numeric, errors="coerce") per_sensor_missing_` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `profiled_compare_summary_df = pd.DataFrame( [ summarize_profiled_state(silver_eda_dataframe, feature_cols=FEATURE_COLUMNS, state_value=PROFILED_NORMAL_CLEAN_VALUE), summarize_profi` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="profiled_state_compare_summary", message="Built high-level profiled-state comparison summary.", data={"profiled_state_compare_summary": profiled_comp` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(profiled_compare_summary_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 42 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abs`
- `add`
- `Any`
- `any`
- `append`
- `Built`
- `clean_center`
- `clean_scale`
- `coerce`
- `comparison`
- `copy`
- `CORRELATION_ARTIFACT_DIR`
- `csv`
- `DataFrame`
- `def`
- `df`
- `drop`
- `dropna`
- `else`
- `errors`

### Outputs

- `ascending`
- `clean_df`
- `clean_series`
- `clean_value`
- `data`
- `feature_cols`
- `kind`
- `logger`
- `message`
- `na_position`
- `robust_state_compare_df`
- `robust_state_compare_path`
- `robust_state_compare_top_df`
- `robust_state_comparison_vs_clean`
- `state_center`
- `state_col`
- `state_q05`
- `state_q95`
- `state_series`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Robust state comparison vs normal_clean`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `def robust_state_comparison_vs_clean( df: pd.DataFrame, *, feature_cols: Sequence[str], state_col: str = "machine_status__profiled", clean_value: str = "normal_clean",`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: rows: list[dict[str, Any]] = [] clean_df = df[df[state_col] == clean_value].copy() for feature_col in feature_cols: clean_series = pd.to_numeric(clean_df[feature`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `robust_state_compare_df = robust_state_comparison_vs_clean( silver_eda_dataframe, feature_cols=FEATURE_COLUMNS, state_col=STATE_COL_PROFILED, clean_value=PROFILED_NORMAL_CLEAN_VALU`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `robust_state_compare_top_df = ( robust_state_compare_df[ robust_state_compare_df["state_value"].isin( [ PROFILED_NORMAL_CONTAMINATED_VALUE, PROFILED_ABNORMAL_VALUE, PROFILED_RECOVE`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `robust_state_compare_path = CORRELATION_ARTIFACT_DIR / "robust_state_comparison_vs_normal_clean.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `robust_state_compare_df.to_csv(robust_state_compare_path, index=False)`: Writes an artifact or output used for review or downstream notebooks.
- `ledger.add( kind="step", step="robust_state_comparison", message="Built robust state comparison vs normal_clean.", data={ "robust_state_compare_path": str(robust_state_compare_path`: Records or exports ledger information for stage-level traceability.

Important functions or methods detected:
- `abs`
- `add`
- `any`
- `append`
- `copy`
- `DataFrame`
- `display`
- `groupby`
- `head`
- `isin`
- `median`
- `notna`
- `quantile`
- `reset_index`
- `robust_center_scale`
- `robust_state_comparison_vs_clean`
- `sort_values`
- `to_csv`
- `to_dict`
- `to_numeric`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Robust state comparison vs normal_clean` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `def robust_state_comparison_vs_clean( df: pd.DataFrame, *, feature_cols: Sequence[str], state_col: str = "machine_status__profiled", clean_value: str = "normal_clean",` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: rows: list[dict[str, Any]] = [] clean_df = df[df[state_col] == clean_value].copy() for feature_col in feature_cols: clean_series = pd.to_numeric(clean_df[feature` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `robust_state_compare_df = robust_state_comparison_vs_clean( silver_eda_dataframe, feature_cols=FEATURE_COLUMNS, state_col=STATE_COL_PROFILED, clean_value=PROFILED_NORMAL_CLEAN_VALU` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `robust_state_compare_top_df = ( robust_state_compare_df[ robust_state_compare_df["state_value"].isin( [ PROFILED_NORMAL_CONTAMINATED_VALUE, PROFILED_ABNORMAL_VALUE, PROFILED_RECOVE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `robust_state_compare_path = CORRELATION_ARTIFACT_DIR / "robust_state_comparison_vs_normal_clean.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `robust_state_compare_df.to_csv(robust_state_compare_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `ledger.add( kind="step", step="robust_state_comparison", message="Built robust state comparison vs normal_clean.", data={ "robust_state_compare_path": str(robust_state_compare_path` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Saved:", robust_state_compare_path)` | Displays a notebook-facing result for inspection. |
| `display(robust_state_compare_top_df.head(25))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 43 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Clean`
- `coolwarm`
- `copy`
- `Correlation`
- `correlation_matrix_normal_clean`
- `drop_duplicates`
- `dropna`
- `else`
- `feature`
- `figsize`
- `figure`
- `found`
- `head`
- `Heatmap`
- `heatmap`
- `loc`
- `No`
- `Normal`
- `robust_state_compare_top_df`
- `Sensors`

### Outputs

- `cbar_kws`
- `center`
- `cmap`
- `linewidths`
- `square`
- `TOP_HEATMAP_SENSOR_COUNT`
- `top_shift_corr_matrix`
- `top_shifted_sensors`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Correlation heatmap for top shifted sensors`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `TOP_HEATMAP_SENSOR_COUNT = 20`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `top_shifted_sensors = ( robust_state_compare_top_df["feature"] .dropna() .drop_duplicates() .head(TOP_HEATMAP_SENSOR_COUNT) .tolist()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if not top_shifted_sensors: print("No top shifted sensors found; skipping heatmap.")`: Displays a notebook-facing result for inspection.
- `else: top_shift_corr_matrix = correlation_matrix_normal_clean.loc[ top_shifted_sensors, top_shifted_sensors ].copy() plt.figure(figsize=(14, 10)) sns.heatmap( top_shift_corr_matrix`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `copy`
- `drop_duplicates`
- `dropna`
- `figure`
- `head`
- `heatmap`
- `show`
- `tight_layout`
- `title`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Correlation heatmap for top shifted sensors` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `TOP_HEATMAP_SENSOR_COUNT = 20` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `top_shifted_sensors = ( robust_state_compare_top_df["feature"] .dropna() .drop_duplicates() .head(TOP_HEATMAP_SENSOR_COUNT) .tolist()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if not top_shifted_sensors: print("No top shifted sensors found; skipping heatmap.")` | Displays a notebook-facing result for inspection. |
| `else: top_shift_corr_matrix = correlation_matrix_normal_clean.loc[ top_shifted_sensors, top_shifted_sensors ].copy() plt.figure(figsize=(14, 10)) sns.heatmap( top_shift_corr_matrix` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 44 — Record traceability output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__distribution_by_profiled_state`
- `add`
- `bbox_inches`
- `Built`
- `by`
- `close`
- `coerce`
- `comparison`
- `comparisons`
- `continue`
- `copy`
- `distribution`
- `dpi`
- `drop_duplicates`
- `dropna`
- `else`
- `empty`
- `errors`
- `exist_ok`
- `f`

### Outputs

- `common_norm`
- `data`
- `distribution_plot_dir`
- `fill`
- `hue`
- `kind`
- `logger`
- `message`
- `out_path`
- `plot_df`
- `plot_states`
- `step`
- `TOP_DISTRIBUTION_SENSOR_COUNT`
- `top_distribution_sensors`
- `x`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Feature distribution comparisons for top shifted sensors`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `TOP_DISTRIBUTION_SENSOR_COUNT = 6`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `top_distribution_sensors = ( robust_state_compare_top_df["feature"] .dropna() .drop_duplicates() .head(TOP_DISTRIBUTION_SENSOR_COUNT) .tolist()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if not top_distribution_sensors: print("No top shifted sensors found; skipping distribution plots.")`: Displays a notebook-facing result for inspection.
- `else: distribution_plot_dir = DISTRIBUTION_PLOT_DIR distribution_plot_dir.mkdir(parents=True, exist_ok=True) plot_states = [ PROFILED_NORMAL_CLEAN_VALUE, PROFILED_NORMAL_CONTAMINAT`: Writes an artifact or output used for review or downstream notebooks.
- `ledger.add( kind="step", step="feature_distribution_comparisons", message="Built feature distribution comparison plots for top shifted sensors.", data={"top_distribution_sensors": `: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `close`
- `copy`
- `drop_duplicates`
- `dropna`
- `figure`
- `head`
- `isin`
- `kdeplot`
- `mkdir`
- `savefig`
- `show`
- `tight_layout`
- `title`
- `to_numeric`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Feature distribution comparisons for top shifted sensors` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `TOP_DISTRIBUTION_SENSOR_COUNT = 6` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `top_distribution_sensors = ( robust_state_compare_top_df["feature"] .dropna() .drop_duplicates() .head(TOP_DISTRIBUTION_SENSOR_COUNT) .tolist()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if not top_distribution_sensors: print("No top shifted sensors found; skipping distribution plots.")` | Displays a notebook-facing result for inspection. |
| `else: distribution_plot_dir = DISTRIBUTION_PLOT_DIR distribution_plot_dir.mkdir(parents=True, exist_ok=True) plot_states = [ PROFILED_NORMAL_CLEAN_VALUE, PROFILED_NORMAL_CONTAMINAT` | Writes an artifact or output used for review or downstream notebooks. |
| `ledger.add( kind="step", step="feature_distribution_comparisons", message="Built feature distribution comparison plots for top shifted sensors.", data={"top_distribution_sensors": ` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: plot/image artifact.

## Code Cell 45 — Record traceability output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__timeline_overlay`
- `across`
- `add`
- `bbox_inches`
- `break`
- `Built`
- `by`
- `close`
- `coerce`
- `column`
- `columns`
- `continue`
- `copy`
- `dpi`
- `drop_duplicates`
- `dropna`
- `else`
- `empty`
- `errors`
- `event_step`

### Outputs

- `alpha`
- `data`
- `hue`
- `kind`
- `logger`
- `message`
- `out_path`
- `plot_df`
- `s`
- `step`
- `time_col`
- `timeline_df`
- `timeline_plot_dir`
- `timeline_sensor_cols`
- `TOP_TIMELINE_SENSOR_COUNT`
- `x`
- `y`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Overlay top features across time`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `TOP_TIMELINE_SENSOR_COUNT = 4`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `timeline_sensor_cols = ( robust_state_compare_top_df["feature"] .dropna() .drop_duplicates() .head(TOP_TIMELINE_SENSOR_COUNT) .tolist()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `time_col = "time_index" if "time_index" in silver_eda_dataframe.columns else None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if time_col is None: for fallback_col in ["event_step", "event_time"]: if fallback_col in silver_eda_dataframe.columns: time_col = fallback_col break`: Controls validation, iteration, file handling, or error handling for this step.
- `if time_col is None: print("No time/order column found; skipping overlay plots.")`: Displays a notebook-facing result for inspection.
- `else: timeline_plot_dir = TIMELINE_OVERLAY_DIR timeline_plot_dir.mkdir(parents=True, exist_ok=True) timeline_df = silver_eda_dataframe.copy() # Keep plot size manageable if len(tim`: Writes an artifact or output used for review or downstream notebooks.
- `ledger.add( kind="step", step="timeline_overlays", message="Built timeline overlay plots for top shifted sensors.", data={"timeline_sensor_cols": timeline_sensor_cols, "time_col": `: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `close`
- `copy`
- `drop_duplicates`
- `dropna`
- `figure`
- `head`
- `mkdir`
- `sample`
- `savefig`
- `scatterplot`
- `show`
- `sort_values`
- `tight_layout`
- `title`
- `to_numeric`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Overlay top features across time` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `TOP_TIMELINE_SENSOR_COUNT = 4` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `timeline_sensor_cols = ( robust_state_compare_top_df["feature"] .dropna() .drop_duplicates() .head(TOP_TIMELINE_SENSOR_COUNT) .tolist()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `time_col = "time_index" if "time_index" in silver_eda_dataframe.columns else None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if time_col is None: for fallback_col in ["event_step", "event_time"]: if fallback_col in silver_eda_dataframe.columns: time_col = fallback_col break` | Controls validation, iteration, file handling, or error handling for this step. |
| `if time_col is None: print("No time/order column found; skipping overlay plots.")` | Displays a notebook-facing result for inspection. |
| `else: timeline_plot_dir = TIMELINE_OVERLAY_DIR timeline_plot_dir.mkdir(parents=True, exist_ok=True) timeline_df = silver_eda_dataframe.copy() # Keep plot size manageable if len(tim` | Writes an artifact or output used for review or downstream notebooks. |
| `ledger.add( kind="step", step="timeline_overlays", message="Built timeline overlay plots for top shifted sensors.", data={"timeline_sensor_cols": timeline_sensor_cols, "time_col": ` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: plot/image artifact.

## Code Cell 46 — Record traceability output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `abnormal_onset_position`
- `add`
- `aligned`
- `aligned_onset_shape`
- `aligned_onset_windows`
- `aligned_rows`
- `aligned_step`
- `alignment`
- `Any`
- `append`
- `arange`
- `build`
- `Built`
- `cast`
- `column`
- `columns`
- `concat`
- `continue`
- `copy`

### Outputs

- `abnormal_positions`
- `ALIGN_POST_STEPS`
- `ALIGN_PRE_STEPS`
- `aligned_onset_df`
- `axis`
- `data`
- `end_pos`
- `episode_df`
- `first_abnormal_pos`
- `ignore_index`
- `kind`
- `logger`
- `message`
- `original_index_col`
- `start_pos`
- `step`
- `window_df`
- `work_df`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Detect abnormal onsets and build aligned window rows`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `ALIGN_PRE_STEPS = 50`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ALIGN_POST_STEPS = 50`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if "meta__episode_id" not in silver_eda_dataframe.columns: print("No meta__episode_id column found; skipping onset alignment.")`: Displays a notebook-facing result for inspection.
- `else: if time_col is None: print("No order/time column found; skipping onset alignment.") else: #aligned_rows: list[dict[str, Any]] = [] aligned_rows: list[pd.DataFrame] = [] work_`: Records or exports ledger information for stage-level traceability.

Important functions or methods detected:
- `add`
- `append`
- `arange`
- `cast`
- `concat`
- `copy`
- `DataFrame`
- `display`
- `eq`
- `flatnonzero`
- `groupby`
- `head`
- `max`
- `min`
- `reset_index`
- `sort_values`
- `to_numpy`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Detect abnormal onsets and build aligned window rows` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `ALIGN_PRE_STEPS = 50` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ALIGN_POST_STEPS = 50` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if "meta__episode_id" not in silver_eda_dataframe.columns: print("No meta__episode_id column found; skipping onset alignment.")` | Displays a notebook-facing result for inspection. |
| `else: if time_col is None: print("No order/time column found; skipping onset alignment.") else: #aligned_rows: list[dict[str, Any]] = [] aligned_rows: list[pd.DataFrame] = [] work_` | Records or exports ledger information for stage-level traceability. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 47 — Run validation guardrails

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abs`
- `add`
- `Agglomerative`
- `AgglomerativeClustering`
- `average`
- `by`
- `clean`
- `Cluster`
- `cluster_`
- `Clustered`
- `clustering`
- `columns`
- `copy`
- `correlation`
- `CORRELATION_ARTIFACT_DIR`
- `correlation_matrix_normal_clean`
- `csv`
- `DataFrame`
- `distance`
- `drop`

### Outputs

- `cluster_labels`
- `clustering_model`
- `corr_for_cluster`
- `data`
- `distance_matrix`
- `FEATURE_CLUSTER_COUNT`
- `feature_cluster_map_df`
- `feature_cluster_map_path`
- `kind`
- `linkage`
- `logger`
- `message`
- `metric`
- `n_clusters`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Cluster numeric features by correlation distance`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `FEATURE_CLUSTER_COUNT = 8`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if correlation_matrix_normal_clean.empty: print("Normal-clean correlation matrix is empty; skipping feature clustering.")`: Displays a notebook-facing result for inspection.
- `else: corr_for_cluster = correlation_matrix_normal_clean.copy().fillna(0.0) distance_matrix = 1.0 - corr_for_cluster.abs() # Agglomerative clustering on the precomputed distance ma`: Records or exports ledger information for stage-level traceability.

Important functions or methods detected:
- `abs`
- `add`
- `AgglomerativeClustering`
- `copy`
- `DataFrame`
- `display`
- `fillna`
- `fit_predict`
- `head`
- `min`
- `nunique`
- `reset_index`
- `sort_values`
- `to_csv`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Cluster numeric features by correlation distance` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `FEATURE_CLUSTER_COUNT = 8` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if correlation_matrix_normal_clean.empty: print("Normal-clean correlation matrix is empty; skipping feature clustering.")` | Displays a notebook-facing result for inspection. |
| `else: corr_for_cluster = correlation_matrix_normal_clean.copy().fillna(0.0) distance_matrix = 1.0 - corr_for_cluster.abs() # Agglomerative clustering on the precomputed distance ma` | Records or exports ledger information for stage-level traceability. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 48 — Run validation guardrails

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__aligned_onset_median_curve`
- `abnormal`
- `add`
- `Aligned`
- `aligned`
- `aligned_onset_df`
- `ALIGNED_ONSET_PLOT_DIR`
- `aligned_onset_sensor_curves`
- `aligned_step`
- `anomaly`
- `axvline`
- `bbox_inches`
- `Built`
- `close`
- `coerce`
- `continue`
- `copy`
- `curve`
- `curves`
- `dataframe`

### Outputs

- `aligned_plot_dir`
- `aligned_sensor_cols`
- `data`
- `hue`
- `kind`
- `logger`
- `message`
- `out_path`
- `plot_df`
- `step`
- `summary_df`
- `x`
- `y`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Aligned anomaly-onset sensor median curves`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `if "aligned_onset_df" not in globals() or aligned_onset_df.empty: print("No aligned onset dataframe found; skipping aligned median curves.")`: Displays a notebook-facing result for inspection.
- `else: aligned_plot_dir = ALIGNED_ONSET_PLOT_DIR aligned_plot_dir.mkdir(parents=True, exist_ok=True) aligned_sensor_cols = ( robust_state_compare_top_df["feature"] .dropna() .drop_d`: Records or exports ledger information for stage-level traceability.

Important functions or methods detected:
- `add`
- `axvline`
- `close`
- `copy`
- `drop_duplicates`
- `dropna`
- `figure`
- `globals`
- `groupby`
- `head`
- `lineplot`
- `median`
- `mkdir`
- `reset_index`
- `savefig`
- `show`
- `tight_layout`
- `title`
- `to_numeric`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Aligned anomaly-onset sensor median curves` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `if "aligned_onset_df" not in globals() or aligned_onset_df.empty: print("No aligned onset dataframe found; skipping aligned median curves.")` | Displays a notebook-facing result for inspection. |
| `else: aligned_plot_dir = ALIGNED_ONSET_PLOT_DIR aligned_plot_dir.mkdir(parents=True, exist_ok=True) aligned_sensor_cols = ( robust_state_compare_top_df["feature"] .dropna() .drop_d` | Records or exports ledger information for stage-level traceability. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: plot/image artifact.

## Code Cell 49 — Compute the correlation matrix

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `apply`
- `build`
- `Built`
- `by`
- `coerce`
- `copy`
- `else`
- `errors`
- `explained_variance_ratio_`
- `FEATURE_COLUMNS`
- `figsize`
- `figure`
- `fillna`
- `fit_transform`
- `impute`
- `ledger`
- `median`
- `Median`
- `n_components`

### Outputs

- `alpha`
- `data`
- `hue`
- `kind`
- `logger`
- `message`
- `pca_model`
- `PCA_N_COMPONENTS`
- `pca_plot_df`
- `PCA_SAMPLE_N`
- `PCA_SCALER`
- `pca_source_df`
- `s`
- `scaler`
- `step`
- `x`
- `X_pca`
- `X_pca_components`
- `X_scaled`
- `y`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# PCA build`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `PCA_SAMPLE_N = 20000`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PCA_N_COMPONENTS = 2`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PCA_SCALER = "robust" # "robust" or "standard"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `pca_source_df = silver_eda_dataframe[[STATE_COL_PROFILED] + FEATURE_COLUMNS].copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if len(pca_source_df) > PCA_SAMPLE_N: pca_source_df = pca_source_df.sample(PCA_SAMPLE_N, random_state=42).copy()`: Controls validation, iteration, file handling, or error handling for this step.
- `X_pca = pca_source_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Median impute for PCA prep`: Documents the purpose or boundary of the surrounding notebook step.
- `X_pca = X_pca.fillna(X_pca.median(numeric_only=True))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if PCA_SCALER == "standard": scaler = StandardScaler()`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `add`
- `apply`
- `copy`
- `figure`
- `fillna`
- `fit_transform`
- `median`
- `PCA`
- `RobustScaler`
- `sample`
- `scatterplot`
- `show`
- `StandardScaler`
- `tight_layout`
- `title`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# PCA build` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `PCA_SAMPLE_N = 20000` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PCA_N_COMPONENTS = 2` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PCA_SCALER = "robust" # "robust" or "standard"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `pca_source_df = silver_eda_dataframe[[STATE_COL_PROFILED] + FEATURE_COLUMNS].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(pca_source_df) > PCA_SAMPLE_N: pca_source_df = pca_source_df.sample(PCA_SAMPLE_N, random_state=42).copy()` | Controls validation, iteration, file handling, or error handling for this step. |
| `X_pca = pca_source_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Median impute for PCA prep` | Documents the purpose or boundary of the surrounding notebook step. |
| `X_pca = X_pca.fillna(X_pca.median(numeric_only=True))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if PCA_SCALER == "standard": scaler = StandardScaler()` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: scaler = RobustScaler()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `X_scaled = scaler.fit_transform(X_pca)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `pca_model = PCA(n_components=PCA_N_COMPONENTS, random_state=42)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `X_pca_components = pca_model.fit_transform(X_scaled)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `pca_plot_df = pca_source_df[[STATE_COL_PROFILED]].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `pca_plot_df["pc1"] = X_pca_components[:, 0]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `pca_plot_df["pc2"] = X_pca_components[:, 1]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="pca_build", message="Built PCA projection for profiled states.", data={ "pca_sample_n": int(len(pca_plot_df)), "pca_explained_variance_ratio": pca_mo` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.figure(figsize=(10, 8))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `sns.scatterplot( data=pca_plot_df, x="pc1", y="pc2", hue=STATE_COL_PROFILED, s=18, alpha=0.6,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.title("PCA projection by profiled state")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.tight_layout()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.show()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 50 — Compute the correlation matrix

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abs`
- `add`
- `ascending`
- `axis`
- `Built`
- `component`
- `components_`
- `csv`
- `cumsum`
- `cumulative_explained_variance_ratio`
- `DataFrame`
- `diagnostics`
- `drop`
- `exist_ok`
- `explained_variance_ratio`
- `explained_variance_ratio_`
- `f`
- `feature`
- `FEATURE_COLUMNS`
- `head`

### Outputs

- `columns`
- `data`
- `index`
- `kind`
- `loading_df`
- `logger`
- `message`
- `pca_diag_dir`
- `pca_explained_variance_df`
- `pca_explained_variance_path`
- `pca_loading_path`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# PCA diagnostics`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `pca_explained_variance_df = pd.DataFrame( { "component": [f"PC{i+1}" for i in range(len(pca_model.explained_variance_ratio_))], "explained_variance_ratio": pca_model.explained_vari`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `loading_df = pd.DataFrame( pca_model.components_.T, index=FEATURE_COLUMNS, columns=[f"PC{i+1}" for i in range(pca_model.n_components_)],`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `).reset_index(names="feature")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `loading_df["pc1_abs_loading"] = loading_df["PC1"].abs()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `loading_df["pc2_abs_loading"] = loading_df["PC2"].abs()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `loading_df["max_abs_loading"] = loading_df[["pc1_abs_loading", "pc2_abs_loading"]].max(axis=1)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `loading_df = loading_df.sort_values("max_abs_loading", ascending=False).reset_index(drop=True)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `pca_diag_dir = PCA_ARTIFACT_DIR`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `abs`
- `add`
- `cumsum`
- `DataFrame`
- `display`
- `head`
- `max`
- `mkdir`
- `range`
- `reset_index`
- `sort_values`
- `to_csv`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# PCA diagnostics` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `pca_explained_variance_df = pd.DataFrame( { "component": [f"PC{i+1}" for i in range(len(pca_model.explained_variance_ratio_))], "explained_variance_ratio": pca_model.explained_vari` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `loading_df = pd.DataFrame( pca_model.components_.T, index=FEATURE_COLUMNS, columns=[f"PC{i+1}" for i in range(pca_model.n_components_)],` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).reset_index(names="feature")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `loading_df["pc1_abs_loading"] = loading_df["PC1"].abs()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `loading_df["pc2_abs_loading"] = loading_df["PC2"].abs()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `loading_df["max_abs_loading"] = loading_df[["pc1_abs_loading", "pc2_abs_loading"]].max(axis=1)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `loading_df = loading_df.sort_values("max_abs_loading", ascending=False).reset_index(drop=True)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `pca_diag_dir = PCA_ARTIFACT_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `pca_diag_dir.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `pca_explained_variance_path = pca_diag_dir / "pca_explained_variance.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `pca_loading_path = pca_diag_dir / "pca_feature_loadings.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `pca_explained_variance_df.to_csv(pca_explained_variance_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `loading_df.to_csv(pca_loading_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `ledger.add( kind="step", step="pca_diagnostics", message="Built PCA diagnostics and saved loadings.", data={ "pca_explained_variance_path": str(pca_explained_variance_path), "pca_l` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Saved:", pca_explained_variance_path)` | Displays a notebook-facing result for inspection. |
| `print("Saved:", pca_loading_path)` | Displays a notebook-facing result for inspection. |
| `display(pca_explained_variance_df)` | Displays a notebook-facing result for inspection. |
| `display(loading_df.head(20))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 51 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `any`
- `append`
- `apply`
- `backfill`
- `bfill`
- `Built`
- `coerce`
- `columns`
- `comparison`
- `copy`
- `csv`
- `DataFrame`
- `ddof`
- `drop`
- `else`
- `errors`
- `fallback`
- `feature`
- `feature_col`

### Outputs

- `ascending`
- `data`
- `ffill_df`
- `ffill_imputed_df`
- `ffill_s`
- `imputation_compare_df`
- `imputation_compare_path`
- `impute_compare_rows`
- `IMPUTE_COMPARE_SAMPLE_N`
- `impute_df`
- `impute_source_with_keys`
- `kind`
- `logger`
- `median_imputed_df`
- `median_s`
- `message`
- `original_s`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Imputation comparison`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `IMPUTE_COMPARE_SAMPLE_N = 10000`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `impute_df = silver_eda_dataframe[FEATURE_COLUMNS].copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if len(impute_df) > IMPUTE_COMPARE_SAMPLE_N: impute_df = impute_df.sample(IMPUTE_COMPARE_SAMPLE_N, random_state=42).copy()`: Controls validation, iteration, file handling, or error handling for this step.
- `impute_df = impute_df.apply(pd.to_numeric, errors="coerce")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Strategy 1: median impute`: Documents the purpose or boundary of the surrounding notebook step.
- `median_imputed_df = impute_df.fillna(impute_df.median(numeric_only=True))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Strategy 2: forward fill within row order, then backfill, then median fallback`: Documents the purpose or boundary of the surrounding notebook step.
- `if "meta__episode_id" in silver_eda_dataframe.columns and time_col is not None: impute_source_with_keys = silver_eda_dataframe[["meta__episode_id", time_col] + FEATURE_COLUMNS].cop`: Controls validation, iteration, file handling, or error handling for this step.
- `else: ffill_imputed_df = impute_df.fillna(method="ffill").fillna(method="bfill") ffill_imputed_df = ffill_imputed_df.fillna(ffill_imputed_df.median(numeric_only=True))`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `any`
- `append`
- `apply`
- `bfill`
- `copy`
- `DataFrame`
- `display`
- `ffill`
- `fillna`
- `groupby`
- `head`
- `isna`
- `mean`
- `median`
- `notna`
- `reset_index`
- `sample`
- `sort_values`
- `std`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Imputation comparison` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `IMPUTE_COMPARE_SAMPLE_N = 10000` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `impute_df = silver_eda_dataframe[FEATURE_COLUMNS].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(impute_df) > IMPUTE_COMPARE_SAMPLE_N: impute_df = impute_df.sample(IMPUTE_COMPARE_SAMPLE_N, random_state=42).copy()` | Controls validation, iteration, file handling, or error handling for this step. |
| `impute_df = impute_df.apply(pd.to_numeric, errors="coerce")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Strategy 1: median impute` | Documents the purpose or boundary of the surrounding notebook step. |
| `median_imputed_df = impute_df.fillna(impute_df.median(numeric_only=True))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Strategy 2: forward fill within row order, then backfill, then median fallback` | Documents the purpose or boundary of the surrounding notebook step. |
| `if "meta__episode_id" in silver_eda_dataframe.columns and time_col is not None: impute_source_with_keys = silver_eda_dataframe[["meta__episode_id", time_col] + FEATURE_COLUMNS].cop` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: ffill_imputed_df = impute_df.fillna(method="ffill").fillna(method="bfill") ffill_imputed_df = ffill_imputed_df.fillna(ffill_imputed_df.median(numeric_only=True))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `impute_compare_rows = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for feature_col in FEATURE_COLUMNS: original_s = pd.to_numeric(impute_df[feature_col], errors="coerce") median_s = pd.to_numeric(median_imputed_df[feature_col], errors="coerce") ff` | Controls validation, iteration, file handling, or error handling for this step. |
| `imputation_compare_df = pd.DataFrame(impute_compare_rows)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `imputation_compare_df["median_mean_shift"] = ( imputation_compare_df["median_imputed_mean"] - imputation_compare_df["original_mean"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `imputation_compare_df["ffill_mean_shift"] = ( imputation_compare_df["ffill_imputed_mean"] - imputation_compare_df["original_mean"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `imputation_compare_df["median_std_shift"] = ( imputation_compare_df["median_imputed_std"] - imputation_compare_df["original_std"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `imputation_compare_df["ffill_std_shift"] = ( imputation_compare_df["ffill_imputed_std"] - imputation_compare_df["original_std"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `imputation_compare_df = imputation_compare_df.sort_values( "original_missing_pct", ascending=False,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).reset_index(drop=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `imputation_compare_path = ( SILVER_EDA_SUMMARY_DIR / "imputation_compare_summary.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `imputation_compare_df.to_csv(imputation_compare_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `ledger.add( kind="step", step="imputation_comparison", message="Built imputation comparison summary.", data={ "imputation_compare_path": str(imputation_compare_path), "top_imputati` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Saved:", imputation_compare_path)` | Displays a notebook-facing result for inspection. |
| `display(imputation_compare_df.head(25))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 52 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `across`
- `add`
- `agg`
- `apply`
- `audit`
- `Built`
- `cleaner`
- `coerce`
- `confirmation`
- `copy`
- `csv`
- `decision_function`
- `downstream`
- `dropna`
- `eq`
- `errors`
- `FEATURE_COLUMNS`
- `fillna`
- `fit`
- `fit_transform`

### Outputs

- `contamination`
- `data`
- `kind`
- `logger`
- `mean_iforest_score`
- `median_iforest_score`
- `message`
- `n_estimators`
- `n_jobs`
- `OUTLIER_AUDIT_SAMPLE_N`
- `OUTLIER_CONTAMINATION`
- `outlier_count`
- `outlier_df`
- `outlier_model`
- `outlier_scaler`
- `outlier_summary_df`
- `outlier_summary_path`
- `random_state`
- `row_count`
- `score_outlier_df`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Outlier audit (downstream confirmation, not primary cleaner)`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `OUTLIER_AUDIT_SAMPLE_N = 20000`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `OUTLIER_CONTAMINATION = 0.05`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `outlier_df = silver_eda_dataframe[[STATE_COL_PROFILED] + FEATURE_COLUMNS].copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Train only on normal_clean`: Documents the purpose or boundary of the surrounding notebook step.
- `train_outlier_df = outlier_df.loc[mask_profiled_normal_clean, FEATURE_COLUMNS].copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `train_outlier_df = train_outlier_df.apply(pd.to_numeric, errors="coerce")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `train_outlier_df = train_outlier_df.fillna(train_outlier_df.median(numeric_only=True))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `score_outlier_df = outlier_df.copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `score_outlier_features = score_outlier_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `agg`
- `apply`
- `audit`
- `copy`
- `decision_function`
- `display`
- `eq`
- `fillna`
- `fit`
- `fit_transform`
- `groupby`
- `IsolationForest`
- `median`
- `predict`
- `reset_index`
- `RobustScaler`
- `sample`
- `to_csv`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Outlier audit (downstream confirmation, not primary cleaner)` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `OUTLIER_AUDIT_SAMPLE_N = 20000` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `OUTLIER_CONTAMINATION = 0.05` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `outlier_df = silver_eda_dataframe[[STATE_COL_PROFILED] + FEATURE_COLUMNS].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Train only on normal_clean` | Documents the purpose or boundary of the surrounding notebook step. |
| `train_outlier_df = outlier_df.loc[mask_profiled_normal_clean, FEATURE_COLUMNS].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `train_outlier_df = train_outlier_df.apply(pd.to_numeric, errors="coerce")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `train_outlier_df = train_outlier_df.fillna(train_outlier_df.median(numeric_only=True))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `score_outlier_df = outlier_df.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `score_outlier_features = score_outlier_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `score_outlier_features = score_outlier_features.fillna(score_outlier_features.median(numeric_only=True))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(train_outlier_df) > OUTLIER_AUDIT_SAMPLE_N: train_outlier_df = train_outlier_df.sample(OUTLIER_AUDIT_SAMPLE_N, random_state=42).copy()` | Controls validation, iteration, file handling, or error handling for this step. |
| `outlier_scaler = RobustScaler()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `X_train_outlier = outlier_scaler.fit_transform(train_outlier_df)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `X_score_outlier = outlier_scaler.transform(score_outlier_features)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `outlier_model = IsolationForest( n_estimators=200, contamination=OUTLIER_CONTAMINATION, random_state=42, n_jobs=-1,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `outlier_model.fit(X_train_outlier)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `score_outlier_df["iforest_score"] = outlier_model.decision_function(X_score_outlier)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `score_outlier_df["iforest_pred"] = outlier_model.predict(X_score_outlier)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `score_outlier_df["iforest_is_outlier"] = score_outlier_df["iforest_pred"].eq(-1)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `outlier_summary_df = ( score_outlier_df.groupby(STATE_COL_PROFILED, dropna=False) .agg( row_count=("iforest_is_outlier", "size"), outlier_count=("iforest_is_outlier", "sum"), mean_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `outlier_summary_df["outlier_rate"] = ( outlier_summary_df["outlier_count"] / outlier_summary_df["row_count"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `outlier_summary_path = ( SILVER_EDA_SUMMARY_DIR / "outlier_audit_profiled_states.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `outlier_summary_df.to_csv(outlier_summary_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `ledger.add( kind="step", step="outlier_audit", message="Built downstream IsolationForest outlier audit across profiled states.", data={ "outlier_summary_path": str(outlier_summary_` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Saved:", outlier_summary_path)` | Displays a notebook-facing result for inspection. |
| `display(outlier_summary_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 53 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `abnormal_percent`
- `add`
- `append`
- `artifact`
- `artifacts`
- `astype`
- `columns`
- `copy`
- `count`
- `counts`
- `Creating`
- `dump`
- `else`
- `encoding`
- `episode`
- `Episode`
- `episode_count`
- `episode_df`
- `episode_id`

### Outputs

- `abnormal_count`
- `data`
- `dropna`
- `episode_source_df`
- `EPISODE_STATUS_EXPORT_PATH`
- `episode_status_rows`
- `episode_status_work_df`
- `episode_total_rows`
- `kind`
- `logger`
- `message`
- `normal_clean_count`
- `normal_contaminated_count`
- `normal_suspect_count`
- `normal_total`
- `recovery_count`
- `sort`
- `state_counts`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Export episode status counts for synthetic generator inputs`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Location:`: Documents the purpose or boundary of the surrounding notebook step.
- `# artifacts/silver_subsets/pump/generator_inputs/episode_status_counts.json`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `EPISODE_STATUS_EXPORT_PATH = ( GENERATOR_INPUT_DIR / "episode_status_counts.json"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if "meta__episode_id" not in silver_eda_dataframe.columns: print("WARNING: meta__episode_id not found. Creating one global fallback episode.") episode_source_df = silver_eda_datafr`: Displays a notebook-facing result for inspection.
- `else: episode_source_df = silver_eda_dataframe.copy()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `episode_status_work_df = episode_source_df.copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `episode_status_work_df[STATE_COL_PROFILED] = ( episode_status_work_df[STATE_COL_PROFILED] .astype(str) .str.lower() .str.strip()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `append`
- `astype`
- `copy`
- `dump`
- `get`
- `groupby`
- `lower`
- `notna`
- `open`
- `strip`
- `to_dict`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Export episode status counts for synthetic generator inputs` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Location:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# artifacts/silver_subsets/pump/generator_inputs/episode_status_counts.json` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `EPISODE_STATUS_EXPORT_PATH = ( GENERATOR_INPUT_DIR / "episode_status_counts.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if "meta__episode_id" not in silver_eda_dataframe.columns: print("WARNING: meta__episode_id not found. Creating one global fallback episode.") episode_source_df = silver_eda_datafr` | Displays a notebook-facing result for inspection. |
| `else: episode_source_df = silver_eda_dataframe.copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `episode_status_work_df = episode_source_df.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `episode_status_work_df[STATE_COL_PROFILED] = ( episode_status_work_df[STATE_COL_PROFILED] .astype(str) .str.lower() .str.strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `episode_status_rows = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for episode_id, episode_df in episode_status_work_df.groupby( "meta__episode_id", dropna=False, sort=True,` | Controls validation, iteration, file handling, or error handling for this step. |
| `): state_counts = episode_df[STATE_COL_PROFILED].value_counts(dropna=False).to_dict() normal_clean_count = int(state_counts.get("normal_clean", 0)) normal_suspect_count = int(state` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `with open(EPISODE_STATUS_EXPORT_PATH, "w", encoding="utf-8") as f: json.dump(episode_status_rows, f, indent=2)` | Controls validation, iteration, file handling, or error handling for this step. |
| `ledger.add( kind="artifact", step="export_episode_status_counts", message="Exported episode-level profiled state counts for synthetic generator inputs.", data={ "episode_status_cou` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Saved episode status counts:", EPISODE_STATUS_EXPORT_PATH)` | Displays a notebook-facing result for inspection. |
| `print("Episode count:", len(episode_status_rows))` | Displays a notebook-facing result for inspection. |
| `print("First row:", episode_status_rows[0] if episode_status_rows else None)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 54 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `added`
- `AFTER`
- `All`
- `append`
- `are`
- `artifact`
- `before`
- `cell`
- `cells`
- `continue`
- `correlation`
- `counts`
- `defined`
- `dictionary`
- `disk`
- `do`
- `does`
- `dropped`
- `dropped_feature_profile_abnormal_path`

### Outputs

- `data`
- `generator_input_manifest`
- `kind`
- `logger`
- `message`
- `missing_manifest_paths`
- `missing_manifest_variables`
- `required_manifest_variables`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Update generator input manifest with episode + dropped profile paths`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# This cell should run AFTER:`: Documents the purpose or boundary of the surrounding notebook step.
- `# 1. generator-ready profile/correlation/group/fault exports`: Documents the purpose or boundary of the surrounding notebook step.
- `# 2. dropped feature profile exports`: Documents the purpose or boundary of the surrounding notebook step.
- `# 3. episode status counts export`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# If the manifest dictionary is not still in memory, reload it from disk.`: Documents the purpose or boundary of the surrounding notebook step.
- `if "generator_input_manifest" not in globals(): if not Path(generator_input_manifest_path).exists(): raise FileNotFoundError( f"generator_input_manifest_path does not exist: {gener`: Controls validation, iteration, file handling, or error handling for this step.
- `required_manifest_variables = [ "EPISODE_STATUS_EXPORT_PATH", "DROPPED_SENSOR_REGISTRY_PATH", "dropped_feature_profile_normal_clean_path", "dropped_feature_profile_abnormal_path", `: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `append`
- `dump`
- `exists`
- `FileNotFoundError`
- `globals`
- `items`
- `load`
- `NameError`
- `open`
- `Path`
- `strip`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Update generator input manifest with episode + dropped profile paths` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# This cell should run AFTER:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 1. generator-ready profile/correlation/group/fault exports` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 2. dropped feature profile exports` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 3. episode status counts export` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# If the manifest dictionary is not still in memory, reload it from disk.` | Documents the purpose or boundary of the surrounding notebook step. |
| `if "generator_input_manifest" not in globals(): if not Path(generator_input_manifest_path).exists(): raise FileNotFoundError( f"generator_input_manifest_path does not exist: {gener` | Controls validation, iteration, file handling, or error handling for this step. |
| `required_manifest_variables = [ "EPISODE_STATUS_EXPORT_PATH", "DROPPED_SENSOR_REGISTRY_PATH", "dropped_feature_profile_normal_clean_path", "dropped_feature_profile_abnormal_path", ` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_manifest_variables = [ variable_name for variable_name in required_manifest_variables if variable_name not in globals()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_manifest_variables: raise NameError( "The following variables are not defined yet. " "Run the dropped-profile export and episode-status export cells before this cell: " ` | Controls validation, iteration, file handling, or error handling for this step. |
| `# Existing required generator inputs` | Documents the purpose or boundary of the surrounding notebook step. |
| `generator_input_manifest["feature_profile_normal_clean_path"] = str(feature_profile_normal_clean_path)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `generator_input_manifest["feature_profile_abnormal_path"] = str(feature_profile_abnormal_path)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `generator_input_manifest["feature_profile_recovery_path"] = str(feature_profile_recovery_path)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `generator_input_manifest["sensor_correlation_pairs_normal_clean_path"] = str(sensor_correlation_pairs_normal_clean_path)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `generator_input_manifest["sensor_group_map_normal_clean_path"] = str(sensor_group_map_normal_clean_path)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `generator_input_manifest["sensor_fault_pairings_normal_path"] = str(sensor_fault_pairings_normal_path)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `generator_input_manifest["sensor_correlation_hotspot_clusters_normal_clean_path"] = str( sensor_correlation_hotspot_clusters_normal_clean_path` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Newly added generator inputs` | Documents the purpose or boundary of the surrounding notebook step. |
| `generator_input_manifest["episode_status_counts_path"] = str(EPISODE_STATUS_EXPORT_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `generator_input_manifest["dropped_sensor_registry_path"] = str( DROPPED_SENSOR_REGISTRY_PATH` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `generator_input_manifest["dropped_feature_profile_normal_clean_path"] = str( dropped_feature_profile_normal_clean_path` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `generator_input_manifest["dropped_feature_profile_abnormal_path"] = str( dropped_feature_profile_abnormal_path` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `generator_input_manifest["dropped_feature_profile_recovery_path"] = str( dropped_feature_profile_recovery_path` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Validate paths before writing manifest` | Documents the purpose or boundary of the surrounding notebook step. |
| `missing_manifest_paths = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for key, value in generator_input_manifest.items(): if value is None or str(value).strip() == "": missing_manifest_paths.append((key, value)) continue if not Path(value).exists(): ` | Controls validation, iteration, file handling, or error handling for this step. |
| `if missing_manifest_paths: print("WARNING: Some manifest paths do not exist yet:") for key, value in missing_manifest_paths: print(f" {key}: {value}")` | Displays a notebook-facing result for inspection. |
| `else: print("PASS: All generator manifest paths exist.")` | Displays a notebook-facing result for inspection. |
| `with open(generator_input_manifest_path, "w", encoding="utf-8") as f: json.dump(generator_input_manifest, f, indent=2)` | Controls validation, iteration, file handling, or error handling for this step. |
| `ledger.add( kind="artifact", step="update_generator_input_manifest", message="Updated generator input manifest with episode status counts and dropped feature profile paths.", data=` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Updated generator input manifest:", generator_input_manifest_path)` | Displays a notebook-facing result for inspection. |
| `for key, value in generator_input_manifest.items(): print(f"{key}: {value}")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 55 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__silver__eda_profile__summary`
- `__silver__truth__`
- `abnormal_rows`
- `Add`
- `add`
- `append_truth_index`
- `artifact`
- `artifact_paths`
- `artifacts`
- `astype`
- `before`
- `Build`
- `build_silver_eda_profile_truth_record`
- `build_truth_record`
- `Built`
- `bundle`
- `Carry`
- `clean`
- `config_snapshot`
- `correlation_matrix_normal_clean_path`

### Outputs

- `column_count`
- `data`
- `dataset_name`
- `expected_silver_eda_profile_truth_path`
- `feature_columns`
- `kind`
- `layer_name`
- `logger`
- `message`
- `meta_columns`
- `missingness_payload`
- `parent_truth_hash`
- `pipeline_mode`
- `process_run_id`
- `profiled_state_counts`
- `row_count`
- `silver_eda_profile_summary`
- `silver_eda_profile_summary_path`
- `silver_eda_profile_truth`
- `SILVER_EDA_PROFILE_TRUTH_HASH`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Finalize Silver EDA Profile truth record`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `profiled_state_counts = ( silver_eda_dataframe[STATE_COL_PROFILED] .astype(str) .value_counts(dropna=False) .to_dict()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_eda_profile_summary = { "dataset_name": DATASET_NAME, "stage": STAGE, "layer_name": "silver", "eda_stage": "eda_profile", "profiled_state_counts": profiled_state_counts, "fe`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `silver_eda_profile_summary_path = ( SILVER_EDA_SUMMARY_DIR / f"{DATASET_NAME}__silver__eda_profile__summary.json"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `with open(silver_eda_profile_summary_path, "w", encoding="utf-8") as f: json.dump(silver_eda_profile_summary, f, indent=2)`: Controls validation, iteration, file handling, or error handling for this step.
- `# ---------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build truth record`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `add`
- `append_truth_index`
- `astype`
- `build_truth_record`
- `display`
- `dump`
- `get`
- `globals`
- `identify_meta_columns`
- `initialize_layer_truth`
- `open`
- `save_truth_record`
- `sum`
- `to_dict`
- `update_truth_section`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Finalize Silver EDA Profile truth record` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `profiled_state_counts = ( silver_eda_dataframe[STATE_COL_PROFILED] .astype(str) .value_counts(dropna=False) .to_dict()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_profile_summary = { "dataset_name": DATASET_NAME, "stage": STAGE, "layer_name": "silver", "eda_stage": "eda_profile", "profiled_state_counts": profiled_state_counts, "fe` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_profile_summary_path = ( SILVER_EDA_SUMMARY_DIR / f"{DATASET_NAME}__silver__eda_profile__summary.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `with open(silver_eda_profile_summary_path, "w", encoding="utf-8") as f: json.dump(silver_eda_profile_summary, f, indent=2)` | Controls validation, iteration, file handling, or error handling for this step. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build truth record` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `silver_eda_profile_truth = initialize_layer_truth( truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name=LAYER_NAME, process_run_id=SILVER_SUBSETS_PROCESS_RUN_ID, pipe` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_profile_truth = update_truth_section( silver_eda_profile_truth, "config_snapshot", { "source_config_stage": "silver_eda", "effective_stage": STAGE, "effective_layer_name` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_profile_truth = update_truth_section( silver_eda_profile_truth, "runtime_facts", { "parent_layer_name": "silver", "parent_truth_hash": SILVER_TRUTH_HASH, "source_profile` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Carry missingness payload forward from Silver PreEDA truth.` | Documents the purpose or boundary of the surrounding notebook step. |
| `missingness_payload = ( silver_truth.get("runtime_facts", {}) .get("missingness_quarantine")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missingness_payload is not None: silver_eda_profile_truth = update_truth_section( silver_eda_profile_truth, "runtime_facts", { "missingness_quarantine": missingness_payload, }, ` | Controls validation, iteration, file handling, or error handling for this step. |
| `silver_eda_profile_truth = update_truth_section( silver_eda_profile_truth, "artifact_paths", { # Source / summary "profiled_dataframe_path": str(PROFILED_DF_PATH), "silver_eda_prof` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_profile_truth = update_truth_section( silver_eda_profile_truth, "notes", { "purpose": ( "Silver profiled subset artifact bundle. This truth record provides " "clean-norm` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_profile_truth = build_truth_record( truth_base=silver_eda_profile_truth, row_count=len(silver_eda_dataframe), column_count=silver_eda_dataframe.shape[1], meta_columns=id` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_EDA_PROFILE_TRUTH_HASH = silver_eda_profile_truth["truth_hash"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Add stage metadata before saving and indexing.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The existing truth utility does not add stage-specific filenames, so stage` | Documents the purpose or boundary of the surrounding notebook step. |
| `# identity is stored in the truth record and truth_index.jsonl.` | Documents the purpose or boundary of the surrounding notebook step. |
| `silver_eda_profile_truth["truth_stage"] = "eda_profile"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_profile_truth["notebook_name"] = "silver_02b_eda_profile"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `expected_silver_eda_profile_truth_path = ( TRUTHS_PATH / "silver" / f"{DATASET_NAME}__silver__truth__{SILVER_EDA_PROFILE_TRUTH_HASH}.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_profile_truth["truth_path"] = str( expected_silver_eda_profile_truth_path` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `silver_eda_profile_truth_path = save_truth_record( silver_eda_profile_truth, truth_dir=TRUTHS_PATH, dataset_name=DATASET_NAME, layer_name="silver",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `append_truth_index( silver_eda_profile_truth, truth_index_path=TRUTH_INDEX_PATH,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="build_silver_eda_profile_truth_record", message="Built and indexed Silver EDA profile truth record.", data={ "silver_eda_profile_truth_hash": SILVER_` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Saved Silver EDA profile summary:", silver_eda_profile_summary_path)` | Displays a notebook-facing result for inspection. |
| `print("Saved Silver EDA profile truth:", silver_eda_profile_truth_path)` | Displays a notebook-facing result for inspection. |
| `print("SILVER_EDA_PROFILE_TRUTH_HASH:", SILVER_EDA_PROFILE_TRUTH_HASH)` | Displays a notebook-facing result for inspection. |
| `display(silver_eda_profile_summary)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: truth record.

## Code Cell 56 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_path`
- `abnormal`
- `above`
- `add`
- `Add`
- `append`
- `artifact`
- `artifact_count`
- `artifact_description`
- `artifact_format`
- `artifact_name`
- `artifact_names`
- `artifact_path_value`
- `artifact_type`
- `audit`
- `based`
- `be`
- `behavior`
- `Build`
- `build_silver_eda_artifact_index`

### Outputs

- `artifact_description_map`
- `artifact_index_records`
- `artifact_path`
- `artifact_suffix`
- `artifact_type_map`
- `data`
- `eda_artifact_index_df`
- `extra`
- `keep`
- `kind`
- `logger`
- `message`
- `step`
- `subset`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build Silver EDA artifact index dataframe`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `artifact_index_records = []`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `artifact_description_map = { "correlation_matrix_normal_clean_path": "Normal-clean sensor correlation matrix.", "sensor_correlation_pairs_normal_clean_path": "Long-form normal-clea`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `artifact_type_map = { ".csv": "table", ".json": "json", ".parquet": "dataset", ".png": "plot", ".jpg": "plot", ".jpeg": "plot", ".pkl": "pickle", ".yaml": "config", ".yml": "config`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Start with paths stored in the final Silver subsets summary.`: Documents the purpose or boundary of the surrounding notebook step.
- `for artifact_name, artifact_path_value in silver_eda_profile_summary.items(): if not str(artifact_name).endswith("_path"): continue if artifact_path_value is None or str(artifact_p`: Controls validation, iteration, file handling, or error handling for this step.
- `# Add generator manifest paths that may not be separately listed above.`: Documents the purpose or boundary of the surrounding notebook step.
- `if "generator_input_manifest" in globals(): for artifact_name, artifact_path_value in generator_input_manifest.items(): if artifact_path_value is None or str(artifact_path_value).s`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `add`
- `append`
- `DataFrame`
- `display`
- `drop_duplicates`
- `endswith`
- `get`
- `globals`
- `info`
- `items`
- `lower`
- `Path`
- `replace`
- `reset_index`
- `strip`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build Silver EDA artifact index dataframe` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `artifact_index_records = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `artifact_description_map = { "correlation_matrix_normal_clean_path": "Normal-clean sensor correlation matrix.", "sensor_correlation_pairs_normal_clean_path": "Long-form normal-clea` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `artifact_type_map = { ".csv": "table", ".json": "json", ".parquet": "dataset", ".png": "plot", ".jpg": "plot", ".jpeg": "plot", ".pkl": "pickle", ".yaml": "config", ".yml": "config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Start with paths stored in the final Silver subsets summary.` | Documents the purpose or boundary of the surrounding notebook step. |
| `for artifact_name, artifact_path_value in silver_eda_profile_summary.items(): if not str(artifact_name).endswith("_path"): continue if artifact_path_value is None or str(artifact_p` | Controls validation, iteration, file handling, or error handling for this step. |
| `# Add generator manifest paths that may not be separately listed above.` | Documents the purpose or boundary of the surrounding notebook step. |
| `if "generator_input_manifest" in globals(): for artifact_name, artifact_path_value in generator_input_manifest.items(): if artifact_path_value is None or str(artifact_path_value).s` | Controls validation, iteration, file handling, or error handling for this step. |
| `eda_artifact_index_df = ( pd.DataFrame(artifact_index_records) .drop_duplicates( subset=[ "dataset_id", "run_id", "notebook_name", "artifact_name", ], keep="last", ) .reset_index(d` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info( "Built Silver 02b EDA artifact index dataframe.", extra={ "dataset_id": DATASET_ID, "run_id": RUN_ID, "artifact_count": int(len(eda_artifact_index_df)), },` | Writes a logger message for traceability during notebook execution. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="artifact", step="build_silver_eda_artifact_index", message="Built SQL-ready artifact index for Silver 02b EDA outputs.", data={ "dataset_id": DATASET_ID, "run_id"` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(eda_artifact_index_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 57 — QA

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append_truth_index`
- `astype`
- `before`
- `cell`
- `columns`
- `Confirm`
- `copy`
- `count`
- `difference`
- `dir`
- `EDA`
- `eda_profile`
- `empty`
- `eq`
- `exists`
- `f`
- `file`
- `FileNotFoundError`
- `files`
- `finalization`

### Outputs

- `missing_columns`
- `required_columns`
- `silver_eda_profile_truth_index_df`
- `truth_dir_check`
- `truth_files`
- `truth_index_df`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Verify Silver EDA profile truth exists`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `truth_dir_check = SILVER_EDA_TRUTH_DIR`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("Silver truth dir:", truth_dir_check)`: Displays a notebook-facing result for inspection.
- `truth_files = sorted(truth_dir_check.glob("*.json"))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("truth file count:", len(truth_files))`: Displays a notebook-facing result for inspection.
- `print("latest truth files:", [p.name for p in truth_files[-5:]])`: Displays a notebook-facing result for inspection.
- `if not truth_files: raise FileNotFoundError( "No Silver truth JSON files were found. " "Run the Silver EDA profile truth finalization cell first." )`: Controls validation, iteration, file handling, or error handling for this step.
- `if not TRUTH_INDEX_PATH.exists() or TRUTH_INDEX_PATH.stat().st_size == 0: raise FileNotFoundError( f"Truth index is missing or empty: {TRUTH_INDEX_PATH}" )`: Controls validation, iteration, file handling, or error handling for this step.
- `truth_index_df = pd.read_json(TRUTH_INDEX_PATH, lines=True)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `required_columns = { "layer_name", "truth_stage", "truth_path", "truth_hash",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `append_truth_index`
- `astype`
- `copy`
- `difference`
- `display`
- `eq`
- `exists`
- `FileNotFoundError`
- `glob`
- `KeyError`
- `read_json`
- `sorted`
- `stat`
- `tail`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Verify Silver EDA profile truth exists` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `truth_dir_check = SILVER_EDA_TRUTH_DIR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Silver truth dir:", truth_dir_check)` | Displays a notebook-facing result for inspection. |
| `truth_files = sorted(truth_dir_check.glob("*.json"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("truth file count:", len(truth_files))` | Displays a notebook-facing result for inspection. |
| `print("latest truth files:", [p.name for p in truth_files[-5:]])` | Displays a notebook-facing result for inspection. |
| `if not truth_files: raise FileNotFoundError( "No Silver truth JSON files were found. " "Run the Silver EDA profile truth finalization cell first." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `if not TRUTH_INDEX_PATH.exists() or TRUTH_INDEX_PATH.stat().st_size == 0: raise FileNotFoundError( f"Truth index is missing or empty: {TRUTH_INDEX_PATH}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `truth_index_df = pd.read_json(TRUTH_INDEX_PATH, lines=True)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `required_columns = { "layer_name", "truth_stage", "truth_path", "truth_hash",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_columns = required_columns.difference(truth_index_df.columns)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if missing_columns: display(truth_index_df.tail(10)) raise KeyError( f"truth_index.jsonl is missing required columns: {sorted(missing_columns)}" )` | Displays a notebook-facing result for inspection. |
| `silver_eda_profile_truth_index_df = truth_index_df.loc[ truth_index_df["layer_name"].astype(str).eq("silver") & truth_index_df["truth_stage"].astype(str).eq("eda_profile")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if silver_eda_profile_truth_index_df.empty: display( truth_index_df.loc[ truth_index_df["layer_name"].astype(str).eq("silver") ].tail(10) ) raise ValueError( "No Silver EDA profile` | Displays a notebook-facing result for inspection. |
| `print("Verified Silver EDA profile truth index rows:")` | Displays a notebook-facing result for inspection. |
| `display(silver_eda_profile_truth_index_df.tail(5))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 58 — Build the Silver EDA profile table

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `analytical`
- `bool`
- `boolean_column_count`
- `categorical_column_count`
- `category`
- `column_count`
- `columns`
- `dataframe`
- `DataFrame`
- `dataframe_name`
- `DATASET_ID`
- `dataset_id`
- `datetime`
- `datetime_column_count`
- `datetimetz`
- `deep`
- `duplicate_row_count`
- `duplicated`
- `EDA`
- `EDA_DATAFRAME_NAME`

### Outputs

- `profile_df`

### Key Operations

- `profile_df = pd.DataFrame( [ { "dataset_id": DATASET_ID, "run_id": RUN_ID, "notebook_name": EDA_NOTEBOOK_NAME, "dataframe_name": EDA_DATAFRAME_NAME, "profile_scope": "full_dataset"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `DataFrame`
- `duplicated`
- `memory_usage`
- `select_dtypes`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `profile_df = pd.DataFrame( [ { "dataset_id": DATASET_ID, "run_id": RUN_ID, "notebook_name": EDA_NOTEBOOK_NAME, "dataframe_name": EDA_DATAFRAME_NAME, "profile_scope": "full_dataset"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 59 — Select numeric columns for EDA statistics

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `astype`
- `count`
- `dataframe_name`
- `DATASET_ID`
- `dataset_id`
- `describe`
- `EDA_DATAFRAME_NAME`
- `EDA_NOTEBOOK_NAME`
- `feature_name`
- `include`
- `index`
- `int64`
- `isna`
- `kurtosis`
- `kurtosis_value`
- `map`
- `max`
- `max_value`
- `mean`
- `mean_value`

### Outputs

- `columns`
- `describe_df`
- `feature_statistics_df`
- `numeric_df`
- `percentiles`

### Key Operations

- `numeric_df = silver_eda_dataframe.select_dtypes(include="number")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `describe_df = numeric_df.describe( percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `).T`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `feature_statistics_df = describe_df.reset_index().rename( columns={ "index": "feature_name", "count": "non_null_count", "mean": "mean_value", "std": "std_value", "min": "min_value"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `feature_statistics_df["dataset_id"] = DATASET_ID`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `feature_statistics_df["run_id"] = RUN_ID`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `feature_statistics_df["notebook_name"] = EDA_NOTEBOOK_NAME`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `feature_statistics_df["dataframe_name"] = EDA_DATAFRAME_NAME`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `feature_statistics_df["null_count"] = feature_statistics_df["feature_name"].map( silver_eda_dataframe.isna().sum().to_dict()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `).astype("int64")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `feature_statistics_df["null_pct"] = ( feature_statistics_df["null_count"] / len(silver_eda_dataframe)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `describe`
- `isna`
- `kurtosis`
- `map`
- `rename`
- `reset_index`
- `select_dtypes`
- `skew`
- `sum`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `numeric_df = silver_eda_dataframe.select_dtypes(include="number")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `describe_df = numeric_df.describe( percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).T` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_statistics_df = describe_df.reset_index().rename( columns={ "index": "feature_name", "count": "non_null_count", "mean": "mean_value", "std": "std_value", "min": "min_value"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_statistics_df["dataset_id"] = DATASET_ID` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_statistics_df["run_id"] = RUN_ID` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_statistics_df["notebook_name"] = EDA_NOTEBOOK_NAME` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_statistics_df["dataframe_name"] = EDA_DATAFRAME_NAME` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_statistics_df["null_count"] = feature_statistics_df["feature_name"].map( silver_eda_dataframe.isna().sum().to_dict()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `).astype("int64")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_statistics_df["null_pct"] = ( feature_statistics_df["null_count"] / len(silver_eda_dataframe)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_statistics_df["skew_value"] = feature_statistics_df["feature_name"].map( numeric_df.skew(numeric_only=True).to_dict()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_statistics_df["kurtosis_value"] = feature_statistics_df["feature_name"].map( numeric_df.kurtosis(numeric_only=True).to_dict()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_statistics_df = feature_statistics_df[ [ "dataset_id", "run_id", "notebook_name", "dataframe_name", "feature_name", "non_null_count", "null_count", "null_pct", "mean_value"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 60 — Build the missingness summary table

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `dataframe_name`
- `dataset_id`
- `DATASET_ID`
- `EDA_DATAFRAME_NAME`
- `EDA_NOTEBOOK_NAME`
- `feature_name`
- `index`
- `isna`
- `non_null_count`
- `notebook_name`
- `null_count`
- `null_pct`
- `rename`
- `reset_index`
- `row_count`
- `run_id`
- `RUN_ID`
- `silver_eda_dataframe`
- `sum`

### Outputs

- `missingness_summary_df`

### Key Operations

- `missingness_summary_df = ( silver_eda_dataframe.isna() .sum() .rename("null_count") .reset_index() .rename(columns={"index": "feature_name"})`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_summary_df["dataset_id"] = DATASET_ID`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_summary_df["run_id"] = RUN_ID`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_summary_df["notebook_name"] = EDA_NOTEBOOK_NAME`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_summary_df["dataframe_name"] = EDA_DATAFRAME_NAME`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_summary_df["row_count"] = len(silver_eda_dataframe)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_summary_df["non_null_count"] = ( missingness_summary_df["row_count"] - missingness_summary_df["null_count"]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_summary_df["null_pct"] = ( missingness_summary_df["null_count"] / missingness_summary_df["row_count"]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_summary_df = missingness_summary_df[ [ "dataset_id", "run_id", "notebook_name", "dataframe_name", "feature_name", "row_count", "null_count", "non_null_count", "null_pct`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `isna`
- `rename`
- `reset_index`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `missingness_summary_df = ( silver_eda_dataframe.isna() .sum() .rename("null_count") .reset_index() .rename(columns={"index": "feature_name"})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_summary_df["dataset_id"] = DATASET_ID` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_summary_df["run_id"] = RUN_ID` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_summary_df["notebook_name"] = EDA_NOTEBOOK_NAME` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_summary_df["dataframe_name"] = EDA_DATAFRAME_NAME` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_summary_df["row_count"] = len(silver_eda_dataframe)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_summary_df["non_null_count"] = ( missingness_summary_df["row_count"] - missingness_summary_df["null_count"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_summary_df["null_pct"] = ( missingness_summary_df["null_count"] / missingness_summary_df["row_count"]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_summary_df = missingness_summary_df[ [ "dataset_id", "run_id", "notebook_name", "dataframe_name", "feature_name", "row_count", "null_count", "non_null_count", "null_pct` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 61 — Compute the correlation matrix

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abs`
- `abs_correlation_value`
- `columns`
- `copy`
- `corr`
- `correlation_method`
- `correlation_value`
- `dataframe_name`
- `dataset_id`
- `DATASET_ID`
- `EDA_DATAFRAME_NAME`
- `EDA_NOTEBOOK_NAME`
- `feature_a`
- `feature_b`
- `level_0`
- `level_1`
- `method`
- `notebook_name`
- `numeric_df`
- `pearson`

### Outputs

- `correlation_matrix`
- `correlation_pairs_df`

### Key Operations

- `correlation_matrix = numeric_df.corr(method="pearson")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `correlation_pairs_df = ( correlation_matrix.stack() .rename("correlation_value") .reset_index() .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `correlation_pairs_df = correlation_pairs_df[ correlation_pairs_df["feature_a"] != correlation_pairs_df["feature_b"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `].copy()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `correlation_pairs_df["dataset_id"] = DATASET_ID`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `correlation_pairs_df["run_id"] = RUN_ID`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `correlation_pairs_df["notebook_name"] = EDA_NOTEBOOK_NAME`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `correlation_pairs_df["dataframe_name"] = EDA_DATAFRAME_NAME`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `correlation_pairs_df["correlation_method"] = "pearson"`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `correlation_pairs_df["abs_correlation_value"] = correlation_pairs_df["correlation_value"].abs()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `correlation_pairs_df = correlation_pairs_df[ [ "dataset_id", "run_id", "notebook_name", "dataframe_name", "correlation_method", "feature_a", "feature_b", "correlation_value", "abs_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `abs`
- `copy`
- `corr`
- `rename`
- `reset_index`
- `stack`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `correlation_matrix = numeric_df.corr(method="pearson")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `correlation_pairs_df = ( correlation_matrix.stack() .rename("correlation_value") .reset_index() .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `correlation_pairs_df = correlation_pairs_df[ correlation_pairs_df["feature_a"] != correlation_pairs_df["feature_b"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `correlation_pairs_df["dataset_id"] = DATASET_ID` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `correlation_pairs_df["run_id"] = RUN_ID` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `correlation_pairs_df["notebook_name"] = EDA_NOTEBOOK_NAME` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `correlation_pairs_df["dataframe_name"] = EDA_DATAFRAME_NAME` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `correlation_pairs_df["correlation_method"] = "pearson"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `correlation_pairs_df["abs_correlation_value"] = correlation_pairs_df["correlation_value"].abs()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `correlation_pairs_df = correlation_pairs_df[ [ "dataset_id", "run_id", "notebook_name", "dataframe_name", "correlation_method", "feature_a", "feature_b", "correlation_value", "abs_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 62 — Build sensor outlier summary records

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append`
- `columns`
- `continue`
- `DataFrame`
- `dataframe_name`
- `DATASET_ID`
- `dataset_id`
- `dropna`
- `EDA_DATAFRAME_NAME`
- `EDA_NOTEBOOK_NAME`
- `else`
- `empty`
- `feature_name`
- `iqr_1_5`
- `method_notes`
- `notebook_name`
- `numeric_df`
- `outlier_method`
- `outlier_pct`
- `quantile`

### Outputs

- `iqr`
- `lower_threshold`
- `outlier_count`
- `outlier_mask`
- `outlier_records`
- `outlier_summary_df`
- `q1`
- `q3`
- `row_count`
- `series`
- `upper_threshold`

### Key Operations

- `outlier_records = []`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `for feature_name in numeric_df.columns: series = numeric_df[feature_name].dropna() if series.empty: continue q1 = series.quantile(0.25) q3 = series.quantile(0.75) iqr = q3 - q1 low`: Controls validation, iteration, file handling, or error handling for this step.
- `outlier_summary_df = pd.DataFrame(outlier_records)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `append`
- `DataFrame`
- `dropna`
- `quantile`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `outlier_records = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for feature_name in numeric_df.columns: series = numeric_df[feature_name].dropna() if series.empty: continue q1 = series.quantile(0.25) q3 = series.quantile(0.75) iqr = q3 - q1 low` | Controls validation, iteration, file handling, or error handling for this step. |
| `outlier_summary_df = pd.DataFrame(outlier_records)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 63 — Build categorical summary records

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append`
- `bool`
- `category`
- `category_count`
- `category_pct`
- `category_value`
- `columns`
- `DataFrame`
- `dataframe_name`
- `dataset_id`
- `DATASET_ID`
- `dropna`
- `EDA_DATAFRAME_NAME`
- `EDA_NOTEBOOK_NAME`
- `else`
- `feature_name`
- `items`
- `notebook_name`
- `object`
- `run_id`

### Outputs

- `categorical_columns`
- `categorical_distribution_df`
- `categorical_records`
- `counts`
- `include`
- `total_count`

### Key Operations

- `categorical_records = []`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `categorical_columns = silver_eda_dataframe.select_dtypes( include=["object", "category", "bool"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `).columns`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `for feature_name in categorical_columns: counts = silver_eda_dataframe[feature_name].value_counts(dropna=False) total_count = int(counts.sum()) for category_value, category_count i`: Controls validation, iteration, file handling, or error handling for this step.
- `categorical_distribution_df = pd.DataFrame(categorical_records)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `append`
- `DataFrame`
- `items`
- `select_dtypes`
- `sum`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `categorical_records = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `categorical_columns = silver_eda_dataframe.select_dtypes( include=["object", "category", "bool"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).columns` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for feature_name in categorical_columns: counts = silver_eda_dataframe[feature_name].value_counts(dropna=False) total_count = int(counts.sum()) for category_value, category_count i` | Controls validation, iteration, file handling, or error handling for this step. |
| `categorical_distribution_df = pd.DataFrame(categorical_records)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 64 — Build the missingness summary table

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `DataFrame`
- `dataframes`
- `durable`
- `EDA`
- `eda_artifact_index_df`
- `EDA_NOTEBOOK_NAME`
- `else`
- `info`
- `items`
- `ledger`
- `notebook`
- `outputs`
- `PostgreSQL`
- `row_count`
- `rows_written`
- `Silver`
- `silver_02b_sql_summary_write`
- `Skipped`
- `skipped`

### Outputs

- `artifact_index_df`
- `categorical_distribution_df`
- `correlation_pairs_df`
- `data`
- `dataset_id`
- `engine`
- `extra`
- `feature_statistics_df`
- `kind`
- `logger`
- `message`
- `missingness_summary_df`
- `notebook_name`
- `outlier_summary_df`
- `profile_df`
- `run_id`
- `silver_eda_sql_rows_written`
- `step`
- `WRITE_SILVER_EDA_SQL_OUTPUTS`

### Key Operations

- `WRITE_SILVER_EDA_SQL_OUTPUTS = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if WRITE_SILVER_EDA_SQL_OUTPUTS: silver_eda_sql_rows_written = write_silver_eda_sql_outputs( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, notebook_name=EDA_NOTEBOOK_NAME, p`: Writes a logger message for traceability during notebook execution.
- `else: logger.info("Skipped Silver 02b EDA SQL summary write.") print("Silver 02b EDA SQL summary write skipped.")`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `add`
- `DataFrame`
- `display`
- `info`
- `items`
- `write_silver_eda_sql_outputs`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `WRITE_SILVER_EDA_SQL_OUTPUTS = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if WRITE_SILVER_EDA_SQL_OUTPUTS: silver_eda_sql_rows_written = write_silver_eda_sql_outputs( engine=engine, dataset_id=DATASET_ID, run_id=RUN_ID, notebook_name=EDA_NOTEBOOK_NAME, p` | Writes a logger message for traceability during notebook execution. |
| `else: logger.info("Skipped Silver 02b EDA SQL summary write.") print("Silver 02b EDA SQL summary write skipped.")` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

