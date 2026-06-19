# Notebook Code Reference: EDA_Notebook_Pump_Gold_01_PreProcessing

Notebook path:

`notebooks/experiments/EDA_Notebook_Pump_Gold_01_PreProcessing.ipynb`

## Notebook Purpose

This notebook prepares modeling-ready Gold features, train/test split context, and lineage outputs from the Silver analytical data.

Notebook stage:

`Gold`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Answer | Code Cell 01, Code Cell 02, Code Cell 17, Code Cell 19, Code Cell 23, Code Cell 46, Code Cell 57, Code Cell 58, Code Cell 59 |
| Load Paths, Configuration, and Runtime Settings | Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08 |
| Review intermediate output | Code Cell 09 |
| Start Logging for the Gold Preprocessing Stage | Code Cell 10, Code Cell 11 |
| Initialize the W&B Run | Code Cell 12 |
| Initialize the Gold Ledger | Code Cell 13, Code Cell 14 |
| Load the Silver Input and Supporting Artifacts | Code Cell 15 |
| Resolve the Parent Truth Record and Confirm the Dataset Identity | Code Cell 16 |
| Create the Working Gold Dataframe and Start Runtime Tracking | Code Cell 18 |
| Stamp Stable Gold Row Identity | Code Cell 20 |
| Reload the Parent Truth Record for Direct Reference | Code Cell 21 |
| Define the Episode-Based Train/Test Split Logic | Code Cell 22 |
| Build the Train/Test Split Mask | Code Cell 24 |
| Define a Helper to Stamp Training Metadata | Code Cell 25 |
| Stamp Train/Test Metadata onto the Working Dataframe | Code Cell 26 |
| Define the Numeric Feature Selection Logic | Code Cell 27 |
| Select the Numeric Feature Set for Gold Modeling | Code Cell 28 |
| Define the One-Hot Encoding Logic Based on Upstream Truth | Code Cell 29 |
| Apply One-Hot Encoding When Required | Code Cell 30 |
| Define the Imputation Logic | Code Cell 31 |
| Apply Numeric Feature Imputation | Code Cell 32 |
| Rebuild the Training Mask After Imputation | Code Cell 33 |
| Freeze a Prescaled Copy of the Gold Dataframe | Code Cell 34 |
| Define the Scaler Factory | Code Cell 35 |
| Define the Scaling Workflow | Code Cell 36 |
| Scale the Gold Feature Set | Code Cell 37 |
| Define the Normal-Only Fit Subset Logic | Code Cell 38 |
| Build the Normal-Only Fit Subset | Code Cell 39 |
| Define the Reference Profile Logic | Code Cell 40 |
| Build the Normal Reference Profile | Code Cell 41 |
| Define the Stage 2 Feature Ranking Logic | Code Cell 42 |
| Choose the Stage 2 Feature Set | Code Cell 43 |
| Define the Stage 3 Sensor Grouping Logic | Code Cell 44 |
| Build the Stage 3 Sensor Groups | Code Cell 45 |
| Save the Stage-Level Gold Artifacts | Code Cell 47 |
| Create the Final Gold Split Dataframes | Code Cell 48 |
| Finalize the Gold Truth Record | Code Cell 49 |
| Update Gold preprocessing truth metadata | Code Cell 50 |
| Save the Final Gold Preprocessing Outputs | Code Cell 51, Code Cell 52 |
| Save pre-scaled Gold feature outputs | Code Cell 53 |
| Save the Preprocessing Summary and Metadata Records | Code Cell 54 |
| Save the Ledger and Close the Tracking Run | Code Cell 55 |
| Run Final Gold Preprocessing Sanity Checks | Code Cell 56 |
| Compare the Prescaled and Scaled Column Structures | Code Cell 60 |
| Verify Final Lineage Columns Across All Gold Outputs | Code Cell 61 |
| Gold Preprocessing SQL Write Cell | Code Cell 62 |

## Code Cell 01 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__future__`
- `annotations`
- `Any`
- `append_truth_index`
- `artifact_file_path`
- `Artifacts`
- `artifacts`
- `build_artifact_dirs_from_config`
- `build_file_fingerprint`
- `build_truth_config_block`
- `build_truth_record`
- `cascade_row_tracking`
- `cast`
- `cluster`
- `columns`
- `config_loader`
- `configure_logging`
- `core`
- `Custom`
- `database`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `from __future__ import annotations`: Imports a dependency or project helper used by later cells.
- `from dataclasses import dataclass, field`: Imports a dependency or project helper used by later cells.
- `from datetime import datetime, timezone`: Imports a dependency or project helper used by later cells.
- `from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Mapping, cast`: Imports a dependency or project helper used by later cells.
- `from pathlib import Path`: Imports a dependency or project helper used by later cells.
- `import yaml`: Imports a dependency or project helper used by later cells.
- `import os`: Imports a dependency or project helper used by later cells.
- `import logging`: Imports a dependency or project helper used by later cells.
- `import wandb`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `import numpy as np`: Imports a dependency or project helper used by later cells.
- `import joblib`: Imports a dependency or project helper used by later cells.

Important functions or methods detected:
- `import`
- `set_option`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `from __future__ import annotations` | Imports a dependency or project helper used by later cells. |
| `from dataclasses import dataclass, field` | Imports a dependency or project helper used by later cells. |
| `from datetime import datetime, timezone` | Imports a dependency or project helper used by later cells. |
| `from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Mapping, cast` | Imports a dependency or project helper used by later cells. |
| `from pathlib import Path` | Imports a dependency or project helper used by later cells. |
| `import yaml` | Imports a dependency or project helper used by later cells. |
| `import os` | Imports a dependency or project helper used by later cells. |
| `import logging` | Imports a dependency or project helper used by later cells. |
| `import wandb` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `import numpy as np` | Imports a dependency or project helper used by later cells. |
| `import joblib` | Imports a dependency or project helper used by later cells. |
| `from sklearn.model_selection import train_test_split, KFold` | Imports a dependency or project helper used by later cells. |
| `from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder` | Imports a dependency or project helper used by later cells. |
| `from sklearn.decomposition import PCA` | Imports a dependency or project helper used by later cells. |
| `from sklearn.cluster import KMeans` | Imports a dependency or project helper used by later cells. |
| `# Custom Utilities Module` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.core.paths import get_paths` | Imports a dependency or project helper used by later cells. |
| `from utils.core.file_io import ( load_data, save_data, save_json, load_json,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.logging_profiler import profile_dataframe` | Imports a dependency or project helper used by later cells. |
| `from utils.core.logging_setup import ( configure_logging, log_layer_paths,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.wandb_utils import finalize_wandb_stage` | Imports a dependency or project helper used by later cells. |
| `from utils.core.truths import ( make_process_run_id, build_file_fingerprint, extract_truth_hash, identify_meta_columns, identify_feature_columns, initialize_layer_truth, update_tru` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.config_loader import ( load_pipeline_config, build_truth_config_block, set_wandb_dir_from_config, export_config_snapshot,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.medallion.gold.cascade_row_tracking import ensure_stable_row_id` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import ( get_engine_from_env, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.layer_postgres import ( read_layer_dataframe, write_layer_dataframe, prepare_layer_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.sql_notebook_helpers import ( delete_dataset_run_rows, execute_many, get_existing_dataframe, get_row_value, log_data_quality_event, log_pipeline_artifact, previ` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.medallion_sql_writers import ( log_gold_05_anomaly_detection_summary_sql, log_silver_eda_sql, write_bronze_sensor_observations_sql, write_gold_baseline_scores_s` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Ledger` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.core.ledger import Ledger` | Imports a dependency or project helper used by later cells. |
| `# Artifacts` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.core.artifacts import ( build_artifact_dirs_from_config, artifact_file_path,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.notebook_context import load_notebook_context` | Imports a dependency or project helper used by later cells. |
| `# Show more columns` | Documents the purpose or boundary of the surrounding notebook step. |
| `pd.set_option("display.max_columns", 100)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `pd.set_option("display.width", 200)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: SQL or medallion table write, truth record.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__name__`
- `a`
- `Any`
- `be`
- `cast`
- `column`
- `column_name`
- `def`
- `dictionary`
- `elif`
- `else`
- `f`
- `feature`
- `feature_columns`
- `got`
- `Handles`
- `into`
- `isinstance`
- `mapping`
- `Mapping`

### Outputs

- `cfg_optional_mapping`
- `cfg_require_mapping`
- `normalize_feature_columns`
- `raw_values`
- `require_dict`
- `require_list`
- `value`

### Key Operations

- `def require_dict(value: Any \| None, name: str) -> Dict[str, Any]: if value is None: return {} if not isinstance(value, dict): raise TypeError( f"{name} must be a dictionary, got {t`: Defines notebook-local logic used later in the notebook.
- `def require_list(value: Any \| None, name: str) -> List[Any]: if value is None: return [] if not isinstance(value, list): raise TypeError( f"{name} must be a list, got {type(value).`: Defines notebook-local logic used later in the notebook.
- `def normalize_feature_columns(value: Any, name: str = "feature_columns") -> list[str]: """ Normalize feature-column results into list[str]. Handles: - list[str] - tuple[list[str], `: Defines notebook-local logic used later in the notebook.
- `def cfg_require_mapping(value: object, name: str) -> Mapping[str, Any]: if not isinstance(value, Mapping): raise TypeError( f"{name} must be a mapping, got {type(value).__name__}: `: Defines notebook-local logic used later in the notebook.
- `def cfg_optional_mapping(value: object \| None, name: str) -> Mapping[str, Any]: if value is None: return {} return cfg_require_mapping(value, name)`: Defines notebook-local logic used later in the notebook.

Important functions or methods detected:
- `cast`
- `cfg_optional_mapping`
- `cfg_require_mapping`
- `isinstance`
- `normalize_feature_columns`
- `require_dict`
- `require_list`
- `strip`
- `type`
- `TypeError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def require_dict(value: Any \| None, name: str) -> Dict[str, Any]: if value is None: return {} if not isinstance(value, dict): raise TypeError( f"{name} must be a dictionary, got {t` | Defines notebook-local logic used later in the notebook. |
| `def require_list(value: Any \| None, name: str) -> List[Any]: if value is None: return [] if not isinstance(value, list): raise TypeError( f"{name} must be a list, got {type(value).` | Defines notebook-local logic used later in the notebook. |
| `def normalize_feature_columns(value: Any, name: str = "feature_columns") -> list[str]: """ Normalize feature-column results into list[str]. Handles: - list[str] - tuple[list[str], ` | Defines notebook-local logic used later in the notebook. |
| `def cfg_require_mapping(value: object, name: str) -> Mapping[str, Any]: if not isinstance(value, Mapping): raise TypeError( f"{name} must be a mapping, got {type(value).__name__}: ` | Defines notebook-local logic used later in the notebook. |
| `def cfg_optional_mapping(value: object \| None, name: str) -> Mapping[str, Any]: if value is None: return {} return cfg_require_mapping(value, name)` | Defines notebook-local logic used later in the notebook. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 03 — Load Paths, Configuration, and Runtime Settings

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
- `execution`
- `gold`
- `gold_preprocessing`
- `info`
- `load_notebook_context`
- `loaded`
- `Loaded`
- `log`
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
- `EXECUTION_CFG`
- `extra`
- `FILENAMES`
- `GOLD_CFG`
- `kind`
- `ledger`
- `log_filename`

### Key Operations

- `# Shared notebook context`: Documents the purpose or boundary of the surrounding notebook step.
- `CONTEXT_STAGE = "gold_preprocessing"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_DATASET = "pump"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LAYER = "gold"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_RUN_MODE = "train"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_PROFILE = "default"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LOG_FILE = "gold_preprocessing.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.gold.preprocessing", log_filena`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `CONTEXT_STAGE = "gold_preprocessing"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LAYER = "gold"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_RUN_MODE = "train"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LOG_FILE = "gold_preprocessing.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.gold.preprocessing", log_filena` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Shared aliases used throughout the notebook` | Documents the purpose or boundary of the surrounding notebook step. |
| `paths = CTX.paths` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_MAP = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE_CFG = CTX.stage_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_CFG = CTX.stage_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
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

## Code Cell 04 — Load Paths, Configuration, and Runtime Settings

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `Artifacts`
- `artifacts`
- `B`
- `Base`
- `build_truth_config_block`
- `capstone`
- `cast`
- `CONFIG`
- `CONFIG_RUN_MODE`
- `data_silver_train_dir`
- `dataset`
- `DATASET_CFG`
- `details`
- `entity`
- `execution_mode`
- `exist_ok`
- `f`
- `failsafes`
- `File`

### Outputs

- `ARTIFACTS_ROOT`
- `CASCADE_DEFAULTS_RESULTS_FILE_NAME_CSV`
- `CASCADE_DEFAULTS_RESULTS_FILE_NAME_PICKLE`
- `CASCADE_DEFAULTS_RESULTS_PATH_CSV`
- `CASCADE_DEFAULTS_RESULTS_PATH_PICKLE`
- `CASCADE_TUNED_RESULTS_FILE_NAME_CSV`
- `CASCADE_TUNED_RESULTS_FILE_NAME_PICKLE`
- `CASCADE_TUNED_RESULTS_PATH_CSV`
- `CASCADE_TUNED_RESULTS_PATH_PICKLE`
- `COMPARISON_FILE_NAME`
- `COMPARISON_PATH`
- `CONFIG_PROFILE`
- `DATASET_NAME`
- `DATASET_NAME_CONFIG`
- `FEATURE_REGISTRY_FILE_NAME`
- `FEATURE_REGISTRY_PATH`
- `GOLD_ARTIFACTS_PATH`
- `GOLD_DATA_PATH`
- `GOLD_FILE_NAME`
- `GOLD_FIT_DATA_PATH`

### Key Operations

- `TRUTH_CONFIG = cast(Dict[str, Any], build_truth_config_block(CONFIG))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_CONFIG["pipeline"] = PIPELINE`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `# ---- Stage details ----`: Documents the purpose or boundary of the surrounding notebook step.
- `STAGE = "gold"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `LAYER_NAME = str(GOLD_CFG["layer_name"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_VERSION = str(VERSIONS_CFG["gold"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_VERSION = str(VERSIONS_CFG["truth"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RECIPE_ID = str(GOLD_CFG["recipe_id"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `PIPELINE_MODE = str(PIPELINE["execution_mode"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RUN_MODE = str(RUNTIME_CFG.get("mode", CONFIG_RUN_MODE))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
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
| `STAGE = "gold"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LAYER_NAME = str(GOLD_CFG["layer_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_VERSION = str(VERSIONS_CFG["gold"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_VERSION = str(VERSIONS_CFG["truth"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RECIPE_ID = str(GOLD_CFG["recipe_id"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `PIPELINE_MODE = str(PIPELINE["execution_mode"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_MODE = str(RUNTIME_CFG.get("mode", CONFIG_RUN_MODE))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = str(RUNTIME_CFG.get("profile", CONFIG_PROFILE))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `DATASET_NAME_CONFIG = str(DATASET_CFG.get("name", "pump"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_NAME = DATASET_NAME_CONFIG.strip().lower()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `GOLD_PROCESS_RUN_ID = make_process_run_id( str(GOLD_CFG["process_run_id_prefix"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- W&B ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `WANDB_PROJECT = str(WANDB_CFG.get("project", "capstone"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WANDB_ENTITY = str(WANDB_CFG.get("entity", ""))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WANDB_RUN_NAME = f"{GOLD_VERSION}"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- File names ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_FILE_NAME = str(FILENAMES["silver_train_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_FILE_NAME = str(FILENAMES["gold_preprocessed_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_TRAIN_FILE_NAME = str(FILENAMES["gold_train_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_TEST_FILE_NAME = str(FILENAMES["gold_test_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_FIT_FILE_NAME = str(FILENAMES["gold_fit_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_PRESCALED_FILE_NAME = str( FILENAMES["gold_preprocessed_prescaled_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_SCALED_FILE_NAME = str( FILENAMES["gold_preprocessed_scaled_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FEATURE_REGISTRY_FILE_NAME = str(FILENAMES["feature_registry_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `IMPUTE_RECOMMENDATION_FILE_NAME = str( FILENAMES["impute_recommendation_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STAGE1_FEATURES_FILE_NAME = str(FILENAMES["stage1_features_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE2_FEATURES_FILE_NAME = str(FILENAMES["stage2_features_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE3_PRIMARY_FILE_NAME = str(FILENAMES["stage3_primary_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE3_SECONDARY_FILE_NAME = str(FILENAMES["stage3_secondary_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CASCADE_DEFAULTS_RESULTS_FILE_NAME_CSV = str( FILENAMES["cascade_defaults_results_file_name_csv"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_DEFAULTS_RESULTS_FILE_NAME_PICKLE = str( FILENAMES["cascade_defaults_results_file_name_pickle"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_TUNED_RESULTS_FILE_NAME_CSV = str( FILENAMES["cascade_tuned_results_file_name_csv"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_TUNED_RESULTS_FILE_NAME_PICKLE = str( FILENAMES["cascade_tuned_results_file_name_pickle"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `COMPARISON_FILE_NAME = str(FILENAMES["comparison_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_PREPROCESSING_LEDGER_FILE_NAME = str( FILENAMES["gold_preprocessing_ledger_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PREPROCESSING_SUMMARY_FILE_NAME = str( FILENAMES["preprocessing_summary_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PREPROCESSING_METADATA_FILE_NAME = str( FILENAMES["preprocessing_metadata_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `REFERENCE_PROFILE_FILE_NAME = str(FILENAMES["reference_profile_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Runtime knobs ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `74 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 05 — Load Paths, Configuration, and Runtime Settings

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

## Code Cell 06 — Load Paths, Configuration, and Runtime Settings

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `check`
- `context`
- `f`
- `globals`
- `Gold`
- `GOLD_CFG`
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

- `gold_required_context_vars`
- `missing_gold_context_vars`

### Key Operations

- `gold_required_context_vars = [ "GOLD_CFG",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_gold_context_vars = [ name for name in gold_required_context_vars if name not in globals()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if missing_gold_context_vars: raise NameError(f"Missing Gold context variables: {missing_gold_context_vars}")`: Controls validation, iteration, file handling, or error handling for this step.
- `logger.info("Gold context sanity check passed")`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `globals`
- `info`
- `NameError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `gold_required_context_vars = [ "GOLD_CFG",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_gold_context_vars = [ name for name in gold_required_context_vars if name not in globals()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_gold_context_vars: raise NameError(f"Missing Gold context variables: {missing_gold_context_vars}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `logger.info("Gold context sanity check passed")` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 07 — Load Paths, Configuration, and Runtime Settings

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold_preprocessing__resolved_config`
- `artifact`
- `build_artifact_dirs_from_config`
- `DATASET_NAME`
- `directories`
- `execution`
- `export_config_snapshot`
- `f`
- `features`
- `FILENAMES`
- `get`
- `Gold`
- `gold_preprocessing`
- `gold_preprocessing_ledger_file_name`
- `lineage`
- `metadata`
- `models`
- `preprocessing`
- `preprocessing_metadata_file_name`
- `preprocessing_summary_file_name`

### Outputs

- `config`
- `CONFIG_SNAPSHOT_PATH`
- `GOLD_ARTIFACTS_PATH`
- `GOLD_CONFIG_DIR`
- `GOLD_FEATURE_DIR`
- `GOLD_LINEAGE_DIR`
- `GOLD_METADATA_DIR`
- `GOLD_MODEL_DIR`
- `GOLD_PREPROCESSING_ARTIFACT_DIRS`
- `GOLD_PREPROCESSING_ROOT`
- `gold_preprocesssing_ledger_path`
- `GOLD_PROFILE_DIR`
- `GOLD_SUMMARY_DIR`
- `PREPROCESSING_METADATA_PATH`
- `PREPROCESSING_SUMMARY_PATH`
- `REFERENCE_PROFILE_PATH`
- `stage_key`
- `STAGE1_FEATURES_PATH`
- `STAGE2_FEATURES_PATH`
- `STAGE3_PRIMARY_PATH`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Gold preprocessing artifact directories`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `GOLD_PREPROCESSING_ARTIFACT_DIRS = build_artifact_dirs_from_config( config=CONFIG, stage_key="gold_preprocessing",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `GOLD_ARTIFACTS_PATH = GOLD_PREPROCESSING_ARTIFACT_DIRS["stage_dataset_root"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_PREPROCESSING_ROOT = GOLD_PREPROCESSING_ARTIFACT_DIRS["root"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_FEATURE_DIR = GOLD_PREPROCESSING_ARTIFACT_DIRS["features"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_PROFILE_DIR = GOLD_PREPROCESSING_ARTIFACT_DIRS["profiles"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_MODEL_DIR = GOLD_PREPROCESSING_ARTIFACT_DIRS["models"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_SUMMARY_DIR = GOLD_PREPROCESSING_ARTIFACT_DIRS["summaries"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_METADATA_DIR = GOLD_PREPROCESSING_ARTIFACT_DIRS["metadata"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `build_artifact_dirs_from_config`
- `export_config_snapshot`
- `get`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Gold preprocessing artifact directories` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `GOLD_PREPROCESSING_ARTIFACT_DIRS = build_artifact_dirs_from_config( config=CONFIG, stage_key="gold_preprocessing",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_ARTIFACTS_PATH = GOLD_PREPROCESSING_ARTIFACT_DIRS["stage_dataset_root"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_PREPROCESSING_ROOT = GOLD_PREPROCESSING_ARTIFACT_DIRS["root"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_FEATURE_DIR = GOLD_PREPROCESSING_ARTIFACT_DIRS["features"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_PROFILE_DIR = GOLD_PREPROCESSING_ARTIFACT_DIRS["profiles"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_MODEL_DIR = GOLD_PREPROCESSING_ARTIFACT_DIRS["models"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_SUMMARY_DIR = GOLD_PREPROCESSING_ARTIFACT_DIRS["summaries"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_METADATA_DIR = GOLD_PREPROCESSING_ARTIFACT_DIRS["metadata"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_CONFIG_DIR = GOLD_PREPROCESSING_ARTIFACT_DIRS["config"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_LINEAGE_DIR = GOLD_PREPROCESSING_ARTIFACT_DIRS["lineage"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE1_FEATURES_PATH = GOLD_FEATURE_DIR / FILENAMES["stage1_features_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE2_FEATURES_PATH = GOLD_FEATURE_DIR / FILENAMES["stage2_features_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE3_PRIMARY_PATH = GOLD_FEATURE_DIR / FILENAMES["stage3_primary_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE3_SECONDARY_PATH = GOLD_FEATURE_DIR / FILENAMES["stage3_secondary_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `REFERENCE_PROFILE_PATH = GOLD_PROFILE_DIR / FILENAMES["reference_profile_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PREPROCESSING_SUMMARY_PATH = ( GOLD_SUMMARY_DIR / FILENAMES["preprocessing_summary_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PREPROCESSING_METADATA_PATH = ( GOLD_METADATA_DIR / FILENAMES["preprocessing_metadata_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CONFIG_SNAPSHOT_PATH = ( GOLD_CONFIG_DIR / f"{DATASET_NAME}__gold_preprocessing__resolved_config.yaml"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_preprocesssing_ledger_path = ( GOLD_LINEAGE_DIR / FILENAMES["gold_preprocessing_ledger_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if CONFIG["execution"].get("save_config_snapshot", True): export_config_snapshot(CONFIG, CONFIG_SNAPSHOT_PATH)` | Controls validation, iteration, file handling, or error handling for this step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 08 — Load Paths, Configuration, and Runtime Settings

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

## Code Cell 10 — Start Logging for the Gold Preprocessing Stage

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

## Code Cell 11 — Start Logging for the Gold Preprocessing Stage

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
- `gold`
- `Gold`
- `gold_preprocessing`
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

### Outputs

- `gold_log_path`
- `level`
- `logger`
- `overwrite_handlers`

### Key Operations

- `"""`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Original Logging Setup`: Documents the purpose or boundary of the surrounding notebook step.
- `# Create gold log path`: Documents the purpose or boundary of the surrounding notebook step.
- `gold_log_path = paths.logs / "gold_preprocessing.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Initial Logger`: Documents the purpose or boundary of the surrounding notebook step.
- `configure_logging( "capstone", gold_log_path, level=logging.DEBUG, overwrite_handlers=True,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Initiate Logger and log file`: Documents the purpose or boundary of the surrounding notebook step.
- `logger = logging.getLogger("capstone.gold")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Log load and initiation`: Documents the purpose or boundary of the surrounding notebook step.
- `logger.info("Gold stage starting")`: Writes a logger message for traceability during notebook execution.
- `# Log paths loads`: Documents the purpose or boundary of the surrounding notebook step.

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
| `# Create gold log path` | Documents the purpose or boundary of the surrounding notebook step. |
| `gold_log_path = paths.logs / "gold_preprocessing.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Initial Logger` | Documents the purpose or boundary of the surrounding notebook step. |
| `configure_logging( "capstone", gold_log_path, level=logging.DEBUG, overwrite_handlers=True,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Initiate Logger and log file` | Documents the purpose or boundary of the surrounding notebook step. |
| `logger = logging.getLogger("capstone.gold")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Log load and initiation` | Documents the purpose or boundary of the surrounding notebook step. |
| `logger.info("Gold stage starting")` | Writes a logger message for traceability during notebook execution. |
| `# Log paths loads` | Documents the purpose or boundary of the surrounding notebook step. |
| `log_layer_paths(paths, current_layer="gold", logger=logger) """` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 12 — Initialize the W&B Run

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `B`
- `dataset`
- `DATASET_NAME`
- `FEATURE_REGISTRY_PATH`
- `feature_registry_path`
- `GOLD_DATA_PATH`
- `gold_output_path`
- `gold_preprocessing`
- `GOLD_VERSION`
- `gold_version`
- `IMPUTE_RECOMMENDATION_PATH`
- `impute_recommendation_path`
- `info`
- `init`
- `initialized`
- `logger`
- `s`
- `SCALER_KIND`
- `scaler_kind`
- `silver_path`

### Outputs

- `config`
- `entity`
- `job_type`
- `name`
- `project`
- `wandb_run`

### Key Operations

- `# W&B`: Documents the purpose or boundary of the surrounding notebook step.
- `wandb_run = wandb.init( project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="gold_preprocessing", config={ "gold_version": GOLD_VERSION, "dataset": DATASET_NA`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger.info("W&B initialized: %s", wandb_run.name)`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `info`
- `init`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# W&B` | Documents the purpose or boundary of the surrounding notebook step. |
| `wandb_run = wandb.init( project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="gold_preprocessing", config={ "gold_version": GOLD_VERSION, "dataset": DATASET_NA` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("W&B initialized: %s", wandb_run.name)` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- Artifact or state outputs detected: optional experiment tracking call.

## Code Cell 13 — Initialize the Gold Ledger

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

## Code Cell 14 — Initialize the Gold Ledger

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `init`
- `Initialized`
- `Original`
- `RECIPE_ID`
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

- `"""`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Original Ledger Setup`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger = Ledger(stage=STAGE, recipe_id=RECIPE_ID)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ledger.add( kind="step", step="init", message="Initialized ledger", logger=logger`: Records or exports ledger information for stage-level traceability.
- `) """`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `Ledger`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `"""` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Original Ledger Setup` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger = Ledger(stage=STAGE, recipe_id=RECIPE_ID)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger.add( kind="step", step="init", message="Initialized ledger", logger=logger` | Records or exports ledger information for stage-level traceability. |
| `) """` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 15 — Load the Silver Input and Supporting Artifacts

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `builder`
- `but`
- `clean`
- `columns`
- `contaminated`
- `copy`
- `dataframe`
- `DataFrame`
- `else`
- `eq`
- `exists`
- `Expected`
- `f`
- `FileNotFoundError`
- `first`
- `found`
- `Gold`
- `gold_train_normal_rows`
- `head`

### Outputs

- `data`
- `gold_contaminated_normal_dataframe`
- `gold_contaminated_normal_mask`
- `gold_train_normal_dataframe`
- `gold_train_normal_mask`
- `kind`
- `logger`
- `message`
- `normal_clean_dataframe`
- `normal_clean_path`
- `normal_contaminated_dataframe`
- `normal_contaminated_path`
- `profiled_silver_path`
- `silver_dataframe`
- `silver_path`
- `step`
- `USE_PROFILED_SILVER_SUBSETS`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Load profiled Silver subset outputs for Gold preprocessing`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `USE_PROFILED_SILVER_SUBSETS = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if USE_PROFILED_SILVER_SUBSETS: profiled_silver_path = SILVER_PROFILED_DATAFRAME_DATA_PATH normal_clean_path = SILVER_NORMAL_CLEAN_DATA_PATH normal_contaminated_path = SILVER_NORMA`: Writes a logger message for traceability during notebook execution.
- `else: silver_path = SILVER_TRAIN_DATA_PATH logger.info("Loading original Silver parquet: %s", silver_path) silver_dataframe = load_data(silver_path) normal_clean_dataframe = pd.Dat`: Writes a logger message for traceability during notebook execution.
- `logger.info("Silver shape=%s", silver_dataframe.shape)`: Writes a logger message for traceability during notebook execution.
- `ledger.add( kind="step", step="load_profiled_silver_inputs", message="Loaded profiled Silver parquet and subset outputs for Gold preprocessing.", data={ "silver_path": str(silver_p`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(silver_dataframe.head(3))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `copy`
- `DataFrame`
- `display`
- `eq`
- `exists`
- `FileNotFoundError`
- `head`
- `info`
- `KeyError`
- `load_data`
- `read_parquet`
- `Series`
- `warning`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Load profiled Silver subset outputs for Gold preprocessing` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `USE_PROFILED_SILVER_SUBSETS = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if USE_PROFILED_SILVER_SUBSETS: profiled_silver_path = SILVER_PROFILED_DATAFRAME_DATA_PATH normal_clean_path = SILVER_NORMAL_CLEAN_DATA_PATH normal_contaminated_path = SILVER_NORMA` | Writes a logger message for traceability during notebook execution. |
| `else: silver_path = SILVER_TRAIN_DATA_PATH logger.info("Loading original Silver parquet: %s", silver_path) silver_dataframe = load_data(silver_path) normal_clean_dataframe = pd.Dat` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Silver shape=%s", silver_dataframe.shape)` | Writes a logger message for traceability during notebook execution. |
| `ledger.add( kind="step", step="load_profiled_silver_inputs", message="Loaded profiled Silver parquet and subset outputs for Gold preprocessing.", data={ "silver_path": str(silver_p` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(silver_dataframe.head(3))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 16 — Resolve the Parent Truth Record and Confirm the Dataset Identity

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__silver__feature_registry`
- `a`
- `artifacts`
- `artifacts_root`
- `astype`
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
- `EDA`
- `else`
- `extract_truth_hash`
- `f`
- `Feature`

### Outputs

- `column_name`
- `dataframe`
- `dataset_name`
- `DATASET_NAME`
- `FEATURE_REGISTRY_PATH`
- `GOLD_PARENT_TRUTH_HASH`
- `gold_truth`
- `IMPUTE_RECOMMENDATION_PATH`
- `layer_name`
- `parent_layer_name`
- `PARENT_PIPELINE_MODE`
- `parent_truth_hash`
- `PIPELINE_MODE`
- `pipeline_mode`
- `process_run_id`
- `SILVER_DATASET_NAME`
- `silver_eda_artifacts_dir`
- `silver_truth`
- `truth_dir`
- `truth_version`

### Key Operations

- `GOLD_PARENT_TRUTH_HASH = extract_truth_hash(silver_dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if GOLD_PARENT_TRUTH_HASH is None: raise ValueError("Gold preprocessing input dataframe does not contain a readable meta__truth_hash value.")`: Controls validation, iteration, file handling, or error handling for this step.
- `SILVER_DATASET_NAME = ( silver_dataframe["meta__dataset"] .dropna() .astype("string") .str.strip()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `SILVER_DATASET_NAME = SILVER_DATASET_NAME[SILVER_DATASET_NAME != ""]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if len(SILVER_DATASET_NAME) == 0: raise ValueError("Gold preprocessing input dataframe is missing usable meta__dataset values.")`: Controls validation, iteration, file handling, or error handling for this step.
- `SILVER_DATASET_NAME = str(SILVER_DATASET_NAME.iloc[0]).strip()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `silver_truth = load_parent_truth_record_from_dataframe( dataframe=silver_dataframe, truth_dir=TRUTHS_PATH, parent_layer_name="silver", dataset_name=SILVER_DATASET_NAME, column_name`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `DATASET_NAME = get_dataset_name_from_truth(silver_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_PARENT_TRUTH_HASH = get_truth_hash(silver_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PARENT_PIPELINE_MODE = get_pipeline_mode_from_truth(silver_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `astype`
- `dropna`
- `extract_truth_hash`
- `fillna`
- `get_artifact_path_from_truth`
- `get_dataset_name_from_truth`
- `get_pipeline_mode_from_truth`
- `get_truth_hash`
- `info`
- `initialize_layer_truth`
- `load_parent_truth_record_from_dataframe`
- `Path`
- `strip`
- `update_truth_section`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `GOLD_PARENT_TRUTH_HASH = extract_truth_hash(silver_dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if GOLD_PARENT_TRUTH_HASH is None: raise ValueError("Gold preprocessing input dataframe does not contain a readable meta__truth_hash value.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `SILVER_DATASET_NAME = ( silver_dataframe["meta__dataset"] .dropna() .astype("string") .str.strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SILVER_DATASET_NAME = SILVER_DATASET_NAME[SILVER_DATASET_NAME != ""]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(SILVER_DATASET_NAME) == 0: raise ValueError("Gold preprocessing input dataframe is missing usable meta__dataset values.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `SILVER_DATASET_NAME = str(SILVER_DATASET_NAME.iloc[0]).strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `silver_truth = load_parent_truth_record_from_dataframe( dataframe=silver_dataframe, truth_dir=TRUTHS_PATH, parent_layer_name="silver", dataset_name=SILVER_DATASET_NAME, column_name` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DATASET_NAME = get_dataset_name_from_truth(silver_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_PARENT_TRUTH_HASH = get_truth_hash(silver_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PARENT_PIPELINE_MODE = get_pipeline_mode_from_truth(silver_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if PARENT_PIPELINE_MODE is not None: PIPELINE_MODE = PARENT_PIPELINE_MODE` | Controls validation, iteration, file handling, or error handling for this step. |
| `FEATURE_REGISTRY_PATH = Path(get_artifact_path_from_truth(silver_truth, "feature_registry_dir")) / "registry" / f"{DATASET_NAME}__silver__feature_registry.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `silver_eda_artifacts_dir = Path(RESOLVED_PATHS["artifacts_root"]) / "silver_eda" / DATASET_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `IMPUTE_RECOMMENDATION_PATH = silver_eda_artifacts_dir / FILENAMES["impute_recommendation_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if "meta__pipeline_mode" not in silver_dataframe.columns: silver_dataframe["meta__pipeline_mode"] = PIPELINE_MODE` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: silver_dataframe["meta__pipeline_mode"] = silver_dataframe["meta__pipeline_mode"].fillna(PIPELINE_MODE)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth = initialize_layer_truth( truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name=LAYER_NAME, process_run_id=GOLD_PROCESS_RUN_ID, pipeline_mode=PIPELINE_MODE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth = update_truth_section( gold_truth, "config_snapshot", { "gold_version": GOLD_VERSION, "recipe_id": RECIPE_ID, "dataset_name_config": DATASET_NAME_CONFIG, "dataset_name_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "parent_layer_name": "silver", "parent_truth_hash": GOLD_PARENT_TRUTH_HASH, "dataset_name_from_parent_truth": DATA` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("Resolved Silver parent truth hash: %s", GOLD_PARENT_TRUTH_HASH)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Resolved Gold preprocessing dataset name from Silver truth: %s", DATASET_NAME)` | Writes a logger message for traceability during notebook execution. |
| `print("Gold preprocessing parent truth hash:", GOLD_PARENT_TRUTH_HASH)` | Displays a notebook-facing result for inspection. |
| `print("Gold preprocessing dataset name from parent truth:", DATASET_NAME)` | Displays a notebook-facing result for inspection. |
| `print("Feature registry path from Silver truth:", FEATURE_REGISTRY_PATH)` | Displays a notebook-facing result for inspection. |
| `print("Impute recommendation path from Silver EDA artifacts:", IMPUTE_RECOMMENDATION_PATH)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 17 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `after`
- `After`
- `are`
- `be`
- `bool`
- `but`
- `column`
- `column_name`
- `columns`
- `count`
- `dataframe`
- `does`
- `empty`
- `exist`
- `exists`
- `Expected`
- `f`
- `Feature`
- `feature`

### Outputs

- `data`
- `default`
- `feature_columns`
- `feature_columns_raw`
- `feature_registry`
- `feature_registry_raw`
- `feature_set_id`
- `imputation_recommendation`
- `imputation_recommendation_raw`
- `kind`
- `logger`
- `message`
- `missing_feature_columns`
- `raise_if_missing`
- `recommended_imputation`
- `STAGE2_FEATURE_COLUMNS`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Load feature registry and imputation recommendation`: Documents the purpose or boundary of the surrounding notebook step.
- `# after resolving Silver parent truth`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `if not FEATURE_REGISTRY_PATH.exists(): raise FileNotFoundError( "Feature registry path was resolved from Silver truth, but the file does not exist.\n" f"Expected: {FEATURE_REGISTRY`: Controls validation, iteration, file handling, or error handling for this step.
- `logger.info( "Loading feature registry from resolved Silver truth path: %s", FEATURE_REGISTRY_PATH,`: Writes a logger message for traceability during notebook execution.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `feature_registry_raw = load_json(FEATURE_REGISTRY_PATH)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `feature_registry = require_dict( feature_registry_raw, "feature_registry",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `feature_columns_raw = require_list( feature_registry.get("feature_columns"), "feature_registry['feature_columns']",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `bool`
- `exists`
- `FileNotFoundError`
- `get`
- `info`
- `load_json`
- `normalize_feature_columns`
- `require_dict`
- `require_list`
- `strip`
- `ValueError`
- `warning`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Load feature registry and imputation recommendation` | Documents the purpose or boundary of the surrounding notebook step. |
| `# after resolving Silver parent truth` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `if not FEATURE_REGISTRY_PATH.exists(): raise FileNotFoundError( "Feature registry path was resolved from Silver truth, but the file does not exist.\n" f"Expected: {FEATURE_REGISTRY` | Controls validation, iteration, file handling, or error handling for this step. |
| `logger.info( "Loading feature registry from resolved Silver truth path: %s", FEATURE_REGISTRY_PATH,` | Writes a logger message for traceability during notebook execution. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_registry_raw = load_json(FEATURE_REGISTRY_PATH)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `feature_registry = require_dict( feature_registry_raw, "feature_registry",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_columns_raw = require_list( feature_registry.get("feature_columns"), "feature_registry['feature_columns']",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_columns = [ str(column_name).strip() for column_name in feature_columns_raw if str(column_name).strip() != ""` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_set_id = str( feature_registry.get("feature_set_id", "unknown_feature_set")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if len(feature_columns) == 0: raise ValueError( "Feature registry loaded successfully, but feature_columns is empty.\n" f"Feature registry path: {FEATURE_REGISTRY_PATH}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `missing_feature_columns = [ column_name for column_name in feature_columns if column_name not in silver_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_feature_columns: logger.warning( "Some feature registry columns are missing from the Gold input dataframe and will be skipped: %s", missing_feature_columns[:20], )` | Writes a logger message for traceability during notebook execution. |
| `feature_columns = [ column_name for column_name in feature_columns if column_name in silver_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if len(feature_columns) == 0: raise ValueError( "After intersecting the feature registry with the Gold input dataframe, " "no usable feature columns remained." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `STAGE2_FEATURE_COLUMNS = normalize_feature_columns( feature_columns, "STAGE2_FEATURE_COLUMNS",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("Loading imputation recommendation: %s", IMPUTE_RECOMMENDATION_PATH)` | Writes a logger message for traceability during notebook execution. |
| `imputation_recommendation_raw = load_json( IMPUTE_RECOMMENDATION_PATH, raise_if_missing=False, default={},` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `imputation_recommendation = require_dict( imputation_recommendation_raw, "imputation_recommendation",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `recommended_imputation = str( imputation_recommendation.get( "recommendation", "forward_fill_within_group_then_median", )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="load_feature_registry_and_imputation_recommendation", message="Loaded feature registry and imputation recommendation after resolving Silver parent tr` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Loaded feature registry:", FEATURE_REGISTRY_PATH)` | Displays a notebook-facing result for inspection. |
| `print("Feature set ID:", feature_set_id)` | Displays a notebook-facing result for inspection. |
| `print("Feature column count:", len(STAGE2_FEATURE_COLUMNS))` | Displays a notebook-facing result for inspection. |
| `print("First 10 feature columns:", STAGE2_FEATURE_COLUMNS[:10])` | Displays a notebook-facing result for inspection. |
| `print("Recommended imputation:", recommended_imputation)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 18 — Create the Working Gold Dataframe and Start Runtime Tracking

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `copy`
- `else`
- `gold_input_source`
- `gold_training_normal_source`
- `gold_version`
- `GOLD_VERSION`
- `machine_status__profiled`
- `normal_clean`
- `now`
- `original_silver_normal_logic`
- `preprocessing_recipe_id`
- `processed_at_utc`
- `RECIPE_ID`
- `runtime_facts`
- `silver_dataframe`
- `silver_subset_profiled_dataframe`
- `Timestamp`
- `tz`
- `update_truth_section`
- `USE_PROFILED_SILVER_SUBSETS`

### Outputs

- `dataframe`
- `GOLD_PROCESSED_AT_UTC`
- `gold_truth`
- `gold_working_dataframe`

### Key Operations

- `dataframe = silver_dataframe.copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_PROCESSED_AT_UTC = pd.Timestamp.now(tz="UTC")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "processed_at_utc": GOLD_PROCESSED_AT_UTC, "gold_version": GOLD_VERSION, "preprocessing_recipe_id": RECIPE_ID, "go`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_working_dataframe = dataframe.copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `copy`
- `now`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `dataframe = silver_dataframe.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_PROCESSED_AT_UTC = pd.Timestamp.now(tz="UTC")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "processed_at_utc": GOLD_PROCESSED_AT_UTC, "gold_version": GOLD_VERSION, "preprocessing_recipe_id": RECIPE_ID, "go` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_working_dataframe = dataframe.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 19 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `contaminated`
- `dropna`
- `gold`
- `gold_contaminated_normal_dataframe`
- `gold_train_normal_dataframe`
- `machine_status__profiled`
- `name`
- `normal`
- `profiled`
- `rename_axis`
- `reset_index`
- `row_count`
- `shape`
- `silver`
- `silver_dataframe`
- `train`
- `USE_PROFILED_SILVER_SUBSETS`
- `value_counts`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `i`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `display`
- `rename_axis`
- `reset_index`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `i` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 20 — Stamp Stable Gold Row Identity

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `before`
- `bool`
- `dataframe`
- `downstream`
- `ensure_stable_row_id`
- `Gold`
- `identity`
- `is_unique`
- `isna`
- `ledger`
- `meta__row_id`
- `onto`
- `row`
- `row_count`
- `row_id_null_count`
- `row_id_unique`
- `row_tracking`
- `runtime_facts`
- `stable`

### Outputs

- `data`
- `gold_truth`
- `gold_working_dataframe`
- `kind`
- `logger`
- `message`
- `row_id_column`
- `step`

### Key Operations

- `gold_working_dataframe = ensure_stable_row_id( gold_working_dataframe, row_id_column="meta__row_id",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="ensure_stable_row_id", message="Stamped stable row identity onto the Gold working dataframe before downstream transformations.", data={ "row_id_colum`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "row_tracking": { "row_id_column": "meta__row_id", "row_count": int(len(gold_working_dataframe)), "row_id_unique":`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `bool`
- `ensure_stable_row_id`
- `isna`
- `sum`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `gold_working_dataframe = ensure_stable_row_id( gold_working_dataframe, row_id_column="meta__row_id",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="ensure_stable_row_id", message="Stamped stable row identity onto the Gold working dataframe before downstream transformations.", data={ "row_id_colum` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "row_tracking": { "row_id_column": "meta__row_id", "row_count": int(len(gold_working_dataframe)), "row_id_unique":` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 21 — Reload the Parent Truth Record for Direct Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `gold_working_dataframe`
- `load_parent_truth_record_from_dataframe`
- `meta__truth_hash`
- `silver`
- `TRUTHS_PATH`

### Outputs

- `column_name`
- `dataframe`
- `dataset_name`
- `parent_layer_name`
- `silver_truth`
- `truth_dir`

### Key Operations

- `silver_truth = load_parent_truth_record_from_dataframe( dataframe=gold_working_dataframe, truth_dir=TRUTHS_PATH, parent_layer_name="silver", dataset_name=DATASET_NAME, column_name=`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `load_parent_truth_record_from_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `silver_truth = load_parent_truth_record_from_dataframe( dataframe=gold_working_dataframe, truth_dir=TRUTHS_PATH, parent_layer_name="silver", dataset_name=DATASET_NAME, column_name=` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 22 — Define the Episode-Based Train/Test Split Logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__row_order_for_episode_split`
- `a`
- `an`
- `any`
- `arange`
- `at`
- `At`
- `be`
- `become`
- `belongs`
- `between`
- `break`
- `build`
- `Build`
- `by`
- `candidate_column`
- `Cannot`
- `chronological`
- `column`
- `columns`

### Outputs

- `build_episode_based_split_mask`
- `episode_order`
- `missing_episode_count`
- `n_episodes`
- `order_column`
- `ordered_episodes`
- `split_info`
- `test_episode_set`
- `test_episodes`
- `train_episode_count`
- `train_episode_set`
- `train_episodes`
- `train_mask`
- `working_dataframe`

### Key Operations

- `def build_episode_based_split_mask( dataframe: pd.DataFrame, *, train_fraction: float, episode_column: str = "meta__episode_id", order_column: str \| None = None,`: Defines notebook-local logic used later in the notebook.
- `) -> tuple[pd.Series, dict]: """ Build a train/test mask at the episode level. The split is chronological by episode start order, not by raw episode id sorting. - Earlier episodes `: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `any`
- `arange`
- `build_episode_based_split_mask`
- `copy`
- `floor`
- `groupby`
- `isin`
- `isna`
- `max`
- `min`
- `sort_values`
- `sum`
- `tolist`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_episode_based_split_mask( dataframe: pd.DataFrame, *, train_fraction: float, episode_column: str = "meta__episode_id", order_column: str \| None = None,` | Defines notebook-local logic used later in the notebook. |
| `) -> tuple[pd.Series, dict]: """ Build a train/test mask at the episode level. The split is chronological by episode start order, not by raw episode id sorting. - Earlier episodes ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 23 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `gold_working_dataframe`

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

## Code Cell 24 — Build the Train/Test Split Mask

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `break`
- `build_episode_based_split_mask`
- `candidate_column`
- `chronological`
- `columns`
- `Created`
- `DataFrame`
- `episode`
- `episode_split_info`
- `event_step`
- `Gold`
- `gold_working_dataframe`
- `ledger`
- `level`
- `meta__episode_id`
- `meta__row_id`
- `modeling`
- `runtime_facts`
- `split`

### Outputs

- `data`
- `episode_column`
- `gold_truth`
- `kind`
- `logger`
- `message`
- `order_column`
- `split_order_column`
- `step`
- `train_fraction`

### Key Operations

- `split_order_column = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `for candidate_column in ["time_index", "event_step", "meta__row_id"]: if candidate_column in gold_working_dataframe.columns: split_order_column = candidate_column break`: Controls validation, iteration, file handling, or error handling for this step.
- `train_mask, split_info = build_episode_based_split_mask( gold_working_dataframe, train_fraction=TRAIN_FRACTION, episode_column="meta__episode_id", order_column=split_order_column,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="build_episode_based_split_mask", message="Created chronological episode-level train/test split for Gold modeling.", data=split_info, logger=logger,`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "episode_split_info": split_info, },`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(pd.DataFrame([split_info]))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `build_episode_based_split_mask`
- `DataFrame`
- `display`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `split_order_column = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for candidate_column in ["time_index", "event_step", "meta__row_id"]: if candidate_column in gold_working_dataframe.columns: split_order_column = candidate_column break` | Controls validation, iteration, file handling, or error handling for this step. |
| `train_mask, split_info = build_episode_based_split_mask( gold_working_dataframe, train_fraction=TRAIN_FRACTION, episode_column="meta__episode_id", order_column=split_order_column,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="build_episode_based_split_mask", message="Created chronological episode-level train/test split for Gold modeling.", data=split_info, logger=logger,` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "episode_split_info": split_info, },` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(pd.DataFrame([split_info]))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 25 — Define a Helper to Stamp Training Metadata

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Adds`
- `astype`
- `bool`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `does`
- `else`
- `fillna`
- `isinstance`
- `length`
- `match`
- `meta__is_train_flag`
- `ndarray`
- `onto`
- `raise`
- `reindex`
- `Series`
- `split`

### Outputs

- `dtype`
- `index`
- `stamp_training_metadata`
- `train_mask_aligned`
- `working_dataframe`

### Key Operations

- `def stamp_training_metadata( dataframe: pd.DataFrame, train_mask: pd.Series \| np.ndarray \| None,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """ Stamp the train/test split onto the dataframe. Adds: - meta__is_train_flag """ working_dataframe = dataframe.copy() if train_mask is None: return working_dat`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `copy`
- `fillna`
- `isinstance`
- `reindex`
- `Series`
- `stamp_training_metadata`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def stamp_training_metadata( dataframe: pd.DataFrame, train_mask: pd.Series \| np.ndarray \| None,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Stamp the train/test split onto the dataframe. Adds: - meta__is_train_flag """ working_dataframe = dataframe.copy() if train_mask is None: return working_dat` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 26 — Stamp Train/Test Metadata onto the Working Dataframe

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `add_train_flag`
- `flag`
- `ledger`
- `level`
- `metadata`
- `only`
- `row`
- `runtime_facts`
- `split`
- `split_info`
- `stamp_training_metadata`
- `Stamped`
- `Store`
- `to`
- `train`
- `Truth`
- `update_truth_section`
- `was`
- `written`

### Outputs

- `data`
- `gold_truth`
- `gold_working_dataframe`
- `kind`
- `logger`
- `message`
- `step`
- `train_mask`

### Key Operations

- `gold_working_dataframe = stamp_training_metadata( gold_working_dataframe, train_mask=train_mask,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "split_info": split_info, },`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="add_train_flag", message="Stamped only row-level train flag; split metadata was written to Truth Store.", data=split_info, logger=logger,`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `stamp_training_metadata`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `gold_working_dataframe = stamp_training_metadata( gold_working_dataframe, train_mask=train_mask,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "split_info": split_info, },` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="add_train_flag", message="Stamped only row-level train flag; split metadata was written to Truth Store.", data=split_info, logger=logger,` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 27 — Define the Numeric Feature Selection Logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `api`
- `append`
- `columns`
- `continue`
- `DataFrame`
- `dataframe`
- `def`
- `feature_columns`
- `feature_name`
- `is_numeric_dtype`
- `numeric_feature_columns`
- `types`

### Outputs

- `select_numeric_feature_columns`

### Key Operations

- `def select_numeric_feature_columns( dataframe: pd.DataFrame, *, feature_columns: list[str],`: Defines notebook-local logic used later in the notebook.
- `) -> list[str]: numeric_feature_columns: list[str] = [] for feature_name in feature_columns: if feature_name not in dataframe.columns: continue if pd.api.types.is_numeric_dtype(dat`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append`
- `is_numeric_dtype`
- `select_numeric_feature_columns`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def select_numeric_feature_columns( dataframe: pd.DataFrame, *, feature_columns: list[str],` | Defines notebook-local logic used later in the notebook. |
| `) -> list[str]: numeric_feature_columns: list[str] = [] for feature_name in feature_columns: if feature_name not in dataframe.columns: continue if pd.api.types.is_numeric_dtype(dat` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 28 — Select the Numeric Feature Set for Gold Modeling

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `add`
- `Capture`
- `columns`
- `dataframe`
- `feature`
- `ledger`
- `Need`
- `numeric`
- `of`
- `select_numeric_feature_columns`
- `shape`
- `silver_dataframe`
- `silver_dataframe_path`
- `silver_dataframe_shape`
- `silver_path`
- `the`
- `TODO`

### Outputs

- `data`
- `feature_columns`
- `kind`
- `logger`
- `message`
- `numeric_feature_columns`
- `step`

### Key Operations

- `numeric_feature_columns = select_numeric_feature_columns( silver_dataframe, feature_columns=feature_columns,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `# TODO: Need Logger`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger.add( kind="step", step="select_numeric_feature_columns", message="Capture the a list of the numeric feature columns from the dataframe", data={ "silver_dataframe_path": str(`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `add`
- `select_numeric_feature_columns`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `numeric_feature_columns = select_numeric_feature_columns( silver_dataframe, feature_columns=feature_columns,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# TODO: Need Logger` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger.add( kind="step", step="select_numeric_feature_columns", message="Capture the a list of the numeric feature columns from the dataframe", data={ "silver_dataframe_path": str(` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 29 — Define the One-Hot Encoding Logic Based on Upstream Truth

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `after`
- `Apply`
- `bool`
- `category`
- `column_name`
- `containing`
- `copy`
- `create`
- `DataFrame`
- `dataframe`
- `Dataframe`
- `def`
- `default`
- `drop`
- `encoded`
- `encoded_column_names`
- `encoding`
- `first`
- `get`

### Outputs

- `apply_one_hot_encoding_from_truths`
- `available_encoding_columns`
- `columns`
- `drop_first`
- `dtype`
- `dummy_na`
- `encoded_gold_dataframe`
- `needs_one_hot_encoding`
- `one_hot_encoding_columns`

### Key Operations

- `def apply_one_hot_encoding_from_truths( gold_dataframe: pd.DataFrame, *, upstream_truth_record: dict, drop_first: bool = False, dummy_na: bool = False,`: Defines notebook-local logic used later in the notebook.
- `) -> tuple[pd.DataFrame, list[str]]: """ Apply one-hot encoding in Gold using the encoding instructions saved in the upstream truth record. Parameters ---------- gold_dataframe : p`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `apply_one_hot_encoding_from_truths`
- `bool`
- `copy`
- `get`
- `get_dummies`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def apply_one_hot_encoding_from_truths( gold_dataframe: pd.DataFrame, *, upstream_truth_record: dict, drop_first: bool = False, dummy_na: bool = False,` | Defines notebook-local logic used later in the notebook. |
| `) -> tuple[pd.DataFrame, list[str]]: """ Apply one-hot encoding in Gold using the encoding instructions saved in the upstream truth record. Parameters ---------- gold_dataframe : p` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 30 — Apply One-Hot Encoding When Required

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `applied_one_hot_encoding_columns`
- `apply_one_hot_encoding_from_truths`
- `bool`
- `get`
- `gold_truth`
- `gold_working_dataframe`
- `needs_one_hot_encoding`
- `one_hot_encoding_columns`
- `silver_truth`

### Outputs

- `drop_first`
- `dummy_na`
- `gold_dataframe`
- `upstream_truth_record`

### Key Operations

- `gold_working_dataframe, applied_one_hot_encoding_columns = apply_one_hot_encoding_from_truths( gold_dataframe=gold_working_dataframe, upstream_truth_record=silver_truth, drop_first`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_truth["needs_one_hot_encoding"] = bool( silver_truth.get("needs_one_hot_encoding", False)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_truth["one_hot_encoding_columns"] = silver_truth.get( "one_hot_encoding_columns", []`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_truth["applied_one_hot_encoding_columns"] = applied_one_hot_encoding_columns`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `apply_one_hot_encoding_from_truths`
- `bool`
- `get`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `gold_working_dataframe, applied_one_hot_encoding_columns = apply_one_hot_encoding_from_truths( gold_dataframe=gold_working_dataframe, upstream_truth_record=silver_truth, drop_first` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth["needs_one_hot_encoding"] = bool( silver_truth.get("needs_one_hot_encoding", False)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth["one_hot_encoding_columns"] = silver_truth.get( "one_hot_encoding_columns", []` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth["applied_one_hot_encoding_columns"] = applied_one_hot_encoding_columns` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 31 — Define the Imputation Logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__original_row_order_for_imputation`
- `append`
- `Apply`
- `arange`
- `are`
- `asset`
- `astype`
- `because`
- `bool`
- `Cannot`
- `columns`
- `copy`
- `damage`
- `dataframe`
- `DataFrame`
- `Decide`
- `def`
- `define`
- `does`
- `drop`

### Outputs

- `aligned_train_mask`
- `applied_method`
- `apply_imputation`
- `grouping_columns`
- `imputation_info`
- `mean_value`
- `median_value`
- `missing_feature_columns`
- `ordering_column`
- `stats_dataframe`
- `working_dataframe`

### Key Operations

- `def apply_imputation( dataframe: pd.DataFrame, *, numeric_feature_columns: list[str], method: str, train_mask: pd.Series \| None = None,`: Defines notebook-local logic used later in the notebook.
- `) -> tuple[pd.DataFrame, dict]: """ Apply numeric feature imputation while preserving the original row order. Supported methods ----------------- - forward_fill_within_group_then_m`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append`
- `apply_imputation`
- `arange`
- `astype`
- `copy`
- `drop`
- `ffill`
- `fillna`
- `groupby`
- `isna`
- `KeyError`
- `mean`
- `median`
- `reindex`
- `reset_index`
- `sort_values`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def apply_imputation( dataframe: pd.DataFrame, *, numeric_feature_columns: list[str], method: str, train_mask: pd.Series \| None = None,` | Defines notebook-local logic used later in the notebook. |
| `) -> tuple[pd.DataFrame, dict]: """ Apply numeric feature imputation while preserving the original row order. Supported methods ----------------- - forward_fill_within_group_then_m` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 32 — Apply Numeric Feature Imputation

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `apply_imputation`
- `astype`
- `bool`
- `forward_fill_within_group_then_median`
- `gold_working_dataframe`
- `imputation_info`
- `meta__is_train_flag`
- `recommended_imputation`
- `runtime_facts`
- `update_truth_section`

### Outputs

- `gold_truth`
- `method`
- `numeric_feature_columns`
- `train_mask`
- `train_mask_for_stats`

### Key Operations

- `train_mask_for_stats = gold_working_dataframe["meta__is_train_flag"].astype(bool)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `gold_working_dataframe, imputation_info = apply_imputation( gold_working_dataframe, numeric_feature_columns=numeric_feature_columns, method="forward_fill_within_group_then_median",`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "imputation_info": imputation_info, "recommended_imputation": recommended_imputation, },`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `apply_imputation`
- `astype`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `train_mask_for_stats = gold_working_dataframe["meta__is_train_flag"].astype(bool)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_working_dataframe, imputation_info = apply_imputation( gold_working_dataframe, numeric_feature_columns=numeric_feature_columns, method="forward_fill_within_group_then_median",` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "imputation_info": imputation_info, "recommended_imputation": recommended_imputation, },` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 33 — Rebuild the Training Mask After Imputation

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `add`
- `after`
- `astype`
- `because`
- `bool`
- `boolean`
- `changed`
- `column`
- `Creating`
- `do`
- `flag`
- `fresh`
- `gold_working_dataframe`
- `have`
- `imputation`
- `index`
- `ledger`
- `mask`
- `may`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `step`
- `train_mask_flag`
- `train_mask_flag_dict`

### Key Operations

- `# Rebuild a fresh Series mask from the stamped column`: Documents the purpose or boundary of the surrounding notebook step.
- `# We have to do this because the index may have changed after imputation`: Documents the purpose or boundary of the surrounding notebook step.
- `train_mask_flag = gold_working_dataframe["meta__is_train_flag"].astype(bool)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `# TODO: Need Logger`: Documents the purpose or boundary of the surrounding notebook step.
- `train_mask_flag_dict = train_mask_flag.to_dict()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ledger.add( kind="step", step="training_mask_flag", message="Creating a mask from the meta__is_train_flag boolean flag", data={ "training_mask_flag_count": train_mask_flag.shape, "`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `add`
- `astype`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Rebuild a fresh Series mask from the stamped column` | Documents the purpose or boundary of the surrounding notebook step. |
| `# We have to do this because the index may have changed after imputation` | Documents the purpose or boundary of the surrounding notebook step. |
| `train_mask_flag = gold_working_dataframe["meta__is_train_flag"].astype(bool)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# TODO: Need Logger` | Documents the purpose or boundary of the surrounding notebook step. |
| `train_mask_flag_dict = train_mask_flag.to_dict()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger.add( kind="step", step="training_mask_flag", message="Creating a mask from the meta__is_train_flag boolean flag", data={ "training_mask_flag_count": train_mask_flag.shape, "` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 34 — Freeze a Prescaled Copy of the Gold Dataframe

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `copy`
- `dataframe`
- `deferred`
- `finalized`
- `Gold`
- `gold_preprocessed_prescaled_shape`
- `gold_working_dataframe`
- `ledger`
- `lineage`
- `memory`
- `prepare_gold_preprocessed_prescaled_dataframe`
- `Prepared`
- `prescaled`
- `Save`
- `shape`
- `truth`
- `until`

### Outputs

- `data`
- `gold_preprocessed_prescaled_dataframe`
- `kind`
- `logger`
- `message`
- `step`

### Key Operations

- `gold_preprocessed_prescaled_dataframe = gold_working_dataframe.copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ledger.add( kind="step", step="prepare_gold_preprocessed_prescaled_dataframe", message="Prepared Gold prescaled dataframe in memory. Save is deferred until truth lineage is finaliz`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(gold_preprocessed_prescaled_dataframe.shape)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `copy`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `gold_preprocessed_prescaled_dataframe = gold_working_dataframe.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger.add( kind="step", step="prepare_gold_preprocessed_prescaled_dataframe", message="Prepared Gold prescaled dataframe in memory. Save is deferred until truth lineage is finaliz` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(gold_preprocessed_prescaled_dataframe.shape)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 35 — Define the Scaler Factory

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `def`
- `elif`
- `else`
- `f`
- `lower`
- `minmax`
- `MinMaxScaler`
- `raise`
- `robust`
- `RobustScaler`
- `scaler`
- `standard`
- `StandardScaler`
- `Unknown`
- `Use`
- `ValueError`

### Outputs

- `kind`
- `make_scaler`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `lower`
- `make_scaler`
- `MinMaxScaler`
- `RobustScaler`
- `StandardScaler`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 36 — Define the Scaling Workflow

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `add`
- `all`
- `apply`
- `artifacts_path`
- `available`
- `Check`
- `config`
- `copy`
- `dataframe`
- `DataFrame`
- `dataset_name`
- `def`
- `dump`
- `else`
- `empty`
- `exist_ok`
- `explicit`
- `feature_columns`
- `feature_count`

### Outputs

- `allow_val_change`
- `data`
- `dataset`
- `fit_and_apply_scaler`
- `fit_mask`
- `fit_rows`
- `fit_source`
- `kind`
- `logger`
- `message`
- `scaled_dataframe`
- `scaler`
- `scaler_filename`
- `scaler_kind`
- `scaler_path`
- `step`

### Key Operations

- `def fit_and_apply_scaler( dataframe: pd.DataFrame, feature_columns: Sequence[str], train_mask: pd.Series, normal_only_mask: Optional[pd.Series], scaler_kind: str, artifacts_path: P`: Defines notebook-local logic used later in the notebook.
- `) -> Tuple[pd.DataFrame, Path]: """ Fit a scaler on normal-only train rows and apply it to all rows. """ if normal_only_mask is not None: fit_mask = train_mask & normal_only_mask f`: Records or exports ledger information for stage-level traceability.

Important functions or methods detected:
- `add`
- `copy`
- `dump`
- `fit`
- `fit_and_apply_scaler`
- `format`
- `log`
- `lower`
- `make_scaler`
- `mkdir`
- `train`
- `transform`
- `update`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def fit_and_apply_scaler( dataframe: pd.DataFrame, feature_columns: Sequence[str], train_mask: pd.Series, normal_only_mask: Optional[pd.Series], scaler_kind: str, artifacts_path: P` | Defines notebook-local logic used later in the notebook. |
| `) -> Tuple[pd.DataFrame, Path]: """ Fit a scaler on normal-only train rows and apply it to all rows. """ if normal_only_mask is not None: fit_mask = train_mask & normal_only_mask f` | Records or exports ledger information for stage-level traceability. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: model artifact, optional experiment tracking call.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 37 — Scale the Gold Feature Set

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `anomaly_flag`
- `build`
- `build_gold_model_ready_dataframe`
- `Built`
- `columns`
- `details`
- `else`
- `eq`
- `f`
- `feature_set_id`
- `fit_and_apply_scaler`
- `Gold`
- `GOLD_MODEL_DIR`
- `gold_preprocessed_scaled_dataframe`
- `gold_working_dataframe`
- `machine_status__profiled`
- `model`
- `normal_clean`
- `numeric_feature_columns`

### Outputs

- `artifacts_path`
- `data`
- `dataframe`
- `dataset_name`
- `feature_columns`
- `gold_build_info`
- `gold_truth`
- `kind`
- `ledger`
- `logger`
- `message`
- `normal_only_mask`
- `scaler_kind`
- `step`
- `train_mask`

### Key Operations

- `normal_only_mask = ( gold_working_dataframe["machine_status__profiled"].eq("normal_clean") if "machine_status__profiled" in gold_working_dataframe.columns else ( (gold_working_data`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_preprocessed_scaled_dataframe, scaler_path = fit_and_apply_scaler( dataframe=gold_working_dataframe, feature_columns=numeric_feature_columns, train_mask=train_mask_flag, norma`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "feature_set_id": feature_set_id, "numeric_feature_count": int(len(numeric_feature_columns)), "scaler_path": str(s`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print(f"Scaler saved to: {scaler_path}")`: Displays a notebook-facing result for inspection.
- `print(f"Scaled dataframe shape: {gold_preprocessed_scaled_dataframe.shape}")`: Displays a notebook-facing result for inspection.
- `gold_build_info = { "numeric_feature_count": int(len(numeric_feature_columns)), "feature_set_id": str(feature_set_id), "recommended_imputation": str(recommended_imputation), "scale`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="build_gold_model_ready_dataframe", message="Built Gold model-ready dataframe. Runtime build details were written to Truth Store.", data=gold_build_in`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `display`
- `else`
- `eq`
- `fit_and_apply_scaler`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `normal_only_mask = ( gold_working_dataframe["machine_status__profiled"].eq("normal_clean") if "machine_status__profiled" in gold_working_dataframe.columns else ( (gold_working_data` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_preprocessed_scaled_dataframe, scaler_path = fit_and_apply_scaler( dataframe=gold_working_dataframe, feature_columns=numeric_feature_columns, train_mask=train_mask_flag, norma` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "feature_set_id": feature_set_id, "numeric_feature_count": int(len(numeric_feature_columns)), "scaler_path": str(s` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"Scaler saved to: {scaler_path}")` | Displays a notebook-facing result for inspection. |
| `print(f"Scaled dataframe shape: {gold_preprocessed_scaled_dataframe.shape}")` | Displays a notebook-facing result for inspection. |
| `gold_build_info = { "numeric_feature_count": int(len(numeric_feature_columns)), "feature_set_id": str(feature_set_id), "recommended_imputation": str(recommended_imputation), "scale` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="build_gold_model_ready_dataframe", message="Built Gold model-ready dataframe. Runtime build details were written to Truth Store.", data=gold_build_in` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(gold_build_info)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 38 — Define the Normal-Only Fit Subset Logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_flag`
- `columns`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `elif`
- `loc`
- `machine_status__profiled`
- `normal_clean`
- `Series`
- `train_mask`

### Outputs

- `get_training_rows_for_unsupervised_model`
- `training_subset`

### Key Operations

- `def get_training_rows_for_unsupervised_model( dataframe: pd.DataFrame, *, train_mask: pd.Series,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: training_subset = dataframe.loc[train_mask].copy() if "machine_status__profiled" in training_subset.columns: training_subset = training_subset[ training_subset["`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `copy`
- `get_training_rows_for_unsupervised_model`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def get_training_rows_for_unsupervised_model( dataframe: pd.DataFrame, *, train_mask: pd.Series,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: training_subset = dataframe.loc[train_mask].copy() if "machine_status__profiled" in training_subset.columns: training_subset = training_subset[ training_subset["` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 39 — Build the Normal-Only Fit Subset

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `anomaly_flag`
- `columns`
- `else`
- `Filtered`
- `fit`
- `generate`
- `get_training_rows_for_unsupervised_model`
- `Gold`
- `gold_normal_only_split_shape`
- `gold_preprocessed_scaled_dataframe`
- `gold_preprocessed_scaled_shape`
- `ledger`
- `machine_status__profiled`
- `Normal`
- `normal`
- `normal_clean`
- `normal_source`
- `Only`
- `only`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `step`
- `train_mask`
- `training_rows_for_fit`

### Key Operations

- `# Normal Only`: Documents the purpose or boundary of the surrounding notebook step.
- `training_rows_for_fit = get_training_rows_for_unsupervised_model( gold_preprocessed_scaled_dataframe, train_mask=train_mask_flag,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger.add( kind="step", step="get_training_rows_for_unsupervised_model", message="Filtered the training split to generate the normal-only Gold fit subset.", data={ "gold_preproces`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `add`
- `get_training_rows_for_unsupervised_model`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Normal Only` | Documents the purpose or boundary of the surrounding notebook step. |
| `training_rows_for_fit = get_training_rows_for_unsupervised_model( gold_preprocessed_scaled_dataframe, train_mask=train_mask_flag,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger.add( kind="step", step="get_training_rows_for_unsupervised_model", message="Filtered the training split to generate the normal-only Gold fit subset.", data={ "gold_preproces` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 40 — Define the Reference Profile Logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `any`
- `append`
- `astype`
- `bool`
- `Build`
- `by`
- `cascade`
- `coerce`
- `column_name`
- `columns`
- `copy`
- `dataframe`
- `DataFrame`
- `def`
- `else`
- `errors`
- `feature_name`
- `Gold`
- `index`
- `isna`

### Outputs

- `build_reference_profile`
- `feature_columns`
- `feature_series`
- `profiled_dataframe`
- `subset_mask`

### Key Operations

- `def build_reference_profile( dataframe: pd.DataFrame, *, feature_columns: Sequence[str], subset_mask: Optional[pd.Series] = None,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """ Build the notebook-style reference profile used by the Gold cascade notebooks. Output columns -------------- - feature_name - median_value - mean_value - sta`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `any`
- `append`
- `astype`
- `build_reference_profile`
- `copy`
- `DataFrame`
- `isna`
- `mean`
- `median`
- `notna`
- `quantile`
- `reindex`
- `std`
- `to_numeric`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_reference_profile( dataframe: pd.DataFrame, *, feature_columns: Sequence[str], subset_mask: Optional[pd.Series] = None,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Build the notebook-style reference profile used by the Gold cascade notebooks. Output columns -------------- - feature_name - median_value - mean_value - sta` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 41 — Build the Normal Reference Profile

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `based`
- `bounds`
- `build_reference_profile`
- `Built`
- `capturing`
- `center`
- `columns`
- `comparison`
- `downstream`
- `feature`
- `ledger`
- `Need`
- `numeric_feature_columns`
- `percentile`
- `profile`
- `reference`
- `refernce_profile`
- `selected`
- `spread`

### Outputs

- `data`
- `feature_columns`
- `kind`
- `logger`
- `message`
- `reference_profile`
- `step`

### Key Operations

- `reference_profile = build_reference_profile( training_rows_for_fit, feature_columns=numeric_feature_columns,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `# TODO: Need Logger`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger.add( kind="step", step="build_reference_profile", message="Built reference profile table from selected feature columns, capturing center, spread, and percentile bounds for d`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `add`
- `build_reference_profile`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `reference_profile = build_reference_profile( training_rows_for_fit, feature_columns=numeric_feature_columns,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# TODO: Need Logger` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger.add( kind="step", step="build_reference_profile", message="Built reference profile table from selected feature columns, capturing center, spread, and percentile bounds for d` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 42 — Define the Stage 2 Feature Ranking Logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `abs`
- `Any`
- `any`
- `append`
- `astype`
- `be`
- `bool`
- `both`
- `chosen`
- `coefficient`
- `coerce`
- `column_name`
- `columns`
- `Compatibility`
- `continue`
- `copy`
- `DataFrame`
- `dataframe`
- `def`

### Outputs

- `ascending`
- `by`
- `choose_stage2_features_from_training_stability`
- `coefficient_of_variation`
- `feature_columns`
- `feature_series`
- `info`
- `median_value`
- `non_null_ratio`
- `ranking_dataframe`
- `series`
- `standard_deviation`
- `train_frame`
- `train_mask`
- `variance`

### Key Operations

- `def choose_stage2_features_from_training_stability( dataframe: pd.DataFrame, *, feature_columns: Sequence[str], train_mask: Optional[pd.Series] = None, target_count: Optional[int] `: Defines notebook-local logic used later in the notebook.
- `) -> Tuple[List[str], Dict[str, Any]] \| List[str]: """ Compatibility helper for both pipeline-style and notebook-style Stage 2 feature selection. Modes ----- - Notebook style: prov`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `abs`
- `any`
- `append`
- `astype`
- `choose_stage2_features_from_training_stability`
- `copy`
- `DataFrame`
- `head`
- `isna`
- `max`
- `mean`
- `median`
- `notna`
- `reindex`
- `reset_index`
- `Returns`
- `sort_values`
- `std`
- `to_numeric`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def choose_stage2_features_from_training_stability( dataframe: pd.DataFrame, *, feature_columns: Sequence[str], train_mask: Optional[pd.Series] = None, target_count: Optional[int] ` | Defines notebook-local logic used later in the notebook. |
| `) -> Tuple[List[str], Dict[str, Any]] \| List[str]: """ Compatibility helper for both pipeline-style and notebook-style Stage 2 feature selection. Modes ----- - Notebook style: prov` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 43 — Choose the Stage 2 Feature Set

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `by`
- `choose_stage2_features_from_training_stability`
- `columns`
- `dataframe_original_column_count`
- `dataframe_original_columns`
- `features`
- `ledger`
- `lowest`
- `modeling`
- `Need`
- `numeric_feature_columns`
- `Ranked`
- `relative`
- `selected`
- `stability`
- `Stage`
- `stage1_feature_column_count`
- `stage2_feature_column_count`
- `STAGE2_TARGET_FEATURE_COUNT`

### Outputs

- `data`
- `feature_columns`
- `kind`
- `logger`
- `message`
- `stage1_feature_columns`
- `stage2_feature_columns`
- `step`
- `target_count`

### Key Operations

- `stage1_feature_columns = list(numeric_feature_columns)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stage2_feature_columns = choose_stage2_features_from_training_stability( training_rows_for_fit, feature_columns=stage1_feature_columns, target_count=STAGE2_TARGET_FEATURE_COUNT,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `# TODO: Need Logger`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger.add( kind="step", step="choose_stage2_features_from_training_stability", message="Ranked training features by stability and selected the top features with the lowest relativ`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `add`
- `choose_stage2_features_from_training_stability`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage1_feature_columns = list(numeric_feature_columns)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage2_feature_columns = choose_stage2_features_from_training_stability( training_rows_for_fit, feature_columns=stage1_feature_columns, target_count=STAGE2_TARGET_FEATURE_COUNT,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# TODO: Need Logger` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger.add( kind="step", step="choose_stage2_features_from_training_stability", message="Ranked training features by stability and selected the top features with the lowest relativ` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 44 — Define the Stage 3 Sensor Grouping Logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `assign`
- `copy`
- `DataFrame`
- `def`
- `deviation`
- `drop`
- `empty`
- `feature_name`
- `features`
- `head`
- `isin`
- `lowest`
- `most`
- `Notebook`
- `primary`
- `primary_count`
- `Rank`
- `reference_profile`
- `reset_index`
- `rule`

### Outputs

- `ascending`
- `build_stage3_sensor_groups`
- `by`
- `primary_rule_sensors`
- `ranked_reference`
- `remaining_features`
- `secondary_rule_sensors`

### Key Operations

- `def build_stage3_sensor_groups( reference_profile: pd.DataFrame, *, stage2_feature_columns: Sequence[str], primary_count: int, secondary_count: int,`: Defines notebook-local logic used later in the notebook.
- `) -> Tuple[List[str], List[str]]: """ Notebook-style Stage 3 sensor selection. Rank Stage 2 features by lowest standard deviation, then assign the most stable features to the prima`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `build_stage3_sensor_groups`
- `copy`
- `head`
- `isin`
- `reset_index`
- `sort_values`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_stage3_sensor_groups( reference_profile: pd.DataFrame, *, stage2_feature_columns: Sequence[str], primary_count: int, secondary_count: int,` | Defines notebook-local logic used later in the notebook. |
| `) -> Tuple[List[str], List[str]]: """ Notebook-style Stage 3 sensor selection. Rank Stage 2 features by lowest standard deviation, then assign the most stable features to the prima` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 45 — Build the Stage 3 Sensor Groups

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `artifacts`
- `astype`
- `baseline`
- `baseline_feature_columns`
- `bool`
- `Build`
- `build_reference_profile`
- `build_stage3_sensor_groups`
- `cascade`
- `choose_stage2_features_from_training_stability`
- `copy`
- `DataFrame`
- `def`
- `downstream`
- `else`
- `Gold`
- `index`
- `loc`
- `newer`

### Outputs

- `build_gold_support_artifacts`
- `feature_columns`
- `min_non_null_ratio`
- `min_variance`
- `primary_count`
- `reference_profile`
- `secondary_count`
- `stage1_feature_columns`
- `stage2_feature_columns`
- `stage2_info`
- `subset_mask`
- `target_count`
- `train_mask`
- `training_rows_for_fit`

### Key Operations

- `def build_gold_support_artifacts( scaled_dataframe: pd.DataFrame, *, selected_feature_columns: Sequence[str], train_mask: pd.Series, baseline_feature_columns: Optional[Sequence[str`: Defines notebook-local logic used later in the notebook.
- `) -> Dict[str, Any]: """ Build downstream support artifacts for baseline and cascade using the newer Gold notebook structure. """ train_mask = train_mask.astype(bool).reindex(scale`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `build_gold_support_artifacts`
- `build_reference_profile`
- `build_stage3_sensor_groups`
- `choose_stage2_features_from_training_stability`
- `copy`
- `reindex`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_gold_support_artifacts( scaled_dataframe: pd.DataFrame, *, selected_feature_columns: Sequence[str], train_mask: pd.Series, baseline_feature_columns: Optional[Sequence[str` | Defines notebook-local logic used later in the notebook. |
| `) -> Dict[str, Any]: """ Build downstream support artifacts for baseline and cascade using the newer Gold notebook structure. """ train_mask = train_mask.astype(bool).reindex(scale` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 46 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `assigning`
- `build_stage3_sensor_groups`
- `Built`
- `by`
- `features`
- `groups`
- `ledger`
- `most`
- `Need`
- `on`
- `primary`
- `profile`
- `ranking`
- `reference`
- `reference_profile`
- `refernce_profile`
- `rule`
- `secondary`
- `sensor`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `primary_count`
- `secondary_count`
- `stage2_feature_columns`
- `step`

### Key Operations

- `stage3_primary_rule_sensors, stage3_secondary_rule_sensors = build_stage3_sensor_groups( reference_profile, stage2_feature_columns=STAGE2_FEATURE_COLUMNS, primary_count=STAGE3_PRIM`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `# TODO: Need Logger`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger.add( kind="step", step="build_stage3_sensor_groups", message="Built Stage 3 sensor groups by ranking Stage 2 features on reference-profile stability and assigning the most s`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `add`
- `build_stage3_sensor_groups`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage3_primary_rule_sensors, stage3_secondary_rule_sensors = build_stage3_sensor_groups( reference_profile, stage2_feature_columns=STAGE2_FEATURE_COLUMNS, primary_count=STAGE3_PRIM` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# TODO: Need Logger` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger.add( kind="step", step="build_stage3_sensor_groups", message="Built Stage 3 sensor groups by ranking Stage 2 features on reference-profile stability and assigning the most s` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 47 — Save the Stage-Level Gold Artifacts

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `artifact_paths`
- `build_stage_feature_sets`
- `Built`
- `feature`
- `index`
- `ledger`
- `reference_profile`
- `REFERENCE_PROFILE_PATH`
- `reference_profile_path`
- `results`
- `runtime_facts`
- `save`
- `save_json`
- `sets`
- `Stage`
- `stage_feature_summary`
- `stage1_feature_columns`
- `stage1_feature_count`
- `stage1_features_path`

### Outputs

- `data`
- `feature_set_summary`
- `gold_truth`
- `kind`
- `logger`
- `message`
- `step`

### Key Operations

- `save_json(stage1_feature_columns, STAGE1_FEATURES_PATH)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `save_json(stage2_feature_columns, STAGE2_FEATURES_PATH)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `save_json(stage3_primary_rule_sensors, STAGE3_PRIMARY_PATH)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `save_json(stage3_secondary_rule_sensors, STAGE3_SECONDARY_PATH)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `reference_profile.to_csv(REFERENCE_PROFILE_PATH, index=False)`: Writes an artifact or output used for review or downstream notebooks.
- `wandb.save(str(STAGE1_FEATURES_PATH))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `wandb.save(str(STAGE2_FEATURES_PATH))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `wandb.save(str(STAGE3_PRIMARY_PATH))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `wandb.save(str(STAGE3_SECONDARY_PATH))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `wandb.save(str(REFERENCE_PROFILE_PATH))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `feature_set_summary = { "stage1_feature_count": int(len(stage1_feature_columns)), "stage2_feature_count": int(len(stage2_feature_columns)), "stage3_primary_rule_count": int(len(sta`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `save`
- `save_json`
- `to_csv`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `save_json(stage1_feature_columns, STAGE1_FEATURES_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `save_json(stage2_feature_columns, STAGE2_FEATURES_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `save_json(stage3_primary_rule_sensors, STAGE3_PRIMARY_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `save_json(stage3_secondary_rule_sensors, STAGE3_SECONDARY_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `reference_profile.to_csv(REFERENCE_PROFILE_PATH, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `wandb.save(str(STAGE1_FEATURES_PATH))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(STAGE2_FEATURES_PATH))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(STAGE3_PRIMARY_PATH))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(STAGE3_SECONDARY_PATH))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(REFERENCE_PROFILE_PATH))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_set_summary = { "stage1_feature_count": int(len(stage1_feature_columns)), "stage2_feature_count": int(len(stage2_feature_columns)), "stage3_primary_rule_count": int(len(sta` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "stage_feature_summary": feature_set_summary, },` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth = update_truth_section( gold_truth, "artifact_paths", { "reference_profile_path": str(REFERENCE_PROFILE_PATH), "stage1_features_path": str(STAGE1_FEATURES_PATH), "stage2` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="build_stage_feature_sets", message="Built feature sets for Stage 1, Stage 2, and Stage 3 and wrote the results to Truth Store.", data=feature_set_sum` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `feature_set_summary` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output, optional experiment tracking call.

## Code Cell 48 — Create the Final Gold Split Dataframes

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `astype`
- `bool`
- `build_final_gold_split_dataframes`
- `columns`
- `copy`
- `correctly`
- `Created`
- `DataFrame`
- `dataframes`
- `else`
- `final`
- `fit`
- `fit_normal_only`
- `fit_rows_normal_only`
- `Gold`
- `gold_preprocessed_scaled_dataframe`
- `ledger`
- `loc`
- `meta__episode_id`

### Outputs

- `data`
- `gold_fit_dataframe`
- `gold_split_summary`
- `gold_test_dataframe`
- `gold_train_dataframe`
- `kind`
- `logger`
- `message`
- `step`
- `train_mask_flag`

### Key Operations

- `if "meta__is_train_flag" in gold_preprocessed_scaled_dataframe.columns: gold_preprocessed_scaled_dataframe["meta__split"] = np.where( gold_preprocessed_scaled_dataframe["meta__is_t`: Controls validation, iteration, file handling, or error handling for this step.
- `else: raise ValueError( "meta__is_train_flag is missing from gold_preprocessed_scaled_dataframe. " "The train/test split was not stamped correctly." )`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `train_mask_flag = gold_preprocessed_scaled_dataframe["meta__is_train_flag"].astype(bool)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `gold_train_dataframe = gold_preprocessed_scaled_dataframe.loc[train_mask_flag].copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `gold_test_dataframe = gold_preprocessed_scaled_dataframe.loc[~train_mask_flag].copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `gold_fit_dataframe = training_rows_for_fit.copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `gold_train_dataframe["meta__split"] = "train"`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_test_dataframe["meta__split"] = "test"`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_fit_dataframe["meta__split"] = "fit_normal_only"`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_split_summary = { "train_rows": int(len(gold_train_dataframe)), "test_rows": int(len(gold_test_dataframe)), "fit_rows_normal_only": int(len(gold_fit_dataframe)), "train_episod`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="build_final_gold_split_dataframes", message="Created final Gold train, test, and fit-normal-only dataframes.", data=gold_split_summary, logger=logger`: Records or exports ledger information for stage-level traceability.

Important functions or methods detected:
- `add`
- `astype`
- `copy`
- `DataFrame`
- `display`
- `nunique`
- `ValueError`
- `where`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `if "meta__is_train_flag" in gold_preprocessed_scaled_dataframe.columns: gold_preprocessed_scaled_dataframe["meta__split"] = np.where( gold_preprocessed_scaled_dataframe["meta__is_t` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: raise ValueError( "meta__is_train_flag is missing from gold_preprocessed_scaled_dataframe. " "The train/test split was not stamped correctly." )` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `train_mask_flag = gold_preprocessed_scaled_dataframe["meta__is_train_flag"].astype(bool)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_train_dataframe = gold_preprocessed_scaled_dataframe.loc[train_mask_flag].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_test_dataframe = gold_preprocessed_scaled_dataframe.loc[~train_mask_flag].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_fit_dataframe = training_rows_for_fit.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_train_dataframe["meta__split"] = "train"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_test_dataframe["meta__split"] = "test"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_fit_dataframe["meta__split"] = "fit_normal_only"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_split_summary = { "train_rows": int(len(gold_train_dataframe)), "test_rows": int(len(gold_test_dataframe)), "fit_rows_normal_only": int(len(gold_fit_dataframe)), "train_episod` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="build_final_gold_split_dataframes", message="Created final Gold train, test, and fit-normal-only dataframes.", data=gold_split_summary, logger=logger` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(pd.DataFrame([gold_split_summary]))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 49 — Finalize the Gold Truth Record

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `agg`
- `anomaly`
- `anomaly_flag`
- `appear`
- `are`
- `assign`
- `astype`
- `Bad`
- `be`
- `between`
- `bool`
- `both`
- `column`
- `column_name`
- `columns`
- `contains`
- `count`
- `Count`
- `DataFrame`
- `dataframe`

### Outputs

- `anomaly_rows`
- `bad_fit_rows`
- `episode_column`
- `fit_dataframe`
- `fit_row_ids`
- `fit_rows_outside_train`
- `full_dataframe`
- `gold_episode_split_summary`
- `overlapping_episodes`
- `overlapping_train_test_rows`
- `required_columns`
- `row_count`
- `row_id_column`
- `split_episode_summary`
- `split_label`
- `test_anomaly_rows`
- `test_dataframe`
- `test_episode_set`
- `test_row_ids`
- `train_dataframe`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Verify episode split integrity`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `def verify_gold_episode_split( full_dataframe: pd.DataFrame, train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame, fit_dataframe: pd.DataFrame, *, episode_column: str = "met`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: required_columns = [episode_column, row_id_column, "meta__is_train_flag"] for column_name in required_columns: if column_name not in full_dataframe.columns: rais`: Writes a logger message for traceability during notebook execution.
- `gold_episode_split_summary = verify_gold_episode_split( full_dataframe=gold_preprocessed_scaled_dataframe, train_dataframe=gold_train_dataframe, test_dataframe=gold_test_dataframe,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(gold_episode_split_summary)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `agg`
- `assign`
- `astype`
- `display`
- `dropna`
- `else`
- `fillna`
- `groupby`
- `reset_index`
- `sorted`
- `sum`
- `ValueError`
- `verify_gold_episode_split`
- `warning`
- `where`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Verify episode split integrity` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `def verify_gold_episode_split( full_dataframe: pd.DataFrame, train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame, fit_dataframe: pd.DataFrame, *, episode_column: str = "met` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: required_columns = [episode_column, row_id_column, "meta__is_train_flag"] for column_name in required_columns: if column_name not in full_dataframe.columns: rais` | Writes a logger message for traceability during notebook execution. |
| `gold_episode_split_summary = verify_gold_episode_split( full_dataframe=gold_preprocessed_scaled_dataframe, train_dataframe=gold_train_dataframe, test_dataframe=gold_test_dataframe,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(gold_episode_split_summary)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 50 — Update Gold preprocessing truth metadata

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `artifact_paths`
- `astype`
- `build_file_fingerprint`
- `columns`
- `dropna`
- `else`
- `FEATURE_REGISTRY_PATH`
- `feature_registry_path`
- `Gold`
- `GOLD_FIT_DATA_PATH`
- `gold_fit_path`
- `gold_preprocessed_path`
- `gold_preprocessed_scaled_dataframe`
- `GOLD_PRESCALED_DATA_PATH`
- `gold_prescaled_path`
- `GOLD_SCALED_DATA_PATH`
- `gold_scaled_path`
- `gold_split_summary`
- `GOLD_TEST_DATA_PATH`
- `gold_test_path`

### Outputs

- `gold_truth`

### Key Operations

- `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "source_run_ids": ( gold_preprocessed_scaled_dataframe["meta__run_id"].dropna().astype(str).unique().tolist() if "`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_truth = update_truth_section( gold_truth, "source_fingerprint", build_file_fingerprint(silver_path),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_truth = update_truth_section( gold_truth, "artifact_paths", { "silver_source_path": str(silver_path), "feature_registry_path": str(FEATURE_REGISTRY_PATH), "imputation_recommen`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_truth = update_truth_section( gold_truth, "notes", { "purpose": "Gold preprocessing truth record", },`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(gold_truth["runtime_facts"])`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `astype`
- `build_file_fingerprint`
- `display`
- `dropna`
- `tolist`
- `unique`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `gold_truth = update_truth_section( gold_truth, "runtime_facts", { "source_run_ids": ( gold_preprocessed_scaled_dataframe["meta__run_id"].dropna().astype(str).unique().tolist() if "` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth = update_truth_section( gold_truth, "source_fingerprint", build_file_fingerprint(silver_path),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth = update_truth_section( gold_truth, "artifact_paths", { "silver_source_path": str(silver_path), "feature_registry_path": str(FEATURE_REGISTRY_PATH), "imputation_recommen` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth = update_truth_section( gold_truth, "notes", { "purpose": "Gold preprocessing truth record", },` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(gold_truth["runtime_facts"])` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 51 — Save the Final Gold Preprocessing Outputs

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `across`
- `add`
- `artifacts`
- `columns`
- `f`
- `Gold`
- `gold_fit_dataframe`
- `gold_fit_has_row_id`
- `gold_preprocessed_scaled_dataframe`
- `gold_scaled_has_row_id`
- `gold_test_dataframe`
- `gold_test_has_row_id`
- `gold_train_dataframe`
- `gold_train_has_row_id`
- `identity`
- `ledger`
- `meta__row_id`
- `missing`
- `raise`
- `required_col`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `required_row_tracking_columns`
- `step`

### Key Operations

- `required_row_tracking_columns = ["meta__row_id"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `for required_col in required_row_tracking_columns: if required_col not in gold_preprocessed_scaled_dataframe.columns: raise ValueError(f"{required_col} missing from gold_preprocess`: Controls validation, iteration, file handling, or error handling for this step.
- `ledger.add( kind="step", step="validate_row_tracking_columns", message="Validated stable row identity across saved Gold artifacts.", data={ "required_columns": required_row_trackin`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `required_row_tracking_columns = ["meta__row_id"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for required_col in required_row_tracking_columns: if required_col not in gold_preprocessed_scaled_dataframe.columns: raise ValueError(f"{required_col} missing from gold_preprocess` | Controls validation, iteration, file handling, or error handling for this step. |
| `ledger.add( kind="step", step="validate_row_tracking_columns", message="Validated stable row identity across saved Gold artifacts.", data={ "required_columns": required_row_trackin` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 52 — Save the Final Gold Preprocessing Outputs

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append_truth_index`
- `build_truth_record`
- `Gold`
- `GOLD_PARENT_TRUTH_HASH`
- `gold_truth`
- `hash`
- `identify_feature_columns`
- `identify_meta_columns`
- `meta__parent_truth_hash`
- `meta__pipeline_mode`
- `meta__truth_hash`
- `save_truth_record`
- `shape`
- `sorted`
- `stamp_truth_columns`
- `truth`
- `truth_hash`
- `truth_index_path`
- `TRUTH_INDEX_PATH`
- `TRUTHS_PATH`

### Outputs

- `column_count`
- `dataset_name`
- `feature_columns`
- `gold_feature_columns`
- `gold_fit_dataframe`
- `gold_meta_columns`
- `gold_preprocessed_prescaled_dataframe`
- `gold_preprocessed_scaled_dataframe`
- `gold_test_dataframe`
- `gold_train_dataframe`
- `GOLD_TRUTH_HASH`
- `gold_truth_path`
- `gold_truth_record`
- `layer_name`
- `meta_columns`
- `parent_truth_hash`
- `pipeline_mode`
- `row_count`
- `truth_base`
- `truth_dir`

### Key Operations

- `gold_meta_columns = identify_meta_columns(gold_preprocessed_scaled_dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `gold_meta_columns = sorted(set(gold_meta_columns + [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_feature_columns = identify_feature_columns(gold_preprocessed_scaled_dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `gold_truth_record = build_truth_record( truth_base=gold_truth, row_count=len(gold_preprocessed_scaled_dataframe), column_count=gold_preprocessed_scaled_dataframe.shape[1] + 3, meta`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `GOLD_TRUTH_HASH = gold_truth_record["truth_hash"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `gold_preprocessed_prescaled_dataframe = stamp_truth_columns( gold_preprocessed_prescaled_dataframe, truth_hash=GOLD_TRUTH_HASH, parent_truth_hash=GOLD_PARENT_TRUTH_HASH, pipeline_m`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_preprocessed_scaled_dataframe = stamp_truth_columns( gold_preprocessed_scaled_dataframe, truth_hash=GOLD_TRUTH_HASH, parent_truth_hash=GOLD_PARENT_TRUTH_HASH, pipeline_mode=PI`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_train_dataframe = stamp_truth_columns( gold_train_dataframe, truth_hash=GOLD_TRUTH_HASH, parent_truth_hash=GOLD_PARENT_TRUTH_HASH, pipeline_mode=PIPELINE_MODE,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `append_truth_index`
- `build_truth_record`
- `identify_feature_columns`
- `identify_meta_columns`
- `save_truth_record`
- `sorted`
- `stamp_truth_columns`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `gold_meta_columns = identify_meta_columns(gold_preprocessed_scaled_dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_meta_columns = sorted(set(gold_meta_columns + [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_feature_columns = identify_feature_columns(gold_preprocessed_scaled_dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_truth_record = build_truth_record( truth_base=gold_truth, row_count=len(gold_preprocessed_scaled_dataframe), column_count=gold_preprocessed_scaled_dataframe.shape[1] + 3, meta` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_TRUTH_HASH = gold_truth_record["truth_hash"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_preprocessed_prescaled_dataframe = stamp_truth_columns( gold_preprocessed_prescaled_dataframe, truth_hash=GOLD_TRUTH_HASH, parent_truth_hash=GOLD_PARENT_TRUTH_HASH, pipeline_m` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_preprocessed_scaled_dataframe = stamp_truth_columns( gold_preprocessed_scaled_dataframe, truth_hash=GOLD_TRUTH_HASH, parent_truth_hash=GOLD_PARENT_TRUTH_HASH, pipeline_mode=PI` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_train_dataframe = stamp_truth_columns( gold_train_dataframe, truth_hash=GOLD_TRUTH_HASH, parent_truth_hash=GOLD_PARENT_TRUTH_HASH, pipeline_mode=PIPELINE_MODE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_test_dataframe = stamp_truth_columns( gold_test_dataframe, truth_hash=GOLD_TRUTH_HASH, parent_truth_hash=GOLD_PARENT_TRUTH_HASH, pipeline_mode=PIPELINE_MODE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_fit_dataframe = stamp_truth_columns( gold_fit_dataframe, truth_hash=GOLD_TRUTH_HASH, parent_truth_hash=GOLD_PARENT_TRUTH_HASH, pipeline_mode=PIPELINE_MODE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth_path = save_truth_record( gold_truth_record, truth_dir=TRUTHS_PATH, dataset_name=DATASET_NAME, layer_name=LAYER_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `append_truth_index( gold_truth_record, truth_index_path=TRUTH_INDEX_PATH,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Gold truth hash:", GOLD_TRUTH_HASH)` | Displays a notebook-facing result for inspection. |
| `print("Gold truth path:", gold_truth_path)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: truth record.

## Code Cell 53 — Save pre-scaled Gold feature outputs

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal_rows`
- `add`
- `anomaly_flag`
- `columns`
- `DataFrame`
- `else`
- `finalize_gold_truth_and_save_artifacts`
- `Finalized`
- `fit`
- `fit_normal_only`
- `Gold`
- `GOLD_FIT_DATA_PATH`
- `gold_fit_dataframe`
- `GOLD_PARENT_TRUTH_HASH`
- `gold_parent_truth_hash`
- `gold_preprocessed_prescaled_dataframe`
- `gold_preprocessed_scaled_dataframe`
- `GOLD_PRESCALED_DATA_PATH`
- `gold_prescaled_path`
- `GOLD_PROCESS_RUN_ID`

### Outputs

- `data`
- `gold_fit_path`
- `gold_preprocessed_prescaled_path`
- `gold_preprocessed_scaled_path`
- `gold_test_path`
- `gold_train_path`
- `kind`
- `logger`
- `message`
- `step`

### Key Operations

- `gold_preprocessed_prescaled_path = save_data( gold_preprocessed_prescaled_dataframe, GOLD_PRESCALED_DATA_PATH.parent, GOLD_PRESCALED_DATA_PATH.name,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_preprocessed_scaled_path = save_data( gold_preprocessed_scaled_dataframe, GOLD_SCALED_DATA_PATH.parent, GOLD_SCALED_DATA_PATH.name,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_train_path = save_data( gold_train_dataframe, GOLD_TRAIN_DATA_PATH.parent, GOLD_TRAIN_DATA_PATH.name,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_test_path = save_data( gold_test_dataframe, GOLD_TEST_DATA_PATH.parent, GOLD_TEST_DATA_PATH.name,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_fit_path = save_data( gold_fit_dataframe, GOLD_FIT_DATA_PATH.parent, GOLD_FIT_DATA_PATH.name,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `wandb.save(str(gold_preprocessed_prescaled_path))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `wandb.save(str(gold_preprocessed_scaled_path))`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `DataFrame`
- `display`
- `save`
- `save_data`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `gold_preprocessed_prescaled_path = save_data( gold_preprocessed_prescaled_dataframe, GOLD_PRESCALED_DATA_PATH.parent, GOLD_PRESCALED_DATA_PATH.name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_preprocessed_scaled_path = save_data( gold_preprocessed_scaled_dataframe, GOLD_SCALED_DATA_PATH.parent, GOLD_SCALED_DATA_PATH.name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_train_path = save_data( gold_train_dataframe, GOLD_TRAIN_DATA_PATH.parent, GOLD_TRAIN_DATA_PATH.name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_test_path = save_data( gold_test_dataframe, GOLD_TEST_DATA_PATH.parent, GOLD_TEST_DATA_PATH.name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_fit_path = save_data( gold_fit_dataframe, GOLD_FIT_DATA_PATH.parent, GOLD_FIT_DATA_PATH.name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(gold_preprocessed_prescaled_path))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(gold_preprocessed_scaled_path))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(gold_train_path))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(gold_test_path))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(gold_fit_path))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="finalize_gold_truth_and_save_artifacts", message="Finalized Gold truth, stamped lineage columns, and saved prescaled/scaled/train/test/fit outputs.",` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display( pd.DataFrame( [ { "split": "train", "rows": int(len(gold_train_dataframe)), "abnormal_rows": int(gold_train_dataframe["anomaly_flag"].sum()) if "anomaly_flag" in gold_trai` | Displays a notebook-facing result for inspection. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: optional experiment tracking call.

## Code Cell 54 — Save the Preprocessing Summary and Metadata Records

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `anomaly_flag`
- `artifacts`
- `DATASET_NAME`
- `dataset_name`
- `else`
- `feature_count`
- `FEATURE_REGISTRY_PATH`
- `feature_set_source`
- `fit_rows_normal_only`
- `get`
- `Gold`
- `gold_fit_dataframe`
- `gold_fit_path`
- `gold_input_source`
- `GOLD_PARENT_TRUTH_HASH`
- `gold_parent_truth_hash`
- `gold_preprocessed_scaled_dataframe`
- `GOLD_PRESCALED_DATA_PATH`
- `gold_prescaled_output_path`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `preprocessing_metadata`
- `preprocessing_summary`
- `step`

### Key Operations

- `preprocessing_summary = { "gold_prescaled_path": str(GOLD_PRESCALED_DATA_PATH), "gold_scaled_path": str(GOLD_SCALED_DATA_PATH), "gold_scaled_shape": list(gold_preprocessed_scaled_d`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `save_json(preprocessing_summary, PREPROCESSING_SUMMARY_PATH)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `preprocessing_metadata = { "recipe_id": RECIPE_ID, "gold_version": GOLD_VERSION, "dataset_name": DATASET_NAME, "feature_set_source": str(FEATURE_REGISTRY_PATH), "imputation_recomme`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `save_json(preprocessing_metadata, PREPROCESSING_METADATA_PATH)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `wandb.save(str(PREPROCESSING_SUMMARY_PATH))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `wandb.save(str(PREPROCESSING_METADATA_PATH))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="save_preprocessing_outputs", message="Saved Gold preprocessing summary and metadata artifacts.", data={ "preprocessing_summary_path": str(PREPROCESSI`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `{ "preprocessing_summary_path": str(PREPROCESSING_SUMMARY_PATH), "preprocessing_metadata_path": str(PREPROCESSING_METADATA_PATH), "gold_truth_hash": GOLD_TRUTH_HASH, "gold_truth_pa`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `get`
- `save`
- `save_json`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `preprocessing_summary = { "gold_prescaled_path": str(GOLD_PRESCALED_DATA_PATH), "gold_scaled_path": str(GOLD_SCALED_DATA_PATH), "gold_scaled_shape": list(gold_preprocessed_scaled_d` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `save_json(preprocessing_summary, PREPROCESSING_SUMMARY_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `preprocessing_metadata = { "recipe_id": RECIPE_ID, "gold_version": GOLD_VERSION, "dataset_name": DATASET_NAME, "feature_set_source": str(FEATURE_REGISTRY_PATH), "imputation_recomme` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `save_json(preprocessing_metadata, PREPROCESSING_METADATA_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(PREPROCESSING_SUMMARY_PATH))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(PREPROCESSING_METADATA_PATH))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="save_preprocessing_outputs", message="Saved Gold preprocessing summary and metadata artifacts.", data={ "preprocessing_summary_path": str(PREPROCESSI` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `{ "preprocessing_summary_path": str(PREPROCESSING_SUMMARY_PATH), "preprocessing_metadata_path": str(PREPROCESSING_METADATA_PATH), "gold_truth_hash": GOLD_TRUTH_HASH, "gold_truth_pa` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: optional experiment tracking call.

## Code Cell 55 — Save the Ledger and Close the Tracking Run

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `finish`
- `GOLD_ARTIFACTS_PATH`
- `GOLD_PREPROCESSING_LEDGER_FILE_NAME`
- `gold_preprocesssing_ledger_path`
- `ledger`
- `save`
- `wandb`
- `wandb_run`
- `write_json`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `#gold_preprocesssing_ledger_path = GOLD_ARTIFACTS_PATH / GOLD_PREPROCESSING_LEDGER_FILE_NAME`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger.write_json(gold_preprocesssing_ledger_path)`: Records or exports ledger information for stage-level traceability.
- `wandb.save(str(gold_preprocesssing_ledger_path))`: Records or exports ledger information for stage-level traceability.
- `wandb_run.finish()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `finish`
- `save`
- `write_json`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `#gold_preprocesssing_ledger_path = GOLD_ARTIFACTS_PATH / GOLD_PREPROCESSING_LEDGER_FILE_NAME` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger.write_json(gold_preprocesssing_ledger_path)` | Records or exports ledger information for stage-level traceability. |
| `wandb.save(str(gold_preprocesssing_ledger_path))` | Records or exports ledger information for stage-level traceability. |
| `wandb_run.finish()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: optional experiment tracking call.

## Code Cell 56 — Run Final Gold Preprocessing Sanity Checks

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `all`
- `anomalous`
- `anomaly_flag`
- `assert`
- `astype`
- `available`
- `bool`
- `check`
- `Checks`
- `checks`
- `clean`
- `column`
- `columns`
- `confirmed`
- `contains`
- `count`
- `dataframe`
- `does`
- `elif`
- `else`

### Outputs

- `dataframe_prescaled`
- `dataframe_scaled`
- `fit_mask_expected`
- `train_mask_flag`

### Key Operations

- `# Gold Preprocessing Sanity Checks`: Documents the purpose or boundary of the surrounding notebook step.
- `print("Running Gold preprocessing sanity checks...\n")`: Displays a notebook-facing result for inspection.
- `assert "gold_preprocessed_scaled_dataframe" in locals(), "gold_preprocessed_scaled_dataframe is missing."`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `assert "gold_preprocessed_prescaled_dataframe" in locals(), "gold_preprocessed_prescaled_dataframe is missing."`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `assert "train_mask_flag" in locals(), "train_mask_flag is missing."`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `assert "numeric_feature_columns" in locals(), "numeric_feature_columns is missing."`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `assert "gold_train_dataframe" in locals(), "gold_train_dataframe is missing."`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `assert "gold_test_dataframe" in locals(), "gold_test_dataframe is missing."`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `assert "gold_fit_dataframe" in locals(), "gold_fit_dataframe is missing."`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `assert "gold_truth_record" in locals(), "gold_truth_record is missing."`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `assert "GOLD_TRUTH_HASH" in locals(), "GOLD_TRUTH_HASH is missing."`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `dataframe_scaled = gold_preprocessed_scaled_dataframe`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `all`
- `assert`
- `astype`
- `Fit`
- `locals`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Gold Preprocessing Sanity Checks` | Documents the purpose or boundary of the surrounding notebook step. |
| `print("Running Gold preprocessing sanity checks...\n")` | Displays a notebook-facing result for inspection. |
| `assert "gold_preprocessed_scaled_dataframe" in locals(), "gold_preprocessed_scaled_dataframe is missing."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `assert "gold_preprocessed_prescaled_dataframe" in locals(), "gold_preprocessed_prescaled_dataframe is missing."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `assert "train_mask_flag" in locals(), "train_mask_flag is missing."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `assert "numeric_feature_columns" in locals(), "numeric_feature_columns is missing."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `assert "gold_train_dataframe" in locals(), "gold_train_dataframe is missing."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `assert "gold_test_dataframe" in locals(), "gold_test_dataframe is missing."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `assert "gold_fit_dataframe" in locals(), "gold_fit_dataframe is missing."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `assert "gold_truth_record" in locals(), "gold_truth_record is missing."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `assert "GOLD_TRUTH_HASH" in locals(), "GOLD_TRUTH_HASH is missing."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `dataframe_scaled = gold_preprocessed_scaled_dataframe` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `dataframe_prescaled = gold_preprocessed_prescaled_dataframe` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `n_rows, n_cols = dataframe_scaled.shape` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"- Scaled Gold shape: {n_rows} rows x {n_cols} columns")` | Displays a notebook-facing result for inspection. |
| `train_mask_flag = train_mask_flag.astype(bool)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `assert len(train_mask_flag) == n_rows, "train_mask_flag length does not match Gold dataframe."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `assert len(gold_train_dataframe) == train_mask_flag.sum(), "Train split row count mismatch."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `assert len(gold_test_dataframe) == (~train_mask_flag).sum(), "Test split row count mismatch."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"- Train rows: {len(gold_train_dataframe)}, Test rows: {len(gold_test_dataframe)}")` | Displays a notebook-facing result for inspection. |
| `if "machine_status__profiled" in dataframe_scaled.columns: fit_mask_expected = train_mask_flag & (dataframe_scaled["machine_status__profiled"] == "normal_clean") assert len(gold_fi` | Displays a notebook-facing result for inspection. |
| `elif "anomaly_flag" in dataframe_scaled.columns: fit_mask_expected = train_mask_flag & (dataframe_scaled["anomaly_flag"] == 0) assert len(gold_fit_dataframe) == fit_mask_expected.s` | Displays a notebook-facing result for inspection. |
| `else: print("- Neither machine_status__profiled nor anomaly_flag available; skipping fit-subset state check.")` | Displays a notebook-facing result for inspection. |
| `for required_column in numeric_feature_columns: assert required_column in dataframe_scaled.columns, ( f"Missing numeric feature column in scaled dataframe: {required_column}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `assert dataframe_scaled.index.is_unique, "Scaled dataframe index is not unique."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `assert dataframe_prescaled.index.is_unique, "Prescaled dataframe index is not unique."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("\nGold preprocessing sanity checks passed.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 57 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `gold_preprocessed_prescaled_dataframe`
- `shape`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `g`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `g` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 58 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `gold_preprocessed_scaled_dataframe`
- `shape`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `g`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `g` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 59 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_flag`
- `columns`
- `counts`
- `dataframe`
- `df`
- `dropna`
- `fit`
- `full`
- `gold_fit_dataframe`
- `gold_preprocessed_scaled_dataframe`
- `gold_test_dataframe`
- `gold_train_dataframe`
- `machine_status__profiled`
- `nfit`
- `ntest`
- `profiled`
- `scaled`
- `states`
- `test`
- `train`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `print("anomaly_flag in full scaled df:", "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns)`: Displays a notebook-facing result for inspection.
- `print("anomaly_flag in train df:", "anomaly_flag" in gold_train_dataframe.columns)`: Displays a notebook-facing result for inspection.
- `print("anomaly_flag in test df:", "anomaly_flag" in gold_test_dataframe.columns)`: Displays a notebook-facing result for inspection.
- `print("anomaly_flag in fit df:", "anomaly_flag" in gold_fit_dataframe.columns)`: Displays a notebook-facing result for inspection.
- `print("\nfit dataframe profiled states:")`: Displays a notebook-facing result for inspection.
- `print(gold_fit_dataframe["machine_status__profiled"].value_counts(dropna=False))`: Displays a notebook-facing result for inspection.
- `print("\nfit dataframe anomaly_flag counts:")`: Displays a notebook-facing result for inspection.
- `print(gold_fit_dataframe["anomaly_flag"].value_counts(dropna=False))`: Displays a notebook-facing result for inspection.
- `print("\ntest dataframe anomaly_flag counts:")`: Displays a notebook-facing result for inspection.
- `print(gold_test_dataframe["anomaly_flag"].value_counts(dropna=False))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print("anomaly_flag in full scaled df:", "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns)` | Displays a notebook-facing result for inspection. |
| `print("anomaly_flag in train df:", "anomaly_flag" in gold_train_dataframe.columns)` | Displays a notebook-facing result for inspection. |
| `print("anomaly_flag in test df:", "anomaly_flag" in gold_test_dataframe.columns)` | Displays a notebook-facing result for inspection. |
| `print("anomaly_flag in fit df:", "anomaly_flag" in gold_fit_dataframe.columns)` | Displays a notebook-facing result for inspection. |
| `print("\nfit dataframe profiled states:")` | Displays a notebook-facing result for inspection. |
| `print(gold_fit_dataframe["machine_status__profiled"].value_counts(dropna=False))` | Displays a notebook-facing result for inspection. |
| `print("\nfit dataframe anomaly_flag counts:")` | Displays a notebook-facing result for inspection. |
| `print(gold_fit_dataframe["anomaly_flag"].value_counts(dropna=False))` | Displays a notebook-facing result for inspection. |
| `print("\ntest dataframe anomaly_flag counts:")` | Displays a notebook-facing result for inspection. |
| `print(gold_test_dataframe["anomaly_flag"].value_counts(dropna=False))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 60 — Compare the Prescaled and Scaled Column Structures

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `all`
- `All`
- `are`
- `both`
- `but`
- `column`
- `columns`
- `Columns`
- `dataframe1`
- `dataframe2`
- `dataframes`
- `difference`
- `each`
- `f`
- `Find`
- `Get`
- `gold_preprocessed_prescaled_dataframe`
- `gold_preprocessed_scaled_dataframe`
- `lists`
- `n`

### Outputs

- `columns_not_in_both`
- `prescaled_columns`
- `scaled_columns`
- `unique_to_prescaled`
- `unique_to_scaled`

### Key Operations

- `# Get column lists`: Documents the purpose or boundary of the surrounding notebook step.
- `prescaled_columns = gold_preprocessed_prescaled_dataframe.columns.tolist()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `scaled_columns = gold_preprocessed_scaled_dataframe.columns.tolist()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print(f"Prescaled Columns columns: {prescaled_columns}")`: Displays a notebook-facing result for inspection.
- `print(f"Scaled columns: {scaled_columns}\n")`: Displays a notebook-facing result for inspection.
- `# Find columns unique to each`: Documents the purpose or boundary of the surrounding notebook step.
- `unique_to_prescaled = set(prescaled_columns) - set(scaled_columns)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `unique_to_scaled = set(scaled_columns) - set(prescaled_columns)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print(f"Columns in dataframe1 but not in dataframe2: {unique_to_prescaled}")`: Displays a notebook-facing result for inspection.
- `print(f"Columns in dataframe2 but not in dataframe1: {unique_to_scaled}\n")`: Displays a notebook-facing result for inspection.
- `# Find all columns that are not in both (symmetric difference)`: Documents the purpose or boundary of the surrounding notebook step.
- `columns_not_in_both = set(prescaled_columns).symmetric_difference(set(scaled_columns))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `both`
- `symmetric_difference`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Get column lists` | Documents the purpose or boundary of the surrounding notebook step. |
| `prescaled_columns = gold_preprocessed_prescaled_dataframe.columns.tolist()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `scaled_columns = gold_preprocessed_scaled_dataframe.columns.tolist()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print(f"Prescaled Columns columns: {prescaled_columns}")` | Displays a notebook-facing result for inspection. |
| `print(f"Scaled columns: {scaled_columns}\n")` | Displays a notebook-facing result for inspection. |
| `# Find columns unique to each` | Documents the purpose or boundary of the surrounding notebook step. |
| `unique_to_prescaled = set(prescaled_columns) - set(scaled_columns)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `unique_to_scaled = set(scaled_columns) - set(prescaled_columns)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print(f"Columns in dataframe1 but not in dataframe2: {unique_to_prescaled}")` | Displays a notebook-facing result for inspection. |
| `print(f"Columns in dataframe2 but not in dataframe1: {unique_to_scaled}\n")` | Displays a notebook-facing result for inspection. |
| `# Find all columns that are not in both (symmetric difference)` | Documents the purpose or boundary of the surrounding notebook step. |
| `columns_not_in_both = set(prescaled_columns).symmetric_difference(set(scaled_columns))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print(f"All columns not present in both dataframes: {columns_not_in_both}")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 61 — Verify Final Lineage Columns Across All Gold Outputs

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `astype`
- `check`
- `column_name`
- `columns`
- `contain`
- `created`
- `dataframe`
- `dataframe_parent`
- `does`
- `dropna`
- `exists`
- `extract_truth_hash`
- `f`
- `file`
- `FileNotFoundError`
- `frame_name`
- `frame_value`
- `get`
- `Gold`

### Outputs

- `frame_parent_values`
- `frame_truth_hash`
- `gold_frames_to_check`
- `loaded_gold_parent_truth_hash`
- `loaded_gold_truth`
- `loaded_gold_truth_hash`
- `loaded_gold_truth_raw`
- `missing_gold_meta_columns`
- `required_gold_meta_columns`

### Key Operations

- `required_gold_meta_columns = [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold_frames_to_check = { "gold_preprocessed_prescaled": gold_preprocessed_prescaled_dataframe, "gold_preprocessed_scaled": gold_preprocessed_scaled_dataframe, "gold_fit": gold_fit_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `for frame_name, frame_value in gold_frames_to_check.items(): missing_gold_meta_columns = [ column_name for column_name in required_gold_meta_columns if column_name not in frame_val`: Controls validation, iteration, file handling, or error handling for this step.
- `if not Path(gold_truth_path).exists(): raise FileNotFoundError(f"Gold truth file was not created: {gold_truth_path}")`: Controls validation, iteration, file handling, or error handling for this step.
- `loaded_gold_truth_raw = load_json(gold_truth_path)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `loaded_gold_truth = require_dict( loaded_gold_truth_raw, "loaded_gold_truth",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `loaded_gold_truth_hash = str( loaded_gold_truth.get("truth_hash", "")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `).strip()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if loaded_gold_truth_hash != GOLD_TRUTH_HASH: raise ValueError( "Saved Gold truth file hash does not match GOLD_TRUTH_HASH:\n" f"file={loaded_gold_truth_hash}\n" f"record={GOLD_TRU`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `astype`
- `dropna`
- `exists`
- `extract_truth_hash`
- `FileNotFoundError`
- `get`
- `items`
- `load_json`
- `Path`
- `require_dict`
- `strip`
- `tolist`
- `unique`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `required_gold_meta_columns = [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_frames_to_check = { "gold_preprocessed_prescaled": gold_preprocessed_prescaled_dataframe, "gold_preprocessed_scaled": gold_preprocessed_scaled_dataframe, "gold_fit": gold_fit_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for frame_name, frame_value in gold_frames_to_check.items(): missing_gold_meta_columns = [ column_name for column_name in required_gold_meta_columns if column_name not in frame_val` | Controls validation, iteration, file handling, or error handling for this step. |
| `if not Path(gold_truth_path).exists(): raise FileNotFoundError(f"Gold truth file was not created: {gold_truth_path}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `loaded_gold_truth_raw = load_json(gold_truth_path)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `loaded_gold_truth = require_dict( loaded_gold_truth_raw, "loaded_gold_truth",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `loaded_gold_truth_hash = str( loaded_gold_truth.get("truth_hash", "")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).strip()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if loaded_gold_truth_hash != GOLD_TRUTH_HASH: raise ValueError( "Saved Gold truth file hash does not match GOLD_TRUTH_HASH:\n" f"file={loaded_gold_truth_hash}\n" f"record={GOLD_TRU` | Controls validation, iteration, file handling, or error handling for this step. |
| `loaded_gold_parent_truth_hash = str( loaded_gold_truth.get("parent_truth_hash", "")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).strip()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if loaded_gold_parent_truth_hash != GOLD_PARENT_TRUTH_HASH: raise ValueError( "Saved Gold truth file parent hash does not match GOLD_PARENT_TRUTH_HASH:\n" f"truth={loaded_gold_pare` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Gold PreProcessing lineage sanity check passed.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 62 — Gold Preprocessing SQL Write Cell

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `else`
- `get`
- `globals`
- `gold_preprocessed_v1`
- `Postgres`
- `skipped`
- `write`
- `write_gold_preprocessed_features_sql`

### Outputs

- `capstone_schema`
- `dataset_id`
- `dataset_name`
- `engine`
- `feature_set_id`
- `gold_preprocessing_sql_summary_dataframe`
- `notebook_globals`
- `run_id`
- `WRITE_TO_POSTGRES`

### Key Operations

- `WRITE_TO_POSTGRES = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if WRITE_TO_POSTGRES: gold_preprocessing_sql_summary_dataframe = write_gold_preprocessed_features_sql( engine=engine, capstone_schema=CAPSTONE_SCHEMA, dataset_id=DATASET_ID, run_id`: Displays a notebook-facing result for inspection.
- `else: print("Postgres write skipped.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `get`
- `globals`
- `write_gold_preprocessed_features_sql`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `WRITE_TO_POSTGRES = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if WRITE_TO_POSTGRES: gold_preprocessing_sql_summary_dataframe = write_gold_preprocessed_features_sql( engine=engine, capstone_schema=CAPSTONE_SCHEMA, dataset_id=DATASET_ID, run_id` | Displays a notebook-facing result for inspection. |
| `else: print("Postgres write skipped.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

