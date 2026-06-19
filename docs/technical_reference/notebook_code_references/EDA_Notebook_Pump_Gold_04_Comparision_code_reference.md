# Notebook Code Reference: EDA_Notebook_Pump_Gold_04_Comparision

Notebook path:

`notebooks/experiments/EDA_Notebook_Pump_Gold_04_Comparision.ipynb`

## Notebook Purpose

This notebook compares baseline and cascade model outputs and builds final model-selection evidence.

Notebook stage:

`Gold`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Gold Comparison Setup and Imports | Code Cell 01, Code Cell 02 |
| Load Configuration, Paths, and Comparison Runtime Settings | Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08 |
| Review intermediate output | Code Cell 09 |
| Start Logging for the Gold Comparison Stage | Code Cell 10, Code Cell 11 |
| Initialize Experiment Tracking | Code Cell 12 |
| Initialize the Comparison Ledger | Code Cell 13 |
| Load the Baseline and Cascade Artifacts and Validate Shared Gold Lineage | Code Cell 14 |
| Resolve and Validate Baseline Stage Truth | Code Cell 15 |
| Shared Parent Truth Validation | Code Cell 16 |
| Resolve comparison dataset identity | Code Cell 17 |
| Build the Main Comparison Table | Code Cell 18 |
| Build the Comparison Summary Metrics | Code Cell 19 |
| Display the Model Comparison Table | Code Cell 20 |
| Paired Statistical Comparison of Baseline and Final Cascade | Code Cell 21 |
| Build statistical comparison summary | Code Cell 22 |
| Visualize Alert Counts and Core Metrics | Code Cell 23 |
| Answer | Code Cell 24, Code Cell 27 |
| Build cascade funnel comparison frame | Code Cell 25, Code Cell 26 |
| Build the Comparison Truth Record and Save the Comparison Artifacts | Code Cell 28 |
| Finalize the Ledger and Close the Tracking Run | Code Cell 29 |
| Visualize the Cascade Filtering Funnel | Code Cell 30 |
| Run Final Lineage and Consistency Checks | Code Cell 31 |
| Gold Comparison SQL Write Cell | Code Cell 32, Code Cell 33 |

## Code Cell 01 — Gold Comparison Setup and Imports

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__future__`
- `annotations`
- `Any`
- `append_truth_index`
- `artifact_file_path`
- `artifacts`
- `average_precision_score`
- `build_artifact_dirs`
- `build_artifact_dirs_from_config`
- `build_file_fingerprint`
- `build_truth_config_block`
- `build_truth_record`
- `cast`
- `classification_report`
- `cluster`
- `columns`
- `config_loader`
- `configure_logging`
- `confusion_matrix`
- `contingency_tables`

### Outputs

- `HAS_STATSMODELS_MCNEMAR`

### Key Operations

- `from __future__ import annotations`: Imports a dependency or project helper used by later cells.
- `from dataclasses import dataclass, field`: Imports a dependency or project helper used by later cells.
- `from datetime import datetime, timezone`: Imports a dependency or project helper used by later cells.
- `from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Mapping, cast`: Imports a dependency or project helper used by later cells.
- `from pathlib import Path`: Imports a dependency or project helper used by later cells.
- `import yaml`: Imports a dependency or project helper used by later cells.
- `import re`: Imports a dependency or project helper used by later cells.
- `import os`: Imports a dependency or project helper used by later cells.
- `import logging`: Imports a dependency or project helper used by later cells.
- `import wandb`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `import numpy as np`: Imports a dependency or project helper used by later cells.

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
| `import re` | Imports a dependency or project helper used by later cells. |
| `import os` | Imports a dependency or project helper used by later cells. |
| `import logging` | Imports a dependency or project helper used by later cells. |
| `import wandb` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `import numpy as np` | Imports a dependency or project helper used by later cells. |
| `import math` | Imports a dependency or project helper used by later cells. |
| `import matplotlib.pyplot as plt` | Imports a dependency or project helper used by later cells. |
| `from matplotlib.figure import Figure` | Imports a dependency or project helper used by later cells. |
| `import seaborn as sns` | Imports a dependency or project helper used by later cells. |
| `import joblib` | Imports a dependency or project helper used by later cells. |
| `from math import sqrt` | Imports a dependency or project helper used by later cells. |
| `from scipy.stats import norm` | Imports a dependency or project helper used by later cells. |
| `try: from statsmodels.stats.contingency_tables import mcnemar HAS_STATSMODELS_MCNEMAR = True` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `except Exception: HAS_STATSMODELS_MCNEMAR = False` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from sklearn.model_selection import train_test_split, KFold` | Imports a dependency or project helper used by later cells. |
| `from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler` | Imports a dependency or project helper used by later cells. |
| `from sklearn.decomposition import PCA` | Imports a dependency or project helper used by later cells. |
| `from sklearn.cluster import KMeans` | Imports a dependency or project helper used by later cells. |
| `from sklearn.ensemble import RandomForestClassifier, IsolationForest` | Imports a dependency or project helper used by later cells. |
| `from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score` | Imports a dependency or project helper used by later cells. |
| `from sklearn.svm import OneClassSVM` | Imports a dependency or project helper used by later cells. |
| `from sklearn.neighbors import LocalOutlierFactor` | Imports a dependency or project helper used by later cells. |
| `import pyarrow.parquet as pq` | Imports a dependency or project helper used by later cells. |
| `import pyarrow as pa` | Imports a dependency or project helper used by later cells. |
| `from IPython.display import HTML` | Imports a dependency or project helper used by later cells. |
| `import hashlib` | Imports a dependency or project helper used by later cells. |
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
| `from utils.core.artifacts import ( build_artifact_dirs_from_config, build_artifact_dirs, artifact_file_path,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.notebook_context import load_notebook_context` | Imports a dependency or project helper used by later cells. |
| `# Show more columns` | Documents the purpose or boundary of the surrounding notebook step. |
| `pd.set_option("display.max_columns", 100)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `1 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: SQL or medallion table write, truth record.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Gold Comparison Setup and Imports

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__name__`
- `a`
- `Any`
- `any`
- `be`
- `cast`
- `config`
- `Convert`
- `def`
- `dictionary`
- `extract`
- `f`
- `get`
- `Got`
- `got`
- `into`
- `isinstance`
- `JSON`
- `key`
- `loaded`

### Outputs

- `cfg_optional_mapping`
- `cfg_require_mapping`
- `get_nested_mapping`
- `normalize_text_value`
- `raw_value`
- `require_mapping`
- `require_truth_record`

### Key Operations

- `# ----`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# ----`: Documents the purpose or boundary of the surrounding notebook step.
- `def cfg_require_mapping(value: object, name: str) -> Mapping[str, Any]: if not isinstance(value, Mapping): raise TypeError( f"{name} must be a mapping, got {type(value).__name__}: `: Defines notebook-local logic used later in the notebook.
- `# ----`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# ----`: Documents the purpose or boundary of the surrounding notebook step.
- `def cfg_optional_mapping(value: object \| None, name: str) -> Mapping[str, Any]: if value is None: return {} return cfg_require_mapping(value, name)`: Defines notebook-local logic used later in the notebook.
- `# --------------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# --------------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `def require_truth_record(value: Any, name: str) -> dict[str, Any]: """ Validate a loaded truth record. """ if not isinstance(value, dict): raise TypeError( f"{name} must be a dicti`: Defines notebook-local logic used later in the notebook.

Important functions or methods detected:
- `cast`
- `cfg_optional_mapping`
- `cfg_require_mapping`
- `get`
- `get_nested_mapping`
- `isinstance`
- `lower`
- `normalize_text_value`
- `require_mapping`
- `require_truth_record`
- `strip`
- `type`
- `TypeError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `def cfg_require_mapping(value: object, name: str) -> Mapping[str, Any]: if not isinstance(value, Mapping): raise TypeError( f"{name} must be a mapping, got {type(value).__name__}: ` | Defines notebook-local logic used later in the notebook. |
| `# ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `def cfg_optional_mapping(value: object \| None, name: str) -> Mapping[str, Any]: if value is None: return {} return cfg_require_mapping(value, name)` | Defines notebook-local logic used later in the notebook. |
| `# --------------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# --------------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `def require_truth_record(value: Any, name: str) -> dict[str, Any]: """ Validate a loaded truth record. """ if not isinstance(value, dict): raise TypeError( f"{name} must be a dicti` | Defines notebook-local logic used later in the notebook. |
| `# --------------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# --------------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `def get_nested_mapping( value: dict[str, Any], key: str, name: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, Any]: """ Safely extract a nested dictionary from a truth/config record. """ raw_value = value.get(key, {}) if raw_value is None: return {} if not isinstance(raw_val` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# --------------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# --------------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `def require_mapping(value: Any, name: str) -> dict[str, Any]: """ Validate that a loaded JSON/config object is a dictionary. """ if not isinstance(value, dict): raise TypeError( f"` | Defines notebook-local logic used later in the notebook. |
| `# --------------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# --------------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `def normalize_text_value(value: Any) -> str: """ Convert any scalar value into lowercase stripped text. """ if value is None: return "" return str(value).strip().lower()` | Defines notebook-local logic used later in the notebook. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 03 — Load Configuration, Paths, and Comparison Runtime Settings

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `aliases`
- `capstone`
- `comparison`
- `context`
- `context_loaded`
- `dataset_config`
- `default`
- `execution`
- `gold`
- `gold_comparison`
- `gold_model_comparison`
- `info`
- `load_notebook_context`
- `loaded`
- `Loaded`
- `log`
- `LOG_PATH`
- `log_path`
- `logger`

### Outputs

- `COMPARISON_CFG`
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
- `CONTEXT_STAGE = "gold_comparison"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_DATASET = "pump"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LAYER = "gold"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_RUN_MODE = "train"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_PROFILE = "default"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LOG_FILE = "gold_model_comparison.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.gold.comparison", log_filename=`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `CONTEXT_STAGE = "gold_comparison"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LAYER = "gold"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_RUN_MODE = "train"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LOG_FILE = "gold_model_comparison.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.gold.comparison", log_filename=` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Shared aliases used throughout the notebook` | Documents the purpose or boundary of the surrounding notebook step. |
| `paths = CTX.paths` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_MAP = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE_CFG = CTX.stage_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `COMPARISON_CFG = CTX.stage_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
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

## Code Cell 04 — Load Configuration, Paths, and Comparison Runtime Settings

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `aliases`
- `Any`
- `Artifacts`
- `B`
- `Backward`
- `Base`
- `Baseline`
- `baseline`
- `build_truth_config_block`
- `capstone`
- `cascade`
- `Cascade`
- `CASCADE_DEFAULTS_THRESHOLDS_PATH`
- `cascade_defaults_thresholds_path`
- `CASCADE_STAGE3_IMPROVED_`
- `CASCADE_STAGE3_IMPROVED_METADATA_FILE_NAME`
- `cascade_stage3_improved_metadata_file_name`
- `CASCADE_STAGE3_IMPROVED_METADATA_PATH`
- `cascade_stage3_improved_metadata_path`
- `cascade_stage3_improved_results_file_name_csv`

### Outputs

- `ARTIFACTS_ROOT`
- `BASELINE_METADATA_FILE_NAME`
- `BASELINE_METADATA_PATH`
- `BASELINE_RESULTS_FILE_NAME_CSV`
- `BASELINE_RESULTS_FILE_NAME_PICKLE`
- `BASELINE_RESULTS_PATH_CSV`
- `BASELINE_RESULTS_PATH_PICKLE`
- `BASELINE_SUMMARY_FILE_NAME`
- `BASELINE_SUMMARY_PATH`
- `BASELINE_THRESHOLDS_FILE_NAME`
- `BASELINE_THRESHOLDS_PATH`
- `CASCADE_DEFAULTS_METADATA_FILE_NAME`
- `CASCADE_DEFAULTS_METADATA_PATH`
- `CASCADE_DEFAULTS_RESULTS_FILE_NAME_CSV`
- `CASCADE_DEFAULTS_RESULTS_FILE_NAME_PICKLE`
- `CASCADE_DEFAULTS_RESULTS_PATH_CSV`
- `CASCADE_DEFAULTS_RESULTS_PATH_PICKLE`
- `CASCADE_DEFAULTS_SUMMARY_FILE_NAME`
- `CASCADE_DEFAULTS_SUMMARY_PATH`
- `CASCADE_DEFAULTS_THRESHOLDS_FILE_NAME`

### Key Operations

- `TRUTH_CONFIG = cast(Dict[str, Any], build_truth_config_block(CONFIG))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_CONFIG["pipeline"] = PIPELINE`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `TRUTH_CONFIG["stage_params"] = COMPARISON_CFG`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `# ---- Stage details ----`: Documents the purpose or boundary of the surrounding notebook step.
- `STAGE = "gold"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `LAYER_NAME = str(COMPARISON_CFG["layer_name"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_VERSION = str(VERSIONS_CFG["gold"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_VERSION = str(VERSIONS_CFG["truth"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RECIPE_ID = str(COMPARISON_CFG["recipe_id"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `PIPELINE_MODE = str(PIPELINE["execution_mode"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

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
| `TRUTH_CONFIG["stage_params"] = COMPARISON_CFG` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Stage details ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `STAGE = "gold"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LAYER_NAME = str(COMPARISON_CFG["layer_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_VERSION = str(VERSIONS_CFG["gold"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_VERSION = str(VERSIONS_CFG["truth"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RECIPE_ID = str(COMPARISON_CFG["recipe_id"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `PIPELINE_MODE = str(PIPELINE["execution_mode"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_MODE = str(RUNTIME_CFG.get("mode", CONFIG_RUN_MODE))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = str(RUNTIME_CFG.get("profile", CONFIG_PROFILE))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `DATASET_NAME_CONFIG = str(DATASET_CFG.get("name", "pump"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_NAME = DATASET_NAME_CONFIG.strip().lower()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `GOLD_PROCESS_RUN_ID = make_process_run_id( str(COMPARISON_CFG["process_run_id_prefix"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PROCESS_RUN_ID = GOLD_PROCESS_RUN_ID` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- W&B ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `WANDB_PROJECT = str(WANDB_CFG.get("project", "capstone"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WANDB_ENTITY = str(WANDB_CFG.get("entity", ""))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WANDB_RUN_NAME = f"{GOLD_VERSION}"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- File names: baseline ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `BASELINE_RESULTS_FILE_NAME_CSV = str( FILENAMES["baseline_results_file_name_csv"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BASELINE_RESULTS_FILE_NAME_PICKLE = str( FILENAMES["baseline_results_file_name_pickle"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BASELINE_SUMMARY_FILE_NAME = str(FILENAMES["baseline_summary_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BASELINE_THRESHOLDS_FILE_NAME = str(FILENAMES["baseline_thresholds_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BASELINE_METADATA_FILE_NAME = str(FILENAMES["baseline_metadata_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- File names: cascade defaults ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `CASCADE_DEFAULTS_RESULTS_FILE_NAME_CSV = str( FILENAMES["cascade_defaults_results_file_name_csv"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_DEFAULTS_RESULTS_FILE_NAME_PICKLE = str( FILENAMES["cascade_defaults_results_file_name_pickle"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_DEFAULTS_SUMMARY_FILE_NAME = str( FILENAMES["cascade_defaults_summary_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_DEFAULTS_THRESHOLDS_FILE_NAME = str( FILENAMES["cascade_defaults_thresholds_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_DEFAULTS_METADATA_FILE_NAME = str( FILENAMES["cascade_defaults_metadata_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- File names: cascade tuned ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `CASCADE_TUNED_RESULTS_FILE_NAME_CSV = str( FILENAMES["cascade_tuned_results_file_name_csv"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_TUNED_RESULTS_FILE_NAME_PICKLE = str( FILENAMES["cascade_tuned_results_file_name_pickle"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_TUNED_SUMMARY_FILE_NAME = str( FILENAMES["cascade_tuned_summary_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_TUNED_THRESHOLDS_FILE_NAME = str( FILENAMES["cascade_tuned_thresholds_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_TUNED_METADATA_FILE_NAME = str( FILENAMES["cascade_tuned_metadata_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `130 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 05 — Load Configuration, Paths, and Comparison Runtime Settings

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

## Code Cell 06 — Load Configuration, Paths, and Comparison Runtime Settings

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `check`
- `COMPARISON_CFG`
- `context`
- `f`
- `globals`
- `Gold`
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

- `gold_required_context_vars = [ "COMPARISON_CFG",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `gold_required_context_vars = [ "COMPARISON_CFG",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_gold_context_vars = [ name for name in gold_required_context_vars if name not in globals()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_gold_context_vars: raise NameError(f"Missing Gold context variables: {missing_gold_context_vars}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `logger.info("Gold context sanity check passed")` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 07 — Load Configuration, Paths, and Comparison Runtime Settings

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold__statistical_test_summary`
- `__gold_comparison__resolved_config`
- `aliases`
- `artifact`
- `Backward`
- `baseline`
- `baseline_metadata_file_name`
- `baseline_results_file_name_csv`
- `baseline_results_file_name_pickle`
- `baseline_summary_file_name`
- `baseline_thresholds_file_name`
- `build`
- `build_artifact_dirs`
- `by`
- `cascade_defaults`
- `cascade_defaults_metadata_file_name`
- `cascade_defaults_results_file_name_csv`
- `cascade_defaults_results_file_name_pickle`
- `cascade_defaults_summary_file_name`
- `cascade_defaults_thresholds_file_name`

### Outputs

- `artifacts_root`
- `ARTIFACTS_ROOT`
- `BASELINE_METADATA_PATH`
- `BASELINE_RESULTS_PATH_CSV`
- `BASELINE_RESULTS_PATH_PICKLE`
- `BASELINE_SUMMARY_PATH`
- `BASELINE_THRESHOLDS_PATH`
- `CASCADE_DEFAULTS_METADATA_PATH`
- `CASCADE_DEFAULTS_RESULTS_PATH_CSV`
- `CASCADE_DEFAULTS_RESULTS_PATH_PICKLE`
- `CASCADE_DEFAULTS_SUMMARY_PATH`
- `CASCADE_DEFAULTS_THRESHOLDS_PATH`
- `CASCADE_STAGE3_IMPROVED_METADATA_PATH`
- `CASCADE_STAGE3_IMPROVED_RESULTS_PATH_CSV`
- `CASCADE_STAGE3_IMPROVED_RESULTS_PATH_PICKLE`
- `CASCADE_STAGE3_IMPROVED_SUMMARY_PATH`
- `CASCADE_STAGE3_IMPROVED_THRESHOLDS_PATH`
- `CASCADE_STAGE3_METADATA_PATH`
- `CASCADE_STAGE3_RESULTS_PATH_CSV`
- `CASCADE_STAGE3_RESULTS_PATH_PICKLE`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Gold comparison artifact directories`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Gold 04 compares outputs from Gold 02 / 03A / 03B / 03C.`: Documents the purpose or boundary of the surrounding notebook step.
- `# Its CONFIG may not contain those notebooks' stage-specific config sections,`: Documents the purpose or boundary of the surrounding notebook step.
- `# so we build the known Gold artifact families directly from the shared artifact root.`: Documents the purpose or boundary of the surrounding notebook step.
- `ARTIFACTS_ROOT = Path(CONFIG["resolved_paths"]["artifacts_root"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_COMMON_SUBDIRS = [ "scores", "summaries", "thresholds", "metadata", "models", "plots", "config", "lineage",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `GOLD_COMPARISON_SUBDIRS = [ "results", "summaries", "plots", "statistics", "metadata", "config", "lineage",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `GOLD_BASELINE_ARTIFACT_DIRS = build_artifact_dirs( artifacts_root=ARTIFACTS_ROOT, stage="gold", dataset_name=DATASET_NAME, family="baseline", subdirs=GOLD_COMMON_SUBDIRS,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `build_artifact_dirs`
- `export_config_snapshot`
- `get`
- `Path`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Gold comparison artifact directories` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Gold 04 compares outputs from Gold 02 / 03A / 03B / 03C.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Its CONFIG may not contain those notebooks' stage-specific config sections,` | Documents the purpose or boundary of the surrounding notebook step. |
| `# so we build the known Gold artifact families directly from the shared artifact root.` | Documents the purpose or boundary of the surrounding notebook step. |
| `ARTIFACTS_ROOT = Path(CONFIG["resolved_paths"]["artifacts_root"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_COMMON_SUBDIRS = [ "scores", "summaries", "thresholds", "metadata", "models", "plots", "config", "lineage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_COMPARISON_SUBDIRS = [ "results", "summaries", "plots", "statistics", "metadata", "config", "lineage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_BASELINE_ARTIFACT_DIRS = build_artifact_dirs( artifacts_root=ARTIFACTS_ROOT, stage="gold", dataset_name=DATASET_NAME, family="baseline", subdirs=GOLD_COMMON_SUBDIRS,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_CASCADE_DEFAULTS_ARTIFACT_DIRS = build_artifact_dirs( artifacts_root=ARTIFACTS_ROOT, stage="gold", dataset_name=DATASET_NAME, family="cascade_defaults", subdirs=GOLD_COMMON_SU` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_CASCADE_TUNED_ARTIFACT_DIRS = build_artifact_dirs( artifacts_root=ARTIFACTS_ROOT, stage="gold", dataset_name=DATASET_NAME, family="cascade_tuned", subdirs=GOLD_COMMON_SUBDIRS,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_CASCADE_STAGE3_ARTIFACT_DIRS = build_artifact_dirs( artifacts_root=ARTIFACTS_ROOT, stage="gold", dataset_name=DATASET_NAME, family="cascade_stage3_improved", subdirs=GOLD_COMM` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_COMPARISON_ARTIFACT_DIRS = build_artifact_dirs( artifacts_root=ARTIFACTS_ROOT, stage="gold", dataset_name=DATASET_NAME, family="comparison", subdirs=GOLD_COMPARISON_SUBDIRS,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_ARTIFACTS_PATH = GOLD_COMPARISON_ARTIFACT_DIRS["stage_dataset_root"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_COMPARISON_CONFIG_DIR = GOLD_COMPARISON_ARTIFACT_DIRS["config"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_SNAPSHOT_PATH = ( GOLD_COMPARISON_CONFIG_DIR / f"{DATASET_NAME}__gold_comparison__resolved_config.yaml"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if CONFIG["execution"].get("save_config_snapshot", True): export_config_snapshot(CONFIG, CONFIG_SNAPSHOT_PATH)` | Controls validation, iteration, file handling, or error handling for this step. |
| `BASELINE_RESULTS_PATH_CSV = ( GOLD_BASELINE_ARTIFACT_DIRS["scores"] / FILENAMES["baseline_results_file_name_csv"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BASELINE_RESULTS_PATH_PICKLE = ( GOLD_BASELINE_ARTIFACT_DIRS["scores"] / FILENAMES["baseline_results_file_name_pickle"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BASELINE_SUMMARY_PATH = ( GOLD_BASELINE_ARTIFACT_DIRS["summaries"] / FILENAMES["baseline_summary_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BASELINE_THRESHOLDS_PATH = ( GOLD_BASELINE_ARTIFACT_DIRS["thresholds"] / FILENAMES["baseline_thresholds_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BASELINE_METADATA_PATH = ( GOLD_BASELINE_ARTIFACT_DIRS["metadata"] / FILENAMES["baseline_metadata_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_DEFAULTS_RESULTS_PATH_CSV = ( GOLD_CASCADE_DEFAULTS_ARTIFACT_DIRS["scores"] / FILENAMES["cascade_defaults_results_file_name_csv"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_DEFAULTS_RESULTS_PATH_PICKLE = ( GOLD_CASCADE_DEFAULTS_ARTIFACT_DIRS["scores"] / FILENAMES["cascade_defaults_results_file_name_pickle"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_DEFAULTS_SUMMARY_PATH = ( GOLD_CASCADE_DEFAULTS_ARTIFACT_DIRS["summaries"] / FILENAMES["cascade_defaults_summary_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_DEFAULTS_THRESHOLDS_PATH = ( GOLD_CASCADE_DEFAULTS_ARTIFACT_DIRS["thresholds"] / FILENAMES["cascade_defaults_thresholds_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_DEFAULTS_METADATA_PATH = ( GOLD_CASCADE_DEFAULTS_ARTIFACT_DIRS["metadata"] / FILENAMES["cascade_defaults_metadata_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_TUNED_RESULTS_PATH_CSV = ( GOLD_CASCADE_TUNED_ARTIFACT_DIRS["scores"] / FILENAMES["cascade_tuned_results_file_name_csv"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_TUNED_RESULTS_PATH_PICKLE = ( GOLD_CASCADE_TUNED_ARTIFACT_DIRS["scores"] / FILENAMES["cascade_tuned_results_file_name_pickle"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_TUNED_SUMMARY_PATH = ( GOLD_CASCADE_TUNED_ARTIFACT_DIRS["summaries"] / FILENAMES["cascade_tuned_summary_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_TUNED_THRESHOLDS_PATH = ( GOLD_CASCADE_TUNED_ARTIFACT_DIRS["thresholds"] / FILENAMES["cascade_tuned_thresholds_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_TUNED_METADATA_PATH = ( GOLD_CASCADE_TUNED_ARTIFACT_DIRS["metadata"] / FILENAMES["cascade_tuned_metadata_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_STAGE3_IMPROVED_RESULTS_PATH_CSV = ( GOLD_CASCADE_STAGE3_ARTIFACT_DIRS["scores"] / FILENAMES["cascade_stage3_improved_results_file_name_csv"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_STAGE3_IMPROVED_RESULTS_PATH_PICKLE = ( GOLD_CASCADE_STAGE3_ARTIFACT_DIRS["scores"] / FILENAMES["cascade_stage3_improved_results_file_name_pickle"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `26 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 08 — Load Configuration, Paths, and Comparison Runtime Settings

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

## Code Cell 10 — Start Logging for the Gold Comparison Stage

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

## Code Cell 11 — Start Logging for the Gold Comparison Stage

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
- `gold_model_comparison`
- `info`
- `Initial`
- `Initiate`
- `initiation`
- `load`
- `loads`
- `Log`
- `log`
- `log_layer_paths`
- `logging`

### Outputs

- `gold_log_path`
- `level`
- `logger`
- `overwrite_handlers`

### Key Operations

- `"""`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Original Logging Setup`: Documents the purpose or boundary of the surrounding notebook step.
- `# Create gold log path`: Documents the purpose or boundary of the surrounding notebook step.
- `gold_log_path = paths.logs / "gold_model_comparison.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Initial Logger`: Documents the purpose or boundary of the surrounding notebook step.
- `configure_logging( "capstone", gold_log_path, level=logging.DEBUG, overwrite_handlers=True,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Initiate Logger and log file`: Documents the purpose or boundary of the surrounding notebook step.
- `logger = logging.getLogger("capstone.gold")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Log load and initiation`: Documents the purpose or boundary of the surrounding notebook step.
- `logger.info("Gold Modeling stage starting")`: Writes a logger message for traceability during notebook execution.
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
| `gold_log_path = paths.logs / "gold_model_comparison.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Initial Logger` | Documents the purpose or boundary of the surrounding notebook step. |
| `configure_logging( "capstone", gold_log_path, level=logging.DEBUG, overwrite_handlers=True,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Initiate Logger and log file` | Documents the purpose or boundary of the surrounding notebook step. |
| `logger = logging.getLogger("capstone.gold")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Log load and initiation` | Documents the purpose or boundary of the surrounding notebook step. |
| `logger.info("Gold Modeling stage starting")` | Writes a logger message for traceability during notebook execution. |
| `# Log paths loads` | Documents the purpose or boundary of the surrounding notebook step. |
| `log_layer_paths(paths, current_layer="gold", logger=logger) """` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 12 — Initialize Experiment Tracking

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `B`
- `BASELINE_RESULTS_PATH_CSV`
- `baseline_results_path_csv`
- `BASELINE_RESULTS_PATH_PICKLE`
- `baseline_results_path_pickle`
- `BASELINE_SUMMARY_PATH`
- `baseline_summary_path`
- `CASCADE_DEFAULTS_METADATA_PATH`
- `cascade_defaults_metadata_path`
- `cascade_defaults_results_path_csv`
- `CASCADE_DEFAULTS_RESULTS_PATH_CSV`
- `cascade_defaults_results_path_pickle`
- `CASCADE_DEFAULTS_RESULTS_PATH_PICKLE`
- `cascade_defaults_summary_path`
- `CASCADE_DEFAULTS_SUMMARY_PATH`
- `CASCADE_DEFAULTS_THRESHOLDS_PATH`
- `cascade_defaults_thresholds_path`
- `CASCADE_STAGE3_IMPROVED_METADATA_PATH`
- `cascade_stage3_improved_metadata_path`
- `CASCADE_STAGE3_IMPROVED_RESULTS_PATH_CSV`

### Outputs

- `config`
- `entity`
- `job_type`
- `name`
- `project`
- `wandb_run`

### Key Operations

- `# W&B`: Documents the purpose or boundary of the surrounding notebook step.
- `wandb_run = wandb.init( project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="gold_model_comparison", config={ "gold_version": GOLD_VERSION, "dataset": DATASET`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger.info("W&B initialized: %s", wandb_run.name)`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `info`
- `init`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# W&B` | Documents the purpose or boundary of the surrounding notebook step. |
| `wandb_run = wandb.init( project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="gold_model_comparison", config={ "gold_version": GOLD_VERSION, "dataset": DATASET` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("W&B initialized: %s", wandb_run.name)` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- Artifact or state outputs detected: optional experiment tracking call.

## Code Cell 13 — Initialize the Comparison Ledger

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `init`
- `Initialized`
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

- `# Ledger Setup`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger = Ledger(stage=STAGE, recipe_id=RECIPE_ID)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ledger.add( kind="step", step="init", message="Initialized ledger", logger=logger`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `Ledger`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Ledger Setup` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger = Ledger(stage=STAGE, recipe_id=RECIPE_ID)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger.add( kind="step", step="init", message="Initialized ledger", logger=logger` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 14 — Load the Baseline and Cascade Artifacts and Validate Shared Gold Lineage

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `baseline_metadata`
- `BASELINE_METADATA_PATH`
- `BASELINE_RESULTS_PATH_PICKLE`
- `baseline_summary`
- `BASELINE_SUMMARY_PATH`
- `baseline_thresholds`
- `BASELINE_THRESHOLDS_PATH`
- `cascade_defaults_metadata`
- `CASCADE_DEFAULTS_METADATA_PATH`
- `CASCADE_DEFAULTS_RESULTS_PATH_PICKLE`
- `cascade_defaults_summary`
- `CASCADE_DEFAULTS_SUMMARY_PATH`
- `cascade_defaults_thresholds`
- `CASCADE_DEFAULTS_THRESHOLDS_PATH`
- `cascade_stage3_improved_metadata`
- `CASCADE_STAGE3_IMPROVED_METADATA_PATH`
- `CASCADE_STAGE3_IMPROVED_RESULTS_PATH_PICKLE`
- `cascade_stage3_improved_summary`
- `CASCADE_STAGE3_IMPROVED_SUMMARY_PATH`

### Outputs

- `baseline_results`
- `baseline_results_pickle`
- `cascade_defaults_results`
- `cascade_defaults_results_pickle`
- `cascade_stage3_improved_results`
- `cascade_stage3_improved_results_pickle`
- `cascade_tuned_results`
- `cascade_tuned_results_pickle`

### Key Operations

- `baseline_results_pickle = pd.read_pickle(BASELINE_RESULTS_PATH_PICKLE)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `baseline_results = baseline_results_pickle`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `baseline_summary: dict[str, Any] = require_truth_record( load_json(BASELINE_SUMMARY_PATH), "baseline_summary",`: Loads input data, configuration, or artifacts required by the current stage.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `baseline_thresholds: dict[str, Any] = require_truth_record( load_json(BASELINE_THRESHOLDS_PATH), "baseline_thresholds",`: Loads input data, configuration, or artifacts required by the current stage.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `baseline_metadata: dict[str, Any] = require_truth_record( load_json(BASELINE_METADATA_PATH), "baseline_metadata",`: Loads input data, configuration, or artifacts required by the current stage.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `cascade_defaults_summary: dict[str, Any] = require_truth_record( load_json(CASCADE_DEFAULTS_SUMMARY_PATH), "cascade_defaults_summary",`: Loads input data, configuration, or artifacts required by the current stage.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `cascade_defaults_thresholds: dict[str, Any] = require_truth_record( load_json(CASCADE_DEFAULTS_THRESHOLDS_PATH), "cascade_defaults_thresholds",`: Loads input data, configuration, or artifacts required by the current stage.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `load_json`
- `read_pickle`
- `require_truth_record`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `baseline_results_pickle = pd.read_pickle(BASELINE_RESULTS_PATH_PICKLE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `baseline_results = baseline_results_pickle` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `baseline_summary: dict[str, Any] = require_truth_record( load_json(BASELINE_SUMMARY_PATH), "baseline_summary",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_thresholds: dict[str, Any] = require_truth_record( load_json(BASELINE_THRESHOLDS_PATH), "baseline_thresholds",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_metadata: dict[str, Any] = require_truth_record( load_json(BASELINE_METADATA_PATH), "baseline_metadata",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_defaults_summary: dict[str, Any] = require_truth_record( load_json(CASCADE_DEFAULTS_SUMMARY_PATH), "cascade_defaults_summary",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_defaults_thresholds: dict[str, Any] = require_truth_record( load_json(CASCADE_DEFAULTS_THRESHOLDS_PATH), "cascade_defaults_thresholds",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_defaults_metadata: dict[str, Any] = require_truth_record( load_json(CASCADE_DEFAULTS_METADATA_PATH), "cascade_defaults_metadata",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_defaults_results_pickle = pd.read_pickle(CASCADE_DEFAULTS_RESULTS_PATH_PICKLE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_defaults_results = cascade_defaults_results_pickle` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_tuned_summary: dict[str, Any] = require_truth_record( load_json(CASCADE_TUNED_SUMMARY_PATH), "cascade_tuned_summary",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_tuned_thresholds: dict[str, Any] = require_truth_record( load_json(CASCADE_TUNED_THRESHOLDS_PATH), "cascade_tuned_thresholds",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_tuned_metadata: dict[str, Any] = require_truth_record( load_json(CASCADE_TUNED_METADATA_PATH), "cascade_tuned_metadata",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_tuned_results_pickle = pd.read_pickle(CASCADE_TUNED_RESULTS_PATH_PICKLE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_tuned_results = cascade_tuned_results_pickle` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_stage3_improved_summary: dict[str, Any] = require_truth_record( load_json(CASCADE_STAGE3_IMPROVED_SUMMARY_PATH), "cascade_stage3_improved_summary",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_stage3_improved_thresholds: dict[str, Any] = require_truth_record( load_json(CASCADE_STAGE3_IMPROVED_THRESHOLDS_PATH), "cascade_stage3_improved_thresholds",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_stage3_improved_metadata: dict[str, Any] = require_truth_record( load_json(CASCADE_STAGE3_IMPROVED_METADATA_PATH), "cascade_stage3_improved_metadata",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_stage3_improved_results_pickle = pd.read_pickle(CASCADE_STAGE3_IMPROVED_RESULTS_PATH_PICKLE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_stage3_improved_results = cascade_stage3_improved_results_pickle` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 15 — Resolve and Validate Baseline Stage Truth

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `Any`
- `artifact_paths`
- `baseline`
- `Baseline`
- `baseline_metadata`
- `baseline_results`
- `baseline_truth`
- `cascade`
- `Cascade`
- `cascade_defaults_metadata`
- `cascade_defaults_results`
- `cascade_defaults_truth`
- `cascade_stage3_improved_metadata`
- `cascade_stage3_improved_results`
- `cascade_stage3_improved_truth`
- `cascade_truth_hash`
- `cascade_truth_path`
- `cascade_tuned_metadata`
- `cascade_tuned_results`

### Outputs

- `BASELINE_PARENT_GOLD_TRUTH_HASH`
- `baseline_results_truth_hash`
- `baseline_truth_artifact_paths`
- `BASELINE_TRUTH_HASH`
- `BASELINE_TRUTH_PATH`
- `BASELINE_TRUTH_PATH_VALUE`
- `baseline_truth_runtime_facts`
- `cascade_defaults_results_truth_hash`
- `CASCADE_DEFAULTS_TRUTH_HASH`
- `CASCADE_DEFAULTS_TRUTH_PATH`
- `CASCADE_DEFAULTS_TRUTH_PATH_VALUE`
- `cascade_stage3_improved_results_truth_hash`
- `CASCADE_STAGE3_IMPROVED_TRUTH_HASH`
- `CASCADE_STAGE3_IMPROVED_TRUTH_PATH`
- `CASCADE_STAGE3_IMPROVED_TRUTH_PATH_VALUE`
- `cascade_tuned_results_truth_hash`
- `CASCADE_TUNED_TRUTH_HASH`
- `CASCADE_TUNED_TRUTH_PATH`
- `CASCADE_TUNED_TRUTH_PATH_VALUE`

### Key Operations

- `# -------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Resolve and validate Baseline stage truth`: Documents the purpose or boundary of the surrounding notebook step.
- `# -------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `BASELINE_TRUTH_HASH = baseline_metadata.get("baseline_truth_hash")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `BASELINE_TRUTH_PATH_VALUE = baseline_metadata.get("baseline_truth_path")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if BASELINE_TRUTH_HASH is None: raise ValueError("baseline_metadata is missing 'baseline_truth_hash'.")`: Controls validation, iteration, file handling, or error handling for this step.
- `if BASELINE_TRUTH_PATH_VALUE is None: raise ValueError("baseline_metadata is missing 'baseline_truth_path'.")`: Controls validation, iteration, file handling, or error handling for this step.
- `BASELINE_TRUTH_PATH = Path(BASELINE_TRUTH_PATH_VALUE)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if not BASELINE_TRUTH_PATH.exists(): raise FileNotFoundError(f"Baseline truth file not found: {BASELINE_TRUTH_PATH}")`: Controls validation, iteration, file handling, or error handling for this step.
- `baseline_truth: dict[str, Any] = require_truth_record( load_json(BASELINE_TRUTH_PATH), "baseline_truth",`: Loads input data, configuration, or artifacts required by the current stage.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if baseline_truth.get("truth_hash") != BASELINE_TRUTH_HASH: raise ValueError( "Baseline metadata truth hash does not match the loaded baseline truth file:\n" f"metadata={BASELINE_T`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `exists`
- `extract_truth_hash`
- `FileNotFoundError`
- `get`
- `get_parent_truth_hash`
- `load_json`
- `Path`
- `require_truth_record`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# -------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Resolve and validate Baseline stage truth` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `BASELINE_TRUTH_HASH = baseline_metadata.get("baseline_truth_hash")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BASELINE_TRUTH_PATH_VALUE = baseline_metadata.get("baseline_truth_path")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if BASELINE_TRUTH_HASH is None: raise ValueError("baseline_metadata is missing 'baseline_truth_hash'.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if BASELINE_TRUTH_PATH_VALUE is None: raise ValueError("baseline_metadata is missing 'baseline_truth_path'.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `BASELINE_TRUTH_PATH = Path(BASELINE_TRUTH_PATH_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not BASELINE_TRUTH_PATH.exists(): raise FileNotFoundError(f"Baseline truth file not found: {BASELINE_TRUTH_PATH}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `baseline_truth: dict[str, Any] = require_truth_record( load_json(BASELINE_TRUTH_PATH), "baseline_truth",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if baseline_truth.get("truth_hash") != BASELINE_TRUTH_HASH: raise ValueError( "Baseline metadata truth hash does not match the loaded baseline truth file:\n" f"metadata={BASELINE_T` | Controls validation, iteration, file handling, or error handling for this step. |
| `baseline_results_truth_hash = extract_truth_hash(baseline_results)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if baseline_results_truth_hash is None: raise ValueError("Could not resolve meta__truth_hash from baseline_results CSV.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if baseline_results_truth_hash != BASELINE_TRUTH_HASH: raise ValueError( "Baseline results CSV truth hash does not match baseline_metadata['baseline_truth_hash']:\n" f"csv={baselin` | Controls validation, iteration, file handling, or error handling for this step. |
| `BASELINE_PARENT_GOLD_TRUTH_HASH = get_parent_truth_hash(baseline_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if BASELINE_PARENT_GOLD_TRUTH_HASH is None: raise ValueError("baseline_truth is missing a usable parent_truth_hash.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `baseline_truth_runtime_facts = baseline_truth.get("runtime_facts", {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `baseline_truth_artifact_paths = baseline_truth.get("artifact_paths", {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# -------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Resolve and validate Cascade Defaults stage truth` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `CASCADE_DEFAULTS_TRUTH_HASH = cascade_defaults_metadata.get("cascade_truth_hash")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CASCADE_DEFAULTS_TRUTH_PATH_VALUE = cascade_defaults_metadata.get("cascade_truth_path")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if CASCADE_DEFAULTS_TRUTH_HASH is None: raise ValueError("cascade_defaults_metadata is missing 'cascade_truth_hash'.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if CASCADE_DEFAULTS_TRUTH_PATH_VALUE is None: raise ValueError("cascade_defaults_metadata is missing 'cascade_truth_path'.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `CASCADE_DEFAULTS_TRUTH_PATH = Path(CASCADE_DEFAULTS_TRUTH_PATH_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not CASCADE_DEFAULTS_TRUTH_PATH.exists(): raise FileNotFoundError(f"Cascade defaults truth file not found: {CASCADE_DEFAULTS_TRUTH_PATH}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `cascade_defaults_truth: dict[str, Any] = require_truth_record( load_json(CASCADE_DEFAULTS_TRUTH_PATH), "cascade_defaults_truth",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if cascade_defaults_truth.get("truth_hash") != CASCADE_DEFAULTS_TRUTH_HASH: raise ValueError( "Cascade defaults metadata truth hash does not match the loaded cascade defaults truth` | Controls validation, iteration, file handling, or error handling for this step. |
| `cascade_defaults_results_truth_hash = extract_truth_hash(cascade_defaults_results)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if cascade_defaults_results_truth_hash is None: raise ValueError("Could not resolve meta__truth_hash from cascade_defaults_results CSV.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if cascade_defaults_results_truth_hash != CASCADE_DEFAULTS_TRUTH_HASH: raise ValueError( "Cascade defaults results CSV truth hash does not match cascade_defaults_metadata['cascade_` | Controls validation, iteration, file handling, or error handling for this step. |
| `# -------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Resolve and validate Cascade Tuned stage truth` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `CASCADE_TUNED_TRUTH_HASH = cascade_tuned_metadata.get("cascade_truth_hash")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CASCADE_TUNED_TRUTH_PATH_VALUE = cascade_tuned_metadata.get("cascade_truth_path")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if CASCADE_TUNED_TRUTH_HASH is None: raise ValueError("cascade_tuned_metadata is missing 'cascade_truth_hash'.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if CASCADE_TUNED_TRUTH_PATH_VALUE is None: raise ValueError("cascade_tuned_metadata is missing 'cascade_truth_path'.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `CASCADE_TUNED_TRUTH_PATH = Path(CASCADE_TUNED_TRUTH_PATH_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not CASCADE_TUNED_TRUTH_PATH.exists(): raise FileNotFoundError(f"Cascade tuned truth file not found: {CASCADE_TUNED_TRUTH_PATH}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `cascade_tuned_truth: dict[str, Any] = require_truth_record( load_json(CASCADE_TUNED_TRUTH_PATH), "cascade_tuned_truth",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if cascade_tuned_truth.get("truth_hash") != CASCADE_TUNED_TRUTH_HASH: raise ValueError( "Cascade tuned metadata truth hash does not match the loaded cascade tuned truth file:\n" f"` | Controls validation, iteration, file handling, or error handling for this step. |
| `cascade_tuned_results_truth_hash = extract_truth_hash(cascade_tuned_results)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if cascade_tuned_results_truth_hash is None: raise ValueError("Could not resolve meta__truth_hash from cascade_tuned_results CSV.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if cascade_tuned_results_truth_hash != CASCADE_TUNED_TRUTH_HASH: raise ValueError( "Cascade tuned results CSV truth hash does not match cascade_tuned_metadata['cascade_truth_hash']` | Controls validation, iteration, file handling, or error handling for this step. |
| `# -------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Resolve and validate Cascade Stage 3 Improved stage truth` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `CASCADE_STAGE3_IMPROVED_TRUTH_HASH = cascade_stage3_improved_metadata.get("cascade_truth_hash")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CASCADE_STAGE3_IMPROVED_TRUTH_PATH_VALUE = cascade_stage3_improved_metadata.get("cascade_truth_path")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if CASCADE_STAGE3_IMPROVED_TRUTH_HASH is None: raise ValueError("cascade_stage3_improved_metadata is missing 'cascade_truth_hash'.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if CASCADE_STAGE3_IMPROVED_TRUTH_PATH_VALUE is None: raise ValueError("cascade_stage3_improved_metadata is missing 'cascade_truth_path'.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `CASCADE_STAGE3_IMPROVED_TRUTH_PATH = Path(CASCADE_STAGE3_IMPROVED_TRUTH_PATH_VALUE)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not CASCADE_STAGE3_IMPROVED_TRUTH_PATH.exists(): raise FileNotFoundError( f"Cascade stage3 improved truth file not found: {CASCADE_STAGE3_IMPROVED_TRUTH_PATH}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `cascade_stage3_improved_truth: dict[str, Any] = require_truth_record( load_json(CASCADE_STAGE3_IMPROVED_TRUTH_PATH), "cascade_stage3_improved_truth",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `4 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 16 — Shared Parent Truth Validation

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `baseline`
- `baseline_results`
- `baseline_results_truth_hash`
- `baseline_truth`
- `BASELINE_TRUTH_HASH`
- `BASELINE_TRUTH_PATH`
- `cascade_default`
- `cascade_defaults_results`
- `cascade_defaults_results_truth_hash`
- `cascade_defaults_truth`
- `CASCADE_DEFAULTS_TRUTH_HASH`
- `CASCADE_DEFAULTS_TRUTH_PATH`
- `cascade_stage3_improved`
- `cascade_stage3_improved_results`
- `cascade_stage3_improved_results_truth_hash`
- `cascade_stage3_improved_truth`
- `CASCADE_STAGE3_IMPROVED_TRUTH_HASH`
- `CASCADE_STAGE3_IMPROVED_TRUTH_PATH`
- `cascade_tuned`
- `cascade_tuned_results`

### Outputs

- `BASELINE_PARENT_GOLD_TRUTH_HASH`
- `DEFAULTS_PARENT_GOLD_TRUTH_HASH`
- `lineage_df`
- `STAGE3_IMPROVED_PARENT_GOLD_TRUTH_HASH`
- `TUNED_PARENT_GOLD_TRUTH_HASH`

### Key Operations

- `DEFAULTS_PARENT_GOLD_TRUTH_HASH = get_parent_truth_hash(cascade_defaults_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TUNED_PARENT_GOLD_TRUTH_HASH = get_parent_truth_hash(cascade_tuned_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `STAGE3_IMPROVED_PARENT_GOLD_TRUTH_HASH = get_parent_truth_hash(cascade_stage3_improved_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `BASELINE_PARENT_GOLD_TRUTH_HASH = get_parent_truth_hash(baseline_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `lineage_df = pd.DataFrame( [ { "model_id": "baseline", "dataset_name": get_dataset_name_from_truth(baseline_truth), "stage_truth_hash": BASELINE_TRUTH_HASH, "parent_gold_truth_hash`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Comparison lineage summary:")`: Displays a notebook-facing result for inspection.
- `display(lineage_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `DataFrame`
- `display`
- `get_dataset_name_from_truth`
- `get_parent_truth_hash`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `DEFAULTS_PARENT_GOLD_TRUTH_HASH = get_parent_truth_hash(cascade_defaults_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TUNED_PARENT_GOLD_TRUTH_HASH = get_parent_truth_hash(cascade_tuned_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE3_IMPROVED_PARENT_GOLD_TRUTH_HASH = get_parent_truth_hash(cascade_stage3_improved_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BASELINE_PARENT_GOLD_TRUTH_HASH = get_parent_truth_hash(baseline_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `lineage_df = pd.DataFrame( [ { "model_id": "baseline", "dataset_name": get_dataset_name_from_truth(baseline_truth), "stage_truth_hash": BASELINE_TRUTH_HASH, "parent_gold_truth_hash` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Comparison lineage summary:")` | Displays a notebook-facing result for inspection. |
| `display(lineage_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 17 — Resolve comparison dataset identity

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold__truth__`
- `a`
- `add`
- `Any`
- `are`
- `artifact`
- `artifact_paths`
- `artifacts`
- `baseline`
- `Baseline`
- `baseline_dataset`
- `baseline_metadata_path`
- `BASELINE_METADATA_PATH`
- `baseline_parent`
- `BASELINE_PARENT_GOLD_TRUTH_HASH`
- `baseline_parent_gold_truth_hash`
- `baseline_result_rows`
- `baseline_results`
- `baseline_results_path_csv`
- `BASELINE_RESULTS_PATH_CSV`

### Outputs

- `CASCADE_DEFAULTS_DATASET_NAME`
- `CASCADE_STAGE3_IMPROVED_DATASET_NAME`
- `CASCADE_TUNED_DATASET_NAME`
- `COMPARISON_PARENT_GOLD_TRUTH_HASH`
- `data`
- `DATASET_NAME`
- `GOLD_PARENT_TRUTH_HASH`
- `gold_truth_artifact_paths`
- `GOLD_TRUTH_PATH`
- `gold_truth_runtime_facts`
- `kind`
- `logger`
- `message`
- `PIPELINE_MODE`
- `PIPELINE_MODE_FROM_BASELINE_TRUTH`
- `step`

### Key Operations

- `DATASET_NAME = get_dataset_name_from_truth(baseline_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CASCADE_DEFAULTS_DATASET_NAME = get_dataset_name_from_truth(cascade_defaults_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CASCADE_TUNED_DATASET_NAME = get_dataset_name_from_truth(cascade_tuned_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CASCADE_STAGE3_IMPROVED_DATASET_NAME = get_dataset_name_from_truth(cascade_stage3_improved_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if DATASET_NAME != CASCADE_DEFAULTS_DATASET_NAME: raise ValueError( "Baseline and Cascade Defaults do not share the same dataset_name:\n" f"baseline_dataset={DATASET_NAME}\n" f"cas`: Controls validation, iteration, file handling, or error handling for this step.
- `if DATASET_NAME != CASCADE_TUNED_DATASET_NAME: raise ValueError( "Baseline and Cascade Tuned do not share the same dataset_name:\n" f"baseline_dataset={DATASET_NAME}\n" f"cascade_t`: Controls validation, iteration, file handling, or error handling for this step.
- `if DATASET_NAME != CASCADE_STAGE3_IMPROVED_DATASET_NAME: raise ValueError( "Baseline and Cascade Stage3 Improved do not share the same dataset_name:\n" f"baseline_dataset={DATASET_`: Controls validation, iteration, file handling, or error handling for this step.
- `COMPARISON_PARENT_GOLD_TRUTH_HASH = DEFAULTS_PARENT_GOLD_TRUTH_HASH`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if COMPARISON_PARENT_GOLD_TRUTH_HASH is None: raise ValueError( "Cascade defaults truth is missing a usable parent_gold_truth_hash. " "Comparison lineage cannot be resolved." )`: Controls validation, iteration, file handling, or error handling for this step.
- `if TUNED_PARENT_GOLD_TRUTH_HASH != COMPARISON_PARENT_GOLD_TRUTH_HASH: raise ValueError( "Cascade tuned does not share the same parent Gold truth hash as cascade defaults.\n" f"casc`: Controls validation, iteration, file handling, or error handling for this step.
- `if STAGE3_IMPROVED_PARENT_GOLD_TRUTH_HASH != COMPARISON_PARENT_GOLD_TRUTH_HASH: raise ValueError( "Cascade stage3 improved does not share the same parent Gold truth hash as cascade`: Controls validation, iteration, file handling, or error handling for this step.
- `if BASELINE_PARENT_GOLD_TRUTH_HASH != COMPARISON_PARENT_GOLD_TRUTH_HASH: raise ValueError( "Baseline artifacts are from a different Gold lineage than the cascade artifacts used for`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `add`
- `display`
- `exists`
- `FileNotFoundError`
- `get`
- `get_dataset_name_from_truth`
- `get_pipeline_mode_from_truth`
- `head`
- `info`
- `load_json`
- `require_truth_record`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `DATASET_NAME = get_dataset_name_from_truth(baseline_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CASCADE_DEFAULTS_DATASET_NAME = get_dataset_name_from_truth(cascade_defaults_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CASCADE_TUNED_DATASET_NAME = get_dataset_name_from_truth(cascade_tuned_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CASCADE_STAGE3_IMPROVED_DATASET_NAME = get_dataset_name_from_truth(cascade_stage3_improved_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if DATASET_NAME != CASCADE_DEFAULTS_DATASET_NAME: raise ValueError( "Baseline and Cascade Defaults do not share the same dataset_name:\n" f"baseline_dataset={DATASET_NAME}\n" f"cas` | Controls validation, iteration, file handling, or error handling for this step. |
| `if DATASET_NAME != CASCADE_TUNED_DATASET_NAME: raise ValueError( "Baseline and Cascade Tuned do not share the same dataset_name:\n" f"baseline_dataset={DATASET_NAME}\n" f"cascade_t` | Controls validation, iteration, file handling, or error handling for this step. |
| `if DATASET_NAME != CASCADE_STAGE3_IMPROVED_DATASET_NAME: raise ValueError( "Baseline and Cascade Stage3 Improved do not share the same dataset_name:\n" f"baseline_dataset={DATASET_` | Controls validation, iteration, file handling, or error handling for this step. |
| `COMPARISON_PARENT_GOLD_TRUTH_HASH = DEFAULTS_PARENT_GOLD_TRUTH_HASH` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if COMPARISON_PARENT_GOLD_TRUTH_HASH is None: raise ValueError( "Cascade defaults truth is missing a usable parent_gold_truth_hash. " "Comparison lineage cannot be resolved." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `if TUNED_PARENT_GOLD_TRUTH_HASH != COMPARISON_PARENT_GOLD_TRUTH_HASH: raise ValueError( "Cascade tuned does not share the same parent Gold truth hash as cascade defaults.\n" f"casc` | Controls validation, iteration, file handling, or error handling for this step. |
| `if STAGE3_IMPROVED_PARENT_GOLD_TRUTH_HASH != COMPARISON_PARENT_GOLD_TRUTH_HASH: raise ValueError( "Cascade stage3 improved does not share the same parent Gold truth hash as cascade` | Controls validation, iteration, file handling, or error handling for this step. |
| `if BASELINE_PARENT_GOLD_TRUTH_HASH != COMPARISON_PARENT_GOLD_TRUTH_HASH: raise ValueError( "Baseline artifacts are from a different Gold lineage than the cascade artifacts used for` | Controls validation, iteration, file handling, or error handling for this step. |
| `GOLD_PARENT_TRUTH_HASH = COMPARISON_PARENT_GOLD_TRUTH_HASH` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PIPELINE_MODE_FROM_BASELINE_TRUTH = get_pipeline_mode_from_truth(baseline_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if PIPELINE_MODE_FROM_BASELINE_TRUTH is not None: PIPELINE_MODE = PIPELINE_MODE_FROM_BASELINE_TRUTH` | Controls validation, iteration, file handling, or error handling for this step. |
| `GOLD_TRUTH_PATH = ( TRUTHS_PATH / "gold" / f"{DATASET_NAME}__gold__truth__{GOLD_PARENT_TRUTH_HASH}.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if not GOLD_TRUTH_PATH.exists(): raise FileNotFoundError(f"Gold truth file not found: {GOLD_TRUTH_PATH}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `gold_truth: dict[str, Any] = require_truth_record( load_json(GOLD_TRUTH_PATH), "gold_truth",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth_runtime_facts = gold_truth.get("runtime_facts", {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_truth_artifact_paths = gold_truth.get("artifact_paths", {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Resolved comparison dataset name from truth: %s", DATASET_NAME)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Resolved comparison Gold parent truth hash: %s", GOLD_PARENT_TRUTH_HASH)` | Writes a logger message for traceability during notebook execution. |
| `print("Comparison dataset name from truth:", DATASET_NAME)` | Displays a notebook-facing result for inspection. |
| `print("Comparison Gold parent truth hash:", GOLD_PARENT_TRUTH_HASH)` | Displays a notebook-facing result for inspection. |
| `ledger.add( kind="step", step="load_comparison_inputs", message=( "Loaded baseline/cascade outputs, validated their stage truth records, " "resolved the cascade comparison lineage,` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(baseline_results.head(10))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 18 — Build the Main Comparison Table

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `alert_count_test_rows`
- `baseline`
- `Baseline`
- `baseline_summary`
- `BASELINE_TRUTH_HASH`
- `Cascade`
- `cascade`
- `cascade_default`
- `cascade_defaults_summary`
- `CASCADE_DEFAULTS_TRUTH_HASH`
- `cascade_metrics`
- `cascade_stage3`
- `cascade_stage3_improved_summary`
- `CASCADE_STAGE3_IMPROVED_TRUTH_HASH`
- `cascade_tuned`
- `cascade_tuned_summary`
- `CASCADE_TUNED_TRUTH_HASH`
- `DataFrame`
- `Default`
- `f1`

### Outputs

- `baseline_metrics`
- `cascade_default_metrics`
- `cascade_stage3_improved_metrics`
- `cascade_tuned_metrics`
- `comparison_df`
- `comparison_rows`

### Key Operations

- `baseline_metrics = baseline_summary["baseline_metrics"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `cascade_default_metrics = cascade_defaults_summary["cascade_metrics"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `cascade_tuned_metrics = cascade_tuned_summary["cascade_metrics"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `cascade_stage3_improved_metrics = cascade_stage3_improved_summary["cascade_metrics"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `comparison_rows = [ { "model_id": "baseline", "model": "Baseline IsolationForest", "variant_family": "baseline", "stage3_mode": "none", "alert_count_test_rows": int(baseline_summar`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `comparison_df = pd.DataFrame(comparison_rows)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `DataFrame`
- `get`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `baseline_metrics = baseline_summary["baseline_metrics"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_default_metrics = cascade_defaults_summary["cascade_metrics"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_tuned_metrics = cascade_tuned_summary["cascade_metrics"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_stage3_improved_metrics = cascade_stage3_improved_summary["cascade_metrics"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `comparison_rows = [ { "model_id": "baseline", "model": "Baseline IsolationForest", "variant_family": "baseline", "stage3_mode": "none", "alert_count_test_rows": int(baseline_summar` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_df = pd.DataFrame(comparison_rows)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 19 — Build the Comparison Summary Metrics

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `alert_count_test_rows`
- `ascending`
- `baseline_f1`
- `baseline_metrics`
- `baseline_precision`
- `baseline_recall`
- `baseline_summary`
- `BASELINE_TRUTH_HASH`
- `baseline_truth_hash`
- `baseline_truth_path`
- `BASELINE_TRUTH_PATH`
- `baseline_vs_default_alert_reduction_count`
- `baseline_vs_default_alert_reduction_ratio`
- `baseline_vs_stage3_improved_alert_reduction_count`
- `baseline_vs_stage3_improved_alert_reduction_ratio`
- `baseline_vs_stage3_medium_alert_reduction_count`
- `baseline_vs_stage3_medium_alert_reduction_ratio`
- `baseline_vs_stage3_relaxed_alert_reduction_count`
- `baseline_vs_stage3_relaxed_alert_reduction_ratio`
- `baseline_vs_stage3_strict_alert_reduction_count`

### Outputs

- `baseline_alert_count_test_rows`
- `cascade_default_alert_count_test_rows`
- `cascade_tuned_alert_count_test_rows`
- `comparison_summary`
- `stage3_improved_alert_count_test_rows`
- `stage3_medium_alert_count_test_rows`
- `stage3_relaxed_alert_count_test_rows`
- `stage3_strict_alert_count_test_rows`

### Key Operations

- `baseline_alert_count_test_rows = int(baseline_summary["alert_count_test_rows"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `cascade_default_alert_count_test_rows = int(cascade_defaults_summary["final_cascade_alert_count_test_rows"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `cascade_tuned_alert_count_test_rows = int(cascade_tuned_summary["final_cascade_alert_count_test_rows"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stage3_improved_alert_count_test_rows = int( cascade_stage3_improved_summary["final_cascade_alert_count_test_rows"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage3_relaxed_alert_count_test_rows = int(cascade_stage3_improved_summary["stage3_relaxed_alert_count_test_rows"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stage3_medium_alert_count_test_rows = int(cascade_stage3_improved_summary["stage3_medium_alert_count_test_rows"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stage3_strict_alert_count_test_rows = int(cascade_stage3_improved_summary["stage3_strict_alert_count_test_rows"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `comparison_summary = { "baseline_alert_count_test_rows": baseline_alert_count_test_rows, "cascade_default_alert_count_test_rows": cascade_default_alert_count_test_rows, "cascade_tu`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `get`
- `max`
- `sort_values`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `baseline_alert_count_test_rows = int(baseline_summary["alert_count_test_rows"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_default_alert_count_test_rows = int(cascade_defaults_summary["final_cascade_alert_count_test_rows"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_tuned_alert_count_test_rows = int(cascade_tuned_summary["final_cascade_alert_count_test_rows"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage3_improved_alert_count_test_rows = int( cascade_stage3_improved_summary["final_cascade_alert_count_test_rows"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage3_relaxed_alert_count_test_rows = int(cascade_stage3_improved_summary["stage3_relaxed_alert_count_test_rows"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage3_medium_alert_count_test_rows = int(cascade_stage3_improved_summary["stage3_medium_alert_count_test_rows"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage3_strict_alert_count_test_rows = int(cascade_stage3_improved_summary["stage3_strict_alert_count_test_rows"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `comparison_summary = { "baseline_alert_count_test_rows": baseline_alert_count_test_rows, "cascade_default_alert_count_test_rows": cascade_default_alert_count_test_rows, "cascade_tu` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 20 — Display the Model Comparison Table

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `alert_count_test_rows`
- `Baseline`
- `Cascade`
- `Comparison`
- `comparison_df`
- `Default`
- `f1`
- `format`
- `Gold`
- `HTML`
- `Model`
- `Performance`
- `precision`
- `recall`
- `set_caption`
- `Stage`
- `style`
- `to_html`
- `Tuned`
- `Variants`

### Outputs

- `styled`

### Key Operations

- `styled = comparison_df.style.format( { "alert_count_test_rows": "{:,.0f}", "precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}", }`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `).set_caption( "Gold Model Performance Comparison: Baseline, Cascade Default, Cascade Tuned, and Stage 3 Variants"`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(HTML(styled.to_html()))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `format`
- `HTML`
- `set_caption`
- `to_html`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `styled = comparison_df.style.format( { "alert_count_test_rows": "{:,.0f}", "precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}", }` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).set_caption( "Gold Model Performance Comparison: Baseline, Cascade Default, Cascade Tuned, and Stage 3 Variants"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(HTML(styled.to_html()))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 21 — Paired Statistical Comparison of Baseline and Final Cascade

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `abs`
- `aggregate`
- `alert`
- `aligned`
- `anomaly_flag`
- `Any`
- `approximation`
- `are`
- `astype`
- `available`
- `baseline`
- `Baseline`
- `baseline_correct`
- `baseline_flag`
- `baseline_pred`
- `baseline_rows`
- `be`
- `binary`
- `bool`

### Outputs

- `_as_binary_series`
- `alternative`
- `b`
- `baseline_correct_comparison_wrong`
- `baseline_counts`
- `baseline_flag_column`
- `baseline_frame`
- `baseline_normal_rows`
- `baseline_required`
- `baseline_results`
- `baseline_wrong_comparison_correct`
- `both_correct`
- `both_wrong`
- `build_paired_model_frame`
- `build_statistical_test_summary`
- `c`
- `cleaned`
- `column_name`
- `comparison_counts`
- `comparison_flag_column`

### Key Operations

- `def _as_binary_series(series: pd.Series, *, column_name: str) -> pd.Series: """ Convert a model flag or label column into a clean 0/1 integer series. This keeps the statistical com`: Defines notebook-local logic used later in the notebook.
- `def build_paired_model_frame( baseline_results: pd.DataFrame, comparison_results: pd.DataFrame, *, baseline_flag_column: str = "baseline_flag", comparison_flag_column: str = "casca`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """ Build a row-aligned dataframe for paired statistical comparison. Expected use: - baseline_results comes from Gold 02 - comparison_results comes from Gold 03C`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def run_mcnemar_paired_test(paired_frame: pd.DataFrame) -> dict[str, Any]: """ Run McNemar's test on paired model correctness. Table layout: [[both correct, baseline correct / comp`: Defines notebook-local logic used later in the notebook.
- `def summarize_confusion_counts( paired_frame: pd.DataFrame, *, prediction_column: str,`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, int \| float]: """ Calculate confusion counts and standard model metrics for one model. """ y_true = paired_frame["y_true"].astype(int) y_pred = paired_frame[predicti`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def run_two_proportion_z_test( *, count_a: int, nobs_a: int, count_b: int, nobs_b: int, alternative: str = "two-sided",`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, float]: """ Two-proportion z-test. Used here as a secondary, aggregate-level check for false-positive-rate and precision differences. The paired McNemar test remains`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def build_statistical_test_summary( baseline_results: pd.DataFrame, comparison_results: pd.DataFrame, *, comparison_name: str = "Stage 3 Improved", comparison_flag_column: str = "c`: Defines notebook-local logic used later in the notebook.
- `) -> tuple[pd.DataFrame, dict[str, Any]]: """ Build the statistical summary used in the Task 3 report. """ paired_frame = build_paired_model_frame( baseline_results=baseline_result`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `_as_binary_series`
- `abs`
- `astype`
- `build_paired_model_frame`
- `build_statistical_test_summary`
- `cdf`
- `copy`
- `DataFrame`
- `fillna`
- `getattr`
- `max`
- `mcnemar`
- `merge`
- `run_mcnemar_paired_test`
- `run_two_proportion_z_test`
- `sf`
- `sorted`
- `sqrt`
- `sum`
- `summarize_confusion_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def _as_binary_series(series: pd.Series, *, column_name: str) -> pd.Series: """ Convert a model flag or label column into a clean 0/1 integer series. This keeps the statistical com` | Defines notebook-local logic used later in the notebook. |
| `def build_paired_model_frame( baseline_results: pd.DataFrame, comparison_results: pd.DataFrame, *, baseline_flag_column: str = "baseline_flag", comparison_flag_column: str = "casca` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Build a row-aligned dataframe for paired statistical comparison. Expected use: - baseline_results comes from Gold 02 - comparison_results comes from Gold 03C` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def run_mcnemar_paired_test(paired_frame: pd.DataFrame) -> dict[str, Any]: """ Run McNemar's test on paired model correctness. Table layout: [[both correct, baseline correct / comp` | Defines notebook-local logic used later in the notebook. |
| `def summarize_confusion_counts( paired_frame: pd.DataFrame, *, prediction_column: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, int \| float]: """ Calculate confusion counts and standard model metrics for one model. """ y_true = paired_frame["y_true"].astype(int) y_pred = paired_frame[predicti` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def run_two_proportion_z_test( *, count_a: int, nobs_a: int, count_b: int, nobs_b: int, alternative: str = "two-sided",` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, float]: """ Two-proportion z-test. Used here as a secondary, aggregate-level check for false-positive-rate and precision differences. The paired McNemar test remains` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def build_statistical_test_summary( baseline_results: pd.DataFrame, comparison_results: pd.DataFrame, *, comparison_name: str = "Stage 3 Improved", comparison_flag_column: str = "c` | Defines notebook-local logic used later in the notebook. |
| `) -> tuple[pd.DataFrame, dict[str, Any]]: """ Build the statistical summary used in the Task 3 report. """ paired_frame = build_paired_model_frame( baseline_results=baseline_result` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 22 — Build statistical comparison summary

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold__statistical_test_summary`
- `build_statistical_test_summary`
- `cascade_final_flag`
- `cascade_stage3_improved_results`
- `csv`
- `DATASET_NAME`
- `f`
- `GOLD_ARTIFACTS_PATH`
- `Improved`
- `index`
- `json`
- `outputs`
- `reporting`
- `Save`
- `save_json`
- `Saved`
- `Stage`
- `statistical`
- `statistical_test_dataframe`
- `statistical_test_summary`

### Outputs

- `baseline_results`
- `comparison_flag_column`
- `comparison_name`
- `comparison_results`

### Key Operations

- `statistical_test_dataframe, statistical_test_summary = build_statistical_test_summary( baseline_results=baseline_results, comparison_results=cascade_stage3_improved_results, compar`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(statistical_test_dataframe)`: Displays a notebook-facing result for inspection.
- `# Save outputs for Task 3 reporting.`: Documents the purpose or boundary of the surrounding notebook step.
- `#STATISTICAL_TEST_SUMMARY_PATH = (`: Documents the purpose or boundary of the surrounding notebook step.
- `# GOLD_ARTIFACTS_PATH / f"{DATASET_NAME}__gold__statistical_test_summary.json"`: Documents the purpose or boundary of the surrounding notebook step.
- `#)`: Documents the purpose or boundary of the surrounding notebook step.
- `#STATISTICAL_TEST_TABLE_PATH = (`: Documents the purpose or boundary of the surrounding notebook step.
- `# GOLD_ARTIFACTS_PATH / f"{DATASET_NAME}__gold__statistical_test_summary.csv"`: Documents the purpose or boundary of the surrounding notebook step.
- `#)`: Documents the purpose or boundary of the surrounding notebook step.
- `statistical_test_dataframe.to_csv(STATISTICAL_TEST_TABLE_PATH, index=False)`: Writes an artifact or output used for review or downstream notebooks.
- `save_json(statistical_test_summary, STATISTICAL_TEST_SUMMARY_PATH)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `build_statistical_test_summary`
- `display`
- `save_json`
- `to_csv`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `statistical_test_dataframe, statistical_test_summary = build_statistical_test_summary( baseline_results=baseline_results, comparison_results=cascade_stage3_improved_results, compar` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(statistical_test_dataframe)` | Displays a notebook-facing result for inspection. |
| `# Save outputs for Task 3 reporting.` | Documents the purpose or boundary of the surrounding notebook step. |
| `#STATISTICAL_TEST_SUMMARY_PATH = (` | Documents the purpose or boundary of the surrounding notebook step. |
| `# GOLD_ARTIFACTS_PATH / f"{DATASET_NAME}__gold__statistical_test_summary.json"` | Documents the purpose or boundary of the surrounding notebook step. |
| `#)` | Documents the purpose or boundary of the surrounding notebook step. |
| `#STATISTICAL_TEST_TABLE_PATH = (` | Documents the purpose or boundary of the surrounding notebook step. |
| `# GOLD_ARTIFACTS_PATH / f"{DATASET_NAME}__gold__statistical_test_summary.csv"` | Documents the purpose or boundary of the surrounding notebook step. |
| `#)` | Documents the purpose or boundary of the surrounding notebook step. |
| `statistical_test_dataframe.to_csv(STATISTICAL_TEST_TABLE_PATH, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `save_json(statistical_test_summary, STATISTICAL_TEST_SUMMARY_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"Saved statistical test table: {STATISTICAL_TEST_TABLE_PATH}")` | Displays a notebook-facing result for inspection. |
| `print(f"Saved statistical test summary: {STATISTICAL_TEST_SUMMARY_PATH}")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 23 — Visualize Alert Counts and Core Metrics

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold__cascade_stage3_improved_summary`
- `__gold__comparison__2panel_test_alerts_and_metrics`
- `_alert_count_test_rows`
- `_f1`
- `_precision`
- `_recall`
- `a`
- `absent`
- `alert`
- `Alert`
- `alert_count_test_rows`
- `alerts`
- `Alerts`
- `all`
- `Any`
- `apply`
- `arange`
- `are`
- `axes`
- `axis`

### Outputs

- `_is_complete_metric_row`
- `_safe_float`
- `_safe_int`
- `_stage3_alert_count`
- `_stage3_metric`
- `_summary_model_row`
- `bars`
- `candidate_comparison_rows`
- `candidate_keys`
- `comparison_2panel_plot_path`
- `comparison_plot_dataframe`
- `excluded_comparison_rows`
- `fontsize`
- `ha`
- `label`
- `metric_labels`
- `metric_names`
- `metric_prefix`
- `model_count`
- `model_name`

### Key Operations

- `STAGE3_IMPROVED_SUMMARY_PATH = Path( RESOLVED_PATHS.get( "cascade_stage3_improved_summary_path", str(GOLD_ARTIFACTS_PATH / f"{DATASET_NAME}__gold__cascade_stage3_improved_summary.j`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage3_improved_summary: dict[str, Any] = require_mapping( load_json(STAGE3_IMPROVED_SUMMARY_PATH), "stage3_improved_summary",`: Loads input data, configuration, or artifacts required by the current stage.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if "cascade_metrics" in stage3_improved_summary: stage3_improved_metrics = stage3_improved_summary["cascade_metrics"]`: Controls validation, iteration, file handling, or error handling for this step.
- `else: stage3_improved_metrics = stage3_improved_summary`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# =============================================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build comparison dataframe for plotting`: Documents the purpose or boundary of the surrounding notebook step.
- `# =============================================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `def _safe_float(value: object) -> float: """ Convert a metric value to float. Missing or invalid values are returned as NaN so optional model variants do not crash the comparison n`: Defines notebook-local logic used later in the notebook.
- `def _safe_int(value: object) -> float: """ Convert a count value to a numeric plotting value. Float is used temporarily so NaN can represent missing integer-like values. """ if val`: Defines notebook-local logic used later in the notebook.
- `def _is_complete_metric_row(row: pd.Series) -> bool: """ Return True when a model row has all required comparison metrics. """ required_fields = ["test_alerts", "precision", "recal`: Defines notebook-local logic used later in the notebook.

Important functions or methods detected:
- `_is_complete_metric_row`
- `_safe_float`
- `_safe_int`
- `_stage3_alert_count`
- `_stage3_metric`
- `_summary_model_row`
- `apply`
- `arange`
- `bar`
- `cast`
- `copy`
- `DataFrame`
- `display`
- `except`
- `figure`
- `get`
- `get_height`
- `get_width`
- `get_x`
- `isnan`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `STAGE3_IMPROVED_SUMMARY_PATH = Path( RESOLVED_PATHS.get( "cascade_stage3_improved_summary_path", str(GOLD_ARTIFACTS_PATH / f"{DATASET_NAME}__gold__cascade_stage3_improved_summary.j` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage3_improved_summary: dict[str, Any] = require_mapping( load_json(STAGE3_IMPROVED_SUMMARY_PATH), "stage3_improved_summary",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if "cascade_metrics" in stage3_improved_summary: stage3_improved_metrics = stage3_improved_summary["cascade_metrics"]` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: stage3_improved_metrics = stage3_improved_summary` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# =============================================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build comparison dataframe for plotting` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =============================================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `def _safe_float(value: object) -> float: """ Convert a metric value to float. Missing or invalid values are returned as NaN so optional model variants do not crash the comparison n` | Defines notebook-local logic used later in the notebook. |
| `def _safe_int(value: object) -> float: """ Convert a count value to a numeric plotting value. Float is used temporarily so NaN can represent missing integer-like values. """ if val` | Defines notebook-local logic used later in the notebook. |
| `def _is_complete_metric_row(row: pd.Series) -> bool: """ Return True when a model row has all required comparison metrics. """ required_fields = ["test_alerts", "precision", "recal` | Defines notebook-local logic used later in the notebook. |
| `def _summary_model_row( *, model_name: str, metric_prefix: str, summary_dict: dict[str, object],` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, object]: """ Build one model-comparison row from the comparison summary dictionary. """ return { "model": model_name, "test_alerts": _safe_int( summary_dict.get(f"{m` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def _stage3_metric( summary_dict: dict[str, object], metric_name: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> float: """ Resolve a Stage 3 metric from possible summary-key variants. """ candidate_keys = [ metric_name, f"stage3_improved_{metric_name}", f"stage3_{metric_name}", ] for ke` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def _stage3_alert_count( summary_dict: dict[str, object],` | Defines notebook-local logic used later in the notebook. |
| `) -> float: """ Resolve a Stage 3 alert count from possible summary-key variants. """ candidate_keys = [ "final_alert_count_test_rows", "final_cascade_alert_count_test_rows", "aler` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `candidate_comparison_rows = [ _summary_model_row( model_name="Baseline IsolationForest", metric_prefix="baseline", summary_dict=comparison_summary, ), _summary_model_row( model_nam` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `raw_comparison_plot_dataframe = pd.DataFrame(candidate_comparison_rows)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `comparison_plot_dataframe = ( raw_comparison_plot_dataframe[ raw_comparison_plot_dataframe.apply(_is_complete_metric_row, axis=1) ] .copy() .reset_index(drop=True)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `excluded_comparison_rows = ( raw_comparison_plot_dataframe[ ~raw_comparison_plot_dataframe.apply(_is_complete_metric_row, axis=1) ] .copy() .reset_index(drop=True)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Comparison rows included in plots:", len(comparison_plot_dataframe))` | Displays a notebook-facing result for inspection. |
| `print("Comparison rows excluded because metrics were missing:", len(excluded_comparison_rows))` | Displays a notebook-facing result for inspection. |
| `display(comparison_plot_dataframe)` | Displays a notebook-facing result for inspection. |
| `if not excluded_comparison_rows.empty: print("Rows excluded from plots because one or more metrics were missing:") display(excluded_comparison_rows)` | Displays a notebook-facing result for inspection. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Chart 1: Alert Count Comparison on Test Rows` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `plt.figure(figsize=(18, 8))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `bars = plt.bar( comparison_plot_dataframe["model"], comparison_plot_dataframe["test_alerts"],` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.title("Alert Count Comparison on Test Rows")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.ylabel("Alert Count")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.xticks(rotation=20)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for bar, value in zip(bars, comparison_plot_dataframe["test_alerts"]): plt.text( bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,}", ha="center", va="bottom", fontsi` | Controls validation, iteration, file handling, or error handling for this step. |
| `plt.tight_layout()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.show()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Chart 2: Test Alerts + Precision / Recall / F1` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `fig, axes = plt.subplots(1, 2, figsize=(18, 8))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Left panel: test alerts` | Documents the purpose or boundary of the surrounding notebook step. |
| `bars = axes[0].bar( comparison_plot_dataframe["model"], comparison_plot_dataframe["test_alerts"],` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `axes[0].set_title("Test Alert Counts")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `axes[0].set_ylabel("Alerts")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `axes[0].tick_params(axis="x", rotation=0)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for bar, value in zip(bars, comparison_plot_dataframe["test_alerts"]): axes[0].text( bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,}", ha="center", va="bottom", fo` | Controls validation, iteration, file handling, or error handling for this step. |
| `# Right panel: precision / recall / f1` | Documents the purpose or boundary of the surrounding notebook step. |
| `metric_names = ["precision", "recall", "f1"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `metric_labels = ["PRECISION", "RECALL", "F1"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `x = np.arange(len(metric_names))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `model_count = len(comparison_plot_dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `width = 0.11` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for index, row in comparison_plot_dataframe.reset_index(drop=True).iterrows(): offsets = x + (index - (model_count - 1) / 2) * width axes[1].bar( offsets, [row["precision"], row["r` | Controls validation, iteration, file handling, or error handling for this step. |
| `axes[1].set_title("Precision / Recall / F1")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `11 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: plot/image artifact.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 24 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `Alerts`
- `Broad`
- `Build`
- `by`
- `cascade`
- `cascade_summary`
- `Confirmed`
- `DataFrame`
- `def`
- `Final`
- `final_cascade_alert_count_test_rows`
- `funnel`
- `model`
- `model_label`
- `Narrow`
- `saved`
- `stage`
- `Stage`
- `stage1_alert_count_test_rows`

### Outputs

- `build_cascade_funnel_dataframe`

### Key Operations

- `def build_cascade_funnel_dataframe( cascade_summary: dict, *, model_label: str,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """ Build a stage-by-stage cascade funnel from saved cascade summary values. """ return pd.DataFrame( [ { "model": model_label, "stage": "Stage 1 Broad IF", "tes`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `build_cascade_funnel_dataframe`
- `DataFrame`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_cascade_funnel_dataframe( cascade_summary: dict, *, model_label: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Build a stage-by-stage cascade funnel from saved cascade summary values. """ return pd.DataFrame( [ { "model": model_label, "stage": "Stage 1 Broad IF", "tes` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 25 — Build cascade funnel comparison frame

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_cascade_funnel_dataframe`
- `Cascade`
- `cascade_defaults_summary`
- `cascade_stage3_improved_summary`
- `cascade_tuned_summary`
- `concat`
- `Default`
- `Improved`
- `Stage`
- `Tuned`

### Outputs

- `cascade_funnel_dataframe`
- `ignore_index`
- `model_label`

### Key Operations

- `cascade_funnel_dataframe = pd.concat( [ build_cascade_funnel_dataframe( cascade_defaults_summary, model_label="03A Default Cascade", ), build_cascade_funnel_dataframe( cascade_tune`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(cascade_funnel_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_cascade_funnel_dataframe`
- `concat`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `cascade_funnel_dataframe = pd.concat( [ build_cascade_funnel_dataframe( cascade_defaults_summary, model_label="03A Default Cascade", ), build_cascade_funnel_dataframe( cascade_tune` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(cascade_funnel_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 26 — Build cascade funnel comparison frame

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `__cascade_funnel`
- `__gold__`
- `Alert`
- `alpha`
- `ax`
- `axis`
- `bar`
- `bbox_inches`
- `bottom`
- `Cascade`
- `cascade`
- `cascade_funnel_dataframe`
- `center`
- `chart`
- `DATASET_NAME`
- `dpi`
- `f`
- `fig`
- `figsize`

### Outputs

- `bars`
- `funnel_plot_path`
- `ha`
- `model_name`
- `safe_model_name`
- `va`

### Key Operations

- `f`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `bar`
- `get_height`
- `get_width`
- `get_x`
- `grid`
- `groupby`
- `lower`
- `replace`
- `savefig`
- `set_title`
- `set_ylabel`
- `show`
- `strip`
- `subplots`
- `text`
- `tight_layout`
- `xticks`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `f` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: plot/image artifact.

## Code Cell 27 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `__gold__stage3_operating_modes_alerts_vs_f1`
- `Alert`
- `alert_count_test_rows`
- `alpha`
- `annotate`
- `ax`
- `bbox_inches`
- `Burden`
- `chart`
- `comparison_df`
- `copy`
- `DATASET_NAME`
- `dpi`
- `f`
- `f1`
- `F1`
- `fig`
- `figsize`
- `get`

### Outputs

- `ascending`
- `by`
- `label_offsets`
- `offset`
- `s`
- `stage3_mode_dataframe`
- `stage3_mode_plot_path`
- `textcoords`
- `xytext`

### Key Operations

- `stage3_mode_dataframe = comparison_df.loc[ comparison_df["model_id"].isin( [ "stage3_improved", "stage3_relaxed", "stage3_medium", "stage3_strict", ] ), ["model", "alert_count_test`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `].copy()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage3_mode_dataframe = stage3_mode_dataframe.sort_values( by="alert_count_test_rows", ascending=False,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage3_mode_dataframe)`: Displays a notebook-facing result for inspection.
- `fig, ax = plt.subplots(figsize=(9, 6))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ax.scatter( stage3_mode_dataframe["alert_count_test_rows"], stage3_mode_dataframe["f1"], s=120,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `label_offsets = { "Stage 3 Relaxed": (8, 8), "Stage 3 Medium": (8, 18), "Stage 3 Improved": (8, -14), "Stage 3 Strict": (8, -10),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `for _, row in stage3_mode_dataframe.iterrows(): offset = label_offsets.get(row["model"], (6, 6)) ax.annotate( row["model"], (row["alert_count_test_rows"], row["f1"]), textcoords="o`: Controls validation, iteration, file handling, or error handling for this step.
- `ax.set_title("Stage 3 Operating Modes: Alert Burden vs F1")`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `annotate`
- `copy`
- `display`
- `get`
- `grid`
- `isin`
- `iterrows`
- `savefig`
- `scatter`
- `set_title`
- `set_xlabel`
- `set_ylabel`
- `show`
- `sort_values`
- `subplots`
- `tight_layout`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage3_mode_dataframe = comparison_df.loc[ comparison_df["model_id"].isin( [ "stage3_improved", "stage3_relaxed", "stage3_medium", "stage3_strict", ] ), ["model", "alert_count_test` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage3_mode_dataframe = stage3_mode_dataframe.sort_values( by="alert_count_test_rows", ascending=False,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage3_mode_dataframe)` | Displays a notebook-facing result for inspection. |
| `fig, ax = plt.subplots(figsize=(9, 6))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ax.scatter( stage3_mode_dataframe["alert_count_test_rows"], stage3_mode_dataframe["f1"], s=120,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `label_offsets = { "Stage 3 Relaxed": (8, 8), "Stage 3 Medium": (8, 18), "Stage 3 Improved": (8, -14), "Stage 3 Strict": (8, -10),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for _, row in stage3_mode_dataframe.iterrows(): offset = label_offsets.get(row["model"], (6, 6)) ax.annotate( row["model"], (row["alert_count_test_rows"], row["f1"]), textcoords="o` | Controls validation, iteration, file handling, or error handling for this step. |
| `ax.set_title("Stage 3 Operating Modes: Alert Burden vs F1")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ax.set_xlabel("Test Alert Rows")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ax.set_ylabel("F1 Score")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ax.grid(True, linewidth=0.5, alpha=0.4)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.tight_layout()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage3_mode_plot_path = ( GOLD_COMPARISON_ARTIFACT_DIRS["plots"] / f"{DATASET_NAME}__gold__stage3_operating_modes_alerts_vs_f1.png"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `fig.savefig(stage3_mode_plot_path, dpi=200, bbox_inches="tight")` | Writes an artifact or output used for review or downstream notebooks. |
| `plt.show()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"Saved Stage 3 operating-mode chart: {stage3_mode_plot_path}")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: plot/image artifact.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 28 — Build the Comparison Truth Record and Save the Comparison Artifacts

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `append_truth_index`
- `artifact_paths`
- `baseline`
- `baseline_metadata_path`
- `BASELINE_METADATA_PATH`
- `baseline_result_row_count`
- `baseline_results`
- `BASELINE_RESULTS_PATH_CSV`
- `baseline_results_path_csv`
- `baseline_results_path_pickle`
- `BASELINE_RESULTS_PATH_PICKLE`
- `BASELINE_SUMMARY_PATH`
- `baseline_summary_path`
- `baseline_thresholds_path`
- `BASELINE_THRESHOLDS_PATH`
- `BASELINE_TRUTH_HASH`
- `baseline_truth_hash`
- `best_model_by_alert_reduction`
- `best_model_by_f1`

### Outputs

- `column_count`
- `comparison_df`
- `comparison_feature_columns`
- `comparison_meta_columns`
- `comparison_process_run_id`
- `comparison_truth`
- `COMPARISON_TRUTH_HASH`
- `comparison_truth_layer_name`
- `comparison_truth_path`
- `comparison_truth_record`
- `config_profile_value`
- `data`
- `dataset_name`
- `feature_columns`
- `gold_process_run_id_value`
- `kind`
- `layer_name`
- `logger`
- `message`
- `meta_columns`

### Key Operations

- `comparison_summary["gold_truth_hash"] = GOLD_PARENT_TRUTH_HASH`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `comparison_summary["gold_truth_path"] = str(GOLD_TRUTH_PATH)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `comparison_summary["gold_process_run_id"] = gold_truth.get("process_run_id")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `comparison_summary["gold_feature_set_id"] = gold_truth_runtime_facts.get("feature_set_id")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `truth_config_object = globals().get("TRUTH_CONFIG")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `run_mode_value = globals().get("RUN_MODE")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `config_profile_value = globals().get("CONFIG_PROFILE", "default")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `gold_process_run_id_value = globals().get("GOLD_PROCESS_RUN_ID")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if isinstance(truth_config_object, dict): truth_config_snapshot = truth_config_object`: Controls validation, iteration, file handling, or error handling for this step.
- `else: truth_config_snapshot = { "runtime": { "stage": "gold_comparison", "dataset": DATASET_NAME, "mode": run_mode_value, "profile": config_profile_value, } }`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `comparison_truth_layer_name = "gold_comparison"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if isinstance(gold_process_run_id_value, str) and gold_process_run_id_value.strip(): comparison_process_run_id = gold_process_run_id_value`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `add`
- `append_truth_index`
- `build_truth_record`
- `get`
- `globals`
- `identify_feature_columns`
- `identify_meta_columns`
- `initialize_layer_truth`
- `isinstance`
- `make_process_run_id`
- `save`
- `save_json`
- `save_truth_record`
- `sorted`
- `stamp_truth_columns`
- `strip`
- `to_csv`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `comparison_summary["gold_truth_hash"] = GOLD_PARENT_TRUTH_HASH` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_summary["gold_truth_path"] = str(GOLD_TRUTH_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_summary["gold_process_run_id"] = gold_truth.get("process_run_id")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_summary["gold_feature_set_id"] = gold_truth_runtime_facts.get("feature_set_id")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `truth_config_object = globals().get("TRUTH_CONFIG")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `run_mode_value = globals().get("RUN_MODE")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `config_profile_value = globals().get("CONFIG_PROFILE", "default")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_process_run_id_value = globals().get("GOLD_PROCESS_RUN_ID")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if isinstance(truth_config_object, dict): truth_config_snapshot = truth_config_object` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: truth_config_snapshot = { "runtime": { "stage": "gold_comparison", "dataset": DATASET_NAME, "mode": run_mode_value, "profile": config_profile_value, } }` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_truth_layer_name = "gold_comparison"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if isinstance(gold_process_run_id_value, str) and gold_process_run_id_value.strip(): comparison_process_run_id = gold_process_run_id_value` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: comparison_process_run_id = make_process_run_id("gold_comparison_process")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_truth = initialize_layer_truth( truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name=comparison_truth_layer_name, process_run_id=comparison_process_run_id,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_truth = update_truth_section( comparison_truth, "config_snapshot", truth_config_snapshot,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_truth = update_truth_section( comparison_truth, "runtime_facts", { "comparison_row_count": int(len(comparison_df)), "baseline_result_row_count": int(len(baseline_results` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_truth = update_truth_section( comparison_truth, "artifact_paths", { "gold_truth_path": str(GOLD_TRUTH_PATH), "baseline_results_path_csv": str(BASELINE_RESULTS_PATH_CSV),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_meta_columns = sorted( set( identify_meta_columns(comparison_df) + [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode", ] )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_feature_columns = identify_feature_columns(comparison_df)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `comparison_truth_record = build_truth_record( truth_base=comparison_truth, row_count=len(comparison_df), column_count=comparison_df.shape[1] + 3, meta_columns=comparison_meta_colum` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `COMPARISON_TRUTH_HASH = comparison_truth_record["truth_hash"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `comparison_df = stamp_truth_columns( comparison_df, truth_hash=COMPARISON_TRUTH_HASH, parent_truth_hash=GOLD_PARENT_TRUTH_HASH, pipeline_mode=PIPELINE_MODE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_truth_path = save_truth_record( comparison_truth_record, truth_dir=TRUTHS_PATH, dataset_name=DATASET_NAME, layer_name=comparison_truth_layer_name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `append_truth_index( comparison_truth_record, truth_index_path=TRUTH_INDEX_PATH,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_summary["comparison_truth_hash"] = COMPARISON_TRUTH_HASH` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_summary["comparison_truth_path"] = str(comparison_truth_path)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_summary["comparison_process_run_id"] = comparison_process_run_id` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `save_json(comparison_summary, MODEL_COMPARISON_SUMMARY_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb_run_object = globals().get("wandb_run")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if wandb_run_object is not None: wandb_run_object.save(str(MODEL_COMPARISON_PATH)) wandb_run_object.save(str(MODEL_COMPARISON_SUMMARY_PATH)) wandb_run_object.save(str(comparison_tr` | Controls validation, iteration, file handling, or error handling for this step. |
| `ledger.add( kind="step", step="save_comparison_outputs", message="Saved final baseline versus cascade comparison outputs and comparison stage truth record.", data={ "comparison_csv` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output, truth record.

## Code Cell 29 — Finalize the Ledger and Close the Tracking Run

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `BASELINE_VS_CASCADE_PATH`
- `BASELINE_VS_CASCADE_SUMMARY_PATH`
- `comparison`
- `comparison_csv`
- `comparison_ledger_path`
- `comparison_summary`
- `comparison_summary_json`
- `complete`
- `finalize_comparison`
- `finish`
- `Gold`
- `GOLD_ARTIFACTS_PATH`
- `GOLD_COMPARISON_LEDGER_FILE_NAME`
- `ledger`
- `MODEL_COMPARISON_PATH`
- `MODEL_COMPARISON_SUMMARY_PATH`
- `notebook`
- `save`
- `wandb`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `step`

### Key Operations

- `ledger.add( kind="step", step="finalize_comparison", message="Gold comparison notebook complete.", data={ #"comparison_csv": str(BASELINE_VS_CASCADE_PATH), #"comparison_summary_jso`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#comparison_ledger_path = GOLD_ARTIFACTS_PATH / GOLD_COMPARISON_LEDGER_FILE_NAME`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger.write_json(comparison_ledger_path)`: Records or exports ledger information for stage-level traceability.
- `wandb.save(str(comparison_ledger_path))`: Records or exports ledger information for stage-level traceability.
- `wandb_run.finish()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `finish`
- `save`
- `write_json`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `ledger.add( kind="step", step="finalize_comparison", message="Gold comparison notebook complete.", data={ #"comparison_csv": str(BASELINE_VS_CASCADE_PATH), #"comparison_summary_jso` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#comparison_ledger_path = GOLD_ARTIFACTS_PATH / GOLD_COMPARISON_LEDGER_FILE_NAME` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger.write_json(comparison_ledger_path)` | Records or exports ledger information for stage-level traceability. |
| `wandb.save(str(comparison_ledger_path))` | Records or exports ledger information for stage-level traceability. |
| `wandb_run.finish()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: optional experiment tracking call.

## Code Cell 30 — Visualize the Cascade Filtering Funnel

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold__alert_count_comparison_test_rows`
- `Alert`
- `alpha`
- `Any`
- `ax`
- `axis`
- `bar`
- `Baseline`
- `baseline_alert_count_test_rows`
- `bottom`
- `Cascade`
- `cascade_default_alert_count_test_rows`
- `cascade_tuned_alert_count_test_rows`
- `center`
- `Chart`
- `Comparison`
- `Count`
- `DATASET_NAME`
- `Default`
- `dpi`

### Outputs

- `bars`
- `comparison_alert_plot_path`
- `counts`
- `ha`
- `stages`
- `va`

### Key Operations

- `# Funnel Filter Chart`: Documents the purpose or boundary of the surrounding notebook step.
- `model_comparison_summary: dict[str, Any] = require_mapping( load_json(MODEL_COMPARISON_SUMMARY_PATH), "model_comparison_summary",`: Loads input data, configuration, or artifacts required by the current stage.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stages = [ "Baseline", "Cascade Default", "Cascade Tuned", "Stage 3 Improved", "Stage 3 Relaxed", "Stage 3 Medium", "Stage 3 Strict",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `counts = [ int(model_comparison_summary["baseline_alert_count_test_rows"]), int(model_comparison_summary["cascade_default_alert_count_test_rows"]), int(model_comparison_summary["ca`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `fig, ax = plt.subplots(figsize=(11, 5))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `bars = ax.bar(stages, counts)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ax.set_title("Alert Count Comparison on Test Rows")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ax.set_ylabel("Alert Count")`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.4)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `bar`
- `get_height`
- `get_width`
- `get_x`
- `grid`
- `load_json`
- `require_mapping`
- `savefig`
- `set_title`
- `set_ylabel`
- `show`
- `subplots`
- `text`
- `tight_layout`
- `xticks`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Funnel Filter Chart` | Documents the purpose or boundary of the surrounding notebook step. |
| `model_comparison_summary: dict[str, Any] = require_mapping( load_json(MODEL_COMPARISON_SUMMARY_PATH), "model_comparison_summary",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stages = [ "Baseline", "Cascade Default", "Cascade Tuned", "Stage 3 Improved", "Stage 3 Relaxed", "Stage 3 Medium", "Stage 3 Strict",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `counts = [ int(model_comparison_summary["baseline_alert_count_test_rows"]), int(model_comparison_summary["cascade_default_alert_count_test_rows"]), int(model_comparison_summary["ca` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `fig, ax = plt.subplots(figsize=(11, 5))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `bars = ax.bar(stages, counts)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ax.set_title("Alert Count Comparison on Test Rows")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ax.set_ylabel("Alert Count")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.4)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for bar in bars: ax.text( bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height()):,}", ha="center", va="bottom", )` | Controls validation, iteration, file handling, or error handling for this step. |
| `plt.xticks(rotation=20)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.tight_layout()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#comparison_alert_plot_path = GOLD_ARTIFACTS_PATH / f"{DATASET_NAME}__gold__alert_count_comparison_test_rows.png"` | Documents the purpose or boundary of the surrounding notebook step. |
| `comparison_alert_plot_path = ( GOLD_COMPARISON_ARTIFACT_DIRS["plots"] / f"{DATASET_NAME}__gold__alert_count_comparison_test_rows.png"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.savefig(comparison_alert_plot_path, dpi=200)` | Writes an artifact or output used for review or downstream notebooks. |
| `plt.show()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: plot/image artifact.

## Code Cell 31 — Run Final Lineage and Consistency Checks

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `Any`
- `astype`
- `baseline`
- `baseline_truth_hash`
- `BASELINE_TRUTH_HASH`
- `cascade_default_truth_hash`
- `CASCADE_DEFAULTS_TRUTH_HASH`
- `CASCADE_STAGE3_IMPROVED_TRUTH_HASH`
- `cascade_stage3_improved_truth_hash`
- `CASCADE_STAGE3_TRUTH_HASH`
- `CASCADE_TUNED_TRUTH_HASH`
- `cascade_tuned_truth_hash`
- `check`
- `column_name`
- `columns`
- `Comparison`
- `comparison_df`
- `comparison_summary`
- `COMPARISON_TRUTH_HASH`

### Outputs

- `comparison_df_truth_hash_check`
- `comparison_parent_values`
- `missing_comparison_meta_columns`
- `required_comparison_meta_columns`

### Key Operations

- `required_comparison_meta_columns = [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_comparison_meta_columns = [ column_name for column_name in required_comparison_meta_columns if column_name not in comparison_df.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if missing_comparison_meta_columns: raise ValueError( f"comparison_df is missing required lineage columns: {missing_comparison_meta_columns}" )`: Controls validation, iteration, file handling, or error handling for this step.
- `comparison_df_truth_hash_check = extract_truth_hash(comparison_df)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if comparison_df_truth_hash_check is None: raise ValueError("comparison_df does not contain a readable meta__truth_hash value.")`: Controls validation, iteration, file handling, or error handling for this step.
- `if comparison_df_truth_hash_check != COMPARISON_TRUTH_HASH: raise ValueError( "comparison_df truth hash does not match COMPARISON_TRUTH_HASH:\n" f"dataframe={comparison_df_truth_ha`: Controls validation, iteration, file handling, or error handling for this step.
- `comparison_parent_values = comparison_df["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if not comparison_parent_values: raise ValueError("comparison_df is missing populated meta__parent_truth_hash values.")`: Controls validation, iteration, file handling, or error handling for this step.
- `if len(comparison_parent_values) != 1: raise ValueError(f"comparison_df has multiple parent truth hashes: {comparison_parent_values}")`: Controls validation, iteration, file handling, or error handling for this step.
- `if comparison_parent_values[0] != GOLD_PARENT_TRUTH_HASH: raise ValueError( "comparison_df parent truth hash does not match GOLD_PARENT_TRUTH_HASH:\n" f"dataframe_parent={compariso`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `astype`
- `dropna`
- `exists`
- `extract_truth_hash`
- `FileNotFoundError`
- `get`
- `load_json`
- `Path`
- `require_truth_record`
- `tolist`
- `unique`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `required_comparison_meta_columns = [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_comparison_meta_columns = [ column_name for column_name in required_comparison_meta_columns if column_name not in comparison_df.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_comparison_meta_columns: raise ValueError( f"comparison_df is missing required lineage columns: {missing_comparison_meta_columns}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `comparison_df_truth_hash_check = extract_truth_hash(comparison_df)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if comparison_df_truth_hash_check is None: raise ValueError("comparison_df does not contain a readable meta__truth_hash value.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if comparison_df_truth_hash_check != COMPARISON_TRUTH_HASH: raise ValueError( "comparison_df truth hash does not match COMPARISON_TRUTH_HASH:\n" f"dataframe={comparison_df_truth_ha` | Controls validation, iteration, file handling, or error handling for this step. |
| `comparison_parent_values = comparison_df["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not comparison_parent_values: raise ValueError("comparison_df is missing populated meta__parent_truth_hash values.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if len(comparison_parent_values) != 1: raise ValueError(f"comparison_df has multiple parent truth hashes: {comparison_parent_values}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if comparison_parent_values[0] != GOLD_PARENT_TRUTH_HASH: raise ValueError( "comparison_df parent truth hash does not match GOLD_PARENT_TRUTH_HASH:\n" f"dataframe_parent={compariso` | Controls validation, iteration, file handling, or error handling for this step. |
| `if not Path(comparison_truth_path).exists(): raise FileNotFoundError(f"Comparison truth file was not created: {comparison_truth_path}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `loaded_comparison_truth: dict[str, Any] = require_truth_record( load_json(comparison_truth_path), "loaded_comparison_truth",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if loaded_comparison_truth.get("truth_hash") != COMPARISON_TRUTH_HASH: raise ValueError( "Saved Comparison truth file hash does not match COMPARISON_TRUTH_HASH:\n" f"file={loaded_c` | Controls validation, iteration, file handling, or error handling for this step. |
| `if loaded_comparison_truth.get("parent_truth_hash") != GOLD_PARENT_TRUTH_HASH: raise ValueError( "Saved Comparison truth file parent hash does not match GOLD_PARENT_TRUTH_HASH:\n" ` | Controls validation, iteration, file handling, or error handling for this step. |
| `saved_comparison_summary: dict[str, Any] = require_truth_record( load_json(MODEL_COMPARISON_SUMMARY_PATH), "saved_comparison_summary",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if saved_comparison_summary.get("baseline_truth_hash") != BASELINE_TRUTH_HASH: raise ValueError( "comparison_summary baseline_truth_hash does not match BASELINE_TRUTH_HASH:\n" f"su` | Controls validation, iteration, file handling, or error handling for this step. |
| `if saved_comparison_summary.get("cascade_default_truth_hash") != CASCADE_DEFAULTS_TRUTH_HASH: raise ValueError( "comparison_summary cascade_default_truth_hash does not match CASCAD` | Controls validation, iteration, file handling, or error handling for this step. |
| `if saved_comparison_summary.get("cascade_tuned_truth_hash") != CASCADE_TUNED_TRUTH_HASH: raise ValueError( "comparison_summary cascade_tuned_truth_hash does not match CASCADE_TUNED` | Controls validation, iteration, file handling, or error handling for this step. |
| `if saved_comparison_summary.get("cascade_stage3_improved_truth_hash") != CASCADE_STAGE3_IMPROVED_TRUTH_HASH: raise ValueError( "comparison_summary cascade_stage3_improved_truth_has` | Controls validation, iteration, file handling, or error handling for this step. |
| `if saved_comparison_summary.get("gold_truth_hash") != GOLD_PARENT_TRUTH_HASH: raise ValueError( "comparison_summary gold_truth_hash does not match GOLD_PARENT_TRUTH_HASH:\n" f"summ` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Gold Comparison lineage sanity check passed.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 32 — Gold Comparison SQL Write Cell

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `add`
- `alert`
- `Alert`
- `alert_count`
- `alert_count_all_rows`
- `alert_count_test_rows`
- `alerts`
- `Alerts`
- `astype`
- `Available`
- `because`
- `column`
- `column_count`
- `columns`
- `comparison`
- `comparison_df`
- `contain`
- `copy`
- `Count`

### Outputs

- `alert_count_candidates`
- `alert_count_source_column`
- `capstone_schema`
- `data`
- `dataframe`
- `dataset_id`
- `dataset_name`
- `engine`
- `extra`
- `gold_comparison_sql_summary_dataframe`
- `gold04_metric_rows_df`
- `kind`
- `logger`
- `message`
- `metric_column_map`
- `missing_columns`
- `notebook_globals`
- `required_columns`
- `run_id`
- `step`

### Key Operations

- `WRITE_TO_POSTGRES = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if WRITE_TO_POSTGRES: gold04_metric_rows_df = comparison_df.copy() # Keep `model` because the current SQL writer expects it. if "model" not in gold04_metric_rows_df.columns and "Mo`: Writes a logger message for traceability during notebook execution.
- `else: print("Postgres write skipped.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `astype`
- `copy`
- `display`
- `get`
- `globals`
- `info`
- `items`
- `KeyError`
- `next`
- `write_gold_model_comparison_results_sql`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `WRITE_TO_POSTGRES = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if WRITE_TO_POSTGRES: gold04_metric_rows_df = comparison_df.copy() # Keep `model` because the current SQL writer expects it. if "model" not in gold04_metric_rows_df.columns and "Mo` | Writes a logger message for traceability during notebook execution. |
| `else: print("Postgres write skipped.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 33 — Gold Comparison SQL Write Cell

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `alert_count_baseline`
- `alert_count_comparison`
- `baseline_model`
- `BY`
- `comparison_model`
- `created_at_utc`
- `DATASET_ID`
- `dataset_id`
- `DESC`
- `engine`
- `f1_baseline`
- `f1_comparison`
- `gold`
- `LIMIT`
- `model_comparison_results`
- `ORDER`
- `precision_baseline`
- `precision_comparison`
- `read_sql_dataframe`
- `recall_baseline`

### Outputs

- `gold04_check_df`
- `gold04_check_sql`
- `params`

### Key Operations

- `gold04_check_sql = """`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SELECT dataset_id, run_id, baseline_model, comparison_model, alert_count_baseline, alert_count_comparison, precision_baseline, precision_comparison, recall_baseline, recall_compari`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `FROM gold.model_comparison_results`: Imports a dependency or project helper used by later cells.
- `WHERE dataset_id = :dataset_id AND run_id = :run_id`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ORDER BY created_at_utc DESC`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `LIMIT 5;`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `"""`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold04_check_df = read_sql_dataframe( engine, gold04_check_sql, params={ "dataset_id": DATASET_ID, "run_id": RUN_ID, },`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(gold04_check_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `gold04_check_sql = """` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SELECT dataset_id, run_id, baseline_model, comparison_model, alert_count_baseline, alert_count_comparison, precision_baseline, precision_comparison, recall_baseline, recall_compari` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FROM gold.model_comparison_results` | Imports a dependency or project helper used by later cells. |
| `WHERE dataset_id = :dataset_id AND run_id = :run_id` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ORDER BY created_at_utc DESC` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `LIMIT 5;` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `"""` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold04_check_df = read_sql_dataframe( engine, gold04_check_sql, params={ "dataset_id": DATASET_ID, "run_id": RUN_ID, },` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(gold04_check_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

