# Notebook Code Reference: EDA_Notebook_Pump_Gold_02_Baseline_Modeling

Notebook path:

`notebooks/experiments/EDA_Notebook_Pump_Gold_02_Baseline_Modeling.ipynb`

## Notebook Purpose

This notebook trains and evaluates the baseline Isolation Forest model for comparison against cascade models.

Notebook stage:

`Gold`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Baseline Modeling Setup and Imports | Code Cell 01, Code Cell 02 |
| Load Configuration, Paths, and Baseline Runtime Settings | Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08 |
| Review intermediate output | Code Cell 09 |
| Start Logging for the Baseline Modeling Stage | Code Cell 10, Code Cell 11 |
| Initialize Experiment Tracking | Code Cell 12 |
| Initialize the Baseline Ledger | Code Cell 13, Code Cell 14 |
| Load Gold Modeling Inputs and Resolve the Parent Truth | Code Cell 15 |
| Validate Stable Row Identity for Baseline Scoring | Code Cell 16 |
| Build the Train and Test Masks | Code Cell 17 |
| Prepare Labels and Feature Matrices | Code Cell 18 |
| Define Baseline Scoring, Thresholding, and Evaluation Helpers | Code Cell 19 |
| Define percentile threshold selection | Code Cell 20 |
| Define label-based model evaluation | Code Cell 21, Code Cell 25 |
| Fit the Baseline Isolation Forest Model | Code Cell 22 |
| Answer | Code Cell 23, Code Cell 24, Code Cell 29 |
| Define baseline output validation checks | Code Cell 26, Code Cell 27 |
| Build the Baseline Truth Record and Save the Baseline Artifacts | Code Cell 28 |
| Finalize the Ledger and Close the Tracking Run | Code Cell 30 |
| Run Final Lineage and Consistency Checks | Code Cell 31 |
| Gold Baseline SQL Write Cell | Code Cell 32 |

## Code Cell 01 — Baseline Modeling Setup and Imports

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__future__`
- `annotations`
- `Any`
- `append_truth_index`
- `artifact_file_path`
- `artifacts`
- `Artifacts`
- `average_precision_score`
- `build_artifact_dirs_from_config`
- `build_file_fingerprint`
- `build_stage_scoring_frame`
- `build_truth_config_block`
- `build_truth_record`
- `cascade_row_tracking`
- `cast`
- `classification_report`
- `cluster`
- `columns`
- `config_loader`
- `configure_logging`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `from __future__ import annotations`: Imports a dependency or project helper used by later cells.
- `from dataclasses import dataclass, field`: Imports a dependency or project helper used by later cells.
- `from datetime import datetime, timezone`: Imports a dependency or project helper used by later cells.
- `from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Mapping, cast`: Imports a dependency or project helper used by later cells.
- `import math`: Imports a dependency or project helper used by later cells.
- `from pathlib import Path`: Imports a dependency or project helper used by later cells.
- `import yaml`: Imports a dependency or project helper used by later cells.
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
| `import math` | Imports a dependency or project helper used by later cells. |
| `from pathlib import Path` | Imports a dependency or project helper used by later cells. |
| `import yaml` | Imports a dependency or project helper used by later cells. |
| `import os` | Imports a dependency or project helper used by later cells. |
| `import logging` | Imports a dependency or project helper used by later cells. |
| `import wandb` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `import numpy as np` | Imports a dependency or project helper used by later cells. |
| `import joblib` | Imports a dependency or project helper used by later cells. |
| `from sklearn.model_selection import train_test_split, KFold` | Imports a dependency or project helper used by later cells. |
| `from sklearn.preprocessing import ( StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from sklearn.decomposition import PCA` | Imports a dependency or project helper used by later cells. |
| `from sklearn.cluster import KMeans` | Imports a dependency or project helper used by later cells. |
| `from sklearn.ensemble import ( RandomForestClassifier, IsolationForest,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from sklearn.metrics import ( classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from sklearn.svm import OneClassSVM` | Imports a dependency or project helper used by later cells. |
| `from sklearn.neighbors import LocalOutlierFactor` | Imports a dependency or project helper used by later cells. |
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
| `from utils.medallion.gold.cascade_row_tracking import ( ensure_stable_row_id, build_stage_scoring_frame, score_isolation_forest_stage, merge_stage_results_back, finalize_stage_flag` | Imports a dependency or project helper used by later cells. |
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

## Code Cell 02 — Baseline Modeling Setup and Imports

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

## Code Cell 03 — Load Configuration, Paths, and Baseline Runtime Settings

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `aliases`
- `baseline`
- `capstone`
- `context`
- `context_loaded`
- `dataset_config`
- `default`
- `execution`
- `gold`
- `gold_baseline`
- `gold_modeling_baseline`
- `info`
- `load_notebook_context`
- `loaded`
- `Loaded`
- `log`
- `LOG_PATH`
- `log_path`
- `logger`

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
- `CONTEXT_STAGE = "gold_baseline"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_DATASET = "pump"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LAYER = "gold"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_RUN_MODE = "train"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_PROFILE = "default"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LOG_FILE = "gold_modeling_baseline.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.gold.baseline", log_filename=CO`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `CONTEXT_STAGE = "gold_baseline"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LAYER = "gold"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_RUN_MODE = "train"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LOG_FILE = "gold_modeling_baseline.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.gold.baseline", log_filename=CO` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
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

## Code Cell 04 — Load Configuration, Paths, and Baseline Runtime Settings

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `Artifacts`
- `B`
- `Base`
- `Baseline`
- `build_truth_config_block`
- `capstone`
- `cast`
- `CONFIG`
- `CONFIG_RUN_MODE`
- `dataset`
- `DATASET_CFG`
- `details`
- `entity`
- `execution_mode`
- `exist_ok`
- `f`
- `failsafes`
- `File`
- `FILENAMES`

### Outputs

- `ARTIFACTS_ROOT`
- `BASELINE_ESTIMATOR_COUNT`
- `BASELINE_METADATA_FILE_NAME`
- `BASELINE_METADATA_PATH`
- `BASELINE_MODEL_ARTIFACT_PATH`
- `BASELINE_MODEL_FILE_NAME`
- `BASELINE_MODELS_PATH`
- `BASELINE_RESULTS_FILE_NAME_CSV`
- `BASELINE_RESULTS_FILE_NAME_PICKLE`
- `BASELINE_RESULTS_PATH_CSV`
- `BASELINE_RESULTS_PATH_PICKLE`
- `BASELINE_SUMMARY_FILE_NAME`
- `BASELINE_SUMMARY_PATH`
- `BASELINE_THRESHOLD_PERCENTILE`
- `BASELINE_THRESHOLDS_FILE_NAME`
- `BASELINE_THRESHOLDS_PATH`
- `CONFIG_PROFILE`
- `DATASET_NAME`
- `DATASET_NAME_CONFIG`
- `GOLD_ARTIFACTS_PATH`

### Key Operations

- `TRUTH_CONFIG = cast(Dict[str, Any], build_truth_config_block(CONFIG))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_CONFIG["pipeline"] = PIPELINE`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `# ---- Stage details ----`: Documents the purpose or boundary of the surrounding notebook step.
- `STAGE = "gold"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `LAYER_NAME = str(GOLD_CFG["layer_name"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RECIPE_ID = str(GOLD_CFG["recipe_id"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_VERSION = str(VERSIONS_CFG["gold"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_VERSION = str(VERSIONS_CFG["truth"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `RECIPE_ID = str(GOLD_CFG["recipe_id"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_VERSION = str(VERSIONS_CFG["gold"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_VERSION = str(VERSIONS_CFG["truth"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
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
| `GOLD_PREPROCESSED_FILE_NAME = str(FILENAMES["gold_preprocessed_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_PREPROCESSED_SCALED_FILE_NAME = str( FILENAMES["gold_preprocessed_scaled_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_FIT_FILE_NAME = str(FILENAMES["gold_fit_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_TEST_FILE_NAME = str(FILENAMES["gold_test_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_TRAIN_FILE_NAME = str(FILENAMES["gold_train_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE1_FEATURES_FILE_NAME = str(FILENAMES["stage1_features_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BASELINE_RESULTS_FILE_NAME_CSV = str(FILENAMES["baseline_results_file_name_csv"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BASELINE_RESULTS_FILE_NAME_PICKLE = str( FILENAMES["baseline_results_file_name_pickle"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BASELINE_MODEL_FILE_NAME = str(FILENAMES["baseline_model_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BASELINE_THRESHOLDS_FILE_NAME = str(FILENAMES["baseline_thresholds_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BASELINE_SUMMARY_FILE_NAME = str(FILENAMES["baseline_summary_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BASELINE_METADATA_FILE_NAME = str(FILENAMES["baseline_metadata_file_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_BASELINE_LEDGER_FILE_NAME = str( FILENAMES.get("gold_baseline_ledger_file_name", "gold_baseline_ledger.jsonl")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `TRUTH_INDEX_FILE_NAME = "truth_index.jsonl"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Runtime knobs ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `TRAIN_FRACTION = float(GOLD_CFG["train_fraction"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RANDOM_SEED = int(GOLD_CFG["random_seed"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BASELINE_THRESHOLD_PERCENTILE = float(GOLD_CFG["baseline_threshold_percentile"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE1_THRESHOLD_PERCENTILE = float(GOLD_CFG["stage1_threshold_percentile"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE2_THRESHOLD_PERCENTILE = float(GOLD_CFG["stage2_threshold_percentile"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BASELINE_ESTIMATOR_COUNT = int(GOLD_CFG["baseline_estimator_count"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Base paths only ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `ARTIFACTS_ROOT = Path(str(RESOLVED_PATHS["artifacts_root"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_PREPROCESSED_DATA_PATH = Path( str(RESOLVED_PATHS["gold_preprocessed_data_path"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_PREPROCESSED_SCALED_DATA_PATH = Path( str(RESOLVED_PATHS["gold_preprocessed_scaled_data_path"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_TRAIN_DATA_PATH = Path(str(RESOLVED_PATHS["gold_train_data_path"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_TEST_DATA_PATH = Path(str(RESOLVED_PATHS["gold_test_data_path"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `47 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 05 — Load Configuration, Paths, and Baseline Runtime Settings

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

## Code Cell 06 — Load Configuration, Paths, and Baseline Runtime Settings

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

## Code Cell 07 — Load Configuration, Paths, and Baseline Runtime Settings

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold_baseline__resolved_config`
- `artifact`
- `baseline`
- `baseline_metadata_file_name`
- `baseline_model_file_name`
- `BASELINE_MODELS_PATH`
- `baseline_results_file_name_csv`
- `baseline_results_file_name_pickle`
- `baseline_summary_file_name`
- `baseline_thresholds_file_name`
- `build_artifact_dirs_from_config`
- `DATASET_NAME`
- `directories`
- `execution`
- `export_config_snapshot`
- `f`
- `FILENAMES`
- `get`
- `Gold`
- `gold_baseline`

### Outputs

- `baseline_ledger_path`
- `BASELINE_METADATA_PATH`
- `BASELINE_MODEL_ARTIFACT_PATH`
- `BASELINE_RESULTS_PATH_CSV`
- `BASELINE_RESULTS_PATH_PICKLE`
- `BASELINE_SUMMARY_PATH`
- `BASELINE_THRESHOLDS_PATH`
- `config`
- `CONFIG_SNAPSHOT_PATH`
- `GOLD_ARTIFACTS_PATH`
- `GOLD_BASELINE_ARTIFACT_DIRS`
- `GOLD_BASELINE_CONFIG_DIR`
- `GOLD_BASELINE_ROOT`
- `stage_key`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Gold baseline artifact directories`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `GOLD_BASELINE_ARTIFACT_DIRS = build_artifact_dirs_from_config( config=CONFIG, stage_key="gold_baseline",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `GOLD_ARTIFACTS_PATH = GOLD_BASELINE_ARTIFACT_DIRS["stage_dataset_root"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_BASELINE_ROOT = GOLD_BASELINE_ARTIFACT_DIRS["root"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_BASELINE_CONFIG_DIR = GOLD_BASELINE_ARTIFACT_DIRS["config"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_SNAPSHOT_PATH = ( GOLD_BASELINE_CONFIG_DIR / f"{DATASET_NAME}__gold_baseline__resolved_config.yaml"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if CONFIG["execution"].get("save_config_snapshot", True): export_config_snapshot(CONFIG, CONFIG_SNAPSHOT_PATH)`: Controls validation, iteration, file handling, or error handling for this step.
- `BASELINE_MODEL_ARTIFACT_PATH = ( GOLD_BASELINE_ARTIFACT_DIRS["models"] / FILENAMES["baseline_model_file_name"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `build_artifact_dirs_from_config`
- `export_config_snapshot`
- `get`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Gold baseline artifact directories` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `GOLD_BASELINE_ARTIFACT_DIRS = build_artifact_dirs_from_config( config=CONFIG, stage_key="gold_baseline",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_ARTIFACTS_PATH = GOLD_BASELINE_ARTIFACT_DIRS["stage_dataset_root"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_BASELINE_ROOT = GOLD_BASELINE_ARTIFACT_DIRS["root"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_BASELINE_CONFIG_DIR = GOLD_BASELINE_ARTIFACT_DIRS["config"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_SNAPSHOT_PATH = ( GOLD_BASELINE_CONFIG_DIR / f"{DATASET_NAME}__gold_baseline__resolved_config.yaml"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if CONFIG["execution"].get("save_config_snapshot", True): export_config_snapshot(CONFIG, CONFIG_SNAPSHOT_PATH)` | Controls validation, iteration, file handling, or error handling for this step. |
| `BASELINE_MODEL_ARTIFACT_PATH = ( GOLD_BASELINE_ARTIFACT_DIRS["models"] / FILENAMES["baseline_model_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#BASELINE_MODELS_PATH = BASELINE_MODEL_ARTIFACT_PATH` | Documents the purpose or boundary of the surrounding notebook step. |
| `BASELINE_RESULTS_PATH_CSV = ( GOLD_BASELINE_ARTIFACT_DIRS["scores"] / FILENAMES["baseline_results_file_name_csv"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BASELINE_RESULTS_PATH_PICKLE = ( GOLD_BASELINE_ARTIFACT_DIRS["scores"] / FILENAMES["baseline_results_file_name_pickle"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BASELINE_THRESHOLDS_PATH = ( GOLD_BASELINE_ARTIFACT_DIRS["thresholds"] / FILENAMES["baseline_thresholds_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BASELINE_SUMMARY_PATH = ( GOLD_BASELINE_ARTIFACT_DIRS["summaries"] / FILENAMES["baseline_summary_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BASELINE_METADATA_PATH = ( GOLD_BASELINE_ARTIFACT_DIRS["metadata"] / FILENAMES["baseline_metadata_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_ledger_path = ( GOLD_BASELINE_ARTIFACT_DIRS["lineage"] / FILENAMES["gold_baseline_ledger_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 08 — Load Configuration, Paths, and Baseline Runtime Settings

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

## Code Cell 10 — Start Logging for the Baseline Modeling Stage

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

## Code Cell 11 — Start Logging for the Baseline Modeling Stage

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
- `gold_modeling_baseline`
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
- `gold_log_path = paths.logs / "gold_modeling_baseline.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `gold_log_path = paths.logs / "gold_modeling_baseline.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
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
- `BASELINE_THRESHOLD_PERCENTILE`
- `baseline_threshold_percentile`
- `dataset`
- `DATASET_NAME`
- `GOLD_FIT_DATA_PATH`
- `gold_input_path`
- `gold_modeling_baseline`
- `GOLD_VERSION`
- `gold_version`
- `info`
- `init`
- `initialized`
- `logger`
- `s`
- `STAGE`
- `stage`
- `train_fraction`
- `TRAIN_FRACTION`
- `W`

### Outputs

- `config`
- `entity`
- `job_type`
- `name`
- `project`
- `wandb_run`

### Key Operations

- `# W&B`: Documents the purpose or boundary of the surrounding notebook step.
- `wandb_run = wandb.init( project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="gold_modeling_baseline", config={ "gold_version": GOLD_VERSION, "dataset": DATASE`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger.info("W&B initialized: %s", wandb_run.name)`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `info`
- `init`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# W&B` | Documents the purpose or boundary of the surrounding notebook step. |
| `wandb_run = wandb.init( project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="gold_modeling_baseline", config={ "gold_version": GOLD_VERSION, "dataset": DATASE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("W&B initialized: %s", wandb_run.name)` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- Artifact or state outputs detected: optional experiment tracking call.

## Code Cell 13 — Initialize the Baseline Ledger

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

## Code Cell 14 — Initialize the Baseline Ledger

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

## Code Cell 15 — Load Gold Modeling Inputs and Resolve the Parent Truth

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold__truth__`
- `__name__`
- `a`
- `add`
- `Any`
- `artifact`
- `artifact_paths`
- `astype`
- `baseline`
- `be`
- `column`
- `contain`
- `dataset`
- `dictionary`
- `dropna`
- `else`
- `empty`
- `f`
- `features`
- `file`

### Outputs

- `column_name`
- `data`
- `dataframe`
- `DATASET_NAME`
- `dataset_name`
- `GOLD_DATASET_NAME`
- `GOLD_FIT_DATA_PATH`
- `gold_fit_dataframe`
- `GOLD_PARENT_TRUTH_HASH`
- `gold_preprocessed_scaled_dataframe`
- `gold_truth`
- `GOLD_TRUTH_PATH`
- `kind`
- `logger`
- `message`
- `parent_layer_name`
- `PARENT_PIPELINE_MODE`
- `PIPELINE_MODE`
- `raw_gold_truth_artifact_paths`
- `raw_gold_truth_runtime_facts`

### Key Operations

- `logger.info("Loading Gold Preprocessed parquet: %s", GOLD_PREPROCESSED_SCALED_DATA_PATH)`: Writes a logger message for traceability during notebook execution.
- `gold_preprocessed_scaled_dataframe = load_data(GOLD_PREPROCESSED_SCALED_DATA_PATH)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_DATASET_NAME = ( gold_preprocessed_scaled_dataframe["meta__dataset"] .dropna() .astype("string") .str.strip()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `GOLD_DATASET_NAME = GOLD_DATASET_NAME[GOLD_DATASET_NAME != ""]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if len(GOLD_DATASET_NAME) == 0: raise ValueError("Gold baseline input dataframe is missing usable meta__dataset values.")`: Controls validation, iteration, file handling, or error handling for this step.
- `GOLD_DATASET_NAME = str(GOLD_DATASET_NAME.iloc[0]).strip()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `gold_truth = load_parent_truth_record_from_dataframe( dataframe=gold_preprocessed_scaled_dataframe, truth_dir=TRUTHS_PATH, parent_layer_name="gold", dataset_name=GOLD_DATASET_NAME,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `DATASET_NAME = get_dataset_name_from_truth(gold_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_PARENT_TRUTH_HASH = get_truth_hash(gold_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PARENT_PIPELINE_MODE = get_pipeline_mode_from_truth(gold_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `astype`
- `display`
- `dropna`
- `get`
- `get_dataset_name_from_truth`
- `get_pipeline_mode_from_truth`
- `get_truth_hash`
- `head`
- `info`
- `isinstance`
- `load_data`
- `load_json`
- `load_parent_truth_record_from_dataframe`
- `Path`
- `strip`
- `type`
- `TypeError`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `logger.info("Loading Gold Preprocessed parquet: %s", GOLD_PREPROCESSED_SCALED_DATA_PATH)` | Writes a logger message for traceability during notebook execution. |
| `gold_preprocessed_scaled_dataframe = load_data(GOLD_PREPROCESSED_SCALED_DATA_PATH)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_DATASET_NAME = ( gold_preprocessed_scaled_dataframe["meta__dataset"] .dropna() .astype("string") .str.strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_DATASET_NAME = GOLD_DATASET_NAME[GOLD_DATASET_NAME != ""]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(GOLD_DATASET_NAME) == 0: raise ValueError("Gold baseline input dataframe is missing usable meta__dataset values.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `GOLD_DATASET_NAME = str(GOLD_DATASET_NAME.iloc[0]).strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_truth = load_parent_truth_record_from_dataframe( dataframe=gold_preprocessed_scaled_dataframe, truth_dir=TRUTHS_PATH, parent_layer_name="gold", dataset_name=GOLD_DATASET_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DATASET_NAME = get_dataset_name_from_truth(gold_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_PARENT_TRUTH_HASH = get_truth_hash(gold_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PARENT_PIPELINE_MODE = get_pipeline_mode_from_truth(gold_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if PARENT_PIPELINE_MODE is not None: PIPELINE_MODE = PARENT_PIPELINE_MODE` | Controls validation, iteration, file handling, or error handling for this step. |
| `GOLD_TRUTH_PATH = ( TRUTHS_PATH / "gold" / f"{DATASET_NAME}__gold__truth__{GOLD_PARENT_TRUTH_HASH}.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if not isinstance(gold_truth, dict): raise TypeError( "Gold parent truth record must be a dictionary. " f"Got {type(gold_truth).__name__}: {gold_truth!r}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `raw_gold_truth_runtime_facts = gold_truth.get("runtime_facts", {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `raw_gold_truth_artifact_paths = gold_truth.get("artifact_paths", {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_truth_runtime_facts: dict[str, Any] = ( raw_gold_truth_runtime_facts if isinstance(raw_gold_truth_runtime_facts, dict) else {}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth_artifact_paths: dict[str, Any] = ( raw_gold_truth_artifact_paths if isinstance(raw_gold_truth_artifact_paths, dict) else {}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_FIT_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_fit_path", str(GOLD_FIT_DATA_PATH)))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE1_FEATURES_PATH = Path(gold_truth_artifact_paths.get("stage1_features_path", str(STAGE1_FEATURES_PATH)))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Resolved Gold baseline dataset name from Gold truth: %s", DATASET_NAME)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Resolved Gold truth path: %s", GOLD_TRUTH_PATH)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Resolved Gold fit parquet from Gold truth: %s", GOLD_FIT_DATA_PATH)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Resolved Stage 1 features path from Gold truth: %s", STAGE1_FEATURES_PATH)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Loading Gold fit parquet: %s", GOLD_FIT_DATA_PATH)` | Writes a logger message for traceability during notebook execution. |
| `gold_fit_dataframe = load_data(GOLD_FIT_DATA_PATH)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Loading Stage 1 features JSON: %s", STAGE1_FEATURES_PATH)` | Writes a logger message for traceability during notebook execution. |
| `raw_stage1_feature_columns = load_json(STAGE1_FEATURES_PATH)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if raw_stage1_feature_columns is None: raise ValueError(f"Stage 1 features file returned None: {STAGE1_FEATURES_PATH}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if not isinstance(raw_stage1_feature_columns, (list, tuple)): raise TypeError( "Stage 1 features JSON must contain a list of column names. " f"Got {type(raw_stage1_feature_columns)` | Controls validation, iteration, file handling, or error handling for this step. |
| `stage1_feature_columns: list[str] = [ str(column_name).strip() for column_name in raw_stage1_feature_columns if str(column_name).strip()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if not stage1_feature_columns: raise ValueError(f"Stage 1 features list is empty: {STAGE1_FEATURES_PATH}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Gold baseline dataset name from parent truth:", DATASET_NAME)` | Displays a notebook-facing result for inspection. |
| `print("Gold baseline parent truth hash:", GOLD_PARENT_TRUTH_HASH)` | Displays a notebook-facing result for inspection. |
| `ledger.add( kind="step", step="load_modeling_inputs", message="Loaded Gold scaled parquet, loaded Gold truth, substituted truth-linked artifact paths, then loaded baseline inputs."` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(gold_fit_dataframe.head(3))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 16 — Validate Stable Row Identity for Baseline Scoring

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `baseline`
- `bool`
- `dataframe`
- `ensure_stable_row_id`
- `Gold`
- `identity`
- `input`
- `is_unique`
- `ledger`
- `meta__row_id`
- `modeling`
- `on`
- `row`
- `row_count`
- `row_id_unique`
- `stable`
- `validate_baseline_row_tracking`
- `Validated`

### Outputs

- `data`
- `gold_preprocessed_scaled_dataframe`
- `kind`
- `logger`
- `message`
- `row_id_column`
- `step`

### Key Operations

- `gold_preprocessed_scaled_dataframe = ensure_stable_row_id( gold_preprocessed_scaled_dataframe, row_id_column="meta__row_id",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="validate_baseline_row_tracking", message="Validated stable row identity on Gold baseline modeling input dataframe.", data={ "row_id_column": "meta__r`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `bool`
- `ensure_stable_row_id`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `gold_preprocessed_scaled_dataframe = ensure_stable_row_id( gold_preprocessed_scaled_dataframe, row_id_column="meta__row_id",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="validate_baseline_row_tracking", message="Validated stable row identity on Gold baseline modeling input dataframe.", data={ "row_id_column": "meta__r` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 17 — Build the Train and Test Masks

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `all`
- `Anomalies`
- `anomaly_flag`
- `astype`
- `before`
- `bool`
- `columns`
- `counts`
- `d`
- `dtype`
- `exist`
- `fillna`
- `Gold`
- `gold_preprocessed_scaled_dataframe`
- `info`
- `it`
- `loc`
- `logger`
- `Masks`
- `meta__is_train_flag`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `# Masks must exist in scaled parquet.`: Documents the purpose or boundary of the surrounding notebook step.
- `if "meta__is_train_flag" not in gold_preprocessed_scaled_dataframe.columns: raise ValueError( "meta__is_train_flag missing from gold_preprocessed_scaled_dataframe. " "Gold preproce`: Controls validation, iteration, file handling, or error handling for this step.
- `train_mask: pd.Series = ( gold_preprocessed_scaled_dataframe["meta__is_train_flag"] .fillna(False) .astype(bool)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `test_mask: pd.Series = (~train_mask).astype(bool)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `train_mask_array: np.ndarray = train_mask.to_numpy(dtype=bool)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `test_mask_array: np.ndarray = test_mask.to_numpy(dtype=bool)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger.info( "Split counts: all=%d train=%d test=%d", len(train_mask), int(train_mask.sum()), int(test_mask.sum()),`: Writes a logger message for traceability during notebook execution.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns: logger.info( "Anomalies: all=%d test=%d", int( gold_preprocessed_scaled_dataframe["anomaly_flag"] .fillna(0) .astyp`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `astype`
- `fillna`
- `info`
- `sum`
- `to_numpy`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Masks must exist in scaled parquet.` | Documents the purpose or boundary of the surrounding notebook step. |
| `if "meta__is_train_flag" not in gold_preprocessed_scaled_dataframe.columns: raise ValueError( "meta__is_train_flag missing from gold_preprocessed_scaled_dataframe. " "Gold preproce` | Controls validation, iteration, file handling, or error handling for this step. |
| `train_mask: pd.Series = ( gold_preprocessed_scaled_dataframe["meta__is_train_flag"] .fillna(False) .astype(bool)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `test_mask: pd.Series = (~train_mask).astype(bool)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `train_mask_array: np.ndarray = train_mask.to_numpy(dtype=bool)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `test_mask_array: np.ndarray = test_mask.to_numpy(dtype=bool)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info( "Split counts: all=%d train=%d test=%d", len(train_mask), int(train_mask.sum()), int(test_mask.sum()),` | Writes a logger message for traceability during notebook execution. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns: logger.info( "Anomalies: all=%d test=%d", int( gold_preprocessed_scaled_dataframe["anomaly_flag"] .fillna(0) .astyp` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 18 — Prepare Labels and Feature Matrices

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_flag`
- `are`
- `arrays`
- `astype`
- `baseline_all_features`
- `baseline_test_features`
- `baseline_train_fit_features`
- `Build`
- `but`
- `column_name`
- `columns`
- `copy`
- `DataFrame`
- `DataFrames`
- `dtype`
- `f`
- `feature`
- `fillna`
- `fitted`
- `gold_fit_dataframe`

### Outputs

- `all_labels`
- `missing_fit_features`
- `missing_stage1_features`
- `test_labels`

### Key Operations

- `all_labels: np.ndarray \| None = None`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `test_labels: np.ndarray \| None = None`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns: all_labels = ( gold_preprocessed_scaled_dataframe["anomaly_flag"] .fillna(0) .astype(int) .to_numpy(dtype=int) ) te`: Controls validation, iteration, file handling, or error handling for this step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build model feature matrices`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Keep these as DataFrames, not NumPy arrays.`: Documents the purpose or boundary of the surrounding notebook step.
- `# This prevents the sklearn warning:`: Documents the purpose or boundary of the surrounding notebook step.
- `# "X has feature names, but IsolationForest was fitted without feature names"`: Documents the purpose or boundary of the surrounding notebook step.
- `missing_stage1_features = [ column_name for column_name in stage1_feature_columns if column_name not in gold_preprocessed_scaled_dataframe.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if missing_stage1_features: raise ValueError( "Stage 1 feature columns are missing from gold_preprocessed_scaled_dataframe:\n" f"{missing_stage1_features[:25]}" )`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `astype`
- `copy`
- `fillna`
- `to_numpy`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `all_labels: np.ndarray \| None = None` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `test_labels: np.ndarray \| None = None` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns: all_labels = ( gold_preprocessed_scaled_dataframe["anomaly_flag"] .fillna(0) .astype(int) .to_numpy(dtype=int) ) te` | Controls validation, iteration, file handling, or error handling for this step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build model feature matrices` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Keep these as DataFrames, not NumPy arrays.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# This prevents the sklearn warning:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# "X has feature names, but IsolationForest was fitted without feature names"` | Documents the purpose or boundary of the surrounding notebook step. |
| `missing_stage1_features = [ column_name for column_name in stage1_feature_columns if column_name not in gold_preprocessed_scaled_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_stage1_features: raise ValueError( "Stage 1 feature columns are missing from gold_preprocessed_scaled_dataframe:\n" f"{missing_stage1_features[:25]}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `missing_fit_features = [ column_name for column_name in stage1_feature_columns if column_name not in gold_fit_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_fit_features: raise ValueError( "Stage 1 feature columns are missing from gold_fit_dataframe:\n" f"{missing_fit_features[:25]}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `baseline_train_fit_features: pd.DataFrame = gold_fit_dataframe.loc[ :, stage1_feature_columns,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_all_features: pd.DataFrame = gold_preprocessed_scaled_dataframe.loc[ :, stage1_feature_columns,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_test_features: pd.DataFrame = gold_preprocessed_scaled_dataframe.loc[ test_mask, stage1_feature_columns,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 19 — Define Baseline Scoring, Thresholding, and Evaluation Helpers

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `anomalous`
- `anomaly`
- `array`
- `be`
- `but`
- `can`
- `Compute`
- `DataFrame`
- `DataFrames`
- `def`
- `feature`
- `feature_matrix`
- `fitted`
- `Higher`
- `IsolationForest`
- `keeps`
- `mean`
- `model`
- `more`

### Outputs

- `anomaly_scores`
- `compute_anomaly_scores_isolation_forest`
- `scores`

### Key Operations

- `def compute_anomaly_scores_isolation_forest( model: IsolationForest, feature_matrix: pd.DataFrame \| np.ndarray,`: Defines notebook-local logic used later in the notebook.
- `) -> np.ndarray: """ Compute anomaly scores from a fitted IsolationForest model. Higher returned values mean more anomalous. Notes ----- The feature_matrix can be a DataFrame or Nu`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `compute_anomaly_scores_isolation_forest`
- `score_samples`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def compute_anomaly_scores_isolation_forest( model: IsolationForest, feature_matrix: pd.DataFrame \| np.ndarray,` | Defines notebook-local logic used later in the notebook. |
| `) -> np.ndarray: """ Compute anomaly scores from a fitted IsolationForest model. Higher returned values mean more anomalous. Notes ----- The feature_matrix can be a DataFrame or Nu` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 20 — Define percentile threshold selection

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_scores`
- `def`
- `ndarray`
- `percentile`

### Outputs

- `choose_threshold_by_percentile`

### Key Operations

- `def choose_threshold_by_percentile( anomaly_scores: np.ndarray, percentile: float,`: Defines notebook-local logic used later in the notebook.
- `) -> float: return float(np.percentile(anomaly_scores, percentile))`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `choose_threshold_by_percentile`
- `percentile`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def choose_threshold_by_percentile( anomaly_scores: np.ndarray, percentile: float,` | Defines notebook-local logic used later in the notebook. |
| `) -> float: return float(np.percentile(anomaly_scores, percentile))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 21 — Define label-based model evaluation

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `anomaly_scores`
- `anomaly_scores_array`
- `asarray`
- `astype`
- `average_precision_score`
- `binary`
- `def`
- `dtype`
- `else`
- `f1`
- `ndarray`
- `pr_auc`
- `precision`
- `precision_recall_fscore_support`
- `recall`
- `results`
- `roc_auc`
- `roc_auc_score`
- `threshold`

### Outputs

- `average`
- `evaluate_against_labels`
- `predicted_labels`
- `zero_division`

### Key Operations

- `def evaluate_against_labels( true_labels: np.ndarray, anomaly_scores: np.ndarray, threshold: float,`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, float \| None]: true_labels_array: np.ndarray = np.asarray(true_labels, dtype=int) anomaly_scores_array: np.ndarray = np.asarray(anomaly_scores, dtype=float) predicte`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `asarray`
- `astype`
- `average_precision_score`
- `evaluate_against_labels`
- `precision_recall_fscore_support`
- `roc_auc_score`
- `unique`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def evaluate_against_labels( true_labels: np.ndarray, anomaly_scores: np.ndarray, threshold: float,` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, float \| None]: true_labels_array: np.ndarray = np.asarray(true_labels, dtype=int) anomaly_scores_array: np.ndarray = np.asarray(anomaly_scores, dtype=float) predicte` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 22 — Fit the Baseline Isolation Forest Model

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BASELINE_ESTIMATOR_COUNT`
- `baseline_train_fit_features`
- `fit`
- `IsolationForest`
- `RANDOM_SEED`

### Outputs

- `baseline_model`
- `n_estimators`
- `n_jobs`
- `random_state`

### Key Operations

- `baseline_model = IsolationForest( n_estimators=BASELINE_ESTIMATOR_COUNT, random_state=RANDOM_SEED, n_jobs=-1,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `baseline_model.fit(baseline_train_fit_features)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `fit`
- `IsolationForest`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `baseline_model = IsolationForest( n_estimators=BASELINE_ESTIMATOR_COUNT, random_state=RANDOM_SEED, n_jobs=-1,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_model.fit(baseline_train_fit_features)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 23 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `alert_count_all_rows`
- `alert_count_test_rows`
- `all`
- `all_rows`
- `Any`
- `asarray`
- `astype`
- `Baseline`
- `baseline`
- `baseline_all_features`
- `baseline_all_scores`
- `BASELINE_ESTIMATOR_COUNT`
- `baseline_flag`
- `baseline_flags`
- `baseline_metrics`
- `baseline_model`
- `baseline_score`
- `baseline_test_flags`
- `baseline_test_scores`

### Outputs

- `baseline_results`
- `baseline_threshold`
- `data`
- `kind`
- `logger`
- `message`
- `step`

### Key Operations

- `# Score normal-only training rows.`: Documents the purpose or boundary of the surrounding notebook step.
- `baseline_train_scores: np.ndarray = compute_anomaly_scores_isolation_forest( baseline_model, baseline_train_fit_features,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Score all rows.`: Documents the purpose or boundary of the surrounding notebook step.
- `baseline_all_scores: np.ndarray = compute_anomaly_scores_isolation_forest( baseline_model, baseline_all_features,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if len(baseline_all_scores) != len(gold_preprocessed_scaled_dataframe): raise ValueError( "Score length mismatch vs all-rows dataframe. Check feature matrix source." )`: Controls validation, iteration, file handling, or error handling for this step.
- `baseline_threshold = choose_threshold_by_percentile( baseline_train_scores, BASELINE_THRESHOLD_PERCENTILE,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `baseline_flags: np.ndarray = ( baseline_all_scores >= baseline_threshold`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `).astype(int)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `baseline_test_scores: np.ndarray = baseline_all_scores[test_mask_array]`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `asarray`
- `astype`
- `choose_threshold_by_percentile`
- `compute_anomaly_scores_isolation_forest`
- `copy`
- `display`
- `evaluate_against_labels`
- `get`
- `sum`
- `update`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Score normal-only training rows.` | Documents the purpose or boundary of the surrounding notebook step. |
| `baseline_train_scores: np.ndarray = compute_anomaly_scores_isolation_forest( baseline_model, baseline_train_fit_features,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Score all rows.` | Documents the purpose or boundary of the surrounding notebook step. |
| `baseline_all_scores: np.ndarray = compute_anomaly_scores_isolation_forest( baseline_model, baseline_all_features,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if len(baseline_all_scores) != len(gold_preprocessed_scaled_dataframe): raise ValueError( "Score length mismatch vs all-rows dataframe. Check feature matrix source." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `baseline_threshold = choose_threshold_by_percentile( baseline_train_scores, BASELINE_THRESHOLD_PERCENTILE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_flags: np.ndarray = ( baseline_all_scores >= baseline_threshold` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `).astype(int)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_test_scores: np.ndarray = baseline_all_scores[test_mask_array]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_test_flags: np.ndarray = baseline_flags[test_mask_array]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_results = gold_preprocessed_scaled_dataframe.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `baseline_results["baseline_score"] = baseline_all_scores` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_results["baseline_flag"] = baseline_flags` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_metrics: dict[str, Any] = { "model": "Baseline IsolationForest", "threshold_percentile": float(BASELINE_THRESHOLD_PERCENTILE), "threshold": float(baseline_threshold), "ale` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if test_labels is not None: test_labels_array: np.ndarray = np.asarray(test_labels, dtype=int) baseline_metrics.update( evaluate_against_labels( test_labels_array, baseline_test_sc` | Controls validation, iteration, file handling, or error handling for this step. |
| `ledger.add( kind="step", step="run_baseline_isolation_forest", message="Ran baseline Isolation Forest fit on normal-only rows and scored the full scaled Gold dataset; evaluated on ` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(baseline_metrics)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 24 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_x`
- `_y`
- `add`
- `anomalous`
- `anomaly`
- `are`
- `astype`
- `Avoid`
- `Baseline`
- `baseline`
- `baseline_alert_count_all_rows`
- `baseline_alert_count_test_rows`
- `baseline_all_features`
- `baseline_decision`
- `baseline_feature_columns`
- `baseline_flag`
- `baseline_flag_count`
- `baseline_flag_count_test_rows`
- `baseline_model`
- `baseline_pred`

### Outputs

- `baseline_all_decisions`
- `baseline_all_preds`
- `baseline_all_scores`
- `baseline_row_tracking_columns`
- `baseline_stage_input_df`
- `baseline_stage_results_df`
- `columns`
- `data`
- `dataframe`
- `errors`
- `feature_columns`
- `how`
- `kind`
- `logger`
- `mask`
- `message`
- `missing_baseline_feature_columns`
- `model`
- `on`
- `row_id_column`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Score baseline with row tracking`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Important:`: Documents the purpose or boundary of the surrounding notebook step.
- `# score_isolation_forest_stage() returns model.score_samples() as the helper`: Documents the purpose or boundary of the surrounding notebook step.
- `# score column. For IsolationForest, score_samples is higher for more normal`: Documents the purpose or boundary of the surrounding notebook step.
- `# rows and lower for more anomalous rows.`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# The project baseline score uses the opposite direction:`: Documents the purpose or boundary of the surrounding notebook step.
- `# baseline_score = -model.score_samples(...)`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# So this cell preserves the helper's raw score_samples output separately`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `add`
- `astype`
- `build_stage_scoring_frame`
- `compute_anomaly_scores_isolation_forest`
- `decision_function`
- `drop`
- `globals`
- `max`
- `median`
- `merge`
- `min`
- `predict`
- `rename`
- `score_isolation_forest_stage`
- `score_samples`
- `sum`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Score baseline with row tracking` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Important:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# score_isolation_forest_stage() returns model.score_samples() as the helper` | Documents the purpose or boundary of the surrounding notebook step. |
| `# score column. For IsolationForest, score_samples is higher for more normal` | Documents the purpose or boundary of the surrounding notebook step. |
| `# rows and lower for more anomalous rows.` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The project baseline score uses the opposite direction:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# baseline_score = -model.score_samples(...)` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# So this cell preserves the helper's raw score_samples output separately` | Documents the purpose or boundary of the surrounding notebook step. |
| `# and restores baseline_score to the project-defined anomaly score.` | Documents the purpose or boundary of the surrounding notebook step. |
| `source_gold_dataframe: pd.DataFrame = gold_preprocessed_scaled_dataframe` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_feature_columns: list[str] = list(stage1_feature_columns)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_baseline_feature_columns = [ column_name for column_name in baseline_feature_columns if column_name not in source_gold_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_baseline_feature_columns: raise ValueError( "Some baseline feature columns are missing from gold_preprocessed_scaled_dataframe:\n" f"{missing_baseline_feature_columns[:2` | Controls validation, iteration, file handling, or error handling for this step. |
| `baseline_stage_input_df = build_stage_scoring_frame( dataframe=source_gold_dataframe, feature_columns=baseline_feature_columns, mask=None, row_id_column="meta__row_id",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_stage_results_df = score_isolation_forest_stage( stage_dataframe=baseline_stage_input_df, model=baseline_model, feature_columns=baseline_feature_columns, stage_name="basel` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Rename the helper outputs so the raw IsolationForest score does not replace` | Documents the purpose or boundary of the surrounding notebook step. |
| `# the project-level anomaly score.` | Documents the purpose or boundary of the surrounding notebook step. |
| `baseline_stage_results_df = baseline_stage_results_df.rename( columns={ "baseline_score": "baseline_score_samples_raw", "baseline_flag": "baseline_predict_flag", }` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_row_tracking_columns = [ "meta__row_id", "baseline_score_samples_raw", "baseline_decision", "baseline_pred", "baseline_predict_flag",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_row_tracking_columns = [ column_name for column_name in baseline_row_tracking_columns if column_name in baseline_stage_results_df.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Drop old baseline columns first so rerunning this cell does not create` | Documents the purpose or boundary of the surrounding notebook step. |
| `# duplicate _x / _y columns.` | Documents the purpose or boundary of the surrounding notebook step. |
| `scored_gold_dataframe: pd.DataFrame = source_gold_dataframe.drop( columns=[ "baseline_score_samples_raw", "baseline_decision", "baseline_pred", "baseline_predict_flag", "baseline_s` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `scored_gold_dataframe = scored_gold_dataframe.merge( baseline_stage_results_df[baseline_row_tracking_columns], on="meta__row_id", how="left",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Recompute the canonical project anomaly score directly from the model.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Higher values mean more anomalous.` | Documents the purpose or boundary of the surrounding notebook step. |
| `baseline_all_scores = compute_anomaly_scores_isolation_forest( baseline_model, baseline_all_features,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_all_decisions = baseline_model.decision_function(baseline_all_features)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `baseline_all_preds = baseline_model.predict(baseline_all_features)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(baseline_all_scores) != len(scored_gold_dataframe): raise ValueError( "baseline_all_scores length does not match gold_preprocessed_scaled_dataframe." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `scored_gold_dataframe["baseline_score"] = baseline_all_scores` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `scored_gold_dataframe["baseline_decision"] = baseline_all_decisions` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `scored_gold_dataframe["baseline_pred"] = baseline_all_preds` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `scored_gold_dataframe["baseline_flag"] = ( scored_gold_dataframe["baseline_score"] >= baseline_threshold` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `).astype(int)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Avoid Pylance's notebook-cell unbound-variable warning while still updating` | Documents the purpose or boundary of the surrounding notebook step. |
| `# the shared notebook variable for later cells.` | Documents the purpose or boundary of the surrounding notebook step. |
| `globals()["gold_preprocessed_scaled_dataframe"] = scored_gold_dataframe` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="score_baseline_with_row_tracking", message="Scored the full Gold dataframe with baseline row-level tracking and preserved project-defined anomaly sco` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Baseline row-tracked scoring complete.")` | Displays a notebook-facing result for inspection. |
| `print( { "baseline_threshold": float(baseline_threshold), "baseline_alert_count_all_rows": int(scored_gold_dataframe["baseline_flag"].sum()), "baseline_alert_count_test_rows": int(` | Displays a notebook-facing result for inspection. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 25 — Define label-based model evaluation

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `actual`
- `add`
- `after`
- `alert`
- `alert_count_all_rows`
- `alert_count_test_rows`
- `alerts`
- `anomaly`
- `asarray`
- `astype`
- `Baseline`
- `baseline_all_scores`
- `baseline_flag`
- `baseline_score`
- `baseline_threshold`
- `baseline_threshold_percentile`
- `BASELINE_THRESHOLD_PERCENTILE`
- `copy`
- `defined`
- `direction`

### Outputs

- `baseline_alert_count_all_rows`
- `baseline_alert_count_test_rows`
- `baseline_metrics`
- `baseline_results`
- `baseline_test_scores`
- `data`
- `kind`
- `logger`
- `message`
- `step`
- `test_labels_array`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Synchronize baseline_results after row-tracked scoring`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `baseline_results = gold_preprocessed_scaled_dataframe.copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Keep the project-defined percentile threshold as the actual alert rule.`: Documents the purpose or boundary of the surrounding notebook step.
- `baseline_results["baseline_score"] = baseline_all_scores`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `baseline_results["baseline_threshold"] = float(baseline_threshold)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `baseline_results["baseline_threshold_percentile"] = float(BASELINE_THRESHOLD_PERCENTILE)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `baseline_results["baseline_flag"] = ( baseline_results["baseline_score"] >= baseline_threshold`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `).astype(int)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `baseline_alert_count_all_rows = int(baseline_results["baseline_flag"].sum())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `baseline_alert_count_test_rows = int( baseline_results.loc[test_mask, "baseline_flag"].sum()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `asarray`
- `astype`
- `copy`
- `direction`
- `display`
- `evaluate_against_labels`
- `get`
- `isinstance`
- `sum`
- `to_numpy`
- `update`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Synchronize baseline_results after row-tracked scoring` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `baseline_results = gold_preprocessed_scaled_dataframe.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Keep the project-defined percentile threshold as the actual alert rule.` | Documents the purpose or boundary of the surrounding notebook step. |
| `baseline_results["baseline_score"] = baseline_all_scores` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_results["baseline_threshold"] = float(baseline_threshold)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_results["baseline_threshold_percentile"] = float(BASELINE_THRESHOLD_PERCENTILE)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_results["baseline_flag"] = ( baseline_results["baseline_score"] >= baseline_threshold` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `).astype(int)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_alert_count_all_rows = int(baseline_results["baseline_flag"].sum())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `baseline_alert_count_test_rows = int( baseline_results.loc[test_mask, "baseline_flag"].sum()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if not isinstance(baseline_metrics, dict): baseline_metrics = {}` | Controls validation, iteration, file handling, or error handling for this step. |
| `if test_labels is not None: test_labels_array = np.asarray(test_labels, dtype=int) baseline_test_scores = baseline_results.loc[ test_mask, "baseline_score", ].to_numpy(dtype=float)` | Controls validation, iteration, file handling, or error handling for this step. |
| `if baseline_alert_count_all_rows == 0: raise ValueError( "Baseline produced zero alerts after synchronization. " "This usually means baseline_score is using raw score_samples inste` | Controls validation, iteration, file handling, or error handling for this step. |
| `baseline_metrics["alert_count_all_rows"] = baseline_alert_count_all_rows` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_metrics["alert_count_test_rows"] = baseline_alert_count_test_rows` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="synchronize_baseline_results_after_row_tracking", message="Synchronized baseline_results after row-tracked scoring and reapplied the percentile-thres` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(baseline_metrics)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 26 — Define baseline output validation checks

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `alert_count_all_rows`
- `alert_count_test_rows`
- `all_rows`
- `anomaly_flag`
- `any`
- `astype`
- `baseline`
- `Baseline`
- `baseline_flag`
- `baseline_score`
- `binary`
- `column`
- `column_name`
- `columns`
- `confusion_matrix`
- `contains`
- `copy`
- `DataFrame`
- `dataframe`

### Outputs

- `average`
- `invalid_flags`
- `labels`
- `required_columns`
- `test_dataframe`
- `validate_baseline_output`
- `validation_summary`
- `y_pred`
- `y_true`
- `zero_division`

### Key Operations

- `def validate_baseline_output( results_dataframe: pd.DataFrame, *, test_mask: pd.Series, label_column: str = "anomaly_flag", flag_column: str = "baseline_flag", score_column: str = `: Defines notebook-local logic used later in the notebook.
- `) -> dict: required_columns = [ row_id_column, flag_column, score_column, "meta__is_train_flag", ] for column_name in required_columns: if column_name not in results_dataframe.colu`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `any`
- `astype`
- `confusion_matrix`
- `copy`
- `dropna`
- `fillna`
- `isna`
- `nunique`
- `precision_recall_fscore_support`
- `ravel`
- `sorted`
- `sum`
- `to_numpy`
- `unique`
- `update`
- `validate_baseline_output`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def validate_baseline_output( results_dataframe: pd.DataFrame, *, test_mask: pd.Series, label_column: str = "anomaly_flag", flag_column: str = "baseline_flag", score_column: str = ` | Defines notebook-local logic used later in the notebook. |
| `) -> dict: required_columns = [ row_id_column, flag_column, score_column, "meta__is_train_flag", ] for column_name in required_columns: if column_name not in results_dataframe.colu` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 27 — Define baseline output validation checks

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `baseline`
- `baseline_results`
- `DataFrame`
- `held`
- `ledger`
- `on`
- `out`
- `output`
- `rows`
- `scored`
- `test`
- `validate_baseline_output`
- `Validated`

### Outputs

- `baseline_output_validation`
- `data`
- `kind`
- `logger`
- `message`
- `step`
- `test_mask`

### Key Operations

- `baseline_output_validation = validate_baseline_output( baseline_results, test_mask=test_mask,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="validate_baseline_output", message="Validated baseline scored output on held-out test rows.", data=baseline_output_validation, logger=logger,`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(pd.DataFrame([baseline_output_validation]))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `DataFrame`
- `display`
- `validate_baseline_output`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `baseline_output_validation = validate_baseline_output( baseline_results, test_mask=test_mask,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="validate_baseline_output", message="Validated baseline scored output on held-out test rows.", data=baseline_output_validation, logger=logger,` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(pd.DataFrame([baseline_output_validation]))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 28 — Build the Baseline Truth Record and Save the Baseline Artifacts

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `alert_count_all_rows`
- `alert_count_test_rows`
- `append_truth_index`
- `artifact_paths`
- `baseline`
- `baseline_estimator_count`
- `BASELINE_ESTIMATOR_COUNT`
- `baseline_flag`
- `BASELINE_METADATA_PATH`
- `baseline_metadata_path`
- `baseline_metrics`
- `baseline_model`
- `baseline_model_artifact_path`
- `BASELINE_MODEL_ARTIFACT_PATH`
- `BASELINE_MODELS_PATH`
- `baseline_models_path`
- `baseline_results_path_csv`
- `BASELINE_RESULTS_PATH_CSV`
- `BASELINE_RESULTS_PATH_PICKLE`

### Outputs

- `baseline_alert_count_all_rows`
- `baseline_alert_count_test_rows`
- `baseline_feature_columns`
- `baseline_meta_columns`
- `baseline_metadata`
- `baseline_process_run_id`
- `baseline_results`
- `baseline_summary`
- `baseline_thresholds`
- `baseline_truth`
- `BASELINE_TRUTH_HASH`
- `baseline_truth_layer_name`
- `baseline_truth_path`
- `baseline_truth_record`
- `column_count`
- `config_profile_value`
- `data`
- `dataset_name`
- `feature_columns`
- `gold_process_run_id_value`

### Key Operations

- `baseline_alert_count_all_rows = int(baseline_results["baseline_flag"].sum())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `baseline_alert_count_test_rows = int(baseline_results.loc[test_mask, "baseline_flag"].sum())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `baseline_thresholds = { "baseline_threshold_percentile": float(BASELINE_THRESHOLD_PERCENTILE), "baseline_threshold": float(baseline_threshold),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `baseline_summary = { "dataset_name": DATASET_NAME, "baseline_metrics": baseline_metrics, "alert_count_all_rows": baseline_alert_count_all_rows, "alert_count_test_rows": baseline_al`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `truth_config_object = globals().get("TRUTH_CONFIG")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `run_mode_value = globals().get("RUN_MODE")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `config_profile_value = globals().get("CONFIG_PROFILE", "default")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `gold_process_run_id_value = globals().get("GOLD_PROCESS_RUN_ID")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if isinstance(truth_config_object, dict): truth_config_snapshot = truth_config_object`: Controls validation, iteration, file handling, or error handling for this step.
- `else: truth_config_snapshot = { "runtime": { "stage": "gold_baseline", "dataset": DATASET_NAME, "mode": run_mode_value, "profile": config_profile_value, } }`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `append_truth_index`
- `build_truth_record`
- `dump`
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
- `sum`
- `to_csv`
- `to_pickle`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `baseline_alert_count_all_rows = int(baseline_results["baseline_flag"].sum())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `baseline_alert_count_test_rows = int(baseline_results.loc[test_mask, "baseline_flag"].sum())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `baseline_thresholds = { "baseline_threshold_percentile": float(BASELINE_THRESHOLD_PERCENTILE), "baseline_threshold": float(baseline_threshold),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_summary = { "dataset_name": DATASET_NAME, "baseline_metrics": baseline_metrics, "alert_count_all_rows": baseline_alert_count_all_rows, "alert_count_test_rows": baseline_al` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `truth_config_object = globals().get("TRUTH_CONFIG")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `run_mode_value = globals().get("RUN_MODE")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `config_profile_value = globals().get("CONFIG_PROFILE", "default")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_process_run_id_value = globals().get("GOLD_PROCESS_RUN_ID")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if isinstance(truth_config_object, dict): truth_config_snapshot = truth_config_object` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: truth_config_snapshot = { "runtime": { "stage": "gold_baseline", "dataset": DATASET_NAME, "mode": run_mode_value, "profile": config_profile_value, } }` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_truth_layer_name = "gold_baseline"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if isinstance(gold_process_run_id_value, str) and gold_process_run_id_value.strip(): baseline_process_run_id = gold_process_run_id_value` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: baseline_process_run_id = make_process_run_id("gold_baseline_process")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_truth = initialize_layer_truth( truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name=baseline_truth_layer_name, process_run_id=baseline_process_run_id, pipel` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_truth = update_truth_section( baseline_truth, "config_snapshot", truth_config_snapshot,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_truth = update_truth_section( baseline_truth, "runtime_facts", { "baseline_threshold_percentile": float(BASELINE_THRESHOLD_PERCENTILE), "baseline_threshold": float(baselin` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_truth = update_truth_section( baseline_truth, "artifact_paths", { "gold_truth_path": str(GOLD_TRUTH_PATH), "gold_preprocessed_scaled_path": str(GOLD_PREPROCESSED_SCALED_DA` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_meta_columns = sorted( set( identify_meta_columns(baseline_results) + [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode", ] )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_feature_columns = identify_feature_columns(baseline_results)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `baseline_truth_record = build_truth_record( truth_base=baseline_truth, row_count=len(baseline_results), column_count=baseline_results.shape[1] + 3, meta_columns=baseline_meta_colum` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `BASELINE_TRUTH_HASH = baseline_truth_record["truth_hash"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `baseline_results = stamp_truth_columns( baseline_results, truth_hash=BASELINE_TRUTH_HASH, parent_truth_hash=GOLD_PARENT_TRUTH_HASH, pipeline_mode=PIPELINE_MODE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_truth_path = save_truth_record( baseline_truth_record, truth_dir=TRUTHS_PATH, dataset_name=DATASET_NAME, layer_name=baseline_truth_layer_name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `append_truth_index( baseline_truth_record, truth_index_path=TRUTH_INDEX_PATH,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_summary["baseline_truth_hash"] = BASELINE_TRUTH_HASH` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_summary["baseline_truth_path"] = str(baseline_truth_path)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_summary["baseline_process_run_id"] = baseline_process_run_id` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_summary["gold_truth_hash"] = GOLD_PARENT_TRUTH_HASH` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_summary["gold_truth_path"] = str(GOLD_TRUTH_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_summary["gold_process_run_id"] = gold_truth.get("process_run_id")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_summary["gold_feature_set_id"] = gold_truth_runtime_facts.get("feature_set_id")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_metadata = { "gold_preprocessed_scaled_path": str(GOLD_PREPROCESSED_SCALED_DATA_PATH), "gold_fit_path": str(GOLD_FIT_DATA_PATH), "baseline_results_path_csv": str(BASELINE_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_results.to_csv(BASELINE_RESULTS_PATH_CSV, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `baseline_results.to_pickle(BASELINE_RESULTS_PATH_PICKLE)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `joblib.dump(baseline_model, BASELINE_MODEL_ARTIFACT_PATH)` | Writes an artifact or output used for review or downstream notebooks. |
| `joblib.dump(baseline_model, BASELINE_MODELS_PATH)` | Writes an artifact or output used for review or downstream notebooks. |
| `save_json(baseline_thresholds, BASELINE_THRESHOLDS_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `save_json(baseline_summary, BASELINE_SUMMARY_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `save_json(baseline_metadata, BASELINE_METADATA_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(BASELINE_RESULTS_PATH_CSV))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(BASELINE_RESULTS_PATH_PICKLE))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(BASELINE_MODEL_ARTIFACT_PATH))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(BASELINE_MODELS_PATH))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(BASELINE_THRESHOLDS_PATH))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(BASELINE_SUMMARY_PATH))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(BASELINE_METADATA_PATH))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `wandb.save(str(baseline_truth_path))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="save_baseline_outputs", message="Saved baseline results, trained Isolation Forest model, thresholds, summary, metadata, and baseline stage truth reco` | Records or exports ledger information for stage-level traceability. |
| `1 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output, model artifact, optional experiment tracking call, truth record.

## Code Cell 29 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `anomaly_flag`
- `baseline`
- `Baseline`
- `baseline_decision`
- `baseline_flag`
- `baseline_pred`
- `baseline_results`
- `baseline_score`
- `bool`
- `Built`
- `columns`
- `count`
- `detected`
- `detected_row_count`
- `event_step`
- `event_time`
- `extract_baseline_detected_rows`
- `f`
- `get_detected_rows_dataframe`

### Outputs

- `ascending`
- `baseline_detected_rows_dataframe`
- `data`
- `dataframe`
- `decision_column`
- `include_columns`
- `kind`
- `logger`
- `message`
- `pred_column`
- `row_id_column`
- `score_column`
- `sort_by`
- `step`
- `target_flag_column`

### Key Operations

- `baseline_detected_rows_dataframe = get_detected_rows_dataframe( dataframe=baseline_results, target_flag_column="baseline_flag", row_id_column="meta__row_id", score_column="baseline`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="extract_baseline_detected_rows", message="Built the baseline detected-rows dataframe from the scored baseline results using stable row tracking.", da`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print(f"Baseline detected row count: {len(baseline_detected_rows_dataframe):,}")`: Displays a notebook-facing result for inspection.
- `display(baseline_detected_rows_dataframe.head(20))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `bool`
- `display`
- `get_detected_rows_dataframe`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `baseline_detected_rows_dataframe = get_detected_rows_dataframe( dataframe=baseline_results, target_flag_column="baseline_flag", row_id_column="meta__row_id", score_column="baseline` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="extract_baseline_detected_rows", message="Built the baseline detected-rows dataframe from the scored baseline results using stable row tracking.", da` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"Baseline detected row count: {len(baseline_detected_rows_dataframe):,}")` | Displays a notebook-facing result for inspection. |
| `display(baseline_detected_rows_dataframe.head(20))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 30 — Finalize the Ledger and Close the Tracking Run

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `baseline`
- `baseline_ledger_path`
- `baseline_metrics`
- `baseline_model_artifact_path`
- `BASELINE_MODEL_ARTIFACT_PATH`
- `baseline_model_path`
- `BASELINE_MODELS_PATH`
- `baseline_results_path_csv`
- `BASELINE_RESULTS_PATH_CSV`
- `BASELINE_RESULTS_PATH_PICKLE`
- `baseline_results_path_pickle`
- `complete`
- `finalize_baseline_modeling`
- `finish`
- `Gold`
- `ledger`
- `modeling`
- `notebook`
- `save`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `step`

### Key Operations

- `ledger.add( kind="step", step="finalize_baseline_modeling", message="Gold baseline modeling notebook complete.", data={ "baseline_metrics": baseline_metrics, "baseline_results_path`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.write_json(baseline_ledger_path)`: Records or exports ledger information for stage-level traceability.
- `wandb.save(str(baseline_ledger_path))`: Records or exports ledger information for stage-level traceability.
- `wandb_run.finish()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `finish`
- `save`
- `write_json`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `ledger.add( kind="step", step="finalize_baseline_modeling", message="Gold baseline modeling notebook complete.", data={ "baseline_metrics": baseline_metrics, "baseline_results_path` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.write_json(baseline_ledger_path)` | Records or exports ledger information for stage-level traceability. |
| `wandb.save(str(baseline_ledger_path))` | Records or exports ledger information for stage-level traceability. |
| `wandb_run.finish()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: optional experiment tracking call.

## Code Cell 31 — Run Final Lineage and Consistency Checks

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__name__`
- `a`
- `Any`
- `astype`
- `baseline`
- `Baseline`
- `baseline_metadata`
- `BASELINE_METADATA_PATH`
- `baseline_results`
- `baseline_truth_hash`
- `BASELINE_TRUTH_HASH`
- `baseline_truth_path`
- `check`
- `column_name`
- `columns`
- `contain`
- `created`
- `dataframe`
- `dataframe_parent`
- `dictionary`

### Outputs

- `baseline_parent_values`
- `baseline_results_truth_hash_check`
- `loaded_baseline_truth_raw`
- `missing_baseline_meta_columns`
- `required_baseline_meta_columns`
- `saved_baseline_metadata_raw`

### Key Operations

- `required_baseline_meta_columns = [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_baseline_meta_columns = [ column_name for column_name in required_baseline_meta_columns if column_name not in baseline_results.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if missing_baseline_meta_columns: raise ValueError( f"baseline_results is missing required lineage columns: {missing_baseline_meta_columns}" )`: Controls validation, iteration, file handling, or error handling for this step.
- `baseline_results_truth_hash_check = extract_truth_hash(baseline_results)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if baseline_results_truth_hash_check is None: raise ValueError("baseline_results does not contain a readable meta__truth_hash value.")`: Controls validation, iteration, file handling, or error handling for this step.
- `if baseline_results_truth_hash_check != BASELINE_TRUTH_HASH: raise ValueError( "baseline_results truth hash does not match BASELINE_TRUTH_HASH:\n" f"dataframe={baseline_results_tru`: Controls validation, iteration, file handling, or error handling for this step.
- `baseline_parent_values = baseline_results["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if not baseline_parent_values: raise ValueError("baseline_results is missing populated meta__parent_truth_hash values.")`: Controls validation, iteration, file handling, or error handling for this step.
- `if len(baseline_parent_values) != 1: raise ValueError(f"baseline_results has multiple parent truth hashes: {baseline_parent_values}")`: Controls validation, iteration, file handling, or error handling for this step.
- `if baseline_parent_values[0] != GOLD_PARENT_TRUTH_HASH: raise ValueError( "baseline_results parent truth hash does not match GOLD_PARENT_TRUTH_HASH:\n" f"dataframe_parent={baseline`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `astype`
- `dropna`
- `exists`
- `extract_truth_hash`
- `FileNotFoundError`
- `get`
- `isinstance`
- `load_json`
- `Path`
- `tolist`
- `type`
- `TypeError`
- `unique`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `required_baseline_meta_columns = [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_baseline_meta_columns = [ column_name for column_name in required_baseline_meta_columns if column_name not in baseline_results.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_baseline_meta_columns: raise ValueError( f"baseline_results is missing required lineage columns: {missing_baseline_meta_columns}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `baseline_results_truth_hash_check = extract_truth_hash(baseline_results)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if baseline_results_truth_hash_check is None: raise ValueError("baseline_results does not contain a readable meta__truth_hash value.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if baseline_results_truth_hash_check != BASELINE_TRUTH_HASH: raise ValueError( "baseline_results truth hash does not match BASELINE_TRUTH_HASH:\n" f"dataframe={baseline_results_tru` | Controls validation, iteration, file handling, or error handling for this step. |
| `baseline_parent_values = baseline_results["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not baseline_parent_values: raise ValueError("baseline_results is missing populated meta__parent_truth_hash values.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if len(baseline_parent_values) != 1: raise ValueError(f"baseline_results has multiple parent truth hashes: {baseline_parent_values}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if baseline_parent_values[0] != GOLD_PARENT_TRUTH_HASH: raise ValueError( "baseline_results parent truth hash does not match GOLD_PARENT_TRUTH_HASH:\n" f"dataframe_parent={baseline` | Controls validation, iteration, file handling, or error handling for this step. |
| `if not Path(baseline_truth_path).exists(): raise FileNotFoundError(f"Baseline truth file was not created: {baseline_truth_path}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `loaded_baseline_truth_raw = load_json(baseline_truth_path)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not isinstance(loaded_baseline_truth_raw, dict): raise TypeError( "Saved baseline truth JSON must load as a dictionary. " f"Got {type(loaded_baseline_truth_raw).__name__}: {load` | Controls validation, iteration, file handling, or error handling for this step. |
| `loaded_baseline_truth: dict[str, Any] = loaded_baseline_truth_raw` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if loaded_baseline_truth.get("truth_hash") != BASELINE_TRUTH_HASH: raise ValueError( "Saved Baseline truth file hash does not match BASELINE_TRUTH_HASH:\n" f"file={loaded_baseline_` | Controls validation, iteration, file handling, or error handling for this step. |
| `if loaded_baseline_truth.get("parent_truth_hash") != GOLD_PARENT_TRUTH_HASH: raise ValueError( "Saved Baseline truth file parent hash does not match GOLD_PARENT_TRUTH_HASH:\n" f"tr` | Controls validation, iteration, file handling, or error handling for this step. |
| `saved_baseline_metadata_raw = load_json(BASELINE_METADATA_PATH)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not isinstance(saved_baseline_metadata_raw, dict): raise TypeError( "Saved baseline metadata JSON must load as a dictionary. " f"Got {type(saved_baseline_metadata_raw).__name__}` | Controls validation, iteration, file handling, or error handling for this step. |
| `saved_baseline_metadata: dict[str, Any] = saved_baseline_metadata_raw` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if saved_baseline_metadata.get("baseline_truth_hash") != BASELINE_TRUTH_HASH: raise ValueError( "baseline_metadata baseline_truth_hash does not match BASELINE_TRUTH_HASH:\n" f"meta` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Gold Baseline lineage sanity check passed.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 32 — Gold Baseline SQL Write Cell

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `else`
- `get`
- `globals`
- `Postgres`
- `skipped`
- `write`
- `write_gold_baseline_scores_sql`

### Outputs

- `capstone_schema`
- `dataset_id`
- `dataset_name`
- `engine`
- `gold_baseline_sql_summary_dataframe`
- `notebook_globals`
- `run_id`
- `WRITE_TO_POSTGRES`

### Key Operations

- `WRITE_TO_POSTGRES = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if WRITE_TO_POSTGRES: gold_baseline_sql_summary_dataframe = write_gold_baseline_scores_sql( engine=engine, capstone_schema=CAPSTONE_SCHEMA, dataset_id=DATASET_ID, run_id=RUN_ID, no`: Displays a notebook-facing result for inspection.
- `else: print("Postgres write skipped.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `get`
- `globals`
- `write_gold_baseline_scores_sql`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `WRITE_TO_POSTGRES = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if WRITE_TO_POSTGRES: gold_baseline_sql_summary_dataframe = write_gold_baseline_scores_sql( engine=engine, capstone_schema=CAPSTONE_SCHEMA, dataset_id=DATASET_ID, run_id=RUN_ID, no` | Displays a notebook-facing result for inspection. |
| `else: print("Postgres write skipped.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

