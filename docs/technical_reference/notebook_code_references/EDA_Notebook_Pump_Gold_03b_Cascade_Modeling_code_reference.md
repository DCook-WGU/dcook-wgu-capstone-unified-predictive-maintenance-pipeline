# Notebook Code Reference: EDA_Notebook_Pump_Gold_03b_Cascade_Modeling

Notebook path:

`notebooks/experiments/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling.ipynb`

## Notebook Purpose

This notebook supports staged cascade anomaly-detection modeling and records model behavior for comparison.

Notebook stage:

`Gold`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Tuned Cascade Modeling Setup and Imports | Code Cell 01, Code Cell 02 |
| Define configuration mapping guards | Code Cell 03 |
| Load Configuration, Paths, and Tuned Cascade Runtime Settings | Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08, Code Cell 09 |
| Review intermediate output | Code Cell 10 |
| Start Logging for the Tuned Cascade Modeling Stage | Code Cell 11, Code Cell 12 |
| Initialize Experiment Tracking | Code Cell 13 |
| Initialize the Cascade Ledger | Code Cell 14, Code Cell 15 |
| Load the Gold Modeling Inputs and Resolve the Parent Truth | Code Cell 16, Code Cell 17 |
| Rebuild the Train and Test Masks | Code Cell 18 |
| Define the Stage 3 Reference Profile Logic | Code Cell 19 |
| Build the Stage 3 Reference Profile | Code Cell 20 |
| Prepare the Feature Matrices and Evaluation Labels | Code Cell 21 |
| Run validation guardrails | Code Cell 22 |
| Answer | Code Cell 23, Code Cell 27, Code Cell 31, Code Cell 45, Code Cell 51, Code Cell 52 |
| Define percentile threshold selection | Code Cell 24 |
| Define label-based model evaluation | Code Cell 25 |
| Run Stage 1: Broad Isolation Forest Screening | Code Cell 26 |
| Build review visualization | Code Cell 28 |
| Define the Stage 2 Selection Logic | Code Cell 29 |
| Run Stage 2 Selection and Keep the Best Narrow Model | Code Cell 30 |
| Quick Verifications Cell | Code Cell 32 |
| Validate That the Stage 3 Rule Sensors Exist | Code Cell 33 |
| Define the Primary Profile Breach Logic | Code Cell 34 |
| Define the Secondary Corroboration Logic | Code Cell 35 |
| Compute the Secondary Breach Count | Code Cell 36 |
| Compute Stage 3 secondary breach evidence | Code Cell 37 |
| Define the Persistence Logic | Code Cell 38 |
| Compute Stage 3 persistence evidence | Code Cell 39 |
| Define the Drift Logic | Code Cell 40 |
| Compute the Drift Flag Inputs | Code Cell 41 |
| Build the Final Stage 3 Evidence Flags and Final Cascade Decision | Code Cell 42 |
| Finalize cascade stage flag columns | Code Cell 43 |
| Build the Main Cascade Metrics | Code Cell 44 |
| Validate cascade output columns and counts | Code Cell 46 |
| Build the Cascade Summary, Threshold Records, and Truth Artifact | Code Cell 47, Code Cell 48, Code Cell 49 |
| Build detected-row review extract | Code Cell 50 |
| Finalize the Ledger and Close the Tracking Run | Code Cell 53 |
| Run Final Lineage and Consistency Checks | Code Cell 54 |
| Gold Cascade SQL Write Cell | Code Cell 55 |

## Code Cell 01 — Tuned Cascade Modeling Setup and Imports

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
- `build_artifact_dirs_from_config`
- `build_cascade_variant_contract`
- `build_file_fingerprint`
- `build_gold_model_output_validation_contract`
- `build_stage_scoring_frame`
- `build_stage3_rule_payload_from_globals`
- `build_truth_config_block`
- `build_truth_record`
- `cascade_row_tracking`
- `cast`
- `classification_report`
- `cluster`
- `columns`

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
| `from sklearn.model_selection import train_test_split, KFold, ParameterGrid` | Imports a dependency or project helper used by later cells. |
| `from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler` | Imports a dependency or project helper used by later cells. |
| `from sklearn.decomposition import PCA` | Imports a dependency or project helper used by later cells. |
| `from sklearn.cluster import KMeans` | Imports a dependency or project helper used by later cells. |
| `from sklearn.ensemble import RandomForestClassifier, IsolationForest` | Imports a dependency or project helper used by later cells. |
| `from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score` | Imports a dependency or project helper used by later cells. |
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
| `from utils.medallion.gold.gold_cascade_validation_contracts import ( build_cascade_variant_contract, build_stage3_rule_payload_from_globals, write_json_contract,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.artifacts import gold_model_validation_contract_path` | Imports a dependency or project helper used by later cells. |
| `from utils.medallion.gold.gold_cascade_validation_contracts import ( build_gold_model_output_validation_contract, write_gold_model_output_validation_contract,` | Imports a dependency or project helper used by later cells. |
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

## Code Cell 02 — Tuned Cascade Modeling Setup and Imports

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `anomaly`
- `Any`
- `array`
- `asarray`
- `behavior`
- `bool`
- `both`
- `Cannot`
- `choose`
- `Choose`
- `Compatibility`
- `def`
- `dtype`
- `empty`
- `max`
- `mean`
- `metadata`
- `min`
- `Notebook`

### Outputs

- `choose_threshold_by_percentile`
- `info`
- `scores_array`
- `threshold`

### Key Operations

- `def choose_threshold_by_percentile( scores: Sequence[float], percentile: float = 95.0, *, return_info: bool = False,`: Defines notebook-local logic used later in the notebook.
- `) -> float \| Tuple[float, Dict[str, Any]]: """ Choose anomaly threshold using a score percentile. Compatibility behavior ---------------------- - Notebook-style usage: threshold = `: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `asarray`
- `choose_threshold_by_percentile`
- `max`
- `mean`
- `min`
- `percentile`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def choose_threshold_by_percentile( scores: Sequence[float], percentile: float = 95.0, *, return_info: bool = False,` | Defines notebook-local logic used later in the notebook. |
| `) -> float \| Tuple[float, Dict[str, Any]]: """ Choose anomaly threshold using a score percentile. Compatibility behavior ---------------------- - Notebook-style usage: threshold = ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 03 — Define configuration mapping guards

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__name__`
- `a`
- `an`
- `Any`
- `array`
- `asarray`
- `astype`
- `be`
- `bool`
- `boolean`
- `cast`
- `choose_threshold_by_percentile`
- `column`
- `config`
- `Convert`
- `def`
- `dictionary`
- `dtype`
- `either`
- `empty`

### Outputs

- `as_bool_array`
- `as_float_array`
- `as_int_array`
- `cfg_optional_mapping`
- `cfg_require_mapping`
- `choose_threshold_value`
- `cleaned_values`
- `require_float`
- `require_mapping`
- `require_str_list`
- `score_values`
- `threshold_result`
- `value`

### Key Operations

- `def cfg_require_mapping(value: object, name: str) -> Mapping[str, Any]: if not isinstance(value, Mapping): raise TypeError( f"{name} must be a mapping, got {type(value).__name__}: `: Defines notebook-local logic used later in the notebook.
- `def cfg_optional_mapping(value: object \| None, name: str) -> Mapping[str, Any]: if value is None: return {} return cfg_require_mapping(value, name)`: Defines notebook-local logic used later in the notebook.
- `def require_mapping(value: Any, name: str) -> dict[str, Any]: """ Validate that a loaded JSON/config object is a dictionary. """ if not isinstance(value, dict): raise TypeError( f"`: Defines notebook-local logic used later in the notebook.
- `def require_str_list(value: Any, name: str) -> list[str]: """ Validate that a loaded JSON/config object is a list of strings. """ if value is None: raise ValueError(f"{name} is Non`: Defines notebook-local logic used later in the notebook.
- `def require_float(value: Any, name: str) -> float: """ Convert a scalar or threshold-return tuple into a float. Some project helpers may return either: threshold or: (threshold, me`: Defines notebook-local logic used later in the notebook.
- `def as_bool_array(value: Any, name: str) -> np.ndarray: """ Convert a Pandas/NumPy boolean mask into a NumPy bool array. """ if isinstance(value, pd.Series): return value.to_numpy(`: Defines notebook-local logic used later in the notebook.
- `def as_int_array(value: Any, name: str) -> np.ndarray: """ Convert labels/flags into a NumPy int array. """ if value is None: raise ValueError(f"{name} is None.") if isinstance(val`: Defines notebook-local logic used later in the notebook.
- `def as_float_array(value: Any, name: str) -> np.ndarray: """ Convert scores into a flat NumPy float array. """ if value is None: raise ValueError(f"{name} is None.") return np.asar`: Defines notebook-local logic used later in the notebook.
- `def choose_threshold_value(scores: Any, percentile: float) -> float: """ Normalize score input and threshold helper output for Pylance. """ score_values = as_float_array(scores, "s`: Defines notebook-local logic used later in the notebook.

Important functions or methods detected:
- `as_bool_array`
- `as_float_array`
- `as_int_array`
- `asarray`
- `astype`
- `cast`
- `cfg_optional_mapping`
- `cfg_require_mapping`
- `choose_threshold_by_percentile`
- `choose_threshold_value`
- `fillna`
- `isinstance`
- `require_float`
- `require_mapping`
- `require_str_list`
- `reshape`
- `strip`
- `to_numpy`
- `tolist`
- `type`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def cfg_require_mapping(value: object, name: str) -> Mapping[str, Any]: if not isinstance(value, Mapping): raise TypeError( f"{name} must be a mapping, got {type(value).__name__}: ` | Defines notebook-local logic used later in the notebook. |
| `def cfg_optional_mapping(value: object \| None, name: str) -> Mapping[str, Any]: if value is None: return {} return cfg_require_mapping(value, name)` | Defines notebook-local logic used later in the notebook. |
| `def require_mapping(value: Any, name: str) -> dict[str, Any]: """ Validate that a loaded JSON/config object is a dictionary. """ if not isinstance(value, dict): raise TypeError( f"` | Defines notebook-local logic used later in the notebook. |
| `def require_str_list(value: Any, name: str) -> list[str]: """ Validate that a loaded JSON/config object is a list of strings. """ if value is None: raise ValueError(f"{name} is Non` | Defines notebook-local logic used later in the notebook. |
| `def require_float(value: Any, name: str) -> float: """ Convert a scalar or threshold-return tuple into a float. Some project helpers may return either: threshold or: (threshold, me` | Defines notebook-local logic used later in the notebook. |
| `def as_bool_array(value: Any, name: str) -> np.ndarray: """ Convert a Pandas/NumPy boolean mask into a NumPy bool array. """ if isinstance(value, pd.Series): return value.to_numpy(` | Defines notebook-local logic used later in the notebook. |
| `def as_int_array(value: Any, name: str) -> np.ndarray: """ Convert labels/flags into a NumPy int array. """ if value is None: raise ValueError(f"{name} is None.") if isinstance(val` | Defines notebook-local logic used later in the notebook. |
| `def as_float_array(value: Any, name: str) -> np.ndarray: """ Convert scores into a flat NumPy float array. """ if value is None: raise ValueError(f"{name} is None.") return np.asar` | Defines notebook-local logic used later in the notebook. |
| `def choose_threshold_value(scores: Any, percentile: float) -> float: """ Normalize score input and threshold helper output for Pylance. """ score_values = as_float_array(scores, "s` | Defines notebook-local logic used later in the notebook. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 04 — Load Configuration, Paths, and Tuned Cascade Runtime Settings

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `aliases`
- `capstone`
- `cascade`
- `context`
- `context_loaded`
- `dataset_config`
- `default`
- `execution`
- `gold`
- `gold_cascade`
- `gold_modeling_cascade_tuned`
- `info`
- `load_notebook_context`
- `Loaded`
- `loaded`
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
- `CONTEXT_STAGE = "gold_cascade"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_DATASET = "pump"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LAYER = "gold"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_RUN_MODE = "train"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_PROFILE = "default"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LOG_FILE = "gold_modeling_cascade_tuned.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.gold.cascade.tuned", log_filena`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `CONTEXT_STAGE = "gold_cascade"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LAYER = "gold"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_RUN_MODE = "train"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LOG_FILE = "gold_modeling_cascade_tuned.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.gold.cascade.tuned", log_filena` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
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

## Code Cell 05 — Load Configuration, Paths, and Tuned Cascade Runtime Settings

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `Artifacts`
- `B`
- `Base`
- `blocks`
- `build_truth_config_block`
- `capstone`
- `Cascade`
- `cascade`
- `cascade_tuned_metadata_file_name`
- `cascade_tuned_metadata_path`
- `cascade_tuned_reference_profile_file_name`
- `cascade_tuned_reference_profile_path`
- `cascade_tuned_results_file_name_csv`
- `cascade_tuned_results_file_name_pickle`
- `cascade_tuned_results_path_csv`
- `cascade_tuned_results_path_pickle`
- `cascade_tuned_stage1_model_artifact_path`
- `cascade_tuned_stage1_model_file_name`
- `cascade_tuned_stage1_models_path`

### Outputs

- `ARTIFACTS_ROOT`
- `CASCADE_METADATA_FILE_NAME`
- `CASCADE_METADATA_PATH`
- `CASCADE_REFERENCE_PROFILE_FILE_NAME`
- `CASCADE_REFERENCE_PROFILE_PATH`
- `CASCADE_RESULTS_FILE_NAME_CSV`
- `CASCADE_RESULTS_FILE_NAME_PICKLE`
- `CASCADE_RESULTS_PATH_CSV`
- `CASCADE_RESULTS_PATH_PICKLE`
- `CASCADE_SUMMARY_FILE_NAME`
- `CASCADE_SUMMARY_PATH`
- `CASCADE_THRESHOLDS_FILE_NAME`
- `CASCADE_THRESHOLDS_PATH`
- `CASCADE_VARIANT`
- `CONFIG_PROFILE`
- `DATASET_NAME`
- `DATASET_NAME_CONFIG`
- `GOLD_ARTIFACTS_PATH`
- `GOLD_CASCADE_LEDGER_FILE_NAME`
- `GOLD_FIT_DATA_PATH`

### Key Operations

- `# Tuned cascade notebook selector.`: Documents the purpose or boundary of the surrounding notebook step.
- `CASCADE_VARIANT = "tuned"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `TRUTH_CONFIG = cast(Dict[str, Any], build_truth_config_block(CONFIG))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_CONFIG["pipeline"] = PIPELINE`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `# ---- Stage details ----`: Documents the purpose or boundary of the surrounding notebook step.
- `STAGE = "gold"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `LAYER_NAME = str(GOLD_CFG["layer_name"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RECIPE_ID = str(GOLD_CFG["recipe_id"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_VERSION = str(VERSIONS_CFG["gold"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_VERSION = str(VERSIONS_CFG["truth"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `build_truth_config_block`
- `cast`
- `cfg_require_mapping`
- `get`
- `items`
- `lower`
- `make_process_run_id`
- `mkdir`
- `Path`
- `set_wandb_dir_from_config`
- `strip`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Tuned cascade notebook selector.` | Documents the purpose or boundary of the surrounding notebook step. |
| `CASCADE_VARIANT = "tuned"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
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
| `# ---- Cascade config blocks ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `STAGE1_CFG = cast( Dict[str, Any], cfg_require_mapping(GOLD_CFG.get("stage1", {}), "CONFIG['gold_cascade']['stage1']"),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STAGE2_CFG = cast( Dict[str, Any], cfg_require_mapping(GOLD_CFG.get("stage2", {}), "CONFIG['gold_cascade']['stage2']"),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STAGE3_CFG = cast( Dict[str, Any], cfg_require_mapping(GOLD_CFG.get("stage3", {}), "CONFIG['gold_cascade']['stage3']"),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STAGE2_FIXED_CFG = cast( Dict[str, Any], cfg_require_mapping( STAGE2_CFG.get("fixed", {}), "CONFIG['gold_cascade']['stage2']['fixed']", ),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STAGE2_SEARCH_CFG = cast( Dict[str, Any], cfg_require_mapping( STAGE2_CFG.get("search", {}), "CONFIG['gold_cascade']['stage2']['search']", ),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STAGE2_FIXED_PARAMS = cast( Dict[str, Any], cfg_require_mapping( STAGE2_FIXED_CFG.get("params", {}), "CONFIG['gold_cascade']['stage2']['fixed']['params']", ),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STAGE2_SEARCH_PARAM_GRID_CFG = cast( Dict[str, Any], cfg_require_mapping( STAGE2_SEARCH_CFG.get("param_grid", {}), "CONFIG['gold_cascade']['stage2']['search']['param_grid']", ),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Runtime knobs ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `RANDOM_SEED = int(GOLD_CFG["random_seed"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE1_ESTIMATOR_COUNT = int(STAGE1_CFG["estimator_count"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE1_THRESHOLD_PERCENTILE = float(STAGE1_CFG["threshold_percentile"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE2_SELECTION_MODE = str(STAGE2_CFG["selection_mode"]).strip().lower()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE2_MIN_RECALL = float(STAGE2_CFG["min_recall"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE2_RANDOM_STATE = int(STAGE2_CFG.get("random_state", RANDOM_SEED))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE2_FIXED_THRESHOLD_PERCENTILE = float( STAGE2_FIXED_CFG["threshold_percentile"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STAGE2_THRESHOLD_GRID = [ float(value) for value in STAGE2_SEARCH_CFG["threshold_grid"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STAGE2_PARAM_GRID = { str(key): list(value) for key, value in STAGE2_SEARCH_PARAM_GRID_CFG.items()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STAGE3_MIN_PRIMARY_SENSOR_HITS = int(STAGE3_CFG["min_primary_sensor_hits"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE3_MIN_SECONDARY_SENSOR_HITS = int(STAGE3_CFG["min_secondary_sensor_hits"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE3_ROLLING_WINDOW_SIZE = int(STAGE3_CFG["rolling_window_size"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE3_MINIMUM_FLAGS_IN_WINDOW = int(STAGE3_CFG["minimum_flags_in_window"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- W&B ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `WANDB_PROJECT = str(WANDB_CFG.get("project", "capstone"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WANDB_ENTITY = str(WANDB_CFG.get("entity", ""))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `108 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 06 — Load Configuration, Paths, and Tuned Cascade Runtime Settings

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

## Code Cell 07 — Load Configuration, Paths, and Tuned Cascade Runtime Settings

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

## Code Cell 08 — Load Configuration, Paths, and Tuned Cascade Runtime Settings

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold_cascade_tuned__resolved_config`
- `build_artifact_dirs_from_config`
- `cascade_tuned_metadata_file_name`
- `cascade_tuned_reference_profile_file_name`
- `cascade_tuned_results_file_name_csv`
- `cascade_tuned_results_file_name_pickle`
- `cascade_tuned_stage1_model_file_name`
- `cascade_tuned_stage2_model_file_name`
- `cascade_tuned_summary_file_name`
- `cascade_tuned_thresholds_file_name`
- `DATASET_NAME`
- `execution`
- `export_config_snapshot`
- `f`
- `FILENAMES`
- `get`
- `gold_cascade`
- `gold_cascade_tuned_ledger_file_name`
- `lineage`
- `metadata`

### Outputs

- `cascade_ledger_path`
- `CASCADE_METADATA_PATH`
- `CASCADE_REFERENCE_PROFILE_PATH`
- `CASCADE_RESULTS_PATH_CSV`
- `CASCADE_RESULTS_PATH_PICKLE`
- `CASCADE_SUMMARY_PATH`
- `CASCADE_THRESHOLDS_PATH`
- `CASCADE_VARIANT`
- `config`
- `CONFIG_SNAPSHOT_PATH`
- `GOLD_ARTIFACTS_PATH`
- `GOLD_CASCADE_ARTIFACT_DIRS`
- `GOLD_CASCADE_CONFIG_DIR`
- `GOLD_CASCADE_ROOT`
- `stage_key`
- `STAGE1_MODEL_ARTIFACT_PATH`
- `STAGE2_MODEL_ARTIFACT_PATH`
- `variant`

### Key Operations

- `CASCADE_VARIANT = "tuned"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_CASCADE_ARTIFACT_DIRS = build_artifact_dirs_from_config( config=CONFIG, stage_key="gold_cascade", variant=CASCADE_VARIANT,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `GOLD_ARTIFACTS_PATH = GOLD_CASCADE_ARTIFACT_DIRS["stage_dataset_root"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_CASCADE_ROOT = GOLD_CASCADE_ARTIFACT_DIRS["root"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CASCADE_RESULTS_PATH_CSV = ( GOLD_CASCADE_ARTIFACT_DIRS["scores"] / FILENAMES["cascade_tuned_results_file_name_csv"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `CASCADE_RESULTS_PATH_PICKLE = ( GOLD_CASCADE_ARTIFACT_DIRS["scores"] / FILENAMES["cascade_tuned_results_file_name_pickle"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `CASCADE_THRESHOLDS_PATH = ( GOLD_CASCADE_ARTIFACT_DIRS["thresholds"] / FILENAMES["cascade_tuned_thresholds_file_name"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `CASCADE_SUMMARY_PATH = ( GOLD_CASCADE_ARTIFACT_DIRS["summaries"] / FILENAMES["cascade_tuned_summary_file_name"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `build_artifact_dirs_from_config`
- `export_config_snapshot`
- `get`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `CASCADE_VARIANT = "tuned"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_CASCADE_ARTIFACT_DIRS = build_artifact_dirs_from_config( config=CONFIG, stage_key="gold_cascade", variant=CASCADE_VARIANT,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_ARTIFACTS_PATH = GOLD_CASCADE_ARTIFACT_DIRS["stage_dataset_root"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_CASCADE_ROOT = GOLD_CASCADE_ARTIFACT_DIRS["root"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CASCADE_RESULTS_PATH_CSV = ( GOLD_CASCADE_ARTIFACT_DIRS["scores"] / FILENAMES["cascade_tuned_results_file_name_csv"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_RESULTS_PATH_PICKLE = ( GOLD_CASCADE_ARTIFACT_DIRS["scores"] / FILENAMES["cascade_tuned_results_file_name_pickle"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_THRESHOLDS_PATH = ( GOLD_CASCADE_ARTIFACT_DIRS["thresholds"] / FILENAMES["cascade_tuned_thresholds_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_SUMMARY_PATH = ( GOLD_CASCADE_ARTIFACT_DIRS["summaries"] / FILENAMES["cascade_tuned_summary_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_METADATA_PATH = ( GOLD_CASCADE_ARTIFACT_DIRS["metadata"] / FILENAMES["cascade_tuned_metadata_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_REFERENCE_PROFILE_PATH = ( GOLD_CASCADE_ARTIFACT_DIRS["profiles"] / FILENAMES["cascade_tuned_reference_profile_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STAGE1_MODEL_ARTIFACT_PATH = ( GOLD_CASCADE_ARTIFACT_DIRS["models"] / FILENAMES["cascade_tuned_stage1_model_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `STAGE2_MODEL_ARTIFACT_PATH = ( GOLD_CASCADE_ARTIFACT_DIRS["models"] / FILENAMES["cascade_tuned_stage2_model_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#STAGE1_MODELS_PATH = STAGE1_MODEL_ARTIFACT_PATH` | Documents the purpose or boundary of the surrounding notebook step. |
| `#STAGE2_MODELS_PATH = STAGE2_MODEL_ARTIFACT_PATH` | Documents the purpose or boundary of the surrounding notebook step. |
| `cascade_ledger_path = ( GOLD_CASCADE_ARTIFACT_DIRS["lineage"] / FILENAMES["gold_cascade_tuned_ledger_file_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_CASCADE_CONFIG_DIR = GOLD_CASCADE_ARTIFACT_DIRS["config"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_SNAPSHOT_PATH = ( GOLD_CASCADE_CONFIG_DIR / f"{DATASET_NAME}__gold_cascade_tuned__resolved_config.yaml"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if CONFIG["execution"].get("save_config_snapshot", True): export_config_snapshot(CONFIG, CONFIG_SNAPSHOT_PATH)` | Controls validation, iteration, file handling, or error handling for this step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 09 — Load Configuration, Paths, and Tuned Cascade Runtime Settings

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

## Code Cell 10 — Review intermediate output

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

## Code Cell 11 — Start Logging for the Tuned Cascade Modeling Stage

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

## Code Cell 12 — Start Logging for the Tuned Cascade Modeling Stage

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
- `gold_modeling_cascade_tuned`
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
- `gold_log_path = paths.logs / "gold_modeling_cascade_tuned.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `gold_log_path = paths.logs / "gold_modeling_cascade_tuned.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
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

## Code Cell 13 — Initialize Experiment Tracking

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `B`
- `dataset`
- `DATASET_NAME`
- `GOLD_FIT_DATA_PATH`
- `gold_fit_path`
- `gold_modeling_cascade`
- `GOLD_PREPROCESSED_SCALED_DATA_PATH`
- `gold_scored_path`
- `GOLD_VERSION`
- `gold_version`
- `info`
- `init`
- `initialized`
- `logger`
- `s`
- `STAGE`
- `stage`
- `stage1_features_path`
- `STAGE1_FEATURES_PATH`
- `stage1_threshold_percentile`

### Outputs

- `config`
- `entity`
- `job_type`
- `name`
- `project`
- `wandb_run`

### Key Operations

- `# W&B`: Documents the purpose or boundary of the surrounding notebook step.
- `wandb_run = wandb.init( project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="gold_modeling_cascade", config={ "gold_version": GOLD_VERSION, "dataset": DATASET`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger.info("W&B initialized: %s", wandb_run.name)`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `info`
- `init`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# W&B` | Documents the purpose or boundary of the surrounding notebook step. |
| `wandb_run = wandb.init( project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="gold_modeling_cascade", config={ "gold_version": GOLD_VERSION, "dataset": DATASET` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("W&B initialized: %s", wandb_run.name)` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- Artifact or state outputs detected: optional experiment tracking call.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 14 — Initialize the Cascade Ledger

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

## Code Cell 15 — Initialize the Cascade Ledger

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

## Code Cell 16 — Load the Gold Modeling Inputs and Resolve the Parent Truth

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold__truth__`
- `add`
- `artifact`
- `artifact_paths`
- `astype`
- `cascade`
- `dataset`
- `dropna`
- `f`
- `features`
- `fit`
- `get`
- `get_dataset_name_from_truth`
- `get_pipeline_mode_from_truth`
- `get_truth_hash`
- `Gold`
- `gold`
- `gold_fit_path`
- `gold_fit_shape`
- `gold_preprocessed_path`

### Outputs

- `column_name`
- `data`
- `dataframe`
- `dataset_name`
- `DATASET_NAME`
- `GOLD_DATASET_NAME`
- `GOLD_FIT_DATA_PATH`
- `gold_fit_dataframe`
- `GOLD_PARENT_TRUTH_HASH`
- `GOLD_PREPROCESSED_DATA_PATH`
- `gold_preprocessed_dataframe`
- `gold_preprocessed_scaled_dataframe`
- `GOLD_TEST_DATA_PATH`
- `gold_test_dataframe`
- `GOLD_TRAIN_DATA_PATH`
- `gold_train_dataframe`
- `gold_truth`
- `gold_truth_artifact_paths`
- `GOLD_TRUTH_PATH`
- `gold_truth_runtime_facts`

### Key Operations

- `logger.info("Loading Gold Preprocessed parquet: %s", GOLD_PREPROCESSED_SCALED_DATA_PATH)`: Writes a logger message for traceability during notebook execution.
- `gold_preprocessed_scaled_dataframe = load_data(GOLD_PREPROCESSED_SCALED_DATA_PATH)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_DATASET_NAME = ( gold_preprocessed_scaled_dataframe["meta__dataset"] .dropna() .astype("string") .str.strip()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `GOLD_DATASET_NAME = GOLD_DATASET_NAME[GOLD_DATASET_NAME != ""]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if len(GOLD_DATASET_NAME) == 0: raise ValueError("Gold cascade input dataframe is missing usable meta__dataset values.")`: Controls validation, iteration, file handling, or error handling for this step.
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
- `load_data`
- `load_json`
- `load_parent_truth_record_from_dataframe`
- `Path`
- `require_str_list`
- `strip`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `logger.info("Loading Gold Preprocessed parquet: %s", GOLD_PREPROCESSED_SCALED_DATA_PATH)` | Writes a logger message for traceability during notebook execution. |
| `gold_preprocessed_scaled_dataframe = load_data(GOLD_PREPROCESSED_SCALED_DATA_PATH)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_DATASET_NAME = ( gold_preprocessed_scaled_dataframe["meta__dataset"] .dropna() .astype("string") .str.strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_DATASET_NAME = GOLD_DATASET_NAME[GOLD_DATASET_NAME != ""]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(GOLD_DATASET_NAME) == 0: raise ValueError("Gold cascade input dataframe is missing usable meta__dataset values.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `GOLD_DATASET_NAME = str(GOLD_DATASET_NAME.iloc[0]).strip()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_truth = load_parent_truth_record_from_dataframe( dataframe=gold_preprocessed_scaled_dataframe, truth_dir=TRUTHS_PATH, parent_layer_name="gold", dataset_name=GOLD_DATASET_NAME,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DATASET_NAME = get_dataset_name_from_truth(gold_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_PARENT_TRUTH_HASH = get_truth_hash(gold_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PARENT_PIPELINE_MODE = get_pipeline_mode_from_truth(gold_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if PARENT_PIPELINE_MODE is not None: PIPELINE_MODE = PARENT_PIPELINE_MODE` | Controls validation, iteration, file handling, or error handling for this step. |
| `GOLD_TRUTH_PATH = ( TRUTHS_PATH / "gold" / f"{DATASET_NAME}__gold__truth__{GOLD_PARENT_TRUTH_HASH}.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold_truth_runtime_facts = gold_truth.get("runtime_facts", {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_truth_artifact_paths = gold_truth.get("artifact_paths", {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_PREPROCESSED_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_preprocessed_path", str(GOLD_PREPROCESSED_DATA_PATH)))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_FIT_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_fit_path", str(GOLD_FIT_DATA_PATH)))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_TEST_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_test_path", str(GOLD_TEST_DATA_PATH)))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_TRAIN_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_train_path", str(GOLD_TRAIN_DATA_PATH)))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE1_FEATURES_PATH = Path(gold_truth_artifact_paths.get("stage1_features_path", str(STAGE1_FEATURES_PATH)))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE2_FEATURES_PATH = Path(gold_truth_artifact_paths.get("stage2_features_path", str(STAGE2_FEATURES_PATH)))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE3_PRIMARY_PATH = Path(gold_truth_artifact_paths.get("stage3_primary_path", str(STAGE3_PRIMARY_PATH)))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE3_SECONDARY_PATH = Path(gold_truth_artifact_paths.get("stage3_secondary_path", str(STAGE3_SECONDARY_PATH)))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Resolved Gold cascade dataset name from Gold truth: %s", DATASET_NAME)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Resolved Gold truth path: %s", GOLD_TRUTH_PATH)` | Writes a logger message for traceability during notebook execution. |
| `print("Gold cascade dataset name from parent truth:", DATASET_NAME)` | Displays a notebook-facing result for inspection. |
| `print("Gold cascade parent truth hash:", GOLD_PARENT_TRUTH_HASH)` | Displays a notebook-facing result for inspection. |
| `logger.info("Loading Gold Preprocessed parquet: %s", GOLD_PREPROCESSED_DATA_PATH)` | Writes a logger message for traceability during notebook execution. |
| `gold_preprocessed_dataframe = load_data(GOLD_PREPROCESSED_DATA_PATH)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Loading Gold fit parquet: %s", GOLD_FIT_DATA_PATH)` | Writes a logger message for traceability during notebook execution. |
| `gold_fit_dataframe = load_data(GOLD_FIT_DATA_PATH)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Loading Gold test parquet: %s", GOLD_TEST_DATA_PATH)` | Writes a logger message for traceability during notebook execution. |
| `gold_test_dataframe = load_data(GOLD_TEST_DATA_PATH)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Loading Gold train parquet: %s", GOLD_TRAIN_DATA_PATH)` | Writes a logger message for traceability during notebook execution. |
| `gold_train_dataframe = load_data(GOLD_TRAIN_DATA_PATH)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Loading Stage 1 features: %s", STAGE1_FEATURES_PATH)` | Writes a logger message for traceability during notebook execution. |
| `stage1_feature_columns: list[str] = require_str_list( load_json(STAGE1_FEATURES_PATH), "stage1_feature_columns",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("Loading Stage 2 features: %s", STAGE2_FEATURES_PATH)` | Writes a logger message for traceability during notebook execution. |
| `stage2_feature_columns: list[str] = require_str_list( load_json(STAGE2_FEATURES_PATH), "stage2_feature_columns",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("Loading Stage 3 primary rule sensors: %s", STAGE3_PRIMARY_PATH)` | Writes a logger message for traceability during notebook execution. |
| `stage3_primary_rule_sensors: list[str] = require_str_list( load_json(STAGE3_PRIMARY_PATH), "stage3_primary_rule_sensors",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("Loading Stage 3 secondary rule sensors: %s", STAGE3_SECONDARY_PATH)` | Writes a logger message for traceability during notebook execution. |
| `stage3_secondary_rule_sensors: list[str] = require_str_list( load_json(STAGE3_SECONDARY_PATH), "stage3_secondary_rule_sensors",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="load_modeling_inputs", message="Loaded Gold scaled parquet, loaded Gold truth, substituted truth-linked artifact paths, then loaded cascade inputs.",` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(gold_test_dataframe.head(3))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 17 — Load the Gold Modeling Inputs and Resolve the Parent Truth

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `bool`
- `cascade`
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
- `validate_cascade_row_tracking`
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
- `ledger.add( kind="step", step="validate_cascade_row_tracking", message="Validated stable row identity on Gold cascade modeling input dataframe.", data={ "row_id_column": "meta__row`: Records or exports ledger information for stage-level traceability.
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
| `ledger.add( kind="step", step="validate_cascade_row_tracking", message="Validated stable row identity on Gold cascade modeling input dataframe.", data={ "row_id_column": "meta__row` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 18 — Rebuild the Train and Test Masks

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_flag`
- `astype`
- `bool`
- `columns`
- `dtype`
- `fillna`
- `gold_preprocessed_scaled_dataframe`
- `loc`
- `meta__is_train_flag`
- `missing`
- `ndarray`
- `raise`
- `Series`
- `test_mask`
- `test_mask_array`
- `to_numpy`
- `train_mask`
- `train_mask_array`
- `ValueError`

### Outputs

- `test_labels`

### Key Operations

- `if "meta__is_train_flag" not in gold_preprocessed_scaled_dataframe.columns: raise ValueError( "meta__is_train_flag missing from gold_preprocessed_scaled_dataframe." )`: Controls validation, iteration, file handling, or error handling for this step.
- `train_mask: pd.Series = ( gold_preprocessed_scaled_dataframe["meta__is_train_flag"] .fillna(False) .astype(bool)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `test_mask: pd.Series = (~train_mask).astype(bool)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `train_mask_array: np.ndarray = train_mask.to_numpy(dtype=bool)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `test_mask_array: np.ndarray = test_mask.to_numpy(dtype=bool)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `test_labels: np.ndarray \| None = None`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns: test_labels = ( gold_preprocessed_scaled_dataframe.loc[test_mask, "anomaly_flag"] .fillna(0) .astype(int) .to_numpy`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `astype`
- `fillna`
- `to_numpy`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `if "meta__is_train_flag" not in gold_preprocessed_scaled_dataframe.columns: raise ValueError( "meta__is_train_flag missing from gold_preprocessed_scaled_dataframe." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `train_mask: pd.Series = ( gold_preprocessed_scaled_dataframe["meta__is_train_flag"] .fillna(False) .astype(bool)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `test_mask: pd.Series = (~train_mask).astype(bool)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `train_mask_array: np.ndarray = train_mask.to_numpy(dtype=bool)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `test_mask_array: np.ndarray = test_mask.to_numpy(dtype=bool)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `test_labels: np.ndarray \| None = None` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns: test_labels = ( gold_preprocessed_scaled_dataframe.loc[test_mask, "anomaly_flag"] .fillna(0) .astype(int) .to_numpy` | Controls validation, iteration, file handling, or error handling for this step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 19 — Define the Stage 3 Reference Profile Logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append`
- `DataFrame`
- `dataframe`
- `def`
- `else`
- `feature_columns`
- `feature_name`
- `isna`
- `lower_bound`
- `mean`
- `mean_value`
- `median`
- `median_value`
- `quantile`
- `reference_rows`
- `standard_deviation`
- `std`
- `upper_bound`

### Outputs

- `build_reference_profile`
- `feature_series`
- `reference_profile`

### Key Operations

- `def build_reference_profile( dataframe: pd.DataFrame, *, feature_columns: list[str],`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: reference_rows: list[dict] = [] for feature_name in feature_columns: feature_series = dataframe[feature_name] reference_rows.append({ "feature_name": feature_nam`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append`
- `build_reference_profile`
- `DataFrame`
- `isna`
- `mean`
- `median`
- `quantile`
- `std`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_reference_profile( dataframe: pd.DataFrame, *, feature_columns: list[str],` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: reference_rows: list[dict] = [] for feature_name in feature_columns: feature_series = dataframe[feature_name] reference_rows.append({ "feature_name": feature_nam` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 20 — Build the Stage 3 Reference Profile

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `build_reference_profile`
- `Built`
- `confirmation`
- `fit`
- `fromkeys`
- `gold_fit_dataframe`
- `head`
- `ledger`
- `logic`
- `period`
- `profile`
- `reference`
- `reference_feature_count`
- `Stage`
- `stage1_feature_columns`
- `stage3_primary_rule_sensors`
- `stage3_secondary_rule_sensors`
- `training_rows`

### Outputs

- `data`
- `feature_columns`
- `kind`
- `logger`
- `message`
- `reference_profile`
- `reference_profile_features`
- `step`

### Key Operations

- `reference_profile_features = list(dict.fromkeys( stage1_feature_columns + stage3_primary_rule_sensors + stage3_secondary_rule_sensors`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `reference_profile = build_reference_profile( gold_fit_dataframe, feature_columns=reference_profile_features,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="build_reference_profile", message="Built fit-period reference profile for Stage 3 confirmation logic.", data={ "training_rows": int(len(gold_fit_data`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(reference_profile.head(10))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `build_reference_profile`
- `display`
- `fromkeys`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `reference_profile_features = list(dict.fromkeys( stage1_feature_columns + stage3_primary_rule_sensors + stage3_secondary_rule_sensors` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `reference_profile = build_reference_profile( gold_fit_dataframe, feature_columns=reference_profile_features,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="build_reference_profile", message="Built fit-period reference profile for Stage 3 confirmation logic.", data={ "training_rows": int(len(gold_fit_data` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(reference_profile.head(10))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 21 — Prepare the Feature Matrices and Evaluation Labels

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `active`
- `all`
- `anomaly_flag`
- `arrays`
- `astype`
- `block`
- `Build`
- `but`
- `can`
- `cell`
- `columns`
- `construction`
- `copy`
- `DataFrames`
- `dataset`
- `Deprecated`
- `feature`
- `features`
- `fillna`
- `fit`

### Outputs

- `stage1_all_features`
- `stage1_train_fit_features`
- `stage2_all_features`
- `stage2_train_fit_features`
- `test_labels`

### Key Operations

- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Deprecated feature-matrix block removed.`: Documents the purpose or boundary of the surrounding notebook step.
- `# The active typed feature-matrix construction is handled in the following cell.`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build model feature matrices`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Keep these as DataFrames, not NumPy arrays.`: Documents the purpose or boundary of the surrounding notebook step.
- `# This prevents the sklearn warning:`: Documents the purpose or boundary of the surrounding notebook step.
- `# "X has feature names, but IsolationForest was fitted without feature names"`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# Labels can still use .values/.to_numpy().`: Documents the purpose or boundary of the surrounding notebook step.
- `# The issue is only with the feature matrix passed into IsolationForest.`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `astype`
- `copy`
- `fillna`
- `to_numpy`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Deprecated feature-matrix block removed.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The active typed feature-matrix construction is handled in the following cell.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build model feature matrices` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Keep these as DataFrames, not NumPy arrays.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# This prevents the sklearn warning:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# "X has feature names, but IsolationForest was fitted without feature names"` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Labels can still use .values/.to_numpy().` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The issue is only with the feature matrix passed into IsolationForest.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Fit features from normal-only fit parquet` | Documents the purpose or boundary of the surrounding notebook step. |
| `stage1_train_fit_features = gold_fit_dataframe.loc[:, stage1_feature_columns].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage2_train_fit_features = gold_fit_dataframe.loc[:, stage2_feature_columns].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Score features from the full scaled dataset, all rows` | Documents the purpose or boundary of the surrounding notebook step. |
| `stage1_all_features = gold_preprocessed_scaled_dataframe.loc[:, stage1_feature_columns].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage2_all_features = gold_preprocessed_scaled_dataframe.loc[:, stage2_feature_columns].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `test_labels = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns: test_labels = ( gold_preprocessed_scaled_dataframe .loc[test_mask, "anomaly_flag"] .fillna(0) .astype(int) .values ` | Controls validation, iteration, file handling, or error handling for this step. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 22 — Run validation guardrails

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_flag`
- `are`
- `astype`
- `Build`
- `column_name`
- `columns`
- `copy`
- `DataFrame`
- `dtype`
- `f`
- `feature`
- `fillna`
- `gold_fit_dataframe`
- `gold_preprocessed_scaled_dataframe`
- `loc`
- `matrices`
- `missing`
- `model`
- `n`
- `ndarray`

### Outputs

- `missing_stage1_features`
- `missing_stage1_scaled_features`
- `missing_stage2_features`
- `missing_stage2_scaled_features`
- `test_labels`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build model feature matrices`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `missing_stage1_features = [ column_name for column_name in stage1_feature_columns if column_name not in gold_fit_dataframe.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_stage2_features = [ column_name for column_name in stage2_feature_columns if column_name not in gold_fit_dataframe.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_stage1_scaled_features = [ column_name for column_name in stage1_feature_columns if column_name not in gold_preprocessed_scaled_dataframe.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_stage2_scaled_features = [ column_name for column_name in stage2_feature_columns if column_name not in gold_preprocessed_scaled_dataframe.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if missing_stage1_features: raise ValueError( "Stage 1 feature columns are missing from gold_fit_dataframe:\n" f"{missing_stage1_features[:25]}" )`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `astype`
- `copy`
- `fillna`
- `to_numpy`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build model feature matrices` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `missing_stage1_features = [ column_name for column_name in stage1_feature_columns if column_name not in gold_fit_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_stage2_features = [ column_name for column_name in stage2_feature_columns if column_name not in gold_fit_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_stage1_scaled_features = [ column_name for column_name in stage1_feature_columns if column_name not in gold_preprocessed_scaled_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_stage2_scaled_features = [ column_name for column_name in stage2_feature_columns if column_name not in gold_preprocessed_scaled_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_stage1_features: raise ValueError( "Stage 1 feature columns are missing from gold_fit_dataframe:\n" f"{missing_stage1_features[:25]}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `if missing_stage2_features: raise ValueError( "Stage 2 feature columns are missing from gold_fit_dataframe:\n" f"{missing_stage2_features[:25]}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `if missing_stage1_scaled_features: raise ValueError( "Stage 1 feature columns are missing from gold_preprocessed_scaled_dataframe:\n" f"{missing_stage1_scaled_features[:25]}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `if missing_stage2_scaled_features: raise ValueError( "Stage 2 feature columns are missing from gold_preprocessed_scaled_dataframe:\n" f"{missing_stage2_scaled_features[:25]}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `stage1_train_fit_features: pd.DataFrame = gold_fit_dataframe.loc[ :, stage1_feature_columns,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage2_train_fit_features: pd.DataFrame = gold_fit_dataframe.loc[ :, stage2_feature_columns,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage1_all_features: pd.DataFrame = gold_preprocessed_scaled_dataframe.loc[ :, stage1_feature_columns,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage2_all_features: pd.DataFrame = gold_preprocessed_scaled_dataframe.loc[ :, stage2_feature_columns,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `test_labels: np.ndarray \| None = None` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns: test_labels = ( gold_preprocessed_scaled_dataframe .loc[test_mask, "anomaly_flag"] .fillna(0) .astype(int) .to_nump` | Controls validation, iteration, file handling, or error handling for this step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 23 — Answer

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

## Code Cell 24 — Define percentile threshold selection

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

- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def choose_threshold_by_percentile( anomaly_scores: np.ndarray, percentile: float,`: Defines notebook-local logic used later in the notebook.
- `) -> float: return float(np.percentile(anomaly_scores, percentile))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `choose_threshold_by_percentile`
- `percentile`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def choose_threshold_by_percentile( anomaly_scores: np.ndarray, percentile: float,` | Defines notebook-local logic used later in the notebook. |
| `) -> float: return float(np.percentile(anomaly_scores, percentile))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 25 — Define label-based model evaluation

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
- `predicted_labels`
- `recall`
- `reshape`
- `results`
- `roc_auc`

### Outputs

- `average`
- `evaluate_against_labels`
- `zero_division`

### Key Operations

- `def evaluate_against_labels( true_labels: np.ndarray, anomaly_scores: np.ndarray, threshold: float,`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, float \| None]: true_labels_array: np.ndarray = np.asarray(true_labels, dtype=int) anomaly_scores_array: np.ndarray = np.asarray(anomaly_scores, dtype=float).reshape(`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `asarray`
- `astype`
- `average_precision_score`
- `evaluate_against_labels`
- `precision_recall_fscore_support`
- `reshape`
- `roc_auc_score`
- `unique`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def evaluate_against_labels( true_labels: np.ndarray, anomaly_scores: np.ndarray, threshold: float,` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, float \| None]: true_labels_array: np.ndarray = np.asarray(true_labels, dtype=int) anomaly_scores_array: np.ndarray = np.asarray(anomaly_scores, dtype=float).reshape(` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 26 — Run Stage 1: Broad Isolation Forest Screening

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `alert_count_all_rows`
- `alert_count_test_rows`
- `all`
- `Any`
- `as_float_array`
- `astype`
- `broad`
- `choose_threshold_value`
- `compute_anomaly_scores_isolation_forest`
- `dataset`
- `estimator_count`
- `feature_count`
- `fit`
- `Forest`
- `Gold`
- `Isolation`
- `IsolationForest`
- `ledger`
- `ndarray`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `n_estimators`
- `n_jobs`
- `random_state`
- `stage1_model`
- `step`

### Key Operations

- `stage1_model = IsolationForest( n_estimators=STAGE1_ESTIMATOR_COUNT, random_state=RANDOM_SEED, n_jobs=-1,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage1_model.fit(stage1_train_fit_features)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage1_train_scores: np.ndarray = as_float_array( compute_anomaly_scores_isolation_forest( stage1_model, stage1_train_fit_features, ), "stage1_train_scores",`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage1_all_scores: np.ndarray = as_float_array( compute_anomaly_scores_isolation_forest( stage1_model, stage1_all_features, ), "stage1_all_scores",`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage1_threshold: float = choose_threshold_value( stage1_train_scores, STAGE1_THRESHOLD_PERCENTILE,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage1_flags: np.ndarray = ( stage1_all_scores >= stage1_threshold`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `).astype(int)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage1_summary: dict[str, Any] = { "threshold_percentile": float(STAGE1_THRESHOLD_PERCENTILE), "threshold": float(stage1_threshold), "alert_count_all_rows": int(stage1_flags.sum())`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `as_float_array`
- `astype`
- `choose_threshold_value`
- `compute_anomaly_scores_isolation_forest`
- `fit`
- `IsolationForest`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage1_model = IsolationForest( n_estimators=STAGE1_ESTIMATOR_COUNT, random_state=RANDOM_SEED, n_jobs=-1,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage1_model.fit(stage1_train_fit_features)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage1_train_scores: np.ndarray = as_float_array( compute_anomaly_scores_isolation_forest( stage1_model, stage1_train_fit_features, ), "stage1_train_scores",` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage1_all_scores: np.ndarray = as_float_array( compute_anomaly_scores_isolation_forest( stage1_model, stage1_all_features, ), "stage1_all_scores",` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage1_threshold: float = choose_threshold_value( stage1_train_scores, STAGE1_THRESHOLD_PERCENTILE,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage1_flags: np.ndarray = ( stage1_all_scores >= stage1_threshold` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `).astype(int)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage1_summary: dict[str, Any] = { "threshold_percentile": float(STAGE1_THRESHOLD_PERCENTILE), "threshold": float(stage1_threshold), "alert_count_all_rows": int(stage1_flags.sum())` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="run_cascade_stage1", message="Ran Stage 1 broad Isolation Forest using saved Gold fit data and scored all rows of the scaled dataset.", data={ "estim` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 27 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `across`
- `add`
- `back`
- `build_stage_scoring_frame`
- `cascade`
- `copy`
- `full`
- `gold_preprocessed_scaled_dataframe`
- `identity`
- `ledger`
- `level`
- `merge_stage_results_back`
- `merged`
- `meta__row_id`
- `outputs`
- `population`
- `row`
- `score_isolation_forest_stage`
- `score_stage1_with_row_tracking`
- `Scored`

### Outputs

- `cascade_results`
- `data`
- `dataframe`
- `feature_columns`
- `kind`
- `logger`
- `mask`
- `master_dataframe`
- `message`
- `model`
- `row_id_column`
- `stage_dataframe`
- `stage_name`
- `stage_results_dataframe`
- `stage1_input_df`
- `stage1_results_df`
- `step`

### Key Operations

- `stage1_input_df = build_stage_scoring_frame( dataframe=gold_preprocessed_scaled_dataframe, feature_columns=stage1_feature_columns, mask=None, row_id_column="meta__row_id",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage1_results_df = score_isolation_forest_stage( stage_dataframe=stage1_input_df, model=stage1_model, feature_columns=stage1_feature_columns, stage_name="stage1", row_id_column="m`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `cascade_results = merge_stage_results_back( master_dataframe=gold_preprocessed_scaled_dataframe.copy(), stage_results_dataframe=stage1_results_df, stage_name="stage1", row_id_colum`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="score_stage1_with_row_tracking", message="Scored Stage 1 across the full cascade population and merged row-level outputs back using stable row identi`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `build_stage_scoring_frame`
- `copy`
- `merge_stage_results_back`
- `score_isolation_forest_stage`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage1_input_df = build_stage_scoring_frame( dataframe=gold_preprocessed_scaled_dataframe, feature_columns=stage1_feature_columns, mask=None, row_id_column="meta__row_id",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage1_results_df = score_isolation_forest_stage( stage_dataframe=stage1_input_df, model=stage1_model, feature_columns=stage1_feature_columns, stage_name="stage1", row_id_column="m` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results = merge_stage_results_back( master_dataframe=gold_preprocessed_scaled_dataframe.copy(), stage_results_dataframe=stage1_results_df, stage_name="stage1", row_id_colum` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="score_stage1_with_row_tracking", message="Scored Stage 1 across the full cascade population and merged row-level outputs back using stable row identi` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 28 — Build review visualization

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `after`
- `alert`
- `astype`
- `cascade_results`
- `configured`
- `count`
- `does`
- `flags`
- `ledger`
- `length`
- `loc`
- `match`
- `percentile`
- `raise`
- `results`
- `row`
- `rule`
- `scoring`
- `Stage`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `step`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Synchronize Stage 1 threshold flags after row-tracked scoring`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `if len(stage1_all_scores) != len(cascade_results): raise ValueError( "stage1_all_scores length does not match cascade_results length. " "Stage 1 row-tracked synchronization is unsa`: Controls validation, iteration, file handling, or error handling for this step.
- `cascade_results["stage1_score"] = stage1_all_scores`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `cascade_results["stage1_threshold"] = float(stage1_threshold)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `cascade_results["stage1_threshold_percentile"] = float(STAGE1_THRESHOLD_PERCENTILE)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `cascade_results["stage1_flag"] = stage1_flags.astype(int)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="synchronize_stage1_threshold_flags", message="Synchronized Stage 1 row-tracked results with the configured percentile-threshold alert rule.", data={ `: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Stage 1 threshold-synchronized alert count:", int(cascade_results["stage1_flag"].sum()))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `astype`
- `sum`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Synchronize Stage 1 threshold flags after row-tracked scoring` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `if len(stage1_all_scores) != len(cascade_results): raise ValueError( "stage1_all_scores length does not match cascade_results length. " "Stage 1 row-tracked synchronization is unsa` | Controls validation, iteration, file handling, or error handling for this step. |
| `cascade_results["stage1_score"] = stage1_all_scores` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results["stage1_threshold"] = float(stage1_threshold)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results["stage1_threshold_percentile"] = float(STAGE1_THRESHOLD_PERCENTILE)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results["stage1_flag"] = stage1_flags.astype(int)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="synchronize_stage1_threshold_flags", message="Synchronized Stage 1 row-tracked results with the configured percentile-threshold alert rule.", data={ ` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Stage 1 threshold-synchronized alert count:", int(cascade_results["stage1_flag"].sum()))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 29 — Define the Stage 2 Selection Logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `a`
- `alert_rate_test_rows`
- `Any`
- `append`
- `as_float_array`
- `asarray`
- `astype`
- `best`
- `binary`
- `bool`
- `choose_threshold_value`
- `compute_anomaly_scores_isolation_forest`
- `confusion_matrix`
- `copy`
- `DataFrame`
- `def`
- `did`
- `drop`
- `dtype`

### Outputs

- `alert_rate`
- `ascending`
- `average`
- `best_model`
- `best_result`
- `by`
- `candidate_model`
- `candidate_result`
- `evaluate_stage2_model_with_thresholds`
- `f1`
- `fn`
- `fp`
- `labels`
- `min_recall`
- `model`
- `model_candidates`
- `model_params`
- `n_jobs`
- `precision`
- `random_state`

### Key Operations

- `def evaluate_stage2_model_with_thresholds( *, model: IsolationForest, model_params: dict, stage2_train_fit_features: pd.DataFrame \| np.ndarray, stage2_all_features: pd.DataFrame \| `: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, Any]: model.fit(stage2_train_fit_features) stage2_train_scores: np.ndarray = as_float_array( compute_anomaly_scores_isolation_forest( model, stage2_train_fit_feature`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def run_stage2_selection( *, selection_mode: str, fixed_params: dict, fixed_threshold_percentile: float, threshold_grid: list[float], param_grid: dict, stage2_train_fit_features: p`: Defines notebook-local logic used later in the notebook.
- `) -> tuple[IsolationForest, dict[str, Any], pd.DataFrame]: selection_mode_clean = str(selection_mode).strip().lower() search_rows: list[dict[str, Any]] = [] best_result: dict[str, `: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append`
- `as_float_array`
- `asarray`
- `astype`
- `choose_threshold_value`
- `compute_anomaly_scores_isolation_forest`
- `confusion_matrix`
- `copy`
- `DataFrame`
- `evaluate_stage2_model_with_thresholds`
- `fit`
- `get`
- `IsolationForest`
- `lower`
- `max`
- `ParameterGrid`
- `precision_recall_fscore_support`
- `ravel`
- `reset_index`
- `run_stage2_selection`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def evaluate_stage2_model_with_thresholds( *, model: IsolationForest, model_params: dict, stage2_train_fit_features: pd.DataFrame \| np.ndarray, stage2_all_features: pd.DataFrame \| ` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, Any]: model.fit(stage2_train_fit_features) stage2_train_scores: np.ndarray = as_float_array( compute_anomaly_scores_isolation_forest( model, stage2_train_fit_feature` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def run_stage2_selection( *, selection_mode: str, fixed_params: dict, fixed_threshold_percentile: float, threshold_grid: list[float], param_grid: dict, stage2_train_fit_features: p` | Defines notebook-local logic used later in the notebook. |
| `) -> tuple[IsolationForest, dict[str, Any], pd.DataFrame]: selection_mode_clean = str(selection_mode).strip().lower() search_rows: list[dict[str, Any]] = [] best_result: dict[str, ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 30 — Run Stage 2 Selection and Keep the Best Narrow Model

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `best`
- `best_params`
- `best_stage2_result`
- `candidate_count`
- `chose`
- `configuration`
- `configured`
- `f1`
- `Forest`
- `head`
- `Isolation`
- `ledger`
- `mode`
- `model_params`
- `narrow`
- `precision`
- `Ran`
- `raw_alert_count_all_rows`
- `raw_alert_count_test_rows`

### Outputs

- `data`
- `fixed_params`
- `fixed_threshold_percentile`
- `kind`
- `logger`
- `message`
- `min_recall`
- `param_grid`
- `random_seed`
- `selection_mode`
- `stage1_flags`
- `stage2_all_features`
- `stage2_all_scores`
- `stage2_best_params`
- `stage2_flags`
- `stage2_raw_flags`
- `stage2_selected_threshold_percentile`
- `stage2_summary`
- `stage2_threshold`
- `stage2_train_fit_features`

### Key Operations

- `stage2_model, best_stage2_result, stage2_search_results = run_stage2_selection( selection_mode=STAGE2_SELECTION_MODE, fixed_params=STAGE2_FIXED_PARAMS, fixed_threshold_percentile=S`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage2_train_scores = best_stage2_result["stage2_train_scores"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stage2_all_scores = best_stage2_result["stage2_all_scores"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stage2_threshold = best_stage2_result["stage2_threshold"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stage2_raw_flags = best_stage2_result["stage2_raw_flags"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stage2_flags = best_stage2_result["stage2_flags"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stage2_selected_threshold_percentile = best_stage2_result["selected_threshold_percentile"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stage2_best_params = best_stage2_result["model_params"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stage2_summary = { "selection_mode": STAGE2_SELECTION_MODE, "selected_threshold_percentile": float(stage2_selected_threshold_percentile), "threshold": float(stage2_threshold), "bes`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="run_cascade_stage2_selection", message="Ran Stage 2 selection using the configured mode and chose the best narrow Isolation Forest configuration.", d`: Records or exports ledger information for stage-level traceability.

Important functions or methods detected:
- `add`
- `display`
- `head`
- `run_stage2_selection`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage2_model, best_stage2_result, stage2_search_results = run_stage2_selection( selection_mode=STAGE2_SELECTION_MODE, fixed_params=STAGE2_FIXED_PARAMS, fixed_threshold_percentile=S` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage2_train_scores = best_stage2_result["stage2_train_scores"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage2_all_scores = best_stage2_result["stage2_all_scores"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage2_threshold = best_stage2_result["stage2_threshold"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage2_raw_flags = best_stage2_result["stage2_raw_flags"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage2_flags = best_stage2_result["stage2_flags"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage2_selected_threshold_percentile = best_stage2_result["selected_threshold_percentile"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage2_best_params = best_stage2_result["model_params"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage2_summary = { "selection_mode": STAGE2_SELECTION_MODE, "selected_threshold_percentile": float(stage2_selected_threshold_percentile), "threshold": float(stage2_threshold), "bes` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="run_cascade_stage2_selection", message="Ran Stage 2 selection using the configured mode and chose the best narrow Isolation Forest configuration.", d` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage2_search_results.head(10))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 31 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `astype`
- `back`
- `based`
- `bool`
- `build_stage_scoring_frame`
- `candidate`
- `do`
- `dtype`
- `fillna`
- `final`
- `flag`
- `helper`
- `identity`
- `ledger`
- `left`
- `level`
- `loc`
- `logic`
- `merge`

### Outputs

- `cascade_results`
- `columns`
- `data`
- `dataframe`
- `feature_columns`
- `how`
- `kind`
- `logger`
- `mask`
- `message`
- `model`
- `on`
- `row_id_column`
- `stage_dataframe`
- `stage_name`
- `stage2_candidate_mask`
- `stage2_input_df`
- `stage2_results_df`
- `step`

### Key Operations

- `stage2_candidate_mask = ( cascade_results["stage1_flag"] .fillna(0) .astype(int) == 1`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage2_candidate_mask_array: np.ndarray = stage2_candidate_mask.to_numpy(dtype=bool)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage2_input_df = build_stage_scoring_frame( dataframe=cascade_results, feature_columns=stage2_feature_columns, mask=stage2_candidate_mask, row_id_column="meta__row_id",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage2_results_df = score_isolation_forest_stage( stage_dataframe=stage2_input_df, model=stage2_model, feature_columns=stage2_feature_columns, stage_name="stage2", row_id_column="m`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Rename raw helper outputs so they do not overwrite your threshold-based final stage2 flag logic.`: Documents the purpose or boundary of the surrounding notebook step.
- `stage2_results_df = stage2_results_df.rename( columns={ "stage2_score": "stage2_model_score", "stage2_decision": "stage2_model_decision", "stage2_pred": "stage2_model_pred", "stage`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `cascade_results = cascade_results.merge( stage2_results_df[ [ "meta__row_id", "stage2_model_score", "stage2_model_decision", "stage2_model_pred", "stage2_model_flag", ] ], on="meta`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `astype`
- `build_stage_scoring_frame`
- `fillna`
- `merge`
- `rename`
- `score_isolation_forest_stage`
- `sum`
- `to_numpy`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage2_candidate_mask = ( cascade_results["stage1_flag"] .fillna(0) .astype(int) == 1` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage2_candidate_mask_array: np.ndarray = stage2_candidate_mask.to_numpy(dtype=bool)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage2_input_df = build_stage_scoring_frame( dataframe=cascade_results, feature_columns=stage2_feature_columns, mask=stage2_candidate_mask, row_id_column="meta__row_id",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage2_results_df = score_isolation_forest_stage( stage_dataframe=stage2_input_df, model=stage2_model, feature_columns=stage2_feature_columns, stage_name="stage2", row_id_column="m` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Rename raw helper outputs so they do not overwrite your threshold-based final stage2 flag logic.` | Documents the purpose or boundary of the surrounding notebook step. |
| `stage2_results_df = stage2_results_df.rename( columns={ "stage2_score": "stage2_model_score", "stage2_decision": "stage2_model_decision", "stage2_pred": "stage2_model_pred", "stage` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results = cascade_results.merge( stage2_results_df[ [ "meta__row_id", "stage2_model_score", "stage2_model_decision", "stage2_model_pred", "stage2_model_flag", ] ], on="meta` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results["stage2_score"] = np.nan` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results.loc[stage2_candidate_mask, "stage2_score"] = stage2_all_scores[ stage2_candidate_mask_array` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results["stage2_raw_flag"] = 0` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results.loc[stage2_candidate_mask, "stage2_raw_flag"] = stage2_raw_flags[ stage2_candidate_mask_array` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results["stage2_flag"] = 0` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results.loc[stage2_candidate_mask, "stage2_flag"] = stage2_flags[ stage2_candidate_mask_array` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results["stage2_raw_flag"] = ( cascade_results["stage2_raw_flag"] .fillna(0) .astype(int)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results["stage2_flag"] = ( cascade_results["stage2_flag"] .fillna(0) .astype(int)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="score_stage2_with_row_tracking", message="Scored Stage 2 candidate rows and merged row-level Stage 2 outputs back using stable row identity.", data={` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 32 — Quick Verifications Cell

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `candidate`
- `count`
- `DataFrame`
- `else`
- `get`
- `globals`
- `head`
- `missing`
- `search`
- `stage2_best_params`
- `stage2_search_results`
- `STAGE2_SELECTION_MODE`

### Outputs

- `stage2_search_results_object`

### Key Operations

- `stage2_search_results_object = globals().get("stage2_search_results")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("STAGE2_SELECTION_MODE:", STAGE2_SELECTION_MODE)`: Displays a notebook-facing result for inspection.
- `print("stage2_best_params:", stage2_best_params)`: Displays a notebook-facing result for inspection.
- `if stage2_search_results_object is None: print("search candidate count: missing")`: Displays a notebook-facing result for inspection.
- `else: print("search candidate count:", len(stage2_search_results_object)) if len(stage2_search_results_object) > 0: display(pd.DataFrame(stage2_search_results_object).head())`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `DataFrame`
- `display`
- `get`
- `globals`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage2_search_results_object = globals().get("stage2_search_results")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("STAGE2_SELECTION_MODE:", STAGE2_SELECTION_MODE)` | Displays a notebook-facing result for inspection. |
| `print("stage2_best_params:", stage2_best_params)` | Displays a notebook-facing result for inspection. |
| `if stage2_search_results_object is None: print("search candidate count: missing")` | Displays a notebook-facing result for inspection. |
| `else: print("search candidate count:", len(stage2_search_results_object)) if len(stage2_search_results_object) > 0: display(pd.DataFrame(stage2_search_results_object).head())` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 33 — Validate That the Stage 3 Rule Sensors Exist

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `cascade_results`
- `check`
- `column`
- `columns`
- `d`
- `dataframe`
- `ensure`
- `exist`
- `info`
- `logger`
- `Missing`
- `missing`
- `primary`
- `PRIMARY`
- `rule`
- `s`
- `sanity`
- `scored`
- `secondary`
- `SECONDARY`

### Outputs

- `missing_primary`
- `missing_secondary`

### Key Operations

- `# --- Stage 3 sanity check: ensure rule sensors exist in scored dataframe`: Documents the purpose or boundary of the surrounding notebook step.
- `missing_primary = [column for column in stage3_primary_rule_sensors if column not in cascade_results.columns]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `missing_secondary = [column for column in stage3_secondary_rule_sensors if column not in cascade_results.columns]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `logger.info("Stage3 missing sensors: primary=%d secondary=%d", len(missing_primary), len(missing_secondary))`: Writes a logger message for traceability during notebook execution.
- `if missing_primary: logger.warning("Missing Stage3 PRIMARY sensors (showing up to 20): %s", missing_primary[:20])`: Writes a logger message for traceability during notebook execution.
- `if missing_secondary: logger.warning("Missing Stage3 SECONDARY sensors (showing up to 20): %s", missing_secondary[:20])`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `info`
- `sensors`
- `warning`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# --- Stage 3 sanity check: ensure rule sensors exist in scored dataframe` | Documents the purpose or boundary of the surrounding notebook step. |
| `missing_primary = [column for column in stage3_primary_rule_sensors if column not in cascade_results.columns]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `missing_secondary = [column for column in stage3_secondary_rule_sensors if column not in cascade_results.columns]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info("Stage3 missing sensors: primary=%d secondary=%d", len(missing_primary), len(missing_secondary))` | Writes a logger message for traceability during notebook execution. |
| `if missing_primary: logger.warning("Missing Stage3 PRIMARY sensors (showing up to 20): %s", missing_primary[:20])` | Writes a logger message for traceability during notebook execution. |
| `if missing_secondary: logger.warning("Missing Stage3 SECONDARY sensors (showing up to 20): %s", missing_secondary[:20])` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 34 — Define the Primary Profile Breach Logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `astype`
- `columns`
- `continue`
- `DataFrame`
- `dataframe`
- `def`
- `dtype`
- `feature_columns`
- `feature_name`
- `index`
- `loc`
- `lower_bound`
- `name`
- `reference_profile`
- `Series`
- `set_index`
- `stage3_profile_breach_count`
- `upper_bound`

### Outputs

- `breach_counts`
- `breach_flag`
- `compute_primary_breach_count`
- `lower`
- `reference_lookup`
- `upper`

### Key Operations

- `def compute_primary_breach_count( dataframe: pd.DataFrame, *, reference_profile: pd.DataFrame, feature_columns: list[str],`: Defines notebook-local logic used later in the notebook.
- `) -> pd.Series: reference_lookup = reference_profile.set_index("feature_name")[["lower_bound", "upper_bound"]] breach_counts = pd.Series(0, index=dataframe.index, dtype=int) for fe`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `compute_primary_breach_count`
- `Series`
- `set_index`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def compute_primary_breach_count( dataframe: pd.DataFrame, *, reference_profile: pd.DataFrame, feature_columns: list[str],` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.Series: reference_lookup = reference_profile.set_index("feature_name")[["lower_bound", "upper_bound"]] breach_counts = pd.Series(0, index=dataframe.index, dtype=int) for fe` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 35 — Define the Secondary Corroboration Logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `cascade_results`
- `compute_primary_breach_count`
- `stage3_primary_rule_sensors`
- `stage3_profile_breach_count`

### Outputs

- `feature_columns`
- `reference_profile`

### Key Operations

- `cascade_results["stage3_profile_breach_count"] = compute_primary_breach_count( cascade_results, reference_profile=reference_profile, feature_columns=stage3_primary_rule_sensors,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `compute_primary_breach_count`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `cascade_results["stage3_profile_breach_count"] = compute_primary_breach_count( cascade_results, reference_profile=reference_profile, feature_columns=stage3_primary_rule_sensors,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 36 — Compute the Secondary Breach Count

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `astype`
- `continue`
- `dataframe`
- `DataFrame`
- `def`
- `dtype`
- `feature_columns`
- `feature_name`
- `index`
- `name`
- `reference_profile`
- `Series`
- `set_index`
- `stage3_secondary_breach_count`
- `to_dict`

### Outputs

- `breach_counts`
- `compute_secondary_breach_count`
- `feature_breach_flag`
- `lower_bound`
- `reference_lookup`
- `upper_bound`

### Key Operations

- `def compute_secondary_breach_count( dataframe: pd.DataFrame, *, reference_profile: pd.DataFrame, feature_columns: list[str],`: Defines notebook-local logic used later in the notebook.
- `) -> pd.Series: reference_lookup = reference_profile.set_index("feature_name").to_dict("index") breach_counts = pd.Series(0, index=dataframe.index, dtype=int) for feature_name in f`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `compute_secondary_breach_count`
- `Series`
- `set_index`
- `to_dict`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def compute_secondary_breach_count( dataframe: pd.DataFrame, *, reference_profile: pd.DataFrame, feature_columns: list[str],` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.Series: reference_lookup = reference_profile.set_index("feature_name").to_dict("index") breach_counts = pd.Series(0, index=dataframe.index, dtype=int) for feature_name in f` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 37 — Compute Stage 3 secondary breach evidence

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `cascade_results`
- `compute_secondary_breach_count`
- `stage3_secondary_breach_count`
- `stage3_secondary_rule_sensors`

### Outputs

- `feature_columns`
- `reference_profile`

### Key Operations

- `cascade_results["stage3_secondary_breach_count"] = compute_secondary_breach_count( cascade_results, reference_profile=reference_profile, feature_columns=stage3_secondary_rule_senso`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `compute_secondary_breach_count`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `cascade_results["stage3_secondary_breach_count"] = compute_secondary_breach_count( cascade_results, reference_profile=reference_profile, feature_columns=stage3_secondary_rule_senso` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 38 — Define the Persistence Logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `astype`
- `def`
- `ge`
- `min_periods`
- `minimum_flags_in_window`
- `name`
- `rolling`
- `rolling_window_size`
- `Series`
- `source_flags`
- `stage3_persistence_flag`
- `sum`
- `window`

### Outputs

- `compute_persistence_flag`
- `persistence_flag`

### Key Operations

- `def compute_persistence_flag( source_flags: pd.Series, *, rolling_window_size: int = 3, minimum_flags_in_window: int = 2,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.Series: persistence_flag = ( source_flags .rolling(window=rolling_window_size, min_periods=1) .sum() .ge(minimum_flags_in_window) .astype(int) ) persistence_flag.name = "st`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `compute_persistence_flag`
- `ge`
- `rolling`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def compute_persistence_flag( source_flags: pd.Series, *, rolling_window_size: int = 3, minimum_flags_in_window: int = 2,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.Series: persistence_flag = ( source_flags .rolling(window=rolling_window_size, min_periods=1) .sum() .ge(minimum_flags_in_window) .astype(int) ) persistence_flag.name = "st` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 39 — Compute Stage 3 persistence evidence

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `cascade_results`
- `compute_persistence_flag`
- `stage2_flag`
- `STAGE3_MINIMUM_FLAGS_IN_WINDOW`
- `stage3_persistence_flag`
- `STAGE3_ROLLING_WINDOW_SIZE`

### Outputs

- `minimum_flags_in_window`
- `rolling_window_size`

### Key Operations

- `cascade_results["stage3_persistence_flag"] = compute_persistence_flag( cascade_results["stage2_flag"], rolling_window_size=STAGE3_ROLLING_WINDOW_SIZE, minimum_flags_in_window=STAGE`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `compute_persistence_flag`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `cascade_results["stage3_persistence_flag"] = compute_persistence_flag( cascade_results["stage2_flag"], rolling_window_size=STAGE3_ROLLING_WINDOW_SIZE, minimum_flags_in_window=STAGE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 40 — Define the Drift Logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abs`
- `astype`
- `continue`
- `dataframe`
- `DataFrame`
- `def`
- `drift_threshold_multiplier`
- `dtype`
- `feature_columns`
- `feature_name`
- `index`
- `isna`
- `median`
- `min_periods`
- `name`
- `rolling`
- `rolling_window_size`
- `Series`
- `stage3_drift_flag`
- `std`

### Outputs

- `compute_drift_flag`
- `drift_flag`
- `drift_trigger_counts`
- `feature_drift_flag`
- `feature_series`
- `feature_standard_deviation`
- `rolling_delta`
- `rolling_median`

### Key Operations

- `def compute_drift_flag( dataframe: pd.DataFrame, *, feature_columns: list[str], rolling_window_size: int = 5, drift_threshold_multiplier: float = 1.0,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.Series: drift_trigger_counts = pd.Series(0, index=dataframe.index, dtype=int) for feature_name in feature_columns: feature_series = dataframe[feature_name] feature_standard`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `abs`
- `astype`
- `compute_drift_flag`
- `isna`
- `median`
- `rolling`
- `Series`
- `std`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def compute_drift_flag( dataframe: pd.DataFrame, *, feature_columns: list[str], rolling_window_size: int = 5, drift_threshold_multiplier: float = 1.0,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.Series: drift_trigger_counts = pd.Series(0, index=dataframe.index, dtype=int) for feature_name in feature_columns: feature_series = dataframe[feature_name] feature_standard` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 41 — Compute the Drift Flag Inputs

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `cascade_results`
- `compute_drift_flag`
- `fromkeys`
- `stage3_drift_flag`
- `stage3_primary_rule_sensors`
- `stage3_secondary_rule_sensors`

### Outputs

- `drift_threshold_multiplier`
- `feature_columns`
- `rolling_window_size`
- `stage3_rule_watch_features`

### Key Operations

- `stage3_rule_watch_features = list(dict.fromkeys( stage3_primary_rule_sensors + stage3_secondary_rule_sensors`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `cascade_results["stage3_drift_flag"] = compute_drift_flag( cascade_results, feature_columns=stage3_rule_watch_features, rolling_window_size=5, drift_threshold_multiplier=1.0,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `compute_drift_flag`
- `fromkeys`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage3_rule_watch_features = list(dict.fromkeys( stage3_primary_rule_sensors + stage3_secondary_rule_sensors` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results["stage3_drift_flag"] = compute_drift_flag( cascade_results, feature_columns=stage3_rule_watch_features, rolling_window_size=5, drift_threshold_multiplier=1.0,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 42 — Build the Final Stage 3 Evidence Flags and Final Cascade Decision

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `all`
- `Applied`
- `astype`
- `cascade_final_flag`
- `cascade_results`
- `checks`
- `confirmation`
- `corroboration_rows_all`
- `corroboration_rows_test`
- `dataset`
- `drift_rows_all`
- `drift_rows_test`
- `final_alert_count_all_rows`
- `final_alert_count_test_rows`
- `head`
- `ledger`
- `loc`
- `of`
- `persistence_rows_all`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `stage3_summary`
- `step`

### Key Operations

- `cascade_results["stage3_profile_breach_flag"] = ( cascade_results["stage3_profile_breach_count"] >= STAGE3_MIN_PRIMARY_SENSOR_HITS`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `).astype(int)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `cascade_results["stage3_corroboration_flag"] = ( cascade_results["stage3_secondary_breach_count"] >= STAGE3_MIN_SECONDARY_SENSOR_HITS`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `).astype(int)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `cascade_results["stage3_rule_evidence_count"] = ( cascade_results["stage3_profile_breach_flag"] + cascade_results["stage3_persistence_flag"] + cascade_results["stage3_drift_flag"] `: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `cascade_results["cascade_final_flag"] = ( (cascade_results["stage1_flag"] == 1) & (cascade_results["stage2_flag"] == 1) & ( (cascade_results["stage3_profile_breach_count"] >= STAGE`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `).astype(int)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage3_summary = { "primary_rule_sensor_count": int(len(stage3_primary_rule_sensors)), "secondary_rule_sensor_count": int(len(stage3_secondary_rule_sensors)), "profile_breach_rows_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="run_cascade_stage3_confirmation", message="Applied Stage 3 confirmation checks to all rows of the scaled dataset.", data=stage3_summary, logger=logge`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `astype`
- `display`
- `head`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `cascade_results["stage3_profile_breach_flag"] = ( cascade_results["stage3_profile_breach_count"] >= STAGE3_MIN_PRIMARY_SENSOR_HITS` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `).astype(int)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results["stage3_corroboration_flag"] = ( cascade_results["stage3_secondary_breach_count"] >= STAGE3_MIN_SECONDARY_SENSOR_HITS` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `).astype(int)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results["stage3_rule_evidence_count"] = ( cascade_results["stage3_profile_breach_flag"] + cascade_results["stage3_persistence_flag"] + cascade_results["stage3_drift_flag"] ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_results["cascade_final_flag"] = ( (cascade_results["stage1_flag"] == 1) & (cascade_results["stage2_flag"] == 1) & ( (cascade_results["stage3_profile_breach_count"] >= STAGE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `).astype(int)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage3_summary = { "primary_rule_sensor_count": int(len(stage3_primary_rule_sensors)), "secondary_rule_sensor_count": int(len(stage3_secondary_rule_sensors)), "profile_breach_rows_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="run_cascade_stage3_confirmation", message="Applied Stage 3 confirmation checks to all rows of the scaled dataset.", data=stage3_summary, logger=logge` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(cascade_results.head(5))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 43 — Finalize cascade stage flag columns

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `after`
- `astype`
- `cascade`
- `cascade_final_flag`
- `cascade_final_flag_count`
- `columns`
- `else`
- `fillna`
- `finalize_stage_flag_columns`
- `finalize_stage_flags`
- `Finalized`
- `flags`
- `ledger`
- `processing`
- `rule`
- `sparse`
- `Stage`
- `stage`
- `stage1`

### Outputs

- `cascade_results`
- `data`
- `kind`
- `logger`
- `message`
- `stage_names`
- `step`

### Key Operations

- `cascade_results = finalize_stage_flag_columns( cascade_results, stage_names=["stage1", "stage2", "stage3"],`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if "stage3_confirmed_flag" in cascade_results.columns and "stage3_flag" not in cascade_results.columns: cascade_results["stage3_flag"] = cascade_results["stage3_confirmed_flag"].fi`: Controls validation, iteration, file handling, or error handling for this step.
- `ledger.add( kind="step", step="finalize_stage_flags", message="Finalized sparse cascade stage flags after Stage 3 rule processing.", data={ "stage1_flag_count": int(cascade_results`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `astype`
- `fillna`
- `finalize_stage_flag_columns`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `cascade_results = finalize_stage_flag_columns( cascade_results, stage_names=["stage1", "stage2", "stage3"],` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if "stage3_confirmed_flag" in cascade_results.columns and "stage3_flag" not in cascade_results.columns: cascade_results["stage3_flag"] = cascade_results["stage3_confirmed_flag"].fi` | Controls validation, iteration, file handling, or error handling for this step. |
| `ledger.add( kind="step", step="finalize_stage_flags", message="Finalized sparse cascade stage flags after Stage 3 rule processing.", data={ "stage1_flag_count": int(cascade_results` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 44 — Build the Main Cascade Metrics

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `Any`
- `asarray`
- `astype`
- `binary`
- `Cascade`
- `cascade_final_flag`
- `cascade_metrics`
- `cascade_results`
- `cascade_test_flags`
- `dtype`
- `f1`
- `fillna`
- `final_alert_count_all_rows`
- `final_alert_count_test_rows`
- `loc`
- `model`
- `ndarray`
- `precision`
- `precision_recall_fscore_support`

### Outputs

- `average`
- `zero_division`

### Key Operations

- `cascade_metrics: dict[str, Any] = { "model": "3-Stage Cascade", "stage1_alert_count_all_rows": int(cascade_results["stage1_flag"].sum()), "stage2_alert_count_all_rows": int(cascade`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if test_labels is not None: test_labels_array: np.ndarray = np.asarray(test_labels, dtype=int) cascade_test_flags: np.ndarray = ( cascade_results .loc[test_mask, "cascade_final_fla`: Controls validation, iteration, file handling, or error handling for this step.
- `display(cascade_metrics)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `asarray`
- `astype`
- `display`
- `fillna`
- `precision_recall_fscore_support`
- `sum`
- `to_numpy`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `cascade_metrics: dict[str, Any] = { "model": "3-Stage Cascade", "stage1_alert_count_all_rows": int(cascade_results["stage1_flag"].sum()), "stage2_alert_count_all_rows": int(cascade` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if test_labels is not None: test_labels_array: np.ndarray = np.asarray(test_labels, dtype=int) cascade_test_flags: np.ndarray = ( cascade_results .loc[test_mask, "cascade_final_fla` | Controls validation, iteration, file handling, or error handling for this step. |
| `display(cascade_metrics)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 45 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `all_rows`
- `anomaly_flag`
- `Any`
- `astype`
- `Bad`
- `binary`
- `Cascade`
- `cascade`
- `cascade_final_flag`
- `column`
- `column_name`
- `columns`
- `confusion_matrix`
- `contains`
- `copy`
- `count`
- `DataFrame`
- `dataframe`
- `def`

### Outputs

- `average`
- `bad_final_gate_rows`
- `bad_stage2_gate_rows`
- `binary_columns`
- `invalid_values`
- `labels`
- `required_columns`
- `test_dataframe`
- `validate_cascade_output`
- `zero_division`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Cascade output validation on held-out test rows`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `def validate_cascade_output( results_dataframe: pd.DataFrame, *, test_mask: pd.Series, label_column: str = "anomaly_flag", row_id_column: str = "meta__row_id", final_flag_column: s`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, Any]: required_columns = [ row_id_column, "meta__is_train_flag", "stage1_flag", "stage2_raw_flag", "stage2_flag", final_flag_column, ] for column_name in required_co`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `astype`
- `confusion_matrix`
- `copy`
- `dropna`
- `fillna`
- `precision_recall_fscore_support`
- `ravel`
- `sorted`
- `sum`
- `to_numpy`
- `unique`
- `update`
- `validate_cascade_output`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Cascade output validation on held-out test rows` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `def validate_cascade_output( results_dataframe: pd.DataFrame, *, test_mask: pd.Series, label_column: str = "anomaly_flag", row_id_column: str = "meta__row_id", final_flag_column: s` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, Any]: required_columns = [ row_id_column, "meta__is_train_flag", "stage1_flag", "stage2_raw_flag", "stage2_flag", final_flag_column, ] for column_name in required_co` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 46 — Validate cascade output columns and counts

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `cascade`
- `cascade_final_flag`
- `cascade_results`
- `DataFrame`
- `final`
- `held`
- `ledger`
- `metrics`
- `out`
- `outputs`
- `staged`
- `test`
- `validate_cascade_output`
- `Validated`

### Outputs

- `cascade_output_validation`
- `data`
- `final_flag_column`
- `kind`
- `logger`
- `message`
- `step`
- `test_mask`

### Key Operations

- `cascade_output_validation = validate_cascade_output( cascade_results, test_mask=test_mask, final_flag_column="cascade_final_flag",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="validate_cascade_output", message="Validated cascade staged outputs and final held-out test metrics.", data=cascade_output_validation, logger=logger,`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(pd.DataFrame([cascade_output_validation]))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `DataFrame`
- `display`
- `validate_cascade_output`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `cascade_output_validation = validate_cascade_output( cascade_results, test_mask=test_mask, final_flag_column="cascade_final_flag",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="validate_cascade_output", message="Validated cascade staged outputs and final held-out test metrics.", data=cascade_output_validation, logger=logger,` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(pd.DataFrame([cascade_output_validation]))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 47 — Build the Cascade Summary, Threshold Records, and Truth Artifact

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `append_truth_index`
- `artifact_paths`
- `build_truth_record`
- `cascade`
- `cascade_final_flag`
- `CASCADE_METADATA_PATH`
- `cascade_metrics`
- `CASCADE_REFERENCE_PROFILE_PATH`
- `CASCADE_RESULTS_PATH_CSV`
- `CASCADE_RESULTS_PATH_PICKLE`
- `CASCADE_SUMMARY_PATH`
- `CASCADE_THRESHOLDS_PATH`
- `cascade_tune_reference_profile_path`
- `cascade_tuned_metadata_path`
- `cascade_tuned_reference_profile_path`
- `cascade_tuned_results_path_csv`
- `cascade_tuned_results_path_pickle`
- `cascade_tuned_stage1_model_artifact_path`
- `cascade_tuned_stage1_models_path`

### Outputs

- `cascade_feature_columns`
- `cascade_meta_columns`
- `cascade_metadata`
- `cascade_process_run_id`
- `cascade_results`
- `cascade_summary`
- `cascade_thresholds`
- `cascade_truth`
- `CASCADE_TRUTH_HASH`
- `cascade_truth_layer_name`
- `cascade_truth_path`
- `cascade_truth_record`
- `column_count`
- `config_profile_value`
- `data`
- `dataset_name`
- `feature_columns`
- `final_cascade_alert_count_all_rows`
- `final_cascade_alert_count_test_rows`
- `gold_process_run_id_value`

### Key Operations

- `stage1_alert_count_all_rows = int(cascade_results["stage1_flag"].sum())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stage2_alert_count_all_rows = int(cascade_results["stage2_flag"].sum())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `final_cascade_alert_count_all_rows = int(cascade_results["cascade_final_flag"].sum())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stage1_alert_count_test_rows = int(cascade_results.loc[test_mask, "stage1_flag"].sum())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stage2_alert_count_test_rows = int(cascade_results.loc[test_mask, "stage2_flag"].sum())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `final_cascade_alert_count_test_rows = int(cascade_results.loc[test_mask, "cascade_final_flag"].sum())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `cascade_thresholds = { "cascade_variant": CASCADE_VARIANT, "stage1_threshold_percentile": float(STAGE1_THRESHOLD_PERCENTILE), "stage1_threshold": float(stage1_threshold), "stage2_s`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `cascade_summary = { "dataset_name": DATASET_NAME, "cascade_variant": CASCADE_VARIANT, "cascade_metrics": cascade_metrics, "stage1_alert_count_all_rows": stage1_alert_count_all_rows`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `truth_config_object = globals().get("TRUTH_CONFIG")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `run_mode_value = globals().get("RUN_MODE")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `append_truth_index`
- `build_truth_record`
- `display`
- `dump`
- `get`
- `get_params`
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

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage1_alert_count_all_rows = int(cascade_results["stage1_flag"].sum())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage2_alert_count_all_rows = int(cascade_results["stage2_flag"].sum())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `final_cascade_alert_count_all_rows = int(cascade_results["cascade_final_flag"].sum())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage1_alert_count_test_rows = int(cascade_results.loc[test_mask, "stage1_flag"].sum())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stage2_alert_count_test_rows = int(cascade_results.loc[test_mask, "stage2_flag"].sum())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `final_cascade_alert_count_test_rows = int(cascade_results.loc[test_mask, "cascade_final_flag"].sum())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_thresholds = { "cascade_variant": CASCADE_VARIANT, "stage1_threshold_percentile": float(STAGE1_THRESHOLD_PERCENTILE), "stage1_threshold": float(stage1_threshold), "stage2_s` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_summary = { "dataset_name": DATASET_NAME, "cascade_variant": CASCADE_VARIANT, "cascade_metrics": cascade_metrics, "stage1_alert_count_all_rows": stage1_alert_count_all_rows` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `truth_config_object = globals().get("TRUTH_CONFIG")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `run_mode_value = globals().get("RUN_MODE")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `config_profile_value = globals().get("CONFIG_PROFILE", "default")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold_process_run_id_value = globals().get("GOLD_PROCESS_RUN_ID")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if isinstance(truth_config_object, dict): truth_config_snapshot = truth_config_object` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: truth_config_snapshot = { "runtime": { "stage": "gold_cascade", "dataset": DATASET_NAME, "cascade_variant": CASCADE_VARIANT, "mode": run_mode_value, "profile": config_profile` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_truth_layer_name = "gold_cascade"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if isinstance(gold_process_run_id_value, str) and gold_process_run_id_value.strip(): cascade_process_run_id = gold_process_run_id_value` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: cascade_process_run_id = make_process_run_id("gold_cascade_process")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_truth = initialize_layer_truth( truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name=cascade_truth_layer_name, process_run_id=cascade_process_run_id, pipeline` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_truth = update_truth_section( cascade_truth, "config_snapshot", truth_config_snapshot,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_truth = update_truth_section( cascade_truth, "runtime_facts", { "cascade_variant": CASCADE_VARIANT, "stage1_threshold_percentile": float(STAGE1_THRESHOLD_PERCENTILE), "stag` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_truth = update_truth_section( cascade_truth, "artifact_paths", { "cascade_variant": CASCADE_VARIANT, "gold_truth_path": str(GOLD_TRUTH_PATH), "gold_fit_path": str(GOLD_FIT_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_meta_columns = sorted( set( identify_meta_columns(cascade_results) + [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode", ] )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_feature_columns = identify_feature_columns(cascade_results)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_truth_record = build_truth_record( truth_base=cascade_truth, row_count=len(cascade_results), column_count=cascade_results.shape[1] + 3, meta_columns=cascade_meta_columns, f` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CASCADE_TRUTH_HASH = cascade_truth_record["truth_hash"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_results = stamp_truth_columns( cascade_results, truth_hash=CASCADE_TRUTH_HASH, parent_truth_hash=GOLD_PARENT_TRUTH_HASH, pipeline_mode=PIPELINE_MODE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_truth_path = save_truth_record( cascade_truth_record, truth_dir=TRUTHS_PATH, dataset_name=DATASET_NAME, layer_name=cascade_truth_layer_name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `append_truth_index( cascade_truth_record, truth_index_path=TRUTH_INDEX_PATH,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_summary["cascade_truth_hash"] = CASCADE_TRUTH_HASH` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_summary["cascade_truth_path"] = str(cascade_truth_path)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_summary["cascade_process_run_id"] = cascade_process_run_id` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_summary["gold_truth_hash"] = GOLD_PARENT_TRUTH_HASH` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_summary["gold_truth_path"] = str(GOLD_TRUTH_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_summary["gold_process_run_id"] = gold_truth.get("process_run_id")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_summary["gold_feature_set_id"] = gold_truth_runtime_facts.get("feature_set_id")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(cascade_summary)` | Displays a notebook-facing result for inspection. |
| `cascade_metadata = { "cascade_variant": CASCADE_VARIANT, "stage2_selection_mode": STAGE2_SELECTION_MODE, "cascade_tuned_results_path_csv": str(CASCADE_RESULTS_PATH_CSV), "cascade_t` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(cascade_metadata)` | Displays a notebook-facing result for inspection. |
| `display(cascade_results)` | Displays a notebook-facing result for inspection. |
| `cascade_results.to_csv(CASCADE_RESULTS_PATH_CSV, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `cascade_results.to_pickle(CASCADE_RESULTS_PATH_PICKLE)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(reference_profile)` | Displays a notebook-facing result for inspection. |
| `reference_profile.to_csv(CASCADE_REFERENCE_PROFILE_PATH, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `joblib.dump(stage1_model, STAGE1_MODEL_ARTIFACT_PATH)` | Writes an artifact or output used for review or downstream notebooks. |
| `joblib.dump(stage1_model, STAGE1_MODELS_PATH)` | Writes an artifact or output used for review or downstream notebooks. |
| `joblib.dump(stage2_model, STAGE2_MODEL_ARTIFACT_PATH)` | Writes an artifact or output used for review or downstream notebooks. |
| `joblib.dump(stage2_model, STAGE2_MODELS_PATH)` | Writes an artifact or output used for review or downstream notebooks. |
| `save_json(cascade_thresholds, CASCADE_THRESHOLDS_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `15 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output, model artifact, optional experiment tracking call, truth record.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 48 — Build the Cascade Summary, Threshold Records, and Truth Artifact

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `alert_count_test_rows`
- `artifact`
- `artifacts`
- `based`
- `build_gold_model_output_validation_contract`
- `Cascade`
- `cascade`
- `cascade_final_flag`
- `cascade_metrics`
- `cascade_results`
- `CASCADE_RESULTS_PATH_CSV`
- `cascade_rule_artifact`
- `cascade_truth_hash`
- `CASCADE_TRUTH_HASH`
- `cascade_tuned`
- `cascade_tuned_final`
- `cascade_variant`
- `CASCADE_VARIANT`
- `downstream_consumer`

### Outputs

- `artifacts_root`
- `cascade_tuned_contract`
- `cascade_tuned_contract_path`
- `contract`
- `data`
- `dataset_id`
- `kind`
- `lineage_payload`
- `logger`
- `message`
- `metrics`
- `model_id`
- `model_label`
- `model_stage`
- `notes`
- `operating_mode`
- `output_artifact_path`
- `output_dataframe`
- `output_flag_column`
- `output_path`

### Key Operations

- `stage2_percentile_for_contract = float(stage2_selected_threshold_percentile)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `cascade_tuned_contract_path = gold_model_validation_contract_path( artifacts_root=paths.artifacts, dataset_id=DATASET_ID, model_id="cascade_tuned",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `cascade_tuned_contract = build_gold_model_output_validation_contract( dataset_id=DATASET_ID, run_id=RUN_ID, model_id="cascade_tuned", model_label="Cascade Tuned", source_notebook="`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `write_gold_model_output_validation_contract( contract=cascade_tuned_contract, output_path=cascade_tuned_contract_path,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="artifact", step="gold03b_cascade_tuned_validation_contract_written", message="Wrote Gold 03b cascade-tuned validation contract for Gold 06.", data={ "path": str(c`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(cascade_tuned_contract_path)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `build_gold_model_output_validation_contract`
- `display`
- `get`
- `globals`
- `gold_model_validation_contract_path`
- `write_gold_model_output_validation_contract`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage2_percentile_for_contract = float(stage2_selected_threshold_percentile)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_tuned_contract_path = gold_model_validation_contract_path( artifacts_root=paths.artifacts, dataset_id=DATASET_ID, model_id="cascade_tuned",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_tuned_contract = build_gold_model_output_validation_contract( dataset_id=DATASET_ID, run_id=RUN_ID, model_id="cascade_tuned", model_label="Cascade Tuned", source_notebook="` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `write_gold_model_output_validation_contract( contract=cascade_tuned_contract, output_path=cascade_tuned_contract_path,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="artifact", step="gold03b_cascade_tuned_validation_contract_written", message="Wrote Gold 03b cascade-tuned validation contract for Gold 06.", data={ "path": str(c` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(cascade_tuned_contract_path)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 49 — Build the Cascade Summary, Threshold Records, and Truth Artifact

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `anomaly_flag`
- `bool`
- `Built`
- `cascade`
- `cascade_final_flag`
- `cascade_results`
- `columns`
- `count`
- `detected`
- `detected_row_count`
- `event_step`
- `event_time`
- `extract_stage1_detected_rows`
- `f`
- `get_detected_rows_dataframe`
- `has_event_step`
- `has_event_time`
- `has_time_index`
- `head`

### Outputs

- `ascending`
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
- `stage1_detected_rows_dataframe`
- `step`
- `target_flag_column`

### Key Operations

- `stage1_detected_rows_dataframe = get_detected_rows_dataframe( dataframe=cascade_results, target_flag_column="stage1_flag", row_id_column="meta__row_id", score_column="stage1_score"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="extract_stage1_detected_rows", message="Built the Stage 1 detected-rows dataframe from the cascade results using stable row tracking.", data={ "targe`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print(f"Stage 1 detected row count: {len(stage1_detected_rows_dataframe):,}")`: Displays a notebook-facing result for inspection.
- `display(stage1_detected_rows_dataframe.head(20))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `bool`
- `display`
- `get_detected_rows_dataframe`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage1_detected_rows_dataframe = get_detected_rows_dataframe( dataframe=cascade_results, target_flag_column="stage1_flag", row_id_column="meta__row_id", score_column="stage1_score"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="extract_stage1_detected_rows", message="Built the Stage 1 detected-rows dataframe from the cascade results using stable row tracking.", data={ "targe` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"Stage 1 detected row count: {len(stage1_detected_rows_dataframe):,}")` | Displays a notebook-facing result for inspection. |
| `display(stage1_detected_rows_dataframe.head(20))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 50 — Build detected-row review extract

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `anomaly_flag`
- `bool`
- `Built`
- `cascade`
- `cascade_final_flag`
- `cascade_results`
- `columns`
- `count`
- `detected`
- `detected_row_count`
- `event_step`
- `event_time`
- `extract_stage2_detected_rows`
- `f`
- `get_detected_rows_dataframe`
- `has_event_step`
- `has_event_time`
- `has_time_index`
- `head`

### Outputs

- `ascending`
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
- `stage2_detected_rows_dataframe`
- `step`
- `target_flag_column`

### Key Operations

- `stage2_detected_rows_dataframe = get_detected_rows_dataframe( dataframe=cascade_results, target_flag_column="stage2_flag", row_id_column="meta__row_id", score_column="stage2_score"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="extract_stage2_detected_rows", message="Built the Stage 2 detected-rows dataframe from the cascade results using stable row tracking.", data={ "targe`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print(f"Stage 2 detected row count: {len(stage2_detected_rows_dataframe):,}")`: Displays a notebook-facing result for inspection.
- `display(stage2_detected_rows_dataframe.head(20))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `bool`
- `display`
- `get_detected_rows_dataframe`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage2_detected_rows_dataframe = get_detected_rows_dataframe( dataframe=cascade_results, target_flag_column="stage2_flag", row_id_column="meta__row_id", score_column="stage2_score"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="extract_stage2_detected_rows", message="Built the Stage 2 detected-rows dataframe from the cascade results using stable row tracking.", data={ "targe` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"Stage 2 detected row count: {len(stage2_detected_rows_dataframe):,}")` | Displays a notebook-facing result for inspection. |
| `display(stage2_detected_rows_dataframe.head(20))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 51 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `anomaly_flag`
- `Built`
- `cascade`
- `cascade_final_flag`
- `cascade_results`
- `count`
- `detected_row_count`
- `evidence`
- `extract_stage3_evidence_rows`
- `f`
- `get_detected_rows_dataframe`
- `head`
- `ledger`
- `meta__row_id`
- `results`
- `row`
- `stable`
- `Stage`
- `stage1_flag`

### Outputs

- `ascending`
- `data`
- `dataframe`
- `include_columns`
- `kind`
- `logger`
- `message`
- `row_id_column`
- `sort_by`
- `stage3_evidence_rows_dataframe`
- `step`
- `target_flag_column`

### Key Operations

- `stage3_evidence_rows_dataframe = get_detected_rows_dataframe( dataframe=cascade_results, target_flag_column="stage3_profile_breach_flag", row_id_column="meta__row_id", include_colu`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="extract_stage3_evidence_rows", message="Built the Stage 3 evidence-row dataframe from the cascade results using stable row tracking.", data={ "target`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print(f"Stage 3 evidence row count: {len(stage3_evidence_rows_dataframe):,}")`: Displays a notebook-facing result for inspection.
- `display(stage3_evidence_rows_dataframe.head(20))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `display`
- `get_detected_rows_dataframe`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage3_evidence_rows_dataframe = get_detected_rows_dataframe( dataframe=cascade_results, target_flag_column="stage3_profile_breach_flag", row_id_column="meta__row_id", include_colu` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="extract_stage3_evidence_rows", message="Built the Stage 3 evidence-row dataframe from the cascade results using stable row tracking.", data={ "target` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"Stage 3 evidence row count: {len(stage3_evidence_rows_dataframe):,}")` | Displays a notebook-facing result for inspection. |
| `display(stage3_evidence_rows_dataframe.head(20))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 52 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `anomaly_flag`
- `bool`
- `Built`
- `cascade`
- `cascade_final_flag`
- `cascade_results`
- `columns`
- `count`
- `detected`
- `detected_row_count`
- `event_step`
- `event_time`
- `extract_final_detected_rows`
- `f`
- `Final`
- `final`
- `get_detected_rows_dataframe`
- `has_event_step`
- `has_event_time`

### Outputs

- `ascending`
- `data`
- `dataframe`
- `decision_column`
- `final_detected_rows_dataframe`
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

- `final_detected_rows_dataframe = get_detected_rows_dataframe( dataframe=cascade_results, target_flag_column="cascade_final_flag", row_id_column="meta__row_id", score_column="stage2_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="extract_final_detected_rows", message="Built the final detected-rows dataframe from the cascade results using stable row tracking.", data={ "target_f`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print(f"Final cascade detected row count: {len(final_detected_rows_dataframe):,}")`: Displays a notebook-facing result for inspection.
- `display(final_detected_rows_dataframe.head(20))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `bool`
- `display`
- `get_detected_rows_dataframe`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `final_detected_rows_dataframe = get_detected_rows_dataframe( dataframe=cascade_results, target_flag_column="cascade_final_flag", row_id_column="meta__row_id", score_column="stage2_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="extract_final_detected_rows", message="Built the final detected-rows dataframe from the cascade results using stable row tracking.", data={ "target_f` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"Final cascade detected row count: {len(final_detected_rows_dataframe):,}")` | Displays a notebook-facing result for inspection. |
| `display(final_detected_rows_dataframe.head(20))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 53 — Finalize the Ledger and Close the Tracking Run

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `cascade`
- `cascade_ledger_path`
- `cascade_metrics`
- `CASCADE_RESULTS_PATH_CSV`
- `cascade_results_path_csv`
- `CASCADE_RESULTS_PATH_PICKLE`
- `cascade_results_path_pickle`
- `complete`
- `finalize_cascade_modeling`
- `finish`
- `Gold`
- `ledger`
- `modeling`
- `notebook`
- `save`
- `stage1_model_artifact_path`
- `STAGE1_MODEL_ARTIFACT_PATH`
- `stage1_models_path`
- `STAGE1_MODELS_PATH`

### Outputs

- `data`
- `kind`
- `logger`
- `message`
- `step`

### Key Operations

- `ledger.add( kind="step", step="finalize_cascade_modeling", message="Gold cascade modeling notebook complete.", data={ "cascade_metrics": cascade_metrics, "cascade_results_path_csv"`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.write_json(cascade_ledger_path)`: Records or exports ledger information for stage-level traceability.
- `wandb.save(str(cascade_ledger_path))`: Records or exports ledger information for stage-level traceability.
- `wandb_run.finish()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `finish`
- `save`
- `write_json`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `ledger.add( kind="step", step="finalize_cascade_modeling", message="Gold cascade modeling notebook complete.", data={ "cascade_metrics": cascade_metrics, "cascade_results_path_csv"` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.write_json(cascade_ledger_path)` | Records or exports ledger information for stage-level traceability. |
| `wandb.save(str(cascade_ledger_path))` | Records or exports ledger information for stage-level traceability. |
| `wandb_run.finish()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: optional experiment tracking call.

## Code Cell 54 — Run Final Lineage and Consistency Checks

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `Any`
- `astype`
- `Cascade`
- `cascade_metadata`
- `CASCADE_METADATA_PATH`
- `cascade_results`
- `CASCADE_TRUTH_HASH`
- `cascade_truth_hash`
- `cascade_truth_path`
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

### Outputs

- `cascade_parent_values`
- `cascade_results_truth_hash_check`
- `loaded_cascade_parent_truth_hash`
- `loaded_cascade_truth_hash`
- `missing_cascade_meta_columns`
- `required_cascade_meta_columns`
- `saved_cascade_metadata_truth_hash`

### Key Operations

- `required_cascade_meta_columns = [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_cascade_meta_columns = [ column_name for column_name in required_cascade_meta_columns if column_name not in cascade_results.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if missing_cascade_meta_columns: raise ValueError( f"cascade_results is missing required lineage columns: {missing_cascade_meta_columns}" )`: Controls validation, iteration, file handling, or error handling for this step.
- `cascade_results_truth_hash_check = extract_truth_hash(cascade_results)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if cascade_results_truth_hash_check is None: raise ValueError("cascade_results does not contain a readable meta__truth_hash value.")`: Controls validation, iteration, file handling, or error handling for this step.
- `if cascade_results_truth_hash_check != CASCADE_TRUTH_HASH: raise ValueError( "cascade_results truth hash does not match CASCADE_TRUTH_HASH:\n" f"dataframe={cascade_results_truth_ha`: Controls validation, iteration, file handling, or error handling for this step.
- `cascade_parent_values = cascade_results["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if not cascade_parent_values: raise ValueError("cascade_results is missing populated meta__parent_truth_hash values.")`: Controls validation, iteration, file handling, or error handling for this step.
- `if len(cascade_parent_values) != 1: raise ValueError(f"cascade_results has multiple parent truth hashes: {cascade_parent_values}")`: Controls validation, iteration, file handling, or error handling for this step.
- `if cascade_parent_values[0] != GOLD_PARENT_TRUTH_HASH: raise ValueError( "cascade_results parent truth hash does not match GOLD_PARENT_TRUTH_HASH:\n" f"dataframe_parent={cascade_pa`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `astype`
- `dropna`
- `exists`
- `extract_truth_hash`
- `FileNotFoundError`
- `get`
- `load_json`
- `Path`
- `require_mapping`
- `tolist`
- `unique`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `required_cascade_meta_columns = [ "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_cascade_meta_columns = [ column_name for column_name in required_cascade_meta_columns if column_name not in cascade_results.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_cascade_meta_columns: raise ValueError( f"cascade_results is missing required lineage columns: {missing_cascade_meta_columns}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `cascade_results_truth_hash_check = extract_truth_hash(cascade_results)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if cascade_results_truth_hash_check is None: raise ValueError("cascade_results does not contain a readable meta__truth_hash value.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if cascade_results_truth_hash_check != CASCADE_TRUTH_HASH: raise ValueError( "cascade_results truth hash does not match CASCADE_TRUTH_HASH:\n" f"dataframe={cascade_results_truth_ha` | Controls validation, iteration, file handling, or error handling for this step. |
| `cascade_parent_values = cascade_results["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not cascade_parent_values: raise ValueError("cascade_results is missing populated meta__parent_truth_hash values.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if len(cascade_parent_values) != 1: raise ValueError(f"cascade_results has multiple parent truth hashes: {cascade_parent_values}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `if cascade_parent_values[0] != GOLD_PARENT_TRUTH_HASH: raise ValueError( "cascade_results parent truth hash does not match GOLD_PARENT_TRUTH_HASH:\n" f"dataframe_parent={cascade_pa` | Controls validation, iteration, file handling, or error handling for this step. |
| `if not Path(cascade_truth_path).exists(): raise FileNotFoundError(f"Cascade truth file was not created: {cascade_truth_path}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `loaded_cascade_truth: dict[str, Any] = require_mapping( load_json(cascade_truth_path), "loaded_cascade_truth",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `loaded_cascade_truth_hash = loaded_cascade_truth.get("truth_hash")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if loaded_cascade_truth_hash != CASCADE_TRUTH_HASH: raise ValueError( "Saved Cascade truth file hash does not match CASCADE_TRUTH_HASH:\n" f"file={loaded_cascade_truth_hash}\n" f"e` | Controls validation, iteration, file handling, or error handling for this step. |
| `loaded_cascade_parent_truth_hash = loaded_cascade_truth.get("parent_truth_hash")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if loaded_cascade_parent_truth_hash != GOLD_PARENT_TRUTH_HASH: raise ValueError( "Saved Cascade truth file parent hash does not match GOLD_PARENT_TRUTH_HASH:\n" f"truth={loaded_cas` | Controls validation, iteration, file handling, or error handling for this step. |
| `saved_cascade_metadata: dict[str, Any] = require_mapping( load_json(CASCADE_METADATA_PATH), "saved_cascade_metadata",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `saved_cascade_metadata_truth_hash = saved_cascade_metadata.get("cascade_truth_hash")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if saved_cascade_metadata_truth_hash != CASCADE_TRUTH_HASH: raise ValueError( "cascade_metadata cascade_truth_hash does not match CASCADE_TRUTH_HASH:\n" f"metadata={saved_cascade_m` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Gold Cascade lineage sanity check passed.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 55 — Gold Cascade SQL Write Cell

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `cascade_results`
- `cascade_tuned_final`
- `else`
- `get`
- `globals`
- `Postgres`
- `skipped`
- `write`
- `write_gold_cascade_scores_sql`

### Outputs

- `capstone_schema`
- `CASCADE_SQL_MODEL_STAGE`
- `dataframe`
- `dataset_id`
- `dataset_name`
- `engine`
- `gold_cascade_sql_summary_dataframe`
- `model_stage`
- `notebook_globals`
- `run_id`
- `WRITE_TO_POSTGRES`

### Key Operations

- `WRITE_TO_POSTGRES = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CASCADE_SQL_MODEL_STAGE = "cascade_tuned_final"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if WRITE_TO_POSTGRES: gold_cascade_sql_summary_dataframe = write_gold_cascade_scores_sql( engine=engine, capstone_schema=CAPSTONE_SCHEMA, dataset_id=DATASET_ID, run_id=RUN_ID, note`: Displays a notebook-facing result for inspection.
- `else: print("Postgres write skipped.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `get`
- `globals`
- `write_gold_cascade_scores_sql`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `WRITE_TO_POSTGRES = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CASCADE_SQL_MODEL_STAGE = "cascade_tuned_final"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if WRITE_TO_POSTGRES: gold_cascade_sql_summary_dataframe = write_gold_cascade_scores_sql( engine=engine, capstone_schema=CAPSTONE_SCHEMA, dataset_id=DATASET_ID, run_id=RUN_ID, note` | Displays a notebook-facing result for inspection. |
| `else: print("Postgres write skipped.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

