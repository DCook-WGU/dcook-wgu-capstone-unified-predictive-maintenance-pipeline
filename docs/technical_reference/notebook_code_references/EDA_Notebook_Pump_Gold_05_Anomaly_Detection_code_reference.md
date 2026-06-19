# Notebook Code Reference: EDA_Notebook_Pump_Gold_05_Anomaly_Detection

Notebook path:

`notebooks/experiments/EDA_Notebook_Pump_Gold_05_Anomaly_Detection.ipynb`

## Notebook Purpose

This notebook builds anomaly timeline, alert review, and early-warning interpretation artifacts from selected Gold outputs.

Notebook stage:

`Gold`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Answer | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 14, Code Cell 15, Code Cell 17, Code Cell 25, Code Cell 28, Code Cell 30, Code Cell 32, Code Cell 33, Code Cell 34, Code Cell 35, Code Cell 36, Code Cell 37, Code Cell 38, Code Cell 43, Code Cell 49, Code Cell 52, Code Cell 56, Code Cell 59, Code Cell 61, Code Cell 64, Code Cell 66, Code Cell 67, Code Cell 69, Code Cell 70, Code Cell 71, Code Cell 72 |
| Configurables | Code Cell 07, Code Cell 08 |
| Review intermediate output | Code Cell 09 |
| Start Logging for the Anomaly Detection Stage | Code Cell 10, Code Cell 11 |
| Initialize the Anomaly Detection Ledger | Code Cell 12, Code Cell 13 |
| Load selected model results for Gold 05 | Code Cell 16 |
| Define row order alignment helper | Code Cell 18, Code Cell 26, Code Cell 44 |
| Define broken-state anchor columns | Code Cell 19 |
| Define alert and normal-like row flags | Code Cell 20 |
| Define forward stable-normal run lengths | Code Cell 21 |
| Define recovery boundary detection | Code Cell 22 |
| Define episode-phase labeling | Code Cell 23 |
| Define detection class labels | Code Cell 24 |
| Build the final anomaly timeline | Code Cell 27 |
| Create Gold 05 anomaly timeline output | Code Cell 29, Code Cell 42, Code Cell 51, Code Cell 55, Code Cell 63, Code Cell 65, Code Cell 68 |
| Define first-alert lookup helper | Code Cell 31 |
| Define early-warning summary payload | Code Cell 39, Code Cell 40 |
| Define alert packet summary table | Code Cell 41 |
| Define per-run detection summary payload | Code Cell 45, Code Cell 46 |
| Define comparison-window center logic | Code Cell 47 |
| Define comparison-window extraction | Code Cell 48 |
| Define focused anomaly timeline plot | Code Cell 50 |
| Define display-only sensor normalization | Code Cell 53, Code Cell 54 |
| Define model lead-time comparison table | Code Cell 57 |
| Multi-Run Lead-Time Comparison Chart | Code Cell 58 |
| Generate focused anomaly timeline plot | Code Cell 60 |
| Define all-sensor heatmap plot | Code Cell 62 |
| Gold 05 Anomaly Detection SQL Summary Cell | Code Cell 73, Code Cell 74, Code Cell 75 |

## Code Cell 01 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__future__`
- `already`
- `annotations`
- `Any`
- `append_truth_index`
- `artifact_file_path`
- `artifacts`
- `axes3d`
- `Axes3D`
- `build_artifact_dirs`
- `build_artifact_dirs_from_config`
- `build_truth_config_block`
- `build_truth_record`
- `cascade_row_tracking`
- `cast`
- `columns`
- `config_loader`
- `configure_logging`
- `core`
- `database`

### Outputs

- `get_stage_detected_rows_dataframe`

### Key Operations

- `from __future__ import annotations`: Imports a dependency or project helper used by later cells.
- `from pathlib import Path`: Imports a dependency or project helper used by later cells.
- `from typing import Optional, Sequence, Any, Dict, Mapping, Final, Literal, cast`: Imports a dependency or project helper used by later cells.
- `import os`: Imports a dependency or project helper used by later cells.
- `import json`: Imports a dependency or project helper used by later cells.
- `import logging`: Imports a dependency or project helper used by later cells.
- `import numpy as np`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `import matplotlib.pyplot as plt`: Imports a dependency or project helper used by later cells.
- `from matplotlib.figure import Figure`: Imports a dependency or project helper used by later cells.
- `from mpl_toolkits.mplot3d.axes3d import Axes3D`: Imports a dependency or project helper used by later cells.
- `from mpl_toolkits.mplot3d import Axes3D # noqa: F401`: Imports a dependency or project helper used by later cells.

Important functions or methods detected:
- `import`
- `set_option`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `from __future__ import annotations` | Imports a dependency or project helper used by later cells. |
| `from pathlib import Path` | Imports a dependency or project helper used by later cells. |
| `from typing import Optional, Sequence, Any, Dict, Mapping, Final, Literal, cast` | Imports a dependency or project helper used by later cells. |
| `import os` | Imports a dependency or project helper used by later cells. |
| `import json` | Imports a dependency or project helper used by later cells. |
| `import logging` | Imports a dependency or project helper used by later cells. |
| `import numpy as np` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `import matplotlib.pyplot as plt` | Imports a dependency or project helper used by later cells. |
| `from matplotlib.figure import Figure` | Imports a dependency or project helper used by later cells. |
| `from mpl_toolkits.mplot3d.axes3d import Axes3D` | Imports a dependency or project helper used by later cells. |
| `from mpl_toolkits.mplot3d import Axes3D # noqa: F401` | Imports a dependency or project helper used by later cells. |
| `from mpl_toolkits.mplot3d.axes3d import Axes3D` | Imports a dependency or project helper used by later cells. |
| `from matplotlib.ticker import MaxNLocator` | Imports a dependency or project helper used by later cells. |
| `from utils.core.paths import get_paths` | Imports a dependency or project helper used by later cells. |
| `from utils.core.logging_setup import configure_logging, log_layer_paths` | Imports a dependency or project helper used by later cells. |
| `from utils.core.ledger import Ledger` | Imports a dependency or project helper used by later cells. |
| `from utils.core.file_io import ( load_data, save_data, save_json,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.config_loader import ( load_pipeline_config, build_truth_config_block, set_wandb_dir_from_config, export_config_snapshot,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.truths import ( identify_meta_columns, identify_feature_columns, extract_truth_hash, initialize_layer_truth, update_truth_section, build_truth_record, save_truth_re` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# optional existing helper module if you already saved it into utils or notebook path` | Documents the purpose or boundary of the surrounding notebook step. |
| `try: from utils.medallion.gold.cascade_row_tracking import get_stage_detected_rows_dataframe` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `except Exception: get_stage_detected_rows_dataframe = None` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
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
| `pd.set_option("display.max_rows", 200)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

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
- `before`
- `cast`
- `converting`
- `could`
- `def`
- `f`
- `got`
- `Helpers`
- `integer`
- `isinstance`
- `it`
- `Mapping`
- `mapping`
- `must`
- `name`
- `nullable`

### Outputs

- `cfg_optional_mapping`
- `cfg_require_mapping`
- `require_int_value`

### Key Operations

- `# --------------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# Helpers`: Documents the purpose or boundary of the surrounding notebook step.
- `# --------------------------------------------------------------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `def require_int_value(value: int \| None, name: str) -> int: """ Validate that a nullable integer value is present before converting it. """ if value is None: raise ValueError(f"{na`: Defines notebook-local logic used later in the notebook.
- `def cfg_require_mapping(value: object, name: str) -> Mapping[str, Any]: if not isinstance(value, Mapping): raise TypeError( f"{name} must be a mapping, got {type(value).__name__}: `: Defines notebook-local logic used later in the notebook.
- `def cfg_optional_mapping(value: object \| None, name: str) -> Mapping[str, Any]: if value is None: return {} return cfg_require_mapping(value, name)`: Defines notebook-local logic used later in the notebook.

Important functions or methods detected:
- `cast`
- `cfg_optional_mapping`
- `cfg_require_mapping`
- `isinstance`
- `require_int_value`
- `type`
- `TypeError`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# --------------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Helpers` | Documents the purpose or boundary of the surrounding notebook step. |
| `# --------------------------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `def require_int_value(value: int \| None, name: str) -> int: """ Validate that a nullable integer value is present before converting it. """ if value is None: raise ValueError(f"{na` | Defines notebook-local logic used later in the notebook. |
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
- `anomaly_detection`
- `capstone`
- `context`
- `context_loaded`
- `dataset_config`
- `default`
- `execution`
- `gold`
- `gold_anomaly_detection`
- `info`
- `load_notebook_context`
- `loaded`
- `Loaded`
- `log`
- `LOG_PATH`
- `log_path`
- `logger`
- `logger_child_name`

### Outputs

- `ANOMALY_DETECTION_CFG`
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
- `CONTEXT_STAGE = "gold_anomaly_detection"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_DATASET = "pump"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LAYER = "gold"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_RUN_MODE = "train"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_PROFILE = "default"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LOG_FILE = "gold_anomaly_detection.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.gold.anomaly_detection", log_fi`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `CONTEXT_STAGE = "gold_anomaly_detection"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LAYER = "gold"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_RUN_MODE = "train"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LOG_FILE = "gold_anomaly_detection.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.gold.anomaly_detection", log_fi` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Shared aliases used throughout the notebook` | Documents the purpose or boundary of the surrounding notebook step. |
| `paths = CTX.paths` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_MAP = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE_CFG = CTX.stage_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ANOMALY_DETECTION_CFG = CTX.stage_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
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

## Code Cell 04 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `alias`
- `ANOMALY_DETECTION_CFG`
- `Any`
- `artifacts`
- `Artifacts`
- `B`
- `Backward`
- `Base`
- `build_truth_config_block`
- `capstone`
- `cast`
- `cells`
- `Comparison`
- `comparison`
- `comparison_plot_with_test_alerts`
- `compatible`
- `CONFIG`
- `CONFIG_RUN_MODE`
- `csv`
- `dataset`

### Outputs

- `ARTIFACTS_ROOT`
- `COMPARISON_PLOT_WITH_TEST_ALERTS_FILE_NAME`
- `COMPARISON_PLOT_WITH_TEST_ALERTS_PATH`
- `CONFIG_PROFILE`
- `DATASET_NAME`
- `DATASET_NAME_CONFIG`
- `GOLD_ANOMALY_DETECTION_LEDGER_FILE_NAME`
- `GOLD_ARTIFACTS_PATH`
- `GOLD_PROCESS_RUN_ID`
- `GOLD_VERSION`
- `LAYER_NAME`
- `LOGS_PATH`
- `MODEL_COMPARISON_FILE_NAME`
- `MODEL_COMPARISON_PATH`
- `MODEL_COMPARISON_SUMMARY_FILE_NAME`
- `MODEL_COMPARISON_SUMMARY_PATH`
- `PIPELINE_MODE`
- `PROCESS_RUN_ID`
- `RECIPE_ID`
- `RUN_MODE`

### Key Operations

- `TRUTH_CONFIG = cast(Dict[str, Any], build_truth_config_block(CONFIG))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_CONFIG["pipeline"] = PIPELINE`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `TRUTH_CONFIG["stage_params"] = ANOMALY_DETECTION_CFG`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `# ---- Stage details ----`: Documents the purpose or boundary of the surrounding notebook step.
- `STAGE = "gold"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `LAYER_NAME = str(ANOMALY_DETECTION_CFG["layer_name"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `GOLD_VERSION = str(VERSIONS_CFG["gold"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_VERSION = str(VERSIONS_CFG["truth"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RECIPE_ID = str(ANOMALY_DETECTION_CFG["recipe_id"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `TRUTH_CONFIG["stage_params"] = ANOMALY_DETECTION_CFG` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Stage details ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `STAGE = "gold"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LAYER_NAME = str(ANOMALY_DETECTION_CFG["layer_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_VERSION = str(VERSIONS_CFG["gold"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_VERSION = str(VERSIONS_CFG["truth"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RECIPE_ID = str(ANOMALY_DETECTION_CFG["recipe_id"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `PIPELINE_MODE = str(PIPELINE["execution_mode"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_MODE = str(RUNTIME_CFG.get("mode", CONFIG_RUN_MODE))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = str(RUNTIME_CFG.get("profile", CONFIG_PROFILE))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `DATASET_NAME_CONFIG = str(DATASET_CFG.get("name", "pump"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_NAME = DATASET_NAME_CONFIG.strip().lower()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `GOLD_PROCESS_RUN_ID = make_process_run_id( str( ANOMALY_DETECTION_CFG.get( "process_run_id_prefix", "gold05_anomaly_detection", ) )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Backward-compatible alias for downstream cells.` | Documents the purpose or boundary of the surrounding notebook step. |
| `PROCESS_RUN_ID = GOLD_PROCESS_RUN_ID` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- W&B ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `WANDB_PROJECT = str(WANDB_CFG.get("project", "capstone"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WANDB_ENTITY = str(WANDB_CFG.get("entity", ""))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `WANDB_RUN_NAME = f"{GOLD_VERSION}"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- File names ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `GOLD_ANOMALY_DETECTION_LEDGER_FILE_NAME = str( FILENAMES.get( "gold_anomaly_detection_ledger_file_name", "gold_anomaly_detection_ledger.jsonl", )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `MODEL_COMPARISON_FILE_NAME = str( FILENAMES.get("model_comparison_file_name", "model_comparison.csv")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `MODEL_COMPARISON_SUMMARY_FILE_NAME = str( FILENAMES.get( "model_comparison_summary_file_name", "model_comparison_summary.json", )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `COMPARISON_PLOT_WITH_TEST_ALERTS_FILE_NAME = str( FILENAMES.get( "comparison_plot_with_test_alerts_file_name", "comparison_plot_with_test_alerts.png", )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---- Base paths only ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `ARTIFACTS_ROOT = Path(str(RESOLVED_PATHS["artifacts_root"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_ARTIFACTS_PATH = Path( str(RESOLVED_PATHS.get("gold_artifacts_dir", ARTIFACTS_ROOT / "gold"))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `MODEL_COMPARISON_PATH = Path( str( RESOLVED_PATHS.get( "model_comparison_path", GOLD_ARTIFACTS_PATH / MODEL_COMPARISON_FILE_NAME, ) )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `MODEL_COMPARISON_SUMMARY_PATH = Path( str( RESOLVED_PATHS.get( "model_comparison_summary_path", GOLD_ARTIFACTS_PATH / MODEL_COMPARISON_SUMMARY_FILE_NAME, ) )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `COMPARISON_PLOT_WITH_TEST_ALERTS_PATH = Path( str( RESOLVED_PATHS.get( "comparison_plot_with_test_alerts_path", GOLD_ARTIFACTS_PATH / COMPARISON_PLOT_WITH_TEST_ALERTS_FILE_NAME, ) ` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `TRUTHS_PATH = Path(str(RESOLVED_PATHS["truths_dir"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_INDEX_PATH = Path(str(RESOLVED_PATHS["truth_index_path"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LOGS_PATH = Path(str(RESOLVED_PATHS["logs_root"]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# W&B` | Documents the purpose or boundary of the surrounding notebook step. |
| `set_wandb_dir_from_config(CONFIG)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Path failsafes` | Documents the purpose or boundary of the surrounding notebook step. |
| `ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `MODEL_COMPARISON_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `11 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

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

- `ANOMALY_DETECTION_CFG`
- `check`
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

- `gold_required_context_vars = [ "ANOMALY_DETECTION_CFG",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `gold_required_context_vars = [ "ANOMALY_DETECTION_CFG",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_gold_context_vars = [ name for name in gold_required_context_vars if name not in globals()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_gold_context_vars: raise NameError(f"Missing Gold context variables: {missing_gold_context_vars}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `logger.info("Gold context sanity check passed")` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 07 — Configurables

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `baseline`
- `Baseline`
- `baseline_decision`
- `baseline_flag`
- `baseline_score`
- `cascade`
- `Cascade`
- `cascade_defaults`
- `cascade_final_flag`
- `cascade_stage3_medium_flag`
- `cascade_stage3_strict_flag`
- `cascade_tuned`
- `Defaults`
- `event_step`
- `event_time`
- `f`
- `Forest`
- `get`
- `Improved`
- `Isolation`

### Outputs

- `ALERT_PACKET_MAX_GAP_ROWS`
- `DEFAULT_SENSOR_FOR_TIMELINE`
- `PLOT_ALERT_MARKER_SIZE`
- `PLOT_ALERT_SPAN_ALPHA`
- `PLOT_EVENT_ALPHA`
- `PLOT_GRID_ALPHA`
- `PLOT_RUN_LABEL`
- `PLOT_RUN_LABEL_MAP`
- `PLOT_SENSOR_CLIP_VALUE`
- `PLOT_SENSOR_NORMALIZATION_METHOD`
- `PLOT_WINDOW_AFTER_CENTER`
- `PLOT_WINDOW_BEFORE_CENTER`
- `PRIMARY_DECISION_COLUMN`
- `PRIMARY_SCORE_COLUMN`
- `RECOVERY_STABILITY_ROWS`
- `RECOVERY_STARTS_AFTER_BROKEN`
- `ROW_ID_COLUMN`
- `RUN_CONFIG_MAP`
- `RUN_FAMILY`
- `SELECTED_RUN_CONFIG`

### Key Operations

- `SELECTED_RUN_KEY = "stage3_improved"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# valid:`: Documents the purpose or boundary of the surrounding notebook step.
- `# baseline`: Documents the purpose or boundary of the surrounding notebook step.
- `# cascade_defaults`: Documents the purpose or boundary of the surrounding notebook step.
- `# cascade_tuned`: Documents the purpose or boundary of the surrounding notebook step.
- `# stage3_improved`: Documents the purpose or boundary of the surrounding notebook step.
- `USE_GLOBAL_RESULTS_IF_AVAILABLE = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RECOVERY_STABILITY_ROWS = 30`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RECOVERY_STARTS_AFTER_BROKEN = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ROW_ID_COLUMN = "meta__row_id"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TIME_AXIS_CANDIDATES = ["time_index", "event_step", "event_time"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `STATUS_COLUMN_CANDIDATES = ["machine_status", "status", "state"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `SELECTED_RUN_KEY = "stage3_improved"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# valid:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# baseline` | Documents the purpose or boundary of the surrounding notebook step. |
| `# cascade_defaults` | Documents the purpose or boundary of the surrounding notebook step. |
| `# cascade_tuned` | Documents the purpose or boundary of the surrounding notebook step. |
| `# stage3_improved` | Documents the purpose or boundary of the surrounding notebook step. |
| `USE_GLOBAL_RESULTS_IF_AVAILABLE = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RECOVERY_STABILITY_ROWS = 30` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RECOVERY_STARTS_AFTER_BROKEN = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ROW_ID_COLUMN = "meta__row_id"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TIME_AXIS_CANDIDATES = ["time_index", "event_step", "event_time"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STATUS_COLUMN_CANDIDATES = ["machine_status", "status", "state"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DEFAULT_SENSOR_FOR_TIMELINE = "sensor_00"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PLOT_WINDOW_BEFORE_CENTER = 300` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PLOT_WINDOW_AFTER_CENTER = 300` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PLOT_ALERT_MARKER_SIZE = 18` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ALERT_PACKET_MAX_GAP_ROWS = 5` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PLOT_SENSOR_NORMALIZATION_METHOD = "robust_zscore"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PLOT_SENSOR_CLIP_VALUE = 5.0` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PLOT_GRID_ALPHA = 0.25` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PLOT_EVENT_ALPHA = 0.85` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PLOT_ALERT_SPAN_ALPHA = 0.08` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PLOT_RUN_LABEL_MAP = { "baseline": "Baseline Isolation Forest", "cascade_defaults": "Cascade Defaults", "cascade_tuned": "Cascade Tuned", "stage3_improved": "Stage 3 Improved", "st` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `RUN_CONFIG_MAP = { "baseline": { "target_flag_column": "baseline_flag", "primary_score_column": "baseline_score", "primary_decision_column": "baseline_decision", "run_family": "bas` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if SELECTED_RUN_KEY not in RUN_CONFIG_MAP: raise ValueError( f"Unsupported SELECTED_RUN_KEY: {SELECTED_RUN_KEY}. " f"Valid keys: {list(RUN_CONFIG_MAP)}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `PLOT_RUN_LABEL = PLOT_RUN_LABEL_MAP.get(SELECTED_RUN_KEY, SELECTED_RUN_KEY)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SELECTED_RUN_CONFIG = RUN_CONFIG_MAP[SELECTED_RUN_KEY]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TARGET_FLAG_COLUMN = SELECTED_RUN_CONFIG["target_flag_column"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PRIMARY_SCORE_COLUMN = SELECTED_RUN_CONFIG["primary_score_column"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PRIMARY_DECISION_COLUMN = SELECTED_RUN_CONFIG["primary_decision_column"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_FAMILY = SELECTED_RUN_CONFIG["run_family"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 08 — Configurables

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

## Code Cell 10 — Start Logging for the Anomaly Detection Stage

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

## Code Cell 11 — Start Logging for the Anomaly Detection Stage

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_detection`
- `capstone`
- `configure_logging`
- `Create`
- `current_layer`
- `DEBUG`
- `file`
- `getLogger`
- `Gold`
- `gold`
- `gold_anomaly_detection`
- `info`
- `Initial`
- `Initiate`
- `initiation`
- `load`
- `loads`
- `Log`
- `log`
- `log_layer_paths`

### Outputs

- `gold_log_path`
- `level`
- `logger`
- `overwrite_handlers`

### Key Operations

- `"""`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Original Logging Setup`: Documents the purpose or boundary of the surrounding notebook step.
- `# Create gold log path`: Documents the purpose or boundary of the surrounding notebook step.
- `gold_log_path = paths.logs / "gold_anomaly_detection.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Initial Logger`: Documents the purpose or boundary of the surrounding notebook step.
- `configure_logging( "capstone", gold_log_path, level=logging.DEBUG, overwrite_handlers=True,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Initiate Logger and log file`: Documents the purpose or boundary of the surrounding notebook step.
- `logger = logging.getLogger("capstone.gold.anomaly_detection")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
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
| `gold_log_path = paths.logs / "gold_anomaly_detection.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Initial Logger` | Documents the purpose or boundary of the surrounding notebook step. |
| `configure_logging( "capstone", gold_log_path, level=logging.DEBUG, overwrite_handlers=True,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Initiate Logger and log file` | Documents the purpose or boundary of the surrounding notebook step. |
| `logger = logging.getLogger("capstone.gold.anomaly_detection")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Log load and initiation` | Documents the purpose or boundary of the surrounding notebook step. |
| `logger.info("Gold Modeling stage starting")` | Writes a logger message for traceability during notebook execution. |
| `# Log paths loads` | Documents the purpose or boundary of the surrounding notebook step. |
| `log_layer_paths(paths, current_layer="gold", logger=logger) """` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 12 — Initialize the Anomaly Detection Ledger

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

## Code Cell 13 — Initialize the Anomaly Detection Ledger

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `align`
- `anomaly`
- `architecture`
- `based`
- `capstone`
- `config`
- `dataset_name`
- `DATASET_NAME`
- `decision`
- `detection`
- `exports`
- `follow`
- `Gold`
- `init`
- `Initialized`
- `loading`
- `logging`
- `logs`
- `notebook`

### Outputs

- `consequence`
- `data`
- `kind`
- `ledger`
- `logger`
- `message`
- `recipe_id`
- `stage`
- `step`
- `why`

### Key Operations

- `"""`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Original Ledger Setup`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger = Ledger( stage=STAGE, recipe_id=PROCESS_RUN_ID)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ledger.add( kind="step", step="init", message="Initialized ledger", logger=logger`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="decision", step="notebook_init", message="Initialized Gold 05 anomaly detection notebook with utils-based paths/config/logging.", why="Gold 05 should align with t`: Records or exports ledger information for stage-level traceability.
- `) """`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `Ledger`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `"""` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Original Ledger Setup` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger = Ledger( stage=STAGE, recipe_id=PROCESS_RUN_ID)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger.add( kind="step", step="init", message="Initialized ledger", logger=logger` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="decision", step="notebook_init", message="Initialized Gold 05 anomaly detection notebook with utils-based paths/config/logging.", why="Gold 05 should align with t` | Records or exports ledger information for stage-level traceability. |
| `) """` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 14 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold_anomaly_detection`
- `__gold_anomaly_detection__resolved_config`
- `__name__`
- `a`
- `add`
- `after`
- `anomaly`
- `are`
- `artifact`
- `artifacts`
- `baseline`
- `baseline_results_file_name_pickle`
- `be`
- `build_artifact_dirs`
- `build_artifact_dirs_from_config`
- `built`
- `but`
- `by`
- `can`
- `cascade_defaults`

### Outputs

- `ANOMALY_DETECTION_ARTIFACT_DIR`
- `ANOMALY_DETECTION_ARTIFACT_DIRS`
- `ANOMALY_DETECTION_CONFIG_DIR`
- `ANOMALY_DETECTION_EXPORT_DIR`
- `ANOMALY_DETECTION_LINEAGE_DIR`
- `ANOMALY_DETECTION_METADATA_DIR`
- `ANOMALY_DETECTION_PACKET_DIR`
- `ANOMALY_DETECTION_PLOT_DIR`
- `ANOMALY_DETECTION_SUMMARY_DIR`
- `ARTIFACTS_ROOT`
- `artifacts_root`
- `config`
- `CONFIG_SNAPSHOT_PATH`
- `consequence`
- `data`
- `dataset_name`
- `family`
- `GOLD_BASELINE_ARTIFACT_DIRS`
- `GOLD_CASCADE_DEFAULTS_ARTIFACT_DIRS`
- `GOLD_CASCADE_STAGE3_ARTIFACT_DIRS`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Gold 05 artifact directories`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Gold 05 reads scored outputs from Gold 02 / 03A / 03B / 03C`: Documents the purpose or boundary of the surrounding notebook step.
- `# and writes its own anomaly-detection outputs.`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# The Gold 05 CONFIG owns gold_anomaly_detection, but it may not contain`: Documents the purpose or boundary of the surrounding notebook step.
- `# the other Gold notebook stage keys. Therefore:`: Documents the purpose or boundary of the surrounding notebook step.
- `# - prior Gold model output directories are built directly`: Documents the purpose or boundary of the surrounding notebook step.
- `# - Gold 05's own output directory is built from CONFIG`: Documents the purpose or boundary of the surrounding notebook step.
- `raw_resolved_paths = CONFIG.get("resolved_paths", {})`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if not isinstance(raw_resolved_paths, dict): raise TypeError( "CONFIG['resolved_paths'] must be a dictionary. " f"Got {type(raw_resolved_paths).__name__}: {raw_resolved_paths!r}" )`: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `add`
- `build_artifact_dirs`
- `build_artifact_dirs_from_config`
- `export_config_snapshot`
- `get`
- `info`
- `isinstance`
- `KeyError`
- `Path`
- `type`
- `TypeError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Gold 05 artifact directories` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Gold 05 reads scored outputs from Gold 02 / 03A / 03B / 03C` | Documents the purpose or boundary of the surrounding notebook step. |
| `# and writes its own anomaly-detection outputs.` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The Gold 05 CONFIG owns gold_anomaly_detection, but it may not contain` | Documents the purpose or boundary of the surrounding notebook step. |
| `# the other Gold notebook stage keys. Therefore:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - prior Gold model output directories are built directly` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - Gold 05's own output directory is built from CONFIG` | Documents the purpose or boundary of the surrounding notebook step. |
| `raw_resolved_paths = CONFIG.get("resolved_paths", {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not isinstance(raw_resolved_paths, dict): raise TypeError( "CONFIG['resolved_paths'] must be a dictionary. " f"Got {type(raw_resolved_paths).__name__}: {raw_resolved_paths!r}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `raw_artifacts_root = raw_resolved_paths.get("artifacts_root")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if raw_artifacts_root is None: raise KeyError( "Could not resolve artifacts root. " "Expected CONFIG['resolved_paths']['artifacts_root']." )` | Controls validation, iteration, file handling, or error handling for this step. |
| `ARTIFACTS_ROOT = Path(raw_artifacts_root)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_MODEL_SUBDIRS = [ "scores", "summaries", "thresholds", "metadata", "models", "plots", "config", "lineage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_BASELINE_ARTIFACT_DIRS = build_artifact_dirs( artifacts_root=ARTIFACTS_ROOT, stage="gold", dataset_name=DATASET_NAME, family="baseline", subdirs=GOLD_MODEL_SUBDIRS,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_CASCADE_DEFAULTS_ARTIFACT_DIRS = build_artifact_dirs( artifacts_root=ARTIFACTS_ROOT, stage="gold", dataset_name=DATASET_NAME, family="cascade_defaults", subdirs=GOLD_MODEL_SUB` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_CASCADE_TUNED_ARTIFACT_DIRS = build_artifact_dirs( artifacts_root=ARTIFACTS_ROOT, stage="gold", dataset_name=DATASET_NAME, family="cascade_tuned", subdirs=GOLD_MODEL_SUBDIRS,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD_CASCADE_STAGE3_ARTIFACT_DIRS = build_artifact_dirs( artifacts_root=ARTIFACTS_ROOT, stage="gold", dataset_name=DATASET_NAME, family="cascade_stage3_improved", subdirs=GOLD_MODE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ANOMALY_DETECTION_ARTIFACT_DIRS = build_artifact_dirs_from_config( config=CONFIG, stage_key="gold_anomaly_detection",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ANOMALY_DETECTION_ARTIFACT_DIR = ANOMALY_DETECTION_ARTIFACT_DIRS["root"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ANOMALY_DETECTION_EXPORT_DIR = ANOMALY_DETECTION_ARTIFACT_DIRS["exports"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ANOMALY_DETECTION_PLOT_DIR = ANOMALY_DETECTION_ARTIFACT_DIRS["plots"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ANOMALY_DETECTION_SUMMARY_DIR = ANOMALY_DETECTION_ARTIFACT_DIRS["summaries"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ANOMALY_DETECTION_PACKET_DIR = ANOMALY_DETECTION_ARTIFACT_DIRS["packets"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ANOMALY_DETECTION_METADATA_DIR = ANOMALY_DETECTION_ARTIFACT_DIRS["metadata"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ANOMALY_DETECTION_CONFIG_DIR = ANOMALY_DETECTION_ARTIFACT_DIRS["config"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ANOMALY_DETECTION_LINEAGE_DIR = ANOMALY_DETECTION_ARTIFACT_DIRS["lineage"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_SNAPSHOT_PATH = ( ANOMALY_DETECTION_CONFIG_DIR / f"{DATASET_NAME}__gold_anomaly_detection__resolved_config.yaml"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if CONFIG["execution"].get("save_config_snapshot", True): export_config_snapshot(CONFIG, CONFIG_SNAPSHOT_PATH)` | Controls validation, iteration, file handling, or error handling for this step. |
| `RUN_RESULT_PATH_MAP = { "baseline": ( GOLD_BASELINE_ARTIFACT_DIRS["scores"] / FILENAMES["baseline_results_file_name_pickle"] ), "cascade_defaults": ( GOLD_CASCADE_DEFAULTS_ARTIFACT` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `GOLD05_LEDGER_PATH = ( ANOMALY_DETECTION_LINEAGE_DIR / FILENAMES.get( "gold_anomaly_detection_ledger_file_name", f"ledger__{DATASET_NAME}__gold_anomaly_detection.json", )` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("Selected run artifact path: %s", RUN_RESULT_PATH_MAP[SELECTED_RUN_KEY])` | Writes a logger message for traceability during notebook execution. |
| `ledger.add( kind="decision", step="resolved_run_paths", message="Resolved selected run artifact path from pipeline config.", why="Gold 05 should load scored outputs from config-der` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 15 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `about`
- `allowed`
- `artifact`
- `baseline`
- `baseline_results`
- `bool`
- `cascade`
- `cascade_defaults`
- `cascade_results`
- `cascade_tuned`
- `cascade_tuned_results`
- `complain`
- `config`
- `copy`
- `def`
- `does`
- `else`
- `exist`
- `exists`
- `explicitly`

### Outputs

- `cascade_results_object`
- `dataframe`
- `load_selected_results_from_utils`
- `primary_global_name`
- `primary_global_name_map`
- `primary_global_object`
- `result_path`

### Key Operations

- `def load_selected_results_from_utils( *, selected_run_key: str, use_globals_if_available: bool = True,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """ Load the selected scored dataframe using: 1. globals fallback when explicitly allowed 2. artifact path from config via utils.file_io / pickle This version us`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `copy`
- `exists`
- `FileNotFoundError`
- `get`
- `globals`
- `info`
- `isinstance`
- `load_data`
- `load_selected_results_from_utils`
- `lower`
- `Path`
- `read_pickle`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def load_selected_results_from_utils( *, selected_run_key: str, use_globals_if_available: bool = True,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Load the selected scored dataframe using: 1. globals fallback when explicitly allowed 2. artifact path from config via utils.file_io / pickle This version us` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 16 — Load selected model results for Gold 05

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `add`
- `All`
- `anomaly`
- `at`
- `column_count`
- `dataframe`
- `downstream`
- `Gold`
- `head`
- `ledger`
- `load_selected_results`
- `load_selected_results_from_utils`
- `Loaded`
- `on`
- `operates`
- `output`
- `plots`
- `row_count`
- `run`

### Outputs

- `consequence`
- `data`
- `kind`
- `logger`
- `message`
- `selected_results`
- `selected_run_key`
- `step`
- `use_globals_if_available`
- `why`

### Key Operations

- `selected_results = load_selected_results_from_utils( selected_run_key=SELECTED_RUN_KEY, use_globals_if_available=USE_GLOBAL_RESULTS_IF_AVAILABLE,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="load_selected_results", message="Loaded selected scored dataframe for Gold 05 anomaly validation.", why="Timeline validation operates on a single sco`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(selected_results.head(4))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `display`
- `head`
- `load_selected_results_from_utils`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `selected_results = load_selected_results_from_utils( selected_run_key=SELECTED_RUN_KEY, use_globals_if_available=USE_GLOBAL_RESULTS_IF_AVAILABLE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="load_selected_results", message="Loaded selected scored dataframe for Gold 05 anomaly validation.", why="Timeline validation operates on a single sco` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(selected_results.head(4))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 17 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `bool`
- `candidates`
- `Candidates`
- `checked`
- `column`
- `columns`
- `Could`
- `DataFrame`
- `dataframe`
- `def`
- `f`
- `label`
- `Optional`
- `raise`
- `required`
- `resolve`
- `Sequence`
- `ValueError`

### Outputs

- `resolve_first_present_column`

### Key Operations

- `def resolve_first_present_column( dataframe: pd.DataFrame, candidates: Sequence[str], required: bool = True, label: str = "column",`: Defines notebook-local logic used later in the notebook.
- `) -> Optional[str]: for column in candidates: if column in dataframe.columns: return column if required: raise ValueError( f"Could not resolve required {label}. " f"Candidates chec`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `resolve_first_present_column`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def resolve_first_present_column( dataframe: pd.DataFrame, candidates: Sequence[str], required: bool = True, label: str = "column",` | Defines notebook-local logic used later in the notebook. |
| `) -> Optional[str]: for column in candidates: if column in dataframe.columns: return column if required: raise ValueError( f"Could not resolve required {label}. " f"Candidates chec` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 18 — Define row order alignment helper

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `any`
- `arange`
- `be`
- `by`
- `columns`
- `contains`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `drop`
- `dtype`
- `else`
- `f`
- `int64`
- `is_unique`
- `isna`
- `must`
- `null`
- `Optional`

### Outputs

- `ensure_row_id_and_plot_order`
- `out`

### Key Operations

- `def ensure_row_id_and_plot_order( dataframe: pd.DataFrame, row_id_column: str = ROW_ID_COLUMN, time_axis_column: Optional[str] = None,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: out = dataframe.copy() if row_id_column not in out.columns: out[row_id_column] = np.arange(len(out), dtype=np.int64) if out[row_id_column].isna().any(): raise Va`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `any`
- `arange`
- `copy`
- `ensure_row_id_and_plot_order`
- `isna`
- `reset_index`
- `sort_values`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def ensure_row_id_and_plot_order( dataframe: pd.DataFrame, row_id_column: str = ROW_ID_COLUMN, time_axis_column: Optional[str] = None,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: out = dataframe.copy() if row_id_column not in out.columns: out[row_id_column] = np.arange(len(out), dtype=np.int64) if out[row_id_column].isna().any(): raise Va` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 19 — Define broken-state anchor columns

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `astype`
- `BROKEN`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `eq`
- `found`
- `index`
- `is_broken_row`
- `loc`
- `No`
- `plot_order_index`
- `raise`
- `row`
- `rows_to_first_broken`
- `scored`
- `status_column`
- `tolist`
- `upper`

### Outputs

- `add_broken_anchor_columns`
- `broken_indices`
- `first_broken_dataframe_index`
- `first_broken_plot_order_index`
- `out`

### Key Operations

- `def add_broken_anchor_columns( dataframe: pd.DataFrame, status_column: str,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: out = dataframe.copy() out["is_broken_row"] = ( out[status_column].astype(str).str.upper().eq("BROKEN").astype(int) ) broken_indices = out.index[out["is_broken_r`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add_broken_anchor_columns`
- `astype`
- `copy`
- `eq`
- `tolist`
- `upper`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def add_broken_anchor_columns( dataframe: pd.DataFrame, status_column: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: out = dataframe.copy() out["is_broken_row"] = ( out[status_column].astype(str).str.upper().eq("BROKEN").astype(int) ) broken_indices = out.index[out["is_broken_r` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 20 — Define alert and normal-like row flags

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `astype`
- `column`
- `columns`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `f`
- `fillna`
- `flag`
- `is_normal_like_row`
- `Missing`
- `raise`
- `required`
- `selected_final_alert_flag`
- `target`
- `target_flag_column`
- `ValueError`

### Outputs

- `add_alert_and_normal_like_columns`
- `out`

### Key Operations

- `def add_alert_and_normal_like_columns( dataframe: pd.DataFrame, target_flag_column: str,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: out = dataframe.copy() if target_flag_column not in out.columns: raise ValueError(f"Missing required target flag column: {target_flag_column}") out["selected_fin`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add_alert_and_normal_like_columns`
- `astype`
- `copy`
- `fillna`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def add_alert_and_normal_like_columns( dataframe: pd.DataFrame, target_flag_column: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: out = dataframe.copy() if target_flag_column not in out.columns: raise ValueError(f"Missing required target flag column: {target_flag_column}") out["selected_fin` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 21 — Define forward stable-normal run lengths

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `astype`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `dtype`
- `else`
- `fillna`
- `forward_normal_like_run_length`
- `idx`
- `int64`
- `is_normal_like_row`
- `normal_like_column`
- `range`
- `to_numpy`
- `zeros`

### Outputs

- `compute_forward_stable_normal_run`
- `current_run`
- `forward_run`
- `out`
- `values`

### Key Operations

- `def compute_forward_stable_normal_run( dataframe: pd.DataFrame, normal_like_column: str = "is_normal_like_row",`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: out = dataframe.copy() values = out[normal_like_column].fillna(0).astype(int).to_numpy() forward_run = np.zeros(len(values), dtype=np.int64) current_run = 0 for `: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `compute_forward_stable_normal_run`
- `copy`
- `fillna`
- `range`
- `to_numpy`
- `zeros`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def compute_forward_stable_normal_run( dataframe: pd.DataFrame, normal_like_column: str = "is_normal_like_row",` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: out = dataframe.copy() values = out[normal_like_column].fillna(0).astype(int).to_numpy() forward_run = np.zeros(len(values), dtype=np.int64) current_run = 0 for ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 22 — Define recovery boundary detection

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `bool`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `else`
- `empty`
- `forward_normal_like_run_length`
- `iloc`
- `loc`
- `plot_order_index`
- `RECOVERY_STABILITY_ROWS`
- `recovery_stability_rows`
- `RECOVERY_STARTS_AFTER_BROKEN`
- `recovery_starts_after_broken`
- `stability_rows`

### Outputs

- `candidate_mask`
- `candidate_rows`
- `first_broken_plot_order_index`
- `out`
- `recovery_end_plot_order_index`
- `recovery_start_plot_order_index`
- `resolve_recovery_boundaries`

### Key Operations

- `def resolve_recovery_boundaries( dataframe: pd.DataFrame, stability_rows: int = RECOVERY_STABILITY_ROWS, recovery_starts_after_broken: bool = RECOVERY_STARTS_AFTER_BROKEN,`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, Any]: out = dataframe.copy() first_broken_plot_order_index = int(out["first_broken_plot_order_index"].iloc[0]) recovery_start_plot_order_index = ( first_broken_plot_`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `copy`
- `resolve_recovery_boundaries`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def resolve_recovery_boundaries( dataframe: pd.DataFrame, stability_rows: int = RECOVERY_STABILITY_ROWS, recovery_starts_after_broken: bool = RECOVERY_STARTS_AFTER_BROKEN,` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, Any]: out = dataframe.copy() first_broken_plot_order_index = int(out["first_broken_plot_order_index"].iloc[0]) recovery_start_plot_order_index = ( first_broken_plot_` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 23 — Define episode-phase labeling

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `append`
- `astype`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `elif`
- `else`
- `episode_phase`
- `failure`
- `first_broken_plot_order_index`
- `is_recovery_row`
- `is_stable_normal_row`
- `phase_values`
- `plot_idx`
- `plot_order_index`
- `pre_failure`
- `recovery`
- `recovery_boundary_payload`

### Outputs

- `add_episode_phase_columns`
- `first_broken`
- `out`
- `recovery_end`
- `recovery_start`

### Key Operations

- `def add_episode_phase_columns( dataframe: pd.DataFrame, recovery_boundary_payload: dict[str, Any],`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: out = dataframe.copy() first_broken = recovery_boundary_payload["first_broken_plot_order_index"] recovery_start = recovery_boundary_payload["recovery_start_plot_`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add_episode_phase_columns`
- `append`
- `astype`
- `copy`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def add_episode_phase_columns( dataframe: pd.DataFrame, recovery_boundary_payload: dict[str, Any],` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: out = dataframe.copy() first_broken = recovery_boundary_payload["first_broken_plot_order_index"] recovery_start = recovery_boundary_payload["recovery_start_plot_` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 24 — Define detection class labels

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `append`
- `continue`
- `copy`
- `dataframe`
- `DataFrame`
- `def`
- `detection_class`
- `detection_classes`
- `early_warning`
- `elif`
- `else`
- `episode_phase`
- `failure`
- `failure_hit`
- `false_positive`
- `get`
- `iterrows`
- `no_alert`
- `pre_failure`

### Outputs

- `classify_detection_rows`
- `is_alert`
- `out`
- `phase`

### Key Operations

- `def classify_detection_rows( dataframe: pd.DataFrame, target_flag_column: str,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: out = dataframe.copy() detection_classes: list[str] = [] for _, row in out.iterrows(): is_alert = int(row.get(target_flag_column, 0) or 0) if is_alert != 1: dete`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append`
- `classify_detection_rows`
- `copy`
- `get`
- `iterrows`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def classify_detection_rows( dataframe: pd.DataFrame, target_flag_column: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: out = dataframe.copy() detection_classes: list[str] = [] for _, row in out.iterrows(): is_alert = int(row.get(target_flag_column, 0) or 0) if is_alert != 1: dete` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 25 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add_alert_and_normal_like_columns`
- `add_broken_anchor_columns`
- `add_episode_phase_columns`
- `axis`
- `be`
- `Candidates`
- `checked`
- `classify_detection_rows`
- `column`
- `compute_forward_stable_normal_run`
- `could`
- `ensure_row_id_and_plot_order`
- `even`
- `f`
- `is_normal_like_row`
- `raise`
- `RECOVERY_STABILITY_ROWS`
- `resolve_first_present_column`
- `resolve_recovery_boundaries`
- `resolved`

### Outputs

- `label`
- `normal_like_column`
- `recovery_boundary_payload`
- `recovery_starts_after_broken`
- `required`
- `row_id_column`
- `stability_rows`
- `status_column`
- `target_flag_column`
- `time_axis_column`
- `timeline_source_df`

### Key Operations

- `time_axis_column = resolve_first_present_column( selected_results, TIME_AXIS_CANDIDATES, required=False, label="time axis column",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `status_column = resolve_first_present_column( selected_results, STATUS_COLUMN_CANDIDATES, required=True, label="status column",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if status_column is None: raise ValueError( "status_column could not be resolved, even though required=True was used. " f"Candidates checked: {STATUS_COLUMN_CANDIDATES}" )`: Controls validation, iteration, file handling, or error handling for this step.
- `status_column_resolved: str = str(status_column)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `timeline_source_df = ensure_row_id_and_plot_order( selected_results, row_id_column=ROW_ID_COLUMN, time_axis_column=time_axis_column,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `timeline_source_df = add_broken_anchor_columns( timeline_source_df, status_column=status_column_resolved,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `timeline_source_df = add_alert_and_normal_like_columns( timeline_source_df, target_flag_column=TARGET_FLAG_COLUMN,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add_alert_and_normal_like_columns`
- `add_broken_anchor_columns`
- `add_episode_phase_columns`
- `classify_detection_rows`
- `compute_forward_stable_normal_run`
- `ensure_row_id_and_plot_order`
- `resolve_first_present_column`
- `resolve_recovery_boundaries`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `time_axis_column = resolve_first_present_column( selected_results, TIME_AXIS_CANDIDATES, required=False, label="time axis column",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `status_column = resolve_first_present_column( selected_results, STATUS_COLUMN_CANDIDATES, required=True, label="status column",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if status_column is None: raise ValueError( "status_column could not be resolved, even though required=True was used. " f"Candidates checked: {STATUS_COLUMN_CANDIDATES}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `status_column_resolved: str = str(status_column)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `timeline_source_df = ensure_row_id_and_plot_order( selected_results, row_id_column=ROW_ID_COLUMN, time_axis_column=time_axis_column,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `timeline_source_df = add_broken_anchor_columns( timeline_source_df, status_column=status_column_resolved,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `timeline_source_df = add_alert_and_normal_like_columns( timeline_source_df, target_flag_column=TARGET_FLAG_COLUMN,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `timeline_source_df = compute_forward_stable_normal_run( timeline_source_df, normal_like_column="is_normal_like_row",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `recovery_boundary_payload = resolve_recovery_boundaries( timeline_source_df, stability_rows=RECOVERY_STABILITY_ROWS, recovery_starts_after_broken=RECOVERY_STARTS_AFTER_BROKEN,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `timeline_source_df = add_episode_phase_columns( timeline_source_df, recovery_boundary_payload=recovery_boundary_payload,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `timeline_source_df = classify_detection_rows( timeline_source_df, target_flag_column=TARGET_FLAG_COLUMN,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 26 — Define row order alignment helper

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add_alert_and_normal_like_columns`
- `add_broken_anchor_columns`
- `add_episode_phase_columns`
- `bool`
- `classify_detection_rows`
- `compute_forward_stable_normal_run`
- `DataFrame`
- `def`
- `ensure_row_id_and_plot_order`
- `is_normal_like_row`
- `Optional`
- `recovery_stability_rows`
- `resolve_recovery_boundaries`
- `selected_results`
- `status_column`

### Outputs

- `build_anomaly_timeline_dataframe`
- `df`
- `normal_like_column`
- `recovery_boundary_payload`
- `recovery_starts_after_broken`
- `row_id_column`
- `stability_rows`
- `target_flag_column`
- `time_axis_column`

### Key Operations

- `def build_anomaly_timeline_dataframe( selected_results: pd.DataFrame, target_flag_column: str, status_column: str, row_id_column: str, time_axis_column: Optional[str], recovery_sta`: Defines notebook-local logic used later in the notebook.
- `): df = ensure_row_id_and_plot_order( selected_results, row_id_column=row_id_column, time_axis_column=time_axis_column, ) df = add_broken_anchor_columns(df, status_column=status_co`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add_alert_and_normal_like_columns`
- `add_broken_anchor_columns`
- `add_episode_phase_columns`
- `build_anomaly_timeline_dataframe`
- `classify_detection_rows`
- `compute_forward_stable_normal_run`
- `ensure_row_id_and_plot_order`
- `resolve_recovery_boundaries`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_anomaly_timeline_dataframe( selected_results: pd.DataFrame, target_flag_column: str, status_column: str, row_id_column: str, time_axis_column: Optional[str], recovery_sta` | Defines notebook-local logic used later in the notebook. |
| `): df = ensure_row_id_and_plot_order( selected_results, row_id_column=row_id_column, time_axis_column=time_axis_column, ) df = add_broken_anchor_columns(df, status_column=status_co` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 27 — Build the final anomaly timeline

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `All`
- `annotations`
- `anomaly`
- `anomaly_timeline_dataframe`
- `build_anomaly_timeline_dataframe`
- `build_timeline_dataframe`
- `Built`
- `c`
- `canonical`
- `column_count`
- `columns`
- `dataframe`
- `derive`
- `detection_class`
- `episode_phase`
- `exports`
- `failure`
- `forward_normal_like_run_length`
- `frame`

### Outputs

- `consequence`
- `data`
- `kind`
- `logger`
- `message`
- `preview_columns`
- `recovery_stability_rows`
- `recovery_starts_after_broken`
- `row_id_column`
- `status_column`
- `step`
- `target_flag_column`
- `time_axis_column`
- `why`

### Key Operations

- `anomaly_timeline_dataframe, recovery_boundary_payload = build_anomaly_timeline_dataframe( selected_results, target_flag_column=TARGET_FLAG_COLUMN, status_column=status_column_resol`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="build_timeline_dataframe", message="Built anomaly timeline dataframe with failure/recovery phase annotations.", why="Gold 05 needs one canonical row-`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `preview_columns = [ "plot_order_index", status_column, "is_broken_row", TARGET_FLAG_COLUMN, "selected_final_alert_flag", "is_normal_like_row", "forward_normal_like_run_length", "ep`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `preview_columns = [c for c in preview_columns if c in anomaly_timeline_dataframe.columns]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `display(anomaly_timeline_dataframe[preview_columns].head(40))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `build_anomaly_timeline_dataframe`
- `display`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `anomaly_timeline_dataframe, recovery_boundary_payload = build_anomaly_timeline_dataframe( selected_results, target_flag_column=TARGET_FLAG_COLUMN, status_column=status_column_resol` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="build_timeline_dataframe", message="Built anomaly timeline dataframe with failure/recovery phase annotations.", why="Gold 05 needs one canonical row-` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `preview_columns = [ "plot_order_index", status_column, "is_broken_row", TARGET_FLAG_COLUMN, "selected_final_alert_flag", "is_normal_like_row", "forward_normal_like_run_length", "ep` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `preview_columns = [c for c in preview_columns if c in anomaly_timeline_dataframe.columns]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `display(anomaly_timeline_dataframe[preview_columns].head(40))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 28 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `append`
- `astype`
- `Build`
- `by`
- `c`
- `column`
- `columns`
- `copy`
- `dataframe`
- `DataFrame`
- `decision_column`
- `def`
- `detected`
- `detection_class`
- `drop`
- `episode_phase`
- `event_step`
- `event_time`
- `fillna`

### Outputs

- `build_detected_rows_review_dataframe`
- `candidate_columns`
- `detected_rows_df`
- `out`

### Key Operations

- `def build_detected_rows_review_dataframe( dataframe: pd.DataFrame, target_flag_column: str, score_column: Optional[str] = None, decision_column: Optional[str] = None, include_colum`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """ Build a filtered detected-row review dataframe for the selected run. """ out = dataframe.copy() candidate_columns = [ ROW_ID_COLUMN, "plot_order_index", "met`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append`
- `astype`
- `build_detected_rows_review_dataframe`
- `copy`
- `fillna`
- `reset_index`
- `sort_values`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_detected_rows_review_dataframe( dataframe: pd.DataFrame, target_flag_column: str, score_column: Optional[str] = None, decision_column: Optional[str] = None, include_colum` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Build a filtered detected-row review dataframe for the selected run. """ out = dataframe.copy() candidate_columns = [ ROW_ID_COLUMN, "plot_order_index", "met` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 29 — Create Gold 05 anomaly timeline output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `across`
- `add`
- `alert`
- `all`
- `anomaly_timeline_dataframe`
- `be`
- `build_detected_rows_review_dataframe`
- `Built`
- `can`
- `columns`
- `compared`
- `dataframe`
- `Detected`
- `detected`
- `detected_row_count`
- `detected_rows_review`
- `else`
- `exported`
- `forward_normal_like_run_length`
- `Gold`

### Outputs

- `consequence`
- `data`
- `decision_column`
- `detected_rows_review_df`
- `include_columns`
- `kind`
- `logger`
- `message`
- `score_column`
- `step`
- `target_flag_column`
- `why`

### Key Operations

- `detected_rows_review_df = build_detected_rows_review_dataframe( anomaly_timeline_dataframe, target_flag_column=TARGET_FLAG_COLUMN, score_column=PRIMARY_SCORE_COLUMN if PRIMARY_SCOR`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="detected_rows_review", message="Built detected-row review dataframe for selected run.", why="Gold 05 should provide row-level review of all alert row`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(detected_rows_review_df.head(30))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `build_detected_rows_review_dataframe`
- `display`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `detected_rows_review_df = build_detected_rows_review_dataframe( anomaly_timeline_dataframe, target_flag_column=TARGET_FLAG_COLUMN, score_column=PRIMARY_SCORE_COLUMN if PRIMARY_SCOR` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="detected_rows_review", message="Built detected-row review dataframe for selected run.", why="Gold 05 should provide row-level review of all alert row` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(detected_rows_review_df.head(30))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 30 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `astype`
- `col`
- `columns`
- `def`
- `df`
- `else`
- `empty`
- `fillna`
- `iloc`
- `loc`
- `plot_order_index`

### Outputs

- `get_first_alert_index`
- `rows`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `fillna`
- `get_first_alert_index`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 31 — Define first-alert lookup helper

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_timeline_dataframe`
- `baseline_flag`
- `cascade_final_flag`
- `first_broken`
- `first_broken_plot_order_index`
- `get_first_alert_index`
- `iloc`
- `stage1_flag`
- `stage2_flag`
- `stage2_raw_flag`

### Outputs

- `stage_alerts`

### Key Operations

- `stage_alerts = { "stage1_flag": get_first_alert_index(anomaly_timeline_dataframe, "stage1_flag"), "stage2_raw_flag": get_first_alert_index(anomaly_timeline_dataframe, "stage2_raw_f`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage_alerts)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `get_first_alert_index`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_alerts = { "stage1_flag": get_first_alert_index(anomaly_timeline_dataframe, "stage1_flag"), "stage2_raw_flag": get_first_alert_index(anomaly_timeline_dataframe, "stage2_raw_f` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage_alerts)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 32 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_timeline_dataframe`
- `build_detected_rows_review_dataframe`
- `cascade_final_flag`
- `columns`
- `detection_class`
- `else`
- `episode_phase`
- `head`
- `machine_status`
- `stage1_decision`
- `stage1_flag`
- `stage1_score`
- `stage2_flag`
- `stage2_raw_flag`

### Outputs

- `decision_column`
- `include_columns`
- `score_column`
- `stage1_detected_rows_df`
- `target_flag_column`

### Key Operations

- `stage1_detected_rows_df = build_detected_rows_review_dataframe( anomaly_timeline_dataframe, target_flag_column="stage1_flag", score_column="stage1_score" if "stage1_score" in anoma`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage1_detected_rows_df.head(150))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `build_detected_rows_review_dataframe`
- `display`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage1_detected_rows_df = build_detected_rows_review_dataframe( anomaly_timeline_dataframe, target_flag_column="stage1_flag", score_column="stage1_score" if "stage1_score" in anoma` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage1_detected_rows_df.head(150))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 33 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `above`
- `already`
- `anomaly_timeline_dataframe`
- `astype`
- `build`
- `cascade_results`
- `cells`
- `columns`
- `created`
- `dataframe`
- `ensure_row_id_and_plot_order`
- `Gold`
- `max`
- `min`
- `missing`
- `on`
- `plot_order_index`
- `raise`
- `ready`
- `receives`

### Outputs

- `errors`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Validate plot_order_index on the selected timeline dataframe`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Gold 05 should use anomaly_timeline_dataframe, not cascade_results.`: Documents the purpose or boundary of the surrounding notebook step.
- `# anomaly_timeline_dataframe is already created from selected_results and`: Documents the purpose or boundary of the surrounding notebook step.
- `# already receives plot_order_index from ensure_row_id_and_plot_order().`: Documents the purpose or boundary of the surrounding notebook step.
- `if "plot_order_index" not in anomaly_timeline_dataframe.columns: raise ValueError( "anomaly_timeline_dataframe is missing plot_order_index. " "Rerun the timeline build cells above.`: Controls validation, iteration, file handling, or error handling for this step.
- `anomaly_timeline_dataframe["plot_order_index"] = ( pd.to_numeric( anomaly_timeline_dataframe["plot_order_index"], errors="raise", ) .astype(int)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("plot_order_index ready on anomaly_timeline_dataframe.")`: Displays a notebook-facing result for inspection.
- `print( { "selected_run_key": SELECTED_RUN_KEY, "min": int(anomaly_timeline_dataframe["plot_order_index"].min()), "max": int(anomaly_timeline_dataframe["plot_order_index"].max()), "`: Displays a notebook-facing result for inspection.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `ensure_row_id_and_plot_order`
- `max`
- `min`
- `to_numeric`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Validate plot_order_index on the selected timeline dataframe` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Gold 05 should use anomaly_timeline_dataframe, not cascade_results.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# anomaly_timeline_dataframe is already created from selected_results and` | Documents the purpose or boundary of the surrounding notebook step. |
| `# already receives plot_order_index from ensure_row_id_and_plot_order().` | Documents the purpose or boundary of the surrounding notebook step. |
| `if "plot_order_index" not in anomaly_timeline_dataframe.columns: raise ValueError( "anomaly_timeline_dataframe is missing plot_order_index. " "Rerun the timeline build cells above.` | Controls validation, iteration, file handling, or error handling for this step. |
| `anomaly_timeline_dataframe["plot_order_index"] = ( pd.to_numeric( anomaly_timeline_dataframe["plot_order_index"], errors="raise", ) .astype(int)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("plot_order_index ready on anomaly_timeline_dataframe.")` | Displays a notebook-facing result for inspection. |
| `print( { "selected_run_key": SELECTED_RUN_KEY, "min": int(anomaly_timeline_dataframe["plot_order_index"].min()), "max": int(anomaly_timeline_dataframe["plot_order_index"].max()), "` | Displays a notebook-facing result for inspection. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 34 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_timeline_dataframe`
- `baseline`
- `baseline_flag`
- `baseline_score`
- `between`
- `cascade`
- `cascade_final_flag`
- `cascade_results`
- `column_name`
- `columns`
- `debug`
- `defaults`
- `detection_class`
- `episode_phase`
- `improved`
- `instead`
- `loc`
- `machine_status`
- `meta__row_id`
- `of`

### Outputs

- `available_debug_columns`
- `debug_columns`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Selected-run debug window`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Use anomaly_timeline_dataframe instead of cascade_results so this works`: Documents the purpose or boundary of the surrounding notebook step.
- `# for baseline, cascade defaults, cascade tuned, and stage3 improved runs.`: Documents the purpose or boundary of the surrounding notebook step.
- `debug_columns = [ "plot_order_index", "time_index", "meta__row_id", "machine_status", "episode_phase", "detection_class", "baseline_flag", "baseline_score", "stage1_flag", "stage1_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `available_debug_columns = [ column_name for column_name in debug_columns if column_name in anomaly_timeline_dataframe.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display( anomaly_timeline_dataframe.loc[ anomaly_timeline_dataframe["plot_order_index"].between(10380, 10460), available_debug_columns, ]`: Displays a notebook-facing result for inspection.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `between`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Selected-run debug window` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Use anomaly_timeline_dataframe instead of cascade_results so this works` | Documents the purpose or boundary of the surrounding notebook step. |
| `# for baseline, cascade defaults, cascade tuned, and stage3 improved runs.` | Documents the purpose or boundary of the surrounding notebook step. |
| `debug_columns = [ "plot_order_index", "time_index", "meta__row_id", "machine_status", "episode_phase", "detection_class", "baseline_flag", "baseline_score", "stage1_flag", "stage1_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `available_debug_columns = [ column_name for column_name in debug_columns if column_name in anomaly_timeline_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display( anomaly_timeline_dataframe.loc[ anomaly_timeline_dataframe["plot_order_index"].between(10380, 10460), available_debug_columns, ]` | Displays a notebook-facing result for inspection. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 35 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_timeline_dataframe`
- `baseline_flag`
- `baseline_score`
- `between`
- `Build`
- `cascade_final_flag`
- `column`
- `column_name`
- `columns`
- `copy`
- `detection_class`
- `Do`
- `early`
- `episode_phase`
- `first`
- `inside`
- `loc`
- `machine_status`
- `plot_order_index`
- `put`

### Outputs

- `stage1_early_window_columns`
- `stage1_early_window_df`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Stage 1 / early-window review`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build the column list first, then select. Do not put None inside .loc[].`: Documents the purpose or boundary of the surrounding notebook step.
- `stage1_early_window_columns = [ "plot_order_index", "machine_status", "episode_phase", "detection_class", "stage1_flag", "stage1_score", "stage2_score", "stage2_raw_flag", "stage2_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage1_early_window_columns = [ column_name for column_name in stage1_early_window_columns if column_name in anomaly_timeline_dataframe.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage1_early_window_df = anomaly_timeline_dataframe.loc[ anomaly_timeline_dataframe["plot_order_index"].between(10380, 10460), stage1_early_window_columns,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `].copy()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage1_early_window_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `between`
- `copy`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Stage 1 / early-window review` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build the column list first, then select. Do not put None inside .loc[].` | Documents the purpose or boundary of the surrounding notebook step. |
| `stage1_early_window_columns = [ "plot_order_index", "machine_status", "episode_phase", "detection_class", "stage1_flag", "stage1_score", "stage2_score", "stage2_raw_flag", "stage2_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage1_early_window_columns = [ column_name for column_name in stage1_early_window_columns if column_name in anomaly_timeline_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage1_early_window_df = anomaly_timeline_dataframe.loc[ anomaly_timeline_dataframe["plot_order_index"].between(10380, 10460), stage1_early_window_columns,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage1_early_window_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 36 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_timeline_dataframe`
- `baseline_flag`
- `baseline_score`
- `between`
- `cascade_final_flag`
- `column_name`
- `columns`
- `copy`
- `detection_class`
- `episode_phase`
- `loc`
- `machine_status`
- `plot_order_index`
- `stage1_flag`
- `stage1_score`
- `stage2_flag`
- `stage2_raw_flag`
- `stage2_score`
- `tail`
- `TARGET_FLAG_COLUMN`

### Outputs

- `stage2_pre_failure_columns`
- `stage2_pre_failure_df`

### Key Operations

- `stage2_pre_failure_columns = [ "plot_order_index", "machine_status", "episode_phase", "detection_class", "baseline_flag", "baseline_score", "stage1_flag", "stage1_score", "stage2_s`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage2_pre_failure_columns = [ column_name for column_name in stage2_pre_failure_columns if column_name in anomaly_timeline_dataframe.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage2_pre_failure_df = anomaly_timeline_dataframe.loc[ anomaly_timeline_dataframe["plot_order_index"].between(17000, 17155), stage2_pre_failure_columns,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `].copy()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage2_pre_failure_df.tail(100))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `between`
- `copy`
- `display`
- `tail`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage2_pre_failure_columns = [ "plot_order_index", "machine_status", "episode_phase", "detection_class", "baseline_flag", "baseline_score", "stage1_flag", "stage1_score", "stage2_s` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage2_pre_failure_columns = [ column_name for column_name in stage2_pre_failure_columns if column_name in anomaly_timeline_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage2_pre_failure_df = anomaly_timeline_dataframe.loc[ anomaly_timeline_dataframe["plot_order_index"].between(17000, 17155), stage2_pre_failure_columns,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage2_pre_failure_df.tail(100))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 37 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_timeline_dataframe`
- `baseline_flag`
- `baseline_score`
- `between`
- `cascade_final_flag`
- `column_name`
- `columns`
- `copy`
- `detection_class`
- `episode_phase`
- `loc`
- `machine_status`
- `plot_order_index`
- `stage1_flag`
- `stage1_score`
- `stage2_flag`
- `stage2_raw_flag`
- `stage2_score`
- `TARGET_FLAG_COLUMN`

### Outputs

- `stage_window_columns`
- `stage2_window_early_df`
- `stage2_window_failure_df`

### Key Operations

- `stage_window_columns = [ "plot_order_index", "machine_status", "episode_phase", "detection_class", "baseline_flag", "baseline_score", "stage1_flag", "stage1_score", "stage2_score",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage_window_columns = [ column_name for column_name in stage_window_columns if column_name in anomaly_timeline_dataframe.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage2_window_early_df = anomaly_timeline_dataframe.loc[ anomaly_timeline_dataframe["plot_order_index"].between(10380, 10460), stage_window_columns,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `].copy()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage2_window_failure_df = anomaly_timeline_dataframe.loc[ anomaly_timeline_dataframe["plot_order_index"].between(17130, 17210), stage_window_columns,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `].copy()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(stage2_window_early_df)`: Displays a notebook-facing result for inspection.
- `display(stage2_window_failure_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `between`
- `copy`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stage_window_columns = [ "plot_order_index", "machine_status", "episode_phase", "detection_class", "baseline_flag", "baseline_score", "stage1_flag", "stage1_score", "stage2_score",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage_window_columns = [ column_name for column_name in stage_window_columns if column_name in anomaly_timeline_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage2_window_early_df = anomaly_timeline_dataframe.loc[ anomaly_timeline_dataframe["plot_order_index"].between(10380, 10460), stage_window_columns,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage2_window_failure_df = anomaly_timeline_dataframe.loc[ anomaly_timeline_dataframe["plot_order_index"].between(17130, 17210), stage_window_columns,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(stage2_window_early_df)` | Displays a notebook-facing result for inspection. |
| `display(stage2_window_failure_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 38 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_plot_order_index`
- `Any`
- `astype`
- `cascade_final_flag`
- `column`
- `columns`
- `DataFrame`
- `dataframe`
- `def`
- `else`
- `empty`
- `f`
- `fillna`
- `first_`
- `iloc`
- `loc`
- `payload`
- `plot_order_index`
- `stage1_flag`
- `stage2_flag`

### Outputs

- `build_optional_cascade_stage_summary`
- `flagged`

### Key Operations

- `def build_optional_cascade_stage_summary( dataframe: pd.DataFrame,`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, Any]: payload: dict[str, Any] = {} for column in ["stage1_flag", "stage2_flag", "stage2_raw_flag", "cascade_final_flag"]: if column in dataframe.columns: flagged = d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `build_optional_cascade_stage_summary`
- `fillna`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_optional_cascade_stage_summary( dataframe: pd.DataFrame,` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, Any]: payload: dict[str, Any] = {} for column in ["stage1_flag", "stage2_flag", "stage2_raw_flag", "cascade_final_flag"]: if column in dataframe.columns: flagged = d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 39 — Define early-warning summary payload

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `astype`
- `build_optional_cascade_stage_summary`
- `cascade`
- `columns`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `detection_class`
- `dropna`
- `else`
- `empty`
- `episode_phase`
- `eq`
- `fillna`
- `iloc`
- `lead_rows_to_failure`
- `lead_time_minutes_to_failure`
- `loc`

### Outputs

- `alert_rows`
- `build_detection_summary_payload`
- `detection_class_counts`
- `first_alert_plot_order_index`
- `first_broken_plot_order_index`
- `lead_rows`
- `out`
- `payload`
- `recovery_end_candidates`
- `recovery_end_plot_order_index`

### Key Operations

- `def build_detection_summary_payload( dataframe: pd.DataFrame,`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, Any]: out = dataframe.copy() first_broken_plot_order_index = int(out["first_broken_plot_order_index"].iloc[0]) alert_rows = out.loc[out[TARGET_FLAG_COLUMN].fillna(0)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `build_detection_summary_payload`
- `build_optional_cascade_stage_summary`
- `copy`
- `eq`
- `fillna`
- `sum`
- `to_dict`
- `update`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_detection_summary_payload( dataframe: pd.DataFrame,` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, Any]: out = dataframe.copy() first_broken_plot_order_index = int(out["first_broken_plot_order_index"].iloc[0]) alert_rows = out.loc[out[TARGET_FLAG_COLUMN].fillna(0)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 40 — Define early-warning summary payload

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_detection_summary_payload`
- `DataFrame`
- `dataframe`
- `def`

### Outputs

- `build_failure_lead_time_dataframe`

### Key Operations

- `def build_failure_lead_time_dataframe( dataframe: pd.DataFrame,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: return pd.DataFrame([build_detection_summary_payload(dataframe)])`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `build_detection_summary_payload`
- `build_failure_lead_time_dataframe`
- `DataFrame`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_failure_lead_time_dataframe( dataframe: pd.DataFrame,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: return pd.DataFrame([build_detection_summary_payload(dataframe)])` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 41 — Define alert packet summary table

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `alert_packet_id`
- `ALERT_PACKET_MAX_GAP_ROWS`
- `Any`
- `any`
- `append`
- `astype`
- `contains_failure_hit`
- `contains_pre_failure_alert`
- `contains_recovery_alert`
- `contains_stable_normal_alert`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `diff`
- `drop`
- `dropna`
- `elif`
- `empty`

### Outputs

- `alert_rows`
- `build_alert_packet_summary`
- `current_packet_id`
- `first_broken_plot_order_index`
- `first_packet_plot_order_index`
- `last_packet_plot_order_index`
- `out`

### Key Operations

- `def build_alert_packet_summary( dataframe: pd.DataFrame, max_gap_rows: int = ALERT_PACKET_MAX_GAP_ROWS,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: out = dataframe.copy() alert_rows = out.loc[out[TARGET_FLAG_COLUMN].fillna(0).astype(int) == 1].copy() if alert_rows.empty: return pd.DataFrame() alert_rows = al`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `any`
- `append`
- `astype`
- `build_alert_packet_summary`
- `copy`
- `DataFrame`
- `diff`
- `fillna`
- `groupby`
- `iterrows`
- `max`
- `min`
- `reset_index`
- `sort_values`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_alert_packet_summary( dataframe: pd.DataFrame, max_gap_rows: int = ALERT_PACKET_MAX_GAP_ROWS,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: out = dataframe.copy() alert_rows = out.loc[out[TARGET_FLAG_COLUMN].fillna(0).astype(int) == 1].copy() if alert_rows.empty: return pd.DataFrame() alert_rows = al` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 42 — Create Gold 05 anomaly timeline output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `alert`
- `alert_packet_count`
- `anomaly_timeline_dataframe`
- `artifacts`
- `behavior`
- `build_alert_packet_summary`
- `build_detection_summary_payload`
- `build_failure_lead_time_dataframe`
- `build_summaries`
- `Built`
- `detection`
- `dumps`
- `early`
- `else`
- `empty`
- `found`
- `get`
- `Gold`
- `has`

### Outputs

- `alert_packet_summary_df`
- `consequence`
- `data`
- `detection_summary_payload`
- `failure_lead_time_df`
- `kind`
- `logger`
- `message`
- `step`
- `why`

### Key Operations

- `failure_lead_time_df = build_failure_lead_time_dataframe(anomaly_timeline_dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `alert_packet_summary_df = build_alert_packet_summary(anomaly_timeline_dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `detection_summary_payload = build_detection_summary_payload(anomaly_timeline_dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ledger.add( kind="step", step="build_summaries", message="Built lead-time, alert-packet, and detection summary outputs.", why="Gold 05 must quantify early warning timing and alert `: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print(json.dumps(detection_summary_payload, indent=2))`: Displays a notebook-facing result for inspection.
- `display(failure_lead_time_df)`: Displays a notebook-facing result for inspection.
- `if not alert_packet_summary_df.empty: display(alert_packet_summary_df.head(20))`: Displays a notebook-facing result for inspection.
- `else: print("No alert packets were found.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `build_alert_packet_summary`
- `build_detection_summary_payload`
- `build_failure_lead_time_dataframe`
- `display`
- `dumps`
- `get`
- `head`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `failure_lead_time_df = build_failure_lead_time_dataframe(anomaly_timeline_dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `alert_packet_summary_df = build_alert_packet_summary(anomaly_timeline_dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `detection_summary_payload = build_detection_summary_payload(anomaly_timeline_dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger.add( kind="step", step="build_summaries", message="Built lead-time, alert-packet, and detection summary outputs.", why="Gold 05 must quantify early warning timing and alert ` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(json.dumps(detection_summary_payload, indent=2))` | Displays a notebook-facing result for inspection. |
| `display(failure_lead_time_df)` | Displays a notebook-facing result for inspection. |
| `if not alert_packet_summary_df.empty: display(alert_packet_summary_df.head(20))` | Displays a notebook-facing result for inspection. |
| `else: print("No alert packets were found.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 43 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `def`
- `f`
- `keys`
- `raise`
- `RUN_CONFIG_MAP`
- `selected_run_key`
- `Unsupported`
- `Valid`
- `ValueError`

### Outputs

- `resolve_run_config`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `resolve_run_config`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 44 — Define row order alignment helper

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `add_alert_and_normal_like_columns`
- `add_broken_anchor_columns`
- `add_episode_phase_columns`
- `aligned`
- `Any`
- `arbitrary`
- `around`
- `axis`
- `be`
- `Build`
- `Candidates`
- `checked`
- `classify_detection_rows`
- `column`
- `compute_forward_stable_normal_run`
- `copy`
- `could`
- `dataframe`
- `DataFrame`

### Outputs

- `build_run_timeline_dataframe`
- `label`
- `load_results_for_run`
- `plot_run_label`
- `recovery_boundary_payload`
- `recovery_starts_after_broken`
- `required`
- `row_id_column`
- `run_config`
- `run_df`
- `run_recovery_boundary_payload`
- `run_results`
- `run_status_column`
- `run_time_axis_column`
- `selected_run_key`
- `stability_rows`
- `target_flag_column`
- `time_axis_column`
- `use_globals_if_available`

### Key Operations

- `def load_results_for_run(selected_run_key: str) -> pd.DataFrame: """ Wrapper around the utils-aligned loader for arbitrary run keys. """ return load_selected_results_from_utils( se`: Defines notebook-local logic used later in the notebook.
- `def build_run_timeline_dataframe( selected_run_key: str,`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, Any]: """ Build a fully prepared timeline dataframe for one run. """ run_config = resolve_run_config(selected_run_key) plot_run_label = PLOT_RUN_LABEL_MAP.get(select`: Loads input data, configuration, or artifacts required by the current stage.

Important functions or methods detected:
- `add_alert_and_normal_like_columns`
- `add_broken_anchor_columns`
- `add_episode_phase_columns`
- `build_run_timeline_dataframe`
- `classify_detection_rows`
- `compute_forward_stable_normal_run`
- `copy`
- `ensure_row_id_and_plot_order`
- `get`
- `load_results_for_run`
- `load_selected_results_from_utils`
- `resolve_first_present_column`
- `resolve_recovery_boundaries`
- `resolve_run_config`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def load_results_for_run(selected_run_key: str) -> pd.DataFrame: """ Wrapper around the utils-aligned loader for arbitrary run keys. """ return load_selected_results_from_utils( se` | Defines notebook-local logic used later in the notebook. |
| `def build_run_timeline_dataframe( selected_run_key: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, Any]: """ Build a fully prepared timeline dataframe for one run. """ run_config = resolve_run_config(selected_run_key) plot_run_label = PLOT_RUN_LABEL_MAP.get(select` | Loads input data, configuration, or artifacts required by the current stage. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 45 — Define per-run detection summary payload

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `an`
- `Any`
- `arbitrary`
- `astype`
- `Build`
- `build_optional_cascade_stage_summary`
- `cascade`
- `columns`
- `copy`
- `def`
- `detection_class`
- `detection_class_counts`
- `dropna`
- `else`
- `empty`
- `episode_phase`
- `eq`
- `fillna`
- `iloc`
- `lead_rows_to_failure`

### Outputs

- `alert_rows`
- `build_run_detection_summary_payload`
- `df`
- `first_alert_plot_order_index`
- `first_broken_plot_order_index`
- `lead_rows`
- `payload`
- `recovery_end_candidates`
- `recovery_end_plot_order_index`
- `run_family`
- `target_flag_column`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `build_optional_cascade_stage_summary`
- `build_run_detection_summary_payload`
- `copy`
- `eq`
- `fillna`
- `sum`
- `to_dict`
- `update`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 46 — Define per-run detection summary payload

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `build_run_detection_summary_payload`
- `DataFrame`
- `def`
- `payload`
- `run_payloads`
- `Sequence`

### Outputs

- `build_comparison_summary_dataframe`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `build_comparison_summary_dataframe`
- `build_run_detection_summary_payload`
- `DataFrame`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 47 — Define comparison-window center logic

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `astype`
- `broken`
- `center_on`
- `DataFrame`
- `dataframe`
- `def`
- `empty`
- `fillna`
- `iloc`
- `loc`
- `plot_order_index`
- `target_flag_column`

### Outputs

- `alert_candidates`
- `first_broken_plot_order_index`
- `resolve_comparison_center_index`

### Key Operations

- `def resolve_comparison_center_index( dataframe: pd.DataFrame, target_flag_column: str, center_on: str = "broken",`: Defines notebook-local logic used later in the notebook.
- `) -> int: first_broken_plot_order_index = int(dataframe["first_broken_plot_order_index"].iloc[0]) if center_on == "broken": return first_broken_plot_order_index alert_candidates = `: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `fillna`
- `resolve_comparison_center_index`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def resolve_comparison_center_index( dataframe: pd.DataFrame, target_flag_column: str, center_on: str = "broken",` | Defines notebook-local logic used later in the notebook. |
| `) -> int: first_broken_plot_order_index = int(dataframe["first_broken_plot_order_index"].iloc[0]) if center_on == "broken": return first_broken_plot_order_index alert_candidates = ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 48 — Define comparison-window extraction

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `center_index`
- `copy`
- `dataframe`
- `DataFrame`
- `def`
- `loc`
- `plot_order_index`
- `relative_plot_index`
- `rows_after`
- `rows_before`

### Outputs

- `extract_comparison_window`
- `max_plot_order_index`
- `min_plot_order_index`
- `out`
- `window_df`

### Key Operations

- `def extract_comparison_window( dataframe: pd.DataFrame, center_index: int, rows_before: int, rows_after: int,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: out = dataframe.copy() min_plot_order_index = center_index - rows_before max_plot_order_index = center_index + rows_after window_df = out.loc[ (out["plot_order_i`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `copy`
- `extract_comparison_window`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def extract_comparison_window( dataframe: pd.DataFrame, center_index: int, rows_before: int, rows_after: int,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: out = dataframe.copy() min_plot_order_index = center_index - rows_before max_plot_order_index = center_index + rows_after window_df = out.loc[ (out["plot_order_i` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 49 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `alert`
- `anchor`
- `anomaly`
- `Any`
- `append`
- `astype`
- `available`
- `ax`
- `ax1`
- `axis`
- `axvline`
- `axvspan`
- `be`
- `before`
- `bold`
- `bool`
- `both`
- `BROKEN`
- `broken`
- `build`

### Outputs

- `add_alert_spans`
- `add_event_reference_lines`
- `alert_candidates`
- `alert_rows`
- `alpha`
- `ax2`
- `center_index`
- `center_on`
- `combine_legends`
- `context`
- `extract_centered_plot_window`
- `first_alert_plot_order_index`
- `first_broken_plot_order_index`
- `fontsize`
- `format_gold05_axis`
- `frameon`
- `get_first_alert_plot_order_index`
- `get_first_phase_plot_order_index`
- `get_timeline_event_context`
- `handles`

### Key Operations

- `def resolve_sensor_column_for_plot( dataframe: pd.DataFrame, preferred_sensor_column: str = DEFAULT_SENSOR_FOR_TIMELINE,`: Defines notebook-local logic used later in the notebook.
- `) -> str: if preferred_sensor_column in dataframe.columns: return preferred_sensor_column sensor_candidates = [c for c in dataframe.columns if c.startswith("sensor_")] if not senso`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def get_first_alert_plot_order_index( dataframe: pd.DataFrame, target_flag_column: str,`: Defines notebook-local logic used later in the notebook.
- `) -> Optional[int]: if target_flag_column not in dataframe.columns: return None alert_candidates = dataframe.loc[ dataframe[target_flag_column].fillna(0).astype(int) == 1, "plot_or`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def get_first_phase_plot_order_index( dataframe: pd.DataFrame, phase_name: str,`: Defines notebook-local logic used later in the notebook.
- `) -> Optional[int]: if "episode_phase" not in dataframe.columns: return None phase_candidates = dataframe.loc[ dataframe["episode_phase"].eq(phase_name), "plot_order_index", ] if p`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def get_timeline_event_context( dataframe: pd.DataFrame, target_flag_column: str,`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, Optional[int]]: out = dataframe.copy() if "first_broken_plot_order_index" not in out.columns: raise ValueError( "Missing first_broken_plot_order_index. " "Rerun the `: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def resolve_plot_center_index( dataframe: pd.DataFrame, target_flag_column: str, center_on: str = "broken",`: Defines notebook-local logic used later in the notebook.
- `) -> int: context = get_timeline_event_context( dataframe, target_flag_column=target_flag_column, ) first_alert_plot_order_index = context.get("first_alert_plot_order_index") first`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def extract_centered_plot_window( dataframe: pd.DataFrame, *, target_flag_column: str, center_on: str = "broken", rows_before: int = PLOT_WINDOW_BEFORE_CENTER, rows_after: int = PL`: Defines notebook-local logic used later in the notebook.
- `) -> tuple[pd.DataFrame, dict[str, Optional[int]]]: out = dataframe.copy() center_index = resolve_plot_center_index( out, target_flag_column=target_flag_column, center_on=center_on`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add_alert_spans`
- `add_event_reference_lines`
- `append`
- `astype`
- `axvline`
- `axvspan`
- `combine_legends`
- `copy`
- `eq`
- `extend`
- `extract_centered_plot_window`
- `fillna`
- `format_gold05_axis`
- `get`
- `get_first_alert_plot_order_index`
- `get_first_phase_plot_order_index`
- `get_legend_handles_labels`
- `get_timeline_event_context`
- `grid`
- `legend`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def resolve_sensor_column_for_plot( dataframe: pd.DataFrame, preferred_sensor_column: str = DEFAULT_SENSOR_FOR_TIMELINE,` | Defines notebook-local logic used later in the notebook. |
| `) -> str: if preferred_sensor_column in dataframe.columns: return preferred_sensor_column sensor_candidates = [c for c in dataframe.columns if c.startswith("sensor_")] if not senso` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def get_first_alert_plot_order_index( dataframe: pd.DataFrame, target_flag_column: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> Optional[int]: if target_flag_column not in dataframe.columns: return None alert_candidates = dataframe.loc[ dataframe[target_flag_column].fillna(0).astype(int) == 1, "plot_or` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def get_first_phase_plot_order_index( dataframe: pd.DataFrame, phase_name: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> Optional[int]: if "episode_phase" not in dataframe.columns: return None phase_candidates = dataframe.loc[ dataframe["episode_phase"].eq(phase_name), "plot_order_index", ] if p` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def get_timeline_event_context( dataframe: pd.DataFrame, target_flag_column: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, Optional[int]]: out = dataframe.copy() if "first_broken_plot_order_index" not in out.columns: raise ValueError( "Missing first_broken_plot_order_index. " "Rerun the ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def resolve_plot_center_index( dataframe: pd.DataFrame, target_flag_column: str, center_on: str = "broken",` | Defines notebook-local logic used later in the notebook. |
| `) -> int: context = get_timeline_event_context( dataframe, target_flag_column=target_flag_column, ) first_alert_plot_order_index = context.get("first_alert_plot_order_index") first` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def extract_centered_plot_window( dataframe: pd.DataFrame, *, target_flag_column: str, center_on: str = "broken", rows_before: int = PLOT_WINDOW_BEFORE_CENTER, rows_after: int = PL` | Defines notebook-local logic used later in the notebook. |
| `) -> tuple[pd.DataFrame, dict[str, Optional[int]]]: out = dataframe.copy() center_index = resolve_plot_center_index( out, target_flag_column=target_flag_column, center_on=center_on` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def add_alert_spans( ax, window_df: pd.DataFrame, target_flag_column: str, *, x_column: str = "relative_plot_index", alpha: float = PLOT_ALERT_SPAN_ALPHA,` | Defines notebook-local logic used later in the notebook. |
| `) -> None: if target_flag_column not in window_df.columns or x_column not in window_df.columns: return alert_rows = window_df.loc[ window_df[target_flag_column].fillna(0).astype(in` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def add_event_reference_lines( ax, event_context: dict[str, Optional[int]], *, x_min: float, x_max: float, center_index: int, include_recovery: bool = True,` | Defines notebook-local logic used later in the notebook. |
| `) -> None: marker_specs = [ ("BROKEN", "first_broken_plot_order_index", "--"), ("FIRST_ALERT", "first_alert_plot_order_index", ":"), ] if include_recovery: marker_specs.extend( [ (` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def format_gold05_axis( ax, *, title: str, xlabel: str, ylabel: str, grid: bool = True,` | Defines notebook-local logic used later in the notebook. |
| `) -> None: ax.set_title(title, fontsize=13, fontweight="bold", pad=12) ax.set_xlabel(xlabel) ax.set_ylabel(ylabel) ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True)) i` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def combine_legends( ax1, ax2=None, *, loc: str = "upper left", ncol: int = 1,` | Defines notebook-local logic used later in the notebook. |
| `) -> None: handles1, labels1 = ax1.get_legend_handles_labels() handles = list(handles1) labels = list(labels1) if ax2 is not None: handles2, labels2 = ax2.get_legend_handles_labels` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 50 — Define focused anomaly timeline plot

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add_alert_spans`
- `add_event_reference_lines`
- `alert`
- `Alert`
- `alerts`
- `anchor`
- `anchors`
- `astype`
- `available`
- `ax1`
- `axis`
- `be`
- `broken`
- `BROKEN`
- `center_plot_order_index`
- `centered`
- `coerce`
- `columns`
- `combine_legends`
- `context`

### Outputs

- `alert_df`
- `alpha`
- `ax2`
- `center_index`
- `center_label`
- `center_on`
- `label`
- `linestyle`
- `linewidth`
- `out`
- `plot_anomaly_timeline_window`
- `raw_center_index`
- `rows_after`
- `rows_before`
- `s`
- `target_flag_column`
- `title`
- `x_column`
- `x_max`
- `x_min`

### Key Operations

- `def plot_anomaly_timeline_window( dataframe: pd.DataFrame, sensor_column: str, center_on: str = "broken", # broken \| alert rows_before: int = PLOT_WINDOW_BEFORE_CENTER, rows_after:`: Defines notebook-local logic used later in the notebook.
- `) -> Figure: """ Plot one sensor, model score, selected alerts, and lifecycle anchors. The x-axis is centered so 0 means the selected anchor: - center_on="broken": 0 is the first B`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add_alert_spans`
- `add_event_reference_lines`
- `astype`
- `combine_legends`
- `copy`
- `extract_centered_plot_window`
- `fillna`
- `format_gold05_axis`
- `get`
- `grid`
- `nanmax`
- `nanmin`
- `plot`
- `plot_anomaly_timeline_window`
- `scatter`
- `set_ylabel`
- `subplots`
- `tight_layout`
- `to_numeric`
- `to_numpy`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def plot_anomaly_timeline_window( dataframe: pd.DataFrame, sensor_column: str, center_on: str = "broken", # broken \| alert rows_before: int = PLOT_WINDOW_BEFORE_CENTER, rows_after:` | Defines notebook-local logic used later in the notebook. |
| `) -> Figure: """ Plot one sensor, model score, selected alerts, and lifecycle anchors. The x-axis is centered so 0 means the selected anchor: - center_on="broken": 0 is the first B` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 51 — Create Gold 05 anomaly timeline output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_timeline_dataframe`
- `broken`
- `DEFAULT_SENSOR_FOR_TIMELINE`
- `plot_anomaly_timeline_window`
- `PLOT_WINDOW_AFTER_CENTER`
- `PLOT_WINDOW_BEFORE_CENTER`
- `resolve_sensor_column_for_plot`
- `show`

### Outputs

- `center_on`
- `preferred_sensor_column`
- `rows_after`
- `rows_before`
- `sensor_column`
- `sensor_column_for_plot`
- `timeline_fig`

### Key Operations

- `sensor_column_for_plot = resolve_sensor_column_for_plot( anomaly_timeline_dataframe, preferred_sensor_column=DEFAULT_SENSOR_FOR_TIMELINE,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `timeline_fig = plot_anomaly_timeline_window( anomaly_timeline_dataframe, sensor_column=sensor_column_for_plot, center_on="broken", rows_before=PLOT_WINDOW_BEFORE_CENTER, rows_after`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `plt.show()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `plot_anomaly_timeline_window`
- `resolve_sensor_column_for_plot`
- `show`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `sensor_column_for_plot = resolve_sensor_column_for_plot( anomaly_timeline_dataframe, preferred_sensor_column=DEFAULT_SENSOR_FOR_TIMELINE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `timeline_fig = plot_anomaly_timeline_window( anomaly_timeline_dataframe, sensor_column=sensor_column_for_plot, center_on="broken", rows_before=PLOT_WINDOW_BEFORE_CENTER, rows_after` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.show()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 52 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `accidentally`
- `alert`
- `avoids`
- `boolean`
- `column_name`
- `columns`
- `compile`
- `d`
- `DataFrame`
- `dataframe`
- `def`
- `derived`
- `Expected`
- `flag`
- `format`
- `found`
- `indicators`
- `like`
- `match`
- `metadata`

### Outputs

- `raw_sensor_pattern`
- `resolve_sensor_columns`
- `sensor_columns`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `compile`
- `match`
- `resolve_sensor_columns`
- `sorted`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 53 — Define display-only sensor normalization

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abs`
- `approximates`
- `artifacts`
- `astype`
- `become`
- `before`
- `bool`
- `Boolean`
- `bools`
- `cannot`
- `change`
- `coerce`
- `column`
- `columns`
- `continue`
- `convertible`
- `Converts`
- `copy`
- `data`
- `dataframe`

### Outputs

- `iqr_value`
- `mad_value`
- `max_value`
- `mean_value`
- `median_value`
- `min_value`
- `normalize_sensor_columns_for_plot`
- `out`
- `q1_value`
- `q3_value`
- `range_value`
- `robust_scale`
- `series`
- `std_value`

### Key Operations

- `def normalize_sensor_columns_for_plot( dataframe: pd.DataFrame, sensor_columns: list[str], method: str = "zscore",`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """ Normalize sensor columns for multi-sensor visualization only. This function is display-only. It does not change model inputs, saved timeline data, truth reco`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `abs`
- `astype`
- `copy`
- `isnan`
- `max`
- `mean`
- `median`
- `min`
- `normalize_sensor_columns_for_plot`
- `notna`
- `quantile`
- `std`
- `sum`
- `to_numeric`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def normalize_sensor_columns_for_plot( dataframe: pd.DataFrame, sensor_columns: list[str], method: str = "zscore",` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Normalize sensor columns for multi-sensor visualization only. This function is display-only. It does not change model inputs, saved timeline data, truth reco` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 54 — Define display-only sensor normalization

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `across`
- `add_alert_spans`
- `add_event_reference_lines`
- `add_isometric_x_shift`
- `alert`
- `All`
- `all`
- `amplitude_scale`
- `anchor`
- `append`
- `around`
- `ax`
- `be`
- `bool`
- `BROKEN`
- `broken`
- `center_plot_order_index`
- `coerce`
- `column_name`
- `columns`

### Outputs

- `alpha`
- `center_index`
- `center_label`
- `center_on`
- `lane_base`
- `method`
- `out`
- `plot_all_sensors_stacked_waveform`
- `raw_center_index`
- `rows_after`
- `rows_before`
- `sensor_columns`
- `sensor_columns_resolved`
- `target_flag_column`
- `title`
- `window_df`
- `x_column`
- `x_max`
- `x_min`
- `x_plot`

### Key Operations

- `def plot_all_sensors_stacked_waveform( dataframe: pd.DataFrame, sensor_columns: Optional[list[str]] = None, center_on: str = "broken", rows_before: int = PLOT_WINDOW_BEFORE_CENTER,`: Defines notebook-local logic used later in the notebook.
- `) -> Figure: """ Plot all sensors as stacked, normalized waveforms around the selected anchor. Each sensor is normalized independently for display. This lets you see simultaneous m`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add_alert_spans`
- `add_event_reference_lines`
- `append`
- `combine_legends`
- `copy`
- `enumerate`
- `extract_centered_plot_window`
- `format_gold05_axis`
- `get`
- `nanmax`
- `nanmin`
- `normalize_sensor_columns_for_plot`
- `plot`
- `plot_all_sensors_stacked_waveform`
- `resolve_sensor_columns`
- `set_ylim`
- `set_yticklabels`
- `set_yticks`
- `subplots`
- `tight_layout`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def plot_all_sensors_stacked_waveform( dataframe: pd.DataFrame, sensor_columns: Optional[list[str]] = None, center_on: str = "broken", rows_before: int = PLOT_WINDOW_BEFORE_CENTER,` | Defines notebook-local logic used later in the notebook. |
| `) -> Figure: """ Plot all sensors as stacked, normalized waveforms around the selected anchor. Each sensor is normalized independently for display. This lets you see simultaneous m` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 55 — Create Gold 05 anomaly timeline output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_timeline_dataframe`
- `broken`
- `plot_all_sensors_stacked_waveform`
- `PLOT_WINDOW_AFTER_CENTER`
- `PLOT_WINDOW_BEFORE_CENTER`
- `resolve_sensor_columns`
- `show`
- `zscore`

### Outputs

- `add_isometric_x_shift`
- `amplitude_scale`
- `center_on`
- `lane_spacing`
- `normalize_method`
- `rows_after`
- `rows_before`
- `sensor_columns`
- `stacked_sensor_columns`
- `stacked_waveform_fig`
- `x_shift_per_sensor`

### Key Operations

- `stacked_sensor_columns = resolve_sensor_columns(anomaly_timeline_dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `stacked_waveform_fig = plot_all_sensors_stacked_waveform( anomaly_timeline_dataframe, sensor_columns=stacked_sensor_columns, center_on="broken", rows_before=PLOT_WINDOW_BEFORE_CENT`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `plt.show()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `plot_all_sensors_stacked_waveform`
- `resolve_sensor_columns`
- `show`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `stacked_sensor_columns = resolve_sensor_columns(anomaly_timeline_dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `stacked_waveform_fig = plot_all_sensors_stacked_waveform( anomaly_timeline_dataframe, sensor_columns=stacked_sensor_columns, center_on="broken", rows_before=PLOT_WINDOW_BEFORE_CENT` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.show()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 56 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `aligned`
- `baseline`
- `baseline_row_count`
- `be`
- `build_comparison_summary_dataframe`
- `build_run_timeline_dataframe`
- `Built`
- `can`
- `cascade`
- `cascade_defaults`
- `cascade_tuned`
- `Comparison`
- `comparison`
- `comparison_build`
- `comparison_row_count`
- `dataframe`
- `direct`
- `generated`
- `Gold`

### Outputs

- `baseline_payload`
- `comparison_payload`
- `COMPARISON_RUN_KEY`
- `comparison_summary_df`
- `consequence`
- `data`
- `kind`
- `logger`
- `message`
- `step`
- `why`

### Key Operations

- `COMPARISON_RUN_KEY = "cascade_tuned"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# valid:`: Documents the purpose or boundary of the surrounding notebook step.
- `# cascade_defaults`: Documents the purpose or boundary of the surrounding notebook step.
- `# cascade_tuned`: Documents the purpose or boundary of the surrounding notebook step.
- `# stage3_improved`: Documents the purpose or boundary of the surrounding notebook step.
- `baseline_payload = build_run_timeline_dataframe("baseline")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `comparison_payload = build_run_timeline_dataframe(COMPARISON_RUN_KEY)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `comparison_summary_df = build_comparison_summary_dataframe( [baseline_payload, comparison_payload]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="comparison_build", message="Built baseline vs comparison-run timeline payloads and summary dataframe.", why="Gold 05 should support direct baseline-v`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(comparison_summary_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `build_comparison_summary_dataframe`
- `build_run_timeline_dataframe`
- `display`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `COMPARISON_RUN_KEY = "cascade_tuned"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# valid:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# cascade_defaults` | Documents the purpose or boundary of the surrounding notebook step. |
| `# cascade_tuned` | Documents the purpose or boundary of the surrounding notebook step. |
| `# stage3_improved` | Documents the purpose or boundary of the surrounding notebook step. |
| `baseline_payload = build_run_timeline_dataframe("baseline")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `comparison_payload = build_run_timeline_dataframe(COMPARISON_RUN_KEY)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `comparison_summary_df = build_comparison_summary_dataframe( [baseline_payload, comparison_payload]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="comparison_build", message="Built baseline vs comparison-run timeline payloads and summary dataframe.", why="Gold 05 should support direct baseline-v` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(comparison_summary_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 57 — Define model lead-time comparison table

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `across`
- `add`
- `alone`
- `ANOMALY_DETECTION_SUMMARY_DIR`
- `artifact`
- `baseline`
- `build_comparison_summary_dataframe`
- `build_run_timeline_dataframe`
- `Built`
- `can`
- `cascade`
- `cascade_defaults`
- `cascade_tuned`
- `column_name`
- `columns`
- `compare`
- `comparison`
- `copy`
- `csv`

### Outputs

- `consequence`
- `data`
- `index`
- `kind`
- `lead_time_comparison_columns`
- `lead_time_comparison_df`
- `LEAD_TIME_RUN_KEYS`
- `lead_time_run_payloads`
- `logger`
- `message`
- `multi_run_lead_time_path`
- `step`
- `why`

### Key Operations

- `LEAD_TIME_RUN_KEYS = [ "baseline", "cascade_defaults", "cascade_tuned", "stage3_improved", "stage3_medium", "stage3_strict",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `lead_time_run_payloads = [ build_run_timeline_dataframe(run_key) for run_key in LEAD_TIME_RUN_KEYS`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `lead_time_comparison_df = build_comparison_summary_dataframe( lead_time_run_payloads`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `lead_time_comparison_columns = [ "selected_run_key", "plot_run_label", "run_family", "target_flag_column", "first_alert_plot_order_index", "first_broken_plot_order_index", "lead_ro`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `lead_time_comparison_columns = [ column_name for column_name in lead_time_comparison_columns if column_name in lead_time_comparison_df.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `lead_time_comparison_df = lead_time_comparison_df.loc[ :, lead_time_comparison_columns,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `].copy()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `build_comparison_summary_dataframe`
- `build_run_timeline_dataframe`
- `copy`
- `display`
- `mkdir`
- `to_csv`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `LEAD_TIME_RUN_KEYS = [ "baseline", "cascade_defaults", "cascade_tuned", "stage3_improved", "stage3_medium", "stage3_strict",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `lead_time_run_payloads = [ build_run_timeline_dataframe(run_key) for run_key in LEAD_TIME_RUN_KEYS` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `lead_time_comparison_df = build_comparison_summary_dataframe( lead_time_run_payloads` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `lead_time_comparison_columns = [ "selected_run_key", "plot_run_label", "run_family", "target_flag_column", "first_alert_plot_order_index", "first_broken_plot_order_index", "lead_ro` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `lead_time_comparison_columns = [ column_name for column_name in lead_time_comparison_columns if column_name in lead_time_comparison_df.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `lead_time_comparison_df = lead_time_comparison_df.loc[ :, lead_time_comparison_columns,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(lead_time_comparison_df)` | Displays a notebook-facing result for inspection. |
| `# ------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Save multi-run lead-time comparison artifact` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ------------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `ANOMALY_DETECTION_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `multi_run_lead_time_path = ( ANOMALY_DETECTION_SUMMARY_DIR / "multi_run_lead_time_comparison.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `lead_time_comparison_df.to_csv( multi_run_lead_time_path, index=False,` | Writes an artifact or output used for review or downstream notebooks. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"Saved multi-run lead-time comparison: {multi_run_lead_time_path}")` | Displays a notebook-facing result for inspection. |
| `ledger.add( kind="step", step="multi_run_lead_time_comparison", message="Built multi-run lead-time comparison across baseline, cascade, and Stage 3 operating modes.", why="Precisio` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 58 — Multi-Run Lead-Time Comparison Chart

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__multi_run_lead_time_comparison`
- `alpha`
- `ANOMALY_DETECTION_PLOT_DIR`
- `ANOMALY_DETECTION_SUMMARY_DIR`
- `ax`
- `axhline`
- `axis`
- `bar`
- `bbox_inches`
- `Before`
- `bottom`
- `Broken`
- `by`
- `center`
- `chart`
- `copy`
- `csv`
- `dpi`
- `Early`
- `else`

### Outputs

- `bars`
- `ha`
- `lead_time_plot_path`
- `multi_run_lead_time_dataframe`
- `multi_run_lead_time_path`
- `plot_frame`
- `va`

### Key Operations

- `multi_run_lead_time_path = ( ANOMALY_DETECTION_SUMMARY_DIR / "multi_run_lead_time_comparison.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `multi_run_lead_time_dataframe = pd.read_csv(multi_run_lead_time_path)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `display(multi_run_lead_time_dataframe)`: Displays a notebook-facing result for inspection.
- `plot_frame = multi_run_lead_time_dataframe.copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `plot_frame["lead_time_hours_to_failure"] = ( plot_frame["lead_time_minutes_to_failure"] / 60`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `fig, ax = plt.subplots(figsize=(10, 5))`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `bars = ax.bar( plot_frame["plot_run_label"], plot_frame["lead_time_hours_to_failure"],`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ax.axhline(0, linewidth=1)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ax.set_title("Early-Warning Lead Time by Model Run")`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `axhline`
- `bar`
- `copy`
- `display`
- `get_height`
- `get_width`
- `get_x`
- `grid`
- `read_csv`
- `Row`
- `savefig`
- `set_title`
- `set_xlabel`
- `set_ylabel`
- `show`
- `subplots`
- `text`
- `tight_layout`
- `xticks`
- `zip`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `multi_run_lead_time_path = ( ANOMALY_DETECTION_SUMMARY_DIR / "multi_run_lead_time_comparison.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `multi_run_lead_time_dataframe = pd.read_csv(multi_run_lead_time_path)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `display(multi_run_lead_time_dataframe)` | Displays a notebook-facing result for inspection. |
| `plot_frame = multi_run_lead_time_dataframe.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `plot_frame["lead_time_hours_to_failure"] = ( plot_frame["lead_time_minutes_to_failure"] / 60` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `fig, ax = plt.subplots(figsize=(10, 5))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `bars = ax.bar( plot_frame["plot_run_label"], plot_frame["lead_time_hours_to_failure"],` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ax.axhline(0, linewidth=1)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ax.set_title("Early-Warning Lead Time by Model Run")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ax.set_ylabel("Lead Time Before First Broken Row (Hours)")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ax.set_xlabel("Model Run")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ax.grid(axis="y", linewidth=0.5, alpha=0.4)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for bar, value in zip(bars, plot_frame["lead_time_hours_to_failure"]): ax.text( bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,.1f}", ha="center", va="bottom" if va` | Controls validation, iteration, file handling, or error handling for this step. |
| `plt.xticks(rotation=20)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.tight_layout()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `lead_time_plot_path = ( ANOMALY_DETECTION_PLOT_DIR / f"{SELECTED_RUN_KEY}__multi_run_lead_time_comparison.png"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `fig.savefig(lead_time_plot_path, dpi=200, bbox_inches="tight")` | Writes an artifact or output used for review or downstream notebooks. |
| `plt.show()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print(f"Saved multi-run lead-time chart: {lead_time_plot_path}")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: plot/image artifact.

## Code Cell 59 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `alert`
- `alerts`
- `Any`
- `astype`
- `ax1`
- `axvline`
- `Baseline`
- `baseline_context`
- `baseline_payload`
- `baseline_window_df`
- `broken`
- `BROKEN`
- `Center`
- `coerce`
- `columns`
- `combine_legends`
- `comparison_context`
- `comparison_payload`
- `comparison_window_df`
- `copy`

### Outputs

- `alpha`
- `ax2`
- `baseline_alert_df`
- `baseline_df`
- `baseline_score_column`
- `baseline_target_flag_column`
- `center_label`
- `center_on`
- `comparison_alert_df`
- `comparison_df`
- `comparison_score_column`
- `comparison_target_flag_column`
- `label`
- `linestyle`
- `linewidth`
- `marker`
- `plot_comparison_overlay`
- `rows_after`
- `rows_before`
- `s`

### Key Operations

- `def plot_comparison_overlay( baseline_payload: dict[str, Any], comparison_payload: dict[str, Any], sensor_column: str, center_on: str = "broken", rows_before: int = PLOT_WINDOW_BEF`: Defines notebook-local logic used later in the notebook.
- `) -> Figure: baseline_df = baseline_payload["timeline_dataframe"].copy() comparison_df = comparison_payload["timeline_dataframe"].copy() baseline_target_flag_column = baseline_payl`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `axvline`
- `combine_legends`
- `copy`
- `extract_centered_plot_window`
- `fillna`
- `format_gold05_axis`
- `grid`
- `plot`
- `plot_comparison_overlay`
- `scatter`
- `set_ylabel`
- `subplots`
- `tight_layout`
- `to_numeric`
- `twinx`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def plot_comparison_overlay( baseline_payload: dict[str, Any], comparison_payload: dict[str, Any], sensor_column: str, center_on: str = "broken", rows_before: int = PLOT_WINDOW_BEF` | Defines notebook-local logic used later in the notebook. |
| `) -> Figure: baseline_df = baseline_payload["timeline_dataframe"].copy() comparison_df = comparison_payload["timeline_dataframe"].copy() baseline_target_flag_column = baseline_payl` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 60 — Generate focused anomaly timeline plot

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `broken`
- `DEFAULT_SENSOR_FOR_TIMELINE`
- `plot_comparison_overlay`
- `PLOT_WINDOW_AFTER_CENTER`
- `PLOT_WINDOW_BEFORE_CENTER`
- `resolve_sensor_column_for_plot`
- `show`
- `timeline_dataframe`

### Outputs

- `baseline_payload`
- `center_on`
- `comparison_fig`
- `comparison_payload`
- `comparison_sensor_column`
- `preferred_sensor_column`
- `rows_after`
- `rows_before`
- `sensor_column`

### Key Operations

- `comparison_sensor_column = resolve_sensor_column_for_plot( baseline_payload["timeline_dataframe"], preferred_sensor_column=DEFAULT_SENSOR_FOR_TIMELINE,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `comparison_fig = plot_comparison_overlay( baseline_payload=baseline_payload, comparison_payload=comparison_payload, sensor_column=comparison_sensor_column, center_on="broken", rows`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `plt.show()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `plot_comparison_overlay`
- `resolve_sensor_column_for_plot`
- `show`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `comparison_sensor_column = resolve_sensor_column_for_plot( baseline_payload["timeline_dataframe"], preferred_sensor_column=DEFAULT_SENSOR_FOR_TIMELINE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_fig = plot_comparison_overlay( baseline_payload=baseline_payload, comparison_payload=comparison_payload, sensor_column=comparison_sensor_column, center_on="broken", rows` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.show()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 61 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `After`
- `append`
- `astype`
- `axis`
- `between`
- `Build`
- `clip`
- `clip_value`
- `coerce`
- `column`
- `column_name`
- `columns`
- `continue`
- `conversion`
- `copy`
- `dataframe`
- `DataFrame`
- `def`
- `drop`

### Outputs

- `available_sensor_columns`
- `build_sensor_matrix_for_plot`
- `converted_series`
- `errors`
- `method`
- `normalized_sensor_df`
- `numeric_sensor_df`
- `out`
- `sensor_columns`
- `window_df`
- `x_values`
- `y_values`
- `z_matrix`

### Key Operations

- `def build_sensor_matrix_for_plot( dataframe: pd.DataFrame, *, sensor_columns: list[str], time_axis_column: str, start_index: int, end_index: int, normalize_method: Optional[str] = `: Defines notebook-local logic used later in the notebook.
- `) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]: """ Build a numeric sensor matrix for heatmap / 3D plotting. Returns ------- window_df: Filtered dataframe for the requ`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `append`
- `astype`
- `between`
- `build_sensor_matrix_for_plot`
- `clip`
- `copy`
- `DataFrame`
- `normalize_sensor_columns_for_plot`
- `notna`
- `reset_index`
- `sort_values`
- `sum`
- `to_numeric`
- `to_numpy`
- `tolist`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_sensor_matrix_for_plot( dataframe: pd.DataFrame, *, sensor_columns: list[str], time_axis_column: str, start_index: int, end_index: int, normalize_method: Optional[str] = ` | Defines notebook-local logic used later in the notebook. |
| `) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]: """ Build a numeric sensor matrix for heatmap / 3D plotting. Returns ------- window_df: Filtered dataframe for the requ` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 62 — Define all-sensor heatmap plot

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add_event_reference_lines`
- `alert`
- `All`
- `arange`
- `auto`
- `ax`
- `be`
- `broken`
- `BROKEN`
- `build_sensor_matrix_for_plot`
- `center_plot_order_index`
- `clipped`
- `colorbar`
- `column_name`
- `columns`
- `combine_legends`
- `context`
- `copy`
- `could`
- `dataframe`

### Outputs

- `aspect`
- `center_index`
- `center_label`
- `center_on`
- `clip_value`
- `colorbar_label`
- `end_index`
- `extent`
- `grid`
- `im`
- `interpolation`
- `normalize_method`
- `out`
- `plot_all_sensors_heatmap`
- `raw_center_index`
- `rows_after`
- `rows_before`
- `sensor_columns`
- `sensor_columns_resolved`
- `start_index`

### Key Operations

- `def plot_all_sensors_heatmap( dataframe: pd.DataFrame, sensor_columns: Optional[list[str]] = None, center_on: str = "broken", time_axis_column: str = "plot_order_index", rows_befor`: Defines notebook-local logic used later in the notebook.
- `) -> Figure: out = dataframe.copy() if sensor_columns is None: sensor_columns_resolved: list[str] = resolve_sensor_columns(out) else: sensor_columns_resolved = [str(column_name) fo`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add_event_reference_lines`
- `arange`
- `build_sensor_matrix_for_plot`
- `colorbar`
- `combine_legends`
- `copy`
- `extract_centered_plot_window`
- `format_gold05_axis`
- `get`
- `Heatmap`
- `imshow`
- `max`
- `min`
- `nanmax`
- `nanmin`
- `plot_all_sensors_heatmap`
- `resolve_sensor_columns`
- `set_yticklabels`
- `set_yticks`
- `subplots`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def plot_all_sensors_heatmap( dataframe: pd.DataFrame, sensor_columns: Optional[list[str]] = None, center_on: str = "broken", time_axis_column: str = "plot_order_index", rows_befor` | Defines notebook-local logic used later in the notebook. |
| `) -> Figure: out = dataframe.copy() if sensor_columns is None: sensor_columns_resolved: list[str] = resolve_sensor_columns(out) else: sensor_columns_resolved = [str(column_name) fo` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 63 — Create Gold 05 anomaly timeline output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `anomaly_timeline_dataframe`
- `broken`
- `plot_all_sensors_heatmap`
- `plot_order_index`
- `PLOT_WINDOW_AFTER_CENTER`
- `PLOT_WINDOW_BEFORE_CENTER`
- `resolve_sensor_columns`
- `show`

### Outputs

- `all_sensor_columns`
- `all_sensor_heatmap_fig`
- `center_on`
- `rows_after`
- `rows_before`
- `sensor_columns`
- `time_axis_column`

### Key Operations

- `all_sensor_columns = resolve_sensor_columns(anomaly_timeline_dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `all_sensor_heatmap_fig = plot_all_sensors_heatmap( anomaly_timeline_dataframe, sensor_columns=all_sensor_columns, center_on="broken", time_axis_column="plot_order_index", rows_befo`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `plt.show()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `plot_all_sensors_heatmap`
- `resolve_sensor_columns`
- `show`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `all_sensor_columns = resolve_sensor_columns(anomaly_timeline_dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `all_sensor_heatmap_fig = plot_all_sensors_heatmap( anomaly_timeline_dataframe, sensor_columns=all_sensor_columns, center_on="broken", time_axis_column="plot_order_index", rows_befo` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.show()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 64 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `add_subplot`
- `alert`
- `all`
- `All`
- `anchor`
- `arange`
- `asarray`
- `Axes3D`
- `axis`
- `azim`
- `be`
- `broken`
- `BROKEN`
- `build_sensor_matrix_for_plot`
- `by`
- `cast`
- `ceil`
- `center_index`
- `center_plot_order_index`

### Outputs

- `alpha`
- `antialiased`
- `ax`
- `center_label`
- `center_on`
- `clip_value`
- `dtype`
- `end_index`
- `fig`
- `label_step`
- `labeled_positions`
- `labeled_values`
- `linewidth`
- `normalize_method`
- `out`
- `pad`
- `plot_all_sensors_3d_surface`
- `raw_center_index`
- `rows_after`
- `rows_before`

### Key Operations

- `def plot_all_sensors_3d_surface( dataframe: pd.DataFrame, *, sensor_columns: list[str], center_on: str = "broken", time_axis_column: str = "plot_order_index", rows_before: int = 30`: Defines notebook-local logic used later in the notebook.
- `) -> Figure: """ Plot all sensor values as a 3D surface. Notes ----- - X axis = rows from the selected anchor - Y axis = numeric sensor position - Z axis = normalized sensor value `: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add_subplot`
- `arange`
- `asarray`
- `build_sensor_matrix_for_plot`
- `cast`
- `ceil`
- `copy`
- `extract_centered_plot_window`
- `Figure`
- `filled`
- `get`
- `max`
- `meshgrid`
- `min`
- `plot_all_sensors_3d_surface`
- `plot_surface`
- `set_title`
- `set_xlabel`
- `set_ylabel`
- `set_yticklabels`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def plot_all_sensors_3d_surface( dataframe: pd.DataFrame, *, sensor_columns: list[str], center_on: str = "broken", time_axis_column: str = "plot_order_index", rows_before: int = 30` | Defines notebook-local logic used later in the notebook. |
| `) -> Figure: """ Plot all sensor values as a 3D surface. Notes ----- - X axis = rows from the selected anchor - Y axis = numeric sensor position - Z axis = normalized sensor value ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 65 — Create Gold 05 anomaly timeline output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `all_sensor_columns`
- `anomaly_timeline_dataframe`
- `broken`
- `plot_all_sensors_3d_surface`
- `plot_order_index`
- `PLOT_WINDOW_AFTER_CENTER`
- `PLOT_WINDOW_BEFORE_CENTER`
- `show`

### Outputs

- `all_sensor_3d_fig`
- `center_on`
- `rows_after`
- `rows_before`
- `sensor_columns`
- `time_axis_column`

### Key Operations

- `all_sensor_3d_fig = plot_all_sensors_3d_surface( anomaly_timeline_dataframe, sensor_columns=all_sensor_columns, center_on="broken", time_axis_column="plot_order_index", rows_before`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `plt.show()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `plot_all_sensors_3d_surface`
- `show`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `all_sensor_3d_fig = plot_all_sensors_3d_surface( anomaly_timeline_dataframe, sensor_columns=all_sensor_columns, center_on="broken", time_axis_column="plot_order_index", rows_before` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `plt.show()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 66 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `alert`
- `Alert`
- `alert_packet_summary_df`
- `Built`
- `can`
- `contains_pre_failure_alert`
- `dataframe`
- `DataFrame`
- `drop`
- `early`
- `else`
- `empty`
- `examples`
- `focused`
- `Gold`
- `head`
- `highlight`
- `illustrative`
- `inspection`

### Outputs

- `ascending`
- `by`
- `consequence`
- `data`
- `kind`
- `logger`
- `message`
- `step`
- `top_alert_packets_df`
- `TOP_K_ALERT_PACKETS`
- `why`

### Key Operations

- `# ============================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Top-K Alert Packet Review`: Documents the purpose or boundary of the surrounding notebook step.
- `# ============================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `TOP_K_ALERT_PACKETS = 5`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `top_alert_packets_df = ( alert_packet_summary_df.sort_values( by=["contains_pre_failure_alert", "rows_from_packet_start_to_broken", "packet_row_count"], ascending=[False, False, Fa`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="top_k_alert_packets", message="Built top-k alert packet review dataframe.", why="Gold 05 should surface the most illustrative alert packets for focus`: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(top_alert_packets_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `add`
- `DataFrame`
- `display`
- `head`
- `reset_index`
- `sort_values`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# ============================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Top-K Alert Packet Review` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ============================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `TOP_K_ALERT_PACKETS = 5` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `top_alert_packets_df = ( alert_packet_summary_df.sort_values( by=["contains_pre_failure_alert", "rows_from_packet_start_to_broken", "packet_row_count"], ascending=[False, False, Fa` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="top_k_alert_packets", message="Built top-k alert packet review dataframe.", why="Gold 05 should surface the most illustrative alert packets for focus` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(top_alert_packets_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 67 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add_alert_spans`
- `add_event_reference_lines`
- `alert`
- `Alert`
- `around`
- `astype`
- `ax1`
- `axvline`
- `center_plot_order_index`
- `Centered`
- `coerce`
- `columns`
- `combine_legends`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `dtype`
- `empty`
- `errors`

### Outputs

- `alert_df`
- `alpha`
- `ax2`
- `center_index`
- `event_context`
- `label`
- `linestyle`
- `linewidth`
- `max_plot_order_index`
- `min_plot_order_index`
- `out`
- `plot_packet_centered_window`
- `s`
- `target_flag_column`
- `title`
- `window_df`
- `x_column`
- `x_max`
- `x_min`
- `x_values`

### Key Operations

- `# ============================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Packet-Centered Plot Helper`: Documents the purpose or boundary of the surrounding notebook step.
- `# ============================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `def plot_packet_centered_window( dataframe: pd.DataFrame, packet_start_plot_order_index: int, sensor_column: str, rows_before: int = 120, rows_after: int = 120, figsize: tuple[int,`: Defines notebook-local logic used later in the notebook.
- `) -> Figure: out = dataframe.copy() min_plot_order_index = packet_start_plot_order_index - rows_before max_plot_order_index = packet_start_plot_order_index + rows_after window_df =`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add_alert_spans`
- `add_event_reference_lines`
- `astype`
- `axvline`
- `combine_legends`
- `copy`
- `fillna`
- `format_gold05_axis`
- `get_timeline_event_context`
- `grid`
- `nanmax`
- `nanmin`
- `plot`
- `plot_packet_centered_window`
- `scatter`
- `set_ylabel`
- `subplots`
- `tight_layout`
- `to_numeric`
- `to_numpy`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# ============================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Packet-Centered Plot Helper` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ============================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `def plot_packet_centered_window( dataframe: pd.DataFrame, packet_start_plot_order_index: int, sensor_column: str, rows_before: int = 120, rows_after: int = 120, figsize: tuple[int,` | Defines notebook-local logic used later in the notebook. |
| `) -> Figure: out = dataframe.copy() min_plot_order_index = packet_start_plot_order_index - rows_before max_plot_order_index = packet_start_plot_order_index + rows_after window_df =` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 68 — Create Gold 05 anomaly timeline output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `anomaly_timeline_dataframe`
- `append`
- `DEFAULT_SENSOR_FOR_TIMELINE`
- `empty`
- `iterrows`
- `packet_row`
- `plot_packet_centered_window`
- `resolve_sensor_column_for_plot`
- `show`
- `top_alert_packets_df`

### Outputs

- `packet_fig`
- `packet_figures`
- `packet_sensor_column`
- `packet_start_plot_order_index`
- `preferred_sensor_column`
- `rows_after`
- `rows_before`
- `sensor_column`

### Key Operations

- `packet_figures = []`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if not top_alert_packets_df.empty: packet_sensor_column = resolve_sensor_column_for_plot( anomaly_timeline_dataframe, preferred_sensor_column=DEFAULT_SENSOR_FOR_TIMELINE, ) for _, `: Controls validation, iteration, file handling, or error handling for this step.

Important functions or methods detected:
- `append`
- `iterrows`
- `plot_packet_centered_window`
- `resolve_sensor_column_for_plot`
- `show`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `packet_figures = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not top_alert_packets_df.empty: packet_sensor_column = resolve_sensor_column_for_plot( anomaly_timeline_dataframe, preferred_sensor_column=DEFAULT_SENSOR_FOR_TIMELINE, ) for _, ` | Controls validation, iteration, file handling, or error handling for this step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 69 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__alert_packet_summary`
- `__detection_summary`
- `__failure_lead_time_summary`
- `__stacked_sensor_waveform`
- `__timeline_export`
- `__timeline_plot`
- `active`
- `add`
- `alert_packet_summary_df`
- `ANOMALY_DETECTION_EXPORT_DIR`
- `ANOMALY_DETECTION_PACKET_DIR`
- `ANOMALY_DETECTION_PLOT_DIR`
- `ANOMALY_DETECTION_SUMMARY_DIR`
- `anomaly_timeline_dataframe`
- `are`
- `artifacts`
- `available`
- `bbox_inches`
- `be`
- `copy`

### Outputs

- `alert_packet_export_path`
- `consequence`
- `data`
- `failure_lead_time_export_path`
- `kind`
- `logger`
- `message`
- `stacked_waveform_export_path`
- `step`
- `summary_payload_export_path`
- `timeline_export_df`
- `timeline_export_path`
- `timeline_plot_export_path`
- `why`

### Key Operations

- `# ============================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Export Outputs`: Documents the purpose or boundary of the surrounding notebook step.
- `# ============================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `timeline_export_path = ( Path(ANOMALY_DETECTION_EXPORT_DIR) / f"{SELECTED_RUN_KEY}__timeline_export.parquet"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `failure_lead_time_export_path = ( Path(ANOMALY_DETECTION_SUMMARY_DIR) / f"{SELECTED_RUN_KEY}__failure_lead_time_summary.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `alert_packet_export_path = ( Path(ANOMALY_DETECTION_PACKET_DIR) / f"{SELECTED_RUN_KEY}__alert_packet_summary.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `summary_payload_export_path = ( Path(ANOMALY_DETECTION_SUMMARY_DIR) / f"{SELECTED_RUN_KEY}__detection_summary.json"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `timeline_plot_export_path = ( Path(ANOMALY_DETECTION_PLOT_DIR) / f"{SELECTED_RUN_KEY}__timeline_plot.png"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `copy`
- `Path`
- `save_data`
- `save_json`
- `savefig`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# ============================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Export Outputs` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ============================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `timeline_export_path = ( Path(ANOMALY_DETECTION_EXPORT_DIR) / f"{SELECTED_RUN_KEY}__timeline_export.parquet"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `failure_lead_time_export_path = ( Path(ANOMALY_DETECTION_SUMMARY_DIR) / f"{SELECTED_RUN_KEY}__failure_lead_time_summary.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `alert_packet_export_path = ( Path(ANOMALY_DETECTION_PACKET_DIR) / f"{SELECTED_RUN_KEY}__alert_packet_summary.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `summary_payload_export_path = ( Path(ANOMALY_DETECTION_SUMMARY_DIR) / f"{SELECTED_RUN_KEY}__detection_summary.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `timeline_plot_export_path = ( Path(ANOMALY_DETECTION_PLOT_DIR) / f"{SELECTED_RUN_KEY}__timeline_plot.png"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stacked_waveform_export_path = ( Path(ANOMALY_DETECTION_PLOT_DIR) / f"{SELECTED_RUN_KEY}__stacked_sensor_waveform.png"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `timeline_export_df = anomaly_timeline_dataframe.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `timeline_export_path = save_data( timeline_export_df, timeline_export_path.parent, timeline_export_path.name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `failure_lead_time_export_path = save_data( failure_lead_time_df, failure_lead_time_export_path.parent, failure_lead_time_export_path.name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if not alert_packet_summary_df.empty: alert_packet_export_path = save_data( alert_packet_summary_df, alert_packet_export_path.parent, alert_packet_export_path.name, )` | Controls validation, iteration, file handling, or error handling for this step. |
| `save_json( detection_summary_payload, summary_payload_export_path,` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `timeline_fig.savefig(timeline_plot_export_path, bbox_inches="tight")` | Writes an artifact or output used for review or downstream notebooks. |
| `stacked_waveform_fig.savefig(stacked_waveform_export_path, bbox_inches="tight")` | Writes an artifact or output used for review or downstream notebooks. |
| `ledger.add( kind="step", step="export_outputs", message="Exported Gold 05 timeline outputs and plots.", why="Notebook artifacts should be persisted for reporting, dashboarding, and` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Exports written:")` | Displays a notebook-facing result for inspection. |
| `print(f" - {timeline_export_path}")` | Displays a notebook-facing result for inspection. |
| `print(f" - {failure_lead_time_export_path}")` | Displays a notebook-facing result for inspection. |
| `print(f" - {alert_packet_export_path}")` | Displays a notebook-facing result for inspection. |
| `print(f" - {summary_payload_export_path}")` | Loads input data, configuration, or artifacts required by the current stage. |
| `print(f" - {timeline_plot_export_path}")` | Displays a notebook-facing result for inspection. |
| `print(f" - {stacked_waveform_export_path}")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: plot/image artifact.

## Code Cell 70 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__all_sensors_3d_surface`
- `__all_sensors_heatmap`
- `__comparison_plot`
- `__comparison_summary`
- `__detected_rows_review`
- `__packet_`
- `__top_alert_packets`
- `__window_plot`
- `a`
- `add`
- `Additional`
- `additional_exports`
- `all_sensor_3d_fig`
- `all_sensor_heatmap_fig`
- `anomaly`
- `ANOMALY_DETECTION_EXPORT_DIR`
- `ANOMALY_DETECTION_PACKET_DIR`
- `ANOMALY_DETECTION_PLOT_DIR`
- `ANOMALY_DETECTION_SUMMARY_DIR`
- `artifact`

### Outputs

- `all_sensor_3d_plot_export_path`
- `all_sensor_heatmap_export_path`
- `bbox_inches`
- `comparison_plot_export_path`
- `comparison_summary_export_path`
- `consequence`
- `data`
- `detected_rows_export_path`
- `kind`
- `lead_time_comparison_export_path`
- `logger`
- `message`
- `packet_plot_export_path`
- `step`
- `top_alert_packets_export_path`
- `why`

### Key Operations

- `detected_rows_export_path = ( ANOMALY_DETECTION_EXPORT_DIR / f"{SELECTED_RUN_KEY}__detected_rows_review.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `comparison_summary_export_path = ( ANOMALY_DETECTION_SUMMARY_DIR / f"baseline_vs_{COMPARISON_RUN_KEY}__comparison_summary.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `comparison_plot_export_path = ( ANOMALY_DETECTION_PLOT_DIR / f"baseline_vs_{COMPARISON_RUN_KEY}__comparison_plot.png"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `lead_time_comparison_export_path = ( ANOMALY_DETECTION_SUMMARY_DIR / "multi_run_lead_time_comparison.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `lead_time_comparison_export_path = save_data( lead_time_comparison_df, lead_time_comparison_export_path.parent, lead_time_comparison_export_path.name,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `all_sensor_heatmap_export_path = ( ANOMALY_DETECTION_PLOT_DIR / f"{SELECTED_RUN_KEY}__all_sensors_heatmap.png"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `save_data`
- `savefig`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `detected_rows_export_path = ( ANOMALY_DETECTION_EXPORT_DIR / f"{SELECTED_RUN_KEY}__detected_rows_review.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_summary_export_path = ( ANOMALY_DETECTION_SUMMARY_DIR / f"baseline_vs_{COMPARISON_RUN_KEY}__comparison_summary.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_plot_export_path = ( ANOMALY_DETECTION_PLOT_DIR / f"baseline_vs_{COMPARISON_RUN_KEY}__comparison_plot.png"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `lead_time_comparison_export_path = ( ANOMALY_DETECTION_SUMMARY_DIR / "multi_run_lead_time_comparison.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `lead_time_comparison_export_path = save_data( lead_time_comparison_df, lead_time_comparison_export_path.parent, lead_time_comparison_export_path.name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `all_sensor_heatmap_export_path = ( ANOMALY_DETECTION_PLOT_DIR / f"{SELECTED_RUN_KEY}__all_sensors_heatmap.png"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `all_sensor_3d_plot_export_path = ( ANOMALY_DETECTION_PLOT_DIR / f"{SELECTED_RUN_KEY}__all_sensors_3d_surface.png"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `top_alert_packets_export_path = ( ANOMALY_DETECTION_PACKET_DIR / f"{SELECTED_RUN_KEY}__top_alert_packets.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `detected_rows_export_path = save_data( detected_rows_review_df, detected_rows_export_path.parent, detected_rows_export_path.name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_summary_export_path = save_data( comparison_summary_df, comparison_summary_export_path.parent, comparison_summary_export_path.name,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `comparison_fig.savefig( comparison_plot_export_path, bbox_inches="tight",` | Writes an artifact or output used for review or downstream notebooks. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `all_sensor_heatmap_fig.savefig( all_sensor_heatmap_export_path, bbox_inches="tight",` | Writes an artifact or output used for review or downstream notebooks. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `all_sensor_3d_fig.savefig( all_sensor_3d_plot_export_path, bbox_inches="tight",` | Writes an artifact or output used for review or downstream notebooks. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if not top_alert_packets_df.empty: top_alert_packets_export_path = save_data( top_alert_packets_df, top_alert_packets_export_path.parent, top_alert_packets_export_path.name, ) for ` | Writes an artifact or output used for review or downstream notebooks. |
| `ledger.add( kind="step", step="additional_exports", message="Exported detected-row review, comparison outputs, heatmap, 3D surface, and packet review artifacts.", why="Gold 05 shou` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Additional exports written:")` | Displays a notebook-facing result for inspection. |
| `print(f" - {detected_rows_export_path}")` | Displays a notebook-facing result for inspection. |
| `print(f" - {comparison_summary_export_path}")` | Displays a notebook-facing result for inspection. |
| `print(f" - {comparison_plot_export_path}")` | Displays a notebook-facing result for inspection. |
| `print(f" - {all_sensor_heatmap_export_path}")` | Displays a notebook-facing result for inspection. |
| `print(f" - {all_sensor_3d_plot_export_path}")` | Displays a notebook-facing result for inspection. |
| `print(f" - {top_alert_packets_export_path}")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: plot/image artifact.

## Code Cell 71 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `alert_packet_export_path`
- `all_sensor_3d_plot_export_path`
- `all_sensor_heatmap_export_path`
- `append_truth_index`
- `are`
- `artifact_paths`
- `artifacts`
- `based`
- `build_truth_record`
- `Built`
- `comparison_plot_export_path`
- `comparison_summary_export_path`
- `CONFIG`
- `config_hash`
- `config_snapshot`
- `CONFIG_SNAPSHOT_PATH`
- `config_snapshot_path`
- `config_sources`
- `DataFrame`

### Outputs

- `anomaly_timeline_dataframe`
- `column_count`
- `config_sources_for_truth`
- `consequence`
- `data`
- `dataset_name`
- `feature_columns`
- `gold05_truth_base`
- `gold05_truth_path`
- `gold05_truth_record`
- `kind`
- `layer_name`
- `logger`
- `message`
- `meta_columns`
- `parent_truth_hash`
- `pipeline_mode`
- `process_run_id`
- `raw_config_sources`
- `row_count`

### Key Operations

- `# ============================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Truth Record for Gold 05`: Documents the purpose or boundary of the surrounding notebook step.
- `# ============================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `parent_truth_hash = extract_truth_hash(selected_results)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `gold05_truth_base = initialize_layer_truth( truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name="gold_anomaly_detection", process_run_id=PROCESS_RUN_ID, pipeline_mod`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `raw_config_sources = CONFIG.get("config_sources", [])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if isinstance(raw_config_sources, list): config_sources_for_truth = [str(source_file) for source_file in raw_config_sources]`: Controls validation, iteration, file handling, or error handling for this step.
- `else: config_sources_for_truth = []`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold05_truth_base = update_truth_section( gold05_truth_base, "config_snapshot", { "selected_run_key": SELECTED_RUN_KEY, "target_flag_column": TARGET_FLAG_COLUMN, "primary_score_col`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold05_truth_base = update_truth_section( gold05_truth_base, "runtime_facts", { "row_count": int(len(anomaly_timeline_dataframe)), "column_count": int(anomaly_timeline_dataframe.sh`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `append_truth_index`
- `build_truth_record`
- `DataFrame`
- `display`
- `extract_truth_hash`
- `get`
- `identify_feature_columns`
- `identify_meta_columns`
- `initialize_layer_truth`
- `isinstance`
- `save_truth_record`
- `stamp_truth_columns`
- `update_truth_section`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# ============================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Truth Record for Gold 05` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ============================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `parent_truth_hash = extract_truth_hash(selected_results)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `gold05_truth_base = initialize_layer_truth( truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name="gold_anomaly_detection", process_run_id=PROCESS_RUN_ID, pipeline_mod` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `raw_config_sources = CONFIG.get("config_sources", [])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if isinstance(raw_config_sources, list): config_sources_for_truth = [str(source_file) for source_file in raw_config_sources]` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: config_sources_for_truth = []` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold05_truth_base = update_truth_section( gold05_truth_base, "config_snapshot", { "selected_run_key": SELECTED_RUN_KEY, "target_flag_column": TARGET_FLAG_COLUMN, "primary_score_col` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold05_truth_base = update_truth_section( gold05_truth_base, "runtime_facts", { "row_count": int(len(anomaly_timeline_dataframe)), "column_count": int(anomaly_timeline_dataframe.sh` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold05_truth_base = update_truth_section( gold05_truth_base, "artifact_paths", { "timeline_export_path": str(timeline_export_path), "failure_lead_time_export_path": str(failure_lea` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold05_truth_record = build_truth_record( truth_base=gold05_truth_base, row_count=len(anomaly_timeline_dataframe), column_count=anomaly_timeline_dataframe.shape[1], meta_columns=id` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold05_truth_path = save_truth_record( gold05_truth_record, truth_dir=paths.truths, dataset_name=DATASET_NAME, layer_name="gold_anomaly_detection",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `append_truth_index( gold05_truth_record, truth_index_path=RESOLVED_PATHS["truth_index_path"],` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `anomaly_timeline_dataframe = stamp_truth_columns( anomaly_timeline_dataframe, truth_hash=gold05_truth_record["truth_hash"], parent_truth_hash=parent_truth_hash, pipeline_mode=PIPEL` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="truth_record", message="Built and saved Gold 05 truth record and stamped the timeline dataframe.", why="Gold 05 outputs are derived artifacts and sho` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(pd.DataFrame([gold05_truth_record]))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: truth record.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 72 — Answer

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `f`
- `Finalize`
- `Gold`
- `GOLD05_LEDGER_PATH`
- `info`
- `ledger`
- `Ledger`
- `logger`
- `s`
- `to`
- `write_json`
- `written`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `# ============================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Finalize Ledger`: Documents the purpose or boundary of the surrounding notebook step.
- `# ============================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `ledger.write_json(GOLD05_LEDGER_PATH)`: Records or exports ledger information for stage-level traceability.
- `logger.info("Gold 05 ledger written to: %s", GOLD05_LEDGER_PATH)`: Writes a logger message for traceability during notebook execution.
- `print(f"Ledger written: {GOLD05_LEDGER_PATH}")`: Records or exports ledger information for stage-level traceability.

Important functions or methods detected:
- `info`
- `write_json`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# ============================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Finalize Ledger` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ============================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `ledger.write_json(GOLD05_LEDGER_PATH)` | Records or exports ledger information for stage-level traceability. |
| `logger.info("Gold 05 ledger written to: %s", GOLD05_LEDGER_PATH)` | Writes a logger message for traceability during notebook execution. |
| `print(f"Ledger written: {GOLD05_LEDGER_PATH}")` | Records or exports ledger information for stage-level traceability. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 73 — Gold 05 Anomaly Detection SQL Summary Cell

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__name__`
- `a`
- `alert_packet_summary`
- `alert_packet_summary_df`
- `anomaly_timeline`
- `anomaly_timeline_dataframe`
- `append`
- `because`
- `column_count`
- `columns`
- `continue`
- `detected_rows_review`
- `detected_rows_review_df`
- `else`
- `failure_lead_time`
- `failure_lead_time_df`
- `get`
- `globals`
- `Gold`
- `isinstance`

### Outputs

- `capstone_schema`
- `dataframe`
- `dataset_id`
- `dataset_name`
- `engine`
- `extra`
- `gold_05_sql_summary_dataframe`
- `gold05_output_frames`
- `gold05_output_manifest_df`
- `gold05_output_manifest_records`
- `notebook_globals`
- `run_id`
- `WRITE_TO_POSTGRES`

### Key Operations

- `WRITE_TO_POSTGRES = True`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if WRITE_TO_POSTGRES: gold05_output_frames = { "failure_lead_time": globals().get("failure_lead_time_df"), "detected_rows_review": globals().get("detected_rows_review_df"), "alert_`: Writes a logger message for traceability during notebook execution.
- `else: print("Postgres write skipped.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `append`
- `DataFrame`
- `display`
- `get`
- `globals`
- `isinstance`
- `items`
- `log_gold_05_anomaly_detection_summary_sql`
- `map`
- `type`
- `warning`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `WRITE_TO_POSTGRES = True` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if WRITE_TO_POSTGRES: gold05_output_frames = { "failure_lead_time": globals().get("failure_lead_time_df"), "detected_rows_review": globals().get("detected_rows_review_df"), "alert_` | Writes a logger message for traceability during notebook execution. |
| `else: print("Postgres write skipped.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 74 — Gold 05 Anomaly Detection SQL Summary Cell

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `capstone`
- `completed_at_utc`
- `dataset_id`
- `DATASET_ID`
- `DESC`
- `engine`
- `gold_anomaly_detection_summary`
- `LIMIT`
- `ORDER`
- `pipeline_runs`
- `pipeline_stage`
- `read_sql_dataframe`
- `RUN_ID`
- `run_id`
- `run_status`
- `runtime_facts`
- `SELECT`
- `source_run_id`
- `WHERE`

### Outputs

- `gold05_pipeline_check_df`
- `gold05_pipeline_check_sql`
- `params`

### Key Operations

- `gold05_pipeline_check_sql = """`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SELECT pipeline_stage, run_status, completed_at_utc, runtime_facts`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `FROM capstone.pipeline_runs`: Imports a dependency or project helper used by later cells.
- `WHERE dataset_id = :dataset_id AND runtime_facts ->> 'source_run_id' = :run_id AND pipeline_stage = 'gold_anomaly_detection_summary'`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ORDER BY completed_at_utc DESC`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `LIMIT 3;`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `"""`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold05_pipeline_check_df = read_sql_dataframe( engine, gold05_pipeline_check_sql, params={ "dataset_id": DATASET_ID, "run_id": RUN_ID, },`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(gold05_pipeline_check_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `gold05_pipeline_check_sql = """` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SELECT pipeline_stage, run_status, completed_at_utc, runtime_facts` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FROM capstone.pipeline_runs` | Imports a dependency or project helper used by later cells. |
| `WHERE dataset_id = :dataset_id AND runtime_facts ->> 'source_run_id' = :run_id AND pipeline_stage = 'gold_anomaly_detection_summary'` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ORDER BY completed_at_utc DESC` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `LIMIT 3;` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `"""` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold05_pipeline_check_df = read_sql_dataframe( engine, gold05_pipeline_check_sql, params={ "dataset_id": DATASET_ID, "run_id": RUN_ID, },` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(gold05_pipeline_check_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 75 — Gold 05 Anomaly Detection SQL Summary Cell

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `BY`
- `capstone`
- `check_name`
- `check_status`
- `created_at_utc`
- `data_quality_events`
- `DATASET_ID`
- `dataset_id`
- `DESC`
- `details_json`
- `engine`
- `gold_05_summary_sql_log`
- `layer_name`
- `LIMIT`
- `ORDER`
- `read_sql_dataframe`
- `row_count`
- `RUN_ID`
- `run_id`
- `SELECT`

### Outputs

- `gold05_dq_check_df`
- `gold05_dq_check_sql`
- `params`

### Key Operations

- `gold05_dq_check_sql = """`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SELECT layer_name, table_name, check_name, check_status, row_count, details_json, created_at_utc`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `FROM capstone.data_quality_events`: Imports a dependency or project helper used by later cells.
- `WHERE dataset_id = :dataset_id AND run_id = :run_id AND check_name = 'gold_05_summary_sql_log'`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ORDER BY created_at_utc DESC`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `LIMIT 5;`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `"""`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `gold05_dq_check_df = read_sql_dataframe( engine, gold05_dq_check_sql, params={ "dataset_id": DATASET_ID, "run_id": RUN_ID, },`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(gold05_dq_check_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `read_sql_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `gold05_dq_check_sql = """` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SELECT layer_name, table_name, check_name, check_status, row_count, details_json, created_at_utc` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FROM capstone.data_quality_events` | Imports a dependency or project helper used by later cells. |
| `WHERE dataset_id = :dataset_id AND run_id = :run_id AND check_name = 'gold_05_summary_sql_log'` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ORDER BY created_at_utc DESC` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `LIMIT 5;` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `"""` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `gold05_dq_check_df = read_sql_dataframe( engine, gold05_dq_check_sql, params={ "dataset_id": DATASET_ID, "run_id": RUN_ID, },` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(gold05_dq_check_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

