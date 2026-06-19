# Notebook Code Reference: EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation

Notebook path:

`notebooks/experiments/EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation.ipynb`

## Notebook Purpose

This notebook replays test data through saved model artifacts and compares replay behavior to training-run outputs.

Notebook stage:

`Gold`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| 1. Imports and notebook context | Code Cell 01, Code Cell 02, Code Cell 03 |
| 2. Resolve source artifacts and output folders | Code Cell 04 |
| 3. Load test source and training artifacts | Code Cell 05 |
| 4. Replay helper functions | Code Cell 06, Code Cell 07 |
| 5. Replay baseline and cascade variants | Code Cell 08 |
| 6. Build replay metrics and compare to training-run artifacts | Code Cell 09 |
| 7. Save Gold 06A outputs | Code Cell 10 |
| 8. Gold 06A interpretation | Code Cell 11 |

## Code Cell 01 — 1. Imports and notebook context

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__future__`
- `annotations`
- `Any`
- `average_precision_score`
- `configure_logging`
- `confusion_matrix`
- `core`
- `file_io`
- `joblib`
- `json`
- `load_json`
- `load_notebook_context`
- `logging`
- `logging_setup`
- `Mapping`
- `math`
- `matplotlib`
- `metrics`
- `notebook_context`
- `numpy`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `from __future__ import annotations`: Imports a dependency or project helper used by later cells.
- `from pathlib import Path`: Imports a dependency or project helper used by later cells.
- `from typing import Any, Mapping, Sequence`: Imports a dependency or project helper used by later cells.
- `import json`: Imports a dependency or project helper used by later cells.
- `import logging`: Imports a dependency or project helper used by later cells.
- `import math`: Imports a dependency or project helper used by later cells.
- `import os`: Imports a dependency or project helper used by later cells.
- `import joblib`: Imports a dependency or project helper used by later cells.
- `import matplotlib.pyplot as plt`: Imports a dependency or project helper used by later cells.
- `import numpy as np`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `import yaml`: Imports a dependency or project helper used by later cells.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `from __future__ import annotations` | Imports a dependency or project helper used by later cells. |
| `from pathlib import Path` | Imports a dependency or project helper used by later cells. |
| `from typing import Any, Mapping, Sequence` | Imports a dependency or project helper used by later cells. |
| `import json` | Imports a dependency or project helper used by later cells. |
| `import logging` | Imports a dependency or project helper used by later cells. |
| `import math` | Imports a dependency or project helper used by later cells. |
| `import os` | Imports a dependency or project helper used by later cells. |
| `import joblib` | Imports a dependency or project helper used by later cells. |
| `import matplotlib.pyplot as plt` | Imports a dependency or project helper used by later cells. |
| `import numpy as np` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `import yaml` | Imports a dependency or project helper used by later cells. |
| `from sklearn.metrics import ( average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.file_io import load_json, save_json` | Imports a dependency or project helper used by later cells. |
| `from utils.core.notebook_context import load_notebook_context` | Imports a dependency or project helper used by later cells. |
| `from utils.core.logging_setup import configure_logging` | Imports a dependency or project helper used by later cells. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 02 — 1. Imports and notebook context

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__name__`
- `a`
- `also`
- `Any`
- `arbitrary`
- `be`
- `can`
- `checker`
- `concrete`
- `contract`
- `def`
- `dictionary`
- `explicit`
- `f`
- `gives`
- `got`
- `helper`
- `instead`
- `isinstance`
- `item`

### Outputs

- `require_mapping`
- `require_string_list`

### Key Operations

- `def require_string_list(value: Any, name: str) -> list[str]: """ Validate a loaded JSON value as a list of strings. This keeps the notebook runtime-safe and also gives Pylance a co`: Defines notebook-local logic used later in the notebook.
- `def require_mapping(value: Any, name: str) -> dict[str, Any]: """ Validate a loaded JSON/YAML value as a dictionary. JSON and YAML loaders can return None or arbitrary objects from`: Defines notebook-local logic used later in the notebook.

Important functions or methods detected:
- `isinstance`
- `load_json`
- `require_mapping`
- `require_string_list`
- `type`
- `TypeError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def require_string_list(value: Any, name: str) -> list[str]: """ Validate a loaded JSON value as a list of strings. This keeps the notebook runtime-safe and also gives Pylance a co` | Defines notebook-local logic used later in the notebook. |
| `def require_mapping(value: Any, name: str) -> dict[str, Any]: """ Validate a loaded JSON/YAML value as a dictionary. JSON and YAML loaders can return None or arbitrary objects from` | Defines notebook-local logic used later in the notebook. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 03 — 1. Imports and notebook context

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `artifacts`
- `capstone`
- `context`
- `context_loaded`
- `dataset_config`
- `default`
- `Gold`
- `gold`
- `gold_model_replay_validation`
- `info`
- `load_notebook_context`
- `loaded`
- `Loaded`
- `log`
- `LOG_PATH`
- `log_path`
- `logger`
- `logger_child_name`
- `message`

### Outputs

- `ARTIFACTS_ROOT`
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
- `DATASET_NAME`
- `extra`
- `FILENAMES`
- `kind`
- `ledger`
- `log_filename`

### Key Operations

- `# Shared notebook context.`: Documents the purpose or boundary of the surrounding notebook step.
- `CONTEXT_STAGE = "gold_model_replay_validation"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_DATASET = "pump"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LAYER = "gold"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_RUN_MODE = "test"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_PROFILE = "default"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LOG_FILE = "gold_model_replay_validation.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.gold.model_replay_validation", `: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `paths = CTX.paths`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG = CTX.config`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_MAP = CTX.config`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `info`
- `load_notebook_context`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Shared notebook context.` | Documents the purpose or boundary of the surrounding notebook step. |
| `CONTEXT_STAGE = "gold_model_replay_validation"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LAYER = "gold"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_RUN_MODE = "test"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LOG_FILE = "gold_model_replay_validation.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.gold.model_replay_validation", ` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `paths = CTX.paths` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_MAP = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE_CFG = CTX.stage_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RESOLVED_PATHS = CTX.resolved_paths` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FILENAMES = CTX.filenames` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUNTIME_CFG = CTX.runtime` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_CFG = CTX.dataset_config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PIPELINE = CTX.pipeline` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger = CTX.logger` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger = CTX.ledger` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LOG_PATH = CTX.log_path` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_RECIPE_ID = CTX.recipe_id` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_NAME = CONTEXT_DATASET` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ARTIFACTS_ROOT = paths.artifacts` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MODELS_ROOT = paths.models` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger.info( "Gold 06A replay validation context loaded", extra={ "stage": CONTEXT_STAGE, "dataset": CONTEXT_DATASET, "mode": CONFIG_RUN_MODE, "profile": CONFIG_PROFILE, "log_path"` | Writes a logger message for traceability during notebook execution. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="context_loaded", message="Loaded Gold 06A replay-validation context.", data={ "stage": CONTEXT_STAGE, "dataset": CONTEXT_DATASET, "mode": CONFIG_RUN_` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 04 — 2. Resolve source artifacts and output folders

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold__baseline_isolation_forest`
- `__gold__baseline_summary`
- `__gold__baseline_thresholds`
- `__gold__cascade_defaults_reference_profile`
- `__gold__cascade_defaults_stage1_isolation_forest`
- `__gold__cascade_defaults_stage2_isolation_forest`
- `__gold__cascade_defaults_summary`
- `__gold__cascade_defaults_thresholds`
- `__gold__cascade_stage3_improved_reference_profile`
- `__gold__cascade_stage3_improved_stage1_isolation_forest`
- `__gold__cascade_stage3_improved_stage2_isolation_forest`
- `__gold__cascade_stage3_improved_summary`
- `__gold__cascade_stage3_improved_thresholds`
- `__gold__cascade_tuned_reference_profile`
- `__gold__cascade_tuned_stage1_isolation_forest`
- `__gold__cascade_tuned_stage2_isolation_forest`
- `__gold__cascade_tuned_summary`
- `__gold__cascade_tuned_thresholds`
- `__gold__preprocessed_scaled`
- `__gold__stage1_features`

### Outputs

- `artifact_inventory`
- `CONFIG_ARTIFACTS`
- `data`
- `FEATURE_DIR`
- `FULL_SCALED_PATH`
- `GOLD_ROOT`
- `kind`
- `logger`
- `message`
- `missing_artifacts`
- `MODEL_ARTIFACTS`
- `PROFILE_ARTIFACTS`
- `STAGE1_FEATURES_PATH`
- `STAGE2_FEATURES_PATH`
- `STAGE3_PRIMARY_PATH`
- `STAGE3_SECONDARY_PATH`
- `step`
- `SUMMARY_ARTIFACTS`
- `TEST_DATA_PATH`
- `THRESHOLD_ARTIFACTS`

### Key Operations

- `GOLD_ROOT = ARTIFACTS_ROOT / "gold" / DATASET_NAME`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `VALIDATION_ROOT = GOLD_ROOT / "model_replay_validation"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `VALIDATION_RESULTS_DIR = VALIDATION_ROOT / "results"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `VALIDATION_SCORES_DIR = VALIDATION_ROOT / "scores"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `VALIDATION_SUMMARY_DIR = VALIDATION_ROOT / "summaries"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `VALIDATION_PLOTS_DIR = VALIDATION_ROOT / "plots"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `for directory in [ VALIDATION_ROOT, VALIDATION_RESULTS_DIR, VALIDATION_SCORES_DIR, VALIDATION_SUMMARY_DIR, VALIDATION_PLOTS_DIR,`: Controls validation, iteration, file handling, or error handling for this step.
- `]: directory.mkdir(parents=True, exist_ok=True)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Gold 01 split outputs. Prefer the full scaled dataframe because Stage 3 uses`: Documents the purpose or boundary of the surrounding notebook step.
- `# rolling/persistence/drift logic. The final metrics are still evaluated only on`: Documents the purpose or boundary of the surrounding notebook step.
- `# held-out test rows via meta__is_train_flag.`: Documents the purpose or boundary of the surrounding notebook step.
- `FULL_SCALED_PATH = Path(RESOLVED_PATHS.get( "gold_preprocessed_scaled_data_path", paths.data / "gold" / f"{DATASET_NAME}__gold__preprocessed_scaled.parquet",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `DataFrame`
- `display`
- `exists`
- `FileNotFoundError`
- `get`
- `items`
- `mkdir`
- `Path`
- `to_string`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `GOLD_ROOT = ARTIFACTS_ROOT / "gold" / DATASET_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `VALIDATION_ROOT = GOLD_ROOT / "model_replay_validation"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `VALIDATION_RESULTS_DIR = VALIDATION_ROOT / "results"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `VALIDATION_SCORES_DIR = VALIDATION_ROOT / "scores"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `VALIDATION_SUMMARY_DIR = VALIDATION_ROOT / "summaries"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `VALIDATION_PLOTS_DIR = VALIDATION_ROOT / "plots"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for directory in [ VALIDATION_ROOT, VALIDATION_RESULTS_DIR, VALIDATION_SCORES_DIR, VALIDATION_SUMMARY_DIR, VALIDATION_PLOTS_DIR,` | Controls validation, iteration, file handling, or error handling for this step. |
| `]: directory.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Gold 01 split outputs. Prefer the full scaled dataframe because Stage 3 uses` | Documents the purpose or boundary of the surrounding notebook step. |
| `# rolling/persistence/drift logic. The final metrics are still evaluated only on` | Documents the purpose or boundary of the surrounding notebook step. |
| `# held-out test rows via meta__is_train_flag.` | Documents the purpose or boundary of the surrounding notebook step. |
| `FULL_SCALED_PATH = Path(RESOLVED_PATHS.get( "gold_preprocessed_scaled_data_path", paths.data / "gold" / f"{DATASET_NAME}__gold__preprocessed_scaled.parquet",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `TEST_DATA_PATH = Path(RESOLVED_PATHS.get( "gold_test_data_path", paths.data / "gold" / f"{DATASET_NAME}__gold__test.parquet",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `))` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `FEATURE_DIR = GOLD_ROOT / "preprocessing" / "features"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE1_FEATURES_PATH = FEATURE_DIR / f"{DATASET_NAME}__gold__stage1_features.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE2_FEATURES_PATH = FEATURE_DIR / f"{DATASET_NAME}__gold__stage2_features.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE3_PRIMARY_PATH = FEATURE_DIR / f"{DATASET_NAME}__gold__stage3_primary_rule_sensors.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STAGE3_SECONDARY_PATH = FEATURE_DIR / f"{DATASET_NAME}__gold__stage3_secondary_rule_sensors.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MODEL_ARTIFACTS = { "baseline": GOLD_ROOT / "baseline" / "models" / f"{DATASET_NAME}__gold__baseline_isolation_forest.joblib", "cascade_default_stage1": GOLD_ROOT / "cascade_defaul` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `THRESHOLD_ARTIFACTS = { "baseline": GOLD_ROOT / "baseline" / "thresholds" / f"{DATASET_NAME}__gold__baseline_thresholds.json", "cascade_default": GOLD_ROOT / "cascade_defaults" / "` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SUMMARY_ARTIFACTS = { "baseline": GOLD_ROOT / "baseline" / "summaries" / f"{DATASET_NAME}__gold__baseline_summary.json", "cascade_default": GOLD_ROOT / "cascade_defaults" / "summar` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PROFILE_ARTIFACTS = { "cascade_default": GOLD_ROOT / "cascade_defaults" / "profiles" / f"{DATASET_NAME}__gold__cascade_defaults_reference_profile.csv", "cascade_tuned": GOLD_ROOT /` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CONFIG_ARTIFACTS = { "cascade_default": GOLD_ROOT / "cascade_defaults" / "config" / f"{DATASET_NAME}__gold_cascade_defaults__resolved_config.yaml", "cascade_tuned": GOLD_ROOT / "ca` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `artifact_inventory = pd.DataFrame( [ {"artifact_group": "model", "artifact_key": key, "path": str(path), "exists": path.exists()} for key, path in MODEL_ARTIFACTS.items() ] + [ {"a` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_artifacts = artifact_inventory.loc[~artifact_inventory["exists"]]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if len(missing_artifacts) > 0: raise FileNotFoundError( "Gold 06A cannot run because required artifacts are missing:\n" + missing_artifacts.to_string(index=False) )` | Controls validation, iteration, file handling, or error handling for this step. |
| `ledger.add( kind="check", step="required_artifacts_found", message="Confirmed required Gold 06A model/config/feature/profile artifacts are available.", data={"artifact_count": int(` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(artifact_inventory)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 05 — 3. Load test source and training artifacts

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `absent`
- `add`
- `anomaly_flag`
- `Any`
- `are`
- `artifacts`
- `astype`
- `because`
- `bool`
- `but`
- `can`
- `check`
- `Checked`
- `column`
- `columns`
- `config`
- `CONFIG_ARTIFACTS`
- `config_payloads`
- `configuration`
- `Could`

### Outputs

- `basic_shape_check`
- `data`
- `kind`
- `label_column_candidates`
- `logger`
- `message`
- `replay_source_dataframe`
- `replay_source_name`
- `resolved_label_column`
- `step`
- `test_labels`
- `test_mask`

### Key Operations

- `if FULL_SCALED_PATH.exists(): replay_source_dataframe = pd.read_parquet(FULL_SCALED_PATH) replay_source_name = "full_scaled_with_test_mask" if "meta__is_train_flag" not in replay_s`: Loads input data, configuration, or artifacts required by the current stage.
- `else: # Fallback for portability. This can run the replay, but rolling Stage 3 rules # may not match the original notebook exactly because training rows are absent. replay_source_d`: Loads input data, configuration, or artifacts required by the current stage.
- `stage1_feature_columns: list[str] = require_string_list( load_json(STAGE1_FEATURES_PATH), "stage1_feature_columns",`: Loads input data, configuration, or artifacts required by the current stage.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage2_feature_columns: list[str] = require_string_list( load_json(STAGE2_FEATURES_PATH), "stage2_feature_columns",`: Loads input data, configuration, or artifacts required by the current stage.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage3_primary_rule_sensors: list[str] = require_string_list( load_json(STAGE3_PRIMARY_PATH), "stage3_primary_rule_sensors",`: Loads input data, configuration, or artifacts required by the current stage.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage3_secondary_rule_sensors: list[str] = require_string_list( load_json(STAGE3_SECONDARY_PATH), "stage3_secondary_rule_sensors",`: Loads input data, configuration, or artifacts required by the current stage.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `models: dict[str, Any] = { key: joblib.load(path) for key, path in MODEL_ARTIFACTS.items()`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `astype`
- `exists`
- `fillna`
- `items`
- `KeyError`
- `load`
- `load_json`
- `next`
- `open`
- `read_csv`
- `read_parquet`
- `require_mapping`
- `require_string_list`
- `safe_load`
- `Series`
- `sum`
- `to_numpy`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `if FULL_SCALED_PATH.exists(): replay_source_dataframe = pd.read_parquet(FULL_SCALED_PATH) replay_source_name = "full_scaled_with_test_mask" if "meta__is_train_flag" not in replay_s` | Loads input data, configuration, or artifacts required by the current stage. |
| `else: # Fallback for portability. This can run the replay, but rolling Stage 3 rules # may not match the original notebook exactly because training rows are absent. replay_source_d` | Loads input data, configuration, or artifacts required by the current stage. |
| `stage1_feature_columns: list[str] = require_string_list( load_json(STAGE1_FEATURES_PATH), "stage1_feature_columns",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage2_feature_columns: list[str] = require_string_list( load_json(STAGE2_FEATURES_PATH), "stage2_feature_columns",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage3_primary_rule_sensors: list[str] = require_string_list( load_json(STAGE3_PRIMARY_PATH), "stage3_primary_rule_sensors",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage3_secondary_rule_sensors: list[str] = require_string_list( load_json(STAGE3_SECONDARY_PATH), "stage3_secondary_rule_sensors",` | Loads input data, configuration, or artifacts required by the current stage. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `models: dict[str, Any] = { key: joblib.load(path) for key, path in MODEL_ARTIFACTS.items()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `thresholds: dict[str, dict[str, Any]] = { key: require_mapping(load_json(path), f"{key} thresholds") for key, path in THRESHOLD_ARTIFACTS.items()` | Loads input data, configuration, or artifacts required by the current stage. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `summaries: dict[str, dict[str, Any]] = { key: require_mapping(load_json(path), f"{key} summary") for key, path in SUMMARY_ARTIFACTS.items()` | Loads input data, configuration, or artifacts required by the current stage. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `profiles: dict[str, pd.DataFrame] = { key: pd.read_csv(path) for key, path in PROFILE_ARTIFACTS.items()` | Loads input data, configuration, or artifacts required by the current stage. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `config_payloads: dict[str, dict[str, Any]] = {}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for key, path in CONFIG_ARTIFACTS.items(): with path.open("r", encoding="utf-8") as file: config_payloads[key] = require_mapping( yaml.safe_load(file), f"{key} config", )` | Controls validation, iteration, file handling, or error handling for this step. |
| `label_column_candidates = ["anomaly_flag", "is_anomaly", "target_flag", "label"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `resolved_label_column = next( (column for column in label_column_candidates if column in replay_source_dataframe.columns), None,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if resolved_label_column is None: raise KeyError(f"Could not find label column. Checked: {label_column_candidates}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `label_column: str = resolved_label_column` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `test_labels = replay_source_dataframe.loc[test_mask, label_column].fillna(0).astype(int).to_numpy(dtype=int)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `basic_shape_check = { "replay_source_name": replay_source_name, "row_count": int(len(replay_source_dataframe)), "test_row_count": int(test_mask.sum()), "label_column": label_column` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="check", step="replay_source_loaded", message="Loaded Gold replay source dataframe and model configuration artifacts.", data=basic_shape_check, logger=logger,` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `basic_shape_check` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.

## Code Cell 06 — 4. Replay helper functions

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `a`
- `abs`
- `across`
- `alert`
- `alert_count_test_rows`
- `already`
- `also`
- `an`
- `anomalous`
- `Any`
- `any`
- `are`
- `astype`
- `average_precision_score`
- `away`
- `being`
- `binary`
- `bounds`
- `by`

### Outputs

- `anomaly_score`
- `average`
- `breach_counts`
- `compute_binary_metrics`
- `compute_drift_flag`
- `compute_persistence_flag`
- `compute_profile_breach_count`
- `decision`
- `drift_trigger_counts`
- `ensure_columns`
- `feature_breach_flag`
- `feature_drift_flag`
- `feature_frame`
- `feature_series`
- `feature_standard_deviation`
- `lower_bound`
- `missing_columns`
- `prediction`
- `reference_lookup`
- `rolling_count`

### Key Operations

- `def ensure_columns( dataframe: pd.DataFrame, columns: Sequence[str], *, context: str,`: Defines notebook-local logic used later in the notebook.
- `) -> None: """ Confirm that a dataframe contains the requested columns. Parameters ---------- dataframe: DataFrame being checked. columns: Column names required by the model or rul`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def score_isolation_forest( model: Any, dataframe: pd.DataFrame, feature_columns: Sequence[str],`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, np.ndarray]: """ Score a dataframe with an already-fitted Isolation Forest. The project uses `-score_samples` so larger values mean more anomalous. The helper also r`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def compute_binary_metrics( *, dataframe: pd.DataFrame, test_mask: pd.Series, label_column: str, flag_column: str, score_column: str \| None = None,`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, Any]: """ Compute test-row alert counts and standard binary classification metrics. """ test_dataframe = dataframe.loc[test_mask].copy() y_true = test_dataframe[labe`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def compute_profile_breach_count( dataframe: pd.DataFrame, *, reference_profile: pd.DataFrame, feature_columns: Sequence[str], output_name: str,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.Series: """ Count how many selected sensor features are outside their reference bounds. """ reference_lookup = reference_profile.set_index("feature_name").to_dict("index") `: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def compute_persistence_flag( flag_series: pd.Series, *, rolling_window_size: int, minimum_flags_in_window: int,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.Series: """ Mark rows where recent Stage 2 flags persist across a rolling window. """ rolling_count = ( flag_series.fillna(0) .astype(int) .rolling(window=rolling_window_si`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def compute_drift_flag( dataframe: pd.DataFrame, *, feature_columns: Sequence[str], rolling_window_size: int, drift_threshold_multiplier: float,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.Series: """ Mark rows where any watched feature drifts away from its rolling median. """ drift_trigger_counts = pd.Series(0, index=dataframe.index, dtype=int) for feature_n`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `abs`
- `astype`
- `average_precision_score`
- `compute_binary_metrics`
- `compute_drift_flag`
- `compute_persistence_flag`
- `compute_profile_breach_count`
- `confusion_matrix`
- `copy`
- `decision_function`
- `ensure_columns`
- `fillna`
- `isna`
- `KeyError`
- `median`
- `precision_recall_fscore_support`
- `predict`
- `ravel`
- `return`
- `roc_auc_score`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def ensure_columns( dataframe: pd.DataFrame, columns: Sequence[str], *, context: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> None: """ Confirm that a dataframe contains the requested columns. Parameters ---------- dataframe: DataFrame being checked. columns: Column names required by the model or rul` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def score_isolation_forest( model: Any, dataframe: pd.DataFrame, feature_columns: Sequence[str],` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, np.ndarray]: """ Score a dataframe with an already-fitted Isolation Forest. The project uses `-score_samples` so larger values mean more anomalous. The helper also r` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def compute_binary_metrics( *, dataframe: pd.DataFrame, test_mask: pd.Series, label_column: str, flag_column: str, score_column: str \| None = None,` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, Any]: """ Compute test-row alert counts and standard binary classification metrics. """ test_dataframe = dataframe.loc[test_mask].copy() y_true = test_dataframe[labe` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def compute_profile_breach_count( dataframe: pd.DataFrame, *, reference_profile: pd.DataFrame, feature_columns: Sequence[str], output_name: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.Series: """ Count how many selected sensor features are outside their reference bounds. """ reference_lookup = reference_profile.set_index("feature_name").to_dict("index") ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def compute_persistence_flag( flag_series: pd.Series, *, rolling_window_size: int, minimum_flags_in_window: int,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.Series: """ Mark rows where recent Stage 2 flags persist across a rolling window. """ rolling_count = ( flag_series.fillna(0) .astype(int) .rolling(window=rolling_window_si` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def compute_drift_flag( dataframe: pd.DataFrame, *, feature_columns: Sequence[str], rolling_window_size: int, drift_threshold_multiplier: float,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.Series: """ Mark rows where any watched feature drifts away from its rolling median. """ drift_trigger_counts = pd.Series(0, index=dataframe.index, dtype=int) for feature_n` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 07 — 4. Replay helper functions

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Any`
- `Apply`
- `astype`
- `broad`
- `by`
- `cascade_final_flag`
- `cascade_stage3_improved_flag`
- `cascade_stage3_medium_flag`
- `cascade_stage3_relaxed_flag`
- `cascade_stage3_strict_flag`
- `compute_drift_flag`
- `compute_persistence_flag`
- `compute_profile_breach_count`
- `confirmation`
- `copy`
- `DataFrame`
- `dataframe`
- `def`
- `fromkeys`
- `get`

### Outputs

- `add_stage3_broad_rules`
- `add_stage3_improved_rules`
- `corroboration_weight`
- `drift_rolling_window_size`
- `drift_threshold_multiplier`
- `drift_weight`
- `feature_columns`
- `min_primary_hits`
- `min_secondary_hits`
- `min_weighted_score`
- `minimum_flags_in_window`
- `output_name`
- `persistence_weight`
- `profile_weight`
- `reference_profile`
- `result`
- `rolling_window_size`
- `selected_params`
- `strong_primary_hits`
- `watch_features`

### Key Operations

- `def add_stage3_broad_rules( dataframe: pd.DataFrame, *, reference_profile: pd.DataFrame, stage3_config: Mapping[str, Any],`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """ Apply the broad Stage 3 confirmation rule used by Gold 03A and Gold 03B. """ result = dataframe.copy() min_primary_hits = int(stage3_config["min_primary_sens`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def add_stage3_improved_rules( dataframe: pd.DataFrame, *, reference_profile: pd.DataFrame, thresholds_payload: Mapping[str, Any], stage3_config: Mapping[str, Any],`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """ Apply the tuned Stage 3 confirmation rules used by Gold 03C. """ result = dataframe.copy() selected_params = dict(thresholds_payload["stage3_selected_params"`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add_stage3_broad_rules`
- `add_stage3_improved_rules`
- `astype`
- `compute_drift_flag`
- `compute_persistence_flag`
- `compute_profile_breach_count`
- `copy`
- `fromkeys`
- `get`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def add_stage3_broad_rules( dataframe: pd.DataFrame, *, reference_profile: pd.DataFrame, stage3_config: Mapping[str, Any],` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Apply the broad Stage 3 confirmation rule used by Gold 03A and Gold 03B. """ result = dataframe.copy() min_primary_hits = int(stage3_config["min_primary_sens` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def add_stage3_improved_rules( dataframe: pd.DataFrame, *, reference_profile: pd.DataFrame, thresholds_payload: Mapping[str, Any], stage3_config: Mapping[str, Any],` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Apply the tuned Stage 3 confirmation rules used by Gold 03C. """ result = dataframe.copy() selected_params = dict(thresholds_payload["stage3_selected_params"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 08 — 5. Replay baseline and cascade variants

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `add_stage3_broad_rules`
- `add_stage3_improved_rules`
- `against`
- `Any`
- `artifacts`
- `astype`
- `baseline`
- `baseline_decision`
- `baseline_flag`
- `baseline_pred`
- `baseline_rows`
- `baseline_score`
- `baseline_threshold`
- `bool`
- `cascade`
- `cascade_default`
- `cascade_default_rows`
- `cascade_default_stage1`
- `cascade_default_stage2`

### Outputs

- `baseline_replay`
- `cascade_default_replay`
- `cascade_tuned_replay`
- `data`
- `gold_cascade_config`
- `kind`
- `logger`
- `message`
- `reference_profile`
- `result`
- `run_baseline_replay`
- `run_cascade_replay`
- `score_payload`
- `stage1_model_key`
- `stage1_scores`
- `stage1_threshold`
- `stage2_model_key`
- `stage2_scores`
- `stage2_threshold`
- `stage3_config`

### Key Operations

- `def run_baseline_replay(source_dataframe: pd.DataFrame) -> pd.DataFrame: """Replay the Gold 02 baseline Isolation Forest using the saved threshold.""" result = source_dataframe.cop`: Defines notebook-local logic used later in the notebook.
- `def run_cascade_replay( source_dataframe: pd.DataFrame, *, variant_key: str, stage1_model_key: str, stage2_model_key: str, use_improved_stage3: bool,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """Replay one saved cascade variant against the replay source dataframe.""" result = source_dataframe.copy() threshold_payload: dict[str, Any] = thresholds[varia`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `baseline_replay = run_baseline_replay(replay_source_dataframe)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `cascade_default_replay = run_cascade_replay( replay_source_dataframe, variant_key="cascade_default", stage1_model_key="cascade_default_stage1", stage2_model_key="cascade_default_st`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `cascade_tuned_replay = run_cascade_replay( replay_source_dataframe, variant_key="cascade_tuned", stage1_model_key="cascade_tuned_stage1", stage2_model_key="cascade_tuned_stage2", u`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `stage3_improved_replay = run_cascade_replay( replay_source_dataframe, variant_key="stage3_improved", stage1_model_key="stage3_improved_stage1", stage2_model_key="stage3_improved_st`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `ledger.add( kind="step", step="model_variants_replayed", message="Replayed saved baseline and cascade model artifacts against the Gold test split source.", data={ "baseline_rows": `: Records or exports ledger information for stage-level traceability.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `add_stage3_broad_rules`
- `add_stage3_improved_rules`
- `astype`
- `copy`
- `get`
- `require_mapping`
- `run_baseline_replay`
- `run_cascade_replay`
- `score_isolation_forest`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def run_baseline_replay(source_dataframe: pd.DataFrame) -> pd.DataFrame: """Replay the Gold 02 baseline Isolation Forest using the saved threshold.""" result = source_dataframe.cop` | Defines notebook-local logic used later in the notebook. |
| `def run_cascade_replay( source_dataframe: pd.DataFrame, *, variant_key: str, stage1_model_key: str, stage2_model_key: str, use_improved_stage3: bool,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """Replay one saved cascade variant against the replay source dataframe.""" result = source_dataframe.copy() threshold_payload: dict[str, Any] = thresholds[varia` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `baseline_replay = run_baseline_replay(replay_source_dataframe)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `cascade_default_replay = run_cascade_replay( replay_source_dataframe, variant_key="cascade_default", stage1_model_key="cascade_default_stage1", stage2_model_key="cascade_default_st` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cascade_tuned_replay = run_cascade_replay( replay_source_dataframe, variant_key="cascade_tuned", stage1_model_key="cascade_tuned_stage1", stage2_model_key="cascade_tuned_stage2", u` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `stage3_improved_replay = run_cascade_replay( replay_source_dataframe, variant_key="stage3_improved", stage1_model_key="stage3_improved_stage1", stage2_model_key="stage3_improved_st` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="step", step="model_variants_replayed", message="Replayed saved baseline and cascade model artifacts against the Gold test split source.", data={ "baseline_rows": ` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Replay complete.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 09 — 6. Build replay metrics and compare to training-run artifacts

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_alert_count_test_rows`
- `_delta`
- `_f1`
- `_precision`
- `_recall`
- `a`
- `above`
- `abs`
- `alert`
- `alert_count_match`
- `alert_count_test_rows`
- `alert_count_test_rows_delta`
- `alert_count_within_tolerance`
- `all`
- `Any`
- `append`
- `artifacts`
- `axis`
- `baseline`
- `Baseline`

### Outputs

- `actual_column`
- `ALERT_COUNT_TOLERANCE`
- `baseline_metrics`
- `build_expected_metrics_from_training_artifacts`
- `cascade_metrics`
- `dataframe`
- `delta_column`
- `expected_column`
- `expected_metrics_dataframe`
- `flag_column`
- `how`
- `label_column`
- `METRIC_TOLERANCE`
- `METRIC_TOLERANCE_RELAXED`
- `metrics`
- `on`
- `replay_comparison_dataframe`
- `replay_metric_rows`
- `replay_metric_specs`
- `replay_metrics_dataframe`

### Key Operations

- `replay_metric_specs = [ { "model_id": "baseline", "model_label": "Baseline IsolationForest", "source_notebook": "gold_02", "dataframe": baseline_replay, "flag_column": "baseline_fl`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `replay_metric_rows = []`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `for spec in replay_metric_specs: metrics = compute_binary_metrics( dataframe=spec["dataframe"], test_mask=test_mask, label_column=label_column, flag_column=spec["flag_column"], sco`: Controls validation, iteration, file handling, or error handling for this step.
- `replay_metrics_dataframe = pd.DataFrame(replay_metric_rows)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `def build_expected_metrics_from_training_artifacts() -> pd.DataFrame: """Collect the saved test-row metrics from the Gold 02/03 training artifacts.""" rows: list[dict[str, Any]] = `: Defines notebook-local logic used later in the notebook.
- `expected_metrics_dataframe = build_expected_metrics_from_training_artifacts()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `replay_comparison_dataframe = replay_metrics_dataframe.merge( expected_metrics_dataframe, on="model_id", how="left",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `for metric_name in ["alert_count_test_rows", "precision", "recall", "f1"]: actual_column = metric_name expected_column = f"expected_{metric_name}" delta_column = f"{metric_name}_de`: Controls validation, iteration, file handling, or error handling for this step.
- `METRIC_TOLERANCE = 1e-9`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `replay_comparison_dataframe["alert_count_match"] = ( replay_comparison_dataframe["alert_count_test_rows_delta"].fillna(np.inf).abs() == 0`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `abs`
- `all`
- `append`
- `build_expected_metrics_from_training_artifacts`
- `compute_binary_metrics`
- `DataFrame`
- `display`
- `eq`
- `fillna`
- `merge`
- `where`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `replay_metric_specs = [ { "model_id": "baseline", "model_label": "Baseline IsolationForest", "source_notebook": "gold_02", "dataframe": baseline_replay, "flag_column": "baseline_fl` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `replay_metric_rows = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for spec in replay_metric_specs: metrics = compute_binary_metrics( dataframe=spec["dataframe"], test_mask=test_mask, label_column=label_column, flag_column=spec["flag_column"], sco` | Controls validation, iteration, file handling, or error handling for this step. |
| `replay_metrics_dataframe = pd.DataFrame(replay_metric_rows)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `def build_expected_metrics_from_training_artifacts() -> pd.DataFrame: """Collect the saved test-row metrics from the Gold 02/03 training artifacts.""" rows: list[dict[str, Any]] = ` | Defines notebook-local logic used later in the notebook. |
| `expected_metrics_dataframe = build_expected_metrics_from_training_artifacts()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `replay_comparison_dataframe = replay_metrics_dataframe.merge( expected_metrics_dataframe, on="model_id", how="left",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for metric_name in ["alert_count_test_rows", "precision", "recall", "f1"]: actual_column = metric_name expected_column = f"expected_{metric_name}" delta_column = f"{metric_name}_de` | Controls validation, iteration, file handling, or error handling for this step. |
| `METRIC_TOLERANCE = 1e-9` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `replay_comparison_dataframe["alert_count_match"] = ( replay_comparison_dataframe["alert_count_test_rows_delta"].fillna(np.inf).abs() == 0` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `replay_comparison_dataframe["precision_match"] = ( replay_comparison_dataframe["precision_delta"].fillna(np.inf).abs() <= METRIC_TOLERANCE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `replay_comparison_dataframe["recall_match"] = ( replay_comparison_dataframe["recall_delta"].fillna(np.inf).abs() <= METRIC_TOLERANCE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `replay_comparison_dataframe["f1_match"] = ( replay_comparison_dataframe["f1_delta"].fillna(np.inf).abs() <= METRIC_TOLERANCE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `replay_comparison_dataframe["validation_status"] = np.where( replay_comparison_dataframe[["alert_count_match", "precision_match", "recall_match", "f1_match"]].all(axis=1), "pass", ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Secondary tolerance check for near-exact replay matches` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------------------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The exact validation_status above remains the strict check.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# If every metric matches exactly, validation_status stays "pass".` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# If a row does not pass exactly, this second check determines whether the row` | Documents the purpose or boundary of the surrounding notebook step. |
| `# is still close enough to be considered a practical replay match.` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Current tolerance rule:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - alert count may differ by no more than 1 row;` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - precision, recall, and F1 may differ by no more than 0.0001.` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# This preserves both interpretations:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - validation_status = strict exact-match result` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - tolerance_validation_status = secondary within-tolerance result` | Documents the purpose or boundary of the surrounding notebook step. |
| `# - final_validation_status = reportable final result` | Documents the purpose or boundary of the surrounding notebook step. |
| `ALERT_COUNT_TOLERANCE = 1` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `METRIC_TOLERANCE_RELAXED = 0.0001` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `replay_comparison_dataframe["alert_count_within_tolerance"] = ( replay_comparison_dataframe["alert_count_test_rows_delta"] .fillna(np.inf) .abs() <= ALERT_COUNT_TOLERANCE` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `replay_comparison_dataframe["precision_within_tolerance"] = ( replay_comparison_dataframe["precision_delta"] .fillna(np.inf) .abs() <= METRIC_TOLERANCE_RELAXED` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `replay_comparison_dataframe["recall_within_tolerance"] = ( replay_comparison_dataframe["recall_delta"] .fillna(np.inf) .abs() <= METRIC_TOLERANCE_RELAXED` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `replay_comparison_dataframe["f1_within_tolerance"] = ( replay_comparison_dataframe["f1_delta"] .fillna(np.inf) .abs() <= METRIC_TOLERANCE_RELAXED` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `replay_comparison_dataframe["tolerance_validation_status"] = np.where( replay_comparison_dataframe["validation_status"].eq("pass"), "not_needed_exact_pass", np.where( replay_compar` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `replay_comparison_dataframe["final_validation_status"] = np.where( replay_comparison_dataframe["validation_status"].eq("pass"), "exact_pass", replay_comparison_dataframe["tolerance` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(replay_comparison_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 10 — 7. Save Gold 06A outputs

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__`
- `__gold06a__test_replay_metrics`
- `__gold06a__test_replay_scores`
- `__gold06a__test_replay_summary`
- `__gold06a__test_replay_vs_training_artifacts`
- `add`
- `against`
- `alert_count_tolerance`
- `ALERT_COUNT_TOLERANCE`
- `all`
- `arange`
- `artifacts`
- `baseline`
- `baseline_flag`
- `baseline_replay`
- `baseline_score`
- `Build`
- `by`
- `cascade_default`
- `cascade_default_replay`

### Outputs

- `comparison_output_path`
- `data`
- `identity_columns`
- `kind`
- `logger`
- `message`
- `metrics_output_path`
- `replay_scores_dataframe`
- `score_sources`
- `scores_output_path`
- `step`
- `summary_output_path`
- `summary_payload`

### Key Operations

- `# Build one wide test-row dataframe with final replay outputs from all model variants.`: Documents the purpose or boundary of the surrounding notebook step.
- `identity_columns = [ column for column in [ "meta__row_id", "meta__asset_id", "meta__dataset", "meta__run_id", "meta__split", "meta__is_train_flag", "event_time", "event_step", "ti`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `replay_scores_dataframe = replay_source_dataframe.loc[test_mask, identity_columns].copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `replay_scores_dataframe = replay_scores_dataframe.reset_index(drop=True)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `replay_scores_dataframe["plot_order_index"] = np.arange(len(replay_scores_dataframe), dtype=int)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `score_sources = { "baseline": (baseline_replay, ["baseline_score", "baseline_flag"]), "cascade_default": (cascade_default_replay, ["stage1_score", "stage1_flag", "stage2_score", "s`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `for prefix, (source_df, columns) in score_sources.items(): for column in columns: if column in source_df.columns: replay_scores_dataframe[f"{prefix}__{column}"] = ( source_df.loc[t`: Controls validation, iteration, file handling, or error handling for this step.
- `metrics_output_path = VALIDATION_RESULTS_DIR / f"{DATASET_NAME}__gold06a__test_replay_metrics.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `comparison_output_path = VALIDATION_RESULTS_DIR / f"{DATASET_NAME}__gold06a__test_replay_vs_training_artifacts.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `scores_output_path = VALIDATION_SCORES_DIR / f"{DATASET_NAME}__gold06a__test_replay_scores.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `arange`
- `copy`
- `eq`
- `isin`
- `items`
- `reset_index`
- `save_json`
- `sum`
- `to_csv`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Build one wide test-row dataframe with final replay outputs from all model variants.` | Documents the purpose or boundary of the surrounding notebook step. |
| `identity_columns = [ column for column in [ "meta__row_id", "meta__asset_id", "meta__dataset", "meta__run_id", "meta__split", "meta__is_train_flag", "event_time", "event_step", "ti` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `replay_scores_dataframe = replay_source_dataframe.loc[test_mask, identity_columns].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `replay_scores_dataframe = replay_scores_dataframe.reset_index(drop=True)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `replay_scores_dataframe["plot_order_index"] = np.arange(len(replay_scores_dataframe), dtype=int)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `score_sources = { "baseline": (baseline_replay, ["baseline_score", "baseline_flag"]), "cascade_default": (cascade_default_replay, ["stage1_score", "stage1_flag", "stage2_score", "s` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for prefix, (source_df, columns) in score_sources.items(): for column in columns: if column in source_df.columns: replay_scores_dataframe[f"{prefix}__{column}"] = ( source_df.loc[t` | Controls validation, iteration, file handling, or error handling for this step. |
| `metrics_output_path = VALIDATION_RESULTS_DIR / f"{DATASET_NAME}__gold06a__test_replay_metrics.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `comparison_output_path = VALIDATION_RESULTS_DIR / f"{DATASET_NAME}__gold06a__test_replay_vs_training_artifacts.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `scores_output_path = VALIDATION_SCORES_DIR / f"{DATASET_NAME}__gold06a__test_replay_scores.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `summary_output_path = VALIDATION_SUMMARY_DIR / f"{DATASET_NAME}__gold06a__test_replay_summary.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `replay_metrics_dataframe.to_csv(metrics_output_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `replay_comparison_dataframe.to_csv(comparison_output_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `replay_scores_dataframe.to_csv(scores_output_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `summary_payload = { "stage": CONTEXT_STAGE, "dataset": DATASET_NAME, "recipe_id": CONTEXT_RECIPE_ID, "replay_source_name": replay_source_name, "replay_source_rows": int(len(replay_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `save_json(summary_payload, summary_output_path)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="result", step="gold06a_outputs_saved", message="Saved Gold 06A replay metrics, replay-vs-training comparison, row-level test replay scores, and summary JSON.", da` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `metrics_output_path, comparison_output_path, scores_output_path, summary_output_path` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 11 — 8. Gold 06A interpretation

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `against`
- `alert`
- `are`
- `artifacts`
- `Baseline`
- `be`
- `benchmark`
- `can`
- `cascade`
- `check`
- `compared`
- `comparison_scope`
- `count`
- `default`
- `delta`
- `deltas`
- `early`
- `eq`
- `evaluated`

### Outputs

- `interpretation`

### Key Operations

- `interpretation = { "purpose": ( "Gold 06A validates that saved Gold 02/03 model artifacts and rule settings can be replayed " "against held-out test rows without retraining." ), "v`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(interpretation)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `eq`
- `isin`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `interpretation = { "purpose": ( "Gold 06A validates that saved Gold 02/03 model artifacts and rule settings can be replayed " "against held-out test rows without retraining." ), "v` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(interpretation)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

