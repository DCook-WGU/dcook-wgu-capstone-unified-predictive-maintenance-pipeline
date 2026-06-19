# Notebook Code Reference: EDA_Notebook_Pump_Gold_06B_Test_Early_Warning_Validation

Notebook path:

`notebooks/experiments/EDA_Notebook_Pump_Gold_06B_Test_Early_Warning_Validation.ipynb`

## Notebook Purpose

This notebook evaluates held-out early-warning behavior using replay outputs and Gold anomaly timeline references.

Notebook stage:

`Gold`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| 1. Imports and context | Code Cell 01, Code Cell 02 |
| 2. Load Gold 06A replay output | Code Cell 03 |
| 3. Early-warning helper functions | Code Cell 04 |
| 4. Build early-warning comparison across replayed model variants | Code Cell 05 |
| 5. Compare test early-warning results to the Gold 05 training-run summary | Code Cell 06 |
| 6. Save outputs and plots | Code Cell 07 |
| 7. Gold 06B interpretation | Code Cell 08 |

## Code Cell 01 — 1. Imports and context

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__future__`
- `annotations`
- `Any`
- `core`
- `file_io`
- `json`
- `load_json`
- `load_notebook_context`
- `logging`
- `Mapping`
- `matplotlib`
- `notebook_context`
- `numpy`
- `os`
- `pandas`
- `pathlib`
- `pyplot`
- `save_json`
- `Sequence`
- `typing`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `from __future__ import annotations`: Imports a dependency or project helper used by later cells.
- `from pathlib import Path`: Imports a dependency or project helper used by later cells.
- `from typing import Any, Mapping, Sequence`: Imports a dependency or project helper used by later cells.
- `import json`: Imports a dependency or project helper used by later cells.
- `import logging`: Imports a dependency or project helper used by later cells.
- `import os`: Imports a dependency or project helper used by later cells.
- `import matplotlib.pyplot as plt`: Imports a dependency or project helper used by later cells.
- `import numpy as np`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `from utils.core.file_io import load_json, save_json`: Imports a dependency or project helper used by later cells.
- `from utils.core.notebook_context import load_notebook_context`: Imports a dependency or project helper used by later cells.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `from __future__ import annotations` | Imports a dependency or project helper used by later cells. |
| `from pathlib import Path` | Imports a dependency or project helper used by later cells. |
| `from typing import Any, Mapping, Sequence` | Imports a dependency or project helper used by later cells. |
| `import json` | Imports a dependency or project helper used by later cells. |
| `import logging` | Imports a dependency or project helper used by later cells. |
| `import os` | Imports a dependency or project helper used by later cells. |
| `import matplotlib.pyplot as plt` | Imports a dependency or project helper used by later cells. |
| `import numpy as np` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `from utils.core.file_io import load_json, save_json` | Imports a dependency or project helper used by later cells. |
| `from utils.core.notebook_context import load_notebook_context` | Imports a dependency or project helper used by later cells. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 02 — 1. Imports and context

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `add`
- `artifacts`
- `capstone`
- `context`
- `context_loaded`
- `default`
- `directory`
- `early`
- `exist_ok`
- `gold`
- `Gold`
- `gold_test_early_warning_validation`
- `load_notebook_context`
- `Loaded`
- `log`
- `mkdir`
- `mode`
- `model_replay_validation`
- `parents`
- `paths`

### Outputs

- `CONFIG`
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
- `DATASET_NAME`
- `GOLD_ROOT`
- `kind`
- `ledger`
- `log_filename`
- `LOG_PATH`
- `logger`
- `logger_child_name`
- `message`

### Key Operations

- `CONTEXT_STAGE = "gold_test_early_warning_validation"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_DATASET = "pump"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LAYER = "gold"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_RUN_MODE = "test"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_PROFILE = "default"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONTEXT_LOG_FILE = "gold_test_early_warning_validation.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.gold.test_early_warning_validat`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `paths = CTX.paths`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG = CTX.config`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `RESOLVED_PATHS = CTX.resolved_paths`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `logger = CTX.logger`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `load_notebook_context`
- `mkdir`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `CONTEXT_STAGE = "gold_test_early_warning_validation"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LAYER = "gold"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_RUN_MODE = "test"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_LOG_FILE = "gold_test_early_warning_validation.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CTX = load_notebook_context( stage=CONTEXT_STAGE, dataset=CONTEXT_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, logger_child_name="capstone.gold.test_early_warning_validat` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `paths = CTX.paths` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG = CTX.config` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RESOLVED_PATHS = CTX.resolved_paths` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `logger = CTX.logger` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ledger = CTX.ledger` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LOG_PATH = CTX.log_path` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONTEXT_RECIPE_ID = CTX.recipe_id` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_NAME = CONTEXT_DATASET` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `GOLD_ROOT = paths.artifacts / "gold" / DATASET_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `VALIDATION_ROOT = GOLD_ROOT / "model_replay_validation"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `VALIDATION_SCORES_DIR = VALIDATION_ROOT / "scores"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `VALIDATION_RESULTS_DIR = VALIDATION_ROOT / "results"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `VALIDATION_SUMMARY_DIR = VALIDATION_ROOT / "summaries"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `VALIDATION_PLOTS_DIR = VALIDATION_ROOT / "plots"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for directory in [VALIDATION_RESULTS_DIR, VALIDATION_SUMMARY_DIR, VALIDATION_PLOTS_DIR]: directory.mkdir(parents=True, exist_ok=True)` | Controls validation, iteration, file handling, or error handling for this step. |
| `ledger.add( kind="step", step="context_loaded", message="Loaded Gold 06B test early-warning validation context.", data={"stage": CONTEXT_STAGE, "dataset": DATASET_NAME, "log_path":` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 03 — 2. Load Gold 06A replay output

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold06a__test_replay_scores`
- `add`
- `anomaly_detection`
- `anomaly_flag`
- `anomaly_rows`
- `arange`
- `astype`
- `bool`
- `check`
- `Checked`
- `column`
- `columns`
- `copy`
- `Could`
- `csv`
- `DATASET_NAME`
- `dtype`
- `early`
- `exists`
- `f`

### Outputs

- `basic_input_check`
- `data`
- `kind`
- `label_column_candidates`
- `logger`
- `message`
- `replay_scores_dataframe`
- `REPLAY_SCORES_PATH`
- `resolved_label_column`
- `step`
- `TRAIN_LEAD_TIME_PATH`

### Key Operations

- `REPLAY_SCORES_PATH = VALIDATION_SCORES_DIR / f"{DATASET_NAME}__gold06a__test_replay_scores.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRAIN_LEAD_TIME_PATH = GOLD_ROOT / "anomaly_detection" / "summaries" / "multi_run_lead_time_comparison.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if not REPLAY_SCORES_PATH.exists(): raise FileNotFoundError( f"Gold 06B needs the Gold 06A replay score output first: {REPLAY_SCORES_PATH}" )`: Controls validation, iteration, file handling, or error handling for this step.
- `replay_scores_dataframe = pd.read_csv(REPLAY_SCORES_PATH)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `label_column_candidates = ["anomaly_flag", "is_anomaly", "target_flag", "label"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `resolved_label_column = next( (column for column in label_column_candidates if column in replay_scores_dataframe.columns), None,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if resolved_label_column is None: raise KeyError(f"Could not find label column. Checked: {label_column_candidates}")`: Controls validation, iteration, file handling, or error handling for this step.
- `label_column: str = resolved_label_column`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if "plot_order_index" not in replay_scores_dataframe.columns: replay_scores_dataframe = replay_scores_dataframe.copy() replay_scores_dataframe["plot_order_index"] = np.arange(len(r`: Controls validation, iteration, file handling, or error handling for this step.
- `basic_input_check = { "replay_scores_path": str(REPLAY_SCORES_PATH), "row_count": int(len(replay_scores_dataframe)), "label_column": label_column, "anomaly_rows": int(replay_scores`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `add`
- `arange`
- `astype`
- `bool`
- `copy`
- `exists`
- `FileNotFoundError`
- `fillna`
- `KeyError`
- `next`
- `read_csv`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `REPLAY_SCORES_PATH = VALIDATION_SCORES_DIR / f"{DATASET_NAME}__gold06a__test_replay_scores.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRAIN_LEAD_TIME_PATH = GOLD_ROOT / "anomaly_detection" / "summaries" / "multi_run_lead_time_comparison.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not REPLAY_SCORES_PATH.exists(): raise FileNotFoundError( f"Gold 06B needs the Gold 06A replay score output first: {REPLAY_SCORES_PATH}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `replay_scores_dataframe = pd.read_csv(REPLAY_SCORES_PATH)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `label_column_candidates = ["anomaly_flag", "is_anomaly", "target_flag", "label"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `resolved_label_column = next( (column for column in label_column_candidates if column in replay_scores_dataframe.columns), None,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if resolved_label_column is None: raise KeyError(f"Could not find label column. Checked: {label_column_candidates}")` | Controls validation, iteration, file handling, or error handling for this step. |
| `label_column: str = resolved_label_column` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if "plot_order_index" not in replay_scores_dataframe.columns: replay_scores_dataframe = replay_scores_dataframe.copy() replay_scores_dataframe["plot_order_index"] = np.arange(len(r` | Controls validation, iteration, file handling, or error handling for this step. |
| `basic_input_check = { "replay_scores_path": str(REPLAY_SCORES_PATH), "row_count": int(len(replay_scores_dataframe)), "label_column": label_column, "anomaly_rows": int(replay_scores` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="check", step="gold06a_replay_scores_loaded", message="Loaded Gold 06A replay scores for test early-warning validation.", data=basic_input_check, logger=logger,` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `basic_input_check` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 04 — 3. Early-warning helper functions

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `actual`
- `actual_failure_column`
- `actual_failure_flag`
- `anomaly`
- `any`
- `Any`
- `astype`
- `because`
- `binary`
- `BROKEN`
- `broken`
- `Build`
- `by`
- `closer`
- `column`
- `columns`
- `contains`
- `dataframe`
- `DataFrame`

### Outputs

- `alert_mask`
- `alerts_at_or_after_failure`
- `alerts_before_failure`
- `broken_status_flag`
- `build_early_warning_summary`
- `first_alert`
- `first_failure`
- `first_flag_index`
- `flagged_rows`
- `lead_rows`
- `resolve_failure_flag`
- `status_text`
- `total_alert_rows`
- `total_failure_rows`

### Key Operations

- `def resolve_failure_flag(dataframe: pd.DataFrame, label_column: str) -> pd.Series: """ Resolve the actual failure/anomaly flag used for early-warning timing. The label column is th`: Defines notebook-local logic used later in the notebook.
- `def first_flag_index(dataframe: pd.DataFrame, flag_column: str) -> int \| None: """Return the first plot_order_index where a binary flag equals 1.""" if flag_column not in dataframe`: Defines notebook-local logic used later in the notebook.
- `def build_early_warning_summary( dataframe: pd.DataFrame, *, run_key: str, run_label: str, flag_column: str, actual_failure_column: str = "actual_failure_flag",`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, Any]: """ Build one early-warning summary row for a replayed model output. """ if flag_column not in dataframe.columns: raise KeyError(f"Missing flag column for {run`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `any`
- `astype`
- `build_early_warning_summary`
- `contains`
- `fillna`
- `first_flag_index`
- `KeyError`
- `lower`
- `resolve_failure_flag`
- `sum`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def resolve_failure_flag(dataframe: pd.DataFrame, label_column: str) -> pd.Series: """ Resolve the actual failure/anomaly flag used for early-warning timing. The label column is th` | Defines notebook-local logic used later in the notebook. |
| `def first_flag_index(dataframe: pd.DataFrame, flag_column: str) -> int \| None: """Return the first plot_order_index where a binary flag equals 1.""" if flag_column not in dataframe` | Defines notebook-local logic used later in the notebook. |
| `def build_early_warning_summary( dataframe: pd.DataFrame, *, run_key: str, run_label: str, flag_column: str, actual_failure_column: str = "actual_failure_flag",` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, Any]: """ Build one early-warning summary row for a replayed model output. """ if flag_column not in dataframe.columns: raise KeyError(f"Missing flag column for {run` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 05 — 4. Build early-warning comparison across replayed model variants

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `actual_failure_flag`
- `are`
- `baseline`
- `Baseline`
- `baseline__baseline_flag`
- `build_early_warning_summary`
- `Cascade`
- `cascade_default__cascade_final_flag`
- `cascade_defaults`
- `cascade_tuned`
- `cascade_tuned__cascade_final_flag`
- `columns`
- `copy`
- `DataFrame`
- `Default`
- `flag`
- `Improved`
- `IsolationForest`
- `logger`
- `Medium`

### Outputs

- `available_run_specs`
- `early_warning_summary_dataframe`
- `flag_column`
- `label_column`
- `missing_run_specs`
- `replay_scores_dataframe`
- `run_key`
- `run_label`
- `run_specs`

### Key Operations

- `replay_scores_dataframe = replay_scores_dataframe.copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `replay_scores_dataframe["actual_failure_flag"] = resolve_failure_flag( replay_scores_dataframe, label_column=label_column,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `run_specs = [ { "selected_run_key": "baseline", "plot_run_label": "Baseline IsolationForest", "target_flag_column": "baseline__baseline_flag", }, { "selected_run_key": "cascade_def`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `available_run_specs = [ spec for spec in run_specs if spec["target_flag_column"] in replay_scores_dataframe.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_run_specs = [ spec for spec in run_specs if spec["target_flag_column"] not in replay_scores_dataframe.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if missing_run_specs: logger.warning("Some replay flag columns are missing: %s", missing_run_specs)`: Writes a logger message for traceability during notebook execution.
- `early_warning_summary_dataframe = pd.DataFrame([ build_early_warning_summary( replay_scores_dataframe, run_key=spec["selected_run_key"], run_label=spec["plot_run_label"], flag_colu`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `])`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `build_early_warning_summary`
- `copy`
- `DataFrame`
- `display`
- `resolve_failure_flag`
- `warning`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `replay_scores_dataframe = replay_scores_dataframe.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `replay_scores_dataframe["actual_failure_flag"] = resolve_failure_flag( replay_scores_dataframe, label_column=label_column,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `run_specs = [ { "selected_run_key": "baseline", "plot_run_label": "Baseline IsolationForest", "target_flag_column": "baseline__baseline_flag", }, { "selected_run_key": "cascade_def` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `available_run_specs = [ spec for spec in run_specs if spec["target_flag_column"] in replay_scores_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_run_specs = [ spec for spec in run_specs if spec["target_flag_column"] not in replay_scores_dataframe.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_run_specs: logger.warning("Some replay flag columns are missing: %s", missing_run_specs)` | Writes a logger message for traceability during notebook execution. |
| `early_warning_summary_dataframe = pd.DataFrame([ build_early_warning_summary( replay_scores_dataframe, run_key=spec["selected_run_key"], run_label=spec["plot_run_label"], flag_colu` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `])` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(early_warning_summary_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 06 — 5. Compare test early-warning results to the Gold 05 training-run summary

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_gold05`
- `_test_replay`
- `columns`
- `copy`
- `DataFrame`
- `early_warning_summary_dataframe`
- `else`
- `exists`
- `gold05_comparison_available`
- `lead_time_delta_minutes`
- `lead_time_minutes_to_failure_gold05`
- `lead_time_minutes_to_failure_test_replay`
- `left`
- `merge`
- `read_csv`
- `selected_run_key`
- `TRAIN_LEAD_TIME_PATH`

### Outputs

- `how`
- `lead_time_comparison_dataframe`
- `on`
- `suffixes`
- `train_lead_time_dataframe`

### Key Operations

- `if TRAIN_LEAD_TIME_PATH.exists(): train_lead_time_dataframe = pd.read_csv(TRAIN_LEAD_TIME_PATH) lead_time_comparison_dataframe = early_warning_summary_dataframe.merge( train_lead_t`: Loads input data, configuration, or artifacts required by the current stage.
- `else: train_lead_time_dataframe = pd.DataFrame() lead_time_comparison_dataframe = early_warning_summary_dataframe.copy() lead_time_comparison_dataframe["gold05_comparison_available`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(lead_time_comparison_dataframe)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `copy`
- `DataFrame`
- `display`
- `exists`
- `merge`
- `read_csv`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `if TRAIN_LEAD_TIME_PATH.exists(): train_lead_time_dataframe = pd.read_csv(TRAIN_LEAD_TIME_PATH) lead_time_comparison_dataframe = early_warning_summary_dataframe.merge( train_lead_t` | Loads input data, configuration, or artifacts required by the current stage. |
| `else: train_lead_time_dataframe = pd.DataFrame() lead_time_comparison_dataframe = early_warning_summary_dataframe.copy() lead_time_comparison_dataframe["gold05_comparison_available` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(lead_time_comparison_dataframe)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 07 — 6. Save outputs and plots

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__gold06b__test_early_warning_summary`
- `__gold06b__test_lead_time_comparison`
- `__gold06b__test_stage3_improved_timeline`
- `__gold06b__test_vs_gold05_lead_time_comparison`
- `Actual`
- `actual_failure_flag`
- `add`
- `alert`
- `anomaly`
- `artifacts`
- `ax`
- `axis`
- `bar`
- `bool`
- `by`
- `columns`
- `comparison`
- `CONTEXT_RECIPE_ID`
- `CONTEXT_STAGE`
- `copy`

### Outputs

- `data`
- `early_warning_summary_path`
- `kind`
- `lead_plot_path`
- `lead_time_comparison_path`
- `logger`
- `message`
- `plot_df`
- `step`
- `summary_json_path`
- `summary_payload`
- `timeline_flag_column`
- `timeline_plot_path`

### Key Operations

- `early_warning_summary_path = VALIDATION_SUMMARY_DIR / f"{DATASET_NAME}__gold06b__test_early_warning_summary.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `lead_time_comparison_path = VALIDATION_SUMMARY_DIR / f"{DATASET_NAME}__gold06b__test_vs_gold05_lead_time_comparison.csv"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `summary_json_path = VALIDATION_SUMMARY_DIR / f"{DATASET_NAME}__gold06b__test_early_warning_summary.json"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `lead_plot_path = VALIDATION_PLOTS_DIR / f"{DATASET_NAME}__gold06b__test_lead_time_comparison.png"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `timeline_plot_path = VALIDATION_PLOTS_DIR / f"{DATASET_NAME}__gold06b__test_stage3_improved_timeline.png"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `early_warning_summary_dataframe.to_csv(early_warning_summary_path, index=False)`: Writes an artifact or output used for review or downstream notebooks.
- `lead_time_comparison_dataframe.to_csv(lead_time_comparison_path, index=False)`: Writes an artifact or output used for review or downstream notebooks.
- `# Lead-time bar plot.`: Documents the purpose or boundary of the surrounding notebook step.
- `plot_df = early_warning_summary_dataframe.dropna(subset=["lead_time_minutes_to_failure"]).copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if not plot_df.empty: fig, ax = plt.subplots(figsize=(10, 5)) ax.bar(plot_df["plot_run_label"], plot_df["lead_time_minutes_to_failure"]) ax.set_title("Gold 06B Test Replay Lead Tim`: Writes an artifact or output used for review or downstream notebooks.
- `# Simple timeline for the selected/recommended Stage 3 output.`: Documents the purpose or boundary of the surrounding notebook step.
- `timeline_flag_column = "stage3_improved__cascade_final_flag"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `add`
- `bar`
- `bool`
- `copy`
- `dropna`
- `exists`
- `failure`
- `legend`
- `plot`
- `save_json`
- `savefig`
- `set_title`
- `set_xlabel`
- `set_ylabel`
- `show`
- `subplots`
- `tick_params`
- `tight_layout`
- `to_csv`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `early_warning_summary_path = VALIDATION_SUMMARY_DIR / f"{DATASET_NAME}__gold06b__test_early_warning_summary.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `lead_time_comparison_path = VALIDATION_SUMMARY_DIR / f"{DATASET_NAME}__gold06b__test_vs_gold05_lead_time_comparison.csv"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `summary_json_path = VALIDATION_SUMMARY_DIR / f"{DATASET_NAME}__gold06b__test_early_warning_summary.json"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `lead_plot_path = VALIDATION_PLOTS_DIR / f"{DATASET_NAME}__gold06b__test_lead_time_comparison.png"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `timeline_plot_path = VALIDATION_PLOTS_DIR / f"{DATASET_NAME}__gold06b__test_stage3_improved_timeline.png"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `early_warning_summary_dataframe.to_csv(early_warning_summary_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `lead_time_comparison_dataframe.to_csv(lead_time_comparison_path, index=False)` | Writes an artifact or output used for review or downstream notebooks. |
| `# Lead-time bar plot.` | Documents the purpose or boundary of the surrounding notebook step. |
| `plot_df = early_warning_summary_dataframe.dropna(subset=["lead_time_minutes_to_failure"]).copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not plot_df.empty: fig, ax = plt.subplots(figsize=(10, 5)) ax.bar(plot_df["plot_run_label"], plot_df["lead_time_minutes_to_failure"]) ax.set_title("Gold 06B Test Replay Lead Tim` | Writes an artifact or output used for review or downstream notebooks. |
| `# Simple timeline for the selected/recommended Stage 3 output.` | Documents the purpose or boundary of the surrounding notebook step. |
| `timeline_flag_column = "stage3_improved__cascade_final_flag"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if timeline_flag_column in replay_scores_dataframe.columns: fig, ax = plt.subplots(figsize=(12, 4)) ax.plot(replay_scores_dataframe["plot_order_index"], replay_scores_dataframe["ac` | Writes an artifact or output used for review or downstream notebooks. |
| `summary_payload = { "stage": CONTEXT_STAGE, "dataset": DATASET_NAME, "recipe_id": CONTEXT_RECIPE_ID, "replay_scores_path": str(REPLAY_SCORES_PATH), "early_warning_summary_path": st` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `save_json(summary_payload, summary_json_path)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ledger.add( kind="result", step="gold06b_outputs_saved", message="Saved Gold 06B test early-warning summaries, comparison artifacts, and plots.", data=summary_payload, logger=logge` | Records or exports ledger information for stage-level traceability. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `summary_payload` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to ledger-based traceability or ledger export behavior.
- Artifact or state outputs detected: CSV output, plot/image artifact.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 08 — 7. Gold 06B interpretation

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `behavior`
- `but`
- `comparison_output`
- `early`
- `early_warning_summary_path`
- `Gold`
- `held`
- `input`
- `lead_time_comparison_path`
- `on`
- `out`
- `outputs`
- `primary_output`
- `purpose`
- `REPLAY_SCORES_PATH`
- `replayed`
- `review`
- `scope`
- `style`
- `test`

### Outputs

- `interpretation`

### Key Operations

- `interpretation = { "purpose": "Validate early-warning behavior on replayed held-out test outputs.", "input": str(REPLAY_SCORES_PATH), "primary_output": str(early_warning_summary_pa`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `interpretation`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `interpretation = { "purpose": "Validate early-warning behavior on replayed held-out test outputs.", "input": str(REPLAY_SCORES_PATH), "primary_output": str(early_warning_summary_pa` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `interpretation` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

