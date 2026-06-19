# Notebook Code Reference: synthetic_01_generate_synethic_data

Notebook path:

`notebooks/synthetic/synthetic_01_generate_synethic_data.ipynb`

## Notebook Purpose

This notebook documents and runs a focused support step in the synthetic pump telemetry workflow.

Notebook stage:

`Synthetic support`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08, Code Cell 09, Code Cell 10, Code Cell 11, Code Cell 12, Code Cell 13, Code Cell 14, Code Cell 15, Code Cell 16, Code Cell 17, Code Cell 18, Code Cell 19, Code Cell 20, Code Cell 21, Code Cell 22, Code Cell 23, Code Cell 24, Code Cell 25, Code Cell 26, Code Cell 27, Code Cell 28, Code Cell 29, Code Cell 30, Code Cell 31, Code Cell 32, Code Cell 33, Code Cell 34, Code Cell 35, Code Cell 36, Code Cell 37, Code Cell 38, Code Cell 39, Code Cell 40, Code Cell 41, Code Cell 42, Code Cell 43, Code Cell 44, Code Cell 45, Code Cell 46, Code Cell 47, Code Cell 48, Code Cell 49, Code Cell 50, Code Cell 51, Code Cell 52, Code Cell 53, Code Cell 54, Code Cell 55, Code Cell 56, Code Cell 57, Code Cell 58, Code Cell 59, Code Cell 60, Code Cell 61, Code Cell 62, Code Cell 63, Code Cell 64, Code Cell 65, Code Cell 66 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__future__`
- `annotations`
- `Any`
- `append_truth_index`
- `build_missingness_spec_from_truth_payload`
- `build_postgres_url`
- `build_truth_config_block`
- `build_truth_record`
- `config_loader`
- `configure_logging`
- `Core`
- `core`
- `database`
- `datetime`
- `ensure_sequence`
- `env_helpers`
- `env_str`
- `EpisodeSpec`
- `execute_sql`
- `export`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `from __future__ import annotations`: Imports a dependency or project helper used by later cells.
- `from pathlib import Path`: Imports a dependency or project helper used by later cells.
- `import logging`: Imports a dependency or project helper used by later cells.
- `import json`: Imports a dependency or project helper used by later cells.
- `from typing import Dict, List, Optional, Tuple, Any, Iterable, Sequence`: Imports a dependency or project helper used by later cells.
- `import inspect`: Imports a dependency or project helper used by later cells.
- `import numpy as np`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `import random`: Imports a dependency or project helper used by later cells.
- `from datetime import datetime, timedelta, timezone`: Imports a dependency or project helper used by later cells.
- `# Core Utils`: Documents the purpose or boundary of the surrounding notebook step.
- `from utils.core.paths import get_paths`: Imports a dependency or project helper used by later cells.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `from __future__ import annotations` | Imports a dependency or project helper used by later cells. |
| `from pathlib import Path` | Imports a dependency or project helper used by later cells. |
| `import logging` | Imports a dependency or project helper used by later cells. |
| `import json` | Imports a dependency or project helper used by later cells. |
| `from typing import Dict, List, Optional, Tuple, Any, Iterable, Sequence` | Imports a dependency or project helper used by later cells. |
| `import inspect` | Imports a dependency or project helper used by later cells. |
| `import numpy as np` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `import random` | Imports a dependency or project helper used by later cells. |
| `from datetime import datetime, timedelta, timezone` | Imports a dependency or project helper used by later cells. |
| `# Core Utils` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.core.paths import get_paths` | Imports a dependency or project helper used by later cells. |
| `from utils.core.file_io import ( save_data, load_json,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.logging_setup import ( configure_logging, log_layer_paths,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.config_loader import ( load_pipeline_config, set_wandb_dir_from_config, export_config_snapshot, build_truth_config_block,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.truths import ( make_process_run_id, initialize_layer_truth, update_truth_section, build_truth_record, save_truth_record, append_truth_index, stamp_truth_columns, l` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Synthetic Generator` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.synthetic.generator.profiles import ( load_rich_profile_csv, load_and_merge_rich_profiles, load_correlation_pairs_csv, load_group_map_csv, load_fault_pairings_csv,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.generator.missingness import build_missingness_spec_from_truth_payload` | Imports a dependency or project helper used by later cells. |
| `from utils.synthetic.generator.generator import ( SyntheticGenerator, EpisodeSpec,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.synthetic.generator.postgres_writer import ( ensure_sequence, reserve_next_batch_id, reserve_cycle_range, reset_sequence, reset_synthetic_sequences, write_stream_batch,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#from utils.synthetic.generator.export import export_synthetic_batch_to_parquet` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Postgres` | Documents the purpose or boundary of the surrounding notebook step. |
| `from utils.database.postgres import ( get_engine_from_env, build_postgres_url, execute_sql, read_sql_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.database.layer_postgres import ( write_layer_dataframe, prepare_layer_dataframe,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.env_helpers import env_str` | Imports a dependency or project helper used by later cells. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: Parquet output, SQL or medallion table write, truth record.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `at`
- `d`
- `Data`
- `datetime`
- `f`
- `Generation`
- `H`
- `hours`
- `M`
- `m`
- `now`
- `Starting`
- `store`
- `strftime`
- `Synthetic`
- `timedelta`
- `Y_`

### Outputs

- `generation_started_adjusted_time`
- `generation_started_current_datetime`
- `generation_started_formatted_datetime`

### Key Operations

- `generation_started_current_datetime = datetime.now()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `generation_started_adjusted_time = generation_started_current_datetime - timedelta(hours=4)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `generation_started_formatted_datetime = generation_started_adjusted_time.strftime("%m%d%Y_%H%M")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print(f"Starting Synthetic Data Generation at {generation_started_formatted_datetime}")`: Displays a notebook-facing result for inspection.
- `%store generation_started_current_datetime`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `now`
- `strftime`
- `timedelta`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `generation_started_current_datetime = datetime.now()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `generation_started_adjusted_time = generation_started_current_datetime - timedelta(hours=4)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `generation_started_formatted_datetime = generation_started_adjusted_time.strftime("%m%d%Y_%H%M")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print(f"Starting Synthetic Data Generation at {generation_started_formatted_datetime}")` | Displays a notebook-facing result for inspection. |
| `%store generation_started_current_datetime` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 03 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `default`
- `Notebook`
- `params`
- `pump`
- `synthetic`
- `train`

### Outputs

- `DATASET`
- `MODE`
- `PROFILE`
- `STAGE`

### Key Operations

- `# --- Notebook params ---`: Documents the purpose or boundary of the surrounding notebook step.
- `STAGE = "synthetic"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET = "pump"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `MODE = "train"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PROFILE = "default"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# --- Notebook params ---` | Documents the purpose or boundary of the surrounding notebook step. |
| `STAGE = "synthetic"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MODE = "train"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 04 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `batch`
- `configs`
- `data`
- `execution_mode`
- `exist_ok`
- `get`
- `get_paths`
- `load_pipeline_config`
- `logs_root`
- `lower`
- `mkdir`
- `name`
- `notebook`
- `orchestration_mode`
- `parents`
- `resolved_paths`
- `root`
- `set_wandb_dir_from_config`
- `strip`
- `synthetic`

### Outputs

- `ARTIFACTS_ROOT`
- `CONFIG`
- `config_obj`
- `config_root`
- `dataset`
- `DATASET_NAME`
- `LOGS_PATH`
- `mode`
- `PATHS`
- `paths`
- `PIPELINE`
- `PIPELINE_MODE`
- `profile`
- `project_root`
- `stage`
- `SYN_CFG`
- `TRUTH_INDEX_PATH`
- `TRUTH_VERSION`
- `TRUTHS_PATH`

### Key Operations

- `paths = get_paths()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `config_obj = load_pipeline_config( config_root=paths.configs, stage=STAGE, dataset=DATASET, mode=MODE, profile=PROFILE, project_root=paths.root,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `CONFIG = config_obj.data`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SYN_CFG = CONFIG["synthetic"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PATHS = CONFIG["resolved_paths"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PIPELINE = CONFIG.get("pipeline", {"execution_mode": "batch", "orchestration_mode": "notebook"})`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PIPELINE_MODE = PIPELINE["execution_mode"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET_NAME = str(CONFIG["dataset"]["name"]).strip().lower()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_VERSION = CONFIG["versions"]["truth"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTHS_PATH = Path(PATHS["truths_dir"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TRUTH_INDEX_PATH = Path(PATHS["truth_index_path"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get`
- `get_paths`
- `load_pipeline_config`
- `lower`
- `mkdir`
- `Path`
- `set_wandb_dir_from_config`
- `strip`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `paths = get_paths()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `config_obj = load_pipeline_config( config_root=paths.configs, stage=STAGE, dataset=DATASET, mode=MODE, profile=PROFILE, project_root=paths.root,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `CONFIG = config_obj.data` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SYN_CFG = CONFIG["synthetic"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PATHS = CONFIG["resolved_paths"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PIPELINE = CONFIG.get("pipeline", {"execution_mode": "batch", "orchestration_mode": "notebook"})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PIPELINE_MODE = PIPELINE["execution_mode"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_NAME = str(CONFIG["dataset"]["name"]).strip().lower()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_VERSION = CONFIG["versions"]["truth"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTHS_PATH = Path(PATHS["truths_dir"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTH_INDEX_PATH = Path(PATHS["truth_index_path"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `LOGS_PATH = Path(PATHS["logs_root"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ARTIFACTS_ROOT = Path(PATHS["artifacts_root"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TRUTHS_PATH.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `LOGS_PATH.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `set_wandb_dir_from_config(CONFIG)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("DATASET_NAME:", DATASET_NAME)` | Displays a notebook-facing result for inspection. |
| `print("TRUTHS_PATH:", TRUTHS_PATH)` | Displays a notebook-facing result for inspection. |
| `print("ARTIFACTS_ROOT:", ARTIFACTS_ROOT)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 05 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `capstone`
- `configure_logging`
- `Create`
- `current_layer`
- `Data`
- `DEBUG`
- `file`
- `Generation`
- `getLogger`
- `gold`
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

- `level`
- `logger`
- `overwrite_handlers`
- `synthetic_log_path`

### Key Operations

- `# Logging Setup`: Documents the purpose or boundary of the surrounding notebook step.
- `# Create gold log path`: Documents the purpose or boundary of the surrounding notebook step.
- `synthetic_log_path = paths.logs / "synthetic_data_generator.log"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Initial Logger`: Documents the purpose or boundary of the surrounding notebook step.
- `configure_logging( "capstone", synthetic_log_path, level=logging.DEBUG, overwrite_handlers=True,`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Initiate Logger and log file`: Documents the purpose or boundary of the surrounding notebook step.
- `logger = logging.getLogger("capstone.synthetic")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Log load and initiation`: Documents the purpose or boundary of the surrounding notebook step.
- `logger.info("Synethetic Data Generation starting")`: Writes a logger message for traceability during notebook execution.
- `# Log paths loads`: Documents the purpose or boundary of the surrounding notebook step.
- `log_layer_paths(paths, current_layer="synthetic", logger=logger)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `configure_logging`
- `getLogger`
- `info`
- `log_layer_paths`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Logging Setup` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Create gold log path` | Documents the purpose or boundary of the surrounding notebook step. |
| `synthetic_log_path = paths.logs / "synthetic_data_generator.log"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Initial Logger` | Documents the purpose or boundary of the surrounding notebook step. |
| `configure_logging( "capstone", synthetic_log_path, level=logging.DEBUG, overwrite_handlers=True,` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Initiate Logger and log file` | Documents the purpose or boundary of the surrounding notebook step. |
| `logger = logging.getLogger("capstone.synthetic")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Log load and initiation` | Documents the purpose or boundary of the surrounding notebook step. |
| `logger.info("Synethetic Data Generation starting")` | Writes a logger message for traceability during notebook execution. |
| `# Log paths loads` | Documents the purpose or boundary of the surrounding notebook step. |
| `log_layer_paths(paths, current_layer="synthetic", logger=logger)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 06 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `Accepts`
- `an`
- `are`
- `bool`
- `def`
- `either`
- `exclusive`
- `Expected`
- `f`
- `Generator`
- `got`
- `inclusive`
- `integer`
- `integers`
- `isinstance`
- `low_inclusive`
- `numpy`
- `raise`
- `random`
- `range`

### Outputs

- `high`
- `low`
- `sample_int_range`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `integers`
- `isinstance`
- `sample_int_range`
- `type`
- `TypeError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 07 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `Accepts`
- `def`
- `either`
- `Expected`
- `f`
- `floating`
- `Generator`
- `got`
- `integer`
- `isinstance`
- `raise`
- `random`
- `range`
- `Returns`
- `rng`
- `tuple`
- `type`
- `TypeError`
- `uniform`

### Outputs

- `high`
- `low`
- `sample_float_range`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `isinstance`
- `sample_float_range`
- `type`
- `TypeError`
- `uniform`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 08 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `are`
- `artifact`
- `artifacts`
- `by`
- `clean`
- `fallback`
- `Keep`
- `layer`
- `New`
- `normal`
- `old`
- `older`
- `Parent`
- `produced`
- `runs`
- `Silver`
- `silver_eda`
- `silver_subsets`
- `the`

### Outputs

- `SILVER_LEGACY_EDA_LAYER`
- `SILVER_PROFILE_LAYER`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Parent Silver artifact layer`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# New clean-normal artifacts are produced by silver_subsets.`: Documents the purpose or boundary of the surrounding notebook step.
- `# Keep the old silver_eda layer as a fallback for older runs.`: Documents the purpose or boundary of the surrounding notebook step.
- `SILVER_PROFILE_LAYER = "silver_subsets"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_LEGACY_EDA_LAYER = "silver_eda"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("SILVER_PROFILE_LAYER:", SILVER_PROFILE_LAYER)`: Displays a notebook-facing result for inspection.
- `print("SILVER_LEGACY_EDA_LAYER:", SILVER_LEGACY_EDA_LAYER)`: Displays a notebook-facing result for inspection.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Parent Silver artifact layer` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# New clean-normal artifacts are produced by silver_subsets.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Keep the old silver_eda layer as a fallback for older runs.` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_PROFILE_LAYER = "silver_subsets"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_LEGACY_EDA_LAYER = "silver_eda"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("SILVER_PROFILE_LAYER:", SILVER_PROFILE_LAYER)` | Displays a notebook-facing result for inspection. |
| `print("SILVER_LEGACY_EDA_LAYER:", SILVER_LEGACY_EDA_LAYER)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 09 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `Any`
- `append`
- `are`
- `artifacts`
- `Assumes`
- `back`
- `behavior`
- `clean`
- `continue`
- `Could`
- `dataset`
- `def`
- `encoding`
- `entries`
- `error`
- `Exception`
- `exists`
- `f`
- `Fall`

### Outputs

- `dataset_name`
- `dataset_name_norm`
- `get_latest_parent_profile_truth_hash`
- `get_latest_truth_hash`
- `latest_record`
- `layer_name`
- `layer_name_norm`
- `line`
- `rec`
- `rec_dataset`
- `rec_layer`
- `truth_hash`
- `truth_index_path`

### Key Operations

- `# Get Latest Truth Hash`: Documents the purpose or boundary of the surrounding notebook step.
- `def get_latest_truth_hash( *, truth_index_path: Path, layer_name: str, dataset_name: str,`: Defines notebook-local logic used later in the notebook.
- `) -> str: """ Return the most recent truth_hash for a given layer + dataset from truth_index.jsonl. Assumes truth_index.jsonl is append-only and newer entries are later in the file`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `def get_latest_parent_profile_truth_hash( *, truth_index_path: Path, dataset_name: str, preferred_layer_name: str = "silver_subsets", fallback_layer_name: str = "silver_eda",`: Defines notebook-local logic used later in the notebook.
- `) -> tuple[str, str]: """ Resolve the latest parent truth hash for the synthetic generator. Preferred behavior: - Use silver_subsets for clean-normal profile artifacts. - Fall back`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `exists`
- `FileNotFoundError`
- `get`
- `get_latest_parent_profile_truth_hash`
- `get_latest_truth_hash`
- `loads`
- `lower`
- `open`
- `strip`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Get Latest Truth Hash` | Documents the purpose or boundary of the surrounding notebook step. |
| `def get_latest_truth_hash( *, truth_index_path: Path, layer_name: str, dataset_name: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> str: """ Return the most recent truth_hash for a given layer + dataset from truth_index.jsonl. Assumes truth_index.jsonl is append-only and newer entries are later in the file` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `def get_latest_parent_profile_truth_hash( *, truth_index_path: Path, dataset_name: str, preferred_layer_name: str = "silver_subsets", fallback_layer_name: str = "silver_eda",` | Defines notebook-local logic used later in the notebook. |
| `) -> tuple[str, str]: """ Resolve the latest parent truth hash for the synthetic generator. Preferred behavior: - Use silver_subsets for clean-normal profile artifacts. - Fall back` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 10 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `a`
- `actual_broken_rows`
- `alias`
- `allowed`
- `append`
- `Backward`
- `batch`
- `bool`
- `break`
- `buildup_range`
- `can`
- `cap`
- `capstone`
- `cells`
- `compatible`
- `continue`
- `control`
- `core`
- `current_rows_per_failure`

### Outputs

- `ALLOW_OVERSHOOT`
- `APPEND_MODE`
- `ARTIFACT_NAME`
- `BUILDUP`
- `BUILDUP_FRACTION`
- `DATASET`
- `dataset_name`
- `EPISODE_MAX_ROWS`
- `EXPORT_DIRECTORY`
- `EXPORT_ENABLED`
- `FAILURE`
- `fallback_layer_name`
- `MAGNITUDE`
- `MAX_EPISODES`
- `MODE`
- `NORMAL_AFTER`
- `NORMAL_BEFORE`
- `OUTPUT_MODE`
- `PG_SCHEMA`
- `preferred_layer_name`

### Key Operations

- `# Updated`: Documents the purpose or boundary of the surrounding notebook step.
- `# --- Notebook params ---`: Documents the purpose or boundary of the surrounding notebook step.
- `STAGE = SYN_CFG["layer_name"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `DATASET = "pump"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `MODE = "train"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PROFILE = "default"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Parent truth hash from the latest Silver profile/subset run`: Documents the purpose or boundary of the surrounding notebook step.
- `SILVER_PARENT_LAYER_NAME, SILVER_PARENT_TRUTH_HASH = get_latest_parent_profile_truth_hash( truth_index_path=TRUTH_INDEX_PATH, dataset_name=DATASET_NAME, preferred_layer_name=SILVER`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Backward-compatible alias so older cells do not immediately break.`: Documents the purpose or boundary of the surrounding notebook step.
- `SILVER_EDA_TRUTH_HASH = SILVER_PARENT_TRUTH_HASH`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("SILVER_PARENT_LAYER_NAME:", SILVER_PARENT_LAYER_NAME)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `bool`
- `control`
- `get_latest_parent_profile_truth_hash`
- `lower`
- `make_process_run_id`
- `overrides`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Updated` | Documents the purpose or boundary of the surrounding notebook step. |
| `# --- Notebook params ---` | Documents the purpose or boundary of the surrounding notebook step. |
| `STAGE = SYN_CFG["layer_name"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MODE = "train"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Parent truth hash from the latest Silver profile/subset run` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_PARENT_LAYER_NAME, SILVER_PARENT_TRUTH_HASH = get_latest_parent_profile_truth_hash( truth_index_path=TRUTH_INDEX_PATH, dataset_name=DATASET_NAME, preferred_layer_name=SILVER` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Backward-compatible alias so older cells do not immediately break.` | Documents the purpose or boundary of the surrounding notebook step. |
| `SILVER_EDA_TRUTH_HASH = SILVER_PARENT_TRUTH_HASH` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("SILVER_PARENT_LAYER_NAME:", SILVER_PARENT_LAYER_NAME)` | Displays a notebook-facing result for inspection. |
| `print("SILVER_PARENT_TRUTH_HASH:", SILVER_PARENT_TRUTH_HASH)` | Displays a notebook-facing result for inspection. |
| `# Faults` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Episode overrides (easy test knobs)` | Documents the purpose or boundary of the surrounding notebook step. |
| `PRIMARY_SENSOR = None # None => first sensor` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PRIMARY_FAULT_TYPE = list(SYN_CFG["faults"]["allowed"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Episode Settings` | Documents the purpose or boundary of the surrounding notebook step. |
| `NORMAL_BEFORE = list(SYN_CFG["episode"]["normal_before_range"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BUILDUP = list(SYN_CFG["episode"]["buildup_range"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FAILURE = list(SYN_CFG["episode"]["failure_range"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RECOVERY = list(SYN_CFG["episode"]["recovery_range"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `NORMAL_AFTER = list(SYN_CFG["episode"]["normal_after_range"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MAGNITUDE = list(SYN_CFG["episode"]["magnitude_range"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BUILDUP_FRACTION = float(SYN_CFG["episode"]["buildup_fraction"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SYNTH_PROCESS_RUN_ID = make_process_run_id(SYN_CFG["process_run_id_prefix"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Outputs` | Documents the purpose or boundary of the surrounding notebook step. |
| `OUTPUT_MODE = SYN_CFG["output_mode"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Postgres settings` | Documents the purpose or boundary of the surrounding notebook step. |
| `PG_SCHEMA = str(SYN_CFG["postgres"]["schema"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TABLE_ARTIFACT_NAME = str(SYN_CFG["postgres"]["table_artifact_name"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# medallion naming: synthetic_<dataset>_<artifact_name>` | Documents the purpose or boundary of the surrounding notebook step. |
| `ARTIFACT_NAME = "stream"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Export` | Documents the purpose or boundary of the surrounding notebook step. |
| `EXPORT_ENABLED = bool(SYN_CFG["export"]["enabled"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `EXPORT_DIRECTORY = str(SYN_CFG["export"]["export_dir_key"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---- Mode switch ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `MODE = str(SYN_CFG["generator"]["mode"]) # "single" \| "batch"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TARGET_ROWS = int(SYN_CFG["generator"]["target_rows"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MAX_EPISODES = int(SYN_CFG["generator"]["max_episodes"]) # safety cap` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---- policy knobs ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `EPISODE_MAX_ROWS = int(SYN_CFG["generator"]["episode_max_rows"]) # prevents monster episodes; forces multiple episodes in a 10k batch` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ALLOW_OVERSHOOT = bool(SYN_CFG["generator"]["allow_overshoot"]) # if True, can overshoot when remaining can't fit minimum core` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---- failure rarity control (match real dataset frequency) ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `#Forumla to determine rows per failure` | Documents the purpose or boundary of the surrounding notebook step. |
| `#new_rows_per_failure = current_rows_per_failure * (actual_broken_rows / target_broken_rows)` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Real dataset: ~7 failures per 220,320 rows => ~1 failure per 35,474 rows // 250,000 / 7 = 142_857 -> 35_714.2857` | Documents the purpose or boundary of the surrounding notebook step. |
| `ROWS_PER_FAILURE = int(SYN_CFG["generator"]["rows_per_failure"]) # ~31_474.2857` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#PG_SCHEMA = "capstone"` | Documents the purpose or boundary of the surrounding notebook step. |
| `ARTIFACT_NAME = "stream"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TABLE_NAME = f"synthetic_{DATASET_NAME.lower()}_{ARTIFACT_NAME}"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---- write mode flags` | Documents the purpose or boundary of the surrounding notebook step. |
| `WRITE_MODE = str(SYN_CFG["generator"]["write_mode"]) # "reset" \| "append"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `APPEND_MODE = str(SYN_CFG["generator"]["append_mode"]) # "continue" \| "renumber" (only matters if WRITE_MODE="append")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 11 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `Backward`
- `BUILDUP`
- `by`
- `case`
- `cells`
- `compatible`
- `debug`
- `Do`
- `env_str`
- `expected`
- `FAILURE`
- `id`
- `Keep`
- `later`
- `Lowercase`
- `MAGNITUDE`
- `new`
- `NORMAL_AFTER`
- `NORMAL_BEFORE`

### Outputs

- `aliases`
- `buildup_range`
- `BUILDUP_RANGE`
- `failure_range`
- `FAILURE_RANGE`
- `MAGNITUDE_RANGE`
- `magnitude_range`
- `normal_after_range`
- `NORMAL_AFTER_RANGE`
- `NORMAL_BEFORE_RANGE`
- `normal_before_range`
- `process_run_id`
- `recovery_range`
- `RECOVERY_RANGE`
- `run_id`
- `RUN_ID`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Backward-compatible runtime aliases`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Keep one stable run id for the whole notebook.`: Documents the purpose or boundary of the surrounding notebook step.
- `# Do not regenerate a new process_run_id later.`: Documents the purpose or boundary of the surrounding notebook step.
- `RUN_ID = env_str( "SYNTHETIC_RUN_ID", SYNTH_PROCESS_RUN_ID, aliases=("RUN_ID",),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `process_run_id = RUN_ID`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Range aliases expected by older/debug cells.`: Documents the purpose or boundary of the surrounding notebook step.
- `NORMAL_BEFORE_RANGE = NORMAL_BEFORE`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `BUILDUP_RANGE = BUILDUP`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `FAILURE_RANGE = FAILURE`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `env_str`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Backward-compatible runtime aliases` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Keep one stable run id for the whole notebook.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Do not regenerate a new process_run_id later.` | Documents the purpose or boundary of the surrounding notebook step. |
| `RUN_ID = env_str( "SYNTHETIC_RUN_ID", SYNTH_PROCESS_RUN_ID, aliases=("RUN_ID",),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `process_run_id = RUN_ID` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Range aliases expected by older/debug cells.` | Documents the purpose or boundary of the surrounding notebook step. |
| `NORMAL_BEFORE_RANGE = NORMAL_BEFORE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `BUILDUP_RANGE = BUILDUP` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FAILURE_RANGE = FAILURE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RECOVERY_RANGE = RECOVERY` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `NORMAL_AFTER_RANGE = NORMAL_AFTER` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MAGNITUDE_RANGE = MAGNITUDE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Lowercase aliases in case older cells still reference them.` | Documents the purpose or boundary of the surrounding notebook step. |
| `normal_before_range = NORMAL_BEFORE_RANGE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `buildup_range = BUILDUP_RANGE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `failure_range = FAILURE_RANGE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `recovery_range = RECOVERY_RANGE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `normal_after_range = NORMAL_AFTER_RANGE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `magnitude_range = MAGNITUDE_RANGE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `run_id = RUN_ID` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("RUN_ID:", RUN_ID)` | Displays a notebook-facing result for inspection. |
| `print("RECOVERY_RANGE:", RECOVERY_RANGE)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 12 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `bool`
- `build`
- `by`
- `calibration`
- `config`
- `correlation_tuning`
- `enabled`
- `extracts`
- `generator`
- `get`
- `info`
- `logger`
- `mean_within_k_std`
- `s`
- `sensor_15`
- `sensor_50`
- `std_ratio_bounds`
- `SYN_CFG`
- `Synthetic`
- `the`

### Outputs

- `CAL_CFG`
- `CALIBRATION_ENABLED`
- `CALIBRATION_MEAN_WITHIN_K_STD`
- `CALIBRATION_STD_RATIO_BOUNDS`
- `CORRELATION_CLUSTER_DERIVATION`
- `CORRELATION_HOTSPOT_CLUSTERS`
- `CORRELATION_TUNING_CFG`
- `FAULT_EXCLUDED_SENSORS`

### Key Operations

- `# --- Synthetic config extracts used by the generator build ---`: Documents the purpose or boundary of the surrounding notebook step.
- `CAL_CFG = SYN_CFG.get("calibration", {}) or {}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CALIBRATION_ENABLED = bool(CAL_CFG.get("enabled", True))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CALIBRATION_MEAN_WITHIN_K_STD = float(CAL_CFG.get("mean_within_k_std", 1.00))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CALIBRATION_STD_RATIO_BOUNDS = tuple(CAL_CFG.get("std_ratio_bounds", [0.90, 1.35]))`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CORRELATION_HOTSPOT_CLUSTERS = SYN_CFG.get("correlation_hotspot_clusters", []) or []`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CORRELATION_CLUSTER_DERIVATION = dict(SYN_CFG.get("correlation_cluster_derivation", {}) or {})`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `FAULT_EXCLUDED_SENSORS = list(SYN_CFG.get("fault_excluded_sensors", ["sensor_15", "sensor_50"]) or [])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CORRELATION_TUNING_CFG = dict(SYN_CFG.get("correlation_tuning", {}) or {})`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("CALIBRATION_ENABLED:", CALIBRATION_ENABLED)`: Displays a notebook-facing result for inspection.
- `print("CALIBRATION_MEAN_WITHIN_K_STD:", CALIBRATION_MEAN_WITHIN_K_STD)`: Displays a notebook-facing result for inspection.
- `print("CALIBRATION_STD_RATIO_BOUNDS:", CALIBRATION_STD_RATIO_BOUNDS)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `bool`
- `get`
- `info`
- `tuple`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# --- Synthetic config extracts used by the generator build ---` | Documents the purpose or boundary of the surrounding notebook step. |
| `CAL_CFG = SYN_CFG.get("calibration", {}) or {}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CALIBRATION_ENABLED = bool(CAL_CFG.get("enabled", True))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CALIBRATION_MEAN_WITHIN_K_STD = float(CAL_CFG.get("mean_within_k_std", 1.00))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CALIBRATION_STD_RATIO_BOUNDS = tuple(CAL_CFG.get("std_ratio_bounds", [0.90, 1.35]))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CORRELATION_HOTSPOT_CLUSTERS = SYN_CFG.get("correlation_hotspot_clusters", []) or []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CORRELATION_CLUSTER_DERIVATION = dict(SYN_CFG.get("correlation_cluster_derivation", {}) or {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FAULT_EXCLUDED_SENSORS = list(SYN_CFG.get("fault_excluded_sensors", ["sensor_15", "sensor_50"]) or [])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CORRELATION_TUNING_CFG = dict(SYN_CFG.get("correlation_tuning", {}) or {})` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("CALIBRATION_ENABLED:", CALIBRATION_ENABLED)` | Displays a notebook-facing result for inspection. |
| `print("CALIBRATION_MEAN_WITHIN_K_STD:", CALIBRATION_MEAN_WITHIN_K_STD)` | Displays a notebook-facing result for inspection. |
| `print("CALIBRATION_STD_RATIO_BOUNDS:", CALIBRATION_STD_RATIO_BOUNDS)` | Displays a notebook-facing result for inspection. |
| `print("CORRELATION_HOTSPOT_CLUSTERS:", CORRELATION_HOTSPOT_CLUSTERS)` | Displays a notebook-facing result for inspection. |
| `print("CORRELATION_CLUSTER_DERIVATION:", CORRELATION_CLUSTER_DERIVATION)` | Displays a notebook-facing result for inspection. |
| `print("FAULT_EXCLUDED_SENSORS:", FAULT_EXCLUDED_SENSORS)` | Displays a notebook-facing result for inspection. |
| `print("CORRELATION_TUNING_CFG:", CORRELATION_TUNING_CFG)` | Displays a notebook-facing result for inspection. |
| `logger.info("CALIBRATION_ENABLED: %s", CALIBRATION_ENABLED)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("CALIBRATION_MEAN_WITHIN_K_STD: %s", CALIBRATION_MEAN_WITHIN_K_STD)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("CALIBRATION_STD_RATIO_BOUNDS: %s", CALIBRATION_STD_RATIO_BOUNDS)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("CORRELATION_HOTSPOT_CLUSTERS: %s", CORRELATION_HOTSPOT_CLUSTERS)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("CORRELATION_CLUSTER_DERIVATION: %s", CORRELATION_CLUSTER_DERIVATION)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("FAULT_EXCLUDED_SENSORS: %s", FAULT_EXCLUDED_SENSORS)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("CORRELATION_TUNING_CFG: %s", CORRELATION_TUNING_CFG)` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 13 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `a`
- `append`
- `ARTIFACT_NAME`
- `batch`
- `can`
- `cap`
- `capstone`
- `continue`
- `control`
- `core`
- `dataset`
- `DATASET_NAME`
- `episodes`
- `f`
- `failure`
- `failures`
- `fit`
- `flags`
- `forces`

### Outputs

- `ALLOW_OVERSHOOT`
- `APPEND_MODE`
- `EPISODE_MAX_ROWS`
- `MAX_EPISODES`
- `MODE`
- `ROWS_PER_FAILURE`
- `TABLE_NAME`
- `TARGET_ROWS`
- `WRITE_MODE`

### Key Operations

- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# ---- Mode switch ----`: Documents the purpose or boundary of the surrounding notebook step.
- `MODE = "batch" # "single" \| "batch"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TARGET_ROWS = 225_000`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `MAX_EPISODES = 1_000_000 # safety cap`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# ---- policy knobs ----`: Documents the purpose or boundary of the surrounding notebook step.
- `EPISODE_MAX_ROWS = 12_000 # prevents monster episodes; forces multiple episodes in a 10k batch`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ALLOW_OVERSHOOT = False # if True, can overshoot when remaining can't fit minimum core`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# ---- failure rarity control (match real dataset frequency) ----`: Documents the purpose or boundary of the surrounding notebook step.
- `# Real dataset: ~7 failures per 250,000 rows => ~1 failure per 35,714 rows`: Documents the purpose or boundary of the surrounding notebook step.
- `ROWS_PER_FAILURE = 32_000 # ~35714.2857`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#PG_SCHEMA = "capstone"`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `control`
- `lower`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# ---- Mode switch ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `MODE = "batch" # "single" \| "batch"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TARGET_ROWS = 225_000` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MAX_EPISODES = 1_000_000 # safety cap` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---- policy knobs ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `EPISODE_MAX_ROWS = 12_000 # prevents monster episodes; forces multiple episodes in a 10k batch` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ALLOW_OVERSHOOT = False # if True, can overshoot when remaining can't fit minimum core` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---- failure rarity control (match real dataset frequency) ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Real dataset: ~7 failures per 250,000 rows => ~1 failure per 35,714 rows` | Documents the purpose or boundary of the surrounding notebook step. |
| `ROWS_PER_FAILURE = 32_000 # ~35714.2857` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#PG_SCHEMA = "capstone"` | Documents the purpose or boundary of the surrounding notebook step. |
| `#ARTIFACT_NAME = "stream"` | Documents the purpose or boundary of the surrounding notebook step. |
| `TABLE_NAME = f"synthetic_{DATASET_NAME.lower()}_{ARTIFACT_NAME}"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---- write mode flags` | Documents the purpose or boundary of the surrounding notebook step. |
| `WRITE_MODE = "reset" # "reset" \| "append"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `APPEND_MODE = "renumber" # "continue" \| "renumber" (only matters if WRITE_MODE="append")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 14 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `alias`
- `B`
- `Backward`
- `build_missingness_spec_from_truth_payload`
- `can`
- `cell`
- `cells`
- `compatible`
- `count`
- `downstream`
- `existing`
- `get`
- `get_parent_truth_hash`
- `get_pipeline_mode_from_truth`
- `get_truth_hash`
- `info`
- `items`
- `keep`
- `load_truth_record_by_hash`
- `logger`

### Outputs

- `dataset_name`
- `layer_name`
- `missingness_payload`
- `MISSINGNESS_SENSOR_OVERRIDES`
- `missingness_spec`
- `parent_mode`
- `PARENT_TRUTH_HASH`
- `PIPELINE_MODE`
- `silver_eda_truth`
- `silver_parent_truth`
- `silver_preeda_truth`
- `SILVER_PREEDA_TRUTH_HASH`
- `truth_dir`
- `truth_hash`

### Key Operations

- `if SILVER_PARENT_TRUTH_HASH is None or str(SILVER_PARENT_TRUTH_HASH).strip() == "": raise ValueError("Set SILVER_PARENT_TRUTH_HASH in the parameter cell.")`: Controls validation, iteration, file handling, or error handling for this step.
- `silver_parent_truth = load_truth_record_by_hash( truth_dir=TRUTHS_PATH, layer_name=SILVER_PARENT_LAYER_NAME, dataset_name=DATASET_NAME, truth_hash=str(SILVER_PARENT_TRUTH_HASH).str`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# Backward-compatible alias so existing downstream cells can keep working.`: Documents the purpose or boundary of the surrounding notebook step.
- `silver_eda_truth = silver_parent_truth`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `PARENT_TRUTH_HASH = get_truth_hash(silver_parent_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `SILVER_PREEDA_TRUTH_HASH = get_parent_truth_hash(silver_parent_truth)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `silver_preeda_truth = load_truth_record_by_hash( truth_dir=TRUTHS_PATH, layer_name="silver", dataset_name=DATASET_NAME, truth_hash=str(SILVER_PREEDA_TRUTH_HASH).strip(),`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_payload = ( (silver_preeda_truth.get("runtime_facts", {}) or {}) .get("missingness_quarantine", {}) or {}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missingness_spec = build_missingness_spec_from_truth_payload(missingness_payload)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `build_missingness_spec_from_truth_payload`
- `get`
- `get_parent_truth_hash`
- `get_pipeline_mode_from_truth`
- `get_truth_hash`
- `info`
- `items`
- `load_truth_record_by_hash`
- `strip`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `if SILVER_PARENT_TRUTH_HASH is None or str(SILVER_PARENT_TRUTH_HASH).strip() == "": raise ValueError("Set SILVER_PARENT_TRUTH_HASH in the parameter cell.")` | Controls validation, iteration, file handling, or error handling for this step. |
| `silver_parent_truth = load_truth_record_by_hash( truth_dir=TRUTHS_PATH, layer_name=SILVER_PARENT_LAYER_NAME, dataset_name=DATASET_NAME, truth_hash=str(SILVER_PARENT_TRUTH_HASH).str` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Backward-compatible alias so existing downstream cells can keep working.` | Documents the purpose or boundary of the surrounding notebook step. |
| `silver_eda_truth = silver_parent_truth` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PARENT_TRUTH_HASH = get_truth_hash(silver_parent_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_PREEDA_TRUTH_HASH = get_parent_truth_hash(silver_parent_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `silver_preeda_truth = load_truth_record_by_hash( truth_dir=TRUTHS_PATH, layer_name="silver", dataset_name=DATASET_NAME, truth_hash=str(SILVER_PREEDA_TRUTH_HASH).strip(),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_payload = ( (silver_preeda_truth.get("runtime_facts", {}) or {}) .get("missingness_quarantine", {}) or {}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missingness_spec = build_missingness_spec_from_truth_payload(missingness_payload)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MISSINGNESS_SENSOR_OVERRIDES = {}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for sensor_name, override_pct in MISSINGNESS_SENSOR_OVERRIDES.items(): missingness_spec.missingness_pct_all[str(sensor_name)] = float(override_pct)` | Controls validation, iteration, file handling, or error handling for this step. |
| `parent_mode = get_pipeline_mode_from_truth(silver_parent_truth)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if parent_mode: PIPELINE_MODE = parent_mode` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("PARENT_TRUTH_HASH:", PARENT_TRUTH_HASH)` | Displays a notebook-facing result for inspection. |
| `print("PIPELINE_MODE:", PIPELINE_MODE)` | Displays a notebook-facing result for inspection. |
| `print("sensor_50 target missing pct:", missingness_spec.missingness_pct_all.get("sensor_50"))` | Displays a notebook-facing result for inspection. |
| `print("sensor_51 target missing pct:", missingness_spec.missingness_pct_all.get("sensor_51"))` | Displays a notebook-facing result for inspection. |
| `print("sensor_15 target missing pct:", missingness_spec.missingness_pct_all.get("sensor_15"))` | Displays a notebook-facing result for inspection. |
| `print("missingness target count:", len(missingness_spec.missingness_pct_all))` | Displays a notebook-facing result for inspection. |
| `logger.info("W&B PARENT_TRUTH_HASH: %s", PARENT_TRUTH_HASH)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("W&B PIPELINE_MODE: %s", PIPELINE_MODE)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("sensor_50 target missing pct: %s", missingness_spec.missingness_pct_all.get("sensor_50"))` | Writes a logger message for traceability during notebook execution. |
| `logger.info("sensor_51 target missing pct: %s", missingness_spec.missingness_pct_all.get("sensor_51"))` | Writes a logger message for traceability during notebook execution. |
| `logger.info("sensor_15 target missing pct: %s", missingness_spec.missingness_pct_all.get("sensor_15"))` | Writes a logger message for traceability during notebook execution. |
| `logger.info("missingness target count: %s", len(missingness_spec.missingness_pct_all))` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 15 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `are`
- `artifact`
- `ARTIFACT_CLUSTER_COUNT`
- `but`
- `clean`
- `cluster`
- `clusters`
- `contains`
- `correlation_cluster_derivation`
- `CORRELATION_HOTSPOT_CLUSTERS`
- `Could`
- `derive`
- `elif`
- `else`
- `empty`
- `encoding`
- `error`
- `exc`
- `Exception`
- `exists`

### Outputs

- `artifact_clusters`
- `HOTSPOT_CLUSTER_ARTIFACT_PATH`
- `HOTSPOT_CLUSTERS_FOR_GENERATOR`
- `hotspot_key`
- `hotspot_payload`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Resolve clean-normal hotspot clusters for generator`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Priority:`: Documents the purpose or boundary of the surrounding notebook step.
- `# 1. Use non-empty hotspot artifact clusters if provided.`: Documents the purpose or boundary of the surrounding notebook step.
- `# 2. If artifact exists but is empty, keep YAML clusters.`: Documents the purpose or boundary of the surrounding notebook step.
- `# 3. If YAML clusters are empty too, let SyntheticGenerator derive clusters`: Documents the purpose or boundary of the surrounding notebook step.
- `# from correlation_cluster_derivation.`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `HOTSPOT_CLUSTER_ARTIFACT_PATH = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `HOTSPOT_CLUSTERS_FOR_GENERATOR = list(CORRELATION_HOTSPOT_CLUSTERS or [])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `hotspot_key = ( SYN_CFG.get("silver_eda_artifact_keys", {}) or {}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get`
- `get_artifact_path_from_truth`
- `isinstance`
- `loads`
- `Path`
- `read_text`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Resolve clean-normal hotspot clusters for generator` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Priority:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 1. Use non-empty hotspot artifact clusters if provided.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 2. If artifact exists but is empty, keep YAML clusters.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 3. If YAML clusters are empty too, let SyntheticGenerator derive clusters` | Documents the purpose or boundary of the surrounding notebook step. |
| `# from correlation_cluster_derivation.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `HOTSPOT_CLUSTER_ARTIFACT_PATH = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `HOTSPOT_CLUSTERS_FOR_GENERATOR = list(CORRELATION_HOTSPOT_CLUSTERS or [])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `hotspot_key = ( SYN_CFG.get("silver_eda_artifact_keys", {}) or {}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).get("hotspot_clusters_normal")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if hotspot_key: try: HOTSPOT_CLUSTER_ARTIFACT_PATH = get_artifact_path_from_truth( silver_parent_truth, hotspot_key, ) except Exception as exc: HOTSPOT_CLUSTER_ARTIFACT_PATH = None` | Displays a notebook-facing result for inspection. |
| `artifact_clusters = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if HOTSPOT_CLUSTER_ARTIFACT_PATH: hotspot_payload = json.loads( Path(HOTSPOT_CLUSTER_ARTIFACT_PATH).read_text(encoding="utf-8") ) if isinstance(hotspot_payload, dict) and "clusters` | Loads input data, configuration, or artifacts required by the current stage. |
| `print("HOTSPOT_CLUSTER_ARTIFACT_PATH:", HOTSPOT_CLUSTER_ARTIFACT_PATH)` | Displays a notebook-facing result for inspection. |
| `print("YAML_CLUSTER_COUNT:", len(CORRELATION_HOTSPOT_CLUSTERS or []))` | Displays a notebook-facing result for inspection. |
| `print("ARTIFACT_CLUSTER_COUNT:", len(artifact_clusters))` | Displays a notebook-facing result for inspection. |
| `print("HOTSPOT_CLUSTERS_FOR_GENERATOR:", HOTSPOT_CLUSTERS_FOR_GENERATOR)` | Displays a notebook-facing result for inspection. |
| `print("HOTSPOT_CLUSTER_COUNT:", len(HOTSPOT_CLUSTERS_FOR_GENERATOR))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 16 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `missingness_payload`

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

## Code Cell 17 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `count`
- `get`
- `missing`
- `missingness`
- `missingness_pct_all`
- `missingness_spec`
- `pct`
- `sensor_15`
- `sensor_50`
- `sensor_51`
- `target`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `print("sensor_50 target missing pct:", missingness_spec.missingness_pct_all.get("sensor_50"))`: Displays a notebook-facing result for inspection.
- `print("sensor_51 target missing pct:", missingness_spec.missingness_pct_all.get("sensor_51"))`: Displays a notebook-facing result for inspection.
- `print("sensor_15 target missing pct:", missingness_spec.missingness_pct_all.get("sensor_15"))`: Displays a notebook-facing result for inspection.
- `print("missingness target count:", len(missingness_spec.missingness_pct_all))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `get`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print("sensor_50 target missing pct:", missingness_spec.missingness_pct_all.get("sensor_50"))` | Displays a notebook-facing result for inspection. |
| `print("sensor_51 target missing pct:", missingness_spec.missingness_pct_all.get("sensor_51"))` | Displays a notebook-facing result for inspection. |
| `print("sensor_15 target missing pct:", missingness_spec.missingness_pct_all.get("sensor_15"))` | Displays a notebook-facing result for inspection. |
| `print("missingness target count:", len(missingness_spec.missingness_pct_all))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 18 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `corr_pairs_normal`
- `fault_pairings_normal`
- `get_artifact_path_from_truth`
- `group_map_normal`
- `info`
- `logger`
- `profile_abnormal`
- `profile_normal`
- `profile_recovery`
- `s`
- `silver_eda_artifact_keys`
- `silver_eda_truth`
- `SYN_CFG`

### Outputs

- `corr_pairs_normal_path`
- `fault_pairings_normal_path`
- `group_map_normal_path`
- `keys`
- `profile_abnormal_path`
- `profile_normal_path`
- `profile_recovery_path`

### Key Operations

- `keys = SYN_CFG["silver_eda_artifact_keys"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `profile_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_normal"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `profile_abnormal_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_abnormal"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `profile_recovery_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_recovery"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `corr_pairs_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["corr_pairs_normal"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `group_map_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["group_map_normal"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `fault_pairings_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["fault_pairings_normal"])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print(profile_normal_path)`: Displays a notebook-facing result for inspection.
- `print(profile_abnormal_path)`: Displays a notebook-facing result for inspection.
- `print(profile_recovery_path)`: Displays a notebook-facing result for inspection.
- `print(corr_pairs_normal_path)`: Displays a notebook-facing result for inspection.
- `print(group_map_normal_path)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `get_artifact_path_from_truth`
- `info`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `keys = SYN_CFG["silver_eda_artifact_keys"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `profile_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_normal"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `profile_abnormal_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_abnormal"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `profile_recovery_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_recovery"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `corr_pairs_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["corr_pairs_normal"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `group_map_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["group_map_normal"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `fault_pairings_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["fault_pairings_normal"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print(profile_normal_path)` | Displays a notebook-facing result for inspection. |
| `print(profile_abnormal_path)` | Displays a notebook-facing result for inspection. |
| `print(profile_recovery_path)` | Displays a notebook-facing result for inspection. |
| `print(corr_pairs_normal_path)` | Displays a notebook-facing result for inspection. |
| `print(group_map_normal_path)` | Displays a notebook-facing result for inspection. |
| `print(fault_pairings_normal_path)` | Displays a notebook-facing result for inspection. |
| `logger.info("silver_eda_artifact_keys: %s", keys)` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 19 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__`
- `__episode_status_counts`
- `a`
- `already`
- `artifact`
- `ARTIFACTS_ROOT`
- `be`
- `because`
- `but`
- `candidates`
- `Could`
- `counts`
- `DATASET_NAME`
- `def`
- `defines`
- `diagnostics`
- `else`
- `encoding`
- `episode`
- `episode_status_counts`

### Outputs

- `data`
- `EPISODE_STATUS_JSON_PATH`
- `episode_status_rows`
- `HAS_EPISODE_STATUS_COUNTS`
- `label`
- `load_episode_status_counts_json`
- `p`
- `resolve_optional_existing_path_from_candidates`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Optional episode status counts resolver`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# This artifact is useful for diagnostics, but it is not required`: Documents the purpose or boundary of the surrounding notebook step.
- `# for synthetic generation because the YAML already defines the`: Documents the purpose or boundary of the surrounding notebook step.
- `# episode ranges and rows_per_failure settings.`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `def load_episode_status_counts_json(path: str \| Path) -> list[dict]: p = Path(path) if not p.exists(): raise FileNotFoundError(f"episode_status_counts.json not found: {p}") data = `: Defines notebook-local logic used later in the notebook.
- `def resolve_optional_existing_path_from_candidates( candidates: list[Path], *, label: str,`: Defines notebook-local logic used later in the notebook.
- `) -> Path \| None: for path in candidates: if path.exists(): return path print( f"WARNING: Could not resolve optional {label}. Tried:\n" + "\n".join(str(path) for path in candidates`: Displays a notebook-facing result for inspection.
- `EPISODE_STATUS_JSON_PATH = resolve_optional_existing_path_from_candidates( [ ARTIFACTS_ROOT / SILVER_PARENT_LAYER_NAME / DATASET_NAME / f"{DATASET_NAME}__{SILVER_PARENT_LAYER_NAME}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `exists`
- `FileNotFoundError`
- `isinstance`
- `join`
- `load_episode_status_counts_json`
- `loads`
- `Path`
- `read_text`
- `resolve_optional_existing_path_from_candidates`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Optional episode status counts resolver` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# This artifact is useful for diagnostics, but it is not required` | Documents the purpose or boundary of the surrounding notebook step. |
| `# for synthetic generation because the YAML already defines the` | Documents the purpose or boundary of the surrounding notebook step. |
| `# episode ranges and rows_per_failure settings.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `def load_episode_status_counts_json(path: str \| Path) -> list[dict]: p = Path(path) if not p.exists(): raise FileNotFoundError(f"episode_status_counts.json not found: {p}") data = ` | Defines notebook-local logic used later in the notebook. |
| `def resolve_optional_existing_path_from_candidates( candidates: list[Path], *, label: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> Path \| None: for path in candidates: if path.exists(): return path print( f"WARNING: Could not resolve optional {label}. Tried:\n" + "\n".join(str(path) for path in candidates` | Displays a notebook-facing result for inspection. |
| `EPISODE_STATUS_JSON_PATH = resolve_optional_existing_path_from_candidates( [ ARTIFACTS_ROOT / SILVER_PARENT_LAYER_NAME / DATASET_NAME / f"{DATASET_NAME}__{SILVER_PARENT_LAYER_NAME}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if EPISODE_STATUS_JSON_PATH is not None: episode_status_rows = load_episode_status_counts_json(EPISODE_STATUS_JSON_PATH) HAS_EPISODE_STATUS_COUNTS = True` | Loads input data, configuration, or artifacts required by the current stage. |
| `else: episode_status_rows = [] HAS_EPISODE_STATUS_COUNTS = False` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("EPISODE_STATUS_JSON_PATH:", EPISODE_STATUS_JSON_PATH)` | Displays a notebook-facing result for inspection. |
| `print("HAS_EPISODE_STATUS_COUNTS:", HAS_EPISODE_STATUS_COUNTS)` | Displays a notebook-facing result for inspection. |
| `print("Loaded episodes:", len(episode_status_rows))` | Displays a notebook-facing result for inspection. |
| `print("First row:", episode_status_rows[0] if episode_status_rows else None)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 20 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `astype`
- `coerce`
- `columns`
- `copy`
- `DataFrame`
- `def`
- `dtype`
- `episode_status_rows`
- `episode_total_median`
- `episode_total_p10`
- `episode_total_p90`
- `episode_total_rows`
- `episodes_n`
- `errors`
- `failure`
- `failure_rows_default`
- `failure_rows_max`
- `fillna`
- `get`

### Outputs

- `abnormal_series`
- `df`
- `failure_series`
- `normal_pct`
- `recovery_pct`
- `summarize_episode_status_rows`
- `total`
- `total_nonzero`
- `zero_series`

### Key Operations

- `d`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `copy`
- `DataFrame`
- `fillna`
- `get`
- `max`
- `maximum`
- `mean`
- `median`
- `nanmedian`
- `nanpercentile`
- `replace`
- `Series`
- `summarize_episode_status_rows`
- `to_numeric`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `d` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 21 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `adjust`
- `after`
- `becomes`
- `before`
- `bool`
- `buildup_fraction_of_normal`
- `choice`
- `choose`
- `clamp`
- `dataset`
- `def`
- `derive`
- `distribution`
- `else`
- `empty`
- `enforce`
- `episode`
- `episode_status_rows`
- `episode_total_rows`

### Outputs

- `buildup`
- `choose_episode_phase_lengths`
- `episode_total`
- `episode_total_check`
- `failure`
- `na`
- `nb`
- `normal_total`
- `real_total`
- `rec`
- `rec_pct`
- `remaining`
- `shift`
- `template`
- `totals`

### Key Operations

- `def choose_episode_phase_lengths( rng: np.random.Generator, *, episode_status_rows: list[dict], # knobs failure_rows_default: int = 1, # how much of "normal" becomes buildup buildu`: Defines notebook-local logic used later in the notebook.
- `) -> dict: """ Returns dict with: normal_before, buildup, failure, recovery, normal_after, episode_total """ if not episode_status_rows: raise ValueError("episode_status_rows is em`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `choice`
- `choose_episode_phase_lengths`
- `failure`
- `get`
- `integers`
- `max`
- `min`
- `round`
- `totals`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def choose_episode_phase_lengths( rng: np.random.Generator, *, episode_status_rows: list[dict], # knobs failure_rows_default: int = 1, # how much of "normal" becomes buildup buildu` | Defines notebook-local logic used later in the notebook. |
| `) -> dict: """ Returns dict with: normal_before, buildup, failure, recovery, normal_after, episode_total """ if not episode_status_rows: raise ValueError("episode_status_rows is em` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 22 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__`
- `__dropped_feature_profiles__abnormal`
- `__dropped_feature_profiles__normal`
- `__dropped_feature_profiles__normal_clean`
- `__dropped_feature_profiles__recovery`
- `__silver_eda__dropped_feature_profiles__abnormal`
- `__silver_eda__dropped_feature_profiles__normal`
- `__silver_eda__dropped_feature_profiles__recovery`
- `abnormal`
- `artifact`
- `artifacts`
- `base`
- `build`
- `Calibration`
- `CALIBRATION_ENABLED`
- `CALIBRATION_MEAN_WITHIN_K_STD`
- `CALIBRATION_STD_RATIO_BOUNDS`
- `candidates`
- `cluster`
- `clusters`

### Outputs

- `_to_target_dict`
- `abnormal_profiles`
- `base_profile_csv_path`
- `build_state_calibration_targets_from_profile_dicts`
- `corr_pairs_df`
- `correlation_cluster_derivation`
- `correlation_hotspot_clusters`
- `correlation_pairs_dataframe`
- `correlation_tuning`
- `dropped_profile_abnormal_path`
- `dropped_profile_csv_path`
- `dropped_profile_normal_path`
- `dropped_profile_recovery_path`
- `fault_excluded_sensors`
- `fault_pairings_dataframe`
- `fault_pairings_df`
- `generator`
- `generator_inputs_dir`
- `group_map_dataframe`
- `group_map_df`

### Key Operations

- `# --- Load profiles (base + dropped) and build generator ---`: Documents the purpose or boundary of the surrounding notebook step.
- `# profile_normal_path should point to:`: Documents the purpose or boundary of the surrounding notebook step.
- `# artifacts/silver_subsets/pump/generator_inputs/feature_profile_normal_clean.csv`: Documents the purpose or boundary of the surrounding notebook step.
- `generator_inputs_dir = Path(profile_normal_path).parent`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# Parent artifact folder:`: Documents the purpose or boundary of the surrounding notebook step.
- `# artifacts/silver_subsets/pump`: Documents the purpose or boundary of the surrounding notebook step.
- `silver_parent_artifact_dir = generator_inputs_dir.parent`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `def resolve_optional_existing_path(candidates: list[Path]) -> str \| None: for path in candidates: if path.exists(): return str(path) return None`: Defines notebook-local logic used later in the notebook.
- `dropped_profile_normal_path = resolve_optional_existing_path( [ generator_inputs_dir / "dropped_feature_profiles__normal_clean.csv", generator_inputs_dir / "dropped_feature_profile`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `dropped_profile_abnormal_path = resolve_optional_existing_path( [ generator_inputs_dir / "dropped_feature_profiles__abnormal.csv", silver_parent_artifact_dir / f"{DATASET_NAME}__{S`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `_to_target_dict`
- `build_state_calibration_targets_from_profile_dicts`
- `exists`
- `globals`
- `info`
- `items`
- `load_and_merge_rich_profiles`
- `load_correlation_pairs_csv`
- `load_fault_pairings_csv`
- `load_group_map_csv`
- `notna`
- `Path`
- `profiles`
- `resolve_optional_existing_path`
- `sorted`
- `SyntheticGenerator`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# --- Load profiles (base + dropped) and build generator ---` | Documents the purpose or boundary of the surrounding notebook step. |
| `# profile_normal_path should point to:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# artifacts/silver_subsets/pump/generator_inputs/feature_profile_normal_clean.csv` | Documents the purpose or boundary of the surrounding notebook step. |
| `generator_inputs_dir = Path(profile_normal_path).parent` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# Parent artifact folder:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# artifacts/silver_subsets/pump` | Documents the purpose or boundary of the surrounding notebook step. |
| `silver_parent_artifact_dir = generator_inputs_dir.parent` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `def resolve_optional_existing_path(candidates: list[Path]) -> str \| None: for path in candidates: if path.exists(): return str(path) return None` | Defines notebook-local logic used later in the notebook. |
| `dropped_profile_normal_path = resolve_optional_existing_path( [ generator_inputs_dir / "dropped_feature_profiles__normal_clean.csv", generator_inputs_dir / "dropped_feature_profile` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `dropped_profile_abnormal_path = resolve_optional_existing_path( [ generator_inputs_dir / "dropped_feature_profiles__abnormal.csv", silver_parent_artifact_dir / f"{DATASET_NAME}__{S` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `dropped_profile_recovery_path = resolve_optional_existing_path( [ generator_inputs_dir / "dropped_feature_profiles__recovery.csv", silver_parent_artifact_dir / f"{DATASET_NAME}__{S` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("generator_inputs_dir:", generator_inputs_dir)` | Displays a notebook-facing result for inspection. |
| `print("silver_parent_artifact_dir:", silver_parent_artifact_dir)` | Displays a notebook-facing result for inspection. |
| `print("Dropped profile normal path:", dropped_profile_normal_path)` | Displays a notebook-facing result for inspection. |
| `print("Dropped profile abnormal path:", dropped_profile_abnormal_path)` | Displays a notebook-facing result for inspection. |
| `print("Dropped profile recovery path:", dropped_profile_recovery_path)` | Displays a notebook-facing result for inspection. |
| `print( "Dropped profile files found:", dropped_profile_normal_path is not None and Path(dropped_profile_normal_path).exists(), dropped_profile_abnormal_path is not None and Path(dr` | Displays a notebook-facing result for inspection. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `normal_profiles = load_and_merge_rich_profiles( base_profile_csv_path=str(profile_normal_path), state_scope="normal", dropped_profile_csv_path=dropped_profile_normal_path,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `abnormal_profiles = load_and_merge_rich_profiles( base_profile_csv_path=str(profile_abnormal_path), state_scope="abnormal", dropped_profile_csv_path=dropped_profile_abnormal_path,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `recovery_profiles = load_and_merge_rich_profiles( base_profile_csv_path=str(profile_recovery_path), state_scope="recovery", dropped_profile_csv_path=dropped_profile_recovery_path,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `corr_pairs_df = load_correlation_pairs_csv(corr_pairs_normal_path)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `group_map_df = load_group_map_csv(group_map_normal_path)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `fault_pairings_df = load_fault_pairings_csv(fault_pairings_normal_path)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `def build_state_calibration_targets_from_profile_dicts( *, normal_profiles: dict, abnormal_profiles: dict, recovery_profiles: dict,` | Defines notebook-local logic used later in the notebook. |
| `) -> dict: def _to_target_dict(profile_dict: dict) -> dict: out = {} for sensor, prof in profile_dict.items(): out[str(sensor)] = { "mean": float(prof.mean), "std": float(prof.std)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `state_calibration_targets = ( build_state_calibration_targets_from_profile_dicts( normal_profiles=normal_profiles, abnormal_profiles=abnormal_profiles, recovery_profiles=recovery_p` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `generator = SyntheticGenerator( normal_profiles=normal_profiles, abnormal_profiles=abnormal_profiles, recovery_profiles=recovery_profiles, correlation_pairs_dataframe=corr_pairs_df` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Sensors:", len(generator.sensors))` | Displays a notebook-facing result for inspection. |
| `print("First sensors:", generator.sensors[:10])` | Displays a notebook-facing result for inspection. |
| `print("Calibration enabled:", CALIBRATION_ENABLED)` | Displays a notebook-facing result for inspection. |
| `print("Calibration mean_within_k_std:", CALIBRATION_MEAN_WITHIN_K_STD)` | Displays a notebook-facing result for inspection. |
| `print("Calibration std_ratio_bounds:", CALIBRATION_STD_RATIO_BOUNDS)` | Displays a notebook-facing result for inspection. |
| `print("Resolved hotspot clusters:", generator.correlation_hotspot_clusters)` | Displays a notebook-facing result for inspection. |
| `print("Resolved hotspot cluster count:", len(generator.correlation_hotspot_clusters))` | Displays a notebook-facing result for inspection. |
| `print("Fault-eligible sensors:", len([s for s in generator.sensors if s not in generator.fault_excluded_sensors]))` | Displays a notebook-facing result for inspection. |
| `print("Excluded sensors:", sorted(generator.fault_excluded_sensors))` | Displays a notebook-facing result for inspection. |
| `print("Correlation tuning:", generator.correlation_tuning)` | Displays a notebook-facing result for inspection. |
| `logger.info("Generator Run")` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Generator Sensors Count: %s", len(generator.sensors))` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Generator Sensors List: %s", generator.sensors)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Calibration Enabled: %s", CALIBRATION_ENABLED)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Calibration mean_within_k_std: %s", CALIBRATION_MEAN_WITHIN_K_STD)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Calibration std_ratio_bounds: %s", CALIBRATION_STD_RATIO_BOUNDS)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Resolved hotspot clusters: %s", generator.correlation_hotspot_clusters)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Fault excluded sensors: %s", sorted(generator.fault_excluded_sensors))` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Correlation tuning: %s", generator.correlation_tuning)` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 23 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `cluster`
- `clusters`
- `correlation_hotspot_clusters`
- `count`
- `generator`
- `hotspot`
- `Resolved`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `print("Resolved hotspot clusters:", generator.correlation_hotspot_clusters)`: Displays a notebook-facing result for inspection.
- `print("Resolved hotspot cluster count:", len(generator.correlation_hotspot_clusters))`: Displays a notebook-facing result for inspection.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print("Resolved hotspot clusters:", generator.correlation_hotspot_clusters)` | Displays a notebook-facing result for inspection. |
| `print("Resolved hotspot cluster count:", len(generator.correlation_hotspot_clusters))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 24 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal_profiles`
- `count`
- `Diagnose`
- `dropped`
- `expected`
- `f`
- `generator`
- `Generator`
- `i`
- `merge`
- `missing`
- `n`
- `nGenerator`
- `normal_profiles`
- `profile`
- `range`
- `recovery_profiles`
- `s`
- `sensor`
- `sensor_`

### Outputs

- `expected_sensor_columns`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Diagnose dropped sensor profile merge`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `for sensor_name in ["sensor_15", "sensor_50"]: print(f"\n{sensor_name}") print(" in normal_profiles:", sensor_name in normal_profiles) print(" in abnormal_profiles:", sensor_name i`: Displays a notebook-facing result for inspection.
- `print("\nGenerator sensor count:", len(generator.sensors))`: Displays a notebook-facing result for inspection.
- `print("Generator sensors missing from expected 52:")`: Displays a notebook-facing result for inspection.
- `expected_sensor_columns = [f"sensor_{i:02d}" for i in range(52)]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print([s for s in expected_sensor_columns if s not in generator.sensors])`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `range`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Diagnose dropped sensor profile merge` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `for sensor_name in ["sensor_15", "sensor_50"]: print(f"\n{sensor_name}") print(" in normal_profiles:", sensor_name in normal_profiles) print(" in abnormal_profiles:", sensor_name i` | Displays a notebook-facing result for inspection. |
| `print("\nGenerator sensor count:", len(generator.sensors))` | Displays a notebook-facing result for inspection. |
| `print("Generator sensors missing from expected 52:")` | Displays a notebook-facing result for inspection. |
| `expected_sensor_columns = [f"sensor_{i:02d}" for i in range(52)]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print([s for s in expected_sensor_columns if s not in generator.sensors])` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 25 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `abnormal_profiles`
- `AFTER`
- `all`
- `assert`
- `astype`
- `be`
- `been`
- `before`
- `build`
- `cell`
- `check`
- `CHECK`
- `continue`
- `count`
- `created`
- `dictionary`
- `Dropped`
- `dropped`
- `dropped_profile_abnormal_path`

### Outputs

- `dropped_df`
- `expected_generated_dropped_sensors`
- `expected_restored_only_sensors`
- `expected_sensor_columns`
- `missing_objects`
- `required_objects`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# HARD CHECK: dropped sensor profile merge before generation`: Documents the purpose or boundary of the surrounding notebook step.
- `# Run this AFTER normal_profiles / abnormal_profiles /`: Documents the purpose or boundary of the surrounding notebook step.
- `# recovery_profiles and generator have been created.`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `required_objects = [ "normal_profiles", "abnormal_profiles", "recovery_profiles", "generator",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `missing_objects = [ name for name in required_objects if name not in globals()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if missing_objects: raise NameError( "Run the generator-build cell before this check. " f"Missing objects: {missing_objects}" )`: Controls validation, iteration, file handling, or error handling for this step.
- `expected_generated_dropped_sensors = ["sensor_50"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `expected_restored_only_sensors = ["sensor_15"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `astype`
- `globals`
- `items`
- `NameError`
- `range`
- `read_csv`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# HARD CHECK: dropped sensor profile merge before generation` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Run this AFTER normal_profiles / abnormal_profiles /` | Documents the purpose or boundary of the surrounding notebook step. |
| `# recovery_profiles and generator have been created.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `required_objects = [ "normal_profiles", "abnormal_profiles", "recovery_profiles", "generator",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `missing_objects = [ name for name in required_objects if name not in globals()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if missing_objects: raise NameError( "Run the generator-build cell before this check. " f"Missing objects: {missing_objects}" )` | Controls validation, iteration, file handling, or error handling for this step. |
| `expected_generated_dropped_sensors = ["sensor_50"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `expected_restored_only_sensors = ["sensor_15"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Dropped profile normal path:", dropped_profile_normal_path)` | Displays a notebook-facing result for inspection. |
| `print("Dropped profile abnormal path:", dropped_profile_abnormal_path)` | Displays a notebook-facing result for inspection. |
| `print("Dropped profile recovery path:", dropped_profile_recovery_path)` | Displays a notebook-facing result for inspection. |
| `for label, path in { "normal_clean": dropped_profile_normal_path, "abnormal": dropped_profile_abnormal_path, "recovery": dropped_profile_recovery_path,` | Controls validation, iteration, file handling, or error handling for this step. |
| `}.items(): print(f"\n{label} dropped profile path:", path) if path is None: print(" MISSING PATH") continue dropped_df = pd.read_csv(path) print(" shape:", dropped_df.shape) print(` | Loads input data, configuration, or artifacts required by the current stage. |
| `print("\nProfile dictionary membership:")` | Displays a notebook-facing result for inspection. |
| `for sensor_name in ["sensor_15", "sensor_50"]: print(sensor_name) print(" in normal_profiles:", sensor_name in normal_profiles) print(" in abnormal_profiles:", sensor_name in abnor` | Displays a notebook-facing result for inspection. |
| `print("\nGenerator sensor count:", len(generator.sensors))` | Displays a notebook-facing result for inspection. |
| `expected_sensor_columns = [f"sensor_{i:02d}" for i in range(52)]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Missing from generator.sensors:")` | Displays a notebook-facing result for inspection. |
| `print([sensor for sensor in expected_sensor_columns if sensor not in generator.sensors])` | Displays a notebook-facing result for inspection. |
| `assert "sensor_50" in normal_profiles, "sensor_50 is missing from normal_profiles."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `assert "sensor_50" in generator.sensors, "sensor_50 is missing from generator.sensors."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `assert "sensor_15" not in generator.sensors, ( "sensor_15 should not be generated. It should be restored later as all-null."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `assert len(generator.sensors) == 51, ( f"Expected 51 generated sensors: 50 normal feature sensors + sensor_50. " f"Got {len(generator.sensors)}."` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("\nPASS: sensor_50 is generated; sensor_15 will be restored as all-null later.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 26 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `astype`
- `COLUMN`
- `columns`
- `continue`
- `dropped`
- `dropped_profile_abnormal_path`
- `dropped_profile_normal_path`
- `dropped_profile_recovery_path`
- `else`
- `files`
- `has`
- `Inspect`
- `items`
- `label`
- `MISSING`
- `n`
- `NO`
- `normal_clean`
- `profile`

### Outputs

- `df`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Inspect dropped profile files`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `for label, path in { "normal_clean": dropped_profile_normal_path, "abnormal": dropped_profile_abnormal_path, "recovery": dropped_profile_recovery_path,`: Controls validation, iteration, file handling, or error handling for this step.
- `}.items(): print("\n", label, path) if path is None: print(" MISSING PATH") continue df = pd.read_csv(path) print(" shape:", df.shape) print(" sensors:", df["sensor"].astype(str).t`: Loads input data, configuration, or artifacts required by the current stage.

Important functions or methods detected:
- `astype`
- `items`
- `read_csv`
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Inspect dropped profile files` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `for label, path in { "normal_clean": dropped_profile_normal_path, "abnormal": dropped_profile_abnormal_path, "recovery": dropped_profile_recovery_path,` | Controls validation, iteration, file handling, or error handling for this step. |
| `}.items(): print("\n", label, path) if path is None: print(" MISSING PATH") continue df = pd.read_csv(path) print(" shape:", df.shape) print(" sensors:", df["sensor"].astype(str).t` | Loads input data, configuration, or artifacts required by the current stage. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 27 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `eligibility`
- `eligible`
- `Excluded`
- `exclusions`
- `Fault`
- `generator`
- `info`
- `logger`
- `resolved`
- `s`
- `sensors`
- `sorted`
- `the`
- `use`

### Outputs

- `FAULT_ELIGIBLE_SENSORS`
- `FAULT_EXCLUDED_SENSORS`

### Key Operations

- `# --- Fault eligibility: use the generator-resolved exclusions ---`: Documents the purpose or boundary of the surrounding notebook step.
- `FAULT_EXCLUDED_SENSORS = set(generator.fault_excluded_sensors)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `FAULT_ELIGIBLE_SENSORS = [s for s in generator.sensors if s not in FAULT_EXCLUDED_SENSORS]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("Fault-eligible sensors:", len(FAULT_ELIGIBLE_SENSORS))`: Displays a notebook-facing result for inspection.
- `print("Excluded sensors:", sorted(FAULT_EXCLUDED_SENSORS))`: Displays a notebook-facing result for inspection.
- `logger.info("Fault-eligible sensors: %s", len(FAULT_ELIGIBLE_SENSORS))`: Writes a logger message for traceability during notebook execution.
- `logger.info("Excluded sensors: %s", sorted(FAULT_EXCLUDED_SENSORS))`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `info`
- `sorted`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# --- Fault eligibility: use the generator-resolved exclusions ---` | Documents the purpose or boundary of the surrounding notebook step. |
| `FAULT_EXCLUDED_SENSORS = set(generator.fault_excluded_sensors)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FAULT_ELIGIBLE_SENSORS = [s for s in generator.sensors if s not in FAULT_EXCLUDED_SENSORS]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Fault-eligible sensors:", len(FAULT_ELIGIBLE_SENSORS))` | Displays a notebook-facing result for inspection. |
| `print("Excluded sensors:", sorted(FAULT_EXCLUDED_SENSORS))` | Displays a notebook-facing result for inspection. |
| `logger.info("Fault-eligible sensors: %s", len(FAULT_ELIGIBLE_SENSORS))` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Excluded sensors: %s", sorted(FAULT_EXCLUDED_SENSORS))` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 28 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__`
- `__episode_status_counts`
- `__silver_eda__episode_status_counts`
- `a`
- `already`
- `artifacts`
- `available`
- `candidates`
- `composition`
- `contain`
- `Continuing`
- `Could`
- `count`
- `DATASET_NAME`
- `def`
- `else`
- `encoding`
- `episode`
- `Episode`
- `episode_status_counts`

### Outputs

- `EPISODE_STATS`
- `episode_stats_path`
- `EPISODE_STATS_PATH_EXISTS`
- `generator_inputs_dir`
- `label`
- `resolve_optional_existing_path_from_candidates`
- `silver_parent_artifact_dir`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Load Silver episode composition stats if available`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Preferred location:`: Documents the purpose or boundary of the surrounding notebook step.
- `# artifacts/silver_subsets/pump/generator_inputs/episode_status_counts.json`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `def resolve_optional_existing_path_from_candidates( candidates: list[Path], *, label: str,`: Defines notebook-local logic used later in the notebook.
- `) -> Path \| None: for path in candidates: if path.exists(): return path print( f"WARNING: Could not resolve optional {label}. Tried:\n" + "\n".join(str(path) for path in candidates`: Displays a notebook-facing result for inspection.
- `# profile_dir should already point to:`: Documents the purpose or boundary of the surrounding notebook step.
- `# artifacts/silver_subsets/pump/generator_inputs`: Documents the purpose or boundary of the surrounding notebook step.
- `generator_inputs_dir = Path(profile_normal_path).parent`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `silver_parent_artifact_dir = generator_inputs_dir.parent`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `exists`
- `isinstance`
- `join`
- `load`
- `open`
- `Path`
- `resolve_optional_existing_path_from_candidates`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Load Silver episode composition stats if available` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Preferred location:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# artifacts/silver_subsets/pump/generator_inputs/episode_status_counts.json` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `def resolve_optional_existing_path_from_candidates( candidates: list[Path], *, label: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> Path \| None: for path in candidates: if path.exists(): return path print( f"WARNING: Could not resolve optional {label}. Tried:\n" + "\n".join(str(path) for path in candidates` | Displays a notebook-facing result for inspection. |
| `# profile_dir should already point to:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# artifacts/silver_subsets/pump/generator_inputs` | Documents the purpose or boundary of the surrounding notebook step. |
| `generator_inputs_dir = Path(profile_normal_path).parent` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `silver_parent_artifact_dir = generator_inputs_dir.parent` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `episode_stats_path = resolve_optional_existing_path_from_candidates( [ generator_inputs_dir / "episode_status_counts.json", silver_parent_artifact_dir / f"{DATASET_NAME}__{SILVER_P` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `EPISODE_STATS = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `EPISODE_STATS_PATH_EXISTS = episode_stats_path is not None and episode_stats_path.exists()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if EPISODE_STATS_PATH_EXISTS: with open(episode_stats_path, "r", encoding="utf-8") as f: EPISODE_STATS = json.load(f) if not isinstance(EPISODE_STATS, list): raise ValueError( f"Ep` | Displays a notebook-facing result for inspection. |
| `else: EPISODE_STATS = [] print("WARNING: Episode status stats unavailable. Continuing with YAML episode ranges.") print("episode_stats_path:", episode_stats_path)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 29 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `copy`
- `DataFrame`
- `else`
- `empty`
- `EPISODE_STATS`
- `median`
- `normal`
- `quantile`
- `recovery`

### Outputs

- `EPISODE_STATS_DF`
- `NORMAL_ROWS_MEDIAN`
- `RECOVERY_ROWS_MEDIAN`
- `RECOVERY_ROWS_Q75`

### Key Operations

- `EPISODE_STATS_DF = pd.DataFrame(EPISODE_STATS).copy() if EPISODE_STATS else pd.DataFrame()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `if not EPISODE_STATS_DF.empty: RECOVERY_ROWS_MEDIAN = int(EPISODE_STATS_DF["recovery"].median()) RECOVERY_ROWS_Q75 = int(EPISODE_STATS_DF["recovery"].quantile(0.75)) NORMAL_ROWS_ME`: Displays a notebook-facing result for inspection.
- `else: RECOVERY_ROWS_MEDIAN = 0 RECOVERY_ROWS_Q75 = 0 NORMAL_ROWS_MEDIAN = 0`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `copy`
- `DataFrame`
- `median`
- `quantile`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `EPISODE_STATS_DF = pd.DataFrame(EPISODE_STATS).copy() if EPISODE_STATS else pd.DataFrame()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not EPISODE_STATS_DF.empty: RECOVERY_ROWS_MEDIAN = int(EPISODE_STATS_DF["recovery"].median()) RECOVERY_ROWS_Q75 = int(EPISODE_STATS_DF["recovery"].quantile(0.75)) NORMAL_ROWS_ME` | Displays a notebook-facing result for inspection. |
| `else: RECOVERY_ROWS_MEDIAN = 0 RECOVERY_ROWS_Q75 = 0 NORMAL_ROWS_MEDIAN = 0` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 30 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `available`
- `batch`
- `before`
- `behavior`
- `copy`
- `DataFrame`
- `else`
- `episode`
- `Episode`
- `EPISODE_STATS`
- `EPISODE_STATS_PATH_EXISTS`
- `episode_total_rows`
- `failure`
- `fallback`
- `generation`
- `generator`
- `head`
- `max`
- `mean`
- `meta__episode_id`

### Outputs

- `episode_stats_df`

### Key Operations

- `# --- Validate episode stats payload before batch generation ---`: Documents the purpose or boundary of the surrounding notebook step.
- `if EPISODE_STATS_PATH_EXISTS: display( pd.DataFrame(EPISODE_STATS)[ ["meta__episode_id", "normal", "failure", "recovery", "episode_total_rows"] ].head(10) ) episode_stats_df = pd.D`: Displays a notebook-facing result for inspection.
- `else: print("EPISODE_STATS not available; batch generator will use fallback recovery behavior.")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `copy`
- `DataFrame`
- `display`
- `head`
- `max`
- `mean`
- `min`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# --- Validate episode stats payload before batch generation ---` | Documents the purpose or boundary of the surrounding notebook step. |
| `if EPISODE_STATS_PATH_EXISTS: display( pd.DataFrame(EPISODE_STATS)[ ["meta__episode_id", "normal", "failure", "recovery", "episode_total_rows"] ].head(10) ) episode_stats_df = pd.D` | Displays a notebook-facing result for inspection. |
| `else: print("EPISODE_STATS not available; batch generator will use fallback recovery behavior.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 31 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__init__`
- `Generator`
- `info`
- `inspect`
- `Inspection`
- `logger`
- `s`
- `Signature`
- `signature`
- `SyntheticGenerator`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `print(inspect.signature(SyntheticGenerator.__init__))`: Displays a notebook-facing result for inspection.
- `logger.info("Generator Signature Inspection: %s", inspect.signature(SyntheticGenerator.__init__))`: Writes a logger message for traceability during notebook execution.

Important functions or methods detected:
- `info`
- `signature`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print(inspect.signature(SyntheticGenerator.__init__))` | Displays a notebook-facing result for inspection. |
| `logger.info("Generator Signature Inspection: %s", inspect.signature(SyntheticGenerator.__init__))` | Writes a logger message for traceability during notebook execution. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.

## Code Cell 32 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `above`
- `accumulated`
- `actual`
- `after`
- `allowed`
- `available`
- `based`
- `Batch`
- `batch`
- `before`
- `Behavior`
- `below`
- `between`
- `bool`
- `broken`
- `BROKEN`
- `bu_max`
- `bu_min`
- `Build`

### Outputs

- `_episode_total_rows`
- `_failure_probability`
- `_is_failure_episode`
- `_rows_after_broken`
- `_rows_before_broken`
- `_safe_scalar_float`
- `_safe_scalar_int`
- `ALLOW_OVERSHOOT`
- `allowed_faults`
- `bu`
- `bu_candidate`
- `buildup`
- `buildup_fraction_of_normal`
- `candidate_bu`
- `candidate_nb`
- `candidate_rows_before_broken`
- `chosen`
- `cut_after`
- `cut_before`
- `effective_gap`

### Key Operations

- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# ---- Mode switch ----`: Documents the purpose or boundary of the surrounding notebook step.
- `MODE = "batch" # "single" \| "batch"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TARGET_ROWS = 72_000`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `MAX_EPISODES = 1_000_000 # safety cap`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# ---- policy knobs ----`: Documents the purpose or boundary of the surrounding notebook step.
- `EPISODE_MAX_ROWS = 3_000 # prevents monster episodes; forces multiple episodes in a 10k batch`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ALLOW_OVERSHOOT = False # if True, can overshoot when remaining can't fit minimum core`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# ---- failure rarity control (match real dataset frequency) ----`: Documents the purpose or boundary of the surrounding notebook step.
- `# Real dataset: ~7 failures per 250,000 rows => ~1 failure per 35,714 rows`: Documents the purpose or boundary of the surrounding notebook step.
- `ROWS_PER_FAILURE = 250_000 / 7 # ~35714.2857`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `_episode_total_rows`
- `_failure_probability`
- `_is_failure_episode`
- `_rows_after_broken`
- `_rows_before_broken`
- `_safe_scalar_float`
- `_safe_scalar_int`
- `bool`
- `choice`
- `choose_episode_phase_lengths`
- `clip`
- `control`
- `default_rng`
- `EpisodeSpec`
- `get`
- `isinstance`
- `max`
- `min`
- `pick_fault`
- `pick_sensor`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# ---- Mode switch ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `MODE = "batch" # "single" \| "batch"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TARGET_ROWS = 72_000` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `MAX_EPISODES = 1_000_000 # safety cap` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---- policy knobs ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `EPISODE_MAX_ROWS = 3_000 # prevents monster episodes; forces multiple episodes in a 10k batch` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ALLOW_OVERSHOOT = False # if True, can overshoot when remaining can't fit minimum core` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---- failure rarity control (match real dataset frequency) ----` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Real dataset: ~7 failures per 250,000 rows => ~1 failure per 35,714 rows` | Documents the purpose or boundary of the surrounding notebook step. |
| `ROWS_PER_FAILURE = 250_000 / 7 # ~35714.2857` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def _episode_total_rows(spec: EpisodeSpec) -> int: return int( spec.normal_before + spec.buildup + spec.failure + spec.recovery + spec.normal_after )` | Defines notebook-local logic used later in the notebook. |
| `def _rows_before_broken(spec: EpisodeSpec) -> int: """ Rows that occur before the BROKEN row inside a fault episode. """ return int(spec.normal_before + spec.buildup)` | Defines notebook-local logic used later in the notebook. |
| `def _rows_after_broken(spec: EpisodeSpec) -> int: """ Rows that occur after the BROKEN row inside a fault episode. """ return int(spec.recovery + spec.normal_after)` | Defines notebook-local logic used later in the notebook. |
| `def _failure_probability( *, rows_since_last_broken: int, candidate_rows_before_broken: int,` | Defines notebook-local logic used later in the notebook. |
| `) -> float: """ Stateful probability trigger. Instead of using a guessed episode length, use the actual accumulated rows since the last broken row plus the rows that would occur be` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def _is_failure_episode( rng: np.random.Generator, *, rows_since_last_broken: int, candidate_rows_before_broken: int,` | Defines notebook-local logic used later in the notebook. |
| `) -> bool: p = _failure_probability( rows_since_last_broken=int(rows_since_last_broken), candidate_rows_before_broken=int(candidate_rows_before_broken), ) return bool(rng.random() ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `rng = np.random.default_rng(int(SYN_CFG.get("random_seed", 42)))` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ep_cfg = SYN_CFG["episode"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `allowed_faults = (SYN_CFG.get("faults", {}) or {}).get("allowed", [])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if not allowed_faults: raise ValueError("SYN_CFG.faults.allowed is empty or missing")` | Controls validation, iteration, file handling, or error handling for this step. |
| `def _safe_scalar_int(x, *, default: int = 0) -> int: if x is None: return int(default) if isinstance(x, (list, tuple)): return int(x[0]) if len(x) else int(default) return int(x)` | Defines notebook-local logic used later in the notebook. |
| `def _safe_scalar_float(x, *, default: float = 0.0) -> float: if x is None: return float(default) if isinstance(x, (list, tuple)): return float(x[0]) if len(x) else float(default) r` | Defines notebook-local logic used later in the notebook. |
| `def pick_fault(rng: np.random.Generator) -> str: if PRIMARY_FAULT_TYPE is None or str(PRIMARY_FAULT_TYPE).strip() == "": return str(rng.choice(allowed_faults)) if isinstance(PRIMAR` | Defines notebook-local logic used later in the notebook. |
| `def pick_sensor(rng: np.random.Generator) -> str: if PRIMARY_SENSOR is None or str(PRIMARY_SENSOR).strip() == "": return str(rng.choice(FAULT_ELIGIBLE_SENSORS)) chosen = str(PRIMAR` | Defines notebook-local logic used later in the notebook. |
| `def sample_episode_spec( rng: np.random.Generator, rows_since_last_broken: int = 0,` | Defines notebook-local logic used later in the notebook. |
| `) -> EpisodeSpec: """ Unconstrained episode sampling (good for MODE='single'). Uses the same stateful failure trigger as batch mode. """ magnitude = sample_float_range(rng, ep_cfg[` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def sample_episode_spec_fit_remaining( rng: np.random.Generator, remaining_rows: int, *, rows_since_last_broken: int,` | Defines notebook-local logic used later in the notebook. |
| `) -> EpisodeSpec: """ Batch-safe sampling with stateful failure spacing. Differences from the old version: 1) failure probability is based on rows_since_last_broken + candidate row` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 33 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_classify_cluster_family`
- `_cluster_avg_abs_corr`
- `append`
- `chain`
- `chain_cluster_avg_abs_corr_threshold`
- `cluster`
- `cluster_size`
- `correlation_hotspot_clusters`
- `correlation_tuning`
- `DataFrame`
- `else`
- `episode`
- `failure`
- `family_split_rules`
- `generator`
- `get`
- `globals`
- `Loaded`
- `per`
- `range`

### Outputs

- `avg_abs_corr`
- `chain_threshold`
- `cluster_debug_df`
- `cluster_debug_rows`
- `family_name`

### Key Operations

- `chain_threshold = ( (generator.correlation_tuning.get("family_split_rules", {}) or {}) .get("chain_cluster_avg_abs_corr_threshold")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Loaded chain threshold:", chain_threshold)`: Displays a notebook-facing result for inspection.
- `print("Rows per failure:", ROWS_PER_FAILURE)`: Displays a notebook-facing result for inspection.
- `print("Recovery range:", RECOVERY_RANGE if "RECOVERY_RANGE" in globals() else SYN_CFG.get("episode", {}).get("recovery_range"))`: Displays a notebook-facing result for inspection.
- `cluster_debug_rows = []`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `for cluster in generator.correlation_hotspot_clusters: avg_abs_corr = generator._cluster_avg_abs_corr(cluster) family_name = generator._classify_cluster_family(cluster) cluster_deb`: Controls validation, iteration, file handling, or error handling for this step.
- `cluster_debug_df = pd.DataFrame(cluster_debug_rows)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `display(cluster_debug_df)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `_classify_cluster_family`
- `_cluster_avg_abs_corr`
- `append`
- `DataFrame`
- `display`
- `get`
- `globals`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `chain_threshold = ( (generator.correlation_tuning.get("family_split_rules", {}) or {}) .get("chain_cluster_avg_abs_corr_threshold")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Loaded chain threshold:", chain_threshold)` | Displays a notebook-facing result for inspection. |
| `print("Rows per failure:", ROWS_PER_FAILURE)` | Displays a notebook-facing result for inspection. |
| `print("Recovery range:", RECOVERY_RANGE if "RECOVERY_RANGE" in globals() else SYN_CFG.get("episode", {}).get("recovery_range"))` | Displays a notebook-facing result for inspection. |
| `cluster_debug_rows = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for cluster in generator.correlation_hotspot_clusters: avg_abs_corr = generator._cluster_avg_abs_corr(cluster) family_name = generator._classify_cluster_family(cluster) cluster_deb` | Controls validation, iteration, file handling, or error handling for this step. |
| `cluster_debug_df = pd.DataFrame(cluster_debug_rows)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `display(cluster_debug_df)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 34 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_episode_total_rows`
- `_rows_after_broken`
- `append`
- `are`
- `batch`
- `be`
- `because`
- `break`
- `cell`
- `column`
- `columns`
- `compatibility`
- `concat`
- `copy`
- `creates`
- `elif`
- `else`
- `episodes`
- `episodes_used`
- `exact`

### Outputs

- `chunks`
- `df_ep`
- `episode`
- `episode_count`
- `episode_id`
- `OBSERVABLE_MIN_CONSECUTIVE`
- `observable_min_consecutive`
- `OBSERVABLE_ZSCORE_THRESHOLD`
- `observable_zscore_threshold`
- `remaining`
- `remaining_rows`
- `row_count`
- `rows_since_last_broken`
- `ROWS_SINCE_LAST_BROKEN`
- `spec`
- `synthetic_df`

### Key Operations

- `# This cell is the ONLY place that creates/overwrites synthetic_df.`: Documents the purpose or boundary of the surrounding notebook step.
- `OBSERVABLE_ZSCORE_THRESHOLD = 2.5`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `OBSERVABLE_MIN_CONSECUTIVE = 3`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#ROWS_SINCE_LAST_BROKEN = 0`: Documents the purpose or boundary of the surrounding notebook step.
- `if MODE == "single": episode = sample_episode_spec(rng) synthetic_df = generator.generate_episode( episode, episode_id=0, observable_zscore_threshold=OBSERVABLE_ZSCORE_THRESHOLD, o`: Displays a notebook-facing result for inspection.
- `elif MODE == "batch": chunks = [] row_count = 0 episode_count = 0 ROWS_SINCE_LAST_BROKEN = 0 while row_count < TARGET_ROWS and episode_count < MAX_EPISODES: remaining = TARGET_ROWS`: Displays a notebook-facing result for inspection.
- `else: raise ValueError("MODE must be 'single' or 'batch'")`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `_episode_total_rows`
- `_rows_after_broken`
- `append`
- `concat`
- `copy`
- `display`
- `generate_episode`
- `nunique`
- `sample_episode_spec`
- `sample_episode_spec_fit_remaining`
- `size`
- `value_counts`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# This cell is the ONLY place that creates/overwrites synthetic_df.` | Documents the purpose or boundary of the surrounding notebook step. |
| `OBSERVABLE_ZSCORE_THRESHOLD = 2.5` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `OBSERVABLE_MIN_CONSECUTIVE = 3` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#ROWS_SINCE_LAST_BROKEN = 0` | Documents the purpose or boundary of the surrounding notebook step. |
| `if MODE == "single": episode = sample_episode_spec(rng) synthetic_df = generator.generate_episode( episode, episode_id=0, observable_zscore_threshold=OBSERVABLE_ZSCORE_THRESHOLD, o` | Displays a notebook-facing result for inspection. |
| `elif MODE == "batch": chunks = [] row_count = 0 episode_count = 0 ROWS_SINCE_LAST_BROKEN = 0 while row_count < TARGET_ROWS and episode_count < MAX_EPISODES: remaining = TARGET_ROWS` | Displays a notebook-facing result for inspection. |
| `else: raise ValueError("MODE must be 'single' or 'batch'")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 35 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `abnormal`
- `Add`
- `additional`
- `additional_missing_needed`
- `ADDS`
- `already`
- `append`
- `applied_missing_n`
- `are`
- `astype`
- `attempt`
- `be`
- `bool`
- `BROKEN`
- `build`
- `choice`
- `closely`
- `col`
- `column`

### Outputs

- `additional_needed`
- `applied_n`
- `apply_missingness_percentage_failsafe`
- `ascending`
- `audit_df`
- `audit_rows`
- `chosen_index`
- `columns`
- `current_missing_mask`
- `current_missing_n`
- `eligible_index`
- `eligible_mask`
- `eligible_n`
- `final_missing_n`
- `final_missing_pct`
- `n_rows`
- `out`
- `overshot_before`
- `phase_series`
- `protected_mask`

### Key Operations

- `def apply_missingness_percentage_failsafe( dataframe: pd.DataFrame, *, missingness_pct_all: dict[str, float], sensor_columns: Optional[Sequence[str]] = None, status_column: str = "`: Defines notebook-local logic used later in the notebook.
- `) -> tuple[pd.DataFrame, pd.DataFrame]: """ Add additional missing values to sensor columns until the final missing percentage matches the desired target as closely as possible. Th`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append`
- `apply_missingness_percentage_failsafe`
- `astype`
- `bool`
- `choice`
- `copy`
- `DataFrame`
- `default_rng`
- `fillna`
- `get`
- `isin`
- `isna`
- `keys`
- `lower`
- `max`
- `min`
- `reset_index`
- `round`
- `Series`
- `sort_values`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def apply_missingness_percentage_failsafe( dataframe: pd.DataFrame, *, missingness_pct_all: dict[str, float], sensor_columns: Optional[Sequence[str]] = None, status_column: str = "` | Defines notebook-local logic used later in the notebook. |
| `) -> tuple[pd.DataFrame, pd.DataFrame]: """ Add additional missing values to sensor columns until the final missing percentage matches the desired target as closely as possible. Th` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 36 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `apply_missingness_percentage_failsafe`
- `BROKEN`
- `f`
- `failed`
- `failure`
- `fault`
- `head`
- `i`
- `machine_status`
- `missingness_failsafe_audit_df`
- `missingness_spec`
- `phase`
- `range`
- `sensor_`
- `synthetic_df`

### Outputs

- `missingness_pct_all`
- `phase_column`
- `protected_phases`
- `protected_statuses`
- `random_seed`
- `sensor_columns`
- `status_column`

### Key Operations

- `synthetic_df, missingness_failsafe_audit_df = apply_missingness_percentage_failsafe( synthetic_df, missingness_pct_all=missingness_spec.missingness_pct_all, sensor_columns=[f"senso`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `display(missingness_failsafe_audit_df.head(20))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `apply_missingness_percentage_failsafe`
- `display`
- `head`
- `range`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `synthetic_df, missingness_failsafe_audit_df = apply_missingness_percentage_failsafe( synthetic_df, missingness_pct_all=missingness_spec.missingness_pct_all, sensor_columns=[f"senso` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(missingness_failsafe_audit_df.head(20))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 37 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `about`
- `added`
- `after`
- `all`
- `are`
- `be`
- `before`
- `by`
- `can`
- `column`
- `columns`
- `copy`
- `count`
- `database`
- `dataframe`
- `diagnostics`
- `Do`
- `dropped`
- `elif`
- `else`

### Outputs

- `EXPECTED_SENSOR_COLUMNS`
- `missing_pct`
- `missing_sensor_columns`
- `non_sensor_columns`
- `synthetic_df`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Restore full 52-sensor schema before diagnostics and database write`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Only sensors that are truly unavailable after generation are added.`: Documents the purpose or boundary of the surrounding notebook step.
- `# sensor_15 is expected to be all-null.`: Documents the purpose or boundary of the surrounding notebook step.
- `# sensor_50 should ideally be generated and then masked by missingness.`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `EXPECTED_SENSOR_COLUMNS = [f"sensor_{i:02d}" for i in range(52)]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `missing_sensor_columns = [ sensor for sensor in EXPECTED_SENSOR_COLUMNS if sensor not in synthetic_df.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Missing sensor columns before schema restore:", missing_sensor_columns)`: Displays a notebook-facing result for inspection.
- `for sensor in missing_sensor_columns: if sensor == "sensor_15": synthetic_df[sensor] = np.nan print("Restored sensor_15 as all-null expected column.") elif sensor == "sensor_50": r`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `copy`
- `isna`
- `mean`
- `range`
- `startswith`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Restore full 52-sensor schema before diagnostics and database write` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Only sensors that are truly unavailable after generation are added.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# sensor_15 is expected to be all-null.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# sensor_50 should ideally be generated and then masked by missingness.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `EXPECTED_SENSOR_COLUMNS = [f"sensor_{i:02d}" for i in range(52)]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `missing_sensor_columns = [ sensor for sensor in EXPECTED_SENSOR_COLUMNS if sensor not in synthetic_df.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Missing sensor columns before schema restore:", missing_sensor_columns)` | Displays a notebook-facing result for inspection. |
| `for sensor in missing_sensor_columns: if sensor == "sensor_15": synthetic_df[sensor] = np.nan print("Restored sensor_15 as all-null expected column.") elif sensor == "sensor_50": r` | Displays a notebook-facing result for inspection. |
| `non_sensor_columns = [ column for column in synthetic_df.columns if not str(column).startswith("sensor_")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `synthetic_df = synthetic_df[ non_sensor_columns + EXPECTED_SENSOR_COLUMNS` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Synthetic dataframe shape after schema restore:", synthetic_df.shape)` | Displays a notebook-facing result for inspection. |
| `print( "Sensor column count after schema restore:", len([column for column in synthetic_df.columns if str(column).startswith("sensor_")]),` | Displays a notebook-facing result for inspection. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `for sensor_name in ["sensor_15", "sensor_50"]: if sensor_name in synthetic_df.columns: missing_pct = float(synthetic_df[sensor_name].isna().mean() * 100.0) print(f"{sensor_name} mi` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 38 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `broken`
- `counts`
- `DataFrame`
- `dropna`
- `else`
- `EPISODE_STATS`
- `expected`
- `mean`
- `no`
- `phase`
- `recovering`
- `recovering_rows_per_broken_row`
- `recovery`
- `rows`
- `stream_state`
- `sum`
- `synthetic_df`
- `value_counts`

### Outputs

- `broken_rows`
- `episode_stats_df`
- `recovering_rows`

### Key Operations

- `print("stream_state counts")`: Displays a notebook-facing result for inspection.
- `display(synthetic_df["stream_state"].value_counts(dropna=False))`: Displays a notebook-facing result for inspection.
- `print("phase counts")`: Displays a notebook-facing result for inspection.
- `display(synthetic_df["phase"].value_counts(dropna=False))`: Displays a notebook-facing result for inspection.
- `broken_rows = int((synthetic_df["stream_state"] == "broken").sum())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `recovering_rows = int((synthetic_df["stream_state"] == "recovering").sum())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("broken_rows:", broken_rows)`: Displays a notebook-facing result for inspection.
- `print("recovering_rows:", recovering_rows)`: Displays a notebook-facing result for inspection.
- `if broken_rows > 0: print("recovering_rows_per_broken_row:", recovering_rows / broken_rows)`: Displays a notebook-facing result for inspection.
- `else: print("recovering_rows_per_broken_row: no broken rows")`: Displays a notebook-facing result for inspection.
- `if EPISODE_STATS: episode_stats_df = pd.DataFrame(EPISODE_STATS) print("expected mean recovery rows from EPISODE_STATS:", float(episode_stats_df["recovery"].mean()))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `DataFrame`
- `display`
- `mean`
- `sum`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print("stream_state counts")` | Displays a notebook-facing result for inspection. |
| `display(synthetic_df["stream_state"].value_counts(dropna=False))` | Displays a notebook-facing result for inspection. |
| `print("phase counts")` | Displays a notebook-facing result for inspection. |
| `display(synthetic_df["phase"].value_counts(dropna=False))` | Displays a notebook-facing result for inspection. |
| `broken_rows = int((synthetic_df["stream_state"] == "broken").sum())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `recovering_rows = int((synthetic_df["stream_state"] == "recovering").sum())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("broken_rows:", broken_rows)` | Displays a notebook-facing result for inspection. |
| `print("recovering_rows:", recovering_rows)` | Displays a notebook-facing result for inspection. |
| `if broken_rows > 0: print("recovering_rows_per_broken_row:", recovering_rows / broken_rows)` | Displays a notebook-facing result for inspection. |
| `else: print("recovering_rows_per_broken_row: no broken rows")` | Displays a notebook-facing result for inspection. |
| `if EPISODE_STATS: episode_stats_df = pd.DataFrame(EPISODE_STATS) print("expected mean recovery rows from EPISODE_STATS:", float(episode_stats_df["recovery"].mean()))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 39 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append`
- `coerce`
- `columns`
- `continue`
- `copy`
- `DataFrame`
- `ddof`
- `dropna`
- `errors`
- `head`
- `items`
- `loc`
- `max`
- `normal`
- `normal_profiles`
- `prof`
- `sensor`
- `sort_values`
- `std`
- `std_actual`

### Outputs

- `normal_df`
- `variance_check_df`
- `variance_check_rows`
- `x`

### Key Operations

- `normal_df = synthetic_df.loc[synthetic_df["stream_state"] == "normal"].copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `variance_check_rows = []`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `for sensor, prof in normal_profiles.items(): if sensor not in normal_df.columns: continue x = pd.to_numeric(normal_df[sensor], errors="coerce").dropna() if len(x) < 5: continue var`: Controls validation, iteration, file handling, or error handling for this step.
- `variance_check_df = pd.DataFrame(variance_check_rows).sort_values("std_ratio")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `display(variance_check_df.head(15))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `append`
- `copy`
- `DataFrame`
- `display`
- `dropna`
- `head`
- `items`
- `max`
- `sort_values`
- `std`
- `to_numeric`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `normal_df = synthetic_df.loc[synthetic_df["stream_state"] == "normal"].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `variance_check_rows = []` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for sensor, prof in normal_profiles.items(): if sensor not in normal_df.columns: continue x = pd.to_numeric(normal_df[sensor], errors="coerce").dropna() if len(x) < 5: continue var` | Controls validation, iteration, file handling, or error handling for this step. |
| `variance_check_df = pd.DataFrame(variance_check_rows).sort_values("std_ratio")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `display(variance_check_df.head(15))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 40 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `astype`
- `broken`
- `counts`
- `dropna`
- `episodes`
- `f`
- `found`
- `issubset`
- `meta__episode_id`
- `normal`
- `normalize`
- `nunique`
- `phase`
- `raise`
- `recovering`
- `rows`
- `sort_index`
- `sorted`
- `stream_state`
- `sum`

### Outputs

- `actual_stream_states`
- `allowed_stream_states`

### Key Operations

- `print("rows:", len(synthetic_df))`: Displays a notebook-facing result for inspection.
- `print(synthetic_df["stream_state"].value_counts().sort_index())`: Displays a notebook-facing result for inspection.
- `print(synthetic_df["stream_state"].value_counts(normalize=True).sort_index())`: Displays a notebook-facing result for inspection.
- `print("broken rows:", int((synthetic_df["stream_state"] == "broken").sum()))`: Displays a notebook-facing result for inspection.
- `print("recovering rows:", int((synthetic_df["stream_state"] == "recovering").sum()))`: Displays a notebook-facing result for inspection.
- `print("episodes:", synthetic_df["meta__episode_id"].nunique())`: Displays a notebook-facing result for inspection.
- `print("phase counts:")`: Displays a notebook-facing result for inspection.
- `print(synthetic_df["phase"].value_counts().sort_index())`: Displays a notebook-facing result for inspection.
- `allowed_stream_states = {"normal", "broken", "recovering"}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `actual_stream_states = set(synthetic_df["stream_state"].dropna().astype(str).unique().tolist())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("allowed_stream_states:", allowed_stream_states)`: Displays a notebook-facing result for inspection.
- `print("actual_stream_states:", actual_stream_states)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `astype`
- `dropna`
- `issubset`
- `nunique`
- `sort_index`
- `sorted`
- `sum`
- `tolist`
- `unique`
- `value_counts`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print("rows:", len(synthetic_df))` | Displays a notebook-facing result for inspection. |
| `print(synthetic_df["stream_state"].value_counts().sort_index())` | Displays a notebook-facing result for inspection. |
| `print(synthetic_df["stream_state"].value_counts(normalize=True).sort_index())` | Displays a notebook-facing result for inspection. |
| `print("broken rows:", int((synthetic_df["stream_state"] == "broken").sum()))` | Displays a notebook-facing result for inspection. |
| `print("recovering rows:", int((synthetic_df["stream_state"] == "recovering").sum()))` | Displays a notebook-facing result for inspection. |
| `print("episodes:", synthetic_df["meta__episode_id"].nunique())` | Displays a notebook-facing result for inspection. |
| `print("phase counts:")` | Displays a notebook-facing result for inspection. |
| `print(synthetic_df["phase"].value_counts().sort_index())` | Displays a notebook-facing result for inspection. |
| `allowed_stream_states = {"normal", "broken", "recovering"}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `actual_stream_states = set(synthetic_df["stream_state"].dropna().astype(str).unique().tolist())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("allowed_stream_states:", allowed_stream_states)` | Displays a notebook-facing result for inspection. |
| `print("actual_stream_states:", actual_stream_states)` | Displays a notebook-facing result for inspection. |
| `if not actual_stream_states.issubset(allowed_stream_states): raise ValueError( f"Unexpected stream_state values found: {sorted(actual_stream_states - allowed_stream_states)}" )` | Controls validation, iteration, file handling, or error handling for this step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 41 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `normalize`
- `sort_index`
- `stream_state`
- `synthetic_df`
- `value_counts`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `print(synthetic_df["stream_state"].value_counts(normalize=True).sort_index())`: Displays a notebook-facing result for inspection.
- `print(synthetic_df["stream_state"].value_counts().sort_index())`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `sort_index`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print(synthetic_df["stream_state"].value_counts(normalize=True).sort_index())` | Displays a notebook-facing result for inspection. |
| `print(synthetic_df["stream_state"].value_counts().sort_index())` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 42 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `actually`
- `All`
- `all`
- `Basic`
- `bool`
- `check`
- `column`
- `Columns`
- `columns`
- `counts`
- `else`
- `have`
- `isna`
- `mean`
- `missing`
- `NA`
- `null`
- `Null`
- `output`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `print("Columns:", int(len(synthetic_df.columns)))`: Displays a notebook-facing result for inspection.
- `print("Rows:", int(len(synthetic_df)))`: Displays a notebook-facing result for inspection.
- `if "sensor_50" in synthetic_df.columns: print("Null % sensor_50:", float(synthetic_df["sensor_50"].isna().mean() * 100.0))`: Displays a notebook-facing result for inspection.
- `else: print("Null % sensor_50: NA (column missing)")`: Displays a notebook-facing result for inspection.
- `if "sensor_15" in synthetic_df.columns: print("All-null sensor_15:", bool(synthetic_df["sensor_15"].isna().all()))`: Displays a notebook-facing result for inspection.
- `else: print("All-null sensor_15: NA (column missing)")`: Displays a notebook-facing result for inspection.
- `# Basic check that we actually have abnormal segments in the output:`: Documents the purpose or boundary of the surrounding notebook step.
- `if "stream_state" in synthetic_df.columns: print("stream_state counts:", synthetic_df["stream_state"].value_counts().to_dict())`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `all`
- `bool`
- `isna`
- `mean`
- `NA`
- `to_dict`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print("Columns:", int(len(synthetic_df.columns)))` | Displays a notebook-facing result for inspection. |
| `print("Rows:", int(len(synthetic_df)))` | Displays a notebook-facing result for inspection. |
| `if "sensor_50" in synthetic_df.columns: print("Null % sensor_50:", float(synthetic_df["sensor_50"].isna().mean() * 100.0))` | Displays a notebook-facing result for inspection. |
| `else: print("Null % sensor_50: NA (column missing)")` | Displays a notebook-facing result for inspection. |
| `if "sensor_15" in synthetic_df.columns: print("All-null sensor_15:", bool(synthetic_df["sensor_15"].isna().all()))` | Displays a notebook-facing result for inspection. |
| `else: print("All-null sensor_15: NA (column missing)")` | Displays a notebook-facing result for inspection. |
| `# Basic check that we actually have abnormal segments in the output:` | Documents the purpose or boundary of the surrounding notebook step. |
| `if "stream_state" in synthetic_df.columns: print("stream_state counts:", synthetic_df["stream_state"].value_counts().to_dict())` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 43 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `broken`
- `counts`
- `dropna`
- `else`
- `machine_status`
- `no`
- `phase`
- `recovering`
- `recovering_rows_per_broken_row`
- `rows`
- `stream_state`
- `sum`
- `synthetic_df`
- `to_dict`
- `value_counts`

### Outputs

- `broken_rows`
- `phase_counts`
- `recovering_rows`

### Key Operations

- `print("machine_status / stream_state counts")`: Displays a notebook-facing result for inspection.
- `display(synthetic_df["stream_state"].value_counts(dropna=False))`: Displays a notebook-facing result for inspection.
- `phase_counts = synthetic_df["phase"].value_counts(dropna=False).to_dict()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("phase_counts:", phase_counts)`: Displays a notebook-facing result for inspection.
- `broken_rows = int((synthetic_df["stream_state"] == "broken").sum())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `recovering_rows = int((synthetic_df["stream_state"] == "recovering").sum())`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("broken_rows:", broken_rows)`: Displays a notebook-facing result for inspection.
- `print("recovering_rows:", recovering_rows)`: Displays a notebook-facing result for inspection.
- `if broken_rows > 0: print("recovering_rows_per_broken_row:", recovering_rows / broken_rows)`: Displays a notebook-facing result for inspection.
- `else: print("recovering_rows_per_broken_row: no broken rows")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `sum`
- `to_dict`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print("machine_status / stream_state counts")` | Displays a notebook-facing result for inspection. |
| `display(synthetic_df["stream_state"].value_counts(dropna=False))` | Displays a notebook-facing result for inspection. |
| `phase_counts = synthetic_df["phase"].value_counts(dropna=False).to_dict()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("phase_counts:", phase_counts)` | Displays a notebook-facing result for inspection. |
| `broken_rows = int((synthetic_df["stream_state"] == "broken").sum())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `recovering_rows = int((synthetic_df["stream_state"] == "recovering").sum())` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("broken_rows:", broken_rows)` | Displays a notebook-facing result for inspection. |
| `print("recovering_rows:", recovering_rows)` | Displays a notebook-facing result for inspection. |
| `if broken_rows > 0: print("recovering_rows_per_broken_row:", recovering_rows / broken_rows)` | Displays a notebook-facing result for inspection. |
| `else: print("recovering_rows_per_broken_row: no broken rows")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 44 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `append`
- `ASC`
- `ascending`
- `b`
- `batch`
- `batch_col`
- `batch_id`
- `batches`
- `be`
- `becomes`
- `behavior`
- `bool`
- `BY`
- `CASCADE`
- `COALESCE`
- `continue`
- `COUNT`
- `def`
- `DENSE_RANK`
- `DISTINCT`

### Outputs

- `append_mode`
- `canonicalize_existing_batch_ids`
- `choose_batch_id`
- `df`
- `drop_table`
- `exists`
- `fq`
- `get_max_batch_id`
- `sql`
- `summary`
- `table_exists`
- `write_mode`

### Key Operations

- `def table_exists(engine, *, schema: str, table_name: str) -> bool: sql = """ SELECT EXISTS( SELECT 1 FROM information_schema.tables WHERE table_schema = :schema AND table_name = :t`: Defines notebook-local logic used later in the notebook.
- `def drop_table(engine, *, schema: str, table_name: str) -> None: execute_sql(engine, f'DROP TABLE IF EXISTS "{schema}"."{table_name}" CASCADE')`: Defines notebook-local logic used later in the notebook.
- `def get_max_batch_id(engine, *, schema: str, table_name: str, batch_col: str = "batch_id") -> int: fq = f'"{schema}"."{table_name}"' df = read_sql_dataframe(engine, f"SELECT COALES`: Defines notebook-local logic used later in the notebook.
- `def canonicalize_existing_batch_ids(engine, *, schema: str, table_name: str, batch_col: str = "batch_id") -> Dict[str, int]: """ Renumber existing batch ids in-place to 1..N, prese`: Defines notebook-local logic used later in the notebook.
- `def choose_batch_id( engine, *, schema: str, table_name: str, write_mode: str, # "reset" \| "append" append_mode: str, # "continue" \| "renumber" (only used if write_mode="append") b`: Defines notebook-local logic used later in the notebook.
- `) -> int: """ Implements your exact behavior: - write_mode="reset": drop table, recreate later via writer, next batch_id = 1 - write_mode="append", append_mode="continue": do NOT r`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `AS`
- `bool`
- `canonicalize_existing_batch_ids`
- `choose_batch_id`
- `COALESCE`
- `COUNT`
- `DENSE_RANK`
- `drop_table`
- `execute_sql`
- `EXISTS`
- `get_max_batch_id`
- `lower`
- `MAX`
- `MIN`
- `N`
- `OVER`
- `read_sql_dataframe`
- `strip`
- `table_exists`
- `ValueError`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def table_exists(engine, *, schema: str, table_name: str) -> bool: sql = """ SELECT EXISTS( SELECT 1 FROM information_schema.tables WHERE table_schema = :schema AND table_name = :t` | Defines notebook-local logic used later in the notebook. |
| `def drop_table(engine, *, schema: str, table_name: str) -> None: execute_sql(engine, f'DROP TABLE IF EXISTS "{schema}"."{table_name}" CASCADE')` | Defines notebook-local logic used later in the notebook. |
| `def get_max_batch_id(engine, *, schema: str, table_name: str, batch_col: str = "batch_id") -> int: fq = f'"{schema}"."{table_name}"' df = read_sql_dataframe(engine, f"SELECT COALES` | Defines notebook-local logic used later in the notebook. |
| `def canonicalize_existing_batch_ids(engine, *, schema: str, table_name: str, batch_col: str = "batch_id") -> Dict[str, int]: """ Renumber existing batch ids in-place to 1..N, prese` | Defines notebook-local logic used later in the notebook. |
| `def choose_batch_id( engine, *, schema: str, table_name: str, write_mode: str, # "reset" \| "append" append_mode: str, # "continue" \| "renumber" (only used if write_mode="append") b` | Defines notebook-local logic used later in the notebook. |
| `) -> int: """ Implements your exact behavior: - write_mode="reset": drop table, recreate later via writer, next batch_id = 1 - write_mode="append", append_mode="continue": do NOT r` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 45 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `_batch_id`
- `_cycle_id`
- `append`
- `based`
- `can`
- `capstone`
- `choose_batch_id`
- `Chosen`
- `continue`
- `cycles`
- `ensure_sequence`
- `f`
- `flags`
- `get_engine_from_env`
- `lower`
- `matters`
- `mode`
- `now`
- `only`

### Outputs

- `APPEND_MODE`
- `append_mode`
- `artifact_name`
- `ARTIFACT_NAME`
- `batch_col`
- `batch_id`
- `cycle_start`
- `dataset_name`
- `engine`
- `n_rows`
- `PG_SCHEMA`
- `schema`
- `sequence_name`
- `table_name`
- `TABLE_NAME`
- `table_written`
- `write_mode`
- `WRITE_MODE`

### Key Operations

- `engine = get_engine_from_env()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `PG_SCHEMA = "capstone"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `ARTIFACT_NAME = "stream"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `TABLE_NAME = f"synthetic_{DATASET_NAME.lower()}_{ARTIFACT_NAME}"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# ---- write mode flags`: Documents the purpose or boundary of the surrounding notebook step.
- `WRITE_MODE = "reset" # "reset" \| "append"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `APPEND_MODE = "renumber" # "continue" \| "renumber" (only matters if WRITE_MODE="append")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `batch_id = choose_batch_id( engine, schema=PG_SCHEMA, table_name=TABLE_NAME, write_mode=WRITE_MODE, append_mode=APPEND_MODE, batch_col="batch_id",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Chosen batch_id:", batch_id)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `choose_batch_id`
- `ensure_sequence`
- `get_engine_from_env`
- `lower`
- `reserve_cycle_range`
- `reset_synthetic_sequences`
- `strip`
- `write_stream_batch`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `engine = get_engine_from_env()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `PG_SCHEMA = "capstone"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ARTIFACT_NAME = "stream"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `TABLE_NAME = f"synthetic_{DATASET_NAME.lower()}_{ARTIFACT_NAME}"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# ---- write mode flags` | Documents the purpose or boundary of the surrounding notebook step. |
| `WRITE_MODE = "reset" # "reset" \| "append"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `APPEND_MODE = "renumber" # "continue" \| "renumber" (only matters if WRITE_MODE="append")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `batch_id = choose_batch_id( engine, schema=PG_SCHEMA, table_name=TABLE_NAME, write_mode=WRITE_MODE, append_mode=APPEND_MODE, batch_col="batch_id",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Chosen batch_id:", batch_id)` | Displays a notebook-facing result for inspection. |
| `if str(WRITE_MODE).strip().lower() == "reset": ensure_sequence(engine, schema=PG_SCHEMA, sequence_name=f"seq_synthetic_{DATASET_NAME.lower()}_batch_id") ensure_sequence(engine, sch` | Controls validation, iteration, file handling, or error handling for this step. |
| `# cycles can stay sequence-based for now` | Documents the purpose or boundary of the surrounding notebook step. |
| `ensure_sequence(engine, schema=PG_SCHEMA, sequence_name=f"seq_synthetic_{DATASET_NAME.lower()}_cycle_id")` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `cycle_start = reserve_cycle_range( engine, schema=PG_SCHEMA, sequence_name=f"seq_synthetic_{DATASET_NAME.lower()}_cycle_id", n_rows=len(synthetic_df),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `table_written = write_stream_batch( engine, synthetic_df, dataset_name=DATASET_NAME, schema=PG_SCHEMA, artifact_name=ARTIFACT_NAME, batch_id=batch_id, cycle_start=cycle_start,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Wrote:", table_written, "batch_id:", batch_id, "cycle_start:", cycle_start)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 46 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `DB_HOST`
- `DB_NAME`
- `DB_PASSWORD`
- `DB_PORT`
- `DB_USER`
- `environ`
- `get`
- `k`
- `os`
- `POSTGRES_DB`
- `POSTGRES_PASSWORD`
- `POSTGRES_USER`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import os`: Imports a dependency or project helper used by later cells.
- `{k: os.environ.get(k) for k in ["DB_HOST","DB_PORT","DB_NAME","DB_USER","DB_PASSWORD","POSTGRES_DB","POSTGRES_USER","POSTGRES_PASSWORD"]}`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `get`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import os` | Imports a dependency or project helper used by later cells. |
| `{k: os.environ.get(k) for k in ["DB_HOST","DB_PORT","DB_NAME","DB_USER","DB_PASSWORD","POSTGRES_DB","POSTGRES_USER","POSTGRES_PASSWORD"]}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 47 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `find_spec`
- `importlib`
- `psycopg`
- `psycopg2`
- `util`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `import importlib.util`: Imports a dependency or project helper used by later cells.
- `print("psycopg:", importlib.util.find_spec("psycopg"))`: Displays a notebook-facing result for inspection.
- `print("psycopg2:", importlib.util.find_spec("psycopg2"))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `find_spec`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `import importlib.util` | Imports a dependency or project helper used by later cells. |
| `print("psycopg:", importlib.util.find_spec("psycopg"))` | Displays a notebook-facing result for inspection. |
| `print("psycopg2:", importlib.util.find_spec("psycopg2"))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 48 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `an`
- `Coerce`
- `convert`
- `def`
- `element`
- `empty`
- `Exception`
- `first`
- `into`
- `isinstance`
- `possible`
- `safely`
- `string`
- `take`
- `the`
- `tuple`

### Outputs

- `_as_float`
- `_as_int`
- `x`

### Key Operations

- `def _as_int(x): """ Coerce x into an int if possible. - If x is list/tuple: take the first element (or None if empty) - If x is None: return None - If x is float/string/int: conver`: Defines notebook-local logic used later in the notebook.
- `def _as_float(x): """ Coerce x into a float if possible. - If x is list/tuple: take the first element (or None if empty) - If x is None: return None """ if x is None: return None i`: Defines notebook-local logic used later in the notebook.

Important functions or methods detected:
- `_as_float`
- `_as_int`
- `element`
- `isinstance`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def _as_int(x): """ Coerce x into an int if possible. - If x is list/tuple: take the first element (or None if empty) - If x is None: return None - If x is float/string/int: conver` | Defines notebook-local logic used later in the notebook. |
| `def _as_float(x): """ Coerce x into a float if possible. - If x is list/tuple: take the first element (or None if empty) - If x is None: return None """ if x is None: return None i` | Defines notebook-local logic used later in the notebook. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 49 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `across`
- `actually`
- `columns`
- `created`
- `different`
- `drop_duplicates`
- `else`
- `episodes`
- `faults`
- `happened`
- `head`
- `how`
- `many`
- `meta__episode_id`
- `meta__primary_fault_type`
- `meta__primary_sensor`
- `NO`
- `nunique`
- `primary`
- `primary_fault_type`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `# episodes actually created`: Documents the purpose or boundary of the surrounding notebook step.
- `print("episodes:", synthetic_df["meta__episode_id"].nunique() if "meta__episode_id" in synthetic_df.columns else "NO meta__episode_id")`: Displays a notebook-facing result for inspection.
- `# how many different primary faults / sensors happened across episodes`: Documents the purpose or boundary of the surrounding notebook step.
- `if "meta__primary_fault_type" in synthetic_df.columns: print("unique primary_fault_type:", synthetic_df["meta__primary_fault_type"].nunique()) print(synthetic_df[["meta__episode_id`: Displays a notebook-facing result for inspection.
- `if "meta__primary_sensor" in synthetic_df.columns: print("unique primary_sensor:", synthetic_df["meta__primary_sensor"].nunique()) print(synthetic_df[["meta__episode_id","meta__pri`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `drop_duplicates`
- `head`
- `nunique`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# episodes actually created` | Documents the purpose or boundary of the surrounding notebook step. |
| `print("episodes:", synthetic_df["meta__episode_id"].nunique() if "meta__episode_id" in synthetic_df.columns else "NO meta__episode_id")` | Displays a notebook-facing result for inspection. |
| `# how many different primary faults / sensors happened across episodes` | Documents the purpose or boundary of the surrounding notebook step. |
| `if "meta__primary_fault_type" in synthetic_df.columns: print("unique primary_fault_type:", synthetic_df["meta__primary_fault_type"].nunique()) print(synthetic_df[["meta__episode_id` | Displays a notebook-facing result for inspection. |
| `if "meta__primary_sensor" in synthetic_df.columns: print("unique primary_sensor:", synthetic_df["meta__primary_sensor"].nunique()) print(synthetic_df[["meta__episode_id","meta__pri` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 50 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `d`
- `datetime`
- `H`
- `hours`
- `m`
- `M`
- `now`
- `strftime`
- `timedelta`
- `Y_`

### Outputs

- `adjusted_time`
- `current_datetime`
- `formatted_datetime`

### Key Operations

- `current_datetime = datetime.now()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `adjusted_time = current_datetime - timedelta(hours=4)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `formatted_datetime = adjusted_time.strftime("%m%d%Y_%H%M")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `now`
- `strftime`
- `timedelta`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `current_datetime = datetime.now()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `adjusted_time = current_datetime - timedelta(hours=4)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `formatted_datetime = adjusted_time.strftime("%m%d%Y_%H%M")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 51 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__`
- `__dict__`
- `__resolved_config`
- `__synthetic__`
- `_as_float`
- `_as_int`
- `a`
- `alias`
- `append_truth_index`
- `artifact`
- `artifact_paths`
- `ARTIFACTS_ROOT`
- `batch`
- `batch_id`
- `build_truth_config_block`
- `build_truth_record`
- `BUILDUP`
- `buildup_range`
- `buildup_selection`
- `column`

### Outputs

- `artifact_paths_payload`
- `column_count`
- `dataset_name`
- `feature_columns`
- `layer_name`
- `LAYER_NAME`
- `meta_columns`
- `out_path`
- `parent_truth_hash`
- `pipeline_mode`
- `process_run_id`
- `resolved_config_dir`
- `resolved_config_path`
- `row_count`
- `suffix`
- `synth_dir`
- `synth_truth_hash`
- `synthetic_df`
- `synthetic_truth`
- `truth_base`

### Key Operations

- `process_run_id = RUN_ID`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `synthetic_truth = initialize_layer_truth( truth_version=str(TRUTH_VERSION), dataset_name=DATASET_NAME, layer_name="synthetic", process_run_id=process_run_id, pipeline_mode=PIPELINE`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `LAYER_NAME = "synthetic"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `resolved_config_dir = ARTIFACTS_ROOT / "synthetic" / DATASET_NAME`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `resolved_config_dir.mkdir(parents=True, exist_ok=True)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `resolved_config_path = resolved_config_dir / f"{DATASET_NAME}__{LAYER_NAME}__resolved_config.yaml"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# export_config_snapshot requires a destination path`: Documents the purpose or boundary of the surrounding notebook step.
- `export_config_snapshot(CONFIG, destination=resolved_config_path)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Resolved config written to:", resolved_config_path)`: Displays a notebook-facing result for inspection.
- `synthetic_truth = update_truth_section( synthetic_truth, "config_snapshot", { # store the path you just wrote "resolved_config_path": str(resolved_config_path), # optional: small i`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `_as_float`
- `_as_int`
- `append_truth_index`
- `build_truth_config_block`
- `build_truth_record`
- `dataframe`
- `dropna`
- `else`
- `export_config_snapshot`
- `getattr`
- `globals`
- `hasattr`
- `initialize_layer_truth`
- `max`
- `min`
- `mkdir`
- `mode`
- `Path`
- `save_data`
- `save_truth_record`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `process_run_id = RUN_ID` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `synthetic_truth = initialize_layer_truth( truth_version=str(TRUTH_VERSION), dataset_name=DATASET_NAME, layer_name="synthetic", process_run_id=process_run_id, pipeline_mode=PIPELINE` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `LAYER_NAME = "synthetic"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `resolved_config_dir = ARTIFACTS_ROOT / "synthetic" / DATASET_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `resolved_config_dir.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `resolved_config_path = resolved_config_dir / f"{DATASET_NAME}__{LAYER_NAME}__resolved_config.yaml"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# export_config_snapshot requires a destination path` | Documents the purpose or boundary of the surrounding notebook step. |
| `export_config_snapshot(CONFIG, destination=resolved_config_path)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Resolved config written to:", resolved_config_path)` | Displays a notebook-facing result for inspection. |
| `synthetic_truth = update_truth_section( synthetic_truth, "config_snapshot", { # store the path you just wrote "resolved_config_path": str(resolved_config_path), # optional: small i` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `synthetic_truth = update_truth_section( synthetic_truth, "runtime_facts", { "primary_sensor": ( getattr(episode, "primary_sensor", None) if "episode" in globals() else (str(synthet` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Optionally save local parquet artifact too (useful for debugging)` | Documents the purpose or boundary of the surrounding notebook step. |
| `synth_dir = ARTIFACTS_ROOT / "synthetic" / DATASET_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `synth_dir.mkdir(parents=True, exist_ok=True)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `suffix = "episode" if ("MODE" in globals() and str(MODE) == "single") else "batch"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `out_path = synth_dir / f"{DATASET_NAME}__synthetic__{suffix}.parquet"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `save_data(synthetic_df, synth_dir, out_path.name)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `artifact_paths_payload = { "parent_truth_hash": PARENT_TRUTH_HASH, "silver_parent_layer_name": SILVER_PARENT_LAYER_NAME, "silver_parent_truth_hash": SILVER_PARENT_TRUTH_HASH, "silv` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `#export_path = Path(PATHS["data_raw_dir"] / "synthetic")` | Documents the purpose or boundary of the surrounding notebook step. |
| `if EXPORT_ENABLED: artifact_paths_payload["export_batch_parquet_path"] = str(out_path)` | Controls validation, iteration, file handling, or error handling for this step. |
| `synthetic_truth = update_truth_section(synthetic_truth, "artifact_paths", artifact_paths_payload)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `meta_columns = sorted(["meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `feature_columns = sorted([column for column in synthetic_df.columns if not str(column).startswith("meta__")])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `truth_record = build_truth_record( truth_base=synthetic_truth, row_count=int(len(synthetic_df)), column_count=int(synthetic_df.shape[1] + 3), meta_columns=meta_columns, feature_col` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `synth_truth_hash = truth_record["truth_hash"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# stamp lineage columns into dataframe (optional)` | Documents the purpose or boundary of the surrounding notebook step. |
| `synthetic_df = stamp_truth_columns( synthetic_df, truth_hash=synth_truth_hash, parent_truth_hash=PARENT_TRUTH_HASH, pipeline_mode=PIPELINE_MODE,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `truth_path = save_truth_record( truth_record, truth_dir=TRUTHS_PATH, dataset_name=DATASET_NAME, layer_name="synthetic",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `append_truth_index(truth_record, truth_index_path=TRUTH_INDEX_PATH)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Synthetic truth hash:", synth_truth_hash)` | Displays a notebook-facing result for inspection. |
| `print("Synthetic truth path:", truth_path)` | Displays a notebook-facing result for inspection. |
| `print("Local episode parquet:", out_path)` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: truth record.

## Code Cell 52 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_missingness_spec_from_truth_payload`
- `EpisodeSpec`
- `export`
- `export_synthetic_batch_to_parquet`
- `generator`
- `Imports`
- `load_and_merge_rich_profiles`
- `missingness`
- `OK`
- `profiles`
- `synthetic`
- `SyntheticGenerator`
- `utils`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `from utils.synthetic.generator.profiles import load_and_merge_rich_profiles`: Imports a dependency or project helper used by later cells.
- `from utils.synthetic.generator.generator import SyntheticGenerator, EpisodeSpec`: Imports a dependency or project helper used by later cells.
- `from utils.synthetic.generator.missingness import build_missingness_spec_from_truth_payload`: Imports a dependency or project helper used by later cells.
- `from utils.synthetic.generator.export import export_synthetic_batch_to_parquet`: Imports a dependency or project helper used by later cells.
- `print("Imports OK")`: Displays a notebook-facing result for inspection.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `from utils.synthetic.generator.profiles import load_and_merge_rich_profiles` | Imports a dependency or project helper used by later cells. |
| `from utils.synthetic.generator.generator import SyntheticGenerator, EpisodeSpec` | Imports a dependency or project helper used by later cells. |
| `from utils.synthetic.generator.missingness import build_missingness_spec_from_truth_payload` | Imports a dependency or project helper used by later cells. |
| `from utils.synthetic.generator.export import export_synthetic_batch_to_parquet` | Imports a dependency or project helper used by later cells. |
| `print("Imports OK")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: Parquet output.

## Code Cell 53 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `else`
- `Episodes`
- `fault`
- `head`
- `meta__episode_id`
- `meta__primary_fault_type`
- `meta__primary_sensor`
- `NA`
- `nunique`
- `Primary`
- `sensors`
- `synthetic_df`
- `types`
- `value_counts`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `print("Episodes:", synthetic_df["meta__episode_id"].nunique() if "meta__episode_id" in synthetic_df else "NA")`: Displays a notebook-facing result for inspection.
- `print("Primary fault types:", synthetic_df["meta__primary_fault_type"].value_counts().head(10) if "meta__primary_fault_type" in synthetic_df else "NA")`: Displays a notebook-facing result for inspection.
- `print("Primary sensors:", synthetic_df["meta__primary_sensor"].value_counts().head(10) if "meta__primary_sensor" in synthetic_df else "NA")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `head`
- `nunique`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print("Episodes:", synthetic_df["meta__episode_id"].nunique() if "meta__episode_id" in synthetic_df else "NA")` | Displays a notebook-facing result for inspection. |
| `print("Primary fault types:", synthetic_df["meta__primary_fault_type"].value_counts().head(10) if "meta__primary_fault_type" in synthetic_df else "NA")` | Displays a notebook-facing result for inspection. |
| `print("Primary sensors:", synthetic_df["meta__primary_sensor"].value_counts().head(10) if "meta__primary_sensor" in synthetic_df else "NA")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 54 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `astype`
- `c`
- `columns`
- `def`
- `df`
- `dropna`
- `else`
- `expected_sensors`
- `extra_sensor_columns`
- `missing_sensor_columns`
- `phase`
- `phase_col`
- `phase_values`
- `row_count`
- `sensor_`
- `sorted`
- `startswith`
- `state_col`
- `state_values`
- `stream_state`

### Outputs

- `cols`
- `exp`
- `extra_sensor_cols`
- `missing_cols`
- `out`
- `verify_schema`

### Key Operations

- `def verify_schema( df, *, expected_sensors: list[str], state_col: str = "stream_state", phase_col: str = "phase",`: Defines notebook-local logic used later in the notebook.
- `) -> dict: cols = set(df.columns) exp = set(expected_sensors) missing_cols = sorted(exp - cols) extra_sensor_cols = sorted([c for c in cols if c.startswith("sensor_") and c not in `: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `astype`
- `dropna`
- `sorted`
- `startswith`
- `unique`
- `verify_schema`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def verify_schema( df, *, expected_sensors: list[str], state_col: str = "stream_state", phase_col: str = "phase",` | Defines notebook-local logic used later in the notebook. |
| `) -> dict: cols = set(df.columns) exp = set(expected_sensors) missing_cols = sorted(exp - cols) extra_sensor_cols = sorted([c for c in cols if c.startswith("sensor_") and c not in ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 55 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abs`
- `actual_missing_rows`
- `actual_pct`
- `append`
- `ascending`
- `c`
- `columns`
- `consistent`
- `continue`
- `DataFrame`
- `def`
- `df`
- `diff_missing_rows`
- `else`
- `empty`
- `expectation`
- `expected_missing_rows`
- `generator`
- `get`
- `integer`

### Outputs

- `actual_missing`
- `diff_rows`
- `expected_missing`
- `expected_present`
- `n`
- `ok`
- `out`
- `rows`
- `target`
- `tol_missing_rows`
- `verify_missingness_exact`

### Key Operations

- `def verify_missingness_exact( df: pd.DataFrame, *, target_missing_pct: dict[str, float], sensors: list[str], tol_missing_rows: int \| None = None,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: n = int(len(df)) if tol_missing_rows is None: tol_missing_rows = max(10, int(round(0.0002 * n))) rows = [] for s in sensors: if s not in df.columns: continue tar`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `abs`
- `append`
- `DataFrame`
- `get`
- `isna`
- `max`
- `min`
- `round`
- `sort_values`
- `sum`
- `verify_missingness_exact`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def verify_missingness_exact( df: pd.DataFrame, *, target_missing_pct: dict[str, float], sensors: list[str], tol_missing_rows: int \| None = None,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: n = int(len(df)) if tol_missing_rows is None: tol_missing_rows = max(10, int(round(0.0002 * n))) rows = [] for s in sensors: if s not in df.columns: continue tar` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 56 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `append`
- `ascending`
- `astype`
- `bound_cols`
- `coerce`
- `columns`
- `continue`
- `copy`
- `DataFrame`
- `ddof`
- `def`
- `df`
- `dropna`
- `else`
- `empty`
- `errors`
- `iloc`
- `loc`
- `mean`

### Outputs

- `df_st`
- `hi`
- `lo`
- `nonnull`
- `out`
- `outside`
- `p`
- `prof`
- `rows`
- `series`
- `verify_profile_bounds`

### Key Operations

- `def verify_profile_bounds( df: pd.DataFrame, *, profile_df: pd.DataFrame, # rows: sensor,state_scope,p01,p99,mean,std,... sensors: list[str], state_col: str = "stream_state", state`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: p01_col, p99_col = bound_cols prof = profile_df.copy() prof["sensor"] = prof["sensor"].astype(str) prof["state_scope"] = prof["state_scope"].astype(str) rows = [`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append`
- `astype`
- `copy`
- `DataFrame`
- `dropna`
- `mean`
- `sort_values`
- `std`
- `to_numeric`
- `verify_profile_bounds`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def verify_profile_bounds( df: pd.DataFrame, *, profile_df: pd.DataFrame, # rows: sensor,state_scope,p01,p99,mean,std,... sensors: list[str], state_col: str = "stream_state", state` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: p01_col, p99_col = bound_cols prof = profile_df.copy() prof["sensor"] = prof["sensor"].astype(str) prof["state_scope"] = prof["state_scope"].astype(str) rows = [` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 57 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `a`
- `abs`
- `actual_abs`
- `actual_signed`
- `all`
- `append`
- `apply`
- `ascending`
- `astype`
- `axis`
- `b`
- `coerce`
- `columns`
- `contain`
- `copy`
- `corr`
- `corr_pairs_df`
- `correlation`
- `DataFrame`

### Outputs

- `actual`
- `df_n`
- `pairs`
- `pear`
- `rows`
- `verify_top_correlations`

### Key Operations

- `def verify_top_correlations( df: pd.DataFrame, *, corr_pairs_df: pd.DataFrame, sensors: list[str], state_col: str = "stream_state", state_value: str = "normal", top_k: int = 25,`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: df_n = df.loc[df[state_col].astype(str) == state_value, sensors].copy() df_n = df_n.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all") pear = df_n.c`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `abs`
- `append`
- `apply`
- `astype`
- `copy`
- `corr`
- `DataFrame`
- `dropna`
- `head`
- `iterrows`
- `sort_values`
- `ValueError`
- `verify_top_correlations`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def verify_top_correlations( df: pd.DataFrame, *, corr_pairs_df: pd.DataFrame, sensors: list[str], state_col: str = "stream_state", state_value: str = "normal", top_k: int = 25,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: df_n = df.loc[df[state_col].astype(str) == state_value, sensors].copy() df_n = df_n.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all") pear = df_n.c` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 58 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__dict__`
- `a`
- `A`
- `abnormal`
- `abnormal_profiles`
- `all`
- `append`
- `are`
- `astype`
- `available`
- `B`
- `best`
- `bool`
- `bounds`
- `bounds_state`
- `Build`
- `buildup`
- `check`
- `Choose`
- `clipping`

### Outputs

- `_profiles_dict_to_df`
- `bound_cols`
- `bounds_check`
- `corr_check`
- `corr_pairs_df`
- `df_check`
- `expected_sensors`
- `ignore_index`
- `missing_report`
- `phase_col`
- `profile_df`
- `r`
- `rows`
- `schema_report`
- `SENSORS`
- `sensors`
- `state_col`
- `state_value`
- `state_values`
- `target_missing_pct`

### Key Operations

- `# ---------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# 0) Choose your sensor list`: Documents the purpose or boundary of the surrounding notebook step.
- `# ---------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `SENSORS = list(generator.sensors) # best source of truth`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# For a dataframe view:`: Documents the purpose or boundary of the surrounding notebook step.
- `# SENSORS = sorted([column for column in synthetic_df.columns if column.startswith("sensor_")])`: Documents the purpose or boundary of the surrounding notebook step.
- `# ---------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `# 1) Schema / state sanity`: Documents the purpose or boundary of the surrounding notebook step.
- `# ---------------------------`: Documents the purpose or boundary of the surrounding notebook step.
- `schema_report = verify_schema( synthetic_df, expected_sensors=SENSORS, state_col="stream_state", phase_col="phase",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Schema report:", schema_report)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `_profiles_dict_to_df`
- `all`
- `append`
- `astype`
- `bool`
- `concat`
- `copy`
- `DataFrame`
- `display`
- `eq`
- `get`
- `getattr`
- `head`
- `isinstance`
- `items`
- `missing`
- `MissingnessSpec`
- `profile_df`
- `sanity`
- `sorted`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 0) Choose your sensor list` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `SENSORS = list(generator.sensors) # best source of truth` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# For a dataframe view:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# SENSORS = sorted([column for column in synthetic_df.columns if column.startswith("sensor_")])` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 1) Schema / state sanity` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `schema_report = verify_schema( synthetic_df, expected_sensors=SENSORS, state_col="stream_state", phase_col="phase",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Schema report:", schema_report)` | Displays a notebook-facing result for inspection. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 2) Missingness sanity` | Documents the purpose or boundary of the surrounding notebook step. |
| `# (GLOBAL target; easiest / most stable)` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Option A: from MissingnessSpec (preferred method)` | Documents the purpose or boundary of the surrounding notebook step. |
| `target_missing_pct = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `if getattr(generator, "missingness_spec", None) is not None: target_missing_pct = dict(generator.missingness_spec.missingness_pct_all)` | Controls validation, iteration, file handling, or error handling for this step. |
| `print("Has missingness_spec?:", getattr(generator, "missingness_spec", None) is not None)` | Displays a notebook-facing result for inspection. |
| `print("target_missing_pct size:", 0 if target_missing_pct is None else len(target_missing_pct))` | Displays a notebook-facing result for inspection. |
| `# Option B: from Silver truth` | Documents the purpose or boundary of the surrounding notebook step. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `target_missing_pct = ( (silver_truth.get("runtime_facts", {}) or {}) .get("missingness_quarantine", {}) .get("missingness_pct_all")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("target_missing_pct size:", 0 if not target_missing_pct else len(target_missing_pct))` | Displays a notebook-facing result for inspection. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if target_missing_pct is None: print("Skipping missingness check: no target_missing_pct available")` | Displays a notebook-facing result for inspection. |
| `else: # Force known all-null sensors to 100% missing (matches Silver PreEDA quarantine reality) if isinstance(target_missing_pct, dict): target_missing_pct = dict(target_missing_pc` | Displays a notebook-facing result for inspection. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 3) Profile bounds sanity` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build one combined profile_df (sensor,state_scope,p01,p99,mean,std,...)` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `def _profiles_dict_to_df(d: dict, state: str) -> pd.DataFrame: rows = [] for sensor, prof in d.items(): # SensorRichProfile is dataclass-like; __dict__ works r = dict(prof.__dict__` | Defines notebook-local logic used later in the notebook. |
| `profile_df = pd.concat( [ _profiles_dict_to_df(normal_profiles, "normal"), _profiles_dict_to_df(abnormal_profiles, "abnormal"), _profiles_dict_to_df(recovery_profiles, "recovery"),` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("profile_df columns:", sorted(profile_df.columns.tolist()))` | Displays a notebook-facing result for inspection. |
| `display(profile_df.head(5))` | Displays a notebook-facing result for inspection. |
| `# --- For verification only: map buildup -> abnormal ---` | Documents the purpose or boundary of the surrounding notebook step. |
| `df_check = synthetic_df.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `df_check["bounds_state"] = df_check["stream_state"].astype(str)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `df_check.loc[df_check["bounds_state"].eq("buildup"), "bounds_state"] = "abnormal"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `bounds_check = verify_profile_bounds( df_check, profile_df=profile_df, sensors=SENSORS, state_col="bounds_state", state_values=["normal", "abnormal", "recovery"], bound_cols=("p01"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(bounds_check.head(50))` | Displays a notebook-facing result for inspection. |
| `# rule of thumb: if pct_outside_bounds is consistently > ~1-3%,` | Documents the purpose or boundary of the surrounding notebook step. |
| `# your generator clipping or distributions are drifting from profile expectations.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 4) Correlation sanity (normal only)` | Documents the purpose or boundary of the surrounding notebook step. |
| `# ---------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `corr_check = verify_top_correlations( synthetic_df, corr_pairs_df=corr_pairs_df, sensors=SENSORS, state_col="stream_state", state_value="normal", top_k=25,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `display(corr_check)` | Displays a notebook-facing result for inspection. |
| `print("Done.")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 59 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `ascending`
- `astype`
- `candidate`
- `copy`
- `dropna`
- `fault_coupling_strength`
- `fault_pairings_df`
- `head`
- `iloc`
- `loc`
- `meta__episode_id`
- `meta__primary_sensor`
- `secondaries`
- `sensor`
- `sensor_primary`
- `show`
- `some`
- `sort_values`
- `stream_state`
- `synthetic_df`

### Outputs

- `ep0`
- `links`
- `primary`

### Key Operations

- `ep0 = synthetic_df.loc[synthetic_df["meta__episode_id"] == 0].copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print(ep0["stream_state"].value_counts())`: Displays a notebook-facing result for inspection.
- `primary = str(ep0["meta__primary_sensor"].dropna().iloc[0])`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("Primary sensor:", primary)`: Displays a notebook-facing result for inspection.
- `# show some candidate secondaries from your fault_pairings_df`: Documents the purpose or boundary of the surrounding notebook step.
- `links = fault_pairings_df.loc[fault_pairings_df["sensor_primary"].astype(str) == primary].copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `display(links.sort_values("fault_coupling_strength", ascending=False).head(10))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `astype`
- `copy`
- `display`
- `dropna`
- `head`
- `sort_values`
- `value_counts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `ep0 = synthetic_df.loc[synthetic_df["meta__episode_id"] == 0].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print(ep0["stream_state"].value_counts())` | Displays a notebook-facing result for inspection. |
| `primary = str(ep0["meta__primary_sensor"].dropna().iloc[0])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Primary sensor:", primary)` | Displays a notebook-facing result for inspection. |
| `# show some candidate secondaries from your fault_pairings_df` | Documents the purpose or boundary of the surrounding notebook step. |
| `links = fault_pairings_df.loc[fault_pairings_df["sensor_primary"].astype(str) == primary].copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `display(links.sort_values("fault_coupling_strength", ascending=False).head(10))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 60 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `after`
- `are`
- `before`
- `but`
- `c`
- `canonical`
- `clean`
- `column`
- `columns`
- `copy`
- `dataframe`
- `Dropped`
- `expected`
- `f`
- `feature`
- `final`
- `full`
- `generated`
- `generator`
- `i`

### Outputs

- `EXPECTED_SENSOR_COLUMNS`
- `missing_pct`
- `missing_sensor_columns`
- `non_sensor_columns`
- `synthetic_df`

### Key Operations

- `'''`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Restore full expected sensor schema`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# The generator may model only the clean feature set, but final synthetic`: Documents the purpose or boundary of the surrounding notebook step.
- `# output should preserve the original wide sensor schema when possible.`: Documents the purpose or boundary of the surrounding notebook step.
- `# Dropped sensors are restored as missing columns unless generated upstream.`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `EXPECTED_SENSOR_COLUMNS = [f"sensor_{i:02d}" for i in range(52)]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `missing_sensor_columns = [ sensor for sensor in EXPECTED_SENSOR_COLUMNS if sensor not in synthetic_df.columns`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `print("Missing sensor columns before schema restore:", missing_sensor_columns)`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `copy`
- `isna`
- `mean`
- `range`
- `startswith`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Restore full expected sensor schema` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# The generator may model only the clean feature set, but final synthetic` | Documents the purpose or boundary of the surrounding notebook step. |
| `# output should preserve the original wide sensor schema when possible.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Dropped sensors are restored as missing columns unless generated upstream.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `EXPECTED_SENSOR_COLUMNS = [f"sensor_{i:02d}" for i in range(52)]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `missing_sensor_columns = [ sensor for sensor in EXPECTED_SENSOR_COLUMNS if sensor not in synthetic_df.columns` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Missing sensor columns before schema restore:", missing_sensor_columns)` | Displays a notebook-facing result for inspection. |
| `for sensor in missing_sensor_columns: synthetic_df[sensor] = np.nan` | Controls validation, iteration, file handling, or error handling for this step. |
| `# Keep sensor columns in canonical order while preserving metadata columns.` | Documents the purpose or boundary of the surrounding notebook step. |
| `non_sensor_columns = [ column for column in synthetic_df.columns if not str(column).startswith("sensor_")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `synthetic_df = synthetic_df[ non_sensor_columns + EXPECTED_SENSOR_COLUMNS` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("Synthetic dataframe shape after schema restore:", synthetic_df.shape)` | Displays a notebook-facing result for inspection. |
| `print("Sensor columns after schema restore:", len([c for c in synthetic_df.columns if str(c).startswith("sensor_")]))` | Displays a notebook-facing result for inspection. |
| `for sensor_name in ["sensor_15", "sensor_50"]: missing_pct = float(synthetic_df[sensor_name].isna().mean() * 100.0) print(f"{sensor_name} missing% after restore: {missing_pct:.2f}"` | Displays a notebook-facing result for inspection. |
| `'''` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 61 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dispose`
- `Dispose`
- `Engine`
- `engine`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `# Dispose Engine`: Documents the purpose or boundary of the surrounding notebook step.
- `engine.dispose()`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `dispose`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Dispose Engine` | Documents the purpose or boundary of the surrounding notebook step. |
| `engine.dispose()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 62 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__`
- `__name__`
- `a`
- `Any`
- `append`
- `astype`
- `auditing`
- `config`
- `CONFIG`
- `config_key`
- `copy`
- `csv`
- `CSV`
- `currently`
- `DataFrame`
- `datetime`
- `def`
- `dtype`
- `dumps`
- `else`

### Outputs

- `_flatten_config_rows`
- `df`
- `export_config_snapshot_csv`
- `file_path`
- `filename`
- `key_str`
- `new_prefix`
- `output_path`
- `rows`

### Key Operations

- `def _flatten_config_rows( obj: Any, prefix: str = "",`: Defines notebook-local logic used later in the notebook.
- `) -> List[Dict[str, Any]]: rows: List[Dict[str, Any]] = [] if isinstance(obj, dict): for key, value in obj.items(): key_str = str(key) new_prefix = f"{prefix}.{key_str}" if prefix `: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def export_config_snapshot_csv( *, config: Dict[str, Any], output_dir: str \| Path, filename_prefix: str = "synthetic_config_snapshot", run_id: Optional[str] = None, extra_settings:`: Defines notebook-local logic used later in the notebook.
- `) -> tuple[pd.DataFrame, str]: """ Export the currently loaded config plus selected resolved runtime settings to a flat CSV for run auditing / upload review. Recommended usage: - c`: Writes an artifact or output used for review or downstream notebooks.

Important functions or methods detected:
- `_flatten_config_rows`
- `append`
- `astype`
- `copy`
- `DataFrame`
- `dumps`
- `export_config_snapshot_csv`
- `extend`
- `isinstance`
- `isoformat`
- `items`
- `mkdir`
- `now`
- `Path`
- `Series`
- `split`
- `strip`
- `to_csv`
- `type`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def _flatten_config_rows( obj: Any, prefix: str = "",` | Defines notebook-local logic used later in the notebook. |
| `) -> List[Dict[str, Any]]: rows: List[Dict[str, Any]] = [] if isinstance(obj, dict): for key, value in obj.items(): key_str = str(key) new_prefix = f"{prefix}.{key_str}" if prefix ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def export_config_snapshot_csv( *, config: Dict[str, Any], output_dir: str \| Path, filename_prefix: str = "synthetic_config_snapshot", run_id: Optional[str] = None, extra_settings:` | Defines notebook-local logic used later in the notebook. |
| `) -> tuple[pd.DataFrame, str]: """ Export the currently loaded config plus selected resolved runtime settings to a flat CSV for run auditing / upload review. Recommended usage: - c` | Writes an artifact or output used for review or downstream notebooks. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 63 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `data_synthetic`
- `get_paths`

### Outputs

- `EXPORT_SETTINGS_DIR`
- `paths`

### Key Operations

- `paths = get_paths()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `EXPORT_SETTINGS_DIR = paths.data_synthetic`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `get_paths`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `paths = get_paths()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `EXPORT_SETTINGS_DIR = paths.data_synthetic` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 64 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `alias`
- `config_snapshot_df`
- `config_snapshot_path`
- `CSV`
- `else`
- `EPISODE_MAX_ROWS`
- `episode_max_rows`
- `export_config_snapshot_csv`
- `EXPORT_SETTINGS_DIR`
- `exported`
- `fault_excluded_sensors`
- `FAULT_EXCLUDED_SENSORS`
- `formatted_datetime`
- `globals`
- `head`
- `hotspot_cluster_count`
- `HOTSPOT_CLUSTERS_FOR_GENERATOR`
- `info`
- `legacy`
- `logger`

### Outputs

- `config`
- `CONFIG_SNAPSHOT_EXTRA`
- `extra_settings`
- `filename_prefix`
- `output_dir`
- `run_id`
- `timestamp`

### Key Operations

- `CONFIG_SNAPSHOT_EXTRA = { "mode": MODE, "target_rows": TARGET_ROWS, "max_episodes": MAX_EPISODES, "rows_per_failure": ROWS_PER_FAILURE, "episode_max_rows": EPISODE_MAX_ROWS, "obser`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `config_snapshot_df, config_snapshot_path = export_config_snapshot_csv( config=CONFIG, # or SYN_CFG if you only want the synthetic section output_dir=EXPORT_SETTINGS_DIR if "EXPORT_`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `logger.info("Config snapshot CSV exported: %s", config_snapshot_path)`: Writes a logger message for traceability during notebook execution.
- `logger.info("Config snapshot rows: %s", len(config_snapshot_df))`: Writes a logger message for traceability during notebook execution.
- `print("Config snapshot CSV:", config_snapshot_path)`: Displays a notebook-facing result for inspection.
- `display(config_snapshot_df.head(50))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `export_config_snapshot_csv`
- `globals`
- `head`
- `info`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `CONFIG_SNAPSHOT_EXTRA = { "mode": MODE, "target_rows": TARGET_ROWS, "max_episodes": MAX_EPISODES, "rows_per_failure": ROWS_PER_FAILURE, "episode_max_rows": EPISODE_MAX_ROWS, "obser` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `config_snapshot_df, config_snapshot_path = export_config_snapshot_csv( config=CONFIG, # or SYN_CFG if you only want the synthetic section output_dir=EXPORT_SETTINGS_DIR if "EXPORT_` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `logger.info("Config snapshot CSV exported: %s", config_snapshot_path)` | Writes a logger message for traceability during notebook execution. |
| `logger.info("Config snapshot rows: %s", len(config_snapshot_df))` | Writes a logger message for traceability during notebook execution. |
| `print("Config snapshot CSV:", config_snapshot_path)` | Displays a notebook-facing result for inspection. |
| `display(config_snapshot_df.head(50))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell contributes to logger-based traceability.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 65 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__`
- `a`
- `ABNORMAL`
- `abs`
- `abs_gap`
- `abs_pct_gap`
- `against`
- `append`
- `asarray`
- `astype`
- `b`
- `BROKEN`
- `broken_count_tolerance`
- `Build`
- `CANDIDATE_STOP`
- `cluster`
- `cluster_abs_gap_pass`
- `cluster_abs_gap_warn`
- `cluster_name`
- `clusters`

### Outputs

- `_cluster_avg_abs_corr`
- `_normalize_status_series`
- `_overall_corr_errors`
- `_pct_missing`
- `_safe_corr`
- `_sensor_cols`
- `_status_label`
- `all_statuses`
- `arr`
- `broken_gap`
- `broken_status`
- `build_synthetic_run_scorecard`
- `c`
- `cluster_df`
- `cluster_rows`
- `cols`
- `compare_two_run_decisions`
- `corr_summary`
- `cr`
- `cs`

### Key Operations

- `DEFAULT_PRIORITY_SENSORS = [ "sensor_15", "sensor_50", "sensor_51", "sensor_06", "sensor_07", "sensor_08", "sensor_09",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `DEFAULT_PRIORITY_PAIRS = [ ("sensor_08", "sensor_09"), ("sensor_14", "sensor_16"), ("sensor_17", "sensor_18"), ("sensor_20", "sensor_21"), ("sensor_22", "sensor_23"), ("sensor_25",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `DEFAULT_PRIORITY_CLUSTERS = [ ["sensor_19", "sensor_20", "sensor_21", "sensor_22", "sensor_23", "sensor_24", "sensor_25"], ["sensor_31", "sensor_32", "sensor_33"], ["sensor_34", "s`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `DEFAULT_THRESHOLDS = { "state_pct_tolerance": 1.0, # percentage points "broken_count_tolerance": 2, # rows "missing_pct_tolerance": 1.0, # percentage points "priority_pair_abs_gap_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `}`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def _normalize_status_series(series: pd.Series) -> pd.Series: s = series.astype("string").str.upper().fillna("UNKNOWN") s = s.replace( { "BROKEN": "BROKEN", "ABNORMAL": "BROKEN", "`: Defines notebook-local logic used later in the notebook.
- `def _sensor_cols(df: pd.DataFrame) -> List[str]: return [c for c in df.columns if str(c).startswith("sensor_")]`: Defines notebook-local logic used later in the notebook.
- `def _pct_missing(df: pd.DataFrame, col: str) -> float: if col not in df.columns or len(df) == 0: return np.nan return float(df[col].isna().mean() * 100.0)`: Defines notebook-local logic used later in the notebook.
- `def _safe_corr(df: pd.DataFrame, a: str, b: str) -> float: if a not in df.columns or b not in df.columns: return np.nan x = pd.to_numeric(df[a], errors="coerce") y = pd.to_numeric(`: Defines notebook-local logic used later in the notebook.

Important functions or methods detected:
- `_cluster_avg_abs_corr`
- `_normalize_status_series`
- `_overall_corr_errors`
- `_pct_missing`
- `_safe_corr`
- `_sensor_cols`
- `_status_label`
- `abs`
- `append`
- `asarray`
- `astype`
- `build_synthetic_run_scorecard`
- `compare_two_run_decisions`
- `concat`
- `copy`
- `corr`
- `DataFrame`
- `enumerate`
- `eq`
- `export_scorecard_bundle`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `DEFAULT_PRIORITY_SENSORS = [ "sensor_15", "sensor_50", "sensor_51", "sensor_06", "sensor_07", "sensor_08", "sensor_09",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DEFAULT_PRIORITY_PAIRS = [ ("sensor_08", "sensor_09"), ("sensor_14", "sensor_16"), ("sensor_17", "sensor_18"), ("sensor_20", "sensor_21"), ("sensor_22", "sensor_23"), ("sensor_25",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DEFAULT_PRIORITY_CLUSTERS = [ ["sensor_19", "sensor_20", "sensor_21", "sensor_22", "sensor_23", "sensor_24", "sensor_25"], ["sensor_31", "sensor_32", "sensor_33"], ["sensor_34", "s` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `DEFAULT_THRESHOLDS = { "state_pct_tolerance": 1.0, # percentage points "broken_count_tolerance": 2, # rows "missing_pct_tolerance": 1.0, # percentage points "priority_pair_abs_gap_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `}` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def _normalize_status_series(series: pd.Series) -> pd.Series: s = series.astype("string").str.upper().fillna("UNKNOWN") s = s.replace( { "BROKEN": "BROKEN", "ABNORMAL": "BROKEN", "` | Defines notebook-local logic used later in the notebook. |
| `def _sensor_cols(df: pd.DataFrame) -> List[str]: return [c for c in df.columns if str(c).startswith("sensor_")]` | Defines notebook-local logic used later in the notebook. |
| `def _pct_missing(df: pd.DataFrame, col: str) -> float: if col not in df.columns or len(df) == 0: return np.nan return float(df[col].isna().mean() * 100.0)` | Defines notebook-local logic used later in the notebook. |
| `def _safe_corr(df: pd.DataFrame, a: str, b: str) -> float: if a not in df.columns or b not in df.columns: return np.nan x = pd.to_numeric(df[a], errors="coerce") y = pd.to_numeric(` | Defines notebook-local logic used later in the notebook. |
| `def _cluster_avg_abs_corr(df: pd.DataFrame, cluster: Sequence[str]) -> float: cols = [c for c in cluster if c in df.columns] if len(cols) < 2: return np.nan vals = [] for i, a in e` | Defines notebook-local logic used later in the notebook. |
| `def _overall_corr_errors( synth_normal: pd.DataFrame, real_normal: pd.DataFrame, sensor_cols: Sequence[str],` | Defines notebook-local logic used later in the notebook. |
| `) -> Dict[str, float]: errs: List[float] = [] cols = [c for c in sensor_cols if c in synth_normal.columns and c in real_normal.columns] for i, a in enumerate(cols): for b in cols[i` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def _status_label(value: float, pass_threshold: float, warn_threshold: float) -> str: if not np.isfinite(value): return "WARN" if value <= pass_threshold: return "PASS" if value <=` | Defines notebook-local logic used later in the notebook. |
| `def build_synthetic_run_scorecard( synthetic_df: pd.DataFrame, reference_df: pd.DataFrame, *, status_col: str = "machine_status", priority_sensors: Optional[Sequence[str]] = None, ` | Defines notebook-local logic used later in the notebook. |
| `) -> Dict[str, pd.DataFrame]: """ Build a scorecard comparing one synthetic run against the reference dataset. """ thresholds = dict(DEFAULT_THRESHOLDS \| (thresholds or {})) priori` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def export_scorecard_bundle( scorecards: Dict[str, pd.DataFrame], *, output_dir: str \| Path, run_id: Optional[str] = None, prefix: str = "synthetic_scorecard",` | Defines notebook-local logic used later in the notebook. |
| `) -> Dict[str, str]: output_dir = Path(output_dir) output_dir.mkdir(parents=True, exist_ok=True) paths: Dict[str, str] = {} for name, df in scorecards.items(): filename = f"{prefix` | Writes an artifact or output used for review or downstream notebooks. |
| `def compare_two_run_decisions( current_decision_df: pd.DataFrame, previous_decision_df: pd.DataFrame,` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: cur = current_decision_df.iloc[0].to_dict() prev = previous_decision_df.iloc[0].to_dict() return pd.DataFrame( [ { "current_run_id": cur.get("run_id"), "previous` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: CSV output.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 66 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `another`
- `com`
- `https`
- `ipynb`
- `jupyter`
- `notebook`
- `questions`
- `Reference`
- `run`
- `running`
- `stackoverflow`
- `synthetic_pipeline_condensed`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `# Reference: https://stackoverflow.com/questions/49817409/running-a-jupyter-notebook-from-another-notebook`: Documents the purpose or boundary of the surrounding notebook step.
- `%run ./synthetic_pipeline_condensed-02_03.ipynb`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Reference: https://stackoverflow.com/questions/49817409/running-a-jupyter-notebook-from-another-notebook` | Documents the purpose or boundary of the surrounding notebook step. |
| `%run ./synthetic_pipeline_condensed-02_03.ipynb` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

