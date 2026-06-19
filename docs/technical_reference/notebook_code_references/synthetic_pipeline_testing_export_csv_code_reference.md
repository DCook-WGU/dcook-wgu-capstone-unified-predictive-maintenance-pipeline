# Notebook Code Reference: synthetic_pipeline_testing_export_csv

Notebook path:

`notebooks/synthetic/synthetic_pipeline_testing_export_csv.ipynb`

## Status

Testing / support reference.

This notebook is documented because it remains in the repository and supports focused CSV export testing for the synthetic pipeline. It should not be treated as the preferred end-to-end synthetic workflow unless it is explicitly selected for testing or troubleshooting.

## Notebook Purpose

This notebook documents and runs a focused support step in the synthetic pump telemetry workflow.

Notebook stage:

`Synthetic support`

## Section Map

| Notebook Section | Related Code Cells |
|---|---|
| Code Reference | Code Cell 01, Code Cell 02, Code Cell 03, Code Cell 04, Code Cell 05, Code Cell 06, Code Cell 07, Code Cell 08, Code Cell 09, Code Cell 10, Code Cell 11, Code Cell 12, Code Cell 13, Code Cell 14, Code Cell 15, Code Cell 16, Code Cell 17, Code Cell 18, Code Cell 19, Code Cell 20, Code Cell 21, Code Cell 22, Code Cell 23, Code Cell 24, Code Cell 25, Code Cell 26, Code Cell 27, Code Cell 28, Code Cell 29, Code Cell 30, Code Cell 31, Code Cell 32, Code Cell 33, Code Cell 34, Code Cell 35, Code Cell 36 |

## Code Cell 01 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__future__`
- `annotations`
- `Any`
- `build_truth_config_block`
- `combinations`
- `config_loader`
- `core`
- `database`
- `datetime`
- `Engine`
- `engine`
- `export_config_snapshot`
- `file_io`
- `get_engine_from_env`
- `get_paths`
- `itertools`
- `load_data`
- `load_pipeline_config`
- `numpy`
- `Optional`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `from __future__ import annotations`: Imports a dependency or project helper used by later cells.
- `import os`: Imports a dependency or project helper used by later cells.
- `from pathlib import Path`: Imports a dependency or project helper used by later cells.
- `from typing import Any, List, Optional, Sequence`: Imports a dependency or project helper used by later cells.
- `from itertools import combinations`: Imports a dependency or project helper used by later cells.
- `import pandas as pd`: Imports a dependency or project helper used by later cells.
- `import numpy as np`: Imports a dependency or project helper used by later cells.
- `from sqlalchemy.engine import Engine`: Imports a dependency or project helper used by later cells.
- `from datetime import datetime, timedelta`: Imports a dependency or project helper used by later cells.
- `from utils.database.postgres import ( sanitize_sql_identifier, read_sql_dataframe, get_engine_from_env,`: Imports a dependency or project helper used by later cells.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `from utils.core.file_io import ( save_data, load_data,`: Imports a dependency or project helper used by later cells.

Important functions or methods detected:
- `import`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `from __future__ import annotations` | Imports a dependency or project helper used by later cells. |
| `import os` | Imports a dependency or project helper used by later cells. |
| `from pathlib import Path` | Imports a dependency or project helper used by later cells. |
| `from typing import Any, List, Optional, Sequence` | Imports a dependency or project helper used by later cells. |
| `from itertools import combinations` | Imports a dependency or project helper used by later cells. |
| `import pandas as pd` | Imports a dependency or project helper used by later cells. |
| `import numpy as np` | Imports a dependency or project helper used by later cells. |
| `from sqlalchemy.engine import Engine` | Imports a dependency or project helper used by later cells. |
| `from datetime import datetime, timedelta` | Imports a dependency or project helper used by later cells. |
| `from utils.database.postgres import ( sanitize_sql_identifier, read_sql_dataframe, get_engine_from_env,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.file_io import ( save_data, load_data,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `from utils.core.paths import get_paths` | Imports a dependency or project helper used by later cells. |
| `from utils.core.config_loader import ( load_pipeline_config, build_truth_config_block, set_wandb_dir_from_config, export_config_snapshot,` | Imports a dependency or project helper used by later cells. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 02 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `CAPSTONE_SCHEMA`
- `configs`
- `data`
- `data_synthetic_dir`
- `default`
- `Get`
- `get_paths`
- `getenv`
- `load_pipeline_config`
- `Object`
- `observation_timestamp`
- `os`
- `pump`
- `pump_asset_001`
- `pump_synthetic_v1`
- `resolved_paths`
- `root`
- `RUN_ID`
- `s`
- `SCHEMA`

### Outputs

- `ASSET_ID`
- `CHUNK_SIZE`
- `CONFIG`
- `CONFIG_DATASET`
- `CONFIG_PROFILE`
- `CONFIG_ROOT`
- `config_root`
- `CONFIG_RUN_MODE`
- `CONFIG_STAGE`
- `dataset`
- `DATASET_ID`
- `FILENAMES`
- `mode`
- `ORDER_BY`
- `OUTPUT_BASE_FILE_NAME`
- `OUTPUT_DIRECTORY`
- `PATHS`
- `paths`
- `profile`
- `project_root`

### Key Operations

- `# Get Path's Object`: Documents the purpose or boundary of the surrounding notebook step.
- `paths = get_paths()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `CONFIG_ROOT = paths.configs`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `CONFIG_STAGE = "synthetic"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_DATASET = "pump"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_RUN_MODE = "train"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `CONFIG_PROFILE = "default"`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####`: Documents the purpose or boundary of the surrounding notebook step.
- `CONFIG = load_pipeline_config( config_root=CONFIG_ROOT, stage=CONFIG_STAGE, dataset=CONFIG_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, project_root=paths.root,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `).data`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `get_paths`
- `getenv`
- `load_pipeline_config`
- `Path`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# Get Path's Object` | Documents the purpose or boundary of the surrounding notebook step. |
| `paths = get_paths()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `CONFIG_ROOT = paths.configs` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `CONFIG_STAGE = "synthetic"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_DATASET = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_RUN_MODE = "train"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CONFIG_PROFILE = "default"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####` | Documents the purpose or boundary of the surrounding notebook step. |
| `CONFIG = load_pipeline_config( config_root=CONFIG_ROOT, stage=CONFIG_STAGE, dataset=CONFIG_DATASET, mode=CONFIG_RUN_MODE, profile=CONFIG_PROFILE, project_root=paths.root,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `).data` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `SYN_CFG = CONFIG["synthetic"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `PATHS = CONFIG["resolved_paths"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `FILENAMES = CONFIG["filenames"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SYNTHETIC_DATA_PATH = Path(PATHS["data_synthetic_dir"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SCHEMA = os.getenv("CAPSTONE_SCHEMA", "synthetic_run_001")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `DATASET_ID = os.getenv("SYNTHETIC_DATASET_ID", "pump_synthetic_v1")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `RUN_ID = os.getenv("SYNTHETIC_RUN_ID", "synthetic_run_001")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ASSET_ID = os.getenv("SYNTHETIC_ASSET_ID", "pump_asset_001")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `CHUNK_SIZE = 100_000` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `ORDER_BY = "observation_timestamp"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SOURCE_TABLE_NAME = "synthetic_observations_timestamped_stage"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `OUTPUT_DIRECTORY = SYNTHETIC_DATA_PATH` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `OUTPUT_BASE_FILE_NAME = "synthetic_timestamped_export"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 03 — Code Reference

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

## Code Cell 04 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `_part_`
- `a`
- `abnormal`
- `append`
- `artifacts`
- `asset_id`
- `base_file_name`
- `broken`
- `BROKEN`
- `BY`
- `capstone`
- `CASE`
- `chunk_size`
- `COALESCE`
- `continue`
- `COUNT`
- `created_files`
- `CSV`
- `csv`

### Outputs

- `chunk_df`
- `count_sql`
- `export_synthetic_table_to_csv_parts`
- `index`
- `order_clause`
- `output_dir`
- `output_path`
- `part_file_name`
- `part_number`
- `safe_order_by`
- `safe_schema`
- `safe_table`
- `sensor_columns`
- `sensor_select_sql`
- `sql`
- `total_rows`
- `total_rows_df`

### Key Operations

- `def export_synthetic_table_to_csv_parts( engine, *, schema: str = "capstone", table_name: str = "synthetic_observations_timestamped_stage", output_dir: str \| Path = "/workspace/art`: Defines notebook-local logic used later in the notebook.
- `) -> List[Path]: """ Read a synthetic Postgres table using the requested projection/machine_status mapping and export the result into multiple CSV parts. Returns ------- List[Path]`: Loads input data, configuration, or artifacts required by the current stage.

Important functions or methods detected:
- `append`
- `COALESCE`
- `COUNT`
- `export_synthetic_table_to_csv_parts`
- `join`
- `LOWER`
- `mkdir`
- `Path`
- `range`
- `read_sql_dataframe`
- `sanitize_sql_identifier`
- `save_data`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def export_synthetic_table_to_csv_parts( engine, *, schema: str = "capstone", table_name: str = "synthetic_observations_timestamped_stage", output_dir: str \| Path = "/workspace/art` | Defines notebook-local logic used later in the notebook. |
| `) -> List[Path]: """ Read a synthetic Postgres table using the requested projection/machine_status mapping and export the result into multiple CSV parts. Returns ------- List[Path]` | Loads input data, configuration, or artifacts required by the current stage. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: CSV output.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 05 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `get_engine_from_env`

### Outputs

- `engine`

### Key Operations

- `e`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `get_engine_from_env`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `e` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 06 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `at`
- `Export`
- `f`
- `formatted_datetime`
- `Starting`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `p`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `p` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 07 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `engine`
- `export_synthetic_table_to_csv_parts`
- `formatted_datetime`
- `OUTPUT_BASE_FILE_NAME`
- `OUTPUT_DIRECTORY`
- `SOURCE_TABLE_NAME`

### Outputs

- `base_file_name`
- `chunk_size`
- `exported_files`
- `order_by`
- `output_dir`
- `schema`
- `table_name`
- `timestamp`

### Key Operations

- `exported_files = export_synthetic_table_to_csv_parts( engine, schema=SCHEMA, table_name=SOURCE_TABLE_NAME, output_dir=OUTPUT_DIRECTORY, base_file_name=OUTPUT_BASE_FILE_NAME, chunk_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `exported_files`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `export_synthetic_table_to_csv_parts`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `exported_files = export_synthetic_table_to_csv_parts( engine, schema=SCHEMA, table_name=SOURCE_TABLE_NAME, output_dir=OUTPUT_DIRECTORY, base_file_name=OUTPUT_BASE_FILE_NAME, chunk_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `exported_files` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 08 — Code Reference

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

## Code Cell 09 — Code Reference

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

## Code Cell 10 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `at`
- `completed`
- `export`
- `exported`
- `f`
- `files`
- `formatted_datetime`
- `Run`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `p`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `p` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 11 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_`
- `a`
- `abnormal`
- `artifacts`
- `asset_id`
- `base_file_name`
- `BROKEN`
- `broken`
- `BY`
- `capstone`
- `CASE`
- `COALESCE`
- `COUNT`
- `CSV`
- `csv`
- `d_`
- `dataset_id`
- `datetime`
- `def`
- `else`

### Outputs

- `count_sql`
- `export_df`
- `export_synthetic_table_to_csv`
- `file_name`
- `index`
- `order_clause`
- `output_dir`
- `output_path`
- `safe_order_by`
- `safe_schema`
- `safe_table`
- `sensor_columns`
- `sensor_select_sql`
- `sql`
- `timestamp`
- `total_rows`
- `total_rows_df`

### Key Operations

- `def export_synthetic_table_to_csv( engine, *, schema: str = "capstone", table_name: str = "synthetic_observations_timestamped_stage", output_dir: str \| Path = "/workspace/artifacts`: Defines notebook-local logic used later in the notebook.
- `) -> Path \| None: """ Read a synthetic Postgres table using the requested projection/machine_status mapping and export the result into a single CSV file. Returns ------- Path \| Non`: Loads input data, configuration, or artifacts required by the current stage.

Important functions or methods detected:
- `COALESCE`
- `COUNT`
- `export_synthetic_table_to_csv`
- `join`
- `LOWER`
- `mkdir`
- `Path`
- `range`
- `read_sql_dataframe`
- `sanitize_sql_identifier`
- `save_data`
- `strftime`
- `utcnow`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def export_synthetic_table_to_csv( engine, *, schema: str = "capstone", table_name: str = "synthetic_observations_timestamped_stage", output_dir: str \| Path = "/workspace/artifacts` | Defines notebook-local logic used later in the notebook. |
| `) -> Path \| None: """ Read a synthetic Postgres table using the requested projection/machine_status mapping and export the result into a single CSV file. Returns ------- Path \| Non` | Loads input data, configuration, or artifacts required by the current stage. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: CSV output.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 12 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `artifacts`
- `capstone`
- `engine`
- `export_synthetic_table_to_csv`
- `exports`
- `formatted_datetime`
- `synthetic_observations_timestamped_stage`
- `synthetic_timestamped_export_full_`
- `workspace`

### Outputs

- `base_file_name`
- `export_path`
- `output_dir`
- `schema`
- `table_name`
- `timestamp`

### Key Operations

- `export_path = export_synthetic_table_to_csv( engine, schema="capstone", table_name="synthetic_observations_timestamped_stage", output_dir="/workspace/artifacts/exports", base_file_`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `export_synthetic_table_to_csv`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `export_path = export_synthetic_table_to_csv( engine, schema="capstone", table_name="synthetic_observations_timestamped_stage", output_dir="/workspace/artifacts/exports", base_file_` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: CSV output.

## Code Cell 13 — Code Reference

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

## Code Cell 14 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `at`
- `f`
- `formatted_datetime`
- `scorecard`
- `starting`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `p`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `p` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 15 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `get_engine_from_env`

### Outputs

- `engine`

### Key Operations

- `e`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `get_engine_from_env`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `e` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 16 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `Any`
- `bool`
- `Bronze`
- `but`
- `column`
- `columns`
- `comparison`
- `copy`
- `CSV`
- `data_raw`
- `def`
- `drop`
- `drop_unnamed_index_columns`
- `else`
- `file_io`
- `file_name`
- `get_paths`
- `Load`
- `load_data`

### Outputs

- `dataframe`
- `file_dir`
- `load_source_dataframe_for_comparison`
- `paths`
- `unnamed_columns`
- `valid_sort_columns`

### Key Operations

- `def load_source_dataframe_for_comparison( *, file_name: str, file_dir: Optional[str \| Path] = None, use_data_raw_dir: bool = True, drop_unnamed_index_columns: bool = True, rename_c`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """ Load a source CSV/Parquet for comparison using the project file_io pattern, but WITHOUT Bronze metadata stamping. """ paths = get_paths() if file_dir is None`: Loads input data, configuration, or artifacts required by the current stage.

Important functions or methods detected:
- `copy`
- `drop`
- `get_paths`
- `load_data`
- `load_source_dataframe_for_comparison`
- `lower`
- `rename`
- `reset_index`
- `sort_values`
- `startswith`
- `strip`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def load_source_dataframe_for_comparison( *, file_name: str, file_dir: Optional[str \| Path] = None, use_data_raw_dir: bool = True, drop_unnamed_index_columns: bool = True, rename_c` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Load a source CSV/Parquet for comparison using the project file_io pattern, but WITHOUT Bronze metadata stamping. """ paths = get_paths() if file_dir is None` | Loads input data, configuration, or artifacts required by the current stage. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 17 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `Any`
- `BY`
- `column`
- `comparison`
- `copy`
- `def`
- `else`
- `f`
- `get_engine_from_env`
- `into`
- `join`
- `LIMIT`
- `limit`
- `Optional`
- `ORDER`
- `order_by`
- `Postgres`
- `Read`
- `read_sql_dataframe`

### Outputs

- `cleaned_columns`
- `cleaned_order`
- `dataframe`
- `engine`
- `params`
- `read_postgres_table_for_comparison`
- `safe_schema`
- `safe_table`
- `select_sql`
- `sql`

### Key Operations

- `def read_postgres_table_for_comparison( *, schema: str, table_name: str, selected_columns: Optional[Sequence[str]] = None, where_sql: Optional[str] = None, order_by: Optional[Seque`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """ Read a Postgres table into a dataframe for comparison work. """ if engine is None: engine = get_engine_from_env() safe_schema = sanitize_sql_identifier(schem`: Loads input data, configuration, or artifacts required by the current stage.

Important functions or methods detected:
- `copy`
- `get_engine_from_env`
- `join`
- `read_postgres_table_for_comparison`
- `read_sql_dataframe`
- `sanitize_sql_identifier`
- `strip`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def read_postgres_table_for_comparison( *, schema: str, table_name: str, selected_columns: Optional[Sequence[str]] = None, where_sql: Optional[str] = None, order_by: Optional[Seque` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Read a Postgres table into a dataframe for comparison work. """ if engine is None: engine = get_engine_from_env() safe_schema = sanitize_sql_identifier(schem` | Loads input data, configuration, or artifacts required by the current stage. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 18 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abnormal`
- `Any`
- `astype`
- `be`
- `BROKEN`
- `broken`
- `BY`
- `by`
- `capstone`
- `CASE`
- `COALESCE`
- `coerce`
- `column`
- `columns`
- `comparison`
- `copy`
- `def`
- `derived`
- `ELSE`
- `END`

### Outputs

- `cleaned_order`
- `dataframe`
- `engine`
- `params`
- `read_synthetic_comparison_projection_dataframe`
- `safe_schema`
- `safe_table`
- `sensor_columns`
- `sensor_select_sql`
- `sql`

### Key Operations

- `def read_synthetic_comparison_projection_dataframe( *, schema: str = "capstone", table_name: str = "synthetic_observations_timestamped_stage", sensor_columns: Optional[Sequence[str`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """ Read the synthetic Postgres table into the same comparison-ready shape used by the multipart export logic: - timestamp - sensor_* columns - machine_status (d`: Loads input data, configuration, or artifacts required by the current stage.

Important functions or methods detected:
- `astype`
- `COALESCE`
- `copy`
- `get_engine_from_env`
- `join`
- `LOWER`
- `machine_status`
- `range`
- `read_sql_dataframe`
- `read_synthetic_comparison_projection_dataframe`
- `sanitize_sql_identifier`
- `strip`
- `to_datetime`
- `upper`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def read_synthetic_comparison_projection_dataframe( *, schema: str = "capstone", table_name: str = "synthetic_observations_timestamped_stage", sensor_columns: Optional[Sequence[str` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Read the synthetic Postgres table into the same comparison-ready shape used by the multipart export logic: - timestamp - sensor_* columns - machine_status (d` | Loads input data, configuration, or artifacts required by the current stage. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.
- This cell may read from or write to PostgreSQL or SQL-facing helpers.

## Code Cell 19 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `a`
- `append`
- `astype`
- `coerce`
- `cols`
- `column`
- `columns`
- `comparison`
- `copy`
- `dataframe`
- `DataFrame`
- `def`
- `errors`
- `extend`
- `identity`
- `loc`
- `machine_status`
- `Optional`
- `ready`
- `Reduce`

### Outputs

- `build_kaggle_style_comparison_frame`
- `keep_identity_columns`
- `out`
- `sensor_columns`

### Key Operations

- `def build_kaggle_style_comparison_frame( dataframe: pd.DataFrame, *, timestamp_column: str = "timestamp", status_column: str = "machine_status", sensor_prefix: str = "sensor_", kee`: Defines notebook-local logic used later in the notebook.
- `) -> pd.DataFrame: """ Reduce a dataframe to a comparison-ready shape: identity cols + timestamp + sensors + machine_status. """ keep_identity_columns = list(keep_identity_columns `: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `append`
- `astype`
- `build_kaggle_style_comparison_frame`
- `copy`
- `extend`
- `sorted`
- `startswith`
- `strip`
- `to_datetime`
- `upper`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_kaggle_style_comparison_frame( dataframe: pd.DataFrame, *, timestamp_column: str = "timestamp", status_column: str = "machine_status", sensor_prefix: str = "sensor_", kee` | Defines notebook-local logic used later in the notebook. |
| `) -> pd.DataFrame: """ Reduce a dataframe to a comparison-ready shape: identity cols + timestamp + sensors + machine_status. """ keep_identity_columns = list(keep_identity_columns ` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 20 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `abs`
- `append`
- `asarray`
- `cluster`
- `column`
- `columns`
- `combinations`
- `corr`
- `DataFrame`
- `dataframe`
- `def`
- `dropna`
- `dtype`
- `isfinite`
- `mean`
- `mean_abs_diff`
- `median`
- `median_abs_diff`
- `nan`
- `p90_abs_diff`

### Outputs

- `_cluster_average_correlation`
- `_get_sensor_columns`
- `_global_correlation_error`
- `_safe_corr`
- `common_sensors`
- `corr_value`
- `diffs`
- `diffs_array`
- `sensors`
- `src_corr`
- `subset`
- `syn_corr`
- `values`

### Key Operations

- `def _get_sensor_columns( dataframe: pd.DataFrame, *, sensor_prefix: str = "sensor_",`: Defines notebook-local logic used later in the notebook.
- `) -> list[str]: return sorted([column for column in dataframe.columns if str(column).startswith(sensor_prefix)])`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def _safe_corr( dataframe: pd.DataFrame, sensor_a: str, sensor_b: str,`: Defines notebook-local logic used later in the notebook.
- `) -> float: if sensor_a not in dataframe.columns or sensor_b not in dataframe.columns: return np.nan subset = dataframe[[sensor_a, sensor_b]].dropna() if len(subset) < 3: return np`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def _cluster_average_correlation( dataframe: pd.DataFrame, cluster: Sequence[str],`: Defines notebook-local logic used later in the notebook.
- `) -> float: sensors = [sensor for sensor in cluster if sensor in dataframe.columns] if len(sensors) < 2: return np.nan values = [] for sensor_a, sensor_b in combinations(sensors, 2`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `def _global_correlation_error( source_df: pd.DataFrame, synthetic_df: pd.DataFrame, *, sensor_columns: Sequence[str],`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, float]: common_sensors = [sensor for sensor in sensor_columns if sensor in source_df.columns and sensor in synthetic_df.columns] if len(common_sensors) < 2: return {`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `_cluster_average_correlation`
- `_get_sensor_columns`
- `_global_correlation_error`
- `_safe_corr`
- `abs`
- `append`
- `asarray`
- `combinations`
- `corr`
- `dropna`
- `isfinite`
- `mean`
- `median`
- `percentile`
- `sorted`
- `startswith`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def _get_sensor_columns( dataframe: pd.DataFrame, *, sensor_prefix: str = "sensor_",` | Defines notebook-local logic used later in the notebook. |
| `) -> list[str]: return sorted([column for column in dataframe.columns if str(column).startswith(sensor_prefix)])` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def _safe_corr( dataframe: pd.DataFrame, sensor_a: str, sensor_b: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> float: if sensor_a not in dataframe.columns or sensor_b not in dataframe.columns: return np.nan subset = dataframe[[sensor_a, sensor_b]].dropna() if len(subset) < 3: return np` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def _cluster_average_correlation( dataframe: pd.DataFrame, cluster: Sequence[str],` | Defines notebook-local logic used later in the notebook. |
| `) -> float: sensors = [sensor for sensor in cluster if sensor in dataframe.columns] if len(sensors) < 2: return np.nan values = [] for sensor_a, sensor_b in combinations(sensors, 2` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `def _global_correlation_error( source_df: pd.DataFrame, synthetic_df: pd.DataFrame, *, sensor_columns: Sequence[str],` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, float]: common_sensors = [sensor for sensor in sensor_columns if sensor in source_df.columns and sensor in synthetic_df.columns] if len(common_sensors) < 2: return {` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 21 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `_cluster_average_correlation`
- `_get_sensor_columns`
- `_global_correlation_error`
- `_safe_corr`
- `a`
- `abs`
- `abs_diff_vs_source`
- `across`
- `append`
- `astype`
- `cluster_19_25`
- `cluster_31_33`
- `cluster_34_36`
- `cluster_name`
- `cluster_sensors`
- `clusters`
- `column`
- `columns`
- `Compare`
- `continue`

### Outputs

- `build_synthetic_vs_source_scorecard`
- `cluster_rows`
- `focus_clusters`
- `focus_clusters_df`
- `focus_pairs`
- `focus_pairs_df`
- `focus_sensors`
- `global_correlation_error_df`
- `global_error`
- `global_error_row`
- `missingness_df`
- `missingness_rows`
- `pair_rows`
- `prev_avg`
- `prev_corr`
- `prev_count`
- `prev_global_error`
- `prev_missing`
- `prev_n`
- `previous_synthetic_status_column`

### Key Operations

- `def build_synthetic_vs_source_scorecard( *, source_df: pd.DataFrame, synthetic_df: pd.DataFrame, previous_synthetic_df: Optional[pd.DataFrame] = None, source_status_column: str = "`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, pd.DataFrame]: """ Compare source vs synthetic and return a scorecard dict of DataFrames. Supports different status-column names across source/synthetic/previous syn`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `_cluster_average_correlation`
- `_get_sensor_columns`
- `_global_correlation_error`
- `_safe_corr`
- `abs`
- `append`
- `astype`
- `build_synthetic_vs_source_scorecard`
- `DataFrame`
- `dropna`
- `isfinite`
- `isna`
- `items`
- `max`
- `mean`
- `sorted`
- `sum`
- `union`
- `unique`
- `upper`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def build_synthetic_vs_source_scorecard( *, source_df: pd.DataFrame, synthetic_df: pd.DataFrame, previous_synthetic_df: Optional[pd.DataFrame] = None, source_status_column: str = "` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, pd.DataFrame]: """ Compare source vs synthetic and return a scorecard dict of DataFrames. Supports different status-column names across source/synthetic/previous syn` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 22 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `csv`
- `load_source_dataframe_for_comparison`
- `pump_sensor_data`
- `sensor`

### Outputs

- `file_name`
- `source_df`
- `use_data_raw_dir`

### Key Operations

- `source_df = load_source_dataframe_for_comparison( file_name="pump_sensor_data/sensor.csv", use_data_raw_dir=True,`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `load_source_dataframe_for_comparison`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `source_df = load_source_dataframe_for_comparison( file_name="pump_sensor_data/sensor.csv", use_data_raw_dir=True,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 23 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `capstone`
- `read_synthetic_comparison_projection_dataframe`
- `synthetic_observations_timestamped_stage`

### Outputs

- `schema`
- `synthetic_df`
- `table_name`

### Key Operations

- `synthetic_df = read_synthetic_comparison_projection_dataframe( schema="capstone", table_name="synthetic_observations_timestamped_stage",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `read_synthetic_comparison_projection_dataframe`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `synthetic_df = read_synthetic_comparison_projection_dataframe( schema="capstone", table_name="synthetic_observations_timestamped_stage",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 24 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `build_kaggle_style_comparison_frame`
- `copy`
- `machine_status`
- `source_df`
- `synthetic_df`
- `timestamp`

### Outputs

- `source_compare_df`
- `status_column`
- `synthetic_compare_df`
- `timestamp_column`

### Key Operations

- `source_compare_df = build_kaggle_style_comparison_frame( source_df, timestamp_column="timestamp", status_column="machine_status",`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `)`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `synthetic_compare_df = synthetic_df.copy()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.

Important functions or methods detected:
- `build_kaggle_style_comparison_frame`
- `copy`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `source_compare_df = build_kaggle_style_comparison_frame( source_df, timestamp_column="timestamp", status_column="machine_status",` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `synthetic_compare_df = synthetic_df.copy()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 25 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__silver_subsets__profiled_dataframe`
- `a`
- `After`
- `against`
- `All`
- `artifact`
- `artifacts`
- `astype`
- `be`
- `Build`
- `build_kaggle_style_comparison_frame`
- `buildup`
- `candidates`
- `cell`
- `cells`
- `Checked`
- `clean`
- `columns`
- `common`
- `Common`

### Outputs

- `artifact_paths`
- `candidate`
- `candidate_paths`
- `DATASET_NAME`
- `dataset_name`
- `direct_candidates`
- `full_kaggle_normal_compare_df`
- `latest_row`
- `layer_name`
- `matches`
- `normal_clean_source_compare_df`
- `normal_clean_source_df`
- `raw_path`
- `resolve_latest_truth_record`
- `resolve_silver_profiled_dataframe_path`
- `search_matches`
- `silver_profiled_df`
- `silver_subsets_artifact_dir`
- `SILVER_SUBSETS_LAYER_NAME`
- `SILVER_SUBSETS_PROFILED_PATH`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build validation target subsets`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# After clean-normal profiling, synthetic NORMAL should be judged primarily`: Documents the purpose or boundary of the surrounding notebook step.
- `# against Silver normal_clean, not the full Kaggle NORMAL population.`: Documents the purpose or boundary of the surrounding notebook step.
- `# Full Kaggle NORMAL remains a diagnostic comparison only.`: Documents the purpose or boundary of the surrounding notebook step.
- `#`: Documents the purpose or boundary of the surrounding notebook step.
- `# This cell resolves the Silver profiled dataframe from:`: Documents the purpose or boundary of the surrounding notebook step.
- `# 1. latest silver_subsets truth record`: Documents the purpose or boundary of the surrounding notebook step.
- `# 2. common direct artifact paths`: Documents the purpose or boundary of the surrounding notebook step.
- `# 3. recursive artifact search`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.

Important functions or methods detected:
- `astype`
- `build_kaggle_style_comparison_frame`
- `copy`
- `DataFrame`
- `eq`
- `exists`
- `FileNotFoundError`
- `get`
- `isin`
- `join`
- `load`
- `lower`
- `open`
- `Path`
- `read_json`
- `read_parquet`
- `resolve_latest_truth_record`
- `resolve_silver_profiled_dataframe_path`
- `rglob`
- `sorted`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build validation target subsets` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# After clean-normal profiling, synthetic NORMAL should be judged primarily` | Documents the purpose or boundary of the surrounding notebook step. |
| `# against Silver normal_clean, not the full Kaggle NORMAL population.` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Full Kaggle NORMAL remains a diagnostic comparison only.` | Documents the purpose or boundary of the surrounding notebook step. |
| `#` | Documents the purpose or boundary of the surrounding notebook step. |
| `# This cell resolves the Silver profiled dataframe from:` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 1. latest silver_subsets truth record` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 2. common direct artifact paths` | Documents the purpose or boundary of the surrounding notebook step. |
| `# 3. recursive artifact search` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `DATASET_NAME = "pump"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `SILVER_SUBSETS_LAYER_NAME = "silver_subsets"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `silver_subsets_artifact_dir = ( paths.artifacts / SILVER_SUBSETS_LAYER_NAME / DATASET_NAME` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `truth_index_path = paths.artifacts / "truths" / "truth_index.jsonl"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `truth_dir = paths.artifacts / "truths"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `def resolve_latest_truth_record( *, truth_index_path: Path, truth_dir: Path, layer_name: str, dataset_name: str,` | Defines notebook-local logic used later in the notebook. |
| `) -> dict \| None: """ Resolve the latest truth record for a layer/dataset from truth_index.jsonl. Returns None if unavailable. """ if not truth_index_path.exists(): print(f"WARNING` | Loads input data, configuration, or artifacts required by the current stage. |
| `def resolve_silver_profiled_dataframe_path() -> Path: """ Resolve profiled dataframe path for validation. """ # ----------------------------------------------------- # 1. Truth-fir` | Defines notebook-local logic used later in the notebook. |
| `SILVER_SUBSETS_PROFILED_PATH = resolve_silver_profiled_dataframe_path()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `silver_profiled_df = pd.read_parquet(SILVER_SUBSETS_PROFILED_PATH)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STATE_COL_PROFILED = "machine_status__profiled"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `STATE_COL_SYNTHETIC = "machine_status"` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("SILVER_SUBSETS_PROFILED_PATH:", SILVER_SUBSETS_PROFILED_PATH)` | Displays a notebook-facing result for inspection. |
| `print("silver_profiled_df shape:", silver_profiled_df.shape)` | Displays a notebook-facing result for inspection. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Source target: clean normal` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `normal_clean_source_df = silver_profiled_df.loc[ silver_profiled_df[STATE_COL_PROFILED] .astype(str) .str.lower() .eq("normal_clean")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `normal_clean_source_compare_df = build_kaggle_style_comparison_frame( normal_clean_source_df, timestamp_column="timestamp", status_column=STATE_COL_PROFILED,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# Give the scorecard a comparable state label.` | Documents the purpose or boundary of the surrounding notebook step. |
| `normal_clean_source_compare_df["machine_status"] = "NORMAL"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Source target: suspect / contaminated normal` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `suspect_normal_source_df = silver_profiled_df.loc[ silver_profiled_df[STATE_COL_PROFILED] .astype(str) .str.lower() .isin(["normal_suspect", "normal_contaminated"])` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `suspect_normal_source_compare_df = build_kaggle_style_comparison_frame( suspect_normal_source_df, timestamp_column="timestamp", status_column=STATE_COL_PROFILED,` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `)` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `suspect_normal_source_compare_df["machine_status"] = "NORMAL"` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Diagnostic target: full Kaggle NORMAL` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `full_kaggle_normal_compare_df = source_compare_df.loc[ source_compare_df["machine_status"] .astype(str) .str.upper() .eq("NORMAL")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Synthetic NORMAL` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `synthetic_normal_compare_df = synthetic_compare_df.loc[ synthetic_compare_df[STATE_COL_SYNTHETIC] .astype(str) .str.upper() .eq("NORMAL")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `].copy()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Synthetic buildup phase` | Documents the purpose or boundary of the surrounding notebook step. |
| `# -----------------------------` | Documents the purpose or boundary of the surrounding notebook step. |
| `if "phase" in synthetic_compare_df.columns: synthetic_buildup_compare_df = synthetic_compare_df.loc[ synthetic_compare_df["phase"] .astype(str) .str.lower() .eq("buildup") ].copy()` | Controls validation, iteration, file handling, or error handling for this step. |
| `else: synthetic_buildup_compare_df = pd.DataFrame()` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `print("normal_clean_source_compare_df:", normal_clean_source_compare_df.shape)` | Displays a notebook-facing result for inspection. |
| `4 additional statements` | Additional statements continue the same notebook step; review the notebook for exact code order. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 26 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `another`
- `comparison`
- `export`
- `load`
- `run`
- `synthetic`
- `table`
- `vs`
- `want`
- `you`

### Outputs

- `previous_synthetic_compare_df`

### Key Operations

- `previous_synthetic_compare_df = None`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `# or load another synthetic table/export if you want run-vs-run comparison`: Documents the purpose or boundary of the surrounding notebook step.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `previous_synthetic_compare_df = None` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `# or load another synthetic table/export if you want run-vs-run comparison` | Documents the purpose or boundary of the surrounding notebook step. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 27 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `columns`
- `nsynthetic_compare_df`
- `source_compare_df`
- `synthetic_compare_df`
- `tolist`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `print("source_compare_df columns:")`: Displays a notebook-facing result for inspection.
- `print(source_compare_df.columns.tolist())`: Displays a notebook-facing result for inspection.
- `print("\nsynthetic_compare_df columns:")`: Displays a notebook-facing result for inspection.
- `print(synthetic_compare_df.columns.tolist())`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `tolist`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print("source_compare_df columns:")` | Displays a notebook-facing result for inspection. |
| `print(source_compare_df.columns.tolist())` | Displays a notebook-facing result for inspection. |
| `print("\nsynthetic_compare_df columns:")` | Displays a notebook-facing result for inspection. |
| `print(synthetic_compare_df.columns.tolist())` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 28 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `c`
- `columns`
- `lower`
- `source`
- `source_compare_df`
- `state`
- `status`
- `synthetic`
- `synthetic_compare_df`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `print("source status columns:", [c for c in source_compare_df.columns if "status" in c.lower() or "state" in c.lower()])`: Displays a notebook-facing result for inspection.
- `print("synthetic status columns:", [c for c in synthetic_compare_df.columns if "status" in c.lower() or "state" in c.lower()])`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `lower`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `print("source status columns:", [c for c in source_compare_df.columns if "status" in c.lower() or "state" in c.lower()])` | Displays a notebook-facing result for inspection. |
| `print("synthetic status columns:", [c for c in synthetic_compare_df.columns if "status" in c.lower() or "state" in c.lower()])` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 29 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `alias`
- `append`
- `Backward`
- `break`
- `Build`
- `build_synthetic_vs_source_scorecard`
- `Building`
- `Built`
- `code`
- `compatible`
- `does`
- `empty`
- `export`
- `f`
- `full_kaggle_normal_compare_df`
- `immediately`
- `keys`
- `machine_status`
- `normal_clean_source_compare_df`
- `older`

### Outputs

- `label`
- `previous_synthetic_df`
- `previous_synthetic_status_column`
- `scorecard`
- `scorecard_targets`
- `scorecards`
- `source_df`
- `source_status_column`
- `synthetic_df`
- `synthetic_status_column`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Build separated scorecards`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `scorecard_targets = [ { "label": "synthetic_vs_normal_clean", "source_df": normal_clean_source_compare_df, "synthetic_df": synthetic_normal_compare_df, "source_status_column": "mac`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `]`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `if not synthetic_buildup_compare_df.empty and not suspect_normal_source_compare_df.empty: scorecard_targets.append( { "label": "synthetic_buildup_vs_suspect_normal", "source_df": s`: Controls validation, iteration, file handling, or error handling for this step.
- `scorecards = {}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `for target in scorecard_targets: label = target["label"] print(f"Building scorecard: {label}") print(" source:", target["source_df"].shape) print(" synthetic:", target["synthetic_d`: Displays a notebook-facing result for inspection.
- `# Backward-compatible alias so older display/export code does not immediately break.`: Documents the purpose or boundary of the surrounding notebook step.
- `scorecard = scorecards["synthetic_vs_full_source"]`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print("Built scorecards:", list(scorecards.keys()))`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `append`
- `build_synthetic_vs_source_scorecard`
- `keys`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Build separated scorecards` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `scorecard_targets = [ { "label": "synthetic_vs_normal_clean", "source_df": normal_clean_source_compare_df, "synthetic_df": synthetic_normal_compare_df, "source_status_column": "mac` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `]` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `if not synthetic_buildup_compare_df.empty and not suspect_normal_source_compare_df.empty: scorecard_targets.append( { "label": "synthetic_buildup_vs_suspect_normal", "source_df": s` | Controls validation, iteration, file handling, or error handling for this step. |
| `scorecards = {}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for target in scorecard_targets: label = target["label"] print(f"Building scorecard: {label}") print(" source:", target["source_df"].shape) print(" synthetic:", target["synthetic_d` | Displays a notebook-facing result for inspection. |
| `# Backward-compatible alias so older display/export code does not immediately break.` | Documents the purpose or boundary of the surrounding notebook step. |
| `scorecard = scorecards["synthetic_vs_full_source"]` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print("Built scorecards:", list(scorecards.keys()))` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 30 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `clusters`
- `correlation`
- `error`
- `focus_clusters`
- `focus_pairs`
- `global_correlation_error`
- `items`
- `label`
- `missingness`
- `mix`
- `n`
- `nFocus`
- `nGlobal`
- `nMissingness`
- `nState`
- `pairs`
- `scorecard`
- `scorecard_dict`
- `scorecards`
- `separated`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Display separated scorecard summaries`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `for label, scorecard_dict in scorecards.items(): print("\n" + "=" * 80) print(label) print("=" * 80) print("\nState mix:") display(scorecard_dict["state_mix"]) print("\nMissingness`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `display`
- `items`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Display separated scorecard summaries` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `for label, scorecard_dict in scorecards.items(): print("\n" + "=" * 80) print(label) print("=" * 80) print("\nState mix:") display(scorecard_dict["state_mix"]) print("\nMissingness` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 31 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `__`
- `csv`
- `dataframe`
- `DataFrame`
- `def`
- `exist_ok`
- `f`
- `filename_prefix`
- `index`
- `items`
- `mkdir`
- `out_paths`
- `parents`
- `scorecard`
- `synthetic_scorecard`
- `table_name`
- `to_csv`

### Outputs

- `export_scorecard_tables`
- `output_dir`
- `path`

### Key Operations

- `def export_scorecard_tables( *, scorecard: dict[str, pd.DataFrame], output_dir: str \| Path, filename_prefix: str = "synthetic_scorecard",`: Defines notebook-local logic used later in the notebook.
- `) -> dict[str, str]: output_dir = Path(output_dir) output_dir.mkdir(parents=True, exist_ok=True) out_paths: dict[str, str] = {} for table_name, dataframe in scorecard.items(): path`: Writes an artifact or output used for review or downstream notebooks.

Important functions or methods detected:
- `export_scorecard_tables`
- `items`
- `mkdir`
- `Path`
- `to_csv`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `def export_scorecard_tables( *, scorecard: dict[str, pd.DataFrame], output_dir: str \| Path, filename_prefix: str = "synthetic_scorecard",` | Defines notebook-local logic used later in the notebook. |
| `) -> dict[str, str]: output_dir = Path(output_dir) output_dir.mkdir(parents=True, exist_ok=True) out_paths: dict[str, str] = {} for table_name, dataframe in scorecard.items(): path` | Writes an artifact or output used for review or downstream notebooks. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- Artifact or state outputs detected: CSV output.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 32 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `data_synthetic`
- `Export`
- `export_scorecard_tables`
- `get_paths`
- `items`
- `label`
- `scorecard_dict`
- `scorecards`
- `separated`

### Outputs

- `all_scorecard_export_paths`
- `export_paths`
- `filename_prefix`
- `output_dir`
- `paths`
- `scorecard`

### Key Operations

- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `# Export separated scorecards`: Documents the purpose or boundary of the surrounding notebook step.
- `# =========================================================`: Documents the purpose or boundary of the surrounding notebook step.
- `paths = get_paths()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `all_scorecard_export_paths = {}`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `for label, scorecard_dict in scorecards.items(): export_paths = export_scorecard_tables( scorecard=scorecard_dict, output_dir=paths.data_synthetic, filename_prefix=label, ) all_sco`: Controls validation, iteration, file handling, or error handling for this step.
- `all_scorecard_export_paths`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `export_scorecard_tables`
- `get_paths`
- `items`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `# Export separated scorecards` | Documents the purpose or boundary of the surrounding notebook step. |
| `# =========================================================` | Documents the purpose or boundary of the surrounding notebook step. |
| `paths = get_paths()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `all_scorecard_export_paths = {}` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `for label, scorecard_dict in scorecards.items(): export_paths = export_scorecard_tables( scorecard=scorecard_dict, output_dir=paths.data_synthetic, filename_prefix=label, ) all_sco` | Controls validation, iteration, file handling, or error handling for this step. |
| `all_scorecard_export_paths` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 33 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `dispose`
- `engine`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `e`: Executes part of the notebook workflow while preserving the existing analytical behavior.

Important functions or methods detected:
- `dispose`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `e` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.

## Code Cell 34 — Code Reference

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

## Code Cell 35 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `at`
- `card`
- `complete`
- `f`
- `formatted_datetime`
- `Score`

### Outputs

- No clear assigned output variables were detected by static review.

### Key Operations

- `p`: Executes part of the notebook workflow while preserving the existing analytical behavior.

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `p` | Executes part of the notebook workflow while preserving the existing analytical behavior. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
- This cell may affect model fitting, scoring, or evaluation state and should not be edited without separate validation.

## Code Cell 36 — Code Reference

### Purpose

This cell supports $heading in the notebook workflow. It keeps the existing execution behavior and should be read in sequence with the cells before it.

### Inputs

- `at`
- `Completed`
- `d`
- `Data`
- `datetime`
- `f`
- `Generation`
- `generation_started_current_datetime`
- `H`
- `hours`
- `m`
- `M`
- `now`
- `r`
- `runtime`
- `store`
- `strftime`
- `Synthetic`
- `timedelta`
- `Total`

### Outputs

- `generation_complete_current_datetime`
- `generation_completed_adjusted_time`
- `generation_completed_formatted_datetime`
- `total_runtime`

### Key Operations

- `%store -r generation_started_current_datetime`: Executes part of the notebook workflow while preserving the existing analytical behavior.
- `generation_complete_current_datetime = datetime.now()`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `generation_completed_adjusted_time = generation_complete_current_datetime - timedelta(hours=4)`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `generation_completed_formatted_datetime = generation_completed_adjusted_time.strftime("%m%d%Y_%H%M")`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `total_runtime = generation_complete_current_datetime - generation_started_current_datetime`: Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream.
- `print(f"Synthetic Data Generation Completed at {generation_completed_formatted_datetime}")`: Displays a notebook-facing result for inspection.
- `print(f"Total runtime: {total_runtime}")`: Displays a notebook-facing result for inspection.

Important functions or methods detected:
- `now`
- `strftime`
- `timedelta`

### Line / Statement Reference

| Line or Statement | Explanation |
|---|---|
| `%store -r generation_started_current_datetime` | Executes part of the notebook workflow while preserving the existing analytical behavior. |
| `generation_complete_current_datetime = datetime.now()` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `generation_completed_adjusted_time = generation_complete_current_datetime - timedelta(hours=4)` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `generation_completed_formatted_datetime = generation_completed_adjusted_time.strftime("%m%d%Y_%H%M")` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `total_runtime = generation_complete_current_datetime - generation_started_current_datetime` | Assigns a value, dataframe, path, model object, configuration object, or intermediate result used downstream. |
| `print(f"Synthetic Data Generation Completed at {generation_completed_formatted_datetime}")` | Displays a notebook-facing result for inspection. |
| `print(f"Total runtime: {total_runtime}")` | Displays a notebook-facing result for inspection. |

### Behavior Notes

- This reference documents the existing code path; it does not change execution behavior.
