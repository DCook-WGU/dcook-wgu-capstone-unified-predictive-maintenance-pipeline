# Utility Module Reference: `utils/core/wandb_utils.py`

## Module Purpose

This module wraps optional Weights & Biases logging and artifact behavior so W&B can remain disabled by configuration.

## Pipeline Role

- Stage support: Core / cross-stage
- Primary responsibility: This module wraps optional Weights & Biases logging and artifact behavior so W&B can remain disabled by configuration.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Silver_01_PreEDA`, `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_require_wandb` | Import wandb with a clear error message if not installed. | deep |
| `_sanitize_dataframe_for_wandb_table` | Return a W&B table-safe dataframe copy with datetimes and missing values normalized. | deep |
| `log_metrics` | Log scalar metrics (or small JSON-serializable values) to an active W&B run. | deep |
| `log_dataframe_head` | Log a small DataFrame sample as a W&B Table. | deep |
| `log_text` | Log a text blob. | deep |
| `log_files_as_artifact` | Create a W&B artifact and attach specific files. | deep |
| `log_dir_as_artifact` | Create a W&B artifact and attach files from a directory matching glob patterns. | deep |
| `finalize_wandb_stage` | End-of-stage W&B finalizer: - optionally profiles df and saves CSVs into project_root/artifacts/<stage>/ - logs metrics + a head table - uploads: * logs/<stage>.log * parquet outputs from dataset_dirs * diagnostics from artifacts/<stage>/ * optional notebook .ipynb Returns a dict with paths used + computed metrics. | deep |

## Configuration Dependencies

- Environment variables where runtime mode or optional integration behavior is configured.
- Project root, resolved path mappings, and artifact directory configuration.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_require_wandb` | `No explicit parameters` | Import wandb with a clear error message if not installed. |
| `_sanitize_dataframe_for_wandb_table` | `frame` | Return a W&B table-safe dataframe copy with datetimes and missing values normalized. |
| `log_metrics` | `run, metrics, step, commit` | Log scalar metrics (or small JSON-serializable values) to an active W&B run. |
| `log_dataframe_head` | `run, dataframe, key, n, *, max_rows` | Log a small DataFrame sample as a W&B Table. |
| `log_text` | `run, key, text` | Log a text blob. |
| `log_files_as_artifact` | `run, *, artifact_name, artifact_type, files, aliases, metadata` | Create a W&B artifact and attach specific files. |
| `log_dir_as_artifact` | `run, *, artifact_name, artifact_type, dir_path, patterns, aliases, metadata, recursive` | Create a W&B artifact and attach files from a directory matching glob patterns. |
| `finalize_wandb_stage` | `run, *, stage, dataframe, project_root, logs_dir, dataset_dirs, dataset_artifact_name, logger, notebook_path, aliases, table_key, table_n, profile, diagnostics_patterns, parquet_patterns` | End-of-stage W&B finalizer: - optionally profiles df and saves CSVs into project_root/artifacts/<stage>/ - logs metrics + a head table - uploads: * logs/<stage>.log * parquet outputs from dataset_dirs * diagnostics from artifacts/<stage>/ * optional notebook .ipynb Returns a dict with paths used + computed metrics. |

## Side Effects

- Source includes directory creation; helpers can create configured output directories.
- Source includes logger usage; helpers can emit project log messages.
- Source includes W&B integration points; behavior depends on the project's optional W&B configuration.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
- W&B: Source references optional Weights & Biases helper behavior.

## Failure Behavior

- Source raises `FileNotFoundError` for invalid input, missing context, or failed validation paths.
- Source raises `ImportError` for invalid input, missing context, or failed validation paths.
- Source raises `NotADirectoryError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Silver_01_PreEDA`, `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Module Importance

This module matters because shared configuration, paths, logging, ledger, truth, artifact, and optional W&B behavior reduce duplicated notebook setup.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
