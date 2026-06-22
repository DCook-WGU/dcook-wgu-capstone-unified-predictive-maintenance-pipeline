# Utility Module Reference: `utils/core/file_io.py`

## Module Purpose

This module centralizes small JSON, text, and directory file I/O helpers used by notebooks and utilities.

## Pipeline Role

- Stage support: Core / cross-stage
- Primary responsibility: This module centralizes small JSON, text, and directory file I/O helpers used by notebooks and utilities.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_resolve_path` | Resolve flexible file path inputs into a Path. | deep |
| `_copy_read_options` | Copy pandas read options into a plain string-keyed dictionary. | short |
| `_copy_write_options` | Copy pandas write options into a plain string-keyed dictionary. | deep |
| `_clean_values` | Return non-null, stripped string values from a series. | short |
| `_normalize_dataset_name` | Normalize a dataset identifier for stable filenames and metadata. | deep |
| `_generate_deterministic_dataset_name_from_file_details` | Build a deterministic dataset name from file details. | deep |
| `_create_record_id` | Create a deterministic record identifier from source lineage fields. | deep |
| `ingest_data` | Ingest a raw dataset into the Bronze layer. | deep |
| `load_data` | Load a CSV or Parquet dataset from disk. | deep |
| `save_data` | Save a dataframe as CSV or Parquet and return the written path. | deep |
| `_json_default` | Convert pandas, NumPy, and datetime values into JSON-serializable values. | deep |
| `save_json` | Save a JSON-serializable object and return the written path. | deep |
| `load_json` | Load a JSON file and return the parsed Python object (dict/list/etc.). | deep |
| `resolve_dataset_name_for_bronze` | Resolve dataset name during Bronze ingestion using the following priority order: 1. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.
- Dataset, run, stage, or recipe identifiers used for traceability.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_resolve_path` | `file_path, file_name` | Resolve flexible file path inputs into a Path. |
| `_copy_read_options` | `read_options` | Copy pandas read options into a plain string-keyed dictionary. |
| `_copy_write_options` | `write_options` | Copy pandas write options into a plain string-keyed dictionary. |
| `_clean_values` | `series` | Return non-null, stripped string values from a series. |
| `_normalize_dataset_name` | `dataset_name` | Normalize a dataset identifier for stable filenames and metadata. |
| `_generate_deterministic_dataset_name_from_file_details` | `path_value` | Build a deterministic dataset name from file details. |
| `_create_record_id` | `dataframe, source_file_column, source_row_column, out_column` | Create a deterministic record identifier from source lineage fields. |
| `ingest_data` | `file_path, file_name, dataset_name, dataset_name_config, dataset_candidates, split, run_id, asset_id, add_record_id, validate, **read_kwargs` | Ingest a raw dataset into the Bronze layer. |
| `load_data` | `file_path, file_name, engine, **read_kwargs` | Load a CSV or Parquet dataset from disk. |
| `save_data` | `dataframe, file_path, file_name, create_dirs, index, **write_kwargs` | Save a dataframe as CSV or Parquet and return the written path. |
| `_json_default` | `value` | Convert pandas, NumPy, and datetime values into JSON-serializable values. |
| `save_json` | `data, file_path, file_name, create_dirs, indent` | Save a JSON-serializable object and return the written path. |
| `load_json` | `file_path, file_name, *, encoding, default, raise_if_missing` | Load a JSON file and return the parsed Python object (dict/list/etc.). |
| `resolve_dataset_name_for_bronze` | `dataframe, *, dataset_candidates, argument_value, config_value, fallback_value, source_path, bronze_source_column, fail_on_multiple_in_dataset` | Resolve dataset name during Bronze ingestion using the following priority order: 1. |

## Side Effects

- Source includes file-write calls; helpers can write configured files or artifacts when those paths are passed by the caller.
- Source includes directory creation; helpers can create configured output directories.
- Source includes logger usage; helpers can emit project log messages.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories and file-write operations.

## Failure Behavior

- Source raises `FileNotFoundError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.
- Source raises `logger` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Module Importance

This module matters because shared configuration, paths, logging, ledger, truth, artifact, and optional W&B behavior reduce duplicated notebook setup.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
