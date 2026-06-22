# Utility Module Reference: `utils/core/truths.py`

## Module Purpose

This module creates and applies truth metadata records that connect dataframes, parent truths, config snapshots, and artifact lineage.

## Pipeline Role

- Stage support: Core / cross-stage
- Primary responsibility: This module creates and applies truth metadata records that connect dataframes, parent truths, config snapshots, and artifact lineage.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `utc_now_iso` | Return the current UTC timestamp as a second-precision ISO string. | deep |
| `make_process_run_id` | Build a UTC timestamped process run identifier with the given prefix. | short |
| `_normalize_for_json` | Convert common project values into deterministic JSON-safe objects. | deep |
| `compute_sha256` | Return a SHA-256 hash for a normalized JSON representation of a payload. | deep |
| `identify_meta_columns` | Return sorted dataframe columns reserved for project metadata. | short |
| `identify_feature_columns` | Return sorted dataframe columns that are not project metadata fields. | short |
| `extract_truth_hash` | Extract the single non-null truth hash from a dataframe metadata column. | deep |
| `build_file_fingerprint` | Build a small deterministic fingerprint payload for a source file. | deep |
| `initialize_layer_truth` | Create the base truth payload for a Medallion layer before row facts are known. | deep |
| `update_truth_section` | Return a copied truth record with one section updated by the supplied values. | deep |
| `build_truth_record` | Build the final truth record and hash from base lineage and dataframe facts. | deep |
| `save_truth_record` | Write a truth record JSON file under the layer truth directory. | deep |
| `append_truth_index` | Append a normalized truth record to the JSONL truth index file. | deep |
| `stamp_truth_columns` | Return a copied dataframe stamped with truth lineage metadata columns. | deep |
| `load_truth_record` | Load a truth record JSON file from disk. | deep |
| `find_truth_record_by_hash` | Resolve the expected truth record path for a dataset, layer, and hash. | deep |
| `load_truth_record_by_hash` | Load a truth record by dataset, layer, and truth hash. | deep |
| `load_parent_truth_record_from_dataframe` | Load the parent truth record referenced by a dataframe truth column. | deep |
| `get_dataset_name_from_truth` | Return the required dataset name from a truth record. | deep |
| `get_truth_hash` | Return the required truth hash from a truth record. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.
- Dataset, run, stage, or recipe identifiers used for traceability.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `utc_now_iso` | `No explicit parameters` | Return the current UTC timestamp as a second-precision ISO string. |
| `make_process_run_id` | `prefix` | Build a UTC timestamped process run identifier with the given prefix. |
| `_normalize_for_json` | `value` | Convert common project values into deterministic JSON-safe objects. |
| `compute_sha256` | `payload` | Return a SHA-256 hash for a normalized JSON representation of a payload. |
| `identify_meta_columns` | `dataframe` | Return sorted dataframe columns reserved for project metadata. |
| `identify_feature_columns` | `dataframe` | Return sorted dataframe columns that are not project metadata fields. |
| `extract_truth_hash` | `dataframe, column_name` | Extract the single non-null truth hash from a dataframe metadata column. |
| `build_file_fingerprint` | `file_path` | Build a small deterministic fingerprint payload for a source file. |
| `initialize_layer_truth` | `*, truth_version, dataset_name, layer_name, process_run_id, pipeline_mode, parent_truth_hash` | Create the base truth payload for a Medallion layer before row facts are known. |
| `update_truth_section` | `truth, section, values` | Return a copied truth record with one section updated by the supplied values. |
| `build_truth_record` | `*, truth_base, row_count, column_count, meta_columns, feature_columns` | Build the final truth record and hash from base lineage and dataframe facts. |
| `save_truth_record` | `truth_record, *, truth_dir, dataset_name, layer_name` | Write a truth record JSON file under the layer truth directory. |
| `append_truth_index` | `truth_record, *, truth_index_path` | Append a normalized truth record to the JSONL truth index file. |
| `stamp_truth_columns` | `dataframe, *, truth_hash, parent_truth_hash, pipeline_mode` | Return a copied dataframe stamped with truth lineage metadata columns. |

## Side Effects

- Source includes directory creation; helpers can create configured output directories.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `FileNotFoundError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Module Importance

This module matters because shared configuration, paths, logging, ledger, truth, artifact, and optional W&B behavior reduce duplicated notebook setup.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
