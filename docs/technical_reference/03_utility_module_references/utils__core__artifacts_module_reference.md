# Utility Module Reference: `utils/core/artifacts.py`

## Module Purpose

This module builds deterministic artifact directories and file paths so notebooks can write outputs under consistent stage, dataset, and family locations.

## Pipeline Role

- Stage support: Core / cross-stage
- Primary responsibility: This module builds deterministic artifact directories and file paths so notebooks can write outputs under consistent stage, dataset, and family locations.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_clean_path_part` | Return a trimmed path component, or None when the value is blank. | deep |
| `_copy_artifact_mapping` | Copy an artifact configuration mapping into a plain string-keyed dictionary. | deep |
| `_require_mapping` | Validate that a value is a mapping and return it as a string-keyed dictionary. | deep |
| `_get_artifact_mapping` | Return a nested artifact mapping as a plain string-keyed dictionary. | deep |
| `_get_optional_artifact_str` | Return an optional artifact config value as a string. | deep |
| `_get_required_artifact_str` | Return a required artifact config value as a non-empty string. | deep |
| `_get_artifact_subdirs` | Return artifact subdirectory names from config as a list of strings. | deep |
| `build_artifact_dirs` | Build standardized artifact directories for a pipeline stage. | deep |
| `build_artifact_dirs_from_config` | Build standardized artifact directories from the resolved pipeline config. | deep |
| `artifact_file_path` | Build a file path inside one standardized artifact subdirectory. | deep |
| `build_gold_model_validation_artifact_dirs` | Build canonical Gold model-validation artifact directories. | deep |
| `gold_model_validation_contracts_dir` | Return the canonical Gold model-validation contracts directory. | deep |
| `gold_model_validation_results_dir` | Return the canonical Gold model-validation results directory. | deep |
| `gold_model_validation_contract_filename` | Return the canonical filename for one Gold output-validation contract. | deep |
| `gold_model_validation_contract_path` | Return the canonical path for one Gold output-validation contract. | deep |

## Configuration Dependencies

- Resolved pipeline configuration dictionaries and YAML-derived stage settings.
- Project root, resolved path mappings, and artifact directory configuration.
- Dataset, run, stage, or recipe identifiers used for traceability.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_clean_path_part` | `value` | Return a trimmed path component, or None when the value is blank. |
| `_copy_artifact_mapping` | `value` | Copy an artifact configuration mapping into a plain string-keyed dictionary. |
| `_require_mapping` | `value, label` | Validate that a value is a mapping and return it as a string-keyed dictionary. |
| `_get_artifact_mapping` | `mapping, key` | Return a nested artifact mapping as a plain string-keyed dictionary. |
| `_get_optional_artifact_str` | `mapping, key, default` | Return an optional artifact config value as a string. |
| `_get_required_artifact_str` | `mapping, key, label` | Return a required artifact config value as a non-empty string. |
| `_get_artifact_subdirs` | `mapping, key, default` | Return artifact subdirectory names from config as a list of strings. |
| `build_artifact_dirs` | `*, artifacts_root, stage, dataset_name, family, subdirs, create` | Build standardized artifact directories for a pipeline stage. |
| `build_artifact_dirs_from_config` | `*, config, stage_key, family_override, variant, subdirs_override, create` | Build standardized artifact directories from the resolved pipeline config. |
| `artifact_file_path` | `artifact_dirs, subdir_key, file_name` | Build a file path inside one standardized artifact subdirectory. |
| `build_gold_model_validation_artifact_dirs` | `*, artifacts_root, dataset_id, create` | Build canonical Gold model-validation artifact directories. |
| `gold_model_validation_contracts_dir` | `*, artifacts_root, dataset_id, create` | Return the canonical Gold model-validation contracts directory. |
| `gold_model_validation_results_dir` | `*, artifacts_root, dataset_id, create` | Return the canonical Gold model-validation results directory. |
| `gold_model_validation_contract_filename` | `*, dataset_id, model_id` | Return the canonical filename for one Gold output-validation contract. |

## Side Effects

- Source includes directory creation; helpers can create configured output directories.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.

## Failure Behavior

- Source raises `KeyError` for invalid input, missing context, or failed validation paths.
- Source raises `TypeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Module Importance

This module matters because shared configuration, paths, logging, ledger, truth, artifact, and optional W&B behavior reduce duplicated notebook setup.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
