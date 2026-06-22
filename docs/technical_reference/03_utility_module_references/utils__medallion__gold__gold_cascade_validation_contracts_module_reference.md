# Utility Module Reference: `utils/medallion/gold/gold_cascade_validation_contracts.py`

## Module Purpose

This module builds validation contracts for Gold cascade outputs so replay and early-warning checks have stable expectations.

## Pipeline Role

- Stage support: Gold
- Primary responsibility: This module builds validation contracts for Gold cascade outputs so replay and early-warning checks have stable expectations.

## Primary Consumers

`EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `to_json_safe` | Convert common notebook objects into JSON-serializable values. | deep |
| `stable_json_hash` | Create a stable SHA-256 hash for a JSON-like payload. | deep |
| `write_json_contract` | Write a validation contract to disk as pretty JSON. | deep |
| `load_json_contract` | Load a JSON validation contract from disk. | deep |
| `_path_exists` | Return whether a non-empty path-like value exists on disk. | deep |
| `_first_existing_path` | Return the first existing path from an iterable, or None. | deep |
| `_metric_value` | Return the first metric value found for the ordered candidate keys. | deep |
| `build_cascade_variant_contract` | Build a validation contract for one cascade notebook output. | deep |
| `build_stage3_rule_payload_from_globals` | Build a Stage 3 rule payload from common Gold 03 notebook variables. | deep |
| `build_gold06_validation_targets` | Return the expected final Gold model outputs Gold 06 should validate. | deep |
| `load_validation_contracts` | Load contracts referenced by the Gold 06 validation target table. | deep |
| `validate_gold04_against_contracts` | Validate that every Gold 04 final model row has a supporting contract. | deep |
| `_first_present_value` | Return the first non-null value found for a list of candidate keys. | short |
| `_as_float_or_none` | Convert a value to float when possible; otherwise return None. | short |
| `_as_int_or_none` | Convert a value to int when possible; otherwise return None. | short |
| `normalize_gold_metric_payload` | Normalize model metrics into the same fields used by Gold 04. | deep |
| `summarize_validation_output_dataframe` | Summarize the dataframe that supports one validation contract. | deep |
| `build_gold_model_output_validation_contract` | Build one explicit validation contract for a final Gold comparison row. | deep |
| `write_gold_model_output_validation_contract` | Write one final-model validation contract to its canonical artifact path. | deep |
| `build_gold06_contract_validation_targets` | Build the Gold 06 contract-validation target table. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `to_json_safe` | `value` | Convert common notebook objects into JSON-serializable values. |
| `stable_json_hash` | `payload` | Create a stable SHA-256 hash for a JSON-like payload. |
| `write_json_contract` | `contract, output_path` | Write a validation contract to disk as pretty JSON. |
| `load_json_contract` | `path` | Load a JSON validation contract from disk. |
| `_path_exists` | `path_value` | Return whether a non-empty path-like value exists on disk. |
| `_first_existing_path` | `paths` | Return the first existing path from an iterable, or None. |
| `_metric_value` | `metrics, *keys` | Return the first metric value found for the ordered candidate keys. |
| `build_cascade_variant_contract` | `*, dataset_id, run_id, model_id, source_notebook, cascade_variant, model_stage, operating_mode, stage3_type, rule_source, cascade_results, cascade_metrics, threshold_payload, artifact_paths, stage1_model_path, stage2_model_path, stage3_rule_payload, gold04_targets, lineage_payload, notes` | Build a validation contract for one cascade notebook output. |
| `build_stage3_rule_payload_from_globals` | `*, notebook_globals, selected_mode` | Build a Stage 3 rule payload from common Gold 03 notebook variables. |
| `build_gold06_validation_targets` | `No explicit parameters` | Return the expected final Gold model outputs Gold 06 should validate. |
| `load_validation_contracts` | `contract_dir, targets` | Load contracts referenced by the Gold 06 validation target table. |
| `validate_gold04_against_contracts` | `*, gold04_dataframe, validation_targets, contracts` | Validate that every Gold 04 final model row has a supporting contract. |
| `_first_present_value` | `mapping, candidates` | Return the first non-null value found for a list of candidate keys. |
| `_as_float_or_none` | `value` | Convert a value to float when possible; otherwise return None. |

## Side Effects

- Source includes directory creation; helpers can create configured output directories.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.

## Failure Behavior

- Source raises `KeyError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`

## Module Importance

This module matters because Gold notebooks depend on stable shared helpers for model input preparation, cascade modeling, evaluation, validation contracts, and artifact traceability.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
