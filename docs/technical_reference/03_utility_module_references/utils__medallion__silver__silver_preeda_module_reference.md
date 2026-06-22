# Utility Module Reference: `utils/medallion/silver/silver_preeda.py`

## Module Purpose

This module supports Silver PreEDA preparation, profiling, and validation before detailed exploratory analysis.

## Pipeline Role

- Stage support: Silver
- Primary responsibility: This module supports Silver PreEDA preparation, profiling, and validation before detailed exploratory analysis.

## Primary Consumers

`EDA_Notebook_Pump_Silver_01_PreEDA`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `remove_junk_import_columns` | Remove common import-artifact columns from a Bronze dataframe copy. | deep |
| `_ensure_regex_pattern` | Ensure a string/regex input is returned as a compiled regex. | short |
| `deduplicate_columns` | Remove duplicate column names while preserving the chosen keep strategy. | deep |
| `_clean_values` | Normalize a series into stripped string values. | short |
| `_normalize_dataset_name` | Normalize dataset names into pipeline-safe identifiers. | deep |
| `validate_dataset_name_for_silver` | Validate or resolve the Silver dataset name from dataframe, truth, or config. | deep |
| `resolve_label_or_status_source` | Resolve the best source column for anomaly/status labeling. | deep |
| `protect_canonical_output_names` | Rename raw columns that would collide with canonical Silver output names. | short |
| `_pick_first_existing_candidate_column` | Return the first candidate column that exists in the dataframe. | short |
| `_ensure_grouping_columns_exists` | Ensure meta__asset_id and meta__run_id exist. | short |
| `evaluate_time_candidates` | Evaluate candidate time columns and choose the best parseable option. | short |
| `evaluate_step_candidates` | Evaluate candidate step/order columns and choose the best numeric option. | short |
| `build_canonical_identity_and_order_master` | Build canonical Silver ordering and identity fields. | deep |
| `normalize_label_to_binary` | Convert mixed label representations into binary anomaly labels. | deep |
| `build_anomaly_flag_from_status` | Build ``anomaly_flag`` using either a label column or a status column. | deep |
| `build_episode_ids_from_anomaly_flag` | Build episode IDs from anomaly-flag transitions within asset/run groups. | deep |
| `classify_column_type` | Coarse column typing for feature registry. | short |
| `should_exclude_by_prefix` | Return True if a column should be excluded based on prefix rules. | short |
| `looks_like_identifier_column` | Heuristic check for identifier-like columns that should not be treated as features. | medium |
| `identify_feature_set` | Identify usable Silver feature columns and light feature groups. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.
- Dataset, run, stage, or recipe identifiers used for traceability.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `remove_junk_import_columns` | `dataframe, *, junk_column_candidates, unnamed_column_regex` | Remove common import-artifact columns from a Bronze dataframe copy. |
| `_ensure_regex_pattern` | `pattern` | Ensure a string/regex input is returned as a compiled regex. |
| `deduplicate_columns` | `dataframe, *, keep` | Remove duplicate column names while preserving the chosen keep strategy. |
| `_clean_values` | `series` | Normalize a series into stripped string values. |
| `_normalize_dataset_name` | `dataset_name` | Normalize dataset names into pipeline-safe identifiers. |
| `validate_dataset_name_for_silver` | `dataframe, *, dataset_column, dataset_name_config, dataset_name_parent_truth` | Validate or resolve the Silver dataset name from dataframe, truth, or config. |
| `resolve_label_or_status_source` | `dataframe, *, status_column_candidates, label_column_candidates, label_exclude_columns` | Resolve the best source column for anomaly/status labeling. |
| `protect_canonical_output_names` | `dataframe, *, canonical_output_columns, raw_prefix` | Rename raw columns that would collide with canonical Silver output names. |
| `_pick_first_existing_candidate_column` | `dataframe, candidates` | Return the first candidate column that exists in the dataframe. |
| `_ensure_grouping_columns_exists` | `dataframe, *, asset_id_default_fallback, run_id_default_fallback` | Ensure meta__asset_id and meta__run_id exist. |
| `evaluate_time_candidates` | `dataframe, *, time_column_candidates` | Evaluate candidate time columns and choose the best parseable option. |
| `evaluate_step_candidates` | `dataframe, *, step_column_candidates` | Evaluate candidate step/order columns and choose the best numeric option. |
| `build_canonical_identity_and_order_master` | `dataframe, *, time_column_candidates, step_column_candidates, tie_breaker_candidates, asset_id_default_fallback, run_id_default_fallback, min_time_parse_success_percent, min_step_parse_success_percent` | Build canonical Silver ordering and identity fields. |
| `normalize_label_to_binary` | `series` | Convert mixed label representations into binary anomaly labels. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Silver_01_PreEDA`

## Module Importance

This module matters because Silver notebooks depend on repeatable profiling and EDA helpers before the modeling-ready Gold layer is built.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
