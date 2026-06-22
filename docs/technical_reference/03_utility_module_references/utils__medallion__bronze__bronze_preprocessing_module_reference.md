# Utility Module Reference: `utils/medallion/bronze/bronze_preprocessing.py`

## Module Purpose

This module prepares raw pump observations for the Bronze layer by cleaning, validating, and shaping source data.

## Pipeline Role

- Stage support: Bronze
- Primary responsibility: This module prepares raw pump observations for the Bronze layer by cleaning, validating, and shaping source data.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Silver_01_PreEDA`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_normalize_dataset_name` | Normalize a dataset name into a stable pipeline-safe identifier. | deep |
| `_generate_deterministic_dataset_name_from_file_details` | Build a deterministic fallback dataset name from source file details. | deep |
| `resolve_dataset_name_for_bronze_pre_ingest` | Resolve dataset name before Bronze ingestion. | deep |
| `write_dataset_resolution_attrs` | Write Bronze dataset resolution metadata into dataframe.attrs. | deep |
| `collect_meta_columns` | Return meta__ columns in their existing order. | short |
| `reorder_bronze_columns` | Move meta__ columns to the front while preserving existing order. | deep |
| `prepare_bronze_dataframe` | Prepare a Bronze dataframe for downstream saving and truth stamping. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.
- Dataset, run, stage, or recipe identifiers used for traceability.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_normalize_dataset_name` | `dataset_name` | Normalize a dataset name into a stable pipeline-safe identifier. |
| `_generate_deterministic_dataset_name_from_file_details` | `path_value` | Build a deterministic fallback dataset name from source file details. |
| `resolve_dataset_name_for_bronze_pre_ingest` | `*, argument_value, config_value, handoff_dataset_name, source_table_name, source_table_dataset_map, fallback_value, source_path` | Resolve dataset name before Bronze ingestion. |
| `write_dataset_resolution_attrs` | `dataframe, *, dataset_column, fallback_dataset_name, fallback_method` | Write Bronze dataset resolution metadata into dataframe.attrs. |
| `collect_meta_columns` | `existing_columns` | Return meta__ columns in their existing order. |
| `reorder_bronze_columns` | `dataframe` | Move meta__ columns to the front while preserving existing order. |
| `prepare_bronze_dataframe` | `dataframe, *, argument_dataset_name, config_dataset_name, handoff_dataset_name, source_table_name, source_table_dataset_map, fallback_dataset_name, source_path, dataset_column, reorder_columns` | Prepare a Bronze dataframe for downstream saving and truth stamping. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Silver_01_PreEDA`

## Module Importance

This module matters because Bronze preprocessing defines the first cleaned analytical layer used by downstream Silver and Gold work.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
