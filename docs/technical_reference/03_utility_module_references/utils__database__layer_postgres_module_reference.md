# Utility Module Reference: `utils/database/layer_postgres.py`

## Module Purpose

This module provides layer-aware PostgreSQL read/write helpers for Medallion outputs.

## Pipeline Role

- Stage support: Database / SQL persistence
- Primary responsibility: This module provides layer-aware PostgreSQL read/write helpers for Medallion outputs.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Silver_01_PreEDA`, `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `build_layer_table_name` | Build a sanitized layer table name from dataset, layer, and artifact parts. | deep |
| `_series_looks_like_json` | Return whether the first non-null series value is a JSON-like object. | short |
| `_infer_sqlalchemy_dtype_for_series` | Infer a conservative SQLAlchemy dtype for a pandas Series. | deep |
| `infer_sqlalchemy_dtypes` | Build a pandas.to_sql dtype mapping, honoring explicit column overrides. | deep |
| `prepare_layer_dataframe` | Return a copy of the dataframe with optional capstone-style meta columns. | deep |
| `write_layer_dataframe` | Generic dataframe writer for Bronze / Silver / Gold / synthetic layers. | deep |
| `read_layer_dataframe` | Read a layer table into pandas with optional projection, filtering, ordering, and limit. | deep |

## Configuration Dependencies

- Dataset, run, stage, or recipe identifiers used for traceability.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `build_layer_table_name` | `*, dataset_name, layer, artifact_name, include_layer_prefix` | Build a sanitized layer table name from dataset, layer, and artifact parts. |
| `_series_looks_like_json` | `series` | Return whether the first non-null series value is a JSON-like object. |
| `_infer_sqlalchemy_dtype_for_series` | `series` | Infer a conservative SQLAlchemy dtype for a pandas Series. |
| `infer_sqlalchemy_dtypes` | `dataframe, dtype_overrides` | Build a pandas.to_sql dtype mapping, honoring explicit column overrides. |
| `prepare_layer_dataframe` | `dataframe, *, truth_hash, parent_truth_hash, pipeline_mode, process_run_id, add_loaded_at_column, loaded_at_column, extra_meta` | Return a copy of the dataframe with optional capstone-style meta columns. |
| `write_layer_dataframe` | `engine, dataframe, *, schema, dataset_name, layer, artifact_name, table_name, include_layer_prefix_in_table_name, if_exists, index, chunksize, method, allow_empty, dtype_overrides, logger` | Generic dataframe writer for Bronze / Silver / Gold / synthetic layers. |
| `read_layer_dataframe` | `engine, *, schema, table_name, dataset_name, layer, artifact_name, include_layer_prefix_in_table_name, columns, where_clause, params, order_by, limit, require_exists` | Read a layer table into pandas with optional projection, filtering, ordering, and limit. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.
- Source includes logger usage; helpers can emit project log messages.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.
- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `FileNotFoundError` for invalid input, missing context, or failed validation paths.
- Source raises `TypeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Silver_01_PreEDA`, `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Module Importance

This module matters because SQL persistence and metadata logging must stay consistent across notebook reruns and Medallion handoffs.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
