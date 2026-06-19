# Database Utility Reference: medallion_sql_writers.py

Source path:

`utils/database/medallion_sql_writers.py`

## Purpose

Provides notebook-facing SQL writers for selected Bronze, Silver, and Gold outputs in the Medallion-style capstone pipeline.

These helpers let notebooks persist important dataframe outputs and metadata to PostgreSQL while keeping generated files, plots, ledgers, and other artifacts in the normal artifact folders.

## Pipeline Role

This module supports SQL persistence after notebook processing steps have already produced a dataframe or summary table.

Typical use:

1. A notebook finishes a Bronze, Silver, or Gold processing step.
2. The notebook passes a dataframe directly, or passes notebook globals so the helper can find a known dataframe variable.
3. The helper deletes prior rows for the same dataset/run when needed.
4. The helper inserts current rows into the target layer table.
5. The helper records SQL-facing metadata in the capstone metadata schema.

## Main Functions

| Function | Main Inputs | SQL Output |
|---|---|---|
| `write_bronze_sensor_observations_sql` | engine, schema, dataset/run ids, Bronze dataframe or notebook globals | `bronze.sensor_observations` plus metadata records |
| `write_silver_sensor_observation_features_sql` | engine, dataset/run ids, Silver dataframe or notebook globals | `silver.sensor_observation_features` plus metadata records |
| `log_silver_eda_sql` | engine, dataset/run ids, Silver EDA dataframe or notebook globals | `capstone.pipeline_runs`, `capstone.data_quality_events`, optional artifact metadata |
| `write_silver_eda_sql_outputs` | engine, dataset/run ids, Silver EDA summary dataframes | Silver EDA summary tables |
| `write_gold_preprocessed_features_sql` | engine, dataset/run ids, Gold preprocessing dataframe or notebook globals | `gold.preprocessed_features` plus metadata records |
| `write_gold_anomaly_scores_sql` | engine, dataset/run ids, model/stage labels, scored dataframe | `gold.anomaly_detection_scores` plus metadata records |
| `write_gold_baseline_scores_sql` | engine, dataset/run ids, baseline scored dataframe or notebook globals | baseline rows in `gold.anomaly_detection_scores` |
| `write_gold_cascade_scores_sql` | engine, dataset/run ids, cascade scored dataframe or notebook globals | cascade rows in `gold.anomaly_detection_scores` |
| `write_gold_model_comparison_results_sql` | engine, dataset/run ids, comparison dataframe or notebook globals | `gold.model_comparison_results` plus metadata records |
| `log_gold_05_anomaly_detection_summary_sql` | engine, dataset/run ids, Gold 05 summary dataframe or notebook globals | `capstone.pipeline_runs`, `capstone.data_quality_events`, optional artifact metadata |

## Configuration and Environment Behavior

- Callers provide a SQLAlchemy engine.
- Callers provide `dataset_id` and `run_id` explicitly.
- Callers provide the metadata schema through `capstone_schema`.
- Layer schemas are fixed by the writer or passed explicitly where supported.
- Dataframes can be passed directly, or resolved from notebook globals using expected variable names.

## Database and Table Assumptions

This module assumes the PostgreSQL bootstrap has created the layer and metadata schemas used by the capstone:

- `bronze.sensor_observations`
- `silver.sensor_observation_features`
- Silver EDA summary tables
- `gold.preprocessed_features`
- `gold.anomaly_detection_scores`
- `gold.model_comparison_results`
- `capstone.pipeline_runs`
- `capstone.data_quality_events`
- `capstone.pipeline_artifacts`

The Gold preprocessed feature writer can create or migrate `gold.preprocessed_features` to protect notebook reruns against older table definitions.

## SQL Side Effects

- Writer functions delete existing rows for the same dataset/run before inserting replacements where idempotent reruns are expected.
- Gold score writers delete existing rows for the same dataset/run/model/stage before inserting replacements.
- Metadata helpers upsert rows into `capstone.pipeline_runs`.
- Data quality helper calls insert rows into `capstone.data_quality_events`.
- Artifact helper calls insert metadata rows into `capstone.pipeline_artifacts`.
- The module does not reset schemas, truncate full tables, rebuild Docker services, or create notebook artifacts.

## Logging, Ledger, and Artifact Behavior

### Logging

The module uses print-style notebook status messages for row counts, deletions, metadata writes, and artifact metadata logging.

### Ledger

The module does not write project ledger entries directly. It records SQL-facing metadata in PostgreSQL tables.

### Artifacts

The module can record artifact metadata when notebooks expose known artifact path variables. It does not create, copy, or move artifact files.

## Notebook Usage Context

The helpers are intended to be called from notebooks after dataframe creation is complete.

For direct use, pass `dataframe=...`.

For notebook-global lookup, pass `notebook_globals=globals()` and optionally `candidate_names=[...]` if the notebook uses a nonstandard dataframe variable name.

## Return Values

Most writer functions return small pandas dataframes read back from PostgreSQL. These are row-count or recent-metadata summaries intended for notebook display and sanity checks.

Functions that only record metadata return `None`.

## Common Failure Points

- Missing dataframe variable when using notebook-global lookup.
- Non-dataframe object passed as `dataframe`.
- Missing score or flag columns in scored Gold outputs.
- Missing `model` column in the comparison dataframe.
- No baseline or cascade rows in the comparison dataframe.
- Missing PostgreSQL tables if bootstrap has not been run.
- Unsafe schema or table identifiers.

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted database tests that use this module.
