# Utility Module Reference: `utils/database/medallion_sql_writers.py`

## Module Purpose

This module writes Medallion-layer notebook outputs and metadata into the project PostgreSQL schema.

## Pipeline Role

- Stage support: Database / SQL persistence
- Primary responsibility: This module writes Medallion-layer notebook outputs and metadata into the project PostgreSQL schema.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`, `EDA_Notebook_Pump_Silver_01_PreEDA`, `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `_resolve_dataframe` | Resolve the dataframe to write. | deep |
| `_execute_many` | Execute a parameterized SQL statement for many rows in chunks. | deep |
| `_delete_dataset_run_rows` | Delete existing rows for one dataset/run before writing notebook outputs. | deep |
| `_delete_model_score_rows` | Delete existing score rows for one dataset/run/model/stage. | deep |
| `_stage_pipeline_run_id` | Build a stage-specific pipeline_runs.run_id. | short |
| `_upsert_pipeline_run` | Upsert one capstone.pipeline_runs row for a notebook/stage. | deep |
| `_log_data_quality_event` | Insert one capstone.data_quality_events row. | deep |
| `_log_pipeline_artifact` | Insert one capstone.pipeline_artifacts row. | deep |
| `write_bronze_sensor_observations_sql` | Write final Bronze dataframe rows to bronze.sensor_observations. | deep |
| `write_silver_sensor_observation_features_sql` | Write final Silver dataframe rows to silver.sensor_observation_features. | deep |
| `log_silver_eda_sql` | Log Silver EDA profile metadata to capstone metadata tables. | deep |
| `write_silver_eda_sql_outputs` | Write Silver 02b EDA summary outputs to PostgreSQL. | deep |
| `_ensure_gold_preprocessed_features_table` | Create or migrate gold.preprocessed_features. | deep |
| `write_gold_preprocessed_features_sql` | Write Gold preprocessed features to gold.preprocessed_features. | deep |
| `write_gold_anomaly_scores_sql` | Generic writer for gold.anomaly_detection_scores. | deep |
| `write_gold_baseline_scores_sql` | Convenience wrapper for baseline Isolation Forest scores. | deep |
| `write_gold_cascade_scores_sql` | Convenience wrapper for cascade anomaly scores. | deep |
| `write_gold_model_comparison_results_sql` | Write baseline-vs-cascade comparison summary to gold.model_comparison_results. | deep |
| `log_gold_05_anomaly_detection_summary_sql` | Log Gold 05 anomaly-detection summary metadata. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.
- Dataset, run, stage, or recipe identifiers used for traceability.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_resolve_dataframe` | `*, dataframe, candidate_names, notebook_globals` | Resolve the dataframe to write. |
| `_execute_many` | `engine, sql, rows, *, chunk_size` | Execute a parameterized SQL statement for many rows in chunks. |
| `_delete_dataset_run_rows` | `engine, *, schema, table, dataset_id, run_id` | Delete existing rows for one dataset/run before writing notebook outputs. |
| `_delete_model_score_rows` | `engine, *, schema, table, dataset_id, run_id, model_name, model_stage` | Delete existing score rows for one dataset/run/model/stage. |
| `_stage_pipeline_run_id` | `run_id, pipeline_stage` | Build a stage-specific pipeline_runs.run_id. |
| `_upsert_pipeline_run` | `engine, *, capstone_schema, dataset_id, run_id, pipeline_stage, dataset_name, pipeline_mode, run_status, source_system, notes, runtime_facts` | Upsert one capstone.pipeline_runs row for a notebook/stage. |
| `_log_data_quality_event` | `engine, *, capstone_schema, dataset_id, run_id, layer_name, table_name, check_name, check_status, severity, row_count, details_json` | Insert one capstone.data_quality_events row. |
| `_log_pipeline_artifact` | `engine, *, capstone_schema, dataset_id, run_id, layer_name, stage_name, artifact_name, artifact_type, artifact_path, truth_hash, parent_truth_hash, metadata_json` | Insert one capstone.pipeline_artifacts row. |
| `write_bronze_sensor_observations_sql` | `engine, *, capstone_schema, layer_schema, dataset_id, run_id, notebook_globals, dataframe, dataset_name, candidate_names` | Write final Bronze dataframe rows to bronze.sensor_observations. |
| `write_silver_sensor_observation_features_sql` | `*, engine, capstone_schema, dataset_id, run_id, notebook_globals, dataframe, dataset_name, candidate_names, feature_set_id` | Write final Silver dataframe rows to silver.sensor_observation_features. |
| `log_silver_eda_sql` | `*, engine, capstone_schema, dataset_id, run_id, notebook_globals, dataframe, dataset_name, candidate_names` | Log Silver EDA profile metadata to capstone metadata tables. |
| `write_silver_eda_sql_outputs` | `engine, dataset_id, run_id, notebook_name, profile_df, feature_statistics_df, missingness_summary_df, correlation_pairs_df, outlier_summary_df, categorical_distribution_df, artifact_index_df, schema` | Write Silver 02b EDA summary outputs to PostgreSQL. |
| `_ensure_gold_preprocessed_features_table` | `engine` | Create or migrate gold.preprocessed_features. |
| `write_gold_preprocessed_features_sql` | `*, engine, capstone_schema, dataset_id, run_id, notebook_globals, dataframe, dataset_name, candidate_names, feature_set_id` | Write Gold preprocessed features to gold.preprocessed_features. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.
- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `KeyError` for invalid input, missing context, or failed validation paths.
- Source raises `NameError` for invalid input, missing context, or failed validation paths.
- Source raises `TypeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`, `EDA_Notebook_Pump_Silver_01_PreEDA`, `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Module Importance

This module matters because SQL persistence and metadata logging must stay consistent across notebook reruns and Medallion handoffs.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
