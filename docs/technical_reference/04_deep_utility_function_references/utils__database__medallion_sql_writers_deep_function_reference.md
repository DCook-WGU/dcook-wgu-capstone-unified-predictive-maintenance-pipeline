# utils/database/medallion_sql_writers.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `utils/database/medallion_sql_writers.py` that need deeper explanation than the 071d module-level reference. The selected functions are the SQL persistence boundary for Bronze observations, Silver features, Gold model inputs, Gold anomaly scores, Gold model comparison output, and Gold 05 summary metadata.

## Source Grounding

Sources used:

- `utils/database/medallion_sql_writers.py`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__database__medallion_sql_writers_module_reference.md`
- `function_inventory.json`
- `technical_reference/01_notebook_workflow_references/`
- `technical_reference/00_project_manual/`
- Documentation standards under `docs_agent_pack/00_STANDARDS/`

The active utility source file is the source of truth.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
| -------- | -------------- | ------------------------ |
| `write_bronze_sensor_observations_sql` | Persists Bronze observation rows and Bronze SQL metadata | Bronze preprocessing SQL handoff |
| `write_silver_sensor_observation_features_sql` | Persists Silver feature rows and feature-set metadata | Silver feature persistence |
| `write_gold_preprocessed_features_sql` | Persists Gold preprocessed model-input rows | Gold preprocessing and model-input handoff |
| `write_gold_anomaly_scores_sql` | Persists Gold anomaly score rows for a model/stage | Baseline and cascade score persistence |
| `write_gold_model_comparison_results_sql` | Persists one baseline-vs-cascade comparison record | Gold model comparison reporting |
| `log_gold_05_anomaly_detection_summary_sql` | Logs Gold 05 summary metadata and known artifact paths | Gold 05 reporting and audit metadata |

## Module-Level Technical Context

`medallion_sql_writers.py` provides notebook-facing PostgreSQL writers for selected Medallion outputs. The assigned functions translate pandas dataframes and notebook metadata into layer tables and project metadata tables while preserving the file/artifact workflow used elsewhere in the project.

The functions are intentionally specific to project table contracts. They do not replace the broader artifact or truth-record system. Their role is to make key notebook outputs queryable, rerunnable by `dataset_id` and `run_id`, and visible through pipeline metadata and data quality event rows.

## Deep Function References

### `write_bronze_sensor_observations_sql`

#### Functional Purpose

`write_bronze_sensor_observations_sql` writes Bronze observation-level records into `bronze.sensor_observations`. It converts a Bronze dataframe into a SQL row contract that keeps dataset/run identity, asset and time identifiers, source-row metadata, the raw row payload, and truth hash fields when present.

#### Pipeline Context

The Bronze workflow references confirm this function as the Bronze SQL persistence step when PostgreSQL writing is enabled. It supports the handoff from Bronze notebook output into a queryable Bronze layer table.

#### Inputs and Assumptions

Important inputs include the SQLAlchemy `engine`, metadata schema name, Bronze layer schema name, `dataset_id`, `run_id`, optional `dataset_name`, and either a direct dataframe or notebook globals containing one of the expected Bronze dataframe names.

The source dataframe is copied by the shared dataframe resolver and then reset with the index retained as a column. The writer looks for project-recognized columns such as `meta__asset_id`, `asset_id`, `event_time`, `timestamp`, `event_step`, `time_index`, `meta__source_file`, `meta__record_id`, `meta__truth_hash`, and `meta__parent_truth_hash`. Missing optional fields are written as null values rather than synthesized from unrelated columns.

#### Outputs and Return Contract

The function returns a dataframe read back from PostgreSQL with `dataset_id`, `run_id`, and the written row count grouped for the current dataset/run.

#### Side Effects

Confirmed side effects are:

- Deletes existing `bronze.sensor_observations` rows for the same `dataset_id` and `run_id`.
- Inserts one SQL row per resolved Bronze dataframe row.
- Writes a JSON raw payload for each source row.
- Upserts a row into the configured metadata schema's `pipeline_runs` table for the Bronze preprocessing stage.
- Inserts a `data_quality_events` row named `bronze_sql_write`.

#### Failure Behavior and Guardrails

The shared resolver raises `TypeError` when a direct dataframe argument is not a pandas dataframe, `ValueError` when neither a dataframe nor notebook globals are available, and `NameError` when no expected dataframe is found in notebook globals. Database write failures are not caught inside this function.

#### Lineage and Reproducibility Role

The SQL rows preserve `dataset_id`, `run_id`, source row identifiers, `meta_truth_hash`, and `meta_parent_truth_hash` when those fields are available in the source dataframe. The delete-before-insert pattern supports reruns for the same dataset/run by replacing the previous SQL representation. The pipeline metadata row records row count, column count, and target table information.

#### Why This Function Matters

Bronze SQL persistence is the earliest database-backed layer handoff. It gives reviewers a way to verify that raw or lightly processed sensor observations were written with run identity and source payload context, which supports auditability before Silver feature construction.

#### Verification Method

- Confirm `bronze.sensor_observations` contains rows for the expected `dataset_id` and `run_id`.
- Confirm the returned summary row count matches the source Bronze dataframe length.
- Inspect `raw_payload` for representative rows to verify source-row preservation.
- Confirm `meta_truth_hash` and `meta_parent_truth_hash` are populated when source columns exist.
- Confirm a `bronze_sql_write` data quality event exists for the same dataset/run.

### `write_silver_sensor_observation_features_sql`

#### Functional Purpose

`write_silver_sensor_observation_features_sql` writes Silver feature rows into `silver.sensor_observation_features`. It separates feature values from quality and metadata columns so the SQL table keeps a compact row contract with `features_json`, `quality_json`, feature-set identity, and lineage fields.

#### Pipeline Context

The function supports Silver feature persistence after Silver cleaning and feature preparation. Existing workflow and project manual references confirm Silver SQL persistence as part of the database-backed Medallion handoff, although specific notebook usage may vary by notebook.

#### Inputs and Assumptions

Important inputs include the SQLAlchemy `engine`, metadata schema, `dataset_id`, `run_id`, optional dataframe or notebook globals, optional `dataset_name`, and `feature_set_id`.

The writer resolves a Silver dataframe from direct input or common notebook variable names. It treats recognized dataset/run/time/asset columns and meta columns as canonical or quality context. Feature columns are inferred from remaining non-meta columns.

#### Outputs and Return Contract

The function returns a dataframe read back from PostgreSQL with row counts grouped by `dataset_id`, `run_id`, and `feature_set_id`.

#### Side Effects

Confirmed side effects are:

- Deletes existing `silver.sensor_observation_features` rows for the same `dataset_id` and `run_id`.
- Inserts Silver feature rows with `features_json` and `quality_json`.
- Upserts a Silver preprocessing pipeline metadata row.
- Inserts a `data_quality_events` row named `silver_sql_write`.

#### Failure Behavior and Guardrails

The shared dataframe resolver provides input-type and missing-dataframe guardrails. The function does not raise a separate error when inferred feature columns are sparse; it writes the available non-canonical, non-meta columns into `features_json`. Database failures propagate from SQL execution.

#### Lineage and Reproducibility Role

Each row carries `dataset_id`, `run_id`, `feature_set_id`, time/asset identifiers when available, and truth hash fields when available. Pipeline metadata records feature and quality column counts, which helps validate that the SQL handoff captured the intended feature population.

#### Why This Function Matters

Silver SQL persistence turns feature-engineered observations into queryable records without flattening every feature into fixed table columns. The JSON payload structure keeps the writer flexible while retaining enough run and feature-set identity for audit checks.

#### Verification Method

- Confirm `silver.sensor_observation_features` row count matches the resolved Silver dataframe.
- Confirm the returned summary includes the expected `feature_set_id`.
- Inspect `features_json` to verify expected feature columns are present.
- Inspect `quality_json` to verify meta or quality-related fields are retained when present.
- Confirm a `silver_sql_write` data quality event exists for the run.

### `write_gold_preprocessed_features_sql`

#### Functional Purpose

`write_gold_preprocessed_features_sql` writes Gold preprocessed model-input rows into `gold.preprocessed_features`. It persists the model-ready feature payload, train/test split metadata when present, and lineage columns needed to verify model input reproducibility.

#### Pipeline Context

Gold workflow references confirm this writer as the SQL persistence path for Gold preprocessed features. It supports the Gold model-input stage before baseline and cascade modeling.

#### Inputs and Assumptions

Important inputs include the SQLAlchemy `engine`, metadata schema, `dataset_id`, `run_id`, optional direct dataframe or notebook globals, optional `dataset_name`, and `feature_set_id`.

The function resolves a Gold preprocessed dataframe from a list of expected notebook variable names. It excludes known identity, split, and metadata columns from the feature payload. Split identity is read from `meta__split` or `split_name`; training status is read from `meta__is_train_flag`, `meta__is_train`, or `is_train` when present.

#### Outputs and Return Contract

The function returns a dataframe read back from PostgreSQL with row counts grouped by `dataset_id`, `run_id`, and `split_name`.

#### Side Effects

Confirmed side effects are:

- Ensures the `gold.preprocessed_features` table structure and indexes exist through the module's table preparation helper.
- Deletes existing rows for the same `dataset_id` and `run_id`.
- Inserts preprocessed feature rows with `features_json`.
- Upserts Gold preprocessing pipeline metadata.
- Inserts a `data_quality_events` row named `gold_preprocessing_sql_write`.

#### Failure Behavior and Guardrails

The shared resolver raises the documented dataframe resolution errors. The table preparation helper creates or updates the target table structure before writing. Database failures propagate from SQL execution.

#### Lineage and Reproducibility Role

The SQL row contract preserves `dataset_id`, `run_id`, feature-set identity, split name, training-row indicator, and truth hash fields when available. This allows a reviewer to compare model-ready SQL rows against the Gold preprocessing output and confirm which rows were eligible for training or evaluation.

#### Why This Function Matters

Gold model validity depends on reproducible model inputs. Persisting the preprocessed feature payload and split metadata makes the modeling stage auditable outside notebook-local variables.

#### Verification Method

- Confirm `gold.preprocessed_features` exists and contains rows for the expected dataset/run.
- Compare grouped `split_name` counts to the Gold preprocessing output.
- Inspect `features_json` for expected model-input feature names.
- Confirm `is_train` values align with the preprocessing split contract when present.
- Confirm a `gold_preprocessing_sql_write` data quality event exists.

### `write_gold_anomaly_scores_sql`

#### Functional Purpose

`write_gold_anomaly_scores_sql` writes scored anomaly detection outputs into `gold.anomaly_detection_scores` for one dataset/run/model/stage combination. It records score values, anomaly flags, alert severity, evidence payloads, model identity, and available lineage fields.

#### Pipeline Context

This is the generic Gold score writer used by model-specific score persistence wrappers. It supports baseline and cascade score persistence by accepting model-specific score and flag column candidates.

#### Inputs and Assumptions

Important inputs include the SQLAlchemy `engine`, metadata schema, `dataset_id`, `run_id`, `model_name`, `model_stage`, ordered score-column candidates, ordered flag-column candidates, and either a direct scored dataframe or notebook globals.

The writer requires one score column and one anomaly flag column to be found in the resolved dataframe. For `evidence_column_mode="basic"`, the evidence payload contains the score and flag columns. For `evidence_column_mode="cascade"`, the evidence payload also captures columns whose names indicate stage, cascade, breach, drift, persistence, corroboration, or evidence context.

#### Outputs and Return Contract

The function returns a dataframe read back from PostgreSQL with `model_name`, `model_stage`, written row count, and alert count for the current dataset/run/model/stage.

#### Side Effects

Confirmed side effects are:

- Deletes existing `gold.anomaly_detection_scores` rows for the same `dataset_id`, `run_id`, `model_name`, and `model_stage`.
- Inserts one anomaly score row per source dataframe row.
- Writes evidence payloads as JSON.
- Upserts a pipeline metadata row for `gold_<model_stage>_modeling`.
- Inserts a data quality event named `<model_stage>_sql_write`.

#### Failure Behavior and Guardrails

The function raises `KeyError` if none of the score-column candidates or none of the flag-column candidates are present. The shared resolver raises dataframe input errors. Database failures propagate from SQL execution.

#### Lineage and Reproducibility Role

Rows carry `dataset_id`, `run_id`, `model_name`, `model_stage`, score and flag values, `meta_truth_hash`, and `meta_parent_truth_hash` when present. The metadata row stores the selected score column, selected flag column, row count, alert count, and evidence column count, which makes score persistence reproducible and reviewable.

#### Why This Function Matters

Gold model comparison depends on score outputs being persisted under stable model and stage identifiers. This function provides the common SQL contract for baseline and cascade results without requiring each notebook to build its own insert logic.

#### Verification Method

- Confirm `gold.anomaly_detection_scores` contains rows for the expected dataset/run/model/stage.
- Confirm the returned `row_count` matches the scored dataframe length.
- Confirm `alert_count` equals the sum of the selected anomaly flag column.
- Inspect `evidence_json` to verify selected score and flag column names are recorded.
- Confirm rerunning the writer replaces prior rows for the same dataset/run/model/stage.

### `write_gold_model_comparison_results_sql`

#### Functional Purpose

`write_gold_model_comparison_results_sql` writes a baseline-vs-cascade comparison record into `gold.model_comparison_results`. It extracts the first comparison row whose model label contains `baseline` and the first row whose model label contains `cascade`, then stores their alert and metric values together with the full source comparison dataframe in `comparison_json`.

#### Pipeline Context

Workflow references confirm this writer as part of Gold 04 comparison persistence. It supports reporting and audit review of the baseline-vs-cascade comparison.

#### Inputs and Assumptions

Important inputs include the SQLAlchemy `engine`, metadata schema, `dataset_id`, `run_id`, optional direct comparison dataframe or notebook globals, optional `dataset_name`, and optional candidate dataframe names.

The resolved comparison dataframe must include a `model` column. Metric extraction looks for alert-count, precision, recall, and F1 columns using source-confirmed candidate names. The function writes one selected comparison record, not one SQL row per source comparison row.

#### Outputs and Return Contract

The function returns up to five recent comparison records for the current dataset/run from `gold.model_comparison_results`, including model labels, alert counts, precision, recall, F1 values, and creation time.

#### Side Effects

Confirmed side effects are:

- Deletes existing `gold.model_comparison_results` rows for the same `dataset_id` and `run_id`.
- Inserts one baseline-vs-cascade comparison record.
- Stores the full source comparison dataframe and source column names in `comparison_json`.
- Upserts Gold model comparison pipeline metadata.
- Inserts a `data_quality_events` row named `gold_comparison_sql_write`.

#### Failure Behavior and Guardrails

The function raises `KeyError` if the comparison dataframe has no `model` column. It raises `ValueError` if no baseline row or no cascade row can be found. The shared resolver raises dataframe input errors. Database failures propagate from SQL execution.

#### Lineage and Reproducibility Role

The record carries `dataset_id`, `run_id`, model labels, selected metrics, and a JSON copy of the source comparison rows. The metadata row stores the same comparison fields so the model comparison can be reviewed without relying only on notebook-local variables.

#### Why This Function Matters

The comparison table is a compact evaluator-facing SQL output. It preserves the key baseline/cascade comparison metrics while retaining the wider source comparison payload for traceability.

#### Verification Method

- Confirm exactly one current comparison record is present for the dataset/run after the delete-and-insert cycle.
- Confirm `baseline_model` and `comparison_model` match the intended source comparison rows.
- Inspect `comparison_json` to verify the full source comparison dataframe was retained.
- Confirm `gold_comparison_sql_write` exists with row count `1`.
- Confirm the function fails when the comparison dataframe lacks a `model` column.

### `log_gold_05_anomaly_detection_summary_sql`

#### Functional Purpose

`log_gold_05_anomaly_detection_summary_sql` logs Gold 05 anomaly-detection summary metadata into the project metadata tables. It summarizes the resolved dataframe, records selected-run context when available, logs a data quality event, and registers known Gold 05 artifact path variables when they are present in notebook globals.

#### Pipeline Context

Gold 05 workflow references confirm this function as the SQL metadata logging step for the anomaly-detection summary notebook. It supports final reporting auditability rather than writing a new layer dataframe.

#### Inputs and Assumptions

Important inputs include the SQLAlchemy `engine`, metadata schema, `dataset_id`, `run_id`, optional summary dataframe or notebook globals, and optional `dataset_name`.

The function resolves a summary dataframe from direct input or Gold 05 candidate variable names. If notebook globals are supplied, it reads `SELECTED_RUN_KEY` and a fixed list of artifact path variables such as `baseline_summary_path`, `cascade_summary_path`, `comparison_summary_path`, `baseline_vs_cascade_path`, `gold_anomaly_detection_ledger_path`, `selected_run_artifact_path`, `model_comparison_plot_path`, and `final_report_table_path`.

#### Outputs and Return Contract

The function returns recent pipeline run metadata rows from the configured metadata schema where the runtime facts identify the source `run_id`.

#### Side Effects

Confirmed side effects are:

- Upserts a `pipeline_runs` row for `gold_anomaly_detection_summary`.
- Inserts a `data_quality_events` row named `gold_05_summary_sql_log`.
- Logs known Gold 05 artifact path variables into `pipeline_artifacts` when those variables are present in notebook globals.
- Does not create artifact files and does not write ledger entries.

#### Failure Behavior and Guardrails

The shared resolver raises dataframe input errors. The function only logs artifact path variables that exist in notebook globals and are not null. Database failures propagate from SQL execution.

#### Lineage and Reproducibility Role

The runtime facts include row count, column count, source columns, selected run key, model labels when a `model` column exists, and alert-count summaries when `alert_count_all_rows` exists. Artifact metadata links known Gold 05 output paths to dataset/run identity and selected-run context.

#### Why This Function Matters

Gold 05 is a final reporting stage. This function makes the selected summary and known output artifacts visible in metadata tables, which supports review of the final anomaly-detection narrative without requiring every reviewer to inspect notebook-local state.

#### Verification Method

- Confirm a `pipeline_runs` row exists for `gold_anomaly_detection_summary`.
- Confirm runtime facts include row count, column count, and selected-run key when available.
- Confirm a `gold_05_summary_sql_log` data quality event exists.
- Confirm expected Gold 05 artifact path variables were logged only when present in notebook globals.
- Confirm no artifact file creation is attributed to this function.

## Cross-Function Relationships

The assigned functions form a SQL persistence path across the Medallion workflow:

- Bronze, Silver, and Gold preprocessing writers persist layer-specific dataframe outputs under dataset/run identity.
- The Gold score writer persists model-stage outputs for later comparison and reporting.
- The Gold comparison writer condenses baseline/cascade comparison metrics into a SQL record while retaining the source comparison rows in JSON.
- The Gold 05 summary logger writes metadata and artifact path references for the final anomaly-detection reporting stage.
- All assigned writer functions use shared metadata logging helpers for pipeline run and data quality event records when source-confirmed.

## Source-Limited Items

- Direct notebook usage of `write_gold_anomaly_scores_sql` itself was not determined from available source; model-specific wrappers and module references indicate its intended score persistence role.
- Whether every SQL table already exists before execution is not fully determined from available source. `write_gold_preprocessed_features_sql` includes source-confirmed table preparation, while the other assigned writers rely on existing SQL table contracts.
- Transaction rollback behavior beyond the shared SQLAlchemy `engine.begin()` helper behavior was not determined from available source.
