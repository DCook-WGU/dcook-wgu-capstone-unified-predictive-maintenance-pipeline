# Bronze 01 Deep Technical Reference

## Purpose of This Deep Reference

This document explains the technical decisions in Bronze 01 that require deeper detail than the workflow reference. It focuses on the design choices that make the Bronze output reproducible, traceable, SQL-ready, and usable as the first Medallion handoff.

## Technical Scope

This reference covers:

- Bronze notebook context and configuration setup.
- Raw file ingestion and the inactive PostgreSQL handoff branch present in source.
- Dataset, run, asset, and source-row identity handling.
- Bronze dataframe metadata construction.
- Artifact directory creation, config snapshot export, and Parquet persistence.
- Bronze truth-record creation, truth hash stamping, and truth validation.
- Profiling, W&B tracking, logging, and ledger behavior.
- Bronze SQL persistence and data quality metadata behavior.
- Source-confirmed downstream handoff limits.

## Source Grounding

Sources used:

- Active notebook source: `notebooks/preprocessing/EDA_Notebook_Pump_Bronze_01_Preprocessing.ipynb`
- `notebook_inventory.json`
- `artifact_io_manifest.json`
- `sql_touchpoints.json`
- Existing Bronze workflow reference: `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Bronze_01_Preprocessing_code_reference.md`
- Project manual relationship maps under `technical_reference/00_project_manual/`
- Relevant utility source and deep references under `technical_reference/04_deep_utility_function_references/`
- Documentation standards under `docs_agent_pack/00_STANDARDS/`

The active Bronze 01 notebook is the source of truth. Existing documentation and manifests are used as supporting context only.

## Stage Role in the Medallion Pipeline

Bronze 01 is the first active notebook in the core Medallion path. Its technical role is to ingest pump telemetry, attach Bronze metadata, establish the root truth record, save the Bronze dataframe as a Parquet artifact, and write a SQL representation of the Bronze observations when SQL writing is enabled.

The active notebook sets `CONTEXT_STAGE = "bronze"`, `CONTEXT_LAYER = "bronze"`, `CONFIG_RUN_MODE = "train"`, and `CONFIG_PROFILE = "default"` before calling `load_notebook_context`. That context supplies the resolved configuration, paths, filenames, logger, ledger, runtime settings, and dataset settings used by the rest of the notebook.

Bronze 01 also contains a PostgreSQL handoff branch for `BRONZE_SOURCE_MODE == "postgres_handoff"`, but the active source sets `BRONZE_SOURCE_MODE = "file"`. The documented active ingestion path is therefore file-based unless the notebook source is changed.

## Input Contract and Lineage

The active input contract is driven by the resolved dataset configuration and Bronze stage configuration:

- `RAW_FILE_PATH / RAW_FILE_NAME` is the active source when `BRONZE_SOURCE_MODE = "file"`.
- `DATASET_CFG["name"]`, `DATASET_CFG["run_id"]`, and `DATASET_CFG["asset_id"]` provide the configured dataset, run, and asset identifiers.
- `BRONZE_CFG["dataset_candidates"]` controls dataset-name discovery inside the raw dataframe.
- `BRONZE_CFG["add_record_id"]`, `BRONZE_CFG["record_id_inputs"]`, and `BRONZE_CFG["record_id_method"]` control stable record-id behavior.
- Environment variables can override SQL-facing `DATASET_ID`, `RUN_ID`, and `ASSET_ID` through `CAPSTONE_DATASET_ID`, `CAPSTONE_RUN_ID`, and `CAPSTONE_ASSET_ID`.
- `CAPSTONE_SCHEMA` defaults to `capstone` when the environment variable is absent.

The notebook resolves two forms of dataset identity. A provisional name is computed before ingestion so tracking metadata can be initialized. The final name is read from `dataframe.attrs["dataset_resolution"]` after ingestion. The notebook raises `ValueError` if that attribute is missing or if the final dataset name is `None`.

The inactive PostgreSQL handoff branch reads `public.bronze_observations_input_stage` through `read_layer_dataframe`, filters by `dataset_id` and `run_id`, orders by `batch_id, row_in_batch`, requires the table to exist, and writes dataset-resolution attributes back to the dataframe. This branch is source-confirmed as available logic, not the active default path.

Lineage matters at Bronze because downstream layers depend on this stage as the first traceable representation of the telemetry. Bronze has no parent truth hash. Its truth record is the root that downstream truth records can reference.

## Bronze Data Preparation Methodology

The file-ingestion path calls `ingest_data` with the raw path, configured dataset candidates, split label, run ID, asset ID, `add_record_id=True`, and `validate=True`. Active utility source confirms that this adds the Bronze metadata contract:

- `meta__dataset`
- `meta__split`
- `meta__run_id`
- `meta__asset_id`
- `meta__ingested_at_utc`
- `meta__source_file`
- `meta__source_row_id`
- Optional `meta__record_id`

The utility resolves the dataset name from argument, config, inside-dataset candidates, file details, or fallback. It stores the selected dataset name, source column, method, and priority order in `dataframe.attrs["dataset_resolution"]`.

Bronze does not perform Silver-style cleaning or feature engineering. The notebook keeps non-metadata sensor columns in their raw loaded form, then moves `meta__` columns to the front for readability and downstream consistency. The reorder helper returns a copy of the dataframe to avoid accidental pandas view behavior.

Timestamp parsing is not explicitly performed in the active Bronze notebook source. Any timestamp-like columns are passed through from the loaded dataframe and later mapped by SQL writer candidates such as `event_time`, `timestamp`, `datetime`, or `date_time` if present.

## Bronze Validation and Data Quality Checks

Bronze 01 uses several guardrails before and after persistence:

| Check | Source-Confirmed Behavior | Failure / Risk Prevented |
|---|---|---|
| Shared context sanity check | Verifies required context variables are present after `load_notebook_context`. | Prevents later execution with missing paths, config, logger, or ledger objects. |
| SQL smoke check | Reads `information_schema.tables` for configured schemas. | Detects database connectivity or schema visibility problems early. |
| Dataset resolution check | Requires `dataframe.attrs["dataset_resolution"]` and non-null final dataset name. | Prevents artifacts and truth records from using an unknown dataset identity. |
| Ingestion row identity checks | `ingest_data(validate=True)` requires unique, zero-based, contiguous `meta__source_row_id` values and no duplicate source-file/source-row keys. | Prevents row expansion, duplicate lineage keys, or broken source-row mapping. |
| Record-id uniqueness check | When record IDs are added, `meta__record_id` must be unique. | Protects stable row joins and deduplication checks. |
| Empty ingest check | `ingest_data` raises `ValueError` for an empty dataframe. | Prevents persisting an empty Bronze stage as valid output. |
| Unsupported source mode check | W&B setup and ingest both raise `ValueError` for unsupported `BRONZE_SOURCE_MODE`. | Prevents ambiguous source metadata or unexpected ingest behavior. |
| Truth metadata check | Requires `meta__truth_hash`, `meta__parent_truth_hash`, and `meta__pipeline_mode` in the final dataframe. | Confirms truth stamping occurred before final validation. |
| Truth hash match check | Compares `extract_truth_hash(dataframe)` to `BRONZE_TRUTH_HASH`. | Confirms row-level lineage matches the saved truth record. |
| Bronze parent hash check | Requires no populated parent truth hash values. | Preserves the Bronze chain-root invariant. |
| Saved truth file check | Confirms the truth JSON exists and contains matching hash, row count, column count, and `parent_truth_hash=None`. | Protects the handoff from pointing at missing or inconsistent truth metadata. |

The SQL writer also logs a `bronze_sql_write` data quality event after rows are inserted. This is part of SQL persistence behavior rather than the notebook's in-memory validation block.

## Artifact and SQL Persistence

Bronze 01 writes several source-confirmed outputs:

| Output | Mechanism | Location / Table | Notes |
|---|---|---|---|
| Resolved config snapshot | `export_config_snapshot` | `BRONZE_CONFIG_DIR / <dataset>__bronze__resolved_config.yaml` | Written only when `CONFIG["execution"].get("save_config_snapshot", True)` is true. |
| Bronze truth record | `save_truth_record` | `TRUTHS_PATH / <layer-specific truth file>` | File path is returned as `bronze_truth_path`. |
| Truth index entry | `append_truth_index` | `TRUTH_INDEX_PATH` | Adds the Bronze truth record to the project-wide index. |
| Bronze dataframe artifact | `save_data` | `paths.data_bronze_train / BRONZE_TRAIN_DATA_FILE_NAME` | Active source saves before profiling so the primary artifact exists even if profiling fails. |
| Bronze profiling outputs | `profile_dataframe` | `BRONZE_PROFILE_DIR` | Returns metrics and saved-output metadata. |
| W&B dataset artifact | `finalize_wandb_stage` | W&B run context | Uses the Bronze dataframe, logs directory, dataset directory, dataset artifact name, and notebook path supplied by the notebook. |
| Bronze SQL observations | `write_bronze_sensor_observations_sql` | `bronze.sensor_observations` by default | Active source sets `WRITE_TO_POSTGRES = True`. |
| SQL metadata row | SQL writer helper | configured `capstone_schema.pipeline_runs` | Records `pipeline_stage="bronze_preprocessing"` and runtime facts. |
| SQL data quality event | SQL writer helper | configured `capstone_schema.data_quality_events` | Logs `check_name="bronze_sql_write"`. |

The SQL writer deletes existing Bronze observation rows for the same `dataset_id` and `run_id`, inserts one SQL row per Bronze dataframe row, stores the source row as `raw_payload` JSON, carries truth hash fields when present, upserts a pipeline run row, logs a data quality event, and returns a row-count summary read back from the SQL table.

The notebook previews the SQL table with `preview_sql_table(schema=LAYER_SCHEMA, table="sensor_observations", limit=5)` and disposes the SQLAlchemy engine at the end.

## Truth, Audit, and Reproducibility Behavior

Bronze 01 establishes the root of the project truth chain:

- `initialize_layer_truth` is called with `parent_truth_hash=None`.
- The truth payload records `truth_version`, `dataset_name`, `layer_name`, `process_run_id`, `pipeline_mode`, Bronze version, split status, run ID, asset ID, record-id settings, raw file path, raw data directory, provisional and final dataset names, dataset resolution method, and Bronze output path information.
- `build_file_fingerprint(RAW_FILE_PATH / RAW_FILE_NAME)` records source file fingerprint data.
- `build_truth_record` computes the final truth hash using the pre-stamped dataframe facts. The notebook adds three to the pre-stamp column count because `stamp_truth_columns` adds `meta__truth_hash`, `meta__parent_truth_hash`, and `meta__pipeline_mode`.
- `stamp_truth_columns` returns a stamped dataframe assigned back to `dataframe`.
- The final validation block verifies the saved truth JSON against the live dataframe.

This design separates stage-level reproducibility from row-level lineage. The truth JSON records how the Bronze output was produced; the dataframe columns carry enough truth metadata for downstream stages to verify which truth record their rows came from.

Logging and ledger behavior are also source-confirmed. The notebook logs context loading, context sanity, layer paths, data review summaries, column reorder counts, truth hash/path, and SQL-related outputs. Ledger entries are added for context loading, context sanity checking, and path logging.

W&B tracking is active in the notebook source. The run starts with provisional source metadata, then the config is updated with the final dataset identity after ingestion. The run is finalized with `finalize_wandb_stage` and closed with `wandb_run.finish()`.

## Downstream Technical Handoff

Bronze 01 source-confirms these handoff objects:

- A Bronze Parquet artifact is written through `save_data` to `paths.data_bronze_train / BRONZE_TRAIN_DATA_FILE_NAME`.
- A Bronze truth record JSON is written and indexed.
- Bronze observations are written to `bronze.sensor_observations` when `WRITE_TO_POSTGRES` is true.
- SQL metadata and data-quality rows are written by the Bronze SQL writer.

Project manual relationship maps and the existing Bronze workflow reference state that Silver 01 consumes the Bronze output and Bronze truth context. The active Bronze notebook source itself does not load or inspect Silver 01, so the exact downstream Silver read path is not established by Bronze source alone.

Direct file-level dependency from Bronze 01 to Silver 01: Not determined from available source.

## Key Technical Decisions

| Decision | Source Evidence | Why It Matters | Verification Method |
|---|---|---|---|
| Use shared notebook context for Bronze setup. | Active notebook calls `load_notebook_context(stage="bronze", dataset="pump", mode="train", profile="default")`. | Centralizes config, paths, logger, ledger, filenames, and runtime metadata before any outputs are written. | Confirm `CTX`, `paths`, `CONFIG`, `BRONZE_CFG`, `logger`, and `ledger` are available after context load. |
| Keep dataset identity provisional until ingest confirms it. | Notebook resolves `PROVISIONAL_DATASET_NAME`, then requires `dataframe.attrs["dataset_resolution"]` before creating artifact directories. | Prevents config snapshots and artifact directories from being written under an unconfirmed dataset name. | Confirm `RESOLVED_DATASET_NAME` is not `None` and artifact paths include the resolved name. |
| Defer artifact directory creation until after dataset resolution. | `BRONZE_ARTIFACT_DIRS = {}` is set before ingest; `build_artifact_dirs` runs only after final dataset identity is confirmed. | Avoids stale folders and snapshots tied to provisional identity. | Confirm Bronze artifact directories are created after ingestion and include config, lineage, profiles, quality, schema, summaries, and metadata subdirectories. |
| Treat Bronze as the truth-chain root. | `initialize_layer_truth` receives `parent_truth_hash=None`; final validation rejects populated parent truth hash values. | Establishes a clean root for Silver and Gold lineage. | Load the saved truth JSON and confirm `parent_truth_hash` is `None`. |
| Compute the truth hash before row-level truth stamping. | Notebook builds `bronze_truth_record` before calling `stamp_truth_columns` and explicitly accounts for three added stamp columns. | Avoids recursive hashing where the truth hash depends on a dataframe column containing that same hash. | Confirm saved `column_count` equals the stamped dataframe column count and `BRONZE_TRUTH_HASH` matches `extract_truth_hash(dataframe)`. |
| Move metadata columns to the front without changing values. | `reorder_bronze_columns` separates `meta__` columns, appends non-meta columns, and returns `.copy()`. | Keeps row metadata visible for review while preserving raw sensor column values. | Confirm all `meta__` columns precede non-meta columns and row count remains unchanged. |
| Save the Bronze Parquet before profiling. | Active notebook comment and source call `save_data` before `profile_dataframe`. | Ensures the main Bronze handoff artifact exists even if profiling fails later. | Confirm the Parquet file exists before relying on profile artifacts. |
| Validate the saved truth record against the live dataframe. | Final validation loads `bronze_truth_path`, checks `truth_hash`, row count, column count, and root parent hash. | Detects mismatches between persisted truth metadata and the dataframe that was saved/written. | Compare loaded truth JSON fields to `BRONZE_TRUTH_HASH` and `dataframe.shape`. |
| Gate SQL persistence through a notebook flag. | Active source sets `WRITE_TO_POSTGRES = True` before calling `write_bronze_sensor_observations_sql`. | Makes the SQL side effect explicit in the notebook source. | Confirm `WRITE_TO_POSTGRES` value and inspect the returned SQL summary dataframe. |
| Use delete-and-insert SQL behavior for dataset/run reruns. | `write_bronze_sensor_observations_sql` deletes existing `bronze.sensor_observations` rows for the same dataset/run before inserting. | Supports reproducible reruns without duplicate Bronze SQL rows for the same run identity. | Query `bronze.sensor_observations` by `dataset_id` and `run_id` and compare row count to the Bronze dataframe. |

## Failure Modes and Guardrails

Source-confirmed failure behavior includes:

- Missing shared context variables raise `NameError`.
- Missing dataset/run/asset identifiers after environment/config fallback resolution raise `ValueError`.
- Unsupported `BRONZE_SOURCE_MODE` raises `ValueError`.
- Unsupported raw file suffixes in `ingest_data` raise `ValueError`.
- Raw file read or write errors propagate after being logged by the file utility.
- Empty ingest output raises `ValueError`.
- Duplicate or non-contiguous source-row identifiers raise `ValueError`.
- Missing dataset-resolution dataframe attributes raise `ValueError`.
- Missing or malformed saved truth fields raise `KeyError`, `TypeError`, `ValueError`, or `FileNotFoundError` depending on the failing check.
- SQL table absence in the PostgreSQL handoff branch raises `FileNotFoundError` through `read_layer_dataframe(require_exists=True)`.
- SQL writer dataframe resolution errors can raise `TypeError`, `ValueError`, or `NameError`.
- Database write failures are not swallowed by the Bronze SQL writer.

Source-confirmed timestamp validation beyond SQL writer candidate mapping was not found in the active Bronze notebook.

## Verification Checklist

- Confirm the resolved active notebook path is `notebooks/preprocessing/EDA_Notebook_Pump_Bronze_01_Preprocessing.ipynb`.
- Confirm `BRONZE_SOURCE_MODE` is `"file"` before describing the active ingest path.
- Confirm the raw file path resolves from `RAW_FILE_PATH / RAW_FILE_NAME`.
- Confirm `dataframe.attrs["dataset_resolution"]` exists after ingestion.
- Confirm `RESOLVED_DATASET_NAME`, `DATASET_SOURCE_COLUMN`, and `DATASET_METHOD` are populated.
- Confirm `meta__dataset`, `meta__split`, `meta__run_id`, `meta__asset_id`, `meta__source_file`, `meta__source_row_id`, and `meta__record_id` exist when record IDs are enabled.
- Confirm `meta__truth_hash`, `meta__parent_truth_hash`, and `meta__pipeline_mode` exist after truth stamping.
- Confirm `meta__parent_truth_hash` contains no populated values for Bronze.
- Confirm the saved truth JSON exists and matches `BRONZE_TRUTH_HASH`, row count, and column count.
- Confirm the Bronze Parquet output exists at `paths.data_bronze_train / BRONZE_TRAIN_DATA_FILE_NAME`.
- Confirm the config snapshot exists when config snapshot saving is enabled.
- Confirm the SQL write summary row count matches the Bronze dataframe length when `WRITE_TO_POSTGRES` is true.
- Confirm no claim of Silver file-level consumption is made unless verified from Silver source or a trusted relationship map.

## Source-Limited Items

- Direct file-level dependency from Bronze 01 to Silver 01: Not determined from available source.
- Active Bronze source does not explicitly parse timestamp columns; timestamp normalization behavior is Not determined from available source.
- Whether W&B is disabled by environment at runtime was not determined from available source; the notebook source calls `wandb.init`, `finalize_wandb_stage`, and `wandb_run.finish()`.
- Runtime database state, row counts, and actual SQL write success were not verified because this task did not execute the notebook or connect to the database.
- The inactive PostgreSQL handoff branch is documented as source-confirmed code, but its runtime behavior in the current run mode is Not determined from available source.
