# Notebook Code Reference: EDA_Notebook_Pump_Bronze_01_Preprocessing

## Notebook Purpose

This notebook is the first stage of the capstone Medallion pipeline. It ingests raw pump telemetry data from a flat file or a PostgreSQL handoff table, assigns row-level identity and lineage metadata, and produces the canonical Bronze-layer dataset that all downstream Silver and Gold stages depend on.

Beyond loading data, it establishes the **truth chain root**. It initializes the Bronze truth record, fingerprints the source file, stamps every row with `meta__truth_hash`, `meta__parent_truth_hash`, and `meta__pipeline_mode`, and writes the truth record to disk. Silver references this record via `parent_truth_hash`. No upstream truth record exists; Bronze is `parent_truth_hash = None`.

Deliverables: 1.1.1 and 1.1.2.

---

## Pipeline Role

- **Stage:** Bronze
- **Position:** First notebook. No upstream notebook dependency.
- **Primary responsibility:** Raw ingest → identity resolution → truth chain root → Parquet artifact → PostgreSQL write → W&B tracking.

---

## Inputs

| Input | Source | Form | Used For |
|---|---|---|---|
| Raw telemetry file | `RAW_FILE_PATH / RAW_FILE_NAME` | CSV | Primary ingest when `BRONZE_SOURCE_MODE == "file"` |
| PostgreSQL handoff table | `public.<POSTGRES_SOURCE_TABLE_NAME>` via `read_layer_dataframe` | SQL table | Alternative ingest when `BRONZE_SOURCE_MODE == "postgres_handoff"` |
| Project config | `load_notebook_context(stage="bronze", dataset="pump")` | YAML → `CTX` object | All runtime constants, paths, stage config, fallbacks |
| Environment variables | OS environment | Strings | DB engine (`get_engine_from_env()`), `CAPSTONE_SCHEMA`, `DATASET_ID` override |

---

## Configuration and Runtime Context

| Item | Source | Purpose |
|---|---|---|
| `BRONZE_SOURCE_MODE` | `BRONZE_CFG` | `"file"` or `"postgres_handoff"` — determines ingest path |
| `DEFAULT_FALLBACKS` | `BRONZE_CFG` (assigned in Notebook Level Configuration) | Provides fallback values for Bronze preprocessing; required by the Bronze-specific sanity check |
| `WRITE_TO_POSTGRES` | Notebook cell (bool gate) | Controls whether the SQL write executes; allows offline runs |
| `DATASET_NAME` / `RUN_ID` / `ASSET_ID` | `DATASET_CFG` via `CTX` | Row-level identity fields written into every Bronze row |
| `BRONZE_VERSION` / `TRUTH_VERSION` | `VERSIONS_CFG` via `CTX` | Version stamps written into the truth record and artifact metadata |
| `PROCESS_RUN_ID` | Constructed from `BRONZE_CFG["process_run_id_prefix"]` + timestamp | Unique identifier for this Bronze execution; written into truth record |
| `PIPELINE_MODE` | `PIPELINE["execution_mode"]` | Propagated to truth record and to `meta__pipeline_mode` column |
| `CONTEXT_RECIPE_ID` | `CTX` | Identifies the cleaning recipe version for this Bronze run |
| `LAYER_SCHEMA` | `BRONZE_CFG` | PostgreSQL schema for the Bronze layer table |
| `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_RUN_NAME` | `WANDB_CFG` via `CTX` | W&B run initialization parameters |

---

## Logical Workflow Map

1. Load shared context via `load_notebook_context`; run general (17-var) and Bronze-specific (`BRONZE_CFG`, `DEFAULT_FALLBACKS`) sanity checks
2. Establish SQL engine; run SQL smoke check to confirm DB connectivity
3. Start logging; record first ledger entry
4. Define and execute dataset name resolution logic; resolve provisional dataset identity
5. Set W&B metadata for ingest source; initialize W&B run with `wandb.init`
6. Ingest raw data (`ingest_data` or `read_layer_dataframe`); confirm final dataset identity from `dataframe.attrs`
7. Create Bronze artifact directories now that dataset name is confirmed; export config snapshot
8. Update W&B run config with confirmed dataset name
9. Initialize Bronze truth record (`initialize_layer_truth`, `parent_truth_hash=None`); attach config snapshot section
10. Review data (shape, dtypes, missingness, descriptive stats)
11. Fingerprint source file; finalize truth record; stamp rows (`stamp_truth_columns`); save truth record to disk; append to truth index
12. Reorder columns — `meta__` columns move to front
13. Save Bronze Parquet (`save_data`); generate profiling artifacts (`profile_dataframe`)
14. Finalize W&B stage (`finalize_wandb_stage`); close run (`wandb_run.finish`)
15. Validate Bronze lineage and truth consistency; run final structural check
16. Write Bronze rows to PostgreSQL (`write_bronze_sensor_observations_sql`) if `WRITE_TO_POSTGRES`; preview output; dispose engine

---

## Section Overview

| Section | Purpose | Key Inputs | Key Outputs / Side Effects |
|---|---|---|---|
| Environment Setup | Import libraries; define local validation helpers | None | All imports and helpers available |
| Context Load | Call `load_notebook_context`; bind paths, config, logger, ledger | Config files, env vars | `CTX`, `BRONZE_CFG`, `DEFAULT_FALLBACKS`, `logger`, `ledger`; first `ledger.add` call |
| Notebook Level Configuration | Expand config into runtime constants | `CTX` attributes | `DATASET_NAME`, `RUN_ID`, `ASSET_ID`, `PROCESS_RUN_ID`, `DEFAULT_FALLBACKS`, all layer/version constants |
| Defer Artifact Folders | Placeholder until dataset name is confirmed post-ingest | None | `BRONZE_ARTIFACT_DIRS = {}` |
| SQL Runtime Context + Sanity Checks | Get DB engine; resolve `DATASET_ID` / `RUN_ID` from env or config; verify 17 general + 2 Bronze-specific context vars | Globals dict, env vars, `BRONZE_CFG` | `engine`, `DATASET_ID`, `RUN_ID`; `ledger.add`; raises `NameError` on missing vars |
| SQL Smoke Check | Query `information_schema.tables` to confirm DB is live | `engine` | `sql_smoke_check_dataframe`; confirms schema accessible |
| Start Logging | Call `log_layer_paths`; record layer start in logger and ledger | `paths`, `logger`, `ledger` | Layer path log written; `ledger.add` for stage start |
| Dataset Name Resolution | Define resolution priority logic; resolve provisional name | Config value, file path or Postgres source | `PROVISIONAL_DATASET_NAME` and resolution method |
| W&B Source Setup | Set W&B metadata fields for the ingest source type | `BRONZE_SOURCE_MODE` | `WANDB_SOURCE_KIND`, `WANDB_SOURCE_REFERENCE`; raises `ValueError` on unknown mode |
| W&B Init | `wandb.init` with source metadata and provisional dataset name | `WANDB_CFG`, provisional identifiers | Active `wandb_run` |
| Ingest | Load raw data (`ingest_data` for file; `read_layer_dataframe` for Postgres) | File path or Postgres engine | `dataframe` with record IDs and `meta__` columns; `dataframe.attrs["dataset_resolution"]` populated |
| Confirm Dataset Identity | Read `dataframe.attrs` to confirm final dataset name post-ingest | `dataframe.attrs` | `RESOLVED_DATASET_NAME`; raises `ValueError` if absent or None |
| Create Artifact Folders | Build Bronze artifact directory tree; export config snapshot | `RESOLVED_DATASET_NAME`, `paths` | Directory tree on disk; `CONFIG_SNAPSHOT_PATH` written; `BRONZE_ARTIFACT_DIRS` populated |
| Update W&B Config | Overwrite W&B run config with confirmed dataset name | `wandb_run`, `RESOLVED_DATASET_NAME` | W&B run metadata updated |
| Log Dataset Name Changes | Log if provisional and resolved names differ | Both name vars | Logger warning if names changed |
| Build Truth Record Foundation | `initialize_layer_truth`; attach config snapshot section | `TRUTH_VERSION`, `DATASET_NAME`, `LAYER_NAME`, `PROCESS_RUN_ID` | `bronze_truth` dict; `parent_truth_hash=None` |
| Bronze Data Review | Shape, dtype, missingness, and descriptive stats display | `dataframe` | Terminal/cell output only; missingness computed per column |
| Finalize Lineage + Save Truth | Fingerprint source; update truth record; build column lists; stamp rows; save truth record; append truth index | `RAW_FILE_PATH`, `dataframe`, `bronze_truth` | `BRONZE_TRUTH_HASH`; `meta__truth_hash` stamped into every row; truth JSON on disk; truth index updated |
| Column Reorder | Move all `meta__` columns to front; return copy | `dataframe` | Reordered `dataframe` (copy, not view) |
| Save Bronze Dataset | `save_data` — write Parquet | `dataframe`, `paths.data_bronze_train`, file name | `.parquet` artifact on disk (before profiling so artifact exists even if profiling fails) |
| Profiling | `profile_dataframe` | `dataframe`, `logger`, `BRONZE_PROFILE_DIR` | Profile files in artifact dir; `metrics` dict |
| W&B Finalization | `finalize_wandb_stage`; `wandb_run.finish` | `wandb_run`, `dataframe`, `paths` | Dataset artifact registered in W&B; run closed |
| Lineage Validation | Verify required `meta__` columns; extract and confirm truth hash; cross-check truth record row/column counts | `dataframe`, saved truth JSON | Raises `ValueError` if any check fails |
| Final Structural Check | Print shape, dtypes, stats | `dataframe` | Terminal output only |
| Bronze SQL Write | `write_bronze_sensor_observations_sql` if `WRITE_TO_POSTGRES` | `engine`, schemas, `DATASET_ID`, `RUN_ID`, `dataframe` | Rows written to `sensor_observations` in Bronze layer schema |
| SQL Preview + Cleanup | `preview_sql_table`; `engine.dispose` | `engine` | Schema/table listing displayed; connection pool released |

---

## Section Details

### Context Load and Sanity Checks

`load_notebook_context(stage="bronze", dataset="pump", mode="train", profile="default")` is the single call that bootstraps the entire notebook runtime. It returns a `CTX` object from which the notebook immediately unpacks `paths`, `CONFIG`, `STAGE_CFG` (aliased as `BRONZE_CFG`), `logger`, `ledger`, and all config sub-dicts. `DEFAULT_FALLBACKS` is extracted from `BRONZE_CFG` in the Notebook Level Configuration cell.

Two sanity checks follow:
- The **general check** verifies 17 variables (`CTX`, `paths`, `CONFIG`, `STAGE_CFG`, `logger`, `ledger`, `PIPELINE`, `WANDB_CFG`, and others). Missing any raises `NameError` immediately.
- The **Bronze-specific check** verifies `BRONZE_CFG` and `DEFAULT_FALLBACKS`. This check exists to catch the previously-observed `NameError: Missing Bronze context variables: ['DEFAULT_FALLBACKS']` that occurred when the stage config block was incomplete.

Both checks use `name not in globals()` to guard against silent `None` bindings. After both pass, `ledger.add` records the context initialization as a ledger step.

---

### Dataset Name Resolution

The dataset name cannot be set as a hard constant because it may differ between file and Postgres ingest paths. Before ingestion, the notebook resolves a *provisional* name through a priority cascade: argument override → config value → Postgres handoff source table name → file-based heuristic (stem normalization + content fingerprint) → fallback string `"unknown_dataset"`.

After `ingest_data` runs, the *final* name is confirmed by reading `dataframe.attrs["dataset_resolution"]`, which `ingest_data` populates from file or Postgres metadata. If the attrs dict is absent or the resolved name is `None`, a `ValueError` halts execution. Only `RESOLVED_DATASET_NAME` enters the truth record, artifact paths, and W&B config. The provisional name is used only for the W&B run's pre-ingest configuration.

---

### Truth Record Initialization and Lineage Finalization

Bronze is the root of the capstone truth chain — `parent_truth_hash=None`. The truth record captures: dataset identity, process run ID, pipeline mode, config snapshot, source file fingerprint, and final column counts. Silver and Gold reference this record via `parent_truth_hash`.

Key operations:
- `initialize_layer_truth(...)` — creates the root truth dict
- `update_truth_section(bronze_truth, "config_snapshot", {...})` — attaches Bronze version and runtime facts
- `build_file_fingerprint(RAW_FILE_PATH / RAW_FILE_NAME)` — SHA-based fingerprint of the source file
- `update_truth_section(bronze_truth, "source_fingerprint", ...)` — attaches fingerprint
- `build_truth_record(bronze_truth, dataframe)` — finalizes the record with row/column counts
- `stamp_truth_columns(dataframe, bronze_truth)` — writes `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` into every row in-place
- `save_truth_record(bronze_truth, bronze_truth_path)` — persists truth record JSON to `BRONZE_LINEAGE_DIR`
- `append_truth_index(bronze_truth, TRUTH_INDEX_PATH)` — adds this record to the project-wide truth index

`stamp_truth_columns` writes into the DataFrame in-place before `reorder_bronze_columns` is called. `reorder_bronze_columns` returns a **copy** to prevent mutation through a pandas view.

---

### Save, Profile, and W&B Finalization

The Bronze Parquet artifact is saved **before** profiling so it exists on disk even if `profile_dataframe` raises an exception. This ordering is deliberate — noted in a source comment.

- `save_data(dataframe, paths.data_bronze_train, BRONZE_TRAIN_DATA_FILE_NAME)` — Parquet write via project wrapper (not detected by pattern-match; confirmed from source). The inventory's `artifact_write_clues` shows only `to_json`; `save_data` wraps `to_parquet` internally.
- `profile_dataframe(dataframe, logger, artifacts_dir=BRONZE_PROFILE_DIR)` — generates profiling outputs; returns `(metrics, saved)` but the side effect (files on disk) is the primary output.
- `finalize_wandb_stage(wandb_run, stage="bronze", dataframe=dataframe, ...)` — registers the dataset artifact, attaches the log and notebook file.
- `wandb_run.finish()` — closes the W&B run.

---

### Lineage Validation

After W&B is closed and before the SQL write, a validation block confirms the Bronze truth chain is intact:
- Checks `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` are present in `dataframe.columns`
- Calls `extract_truth_hash(dataframe)` and raises `ValueError` if result is `None`
- Reads the saved truth JSON from disk; cross-checks `row_count` and `column_count` against the live dataframe
- Confirms `parent_truth_hash` in the loaded record is `None` (as expected for the chain root)

This block uses locally-defined `require_dict` and `require_int_value` helpers to produce clear error messages when truth metadata is malformed.

---

### Bronze SQL Write

`write_bronze_sensor_observations_sql` writes the Bronze dataframe to the `sensor_observations` table in the Bronze layer schema. The `WRITE_TO_POSTGRES` bool gate allows the notebook to run fully in offline or read-only environments without touching the database. When disabled, a print statement confirms the skip; all Parquet and truth artifacts are unaffected.

After the write, `preview_sql_table` displays a 5-row sample for visual confirmation. `engine.dispose()` releases the connection pool before the kernel exits.

---

## Key Function Calls and In-Place Usage

| Function | Section | Inputs Provided | Return / Side Effect |
|---|---|---|---|
| `load_notebook_context(...)` | Context Load | `stage="bronze"`, `dataset="pump"`, `mode`, `profile` | `CTX`; unpacked into `paths`, `BRONZE_CFG`, `logger`, `ledger` |
| `get_engine_from_env()` | SQL Runtime Context | None (reads env vars) | `engine` — SQLAlchemy engine |
| `read_sql_dataframe(engine, ...)` | SQL Smoke Check | Inline `SELECT` on `information_schema.tables` | `sql_smoke_check_dataframe` |
| `ingest_data(...)` | Ingest | File path, dataset candidates, `run_id`, `asset_id`, `validate=True` | `dataframe` with record IDs; resolution metadata in `.attrs` |
| `read_layer_dataframe(...)` | Ingest (Postgres path) | `engine`, `schema="public"`, table name | `dataframe` |
| `build_artifact_dirs(...)` | Create Artifact Folders | `artifacts_root`, `stage="bronze"`, `dataset_name` | Directory tree; returns dict of named paths |
| `export_config_snapshot(...)` | Create Artifact Folders | `CONFIG`, `CONFIG_SNAPSHOT_PATH` | Config JSON written to `BRONZE_CONFIG_DIR` |
| `wandb.init(...)` | W&B Init | `WANDB_CFG` fields, provisional dataset name, source metadata | Active `wandb_run` |
| `initialize_layer_truth(...)` | Build Truth Record | `truth_version`, `dataset_name`, `process_run_id`, `parent_truth_hash=None` | `bronze_truth` dict (chain root) |
| `build_file_fingerprint(...)` | Finalize Lineage | Path to raw file | Fingerprint dict |
| `build_truth_record(...)` | Finalize Lineage | `bronze_truth`, `dataframe` | Finalized truth dict with row/column counts |
| `stamp_truth_columns(...)` | Finalize Lineage | `dataframe`, `bronze_truth` | `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` written in-place |
| `save_truth_record(...)` | Finalize Lineage | `bronze_truth`, `bronze_truth_path` | Truth JSON on disk in `BRONZE_LINEAGE_DIR` |
| `append_truth_index(...)` | Finalize Lineage | `bronze_truth`, `TRUTH_INDEX_PATH` | Project-wide truth index updated |
| `reorder_bronze_columns(...)` | Column Reorder | `dataframe` | Returns copy with `meta__` columns leading |
| `save_data(...)` | Save Bronze Dataset | `dataframe`, `paths.data_bronze_train`, file name | `.parquet` written; wrapper around `to_parquet` |
| `profile_dataframe(...)` | Create Profiling Outputs | `dataframe`, `logger`, `BRONZE_PROFILE_DIR` | Profile files in artifact dir; `metrics` dict |
| `finalize_wandb_stage(...)` | Finalize Run Tracking | `wandb_run`, `stage`, `dataframe`, `paths` | Dataset artifact in W&B |
| `extract_truth_hash(...)` | Lineage Validation | `dataframe` | Hash string or `None` |
| `write_bronze_sensor_observations_sql(...)` | Bronze SQL Write | `engine`, schemas, `DATASET_ID`, `RUN_ID`, `dataframe` | Rows written to `sensor_observations` |
| `preview_sql_table(...)` | SQL Preview | `schema`, `table="sensor_observations"`, `limit=5` | 5-row display |

---

## Outputs and Artifacts

| Output | Type | Location | Downstream Consumer |
|---|---|---|---|
| Bronze Parquet dataset | `.parquet` | `paths.data_bronze_train / BRONZE_TRAIN_DATA_FILE_NAME` | Silver_01_PreEDA |
| Bronze truth record | `.json` | `BRONZE_LINEAGE_DIR / <truth_hash>.json` | Silver truth chain (`parent_truth_hash`) |
| Config snapshot | `.json` | `BRONZE_CONFIG_DIR / CONFIG_SNAPSHOT_PATH` | Audit / reproducibility |
| Profiling artifacts | Files (profiler format) | `BRONZE_PROFILE_DIR` | QA reference |
| `sensor_observations` table | PostgreSQL rows | Bronze `LAYER_SCHEMA` | Synthetic pipeline `postgres_to_bronze` path; Silver SQL read path |
| W&B dataset artifact | W&B artifact | W&B project | Experiment traceability |
| Truth index entry | Appended JSON index | `TRUTH_INDEX_PATH` | Cross-run lineage lookup |
| Bronze log | Log file | `paths.logs / "bronze.log"` | Ops / audit |
| Ledger entries | JSON via `ledger` | Written at stage close | Audit trail |

---

## Data Quality / Validation Behavior

| Check | Where | Failure / Risk Prevented |
|---|---|---|
| General context sanity check (17 vars) | After `load_notebook_context` | Prevents silent `NameError` from incomplete context |
| Bronze-specific check (`BRONZE_CFG`, `DEFAULT_FALLBACKS`) | After general check | Guards against known `NameError: Missing Bronze context variables` |
| SQL smoke check | Before ingest | Catches DB unavailability before any processing runs |
| Dataset resolution attrs present post-ingest | After `ingest_data` | Prevents proceeding with unknown dataset identity |
| `RESOLVED_DATASET_NAME` not None | After attrs read | Prevents None embedded in artifact paths or truth records |
| W&B source mode guard | W&B Source Setup | Raises `ValueError` on unrecognized `BRONZE_SOURCE_MODE` before W&B init |
| Required `meta__` columns present (post-truth-stamp) | Lineage Validation | Confirms `stamp_truth_columns` succeeded |
| Truth hash extractable from dataframe | Lineage Validation | Confirms rows carry readable `meta__truth_hash` |
| Truth record row/column count cross-check | Lineage Validation | Confirms saved truth record matches live dataframe |
| `parent_truth_hash` is None in loaded record | Lineage Validation | Confirms chain-root invariant is preserved in the saved file |

---

## Downstream Handoff

`Silver_01_PreEDA` reads the Bronze Parquet file from `paths.data_bronze_train`. Before any transformation, it calls `extract_truth_hash(dataframe)` to capture `SILVER_PARENT_TRUTH_HASH`, linking the Silver truth record back to the Bronze root.

The `sensor_observations` PostgreSQL table is consumed by the Synthetic pipeline's `postgres_to_bronze` path when operating without Kafka.

The truth record JSON in `BRONZE_LINEAGE_DIR` and the truth index entry at `TRUTH_INDEX_PATH` are available for cross-stage audit and for any future truth record lookup by hash.

---

## Relationship to Other Notebooks

### Upstream Context

Bronze_01_Preprocessing has no upstream notebook dependency. It reads raw synthetic pump telemetry directly from the PostgreSQL operational layer via `read_layer_dataframe` from the `capstone` schema. The synthetic data pipeline (synthetic notebooks) populates this schema before Bronze_01 runs.

### Downstream Handoff

Bronze_01 provides the preprocessed Bronze Parquet (written to the `capstone.bronze` schema via `write_layer_dataframe`) to Silver_01_PreEDA. It registers a `bronze_preprocessing` truth record under `truths/` and logs a run record to `capstone.pipeline_runs`.

### Pipeline Position

Entry point of the Bronze → Silver → Gold medallion pipeline. First notebook to process raw synthetic pump telemetry data. All downstream Silver and Gold notebooks trace back to Bronze_01's preprocessed output as the origin of their telemetry data.

### Relationship Summary

- Reads from the PostgreSQL `capstone` schema populated by the synthetic data pipeline; no upstream notebook dependency
- Produces preprocessed Bronze Parquet consumed by Silver_01_PreEDA
- Establishes the `bronze_preprocessing` truth record as the first link in the project lineage chain
- Writes run records to `capstone.pipeline_runs` for operational audit
- No direct relationships with Silver_02x, Gold, or cascade notebooks

---

## Notes / Risks / Deferred Cleanup

- `BRONZE_SOURCE_MODE` must be `"file"` or `"postgres_handoff"`. An unknown value raises `ValueError` at W&B source setup, before ingest begins.
- `WRITE_TO_POSTGRES = False` skips the SQL write silently. All file artifacts are unaffected.
- Artifact folder creation is deferred until `RESOLVED_DATASET_NAME` is confirmed. A pre-ingest failure leaves no artifact directories created.
- `save_data()` is a project-level wrapper around `to_parquet`. It is not detected by the inventory script's pattern-match clues (which only see `to_json`). Artifact output has been confirmed by direct source inspection.
- `MissingIDFieldWarning` appears during `nbconvert --execute` because cells lack the `id` field required by nbformat 5.1+. Non-fatal today; normalize cell IDs before a future nbformat upgrade.
- `MODEL_EVALUATION` decision tag in the inventory is triggered by `profile_dataframe`. The profiler computes descriptive statistics that resemble evaluation metrics but this is data profiling, not model evaluation in the modeling sense.
