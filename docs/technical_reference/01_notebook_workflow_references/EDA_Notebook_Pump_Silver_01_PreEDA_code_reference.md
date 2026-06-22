# Notebook Code Reference: EDA_Notebook_Pump_Silver_01_PreEDA

**Source:** `notebooks/eda/EDA_Notebook_Pump_Silver_01_PreEDA.ipynb`
**Stage:** Silver — Pre-EDA
**Cells:** 213 (89 code, 124 markdown), 64 sections

---

## Notebook Purpose

This notebook is the Silver-layer entry point. It receives the Bronze Parquet artifact, resolves the parent truth record from the Bronze data, and produces a cleaned, feature-selected dataset ready for modeling in Gold.

The core responsibilities are:
- **Structural cleaning:** Remove junk import columns, duplicate column names, and canonical name conflicts
- **Identity canonicalization:** Resolve time, step, asset, and run columns into a consistent row-ordering structure
- **Anomaly flag construction:** Build a binary `anomaly_flag` column from either a label column or a status column, depending on what the dataset provides
- **Episode identification:** Assign contiguous episode IDs from the anomaly signal
- **Feature selection:** Classify, filter, and register the model-ready feature set
- **Missingness quarantine:** Drop features exceeding the missingness threshold; write quarantined column data to a separate artifact
- **Truth chain continuation:** Extract Bronze `meta__truth_hash` as `SILVER_PARENT_TRUTH_HASH`, initialize Silver truth record, stamp Silver truth hash into every row, and save the Silver truth record

Deliverables: cleaned Silver Parquet, feature registry JSON, quarantined sensors Parquet, Silver truth record, ledger JSON, and an optional Silver EDA SQL summary.

---

## Pipeline Role

| Attribute | Value |
|---|---|
| Stage | Silver |
| Position | Second in the active chain; directly downstream of Bronze_01 |
| Upstream input | Bronze Parquet artifact from `BRONZE_TRAIN_DATA_PATH` |
| Downstream output | Silver Parquet + feature registry consumed by Gold_01_PreProcessing |
| Truth chain | Reads Bronze `meta__truth_hash` as parent; writes Silver `meta__truth_hash` for Gold |

---

## Inputs

| Input | Source | Form | Used For |
|---|---|---|---|
| Bronze Parquet dataset | `BRONZE_TRAIN_DATA_PATH / BRONZE_TRAIN_DATA_FILE_NAME` | Parquet | Primary data load |
| Bronze PostgreSQL table | `sensor_observations` via `read_layer_dataframe` | SQL (alternative path) | SQL ingest mode if Parquet unavailable |
| Project config | `load_notebook_context(stage="silver", dataset="pump")` | YAML → `CTX` | All runtime constants, paths, stage config, fallbacks |
| Environment variables | OS environment | Strings | DB engine, `CAPSTONE_SCHEMA`, `DATASET_ID` / `RUN_ID` override |

---

## Configuration and Runtime Context

| Item | Source | Purpose |
|---|---|---|
| `SILVER_CFG` / `DEFAULTS_FALLBACKS_CFG` / `DEFAULT_FALLBACKS_CFG` | `CTX` | Silver stage config and fallback values; required by Silver-specific sanity check |
| `CLEANING_RECIPE_ID` / `CONTEXT_RECIPE_ID` | `SILVER_CFG` | Identifies the cleaning recipe version for this Silver run |
| `SILVER_PROCESS_RUN_ID` | `make_process_run_id(...)` | Unique identifier for this Silver execution; written into truth record |
| `QUARANTINE_MISSING_PCT` | `SILVER_CFG` / `DEFAULTS_FALLBACKS_CFG` | Threshold above which a feature column is dropped and quarantined |
| `CORRELATION_THRESHOLD` | `SILVER_CFG` | Feature correlation threshold for deselection (used downstream or in feature selection logic) |
| `MIN_ROWS_PER_STATE_FOR_GATE` | `SILVER_CFG` | Minimum rows per anomaly state required for state-stratified missingness gate to apply |
| `MISSING_STATE_DIFF_GATE_PCT` | `SILVER_CFG` | Missingness difference gate between normal and anomaly states |
| `JUNK_COLUMN_CANDIDATES` | `SILVER_CFG` | Regex/prefix list for columns added by CSV import tooling |
| `LABEL_COLUMN_CANDIDATES` / `STATUS_COLUMN_CANDIDATES` | `SILVER_CFG` | Column names searched for the anomaly label or operational status signal |
| `NORMAL_STATUS_VALUES` | `SILVER_CFG` | Values in the status column that map to `anomaly_flag = 0` |
| `CANONICAL_OUTPUT_COLUMNS` / `CANONICAL_NON_META_ORDER` | `SILVER_CFG` | Column ordering lists enforced during `reorder_silver_columns` |
| `WRITE_TO_POSTGRES` | Notebook cell (bool gate) | Controls whether `log_silver_eda_sql` executes; allows offline runs |
| `DATASET_NAME` / `RUN_ID` / `ASSET_ID` | Resolved from `CTX` and env vars | Row-level identity; written into Silver rows and truth record |
| `SILVER_VERSION` / `TRUTH_VERSION` | `VERSIONS_CFG` via `CTX` | Version stamps in truth record and artifact metadata |
| `PIPELINE_MODE` | Resolved from parent truth record post-ingest | Propagated from Bronze; written into Silver truth record and `meta__pipeline_mode` |
| `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_RUN_NAME` | `WANDB_CFG` via `CTX` | W&B run initialization |

---

## Logical Workflow Map

1. Load shared context via `load_notebook_context`; run general + Silver-specific sanity checks
2. Establish SQL engine; run SQL smoke check
3. Start logging; initialize ledger; record stage-start entries
4. Initialize W&B run with `wandb.init`
5. Load Bronze Parquet via `load_data`; record ledger step
6. Extract Bronze `meta__truth_hash` → `SILVER_PARENT_TRUTH_HASH`; load parent truth record; confirm `DATASET_NAME` and `PIPELINE_MODE` from parent truth; initialize Silver truth record (`initialize_layer_truth`) with `parent_truth_hash=SILVER_PARENT_TRUTH_HASH`
7. Create Silver artifact directories now that dataset name is confirmed; export config snapshot
8. Initial input review (shape, dtypes, describe)
9. Remove junk import columns (`remove_junk_import_columns`); remove duplicate column names (`deduplicate_columns`)
10. Validate dataset identity for Silver (`validate_dataset_name_for_silver`); update truth section
11. Add required Silver metadata columns (`meta__row_seq`, `meta__processed_at_utc`)
12. Resolve anomaly label source (`resolve_label_or_status_source`) — label column or status column path; update truth section
13. Protect canonical output column names from collision (`protect_canonical_output_names`)
14. Build canonical row identity and ordering (`build_canonical_identity_and_order_master`) — time, step, asset, run columns; update truth section
15. Build binary anomaly flag (`normalize_label_to_binary` or `build_anomaly_flag_from_status`); build episode IDs (`build_episode_ids_from_anomaly_flag`)
16. Classify and select candidate feature columns (`classify_column_type`, `identify_feature_set`, `identify_one_hot_encoding_columns`, `build_feature_set_identifier`); update truth section
17. Mid-workflow structural review; compute state-stratified missingness audit
18. Quarantine features with excessive missingness (`quarantine_features_by_missingness`); write dropped columns to Parquet; summarize global missingness; finalize feature lists
19. Create stable feature set identifier (MD5 of sorted feature list)
20. Reorder Silver columns into canonical layout (`reorder_silver_columns`)
21. Run final quick quality checks (`compute_quick_quality_checks`); build Silver feature registry
22. Stamp Silver truth columns (`stamp_truth_columns`); build and save truth record; append truth index; write Silver Parquet and feature registry JSON; write ledger JSON
23. Finalize W&B run (`finalize_wandb_stage`, `wandb_run.finish`)
24. Run final lineage and consistency check — load saved truth JSON from disk; cross-check row/column counts; verify parent hash
25. Write Silver EDA summary to PostgreSQL (`log_silver_eda_sql`) if `WRITE_TO_POSTGRES`

---

## Section Overview

| Section | Key Inputs | Key Outputs / Side Effects |
|---|---|---|
| Environment Setup (imports, `DropKeep` dataclass) | None | All imports; `DropKeep` dataclass; helper functions |
| Environment Setup, Paths, and Runtime Configuration | Config files, env vars | `CTX`, `SILVER_CFG`, `DEFAULT_FALLBACKS_CFG`, all layer/version/path constants, `logger`, `ledger`; `ledger.add`; `set_wandb_dir_from_config` |
| Context Sanity Check | Globals dict, `SILVER_CFG` | Raises `NameError` if any of the required general or Silver-specific variables are missing; `ledger.add` on pass |
| Defer Silver Artifact Folder Creation | None | Placeholder artifact path vars; no dirs created until dataset identity confirmed |
| SQL Runtime Context | `CTX`, env vars | `engine`; `DATASET_ID`, `RUN_ID`, `ASSET_ID` resolved from env → config → fallback cascade |
| SQL Smoke Check | `engine` | `sql_smoke_check_dataframe`; confirms DB connectivity |
| Logging Setup | `paths`, `logger`, `ledger` | Layer path log written; `ledger.add` for stage start |
| Initialize Experiment Tracking | `WANDB_CFG` | `wandb_run` active |
| Initialize the Silver Ledger | `ledger` | `ledger.add` step recorded (Ledger constructor called or re-initialized) |
| Load the Bronze Dataset into the Silver Workflow | `BRONZE_TRAIN_DATA_PATH`, `BRONZE_TRAIN_DATA_FILE_NAME` | `dataframe`; `ledger.add`; `wandb_run.log` with row/column count |
| Resolve the Parent Truth Record and Confirm Dataset Identity | `dataframe` (`meta__truth_hash` column), truth store | `SILVER_PARENT_TRUTH_HASH`, `BRONZE_DATASET_NAME`, `PIPELINE_MODE`, `parent_truth`; `silver_truth` initialized with `parent_truth_hash=SILVER_PARENT_TRUTH_HASH` |
| Create the Silver Pre-EDA Artifact Folders After Dataset Resolution | `DATASET_NAME`, `paths` | Directory tree on disk; `CONFIG_SNAPSHOT_PATH`; `SILVER_PREEDA_ARTIFACT_DIRS` populated; `export_config_snapshot` |
| Initial Silver Input Review | `dataframe` | Shape, dtypes, describe displayed; no artifact |
| Remove Junk Import Columns (definition) | `JUNK_COLUMN_CANDIDATES` regex | Helper function defined |
| Remove import-generated junk columns | `dataframe`, `JUNK_COLUMN_CANDIDATES` | `dataframe` with junk columns removed; `ledger.add` |
| Remove Duplicate Column Names (definition) | None | Helper function defined |
| Remove duplicate columns before canonicalization | `dataframe` | `dataframe` with duplicates removed; `ledger.add` |
| Validate the Dataset Name for Silver (definition + helpers) | None | `validate_dataset_name_for_silver` and normalization helpers defined |
| Validate dataset identity for the Silver layer | `dataframe`, `DATASET_NAME` from config, truth record name | `VALIDATED_DATASET_NAME`, `DATASET_METHOD`, `DATASET_SOURCE_COLUMN`; `silver_truth` updated; `ledger.add` |
| Add and Confirm Required Silver Metadata Columns | `dataframe` | `meta__row_seq` (RangeIndex), `meta__processed_at_utc` (UTC timestamp) added; `silver_truth` updated; `ledger.add` |
| Resolve the Label or Status Source (definition) | None | Resolution function defined |
| Review current column layout | `dataframe` | Display only |
| Resolve the label source used for anomaly flags | `dataframe`, `LABEL_COLUMN_CANDIDATES`, `STATUS_COLUMN_CANDIDATES` | `LABEL_SOURCE_COLUMN`, `LABEL_SOURCE_TYPE`, `LABEL_SOURCE_INFO`; `silver_truth` updated; `ledger.add` |
| Protect Canonical Output Column Names (definition) | None | `protect_canonical_output_names` function defined |
| Protect canonical metadata column names | `dataframe`, `CANONICAL_OUTPUT_COLUMNS` | `dataframe` with any shadow-name columns renamed; `rename_map` recorded; `ledger.add` |
| Build the Canonical Identity and Ordering Fields (definition) | None | `build_canonical_identity_and_order_master` and related helpers defined |
| Build canonical row identity and ordering fields | `dataframe`, time/step/asset/run candidates | Ordering columns resolved; `dataframe` sorted/reindexed; `canonical_info`; `silver_truth` updated; `ledger.add` |
| Build the Binary Anomaly Flag (definition) | None | `ANOMALY_FLAG_COLUMN` constant set |
| Define label-to-binary normalization | `LABEL_SOURCE_TYPE`, `LABEL_SOURCE_COLUMN` | `normalize_label_to_binary` (label path) and `build_anomaly_flag_from_status` (status path) defined |
| Define status-to-anomaly conversion | None | `build_anomaly_flag_from_status` inner logic defined |
| Build Episode IDs from the Anomaly Signal (definition) | None | `build_episode_ids_from_anomaly_flag` defined |
| Build anomaly episode identifiers | `dataframe`, `ANOMALY_FLAG_COLUMN` | `dataframe` with `anomaly_flag` + episode ID columns; `episode_counts`; `ledger.add` |
| Identify the Candidate Feature Set (definitions) | None | `classify_column_type`, prefix exclusion helpers, identifier-heuristic helpers defined |
| Select model-ready candidate features | `dataframe`, exclusion lists | `FEATURE_COLUMNS`, `FEATURE_GROUPS`, `FEATURE_INFO` |
| Identify Columns That May Need One-Hot Encoding Later | `dataframe` | `silver_one_hot_encoding_columns` list |
| Apply feature exclusion rules | `FEATURE_COLUMNS`, `FEATURE_GROUPS` | `FEATURE_SET_IDENTIFIER`; `silver_truth` updated; `ledger.add` |
| Mid-Workflow Structural Review | `dataframe` | Dataframe backup created; structural review displayed; state-stratified missingness audit computed |
| Review intermediate output | `dataframe`, `silver_truth` | State-level missingness percentages per feature computed; `MISSINGNESS_REPLAY` logic gate evaluated |
| Evaluate Missingness and Quarantine High-Missing Features (definition) | None | `compute_missingness_percentage`, `quarantine_features_by_missingness` defined |
| Quarantine features with excessive missingness | `dataframe`, `QUARANTINE_MISSING_PCT`, `DROPPED_SENSORS_DATA_PATH` | `dataframe` with quarantined columns removed; `dropped_dataframe.to_parquet(DROPPED_SENSORS_DATA_PATH)` written; `dropped_features`; `silver_truth` updated |
| Summarize global missingness after quarantine | `dataframe`, `FEATURE_COLUMNS` | `compute_global_missingness` result; `missingness_audit`; `silver_truth` updated; `ledger.add` |
| Define configuration mapping guards | `missingness_audit_map` | `cfg_require_mapping` helper; per-state missingness summary validated |
| Finalize the Feature Lists After Quarantine | `FEATURE_COLUMNS`, `dropped_features` | `FEATURE_COLUMNS`, `FEATURE_GROUPS`, `FEATURE_INFO` updated after drops |
| Create a Stable Feature Set Identifier | `FEATURE_COLUMNS` | `FEATURE_SET_IDENTIFIER` (MD5 of sorted feature column list) |
| Reorder the Silver Columns (definitions) | None | `collect_meta_columns`, `reorder_silver_columns` defined |
| Define final Silver column ordering | `dataframe`, `CANONICAL_OUTPUT_COLUMNS` | `dataframe` with canonical column order applied |
| Run Final Quick Quality Checks (definition) | None | `compute_quick_quality_checks` defined |
| Run final Silver quality checks | `dataframe` | `quality_info` dict (duplicate counts, anomaly rate, missingness, numeric feature count) |
| Build the Silver Feature Registry | `dataframe`, `FEATURE_GROUPS`, `FEATURE_COLUMNS` | `feature_registry` dict |
| Build the Silver Truth Record and Save the Final Outputs | `dataframe`, `silver_truth`, `SILVER_TRUTH_HASH` | `stamp_truth_columns` → rows stamped; `build_truth_record` → finalized; `save_truth_record` → JSON on disk; `append_truth_index`; `save_data` → Silver Parquet; `save_json` → feature registry JSON; `ledger.add` |
| Save the Ledger Artifact | `ledger`, `SILVER_LINEAGE_DIR` | `ledger.write_json` → ledger JSON on disk |
| Finalize Experiment Tracking | `wandb_run`, `dataframe`, `paths` | `finalize_wandb_stage`; `wandb_run.finish`; dataset artifact registered in W&B |
| Run a Final Lineage and Consistency Check (definitions + validation) | Saved truth JSON, `dataframe` | Raises `ValueError` if Silver `meta__` columns missing, hash unreadable, or row/col counts mismatch |
| Silver SQL Write Cell | `engine`, `CAPSTONE_SCHEMA`, `globals()` | `log_silver_eda_sql` writes Silver EDA summary to PostgreSQL; `display` summary; skipped if `WRITE_TO_POSTGRES = False` |

---

## Section Details

### Context Load and Sanity Checks

`load_notebook_context(stage="silver", dataset="pump")` is the single bootstrap call. It returns `CTX` from which the notebook immediately unpacks `paths`, `CONFIG`, `STAGE_CFG` (aliased as `SILVER_CFG`), `logger`, `ledger`, `DEFAULTS_FALLBACKS_CFG`, `DEFAULT_FALLBACKS_CFG`, and all config sub-dicts.

Two sanity checks follow:
- The **general check** verifies the standard set of required context variables (same 17 vars as Bronze).
- The **Silver-specific check** verifies `SILVER_CFG` and `DEFAULT_FALLBACKS_CFG` (and a separate `DEFAULTS_FALLBACKS_CFG` variant). A missing variable raises `NameError` immediately.

Both checks use `name not in globals()` to guard against silent `None` bindings. `ledger.add` records the sanity check pass.

`SILVER_PROCESS_RUN_ID` is constructed via `make_process_run_id(...)` in the configuration block and is written into the Silver truth record to give each Silver execution a unique identifier.

---

### Artifact Folder Deferral

Silver artifact directories cannot be created until `DATASET_NAME` is confirmed from the parent truth record (extracted post-load). The "Defer Silver artifact folder creation" section assigns placeholder path variables. The actual `build_artifact_dirs` call runs in "Create the Silver Pre-EDA Artifact Folders After Dataset Resolution," which runs after `initialize_layer_truth`. Directories created: `SILVER_ARTIFACTS_PATH`, `SILVER_CONFIG_DIR`, `SILVER_LINEAGE_DIR`, `SILVER_METADATA_DIR`, `SILVER_PREEDA_ARTIFACT_DIRS`, `SILVER_PROFILE_DIR`, `SILVER_QUALITY_DIR`, `SILVER_REGISTRY_DIR`, `SILVER_SUMMARY_DIR`.

---

### Load Bronze Data and Parent Truth Resolution

`load_data(bronze_data_path, ...)` is the Parquet read wrapper. It resolves the Bronze file through a fallback search over `BRONZE_TRAIN_DATA_PATH`, using `BRONZE_TRAIN_DATA_FILE_NAME` as the preferred filename. If the exact file is not found, it falls back to the most-recently-modified `.parquet` in the directory.

After `dataframe` is loaded:
1. `extract_truth_hash(dataframe)` reads the `meta__truth_hash` column → `SILVER_PARENT_TRUTH_HASH`.
2. `load_parent_truth_record_from_dataframe(dataframe, ...)` reads the parent truth JSON from disk using `SILVER_PARENT_TRUTH_HASH` as the lookup key.
3. `get_dataset_name_from_truth(parent_truth)` confirms `DATASET_NAME` from the parent record rather than config alone.
4. `get_pipeline_mode_from_truth(parent_truth)` propagates `PIPELINE_MODE` from Bronze into Silver.
5. `initialize_layer_truth(..., parent_truth_hash=SILVER_PARENT_TRUTH_HASH)` creates `silver_truth` — the Silver truth root.

If `SILVER_PARENT_TRUTH_HASH` is `None` or the parent truth record cannot be loaded, `ValueError` halts execution. Silver is never the chain root; it always requires a Bronze parent.

---

### Anomaly Flag Construction — Two Paths

`resolve_label_or_status_source` inspects the incoming dataframe for columns matching `LABEL_COLUMN_CANDIDATES` and `STATUS_COLUMN_CANDIDATES`. It returns `LABEL_SOURCE_TYPE` (`"label"` or `"status"`) and `LABEL_SOURCE_COLUMN`.

**Label path:** If a label column is found (e.g., a direct 0/1 or True/False column), `normalize_label_to_binary` maps it to a binary integer series. Numeric values are coerced; string values are mapped to `{True/anomaly/1 → 1, False/normal/0 → 0}`.

**Status path:** If only a status column is found (e.g., `"operating"`, `"fault"`, `"recovery"`), `build_anomaly_flag_from_status` compares each value against `NORMAL_STATUS_VALUES`. Rows matching normal values get `0`; all others get `1`.

Both paths write the result into `ANOMALY_FLAG_COLUMN` in the dataframe. The `LABEL_SOURCE_TYPE` and mapping details are recorded in `silver_truth` via `update_truth_section`. This two-path design makes Silver dataset-agnostic: labeled datasets and status-only datasets both produce a consistent `anomaly_flag` column.

---

### Canonical Identity and Ordering

`build_canonical_identity_and_order_master` resolves four canonical fields:
- **Time column:** Searched from `TIME_COLUMN_CANDIDATES`; tested with `pd.to_datetime`; `MIN_TIME_PARSE_SUCCESS_PERCENT` gate applied.
- **Step column:** Searched from `STEP_COLUMN_CANDIDATES`; tested with `pd.to_numeric`; `MIN_STEP_PARSE_SUCCESS_PERCENT` gate applied.
- **Asset column:** First non-null match from `TIE_BREAKER_CANDIDATES` used as the asset identifier.
- **Run column:** Similarly resolved from candidates; used in groupby sort when present.

If a time column is found, `dataframe` is sorted by `[asset_column, run_column, time_column]` (whichever are available). A `group_count` column tracks row position within each group. The resolution results are stored in `canonical_info` and written into `silver_truth`.

---

### Feature Selection Pipeline

Feature selection runs in three stages:

**1. Classify:** `classify_column_type` assigns each column to one of: `meta`, `identity`, `label`, `numeric_sensor`, `categorical`, or `exclude`. Columns that match `DEFAULT_EXCLUDE_PREFIXES` (e.g., `"meta__"`, `"raw__"`) are excluded immediately. The `looks_like_identifier_column` heuristic identifies high-cardinality string columns as identifiers.

**2. Select:** `identify_feature_set` applies the classification results plus explicit exclusion lists (`CANONICAL_EXCLUDE_COLUMNS`, `LABEL_EXCLUDE_COLUMNS`, resolved label/meta column names) to produce `FEATURE_COLUMNS` and `FEATURE_GROUPS`.

**3. One-hot candidates:** `identify_one_hot_encoding_columns` flags categorical columns in `FEATURE_COLUMNS` that will need encoding before modeling. These are recorded in `silver_one_hot_encoding_truths` and written into `silver_truth`.

`build_feature_set_identifier` constructs `FEATURE_SET_IDENTIFIER` — an MD5 hash of the sorted feature column list. It changes any time the selected features change, giving Gold a stable key for checking whether its expected feature set matches Silver's output.

---

### Missingness Quarantine

`quarantine_features_by_missingness` calls `compute_missingness_percentage` to compute per-feature missing percentage, then drops columns where the percentage exceeds `QUARANTINE_MISSING_PCT`. Quarantined column data (record IDs and dropped column values) is written to `DROPPED_SENSORS_DATA_PATH` as a Parquet file using `dropped_dataframe.to_parquet(...)` directly — this is distinct from `save_data()`.

Before quarantine, the "Review intermediate output" section computes **state-stratified missingness**: missingness is computed separately for normal, anomaly, and recovery rows (using the `anomaly_flag` column as the state signal). `MISSING_STATE_DIFF_GATE_PCT` sets the threshold at which a feature's missingness is flagged as state-dependent — indicating the missing data may be structurally tied to the fault condition rather than random. This is the `MISSINGNESS_REPLAY` behavior: documenting which features have state-correlated gaps so Gold can handle them deliberately.

After quarantine, `compute_global_missingness` re-evaluates the remaining columns, `FEATURE_COLUMNS` is updated to remove the dropped columns from `FEATURE_GROUPS`, and both `silver_truth` and `ledger` record the final missingness audit summary.

---

### Save, Truth Stamp, and Ledger

The finalization sequence (cells 196–200) runs in order:

1. `stamp_truth_columns(dataframe, silver_truth)` — writes `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` into every row in-place
2. `build_truth_record(silver_truth, dataframe)` — finalizes row/column counts in the truth dict
3. `save_truth_record(silver_truth, silver_truth_path)` — writes Silver truth JSON to `SILVER_LINEAGE_DIR`
4. `append_truth_index(silver_truth, TRUTH_INDEX_PATH)` — adds Silver entry to the project-wide truth index
5. `save_data(dataframe, file_path=SILVER_TRAIN_DATA_PATH, file_name=SILVER_TRAIN_DATA_FILE_NAME, ...)` — writes the Silver Parquet artifact
6. `save_json(feature_registry, file_path=SILVER_REGISTRY_DIR, file_name=FEATURE_REGISTRY_FILE_NAME, ...)` — writes the feature registry JSON
7. `ledger.add(kind="step", step="silver_finalize_export", ...)` — records final export metadata
8. `ledger.write_json(SILVER_LINEAGE_DIR / "silver__<dataset_name>__ledger.json")` — writes the ledger artifact

`save_data` is a project-level wrapper around `to_parquet`. The inventory's `artifact_write_clues` includes `to_parquet` (from the missingness quarantine path) and `save_json`. The Silver Parquet is written by `save_data`, confirmed directly from source cell 197.

---

### Silver SQL Write

`log_silver_eda_sql` is the Silver EDA summary write function. It is **not** a generic `write_layer_dataframe` call — it writes a structured summary of the Silver EDA results (feature counts, missingness, anomaly rate, feature set ID, dataset name) to a dedicated PostgreSQL table. The `WRITE_TO_POSTGRES = True` gate controls whether it runs. When `False`, a print statement confirms the skip; all file artifacts are unaffected.

`write_layer_dataframe` appears in the import list and in the `sql_touchpoints` clue, but is not called directly in the Silver SQL write cell — it is available for other paths (e.g., utility internals or commented-out alternatives).

---

## Key Function Calls and In-Place Usage

| Function | Section | Inputs Provided | Return / Side Effect |
|---|---|---|---|
| `load_notebook_context(...)` | Environment Setup | `stage="silver"`, `dataset="pump"`, `mode`, `profile` | `CTX`; unpacked into `paths`, `SILVER_CFG`, `logger`, `ledger` |
| `make_process_run_id(...)` | Environment Setup | prefix from `SILVER_CFG`, timestamp | `SILVER_PROCESS_RUN_ID` string |
| `get_engine_from_env()` | SQL Runtime Context | None (reads env vars) | `engine` — SQLAlchemy engine |
| `log_layer_paths(...)` | Logging Setup | `paths`, `logger` | Layer path log written |
| `wandb.init(...)` | Initialize Experiment Tracking | `WANDB_CFG`, dataset name | `wandb_run` |
| `load_data(...)` | Load Bronze Dataset | Bronze file path | `dataframe` (Parquet read via wrapper) |
| `extract_truth_hash(dataframe)` | Resolve Parent Truth Record | `dataframe` | `SILVER_PARENT_TRUTH_HASH` (str or None) |
| `load_parent_truth_record_from_dataframe(...)` | Resolve Parent Truth Record | `dataframe`, truth store path | `parent_truth` dict |
| `get_dataset_name_from_truth(...)` | Resolve Parent Truth Record | `parent_truth` | `BRONZE_DATASET_NAME` / `DATASET_NAME` |
| `get_pipeline_mode_from_truth(...)` | Resolve Parent Truth Record | `parent_truth` | `PARENT_PIPELINE_MODE` / `PIPELINE_MODE` |
| `initialize_layer_truth(...)` | Resolve Parent Truth Record | `truth_version`, `dataset_name`, `layer_name`, `process_run_id`, `parent_truth_hash=SILVER_PARENT_TRUTH_HASH` | `silver_truth` dict |
| `build_artifact_dirs(...)` | Create Silver Artifact Folders | `artifacts_root`, `stage="silver"`, `dataset_name` | Directory tree; `SILVER_PREEDA_ARTIFACT_DIRS` dict |
| `export_config_snapshot(...)` | Create Silver Artifact Folders | `CONFIG`, `CONFIG_SNAPSHOT_PATH` | Config JSON in `SILVER_CONFIG_DIR` |
| `remove_junk_import_columns(...)` | Remove junk import columns | `dataframe`, `JUNK_COLUMN_CANDIDATES` | `dataframe` without junk columns; `junk_columns_found` list |
| `deduplicate_columns(...)` | Remove duplicate column names | `dataframe` | `dataframe` without duplicate column names; `duplicates_columns_found` |
| `validate_dataset_name_for_silver(...)` | Validate dataset identity | `dataframe`, config name, truth name | `VALIDATED_DATASET_NAME`, `DATASET_METHOD` |
| `resolve_label_or_status_source(...)` | Resolve label source | `dataframe`, candidate lists, `NORMAL_STATUS_VALUES` | `LABEL_SOURCE_COLUMN`, `LABEL_SOURCE_TYPE`, `LABEL_SOURCE_INFO` |
| `protect_canonical_output_names(...)` | Protect canonical column names | `dataframe`, `CANONICAL_OUTPUT_COLUMNS` | Renamed `dataframe`; `rename_map` |
| `build_canonical_identity_and_order_master(...)` | Build canonical identity | `dataframe`, candidate lists, parse success thresholds | Sorted/reindexed `dataframe`; `canonical_info` |
| `normalize_label_to_binary(...)` | Build binary anomaly flag | `raw_series` (label column) | `anomaly_series` (0/1 int series) |
| `build_anomaly_flag_from_status(...)` | Build binary anomaly flag | `raw_series` (status column), `NORMAL_STATUS_VALUES` | `anomaly_flag` series (0/1) |
| `build_episode_ids_from_anomaly_flag(...)` | Build anomaly episode IDs | `dataframe`, `ANOMALY_FLAG_COLUMN`, ordering columns | `dataframe` with episode ID column |
| `classify_column_type(...)` | Classify feature candidates | Column name, series | Type string (`"numeric_sensor"`, `"categorical"`, etc.) |
| `identify_feature_set(...)` | Select candidate features | `dataframe`, exclusion lists | `FEATURE_COLUMNS`, `FEATURE_GROUPS`, `FEATURE_INFO` |
| `identify_one_hot_encoding_columns(...)` | Flag categorical features | `dataframe`, `FEATURE_COLUMNS` | `silver_one_hot_encoding_columns` list |
| `build_feature_set_identifier(...)` | Create feature set fingerprint | `FEATURE_COLUMNS` | `FEATURE_SET_IDENTIFIER` (MD5 hex string) |
| `compute_missingness_percentage(...)` | Evaluate per-feature missingness | `dataframe`, `FEATURE_COLUMNS` | `missing_percentage` series |
| `quarantine_features_by_missingness(...)` | Drop high-missing features | `dataframe`, threshold, `DROPPED_SENSORS_DATA_PATH`, `save_dropped_dataframe=True` | Cleaned `dataframe`; `dropped_dataframe.to_parquet(...)` written |
| `compute_global_missingness(...)` | Summarize remaining missingness | `dataframe`, `FEATURE_COLUMNS` | `missingness_audit` dict |
| `reorder_silver_columns(...)` | Canonical column ordering | `dataframe`, `CANONICAL_OUTPUT_COLUMNS` | Reordered `dataframe` |
| `compute_quick_quality_checks(...)` | Final structural QA | `dataframe` | `quality_info` dict |
| `stamp_truth_columns(...)` | Write Silver truth hash into rows | `dataframe`, `silver_truth` | `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` in every row (in-place) |
| `build_truth_record(...)` | Finalize Silver truth dict | `silver_truth`, `dataframe` | `silver_truth_record` with row/col counts |
| `save_truth_record(...)` | Persist Silver truth JSON | `silver_truth_record`, `silver_truth_path` | JSON file in `SILVER_LINEAGE_DIR` |
| `append_truth_index(...)` | Add to project truth index | `silver_truth_record`, `TRUTH_INDEX_PATH` | Project-wide truth index updated |
| `save_data(...)` | Write Silver Parquet | `dataframe`, `SILVER_TRAIN_DATA_PATH`, `SILVER_TRAIN_DATA_FILE_NAME` | `.parquet` on disk (wrapper around `to_parquet`) |
| `save_json(...)` | Write feature registry | `feature_registry`, `SILVER_REGISTRY_DIR`, `FEATURE_REGISTRY_FILE_NAME` | `.json` on disk (wrapper around `to_json`) |
| `ledger.write_json(...)` | Write ledger artifact | `SILVER_LINEAGE_DIR / "silver__<name>__ledger.json"` | Ledger JSON on disk |
| `finalize_wandb_stage(...)` | Register dataset artifact | `wandb_run`, `stage`, `dataframe`, `paths` | Silver dataset artifact in W&B |
| `log_silver_eda_sql(...)` | Write EDA summary to PostgreSQL | `engine`, `CAPSTONE_SCHEMA`, `DATASET_ID`, `RUN_ID`, `globals()` | Silver EDA row(s) in PostgreSQL audit table |

---

## Outputs and Artifacts

| Output | Type | Location | Downstream Consumer |
|---|---|---|---|
| Silver Parquet dataset | `.parquet` | `SILVER_TRAIN_DATA_PATH / SILVER_TRAIN_DATA_FILE_NAME` | Gold_01_PreProcessing |
| Feature registry JSON | `.json` | `SILVER_REGISTRY_DIR / FEATURE_REGISTRY_FILE_NAME` | Gold reads `FEATURE_COLUMNS` and `FEATURE_GROUPS` from this file |
| Quarantined sensors Parquet | `.parquet` | `DROPPED_SENSORS_DATA_PATH` | Reference / QA only; not consumed by Gold |
| Silver truth record | `.json` | `SILVER_LINEAGE_DIR / <silver_truth_hash>.json` | Gold truth chain (`parent_truth_hash` reference) |
| Silver ledger JSON | `.json` | `SILVER_LINEAGE_DIR / silver__<dataset_name>__ledger.json` | Audit trail |
| Truth index entry | Appended JSON index | `TRUTH_INDEX_PATH` | Cross-run lineage lookup |
| Config snapshot | `.json` | `SILVER_CONFIG_DIR / CONFIG_SNAPSHOT_PATH` | Reproducibility / audit |
| Silver EDA SQL summary | PostgreSQL rows | `CAPSTONE_SCHEMA` audit table (via `log_silver_eda_sql`) | Operational monitoring; only written when `WRITE_TO_POSTGRES = True` |
| W&B dataset artifact | W&B artifact | W&B project | Experiment traceability |
| Silver log | Log file | `paths.logs` | Ops / audit |

---

## Data Quality / Validation Behavior

| Check | Where | Failure / Risk Prevented |
|---|---|---|
| General context sanity check | After `load_notebook_context` | Prevents silent `NameError` from incomplete context |
| Silver-specific check (`SILVER_CFG`, `DEFAULT_FALLBACKS_CFG`) | After general check | Prevents `NameError: Missing Silver context variables` |
| SQL smoke check | Before data load | Catches DB unavailability before processing begins |
| `SILVER_PARENT_TRUTH_HASH` not None | After `extract_truth_hash` | Prevents Silver from running without a Bronze parent; catches Bronze Parquet that never had truth stamped |
| Parent truth record loadable | After `load_parent_truth_record_from_dataframe` | Confirms Bronze truth record exists on disk |
| `DATASET_NAME` resolved from parent truth | After `get_dataset_name_from_truth` | Prevents Silver from inheriting a stale or wrong dataset name from config |
| `validate_dataset_name_for_silver` | Dataset identity section | Raises `ValueError` if Silver's resolved name conflicts with the parent truth record name |
| Anomaly label source resolved | After `resolve_label_or_status_source` | Prevents proceeding with no anomaly signal |
| State-stratified missingness gate | Mid-workflow review | Documents state-correlated missingness before quarantine; `MISSINGNESS_REPLAY` flag |
| Quarantine threshold | `quarantine_features_by_missingness` | Removes features that would degrade modeling with excessive NaNs |
| `meta__` columns present post-stamp | Final lineage check | Confirms `stamp_truth_columns` wrote required columns |
| Truth hash extractable from dataframe | Final lineage check | Confirms Silver rows carry a readable `meta__truth_hash` |
| Saved truth record row/column cross-check | Final lineage check | Confirms JSON on disk matches live dataframe |
| `parent_truth_hash` in loaded record matches `SILVER_PARENT_TRUTH_HASH` | Final lineage check | Confirms continuity of the Bronze → Silver truth chain |

---

## Downstream Handoff

`Gold_01_PreProcessing` reads:
- The **Silver Parquet** from `SILVER_TRAIN_DATA_PATH` for the feature matrix and truth metadata
- The **feature registry JSON** from `SILVER_REGISTRY_DIR` to recover `FEATURE_COLUMNS`, `FEATURE_GROUPS`, and `FEATURE_SET_IDENTIFIER`

Before any Gold transformation, Gold calls `extract_truth_hash(dataframe)` on the Silver Parquet to recover `GOLD_PARENT_TRUTH_HASH`, pointing back to the Silver truth record. The Silver truth record JSON in `SILVER_LINEAGE_DIR` is the authoritative source for the Silver-to-Gold truth link.

---

## Relationship to Other Notebooks

### Upstream Context

Silver_01_PreEDA reads the Bronze layer Parquet produced by Bronze_01_Preprocessing via `read_layer_dataframe` from the `capstone.bronze` schema. The `bronze_preprocessing` truth record provides lineage context. W&B is not active in Silver notebooks; no W&B dependency exists here.

### Downstream Handoff

Silver_01 provides pre-EDA profile Parquets, null analysis summaries, and distribution statistics to Silver_02a_EDA_Building_Subsets_v3. It writes Silver layer frames to `capstone.silver` via `log_silver_eda_sql` and registers a truth record under `truths/`.

### Pipeline Position

First Silver-layer notebook. Converts Bronze-layer preprocessed telemetry into pre-EDA analytical profiles that seed Silver_02a's subset construction and imputation decisions. Sits between Bronze_01 and Silver_02a in the linear pipeline.

### Relationship Summary

- Reads Bronze layer Parquet from `capstone.bronze` via `read_layer_dataframe`
- Produces pre-EDA profiles and distribution summaries consumed by Silver_02a
- Writes to `capstone.silver` via `log_silver_eda_sql`; W&B is not active
- No Gold, cascade, or W&B relationships
- Direct downstream consumer: Silver_02a only

---

## Notes / Risks / Deferred Cleanup

- If neither `LABEL_COLUMN_CANDIDATES` nor `STATUS_COLUMN_CANDIDATES` produces a match, `resolve_label_or_status_source` returns no source and downstream anomaly flag construction will fail. The notebook raises `ValueError` in this case before the flag build.
- `WRITE_TO_POSTGRES = True` is the default. Setting it to `False` skips `log_silver_eda_sql` but does not affect any file artifact. The Silver Parquet, feature registry, truth record, and ledger are all written regardless.
- `save_data` is a project wrapper around `to_parquet`. The inventory `artifact_write_clues` shows `to_parquet` (from the quarantine path), `save_json`, and `to_json`. The Silver Parquet write is by `save_data`, confirmed from source cell 197. These are two distinct Parquet writes: quarantine via `dropped_dataframe.to_parquet(...)`, and main Silver dataset via `save_data(...)`.
- `write_layer_dataframe` is imported but is not the Silver SQL write function — `log_silver_eda_sql` is used instead. `write_layer_dataframe` is available in scope for ad-hoc use or future SQL path alternatives.
- `MissingIDFieldWarning` is emitted during `nbconvert --execute` because cells lack the `id` field required by nbformat 5.1+. Non-fatal; normalize cell IDs before a future nbformat upgrade.
- The `MODEL_TRAINING` and `MODEL_EVALUATION` decision tags in the inventory stem from `sklearn` imports in the imports cell — these are imported at the top of the notebook for use if inline validation runs are needed, but no model training or evaluation occurs in Silver_01's primary workflow. The tags reflect import presence, not active use.
