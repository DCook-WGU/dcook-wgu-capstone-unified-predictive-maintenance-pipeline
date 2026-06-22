# Silver 01 Deep Technical Reference

## Purpose of This Deep Reference

This document covers technical decisions in Silver 01 (Pre-EDA) that require deeper explanation than the workflow reference provides. The workflow reference describes what the notebook does step by step. This document explains why the Bronze input is loaded and verified before any transformation, why artifact directory creation is deferred, why `DATASET_NAME` is verified rather than assigned, why the anomaly flag is built through two distinct paths, why state-stratified missingness is computed before quarantine, why the feature set identifier is an MD5 hash of sorted column names, why the final lineage consistency check re-reads the saved truth JSON, and why the SQL write is scoped to an EDA summary rather than the full Silver dataframe.

## Technical Scope

- Bronze Parquet loading strategy and fallback behavior
- Parent truth extraction and parent-truth-first initialization sequence
- Artifact directory deferral until `DATASET_NAME` is confirmed
- `DATASET_NAME` verification (not assignment) via parent truth and `meta__dataset` column
- `PIPELINE_MODE` propagation from Bronze parent truth
- Two-level sanity check design (general + Silver-specific)
- Defensive copy before column addition
- Junk import column removal — prefix candidates plus regex
- Canonical output column name protection before canonicalization
- Two-path anomaly flag construction (label path vs status path)
- Episode ID construction per asset/run group from anomaly signal
- Feature selection defaults and identifier-column heuristic exclusion
- State-stratified missingness audit before quarantine
- Missingness quarantine with dropped sensor Parquet preservation
- MD5-based feature set identifier (sorted columns, order-invariant)
- Canonical column reordering layout
- Final quality checks and feature registry construction
- Truth record finalization and stamp-before-save sequencing
- Final lineage consistency check: re-read saved truth JSON from disk
- SQL write scope: EDA summary via `log_silver_eda_sql`, not full dataframe
- W&B run scope across full transformation window
- Ledger as structured audit trail throughout

## Source Grounding

Sources used:

- `notebooks/eda/EDA_Notebook_Pump_Silver_01_PreEDA.ipynb`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Silver_01_PreEDA_code_reference.md`
- `notebook_inventory.json`
- `artifact_io_manifest.json`
- `sql_touchpoints.json`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Bronze_01_Preprocessing_deep_technical_reference.md`
- `technical_reference/00_project_manual/notebook_relationship_map.md`
- `technical_reference/04_deep_utility_function_references/utils__core__truths_deep_function_reference.md`
- `technical_reference/04_deep_utility_function_references/utils__core__artifacts_deep_function_reference.md`
- `technical_reference/04_deep_utility_function_references/utils__database__layer_postgres_deep_function_reference.md`

The active Silver 01 notebook source is the source of truth for all function behavior, variable names, and design decisions documented here.

## Stage Role in the Medallion Pipeline

Silver 01 is the first Silver-layer notebook. It receives the Bronze Parquet output and produces a cleaned, schema-canonicalized, feature-selected Silver Parquet that Gold 01 consumes. Its position in the chain is: Bronze 01 → **Silver 01** → Gold 01.

Silver 01 performs every structural transformation required before modeling: it verifies dataset identity from the Bronze parent truth, removes import-generated noise, builds the anomaly flag from whatever label or status signal the dataset provides, constructs episode identifiers, selects the model-ready feature set, quarantines high-missingness sensors, stamps a Silver truth hash into every row, and writes all outputs to disk. It is not a partial pre-processing step — it is the full Silver transformation layer.

Silver 01 does not train models. It does not apply thresholds. It does not join additional datasets. It does not modify Bronze data in place.

## Input Contract and Lineage

### Bronze Parquet as Primary Input

The primary input is the Bronze Parquet artifact at `BRONZE_TRAIN_DATA_PATH / BRONZE_TRAIN_DATA_FILE_NAME`. The load sequence prefers the exact filename:

```
if preferred_bronze.exists():
    bronze_data_path = preferred_bronze
else:
    parquet_files = sorted(BRONZE_TRAIN_DATA_PATH.glob("*.parquet"))
    ...
    bronze_data_path = parquet_files[0]
```

The fallback to the alphabetically-first Parquet in the directory provides resilience for minor filename changes between pipeline stages without silently reading a wrong file. If no Parquet files are found, `FileNotFoundError` is raised. If multiple Parquet files are found under the fallback path, a `logger.warning` is emitted and the first is used.

### SQL Alternative Read Path

The artifact manifest clue includes `read_layer_dataframe` and `read_sql` as SQL read clues. The workflow reference describes a SQL Bronze read path (`sensor_observations` table via `read_layer_dataframe`) as an alternative when the Parquet artifact is unavailable. This alternative path is present in the import list but the source shows the Parquet path as the primary — the SQL read is a fallback option, not the standard run path.

### Parent Truth as Lineage Anchor

Immediately after `load_data` returns, before any column is added or removed, `extract_truth_hash(dataframe)` reads the `meta__truth_hash` column from the Bronze-loaded frame. This becomes `SILVER_PARENT_TRUTH_HASH`. The design intent is documented in Cell 39's comment: "capture parent hash before any transformation so Silver truth links back to the exact Bronze artifact."

`load_parent_truth_record_from_dataframe` then loads the Bronze truth JSON from disk using `SILVER_PARENT_TRUTH_HASH` as the lookup key. Silver inherits two critical values from the parent truth record:

- `DATASET_NAME` — confirmed from `get_dataset_name_from_truth(parent_truth)` rather than from config alone
- `PIPELINE_MODE` — propagated via `get_pipeline_mode_from_truth(parent_truth)` so the synthetic vs real distinction carries forward without manual config

If `SILVER_PARENT_TRUTH_HASH` is `None` or the parent truth record cannot be loaded, execution halts with `ValueError`. Silver is never the chain root and must always have a Bronze parent.

### Project Config via `load_notebook_context`

`load_notebook_context(stage="silver", dataset="pump", mode="train", profile="default")` is the single bootstrap call. It returns `CTX` from which the notebook unpacks:

- `SILVER_CFG` — Silver-specific stage configuration (thresholds, column candidates, recipe ID)
- `DEFAULT_FALLBACKS_CFG` / `DEFAULTS_FALLBACKS_CFG` — fallback value configuration
- `logger`, `ledger`, `CONFIG`, `paths`, `RESOLVED_PATHS`, `FILENAMES`, `VERSIONS_CFG`, `RUNTIME_CFG`, `DATASET_CFG`, `WANDB_CFG`, `EXECUTION_CFG`, `PIPELINE`
- `CONTEXT_RECIPE_ID` — recipe identifier carried into the truth record

The two-alias pattern (`DEFAULTS_FALLBACKS_CFG = CTX.default_fallbacks` and `DEFAULT_FALLBACKS_CFG = CTX.default_fallbacks`) ensures both naming variants resolve to the same object, avoiding downstream `NameError` regardless of which alias is referenced.

### Environment Variables for SQL Identity

`DATASET_ID`, `RUN_ID`, and `ASSET_ID` are resolved via a `first_non_empty_string()` cascade that checks environment variables, `globals()` entries, config values, and defaults in order. This design accommodates both Docker environment injection and local notebook execution without requiring code changes.

### Lineage Significance at Silver

Silver is the first layer where model-ready data is produced. Any error in lineage at this stage propagates into Gold training, validation, and truth comparison. Silver therefore continues the truth chain established at Bronze rather than starting a new one: `SILVER_PARENT_TRUTH_HASH` links the Silver output unambiguously to the exact Bronze artifact it was derived from.

## Silver Data Preparation Methodology

### Two-Level Sanity Check

Two sanity checks run immediately after context load:

- **General check**: Validates 16 required context variables (`CTX`, `paths`, `CONFIG`, `CONFIG_MAP`, `STAGE_CFG`, `RESOLVED_PATHS`, `FILENAMES`, `VERSIONS_CFG`, `RUNTIME_CFG`, `DATASET_CFG`, `WANDB_CFG`, `EXECUTION_CFG`, `PIPELINE`, `logger`, `ledger`, `LOG_PATH`). Uses `name not in globals()` rather than checking for `None`, which means a variable silently bound to `None` does not pass this check.
- **Silver-specific check**: Validates `SILVER_CFG` alone. This is a separate check rather than extending the general list because the Silver-specific config depends on stage context resolution that may fail independently of general context resolution.

Both checks raise `NameError` on failure and record `ledger.add(kind="check", ...)` on pass.

### Artifact Directory Deferral

Silver artifact directory paths are initialized to `None` in Cell 16. The actual `build_artifact_dirs` call runs in Cell 41, after `DATASET_NAME` is confirmed from the parent truth record. The deferral prevents creating directories under a wrong or placeholder dataset name if the parent truth lookup fails mid-run. All downstream path variables (`SILVER_ARTIFACTS_PATH`, `SILVER_REGISTRY_DIR`, `SILVER_LINEAGE_DIR`, etc.) remain `None` until confirmed.

### Junk Import Column Removal

`remove_junk_import_columns` applies two detection passes:

1. **Prefix candidates**: Columns whose normalized lowercase names start with any entry in `JUNK_COLUMN_CANDIDATES` are identified.
2. **Regex match**: The `UNNAMED_COLUMN_REGEX` pattern catches columns added by CSV import tools (e.g., `Unnamed: 0`, `Unnamed: 1`).

Both passes apply to all columns; a column matching either criterion is dropped. This dual approach catches both project-configured junk columns and generic unnamed index columns that CSV export tools add regardless of configuration.

### Deduplication of Column Names

`deduplicate_columns(dataframe, keep="first")` removes duplicate column names using the `"first"` keep policy. A `keep="first"` policy is consistent with preserving the leftmost occurrence, which is conventionally the original source column rather than any repeated or appended copy.

### `DATASET_NAME` Verification (Not Assignment)

`validate_dataset_name_for_silver` performs verification, not assignment. Its docstring states: "Silver does not resolve or assign dataset identity. Silver verifies that Bronze-stamped dataset identity exists and is consistent."

The function reads `meta__dataset` column values, normalizes them (lowercase, replace spaces and hyphens with underscores, strip non-alphanumeric), and requires exactly one unique value (`fail_on_multiple_in_bronze=True`). If multiple values are found in `meta__dataset`, execution halts. This prevents a dataset containing mixed-source rows from passing through Silver as a single entity without detection.

### Defensive Copy Before Metadata Column Addition

Cell 63 contains `dataframe = dataframe.copy()` before adding any Silver metadata columns. This ensures the Bronze-loaded frame is not mutated in place. The pattern matters on re-run: without the copy, re-executing Cell 63 after a downstream failure would attempt to add columns to a frame that may already have them from the prior run's partially-modified state.

`SILVER_PROCESSED_AT_UTC` is recorded in the truth record rather than row-stamped into the dataframe. The Cell 63 comment confirms: "Silver runtime context goes to truth, not dataframe."

### Required Metadata Column Backfill

After the defensive copy, `META_REQUIRED_COLUMNS` is iterated. Any missing column is added with a column-specific default:
- `meta__dataset` → `pd.NA`
- `meta__split` → `"unsplit"`
- `meta__run_id` → `RUN_ID_DEFAULT_FALLBACK`
- `meta__asset_id` → `ASSET_ID_DEFAULT_FALLBACK`
- `meta__source_file` → `""`
- `meta__source_row_id` → `pd.RangeIndex(0, len(dataframe))`
- Other required columns → `pd.NA`

This backfill makes Silver tolerant of Bronze variants that may omit a specific metadata column without crashing downstream truth stamping.

### Canonical Output Column Name Protection

Before building canonical columns (`event_time`, `event_step`, `time_index`), `protect_canonical_output_names` inspects whether any of the target canonical names already exist in the source dataframe. If found, they are renamed to `raw__<name>` (with a numeric suffix for uniqueness). The ledger records the rename map.

This step prevents the canonical identity builder from overwriting source data that happens to share a name with a canonical output. The `raw__` prefix convention signals that the original source column is preserved for inspection even after canonicalization.

## Silver Transformation and Feature Logic

### Label Source Resolution (Policy: Label Preferred)

`resolve_label_or_status_source` scans `LABEL_COLUMN_CANDIDATES` and `STATUS_COLUMN_CANDIDATES` lists. Its policy: label column preferred over status column. The function comment states this preference is implemented by checking label candidates first and returning on the first match.

The resolution records column coverage statistics (`top_values`, `unique_count`, `non_null_count`, `coverage_pct`) and writes them to `silver_truth` via `update_truth_section`. This makes the label resolution decision auditable without re-running the notebook.

### Binary Anomaly Flag — Two Paths

Depending on `LABEL_SOURCE_TYPE`, the anomaly flag is built through one of three paths:

**Label path** (`LABEL_SOURCE_TYPE == "label"`): `normalize_label_to_binary` maps the label column to a binary integer series. Numeric values are coerced; string values are mapped to `{True/anomaly/1 → 1, False/normal/0 → 0}`.

**Status path** (`LABEL_SOURCE_TYPE == "status"`): `build_anomaly_flag_from_status` compares each value against `NORMAL_STATUS_VALUES`. Values matching normal → 0; all others → 1. Optional helper columns `status_normal_value`, `is_normal`, and `is_anomaly` are added for EDA convenience (they expose the intermediate decision without changing `anomaly_flag`).

**No source path**: When neither label nor status column is found, `anomaly_flag` defaults to all zeros (all normal). `anomaly_build_info["method"]` records `"no_label_or_status_available_default_all_normal"`. This prevents a crash on datasets that lack an anomaly signal at load time.

The two-path design makes Silver dataset-agnostic: labeled datasets and operational status datasets both produce a consistent `anomaly_flag` column for downstream use.

### Episode ID Construction

`build_episode_ids_from_anomaly_flag` assigns a `meta__episode_id` value to each row. An episode is a contiguous anomaly window (where `anomaly_flag == 1`); the episode ID increments each time the signal transitions from anomaly back to normal. Episodes are computed per group (`meta__asset_id` / `meta__run_id`), sorted by `time_index` or `event_step` within each group.

The episode ID is not a prediction; it is a segmentation key that allows downstream evaluation to group rows by fault window rather than treating each row independently. This distinction matters for episode-level recall measurement in Gold.

### Feature Column Classification

`classify_column_type` assigns each column one of: `meta`, `identity`, `label`, `numeric_sensor`, `categorical`, `boolean`, `text`, `datetime`, or `other`.

`identify_feature_set` then applies layered exclusion:

1. Columns matching `DEFAULT_EXCLUDE_PREFIXES` (e.g., `"meta__"`, `"raw__"`) — excluded unconditionally
2. Explicit `CANONICAL_EXCLUDE_COLUMNS` and `LABEL_EXCLUDE_COLUMNS` — excluded by config
3. The resolved `LABEL_SOURCE_COLUMN` — excluded to prevent label leakage
4. High-cardinality string columns identified by `looks_like_identifier_column` — excluded by heuristic

Default feature selection includes only `numeric` and `boolean` columns. Categorical, text, and datetime columns require explicit `include_*=True` flags. This conservative default prevents accidental injection of non-numeric columns into the feature set without deliberate choice.

`identify_one_hot_encoding_columns` flags any categorical columns remaining in `FEATURE_COLUMNS` as candidates for future encoding. Their identity is recorded in `silver_truth` but no encoding is applied in Silver 01.

### MD5 Feature Set Identifier

`build_feature_set_identifier` computes:

```python
normalized = "|".join(sorted([str(name) for name in feature_columns]))
digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
return f"feature_set__{digest}"
```

Columns are sorted before hashing so column reordering cannot create a false hash mismatch between runs. The `FEATURE_SET_IDENTIFIER` changes only when the set of selected feature columns changes. Gold 01 uses this identifier to verify that the feature set it receives matches its expected configuration.

### Canonical Column Reordering

`reorder_silver_columns` enforces the layout: meta columns (sorted alphabetically) → canonical columns (`CANONICAL_NON_META_ORDER`) → label columns → feature columns → remainder (any column not in any group, in original order). A `.copy()` is returned to prevent the caller from receiving a view — the subsequent truth stamp requires an owned frame.

This ordering is a structural convention, not a semantic transformation. It makes visual inspection of the Silver Parquet predictable and simplifies column-range operations in downstream notebooks.

## Silver Validation and Data Quality Checks

### Schema and Column Validation

The sanity checks validate the config objects and runtime context. `validate_dataset_name_for_silver` validates that `meta__dataset` is present, non-empty, and unique across all rows. These are hard failures — the notebook does not proceed past them on mismatch.

The metadata column backfill (Cell 63) is a tolerance mechanism rather than a validation failure: missing required columns are added with defaults rather than raising. The validation philosophy is: fatal on identity ambiguity, tolerant on missing operational columns with known defaults.

### State-Stratified Missingness Audit

Before the quarantine step, the mid-workflow structural review computes missingness separately per anomaly state (normal, anomaly, recovery rows). `MISSING_STATE_DIFF_GATE_PCT` is the threshold for flagging a feature as having state-correlated gaps. A sensor that is systematically absent during fault conditions has a different implication than one with random dropouts — it may indicate the sensor stops reporting during the fault event rather than having data quality issues. This `MISSINGNESS_REPLAY` audit documents which features have state-dependent gaps so Gold can handle them deliberately.

### Missingness Quarantine

`quarantine_features_by_missingness` drops from `FEATURE_COLUMNS` any numeric column exceeding `QUARANTINE_MISSING_PCT` missing rows. `drop_all_null=True` ensures fully null columns are also dropped regardless of threshold.

Quarantined column data is preserved: `dropped_dataframe.to_parquet(DROPPED_SENSORS_DATA_PATH)` writes the dropped sensor columns (with their row identifiers) to `DROPPED_SENSORS_DATA_PATH`. This direct `to_parquet()` call is distinct from the `save_data()` wrapper used for the primary Silver Parquet — the dropped sensors file is a secondary artifact, not the main output.

The missingness audit results (threshold, kept features, dropped features, per-column missing percentage, drop reasons) are recorded in `silver_truth` under `"sensor_drop_audit"`.

### Final Quick Quality Checks

`compute_quick_quality_checks` produces a summary of: total rows, duplicate row count, duplicate `meta__event_id` count (if present), per-feature numeric missingness (top 25 by percentage), anomaly rate percentage, and anomaly flag value counts. The top-25 cap on missingness entries prevents an excessively large ledger payload.

This check runs after quarantine and reordering, before truth stamping, so it reflects the final Silver output state that will be written to disk.

## Artifact and SQL Persistence

### Truth Stamp Before File Write

The finalization sequence in cells 194–200 runs in strict order:

1. `stamp_truth_columns(dataframe, silver_truth)` — writes `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` into every row in-place. This must run before the Parquet write so the file carries the truth hash.
2. `build_truth_record(silver_truth, dataframe)` — finalizes row/column counts in the truth dict.
3. `save_truth_record(silver_truth, silver_truth_path)` — writes Silver truth JSON to `SILVER_LINEAGE_DIR`.
4. `append_truth_index(silver_truth, TRUTH_INDEX_PATH)` — adds the Silver entry to the project-wide truth index.
5. `save_data(dataframe, ...)` — writes the Silver Parquet artifact.
6. `save_json(feature_registry, ...)` — writes the feature registry JSON.
7. `ledger.add(kind="step", step="silver_finalize_export", ...)` — records final export metadata.
8. `ledger.write_json(SILVER_LINEAGE_DIR / ...)` — writes the ledger artifact.

The stamp-before-save ordering is required: if the truth stamp were applied after the Parquet write, the saved file would lack the `meta__truth_hash` column and the lineage consistency check would fail on reload.

### Silver Output Artifacts

| Artifact | Path | Format | Function |
|---|---|---|---|
| Silver Parquet | `SILVER_TRAIN_DATA_PATH / SILVER_TRAIN_DATA_FILE_NAME` | Parquet | Primary Silver dataset; consumed by Gold 01 |
| Feature registry | `SILVER_REGISTRY_DIR / FEATURE_REGISTRY_FILE_NAME` | JSON | Feature set metadata; `FEATURE_SET_IDENTIFIER`, selected columns, exclusion decisions |
| Dropped sensors Parquet | `SILVER_PROFILE_DIR / DROPPED_SENSORS_FILE_NAME` | Parquet | Quarantined column values; row identifiers preserved |
| Silver truth record | `SILVER_LINEAGE_DIR / ...` | JSON | Truth hash, parent hash, config snapshot, all runtime fact sections |
| Config snapshot | `SILVER_CONFIG_DIR / ...` | YAML | Full resolved config at run time |
| Ledger | `SILVER_LINEAGE_DIR / silver__{dataset_name}__ledger.json` | JSON | Step-by-step audit trail of decisions |

The artifact manifest confirms `save_json` and `to_parquet` write clues for this notebook. The `to_json` clue reflects truth record or feature registry JSON writes.

### Feature Registry Structure

The feature registry JSON contains: `dataset_name`, `row_count`, `column_count`, `feature_set_id`, `feature_count`, `feature_columns`, `feature_groups` (by type), `feature_info`, `exclude_prefixes`, `exclude_columns`, `label_source_column`, `label_source_type`, `quarantine_missing_pct`, `pipeline_mode`, `process_run_id`.

This makes the feature registry a self-contained description of the Silver feature set, usable by downstream notebooks without reloading the Parquet.

### SQL Persistence: EDA Summary Only

The SQL write is controlled by `WRITE_TO_POSTGRES = True`. When enabled, `log_silver_eda_sql` writes a Silver EDA summary to PostgreSQL — feature counts, missingness summary, anomaly rate, feature set identifier, and dataset name. This is not a write of the full Silver dataframe.

The SQL clue manifest shows `write_layer_dataframe` as a clue, but the source Cell 211 calls `log_silver_eda_sql` (not `write_layer_dataframe` directly). `write_layer_dataframe` is available in the imports and may be used inside utility internals or in commented-out alternative paths. The confirmed Silver SQL write function is `log_silver_eda_sql`.

The `globals()` manifest pattern (`notebook_globals=globals()`) passes all resolved runtime variables to the SQL helper. This is the same pattern used in Gold notebooks for manifest-driven SQL column resolution.

`read_layer_dataframe` and `read_sql` clues reflect either the SQL Bronze read alternative path or the SQL smoke check query.

## Truth, Audit, and Reproducibility Behavior

### Truth Record Sections

`silver_truth` is initialized with `initialize_layer_truth` immediately after parent truth resolution, carrying `parent_truth_hash=SILVER_PARENT_TRUTH_HASH`. Subsequent `update_truth_section` calls add named sections throughout the run:

- `"config_snapshot"` — cleaning recipe ID, quarantine threshold, parse success thresholds, pipeline mode
- `"runtime_facts"` — processed at UTC, parent hash, dataset name from parent truth, pipeline mode
- `"dataset_validation"` — validated dataset name, source column, method
- `"label_resolution"` — label source column, type, coverage statistics, has-label/has-status flags
- `"canonical_info"` — resolved time/step/asset/run columns and their parse success rates
- `"feature_set"` — feature set ID, feature count (pre-quarantine)
- `"sensor_drop_audit"` — quarantine threshold, kept/dropped features, missingness percentages, drop reasons
- `"runtime_facts"` updates again for label source and feature set after quarantine

`build_truth_record` finalizes row and column counts, then `save_truth_record` writes the completed truth JSON.

### `meta__truth_hash` and `meta__parent_truth_hash` in Every Row

`stamp_truth_columns` writes the Silver truth hash and parent truth hash into every dataframe row before the Parquet write. This means every row of the Silver Parquet carries its own lineage anchor — a downstream notebook loading only a slice of the Silver Parquet can still confirm which Bronze artifact it traces back to.

### Ledger as Structured Audit Trail

The ledger records every major step with `kind`, `step`, `message`, `why` (where supplied), `consequence` (where supplied), and `data` payload. Decision-category entries (`kind="decision"`) cover the five most consequential choices: dataset name validation, label source resolution, canonical identity construction, feature set finalization, and EDA summary output. Step-category entries cover all data transformations. Check-category entries cover sanity check passes.

The ledger JSON written at Cell 200 is the structured timeline of the Silver run — reproducible without re-executing the notebook.

### `SILVER_PROCESS_RUN_ID`

`make_process_run_id(...)` constructs a unique identifier for each Silver execution. It is written into the truth record and the feature registry. A fresh execution of the same notebook produces a new `SILVER_PROCESS_RUN_ID` even with identical inputs, allowing multiple Silver runs on the same dataset to be distinguished in the audit trail.

### W&B Tracking

`wandb.init` is called before Bronze loading; `finalize_wandb_stage` and `wandb_run.finish` are called after the Parquet is written. Bronze row/column counts are logged via `wandb_run.log`. The W&B run covers the full Silver transformation window, providing an external audit of the run even if the local ledger JSON is unavailable.

## Downstream Technical Handoff

The Silver Parquet (`SILVER_TRAIN_DATA_PATH / SILVER_TRAIN_DATA_FILE_NAME`) is the primary artifact consumed downstream. The workflow reference identifies Gold 01 as the direct downstream consumer. The feature registry JSON is also expected to be read downstream for feature set verification.

Whether Gold 01 reads directly from `SILVER_TRAIN_DATA_PATH` or via a SQL Silver table is Not determined from available source. The Silver SQL EDA summary table exists as metadata, not as the primary data path.

The Silver truth hash stamped into each row is consumed by downstream notebooks that call `extract_truth_hash(dataframe)` to retrieve `SILVER_PARENT_TRUTH_HASH` for their own truth chain.

The `FEATURE_SET_IDENTIFIER` written into the feature registry JSON is used downstream to verify that the feature set Gold receives matches what Silver selected.

## Key Technical Decisions

| Decision | Source Evidence | Why It Matters | Verification Method |
|---|---|---|---|
| Bronze load preferred path + alphabetical fallback | Cell 36: `if preferred_bronze.exists()` → `parquet_files = sorted(...).glob("*.parquet")`; `parquet_files[0]` | Provides resilience for minor filename changes between pipeline stages; prevents silent use of wrong files by logging a warning on multi-file ambiguity | Confirm `bronze_data_path` in ledger `"load_bronze"` step matches expected file path |
| Parent truth extraction before any transformation | Cell 39 comment: "capture parent hash before any transformation so Silver truth links back to the exact Bronze artifact" | The hash extracted at this point reflects the Bronze Parquet as received, not a Silver-modified copy; any downstream hash mismatch is attributable to Silver transforms, not the input | Run final lineage consistency check (Cell 208) and confirm `meta__parent_truth_hash` in Parquet matches logged `SILVER_PARENT_TRUTH_HASH` |
| Artifact directory deferral until `DATASET_NAME` confirmed | Cell 16: all paths set to `None`; Cell 41: `if DATASET_NAME is None: raise ValueError` guard before `build_artifact_dirs` | Prevents creation of artifact directories under a wrong or placeholder name if the parent truth lookup fails | Confirm `SILVER_ARTIFACTS_PATH` is non-None only after Cell 41 executes; confirm directory path contains resolved `DATASET_NAME` |
| `DATASET_NAME` verified from parent truth, not assigned | `validate_dataset_name_for_silver` docstring: "Silver does not resolve or assign dataset identity"; `fail_on_multiple_in_bronze=True` | Silver must not introduce a dataset name different from what Bronze established; mixed-source rows are detected and halted here before propagating to Gold | Confirm `DATASET_NAME` in Silver Parquet `meta__dataset` column matches Bronze parent truth `dataset_name` field |
| `PIPELINE_MODE` propagated from Bronze parent truth | Cell 39: `PARENT_PIPELINE_MODE = get_pipeline_mode_from_truth(parent_truth)`; `if PARENT_PIPELINE_MODE is not None: PIPELINE_MODE = PARENT_PIPELINE_MODE` | Silver must not override the synthetic vs real distinction that Bronze established; propagation ensures the correct `meta__pipeline_mode` value is stamped into Silver rows | Confirm Silver Parquet `meta__pipeline_mode` column matches Bronze truth `pipeline_mode` field |
| Two-level sanity check (general + Silver-specific) | Cell 12: 16 general variables; Cell 13: `SILVER_CFG` alone; both use `name not in globals()` | The Silver-specific check can fail independently if `load_notebook_context` returns without populating `stage_config`; separating the checks isolates the failure source | Confirm both ledger steps `"context_sanity_check"` and a silver-specific step are present in the ledger JSON |
| Defensive copy before metadata column addition | Cell 63: `dataframe = dataframe.copy()` with comment "Bronze-loaded frame is not mutated in-place" | Re-running Cell 63 after a downstream failure is safe because the Bronze-loaded state is not accumulated; prevents duplicate-column errors on re-run | Re-execute Cell 63 twice without re-running Cell 36; confirm row count and column count are unchanged |
| Junk import column removal: prefix candidates + regex | Cell 48: `is_junk_prefix or is_regex_match`; `UNNAMED_COLUMN_REGEX` for generic unnamed columns | Prefix candidates handle project-specific junk; regex handles standard CSV import index columns regardless of config; neither alone is sufficient | Confirm `junk_columns_found` in ledger `"remove_junk_import_columns"` step lists expected columns only |
| Canonical output name protection to `raw__*` | Cell 77: rename to `f"{raw_prefix}{canonical_name}"` before canonical build | Prevents canonical column builder from overwriting source data that happens to share a name with target canonical columns (`event_time`, `event_step`, `time_index`) | Inspect `rename_map` in ledger `"canonical_name_collision_protection"` step for any `raw__` renames |
| Label preferred over status in anomaly flag resolution | Cell 66: label candidates checked first; status candidates only if no label found | Label columns carry explicit human-annotated anomaly truth; status columns carry operational state; mixing both would produce a different `anomaly_flag` than either alone | Confirm `LABEL_SOURCE_TYPE` in Silver truth `label_resolution` section matches the column type actually present |
| No-source path defaults all rows to normal | Cell 94: `else: dataframe[ANOMALY_FLAG_COLUMN] = np.zeros(...)`; `method="no_label_or_status_available_default_all_normal"` | Prevents a crash on datasets temporarily missing a label column; produces a valid (if trivially normal) Silver output that can proceed through the pipeline | Confirm anomaly rate is 0 in quality checks when no label or status column is found |
| Episode IDs per asset/run group ordered by time | Cell 99: `group_columns` resolved from `meta__asset_id` and `meta__run_id`; sorted by `time_index` or `event_step` | Per-group ordering prevents cross-run or cross-asset episode IDs from being computed over a mixed temporal sequence; Gold evaluation uses episode IDs for window-level recall | Confirm `meta__episode_id` increments correctly at anomaly-to-normal transitions in a single asset/run slice |
| Feature selection: numeric + boolean only by default | Cell 114: `include_categorical_features=False`, `include_text_features=False`, `include_datetime_features=False` | Conservative default avoids accidental inclusion of high-cardinality string or datetime columns; categorical encoding is deferred and tracked separately | Confirm `FEATURE_GROUPS["categorical"]` is empty in `feature_registry.json` when default flags are used |
| MD5 hash of sorted feature column names | Cell 162: `"|".join(sorted(...))` before MD5; comment "sorted before hashing so column order cannot create a false mismatch" | If column order changed between runs but the same columns were selected, a non-sorted hash would produce a false mismatch; sorted hash detects only actual feature set changes | Reorder `FEATURE_COLUMNS` before calling `build_feature_set_identifier` and confirm the same hash is returned |
| Truth stamp before Parquet write | Finalization sequence: `stamp_truth_columns` → `save_data`; Silver truth hash present in every Parquet row | The Parquet must carry `meta__truth_hash` in its rows so downstream notebooks can read the Silver hash without the truth JSON; reversing the order would write a Parquet without the hash | Confirm `meta__truth_hash` column is present and non-null in the saved Silver Parquet |
| Final lineage consistency check re-reads saved JSON from disk | Cell 208: `load_json(silver_truth_path)` and `isinstance` check; hash equality assertions | Confirms that the truth JSON file was actually written (not just built in memory), that it is valid, and that the in-memory hash matches what the file contains | Confirm Cell 208 completes without `ValueError` or `FileNotFoundError` |
| SQL write scoped to EDA summary via `log_silver_eda_sql` | Cell 211: `log_silver_eda_sql(engine, ..., notebook_globals=globals())`; not `write_layer_dataframe` for the full dataframe | Writing the full Silver dataframe to SQL at the Pre-EDA stage would be premature; the EDA summary is a structured metadata record, not a data lake write | Confirm `log_silver_eda_sql` is the only confirmed SQL write in Cell 211; `write_layer_dataframe` is a clue artifact, not a direct Silver 01 call |

## Failure Modes and Guardrails

| Failure Condition | Behavior | Prevention / Guardrail |
|---|---|---|
| Bronze Parquet file absent from `BRONZE_TRAIN_DATA_PATH` | `FileNotFoundError` with path in message | Explicit existence check before `load_data` |
| Multiple Parquet files in Bronze path (fallback triggered) | `logger.warning` identifying first file used | Warning surfaced but execution continues with alphabetically first file |
| `meta__truth_hash` absent or `None` in Bronze frame | `ValueError`: "Silver input dataframe does not contain a readable meta__truth_hash value" | `extract_truth_hash` return value checked before use |
| Parent truth record not loadable from disk | `ValueError` raised by `load_parent_truth_record_from_dataframe` | Truth dir and hash confirmed before `initialize_layer_truth` |
| `DATASET_NAME` resolved to empty or None from parent truth | `ValueError`: "DATASET_NAME must be resolved from the Bronze parent truth before creating Silver artifacts" | Guard in Cell 41 before `build_artifact_dirs` |
| `meta__dataset` column missing from Bronze frame | `ValueError` in `validate_dataset_name_for_silver` | Checks column presence before reading values |
| Multiple unique values in `meta__dataset` | `ValueError`: "multiple values were found in 'meta__dataset'" | `fail_on_multiple_in_bronze=True` |
| Required general context variables missing | `NameError` with list of missing names (Cell 12) | `name not in globals()` check over all 16 required variables |
| `SILVER_CFG` missing | `NameError`: "Missing Silver context variables: ['SILVER_CFG']" | Silver-specific sanity check (Cell 13) |
| No `anomaly_flag` source found | All rows default to `anomaly_flag = 0`; no crash | `else` branch in Cell 94; method recorded as `"no_label_or_status_available_default_all_normal"` |
| Silver dataframe `meta__truth_hash` mismatch after stamp | `ValueError` comparing dataframe hash to `SILVER_TRUTH_HASH` | Cell 208 final lineage consistency check |
| `meta__parent_truth_hash` has multiple unique values in Silver Parquet | `ValueError`: "Silver dataframe has multiple parent truth hashes" | Cell 208 checks `len(silver_parent_values) != 1` |
| Silver truth JSON not created on disk | `FileNotFoundError` in Cell 208 | `Path(silver_truth_path).exists()` check |
| Silver truth JSON is not a valid dict | `TypeError` in Cell 208 | `isinstance(loaded_silver_truth_raw, dict)` check |
| Required lineage columns absent from Silver dataframe | `ValueError`: "Silver dataframe is missing required lineage columns: [...]" | Cell 208 checks `["meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode"]` presence |
| `WRITE_TO_POSTGRES = False` | SQL write skipped; print statement confirms | Boolean gate in Cell 211; all file artifacts unaffected |
| Junk column candidates list empty AND no unnamed columns present | No columns dropped; ledger records `"dropped": []` | Graceful no-op path in `remove_junk_import_columns` |
| Feature columns all quarantined by missingness | `FEATURE_COLUMNS` is empty after quarantine; Silver Parquet written with no model features | Not a hard failure; `FEATURE_COUNT = 0` in feature registry; downstream Gold will fail if it requires features |

## Verification Checklist

- Active notebook path is `notebooks/eda/EDA_Notebook_Pump_Silver_01_PreEDA.ipynb`
- Bronze Parquet exists at `BRONZE_TRAIN_DATA_PATH / BRONZE_TRAIN_DATA_FILE_NAME`
- `meta__truth_hash` column is present and non-null in Bronze Parquet (parent truth extractable)
- Bronze parent truth JSON exists at `TRUTHS_PATH / bronze / {dataset_name} / ...`
- `DATASET_NAME` in Silver Parquet `meta__dataset` column matches Bronze parent truth `dataset_name`
- `PIPELINE_MODE` in Silver Parquet `meta__pipeline_mode` column matches Bronze parent truth `pipeline_mode`
- Silver artifact directory exists at `SILVER_ARTIFACTS_PATH` containing expected subdirectories
- `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` columns present and non-null in Silver Parquet
- `meta__parent_truth_hash` in Silver Parquet has exactly one unique value matching `SILVER_PARENT_TRUTH_HASH`
- Silver truth JSON exists in `SILVER_LINEAGE_DIR` and loads as valid dict
- Feature registry JSON exists in `SILVER_REGISTRY_DIR` with non-empty `feature_columns` list
- `FEATURE_SET_IDENTIFIER` in feature registry matches `build_feature_set_identifier(FEATURE_COLUMNS)` computed from loaded Silver Parquet
- Dropped sensors Parquet exists in `SILVER_PROFILE_DIR` (if any features were quarantined)
- Ledger JSON exists in `SILVER_LINEAGE_DIR`
- Ledger contains `"context_sanity_check"`, `"load_bronze"`, `"validate_dataset_name"`, `"resolve_label_or_status_source"`, `"build_canonical_identity_and_order_master"`, `"finalize_feature_set"`, `"silver_finalize_export"` steps
- `anomaly_flag` column present in Silver Parquet with only 0 and 1 values
- `meta__episode_id` column present in Silver Parquet
- Column layout in Silver Parquet follows the canonical ordering: `meta__*` columns first, then canonical columns, then label columns, then feature columns
- Final lineage consistency check (Cell 208) completes without error
- If `WRITE_TO_POSTGRES = True`: Silver EDA summary is readable from the appropriate SQL table

## Source-Limited Items

- Whether Gold 01 reads the Silver Parquet directly from `SILVER_TRAIN_DATA_PATH` or from the Silver SQL table is Not determined from available source.
- The exact SQL table name that `log_silver_eda_sql` writes to is Not determined from available source; it is resolved inside the `log_silver_eda_sql` utility function.
- Whether `CORRELATION_THRESHOLD` from `SILVER_CFG` is applied in Silver 01 (feature correlation deselection) is not confirmed from visible source cells; the workflow reference mentions it as a config value but no correlation-based deselection call is visible in the cell listing.
- Whether the ledger object (`CTX.ledger`) persists across session boundaries or is re-initialized on each `load_notebook_context` call is Not determined from available source.
- Whether the W&B dataset artifact registration in `finalize_wandb_stage` includes the Silver Parquet path or only metadata is Not determined from available source.
- Whether Silver 02a or Silver 02b reads directly from the Silver 01 Parquet output path or from the SQL Silver table is Not determined from available source.
