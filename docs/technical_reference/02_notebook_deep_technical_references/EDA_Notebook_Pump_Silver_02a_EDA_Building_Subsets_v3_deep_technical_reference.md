# Silver 02a Deep Technical Reference

## Purpose of This Deep Reference

This document covers technical decisions in Silver 02a (Subset Builder and Clean-Normal Constructor) that require deeper explanation than the workflow reference provides. The workflow reference describes what the notebook does step by step. This document explains why Silver 02a uses a separate EDA stage config distinct from Silver 01, why artifact directories are created before the Silver Parquet is loaded (unlike Silver 01's deferred pattern), why the feature registry is resolved from the Silver parent truth record rather than config, why state normalization uses a priority column detection pattern, why the clean-normal profiling pipeline trims episode edges before windowing, why window quality is scored across three dimensions, why row deviation is normalized by IQR with a floor, why the `exclude` quality class maps to `normal_contaminated` rather than `abnormal`, why data output Parquets go to `SILVER_TRAIN_DATA_PATH` while EDA artifacts go to a separate tree, and why Silver 02a creates its own truth record rather than extending Silver 01's.

## Technical Scope

- Separate `silver_eda` stage config (`SILVER_EDA_CFG`) vs Silver 01's `silver` stage config
- Artifact directory creation before Silver Parquet load (contrast with Silver 01's deferred pattern)
- W&B disabled and preserved as triple-quoted string
- Silver parent truth hash extraction before any filtering
- Feature registry resolution from Silver parent truth record via `get_artifact_path_from_truth`
- Feature column intersection with dataframe and fallback to `sensor_*` prefix detection
- Source state normalization: priority column detection and canonical three-value mapping
- `LABEL_COLUMN` bound to normalized column, not raw source column
- Five-stage clean-normal profiling pipeline: episode assignment → edge trim → windowing → stats → scoring/baseline
- Edge trimming design: transitional-row exclusion and short-episode drop
- Window quality scoring: three-dimension rank-based composite score
- IQR floor (`safe_iqr_floor`) preventing division by zero in value deviation
- Delta deviation computed within episode groups via groupby diff
- Two-threshold quality classification: `suspect` and `exclude` classes
- `normal_contaminated` as the profiled label for excluded normal rows
- `silver_subset_df` rebuilt from `.copy()` after profiled-state constants are defined
- Three data Parquet outputs written to `SILVER_TRAIN_DATA_PATH`
- Sensor baseline artifacts written with direct `to_csv` and `json.dump`
- Subset truth record (`silver_eda_subsets_truth`) independent of Silver 01's truth record
- Two process run IDs: `SILVER_PROCESS_RUN_ID` and `SILVER_SUBSET_PROCESS_RUN_ID`
- SQL write gate and `log_silver_eda_sql` scope

## Source Grounding

Sources used:

- `notebooks/eda/EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3.ipynb`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3_code_reference.md`
- `notebook_inventory.json`
- `artifact_io_manifest.json`
- `sql_touchpoints.json`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Silver_01_PreEDA_deep_technical_reference.md`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Bronze_01_Preprocessing_deep_technical_reference.md`
- `technical_reference/00_project_manual/notebook_relationship_map.md`
- `technical_reference/04_deep_utility_function_references/utils__core__truths_deep_function_reference.md`

The active Silver 02a notebook source is the source of truth for all function behavior, variable names, and design decisions documented here.

## Stage Role in the Medallion Pipeline

Silver 02a is the third notebook in the active chain (Bronze 01 → Silver 01 → Silver 02a). Its role is not cleaning or schema normalization — those are Silver 01's responsibilities. Silver 02a performs analytical profiling: it identifies which normal-labeled rows are genuinely stable normal behavior and which may be transitional or contaminated, then assigns a `machine_status__profiled` label to every row.

This profiling distinction matters because Gold's Isolation Forest models learn what normal looks like. If normal rows from the edges of fault transitions are included in the normal reference, the model's normal boundary is degraded. Silver 02a addresses this before data reaches Gold by separating `normal_clean`, `normal_suspect`, `normal_contaminated`, `abnormal`, and `recovery` rows into distinct labels.

Silver 02a is not a validation notebook. It produces analytical artifacts (sensor baseline profiles, per-sensor plots, profiled Parquet subsets) that serve both downstream EDA and downstream modeling. Its `CONTEXT_STAGE = "silver_eda"` distinguishes it from Silver 01's `"silver"` stage throughout the config and truth system.

## Input Contract and Lineage

### Silver Parquet as Primary Input

The primary input is the Silver Pre-EDA Parquet at `SILVER_TRAIN_DATA_PATH / SILVER_TRAIN_DATA_FILE_NAME`. The load pattern mirrors Silver 01: the canonical filename is tried first; if absent, a glob finds the first `.parquet` in the directory with a `logger.warning`. `FileNotFoundError` is raised if no Parquet files are found.

After loading, `SILVER_TRUTH_HASH = extract_truth_hash(silver_eda_df)` is called immediately before any filtering or column additions. The Cell 34 comment states: "Capture parent truth hash before any filtering so the subset truth record can link back to this Silver version." If the hash is `None`, `ValueError` halts execution.

`load_parent_truth_record_from_dataframe(dataframe=silver_eda_df, truth_dir=TRUTHS_PATH, parent_layer_name="silver", ...)` then loads the Silver 01 truth JSON. From this record, Silver 02a reads:

- `DATASET_NAME` via `get_dataset_name_from_truth`
- `SILVER_TRUTH_HASH` via `get_truth_hash`
- `SILVER_PARENT_TRUTH_HASH` via `get_parent_truth_hash` (Bronze's hash, one level deeper)
- `PIPELINE_MODE` via `get_pipeline_mode_from_truth`
- `LABEL_SOURCE_COLUMN`, `LABEL_SOURCE_TYPE` from `silver_truth["runtime_facts"]["label_resolution"]`
- `feature_registry_dir` from `get_artifact_path_from_truth(silver_truth, "feature_registry_dir")`

### Feature Registry Resolved from Silver Truth

`FEATURE_REGISTRY_PATH` is constructed from the `feature_registry_dir` extracted from the Silver 01 truth record plus the canonical registry filename `{DATASET_NAME}__silver__feature_registry.json`. The Cell 36 ledger comment explains the design: "Silver subset building should inherit resolved feature metadata from Silver rather than rebuilding it from config."

This is a deliberate lineage decision: the subset builder uses exactly the feature set that Silver 01 finalized, preserving the `FEATURE_SET_IDENTIFIER` chain without independently re-selecting features.

The feature registry is validated as non-None and as a `dict` before use. Missing or malformed registry files raise `ValueError` or `TypeError`. After loading, `FEATURE_COLUMNS` is constructed from the registry's `feature_columns` list, filtered to only those columns present in the Silver dataframe. Columns in the registry but absent from the dataframe are logged as warnings. An empty result after intersection raises `ValueError`.

If `FEATURE_REGISTRY_PATH` is `None` (i.e., the parent truth record did not record a feature registry directory), `ValueError` halts execution before any registry load is attempted.

### Fallback to `sensor_*` Prefix Detection

Cell 43 contains: "Prefer the feature registry columns when available, then fall back to dataframe sensor_* columns. This keeps the subset builder aligned with the Silver feature lineage." When `CONFIG_SENSOR_COLUMNS` is empty, `SENSOR_COLUMNS` is derived by scanning the Silver dataframe for columns beginning with `sensor_`. This fallback makes the notebook runnable even when the feature registry is unreachable, at the cost of using an unregistered feature set.

### Config via `load_notebook_context(stage="silver_eda")`

`load_notebook_context(stage="silver_eda", dataset="pump", mode="train", profile="default")` is the bootstrap call. The stage string `"silver_eda"` is distinct from Silver 01's `"silver"` — Silver 02a uses a dedicated EDA config block (`SILVER_EDA_CFG`, aliased from `CTX.stage_config`) that carries its own threshold values: `TRIM_FRAC`, `WINDOWS_PER_EPISODE`, `MIN_WINDOW_ROWS`, `KEEP_WINDOW_FRAC`, `VALUE_DEVIATION_THRESHOLD`, `DELTA_DEVIATION_THRESHOLD`, `SUSPECT_SENSOR_COUNT`, `EXCLUDE_SENSOR_COUNT`.

### Lineage Significance

Silver 02a occupies a middle position in the lineage chain. Its subset truth record (`silver_eda_subsets_truth`) records `SILVER_TRUTH_HASH` as its parent — Silver 01's hash — making Silver 02a's outputs traceable to the Silver Pre-EDA run without depending on a direct file path reference.

## Silver Data Preparation Methodology

### Artifact Directory Creation Before Silver Load

Silver 02a creates its artifact directory tree immediately after context setup, before loading the Silver Parquet. This is the inverse of Silver 01's deferred pattern. The workflow reference explains: "Silver 02a already has `DATASET_NAME` from config and trusts that it matches the Silver Parquet it will load." The directory names include `DATASET_NAME_CONFIG` from `DATASET_CFG`, so no parent truth lookup is required before creating them.

Directories created: `SILVER_EDA_ARTIFACT_DIR` (root under `artifacts/silver/<dataset>/eda/`) and all subdirectories: `sensor_profiles`, `summaries`, `subsets`, `lineage`, `config`, `metadata`, `correlation_analysis`, `distribution_plots`, `pca`, `generator_inputs`, `aligned_onset_plots`, `timeline_overlays`. Backward-compatible aliases (`SENSOR_PROFILE_DIR`, `CORRELATION_ARTIFACT_DIR`, etc.) point to canonical paths.

`export_config_snapshot` writes the full resolved config to `SILVER_EDA_CONFIG_DIR` immediately after directory creation.

### Source State Normalization

The source state column is detected by probing `silver_eda_df.columns` for candidates in priority order: `"machine_status__synthetic"` → `"machine_status"` → `"status"` → `"state"`. The detected column is stored as `STATE_COL_SOURCE_RAW`. If none are found, `KeyError` is raised with the list of expected candidates.

`normalize_machine_state` maps raw variants to three canonical values:
- `"normal"`, `"norm"`, `"ok"` → `"normal"`
- `"broken"`, `"abnormal"`, `"fault"`, `"failure"`, `"failed"` → `"abnormal"`
- `"recovering"`, `"recovery"`, `"cooldown"`, `"cooling"` → `"recovery"`

The normalized values are written into `silver_eda_df["machine_status__synthetic"]`. When `STATE_COL_SOURCE_RAW` is already `"machine_status__synthetic"`, the normalization is applied in-place as a lowercasing strip. When a different column is detected, the mapping is applied via `.map(normalize_machine_state)`.

### `LABEL_COLUMN` Bound to Normalized Column

Cell 43 explicitly binds `LABEL_COLUMN = STATE_COL_SOURCE` (i.e., `"machine_status__synthetic"`), the normalized column. The comment states: "This avoids mixed casing such as NORMAL / normal and keeps all downstream profiling logic using the same state values." All windowing functions receive `LABEL_COLUMN` as the label reference, not `STATE_COL_SOURCE_RAW`.

### Two Sanity Checks (General + Silver EDA–Specific)

The general sanity check validates 16 shared context variables using `name not in globals()`. The Silver EDA–specific check validates `SILVER_EDA_CFG` alone. Missing variables raise `NameError`. Both checks add a `ledger.add(kind="check", ...)` entry on pass.

The two original setup blocks (standalone `configure_logging` and `Ledger` initialization) exist as triple-quoted strings in Cells 23 and 29. They are not executed; `load_notebook_context` handles both. These serve as documentation of the pre-shared-context setup.

## Silver Transformation, EDA, and Feature Logic

### Five-Stage Clean-Normal Profiling Pipeline

The core analytical block executes five functions in sequence in Cell 57 on the full `silver_eda_df`:

**Stage 1 — `assign_normal_episodes`**: Detects transitions between normal and non-normal rows using a cumsum on a boolean status-change mask. Each contiguous run of `normal` rows receives a monotonically incrementing `__normal_episode_id`. Non-normal rows receive `NaN` episode IDs. Episode boundaries are identified at every status-change transition, not only at anomaly onset.

**Stage 2 — `trim_normal_episode_edges`**: Removes the first and last `TRIM_FRAC` fraction of each episode. The design rationale is documented in the workflow reference: rows near episode boundaries may capture fault-onset transients at the end or post-recovery stabilization at the start. Trimming isolates the stable interior of each normal run. Episodes yielding fewer than `min_rows_after_trim` (1000 rows) after trimming are dropped entirely by skipping the `kept_index` assignment — they contribute zero rows to windowing. The result is the `__keep_after_episode_trim` boolean column.

**Stage 3 — `create_episode_windows`**: Splits each trimmed episode into `WINDOWS_PER_EPISODE` equal segments using `np.array_split`. Each window receives a string label `"episode_{N}__window_{M}"`. Windows below `MIN_WINDOW_ROWS` (500) are excluded from the `__window_id` assignment. `np.array_split` produces windows as equal as possible given integer division; edge windows may be one row shorter. Only rows in kept episodes with `__keep_after_episode_trim == True` participate.

**Stage 4 — `calculate_window_sensor_stats`**: For every (window × sensor) pair, computes: median, mean, std, 5th percentile, 25th percentile (Q25), 75th percentile (Q75), 95th percentile, IQR (Q75 − Q25), and the corresponding delta statistics (row-to-row change via `.diff()`): delta median, delta mean, delta std, `delta_abs_median`, `delta_abs_q95`, `delta_iqr`. The result is `window_sensor_stats_df` with one row per (window × sensor).

**Stage 5 — `score_and_filter_windows` + `build_final_sensor_baseline`**: Windows are ranked on three dimensions:
- IQR rank percentile (lower IQR = more stable sensor behavior)
- `delta_abs_q95` rank percentile (lower delta = smoother sensor changes)
- Missing data rate rank percentile (lower missing rate = more complete)

The composite `window_quality_score` is the mean of the three rank percentiles. The bottom `KEEP_WINDOW_FRAC` (80%) by score are classified `"keep"`. `build_final_sensor_baseline` then aggregates per-sensor statistics from kept windows only, taking medians of medians to produce `final_sensor_baseline_df` — the learned stable-normal sensor profile.

### IQR Normalization with Floor

`score_rows_against_sensor_baseline` normalizes value deviation as:

```
value_deviation = abs(sensor_value - baseline_median) / max(baseline_iqr, safe_iqr_floor)
```

`safe_iqr_floor = 1e-6` prevents division by zero for sensors with perfectly flat normal behavior (IQR = 0 in the baseline). This ensures that even an exactly constant sensor can produce a non-zero deviation when a row deviates from the baseline median.

Similarly, `safe_delta_floor = 1e-6` applies to `baseline_delta_abs_q95` in the delta deviation normalization.

### Delta Deviation Within Episode Groups

Delta deviation uses `scored_df.groupby("__normal_episode_id")[sensor].diff()`. Computing diff within episode groups prevents the first row of a new episode from producing a spuriously large delta against the last row of the previous (non-normal or different-episode) section. Without the groupby, cross-episode boundaries would introduce artificial delta flags at episode starts.

### Per-Sensor Deviation Flags

For each sensor, `score_rows_against_sensor_baseline` computes four derived columns:
- `{sensor}__value_deviation`: the continuous normalized value deviation
- `{sensor}__delta_deviation`: the continuous normalized delta deviation
- `{sensor}__value_abnormal_flag`: boolean, True when `value_deviation > VALUE_DEVIATION_THRESHOLD`
- `{sensor}__delta_abnormal_flag`: boolean, True when `delta_deviation > DELTA_DEVIATION_THRESHOLD`
- `{sensor}__any_abnormal_flag`: logical OR of the two flags

Three aggregate counts summarize across all sensors per row: `normal_value_abnormal_sensor_count`, `normal_delta_abnormal_sensor_count`, `normal_total_abnormal_sensor_count`.

Only normal rows have flags applied; non-normal rows accumulate zero flags.

### Two-Threshold Quality Classification

`classify_normal_training_quality` assigns `normal_training_quality_class` using a two-threshold design:

- Non-normal rows → `"not_normal"`
- Normal rows, `normal_total_abnormal_sensor_count < SUSPECT_SENSOR_COUNT (8)` → `"clean"`
- Normal rows, `count >= SUSPECT_SENSOR_COUNT (8)` → `"suspect"`
- Normal rows, `count >= EXCLUDE_SENSOR_COUNT (21)` → `"exclude"` (overrides `"suspect"`)

`is_clean_normal_for_training` boolean is also set per row. The two-threshold design provides more analytical granularity than a binary clean/dirty split: `suspect` rows are questionable but not definitively contaminated, while `exclude` rows show enough deviant sensors to warrant exclusion from the clean-normal reference. Both classes can still be preserved in the output dataframe for EDA purposes.

### Profiled-State Mapping

Cell 104 declares canonical profiled-state constants and maps `machine_status__profiled`:

| Source State (`machine_status__synthetic`) | Quality Class (`final_row_quality_class`) | `machine_status__profiled` |
|---|---|---|
| `normal` | `clean` | `"normal_clean"` |
| `normal` | `suspect` | `"normal_suspect"` |
| `normal` | `exclude` | `"normal_contaminated"` |
| `abnormal` | any | `"abnormal"` |
| `recovery` | any | `"recovery"` |

The `exclude` quality class maps to `"normal_contaminated"`, not `"abnormal"`. This preserves the semantic distinction: excluded normal rows are machine-status-normal rows that were flagged by the profiling pipeline for sensor behavior, not confirmed fault-state rows. Mixing them with `abnormal` would misrepresent their source label.

`silver_subset_df` is rebuilt from `scored_normal_quality_df.copy()` in Cell 104 (after the profiled-state constants are defined). This rebuild is required because the constants (`PROFILED_NORMAL_CLEAN_VALUE`, etc.) are declared in Cell 104 and are unavailable in Cell 89 where the initial quality columns were added.

### Sensor Profile Plots

`plot_all_sensor_profiles` generates one PNG per sensor, saved to `SENSOR_PROFILE_DIR/plots`. Each plot shows raw sensor values with the baseline median and IQR range overlaid. `SHOW_ALL_SENSOR_PROFILE_PLOTS = False` by default to prevent inline rendering of ~25+ plots; files are still saved to disk. Spot-check cells call `display_sensor_profile` and `plot_sensor_profile_with_baseline` for a small set of named sensors.

These plots are visual review artifacts; they do not enter the modeling pipeline. They provide a human-readable check on whether the learned baseline is reasonable for each sensor.

## Silver Validation and Data Quality Checks

### Hard Failures

| Condition | Failure |
|---|---|
| No Silver Parquet found in `SILVER_TRAIN_DATA_PATH` | `FileNotFoundError` |
| `SILVER_TRUTH_HASH` is `None` after load | `ValueError`: "Could not resolve meta__truth_hash from Silver dataframe" |
| `FEATURE_REGISTRY_PATH` is `None` from parent truth | `ValueError` before `load_json` |
| Feature registry loads as `None` | `ValueError` |
| Feature registry is not a `dict` | `TypeError` |
| `feature_columns` in registry is empty | `ValueError` |
| All feature columns absent from Silver dataframe after intersection | `ValueError` |
| No state column found among candidates | `KeyError` with expected candidate list |
| No sensor columns resolved (registry empty and no `sensor_*` prefix columns) | `ValueError` |
| `SILVER_EDA_CFG` missing from globals | `NameError` |

### Soft Warnings

Feature columns in the registry absent from the Silver dataframe are logged as `logger.warning` with the list of missing columns (up to 20). Multiple Silver Parquet files in the glob fallback path produce `logger.warning` identifying the first file used.

Episodes too short to produce windows are silently dropped by `create_episode_windows` and `trim_normal_episode_edges`. No error is raised; coverage reduces. The workflow reference identifies this as an implicit check rather than a hard failure.

### SQL Smoke Check

`read_sql_dataframe(engine, "SELECT table_schema, table_name ...")` against `information_schema.tables` runs before any data load. This is a read-only connectivity check independent of the `WRITE_TO_POSTGRES` gate.

## Artifact and SQL Persistence

### Data Output Parquets (in `SILVER_TRAIN_DATA_PATH`)

Three profiled Parquet files are written to `SILVER_TRAIN_DATA_PATH` (the same directory as the Silver 01 Parquet):

| File | Content |
|---|---|
| `{dataset}__silver_subsets__profiled_dataframe.parquet` | All rows from `silver_subset_df` with `machine_status__profiled` column |
| `{dataset}__silver_subsets__normal_clean.parquet` | Rows where `machine_status__profiled == "normal_clean"` |
| `{dataset}__silver_subsets__normal_contaminated.parquet` | Rows where `machine_status__profiled == "normal_contaminated"` |

All three use direct `.to_parquet(..., index=False)`, not the `save_data()` wrapper. The data directory (`SILVER_SUBSET_DATA_DIR`) resolves to `SILVER_TRAIN_DATA_PATH` directly (a commented-out `subset_outputs/<dataset>` path exists but is not active). Placing outputs alongside the Silver 01 Parquet makes them accessible to downstream notebooks using the same `SILVER_TRAIN_DATA_PATH` path resolution.

A commented-out `gold_train__normal_clean.parquet` alias for Gold is present but not executed.

### EDA and Sensor Baseline Artifacts (in Artifact Tree)

| File | Format | Write Method | Location |
|---|---|---|---|
| `silver_sensor_baseline_profiles.csv` | CSV | `to_csv(index=False)` | `SENSOR_PROFILE_DIR` |
| `silver_sensor_baseline_profiles.json` | JSON | `json.dump(..., default=str)` | `SENSOR_PROFILE_DIR` |
| `{dataset}__silver_subsets__summary.json` | JSON | `json.dump` | `SILVER_EDA_SUMMARY_DIR` |
| Per-sensor profile PNGs | PNG | `plt.savefig(...)` | `SENSOR_PROFILE_DIR/plots/` |
| Config snapshot | YAML/JSON | `export_config_snapshot` | `SILVER_EDA_CONFIG_DIR` |
| Ledger JSON | JSON | `ledger.write_json` (via `ledger`) | `SILVER_EDA_LINEAGE_DIR` |

All EDA artifacts use direct write calls (`to_csv`, `json.dump`, `plt.savefig`) rather than project-level wrappers. The artifact manifest confirms `to_csv`, `to_parquet`, `to_json`, `save_json` as write clues. `save_json` likely reflects the `save_json` import used for the truth record (from `utils.core.file_io`) or the truth utility internals.

The baseline JSON carries `artifact_type`, `created_at_utc`, `source_notebook`, `profile_method`, `state_column_used_for_profile`, `normal_values_used_for_profile`, `baseline_config` (all threshold constants), and `summary` (sensor count, median baseline rows/windows). This self-describing format allows downstream consumers to understand the baseline configuration without re-reading the notebook source.

### Subset Truth Record

`initialize_layer_truth(layer_name="silver", parent_truth_hash=SILVER_TRUTH_HASH, process_run_id=SILVER_SUBSET_PROCESS_RUN_ID, ...)` creates `silver_eda_subsets_truth`. This is not Silver 01's truth record — it is a new truth record for the subset-building stage, with Silver 01's hash as its parent.

Four sections are populated:
- `"config_snapshot"`: stage, layer, dataset name, pipeline mode, run mode
- `"runtime_facts"`: parent hash, state columns, profiling method, profiled/source state counts, row count
- `"artifact_paths"`: all three Parquet paths, summary JSON, data dir, artifact dir
- `"notes"`: human-readable purpose statement

`build_truth_record` finalizes row/column counts. `save_truth_record` writes to `TRUTHS_PATH/silver/<dataset>__silver__truth__<SILVER_EDA_SUBSETS_TRUTH_HASH>.json`. `append_truth_index` adds the entry to the project-wide truth index. `ledger.add` records both `SILVER_EDA_SUBSETS_TRUTH_HASH` and `SILVER_TRUTH_HASH`.

### SQL Persistence

`WRITE_TO_POSTGRES = True` controls the SQL write gate. When enabled, `log_silver_eda_sql(engine, CAPSTONE_SCHEMA, DATASET_ID, RUN_ID, notebook_globals=globals(), ...)` writes summary metadata to `capstone.pipeline_runs`, `capstone.data_quality_events`, and `capstone.pipeline_artifacts`. This is a metadata/summary write, not a full dataframe write. The `globals()` manifest pattern is the same as in Silver 01 and the Gold notebooks.

The SQL touchpoint manifest shows `read_layer_dataframe`, `read_sql`, and `write_layer_dataframe` as clues. `read_sql` reflects the smoke check query. `read_layer_dataframe` and `write_layer_dataframe` are in the import list; `write_layer_dataframe` is the confirmed write clue and likely refers to internals of `log_silver_eda_sql`.

## Truth, Audit, and Reproducibility Behavior

### Two Process Run IDs

Two independent process run IDs are generated:
- `SILVER_PROCESS_RUN_ID` — identifies the parent Silver Pre-EDA run (uses a prefix from `SILVER_EDA_CFG["process_run_id_prefix"]`)
- `SILVER_SUBSET_PROCESS_RUN_ID` — identifies this specific Silver 02a subset-building execution (uses `SILVER_EDA_CFG.get("subset_process_run_id_prefix", "silver_subset_process")`)

The subset truth record uses `SILVER_SUBSET_PROCESS_RUN_ID`. Having two distinct IDs allows each Silver 02a execution to be uniquely identified in the truth index even when run against the same parent Silver Parquet.

### Truth Chain Continuity

Silver 02a reads Silver 01's truth hash from the Silver Parquet (`extract_truth_hash`) and stores it as `SILVER_TRUTH_HASH`. This hash becomes `parent_truth_hash` in the subset truth record. The result is a verifiable lineage chain: Bronze → Silver 01 → Silver 02a subsets. Silver 02a's truth record also carries `SILVER_PARENT_TRUTH_HASH` (the Bronze hash read from Silver 01's truth record) in its `runtime_facts`.

### Ledger Throughout

The ledger records: `"context_loaded"`, `"context_sanity_check"`, `"load_silver"`, `"normalize_source_state"`, `"save_sensor_baseline_profiles"`, and `"save_subset_outputs"` (with all four artifact paths). The ledger JSON is written via the ledger object at completion. Every significant decision and artifact write is represented.

### W&B Disabled

Cell 25 contains the `wandb.init` call in a triple-quoted string that is not executed. The workflow reference confirms: "W&B integration is disabled; block is preserved in triple-quotes for future activation." Silver 02a does not produce W&B runs, dataset artifacts, or W&B logs.

## Downstream Technical Handoff

The three profiled Parquet files in `SILVER_TRAIN_DATA_PATH` are the primary downstream consumables. The workflow reference identifies Silver 02b and Gold 01 as consumers:

- The profiled dataframe Parquet (`*__profiled_dataframe.parquet`) carries `machine_status__profiled` for all rows — used by Silver 02b for EDA and Gold 01 for preprocessing
- The clean-normal Parquet (`*__normal_clean.parquet`) carries only rows where profiling confirmed stable normal behavior — used as the reference for Gold Isolation Forest training
- The contaminated-normal Parquet (`*__normal_contaminated.parquet`) carries excluded normal rows — available for EDA reference

The sensor baseline artifacts (`silver_sensor_baseline_profiles.csv` and `.json`) are referenced by Silver 02b and the synthetic generator. The subset truth record's `"artifact_paths"` section records all three Parquet paths, allowing downstream truth-record readers to locate the outputs without hardcoding paths.

Whether Silver 02b or Gold 01 read these artifacts via the Silver 02a truth record's `artifact_paths` section or via directly configured path constants is not determined from Silver 02a source alone.

## Key Technical Decisions

| Decision | Source Evidence | Why It Matters | Verification Method |
|---|---|---|---|
| `CONTEXT_STAGE = "silver_eda"` gives Silver 02a its own config block | Cell 7: `SILVER_EDA_CFG = CTX.stage_config` from `stage="silver_eda"` context; Cell 11: Silver-specific check verifies `SILVER_EDA_CFG` | Silver 02a's windowing, scoring, and quality thresholds are independent of Silver 01's cleaning config; changing Silver 01's config cannot accidentally alter Silver 02a's profiling behavior | Confirm `SILVER_EDA_CFG` in ledger `"context_loaded"` step; confirm stage string is `"silver_eda"` |
| Artifact directories created before Silver Parquet load (no deferral) | Cell 14: directories created before Cell 32 (Silver load); workflow reference: "Silver 02a already has `DATASET_NAME` from config" | Silver 02a trusts config-sourced `DATASET_NAME` so no post-load verification is needed before directory creation; unlike Silver 01 which cannot trust config alone before parent truth confirmation | Confirm `SILVER_EDA_ARTIFACT_DIR` exists after Cell 14 but before Cell 32 runs |
| W&B disabled — preserved as triple-quoted string | Cell 25: `''' wandb_run = wandb.init(...) '''`; workflow reference: "not executed — `load_notebook_context` handles both" | Silver 02a's analytical outputs are file-based artifacts; W&B overhead is deferred to future activation; the preserved block documents what would be tracked | Confirm no `wandb.init()` is called in any executed cell; confirm `wandb_run` is not in globals |
| Parent truth hash extracted before any filtering | Cell 34 comment: "Capture parent truth hash before any filtering so the subset truth record can link back to this Silver version" | If extracted after row filtering, the hash would reflect a derived slice, not the complete Silver 01 output; the link in the subset truth record would be ambiguous | Confirm `SILVER_TRUTH_HASH` equals the `meta__truth_hash` value in the unfiltered `silver_eda_df` before any transformation |
| Feature registry resolved from Silver parent truth, not config | Cell 36 ledger `why` field: "Silver subset building should inherit resolved feature metadata from Silver rather than rebuilding it from config" | Ensures subset builder uses the same `FEATURE_SET_IDENTIFIER` and `FEATURE_COLUMNS` that Silver 01 finalized; prevents silent feature-set drift if Silver 01 was re-run with different columns | Confirm `FEATURE_REGISTRY_PATH` in ledger `"load_silver_feature_registry"` step matches path derived from Silver 01 truth |
| `LABEL_COLUMN = STATE_COL_SOURCE` — normalized column used for all profiling | Cell 43 comment: "Avoids mixed casing such as NORMAL / normal" | All windowing and scoring functions receive a guaranteed lowercase-normalized state column; mixed-case values would produce missed `isin(normal_values)` matches in episode assignment | Confirm `silver_eda_df["machine_status__synthetic"]` is fully lowercase before windowing begins |
| Feature registry columns intersected with dataframe before use | Cell 36: `FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if col in silver_eda_df.columns]` with `ValueError` on empty result | Registry may reference columns quarantined by a different Silver 01 run; intersection prevents `KeyError` in windowing stats while the `ValueError` surfaces total feature loss | Confirm `missing_feature_columns` warning appears only for genuinely absent columns |
| Edge trimming (`TRIM_FRAC = 0.10`) excludes transitional rows | Cell 47: removes first and last 10% of each episode; episodes < 1000 rows after trim are dropped entirely | Rows at episode edges may contain fault-onset or post-recovery transients that contaminate the normal baseline; excluding them produces a cleaner stable-behavior reference | Confirm trimmed episode rows do not appear in `window_sensor_stats_df`; verify `__keep_after_episode_trim == False` for edge rows |
| Window quality: three-dimension rank composite | Cell 53: `window_quality_score = (iqr_rank_pct + delta_rank_pct + missing_rank_pct) / 3`; bottom `KEEP_WINDOW_FRAC` (80%) retained | No single metric captures all sources of window instability; IQR measures sensor variability, delta measures rate-of-change, missingness measures coverage; combining all three gives a more robust stability signal | Inspect `window_quality_df` to confirm high-IQR or high-missing-rate windows are classified `"exclude"` |
| `safe_iqr_floor = 1e-6` prevents zero-division in deviation normalization | Cell 77: `safe_iqr = max(float(baseline_iqr), safe_iqr_floor)` | Sensors with constant normal behavior produce `baseline_iqr = 0`; dividing by zero would produce `NaN` or `inf` deviation values; the floor ensures any actual deviation from a constant sensor still produces a meaningful score | Test with a synthetic constant-sensor column; confirm `value_deviation` is finite |
| Delta diff within episode groups — `groupby("__normal_episode_id").diff()` | Cell 77: `scored_df.groupby("__normal_episode_id")[sensor].diff()` | Cross-episode diff would produce a large spurious delta at the start of each episode; per-episode diff restricts delta computation to within-episode changes | Confirm `delta_deviation` for the first row of each episode is `NaN` (not a cross-boundary value) |
| Two-threshold quality classification (suspect at 8, exclude at 21) | Cell 43: `SUSPECT_SENSOR_COUNT = 8`, `EXCLUDE_SENSOR_COUNT = 21`; Cell 79: `exclude` overrides `suspect` when count meets the higher threshold | `suspect` rows are flagged but not excluded — they may be transitional or sensor-noisy; `exclude` rows show enough total deviant sensors to warrant exclusion from the clean-normal reference | Inspect `normal_training_quality_class` distribution in `scored_normal_quality_df`; confirm `exclude` rows have `normal_total_abnormal_sensor_count >= 21` |
| `normal_contaminated` maps to `exclude` class, not to `abnormal` | Cell 104: `mask_normal_exclude = mask_source_normal & quality_class.eq("exclude")`; `silver_subset_df.loc[mask_normal_exclude, STATE_COL_PROFILED] = PROFILED_NORMAL_CONTAMINATED_VALUE` | These rows are source-labeled normal; mixing them with `abnormal` would falsely inflate the fault category; `normal_contaminated` preserves the semantic distinction for downstream EDA and modeling | Confirm `machine_status__profiled == "normal_contaminated"` only where `machine_status__synthetic == "normal"` |
| `silver_subset_df` rebuilt from `.copy()` in Cell 104, not extended from Cell 89 | Cell 104: `silver_subset_df = scored_normal_quality_df.copy()` after profiled-state constants defined | Constants (`PROFILED_NORMAL_CLEAN_VALUE`, etc.) are declared in Cell 104 and unavailable in Cell 89; a single-pass build would require forward-declaring the constants or accepting inconsistent labeling between the two cells | Confirm running Cell 104 without Cell 89 produces the same final `silver_subset_df` |
| Data Parquets written to `SILVER_TRAIN_DATA_PATH`, not to artifact EDA tree | Cell 106: `SILVER_SUBSET_DATA_DIR = SILVER_TRAIN_DATA_PATH`; commented-out `subset_outputs/<dataset>` alternative | Downstream notebooks (Gold 01) read from `SILVER_TRAIN_DATA_PATH`; placing outputs there avoids requiring Gold to resolve a different path; the EDA tree holds analysis artifacts, not pipeline data | Confirm profiled Parquet files exist in `SILVER_TRAIN_DATA_PATH`, not in `SILVER_EDA_ARTIFACT_DIR` |
| Subset truth record independent of Silver 01's truth record | Cell 109: `initialize_layer_truth(layer_name="silver", parent_truth_hash=SILVER_TRUTH_HASH, ...)` creates a new record; not `update_truth_section` on Silver 01's record | Silver 02a's subset profiling is a separate analytical act from Silver 01's cleaning; independent truth records allow Silver 02a to be re-run without invalidating Silver 01's record | Confirm two Silver-layer truth JSON files exist in `TRUTHS_PATH/silver/` — one for Silver 01 and one with the subset hash |

## Failure Modes and Guardrails

| Failure Condition | Behavior | Prevention / Guardrail |
|---|---|---|
| No Silver Parquet at canonical path and none found by glob | `FileNotFoundError` | Glob with explicit empty-list check before file assignment |
| `SILVER_TRUTH_HASH` is `None` | `ValueError` | Checked immediately after `extract_truth_hash` returns |
| `meta__dataset` column absent or all-empty in Silver dataframe | `ValueError` | Checked after extracting `SILVER_DATASET_NAME` from `meta__dataset` values |
| Parent truth record not loadable from disk | `ValueError` inside `load_parent_truth_record_from_dataframe` | Hash confirmed non-None before parent truth load |
| `SILVER_EDA_CFG` absent | `NameError` | Silver EDA–specific sanity check (Cell 11) |
| `FEATURE_REGISTRY_PATH` is `None` from parent truth | `ValueError` before any load attempt (Cell 36) | Explicit None check |
| Feature registry is `None` or not a `dict` | `ValueError` or `TypeError` | Type checks after `load_json` |
| `feature_columns` list empty in registry | `ValueError` | Empty list check |
| All feature columns absent from Silver dataframe | `ValueError` after intersection | Post-intersection empty check |
| No state column found among candidates | `KeyError` with candidate list | Priority detection loop with raise on `None` result |
| No sensor columns resolved | `ValueError` | Checked after both registry and `sensor_*` fallback paths |
| Multiple Silver Parquet files in glob fallback | `logger.warning`; first file used | Warning emitted; execution continues |
| Episodes too short after trimming | Silently dropped from windowing | No error; windowing simply produces no `__window_id` rows for short episodes |
| No normal rows survive trimming | `window_sensor_stats_df` is empty; `final_sensor_baseline_df` is empty; downstream scoring produces no results | Not a hard failure; subsequent Parquet outputs may be empty |
| `WRITE_TO_POSTGRES = False` | SQL write skipped; print statement confirms | Boolean gate in Cell 113; all file artifacts unaffected |

## Verification Checklist

- Active notebook path is `notebooks/eda/EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3.ipynb`
- Silver Pre-EDA Parquet exists at `SILVER_TRAIN_DATA_PATH / SILVER_TRAIN_DATA_FILE_NAME`
- `meta__truth_hash` column is present and non-null in Silver Parquet
- Silver 01 parent truth JSON exists in `TRUTHS_PATH/silver/`
- `FEATURE_REGISTRY_PATH` is non-None after Cell 34 executes
- Feature registry JSON exists and loads as a non-empty `dict`
- `SENSOR_COLUMNS` is non-empty after Cell 43 executes
- `machine_status__synthetic` column is present and fully lowercase in `silver_eda_df` after normalization
- `normal_profile_df` contains a non-zero count of rows with `__keep_after_episode_trim == True`
- `window_sensor_stats_df` is non-empty (at least one episode produced windows above `MIN_WINDOW_ROWS`)
- `final_sensor_baseline_df` has one row per sensor in `SENSOR_COLUMNS`
- `silver_sensor_baseline_profiles.csv` and `silver_sensor_baseline_profiles.json` exist in `SENSOR_PROFILE_DIR`
- `scored_normal_quality_df` contains `normal_training_quality_class` column
- `silver_subset_df` contains `machine_status__profiled` column with only: `normal_clean`, `normal_suspect`, `normal_contaminated`, `abnormal`, `recovery`
- `{dataset}__silver_subsets__profiled_dataframe.parquet` exists in `SILVER_TRAIN_DATA_PATH`
- `{dataset}__silver_subsets__normal_clean.parquet` exists in `SILVER_TRAIN_DATA_PATH`
- `{dataset}__silver_subsets__normal_contaminated.parquet` exists in `SILVER_TRAIN_DATA_PATH`
- `{dataset}__silver_subsets__summary.json` exists in `SILVER_EDA_SUMMARY_DIR`
- Silver EDA Subsets truth JSON exists in `TRUTHS_PATH/silver/` with a hash distinct from Silver 01's truth hash
- `SILVER_EDA_SUBSETS_TRUTH_HASH` appears in ledger `"save_subset_outputs"` step
- If `WRITE_TO_POSTGRES = True`: `log_silver_eda_sql` completed without error
- Ledger contains `"load_silver"`, `"normalize_source_state"`, `"save_sensor_baseline_profiles"`, `"save_subset_outputs"` steps

## Source-Limited Items

- Whether Silver 02b or Gold 01 consume the profiled Parquet artifacts via path constants or via Silver 02a's truth record `artifact_paths` section is not determined from Silver 02a source alone.
- The exact `SILVER_EDA_CFG` YAML key structure (field names for all threshold constants) is not confirmed from source cells visible; Cell 43 shows hardcoded defaults (`TRIM_FRAC = 0.10`, etc.) suggesting the notebook does not strictly require these from config.
- Whether the ledger JSON is explicitly written to a file path in Silver 02a source is not confirmed from visible cells; no `ledger.write_json(...)` call appears in the cell listing above Cell 113.
- The exact SQL table columns and row structure written by `log_silver_eda_sql` to `capstone.pipeline_runs`, `capstone.data_quality_events`, and `capstone.pipeline_artifacts` is not determined from Silver 02a source cells.
- Whether per-sensor profile PNGs are named consistently (e.g., `sensor_00.png`) or use a different naming convention is Not determined from available source beyond the function signature of `plot_all_sensor_profiles`.
- Whether the subset summary JSON is read by any downstream notebook or is an audit-only artifact is Not determined from available source.
