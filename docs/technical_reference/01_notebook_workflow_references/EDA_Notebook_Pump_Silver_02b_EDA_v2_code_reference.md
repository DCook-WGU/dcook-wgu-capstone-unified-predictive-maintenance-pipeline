# Notebook Code Reference: EDA_Notebook_Pump_Silver_02b_EDA_v2

**Source:** `notebooks/eda/EDA_Notebook_Pump_Silver_02b_EDA_v2.ipynb`
**Stage:** Silver EDA — Profiled State EDA and Generator Input Bundle
**Cells:** 139 (64 code, 75 markdown), 22 inventory sections

---

## Notebook Purpose

Silver 02b performs profiled-state EDA on the Silver dataset produced by Silver 02a and builds the artifact bundle consumed by the synthetic generator, Gold preprocessing, and the truth lineage chain.

Where Silver 02a creates the profiled state labels (`normal_clean`, `normal_contaminated`, `abnormal`, `recovery`), Silver 02b characterizes sensor behavior across those four states and produces structured, reusable outputs that downstream stages can consume without re-running the EDA. The three most critical contributions are:

**1. Generator input bundle:** Per-state feature profiles, correlation matrices, sensor grouping maps, dropped sensor registry, and episode status counts. These are the direct contract used by the synthetic generator to reproduce realistic fault and normal-behavior signals.

**2. Gold modeling context:** Correlation structure, PCA variance diagnostics, imputation strategy comparison, and IsolationForest outlier audit. These inform Gold preprocessing decisions before modeling begins.

**3. Silver EDA Profile truth record:** A third Silver-layer truth record (`silver_eda_profile_truth`) with `parent_truth_hash` pointing to Silver_01's hash, establishing a verifiable lineage link from the EDA artifact bundle back to the source Silver run. The truth record also carries the missingness quarantine payload forward from Silver_01's truth, giving Gold a complete missingness audit without re-reading Silver_01.

Silver 02b does not modify the profiled dataframe or its truth stamps. It reads, characterizes, and exports.

---

## Pipeline Role

| Attribute | Value |
|---|---|
| Stage | Silver EDA (`silver_eda`) |
| Position | Fourth in the active chain; downstream of Silver_02a |
| Upstream input | Silver_02a profiled dataframe (`*__silver_subsets__profiled_dataframe.parquet`) |
| Downstream outputs | Generator input bundle (synthetic generator); EDA artifact index (Gold); Silver EDA Profile truth record (lineage) |
| Truth chain | Reads Silver_01's `meta__truth_hash` as parent; creates `silver_eda_profile_truth` record (`truth_stage="eda_profile"`) |
| W&B | Disabled via `USE_EXPERIMENT_TRACKING = False` toggle; not triple-quoted — the flag is evaluated, just resolves to false |
| SQL write function | `write_silver_eda_sql_outputs` (not `log_silver_eda_sql`) — writes to 7 Silver EDA-specific tables |

---

## Inputs

| Input | Source | Form | Used For |
|---|---|---|---|
| Silver_02a profiled dataframe | `SILVER_TRAIN_DATA_PATH / {DATASET_NAME}__silver_subsets__profiled_dataframe.parquet` | Parquet (fallback: recursive glob) | All EDA, correlation, and generator input analysis |
| Feature registry JSON | Path resolved from parent Silver truth record | JSON dict | `FEATURE_COLUMNS`; sensor columns for all per-feature EDA |
| Dropped sensors Parquet | `DROPPED_SENSORS_PARQUET_PATH` from Silver_01 quarantine output | Parquet | Dropped feature profiles (per state) for generator inputs |
| Dropped sensor metadata | PreEDA feature registry (`drop_reasons`, `dropped_missing_pct`) | JSON / registry | `dropped_sensor_registry.json` for generator |
| Parent Silver truth record | `TRUTHS_PATH/silver/` via `load_parent_truth_record_from_dataframe` | JSON | `FEATURE_REGISTRY_PATH`; missingness quarantine payload forwarding |
| Project config | `load_notebook_context(stage="silver_eda", dataset="pump")` | YAML → `CTX` | All runtime constants, stage config, paths |
| Environment variables | OS environment | Strings | DB engine, `CAPSTONE_SCHEMA`, `DATASET_ID` / `RUN_ID` |

---

## Configuration and Runtime Context

| Item | Source | Purpose |
|---|---|---|
| `SILVER_EDA_CFG` | `CTX` | Silver EDA stage config block; required by Silver-EDA–specific sanity check |
| `DEFAULT_FALLBACKS_CFG` | `CTX` | Fallback values for `ASSET_ID` / `RUN_ID` resolution |
| `SILVER_PROCESS_RUN_ID` / `SILVER_SUBSETS_PROCESS_RUN_ID` | `make_process_run_id(...)` | Unique IDs for parent Silver run and EDA profile run; written into truth record |
| `USE_EXPERIMENT_TRACKING` | Notebook cell (bool gate, default `False`) | Controls W&B; when `False`, experiment tracking is skipped silently |
| `WRITE_SILVER_EDA_SQL_OUTPUTS` | QA section cell (bool gate, default `True`) | Controls `write_silver_eda_sql_outputs`; allows offline/dry runs |
| `SUBSYSTEM_CORR_THRESHOLD` | Notebook cell | Correlation threshold for sensor subsystem grouping via connected components (default 0.80) |
| `FEATURE_CLUSTER_COUNT` | Notebook cell | Number of clusters for `AgglomerativeClustering` on the correlation matrix |
| `PCA_SAMPLE_N` / `PCA_N_COMPONENTS` / `PCA_SCALER` | Notebook cell | PCA sampling and scaler configuration (default: 20000 rows, 2 components, `"robust"`) |
| `OUTLIER_AUDIT_SAMPLE_N` / `OUTLIER_CONTAMINATION` | Notebook cell | `IsolationForest` sample cap and contamination rate for outlier audit |
| `IMPUTE_COMPARE_SAMPLE_N` | Notebook cell | Sample size for imputation strategy comparison |
| `TOP_HEATMAP_SENSOR_COUNT` / `TOP_DISTRIBUTION_SENSOR_COUNT` / `TOP_TIMELINE_SENSOR_COUNT` | Notebook cells | Number of top shifted sensors to include in heatmap, distribution, and timeline plots |
| `ALIGN_PRE_STEPS` / `ALIGN_POST_STEPS` | Notebook cells | Window size for aligned anomaly-onset plots (steps before and after onset) |
| `DATASET_NAME` / `RUN_ID` / `ASSET_ID` / `DATASET_ID` | Resolved from env → config → fallback | Identity fields written into SQL rows and truth record |
| `PIPELINE_MODE` / `RUN_MODE` | `PIPELINE` via `CTX` | Propagated into truth record |
| `EDA_NOTEBOOK_NAME` / `EDA_DATAFRAME_NAME` | Config/cell | Written into SQL rows for provenance |

---

## Logical Workflow Map

1. Load shared context via `load_notebook_context(stage="silver_eda")`; run sanity checks; `ledger.add` checkpoint
2. Create Silver EDA artifact directory tree; export config snapshot
3. Establish SQL engine; SQL smoke check; resolve `DATASET_ID` / `RUN_ID` / `ASSET_ID`
4. Start logging (`log_layer_paths`); `ledger.add`
5. Set `USE_EXPERIMENT_TRACKING = False` (W&B disabled; `ledger.add` checkpoint instead)
6. Load Silver_02a profiled Parquet via `load_data`; fallback: recursive glob for `*__silver_subsets__profiled_dataframe.parquet`
7. Extract `SILVER_TRUTH_HASH` **before any filtering**; load parent Silver truth record; confirm `DATASET_NAME`; resolve `FEATURE_REGISTRY_PATH`
8. Load feature registry JSON; validate and resolve `FEATURE_COLUMNS`
9. Define robust scaling helpers; define profiled state constants and masks (`mask_profiled_normal_clean`, `mask_profiled_normal_contaminated`, `mask_profiled_abnormal`, `mask_profiled_recovery`)
10. Build profiled data overview summary (row counts per state, feature/meta column counts); log
11. Compute full-dataset missingness; `ledger.add`
12. Check duplicate candidate keys (full rows + key combinations)
13. Compute numeric feature summaries (describe, skew, kurtosis per feature)
14. Display source state counts (`machine_status__synthetic` distribution)
15. Build `PROFILED_STATE_SUBSETS` dict — copy of each profiled state; profiled subset summary displayed
16. Build dropped/weak sensor review (sort by missing_pct, flag all-null and near-constant)
17. Build episode-by-source-state and episode-by-profiled-state summary tables; `ledger.add` each
18. Build per-state missingness tables — source state missingness + profiled state missingness (MISSINGNESS_REPLAY)
19. Build state transition and dwell tables via `build_state_transition_tables` (source and profiled)
20. Build correlation artifacts:
    - Normal-clean correlation matrix → CSV + long-form pair table → CSV
    - Normal-contaminated correlation matrix → CSV (if rows > 1)
    - Abnormal correlation matrix → CSV; fault-pairing table (correlation delta vs. normal-clean) → CSV
    - `ledger.add`
21. Build sensor grouping from normal-clean correlation structure:
    - Connected-components adjacency graph at `SUBSYSTEM_CORR_THRESHOLD` → `sensor_group_map_normal_clean.csv`
    - `AgglomerativeClustering` on correlation distance matrix → `feature_cluster_map.csv`
    - `ledger.add`
22. Build generator input artifacts:
    - Load dropped sensor metadata from PreEDA registry; build `dropped_sensor_registry.json`
    - `build_rich_feature_profile` for each state (normal_clean, abnormal, recovery) → per-state CSV profiles
    - `build_rich_feature_profile` for dropped sensors per state → dropped feature profile CSVs
    - Build sensor hotspot summary via `build_state_median_lookup` + `robust_state_comparison_vs_clean`
    - Save hotspot, transition, and dwell artifacts → CSV; `ledger.add`
23. Summarize each profiled state (`summarize_profiled_state`); compare all states against normal-clean via `robust_state_comparison_vs_clean` → `robust_state_compare_top_df` (ranked by shift magnitude)
24. Build top-sensor heatmap (seaborn; `plt.show`)
25. Build per-state distribution plots for top shifted sensors → PNGs in `DISTRIBUTION_PLOT_DIR`; `ledger.add`
26. Build timeline sensor overlay plots for top shifted sensors → PNGs in `TIMELINE_OVERLAY_DIR`; `ledger.add`
27. Build aligned anomaly-onset windows (`ALIGN_PRE_STEPS` / `ALIGN_POST_STEPS` around onset boundaries) → `aligned_onset_df`; plot aligned median curves → PNGs in `ALIGNED_ONSET_PLOT_DIR`; `ledger.add`
28. Run agglomerative clustering validation guardrails; feature cluster map; `ledger.add`
29. Run PCA diagnostics (2-component, median-imputed, `RobustScaler` or `StandardScaler`); plot scatter colored by profiled state; save explained variance + loadings → CSV; `ledger.add`
30. Run imputation comparison (median impute vs. forward-fill+backfill on sample); save comparison → CSV
31. Run IsolationForest outlier audit (trained on `normal_clean` rows only; scored on all rows); per-state outlier summary → CSV
32. Export episode status counts (per-episode profiled state distribution) → `episode_status_counts.json`
33. Update generator input manifest (`generator_input_manifest.json`) with all artifact paths; validate all paths exist; `ledger.add`
34. **Finalization (cell 118):** Build `silver_eda_profile_summary` JSON; initialize `silver_eda_profile_truth` with `parent_truth_hash=SILVER_TRUTH_HASH`; carry missingness quarantine payload forward from Silver_01 truth; populate config_snapshot / runtime_facts / artifact_paths / notes sections; `build_truth_record`; `save_truth_record`; `append_truth_index`; `ledger.add`
35. Build EDA artifact index dataframe from all generator manifest paths; `ledger.add`
36. **QA section:** Verify Silver truth dir and truth index exist; build SQL staging tables (profile, feature_statistics, missingness_summary, correlation_pairs, outlier_summary, categorical_distribution)
37. `write_silver_eda_sql_outputs` (gated by `WRITE_SILVER_EDA_SQL_OUTPUTS`) → 7 Silver EDA PostgreSQL tables; `ledger.add`

---

## Section Overview

| Section | Key Outputs / Side Effects |
|---|---|
| Imports and environment setup | All imports; `cfg_require_mapping`, `require_dict`, `require_list` helpers defined |
| Context load and configuration | `CTX`, `SILVER_EDA_CFG`, `DEFAULT_FALLBACKS_CFG`, all pipeline constants, `logger`, `ledger`; sanity checks |
| Silver EDA artifact directories | Directory tree created; `export_config_snapshot`; `CONFIG_SNAPSHOT_PATH` |
| SQL Runtime Context | `engine`; `DATASET_ID`, `RUN_ID`, `ASSET_ID` resolved |
| SQL smoke check | `sql_smoke_check_dataframe` displayed; DB connectivity confirmed |
| Logging, experiment tracking, ledger | `log_layer_paths`; `USE_EXPERIMENT_TRACKING = False`; ledger checkpoint `ledger.add` |
| Load profiled Silver dataframe | `silver_eda_dataframe` (Silver_02a profiled Parquet); `ledger.add` |
| Resolve parent truth and dataset identity | `SILVER_TRUTH_HASH` (before filtering); `silver_truth`; `DATASET_NAME`; `FEATURE_REGISTRY_PATH` |
| Load feature registry | `feature_registry`; `FEATURE_COLUMNS`; validation of column presence |
| Robust scaling helpers and state constants | `robust_center_scale` helper; profiled-state masks; `PROFILED_STATE_SUBSETS` dict |
| Build profiled data overview | `overview_summary_df` with per-state row counts; log |
| Evaluate overall missingness | Full-dataset `missingness_df`; sorted by missing_pct; `ledger.add` |
| Check duplicate candidate keys | Per-key duplicate counts; displayed |
| Build numeric feature summaries | `numeric_profile_df` (describe + skew + kurtosis per feature column) |
| Source and profiled state counts | Source state value counts; profiled state value counts and percentages displayed |
| Define state-level sensor profile builder | `build_rich_feature_profile` helper defined |
| Review dropped/weak sensor behavior | `dropped_sensor_review_df` sorted by missing_pct; all-null / near-constant flags; `ledger.add` |
| Summarize states by episode | `episode_source_counts_df`; `episode_profiled_counts_df`; `ledger.add` each |
| Build missingness tables by state | Source-state missingness per feature; profiled-state missingness per feature (MISSINGNESS_REPLAY) |
| Build state transition and dwell tables | `source_transition_counts_df`, `profiled_transition_counts_df`, dwell summaries |
| Build normal-clean correlation artifacts | `correlation_matrix_normal_clean`; `clean_corr_pairs`; contaminated and abnormal correlation matrices; fault pairings; all CSV; `ledger.add` |
| Build sensor grouping from correlation | Connected-components sensor groups at 0.80 threshold; `AgglomerativeClustering` feature clusters; CSVs; `ledger.add` |
| Build generator input artifacts | Dropped sensor registry JSON; per-state feature profiles (CSVs); `build_state_median_lookup`; `ledger.add` |
| Save hotspot / transition / dwell artifacts | Hotspot sensor summary; transition counts; dwell summaries → CSVs; `ledger.add` |
| Summarize and compare profiled states | Per-state summary dicts; `robust_state_compare_top_df`; top shifted sensors ranked |
| Top-sensor heatmap | Seaborn heatmap for top shifted sensors vs. normal-clean correlation (inline display) |
| Distribution plots | Per-state distribution PNG per top shifted sensor; saved to `DISTRIBUTION_PLOT_DIR`; `ledger.add` |
| Timeline overlays | Timeline PNG per top shifted sensor; saved to `TIMELINE_OVERLAY_DIR`; `ledger.add` |
| Aligned anomaly-onset windows | `aligned_onset_df`; aligned median curve PNGs in `ALIGNED_ONSET_PLOT_DIR`; `ledger.add` |
| Agglomerative clustering validation | `feature_cluster_map.csv`; `ledger.add` |
| PCA diagnostics | Explained variance + loadings CSVs; scatter plot; `ledger.add` |
| Imputation comparison | Median vs. forward-fill comparison on sample; comparison CSV |
| Outlier audit by profiled state | `IsolationForest` trained on `normal_clean`; per-state outlier summary CSV |
| Export episode status counts | `episode_status_counts.json` → `GENERATOR_INPUT_DIR`; `ledger.add` |
| Update generator input manifest | `generator_input_manifest.json` consolidated; all paths validated; `ledger.add` |
| Build truth record and save final outputs | `silver_eda_profile_summary` JSON; `silver_eda_profile_truth` initialized + populated; `save_truth_record`; `append_truth_index`; `ledger.add` |
| Build EDA artifact index | `eda_artifact_index_df` over all manifest paths; `ledger.add` |
| QA / SQL staging | Truth dir / truth index verified; SQL staging tables built |
| SQL write gate | `write_silver_eda_sql_outputs` → 7 Silver EDA PostgreSQL tables; `WRITE_SILVER_EDA_SQL_OUTPUTS` gate |

---

## Section Details

### Context Load and Sanity Checks

`load_notebook_context(stage="silver_eda", dataset="pump", mode="train", profile="default")` shares the same stage string as Silver_02a. Both notebooks use `SILVER_EDA_CFG` and `DEFAULT_FALLBACKS_CFG`. Both produce independent truth records under `truth_stage="eda_subsets"` (Silver_02a) and `truth_stage="eda_profile"` (Silver_02b).

A context sanity check raises `NameError` listing missing variables if any required shared context name is absent. The Silver-EDA–specific check verifies `SILVER_EDA_CFG`.

The original standalone `Ledger(...)` initialization is preserved in a triple-quoted string (cell 30). It is not executed; `load_notebook_context` initializes the ledger. The cell serves as migration documentation.

**Experiment tracking:** `USE_EXPERIMENT_TRACKING = False` is set in cell 27 via a conditional block. Unlike Silver_02a where `wandb.init` is triple-quoted, Silver_02b uses an explicit boolean toggle. The `if USE_EXPERIMENT_TRACKING: ...` branch prints a confirmation but takes no action when false. No W&B run is started.

---

### Load Profiled Silver Dataframe and Parent Truth Resolution

`load_data(...)` reads the Silver_02a profiled Parquet from `SILVER_TRAIN_DATA_PATH`. The primary path is `{SILVER_TRAIN_DATA_PATH}/{DATASET_NAME}__silver_subsets__profiled_dataframe.parquet`. If that exact file does not exist, the notebook globs recursively through `SILVER_EDA_OUTPUT_DIR` and `SILVER_EDA_ARTIFACT_DIR` for `*__silver_subsets__profiled_dataframe.parquet`.

`SILVER_TRUTH_HASH = extract_truth_hash(silver_eda_dataframe)` runs **immediately after load, before any subsetting or filtering**. This is the same pattern as Silver_02a — the hash must reflect the full unmodified Silver output to ensure the truth lineage link is valid.

`load_parent_truth_record_from_dataframe` loads the Silver_01 truth JSON to retrieve `FEATURE_REGISTRY_PATH`. The feature registry is then loaded, validated as a non-None dict, and validated to contain a non-empty `feature_columns` list. Missing feature columns (present in registry but absent from the dataframe) emit `logger.warning`.

---

### Profiled State Masks and Subsets

Four boolean masks are built from `silver_eda_dataframe[STATE_COL_PROFILED]`:
- `mask_profiled_normal_clean`
- `mask_profiled_normal_contaminated`
- `mask_profiled_abnormal`
- `mask_profiled_recovery`

`PROFILED_STATE_SUBSETS` is a dict of dataframe copies per profiled state. These copies protect `silver_eda_dataframe` from mutation during state-level analysis. All per-state EDA operations (correlation, profiling, comparison) work from these copies or from direct mask indexing on `silver_eda_dataframe`.

---

### Generator Input Artifacts

This is Silver_02b's most consequential output group. The synthetic generator reads these files directly; changes to their schema or content require corresponding changes to the generator.

**Per-state feature profiles (via `build_rich_feature_profile`):** For each of `normal_clean`, `abnormal`, and `recovery`, the notebook builds a per-sensor profile that includes median, IQR, min/max, distribution family classification (`infer_distribution_family`), missingness rate, and delta statistics. Recovery is included because the synthetic generator needs to reproduce post-fault behavior in addition to normal and fault states.

**Dropped sensor registry:** Loads dropped sensor metadata from the PreEDA feature registry (via `drop_reasons` and `dropped_missing_pct` fields). Cross-references with the dropped sensors Parquet to build per-state feature profiles for quarantined sensors. These tell the generator how to handle sensors that were excluded from modeling: some should be generated then masked (e.g., `sensor_50`), while others should not be generated at all (e.g., an all-null sensor).

**Generator input manifest:** After all artifacts are written, `generator_input_manifest` is populated (or reloaded from disk) and updated with all artifact paths. Cell 116 validates that every path in the manifest exists on disk before marking the manifest complete. The manifest is the single index the generator reads to locate all input files.

---

### Correlation Artifacts and Sensor Grouping

**Correlation matrices:** Computed for `normal_clean`, `normal_contaminated`, and `abnormal` subsets using `df.corr()`. Upper-triangle pairs are extracted as long-form tables for import into SQL and generator use. A fault-pairing table is built by merging the normal-clean and abnormal pair tables to identify sensors whose pairwise correlation changes substantially between normal and fault states.

**Sensor grouping — two methods:**

1. **Connected components (cell 74):** Builds an adjacency graph from the normal-clean absolute correlation matrix at `SUBSYSTEM_CORR_THRESHOLD` (default 0.80). Sensors sharing a correlation above the threshold are added to each other's adjacency set. Connected components are extracted with a visited-set DFS. Each sensor is assigned a `subsystem_group` integer. Output: `sensor_group_map_normal_clean.csv`.

2. **Agglomerative clustering (cell 100-101):** Converts the normal-clean correlation matrix to a distance matrix (`1 - abs_corr`) and applies `AgglomerativeClustering(n_clusters=FEATURE_CLUSTER_COUNT)`. Output: `feature_cluster_map.csv`. This provides a second, orthogonal grouping that does not depend on the 0.80 threshold.

Both groupings are surfaced in the generator manifest and the truth record's `artifact_paths`.

---

### Profiled State Comparison Against Normal-Clean

`robust_state_comparison_vs_clean` computes, for each feature column and each profiled state:
- `clean_center`: median of the feature in `normal_clean` rows
- `clean_scale`: robust IQR-based scale
- `state_center`: median of the feature in that state
- Normalized shift: `(state_center - clean_center) / clean_scale`
- State-level 5th / 95th percentile values

The output (`robust_state_compare_top_df`) ranks features by the magnitude of their shift from `normal_clean` across any state. The top N sensors from this ranking drive the heatmap, distribution plots, timeline overlays, and aligned onset plots.

---

### PCA Diagnostics

PCA runs on up to `PCA_SAMPLE_N` (default 20000) rows sampled from `silver_eda_dataframe[FEATURE_COLUMNS]`. Missing values are median-imputed before scaling. `RobustScaler` is used by default (configurable via `PCA_SCALER`). A 2-component `sklearn.decomposition.PCA` is fitted.

Outputs:
- Scatter plot of PC1 vs PC2 colored by `STATE_COL_PROFILED` (inline display; not saved as PNG)
- `pca_explained_variance.csv` — one row per component with explained variance ratio and cumulative ratio
- `pca_loading_path.csv` — feature loadings on each PC, sorted by PC1 absolute loading

Purpose: diagnostic only. Confirms whether the profiled states separate visually in a reduced space. Not used as a modeling step.

---

### IsolationForest Outlier Audit

`IsolationForest(contamination=OUTLIER_CONTAMINATION)` is trained exclusively on `normal_clean` rows (up to `OUTLIER_AUDIT_SAMPLE_N`, default 20000). The fitted model scores all rows in `silver_eda_dataframe`. Results are grouped by `STATE_COL_PROFILED` to compare outlier rates per state.

This is the source of the `MODEL_TRAINING` and `MODEL_EVALUATION` decision tags in the inventory. Unlike Silver_01 (where those tags reflected import-only presence), this notebook runs a genuine Isolation Forest fit. The output is diagnostic — it informs Gold's contamination parameter choice but does not become a production artifact.

---

### Finalization — Truth Record and Summary Export

This is the largest single code cell in the notebook. Operations in order:

1. Build `silver_eda_profile_summary` dict — profiled state counts, all artifact paths consolidated
2. Write summary JSON to `SILVER_EDA_SUMMARY_DIR / {DATASET_NAME}__silver__eda_profile__summary.json`
3. `initialize_layer_truth(layer_name="silver_eda", parent_truth_hash=SILVER_TRUTH_HASH, ...)` — creates `silver_eda_profile_truth`
4. `update_truth_section(... "config_snapshot" ...)` — stage, layer, pipeline mode, run mode
5. `update_truth_section(... "runtime_facts" ...)` — parent hash, profiled state counts per mask, source dataframe path
6. **Missingness quarantine payload forwarding:** reads `silver_truth.get("runtime_facts", {}).get("missingness_quarantine")` from the Silver_01 parent truth record and appends it to Silver_02b's `runtime_facts`. This is the mechanism by which Gold can read the quarantine audit without accessing Silver_01's truth record directly.
7. `update_truth_section(... "artifact_paths" ...)` — extensive list: all generator-input profiles, dropped sensor registry, correlation artifacts, PCA/imputation/outlier artifact paths, episode status counts, feature registry passthrough
8. `update_truth_section(... "notes" ...)` — human-readable purpose statement
9. `build_truth_record(...)` → `SILVER_EDA_PROFILE_TRUTH_HASH`
10. `silver_eda_profile_truth["truth_stage"] = "eda_profile"` and `["notebook_name"] = "silver_02b_eda_profile"` added directly (not via `update_truth_section`)
11. `save_truth_record(...)` → JSON at `TRUTHS_PATH/silver/{DATASET_NAME}__silver__truth__{hash}.json`
12. `append_truth_index(...)` → truth index updated
13. `ledger.add(step="build_silver_eda_profile_truth_record", ...)` — records both the new hash and the parent Silver hash

---

### QA Section and SQL Write

The QA section (cells 122–137) runs after the truth record is saved and before the SQL write. It:
1. Verifies the Silver truth directory contains at least one truth JSON
2. Verifies the truth index (`TRUTH_INDEX_PATH`) exists and is non-empty
3. Reads the truth index into `truth_index_df` and confirms required columns (`layer_name`, `truth_stage`, `truth_path`, `truth_hash`)
4. Builds SQL staging dataframes in memory: `profile_df`, `feature_statistics_df`, `missingness_summary_df`, `correlation_pairs_df`, `outlier_summary_df`, `categorical_distribution_df`

`write_silver_eda_sql_outputs(engine, dataset_id, run_id, notebook_name, profile_df, ...)` (cell 137) writes to seven tables:

| Table | Content |
|---|---|
| `silver.eda_dataset_profile` | Row/column counts, memory usage, duplicate counts, column-type counts |
| `silver.eda_feature_statistics` | Per-feature describe output (percentiles, mean, std, skew, kurtosis) |
| `silver.eda_missingness_summary` | Per-feature null count, non-null count, null percentage |
| `silver.eda_correlation_pairs` | All pairwise Pearson correlations (full matrix, not upper-triangle only) |
| `silver.eda_outlier_summary` | Per-feature IQR-based outlier count and rate |
| `silver.eda_categorical_distribution` | Value counts for all categorical/boolean columns |
| `silver.eda_artifact_index` | One row per artifact in the EDA artifact index dataframe |

`WRITE_SILVER_EDA_SQL_OUTPUTS = True` is the default. Setting it to `False` skips all database writes; all file artifacts are unaffected.

**Note:** Silver_02b uses `write_silver_eda_sql_outputs`, not `log_silver_eda_sql` (which Silver_01 and Silver_02a use). These are different functions writing to different table schemas.

---

## Key Function Calls and In-Place Usage

| Function | Section | Return / Side Effect |
|---|---|---|
| `load_notebook_context(stage="silver_eda", ...)` | Context load | `CTX`; `SILVER_EDA_CFG`, `logger`, `ledger` |
| `export_config_snapshot(...)` | Artifact dirs | Config JSON in `SILVER_EDA_CONFIG_DIR` |
| `get_engine_from_env()` | SQL Runtime Context | `engine` |
| `read_sql_dataframe(engine, SELECT ...)` | SQL smoke check | `sql_smoke_check_dataframe` |
| `log_layer_paths(paths, logger)` | Logging setup | Layer path log written |
| `load_data(profiled_path.parent, file_name)` | Load profiled Silver | `silver_eda_dataframe` |
| `extract_truth_hash(silver_eda_dataframe)` | Resolve parent truth | `SILVER_TRUTH_HASH` (before any filtering) |
| `load_parent_truth_record_from_dataframe(...)` | Resolve parent truth | `silver_truth`; `FEATURE_REGISTRY_PATH` |
| `load_json(FEATURE_REGISTRY_PATH.parent, name)` | Load feature registry | `feature_registry_raw` |
| `build_rich_feature_profile(df, sensor_cols, ...)` | Generator inputs | Per-state feature profile dict (per sensor: median, IQR, distribution family, missingness) |
| `infer_distribution_family(std, iqr, skewness, ...)` | Inside `build_rich_feature_profile` | Distribution family string (`"near_constant"`, `"normal"`, `"skewed"`, etc.) |
| `build_state_transition_tables(df, state_col, ...)` | State transitions | `(transition_counts_df, dwell_summary_df)` |
| `build_state_median_lookup(df, sensor_cols, state_col)` | Generator inputs | Per-state median per sensor |
| `robust_state_comparison_vs_clean(df, feature_cols)` | State comparison | `robust_state_compare_top_df` ranked by shift magnitude |
| `robust_center_scale(series)` | Inside state comparison | `(center, scale)` using median and IQR |
| `AgglomerativeClustering(n_clusters=...)` | Feature clustering | `cluster_labels` array → `feature_cluster_map.csv` |
| `PCA(n_components=PCA_N_COMPONENTS)` | PCA diagnostics | `pca_model`; explained variance; loadings |
| `RobustScaler()` or `StandardScaler()` | PCA diagnostics | Scaled feature matrix `X_scaled` |
| `IsolationForest(contamination=...)` | Outlier audit | Trained on `normal_clean`; scores all rows |
| `initialize_layer_truth(layer_name="silver_eda", parent_truth_hash=SILVER_TRUTH_HASH, ...)` | Finalization | `silver_eda_profile_truth` dict |
| `update_truth_section(...)` × 5 | Finalization | config_snapshot, runtime_facts (×2 incl. missingness passthrough), artifact_paths, notes |
| `build_truth_record(...)` | Finalization | `SILVER_EDA_PROFILE_TRUTH_HASH` |
| `save_truth_record(...)` | Finalization | JSON at `TRUTHS_PATH/silver/` |
| `append_truth_index(...)` | Finalization | Truth index updated |
| `write_silver_eda_sql_outputs(engine, ..., profile_df, ...)` | SQL write | Rows in 7 `silver.eda_*` PostgreSQL tables |

---

## Outputs and Artifacts

| Output | Type | Location | Downstream Consumer |
|---|---|---|---|
| Feature profile — normal_clean | CSV / JSON | `GENERATOR_INPUT_DIR` | Synthetic generator |
| Feature profile — abnormal | CSV / JSON | `GENERATOR_INPUT_DIR` | Synthetic generator |
| Feature profile — recovery | CSV / JSON | `GENERATOR_INPUT_DIR` | Synthetic generator |
| Dropped sensor registry | JSON | `GENERATOR_INPUT_DIR / dropped_sensor_registry.json` | Synthetic generator |
| Dropped feature profiles (per state) | CSV | `GENERATOR_INPUT_DIR` | Synthetic generator |
| Episode status counts | JSON | `GENERATOR_INPUT_DIR / episode_status_counts.json` | Synthetic generator |
| Generator input manifest | JSON | `GENERATOR_INPUT_DIR / generator_input_manifest.json` | Synthetic generator (single index) |
| Normal-clean correlation matrix | CSV | `CORRELATION_ARTIFACT_DIR` | Gold; generator; EDA reference |
| Normal-clean correlation pairs | CSV | `CORRELATION_ARTIFACT_DIR` | Gold; generator |
| Normal-contaminated correlation matrix | CSV | `CORRELATION_ARTIFACT_DIR` | EDA reference |
| Abnormal correlation matrix + fault pairings | CSV | `CORRELATION_ARTIFACT_DIR` | Gold; EDA reference |
| Sensor group map (connected components) | CSV | `CORRELATION_ARTIFACT_DIR / sensor_group_map_normal_clean.csv` | Generator; Gold |
| Feature cluster map (agglomerative) | CSV | `CORRELATION_ARTIFACT_DIR / feature_cluster_map.csv` | Generator; Gold |
| Hotspot sensor summary | CSV | `CORRELATION_ARTIFACT_DIR / sensor_hotspot_summary_profiled_states.csv` | EDA reference |
| State transition tables (source + profiled) | CSV | `CORRELATION_ARTIFACT_DIR` | EDA reference |
| State dwell summaries (source + profiled) | CSV | `CORRELATION_ARTIFACT_DIR` | EDA reference |
| Distribution plots (per top sensor, per state) | PNG | `DISTRIBUTION_PLOT_DIR` | Visual review |
| Timeline overlay plots (per top sensor) | PNG | `TIMELINE_OVERLAY_DIR` | Visual review |
| Aligned onset plots (per sensor) | PNG | `ALIGNED_ONSET_PLOT_DIR` | Visual review |
| PCA explained variance | CSV | `PCA_ARTIFACT_DIR / pca_explained_variance.csv` | EDA reference |
| PCA loadings | CSV | `PCA_ARTIFACT_DIR / pca_loading_path.csv` | EDA reference |
| Imputation comparison | CSV | `SILVER_EDA_ARTIFACT_DIR` | Gold preprocessing guidance |
| Outlier summary | CSV | `SILVER_EDA_ARTIFACT_DIR` | Gold contamination parameter guidance |
| Silver EDA profile summary | JSON | `SILVER_EDA_SUMMARY_DIR / {DATASET_NAME}__silver__eda_profile__summary.json` | Truth record artifact index; lineage |
| Silver EDA Profile truth record | JSON | `TRUTHS_PATH/silver/{DATASET_NAME}__silver__truth__{hash}.json` | Gold truth chain; lineage |
| Truth index entry | Appended JSONL | `TRUTH_INDEX_PATH` | Cross-run lineage lookup |
| Config snapshot | JSON | `SILVER_EDA_CONFIG_DIR` | Reproducibility / audit |
| SQL EDA rows (7 tables) | PostgreSQL rows | `silver.eda_dataset_profile`, `silver.eda_feature_statistics`, `silver.eda_missingness_summary`, `silver.eda_correlation_pairs`, `silver.eda_outlier_summary`, `silver.eda_categorical_distribution`, `silver.eda_artifact_index` | Operational monitoring |
| Ledger entries | Via `ledger.add` | Audit trail | Audit |

---

## Data Quality / Validation Behavior

| Check | Where | Failure / Risk Prevented |
|---|---|---|
| General + Silver-EDA context sanity check | After `load_notebook_context` | `NameError` if any required variable is missing |
| SQL smoke check | Before data load | Catches DB unavailability before processing |
| Profiled dataframe found | `load_data` with recursive fallback | `FileNotFoundError` if no profiled Parquet anywhere in fallback dirs |
| `SILVER_TRUTH_HASH` not None | Immediately after `load_data` | `ValueError` if profiled dataframe has no truth stamp |
| Feature registry non-None dict | After `load_json` | `ValueError` / `TypeError` if registry is missing or malformed |
| `FEATURE_COLUMNS` non-empty | After registry load | `ValueError` if feature registry contains no feature columns |
| Missing feature columns logged | After registry validation | `logger.warning` for features in registry but absent from dataframe |
| `FEATURE_REGISTRY_PATH` not None | Before `load_json` | `NameError` if parent truth did not resolve a registry path |
| `meta__episode_id` column present | Episode-level summaries | `KeyError` if episode column is absent; fallback creates global episode 0 for generator export |
| Correlation computed only when rows > 1 | Contaminated/abnormal correlation | Skips `.corr()` on empty or single-row subsets |
| Generator manifest path validation | After manifest update | Logs `missing_manifest_paths` for any artifact path that does not exist on disk |
| Truth dir non-empty | QA section | `FileNotFoundError` if Silver truth dir has no JSON files |
| Truth index exists and non-empty | QA section | `FileNotFoundError` if index is missing or zero bytes |
| Truth index has required columns | QA section | `KeyError` if `layer_name`, `truth_stage`, `truth_path`, `truth_hash` are absent |

---

## Downstream Handoff

**Synthetic generator** reads the generator input manifest (`generator_input_manifest.json`) as its single entry point and loads all referenced per-state feature profiles, dropped sensor registry, dropped feature profiles, episode status counts, sensor group map, and correlation pair tables. This is the direct contract: artifact schema changes in Silver_02b require matching changes in the generator.

**Gold_01_PreProcessing** uses:
- The Silver EDA Profile truth record (`SILVER_EDA_PROFILE_TRUTH_HASH`) as a navigation key to locate all EDA artifact paths via the `artifact_paths` section of the truth JSON
- The missingness quarantine payload (forwarded from Silver_01 into the `runtime_facts` section of Silver_02b's truth record) to recover the full quarantine audit without reading Silver_01's truth record directly
- The correlation matrix and sensor grouping artifacts to inform feature selection and preprocessing decisions
- The PCA, imputation comparison, and outlier audit CSVs as Gold preprocessing guidance

The truth index entry written by `append_truth_index` allows any downstream stage to find the Silver EDA Profile truth record by `truth_stage="eda_profile"` and `layer_name="silver_eda"` without knowing the hash in advance.

---

## Relationship to Other Notebooks

### Upstream Context

Silver_02b reads Silver_02a's clean analytical subset Parquets. It applies deeper EDA including distribution analysis, outlier audit (via `IsolationForest.fit_predict` on the normal_clean subset — a diagnostic computation, not a production model), PCA, and aligned onset analysis. No dependency on Silver_01 directly, Gold notebooks, or cascade notebooks.

### Downstream Handoff

Silver_02b's outputs are EDA analysis artifacts (CSVs, JSONs, distribution summaries, outlier audit results). No direct downstream pipeline notebook is confirmed from available source. A generator input manifest is produced but its downstream consumer is not confirmed from the 071b reference.

### Pipeline Position

Final Silver EDA notebook. Provides analytical context for capstone submission and EDA documentation. Its relationship to Gold_01 is contextual (informing feature selection decisions) rather than confirmed as a direct file-level pipeline input.

### Relationship Summary

- Reads Silver_02a clean subset Parquets
- Produces EDA analysis artifacts for review and submission documentation
- No confirmed direct downstream pipeline notebook consumer
- Does not produce model artifacts, SQL tables used by Gold notebooks, or truth records with downstream Gold consumers
- W&B is not active in Silver notebooks

---

## Notes / Risks / Deferred Cleanup

- **`write_silver_eda_sql_outputs` vs. `log_silver_eda_sql`:** Silver_02b uses a different SQL write function than Silver_01 and Silver_02a. `write_silver_eda_sql_outputs` targets 7 Silver EDA-specific tables (`silver.eda_*`); `log_silver_eda_sql` targets pipeline metadata tables (`capstone.pipeline_runs`, etc.). The two are not interchangeable.
- **`USE_EXPERIMENT_TRACKING = False`:** Unlike Silver_02a where `wandb.init` is triple-quoted, Silver_02b uses an explicit boolean gate that is evaluated at runtime. If set to `True` in the future, the `if` branch must be populated with `wandb.init` and `finalize_wandb_stage` / `wandb_run.finish` calls.
- **IsolationForest is a genuine model fit, not a diagnostic import:** The `MODEL_TRAINING` decision tag reflects a real `IsolationForest.fit_predict` call on the `normal_clean` subset. The result (`outlier_summary.csv`) is diagnostic — it does not feed into the production pipeline — but the computation is real. Sample size is capped at `OUTLIER_AUDIT_SAMPLE_N`.
- **PCA scatter plot not saved:** The 2D PCA scatter colored by profiled state is displayed inline but not written to disk. Only the explained variance and loadings CSVs are saved. If the inline plot needs to be preserved for reporting, a `fig.savefig(...)` call should be added.
- **Aligned onset plots contingent on `aligned_onset_df`:** Cell 103 guards on `if "aligned_onset_df" not in globals() or aligned_onset_df.empty`. If the episode structure in the profiled dataframe does not yield valid onset boundaries, this section is silently skipped.
- **Truth record is for `eda_profile`, not for the Silver dataset:** Silver_02b's truth record documents the EDA artifact bundle, not a transformed version of the Silver dataset. It does not stamp `meta__truth_hash` into the profiled dataframe rows. The `SILVER_EDA_PROFILE_TRUTH_HASH` lives in the truth JSON and index only.
- **Generator input manifest carries only path strings:** The manifest validates that listed paths exist at time of write, but does not re-validate them at read time. If artifact files are deleted or moved after Silver_02b runs, the generator will encounter missing-path errors at its own load step.
- **Missingness quarantine passthrough depends on Silver_01 truth record structure:** Cell 118 reads `silver_truth.get("runtime_facts", {}).get("missingness_quarantine")`. If Silver_01's truth record structure changes (e.g., the `runtime_facts` key is renamed), this passthrough silently passes `None` and the quarantine payload is lost from Silver_02b's truth record.
- **`write_layer_dataframe` imported but not used for writes:** Same as Silver_02a — in scope but not called directly.
- **`MissingIDFieldWarning` during `nbconvert --execute`:** Non-fatal; normalize cell IDs before a future nbformat upgrade.
