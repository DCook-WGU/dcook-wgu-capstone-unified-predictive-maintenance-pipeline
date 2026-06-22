# Notebook Code Reference: EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3

**Source:** `notebooks/eda/EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3.ipynb`
**Stage:** Silver EDA — Subset Builder and Clean-Normal Constructor
**Cells:** 115 (52 code, 63 markdown), 17 inventory sections

---

## Notebook Purpose

Silver 02a builds the profiled Silver subset that separates genuinely stable normal behavior from normal rows that may already be contaminated by early fault onset. It loads the Silver Pre-EDA Parquet produced by Silver 01, profiles normal episodes through a windowing and sensor-quality scoring pipeline, and assigns a `machine_status__profiled` label to every row — yielding `normal_clean`, `normal_contaminated`, `normal_suspect`, `abnormal`, and `recovery` subsets.

The core analytical contribution is the **clean-normal profiling pipeline**: rather than treating all rows labeled `normal` as equally valid training data, Silver 02a identifies which normal rows come from stable, non-transitional behavior by windowing normal episodes, scoring window quality against sensor statistics, and building a per-sensor baseline. Rows are then scored against that baseline and classified by quality. This distinction matters because Gold's Isolation Forest models learn what normal looks like — contaminated or transitional normal rows degrade that reference.

Silver 02a also creates its own truth record (`silver_eda_subsets_truth`) with `parent_truth_hash` pointing to Silver_01's hash, establishing a verifiable lineage link from the profiled subsets back to the Silver Pre-EDA run.

Deliverables: three profiled Parquet files, two sensor baseline artifacts, a subset summary JSON, per-sensor profile plots, a subset truth record, and an optional Silver EDA SQL log entry.

---

## Pipeline Role

| Attribute | Value |
|---|---|
| Stage | Silver EDA (`silver_eda`) |
| Position | Third in the active chain; downstream of Silver_01_PreEDA |
| Upstream input | Silver Pre-EDA Parquet from `SILVER_TRAIN_DATA_PATH`; feature registry JSON from Silver_01 truth record |
| Downstream output | Profiled Parquet files consumed by Silver_02b EDA and Gold_01_PreProcessing |
| Truth chain | Reads Silver_01 `meta__truth_hash` as parent; creates new `silver_eda_subsets_truth` record |
| W&B | Disabled; `wandb.init` preserved in triple-quoted string, not executed |

---

## Inputs

| Input | Source | Form | Used For |
|---|---|---|---|
| Silver Pre-EDA Parquet | `SILVER_TRAIN_DATA_PATH / SILVER_TRAIN_DATA_FILE_NAME` | Parquet (fallback: glob for any `.parquet`) | Primary dataframe for all profiling |
| Feature registry JSON | Path resolved from parent Silver truth record | JSON dict | Resolves `SENSOR_COLUMNS`; fallback to `sensor_`-prefix detection |
| Parent Silver truth record | `TRUTHS_PATH/silver/` via `load_parent_truth_record_from_dataframe` | JSON | Provides `FEATURE_REGISTRY_PATH`; truth chain link |
| Project config | `load_notebook_context(stage="silver_eda", dataset="pump")` | YAML → `CTX` | All runtime constants, paths, stage config, fallbacks |
| Environment variables | OS environment | Strings | DB engine, `CAPSTONE_SCHEMA`, `DATASET_ID` / `RUN_ID` override |

---

## Configuration and Runtime Context

| Item | Source | Purpose |
|---|---|---|
| `SILVER_EDA_CFG` | `CTX` | Silver EDA stage config block; required by Silver-specific sanity check |
| `DEFAULT_FALLBACKS_CFG` | `CTX` | Fallback values for `ASSET_ID` / `RUN_ID` resolution |
| `SILVER_PROCESS_RUN_ID` / `SILVER_SUBSET_PROCESS_RUN_ID` | `make_process_run_id(...)` | Unique IDs for the parent Silver run and this subset run; written into truth record |
| `TRIM_FRAC` | `SILVER_EDA_CFG` | Fraction of rows trimmed from each normal episode edge (default ~0.10) |
| `WINDOWS_PER_EPISODE` | `SILVER_EDA_CFG` | Number of equal-size windows per trimmed normal episode (default 5) |
| `MIN_WINDOW_ROWS` | `SILVER_EDA_CFG` | Minimum rows per window; windows below this are discarded |
| `KEEP_WINDOW_FRAC` | `SILVER_EDA_CFG` | Fraction of best-scoring windows to retain for baseline (default 0.80) |
| `VALUE_DEVIATION_THRESHOLD` | `SILVER_EDA_CFG` | IQR multiples above which a row's sensor value is flagged as deviant |
| `DELTA_DEVIATION_THRESHOLD` | `SILVER_EDA_CFG` | IQR multiples above which a row's delta (row-to-row change) is flagged |
| `SUSPECT_SENSOR_COUNT` | `SILVER_EDA_CFG` | Deviant-sensor count at which a row is classified `"suspect"` |
| `EXCLUDE_SENSOR_COUNT` | `SILVER_EDA_CFG` | Deviant-sensor count at which a row is classified `"exclude"` |
| `LABEL_COLUMN` / `NORMAL_VALUES` | `SILVER_EDA_CFG` | Column and values that identify rows as normal for episode building |
| `WRITE_TO_POSTGRES` | Notebook cell (bool gate) | Controls whether `log_silver_eda_sql` executes; allows offline/dry runs |
| `DATASET_NAME` / `RUN_ID` / `ASSET_ID` / `DATASET_ID` | Resolved from env → config → fallback | Row/run identity; written into truth record and SQL rows |
| `SILVER_VERSION` / `TRUTH_VERSION` | `VERSIONS_CFG` via `CTX` | Version stamps in truth record metadata |
| `PIPELINE_MODE` / `RUN_MODE` | `PIPELINE` via `CTX` | Propagated into truth record; drives dry-run vs live-run distinctions |

---

## Logical Workflow Map

1. Load shared context via `load_notebook_context(stage="silver_eda")`; run general + Silver-EDA-specific sanity checks; `ledger.add` on pass
2. Create Silver EDA artifact directory tree; export config snapshot
3. Establish SQL engine; run read-only smoke-check query; resolve `DATASET_ID` / `RUN_ID` / `ASSET_ID`
4. Start logging (`log_layer_paths`); record ledger checkpoint
5. (W&B disabled — `wandb.init` preserved in triple-quoted string, not executed)
6. Load Silver Pre-EDA Parquet via `load_data`; log shape; `ledger.add`
7. Extract `SILVER_TRUTH_HASH` from `silver_eda_df` **before any filtering** — this hash is `parent_truth_hash` for the subset truth record; load parent Silver truth record; resolve `FEATURE_REGISTRY_PATH`
8. Load feature registry JSON; resolve `SENSOR_COLUMNS` (fallback: `sensor_`-prefix detection)
9. Detect source state column; normalize raw status labels to `normal / abnormal / recovery`; review state counts; `ledger.add`
10. Define and execute the five-stage clean-normal profiling pipeline:
    - `assign_normal_episodes` — assign episode IDs to contiguous normal runs
    - `trim_normal_episode_edges` — discard first and last `TRIM_FRAC` of each episode
    - `create_episode_windows` — split trimmed episodes into `WINDOWS_PER_EPISODE` segments
    - `calculate_window_sensor_stats` — compute per-sensor statistics per window
    - `score_and_filter_windows` → `build_final_sensor_baseline` — rank windows and aggregate kept windows into `final_sensor_baseline_df`
11. Review lowest-scoring windows, quality distribution, and baseline shape (display only)
12. Save sensor baseline artifacts: `silver_sensor_baseline_profiles.csv` and `.json` to `SENSOR_PROFILE_DIR`; `ledger.add`
13. Generate and save per-sensor profile plots to `SENSOR_PROFILE_DIR/plots`; optionally display inline
14. Score every row in `normal_profile_df` against the baseline (`score_rows_against_sensor_baseline`); classify by quality (`classify_normal_training_quality`) → `scored_normal_quality_df`
15. Construct `silver_subset_df` from `scored_normal_quality_df`; add `final_row_quality_class` and boolean quality columns
16. Review normal-only subset; plot sensor profiles against baseline
17. Define profiled-state label constants; rebuild `silver_subset_df`; map `machine_status__profiled` using source state + quality class
18. Save three profiled Parquet files and subset summary JSON to `SILVER_TRAIN_DATA_PATH`; `ledger.add`
19. Initialize and populate `silver_eda_subsets_truth` (with `parent_truth_hash=SILVER_TRUTH_HASH`); build truth record; save and index; `ledger.add`
20. Review final `silver_subset_df` structure (`.info()`)
21. Write Silver EDA summary to PostgreSQL (`log_silver_eda_sql`) if `WRITE_TO_POSTGRES`

---

## Section Overview

| Section | Key Inputs | Key Outputs / Side Effects |
|---|---|---|
| Imports and environment setup | None | All imports; `cfg_require_mapping` helper defined |
| Context load and configuration | Config files, env vars | `CTX`, `SILVER_EDA_CFG`, `DEFAULT_FALLBACKS_CFG`, all layer/version/process-run constants, `logger`, `ledger` |
| Context sanity checks | Globals dict | Raises `NameError` if general or `SILVER_EDA_CFG` check fails; `ledger.add` on pass |
| Silver EDA artifact directories | `DATASET_NAME`, `paths` | Full `SILVER_EDA_ARTIFACT_DIR` tree created on disk; `export_config_snapshot` |
| SQL Runtime Context | `CTX`, env vars | `engine`; `DATASET_ID`, `RUN_ID`, `ASSET_ID` resolved |
| SQL smoke check | `engine` | `sql_smoke_check_dataframe` displayed; confirms DB connectivity |
| Logging and ledger initialization | `paths`, `logger`, `ledger` | `log_layer_paths`; `ledger.add` checkpoint |
| W&B guard (disabled) | None | `wandb.init` in triple-quoted string — not executed |
| Load Silver Pre-EDA Output | `SILVER_TRAIN_DATA_PATH` | `silver_eda_df`; `ledger.add` with shape |
| Resolve parent truth record and confirm dataset identity | `silver_eda_df` | `SILVER_TRUTH_HASH` (before any filtering); `silver_truth`; `FEATURE_REGISTRY_PATH` |
| Load feature registry and sensor columns | `FEATURE_REGISTRY_PATH` | `feature_registry`; `SENSOR_COLUMNS` |
| Source state normalization | `silver_eda_df`, state-column candidates | Normalized state column in `silver_eda_df`; `STATE_COL_SOURCE`; `ledger.add` |
| Define windowing helpers | None | `assign_normal_episodes`, `trim_normal_episode_edges`, `create_episode_windows`, `calculate_window_sensor_stats`, `score_and_filter_windows`, `build_final_sensor_baseline` defined |
| Run normal windowing workflow | `silver_eda_df`, windowing config constants | `normal_profile_df`, `window_sensor_stats_df`, `window_quality_df`, `final_sensor_baseline_df`, `kept_window_stats_df` |
| Baseline inspection checkpoints | `window_quality_df`, `final_sensor_baseline_df` | Display only; window quality distribution, baseline shape review |
| Save sensor baseline artifact | `final_sensor_baseline_df`, `SENSOR_PROFILE_DIR` | `silver_sensor_baseline_profiles.csv`; `silver_sensor_baseline_profiles.json`; `ledger.add` |
| Sensor profile plot helpers | None | `display_sensor_profile`, `plot_all_sensor_profiles`, `plot_sensor_profile_with_baseline` defined |
| Generate sensor profile plots | `normal_only_df`, `final_sensor_baseline_df`, `SENSOR_PROFILE_PLOT_DIR` | Per-sensor PNG files saved; optional inline display |
| Row scoring configuration | Threshold constants | `VALUE_DEVIATION_THRESHOLD`, `DELTA_DEVIATION_THRESHOLD`, `SUSPECT_SENSOR_COUNT`, `EXCLUDE_SENSOR_COUNT` resolved; config summary displayed |
| Define scoring and classification helpers | None | `score_rows_against_sensor_baseline`, `classify_normal_training_quality` defined |
| Apply row-level scoring | `normal_profile_df`, `final_sensor_baseline_df` | `scored_normal_quality_df` with per-row sensor-deviation counts and quality class |
| Build final profiled-state dataframe | `scored_normal_quality_df` | `silver_subset_df` with `final_row_quality_class` and boolean quality columns |
| Review normal-only subset | `silver_subset_df` | `normal_only_df`; sensor-baseline overlay plot displayed |
| Define profiled-state label constants and rebuild | `scored_normal_quality_df`, profiled-state constants | `silver_subset_df` with `machine_status__profiled` column; profiled/quality counts displayed |
| Save profiled subset artifacts | `silver_subset_df`, `SILVER_TRAIN_DATA_PATH`, `SILVER_EDA_SUMMARY_DIR` | Three Parquet files; `subset_summary` JSON; `ledger.add` |
| Finalize Silver EDA Subsets truth record | `silver_subset_df`, `SILVER_TRUTH_HASH` | `silver_eda_subsets_truth_record`; truth JSON saved; `append_truth_index`; `ledger.add` |
| Final structure review | `silver_subset_df` | `.info()` display only |
| Silver EDA SQL write | `engine`, `globals()` | `log_silver_eda_sql` → PostgreSQL rows in pipeline tables; skipped when `WRITE_TO_POSTGRES = False` |

---

## Section Details

### Context Load and Sanity Checks

`load_notebook_context(stage="silver_eda", dataset="pump", mode="train", profile="default")` bootstraps the shared runtime. Note: the stage string is `"silver_eda"`, not `"silver"` — Silver 02a uses a dedicated EDA stage config block (`SILVER_EDA_CFG`) rather than the base Silver stage config used by Silver_01.

Two sanity checks follow:
- The **general check** verifies the standard set of required shared context variables.
- The **Silver EDA–specific check** verifies `SILVER_EDA_CFG` is bound. Absence raises `NameError`.

Both preserved original standalone setup blocks (original `configure_logging` and `Ledger` initialization) exist as triple-quoted strings in cells 23 and 29. They are not executed — `load_notebook_context` handles both. They document the pre-shared-context setup sequence.

---

### Silver EDA Artifact Directory Tree

The artifact directory tree is created early, before the Silver Parquet is loaded, because the config-snapshot export also runs here. This is unlike Bronze and Silver_01, which defer directory creation until dataset name is confirmed post-ingest — Silver 02a already has `DATASET_NAME` from config and trusts that it matches the Silver Parquet it will load.

Directories created: `SILVER_EDA_ARTIFACT_DIR`, `SILVER_EDA_SENSOR_PROFILES_DIR`, `SILVER_EDA_SUMMARY_DIR`, `SILVER_EDA_SUBSETS_DIR`, `SILVER_EDA_LINEAGE_DIR`, `SILVER_EDA_CONFIG_DIR`, `SILVER_EDA_METADATA_DIR`, and several plot directories (`ALIGNED_ONSET_PLOT_DIR`, `CORRELATION_ARTIFACT_DIR`, `DISTRIBUTION_PLOT_DIR`, `PCA_ARTIFACT_DIR`, `TIMELINE_OVERLAY_DIR`). `export_config_snapshot` writes the config JSON to `SILVER_EDA_CONFIG_DIR`.

---

### Load Silver Pre-EDA Output and Parent Truth Resolution

`load_data(silver_data_path.parent, silver_data_path.name)` reads the Silver Pre-EDA Parquet into `silver_eda_df`. The path is resolved by first checking for the canonical file at `SILVER_TRAIN_DATA_PATH / SILVER_TRAIN_DATA_FILE_NAME`; if absent, a glob finds the first `.parquet` candidate with a logger warning.

`SILVER_TRUTH_HASH = extract_truth_hash(silver_eda_df)` runs **immediately after load, before any filtering or transformation**. This is intentional: the hash must reflect the full unmodified Silver_01 output so the lineage link in the downstream truth record is valid even when the subset build drops rows. If the hash is `None`, `ValueError` halts execution.

`load_parent_truth_record_from_dataframe(dataframe=silver_eda_df, truth_dir=TRUTHS_PATH, ...)` then loads the Silver_01 truth JSON to retrieve `FEATURE_REGISTRY_PATH`. The feature registry is validated as a non-None dict before use; `SENSOR_COLUMNS` is resolved from the registry's `feature_columns` list. If no sensor columns are found there, the notebook falls back to detecting columns beginning with `sensor_` in the Silver dataframe and emits a `logger.warning`.

---

### Source State Normalization

The source state column is detected by probing `silver_eda_df.columns` for candidates in priority order: `"machine_status__synthetic"`, `"machine_status"`, `"status"`, `"state"`. The detected column name is stored as `STATE_COL_SOURCE_RAW`; the canonical alias `STATE_COL_SOURCE` is fixed to `"machine_status__synthetic"`.

A `normalize_machine_state` function maps raw label variants (e.g., `"ok"`, `"fault"`, `"cooldown"`, `"normal"`) to three normalized values: `"normal"`, `"abnormal"`, or `"recovery"`. This normalization is written to a new column in `silver_eda_df`. Cell 40 displays post-normalization value counts as a checkpoint and records a `ledger.add` with the state distribution.

---

### Clean-Normal Profiling Pipeline

The core analytical block runs five functions in sequence on `silver_eda_df` (cell 57).

**1. `assign_normal_episodes`:** Detects transitions between normal and non-normal rows. Uses a cumsum on the status-change mask to assign a monotonically incrementing `__normal_episode_id` to each contiguous normal run. Non-normal rows receive `NaN` episode IDs.

**2. `trim_normal_episode_edges`:** Removes the first and last `TRIM_FRAC` fraction of each normal episode. The rationale: rows near episode boundaries may capture fault-onset transients (end of episode) or post-recovery stabilization (start of episode). Episodes too short to yield at least `min_rows_after_trim` (1000 rows) after trimming are dropped entirely. Result: `__keep_after_episode_trim` boolean column; only kept rows participate in windowing.

**3. `create_episode_windows`:** Splits each trimmed episode into `WINDOWS_PER_EPISODE` equal segments using `np.array_split`. Each window receives a string label in the form `episode_<N>__window_<M>`. Windows below `MIN_WINDOW_ROWS` are discarded. Result: `__window_id` column on normal rows.

**4. `calculate_window_sensor_stats`:** For every sensor in every window, computes: median, mean, std, IQR, 5th and 95th percentile values, delta IQR, and `delta_abs_q95` (95th percentile of absolute row-to-row change). Result: `window_sensor_stats_df` — one row per (window × sensor).

**5. `score_and_filter_windows` → `build_final_sensor_baseline`:** `score_and_filter_windows` aggregates per-window statistics across sensors, ranks windows by IQR magnitude, delta magnitude, and missing-data rate, and classifies the top `KEEP_WINDOW_FRAC` as `keep`. `build_final_sensor_baseline` aggregates per-sensor statistics from kept windows only, producing `final_sensor_baseline_df` — the learned stable-normal sensor profile.

---

### Save Sensor Baseline Artifact

Cell 67 writes two artifacts to `SENSOR_PROFILE_DIR`:
- `silver_sensor_baseline_profiles.csv` — `final_sensor_baseline_df` in tabular form.
- `silver_sensor_baseline_profiles.json` — structured JSON with metadata (source notebook, profile method, state column used, normal values, baseline config thresholds) and the full per-sensor profile dict.

The JSON artifact includes `artifact_type`, `created_at_utc`, baseline config (all threshold constants), and a `summary` (sensor count, median baseline rows/windows per sensor). These files are reusable by Silver 02b, the synthetic generator, and Gold preprocessing without re-running the windowing pipeline. `ledger.add(kind="artifact", ...)` records the saved paths.

---

### Sensor Profile Plots

`plot_all_sensor_profiles` generates one PNG per sensor in `SENSOR_PROFILE_DIR/plots`. Each plot shows the raw sensor values alongside the baseline median and IQ range. `SHOW_ALL_SENSOR_PROFILE_PLOTS = False` by default to avoid rendering ~25+ plots inline; files are still saved. Spot-check cells call `display_sensor_profile` and `plot_sensor_profile_with_baseline` for a small set of named sensors.

These plots are visual confirmation artifacts — they do not feed into the modeling pipeline. If a sensor profile looks anomalous (extreme IQR, high missing rate, implausible baseline), it points to a sensor that may need flagging or exclusion.

---

### Row-Level Quality Scoring

`score_rows_against_sensor_baseline` processes every row in `normal_profile_df` (not the full Silver dataframe — only normal-episode rows). For each sensor in each row, it computes:
- `value_deviation`: absolute distance from baseline median, normalized by baseline IQR → `value_flag` when > `VALUE_DEVIATION_THRESHOLD`
- `delta_deviation`: absolute row-to-row delta, normalized by `baseline_delta_abs_q95` → `delta_flag` when > `DELTA_DEVIATION_THRESHOLD`

Per-row counts of sensors exceeding each threshold are summed into `normal_value_abnormal_sensor_count`, `normal_delta_abnormal_sensor_count`, and `normal_total_abnormal_sensor_count`.

`classify_normal_training_quality` assigns `normal_training_quality_class` to every row:
- Non-normal rows → `"not_normal"`
- Normal rows: `"clean"` by default; elevated to `"suspect"` if `total_abnormal_sensor_count >= SUSPECT_SENSOR_COUNT`; elevated to `"exclude"` if count meets `EXCLUDE_SENSOR_COUNT`
- Boolean `is_clean_normal_for_training` is also written per row

Result: `scored_normal_quality_df` — all normal-profile rows with quality classification.

---

### Profiled-State Mapping

`silver_subset_df` is built from a copy of `scored_normal_quality_df`. Three boolean columns are added first: `row_is_clean_normal`, `row_is_suspect_normal`, `row_is_exclude_from_normal_training`.

Cell 104 then declares the canonical profiled-state label constants and maps `machine_status__profiled` for every row:

| Source State | Quality Class | `machine_status__profiled` Value |
|---|---|---|
| `normal` | `clean` | `"normal_clean"` |
| `normal` | `suspect` | `"normal_suspect"` |
| `normal` | `exclude` | `"normal_contaminated"` |
| `abnormal` | any | `"abnormal"` |
| `recovery` | any | `"recovery"` |

`silver_subset_df` is rebuilt from `scored_normal_quality_df` with `.copy()` (not mutated in place) to ensure clean column assignment. The `machine_status__profiled` column is the primary label consumed by Silver 02b and Gold_01.

---

### Save Profiled Subset Artifacts

Cell 106 writes the three data-output Parquet files to `SILVER_SUBSET_DATA_DIR` (which resolves to `SILVER_TRAIN_DATA_PATH`) and a JSON summary to `SILVER_EDA_SUMMARY_DIR`. All writes use direct `to_parquet` (not the `save_data` wrapper) and `json.dump` (not `save_json`).

| File | Content | Path |
|---|---|---|
| `<dataset>__silver_subsets__profiled_dataframe.parquet` | All rows with `machine_status__profiled` | `SILVER_TRAIN_DATA_PATH` |
| `<dataset>__silver_subsets__normal_clean.parquet` | Rows where `machine_status__profiled == "normal_clean"` | `SILVER_TRAIN_DATA_PATH` |
| `<dataset>__silver_subsets__normal_contaminated.parquet` | Rows where `machine_status__profiled == "normal_contaminated"` | `SILVER_TRAIN_DATA_PATH` |
| `<dataset>__silver_subsets__summary.json` | Profiled/source state counts, row quality counts, artifact paths, run metadata | `SILVER_EDA_SUMMARY_DIR` |

The commented-out `normal_contaminated` alias for Gold (`gold_train__normal_clean.parquet`) is present in cell 106 but not executed. `ledger.add(kind="step", step="save_subset_outputs", ...)` records all four artifact paths.

---

### Finalize Silver EDA Subsets Truth Record

`initialize_layer_truth(layer_name="silver", parent_truth_hash=SILVER_TRUTH_HASH, process_run_id=SILVER_SUBSET_PROCESS_RUN_ID, ...)` creates the subset truth root. This is not the same truth record as Silver_01's — it is a secondary Silver-layer truth record specific to the subset-building stage.

The truth record is populated in four named sections via `update_truth_section`:
- `config_snapshot`: stage, layer, dataset name, pipeline mode, run mode
- `runtime_facts`: parent hash, state columns used, profiling method, profiled/source counts, row count
- `artifact_paths`: all three Parquet paths, summary JSON, data dir, artifact dir
- `notes`: human-readable purpose statement

`build_truth_record(row_count=len(silver_subset_df), ...)` finalizes with row/column counts. `save_truth_record` writes the JSON to `TRUTHS_PATH/silver/<dataset>__silver__truth__<hash>.json`. `append_truth_index` adds the entry to the project-wide truth index. `ledger.add` records both the new hash (`SILVER_EDA_SUBSETS_TRUTH_HASH`) and the parent Silver hash.

---

### SQL Write Gate

`log_silver_eda_sql(engine, CAPSTONE_SCHEMA, DATASET_ID, RUN_ID, notebook_globals=globals(), ...)` is the only SQL write call in this notebook. It writes summary metadata to `capstone.pipeline_runs`, `capstone.data_quality_events`, and `capstone.pipeline_artifacts`. The `WRITE_TO_POSTGRES = True` gate controls whether it executes; setting it to `False` skips all database side effects without affecting any file artifact. The read-only smoke check at startup is independent of this gate.

---

## Key Function Calls and In-Place Usage

| Function | Section | Inputs Provided | Return / Side Effect |
|---|---|---|---|
| `load_notebook_context(...)` | Context load | `stage="silver_eda"`, `dataset="pump"` | `CTX`; `SILVER_EDA_CFG`, `logger`, `ledger` unpacked |
| `make_process_run_id(...)` | Context load | Prefix from `SILVER_EDA_CFG` | `SILVER_PROCESS_RUN_ID`, `SILVER_SUBSET_PROCESS_RUN_ID` |
| `export_config_snapshot(...)` | Artifact dirs | `CONFIG`, `CONFIG_SNAPSHOT_PATH` | Config JSON written to `SILVER_EDA_CONFIG_DIR` |
| `get_engine_from_env()` | SQL Runtime Context | None | `engine` |
| `read_sql_dataframe(engine, ...)` | SQL smoke check | Inline `SELECT` on `information_schema.tables` | `sql_smoke_check_dataframe` |
| `log_layer_paths(...)` | Logging setup | `paths`, `logger` | Layer path log written |
| `load_data(...)` | Load Silver Pre-EDA | `silver_data_path.parent`, file name | `silver_eda_df` |
| `extract_truth_hash(silver_eda_df)` | Resolve parent truth | `silver_eda_df` (before any filtering) | `SILVER_TRUTH_HASH` |
| `load_parent_truth_record_from_dataframe(...)` | Resolve parent truth | `silver_eda_df`, `TRUTHS_PATH` | `silver_truth`; `FEATURE_REGISTRY_PATH` |
| `load_json(...)` | Load feature registry | `FEATURE_REGISTRY_PATH.parent`, file name | `feature_registry` dict |
| `assign_normal_episodes(...)` | Windowing pipeline | `silver_eda_df`, `LABEL_COLUMN`, `NORMAL_VALUES` | `normal_profile_df` with episode IDs |
| `trim_normal_episode_edges(...)` | Windowing pipeline | `normal_profile_df`, `TRIM_FRAC` | Rows with `__keep_after_episode_trim` flag |
| `create_episode_windows(...)` | Windowing pipeline | Trimmed df, `WINDOWS_PER_EPISODE`, `MIN_WINDOW_ROWS` | `__window_id` column |
| `calculate_window_sensor_stats(...)` | Windowing pipeline | `normal_profile_df`, `SENSOR_COLUMNS` | `window_sensor_stats_df` |
| `score_and_filter_windows(...)` | Windowing pipeline | `window_sensor_stats_df`, `KEEP_WINDOW_FRAC` | `window_quality_df` with `keep`/`drop` classification |
| `build_final_sensor_baseline(...)` | Windowing pipeline | `window_sensor_stats_df`, `window_quality_df` | `final_sensor_baseline_df`, `kept_window_stats_df` |
| `plot_all_sensor_profiles(...)` | Sensor profile plots | `normal_only_df`, `final_sensor_baseline_df`, `SENSOR_COLUMNS`, `SENSOR_PROFILE_PLOT_DIR` | PNG files saved; result dict |
| `score_rows_against_sensor_baseline(...)` | Row scoring | `normal_profile_df`, `final_sensor_baseline_df`, thresholds | `scored_normal_quality_df` with per-sensor deviation counts |
| `classify_normal_training_quality(...)` | Row scoring | `scored_normal_quality_df`, `SUSPECT_SENSOR_COUNT`, `EXCLUDE_SENSOR_COUNT` | `normal_training_quality_class` and boolean columns added |
| `initialize_layer_truth(...)` | Finalize truth record | `layer_name="silver"`, `parent_truth_hash=SILVER_TRUTH_HASH`, `SILVER_SUBSET_PROCESS_RUN_ID` | `silver_eda_subsets_truth` dict |
| `update_truth_section(...)` × 4 | Finalize truth record | `silver_eda_subsets_truth`, section name, payload dict | Truth dict updated |
| `build_truth_record(...)` | Finalize truth record | `silver_eda_subsets_truth`, row/column counts | `silver_eda_subsets_truth_record` with `truth_hash` |
| `save_truth_record(...)` | Finalize truth record | `silver_eda_subsets_truth_record`, `TRUTHS_PATH`, `DATASET_NAME`, `layer_name="silver"` | JSON at `TRUTHS_PATH/silver/<dataset>__silver__truth__<hash>.json` |
| `append_truth_index(...)` | Finalize truth record | `silver_eda_subsets_truth_record`, `TRUTH_INDEX_PATH` | Truth index updated |
| `log_silver_eda_sql(...)` | SQL write | `engine`, `CAPSTONE_SCHEMA`, `DATASET_ID`, `RUN_ID`, `globals()` | Rows in pipeline SQL tables |

---

## Outputs and Artifacts

| Output | Type | Location | Downstream Consumer |
|---|---|---|---|
| Profiled dataframe Parquet | `.parquet` | `SILVER_TRAIN_DATA_PATH / <dataset>__silver_subsets__profiled_dataframe.parquet` | Silver_02b EDA; Gold_01_PreProcessing |
| Clean-normal Parquet | `.parquet` | `SILVER_TRAIN_DATA_PATH / <dataset>__silver_subsets__normal_clean.parquet` | Gold Isolation Forest training reference |
| Contaminated-normal Parquet | `.parquet` | `SILVER_TRAIN_DATA_PATH / <dataset>__silver_subsets__normal_contaminated.parquet` | EDA reference; not used in primary modeling |
| Sensor baseline CSV | `.csv` | `SENSOR_PROFILE_DIR / silver_sensor_baseline_profiles.csv` | Silver_02b; synthetic generator; Gold EDA |
| Sensor baseline JSON | `.json` | `SENSOR_PROFILE_DIR / silver_sensor_baseline_profiles.json` | Same consumers; carries metadata and config |
| Subset summary JSON | `.json` | `SILVER_EDA_SUMMARY_DIR / <dataset>__silver_subsets__summary.json` | Audit / reproducibility |
| Per-sensor profile plots | `.png` | `SENSOR_PROFILE_DIR/plots/` | Visual review; documentation |
| Silver EDA Subsets truth record | `.json` | `TRUTHS_PATH/silver/<dataset>__silver__truth__<hash>.json` | Cross-stage lineage; downstream truth lookup |
| Truth index entry | Appended JSONL | `TRUTH_INDEX_PATH` | Cross-run lineage lookup |
| Config snapshot | `.json` | `SILVER_EDA_CONFIG_DIR` | Audit / reproducibility |
| Silver EDA SQL rows | PostgreSQL rows | `capstone.pipeline_runs`, `capstone.data_quality_events`, `capstone.pipeline_artifacts` | Operational monitoring (when `WRITE_TO_POSTGRES = True`) |
| Ledger entries | In-memory + ledger file | Via `ledger.add` throughout | Audit trail |

---

## Data Quality / Validation Behavior

| Check | Where | Failure / Risk Prevented |
|---|---|---|
| General context sanity check | After `load_notebook_context` | Prevents `NameError` from incomplete shared context |
| `SILVER_EDA_CFG` check | After general check | Prevents `NameError: Missing Silver context variables` |
| SQL smoke check | Before data load | Catches DB unavailability before any processing |
| Silver Parquet exists or fallback glob | Load Silver Pre-EDA | Raises `FileNotFoundError` if no `.parquet` found in `SILVER_TRAIN_DATA_PATH` |
| `SILVER_TRUTH_HASH` not None | Immediately after `load_data` | Prevents proceeding without a valid parent truth link |
| Feature registry non-None dict | After `load_json` | Raises `ValueError` / `TypeError` if registry is missing or malformed |
| State column resolution | Source state normalization | Raises `KeyError` if none of the candidate columns are found |
| `FEATURE_REGISTRY_PATH` not None | Before `load_json` | Raises `ValueError` if parent truth record did not yield a registry path |
| Window count > 0 after trimming | Implicit in `create_episode_windows` | Drops episodes too short to window; no error but coverage reduces |
| `scored_normal_quality_df` not empty | Implicit in row scoring | If all normal rows are excluded by windowing, downstream profile parquets will be empty |
| Truth record row/column count | `build_truth_record` | Finalizes row/column counts in truth dict; downstream can verify against loaded dataframe |

---

## Downstream Handoff

**Silver_02b EDA** reads:
- `<dataset>__silver_subsets__profiled_dataframe.parquet` — the full profiled dataframe with `machine_status__profiled` for state-stratified EDA
- `silver_sensor_baseline_profiles.json` / `.csv` — the clean-normal sensor baseline for comparative analysis

**Gold_01_PreProcessing** reads:
- `<dataset>__silver_subsets__profiled_dataframe.parquet` — uses `machine_status__profiled` to separate clean-normal training rows from anomaly rows
- `<dataset>__silver_subsets__normal_clean.parquet` — may be used as a direct training-data input for the Isolation Forest baseline model

Both consumers can look up the Silver EDA Subsets truth record via `SILVER_EDA_SUBSETS_TRUTH_HASH` to verify the profiled artifacts match their expected row/column counts and to follow the truth chain back to the originating Silver_01 run.

---

## Relationship to Other Notebooks

### Upstream Context

Silver_02a reads Silver_01_PreEDA's pre-EDA profile Parquets. It is the first Silver notebook to produce clean analytical subsets from the pre-EDA profiles. No dependency on Silver_02b, Gold, or any cascade notebooks.

### Downstream Handoff

Silver_02a provides:
- The clean analytical Parquet subset consumed by Gold_01_PreProcessing as its primary pipeline input
- An imputation recommendation JSON consumed by Gold_01 to guide scaling and fill-strategy decisions
- Profiled Silver subsets consumed by Silver_02b for deeper EDA
- Silver layer frames written to `capstone.silver` via `log_silver_eda_sql`

### Pipeline Position

The pivot point between Silver EDA and Gold preprocessing. Silver_02a is the last Silver notebook whose file outputs have confirmed direct artifact-level dependencies in a downstream Gold notebook (Gold_01). Its clean subset Parquet is the most important Silver-to-Gold handoff artifact.

### Relationship Summary

- Reads Silver_01 pre-EDA profile Parquets
- Produces the clean analytical Parquet that is Gold_01's primary input — the key Silver-to-Gold handoff
- Provides imputation recommendation JSON that informs Gold_01's preprocessing strategy
- Also feeds Silver_02b for supplementary EDA (Silver_02b has no confirmed downstream pipeline consumer)
- Direct downstream notebook consumers: Silver_02b; Gold_01

---

## Notes / Risks / Deferred Cleanup

- **W&B disabled:** `wandb.init` is preserved in triple-quoted string form (cell 25) and is not executed. No W&B run is created. The inventory `wandb_clues` reflect the import and the triple-quoted block, not active calls. If W&B is re-enabled, the block should be unquoted and a matching `finalize_wandb_stage` / `wandb_run.finish` sequence should be added.
- **Preserved original-setup blocks:** The standalone `configure_logging` (cell 23) and `Ledger` initialization (cell 29) blocks are triple-quoted. They document the original pre-shared-context setup and are not executed.
- **Sensor column fallback:** If the feature registry contains no `feature_columns`, the notebook falls back to `sensor_`-prefix detection in the Silver dataframe. This preserves execution continuity but emits a `logger.warning` so the gap is visible in the run log.
- **No mutation of input dataframe:** `silver_eda_df` is never modified in place. All transformations produce new copies (`normal_profile_df`, `scored_normal_quality_df`, `silver_subset_df`). Cell 104 explicitly rebuilds `silver_subset_df` from a `.copy()` before writing `machine_status__profiled`.
- **`SILVER_SUBSET_DATA_DIR` resolves to `SILVER_TRAIN_DATA_PATH`:** The alternative nested `subset_outputs/<dataset>/` path is commented out in cell 106. Downstream consumers should expect the profiled Parquet files to sit alongside the Silver Pre-EDA Parquet, not in a subdirectory.
- **Profiled subsets are not truth-stamped:** Unlike Silver_01, Silver_02a does not call `stamp_truth_columns` on the output dataframe. The three output Parquet files carry the original `meta__truth_hash` from Silver_01, not a Silver_02a hash. The `SILVER_EDA_SUBSETS_TRUTH_HASH` lives in the subset truth record JSON and truth index, not in the rows.
- **`write_layer_dataframe` imported but not used for writes:** This function is in scope but Silver_02a's primary SQL write is `log_silver_eda_sql`. `write_layer_dataframe` is available for ad-hoc use.
- **`MissingIDFieldWarning` during `nbconvert --execute`:** Cells lack the `id` field required by nbformat 5.1+. Non-fatal; normalize cell IDs before a future nbformat upgrade.
