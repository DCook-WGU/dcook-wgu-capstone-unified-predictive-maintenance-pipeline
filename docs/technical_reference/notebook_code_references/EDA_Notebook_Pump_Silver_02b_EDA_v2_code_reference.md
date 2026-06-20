# Notebook Code Reference: EDA_Notebook_Pump_Silver_02b_EDA_v2

Notebook path: `notebooks/eda/EDA_Notebook_Pump_Silver_02b_EDA_v2.ipynb`

## Notebook Purpose

Silver 02b performs profiled-state EDA on the Silver dataset produced by Silver 02a. Where Silver 02a creates the profiled status labels (`normal_clean`, `normal_contaminated`, `abnormal`, `recovery`), this notebook consumes them to characterize sensor behavior across those four states. It generates the artifact set that the synthetic generator, Gold preprocessing, and truth lineage chain all depend on: correlation matrices, sensor grouping maps, generator-input profiles, episode status counts, and the Silver EDA Profile truth record.

Notebook stage: `Silver`

## Section Overview

| Section | Cells (approx) |
|---|---|
| Imports and Environment Setup | 0–17 |
| SQL Runtime Context | 18–22 |
| Logging, Experiment Tracking, Ledger | 23–30 |
| Load Profiled Silver Dataframe | 31–32 |
| Resolve Parent Truth and Dataset Identity | 33–34 |
| Load Feature Registry | 35–36 |
| Robust Scaling Helpers and State Constants | 37–41 |
| Profiled Data Overview | 42–43 |
| Missingness and Duplicate Checks | 44–47 |
| Numeric Feature Summaries | 48–49 |
| Source State Counts | 50–51 |
| State-Level Sensor Profile Builder | 52–53 |
| Build Profiled-State Subsets | 54–55 |
| Dropped and Weak Sensor Review | 56–57 |
| Episode-Level State Summaries | 58–61 |
| Missingness by State (Source and Profiled) | 62–65 |
| State Transition and Dwell Tables | 66–67 |
| Correlation Artifacts (Normal-Clean, Contaminated, Abnormal) | 68–76 |
| Sensor Grouping from Correlation Structure | 73–74 |
| Generator Input Artifacts | 77–84 |
| Profiled State Summaries and Robust Comparison | 85–90 |
| Distribution Plots and Timeline Overlays | 91–98 |
| Aligned Anomaly-Onset Windows | 95–103 |
| Agglomerative Feature Clustering | 100–101 |
| PCA Diagnostics | 104–108 |
| Imputation Comparison | 109–110 |
| Outlier Audit by Profiled State | 111–112 |
| Episode Status Counts Export | 113–114 |
| Generator Input Manifest Update | 115–116 |
| Build Truth Record and Save Final Outputs | 117–120 |
| QA / SQL Staging and Write | 121–138 |

---

## Section Details

### Imports and Environment Setup

Cells 0–17 handle all library imports, configuration loading, context initialization, and artifact directory setup. Standard libraries (pathlib, json, logging), data/science libraries (numpy, pandas, matplotlib, seaborn, scipy, sklearn), and project utilities (paths, file I/O, logging setup, truth utilities, ledger, context loader) are imported in a single block.

`load_notebook_context()` is called with `stage="silver_eda"`, `dataset="pump"`, `mode="train"`, and `profile="default"`. This returns a `CTX` object whose attributes are aliased to the short names (`CONFIG`, `STAGE_CFG`, `RESOLVED_PATHS`, `PIPELINE`, `logger`, `ledger`, etc.) used throughout the notebook.

`TRUTH_CONFIG` is built from the resolved config and extended with the PIPELINE block so that any truth records written in this notebook carry execution-mode metadata. Canonical Silver EDA artifact directories are pinned under `artifacts/silver/<dataset>/eda/`, with named subdirectories for aligned onset plots, config, correlation analysis, distribution plots, generator inputs, lineage, metadata, PCA, and sensor profiles.

A context sanity check (cell 11) raises `NameError` if any required variable from `load_notebook_context()` is missing before any other work proceeds.

**Key variables:** `CTX`, `paths`, `CONFIG`, `STAGE_CFG`, `SILVER_EDA_CFG`, `PIPELINE`, `TRUTH_CONFIG`, `SILVER_EDA_ARTIFACT_DIR` and its subdirectory constants, `logger`, `ledger`.

---

### SQL Runtime Context

Cell 19 establishes the PostgreSQL engine via `get_engine_from_env()` and resolves `DATASET_ID` and `RUN_ID` using `first_non_empty_string()`, a helper that skips `None`, empty strings, whitespace-only strings, and dict values. A SQL smoke-check query (cell 21) reads from `information_schema.tables` to confirm that the expected Silver EDA target tables exist and the connection is working.

**Key variables:** `engine`, `CAPSTONE_SCHEMA`, `DATASET_ID`, `RUN_ID`.

---

### Logging, Experiment Tracking, and Ledger Initialization

Cells 23–30 configure the run-level logger, set `USE_EXPERIMENT_TRACKING = False` (W&B is disabled by default for this notebook), and confirm that the ledger from `load_notebook_context()` is ready. The ledger receives a `ledger_context_ready` step entry here to mark the run's starting point in the ledger record.

---

### Load Profiled Silver Dataframe

Cell 32 resolves the path to the Silver subsets parquet produced by Silver 02a. The canonical location is `<SILVER_TRAIN_DATA_PATH>/<dataset>__silver_subsets__profiled_dataframe.parquet`. If that path does not exist, the notebook falls back to searching the Silver EDA output and artifact directories using `rglob()`. The resolved path is logged and the dataframe is loaded as `silver_eda_dataframe`.

**Key variables:** `PROFILED_DF_PATH`, `silver_eda_dataframe`.

---

### Resolve Parent Truth and Dataset Identity

Cell 34 extracts the truth hash from the loaded dataframe using `extract_truth_hash(silver_eda_dataframe)` and raises `ValueError` if it is absent. This hash is stored as `SILVER_TRUTH_HASH` and will become the `parent_truth_hash` in the Silver EDA Profile truth record. The parent truth record is loaded by `load_parent_truth_record_from_dataframe()`, and the `FEATURE_REGISTRY_PATH` is extracted from it for use in the next section.

This step must run on the unfiltered `silver_eda_dataframe` — before any row filtering — to ensure the hash reflects the same data that was produced upstream.

**Key variables:** `SILVER_TRUTH_HASH`, `silver_truth`, `FEATURE_REGISTRY_PATH`, `SILVER_DATASET_NAME`.

---

### Load Feature Registry

Cell 36 loads the feature registry JSON from `FEATURE_REGISTRY_PATH` and extracts `FEATURE_COLUMNS` — the list of sensor/feature column names that should be used for all EDA computations. Missing or empty feature registries raise `ValueError`. Any feature columns listed in the registry but absent from the loaded dataframe are also flagged here.

**Key variables:** `feature_registry`, `FEATURE_COLUMNS`.

---

### Robust Scaling Helpers and State Constants

Cell 39 defines `robust_center_scale()` (returns median and MAD-based scale, falling back to std if MAD is zero) and `robust_abs_z()` (robust Z-score). These helpers are used throughout the notebook wherever sensor comparisons need to be outlier-resistant.

Cell 41 defines the string constants for the source state column (`machine_status__synthetic`) and profiled state column (`machine_status__profiled`), and the allowed values for each: `SOURCE_NORMAL_VALUE = "normal"` and profiled values `normal_clean`, `normal_contaminated`, `abnormal`, `recovery`. Boolean masks for each profiled state are computed here and reused throughout later cells.

**Key variables:** `STATE_COL_SOURCE`, `STATE_COL_PROFILED`, `mask_profiled_normal_clean`, `mask_profiled_normal_contaminated`, `mask_profiled_abnormal`, `mask_profiled_recovery`.

---

### Profiled Data Overview

Cell 43 assembles a compact `overview_summary_df` DataFrame with dataset name, total row count, column count, feature count, meta column count, and the row count for each profiled state. This is the first notebook checkpoint confirming that the profiled input loaded with the expected structure before any analysis begins.

**Key outputs:** `overview_summary_df`.

---

### Missingness and Duplicate Checks

Cell 45 builds a full-dataframe missingness table (`missingness_df`) sorted by descending missing percentage. Cell 47 checks duplicate keys using `DUPLICATE_KEY_CANDIDATES` — combinations of `meta__asset_id`, `time_index`, `event_step`, `event_time`, and `meta__record_id` — and records both full-row duplicate counts and per-key-combination duplicate counts.

**Key outputs:** `missingness_df`, duplicate check rows.

---

### Numeric Feature Summaries

Cell 49 iterates over `FEATURE_COLUMNS` and builds `numeric_profile_df` — a per-feature table with row count, non-null count, missing count, mean, std, min, max, median, IQR, skewness, kurtosis, and quantiles at p01/p05/p25/p75/p95/p99. This is the broadest feature-level profile and is referenced later for generator inputs and the dropped sensor review.

**Key outputs:** `numeric_profile_df`, `numeric_columns`, `categorical_columns`.

---

### Source State Counts

Cell 51 produces `source_state_counts_df` — the row count and row percentage for each value in `machine_status__synthetic`. This preserves visibility into the original source-label distribution alongside the profiled labels.

---

### State-Level Sensor Profile Builder

Cell 53 defines `build_state_sensor_profile_table()`, which iterates over each profiled state group and each sensor column to produce per-state per-sensor summary rows (count, missingness, median, MAD, mean, std, min, max, IQR, skewness). This helper is called later for normal-clean, abnormal, and recovery subsets to create the generator-input profiles and the dropped sensor profile tables.

---

### Build Profiled-State Subsets

Cell 55 creates `PROFILED_STATE_SUBSETS`, a dictionary mapping state names to `.copy()` slices of `silver_eda_dataframe`. Explicit `.copy()` calls prevent accidental mutations to the source dataframe during the per-state analysis that follows.

**Key variable:** `PROFILED_STATE_SUBSETS` (keys: `normal_clean`, `normal_contaminated`, `abnormal`, `recovery`).

---

### Dropped and Weak Sensor Review

Cell 57 derives `dropped_sensor_review_df` from `numeric_profile_df`, adding `is_all_missing` (non-null count is zero) and `is_constant_or_near_constant` (std <= 1e-12) flags. This documents which sensors are structurally absent or degenerate before any generator input decisions are made.

---

### Episode-Level State Summaries

Cells 59 and 61 build episode-level summaries for source states and profiled states respectively. Each groups by `meta__episode_id` and the relevant state column, computing row counts and percentages per episode-state combination. These summaries expose whether a small number of episodes dominate the dataset and provide the episode context needed for onset alignment and synthetic generation planning.

**Key outputs:** `episode_source_counts_df`, `episode_profiled_counts_df`.

---

### Missingness by State (Source and Profiled)

Cells 63 and 65 build per-state per-feature missingness tables for both the source-state dimension and the profiled-state dimension. These tables reveal whether missingness is correlated with specific operational periods rather than being uniformly distributed, which matters for imputation design and synthetic generation.

**Key outputs:** `missingness_by_source_state_df`, `missingness_by_profiled_state_df`.

---

### State Transition and Dwell Tables

Cell 67 defines `build_state_transition_tables()`, which builds a from-state/to-state transition count table and a consecutive-run dwell summary for a given state column. It is called for both the source state column and the profiled state column. Dwell times reveal the typical length of normal, abnormal, and recovery periods, and transition tables document how source labels map into profiled labels.

**Key outputs:** `source_transition_counts_df`, `profiled_transition_counts_df`, `source_dwell_summary_df`, `profiled_dwell_summary_df`.

---

### Correlation Artifacts (Normal-Clean, Contaminated, Abnormal)

Cell 69 computes the Pearson correlation matrix for the `normal_clean` subset over `FEATURE_COLUMNS` and saves it to `correlation_analysis/correlation_matrix_normal_clean.csv`. It also builds a long-form upper-triangle sensor pair table (`clean_corr_pairs`). This normal-clean matrix becomes the reference for sensor grouping and generator input.

Cell 72 repeats the process for the `normal_contaminated` subset and saves the result to `correlation_matrix_normal_contaminated.csv`.

Cell 76 repeats for the `abnormal` subset and constructs `fault_pairings_df` by joining clean and abnormal correlation pairs to identify sensor relationships that change under fault conditions.

**Key artifacts:** `correlation_matrix_normal_clean.csv`, `correlation_matrix_normal_contaminated.csv`, sensor pair CSVs, `fault_pairings_df`.

---

### Sensor Grouping from Correlation Structure

Cell 74 builds sensor subsystem groups using a threshold (`SUBSYSTEM_CORR_THRESHOLD = 0.80`) on the absolute normal-clean correlation matrix. A connected-component traversal identifies groups of sensors that are highly correlated with each other. Cell 101 extends this by converting the correlation matrix to a distance matrix using `distance = 1.0 - |r|` and running `AgglomerativeClustering` (metric=`precomputed`, linkage=`average`, `FEATURE_CLUSTER_COUNT = 8`) to produce `feature_cluster_map_df`. The distance-based approach ensures that highly correlated sensors have small distances and cluster together.

**Key variables:** `adjacency_map`, `components` (correlation-based groups), `feature_cluster_map_df` (agglomerative clusters).

---

### Generator Input Artifacts

Cells 77–84 build and write the structured input files consumed by the synthetic generator. `GENERATOR_INPUT_DIR` is set to `artifacts/silver/<dataset>/eda/generator_inputs/`. `infer_distribution_family()` classifies each sensor's distribution as `near_constant`, `normal`, `skewed`, or `heavy_tailed` based on std, IQR, skewness, and kurtosis.

Cell 80 loads the dropped sensor registry from the PreEDA feature registry — not from `silver_eda_dataframe` directly — because some dropped sensors (e.g. `sensor_15`, all-null) should not appear in synthetic data at all, while others (e.g. `sensor_50`) should be generated and then missingness-masked. This distinction requires reading the PreEDA source rather than inferring from the current dataframe.

`build_state_median_lookup()` (cell 82) produces a per-state per-sensor median table used by the generator for state-conditional value targeting.

Cell 84 saves the subsystem/fault/hotspot artifacts to the correlation analysis directory: `sensor_hotspot_summary_profiled_states.csv`, source and profiled transition/dwell CSVs.

**Key outputs:** Generator-input profile CSVs, `generator_input_manifest.json`, `dropped_sensor_registry.json`, `EPISODE_STATUS_EXPORT_PATH` path defined here.

---

### Profiled State Summaries and Robust Comparison

Cell 86 defines `summarize_profiled_state()`, which returns a compact dict for each state with row count, row percentage, mean sensor missingness, and max sensor missingness. Cell 88 defines `robust_state_comparison_vs_clean()`, which computes per-feature robust Z-scores of each profiled state's median relative to the `normal_clean` reference distribution. The result, `robust_state_compare_top_df`, ranks features by their maximum deviation from normal-clean and drives all downstream top-N sensor visualizations.

**Key outputs:** `robust_state_compare_top_df`, `profiled_state_summary_rows`.

---

### Distribution Plots and Timeline Overlays

Cell 90 selects the top `TOP_HEATMAP_SENSOR_COUNT = 20` shifted sensors and renders a seaborn heatmap of their normal-clean correlations. Cell 94 selects the top `TOP_DISTRIBUTION_SENSOR_COUNT = 6` sensors and saves KDE or histogram overlay plots per state to the distribution plot directory. Cell 97 selects the top `TOP_TIMELINE_SENSOR_COUNT = 4` sensors and renders timeline overlay plots, subsampling to 25,000 rows if the dataframe is larger. All plot paths are logged via the ledger.

---

### Aligned Anomaly-Onset Windows

Cell 99 builds `aligned_onset_df` by identifying the first abnormal row within each episode, then collecting `ALIGN_PRE_STEPS = 50` rows before and `ALIGN_POST_STEPS = 50` rows after that position. An `aligned_step` column (centered at 0) enables cross-episode alignment. Cell 103 uses this aligned dataframe to render per-sensor median curves by profiled state across the aligned window, saving plots to the aligned onset plot directory.

**Key variables:** `ALIGN_PRE_STEPS`, `ALIGN_POST_STEPS`, `aligned_onset_df`.

---

### PCA Diagnostics

Cells 106–108 run a two-component PCA diagnostic on the feature set. The source dataframe is downsampled to `PCA_SAMPLE_N = 20000` rows and missing values are median-imputed before scaling (RobustScaler by default). PCA is fit with `PCA_N_COMPONENTS = 2`. Outputs include explained variance ratios, cumulative explained variance, and a per-feature loading table (`loading_df`) sorted by maximum absolute loading. A scatter plot of PC1 vs PC2 colored by profiled state is saved to the PCA artifact directory. These outputs show whether profiled states separate in reduced space without using PCA as a modeling step.

**Key outputs:** `pca_explained_variance_df`, `loading_df`, PC scatter plot.

---

### Imputation Comparison

Cell 110 compares two imputation strategies on a `IMPUTE_COMPARE_SAMPLE_N = 10000` row sample: (1) global median imputation, and (2) within-episode forward-fill followed by backfill and median fallback. The comparison documents which strategy disturbs feature distributions less and provides a reference for Gold preprocessing decisions.

---

### Outlier Audit by Profiled State

Cell 112 trains an `IsolationForest` on the `normal_clean` subset (up to `OUTLIER_AUDIT_SAMPLE_N = 20000` rows, contamination=0.05) scaled with `RobustScaler`, then scores the full dataset. Per-state outlier rates are aggregated to show whether abnormal and recovery rows score differently from clean normal rows. IQR-based outlier counts are also computed per feature for the SQL export.

---

### Episode Status Counts Export

Cell 114 aggregates profiled state row counts per episode into `episode_status_counts.json` at `EPISODE_STATUS_EXPORT_PATH`. For each episode, the export records the count of `normal_clean`, `normal_contaminated`, `abnormal`, and `recovery` rows. The synthetic generator reads this file to reproduce realistic episode structures. If `meta__episode_id` is absent from the dataframe, a single fallback episode covering all rows is used.

---

### Generator Input Manifest Update

Cell 116 updates the generator input manifest JSON with paths to the episode status counts file and the dropped sensor profile files. The manifest is reloaded from disk if it is no longer in memory. Required path variables are validated before the manifest is written back.

---

### Build Truth Record and Save Final Profiled EDA Outputs

Cell 118 assembles `silver_eda_profile_summary` — a dictionary capturing profiled state counts, feature count, row counts per state, and all artifact paths — and saves it as the Silver EDA Profile truth record via `save_truth_record()`. The truth record is written with `parent_truth_hash = SILVER_TRUTH_HASH` so the lineage chain connects this EDA output back to the upstream Silver layer record.

Cell 120 builds `eda_artifact_index_df`, a structured DataFrame cataloguing each named artifact path alongside a human-readable description. This index is saved as a CSV and passed to the SQL write in the QA section.

**Key outputs:** Silver EDA Profile truth record (with `parent_truth_hash`), `eda_artifact_index_df`.

---

### QA / SQL Staging and Write

Cells 122–138 build the staging DataFrames used to populate the Silver EDA metadata tables in PostgreSQL:

- `profile_df` — one-row dataset-level profile (row count, column counts, memory usage, duplicate count).
- `feature_statistics_df` — per-feature descriptive statistics at multiple percentiles.
- `missingness_summary_df` — per-feature null count and null percentage.
- `correlation_pairs_df` — upper-triangle Pearson correlation pairs for all numeric columns.
- `outlier_summary_df` — per-feature IQR-fence outlier counts.
- `categorical_distribution_df` — per-category counts and percentages for categorical columns.

Cell 137 gates all SQL writes behind `WRITE_SILVER_EDA_SQL_OUTPUTS = True`. When the flag is `True`, `write_silver_eda_sql_outputs()` is called with all staging DataFrames, writing to the target schema tables. When `False`, the notebook completes without touching the database.

---

## Key Outputs

- `correlation_analysis/correlation_matrix_normal_clean.csv` — Pearson correlation matrix for normal-clean sensor rows.
- `correlation_analysis/correlation_matrix_normal_contaminated.csv` — Correlation matrix for normal-contaminated rows.
- `correlation_analysis/sensor_hotspot_summary_profiled_states.csv` — Sensor subsystem and hotspot grouping summary.
- `correlation_analysis/source_state_transition_counts.csv` and `profiled_state_transition_counts.csv` — State transition counts.
- `correlation_analysis/source_state_dwell_summary.csv` and `profiled_state_dwell_summary.csv` — Consecutive-run dwell summaries.
- `generator_inputs/episode_status_counts.json` — Per-episode profiled state row counts consumed by the synthetic generator.
- `generator_inputs/dropped_sensor_registry.json` — Registry of dropped and missingness-masked sensors.
- `generator_inputs/generator_input_manifest.json` — Manifest of all generator input artifact paths.
- Feature profile CSVs for `normal_clean`, `abnormal`, and `recovery` subsets.
- `pca/` — Explained variance CSV, loading table, and PC scatter plot.
- `distribution_plots/` — Per-sensor KDE/histogram overlays by profiled state.
- `aligned_onset_plots/` — Per-sensor median curves aligned around anomaly onset.
- Silver EDA Profile truth record (written to `truths/silver/`) linking to the upstream Silver truth hash via `parent_truth_hash`.
- Artifact index CSV cataloguing all output paths.
- SQL table rows in `silver.eda_dataset_profile`, `silver.eda_feature_statistics`, `silver.eda_missingness_summary`, `silver.eda_correlation_pairs`, `silver.eda_outlier_summary`, `silver.eda_categorical_distribution`, `silver.eda_artifact_index`.

---

## Dependencies and Inputs

| Input | Description |
|---|---|
| `<SILVER_TRAIN_DATA_PATH>/<dataset>__silver_subsets__profiled_dataframe.parquet` | Main input: profiled Silver dataframe from Silver 02a with `machine_status__profiled` column. Falls back to rglob search in Silver EDA artifact dirs if canonical path is absent. |
| Feature registry JSON (path resolved from parent Silver truth record) | Defines the canonical `FEATURE_COLUMNS` list for all EDA computations. |
| Parent Silver truth record (in `truths/silver/`) | Supplies `FEATURE_REGISTRY_PATH`, `parent_truth_hash`, and dataset identity. |
| `utils/core/` and `utils/medallion/silver/` utilities | Context loading, truth utilities, logging, ledger, file I/O, SQL writers. |
| PostgreSQL (via `engine` from `get_engine_from_env()`) | SQL smoke check on startup; SQL writes at end under `WRITE_SILVER_EDA_SQL_OUTPUTS` gate. |
| Environment variables: `CAPSTONE_SCHEMA`, PostgreSQL connection vars | Required for SQL operations. |

---

## SQL / Database Operations

| Operation | Target Tables | Gate |
|---|---|---|
| Smoke check on `information_schema.tables` | Read-only | Always runs |
| Write dataset-level profile | `silver.eda_dataset_profile` | `WRITE_SILVER_EDA_SQL_OUTPUTS = True` |
| Write per-feature statistics | `silver.eda_feature_statistics` | `WRITE_SILVER_EDA_SQL_OUTPUTS = True` |
| Write missingness summary | `silver.eda_missingness_summary` | `WRITE_SILVER_EDA_SQL_OUTPUTS = True` |
| Write correlation pairs | `silver.eda_correlation_pairs` | `WRITE_SILVER_EDA_SQL_OUTPUTS = True` |
| Write outlier summary | `silver.eda_outlier_summary` | `WRITE_SILVER_EDA_SQL_OUTPUTS = True` |
| Write categorical distribution | `silver.eda_categorical_distribution` | `WRITE_SILVER_EDA_SQL_OUTPUTS = True` |
| Write artifact index | `silver.eda_artifact_index` | `WRITE_SILVER_EDA_SQL_OUTPUTS = True` |

All Silver EDA SQL writes are executed through `write_silver_eda_sql_outputs()` in a single call at cell 137. Setting `WRITE_SILVER_EDA_SQL_OUTPUTS = False` performs a complete dry run without writing to the database.

---

## Important Behavioral Notes

**Parent truth hash capture before filtering.** `SILVER_TRUTH_HASH` is extracted from `silver_eda_dataframe` before any row filtering or subsetting. This guarantees that the hash in the Silver EDA Profile truth record matches the full upstream Silver output, not a filtered slice.

**State subset copy pattern.** `PROFILED_STATE_SUBSETS` uses explicit `.copy()` calls when slicing by profiled state. This prevents mutation of `silver_eda_dataframe` during per-state analysis and keeps the source dataframe stable for operations that span multiple states.

**Fallback path for profiled dataframe.** If the canonical Silver 02a parquet path does not exist, the notebook searches Silver EDA artifact directories recursively using `rglob()`. This allows the notebook to run during development without requiring the full upstream pipeline to have written to the exact canonical location.

**Dropped sensor sourcing from PreEDA.** The dropped sensor registry is built from the PreEDA feature registry, not from `silver_eda_dataframe`. Sensors that are all-null should not appear in synthetic data at all; sensors that are present but missingness-masked should be generated and then selectively nulled. Inferring this from the current dataframe alone would collapse both cases into a single "missing column" result.

**Correlation-to-distance conversion for agglomerative clustering.** Feature clustering uses `distance = 1.0 - |r|` computed from the normal-clean correlation matrix. This converts high correlation into small distance so `AgglomerativeClustering` with a precomputed distance metric groups correlated sensors together rather than apart.

**W&B gate.** `USE_EXPERIMENT_TRACKING = False` is set explicitly near the top of the notebook. The flag can be flipped to enable W&B, but the default keeps the notebook runnable without an active tracking connection.

**SQL write gate.** `WRITE_SILVER_EDA_SQL_OUTPUTS = True` in cell 137 controls all PostgreSQL writes in the QA section. Setting it to `False` produces a complete dry run with no database writes.

**Episode fallback for status counts export.** If `meta__episode_id` is absent when building `episode_status_counts.json`, a single synthetic episode covering all rows is used. This prevents the export from failing on datasets that omit episode metadata.

**Truth directory contract.** The Silver EDA Profile truth record is saved using `save_truth_record(..., truth_dir=TRUTHS_PATH, layer_name="silver")`. The utility appends the layer subfolder internally. Passing a pre-nested path such as `TRUTHS_PATH / "silver" / "eda"` would create a double-nested directory and break truth resolution in downstream notebooks.
