# Notebook Code Reference: EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3

Notebook path: `notebooks/eda/EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3.ipynb`

## Notebook Purpose

Silver 02a builds the profiled Silver subset used to separate clean normal behavior from normal rows
that may already be contaminated by early fault onset. It loads the Silver Pre-EDA parquet output
produced by Silver 01, profiles normal episodes by windowing and sensor-quality scoring, and assigns
a `machine_status__profiled` label to every row — yielding `normal_clean`, `normal_contaminated`,
`abnormal`, and `recovery` subsets. The resulting parquet files and truth record are the primary
inputs for Silver 02b EDA and Gold preprocessing.

Notebook stage: `Silver`

---

## Section Overview

| Section | Cells (approx.) | Description |
|---|---|---|
| Imports and Environment Setup | 3–6 | Standard library, project utility, and display imports |
| Pipeline Context and Configuration | 7–11 | `load_notebook_context`, pipeline variables, sanity checks |
| Silver EDA Artifact Directories | 13–15 | Canonical artifact path constants for all Silver EDA outputs |
| SQL Runtime Context | 16–20 | Postgres engine, schema, dataset/run ID resolution, smoke-check query |
| Logging and Ledger Initialization | 21–30 | Logger activation, preserved original-setup blocks (triple-quoted), ledger checkpoint |
| W&B Guard (disabled) | 25 | `wandb.init` block preserved in triple-quotes, not executed |
| Load Silver Pre-EDA Output | 31–34 | Parquet load with fallback, parent truth hash capture |
| Feature Registry and Sensor Columns | 35–36 | Load feature registry JSON, resolve `SENSOR_COLUMNS` |
| Source State Normalization | 37–40 | Detect state column, normalize labels to `normal/abnormal/recovery` |
| Normal Episode Assignment and Edge Trimming | 41–48 | Assign episode IDs to normal runs, trim leading/trailing 10% |
| Episode Windowing | 48–50 | Split each trimmed episode into fixed-count windows |
| Window Sensor Statistics | 50–52 | Compute per-sensor stats (median, IQR, delta quantiles) per window |
| Window Quality Scoring and Filtering | 52–55 | Score and rank windows; classify `keep` vs `drop` |
| Sensor Baseline Build | 55–57 | Aggregate kept-window stats into `final_sensor_baseline_df` |
| Run Normal Windowing Workflow | 57 | Execute the episode → trim → window → stats → score → baseline pipeline |
| Baseline Inspection and Checkpoints | 58–65 | Review lowest-scoring windows, quality distribution, baseline shape |
| Save Sensor Baseline Artifact | 66–68 | Write CSV and JSON sensor baseline profile to `SENSOR_PROFILE_DIR` |
| Sensor Profile Plot Helpers | 69–73 | `display_sensor_profile` helper, spot-check individual sensors |
| Row-Level Quality Scoring Configuration | 74–75 | Deviation threshold constants, scoring config display |
| Row Scoring and Quality Classification | 76–80 | `score_rows_against_sensor_baseline`, `classify_normal_training_quality` definitions |
| Apply Row-Level Scoring | 81–87 | Run scoring and classification on `normal_profile_df` |
| Build Final Profiled-State Dataframe | 88–90 | Construct `silver_subset_df` with row quality columns |
| Normal-Only Review and Profile Plots | 91–102 | Normal-only subset inspection, baseline vs sensor overlay plots |
| Define Final Profiled-State Mapping | 103–104 | Standardize profiled label constants, rebuild `silver_subset_df` |
| Save Profiled Subset Artifacts | 105–107 | Write parquet data outputs and JSON summary |
| Finalize Silver EDA Subsets Truth Record | 108–110 | Initialize, populate, build, save, and index the truth record |
| Final Structure Review | 110–111 | `silver_subset_df.info()` checkpoint |
| SQL Write Gate | 112–113 | `WRITE_TO_POSTGRES` flag; log to pipeline SQL tables |
| Summary | 114 | Next-stage handoff notes |

---

## Section Details

### Imports and Environment Setup

Cells 3–6 import the standard library, NumPy, Pandas, Matplotlib/Seaborn, and the full set of
project utilities used across the notebook. Utilities drawn from `utils.core` cover paths, file I/O,
logging, the run ledger, truth-record operations, config loading, and artifact directory helpers.
Database utilities from `utils.database` cover the Postgres engine, layer read/write helpers, SQL
notebook helpers, and the medallion SQL writer functions. `load_notebook_context` from
`utils.core.notebook_context` is the shared pipeline context entry point. Cell 5 defines
`cfg_require_mapping`, a guard used when accessing config sub-blocks that must be dictionaries.

---

### Pipeline Context and Configuration

Cells 7–11 establish the shared pipeline context. `load_notebook_context` is called with
`stage="silver_eda"`, `dataset="pump"`, `mode="train"`, and `profile="default"`. It returns a
context object (`CTX`) from which the notebook unpacks aliases for paths, config maps, stage config,
resolved paths, filenames, versions, runtime config, dataset config, W&B config, and pipeline config.
These aliases (`paths`, `CONFIG`, `STAGE_CFG`, `SILVER_EDA_CFG`, `PIPELINE`, etc.) are the only
configuration interface used throughout the rest of the notebook.

Cell 9 resolves the primary pipeline constants: `DATASET_NAME`, `TRUTH_CONFIG` (with the `PIPELINE`
block attached so truth records carry execution-mode metadata), `STAGE`, `LAYER_NAME`,
`SILVER_VERSION`, `TRUTH_VERSION`, `PIPELINE_MODE`, `RUN_MODE`, `SILVER_PROCESS_RUN_ID`, and
`SILVER_SUBSET_PROCESS_RUN_ID`. `ASSET_ID` and `RUN_ID` are resolved in priority order:
environment variable → dataset config → `DEFAULT_FALLBACKS_CFG` → hard-coded default string.

Cells 10–11 run two sanity checks. Cell 10 verifies all required shared context variables are
present in globals and raises `NameError` listing any missing names. Cell 11 checks that
`SILVER_EDA_CFG` is available. Both checks log a pass message and add a ledger entry on success.

**Key variables:** `CTX`, `paths`, `CONFIG`, `SILVER_EDA_CFG`, `PIPELINE`, `PIPELINE_MODE`,
`DATASET_NAME`, `SILVER_PROCESS_RUN_ID`, `SILVER_SUBSET_PROCESS_RUN_ID`, `ASSET_ID`, `RUN_ID`,
`TRUTH_VERSION`, `SILVER_VERSION`.

---

### Silver EDA Artifact Directories

Cell 14 establishes the canonical Silver EDA artifact directory tree rooted at
`artifacts/silver/<dataset>/eda/`. Sub-directories for aligned onset plots, config snapshots,
correlation analysis, distribution plots, generator inputs, lineage, metadata, PCA, sensor profiles,
subsets, summary, and window profiles are all declared here as constants. Keeping these declarations
in one cell ensures that all output paths across the notebook are derived from the same root, making
the directory structure auditable and reproducible across runs.

**Key variables:** `SILVER_EDA_ARTIFACT_DIR`, `SILVER_EDA_SENSOR_PROFILES_DIR`,
`SENSOR_PROFILE_DIR`, `SILVER_EDA_SUMMARY_DIR`.

---

### SQL Runtime Context

Cells 16–20 create the Postgres engine via `get_engine_from_env()` and resolve `CAPSTONE_SCHEMA`
from the environment (default `"capstone"`). A `first_non_empty_string` helper resolves `DATASET_ID`
and `RUN_ID` from the environment, dataset config, and fallback values, skipping None, empty strings,
and dict objects. Cell 19 runs a read-only smoke-check query against `information_schema.tables` to
confirm connectivity and displays the result as a checkpoint before any write operations.

**Key variables:** `engine`, `CAPSTONE_SCHEMA`, `DATASET_ID`, `RUN_ID`.

---

### Logging and Ledger Initialization

Cells 21–30 activate the logger and ledger for the Silver subset stage. Cell 22 calls
`log_layer_paths` to write the resolved project path tree to the log at startup. Cell 23 preserves
the original standalone `configure_logging(...)` call in a triple-quoted string; it is not executed
because logging is already configured by `load_notebook_context`. Cell 28–29 similarly preserves the
original `Ledger(...)` initialization in a visible checkpoint comment. These blocks serve as
documentation of the original setup sequence and are kept to make the migration from standalone setup
to shared context visible.

---

### W&B Guard (disabled)

Cell 25 preserves the `wandb.init(...)` call in triple-quoted string form so it is inert at runtime.
The block is retained for future activation if W&B experiment tracking is re-enabled. A comment above
the block states: "W&B integration is disabled; block is preserved in triple-quotes for future
activation." No W&B run is created during normal execution of this notebook.

---

### Load Silver Pre-EDA Output

Cells 31–34 load the Silver parquet produced by Silver 01. Cell 32 first checks for the canonical
file at `SILVER_TRAIN_DATA_PATH / SILVER_TRAIN_DATA_FILE_NAME`. If that path does not exist, it
globs for any `.parquet` file in the directory and uses the first candidate alphabetically, logging a
warning if multiple files are found. The loaded dataframe is assigned to `silver_eda_df`.

Cell 34 immediately extracts `SILVER_TRUTH_HASH` from `silver_eda_df` via `extract_truth_hash`
**before any filtering or transformation**. This timing is intentional: the hash must reflect the
full unmodified Silver output so that the lineage link in the downstream truth record is valid even
if the subset build process drops rows. If the hash cannot be resolved, the cell raises `ValueError`
and halts. The parent Silver truth record is then loaded via `load_parent_truth_record_from_dataframe`
to retrieve the feature registry path for the next section.

**Key variables:** `silver_eda_df`, `SILVER_TRUTH_HASH`, `silver_truth`, `FEATURE_REGISTRY_PATH`.

---

### Feature Registry and Sensor Columns

Cells 35–36 load the feature registry JSON identified from the parent Silver truth record. The
registry is validated as a non-None dict before use. `SENSOR_COLUMNS` is resolved from the
registry's `feature_columns` list. If the registry provides no sensor columns, the notebook falls
back to detecting columns whose names start with `sensor_` in the Silver dataframe, logging a warning
so the gap is visible in the run log.

**Key variables:** `feature_registry`, `SENSOR_COLUMNS`.

---

### Source State Normalization

Cells 37–40 detect the source machine-status column by probing for `machine_status__synthetic`,
`machine_status`, `status`, and `state` in that priority order. The detected column name is stored as
`STATE_COL_SOURCE_RAW`; the canonical output alias `STATE_COL_SOURCE` is fixed to
`"machine_status__synthetic"`. A `normalize_machine_state` function maps raw label variants (e.g.
`"ok"`, `"fault"`, `"cooldown"`) to the three normalized values `"normal"`, `"abnormal"`, or
`"recovery"`, which are written to a new column in `silver_eda_df`. Cell 40 displays post-
normalization value counts as a review checkpoint.

**Key variables:** `STATE_COL_SOURCE`, `STATE_COL_SOURCE_RAW`.

---

### Normal Episode Assignment, Edge Trimming, and Windowing

Cells 41–57 build a clean normal reference by profiling contiguous runs of normal-labeled data
through a four-stage pipeline.

**Episode assignment (`assign_normal_episodes`, cell 45):** Detects transitions between normal and
non-normal rows and assigns a monotonically incrementing `__normal_episode_id` to each contiguous
normal run using a cumulative-sum approach on the status-change mask.

**Edge trimming (`trim_normal_episode_edges`, cell 47):** Removes the first and last `TRIM_FRAC`
(default 10%) of each normal episode to discard rows that may capture fault-onset or fault-recovery
transients near episode boundaries. Episodes too short to survive trimming while still reaching
`min_rows_after_trim` (1000 rows) are dropped entirely.

**Windowing (`create_episode_windows`, cell 49):** Splits each trimmed episode into
`WINDOWS_PER_EPISODE` equal segments (default 5). Windows shorter than `MIN_WINDOW_ROWS` are
discarded. Each window receives a string label in the form `episode_<N>__window_<M>`.

**Window statistics (`calculate_window_sensor_stats`, cell 51):** For every sensor in every window,
computes median, mean, std, IQR, 5th/95th percentile values, delta IQR, and `delta_abs_q95` (95th
percentile of absolute row-to-row change). These statistics are the basis for both window quality
scoring and the final sensor baseline.

**Window quality scoring (`score_and_filter_windows`, cell 53):** Aggregates per-window statistics
and ranks windows by IQR magnitude, delta magnitude, and missing-data rate. The combined
`window_quality_score` classifies windows as `keep` (top `KEEP_WINDOW_FRAC`, default 80%) or `drop`.

**Sensor baseline build (`build_final_sensor_baseline`, cell 55):** Aggregates per-sensor statistics
from the kept windows only, producing `final_sensor_baseline_df` — the learned stable-normal sensor
profile for this dataset.

**Workflow execution (cell 57):** Applies all five functions in sequence on `silver_eda_df` to
produce `normal_profile_df`, `window_sensor_stats_df`, `window_quality_df`,
`final_sensor_baseline_df`, and `kept_window_stats_df`.

**Key variables:** `LABEL_COLUMN`, `NORMAL_VALUES`, `TRIM_FRAC`, `WINDOWS_PER_EPISODE`,
`MIN_WINDOW_ROWS`, `KEEP_WINDOW_FRAC`, `normal_profile_df`, `window_quality_df`,
`final_sensor_baseline_df`.

---

### Save Sensor Baseline Artifact

Cell 67 writes two sensor baseline artifacts to `SENSOR_PROFILE_DIR`:

- `silver_sensor_baseline_profiles.csv` — the baseline dataframe in tabular form.
- `silver_sensor_baseline_profiles.json` — a structured JSON artifact with metadata (source notebook,
  profile method, state column, normal values used, baseline config) and a per-sensor profile dict.

These files capture the learned normal profile independently of the downstream subset outputs so they
can be reviewed, reused by other notebooks, or fed into the synthetic generator without re-running
the full subset-build workflow.

---

### Row-Level Quality Scoring

Cells 74–81 score every row in `normal_profile_df` against the learned sensor baseline to identify
which normal-labeled rows are genuinely stable and which show early deviation.

**Configuration (cell 75):** The scoring parameters `VALUE_DEVIATION_THRESHOLD` and
`DELTA_DEVIATION_THRESHOLD` (both expressed as IQR multiples), `SUSPECT_SENSOR_COUNT`, and
`EXCLUDE_SENSOR_COUNT` are displayed in a summary dataframe for review.

**`score_rows_against_sensor_baseline` (cell 77):** For each sensor in each normal row, measures
value deviation relative to the baseline IQR and delta deviation relative to `baseline_delta_abs_q95`.
Per-row counts of sensors exceeding each threshold are accumulated in
`normal_value_abnormal_sensor_count`, `normal_delta_abnormal_sensor_count`, and
`normal_total_abnormal_sensor_count`.

**`classify_normal_training_quality` (cell 79):** Assigns a `normal_training_quality_class` to each
row. Non-normal rows receive `"not_normal"`. Normal rows default to `"clean"`; elevated to
`"suspect"` when `normal_total_abnormal_sensor_count >= SUSPECT_SENSOR_COUNT`; elevated to
`"exclude"` when the count meets `EXCLUDE_SENSOR_COUNT`. An `is_clean_normal_for_training` boolean
column is also added for direct downstream use.

**Application (cell 81):** Both functions are applied to `normal_profile_df` to produce
`scored_normal_quality_df`.

**Key variables:** `VALUE_DEVIATION_THRESHOLD`, `DELTA_DEVIATION_THRESHOLD`,
`SUSPECT_SENSOR_COUNT`, `EXCLUDE_SENSOR_COUNT`, `scored_normal_quality_df`.

---

### Build Final Profiled-State Dataframe and Label Standardization

Cells 88–104 construct the final profiled dataframe.

Cell 89 copies `scored_normal_quality_df` to `silver_subset_df` and adds `final_row_quality_class`,
`row_is_clean_normal`, `row_is_suspect_normal`, and `row_is_exclude_from_normal_training` boolean
columns.

Cell 104 declares the canonical profiled-state label constants:

| Constant | Value |
|---|---|
| `PROFILED_NORMAL_CLEAN_VALUE` | `"normal_clean"` |
| `PROFILED_NORMAL_SUSPECT_VALUE` | `"normal_suspect"` |
| `PROFILED_NORMAL_CONTAMINATED_VALUE` | `"normal_contaminated"` |
| `PROFILED_ABNORMAL_VALUE` | `"abnormal"` |
| `PROFILED_RECOVERY_VALUE` | `"recovery"` |

`silver_subset_df` is rebuilt from `scored_normal_quality_df` (not mutated in place) and a
`machine_status__profiled` column is written with these standardized values for every row. This
column is the primary output label consumed by Silver 02b and Gold preprocessing.

**Key variables:** `STATE_COL_PROFILED`, `silver_subset_df`, `machine_status__profiled`.

---

### Save Profiled Subset Artifacts

Cell 106 writes the profiled subset data outputs to `SILVER_SUBSET_DATA_DIR` (resolves to
`SILVER_TRAIN_DATA_PATH`) and diagnostic artifacts to `SILVER_EDA_ARTIFACT_DIR`. Three parquet files
are written:

- `<dataset>__silver_subsets__profiled_dataframe.parquet` — full profiled dataframe for all rows.
- `<dataset>__silver_subsets__normal_clean.parquet` — rows where `machine_status__profiled == "normal_clean"`.
- `<dataset>__silver_subsets__normal_contaminated.parquet` — rows where `machine_status__profiled == "normal_contaminated"`.

A JSON summary file (`<dataset>__silver_subsets__summary.json`) is written to
`SILVER_EDA_SUMMARY_DIR` with profiled and source state counts, row counts, artifact paths, and run
metadata.

---

### Finalize Silver EDA Subsets Truth Record

Cell 109 initializes the Silver EDA Subsets truth record via `initialize_layer_truth` with
`layer_name="silver"`, `parent_truth_hash=SILVER_TRUTH_HASH`, and `SILVER_SUBSET_PROCESS_RUN_ID`.
The `parent_truth_hash` field carries the hash extracted from the raw Silver dataframe at load time,
establishing a verifiable lineage link from this subset back to the Silver Pre-EDA parent. The truth
record is populated in four sections:

| Section | Contents |
|---|---|
| `config_snapshot` | Stage, layer, dataset, pipeline mode, run mode |
| `runtime_facts` | Parent hash, state columns used, profiling method, profiled/source state counts, row count |
| `artifact_paths` | Paths for all three parquet outputs, summary JSON, data dir, artifact dir |
| `notes` | Human-readable purpose statement |

`build_truth_record` finalizes the record and computes `SILVER_EDA_SUBSETS_TRUTH_HASH`. The record
is saved via `save_truth_record` to `TRUTHS_PATH/silver/<dataset>__silver__truth__<hash>.json` and
indexed via `append_truth_index` so it participates in cross-notebook truth lookups. The ledger
records the step with both the new hash and the parent hash.

**Key variables:** `silver_eda_subsets_truth_record`, `SILVER_EDA_SUBSETS_TRUTH_HASH`,
`silver_eda_subsets_truth_path`.

---

### SQL Write Gate

Cell 113 is guarded by `WRITE_TO_POSTGRES = True`. When true, `log_silver_eda_sql` is called with
the active engine, schema, `DATASET_ID`, `RUN_ID`, and the notebook globals dict. This function
writes Silver EDA summary metadata to `capstone.pipeline_runs`, `capstone.data_quality_events`, and
`capstone.pipeline_artifacts`. Setting `WRITE_TO_POSTGRES = False` skips all database side effects
for dry-run execution without changing any other notebook behavior.

---

## Key Outputs

- `<dataset>__silver_subsets__profiled_dataframe.parquet` — full profiled dataframe with `machine_status__profiled` and row quality columns; written to `SILVER_TRAIN_DATA_PATH`
- `<dataset>__silver_subsets__normal_clean.parquet` — rows classified as `normal_clean`; written to `SILVER_TRAIN_DATA_PATH`
- `<dataset>__silver_subsets__normal_contaminated.parquet` — rows classified as `normal_contaminated`; written to `SILVER_TRAIN_DATA_PATH`
- `silver_sensor_baseline_profiles.csv` — learned stable-normal sensor profile table; written to `SENSOR_PROFILE_DIR`
- `silver_sensor_baseline_profiles.json` — JSON version of the sensor baseline with metadata
- `<dataset>__silver_subsets__summary.json` — subset run summary with state counts, artifact paths, and run metadata
- Silver EDA Subsets truth record — JSON file at `TRUTHS_PATH/silver/<dataset>__silver__truth__<hash>.json`
- Truth index entry — appended to `truth_index.jsonl`
- SQL metadata rows — in `capstone.pipeline_runs`, `capstone.data_quality_events`, and `capstone.pipeline_artifacts` (when `WRITE_TO_POSTGRES = True`)

---

## Dependencies and Inputs

| Input | Source | Notes |
|---|---|---|
| Silver Pre-EDA parquet | `SILVER_TRAIN_DATA_PATH / SILVER_TRAIN_DATA_FILE_NAME` | Produced by Silver 01. Fallback glob if canonical name is missing. |
| Feature registry JSON | Path resolved from parent Silver truth record | Must be a dict with a `feature_columns` list |
| Parent Silver truth record | `TRUTHS_PATH/silver/` | Loaded via `load_parent_truth_record_from_dataframe` from the Silver dataframe |
| Pipeline/stage config | `load_notebook_context` | Controls dataset name, versions, process run IDs, deviation thresholds |
| `DEFAULT_FALLBACKS_CFG` | Config sub-block | Used for `ASSET_ID` and `RUN_ID` resolution fallbacks |
| Postgres connection | Environment variables | Required for SQL logging; all DB writes skipped when `WRITE_TO_POSTGRES = False` |

---

## SQL / Database Operations

All SQL write operations in this notebook are gated by `WRITE_TO_POSTGRES = True` in cell 113.
When the gate is open, `log_silver_eda_sql` writes summary rows to three tables:

| Table | Purpose |
|---|---|
| `capstone.pipeline_runs` | Records the subset pipeline run metadata (stage, run ID, dataset ID, mode) |
| `capstone.data_quality_events` | Records data quality summary events from the Silver EDA subset stage |
| `capstone.pipeline_artifacts` | Records artifact paths for the profiled parquet outputs and summary file |

Cell 19 also runs a read-only smoke-check query against `information_schema.tables` at startup to
confirm Postgres connectivity before any write operations are attempted.

---

## Important Behavioral Notes

**Parent truth hash capture timing:** `SILVER_TRUTH_HASH` is extracted from `silver_eda_df`
immediately after load, before any filtering, normalization, or subsetting. This is intentional: the
hash must reflect the full unmodified Silver output so the lineage link in the truth record is valid
even when the subset build drops rows.

**Truth record lineage:** The Silver EDA Subsets truth record stores `parent_truth_hash` pointing to
the Silver Pre-EDA truth hash. This allows downstream notebooks and Gold-stage notebooks to trace any
subset artifact back to the originating Silver run by following the hash chain.

**W&B guard:** The `wandb.init` block is stored in a triple-quoted string and is never executed. This
is documented explicitly in the cell comment. No W&B run is started during normal notebook execution.

**Dry-run gate:** Setting `WRITE_TO_POSTGRES = False` in cell 113 suppresses all database writes
without affecting parquet, JSON, or truth-record outputs. The notebook is safe to re-run for
profiling and artifact updates without touching the operational SQL tables.

**Preserved original-setup blocks:** The original standalone `configure_logging(...)` and
`Ledger(...)` initialization blocks are preserved in triple-quoted strings in cells 23 and 29. They
are not executed; `load_notebook_context` handles both. They are kept to document the original setup
sequence before the shared context refactor.

**Sensor column fallback:** If the feature registry contains no `feature_columns`, the notebook falls
back to detecting columns beginning with `sensor_` in the Silver dataframe. This fallback preserves
execution continuity but emits a logger warning so the gap is visible in the run log.

**No mutation of input dataframe:** `silver_eda_df` is never modified in place. All transformations
produce new dataframe copies (`normal_profile_df`, `scored_normal_quality_df`, `silver_subset_df`).
The explicit copy in cell 104 ensures `silver_subset_df` does not alias `scored_normal_quality_df`
when profiled labels are assigned.

**Subset data directory:** `SILVER_SUBSET_DATA_DIR` resolves to `SILVER_TRAIN_DATA_PATH` rather than
a nested `subset_outputs/` subdirectory. The commented-out nested path alternative in cell 106 is
retained to document this design decision.
