# EDA_Notebook_Pump_Gold_05_Anomaly_Detection — Workflow Reference

**Source notebook:** `notebooks/experiments/EDA_Notebook_Pump_Gold_05_Anomaly_Detection.ipynb`
**Stage:** Gold — Anomaly Detection Analysis
**Layer:** Gold
**Reference type:** Workflow-level

---

## Notebook Purpose

Gold_05 is the anomaly timeline analysis and early-warning evaluation notebook. It does not train models, apply anomaly detection thresholds, or score raw sensor data. Its job is to take one scored output DataFrame (from Gold_02 or one of the Gold_03 cascade variants), annotate it with failure lifecycle phases, compute early-warning lead times, build a comprehensive visualization suite, export structured analysis artifacts, and log a summary to PostgreSQL.

Gold_05 is designed to run against any of the six available model runs:

| `SELECTED_RUN_KEY` | `target_flag_column` | `run_family` |
|---|---|---|
| `baseline` | `baseline_flag` | baseline |
| `cascade_defaults` | `cascade_final_flag` | cascade |
| `cascade_tuned` | `cascade_final_flag` | cascade |
| `stage3_improved` | `cascade_final_flag` | cascade |
| `stage3_medium` | `cascade_stage3_medium_flag` | cascade |
| `stage3_strict` | `cascade_stage3_strict_flag` | cascade |

The default `SELECTED_RUN_KEY` is `"stage3_improved"`. Changing it reconfigures the entire notebook — all analysis, summaries, plots, exports, and the truth record derive from whichever key is selected.

Gold_05 also produces a multi-run lead-time comparison across all six run keys, regardless of the selected primary run.

---

## Pipeline Role

- Stage: `gold_anomaly_detection`
- Layer: Gold
- Position in workflow: Runs after the Gold modeling notebooks (Gold_02 through Gold_03c); does not consume Gold_04 comparison outputs; before Gold_06A/06B
- Primary responsibility: Produce annotated anomaly timelines, early-warning lead-time summaries, visualization suites, and a structured detection summary for the selected model run; builds a multi-run lead-time comparison across all six model variants
- Does not train models, apply anomaly thresholds, or re-score raw sensor data
- No W&B run is opened; no truth record chains downstream to Gold_06A

## Configuration and Runtime Context

| Item | Source | Value / Purpose |
|---|---|---|
| `CONTEXT_STAGE` | Notebook constant | `"gold_anomaly_detection"` |
| `ANOMALY_DETECTION_CFG` | `CTX.stage_config` | Stage-specific config section for Gold_05 |
| `SELECTED_RUN_KEY` | Notebook constant (default `"stage3_improved"`) | Selects which scored output to analyze; reconfigures all analysis, exports, and truth record |
| `USE_GLOBAL_RESULTS_IF_AVAILABLE` | Notebook constant (default `True`) | When `True`, prefers in-memory globals over disk reload |
| `RECOVERY_STABILITY_ROWS` | Notebook constant (default `30`) | Minimum consecutive normal-like rows to mark recovery start |
| `RUN_CONFIG_MAP` | Notebook constant dict | Maps each of 6 run keys to `target_flag_column`, score column, decision column, `run_family` |
| `DATASET_ID`, `RUN_ID`, `ASSET_ID` | Env vars / config / `DATASET_CFG`; `first_non_empty_string` fallback chain | SQL write targets |
| `PIPELINE_MODE`, `RUN_MODE`, `CONFIG_PROFILE` | Config sections | Inherited from Gold_01 preprocessing mode |

## Section Overview

| Section | Purpose | Key Outputs |
|---|---|---|
| Bootstrap | Context, dirs, DB (no W&B run opened) | `CTX`, `engine`, `ANOMALY_DETECTION_CFG` |
| Run selection | `SELECTED_RUN_KEY`, `TARGET_FLAG_COLUMN`, plot constants | `RUN_CONFIG_MAP`, `TARGET_FLAG_COLUMN`, `RUN_FAMILY` |
| Upstream data load | Load scored results for selected run | `selected_results` DataFrame |
| Anomaly timeline | Annotate episodes, phases, detection classes | `anomaly_timeline_dataframe`, `recovery_boundary_payload` |
| Detection analysis | Alert packets, stage alerts dict, lead-time summary | `detection_summary_payload`, `alert_packet_summary_df` |
| Multi-run comparison | Full 6-run lead-time table and chart | `lead_time_comparison_df`, PNG |
| Sensor visualizations | Timeline, stacked waveform, heatmap, 3D surface | PNGs |
| Artifact exports | All analysis artifacts to disk | Parquet, CSVs, JSONs, PNGs |
| Truth record | Stamp and register `gold_anomaly_detection` truth | Truth JSON |
| SQL write | Log output manifest | `capstone.pipeline_runs`, `capstone.data_quality_events` |

## Section Details

## 2. Context Stage and Bootstrap

```python
CONTEXT_STAGE = "gold_anomaly_detection"

CTX = load_notebook_context(
    stage=CONTEXT_STAGE,
    dataset=CONTEXT_DATASET,
    mode=CONFIG_RUN_MODE,
    profile=CONFIG_PROFILE,
    logger_child_name="capstone.gold.anomaly_detection",
    log_filename="gold_anomaly_detection.log",
)
```

`ANOMALY_DETECTION_CFG = CTX.stage_config` holds the Gold_05 stage configuration section.

The `Ledger` is provided by the bootstrap — `ledger = CTX.ledger` — rather than being separately instantiated. Ledger steps are added throughout the notebook with `kind`, `step`, `message`, `why`, `consequence`, and `data` fields.

Two sequential sanity checks guard the bootstrap:

- A general check verifying `CTX`, `paths`, `CONFIG`, `CONFIG_MAP`, `STAGE_CFG`, `RESOLVED_PATHS`, `FILENAMES`, `VERSIONS_CFG`, `RUNTIME_CFG`, `DATASET_CFG`, `WANDB_CFG`, `EXECUTION_CFG`, `PIPELINE`, `logger`, `ledger`, and `LOG_PATH`.
- A Gold-specific check verifying `ANOMALY_DETECTION_CFG` is present before any downstream config access.

---

## 3. Configuration Block and Run Selection

After bootstrap, key variables are resolved:

- `STAGE = "gold"`, `LAYER_NAME`, `GOLD_VERSION`, `TRUTH_VERSION`, `RECIPE_ID` from config and version sections
- `PIPELINE_MODE`, `RUN_MODE`, `CONFIG_PROFILE`, `DATASET_NAME`
- `GOLD_PROCESS_RUN_ID` via `make_process_run_id(ANOMALY_DETECTION_CFG["process_run_id_prefix"])` (default prefix: `"gold05_anomaly_detection"`)

The run selection configuration block defines:

```python
SELECTED_RUN_KEY = "stage3_improved"
USE_GLOBAL_RESULTS_IF_AVAILABLE = True
RECOVERY_STABILITY_ROWS = 30
RECOVERY_STARTS_AFTER_BROKEN = True
ROW_ID_COLUMN = "meta__row_id"
```

`RUN_CONFIG_MAP` maps each valid run key to its `target_flag_column`, `primary_score_column`, `primary_decision_column`, and `run_family`. For cascade variants the flag column is `cascade_final_flag` and score is `stage2_score`. For stage3_medium and stage3_strict the flag columns are `cascade_stage3_medium_flag` and `cascade_stage3_strict_flag` respectively, with `stage3_weighted_score` as the primary score.

The resolved run configuration variables — `TARGET_FLAG_COLUMN`, `PRIMARY_SCORE_COLUMN`, `PRIMARY_DECISION_COLUMN`, `RUN_FAMILY` — are set from `RUN_CONFIG_MAP[SELECTED_RUN_KEY]` and used throughout the notebook.

Plot display constants are also set: `PLOT_WINDOW_BEFORE_CENTER = 300`, `PLOT_WINDOW_AFTER_CENTER = 300`, `PLOT_ALERT_MARKER_SIZE = 18`, `ALERT_PACKET_MAX_GAP_ROWS = 5`, `PLOT_SENSOR_NORMALIZATION_METHOD = "robust_zscore"`, `PLOT_SENSOR_CLIP_VALUE = 5.0`.

`PLOT_RUN_LABEL_MAP` provides human-readable run labels for plots and summaries. `COMPARISON_RUN_KEY = "cascade_tuned"` selects the comparison run for the baseline-vs-cascade overlay plot.

---

## 4. Artifact Directory Setup

Gold_05 uses two separate artifact directory builders.

**Upstream model families (read-only input references):**

```python
GOLD_BASELINE_ARTIFACT_DIRS    = build_artifact_dirs(..., family="baseline",              subdirs=GOLD_MODEL_SUBDIRS)
GOLD_CASCADE_DEFAULTS_ARTIFACT_DIRS = build_artifact_dirs(..., family="cascade_defaults", subdirs=GOLD_MODEL_SUBDIRS)
GOLD_CASCADE_TUNED_ARTIFACT_DIRS    = build_artifact_dirs(..., family="cascade_tuned",    subdirs=GOLD_MODEL_SUBDIRS)
GOLD_CASCADE_STAGE3_ARTIFACT_DIRS   = build_artifact_dirs(..., family="cascade_stage3_improved", subdirs=GOLD_MODEL_SUBDIRS)
```

`GOLD_MODEL_SUBDIRS = ["scores", "summaries", "thresholds", "metadata", "models", "plots", "config", "lineage"]`

**Gold_05 own output directory:**

```python
ANOMALY_DETECTION_ARTIFACT_DIRS = build_artifact_dirs_from_config(
    config=CONFIG,
    stage_key="gold_anomaly_detection",
)
```

Subdirectories extracted from the result:
- `ANOMALY_DETECTION_EXPORT_DIR` = `["exports"]`
- `ANOMALY_DETECTION_PLOT_DIR` = `["plots"]`
- `ANOMALY_DETECTION_SUMMARY_DIR` = `["summaries"]`
- `ANOMALY_DETECTION_PACKET_DIR` = `["packets"]`
- `ANOMALY_DETECTION_METADATA_DIR` = `["metadata"]`
- `ANOMALY_DETECTION_CONFIG_DIR` = `["config"]`
- `ANOMALY_DETECTION_LINEAGE_DIR` = `["lineage"]`

A config snapshot is saved at `ANOMALY_DETECTION_CONFIG_DIR / f"{DATASET_NAME}__gold_anomaly_detection__resolved_config.yaml"`.

`RUN_RESULT_PATH_MAP` maps each run key to its pickle artifact path inside the corresponding upstream model family's `"scores"` subdirectory. Note: `stage3_medium` and `stage3_strict` both point to the same `cascade_stage3_improved_results_file_name_pickle` artifact, since those operating modes are embedded in the Gold_03c output file.

`GOLD05_LEDGER_PATH` is resolved from `ANOMALY_DETECTION_LINEAGE_DIR / FILENAMES["gold_anomaly_detection_ledger_file_name"]`.

---

## 5. Database Connection and Asset Resolution

`engine = get_engine_from_env()` establishes the PostgreSQL connection. `CAPSTONE_SCHEMA`, `DATASET_ID`, `RUN_ID`, and `ASSET_ID` are resolved using `first_non_empty_string`, probing environment variables, `globals()`, and dataset config values in priority order. The `is_synthetic_run` flag determines which fallback defaults apply.

A SQL smoke check queries `information_schema.tables` to confirm the database connection and confirm schema presence before any data operations. `log_layer_paths` records resolved path state.

---

## 6. W&B Configuration

`set_wandb_dir_from_config(CONFIG)` sets the W&B working directory. Gold_05 resolves `WANDB_PROJECT`, `WANDB_ENTITY`, and `WANDB_RUN_NAME` from config, but does not call `wandb.init()`. No W&B run is opened and no artifacts are uploaded to W&B in Gold_05.

---

## 7. Upstream Data Loading

`load_selected_results_from_utils` loads the selected scored DataFrame. It follows a two-path resolution:

1. **Globals fallback (primary):** When `USE_GLOBAL_RESULTS_IF_AVAILABLE = True`, checks `globals()` for a DataFrame under the expected variable name for the selected run key. This allows Gold_05 to run immediately after Gold_02 or Gold_03 in the same kernel without reloading from disk. A secondary fallback checks `cascade_results` when a cascade-family run key is selected but its specific global is absent.

2. **Disk fallback:** If no in-memory DataFrame is found, loads from the path in `RUN_RESULT_PATH_MAP[SELECTED_RUN_KEY]` using `pd.read_pickle` (or `load_data` for non-pickle formats).

A ledger step `"load_selected_results"` records the run key, row count, and column count.

---

## 8. Anomaly Timeline Construction

The timeline is assembled through a sequential pipeline of inline helper functions applied to the loaded results DataFrame. Each step enriches the DataFrame in place:

**`ensure_row_id_and_plot_order`** — validates `meta__row_id` for presence and uniqueness, optionally sorts by a time axis column, and assigns a monotone integer `plot_order_index` column. `plot_order_index` becomes the stable x-axis for all plots and range queries.

**`add_broken_anchor_columns`** — locates the first row where `machine_status == "BROKEN"`, records its `plot_order_index` as `first_broken_plot_order_index`, and adds `is_broken_row` and `rows_to_first_broken`. Raises `ValueError` if no BROKEN row exists.

**`add_alert_and_normal_like_columns`** — converts `TARGET_FLAG_COLUMN` to a clean binary `selected_final_alert_flag` column; creates `is_normal_like_row = (selected_final_alert_flag == 0)`.

**`compute_forward_stable_normal_run`** — backward scan computing `forward_normal_like_run_length`: the number of consecutive normal-like rows from each position onward. This enables efficient recovery detection without a look-ahead pass.

**`resolve_recovery_boundaries`** — finds the first row at or after `first_broken_plot_order_index + 1` where `forward_normal_like_run_length >= RECOVERY_STABILITY_ROWS`. Returns a `recovery_boundary_payload` dict with `first_broken_plot_order_index`, `recovery_start_plot_order_index`, `recovery_end_plot_order_index`, and `recovery_stability_rows`.

**`add_episode_phase_columns`** — stamps `episode_phase` per row: `pre_failure`, `failure`, `recovery`, or `stable_normal`, based on the recovery boundary payload. Also adds `is_recovery_row` and `is_stable_normal_row`.

**`classify_detection_rows`** — assigns `detection_class` to each alert row based on its `episode_phase`: `early_warning` (pre_failure alert), `failure_hit` (failure row alert), `recovery_alert`, `false_positive` (stable_normal alert), or `no_alert` / `unclassified_alert`.

The result is:

```python
anomaly_timeline_dataframe, recovery_boundary_payload = build_anomaly_timeline_dataframe(
    selected_results,
    target_flag_column=TARGET_FLAG_COLUMN,
    status_column=status_column_resolved,
    row_id_column=ROW_ID_COLUMN,
    time_axis_column=time_axis_column,
    recovery_stability_rows=RECOVERY_STABILITY_ROWS,
    recovery_starts_after_broken=RECOVERY_STARTS_AFTER_BROKEN,
)
```

A ledger step `"build_timeline_dataframe"` records the complete `recovery_boundary_payload` and the annotated DataFrame shape.

`build_anomaly_timeline_dataframe` wraps all the above steps into one callable, making it reusable for multi-run comparisons.

---

## 9. Detection Analysis

**Detected rows review:** `build_detected_rows_review_dataframe` filters `anomaly_timeline_dataframe` to alert rows and selects context columns including `meta__row_id`, `plot_order_index`, `machine_status`, `episode_phase`, `detection_class`, the target flag column, and optional score/decision columns.

**Stage alerts dict:** A `stage_alerts` dict is built using `get_first_alert_index`, recording the first `plot_order_index` where each flag column fires: `stage1_flag`, `stage2_raw_flag`, `stage2_flag`, `cascade_final_flag`, `baseline_flag`. This provides a single-dict view of when each cascade stage first triggered.

**Stage 1 detected rows:** A separate `stage1_detected_rows_df` is built specifically for Stage 1 alerts, enriched with `stage2_raw_flag`, `stage2_flag`, and `cascade_final_flag` columns, allowing inspection of which Stage 1 alerts passed or were filtered by later cascade stages.

**Detection summary payload:** `build_detection_summary_payload` produces a JSON-compatible dict with:
- `selected_run_key`, `plot_run_label`, `run_family`, `target_flag_column`
- `first_alert_plot_order_index`, `first_broken_plot_order_index`, `recovery_end_plot_order_index`
- `lead_rows_to_failure` = `first_broken_plot_order_index - first_alert_plot_order_index` (None if no alert found)
- `lead_time_minutes_to_failure` (same value, kept as separate key)
- `total_final_alert_rows`
- `detection_class_counts` dict
- Cascade stage first-trigger indices (when `RUN_FAMILY == "cascade"`)

**Failure lead time DataFrame:** `build_failure_lead_time_dataframe` wraps `build_detection_summary_payload` into a single-row DataFrame.

**Alert packet summary:** `build_alert_packet_summary` groups consecutive alert rows separated by at most `ALERT_PACKET_MAX_GAP_ROWS = 5` rows into packets. Each packet record includes `packet_start_plot_order_index`, `packet_end_plot_order_index`, `packet_row_count`, `rows_from_packet_start_to_broken`, and boolean flags for whether the packet contains `pre_failure`, `failure_hit`, `recovery`, or `stable_normal` alerts.

A ledger step `"build_summaries"` records `lead_time_minutes_to_failure`, `total_final_alert_rows`, and `alert_packet_count`.

---

## 10. Debug Inspection Windows

The notebook includes explicit window slice cells for manual inspection at known analysis points:

- `plot_order_index` range 10380–10460 — early-warning window (pre-failure alert cluster)
- `plot_order_index` range 17000–17155 — pre-failure late-stage window
- `plot_order_index` range 17130–17210 — Stage 2 confirmation window near failure

These slices display available flag, score, and phase columns as DataFrames for direct inspection. They are display-only and do not produce saved artifacts.

---

## 11. Multi-Run Lead-Time Comparison

Gold_05 builds a cross-run early-warning summary covering all six model runs:

```python
LEAD_TIME_RUN_KEYS = [
    "baseline", "cascade_defaults", "cascade_tuned",
    "stage3_improved", "stage3_medium", "stage3_strict",
]
```

For each key, `build_run_timeline_dataframe` applies the full annotation pipeline (the same steps as section 8 above) and returns a payload dict containing the timeline DataFrame plus run-specific metadata. `build_comparison_summary_dataframe` tabulates these payloads into `lead_time_comparison_df`.

The comparison table includes `selected_run_key`, `plot_run_label`, `run_family`, `lead_rows_to_failure`, `lead_time_minutes_to_failure`, `total_final_alert_rows`, and cascade stage first-trigger indices where available.

`lead_time_comparison_df` is saved to `ANOMALY_DETECTION_SUMMARY_DIR / "multi_run_lead_time_comparison.csv"`.

A lead-time bar chart is generated showing `lead_time_hours_to_failure` per model run and saved to `ANOMALY_DETECTION_PLOT_DIR / f"{SELECTED_RUN_KEY}__multi_run_lead_time_comparison.png"`.

A ledger step `"multi_run_lead_time_comparison"` records all run keys, row count, and output path.

---

## 12. Comparison Overlay

`plot_comparison_overlay` renders both the baseline and `COMPARISON_RUN_KEY` timelines on the same axes around the same failure anchor. It overlays sensor values, alert markers, and model scores for both runs and adds lifecycle reference lines. The figure is displayed and later exported.

---

## 13. Sensor Visualization Suite

All visualizations center on a configurable anchor point (`"broken"` = first BROKEN row; `"alert"` = first selected alert row). `extract_centered_plot_window` extracts a ±300-row window and computes `relative_plot_index` where 0 is the anchor.

**`plot_anomaly_timeline_window`** — single-sensor plot with model score on a secondary y-axis, alert spans shaded, and lifecycle event reference lines. Centered on `"broken"` by default.

**`plot_all_sensors_stacked_waveform`** — all raw sensor columns (pattern `sensor_NN`) rendered as stacked normalized waveforms. Sensors are offset vertically by `lane_spacing`. Normalization (`zscore` by default for this plot) applies to a copy of the DataFrame only and does not alter saved artifacts.

**`plot_all_sensors_heatmap`** — 2D heatmap of all sensors across the plotting window. `build_sensor_matrix_for_plot` assembles the numeric matrix; normalization (`robust_zscore`) and optional clipping are applied before rendering.

**`plot_all_sensors_3d_surface`** — 3D surface plot using `mpl_toolkits.mplot3d`. X = rows from anchor, Y = sensor position, Z = normalized sensor value.

**`normalize_sensor_columns_for_plot`** — supports `"zscore"` and `"robust_zscore"` normalization methods. Operates on a copy of the DataFrame only; source data and saved exports always use original values.

**`resolve_sensor_columns`** — selects only raw sensor columns matching the strict regex `^sensor_\d{2}$`, excluding derived flag, profile, or metadata columns that share the `sensor_` prefix.

**Alert packet window plots:** For the top-K alert packets (sorted by `contains_pre_failure_alert DESC`, `rows_from_packet_start_to_broken DESC`, `packet_row_count DESC`), `plot_packet_centered_window` renders a ±120-row window around the packet start, showing the sensor value, model score, alert markers, and lifecycle reference lines.

All generated figures are collected and later exported to disk.

---

## 14. Artifact Exports

Exports occur in two batches. All file writes use `save_data` (for DataFrames) or `save_json` (for JSON payloads) or `fig.savefig` (for plots).

**Primary export batch (ledger step `"export_outputs"`):**

| Artifact | Directory | Filename pattern |
|---|---|---|
| Annotated timeline | `EXPORT_DIR` | `{SELECTED_RUN_KEY}__timeline_export.parquet` |
| Failure lead-time summary | `SUMMARY_DIR` | `{SELECTED_RUN_KEY}__failure_lead_time_summary.csv` |
| Alert packet summary | `PACKET_DIR` | `{SELECTED_RUN_KEY}__alert_packet_summary.csv` |
| Detection summary JSON | `SUMMARY_DIR` | `{SELECTED_RUN_KEY}__detection_summary.json` |
| Single-sensor timeline plot | `PLOT_DIR` | `{SELECTED_RUN_KEY}__timeline_plot.png` |
| Stacked sensor waveform | `PLOT_DIR` | `{SELECTED_RUN_KEY}__stacked_sensor_waveform.png` |

**Secondary export batch (ledger step `"additional_exports"`):**

| Artifact | Directory | Filename pattern |
|---|---|---|
| Detected rows review | `EXPORT_DIR` | `{SELECTED_RUN_KEY}__detected_rows_review.csv` |
| Baseline vs comparison summary | `SUMMARY_DIR` | `baseline_vs_{COMPARISON_RUN_KEY}__comparison_summary.csv` |
| Comparison overlay plot | `PLOT_DIR` | `baseline_vs_{COMPARISON_RUN_KEY}__comparison_plot.png` |
| Multi-run lead-time table | `SUMMARY_DIR` | `multi_run_lead_time_comparison.csv` |
| All-sensor heatmap | `PLOT_DIR` | `{SELECTED_RUN_KEY}__all_sensors_heatmap.png` |
| All-sensor 3D surface | `PLOT_DIR` | `{SELECTED_RUN_KEY}__all_sensors_3d_surface.png` |
| Top-K alert packets | `PACKET_DIR` | `{SELECTED_RUN_KEY}__top_alert_packets.csv` |
| Lead-time bar chart | `PLOT_DIR` | `{SELECTED_RUN_KEY}__multi_run_lead_time_comparison.png` |
| Per-packet window plots | `PACKET_DIR` | `{SELECTED_RUN_KEY}__packet_{idx}__window_plot.png` (one per top-K packet) |

`alert_packet_summary_df` and `top_alert_packets_df` exports are conditional on the DataFrames being non-empty.

---

## 15. Truth Record Construction and Stamping

The parent truth hash is extracted from the loaded scored results DataFrame rather than from config:

```python
parent_truth_hash = extract_truth_hash(selected_results)
```

This links the Gold_05 truth record directly to the upstream run that produced the scored output.

The truth record is initialized with `layer_name = "gold_anomaly_detection"` and `parent_truth_hash`. Three sections are populated:

- `config_snapshot`: `selected_run_key`, `target_flag_column`, `primary_score_column`, `primary_decision_column`, `recovery_stability_rows`, `config_hash`, `config_sources`, `config_snapshot_path`
- `runtime_facts`: `row_count`, `column_count`, `first_broken_plot_order_index`, `recovery_start_plot_order_index`, `recovery_end_plot_order_index`, `lead_time_minutes_to_failure`, `run_family`
- `artifact_paths`: all exported file paths (timeline, lead-time, packets, JSON summary, plots, detected rows, comparison outputs, heatmap, 3D, top packets, ledger path)

```python
gold05_truth_record = build_truth_record(
    truth_base=gold05_truth_base,
    row_count=len(anomaly_timeline_dataframe),
    column_count=anomaly_timeline_dataframe.shape[1],
    meta_columns=identify_meta_columns(anomaly_timeline_dataframe),
    feature_columns=identify_feature_columns(anomaly_timeline_dataframe),
)

gold05_truth_path = save_truth_record(gold05_truth_record, ...)
append_truth_index(gold05_truth_record, truth_index_path=RESOLVED_PATHS["truth_index_path"])

anomaly_timeline_dataframe = stamp_truth_columns(
    anomaly_timeline_dataframe,
    truth_hash=gold05_truth_record["truth_hash"],
    parent_truth_hash=parent_truth_hash,
    pipeline_mode=PIPELINE_MODE,
)
```

A ledger step `"truth_record"` records the truth hash, truth path, and parent truth hash.

---

## 16. Ledger Finalization

```python
ledger.write_json(GOLD05_LEDGER_PATH)
```

The ledger is written directly to `GOLD05_LEDGER_PATH` in the `ANOMALY_DETECTION_LINEAGE_DIR`. Unlike Gold_02 through Gold_04, there is no W&B run, so no `wandb.save` call is made for the ledger file.

---

## 17. SQL Persistence

```python
WRITE_TO_POSTGRES = True

gold_05_sql_summary_dataframe = log_gold_05_anomaly_detection_summary_sql(
    engine=engine,
    capstone_schema=CAPSTONE_SCHEMA,
    dataset_id=DATASET_ID,
    run_id=RUN_ID,
    notebook_globals=globals(),
    dataframe=gold05_output_manifest_df,
    dataset_name=DATASET_NAME,
)
```

Before calling the writer, an output manifest DataFrame (`gold05_output_manifest_df`) is built by inspecting up to seven named output frames: `failure_lead_time`, `detected_rows_review`, `alert_packet_summary`, `top_alert_packets`, `lead_time_comparison`, `multi_run_lead_time`, and `anomaly_timeline`. For each non-None DataFrame, a manifest record is built with `dataset_id`, `run_id`, `output_name`, `output_type`, `row_count`, `column_count`, and `columns`. `globals().get()` is used throughout so optional frames (those conditionally skipped on empty detection results) do not raise `NameError`.

`log_gold_05_anomaly_detection_summary_sql` logs the manifest to `capstone.pipeline_runs` and `capstone.data_quality_events`.

Two post-write verification queries are run:

1. Queries `capstone.pipeline_runs` where `pipeline_stage = 'gold_anomaly_detection_summary'`, returning `run_status`, `completed_at_utc`, and `runtime_facts`.
2. Queries `capstone.data_quality_events` where `check_name = 'gold_05_summary_sql_log'`, returning check status and details.

---

## Inputs

| Source | Type | Load method |
|---|---|---|
| Selected scored results (one of six run keys) | Pickle DataFrame | `load_selected_results_from_utils` (globals → `pd.read_pickle` fallback) |
| Gold_04 comparison artifacts | Not loaded | Gold_05 does not consume Gold_04 outputs |
| Upstream model artifact paths | Resolved from config | `build_artifact_dirs` (read-only directory references) |

Gold_05 does not load Gold_04 comparison CSV, summary JSON, truth records, or statistical test artifacts. It reads directly from the individual model scored outputs produced by Gold_02 or Gold_03 notebooks.

---

## Outputs and Artifacts

| Artifact | Directory anchor | Format |
|---|---|---|
| Annotated timeline export | `EXPORT_DIR` | Parquet |
| Detected rows review | `EXPORT_DIR` | CSV |
| Failure lead-time summary | `SUMMARY_DIR` | CSV |
| Detection summary payload | `SUMMARY_DIR` | JSON |
| Multi-run lead-time comparison | `SUMMARY_DIR` | CSV |
| Baseline vs comparison summary | `SUMMARY_DIR` | CSV |
| Alert packet summary | `PACKET_DIR` | CSV |
| Top-K alert packets | `PACKET_DIR` | CSV |
| Single-sensor timeline plot | `PLOT_DIR` | PNG |
| Stacked sensor waveform | `PLOT_DIR` | PNG |
| All-sensor heatmap | `PLOT_DIR` | PNG |
| All-sensor 3D surface | `PLOT_DIR` | PNG |
| Multi-run lead-time chart | `PLOT_DIR` | PNG |
| Comparison overlay plot | `PLOT_DIR` | PNG |
| Per-packet window plots | `PACKET_DIR` | PNG (one per top-K packet) |
| Config snapshot | `CONFIG_DIR` | YAML |
| Gold_05 truth record | `paths.truths / gold_anomaly_detection` | JSON |
| Gold_05 ledger | `LINEAGE_DIR` | JSONL |
| SQL pipeline run summary | `capstone.pipeline_runs` | PostgreSQL |
| SQL data quality event | `capstone.data_quality_events` | PostgreSQL |

---

## 20. Truth Chain and Lineage Continuity

| Field | Source | Value |
|---|---|---|
| `parent_truth_hash` | Extracted from `selected_results` via `extract_truth_hash` | Links to the upstream Gold_02 or Gold_03 scoring run |
| `gold05_truth_record["truth_hash"]` | Computed by `build_truth_record` | Stamps into `anomaly_timeline_dataframe` via `stamp_truth_columns` |
| `DATASET_ID`, `RUN_ID` | Resolved from env/config | Passed to SQL writer |

The Gold_05 truth record's `parent_truth_hash` points to the truth hash embedded in the scored results DataFrame of the selected run, which in turn carries the lineage from Gold_01 preprocessing through the upstream scoring notebook. Gold_05 does not reconstruct or revalidate the upstream truth chain — it propagates the existing hash.

---

## Data Quality / Validation Behavior

| Check | Purpose | Failure / Risk Prevented |
|---|---|---|
| `add_broken_anchor_columns` raises `ValueError` if no BROKEN row | Confirm `machine_status == "BROKEN"` row exists before timeline construction | Hard stop; prevents constructing a timeline with no failure anchor |
| `ensure_row_id_and_plot_order` validates `meta__row_id` presence and uniqueness | Row identity before timeline annotation | Guards all downstream join and offset operations |
| `build_paired_model_frame` merge raises `ValueError` on row loss | Merged baseline and comparison frames must have identical row sets (statistical tests) | Prevents phantom merges from misaligned result frames |
| `meta__row_id` uniqueness in both frames for statistical test merge | Structural integrity check | `ValueError` if either frame has duplicated `meta__row_id` |
| Post-write verification queries | Re-read `capstone.pipeline_runs` and `capstone.data_quality_events` after SQL write | Confirms write reached the database |

---

## Downstream Handoff

Gold_05 does not produce model scoring outputs, flag columns on raw sensor data, or comparison metrics between models. Its exports are analysis artifacts: annotated timelines, lead-time summaries, packet summaries, and visualizations.

Gold_06A (Test Replay Validation) and Gold_06B (Early Warning Validation) follow in the pipeline but Gold_05 does not explicitly configure paths for their consumption. The `anomaly_timeline_dataframe` export (Parquet) and `detection_summary_payload` (JSON) are the most structured outputs that downstream analysis could consume, but no direct handoff contract is confirmed in the Gold_05 source.

---

## Key Function Calls and In-Place Usage

| Function | Module | Purpose |
|---|---|---|
| `load_notebook_context` | `utils.core.notebook_context` | Bootstrap CTX, paths, config, logger, ledger |
| `build_artifact_dirs` | `utils.core.artifacts` | Input artifact directories for 4 upstream model families |
| `build_artifact_dirs_from_config` | `utils.core.artifacts` | Gold_05 own output directory tree |
| `export_config_snapshot` | `utils.core.config_loader` | Save resolved config YAML |
| `set_wandb_dir_from_config` | `utils.core.config_loader` | Set W&B working directory (no run opened) |
| `get_engine_from_env` | `utils.database.postgres` | PostgreSQL engine |
| `read_sql_dataframe` | `utils.database.postgres` | Smoke check and post-write verification |
| `log_layer_paths` | `utils.core.logging_setup` | Log resolved paths |
| `load_data` / `save_data` | `utils.core.file_io` | Parquet/CSV I/O |
| `save_json` | `utils.core.file_io` | JSON artifact saves |
| `extract_truth_hash` | `utils.core.truths` | Extract `meta__truth_hash` from loaded scored DataFrame |
| `initialize_layer_truth` | `utils.core.truths` | Create blank truth record |
| `update_truth_section` | `utils.core.truths` | Populate config_snapshot, runtime_facts, artifact_paths |
| `build_truth_record` | `utils.core.truths` | Finalize and hash truth record |
| `stamp_truth_columns` | `utils.core.truths` | Add `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` to timeline DataFrame |
| `save_truth_record` | `utils.core.truths` | Write truth record to disk |
| `append_truth_index` | `utils.core.truths` | Register truth record in index |
| `identify_meta_columns` / `identify_feature_columns` | `utils.core.truths` | Column classification for truth record |
| `make_process_run_id` | `utils.core.truths` | Generate process run ID |
| `log_gold_05_anomaly_detection_summary_sql` | `utils.database.medallion_sql_writers` | Log output manifest to `capstone.pipeline_runs` and `capstone.data_quality_events` |

---

## Logical Workflow Map

1. Imports
2. Inline helper definitions (`require_int_value`, `cfg_require_mapping`, etc.)
3. Context bootstrap (`load_notebook_context`)
4. Config variable resolution
5. Sanity checks (general, then Gold-specific)
6. Run selection config block (`SELECTED_RUN_KEY`, `RUN_CONFIG_MAP`, `TARGET_FLAG_COLUMN`, `RUN_FAMILY`, plot constants)
7. Database connection + asset resolution (`DATASET_ID`, `RUN_ID`, `ASSET_ID`)
8. SQL smoke check
9. Path logging (`log_layer_paths`)
10. Artifact directory setup (4 upstream + Gold_05 own via `build_artifact_dirs_from_config`)
11. Config snapshot save
12. `RUN_RESULT_PATH_MAP` + `GOLD05_LEDGER_PATH` setup
13. Ledger checkpoint step
14. `load_selected_results_from_utils` — load selected scored DataFrame
15. Column resolver helpers (`resolve_first_present_column` for time axis and status column)
16. Timeline pipeline helper definitions
17. `build_anomaly_timeline_dataframe` — produces `anomaly_timeline_dataframe` and `recovery_boundary_payload`
18. `build_detected_rows_review_dataframe` — all alert rows
19. `stage_alerts` dict construction
20. Stage 1 detected rows review DataFrame
21. `plot_order_index` validation
22. Debug window slice displays (10380–10460, 17000–17155, 17130–17210)
23. `build_optional_cascade_stage_summary` + `build_detection_summary_payload` helper definitions
24. `build_failure_lead_time_dataframe` helper definition
25. `build_alert_packet_summary` helper definition
26. `failure_lead_time_df`, `alert_packet_summary_df`, `detection_summary_payload` construction
27. Multi-run helper definitions (`resolve_run_config`, `load_results_for_run`, `build_run_timeline_dataframe`, `build_run_detection_summary_payload`, `build_comparison_summary_dataframe`)
28. Window extraction and plot helper definitions
29. `baseline_payload` + `comparison_payload` via `build_run_timeline_dataframe`
30. `comparison_summary_df` via `build_comparison_summary_dataframe`
31. Multi-run lead-time comparison across 6 run keys → `lead_time_comparison_df` + CSV save
32. Lead-time bar chart save to `PLOT_DIR`
33. Comparison overlay plot helper definition + `plot_comparison_overlay` call
34. `plot_anomaly_timeline_window` → `timeline_fig`
35. `resolve_sensor_columns` + `normalize_sensor_columns_for_plot` helper definitions
36. `plot_all_sensors_stacked_waveform` → `stacked_waveform_fig`
37. `build_sensor_matrix_for_plot` + `plot_all_sensors_heatmap` helper definitions
38. `plot_all_sensors_heatmap` call → `all_sensor_heatmap_fig`
39. `plot_all_sensors_3d_surface` helper definition
40. `plot_all_sensors_3d_surface` call → `all_sensor_3d_fig`
41. Top-K packet selection → `top_alert_packets_df`
42. `plot_packet_centered_window` helper definition
43. Per-packet window plot generation → `packet_figures`
44. Primary export batch (`save_data` and `savefig` for timeline, lead-time, packets, JSON, timeline plot, stacked waveform)
45. Secondary export batch (detected rows, comparison summary, comparison plot, heatmap, 3D, top packets, packet plots)
46. Truth record construction and stamping (`extract_truth_hash` → `initialize_layer_truth` → `update_truth_section` ×3 → `build_truth_record` → `stamp_truth_columns` → `save_truth_record` → `append_truth_index`)
47. Ledger write to `GOLD05_LEDGER_PATH`
48. `log_gold_05_anomaly_detection_summary_sql` — SQL write
49. Post-write verification: `capstone.pipeline_runs` query
50. Post-write verification: `capstone.data_quality_events` query

---

## Relationship to Other Notebooks

### Upstream Context

Gold_05 loads scored results for the selected model run via `SELECTED_RUN_KEY` (default `"stage3_improved"` — Gold_03c's results). It iterates over all six run keys for the multi-run lead-time comparison. Gold_05 does not consume Gold_04 comparison outputs as pipeline inputs, and does not depend on Gold_06A or Gold_06B.

### Downstream Handoff

Gold_05 provides:
- Anomaly timeline Parquet and detection summary JSON for review and submission support
- Multi-run lead-time comparison CSV (`multi_run_lead_time_comparison.csv`) consumed optionally by Gold_06B
- Visualization PNGs for submission documentation
- `gold_anomaly_detection` truth record (does not chain to Gold_06A or Gold_06B)
- SQL rows via `log_gold_05_anomaly_detection_summary_sql`

### Pipeline Position

Post-modeling analysis notebook. Follows Gold_04 in execution sequence but operates independently of it. Applies anomaly timeline annotation and early-warning analysis to a selected model run. Provides the training-run early-warning reference optionally consumed by Gold_06B. Does not train, score, or route model decisions.

### Relationship Summary

- Follows Gold_04 in execution sequence but does not consume Gold_04 outputs as pipeline inputs
- Loads scored results for `SELECTED_RUN_KEY` directly (default: Gold_03c stage3_improved results)
- Multi-run lead-time CSV is the only confirmed handoff to another notebook (Gold_06B, optional with graceful fallback)
- Gold_06A runs independently of Gold_05; they are parallel validation paths that converge at Gold_06B
- Does not open a W&B run; truth record does not chain to Gold_06A or Gold_06B
