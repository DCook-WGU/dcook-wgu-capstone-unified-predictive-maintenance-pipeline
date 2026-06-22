# Gold 05 Deep Technical Reference

## Purpose of This Deep Reference

This document covers technical decisions in Gold 05 that require deeper explanation than the workflow reference provides. The workflow reference describes what the notebook does step by step. This document explains why the anomaly timeline pipeline, run-selection pattern, episode-phase classification, recovery detection algorithm, export strategy, truth-record design, and SQL persistence behavior are designed the way they are.

## Technical Scope

- Single-run-key analysis design and run-key reconfiguration
- Two-path data loading: globals-first with disk fallback
- Canonical timeline frame construction and its role as the single downstream source
- Failure anchor hard prerequisite and BROKEN-row dependency
- Synthetic plot order index as a dataset-agnostic x-axis
- Forward stable normal run length: backward scan implementation
- episode_phase-based detection classification
- Alert packet grouping and top-K prioritization
- Multi-run lead-time comparison covering all six model variants
- stage3_medium and stage3_strict shared artifact behavior
- Parent truth hash extraction from scored results rather than config
- Strict sensor column regex and display-only normalization
- Export batching and conditional artifact persistence
- No W&B run behavior
- SQL persistence and post-write verification

## Source Grounding

Sources used:

- `notebooks/experiments/EDA_Notebook_Pump_Gold_05_Anomaly_Detection.ipynb`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_05_Anomaly_Detection_code_reference.md`
- `notebook_inventory.json`
- `artifact_io_manifest.json`
- `sql_touchpoints.json`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_04_Comparison_deep_technical_reference.md`
- `technical_reference/00_project_manual/notebook_relationship_map.md`
- `technical_reference/00_project_manual/notebook_dependency_matrix.md`

The active Gold 05 notebook source is the source of truth for all function behavior, variable names, and design decisions documented here. The workflow reference and project manual provide consumer and handoff context only.

## Stage Role in the Gold Modeling Sequence

Gold 05 is the anomaly timeline analysis and early-warning evaluation notebook. It occupies a distinct role from the modeling notebooks (Gold_02 through Gold_03c) and the comparison notebook (Gold_04). Gold 05 does not train models, apply detection thresholds, or re-score sensor data. It consumes one scored output DataFrame produced by an upstream modeling notebook, annotates it with failure lifecycle phases and detection classes, quantifies early-warning lead time, produces a multi-run comparison across all six model variants, and exports reporting-ready analysis artifacts.

Gold 05 sits after Gold_04 in execution sequence but operates independently of it. Gold_04 aggregates model comparison metrics across variants. Gold_05 takes a selected single variant and performs deep timeline analysis on its row-level outputs. These are parallel reporting responsibilities, not a sequential dependency.

The `CONTEXT_STAGE = "gold_anomaly_detection"` constant distinguishes Gold_05 outputs in the truth index, SQL metadata tables, and artifact directory tree from all upstream modeling stages.

## Input Contract and Lineage

### Selected Run Input

Gold 05 loads exactly one scored DataFrame through `load_selected_results_from_utils`. The selected run is controlled by `SELECTED_RUN_KEY`, which defaults to `"stage3_improved"`. This single variable reconfigures every downstream operation in the notebook: the target flag column, the primary score column, the primary decision column, all summaries, all plots, all exports, and the truth record.

Valid run keys and their corresponding flag columns:

| `SELECTED_RUN_KEY` | `target_flag_column` | `run_family` |
|---|---|---|
| `baseline` | `baseline_flag` | `baseline` |
| `cascade_defaults` | `cascade_final_flag` | `cascade` |
| `cascade_tuned` | `cascade_final_flag` | `cascade` |
| `stage3_improved` | `cascade_final_flag` | `cascade` |
| `stage3_medium` | `cascade_stage3_medium_flag` | `cascade` |
| `stage3_strict` | `cascade_stage3_strict_flag` | `cascade` |

For `stage3_medium` and `stage3_strict`, the primary score column is `stage3_weighted_score` rather than `stage2_score`. Both operating modes are embedded in the Gold_03c output pickle, so their `RUN_RESULT_PATH_MAP` entries both point to the same `cascade_stage3_improved_results_file_name_pickle` artifact. If that artifact is missing, both modes fail together.

### No Gold 04 Dependency

Gold 05 does not consume Gold_04 comparison outputs. It does not load Gold_04 comparison CSV, summary JSON, truth records, or statistical test artifacts. The upstream modeling artifact directories for all four model families are resolved as read-only input references for path computation, but Gold_05 does not need Gold_04 to have run first. This separation keeps Gold 05 usable when only a subset of comparison runs have been completed.

### Required Input Columns

The loaded scored DataFrame must contain:

- A status column (probed from `machine_status`, `status`, `state` in that order via `resolve_first_present_column`) — required; raises if none is found
- `TARGET_FLAG_COLUMN` (resolved from `RUN_CONFIG_MAP[SELECTED_RUN_KEY]`) — checked by `add_alert_and_normal_like_columns`; raises `ValueError` if absent
- `meta__row_id` — used by `ensure_row_id_and_plot_order`; created as a zero-based integer sequence if absent, then validated for null-free uniqueness
- An optional time axis column (probed from `time_index`, `event_step`, `event_time`) — used for sort order if present; not required

Score and decision columns (`PRIMARY_SCORE_COLUMN`, `PRIMARY_DECISION_COLUMN`) are used in visualization and the detection review DataFrame but are optional. Their absence skips visualization detail rather than raising.

### Dataset and Run Identity

`DATASET_ID`, `RUN_ID`, and `ASSET_ID` are resolved via `first_non_empty_string`, probing environment variables, `globals()`, and `DATASET_CFG` in priority order. The `is_synthetic_run` flag determines which fallback defaults apply. These values are passed to `log_gold_05_anomaly_detection_summary_sql` and embedded in SQL metadata records.

### Parent Truth Hash

The parent truth hash is extracted from the loaded scored results DataFrame via `extract_truth_hash(selected_results)`. The notebook comment confirms the design intent: the truth hash is read from the DataFrame rather than from config so that the Gold 05 truth record links directly to the upstream run that produced the scores. This means if two different Gold_03c runs produced different scored DataFrames, each Gold 05 run against them would carry a different parent hash without requiring any config change.

## Final Output Preparation

### Single-Frame Canonical Design

All Gold 05 analysis derives from one canonical output: `anomaly_timeline_dataframe`. This frame is built by `build_anomaly_timeline_dataframe`, which applies a sequential pipeline of inline helper functions to the loaded scored results. Every downstream summary, plot, export, and truth record derives from this single frame. The ledger step `"build_timeline_dataframe"` records the complete `recovery_boundary_payload` and frame shape so the canonical frame is auditable without re-running.

### Plot Order Index

`ensure_row_id_and_plot_order` creates a monotone integer `plot_order_index` column assigned as `np.arange(len(out), dtype=np.int64)` after optional time-axis sorting. The notebook comment explains the design: "A stable integer plot_order_index is created even when no time column exists so all downstream phase logic and plots have a consistent x-axis regardless of dataset." This avoids coupling window extraction, boundary detection, and visualization to timestamp arithmetic. All lead-time metrics are measured in rows (`lead_rows_to_failure`, `lead_time_minutes_to_failure` as a second key holding the same value) rather than calendar time.

### Run Family Gating

`build_optional_cascade_stage_summary` and `build_detection_summary_payload` gate cascade-specific fields on `RUN_FAMILY == "cascade"`. This allows the same summary functions to operate for both `baseline` and cascade run keys without `KeyError` when cascade stage columns (`stage1_flag`, `stage2_flag`, `stage2_raw_flag`) are absent from baseline scored outputs. The run family is set from `RUN_CONFIG_MAP[SELECTED_RUN_KEY]["run_family"]` at the configuration stage and remains constant for the entire notebook run.

## Final Anomaly Detection Methodology

### Episode Phase Pipeline

The timeline annotation pipeline applies six sequential transformations to the loaded scored results:

1. `ensure_row_id_and_plot_order` — validates `meta__row_id`, optionally sorts by time axis, assigns `plot_order_index`
2. `add_broken_anchor_columns` — locates the first BROKEN row; raises `ValueError` if none exists; assigns `is_broken_row`, `first_broken_plot_order_index`, `rows_to_first_broken`
3. `add_alert_and_normal_like_columns` — converts `TARGET_FLAG_COLUMN` to `selected_final_alert_flag` (integer binary); creates `is_normal_like_row`
4. `compute_forward_stable_normal_run` — assigns `forward_normal_like_run_length` via backward scan
5. `resolve_recovery_boundaries` — finds recovery start; returns `recovery_boundary_payload`
6. `add_episode_phase_columns` — stamps `episode_phase` per row from boundary payload
7. `classify_detection_rows` — assigns `detection_class` to each alert row from its `episode_phase`

The BROKEN anchor is a hard prerequisite. `add_broken_anchor_columns` raises `ValueError` immediately if no row has `machine_status == "BROKEN"`. All downstream phase computation, lead-time calculation, and detection classification depend on this anchor existing.

### Recovery Detection Algorithm

`compute_forward_stable_normal_run` uses a backward scan across the `is_normal_like_row` array. Starting from the last row and scanning toward the first, it accumulates consecutive normal-like rows and writes that count as `forward_normal_like_run_length[i]`. The notebook comment explains the approach: "The backward scan computes how many consecutive normal-like rows follow each row. This avoids needing a look-ahead and runs in one pass."

`resolve_recovery_boundaries` then finds the first row at or after `first_broken_plot_order_index + 1` (when `RECOVERY_STARTS_AFTER_BROKEN = True`) where `forward_normal_like_run_length >= RECOVERY_STABILITY_ROWS`. `RECOVERY_STABILITY_ROWS` defaults to 30. The requirement for 30 consecutive rows prevents a single clean observation from prematurely ending the recovery phase — a single anomaly-free row would set `forward_normal_like_run_length` to something small and not qualify. When no 30-row stable run is found post-failure, `recovery_end_plot_order_index` is `None` and all rows after the failure anchor are labeled `"recovery"`.

### Detection Classification

`classify_detection_rows` assigns one of five detection classes to each alert row based on its `episode_phase`:

| Detection Class | Condition |
|---|---|
| `early_warning` | Alert in `pre_failure` phase |
| `failure_hit` | Alert in `failure` phase |
| `recovery_alert` | Alert in `recovery` phase |
| `false_positive` | Alert in `stable_normal` phase |
| `no_alert` | Row has no alert |
| `unclassified_alert` | Alert in unrecognized phase |

The comment in the notebook confirms the design intent: "Classification uses episode_phase rather than just the alert flag so each alert row is labelled with its operational context." This separation matters for reporting: an alert in `pre_failure` is an early-warning success, while an alert in `stable_normal` is a false positive. A simple alert count would not distinguish these.

### Alert Packet Grouping

`build_alert_packet_summary` groups consecutive alert rows whose `plot_order_index` values are separated by at most `ALERT_PACKET_MAX_GAP_ROWS = 5` rows into named packets. The notebook comment states: "Packets group consecutive alert rows separated by at most max_gap_rows. This surfaces burst patterns rather than individual points, which is more useful for evaluating pre-failure alarm clustering." Each packet record includes the start/end `plot_order_index`, row count, distance to the first BROKEN row, and boolean flags for the episode phases present in the packet.

Top-K packet selection sorts by `contains_pre_failure_alert DESC`, then `rows_from_packet_start_to_broken DESC`, then `packet_row_count DESC`. The comment explains: "Sort puts pre-failure packets first and then by distance to failure, so the most diagnostically valuable examples are reviewed before false-positive clusters."

When `alert_packet_summary_df` is empty (no alerts were found), the packet artifact exports are skipped rather than writing empty files.

## Reporting and Summary Construction

### Detection Summary Payload

`build_detection_summary_payload` produces a JSON-compatible dict containing `selected_run_key`, `plot_run_label`, `run_family`, `target_flag_column`, first alert and broken row indices, `lead_rows_to_failure`, `lead_time_minutes_to_failure`, `total_final_alert_rows`, `detection_class_counts`, and cascade stage first-trigger indices when `RUN_FAMILY == "cascade"`. This dict is saved as `{SELECTED_RUN_KEY}__detection_summary.json` and is the primary structured reporting artifact for the selected run.

`lead_rows_to_failure` and `lead_time_minutes_to_failure` hold the same numerical value. The workflow reference confirms these are stored under two keys for reporting compatibility — some downstream consumers may expect one name or the other.

### Multi-Run Lead-Time Comparison

Gold 05 builds a 6-run comparison table regardless of which `SELECTED_RUN_KEY` is active. `LEAD_TIME_RUN_KEYS = ["baseline", "cascade_defaults", "cascade_tuned", "stage3_improved", "stage3_medium", "stage3_strict"]` is hardcoded. For each key, `build_run_timeline_dataframe` applies the full annotation pipeline and returns a payload dict. `build_comparison_summary_dataframe` tabulates these into `lead_time_comparison_df`.

The multi-run comparison does not depend on the primary `anomaly_timeline_dataframe`. Each run builds its own independent timeline via `load_results_for_run` (which calls `load_selected_results_from_utils` for each key in turn). This means the comparison covers the full model family including the non-selected runs.

`lead_time_comparison_df` is saved as `multi_run_lead_time_comparison.csv` and a bar chart is generated in hours (`lead_time_minutes_to_failure / 60`). The CSV is the only artifact confirmed to be consumed optionally by Gold_06B.

### Visualization Suite

All visualizations center on a configurable anchor (`"broken"` or `"alert"`). `extract_centered_plot_window` extracts a ±300-row window and assigns `relative_plot_index = 0` at the anchor. All four plot types — single-sensor timeline, stacked sensor waveform, heatmap, and 3D surface — operate on this centered window.

Sensor normalization is intentionally isolated from saved artifacts. `normalize_sensor_columns_for_plot` operates on a DataFrame copy. The notebook comment states: "Normalization is applied to a copy of the dataframe and affects only plot rendering. Saved artifacts (timeline export, truth records) always use the original sensor values." This prevents normalized values from entering the exported Parquet or truth record.

`resolve_sensor_columns` uses the strict regex `r"^sensor_\d{2}$"` to select raw pump sensor columns. The notebook comment explains: "The strict regex (sensor_NN) is intentional: it excludes derived columns like sensor_profile_min or boolean flag columns that share a sensor_ prefix." The regex raises `ValueError` if no matching columns are found.

## Validation and Quality Checks

### Bootstrap Sanity Checks

Two sequential checks guard the bootstrap:

1. General context check verifies 16 required variables: `CTX`, `paths`, `CONFIG`, `CONFIG_MAP`, `STAGE_CFG`, `RESOLVED_PATHS`, `FILENAMES`, `VERSIONS_CFG`, `RUNTIME_CFG`, `DATASET_CFG`, `WANDB_CFG`, `EXECUTION_CFG`, `PIPELINE`, `logger`, `ledger`, `LOG_PATH`. Raises `NameError` with the list of missing names if any are absent.
2. Gold-specific check verifies `ANOMALY_DETECTION_CFG` is present before any stage config access. Raises `NameError` with the missing name.

### Pipeline Validation

- `ensure_row_id_and_plot_order` raises `ValueError` if `meta__row_id` contains null values or is not unique.
- `add_broken_anchor_columns` raises `ValueError` if no BROKEN row exists in the scored DataFrame.
- `add_alert_and_normal_like_columns` raises `ValueError` if `target_flag_column` is absent from the DataFrame.
- `resolve_first_present_column` with `required=True` raises when no candidate column is found (used for status column resolution).
- `resolve_sensor_columns` raises `ValueError` if no column matches `r"^sensor_\d{2}$"`.
- `resolve_run_config` raises `ValueError` if `SELECTED_RUN_KEY` is not in `RUN_CONFIG_MAP`.

### SQL Verification

Two post-write verification queries run after `log_gold_05_anomaly_detection_summary_sql`:

1. Queries `capstone.pipeline_runs` where `pipeline_stage = 'gold_anomaly_detection_summary'` to confirm write with `run_status`, `completed_at_utc`, and `runtime_facts`.
2. Queries `capstone.data_quality_events` where `check_name = 'gold_05_summary_sql_log'` to confirm the quality event write.

These queries do not raise if no rows are returned — they display the result for visual review.

### Database Smoke Check

A SQL smoke check queries `information_schema.tables` immediately after `get_engine_from_env()` to confirm the database connection and schema presence before any data operations.

## Artifact and SQL Persistence

### Export Batches

Exports occur in two batches, each recorded as a ledger step.

**Primary batch (`"export_outputs"`):**

| Artifact | Format | Notes |
|---|---|---|
| Annotated timeline | Parquet | `{SELECTED_RUN_KEY}__timeline_export.parquet` |
| Failure lead-time summary | CSV | `{SELECTED_RUN_KEY}__failure_lead_time_summary.csv` |
| Alert packet summary | CSV | Conditional on non-empty |
| Detection summary payload | JSON | `{SELECTED_RUN_KEY}__detection_summary.json` |
| Single-sensor timeline plot | PNG | `{SELECTED_RUN_KEY}__timeline_plot.png` |
| Stacked sensor waveform | PNG | `{SELECTED_RUN_KEY}__stacked_sensor_waveform.png` |

**Secondary batch (`"additional_exports"`):**

| Artifact | Format | Notes |
|---|---|---|
| Detected rows review | CSV | All alert rows with context columns |
| Baseline vs comparison summary | CSV | `baseline_vs_{COMPARISON_RUN_KEY}__comparison_summary.csv` |
| Comparison overlay plot | PNG | |
| Multi-run lead-time comparison | CSV | Only confirmed handoff artifact for Gold_06B |
| All-sensor heatmap | PNG | |
| All-sensor 3D surface | PNG | |
| Top-K alert packets | CSV | Conditional on non-empty |
| Per-packet window plots | PNG | One per top-K packet |

A config snapshot is saved as YAML at `ANOMALY_DETECTION_CONFIG_DIR / {DATASET_NAME}__gold_anomaly_detection__resolved_config.yaml`.

### SQL Persistence

`WRITE_TO_POSTGRES = True` is hardcoded. `log_gold_05_anomaly_detection_summary_sql` logs an output manifest DataFrame to `capstone.pipeline_runs` and `capstone.data_quality_events`.

Before calling the writer, up to seven named output frames are inspected using `globals().get()`. The notebook comment states: "globals().get() is used so optional frames (e.g. alert_packet_summary) do not raise NameError when skipped earlier due to empty detection results." For each non-None DataFrame, a manifest record is built containing `dataset_id`, `run_id`, `output_name`, `output_type`, `row_count`, `column_count`, and `columns`. This design allows the SQL manifest to represent the actual set of frames produced, rather than assuming all frames were created.

### W&B Behavior

Gold 05 calls `set_wandb_dir_from_config(CONFIG)` and resolves `WANDB_PROJECT`, `WANDB_ENTITY`, and `WANDB_RUN_NAME` from config. It does not call `wandb.init()`, does not open a W&B run, and does not upload artifacts or log metrics to W&B. The workflow reference confirms: "No W&B run is opened; no truth record chains downstream to Gold_06A."

### Ledger

The `Ledger` is provided by `CTX.ledger` rather than separately instantiated. Ledger steps added throughout the notebook include:

- `"context_loaded"` — bootstrap complete
- `"context_sanity_check"` — 16 required variables verified
- `"load_selected_results"` — selected DataFrame loaded
- `"build_timeline_dataframe"` — canonical timeline frame built
- `"build_summaries"` — lead-time, packet, and detection summary produced
- `"multi_run_lead_time_comparison"` — 6-run table and chart complete
- `"top_k_alert_packets"` — top-K packet selection complete
- `"export_outputs"` — primary batch
- `"additional_exports"` — secondary batch
- `"truth_record"` — truth hash and path recorded

The ledger is written to `GOLD05_LEDGER_PATH` in `ANOMALY_DETECTION_LINEAGE_DIR`. No W&B save of the ledger file occurs.

## Truth, Audit, and Reproducibility Behavior

### Parent Truth Hash

`parent_truth_hash = extract_truth_hash(selected_results)` extracts the truth hash embedded in the scored results DataFrame. This links the Gold 05 truth record to whichever upstream scoring run produced the input. The notebook comment confirms the intent: "The parent truth hash is extracted from the scored results dataframe rather than config so the Gold 05 truth record links directly to the run that produced the scores." If the upstream run re-scored with different parameters and produced a different truth hash, the Gold 05 parent hash would reflect the new run without any config change.

### Truth Record Construction

The truth record is initialized with `layer_name = "gold_anomaly_detection"` and the extracted parent truth hash. Three sections are populated:

- `config_snapshot`: `selected_run_key`, `target_flag_column`, `primary_score_column`, `primary_decision_column`, `recovery_stability_rows`, `config_hash`, `config_sources`, `config_snapshot_path`
- `runtime_facts`: `row_count`, `column_count`, `first_broken_plot_order_index`, `recovery_start_plot_order_index`, `recovery_end_plot_order_index`, `lead_time_minutes_to_failure`, `run_family`
- `artifact_paths`: all exported file paths

After building the truth record, `stamp_truth_columns` adds `meta__truth_hash`, `meta__parent_truth_hash`, and `meta__pipeline_mode` to the `anomaly_timeline_dataframe`. The annotated timeline is then exported with these columns present. The truth record is saved via `save_truth_record` and registered in the truth index via `append_truth_index`.

### No Downstream Chain

The workflow reference confirms the Gold 05 truth record does not chain to Gold_06A or Gold_06B. The truth system records Gold 05 in the index as `gold_anomaly_detection`, but Gold_06A operates from its own input contracts and artifact paths rather than consuming a parent hash from Gold 05.

## Downstream Technical Handoff

The only confirmed handoff artifact for another notebook is `multi_run_lead_time_comparison.csv`, which is consumed optionally by Gold_06B. The workflow reference states Gold_06B uses this with a graceful fallback if absent.

The `anomaly_timeline_dataframe` Parquet export and `detection_summary_payload` JSON are the most structured artifacts available for downstream use, but no direct file-level handoff contract from Gold_06A or Gold_06B to these artifacts is confirmed from available Gold_05 source.

Gold_05 SQL rows in `capstone.pipeline_runs` and `capstone.data_quality_events` serve as metadata records for pipeline monitoring and are not consumed by downstream notebooks as primary inputs.

## Key Technical Decisions

| Decision | Source Evidence | Why It Matters | Verification Method |
|---|---|---|---|
| `SELECTED_RUN_KEY` reconfigures all analysis | Cell 15 `RUN_CONFIG_MAP` and resolved variables used throughout | Changing the key rewires the entire notebook — flag column, score column, exports, and truth record — without requiring manual updates to downstream cells | Run with `SELECTED_RUN_KEY = "baseline"` and verify `TARGET_FLAG_COLUMN == "baseline_flag"` in all plot and summary calls |
| Globals-first data loading | Cell 33 `load_selected_results_from_utils` with `globals()` probe | Avoids reloading large scored DataFrames from disk when Gold_05 follows an upstream Gold notebook in the same kernel session | Confirm logger emits "Using in-memory ... from globals" when the upstream variable is present |
| Failure anchor as a hard prerequisite | Cell 41 `add_broken_anchor_columns` raises `ValueError` if no BROKEN row | All phase boundaries, lead-time calculations, and detection classifications are anchored to the first BROKEN row; there is no meaningful timeline without it | Call `add_broken_anchor_columns` on a DataFrame with no BROKEN rows and confirm `ValueError` |
| Synthetic `plot_order_index` x-axis | Cell 39 notebook comment; `np.arange` assigned after optional time sort | Ensures consistent x-axis for window extraction, phase boundaries, and plots regardless of whether a time column exists in the scored output | Confirm `plot_order_index` is monotone integer 0..N-1 even when `time_axis_column` is None |
| Backward scan for forward run length | Cell 45 `compute_forward_stable_normal_run`; single-pass backward loop | Avoids a forward look-ahead pass; computes how many consecutive normal-like rows follow each row in one O(N) scan | Verify `forward_normal_like_run_length[i]` equals the length of the consecutive normal run starting at row i |
| 30-row recovery stability threshold | Cell 47 `RECOVERY_STABILITY_ROWS = 30`; `resolve_recovery_boundaries` | Prevents a single anomaly-free row from prematurely ending recovery; requires a stable run of 30 rows before declaring recovery start | Inject a single normal row post-failure and confirm recovery is not declared; inject 30+ consecutive normal rows and confirm recovery_end is set |
| episode_phase-based detection classification | Cell 51 `classify_detection_rows`; phase used rather than flag alone | Separates early-warning successes from false positives at the alert-row level; a count of "early_warning" is operationally meaningful where a raw alert count is not | Confirm alert rows in `pre_failure` are classified `early_warning` and alert rows in `stable_normal` are classified `false_positive` |
| Alert packets with max gap | Cell 86 `build_alert_packet_summary`; `ALERT_PACKET_MAX_GAP_ROWS = 5` | Surfaces burst alarm patterns rather than isolated points; a 5-row gap threshold keeps closely spaced alerts in the same packet | Confirm two alert rows separated by 4 rows share a packet ID; confirm two rows separated by 6 rows get different packet IDs |
| Top-K packet sort priority | Cell 136 sort by `contains_pre_failure_alert DESC`, `rows_from_packet_start_to_broken DESC` | Puts diagnostically valuable pre-failure packets first in the review table rather than ordering by sequence | Confirm the top row of `top_alert_packets_df` has `contains_pre_failure_alert = True` when any pre-failure packet exists |
| Multi-run comparison covers all 6 keys | Cell 118 `LEAD_TIME_RUN_KEYS`; each key runs its own full pipeline | Provides a side-by-side lead-time table regardless of which run is selected for deep analysis; selected run key does not limit comparison coverage | Confirm `lead_time_comparison_df` always has 6 rows regardless of `SELECTED_RUN_KEY` value |
| `stage3_medium` and `stage3_strict` share one artifact | Cell 31 `RUN_RESULT_PATH_MAP`; both point to `cascade_stage3_improved_results_file_name_pickle` | Both operating modes are embedded in the single Gold_03c scored output; loading both from the same artifact is correct, not an oversight | Confirm both `RUN_RESULT_PATH_MAP["stage3_medium"]` and `["stage3_strict"]` resolve to the same path |
| Parent truth hash from scored results | Cell 146 `extract_truth_hash(selected_results)` with notebook comment | Links Gold_05 truth record to the specific upstream run that produced the scores without requiring config to track the upstream hash separately | Confirm `gold05_truth_base["parent_truth_hash"]` matches the `meta__truth_hash` value embedded in `selected_results` |
| Strict sensor regex `^sensor_\d{2}$` | Cell 108 `resolve_sensor_columns`; pattern excludes derived columns | Prevents profile columns, boolean flag columns, or alert indicators with a `sensor_` prefix from entering multi-sensor plots | Confirm `sensor_profile_min`, `sensor_00_alert`, and similar columns are excluded from the resolved sensor list |
| Display-only sensor normalization | Cell 110 `normalize_sensor_columns_for_plot` operates on a copy | Keeps original sensor values in the exported Parquet and truth record; normalization is exclusively a visual rendering step | Confirm `anomaly_timeline_dataframe["sensor_00"]` is unchanged before and after visualization calls |
| No W&B run opened | Cell (W&B section); only `set_wandb_dir_from_config` called | Gold_05 is an analysis and reporting notebook, not a model training or tracking notebook; W&B run overhead is not appropriate for timeline annotation | Confirm no `wandb.init()` call appears in the notebook source |
| `globals().get()` for SQL manifest | Cell 151; prevents `NameError` for optional frames | `alert_packet_summary_df` and similar frames may not exist when detection produced no alerts; `globals().get()` returns `None` safely and the loop skips None entries | Confirm the SQL manifest correctly excludes `alert_packet_summary` when `alert_packet_summary_df` is empty |
| `WRITE_TO_POSTGRES = True` hardcoded | Cell 151 constant | SQL metadata logging is always on in Gold_05; there is no conditional gate to skip it | Confirm the SQL write and post-write queries always execute regardless of detection results |

## Failure Modes and Guardrails

| Failure Condition | Behavior | Prevented By |
|---|---|---|
| No BROKEN row in scored DataFrame | `ValueError`: "No BROKEN row found in scored dataframe" | `add_broken_anchor_columns` hard stop |
| `target_flag_column` absent from scored DataFrame | `ValueError`: "Missing required target flag column" | `add_alert_and_normal_like_columns` check |
| `meta__row_id` contains null values | `ValueError` | `ensure_row_id_and_plot_order` null check |
| `meta__row_id` is not unique | `ValueError` | `ensure_row_id_and_plot_order` uniqueness check |
| Status column not found among candidates | `ValueError` via `resolve_first_present_column` with `required=True` | Pre-timeline status column resolution |
| `SELECTED_RUN_KEY` not in `RUN_CONFIG_MAP` | `ValueError`: "Unsupported selected_run_key" | `resolve_run_config` guard |
| Globals empty and disk artifact missing | `FileNotFoundError` or `pd.read_pickle` failure | No graceful fallback; load fails |
| `stage3_medium`/`stage3_strict` artifact missing | Both run keys fail to load; multi-run comparison will have missing rows | Shared artifact design; both fail together |
| Empty detection results (no alerts) | Alert packet and top-K exports are skipped; summary payload carries `total_final_alert_rows = 0` | Conditional exports on non-empty DataFrame |
| No recovery period found (insufficient stable run) | `recovery_end_plot_order_index = None`; all post-failure rows labeled `"recovery"` | Designed fallback; not a raise |
| No raw sensor columns matching strict regex | `ValueError`: "No raw sensor columns found" from `resolve_sensor_columns` | Regex check with explicit error |
| Optional output frames not in globals at SQL time | `globals().get()` returns `None`; frame skipped in manifest | Designed defensively; not a raise |
| SQL write failure | Post-write verification queries return empty results; no automatic retry | Visual confirmation from verification query display |
| Database connection failure | `get_engine_from_env()` raises; smoke check fails before any data operations | SQL smoke check gates all DB operations |
| Bootstrap context variable missing | `NameError` with list of missing variable names | Two sequential sanity checks |
| `ANOMALY_DETECTION_CFG` missing | `NameError`: "Missing Gold context variables" | Gold-specific sanity check before any config access |

## Verification Checklist

- Active notebook path is `notebooks/experiments/EDA_Notebook_Pump_Gold_05_Anomaly_Detection.ipynb`
- `SELECTED_RUN_KEY` is set to the intended model variant before run
- Scored results DataFrame exists and contains `machine_status` (or an equivalent status column), `TARGET_FLAG_COLUMN`, and `meta__row_id`
- At least one row with `machine_status == "BROKEN"` is present in the scored results
- `anomaly_timeline_dataframe` contains `episode_phase`, `detection_class`, `selected_final_alert_flag`, `plot_order_index`, `forward_normal_like_run_length`
- `detection_summary_payload` contains `lead_rows_to_failure`, `lead_time_minutes_to_failure`, `total_final_alert_rows`, and `detection_class_counts`
- `failure_lead_time_df` contains a single row with the selected run key and lead-time values
- `lead_time_comparison_df` has 6 rows, one per `LEAD_TIME_RUN_KEYS` entry
- `{SELECTED_RUN_KEY}__timeline_export.parquet` exists in `ANOMALY_DETECTION_EXPORT_DIR`
- `{SELECTED_RUN_KEY}__detection_summary.json` exists in `ANOMALY_DETECTION_SUMMARY_DIR`
- `multi_run_lead_time_comparison.csv` exists in `ANOMALY_DETECTION_SUMMARY_DIR`
- Gold_05 truth record exists with `layer_name = "gold_anomaly_detection"` and `parent_truth_hash` matching the upstream scored results
- `anomaly_timeline_dataframe` columns include `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` after truth stamping
- `GOLD05_LEDGER_PATH` exists and contains expected step names
- `capstone.pipeline_runs` post-write query returns a row for `pipeline_stage = 'gold_anomaly_detection_summary'`
- `capstone.data_quality_events` post-write query returns a row for `check_name = 'gold_05_summary_sql_log'`
- No W&B `wandb.init()` call appears in the notebook source
- `stage3_medium` and `stage3_strict` entries in `RUN_RESULT_PATH_MAP` both resolve to the same pickle artifact path

## Source-Limited Items

- Direct invocation path for `multi_run_lead_time_comparison.csv` by Gold_06B is confirmed by the workflow reference as optional with graceful fallback, but the exact loading mechanism in Gold_06B is Not determined from available Gold_05 source.
- Whether Gold_06A or Gold_06B read the annotated timeline Parquet as a pipeline input is Not determined from available Gold_05 source.
- The exact fallback behavior in `load_selected_results_from_utils` when the disk artifact path is invalid (no valid suffix, wrong extension) is Not determined from available source beyond "uses `pd.read_pickle` or `load_data`."
- Whether `failure_lead_time_minutes_to_failure` and `lead_rows_to_failure` are guaranteed to be equal at all times or can diverge under specific configurations is Not determined from available source — the workflow reference describes them as the same value stored under two keys.
- The behavior of `plot_comparison_overlay` when the `COMPARISON_RUN_KEY` result is unavailable in globals and the disk artifact is missing is Not determined from available source.
- Whether the `RECOVERY_STABILITY_ROWS = 30` constant is config-driven or always a notebook constant is Not determined from available source — it appears as a notebook constant in Cell 15 with no config override shown.
