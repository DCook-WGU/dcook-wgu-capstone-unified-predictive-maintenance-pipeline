# Gold 06B Deep Technical Reference

## Purpose of This Deep Reference

This document covers technical decisions in Gold 06B that require deeper explanation than the workflow reference provides. The workflow reference describes what the notebook does step by step. This document explains why early-warning timing is measured rather than precision/recall, why the notebook accepts partial Gold 06A outputs, why `machine_status` is preferred over the label column for failure detection, why the Gold 05 comparison is structured as an optional left merge, why `selected_run_key` and `target_flag_column` use different naming conventions, and why Gold 06B is terminal rather than producing pipeline artifacts for a downstream notebook.

## Technical Scope

- Terminal notebook role and test-mode consistency with Gold 06A
- Required vs optional upstream artifacts: Gold 06A scores (required) vs Gold 05 lead-time CSV (optional)
- `FileNotFoundError` validation gate enforcing execution order
- `machine_status` preference over label column for failure anchor
- `plot_order_index` synthesis when absent from replay scores
- Defensive copy before adding derived columns
- Available-spec filtering for partial Gold 06A output tolerance
- Lead-time timing metric design: onset position, not precision/recall
- `alerts_before_failure` vs `alerts_at_or_after_failure` split
- `selected_run_key` naming aligned to Gold 05 vs `target_flag_column` naming aligned to Gold 06A
- Optional Gold 05 comparison: left merge with graceful fallback
- `lead_time_delta_minutes` as test-vs-training divergence signal
- Stage 3 Improved timeline plot: conditional on flag column presence
- No W&B, no SQL, no truth record design
- Shared `VALIDATION_ROOT` with Gold 06A

## Source Grounding

Sources used:

- `notebooks/experiments/EDA_Notebook_Pump_Gold_06B_Test_Early_Warning_Validation.ipynb`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_06B_Test_Early_Warning_Validation_code_reference.md`
- `notebook_inventory.json`
- `artifact_io_manifest.json`
- `sql_touchpoints.json`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_05_Anomaly_Detection_deep_technical_reference.md`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation_deep_technical_reference.md`
- `technical_reference/00_project_manual/notebook_relationship_map.md`
- `technical_reference/00_project_manual/notebook_dependency_matrix.md`

The active Gold 06B notebook source is the source of truth for all function behavior, variable names, and design decisions documented here.

## Stage Role in the Final Gold Validation Sequence

Gold 06B is the terminal notebook in the Gold validation sequence. Its source markdown states: "This notebook closes the Gold validation sequence and supports final submission interpretation." No downstream pipeline notebook is identified in the source; all outputs are submission-facing analysis artifacts.

Gold 06B applies Gold 05-style early-warning analysis to the held-out test outputs produced by Gold 06A. Gold 05 performs early-warning analysis on training-run scores. Gold 06B performs equivalent analysis on the replayed test scores from Gold 06A. The resulting comparison between test-replay lead times and training-run lead times provides evidence of whether the model's early-warning timing generalizes from training data to unseen held-out data.

`CONTEXT_STAGE = "gold_test_early_warning_validation"` distinguishes Gold 06B outputs in the ledger and artifact tree from Gold 06A's `gold_model_replay_validation` stage.

Gold 06B does not train models, apply decision thresholds, re-score sensor data, write to PostgreSQL, open a W&B run, or construct a truth record.

## Input Contract and Lineage

### Required Input: Gold 06A Replay Scores

`REPLAY_SCORES_PATH = VALIDATION_SCORES_DIR / f"{DATASET_NAME}__gold06a__test_replay_scores.csv"`

This file must exist before Gold 06B can run. The notebook raises `FileNotFoundError` immediately if it is absent. The code comment explains the design intent: "The explicit existence check acts as a validation gate: 06B cannot run until 06A has written the replay score output, enforcing notebook order."

The replay scores CSV is Gold 06A's wide test-row output: each row is one held-out test observation, and each column group is prefixed by model key (`baseline__`, `cascade_default__`, `cascade_tuned__`, `stage3_improved__`). Gold 06B reads the prefixed flag columns from this file rather than loading any fitted model, threshold, profile, or config artifact directly.

### Optional Input: Gold 05 Training Lead Time

`TRAIN_LEAD_TIME_PATH = GOLD_ROOT / "anomaly_detection" / "summaries" / "multi_run_lead_time_comparison.csv"`

This is the multi-run lead-time comparison CSV produced by Gold 05. Its presence is checked with `.exists()` before any use. When absent, the comparison section of Gold 06B produces a copy of the test-replay summary with a `gold05_comparison_available = False` column rather than raising. This graceful fallback means Gold 06B is fully executable even when Gold 05 has not been run or its outputs have not been preserved.

### No Other Upstream Dependencies

Gold 06B loads only two upstream artifact paths. It does not load model joblibs, threshold JSONs, reference profiles, Gold 04 comparison outputs, Gold 06A metrics CSVs, or Gold 06A comparison CSVs. All scoring and metric computation is delegated entirely to the Gold 06A replay scores CSV and the Gold 05 lead-time reference.

### Dataset Identity

`DATASET_NAME = CONTEXT_DATASET` — set directly from the context constant, consistent with Gold 06A. No `DATASET_ID`/`RUN_ID` resolution is performed; SQL writes are absent.

## Final Validation Preparation Methodology

### Existence Gate and Data Load

The replay scores path is checked with `REPLAY_SCORES_PATH.exists()` before `pd.read_csv` is called. This makes the absence of Gold 06A output an explicit, named error rather than a downstream `FileNotFoundError` from pandas. The message identifies the specific file and names Gold 06A as its source.

After loading, the label column is resolved by probing candidates `["anomaly_flag", "is_anomaly", "target_flag", "label"]`. A `KeyError` is raised if none are found. This probe-based approach is the same pattern used in Gold 06A and accommodates different dataset naming conventions without hardcoding a column name.

### `plot_order_index` Synthesis

If `plot_order_index` is absent from the replay scores CSV, it is synthesized as `np.arange(len(replay_scores_dataframe), dtype=int)`. The code comment states: "Synthesize plot_order_index when absent so early-warning timing logic has a stable integer position axis even on test-only dataframes without a time column." Gold 06A assigns `plot_order_index` before writing the scores CSV, so this fallback handles edge cases where the upstream CSV is missing that column.

### Defensive Copy Before Column Addition

Cell 10 contains: "Copy before adding columns so the loaded CSV state is not mutated in place; this matters if the cell is re-run without reloading." `replay_scores_dataframe = replay_scores_dataframe.copy()` is called before `actual_failure_flag` is added. This prevents a cell re-run from applying `resolve_failure_flag` to a DataFrame that already has the derived column.

### Available-Spec Filtering

Gold 06B defines 7 run specs. `available_run_specs` filters these to only those whose `target_flag_column` is present in the loaded CSV. Missing specs are logged as warnings rather than raising. The code comment states: "Filter to only specs whose flag column is present in the replay output; this makes 06B runnable against partial replay outputs without crashing." This design anticipates scenarios where Gold 06A was run for only a subset of model variants or where the Stage 3 operating-mode flag columns are absent.

## Final Validation Methodology

### Timing-Based Metric Design

The core methodological decision in Gold 06B is documented in the Cell 14 comment: "Early-warning validation measures onset detection timing (first alert vs first BROKEN) rather than precision/recall, because the goal is advance notice, not alert count."

Gold 06A computes precision, recall, F1, and confusion matrix counts — binary classification metrics that measure how many rows the model correctly labeled. Gold 06B measures something different: how early the model fired relative to the first failure event. A model that achieves high precision may still alert only after the failure has begun; a model that achieves a large positive `lead_rows_to_failure` value provided actionable advance notice. These are complementary measures that ask different questions about model utility.

### `lead_rows_to_failure` Computation

`build_early_warning_summary` computes `lead_rows = first_failure - first_alert`. A positive value means the first alert preceded the first failure event. `None` is returned when either index is absent (no alert fired, or no failure row found). `lead_time_minutes_to_failure` is set equal to `lead_rows_to_failure`; both fields carry the same value in row units via `plot_order_index`, not calendar minutes. The field name includes "minutes" for compatibility with Gold 05's `lead_time_minutes_to_failure` column name, which uses the same row-based measurement.

### `alerts_before_failure` vs `alerts_at_or_after_failure`

`build_early_warning_summary` partitions alert rows by position relative to the first failure event:

- `alerts_before_failure`: rows where the flag fires and `plot_order_index < first_broken_plot_order_index`
- `alerts_at_or_after_failure`: rows where the flag fires and `plot_order_index >= first_broken_plot_order_index`

This split distinguishes genuine early-warning alerts (fired before failure) from failure-coincident or post-failure false alarms. Both counts are included in the summary row for each model variant.

### Failure Anchor: `machine_status` Over Label Column

`resolve_failure_flag` uses `machine_status` values matching `"broken|failure|failed"` (case-insensitive regex) as the primary failure anchor when available. The code comment states: "machine_status BROKEN is preferred over the label column because it reflects the operational failure event used throughout the project narrative. The label column is a fallback for datasets that lack machine_status."

The distinction matters for timing calculations. `machine_status == "BROKEN"` marks the operational failure state consistently used by all Gold training and analysis notebooks for phase boundary detection. The label column may mark anomalies that begin before the operational failure state is reached. Using `machine_status` as the primary anchor keeps Gold 06B's timing reference consistent with the Gold 05 episode-phase convention.

### No Hard Validation Status

Gold 06B does not compute a pass/fail or `final_validation_status` field. It reports early-warning timing observations rather than producing an accept/reject decision. The `final_validation_status` concept from Gold 06A (which validates reproducibility of model metrics) does not apply here because there is no expected lead time value to compare against — the Gold 05 training-run lead time is a reference for understanding test behavior, not a threshold that the test replay must meet.

## Relationship to Gold 06A

Gold 06A and Gold 06B address different validation questions:

- **Gold 06A** validates reproducibility: does re-applying the saved model artifacts to held-out test rows produce the same binary classification metrics (precision, recall, F1, alert count) that the training notebooks reported? It re-scores rows using loaded model joblibs, produces `final_validation_status` per model variant, and writes the row-level scored output.
- **Gold 06B** validates early-warning timing: does the replayed model detect the failure event early enough on held-out test data, and how does that timing compare to the training-run early-warning reference? It reads Gold 06A's row-level scored output without any model loading, computes timing metrics, and optionally compares against Gold 05.

The separation exists because these are genuinely different questions that require different artifacts. Gold 06A needs the fitted models to verify reproducibility. Gold 06B needs only the scored flag columns. Combining them would conflate metric validation with timing analysis.

Gold 06B's required input is Gold 06A's scored output. This creates a hard sequential dependency: Gold 06A must complete successfully before Gold 06B can run. The `FileNotFoundError` gate enforces this dependency without requiring any shared session state or kernel variable.

Both notebooks run in `CONFIG_RUN_MODE = "test"`. Both write into the same `VALIDATION_ROOT = GOLD_ROOT / "model_replay_validation"` artifact tree.

## Final Artifact and Report-Readiness Checks

### Required Artifact Gate

The only hard artifact check is `REPLAY_SCORES_PATH.exists()`. This gate prevents any analysis from running against a missing or stale Gold 06A output.

### Available-Column Filtering

`available_run_specs` filtering acts as a column-level existence check for each model variant's flag column. Missing columns are tolerated with a warning rather than a hard failure. This is appropriate because Gold 06B's outputs remain valid for the variants that are present.

### Gold 05 Comparison Availability Flag

`gold05_training_lead_time_available` is recorded in the summary JSON payload as a boolean. A reviewer can inspect this field to confirm whether the Gold 05 comparison was populated. When `False`, `lead_time_comparison_dataframe` contains `gold05_comparison_available = False` for all rows rather than `lead_time_delta_minutes` values.

### Label Column Validation

The label column probe raises `KeyError` if no candidate column is found in the replay scores DataFrame. This prevents timing computation from proceeding without a ground-truth failure indicator.

## Validation Result Construction

### `early_warning_summary_dataframe` Structure

One row per available model variant. Each row contains:

- `selected_run_key`, `plot_run_label`, `target_flag_column`
- `row_count` — total rows in the replay scores DataFrame
- `first_alert_plot_order_index` — first position where the model flag fired (`None` if no alerts)
- `first_broken_plot_order_index` — first position where `actual_failure_flag` is 1 (`None` if no failure rows)
- `lead_rows_to_failure` — `first_failure - first_alert` (`None` if either is absent)
- `lead_time_minutes_to_failure` — same value as `lead_rows_to_failure` (row units)
- `total_final_alert_rows` — sum of the model flag column
- `total_failure_rows` — sum of `actual_failure_flag`
- `alerts_before_failure` — alert rows where `plot_order_index < first_broken_plot_order_index`
- `alerts_at_or_after_failure` — alert rows where `plot_order_index >= first_broken_plot_order_index`

### `lead_time_comparison_dataframe` Structure

When Gold 05 lead time is available: a left merge of `early_warning_summary_dataframe` with `train_lead_time_dataframe` on `selected_run_key`, with suffixes `"_test_replay"` and `"_gold05"`. An additional `lead_time_delta_minutes` column is added: `lead_time_minutes_to_failure_test_replay - lead_time_minutes_to_failure_gold05`. A negative delta means the test-replay alert arrived later than the training-run alert for that variant.

When Gold 05 lead time is absent: a copy of `early_warning_summary_dataframe` with `gold05_comparison_available = False`.

### `selected_run_key` vs `target_flag_column` Naming Convention

Gold 06B's run specs use two different naming schemes for each model variant:

- `selected_run_key` (e.g., `"cascade_defaults"` with trailing 's'): matches the Gold 05 `lead_time_comparison_df` naming convention, enabling the merge on `selected_run_key` to align rows correctly.
- `target_flag_column` (e.g., `"cascade_default__cascade_final_flag"` without trailing 's'): matches the column prefix naming convention used in the Gold 06A replay scores CSV.

This dual naming is required because the Gold 05 multi-run comparison and the Gold 06A scores CSV use different naming conventions for the same model variant. The run spec structure carries both fields so each can be used in the appropriate context without renaming columns.

### Summary JSON Payload

The `summary_payload` saved to `{DATASET_NAME}__gold06b__test_early_warning_summary.json` contains: `stage`, `dataset`, `recipe_id`, `replay_scores_path`, `early_warning_summary_path`, `lead_time_comparison_path`, `lead_plot_path`, `timeline_plot_path`, `model_count`, `gold05_training_lead_time_available`, and `note`. This provides a compact summary of what Gold 06B produced and whether the Gold 05 comparison was available.

### Visualization Outputs

**Lead-time bar plot** — bars of `lead_time_minutes_to_failure` per `plot_run_label`, produced only for variants where the value is not null (i.e., at least one alert fired before the failure event). Saved at 150 dpi.

**Stage 3 Improved timeline plot** — line chart of `actual_failure_flag` and `stage3_improved__cascade_final_flag` over `plot_order_index`. Produced only when `stage3_improved__cascade_final_flag` is present in `replay_scores_dataframe`. This plot specifically highlights the selected/recommended operating mode rather than all seven variants. Saved at 150 dpi.

## Artifact and SQL Persistence

### Output Artifacts

| Artifact | Directory | Filename |
|---|---|---|
| Early-warning summary (per model) | `VALIDATION_SUMMARY_DIR` | `{DATASET_NAME}__gold06b__test_early_warning_summary.csv` |
| Test replay vs Gold 05 comparison | `VALIDATION_SUMMARY_DIR` | `{DATASET_NAME}__gold06b__test_vs_gold05_lead_time_comparison.csv` |
| Summary JSON | `VALIDATION_SUMMARY_DIR` | `{DATASET_NAME}__gold06b__test_early_warning_summary.json` |
| Lead time bar plot | `VALIDATION_PLOTS_DIR` | `{DATASET_NAME}__gold06b__test_lead_time_comparison.png` |
| Stage 3 Improved timeline plot | `VALIDATION_PLOTS_DIR` | `{DATASET_NAME}__gold06b__test_stage3_improved_timeline.png` |

CSVs are written with `.to_csv(..., index=False)`. The summary JSON uses `save_json`. Plot files use `fig.savefig(..., dpi=150)`.

### Shared `VALIDATION_ROOT` With Gold 06A

Gold 06B writes its outputs into the same `model_replay_validation` artifact tree as Gold 06A. Both notebooks write to `VALIDATION_SUMMARY_DIR` and `VALIDATION_PLOTS_DIR`. Gold 06B does not write to `VALIDATION_SCORES_DIR`, which is Gold 06A's write target. This shared directory structure keeps all replay validation outputs together under one path for audit and submission support.

### No SQL Writes

Gold 06B does not write to PostgreSQL. No `get_engine_from_env()` call appears. No SQL touchpoints are confirmed from available source.

### No W&B Uploads

Gold 06B does not call `wandb.init()` and does not upload artifacts to W&B.

### No Truth Record

Gold 06B does not construct a truth record. No `initialize_layer_truth`, `stamp_truth_columns`, or `save_truth_record` calls appear in the source. The validation outputs do not carry `meta__truth_hash` or `meta__parent_truth_hash` columns.

### Ledger

The `Ledger` is provided by `CTX.ledger`. Ledger steps added:

- `"context_loaded"` — bootstrap complete
- `"gold06a_replay_scores_loaded"` — replay scores loaded; records `replay_scores_path`, `row_count`, `label_column`, `anomaly_rows`, `train_lead_time_artifact_available`
- `"gold06b_outputs_saved"` — all outputs saved; records the full summary payload

The exact ledger write path is Not determined from available source beyond CTX management.

## Truth, Audit, and Reproducibility Behavior

Gold 06B preserves auditability through its summary JSON rather than through a truth record.

The summary JSON records `replay_scores_path` (the exact Gold 06A artifact that was read), `gold05_training_lead_time_available` (whether the comparison was populated), `recipe_id`, and `dataset`. A reviewer inspecting the summary JSON can determine precisely which artifacts were used and whether the Gold 05 comparison was available.

The `selected_run_key` values in the early-warning summary allow rows to be traced back to specific model variants in both the Gold 06A comparison DataFrame and the Gold 05 training-run lead-time reference.

The lead-time bar plot and timeline plot provide visual audit evidence of early-warning timing behavior on held-out test data. These are the primary submission-facing deliverables for demonstrating model early-warning capability.

## Downstream Technical Handoff

Gold 06B is the terminal notebook in the Gold validation sequence. No downstream pipeline notebook is identified in the source. The workflow reference states: "This notebook closes the Gold validation sequence and supports final submission interpretation."

The early-warning summary CSV, lead-time comparison CSV, summary JSON, and plot PNGs are the submission-facing outputs. Whether any of these are consumed by a final report or appendix notebook is Not determined from available source.

## Key Technical Decisions

| Decision | Source Evidence | Why It Matters | Verification Method |
|---|---|---|---|
| Terminal notebook design — no downstream consumers | Cell 17 source markdown; "closes the Gold validation sequence and supports final submission interpretation"; no `## Next Stage` pipeline reference | Reflects that Gold 06B produces submission-facing evidence, not pipeline artifacts; prevents misuse as a pipeline stage with downstream consumers | Confirm no downstream notebook loads any `gold06b` artifact path |
| `CONFIG_RUN_MODE = "test"` consistent with Gold 06A | Cell 4 bootstrap constant | Keeps both validation notebooks in test mode; Gold 06B's early-warning analysis is on held-out test rows, not training rows | Confirm `CTX.mode == "test"` |
| `FileNotFoundError` gate on Gold 06A replay scores | Cell 6 existence check; code comment: "acts as a validation gate: 06B cannot run until 06A has written the replay score output" | Enforces notebook execution order without requiring a shared kernel session; prevents silently running against stale or missing scored outputs | Remove the replay scores CSV temporarily and confirm `FileNotFoundError` with the expected message |
| Gold 05 lead-time as optional with graceful fallback | Cell 12 `TRAIN_LEAD_TIME_PATH.exists()` check; `gold05_comparison_available = False` fallback | Gold 06B is executable without Gold 05 having been run; decouples final test validation from the training-run analysis notebook | Confirm Gold 06B produces all outputs including comparison CSV when `TRAIN_LEAD_TIME_PATH` does not exist |
| Timing metric rather than precision/recall | Cell 14 comment: "measures onset detection timing (first alert vs first BROKEN) rather than precision/recall, because the goal is advance notice, not alert count" | Gold 06A already validated binary classification metrics; Gold 06B asks a different question — does the model fire early enough to be operationally useful on unseen data? | Confirm `early_warning_summary_dataframe` contains `lead_rows_to_failure` but not `precision`, `recall`, or `f1` |
| `machine_status` preferred over label column | Cell 8 `resolve_failure_flag`; code comment: "machine_status BROKEN is preferred because it reflects the operational failure event used throughout the project narrative" | Keeps timing anchor consistent with the Gold 05 phase-boundary definition; `machine_status == BROKEN` is the event the whole pipeline narrative references | Confirm `actual_failure_flag` column is derived from `machine_status` BROKEN values when that column is present with matching rows |
| `plot_order_index` synthesis when absent | Cell 6 code comment: "Synthesize when absent so early-warning timing logic has a stable integer position axis" | Prevents timing computation from failing on a replay scores CSV that lacked `plot_order_index`; Gold 06A assigns it, but the fallback handles edge cases | Remove `plot_order_index` from a copy of the replay CSV and confirm `early_warning_summary_dataframe` is still produced |
| Defensive copy before `actual_failure_flag` addition | Cell 10 code comment: "Copy before adding columns so the loaded CSV state is not mutated in place; this matters if the cell is re-run" | Prevents accumulated column additions on re-run from producing incorrect `actual_failure_flag` values or raising duplicate-column errors | Re-run Cell 10 twice without reloading Cell 6 and confirm `actual_failure_flag` values are unchanged |
| `available_run_specs` filtering for partial tolerance | Cell 10 code comment: "makes 06B runnable against partial replay outputs without crashing" | Gold 06B remains usable when only a subset of model variants are present in the replay scores CSV; missing specs are warned, not raised | Run against a replay scores CSV with `stage3_improved__cascade_stage3_strict_flag` absent and confirm the remaining 6 rows are produced |
| `alerts_before_failure` vs `alerts_at_or_after_failure` | Cell 8 `build_early_warning_summary`; both counts computed relative to `first_broken_plot_order_index` | Separates genuine early-warning alerts from failure-coincident and post-failure alerts; a model with many alerts after failure but few before is not operationally useful for advance warning | Confirm `alerts_before_failure + alerts_at_or_after_failure == total_final_alert_rows` for each variant |
| `selected_run_key` matches Gold 05 naming; `target_flag_column` matches Gold 06A prefix | Cell 10 `run_specs`; `cascade_defaults` (with 's') in `selected_run_key`; `cascade_default__` (without 's') in `target_flag_column` | Gold 05 CSV and Gold 06A CSV use different naming conventions for the same variant; the run spec carries both so the merge and the column lookup each use the correct name | Confirm the merge on `selected_run_key` aligns `cascade_defaults` rows with Gold 05 data; confirm `cascade_default__cascade_final_flag` is resolved from the Gold 06A CSV |
| Left merge for Gold 05 comparison | Cell 12 `how="left"` in merge; `suffixes=("_test_replay", "_gold05")` | Preserves all test-replay rows even when the Gold 05 CSV lacks a matching `selected_run_key` row; suffix naming makes the disambiguated columns explicit | Confirm `lead_time_comparison_dataframe` has the same row count as `early_warning_summary_dataframe` |
| `lead_time_delta_minutes` as test-vs-training divergence signal | Cell 12 delta computation: `_test_replay - _gold05` | Quantifies whether the model detects failure earlier or later on unseen test data vs training data; a large negative delta signals timing degradation on held-out data | Confirm `lead_time_delta_minutes` is `None` when either suffix column is `NaN` |
| Stage 3 Improved timeline plot conditional on column presence | Cell 14 `if timeline_flag_column in replay_scores_dataframe.columns` | Prevents `KeyError` when the preferred Stage 3 variant is absent from the replay scores CSV; plot is a visualization artifact, not a required output | Remove `stage3_improved__cascade_final_flag` from a copy of the CSV and confirm no exception is raised |
| No W&B, no SQL, no truth record | Cells 4, 14; no `wandb.init()`, no DB connection, no `initialize_layer_truth` | Gold 06B is a terminal analysis notebook producing submission-facing evidence; adding infrastructure overhead is disproportionate to its scope | Confirm no `wandb.init()`, `get_engine_from_env()`, or `initialize_layer_truth` calls appear in source |

## Failure Modes and Guardrails

| Failure Condition | Behavior | Prevention / Guardrail |
|---|---|---|
| Gold 06A replay scores CSV absent | `FileNotFoundError`: "Gold 06B needs the Gold 06A replay score output first: {path}" | Explicit existence check before `pd.read_csv`; names Gold 06A as the responsible notebook |
| Label column not found among candidates | `KeyError` with candidate list | Probe-and-raise in label column resolution block |
| `machine_status` column absent or no BROKEN rows | Falls back to label column for `actual_failure_flag` | `resolve_failure_flag` checks presence and match count before returning `machine_status` flag |
| `plot_order_index` absent from replay CSV | Synthesized as `np.arange(len(replay_scores_dataframe), dtype=int)` | Explicit absence check and synthesis |
| Model variant flag column absent from replay CSV | Spec filtered out of `available_run_specs`; `logger.warning` issued | Filter-before-use pattern; no raise |
| No flag columns present at all in replay CSV | `early_warning_summary_dataframe` is empty; no rows produced | `available_run_specs` is empty list; downstream CSV writes produce empty files |
| No alerts fired for a variant (`first_alert = None`) | `lead_rows_to_failure = None`; bars dropped from lead-time plot (via `dropna`) | `first_flag_index` returns `None` safely; `lead_time_minutes_to_failure` is `None` |
| No failure rows found (`first_broken = None`) | `lead_rows_to_failure = None`; `alerts_before_failure = 0` by default | `first_flag_index` returns `None`; `alerts_before_failure` block guarded by `if first_failure is not None` |
| Gold 05 lead-time CSV absent | `lead_time_comparison_dataframe` becomes copy of early-warning summary with `gold05_comparison_available = False` | `TRAIN_LEAD_TIME_PATH.exists()` check; no raise |
| Gold 05 `selected_run_key` values do not align with Gold 06B keys | Left merge produces NaN Gold 05 columns for unmatched rows | `how="left"` preserves all test-replay rows; unmatched Gold 05 rows are `NaN` |
| `lead_time_minutes_to_failure_test_replay` or `_gold05` column absent after merge | `lead_time_delta_minutes` column is not added | Column presence check before delta computation |
| `stage3_improved__cascade_final_flag` absent from replay CSV | Stage 3 Improved timeline plot is skipped | `if timeline_flag_column in replay_scores_dataframe.columns` guard |
| `lead_time_minutes_to_failure` all null for lead-time bar plot | `plot_df` is empty after `dropna`; `if not plot_df.empty` guard prevents empty bar chart | `dropna(subset=["lead_time_minutes_to_failure"])` and empty check |

## Verification Checklist

- Active notebook path is `notebooks/experiments/EDA_Notebook_Pump_Gold_06B_Test_Early_Warning_Validation.ipynb`
- `CONFIG_RUN_MODE = "test"` is hardcoded in the bootstrap cell
- `{DATASET_NAME}__gold06a__test_replay_scores.csv` exists in `VALIDATION_SCORES_DIR`
- Replay scores CSV contains `plot_order_index` or the synthesis fallback produces it correctly
- Replay scores CSV contains at least one of the 7 expected flag columns (e.g., `baseline__baseline_flag`)
- Label column is present in replay scores CSV (`anomaly_flag`, `is_anomaly`, `target_flag`, or `label`)
- `actual_failure_flag` column is derived from `machine_status` (preferred) or label column (fallback)
- `early_warning_summary_dataframe` has one row per available model variant
- `lead_rows_to_failure` is positive for variants that detected failure before the first BROKEN row
- `{DATASET_NAME}__gold06b__test_early_warning_summary.csv` exists in `VALIDATION_SUMMARY_DIR`
- `{DATASET_NAME}__gold06b__test_vs_gold05_lead_time_comparison.csv` exists in `VALIDATION_SUMMARY_DIR`
- `{DATASET_NAME}__gold06b__test_early_warning_summary.json` exists in `VALIDATION_SUMMARY_DIR`
- Lead-time bar plot exists in `VALIDATION_PLOTS_DIR` (when at least one variant has non-null lead time)
- Stage 3 Improved timeline plot exists in `VALIDATION_PLOTS_DIR` (when `stage3_improved__cascade_final_flag` is present)
- Summary JSON `gold05_training_lead_time_available` reflects whether `TRAIN_LEAD_TIME_PATH.exists()` returned `True`
- When Gold 05 CSV is present: `lead_time_comparison_dataframe` contains `lead_time_delta_minutes` column
- Ledger contains `"context_loaded"`, `"gold06a_replay_scores_loaded"`, `"gold06b_outputs_saved"` steps
- No `wandb.init()`, `get_engine_from_env()`, or `initialize_layer_truth` calls in notebook source
- No downstream pipeline notebook identified in source; outputs are submission-facing

## Source-Limited Items

- Whether any final report, appendix notebook, or submission documentation directly consumes Gold 06B outputs is Not determined from available source.
- The exact ledger write path for `CTX.ledger` is Not determined from available Gold 06B source; no explicit `ledger.write_json(path)` call appears.
- Whether `load_json` (imported in Cell 3 but not called in any visible code cell) is used for a purpose not visible in the available source is Not determined.
- Whether the lead-time bar plot is saved when all variants produce `None` for `lead_rows_to_failure` (i.e., no model fired before failure) is Not determined beyond the `if not plot_df.empty` guard producing no output file.
- The exact behavior when the Gold 05 lead-time CSV is present but does not contain a `lead_time_minutes_to_failure` column under the `_gold05` suffix is Not determined; the column presence check before `lead_time_delta_minutes` computation would prevent a `KeyError` but the delta column would be absent.
