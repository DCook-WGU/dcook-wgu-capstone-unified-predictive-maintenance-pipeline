# EDA_Notebook_Pump_Gold_06B_Test_Early_Warning_Validation — Workflow Reference

**Source notebook:** `notebooks/experiments/EDA_Notebook_Pump_Gold_06B_Test_Early_Warning_Validation.ipynb`
**Stage:** Gold — Test Early-Warning Validation
**Layer:** Gold
**Reference type:** Workflow-level

---

## Notebook Purpose

Gold_06B applies Gold 05-style early-warning analysis to the replayed held-out test outputs produced by Gold_06A. Rather than scoring raw sensor data or rerunning models, it reads the row-level test replay score CSV from Gold_06A, computes per-model early-warning timing metrics (first alert position relative to first failure position), and optionally compares those test-replay lead times against the Gold_05 training-run reference.

The notebook is short — 8 code cells — and proportional to that scope. Its declared role in the source markdown is to close the Gold validation sequence and support final submission interpretation. It does not train models, tune thresholds, write to PostgreSQL, open a W&B run, or construct a truth record.

---

## Pipeline Role

- Stage: `gold_test_early_warning_validation`
- Layer: Gold
- `CONFIG_RUN_MODE`: `"test"`
- Position in workflow: Terminal notebook; runs after Gold_06A (Test Replay Validation); no downstream notebook
- Primary responsibility: Compute per-model early-warning timing metrics from the Gold_06A replay scores; optionally compare test-replay lead times against the Gold_05 training-run reference; produce visualizations and summary artifacts for final submission interpretation
- Does not train models, score rows, write to PostgreSQL, open a W&B run, or construct a truth record

## Configuration and Runtime Context

| Item | Source | Value / Purpose |
|---|---|---|
| `CONTEXT_STAGE` | Notebook constant | `"gold_test_early_warning_validation"` |
| `CONFIG_RUN_MODE` | `"test"` (from bootstrap mode argument) | Consistent with Gold_06A test mode |
| `DATASET_NAME` | `CONTEXT_DATASET` from bootstrap | Dataset identifier |
| `GOLD_ROOT` | `paths.artifacts / "gold" / DATASET_NAME` | Root Gold artifact directory |
| `VALIDATION_ROOT` | `GOLD_ROOT / "model_replay_validation"` | Shared with Gold_06A; Gold_06B writes into same tree |
| `REPLAY_SCORES_PATH` | `VALIDATION_SCORES_DIR / f"{DATASET_NAME}__gold06a__test_replay_scores.csv"` | Required — `FileNotFoundError` if absent |
| `TRAIN_LEAD_TIME_PATH` | `GOLD_ROOT / "anomaly_detection" / "summaries" / "multi_run_lead_time_comparison.csv"` | Optional — Gold_05 training lead-time reference; graceful fallback if absent |

## Section Overview

| Section | Purpose | Key Outputs |
|---|---|---|
| Bootstrap | Context, output dirs | `CTX`, `VALIDATION_SUMMARY_DIR`, `VALIDATION_PLOTS_DIR` |
| Upstream inputs | Declare required and optional artifact paths | Path declarations; `FileNotFoundError` if replay scores absent |
| Data loading | Load replay scores; resolve label column; synthesize `plot_order_index` if absent | `replay_scores_dataframe` |
| Failure flag | `resolve_failure_flag` adds `actual_failure_flag` | `actual_failure_flag` column |
| Early-warning helpers | Define `first_flag_index`, `build_early_warning_summary` | Inline helper functions |
| Per-variant summary | 7 run specs; filtered to available flag columns | `early_warning_summary_dataframe` |
| Gold_05 comparison | Merge test-replay summary vs. Gold_05 lead-time CSV (optional) | `lead_time_comparison_dataframe` |
| Output saves | CSVs, JSON, 2 plot PNGs | 5 artifacts in `VALIDATION_SUMMARY_DIR` / `VALIDATION_PLOTS_DIR` |

## Section Details

## 2. Context Stage and Bootstrap

```python
CONTEXT_STAGE = "gold_test_early_warning_validation"

CTX = load_notebook_context(
    stage=CONTEXT_STAGE,
    dataset="pump",
    mode="test",
    profile="default",
    logger_child_name="capstone.gold.test_early_warning_validation",
    log_filename="gold_test_early_warning_validation.log",
)
```

`CONFIG_RUN_MODE = "test"` — Gold_06B runs in test mode, consistent with Gold_06A.

`ledger = CTX.ledger` — from the bootstrap, not separately instantiated.

`DATASET_NAME = CONTEXT_DATASET` — set directly from the context constant.

`GOLD_ROOT = paths.artifacts / "gold" / DATASET_NAME`

A ledger step `"context_loaded"` is written immediately after bootstrap.

There is no `wandb.init()`. There are no database connections. Gold_06B does not write to PostgreSQL. There is no truth record construction.

---

## 3. Output Directory Setup

Gold_06B writes its outputs into the same `model_replay_validation` artifact tree established by Gold_06A:

```
VALIDATION_ROOT       = GOLD_ROOT / "model_replay_validation"
VALIDATION_RESULTS_DIR  = VALIDATION_ROOT / "results"
VALIDATION_SUMMARY_DIR  = VALIDATION_ROOT / "summaries"
VALIDATION_PLOTS_DIR    = VALIDATION_ROOT / "plots"
```

All three directories are created with `mkdir(parents=True, exist_ok=True)`. Gold_06B does not create or write to `VALIDATION_SCORES_DIR`; that directory is Gold_06A's write target.

---

## 4. Upstream Inputs

Gold_06B declares two upstream artifact paths at the top of its data-loading section:

**Required — Gold_06A replay scores:**
```python
REPLAY_SCORES_PATH = VALIDATION_SCORES_DIR / f"{DATASET_NAME}__gold06a__test_replay_scores.csv"
```
This file must exist before Gold_06B can run. The notebook raises a `FileNotFoundError` with an explicit message if it is absent, enforcing notebook execution order. This is the only required upstream artifact.

**Optional — Gold_05 training lead time:**
```python
TRAIN_LEAD_TIME_PATH = GOLD_ROOT / "anomaly_detection" / "summaries" / "multi_run_lead_time_comparison.csv"
```
This is the multi-run lead time comparison CSV produced by Gold_05. Its presence is checked with `.exists()` before each use. When absent, the comparison section produces a copy of the test-replay summary with a `gold05_comparison_available = False` column rather than raising an error.

---

## 5. Data Loading

```python
replay_scores_dataframe = pd.read_csv(REPLAY_SCORES_PATH)
```

After loading, the label column is resolved by probing candidates `["anomaly_flag", "is_anomaly", "target_flag", "label"]`; a `KeyError` is raised if none are found.

If `plot_order_index` is absent from the CSV, it is synthesized as `np.arange(len(replay_scores_dataframe), dtype=int)`. This ensures early-warning timing logic has a stable integer position axis even when the upstream CSV does not carry a time column.

A ledger step `"gold06a_replay_scores_loaded"` records: `replay_scores_path`, `row_count`, `label_column`, `anomaly_rows`, and `train_lead_time_artifact_available`.

---

## 6. Failure Flag Resolution

Before any early-warning timing is computed, `actual_failure_flag` is added to a copy of the replay DataFrame:

```python
replay_scores_dataframe = replay_scores_dataframe.copy()
replay_scores_dataframe["actual_failure_flag"] = resolve_failure_flag(
    replay_scores_dataframe,
    label_column=label_column,
)
```

`resolve_failure_flag` prefers `machine_status` values matching `"broken|failure|failed"` (case-insensitive regex) over the label column, because the `machine_status` column more directly reflects the operational failure event used throughout the project narrative. The label column is used as a fallback when `machine_status` is absent or contains no matching values.

---

## 7. Early-Warning Helper Functions

All three helper functions are defined inline:

**`first_flag_index(dataframe, flag_column)`** — returns the first `plot_order_index` value where the binary flag column equals 1, or `None` if the column is absent or no rows are flagged.

**`build_early_warning_summary(dataframe, *, run_key, run_label, flag_column, actual_failure_column="actual_failure_flag")`** — builds one summary row for a model variant. Computes:

- `first_alert_plot_order_index` — first row index where the model flag fires
- `first_broken_plot_order_index` — first row index where `actual_failure_flag` is 1
- `lead_rows_to_failure` — `first_broken_plot_order_index - first_alert_plot_order_index` (positive = alert preceded failure; `None` if either is absent)
- `lead_time_minutes_to_failure` — set equal to `lead_rows_to_failure`; measured in row units via `plot_order_index`, not calendar minutes
- `total_final_alert_rows` — total rows where the model flag is 1
- `total_failure_rows` — total rows where `actual_failure_flag` is 1
- `alerts_before_failure` — alert rows with `plot_order_index < first_broken_plot_order_index`
- `alerts_at_or_after_failure` — alert rows with `plot_order_index >= first_broken_plot_order_index`

---

## 8. Per-Variant Early-Warning Summary

Seven run specs are defined, one per model variant and Stage 3 operating mode. Each spec provides a `selected_run_key`, a `plot_run_label`, and the `target_flag_column` that should be present in the Gold_06A replay scores CSV:

| `selected_run_key` | `target_flag_column` |
|---|---|
| `baseline` | `baseline__baseline_flag` |
| `cascade_defaults` | `cascade_default__cascade_final_flag` |
| `cascade_tuned` | `cascade_tuned__cascade_final_flag` |
| `stage3_improved` | `stage3_improved__cascade_final_flag` |
| `stage3_relaxed` | `stage3_improved__cascade_stage3_relaxed_flag` |
| `stage3_medium` | `stage3_improved__cascade_stage3_medium_flag` |
| `stage3_strict` | `stage3_improved__cascade_stage3_strict_flag` |

`available_run_specs` is filtered to only those specs whose `target_flag_column` is present in the loaded CSV. Missing specs are logged with `logger.warning` but do not raise. This makes the notebook runnable against partial Gold_06A outputs.

`build_early_warning_summary` is applied to each available spec, producing `early_warning_summary_dataframe` (one row per available model variant).

---

## 9. Gold_05 Lead-Time Comparison

When `TRAIN_LEAD_TIME_PATH` exists:

```python
train_lead_time_dataframe = pd.read_csv(TRAIN_LEAD_TIME_PATH)
lead_time_comparison_dataframe = early_warning_summary_dataframe.merge(
    train_lead_time_dataframe,
    on="selected_run_key",
    how="left",
    suffixes=("_test_replay", "_gold05"),
)
lead_time_comparison_dataframe["lead_time_delta_minutes"] = (
    lead_time_comparison_dataframe["lead_time_minutes_to_failure_test_replay"]
    - lead_time_comparison_dataframe["lead_time_minutes_to_failure_gold05"]
)
```

`lead_time_delta_minutes` quantifies the difference in onset detection position between the held-out test replay (Gold_06B) and the original training-run analysis (Gold_05). A negative delta means the test-replay alert arrives later than the training-run alert.

When the Gold_05 artifact is absent, `lead_time_comparison_dataframe` is set to a copy of `early_warning_summary_dataframe` with `gold05_comparison_available = False` appended; no error is raised.

---

## Outputs and Artifacts

All file paths resolve under `VALIDATION_SUMMARY_DIR` and `VALIDATION_PLOTS_DIR`:

| Artifact | Directory | Filename |
|---|---|---|
| Early-warning summary (per model) | `VALIDATION_SUMMARY_DIR` | `{DATASET_NAME}__gold06b__test_early_warning_summary.csv` |
| Test replay vs Gold_05 comparison | `VALIDATION_SUMMARY_DIR` | `{DATASET_NAME}__gold06b__test_vs_gold05_lead_time_comparison.csv` |
| Summary JSON | `VALIDATION_SUMMARY_DIR` | `{DATASET_NAME}__gold06b__test_early_warning_summary.json` |
| Lead time bar plot | `VALIDATION_PLOTS_DIR` | `{DATASET_NAME}__gold06b__test_lead_time_comparison.png` |
| Stage 3 Improved timeline plot | `VALIDATION_PLOTS_DIR` | `{DATASET_NAME}__gold06b__test_stage3_improved_timeline.png` |

**Lead-time bar plot** — bar chart of `lead_time_minutes_to_failure` by `plot_run_label` for all model variants where the value is not null. Saved at 150 dpi.

**Stage 3 Improved timeline plot** — line chart of `actual_failure_flag` and `stage3_improved__cascade_final_flag` over `plot_order_index` for the full test set. Produced only when `stage3_improved__cascade_final_flag` is present in `replay_scores_dataframe`. Saved at 150 dpi.

The summary JSON payload contains: `stage`, `dataset`, `recipe_id`, `replay_scores_path`, `early_warning_summary_path`, `lead_time_comparison_path`, `lead_plot_path`, `timeline_plot_path`, `model_count`, `gold05_training_lead_time_available`, and `note`.

A ledger step `"gold06b_outputs_saved"` records the full summary payload after all saves complete.

---

## 11. Interpretation Object

The final code cell produces a display-only `interpretation` dict:

```python
interpretation = {
    "purpose": "Validate early-warning behavior on replayed held-out test outputs.",
    "input": str(REPLAY_SCORES_PATH),
    "primary_output": str(early_warning_summary_path),
    "comparison_output": str(lead_time_comparison_path),
    "scope": "Gold 05-style early-warning review, but on Gold 06A replayed test-set outputs.",
}
```

This object is not saved to disk. The source markdown states: "This notebook closes the Gold validation sequence and supports final submission interpretation."

---

## Inputs

| Source | Artifact | Required |
|---|---|---|
| Gold_06A | `{DATASET_NAME}__gold06a__test_replay_scores.csv` | Yes — `FileNotFoundError` if absent |
| Gold_05 | `multi_run_lead_time_comparison.csv` (anomaly detection summaries dir) | No — graceful fallback |

Gold_06B does not load model artifacts, threshold files, reference profiles, or any other upstream Gold outputs beyond these two files.

---

## Key Function Calls and In-Place Usage

| Function | Source | Purpose |
|---|---|---|
| `load_notebook_context` | `utils.core.notebook_context` | Bootstrap CTX, paths, config, logger, ledger |
| `load_json` | `utils.core.file_io` | (imported but not called in any code cell) |
| `save_json` | `utils.core.file_io` | Save summary JSON |
| `resolve_failure_flag` | inline | Resolve operational failure column; prefers `machine_status` BROKEN over label |
| `first_flag_index` | inline | First `plot_order_index` where a flag fires |
| `build_early_warning_summary` | inline | Per-variant early-warning timing summary row |

---

## Data Quality / Validation Behavior

| Check | Purpose | Failure / Risk Prevented |
|---|---|---|
| `REPLAY_SCORES_PATH` existence check | Required upstream artifact from Gold_06A | `FileNotFoundError` with explicit message; enforces notebook execution order |
| Label column resolution from candidates | Probes `["anomaly_flag", "is_anomaly", "target_flag", "label"]` | `KeyError` if none found; prevents early-warning computation with no failure ground truth |
| `plot_order_index` synthesis when absent | `np.arange(len(...))` fallback if column not in replay CSV | Ensures timing logic has a stable integer position axis regardless of upstream CSV content |
| `available_run_specs` filtering | Filters 7 run specs to only those with flag columns present in loaded CSV | `logger.warning` for missing specs; prevents `KeyError` on absent flag columns |
| Gold_05 lead-time `exists()` check | Checked before each use | Graceful fallback with `gold05_comparison_available = False`; no error raised |

---

## Downstream Handoff

Gold_06B is the terminal notebook in the Gold validation sequence. No downstream notebook is identified in the source. The "## Next Stage" markdown section states: "This notebook closes the Gold validation sequence and supports final submission interpretation."

Its outputs are intended for review and capstone write-up support rather than as inputs to another pipeline notebook.

---

## Logical Workflow Map

1. Imports
2. Context bootstrap (`load_notebook_context`)
3. Config variable extraction from CTX
4. Output directory setup (`VALIDATION_SUMMARY_DIR`, `VALIDATION_PLOTS_DIR`)
5. Ledger step: `context_loaded`
6. Upstream path declarations (`REPLAY_SCORES_PATH`, `TRAIN_LEAD_TIME_PATH`)
7. Gold_06A score CSV existence check → `FileNotFoundError` if absent
8. `pd.read_csv(REPLAY_SCORES_PATH)`
9. Label column resolution; `plot_order_index` synthesis if absent
10. Ledger step: `gold06a_replay_scores_loaded`
11. Helper function definitions (`resolve_failure_flag`, `first_flag_index`, `build_early_warning_summary`)
12. `actual_failure_flag` added via `resolve_failure_flag`
13. `run_specs` definition (7 variant entries)
14. `available_run_specs` filtering; `logger.warning` for missing flag columns
15. `build_early_warning_summary` applied to all available specs → `early_warning_summary_dataframe`
16. Gold_05 lead time CSV load (if available) and merge → `lead_time_comparison_dataframe`
17. CSV saves (early-warning summary, lead-time comparison)
18. Lead-time bar plot (`fig.savefig`)
19. Stage 3 Improved timeline plot (`fig.savefig`)
20. `save_json(summary_payload, summary_json_path)`
21. Ledger step: `gold06b_outputs_saved`
22. `interpretation` dict display

---

## Relationship to Other Notebooks

### Upstream Context

Gold_06B requires Gold_06A's test replay scores CSV (`REPLAY_SCORES_PATH` — `FileNotFoundError` if absent, enforcing execution order). It optionally loads Gold_05's multi-run lead-time comparison CSV (`TRAIN_LEAD_TIME_PATH`) for training-vs-test early-warning comparison, with graceful fallback if absent. No other upstream notebook dependencies are confirmed.

### Downstream Handoff

Gold_06B is the terminal notebook in the Gold validation sequence. It produces early-warning summary artifacts and plots for capstone submission support. No downstream pipeline notebook is identified; all outputs are submission-facing.

### Pipeline Position

Terminal notebook. Closes the Gold validation sequence. Its early-warning analysis on held-out test data provides the final evidence of model behavior on unseen data for capstone submission. Convergence point between Gold_06A's replay outputs and Gold_05's training-run early-warning reference.

### Relationship Summary

- Reads Gold_06A test replay scores CSV (required) and Gold_05 lead-time CSV (optional; graceful fallback)
- Produces early-warning timing artifacts and plots as the final capstone submission-support deliverables
- Does not write to SQL, W&B, or produce a truth record
- Has no downstream pipeline notebook consumers — terminal by design
- Gold_05 comparison is fully optional: absent Gold_05 CSV does not block Gold_06B execution
