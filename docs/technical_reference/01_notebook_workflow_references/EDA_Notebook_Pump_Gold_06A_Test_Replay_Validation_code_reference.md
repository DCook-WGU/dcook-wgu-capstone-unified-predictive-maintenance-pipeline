# EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation — Workflow Reference

**Source notebook:** `notebooks/experiments/EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation.ipynb`
**Stage:** Gold — Model Replay Validation
**Layer:** Gold
**Reference type:** Workflow-level

---

## Notebook Purpose

Gold_06A is a post-training validation notebook. It does not train models, tune thresholds, or score raw data as part of a discovery process. Its job is narrow and deliberate: reload the model and configuration artifacts that were produced by Gold_02, Gold_03A, Gold_03B, and Gold_03C, apply them to the held-out test split from Gold_01, compute test-row metrics from the replayed output, compare those replayed metrics against the saved training-notebook summary artifacts, and write the comparison and row-level score outputs for Gold_06B.

The notebook is small — 11 code cells — and proportional to that scope. It does not open a W&B run, write to PostgreSQL, or construct a truth record.

---

## Pipeline Role

- Stage: `gold_model_replay_validation`
- Layer: Gold
- `CONFIG_RUN_MODE`: `"test"` — the only Gold notebook confirmed to run in test mode
- Position in workflow: Runs after Gold_02 and Gold_03a/03b/03c training notebooks; before Gold_06B (Early Warning Validation)
- Primary responsibility: Reload saved model artifacts, re-score held-out test rows without retraining, compare replayed metrics against training-notebook summary artifacts, and write row-level replay scores for Gold_06B
- Does not open a W&B run, write to PostgreSQL, or construct a truth record

## Configuration and Runtime Context

| Item | Source | Value / Purpose |
|---|---|---|
| `CONTEXT_STAGE` | Notebook constant | `"gold_model_replay_validation"` |
| `CONFIG_RUN_MODE` | `"test"` (hardcoded in bootstrap) | Gold_06A runs in test mode |
| `DATASET_NAME` | `CONTEXT_DATASET` from bootstrap | Dataset identifier |
| `GOLD_ROOT` | `paths.artifacts / "gold" / DATASET_NAME` | Root artifact directory for all Gold outputs |
| `FULL_SCALED_PATH` | `RESOLVED_PATHS["gold_preprocessed_scaled_data_path"]` or fallback | Preferred replay input — full-split Parquet with `meta__is_train_flag` |
| `TEST_DATA_PATH` | `RESOLVED_PATHS["gold_test_data_path"]` or fallback | Portability fallback for replay source |
| `METRIC_TOLERANCE` | Notebook constant `1e-9` | Strict tolerance for exact match validation pass |
| `METRIC_TOLERANCE_RELAXED` | Notebook constant `0.0001` | Secondary tolerance for pass-with-tolerance status |
| `ALERT_COUNT_TOLERANCE` | Notebook constant `1` | Maximum alert count delta for tolerance pass |

## Section Overview

| Section | Purpose | Key Outputs |
|---|---|---|
| Bootstrap | Context, output dirs | `CTX`, `VALIDATION_ROOT` tree |
| Artifact resolution | Build all upstream artifact paths | `MODEL_ARTIFACTS`, threshold/summary/profile/config path dicts |
| Artifact existence check | `artifact_inventory` DataFrame | Ledger step `required_artifacts_found`; `FileNotFoundError` if any missing |
| Data loading | Load replay source, feature lists, models, thresholds, summaries, profiles, configs | `replay_source_dataframe`, 7 fitted models |
| Replay helpers | Define inline scoring and rule functions | `score_isolation_forest`, `add_stage3_broad_rules`, `add_stage3_improved_rules`, etc. |
| Replay execution | Run 4 replay pipelines | `baseline_replay`, 3 cascade variant replays |
| Metric computation | `compute_binary_metrics` on all 7 model specs | `replay_metrics_dataframe` |
| Validation comparison | Merge replay vs. training metrics; two-pass tolerance check | `replay_comparison_dataframe`, `final_validation_status` |
| Row-level score export | Wide test-row DataFrame with all prefixed flag/score columns | `replay_scores_dataframe` |
| Output saves | 4 CSV/JSON artifacts | Files in `VALIDATION_ROOT` tree for Gold_06B handoff |

## Section Details

## 2. Context Stage and Bootstrap

```python
CONTEXT_STAGE = "gold_model_replay_validation"

CTX = load_notebook_context(
    stage=CONTEXT_STAGE,
    dataset="pump",
    mode="test",
    profile="default",
    logger_child_name="capstone.gold.model_replay_validation",
    log_filename="gold_model_replay_validation.log",
)
```

`CONFIG_RUN_MODE = "test"` — Gold_06A runs in test mode, not train mode. This is the only Gold notebook confirmed to run in test mode rather than train mode.

`ledger = CTX.ledger` — provided by the bootstrap, not separately instantiated.

`DATASET_NAME = CONTEXT_DATASET` — set directly from the context constant rather than from the config dataset section.

`ARTIFACTS_ROOT = paths.artifacts`, `MODELS_ROOT = paths.models` — both taken from the context path object.

A ledger step `"context_loaded"` is added immediately.

There is no `wandb.init()` and no database connection. Gold_06A does not write to PostgreSQL and does not upload artifacts to W&B.

---

## 3. Output Directory Setup

All Gold_06A output paths are rooted under a validation subdirectory inside the Gold artifact tree:

```
GOLD_ROOT                = ARTIFACTS_ROOT / "gold" / DATASET_NAME
VALIDATION_ROOT          = GOLD_ROOT / "model_replay_validation"
VALIDATION_RESULTS_DIR   = VALIDATION_ROOT / "results"
VALIDATION_SCORES_DIR    = VALIDATION_ROOT / "scores"
VALIDATION_SUMMARY_DIR   = VALIDATION_ROOT / "summaries"
VALIDATION_PLOTS_DIR     = VALIDATION_ROOT / "plots"
```

All five directories are created with `mkdir(parents=True, exist_ok=True)`. `VALIDATION_PLOTS_DIR` is created but no plot files are saved in the current notebook.

---

## 4. Upstream Artifact Resolution

Gold_06A constructs explicit path dicts for each category of upstream artifact before loading anything:

**Input data paths (from Gold_01):**

- `FULL_SCALED_PATH` — resolved from `RESOLVED_PATHS["gold_preprocessed_scaled_data_path"]` or a default under `paths.data / "gold"`. This is the preferred source.
- `TEST_DATA_PATH` — resolved from `RESOLVED_PATHS["gold_test_data_path"]` or a default. Used as a portability fallback only.

**Feature JSON files (from Gold_01 preprocessing):**

- `STAGE1_FEATURES_PATH` — `{DATASET_NAME}__gold__stage1_features.json`
- `STAGE2_FEATURES_PATH` — `{DATASET_NAME}__gold__stage2_features.json`
- `STAGE3_PRIMARY_PATH` — `{DATASET_NAME}__gold__stage3_primary_rule_sensors.json`
- `STAGE3_SECONDARY_PATH` — `{DATASET_NAME}__gold__stage3_secondary_rule_sensors.json`

All four are under `GOLD_ROOT / "preprocessing" / "features"`.

**Model artifacts (from Gold_02 and Gold_03A/03B/03C):**

| Key | Path |
|---|---|
| `baseline` | `baseline/models/{DATASET_NAME}__gold__baseline_isolation_forest.joblib` |
| `cascade_default_stage1` | `cascade_defaults/models/{DATASET_NAME}__gold__cascade_defaults_stage1_isolation_forest.joblib` |
| `cascade_default_stage2` | `cascade_defaults/models/{DATASET_NAME}__gold__cascade_defaults_stage2_isolation_forest.joblib` |
| `cascade_tuned_stage1` | `cascade_tuned/models/{DATASET_NAME}__gold__cascade_tuned_stage1_isolation_forest.joblib` |
| `cascade_tuned_stage2` | `cascade_tuned/models/{DATASET_NAME}__gold__cascade_tuned_stage2_isolation_forest.joblib` |
| `stage3_improved_stage1` | `cascade_stage3_improved/models/{DATASET_NAME}__gold__cascade_stage3_improved_stage1_isolation_forest.joblib` |
| `stage3_improved_stage2` | `cascade_stage3_improved/models/{DATASET_NAME}__gold__cascade_stage3_improved_stage2_isolation_forest.joblib` |

**Threshold JSON files:** baseline, cascade_default, cascade_tuned, stage3_improved — one per model family.

**Summary JSON files:** baseline, cascade_default, cascade_tuned, stage3_improved — the same summary artifacts saved by the training notebooks. Gold_06A reads the `baseline_metrics` and `cascade_metrics` sections from these files to build the expected metric table.

**Reference profile CSV files:** cascade_default, cascade_tuned, stage3_improved — one per cascade family.

**Resolved config YAML files:** cascade_default, cascade_tuned, stage3_improved — the `resolved_config.yaml` snapshots saved by Gold_03A, Gold_03B, and Gold_03C. Gold_06A reads the `gold_cascade.stage3` config section from each.

**Artifact existence check:** An `artifact_inventory` DataFrame is assembled covering all model, threshold, summary, profile, and feature artifacts. If any artifact is missing (`path.exists() == False`), a `FileNotFoundError` is raised with the full missing-artifact table before any data is loaded. A ledger step `"required_artifacts_found"` is added on success.

---

## 5. Data Loading

```python
if FULL_SCALED_PATH.exists():
    replay_source_dataframe = pd.read_parquet(FULL_SCALED_PATH)
    replay_source_name = "full_scaled_with_test_mask"
    test_mask = ~replay_source_dataframe["meta__is_train_flag"].fillna(False).astype(bool)
else:
    replay_source_dataframe = pd.read_parquet(TEST_DATA_PATH)
    replay_source_name = "test_only_fallback"
    test_mask = pd.Series(True, index=replay_source_dataframe.index)
```

The full scaled dataframe is strongly preferred because Stage 3 confirmation rules use rolling windows and persistence logic that require training-row history to match the original notebook run. When only the test-only file is available, Stage 3 replay fidelity may differ.

If the full scaled path is used, `meta__is_train_flag` must be present or a `KeyError` is raised. Final metrics are always computed only on test rows via `test_mask`.

Feature lists are loaded with `require_string_list(load_json(path), name)`. Model artifacts are loaded with `joblib.load`. Thresholds, summaries, and config payloads are loaded with `load_json` / `yaml.safe_load`, each validated with `require_mapping`.

The label column is resolved by probing candidates `["anomaly_flag", "is_anomaly", "target_flag", "label"]`; a `KeyError` is raised if none are found.

A ledger step `"replay_source_loaded"` records the replay source name, row counts, label column, anomaly count, and feature list sizes.

---

## 6. Replay Helper Functions

All replay helpers are defined inline. None are imported from utilities.

**`score_isolation_forest`** — applies an already-fitted `IsolationForest` to a DataFrame using selected feature columns. Returns a dict of `score` (`-score_samples`, higher = more anomalous), `decision` (`decision_function`), and `prediction` (`predict`). Calls `ensure_columns` to validate required features are present before scoring.

**`compute_binary_metrics`** — filters to test rows via `test_mask`, computes precision/recall/F1/confusion matrix counts. Optionally appends `roc_auc` and `pr_auc` when a score column is available and both label classes are present.

**`compute_profile_breach_count`** — per-row count of sensor features that fall outside the reference profile `lower_bound`/`upper_bound` bounds. Uses `reference_profile.set_index("feature_name")` lookup.

**`compute_persistence_flag`** — rolling-window Stage 2 flag persistence: marks rows where the rolling sum of `stage2_flag` over `rolling_window_size` rows meets `minimum_flags_in_window`.

**`compute_drift_flag`** — per-row rolling-median drift: marks rows where any watched feature's absolute deviation from its rolling median exceeds `feature_standard_deviation * drift_threshold_multiplier`.

**`add_stage3_broad_rules`** — implements the Stage 3 confirmation logic from Gold_03A and Gold_03B. Reads `min_primary_sensor_hits`, `min_secondary_sensor_hits`, `rolling_window_size`, and `minimum_flags_in_window` from the variant config. Produces `stage3_profile_breach_flag`, `stage3_corroboration_flag`, `stage3_persistence_flag`, `stage3_drift_flag`, `stage3_rule_evidence_count`, and `cascade_final_flag = (stage1_flag == 1) & (stage2_flag == 1) & (profile_breach >= min_primary OR rule_evidence_count >= 2)`.

**`add_stage3_improved_rules`** — implements the Stage 3 logic from Gold_03C. Reads `stage3_selected_params` from the threshold artifact and `stage3` config from the resolved config YAML. Produces the weighted evidence score (`profile_breach * weight + corroboration * weight + persistence * weight + drift * weight`), `stage3_confirmed_flag`, and operating-mode flag columns: `cascade_stage3_improved_flag` (= `cascade_final_flag`), `cascade_stage3_relaxed_flag` (weighted_score >= 2.0), `cascade_stage3_medium_flag` (>= 3.0), `cascade_stage3_strict_flag` (>= 5.0).

---

## 7. Replay Execution

**`run_baseline_replay`** — scores the full replay source DataFrame with the baseline Isolation Forest using Stage 1 feature columns, applies `baseline_threshold`, and produces `baseline_score`, `baseline_decision`, `baseline_pred`, `baseline_threshold`, `baseline_flag`.

**`run_cascade_replay`** — handles all three cascade variants. Applies Stage 1 and Stage 2 Isolation Forest scoring with separate feature column sets, applies `stage1_threshold` and `stage2_threshold`, computes `stage1_flag`, `stage2_raw_flag`, `stage2_flag` (gated: `stage1_flag == 1 AND stage2_raw_flag == 1`), then routes to `add_stage3_broad_rules` (`use_improved_stage3=False`) or `add_stage3_improved_rules` (`use_improved_stage3=True`).

Four replay DataFrames are produced:

```python
baseline_replay        = run_baseline_replay(replay_source_dataframe)
cascade_default_replay = run_cascade_replay(..., variant_key="cascade_default", use_improved_stage3=False)
cascade_tuned_replay   = run_cascade_replay(..., variant_key="cascade_tuned",   use_improved_stage3=False)
stage3_improved_replay = run_cascade_replay(..., variant_key="stage3_improved", use_improved_stage3=True)
```

A ledger step `"model_variants_replayed"` records row counts for all four replay DataFrames plus the test row count.

---

## 8. Metric Computation and Validation Comparison

`replay_metric_specs` defines seven model rows for metric computation:

| `model_id` | Source DataFrame | `flag_column` | `score_column` |
|---|---|---|---|
| `baseline` | `baseline_replay` | `baseline_flag` | `baseline_score` |
| `cascade_default` | `cascade_default_replay` | `cascade_final_flag` | `stage2_score` |
| `cascade_tuned` | `cascade_tuned_replay` | `cascade_final_flag` | `stage2_score` |
| `stage3_improved` | `stage3_improved_replay` | `cascade_final_flag` | `stage3_weighted_score` |
| `stage3_relaxed` | `stage3_improved_replay` | `cascade_stage3_relaxed_flag` | `stage3_weighted_score` |
| `stage3_medium` | `stage3_improved_replay` | `cascade_stage3_medium_flag` | `stage3_weighted_score` |
| `stage3_strict` | `stage3_improved_replay` | `cascade_stage3_strict_flag` | `stage3_weighted_score` |

`compute_binary_metrics` is applied to each row spec, producing `replay_metrics_dataframe`.

`build_expected_metrics_from_training_artifacts` reads the saved Gold 02 / 03A / 03B / 03C summary JSON files to build `expected_metrics_dataframe` with `expected_alert_count_test_rows`, `expected_precision`, `expected_recall`, and `expected_f1` for all seven model rows. Stage 3 operating mode expected values are extracted from the `cascade_metrics` section of the `stage3_improved` summary.

The two DataFrames are merged on `model_id` into `replay_comparison_dataframe`. Delta columns are computed for all four metrics.

**Validation status logic (two-pass):**

**Pass 1 — Exact match** (`METRIC_TOLERANCE = 1e-9`):
- All four metric deltas within tolerance → `validation_status = "pass"`
- Otherwise → `validation_status = "review_delta"`

**Pass 2 — Secondary tolerance** (`ALERT_COUNT_TOLERANCE = 1`, `METRIC_TOLERANCE_RELAXED = 0.0001`):
- Applied only when `validation_status != "pass"`
- Alert count delta ≤ 1 AND precision/recall/F1 deltas ≤ 0.0001 → `tolerance_validation_status = "pass_with_tolerance"`
- Otherwise → `tolerance_validation_status = "review_delta"`

**Final reportable status** (`final_validation_status`):
- `"exact_pass"` — strict exact match
- `"pass_with_tolerance"` — within secondary tolerance
- `"review_delta"` — exceeds both thresholds; requires review

---

## 9. Row-Level Score Export

A wide test-row DataFrame (`replay_scores_dataframe`) is assembled from the held-out test rows only:

- Identity and label columns present in `replay_source_dataframe`: `meta__row_id`, `meta__asset_id`, `meta__dataset`, `meta__run_id`, `meta__split`, `meta__is_train_flag`, `event_time`, `event_step`, `time_index`, `timestamp`, `machine_status`, `anomaly_flag` (or whichever label column was resolved)
- `plot_order_index` = integer position within test rows
- Per-model prefixed score columns:
  - `baseline__baseline_score`, `baseline__baseline_flag`
  - `cascade_default__stage1_score`, `cascade_default__stage1_flag`, `cascade_default__stage2_score`, `cascade_default__stage2_raw_flag`, `cascade_default__stage2_flag`, `cascade_default__cascade_final_flag`
  - `cascade_tuned__*` — same set as cascade_default
  - `stage3_improved__stage1_score`, `stage3_improved__stage1_flag`, `stage3_improved__stage2_score`, `stage3_improved__stage2_raw_flag`, `stage3_improved__stage2_flag`, `stage3_improved__stage3_weighted_score`, `stage3_improved__cascade_final_flag`, `stage3_improved__cascade_stage3_relaxed_flag`, `stage3_improved__cascade_stage3_medium_flag`, `stage3_improved__cascade_stage3_strict_flag`

Columns are included only when the source DataFrame contains them (`if column in source_df.columns`). The index is reset before joining so position-aligned merging is safe.

---

## Outputs and Artifacts

```python
replay_metrics_dataframe.to_csv(metrics_output_path, index=False)
replay_comparison_dataframe.to_csv(comparison_output_path, index=False)
replay_scores_dataframe.to_csv(scores_output_path, index=False)
save_json(summary_payload, summary_output_path)
```

| Artifact | Directory | Filename |
|---|---|---|
| Replay metrics (7 model rows) | `VALIDATION_RESULTS_DIR` | `{DATASET_NAME}__gold06a__test_replay_metrics.csv` |
| Replay vs training comparison | `VALIDATION_RESULTS_DIR` | `{DATASET_NAME}__gold06a__test_replay_vs_training_artifacts.csv` |
| Row-level test replay scores | `VALIDATION_SCORES_DIR` | `{DATASET_NAME}__gold06a__test_replay_scores.csv` |
| Summary JSON | `VALIDATION_SUMMARY_DIR` | `{DATASET_NAME}__gold06a__test_replay_summary.json` |

`summary_payload` includes: `stage`, `dataset`, `recipe_id`, `replay_source_name`, `replay_source_rows`, `test_rows`, `label_column`, `model_count`, `exact_pass_count`, `tolerance_pass_count`, `pass_count`, `review_count`, `alert_count_tolerance`, `metric_tolerance_relaxed`, and all four output paths.

A ledger step `"gold06a_outputs_saved"` records the full summary payload.

No W&B uploads. No SQL writes. No truth record is constructed or saved.

---

## 11. Interpretation Object

The final code cell produces an `interpretation` dict displayed inline:

```python
interpretation = {
    "purpose": "Gold 06A validates that saved Gold 02/03 model artifacts and rule settings can be replayed against held-out test rows without retraining.",
    "comparison_scope": "Replay metrics are compared against the saved training-notebook summary artifacts, not against Gold 04 as the primary benchmark.",
    "exact_pass_count": ...,
    "tolerance_pass_count": ...,
    "pass_count": ...,
    "review_count": ...,
    "next_notebook": "Gold 06B uses the saved Gold 06A replay score output for test-set early-warning validation.",
}
```

This object is display-only and is not saved to disk.

---

## Inputs

| Source | Artifact type | Load method |
|---|---|---|
| Gold_01 scaled Parquet (preferred) | Full-split DataFrame with `meta__is_train_flag` | `pd.read_parquet` |
| Gold_01 test Parquet (fallback) | Test-only DataFrame | `pd.read_parquet` |
| Stage 1 / 2 feature JSON lists | Feature column lists | `load_json` + `require_string_list` |
| Stage 3 primary / secondary sensor JSON lists | Sensor feature lists | `load_json` + `require_string_list` |
| Baseline Isolation Forest joblib | Fitted model | `joblib.load` |
| Cascade (defaults/tuned/stage3_improved) Stage 1 + Stage 2 joblib × 6 | Fitted models | `joblib.load` |
| Baseline / cascade threshold JSON × 4 | Decision thresholds | `load_json` + `require_mapping` |
| Baseline / cascade summary JSON × 4 | Saved training-run metrics | `load_json` + `require_mapping` |
| Cascade reference profile CSV × 3 | Sensor bound tables | `pd.read_csv` |
| Cascade resolved config YAML × 3 | Stage 3 config params | `yaml.safe_load` + `require_mapping` |

Gold_06A does not load Gold_04 comparison outputs or Gold_05 anomaly timeline outputs.

---

## Data Quality / Validation Behavior

| Check | Purpose | Failure / Risk Prevented |
|---|---|---|
| `artifact_inventory` existence check | All model, threshold, summary, profile, and feature artifacts must exist before any data is loaded | `FileNotFoundError` with full missing-artifact table; prevents partial replay |
| `require_string_list` on all feature/sensor JSON lists | Non-empty string list validation | `TypeError`/`ValueError` before any model is scored |
| `require_mapping` on all JSON/YAML dicts | Non-empty dict validation | Catches empty or malformed artifact files before scoring begins |
| Label column resolution from candidate list | Probes `["anomaly_flag", "is_anomaly", "target_flag", "label"]` | `KeyError` if none found; prevents scoring without a ground-truth label |
| `meta__is_train_flag` presence in full-scaled path | Required for test mask derivation | `KeyError` if column absent; prevents computing metrics on wrong rows |
| Two-pass validation status | Metric delta checked at `1e-9` (strict) and `0.0001` (secondary) | Flags any replay metric divergence from training-notebook summary artifacts as `review_delta` |

---

## Downstream Handoff

Gold_06A's interpretation cell explicitly states: *"Gold 06B uses the saved Gold 06A replay score output for test-set early-warning validation."*

The primary handoff artifact is `{DATASET_NAME}__gold06a__test_replay_scores.csv` in `VALIDATION_SCORES_DIR`. This file contains test-row identifiers, labels, `plot_order_index`, and prefixed replay flag and score columns for all four model families and their Stage 3 operating mode variants. Gold_06B uses this file to evaluate early-warning detection behavior on the replayed test output.

---

## Key Function Calls and In-Place Usage

| Function | Source | Purpose |
|---|---|---|
| `load_notebook_context` | `utils.core.notebook_context` | Bootstrap CTX, paths, config, logger, ledger |
| `load_json` | `utils.core.file_io` | Load threshold, summary JSON files |
| `save_json` | `utils.core.file_io` | Save summary JSON output |
| `require_string_list` | inline | Validate feature JSON list values |
| `require_mapping` | inline | Validate JSON/YAML dict values |
| `ensure_columns` | inline | Pre-scoring column presence check |
| `score_isolation_forest` | inline | Apply fitted IF with `-score_samples` |
| `compute_binary_metrics` | inline | Test-row precision/recall/F1/confusion metrics |
| `compute_profile_breach_count` | inline | Sensor-vs-reference-profile breach counting |
| `compute_persistence_flag` | inline | Rolling Stage 2 flag persistence |
| `compute_drift_flag` | inline | Rolling-median sensor drift |
| `add_stage3_broad_rules` | inline | Stage 3 logic for Gold_03A/03B |
| `add_stage3_improved_rules` | inline | Stage 3 weighted logic for Gold_03C |
| `run_baseline_replay` | inline | Full baseline replay pipeline |
| `run_cascade_replay` | inline | Full cascade replay pipeline (all variants) |
| `build_expected_metrics_from_training_artifacts` | inline | Read saved summary metrics for comparison |
| `joblib.load` | joblib | Load fitted model artifacts |

---

## Logical Workflow Map

1. Imports
2. Inline helper definitions (`require_string_list`, `require_mapping`)
3. Context bootstrap (`load_notebook_context`)
4. Config variable extraction from CTX
5. Output directory setup (`VALIDATION_ROOT` tree)
6. Upstream artifact path construction (models, thresholds, summaries, profiles, configs)
7. Artifact existence check → `artifact_inventory` DataFrame; `FileNotFoundError` if any missing
8. Ledger step: `required_artifacts_found`
9. Replay source DataFrame load (full scaled preferred, test-only fallback)
10. Feature list and label column loading and validation
11. Model, threshold, summary, profile, and config artifact loading
12. Ledger step: `replay_source_loaded`
13. Replay helper function definitions
14. Replay function definitions (`run_baseline_replay`, `run_cascade_replay`)
15. Four replay executions (`baseline_replay`, `cascade_default_replay`, `cascade_tuned_replay`, `stage3_improved_replay`)
16. Ledger step: `model_variants_replayed`
17. `replay_metric_specs` definition
18. `compute_binary_metrics` applied to all 7 model specs → `replay_metrics_dataframe`
19. `build_expected_metrics_from_training_artifacts` → `expected_metrics_dataframe`
20. Merge and delta computation → `replay_comparison_dataframe`
21. Two-pass validation status computation (`validation_status`, `tolerance_validation_status`, `final_validation_status`)
22. Wide test-row score DataFrame construction → `replay_scores_dataframe`
23. CSV saves (metrics, comparison, scores) + JSON save (summary)
24. Ledger step: `gold06a_outputs_saved`
25. `interpretation` dict construction and display

---

## Relationship to Other Notebooks

### Upstream Context

Gold_06A loads:
- Gold_01 scaled Parquet (`FULL_SCALED_PATH`, preferring the full split with `meta__is_train_flag`) or `TEST_DATA_PATH` fallback
- Gold_02: baseline model, thresholds, summary, validation contract
- Gold_03a: Stage 1/2 models, thresholds, reference profile, summary, config, validation contract
- Gold_03b: same set as Gold_03a
- Gold_03c: same set + 4 operating-mode validation contracts (default, relaxed, medium, strict)

Gold_06A does not consume Gold_04 comparison outputs or Gold_05 anomaly detection outputs.

### Downstream Handoff

Gold_06A provides to Gold_06B:
- Test replay scores CSV (`{DATASET_NAME}__gold06a__test_replay_scores.csv`) — required by Gold_06B with no fallback
- Also produces: replay metrics CSV, validation comparison CSV, final validation status JSON (for audit/reporting only; not consumed by Gold_06B directly)

### Pipeline Position

Hold-out validation notebook. The only Gold notebook confirmed to run in `CONFIG_RUN_MODE="test"`. Proves model reproducibility on unseen data. Sits between the modeling phase (Gold_02 through Gold_03c) and the final early-warning validation (Gold_06B). Does not write to PostgreSQL, open a W&B run, or produce a truth record.

### Relationship Summary

- Loads fitted models, thresholds, profiles, and contracts from Gold_02, Gold_03a, Gold_03b, Gold_03c
- Does not consume Gold_04 or Gold_05 outputs
- Produces the test replay scores CSV that Gold_06B requires (`FileNotFoundError` if absent)
- Does not write to SQL, W&B, or truth records — intentionally lightweight validation pass
- `final_validation_status` field (`pass`, `pass_with_tolerance`, `review_delta`) is the key audit deliverable
