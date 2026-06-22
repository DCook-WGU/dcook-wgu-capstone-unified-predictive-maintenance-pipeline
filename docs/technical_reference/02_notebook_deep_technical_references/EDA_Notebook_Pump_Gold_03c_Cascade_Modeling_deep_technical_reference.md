# Gold 03c Deep Technical Reference

## Purpose of This Deep Reference

This document covers Gold 03c technical decisions requiring deeper explanation than the workflow reference. The 071b workflow reference (`EDA_Notebook_Pump_Gold_03c_Cascade_Modeling_code_reference.md`) describes what each section does; this document explains why the important methods, configurations, and output designs are structured the way they are — with particular focus on the Stage 2 threshold reuse mechanism, the Stage 3 weighted grid search, the three operating-mode variant system, and the four-contract output design that distinguishes Gold 03c from all prior cascade notebooks.

---

## Technical Scope

Decision tags applicable to this notebook:

| Tag | Reason Applicable |
|---|---|
| `MODEL_TRAINING` | `IsolationForest.fit()` for Stage 1; Stage 2 fitted from Gold_03b best params; no new Stage 2 search |
| `MODEL_EVALUATION` | per-candidate Stage 3 grid metrics (precision/recall/F1); per-mode metrics including ROC AUC and PR AUC |
| `DATA_VALIDATION` | `validate_cascade_output` with final gate checks; variant-column gate validation; 4-contract rule_config checks |
| `TRUTH_METADATA` | `gold_cascade` truth record with `stage3_variant = "tuned_confirmation_layer"`; `GOLD_PARENT_TRUTH_HASH` from scaled Parquet column |
| `ARTIFACT_WRITE` | results CSV/pickle, Stage 1 and Stage 2 joblib models, reference profile CSV, multiple JSON artifacts, 4 validation contracts |
| `SQL_WRITE` | `write_gold_cascade_scores_sql` → `gold.anomaly_detection_scores` with `model_stage="cascade_stage3_improved_final"` |
| `WANDB_LOGGING` | `wandb.init`, `wandb.save` for all artifacts, `wandb_run.finish()` |
| `LEDGER_UPDATE` | `ledger.add` at each key step including `comparison_ready: True`; `ledger.write_json` at close |

Tags not applicable: `TEMPORAL_SMOOTHING`, `CORRELATION_REPAIR`, `VARIANCE_CONTROL`, `MEAN_ANCHORING`, `BOUNDS_CLIPPING`, `FAULT_INJECTION`, `PHASE_SPECIFIC_LOGIC`, `MISSINGNESS_REPLAY`.

---

## Source Grounding

| Source | Role |
|---|---|
| `notebooks/experiments/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling.ipynb` (178 cells) | Primary source of truth for all technical claims |
| `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling_code_reference.md` | 071b workflow reference (read-only context) |
| `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling_deep_technical_reference.md` | Gold 03a sequence context (read-only) |
| `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling_deep_technical_reference.md` | Gold 03b sequence context (read-only) |
| `notebook_inventory.json` | Active notebook path resolution |

The active Gold 03c notebook is the source of truth. Claims not confirmable from source are labeled `Not determined from available source`.

---

## Stage Role in the Cascade

Gold 03c implements the `"stage3_improved"` cascade variant (`CASCADE_VARIANT = "stage3_improved"`, hardcoded). Its `stage3_variant` label is `"tuned_confirmation_layer"`. It is the final cascade modeling notebook and the only one in the project with a confirmed direct file-level dependency on another Gold modeling notebook (Gold_03b).

Gold 03c's primary technical distinction from Gold 03b is Stage 3. Rather than evaluating a fixed rule-based evidence count, Gold 03c introduces a configurable weighted evidence score with grid-searched parameters and three calibrated operating modes (relaxed, medium, strict). This produces four separate output configurations and four corresponding validation contracts — one default and one per severity level — allowing Gold_06A to replay and compare all four configurations in the held-out window.

Gold 03c does not introduce a new Stage 2 selection search. It always reuses Gold_03b's selected Stage 2 model and threshold configuration. This design choice keeps Stage 2 constant across Gold 03b and Gold 03c so that any performance difference between the two notebooks is attributable entirely to Stage 3, not Stage 2 configuration.

The cascade variant comparison in the project:

| Notebook | `CASCADE_VARIANT` | Stage 2 Source | Stage 3 Method | Validation Contracts |
|---|---|---|---|---|
| Gold_03a | `"default"` | Fixed (hardcoded) | Evidence count ≥ 2 OR primary breach | 1 |
| Gold_03b | `"tuned"` | Config-driven grid search | Evidence count ≥ 2 OR primary breach | 1 |
| Gold_03c | `"stage3_improved"` | Always `"previous_best"` (Gold_03b) | Weighted score ≥ tuned threshold OR strong primary | 4 |

---

## Input Contract and Lineage

### Confirmed Input Files

| Input | Source / Load Method | Guard |
|---|---|---|
| Gold_01 scaled Parquet | `load_data(GOLD_PREPROCESSED_SCALED_DATA_PATH)` | Parent truth resolved from `meta__truth_hash` column |
| Gold_01 truth record | `load_parent_truth_record_from_dataframe(dataframe, truth_dir, ...)` | Raises on empty / malformed dict |
| Fit / test / train / preprocessed Parquets | `load_data(path)` per truth-overridden path | 8 path overrides from Gold truth `artifact_paths` |
| Stage 1 / Stage 2 feature JSONs | `require_str_list(load_json(path), name)` | `TypeError`/`ValueError` on malformed lists |
| Stage 3 primary / secondary sensor JSONs | `require_str_list(load_json(path), name)` | Same guard |
| Gold_03b thresholds JSON | `require_mapping(load_json(CASCADE_TUNED_THRESHOLDS_PATH), ...)` | `KeyError` if `stage2_best_params` or `stage2_selected_threshold_percentile` absent |
| Gold_03b Stage 2 model | `joblib.load(CASCADE_TUNED_STAGE2_MODEL_PATH)` | Applied without retraining |

### Parent Truth Resolution

Gold 03c resolves the parent truth hash differently from Gold 03b. Rather than loading a separate Gold truth JSON file, it reads the parent truth record from the `meta__truth_hash` column already stamped into the scaled Parquet:

```python
gold_truth = load_parent_truth_record_from_dataframe(
    dataframe=gold_preprocessed_scaled_dataframe,
    truth_dir=TRUTHS_PATH,
    parent_layer_name="gold",
    dataset_name=GOLD_DATASET_NAME,
    column_name="meta__truth_hash",
)

DATASET_NAME = get_dataset_name_from_truth(gold_truth)
GOLD_PARENT_TRUTH_HASH = get_truth_hash(gold_truth)
```

A secondary lineage variable `STAGE3_IMPROVED_PARENT_GOLD_TRUTH_HASH` is resolved by `resolve_single_parent_gold_truth_hash` — an inline-defined function that reads `meta__parent_truth_hash` from the scaled Parquet and validates that exactly one unique non-null value is present.

### Eight Path Overrides

Eight artifact paths are overridden from `gold_truth["artifact_paths"]` using the same fallback pattern as Gold_03a and Gold_03b — each path falls back to the config-resolved default if the truth record key is absent. The overridden paths are: `GOLD_PREPROCESSED_DATA_PATH`, `GOLD_FIT_DATA_PATH`, `GOLD_TEST_DATA_PATH`, `GOLD_TRAIN_DATA_PATH`, `STAGE1_FEATURES_PATH`, `STAGE2_FEATURES_PATH`, `STAGE3_PRIMARY_PATH`, `STAGE3_SECONDARY_PATH`.

Two additional Gold_03b-specific paths are resolved from `RESOLVED_PATHS`:
- `CASCADE_TUNED_THRESHOLDS_PATH` — Gold_03b's saved Stage 2 thresholds JSON
- `CASCADE_TUNED_SUMMARY_PATH` — Gold_03b's saved summary JSON

### Train/Test Split Recovery

The train/test split is restored from `meta__is_train_flag`:

```python
train_mask = gold_preprocessed_scaled_dataframe["meta__is_train_flag"].fillna(False).astype(bool)
test_mask = (~train_mask).astype(bool)
```

A hard `ValueError` fires if `meta__is_train_flag` is absent. `anomaly_flag` is extracted for test rows when present; when absent, `test_labels = None` and metrics fall back to alert-count minimization.

### Why Lineage Matters for Gold 03c

Gold_04 performs a four-source cross-validation of parent truth hashes, requiring `GOLD_PARENT_TRUTH_HASH` to agree across Gold_02, Gold_03a, Gold_03b, and Gold_03c. Any run of Gold_03c against a different Gold_01 preprocessing output would fail this check. Gold_03c also embeds `parent_gold_truth_hash` in all four validation contracts' `lineage_payload`, so Gold_06A can verify provenance before replaying the cascade.

---

## Model Input Preparation

### Reference Profile Construction

The reference profile is built from `gold_fit_dataframe` (the normal-only fit subset) using `build_reference_profile`, an inline-defined function:

```python
reference_profile_features = list(dict.fromkeys(
    stage1_feature_columns + stage3_primary_rule_sensors + stage3_secondary_rule_sensors
))

reference_profile = build_reference_profile(
    gold_fit_dataframe,
    feature_columns=reference_profile_features,
)
```

Columns produced: `feature_name`, `median_value`, `mean_value`, `standard_deviation`, `lower_bound` (5th percentile), `upper_bound` (95th percentile). Building from `gold_fit_dataframe` prevents anomalous test-window values from shifting the normal operating bounds used in Stage 3 breach detection. The workflow reference also lists `CASCADE_TUNED_REFERENCE_PROFILE_PATH` (Gold_03b's saved reference profile) as an input; the exact relationship between this loaded profile and the rebuilt profile is not determined from the notebook source examined.

### Feature Matrix Assembly

Four typed DataFrames:

| Matrix | Source DataFrame | Columns |
|---|---|---|
| `stage1_train_fit_features` | `gold_fit_dataframe` | `stage1_feature_columns` |
| `stage2_train_fit_features` | `gold_fit_dataframe` | `stage2_feature_columns` |
| `stage1_all_features` | `gold_preprocessed_scaled_dataframe` | `stage1_feature_columns` |
| `stage2_all_features` | `gold_preprocessed_scaled_dataframe` | `stage2_feature_columns` |

All four are validated for missing feature columns before any model training begins. Missing column lists are reported in `ValueError` messages.

### `plot_order_index` Column

After Stage 1 scoring, Gold 03c creates `plot_order_index` — derived from `time_index` if available, `meta__row_id` as integer if parseable, or row order as fallback. This column supports timeline plots and debug displays. It is unique to Gold 03c among the three cascade notebooks.

---

## Cascade Modeling Methodology

### Stage 1 — Broad Isolation Forest

Stage 1 is structurally identical to Gold_03a and Gold_03b:

```python
stage1_model = IsolationForest(
    n_estimators=STAGE1_ESTIMATOR_COUNT,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
stage1_model.fit(stage1_train_fit_features)
```

All rows are scored via `compute_anomaly_scores_isolation_forest` (returns `-score_samples()`, higher = more anomalous). The threshold is calibrated on training scores only via `choose_threshold_value(stage1_train_scores, STAGE1_THRESHOLD_PERCENTILE)`, preventing test anomaly distributions from shifting the cut-point.

Stage 1 results are attached to `cascade_results` via `build_stage_scoring_frame` → `score_isolation_forest_stage` → `merge_stage_results_back` on `meta__row_id`. This also uses a `compute_anomaly_scores_isolation_forest` call to produce `stage1_all_scores` and `stage1_threshold` columns separately, supporting alignment tracing.

### Why Gold 03c Is a Separate Notebook

Gold 03c exists to isolate the Stage 3 improvement from Stage 2 tuning. If Stage 3 improvements were combined with Stage 2 tuning in a single notebook, it would be impossible to attribute metric differences to Stage 2 vs. Stage 3 changes. By fixing Stage 2 to Gold_03b's selected configuration and varying only Stage 3, Gold_03c provides a controlled comparison. Gold_04 can then compare Gold_03b and Gold_03c directly, with Stage 2 held constant.

---

## Stage 2 Threshold Reuse and Candidate Refinement

### Selection Source

`STAGE2_SELECTION_SOURCE` is hardcoded to `"previous_best"` in the notebook source. Three modes are defined in comments:
- `"previous_best"` — reuse Gold_03b's saved best params + threshold percentile (active)
- `"configured_search"` — run Gold_03c's own configured `STAGE2_SELECTION_MODE`
- `"auto"` — use `"previous_best"` if Gold_03b params are available and non-empty, else fall back to `"configured_search"`

The hardcoded `"previous_best"` selection is intentional: Gold_03c must reuse the same Stage 2 configuration that Gold_03b selected in order to hold Stage 2 constant for the Stage 3 comparison.

### Gold_03b Thresholds Load

Gold_03b's thresholds JSON is loaded with a `require_mapping` guard:

```python
previous_03b_stage2_thresholds = require_mapping(
    load_json(CASCADE_TUNED_THRESHOLDS_PATH),
    "previous_03b_stage2_thresholds",
)

previous_03b_stage2_selected_threshold_percentile = float(
    previous_03b_stage2_thresholds.get(
        "stage2_selected_threshold_percentile",
        STAGE2_FIXED_THRESHOLD_PERCENTILE,
    )
)

previous_03b_stage2_best_params = {
    str(key): value
    for key, value in previous_03b_stage2_thresholds.get("stage2_best_params", {}).items()
}
```

A `TypeError` fires if `stage2_best_params` is not a dict. `previous_03b_stage2_available = bool(previous_03b_stage2_best_params)` — a boolean that signals whether the Gold_03b params are usable; if `False`, `"previous_best"` mode would behave differently, though in practice this reflects a Gold_03b execution state issue.

### `run_stage2_selection_decision`

`run_stage2_selection_decision` is defined inline. It routes:
- `"previous_best"` path → creates a fixed-mode Stage 2 from Gold_03b params, bypassing any new grid search
- `"configured_search"` path → runs `evaluate_stage2_model_with_thresholds` across the configured grid
- `"auto"` path → selects based on `previous_best_available` flag

Return tuple: `(stage2_model, best_stage2_result, stage2_search_results)`.

`best_stage2_result` records: `selection_source`, `selection_mode`, `reused_previous_03b_best` (boolean), `selected_threshold_percentile`, `threshold`, `best_params`, and the full metric payload. `stage2_selection_source_used` and `stage2_selection_mode_used` are extracted for ledger logging.

### Stage 2 Gate

Stage 2 applies to only those rows that passed Stage 1 (`stage2_candidate_mask = stage1_flag == 1`). The helper column rename pattern prevents the `score_isolation_forest_stage` defaults from overwriting the threshold-calibrated `stage2_flag` logic:

```python
stage2_results_df = stage2_results_df.rename(columns={
    "stage2_score":    "stage2_model_score",
    "stage2_decision": "stage2_model_decision",
    "stage2_pred":     "stage2_model_pred",
    "stage2_flag":     "stage2_model_flag",
})
```

`stage2_score = NaN` for all rows; then candidate rows are overwritten:

```python
cascade_results["stage2_score"] = np.nan
cascade_results.loc[stage2_candidate_mask, "stage2_score"] = stage2_all_scores[stage2_candidate_mask_array]
```

The in-code comment states: "NaN on non-candidate rows signals that Stage 2 did not evaluate them, which distinguishes absent scores from zero scores." Stage 2 gate enforces: `stage2_flag == 1` only where both `stage1_flag == 1` AND `stage2_raw_flag == 1`.

---

## Stage 3 Improved Rule or Confirmation Logic

Stage 3 in Gold 03c replaces the fixed evidence-count threshold of Gold_03a and Gold_03b with a weighted, grid-searched confirmation layer. This is the primary technical improvement of the notebook.

### Evidence Signal Pre-Computation

Before the tuning grid runs, four evidence signals are pre-computed on `cascade_results`:

- `stage3_profile_breach_count` — via `compute_primary_breach_count`: counts primary sensors outside reference profile upper bounds
- `stage3_secondary_breach_count` — via `compute_secondary_breach_count`: counts secondary sensor violations
- `stage3_persistence_flag` — via `compute_persistence_flag(cascade_results["stage2_flag"], rolling_window_size=STAGE3_ROLLING_WINDOW_SIZE, minimum_flags_in_window=STAGE3_MINIMUM_FLAGS_IN_WINDOW)` at notebook-level defaults

The tuning grid then re-computes persistence and drift per candidate with varying parameters, independently of these pre-computed values.

### Stage 3 Weighted Evidence Score

`build_stage3_candidate_output` computes, for each candidate from `product(STAGE3_TUNING_GRID values)`:

1. `profile_breach_flag` = `stage3_profile_breach_count >= STAGE3_MIN_PRIMARY_SENSOR_HITS`
2. `strong_primary_flag` = `stage3_profile_breach_count >= strong_primary_hits` (candidate-specific threshold)
3. `corroboration_flag` = `stage3_secondary_breach_count >= STAGE3_MIN_SECONDARY_SENSOR_HITS`
4. `persistence_flag` = `compute_persistence_flag(stage2_flag, candidate rolling_window_size, candidate minimum_flags_in_window)`
5. `drift_flag` = `compute_drift_flag(dataframe, stage3_watch_features, STAGE3_DRIFT_ROLLING_WINDOW_SIZE, candidate drift_threshold_multiplier)`
6. Weighted evidence score:

```python
weighted_evidence_score = (
    profile_breach_flag * STAGE3_PROFILE_BREACH_WEIGHT
    + corroboration_flag * STAGE3_CORROBORATION_WEIGHT
    + persistence_flag * STAGE3_PERSISTENCE_WEIGHT
    + drift_flag * STAGE3_DRIFT_WEIGHT
)
```

7. Final confirmation decision:

```python
confirmed_flag = (
    (strong_primary_flag == 1)
    | ((profile_breach_flag == 1) & (weighted_evidence_score >= min_weighted_score))
).astype(int)

final_flag = (
    (stage2_flag == 1) & (confirmed_flag == 1)
).astype(int)
```

The strong primary breach shortcut bypasses the weighted score requirement: a single overwhelming primary breach is sufficient confirmation regardless of other evidence. This prevents the weighted scoring from inadvertently suppressing high-confidence alerts when secondary and temporal evidence are absent.

### Stage 3 Grid Search Candidate Selection

`evaluate_stage3_candidate` scores each candidate against test labels using the same formula as Stage 2:

```python
if recall >= STAGE3_MIN_SELECTION_RECALL:
    selection_score = 3.0 * f1 + 1.0 * precision - 1.0 * alert_rate
else:
    selection_score = -1000.0 + recall   # penalized — fails recall floor
```

All candidates are sorted by `selection_score` descending into `stage3_search_results`. The highest-scoring candidate's output columns are promoted to `cascade_results`:

```python
cascade_results["stage3_profile_breach_flag"]        = stage3_best_output["profile_breach_flag"]
cascade_results["stage3_strong_primary_flag"]        = stage3_best_output["strong_primary_flag"]
cascade_results["stage3_corroboration_flag"]         = stage3_best_output["corroboration_flag"]
cascade_results["stage3_persistence_flag"]           = stage3_best_output["persistence_flag"]
cascade_results["stage3_drift_flag"]                 = stage3_best_output["drift_flag"]
cascade_results["stage3_rule_evidence_count"]        = stage3_best_output["rule_evidence_count"]
cascade_results["stage3_weighted_evidence_score"]    = stage3_best_output["weighted_evidence_score"]
cascade_results["stage3_weighted_score"]             = cascade_results["stage3_weighted_evidence_score"]
cascade_results["stage3_confirmed_flag"]             = stage3_best_output["confirmed_flag"]
cascade_results["cascade_stage3_improved_flag"]      = stage3_best_output["final_flag"]
cascade_results["cascade_final_flag"]                = cascade_results["cascade_stage3_improved_flag"]
```

`cascade_final_flag` is derived from `cascade_stage3_improved_flag`, making Stage 2 + Stage 3 weighted confirmation the definition of the final alert decision.

### Three Operating Mode Variants

Three additional flag columns provide calibrated severity levels:

```python
cascade_results["cascade_stage3_relaxed_flag"] = ((stage2_flag == 1) & (weighted >= 2.0)).astype(int)
cascade_results["cascade_stage3_medium_flag"]  = ((stage2_flag == 1) & (weighted >= 3.0)).astype(int)
cascade_results["cascade_stage3_strict_flag"]  = ((stage2_flag == 1) & (weighted >= 5.0)).astype(int)
```

These thresholds (2, 3, 5) are defined as named constants in the notebook:
- `RELAXED_STAGE3_WEIGHTED_EVIDENCE_SCORE_COMPARISON = 2`
- `MEDIUM_STAGE3_WEIGHTED_EVIDENCE_SCORE_COMPARISON = 3`
- `STRICT_STAGE3_WEIGHTED_EVIDENCE_SCORE_COMPARISON = 5`

All three modes gate on `stage2_flag == 1` — they refine within Stage 2 confirmed alerts, not the full population.

The purpose of fixed thresholds rather than grid-searched thresholds for the operating modes is to provide stable, named operational reference points that Gold_06A can replay without re-running the grid search.

### Alert Priority Classification

```python
cascade_results["alert_priority"] = np.select(
    [
        (stage2_flag == 1) & (stage3_strong_primary_flag == 1),   # "high"
        (stage2_flag == 1) & (stage3_confirmed_flag == 1),         # "medium"
        (stage2_flag == 1),                                         # "low"
    ],
    ["high", "medium", "low"],
    default="none",
)
cascade_results["stage3_priority_class"] = cascade_results["alert_priority"]
```

Priority levels: `high` (strong primary breach confirmed by Stage 2), `medium` (Stage 2 confirmed plus weighted confirmation), `low` (Stage 2 confirmed only), `none` (no alert). This column allows downstream consumers to filter alerts by operational severity without rerunning the cascade.

---

## Candidate Generation and Row Tracking

### Stable Row Identity

`ensure_stable_row_id(gold_preprocessed_scaled_dataframe, row_id_column="meta__row_id")` is called before Stage 1 scoring. All `merge_stage_results_back` calls join on `meta__row_id`. `validate_cascade_output` confirms uniqueness before truth record creation.

### Stage Flag Column Summary

| Column | Semantics | Non-Candidate Value |
|---|---|---|
| `stage1_flag` | 1 = Stage 1 alert | N/A (all rows scored) |
| `stage1_score` | `-score_samples()` | N/A (all rows scored) |
| `stage2_score` | `-score_samples()` for Stage 1 candidates | NaN (explicit, not zero) |
| `stage2_model_score` | Raw model output for Stage 1 candidates | NaN (from left join) |
| `stage2_raw_flag` | Stage 2 threshold binary (independent) | 0 |
| `stage2_flag` | Gated: stage1==1 AND stage2_raw==1 | 0 |
| `stage3_profile_breach_count` | Primary sensor out-of-bounds count (all rows) | 0 |
| `stage3_secondary_breach_count` | Secondary sensor out-of-bounds count (all rows) | 0 |
| `stage3_profile_breach_flag` | Binary from breach count | 0 |
| `stage3_strong_primary_flag` | Binary from strong_primary_hits threshold | 0 |
| `stage3_corroboration_flag` | Binary from secondary breach count | 0 |
| `stage3_persistence_flag` | Rolling window on stage2_flag | 0 |
| `stage3_drift_flag` | Rolling drift check | 0 |
| `stage3_rule_evidence_count` | Sum of four evidence flags | 0 |
| `stage3_weighted_evidence_score` | Weighted sum of evidence signals | 0.0 |
| `stage3_confirmed_flag` | Strong primary OR (breach AND weighted ≥ threshold) | 0 |
| `cascade_stage3_improved_flag` | Stage 2 AND stage3_confirmed_flag | 0 |
| `cascade_final_flag` | Alias of cascade_stage3_improved_flag | 0 |
| `cascade_stage3_relaxed_flag` | Stage 2 AND weighted ≥ 2 | 0 |
| `cascade_stage3_medium_flag` | Stage 2 AND weighted ≥ 3 | 0 |
| `cascade_stage3_strict_flag` | Stage 2 AND weighted ≥ 5 | 0 |
| `alert_priority` | `"high"` / `"medium"` / `"low"` / `"none"` | `"none"` |
| `stage3_priority_class` | Alias of alert_priority | `"none"` |

`finalize_stage_flag_columns(cascade_results, stage_names=["stage1", "stage2", "stage3"])` fills NaN values and enforces integer types across all flag columns before SQL writes and lineage checks.

### Detected Row Frames

Four detected-row frames are built via `get_detected_rows_dataframe` for interactive post-hoc review:

| Frame | Target Flag | Key Additional Columns |
|---|---|---|
| `stage1_detected_rows_dataframe` | `stage1_flag` | `stage1_score`, `stage1_decision`, `stage1_pred`, all stage flags |
| `stage2_detected_rows_dataframe` | `stage2_flag` | `stage2_score`, `stage2_model_*` columns, all stage flags |
| `stage3_evidence_rows_dataframe` | `stage3_profile_breach_flag` | All Stage 3 evidence columns, `stage3_rule_evidence_count` |
| `final_detected_rows_dataframe` | `cascade_final_flag` | Stage 2 model columns, all Stage 3 evidence columns |

All frames are sorted by `time_index` ascending and logged to the ledger. They are **not saved to disk** — same behavior as Gold_03b, distinct from Gold_03a which saves four row-tracking CSVs.

---

## Evaluation and Metrics

### Stage 3 Candidate-Level Evaluation

Each Stage 3 candidate is evaluated inside `evaluate_stage3_candidate` using the same selection score formula as Stage 2: `3.0 * f1 + 1.0 * precision - 1.0 * alert_rate` when `recall >= STAGE3_MIN_SELECTION_RECALL`, else `-1000.0 + recall`. All candidates and their scores appear in `stage3_search_results`, sorted by `selection_score` descending.

`stage3_search_candidate_count` is extracted from `len(stage3_search_results)` and recorded in: the cascade thresholds JSON, cascade summary JSON, cascade metadata JSON, and all four validation contracts' `rule_config`.

### Operating-Mode Metrics

After the final cascade decision, per-mode metrics are computed using `precision_recall_fscore_support` and, uniquely to Gold 03c, `roc_auc_score` and `average_precision_score` against test labels. A resolver function `_resolve_stage3_variant_flag_column` locates the correct flag column for each mode by checking a candidate column name list (`cascade_stage3_{mode}_flag`, `stage3_{mode}_flag`, etc.) and raising `KeyError` if none is found. A second resolver `_resolve_stage3_score_column` finds a continuous evidence score for ROC/PR AUC computation when available.

Mode-level metrics are pushed into `cascade_metrics` before the summary JSON is built, recorded in `stage3_summary["stage3_operating_mode_metrics"]`, and written into all four validation contracts.

### Final Cascade Metrics

`cascade_metrics` records alert counts at all five operating levels (Stage 1, Stage 2, final/improved, relaxed, medium, strict) across all rows and the test window. When `test_labels` is available, precision, recall, and F1 on `cascade_final_flag` vs. `test_labels` are added.

---

## Artifact and SQL Persistence

### Model Persistence

| Artifact | Persistence | Notes |
|---|---|---|
| Stage 1 Isolation Forest | `joblib.dump` × 2 (`STAGE1_MODEL_ARTIFACT_PATH` and `STAGE1_MODELS_PATH`) | Retrained in Gold 03c |
| Stage 2 Isolation Forest | `joblib.dump` × 2 (`STAGE2_MODEL_ARTIFACT_PATH` and `STAGE2_MODELS_PATH`) | Gold_03b best model, re-saved under `cascade_stage3_improved_*` path keys |
| Stage 3 rule logic | **Not persisted as joblib** | `stage3_saved_as_joblib=False` in all 4 contracts |

Stage 2 is re-saved rather than left at Gold_03b's path — Gold_06A reads Stage 2 from the contract's `stage2_model_path`, which points to Gold_03c's `cascade_stage3_improved_stage2_model_artifact_path`, not Gold_03b's original path.

### JSON Artifacts

| Artifact | Key Contents |
|---|---|
| `cascade_thresholds` | `cascade_variant`, stage thresholds, `stage2_best_params`, `stage3_variant = "tuned_confirmation_layer"`, `stage3_selected_params`, `stage3_search_candidate_count`, `stage3_min_selection_recall` |
| `cascade_summary` | Alert counts per operating mode (5 modes), `stage3_operating_mode_metrics`, `stage3_summary`, truth hashes, feature/sensor counts |
| `cascade_metadata` | Gold_01 provenance (`gold_scaler_kind`, `gold_recommended_imputation`, `gold_feature_set_id`), `stage3_selected_params`, `cascade_truth_hash` |
| `stage3_search_results` | All candidate evaluations from the Stage 3 grid search; saved for audit |

### Validation Contracts (Four)

Four separate contracts are written — one per operating mode — using `build_gold_model_output_validation_contract`:

| Contract `model_id` | `operating_mode` | `flag_column` |
|---|---|---|
| `"stage3_improved"` | `"selected_improved"` | `"cascade_final_flag"` |
| `"stage3_relaxed"` | `"relaxed"` | `"cascade_stage3_relaxed_flag"` |
| `"stage3_medium"` | `"medium"` | `"cascade_stage3_medium_flag"` |
| `"stage3_strict"` | `"strict"` | `"cascade_stage3_strict_flag"` |

All four share:
- `source_notebook = "gold_03c_cascade_modeling"`
- `validation_type = "stage3_rule_artifact"`
- `model_stage = "cascade_stage3_improved_final"`
- `stage3_type = "rule_based"`
- `stage3_saved_as_joblib = False`
- `stage1_model_path`, `stage2_model_path`, `output_artifact_path` (= `CASCADE_RESULTS_PATH_CSV`)
- `lineage_payload` includes `cascade_truth_hash`, `parent_gold_truth_hash`, `stage3_input_source`
- `rule_config` includes `stage3_selected_params`, `stage3_operating_mode_metrics`, `stage3_tuning_grid`, `stage3_search_candidate_count`, and all three severity threshold comparisons

Each contract is registered with a separate ledger entry keyed by `model_id`. The four contracts allow Gold_06A to replay all operating modes independently without re-running the notebook.

### SQL Persistence

```python
WRITE_TO_POSTGRES = True
CASCADE_SQL_MODEL_STAGE = "cascade_stage3_improved_final"

if WRITE_TO_POSTGRES:
    gold_cascade_sql_summary_dataframe = write_gold_cascade_scores_sql(
        engine=engine,
        capstone_schema=CAPSTONE_SCHEMA,
        dataset_id=DATASET_ID,
        run_id=RUN_ID,
        notebook_globals=globals(),
        dataframe=cascade_results,
        dataset_name=globals().get("DATASET_NAME", DATASET_ID),
        model_stage=CASCADE_SQL_MODEL_STAGE,
    )
```

Target: `gold.anomaly_detection_scores`. `WRITE_TO_POSTGRES = True` gate allows dry-run mode. Two post-write verification reads run after the write: one grouping all `(model_name, model_stage)` combinations for the current `dataset_id` / `run_id`, and one filtering to `model_name = 'cascade_isolation_forest_rule_confirmation'` and returning row count, alert count, and `MIN`/`MAX` of `meta_scored_at_utc`. A commented-out `DELETE FROM gold.anomaly_detection_scores WHERE ... model_stage = 'cascade_final'` statement is preserved as a reference for cleaning stale rows from prior runs; it is not active code.

---

## Truth, Audit, and Reproducibility Behavior

### Truth Record Chain

Gold 03c initializes its truth record with `layer_name="gold_cascade"` (same as Gold_03a and Gold_03b) and `parent_truth_hash=GOLD_PARENT_TRUTH_HASH`:

```python
cascade_truth = initialize_layer_truth(
    truth_version=TRUTH_VERSION,
    dataset_name=DATASET_NAME,
    layer_name="gold_cascade",
    process_run_id=cascade_process_run_id,
    pipeline_mode=PIPELINE_MODE,
    parent_truth_hash=GOLD_PARENT_TRUTH_HASH,
)
```

Three truth sections are populated via `update_truth_section`:
- **`config_snapshot`** — runtime config snapshot
- **`runtime_facts`** — `cascade_variant`, `stage3_variant = "tuned_confirmation_layer"`, alert counts across all 5 operating modes, Stage 2 and Stage 3 search results, `stage2_selection_mode`, `stage2_best_params`, `stage3_selected_params`, feature and rule sensor counts
- **`artifact_paths`** — all input paths (from Gold_01 truth) plus all `cascade_stage3_improved_*` output paths

Truth columns stamped into all rows:
- `meta__truth_hash` = `CASCADE_TRUTH_HASH`
- `meta__parent_truth_hash` = `GOLD_PARENT_TRUTH_HASH`
- `meta__pipeline_mode` = `PIPELINE_MODE`

### W&B Lifecycle

`wandb_run.finish()` fires in the ledger-close section, before the final lineage checks and SQL write:

```python
ledger.write_json(cascade_ledger_path)
wandb.save(str(cascade_ledger_path))
wandb_run.finish()
```

The ledger entry at this step includes `comparison_ready: True`, signaling to Gold_04 that this notebook has produced a final cascade output ready for cross-model comparison. Artifacts logged after `finish()` are not registered with the W&B run record.

### Final Lineage Checks (Seven Invariants + Metadata Check)

After `wandb_run.finish()`, seven lineage invariants are verified — identical in structure to Gold_03a and Gold_03b:

1. Required columns present: `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode`
2. `extract_truth_hash(cascade_results)` must match `CASCADE_TRUTH_HASH`
3. `meta__parent_truth_hash` must have exactly one unique non-null value
4. That single value must match `GOLD_PARENT_TRUTH_HASH`
5. `cascade_truth_path` must exist on disk
6. Loaded truth file hash must match `CASCADE_TRUTH_HASH`
7. Loaded truth file parent hash must match `GOLD_PARENT_TRUTH_HASH`

Gold 03c adds one check beyond Gold_03a and Gold_03b: `CASCADE_METADATA_PATH` is reloaded and the embedded `cascade_truth_hash` field is verified to match `CASCADE_TRUTH_HASH`. This confirms that the metadata JSON produced by Gold 03c is internally consistent with the live truth record.

### Cascade Output Validation

`validate_cascade_output(cascade_results, test_mask=test_mask, final_flag_column="cascade_final_flag")` enforces structural and gate integrity:

1. Required columns present: `meta__row_id`, `meta__is_train_flag`, `stage1_flag`, `stage2_raw_flag`, `stage2_flag`, `cascade_final_flag`
2. `meta__row_id` uniqueness
3. Binary values only in all flag columns (NaN-dropped)
4. Stage 2 gate: no row where `stage2_flag == 1` and `stage1_flag != 1`
5. Final gate: no row where `cascade_final_flag == 1` and `stage2_flag != 1`

A separate validation cell checks the three operating-mode columns (`cascade_stage3_relaxed_flag`, `cascade_stage3_medium_flag`, `cascade_stage3_strict_flag`) for binary values and Stage 2 gate compliance per mode. This two-pass validation ensures both the default and operating-mode outputs are internally consistent before contracts are written.

---

## Downstream Technical Handoff

### Gold_04 Comparison

Gold_04 reads Gold_03c's cascade results CSV, cascade summary JSON, and truth record as one of its four upstream model sources. Gold_04 validates Gold_03c's truth hash via a multi-source check and cross-validates that Gold_03c shares `GOLD_PARENT_TRUTH_HASH` with Gold_02, Gold_03a, and Gold_03b.

### Gold_06A Test Replay Validation

Gold_06A reads all four validation contracts, the Stage 1 and Stage 2 joblib models, thresholds JSON, summary JSON, reference profile CSV, and results CSV to replay the Stage 3 Improved cascade (and its three operating modes) against the held-out test set. The four contracts are self-contained: Gold_06A does not need to re-read the Gold_03c truth record to replay the cascade.

### `gold.anomaly_detection_scores`

SQL rows are written with `model_stage="cascade_stage3_improved_final"`. Two verification reads confirm row counts and alert counts after the write.

### Gold_05, Gold_06B

Not determined from available source. Direct file-level dependencies of Gold_05 and Gold_06B on Gold_03c artifacts are not confirmed from the notebook source examined.

---

## Key Technical Decisions

| Decision | Source Evidence | Why It Matters | Verification Method |
|---|---|---|---|
| `STAGE2_SELECTION_SOURCE = "previous_best"` hardcoded | `STAGE2_SELECTION_SOURCE = "previous_best"` in notebook source; comment explains three mode options | Holds Stage 2 constant relative to Gold_03b so any metric difference between Gold_03b and Gold_03c is attributable to Stage 3 changes only | Confirm `best_stage2_result["reused_previous_03b_best"] == True` in ledger or summary; confirm `stage2_selection_source` in cascade summary matches `"previous_best"` |
| Direct file-level dependency on `CASCADE_TUNED_THRESHOLDS_PATH` | `require_mapping(load_json(CASCADE_TUNED_THRESHOLDS_PATH), ...)` in notebook source | Gold_03c cannot run in `"previous_best"` mode without Gold_03b's saved thresholds JSON; execution order constraint is enforced by code, not just convention | Confirm `CASCADE_TUNED_THRESHOLDS_PATH` exists before Gold_03c runs; confirm `previous_03b_stage2_best_params` is non-empty in the ledger |
| Stage 2 model loaded from Gold_03b (not retrained) | `CASCADE_TUNED_STAGE2_MODEL_PATH` in Inputs table; loaded and used in `run_stage2_selection_decision` | Ensures Stage 2 scoring behavior is byte-for-byte identical to Gold_03b — same fitted model, same thresholds; eliminates retraining variance | Load both models; confirm `.get_params()` output is identical; confirm model scores match on a known feature matrix |
| Stage 2 model re-saved under `cascade_stage3_improved_*` path | `joblib.dump(stage2_model, STAGE2_MODEL_ARTIFACT_PATH)` where path uses `cascade_stage3_improved_*` prefix | Gold_06A reads Stage 2 model from the contract's `stage2_model_path`, which points to Gold_03c's artifact directory, not Gold_03b's; Gold_06A contracts and artifact paths must be self-contained | Confirm `STAGE2_MODEL_ARTIFACT_PATH` in Gold_03c contract matches a `cascade_stage3_improved_*` prefix; confirm joblib file exists at that path |
| Strong primary breach shortcut in Stage 3 confirmation | `(strong_primary_flag == 1) \| ((profile_breach_flag == 1) & (weighted_evidence_score >= min_weighted_score))` in `build_stage3_candidate_output` | Prevents the weighted score mechanism from suppressing high-confidence alerts where a single overwhelming primary sensor breach leaves no ambiguity | Confirm rows with `stage3_strong_primary_flag == 1` have `cascade_final_flag == 1` regardless of other evidence counts |
| Stage 3 grid-searched candidate selection uses same formula as Stage 2 | `selection_score = 3.0 * f1 + 1.0 * precision - 1.0 * alert_rate` with `STAGE3_MIN_SELECTION_RECALL` penalty | Keeps recall floors and selection scoring consistent across both tuning stages; `STAGE3_MIN_SELECTION_RECALL` prevents a high-precision, low-recall Stage 3 configuration from winning the search | Read `stage3_search_results`; confirm all winning candidates have `recall >= STAGE3_MIN_SELECTION_RECALL` or confirm they have the highest `selection_score` among all candidates |
| Three fixed operating-mode thresholds (2, 3, 5) rather than grid-searched | Hardcoded constants: `RELAXED = 2`, `MEDIUM = 3`, `STRICT = 5` | Grid-searched thresholds would be optimized but unstable across runs; fixed named thresholds provide stable operational reference points that Gold_06A can replay and operational users can reason about | Confirm `cascade_stage3_relaxed_flag`, `cascade_stage3_medium_flag`, `cascade_stage3_strict_flag` are each a strict superset of the next higher severity level |
| Four validation contracts (one per mode) | Source: `build_gold_model_output_validation_contract` called 4 times in notebook | Gold_06A cannot replay all three operating modes using a single contract; separate contracts expose the full rule_config, metrics, and flag column per mode | Confirm 4 contract JSON files exist at per-mode paths; confirm each contract's `flag_column` matches the corresponding `cascade_stage3_{mode}_flag` column |
| `stage3_saved_as_joblib = False` in all four contracts | `stage3_saved_as_joblib=False` in `build_gold_model_output_validation_contract` call | Signals Gold_06A that Stage 3 must be reconstructed from the reference profile and the contract's `rule_config`, not loaded from a joblib file | Read any contract JSON; confirm `stage3_saved_as_joblib: false` |
| `comparison_ready: True` in ledger at close | `"comparison_ready": True` in `ledger.add(step="finalize_cascade_modeling", ...)` source | Signals Gold_04 that this notebook has produced a final cascade output ready for cross-model comparison; absent from Gold_03a and Gold_03b | Read cascade ledger JSON; confirm `comparison_ready == True` in the `finalize_cascade_modeling` entry |
| Operating-mode metrics include ROC AUC and PR AUC | `roc_auc_score`, `average_precision_score` in cell source for mode-level metrics | More informative than precision/recall/F1 alone when comparing cascade variants at different sensitivity levels; unique to Gold_03c in the cascade sequence | Confirm `stage3_operating_mode_metrics` in cascade summary JSON contains `roc_auc` or `pr_auc` fields per mode |
| `wandb_run.finish()` before lineage checks and SQL | `wandb_run.finish()` at end of ledger-close cell; final lineage checks and SQL write in subsequent cells | Consistent with Gold_03a and Gold_03b; artifacts logged after `finish()` are not registered with the run record, but this is intentional | Confirm SQL rows exist in `gold.anomaly_detection_scores` after the notebook runs regardless of W&B run state |
| Metadata JSON `cascade_truth_hash` re-verified in final lineage checks | `CASCADE_METADATA_PATH` reloaded and `cascade_truth_hash` compared to `CASCADE_TRUTH_HASH` | Additional confirmation that the metadata JSON is internally consistent; absent from Gold_03a and Gold_03b final checks; prevents a scenario where metadata was written before truth was finalized | Read `cascade_metadata` JSON; confirm `cascade_truth_hash` field matches `CASCADE_TRUTH_HASH` |
| `GOLD_PARENT_TRUTH_HASH` resolved from scaled Parquet column (not separate JSON load) | `load_parent_truth_record_from_dataframe(dataframe, truth_dir, parent_layer_name="gold", ...)` in source | Alternative resolution path from Gold_03b (which loads a separate Gold truth JSON); both resolve to the same truth record via the truth index; the embedded column approach avoids a separate file dependency | Confirm `GOLD_PARENT_TRUTH_HASH` in Gold_03c matches `GOLD_PARENT_TRUTH_HASH` in Gold_03a and Gold_03b |

---

## Failure Modes and Guardrails

| Failure Condition | Guard | Behavior |
|---|---|---|
| `CASCADE_TUNED_THRESHOLDS_PATH` missing or empty | `require_mapping(load_json(CASCADE_TUNED_THRESHOLDS_PATH), ...)` | `ValueError` on empty dict; `FileNotFoundError` if path does not exist |
| `stage2_best_params` not a dict in Gold_03b thresholds | `TypeError` check in cell source | `TypeError` with type name in message; halts before Stage 2 model creation |
| `stage2_best_params` is empty dict | `previous_03b_stage2_available = bool(previous_03b_stage2_best_params)` evaluated as `False` | `"previous_best"` path behavior changes; in practice signals a Gold_03b execution state issue |
| Gold_01 truth record missing, empty, or non-dict | `load_parent_truth_record_from_dataframe(...)` | Raises on malformed or absent truth record |
| Feature or sensor JSON empty or non-string | `require_str_list(load_json(path), name)` | `TypeError` or `ValueError` before any model fit |
| `meta__is_train_flag` absent from scaled Parquet | Hard `ValueError` check | Cascade halts; prevents re-deriving a different split boundary |
| Feature column missing from any of the four matrices | Four pre-flight column presence checks | `ValueError` with missing column list before any model is trained |
| Stage 2 gate violation (`stage2_flag==1` where `stage1_flag!=1`) | `validate_cascade_output` gate check | `ValueError` |
| Final gate violation (`cascade_final_flag==1` where `stage2_flag!=1`) | `validate_cascade_output` gate check | `ValueError` |
| Operating-mode column missing (`cascade_stage3_{mode}_flag`) | Variant validation cell explicit column check | `ValueError` with column name |
| Operating-mode gate violation (variant flag where `stage2_flag!=1`) | Variant validation cell gate check per mode | `ValueError` with bad row count |
| Stage 3 operating-mode flag column not resolvable | `_resolve_stage3_variant_flag_column` resolver raises `KeyError` | `KeyError` with checked candidate names and available Stage 3 flag columns |
| `meta__truth_hash` absent or non-matching in `cascade_results` | Final lineage check 1–2 | `ValueError` with observed vs. expected hashes |
| Multiple distinct parent hashes in `cascade_results` | Final lineage check 3 | `ValueError`; catches mixed-upstream rows |
| Parent hash mismatch | Final lineage check 4 | `ValueError` |
| Truth file not written to disk | Final lineage check 5 | `FileNotFoundError` |
| Saved truth file hash mismatch | Final lineage checks 6–7 | `ValueError` on either field |
| `cascade_truth_hash` in metadata JSON mismatches `CASCADE_TRUTH_HASH` | Metadata re-read check (Gold 03c specific) | `ValueError` |

---

## Verification Checklist

- [ ] Active notebook path confirmed: `notebooks/experiments/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling.ipynb`
- [ ] `CASCADE_VARIANT == "stage3_improved"` in all artifact metadata, summary JSONs, and validation contracts
- [ ] `stage3_variant == "tuned_confirmation_layer"` in cascade thresholds JSON and truth `runtime_facts`
- [ ] `STAGE2_SELECTION_SOURCE == "previous_best"` hardcoded; `best_stage2_result["reused_previous_03b_best"] == True`
- [ ] `CASCADE_TUNED_THRESHOLDS_PATH` (Gold_03b thresholds JSON) exists and is non-empty
- [ ] `previous_03b_stage2_available == True` in ledger; `previous_03b_stage2_best_params` non-empty
- [ ] Stage 2 model joblib exists at `cascade_stage3_improved_stage2_model_artifact_path`
- [ ] Stage 2 model params in Gold_03c contract match params from Gold_03b cascade summary
- [ ] `stage3_search_candidate_count` in all four contracts matches `len(stage3_search_results)`
- [ ] All Stage 3 candidates in `stage3_search_results` with `recall >= STAGE3_MIN_SELECTION_RECALL` have `selection_score > -1000`
- [ ] All rows with `stage3_strong_primary_flag == 1` have `cascade_final_flag == 1`
- [ ] `(cascade_results["stage2_flag"] == 1) & (cascade_results["stage1_flag"] != 1)` returns empty frame
- [ ] `(cascade_results["cascade_final_flag"] == 1) & (cascade_results["stage2_flag"] != 1)` returns empty frame
- [ ] All three operating-mode flags gate correctly on `stage2_flag == 1`
- [ ] `cascade_stage3_relaxed_flag >= cascade_stage3_medium_flag >= cascade_stage3_strict_flag` (row-wise)
- [ ] `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` all present in `cascade_results`
- [ ] All non-null `meta__truth_hash` values equal `CASCADE_TRUTH_HASH`
- [ ] All non-null `meta__parent_truth_hash` values equal `GOLD_PARENT_TRUTH_HASH`
- [ ] `GOLD_PARENT_TRUTH_HASH` matches that of Gold_03a and Gold_03b
- [ ] Saved truth file exists; its `truth_hash` and `parent_truth_hash` match in-memory values
- [ ] `cascade_metadata` JSON `cascade_truth_hash` field matches `CASCADE_TRUTH_HASH`
- [ ] Four contract JSON files exist at per-mode paths; each `flag_column` matches the correct mode column
- [ ] `stage3_saved_as_joblib = False` in all four contracts
- [ ] `model_stage = "cascade_stage3_improved_final"` in all four contracts and in SQL rows
- [ ] `comparison_ready == True` in ledger `finalize_cascade_modeling` entry
- [ ] `gold.anomaly_detection_scores` contains rows with `model_stage = "cascade_stage3_improved_final"` for expected `dataset_id` and `run_id`
- [ ] Stage 3 search results JSON exists at `CASCADE_STAGE3_SEARCH_RESULTS_PATH`
- [ ] Detected row frames are not saved as CSV files (display only; no `cascade_stage3_improved__row_tracking_*` CSVs in artifact directories)

---

## Source-Limited Items

| Item | Limitation |
|---|---|
| Whether `CASCADE_TUNED_REFERENCE_PROFILE_PATH` is the active reference profile used in Stage 3, or whether the rebuilt `reference_profile` from `build_reference_profile(gold_fit_dataframe, ...)` is used | The workflow reference Inputs table and Logical Workflow Map state that Gold_03b's saved profile is loaded; the reference profile construction section of the notebook source clearly calls `build_reference_profile(gold_fit_dataframe, ...)`. Whether both are loaded and one is selected, or whether the Logical Workflow Map description reflects an older version, is not resolved from the source examined. |
| Exact columns written to `gold.anomaly_detection_scores` | `write_gold_cascade_scores_sql` is a project utility; column selection is not defined inline in this notebook. Full column mapping requires reading the utility source. |
| Whether `stage3_search_results` is saved to disk as a JSON artifact | The workflow reference mentions `CASCADE_STAGE3_SEARCH_RESULTS_PATH` as an output. The artifact save section references this path variable but the exact `save_json` call targeting it was not confirmed in the source examined. |
| Gold_05 direct dependency on Gold_03c artifacts | Not determined from available source. |
| Gold_06B direct dependency on Gold_03c artifacts | Not determined from available source. |
| Whether `STAGE3_IMPROVED_PARENT_GOLD_TRUTH_HASH` (from `resolve_single_parent_gold_truth_hash`) is used in any downstream check beyond the input lineage verification section | The variable is confirmed to be set; its downstream propagation within the notebook is not determined from available source. |
| `stage3_input_source` global value | Read via `globals().get("STAGE3_INPUT_SOURCE", "gold03c_recomputed_scores_or_existing_cas...")` in contract-building cell. Default fallback string was truncated in source examined. |
| W&B artifact metadata types and aliases | `wandb.save(...)` is confirmed for all artifact files. Whether artifact types, aliases, or metadata dicts are passed is not confirmed from source. |
