# Gold 03b Deep Technical Reference

## Purpose of This Deep Reference

This document covers Gold 03b technical decisions requiring deeper explanation than the workflow reference. The 071b workflow reference (`EDA_Notebook_Pump_Gold_03b_Cascade_Modeling_code_reference.md`) describes what each section does; this document explains why the important methods, configurations, and output designs are structured the way they are — with particular focus on the Stage 2 multi-candidate selection mechanism, which is the primary technical differentiator between Gold 03b and the adjacent cascade notebooks.

---

## Technical Scope

Decision tags applicable to this notebook:

| Tag | Reason Applicable |
|---|---|
| `MODEL_TRAINING` | `IsolationForest.fit()` for Stage 1 (direct) and for each Stage 2 candidate within `run_stage2_selection` |
| `MODEL_EVALUATION` | precision, recall, F1 computed per Stage 2 candidate and for `cascade_final_flag` on the test partition |
| `DATA_VALIDATION` | `validate_cascade_output` gate integrity checks, `require_str_list`, `require_mapping`, feature column presence guards, score-length synchronization |
| `TRUTH_METADATA` | `gold_cascade` truth record initialized with `GOLD_PARENT_TRUTH_HASH`; `CASCADE_TRUTH_HASH` stamped into all result rows |
| `ARTIFACT_WRITE` | cascade results CSV/pickle, Stage 1 and Stage 2 joblib models, reference profile CSV, four JSON artifacts, validation contract |
| `SQL_WRITE` | `write_gold_cascade_scores_sql` → `gold.anomaly_detection_scores` with `model_stage="cascade_tuned_final"` |
| `WANDB_LOGGING` | `wandb.init`, `wandb.save` for all artifact and truth files, `wandb_run.finish()` |
| `LEDGER_UPDATE` | `ledger.add` at each key step; `ledger.write_json` at close |

Tags not applicable: `TEMPORAL_SMOOTHING`, `CORRELATION_REPAIR`, `VARIANCE_CONTROL`, `MEAN_ANCHORING`, `BOUNDS_CLIPPING`, `FAULT_INJECTION`, `PHASE_SPECIFIC_LOGIC`, `MISSINGNESS_REPLAY`.

---

## Source Grounding

| Source | Role |
|---|---|
| `notebooks/experiments/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling.ipynb` (144 cells) | Primary source of truth for all technical claims |
| `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling_code_reference.md` | 071b workflow reference (read-only context) |
| `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling_deep_technical_reference.md` | Gold 03a sequence context (read-only) |
| `notebook_inventory.json` | Active notebook path resolution |

The active Gold 03b notebook is the source of truth. Claims not confirmable from source are labeled `Not determined from available source`.

---

## Stage Role in the Cascade

Gold 03b implements the `"tuned"` cascade variant (`CASCADE_VARIANT = "tuned"`, hardcoded). Its primary technical distinction from Gold 03a is a configurable multi-candidate Stage 2 selection mechanism: rather than using fixed Stage 2 parameters, Gold 03b evaluates one or more Isolation Forest configurations across one or more threshold percentiles and selects the best-scoring candidate based on a precision-weighted selection score with a recall floor constraint.

Gold 03b's `CASCADE_TUNED_THRESHOLDS_PATH` (the saved thresholds JSON) is directly consumed by Gold_03c as the `"previous_best"` Stage 2 source. Gold_03c cannot execute its `"previous_best"` mode without Gold_03b's thresholds file. This is the only direct stage-to-stage artifact dependency between cascade modeling notebooks in the project.

Gold 03b also produces the `"cascade_tuned"` validation contract for Gold_06A and writes scored rows to `gold.anomaly_detection_scores` with `model_stage="cascade_tuned_final"`. Gold_04 (Comparison) validates `CASCADE_TUNED_TRUTH_HASH` from this notebook alongside baseline and other cascade hashes in a three-source check.

Gold 03b shares the same `GOLD_PARENT_TRUTH_HASH` as Gold 03a, Gold 02, and Gold 03c — all cascade notebooks must derive from the same Gold_01 preprocessing run. Gold 04 enforces this by cross-checking parent truth hashes from all three cascade variants.

---

## Input Contract and Lineage

### Primary Inputs

Gold 03b loads its inputs through the same mechanism as Gold 03a: the Gold_01 PreProcessing truth record is the authoritative source for all eight artifact paths.

| Input | Load Mechanism | Strict Validation |
|---|---|---|
| Gold_01 scaled Parquet | `load_data(GOLD_PREPROCESSED_SCALED_DATA_PATH)` | `GOLD_DATASET_NAME` from `meta__dataset` column |
| Gold parent truth record | `require_mapping(load_json(GOLD_TRUTH_PATH))` | Raises `ValueError` on empty dict or non-dict result |
| Four feature/sensor JSON lists | `require_str_list(load_json(path), name)` per list | `TypeError`/`ValueError` if any list is non-string or empty |
| Five Gold Parquets | `load_data(path)` per truth-overridden path | Path fallback to config value if key absent from truth record |

The truth record is loaded explicitly from `GOLD_TRUTH_PATH` (not extracted from an embedded column) — this is a minor structural difference from Gold 03a, which uses `load_parent_truth_record_from_dataframe` to read from the `meta__truth_hash` column. Both resolve to the same Gold_01 truth record.

### Path Override Mechanism

Eight artifact paths are overridden from `gold_truth["artifact_paths"]`:

```python
GOLD_PREPROCESSED_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_preprocessed_path", str(GOLD_PREPROCESSED_DATA_PATH)))
GOLD_FIT_DATA_PATH          = Path(gold_truth_artifact_paths.get("gold_fit_path", str(GOLD_FIT_DATA_PATH)))
GOLD_TEST_DATA_PATH         = Path(gold_truth_artifact_paths.get("gold_test_path", str(GOLD_TEST_DATA_PATH)))
GOLD_TRAIN_DATA_PATH        = Path(gold_truth_artifact_paths.get("gold_train_path", str(GOLD_TRAIN_DATA_PATH)))
STAGE1_FEATURES_PATH        = Path(gold_truth_artifact_paths.get("stage1_features_path", str(STAGE1_FEATURES_PATH)))
STAGE2_FEATURES_PATH        = Path(gold_truth_artifact_paths.get("stage2_features_path", str(STAGE2_FEATURES_PATH)))
STAGE3_PRIMARY_PATH         = Path(gold_truth_artifact_paths.get("stage3_primary_path", str(STAGE3_PRIMARY_PATH)))
STAGE3_SECONDARY_PATH       = Path(gold_truth_artifact_paths.get("stage3_secondary_path", str(STAGE3_SECONDARY_PATH)))
```

Each path falls back to the config-resolved default if the truth record does not contain the key — making the override mechanism non-destructive in environments where the truth record is present but incomplete.

`GOLD_PARENT_TRUTH_HASH = get_truth_hash_from_truth_dict(gold_truth)` is extracted at this point and propagated into the cascade truth record, validation contract lineage payload, cascade metadata JSON, and cascade summary JSON.

### Train/Test Split Recovery

The train/test split is recovered from `meta__is_train_flag`:

```python
train_mask = gold_preprocessed_scaled_dataframe["meta__is_train_flag"] == 1
test_mask  = gold_preprocessed_scaled_dataframe["meta__is_train_flag"] == 0
```

`anomaly_flag` is extracted as `test_labels` for the test-window rows only, when present. The availability of `test_labels` determines which Stage 2 selection scoring path is used — the weighted precision/recall formula when labels are available, or alert-count minimization otherwise.

A hard `ValueError` fires if `meta__is_train_flag` is absent from the input DataFrame. Re-deriving a split independently would risk a different boundary from Gold_01's original split, which would invalidate test-partition evaluation.

---

## Model Input Preparation

### Stage 3 Reference Profile

Built from the normal-only fit subset before any cascade scoring begins:

```python
reference_profile_features = list(dict.fromkeys(
    stage1_feature_columns + stage3_primary_rule_sensors + stage3_secondary_rule_sensors
))
reference_profile = build_reference_profile(gold_fit_dataframe, feature_columns=reference_profile_features)
```

Columns produced: `feature_name`, `median_value`, `mean_value`, `standard_deviation`, `lower_bound` (5th percentile), `upper_bound` (95th percentile). Using fit-only data prevents anomalous test-window values from expanding the normal operating envelope and reducing Stage 3 breach sensitivity. The reference profile is used exclusively for Stage 3 rule confirmation.

### Feature Matrix Assembly

Four typed DataFrames (not NumPy arrays) are constructed:

| Matrix | Source DataFrame | Columns | Purpose |
|---|---|---|---|
| `stage1_train_fit_features` | `gold_fit_dataframe` | `stage1_feature_columns` | Stage 1 `IsolationForest.fit()` |
| `stage2_train_fit_features` | `gold_fit_dataframe` | `stage2_feature_columns` | Stage 2 model fit (inside `evaluate_stage2_model_with_thresholds`) |
| `stage1_all_features` | `gold_preprocessed_scaled_dataframe` | `stage1_feature_columns` | Stage 1 all-rows scoring |
| `stage2_all_features` | `gold_preprocessed_scaled_dataframe` | `stage2_feature_columns` | Stage 2 all-rows scoring (inside selection function) |

All four combinations are validated for missing feature columns before model training begins. Keeping DataFrames rather than `.values` preserves feature names for sklearn and prevents silent column-order mismatches.

---

## Cascade Modeling Methodology

### Stage 1 — Broad Isolation Forest

Stage 1 is structurally identical to Gold 03a:

```python
stage1_model = IsolationForest(n_estimators=STAGE1_ESTIMATOR_COUNT, random_state=RANDOM_SEED, n_jobs=-1)
stage1_model.fit(stage1_train_fit_features)
```

All rows are scored (`stage1_all_scores` from `stage1_all_features`) rather than test rows only, because Stage 2 requires a full-population Stage 1 flag mask to gate its candidate set. The threshold is calibrated on training scores only (`choose_threshold_value(stage1_train_scores, STAGE1_THRESHOLD_PERCENTILE)`) so test anomaly score distributions cannot shift the cut-point.

Stage 1 results are merged into a copy of the Gold scaled DataFrame via `build_stage_scoring_frame` → `score_isolation_forest_stage` → `merge_stage_results_back`, using `meta__row_id` as the join key. This initializes `cascade_results` as an accumulating output frame without mutating the original scaled DataFrame.

### Stage 3 — Rule-Based Confirmation

Stage 3 uses four evidence signals (same structure as Gold 03a):

- **Primary breach**: `compute_primary_breach_count` counts primary sensors outside reference profile bounds
- **Secondary corroboration**: `compute_secondary_breach_count` counts secondary sensors outside bounds
- **Temporal persistence**: `compute_persistence_flag` applied to `stage2_flag` (not `stage1_flag`)
- **Rolling drift**: `compute_drift_flag` with `rolling_window_size=5, drift_threshold_multiplier=1.0` (hardcoded in call, not from `STAGE3_CFG`)

Stage 3 logic runs on all rows before gating is applied. The final cascade decision enforces the cascade contract in the expression:

```python
cascade_results["cascade_final_flag"] = (
    (cascade_results["stage1_flag"] == 1) &
    (cascade_results["stage2_flag"] == 1) &
    (
        (cascade_results["stage3_profile_breach_count"] >= STAGE3_MIN_PRIMARY_SENSOR_HITS) |
        (cascade_results["stage3_rule_evidence_count"] >= 2)
    )
).astype(int)
```

This requires both Stage 1 and Stage 2 confirmation before any Stage 3 evidence is checked. Stage 3 is not saved as a joblib artifact — it is fully reconstructable from the reference profile and config thresholds. The validation contract records `stage3_saved_as_joblib=False`.

---

## Stage 2 Selection and Candidate Refinement

This is the primary technical differentiator of Gold 03b.

### Selection Mode

`STAGE2_SELECTION_MODE` is read from `CONFIG["cascade"]["stage2"]["selection_mode"]` — not hardcoded. Three modes are supported:

| Mode | Candidates Evaluated | Thresholds Evaluated |
|---|---|---|
| `"fixed"` | One candidate (`STAGE2_FIXED_PARAMS`) | One (`STAGE2_FIXED_THRESHOLD_PERCENTILE`) |
| `"threshold_grid"` | One candidate (`STAGE2_FIXED_PARAMS`) | All values in `STAGE2_THRESHOLD_GRID` |
| `"parameter_search"` | All combinations from `ParameterGrid(STAGE2_PARAM_GRID)` | All values in `STAGE2_THRESHOLD_GRID` |

An unsupported mode value causes `run_stage2_selection` to raise `ValueError` before any model is trained.

### Evaluation Function: `evaluate_stage2_model_with_thresholds`

For each `(model_params, threshold_percentile)` pair, `evaluate_stage2_model_with_thresholds`:

1. Fits the Stage 2 Isolation Forest on `stage2_train_fit_features`
2. Scores all rows using `stage2_all_features`
3. Applies the threshold to produce `stage2_raw_flags`
4. Gates Stage 2 flags: `stage2_flags = (stage1_flags == 1) & (stage2_raw_flags == 1)` — Stage 2 can only confirm Stage 1 candidates, not introduce new alerts
5. Computes `precision`, `recall`, `f1`, and `alert_rate` on the test partition (when `test_labels` is available)
6. Computes the selection score:

```python
if float(recall) < float(min_recall):
    selection_score = -1000.0 + float(recall)   # penalized — fails recall floor
else:
    selection_score = (3.0 * float(f1)) + (1.0 * float(precision)) - (1.0 * float(alert_rate))
```

When `test_labels` is `None`, the selection score falls back to alert count minimization:

```python
selection_score = -float(stage2_confirmed_count_test_rows)
```

The function loops over all threshold percentiles for a given model and returns the best single result (highest `selection_score`). The result dict includes the full confusion matrix (`tn`, `fp`, `fn`, `tp`), score arrays, and flag arrays for the winning threshold.

### Recall Floor Design

`STAGE2_MIN_RECALL` is a floor constraint from config, not a target. Any Stage 2 candidate whose recall on the test partition falls below this value receives a penalized score (`-1000.0 + recall`) that cannot exceed any compliant candidate's score regardless of precision or F1. This prevents a degenerate high-precision, near-zero-recall candidate from winning the selection — which would effectively disable anomaly detection at Stage 2.

The recall floor only applies when `test_labels` is available. Without ground truth, alert count minimization is the selection criterion.

### Selection Function: `run_stage2_selection`

`run_stage2_selection` iterates over all `(model_params, threshold_percentile)` pairs, collecting results into a `search_rows` list. After iterating, it returns:
- `best_model` — the winning `IsolationForest` object
- `best_result` — the result dict from the winning `(model_params, threshold_percentile)` pair
- `search_results` — a DataFrame sorted by `selection_score` descending

`stage2_search_results` is logged to the ledger and displayed for post-hoc review. The `stage2_best_params` and `stage2_selected_threshold_percentile` extracted from `best_result` are carried into the cascade thresholds JSON, cascade summary JSON, and validation contract.

`run_stage2_selection` raises `ValueError` if no result was produced (no candidates evaluated). This can only occur if both `model_candidates` and `threshold_candidates` are empty — a configuration error, not a model quality failure.

### Stage 2 Final Scoring

After selection, the winning model is used to produce final Stage 2 columns via `build_stage_scoring_frame` → `score_isolation_forest_stage` → `merge_stage_results_back` (same row-tracking pattern as Stage 1). The raw helper output columns are renamed before merge to avoid overwriting the selection-calibrated flags:

```python
stage2_results_df = stage2_results_df.rename(columns={
    "stage2_score":    "stage2_model_score",
    "stage2_decision": "stage2_model_decision",
    "stage2_pred":     "stage2_model_pred",
    "stage2_flag":     "stage2_model_flag",
})
```

Non-candidate rows (where `stage1_flag != 1`) receive `stage2_score = NaN` — not zero. The in-code comment states this explicitly: "NaN on non-candidate rows signals that Stage 2 did not evaluate them, which distinguishes absent scores from zero scores." Score NaN values are preserved in the final results; only flag column NaN values are filled to integers by `finalize_stage_flag_columns`.

---

## Candidate Generation and Row Tracking

### Stable Row Identity

`ensure_stable_row_id(gold_preprocessed_scaled_dataframe, row_id_column="meta__row_id")` is called before any scoring. All subsequent `merge_stage_results_back` calls join on `meta__row_id`. `validate_cascade_output` confirms `meta__row_id` is unique before truth record creation.

### Stage Flag Columns

All flag columns in `cascade_results` are produced as follows:

| Column | Semantics | Non-Candidate Value |
|---|---|---|
| `stage1_flag` | 1 = Stage 1 alert; 0 = no alert | N/A (all rows scored) |
| `stage1_score` | `-score_samples()` value | N/A (all rows scored) |
| `stage2_score` | `-score_samples()` for Stage 1 candidates | NaN (explicit: not zero) |
| `stage2_model_score` | Raw model output for Stage 1 candidates | NaN (from left join) |
| `stage2_raw_flag` | Stage 2 threshold binary (independent of Stage 1) | 0 |
| `stage2_flag` | Gated Stage 2: 1 only where `stage1_flag==1 AND stage2_raw_flag==1` | 0 |
| `stage3_profile_breach_count` | Primary sensor breach count (all rows) | 0 |
| `stage3_secondary_breach_count` | Secondary sensor breach count (all rows) | 0 |
| `stage3_profile_breach_flag` | Binary from `stage3_profile_breach_count >= STAGE3_MIN_PRIMARY_SENSOR_HITS` | 0 |
| `stage3_corroboration_flag` | Binary from secondary breach count | 0 |
| `stage3_persistence_flag` | Rolling window on `stage2_flag` | 0 |
| `stage3_drift_flag` | Rolling drift check on union sensors | 0 |
| `stage3_rule_evidence_count` | Sum of four Stage 3 binary evidence flags | 0 |
| `cascade_final_flag` | Final decision: stage1 AND stage2 AND (primary_breach OR evidence_count≥2) | 0 |

### Detected Row Frames

Four detected-row frames are extracted from `cascade_results` via `get_detected_rows_dataframe`, targeting `stage1_flag`, `stage2_flag`, `stage3_profile_breach_flag`, and `cascade_final_flag` respectively. These frames are logged to the ledger and displayed for post-hoc review within the notebook. **They are not saved to disk as CSV files in Gold 03b.** This is a meaningful difference from Gold 03a, which saves all four row-tracking frames to `CASCADE_ROW_TRACKING_DIR` as named CSVs.

---

## Evaluation and Metrics

### Stage 2 Candidate-Level Evaluation

Each Stage 2 candidate is evaluated inside `evaluate_stage2_model_with_thresholds` against the test partition when `test_labels` is available:

- `precision`, `recall`, `f1` via `precision_recall_fscore_support(average="binary", zero_division=0)`
- `confusion_matrix(labels=[0, 1]).ravel()` → `tn, fp, fn, tp`
- `alert_rate = stage2_confirmed_count_test_rows / max(test_row_count, 1)`
- `selection_score` (recall-floored weighted formula or alert-count minimization)

All candidate results appear in the `stage2_search_results` DataFrame, sorted by `selection_score` descending, for audit and display.

### Final Cascade Metrics

After `cascade_final_flag` is assigned, per-stage alert counts and optional final cascade evaluation:

```python
cascade_metrics = {
    "model": "3-Stage Cascade",
    "stage1_alert_count_all_rows": ..., "stage1_alert_count_test_rows": ...,
    "stage2_alert_count_all_rows": ..., "stage2_alert_count_test_rows": ...,
    "final_alert_count_all_rows": ...,  "final_alert_count_test_rows": ...,
}
```

If `test_labels` is not None, `precision_recall_fscore_support(average="binary", zero_division=0)` is applied to `cascade_final_flag` vs. `test_labels` on the test partition. The resulting `precision`, `recall`, and `f1` are appended to `cascade_metrics` and carried into the cascade summary JSON, validation contract, and ledger close step.

---

## Artifact and SQL Persistence

### Model Persistence

| Artifact | Persistence | Paths |
|---|---|---|
| Stage 1 Isolation Forest | `joblib.dump` | `STAGE1_MODEL_ARTIFACT_PATH` and `STAGE1_MODELS_PATH` (×2) |
| Stage 2 Isolation Forest (selected) | `joblib.dump` | `STAGE2_MODEL_ARTIFACT_PATH` and `STAGE2_MODELS_PATH` (×2) |
| Stage 3 rule logic | **Not persisted** | Reconstructable from `reference_profile` + config thresholds; `stage3_saved_as_joblib=False` in contract |

Stage 2 is saved after selection — the object returned by `run_stage2_selection` as `best_model` is the winning fitted model. It is the model that produced `best_result` in the search loop.

### JSON Artifacts

| Artifact | Key Contents | Consumer |
|---|---|---|
| `cascade_thresholds` | `cascade_variant`, `stage1_threshold_percentile`, `stage1_threshold`, `stage2_selection_mode`, `stage2_selected_threshold_percentile`, `stage2_threshold`, `stage2_best_params` | **Gold_03c `"previous_best"` source**; Gold_06A |
| `cascade_summary` | Per-stage alert counts, metrics, feature/rule counts, `stage2_search_candidate_count`, `stage2_selection_mode`, truth hashes | Gold_06A |
| `cascade_metadata` | Gold_01 provenance (`gold_process_run_id`, `gold_feature_set_id`, `gold_scaler_path`, `gold_scaler_kind`, `gold_recommended_imputation`), cascade truth linkage | Gold_06A |
| Validation contract (`cascade_tuned_contract`) | `model_id="cascade_tuned"`, `model_stage="cascade_tuned_final"`, `rule_config` (all stage params + `stage2_search_candidate_count`), `stage3_saved_as_joblib=False`, lineage payload | Gold_06A |

The `cascade_thresholds` JSON is the only Gold 03b artifact that Gold_03c reads directly. Gold_03c extracts `stage2_selected_threshold_percentile` and `stage2_best_params` from it to initialize its `"previous_best"` Stage 2 source.

The validation contract's `rule_config` includes `stage2_search_candidate_count`, which is absent from the Gold 03a contract. This allows Gold_06A to verify how many Stage 2 candidates were evaluated during the tuned selection.

### Detected Row Artifacts

Four detected-row frames are built for display only — they are **not saved as CSV files** in Gold 03b. This differs from Gold 03a which saves four row-tracking CSVs to `CASCADE_ROW_TRACKING_DIR`.

### SQL Write

```python
WRITE_TO_POSTGRES = True
CASCADE_SQL_MODEL_STAGE = "cascade_tuned_final"

if WRITE_TO_POSTGRES:
    write_gold_cascade_scores_sql(
        engine=engine, capstone_schema=CAPSTONE_SCHEMA,
        dataset_id=DATASET_ID, run_id=RUN_ID,
        notebook_globals=globals(), dataframe=cascade_results,
        dataset_name=DATASET_NAME, model_stage=CASCADE_SQL_MODEL_STAGE,
    )
```

Target: `gold.anomaly_detection_scores`. The `WRITE_TO_POSTGRES = True` gate allows dry-run mode. SQL write occurs after `wandb_run.finish()` and after the final lineage verification.

All artifacts are registered with W&B via `wandb.save(...)`.

---

## Truth, Audit, and Reproducibility Behavior

### Truth Record Chain

Gold 03b initializes its truth record with `layer_name="gold_cascade"` (same layer name as Gold 03a) and `parent_truth_hash=GOLD_PARENT_TRUTH_HASH`:

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

The `runtime_facts` section captures:
- All stage thresholds and threshold percentiles
- `stage2_selection_mode`, `stage2_best_params`, `stage2_search_candidate_count`
- Estimator counts for Stage 1 and Stage 2 (from `stage2_model.get_params()`)
- Feature and rule sensor counts, result row count
- `gold_process_run_id`, `gold_feature_set_id` (from Gold_01 truth runtime facts)

The `artifact_paths` section records all input paths (with `gold_truth_path`, `gold_fit_path`, `gold_scored_path`, feature/sensor JSON paths) and all output paths (prefixed `cascade_tuned_*`).

Truth columns stamped into all rows of `cascade_results`:
- `meta__truth_hash` = `CASCADE_TRUTH_HASH`
- `meta__parent_truth_hash` = `GOLD_PARENT_TRUTH_HASH`
- `meta__pipeline_mode` = `PIPELINE_MODE`

### W&B Lifecycle

`wandb_run.finish()` fires in the ledger-close section before the final lineage checks and SQL write. All `wandb.save(...)` calls complete before `finish()`. The final lineage checks and SQL write are not part of the W&B run record.

### Final Lineage Checks (Seven Invariants)

After `wandb_run.finish()`, seven invariants are verified:

1. `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` all present in `cascade_results`
2. `extract_truth_hash(cascade_results)` equals `CASCADE_TRUTH_HASH`
3. `cascade_results["meta__parent_truth_hash"]` contains exactly one unique non-null value — multiple values indicate rows from different upstream runs were mixed
4. That unique parent hash equals `GOLD_PARENT_TRUTH_HASH`
5. `cascade_truth_path` exists on disk (`FileNotFoundError` if not)
6. Loaded truth file `truth_hash` equals `CASCADE_TRUTH_HASH`
7. Loaded truth file `parent_truth_hash` equals `GOLD_PARENT_TRUTH_HASH`

`ValueError` is raised on any mismatch with expected and observed values shown side by side.

### Cascade Output Validation

`validate_cascade_output(cascade_results, test_mask, final_flag_column="cascade_final_flag")` enforces structural and gate integrity before truth record creation:

1. Required columns present: `meta__row_id`, `meta__is_train_flag`, `stage1_flag`, `stage2_raw_flag`, `stage2_flag`, `cascade_final_flag`
2. `meta__row_id` unique in `cascade_results`
3. All binary flag columns contain only `{0, 1}` (NaN-dropped)
4. No row where `stage2_flag==1` and `stage1_flag!=1` (Stage 2 gate invariant)
5. No row where `cascade_final_flag==1` and `stage2_flag!=1` (final gate invariant)

The validation result is logged to the ledger before proceeding to truth record creation.

### Why Lineage Matters for Gold 03b

Gold 04 performs a three-source check on `CASCADE_TUNED_TRUTH_HASH` — reading it from the cascade truth record, the cascade summary JSON, and the cascade metadata JSON and requiring all three to agree. If any of these artifacts were produced from a different truth chain than the others, the check fails. Gold 03b's truth stamping and artifact path recording ensure all three sources are consistent.

Gold_03c's `"previous_best"` reuse is lineage-dependent: it reads Gold_03b's thresholds JSON and must operate on the same Gold_01 preprocessing truth chain. If Gold_03b were run against a different Gold_01 truth run, Gold_03c's `"previous_best"` Stage 2 configuration would no longer correspond to the same feature set or preprocessing that Gold_03c uses.

---

## Downstream Technical Handoff

### Gold_03c (Stage 3 Improved)

Gold_03c reads `CASCADE_TUNED_THRESHOLDS_PATH` to extract `stage2_selected_threshold_percentile` and `stage2_best_params`, using them as the `"previous_best"` Stage 2 source. This is a **confirmed direct file-level dependency** — Gold_03c cannot run in `"previous_best"` mode without Gold_03b's saved thresholds JSON.

The workflow reference also notes that Gold_03c loads Gold_03b's Stage 2 model and reference profile without retraining. The validation contract `lineage_payload` includes `"downstream_consumer": "gold_03c_if_stage3_improved_uses_tuned_output"` — explicitly noting that this handoff is conditional on Gold_03c's Stage 3 improvement.

### Gold_06A (Test Replay Validation)

Gold_06A reads:
- Validation contract (`cascade_tuned_contract_path`): provides `model_id="cascade_tuned"`, `model_stage="cascade_tuned_final"`, full `rule_config`, `stage3_saved_as_joblib=False`, model paths, output artifact path, lineage payload
- Stage 1 and Stage 2 joblib models (referenced by paths in the contract)
- Thresholds, summary, reference profile, and results CSV

The validation contract is self-contained: Gold_06A does not need to re-read the Gold_03b truth record to perform replay validation.

### Gold_04 (Comparison)

Gold_04 reads cascade results CSV/pickle and the `CASCADE_TUNED_TRUTH_HASH` for cross-variant comparison. It validates that Gold_03b shares `GOLD_PARENT_TRUTH_HASH` with Gold_03a, Gold_02, and Gold_03c. Gold_04 establishes Gold_03b's tuned cascade results as one of three cascade configurations compared against the Gold_02 baseline.

### Gold_05 (Anomaly Detection)

Not determined from available source. The workflow reference does not confirm Gold_05 as a direct consumer of Gold_03b artifacts.

### Gold_06B

Not determined from available source. Gold_06B dependency on Gold_03b is not confirmed from available source.

---

## Key Technical Decisions

| Decision | Source Evidence | Why It Matters | Verification Method |
|---|---|---|---|
| `STAGE2_SELECTION_MODE` read from config (not hardcoded) | `STAGE2_SELECTION_MODE = str(STAGE2_CFG["selection_mode"]).strip().lower()` in config block | Allows switching between single-candidate, threshold grid, and parameter search without code changes; decouples selection breadth from notebook logic | Read config value; confirm matches `stage2_summary["selection_mode"]` and `cascade_thresholds["stage2_selection_mode"]` |
| `STAGE2_MIN_RECALL` penalizes candidates below recall floor | `if float(recall) < float(min_recall): selection_score = -1000.0 + float(recall)` inside `evaluate_stage2_model_with_thresholds` | Prevents a degenerate high-precision, near-zero-recall Stage 2 from winning selection and effectively disabling anomaly detection | Confirm all winning candidates in `stage2_search_results` have recall ≥ `STAGE2_MIN_RECALL`, or confirm they have the highest selection score among all candidates |
| Selection score formula weights F1 × 3, precision × 1, alert_rate × -1 | Confirmed from `evaluate_stage2_model_with_thresholds` source: `(3.0 * f1) + (1.0 * precision) - (1.0 * alert_rate)` | Prioritizes F1 (balanced precision/recall) while penalizing high alert rates; prevents a model that alerts on nearly every row from winning if recall is borderline | Read `stage2_search_results["selection_score"]` and recompute from `f1`, `precision`, `alert_rate` columns |
| Alert-count minimization fallback when `test_labels is None` | `selection_score = -float(stage2_confirmed_count_test_rows)` inside `evaluate_stage2_model_with_thresholds` else branch | Without ground truth, F1-based selection is not possible; minimizing confirmed alerts is the only unsupervised surrogate; this behavior is explicit and logged | Confirm `test_labels is None` path by checking whether `stage2_summary["precision"]` is None/missing |
| Stage 2 candidate search returns `search_results` DataFrame sorted by `selection_score` | `search_results = pd.DataFrame(search_rows).sort_values(by=["selection_score"], ascending=[False])` | Allows full audit of all evaluated candidates; `stage2_search_candidate_count` is recorded in summary and contract | Confirm `len(stage2_search_results) == cascade_summary["stage2_search_candidate_count"]` |
| Stage 2 raw output columns renamed before merge | `stage2_results_df.rename({"stage2_flag": "stage2_model_flag", ...})` before `cascade_results.merge(...)` | Prevents the helper's auto-generated `stage2_flag` from overwriting the selection-calibrated `stage2_flags` produced during `run_stage2_selection` | Confirm `"stage2_model_flag"` and `"stage2_flag"` both exist in `cascade_results` and have distinct values |
| `stage2_score = NaN` for non-candidate rows (not zero) | `cascade_results["stage2_score"] = np.nan`; set only for `stage2_candidate_mask` rows; in-code comment: "distinguishes absent scores from zero scores" | Makes it unambiguous which rows Stage 2 evaluated; a zero score would suggest Stage 2 found low anomaly evidence rather than "Stage 2 was not reached" | `cascade_results["stage2_score"].notna().sum() == cascade_results["stage1_flag"].sum()` |
| Detected row frames displayed but not saved to disk | No `save_data` calls for the four detected-row frames in the notebook source; workflow reference confirms "not saved to disk as CSVs" | Gold 03b's per-stage audit frames are interactive-review artifacts rather than pipeline artifacts; absence of CSV outputs distinguishes this notebook from Gold 03a | Confirm no CSV files named `cascade_03b__row_tracking` exist in artifact directories after a run |
| `stage2_search_candidate_count` recorded in contract `rule_config` | `"stage2_search_candidate_count": int(len(stage2_search_results))` in `cascade_tuned_contract` `rule_config` dict | Allows Gold_06A to verify replay used the same number of candidates as the original selection; catches re-runs with different search breadth | Read contract JSON; confirm `rule_config.stage2_search_candidate_count` matches `cascade_summary["stage2_search_candidate_count"]` |
| `downstream_consumer` in validation contract lineage payload | `"downstream_consumer": "gold_03c_if_stage3_improved_uses_tuned_output"` in `lineage_payload` | Makes the inter-cascade dependency explicit and conditional; Gold_04 and Gold_06A can confirm this notebook's role without reading Gold_03c source | Read contract JSON; confirm `lineage_payload.downstream_consumer` is present |
| Truth record `layer_name = "gold_cascade"` (same as Gold 03a) | `cascade_truth_layer_name = "gold_cascade"` before `initialize_layer_truth(...)` | The truth layer name does not disambiguate cascade variants; `CASCADE_VARIANT = "tuned"` and artifact path prefixes (`cascade_tuned_*`) carry the variant identity; the shared layer name means Gold_04 looks for both variants under `gold_cascade` in the truth index | Confirm `gold_truth["layer_name"] == "gold_cascade"` and artifact paths carry the `cascade_tuned_*` prefix |
| `wandb_run.finish()` before final lineage checks and SQL write | `ledger.write_json(...); wandb.save(...); wandb_run.finish()` — then lineage verification and SQL write | Artifacts logged after `finish()` are not registered with the W&B run for this execution; closing before these steps is intentional and consistent with Gold 03a and Gold 02 | Confirm the W&B run record shows no artifacts logged after the ledger file; confirm SQL rows exist in `gold.anomaly_detection_scores` regardless of W&B run state |
| `GOLD_PARENT_TRUTH_HASH` shared across all cascade variants | Extracted from Gold_01 truth record; carried into `runtime_facts`, cascade metadata JSON, and lineage payload | Gold_04 cross-validates that Gold_03a, Gold_03b, and Gold_03c all share the same `GOLD_PARENT_TRUTH_HASH`; any run against a different Gold_01 preprocessing output would fail this check | Read `cascade_metadata["parent_gold_truth_hash"]` from Gold_03a, Gold_03b, and Gold_03c; confirm all three are equal |

---

## Failure Modes and Guardrails

| Failure Condition | Guard | Behavior |
|---|---|---|
| Gold_01 truth record missing, empty, or non-dict | `require_mapping(load_json(GOLD_TRUTH_PATH))` | `ValueError` before cascade begins; depends on `GOLD_TRUTH_PATH` existing |
| Feature or sensor JSON empty or non-string | `require_str_list(load_json(path), name)` | `TypeError` or `ValueError` before model fit |
| `meta__is_train_flag` absent from input | Hard `ValueError` check | Cascade halts; prevents re-deriving a different split |
| Stage 1 or Stage 2 feature column missing from either DataFrame | Four pre-flight column presence checks | `ValueError` with missing column list; raised before any model is trained |
| Stage 1 score array length differs from `cascade_results` length | Length equality guard before column assignment | `ValueError`; prevents misaligned score-to-row assignment |
| Unsupported `STAGE2_SELECTION_MODE` value | `else: raise ValueError(...)` in `run_stage2_selection` | `ValueError` before any Stage 2 candidate is evaluated |
| No Stage 2 candidates evaluated (empty grids) | `if best_result is None: raise ValueError(...)` in both evaluation and selection functions | `ValueError`; cannot produce a best model |
| Stage 3 rule sensor absent from `cascade_results` | Sanity check (warns on missing sensors) | Warning logged; Stage 3 computation proceeds on available sensors |
| Required cascade output columns missing | `validate_cascade_output` column check | `ValueError` before truth record creation |
| `meta__row_id` not unique | `validate_cascade_output` uniqueness check | `ValueError` |
| Stage 2 gate violation (`stage2_flag==1` where `stage1_flag!=1`) | `validate_cascade_output` gate check | `ValueError` |
| Final gate violation (`cascade_final_flag==1` where `stage2_flag!=1`) | `validate_cascade_output` gate check | `ValueError` |
| Truth hash mismatch between `cascade_results` and `CASCADE_TRUTH_HASH` | Final lineage check 2 | `ValueError` with observed vs. expected shown side by side |
| Multiple distinct parent hashes in `cascade_results` | Final lineage check 3 | `ValueError`; catches mixed-upstream rows |
| Truth file not written to disk | `Path(cascade_truth_path).exists()` check | `FileNotFoundError` |
| Saved truth file hash mismatch | Final lineage checks 6 and 7 | `ValueError` on any field mismatch |

---

## Verification Checklist

- [ ] Active notebook path confirmed: `notebooks/experiments/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling.ipynb`
- [ ] `CASCADE_VARIANT == "tuned"` in all artifact metadata and summary JSONs
- [ ] `STAGE2_SELECTION_MODE` in `cascade_thresholds` JSON matches config value
- [ ] `stage2_search_candidate_count` in `cascade_summary` and validation contract match `len(stage2_search_results)`
- [ ] All Stage 2 candidates in `stage2_search_results` with `recall >= STAGE2_MIN_RECALL` have `selection_score > -1000`
- [ ] `cascade_results["stage2_score"].notna().sum() == cascade_results["stage1_flag"].sum()` — NaN only for non-candidates
- [ ] `(cascade_results["stage2_flag"] == 1) & (cascade_results["stage1_flag"] != 1)` returns empty frame
- [ ] `(cascade_results["cascade_final_flag"] == 1) & (cascade_results["stage2_flag"] != 1)` returns empty frame
- [ ] `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` all present in `cascade_results`
- [ ] All non-null `meta__truth_hash` values in `cascade_results` equal `CASCADE_TRUTH_HASH`
- [ ] All non-null `meta__parent_truth_hash` values in `cascade_results` equal `GOLD_PARENT_TRUTH_HASH`
- [ ] Saved truth file exists at `cascade_truth_path`; its `truth_hash` and `parent_truth_hash` match in-memory values
- [ ] `stage3_saved_as_joblib = False` in validation contract JSON
- [ ] `model_id = "cascade_tuned"` and `model_stage = "cascade_tuned_final"` in validation contract
- [ ] `CASCADE_TUNED_THRESHOLDS_PATH` (thresholds JSON) exists — required for Gold_03c `"previous_best"` mode
- [ ] Stage 1 and Stage 2 joblib models exist at configured artifact paths
- [ ] `gold.anomaly_detection_scores` contains rows with `model_stage = "cascade_tuned_final"` for expected `dataset_id` and `run_id`
- [ ] Ledger JSON exists at `cascade_ledger_path` and contains `finalize_cascade_modeling` step entry
- [ ] `cascade_metadata` JSON records `gold_feature_set_id` and `gold_scaler_kind` from Gold_01 provenance
- [ ] No row-tracking CSV files named `cascade_03b__row_tracking` are present in artifact directories (detected rows display-only in Gold 03b)

---

## Source-Limited Items

| Item | Limitation |
|---|---|
| Exact columns written to `gold.anomaly_detection_scores` | `write_gold_cascade_scores_sql` is a project utility; its column selection is not defined inline in this notebook. Full column mapping requires reading the utility source. |
| Exact path structure for `GOLD_TRUTH_PATH` resolution | `GOLD_TRUTH_PATH` is read from `RESOLVED_PATHS` using a `cascade_tuned_*` key convention. The exact resolved path depends on the config loader and environment variables, not visible in the notebook alone. |
| Gold_05 dependency on Gold 03b artifacts | Not confirmed from available source. The workflow reference does not list Gold_05 as a confirmed direct consumer of Gold_03b outputs. |
| Gold_06B dependency on Gold 03b | Not determined from available source. |
| W&B artifact log types and metadata | `wandb.save(...)` is confirmed for all artifact files. Whether artifact types, aliases, or metadata dicts are passed alongside the file path is not confirmed from the notebook source. |
| Whether `stage3_confirmed_flag` is ever present in `cascade_results` | The notebook contains a conditional `cascade_results["stage3_flag"] = cascade_results["stage3_confirmed_flag"].fillna(0).astype(int)` in `finalize_stage_flag_columns`. Whether `stage3_confirmed_flag` is populated during a standard Gold_03b run is not determined from available source. |
| `STAGE2_RANDOM_STATE` source | `run_stage2_selection` receives `random_seed=STAGE2_RANDOM_STATE`. Whether this is read from config separately from `STAGE1_RANDOM_STATE` or is an alias for `RANDOM_SEED` is not confirmed from the available notebook source. |
