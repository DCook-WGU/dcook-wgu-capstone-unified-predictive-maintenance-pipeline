# Notebook Workflow Reference: EDA_Notebook_Pump_Gold_03c_Cascade_Modeling

**Source notebook:** `notebooks/experiments/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling.ipynb`
**Reference type:** Workflow-level notebook code reference (071b format)
**Layer:** Gold — Cascade Modeling (Stage 3 Improved Variant)
**Context:** This notebook is the third and final cascade modeling stage in the Gold layer. It introduces a tunable, weighted Stage 3 confirmation layer with multi-candidate parameter search and three calibrated operating modes (relaxed, medium, strict). It explicitly reuses the Gold_03b tuned Stage 2 model selection by loading Gold_03b's saved thresholds, and produces the final cascade output consumed by Gold_04 Comparison and Gold_06A Test Replay Validation.

---

## Notebook Purpose

Gold_03c implements the Stage 3 Improved cascade pipeline (`CASCADE_VARIANT = "stage3_improved"`). It reuses Gold_03b's Stage 2 model selection (`STAGE2_SELECTION_SOURCE = "previous_best"`) and introduces tunable, weighted Stage 3 confirmation with multi-candidate parameter grid search, producing three calibrated operating modes: relaxed, medium, and strict. It writes four separate validation contracts (one per mode plus one default), writes scored rows to `gold.anomaly_detection_scores` with `model_stage="cascade_stage3_improved_final"`, and is the final cascade notebook consumed by Gold_04 Comparison and Gold_06A Test Replay Validation.

## Inputs

| Input | Source | Used For |
|---|---|---|
| Gold_01 scaled Parquet | `GOLD_PREPROCESSED_SCALED_DATA_PATH` | Primary cascade scoring input; parent truth extraction |
| Gold_01 truth record | `GOLD_TRUTH_PATH` (JSON) + `require_mapping` | Parent hash; 8 artifact path overrides; runtime facts |
| Fit / test / train / preprocessed Parquets | Truth-overridden paths | Stage 1/2 fit data; reference profile; partition reference |
| Stage 1 / Stage 2 feature JSONs | Truth-overridden paths | IF feature sets |
| Stage 3 primary / secondary sensor JSONs | Truth-overridden paths | Rule sensor sets |
| Gold_03b thresholds JSON | `CASCADE_TUNED_THRESHOLDS_PATH` | `stage2_selected_threshold_percentile`, `stage2_best_params` for `"previous_best"` Stage 2 reuse |
| Gold_03b Stage 2 model | `CASCADE_TUNED_STAGE2_MODEL_PATH` (joblib) | Loaded and fitted; Stage 2 scoring in cascade |
| Gold_03b reference profile | `CASCADE_TUNED_REFERENCE_PROFILE_PATH` (CSV) | Stage 3 breach bounds |

## Configuration and Runtime Context

| Item | Source | Value / Purpose |
|---|---|---|
| `CASCADE_VARIANT` | Hardcoded `"stage3_improved"` | Artifact naming key |
| `STAGE2_SELECTION_SOURCE` | Hardcoded `"previous_best"` | Reuse Gold_03b Stage 2 best config — no new Stage 2 search |
| `STAGE3_WEIGHT_GRID` | Config | Candidate weight tuples `(w_primary, w_secondary, w_persistence, w_drift)` for Stage 3 search |
| `STAGE3_RECALL_FLOOR` | Config | Recall floor; candidates below this are penalized |
| `STAGE3_OPERATING_MODES` | Config | `{"relaxed": ..., "medium": ..., "strict": ...}` score thresholds |
| `STAGE1_ESTIMATOR_COUNT`, `STAGE1_THRESHOLD_PERCENTILE` | `STAGE1_CFG` | Stage 1 IF parameters (same as Gold_03b) |
| `GOLD_PARENT_TRUTH_HASH` | Extracted from Gold_01 truth record | Parent hash for `gold_cascade` truth chain |
| `DATASET_NAME` | Extracted from Gold_01 truth record | Confirmed from truth, not just config |
| `CAPSTONE_SCHEMA`, `DATASET_ID`, `RUN_ID`, `ASSET_ID` | Env vars / config | SQL write targets and row identity |

## Logical Workflow Map

1. Bootstrap: `load_notebook_context()`, config extraction, artifact dirs (`variant="stage3_improved"`), DB engine, W&B init, SQL smoke check
2. Data load: Gold_01 scaled Parquet → Gold truth record → 8 path overrides → 5 Parquets + 4 JSON lists; load Gold_03b Stage 2 model and thresholds
3. Row identity: `ensure_stable_row_id`
4. Split recovery: from `meta__is_train_flag`
5. Reference profile: load Gold_03b's saved profile (rather than rebuilding)
6. Feature matrices: 4 DataFrames (Stage 1/2 × fit/all-rows)
7. Stage 1: Broad IF fit, score, threshold calibration, row tracking, `cascade_results` initialized
8. Stage 2: `run_stage2_selection_decision(selection_source="previous_best", ...)` → apply Gold_03b's Stage 2 model; final scoring, Stage 2 gate
9. Stage 3 grid search: `evaluate_stage3_model_with_weights(...)` across `STAGE3_WEIGHT_GRID`; select best by weighted recall-precision tradeoff
10. Stage 3 scoring: best-weight evidence count; `cascade_final_flag`; three operating mode flags (relaxed/medium/strict) from score thresholds
11. `finalize_stage_flag_columns`
12. Cascade metrics: alert counts per mode; precision/recall/F1 on test window
13. `validate_cascade_output` (run once per mode)
14. Truth record: build → stamp → save → index
15. Artifact saves: results CSV/pickle, 4 operating mode flags, Stage 1 model, search results, threshold/summary/metadata JSONs; W&B uploads
16. Validation contracts: 4 × `build_gold_model_output_validation_contract` + write (default, relaxed, medium, strict)
17. Detected row frames (display only)
18. Ledger close + `wandb_run.finish()`
19. Final lineage checks: 7-step hash invariant verification
20. SQL write: `write_gold_cascade_scores_sql` → `gold.anomaly_detection_scores` (`model_stage="cascade_stage3_improved_final"`)

## Section Overview

| Section | Purpose | Key Outputs |
|---|---|---|
| Bootstrap | Context, dirs, DB, W&B | `CTX`, `GOLD_CASCADE_ARTIFACT_DIRS`, `wandb_run` |
| Data load and Gold truth | Load inputs; inherit parent truth; 8 path overrides; load Gold_03b Stage 2 | 5 DataFrames, 4 JSON lists, `stage2_model`, `gold_03b_thresholds` |
| Row identity | `ensure_stable_row_id` | Unique `meta__row_id` on all rows |
| Train/test split | Recover from `meta__is_train_flag` | `train_mask`, `test_mask` |
| Reference profile | Load Gold_03b's saved profile | `reference_profile` |
| Feature matrices | 4 typed DataFrames for Stage 1/2 | `stage1/2_train_fit_features`, `stage1/2_all_features` |
| Stage 1 | Broad IF screening | `stage1_model`, `cascade_results` initialized |
| Stage 2 | Apply Gold_03b best Stage 2; gate | `stage2_flag` |
| Stage 3 grid search | Multi-candidate weight search | `stage3_best_weights`, `stage3_search_results` |
| Stage 3 scoring + operating modes | Best-weight scoring; relaxed/medium/strict flags | `cascade_final_flag`, 3 mode flags |
| Cascade metrics + validation | Alert counts per mode; `validate_cascade_output` | `cascade_metrics` |
| Truth record | Build, stamp, save, index | `CASCADE_TRUTH_HASH` |
| Artifact saves | All outputs to disk and W&B | Results, models, JSONs |
| Validation contracts | 4 contracts for Gold_06A | 4 contract JSON files |
| Ledger / W&B close | Finalize run record | Ledger JSONL; W&B closed |
| Final lineage checks | 7-step hash invariant verification | `ValueError` on mismatch |
| SQL write | Scored rows to database | `gold.anomaly_detection_scores` |

## Section Details

Detailed workflow sections follow. See **Notebook Bootstrap**, **Configuration Block**, **Artifact Directory Setup**, **Data Load and Gold Truth**, **Stage 1**, **Stage 2**, **Stage 3 Grid Search**, **Stage 3 Operating Modes**, and subsequent sections.

## Key Function Calls and In-Place Usage

| Function | Context | Return / Side Effect |
|---|---|---|
| `load_notebook_context("gold_cascade", ...)` | Bootstrap | `CTX` with all shared context aliases |
| `build_artifact_dirs_from_config(config, stage_key="gold_cascade", variant="stage3_improved")` | Artifact dirs | `GOLD_CASCADE_ARTIFACT_DIRS` dict |
| `require_mapping(load_json(GOLD_TRUTH_PATH))` | Load Gold truth | `gold_truth` dict; raises on empty/non-dict |
| `require_str_list(load_json(path), name)` | Load feature/sensor JSON lists | Validated list; raises on malformed |
| `ensure_stable_row_id(df, "meta__row_id")` | Row identity | `meta__row_id` stamped and unique |
| `run_stage2_selection_decision(selection_source="previous_best", previous_best=gold_03b_thresholds, ...)` | Stage 2 reuse | Best Stage 2 config from Gold_03b thresholds |
| `evaluate_stage3_model_with_weights(cascade_results, reference_profile, weights, ...)` | Stage 3 search | Result dict with `selection_score`, metrics per weight candidate |
| `compute_primary_breach_count`, `compute_secondary_breach_count` | Stage 3 rule | Breach count Series |
| `compute_persistence_flag`, `compute_drift_flag` | Stage 3 rule | Binary flag Series |
| `finalize_stage_flag_columns(cascade_results, stage_names)` | Fill NaN | All flag columns as integers |
| `validate_cascade_output(cascade_results, test_mask)` | Integrity per mode | `ValueError` on gate violation |
| `build_truth_record(...)` / `stamp_truth_columns(...)` / `save_truth_record(...)` | Truth chain | `CASCADE_TRUTH_HASH`; stamped results |
| `build_gold_model_output_validation_contract(model_id, ...)` | Gold_06A contracts | 4 contract dicts (default + 3 modes) |
| `write_gold_cascade_scores_sql(engine, ..., model_stage="cascade_stage3_improved_final")` | SQL write | Rows in `gold.anomaly_detection_scores` |

## Outputs and Artifacts

| Output | Type | Location | Downstream Consumer |
|---|---|---|---|
| Cascade results | CSV + Pickle | `CASCADE_RESULTS_PATH_CSV/PICKLE` | Gold_04; Gold_06A |
| Stage 1 model | Joblib | `STAGE1_MODEL_ARTIFACT_PATH` | Gold_06A |
| Reference profile | CSV | `CASCADE_REFERENCE_PROFILE_PATH` | Gold_06A |
| Cascade thresholds | JSON | `CASCADE_THRESHOLDS_PATH` | Gold_06A |
| Cascade summary | JSON | `CASCADE_SUMMARY_PATH` | Gold_04; Gold_06A |
| Cascade metadata | JSON | `CASCADE_METADATA_PATH` | Gold_06A |
| Stage 3 search results | JSON | `CASCADE_STAGE3_SEARCH_RESULTS_PATH` | Audit |
| Validation contracts (4) | JSON | Per-mode contract paths | Gold_06A (all 4 modes) |
| `gold_cascade` truth record | JSON | `TRUTHS_PATH/gold_cascade/` | Truth index; lineage checks |
| Cascade ledger | JSONL | `cascade_ledger_path` | Lineage audit |
| SQL rows | `gold.anomaly_detection_scores` (model_stage=`"cascade_stage3_improved_final"`) | Operational layer |

## Data Quality / Validation Behavior

| Check | Purpose | Failure / Risk Prevented |
|---|---|---|
| `require_mapping` on Gold truth record | Gold_01 truth must be a non-empty dict | `ValueError` before any cascade begins |
| `require_str_list` on 4 JSON lists | Non-empty string lists for features/sensors | `TypeError`/`ValueError` before model fit |
| Gold_03b thresholds presence | `stage2_selected_threshold_percentile` and `stage2_best_params` must exist in loaded dict | `KeyError` if Gold_03b thresholds absent or malformed |
| Missing feature validation (4 frame checks) | All Stage 1/2 columns present in both DataFrames | `ValueError` with missing column list |
| `STAGE3_RECALL_FLOOR` penalty | Penalize Stage 3 weight candidates below recall floor | Prevents an imprecise weight from winning the search |
| `validate_cascade_output` per mode | Required columns, binary flags, gate integrity per operating mode | `ValueError` on structural or gate violation |
| Final 7-step lineage invariants | `meta__truth_hash` roundtrip, parent hash uniqueness, truth file re-read | `ValueError` on any mismatch |

## Downstream Handoff

Gold_04 (Comparison) reads Gold_03c's cascade results CSV, summary JSON, and truth record as one of its four upstream model sources. Gold_04 validates Gold_03c's truth hash via a three-source check and cross-validates it shares the same `GOLD_PARENT_TRUTH_HASH` as Gold_02 and Gold_03a/b.

Gold_06A reads all four validation contracts, the Stage 1 joblib model, thresholds, summary, reference profile, and results CSV to replay the Stage 3 Improved cascade (and its three operating mode variants) against the held-out test set.

`gold.anomaly_detection_scores` receives cascade-scored rows with `model_stage="cascade_stage3_improved_final"`.

---

## Notebook at a Glance

| Property | Value |
|---|---|
| Cell count | 178 total (67 code, 111 markdown) |
| CASCADE_VARIANT | `"stage3_improved"` (hardcoded) |
| Stage 3 variant label | `"tuned_confirmation_layer"` |
| STAGE2_SELECTION_SOURCE | `"previous_best"` (hardcoded — reuses Gold_03b Stage 2) |
| Parent truth source | Gold_01 PreProcessing truth record |
| Output truth layer | `"gold_cascade"` |
| W&B job_type | `gold_modeling_cascade` |
| Validation contracts written | 4 — one per Stage 3 operating mode |
| SQL model_stage | `"cascade_stage3_improved_final"` |
| Downstream consumers | Gold_04 Comparison, Gold_06A Test Replay Validation |

---

## Pipeline Role

Gold_03c is the improved cascade notebook. Its specific responsibilities relative to the prior cascade notebooks are:

- **Gold_03a (default)**: Fixed Stage 2 selection; basic rule-based Stage 3
- **Gold_03b (tuned)**: Multi-candidate Stage 2 selection; same Stage 3 rule logic; saves best Stage 2 model and thresholds for reuse
- **Gold_03c (stage3_improved)**: Reuses Gold_03b's best Stage 2 configuration; introduces tunable, weighted Stage 3 with grid search; produces three calibrated operating-mode variants; generates four separate validation contracts; is the final cascade output consumed by downstream comparison and validation notebooks

---

## Imports and Library Setup

Standard scientific Python (`numpy`, `pandas`, `sklearn`, `joblib`, `wandb`) plus the full project utility stack. Key utility imports:

- `load_data`, `save_data`, `save_json`, `load_json` — from `utils.core.file_io`
- `load_notebook_context` — notebook context bootstrap
- Truth utilities: `initialize_layer_truth`, `update_truth_section`, `build_truth_record`, `save_truth_record`, `append_truth_index`, `stamp_truth_columns`, `extract_truth_hash`, `identify_meta_columns`, `identify_feature_columns`, `make_process_run_id`, `load_parent_truth_record_from_dataframe`, `get_dataset_name_from_truth`, `get_truth_hash`, `get_pipeline_mode_from_truth` — from `utils.core.truths`
- `load_pipeline_config`, `build_truth_config_block`, `export_config_snapshot` — from `utils.core.config_loader`
- Cascade row-tracking utilities: `ensure_stable_row_id`, `build_stage_scoring_frame`, `score_isolation_forest_stage`, `merge_stage_results_back`, `finalize_stage_flag_columns`, `get_detected_rows_dataframe`, `get_stage_detected_rows_dataframe` — from `utils.medallion.gold.cascade_row_tracking`
- `build_cascade_variant_contract`, `build_stage3_rule_payload_from_globals`, `write_json_contract` — from `utils.medallion.gold.gold_cascade_validation_contracts`
- `build_gold_model_output_validation_contract`, `write_gold_model_output_validation_contract`, `gold_model_validation_contract_path` — validation contract utilities for Gold_06A consumption
- `write_gold_cascade_scores_sql`, `read_sql_dataframe`, `get_engine_from_env` — database utilities
- `require_str_list`, `require_mapping` — strict validation wrappers (inline-defined in notebook)
- `ParameterGrid` — from `sklearn.model_selection`
- `precision_recall_fscore_support`, `confusion_matrix` — from `sklearn.metrics`
- `product` — from `itertools` (used in Stage 3 tuning grid search)

Several utility functions are also defined inline in the notebook: `choose_threshold_by_percentile`, `compute_anomaly_scores_isolation_forest`, mapping guard helpers (`cfg_require_mapping`, `cfg_optional_mapping`), type conversion helpers (`as_bool_array`, `as_int_array`, `as_float_array`, `require_float`), `resolve_single_parent_gold_truth_hash`, `build_reference_profile`, and all Stage 3 computation functions.

---

## Notebook Bootstrap

```python
CTX = load_notebook_context(
    stage="gold_cascade",
    dataset="pump",
    mode="train",
    profile="default",
    logger_child_name="capstone.gold.cascade.stage3_improved",
    log_filename="gold_modeling_cascade_stage3_improved.log",
)
```

Key aliases unpacked from `CTX`: `paths`, `CONFIG`, `STAGE_CFG` (aliased as `GOLD_CFG`), `RESOLVED_PATHS`, `FILENAMES`, `VERSIONS_CFG`, `RUNTIME_CFG`, `DATASET_CFG`, `WANDB_CFG`, `EXECUTION_CFG`, `PIPELINE`, `logger`, `ledger`, `LOG_PATH`, `CONTEXT_RECIPE_ID`.

A context sanity check verifies all required context variables are present before any configuration is read.

---

## Configuration Block

```python
CASCADE_VARIANT = "stage3_improved"   # hardcoded — determines filenames and artifact path keys
```

**Stage 2 configuration** (same structure as Gold_03b; config-driven, with `"previous_best"` override applied later):
- `STAGE2_SELECTION_SOURCE` — initially read from config; overridden to `"previous_best"` in the Stage 2 load cell
- `STAGE2_SELECTION_MODE` — from config (`"parameter_search"` default); used for fallback search if not reusing Gold_03b
- `STAGE2_MIN_RECALL`, `STAGE2_RANDOM_STATE`, `STAGE2_FIXED_THRESHOLD_PERCENTILE`, `STAGE2_FIXED_PARAMS`
- `STAGE2_WARNING_THRESHOLD_PERCENTILE`, `STAGE2_CONFIRMED_THRESHOLD_PERCENTILE` — additional threshold percentile levels for two-level Stage 2 alerting
- `STAGE2_THRESHOLD_GRID`, `STAGE2_PARAM_GRID` — for fallback search

**Stage 3 configuration** (new to Gold_03c — not present in Gold_03a or Gold_03b):
- `STAGE3_TUNING_GRID` — grid of candidate values for `min_weighted_score`, `rolling_window_size`, `minimum_flags_in_window`, `strong_primary_hits`, `drift_threshold_multiplier`
- `STAGE3_MIN_WEIGHTED_SCORE`, `STAGE3_STRONG_PRIMARY_HITS`, `STAGE3_DRIFT_THRESHOLD_MULTIPLIER` — defaults from `STAGE3_TUNING_GRID`
- `STAGE3_MIN_SELECTION_RECALL` — recall floor for Stage 3 candidate selection
- Evidence weights: `STAGE3_PROFILE_BREACH_WEIGHT`, `STAGE3_CORROBORATION_WEIGHT`, `STAGE3_PERSISTENCE_WEIGHT`, `STAGE3_DRIFT_WEIGHT`
- `STAGE3_DRIFT_ROLLING_WINDOW_SIZE` — separate window size for drift detection
- Standard Stage 3 params: `STAGE3_MIN_PRIMARY_SENSOR_HITS`, `STAGE3_MIN_SECONDARY_SENSOR_HITS`, `STAGE3_ROLLING_WINDOW_SIZE`, `STAGE3_MINIMUM_FLAGS_IN_WINDOW`

---

## Artifact Directory Setup

```python
GOLD_CASCADE_ARTIFACT_DIRS = build_artifact_dirs_from_config(
    config=CONFIG,
    stage_key="gold_cascade",
    variant=CASCADE_VARIANT,   # "stage3_improved"
)
```

All output paths use `FILENAMES` keys with the `cascade_stage3_improved_*` prefix:

| Variable | Artifact |
|---|---|
| `CASCADE_RESULTS_PATH_CSV` | `scores/cascade_stage3_improved_results_*.csv` |
| `CASCADE_RESULTS_PATH_PICKLE` | `scores/cascade_stage3_improved_results_*.pkl` |
| `CASCADE_THRESHOLDS_PATH` | `thresholds/cascade_stage3_improved_thresholds_*.json` |
| `CASCADE_SUMMARY_PATH` | `summaries/cascade_stage3_improved_summary_*.json` |
| `CASCADE_METADATA_PATH` | `metadata/cascade_stage3_improved_metadata_*.json` |
| `CASCADE_REFERENCE_PROFILE_PATH` | `profiles/cascade_stage3_improved_reference_profile_*.csv` |
| `STAGE1_MODEL_ARTIFACT_PATH` | `models/cascade_stage3_improved_stage1_model_*.joblib` |
| `STAGE2_MODEL_ARTIFACT_PATH` | `models/cascade_stage3_improved_stage2_model_*.joblib` |
| `cascade_ledger_path` | `lineage/gold_cascade_stage3_improved_ledger_*.json` |
| `CONFIG_SNAPSHOT_PATH` | `config/{dataset}__gold_cascade_stage3_improved__resolved_config.yaml` |

Two additional Gold_03b artifact paths are also resolved from `RESOLVED_PATHS`:
- `CASCADE_TUNED_THRESHOLDS_PATH` — Gold_03b's saved thresholds JSON; consumed in Stage 2 source loading
- `CASCADE_TUNED_SUMMARY_PATH` — Gold_03b's saved summary JSON

A config snapshot is exported if `CONFIG["execution"]["save_config_snapshot"]` is `True`.

---

## SQL Smoke Check

After artifact directories are created, a lightweight read query against `information_schema.tables` confirms schema connectivity. This check does not write any data.

---

## Weights & Biases Initialization

```python
wandb_run = wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name=WANDB_RUN_NAME,
    job_type="gold_modeling_cascade",
    config={
        "cascade_variant": CASCADE_VARIANT,
        "stage2_selection_mode": STAGE2_SELECTION_MODE,
        "stage3_strong_primary_hits": STAGE3_STRONG_PRIMARY_HITS,
        "stage3_profile_breach_weight": STAGE3_PROFILE_BREACH_WEIGHT,
        "stage3_corroboration_weight": STAGE3_CORROBORATION_WEIGHT,
        "stage3_persistence_weight": STAGE3_PERSISTENCE_WEIGHT,
        "stage3_drift_weight": STAGE3_DRIFT_WEIGHT,
        "stage3_min_weighted_score": STAGE3_MIN_WEIGHTED_SCORE,
        "stage3_drift_threshold_multiplier": STAGE3_DRIFT_THRESHOLD_MULTIPLIER,
        ...
    },
)
```

The W&B config captures all Stage 3 weights and the `cascade_variant` identifier.

---

## Data Load — Gold Scaled Input + Parent Truth Resolution

Gold_03c resolves the parent truth hash differently from Gold_03a and Gold_03b. Rather than loading a separate Gold truth JSON first, it resolves the parent truth record directly from the `meta__truth_hash` column already stamped into the scaled Parquet:

```python
gold_preprocessed_scaled_dataframe = load_data(GOLD_PREPROCESSED_SCALED_DATA_PATH)

# DATASET_NAME resolved from the meta__dataset column of the scaled dataframe
GOLD_DATASET_NAME = gold_preprocessed_scaled_dataframe["meta__dataset"].dropna()...

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

`GOLD_PARENT_TRUTH_HASH` is the truth hash from Gold_01 PreProcessing — captured before any cascade modifications so every stage truth record can trace back to its preprocessing parent.

Eight input paths are then overridden from the Gold truth record's `artifact_paths` section (same fallback pattern as Gold_03a/03b):

```python
GOLD_PREPROCESSED_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_preprocessed_path", ...))
GOLD_FIT_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_fit_path", ...))
GOLD_TEST_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_test_path", ...))
GOLD_TRAIN_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_train_path", ...))
STAGE1_FEATURES_PATH = Path(gold_truth_artifact_paths.get("stage1_features_path", ...))
STAGE2_FEATURES_PATH = Path(gold_truth_artifact_paths.get("stage2_features_path", ...))
STAGE3_PRIMARY_PATH = Path(gold_truth_artifact_paths.get("stage3_primary_path", ...))
STAGE3_SECONDARY_PATH = Path(gold_truth_artifact_paths.get("stage3_secondary_path", ...))
```

Five Parquets are then loaded via `load_data`: `gold_preprocessed_scaled_dataframe` (already loaded), `gold_preprocessed_dataframe`, `gold_fit_dataframe`, `gold_test_dataframe`, `gold_train_dataframe`.

Four JSON lists are loaded using `require_str_list(load_json(path), name)`: `stage1_feature_columns`, `stage2_feature_columns`, `stage3_primary_rule_sensors`, `stage3_secondary_rule_sensors`.

A secondary lineage variable `STAGE3_IMPROVED_PARENT_GOLD_TRUTH_HASH` is resolved from `resolve_single_parent_gold_truth_hash` — an inline-defined function that reads `meta__parent_truth_hash` from the dataframe and validates that exactly one unique non-null value is present.

---

## Row Identity Stabilization

```python
gold_preprocessed_scaled_dataframe = ensure_stable_row_id(
    gold_preprocessed_scaled_dataframe,
    row_id_column="meta__row_id",
)
```

`meta__row_id` is guaranteed unique before any stage scoring. All `merge_stage_results_back` calls join on `meta__row_id`.

---

## Train/Test Split Recovery

```python
train_mask = gold_preprocessed_scaled_dataframe["meta__is_train_flag"].fillna(False).astype(bool)
test_mask = (~train_mask).astype(bool)
```

The split is restored from the embedded `meta__is_train_flag` column; no re-splitting occurs. `anomaly_flag` is extracted for test-window rows when present; when absent, `test_labels = None` and all metrics fall back to alert-rate-based selection.

---

## Reference Profile Construction

```python
reference_profile_features = list(dict.fromkeys(
    stage1_feature_columns + stage3_primary_rule_sensors + stage3_secondary_rule_sensors
))

reference_profile = build_reference_profile(
    gold_fit_dataframe,
    feature_columns=reference_profile_features,
)
```

`build_reference_profile` is defined inline. It records `median_value`, `mean_value`, `standard_deviation`, `lower_bound` (5th percentile), and `upper_bound` (95th percentile) per feature from the normal-only fit subset. Building from `gold_fit_dataframe` prevents anomalous test-window values from affecting the normal bounds used in Stage 3 rule evaluation.

---

## Feature Matrix Assembly

Four typed DataFrames are constructed:

| Variable | Source | Columns |
|---|---|---|
| `stage1_train_fit_features` | `gold_fit_dataframe` | `stage1_feature_columns` |
| `stage2_train_fit_features` | `gold_fit_dataframe` | `stage2_feature_columns` |
| `stage1_all_features` | `gold_preprocessed_scaled_dataframe` | `stage1_feature_columns` |
| `stage2_all_features` | `gold_preprocessed_scaled_dataframe` | `stage2_feature_columns` |

Missing feature validation raises `ValueError` before any model is trained.

---

## Stage 1 — Broad Isolation Forest

A single Isolation Forest is trained on `stage1_train_fit_features` using `STAGE1_ESTIMATOR_COUNT` and `RANDOM_SEED`. `compute_anomaly_scores_isolation_forest` returns `-score_samples()` (higher = more anomalous). `choose_threshold_value` converts `STAGE1_THRESHOLD_PERCENTILE` over training scores to an absolute threshold, calibrated on training scores only.

Stage 1 results are attached to `cascade_results` via `build_stage_scoring_frame` → `score_isolation_forest_stage` → `merge_stage_results_back` (same row-tracking pattern as Gold_03a/03b).

`cascade_results` is initialized from `gold_preprocessed_scaled_dataframe.copy()`. Columns added: `stage1_score`, `stage1_decision`, `stage1_pred`, `stage1_flag`.

An additional `plot_order_index` column is created after Stage 1 — derived from `time_index` if available, `meta__row_id` as integer if parseable, or row order as fallback. This column supports debug displays and timeline plots downstream.

Stage 1 scores are also written directly to `cascade_results["stage1_score"]` and `cascade_results["stage1_threshold"]` for alignment tracing, since Stage 1 uses both `build_stage_scoring_frame` and a separate `compute_anomaly_scores_isolation_forest` call for threshold calibration.

---

## Stage 2 — Gold_03b Model Reuse via `run_stage2_selection_decision`

Gold_03c always loads Gold_03b's saved Stage 2 model selection by default. The selection source is hardcoded:

```python
STAGE2_SELECTION_SOURCE = "previous_best"
# "previous_best"     -> reuse 03B's saved best params + threshold percentile
# "configured_search" -> run 03C's configured STAGE2_SELECTION_MODE
# "auto"              -> use previous_best if available, otherwise configured_search
```

Gold_03b's thresholds JSON is loaded from `CASCADE_TUNED_THRESHOLDS_PATH`:

```python
previous_03b_stage2_thresholds = require_mapping(
    load_json(CASCADE_TUNED_THRESHOLDS_PATH),
    "previous_03b_stage2_thresholds",
)
previous_03b_stage2_selected_threshold_percentile = float(
    previous_03b_stage2_thresholds.get("stage2_selected_threshold_percentile", ...)
)
previous_03b_stage2_best_params = dict(previous_03b_stage2_thresholds.get("stage2_best_params", {}))
```

The Stage 2 model is then selected and trained via `run_stage2_selection_decision`:

```python
stage2_model, best_stage2_result, stage2_search_results = run_stage2_selection_decision(
    selection_source=STAGE2_SELECTION_SOURCE,
    previous_best_available=previous_03b_stage2_available,
    previous_best_params=previous_03b_stage2_best_params,
    previous_threshold_percentile=previous_03b_stage2_selected_threshold_percentile,
    selection_mode=STAGE2_SELECTION_MODE,
    fixed_params=STAGE2_FIXED_PARAMS,
    fixed_threshold_percentile=STAGE2_FIXED_THRESHOLD_PERCENTILE,
    threshold_grid=STAGE2_THRESHOLD_GRID,
    param_grid=STAGE2_PARAM_GRID,
    stage2_train_fit_features=stage2_train_fit_features,
    stage2_all_features=stage2_all_features,
    stage1_flags=stage1_flags,
    test_mask=test_mask,
    test_labels=test_labels,
    random_seed=STAGE2_RANDOM_STATE,
    min_recall=STAGE2_MIN_RECALL,
)
```

`run_stage2_selection_decision` is defined inline. It routes to either the `"previous_best"` path (creates a fixed-mode Stage 2 from Gold_03b params) or `"configured_search"` path (runs `evaluate_stage2_model_with_thresholds`). The `"auto"` mode selects `"previous_best"` if Gold_03b params are available and non-empty, otherwise falls back to `"configured_search"`.

`best_stage2_result` records `selection_source`, `selection_mode`, `reused_previous_03b_best` (boolean), `selected_threshold_percentile`, `threshold`, `best_params`, and full metric payload. `stage2_selection_source_used` and `stage2_selection_mode_used` are extracted and logged to the ledger.

### Stage 2 Gate

Stage 2 results are merged using `build_stage_scoring_frame` → `score_isolation_forest_stage` → `merge_stage_results_back` (candidate rows only, masked by `stage2_candidate_mask` where `stage1_flag == 1`). Stage 2 column names are renamed from `score_isolation_forest_stage` defaults to `stage2_model_*` columns to avoid overwriting threshold-calibrated flag logic.

Stage 2 adds: `stage2_model_score`, `stage2_model_decision`, `stage2_model_pred`, `stage2_model_flag`, `stage2_score`, `stage2_raw_flag`, `stage2_flag`. Non-candidate rows receive `NaN` for `stage2_score` to distinguish absent scores from zero scores. The Stage 2 gate enforces: `stage2_flag = 1` only where both `stage1_flag == 1` AND `stage2_raw_flag == 1`.

---

## Stage 3 — Weighted Tunable Confirmation Layer (Improved)

Stage 3 in Gold_03c uses a grid-searched, weighted evidence score rather than simple evidence counts. It is the primary improvement over Gold_03a and Gold_03b.

### Pre-Computation of Evidence Signals

Before the tuning grid runs, four evidence signals are pre-computed on `cascade_results`:

- **`stage3_profile_breach_count`** — via `compute_primary_breach_count`: counts how many primary rule sensors exceed the reference profile's upper bound
- **`stage3_secondary_breach_count`** — via `compute_secondary_breach_count`: counts secondary sensor violations
- Pre-computation of `stage3_persistence_flag` and `stage3_drift_flag` at notebook-level defaults also occurs before the tuning grid, though the tuning grid re-computes these per candidate with varied parameters

`compute_drift_flag` in Gold_03c accepts `rolling_window_size` and `drift_threshold_multiplier` parameters (configurable per candidate), and checks whether rolling delta exceeds `feature_std * drift_threshold_multiplier`.

### Stage 3 Tuning Grid Search

Three operating-mode severity levels are defined:

```python
RELAXED_STAGE3_WEIGHTED_EVIDENCE_SCORE_COMPARISON = 2   # cascade_stage3_relaxed_flag
MEDIUM_STAGE3_WEIGHTED_EVIDENCE_SCORE_COMPARISON  = 3   # cascade_stage3_medium_flag
STRICT_STAGE3_WEIGHTED_EVIDENCE_SCORE_COMPARISON  = 5   # cascade_stage3_strict_flag
```

`build_stage3_candidate_output` is defined inline. For each candidate parameter set from `product(STAGE3_TUNING_GRID values)`, it computes:

1. `profile_breach_flag` = `stage3_profile_breach_count >= STAGE3_MIN_PRIMARY_SENSOR_HITS`
2. `strong_primary_flag` = `stage3_profile_breach_count >= strong_primary_hits` (candidate-specific threshold)
3. `corroboration_flag` = `stage3_secondary_breach_count >= STAGE3_MIN_SECONDARY_SENSOR_HITS`
4. `persistence_flag` = via `compute_persistence_flag` with candidate `rolling_window_size` and `minimum_flags_in_window`
5. `drift_flag` = via `compute_drift_flag` with candidate `drift_threshold_multiplier`
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

A strong primary breach alone is sufficient to confirm a Stage 2 alert, regardless of the weighted score.

`evaluate_stage3_candidate` scores each candidate output against test labels using the same selection score formula as Stage 2:

```python
if recall >= STAGE3_MIN_SELECTION_RECALL:
    selection_score = 3.0 * f1 + 1.0 * precision - 1.0 * alert_rate
else:
    selection_score = -1000.0 + recall   # penalized — fails min_recall floor
```

All candidates are evaluated and sorted by `selection_score` into `stage3_search_results`.

### Best Candidate Selection

The highest-scoring candidate's output is promoted to `cascade_results`:

```python
cascade_results["stage3_profile_breach_flag"]  = stage3_best_output["profile_breach_flag"]
cascade_results["stage3_strong_primary_flag"]  = stage3_best_output["strong_primary_flag"]
cascade_results["stage3_corroboration_flag"]   = stage3_best_output["corroboration_flag"]
cascade_results["stage3_persistence_flag"]     = stage3_best_output["persistence_flag"]
cascade_results["stage3_drift_flag"]           = stage3_best_output["drift_flag"]
cascade_results["stage3_rule_evidence_count"]  = stage3_best_output["rule_evidence_count"]
cascade_results["stage3_weighted_evidence_score"] = stage3_best_output["weighted_evidence_score"]
cascade_results["stage3_weighted_score"]       = cascade_results["stage3_weighted_evidence_score"]
cascade_results["stage3_confirmed_flag"]       = stage3_best_output["confirmed_flag"]
cascade_results["cascade_stage3_improved_flag"] = stage3_best_output["final_flag"]
cascade_results["cascade_final_flag"]          = cascade_results["cascade_stage3_improved_flag"]
```

### Stage 3 Operating Mode Variants

Three additional flag columns are computed using fixed `weighted_evidence_score` thresholds:

```python
cascade_results["cascade_stage3_relaxed_flag"] = ((stage2_flag == 1) & (weighted >= 2.0)).astype(int)
cascade_results["cascade_stage3_medium_flag"]  = ((stage2_flag == 1) & (weighted >= 3.0)).astype(int)
cascade_results["cascade_stage3_strict_flag"]  = ((stage2_flag == 1) & (weighted >= 5.0)).astype(int)
```

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

Three priority levels are assigned: `high` (strong primary breach confirmed by Stage 2), `medium` (Stage 2 confirmed plus weighted confirmation), `low` (Stage 2 confirmed only), `none` (no alert).

`finalize_stage_flag_columns` is called with `stage_names=["stage1", "stage2", "stage3"]` to fill NaN values and enforce integer types across all flag columns.

---

## Stage 3 Operating Mode Metrics

`stage3_operating_mode_metrics` is computed for all three variant modes (relaxed, medium, strict) against test labels. For each mode, `precision_recall_fscore_support` and `roc_auc_score`/`average_precision_score` are computed via a resolver function that looks for the `cascade_stage3_{mode}_flag` column. These metrics are:

- Pushed into `cascade_metrics` before the summary JSON is built
- Stored in `stage3_summary["stage3_operating_mode_metrics"]`
- Written into the four validation contracts (one per mode)

---

## Cascade Output Validation

`validate_cascade_output` runs the same gate checks as Gold_03a and Gold_03b:
1. Required columns present: `meta__row_id`, `meta__is_train_flag`, `stage1_flag`, `stage2_raw_flag`, `stage2_flag`, `cascade_final_flag`
2. `meta__row_id` uniqueness
3. Binary values only in all flag columns
4. Stage 2 gate: no `stage2_flag == 1` where `stage1_flag != 1`
5. Final gate: no `cascade_final_flag == 1` where `stage2_flag != 1`

A separate variant validation cell checks the three operating-mode columns (`cascade_stage3_relaxed_flag`, `cascade_stage3_medium_flag`, `cascade_stage3_strict_flag`) for binary values and Stage 2 gate compliance.

---

## Truth Record Construction and Output Stamping

Truth layer: `"gold_cascade"` (same as Gold_03a/03b — all three cascade notebooks share this layer name; `CASCADE_VARIANT` and `stage3_variant` distinguish the artifact content).

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

**`config_snapshot`** — runtime config snapshot.

**`runtime_facts`** — `cascade_variant`, `stage3_variant` (`"tuned_confirmation_layer"`), alert counts across all 5 operating modes, Stage 2 and Stage 3 search results, `stage2_selection_mode`, `stage2_best_params`, `stage3_selected_params`, feature and rule sensor counts.

**`artifact_paths`** — input paths from Gold_01 truth plus all `cascade_stage3_improved_*` output paths.

`stamp_truth_columns` embeds `meta__truth_hash = CASCADE_TRUTH_HASH`, `meta__parent_truth_hash = GOLD_PARENT_TRUTH_HASH`, `meta__pipeline_mode = PIPELINE_MODE` into every row of `cascade_results`.

---

## Artifact Saves

All outputs are written after truth stamping:

**Parquet/CSV:**
- `cascade_results.to_csv(CASCADE_RESULTS_PATH_CSV, index=False)` — final scored dataset with all stage flags, weighted scores, operating-mode variants, and alert priority
- `cascade_results.to_pickle(CASCADE_RESULTS_PATH_PICKLE)`

**Reference profile:**
- `reference_profile.to_csv(CASCADE_REFERENCE_PROFILE_PATH, index=False)`

**Models (joblib) — Stage 3 is not saved as joblib:**
- `joblib.dump(stage1_model, STAGE1_MODEL_ARTIFACT_PATH)` — Stage 1 broad IF
- `joblib.dump(stage1_model, STAGE1_MODELS_PATH)` — duplicate
- `joblib.dump(stage2_model, STAGE2_MODEL_ARTIFACT_PATH)` — Stage 2 (Gold_03b best params)
- `joblib.dump(stage2_model, STAGE2_MODELS_PATH)` — duplicate

**JSON artifacts:**
- `save_json(cascade_thresholds, CASCADE_THRESHOLDS_PATH)` — all stage thresholds, Stage 3 selected params, evidence weights
- `save_json(cascade_summary, CASCADE_SUMMARY_PATH)` — full cascade summary including Stage 3 variant alert counts, operating mode metrics, `stage3_summary`, truth hashes
- `save_json(cascade_metadata, CASCADE_METADATA_PATH)` — upstream lineage from Gold_01 (`gold_scaler_kind`, `gold_recommended_imputation`, `gold_feature_set_id`) plus Stage 3 selected params and cascade truth hash

All artifacts are uploaded to W&B via `wandb.save`.

---

## Validation Contracts for Gold_06A

Four separate validation contracts are written — one per Stage 3 operating mode — using `build_gold_model_output_validation_contract` and `write_gold_model_output_validation_contract`:

| Contract | `model_id` | `operating_mode` | `flag_column` |
|---|---|---|---|
| Stage 3 Improved | `"stage3_improved"` | `"selected_improved"` | `"cascade_final_flag"` |
| Stage 3 Relaxed | `"stage3_relaxed"` | `"relaxed"` | `"cascade_stage3_relaxed_flag"` |
| Stage 3 Medium | `"stage3_medium"` | `"medium"` | `"cascade_stage3_medium_flag"` |
| Stage 3 Strict | `"stage3_strict"` | `"strict"` | `"cascade_stage3_strict_flag"` |

All four contracts share:
- `source_notebook = "gold_03c_cascade_modeling"`
- `validation_type = "stage3_rule_artifact"`
- `model_stage = "cascade_stage3_improved_final"`
- `stage3_type = "rule_based"`
- `stage3_saved_as_joblib = False`
- `stage1_model_path = STAGE1_MODEL_ARTIFACT_PATH`
- `stage2_model_path = STAGE2_MODEL_ARTIFACT_PATH`
- `output_artifact_path = CASCADE_RESULTS_PATH_CSV`
- `lineage_payload` includes `cascade_truth_hash`, `parent_gold_truth_hash`, `stage3_input_source`
- `rule_config` includes the tuning grid, selected params, operating mode metrics, and three severity threshold comparisons

Each contract is registered with a separate ledger entry keyed by `model_id`.

---

## Detected Row Extracts

Four detected-row frames are built from `cascade_results` via `get_detected_rows_dataframe`:

| Frame | Target Flag | Key Additional Columns |
|---|---|---|
| `stage1_detected_rows_dataframe` | `stage1_flag` | `stage1_score`, `stage1_decision`, `stage1_pred`, all stage flags |
| `stage2_detected_rows_dataframe` | `stage2_flag` | `stage2_score`, `stage2_model_*` columns, all stage flags |
| `stage3_evidence_rows_dataframe` | `stage3_profile_breach_flag` | All Stage 3 evidence columns including `stage3_rule_evidence_count` |
| `final_detected_rows_dataframe` | `cascade_final_flag` | Stage 2 model columns, all Stage 3 evidence columns |

All frames are sorted by `time_index` ascending and logged to the ledger. These frames are not saved to disk.

---

## Ledger Close and W&B Finish

```python
ledger.write_json(cascade_ledger_path)
wandb.save(str(cascade_ledger_path))
wandb_run.finish()
```

W&B is closed **before** final lineage checks and the SQL write. The ledger entry at this step includes `comparison_ready: True`, signaling to Gold_04 that this notebook has produced a comparable final cascade output.

---

## Final Lineage Checks

Seven-step lineage verification runs after `wandb_run.finish()`:

1. Required columns present: `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode`
2. `extract_truth_hash(cascade_results)` must match `CASCADE_TRUTH_HASH`
3. `meta__parent_truth_hash` must have exactly one unique non-null value (multiple hashes indicate mixed upstream runs)
4. That single parent value must match `GOLD_PARENT_TRUTH_HASH`
5. `cascade_truth_path` must exist on disk
6. Loaded truth file hash must match `CASCADE_TRUTH_HASH`
7. Loaded truth file parent hash must match `GOLD_PARENT_TRUTH_HASH`

An additional check reloads `CASCADE_METADATA_PATH` and verifies that the `cascade_truth_hash` embedded in the metadata JSON also matches `CASCADE_TRUTH_HASH`.

---

## SQL Write and Verification

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
        dataset_name=DATASET_NAME,
        model_stage=CASCADE_SQL_MODEL_STAGE,
    )
```

`write_gold_cascade_scores_sql` writes to `gold.anomaly_detection_scores` with `model_stage = "cascade_stage3_improved_final"`. The `WRITE_TO_POSTGRES` gate allows read-only mode.

After the write, two verification read queries run against `gold.anomaly_detection_scores`:

1. **General stage check** — groups all `(model_name, model_stage)` combinations for the current `dataset_id` and `run_id`, returning `row_count` and `alert_count`
2. **Cascade-specific check** — filters to `model_name = 'cascade_isolation_forest_rule_confirmation'` and groups by `model_stage`, returning row count, alert count, and `MIN`/`MAX` of `meta_scored_at_utc`

### Commented-Out DELETE Statement

A commented-out cell contains a `DELETE FROM gold.anomaly_detection_scores WHERE ... AND model_stage = 'cascade_final'` statement. This is not active code. It is preserved as a reference for cleaning stale rows from prior runs that used the old `model_stage = 'cascade_final'` label. No deletion occurs during normal notebook execution.

---

## Key Differences from Gold_03a and Gold_03b

| Aspect | Gold_03a (default) | Gold_03b (tuned) | Gold_03c (stage3_improved) |
|---|---|---|---|
| `CASCADE_VARIANT` | `"default"` | `"tuned"` | `"stage3_improved"` |
| Stage 2 source | Hardcoded fixed | Config-driven search | Always `"previous_best"` — reuses Gold_03b |
| Gold_03b dependency | None | None | Loads `CASCADE_TUNED_THRESHOLDS_PATH` |
| Stage 3 decision | Evidence count >= 2 OR primary breach | Same | Weighted score >= tuned threshold OR strong primary |
| Stage 3 grid search | None | None | `product(STAGE3_TUNING_GRID)` + `STAGE3_MIN_SELECTION_RECALL` |
| Stage 3 operating modes | None | None | relaxed/medium/strict variant flags |
| `alert_priority` column | None | None | `high`, `medium`, `low`, `none` |
| `cascade_final_flag` formula | Stage 2 AND evidence count | Stage 2 AND evidence count | Stage 2 AND weighted confirmation |
| Validation contracts | 1 | 1 | 4 (one per operating mode) |
| SQL `model_stage` | `cascade_default_final` | `cascade_tuned_final` | `cascade_stage3_improved_final` |
| SQL verification queries | None | None | 2 post-write verification reads |
| Artifact filename prefix | `cascade_default_*` | `cascade_tuned_*` | `cascade_stage3_improved_*` |
| `stage3_variant` in truth | Not set | Not set | `"tuned_confirmation_layer"` |
| `comparison_ready` ledger flag | Not present | Not present | `True` |
| `plot_order_index` column | Not present | Not present | Present |
| `stage3_weighted_evidence_score` | Not present | Not present | Present |

---

## Upstream / Downstream Data Flow

**Reads from Gold_01 PreProcessing (via truth record in scaled Parquet):**
- `GOLD_PREPROCESSED_SCALED_DATA_PATH` — primary scored input (all rows)
- `GOLD_PREPROCESSED_DATA_PATH`, `GOLD_FIT_DATA_PATH`, `GOLD_TEST_DATA_PATH`, `GOLD_TRAIN_DATA_PATH` — override from Gold truth artifact paths
- `STAGE1_FEATURES_PATH`, `STAGE2_FEATURES_PATH`, `STAGE3_PRIMARY_PATH`, `STAGE3_SECONDARY_PATH` — override from Gold truth

**Reads from Gold_03b (explicit file load):**
- `CASCADE_TUNED_THRESHOLDS_PATH` — Gold_03b's saved Stage 2 thresholds JSON; consumed for `previous_03b_stage2_selected_threshold_percentile` and `previous_03b_stage2_best_params`

**Writes (consumed by downstream notebooks):**

| Artifact | Path Key | Consumer |
|---|---|---|
| `cascade_results` (CSV) | `cascade_stage3_improved_results_path_csv` | Gold_04 Comparison; Gold_06A (via contracts) |
| `cascade_results` (Pickle) | `cascade_stage3_improved_results_path_pickle` | Gold_04 Comparison |
| Stage 1 model (joblib) | `cascade_stage3_improved_stage1_model_artifact_path` | Gold_06A |
| Stage 2 model (joblib) | `cascade_stage3_improved_stage2_model_artifact_path` | Gold_06A |
| Reference profile (CSV) | `cascade_stage3_improved_reference_profile_path` | Gold_06A |
| Thresholds JSON | `cascade_stage3_improved_thresholds_path` | Gold_04; Gold_06A |
| Summary JSON | `cascade_stage3_improved_summary_path` | Gold_04; Gold_06A |
| Metadata JSON | `cascade_stage3_improved_metadata_path` | Gold_06A |
| 4 validation contracts | `gold_model_validation_contract_path(model_id=...)` | Gold_06A |
| Truth record | `TRUTHS_PATH / gold_cascade / {dataset}__gold_cascade__truth__.json` | Truth index; downstream lineage checks |
| Ledger JSON | `cascade_ledger_path` | Lineage audit |
| SQL rows | `gold.anomaly_detection_scores` (model_stage=`"cascade_stage3_improved_final"`) | Operational layer |

---

## Relationship to Other Notebooks

### Upstream Context

Gold_03c loads:
- Gold_01 scaled Parquet and truth record (8 path overrides)
- Gold_03b thresholds JSON (`CASCADE_TUNED_THRESHOLDS_PATH`) for `STAGE2_SELECTION_SOURCE="previous_best"`
- Gold_03b Stage 2 model (`CASCADE_TUNED_STAGE2_MODEL_PATH`) — applied without retraining
- Gold_03b reference profile (`CASCADE_TUNED_REFERENCE_PROFILE_PATH`) for Stage 3 breach bounds

Gold_03c is the only Gold modeling notebook with a confirmed direct file-level dependency on another Gold modeling notebook. It cannot run before Gold_03b has written its thresholds JSON.

### Downstream Handoff

Gold_03c provides:
- Cascade results CSV/pickle (the final cascade output) and `CASCADE_STAGE3_IMPROVED_TRUTH_HASH` consumed by Gold_04_Comparison
- 4 validation contracts (default + relaxed + medium + strict) consumed by Gold_06A_Test_Replay_Validation
- Stage 1 joblib model, thresholds, summary, and reference profile consumed by Gold_06A
- SQL rows in `gold.anomaly_detection_scores` with `model_stage="cascade_stage3_improved_final"` (preceded by a `DELETE FROM` to clear prior cascade_final rows — confirmed from `sql_touchpoints.json`)

### Pipeline Position

Terminal cascade modeling notebook. Produces the final cascade evaluation output used by both Gold_04 and Gold_06A. Its three operating modes (relaxed, medium, strict) and four validation contracts provide the most granular cascade evaluation in the pipeline.

### Relationship Summary

- Has a confirmed direct file-level dependency on Gold_03b (thresholds JSON, Stage 2 model, reference profile)
- Produces the final cascade results consumed by Gold_04 and all model artifacts consumed by Gold_06A
- The only cascade notebook producing 4 operating modes and 4 corresponding validation contracts
- `GOLD_PARENT_TRUTH_HASH` cross-validated in Gold_04 against Gold_02, Gold_03a, and Gold_03b
- `DELETE FROM` confirms Gold_03c manages its own SQL state before writing
