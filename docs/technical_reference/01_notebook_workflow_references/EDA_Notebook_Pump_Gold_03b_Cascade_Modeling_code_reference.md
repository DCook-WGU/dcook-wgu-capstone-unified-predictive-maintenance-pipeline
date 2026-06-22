# Notebook Workflow Reference: EDA_Notebook_Pump_Gold_03b_Cascade_Modeling

**Source notebook:** `notebooks/experiments/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling.ipynb`
**Reference type:** Workflow-level notebook code reference (071b format)
**Layer:** Gold — Cascade Modeling (Tuned Variant)
**Context:** This notebook trains and evaluates the *tuned* 3-stage cascade anomaly detection pipeline for industrial pump telemetry. It is the second cascade notebook in the Gold layer. The primary distinction from Gold_03a is that Stage 2 configuration is driven by config rather than hardcoded, enabling threshold grid search and parameter search as alternatives to the fixed-mode selection used in Gold_03a.

---

## Notebook Purpose

Gold_03b implements the tuned 3-stage cascade anomaly detection pipeline (`CASCADE_VARIANT = "tuned"`). Stage 2 configuration is config-driven, enabling `run_stage2_selection` with `"fixed"`, `"threshold_grid"`, or `"parameter_search"` modes. The notebook trains Stage 1 and Stage 2 Isolation Forest models, applies rule-based Stage 3 confirmation, writes scored rows to `gold.anomaly_detection_scores` with `model_stage="cascade_tuned_final"`, and produces a validation contract for Gold_06A. Its saved thresholds JSON is consumed by Gold_03c as the `"previous_best"` Stage 2 source.

## Pipeline Role

- Stage: `gold_cascade`; Variant: `"tuned"` (hardcoded); Layer: Gold
- W&B job_type: `gold_modeling_cascade`
- Position in workflow: Second cascade modeling notebook; runs after Gold_03a; before Gold_03c
- Primary responsibility: Full three-stage cascade fit and scoring with configurable multi-candidate Stage 2 selection; produce per-row cascade decisions, per-stage model artifacts, and traceability records
- Gold_03c reads Gold_03b's saved thresholds JSON (`CASCADE_TUNED_THRESHOLDS_PATH`) for `"previous_best"` Stage 2 reuse

## Inputs

| Input | Source | Used For |
|---|---|---|
| Gold_01 scaled Parquet | `GOLD_PREPROCESSED_SCALED_DATA_PATH` | Primary cascade scoring input; parent truth extraction |
| Gold_01 truth record | `GOLD_TRUTH_PATH` (JSON) + `require_mapping` | Parent hash; 8 artifact path overrides; runtime facts |
| Fit Parquet | Truth-overridden `gold_fit_path` | Stage 1/2 model fit (normal-only); reference profile |
| Test / train / preprocessed Parquets | Truth-overridden paths | Partition reference; unscaled reference |
| Stage 1 feature JSON | Truth-overridden `stage1_features_path` | Stage 1 Isolation Forest feature set |
| Stage 2 feature JSON | Truth-overridden `stage2_features_path` | Stage 2 Isolation Forest reduced feature set |
| Stage 3 primary / secondary sensor JSONs | Truth-overridden `stage3_primary/secondary_path` | Stage 3 rule sensor sets |
| `CONFIG`, `RESOLVED_PATHS`, `FILENAMES` | `load_notebook_context()` | All cascade parameters, output paths |

## Configuration and Runtime Context

| Item | Source | Value / Purpose |
|---|---|---|
| `CASCADE_VARIANT` | Hardcoded `"tuned"` | Artifact naming key; distinguishes from Gold_03a/03c |
| `STAGE2_SELECTION_MODE` | `STAGE2_CFG["selection_mode"]` | `"fixed"`, `"threshold_grid"`, or `"parameter_search"` |
| `STAGE2_MIN_RECALL` | Config | Recall floor; candidates below this receive penalized selection score |
| `STAGE2_THRESHOLD_GRID` | `STAGE2_SEARCH_CFG["threshold_grid"]` | Percentile candidates for grid/search modes |
| `STAGE2_PARAM_GRID` | `STAGE2_SEARCH_PARAM_GRID_CFG` | Parameter combinations for `"parameter_search"` mode |
| `STAGE1_ESTIMATOR_COUNT`, `STAGE1_THRESHOLD_PERCENTILE` | `STAGE1_CFG` | Stage 1 IF tree count and threshold calibration percentile |
| `STAGE3_MIN_PRIMARY_SENSOR_HITS`, `STAGE3_MIN_SECONDARY_SENSOR_HITS` | `STAGE3_CFG` | Stage 3 profile breach and corroboration thresholds |
| `STAGE3_ROLLING_WINDOW_SIZE`, `STAGE3_MINIMUM_FLAGS_IN_WINDOW` | `STAGE3_CFG` | Stage 3 persistence window |
| `GOLD_PARENT_TRUTH_HASH` | Extracted from Gold_01 truth record | Parent hash for `gold_cascade` truth chain |
| `DATASET_NAME` | Extracted from Gold_01 truth record | Confirmed from truth, not just config |
| `CAPSTONE_SCHEMA`, `DATASET_ID`, `RUN_ID`, `ASSET_ID` | Env vars / config | SQL write targets and row identity |

## Logical Workflow Map

1. Bootstrap: `load_notebook_context()`, config extraction, artifact dirs (`variant="tuned"`), DB engine, W&B init, SQL smoke check
2. Data load: Gold_01 scaled Parquet → Gold truth record → 8 artifact path overrides → 5 Parquets + 4 JSON lists (`require_str_list`)
3. Row identity: `ensure_stable_row_id(gold_preprocessed_scaled_dataframe, "meta__row_id")`
4. Split recovery: `train_mask` / `test_mask` from `meta__is_train_flag`; optional `test_labels` extraction
5. Reference profile: `build_reference_profile` from normal-only fit data
6. Feature matrices: 4 DataFrames (Stage 1/2 × fit/all-rows); missing feature validation
7. Stage 1: Broad IF fit, score, threshold calibration, row tracking, `cascade_results` initialized
8. Stage 2: `run_stage2_selection` (multi-candidate eval) → final scoring, column rename, Stage 2 gate
9. Stage 3: Primary breach, secondary breach, persistence, drift, evidence count, `cascade_final_flag`; `finalize_stage_flag_columns`
10. Cascade metrics: alert counts; precision/recall/F1 on test window when `test_labels` available
11. `validate_cascade_output`: required columns, binary flags, Stage 2 gate, final gate
12. Truth record: `initialize_layer_truth` → `build_truth_record` → `stamp_truth_columns` → `save_truth_record` → `append_truth_index`
13. Artifact saves: results CSV/pickle, reference profile, 2 joblib models, threshold/summary/metadata JSONs; W&B uploads
14. Validation contract: `build_gold_model_output_validation_contract(model_id="cascade_tuned")` + write
15. Detected row frames: 4 `get_detected_rows_dataframe` extracts (display only)
16. Ledger close: `ledger.write_json`; `wandb_run.finish()`
17. Final lineage checks: 7-step hash invariant verification
18. SQL write: `write_gold_cascade_scores_sql` → `gold.anomaly_detection_scores` (`model_stage="cascade_tuned_final"`)

## Section Overview

| Section | Purpose | Key Outputs |
|---|---|---|
| Bootstrap | Context, dirs, DB, W&B | `CTX`, `GOLD_CASCADE_ARTIFACT_DIRS`, `wandb_run` |
| Data load and Gold truth | Load inputs; inherit parent truth; 8 path overrides | 5 DataFrames, 4 JSON lists, `GOLD_PARENT_TRUTH_HASH` |
| Row identity | `ensure_stable_row_id` | Unique `meta__row_id` on all rows |
| Train/test split | Recover from `meta__is_train_flag` | `train_mask`, `test_mask`, `test_labels` |
| Reference profile | Build normal operating bounds for Stage 3 | `reference_profile` DataFrame |
| Feature matrices | 4 typed DataFrames for Stage 1/2 | `stage1/2_train_fit_features`, `stage1/2_all_features` |
| Stage 1 | Broad IF screening | `stage1_model`, `cascade_results` initialized |
| Stage 2 | Multi-candidate selection and gating | `stage2_model`, best params, `stage2_flag` |
| Stage 3 | Rule-based confirmation | `cascade_final_flag`, evidence columns |
| Cascade metrics + validation | Alert counts; `validate_cascade_output` | `cascade_metrics` |
| Truth record | Build, stamp, save, index | `CASCADE_TRUTH_HASH` |
| Artifact saves | All outputs to disk and W&B | Results CSV/pickle, models, JSONs |
| Validation contract | Contract JSON for Gold_06A | `cascade_tuned_contract` JSON |
| Ledger / W&B close | Finalize run record | Ledger JSONL; W&B closed |
| Final lineage checks | 7-step hash invariant verification | `ValueError` on mismatch |
| SQL write | Scored rows to database | `gold.anomaly_detection_scores` |

## Section Details

Detailed workflow sections follow. See **Notebook Bootstrap**, **Configuration Block**, **Artifact Directory Setup**, **Data Load and Gold Truth**, **Stage 1**, **Stage 2**, **Stage 3**, and subsequent sections.

## Key Function Calls and In-Place Usage

| Function | Context | Return / Side Effect |
|---|---|---|
| `load_notebook_context("gold_cascade", ...)` | Bootstrap | `CTX` with all shared context aliases |
| `build_artifact_dirs_from_config(config, stage_key="gold_cascade", variant="tuned")` | Artifact dirs | `GOLD_CASCADE_ARTIFACT_DIRS` dict |
| `require_mapping(load_json(GOLD_TRUTH_PATH))` | Load Gold truth | `gold_truth` dict; raises on empty/non-dict |
| `require_str_list(load_json(path), name)` | Load feature/sensor JSON lists | Validated list; `TypeError`/`ValueError` on malformed |
| `ensure_stable_row_id(df, "meta__row_id")` | Row identity | `meta__row_id` stamped and unique |
| `build_reference_profile(gold_fit_dataframe, feature_columns)` | Stage 3 bounds | `reference_profile` DataFrame (5th/95th pct per feature) |
| `evaluate_stage2_model_with_thresholds(...)` | Stage 2 candidate eval | Result dict with `selection_score`, metrics |
| `run_stage2_selection(selection_mode, ...)` | Stage 2 selection | Best model, best result, `stage2_search_results` list |
| `compute_primary_breach_count`, `compute_secondary_breach_count` | Stage 3 rule | Breach count Series |
| `compute_persistence_flag`, `compute_drift_flag` | Stage 3 rule | Binary flag Series |
| `finalize_stage_flag_columns(cascade_results, stage_names)` | Fill NaN | All flag columns as integers |
| `validate_cascade_output(cascade_results, test_mask)` | Integrity | `ValueError` on gate violation or structural failure |
| `build_truth_record(...)` / `stamp_truth_columns(...)` / `save_truth_record(...)` | Truth chain | `CASCADE_TRUTH_HASH`; stamped `cascade_results`; truth JSON |
| `build_gold_model_output_validation_contract(model_id="cascade_tuned", ...)` | Gold_06A contract | Contract dict |
| `write_gold_cascade_scores_sql(engine, ..., model_stage="cascade_tuned_final")` | SQL write | Rows in `gold.anomaly_detection_scores` |

## Outputs and Artifacts

| Output | Type | Location | Downstream Consumer |
|---|---|---|---|
| Cascade results | CSV + Pickle | `CASCADE_RESULTS_PATH_CSV/PICKLE` | Gold_03c; Gold_06A |
| Stage 1 model | Joblib | `STAGE1_MODEL_ARTIFACT_PATH` | Gold_06A |
| Stage 2 model | Joblib | `STAGE2_MODEL_ARTIFACT_PATH` | Gold_06A |
| Reference profile | CSV | `CASCADE_REFERENCE_PROFILE_PATH` | Gold_06A |
| Cascade thresholds | JSON | `CASCADE_THRESHOLDS_PATH` | Gold_03c (`previous_best`); Gold_06A |
| Cascade summary | JSON | `CASCADE_SUMMARY_PATH` | Gold_06A |
| Cascade metadata | JSON | `CASCADE_METADATA_PATH` | Gold_06A |
| Validation contract | JSON | `cascade_tuned_contract_path` | Gold_06A |
| `gold_cascade` truth record | JSON | `TRUTHS_PATH/gold_cascade/` | Truth index; lineage checks |
| Cascade ledger | JSONL | `cascade_ledger_path` | Lineage audit |
| SQL rows | `gold.anomaly_detection_scores` (model_stage=`"cascade_tuned_final"`) | Operational layer |

## Data Quality / Validation Behavior

| Check | Purpose | Failure / Risk Prevented |
|---|---|---|
| `require_mapping` on Gold truth record | Gold_01 truth must be a non-empty dict | `ValueError` before any cascade begins |
| `require_str_list` on 4 JSON lists | Feature and sensor lists must be non-empty string lists | `TypeError`/`ValueError` before model fit |
| Missing feature validation (4 frame checks) | All Stage 1/2 feature columns present in both DataFrames | `ValueError` with missing column list |
| `STAGE2_MIN_RECALL` floor | Penalize Stage 2 candidates below recall floor | Prevents low-recall candidate from winning selection |
| `validate_cascade_output` | Required columns, binary flags, gate integrity | `ValueError` on structural or gate violation |
| Final 7-step lineage invariants | `meta__truth_hash` roundtrip, parent hash uniqueness, truth file re-read | `ValueError` on any mismatch |

## Downstream Handoff

Gold_03c reads `CASCADE_TUNED_THRESHOLDS_PATH` (saved thresholds JSON) to extract `stage2_selected_threshold_percentile` and `stage2_best_params`, using them as the `"previous_best"` Stage 2 source via `run_stage2_selection_decision(selection_source="previous_best", ...)`.

Gold_06A reads the validation contract, Stage 1/Stage 2 joblib models, thresholds, summary, reference profile, and results CSV to replay the tuned cascade against the held-out test set.

`gold.anomaly_detection_scores` receives cascade-scored rows with `model_stage="cascade_tuned_final"`.

---

## Notebook at a Glance

| Property | Value |
|---|---|
| Cell count | 144 total (55 code, 89 markdown) |
| CASCADE_VARIANT | `"tuned"` (hardcoded) |
| STAGE2_SELECTION_MODE | Read from config — supports `"fixed"`, `"threshold_grid"`, or `"parameter_search"` |
| Parent truth source | Gold_01 PreProcessing truth record |
| Output truth layer | `"gold_cascade"` |
| W&B job_type | `gold_modeling_cascade` |
| Downstream consumer | Gold_03c Cascade Modeling; Gold_06A Test Replay Validation (via validation contract) |

---

## Imports and Library Setup

The notebook imports standard scientific Python (`numpy`, `pandas`, `sklearn`, `joblib`, `wandb`) alongside the full project utility stack. Key utility imports:

- `load_data`, `save_data`, `save_json`, `load_json` — from `utils.core.file_io`
- `load_pipeline_config`, `build_truth_config_block`, `export_config_snapshot` — from `utils.core.config_loader`
- `load_notebook_context` — from the notebook context bootstrap module
- Truth utilities: `initialize_layer_truth`, `update_truth_section`, `build_truth_record`, `save_truth_record`, `append_truth_index`, `stamp_truth_columns`, `extract_truth_hash`, `identify_meta_columns`, `identify_feature_columns`, `make_process_run_id`, `load_json` — from `utils.core.truths`
- Gold modeling utilities: `build_artifact_dirs_from_config`, `build_reference_profile`, `build_stage_scoring_frame`, `score_isolation_forest_stage`, `merge_stage_results_back`, `ensure_stable_row_id`, `compute_anomaly_scores_isolation_forest`, `choose_threshold_value`, `finalize_stage_flag_columns`, `get_detected_rows_dataframe`, `compute_primary_breach_count`, `compute_secondary_breach_count`, `compute_persistence_flag`, `compute_drift_flag` — from Gold modeling utilities
- `write_gold_cascade_scores_sql` — SQL write utility for `gold.anomaly_detection_scores`
- `build_gold_model_output_validation_contract`, `write_gold_model_output_validation_contract`, `gold_model_validation_contract_path` — validation contract utilities for Gold_06A consumption
- `require_str_list`, `require_mapping` — strict validation wrappers
- `ParameterGrid` — from `sklearn.model_selection` (used in `"parameter_search"` mode)
- `precision_recall_fscore_support` — from `sklearn.metrics`

---

## Notebook Bootstrap

`load_notebook_context()` is called once to initialize paths, config, logger, ledger, and database engine into a `CTX` context object:

```python
CTX = load_notebook_context(
    stage="gold_cascade",
    logger_child_name="capstone.gold.cascade.tuned",
    log_filename="gold_modeling_cascade_tuned.log",
)
```

Key values unpacked from `CTX`: `paths`, `config` (`CONFIG`), `logger`, `ledger`, `engine`, `CAPSTONE_SCHEMA`, `DATASET_ID`, `RUN_ID`, `PIPELINE_MODE`, `TRUTH_VERSION`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME`, `TRUTHS_PATH`, `TRUTH_INDEX_PATH`.

`DATASET_NAME` and `STAGE` are resolved from config after bootstrap. `STAGE` is set to `"gold_cascade"`.

---

## Configuration Block

Cascade variant and Stage 2 selection configuration are read from config in a single block:

```python
CASCADE_VARIANT = "tuned"   # hardcoded — determines filenames, artifact path keys, and truth layer

STAGE2_CFG = CONFIG["cascade"]["stage2"]
STAGE2_SEARCH_CFG = CONFIG["cascade"]["stage2_search"]
STAGE2_SEARCH_PARAM_GRID_CFG = CONFIG["cascade"]["stage2_param_grid"]

STAGE2_SELECTION_MODE = str(STAGE2_CFG["selection_mode"]).strip().lower()
STAGE2_MIN_RECALL = float(STAGE2_CFG["min_recall"])
STAGE2_FIXED_THRESHOLD_PERCENTILE = float(STAGE2_CFG["fixed_threshold_percentile"])
STAGE2_FIXED_PARAMS = dict(STAGE2_CFG["fixed_params"])
STAGE2_THRESHOLD_GRID = [float(v) for v in STAGE2_SEARCH_CFG["threshold_grid"]]
STAGE2_PARAM_GRID = {str(k): list(v) for k, v in STAGE2_SEARCH_PARAM_GRID_CFG.items()}
```

`STAGE2_SELECTION_MODE` controls the breadth of Stage 2 search:
- `"fixed"` — single candidate evaluated at a single threshold
- `"threshold_grid"` — single model candidate evaluated across all `STAGE2_THRESHOLD_GRID` percentiles
- `"parameter_search"` — all combinations from `STAGE2_PARAM_GRID` evaluated across all `STAGE2_THRESHOLD_GRID` percentiles

`STAGE2_MIN_RECALL` is a floor constraint: any Stage 2 candidate whose recall on the test window falls below this value receives a penalized selection score (`-1000.0 + recall`) and cannot win against a compliant candidate.

Additional Stage 1 and Stage 3 config values read from config:
- `STAGE1_ESTIMATOR_COUNT`, `STAGE1_THRESHOLD_PERCENTILE`, `STAGE1_RANDOM_STATE`, `STAGE1_CONTAMINATION`
- `STAGE3_MIN_PRIMARY_SENSOR_HITS`, `STAGE3_MIN_SECONDARY_SENSOR_HITS`, `STAGE3_ROLLING_WINDOW_SIZE`, `STAGE3_MINIMUM_FLAGS_IN_WINDOW`

`CONTEXT_LOG_FILE = "gold_modeling_cascade_tuned.log"` — ledger and log file name for the tuned variant.

---

## Artifact Directory Setup

`build_artifact_dirs_from_config` creates the cascade-specific artifact subdirectory layout:

```python
GOLD_CASCADE_ARTIFACT_DIRS = build_artifact_dirs_from_config(
    config=CONFIG,
    stage_key="gold_cascade",
    variant=CASCADE_VARIANT,   # "tuned"
)
```

All output paths for this notebook are resolved from `GOLD_CASCADE_ARTIFACT_DIRS`, using `FILENAMES` keys with the `cascade_tuned_*` prefix:

| Variable | Artifact |
|---|---|
| `CASCADE_RESULTS_PATH_CSV` | `scores/cascade_tuned_results_*.csv` |
| `CASCADE_RESULTS_PATH_PICKLE` | `scores/cascade_tuned_results_*.pkl` |
| `CASCADE_THRESHOLDS_PATH` | `thresholds/cascade_tuned_thresholds_*.json` |
| `CASCADE_SUMMARY_PATH` | `summaries/cascade_tuned_summary_*.json` |
| `CASCADE_METADATA_PATH` | `metadata/cascade_tuned_metadata_*.json` |
| `CASCADE_REFERENCE_PROFILE_PATH` | `profiles/cascade_tuned_reference_profile_*.csv` |
| `STAGE1_MODEL_ARTIFACT_PATH` | `models/cascade_tuned_stage1_model_*.joblib` |
| `STAGE2_MODEL_ARTIFACT_PATH` | `models/cascade_tuned_stage2_model_*.joblib` |
| `cascade_ledger_path` | `lineage/gold_cascade_tuned_ledger_*.json` |
| `CONFIG_SNAPSHOT_PATH` | `config/{dataset}__gold_cascade_tuned__resolved_config.yaml` |

A config snapshot is exported immediately if `CONFIG["execution"]["save_config_snapshot"]` is `True`.

`GOLD_TRUTH_PATH` and `GOLD_PREPROCESSED_SCALED_DATA_PATH` are read from `RESOLVED_PATHS` using the `cascade_tuned_*` key convention.

---

## SQL Smoke Check

After artifact directories are created, a lightweight connectivity check confirms that the database engine and schema are reachable before any long-running computation begins. This smoke check does not execute any writes.

---

## Weights & Biases Initialization

W&B is initialized with `job_type="gold_modeling_cascade"` — same as Gold_03a. The run config captures all cascade-specific hyperparameters at run time:

```python
wandb_run = wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name=WANDB_RUN_NAME,
    job_type="gold_modeling_cascade",
    config={
        "gold_version": GOLD_VERSION,
        "dataset": DATASET_NAME,
        "stage": STAGE,
        "stage1_threshold_percentile": STAGE1_THRESHOLD_PERCENTILE,
        "stage2_selection_mode": STAGE2_SELECTION_MODE,
        "stage2_fixed_threshold_percentile": STAGE2_FIXED_THRESHOLD_PERCENTILE,
        "stage2_search_threshold_grid": STAGE2_THRESHOLD_GRID,
        "stage3_min_primary_sensor_hits": STAGE3_MIN_PRIMARY_SENSOR_HITS,
        "stage3_min_secondary_sensor_hits": STAGE3_MIN_SECONDARY_SENSOR_HITS,
        ...
    },
)
```

---

## Data Load — Gold Scaled Input + Parent Truth Resolution

The Gold_01 PreProcessing truth record is the authoritative source for this notebook's input data paths. The load sequence is:

1. Load `gold_preprocessed_scaled_dataframe` from `GOLD_PREPROCESSED_SCALED_DATA_PATH` via `load_data`.
2. Load the Gold_01 truth record from `GOLD_TRUTH_PATH` via `load_json` + `require_mapping`.
3. Extract `GOLD_PARENT_TRUTH_HASH` and `DATASET_NAME` from the truth record.
4. Override eight input paths from the truth record's `artifact_paths` section:

```python
GOLD_PREPROCESSED_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_preprocessed_path", str(GOLD_PREPROCESSED_DATA_PATH)))
GOLD_FIT_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_fit_path", str(GOLD_FIT_DATA_PATH)))
GOLD_TEST_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_test_path", str(GOLD_TEST_DATA_PATH)))
GOLD_TRAIN_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_train_path", str(GOLD_TRAIN_DATA_PATH)))
STAGE1_FEATURES_PATH = Path(gold_truth_artifact_paths.get("stage1_features_path", str(STAGE1_FEATURES_PATH)))
STAGE2_FEATURES_PATH = Path(gold_truth_artifact_paths.get("stage2_features_path", str(STAGE2_FEATURES_PATH)))
STAGE3_PRIMARY_PATH = Path(gold_truth_artifact_paths.get("stage3_primary_path", str(STAGE3_PRIMARY_PATH)))
STAGE3_SECONDARY_PATH = Path(gold_truth_artifact_paths.get("stage3_secondary_path", str(STAGE3_SECONDARY_PATH)))
```

Each path falls back to the config-resolved value if the truth record does not contain the key.

5. Load four Parquets via `load_data`: `gold_preprocessed_dataframe`, `gold_fit_dataframe`, `gold_test_dataframe`, `gold_train_dataframe`.
6. Load four JSON lists using `require_str_list(load_json(path), name)`:
   - `stage1_feature_columns` from `STAGE1_FEATURES_PATH`
   - `stage2_feature_columns` from `STAGE2_FEATURES_PATH`
   - `stage3_primary_rule_sensors` from `STAGE3_PRIMARY_PATH`
   - `stage3_secondary_rule_sensors` from `STAGE3_SECONDARY_PATH`

`require_str_list` raises `TypeError` if the loaded JSON is not a flat list of strings — this enforces that feature and sensor lists are not silently misread as nested structures.

---

## Row Identity Stabilization

Before any scoring begins, `ensure_stable_row_id` is called on `gold_preprocessed_scaled_dataframe`:

```python
gold_preprocessed_scaled_dataframe = ensure_stable_row_id(
    gold_preprocessed_scaled_dataframe,
    row_id_column="meta__row_id",
)
```

This guarantees that `meta__row_id` is populated and unique across all rows. All subsequent stage scoring steps use `meta__row_id` as the join key when merging stage results back into `cascade_results`.

---

## Train/Test Split Recovery

The train/test partition is recovered from the `meta__is_train_flag` column already embedded in `gold_preprocessed_scaled_dataframe` — no re-splitting occurs:

```python
train_mask = gold_preprocessed_scaled_dataframe["meta__is_train_flag"] == 1
test_mask = gold_preprocessed_scaled_dataframe["meta__is_train_flag"] == 0
```

`anomaly_flag` is extracted from `gold_preprocessed_scaled_dataframe` for the test window rows only, if present. When available, it populates `test_labels` and enables precision/recall/F1 computation in Stage 2 candidate evaluation. When absent, `test_labels = None` and Stage 2 selection falls back to alert rate minimization only.

---

## Reference Profile Construction

A normal-operating-bounds profile is built from the fit (normal-only) subset before cascade scoring begins. This profile is used exclusively in Stage 3 rule evaluation:

```python
reference_profile_features = list(dict.fromkeys(
    stage1_feature_columns + stage3_primary_rule_sensors + stage3_secondary_rule_sensors
))

reference_profile = build_reference_profile(
    gold_fit_dataframe,
    feature_columns=reference_profile_features,
)
```

`build_reference_profile` is defined inline in the notebook. For each feature it records: `median_value`, `mean_value`, `standard_deviation`, `lower_bound` (5th percentile), `upper_bound` (95th percentile). The union of Stage 1, Stage 3 primary, and Stage 3 secondary sensors is profiled. Building exclusively from `gold_fit_dataframe` prevents anomalous test-window values from contaminating the normal operating bounds.

---

## Feature Matrix Assembly

Four typed DataFrames (not NumPy arrays) are constructed from the loaded data. Keeping DataFrames rather than `.values` avoids sklearn feature-name warnings when fitting Isolation Forest:

| Variable | Source DataFrame | Columns |
|---|---|---|
| `stage1_train_fit_features` | `gold_fit_dataframe` | `stage1_feature_columns` |
| `stage2_train_fit_features` | `gold_fit_dataframe` | `stage2_feature_columns` |
| `stage1_all_features` | `gold_preprocessed_scaled_dataframe` | `stage1_feature_columns` |
| `stage2_all_features` | `gold_preprocessed_scaled_dataframe` | `stage2_feature_columns` |

Missing feature validation raises `ValueError` before any model is trained — one check for each of the four combinations.

A candidate Stage 2 mask is derived from Stage 1 flags to restrict Stage 2 fitting to Stage 1-flagged rows:

```python
stage2_candidate_mask = stage1_flags == 1
```

---

## Stage 1 — Broad Isolation Forest

A single Isolation Forest is trained on the normal-only fit data (`stage1_train_fit_features`) using `STAGE1_ESTIMATOR_COUNT`, `STAGE1_RANDOM_STATE`, and `STAGE1_CONTAMINATION` from config.

Scoring uses `compute_anomaly_scores_isolation_forest`, which returns `-score_samples()` so that higher values indicate stronger anomaly evidence (project convention). `choose_threshold_value` converts `STAGE1_THRESHOLD_PERCENTILE` over the scored fit population into an absolute threshold.

Stage 1 results are attached to a copy of `gold_preprocessed_scaled_dataframe` via `build_stage_scoring_frame` and `score_isolation_forest_stage`, then merged back using `merge_stage_results_back` with `meta__row_id` as the join key:

```python
stage1_input_df = build_stage_scoring_frame(...)
stage1_results_df = score_isolation_forest_stage(...)

cascade_results = merge_stage_results_back(
    master_dataframe=gold_preprocessed_scaled_dataframe.copy(),
    stage_results_dataframe=stage1_results_df,
    stage_name="stage1",
    row_id_column="meta__row_id",
)
```

`cascade_results` becomes the accumulating output frame; the original scaled dataframe is not mutated.

Stage 1 adds these columns to `cascade_results`:
- `stage1_score` — anomaly score (`-score_samples()`)
- `stage1_decision` — threshold decision string
- `stage1_pred` — raw Isolation Forest prediction (−1/+1)
- `stage1_flag` — binary flag (1 = alert, 0 = normal)

Stage 1 row tracking is logged to the ledger immediately after scoring. `get_detected_rows_dataframe` extracts Stage 1 flagged rows into a separate frame for post-hoc review.

---

## Stage 2 — Tuned Narrow Isolation Forest (Multi-Candidate Selection)

Stage 2 uses the `run_stage2_selection` function defined inline in the notebook to evaluate one or more candidate models across one or more threshold percentiles, selecting the best by a weighted selection score.

### Selection Function: `run_stage2_selection`

```python
stage2_model, best_stage2_result, stage2_search_results = run_stage2_selection(
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
stage2_selected_threshold_percentile = best_stage2_result["selected_threshold_percentile"]
stage2_best_params = best_stage2_result["model_params"]
```

**`run_stage2_selection` logic by mode:**
- `"fixed"`: evaluates a single candidate (`fixed_params`) at a single threshold (`fixed_threshold_percentile`)
- `"threshold_grid"`: evaluates a single candidate across all `STAGE2_THRESHOLD_GRID` percentiles
- `"parameter_search"`: evaluates all `ParameterGrid(STAGE2_PARAM_GRID)` combinations across all `STAGE2_THRESHOLD_GRID` percentiles

Each candidate is evaluated by `evaluate_stage2_model_with_thresholds`.

### Evaluation Function: `evaluate_stage2_model_with_thresholds`

For each threshold percentile, this function:
1. Fits the Stage 2 model on Stage 1 candidate rows from the fit population (`stage2_train_fit_features`, masked by `stage2_candidate_mask`)
2. Scores all rows using `stage2_all_features`
3. Applies the threshold to produce `stage2_raw_flags`
4. Gates Stage 2 flags: `stage2_flags = (stage1_flags == 1) & (stage2_raw_flags == 1)` — Stage 2 can only confirm Stage 1 candidates, not introduce new alerts
5. Computes `precision`, `recall`, `f1`, and `alert_rate` on the test window (if `test_labels` is available)
6. Computes `selection_score`:

```python
if recall >= min_recall:
    selection_score = 3.0 * f1 + 1.0 * precision - 1.0 * alert_rate
else:
    selection_score = -1000.0 + recall   # penalized — fails min_recall floor
```

`STAGE2_MIN_RECALL` prevents an overly strict candidate from winning by achieving high precision at the expense of recall. Any candidate falling below this floor cannot outcompete a compliant candidate regardless of F1 or precision. When `test_labels` is `None`, selection falls back to alert rate minimization.

`run_stage2_selection` returns a `search_results` list (sorted by `selection_score` descending), the best model object, and the best result dict. `stage2_search_results` is logged to the ledger and displayed for post-hoc review.

### Stage 2 Scoring

After selection, the winning model is used to produce final Stage 2 columns. `stage2_threshold` is computed via `choose_threshold_value` at `stage2_selected_threshold_percentile`.

Stage 2 results are scored across the full population using `build_stage_scoring_frame` → `score_isolation_forest_stage` → `merge_stage_results_back` (same pattern as Stage 1). Stage 2 adds:
- `stage2_score` — anomaly score from selected model
- `stage2_model_score` — raw `score_samples()` preserved separately
- `stage2_model_decision`, `stage2_model_pred`, `stage2_model_flag` — per-row model outputs
- `stage2_raw_flag` — threshold binary (1 = above threshold)
- `stage2_flag` — gated binary: `1` only where both `stage1_flag==1` AND `stage2_raw_flag==1`

Stage 2 row tracking is logged to the ledger. Non-candidate rows (where `stage1_flag != 1`) receive NaN for Stage 2-specific columns to make the boundary explicit.

The Stage 2 selection summary — `stage2_best_params`, `stage2_selected_threshold_percentile`, `STAGE2_SELECTION_MODE`, and `stage2_search_candidate_count` — is logged to the ledger and displayed for review.

---

## Stage 3 — Rule-Based Confirmation

Stage 3 is rule-based and uses the reference profile built from fit data only. It produces four independent evidence signals from `cascade_results`, then combines them into a final cascade decision. Stage 3 is not saved as a joblib model.

A sanity check confirms that all Stage 3 primary and secondary rule sensors are present in `cascade_results` before any Stage 3 computation begins.

### Stage 3 Evidence Signals

**Primary breach count** — how many primary rule sensors exceed the reference profile's upper bound:
```python
cascade_results["stage3_profile_breach_count"] = compute_primary_breach_count(
    cascade_results, stage3_primary_rule_sensors, reference_profile,
)
cascade_results["stage3_profile_breach_flag"] = (
    cascade_results["stage3_profile_breach_count"] >= STAGE3_MIN_PRIMARY_SENSOR_HITS
).astype(int)
```

**Secondary breach count** — how many secondary rule sensors exceed the reference profile's upper bound:
```python
cascade_results["stage3_secondary_breach_count"] = compute_secondary_breach_count(
    cascade_results, stage3_secondary_rule_sensors, reference_profile,
)
cascade_results["stage3_corroboration_flag"] = (
    cascade_results["stage3_secondary_breach_count"] >= STAGE3_MIN_SECONDARY_SENSOR_HITS
).astype(int)
```

**Persistence flag** — rolling window check on `stage2_flag`:
```python
cascade_results["stage3_persistence_flag"] = compute_persistence_flag(
    cascade_results["stage2_flag"],
    window_size=STAGE3_ROLLING_WINDOW_SIZE,
    min_flags_in_window=STAGE3_MINIMUM_FLAGS_IN_WINDOW,
)
```

**Drift flag** — union-sensor drift check across both primary and secondary sensors:
```python
union_sensors = list(dict.fromkeys(stage3_primary_rule_sensors + stage3_secondary_rule_sensors))
cascade_results["stage3_drift_flag"] = compute_drift_flag(
    cascade_results, union_sensors, reference_profile,
)
```

### Stage 3 Evidence Count and Final Decision

All four evidence signals are summed into `stage3_rule_evidence_count`:

```python
cascade_results["stage3_rule_evidence_count"] = (
    cascade_results["stage3_profile_breach_flag"]
    + cascade_results["stage3_corroboration_flag"]
    + cascade_results["stage3_persistence_flag"]
    + cascade_results["stage3_drift_flag"]
)
```

The final cascade decision is a Stage 2-gated union of Stage 3 evidence:

```python
cascade_results["cascade_final_flag"] = (
    (cascade_results["stage2_flag"] == 1)
    & (
        (cascade_results["stage3_rule_evidence_count"] >= 2)
        | (cascade_results["stage3_profile_breach_flag"] == 1)
    )
).astype(int)
```

This formula requires Stage 2 confirmation plus either: at least two independent Stage 3 evidence signals, or a primary sensor breach alone. `finalize_stage_flag_columns` is called after the assignment to enforce consistent integer types across all flag columns.

---

## Cascade Metrics Computation

After all stage flags are assigned, metrics are computed on the test window:

```python
cascade_metrics = {
    "model": "3-Stage Cascade",
    "stage1_alert_count_all_rows": ...,
    "stage2_alert_count_all_rows": ...,
    "final_alert_count_all_rows": ...,
    "stage1_alert_count_test_rows": ...,
    "stage2_alert_count_test_rows": ...,
    "final_alert_count_test_rows": ...,
}
```

When `test_labels` is available, `precision_recall_fscore_support` (binary average) is used to compute `precision`, `recall`, and `f1` on `cascade_final_flag` vs. `test_labels`. These values are appended to `cascade_metrics`.

---

## Cascade Output Validation

`validate_cascade_output` runs gate checks on the final `cascade_results` frame:

1. Required columns check: `meta__row_id`, `meta__is_train_flag`, `stage1_flag`, `stage2_raw_flag`, `stage2_flag`, `cascade_final_flag`
2. `meta__row_id` uniqueness check
3. Binary value check: all flag columns must contain only `{0, 1}` (NaN-dropped)
4. Stage 2 gate check: no row where `stage2_flag == 1` and `stage1_flag != 1`
5. Final cascade gate check: no row where `cascade_final_flag == 1` and `stage2_flag != 1`

The validation summary is added to the ledger.

---

## Truth Record Construction and Output Stamping

The truth record for this notebook uses `layer_name = "gold_cascade"` (same as Gold_03a — both cascade notebooks share this layer name; `CASCADE_VARIANT` distinguishes the artifact filenames rather than the truth layer):

```python
cascade_truth_layer_name = "gold_cascade"

cascade_truth = initialize_layer_truth(
    truth_version=TRUTH_VERSION,
    dataset_name=DATASET_NAME,
    layer_name=cascade_truth_layer_name,
    process_run_id=cascade_process_run_id,
    pipeline_mode=PIPELINE_MODE,
    parent_truth_hash=GOLD_PARENT_TRUTH_HASH,
)
```

`GOLD_PARENT_TRUTH_HASH` is the truth hash from Gold_01 PreProcessing — read from `gold_truth` loaded earlier.

The truth base is then updated in three passes:

**`config_snapshot`** — a snapshot of `TRUTH_CONFIG` or a manually constructed runtime dict.

**`runtime_facts`** — final selected thresholds and search results:
- `cascade_variant`, `stage1_threshold_percentile`, `stage1_threshold`
- `stage2_selection_mode`, `stage2_selected_threshold_percentile`, `stage2_threshold`, `stage2_best_params`
- `stage1_estimator_count`, `stage2_estimator_count` (from `stage2_model.get_params()`)
- `stage2_search_candidate_count`
- Feature counts, rule sensor counts, result row count
- `parent_truth_hash`, `gold_process_run_id`, `gold_feature_set_id` (from Gold_01 truth)

**`artifact_paths`** — all input and output artifact paths, prefixed with `cascade_tuned_`:
- Input paths: `gold_truth_path`, `gold_fit_path`, `gold_scored_path`, `stage1_features_path`, `stage2_features_path`, `stage3_primary_path`, `stage3_secondary_path`
- Output paths: `cascade_tuned_results_path_csv`, `cascade_tuned_results_path_pickle`, `cascade_tuned_stage1_model_artifact_path`, `cascade_tuned_stage2_model_artifact_path`, `cascade_tuned_thresholds_path`, `cascade_tuned_summary_path`, `cascade_tuned_metadata_path`, `cascade_tuned_reference_profile_path`

`build_truth_record` finalizes and hashes the truth base. `CASCADE_TRUTH_HASH` is set from the result.

`stamp_truth_columns` embeds lineage columns into `cascade_results`:
- `meta__truth_hash` = `CASCADE_TRUTH_HASH`
- `meta__parent_truth_hash` = `GOLD_PARENT_TRUTH_HASH`
- `meta__pipeline_mode` = `PIPELINE_MODE`

`save_truth_record` writes the truth file to `TRUTHS_PATH`. `append_truth_index` registers it in `TRUTH_INDEX_PATH`.

---

## Artifact Saves

All primary outputs are written after the truth record is stamped and the validation contract is written:

**Parquet/CSV results:**
- `cascade_results.to_csv(CASCADE_RESULTS_PATH_CSV, index=False)` — full scored dataset including all stage flags
- `cascade_results.to_pickle(CASCADE_RESULTS_PATH_PICKLE)` — Pickle format for downstream notebooks

**Reference profile:**
- `reference_profile.to_csv(CASCADE_REFERENCE_PROFILE_PATH, index=False)`

**Models (joblib):**
- `joblib.dump(stage1_model, STAGE1_MODEL_ARTIFACT_PATH)` — Stage 1 broad IF
- `joblib.dump(stage2_model, STAGE2_MODEL_ARTIFACT_PATH)` — Stage 2 narrow IF (selected by `run_stage2_selection`)
- Note: Stage 3 is rule-based and is **not** saved as joblib

**JSON artifacts:**
- `save_json(cascade_thresholds, CASCADE_THRESHOLDS_PATH)` — `cascade_variant`, selected threshold percentiles and values, `stage2_best_params`, `stage2_selection_mode`
- `save_json(cascade_summary, CASCADE_SUMMARY_PATH)` — alert counts, metrics, feature counts, search metadata, truth hashes
- `save_json(cascade_metadata, CASCADE_METADATA_PATH)` — upstream/downstream linkage including `gold_truth_hash`, `gold_process_run_id`, `gold_feature_set_id`, `gold_scaler_path`, `gold_scaler_kind`, `gold_recommended_imputation`, `cascade_truth_hash`, `cascade_process_run_id`

All artifacts are uploaded to W&B via `wandb.save`.

---

## Validation Contract for Gold_06A

A validation contract is written for downstream consumption by Gold_06A Test Replay Validation:

```python
cascade_tuned_contract = build_gold_model_output_validation_contract(
    dataset_id=DATASET_ID,
    run_id=RUN_ID,
    model_id="cascade_tuned",
    model_label="Cascade Tuned",
    source_notebook="gold_03b_cascade_modeling",
    validation_type="cascade_rule_artifact",
    model_stage="cascade_tuned_final",
    operating_mode="tuned",
    metrics=cascade_metrics,
    output_dataframe=cascade_results,
    output_flag_column="cascade_final_flag",
    test_mask=test_mask,
    rule_config={
        "cascade_variant": CASCADE_VARIANT,
        "stage1_threshold_percentile": float(STAGE1_THRESHOLD_PERCENTILE),
        "stage1_threshold": float(stage1_threshold),
        "stage2_selection_mode": STAGE2_SELECTION_MODE,
        "stage2_selected_threshold_percentile": stage2_percentile_for_contract,
        "stage2_threshold": float(stage2_threshold),
        "stage2_best_params": stage2_best_params,
        "stage2_search_candidate_count": int(len(stage2_search_results)),
        "stage3_min_primary_sensor_hits": int(STAGE3_MIN_PRIMARY_SENSOR_HITS),
        "stage3_min_secondary_sensor_hits": int(STAGE3_MIN_SECONDARY_SENSOR_HITS),
        "stage3_rolling_window_size": int(STAGE3_ROLLING_WINDOW_SIZE),
        "stage3_minimum_flags_in_window": int(STAGE3_MINIMUM_FLAGS_IN_WINDOW),
    },
    rule_source="gold_03b_summary_thresholds_tuning_and_stage3_rule_cells",
    stage3_type="rule_based",
    stage3_saved_as_joblib=False,
    stage1_model_path=STAGE1_MODEL_ARTIFACT_PATH,
    stage2_model_path=STAGE2_MODEL_ARTIFACT_PATH,
    output_artifact_path=CASCADE_RESULTS_PATH_CSV,
    lineage_payload={
        "cascade_truth_hash": CASCADE_TRUTH_HASH,
        "parent_gold_truth_hash": GOLD_PARENT_TRUTH_HASH,
        "downstream_consumer": "gold_03c_if_stage3_improved_uses_tuned_output",
    },
    notes="Gold 03b tuned cascade contract. Stage 3 is rule-based and not saved as joblib.",
)
write_gold_model_output_validation_contract(cascade_tuned_contract, cascade_tuned_contract_path)
```

The `downstream_consumer` note indicates that Gold_03c consumes the tuned output only if Stage 3 shows improvement — the tuned cascade is a candidate, not a guaranteed downstream source.

---

## Detected Row Extracts

Four detected-row frames are built from `cascade_results` via `get_detected_rows_dataframe`, each targeting a different stage flag:

| Frame | Target Flag | Key Columns |
|---|---|---|
| `stage1_detected_rows_dataframe` | `stage1_flag` | `stage1_score`, `stage1_decision`, `stage1_pred`, all stage flags |
| `stage2_detected_rows_dataframe` | `stage2_flag` | `stage2_score`, `stage2_model_*` columns, all stage flags |
| `stage3_evidence_rows_dataframe` | `stage3_profile_breach_flag` | All Stage 3 evidence columns, `cascade_final_flag` |
| `final_detected_rows_dataframe` | `cascade_final_flag` | All Stage 2 and Stage 3 evidence columns |

All four frames are sorted by `time_index` ascending and logged to the ledger. They are used for interactive post-hoc review within the notebook and are not saved to disk as CSVs.

---

## Ledger Close and W&B Finish

After all outputs are written and the detected-row frames are built, the ledger is written and the W&B run is closed:

```python
ledger.write_json(cascade_ledger_path)
wandb.save(str(cascade_ledger_path))
wandb_run.finish()
```

W&B is closed **before** the final lineage checks and SQL write. This matches the pattern in Gold_02 and Gold_03a. Any errors in the post-close steps would not be captured in the W&B run for this execution.

---

## Final Lineage Checks

After `wandb_run.finish()`, seven lineage invariants are verified against the live `cascade_results` frame:

1. **Required lineage columns present**: `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode`
2. **Truth hash roundtrip**: `extract_truth_hash(cascade_results)` must match `CASCADE_TRUTH_HASH`
3. **Parent hash uniqueness**: `cascade_results["meta__parent_truth_hash"]` must have exactly one unique non-null value (multiple hashes would indicate rows from different upstream runs were mixed, breaking traceability — this invariant is documented inline in the source)
4. **Parent hash match**: that single parent value must equal `GOLD_PARENT_TRUTH_HASH`
5. **Truth file existence**: `cascade_truth_path` must exist on disk
6. **Saved truth file roundtrip**: loaded truth file hash must match `CASCADE_TRUTH_HASH`
7. **Saved truth file parent hash**: loaded truth file parent hash must match `GOLD_PARENT_TRUTH_HASH`

Any mismatch raises `ValueError` with the expected and observed values shown side by side.

---

## SQL Write

```python
WRITE_TO_POSTGRES = True
CASCADE_SQL_MODEL_STAGE = "cascade_tuned_final"

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

`write_gold_cascade_scores_sql` writes to `gold.anomaly_detection_scores` with `model_stage = "cascade_tuned_final"`. The `WRITE_TO_POSTGRES = True` gate allows the notebook to run in read-only mode without touching SQL by setting the flag to `False`.

---

## Key Differences from Gold_03a

| Aspect | Gold_03a (default) | Gold_03b (tuned) |
|---|---|---|
| `CASCADE_VARIANT` | `"default"` (hardcoded) | `"tuned"` (hardcoded) |
| `STAGE2_SELECTION_MODE` | `"fixed"` (hardcoded) | Read from config — `"fixed"`, `"threshold_grid"`, or `"parameter_search"` |
| `STAGE2_MIN_RECALL` | Not present | Required floor constraint on Stage 2 recall |
| `STAGE2_THRESHOLD_GRID` | Not present | Config-driven list of percentiles for search |
| `STAGE2_PARAM_GRID` | Not present | Config-driven parameter combinations for `"parameter_search"` mode |
| Stage 2 function | `run_stage2_selection` (fixed path only) | `run_stage2_selection` with `evaluate_stage2_model_with_thresholds` — multi-candidate |
| Artifact filename prefix | `cascade_default_*` or `cascade_defaults_*` | `cascade_tuned_*` |
| `RESOLVED_PATHS` keys | `cascade_default_*` or `cascade_defaults_*` | `cascade_tuned_*` |
| `model_id` in validation contract | `"cascade_default"` | `"cascade_tuned"` |
| `model_stage` in contract and SQL | `"cascade_default_final"` | `"cascade_tuned_final"` |
| `operating_mode` in contract | Not present | `"tuned"` |
| Validation contract `notes` | Gold_03a contract | "Gold 03b tuned cascade contract. Stage 3 is rule-based and not saved as joblib." |
| Row tracking CSVs via `save_data` | 4 CSVs saved to `CASCADE_ROW_TRACKING_DIR` | Detected-row frames built but **not** saved to disk |
| Downstream consumer in lineage_payload | Not present | `"gold_03c_if_stage3_improved_uses_tuned_output"` |

---

## Upstream / Downstream Data Flow

**Reads from Gold_01 PreProcessing (via truth record):**
- `GOLD_PREPROCESSED_SCALED_DATA_PATH` — primary scored input (all rows)
- `GOLD_PREPROCESSED_DATA_PATH` — original preprocessed Parquet
- `GOLD_FIT_DATA_PATH` — normal-only fit subset (Stage 1/2 training, reference profile)
- `GOLD_TEST_DATA_PATH` — test window Parquet
- `GOLD_TRAIN_DATA_PATH` — train window Parquet
- `STAGE1_FEATURES_PATH`, `STAGE2_FEATURES_PATH`, `STAGE3_PRIMARY_PATH`, `STAGE3_SECONDARY_PATH` — feature/sensor JSON lists

**Writes (consumed by downstream notebooks):**

| Artifact | Path Key | Consumer |
|---|---|---|
| `cascade_results` (CSV) | `cascade_tuned_results_path_csv` | Gold_03c; Gold_06A (via validation contract) |
| `cascade_results` (Pickle) | `cascade_tuned_results_path_pickle` | Gold_03c |
| Stage 1 model (joblib) | `cascade_tuned_stage1_model_artifact_path` | Gold_06A |
| Stage 2 model (joblib) | `cascade_tuned_stage2_model_artifact_path` | Gold_06A |
| Reference profile (CSV) | `cascade_tuned_reference_profile_path` | Gold_06A |
| Thresholds JSON | `cascade_tuned_thresholds_path` | Gold_06A |
| Summary JSON | `cascade_tuned_summary_path` | Gold_06A |
| Metadata JSON | `cascade_tuned_metadata_path` | Gold_06A |
| Validation contract | `gold_model_validation_contract_path(model_id="cascade_tuned")` | Gold_06A |
| Truth record | `TRUTHS_PATH / {layer}_{dataset}.json` | Truth index; Gold lineage checks |
| Ledger JSON | `cascade_ledger_path` | Lineage audit |
| SQL rows | `gold.anomaly_detection_scores` (model_stage=`"cascade_tuned_final"`) | Operational layer |

---

## Relationship to Other Notebooks

### Upstream Context

Gold_03b loads Gold_01's scaled Parquet and truth record (8 path overrides: 5 Parquets + 4 feature/sensor JSON lists). No dependency on Gold_02 or Gold_03a. Its Stage 2 selection is config-driven (`STAGE2_SELECTION_MODE` from config), distinguishing it from Gold_03a's fixed selection.

### Downstream Handoff

Gold_03b provides:
- Thresholds JSON (`CASCADE_TUNED_THRESHOLDS_PATH`) to Gold_03c as the `"previous_best"` Stage 2 source — a direct stage-to-stage artifact handoff
- Stage 2 model (`CASCADE_TUNED_STAGE2_MODEL_PATH`) and reference profile to Gold_03c (loaded without retraining)
- Cascade results CSV/pickle and `CASCADE_TUNED_TRUTH_HASH` consumed by Gold_04_Comparison
- Stage 1 + Stage 2 joblib models, thresholds, reference profile, and validation contract consumed by Gold_06A_Test_Replay_Validation
- SQL rows in `gold.anomaly_detection_scores` with `model_stage="cascade_tuned_final"`

### Pipeline Position

Second cascade notebook. Introduces config-driven Stage 2 multi-candidate selection. Its primary inter-notebook role is producing the best Stage 2 configuration for Gold_03c's `"previous_best"` reuse. Gold_03c cannot run without Gold_03b's saved thresholds JSON.

### Relationship Summary

- The only Gold modeling notebook that acts as a direct artifact-level input to another Gold modeling notebook (Gold_03c)
- Reads Gold_01 outputs; no dependency on Gold_02 or Gold_03a
- Gold_03c loads Gold_03b's thresholds JSON, Stage 2 model, and reference profile — the stage-to-stage handoff
- Also produces independent model results consumed by Gold_04 and Gold_06A
- `CASCADE_TUNED_TRUTH_HASH` three-source-validated in Gold_04 alongside baseline and other cascade hashes
