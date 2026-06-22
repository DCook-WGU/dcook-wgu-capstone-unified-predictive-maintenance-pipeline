# Notebook Code Reference: EDA_Notebook_Pump_Gold_03a_Cascade_Modeling

## Notebook Purpose

Gold_03a implements the project's three-stage cascade anomaly detection pipeline under the `"default"` cascade variant configuration. It runs the full cascade sequence — Stage 1 broad Isolation Forest screening, Stage 2 narrow Isolation Forest confirmation, and Stage 3 rule-based evidence confirmation — on Gold-layer scaled pump telemetry data, then writes the cascade-scored results, per-stage model artifacts, per-stage row tracking exports, a Stage 3 reference profile, a model output validation contract, a `gold_cascade` truth record, and cascade metrics to the `gold.anomaly_detection_scores` SQL table.

The cascade is designed to reduce false positives progressively: Stage 1 casts a wide net, Stage 2 narrows the candidate set to rows that also trigger a focused Isolation Forest on a reduced feature set, and Stage 3 applies rule-based evidence checks (sensor profile breach, temporal persistence, and drift detection) to produce the final cascade alert.

## Pipeline Role

- Stage: `gold_cascade`
- Layer: `gold`
- W&B job type: `gold_modeling_cascade`
- Cascade variant: `"default"` (hardcoded; Gold_03b and Gold_03c implement other variants)
- Stage 2 selection mode: `"fixed"` (hardcoded in this notebook)
- Position in workflow: First cascade modeling notebook; runs after Gold_01 preprocessing and Gold_02 baseline
- Primary responsibility: Full three-stage cascade model fit and scoring; produce per-row cascade decision with traceability artifacts

## Inputs

| Input | Source | Expected Form | Used For |
|---|---|---|---|
| `GOLD_PREPROCESSED_SCALED_DATA_PATH` | Gold_01 output Parquet (configured, truth-path not overridden) | Parquet → DataFrame via `load_data()` | Primary cascade scoring input; all-rows feature matrices; `cascade_results` base |
| `GOLD_PREPROCESSED_DATA_PATH` | Gold_01 output (overridden from Gold truth `artifact_paths["gold_preprocessed_path"]`) | Parquet → DataFrame via `load_data()` | Unscaled Gold reference (loaded but not used for model scoring) |
| `GOLD_FIT_DATA_PATH` | Gold_01 output (overridden from Gold truth `artifact_paths["gold_fit_path"]`) | Parquet → DataFrame via `load_data()` | Normal-only rows for Stage 1 and Stage 2 model fit; Stage 3 reference profile |
| `GOLD_TEST_DATA_PATH` | Gold_01 output (overridden from Gold truth `artifact_paths["gold_test_path"]`) | Parquet → DataFrame via `load_data()` | Loaded for test-partition reference |
| `GOLD_TRAIN_DATA_PATH` | Gold_01 output (overridden from Gold truth `artifact_paths["gold_train_path"]`) | Parquet → DataFrame via `load_data()` | Loaded for train-partition reference |
| `STAGE1_FEATURES_PATH` | Gold_01 artifact JSON (overridden from Gold truth `artifact_paths["stage1_features_path"]`) | JSON list of column names | Stage 1 Isolation Forest feature set |
| `STAGE2_FEATURES_PATH` | Gold_01 artifact JSON (overridden from Gold truth `artifact_paths["stage2_features_path"]`) | JSON list of column names | Stage 2 Isolation Forest reduced feature set |
| `STAGE3_PRIMARY_PATH` | Gold_01 artifact JSON (overridden from Gold truth `artifact_paths["stage3_primary_path"]`) | JSON list of sensor names | Stage 3 primary rule sensors for profile breach check |
| `STAGE3_SECONDARY_PATH` | Gold_01 artifact JSON (overridden from Gold truth `artifact_paths["stage3_secondary_path"]`) | JSON list of sensor names | Stage 3 secondary rule sensors for corroboration check |
| Gold_01 truth record | Loaded from `TRUTHS_PATH/gold/` via `meta__truth_hash` column | JSON truth record | Parent hash, dataset name, pipeline mode, eight artifact path overrides, runtime facts (scaler_kind, feature_set_id) |
| `CONFIG`, `RESOLVED_PATHS`, `FILENAMES` | `load_notebook_context()` bootstrap | Config dict, path dict, filename dict | All cascade model parameters, thresholds, output paths |
| `GOLD_CFG` / `STAGE1_CFG` / `STAGE2_CFG` / `STAGE3_CFG` | Nested config sections | Config dicts | Per-stage estimator counts, threshold percentiles, rule parameters |

## Configuration and Runtime Context

| Item | Source | Purpose |
|---|---|---|
| `CASCADE_VARIANT` | Hardcoded `"default"` | Identifies this notebook's artifact set and distinguishes it from Gold_03b/03c |
| `STAGE1_ESTIMATOR_COUNT` | `STAGE1_CFG["estimator_count"]` | Number of trees in Stage 1 Isolation Forest |
| `STAGE1_THRESHOLD_PERCENTILE` | `STAGE1_CFG["threshold_percentile"]` | Percentile of training scores used to set Stage 1 anomaly threshold |
| `STAGE2_SELECTION_MODE` | Hardcoded `"fixed"` | Stage 2 always uses the `fixed` config branch in this notebook |
| `STAGE2_FIXED_PARAMS` | `STAGE2_CFG["fixed"]["params"]` | Model parameters passed to Stage 2 `IsolationForest` (includes `n_estimators`) |
| `STAGE2_FIXED_THRESHOLD_PERCENTILE` | `STAGE2_CFG["fixed"]["threshold_percentile"]` | Percentile of Stage 2 training scores for Stage 2 threshold |
| `STAGE3_MIN_PRIMARY_SENSOR_HITS` | `STAGE3_CFG["min_primary_sensor_hits"]` | Minimum primary sensor breach count for Stage 3 profile breach flag |
| `STAGE3_MIN_SECONDARY_SENSOR_HITS` | `STAGE3_CFG["min_secondary_sensor_hits"]` | Minimum secondary sensor breach count for Stage 3 corroboration flag |
| `STAGE3_ROLLING_WINDOW_SIZE` | `STAGE3_CFG["rolling_window_size"]` | Rolling window width for Stage 3 persistence flag |
| `STAGE3_MINIMUM_FLAGS_IN_WINDOW` | `STAGE3_CFG["minimum_flags_in_window"]` | Minimum Stage 2 flags within rolling window for persistence flag |
| `RANDOM_SEED` | `GOLD_CFG["random_seed"]` | Shared random state for Stage 1 and Stage 2 Isolation Forest |
| `DATASET_NAME` | Initial from `DATASET_CFG.get("name")` → overridden by `get_dataset_name_from_truth(gold_truth)` | Resolved from Gold parent truth |
| `GOLD_PARENT_TRUTH_HASH` | `get_truth_hash(gold_truth)` | Gold_01's preprocessing truth hash; becomes parent in `gold_cascade` truth record |
| `PIPELINE_MODE` | Initial from `PIPELINE["execution_mode"]` → overridden from Gold truth if non-None | Inherited from Gold_01's pipeline mode |
| `GOLD_PROCESS_RUN_ID` | `make_process_run_id(GOLD_CFG["process_run_id_prefix"])` | Run identity stamp for cascade process |
| `gold_truth_runtime_facts` | `gold_truth.get("runtime_facts", {})` | Gold_01 provenance: `feature_set_id`, `scaler_kind_runtime`, `recommended_imputation` |
| `gold_truth_artifact_paths` | `gold_truth.get("artifact_paths", {})` | Source of all eight artifact path overrides |
| `GOLD_CASCADE_ARTIFACT_DIRS` | `build_artifact_dirs_from_config(config, stage_key="gold_cascade", variant="default")` | Cascade-specific artifact subdirectory layout |
| `DATASET_ID`, `RUN_ID`, `ASSET_ID` | `CAPSTONE_DATASET_ID` / `CAPSTONE_RUN_ID` / `CAPSTONE_ASSET_ID` env vars with multi-source fallback | Used for SQL writes and artifact identification |
| `CAPSTONE_SCHEMA` | `CAPSTONE_SCHEMA` env var (default `"capstone"`) | Postgres schema target for SQL smoke check and cascade scores write |

## Logical Workflow Map

1. Bootstrap: `load_notebook_context()`, constants extraction, cascade artifact dir setup, DB engine, W&B init, SQL smoke check
2. Data load and Gold truth propagation: load Gold_01 scaled Parquet, load Gold parent truth, override eight artifact paths, load five Gold Parquets and four feature/sensor JSON files
3. Row identity validation: `ensure_stable_row_id` on Gold scaled dataframe
4. Train/test split recovery: read `meta__is_train_flag` from Gold_01's stamped column
5. Optional label handling: extract `anomaly_flag` as `test_labels` if present
6. Stage 3 reference profile build: `build_reference_profile` from normal-only fit data
7. Feature matrix assembly: Stage 1 and Stage 2 fit and all-rows DataFrames with validation
8. Stage 1 — Broad Isolation Forest: fit, score all rows, calibrate threshold, row-track, synchronize flags
9. Stage 2 — Narrow Isolation Forest: fit, score Stage 1 candidates only, row-track, synchronize gated flags
10. Stage 3 — Rule confirmation: validate rule sensor columns, compute primary breach count, secondary breach count, persistence flag, drift flag, evidence count, final cascade decision
11. Finalize stage flag columns: `finalize_stage_flag_columns` fills sparse flag NaN values
12. Cascade metrics: per-stage alert counts, optional evaluation against test labels
13. Cascade output validation: `validate_cascade_output` gate and flag integrity checks
14. Truth record build: initialize, populate runtime_facts and artifact_paths, build hash, stamp results, save, index
15. Validation contract: `build_gold_model_output_validation_contract` + `write_gold_model_output_validation_contract` for Gold_06A
16. Per-stage detected row exports: four `get_detected_rows_dataframe` extracts saved to `CASCADE_ROW_TRACKING_DIR`
17. Artifact saves: results CSV/pickle, reference profile CSV, Stage 1 model ×2 paths, Stage 2 model ×2 paths, threshold/summary/metadata JSONs, ledger, W&B saves
18. Ledger close and W&B finish: `ledger.write_json`, `wandb_run.finish()`
19. Final lineage verification: meta column presence, truth hash cross-check, parent hash uniqueness, saved truth file re-read check
20. SQL write: `write_gold_cascade_scores_sql` → `gold.anomaly_detection_scores`

## Section Overview

| Section | Purpose | Key Inputs | Key Outputs / Side Effects |
|---|---|---|---|
| Bootstrap | Initialize cascade runtime context, artifact dirs, DB, W&B | `load_notebook_context()`, `CONFIG`, env vars | `CTX`, `logger`, `ledger`, `GOLD_CASCADE_ARTIFACT_DIRS`, `wandb_run` active, config snapshot saved |
| Data load and Gold truth | Load Gold_01 outputs; inherit parent truth; override 8 artifact paths | `GOLD_PREPROCESSED_SCALED_DATA_PATH`, Gold truth record | Five DataFrames, four feature/sensor lists, `GOLD_PARENT_TRUTH_HASH`, path overrides |
| Row identity | Stamp stable `meta__row_id` on Gold scaled dataframe before cascade scoring | `gold_preprocessed_scaled_dataframe` | `meta__row_id` present and unique |
| Split recovery | Recover train/test partition from Gold_01 stamped column | `meta__is_train_flag` | `train_mask`, `test_mask`, `test_labels` (if labels present) |
| Reference profile | Build Stage 3 rule confirmation bounds from normal-only data | `gold_fit_dataframe`, Stage 1 and Stage 3 sensor lists | `reference_profile` DataFrame (median, SD, 5th/95th pct per feature) |
| Feature matrices | Assemble typed fit and all-rows DataFrames for Stage 1 and Stage 2 | `stage1/2_feature_columns`, fit and scaled DataFrames | `stage1/2_train_fit_features`, `stage1/2_all_features` |
| Stage 1 | Broad IF screening on all rows | `stage1_train_fit_features`, `stage1_all_features` | `stage1_model`, `stage1_flags`, `stage1_score/threshold/flag` columns in `cascade_results` |
| Stage 2 | Narrow IF confirmation on Stage 1 candidates only | `stage2_train_fit_features`, Stage 1 candidate mask | `stage2_model`, gated `stage2_flag`, `stage2_score` (NaN for non-candidates), row-tracking columns |
| Stage 3 | Rule-based evidence checks on all rows | `cascade_results`, `reference_profile`, `stage3_*_rule_sensors` | `stage3_profile_breach_count/flag`, `stage3_secondary_breach_count`, `stage3_corroboration_flag`, `stage3_persistence_flag`, `stage3_drift_flag`, `stage3_rule_evidence_count`, `cascade_final_flag` |
| Finalize flags | Fill NaN values from masked stage scoring | `cascade_results` with sparse flags | All flag columns filled as integers |
| Cascade metrics | Compute per-stage alert counts and optional evaluation | `cascade_results`, `test_labels` | `cascade_metrics` dict with precision, recall, F1 if labels available |
| Output validation | Cascade integrity checks | `cascade_results`, `test_mask` | `validate_cascade_output` passes or raises `ValueError` |
| Truth record | Build and stamp `gold_cascade` truth record | `GOLD_PARENT_TRUTH_HASH`, runtime facts, artifact paths | `cascade_truth_record`, `CASCADE_TRUTH_HASH`, stamped `cascade_results` |
| Validation contract | Write model output contract for Gold_06A replay | `cascade_results`, all stage params and paths | `cascade_default_contract` JSON file at `cascade_default_contract_path` |
| Detected rows | Per-stage row tracking exports | `cascade_results`, per-stage flag columns | Four `get_detected_rows_dataframe` extracts saved to `CASCADE_ROW_TRACKING_DIR` |
| Artifact saves | Persist all cascade outputs | All results and models | Results CSV/pickle, profile CSV, 2 models ×2 paths each, threshold/summary/metadata JSONs, truth record, `wandb.save` all |
| Ledger and W&B close | Finalize run record | Session ledger | `cascade_ledger_path` written; W&B run closed |
| Final lineage checks | Assert hash consistency and truth file integrity | `cascade_results`, saved truth file | `ValueError` on any mismatch; truth file re-read and verified |
| SQL write | Write cascade results to Postgres | `cascade_results`, `engine`, `CAPSTONE_SCHEMA` | Rows inserted to `gold.anomaly_detection_scores` as `model_stage="cascade_default_final"` |

## Section Details

### Bootstrap

#### Purpose

Loads the notebook runtime context, initializes the cascade artifact directory layout, sets up the Postgres engine and SQL identifiers, starts W&B, and performs a SQL smoke check against the schema.

#### Key Operations

- `load_notebook_context(stage="gold_cascade", ...)` → `CTX`, `logger`, `ledger`, `RESOLVED_PATHS`, `CONFIG`, `FILENAMES`, `GOLD_CFG`, `engine`, `PIPELINE`, etc.
- Config extraction provides all working constants: `CASCADE_VARIANT`, all stage config sections (`STAGE1_CFG`, `STAGE2_CFG`, `STAGE3_CFG`, `STAGE2_FIXED_CFG`, `STAGE2_FIXED_PARAMS`), all stage thresholds, all input/output paths
- `STAGE2_SELECTION_MODE = "fixed"` — hardcoded; this notebook always uses the fixed Stage 2 branch
- `build_artifact_dirs_from_config(config, stage_key="gold_cascade", variant="default")` → `GOLD_CASCADE_ARTIFACT_DIRS` — provides named subdirectory paths: `models`, `scores`, `thresholds`, `summaries`, `metadata`, `profiles`, `lineage`, `row_tracking`, `plots`, `config`
- `CASCADE_ROW_TRACKING_DIR = GOLD_CASCADE_ARTIFACT_DIRS["row_tracking"]`
- `get_engine_from_env()` → `engine`; `CAPSTONE_SCHEMA`, `DATASET_ID`, `RUN_ID`, `ASSET_ID` resolved via multi-source fallback chain (env vars → config → hardcoded defaults); `ValueError` if any cannot be resolved
- `export_config_snapshot(CONFIG, ...)` — saves a resolved config YAML snapshot if `execution.save_config_snapshot` is true
- `wandb.init(job_type="gold_modeling_cascade", config={...})` — W&B run opened; config includes all three stages' threshold percentiles and selection mode plus input paths
- SQL smoke check via `read_sql_dataframe(engine, "SELECT table_schema, table_name FROM information_schema.tables WHERE ...")` — confirms Postgres connectivity

---

### Data Load and Gold Truth Propagation

#### Purpose

Loads Gold_01's full-scaled Parquet, reads the Gold parent truth record from the embedded hash, overrides eight artifact paths to match what Gold_01 actually saved, then loads five Parquets and four feature/sensor JSON files using the truth-resolved paths.

#### Key Operations

- `gold_preprocessed_scaled_dataframe = load_data(GOLD_PREPROCESSED_SCALED_DATA_PATH)` — primary input
- `GOLD_DATASET_NAME` extracted from `meta__dataset` column; `ValueError` if empty
- `gold_truth = load_parent_truth_record_from_dataframe(parent_layer_name="gold", column_name="meta__truth_hash", ...)` — loads Gold_01's truth record
- `DATASET_NAME = get_dataset_name_from_truth(gold_truth)`; `GOLD_PARENT_TRUTH_HASH = get_truth_hash(gold_truth)`
- `PARENT_PIPELINE_MODE = get_pipeline_mode_from_truth(gold_truth)` — overrides `PIPELINE_MODE` if non-None
- Eight path overrides from `gold_truth["artifact_paths"]`: `GOLD_PREPROCESSED_DATA_PATH`, `GOLD_FIT_DATA_PATH`, `GOLD_TEST_DATA_PATH`, `GOLD_TRAIN_DATA_PATH`, `STAGE1_FEATURES_PATH`, `STAGE2_FEATURES_PATH`, `STAGE3_PRIMARY_PATH`, `STAGE3_SECONDARY_PATH`
- `gold_truth_runtime_facts = gold_truth.get("runtime_facts", {})` — carries forward Gold_01's `feature_set_id`, `scaler_kind_runtime`, `recommended_imputation`
- `gold_truth_artifact_paths = gold_truth.get("artifact_paths", {})` — recorded in cascade metadata JSON for downstream traceability
- `GOLD_TRUTH_PATH` constructed from `TRUTHS_PATH / "gold" / f"{DATASET_NAME}__gold__truth__{GOLD_PARENT_TRUTH_HASH}.json"`
- Loads (all via truth-overridden paths): `gold_preprocessed_dataframe`, `gold_fit_dataframe`, `gold_test_dataframe`, `gold_train_dataframe`
- `stage1_feature_columns = require_str_list(load_json(STAGE1_FEATURES_PATH))` — strict validated list
- `stage2_feature_columns = require_str_list(load_json(STAGE2_FEATURES_PATH))`
- `stage3_primary_rule_sensors = require_str_list(load_json(STAGE3_PRIMARY_PATH))`
- `stage3_secondary_rule_sensors = require_str_list(load_json(STAGE3_SECONDARY_PATH))`

#### In-Place Usage Notes

Gold_03a overrides eight artifact paths from the Gold parent truth record, compared to Gold_02's two. The `require_str_list` wrapper (used instead of bare `load_json`) enforces that the loaded JSON is a non-empty list of strings, raising `TypeError`/`ValueError` before the cascade begins if any feature or sensor list is malformed. The `GOLD_PARENT_TRUTH_HASH` is captured at this point and carried forward through all truth records, validation contracts, and metadata JSONs so every downstream notebook can trace back to exactly which Gold_01 preprocessing run produced the modeling inputs.

---

### Row Identity Validation

#### Purpose

Confirms that `meta__row_id` is stable and unique on the Gold scaled dataframe before any cascade scoring begins. All later `merge_stage_results_back` calls join on this key.

#### Key Operations

- `gold_preprocessed_scaled_dataframe = ensure_stable_row_id(gold_preprocessed_scaled_dataframe, row_id_column="meta__row_id")`
- `ledger.add(kind="step", step="validate_cascade_row_tracking", data={"row_id_unique": bool(...)})`

---

### Train/Test Split Recovery

#### Purpose

Recovers the train/test partition from Gold_01's `meta__is_train_flag` column. The split is not re-derived; it is restored from the stamped column so the cascade evaluates on exactly the same held-out rows as preprocessing.

#### Key Operations

- `if "meta__is_train_flag" not in gold_preprocessed_scaled_dataframe.columns: raise ValueError(...)` — hard stop
- `train_mask = gold_preprocessed_scaled_dataframe["meta__is_train_flag"].fillna(False).astype(bool)`
- `test_mask = (~train_mask).astype(bool)`
- `if "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns`: extracts `test_labels` as integer NumPy array; else `test_labels = None`

---

### Stage 3 Reference Profile Build

#### Purpose

Constructs a normal operating envelope for Stage 3 rule confirmation by computing statistical bounds on the fit (normal-only) data. The profile is shared across the primary and secondary sensor checks.

#### Key Operations

- `reference_profile_features = list(dict.fromkeys(stage1_feature_columns + stage3_primary_rule_sensors + stage3_secondary_rule_sensors))` — deduplicated union
- `build_reference_profile(gold_fit_dataframe, feature_columns=reference_profile_features)` → `reference_profile` DataFrame with columns: `feature_name`, `median_value`, `mean_value`, `standard_deviation`, `lower_bound` (5th percentile), `upper_bound` (95th percentile)
- Built from fit (normal-only) data so that anomalous test-window values do not contaminate the normal operating bounds

---

### Feature Matrix Assembly

#### Purpose

Builds all four feature DataFrames needed for Stage 1 and Stage 2 model fit and scoring, with pre-flight validation.

#### Key Operations

- Validates all Stage 1 and Stage 2 columns present in both `gold_fit_dataframe` and `gold_preprocessed_scaled_dataframe`; `ValueError` with list of missing columns on failure
- `stage1_train_fit_features = gold_fit_dataframe[stage1_feature_columns]` — Stage 1 fit input (normal-only)
- `stage2_train_fit_features = gold_fit_dataframe[stage2_feature_columns]` — Stage 2 fit input (normal-only)
- `stage1_all_features = gold_preprocessed_scaled_dataframe[stage1_feature_columns]` — Stage 1 scoring input (all rows)
- `stage2_all_features = gold_preprocessed_scaled_dataframe[stage2_feature_columns]` — Stage 2 scoring input (all rows)
- All kept as DataFrames to preserve feature names for sklearn

---

### Stage 1 — Broad Isolation Forest Screening

#### Purpose

Trains a broad-sensitivity Isolation Forest on normal-only data, scores all rows, and sets the Stage 1 alert threshold. Stage 1 produces a wide candidate set with high recall; strict specificity is handled downstream by Stage 2 and Stage 3.

#### Key Operations

- `stage1_model = IsolationForest(n_estimators=STAGE1_ESTIMATOR_COUNT, random_state=RANDOM_SEED, n_jobs=-1)`
- `stage1_model.fit(stage1_train_fit_features)` — fit on normal-only fit rows
- `compute_anomaly_scores_isolation_forest(stage1_model, stage1_train_fit_features)` = `-model.score_samples()` (higher = more anomalous)
- `stage1_train_scores` for threshold calibration only; `stage1_all_scores` for all-rows scoring
- `choose_threshold_value(stage1_train_scores, STAGE1_THRESHOLD_PERCENTILE)` → `stage1_threshold`; calibrated from training scores only (test scores unseen)
- `stage1_flags = (stage1_all_scores >= stage1_threshold).astype(int)` — broad candidate array
- Row-tracked scoring via `build_stage_scoring_frame` + `score_isolation_forest_stage(stage_name="stage1")` + `merge_stage_results_back(master_dataframe=gold_preprocessed_scaled_dataframe.copy())`:
  - `cascade_results` initialized here as a standalone copy from the Gold scaled dataframe
- Synchronized onto `cascade_results`: `stage1_score`, `stage1_threshold`, `stage1_threshold_percentile`, `stage1_flag`

#### In-Place Usage Notes

`cascade_results` is created here as a copy of the Gold scaled dataframe with Stage 1 row-level outputs merged on `meta__row_id`. From this point forward, all subsequent stages append their columns to `cascade_results` rather than creating new frames. The copy ensures the original scaled dataframe is not mutated.

---

### Stage 2 — Narrow Isolation Forest Confirmation

#### Purpose

Trains a focused Isolation Forest on a reduced feature set and applies it only to the Stage 1 candidate rows. Stage 2 is gated by Stage 1: a row can receive a `stage2_flag` only if it already has `stage1_flag == 1`. Non-candidate rows receive `stage2_score = NaN` to distinguish absent scores from zero scores.

#### Key Operations

- `stage2_model = IsolationForest(random_state=STAGE2_RANDOM_STATE, n_jobs=-1, **STAGE2_FIXED_PARAMS)` — uses the `fixed` params branch
- `stage2_model.fit(stage2_train_fit_features)` — fit on normal-only rows
- `stage2_threshold = choose_threshold_value(stage2_train_scores, STAGE2_FIXED_THRESHOLD_PERCENTILE)` — calibrated from training scores only
- `stage2_raw_flags = (stage2_all_scores >= stage2_threshold).astype(int)` — Stage 2 independent alert array
- `stage2_flags = (stage1_flags == 1) & (stage2_raw_flags == 1)` — confirmed alert requires passing both Stage 1 AND Stage 2
- `stage2_candidate_mask = (cascade_results["stage1_flag"] == 1)` — Stage 2 row-tracking processes only Stage 1 positives
- `build_stage_scoring_frame(cascade_results, feature_columns=stage2_feature_columns, mask=stage2_candidate_mask)` — subset to candidates
- `score_isolation_forest_stage(stage2_input_df, stage2_model, stage_name="stage2")` → raw helper outputs renamed before merge: `stage2_score → stage2_model_score`, `stage2_flag → stage2_model_flag`, `stage2_decision → stage2_model_decision`, `stage2_pred → stage2_model_pred`
- Merged back to `cascade_results` on `meta__row_id` (left join; non-candidates get NaN)
- `cascade_results["stage2_score"] = NaN`; set only for Stage 1 candidates
- `cascade_results["stage2_raw_flag"]` and `cascade_results["stage2_flag"]` filled for candidates; zero elsewhere

#### In-Place Usage Notes

Renaming the Stage 2 helper's raw output columns before merging preserves the distinction between the threshold-calibrated `stage2_flag` (which enforces the gate) and the raw helper output `stage2_model_flag`. The NaN fill for non-candidate Stage 2 scores is intentional — it signals "Stage 2 was not evaluated for this row" rather than "Stage 2 produced a zero anomaly score."

---

### Stage 3 — Rule-Based Evidence Confirmation

#### Purpose

Applies four rule-based evidence checks to all rows of `cascade_results` using the reference profile computed from normal-only data. The final cascade decision requires a row to have passed both Stage 1 and Stage 2, and then meet at least one Stage 3 evidence threshold.

#### Key Operations

**Primary breach (profile boundary check):**
- `compute_primary_breach_count(cascade_results, reference_profile, feature_columns=stage3_primary_rule_sensors)` → `stage3_profile_breach_count`: counts how many primary sensors fall outside their reference `[lower_bound, upper_bound]` range
- `stage3_profile_breach_flag = (stage3_profile_breach_count >= STAGE3_MIN_PRIMARY_SENSOR_HITS).astype(int)`

**Secondary breach (corroboration check):**
- `compute_secondary_breach_count(cascade_results, reference_profile, feature_columns=stage3_secondary_rule_sensors)` → `stage3_secondary_breach_count`: counts secondary sensor boundary violations
- `stage3_corroboration_flag = (stage3_secondary_breach_count >= STAGE3_MIN_SECONDARY_SENSOR_HITS).astype(int)`

**Persistence check (temporal confirmation):**
- `compute_persistence_flag(cascade_results["stage2_flag"], rolling_window_size=STAGE3_ROLLING_WINDOW_SIZE, minimum_flags_in_window=STAGE3_MINIMUM_FLAGS_IN_WINDOW)` → `stage3_persistence_flag`
- Applied to `stage2_flag` (not `stage1_flag`) so persistence reflects consecutive confirmed Stage 2 alerts

**Drift check (rolling deviation):**
- `stage3_rule_watch_features = list(dict.fromkeys(stage3_primary_rule_sensors + stage3_secondary_rule_sensors))`
- `compute_drift_flag(cascade_results, feature_columns=stage3_rule_watch_features, rolling_window_size=5, drift_threshold_multiplier=1.0)` → `stage3_drift_flag`
- Triggers when any watch-feature's absolute deviation from its rolling median exceeds 1.0 × its standard deviation

**Evidence aggregation and final decision:**
- `stage3_rule_evidence_count = stage3_profile_breach_flag + stage3_persistence_flag + stage3_drift_flag + stage3_corroboration_flag`
- `cascade_final_flag = (stage1_flag == 1) AND (stage2_flag == 1) AND (stage3_profile_breach_count >= STAGE3_MIN_PRIMARY_SENSOR_HITS OR stage3_rule_evidence_count >= 2)`
- `finalize_stage_flag_columns(cascade_results, stage_names=["stage1","stage2","stage3"])` — fills NaN values introduced by masked Stage 2 scoring so all flag columns are integer-typed

#### In-Place Usage Notes

The Stage 3 rule checks run over all rows, not just Stage 2 candidates. The cascade gating (`stage1_flag == 1 AND stage2_flag == 1`) is enforced in the final decision expression rather than by filtering rows first, so Stage 3 evidence values are computed for every row and the full evidence count is available for inspection even on non-alert rows.

---

### Cascade Metrics

#### Purpose

Assembles per-stage alert counts and optionally evaluates `cascade_final_flag` against ground-truth labels on the test partition.

#### Key Operations

- `cascade_metrics` dict: `model="3-Stage Cascade"`, per-stage alert counts for all rows and test rows
- If `test_labels` is not None: `precision_recall_fscore_support(test_labels_array, cascade_test_flags, average="binary")` → precision, recall, F1 added to `cascade_metrics`

---

### Cascade Output Validation

#### Purpose

Enforces structural and gate integrity constraints on `cascade_results` before truth record creation.

#### Key Operations

- `validate_cascade_output(cascade_results, test_mask, final_flag_column="cascade_final_flag")` checks:
  - Required columns present: `meta__row_id`, `meta__is_train_flag`, `stage1_flag`, `stage2_raw_flag`, `stage2_flag`, `cascade_final_flag`
  - test_mask length matches dataframe length
  - `meta__row_id` is unique
  - All binary flag columns contain only `{0, 1}` values
  - Stage 2 gate: no row has `stage2_flag == 1` where `stage1_flag != 1`
  - Final gate: no row has `cascade_final_flag == 1` where `stage2_flag != 1`

---

### Truth Record Build

#### Purpose

Initializes the `gold_cascade` truth record linking this cascade run to Gold_01's preprocessing truth, builds the hash, stamps truth columns into `cascade_results`, saves the record, and appends the truth index.

#### Key Operations

- `initialize_layer_truth(layer_name="gold_cascade", parent_truth_hash=GOLD_PARENT_TRUTH_HASH)` — parent hash is Gold_01's truth hash
- `update_truth_section(..., "config_snapshot", truth_config_snapshot)` — includes stage, dataset, cascade_variant, mode, profile
- `update_truth_section(..., "runtime_facts", {...})` — records: all stage thresholds, estimator counts, Stage 2 best params, feature/rule counts, result row count, `gold_process_run_id` (from `gold_truth.get("process_run_id")`), `gold_feature_set_id` (from `gold_truth_runtime_facts.get("feature_set_id")`)
- `update_truth_section(..., "artifact_paths", {...})` — records all Gold input paths and all cascade output paths
- `build_truth_record(...)` → `CASCADE_TRUTH_HASH`
- `stamp_truth_columns(cascade_results, truth_hash=CASCADE_TRUTH_HASH, parent_truth_hash=GOLD_PARENT_TRUTH_HASH, pipeline_mode=PIPELINE_MODE)`
- `save_truth_record(cascade_truth_record, truth_dir=TRUTHS_PATH, layer_name="gold_cascade")` → `cascade_truth_path`
- `append_truth_index(cascade_truth_record, truth_index_path=TRUTH_INDEX_PATH)`

---

### Validation Contract

#### Purpose

Writes a structured model output validation contract that Gold_06A (Test Replay Validation) uses to verify the cascade model's behavior against held-out test scenarios.

#### Key Operations

- `cascade_default_contract_path = gold_model_validation_contract_path(artifacts_root=paths.artifacts, dataset_id=DATASET_ID, model_id="cascade_default")`
- `build_gold_model_output_validation_contract(...)` with: `model_id="cascade_default"`, `model_stage="cascade_default_final"`, `source_notebook="gold_03a_cascade_modeling"`, `stage3_type="rule_based"`, `stage3_saved_as_joblib=False`, `rule_config` (all three stages' threshold percentiles, Stage 2 params, Stage 3 rule thresholds), `lineage_payload={"cascade_truth_hash": CASCADE_TRUTH_HASH, "parent_gold_truth_hash": GOLD_PARENT_TRUTH_HASH}`
- `write_gold_model_output_validation_contract(cascade_default_contract, output_path=cascade_default_contract_path)`

---

### Per-Stage Detected Row Exports

#### Purpose

Extracts per-stage alert rows into separate DataFrames for traceability, comparison, and audit, then saves them to named row-tracking CSV files.

#### Key Operations

- `get_detected_rows_dataframe(cascade_results, target_flag_column="stage1_flag", ...)` → `stage1_detected_rows_dataframe`
- `get_detected_rows_dataframe(cascade_results, target_flag_column="stage2_flag", ...)` → `stage2_detected_rows_dataframe`
- `get_detected_rows_dataframe(cascade_results, target_flag_column="stage3_profile_breach_flag", include_columns=[...all stage3 columns...])` → `stage3_evidence_rows_dataframe`
- `get_detected_rows_dataframe(cascade_results, target_flag_column="cascade_final_flag", include_columns=[...all stage columns...])` → `final_detected_rows_dataframe`
- All saved via `save_data(frame, CASCADE_ROW_TRACKING_DIR, filename)`:
  - `{DATASET_NAME}__gold__cascade_03a__row_tracking__stage1_detection.csv`
  - `{DATASET_NAME}__gold__cascade_03a__row_tracking__stage2_detection.csv`
  - `{DATASET_NAME}__gold__cascade_03a__row_tracking__stage3_evidence_detection.csv`
  - `{DATASET_NAME}__gold__cascade_03a__row_tracking__final_detection.csv`
- All four `wandb.save(...)` registered with W&B

---

### Artifact Saves

#### Purpose

Persists all cascade outputs to disk and registers them with W&B.

#### Key Operations

- `cascade_results.to_csv(CASCADE_RESULTS_PATH_CSV)` — full scored results with all meta, lineage, and per-stage columns
- `cascade_results.to_pickle(CASCADE_RESULTS_PATH_PICKLE)`
- `reference_profile.to_csv(CASCADE_REFERENCE_PROFILE_PATH)` — normal operating bounds used by Stage 3
- `joblib.dump(stage1_model, STAGE1_MODEL_ARTIFACT_PATH)` and `STAGE1_MODELS_PATH` (×2 paths)
- `joblib.dump(stage2_model, STAGE2_MODEL_ARTIFACT_PATH)` and `STAGE2_MODELS_PATH` (×2 paths)
- Stage 3 is not persisted via joblib — it is rule-based and fully reconstructable from the reference profile and the config thresholds
- `save_json(cascade_thresholds, CASCADE_THRESHOLDS_PATH)` — both stages' threshold percentiles and values
- `save_json(cascade_summary, CASCADE_SUMMARY_PATH)` — metrics, alert counts, truth hashes, process run IDs, feature/rule counts
- `save_json(cascade_metadata, CASCADE_METADATA_PATH)` — all artifact paths, Gold_01 provenance (scaler_path, scaler_kind, recommended_imputation, feature_set_id), cascade truth linkage
- `wandb.save(...)` for all artifact files and the truth record

---

### Ledger and W&B Close

#### Purpose

Finalizes the cascade session ledger and closes the W&B run.

#### Key Operations

- `ledger.write_json(cascade_ledger_path)` — path is `GOLD_CASCADE_ARTIFACT_DIRS["lineage"] / GOLD_CASCADE_LEDGER_FILE_NAME`
- `wandb.save(str(cascade_ledger_path))`
- `wandb_run.finish()` — no W&B logging valid after this point

---

### Final Lineage Verification

#### Purpose

Asserts that truth hash and parent hash columns are correctly stamped in `cascade_results`, and re-reads the saved truth file to verify the persisted hash matches the in-memory value.

#### Key Operations

- Checks `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` are present in `cascade_results`
- `extract_truth_hash(cascade_results)` must equal `CASCADE_TRUTH_HASH`; `ValueError` on mismatch
- `cascade_results["meta__parent_truth_hash"]` must be exactly one unique non-null value matching `GOLD_PARENT_TRUTH_HASH`; `ValueError` if empty or multiple values
- `load_json(cascade_truth_path)` re-reads the saved truth file; verifies its `truth_hash` and `parent_truth_hash` fields match `CASCADE_TRUTH_HASH` and `GOLD_PARENT_TRUTH_HASH`; `FileNotFoundError` if the truth file was not created

---

### SQL Write

#### Purpose

Writes the cascade-scored results to the `gold.anomaly_detection_scores` table in Postgres, using `model_stage="cascade_default_final"` as the discriminator.

#### Key Operations

- `WRITE_TO_POSTGRES = True`; gated `if WRITE_TO_POSTGRES:` allows dry-run by toggling to `False`
- `write_gold_cascade_scores_sql(engine, capstone_schema, dataset_id, run_id, notebook_globals=globals(), dataframe=cascade_results, dataset_name, model_stage="cascade_default_final")`
- Target: `gold.anomaly_detection_scores`

## Key Function Calls and In-Place Usage

| Function / Method | Context in This Notebook | Inputs Provided Here | Return / Side Effect |
|---|---|---|---|
| `load_notebook_context("gold_cascade")` | Bootstrap | stage, dataset, mode, profile | `CTX` with all shared context aliases |
| `build_artifact_dirs_from_config(config, stage_key="gold_cascade", variant="default")` | Cascade artifact dir layout | `CONFIG`, `"default"` variant | `GOLD_CASCADE_ARTIFACT_DIRS` dict of named subdirs |
| `load_data(path)` | Loads five Gold Parquets | Configured paths, truth-overridden | DataFrames: scaled, preprocessed, fit, test, train |
| `load_parent_truth_record_from_dataframe(parent_layer_name="gold")` | Load Gold_01 truth record | `meta__truth_hash` column, `TRUTHS_PATH` | `gold_truth` dict; source of all 8 path overrides and parent hash |
| `require_str_list(load_json(path))` | Load feature/sensor lists with validation | Four truth-overridden JSON paths | `stage1/2_feature_columns`, `stage3_primary/secondary_rule_sensors` |
| `ensure_stable_row_id(dataframe, row_id_column="meta__row_id")` | Row identity before cascade scoring | `gold_preprocessed_scaled_dataframe` | `meta__row_id` stamped and unique on all rows |
| `build_reference_profile(gold_fit_dataframe, feature_columns=reference_profile_features)` | Stage 3 normal operating bounds | Fit-only data; Stage 1 + Stage 3 sensor union | `reference_profile` DataFrame (5th/95th pct bounds per feature) |
| `compute_anomaly_scores_isolation_forest(model, features)` | Score all rows for Stage 1 and Stage 2 | `stage1/2_model`, feature DataFrames | `-model.score_samples()` arrays (higher = more anomalous) |
| `choose_threshold_value(train_scores, percentile)` | Calibrate Stage 1 and Stage 2 thresholds | Training-only score arrays | `stage1/2_threshold` floats |
| `build_stage_scoring_frame(..., mask=None/candidate_mask)` | Prepare row-indexed input for scoring helper | Stage 1: all rows; Stage 2: Stage 1 candidates only | Masked scoring input frame aligned to `meta__row_id` |
| `score_isolation_forest_stage(stage_dataframe, model, stage_name)` | Row-tracked scoring with stage-prefixed columns | `stage1/2_model`, feature columns | Frame with `stageN_score`, `stageN_decision`, `stageN_pred`, `stageN_flag` |
| `merge_stage_results_back(master_dataframe, stage_results_dataframe, stage_name)` | Add stage outputs to cascade results | `gold_preprocessed_scaled_dataframe.copy()` (Stage 1) or `cascade_results` (Stage 2) | `cascade_results` with stage columns merged on `meta__row_id` |
| `compute_primary_breach_count(cascade_results, reference_profile, stage3_primary_rule_sensors)` | Stage 3 profile breach | `reference_profile` lower/upper bounds | `stage3_profile_breach_count` Series |
| `compute_secondary_breach_count(cascade_results, reference_profile, stage3_secondary_rule_sensors)` | Stage 3 corroboration | `reference_profile` lower/upper bounds | `stage3_secondary_breach_count` Series |
| `compute_persistence_flag(cascade_results["stage2_flag"], rolling_window_size, minimum_flags_in_window)` | Stage 3 temporal persistence | Stage 2 flag column | `stage3_persistence_flag` Series |
| `compute_drift_flag(cascade_results, stage3_rule_watch_features, rolling_window_size=5)` | Stage 3 drift detection | Union of primary + secondary sensors | `stage3_drift_flag` Series |
| `finalize_stage_flag_columns(cascade_results, stage_names=["stage1","stage2","stage3"])` | Fill NaN in sparse stage columns | `cascade_results` | All stage flag columns filled as integers |
| `validate_cascade_output(cascade_results, test_mask)` | Structural and gate integrity check | `final_flag_column="cascade_final_flag"` | Summary dict; raises `ValueError` on gate violation or structural failure |
| `initialize_layer_truth(layer_name="gold_cascade", parent_truth_hash=GOLD_PARENT_TRUTH_HASH)` | Start cascade truth record | `GOLD_PARENT_TRUTH_HASH` | `cascade_truth` dict with parent linkage |
| `build_truth_record(cascade_truth, ...)` | Finalize truth record | Row/column counts, meta/feature columns | `cascade_truth_record`; `CASCADE_TRUTH_HASH` |
| `stamp_truth_columns(cascade_results, CASCADE_TRUTH_HASH, GOLD_PARENT_TRUTH_HASH)` | Write truth lineage into all rows | `cascade_results` | `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` stamped |
| `build_gold_model_output_validation_contract(...)` | Write contract for Gold_06A | All stage params, model paths, lineage payload | `cascade_default_contract` dict |
| `write_gold_model_output_validation_contract(contract, cascade_default_contract_path)` | Persist contract JSON | `cascade_default_contract` | Contract file written at `cascade_default_contract_path` |
| `get_detected_rows_dataframe(cascade_results, target_flag_column, ...)` | Per-stage alert row export | `"stage1_flag"`, `"stage2_flag"`, `"stage3_profile_breach_flag"`, `"cascade_final_flag"` | Four DataFrames of flagged rows |
| `save_data(frame, CASCADE_ROW_TRACKING_DIR, filename)` | Persist row tracking CSVs | Four row-tracking DataFrames | Four CSVs in `CASCADE_ROW_TRACKING_DIR` |
| `joblib.dump(stage1_model, path)` | Persist Stage 1 model | `STAGE1_MODEL_ARTIFACT_PATH` and `STAGE1_MODELS_PATH` | Stage 1 model saved to two paths |
| `joblib.dump(stage2_model, path)` | Persist Stage 2 model | `STAGE2_MODEL_ARTIFACT_PATH` and `STAGE2_MODELS_PATH` | Stage 2 model saved to two paths |
| `write_gold_cascade_scores_sql(engine, ..., model_stage="cascade_default_final")` | SQL write | `cascade_results`, `CAPSTONE_SCHEMA`, `DATASET_ID`, `RUN_ID` | Rows inserted to `gold.anomaly_detection_scores` |

## Outputs and Artifacts

| Output | Type | Location / Destination | Downstream Consumer |
|---|---|---|---|
| Cascade results | CSV | `CASCADE_RESULTS_PATH_CSV` | Gold_04 comparison, Gold_05 anomaly detection |
| Cascade results | Pickle | `CASCADE_RESULTS_PATH_PICKLE` | Gold_04 comparison, Gold_05 anomaly detection |
| Stage 3 reference profile | CSV | `CASCADE_REFERENCE_PROFILE_PATH` | Gold_04, diagnostic audit |
| Stage 1 model | Joblib | `STAGE1_MODEL_ARTIFACT_PATH` (primary), `STAGE1_MODELS_PATH` (models root) | Gold_04 comparison |
| Stage 2 model | Joblib | `STAGE2_MODEL_ARTIFACT_PATH` (primary), `STAGE2_MODELS_PATH` (models root) | Gold_04 comparison |
| Cascade thresholds | JSON | `CASCADE_THRESHOLDS_PATH` | Gold_04 comparison |
| Cascade summary | JSON | `CASCADE_SUMMARY_PATH` | Gold_04 comparison |
| Cascade metadata | JSON | `CASCADE_METADATA_PATH` | Gold_04, Gold_05 provenance traceability |
| `gold_cascade` truth record | JSON | `TRUTHS_PATH/gold_cascade/` | Gold_04, Gold_06A parent truth chain |
| Truth index entry | JSONL append | `TRUTH_INDEX_PATH` | Cross-stage audit |
| Model output validation contract | JSON | `cascade_default_contract_path` | Gold_06A Test Replay Validation |
| Stage 1 detected rows | CSV | `CASCADE_ROW_TRACKING_DIR/...stage1_detection.csv` | Diagnostic review, Gold_06A |
| Stage 2 detected rows | CSV | `CASCADE_ROW_TRACKING_DIR/...stage2_detection.csv` | Diagnostic review |
| Stage 3 evidence rows | CSV | `CASCADE_ROW_TRACKING_DIR/...stage3_evidence_detection.csv` | Diagnostic review |
| Final cascade detected rows | CSV | `CASCADE_ROW_TRACKING_DIR/...final_detection.csv` | Diagnostic review, Gold_06A |
| Cascade ledger | JSONL | `GOLD_CASCADE_ARTIFACT_DIRS["lineage"]/...` | Run audit, W&B |
| Scored cascade rows | SQL rows | `gold.anomaly_detection_scores` (model_stage `cascade_default_final`) | Operational monitoring, reporting |
| W&B run artifacts | W&B | All result/model/JSON/tracking/truth files | Experiment tracking |

## Data Quality / Validation Behavior

| Check | Purpose | Failure / Risk Prevented |
|---|---|---|
| `GOLD_DATASET_NAME` non-empty from `meta__dataset` | Confirm input carries valid dataset identity | `ValueError`; prevents empty dataset name in truth records |
| `load_parent_truth_record_from_dataframe` succeeds | Confirm Gold_01 truth is readable | Raises if truth file not found; prevents cascade on uncertified preprocessing output |
| `require_str_list` on all four JSON loads | Enforce non-empty string lists for features and sensors | `TypeError`/`ValueError` before cascade begins; catches empty or malformed JSON |
| `meta__is_train_flag` column present | Ensure Gold_01 stamped the split | `ValueError`; prevents re-deriving a different split |
| Stage 1 and Stage 2 feature column presence in both DataFrames | Confirm feature sets are available before model fit | `ValueError` with missing column list; prevents silent shape mismatch |
| Score length vs cascade_results length | Synchronized scoring guard after Stage 1 | `ValueError`; prevents misaligned score-to-row assignment |
| Stage 3 rule sensor presence check | Warn on missing Stage 3 columns before rule computation | Logs warning; partial computation instead of hard failure allows audit of missing sensors |
| `validate_cascade_output(...)` | Structural and gate integrity | `ValueError` on missing required columns, non-unique `meta__row_id`, non-binary flags, Stage 2 gate violation, final gate violation |
| Final `meta__truth_hash` cross-check | Confirm stamped hash matches computed hash | `ValueError` on mismatch; catches late-stage column overwrites |
| Final `meta__parent_truth_hash` uniqueness | Confirm single consistent parent hash | `ValueError` if empty or multiple values |
| Truth file re-read verification | Confirm saved JSON file contains correct hashes | `FileNotFoundError` if not created; `ValueError` on hash mismatch in file vs in-memory |

## Downstream Handoff

Gold_04 (Comparison) reads the cascade results CSV/pickle and summary/threshold/metadata JSONs alongside Gold_02 baseline results to produce a side-by-side comparison. The `CASCADE_TRUTH_HASH` becomes the upstream anchor for Gold_04's truth record.

Gold_06A (Test Replay Validation) reads the model output validation contract written by this notebook (`cascade_default_contract`). The contract includes the full rule configuration, model paths, stage threshold values, and lineage payload — giving Gold_06A a self-contained specification against which to verify the cascade replay.

Gold_05 (Anomaly Detection) reads the cascade results and metadata for final anomaly scoring and reporting.

Gold_03b reads this notebook's summary notes — the source notebook's `## Next Stage` heading states: "Gold 03b continues cascade tuning using the same Gold modeling foundation."

The `cascade_metadata.json` records the full Gold_01 provenance trail (scaler path, scaler kind, recommended imputation, feature_set_id) alongside the cascade's own artifact paths, so comparison and audit notebooks can trace the exact preprocessing run without re-reading the Gold_01 truth record directly.

---

## Relationship to Other Notebooks

### Upstream Context

Gold_03a loads Gold_01's scaled Parquet and truth record (8 path overrides: 5 Parquets + 4 feature/sensor JSON lists). No dependency on Gold_02, Gold_03b, or Gold_03c. Stage 2 selection is fixed (`STAGE2_SELECTION_MODE="fixed"` hardcoded); no external configuration search.

### Downstream Handoff

Gold_03a provides:
- Cascade results CSV/pickle and metadata JSON consumed by Gold_04_Comparison as the `"defaults"` model comparison input
- `CASCADE_DEFAULTS_TRUTH_HASH` validated by Gold_04 via three-source check
- Stage 1 and Stage 2 joblib models, thresholds JSON, reference profile, and validation contract consumed by Gold_06A_Test_Replay_Validation
- SQL rows in `gold.anomaly_detection_scores` with `model_stage="cascade_defaults_final"`

### Pipeline Position

First cascade modeling notebook. Establishes the `"defaults"` variant with fixed Stage 2 selection and hardcoded Stage 3 rules. Its outputs are compared against Gold_03b and Gold_03c in Gold_04 to establish the value of tuned cascade configurations.

### Relationship Summary

- Reads Gold_01 scaled Parquet and truth record; no dependency on Gold_02 or Gold_03b/c
- Produces `"defaults"` cascade results and model artifacts consumed by Gold_04 and Gold_06A
- `CASCADE_DEFAULTS_TRUTH_HASH` is the baseline cascade reference point in Gold_04 comparison
- Gold_04 cross-validates that Gold_03a shares `GOLD_PARENT_TRUTH_HASH` with Gold_02, Gold_03b, and Gold_03c
- No stage-to-stage artifact dependency with Gold_03b or Gold_03c

## Notes / Risks / Deferred Cleanup

- `CASCADE_VARIANT = "default"` is hardcoded; this notebook's outputs are keyed to the "default" variant. Gold_03b and Gold_03c implement other cascade configurations.
- `STAGE2_SELECTION_MODE = "fixed"` is hardcoded; Stage 2 always uses the `fixed` config branch (`STAGE2_FIXED_PARAMS`, `STAGE2_FIXED_THRESHOLD_PERCENTILE`). A `STAGE2_FIXED_WARNING_THRESHOLD_PERCENTILE` and `STAGE2_FIXED_CONFIRMED_THRESHOLD_PERCENTILE` are also resolved from config but not directly used in the Stage 2 gating logic in this notebook.
- Stage 3 rule logic is entirely in-notebook function definitions (`compute_primary_breach_count`, `compute_secondary_breach_count`, `compute_persistence_flag`, `compute_drift_flag`). It is not saved as a joblib artifact; this is recorded in the validation contract (`stage3_saved_as_joblib=False`).
- Eight artifact paths are overridden from the Gold parent truth record. The configured values from cell 11 serve as fallbacks only. Any consumer running Gold_03a in a non-standard environment should confirm the Gold_01 truth record exists and contains correct `artifact_paths` before execution.
- `wandb_run.finish()` fires in the ledger-close section before the final lineage checks and SQL write. Artifacts logged after that point are not registered with W&B.
- The Stage 2 row-tracking merge uses a left join so non-candidate rows receive NaN model score columns. `finalize_stage_flag_columns` fills these NaN values as integers before truth record creation, but NaN score values for non-candidates are preserved in the final results.
- The drift detection window size (`rolling_window_size=5`) and multiplier (`drift_threshold_multiplier=1.0`) are hardcoded in the drift function call in this notebook, not driven from `STAGE3_CFG`.
