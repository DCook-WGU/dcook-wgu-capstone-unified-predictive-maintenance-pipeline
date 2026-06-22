# Notebook Code Reference: EDA_Notebook_Pump_Gold_02_Baseline_Modeling

## Notebook Purpose

Gold_02 trains and evaluates the project's baseline anomaly detection model — a single Isolation Forest — on Gold-layer scaled pump telemetry data. It is the first modeling notebook in the Gold layer and establishes the baseline against which all cascade models in Gold_03a–Gold_03c are later compared.

The notebook reads Gold_01's full-scaled Parquet and its fit-normal-only Parquet, then trains the Isolation Forest exclusively on normal-condition rows. It scores the complete dataset, calibrates the anomaly threshold from training scores alone (keeping test scores unseen during calibration), and outputs a scored results frame, a saved model artifact, per-run threshold and summary JSONs, a metadata JSON recording full Gold provenance, a `gold_baseline` truth record, and scored results written to the `gold.anomaly_detection_scores` SQL table.

## Pipeline Role

- Stage: `gold_baseline`
- Layer: `gold`
- W&B job type: `gold_modeling_baseline`
- Position in workflow: First Gold modeling notebook; immediately follows Gold_01 preprocessing
- Primary responsibility: Train baseline Isolation Forest on normal-only fit data; score the full Gold dataset; calibrate and apply the anomaly threshold; write scored results, model, and truth record

## Inputs

| Input | Source | Expected Form | Used For |
|---|---|---|---|
| `GOLD_PREPROCESSED_SCALED_DATA_PATH` | Gold_01 output Parquet (configured path, resolved from `RESOLVED_PATHS`) | Parquet → DataFrame via `load_data()` | Primary modeling input; all-rows feature matrix and results frame base |
| `GOLD_FIT_DATA_PATH` | Gold_01 output Parquet (resolved path **overridden** from Gold truth `artifact_paths["gold_fit_path"]`) | Parquet → DataFrame via `load_data()` | Normal-only training rows used to fit the Isolation Forest |
| `STAGE1_FEATURES_PATH` | Gold_01 artifact JSON (resolved path **overridden** from Gold truth `artifact_paths["stage1_features_path"]`) | JSON list of column names | Stage 1 feature column set used for all feature matrices |
| Gold_01 truth record | Loaded from `TRUTHS_PATH/gold/` via `meta__truth_hash` column in scaled Parquet | JSON truth record | Parent hash, dataset name, pipeline mode, artifact path overrides, runtime facts |
| `CONFIG`, `RESOLVED_PATHS`, `FILENAMES` | `load_notebook_context()` bootstrap | Config dict, path dict, filename dict | All path resolution, model hyperparameters, threshold config |
| `GOLD_CFG` | Nested section of `CONFIG` | Config dict | `baseline_estimator_count`, `baseline_threshold_percentile`, `train_fraction`, `random_seed`, `layer_name`, `recipe_id`, `process_run_id_prefix` |

## Configuration and Runtime Context

| Item | Source | Purpose |
|---|---|---|
| `BASELINE_ESTIMATOR_COUNT` | `GOLD_CFG["baseline_estimator_count"]` | Number of trees in the Isolation Forest |
| `BASELINE_THRESHOLD_PERCENTILE` | `GOLD_CFG["baseline_threshold_percentile"]` | Percentile of training anomaly scores used to set the alert threshold |
| `RANDOM_SEED` | `GOLD_CFG["random_seed"]` | Ensures Isolation Forest tree ensemble is reproducible across runs |
| `TRAIN_FRACTION` | `GOLD_CFG["train_fraction"]` | Recorded in truth record runtime_facts; split is read from Gold_01, not re-derived |
| `DATASET_NAME` | Initial: `DATASET_CFG.get("name")` → overridden by `get_dataset_name_from_truth(gold_truth)` | Resolved from Gold parent truth so it always matches Gold_01's exact run |
| `PIPELINE_MODE` | Initial: from `PIPELINE["execution_mode"]` → overridden by `get_pipeline_mode_from_truth(gold_truth)` if non-None | Inherited from Gold_01's pipeline mode |
| `GOLD_PARENT_TRUTH_HASH` | `get_truth_hash(gold_truth)` | Gold_01's truth hash; becomes the parent hash in Gold_02's truth record |
| `GOLD_FIT_DATA_PATH` | Initial: from `RESOLVED_PATHS` → overridden from `gold_truth["artifact_paths"]["gold_fit_path"]` | Ensures Gold_02 uses the exact fit Parquet saved by Gold_01, not a stale configured path |
| `STAGE1_FEATURES_PATH` | Initial: from `RESOLVED_PATHS` → overridden from `gold_truth["artifact_paths"]["stage1_features_path"]` | Ensures Gold_02 uses the exact feature list saved by Gold_01 |
| `gold_truth_runtime_facts` | `gold_truth.get("runtime_facts", {})` | Carries forward Gold_01 provenance (e.g., `feature_set_id`, `scaler_kind_runtime`) into baseline metadata |
| `gold_truth_artifact_paths` | `gold_truth.get("artifact_paths", {})` | Source of path overrides; also recorded in baseline metadata for downstream traceability |
| `TRUTH_VERSION`, `GOLD_VERSION` | `VERSIONS_CFG` | Written into truth record metadata |
| `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME` | `WANDB_CFG`, `GOLD_VERSION` | W&B experiment tracking; run is ACTIVE in this notebook |

## Logical Workflow Map

1. Bootstrap: `load_notebook_context()`, path/config extraction, W&B init, SQL smoke check
2. Data load and Gold truth propagation: load Gold_01 scaled Parquet, load Gold parent truth, override artifact paths from truth record, load Gold_01 fit Parquet, load Stage 1 feature list
3. Train/test split recovery: read `meta__is_train_flag` column from Gold_01's Parquet
4. Optional label handling: conditionally extract `anomaly_flag` arrays for evaluation
5. Feature matrix assembly: select Stage 1 feature columns from full and fit DataFrames
6. Model fit: train `IsolationForest` on normal-only fit rows
7. Initial scoring and threshold calibration: score all rows, calibrate threshold from training scores, compute metrics
8. Row-tracked scoring: re-score via `score_isolation_forest_stage` helper for per-row lineage; preserve both score directions
9. Results synchronization: rebuild canonical results frame, reapply alert rule, confirm zero-alert guard
10. Output validation: check required columns and `meta__row_id` uniqueness
11. Truth record build: initialize, populate runtime_facts and artifact_paths, build, stamp columns, save, index
12. Artifact saves: results CSV/pickle, model (two paths), threshold/summary/metadata JSONs, ledger, W&B saves
13. Detected rows export: extract flagged rows via `get_detected_rows_dataframe`
14. Ledger close and W&B finish: `ledger.write_json`, `wandb_run.finish()`
15. Final lineage verification: meta column presence, truth hash cross-check, parent hash uniqueness
16. SQL write: `write_gold_baseline_scores_sql` → `gold.anomaly_detection_scores`

## Section Overview

| Section | Purpose | Key Inputs | Key Outputs / Side Effects |
|---|---|---|---|
| Bootstrap | Initialize notebook runtime context | `load_notebook_context()`, `CONFIG`, `WANDB_CFG` | `CTX`, `logger`, `ledger`, `RESOLVED_PATHS`, `wandb_run` active |
| Data load and Gold truth | Load Gold_01 outputs; inherit parent truth; override artifact paths | `GOLD_PREPROCESSED_SCALED_DATA_PATH`, Gold truth record | `gold_preprocessed_scaled_dataframe`, `gold_fit_dataframe`, `stage1_feature_columns`, `GOLD_PARENT_TRUTH_HASH`, path overrides |
| Split recovery | Recover train/test split from Gold_01's stamped column | `meta__is_train_flag` in scaled Parquet | `train_mask`, `test_mask` |
| Label handling | Conditionally prepare anomaly label arrays | `anomaly_flag` column (optional) | `all_labels`, `test_labels` (None if absent) |
| Feature matrix | Assemble DataFrames from Stage 1 feature list | `stage1_feature_columns`, scaled and fit DataFrames | `baseline_train_fit_features`, `baseline_all_features`, `baseline_test_features` |
| Model fit | Train Isolation Forest on normal-only fit rows | `baseline_train_fit_features` | `baseline_model` fitted |
| Initial scoring | Score all rows; calibrate threshold; compute baseline metrics | `baseline_model`, feature matrices, `BASELINE_THRESHOLD_PERCENTILE` | `baseline_train_scores`, `baseline_all_scores`, `baseline_threshold`, `baseline_metrics` |
| Row-tracked scoring | Re-score with per-row tracking helper; preserve dual score columns | `baseline_model`, `gold_preprocessed_scaled_dataframe`, `meta__row_id` | `scored_gold_dataframe` with `baseline_score`, `baseline_score_samples_raw`, `baseline_decision`, `baseline_pred`, `baseline_flag` |
| Results sync | Rebuild results frame on canonical score direction; re-confirm alert rule | `scored_gold_dataframe`, `baseline_threshold` | `baseline_results` with scored columns; zero-alert guard |
| Output validation | Verify results frame integrity | `baseline_results`, `test_mask` | `validate_baseline_output` pass or `ValueError` |
| Truth record | Build and stamp `gold_baseline` truth record linked to Gold_01's hash | `GOLD_PARENT_TRUTH_HASH`, runtime_facts, artifact_paths | `baseline_truth_record`, `BASELINE_TRUTH_HASH`, stamped `baseline_results` |
| Artifact saves | Persist all baseline outputs | `baseline_results`, `baseline_model`, truth record | Results CSV/pickle, model joblib ×2, threshold/summary/metadata JSONs, truth record, `wandb.save` all |
| Detected rows | Extract alert-flagged rows for downstream traceability | `baseline_results`, `baseline_flag` | `baseline_detected_rows_dataframe` |
| Ledger and W&B close | Finalize run record | Full session ledger | `baseline_ledger_path` written; W&B run closed |
| Final lineage checks | Assert truth hash consistency and parent hash uniqueness | `baseline_results`, `BASELINE_TRUTH_HASH`, `GOLD_PARENT_TRUTH_HASH` | `ValueError` on any mismatch |
| SQL write | Write scored results to Postgres Gold layer | `baseline_results`, `engine`, `CAPSTONE_SCHEMA` | Rows inserted to `gold.anomaly_detection_scores` |

## Section Details

### Bootstrap

#### Purpose

Loads the notebook execution context, resolves all paths and filenames, initializes the W&B run, and performs an early SQL connectivity smoke check.

#### Key Operations

- `load_notebook_context()` populates `CTX`; extracts `CONFIG`, `RESOLVED_PATHS`, `FILENAMES`, `logger`, `ledger`, `paths`, `engine`, `PIPELINE`, `RUNTIME_CFG`, `DATASET_CFG`, `GOLD_CFG`, `WANDB_CFG`, `VERSIONS_CFG`
- Config extraction provides all working constants from config and resolved paths
- `GOLD_PROCESS_RUN_ID = make_process_run_id(GOLD_CFG["process_run_id_prefix"])` — unique run stamp generated at launch
- `TRUTH_CONFIG = build_truth_config_block(CONFIG)` — config snapshot object written into truth record
- `wandb.init(job_type="gold_modeling_baseline", ...)` — W&B run opened immediately; active for the full session
- All output directories created with `mkdir(parents=True, exist_ok=True)` for failsafe path resolution
- SQL smoke check via `read_layer_dataframe` or equivalent connectivity probe

#### In-Place Usage Notes

All downstream path resolution happens once in cell 9 from `RESOLVED_PATHS`. Two key paths — `GOLD_FIT_DATA_PATH` and `STAGE1_FEATURES_PATH` — carry configured values from this cell but are later overridden by the Gold parent truth record. The effective runtime values come from the truth overrides, not from the initial cell 9 assignments.

---

### Data Load and Gold Truth Propagation

#### Purpose

Loads Gold_01's full-scaled Parquet, extracts the dataset name from the `meta__dataset` column, then loads the parent Gold truth record using the `meta__truth_hash` column. After loading the truth record, resolves the authoritative values for `DATASET_NAME`, `GOLD_PARENT_TRUTH_HASH`, and `PIPELINE_MODE`, and overrides the configured fit Parquet path and Stage 1 features path to match what Gold_01 actually saved.

#### Key Operations

- `gold_preprocessed_scaled_dataframe = load_data(GOLD_PREPROCESSED_SCALED_DATA_PATH)` — loads Gold_01's full scaled output
- `GOLD_DATASET_NAME` extracted from `meta__dataset` column; `ValueError` if empty
- `gold_truth = load_parent_truth_record_from_dataframe(dataframe=gold_preprocessed_scaled_dataframe, truth_dir=TRUTHS_PATH, parent_layer_name="gold", dataset_name=GOLD_DATASET_NAME, column_name="meta__truth_hash")`
- `DATASET_NAME = get_dataset_name_from_truth(gold_truth)` — overrides initial config value
- `GOLD_PARENT_TRUTH_HASH = get_truth_hash(gold_truth)` — Gold_01's hash; becomes parent in Gold_02's truth chain
- `PARENT_PIPELINE_MODE = get_pipeline_mode_from_truth(gold_truth)` → if non-None, overrides `PIPELINE_MODE`
- `GOLD_FIT_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_fit_path", str(GOLD_FIT_DATA_PATH)))` — path override from Gold_01's truth
- `STAGE1_FEATURES_PATH = Path(gold_truth_artifact_paths.get("stage1_features_path", str(STAGE1_FEATURES_PATH)))` — path override from Gold_01's truth
- `gold_truth_runtime_facts = gold_truth.get("runtime_facts", {})` — carries Gold_01's feature_set_id, scaler metadata, imputation recommendation forward
- `gold_truth_artifact_paths = gold_truth.get("artifact_paths", {})` — source of path overrides and downstream metadata records
- `GOLD_TRUTH_PATH` constructed from `TRUTHS_PATH / "gold" / f"{DATASET_NAME}__gold__truth__{GOLD_PARENT_TRUTH_HASH}.json"`
- `gold_fit_dataframe = load_data(GOLD_FIT_DATA_PATH)` — loads fit-normal Parquet from truth-linked path
- `stage1_feature_columns = load_json(STAGE1_FEATURES_PATH)` — normalized list of feature column names; `ValueError` if None or empty

#### In-Place Usage Notes

The path overrides are the key lineage mechanism in this section. Gold_02 does not use its configured `GOLD_FIT_DATA_PATH` and `STAGE1_FEATURES_PATH` directly — it substitutes the values recorded in Gold_01's truth record. This guarantees that even if config is updated between runs, Gold_02 always uses the exact fit data and feature list that Gold_01 computed, not a potentially different version. The `gold_truth_runtime_facts` dict is also carried forward directly into the baseline metadata JSON, recording Gold_01's scaler kind, scaler path, recommended imputation method, and feature_set_id alongside Gold_02's own outputs.

---

### Train/Test Split Recovery

#### Purpose

Recovers the train/test partition stamped by Gold_01 into the `meta__is_train_flag` column. Gold_02 does not re-derive the split — it reads the decision from Gold_01's output to keep the partition definition consistent across all Gold modeling notebooks.

#### Key Operations

- `if "meta__is_train_flag" not in gold_preprocessed_scaled_dataframe.columns: raise ValueError(...)` — hard guard; pipeline cannot proceed without the Gold_01 split stamp
- `train_mask = gold_preprocessed_scaled_dataframe["meta__is_train_flag"].fillna(False).astype(bool)`
- `test_mask = (~train_mask).astype(bool)`
- `test_mask_array = test_mask.to_numpy()` — used for NumPy-aligned subsetting in scoring

#### In-Place Usage Notes

The `ValueError` on a missing `meta__is_train_flag` is an intentional design constraint: if the column is absent, Gold_01 preprocessing did not complete correctly and the modeling stage should not run. The masks are used throughout the notebook to separate training-set scores from test-set scores, to evaluate metrics on the test partition, and to construct the detected-rows export.

---

### Optional Label Handling

#### Purpose

Conditionally extracts ground-truth anomaly label arrays from the `anomaly_flag` column. Labels are used only if present; the notebook proceeds without evaluation metrics if they are absent.

#### Key Operations

- `if "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns`: extracts `all_labels` and `test_labels` as integer NumPy arrays
- If absent: `all_labels = None`, `test_labels = None`
- All evaluation paths that call `evaluate_against_labels` are guarded by `if test_labels is not None`

---

### Feature Matrix Assembly

#### Purpose

Builds the three DataFrame subsets that all scoring operations use. Feature matrices are kept as DataFrames (not converted to NumPy arrays) to preserve column names and avoid the sklearn feature-name mismatch warning.

#### Key Operations

- Validates that all `stage1_feature_columns` are present in both `gold_preprocessed_scaled_dataframe` and `gold_fit_dataframe`; `ValueError` on any missing column
- `baseline_train_fit_features = gold_fit_dataframe[stage1_feature_columns]` — model fit input (normal-only rows from Gold_01)
- `baseline_all_features = gold_preprocessed_scaled_dataframe[stage1_feature_columns]` — scoring input for all rows
- `baseline_test_features = gold_preprocessed_scaled_dataframe.loc[test_mask, stage1_feature_columns]` — test-only scoring input

#### In-Place Usage Notes

`baseline_train_fit_features` comes from `gold_fit_dataframe`, not from a `train_mask` applied to the scaled Parquet. This is important: Gold_01's fit Parquet contains only the normal-condition training rows after anomaly exclusion; it is a stricter subset than the rows identified by `train_mask`. The distinction matters because the model learns only from confirmed normal behavior.

---

### Model Fit

#### Purpose

Trains the baseline Isolation Forest on normal-only fit rows. Using `RANDOM_SEED` ensures the tree ensemble is fully reproducible across runs, keeping artifact hashes stable.

#### Key Operations

- `baseline_model = IsolationForest(n_estimators=BASELINE_ESTIMATOR_COUNT, random_state=RANDOM_SEED, n_jobs=-1)`
- `baseline_model.fit(baseline_train_fit_features)`

---

### Initial Scoring and Threshold Calibration

#### Purpose

Scores training and all-rows feature matrices with the fitted model, calibrates the anomaly threshold from training scores only, and computes an initial set of evaluation metrics.

#### Key Operations

- `compute_anomaly_scores_isolation_forest(model, features)` returns `-model.score_samples(features)` so that higher values mean more anomalous
- `baseline_train_scores` — scores on the normal-only fit rows; used exclusively for threshold calibration
- `baseline_all_scores` — scores for all rows; length validated against full dataframe before use
- `baseline_threshold = choose_threshold_by_percentile(baseline_train_scores, BASELINE_THRESHOLD_PERCENTILE)` — threshold derived from training distribution only; test scores are unseen at calibration time
- `baseline_flags = (baseline_all_scores >= baseline_threshold).astype(int)` — binary alert array
- `baseline_test_scores`, `baseline_test_flags` — test-partition subsets via `test_mask_array`
- `baseline_metrics` dict: model name, threshold percentile, threshold value, alert counts; extended with `evaluate_against_labels(test_labels_array, baseline_test_scores, baseline_threshold)` if labels exist
- `evaluate_against_labels` returns precision, recall, F1, and ROC-AUC; adds PR-AUC when both label classes present

#### In-Place Usage Notes

The calibration constraint — threshold from training-only scores — is an explicit design requirement encoded in the source comment: "Threshold is derived from the normal-only training scores only, keeping test scores unseen during calibration." Setting the threshold from `baseline_train_scores` (normal-condition fit rows) rather than from all-rows scores ensures the threshold reflects normal operational behavior, not an average that blends normal and anomalous rows.

---

### Row-Tracked Scoring

#### Purpose

Re-scores the full Gold dataset through a project-level row-tracking helper that pairs each output row to its `meta__row_id`. This produces both the native IsolationForest score direction (raw `score_samples`) and additional decision-boundary outputs (`decision_function`, `predict`) alongside the project-defined anomaly score.

#### Key Operations

- `build_stage_scoring_frame(dataframe=source_gold_dataframe, feature_columns=baseline_feature_columns, mask=None, row_id_column="meta__row_id")` — prepares a row-indexed input frame
- `score_isolation_forest_stage(stage_dataframe, model=baseline_model, feature_columns=baseline_feature_columns, stage_name="baseline", row_id_column="meta__row_id")` — produces a tracking frame with stage-prefixed output columns
- Renames helper output columns before merging to avoid collision with the project score convention:
  - `baseline_score` → `baseline_score_samples_raw` (raw `score_samples`, higher = more normal)
  - `baseline_flag` → `baseline_predict_flag` (IsolationForest `predict`-based flag)
- Drops any stale baseline columns from `scored_gold_dataframe` before the merge to prevent duplicate `_x`/`_y` columns on re-execution
- Merges tracking columns back on `meta__row_id`: left join so all rows are retained
- Recomputes canonical project anomaly score: `baseline_all_scores = -model.score_samples(baseline_all_features)`
- Adds `baseline_decision = model.decision_function(baseline_all_features)` and `baseline_pred = model.predict(baseline_all_features)`
- Reapplies `baseline_flag = (baseline_score >= baseline_threshold).astype(int)`
- Updates the shared `gold_preprocessed_scaled_dataframe` reference via `globals()[...]` so the canonical scored frame is accessible in subsequent cells

#### In-Place Usage Notes

The scoring section preserves two parallel score representations: `baseline_score_samples_raw` retains the raw `score_samples` output for any consumer that needs the IsolationForest's native direction, while `baseline_score` is always the project convention (`-score_samples`, higher = more anomalous). This dual-column design avoids the confusion documented in the source comment: "The project baseline score uses the opposite direction: baseline_score = -model.score_samples(...). So this cell preserves the helper's raw score_samples output separately and restores baseline_score to the project-defined anomaly score."

---

### Results Synchronization

#### Purpose

Constructs the canonical `baseline_results` DataFrame from the scored Gold frame, locks in the final column set, re-evaluates metrics on the synchronized results, and applies a zero-alert guard.

#### Key Operations

- `baseline_results = gold_preprocessed_scaled_dataframe.copy()` — preserves all meta and lineage columns from the scored frame
- Adds: `baseline_score`, `baseline_threshold`, `baseline_threshold_percentile`, `baseline_flag`
- If labels present: re-runs `evaluate_against_labels` on synchronized test scores and updates `baseline_metrics`
- `if baseline_alert_count_all_rows == 0: raise ValueError(...)` — guards against the entire run having zero alerts, which indicates the wrong score direction was used

---

### Output Validation

#### Purpose

Confirms that `baseline_results` meets minimum structural requirements before truth record creation and artifact saves proceed.

#### Key Operations

- `validate_baseline_output(results_dataframe=baseline_results, test_mask=test_mask, label_column="anomaly_flag", flag_column="baseline_flag", score_column="baseline_score", row_id_column="meta__row_id", episode_column="episode_column")` — project-specific validation function
- Checks: required columns present (`meta__row_id`, `baseline_flag`, `baseline_score`, `meta__is_train_flag`); test_mask length matches dataframe length; `meta__row_id` uniqueness across all rows

---

### Truth Record Build

#### Purpose

Initializes the `gold_baseline` truth record with all runtime facts, artifact paths, and upstream lineage; builds the hash; stamps truth columns into `baseline_results`; saves the record and updates the truth index.

#### Key Operations

- `initialize_layer_truth(truth_version=TRUTH_VERSION, dataset_name=DATASET_NAME, layer_name="gold_baseline", process_run_id=baseline_process_run_id, pipeline_mode=PIPELINE_MODE, parent_truth_hash=GOLD_PARENT_TRUTH_HASH)` — parent hash is Gold_01's truth hash
- `update_truth_section(..., "config_snapshot", truth_config_snapshot)` — includes stage, dataset, mode, profile; falls back to a minimal inline dict if `TRUTH_CONFIG` is not a dict
- `update_truth_section(..., "runtime_facts", {...})` — records: `baseline_threshold_percentile`, `baseline_threshold`, `baseline_estimator_count`, `train_fraction`, `random_seed`, `alert_count_all_rows`, `alert_count_test_rows`, `result_row_count`, `parent_truth_hash`, `gold_process_run_id` (from `gold_truth.get("process_run_id")`), `gold_feature_set_id` (from `gold_truth_runtime_facts.get("feature_set_id")`)
- `update_truth_section(..., "artifact_paths", {...})` — records paths for: gold truth, gold scaled Parquet, gold fit Parquet, baseline results CSV/pickle, model artifact path, models path, thresholds, summary, metadata
- `build_truth_record(truth_base=baseline_truth, row_count=len(baseline_results), column_count=baseline_results.shape[1] + 3, meta_columns=baseline_meta_columns, feature_columns=baseline_feature_columns)` → `BASELINE_TRUTH_HASH`
- `stamp_truth_columns(baseline_results, truth_hash=BASELINE_TRUTH_HASH, parent_truth_hash=GOLD_PARENT_TRUTH_HASH, pipeline_mode=PIPELINE_MODE)` — writes `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` into every row of `baseline_results`
- `save_truth_record(baseline_truth_record, truth_dir=TRUTHS_PATH, dataset_name=DATASET_NAME, layer_name="gold_baseline")` → `baseline_truth_path`
- `append_truth_index(baseline_truth_record, truth_index_path=TRUTH_INDEX_PATH)` — adds entry to cross-stage truth index

#### In-Place Usage Notes

The `gold_process_run_id` and `gold_feature_set_id` fields carried from `gold_truth` into Gold_02's `runtime_facts` mean that any downstream notebook reading this truth record can trace backward to exactly which Gold_01 run produced the input data and which feature set was selected. The `baseline_process_run_id` may be inherited from `GOLD_PROCESS_RUN_ID` (generated at bootstrap from config) or generated fresh via `make_process_run_id("gold_baseline_process")` if `GOLD_PROCESS_RUN_ID` is not a populated string.

---

### Artifact Saves

#### Purpose

Persists all baseline outputs — results, model, metrics JSONs, truth record, and metadata — to disk and registers all paths with W&B.

#### Key Operations

- `baseline_results.to_csv(BASELINE_RESULTS_PATH_CSV, index=False)` — scored results with all meta and lineage columns
- `baseline_results.to_pickle(BASELINE_RESULTS_PATH_PICKLE)` — same frame in pickle format for downstream Python consumers
- `joblib.dump(baseline_model, BASELINE_MODEL_ARTIFACT_PATH)` — primary model artifact path (registered in truth artifact_paths)
- `joblib.dump(baseline_model, BASELINE_MODELS_PATH)` — secondary path under the models root directory
- `save_json(baseline_thresholds, BASELINE_THRESHOLDS_PATH)` — `{"baseline_threshold_percentile": ..., "baseline_threshold": ...}`
- `save_json(baseline_summary, BASELINE_SUMMARY_PATH)` — dataset name, metrics, alert counts, truth hash, process run ID, gold truth hash, gold feature_set_id
- `save_json(baseline_metadata, BASELINE_METADATA_PATH)` — full provenance: all artifact paths for both Gold_01 and Gold_02 inputs/outputs; `gold_scaler_path`, `gold_scaler_kind`, `gold_recommended_imputation`, `gold_feature_set_id` from Gold_01's truth record
- All seven artifact files registered with `wandb.save(...)` including the truth record
- `ledger.add(kind="step", step="save_baseline_outputs", ...)` records all artifact paths and alert counts

---

### Detected Rows Export

#### Purpose

Extracts only the alert-flagged rows from `baseline_results` into a separate DataFrame for traceability and downstream inspection.

#### Key Operations

- `get_detected_rows_dataframe(dataframe=baseline_results, target_flag_column="baseline_flag", row_id_column="meta__row_id", score_column="baseline_score", decision_column="baseline_decision", pred_column="baseline_pred", include_columns=["baseline_flag", "anomaly_flag"], sort_by="time_index", ascending=True)` → `baseline_detected_rows_dataframe`
- `ledger.add(kind="step", step="extract_baseline_detected_rows", ...)` records detected row count and available sort/time columns

---

### Ledger and W&B Close

#### Purpose

Finalizes the session run record and closes the W&B experiment run.

#### Key Operations

- `ledger.write_json(baseline_ledger_path)` — writes the full sequential step log for this notebook run
- `wandb.save(str(baseline_ledger_path))` — registers ledger with W&B
- `wandb_run.finish()` — closes the W&B run; no further W&B calls are valid after this point

---

### Final Lineage Verification

#### Purpose

Confirms that truth hash and parent hash columns are correctly stamped into `baseline_results` before SQL write. Catches any late-stage column drop or hash mismatch that would produce a corrupt downstream record.

#### Key Operations

- Checks presence of `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` in `baseline_results`; `ValueError` on missing
- `extract_truth_hash(baseline_results)` → cross-checks against `BASELINE_TRUTH_HASH`; `ValueError` on mismatch
- `baseline_results["meta__parent_truth_hash"].dropna().astype(str).unique()` — must be exactly one value; `ValueError` if empty or multiple values; `ValueError` if it does not match `GOLD_PARENT_TRUTH_HASH`

---

### SQL Write

#### Purpose

Writes the scored results to the `gold.anomaly_detection_scores` table in Postgres.

#### Key Operations

- `WRITE_TO_POSTGRES = True` — gate is enabled by default in this notebook
- `write_gold_baseline_scores_sql(engine=engine, capstone_schema=CAPSTONE_SCHEMA, dataset_id=DATASET_ID, run_id=RUN_ID, notebook_globals=globals(), dataset_name=DATASET_NAME)` → summary display
- Target schema/table: `gold.anomaly_detection_scores`

## Key Function Calls and In-Place Usage

| Function / Method | Context in This Notebook | Inputs Provided Here | Return / Side Effect |
|---|---|---|---|
| `load_notebook_context()` | Bootstrap | (none) | `CTX`, logger, ledger, resolved paths, config, Postgres engine |
| `load_data(path)` | Load Gold_01 scaled Parquet and fit Parquet | `GOLD_PREPROCESSED_SCALED_DATA_PATH`, `GOLD_FIT_DATA_PATH` | DataFrames `gold_preprocessed_scaled_dataframe`, `gold_fit_dataframe` |
| `load_parent_truth_record_from_dataframe(...)` | Load Gold_01 truth record via hash column | `parent_layer_name="gold"`, `column_name="meta__truth_hash"` | `gold_truth` dict; source of all path overrides and parent hash |
| `get_truth_hash(gold_truth)` | Extract Gold_01's hash for parent linkage | `gold_truth` | `GOLD_PARENT_TRUTH_HASH` |
| `get_dataset_name_from_truth(gold_truth)` | Override `DATASET_NAME` from truth | `gold_truth` | Resolved `DATASET_NAME` |
| `get_pipeline_mode_from_truth(gold_truth)` | Inherit pipeline mode from Gold_01 | `gold_truth` | `PARENT_PIPELINE_MODE`; if non-None, overrides `PIPELINE_MODE` |
| `load_json(STAGE1_FEATURES_PATH)` | Load Stage 1 feature column list | Truth-overridden path | `stage1_feature_columns` list |
| `IsolationForest.fit(baseline_train_fit_features)` | Train model on normal-only fit rows | `n_estimators`, `random_state=RANDOM_SEED`, `n_jobs=-1` | `baseline_model` fitted in place |
| `compute_anomaly_scores_isolation_forest(model, features)` | Compute anomaly scores (project convention) | `baseline_model`, feature DataFrame | `-model.score_samples(features)` — higher = more anomalous |
| `choose_threshold_by_percentile(scores, percentile)` | Calibrate threshold from training scores only | `baseline_train_scores`, `BASELINE_THRESHOLD_PERCENTILE` | `baseline_threshold` float |
| `evaluate_against_labels(true_labels, scores, threshold)` | Evaluate on test partition if labels present | `test_labels_array`, `baseline_test_scores`, `baseline_threshold` | Precision, recall, F1, ROC-AUC, PR-AUC dict |
| `build_stage_scoring_frame(...)` | Prepare row-indexed input for scoring helper | `gold_preprocessed_scaled_dataframe`, `baseline_feature_columns`, `row_id_column="meta__row_id"` | `baseline_stage_input_df` aligned to row IDs |
| `score_isolation_forest_stage(...)` | Add row-tracked score, decision, pred columns | `baseline_model`, `baseline_feature_columns`, `stage_name="baseline"` | `baseline_stage_results_df` with `baseline_score`, `baseline_decision`, `baseline_pred`, `baseline_flag` columns |
| `validate_baseline_output(...)` | Structural guardrail on results frame | `baseline_results`, `test_mask`, column names | Pass or `ValueError` |
| `initialize_layer_truth(...)` | Start `gold_baseline` truth record | `layer_name="gold_baseline"`, `parent_truth_hash=GOLD_PARENT_TRUTH_HASH` | `baseline_truth` dict with parent linkage |
| `build_truth_record(...)` | Finalize truth record and generate hash | `baseline_truth`, row/column counts, meta and feature columns | `baseline_truth_record` dict; `BASELINE_TRUTH_HASH` |
| `stamp_truth_columns(...)` | Write truth hash and lineage into all rows | `baseline_results`, `BASELINE_TRUTH_HASH`, `GOLD_PARENT_TRUTH_HASH` | `baseline_results` updated with `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` |
| `save_truth_record(...)` | Persist truth JSON | `baseline_truth_record`, `TRUTHS_PATH`, `layer_name="gold_baseline"` | Truth file written; `baseline_truth_path` returned |
| `append_truth_index(...)` | Add to cross-stage truth index | `baseline_truth_record`, `TRUTH_INDEX_PATH` | `truth_index.jsonl` updated |
| `joblib.dump(baseline_model, path)` | Persist fitted model | `BASELINE_MODEL_ARTIFACT_PATH` and `BASELINE_MODELS_PATH` | Model saved to two paths |
| `get_detected_rows_dataframe(...)` | Extract flagged rows for traceability | `baseline_results`, `target_flag_column="baseline_flag"`, `sort_by="time_index"` | `baseline_detected_rows_dataframe` |
| `ledger.write_json(baseline_ledger_path)` | Finalize session step log | (session accumulation) | JSONL ledger written to disk |
| `wandb_run.finish()` | Close W&B experiment run | (active run) | W&B run closed; further saves invalid |
| `write_gold_baseline_scores_sql(...)` | Write scored results to Postgres | `engine`, `CAPSTONE_SCHEMA`, `DATASET_ID`, `RUN_ID`, `globals()` | Rows inserted to `gold.anomaly_detection_scores` |

## Outputs and Artifacts

| Output | Type | Location / Destination | Downstream Consumer |
|---|---|---|---|
| Baseline results | CSV | `BASELINE_RESULTS_PATH_CSV` | Gold_04 comparison, Gold_05 anomaly detection |
| Baseline results | Pickle | `BASELINE_RESULTS_PATH_PICKLE` | Gold_04 comparison, Gold_05 anomaly detection |
| Baseline model | Joblib | `BASELINE_MODEL_ARTIFACT_PATH` (primary), `BASELINE_MODELS_PATH` (models root) | Gold_04 comparison |
| Threshold JSON | JSON | `BASELINE_THRESHOLDS_PATH` | Gold_04 comparison |
| Summary JSON | JSON | `BASELINE_SUMMARY_PATH` | Gold_04 comparison |
| Metadata JSON | JSON | `BASELINE_METADATA_PATH` | Gold_04, Gold_05 provenance traceability |
| `gold_baseline` truth record | JSON | `TRUTHS_PATH/gold_baseline/` | Gold_04 parent truth chain |
| Truth index entry | JSONL append | `TRUTH_INDEX_PATH` | Cross-stage audit |
| Baseline ledger | JSONL | `baseline_ledger_path` | Run audit, W&B |
| Scored results table | SQL rows | `gold.anomaly_detection_scores` | Operational monitoring, reporting |
| W&B run artifacts | W&B | All 7 artifact files + truth record + ledger | Experiment tracking |

## Data Quality / Validation Behavior

| Check | Purpose | Failure / Risk Prevented |
|---|---|---|
| `meta__is_train_flag` column present | Ensure Gold_01 stamped the split before modeling begins | Raises `ValueError`; prevents re-deriving a different split that would silently break evaluation |
| `GOLD_DATASET_NAME` non-empty from `meta__dataset` | Confirm Gold_01 populated dataset identity | Raises `ValueError`; prevents empty or wrong dataset name propagating to truth records |
| `load_parent_truth_record_from_dataframe` succeeds | Confirm Gold_01 truth is readable and hash resolves | Raises if truth file not found; prevents running on stale or uncertified Gold_01 data |
| Stage 1 feature presence in both DataFrames | Confirm feature columns exist in scaled and fit frames | Raises `ValueError` with list of missing columns; prevents silent shape mismatch at fit/score time |
| Score length vs dataframe length | Guard after all-rows scoring | Raises `ValueError` on mismatch; prevents misaligned score-to-row assignment |
| Zero-alert guard | Detect wrong score direction | Raises `ValueError("Baseline produced zero alerts...")`; catches score-direction inversion errors |
| `validate_baseline_output(...)` | Structural check on results frame | Raises `ValueError` on missing required columns or non-unique `meta__row_id` |
| Final `meta__truth_hash` cross-check | Confirm stamped hash matches computed hash | Raises `ValueError` on any mismatch; catches late-stage column overwrites or copy errors |
| Final `meta__parent_truth_hash` uniqueness | Confirm all rows share one parent hash | Raises `ValueError` if empty or if multiple values found; prevents partial-stamp or merge-artifact corruption |

## Downstream Handoff

Gold_03a (Cascade Stage 1 modeling) and Gold_04 (Comparison) each receive the `BASELINE_TRUTH_HASH` as their upstream anchor. Gold_04 reads the baseline results CSV/pickle and the `baseline_summary.json` and `baseline_thresholds.json` to construct a side-by-side comparison against the cascade models' outputs. Gold_05 reads the scored results for anomaly detection analysis.

The `baseline_metadata.json` records the full Gold_01 provenance chain — scaler path, scaler kind, imputation method, feature_set_id — alongside Gold_02's own artifacts, so any comparison or audit notebook can trace the exact preprocessing run that produced the modeling inputs without re-reading the Gold_01 truth record directly.

The `gold.anomaly_detection_scores` SQL rows make the baseline scores available to the operational monitoring layer independently of the file-based artifact path.

---

## Relationship to Other Notebooks

### Upstream Context

Gold_02 loads Gold_01's scaled Parquet (`GOLD_PREPROCESSED_SCALED_DATA_PATH`) and Gold_01's truth record to resolve 5 Parquet paths and 2 feature JSON lists. It inherits `GOLD_PARENT_TRUTH_HASH` from the Gold_01 truth record. No dependency on Gold_03a, Gold_03b, or Gold_03c.

### Downstream Handoff

Gold_02 provides:
- Baseline results CSV and pickle consumed by Gold_04_Comparison as one of four model comparison inputs
- `BASELINE_TRUTH_HASH` validated by Gold_04 via three-source check
- Baseline model joblib and validation contract consumed by Gold_06A_Test_Replay_Validation for test replay
- SQL rows in `gold.anomaly_detection_scores` with `model_stage="baseline_final"` for the operational layer

### Pipeline Position

First Gold modeling notebook. Establishes a single Isolation Forest baseline with threshold calibration. All subsequent cascade modeling and comparison notebooks treat Gold_02's results as the reference baseline. Runs independently of Gold_03x notebooks.

### Relationship Summary

- Reads Gold_01 scaled Parquet and truth record (8 path overrides); no dependency on Gold_03x
- Produces baseline results consumed by Gold_04 and model artifacts consumed by Gold_06A
- `BASELINE_TRUTH_HASH` registered and three-source-validated in Gold_04 alongside cascade hashes
- No stage-to-stage artifact dependency with Gold_03a, Gold_03b, or Gold_03c
- SQL rows persisted to `gold.anomaly_detection_scores` for the operational monitoring layer

## Notes / Risks / Deferred Cleanup

- `GOLD_FIT_DATA_PATH` and `STAGE1_FEATURES_PATH` are configured in cell 9 but overridden from the Gold parent truth record in cell 34. The effective runtime paths are always from the truth overrides. The cell 9 values serve as fallbacks only if the truth record's `artifact_paths` does not contain the key.
- `baseline_score_samples_raw` stores the raw IsolationForest `score_samples` output (higher = more normal). `baseline_score` stores the project convention (higher = more anomalous). Both columns are present in `baseline_results` for consumers that need either direction.
- The model is saved to two paths: `BASELINE_MODEL_ARTIFACT_PATH` is registered in the truth record's `artifact_paths`; `BASELINE_MODELS_PATH` is the models-root path. Downstream consumers should prefer the truth-record path for reproducible artifact resolution.
- `baseline_process_run_id` is inherited from `GOLD_PROCESS_RUN_ID` (generated at bootstrap) if it is a non-empty string. If it is empty or not a string, `make_process_run_id("gold_baseline_process")` generates a fresh ID. This covers re-run scenarios where the bootstrap process run ID may not have propagated.
- The `wandb_run.finish()` in cell 76 closes the W&B run before the final lineage checks and SQL write in cells 79 and 83. Artifacts logged after cell 76 are not registered with W&B.
