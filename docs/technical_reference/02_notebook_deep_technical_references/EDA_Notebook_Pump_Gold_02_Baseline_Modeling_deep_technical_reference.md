# Gold 02 Deep Technical Reference

## Purpose of This Deep Reference

This document covers the technical decisions in Gold 02 (Baseline Modeling) that require deeper explanation than the workflow reference provides. The workflow reference describes what each section does. This document explains why Gold 02 trains a single Isolation Forest as the project baseline, why it fits exclusively on Gold 01's normal-only fit Parquet rather than on the train mask, why the anomaly threshold is calibrated from training scores alone, why the project inverts the Isolation Forest score direction and keeps two parallel score columns, why the train/test split is read from Gold 01 rather than re-derived, why artifact paths and feature lists are overridden from the Gold 01 truth record, why a zero-alert guard exists, why the `gold_baseline` truth record links back to Gold 01's hash, and why the baseline is persisted and shaped to serve as the comparison reference for the cascade models.

## Technical Scope

- `gold_baseline` stage context and two-level sanity check (general + `GOLD_CFG`)
- Gold 01 input contract: scaled Parquet, fit-normal Parquet, Stage 1 feature list, parent truth record
- Truth-driven path overrides (`gold_fit_path`, `stage1_features_path`) and identity inheritance
- Train/test split recovery from `meta__is_train_flag` (not re-derived)
- Optional ground-truth label handling for evaluation
- Feature-matrix assembly from the Stage 1 feature list (DataFrames, not arrays)
- Baseline Isolation Forest fit on normal-only fit rows with a fixed random seed
- Project anomaly score convention (`-score_samples`) and training-only threshold calibration
- Row-tracked re-scoring with dual score columns and decision/predict outputs
- Results synchronization, alert rule, and zero-alert direction guard
- Output validation, lineage cross-checks, and the `gold_baseline` truth record
- Artifact persistence (results, model ×2, threshold/summary/metadata JSON), W&B, ledger
- SQL persistence via `write_gold_baseline_scores_sql` to `gold.anomaly_detection_scores`

## Source Grounding

Sources used:

- `notebooks/experiments/EDA_Notebook_Pump_Gold_02_Baseline_Modeling.ipynb` (active notebook — source of truth)
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_02_Baseline_Modeling_code_reference.md` (read-only context)
- `notebook_inventory.json`
- `artifact_io_manifest.json`
- `sql_touchpoints.json`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_01_PreProcessing_deep_technical_reference.md` (read-only upstream context)
- `technical_reference/00_project_manual/` relationship maps (read-only context)

The active Gold 02 notebook source is the source of truth for all function behavior, variable names, output paths, and design decisions documented here.

## Stage Role in the Gold Modeling Sequence

Gold 02 is the first modeling notebook in the Gold layer. It trains and evaluates the project's baseline anomaly detector — a single Isolation Forest — on Gold 01's scaled telemetry and establishes the reference that the cascade models (Gold 03a–03c) and the comparison notebook (Gold 04) are measured against. It does not preprocess data (Gold 01's responsibility) and it does not implement the multi-stage cascade (Gold 03's responsibility); it produces one deliberately simple model whose results define "what a single Isolation Forest achieves" on this dataset.

Source-confirmed, Gold 02 performs: baseline model training on normal-only rows, full-dataset scoring, training-only threshold calibration, optional supervised evaluation against `anomaly_flag` labels, construction of a scored results frame, persistence of the model and metric artifacts, creation of a `gold_baseline` truth record linked to Gold 01's hash, and a write of scored results to `gold.anomaly_detection_scores`. W&B is active here with job type `gold_modeling_baseline`.

## Input Contract and Lineage

### Gold 01 Inputs

The primary input is Gold 01's full scaled Parquet (`GOLD_PREPROCESSED_SCALED_DATA_PATH`), loaded via `load_data`. From its `meta__dataset` column the dataset name is resolved (`ValueError` if empty), and `load_parent_truth_record_from_dataframe(parent_layer_name="gold", column_name="meta__truth_hash")` loads the Gold 01 truth record. From that record Gold 02 resolves:

- `DATASET_NAME` (overriding config) and `GOLD_PARENT_TRUTH_HASH = get_truth_hash(gold_truth)` — Gold 01's hash, which becomes the parent of Gold 02's own truth record;
- `PARENT_PIPELINE_MODE`, which overrides `PIPELINE_MODE` when non-None;
- `GOLD_FIT_DATA_PATH` and `STAGE1_FEATURES_PATH`, **overridden** from `gold_truth["artifact_paths"]` (`gold_fit_path`, `stage1_features_path`).

The fit-normal Parquet is then loaded from the truth-linked path, and the Stage 1 feature list is loaded from `STAGE1_FEATURES_PATH` (validated: not None, a list/tuple, non-empty after stripping). `gold_truth["runtime_facts"]` and `gold_truth["artifact_paths"]` are captured for forward provenance (scaler path/kind, recommended imputation, `feature_set_id`, `process_run_id`).

### Why the Path Overrides Matter

Gold 02 does not trust its configured paths for the fit data or feature list; it substitutes the exact paths Gold 01 recorded in its truth record. This is the core lineage mechanism: even if config changes between runs, Gold 02 always reads the precise fit Parquet and feature set that the certified Gold 01 run produced. Combined with reading the train/test split from Gold 01's stamped column rather than re-deriving it, this guarantees every Gold modeling notebook operates on an identical, provenance-pinned input definition. Lineage matters at the modeling stage because the baseline's results are only meaningful as a comparison anchor if every downstream model is demonstrably trained and evaluated on the same data partition and feature space.

## Baseline Input Preparation

`ensure_stable_row_id` re-confirms `meta__row_id` on the scaled dataframe so all row-tracked scoring and merges have a stable key. The train/test split is recovered, not recomputed: if `meta__is_train_flag` is absent, `ValueError` halts the run (a missing flag means Gold 01 did not complete correctly). `train_mask` is the column cast to bool; `test_mask = ~train_mask`; `test_mask_array` is the NumPy form used for aligned subsetting.

Ground-truth labels are handled conditionally: if `anomaly_flag` is present, `all_labels` and `test_labels` are extracted as integer arrays; otherwise both are `None` and every evaluation path is guarded by `if test_labels is not None`. This keeps the notebook runnable on unlabeled data while enabling supervised evaluation when synthetic truth labels exist.

Three feature matrices are built from the Stage 1 feature list after validating every column exists in both the scaled and fit frames (`ValueError` lists any missing):

- `baseline_train_fit_features = gold_fit_dataframe[stage1_feature_columns]` — model fit input;
- `baseline_all_features = gold_preprocessed_scaled_dataframe[stage1_feature_columns]` — all-rows scoring input;
- `baseline_test_features` — the test-partition slice.

Crucially, the fit features come from Gold 01's fit Parquet, not from `train_mask` applied to the scaled frame. The fit Parquet is the stricter `train ∩ normal_clean` subset Gold 01 produced after anomaly exclusion, so the model learns only confirmed-normal behavior. Matrices are kept as DataFrames to preserve feature names and avoid scikit-learn's feature-name-mismatch warning.

## Baseline Modeling Methodology

The baseline is a single `IsolationForest(n_estimators=BASELINE_ESTIMATOR_COUNT, random_state=RANDOM_SEED, n_jobs=-1)` fit on `baseline_train_fit_features`. The fixed `RANDOM_SEED` makes the tree ensemble reproducible across runs, which keeps the model artifact and truth hashes stable — a requirement for the lineage chain.

The design intent is to be deliberately simple. Isolation Forest is an unsupervised detector trained only on normal data; it learns the structure of normal pump behavior and flags rows that isolate easily as anomalous. Fitting on the normal-only subset (rather than the full train split) is what makes the learned normal boundary clean — contaminated or abnormal rows are excluded so they cannot pull the model's notion of normal toward the anomalies it is meant to detect. The baseline exists before the cascade variants precisely so that the multi-stage cascade can be evaluated as an improvement (or not) over a single-forest detector under identical inputs and the same threshold-calibration discipline. Gold 02 makes no claim about model quality; it records metrics and leaves comparison to Gold 04.

## Scoring, Thresholding, and Prediction Logic

### Project Score Convention

`compute_anomaly_scores_isolation_forest` returns `-model.score_samples(features)`, inverting scikit-learn's native direction so that **higher means more anomalous**. This project convention is applied consistently to the project score column.

### Training-Only Threshold Calibration

`choose_threshold_by_percentile(baseline_train_scores, BASELINE_THRESHOLD_PERCENTILE)` is `np.percentile` over the normal-only training scores only. Calibrating from training scores keeps the test scores unseen at calibration time and anchors the alert threshold to normal operational behavior rather than a blend of normal and anomalous rows. The alert rule is `baseline_flag = (baseline_score >= baseline_threshold)`.

### Dual Score Columns and Row-Tracked Scoring

The dataset is re-scored through `score_isolation_forest_stage` (via `build_stage_scoring_frame`) keyed on `meta__row_id`, producing per-row tracked outputs. Because that helper emits the raw `score_samples` (higher = more normal) as its score column, the notebook renames the helper outputs to avoid overwriting the project convention:

- `baseline_score_samples_raw` — the helper's raw `score_samples` (higher = more normal), preserved for any consumer needing the native direction;
- `baseline_predict_flag` — the helper's `predict`-based flag.

Stale baseline columns are dropped before the left-join merge on `meta__row_id` so re-execution does not create duplicate `_x`/`_y` columns. The canonical project columns are then recomputed directly from the model: `baseline_score = -score_samples`, `baseline_decision = decision_function`, `baseline_pred = predict`, and `baseline_flag = (baseline_score >= baseline_threshold)`. A length check guards score-to-row alignment, and the shared `gold_preprocessed_scaled_dataframe` reference is updated via `globals()` so later cells see the scored frame.

### Results Synchronization and Zero-Alert Guard

`baseline_results` is rebuilt as a copy of the scored frame with `baseline_score`, `baseline_threshold`, `baseline_threshold_percentile`, and `baseline_flag` reapplied. If `baseline_alert_count_all_rows == 0`, `ValueError` is raised — a direction guard whose own message explains the likely cause: `baseline_score` was left as raw `score_samples` instead of the inverted project score. A run that produces zero alerts under a percentile threshold almost always indicates an inverted score, so this guard converts a silent correctness bug into a hard failure.

## Evaluation and Metrics

`evaluate_against_labels(true_labels, anomaly_scores, threshold)` is the supervised evaluation path, used only when `anomaly_flag` labels are present. It derives predicted labels as `anomaly_scores >= threshold` and computes binary precision, recall, and F1 with `zero_division=0`. ROC-AUC and PR-AUC are computed only when both label classes are present in the test partition; otherwise both are `None`. The distinction between unsupervised training and supervised evaluation is explicit in the design: the model is fit without labels, but synthetic truth labels — when available — are used after the fact to quantify detection quality on the held-out test rows. The `baseline_metrics` dict records the model name, threshold percentile, threshold value, and alert counts, extended with the label metrics when labels exist. Metrics are recomputed on the synchronized test scores during results synchronization so the persisted metrics reflect the final results frame, not an intermediate scoring pass.

## Artifact and SQL Persistence

### File Artifacts

Consistent with the audit clues (`joblib.dump`, `save_json`, `to_csv`, `to_json`), Gold 02 persists:

- **Baseline results** as both CSV (`BASELINE_RESULTS_PATH_CSV`) and pickle (`BASELINE_RESULTS_PATH_PICKLE`) — the full scored frame with all meta/lineage columns.
- **Fitted model** via `joblib.dump` to two paths: `BASELINE_MODEL_ARTIFACT_PATH` (registered in the truth record's `artifact_paths`) and `BASELINE_MODELS_PATH` (the models-root path). Downstream consumers should prefer the truth-registered path for reproducible resolution.
- **Threshold JSON** (`baseline_threshold_percentile`, `baseline_threshold`), **summary JSON** (dataset name, metrics, alert counts, baseline/gold truth hashes, process run id, `feature_set_id`), and **metadata JSON** carrying the full Gold 01 provenance chain (`gold_scaler_path`, `gold_scaler_kind`, `gold_recommended_imputation`, `gold_feature_set_id`) alongside Gold 02's own artifact paths.
- The **`gold_baseline` truth record** and a **truth index** entry.

The model and the truth record are saved in the same step so the artifact path recorded in the truth always resolves. All seven artifact files plus the truth record are registered with `wandb.save`. A flagged-rows export is produced via `get_detected_rows_dataframe` for downstream inspection. The ledger is written to disk and registered with W&B, then `wandb_run.finish()` closes the run before the final lineage checks and SQL write (artifacts logged after that point would not register with W&B).

### SQL Persistence

`write_gold_baseline_scores_sql(engine, capstone_schema=CAPSTONE_SCHEMA, dataset_id=DATASET_ID, run_id=RUN_ID, notebook_globals=globals(), dataset_name=...)` writes the scored results to `gold.anomaly_detection_scores`, gated by `WRITE_TO_POSTGRES = True`. The globals manifest is passed rather than individual frames; the exact table columns are defined in `utils/medallion/gold/` and are not fully determinable from the notebook. The SQL clue `read_sql` corresponds to the bootstrap smoke check; `read_layer_dataframe`/`write_layer_dataframe` appear in imports but are not the direct baseline write path.

## Truth, Audit, and Reproducibility Behavior

`initialize_layer_truth(layer_name="gold_baseline", parent_truth_hash=GOLD_PARENT_TRUTH_HASH, process_run_id=baseline_process_run_id, pipeline_mode=PIPELINE_MODE)` creates the baseline truth record. `baseline_process_run_id` reuses `GOLD_PROCESS_RUN_ID` when it is a populated string, otherwise a fresh `make_process_run_id("gold_baseline_process")` is generated — covering re-run scenarios where the bootstrap id did not propagate. Three sections are populated:

- **config_snapshot** — `TRUTH_CONFIG` when it is a dict, else a minimal inline runtime block;
- **runtime_facts** — threshold percentile/value, estimator count, train fraction, random seed, alert counts, result row count, parent hash, plus `gold_process_run_id` and `gold_feature_set_id` carried from Gold 01's truth so any downstream reader can trace back to the exact preprocessing run and feature set;
- **artifact_paths** — gold truth/scaled/fit paths and all baseline output paths.

`build_truth_record(... row_count=len(baseline_results), column_count=baseline_results.shape[1] + 3, ...)` produces `BASELINE_TRUTH_HASH` (the `+ 3` accounts for the three lineage columns stamped next). `stamp_truth_columns` writes `meta__truth_hash` (= `BASELINE_TRUTH_HASH`), `meta__parent_truth_hash` (= `GOLD_PARENT_TRUTH_HASH`), and `meta__pipeline_mode` into every row. `save_truth_record(layer_name="gold_baseline")` and `append_truth_index` persist and register the record.

A dedicated final lineage check enforces correctness before the SQL write: the three lineage columns must be present; `extract_truth_hash(baseline_results)` must equal `BASELINE_TRUTH_HASH`; `meta__parent_truth_hash` must hold exactly one value equal to `GOLD_PARENT_TRUTH_HASH`; the saved truth JSON must reload with matching `truth_hash` and `parent_truth_hash`; and the saved metadata JSON must carry the matching `baseline_truth_hash`. This matters before cascade modeling, comparison, and validation because those stages anchor to `BASELINE_TRUTH_HASH` and the carried provenance — a corrupt or inconsistent stamp here would silently poison every downstream comparison.

## Downstream Technical Handoff

Source-confirmed outputs produced, stamped, and registered by Gold 02:

- Baseline results (CSV + pickle), each carrying `BASELINE_TRUTH_HASH` and the Gold parent hash.
- Fitted model (two joblib paths), threshold/summary/metadata JSONs.
- The `gold_baseline` truth record (parent = `GOLD_PARENT_TRUTH_HASH`) and its truth index entry.
- Scored rows in `gold.anomaly_detection_scores`.

The workflow reference maps these to specific consumers — Gold 04 (comparison reads the results, summary, thresholds, and validates `BASELINE_TRUTH_HASH`), Gold 05 (anomaly detection reads scored results), and Gold 06A (replay validation uses the baseline model artifact). From Gold 02 source alone, these artifacts and the truth record are confirmed to be produced, stamped, and indexed; the precise file-level read performed by each downstream Gold notebook is governed by those notebooks and is **Not determined from available source** here. Notebook order alone is not treated as evidence of direct handoff. The `baseline_metadata.json` deliberately embeds the Gold 01 provenance chain so a comparison or audit notebook can trace the exact preprocessing run without re-reading Gold 01's truth record directly.

## Key Technical Decisions

| Decision | Source Evidence | Why It Matters | Verification Method |
|---|---|---|---|
| Train a single Isolation Forest as the baseline | Cell 54 `IsolationForest(...)`; stage `gold_baseline` | Establishes a deliberately simple reference the cascade variants are measured against | Confirm one `IsolationForest.fit` call on the fit features |
| Fit on Gold 01's normal-only fit Parquet, not the train mask | Cell 34 loads `gold_fit_dataframe`; cell with `baseline_train_fit_features = gold_fit_dataframe[stage1...]` | The fit Parquet is the stricter `train ∩ normal_clean` subset; the model learns only confirmed-normal behavior | Confirm fit input rows equal `gold_fit_dataframe`, not `train_mask` rows |
| Override fit and feature-list paths from the Gold 01 truth record | Cell 34: `GOLD_FIT_DATA_PATH`/`STAGE1_FEATURES_PATH` from `gold_truth["artifact_paths"]` | Pins inputs to the exact artifacts the certified Gold 01 run produced, regardless of config drift | Confirm effective paths match the truth record's `gold_fit_path`/`stage1_features_path` |
| Recover the train/test split from `meta__is_train_flag` | Cell 40: `ValueError` if column absent | Keeps the partition identical across all Gold modeling notebooks; a missing flag means Gold 01 failed | Confirm split is read from the column, not recomputed |
| Invert the score direction (`-score_samples`) | Cell 48 `compute_anomaly_scores_isolation_forest` | Makes higher mean more anomalous so a percentile threshold flags the top scores | Confirm `baseline_score = -score_samples` and alerts are the high-score rows |
| Calibrate threshold from training scores only | Cell 57 comment + `choose_threshold_by_percentile(baseline_train_scores, ...)` | Anchors the threshold to normal behavior; keeps test scores unseen at calibration | Confirm threshold input is `baseline_train_scores` (normal-only fit rows) |
| Preserve dual score columns | Cell 61 rename to `baseline_score_samples_raw`; recompute `baseline_score` | Lets consumers use either the native or project direction without ambiguity | Confirm both columns exist with opposite orientation |
| Zero-alert direction guard | Cell 63 `if baseline_alert_count_all_rows == 0: raise ValueError` | Converts a silent score-inversion bug into a hard failure | Confirm the guard raises when no alerts are produced |
| Reproducible model via fixed seed | Cell 54 `random_state=RANDOM_SEED`; comment on stable artifact hashes | Keeps the model artifact and truth hashes stable across runs | Confirm identical seed yields identical model/hash |
| Supervised metrics only when labels and both classes present | Cell 52 `evaluate_against_labels`; guarded by `if test_labels is not None` | Runs unlabeled while quantifying detection quality when truth labels exist | Confirm ROC/PR-AUC are `None` when a single class is present |
| `gold_baseline` truth record linked to Gold 01's hash | Cell 70 `initialize_layer_truth(layer_name="gold_baseline", parent_truth_hash=GOLD_PARENT_TRUTH_HASH)` | Extends the truth chain so downstream stages can anchor to the baseline | Confirm a `gold_baseline` record with parent = Gold 01 hash exists |
| Carry Gold 01 provenance into baseline metadata | Cell 70 metadata: scaler path/kind, recommended imputation, `feature_set_id` | Lets comparison/audit notebooks trace the preprocessing run without re-reading Gold 01 truth | Inspect `baseline_metadata.json` for the Gold 01 provenance fields |
| Final lineage cross-check before SQL write | Cell 79 hash/parent/reload checks | Catches late column drops or stamp corruption before persisting downstream rows | Confirm the checks raise on any hash mismatch |
| SQL via `write_gold_baseline_scores_sql` behind `WRITE_TO_POSTGRES` | Cell 83 → `gold.anomaly_detection_scores` | Durable scored-results persistence with an offline-run escape hatch | Confirm the gate controls the write; file artifacts unaffected when `False` |

## Failure Modes and Guardrails

| Failure Condition | Behavior | Guardrail |
|---|---|---|
| Gold 01 scaled Parquet lacks usable `meta__dataset` | `ValueError` | Non-empty dataset-name check after load |
| Gold 01 parent truth not loadable / not a dict | `TypeError`/error from loader | Type check on `gold_truth` |
| `meta__is_train_flag` absent | `ValueError` | Split-recovery guard before modeling |
| Stage 1 feature list None / wrong type / empty | `ValueError`/`TypeError` | Validated after `load_json` |
| Stage 1 feature column missing from scaled or fit frame | `ValueError` listing missing columns | Presence check before matrix assembly |
| All-rows score length ≠ dataframe length | `ValueError` | Length guards after scoring (twice) |
| Zero alerts after synchronization | `ValueError` (score-direction message) | Cell 63 direction guard |
| Results frame missing required columns or non-unique `meta__row_id` | `ValueError` | `validate_baseline_output` |
| Stamped hash ≠ computed hash | `ValueError` | Final `extract_truth_hash` cross-check |
| Multiple or empty parent hashes in results | `ValueError` | Parent-hash uniqueness check |
| Saved truth/metadata JSON hash mismatch | `ValueError` | Reload-and-compare in final check |
| Baseline truth file not created | `FileNotFoundError` | Existence check on `baseline_truth_path` |
| `WRITE_TO_POSTGRES = False` | SQL write skipped; file artifacts unaffected | Explicit boolean gate |
| W&B closed before later cells | Post-`finish` saves do not register | `wandb_run.finish()` ordered after artifact saves |

## Verification Checklist

- Active notebook path is `notebooks/experiments/EDA_Notebook_Pump_Gold_02_Baseline_Modeling.ipynb`
- Gold 01 scaled Parquet exists and carries `meta__truth_hash`; parent truth record loads
- `GOLD_FIT_DATA_PATH` and `STAGE1_FEATURES_PATH` are overridden from the Gold 01 truth record
- `meta__is_train_flag` is present; train/test masks are read, not recomputed
- All Stage 1 feature columns exist in both scaled and fit frames
- `baseline_train_fit_features` come from `gold_fit_dataframe` (normal-only), not from `train_mask`
- Isolation Forest is fit with `random_state=RANDOM_SEED`
- `baseline_score = -score_samples`; threshold is calibrated from training scores only
- `baseline_score` and `baseline_score_samples_raw` both exist with opposite orientation
- Alert count is non-zero (direction guard passes)
- ROC/PR-AUC are present only when test labels have both classes
- `gold_baseline` truth record exists with parent = `GOLD_PARENT_TRUTH_HASH`; truth index updated
- `baseline_results` carries `meta__truth_hash` (= `BASELINE_TRUTH_HASH`), parent hash, and pipeline mode
- Results CSV/pickle, model joblib (×2), threshold/summary/metadata JSONs exist
- Final lineage checks pass (hash, single parent hash, reload comparisons)
- If `WRITE_TO_POSTGRES = True`: rows written to `gold.anomaly_detection_scores`
- A W&B run with `job_type="gold_modeling_baseline"` is created and finished

## Source-Limited Items

- The exact `gold.anomaly_detection_scores` table columns and the internal behavior of `write_gold_baseline_scores_sql` are Not determined from Gold 02 source (the function is called, not defined, in the notebook).
- The precise file-level reads performed by Gold 04, Gold 05, and Gold 06A against Gold 02's artifacts are Not determined from Gold 02 source; only the production, stamping, and indexing of those artifacts are confirmed here.
- The internal definitions of `build_stage_scoring_frame`, `score_isolation_forest_stage`, `validate_baseline_output`, `get_detected_rows_dataframe`, `build_truth_record`, and `stamp_truth_columns` are in the utility modules; their observable effects are documented, but their full implementations are Not determined from Gold 02 source.
- The concrete values of `BASELINE_ESTIMATOR_COUNT`, `BASELINE_THRESHOLD_PERCENTILE`, `RANDOM_SEED`, and `TRAIN_FRACTION` are config-driven (`GOLD_CFG`) and not fixed in the notebook source.
