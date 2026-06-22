# Artifact and Table Handoff Map

## Purpose

This document catalogs the known artifact and SQL table handoffs across the Bronze → Silver → Gold pipeline. Each row identifies what was produced, by whom, consumed by whom, and at what location or table. Evidence confidence reflects whether the relationship was confirmed from a notebook workflow reference, inventory/manifest, or is uncertain.

Note on wrapper functions: The `artifact_io_manifest.json` confirms `save_json`, `to_parquet`, `to_csv`, and `joblib.dump` call clues per notebook but does not expose resolved file paths (paths are determined at runtime from config). Where a wrapper such as `save_data()` is used, the artifact is marked as confirmed only if the 071b reference also names the specific path variable or filename pattern.

---

## SQL Tables

| Object | Object Type | Produced By | Consumed By | Location / Table | Notes | Evidence Confidence |
|---|---|---|---|---|---|---|
| Bronze layer frames | SQL table | Bronze_01_Preprocessing | Silver_01_PreEDA | `capstone.bronze` (via `write_layer_dataframe`) | Read by Silver_01 via `read_layer_dataframe` | Confirmed from inventory or manifest |
| Silver layer frames | SQL table | Silver_01; Silver_02a; Silver_02b | Downstream Silver/Gold readers (via `read_layer_dataframe`) | `capstone.silver` (via `write_layer_dataframe`) | All three Silver notebooks write; `sql_touchpoints.json` confirms calls | Confirmed from inventory or manifest |
| Gold layer frames | SQL table | Gold_01; Gold_02; Gold_03a; Gold_03b; Gold_03c; Gold_04 | Operational/reporting layer | `capstone.gold` or named sub-tables (via `write_layer_dataframe`) | Schema and table names fully resolved at runtime | Confirmed from inventory or manifest |
| `gold.anomaly_detection_scores` | SQL table | Gold_02 (`model_stage="baseline_final"`); Gold_03a (`model_stage="cascade_defaults_final"`); Gold_03b (`model_stage="cascade_tuned_final"`); Gold_03c (`model_stage="cascade_stage3_improved_final"`) | Reporting / operational layer | `gold.anomaly_detection_scores` | Each model writes its own rows distinguished by `model_stage`; Gold_03c clears prior rows (`DELETE FROM` confirmed in `sql_touchpoints.json`) | Confirmed from generated 071b reference |
| `gold.model_comparison_results` | SQL table | Gold_04_Comparison | Reporting / operational layer | `gold.model_comparison_results` | 7 comparison rows per run; written by `write_gold_model_comparison_results_sql` | Confirmed from generated 071b reference |
| `capstone.pipeline_runs` | SQL table | Bronze_01; Silver_01; Silver_02a; Silver_02b; Gold_01; Gold_05 | Audit layer | `capstone.pipeline_runs` | All notebooks with `write_layer_dataframe` clues write pipeline run records; confirmed for Gold_05 specifically in 071b reference | Confirmed from inventory or manifest |
| `capstone.data_quality_events` | SQL table | Gold_05_Anomaly_Detection | Audit layer | `capstone.data_quality_events` | Gold_05 071b reference confirms write after SQL write | Confirmed from generated 071b reference |

---

## Gold_01 Preprocessing Artifacts

| Object | Object Type | Produced By | Consumed By | Location / Table | Notes | Evidence Confidence |
|---|---|---|---|---|---|---|
| `gold_preprocessed_scaled_data` | Parquet artifact | Gold_01_PreProcessing | Gold_02; Gold_03a; Gold_03b; Gold_03c; Gold_06A | `GOLD_PREPROCESSED_SCALED_DATA_PATH` (resolved from truth record override) | Includes `meta__is_train_flag` column used by Gold_06A for test mask derivation | Confirmed from generated 071b reference |
| `gold_preprocessed_data` (unscaled) | Parquet artifact | Gold_01_PreProcessing | Gold_02; Gold_03a; Gold_03b; Gold_03c | `gold_preprocessed_path` (truth override) | Reference / unscaled copy | Confirmed from generated 071b reference |
| `gold_fit_data` (normal-only) | Parquet artifact | Gold_01_PreProcessing | Gold_02; Gold_03a; Gold_03b; Gold_03c | `gold_fit_path` (truth override) | Used for model fit (normal operating conditions only) | Confirmed from generated 071b reference |
| `gold_test_data` | Parquet artifact | Gold_01_PreProcessing | Gold_02; Gold_03a; Gold_03b; Gold_03c; Gold_06A | `gold_test_path` (truth override) | Held-out test split | Confirmed from generated 071b reference |
| `gold_train_data` | Parquet artifact | Gold_01_PreProcessing | Gold_02; Gold_03a; Gold_03b; Gold_03c | `gold_train_path` (truth override) | Training split | Confirmed from generated 071b reference |
| Stage 1 feature JSON | Feature registry | Gold_01_PreProcessing | Gold_02; Gold_03a; Gold_03b; Gold_03c; Gold_06A | `stage1_features_path` (truth override) | IF Stage 1 feature set | Confirmed from generated 071b reference |
| Stage 2 feature JSON | Feature registry | Gold_01_PreProcessing | Gold_03a; Gold_03b; Gold_03c; Gold_06A | `stage2_features_path` (truth override) | IF Stage 2 reduced feature set | Confirmed from generated 071b reference |
| Stage 3 primary sensor JSON | Feature registry | Gold_01_PreProcessing | Gold_03a; Gold_03b; Gold_03c; Gold_06A | `stage3_primary_path` (truth override) | Stage 3 rule primary sensor set | Confirmed from generated 071b reference |
| Stage 3 secondary sensor JSON | Feature registry | Gold_01_PreProcessing | Gold_03a; Gold_03b; Gold_03c; Gold_06A | `stage3_secondary_path` (truth override) | Stage 3 rule secondary sensor set | Confirmed from generated 071b reference |
| `gold_preprocessing` truth record | Truth record | Gold_01_PreProcessing | Gold_02; Gold_03a; Gold_03b; Gold_03c | `GOLD_TRUTH_PATH` JSON | Contains 8 artifact path overrides and `parent_truth_hash`; consumed by all downstream Gold modeling notebooks | Confirmed from generated 071b reference |

---

## Gold Modeling Artifacts (per model variant)

| Object | Object Type | Produced By | Consumed By | Location / Table | Notes | Evidence Confidence |
|---|---|---|---|---|---|---|
| Baseline results CSV | Prediction output | Gold_02 | Gold_04 | `BASELINE_RESULTS_PATH_CSV` | Comparison table input for Gold_04 | Confirmed from generated 071b reference |
| Baseline results pickle | Prediction output | Gold_02 | Gold_04 | `BASELINE_RESULTS_PATH_PICKLE` | Gold_04 loads pickle for in-memory processing | Confirmed from generated 071b reference |
| Baseline model (joblib) | Model artifact | Gold_02 | Gold_06A | `BASELINE_MODEL_ARTIFACT_PATH` | Loaded by Gold_06A for test replay | Confirmed from generated 071b reference |
| Baseline validation contract | Validation output | Gold_02 | Gold_06A | `baseline_contract_path` | Contract for Gold_06A replay metric comparison | Confirmed from generated 071b reference |
| `gold_baseline` truth record | Truth record | Gold_02 | Gold_04 (three-source hash check) | `TRUTHS_PATH/gold_baseline/` | `BASELINE_TRUTH_HASH` validated in Gold_04 | Confirmed from generated 071b reference |
| Cascade results CSV + pickle (per variant) | Prediction output | Gold_03a; Gold_03b; Gold_03c | Gold_04; Gold_06A | `CASCADE_RESULTS_PATH_CSV/PICKLE` per variant | Gold_03c results are the "final" cascade output in Gold_04 | Confirmed from generated 071b reference |
| Stage 1 + Stage 2 models (joblib, per variant) | Model artifact | Gold_03a; Gold_03b; Gold_03c | Gold_06A | `STAGE1/2_MODEL_ARTIFACT_PATH` per variant | Gold_06A loads all variant models for replay | Confirmed from generated 071b reference |
| Reference profile CSV (per variant) | Row-tracking artifact | Gold_03a; Gold_03b; Gold_03c | Gold_06A | `CASCADE_REFERENCE_PROFILE_PATH` per variant | Stage 3 breach bounds; Gold_03c loads Gold_03b's profile | Confirmed from generated 071b reference |
| Cascade thresholds JSON (per variant) | JSON summary | Gold_03a; Gold_03b; Gold_03c | Gold_03c (Gold_03b only); Gold_06A | `CASCADE_THRESHOLDS_PATH` per variant | Gold_03b thresholds JSON is direct dependency for Gold_03c `"previous_best"` Stage 2 reuse | Confirmed from generated 071b reference |
| Cascade summary JSON (per variant) | JSON summary | Gold_03a; Gold_03b; Gold_03c | Gold_04; Gold_06A | `CASCADE_SUMMARY_PATH` per variant | Contains per-model metric summaries consumed by Gold_04 and Gold_06A | Confirmed from generated 071b reference |
| Cascade metadata JSON (per variant) | JSON summary | Gold_03a; Gold_03b; Gold_03c | Gold_04; Gold_06A | `CASCADE_METADATA_PATH` per variant | Contains metadata dict consumed in Gold_04 and Gold_06A | Confirmed from generated 071b reference |
| Validation contracts (1 per Gold_02/03a/03b; 4 for Gold_03c) | Validation output | Gold_02; Gold_03a; Gold_03b; Gold_03c | Gold_06A | Per-variant and per-mode contract JSON paths | Gold_03c writes 4 contracts (default + relaxed + medium + strict) | Confirmed from generated 071b reference |
| `gold_cascade` truth records (per variant) | Truth record | Gold_03a; Gold_03b; Gold_03c | Gold_04 (three-source hash check) | `TRUTHS_PATH/gold_cascade/` per variant | `CASCADE_DEFAULTS/TUNED/STAGE3_IMPROVED_TRUTH_HASH` validated in Gold_04 | Confirmed from generated 071b reference |

---

## Gold_03b → Gold_03c Direct Handoff Artifacts

| Object | Object Type | Produced By | Consumed By | Location / Table | Notes | Evidence Confidence |
|---|---|---|---|---|---|---|
| `cascade_tuned_thresholds.json` | JSON summary | Gold_03b | Gold_03c | `CASCADE_TUNED_THRESHOLDS_PATH` | Contains `stage2_selected_threshold_percentile` and `stage2_best_params`; loaded by Gold_03c as `"previous_best"` Stage 2 source | Confirmed from generated 071b reference |
| Gold_03b Stage 2 model | Model artifact | Gold_03b | Gold_03c | `CASCADE_TUNED_STAGE2_MODEL_PATH` | Loaded by Gold_03c; applied without retraining | Confirmed from generated 071b reference |
| Gold_03b reference profile | Row-tracking artifact | Gold_03b | Gold_03c | `CASCADE_TUNED_REFERENCE_PROFILE_PATH` | Stage 3 breach bounds reused by Gold_03c | Confirmed from generated 071b reference |

---

## Gold_04 Comparison Artifacts

| Object | Object Type | Produced By | Consumed By | Location / Table | Notes | Evidence Confidence |
|---|---|---|---|---|---|---|
| Comparison table CSV | Metric summary | Gold_04_Comparison | Reporting / audit layer | `MODEL_COMPARISON_PATH` | 7-row comparison across all 4 models | Confirmed from generated 071b reference |
| Comparison summary JSON | JSON summary | Gold_04_Comparison | Reporting / audit layer | `MODEL_COMPARISON_SUMMARY_PATH` | Contains all 4 stage truth hashes and summary fields | Confirmed from generated 071b reference |
| Statistical test CSV + JSON | Metric summary | Gold_04_Comparison | Reporting / audit layer | `GOLD_COMPARISON_ARTIFACT_DIRS["statistics"]` | McNemar + z-test results | Confirmed from generated 071b reference |
| Cascade funnel charts | Plot/report artifact | Gold_04_Comparison | Reporting / audit layer | `GOLD_COMPARISON_ARTIFACT_DIRS["plots"]` | PNGs per cascade model | Confirmed from generated 071b reference |
| `gold_comparison` truth record | Truth record | Gold_04_Comparison | Lineage audit | `TRUTHS_PATH/gold_comparison/` | `COMPARISON_TRUTH_HASH` anchored on `GOLD_PARENT_TRUTH_HASH` | Confirmed from generated 071b reference |
| `gold.model_comparison_results` rows | SQL table | Gold_04_Comparison | Reporting / operational layer | `gold.model_comparison_results` | 7 rows per run | Confirmed from generated 071b reference |

---

## Gold_05 Anomaly Detection Artifacts

| Object | Object Type | Produced By | Consumed By | Location / Table | Notes | Evidence Confidence |
|---|---|---|---|---|---|---|
| Anomaly timeline Parquet | Prediction output | Gold_05 | Review / submission support | `anomaly_timeline_dataframe` path | Annotated episode-phase DataFrame; not a confirmed direct input to Gold_06A/06B | Confirmed from generated 071b reference |
| Detection summary JSON | JSON summary | Gold_05 | Review / submission support | `detection_summary_payload` path | Structured alert packet and lead-time summary | Confirmed from generated 071b reference |
| Multi-run lead-time comparison CSV | Metric summary | Gold_05 | Gold_06B (optional) | `multi_run_lead_time_comparison.csv` in anomaly detection summaries dir | Gold_06B loads this with graceful fallback if absent | Confirmed from generated 071b reference |
| Sensor visualization PNGs | Plot/report artifact | Gold_05 | Submission support | `GOLD_ANOMALY_DETECTION_ARTIFACT_DIRS["plots"]` | Timeline, heatmap, waveform, 3D surface plots | Confirmed from generated 071b reference |
| `gold_anomaly_detection` truth record | Truth record | Gold_05 | Lineage audit | `TRUTHS_PATH/gold_anomaly_detection/` | Does not chain to Gold_06A or Gold_06B | Confirmed from generated 071b reference |

---

## Gold_06A and Gold_06B Validation Artifacts

| Object | Object Type | Produced By | Consumed By | Location / Table | Notes | Evidence Confidence |
|---|---|---|---|---|---|---|
| Test replay scores CSV | Prediction output | Gold_06A | Gold_06B | `{DATASET_NAME}__gold06a__test_replay_scores.csv` in `VALIDATION_SCORES_DIR` | Wide DataFrame with per-model flag/score columns; Gold_06B requires this file | Confirmed from generated 071b reference |
| Replay validation comparison CSV | Validation output | Gold_06A | Review / audit | `{DATASET_NAME}__gold06a__test_replay_validation_comparison.csv` | Replayed vs. training metrics with tolerance status | Confirmed from generated 071b reference |
| Final validation status JSON | Validation output | Gold_06A | Review / audit | `{DATASET_NAME}__gold06a__final_validation_status.json` | `final_validation_status` field (`pass`, `pass_with_tolerance`, `review_delta`) | Confirmed from generated 071b reference |
| Early-warning summary CSV | Metric summary | Gold_06B | Submission support | `{DATASET_NAME}__gold06b__test_early_warning_summary.csv` | Per-model alert lead-time on held-out test data | Confirmed from generated 071b reference |
| Lead-time comparison CSV | Metric summary | Gold_06B | Submission support | `{DATASET_NAME}__gold06b__test_vs_gold05_lead_time_comparison.csv` | Test replay vs. Gold_05 training reference (when available) | Confirmed from generated 071b reference |
| Summary JSON | JSON summary | Gold_06B | Submission support | `{DATASET_NAME}__gold06b__test_early_warning_summary.json` | Full summary payload including availability flags | Confirmed from generated 071b reference |
| Lead-time bar plot; Stage 3 timeline plot | Plot/report artifact | Gold_06B | Submission support | `VALIDATION_PLOTS_DIR` | 2 PNGs; terminal capstone submission artifacts | Confirmed from generated 071b reference |

---

## Bronze and Silver Layer Artifacts

| Object | Object Type | Produced By | Consumed By | Location / Table | Notes | Evidence Confidence |
|---|---|---|---|---|---|---|
| Bronze preprocessed data | Parquet artifact | Bronze_01_Preprocessing | Silver_01_PreEDA | `capstone.bronze` (SQL) or Bronze artifact Parquet | `artifact_io_manifest.json` confirms `to_json` output clue (Bronze writes JSON truth record); SQL clue confirms `write_layer_dataframe` | Confirmed from inventory or manifest |
| Silver pre-EDA profiles | Parquet artifact | Silver_01_PreEDA | Silver_02a | Silver artifact path (runtime-resolved) | `artifact_io_manifest.json` confirms `to_parquet`, `save_json`, `to_json` output clues | Confirmed from inventory or manifest |
| Silver clean subsets | Parquet artifact | Silver_02a | Gold_01 | Silver artifact path (runtime-resolved) | `artifact_io_manifest.json` confirms `to_parquet`, `to_csv`, `save_json` output clues | Confirmed from inventory or manifest |
| Imputation recommendation | JSON summary | Silver_02a | Gold_01 | Silver artifact path (runtime-resolved) | `save_json` clue confirmed in `artifact_io_manifest.json` for Silver_02a | Confirmed from inventory or manifest |
| Silver EDA outputs | JSON summary | Silver_02b | Review / EDA record | Silver artifact path (runtime-resolved) | `save_json`, `to_csv` clues confirmed; no confirmed downstream notebook | Confirmed from inventory or manifest |
