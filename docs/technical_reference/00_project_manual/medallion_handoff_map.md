# Medallion Handoff Map

## Purpose

This document describes the layer-by-layer data handoffs across the Bronze → Silver → Gold medallion architecture for the WGU Data Analytics Capstone project. Each handoff row identifies the producer, consumer, object or table exchanged, its purpose in the pipeline, and the evidence confidence level.

---

## Bronze → Silver Handoff

| Handoff | Producer | Consumer | Object / Table / Artifact | Purpose | Evidence | Evidence Confidence |
|---|---|---|---|---|---|---|
| Bronze preprocessed data | Bronze_01_Preprocessing | Silver_01_PreEDA | Preprocessed Bronze Parquet (via `write_layer_dataframe` → `capstone.bronze` schema) | Provides cleaned, schema-validated pump telemetry rows as the Silver EDA input | `sql_touchpoints.json` confirms `write_layer_dataframe` call; 071b reference confirms `bronze_preprocessing` truth record and Bronze layer write | Confirmed from generated 071b reference |
| Bronze truth record | Bronze_01_Preprocessing | Silver_01_PreEDA (and downstream) | `bronze_preprocessing` truth JSON under `truths/` | Anchors lineage chain; parent hash propagated to Silver and Gold | 071b reference confirms `save_truth_record` call | Confirmed from generated 071b reference |
| `capstone.pipeline_runs` row | Bronze_01_Preprocessing | Operational/audit layer | SQL row in `capstone.pipeline_runs` | Records Bronze run identity, timing, and metadata | `sql_touchpoints.json` confirms `write_layer_dataframe` and `read_layer_dataframe` in Bronze_01 | Confirmed from inventory or manifest |

---

## Silver → Silver Handoff (within Silver layer)

| Handoff | Producer | Consumer | Object / Table / Artifact | Purpose | Evidence | Evidence Confidence |
|---|---|---|---|---|---|---|
| Silver pre-EDA profiles | Silver_01_PreEDA | Silver_02a_EDA_Building_Subsets_v3 | Pre-EDA profile Parquets; null/distribution summary files | Provides column quality metadata and baseline distribution statistics for subset construction | 071b references confirm `to_parquet` and `save_json` output clues; Silver_02a 071b reference lists Silver_01 as upstream | Confirmed from generated 071b reference |
| Silver_02a clean subsets | Silver_02a_EDA_Building_Subsets_v3 | Silver_02b_EDA_v2 | Clean subset Parquets; analytical subsets; imputation recommendation JSON | Silver_02b performs deeper EDA on clean subset data | 071b reference confirms Silver_02b reads from Silver_02a output | Confirmed from generated 071b reference |
| Silver layer SQL write | Silver_01; Silver_02a; Silver_02b | Operational/audit layer | `silver` schema layer frames via `write_layer_dataframe` | Persists Silver data into PostgreSQL for downstream reads and audit | `sql_touchpoints.json` confirms `write_layer_dataframe` + `read_layer_dataframe` in all three Silver notebooks | Confirmed from inventory or manifest |

---

## Silver → Gold Handoff

| Handoff | Producer | Consumer | Object / Table / Artifact | Purpose | Evidence | Evidence Confidence |
|---|---|---|---|---|---|---|
| Clean analytical Parquet | Silver_02a_EDA_Building_Subsets_v3 | Gold_01_PreProcessing | Clean pump telemetry subset Parquet (imputed, deduplicated, column-standardized) | Primary Gold preprocessing input | 071b reference for Gold_01 confirms Silver_02a clean subset as the upstream data source | Confirmed from generated 071b reference |
| Imputation recommendation JSON | Silver_02a_EDA_Building_Subsets_v3 | Gold_01_PreProcessing | JSON artifact documenting recommended imputation choices | Informs Gold_01 imputation strategy | 071b reference mentions imputation recommendation artifact; `artifact_io_manifest.json` confirms `save_json` in Silver_02a | Confirmed from generated 071b reference |
| EDA context | Silver_02b_EDA_v2 | Gold_01 (context; not a confirmed direct file-level dependency) | EDA analysis JSONs; feature commentary | Informs feature selection decisions made during Gold_01 development | 071b reference for Silver_02b; direct artifact-level file dependency not confirmed | Not determined from available source |

---

## Gold Preprocessing → Gold Modeling Handoff

| Handoff | Producer | Consumer | Object / Table / Artifact | Purpose | Evidence | Evidence Confidence |
|---|---|---|---|---|---|---|
| Scaled Parquet | Gold_01_PreProcessing | Gold_02; Gold_03a; Gold_03b; Gold_03c | `gold_preprocessed_scaled_data` Parquet | Primary model input; all Gold modeling notebooks load this as their scoring and fit base | 071b references for Gold_02 through Gold_03c confirm `GOLD_PREPROCESSED_SCALED_DATA_PATH` as primary input | Confirmed from generated 071b reference |
| `gold_preprocessing` truth record | Gold_01_PreProcessing | Gold_02; Gold_03a; Gold_03b; Gold_03c | Truth JSON (`GOLD_TRUTH_PATH`) with `parent_truth_hash` and 8 artifact path overrides | Enables all downstream Gold modeling notebooks to resolve canonical Gold_01 artifact paths; anchors `GOLD_PARENT_TRUTH_HASH` | 071b references confirm `require_mapping(load_json(GOLD_TRUTH_PATH))` and 8 override fields | Confirmed from generated 071b reference |
| Feature JSON lists (4) | Gold_01_PreProcessing | Gold_02; Gold_03a; Gold_03b; Gold_03c | `stage1_features`, `stage2_features`, `stage3_primary_sensors`, `stage3_secondary_sensors` JSON files | Stage 1/2 IF feature sets and Stage 3 rule sensor sets for all modeling notebooks | 071b references confirm `require_str_list(load_json(...))` for all 4 lists in each modeling notebook | Confirmed from generated 071b reference |
| Fit / test / train / preprocessed Parquets | Gold_01_PreProcessing | Gold_02; Gold_03a; Gold_03b; Gold_03c | 4 additional Parquet splits resolved via truth-record path overrides | Partition reference for model fit, evaluation, and preprocessing reference | 071b references confirm loading of these 4 paths | Confirmed from generated 071b reference |
| Gold layer SQL write | Gold_01_PreProcessing | Operational/audit layer | `gold` schema layer frames via `write_layer_dataframe`; `capstone.pipeline_runs` row | Persists Gold preprocessing metadata into PostgreSQL | `sql_touchpoints.json` confirms `write_layer_dataframe` in Gold_01 | Confirmed from inventory or manifest |

---

## Gold Cascade Stage-to-Stage Handoff (Gold_03b → Gold_03c)

| Handoff | Producer | Consumer | Object / Table / Artifact | Purpose | Evidence | Evidence Confidence |
|---|---|---|---|---|---|---|
| Tuned Stage 2 thresholds JSON | Gold_03b_Cascade_Modeling | Gold_03c_Cascade_Modeling | `cascade_tuned_thresholds.json` (`CASCADE_TUNED_THRESHOLDS_PATH`) | Gold_03c reads `stage2_selected_threshold_percentile` and `stage2_best_params` as the `"previous_best"` Stage 2 source | 071b references for Gold_03b and Gold_03c confirm this path and the `STAGE2_SELECTION_SOURCE="previous_best"` mechanism | Confirmed from generated 071b reference |
| Gold_03b Stage 2 model | Gold_03b_Cascade_Modeling | Gold_03c_Cascade_Modeling | `cascade_tuned_stage2_model.joblib` (`CASCADE_TUNED_STAGE2_MODEL_PATH`) | Gold_03c loads and applies Gold_03b's trained Stage 2 IF model without retraining | 071b reference for Gold_03c confirms loading this model | Confirmed from generated 071b reference |
| Reference profile CSV | Gold_03b_Cascade_Modeling | Gold_03c_Cascade_Modeling | `CASCADE_TUNED_REFERENCE_PROFILE_PATH` | Stage 3 breach bound profile from Gold_03b's normal-only fit data | 071b reference for Gold_03c confirms loading Gold_03b's saved profile | Confirmed from generated 071b reference |

---

## Gold Modeling → Comparison / Reporting Handoff

| Handoff | Producer | Consumer | Object / Table / Artifact | Purpose | Evidence | Evidence Confidence |
|---|---|---|---|---|---|---|
| Baseline results pickle + JSON + truth | Gold_02_Baseline_Modeling | Gold_04_Comparison | Results pickle; metadata JSON; `gold_baseline` truth record | Gold_04 loads and validates Gold_02 output as one of four model comparison inputs | 071b reference for Gold_04 confirms three-source truth hash validation for Gold_02 | Confirmed from generated 071b reference |
| Cascade defaults results + truth | Gold_03a_Cascade_Modeling | Gold_04_Comparison | Results pickle; metadata JSON; `gold_cascade (defaults)` truth record | Gold_04 loads and validates Gold_03a output | 071b reference for Gold_04 | Confirmed from generated 071b reference |
| Cascade tuned results + truth | Gold_03b_Cascade_Modeling | Gold_04_Comparison | Results pickle; metadata JSON; `gold_cascade (tuned)` truth record | Gold_04 loads and validates Gold_03b output | 071b reference for Gold_04 | Confirmed from generated 071b reference |
| Cascade stage3_improved results + truth | Gold_03c_Cascade_Modeling | Gold_04_Comparison | Results pickle; metadata JSON; `gold_cascade (stage3_improved)` truth record | Gold_04 loads and validates Gold_03c output; this is the final cascade output | 071b references for Gold_03c and Gold_04 | Confirmed from generated 071b reference |
| Selected model run scored results | Gold_02 or Gold_03x (via `SELECTED_RUN_KEY`) | Gold_05_Anomaly_Detection | Scored results Parquet/CSV for the selected run key | Gold_05 performs anomaly timeline analysis on the selected run's output | 071b reference for Gold_05 confirms `SELECTED_RUN_KEY` (default `"stage3_improved"`) and upstream data load | Confirmed from generated 071b reference |
| `gold.model_comparison_results` | Gold_04_Comparison | Operational/audit layer | 7 comparison rows in `gold.model_comparison_results` | Persists cross-model comparison metrics into PostgreSQL for reporting | 071b reference for Gold_04 confirms `write_gold_model_comparison_results_sql` call | Confirmed from generated 071b reference |
| `gold.anomaly_detection_scores` | Gold_02; Gold_03a; Gold_03b; Gold_03c | Operational/audit layer | Scored rows per model variant | Persists row-level anomaly scores and flags per model into PostgreSQL | 071b references confirm `write_gold_cascade_scores_sql` / baseline equivalent | Confirmed from generated 071b reference |

---

## Comparison / Reporting → Final Validation Handoff

| Handoff | Producer | Consumer | Object / Table / Artifact | Purpose | Evidence | Evidence Confidence |
|---|---|---|---|---|---|---|
| Validation contracts (Gold_02, Gold_03a, Gold_03b, Gold_03c × modes) | Gold_02; Gold_03a; Gold_03b; Gold_03c | Gold_06A_Test_Replay_Validation | Validation contract JSON files | Gold_06A uses contracts to confirm replayed metrics match training-notebook summary values | 071b references for Gold_03b, Gold_03c, Gold_06A confirm `build_gold_model_output_validation_contract` and contract loading | Confirmed from generated 071b reference |
| Fitted models (joblib) | Gold_02; Gold_03a; Gold_03b; Gold_03c | Gold_06A_Test_Replay_Validation | Stage 1 and Stage 2 IF models (joblib) per variant | Gold_06A replays all models against the held-out test set without retraining | 071b reference for Gold_06A confirms loading fitted models | Confirmed from generated 071b reference |
| Test replay scores CSV | Gold_06A_Test_Replay_Validation | Gold_06B_Test_Early_Warning_Validation | `{DATASET_NAME}__gold06a__test_replay_scores.csv` | Gold_06B reads row-level replay scores to compute per-model early-warning timing metrics | 071b references for Gold_06A and Gold_06B confirm this file and path | Confirmed from generated 071b reference |
| Multi-run lead-time comparison CSV | Gold_05_Anomaly_Detection | Gold_06B_Test_Early_Warning_Validation | `multi_run_lead_time_comparison.csv` | Gold_06B optionally compares test-replay lead times to Gold_05 training-run reference | 071b reference for Gold_06B confirms optional load of this file with graceful fallback | Confirmed from generated 071b reference |

---

## Truth Hash and Lineage Chain

The following lineage chain is confirmed from 071b references:

```
bronze_preprocessing truth hash
  └─► silver_preprocessing parent_truth_hash (not directly confirmed from available source)
        └─► gold_preprocessing truth hash (GOLD_PARENT_TRUTH_HASH)
              ├─► gold_baseline truth hash (BASELINE_TRUTH_HASH)
              ├─► gold_cascade (defaults) truth hash (CASCADE_DEFAULTS_TRUTH_HASH)
              ├─► gold_cascade (tuned) truth hash (CASCADE_TUNED_TRUTH_HASH)
              └─► gold_cascade (stage3_improved) truth hash (CASCADE_STAGE3_IMPROVED_TRUTH_HASH)
                    └─► gold_comparison truth hash (COMPARISON_TRUTH_HASH)
```

All four Gold modeling notebooks share `GOLD_PARENT_TRUTH_HASH` (sourced from Gold_01 truth record). Gold_04 validates this cross-notebook parent hash invariant. Silver-to-Gold truth chain continuity is not directly confirmed from available 071b references or manifests.
