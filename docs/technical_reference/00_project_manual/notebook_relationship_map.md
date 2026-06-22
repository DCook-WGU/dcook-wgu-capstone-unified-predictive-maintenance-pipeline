# Notebook Relationship Map

## Purpose

This document connects the standalone notebook workflow references under `technical_reference/01_notebook_workflow_references/` into a coherent project manual. Each workflow reference describes a single notebook in depth; this document explains how they relate to each other across the full Bronze → Silver → Gold pipeline for the WGU Data Analytics Capstone — Industrial Pump Telemetry Anomaly Detection project.

---

## End-to-End Notebook Flow

The following notebooks are verified as active under `notebooks/`:

```
Bronze_01_Preprocessing
  └─► Silver_01_PreEDA
        └─► Silver_02a_EDA_Building_Subsets_v3
              └─► Silver_02b_EDA_v2
              └─► Gold_01_PreProcessing
                    ├─► Gold_02_Baseline_Modeling ──────────────────────────► Gold_04_Comparison
                    ├─► Gold_03a_Cascade_Modeling ──────────────────────────► Gold_04_Comparison ──► Gold_05_Anomaly_Detection
                    ├─► Gold_03b_Cascade_Modeling ──► Gold_03c ─────────────► Gold_04_Comparison      └─► Gold_06B (optional)
                    └─► Gold_03c_Cascade_Modeling ──────────────────────────► Gold_04_Comparison
                                                                              Gold_06A_Test_Replay
                                                                                └─► Gold_06B_Test_Early_Warning
```

Source: Confirmed from generated 071b workflow references.

---

## Layer-by-Layer Relationship Summary

### Bronze Layer

Bronze_01_Preprocessing ingests raw synthetic pump telemetry from the PostgreSQL operational layer (via `read_layer_dataframe` and the `capstone` schema), applies Bronze-level preprocessing (schema validation, column cleaning, row identity stamping with `meta__row_id`), writes a preprocessed Bronze Parquet to the artifact store, and establishes a `bronze_preprocessing` truth record. Silver_01_PreEDA depends on this Bronze output as its starting data.

### Silver Layer

Silver_01_PreEDA reads the Bronze Parquet, performs pre-EDA profiling (null analysis, distribution summaries, column metadata), and hands its profiling outputs to Silver_02a. Silver_02a builds clean analytical subsets, applies imputation recommendations, and produces the clean Silver Parquet consumed by Gold_01. Silver_02b performs deeper EDA analysis on Silver_02a outputs and provides supplementary distribution and feature analysis context; it is not a direct pipeline input to Gold_01 in the file-based sense but informs feature selection decisions documented in the EDA record.

### Gold Layer

Gold_01_PreProcessing reads Silver_02a clean data, applies scaling, train/test splitting, feature registry construction, and produces the canonical set of Gold preprocessing artifacts (5 Parquets + 4 feature/sensor JSON lists) used by all downstream Gold modeling notebooks. All Gold_02 through Gold_03c notebooks are rooted on Gold_01 outputs via a truth-record path override mechanism.

Gold_02 (Baseline) and Gold_03a/03b/03c (Cascade variants) each receive Gold_01 outputs and produce independent model results that converge in Gold_04 (Comparison). Gold_04 aggregates and compares all four model outputs and writes to `gold.model_comparison_results`.

Gold_05 (Anomaly Detection) follows Gold_04 in execution sequence but loads a selected model run's scored results independently to produce anomaly timeline analysis and early-warning lead-time summaries. Gold_06A replays held-out test data against all models from Gold_02 through Gold_03c to produce a final replay validation. Gold_06B applies early-warning analysis to Gold_06A's test replay outputs and optionally compares to Gold_05's training-run reference.

---

## Notebook Relationship Table

| Notebook | Receives From | Provides To | Main Handoff Object(s) | Relationship Type | Evidence Confidence |
|---|---|---|---|---|---|
| Bronze_01_Preprocessing | PostgreSQL (`capstone` schema via `read_layer_dataframe`) | Silver_01_PreEDA | Preprocessed Bronze Parquet; `bronze_preprocessing` truth record | Layer handoff | Confirmed from generated 071b reference |
| Silver_01_PreEDA | Bronze_01 Bronze Parquet | Silver_02a | Pre-EDA profile Parquets; null/distribution summaries; truth record | Layer handoff; EDA/profile support | Confirmed from generated 071b reference |
| Silver_02a_EDA_Building_Subsets_v3 | Silver_01 Parquet outputs | Silver_02b; Gold_01 | Clean subset Parquets; imputation recommendation JSON; truth record | Layer handoff; model input preparation | Confirmed from generated 071b reference |
| Silver_02b_EDA_v2 | Silver_02a subset Parquets | EDA record (no direct downstream pipeline notebook confirmed) | EDA analysis JSONs; distribution summaries; feature commentary | EDA/profile support | Confirmed from generated 071b reference |
| Gold_01_PreProcessing | Silver_02a clean Parquet | Gold_02; Gold_03a; Gold_03b; Gold_03c | Scaled Parquet; 4 feature JSON lists; `gold_preprocessing` truth record (with 8 path overrides) | Model input preparation; lineage/truth propagation | Confirmed from generated 071b reference |
| Gold_02_Baseline_Modeling | Gold_01 scaled Parquet + truth record | Gold_04 Comparison; Gold_06A | Baseline results CSV; baseline joblib model; validation contract; `gold_baseline` truth hash | Baseline modeling handoff; artifact handoff | Confirmed from generated 071b reference |
| Gold_03a_Cascade_Modeling | Gold_01 scaled Parquet + truth record (via path overrides) | Gold_04 Comparison; Gold_06A | Cascade results CSV/pickle; 2 joblib models; thresholds JSON; validation contract; `gold_cascade` truth hash | Cascade stage handoff; artifact handoff | Confirmed from generated 071b reference |
| Gold_03b_Cascade_Modeling | Gold_01 scaled Parquet + truth record (via path overrides) | Gold_03c (thresholds JSON); Gold_04; Gold_06A | Thresholds JSON (Stage 2 `previous_best`); cascade results; 2 joblib models; validation contract | Cascade stage handoff; artifact handoff | Confirmed from generated 071b reference |
| Gold_03c_Cascade_Modeling | Gold_01 scaled Parquet + truth; Gold_03b thresholds JSON + Stage 2 model + reference profile | Gold_04 Comparison; Gold_06A | Cascade results CSV/pickle; 4 validation contracts (default + 3 operating modes); `gold_cascade` truth hash | Cascade stage handoff; comparison/evaluation handoff | Confirmed from generated 071b reference |
| Gold_04_Comparison | Gold_02, Gold_03a, Gold_03b, Gold_03c (pickles, JSONs, truth records) | Operational reporting layer | Comparison CSV; summary JSON; statistical test files; `gold.model_comparison_results` SQL rows; `gold_comparison` truth hash | Comparison/evaluation handoff; SQL persistence handoff | Confirmed from generated 071b reference |
| Gold_05_Anomaly_Detection | Selected model run scored results (Gold_02 or Gold_03x output); does not consume Gold_04 outputs as pipeline inputs | Gold_06B (lead-time CSV, optional) | Anomaly timeline Parquet; detection summary JSON; multi-run lead-time comparison CSV; PNGs; truth record | Reporting/submission validation | Confirmed from generated 071b reference |
| Gold_06A_Test_Replay_Validation | Gold_01 scaled Parquet (test split); Gold_02 + Gold_03a/b/c models, thresholds, profiles, validation contracts | Gold_06B | Test replay scores CSV; replay validation comparison CSV; final validation status JSON | Audit/artifact support; reporting/submission validation | Confirmed from generated 071b reference |
| Gold_06B_Test_Early_Warning_Validation | Gold_06A replay scores CSV; Gold_05 lead-time CSV (optional) | None (terminal) | Early-warning summary CSV; lead-time comparison CSV; summary JSON; 2 plot PNGs | Reporting/submission validation | Confirmed from generated 071b reference |

---

## Cascade-Specific Relationship

### Gold_03a — Default Cascade

Gold_03a receives Gold_01's scaled Parquet, 5 Parquet splits, and 4 JSON feature/sensor lists via truth-record path overrides. It fits Stage 1 (broad Isolation Forest) and Stage 2 (fixed-configuration narrow IF) and applies rule-based Stage 3 confirmation with hardcoded parameters. It produces a `cascade_defaults` result set, 2 joblib models, thresholds JSON, a reference profile, and a validation contract for Gold_06A. Its `gold_cascade` truth hash is registered with `CASCADE_VARIANT="defaults"`. Gold_04 reads Gold_03a's results as one of its four model comparison inputs.

### Gold_03b — Tuned Cascade

Gold_03b receives the same Gold_01 inputs as Gold_03a (independently via its own truth-record path override). Its distinguishing feature is that Stage 2 configuration is config-driven rather than hardcoded: `STAGE2_SELECTION_MODE` can be `"fixed"`, `"threshold_grid"`, or `"parameter_search"`, enabling multi-candidate Stage 2 selection via `run_stage2_selection`. Gold_03b saves its best Stage 2 configuration in a thresholds JSON that Gold_03c reads as the `"previous_best"` Stage 2 source. It also writes a `CASCADE_VARIANT="tuned"` result set, 2 joblib models, a reference profile, a validation contract, and SQL rows with `model_stage="cascade_tuned_final"`.

### Gold_03c — Stage 3 Improved Cascade

Gold_03c receives Gold_01 inputs plus Gold_03b's thresholds JSON, Stage 2 model, and reference profile. It reuses Gold_03b's best Stage 2 selection (`STAGE2_SELECTION_SOURCE="previous_best"`) without re-training Stage 2. Its unique contribution is a tunable, weighted Stage 3 confirmation layer with multi-candidate parameter grid search and three calibrated operating modes: relaxed, medium, and strict. It produces four separate validation contracts (one per mode plus a default) for Gold_06A, and its cascade results are the final cascade output consumed by both Gold_04 Comparison and Gold_06A Test Replay Validation. Its `model_stage` in SQL is `"cascade_stage3_improved_final"`.

---

## Reporting and Validation Relationship

### Gold_04 — Comparison and Evaluation

Gold_04 aggregates all four model outputs (Gold_02 Baseline + Gold_03a/b/c Cascade variants) into a single comparison table. It performs three-source truth hash validation per model, cross-validates that all four models share `GOLD_PARENT_TRUTH_HASH`, runs McNemar and z-test statistical comparisons, and writes 7 comparison rows to `gold.model_comparison_results`. Gold_04 is the convergence point for all training-run model evaluation; it does not train, score, or route model decisions.

### Gold_05 — Anomaly Detection Reporting and Analysis

Gold_05 selects one model run (via `SELECTED_RUN_KEY`, default `"stage3_improved"`) and performs in-depth anomaly timeline analysis: episode annotation, failure phase detection, early-warning lead-time measurement, and multi-run lead-time comparison across all six model run keys. It produces visualizations, a structured detection summary, and an anomaly timeline Parquet intended for review and submission support. Gold_05 operates independently of Gold_04 — it does not consume Gold_04 comparison outputs as pipeline inputs. Its multi-run lead-time comparison CSV is optionally consumed by Gold_06B.

### Gold_06A — Test Replay Validation

Gold_06A replays all trained models against the held-out test set without retraining, confirms that replayed metrics match training-notebook summary artifacts within tolerance, and writes a row-level replay scores CSV for Gold_06B. It is the only Gold notebook that explicitly runs in `CONFIG_RUN_MODE="test"`. It validates the reproducibility of the full model pipeline on unseen data, covering all four models and all three Gold_03c operating modes.

### Gold_06B — Final Validation and Submission Readiness

Gold_06B applies Gold_05-style early-warning analysis to Gold_06A's test replay outputs, computing per-model alert lead times on held-out test data and optionally comparing them to Gold_05's training-run reference. It closes the Gold validation sequence. It does not train models, write to PostgreSQL, or construct a truth record. Its outputs are the final capstone submission artifacts: early-warning summary, lead-time comparison, and visualizations confirming model behavior on unseen data.

---

## Unresolved or Source-Limited Relationships

- **Silver_02b → Gold_01 direct pipeline dependency**: Silver_02b's EDA outputs inform feature selection context, but a direct artifact-level dependency (e.g., Silver_02b Parquet loaded by Gold_01) is not confirmed from available 071b references or manifests. Marked as: "Not determined from available source."
- **Gold_04 → Gold_05 direct handoff**: Gold_05 does not consume Gold_04 comparison outputs as pipeline inputs per the 071b reference. The execution sequence relationship (Gold_05 follows Gold_04) is confirmed, but no direct artifact handoff from Gold_04 to Gold_05 is confirmed from available source.
- **W&B artifact cross-notebook dependencies**: All Gold modeling notebooks open W&B runs and log artifacts. Whether any downstream notebook reads W&B artifacts from a prior notebook's run (rather than disk Parquets/JSONs) is not determined from available source.
