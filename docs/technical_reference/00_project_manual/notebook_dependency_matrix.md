# Notebook Dependency Matrix

## Purpose

This matrix documents verified upstream dependencies and downstream consumers for each active notebook in the Bronze → Silver → Gold pipeline. Claims are limited to dependencies confirmed from 071b workflow references, `artifact_io_manifest.json`, `sql_touchpoints.json`, or notebook inventory. Direct dependency means the notebook loads or reads the artifact in a code cell; sequential relationship alone is not treated as a confirmed dependency.

---

## Dependency Matrix

| Notebook | Upstream Dependencies | Downstream Consumers | Shared Config / Runtime Context | Notes | Evidence Confidence |
|---|---|---|---|---|---|
| **Bronze_01_Preprocessing** | PostgreSQL `capstone` schema (synthetic telemetry via `read_layer_dataframe`); `load_notebook_context("bronze_preprocessing")` | Silver_01_PreEDA | `BRONZE_CFG`; `DEFAULT_FALLBACKS`; `DATASET_ID`; `RUN_ID`; `ASSET_ID` from env/config; `CONFIG_PROFILE` | Reads synthetic pump telemetry from the operational PostgreSQL layer; no file-based upstream notebook dependency | Confirmed from generated 071b reference |
| **Silver_01_PreEDA** | Bronze_01 Bronze Parquet (via `read_layer_dataframe` or `load_data`); `bronze_preprocessing` truth record (parent context); `load_notebook_context("silver_pre_eda")` | Silver_02a | `SILVER_CFG`; `DATASET_ID`; `RUN_ID`; pipeline mode from config | Reads Bronze layer data; produces pre-EDA profile artifacts consumed by Silver_02a | Confirmed from generated 071b reference |
| **Silver_02a_EDA_Building_Subsets_v3** | Silver_01 pre-EDA profile Parquets; `load_notebook_context("silver_eda")` | Silver_02b; Gold_01 | `SILVER_EDA_CFG`; `DATASET_ID`; `RUN_ID`; `CONFIG_PROFILE` | Produces clean analytical Parquet and imputation recommendation JSON both consumed by Gold_01; also feeds Silver_02b | Confirmed from generated 071b reference |
| **Silver_02b_EDA_v2** | Silver_02a subset Parquets; `load_notebook_context("silver_eda")` | No confirmed downstream pipeline notebook | `SILVER_EDA_CFG`; `DATASET_ID`; `RUN_ID` | EDA reference; provides distribution and feature analysis context but direct artifact-level dependency from Gold_01 is not confirmed | Confirmed from generated 071b reference |
| **Gold_01_PreProcessing** | Silver_02a clean Parquet; imputation recommendation JSON; `load_notebook_context("gold_preprocessing")` | Gold_02; Gold_03a; Gold_03b; Gold_03c | `GOLD_PREPROCESSING_CFG`; `GOLD_VERSION`; `TRUTH_VERSION`; `RECIPE_ID`; `DATASET_ID`; `RUN_ID`; `PIPELINE_MODE`; `CONFIG_PROFILE` | Truth record carries 8 artifact path overrides consumed by all downstream Gold notebooks; `GOLD_PARENT_TRUTH_HASH` shared across Gold modeling layer | Confirmed from generated 071b reference |
| **Gold_02_Baseline_Modeling** | Gold_01 scaled Parquet; Gold_01 truth record (path overrides for 4 additional Parquets + 2 feature JSONs); `load_notebook_context("gold_baseline")` | Gold_04 Comparison; Gold_06A | `BASELINE_CFG`; `GOLD_VERSION`; `TRUTH_VERSION`; `RECIPE_ID`; `DATASET_ID`; `RUN_ID`; `GOLD_PARENT_TRUTH_HASH` inherited from Gold_01 | No Stage 2 or Stage 3 — single Isolation Forest with threshold calibration; `BASELINE_TRUTH_HASH` registered in Gold_04 | Confirmed from generated 071b reference |
| **Gold_03a_Cascade_Modeling** | Gold_01 scaled Parquet; Gold_01 truth record (8 path overrides: 5 Parquets + 4 feature/sensor JSONs); `load_notebook_context("gold_cascade")` | Gold_04 Comparison; Gold_06A | `CASCADE_CFG`; `CASCADE_VARIANT="defaults"` (hardcoded); `GOLD_VERSION`; `GOLD_PARENT_TRUTH_HASH` | Fixed Stage 2 selection; produces `CASCADE_DEFAULTS_TRUTH_HASH` | Confirmed from generated 071b reference |
| **Gold_03b_Cascade_Modeling** | Gold_01 scaled Parquet; Gold_01 truth record (8 path overrides); `load_notebook_context("gold_cascade")` | Gold_03c (thresholds JSON + Stage 2 model + reference profile); Gold_04 Comparison; Gold_06A | `CASCADE_CFG`; `CASCADE_VARIANT="tuned"` (hardcoded); `STAGE2_SELECTION_MODE` from config; `GOLD_PARENT_TRUTH_HASH` | Multi-candidate Stage 2 selection; thresholds JSON is direct dependency for Gold_03c `STAGE2_SELECTION_SOURCE="previous_best"` | Confirmed from generated 071b reference |
| **Gold_03c_Cascade_Modeling** | Gold_01 scaled Parquet; Gold_01 truth record (8 path overrides); Gold_03b thresholds JSON (`CASCADE_TUNED_THRESHOLDS_PATH`); Gold_03b Stage 2 model (`CASCADE_TUNED_STAGE2_MODEL_PATH`); Gold_03b reference profile (`CASCADE_TUNED_REFERENCE_PROFILE_PATH`); `load_notebook_context("gold_cascade")` | Gold_04 Comparison; Gold_06A (4 validation contracts) | `CASCADE_CFG`; `CASCADE_VARIANT="stage3_improved"` (hardcoded); `STAGE2_SELECTION_SOURCE="previous_best"`; `STAGE3_WEIGHT_GRID`; `GOLD_PARENT_TRUTH_HASH` | Only Gold modeling notebook with direct file-level dependency on another Gold modeling notebook (Gold_03b); produces 4 validation contracts for Gold_06A | Confirmed from generated 071b reference |
| **Gold_04_Comparison** | Gold_02 results pickle + metadata JSON + truth record; Gold_03a results pickle + metadata JSON + truth record; Gold_03b results pickle + metadata JSON + truth record; Gold_03c results pickle + metadata JSON + truth record; `load_notebook_context("gold_comparison")` | `gold.model_comparison_results` (SQL); W&B run (audit) | `COMPARISON_CFG`; `GOLD_VERSION`; `GOLD_PARENT_TRUTH_HASH` cross-validated against all 4 upstream truth records | Does not route model decisions; aggregates and compares; `GOLD_PARENT_TRUTH_HASH` must be shared by all four upstream models | Confirmed from generated 071b reference |
| **Gold_05_Anomaly_Detection** | Scored results for `SELECTED_RUN_KEY` model (Gold_02 or Gold_03a/b/c output, loaded by run key); `load_notebook_context("gold_anomaly_detection")` | Gold_06B (multi-run lead-time comparison CSV, optional) | `ANOMALY_DETECTION_CFG`; `SELECTED_RUN_KEY` (default `"stage3_improved"`); `RECOVERY_STABILITY_ROWS`; `RUN_CONFIG_MAP`; `DATASET_ID`; `RUN_ID` | Does not consume Gold_04 outputs as pipeline inputs; does not open a W&B run; truth record does not chain to Gold_06A/06B | Confirmed from generated 071b reference |
| **Gold_06A_Test_Replay_Validation** | Gold_01 scaled Parquet (`FULL_SCALED_PATH` with `meta__is_train_flag`; or `TEST_DATA_PATH` fallback); fitted Stage 1/Stage 2 models for all 4 variants; thresholds, summaries, reference profiles, configs for all 4 variants; validation contracts for all 4 variants; `load_notebook_context()` | Gold_06B | `CONFIG_RUN_MODE="test"`; `DATASET_NAME`; `GOLD_ROOT`; `METRIC_TOLERANCE=1e-9`; `METRIC_TOLERANCE_RELAXED=0.0001` | Does not open W&B run, write to PostgreSQL, or construct a truth record; two-pass metric tolerance check; Gold_06A is the only Gold notebook with confirmed test-mode execution | Confirmed from generated 071b reference |
| **Gold_06B_Test_Early_Warning_Validation** | Gold_06A test replay scores CSV (`REPLAY_SCORES_PATH` — required); Gold_05 multi-run lead-time comparison CSV (`TRAIN_LEAD_TIME_PATH` — optional); `load_notebook_context()` | None (terminal) | `CONFIG_RUN_MODE="test"`; `DATASET_NAME`; `GOLD_ROOT`; `VALIDATION_ROOT` shared with Gold_06A | Terminal notebook; no downstream pipeline notebook; does not write to SQL, open W&B, or produce a truth record; Gold_05 comparison is fully optional | Confirmed from generated 071b reference |

---

## Shared Runtime Context

All notebooks use `load_notebook_context()` as the single bootstrap call. This provides:

- `CTX` with config, paths, logger, ledger, and DB engine factory
- `DATASET_ID`, `RUN_ID`, `ASSET_ID`, `CAPSTONE_SCHEMA` from env/config
- `CONFIG_PROFILE` and `PIPELINE_MODE` (execution mode) from config
- `paths` object resolving `artifacts/`, `logs/`, `truths/` roots

These shared context objects are consistent across all notebooks per the `load_notebook_context()` bootstrap pattern. Variations per notebook are in stage-specific config sections (e.g., `BRONZE_CFG`, `GOLD_PREPROCESSING_CFG`, `CASCADE_CFG`).

Source: Confirmed from generated 071b references.

---

## Truth Hash Dependency Chain

| Notebook | Truth Layer | Truth Hash Variable | Parent Hash Source | Evidence Confidence |
|---|---|---|---|---|
| Bronze_01 | `bronze_preprocessing` | `BRONZE_TRUTH_HASH` | Not confirmed from available source | Confirmed from generated 071b reference |
| Silver_01 | `silver_pre_eda` | Not determined | Not determined from available source | Not determined from available source |
| Silver_02a | `silver_eda` | Not determined | Not determined from available source | Not determined from available source |
| Silver_02b | `silver_eda` | Not determined | Not determined from available source | Not determined from available source |
| Gold_01 | `gold_preprocessing` | `GOLD_PREPROCESSING_TRUTH_HASH` | Not determined from available source | Confirmed from generated 071b reference |
| Gold_02 | `gold_baseline` | `BASELINE_TRUTH_HASH` | `GOLD_PARENT_TRUTH_HASH` (from Gold_01 truth record) | Confirmed from generated 071b reference |
| Gold_03a | `gold_cascade` | `CASCADE_DEFAULTS_TRUTH_HASH` | `GOLD_PARENT_TRUTH_HASH` (from Gold_01 truth record) | Confirmed from generated 071b reference |
| Gold_03b | `gold_cascade` | `CASCADE_TUNED_TRUTH_HASH` | `GOLD_PARENT_TRUTH_HASH` (from Gold_01 truth record) | Confirmed from generated 071b reference |
| Gold_03c | `gold_cascade` | `CASCADE_STAGE3_IMPROVED_TRUTH_HASH` | `GOLD_PARENT_TRUTH_HASH` (from Gold_01 truth record) | Confirmed from generated 071b reference |
| Gold_04 | `gold_comparison` | `COMPARISON_TRUTH_HASH` | `GOLD_PARENT_TRUTH_HASH` (cross-validated across all 4 models) | Confirmed from generated 071b reference |
| Gold_05 | `gold_anomaly_detection` | Not determined from available source | `SELECTED_RUN_KEY` model's truth hash | Confirmed from generated 071b reference |
| Gold_06A | None | None | N/A — no truth record | Confirmed from generated 071b reference |
| Gold_06B | None | None | N/A — no truth record | Confirmed from generated 071b reference |
