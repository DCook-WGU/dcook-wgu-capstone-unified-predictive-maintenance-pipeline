# EDA_Notebook_Pump_Gold_04_Comparison — Workflow Reference

**Source notebook:** `notebooks/experiments/EDA_Notebook_Pump_Gold_04_Comparison.ipynb`
**Stage:** Gold — Model Comparison
**Layer:** Gold
**Reference type:** Workflow-level

---

## Notebook Purpose

Gold_04 is the cross-model comparison and evaluation notebook. It does not train any model, score any rows, or apply any anomaly detection logic. Its job is to collect the saved output artifacts from the four upstream Gold modeling notebooks, align them against a shared lineage anchor, compute a unified comparison table, run statistical significance tests, produce comparison charts and summary reports, stamp a comparison-stage truth record, and persist the comparison payload to both disk and PostgreSQL.

The four models compared are:

| Notebook | Model ID in comparison | Variant |
|---|---|---|
| Gold_02 | `baseline` | Single Isolation Forest |
| Gold_03a | `cascade_default` | Cascade — default settings |
| Gold_03b | `cascade_tuned` | Cascade — tuned Stage 2 |
| Gold_03c | `stage3_improved` / `stage3_relaxed` / `stage3_medium` / `stage3_strict` | Cascade — Stage 3 confirmation layer |

Gold_04 produces seven comparison rows — one for each model or Stage 3 operating mode variant. All comparison rows carry the same shared `parent_gold_truth_hash`, establishing that all models were evaluated against the same Gold-layer feature data lineage.

Gold_04 does not declare a winning model or direct which model carries forward into Gold_05. The summary records `best_model_by_precision`, `best_model_by_recall`, `best_model_by_f1`, and `best_model_by_alert_reduction` fields, but these are informational identifiers derived from the comparison table, not pipeline-routing flags.

---

## Pipeline Role

- Stage: `gold_comparison`
- Layer: Gold
- Position in workflow: Runs after Gold_02 (Baseline Modeling), Gold_03a, Gold_03b, and Gold_03c (Cascade variants); before Gold_05 (Anomaly Detection)
- Primary responsibility: Aggregate all saved modeling outputs, validate truth hash lineage across all four upstream runs, compute a unified 7-row comparison table with statistical tests, produce comparison charts, stamp a `gold_comparison` truth record, and write rows to `gold.model_comparison_results`
- Does not train models, score rows, or determine which model routes into Gold_05

## Configuration and Runtime Context

| Item | Source | Value / Purpose |
|---|---|---|
| `CONTEXT_STAGE` | Notebook constant | `"gold_comparison"` |
| `COMPARISON_CFG` | `CTX.stage_config` | Stage-specific config section for Gold_04 |
| `PIPELINE_MODE` | `PIPELINE["execution_mode"]`; overridden from baseline truth if non-None | Execution mode inherited from Gold_01 |
| `GOLD_VERSION`, `TRUTH_VERSION`, `RECIPE_ID` | Version and stage config sections | Artifact versioning identifiers |
| `GOLD_PROCESS_RUN_ID` | `make_process_run_id(COMPARISON_CFG["process_run_id_prefix"])` | Run identity stamp |
| `DATASET_NAME` | `DATASET_CFG`; confirmed from all four upstream truth records | `ValueError` if any truth record disagrees |
| `GOLD_PARENT_TRUTH_HASH` | Anchored on `CASCADE_DEFAULTS_TRUTH_HASH` parent; cross-validated against all 4 models | Shared lineage anchor |
| `CAPSTONE_SCHEMA`, `DATASET_ID`, `RUN_ID`, `ASSET_ID` | Env vars / config; `first_non_empty_string` fallback chain | SQL write targets and row identity |

## Section Overview

| Section | Purpose | Key Outputs |
|---|---|---|
| Bootstrap | Context, dirs, DB, W&B | `CTX`, `engine`, `wandb_run`, sanity checks |
| Upstream data load | Pickles and JSON artifacts for all 4 models | 4 result DataFrames; 12 JSON dicts |
| Truth hash validation | Three-source check per model | 4 validated stage truth hashes |
| Lineage cross-check | All 4 models share `GOLD_PARENT_TRUTH_HASH` | `GOLD_PARENT_TRUTH_HASH` |
| Comparison table | 7-row metric table across all models | `comparison_df` |
| Statistical tests | McNemar + z-tests (baseline vs. Stage 3 Improved) | `statistical_test_dataframe`, `statistical_test_summary` |
| Stage 3 analysis | Operating mode scatter plot | PNG |
| Cascade funnel charts | Per-model stage-by-stage alert counts | PNGs |
| Truth record + artifact saves | Stamp, persist, W&B upload | `COMPARISON_TRUTH_HASH`, CSV, JSON |
| Ledger / W&B close | Finalize run record | Ledger JSONL |
| Lineage invariant checks | Post-close hash verification | `ValueError` on mismatch |
| SQL write | Write 7 comparison rows | `gold.model_comparison_results` |

## Section Details

## 2. Context Stage and Bootstrap

The notebook declares `CONTEXT_STAGE = "gold_comparison"` and bootstraps runtime context with:

```python
load_notebook_context(
    stage="gold_comparison",
    logger_child_name="capstone.gold.comparison",
    log_filename="gold_model_comparison.log",
)
```

This produces the `CTX` object and populates `paths`, `CONFIG`, `CONFIG_MAP`, `RESOLVED_PATHS`, `VERSIONS_CFG`, `WANDB_CFG`, `DATASET_CFG`, `RUNTIME_CFG`, `PIPELINE`, `FILENAMES`, and the shared `logger`.

`COMPARISON_CFG = CTX.stage_config` holds the stage-specific configuration section for Gold_04.

Two sequential sanity checks guard the bootstrap:

- A general context check verifying `CTX`, `paths`, `CONFIG`, `CONFIG_MAP`, and related keys.
- A Gold-specific check verifying that `COMPARISON_CFG` is populated before any comparison variables are resolved.

---

## 3. Configuration Block

After bootstrap, the notebook resolves a large number of named variables from config and `RESOLVED_PATHS`. Key variables include:

- `STAGE = "gold"`, `LAYER_NAME` from config
- `GOLD_VERSION`, `TRUTH_VERSION`, `RECIPE_ID` from version and stage config sections
- `PIPELINE_MODE` from `PIPELINE["execution_mode"]`, later overridden if the loaded baseline truth record carries a different mode
- `GOLD_PROCESS_RUN_ID` via `make_process_run_id(COMPARISON_CFG["process_run_id_prefix"])`
- `DATASET_NAME` from `DATASET_CFG`, later re-confirmed from the baseline truth record
- `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME`
- Per-model filename variables resolved from `FILENAMES` for all four upstream models (CSV, pickle, summary, thresholds, metadata)
- Output filename variables: `MODEL_COMPARISON_FILE_NAME`, `MODEL_COMPARISON_SUMMARY_FILE_NAME`, `COMPARISON_PLOT_WITH_TEST_ALERTS_FILE_NAME`, `GOLD_COMPARISON_LEDGER_FILE_NAME`

Input file paths for all four upstream models are resolved from `RESOLVED_PATHS`. Backward-compatible `CASCADE_STAGE3_*` aliases point to the same paths as the canonical `CASCADE_STAGE3_IMPROVED_*` names, since Gold_03c renamed them between notebook versions.

---

## 4. Artifact Directory Setup

Gold_04 uses `build_artifact_dirs` (not `build_artifact_dirs_from_config`) because it must reference artifact directory layouts for multiple upstream model families that may not appear in Gold_04's own config section:

```python
GOLD_BASELINE_ARTIFACT_DIRS = build_artifact_dirs(
    artifacts_root=ARTIFACTS_ROOT,
    stage="gold",
    dataset_name=DATASET_NAME,
    family="baseline",
    subdirs=GOLD_COMMON_SUBDIRS,  # scores, summaries, thresholds, metadata, models, plots, config, lineage
)

GOLD_CASCADE_DEFAULTS_ARTIFACT_DIRS = build_artifact_dirs(..., family="cascade_defaults", ...)
GOLD_CASCADE_TUNED_ARTIFACT_DIRS    = build_artifact_dirs(..., family="cascade_tuned", ...)
GOLD_CASCADE_STAGE3_ARTIFACT_DIRS   = build_artifact_dirs(..., family="cascade_stage3_improved", ...)

GOLD_COMPARISON_ARTIFACT_DIRS = build_artifact_dirs(
    artifacts_root=ARTIFACTS_ROOT,
    stage="gold",
    dataset_name=DATASET_NAME,
    family="comparison",
    subdirs=GOLD_COMPARISON_SUBDIRS,  # results, summaries, plots, statistics, metadata, config, lineage
)
```

The first four families (`baseline`, `cascade_defaults`, `cascade_tuned`, `cascade_stage3_improved`) are used as read-only input directory references. The `comparison` family provides all output paths used by Gold_04.

Input file paths for each upstream model are then composed by combining the appropriate artifact family directory key with the filename variable. For example:

```python
BASELINE_RESULTS_PATH_CSV    = GOLD_BASELINE_ARTIFACT_DIRS["scores"] / FILENAMES["baseline_results_file_name_csv"]
CASCADE_TUNED_SUMMARY_PATH   = GOLD_CASCADE_TUNED_ARTIFACT_DIRS["summaries"] / FILENAMES["cascade_tuned_summary_file_name"]
```

Output paths are composed from `GOLD_COMPARISON_ARTIFACT_DIRS`:

```python
MODEL_COMPARISON_PATH              = GOLD_COMPARISON_ARTIFACT_DIRS["results"]    / ...
MODEL_COMPARISON_SUMMARY_PATH      = GOLD_COMPARISON_ARTIFACT_DIRS["summaries"]  / ...
STATISTICAL_TEST_SUMMARY_PATH      = GOLD_COMPARISON_ARTIFACT_DIRS["statistics"] / ...
STATISTICAL_TEST_TABLE_PATH        = GOLD_COMPARISON_ARTIFACT_DIRS["statistics"] / ...
comparison_ledger_path             = GOLD_COMPARISON_ARTIFACT_DIRS["lineage"]    / ...
```

A config snapshot is saved at `GOLD_COMPARISON_CONFIG_DIR / f"{DATASET_NAME}__gold_comparison__resolved_config.yaml"` when `CONFIG["execution"]["save_config_snapshot"]` is True (default True).

---

## 5. Database Connection and Asset Resolution

`engine = get_engine_from_env()` establishes the PostgreSQL connection. `CAPSTONE_SCHEMA`, `DATASET_ID`, `RUN_ID`, and `ASSET_ID` are resolved from environment variables and `CONFIG`, with `first_non_empty_string` selecting the first non-empty value across candidate sources.

A SQL smoke check (`read_sql_dataframe`) confirms the database connection is live before any data loading begins. `log_layer_paths` records resolved path state to the logger.

---

## 6. W&B Initialization and Ledger Setup

```python
wandb_run = wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name=WANDB_RUN_NAME,
    job_type="gold_model_comparison",
    config={
        "gold_version": GOLD_VERSION,
        "dataset": DATASET_NAME,
        "stage": STAGE,
        # all 4 upstream input paths logged to W&B config
        ...
    },
)
```

All upstream input paths (results CSV, results pickle, summary, thresholds, metadata for all four models) are included in the W&B run config at initialization time.

A `Ledger` is initialized with `stage=STAGE` and `recipe_id=RECIPE_ID`. Ledger steps are added throughout the notebook as named events with `kind`, `step`, `message`, and `data` fields.

---

## 7. Upstream Data Loading

Gold_04 loads two categories of upstream artifacts for each model:

**Row-level result DataFrames (pickle):** Each upstream model's scored output is loaded with `pd.read_pickle`. These DataFrames carry `meta__row_id`, truth hash columns, flag columns, and anomaly labels.

**Structured summary records (JSON):** Each model's summary, thresholds, and metadata are loaded with `load_json`, then validated with the inline helper `require_truth_record`. This helper raises `ValueError` if the loaded object is not a non-empty dict.

```python
baseline_results          = pd.read_pickle(BASELINE_RESULTS_PATH_PICKLE)
baseline_summary          = require_truth_record(load_json(BASELINE_SUMMARY_PATH),     "baseline_summary")
baseline_thresholds       = require_truth_record(load_json(BASELINE_THRESHOLDS_PATH),  "baseline_thresholds")
baseline_metadata         = require_truth_record(load_json(BASELINE_METADATA_PATH),    "baseline_metadata")

cascade_defaults_results  = pd.read_pickle(CASCADE_DEFAULTS_RESULTS_PATH_PICKLE)
cascade_defaults_summary  = require_truth_record(load_json(CASCADE_DEFAULTS_SUMMARY_PATH), ...)
...
cascade_stage3_improved_results = pd.read_pickle(CASCADE_STAGE3_IMPROVED_RESULTS_PATH_PICKLE)
...
```

All four models follow the same pattern.

---

## 8. Truth Hash Validation

Gold_04 performs a three-source truth hash check for each upstream model before any comparison logic runs.

For baseline, this involves:

1. Extracting `BASELINE_TRUTH_HASH` and `BASELINE_TRUTH_PATH` from `baseline_metadata`.
2. Loading the truth file at that path and validating the file's `truth_hash` field matches `BASELINE_TRUTH_HASH`.
3. Calling `extract_truth_hash(baseline_results)` on the loaded pickle DataFrame and validating the extracted hash matches `BASELINE_TRUTH_HASH`.

A mismatch at any of the three sources raises `ValueError` with a diagnostic message. The same three-source check is applied to `cascade_defaults`, `cascade_tuned`, and `cascade_stage3_improved`.

The cascade truth records use `"cascade_truth_hash"` and `"cascade_truth_path"` as metadata keys (compared to `"baseline_truth_hash"` / `"baseline_truth_path"` for the baseline model).

---

## 9. Lineage Cross-Check and Gold Parent Truth Resolution

After loading all four model truth records, Gold_04 calls `get_parent_truth_hash` on each to extract its `parent_truth_hash`. A `lineage_df` DataFrame is built for display, listing `model_id`, `stage_truth_hash`, `parent_gold_truth_hash`, `results_truth_hash`, and `result_rows` for all four models.

A series of strict equality checks then confirms that all four models share the same parent Gold truth hash:

```python
COMPARISON_PARENT_GOLD_TRUTH_HASH = DEFAULTS_PARENT_GOLD_TRUTH_HASH  # anchored on cascade_defaults

if TUNED_PARENT_GOLD_TRUTH_HASH != COMPARISON_PARENT_GOLD_TRUTH_HASH:
    raise ValueError(...)
if STAGE3_IMPROVED_PARENT_GOLD_TRUTH_HASH != COMPARISON_PARENT_GOLD_TRUTH_HASH:
    raise ValueError(...)
if BASELINE_PARENT_GOLD_TRUTH_HASH != COMPARISON_PARENT_GOLD_TRUTH_HASH:
    raise ValueError(...)
```

`cascade_defaults` is used as the canonical lineage anchor because all cascade variants must share the same parent Gold truth hash, and defaults is the simplest variant to anchor on.

Once all checks pass, `GOLD_PARENT_TRUTH_HASH = COMPARISON_PARENT_GOLD_TRUTH_HASH`. The notebook then resolves and loads the shared Gold truth file at `TRUTHS_PATH / "gold" / f"{DATASET_NAME}__gold__truth__{GOLD_PARENT_TRUTH_HASH}.json"`, raising `FileNotFoundError` if it does not exist.

Dataset name consistency is also enforced: all four truth records must agree on `DATASET_NAME` or `ValueError` is raised.

A ledger step `"load_comparison_inputs"` records all resolved input paths, truth hashes, parent truth hashes, and row counts for all four models.

---

## 10. Comparison Table Construction

Metrics are extracted from saved summary dicts rather than recomputed from raw scores, so the comparison table matches exactly what the individual training notebooks reported.

```python
baseline_metrics               = baseline_summary["baseline_metrics"]
cascade_default_metrics        = cascade_defaults_summary["cascade_metrics"]
cascade_tuned_metrics          = cascade_tuned_summary["cascade_metrics"]
cascade_stage3_improved_metrics = cascade_stage3_improved_summary["cascade_metrics"]
```

Seven comparison rows are constructed and assembled into `comparison_df`:

| `model_id` | `model` | `variant_family` | `stage3_mode` | Alert count source |
|---|---|---|---|---|
| `baseline` | Baseline IsolationForest | baseline | none | `baseline_summary["alert_count_test_rows"]` |
| `cascade_default` | Cascade Default | cascade | none | `cascade_defaults_summary["final_cascade_alert_count_test_rows"]` |
| `cascade_tuned` | Cascade Tuned | cascade | none | `cascade_tuned_summary["final_cascade_alert_count_test_rows"]` |
| `stage3_improved` | Stage 3 Improved | cascade_stage3 | selected_improved | `cascade_stage3_improved_summary["final_cascade_alert_count_test_rows"]` |
| `stage3_relaxed` | Stage 3 Relaxed | cascade_stage3 | relaxed | `cascade_stage3_improved_summary["stage3_relaxed_alert_count_test_rows"]` |
| `stage3_medium` | Stage 3 Medium | cascade_stage3 | medium | `cascade_stage3_improved_summary["stage3_medium_alert_count_test_rows"]` |
| `stage3_strict` | Stage 3 Strict | cascade_stage3 | strict | `cascade_stage3_improved_summary["stage3_strict_alert_count_test_rows"]` |

Each row carries `precision`, `recall`, `f1`, `stage_truth_hash` (the model's own stage hash), and `parent_gold_truth_hash` (`GOLD_PARENT_TRUTH_HASH`).

---

## 11. Comparison Summary Construction

`comparison_summary` is a flat dict assembled from the comparison table. It contains:

- Alert counts for all 7 model rows
- Pairwise alert reduction counts and ratios relative to baseline: baseline-vs-default, baseline-vs-tuned, baseline-vs-stage3-improved, baseline-vs-stage3-relaxed, baseline-vs-stage3-medium, baseline-vs-stage3-strict
- Cross-variant change counts: default-vs-tuned, tuned-vs-stage3-improved, tuned-vs-stage3-relaxed, tuned-vs-stage3-medium, tuned-vs-stage3-strict
- Per-model precision, recall, and F1 for all seven rows
- Best-model identifiers (`best_model_by_precision`, `best_model_by_recall`, `best_model_by_f1`, `best_model_by_alert_reduction`) derived by sorting the comparison table
- All four model stage truth hashes and truth paths
- `gold_truth_hash` and `gold_truth_path`

These fields are carried forward into the comparison truth record's `runtime_facts` section.

---

## 12. Statistical Significance Testing

Gold_04 defines a family of inline statistical helper functions and runs three significance tests comparing the baseline (Gold_02) against Stage 3 Improved (Gold_03c):

**`build_paired_model_frame`:** Merges baseline and comparison results on `meta__row_id` using a validated one-to-one inner join, then filters to test rows (`meta__is_train_flag == False`). Raises `ValueError` if any merge row loss occurs or if `meta__row_id` is not unique in either frame.

**`run_mcnemar_paired_test`:** Builds a 2×2 contingency table of row-level correctness patterns (both correct / baseline-only correct / comparison-only correct / both wrong). Applies McNemar's exact test (`statsmodels`) when `discordant_count < 25`, otherwise uses the chi-square approximation. Falls back to a manual continuity-corrected McNemar formula when `statsmodels` is unavailable.

**`run_two_proportion_z_test`:** Two-proportion z-test used for aggregate false-positive rate and precision comparisons.

**`build_statistical_test_summary`:** Orchestrates all three tests and returns both a report DataFrame and a summary dict. The paired McNemar test is the primary comparison; the two z-tests are secondary aggregate-level checks.

```python
statistical_test_dataframe, statistical_test_summary = build_statistical_test_summary(
    baseline_results=baseline_results,
    comparison_results=cascade_stage3_improved_results,
    comparison_name="Stage 3 Improved",
    comparison_flag_column="cascade_final_flag",
)
```

The statistical results are saved to:

- `STATISTICAL_TEST_TABLE_PATH` — CSV via `statistical_test_dataframe.to_csv`
- `STATISTICAL_TEST_SUMMARY_PATH` — JSON via `save_json`

Both files reside under `GOLD_COMPARISON_ARTIFACT_DIRS["statistics"]`.

---

## 13. Stage 3 Operating Mode Analysis

A subset of `comparison_df` filtered to `model_id in ["stage3_improved", "stage3_relaxed", "stage3_medium", "stage3_strict"]` is displayed and plotted. A scatter chart of test alert count (x-axis) versus F1 score (y-axis) annotates each operating mode, allowing visual inspection of the alert-burden / accuracy trade-off across the three Stage 3 threshold settings.

The plot is saved to `GOLD_COMPARISON_ARTIFACT_DIRS["plots"] / f"{DATASET_NAME}__gold__stage3_operating_modes_alerts_vs_f1.png"`.

---

## 14. Cascade Funnel Charts

`build_cascade_funnel_dataframe` constructs a stage-by-stage alert count series from saved cascade summary values for each cascade model:

```
Stage 1 Broad IF → Stage 2 Narrow IF → Final Confirmed Alerts
```

Funnel DataFrames for cascade_defaults (Gold_03a), cascade_tuned (Gold_03b), and cascade_stage3_improved (Gold_03c) are concatenated and grouped by model. Per-model bar charts are generated and saved under `GOLD_COMPARISON_ARTIFACT_DIRS["plots"]` with filename pattern `{DATASET_NAME}__gold__{safe_model_name}__cascade_funnel.png`.

---

## 15. Alert Count Comparison Bar Chart

After the comparison summary is saved to disk, the notebook re-reads `MODEL_COMPARISON_SUMMARY_PATH` via `load_json` and uses the loaded dict to drive a seven-bar comparison chart:

```
Baseline | Cascade Default | Cascade Tuned | Stage 3 Improved | Stage 3 Relaxed | Stage 3 Medium | Stage 3 Strict
```

The chart is saved to `GOLD_COMPARISON_ARTIFACT_DIRS["plots"] / f"{DATASET_NAME}__gold__alert_count_comparison_test_rows.png"`.

---

## 16. Truth Record Construction, Stamping, and Artifact Saves

Gold_04 constructs its own comparison-stage truth record using the standard truth chain functions:

```python
comparison_truth = initialize_layer_truth(
    truth_version=TRUTH_VERSION,
    dataset_name=DATASET_NAME,
    layer_name="gold_comparison",
    process_run_id=comparison_process_run_id,
    pipeline_mode=PIPELINE_MODE,
    parent_truth_hash=GOLD_PARENT_TRUTH_HASH,
)
```

The truth record is populated with three sections:

- `config_snapshot`: the full `TRUTH_CONFIG` dict (or a minimal runtime block if `TRUTH_CONFIG` is not a dict)
- `runtime_facts`: comparison row counts, shared Gold truth hash, all four model stage truth hashes, and the four best-model identifiers
- `artifact_paths`: all upstream input paths and the two output paths (comparison CSV and summary JSON)

`build_truth_record` finalizes the record and stamps `COMPARISON_TRUTH_HASH`. The truth hash is then propagated into `comparison_df`:

```python
comparison_df = stamp_truth_columns(
    comparison_df,
    truth_hash=COMPARISON_TRUTH_HASH,
    parent_truth_hash=GOLD_PARENT_TRUTH_HASH,
    pipeline_mode=PIPELINE_MODE,
)
```

The truth record is saved via `save_truth_record` and indexed via `append_truth_index`. The comparison summary dict is updated with `comparison_truth_hash`, `comparison_truth_path`, and `comparison_process_run_id`.

Artifact saves in this step:

- `comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)` — stamped comparison table
- `save_json(comparison_summary, MODEL_COMPARISON_SUMMARY_PATH)` — full comparison summary
- `wandb_run.save(MODEL_COMPARISON_PATH)` / `wandb_run.save(MODEL_COMPARISON_SUMMARY_PATH)` / `wandb_run.save(comparison_truth_path)` — W&B uploads

A ledger step `"save_comparison_outputs"` records all output paths and both truth hashes.

---

## 17. Ledger Finalization and W&B Close

```python
ledger.add(
    kind="step",
    step="finalize_comparison",
    message="Gold comparison notebook complete.",
    data={
        "comparison_csv": str(MODEL_COMPARISON_PATH),
        "comparison_summary_json": str(MODEL_COMPARISON_SUMMARY_PATH),
        "comparison_summary": comparison_summary,
    },
    logger=logger,
)

ledger.write_json(comparison_ledger_path)
wandb.save(str(comparison_ledger_path))
wandb_run.finish()
```

`wandb_run.finish()` fires at this point — before the lineage invariant checks and before the SQL write.

---

## 18. Lineage Invariant Checks

After `wandb_run.finish()`, a post-completion validation block checks the stamped `comparison_df` and the saved truth file:

- `required_comparison_meta_columns` — verifies `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` are present in `comparison_df`
- `extract_truth_hash(comparison_df)` — checks the DataFrame-embedded truth hash matches `COMPARISON_TRUTH_HASH`
- `comparison_df["meta__parent_truth_hash"]` — verifies exactly one unique non-null parent hash, matching `GOLD_PARENT_TRUTH_HASH`
- Re-loads the saved truth file and checks both `truth_hash` and `parent_truth_hash` fields
- Re-loads the saved `MODEL_COMPARISON_SUMMARY_PATH` and checks `baseline_truth_hash`, `cascade_default_truth_hash`, `cascade_tuned_truth_hash`, and `cascade_stage3_improved_truth_hash` fields match the known stage hashes

Any mismatch raises `ValueError` with the conflicting values.

---

## 19. SQL Persistence

```python
WRITE_TO_POSTGRES = True

gold_comparison_sql_summary_dataframe = write_gold_model_comparison_results_sql(
    engine=engine,
    capstone_schema=CAPSTONE_SCHEMA,
    dataset_id=DATASET_ID,
    run_id=RUN_ID,
    notebook_globals=globals(),
    dataset_name=DATASET_NAME,
    dataframe=gold04_metric_rows_df,
)
```

Before calling the writer, `comparison_df` is copied and the copy is normalized to ensure the SQL writer receives:

- A `model` column (copied from `Model` if needed)
- A `model_label` column derived from `model`
- A normalized `alert_count` column (probed from a candidate list: `alert_count_test_rows`, `test_alert_count`, `test_alerts`, and others)
- Lowercase metric column names (`precision`, `recall`, `f1`) from any title-case variants
- `dataset_id` and `run_id` columns

Required columns `["model", "alert_count", "precision", "recall", "f1"]` are validated before the write. A `KeyError` is raised if any are missing.

The write target is `gold.model_comparison_results`. A ledger step `"gold04_sql_metric_dataframe_prepared"` records the normalized frame shape.

A post-write verification query reads the five most recent rows from `gold.model_comparison_results` filtered by `dataset_id` and `run_id` and displays the result. The query returns `baseline_model`, `comparison_model`, alert counts, precision, recall, F1, and `created_at_utc` for each written row.

---

## Inputs

| Source | Type | Key |
|---|---|---|
| Gold_02 baseline results | Pickle DataFrame | `BASELINE_RESULTS_PATH_PICKLE` |
| Gold_02 baseline summary | JSON | `BASELINE_SUMMARY_PATH` |
| Gold_02 baseline thresholds | JSON | `BASELINE_THRESHOLDS_PATH` |
| Gold_02 baseline metadata | JSON | `BASELINE_METADATA_PATH` |
| Gold_02 baseline truth | JSON (resolved from metadata) | `BASELINE_TRUTH_PATH` |
| Gold_03a cascade defaults results | Pickle DataFrame | `CASCADE_DEFAULTS_RESULTS_PATH_PICKLE` |
| Gold_03a cascade defaults summary | JSON | `CASCADE_DEFAULTS_SUMMARY_PATH` |
| Gold_03a cascade defaults thresholds | JSON | `CASCADE_DEFAULTS_THRESHOLDS_PATH` |
| Gold_03a cascade defaults metadata | JSON | `CASCADE_DEFAULTS_METADATA_PATH` |
| Gold_03a cascade defaults truth | JSON (resolved from metadata) | `CASCADE_DEFAULTS_TRUTH_PATH` |
| Gold_03b cascade tuned results | Pickle DataFrame | `CASCADE_TUNED_RESULTS_PATH_PICKLE` |
| Gold_03b cascade tuned summary | JSON | `CASCADE_TUNED_SUMMARY_PATH` |
| Gold_03b cascade tuned thresholds | JSON | `CASCADE_TUNED_THRESHOLDS_PATH` |
| Gold_03b cascade tuned metadata | JSON | `CASCADE_TUNED_METADATA_PATH` |
| Gold_03b cascade tuned truth | JSON (resolved from metadata) | `CASCADE_TUNED_TRUTH_PATH` |
| Gold_03c stage3_improved results | Pickle DataFrame | `CASCADE_STAGE3_IMPROVED_RESULTS_PATH_PICKLE` |
| Gold_03c stage3_improved summary | JSON | `CASCADE_STAGE3_IMPROVED_SUMMARY_PATH` |
| Gold_03c stage3_improved thresholds | JSON | `CASCADE_STAGE3_IMPROVED_THRESHOLDS_PATH` |
| Gold_03c stage3_improved metadata | JSON | `CASCADE_STAGE3_IMPROVED_METADATA_PATH` |
| Gold_03c stage3_improved truth | JSON (resolved from metadata) | `CASCADE_STAGE3_IMPROVED_TRUTH_PATH` |
| Shared Gold parent truth | JSON (resolved from parent hash) | `GOLD_TRUTH_PATH` |
| Truth index | JSONL (appended) | `TRUTH_INDEX_PATH` |

---

## Outputs and Artifacts

| Artifact | Path anchor | Format |
|---|---|---|
| Comparison table (stamped) | `GOLD_COMPARISON_ARTIFACT_DIRS["results"]` | CSV |
| Comparison summary | `GOLD_COMPARISON_ARTIFACT_DIRS["summaries"]` | JSON |
| Statistical test report | `GOLD_COMPARISON_ARTIFACT_DIRS["statistics"]` | CSV |
| Statistical test summary | `GOLD_COMPARISON_ARTIFACT_DIRS["statistics"]` | JSON |
| Cascade funnel charts (per model) | `GOLD_COMPARISON_ARTIFACT_DIRS["plots"]` | PNG (×3) |
| Stage 3 operating modes chart | `GOLD_COMPARISON_ARTIFACT_DIRS["plots"]` | PNG |
| Alert count comparison chart | `GOLD_COMPARISON_ARTIFACT_DIRS["plots"]` | PNG |
| Config snapshot | `GOLD_COMPARISON_ARTIFACT_DIRS["config"]` | YAML |
| Comparison truth record | `TRUTHS_PATH / "gold_comparison" / ...` | JSON |
| Comparison ledger | `GOLD_COMPARISON_ARTIFACT_DIRS["lineage"]` | JSONL |
| SQL rows | `gold.model_comparison_results` | PostgreSQL |

All output files registered to W&B via `wandb_run.save` are: `MODEL_COMPARISON_PATH`, `MODEL_COMPARISON_SUMMARY_PATH`, `comparison_truth_path`, and `comparison_ledger_path`.

---

## 22. Truth Chain and Lineage Continuity

Gold_04 preserves and propagates the following lineage fields:

| Field | Source | Value |
|---|---|---|
| `GOLD_PARENT_TRUTH_HASH` | Shared across all 4 upstream truth records | Anchored on `CASCADE_DEFAULTS_TRUTH_HASH` parent |
| `BASELINE_TRUTH_HASH` | Gold_02 truth file | Validated 3-way: metadata → truth file → results CSV |
| `CASCADE_DEFAULTS_TRUTH_HASH` | Gold_03a truth file | Validated 3-way |
| `CASCADE_TUNED_TRUTH_HASH` | Gold_03b truth file | Validated 3-way |
| `CASCADE_STAGE3_IMPROVED_TRUTH_HASH` | Gold_03c truth file | Validated 3-way |
| `COMPARISON_TRUTH_HASH` | Gold_04 truth record | Stamped into `comparison_df` via `stamp_truth_columns` |

`parent_truth_hash = GOLD_PARENT_TRUTH_HASH` in the Gold_04 truth record establishes the comparison stage as a direct child of the same Gold feature lineage that produced all four upstream models.

All four stage truth hashes are carried in `comparison_summary`, the `comparison_truth` runtime facts section, and the written SQL rows (where the SQL writer derives them from `notebook_globals`).

---

## Data Quality / Validation Behavior

| Check | Purpose | Failure / Risk Prevented |
|---|---|---|
| Three-source truth hash validation per model | Confirm each model's truth hash is consistent across metadata JSON, truth file, and results pickle | `ValueError` on any mismatch; prevents comparison of outputs from different model runs |
| All 4 models share `GOLD_PARENT_TRUTH_HASH` | Confirm all models evaluated against the same Gold_01 preprocessing output | `ValueError` if any model has a different parent hash; prevents invalid cross-model comparisons |
| `DATASET_NAME` consistency across all 4 truth records | Same dataset name required across all models | `ValueError` on mismatch |
| `comparison_df` parent hash uniqueness | Single unique non-null parent hash in stamped DataFrame | `ValueError` if multiple values present |
| Truth file roundtrip after `save_truth_record` | Re-read saved truth file; check `truth_hash` and `parent_truth_hash` fields | `ValueError` on hash mismatch; catches file corruption after save |
| Saved summary hash fields | Re-read `MODEL_COMPARISON_SUMMARY_PATH` and verify all 4 stage truth hashes match known values | `ValueError` on mismatch; prevents silent summary corruption |

---

## Downstream Handoff

Gold_04 does not produce a scored pump telemetry DataFrame. It does not write row-level anomaly flags, model artifacts (joblib files), or per-sensor prediction scores. Its outputs are comparison-layer artifacts: the comparison CSV, summary JSON, statistical test files, plots, and truth record.

Gold_05 (Anomaly Detection) follows Gold_04 in the pipeline sequence but operates independently — it does not directly consume Gold_04 comparison outputs as pipeline inputs. The `write_gold_model_comparison_results_sql` write target (`gold.model_comparison_results`) serves reporting and audit purposes.

The `comparison_summary` fields `best_model_by_precision`, `best_model_by_recall`, `best_model_by_f1`, and `best_model_by_alert_reduction` are informational identifiers derived from the comparison table. No model selection or routing flag is set in Gold_04 that directs downstream pipeline execution.

---

## Key Function Calls and In-Place Usage

| Function | Module | Purpose |
|---|---|---|
| `load_notebook_context` | `utils.core.notebook_context` | Bootstrap CTX, paths, config, logger |
| `build_artifact_dirs` | `utils.core.artifacts` | Create artifact directory dicts for 5 model families |
| `export_config_snapshot` | `utils.core.config_loader` | Save resolved config YAML |
| `get_engine_from_env` | `utils.database.postgres` | PostgreSQL engine |
| `read_sql_dataframe` | `utils.database.postgres` | SQL smoke check and post-write verification |
| `log_layer_paths` | `utils.core.logging_setup` | Log resolved paths |
| `load_json` | `utils.core.file_io` | Load JSON artifacts |
| `save_json` | `utils.core.file_io` | Save JSON artifacts |
| `extract_truth_hash` | `utils.core.truths` | Extract `meta__truth_hash` from DataFrame |
| `get_parent_truth_hash` | `utils.core.truths` | Extract `parent_truth_hash` from truth record |
| `get_dataset_name_from_truth` | `utils.core.truths` | Extract dataset name for cross-check |
| `get_pipeline_mode_from_truth` | `utils.core.truths` | Extract pipeline mode from baseline truth |
| `initialize_layer_truth` | `utils.core.truths` | Create blank truth record |
| `update_truth_section` | `utils.core.truths` | Populate config_snapshot, runtime_facts, artifact_paths |
| `build_truth_record` | `utils.core.truths` | Finalize truth record, compute hash |
| `stamp_truth_columns` | `utils.core.truths` | Add `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` to comparison_df |
| `save_truth_record` | `utils.core.truths` | Write truth record to disk |
| `append_truth_index` | `utils.core.truths` | Register truth record in index |
| `make_process_run_id` | `utils.core.truths` | Generate process run ID |
| `build_truth_config_block` | `utils.core.config_loader` | Build config snapshot block |
| `write_gold_model_comparison_results_sql` | `utils.database.medallion_sql_writers` | Write comparison rows to `gold.model_comparison_results` |
| `Ledger` | `utils.core.ledger` | Structured run log with named steps |

---

## Logical Workflow Map

1. Imports and inline helper definitions
2. Context bootstrap (`load_notebook_context`)
3. Config variable resolution from `CTX`
4. Sanity checks (general, then Gold-specific)
5. Artifact directory creation (`build_artifact_dirs` ×5)
6. Input and output path resolution
7. Config snapshot save
8. Database connection + asset resolution
9. SQL smoke check
10. W&B initialization
11. Ledger initialization
12. Upstream data loading (pickles + JSON summaries, thresholds, metadata)
13. Three-source truth hash validation for all four models
14. Lineage cross-check — all models must share `GOLD_PARENT_TRUTH_HASH`
15. Gold parent truth file load
16. Ledger step: `load_comparison_inputs`
17. Comparison table construction (`comparison_df`, 7 rows)
18. Comparison summary construction
19. Styled display of comparison table
20. Statistical test helper definitions
21. `build_statistical_test_summary` — McNemar + z-tests (baseline vs. Stage 3 Improved)
22. Statistical output saves (CSV + JSON)
23. Stage 3 operating mode summary and scatter plot
24. Stage 3 summary reload and extended metric helper definitions
25. `build_cascade_funnel_dataframe` helper definition
26. Cascade funnel DataFrame construction
27. Per-model cascade funnel bar chart generation and save
28. Truth record construction and stamping
29. Truth record save and index append
30. Comparison CSV and summary JSON saves
31. W&B uploads (comparison CSV, summary JSON, truth file)
32. Ledger step: `save_comparison_outputs`
33. Ledger step: `finalize_comparison`
34. Ledger JSONL write + W&B save of ledger
35. `wandb_run.finish()`
36. Alert count comparison bar chart (post-close, re-reads saved summary)
37. Lineage invariant checks (post-close)
38. SQL normalization + `write_gold_model_comparison_results_sql`
39. Post-write verification query from `gold.model_comparison_results`

---

## Relationship to Other Notebooks

### Upstream Context

Gold_04 loads from Gold_02 (baseline), Gold_03a (cascade defaults), Gold_03b (cascade tuned), and Gold_03c (cascade stage3_improved): results pickles, metadata JSONs, and truth records. It performs three-source truth hash validation per model and cross-validates that all four models share `GOLD_PARENT_TRUTH_HASH`. `DATASET_NAME` is confirmed consistent across all four truth records.

### Downstream Handoff

Gold_04 produces:
- Comparison CSV and summary JSON for the reporting layer
- Statistical test files and cascade funnel charts for audit
- `gold.model_comparison_results` SQL rows for the operational layer
- `COMPARISON_TRUTH_HASH` registered under `truths/gold_comparison/`

Gold_05 follows in execution sequence but does not consume Gold_04 outputs as pipeline inputs. Gold_04 does not produce scored telemetry rows or route model decisions.

### Pipeline Position

Convergence point for all Gold modeling. The first notebook in the pipeline to load results from four independent modeling notebooks simultaneously. Bridges the modeling phase and the reporting/validation phase. Gold_05, Gold_06A, and Gold_06B all follow Gold_04 in execution sequence but operate independently of its outputs.

### Relationship Summary

- Reads results and truth records from Gold_02, Gold_03a, Gold_03b, and Gold_03c
- Cross-validates `GOLD_PARENT_TRUTH_HASH` across all four upstream models — the lineage convergence check
- No confirmed downstream pipeline notebook consumer of Gold_04's file outputs (Gold_05 operates independently)
- Writes to `gold.model_comparison_results`; does not write to `gold.anomaly_detection_scores`
- Closes the training-run model evaluation phase
