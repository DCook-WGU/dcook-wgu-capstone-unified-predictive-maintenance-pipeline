# Gold 04 Deep Technical Reference

## Purpose of This Deep Reference

This document covers the important technical decisions in the Gold 04 Comparison notebook that require deeper explanation than the workflow reference provides. The workflow reference describes what the notebook does. This reference explains why comparison inputs are loaded and validated the way they are, why the comparison table is structured as seven rows from four model notebooks, why lineage validation is a blocking precondition rather than an optional audit step, why the statistical tests are designed and scoped the way they are, and why the output persistence and final consistency checks are implemented as they are.

---

## Technical Scope

Technical areas covered in this document:

- Comparison input contract: artifact families, truth record validation, three-source hash check
- Shared parent truth hash: anchoring, cross-model validation, failure modes
- Comparison table construction: seven-row design, metric-read-not-recompute principle
- Comparison summary metrics: alert reduction ratios, best-model rankings
- Statistical significance testing: McNemar, z-tests, paired alignment design
- Stage 3 operating-mode analysis: scatter-plot design, separation from family comparison
- Cascade funnel data: built from summary dicts, not from raw results
- Missing metric handling: NaN semantics and plot-frame filtering
- Artifact path fallback: backward compatibility with pre-resolved-paths artifact layouts
- Comparison truth record and lineage stamping
- Artifact persistence: CSV, JSON summary, truth record, W&B uploads
- Ledger and W&B lifecycle
- Final lineage invariant checks: eight post-save consistency assertions
- SQL persistence: `gold.model_comparison_results`, column normalization, WRITE_TO_POSTGRES=True

---

## Source Grounding

Sources used:

- `notebooks/experiments/EDA_Notebook_Pump_Gold_04_Comparison.ipynb` (85 cells, 33 code cells) — primary source of truth
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_04_Comparison_code_reference.md` — read-only context
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling_deep_technical_reference.md` — read-only upstream context
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling_deep_technical_reference.md` — read-only upstream context
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling_deep_technical_reference.md` — read-only upstream context
- `notebook_inventory.json`
- `artifact_io_manifest.json`
- `sql_touchpoints.json`

The active notebook source is the source of truth for all technical claims. Upstream deep references and the workflow reference provide consumer and pipeline context only.

---

## Stage Role in the Gold Modeling Sequence

Gold_04 is the comparison and evaluation notebook in the Gold sequence. It does not train any model, score any rows, or apply anomaly detection logic. Its role is to:

- Collect the saved output artifacts from four upstream Gold modeling notebooks (Gold_02, Gold_03a, Gold_03b, Gold_03c)
- Validate that all four sets of artifacts share a common Gold-layer lineage anchor before any comparison values are computed
- Build a unified seven-row comparison table from pre-saved summary metrics
- Run statistical significance tests between the baseline and the final cascade variant
- Produce comparison charts and summary reports
- Stamp and persist a `gold_comparison` truth record
- Write comparison rows to `gold.model_comparison_results` in PostgreSQL

Gold_04 does not select a winning model. The comparison summary records which model ranks highest by precision, recall, F1, and alert reduction, but these are informational fields derived from the comparison table, not pipeline-routing flags that control which model enters Gold_05.

The four upstream model notebooks and their identifiers in the comparison table are:

| Notebook | Model IDs in Gold_04 comparison table | Variant family |
|---|---|---|
| Gold_02 | `baseline` | `baseline` |
| Gold_03a | `cascade_default` | `cascade` |
| Gold_03b | `cascade_tuned` | `cascade` |
| Gold_03c | `stage3_improved`, `stage3_relaxed`, `stage3_medium`, `stage3_strict` | `cascade_stage3` |

All seven rows carry the same `parent_gold_truth_hash`, establishing that every compared model was evaluated against the same Gold-layer feature data.

---

## Input Contract and Lineage

### Upstream Artifact Families

Gold_04 loads three artifact types per upstream model variant:

1. **Results pickle** (`.pkl`): row-level DataFrame with model scores, flags, and meta columns
2. **Summary JSON**: pre-computed metrics and alert counts produced by the modeling notebook
3. **Metadata JSON**: truth hash pointer and truth file path for the modeling run

For the baseline (Gold_02) these are loaded from `BASELINE_*_PATH` variables. For the three cascade variants (Gold_03a, Gold_03b, Gold_03c) the equivalents are `CASCADE_DEFAULTS_*`, `CASCADE_TUNED_*`, and `CASCADE_STAGE3_IMPROVED_*` variables. All path variables are resolved from `RESOLVED_PATHS`, which is populated via the shared notebook context (`load_notebook_context`).

The `require_truth_record` inline helper validates that every loaded JSON artifact is a dictionary rather than a list or None before any fields are accessed from it.

### Artifact Directory Construction

Gold_04 builds separate `build_artifact_dirs` directory structures for each model family because its own config does not contain the stage-specific config sections of the modeling notebooks. The four upstream model families (`baseline`, `cascade_defaults`, `cascade_tuned`, `cascade_stage3_improved`) each receive a directory structure under `GOLD_COMMON_SUBDIRS = ["scores", "summaries", "thresholds", "metadata", "models", "plots", "config", "lineage"]`. The comparison output family (`comparison`) receives `GOLD_COMPARISON_SUBDIRS = ["results", "summaries", "plots", "statistics", "metadata", "config", "lineage"]`, which differs by having `results` and `statistics` in place of `scores`, `models`, and `thresholds`.

### Three-Source Truth Hash Validation

For each of the four artifact families, Gold_04 validates the truth hash against three independent sources before proceeding:

1. **Metadata JSON** — contains `baseline_truth_hash` or `cascade_truth_hash`
2. **Truth file** — loaded from the path stored in the metadata; must have `truth_hash` matching the metadata value
3. **Results CSV** — `extract_truth_hash(results)` reads `meta__truth_hash` from the results DataFrame; must match the metadata value

The comment in the notebook source states the reason explicitly: a mismatch means the comparison would silently mix artifacts from different runs. This three-source check catches cases where only the metadata or only the CSV was refreshed without re-running the full modeling notebook. Any mismatch raises `ValueError` immediately.

### Required Metadata Fields

For the baseline, metadata must contain `baseline_truth_hash` and `baseline_truth_path`. For cascade variants, metadata must contain `cascade_truth_hash` and `cascade_truth_path`. Missing any of these raises `ValueError` before loading the truth file.

### Shared Parent Truth Hash — Anchoring and Enforcement

After validating individual artifact truth hashes, Gold_04 confirms that all four model variants share the same `parent_gold_truth_hash`. This hash represents the Gold_01 preprocessing truth record that all modeling notebooks must have consumed.

The canonical lineage anchor is `DEFAULTS_PARENT_GOLD_TRUTH_HASH` (from Gold_03a, the cascade defaults). All other parent hashes — tuned, stage3_improved, baseline — must equal this value. If any differ, a `ValueError` is raised with both the differing hash and the expected anchor, plus remediation guidance: re-run the mismatched notebook from the same Gold parent lineage. This check enforces that the comparison is scientifically valid — all models must have been built on the same feature data.

Once the shared parent hash is confirmed, Gold_04 loads the Gold parent truth file directly (`gold_truth`), which must exist at `{TRUTHS_PATH}/gold/{DATASET_NAME}__gold__truth__{GOLD_PARENT_TRUTH_HASH}.json`. The Gold truth's `runtime_facts` and `artifact_paths` sections carry context into the comparison truth record.

### Dataset Name Cross-Validation

After establishing the shared parent hash, Gold_04 reads `DATASET_NAME` from each upstream truth record using `get_dataset_name_from_truth`. If any truth record disagrees with the baseline's dataset name, a `ValueError` is raised. This guards against a comparison that silently mixes artifacts from different datasets.

### Pipeline Mode

`PIPELINE_MODE` is initialized from `PIPELINE["execution_mode"]` and then overridden from `baseline_truth` if `get_pipeline_mode_from_truth(baseline_truth)` returns a non-None value. This ensures the comparison notebook inherits the same pipeline mode as the upstream run rather than relying solely on the current session config.

---

## Comparison Input Preparation

### Metric Read vs. Recompute Principle

The most important design decision in Gold_04 is that **all comparison metrics are read from the saved summary dicts rather than recomputed from the raw results**. The notebook source states this explicitly:

> Metrics will be extracted from saved summary/truth records rather than recomputed
> so the comparison reflects the exact same evaluation that produced the training artifacts.

This means the `comparison_df` values for precision, recall, and F1 come directly from `baseline_summary["baseline_metrics"]`, `cascade_defaults_summary["cascade_metrics"]`, and so on. Alert counts come from `baseline_summary["alert_count_test_rows"]` and `cascade_*_summary["final_cascade_alert_count_test_rows"]`.

Re-evaluating metrics from raw scores in a comparison notebook would introduce risk: threshold choices, test-row masks, and evaluation logic would have to be replicated exactly. Reading the saved values eliminates this risk by using the evaluation that already ran inside the modeling notebooks.

### Summary Key Differences Across Model Families

The baseline summary uses `baseline_summary["alert_count_test_rows"]` while cascade summaries use `cascade_*_summary["final_cascade_alert_count_test_rows"]`. Gold_04 accesses each key by name rather than applying a generic normalization helper, because the metric-read-not-recompute principle requires using the exact keys the modeling notebooks wrote.

For Stage 3 operating-mode metrics, the keys follow the pattern `stage3_{mode}_alert_count_test_rows`, `stage3_{mode}_precision`, `stage3_{mode}_recall`, and `stage3_{mode}_f1` within `cascade_stage3_improved_summary`.

### Artifact Path Fallback for CASCADE_STAGE3_IMPROVED

The `STAGE3_IMPROVED_SUMMARY_PATH` in the visualization section (which reloads the summary for charting) uses a `RESOLVED_PATHS.get(...)` call with a computed fallback path. The notebook comment explains this supports artifact layouts that predate the resolved_paths naming convention. This backward compatibility path ensures the visualization section runs against older artifact trees.

### Missing Metric NaN Handling

The comparison plot dataframe is built from `candidate_comparison_rows` using `_safe_float` and `_safe_int` helpers. These helpers return `float("nan")` for `None` inputs or any value that cannot be cast to float or int. An `_is_complete_metric_row` helper then filters out any row where `test_alerts`, `precision`, `recall`, or `f1` is `None` or NaN. The filtered-out rows are collected into `excluded_comparison_rows` and displayed for inspection. This design allows the comparison to proceed when only a subset of Stage 3 operating modes produced complete metrics, without crashing or silently plotting empty bars.

---

## Model Comparison Methodology

### Seven-Row Comparison Design

The `comparison_df` produced by Gold_04 has seven rows, one per model or Stage 3 operating mode:

| Row | `model_id` | `variant_family` | `stage3_mode` |
|---|---|---|---|
| 1 | `baseline` | `baseline` | `none` |
| 2 | `cascade_default` | `cascade` | `none` |
| 3 | `cascade_tuned` | `cascade` | `none` |
| 4 | `stage3_improved` | `cascade_stage3` | `selected_improved` |
| 5 | `stage3_relaxed` | `cascade_stage3` | `relaxed` |
| 6 | `stage3_medium` | `cascade_stage3` | `medium` |
| 7 | `stage3_strict` | `cascade_stage3` | `strict` |

Every row carries `stage_truth_hash` (the hash of the modeling notebook's truth record) and `parent_gold_truth_hash` (the shared Gold parent). This links every comparison row to its upstream artifact lineage. The `variant_family` and `stage3_mode` columns allow downstream consumers (Gold_05, Gold_06A, reports) to filter the comparison table without parsing `model_id` strings.

### Comparison Metrics

The metrics columns in `comparison_df` are: `alert_count_test_rows`, `precision`, `recall`, `f1`. These are the four metrics normalized and persisted by the individual modeling notebooks. Gold_04 does not add ROC AUC, PR AUC, or score-distribution comparisons, because the modeling notebooks produced alerts against synthetic truth labels using classification metrics, and the comparison is restricted to those evaluation-compatible metrics.

### Alert Reduction Metrics

`comparison_summary` includes per-variant alert reduction count and ratio relative to the baseline:

- `baseline_vs_{variant}_alert_reduction_count`: `baseline_count - variant_count`
- `baseline_vs_{variant}_alert_reduction_ratio`: `(baseline_count - variant_count) / max(baseline_count, 1)`

These are computed for cascade_default, cascade_tuned, stage3_improved, stage3_relaxed, stage3_medium, and stage3_strict. The division guard `max(baseline_count, 1)` prevents zero-division when the baseline has no alerts on the test set.

### Best-Model Rankings

`comparison_summary` includes four best-model identifiers:

- `best_model_by_precision`: `comparison_df.sort_values("precision", ascending=False).iloc[0]["model_id"]`
- `best_model_by_recall`: sorted descending
- `best_model_by_f1`: sorted descending
- `best_model_by_alert_reduction`: sorted by `alert_count_test_rows` ascending (lowest alert count = most reduction from baseline)

These rankings are stored in `comparison_summary`, passed into the comparison truth record's `runtime_facts`, and written to the saved JSON summary. They are informational — Gold_04 does not use them to route a model into Gold_05. The notebook summary note confirms this design: "Gold_04 does not declare a winning model."

### Stage 3 Operating-Mode Analysis

The Stage 3 operating modes are compared separately from the full comparison family. `stage3_mode_dataframe` filters `comparison_df` to the four `stage3_*` rows and plots a scatter chart of `alert_count_test_rows` vs `f1`. This separates the operating-mode tradeoff view (alert burden vs. detection quality within Stage 3) from the baseline-vs-cascade family comparison. The chart is saved to `{DATASET_NAME}__gold__stage3_operating_modes_alerts_vs_f1.png`.

---

## Validation Contract Behavior

Gold_04 does not read or verify Gold model-output validation contracts. The validation contracts produced by Gold_03a, Gold_03b, and Gold_03c are intended for Gold_06A (test replay validation). Gold_04 validates truth hash lineage directly through the metadata JSON, truth file, and results CSV three-source check, which covers the integrity Gold_04 needs for the comparison itself.

Contract-based validation of the comparison rows against the upstream model-output contracts is not performed in Gold_04 source. Whether Gold_06A or another notebook performs this check is not determined from Gold_04 source.

---

## Comparison Output Construction

### comparison_df Structure

The `comparison_df` is a seven-row DataFrame with columns:

- `model_id`: canonical string identifier (e.g., `"baseline"`, `"cascade_default"`, `"stage3_improved"`)
- `model`: human-readable label (e.g., `"Baseline IsolationForest"`, `"Stage 3 Relaxed"`)
- `variant_family`: `"baseline"`, `"cascade"`, or `"cascade_stage3"`
- `stage3_mode`: `"none"` or the operating mode string
- `alert_count_test_rows`: integer alert count on test rows
- `precision`, `recall`, `f1`: metrics read from upstream summaries
- `stage_truth_hash`: individual model truth hash
- `parent_gold_truth_hash`: shared Gold parent truth hash (same for all rows)
- `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode`: stamped by `stamp_truth_columns` before CSV save

The lineage columns are stamped via `stamp_truth_columns(comparison_df, truth_hash=COMPARISON_TRUTH_HASH, parent_truth_hash=GOLD_PARENT_TRUTH_HASH, pipeline_mode=PIPELINE_MODE)`. This makes `comparison_df` a first-class pipeline artifact, not just a notebook table.

### Cascade Funnel Dataframe

`build_cascade_funnel_dataframe` (inline function) constructs a per-stage alert-count frame from saved cascade summary values. It reads stage-level counts from `cascade_summary` dict rather than from raw result DataFrames. The funnel frame is concatenated across `03A Default Cascade` and `03B Tuned Cascade` variants and displayed per model as a bar chart showing stage-by-stage alert reduction.

This design (reading from summary rather than raw results) is consistent with the metric-read-not-recompute principle: the stage counts were computed inside the cascade notebooks; Gold_04 visualizes what was reported without re-filtering raw rows.

### Saved Outputs

The following artifacts are written to disk:

| Artifact | Path Variable | Format |
|---|---|---|
| Comparison CSV | `MODEL_COMPARISON_PATH` | CSV with meta columns stamped |
| Comparison summary | `MODEL_COMPARISON_SUMMARY_PATH` | JSON (comparison_summary dict) |
| Comparison truth record | `comparison_truth_path` | JSON (truth record) |
| Comparison ledger | `comparison_ledger_path` | JSONL |
| Statistical test table | `STATISTICAL_TEST_TABLE_PATH` | CSV |
| Statistical test summary | `STATISTICAL_TEST_SUMMARY_PATH` | JSON |
| Alert count bar chart | `comparison_alert_plot_path` | PNG |
| 2-panel metrics chart | `comparison_2panel_plot_path` | PNG |
| Stage 3 mode scatter | `stage3_mode_plot_path` | PNG |
| Cascade funnel PNGs | Per-model path under `GOLD_COMPARISON_ARTIFACT_DIRS["plots"]` | PNG |

`comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)` and `save_json(comparison_summary, MODEL_COMPARISON_SUMMARY_PATH)` are both called before the truth record is finalized, so the truth record can reference the artifact paths in its `artifact_paths` section.

---

## Evaluation and Metrics

### Comparison Metric Fidelity

The comparison metrics in `comparison_df` and `comparison_summary` are not independently evaluated by Gold_04. They are read from the upstream summary dicts produced by each modeling notebook's own evaluation phase. This preserves metric fidelity: the precision, recall, and F1 values in the comparison table are identical to what each modeling notebook reported at run time. Any divergence between what a modeling notebook reports and what Gold_04 shows would indicate that a different artifact version was loaded, which the three-source truth hash check would catch before the comparison table is built.

### Statistical Comparison — McNemar's Test

Gold_04 runs a paired statistical comparison between `baseline_results` and `cascade_stage3_improved_results`. The comparison is restricted to these two models; the intermediate cascade variants (default, tuned) are not included in the statistical test.

The `build_paired_model_frame` function merges the two result DataFrames on `meta__row_id` using an inner join with `validate="one_to_one"`. If the row counts after the join differ from the baseline frame size, a `ValueError` is raised. After the join, only test rows are retained using `meta__is_train_flag == False`.

The `run_mcnemar_paired_test` function builds the McNemar contingency table as:

```
[[both correct, baseline correct / comparison wrong],
 [baseline wrong / comparison correct, both wrong]]
```

When `discordant_count < 25`, the exact McNemar test is used (`exact=True` via statsmodels). When discordant pairs are ≥ 25, the chi-square approximation with Yates continuity correction is applied. If `statsmodels` is unavailable, a manual continuity-corrected McNemar chi-square fallback using `scipy.stats.norm` is applied.

McNemar is appropriate here because the comparison is paired — the same test rows are evaluated by both models — and the outcome variable is binary (correct vs. incorrect classification).

### Secondary Statistical Tests — Two-Proportion Z-Tests

Two supplementary z-tests are computed:

1. **False-positive-rate z-test** (`alternative="greater"`): Tests whether the baseline's false-positive rate on normal rows is higher than the cascade's. Uses `count_a=baseline_fp, nobs_a=baseline_normal_rows, count_b=comparison_fp, nobs_b=comparison_normal_rows`.

2. **Precision z-test** (`alternative="less"`): Tests whether the cascade achieved higher precision than the baseline. Uses TP / (TP + FP) for each model.

These are aggregate-level tests that do not require paired row alignment, making them complementary to McNemar rather than primary. The `run_two_proportion_z_test` helper computes a pooled proportion standard error and returns the z-statistic and p-value.

### Confusion Matrix Counts

`summarize_confusion_counts` computes TP, FP, TN, FN, precision, recall, F1, and false_positive_rate from the `paired_frame` for each of `baseline_pred` and `comparison_pred`. These counts appear in `statistical_test_summary["baseline_counts"]` and `statistical_test_summary["comparison_counts"]`.

---

## Artifact and SQL Persistence

### SQL Write Target

Gold_04 writes comparison rows to the `gold.model_comparison_results` table using `write_gold_model_comparison_results_sql`. `WRITE_TO_POSTGRES = True` is hardcoded in the notebook — there is no dev/prod toggle for the SQL write.

Before calling the SQL writer, the notebook normalizes the comparison DataFrame for the SQL interface:

- Adds `model_label` from `model` if absent
- Probes multiple column name candidates for alert count: `alert_count_test_rows`, `test_alert_count`, `test_alerts`, `Test Alerts`, `alert_count_all_rows`, `alert_count`, `alerts`, `final_alert_count`. The first match becomes `alert_count` and `alert_count_all_rows`.
- Maps alternative metric column names (`Precision`, `Recall`, `F1`, `F1 Score`, `f1_score`) to their lowercase equivalents
- Adds `dataset_id` and `run_id` from the comparison runtime context

If the alert-count probe finds no recognized column, a `KeyError` is raised. If required normalized columns (`model`, `alert_count`, `precision`, `recall`, `f1`) are absent after normalization, a `KeyError` is raised with the list of missing columns.

The multi-candidate column probe is needed because `comparison_df` column naming evolved across notebook versions; the SQL writer requires a stable `alert_count` column regardless of how the comparison table was built.

A post-write verification query reads from `gold.model_comparison_results` filtered to `dataset_id` and `run_id` and displays up to 5 rows for inspection.

### W&B Artifact Uploads

`wandb_run_object.save(str(...))` is called for `MODEL_COMPARISON_PATH`, `MODEL_COMPARISON_SUMMARY_PATH`, and `comparison_truth_path`. The W&B run is initialized with `job_type="gold_model_comparison"` and the config includes `gold_version` and `gold_comparison_recipe_id`. `wandb_run.finish()` is called after the ledger is written to disk.

### Ledger Lifecycle

A `Ledger` object is created at bootstrap and accumulates step records throughout the notebook run. The final ledger step (`step="finalize_comparison"`) records the comparison CSV path, summary JSON path, and the full `comparison_summary` dict. The ledger is written to `comparison_ledger_path` (JSONL format) after `wandb.save(str(comparison_ledger_path))` and immediately before `wandb_run.finish()`.

---

## Truth, Audit, and Reproducibility Behavior

### Comparison Truth Record

Gold_04 creates a distinct `gold_comparison` layer truth record. Key design points:

- `layer_name = "gold_comparison"` — distinct from the `gold` layer truth records produced by modeling notebooks
- `parent_truth_hash = GOLD_PARENT_TRUTH_HASH` — the shared Gold_01 preprocessing truth hash
- `runtime_facts` section includes: per-model result row counts, shared Gold truth hash, per-model stage truth hashes, and the four best-model ranking fields
- `artifact_paths` section records paths to all four models' artifacts plus the comparison CSV and summary JSON

The truth record is built with `build_truth_record`, saved with `save_truth_record` (to `{TRUTHS_PATH}/gold_comparison/{DATASET_NAME}__gold_comparison__truth__{hash}.json`), and appended to the truth index.

### Comparison Truth Stamping on comparison_df

`stamp_truth_columns` adds `meta__truth_hash`, `meta__parent_truth_hash`, and `meta__pipeline_mode` to `comparison_df` after the truth record hash (`COMPARISON_TRUTH_HASH`) is resolved. The stamped `comparison_df` is then saved to CSV, so the on-disk CSV carries the truth hash as a column in every row.

### Reproducibility

The comparison is fully reproducible given the same upstream artifact set. All comparison metrics come from saved upstream summaries; no probabilistic re-scoring occurs in Gold_04. The `GOLD_PROCESS_RUN_ID` is generated via `make_process_run_id` (timestamped unique identifier) and recorded in the truth record, providing a stable process stamp even if the comparison notebook is re-run.

---

## Downstream Technical Handoff

### To Gold_05

The workflow reference and notebook summary confirm that Gold_05 uses selected model outputs for anomaly timeline and alert review artifacts. What specific artifact Gold_05 reads from Gold_04 is not confirmed from Gold_04 source. Whether Gold_05 reads `MODEL_COMPARISON_PATH` (CSV), `MODEL_COMPARISON_SUMMARY_PATH` (JSON), or the SQL table directly is not determined from available source.

### To Gold_06A

Gold_06A is described as a test replay validation notebook. From Gold_04 source, Gold_06A is not directly referenced. Whether Gold_06A reads the comparison CSV, comparison summary JSON, or the SQL table is not determined from Gold_04 source.

### To Gold_06B

Not determined from available source.

### Via SQL

The `gold.model_comparison_results` table written by Gold_04 is available to downstream SQL consumers. The post-write verification query columns include `dataset_id`, `run_id`, `baseline_model`, `comparison_model`, `alert_count_baseline`, `alert_count_comparison`, `precision_baseline`, `precision_comparison`, `recall_baseline`, `recall_comparison`, `f1_baseline`, `f1_comparison`, `created_at_utc`.

---

## Key Technical Decisions

| Decision | Source Evidence | Why It Matters | Verification Method |
|---|---|---|---|
| Metrics read from saved summaries, not recomputed | Notebook comment: "Metrics will be extracted from saved summary/truth records rather than recomputed so the comparison reflects the exact same evaluation that produced the training artifacts" | Prevents Gold_04 from introducing evaluation divergence; comparison values match exactly what each modeling notebook reported | Confirm `comparison_df` precision values equal `baseline_summary["baseline_metrics"]["precision"]` etc. |
| Triple truth-hash validation per artifact | Three-source check in the artifact loading section: metadata → truth file → results CSV | Detects cross-artifact contamination before any comparison values are computed; catches cases where only one of three sources was refreshed | Manually set a mismatched truth hash in a metadata copy and confirm `ValueError` is raised |
| Shared parent truth hash required for all four model families | `ValueError` if any `parent_gold_truth_hash` differs from `DEFAULTS_PARENT_GOLD_TRUTH_HASH` | Enforces that all models were built on identical feature data; comparison is meaningless if models were built from different preprocessing runs | Confirm `lineage_df` shows identical `parent_gold_truth_hash` for all rows |
| `cascade_defaults` used as the canonical lineage anchor | Comment: "cascade_defaults is used as the canonical lineage reference because all cascade variants must share the same parent Gold truth, and defaults is the simplest variant to anchor on" | Provides a stable, predictable anchor; anchoring on the simplest variant reduces the chance that the anchor itself is a derived or tuned artifact | Confirm `COMPARISON_PARENT_GOLD_TRUTH_HASH` equals `DEFAULTS_PARENT_GOLD_TRUTH_HASH` |
| Seven comparison rows including four Stage 3 modes | `comparison_rows` list in comparison table build cell | Provides complete coverage of the evaluation grid; Gold_03c produces four operating modes and all are included so reviewers can see the alert-burden vs. detection-quality tradeoff within Stage 3 | Confirm `comparison_df` has exactly seven rows with the expected `model_id` values |
| McNemar test for paired significance | `run_mcnemar_paired_test`, inner merge `validate="one_to_one"` on `meta__row_id` | McNemar is the correct paired test for binary classification on the same test rows; avoids treating the two classifiers as independent | Confirm `paired_frame` row count equals the baseline test-row count; confirm McNemar contingency table sums to total test rows |
| Exact McNemar when discordant count < 25 | `use_exact = discordant_count < 25` | Small samples: chi-square approximation is unreliable; exact test is appropriate | Confirm test method string contains "exact" when discordant pairs are few |
| Statistical tests restricted to baseline vs. Stage 3 Improved | `build_statistical_test_summary` called with `cascade_stage3_improved_results` only | Tests the primary experimental question: does the final cascade variant differ significantly from the baseline? Intermediate variants are not the primary comparison target | Confirm `statistical_test_summary["comparison_name"]` is `"Stage 3 Improved"` |
| `_safe_float`/`_safe_int` NaN semantics for optional metrics | Inline helpers; `_is_complete_metric_row` filters plot frame | Allows the notebook to run when Stage 3 operating modes are absent or incomplete without crashing or producing misleading charts | Confirm excluded_comparison_rows is non-empty when a Stage 3 mode lacks metrics; confirm plot frame only includes complete rows |
| `layer_name="gold_comparison"` for truth record | `comparison_truth_layer_name = "gold_comparison"` | Keeps the comparison truth record distinct from the Gold-layer model truth records; prevents truth hash collisions between modeling and comparison stages | Confirm saved truth record path contains `gold_comparison` and not `gold` as the layer component |
| `WRITE_TO_POSTGRES = True` hardcoded | Cell 81 | SQL write is unconditional; no dev/prod skipping flag for the comparison write | Confirm the SQL write always executes when the cell is run; confirm post-write verification query returns rows |
| Alert-count column name probe for SQL normalization | `alert_count_candidates` list with 9 candidates | The comparison_df column naming evolved; the SQL writer requires a stable `alert_count` column; the probe provides backward compatibility | Confirm `alert_count_source_column` resolves to `"alert_count_test_rows"` for current comparison_df |
| Eight post-save lineage invariant checks | Final lineage check section (8 `ValueError`/`FileNotFoundError` guards) | Confirms that on-disk artifacts match in-memory state; prevents notebook from completing when any save was partial or silently failed | Run the final check section and confirm "Gold Comparison lineage sanity check passed" |
| Best-model rankings are informational, not routing flags | `best_model_by_*` fields in comparison_summary; notebook summary: "does not declare a winning model" | Keeps Gold_04 a comparison notebook, not a model selection notebook; routing decisions belong in downstream stages or human review | Confirm `comparison_df` has no `is_recommended` or routing flag column |

---

## Failure Modes and Guardrails

| Condition | Guardrail | Behavior |
|---|---|---|
| Upstream truth file missing | `FileNotFoundError` check before `load_json` | Raises `FileNotFoundError` with the missing truth path |
| Truth hash mismatch: metadata vs. truth file | `baseline_truth.get("truth_hash") != BASELINE_TRUTH_HASH` | Raises `ValueError` with both hashes for both sides |
| Truth hash mismatch: metadata vs. results CSV | `extract_truth_hash(results) != metadata_hash` | Raises `ValueError` with CSV hash and metadata hash |
| Missing `baseline_truth_hash` or `baseline_truth_path` in metadata | None-check after `.get()` | Raises `ValueError` with the missing field name |
| Any model's parent truth hash differs from cascade_defaults | Cross-validation check | Raises `ValueError` with the differing hash and the expected anchor; includes remediation message |
| Dataset name mismatch across model truths | `DATASET_NAME != CASCADE_*_DATASET_NAME` | Raises `ValueError` for each mismatch |
| Comparison parent gold truth hash is None | `COMPARISON_PARENT_GOLD_TRUTH_HASH is None` check | Raises `ValueError`: "Comparison lineage cannot be resolved" |
| `COMPARISON_CFG` missing from context | `gold_required_context_vars` sanity check | Raises `NameError(f"Missing Gold context variables: ...")` |
| Any shared context variable missing | 16-variable sanity check | Raises `NameError(f"Missing required shared context variables: ...")` |
| `paired_frame` row count differs from baseline test-row count after merge | Row count guard in `build_paired_model_frame` | Raises `ValueError` with baseline_rows and paired_rows counts |
| Duplicate `meta__row_id` in either result DataFrame for paired comparison | Uniqueness check in `build_paired_model_frame` | Raises `ValueError` with the non-unique DataFrame name |
| Missing required column in baseline or comparison results | Missing-column check in `build_paired_model_frame` | Raises `ValueError` with the list of missing columns |
| No recognized alert-count column in comparison_df for SQL normalization | `alert_count_source_column is None` check | Raises `KeyError` with available column list |
| SQL required column absent after normalization | `missing_columns` check | Raises `KeyError` with missing column names and available columns |
| `comparison_df` missing `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` after save | Final lineage check | Raises `ValueError` |
| `comparison_df` truth hash does not match `COMPARISON_TRUTH_HASH` | Post-save hash check | Raises `ValueError` with both hashes |
| `comparison_df` has multiple or missing `meta__parent_truth_hash` values | Unique parent hash check | Raises `ValueError` |
| Comparison truth file missing after save | `FileNotFoundError` | Raises `FileNotFoundError` |
| Saved truth file hash or parent hash mismatches | Load and compare saved truth | Raises `ValueError` with both hashes |
| Saved summary hash mismatches for any of 5 upstream truth hashes | Post-save summary verification | Raises `ValueError` with both hashes |
| Binary flag column contains values other than 0 and 1 | `_as_binary_series` validation | Raises `ValueError` with the invalid values found |

---

## Verification Checklist

- [ ] Active notebook path confirmed: `notebooks/experiments/EDA_Notebook_Pump_Gold_04_Comparison.ipynb`
- [ ] All four upstream model result pickles and JSON artifacts exist at their resolved paths
- [ ] Three-source truth hash check passes for all four model families (metadata, truth file, CSV all agree)
- [ ] All four model families share the same `parent_gold_truth_hash`
- [ ] `comparison_df` has exactly seven rows with `model_id` values: `baseline`, `cascade_default`, `cascade_tuned`, `stage3_improved`, `stage3_relaxed`, `stage3_medium`, `stage3_strict`
- [ ] `comparison_df` metric values for `baseline` match `baseline_summary["baseline_metrics"]["precision"]` etc.
- [ ] `comparison_summary` includes `best_model_by_precision`, `best_model_by_recall`, `best_model_by_f1`, `best_model_by_alert_reduction`
- [ ] `statistical_test_summary` includes McNemar result and both z-test results
- [ ] McNemar `test_method` reflects exact or approximation based on discordant count
- [ ] `MODEL_COMPARISON_PATH` (CSV), `MODEL_COMPARISON_SUMMARY_PATH` (JSON), comparison truth record all exist on disk after run
- [ ] `comparison_df` carries `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` columns
- [ ] `comparison_df["meta__parent_truth_hash"]` is a single unique value matching `GOLD_PARENT_TRUTH_HASH`
- [ ] Comparison truth record `layer_name` is `"gold_comparison"`, parent hash is `GOLD_PARENT_TRUTH_HASH`
- [ ] Final lineage checks output "Gold Comparison lineage sanity check passed" without raising
- [ ] `gold.model_comparison_results` post-write query returns rows for `dataset_id` and `run_id`
- [ ] W&B run finishes cleanly; ledger JSONL written before `wandb_run.finish()`
- [ ] `WRITE_TO_POSTGRES = True` is present; SQL write is not gated by a dev flag

---

## Source-Limited Items

| Item | Limitation |
|---|---|
| Exact downstream consumer of `MODEL_COMPARISON_PATH` in Gold_05 | Whether Gold_05 reads the comparison CSV, the summary JSON, or the SQL table is not determined from Gold_04 source |
| Gold_06A's use of Gold_04 outputs | Not determined from Gold_04 source |
| Gold_06B dependencies on Gold_04 | Not determined from available source |
| Whether `validate_gold04_against_output_contracts` from `gold_cascade_validation_contracts.py` is called anywhere in Gold_04 | Not confirmed from Gold_04 source; validation contract checking for Gold_04 comparison rows appears to be a Gold_06A responsibility |
| How `DATASET_ID`, `RUN_ID`, `ASSET_ID` are resolved in the SQL write cell | These globals are used in the SQL cell but their resolution from env vars vs. config is not fully traced in the available cell reads for this reference |
| Whether the statistical test results (McNemar, z-tests) are consumed by any downstream notebook | Not determined from Gold_04 source; they are saved to disk but no downstream reader is confirmed |
| `STATISTICAL_TEST_TABLE_PATH` and `STATISTICAL_TEST_SUMMARY_PATH` variable resolution | These path variables are referenced in the save calls in the statistical test section but their definition cells were not included in the deep read; their exact paths are not confirmed here |
