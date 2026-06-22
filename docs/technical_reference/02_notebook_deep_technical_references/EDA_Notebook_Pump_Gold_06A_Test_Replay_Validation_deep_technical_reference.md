# Gold 06A Deep Technical Reference

## Purpose of This Deep Reference

This document covers technical decisions in Gold 06A that require deeper explanation than the workflow reference provides. The workflow reference describes what the notebook does step by step. This document explains why the replay validation strategy, artifact-first existence check, full-split data preference, two-pass tolerance comparison, Stage 3 rule reimplementation, wide score export design, and lightweight persistence posture are designed the way they are.

## Technical Scope

- Test-mode-only bootstrap and why that matters
- Artifact existence check before any data load
- Full scaled path preference over test-only slice for Stage 3 fidelity
- All helper functions defined inline rather than imported from utilities
- Negated anomaly score convention (`-score_samples`)
- `stage2_flag` cascade gate implementation
- `add_stage3_broad_rules` vs `add_stage3_improved_rules` routing
- Stage 3 operating mode replay from a single DataFrame
- Seven-row metric computation from four replay DataFrames
- Two-pass validation status: exact match → secondary tolerance → `review_delta`
- Comparison against training summary artifacts rather than Gold 04 outputs
- Wide test-row score export with per-model prefixes
- No W&B, no SQL, no truth record design
- Primary handoff artifact for Gold 06B

## Source Grounding

Sources used:

- `notebooks/experiments/EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation.ipynb`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation_code_reference.md`
- `notebook_inventory.json`
- `artifact_io_manifest.json`
- `sql_touchpoints.json`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_04_Comparison_deep_technical_reference.md`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_05_Anomaly_Detection_deep_technical_reference.md`
- `technical_reference/04_deep_utility_function_references/utils__medallion__gold__gold_cascade_validation_contracts_deep_function_reference.md`
- `technical_reference/00_project_manual/notebook_relationship_map.md`
- `technical_reference/00_project_manual/notebook_dependency_matrix.md`

The active Gold 06A notebook source is the source of truth for all function behavior, variable names, and design decisions documented here.

## Stage Role in the Gold Validation Sequence

Gold 06A is a hold-out validation notebook. Its role is narrow and deliberate: reload the fitted model artifacts from Gold_02, Gold_03A, Gold_03B, and Gold_03C, apply them to the held-out test rows from Gold_01 without retraining, compute test-row binary classification metrics from the replayed outputs, compare those replayed metrics against the saved training-notebook summary artifacts, and write the row-level replay scores for Gold_06B consumption.

Gold_06A does not train models, tune thresholds, apply new decision logic, or produce reporting narratives. The interpretation cell in the notebook states its purpose directly: "Gold 06A validates that saved Gold 02/03 model artifacts and rule settings can be replayed against held-out test rows without retraining."

Gold_06A is the only Gold notebook confirmed to run with `CONFIG_RUN_MODE = "test"`. All training notebooks (Gold_02 through Gold_03c) and all analysis/reporting notebooks (Gold_04, Gold_05) run in train mode. Running in test mode is what makes Gold_06A a genuine hold-out validation step rather than a training or analysis step.

The validation produced here is a reproducibility check: does re-applying the saved artifacts to unseen rows produce the same aggregate metrics the training notebooks reported? The answer is captured in `final_validation_status` per model row.

## Input Contract and Lineage

### Replay Source Data

Gold_06A loads either the full scaled Parquet (`FULL_SCALED_PATH`) or a test-only fallback (`TEST_DATA_PATH`), both from Gold_01 preprocessing outputs. The full scaled path is strongly preferred. The notebook comment states the reason: "The full scaled dataframe is preferred over the test-only slice because Stage 3 rolling/persistence/drift rules need training-row history to match the original notebook exactly. Final metrics still use only test rows."

When the full scaled path is used, `test_mask = ~replay_source_dataframe["meta__is_train_flag"].fillna(False).astype(bool)` isolates held-out test rows. All metric computation is performed on test rows only via this mask. The training rows are present in the replay source only to provide rolling-window context for Stage 3 persistence and drift logic; they do not affect the final metric values.

When the test-only fallback is used, `test_mask = pd.Series(True, index=replay_source_dataframe.index)` treats all rows as test rows. Stage 3 replay fidelity may differ because training-row history is absent.

### Model and Configuration Artifacts

Gold_06A loads the following artifact categories from the Gold artifact tree:

- **Feature JSON lists** (from Gold_01): Stage 1 features, Stage 2 features, Stage 3 primary rule sensors, Stage 3 secondary rule sensors — from `GOLD_ROOT / "preprocessing" / "features"`
- **Fitted model joblibs** (from Gold_02 and Gold_03A/03B/03C): 7 fitted models — baseline, cascade_default Stage 1/2, cascade_tuned Stage 1/2, stage3_improved Stage 1/2
- **Threshold JSON files** (from Gold_02 and Gold_03A/03B/03C): 4 files — baseline, cascade_default, cascade_tuned, stage3_improved
- **Summary JSON files** (from Gold_02 and Gold_03A/03B/03C): 4 files — the same summary artifacts saved by the training notebooks; read to build the expected metric table
- **Reference profile CSV files** (from Gold_03A/03B/03C): 3 files — one per cascade variant
- **Resolved config YAML files** (from Gold_03A/03B/03C): 3 files — the `resolved_config.yaml` snapshots; read for the `gold_cascade.stage3` config section

Gold_06A does not load Gold_04 comparison outputs. It does not load Gold_05 anomaly-detection outputs. Its comparison baseline is the saved training-notebook summary artifacts only.

### Formal Validation Contract Files

Gold_06A does not load the formal `gold_model_output_validation_contract` JSON files produced by `build_gold_model_output_validation_contract` / `write_gold_model_output_validation_contract`. The notebook's artifact path construction in Cell 7 defines `MODEL_ARTIFACTS`, `THRESHOLD_ARTIFACTS`, `SUMMARY_ARTIFACTS`, `PROFILE_ARTIFACTS`, and `CONFIG_ARTIFACTS` — no validation contract artifact dict is constructed. The comparison is built directly from saved summary JSON metrics rather than from structured contract payloads.

### Dataset Identity

`DATASET_NAME = CONTEXT_DATASET` — set directly from the context constant. `ARTIFACTS_ROOT = paths.artifacts`, `MODELS_ROOT = paths.models` — both taken from the CTX path object. No `DATASET_ID`/`RUN_ID`/`ASSET_ID` resolution via `first_non_empty_string` is present; SQL writes are absent and no SQL identity lookup is needed.

## Validation Preparation Methodology

### Artifact Existence Check Before Data Load

Before any data or model is loaded, Gold_06A constructs an `artifact_inventory` DataFrame covering all model, threshold, summary, profile, and feature artifacts. Every expected path is checked with `path.exists()`. If any artifact is missing, a `FileNotFoundError` is raised immediately with the full missing-artifact table. A ledger step `"required_artifacts_found"` is added only on complete success.

This pre-load check is deliberate: it prevents partial replay states where some models are scored and others are not due to a missing file discovered mid-run. The `FileNotFoundError` message surfaces all missing artifacts at once rather than one at a time.

### Type-Safe Loading

All JSON and YAML values are validated through inline helpers:

- `require_string_list(value, name)` — validates that a loaded JSON value is a list; converts all elements to strings. The helper comment states the design intent: "This keeps the notebook runtime-safe and also gives Pylance a concrete type instead of `Any | None` from `load_json()`."
- `require_mapping(value, name)` — validates that a loaded JSON or YAML value is a dictionary.

These helpers convert the untyped `Any` return from `load_json` / `yaml.safe_load` into concrete types before any indexing or iteration occurs.

### Label Column Resolution

The label column is resolved by probing candidates `["anomaly_flag", "is_anomaly", "target_flag", "label"]` in order. A `KeyError` is raised if none are found. This probe-based approach accommodates different dataset naming conventions without hardcoding a column name that may not be present.

### Replay Source Name Recording

Whether the full scaled path or the test-only fallback was used is recorded as `replay_source_name` in the ledger step and in the final `summary_payload`. This makes the replay source choice auditable after the fact.

## Validation Methodology

### Replay-First Design

Gold_06A validates by replaying rather than by reading back scored outputs. It loads the same fitted model artifacts that produced the training-run scores, applies them to the same feature columns, applies the same thresholds, and reruns the Stage 3 rule logic from config. The replay output is then compared to the training-notebook summary metrics. A match confirms that the saved artifacts are self-consistent and reproducible on unseen data.

This approach is stricter than checking only that a saved output file exists. It confirms that the model artifacts themselves can regenerate equivalent results, which is the relevant guarantee for downstream deployment or audit.

### Score Convention

`score_isolation_forest` uses `-model.score_samples(feature_frame)` so larger values indicate more anomalous rows. The helper comment states: "score_samples returns negative values where more negative = more anomalous. Negating aligns with the project convention where higher score = more anomalous." This is the same convention used by the training notebooks' inline helpers, so score column values are directly comparable across training and replay outputs.

### Cascade Stage Gate

In `run_cascade_replay`, Stage 2 flag gating is implemented as:

```
stage2_flag = (stage1_flag == 1) AND (stage2_raw_flag == 1)
```

`stage2_raw_flag` is produced by applying the Stage 2 model and threshold to all rows. `stage2_flag` gates this to rows that also fired Stage 1. This mirrors the original cascade design: Stage 2 is a confirmation stage that narrows Stage 1 candidates, not an independent scorer.

### Stage 3 Routing

Two separate Stage 3 implementations are defined inline:

**`add_stage3_broad_rules`** — applies the broad Stage 3 logic used by Gold_03A and Gold_03B. Reads `min_primary_sensor_hits`, `min_secondary_sensor_hits`, `rolling_window_size`, and `minimum_flags_in_window` from the variant's config. Combines profile breach, corroboration, persistence, and drift into `stage3_rule_evidence_count`. Final flag: `stage1_flag == 1 AND stage2_flag == 1 AND (profile_breach >= min_primary OR rule_evidence_count >= 2)`.

**`add_stage3_improved_rules`** — applies the weighted Stage 3 logic from Gold_03C. Reads `stage3_selected_params` from the threshold artifact and `stage3` config from the resolved YAML. Produces a `stage3_weighted_score` (sum of weighted evidence components) and four operating-mode flag columns: `cascade_final_flag` (default), `cascade_stage3_relaxed_flag` (weighted_score >= 2.0), `cascade_stage3_medium_flag` (>= 3.0), `cascade_stage3_strict_flag` (>= 5.0).

The `use_improved_stage3` parameter in `run_cascade_replay` routes between these two implementations. `cascade_default` and `cascade_tuned` use `add_stage3_broad_rules`. `stage3_improved` uses `add_stage3_improved_rules`.

### Seven-Row Metric Computation

Metrics are computed for seven model rows from four replay DataFrames:

| `model_id` | Source DataFrame | `flag_column` |
|---|---|---|
| `baseline` | `baseline_replay` | `baseline_flag` |
| `cascade_default` | `cascade_default_replay` | `cascade_final_flag` |
| `cascade_tuned` | `cascade_tuned_replay` | `cascade_final_flag` |
| `stage3_improved` | `stage3_improved_replay` | `cascade_final_flag` |
| `stage3_relaxed` | `stage3_improved_replay` | `cascade_stage3_relaxed_flag` |
| `stage3_medium` | `stage3_improved_replay` | `cascade_stage3_medium_flag` |
| `stage3_strict` | `stage3_improved_replay` | `cascade_stage3_strict_flag` |

The last four rows all read from `stage3_improved_replay` using different flag columns. This avoids running a separate replay pipeline for each Stage 3 operating mode: `add_stage3_improved_rules` produces all four flag columns in one pass.

`compute_binary_metrics` filters to test rows via `test_mask` and computes `alert_count_test_rows`, `precision`, `recall`, `f1`, `tn`, `fp`, `fn`, `tp`. Optional `roc_auc` and `pr_auc` are appended when a score column is provided and both label classes are present in the test rows.

### Two-Pass Validation Status

Expected metrics are read from the saved Gold_02/Gold_03 summary JSON files by `build_expected_metrics_from_training_artifacts`. The function extracts `expected_alert_count_test_rows`, `expected_precision`, `expected_recall`, `expected_f1` for all seven model rows. Stage 3 operating mode expected values come from the `cascade_metrics` section of the `stage3_improved` summary.

The two DataFrames are merged on `model_id` into `replay_comparison_dataframe`. Delta columns are computed for all four metrics. Validation status is then assigned in two passes:

**Pass 1 — Exact match** (`METRIC_TOLERANCE = 1e-9`):
- All four metric deltas within tolerance → `validation_status = "pass"`
- Otherwise → `validation_status = "review_delta"`

**Pass 2 — Secondary tolerance** (applied only when Pass 1 is not `"pass"`):
- `ALERT_COUNT_TOLERANCE = 1`: alert count delta ≤ 1
- `METRIC_TOLERANCE_RELAXED = 0.0001`: precision/recall/F1 deltas ≤ 0.0001
- Both conditions met → `tolerance_validation_status = "pass_with_tolerance"`
- Otherwise → `tolerance_validation_status = "review_delta"`

**Final reportable status** (`final_validation_status`):
- `"exact_pass"` — strict exact match within `1e-9`
- `"pass_with_tolerance"` — outside exact tolerance but within secondary threshold
- `"review_delta"` — exceeds both thresholds; requires review

The `interpretation` cell confirms the tolerance design intent: "Rows that fail the exact replay check are evaluated with a small tolerance: alert-count delta <= 1 and precision/recall/F1 deltas <= 0.0001."

### Comparison Against Training Summary Artifacts, Not Gold 04

The interpretation cell states directly: "Replay metrics are compared against the saved training-notebook summary artifacts, not against Gold 04 as the primary benchmark." Gold 04 performs statistical comparison between model variants. Gold_06A tests a different question: does the same model produce the same numbers on unseen data? These are separate validation concerns. Using Gold 04 as the comparison baseline would conflate model selection with reproducibility validation.

## Model Output Contract Behavior

Gold_06A does not load formal `gold_model_output_validation_contract` JSON files as defined by the `build_gold_model_output_validation_contract` utility. The artifact path dictionaries in Cell 7 cover `MODEL_ARTIFACTS`, `THRESHOLD_ARTIFACTS`, `SUMMARY_ARTIFACTS`, `PROFILE_ARTIFACTS`, and `CONFIG_ARTIFACTS` — no contract artifact dictionary is constructed.

The de facto contract is the summary JSON metric sections. `build_expected_metrics_from_training_artifacts` reads the `baseline_metrics` section from the baseline summary and the `cascade_metrics` section from each cascade summary. These section keys and their expected subkeys (`alert_count_test_rows`, `precision`, `recall`, `f1`) serve as the expectation source. If a summary file lacks the expected section key, indexing will raise `KeyError` before any comparison occurs.

The absence of formal contract files does not weaken the validation: the summary JSON artifacts are the primary training-run output records, and comparing replayed metrics against them is a direct reproducibility check.

## Artifact and Output Integrity Checks

### Pre-Load Existence Inventory

The `artifact_inventory` DataFrame covers all artifact categories before any data is loaded. This provides a complete pre-flight view of artifact state. A single missing file stops the notebook with a `FileNotFoundError` that names all absent artifacts, preventing any partial-load state.

### Column Presence Checks

`ensure_columns(dataframe, columns, context=...)` is called before each `score_isolation_forest` invocation. Missing feature columns raise `KeyError` with the column list and a context label identifying which stage failed. This prevents cryptic pandas `KeyError` messages during multi-variant scoring.

### `meta__is_train_flag` Integrity

When the full scaled path is used, `meta__is_train_flag` presence is checked immediately after loading. The notebook raises `KeyError` with an explicit message: "Gold 06A needs this column to evaluate held-out test rows." A missing `meta__is_train_flag` would silently apply metrics to all rows rather than held-out test rows, so the check is a hard stop.

### Post-Replay Row Count Verification

A ledger step `"model_variants_replayed"` records row counts for all four replay DataFrames plus the test row count. This provides an auditable record that the replay DataFrames match the expected source row count.

## Validation Result Construction

### `replay_comparison_dataframe` Structure

`replay_comparison_dataframe` is produced by merging `replay_metrics_dataframe` and `expected_metrics_dataframe` on `model_id`. Each row contains:

- `model_id`, `model_label`, `source_notebook`, `flag_column`, `score_column`
- Replayed metric values: `alert_count_test_rows`, `precision`, `recall`, `f1`, `tn`, `fp`, `fn`, `tp`; optional `roc_auc`, `pr_auc`
- Expected metric values: `expected_alert_count_test_rows`, `expected_precision`, `expected_recall`, `expected_f1`
- Delta columns for all four metrics
- `validation_status` (Pass 1 result)
- `tolerance_validation_status` (Pass 2 result)
- `final_validation_status` (`"exact_pass"`, `"pass_with_tolerance"`, or `"review_delta"`)

The `final_validation_status` column is the primary audit deliverable.

### `replay_scores_dataframe` Structure

The wide test-row score DataFrame contains identity columns (probed from a candidate list including `meta__row_id`, `meta__asset_id`, `machine_status`, and the resolved label column) plus a `plot_order_index` (integer position within test rows only) plus per-model prefixed columns.

Prefix pattern: `{model_key}__{column_name}`. Columns are included only when present in the source replay DataFrame. Score sources:

- `baseline__baseline_score`, `baseline__baseline_flag`
- `cascade_default__stage1_score`, `cascade_default__stage1_flag`, `cascade_default__stage2_score`, `cascade_default__stage2_raw_flag`, `cascade_default__stage2_flag`, `cascade_default__cascade_final_flag`
- `cascade_tuned__*` — same set as `cascade_default`
- `stage3_improved__stage1_score`, `stage3_improved__stage1_flag`, `stage3_improved__stage2_score`, `stage3_improved__stage2_raw_flag`, `stage3_improved__stage2_flag`, `stage3_improved__stage3_weighted_score`, `stage3_improved__cascade_final_flag`, `stage3_improved__cascade_stage3_relaxed_flag`, `stage3_improved__cascade_stage3_medium_flag`, `stage3_improved__cascade_stage3_strict_flag`

The index is reset before joining: `replay_scores_dataframe = replay_scores_dataframe.reset_index(drop=True)` and each source DataFrame's test-row slice is also `.reset_index(drop=True)` before column assignment. This makes position-aligned joins safe even when the source DataFrame index is not contiguous.

The `plot_order_index` column provides Gold_06B with a monotone integer position within test rows, analogous to the `plot_order_index` used by Gold_05 for timeline analysis.

### `summary_payload` Structure

The summary JSON contains: `stage`, `dataset`, `recipe_id`, `replay_source_name`, `replay_source_rows`, `test_rows`, `label_column`, `model_count`, `exact_pass_count`, `tolerance_pass_count`, `pass_count`, `review_count`, `alert_count_tolerance`, `metric_tolerance_relaxed`, and all four output artifact paths. This provides a compact run-level summary for audit or reporting without requiring the full comparison DataFrame.

## Artifact and SQL Persistence

### Output Artifacts

| Artifact | Directory | Filename |
|---|---|---|
| Replay metrics (7 rows) | `VALIDATION_RESULTS_DIR` | `{DATASET_NAME}__gold06a__test_replay_metrics.csv` |
| Replay vs training comparison | `VALIDATION_RESULTS_DIR` | `{DATASET_NAME}__gold06a__test_replay_vs_training_artifacts.csv` |
| Row-level test replay scores | `VALIDATION_SCORES_DIR` | `{DATASET_NAME}__gold06a__test_replay_scores.csv` |
| Summary JSON | `VALIDATION_SUMMARY_DIR` | `{DATASET_NAME}__gold06a__test_replay_summary.json` |

All four are saved in Cell 18. The metrics and comparison DataFrames use `.to_csv(..., index=False)`. The scores DataFrame uses `.to_csv(..., index=False)`. The summary payload uses `save_json`.

A `VALIDATION_PLOTS_DIR` is created under `VALIDATION_ROOT / "plots"` but no plot files are written in the current notebook source.

### No SQL Writes

Gold_06A does not write to PostgreSQL. No `get_engine_from_env()` call appears. No SQL touchpoints are confirmed from available source. This is intentional: Gold_06A is a lightweight validation pass. SQL metadata logging of this stage would require a database connection that is unnecessary for the reproducibility check this notebook performs.

### No W&B Uploads

Gold_06A does not call `wandb.init()` and does not upload artifacts to W&B. The workflow reference confirms: "There is no `wandb.init()` and no database connection."

### No Truth Record

Gold_06A does not construct a truth record, call `initialize_layer_truth`, or use `stamp_truth_columns`. The validation outputs do not carry `meta__truth_hash` or `meta__parent_truth_hash` columns. This is consistent with the notebook's role as a validation check rather than a pipeline stage that produces scored outputs for downstream Gold modeling.

### Ledger

The `Ledger` is provided by `CTX.ledger`. Ledger steps added:

- `"context_loaded"` — bootstrap complete
- `"required_artifacts_found"` — artifact existence check passed
- `"replay_source_loaded"` — data loaded with row count, label column, anomaly count, feature list sizes
- `"model_variants_replayed"` — four replay DataFrames produced with row counts
- `"gold06a_outputs_saved"` — complete summary payload recorded

The ledger is written to disk as part of `CTX.ledger` management. No explicit `ledger.write_json(path)` call appears in the source; the ledger write path is Not determined from available source beyond CTX management.

## Truth, Audit, and Reproducibility Behavior

Gold_06A provides reproducibility evidence rather than constructing its own truth record.

The `replay_comparison_dataframe` and its `final_validation_status` column are the primary audit evidence. A reviewer can inspect whether each of the seven model rows achieved `exact_pass`, `pass_with_tolerance`, or `review_delta` and trace any delta to the specific metric (`alert_count`, `precision`, `recall`, `f1`).

The `replay_source_name` field (`"full_scaled_with_test_mask"` or `"test_only_fallback"`) in the summary JSON makes the replay conditions auditable. A reviewer looking at the summary can determine whether the full training-row context was available for Stage 3 rules.

The summary JSON includes `recipe_id` and `dataset`, which identify which pipeline configuration was active. Model artifact paths embedded in `MODEL_ARTIFACTS` provide a direct link back to the joblib files that were replayed.

The tolerance constants (`METRIC_TOLERANCE = 1e-9`, `METRIC_TOLERANCE_RELAXED = 0.0001`, `ALERT_COUNT_TOLERANCE = 1`) are notebook-level constants visible in the source. Any reviewer can verify which tolerance was applied to a given `final_validation_status` row.

## Downstream Technical Handoff

The primary handoff artifact for Gold_06B is `{DATASET_NAME}__gold06a__test_replay_scores.csv` in `VALIDATION_SCORES_DIR`. The interpretation cell states: "Gold 06B uses the saved Gold 06A replay score output for test-set early-warning validation." The workflow reference confirms Gold_06B requires this file with no fallback (`FileNotFoundError` if absent).

The secondary artifacts — replay metrics CSV, comparison CSV, and summary JSON — are available for audit and reporting but are not confirmed as direct pipeline inputs to Gold_06B.

No direct handoff to final report assets or presentation artifacts is confirmed from available Gold_06A source. Whether the validation comparison CSV or summary JSON are consumed by a final report or appendix notebook is Not determined from available source.

## Key Technical Decisions

| Decision | Source Evidence | Why It Matters | Verification Method |
|---|---|---|---|
| `CONFIG_RUN_MODE = "test"` hardcoded | Cell 5 bootstrap constant | Gold_06A is the only Gold notebook that runs in test mode; this distinguishes it as a hold-out validation step rather than a training or analysis step | Confirm `CTX.mode == "test"` in the loaded context |
| Full scaled path preferred over test-only slice | Cell 7 comment; Cell 9 load logic | Stage 3 rolling/persistence/drift rules require training-row history to match the original notebook exactly; test-only rows would break rolling-window continuity for Stage 3 | Confirm `replay_source_name == "full_scaled_with_test_mask"` in summary JSON when the full scaled artifact exists |
| Artifact existence check before any data load | Cell 7 `artifact_inventory` construction; `FileNotFoundError` on missing artifacts | Prevents partial replay states where some models are scored and others fail mid-run; surfaces all missing artifacts at once | Rename a model joblib temporarily and confirm `FileNotFoundError` names all missing artifacts |
| All helpers defined inline | Cells 11, 12, 14, 16 — no utility imports for scoring, metric computation, or rule logic | Makes Gold_06A self-contained and auditable without dependency on utility module state; reduces risk that a utility module change silently changes validation behavior | Confirm no `from utils.medallion.gold` imports appear in the notebook; all scoring and rule logic is local |
| `-score_samples` convention in `score_isolation_forest` | Cell 11 helper comment | Aligns replay score convention with training-notebook helpers so replayed score values are directly comparable | Confirm `score_isolation_forest` returns `-model.score_samples(...)` and that higher values are more anomalous |
| `stage2_flag` gating: Stage 2 confirms Stage 1 | Cell 14 `run_cascade_replay`; `stage2_flag = stage1_flag == 1 AND stage2_raw_flag == 1` | Mirrors the original cascade design; gating ensures Stage 2 metric counts are computed on the same population that the training notebooks gated | Confirm `stage2_flag.sum() <= stage2_raw_flag.sum()` in any cascade replay result |
| `add_stage3_broad_rules` vs `add_stage3_improved_rules` routing | Cell 12; `use_improved_stage3` parameter in `run_cascade_replay` | Cascade defaults and tuned use simple primary/secondary breach rule; stage3_improved uses weighted evidence with four operating modes; separate implementations prevent logic cross-contamination | Confirm cascade_default and cascade_tuned replay DataFrames contain `stage3_profile_breach_flag`; confirm stage3_improved replay contains `stage3_weighted_score` |
| Four operating mode flags from one replay | Cell 12 `add_stage3_improved_rules`; all four flag columns produced in one pass | Avoids running four separate replay pipelines for Stage 3 operating modes; the weighted score is computed once and thresholded four ways | Confirm `stage3_improved_replay` contains `cascade_stage3_relaxed_flag`, `cascade_stage3_medium_flag`, `cascade_stage3_strict_flag` all at once |
| Comparison against training summary artifacts, not Gold 04 | Cell 20 `interpretation["comparison_scope"]` | Gold 04 compares model variants; Gold_06A checks reproducibility; using Gold 04 as the baseline would conflate model selection with validation | Confirm `build_expected_metrics_from_training_artifacts` reads from `summaries["baseline"]` and `summaries["stage3_improved"]` — not from any Gold_04 artifact path |
| Two-pass tolerance validation | Cell 16 `METRIC_TOLERANCE = 1e-9`, `METRIC_TOLERANCE_RELAXED = 0.0001`, `ALERT_COUNT_TOLERANCE = 1` | Exact match at `1e-9` confirms perfect reproducibility; secondary tolerance (`pass_with_tolerance`) allows for acceptable floating-point divergence without masking real regression; `review_delta` flags genuine divergence | Inject a synthetic replay row with a 0.00005 F1 delta and confirm `final_validation_status == "pass_with_tolerance"` |
| Wide prefixed score export | Cell 18; `{prefix}__{column}` naming | Lets Gold_06B read any model variant's flag and score columns without ambiguity; prefix-per-model makes the column source explicit | Confirm `replay_scores_dataframe` contains `baseline__baseline_flag` and `stage3_improved__cascade_stage3_medium_flag` |
| Index reset before score joining | Cell 18 `.reset_index(drop=True)` on base and source DataFrames | Makes position-aligned column assignment safe when the source DataFrame index is not contiguous after test-row masking | Confirm `replay_scores_dataframe.shape[0] == test_mask.sum()` after construction |
| No W&B, no SQL, no truth record | Cells 5, 18; no `wandb.init()`, no DB connection, no `initialize_layer_truth` | Gold_06A is a lightweight validation pass; adding W&B, SQL, or truth record overhead is disproportionate to its narrow scope | Confirm no `wandb.init()`, `get_engine_from_env()`, or `initialize_layer_truth` calls appear in the source |
| `require_string_list` and `require_mapping` defined inline | Cell 4; inline helpers with type comments | Converts `Any` returns from `load_json` / `yaml.safe_load` to concrete types immediately; the comment confirms the Pylance type-safety intent | Confirm both helpers raise `TypeError` when supplied with a non-list or non-dict value |

## Failure Modes and Guardrails

| Failure Condition | Behavior | Prevention / Guardrail |
|---|---|---|
| Any model, threshold, summary, profile, or feature artifact missing | `FileNotFoundError` with full missing-artifact table | Pre-load `artifact_inventory` existence check |
| `meta__is_train_flag` absent from full scaled path | `KeyError`: "Gold 06A needs this column to evaluate held-out test rows" | Explicit column check immediately after load |
| Label column not found among candidates | `KeyError` with candidate list | Probe-and-raise in label column resolution block |
| Feature columns missing from replay source | `KeyError` from `ensure_columns(context=...)` | Called before every `score_isolation_forest` invocation |
| Feature JSON or threshold JSON is not a list/dict | `TypeError` from `require_string_list` or `require_mapping` | Applied to all `load_json` returns before use |
| Config YAML is not a dict | `TypeError` from `require_mapping` on `yaml.safe_load` result | Applied to each YAML payload |
| Summary JSON missing expected section key (`baseline_metrics`, `cascade_metrics`) | `KeyError` during `build_expected_metrics_from_training_artifacts` | No explicit guard; fails at indexing |
| Stage 3 config section missing from resolved YAML | `KeyError` from `require_mapping(gold_cascade_config.get("stage3"), ...)` | `require_mapping` rejects `None` |
| `stage3_selected_params` missing from threshold artifact | `KeyError` from `dict(thresholds_payload["stage3_selected_params"])` | No explicit guard; fails at indexing |
| Both label classes absent from test rows (AUC computation) | `roc_auc` / `pr_auc` are not computed; only binary metrics are returned | `len(np.unique(y_true)) == 2` guard before AUC calls |
| `replay_comparison_dataframe` has no matching `model_id` | Merge produces NaN expected values; delta columns are NaN; `final_validation_status` is undefined | No explicit guard; NaN propagation in status computation |
| Full scaled path missing and test-only path also missing | `FileNotFoundError` at `pd.read_parquet` on `TEST_DATA_PATH` | No explicit guard beyond pre-load artifact inventory (which checks model/threshold/summary/profile/feature artifacts, not the data Parquet) |
| `cascade_stage3_relaxed_flag` or other operating-mode columns absent from `stage3_improved_replay` | Column absent from `replay_scores_dataframe`; metric computation falls back to available flag columns only | `if column in source_df.columns` guard in score source loop |
| Score column absent when AUC is attempted | AUC computation is skipped; only binary metrics are included | `score_column is not None and score_column in test_dataframe.columns` guard |

## Verification Checklist

- Active notebook path is `notebooks/experiments/EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation.ipynb`
- `CONFIG_RUN_MODE = "test"` is hardcoded in Cell 5
- Full scaled Parquet exists at the path in `RESOLVED_PATHS["gold_preprocessed_scaled_data_path"]` and contains `meta__is_train_flag`
- All 7 model joblib artifacts exist; all 4 threshold, summary, and config artifacts exist; all 3 profile CSV artifacts exist; all 4 feature JSON artifacts exist
- `artifact_inventory` DataFrame shows all artifacts present before any data is loaded
- `replay_source_name == "full_scaled_with_test_mask"` is recorded in ledger and summary JSON when the full scaled path is available
- `replay_metrics_dataframe` has 7 rows, one per model spec
- `expected_metrics_dataframe` has 7 rows with populated `expected_alert_count_test_rows`, `expected_precision`, `expected_recall`, `expected_f1`
- `replay_comparison_dataframe` contains `final_validation_status` for all 7 model rows with values in `{"exact_pass", "pass_with_tolerance", "review_delta"}`
- `{DATASET_NAME}__gold06a__test_replay_metrics.csv` exists in `VALIDATION_RESULTS_DIR`
- `{DATASET_NAME}__gold06a__test_replay_vs_training_artifacts.csv` exists in `VALIDATION_RESULTS_DIR`
- `{DATASET_NAME}__gold06a__test_replay_scores.csv` exists in `VALIDATION_SCORES_DIR`
- `{DATASET_NAME}__gold06a__test_replay_summary.json` exists in `VALIDATION_SUMMARY_DIR`
- `replay_scores_dataframe` contains `baseline__baseline_flag` and `stage3_improved__cascade_stage3_medium_flag`
- `replay_scores_dataframe` row count equals `test_mask.sum()`
- No `wandb.init()` call in the notebook source
- No `get_engine_from_env()` or SQL write in the notebook source
- No `initialize_layer_truth` or `stamp_truth_columns` in the notebook source
- Ledger contains `"required_artifacts_found"`, `"replay_source_loaded"`, `"model_variants_replayed"`, `"gold06a_outputs_saved"` steps

## Source-Limited Items

- The exact mechanism by which `CTX.ledger` writes the ledger to disk (path, format) is Not determined from available Gold_06A source; no explicit `ledger.write_json(path)` call appears in Cell 18 or elsewhere.
- Whether `VALIDATION_PLOTS_DIR` is used by a future version of Gold_06A is Not determined from available source; the directory is created but no plot files are confirmed.
- Whether the replay comparison CSV or summary JSON are consumed by a final report or appendix notebook beyond Gold_06B is Not determined from available source.
- Whether a `stage3_improved` summary JSON always contains the `cascade_metrics` section with all four operating-mode metric entries is Not determined from available source; indexing will raise `KeyError` if any expected key is missing.
- The exact Stage 3 weighted score thresholds (2.0, 3.0, 5.0) for operating modes in `add_stage3_improved_rules` are read from `thresholds_payload["stage3_selected_params"]`; whether these are fixed constants or configurable is Not determined from available source beyond the threshold artifact content at run time.
