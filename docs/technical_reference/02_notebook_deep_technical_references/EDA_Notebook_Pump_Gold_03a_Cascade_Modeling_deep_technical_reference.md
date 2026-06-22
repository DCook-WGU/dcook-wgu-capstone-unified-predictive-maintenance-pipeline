# Gold 03a Deep Technical Reference

## Purpose of This Deep Reference

This document explains the technical decisions and design invariants behind Gold_03a's three-stage cascade anomaly detection implementation. It is not a workflow summary — the 071b workflow reference covers section-level operations. This document explains why specific methodological choices were made, how the stages interact as a cascade, what invariants the code enforces before and after scoring, and what would break if key design choices were changed.

---

## Technical Scope

Decision tags applicable to this notebook:

| Tag | Reason Applicable |
|---|---|
| `MODEL_TRAINING` | `IsolationForest.fit()` for Stage 1 and Stage 2 on normal-only fit data |
| `MODEL_EVALUATION` | Optional precision, recall, F1 against ground-truth labels on the test partition |
| `DATA_VALIDATION` | Cascade gate integrity checks, feature column presence guards, score-length synchronization, `require_str_list` enforcement |
| `TRUTH_METADATA` | `gold_cascade` truth record initialized with `GOLD_PARENT_TRUTH_HASH`; truth hash stamped into all result rows |
| `ARTIFACT_WRITE` | Cascade results CSV/pickle, Stage 1 and Stage 2 joblib models, reference profile CSV, four JSON artifacts, four row-tracking CSVs, validation contract |
| `SQL_WRITE` | `write_gold_cascade_scores_sql` → `gold.anomaly_detection_scores` with `model_stage="cascade_default_final"` |
| `WANDB_LOGGING` | `wandb.init`, `wandb.save` for all artifact and truth files, `wandb_run.finish()` |
| `LEDGER_UPDATE` | `ledger.add` at each key step; `ledger.write_json` at close |

Tags not applicable: `TEMPORAL_SMOOTHING`, `CORRELATION_REPAIR`, `VARIANCE_CONTROL`, `MEAN_ANCHORING`, `BOUNDS_CLIPPING`, `FAULT_INJECTION`, `PHASE_SPECIFIC_LOGIC`, `MISSINGNESS_REPLAY`.

---

## Source Grounding

| Source | Role |
|---|---|
| `notebooks/experiments/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling.ipynb` (155 cells) | Primary source of truth for all technical claims |
| `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling_code_reference.md` | 071b workflow reference (read-only context) |
| `notebook_inventory.json` | Active notebook path resolution and cell count confirmation |

All claims in this document are grounded in one of the above sources. Items that could not be confirmed from any source are labeled `Not determined from available source`.

---

## Stage Role in the Cascade

Gold_03a implements the `"default"` cascade variant — the untuned baseline configuration against which Gold_03b (`"tuned"`) and Gold_03c (`"stage3_improved"`) are compared in Gold_04. The `CASCADE_VARIANT = "default"` and `STAGE2_SELECTION_MODE = "fixed"` identifiers are both hardcoded; neither is driven from config or environment variables.

The cascade variant string controls:
- Artifact directory layout via `build_artifact_dirs_from_config(config, stage_key="gold_cascade", variant="default")`
- Model and output file naming (files include the `03a` marker)
- The `model_id="cascade_default"` and `model_stage="cascade_default_final"` values written to the validation contract and SQL table

Gold_04's three-source parent hash check explicitly requires that Gold_03a, Gold_03b, and Gold_03c all share the same `GOLD_PARENT_TRUTH_HASH`. This means all three cascade variants must derive from the same Gold_01 preprocessing run.

---

## Input Contract and Lineage

### Primary Inputs

| Input | Load Mechanism | Strict Validation |
|---|---|---|
| Gold_01 scaled Parquet (`GOLD_PREPROCESSED_SCALED_DATA_PATH`) | `load_data(path)` | `GOLD_DATASET_NAME` extracted from `meta__dataset` column; `ValueError` if empty |
| Gold parent truth record | `load_parent_truth_record_from_dataframe(parent_layer_name="gold", column_name="meta__truth_hash")` | Raises if truth file not found; entire cascade halts |
| Four feature/sensor JSON lists | `require_str_list(load_json(path))` on each | `TypeError`/`ValueError` before cascade begins if any list is empty or non-string |
| Five Gold Parquets (preprocessed, fit, test, train, scaled) | `load_data(path)` per path | Loaded using truth-overridden paths, not configured defaults |

### Path Override Mechanism

Eight artifact paths are overridden from the Gold parent truth record at runtime:

- `GOLD_PREPROCESSED_DATA_PATH`, `GOLD_FIT_DATA_PATH`, `GOLD_TEST_DATA_PATH`, `GOLD_TRAIN_DATA_PATH`
- `STAGE1_FEATURES_PATH`, `STAGE2_FEATURES_PATH`, `STAGE3_PRIMARY_PATH`, `STAGE3_SECONDARY_PATH`

The values in the notebook's config block at session start serve as defaults only. The actual paths used for all eight inputs come from `gold_truth["artifact_paths"]`. This is the mechanism that guarantees Gold_03a loads exactly the same files that Gold_01 saved, regardless of how the config was initialized in the current session.

`GOLD_PARENT_TRUTH_HASH = get_truth_hash(gold_truth)` is captured at this point and carried forward into the cascade truth record, the validation contract lineage payload, and the cascade summary JSON. It is also the value verified in Gold_04's three-source check.

### Train/Test Split Recovery

The train/test partition is not re-derived inside Gold_03a. It is restored from the `meta__is_train_flag` column that Gold_01 stamped into the scaled Parquet. A hard `ValueError` fires if the column is absent. This design ensures the cascade evaluates on exactly the same held-out rows as the preprocessing evaluation — re-deriving the split independently would risk a different partition boundary.

---

## Model Input Preparation

### Stage 3 Reference Profile

The reference profile is built from the **fit (normal-only) subset** of the Gold data, not from the full scaled dataset. The feature union is `stage1_feature_columns + stage3_primary_rule_sensors + stage3_secondary_rule_sensors` deduplicated by `dict.fromkeys`.

```
build_reference_profile(gold_fit_dataframe, feature_columns=reference_profile_features)
```

Columns in the resulting `reference_profile` DataFrame: `feature_name`, `median_value`, `mean_value`, `standard_deviation`, `lower_bound` (5th percentile), `upper_bound` (95th percentile).

Using fit-only data for the profile bounds ensures that anomalous test-window values do not expand the normal operating envelope. If the full dataset were used, true anomalies with out-of-range sensor values would raise the upper bound or lower the lower bound, reducing Stage 3 profile breach sensitivity.

### Feature Matrix Assembly

Four matrices are constructed from the feature column lists and the two source DataFrames:

| Matrix | Source DataFrame | Purpose |
|---|---|---|
| `stage1_train_fit_features` | `gold_fit_dataframe` (normal-only) | Stage 1 `IsolationForest.fit()` |
| `stage2_train_fit_features` | `gold_fit_dataframe` (normal-only) | Stage 2 `IsolationForest.fit()` |
| `stage1_all_features` | `gold_preprocessed_scaled_dataframe` (all rows) | Stage 1 score all rows |
| `stage2_all_features` | `gold_preprocessed_scaled_dataframe` (all rows) | Stage 2 score all rows |

All four are kept as DataFrames (not NumPy arrays) to preserve feature names for sklearn and to prevent single-column slices from being silently converted to Series. Pre-flight validation raises `ValueError` with the list of missing columns if any Stage 1 or Stage 2 feature is absent from either DataFrame.

---

## Stage-One Modeling Methodology

### Code Decision

Stage 1 trains a broad Isolation Forest on normal-only data and scores **all rows** — not just training rows — producing a wide candidate set with high recall. The anomaly score is computed as `-model.score_samples()` so that higher values indicate greater anomaly likelihood.

### How We Know From the Code

```python
stage1_model = IsolationForest(
    n_estimators=STAGE1_ESTIMATOR_COUNT,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
stage1_model.fit(stage1_train_fit_features)
stage1_train_scores = compute_anomaly_scores_isolation_forest(stage1_model, stage1_train_fit_features)
stage1_all_scores   = compute_anomaly_scores_isolation_forest(stage1_model, stage1_all_features)
stage1_threshold    = choose_threshold_value(stage1_train_scores, STAGE1_THRESHOLD_PERCENTILE)
stage1_flags        = (stage1_all_scores >= stage1_threshold).astype(int)
```

The comment in the source code states explicitly: "All rows are scored (not just test) because Stage 2 must receive a full-population Stage 1 flag mask to gate its candidate set."

### Why It Is Done

Threshold calibration on training scores only prevents test anomaly score distributions from influencing the cut-point. If the threshold were set from all-row scores, the presence of genuine anomalies in the test window would shift the percentile-based threshold, reducing Stage 1 recall on exactly the rows Stage 2 needs to see.

Scoring all rows in Stage 1 (rather than test rows only) is necessary because Stage 2 is gated on `stage1_flag`. A gate that covers only the test partition would make Stage 2 unreachable for any training-partition row, which would break the cascade for production scoring on a mixed population.

### How To Verify Programmatically

- Confirm `stage1_threshold` equals the `STAGE1_THRESHOLD_PERCENTILE`th percentile of `stage1_train_scores` — not of `stage1_all_scores`.
- Confirm `len(stage1_all_scores) == len(cascade_results)` (enforced by the synchronization guard before column assignment).
- Confirm `stage1_flags.sum() >= cascade_results["stage2_flag"].sum()` — every Stage 2 positive must have passed Stage 1.

---

## Candidate Generation Logic

### Stage 2: Gated Isolation Forest Confirmation

Stage 2 trains a separate Isolation Forest on a **reduced feature set** (`stage2_feature_columns` rather than `stage1_feature_columns`). It scores all rows but the confirmation flag is gated on Stage 1:

```python
stage2_raw_flags = (stage2_all_scores >= stage2_threshold).astype(int)
stage2_flags     = ((stage1_flags == 1) & (stage2_raw_flags == 1)).astype(int)
```

Row-tracking in Stage 2 uses `build_stage_scoring_frame(..., mask=stage2_candidate_mask)` to build a subset frame containing only Stage 1 positives. The raw helper output columns are renamed before merging back:

```python
stage2_results_df = stage2_results_df.rename(columns={
    "stage2_score":    "stage2_model_score",
    "stage2_decision": "stage2_model_decision",
    "stage2_pred":     "stage2_model_pred",
    "stage2_flag":     "stage2_model_flag",
})
```

This rename prevents the helper's `stage2_flag` output from overwriting the threshold-calibrated `stage2_flags` array computed from the Stage 2 model evaluation step.

Non-candidate rows (Stage 1 negatives) receive `stage2_score = NaN` in `cascade_results`, not zero. NaN is intentional — it signals "Stage 2 was not evaluated for this row" rather than "Stage 2 produced a zero anomaly score." This distinction is preserved in the final results and is visible in the row-tracking exports.

`finalize_stage_flag_columns(cascade_results, stage_names=["stage1","stage2","stage3"])` fills NaN values in flag columns as integers before truth record creation. Score NaN values for non-candidates are not filled.

### Stage 3: Rule-Based Evidence Confirmation

Stage 3 applies four evidence checks to **all rows** of `cascade_results`. The gating is enforced in the final expression rather than by filtering rows first:

```python
cascade_results["cascade_final_flag"] = (
    (cascade_results["stage1_flag"] == 1) &
    (cascade_results["stage2_flag"] == 1) &
    (
        (cascade_results["stage3_profile_breach_count"] >= STAGE3_MIN_PRIMARY_SENSOR_HITS) |
        (cascade_results["stage3_rule_evidence_count"] >= 2)
    )
).astype(int)
```

Computing Stage 3 evidence for all rows (not just Stage 2 positives) allows the evidence counts to be inspected on any row during audit, without having to identify whether the row was a Stage 2 candidate. It also avoids masking logic inside Stage 3 helper functions.

The four Stage 3 evidence checks and their sources:

| Check | Function | Source Input | Threshold Parameter |
|---|---|---|---|
| Primary breach (profile boundary) | `compute_primary_breach_count` | `stage3_primary_rule_sensors`, `reference_profile` | `STAGE3_MIN_PRIMARY_SENSOR_HITS` |
| Secondary corroboration | `compute_secondary_breach_count` | `stage3_secondary_rule_sensors`, `reference_profile` | `STAGE3_MIN_SECONDARY_SENSOR_HITS` |
| Temporal persistence | `compute_persistence_flag` | `cascade_results["stage2_flag"]` (not `stage1_flag`) | `STAGE3_ROLLING_WINDOW_SIZE`, `STAGE3_MINIMUM_FLAGS_IN_WINDOW` |
| Rolling drift | `compute_drift_flag` | `stage3_primary_rule_sensors + stage3_secondary_rule_sensors` union | `rolling_window_size=5`, `drift_threshold_multiplier=1.0` (hardcoded in call) |

The persistence check uses `stage2_flag`, not `stage1_flag`, so that persistence reflects consecutive confirmed two-stage alerts rather than the broader Stage 1 candidate population. A row that triggers Stage 1 repeatedly but never advances to Stage 2 confirmation does not contribute to the persistence count.

The drift check parameters (`rolling_window_size=5`, `drift_threshold_multiplier=1.0`) are hardcoded in the function call, not driven from `STAGE3_CFG`. This means changing `STAGE3_CFG` alone will not affect drift behavior in this notebook.

---

## Evaluation and Metrics

### Cascade Metrics Structure

`cascade_metrics` is a dict with per-stage alert counts for both all rows and the test partition:

```
"model": "3-Stage Cascade"
"stage1_alert_count_all_rows", "stage1_alert_count_test_rows"
"stage2_alert_count_all_rows", "stage2_alert_count_test_rows"
"final_alert_count_all_rows", "final_alert_count_test_rows"
```

If `test_labels` is not None (i.e., `anomaly_flag` is present in the Gold scaled DataFrame), supervised evaluation is appended:

```python
precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels_array,
    cascade_test_flags,
    average="binary",
    zero_division=0,
)
```

Evaluation is applied to the test partition only (`test_mask`). The `zero_division=0` argument means precision and recall return 0.0 if there are no positive predictions, rather than raising an error — this prevents notebook failure in edge cases where the cascade produces no alerts on the test set.

`cascade_metrics` is carried into `cascade_summary`, the validation contract, and the ledger close step, making it available to downstream notebooks and W&B without requiring them to reload the scored results.

---

## Artifact and SQL Persistence

### Model Persistence

| Artifact | Persistence | Paths |
|---|---|---|
| Stage 1 Isolation Forest | `joblib.dump` | `STAGE1_MODEL_ARTIFACT_PATH` (primary), `STAGE1_MODELS_PATH` (models root) |
| Stage 2 Isolation Forest | `joblib.dump` | `STAGE2_MODEL_ARTIFACT_PATH` (primary), `STAGE2_MODELS_PATH` (models root) |
| Stage 3 rule logic | **Not persisted** | Fully reconstructable from `reference_profile` + config thresholds |

Stage 3 is not saved as a joblib artifact because it consists entirely of threshold comparisons against the reference profile. The profile bounds and config parameters are recorded in the validation contract, making Stage 3 reproducible without model serialization. The contract explicitly records `stage3_saved_as_joblib=False` so Gold_06A does not attempt to load a Stage 3 model file.

### JSON Artifacts

| Artifact | Key Contents | Consumer |
|---|---|---|
| `cascade_thresholds` | `stage1_threshold_percentile`, `stage1_threshold`, `stage2_selection_mode`, `stage2_selected_threshold_percentile`, `stage2_threshold`, `stage2_best_params` | Gold_04 comparison |
| `cascade_summary` | Per-stage alert counts, metrics, feature/rule counts, truth hashes, process run IDs | Gold_04 comparison |
| `cascade_metadata` | All Gold_01 input paths, Gold_01 provenance (`scaler_kind`, `recommended_imputation`, `feature_set_id`), cascade artifact paths | Gold_04, Gold_05 provenance traceability |
| Validation contract (`cascade_default_contract`) | `model_id="cascade_default"`, `model_stage="cascade_default_final"`, full `rule_config`, `stage3_saved_as_joblib=False`, lineage payload with `CASCADE_TRUTH_HASH` and `GOLD_PARENT_TRUTH_HASH` | Gold_06A Test Replay Validation |

The `cascade_metadata` JSON records Gold_01 provenance fields (`gold_scaler_kind`, `gold_recommended_imputation`, `gold_feature_set_id`) alongside the cascade's own artifact paths. This gives comparison and audit notebooks a self-contained provenance record without requiring them to re-read the Gold_01 truth record directly.

### Row-Tracking Exports

Four CSV files are exported to `CASCADE_ROW_TRACKING_DIR` using dataset-namespaced filenames:

```
{DATASET_NAME}__gold__cascade_03a__row_tracking__stage1_detection.csv
{DATASET_NAME}__gold__cascade_03a__row_tracking__stage2_detection.csv
{DATASET_NAME}__gold__cascade_03a__row_tracking__stage3_evidence_detection.csv
{DATASET_NAME}__gold__cascade_03a__row_tracking__final_detection.csv
```

All four are registered with W&B via `wandb.save(...)`.

The Stage 3 evidence export flags on `stage3_profile_breach_flag`, not on `cascade_final_flag`. This captures rows that triggered Stage 3 profile bounds, including rows that were not Stage 2 candidates and therefore could not become final cascade alerts.

### SQL Write

```python
WRITE_TO_POSTGRES = True
CASCADE_SQL_MODEL_STAGE = "cascade_default_final"

if WRITE_TO_POSTGRES:
    write_gold_cascade_scores_sql(
        engine=engine,
        capstone_schema=CAPSTONE_SCHEMA,
        dataset_id=DATASET_ID,
        run_id=RUN_ID,
        notebook_globals=globals(),
        dataframe=cascade_results,
        dataset_name=DATASET_NAME,
        model_stage=CASCADE_SQL_MODEL_STAGE,
    )
```

Target: `gold.anomaly_detection_scores`. The `WRITE_TO_POSTGRES = True` gate allows the notebook to run in review mode by setting it to `False` without touching any SQL logic. The SQL write occurs after `wandb_run.finish()` and after the final lineage verification — it is the last substantive operation in the notebook.

---

## Truth, Audit, and Reproducibility Behavior

### Truth Record Chain

Gold_03a initializes its truth record with:

```python
initialize_layer_truth(layer_name="gold_cascade", parent_truth_hash=GOLD_PARENT_TRUTH_HASH)
```

The `gold_cascade` truth record's `parent_truth_hash` is Gold_01's preprocessing truth hash. When Gold_04 verifies its three-source check, it reads `GOLD_PARENT_TRUTH_HASH` from the cascade truth record, the cascade summary JSON, and the cascade metadata JSON, and requires all three to agree.

The `runtime_facts` section of the truth record captures:
- All stage thresholds and threshold percentiles
- Estimator counts for Stage 1 and Stage 2
- Stage 2 best params
- Feature and rule sensor counts
- Total result row count
- `gold_process_run_id` (from the Gold parent truth's own `process_run_id`)
- `gold_feature_set_id` (from `gold_truth_runtime_facts.get("feature_set_id")`)

Truth columns stamped into all rows of `cascade_results`:
- `meta__truth_hash` = `CASCADE_TRUTH_HASH`
- `meta__parent_truth_hash` = `GOLD_PARENT_TRUTH_HASH`
- `meta__pipeline_mode` = `PIPELINE_MODE`

### W&B Lifecycle

`wandb_run.finish()` is called in the ledger-close section, **before** the final lineage verification checks and before the SQL write. Artifacts saved after `wandb_run.finish()` are not registered with W&B. In Gold_03a's execution order, all `wandb.save(...)` calls complete before `wandb_run.finish()` — the final lineage checks and SQL write do not add to the W&B run.

### Final Lineage Checks (Seven Invariants)

After `wandb_run.finish()`, seven invariants are verified against `cascade_results` and the saved truth file:

1. `meta__truth_hash` column present in `cascade_results`
2. `meta__parent_truth_hash` column present in `cascade_results`
3. `meta__pipeline_mode` column present in `cascade_results`
4. `extract_truth_hash(cascade_results)` equals `CASCADE_TRUTH_HASH`
5. `cascade_results["meta__parent_truth_hash"]` contains exactly one unique non-null value
6. That unique parent hash value equals `GOLD_PARENT_TRUTH_HASH`
7. The saved truth file (`cascade_truth_path`) exists, is readable, and its `truth_hash` and `parent_truth_hash` fields match the in-memory values

`ValueError` is raised on any mismatch. `FileNotFoundError` is raised if the truth file does not exist. These checks catch late-stage column overwrites and confirm the truth record was actually written to disk.

### Ledger Behavior

`ledger.add(kind="step", step=..., message=..., data={...})` is called at each key step, including:
- After Stage 1 model fit and scoring
- After Stage 1 row-tracking merge
- After Stage 1 threshold synchronization
- After cascade output validation
- After Stage 3 flag finalization (twice — the notebook contains two calls to `finalize_stage_flag_columns` at cells 106 and 116)
- After row-tracking artifact saves
- At cascade modeling completion (final ledger close)

`ledger.write_json(cascade_ledger_path)` is called at the close section. Path: `GOLD_CASCADE_ARTIFACT_DIRS["lineage"] / GOLD_CASCADE_LEDGER_FILE_NAME`. The ledger JSON is registered with W&B before `wandb_run.finish()`.

---

## Downstream Technical Handoff

### Gold_04 (Comparison)

Gold_04 reads `cascade_results` CSV/pickle and the `cascade_summary`, `cascade_thresholds`, and `cascade_metadata` JSONs for the default variant. It cross-validates that `GOLD_PARENT_TRUTH_HASH` from Gold_03a, Gold_03b, and Gold_03c all match the value carried in Gold_02's parent truth hash.

Gold_04 establishes Gold_03a's `CASCADE_DEFAULTS_TRUTH_HASH` as the baseline cascade reference point in the comparison, against which tuned variants from Gold_03b and Gold_03c are evaluated.

### Gold_06A (Test Replay Validation)

Gold_06A reads the model output validation contract written at `cascade_default_contract_path`. The contract provides Gold_06A with:
- `model_id = "cascade_default"`, `model_stage = "cascade_default_final"`
- Full `rule_config` dict (all stage thresholds and Stage 3 rule parameters)
- `stage1_model_path`, `stage2_model_path` for model reload
- `stage3_saved_as_joblib = False` (Stage 3 is reconstructed from rule config)
- `output_artifact_path` (cascade results CSV path)
- Lineage payload: `cascade_truth_hash` and `parent_gold_truth_hash`

The contract is self-contained: Gold_06A does not need to read the Gold_03a truth record or config to perform validation replay.

Gold_06A also reads the per-stage detected rows from `CASCADE_ROW_TRACKING_DIR` for audit verification.

### Gold_05 (Anomaly Detection)

Gold_05 reads cascade results and metadata for final anomaly scoring and reporting. The `cascade_metadata` JSON provides Gold_05 with Gold_01 provenance (scaler kind, recommended imputation, feature set ID) so that the anomaly detection stage can trace preprocessing decisions without reloading the Gold_01 truth record.

### Gold_03b and Gold_03c

Gold_03b and Gold_03c do not read artifact files produced by Gold_03a. Their relationship is sequence order only — all three cascade variants are derived from the same Gold_01 preprocessing output and use the same `GOLD_PARENT_TRUTH_HASH`. Cross-variant artifact comparison is handled in Gold_04.

---

## Key Technical Decisions

| Decision | Source Evidence | Why It Matters | Verification Method |
|---|---|---|---|
| Threshold calibrated from training scores only | `choose_threshold_value(stage1_train_scores, STAGE1_THRESHOLD_PERCENTILE)` — training array, not all-row array; comment: "test scores unseen" | Test anomaly score distributions cannot shift the percentile cut-point; prevents leakage from evaluation set into threshold calibration | Confirm `stage1_threshold == np.percentile(stage1_train_scores, STAGE1_THRESHOLD_PERCENTILE)` |
| Stage 1 scores all rows (not just candidates) | `stage1_all_scores = compute_anomaly_scores_isolation_forest(stage1_model, stage1_all_features)` — `stage1_all_features` is derived from the full scaled DataFrame; in-code comment explains Stage 2 gating requires full-population flags | Stage 2 gate mask (`stage1_flag`) must cover the same row population that will be operationally scored; restricting to test partition breaks cascade for production or training-partition rows | `len(stage1_all_scores) == len(cascade_results)` enforced by synchronization guard before column assignment |
| Stage 2 NaN for non-candidate rows (not zero) | `cascade_results["stage2_score"] = NaN`; set only for `stage2_candidate_mask` rows; `finalize_stage_flag_columns` fills flag NaNs but not score NaNs | Distinguishes "Stage 2 not evaluated" from "Stage 2 score was zero"; auditors and downstream notebooks can filter `stage2_score.notna()` to identify Stage 2 candidates | `cascade_results["stage2_score"].notna().sum() == cascade_results["stage1_flag"].sum()` |
| Stage 2 raw output columns renamed before merge | `stage2_results_df.rename({"stage2_flag": "stage2_model_flag", ...})` before `cascade_results.merge(...)` | Prevents the helper's auto-generated `stage2_flag` from overwriting the threshold-gated `stage2_flags` array computed during model evaluation | Confirm `"stage2_model_flag"` and `"stage2_flag"` both present in `cascade_results`; confirm `stage2_flag` values match `stage2_flags` array, not the helper's raw flag |
| Stage 3 runs on all rows; gating enforced in final expression | `compute_primary_breach_count(cascade_results, ...)` — no mask; `cascade_final_flag` expression gates on `stage1_flag==1 & stage2_flag==1` | Evidence counts are available for inspection on every row during audit; no masking logic inside Stage 3 helper functions | `cascade_results["stage3_profile_breach_count"].notna().sum() == len(cascade_results)` |
| Persistence applied to `stage2_flag`, not `stage1_flag` | `compute_persistence_flag(cascade_results["stage2_flag"], ...)` — in-code comment: "not Stage 1 candidates" | Persistence reflects temporal clustering of confirmed two-stage alerts; repeated Stage 1 positives that never advance to Stage 2 do not trigger temporal evidence | Compare `stage3_persistence_flag` values to `stage2_flag` rolling window; confirm persistence is zero for rows where `stage2_flag` is consistently 0 in the window |
| Drift parameters hardcoded in call (not from `STAGE3_CFG`) | `compute_drift_flag(..., rolling_window_size=5, drift_threshold_multiplier=1.0)` — values are literals in the function call | Config-only changes to `STAGE3_CFG` will not affect drift behavior in this notebook; must update the function call directly | Read `STAGE3_CFG` and confirm `rolling_window_size` and `drift_threshold_multiplier` are absent from its keys (or differ from the hardcoded call values) |
| Stage 3 not persisted as joblib | `stage3_saved_as_joblib=False` in validation contract; no `joblib.dump` for Stage 3 logic anywhere in notebook | Stage 3 is purely rule-based; it is fully reconstructable from `reference_profile` + config thresholds; persisting it as a model would create an artifact with no additional reproducibility benefit | Confirm no Stage 3 joblib path appears in `GOLD_CASCADE_ARTIFACT_DIRS`; confirm `stage3_saved_as_joblib=False` in the written contract JSON |
| `wandb_run.finish()` before final lineage checks and SQL write | `ledger.write_json(cascade_ledger_path); wandb.save(str(cascade_ledger_path)); wandb_run.finish()` — then lineage verification and `write_gold_cascade_scores_sql` follow | Final lineage checks and SQL write are not part of the W&B run; no W&B logging is valid after `finish()`; the run record closes on successful artifact registration, not on SQL completion | Read the ledger JSON — it must exist and contain the `finalize_cascade_modeling` step; confirm the SQL write function was called after `wandb_run.finish()` |
| Eight artifact paths overridden from Gold parent truth record | `gold_truth["artifact_paths"]` provides all eight overrides; configured values from bootstrap are not used for these paths | Guarantees Gold_03a loads exactly the files Gold_01 saved, regardless of local config state; any environment where the Gold_01 truth record is missing or corrupted will fail at path-override time rather than at model fit time | Confirm `GOLD_FIT_DATA_PATH` after override matches a key in `gold_truth["artifact_paths"]`; load and compare paths before and after override |
| `validate_cascade_output` enforces cascade gate invariants | `validate_cascade_output(cascade_results, test_mask, final_flag_column="cascade_final_flag")` raises `ValueError` for gate violations; checks: required columns present, `meta__row_id` unique, binary flag values only, no `stage2_flag==1` where `stage1_flag!=1`, no `cascade_final_flag==1` where `stage2_flag!=1` | Prevents truth record creation on corrupted cascade results; gate invariants must hold before lineage columns are stamped into the output | Run `validate_cascade_output` on any checkpoint copy of `cascade_results`; any gate violation raises before truth record creation |
| Reference profile built from fit-only (normal) data | `build_reference_profile(gold_fit_dataframe, feature_columns=reference_profile_features)` — `gold_fit_dataframe` is the normal-only subset | Anomalous test-window sensor values must not expand the normal operating envelope; if the full dataset were used for bounds, genuine anomalies would raise upper bounds and reduce Stage 3 profile breach sensitivity | Confirm `reference_profile` was built from `gold_fit_dataframe` rows count, not `gold_preprocessed_scaled_dataframe` rows count |

---

## Failure Modes and Guardrails

| Failure Condition | Guard | Behavior |
|---|---|---|
| Gold_01 truth record missing or unreadable | `load_parent_truth_record_from_dataframe` | Raises; entire cascade halts before any input data is loaded |
| Feature or sensor JSON empty or non-string | `require_str_list(load_json(path))` | `TypeError` or `ValueError` before cascade begins |
| `meta__is_train_flag` column absent from input | Hard `ValueError` check before mask construction | Stops cascade; prevents re-derivation of a different split |
| Stage 1 or Stage 2 feature column missing from either DataFrame | Pre-flight column presence check | `ValueError` with list of missing columns; prevents silent shape mismatch in `fit()` |
| `stage1_all_scores` length differs from `cascade_results` length | Length equality guard before column assignment | `ValueError`; prevents misaligned score-to-row assignment in `stage1_score` column |
| Stage 3 rule sensor absent from scored DataFrame | Sensor presence check (cells 79–80) | Logs warning; Stage 3 rule computation proceeds on available sensors; hard stop is not enforced at this point |
| Required cascade output columns missing | `validate_cascade_output` required column check | `ValueError` before truth record creation |
| `meta__row_id` not unique in `cascade_results` | `validate_cascade_output` uniqueness check | `ValueError`; row-tracking merge validity depends on unique row IDs |
| Stage 2 gate violation (stage2_flag==1 without stage1_flag==1) | `validate_cascade_output` gate check | `ValueError` before truth record creation |
| Final cascade gate violation (cascade_final_flag==1 without stage2_flag==1) | `validate_cascade_output` gate check | `ValueError` before truth record creation |
| Truth hash stamped into results does not match computed hash | `extract_truth_hash(cascade_results) != CASCADE_TRUTH_HASH` check | `ValueError`; catches late-stage column overwrites |
| Multiple distinct parent hashes in `cascade_results` | `len(cascade_parent_values) != 1` check | `ValueError`; multiple values indicate rows from different upstream runs were mixed |
| Truth file not written to disk | `Path(cascade_truth_path).exists()` check | `FileNotFoundError`; confirms truth record was persisted before lineage checks pass |
| Hash mismatch between in-memory truth record and saved JSON file | `loaded_cascade_truth.get("truth_hash") != CASCADE_TRUTH_HASH` check | `ValueError`; confirms file contents match the in-memory record |

---

## Verification Checklist

Use these checks to verify that a Gold_03a run completed correctly.

- [ ] `CASCADE_VARIANT == "default"` and `STAGE2_SELECTION_MODE == "fixed"` in all artifact metadata
- [ ] `stage1_threshold` equals `np.percentile(stage1_train_scores, STAGE1_THRESHOLD_PERCENTILE)` — not derived from all-row scores
- [ ] `len(cascade_results["stage2_score"].dropna()) == cascade_results["stage1_flag"].sum()` — Stage 2 scores present only for Stage 1 positives
- [ ] `(cascade_results["stage2_flag"] == 1) & (cascade_results["stage1_flag"] != 1)` returns an empty frame — Stage 2 gate invariant holds
- [ ] `(cascade_results["cascade_final_flag"] == 1) & (cascade_results["stage2_flag"] != 1)` returns an empty frame — final gate invariant holds
- [ ] `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` all present in `cascade_results`
- [ ] All `meta__truth_hash` values in `cascade_results` equal `CASCADE_TRUTH_HASH`
- [ ] All non-null `meta__parent_truth_hash` values in `cascade_results` equal `GOLD_PARENT_TRUTH_HASH`
- [ ] Saved truth file exists at `cascade_truth_path`; its `truth_hash` and `parent_truth_hash` match in-memory values
- [ ] `stage3_saved_as_joblib = False` in the written validation contract JSON
- [ ] `model_id = "cascade_default"` and `model_stage = "cascade_default_final"` in the written validation contract JSON
- [ ] Both Stage 1 and Stage 2 joblib files exist at their configured artifact paths
- [ ] `gold.anomaly_detection_scores` contains rows with `model_stage = "cascade_default_final"` for the expected `dataset_id` and `run_id`
- [ ] Four row-tracking CSVs present in `CASCADE_ROW_TRACKING_DIR` with `cascade_03a` in their names
- [ ] Ledger JSON exists at `GOLD_CASCADE_ARTIFACT_DIRS["lineage"]`; contains `finalize_cascade_modeling` step entry
- [ ] `cascade_metadata` JSON records `gold_feature_set_id` and `gold_scaler_kind` from Gold_01 provenance

---

## Source-Limited Items

| Item | Limitation |
|---|---|
| Exact SQL column mapping in `write_gold_cascade_scores_sql` | Function is a project utility; its column-selection and upsert logic are not defined inline in this notebook. Full column mapping requires reading the utility source. |
| `CAPSTONE_SCHEMA` effect on SQL table name | The `CAPSTONE_SCHEMA` env var is used in the SQL smoke check and in `write_gold_cascade_scores_sql`. Whether the target table is `{CAPSTONE_SCHEMA}.anomaly_detection_scores` or always `gold.anomaly_detection_scores` cannot be confirmed without reading the utility source. |
| Exact `build_artifact_dirs_from_config` subdirectory layout | The named subdirectory keys (`models`, `scores`, `thresholds`, `summaries`, `metadata`, `profiles`, `lineage`, `row_tracking`, `plots`, `config`) are confirmed from the workflow reference. The physical path structure depends on the utility implementation. |
| `validate_cascade_output` return value structure | The function's summary dict content beyond the gate violation checks is not confirmed from the inline notebook source. |
| W&B artifact log types and metadata | `wandb.save(...)` is confirmed for all artifact files. Whether artifact types, aliases, or metadata dicts are passed is not confirmed from the available source. |
| `finalize_stage_flag_columns` behavior on `stage3_flag` alias | The notebook contains a conditional `cascade_results["stage3_flag"] = cascade_results["stage3_confirmed_flag"].fillna(0).astype(int)` at two points (cells 106 and 116). Whether `stage3_confirmed_flag` is ever present in the frame during a standard run is not determined from available source. |
