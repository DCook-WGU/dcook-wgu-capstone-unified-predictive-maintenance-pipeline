# Gold 01 Deep Technical Reference

## Purpose of This Deep Reference

This document covers the technical decisions in Gold 01 (Gold Preprocessing) that require deeper explanation than the workflow reference provides. The workflow reference describes what each section does. This document explains why Gold 01 is the single preprocessing foundation for every downstream Gold notebook, why it captures the Silver parent truth hash before any subsetting, why it resolves the feature registry from the Silver truth record rather than config, why the train/test split is built at the episode level and ordered chronologically, why imputation statistics are restricted to training rows, why the scaler is fit only on confirmed-normal training rows, why a prescaled snapshot is frozen before scaling, why the normal-only fit subset and reference profile are constructed the way they are, why Stage 2 and Stage 3 feature sets are derived from training stability, why the same Gold truth hash is stamped into all five output frames, and why W&B is active here when the Silver notebooks disable it.

## Technical Scope

- `gold_preprocessing` stage context and two-level sanity check (general + `GOLD_CFG`)
- Silver input contract: profiled dataframe plus normal-clean and normal-contaminated subsets
- Parent truth hash capture before subsetting; dataset, registry, impute-recommendation, and pipeline-mode resolution from Silver truth
- Stable row identity (`meta__row_id`) stamped before any Gold transform
- Chronological, episode-level, non-leaking train/test split
- Train-only imputation statistics; forward-fill within asset/run group then median fallback
- Prescaled snapshot in original sensor units
- Scaler fit on `train ∩ normal_clean` rows; fitted scaler persisted via joblib
- Normal-only fit subset for unsupervised model training
- Reference profile (median/mean/std plus 5th/95th-percentile bounds) on the fit subset
- Stage 2 feature ranking by coefficient of variation; Stage 3 sensor grouping by reference standard deviation
- Five output Parquets all stamped with the same `GOLD_TRUTH_HASH`
- Gold truth record, truth index, ledger, config snapshot, and W&B run
- SQL persistence via `write_gold_preprocessed_features_sql` behind a write gate
- Final lineage, column-symmetry, and episode-split verification

## Source Grounding

Sources used:

- `notebooks/experiments/EDA_Notebook_Pump_Gold_01_PreProcessing.ipynb` (active notebook — source of truth)
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_01_PreProcessing_code_reference.md` (read-only context)
- `notebook_inventory.json`
- `artifact_io_manifest.json`
- `sql_touchpoints.json`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Silver_01_PreEDA_deep_technical_reference.md` (read-only upstream context)
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3_deep_technical_reference.md` (read-only upstream context)
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Silver_02b_EDA_v2_deep_technical_reference.md` (read-only upstream context)
- `technical_reference/00_project_manual/` relationship maps (read-only context)

The active Gold 01 notebook source is the source of truth for all function behavior, variable names, output paths, and design decisions documented here.

## Stage Role in the Medallion Pipeline

Gold 01 is the first notebook in the Gold layer and the entry point to all Gold modeling. It converts the Silver analytical dataset into modeling-ready features and produces the stage-level artifacts that every subsequent Gold notebook consumes. No Gold modeling notebook re-runs Gold 01's preprocessing; they all root on its outputs.

Gold 01 performs, in a fixed and largely irreversible order: stable row identity stamping → chronological episode-based train/test split → numeric feature selection → conditional one-hot encoding → train-only imputation → prescaled snapshot → normal-only scaler fit and scaling → normal-only fit subset → reference profile → Stage 2 feature ranking → Stage 3 sensor grouping → truth record finalization and output stamping → artifact and SQL persistence. Each step is recorded in the truth record, the ledger, and W&B, so any downstream notebook can reconstruct exactly which rows and features it received and how they were preprocessed.

Gold 01 also creates the first Gold-layer truth record (`gold_preprocessing` stage). The same `GOLD_TRUTH_HASH` is stamped into all five output Parquets, which is the mechanism downstream notebooks use to confirm they are reading from one consistent preprocessing run. W&B is active in Gold 01 (it is disabled across the Silver EDA notebooks); Gold 01 establishes the W&B run context for the Gold layer.

## Input Contract and Lineage

### Silver Inputs

With `USE_PROFILED_SILVER_SUBSETS = True` (the default and expected runtime state), Gold 01 loads three distinct Silver 02a artifacts:

- the profiled Silver dataframe (`*__silver_subsets__profiled_dataframe.parquet`) — base dataset for all Gold transforms;
- the normal-clean subset Parquet;
- the normal-contaminated subset Parquet.

All three must exist; any missing file raises `FileNotFoundError` naming the file and the upstream notebook that produces it. The profiled dataframe is the only one transformed; the two subsets are validated at load time and the normal-clean definition is used through the `machine_status__profiled` column.

### Parent Truth and Registry Resolution

`GOLD_PARENT_TRUTH_HASH = extract_truth_hash(silver_dataframe)` runs before any copy or filter; a `None` result raises `ValueError`. The dataset name is taken from non-empty `meta__dataset` values, then `load_parent_truth_record_from_dataframe(parent_layer_name="silver", ...)` loads the Silver truth JSON. From that record Gold 01 resolves:

- `DATASET_NAME` (overriding the config-derived value);
- `GOLD_PARENT_TRUTH_HASH` re-read from the truth record (this confirmed value is carried into `initialize_layer_truth` and stamped as `meta__parent_truth_hash` in all outputs);
- `PARENT_PIPELINE_MODE`, which when non-None overrides `PIPELINE_MODE` and is written into `meta__pipeline_mode` on the Silver dataframe;
- `FEATURE_REGISTRY_PATH = feature_registry_dir / "registry" / {DATASET_NAME}__silver__feature_registry.json` — resolved from the Silver truth's artifact paths so Gold uses the exact registry Silver published, not a config default;
- `IMPUTE_RECOMMENDATION_PATH = artifacts_root / "silver_eda" / DATASET_NAME / FILENAMES["impute_recommendation_file_name"]`.

`gold_truth` is then initialized with `parent_truth_hash=GOLD_PARENT_TRUTH_HASH` and seeded with `config_snapshot` (gold version, recipe id, train fraction, random seed, scaler kind, Stage 2/3 counts, pipeline mode) and an initial `runtime_facts` section.

### Why Lineage Matters Here

Gold 01 is the root of the Gold truth chain. Every Gold modeling and validation notebook reads one of Gold 01's Parquets and verifies its `meta__truth_hash` against the `GOLD_TRUTH_HASH` recorded in the truth index. If the parent hash, dataset identity, or feature registry were resolved inconsistently at this stage, all downstream models would silently train and evaluate on artifacts whose provenance could not be confirmed. Resolving identity from the Silver truth record (rather than config) is what keeps the Gold feature space pinned to the exact Silver run that produced it.

## Gold Data Preparation Methodology

### Stable Row Identity First

`ensure_stable_row_id(gold_working_dataframe, row_id_column="meta__row_id")` stamps a stable integer identity before any Gold transformation and validates uniqueness. This identity is the anchor that survives the sort-and-restore operations later in the pipeline (imputation reorders rows within groups) and is required to be present in all output frames at the end. Row identity is stamped on a `.copy()` of the Silver dataframe so the original input remains unmutated for lineage checks.

### Chronological Episode-Based Split

`build_episode_based_split_mask(train_fraction=TRAIN_FRACTION, episode_column="meta__episode_id")` builds a non-leaking split:

- episodes are ordered by the minimum of their order column, resolved in priority `time_index → event_step → meta__row_id` (falling back to a synthetic row-order column if none exist);
- earlier episodes become train, later episodes become test;
- each episode belongs entirely to one split;
- the train episode count is `floor(n_episodes * train_fraction)` clamped to `[1, n_episodes - 1]`, so at least one episode is always reserved for test.

The function raises `ValueError` if the episode column is missing, if `train_fraction` is not strictly between 0 and 1, if any episode id is NaN, or if fewer than two episodes exist. Episode-level (rather than row-level) splitting prevents rows from the same fault episode appearing in both train and test, which would leak future behavior into training. `meta__is_train_flag` is then stamped on every row, and the `split_info` dict is written to both the ledger and the truth record.

### Numeric Feature Selection and Conditional Encoding

`select_numeric_feature_columns` filters the registry's `feature_columns` to those present in the dataframe with a numeric dtype. It does not alter the registry — it only restricts to what is usable at Gold. `apply_one_hot_encoding_from_truths` reads `needs_one_hot_encoding` and `one_hot_encoding_columns` from the Silver truth and applies `pd.get_dummies` only when the upstream truth signals it is needed; for the pump synthetic dataset (no categorical features) it passes through unchanged with an empty applied-columns list. This keeps Gold preprocessing dataset-agnostic: a categorical dataset would trigger encoding without any notebook change.

### Train-Only Imputation

`apply_imputation(method="forward_fill_within_group_then_median", train_mask=...)` derives all fill statistics from training rows only. Concretely, for this method it groups by the available identity columns (`meta__asset_id`, and `meta__run_id` when present), orders within group by `event_step` (or `time_index`), forward-fills each numeric feature within group, then fills any remaining NaNs with that feature's median computed from the training-row statistics frame. A per-feature `ValueError` is raised if the training median is itself NaN (the feature is entirely missing in the statistics rows). Original row order is preserved via a temporary `__original_row_order_for_imputation` column that is sorted on and dropped at the end.

Restricting statistics to training rows prevents any test-set signal from leaking into the imputation. The `recommended_imputation` value loaded from `IMPUTE_RECOMMENDATION_PATH` is recorded in the truth record for traceability but does not change the applied method — Gold 01 applies the fixed method deterministically. After imputation, `train_mask_flag` is rebuilt from the stamped `meta__is_train_flag` column because the sort-and-restore can invalidate a mask derived from the prior index.

## Feature, Imputation, and Scaling Logic

### Prescaled Snapshot

`gold_preprocessed_prescaled_dataframe = gold_working_dataframe.copy()` freezes the fully-imputed dataframe in original sensor units before scaling. It is saved and truth-stamped separately. Its purpose is inspection, replay, and a column-symmetry check against the scaled frame: the two frames must carry the same feature columns, and retaining raw magnitudes makes anomalies easier to diagnose by original sensor range.

### Scaler Fit — Normal-Only Training Strategy

`fit_and_apply_scaler` fits the scaler on `train_mask & normal_only_mask` and applies the transform to all rows. `normal_only_mask` is `machine_status__profiled == "normal_clean"`, falling back to `anomaly_flag == 0` if the profiled column is absent, and to train-only if neither exists. If the fit subset is empty, `ValueError` is raised. `make_scaler(SCALER_KIND)` returns `RobustScaler` (default), `StandardScaler`, or `MinMaxScaler`. The fitted scaler is persisted with `joblib.dump` to `GOLD_MODEL_DIR` under a dataset/scaler-kind filename so downstream notebooks can apply identical scaling without re-fitting.

Fitting only on confirmed-normal rows is the central modeling decision of this notebook: it prevents anomalous or contaminated readings from distorting the center and spread statistics that define the feature space all downstream unsupervised models depend on.

### Normal-Only Fit Subset and Reference Profile

`get_training_rows_for_unsupervised_model` extracts `training_rows_for_fit` — the scaled `train ∩ normal_clean` rows. This subset never contains test, recovery, abnormal, or contaminated rows, and it is what the Gold Isolation Forest models fit on downstream.

`build_reference_profile` is computed on the fit subset and produces, per feature: `median_value`, `mean_value`, `standard_deviation`, and `lower_bound`/`upper_bound` set to the 5th and 95th percentiles. This profile is the numeric fingerprint of normal pump behavior; the percentile bounds are intended for Stage 3 bounds-based confirmation in the cascade notebooks.

### Stage 2 and Stage 3 Selection

`stage1_feature_columns` is the full numeric feature list. `choose_stage2_features_from_training_stability(..., target_count=STAGE2_TARGET_FEATURE_COUNT)` ranks features on the fit subset by coefficient of variation, defined as `standard_deviation / max(abs(median), 1e-6)`, sorts ascending (most stable first), and selects the top N. Stage 2 deliberately uses a narrower, more stable feature set than Stage 1 to reduce false-positive sensitivity.

`build_stage3_sensor_groups` takes the Stage 2 features, ranks them by the reference profile's `standard_deviation` ascending, assigns the most stable `STAGE3_PRIMARY_COUNT` to the primary rule set and the next `STAGE3_SECONDARY_COUNT` to the secondary rule set. These small, high-confidence lists drive Stage 3 rule-based confirmation downstream. (A `build_gold_support_artifacts` helper that bundles these steps exists but is not the active path; the notebook calls each step explicitly.)

## Validation and Data Quality Checks

| Check | Location | Behavior |
|---|---|---|
| General context sanity (16 shared vars) | After context load | `NameError` listing any missing variable |
| Gold sanity (`GOLD_CFG`) | After context load | `NameError` if absent |
| SQL smoke check | Before data load | Read-only query confirms DB connectivity |
| All three Silver inputs present | Data load | `FileNotFoundError` naming the missing file and producing notebook |
| Parent truth hash not `None` | After `extract_truth_hash` | `ValueError` if the Silver dataframe has no truth stamp |
| `meta__dataset` usable | After load | `ValueError` if no non-empty values |
| Feature registry exists and is a non-empty dict with `feature_columns` | After path resolution | `FileNotFoundError` / `TypeError` / `ValueError` |
| `meta__row_id` uniqueness | After `ensure_stable_row_id` | Uniqueness validated and logged |
| Episode column present, ≥2 episodes, no NaN episode ids, valid train fraction | Split build | `ValueError` for each failed precondition |
| Imputation training median not NaN | Imputation | Per-feature `ValueError` if a training median is NaN |
| Scaler fit subset non-empty | `fit_and_apply_scaler` | `ValueError` if no normal-only train rows exist |
| `meta__row_id` present in all output frames | Output validation | `ValueError` if stable identity was lost |
| Prescaled vs scaled column symmetry | Final checks | `symmetric_difference` should be empty |
| Lineage columns in all five frames | Final checks | `ValueError` if `meta__truth_hash` / `meta__parent_truth_hash` / `meta__pipeline_mode` missing |
| Truth hash consistent across all five frames | Final checks | `ValueError` if any frame returns a different hash |
| Episode non-overlap (train vs test) | `verify_gold_episode_split` | `ValueError` listing leaking episodes if overlap found |

The split between hard failures (missing inputs, leaking episodes, lost row identity, empty scaler fit) and softer behavior (one-hot pass-through, recommended-imputation recorded but not applied) reflects the notebook's role as a foundation: anything that would corrupt the lineage chain or the modeling feature space fails loudly.

## Artifact and SQL Persistence

### File Artifacts

Consistent with the audit clues (`joblib.dump`, `save_json`, `to_csv`, `to_json`), Gold 01 writes:

- **Five Parquets** via `save_data`: prescaled, scaled (full), train, test, and fit-normal-only. All five are stamped with the same `GOLD_TRUTH_HASH` before saving.
- **Four JSON feature/sensor lists** via `save_json`: Stage 1 features, Stage 2 features, Stage 3 primary sensors, Stage 3 secondary sensors.
- **Reference profile CSV** via `reference_profile.to_csv`.
- **Fitted scaler** via `joblib.dump` to `GOLD_MODEL_DIR`.
- **Preprocessing summary JSON**, **config snapshot JSON**, and the **ledger JSON**.
- **Gold truth record JSON** under `TRUTHS_PATH/gold/` plus a **truth index** entry (`layer_name="gold"`, `truth_stage="gold_preprocessing"`).

Each major save is mirrored to W&B via `wandb.save` (stage artifacts, the five Parquets, and the ledger).

### SQL Persistence

`write_gold_preprocessed_features_sql(engine, capstone_schema=CAPSTONE_SCHEMA, dataset_id=DATASET_ID, run_id=RUN_ID, notebook_globals=globals(), dataset_name=..., feature_set_id=...)` writes Gold preprocessing summary outputs to PostgreSQL, gated by `WRITE_TO_POSTGRES = True`. Setting the gate to `False` skips all DB writes and leaves file artifacts unaffected. The function is passed the notebook globals manifest rather than individual frames; the exact target tables and column structure are defined in `utils/medallion/gold/` and are not fully determinable from the notebook alone. The SQL clue `write_layer_dataframe` appears in imports but is not the direct write path here; `read_sql` corresponds to the smoke check.

## Truth, Audit, and Reproducibility Behavior

The Gold truth record is built incrementally with `update_truth_section` throughout the notebook (config snapshot, runtime facts, imputation info, scaler info, stage feature summary, split summary, source fingerprint, artifact paths) and finalized with `build_truth_record(truth_base=gold_truth, row_count=len(scaled), column_count=scaled.shape[1] + 3, meta_columns=..., feature_columns=...)`. The `+ 3` accounts for the three lineage columns stamped immediately afterward. `build_truth_record` produces `GOLD_TRUTH_HASH`.

`stamp_truth_columns` then writes `meta__truth_hash` (= `GOLD_TRUTH_HASH`), `meta__parent_truth_hash` (= `GOLD_PARENT_TRUTH_HASH`), and `meta__pipeline_mode` into all five output frames. Stamping the same hash into every frame is the key lineage guarantee: any downstream notebook reading any Gold Parquet obtains the same hash, and cross-frame consistency is verifiable with `extract_truth_hash` (which the final checks perform). `save_truth_record` writes to `TRUTHS_PATH/gold/`, and `append_truth_index` registers the record so downstream stages can locate it by layer and stage without knowing the hash in advance.

Reproducibility is reinforced by the `source_fingerprint` section (`build_file_fingerprint` of the Silver input), the config snapshot export, the carried-forward pipeline mode and process run id, the recorded `recommended_imputation`, the persisted scaler path, and the ledger written to disk. This matters before baseline modeling, cascade modeling, and validation because those stages must be able to prove they trained and evaluated on a specific, fingerprinted preprocessing run.

W&B is active throughout: `wandb.init(job_type="gold_preprocessing", config={...})` opens the run, `wandb.save` registers artifacts as they are written, scaler facts are logged via `wandb.log`/`wandb.config.update`, and `wandb_run.finish()` closes the run before the final sanity checks.

## Downstream Technical Handoff

Source-confirmed outputs produced and stamped by Gold 01 for downstream consumption:

- Five Parquets (prescaled, scaled, train, test, fit-normal-only), each carrying `GOLD_TRUTH_HASH`.
- Four JSON feature/sensor lists (Stage 1, Stage 2, Stage 3 primary, Stage 3 secondary).
- Reference profile CSV and the fitted scaler joblib.
- The `gold_preprocessing` truth record (with `GOLD_PARENT_TRUTH_HASH` and artifact path overrides) and its truth index entry.

The workflow reference maps these to specific consumers — Gold 02 (fit/train/test Parquets, Stage 1 features, reference profile, truth hash as its parent), Gold 03a/b/c (additionally Stage 2 features and Stage 3 primary/secondary sensor lists), and Gold 06A/06B (the fitted scaler and reference profile for replay/streaming validation). From Gold 01 source alone, the artifacts, truth record, and artifact-path overrides are confirmed to be produced, stamped, and indexed; the precise file-level read performed by each individual downstream Gold notebook is governed by those notebooks and is **Not determined from available source** here. Notebook order alone is not treated as evidence of direct handoff. The shared `GOLD_TRUTH_HASH` is the lineage anchor that downstream notebooks use to confirm a consistent preprocessing run.

## Key Technical Decisions

| Decision | Source Evidence | Why It Matters | Verification Method |
|---|---|---|---|
| Capture `GOLD_PARENT_TRUTH_HASH` before any subsetting, then confirm from the Silver truth record | Cell 35: `extract_truth_hash` before copy/filter; re-read via `get_truth_hash(silver_truth)` | Pins the Gold lineage to the complete Silver run and to the authoritative truth record, not a partial frame | Confirm the parent hash equals the Silver truth record's hash and matches `meta__parent_truth_hash` in outputs |
| Resolve feature registry and identity from Silver truth, not config | Cell 35 comment: "use the exact registry that Silver published, not a config default" | Keeps the Gold feature space pinned to the exact Silver run; prevents config drift | Confirm `FEATURE_REGISTRY_PATH` derives from the Silver truth `feature_registry_dir` |
| Stamp a stable `meta__row_id` before any transform | Cell 46 `ensure_stable_row_id` | Provides an identity anchor that survives imputation's sort-and-restore and is required in all outputs | Confirm `meta__row_id` is unique and present in all five frames |
| Episode-level chronological split with ≥1 test episode reserved | Cell 52 `build_episode_based_split_mask` | Prevents rows from one fault episode leaking across train/test; ordering by time avoids future leakage | Run `verify_gold_episode_split`; confirm zero train/test episode overlap |
| Imputation statistics from training rows only | Cell 74/76: stats frame restricted to `train_mask` | Stops test-set signal from leaking into fill values | Confirm imputation medians are computed only over `meta__is_train_flag == True` rows |
| Forward-fill within asset/run group then median fallback | Cell 74: group by `meta__asset_id`/`meta__run_id`, order by `event_step`/`time_index`, ffill, then train-median fillna | Respects per-asset time order before falling back to a global training statistic | Inspect `imputation_info` for grouping/ordering columns and applied method |
| Rebuild train mask after imputation | Cell 80 | Imputation's sort-and-restore can invalidate an index-derived mask | Confirm `train_mask_flag` is rebuilt from the stamped `meta__is_train_flag` column |
| Freeze a prescaled snapshot before scaling | Cell 83 copy | Enables replay, raw-magnitude diagnosis, and a column-symmetry check | Confirm prescaled and scaled frames share identical feature columns |
| Fit scaler only on `train ∩ normal_clean` rows | Cells 88/90; `normal_only_mask = machine_status__profiled == "normal_clean"` (fallback `anomaly_flag == 0`) | Prevents anomalous readings from distorting the center/spread that define the feature space for all models | Confirm the ledger `fit_source` is "train ∩ normal-only" and fit row count > 0 |
| Persist the fitted scaler via joblib | Cell 88 `joblib.dump` to `GOLD_MODEL_DIR` | Lets inference and validation apply identical scaling without re-fitting | Confirm the scaler file exists under `GOLD_MODEL_DIR` |
| Build the normal-only fit subset distinct from the train split | Cells 94/96 `get_training_rows_for_unsupervised_model` | Isolation Forest must see only normal behavior; excludes test/recovery/abnormal/contaminated rows | Confirm `gold_fit_dataframe` contains only `normal_clean` train rows |
| Reference profile bounds set to 5th/95th percentiles | Cell 99 `quantile(0.05)` / `quantile(0.95)` | Provides robust per-sensor envelopes for Stage 3 bounds-based confirmation | Recompute bounds on the fit subset and compare to the CSV |
| Stage 2 selection by coefficient of variation (`std / max(\|median\|, 1e-6)`) | Cell 104 ranking ascending, top N | Selects the most stable features to cut false-positive sensitivity in cascade Stage 2 | Confirm Stage 2 features are the lowest-CV subset of Stage 1 |
| Stage 3 grouping by reference standard deviation ascending | Cell 109 | Assigns the most stable sensors to high-confidence rule sets | Confirm primary sensors have the lowest reference SD among Stage 2 features |
| Stamp the same `GOLD_TRUTH_HASH` into all five frames | Cell 129 `stamp_truth_columns` × 5 | Any downstream notebook can verify it read artifacts from one preprocessing run | `extract_truth_hash` returns the same hash for all five Parquets |
| SQL via `write_gold_preprocessed_features_sql` behind `WRITE_TO_POSTGRES` | Cell 155 | Durable summary persistence with an offline-run escape hatch | Confirm the gate controls the write and file artifacts are unaffected when `False` |
| W&B active in Gold 01 | Cells 26/88/116/131 `wandb.init`/`save`/`finish` | First active tracking run in the chain; establishes Gold experiment context | Confirm a W&B run with `job_type="gold_preprocessing"` is created and finished |

## Failure Modes and Guardrails

| Failure Condition | Behavior | Guardrail |
|---|---|---|
| Any of the three Silver inputs missing | `FileNotFoundError` naming file and producing notebook | Explicit existence check under `USE_PROFILED_SILVER_SUBSETS` |
| Silver dataframe lacks a readable `meta__truth_hash` | `ValueError` | Checked immediately after `extract_truth_hash` |
| `meta__dataset` empty/missing | `ValueError` | Non-empty value check before parent truth load |
| Feature registry missing, non-dict, or empty `feature_columns` | `FileNotFoundError` / `TypeError` / `ValueError` | Path existence plus `require_dict`/`require_list` validation |
| Episode column missing, NaN episode ids, <2 episodes, or invalid train fraction | `ValueError` (specific message per case) | Preconditions enforced in `build_episode_based_split_mask` |
| A training median is NaN during imputation | Per-feature `ValueError` | Median computed from training stats and checked before fillna |
| No normal-only train rows for the scaler | `ValueError` | Empty fit-subset check in `fit_and_apply_scaler` |
| `meta__row_id` lost during transforms | `ValueError` | Presence validated in all output frames |
| Prescaled vs scaled columns differ | Surfaced by `symmetric_difference` check | Final column-symmetry verification |
| Lineage columns missing or hashes inconsistent across frames | `ValueError` | Final lineage verification with `extract_truth_hash` |
| Train/test episode overlap | `ValueError` listing leaking episodes | `verify_gold_episode_split` run on final frames |
| `WRITE_TO_POSTGRES = False` | SQL write skipped; file artifacts unaffected | Explicit boolean gate |
| W&B unconfigured at runtime | W&B calls require valid project/entity | `wandb.init` config and `wandb.run is not None` guards around logging |

## Verification Checklist

- Active notebook path is `notebooks/experiments/EDA_Notebook_Pump_Gold_01_PreProcessing.ipynb`
- Silver profiled, normal-clean, and normal-contaminated Parquets all exist
- `meta__truth_hash` present in the Silver profiled dataframe; parent hash resolves from the Silver truth record
- `DATASET_NAME`, `FEATURE_REGISTRY_PATH`, `IMPUTE_RECOMMENDATION_PATH`, and `PARENT_PIPELINE_MODE` resolve from Silver truth
- `meta__row_id` is unique and present in all five output frames
- Train/test split is episode-level with at least one test episode and zero overlap
- `meta__is_train_flag` is stamped; `train_mask_flag` is rebuilt after imputation
- Imputation statistics are computed only from training rows
- Prescaled and scaled frames carry identical feature columns
- Scaler is fit on `train ∩ normal_clean` rows and persisted under `GOLD_MODEL_DIR`
- `gold_fit_dataframe` contains only `normal_clean` train rows
- Reference profile CSV exists with median/mean/std and 5th/95th-percentile bounds
- Stage 1/2/3 JSON lists exist; Stage 2 is the lowest-CV subset; Stage 3 primary has the lowest reference SD
- All five Parquets carry the same `GOLD_TRUTH_HASH`, plus `meta__parent_truth_hash` and `meta__pipeline_mode`
- Gold truth record exists under `TRUTHS_PATH/gold/` with `truth_stage="gold_preprocessing"` and a truth index entry
- If `WRITE_TO_POSTGRES = True`: `write_gold_preprocessed_features_sql` completes and returns a summary
- A W&B run with `job_type="gold_preprocessing"` is created and finished

## Source-Limited Items

- The exact PostgreSQL target tables and column structure written by `write_gold_preprocessed_features_sql` are Not determined from Gold 01 source (the function is called, not defined, in the notebook).
- The precise file-level reads performed by each downstream Gold notebook (Gold 02, 03a/b/c, 04, 05, 06A, 06B) against Gold 01's artifacts are Not determined from Gold 01 source; only the production, stamping, and indexing of those artifacts are confirmed here.
- Whether a direct file-level dependency on Silver 02b exists is Not determined from available source; Gold 01 resolves its inputs from Silver 02a outputs and the Silver truth record.
- The internal definitions of `ensure_stable_row_id`, `stamp_truth_columns`, `build_truth_record`, and `save_data` (in the utility modules) are outside this notebook; their observable effects are documented, but their full implementations are Not determined from Gold 01 source.
