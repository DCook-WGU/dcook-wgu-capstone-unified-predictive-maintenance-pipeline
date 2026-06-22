# Notebook Code Reference: EDA_Notebook_Pump_Gold_01_PreProcessing

**Source:** `notebooks/experiments/EDA_Notebook_Pump_Gold_01_PreProcessing.ipynb`
**Stage:** Gold Preprocessing (`gold_preprocessing`)
**Cells:** 157 (62 code, 95 markdown), 47 inventory sections

---

## Notebook Purpose

Gold_01 converts the Silver analytical dataset into modeling-ready features, establishes the train/test/fit-normal split structure, and produces the stage-level artifacts consumed by every subsequent Gold notebook. It is the entry point to the Gold layer — nothing downstream runs without the outputs it produces.

The notebook's core contribution is a pipeline of irreversible transformations applied in a fixed order: stable row identity → episode-based chronological split → numeric feature selection → conditional one-hot encoding → train-only imputation → prescaled snapshot → normal-only scaler fit → scaling → normal-only fit subset → reference profile → Stage 2 feature ranking → Stage 3 sensor grouping. Each step is recorded in the truth record, the ledger, and where applicable in W&B, so every downstream notebook can reconstruct exactly which rows and features it received and under what conditions they were preprocessed.

This notebook also produces the Gold truth record (`gold_preprocessing` stage), which is the first Gold-layer entry in the truth chain. The same `GOLD_TRUTH_HASH` is stamped into all five output Parquets so downstream notebooks can verify they're reading artifacts from the same preprocessing run.

---

## Pipeline Role

| Attribute | Value |
|---|---|
| Stage | Gold Preprocessing (`gold_preprocessing`) |
| Position | First Gold-layer notebook; immediately downstream of Silver_02a |
| Primary input | Silver_02a profiled dataframe (`*__silver_subsets__profiled_dataframe.parquet`) |
| Supporting inputs | Normal-clean subset Parquet, normal-contaminated subset Parquet, feature registry JSON, imputation recommendation JSON |
| Downstream outputs | Five Gold Parquets (scaled, prescaled, train, test, fit-normal); stage feature lists (JSON); reference profile (CSV); fitted scaler (joblib); Gold truth record |
| Truth chain | Parent hash = `meta__truth_hash` extracted from Silver profiled dataframe before any filtering; creates `gold_preprocessing` truth record; stamps all five output frames with `GOLD_TRUTH_HASH` |
| W&B | ACTIVE — `wandb.init` runs with `job_type="gold_preprocessing"`; `wandb.save` uploads all stage and output artifacts; `wandb_run.finish()` closes the run |
| SQL write function | `write_gold_preprocessed_features_sql` — gated by `WRITE_TO_POSTGRES = True` |

---

## Inputs

| Input | Source | Form | Used For |
|---|---|---|---|
| Silver profiled dataframe | `SILVER_PROFILED_DATAFRAME_DATA_PATH` (Silver_02a output) | Parquet | Base dataset for all Gold transforms |
| Normal-clean subset | `SILVER_NORMAL_CLEAN_DATA_PATH` (Silver_02a output) | Parquet | Validated at load time; used in scaler fit strategy |
| Normal-contaminated subset | `SILVER_NORMAL_CONTAMINATED_DATA_PATH` (Silver_02a output) | Parquet | Validated at load time |
| Feature registry JSON | `FEATURE_REGISTRY_PATH` (resolved from Silver truth record) | JSON dict | `FEATURE_COLUMNS` list for feature selection |
| Imputation recommendation JSON | `IMPUTE_RECOMMENDATION_PATH` — constructed as `silver_eda_artifacts_dir / FILENAMES["impute_recommendation_file_name"]`, where `silver_eda_artifacts_dir = artifacts_root / "silver_eda" / DATASET_NAME`; `DATASET_NAME` is resolved from Silver truth in cell 35 | JSON dict | `recommended_imputation` field written to truth record |
| Silver truth record | `TRUTHS_PATH/silver/` via `load_parent_truth_record_from_dataframe` | JSON | `needs_one_hot_encoding`, `one_hot_encoding_columns`; parent hash verification |
| Project config | `load_notebook_context(stage="gold_preprocessing", dataset="pump")` | YAML → `CTX` | All runtime constants, stage config, paths |
| Environment variables | OS environment | Strings | DB engine, `CAPSTONE_SCHEMA`, `DATASET_ID` / `RUN_ID` |

**Load dependency guard (cell 33):** If `USE_PROFILED_SILVER_SUBSETS = True` (the default), all three Silver subset files must exist or the notebook raises `FileNotFoundError` immediately, naming which file is missing and which notebook must be run first.

---

## Configuration and Runtime Context

| Item | Source | Purpose |
|---|---|---|
| `GOLD_CFG` | `CTX` | Gold-specific stage config block; required by Gold-specific sanity check |
| `GOLD_PROCESS_RUN_ID` | `make_process_run_id(...)` | Unique process run ID written into truth record |
| `TRAIN_FRACTION` | Config / `GOLD_CFG` | Fraction of episodes assigned to train split (chronological) |
| `SCALER_KIND` | Config / `GOLD_CFG` (default `"robust"`) | Which scaler to fit: `"robust"`, `"standard"`, or `"minmax"` |
| `STAGE2_TARGET_FEATURE_COUNT` | Config / `GOLD_CFG` | Target number of features selected for Stage 2 modeling |
| `STAGE3_PRIMARY_COUNT` / `STAGE3_SECONDARY_COUNT` | Config / `GOLD_CFG` | Number of sensors assigned to Stage 3 primary and secondary rule sets |
| `USE_PROFILED_SILVER_SUBSETS` | Notebook cell (default `True`) | Whether to load Silver_02a profiled dataframe (required for `machine_status__profiled` column) |
| `WRITE_TO_POSTGRES` | SQL write cell (default `True`) | Controls `write_gold_preprocessed_features_sql`; allows offline runs |
| `DATASET_NAME` | Set from config as `DATASET_CFG["name"]` in cell 9; then overridden by `get_dataset_name_from_truth(silver_truth)` in cell 35 — the effective runtime value comes from Silver truth | Dataset identity written into truth record, artifact paths, and SQL rows |
| `RUN_ID` / `DATASET_ID` | Resolved from env → config → fallback | Identity fields written into SQL rows and truth record |
| `PARENT_PIPELINE_MODE` | Read from Silver truth via `get_pipeline_mode_from_truth(silver_truth)` in cell 35; overrides `PIPELINE_MODE` if non-None; the resolved value is propagated into `meta__pipeline_mode` on `silver_dataframe` and carried into the Gold truth record | Ensures Gold inherits the pipeline execution mode from the Silver run |
| `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_RUN_NAME` | `WANDB_CFG` via `CTX` | W&B run parameters |

---

## Logical Workflow Map

1. Import all libraries including `wandb`, `sklearn`, `joblib`; define local helpers (`require_dict`, `require_list`, `cfg_require_mapping`)
2. `load_notebook_context(stage="gold_preprocessing", ...)` → `CTX`, `GOLD_CFG`, `logger`, `ledger`, all pipeline constants; `TRUTH_CONFIG` built from config + PIPELINE block
3. Sanity checks: general context vars + Gold-specific `GOLD_CFG`; `NameError` on any missing variable
4. `build_artifact_dirs_from_config(stage_key="gold_preprocessing")` → `GOLD_ARTIFACTS_PATH`, `GOLD_FEATURE_DIR`, `GOLD_PROFILE_DIR`, `GOLD_MODEL_DIR`, `GOLD_SUMMARY_DIR`, `GOLD_METADATA_DIR`, `GOLD_CONFIG_DIR`, `GOLD_LINEAGE_DIR`; `export_config_snapshot`
5. SQL engine via `get_engine_from_env()`; resolve `DATASET_ID` / `RUN_ID` / `ASSET_ID` via `first_non_empty_string`; SQL smoke check displaying schema/table list
6. `log_layer_paths(paths, current_layer="gold")`; `ledger.add` checkpoint
7. `wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME, job_type="gold_preprocessing", config={...})` → `wandb_run` (W&B run is ACTIVE throughout this notebook)
8. `ledger.add` (Gold ledger checkpoint); original `Ledger(...)` setup preserved in triple-quoted string (not executed)
9. Load Silver profiled dataframe via `load_data` from `SILVER_PROFILED_DATAFRAME_DATA_PATH`; load normal_clean, normal_contaminated subsets; `FileNotFoundError` raised if any missing
10. Parent truth resolution and Gold truth initialization (before `gold_working_dataframe` is created):
    - `GOLD_PARENT_TRUTH_HASH = extract_truth_hash(silver_dataframe)` — initial value from the dataframe column, before any filtering
    - `load_parent_truth_record_from_dataframe(dataframe=silver_dataframe, ...)` → `silver_truth` (first load, from original Silver input)
    - `DATASET_NAME = get_dataset_name_from_truth(silver_truth)` — overrides the config-derived value
    - `GOLD_PARENT_TRUTH_HASH = get_truth_hash(silver_truth)` — updated from the loaded truth JSON; this is the value carried into the Gold truth record
    - `PARENT_PIPELINE_MODE = get_pipeline_mode_from_truth(silver_truth)` — if non-None, overrides `PIPELINE_MODE` and stamps `meta__pipeline_mode` on `silver_dataframe`
    - `FEATURE_REGISTRY_PATH` constructed via `get_artifact_path_from_truth(silver_truth, "feature_registry_dir")` — resolved directly from Silver truth's `artifact_paths`
    - `IMPUTE_RECOMMENDATION_PATH` constructed as `silver_eda_artifacts_dir / FILENAMES[...]` where dir = `artifacts_root / "silver_eda" / DATASET_NAME`
    - `gold_truth = initialize_layer_truth(parent_truth_hash=GOLD_PARENT_TRUTH_HASH)` + initial `config_snapshot` and `runtime_facts` sections written
11. Load feature registry from `FEATURE_REGISTRY_PATH`; load imputation recommendation JSON from `IMPUTE_RECOMMENDATION_PATH`; validate `feature_columns` list
12. Validate required Silver support artifact paths (covered in step 11 above)
13. `dataframe = silver_dataframe.copy()` — protects original from mutation; `GOLD_PROCESSED_AT_UTC` timestamped; initial `runtime_facts` written to `gold_truth`; `ledger.add`
14. Display profiled Silver split inputs (if `USE_PROFILED_SILVER_SUBSETS`)
15. `ensure_stable_row_id(gold_working_dataframe, row_id_column="meta__row_id")` — stamps a stable integer row identity before any Gold transformations; validates uniqueness and null count; `ledger.add`; `update_truth_section` with row tracking info
16. `load_parent_truth_record_from_dataframe(dataframe=gold_working_dataframe, ...)` → `silver_truth` (second load; now works from the working copy post-row-id stamping)
17. Define `build_episode_based_split_mask` — chronological episode-level split; earlier episodes → train, later episodes → test; each episode belongs entirely to one split; at least one episode reserved for test
18. Display Gold working dataframe columns for review
19. Build split: resolve `split_order_column` (prefers `time_index` → `event_step` → `meta__row_id`); `build_episode_based_split_mask(train_fraction=TRAIN_FRACTION, episode_column="meta__episode_id")` → `train_mask`, `split_info`; `ledger.add`
20. Define `stamp_training_metadata` — stamps `meta__is_train_flag` (bool) from `train_mask`; apply: `gold_working_dataframe = stamp_training_metadata(...)`; write `split_info` to `gold_truth`; `ledger.add`
21. Define and apply `select_numeric_feature_columns` → `numeric_feature_columns` (filters `feature_columns` to those present in dataframe and detected as numeric dtype)
22. Define and apply `apply_one_hot_encoding_from_truths` — reads `needs_one_hot_encoding` and `one_hot_encoding_columns` from `silver_truth`; applies `pd.get_dummies` only when upstream truth signals it's needed; truth fields mirrored into `gold_truth`; `ledger.add`
23. Define `apply_imputation` (method: `"forward_fill_within_group_then_median"` — forward fill within episode group by time order, backfill, then median fallback from train rows); apply: stats derived **exclusively from train rows** (`meta__is_train_flag == True`) to prevent test-set leakage; `update_truth_section` with `imputation_info` and `recommended_imputation`; `ledger.add`
24. Rebuild `train_mask_flag` from stamped `meta__is_train_flag` column (index may have shifted during imputation)
25. `gold_preprocessed_prescaled_dataframe = gold_working_dataframe.copy()` — snapshot in original sensor units before scaling; `ledger.add`
26. Define `make_scaler(kind)` factory → returns `RobustScaler` (default), `StandardScaler`, or `MinMaxScaler`
27. Define `fit_and_apply_scaler` — fits on `train ∩ normal_clean` rows (`train_mask & normal_only_mask`), applies transform to all rows; saves fitted scaler to `GOLD_MODEL_DIR` via `joblib.dump`; returns scaled dataframe + scaler path
28. Apply scaling: `normal_only_mask = (machine_status__profiled == "normal_clean")` (with fallback to `anomaly_flag == 0` if profiled column absent); `fit_and_apply_scaler(...)` → `gold_preprocessed_scaled_dataframe`, `scaler_path`; `update_truth_section(runtime_facts, scaler_kind/scaler_path)`; `ledger.add`
29. Define and apply `get_training_rows_for_unsupervised_model` — filters `gold_preprocessed_scaled_dataframe` to `train ∩ normal_clean` rows → `training_rows_for_fit`; `ledger.add`
30. Define and apply `build_reference_profile` — per-sensor median, mean, std, lower_bound, upper_bound computed on `training_rows_for_fit`; returns a DataFrame with one row per sensor → `reference_profile`; `ledger.add`
31. Assign `stage1_feature_columns = list(numeric_feature_columns)`; define and apply `choose_stage2_features_from_training_stability` — ranks Stage 1 features by coefficient of variation on `training_rows_for_fit`, selects top `STAGE2_TARGET_FEATURE_COUNT` with lowest relative variability → `stage2_feature_columns`; assign `STAGE2_FEATURE_COLUMNS`; `ledger.add`
32. Define and apply `build_stage3_sensor_groups` — filters `reference_profile` to Stage 2 features, ranks by `standard_deviation` ascending, assigns top `STAGE3_PRIMARY_COUNT` to primary and next `STAGE3_SECONDARY_COUNT` to secondary rule sets → `stage3_primary_rule_sensors`, `stage3_secondary_rule_sensors`; `ledger.add`
33. Define `build_gold_support_artifacts` helper (combines steps 29–32 in a single call; not invoked directly in the notebook — each step is called individually above)
34. **Save stage-level artifacts:** `save_json` × 4 + `reference_profile.to_csv`; `wandb.save` × 5; `update_truth_section` with feature set summary; `ledger.add`
35. Build final split dataframes: derive `meta__split` column from `meta__is_train_flag`; `gold_train_dataframe`, `gold_test_dataframe`, `gold_fit_dataframe = training_rows_for_fit.copy()`; `meta__split` values stamped on each
36. Define `verify_gold_episode_split` — verifies zero episode overlap between train and test; raises `ValueError` listing leaking episodes if overlap found; apply: `gold_split_summary` built; display
37. **Finalize truth record:** `update_truth_section(runtime_facts, source_run_ids/split_summary)`; `update_truth_section(source_fingerprint, build_file_fingerprint(silver_path))`; `update_truth_section(artifact_paths, all output paths)`; `ledger.add`
38. Validate `meta__row_id` is present in all four output frames; `ledger.add`
39. `identify_meta_columns` + `identify_feature_columns`; `build_truth_record(truth_base=gold_truth, row_count, column_count, meta_columns, feature_columns)` → `gold_truth_record`, `GOLD_TRUTH_HASH`
40. `stamp_truth_columns(...)` × 5 — stamps `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` into all five output frames with the same `GOLD_TRUTH_HASH`
41. **Save all output Parquets:** `save_data` × 5 (prescaled, scaled, train, test, fit); `wandb.save` × 5; `ledger.add`
42. Build and save preprocessing summary JSON; `save_json`; `save_truth_record`; `append_truth_index`; `ledger.add`
43. `ledger.write_json(gold_preprocesssing_ledger_path)`; `wandb.save(str(gold_preprocesssing_ledger_path))`; `wandb_run.finish()` — W&B run closed
44. **Final sanity checks:** assert all key variables in locals; check prescaled vs scaled column symmetry; verify lineage columns in all 5 frames; extract and cross-check truth hashes; `verify_gold_episode_split` called again on final frames
45. `write_gold_preprocessed_features_sql(engine, capstone_schema, dataset_id, run_id, ...)` gated by `WRITE_TO_POSTGRES = True` → `gold_preprocessing_sql_summary_dataframe`; displayed

---

## Section Overview

| Section | Key Outputs / Side Effects |
|---|---|
| Imports and setup | All imports; `require_dict`, `require_list`, `cfg_require_mapping` helpers defined |
| Load paths, config, runtime settings | `CTX`, `GOLD_CFG`, all pipeline constants, `logger`, `ledger`, `TRUTH_CONFIG`, `GOLD_PROCESS_RUN_ID`; sanity checks |
| Gold preprocessing artifact directories | Directory tree created; `export_config_snapshot` |
| SQL runtime context | `engine`; `DATASET_ID`, `RUN_ID`, `ASSET_ID` resolved; SQL smoke check |
| Start logging | `log_layer_paths`; `ledger.add` |
| Initialize W&B run | `wandb_run = wandb.init(...)` with full config dict |
| Initialize Gold ledger | `ledger.add` checkpoint (initialized by `load_notebook_context`; not re-created) |
| Load Silver input and supporting artifacts | `silver_dataframe`, `silver_normal_clean`, `silver_normal_contaminated` loaded; `FileNotFoundError` if any missing |
| Resolve parent truth and confirm dataset identity | `silver_truth` first load (from `silver_dataframe`); `DATASET_NAME` overridden from Silver truth; `GOLD_PARENT_TRUTH_HASH` updated from truth JSON; `PARENT_PIPELINE_MODE` inherited; `FEATURE_REGISTRY_PATH` and `IMPUTE_RECOMMENDATION_PATH` constructed; `gold_truth` initialized with `config_snapshot` and initial `runtime_facts` |
| Validate required Silver support artifacts | Feature registry and imputation recommendation validated; `feature_columns` list resolved |
| Create working Gold dataframe and start runtime tracking | `gold_working_dataframe = silver_dataframe.copy()`; `GOLD_PROCESSED_AT_UTC`; `gold_truth` runtime_facts updated |
| Review profiled Silver split inputs | Shape / state count display |
| Stamp stable Gold row identity | `ensure_stable_row_id` → `meta__row_id`; uniqueness verified; `ledger.add` |
| Reload parent truth record | `silver_truth` reloaded from working copy |
| Define episode-based train/test split logic | `build_episode_based_split_mask` defined |
| Build train/test split mask | `train_mask`, `split_info`; `ledger.add` |
| Define training metadata stamp helper | `stamp_training_metadata` defined |
| Stamp train/test metadata | `meta__is_train_flag` stamped; `split_info` written to `gold_truth`; `ledger.add` |
| Define numeric feature selection logic | `select_numeric_feature_columns` defined |
| Select numeric feature set | `numeric_feature_columns`; `ledger.add` |
| Define one-hot encoding logic | `apply_one_hot_encoding_from_truths` defined |
| Apply one-hot encoding | `gold_working_dataframe` updated; `applied_one_hot_encoding_columns`; truth fields mirrored |
| Define imputation logic | `apply_imputation` defined (train-only stats, forward-fill within episode group) |
| Apply numeric feature imputation | `gold_working_dataframe` imputed; `imputation_info`, `recommended_imputation` in truth |
| Rebuild training mask after imputation | `train_mask_flag` rebuilt from stamped column; `ledger.add` |
| Freeze prescaled copy | `gold_preprocessed_prescaled_dataframe` snapshot; `ledger.add` |
| Define scaler factory | `make_scaler(kind)` defined |
| Define scaling workflow | `fit_and_apply_scaler` defined (fits on normal-only train rows; saves joblib) |
| Scale Gold feature set | `gold_preprocessed_scaled_dataframe`, `scaler_path`; truth updated; `ledger.add` |
| Define normal-only fit subset logic | `get_training_rows_for_unsupervised_model` defined |
| Build normal-only fit subset | `training_rows_for_fit` (normal_clean train rows only); `ledger.add` |
| Define reference profile logic | `build_reference_profile` defined |
| Build normal reference profile | `reference_profile` (per-sensor median/mean/std/bounds); `ledger.add` |
| Define Stage 2 feature ranking logic | `choose_stage2_features_from_training_stability` defined |
| Choose Stage 2 feature set | `stage2_feature_columns`, `STAGE2_FEATURE_COLUMNS`; `ledger.add` |
| Define Stage 3 sensor grouping logic | `build_stage3_sensor_groups` defined |
| Build Stage 3 sensor groups | `stage3_primary_rule_sensors`, `stage3_secondary_rule_sensors`; `ledger.add` |
| Define Gold support artifact builder | `build_gold_support_artifacts` helper defined (integrates reference profile + stage feature selection; not invoked directly) |
| Save stage-level Gold artifacts | JSON × 4 + CSV × 1 + `wandb.save` × 5; feature set summary in truth |
| Create final Gold split dataframes | `gold_train_dataframe`, `gold_test_dataframe`, `gold_fit_dataframe`; `meta__split` stamped; `verify_gold_episode_split` called |
| Finalize Gold truth record | `update_truth_section` × 3; `build_file_fingerprint`; `build_truth_record` → `GOLD_TRUTH_HASH`; `stamp_truth_columns` × 5 |
| Save final Gold preprocessing outputs | `save_data` × 5 (prescaled, scaled, train, test, fit) + `wandb.save` × 5; `ledger.add` |
| Save pre-scaled Gold feature outputs | Recorded in `ledger.add` as traceability step |
| Save preprocessing summary and metadata | `preprocessing_summary` JSON; `save_truth_record`; `append_truth_index`; `ledger.add` |
| Save ledger and close tracking run | `ledger.write_json`; `wandb.save`; `wandb_run.finish()` |
| Final sanity checks | Assert variables; verify prescaled vs scaled column symmetry; verify lineage columns in all 5 frames; cross-check truth hashes |
| Compare prescaled and scaled column structures | `symmetric_difference` check displayed |
| Verify final lineage columns | Required columns confirmed in all 5 output frames; `extract_truth_hash` cross-check |
| Gold preprocessing SQL write | `write_gold_preprocessed_features_sql` → SQL write; `WRITE_TO_POSTGRES` gate |

---

## Section Details

### Context Load and Sanity Checks

`load_notebook_context(stage="gold_preprocessing", dataset="pump", mode="train", profile="default")` sets the stage to `gold_preprocessing`, distinct from the Silver notebooks' `silver_eda`. The `GOLD_CFG` block is the stage-specific config for Gold.

Two sanity check layers run before any data is loaded:
1. **General context check:** 16 required variables — `CTX`, `paths`, `CONFIG`, `CONFIG_MAP`, `STAGE_CFG`, `RESOLVED_PATHS`, `FILENAMES`, `VERSIONS_CFG`, `RUNTIME_CFG`, `DATASET_CFG`, `WANDB_CFG`, `EXECUTION_CFG`, `PIPELINE`, `logger`, `ledger`, `LOG_PATH` — must all be in globals; `NameError` lists any missing
2. **Gold-specific check:** `GOLD_CFG` must be in globals; `NameError` lists if missing

`TRUTH_CONFIG` is built from `build_truth_config_block(CONFIG)` and extended with the PIPELINE block so that Gold truth records carry execution-mode metadata.

---

### W&B Initialization

W&B is **ACTIVE** in Gold_01, unlike the Silver EDA notebooks. `wandb.init(...)` runs in cell 26 with:
- `project`, `entity`, `name` from `WANDB_CFG`
- `job_type="gold_preprocessing"`
- Full config dict: gold version, dataset, stage, train fraction, Silver/feature-registry/gold output paths, scaler kind, Stage 2 and Stage 3 counts

`wandb.save(...)` is called after each major artifact save (stage features, reference profile, Parquets, ledger). `wandb_run.finish()` is called in cell 136 after the ledger is written and before the final sanity checks run.

---

### Silver Input Load and Truth Hash Capture

`USE_PROFILED_SILVER_SUBSETS = True` (cell 33) is the default and the expected runtime state. Under this flag, the notebook loads three distinct Silver artifacts:
- Profiled Silver dataframe (from Silver_02a)
- Normal-clean subset (from Silver_02a)
- Normal-contaminated subset (from Silver_02a)

All three must exist. `FileNotFoundError` is raised for any missing file, naming which notebook must be run to produce it.

`GOLD_PARENT_TRUTH_HASH` is resolved in two steps, both in cell 35:

1. **Initial value:** `extract_truth_hash(silver_dataframe)` reads `meta__truth_hash` from the loaded Silver Parquet column, before any copy or filter. This captures the hash embedded in the upstream file.
2. **Confirmed value:** `load_parent_truth_record_from_dataframe(dataframe=silver_dataframe, ...)` loads `silver_truth` (the Silver truth JSON from `TRUTHS_PATH/silver/`), then `GOLD_PARENT_TRUTH_HASH = get_truth_hash(silver_truth)` updates the hash from the truth record itself. This is the value carried into `initialize_layer_truth` and written as `meta__parent_truth_hash` in all five output frames.

`ValueError` is raised if the initial column read returns None. The truth record load also resolves `DATASET_NAME`, `FEATURE_REGISTRY_PATH`, `IMPUTE_RECOMMENDATION_PATH`, and `PARENT_PIPELINE_MODE` from the same cell before `gold_truth` is initialized.

---

### Chronological Episode-Based Train/Test Split

`build_episode_based_split_mask` creates a non-leaking split at the episode level:
- Episodes are sorted by their first row's time/order column (not by episode ID integer)
- Earlier episodes → train; later episodes → test
- Each episode belongs entirely to one split — no episode is partially in train and partially in test
- At least one episode is always reserved for test regardless of `TRAIN_FRACTION`

The `split_order_column` resolution: `time_index` → `event_step` → `meta__row_id`. If the first candidate exists in the dataframe, it is used for chronological ordering.

After applying the split, `meta__is_train_flag` is stamped onto every row. The ledger records the exact `split_info` dict (episode counts, row counts per split) and the truth record also receives it under `runtime_facts`.

`train_mask_flag` is rebuilt from the stamped `meta__is_train_flag` column after imputation (cell 80) because the index may have shifted. This pattern is essential: the imputation step sorts rows within episodes and then restores original order, which can invalidate a boolean mask derived from a prior index.

---

### Feature Selection, Encoding, and Imputation

**Feature selection:** `select_numeric_feature_columns` filters `feature_columns` (from Silver truth's feature registry) to those present in the dataframe that `pd.api.types.is_numeric_dtype` returns `True` for. This is a simple but safe filter — it doesn't alter the feature registry, only restricts to what is usable at Gold.

**One-hot encoding:** `apply_one_hot_encoding_from_truths` reads `needs_one_hot_encoding` (bool) and `one_hot_encoding_columns` (list) from `silver_truth`. If `needs_one_hot_encoding` is `False` or the list is empty, no encoding is applied. This design makes Gold preprocessing dataset-agnostic — a dataset with categorical features would trigger encoding without changing the notebook.

**Imputation:** `apply_imputation(method="forward_fill_within_group_then_median", train_mask=train_mask_for_stats)`:
1. Computes fill statistics (median fallback values) from `train_mask_for_stats` rows only — test rows are excluded to prevent test-set leakage into imputation statistics
2. Within each episode group, applies forward fill sorted by time order, then backfill
3. Applies median fallback where forward/backfill leaves NaNs (e.g., single-row episodes or leading NaNs)
4. Preserves original row order via a temporary `__original_row_order_for_imputation` column

The `recommended_imputation` field (loaded from `IMPUTE_RECOMMENDATION_PATH`) is recorded in the truth record for traceability but does not change the applied method — the notebook applies the fixed method.

---

### Prescaled Snapshot

Before scaling runs, `gold_preprocessed_prescaled_dataframe = gold_working_dataframe.copy()` snapshots the fully-imputed, fully-cleaned dataframe in original sensor units. This artifact is saved and truth-stamped separately. Its purpose is inspection, replay, and comparison:
- Feature columns in prescaled and scaled frames should be the same list
- The prescaled frame retains raw sensor ranges, making it easier to diagnose anomalies by original magnitude
- The final sanity check compares column structures between prescaled and scaled frames

---

### Scaler Fit — Normal-Only Train Strategy

`fit_and_apply_scaler` fits the scaler on `train ∩ normal_clean` rows (the intersection of the train mask and the `machine_status__profiled == "normal_clean"` mask). Fallback to `train ∩ (anomaly_flag == 0)` if the profiled column is absent. This strategy prevents anomalous sensor readings from distorting the center and spread statistics that define the feature space for all downstream models.

The fitted scaler is saved via `joblib.dump` to `GOLD_MODEL_DIR`. Downstream notebooks that need to apply the same scaling to new data can load this file directly without re-fitting.

After scaling, `normal_only_mask` is also used to extract `training_rows_for_fit` — the subset that will be passed to Isolation Forest during model training in Gold_02 and the Cascade notebooks. This subset never includes test rows, recovery rows, abnormal rows, or contaminated rows.

---

### Reference Profile and Stage Feature Selection

**Reference profile (`build_reference_profile`):** Computed on `training_rows_for_fit` (normal_clean train rows). Per sensor columns:
- `median_value`, `mean_value`, `standard_deviation` — central tendency and spread
- `lower_bound`, `upper_bound` — used by Stage 3 rule-based confirmation logic

This profile is the numeric "fingerprint" of normal pump behavior. Gold Cascade notebooks read this CSV to implement Stage 3 profile-based anomaly confirmation.

**Stage 2 feature ranking (`choose_stage2_features_from_training_stability`):** Computes coefficient of variation (std / mean, or fallback if mean ≈ 0) on `training_rows_for_fit` per feature. The top `STAGE2_TARGET_FEATURE_COUNT` features with the lowest relative variability are selected. Features excluded by the `min_non_null_ratio` or `min_variance` threshold are dropped before ranking. Stage 2 uses a narrower feature set than Stage 1 to reduce false-positive sensitivity.

**Stage 3 sensor grouping (`build_stage3_sensor_groups`):** From the Stage 2 feature list, ranks by `standard_deviation` in the reference profile (ascending). Assigns the most stable sensors to the primary rule set and the next group to the secondary rule set. These small, high-confidence lists drive Stage 3 rule-based confirmation in the Cascade notebooks.

---

### Gold Truth Record and Output Stamping

Gold's truth record is initialized via `update_truth_section(gold_truth, ...)` incrementally throughout the notebook, then finalized in cell 129:

1. `build_truth_record(truth_base=gold_truth, row_count, column_count, meta_columns, feature_columns)` → `gold_truth_record`, `GOLD_TRUTH_HASH`
2. `stamp_truth_columns(gold_preprocessed_prescaled_dataframe, ...)` × 5 — the **same** `GOLD_TRUTH_HASH` is written into `meta__truth_hash` in all five output frames

The stamp-all-frames pattern is the key lineage guarantee: any downstream notebook that reads any of the five Gold Parquets gets the same hash. Cross-frame consistency can be verified with `extract_truth_hash(frame)` on any output, which the final sanity checks do.

`save_truth_record` → `TRUTHS_PATH/gold/`; `append_truth_index` → truth index updated with `layer_name="gold"` and `truth_stage="gold_preprocessing"`.

---

### Final Sanity Checks

Cells 139–153 run a five-part check after all saves are complete:

1. **Variable existence:** `assert "gold_preprocessed_scaled_dataframe" in locals()` etc. — confirms the full pipeline ran
2. **Shape display:** `gold_preprocessed_prescaled_dataframe.shape` and `gold_preprocessed_scaled_dataframe.shape` displayed for manual inspection
3. **Column structure comparison (cell 149):** `symmetric_difference(prescaled_columns, scaled_columns)` — the sets should be identical (same column names, same order); differences indicate an encoding or scaling step added or dropped columns unexpectedly
4. **Lineage column verification (cell 152):** `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` must be present in all five frames; `extract_truth_hash` must return a non-None matching hash in each; `ValueError` raised if any check fails
5. **Episode split verification:** `verify_gold_episode_split` confirms zero episode overlap between train and test frames, and that the union of train and test episodes equals the full episode set

---

## Key Function Calls and In-Place Usage

| Function | Section | Return / Side Effect |
|---|---|---|
| `load_notebook_context(stage="gold_preprocessing", ...)` | Context load | `CTX`; `GOLD_CFG`, `logger`, `ledger` |
| `export_config_snapshot(...)` | Artifact dirs | Config JSON in `GOLD_CONFIG_DIR` |
| `get_engine_from_env()` | SQL Runtime Context | `engine` |
| `read_sql_dataframe(engine, SELECT ...)` | SQL smoke check | `sql_smoke_check_dataframe` |
| `log_layer_paths(paths, current_layer="gold")` | Logging setup | Layer path log written |
| `wandb.init(project, entity, name, job_type, config)` | W&B init | `wandb_run` (ACTIVE) |
| `load_data(profiled_silver_path.parent, name)` | Load Silver input | `silver_dataframe` |
| `extract_truth_hash(silver_dataframe)` | Resolve parent truth | `GOLD_PARENT_TRUTH_HASH` (before any filtering) |
| `initialize_layer_truth(...)` | Context setup | `gold_truth` dict |
| `load_parent_truth_record_from_dataframe(...)` | Reload parent truth | `silver_truth` |
| `load_json(FEATURE_REGISTRY_PATH)` | Validate artifacts | `feature_registry_raw` |
| `ensure_stable_row_id(gold_working_dataframe, "meta__row_id")` | Stamp row identity | `gold_working_dataframe` with stable `meta__row_id` |
| `build_episode_based_split_mask(df, train_fraction, ...)` | Split | `(train_mask, split_info)` — chronological, episode-level |
| `stamp_training_metadata(df, train_mask)` | Stamp split | `meta__is_train_flag` on every row |
| `select_numeric_feature_columns(df, feature_columns)` | Feature selection | `numeric_feature_columns` |
| `apply_one_hot_encoding_from_truths(df, silver_truth, ...)` | Encoding | `(gold_working_dataframe, applied_one_hot_encoding_columns)` |
| `apply_imputation(df, method, train_mask)` | Imputation | `(gold_working_dataframe, imputation_info)` |
| `make_scaler(kind)` | Scaler factory | `RobustScaler` / `StandardScaler` / `MinMaxScaler` |
| `fit_and_apply_scaler(df, feature_cols, train_mask, normal_only_mask, ...)` | Scaling | `(gold_preprocessed_scaled_dataframe, scaler_path)` |
| `get_training_rows_for_unsupervised_model(df, train_mask)` | Fit subset | `training_rows_for_fit` (normal_clean train rows) |
| `build_reference_profile(training_rows_for_fit, feature_cols)` | Reference profile | `reference_profile` DataFrame (median/mean/std/bounds per sensor) |
| `choose_stage2_features_from_training_stability(df, feature_cols, target_count)` | Stage 2 selection | `stage2_feature_columns` (lowest CV, top-N) |
| `build_stage3_sensor_groups(reference_profile, stage2_features, primary_count, secondary_count)` | Stage 3 grouping | `(stage3_primary_rule_sensors, stage3_secondary_rule_sensors)` |
| `save_json(list, path)` × 4 | Stage artifacts | JSON files for stage1/2/3 feature lists |
| `reference_profile.to_csv(REFERENCE_PROFILE_PATH)` | Stage artifacts | Reference profile CSV |
| `wandb.save(str(path))` × multiple | W&B logging | Artifact files registered in W&B run |
| `verify_gold_episode_split(full, train, test, fit)` | Sanity check | Split verification report; `ValueError` on episode overlap |
| `update_truth_section(gold_truth, section, data)` × multiple | Truth build | Incrementally populates `gold_truth` dict |
| `build_file_fingerprint(silver_path)` | Truth build | Source file fingerprint for `source_fingerprint` section |
| `build_truth_record(truth_base, row_count, ...)` | Finalization | `gold_truth_record`, `GOLD_TRUTH_HASH` |
| `stamp_truth_columns(df, truth_record, ...)` × 5 | Finalization | `meta__truth_hash`, `meta__parent_truth_hash`, `meta__pipeline_mode` stamped in all 5 frames |
| `save_data(df, parent_dir, filename)` × 5 | Save outputs | Five Gold Parquets saved |
| `save_truth_record(...)` | Finalization | JSON at `TRUTHS_PATH/gold/` |
| `append_truth_index(...)` | Finalization | Truth index updated |
| `ledger.write_json(path)` | Close | Ledger JSON written |
| `wandb_run.finish()` | Close | W&B run closed |
| `write_gold_preprocessed_features_sql(engine, capstone_schema, ...)` | SQL write | Rows in Gold SQL table(s) |

---

## Outputs and Artifacts

| Output | Type | Location | Downstream Consumer |
|---|---|---|---|
| Gold prescaled Parquet | Parquet | `GOLD_PRESCALED_DATA_PATH` | Inspection; replay; column structure check |
| Gold scaled Parquet (full) | Parquet | `GOLD_SCALED_DATA_PATH` | Gold_02 baseline modeling; Gold_03* cascade |
| Gold train Parquet | Parquet | `GOLD_TRAIN_DATA_PATH` | Model training in Gold_02, Gold_03* |
| Gold test Parquet | Parquet | `GOLD_TEST_DATA_PATH` | Model evaluation in Gold_02, Gold_03* |
| Gold fit-normal-only Parquet | Parquet | `GOLD_FIT_DATA_PATH` | Isolation Forest fit in Gold_02, Gold_03* |
| Fitted scaler | joblib | `GOLD_MODEL_DIR / {DATASET_NAME}__scaler.joblib` | Replay scaling in inference; Gold_06 validation |
| Stage 1 feature list | JSON | `STAGE1_FEATURES_PATH` | Gold_02, Gold_03* baseline features |
| Stage 2 feature list | JSON | `STAGE2_FEATURES_PATH` | Gold_03* cascade Stage 2 reduced feature set |
| Stage 3 primary sensor list | JSON | `STAGE3_PRIMARY_PATH` | Gold_03* cascade Stage 3 rule confirmation |
| Stage 3 secondary sensor list | JSON | `STAGE3_SECONDARY_PATH` | Gold_03* cascade Stage 3 rule confirmation |
| Normal reference profile | CSV | `REFERENCE_PROFILE_PATH` | Gold_03* Stage 3 bounds-based confirmation |
| Gold preprocessing summary | JSON | `GOLD_SUMMARY_DIR` | Lineage; audit |
| Config snapshot | JSON | `GOLD_CONFIG_DIR` | Reproducibility |
| Gold truth record | JSON | `TRUTHS_PATH/gold/{DATASET_NAME}__gold__truth__{hash}.json` | Downstream truth chain; lineage verification |
| Truth index entry | Appended JSONL | `TRUTH_INDEX_PATH` | Cross-run truth lookup |
| Ledger JSON | JSON | `gold_preprocesssing_ledger_path` | Audit trail |
| SQL rows | PostgreSQL | `gold.preprocessed_features` (and related tables via `write_gold_preprocessed_features_sql`) | Operational monitoring |
| W&B artifacts | W&B run | Project run | Experiment tracking dashboard |

---

## Data Quality / Validation Behavior

| Check | Where | Failure / Risk Prevented |
|---|---|---|
| General context sanity check (16 vars) | After `load_notebook_context` | `NameError` if any required variable is missing |
| Gold-specific context check (`GOLD_CFG`) | After general check | `NameError` if Gold stage config is absent |
| SQL smoke check | Before data load | Catches DB unavailability early |
| Profiled Silver, normal-clean, normal-contaminated all exist | Data load | `FileNotFoundError` naming which Silver notebook must be run |
| `GOLD_PARENT_TRUTH_HASH` not None | After `extract_truth_hash` | `ValueError` if profiled dataframe has no truth stamp |
| Feature registry exists at path | After path resolution | `FileNotFoundError` if registry file missing |
| Feature registry is non-empty dict | After `load_json` | `require_dict` / `require_list` raise `TypeError` / `ValueError` |
| `meta__row_id` uniqueness | After `ensure_stable_row_id` | Logs `row_id_unique=False` if IDs collide |
| `meta__is_train_flag` present before imputation | Imputation | Implicit requirement for train-only stats |
| `train_mask_flag` rebuilt after imputation | After imputation | Prevents index mismatch from episode sort-and-restore |
| Scaler fit on non-empty normal-only rows | `fit_and_apply_scaler` | `ValueError` if no normal rows exist in train split |
| `meta__row_id` in all 4 output frames | Output validation | `ValueError` if stable row identity was lost during transforms |
| Prescaled vs scaled column symmetry | Scaling | Reveals unexpected column additions or drops during scaling |
| Lineage columns in all 5 frames | Final lineage check | `ValueError` if `meta__truth_hash` / `meta__parent_truth_hash` / `meta__pipeline_mode` missing |
| Truth hash consistent across all 5 frames | Final lineage check | `ValueError` if any frame returns a different hash |
| Episode non-overlap between train and test | `verify_gold_episode_split` | `ValueError` listing episode IDs if leakage detected |

---

## Decision Tag Notes

| Tag | Source |
|---|---|
| `ARTIFACT_WRITE` | Five Parquets + five JSONs/CSVs + scaler joblib + ledger + truth record |
| `BOUNDS_CLIPPING` | `lower_bound` / `upper_bound` in reference profile; used downstream for Stage 3 bounds confirmation |
| `DATA_VALIDATION` | Feature registry validation; train/test split verification; lineage column checks |
| `LEDGER_UPDATE` | `ledger.add` at every major step |
| `MODEL_EVALUATION` | Stage 2 CV-based stability ranking; Stage 3 SD ranking; both use training data statistics |
| `MODEL_TRAINING` | Scaler fit on normal-only train rows via `sklearn`; `joblib.dump` persists fitted scaler |
| `SQL_READ` | SQL smoke check (`SELECT ... FROM information_schema.tables`) |
| `SQL_WRITE` | `write_gold_preprocessed_features_sql` |
| `TRUTH_METADATA` | Gold truth record initialized, populated, built, and saved; truth hash stamped onto all outputs |
| `VARIANCE_CONTROL` | Stage 2 ranking by coefficient of variation; features with near-zero variance excluded |
| `WANDB_LOGGING` | `wandb.init` + `wandb.save` × many + `wandb_run.finish()` — fully active W&B run |

---

## Downstream Handoff

**Gold_02_Baseline_Modeling** reads:
- `GOLD_FIT_DATA_PATH` (fit-normal-only Parquet) → trains Isolation Forest
- `GOLD_TRAIN_DATA_PATH`, `GOLD_TEST_DATA_PATH` → evaluation split
- `STAGE1_FEATURES_PATH` → baseline Stage 1 feature set
- `REFERENCE_PROFILE_PATH` → sensor bounds for Stage 3 comparison
- Gold truth record → `GOLD_TRUTH_HASH` as parent hash for its own truth record

**Gold_03a/b/c Cascade Modeling** additionally reads:
- `STAGE2_FEATURES_PATH` → reduced Stage 2 feature set for cascade Stage 2
- `STAGE3_PRIMARY_PATH` / `STAGE3_SECONDARY_PATH` → rule sensors for cascade Stage 3

**Gold_06A/06B Validation** reads:
- The fitted scaler (`GOLD_MODEL_DIR / *__scaler.joblib`) to apply consistent scaling to replay/streaming data
- The reference profile for threshold-based comparison

All downstream notebooks verify their input by reading `meta__truth_hash` from whatever Gold Parquet they load and confirming it matches the expected `GOLD_TRUTH_HASH` recorded in the truth index. This pattern ensures no downstream notebook silently reads from a different preprocessing run.

---

## Relationship to Other Notebooks

### Upstream Context

Gold_01_PreProcessing reads Silver_02a's clean analytical Parquet and imputation recommendation JSON as its primary pipeline inputs. Silver_02b's EDA context may inform feature selection decisions, but a direct file-level dependency on Silver_02b is not confirmed from available source. No dependency on Gold_02, Gold_03x, or any cascade notebooks.

### Downstream Handoff

Gold_01 provides to Gold_02, Gold_03a, Gold_03b, and Gold_03c:
- 5 Parquet artifacts (scaled, preprocessed, fit, test, train) resolved via truth-record path overrides
- 4 JSON feature/sensor lists (Stage 1 features, Stage 2 features, Stage 3 primary sensors, Stage 3 secondary sensors)
- `gold_preprocessing` truth record containing `GOLD_PARENT_TRUTH_HASH` and 8 artifact path overrides

`GOLD_PARENT_TRUTH_HASH` extracted from this truth record is the shared lineage anchor cross-validated across all Gold modeling notebooks in Gold_04.

### Pipeline Position

The single preprocessing foundation for all Gold modeling. All Gold_02 through Gold_03c notebooks are rooted on Gold_01's outputs via the truth-record path override mechanism. W&B is first active here; Silver notebooks disable it. No Gold modeling notebook re-runs Gold_01's preprocessing steps.

### Relationship Summary

- Reads Silver_02a clean Parquet and imputation recommendation JSON as primary inputs
- Produces 5 Parquets + 4 JSON lists + truth record consumed by Gold_02 and all three Gold_03 cascade variants
- `GOLD_PARENT_TRUTH_HASH` is the shared lineage anchor cross-validated by Gold_04 across all four modeling notebooks
- W&B is first active here; establishes the W&B run context for the Gold layer
- Direct downstream consumers: Gold_02; Gold_03a; Gold_03b; Gold_03c

---

## Notes / Risks / Deferred Cleanup

- **`build_gold_support_artifacts` is defined but not called as the primary execution path.** The notebook calls `get_training_rows_for_unsupervised_model`, `build_reference_profile`, `choose_stage2_features_from_training_stability`, and `build_stage3_sensor_groups` individually and explicitly. The `build_gold_support_artifacts` helper (cell 111) is available as an alternative entry point for pipeline-mode callers, but it is not the active code path.
- **`STAGE2_FEATURE_COLUMNS` assignment:** After `choose_stage2_features_from_training_stability` returns `stage2_feature_columns`, the notebook also assigns `STAGE2_FEATURE_COLUMNS = stage2_feature_columns`. The uppercase constant is used in `build_stage3_sensor_groups` and in JSON saves. Both names refer to the same list.
- **`USE_PROFILED_SILVER_SUBSETS = True` is required for `machine_status__profiled`.** The scaler, normal-only fit subset, and reference profile all depend on `machine_status__profiled == "normal_clean"` as the primary mask. The fallback to `anomaly_flag == 0` exists but is not the expected runtime path.
- **W&B is fully active here; Silver EDA notebooks disable it.** Gold_01 is the first notebook in the active chain where `wandb.init` actually runs. Ensure `WANDB_PROJECT` and `WANDB_ENTITY` are configured before running offline.
- **`ledger.write_json` variable name has a typo:** `gold_preprocesssing_ledger_path` (three `s`). The variable resolves correctly because it's defined earlier with the same spelling; the typo is consistent and should not be fixed in isolation.
- **`apply_one_hot_encoding_from_truths` passes through silently when encoding is not needed.** The Silver pump synthetic dataset does not have categorical columns requiring encoding. The function returns the original dataframe unchanged and an empty `applied_one_hot_encoding_columns` list. This is correct and expected behavior.
- **`write_gold_preprocessed_features_sql` table target:** The SQL target appears to be `gold.preprocessed_features` and/or related Gold schema tables based on the function signature; exact table names should be confirmed from the function definition in `utils/medallion/gold/`.
- **Scaler fit requires at least one normal_clean train row.** If the split produces zero normal_clean train rows (pathological dataset), `fit_and_apply_scaler` raises `ValueError`. This is an expected guard, not a silent failure.
