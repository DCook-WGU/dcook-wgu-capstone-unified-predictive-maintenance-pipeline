# Silver 02b Deep Technical Reference

## Purpose of This Deep Reference

This document covers the technical decisions in Silver 02b (Profiled-State EDA and Generator Input Bundle) that require deeper explanation than the workflow reference provides. The workflow reference describes what each section of the notebook does. This document explains why Silver 02b reads the Silver 02a profiled dataframe rather than re-deriving profiled states, why the parent truth hash is captured before any subsetting, why the feature set is inherited from the Silver parent truth record rather than reconstructed from config, why the generator input bundle uses a fixed "rich profile" column contract, why correlation artifacts are computed separately per profiled state, why two independent sensor-grouping methods are produced, why the IsolationForest outlier audit is treated as diagnostic rather than a production model, why a third Silver-layer truth record is created with `truth_stage="eda_profile"`, why the Silver 01 missingness quarantine payload is forwarded into this truth record, and why SQL persistence uses `write_silver_eda_sql_outputs` rather than the `log_silver_eda_sql` function used by Silver 01 and Silver 02a.

## Technical Scope

- Shared `silver_eda` stage context and two-level sanity check (general + `SILVER_EDA_CFG`)
- Profiled Silver Parquet consumption with canonical-path-then-recursive-glob fallback
- Parent truth hash capture before any filtering; dataset and registry resolution from parent truth
- Feature column inheritance from the Silver feature registry with dataframe intersection
- Profiled-state masks and copy-protected per-state subsets
- Episode-level requirement (`meta__episode_id`) for state summaries
- Per-state correlation artifacts and conditional computation guards (rows > 1)
- Two sensor-grouping methods: connected components at a correlation threshold and agglomerative clustering
- Generator input bundle: rich per-state feature profiles, dropped-sensor registry, fault pairings, group map, episode status counts, and consolidated manifest
- Distribution-family inference and robust bound construction inside the generator profile builder
- Dropped-sensor handling distinction: generate-then-mask versus never-generate
- Diagnostic-only PCA, imputation comparison, and IsolationForest outlier audit
- `silver_eda_profile_truth` record creation, missingness quarantine passthrough, and indexing
- W&B disabled via an evaluated boolean toggle (`USE_EXPERIMENT_TRACKING = False`)
- SQL persistence through `write_silver_eda_sql_outputs` to seven `silver.eda_*` tables, behind a write gate

## Source Grounding

Sources used:

- `notebooks/eda/EDA_Notebook_Pump_Silver_02b_EDA_v2.ipynb` (active notebook — source of truth)
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Silver_02b_EDA_v2_code_reference.md` (read-only context)
- `notebook_inventory.json`
- `artifact_io_manifest.json`
- `sql_touchpoints.json`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Silver_01_PreEDA_deep_technical_reference.md` (read-only upstream context)
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3_deep_technical_reference.md` (read-only upstream context)
- `technical_reference/00_project_manual/` relationship maps (read-only context)

The active Silver 02b notebook source is the source of truth for all function behavior, variable names, output paths, and design decisions documented here.

## Stage Role in the Medallion Pipeline

Silver 02b is the fourth notebook in the active chain (Bronze 01 → Silver 01 → Silver 02a → Silver 02b). Its role is analytical characterization and export, not data transformation. Silver 02a produced the profiled-state labels (`normal_clean`, `normal_contaminated`, `abnormal`, `recovery`) on the Silver dataframe. Silver 02b reads that profiled dataframe and characterizes sensor behavior across those states, then writes a structured bundle of reusable artifacts.

Silver 02b performs four source-confirmed kinds of work:

1. **Profiled-state EDA** — per-state correlation matrices, missingness tables, numeric summaries, state transition/dwell tables, distribution and timeline plots, aligned anomaly-onset windows.
2. **Generator input construction** — a fixed-schema bundle of per-state "rich" feature profiles, a dropped-sensor registry, fault-pairing tables, sensor group maps, and episode status counts consolidated by a manifest.
3. **Diagnostic modeling context** — PCA variance diagnostics, an imputation-strategy comparison, and an IsolationForest outlier audit trained only on `normal_clean` rows.
4. **Truth and SQL persistence** — a third Silver-layer truth record (`truth_stage="eda_profile"`), a summary JSON, and a write to seven `silver.eda_*` PostgreSQL tables.

Silver 02b does not modify the profiled dataframe or its truth stamps. It reads, characterizes, and exports. It does not stamp `meta__truth_hash` into any dataframe rows — its truth record documents the EDA artifact bundle, not a transformed dataset.

## Input Contract and Lineage

### Profiled Silver Parquet (Primary Input)

The primary input is the Silver 02a profiled dataframe. The canonical path is `SILVER_TRAIN_DATA_PATH / {DATASET_NAME_CONFIG}__silver_subsets__profiled_dataframe.parquet`. If that exact file is absent, the notebook recursively globs two fallback directories (`SILVER_EDA_OUTPUT_DIR`, `SILVER_EDA_ARTIFACT_DIR`) for `*__silver_subsets__profiled_dataframe.parquet`, sorts the resolved candidates, and uses the first with a `logger.warning`. If no candidate exists anywhere, `FileNotFoundError` is raised listing the primary path and fallback directories. The dataframe is read with `pd.read_parquet(PROFILED_DF_PATH, engine="auto")` directly.

### Parent Truth Resolution

`SILVER_TRUTH_HASH = extract_truth_hash(silver_eda_dataframe)` runs immediately after the load, before any subsetting or filtering. If the hash is `None`, `ValueError` halts execution. Capturing the hash on the full unmodified dataframe ensures the lineage link in the profile truth record references the complete Silver output rather than a derived slice — the same discipline used in Silver 02a.

`SILVER_DATASET_NAME` is extracted from non-empty `meta__dataset` values (`ValueError` if none are usable). `load_parent_truth_record_from_dataframe(parent_layer_name="silver", ...)` then loads the Silver truth JSON, from which Silver 02b reads `DATASET_NAME`, `SILVER_TRUTH_HASH` (re-resolved from the record), `SILVER_PARENT_TRUTH_HASH` (Bronze's hash), `PIPELINE_MODE` (when present), `LABEL_SOURCE_COLUMN`/`LABEL_SOURCE_TYPE`, one-hot-encoding hints, and the `feature_registry_dir` artifact path.

### Feature Registry Inheritance

`FEATURE_REGISTRY_PATH` is constructed as `Path(feature_registry_dir) / "registry" / {DATASET_NAME}__silver__feature_registry.json`. If the parent truth did not resolve a registry directory, `ValueError` is raised before any load attempt. The registry is loaded, required to be a dict with a non-empty `feature_columns` list, and intersected with the dataframe columns. Columns in the registry but absent from the dataframe are logged (`logger.warning`) and skipped; an empty intersection raises `ValueError`. The ledger entry records the rationale: "Silver EDA should inherit resolved feature metadata from Silver rather than rebuilding it from config." This keeps the EDA and generator exports on the exact feature lineage finalized by Silver 01.

### Identity and Config

`load_notebook_context(stage="silver_eda", dataset="pump", mode="train", profile="default")` provides `CTX`, `SILVER_EDA_CFG`, `DEFAULT_FALLBACKS_CFG`, `logger`, and `ledger`. `DATASET_ID` / `RUN_ID` / `ASSET_ID` are resolved from environment → config → `DEFAULT_FALLBACKS_CFG`. A SQL engine is established and a read-only smoke check (`read_sql_dataframe`) confirms connectivity before any data work.

### Why Lineage Matters at This Stage

Silver 02b is the last Silver-layer notebook before the synthetic generator and Gold preprocessing consume Silver outputs. Its truth record is the navigation key that lets downstream stages locate every EDA artifact and recover the upstream missingness audit. If the parent truth hash, dataset identity, or feature set were resolved inconsistently here, the generator bundle and the Gold-facing truth chain would describe a different feature space than the one Silver actually produced.

## Silver Data Preparation Methodology

Silver 02b does not clean or reshape the Silver dataframe; it indexes into it. Preparation is limited to establishing the analysis surface:

- **State columns and masks.** `STATE_COL_SOURCE = "machine_status__synthetic"` and `STATE_COL_PROFILED = "machine_status__profiled"` are required; their absence raises `KeyError`. A source-normal mask plus four profiled-state masks (`normal_clean`, `normal_contaminated`, `abnormal`, `recovery`) are built with a consistent `astype(str).str.lower().str.strip().eq(...)` pattern so casing or whitespace variants do not silently drop rows.
- **Copy-protected subsets.** `PROFILED_STATE_SUBSETS` is a dict of `.copy()` dataframes per profiled state. The explicit copy prevents per-state analysis from mutating `silver_eda_dataframe`; all downstream per-state operations work from these copies or from direct mask indexing.
- **Numeric coercion at point of use.** Correlation and profile builders apply `pd.to_numeric(..., errors="coerce")` to feature columns before computing statistics, so non-numeric contamination becomes `NaN` rather than raising inside `.corr()` or `.quantile()`.
- **Episode requirement.** Episode-level summaries require `meta__episode_id`; its absence raises `KeyError`. Episode counts are built by grouping on episode id crossed with source and profiled state.

These choices reflect the notebook's read-only stance: the profiled dataframe is treated as an immutable contract from Silver 02a, and preparation is confined to safe indexing and coercion.

## Feature Engineering and Gold-Readiness Logic

Silver 02b does not engineer model-input features for Gold; it characterizes the existing feature set and produces the synthetic generator contract. The source-confirmed construction logic is:

### Per-State Rich Feature Profiles (Generator Contract)

`build_rich_feature_profile` produces, for each of `normal_clean`, `abnormal`, and `recovery`, a per-sensor row with a fixed column set required by `utils.synthetic.generator.synthetic_profiles`: `sensor, state_scope, mean, std, min, max, median, iqr, p01, p05, p25, p50, p75, p95, p99, skewness, kurtosis, robust_std, distribution_family, lower_bound, upper_bound`. The schema is fixed because the generator reads these columns directly — any change to the column set is a breaking change to the generator contract.

Within the builder:
- `robust_std = iqr / 1.349` when IQR > 0 (an IQR-to-sigma conversion), falling back to `max(std, 1e-6)` otherwise. This gives the generator a scale estimate that resists outliers.
- `lower_bound`/`upper_bound` are robust bounds: `median ± 4 * robust_std`, then clamped within the empirical `[p01, p99]` and hard `[min, max]` ranges, with a final guard that resets to `[min, max]` if the bounds invert. This bounds generated values to a plausible per-state envelope without letting a few extreme rows widen it indefinitely.
- `distribution_family` is assigned by `infer_distribution_family` from std/IQR/skewness/kurtosis: `near_constant`, `bounded_near_constant`, `right_skewed`, `left_skewed`, `robust_empirical`, or `bounded_normal`. The generator uses the family to choose a sampling strategy per sensor.
- Recovery is profiled alongside normal and abnormal because the generator must reproduce post-fault behavior, not only normal and fault states.

The builder raises `ValueError` if the `normal_clean`, `abnormal`, or `recovery` source dataframe is empty — the generator cannot reproduce a state it has no profile for.

### Fault Pairing Table

`fault_pairings_generator_df` is derived from the normal-clean vs abnormal correlation delta. Coupling strength is taken from `abs_correlation_delta_abnormal_vs_clean` (falling back to `abs_correlation_abnormal`), filled to 0.05 and clipped to `[0.05, 0.95]`, then the table is filtered to feature-column pairs, deduplicated, sorted by strength, and capped at the top 150 pairs. Fixed fields `lag_cycles = 0` and `recommended_secondary_fault = "variance_burst"` are attached. The cap and clip keep the generator's coupling model bounded and finite.

### Dropped-Sensor Registry

The registry is built from the PreEDA feature registry's `feature_info.missingness_quarantine` block (`dropped_features`, `dropped_missing_pct`, `drop_reasons`), restricted to raw physical sensors matching `^sensor_\d{2}$`. A critical distinction is encoded: sensors whose drop reason is `all_null` are excluded from profiling (they should be restored downstream as `NaN`, never generated), while other dropped sensors remain profile-eligible (generated, then missingness-masked). The notebook comments name the concrete cases (`sensor_15` all-null → not generated; `sensor_50` → generate then mask). This tells the generator how to treat quarantined sensors so the synthetic dataset preserves the original missingness structure.

### Sensor Grouping (Two Methods)

1. **Connected components** (`SUBSYSTEM_CORR_THRESHOLD = 0.80`): an adjacency graph is built over the absolute normal-clean correlation matrix; sensor pairs at or above the threshold become neighbors; a visited-set DFS extracts components, each labeled `subsystem_NN` and sized. Output: `sensor_group_map_normal_clean.csv`.
2. **Agglomerative clustering** (`FEATURE_CLUSTER_COUNT`): the normal-clean correlation matrix is converted to a distance matrix and clustered with `AgglomerativeClustering`. Output: `feature_cluster_map.csv`.

The two methods are produced together because they are orthogonal: connected components depends on the 0.80 threshold and yields variable group counts; agglomerative clustering yields a fixed cluster count independent of any threshold. Both groupings are surfaced to the generator and recorded in the truth record's `artifact_paths`.

### Correlation Artifacts

Correlation matrices and upper-triangle pair tables are computed for `normal_clean` (always), and for `normal_contaminated` and `abnormal` only when the subset has more than one row. The abnormal matrix is merged against the normal-clean pairs to build a fault-pairing delta table identifying sensor pairs whose coupling changes most between normal and fault states.

## Silver Validation and Data Quality Checks

| Check | Location | Behavior |
|---|---|---|
| General context sanity (16 shared vars) | After context load | `NameError` listing any missing variable |
| Silver-EDA sanity (`SILVER_EDA_CFG`) | After context load | `NameError` if absent |
| SQL smoke check | Before data load | Read-only query confirms DB connectivity |
| Profiled Parquet present | Load with recursive fallback | `FileNotFoundError` if no candidate anywhere |
| `SILVER_TRUTH_HASH` not `None` | Immediately after load | `ValueError` if profiled dataframe has no truth stamp |
| `meta__dataset` usable | After load | `ValueError` if no non-empty values |
| `FEATURE_REGISTRY_PATH` not `None` | Before registry load | `ValueError` |
| Feature registry is a dict with non-empty `feature_columns` | After load | `ValueError` / `TypeError` |
| Feature columns intersect dataframe | After registry load | `logger.warning` for missing; `ValueError` if intersection empty |
| Required state columns present | State masks | `KeyError` if `machine_status__synthetic` or `machine_status__profiled` absent |
| `meta__episode_id` present | Episode summaries | `KeyError` |
| Correlation computed only when rows > 1 | Contaminated / abnormal correlation | Subset skipped with a printed note; empty dataframe assigned |
| Generator source states non-empty | Rich profile build | `ValueError` if `normal_clean` / `abnormal` / `recovery` is empty |
| `sensor_group_map_df` / `fault_pairings_df` present and non-empty | Generator export | `ValueError` / `TypeError` |
| Manifest required variables defined | Manifest update | `NameError` listing missing variables |
| Manifest path existence | Manifest update | Soft `WARNING` print listing non-existent paths; does not raise |
| Truth dir / truth index present and non-empty | QA section | `FileNotFoundError` |
| Truth index required columns | QA section | `KeyError` if `layer_name` / `truth_stage` / `truth_path` / `truth_hash` absent |

The split between hard failures (missing inputs, empty required states, missing truth structure) and soft warnings (missing registry columns, missing manifest paths) is deliberate: structural prerequisites for the generator contract and lineage chain fail loudly, while incidental gaps degrade coverage with a logged warning.

## Artifact and SQL Persistence

### File Artifacts

All file artifacts are written with direct `to_csv`, `to_json`, `json.dump`, or `plt.savefig` calls (consistent with the audit clues `save_json`, `to_csv`, `to_json`). The major outputs:

- **Generator inputs** (`GENERATOR_INPUT_DIR`): `feature_profile_normal_clean.csv`, `feature_profile_abnormal.csv`, `feature_profile_recovery.csv`; dropped-sensor `dropped_sensor_registry.json` and per-state dropped feature profile CSVs; `sensor_group_map_normal_clean.csv`; `sensor_fault_pairings_normal.csv`; `sensor_correlation_hotspot_clusters_normal_clean.json` (intentionally an empty `{"clusters": []}` placeholder); `episode_status_counts.json`; and the consolidated `generator_input_manifest.json`.
- **Correlation artifacts** (`CORRELATION_ARTIFACT_DIR`): normal-clean correlation matrix and pair table; conditional contaminated matrix and clean-vs-contaminated delta table; conditional abnormal matrix and fault-pairing table; connected-components group map; agglomerative feature cluster map; hotspot/transition/dwell summaries.
- **Diagnostic artifacts**: PCA explained variance and loadings CSVs; imputation comparison CSV; outlier summary CSV; distribution, timeline, and aligned-onset PNGs.
- **Summary and truth**: `{DATASET_NAME}__silver__eda_profile__summary.json` in `SILVER_EDA_SUMMARY_DIR`; the `silver_eda_profile_truth` JSON under `TRUTHS_PATH/silver/`; a truth index entry; and a config snapshot.

The generator input manifest is the single index the generator reads. It is rebuilt (or reloaded from disk if not in memory), updated with every artifact path, validated for path existence at write time, and re-written. The manifest stores only path strings and does not re-validate them at read time.

### SQL Persistence

SQL writes go through `write_silver_eda_sql_outputs(engine, dataset_id, run_id, notebook_name, profile_df, feature_statistics_df, missingness_summary_df, correlation_pairs_df, outlier_summary_df, categorical_distribution_df, artifact_index_df)`, gated by `WRITE_SILVER_EDA_SQL_OUTPUTS = True`. It writes seven `silver.eda_*` tables: `eda_dataset_profile`, `eda_feature_statistics`, `eda_missingness_summary`, `eda_correlation_pairs`, `eda_outlier_summary`, `eda_categorical_distribution`, and `eda_artifact_index`. Setting the gate to `False` skips all DB writes and leaves file artifacts unaffected. This function is distinct from `log_silver_eda_sql` (used by Silver 01 and Silver 02a, which targets pipeline-metadata tables); the two write different schemas and are not interchangeable. The SQL clue `write_layer_dataframe` appears in imports but is not the direct write path here.

## Truth, Audit, and Reproducibility Behavior

Silver 02b creates a third Silver-layer truth record, `silver_eda_profile_truth`, via `initialize_layer_truth(... parent_truth_hash=SILVER_TRUTH_HASH, process_run_id=SILVER_SUBSETS_PROCESS_RUN_ID, pipeline_mode=PIPELINE_MODE)`. Four sections are populated with `update_truth_section`:

- **config_snapshot**: source config stage (`silver_eda`), effective stage/layer, dataset name, pipeline mode, run mode.
- **runtime_facts**: parent layer/hash, source profiled dataframe path, profiled state column, and per-state row counts. A second `update_truth_section` call appends the **missingness quarantine passthrough** — `silver_truth["runtime_facts"]["missingness_quarantine"]` read from the Silver 01 parent truth record. This is the mechanism by which Gold recovers the full quarantine audit without reading Silver 01's truth record directly. If the payload is absent, it is silently omitted.
- **artifact_paths**: the full generator bundle, dropped-sensor artifacts, correlation/grouping artifacts, diagnostic CSVs, episode status counts, and the parent feature registry passthrough.
- **notes**: a human-readable purpose statement.

`build_truth_record` finalizes row/column counts and produces `SILVER_EDA_PROFILE_TRUTH_HASH`. Stage identity is then attached directly (`truth_stage = "eda_profile"`, `notebook_name = "silver_02b_eda_profile"`, `truth_path`) because the truth utility does not encode stage-specific filenames — stage identity lives in the record body and the truth index, not in nested folders. `save_truth_record(... layer_name="silver")` writes the JSON to `TRUTHS_PATH/silver/`, and `append_truth_index` registers it so downstream stages can find it by `truth_stage="eda_profile"` without knowing the hash in advance.

Reproducibility and auditability matter here because this record is the bridge from Silver EDA into the synthetic generator and Gold preprocessing. The profile truth hash links the artifact bundle to the exact Silver run that produced it, the forwarded missingness payload preserves the quarantine audit one hop closer to Gold, and the ledger records both the new hash and the parent Silver hash at each significant step.

W&B is disabled through an evaluated boolean toggle: `USE_EXPERIMENT_TRACKING = False`. Unlike Silver 02a (where `wandb.init` is preserved as a triple-quoted string), Silver 02b evaluates the flag at runtime; the `if USE_EXPERIMENT_TRACKING:` branch only prints a confirmation and starts no run. The original standalone logging and ledger setup blocks are preserved as triple-quoted strings and are not executed; `load_notebook_context` performs both.

## Downstream Technical Handoff

Source-confirmed handoffs from Silver 02b:

- **Synthetic generator (generator input bundle).** The notebook's own comments and the generator profile column contract confirm the bundle is built for `utils.synthetic.generator`. The `generator_input_manifest.json` is the single entry point that indexes the per-state rich profiles, dropped-sensor registry, dropped feature profiles, fault pairings, sensor group map, correlation pairs, hotspot-clusters placeholder, and episode status counts. Because the generator reads these schemas directly, changes to artifact columns require matching generator changes.
- **Silver EDA Profile truth record and truth index.** The `silver_eda_profile_truth` record (`truth_stage="eda_profile"`, parent hash = Silver 01's hash) and its truth index entry are written for downstream navigation. The record carries the forwarded missingness quarantine payload and the full `artifact_paths` list.

For Gold notebooks specifically, the workflow reference describes Gold 01 using the profile truth record, missingness passthrough, correlation/grouping artifacts, and diagnostic CSVs as preprocessing guidance. From Silver 02b source alone, the truth record and artifact bundle are produced and indexed, but a direct file-level dependency of any specific Gold notebook on these outputs is **Not determined from available source**. Notebook order alone is not treated as evidence of direct handoff.

## Key Technical Decisions

| Decision | Source Evidence | Why It Matters | Verification Method |
|---|---|---|---|
| Read the Silver 02a profiled dataframe rather than re-derive profiled states | Cell 32 loads `*__silver_subsets__profiled_dataframe.parquet`; masks read `machine_status__profiled` (cell 41) | Keeps profiled-state definitions single-sourced in Silver 02a; Silver 02b characterizes states it does not own | Confirm Silver 02b never writes `machine_status__profiled`; only reads it |
| Capture `SILVER_TRUTH_HASH` before any subsetting | Cell 34 comment: link back to the Silver source for lineage; `extract_truth_hash` runs before masks | A hash taken after filtering would reference a slice, breaking the lineage link in the profile truth record | Confirm `SILVER_TRUTH_HASH` equals `meta__truth_hash` on the full loaded dataframe |
| Inherit `FEATURE_COLUMNS` from the Silver feature registry, intersected with the dataframe | Cell 36 ledger `why`: inherit resolved feature metadata rather than rebuild from config; `ValueError` on empty intersection | Keeps EDA and generator exports on the exact feature lineage Silver 01 finalized; prevents silent feature drift | Confirm `FEATURE_COLUMNS` matches the registry's `feature_columns` minus columns absent from the dataframe |
| Fixed "rich profile" column contract for generator inputs | Cell 78 `build_rich_feature_profile` docstring lists required columns | The synthetic generator reads these columns directly; the schema is a hard contract | Diff generator profile CSV headers against the documented required column set |
| Robust bounds `median ± 4·(IQR/1.349)` clamped within `[p01,p99]` and `[min,max]` | Cell 78 bound construction | Bounds the generated per-state envelope using outlier-resistant scale without letting extremes widen it | Recompute bounds from a state subset and compare to the CSV |
| Generate-then-mask vs never-generate for dropped sensors | Cell 80: `all_null` sensors excluded from profiling; others profile-eligible; comments name `sensor_15` / `sensor_50` | Preserves the original missingness structure in the synthetic dataset | Confirm `all_null` drop-reason sensors are absent from dropped feature profiles |
| Two independent sensor-grouping methods | Cell 74 connected components at 0.80; cell ~101 agglomerative clustering | Threshold-based and fixed-count groupings are orthogonal; downstream consumers can choose either | Confirm both `sensor_group_map_normal_clean.csv` and `feature_cluster_map.csv` exist |
| Per-state correlation guarded by `rows > 1` | Cells 72, 76 conditional blocks | `.corr()` on empty or single-row subsets is meaningless; guard prevents degenerate artifacts | Confirm empty dataframes are assigned and a note is printed when a state is too small |
| Fault pairings clipped to `[0.05,0.95]` and capped at top 150 | Cell 78 strength clip and `.head(150)` | Keeps the generator's coupling model bounded, finite, and small enough to be tractable | Confirm `sensor_fault_pairings_normal.csv` has ≤150 rows and coupling within the clip range |
| IsolationForest outlier audit is diagnostic, trained only on `normal_clean` | Workflow ref + outlier audit cells; output is `outlier_summary.csv` | Provides Gold contamination guidance without becoming a production model artifact | Confirm the fitted model output is a CSV summary, not a saved model file |
| Third Silver-layer truth record with `truth_stage="eda_profile"` | Cell 118 `initialize_layer_truth` + direct `truth_stage` / `notebook_name` assignment | Documents the EDA bundle as a distinct, indexable lineage node under the silver layer | Confirm a `truth_stage="eda_profile"` entry exists in the truth index |
| Forward Silver 01 missingness quarantine into the profile truth record | Cell 118 reads `silver_truth["runtime_facts"]["missingness_quarantine"]` and re-adds it | Lets Gold recover the quarantine audit without reading Silver 01's truth record directly | Confirm the profile truth `runtime_facts` contains the quarantine payload when the parent has one |
| SQL via `write_silver_eda_sql_outputs` (not `log_silver_eda_sql`) | Cell 137 call to seven `silver.eda_*` tables | Silver EDA outputs target a different table family than pipeline-metadata logging | Confirm the seven `silver.eda_*` tables receive rows when the gate is `True` |
| W&B disabled via evaluated boolean toggle | Cell 27 `USE_EXPERIMENT_TRACKING = False` | Keeps the notebook light while leaving a clear, runtime-evaluated activation point | Confirm no `wandb` run is started and the false branch only prints |

## Failure Modes and Guardrails

| Failure Condition | Behavior | Guardrail |
|---|---|---|
| No profiled Parquet at primary path or in fallback dirs | `FileNotFoundError` listing all checked locations | Canonical path tried first; recursive glob fallback before raising |
| Profiled dataframe missing `meta__truth_hash` | `ValueError` | Checked immediately after load |
| `meta__dataset` empty/missing | `ValueError` | Non-empty value check before parent truth load |
| Parent truth did not resolve a feature registry dir | `ValueError` before load | Explicit `None` check on `FEATURE_REGISTRY_PATH` |
| Feature registry missing, non-dict, or empty `feature_columns` | `ValueError` / `TypeError` | Type and content validation after `load_json` |
| All feature columns absent from dataframe | `ValueError` after intersection | Post-intersection empty check; warning lists skipped columns |
| Required state columns absent | `KeyError` | Explicit required-column check before mask construction |
| `meta__episode_id` absent | `KeyError` | Checked before episode summaries |
| A required generator source state empty (`normal_clean` / `abnormal` / `recovery`) | `ValueError` | Per-state emptiness checks before profile build |
| `sensor_group_map_df` / `fault_pairings_df` missing or empty | `ValueError` / `TypeError` | Presence and type guards in the generator export cell |
| Manifest prerequisite variables undefined | `NameError` listing them | Required-variable check before manifest update |
| Manifest references a non-existent path | Printed `WARNING`; manifest still written | Soft validation — does not raise (paths re-checked by the generator at its own load) |
| Truth dir or truth index missing/empty | `FileNotFoundError` | QA section verifies before SQL write |
| Truth index missing required columns | `KeyError` | QA section validates `layer_name` / `truth_stage` / `truth_path` / `truth_hash` |
| `WRITE_SILVER_EDA_SQL_OUTPUTS = False` | All DB writes skipped; file artifacts unaffected | Explicit boolean gate |

## Verification Checklist

- Active notebook path is `notebooks/eda/EDA_Notebook_Pump_Silver_02b_EDA_v2.ipynb`
- Silver 02a profiled Parquet exists at the canonical path or a fallback dir
- `meta__truth_hash` is present and non-null in the profiled dataframe
- Silver 01 parent truth JSON exists and resolves a `feature_registry_dir`
- `FEATURE_COLUMNS` is non-empty after intersecting registry with dataframe columns
- `machine_status__synthetic` and `machine_status__profiled` columns are present
- `meta__episode_id` is present for episode summaries
- Per-state subsets are `.copy()` and `silver_eda_dataframe` is not mutated
- Generator bundle files exist in `GENERATOR_INPUT_DIR`, including `generator_input_manifest.json`
- Rich profile CSV headers match the documented generator column contract
- `dropped_sensor_registry.json` excludes `all_null` sensors from profiling
- Both `sensor_group_map_normal_clean.csv` and `feature_cluster_map.csv` exist
- `sensor_fault_pairings_normal.csv` has ≤150 rows with coupling in `[0.05,0.95]`
- Correlation artifacts for contaminated/abnormal exist only when those states have >1 row
- `silver_eda_profile_truth` JSON exists under `TRUTHS_PATH/silver/` with `truth_stage="eda_profile"` and parent hash = Silver 01's hash
- The profile truth `runtime_facts` carries the forwarded missingness quarantine payload when the parent has one
- A truth index entry for `truth_stage="eda_profile"` exists
- `{DATASET_NAME}__silver__eda_profile__summary.json` exists in `SILVER_EDA_SUMMARY_DIR`
- If `WRITE_SILVER_EDA_SQL_OUTPUTS = True`: seven `silver.eda_*` tables receive rows
- No `wandb` run is started (`USE_EXPERIMENT_TRACKING = False`)

## Source-Limited Items

- Whether any specific Gold notebook directly reads Silver 02b artifacts (generator bundle, truth record, diagnostic CSVs) at the file level is Not determined from available source. The workflow reference describes Gold 01 using them as preprocessing guidance, but Silver 02b source confirms only production and indexing of these outputs, not a confirmed downstream file dependency.
- The downstream consumer of the synthetic generator's output relative to Silver 02b (i.e., what the generator feeds next) is Not determined from Silver 02b source.
- The exact internal schema and column structure written by `write_silver_eda_sql_outputs` into each of the seven `silver.eda_*` tables is Not determined from Silver 02b source (the function is called, not defined, in the notebook).
- The precise resolution order and final value of the dropped-sensors Parquet path (`DROPPED_SENSORS_PARQUET_PATH`) beyond the candidate-list pattern is Not fully determined from the visible portion of the dropped-sensor cell.
- Whether the `silver_correlation_hotspot_clusters_normal_clean.json` placeholder (`{"clusters": []}`) is ever populated by a later run or consumed by the generator is Not determined from available source.
