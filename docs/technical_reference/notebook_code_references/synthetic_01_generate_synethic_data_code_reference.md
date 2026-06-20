# Notebook Code Reference: synthetic_01_generate_synethic_data

Notebook path: `notebooks/synthetic/synthetic_01_generate_synethic_data.ipynb`

## Notebook Purpose

This notebook is the primary synthetic pump telemetry generator for the capstone project. It reads
sensor statistical profiles produced by the Silver EDA layer (normal, abnormal, and recovery state
profiles, correlation pairs, and fault pairings), configures `SyntheticGenerator` with those
profiles, and runs a batch generation loop to produce a labeled dataset of synthetic pump sensor
readings. The output dataset carries known ground-truth labels (`stream_state`, `phase`,
`machine_status`) and injected anomalies, and is written to Postgres and exported as
parquet/CSV artifacts for use by downstream Bronze and streaming notebooks. A synthetic truth
record is also saved, linking this run back to the Silver EDA parent via `parent_truth_hash`.

Notebook stage: `Synthetic`

---

## Section Overview

| Section | Cells | Description |
|---|---|---|
| Imports and Environment Setup | 02–08 | Imports, timestamp capture, logging init |
| Configuration and Runtime Context | 04–06, 12, 16 | Config load, DATASET\_ID, RUN\_ID, calibration/correlation config extracts |
| Parent Silver Artifact Layer | 09–12, 20 | Silver layer resolution, truth hash lookup, truth record load |
| Backward-Compatible Runtime Aliases | 13–14 | Stable RUN\_ID pin, episode range aliases |
| Generation Parameter Block | 12, 18 | MODE, TARGET\_ROWS, ROWS\_PER\_FAILURE, EPISODE\_MAX\_ROWS, WRITE\_MODE |
| Resolve Hotspot Clusters | 21–22 | Load cluster artifact from Silver; fall back to YAML or internal derivation |
| Load Silver EDA Profile Paths | 25 | Resolve artifact paths for profiles, correlation pairs, group map, fault pairings |
| Optional Episode Status Counts | 26–29, 39–42 | Load episode\_status\_counts.json; compute recovery/normal row medians |
| Dropped Sensor Profile Merge | 30–37 | Resolve and merge dropped-sensor profile CSVs; hard check before generation |
| Generator Build | 30 | Construct SyntheticGenerator with merged profiles, correlation structure, calibration |
| Pre-Generation Checks | 31–35, 38, 43, 46 | Required-object guard, fault sensor list, signature and chain threshold inspection |
| Generation Run | 48 | Core batch/single episode loop; OBSERVABLE threshold control |
| Missingness Failsafe | 50–51 | Enforce Silver EDA missingness targets; protect BROKEN/failure rows |
| Schema Restoration (52-sensor) | 53–54 | Restore missing sensor columns (sensor\_15 all-null; sensor\_50 must exist) |
| Post-Generation Diagnostics | 56–61, 67, 71, 76–77 | State/phase counts, variance check, schema/missingness/correlation verification |
| Postgres Write | 62–63 | Batch/cycle ID management, stream batch write (reset or append mode) |
| Truth Record Initialization | 69 | Initialize synthetic\_truth with parent\_truth\_hash; export resolved config |
| engine.dispose() | 81 | Release connection pool before export |
| Config Snapshot Export | 83–85 | Flatten and export resolved runtime config to CSV |
| Post-Generation Scorecard | 87 | Run scorecard: state mix, missingness, correlation pairs, clusters, decision |
| Pipeline Continuation | 88 | %run to synthetic\_pipeline\_condensed-02\_03.ipynb |

---

## Section Details

### Imports and Environment Setup

Cells 02–08 bring in standard library modules (`pathlib`, `logging`, `json`, `datetime`,
`inspect`, `random`), numeric libraries (`numpy`, `pandas`), and the full set of project utility
imports from `utils.core` (paths, file I/O, logging setup, config loader, truth utilities,
database utilities) and `utils.synthetic.generator` (profiles, generator, missingness, export).

A generation start timestamp is captured at cell 03 immediately after imports. The container clock
runs UTC; the notebook records an EST/EDT-equivalent time by subtracting 4 hours before formatting
the timestamp. This adjusted timestamp is stored in `generation_started_formatted_datetime` and
`%store`d for downstream cells and the artifact export filename.

Logging is configured at cell 06 to write to `logs/synthetic_data_generator.log`. The logger
name is `capstone.synthetic`.

**Key variables produced:**

- `generation_started_formatted_datetime` — adjusted local timestamp string used in artifact filenames
- `logger` — notebook-level logger

---

### Configuration and Runtime Context

Cells 04–05 load the pipeline config via `load_pipeline_config` and populate `CONFIG`, `SYN_CFG`,
`PATHS`, and `PIPELINE`. `PIPELINE_MODE` (`"batch"` or `"notebook"`) is used to qualify truth
records and artifact exports. Cell 16 extracts calibration and correlation settings from
`SYN_CFG`: `CALIBRATION_ENABLED`, `CALIBRATION_MEAN_WITHIN_K_STD`, `CALIBRATION_STD_RATIO_BOUNDS`
(control per-state calibration of generated sensor means/std), `CORRELATION_HOTSPOT_CLUSTERS`
and `CORRELATION_CLUSTER_DERIVATION` (YAML cluster groups; may be overridden by Silver artifact),
`FAULT_EXCLUDED_SENSORS` (default: sensor\_15, sensor\_50), and `CORRELATION_TUNING_CFG`.

**Key variables produced:**

- `SYN_CFG`, `CONFIG`, `PIPELINE_MODE`, `DATASET_NAME`, `TRUTH_VERSION`
- `TRUTHS_PATH`, `TRUTH_INDEX_PATH`, `ARTIFACTS_ROOT`, `LOGS_PATH`
- `CALIBRATION_ENABLED`, `CALIBRATION_MEAN_WITHIN_K_STD`, `CALIBRATION_STD_RATIO_BOUNDS`
- `FAULT_EXCLUDED_SENSORS`, `CORRELATION_HOTSPOT_CLUSTERS`, `CORRELATION_TUNING_CFG`

---

### Parent Silver Artifact Layer

Cells 09–12 identify the Silver parent. `get_latest_parent_profile_truth_hash` reads
`truth_index.jsonl` in append order, preferring `silver_subsets` and falling back to `silver_eda`.
Cell 20 loads the truth record JSON via `load_truth_record_by_hash`. The alias
`silver_eda_truth = silver_parent_truth` keeps older cells working without changes.
`PARENT_TRUTH_HASH` is the direct parent hash; `SILVER_PREEDA_TRUTH_HASH` is the grandparent.
Cell 20 raises `ValueError` immediately if `SILVER_PARENT_TRUTH_HASH` is unset.

**Key variables produced:**

- `SILVER_PARENT_LAYER_NAME`, `SILVER_PARENT_TRUTH_HASH`
- `silver_parent_truth`, `silver_eda_truth`, `PARENT_TRUTH_HASH`, `SILVER_PREEDA_TRUTH_HASH`

---

### Backward-Compatible Runtime Aliases

Cell 14 pins `RUN_ID` from the `SYNTHETIC_RUN_ID` environment variable (fallback:
`SYNTH_PROCESS_RUN_ID`). `RUN_ID` must not be regenerated later; all downstream cells re-use it
as `process_run_id = RUN_ID`. Episode phase length range aliases (`NORMAL_BEFORE_RANGE`,
`BUILDUP_RANGE`, `FAILURE_RANGE`, `RECOVERY_RANGE`, `NORMAL_AFTER_RANGE`, `MAGNITUDE_RANGE`) and
their lowercase equivalents are assigned so that older debug cells continue to work without
modification.

**Key variables produced:**

- `RUN_ID`, `NORMAL_BEFORE_RANGE`, `BUILDUP_RANGE`, `FAILURE_RANGE`, `RECOVERY_RANGE`, `NORMAL_AFTER_RANGE`, `MAGNITUDE_RANGE`

---

### Generation Parameter Block

The active generation parameters live in cell 12 and are supplemented by commented reference
blocks (cells 18, 44) that show tuning history. The active values that control output size and
episode composition are:

- `MODE` — `"batch"` (multi-episode loop) or `"single"` (one episode for debug)
- `TARGET_ROWS` — total number of rows to produce in batch mode (reference block shows 225,000)
- `MAX_EPISODES` — safety cap on the episode loop
- `EPISODE_MAX_ROWS` — maximum rows per episode, preventing runaway single-episode datasets
- `ROWS_PER_FAILURE` — target spacing between fault episodes (reference: ~32,000 rows)
- `WRITE_MODE` — `"reset"` (delete existing rows before insert) or `"append"`
- `APPEND_MODE` — `"renumber"` or `"continue"` (controls episode/batch ID renumbering on append)

---

### Resolve Hotspot Clusters for Generator

Cell 22 resolves the correlation hotspot clusters that `SyntheticGenerator` uses to model
sensor co-movement during normal operation. The resolution priority is:

1. Load the hotspot cluster artifact from the Silver parent truth record (key
   `hotspot_clusters_normal` from `SYN_CFG["silver_eda_artifact_keys"]`). Use this if the file
   exists and is non-empty.
2. If the artifact is absent or empty, use the YAML-configured `CORRELATION_HOTSPOT_CLUSTERS`.
3. If both are empty, pass an empty list to `SyntheticGenerator`, which will then derive clusters
   from `correlation_cluster_derivation` config internally.

**Key variables produced:**

- `HOTSPOT_CLUSTERS_FOR_GENERATOR` — the resolved cluster list passed to the generator

---

### Load Silver EDA Profile Paths

Cell 25 resolves artifact paths from the Silver parent truth record for the six profile/structure
files the generator requires:

- `profile_normal_path` — clean-normal sensor statistical profile CSV
- `profile_abnormal_path` — abnormal state profile CSV
- `profile_recovery_path` — recovery state profile CSV
- `corr_pairs_normal_path` — Pearson correlation pairs CSV for normal state
- `group_map_normal_path` — sensor group membership CSV
- `fault_pairings_normal_path` — fault coupling strength CSV

All paths are resolved via `get_artifact_path_from_truth` using keys from
`SYN_CFG["silver_eda_artifact_keys"]`.

---

### Optional Episode Status Counts Resolver

Cells 26–29 and 39–42 load `episode_status_counts.json` from the Silver generator inputs
directory if it exists. This file is a list of per-episode row counts (normal, failure, recovery,
episode\_total\_rows) extracted from the real Silver dataset. It is optional: the YAML already
defines episode range parameters, so generation proceeds without it. If present, the file is used
to compute `RECOVERY_ROWS_MEDIAN`, `RECOVERY_ROWS_Q75`, and `NORMAL_ROWS_MEDIAN` for diagnostic
display, and can be used by `choose_episode_phase_lengths` to sample episode sizes from the real
distribution.

**Key variables produced:**

- `EPISODE_STATS` — list of dicts from episode\_status\_counts.json (empty list if file absent)
- `EPISODE_STATS_PATH_EXISTS` — boolean flag used to gate display cells
- `RECOVERY_ROWS_MEDIAN`, `RECOVERY_ROWS_Q75`, `NORMAL_ROWS_MEDIAN`

---

### Dropped Sensor Profile Merge and Diagnostics

Cell 30 resolves optional dropped-sensor profile CSVs (normal-clean, abnormal, recovery) covering
sensors excluded from the main Silver EDA profile due to excessive missingness or analytical
exclusion (e.g., sensor\_15, sensor\_50). The resolver checks a prioritized list of candidate
paths and returns `None` if none exist. `load_and_merge_rich_profiles` merges the dropped sensor
rows back into each state's profile dict so the generator has complete sensor coverage.

Cells 32–37 inspect whether `sensor_15` and `sensor_50` appear in each profile dict and
`generator.sensors`. Cell 35 is a hard guard that raises `NameError` if the required generator
objects (`normal_profiles`, `abnormal_profiles`, `recovery_profiles`, `generator`) are not in
scope, ensuring the build cell ran before proceeding.

**Key variables produced:**

- `dropped_profile_normal_path`, `dropped_profile_abnormal_path`, `dropped_profile_recovery_path`
- `normal_profiles`, `abnormal_profiles`, `recovery_profiles`
- `corr_pairs_df`, `group_map_df`, `fault_pairings_df`

---

### Generator Build

The `SyntheticGenerator` is constructed at the bottom of cell 30 with the merged
normal/abnormal/recovery profile dicts; correlation pairs, group map, and fault pairings
DataFrames; `HOTSPOT_CLUSTERS_FOR_GENERATOR`; `CORRELATION_CLUSTER_DERIVATION`;
`FAULT_EXCLUDED_SENSORS` (sensor\_15, sensor\_50); `CORRELATION_TUNING_CFG`; the config random
seed; `missingness_spec` from the Silver parent truth payload; and `state_calibration_targets`
(per-sensor mean/std by state, if `CALIBRATION_ENABLED`). Cell 38 re-derives
`FAULT_EXCLUDED_SENSORS` and `FAULT_ELIGIBLE_SENSORS` from the generator's resolved exclusion
list.

**Key variables produced:**

- `generator` — the configured `SyntheticGenerator` instance

---

### Generation Run

Cell 48 is the only cell that creates or overwrites `synthetic_df`. `OBSERVABLE_ZSCORE_THRESHOLD`
(default: 2.5) and `OBSERVABLE_MIN_CONSECUTIVE` (default: 3) control anomaly-onset detection
sensitivity — the z-score level and run-length at which the generator marks a sensor as entering
an observable broken state. Both values are passed to every `generator.generate_episode(...)` call.

In `batch` mode, the loop calls `sample_episode_spec_fit_remaining` to size each episode within
the remaining `TARGET_ROWS` budget. `ROWS_SINCE_LAST_BROKEN` spaces fault episodes at
approximately `ROWS_PER_FAILURE` intervals. After the loop, episodes are concatenated and
hard-trimmed to exactly `TARGET_ROWS`. In `single` mode, one episode is generated for debugging.

**Key variables produced:**

- `synthetic_df`, `OBSERVABLE_ZSCORE_THRESHOLD`, `OBSERVABLE_MIN_CONSECUTIVE`

---

### Missingness Failsafe

Cells 50–51 call `apply_missingness_percentage_failsafe` to add NaN values until each sensor's
actual missing percentage matches the Silver EDA target. It only adds missingness; it never
restores values. BROKEN rows and abnormal/failure/fault phase rows are protected. Targets come
from `missingness_spec.missingness_pct_all`. An audit DataFrame (`missingness_failsafe_audit_df`)
is returned and displayed for review.

---

### Schema Restoration (Full 52-Sensor Schema)

Cells 53–54 add any sensor columns absent from `synthetic_df` after generation. `sensor_15` is
expected to be absent and is restored as all-null. `sensor_50` must be present (it has a
dropped profile covering generation parameters); if absent the cell raises `ValueError`. This
step runs before Postgres write and diagnostics so the full 52-column schema is always present.

---

### Post-Generation Diagnostics

Cells 56–61, 67, 71, and 76–77 provide diagnostic output against `synthetic_df`: state/phase
value counts, broken/recovering row ratio, episode count and primary fault/sensor distributions,
and a variance check comparing actual sensor standard deviations in normal rows against Silver EDA
profile values. Four verification helpers are run:

- `verify_schema` — checks for missing or unexpected sensor columns
- `verify_missingness_exact` — compares actual missing percentages against Silver EDA targets
  within a configurable row tolerance
- `verify_profile_bounds` — checks that generated values fall within Silver EDA p01/p99 bounds by state
- `verify_top_correlations` — compares top Pearson pairs in synthetic normal rows against Silver
  EDA correlation pairs

---

### Postgres Write

Cells 62–63 define table-check helpers (`table_exists`, `drop_table`, `get_max_batch_id`) and
execute the write. `get_engine_from_env()` creates a SQLAlchemy engine from environment variables.
The write target is `capstone.synthetic_{dataset_name}_stream`.

When `WRITE_MODE="reset"`, the two Postgres sequences (`seq_synthetic_{dataset_name}_batch_id`,
`seq_synthetic_{dataset_name}_cycle_id`) are reset before insertion, making the run idempotent.
`choose_batch_id` selects the correct `batch_id` for the current mode; `reserve_cycle_range`
claims a contiguous block of cycle IDs equal to `len(synthetic_df)` before the insert.
`write_stream_batch` performs the bulk insert. Cells 64–65 are diagnostic: they print the resolved
DB environment variables and check that a psycopg driver is available.

---

### Truth Record Initialization

Cell 69 calls `initialize_layer_truth` with `parent_truth_hash=PARENT_TRUTH_HASH` to link this
synthetic run directly to the Silver EDA/subsets truth record that supplied the profiles. The
resolved config is exported to
`artifacts/synthetic/{dataset_name}/{dataset_name}__synthetic__resolved_config.yaml` and
referenced in the truth record's `config_snapshot` section. `process_run_id = RUN_ID` re-uses
the pinned stable `RUN_ID`.

**Key variables produced:**

- `synthetic_truth` — initialized truth record dict

---

### engine.dispose() Before Export

Cell 81 calls `engine.dispose()` before the export cell creates any new database connections.
This releases pooled connections from the SQLAlchemy engine built for the Postgres write step.
Without this call, a connection pool held open from the write step can conflict with the fresh
engine created during artifact export. The `engine.dispose()` call is unconditional; it runs
even if no rows were written.

---

### Config Snapshot Export

Cells 83–85 define `export_config_snapshot_csv`, a local helper that flattens the nested config
dict into a two-column CSV (`config_key`, `value`) and writes it to `data/synthetic/`. The
filename embeds `RUN_ID` and the adjusted local timestamp. `CONFIG_SNAPSHOT_EXTRA` (cell 85)
appends resolved runtime values not present in the YAML: generation mode, target rows, episode
sizing, `OBSERVABLE_ZSCORE_THRESHOLD`, `OBSERVABLE_MIN_CONSECUTIVE`, and the Silver parent layer
name and truth hashes.

---

### Post-Generation Scorecard

Cell 87 defines `build_synthetic_run_scorecard`, which compares `synthetic_df` against a reference
dataset across five metric groups: state mix (NORMAL/BROKEN/RECOVERING percentages), missingness
for priority sensors, Pearson correlations for priority sensor pairs in normal rows, average
absolute cluster correlation, and overall pairwise correlation error (mean, median, p90). Each
metric is rated PASS/WARN/FAIL. A `decision_scorecard` aggregates counts and produces one of three
recommendations: `CANDIDATE_STOP`, `ONE_MORE_TARGETED_PASS`, or `KEEP_TUNING`. Scorecard tables
can be exported via `export_scorecard_bundle`.

---

### Pipeline Continuation

Cell 88 chains to the next notebook stage via `%run ./synthetic_pipeline_condensed-02_03.ipynb`.
This passes execution to the premelt observation and timestamping stages that consume
`synthetic_df` and the artifacts produced by this notebook.

---

## Key Outputs

- `synthetic_df` — labeled DataFrame with 52 sensor columns, `stream_state`, `phase`,
  `machine_status`, `meta__episode_id`, `meta__primary_sensor`, `meta__primary_fault_type`,
  `meta__magnitude`, and cycle/batch metadata
- Postgres table `capstone.synthetic_{dataset_name}_stream` — written with `batch_id` and `cycle_id`
- Parquet artifact — exported by the continuation notebook (`synthetic_pipeline_condensed-02_03`)
- CSV config snapshot: `data/synthetic/synthetic_config_snapshot__{RUN_ID}__{timestamp}.csv`
- Resolved config YAML: `artifacts/synthetic/{dataset_name}/{dataset_name}__synthetic__resolved_config.yaml`
- Synthetic truth record: `truths/synthetic/` with `parent_truth_hash` linking to Silver EDA
- Scorecard CSVs (if `export_scorecard_bundle` is called): per-metric-group CSVs in `data/synthetic/`

---

## Dependencies and Inputs

- **Silver EDA/Subsets truth record** — resolved from `truth_index.jsonl`; must exist for
  `silver_subsets` or the fallback `silver_eda` layer
- **Silver profile CSVs** — normal-clean, abnormal, recovery feature profile CSVs under
  `artifacts/silver_subsets/pump/generator_inputs/`
- **Silver correlation/structure CSVs** — `corr_pairs_normal.csv`, `group_map_normal.csv`,
  `fault_pairings_normal.csv` from the same directory
- **Dropped sensor profile CSVs** (optional) — `dropped_feature_profiles__normal_clean.csv`,
  `dropped_feature_profiles__abnormal.csv`, `dropped_feature_profiles__recovery.csv`
- **Hotspot cluster artifact** (optional) — JSON artifact at
  `silver_eda_artifact_keys.hotspot_clusters_normal`
- **Episode status counts JSON** (optional) — `episode_status_counts.json` in the Silver
  generator inputs directory
- **Pipeline config YAML** — `configs/synthetic/pump/train/default/`
- **PostgreSQL** — `capstone` schema; credentials from environment variables

---

## SQL / Database Operations

| Operation | Condition |
|---|---|
| `SELECT EXISTS(...)` on `information_schema.tables` | Always |
| `DROP TABLE IF EXISTS ... CASCADE` | Only when `WRITE_MODE="reset"` |
| Sequence reset (`SETVAL`) on batch\_id and cycle\_id sequences | Only when `WRITE_MODE="reset"` |
| `CREATE SEQUENCE IF NOT EXISTS` for both sequences | Always |
| `NEXTVAL` reservation for the full cycle range | Before insert |
| Bulk `INSERT` via `write_stream_batch` | Always |

`WRITE_MODE="reset"` is the idempotent path: the table is dropped and sequences are reset before
each insert, so repeated runs rebuild from scratch.

---

## Important Behavioral Notes

- **UTC-to-local timestamp shift**: The container runs UTC. The notebook subtracts 4 hours from
  `datetime.now()` at startup and before artifact naming to record approximate EST/EDT time.
  This affects `generation_started_formatted_datetime`, `formatted_datetime`, and artifact filenames.

- **Stable RUN\_ID**: `RUN_ID` is pinned once at cell 14 from `SYNTHETIC_RUN_ID` env var. It
  must not be regenerated mid-notebook. All later cells re-use `process_run_id = RUN_ID`.

- **parent\_truth\_hash linking**: The synthetic truth record's `parent_truth_hash` points to the
  Silver EDA/subsets truth hash, creating a traceable chain from synthetic data back to its
  statistical source.

- **OBSERVABLE thresholds**: `OBSERVABLE_ZSCORE_THRESHOLD=2.5` and
  `OBSERVABLE_MIN_CONSECUTIVE=3` control anomaly-onset detection sensitivity. They determine the
  z-score magnitude and run-length at which the generator transitions a sensor into an observable
  broken state. Adjusting these changes how sharply anomalies are labeled in the output.

- **Full 52-sensor schema restoration**: The generator models only sensors that have profiles.
  After generation, `sensor_15` is restored as an all-null column (expected; it is excluded from
  the modeling feature set). `sensor_50` must be present in generator output; if absent the
  notebook raises rather than silently adding a null column.

- **Dropped sensor profiles**: `sensor_50` is covered by a separate dropped-profile CSV that is
  merged into the normal/abnormal/recovery profile dicts before generator construction. If the
  dropped profile files are not found, `sensor_50` will be absent from `generator.sensors` and
  the schema restore step will raise.

- **Silver layer preference and fallback**: Prefers `silver_subsets` (Silver notebook 02a output)
  as the parent layer; falls back to `silver_eda` for older runs. The resolved name is stored in
  `SILVER_PARENT_LAYER_NAME` and recorded in the truth record and config snapshot.

- **W&B**: `set_wandb_dir_from_config(CONFIG)` configures the W&B output directory but there is
  no active `wandb.init` / `wandb.log` / `wandb.finish` block. W&B is directory-configured only.

- **engine.dispose() before export**: Cell 81 releases the SQLAlchemy connection pool before the
  export cells, preventing connection leaks when the export utility creates its own engine or on
  kernel-free re-runs.

- **Commented parameter blocks**: Cells 18 and 44 contain disabled parameter blocks (triple-quoted
  strings) showing earlier tuning values for `TARGET_ROWS` and `EPISODE_MAX_ROWS`. Active
  parameters are in cell 12 and the YAML config.

- **Pipeline continuation via %run**: Cell 88 chains to
  `synthetic_pipeline_condensed-02_03.ipynb`. Parquet artifact export is handled by that notebook,
  not by this one directly.
