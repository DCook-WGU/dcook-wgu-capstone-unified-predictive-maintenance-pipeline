# `utils/core/truths.py` Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `utils/core/truths.py` that need deeper explanation than the module-level utility reference. The focus is truth-record initialization, final hash construction, truth JSON persistence, and dataframe lineage stamping.

## Source Grounding

Sources used:

- Active utility source: `utils/core/truths.py`
- Function inventory: `function_inventory.json`
- Scope plan: `technical_reference/03_utility_module_references/071e_scope_plan.md`
- Module reference: `technical_reference/03_utility_module_references/utils__core__truths_module_reference.md`
- Notebook workflow references under `technical_reference/01_notebook_workflow_references/`
- Project manual files under `technical_reference/00_project_manual/`
- Documentation standards under `docs_agent_pack/00_STANDARDS/`

The active utility source file is the source of truth for function behavior.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
|---|---|---|
| `initialize_layer_truth` | Creates the base truth payload for a Medallion layer before row facts and output paths are known. | Bronze, Silver, Gold lineage initialization. |
| `build_truth_record` | Finalizes a truth record and computes the deterministic truth hash. | Stage finalization and lineage audit. |
| `save_truth_record` | Persists a truth record JSON file under a layer-specific truth directory. | Downstream truth lookup and manual review. |
| `stamp_truth_columns` | Returns a dataframe copy stamped with row-level truth lineage metadata. | Medallion dataframe handoff and downstream validation. |

## Module-Level Technical Context

`truths.py` implements the project's core lineage record pattern. The selected functions create a base truth payload, finalize that payload with row and column facts, persist it to disk, and propagate its hash into dataframe rows. This supports the capstone's Medallion handoff design, where each stage can identify its parent truth and downstream consumers can inspect row-level lineage columns.

## Deep Function References

### `initialize_layer_truth`

#### Functional Purpose

`initialize_layer_truth` creates the base truth payload for a Medallion layer before final row counts, column counts, file fingerprints, artifact paths, or runtime facts are complete. It establishes the layer identity and parent linkage that later truth updates build on.

#### Pipeline Context

Workflow references confirm use in Bronze_01, Silver_01, Silver_02a, Silver_02b, Gold_01, Gold_02, Gold_03a, Gold_03b, Gold_03c, Gold_04, and Gold_05. The project manual identifies Bronze truth as the chain root and describes parent truth hashes propagating into Silver and Gold stages.

#### Inputs and Assumptions

Important inputs:

- `truth_version`: the truth record schema/version identifier used by the notebook.
- `dataset_name`: dataset identity carried through the truth chain.
- `layer_name`: Medallion or stage layer identity, such as `bronze`, `silver`, `gold`, `gold_baseline`, or `gold_cascade`.
- `process_run_id`: run identifier for this process execution.
- `pipeline_mode`: mode context, such as train or test.
- `parent_truth_hash`: the upstream truth hash, or `None` for a root layer such as Bronze.

The function does not validate these values for blank strings. It records the values supplied by the caller.

#### Outputs and Return Contract

The return value is a dictionary containing:

- `truth_version`
- `dataset_name`
- `layer_name`
- `process_run_id`
- `pipeline_mode`
- `parent_truth_hash`
- Empty sections for `source_fingerprint`, `config_snapshot`, `runtime_facts`, `artifact_paths`, and `notes`

The returned dictionary is a base payload. It is not the final truth record because it does not yet include `truth_hash`, `created_at_utc`, row count, column count, meta columns, or feature columns.

#### Side Effects

No file writes, directory creation, dataframe mutation, or external calls are performed by this function.

#### Failure Behavior and Guardrails

No explicit validation or exceptions are defined in the function body. Normal Python call-time errors apply if required keyword arguments are omitted.

#### Lineage and Reproducibility Role

This function creates the parent/child lineage root for a stage. The `parent_truth_hash` field is the key chain element that connects a child layer to its upstream truth record. The function also preserves `pipeline_mode` and `process_run_id`, which help distinguish how and when the stage was executed.

#### Why This Function Matters

Truth records are built incrementally. If the base truth payload is inconsistent, every later update and the final truth hash inherit that inconsistency. This function is important because it sets the required identity fields before artifacts, runtime facts, and dataframe facts are known.

#### Verification Method

Practical verification checks:

- Call the function with a known parent hash and confirm the returned dict contains the same `parent_truth_hash`.
- Confirm a Bronze root call can use `parent_truth_hash=None`.
- Confirm the returned dict contains empty `config_snapshot`, `runtime_facts`, `artifact_paths`, and `notes` sections.
- Confirm no `truth_hash` exists until `build_truth_record` is called.

### `build_truth_record`

#### Functional Purpose

`build_truth_record` converts a base truth payload into the final truth record for a stage output. It adds dataframe facts, computes a deterministic truth hash, adds a creation timestamp, and returns the completed record.

#### Pipeline Context

Workflow references confirm this function across Bronze, Silver, and Gold finalization steps. It is used to produce stage-level truth hashes such as Bronze truth, Silver truth, Gold preprocessing truth, baseline truth, cascade truth, comparison truth, and anomaly-detection truth.

#### Inputs and Assumptions

Important inputs and assumptions:

- `truth_base` must contain `truth_version`, `dataset_name`, `layer_name`, `process_run_id`, `pipeline_mode`, and `parent_truth_hash`.
- `row_count` and `column_count` are converted to `int`.
- `meta_columns` and `feature_columns` are sorted before being stored.
- Optional sections `source_fingerprint`, `config_snapshot`, `runtime_facts`, `artifact_paths`, and `notes` are read from `truth_base` when present and default to `{}` otherwise.

The function assumes the caller has already gathered correct row counts, column counts, and column lists from the output dataframe or artifact being documented.

#### Outputs and Return Contract

The returned truth record includes:

- `truth_hash`
- `created_at_utc`
- Identity fields from `truth_base`
- `source_fingerprint`
- `row_count`
- `column_count`
- sorted `meta_columns`
- sorted `feature_columns`
- `config_snapshot`
- `runtime_facts`
- `artifact_paths`
- `notes`

The `truth_hash` is computed from the payload before `truth_hash`, `created_at_utc`, and `notes` are added.

#### Side Effects

No file writes, directory creation, dataframe mutation, or external calls are performed by this function.

#### Failure Behavior and Guardrails

Confirmed guardrails:

- Missing required keys in `truth_base` raise `KeyError`.
- `row_count` and `column_count` are coerced to `int`, so invalid count values can raise conversion errors.
- The hash uses `_normalize_for_json` and sorted JSON keys to avoid hash variation from mapping insertion order.

#### Lineage and Reproducibility Role

The truth hash is the central lineage identifier. It is computed from identity fields, parent hash, file/source fingerprint, row and column facts, config snapshot, runtime facts, and artifact paths. Because `truth_hash` and `created_at_utc` are excluded from the hash input, the hash describes the payload content rather than describing itself.

Sorting `meta_columns` and `feature_columns` reduces hash instability caused by caller-side list ordering.

#### Why This Function Matters

This function is the point where stage facts become a stable lineage record. Downstream notebooks and manual reviewers use the resulting hash to confirm that dataframe rows, artifact paths, parent truth, and runtime context belong to the same stage output.

#### Verification Method

Practical verification checks:

- Confirm the returned record includes `truth_hash` and `created_at_utc`.
- Confirm `row_count` and `column_count` are stored as integers.
- Confirm `meta_columns` and `feature_columns` are sorted.
- Rebuild the same base payload and dataframe facts, then confirm the computed hash is stable apart from timestamp-bearing fields excluded from the hash input.
- Confirm changing a lineage-relevant input, such as `parent_truth_hash`, `runtime_facts`, or `artifact_paths`, changes the resulting `truth_hash`.

### `save_truth_record`

#### Functional Purpose

`save_truth_record` writes a truth record to a JSON file under a layer-specific truth directory and returns the output path. It makes the in-memory truth record available for downstream lookup, manual review, and later lineage validation.

#### Pipeline Context

Workflow references confirm truth record persistence in Bronze_01, Silver_01, Silver_02a, Silver_02b, Gold_01, Gold_02, Gold_03a, Gold_03b, Gold_03c, Gold_04, and Gold_05. The project manual states that Bronze truth anchors the lineage chain and downstream notebooks use parent truth hashes.

#### Inputs and Assumptions

Important inputs and assumptions:

- `truth_record` must contain a `truth_hash` key.
- `truth_dir` is the shared truth root supplied by the caller.
- `layer_name` is appended to `truth_dir` to keep truths separated by layer.
- `dataset_name`, `layer_name`, and `truth_hash` are used to build the filename:
  `{dataset_name}__{layer_name}__truth__{truth_hash}.json`

The function does not sanitize `dataset_name` or `layer_name`; it uses the values supplied by the caller.

#### Outputs and Return Contract

The return value is the `Path` to the written JSON truth file.

#### Side Effects

Source-confirmed side effects:

- Creates the layer truth directory with `parents=True, exist_ok=True`.
- Writes the normalized truth record as UTF-8 JSON with two-space indentation and `ensure_ascii=False`.

#### Failure Behavior and Guardrails

Confirmed guardrails:

- Missing `truth_record["truth_hash"]` raises `KeyError`.
- File-system errors from directory creation or file writing propagate to the caller.
- JSON serialization is routed through `_normalize_for_json`, which handles common project values such as `Path` and pandas timestamps.

#### Lineage and Reproducibility Role

Persisted truth JSON is the durable record that downstream stages can load by path or hash. It contains the final truth hash and the parent truth hash, letting reviewers trace a layer output back to its upstream data and config context.

#### Why This Function Matters

Without a saved truth record, dataframe-level `meta__truth_hash` values point to a hash that cannot be inspected. The JSON file is the bridge between row-level lineage columns and human-readable audit information.

#### Verification Method

Practical verification checks:

- Confirm the returned path exists after the function runs.
- Confirm the path is under `truth_dir / layer_name`.
- Confirm the filename contains dataset name, layer name, and truth hash.
- Load the JSON and confirm `truth_hash` matches the input truth record.
- Confirm path-like values in the truth record were serialized as strings.

### `stamp_truth_columns`

#### Functional Purpose

`stamp_truth_columns` returns a copy of a dataframe with row-level lineage columns added or overwritten. It stamps the current truth hash, optional parent truth hash, and optional pipeline mode into every row of the returned dataframe.

#### Pipeline Context

Workflow references confirm this function in Bronze_01, Silver_01, Gold_01, Gold_02, Gold_03a, Gold_03b, Gold_03c, Gold_04, and Gold_05. Silver_02a is explicitly documented as a case where output parquet rows are not stamped with the Silver_02a truth hash; the truth hash lives in the truth record and index instead.

#### Inputs and Assumptions

Important inputs and assumptions:

- `dataframe` is a pandas `DataFrame`.
- `truth_hash` is written to `meta__truth_hash`.
- `parent_truth_hash` is written to `meta__parent_truth_hash`, including `None` when no parent exists.
- `pipeline_mode` is written to `meta__pipeline_mode` only when it is not `None`.

The function does not validate hash format or check for existing conflicting values.

#### Outputs and Return Contract

The return value is a copied dataframe. Active source calls `dataframe.copy()` before assigning lineage columns, so the function does not intentionally mutate the input dataframe in place.

The returned dataframe always includes:

- `meta__truth_hash`
- `meta__parent_truth_hash`

It includes `meta__pipeline_mode` only when `pipeline_mode` is provided.

#### Side Effects

No file writes, directory creation, database calls, or external service calls are performed. The source-confirmed dataframe behavior is copy-and-return, not in-place mutation.

#### Failure Behavior and Guardrails

No explicit validation or custom exceptions are defined in the function body. Errors would come from pandas dataframe operations if the supplied object does not support `.copy()` or column assignment.

#### Lineage and Reproducibility Role

This function propagates truth metadata from the stage-level truth record into row-level data. Downstream notebooks can inspect `meta__truth_hash` and `meta__parent_truth_hash` to verify that rows came from the expected stage output and parent truth chain. `meta__pipeline_mode` allows downstream checks to distinguish train/test or other mode contexts when the caller supplies it.

#### Why This Function Matters

Truth records are stage-level artifacts, but downstream notebooks usually consume dataframes. Row-level truth columns connect dataframe rows back to truth JSON files and parent hashes. Without this stamping step, downstream data could lose the link between the rows being modeled and the truth record that describes how those rows were produced.

#### Verification Method

Practical verification checks:

- Confirm the returned dataframe contains `meta__truth_hash` equal to the supplied truth hash for every row.
- Confirm `meta__parent_truth_hash` equals the supplied parent hash, or is null/None when no parent is supplied.
- Confirm `meta__pipeline_mode` appears only when a pipeline mode is supplied.
- Confirm the original dataframe object is not the same object as the returned dataframe.
- Confirm changing the returned dataframe's lineage columns does not mutate the original dataframe unless the caller separately assigns the returned object back to the same variable.

## Cross-Function Relationships

- `initialize_layer_truth` creates the base truth payload with identity and parent linkage.
- Callers can update the base payload with config, runtime facts, artifact paths, and source fingerprints before finalization.
- `build_truth_record` finalizes the payload and computes the truth hash.
- `save_truth_record` persists the completed truth record so downstream stages and reviewers can load it.
- `stamp_truth_columns` propagates the completed truth hash and parent truth hash into dataframe rows.
- `build_truth_config_block` in `config_loader.py` provides the compact config payload commonly stored in a truth record's `config_snapshot` section.
- Artifact paths created through `artifacts.py` helpers are commonly recorded in truth payload `artifact_paths` before `build_truth_record` runs.

## Source-Limited Items

- Some workflow references describe stamping as in-place. Active source returns a copied dataframe. Caller-side assignment determines whether the calling notebook replaces its original dataframe variable.
- The exact validation checks performed after stamping are notebook-specific. Workflow references confirm several final lineage checks, but those checks are not implemented inside `stamp_truth_columns`.
- The function does not itself save truth index entries; that behavior belongs to `append_truth_index`, which is outside this assigned first batch.
