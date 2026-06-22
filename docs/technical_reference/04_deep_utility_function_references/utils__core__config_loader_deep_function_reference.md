# `utils/core/config_loader.py` Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `utils/core/config_loader.py` that need deeper explanation than the module-level utility reference. The focus is the configuration foundation used by notebook context setup, artifact path resolution, truth records, and reproducible stage execution.

## Source Grounding

Sources used:

- Active utility source: `utils/core/config_loader.py`
- Function inventory: `function_inventory.json`
- Scope plan: `technical_reference/03_utility_module_references/071e_scope_plan.md`
- Module reference: `technical_reference/03_utility_module_references/utils__core__config_loader_module_reference.md`
- Notebook workflow references under `technical_reference/01_notebook_workflow_references/`
- Project manual files under `technical_reference/00_project_manual/`
- Documentation standards under `docs_agent_pack/00_STANDARDS/`

The active utility source file is the source of truth for function behavior.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
|---|---|---|
| `load_pipeline_config` | Loads layered YAML config, resolves templates, builds filenames and paths, and returns a hashed `LoadedConfig`. | Shared notebook context and stage configuration setup. |
| `export_config_snapshot` | Persists a resolved config mapping to a YAML snapshot. | Stage artifact/config output for reproducibility review. |
| `build_truth_config_block` | Extracts a compact config payload for embedding in truth records. | Truth record construction and lineage audit. |

## Module-Level Technical Context

`config_loader.py` is the central configuration resolver for the capstone notebooks. It takes the project's layered YAML configuration, injects runtime context, resolves templated values, builds canonical filenames, builds resolved path mappings, and computes a stable config hash. The selected functions matter because later artifact and truth utilities depend on their output shape.

## Deep Function References

### `load_pipeline_config`

#### Functional Purpose

`load_pipeline_config` loads the selected pipeline configuration fragments and returns a frozen `LoadedConfig` object containing:

- `data`: the fully merged and enriched config dictionary.
- `config_hash`: a SHA-256 hash of the resolved config payload.
- `source_files`: the resolved config file paths used to build the config.

In project terms, this function turns the requested stage, dataset, mode, and profile into the config object that notebooks use for paths, filenames, stage parameters, runtime metadata, and truth-record configuration context.

#### Pipeline Context

The module reference confirms direct consumers in Gold_03b, Gold_03c, Gold_04, and Gold_05. The project manual also describes `load_notebook_context()` as the common notebook bootstrap pattern, and that context depends on resolved config, paths, logger, ledger, and related shared objects. The workflow references confirm that Gold notebooks call or depend on `load_pipeline_config`, `build_truth_config_block`, and `export_config_snapshot` as part of their setup and lineage flow.

#### Inputs and Assumptions

Important parameters and assumptions:

- `config_root` must resolve to a directory containing `base.yaml`, `datasets/`, `modes/`, and `stages/`.
- `stage`, `dataset`, and `mode` select config fragments in `stages/<stage>`, `datasets/<dataset>`, and `modes/<mode>`.
- `profile` is stored in `config["runtime"]["profile"]`; source-confirmed profile-specific merge behavior beyond this assignment is not present in the function body.
- `project_root` controls how resolved path mappings are built. If omitted, the current working directory is used.
- YAML fragments must load to mappings; `_read_yaml` raises `ConfigError` if a file is missing or not a top-level mapping.
- The merged config must contain required keys used by `_build_filename_map` and `_build_path_map`, including dataset metadata, version fields, and path root/subdirectory definitions.

#### Outputs and Return Contract

The return value is a `LoadedConfig` dataclass with:

- `data`: the enriched config dictionary.
- `config_hash`: the SHA-256 hash computed from the enriched config before the hash is written into `config_meta`.
- `source_files`: a list of source config file paths as strings.

The enriched config includes:

- `runtime.stage`, `runtime.dataset`, `runtime.mode`, and `runtime.profile`.
- `filenames`, built from dataset and version metadata.
- `resolved_paths`, built from `project_root`, configured roots, dataset metadata, and canonical filenames.
- `config_meta.source_files`, `config_meta.project_root`, and `config_meta.config_hash`.

#### Side Effects

No file writes or directory creation are performed directly by `load_pipeline_config`. Source-confirmed side effects are limited to reading YAML config files from disk.

#### Failure Behavior and Guardrails

Confirmed guardrails:

- `_resolve_config_file` raises `ConfigError` when a selected config file cannot be found, including `.yaml` and `.yml` suffix discovery for dataset, mode, and stage fragments.
- `_read_yaml` raises `ConfigError` for missing files or non-mapping YAML content.
- Missing required keys in the merged config can surface as `KeyError` through `_build_filename_map`, `_build_path_map`, or later config access.
- Template rendering leaves unknown placeholders unchanged rather than failing immediately.

#### Lineage and Reproducibility Role

`load_pipeline_config` is the upstream source for the config hash and resolved paths used by later functions. The config hash is stored under `config_meta.config_hash`, and `build_truth_config_block` later embeds that value into truth records. The function also records the exact source config files and project root, which gives a reviewer a way to reconstruct how a notebook's runtime config was formed.

The two-pass template resolution matters for reproducibility: the first pass resolves values available in the raw merged config, then filenames and resolved paths are added, and the second pass resolves templates that depend on those generated sections.

#### Why This Function Matters

This function is the root of config-driven execution. If its merge order, runtime metadata, filename map, path map, or hash behavior changes, downstream notebooks can write artifacts to different locations, use different stage parameters, or produce truth records that no longer explain the actual runtime settings.

#### Verification Method

Practical verification checks:

- Confirm the returned object has `data`, `config_hash`, and `source_files`.
- Confirm `source_files` contains `base.yaml`, the selected dataset file, the selected mode file, and the selected stage file.
- Confirm `data["runtime"]` contains the requested `stage`, `dataset`, `mode`, and `profile`.
- Confirm `data["filenames"]` includes dataset-specific filenames.
- Confirm `data["resolved_paths"]` includes expected roots such as `artifacts_root`, `truths_dir`, and stage data paths.
- Recompute a config hash from the returned `data` before `config_meta.config_hash` is inserted only if reproducing the source hash behavior exactly; the function computes the hash before storing it in the config.
- Pass a missing stage name and confirm `ConfigError` is raised by config file resolution.

### `export_config_snapshot`

#### Functional Purpose

`export_config_snapshot` writes a resolved config mapping to a YAML file and returns the destination path. It converts config values into plain serializable values before writing, so `Path`, mapping, list, and tuple values are represented in a stable YAML-friendly form.

#### Pipeline Context

Workflow references confirm config snapshots in Bronze, Silver, and Gold stages. Examples include Bronze config output, Silver Pre-EDA config output, Silver EDA config output, Gold preprocessing config output, and Gold cascade/anomaly-detection config output.

#### Inputs and Assumptions

Important inputs and assumptions:

- `config` is a mapping, typically `LoadedConfig.data` or the notebook's resolved `CONFIG`.
- `destination` is the intended snapshot file path.
- The destination parent directory may be absent; the function creates it.
- Values must be serializable after `_plain_config_dict` and `_plain_config_value` conversion.

#### Outputs and Return Contract

The function returns `destination` as a `Path`. The written YAML preserves input key order by using `sort_keys=False`.

#### Side Effects

Source-confirmed side effects:

- Creates `destination.parent` with `parents=True, exist_ok=True`.
- Writes a YAML file to `destination` with UTF-8 encoding.

#### Failure Behavior and Guardrails

Confirmed guardrails:

- Parent directories are created before writing.
- `Path` values are converted to strings before serialization.
- Mapping keys are converted to strings.

Source-confirmed explicit exception handling is not present in this function. File-system permission errors or serialization errors would propagate from `Path.mkdir`, `Path.open`, or `yaml.safe_dump`.

#### Lineage and Reproducibility Role

The snapshot gives reviewers a persisted copy of the resolved config used during a stage run. It complements truth records: truth records carry a compact config payload, while the snapshot can preserve the broader resolved config dictionary for inspection.

#### Why This Function Matters

Configuration is one of the main reproducibility controls in this project. Without a resolved config snapshot, a reviewer can see output files but may not be able to confirm which dataset, mode, stage parameters, path mappings, and optional settings produced them.

#### Verification Method

Practical verification checks:

- Confirm the returned value equals the requested destination path as a `Path`.
- Confirm the destination file exists after the function runs.
- Open the YAML file and confirm it contains expected top-level sections such as `runtime`, `dataset`, `versions`, `filenames`, `resolved_paths`, and `config_meta` when those are present in the input config.
- Confirm path-like values are written as strings.
- Confirm a nested config mapping remains nested rather than flattened.

### `build_truth_config_block`

#### Functional Purpose

`build_truth_config_block` extracts the subset of resolved configuration that truth records need for auditability. It does not copy the entire config. Instead, it builds a compact payload containing config identity, runtime context, version context, optional integration settings, lineage settings, and the current stage's parameter block.

#### Pipeline Context

Workflow references confirm `TRUTH_CONFIG = build_truth_config_block(CONFIG)` in Gold_02 and describe the same config block pattern in Gold_01, Gold_03b, Gold_03c, Gold_04, and other truth-record flows. The output is used as part of truth record construction, usually through a `config_snapshot` section in the truth payload.

#### Inputs and Assumptions

Important assumptions:

- `config["config_meta"]["config_hash"]` exists.
- `config["config_meta"]["source_files"]` exists.
- `config["runtime"]` exists and contains a `stage` key.
- `config["versions"]` exists.
- The stage-specific config block is available at `config[config["runtime"]["stage"]]`; if not, the returned `stage_params` defaults to `{}` only when `.get()` is reached with a valid runtime stage value.
- Optional blocks `wandb`, `execution`, and `lineage` are included if present and default to `{}` otherwise.

#### Outputs and Return Contract

The return value is a dictionary with:

- `config_hash`
- `source_files`
- `runtime`
- `versions`
- `wandb`
- `execution`
- `lineage`
- `stage_params`

The `stage_params` value is selected from the stage key stored in `config["runtime"]["stage"]`.

#### Side Effects

No file writes, directory creation, mutation, database access, or external service calls are performed by this function.

#### Failure Behavior and Guardrails

The function relies on direct dictionary indexing for `config_meta`, `runtime`, and `versions`. Missing required keys raise normal mapping exceptions such as `KeyError`. Optional blocks use `.get(..., {})`.

#### Lineage and Reproducibility Role

This function is the bridge between configuration resolution and truth-record auditability. It carries the config hash and source file list into a truth payload so a reviewer can connect a dataframe or artifact back to the resolved configuration that produced it. It also preserves the runtime block and stage parameters, which are the most relevant pieces for understanding stage-specific behavior.

#### Why This Function Matters

Truth records should be compact enough to inspect but complete enough to explain a run. This helper enforces that balance. If it omits the config hash, source files, runtime mode, versions, or stage parameters, downstream lineage review becomes weaker.

#### Verification Method

Practical verification checks:

- Confirm the returned dictionary contains the eight expected keys.
- Confirm `config_hash` matches `config["config_meta"]["config_hash"]`.
- Confirm `source_files` matches `config["config_meta"]["source_files"]`.
- Confirm `stage_params` comes from the config section named by `config["runtime"]["stage"]`.
- Confirm optional `wandb`, `execution`, and `lineage` blocks are `{}` when missing.

## Cross-Function Relationships

- `load_pipeline_config` produces the resolved config, config hash, source-file list, filenames, and resolved paths used by the rest of the core utility layer.
- `export_config_snapshot` persists that resolved config for review.
- `build_truth_config_block` extracts the compact lineage-relevant subset of the resolved config for truth records.
- The config block produced here is used with truth helpers in `utils/core/truths.py`; the resolved paths produced here are used by artifact helpers in `utils/core/artifacts.py`.

## Source-Limited Items

- Direct consumer coverage is strongest for Gold notebooks in the module reference and workflow references. Broader Bronze and Silver use is confirmed through notebook context and workflow references, but direct `load_pipeline_config` calls are not always visible in the final-facing references.
- The exact caller-side policy for when to skip `export_config_snapshot` is notebook-controlled and not determined solely from this source file.
