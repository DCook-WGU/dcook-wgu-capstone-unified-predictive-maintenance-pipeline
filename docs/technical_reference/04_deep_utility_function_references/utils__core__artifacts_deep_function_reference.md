# `utils/core/artifacts.py` Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `utils/core/artifacts.py` that need deeper explanation than the module-level utility reference. The focus is standardized artifact directory construction and canonical artifact file-path creation.

## Source Grounding

Sources used:

- Active utility source: `utils/core/artifacts.py`
- Function inventory: `function_inventory.json`
- Scope plan: `technical_reference/03_utility_module_references/071e_scope_plan.md`
- Module reference: `technical_reference/03_utility_module_references/utils__core__artifacts_module_reference.md`
- Notebook workflow references under `technical_reference/01_notebook_workflow_references/`
- Project manual files under `technical_reference/00_project_manual/`
- Documentation standards under `docs_agent_pack/00_STANDARDS/`

The active utility source file is the source of truth for function behavior.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
|---|---|---|
| `build_artifact_dirs_from_config` | Converts resolved config artifact layout into standardized stage/dataset/family directory mappings. | Bronze, Silver, Gold artifact setup. |
| `artifact_file_path` | Builds a file path inside a named artifact directory. | Stage artifact writes and consistent output naming. |

## Module-Level Technical Context

`artifacts.py` provides path helpers that keep notebook outputs under consistent stage, dataset, family, and subdirectory layouts. The selected functions are the core bridge between resolved configuration and concrete artifact paths used by notebooks for config snapshots, summaries, metadata, lineage files, model outputs, and validation outputs.

## Deep Function References

### `build_artifact_dirs_from_config`

#### Functional Purpose

`build_artifact_dirs_from_config` reads a stage's `artifact_layout` section from the resolved config and returns a dictionary of standardized `Path` objects. It supports stage-level defaults, optional variant-specific layout overrides, explicit family overrides, explicit subdirectory overrides, and optional directory creation.

In project terms, it converts config into the directory structure where notebooks write stage outputs.

#### Pipeline Context

The workflow references confirm this helper in Gold_01, Gold_03a, Gold_03b, Gold_03c, and Gold_05. The module reference also lists Bronze_01, Gold_01, Gold_02, Gold_03a, Gold_03b, Gold_03c, Gold_04, and Gold_05 as consumers of artifact helpers. Silver workflow references confirm similar artifact directory creation patterns and config snapshot output, although not every reference names this exact function.

#### Inputs and Assumptions

Important inputs and assumptions:

- `config` must be a resolved configuration mapping.
- `stage_key` must exist as a top-level key in `config`.
- `config[stage_key]` must be a mapping.
- `config[stage_key]["artifact_layout"]` is expected when layout-driven directories are needed.
- `config["dataset"]["name"]` must exist and be non-blank.
- `config["resolved_paths"]["artifacts_root"]` must exist and be non-blank.
- If `variant` is provided, `config[stage_key]["artifact_layout"]["variants"][variant]` must exist and be a mapping.
- `family_override` supersedes the layout family when provided.
- `subdirs_override` supersedes the layout subdirectory list when provided.
- `create=True` means returned directories are created on disk through `build_artifact_dirs`.

#### Outputs and Return Contract

The return value is a dictionary mapping string keys to `Path` objects. It is the same structure returned by `build_artifact_dirs`:

- `stage_dataset_root`: `artifacts_root / stage / dataset_name`
- `root`: the stage/dataset root or family-specific root
- One key per configured subdirectory, each pointing under `root`

For variant layouts, the base layout is copied without the `variants` key, then the selected variant's layout values override the base layout.

#### Side Effects

Source-confirmed side effects:

- When `create=True`, the function creates all returned directories through `Path.mkdir(parents=True, exist_ok=True)`.
- When `create=False`, the function returns paths without creating directories.

#### Failure Behavior and Guardrails

Confirmed guardrails:

- Raises `KeyError` if `stage_key` is missing.
- Raises `TypeError` if expected config sections are not mappings.
- Raises `KeyError` if a requested variant is not defined.
- Raises `KeyError` if artifact stage cannot be resolved from layout, stage config, or runtime config.
- Raises `KeyError` or `ValueError` for missing or blank required values such as `dataset.name` and `resolved_paths.artifacts_root`.
- Raises `TypeError` if subdirectory configuration is not a string, bytes, list, or tuple.
- Delegated `build_artifact_dirs` validation raises `ValueError` for blank stage or dataset names and for subdirectory keys that conflict with reserved keys.

#### Lineage and Reproducibility Role

Artifact directories are part of the project's lineage design because truth records and summaries often record artifact paths written under these locations. A stable layout lets downstream notebooks and manual reviewers know where to find config snapshots, lineage records, metadata, summaries, model files, and plots for a stage.

Variant handling is especially important for Gold cascade stages because default, tuned, and stage-three-improved cascade outputs use different artifact families while sharing the same high-level stage.

#### Why This Function Matters

Without this helper, each notebook could construct output directories slightly differently. That would increase handoff risk, make truth-record artifact paths harder to compare, and make evaluator review of stage outputs less consistent. This function centralizes the mapping from config to file-system layout.

#### Verification Method

Practical verification checks:

- Call with a resolved config and known `stage_key`, then confirm the return value includes `stage_dataset_root` and `root`.
- Confirm configured subdirectory keys such as `config`, `lineage`, `metadata`, `summaries`, or stage-specific directories appear when configured.
- Confirm each returned value is a `Path`.
- Confirm directories exist when `create=True`.
- Confirm directories are not required to exist when `create=False`.
- Call with an invalid variant and confirm a `KeyError`.
- Confirm `family_override` changes the returned `root` path without changing the configured stage or dataset root.

### `artifact_file_path`

#### Functional Purpose

`artifact_file_path` appends a file name to a named artifact directory from a directory mapping. It is a small helper, but it enforces that artifact files are built from known directory keys rather than arbitrary path strings.

#### Pipeline Context

The module reference confirms broad Bronze and Gold use of artifact helpers. Workflow references show notebooks writing config snapshots, summaries, metadata, lineage files, and model-support artifacts into directories created by artifact helper functions. This function supports that pattern by constructing individual file paths under those directories.

#### Inputs and Assumptions

Important inputs and assumptions:

- `artifact_dirs` is a mapping such as the output of `build_artifact_dirs` or `build_artifact_dirs_from_config`.
- `subdir_key` must exist in `artifact_dirs`.
- `file_name` is appended as the final path component.
- The function does not validate whether `file_name` is blank, contains separators, or matches a configured filename key.

#### Outputs and Return Contract

The return value is:

`Path(artifact_dirs[subdir_key]) / file_name`

It returns a `Path` object and does not write the file.

#### Side Effects

No file writes, directory creation, config mutation, or dataframe mutation are performed by this function.

#### Failure Behavior and Guardrails

Confirmed guardrail:

- Raises `KeyError` if `subdir_key` is not present in `artifact_dirs`. The error includes the requested key and the available keys.

No source-confirmed guardrail validates `file_name`.

#### Lineage and Reproducibility Role

This helper contributes to lineage by keeping output files anchored to named artifact directories. When notebooks use a named directory key such as `config`, `lineage`, `metadata`, or `summaries`, file paths remain aligned with the directory layout produced from config.

#### Why This Function Matters

The risk this helper reduces is silent path drift. A notebook can still choose the wrong filename, but it cannot use an unknown artifact subdirectory key without getting a clear exception. That makes configured artifact directory mappings easier to verify.

#### Verification Method

Practical verification checks:

- Build an artifact directory mapping with `build_artifact_dirs_from_config`.
- Call `artifact_file_path(artifact_dirs, "config", "resolved_config.yaml")` when `config` exists.
- Confirm the returned path's parent is `artifact_dirs["config"]`.
- Call with a missing key and confirm `KeyError` lists available keys.
- Confirm no file is created until a separate write operation uses the returned path.

## Cross-Function Relationships

- `build_artifact_dirs_from_config` uses the resolved config produced by `load_pipeline_config` to build stage-aware directories.
- `artifact_file_path` uses the directory mapping from `build_artifact_dirs_from_config` or related artifact helpers to build individual file paths.
- `export_config_snapshot` in `config_loader.py` commonly writes to a path created from artifact directories.
- Truth helpers in `truths.py` often record artifact paths under truth payload sections after notebooks write files into these directories.

## Source-Limited Items

- The exact set of subdirectory keys depends on active YAML config, not only on this source file.
- Some workflow references describe artifact directory creation without naming `build_artifact_dirs_from_config`; exact call sites should be verified in active notebook source when a function-specific consumer list is required.
- `artifact_file_path` does not confirm downstream file writes by itself; writes happen in caller code.
