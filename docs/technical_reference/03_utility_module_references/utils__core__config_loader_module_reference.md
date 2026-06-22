# Utility Module Reference: `utils/core/config_loader.py`

## Module Purpose

This module loads, merges, resolves, and snapshots the YAML pipeline configuration used across notebook stages.

## Pipeline Role

- Stage support: Core / cross-stage
- Primary responsibility: This module loads, merges, resolves, and snapshots the YAML pipeline configuration used across notebook stages.

## Primary Consumers

`EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `ConfigError` | Raised when a configuration file is missing or structurally invalid. | deep |
| `LoadedConfig` | Container for the resolved pipeline config. | deep |
| `_SafeFormatDict` | Dictionary that preserves unknown format placeholders. | short |
| `_read_yaml` | Read a YAML mapping from disk and raise ConfigError for invalid input. | deep |
| `_deep_merge` | Recursively merge override values into a deep copy of base. | short |
| `_flatten_for_templates` | Flatten nested config values into dotted and short template keys. | short |
| `_render_template_string` | Render template placeholders using literal key matching. | deep |
| `_render_templates` | Recursively render template strings within dictionaries and lists. | short |
| `_build_filename_map` | Build canonical dataset, artifact, model, and ledger filenames. | deep |
| `_build_path_map` | Build resolved project paths from roots, dataset metadata, and filenames. | deep |
| `_normalize_mode_overrides` | Apply runtime mode-specific override blocks to the merged config. | short |
| `_resolve_config_file` | Resolve a config path with optional .yaml or .yml suffix discovery. | deep |
| `load_pipeline_config` | Load, merge, render, and enrich a stage-specific pipeline config. | deep |
| `_plain_config_value` | Convert config values into plain Python values suitable for YAML/JSON export. | deep |
| `_plain_config_dict` | Convert a config mapping into a plain string-keyed dictionary. | deep |
| `export_config_snapshot` | Write a resolved config snapshot to YAML and return the destination path. | deep |
| `build_truth_config_block` | Build the compact config payload embedded in a truth record. | deep |
| `set_wandb_dir_from_config` | Create the configured W&B directory and set the WANDB_DIR environment variable. | deep |
| `_SafeFormatDict.__missing__` | Preserves unresolved template placeholders by returning the original `{key}` token. | deep |

## Configuration Dependencies

- Resolved pipeline configuration dictionaries and YAML-derived stage settings.
- Environment variables where runtime mode or optional integration behavior is configured.
- Project root, resolved path mappings, and artifact directory configuration.
- Dataset, run, stage, or recipe identifiers used for traceability.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_read_yaml` | `path` | Read a YAML mapping from disk and raise ConfigError for invalid input. |
| `_deep_merge` | `base, override` | Recursively merge override values into a deep copy of base. |
| `_flatten_for_templates` | `data, prefix` | Flatten nested config values into dotted and short template keys. |
| `_render_template_string` | `template, context` | Render template placeholders using literal key matching. |
| `_render_templates` | `obj, context` | Recursively render template strings within dictionaries and lists. |
| `_build_filename_map` | `cfg` | Build canonical dataset, artifact, model, and ledger filenames. |
| `_build_path_map` | `project_root, cfg, filenames` | Build resolved project paths from roots, dataset metadata, and filenames. |
| `_normalize_mode_overrides` | `cfg` | Apply runtime mode-specific override blocks to the merged config. |
| `_resolve_config_file` | `path` | Resolve a config path with optional .yaml or .yml suffix discovery. |
| `load_pipeline_config` | `*, config_root, stage, dataset, mode, profile, project_root` | Load, merge, render, and enrich a stage-specific pipeline config. |
| `_plain_config_value` | `value` | Convert config values into plain Python values suitable for YAML/JSON export. |
| `_plain_config_dict` | `config` | Convert a config mapping into a plain string-keyed dictionary. |
| `export_config_snapshot` | `config, destination` | Write a resolved config snapshot to YAML and return the destination path. |
| `build_truth_config_block` | `config` | Build the compact config payload embedded in a truth record. |

## Side Effects

- Source includes directory creation; helpers can create configured output directories.
- Source includes W&B integration points; behavior depends on the project's optional W&B configuration.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.
- W&B: Source references optional Weights & Biases helper behavior.

## Failure Behavior

- Source raises `ConfigError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Module Importance

This module matters because shared configuration, paths, logging, ledger, truth, artifact, and optional W&B behavior reduce duplicated notebook setup.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
