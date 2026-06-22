# Utility Module Reference: `utils/core/paths.py`

## Module Purpose

This module resolves project-relative paths and root discovery for notebooks and utilities.

## Pipeline Role

- Stage support: Core / cross-stage
- Primary responsibility: This module resolves project-relative paths and root discovery for notebooks and utilities.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `ProjectPaths` | Centralized project path map used by notebooks, scripts, and utilities. | deep |
| `find_project_root` | Resolve the capstone project root from a notebook, script, or container path. | deep |
| `get_paths` | Resolve the project root and return standardized project directory paths. | deep |

## Configuration Dependencies

- Resolved pipeline configuration dictionaries and YAML-derived stage settings.
- Environment variables where runtime mode or optional integration behavior is configured.
- Project root, resolved path mappings, and artifact directory configuration.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `find_project_root` | `start_path` | Resolve the capstone project root from a notebook, script, or container path. |
| `get_paths` | `project_root` | Resolve the project root and return standardized project directory paths. |

## Side Effects

- Source includes logger usage; helpers can emit project log messages.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Module Importance

This module matters because shared configuration, paths, logging, ledger, truth, artifact, and optional W&B behavior reduce duplicated notebook setup.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
