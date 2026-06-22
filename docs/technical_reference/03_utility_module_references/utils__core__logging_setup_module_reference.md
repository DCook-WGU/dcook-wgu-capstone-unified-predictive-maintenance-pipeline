# Utility Module Reference: `utils/core/logging_setup.py`

## Module Purpose

This module configures notebook-safe project loggers with consistent console and file handlers.

## Pipeline Role

- Stage support: Core / cross-stage
- Primary responsibility: This module configures notebook-safe project loggers with consistent console and file handlers.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`, `EDA_Notebook_Pump_Silver_01_PreEDA`, `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `configure_logging` | Configure a named logger with console and file handlers. | deep |
| `log_layer_paths` | Log common project paths, the current layer paths, and the previous layer paths when applicable. | deep |

## Configuration Dependencies

- Resolved pipeline configuration dictionaries and YAML-derived stage settings.
- Project root, resolved path mappings, and artifact directory configuration.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `configure_logging` | `name, log_file, level, overwrite_handlers` | Configure a named logger with console and file handlers. |
| `log_layer_paths` | `paths, current_layer, logger` | Log common project paths, the current layer paths, and the previous layer paths when applicable. |

## Side Effects

- Source includes directory creation; helpers can create configured output directories.
- Source includes logger usage; helpers can emit project log messages.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`, `EDA_Notebook_Pump_Silver_01_PreEDA`, `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Module Importance

This module matters because shared configuration, paths, logging, ledger, truth, artifact, and optional W&B behavior reduce duplicated notebook setup.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
