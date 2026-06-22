# Utility Module Reference: `utils/synthetic/generator/run.py`

## Module Purpose

This module orchestrates synthetic generator setup and execution from configuration.

## Pipeline Role

- Stage support: Synthetic generator
- Primary responsibility: This module orchestrates synthetic generator setup and execution from configuration.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `parse_args` | Parse CLI options for the standalone synthetic generation stage. | deep |
| `main` | Run the generator stage from config, parent truth, and CLI overrides. | deep |

## Configuration Dependencies

- Resolved pipeline configuration dictionaries and YAML-derived stage settings.
- Environment variables where runtime mode or optional integration behavior is configured.
- Project root, resolved path mappings, and artifact directory configuration.
- Dataset, run, stage, or recipe identifiers used for traceability.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `parse_args` | `No explicit parameters` | Parse CLI options for the standalone synthetic generation stage. |
| `main` | `No explicit parameters` | Run the generator stage from config, parent truth, and CLI overrides. |

## Side Effects

- Source includes directory creation; helpers can create configured output directories.
- Source includes logger usage; helpers can emit project log messages.
- Source includes W&B integration points; behavior depends on the project's optional W&B configuration.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.
- W&B: Source references optional Weights & Biases helper behavior.

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Module Importance

This module matters because the synthetic subsystem depends on repeatable staging, truth handling, streaming handoff, or final alignment behavior to make controlled pump telemetry tests explainable.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
