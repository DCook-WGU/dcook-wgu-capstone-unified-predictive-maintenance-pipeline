# Utility Module Reference: `utils/core/notebook_context.py`

## Module Purpose

This module initializes common notebook runtime context such as project root detection, imports, configuration, logging, and ledger setup.

## Pipeline Role

- Stage support: Core / cross-stage
- Primary responsibility: This module initializes common notebook runtime context such as project root detection, imports, configuration, logging, and ledger setup.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `NotebookContext` | Container for resolved notebook configuration, logging, and ledger state. | short |
| `_require_mapping` | Return a copied mapping or raise TypeError for required config sections. | deep |
| `_optional_mapping` | Return a copied optional mapping, using an empty dict when omitted. | deep |
| `load_notebook_context` | Load the shared runtime context used by capstone notebooks. | deep |

## Configuration Dependencies

- Resolved pipeline configuration dictionaries and YAML-derived stage settings.
- Environment variables where runtime mode or optional integration behavior is configured.
- Project root, resolved path mappings, and artifact directory configuration.
- Dataset, run, stage, or recipe identifiers used for traceability.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_require_mapping` | `value, name` | Return a copied mapping or raise TypeError for required config sections. |
| `_optional_mapping` | `value, name` | Return a copied optional mapping, using an empty dict when omitted. |
| `load_notebook_context` | `*, stage, recipe_id, dataset, mode, profile, logger_name, logger_child_name, log_filename, log_level, overwrite_handlers` | Load the shared runtime context used by capstone notebooks. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.
- Source includes logger usage; helpers can emit project log messages.
- Source includes ledger handling; helpers can append or export ledger-style run metadata when called by notebook stages.
- Source includes W&B integration points; behavior depends on the project's optional W&B configuration.

## Artifact / SQL / File-System Interactions

- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.
- W&B: Source references optional Weights & Biases helper behavior.

## Failure Behavior

- Source raises `TypeError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Module Importance

This module matters because shared configuration, paths, logging, ledger, truth, artifact, and optional W&B behavior reduce duplicated notebook setup.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
