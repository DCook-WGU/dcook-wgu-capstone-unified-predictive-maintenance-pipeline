# Utility Module Reference: `utils/core/ledger.py`

## Module Purpose

This module records lightweight run decisions and stage events in a structured ledger that can be exported with artifacts.

## Pipeline Role

- Stage support: Core / cross-stage
- Primary responsibility: This module records lightweight run decisions and stage events in a structured ledger that can be exported with artifacts.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `Ledger` | Collect stage-level lineage entries and export them as JSON. | deep |
| `_now_utc_iso` | Return the current UTC timestamp as an ISO-8601 string. | short |
| `Ledger.add` | Append a structured ledger entry and optionally log it. | deep |
| `Ledger.write_json` | Write accumulated ledger entries to ``out_path`` as indented JSON. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_now_utc_iso` | `No explicit parameters` | Return the current UTC timestamp as an ISO-8601 string. |
| `Ledger.add` | `self, kind, step, message, why, consequence, data, logger` | Append a structured ledger entry and optionally log it. |
| `Ledger.write_json` | `self, out_path` | Write accumulated ledger entries to ``out_path`` as indented JSON. |

## Side Effects

- Source includes file-write calls; helpers can write configured files or artifacts when those paths are passed by the caller.
- Source includes directory creation; helpers can create configured output directories.
- Source includes logger usage; helpers can emit project log messages.
- Source includes ledger handling; helpers can append or export ledger-style run metadata when called by notebook stages.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories and file-write operations.

## Failure Behavior

- Not determined from available source

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`, `EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_02_Baseline_Modeling`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_04_Comparison`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`

## Module Importance

This module matters because shared configuration, paths, logging, ledger, truth, artifact, and optional W&B behavior reduce duplicated notebook setup.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
