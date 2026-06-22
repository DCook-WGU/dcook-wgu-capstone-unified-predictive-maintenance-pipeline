# Utility Module Reference: `utils/core/logging_profiler.py`

## Module Purpose

This module summarizes dataframe shape, columns, missingness, and profiling details through the project logger.

## Pipeline Role

- Stage support: Core / cross-stage
- Primary responsibility: This module summarizes dataframe shape, columns, missingness, and profiling details through the project logger.

## Primary Consumers

`EDA_Notebook_Pump_Bronze_01_Preprocessing`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `profile_dataframe` | Log dataframe shape, memory, dtypes, and head sample. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `profile_dataframe` | `dataframe, logger, artifacts_dir, head` | Log dataframe shape, memory, dtypes, and head sample. |

## Side Effects

- Source includes file-write calls; helpers can write configured files or artifacts when those paths are passed by the caller.
- Source includes directory creation; helpers can create configured output directories.
- Source includes logger usage; helpers can emit project log messages.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories and file-write operations.

## Failure Behavior

- Not determined from available source

## Downstream Usage

`EDA_Notebook_Pump_Bronze_01_Preprocessing`

## Module Importance

This module matters because shared configuration, paths, logging, ledger, truth, artifact, and optional W&B behavior reduce duplicated notebook setup.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
