# Synthetic Utility Reference: melt_stage_writer_original.py

Source path:

`utils/synthetic/pipeline/melt_stage_writer_original.py`

## Status

Legacy / retained reference.

This module is documented because it remains in the repository, but it should not be treated as the preferred current implementation unless it is explicitly referenced by active notebooks, utility imports, or pipeline workflows.

## Purpose

Retains an original melt-stage helper variant for comparison or legacy reference.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `_build_sensor_columns` | `n_sensors` | No docstring summary is available; review the function body for detailed behavior. |
| Function | `_validate_source_columns` | `dataframe, required_columns` | No docstring summary is available; review the function body for detailed behavior. |
| Function | `_extract_sensor_index` | `sensor_name_series` | No docstring summary is available; review the function body for detailed behavior. |
| Function | `_build_message_sequence_index_with_rng` | `` | Build a randomized 0..(n_sensors-1) sequence for each observation using one shared RNG so chunking stays deterministic across the full run. |
| Function | `build_sensor_messages_stage` | `engine` | Build the long-format sensor message stage from the premelt observation stage in chunks instead of loading/melting the full table at once. |
| Function | `validate_sensor_messages_stage` | `engine` | No docstring summary is available; review the function body for detailed behavior. |

## Configuration Dependencies

- Uses SQL schema or table name settings.
- Uses dataset identity values.
- Uses run or recipe identity values.

## Inputs and Outputs

Key inputs:
- Configuration values, dataset identity, run identity, or recipe identity
- Database engine, schema, table, or SQL runtime context
- Pandas dataframes or dataframe-like stage inputs

Key outputs:
- Dataframes or transformed stage outputs
- SQL table rows, status updates, or database-stage records

## Logging, Ledger, and Artifact Behavior

### Logging

- No direct logger calls detected in this module.

### Ledger

- No direct ledger behavior detected in this module.

### SQL/database

- Uses SQL, PostgreSQL, engine, table, or database write/read behavior.

### Artifacts

- No direct artifact write pattern detected in this module.

## Downstream Usage

- No direct notebook reference was detected by static search. The module may be called through another utility or retained for support use.

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
