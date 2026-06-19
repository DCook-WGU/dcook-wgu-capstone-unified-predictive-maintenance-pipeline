# Synthetic Utility Reference: output_to_bronze_runner.py

Source path:

`utils/synthetic/pipeline/output_to_bronze_runner.py`

## Purpose

Coordinates late-stage synthetic output to Bronze handoff steps.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `run_synthetic_to_bronze_once` | `engine` | Run one rebuild, final-align, and Bronze handoff pass. This wrapper keeps the late synthetic pipeline stages in the same order: consumed messages become rebuilt observations, rebuilt observations become final-aligned rows, and final-aligned rows are handed to the Bronze input. |
| Function | `run_synthetic_to_bronze_loop` | `engine` | Repeat synthetic-to-Bronze passes until no stage writes new rows. |

## Configuration Dependencies

- Uses SQL schema or table name settings.
- Uses dataset identity values.
- Uses run or recipe identity values.

## Inputs and Outputs

Key inputs:
- Configuration values, dataset identity, run identity, or recipe identity
- Database engine, schema, table, or SQL runtime context

Key outputs:
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
