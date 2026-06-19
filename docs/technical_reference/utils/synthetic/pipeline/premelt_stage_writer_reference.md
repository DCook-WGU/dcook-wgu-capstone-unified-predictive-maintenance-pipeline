# Synthetic Utility Reference: premelt_stage_writer.py

Source path:

`utils/synthetic/pipeline/premelt_stage_writer.py`

## Purpose

Builds the premelt synthetic observation stage before timestamping and message shaping.

## Pipeline Role

Pipeline-side utility used after synthetic data generation to stage observations, prepare queue records, interact with Kafka/PostgreSQL helpers, rebuild rows, align outputs, or hand data into Bronze-facing workflows.

## Functions and Classes

| Type | Name | Parameters | Purpose |
|---|---|---|---|
| Function | `_get_table_columns` | `engine` | Return source table columns in database ordinal order. |
| Function | `_validate_source_columns` | `columns, required_columns` | Validate that the synthetic stream table has required premelt inputs. |
| Function | `_build_select_sql` | `` | Build the SQL SELECT that assigns observation identity and metadata. |
| Function | `_write_stage_sql_native` | `engine` | Create, append to, or fail on the premelt target table in Postgres. |
| Function | `scalar_to_int` | `value, name` | Convert a scalar SQL result to int and reject missing values. |
| Function | `dataframe_row_count_to_int` | `dataframe` | Return a count value from a one-row dataframe as a plain int. |
| Function | `build_observations_premelt_stage` | `engine` | Build the premelt stage directly inside Postgres. This preserves the same notebook call signature as the pandas version, but avoids a full wide-table round-trip through pandas. The stage assigns dataset/run/asset identity, generated row IDs, observation indexes, and producer metadata while keeping sensor columns wide for the next stage. |
| Function | `validate_observations_premelt_stage` | `engine` | Return row-count and identity checks for the premelt stage table. |

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

- Writes or prepares files/artifacts such as CSV, Parquet, JSON, or metadata outputs.

## Downstream Usage

- `notebooks/synthetic/synthetic_02_build_premilt_observations_stage.ipynb`
- `notebooks/synthetic/synthetic_all_in_one_wip_v1.ipynb`
- `notebooks/synthetic/synthetic_pipeline_condensed-02_03.ipynb`

## Behavior Notes

- This document describes existing utility behavior for project technical reference.
- It does not change source code, function signatures, imports, SQL behavior, configuration behavior, logging behavior, ledger behavior, or artifact behavior.
- Runtime behavior should be validated through the notebooks or targeted pipeline tests that use this module.
