# utils/database/layer_postgres.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `utils/database/layer_postgres.py` that need deeper explanation than the 071d module-level reference. The selected functions provide the shared dataframe read/write boundary used by layer-oriented PostgreSQL handoffs.

## Source Grounding

Sources used:

- `utils/database/layer_postgres.py`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__database__layer_postgres_module_reference.md`
- `function_inventory.json`
- `technical_reference/01_notebook_workflow_references/`
- `technical_reference/00_project_manual/`
- Documentation standards under `docs_agent_pack/00_STANDARDS/`

The active utility source file is the source of truth.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
| -------- | -------------- | ------------------------ |
| `write_layer_dataframe` | Shared dataframe-to-PostgreSQL layer writer | Bronze, Silver, Gold, and synthetic layer handoffs |
| `read_layer_dataframe` | Shared PostgreSQL layer table reader | Notebook and pipeline reads from persisted layer tables |

## Module-Level Technical Context

`layer_postgres.py` centralizes common layer table I/O behavior that would otherwise be repeated across notebooks. The assigned functions route dataframes to sanitized schema/table names, preserve caller-selected write/read options, and use shared database helpers for schema creation, table existence checks, SQL reads, and conservative SQLAlchemy dtype inference.

These functions are lower-level than the project-specific Medallion writers. They do not define Bronze, Silver, or Gold table contracts by themselves; instead, they provide a reusable persistence pattern for dataframe-oriented layer artifacts.

## Deep Function References

### `write_layer_dataframe`

#### Functional Purpose

`write_layer_dataframe` writes a pandas dataframe to a PostgreSQL table under a requested schema and table naming contract. It is the shared layer writer for cases where a notebook or pipeline already has a dataframe and needs a consistent SQL handoff without building a module-specific insert statement.

#### Pipeline Context

Project manual and workflow references confirm this helper as part of the database-backed layer handoff pattern across Bronze, Silver, Gold, and synthetic contexts. Specific notebook usage varies by workflow; some notebooks import the helper directly, while others use more specialized SQL writer functions.

#### Inputs and Assumptions

Important inputs include the SQLAlchemy `engine`, source `dataframe`, target `schema`, and either an explicit `table_name` or a `dataset_name` used to build the table name. Optional `layer`, `artifact_name`, and `include_layer_prefix_in_table_name` settings affect derived table naming.

The function expects a pandas dataframe. Empty dataframes are rejected unless `allow_empty=True`. The `if_exists` mode must be one of `fail`, `replace`, or `append`. Column names are sanitized before writing. SQLAlchemy dtypes are inferred from the dataframe after sanitization, with caller-supplied dtype overrides taking precedence.

#### Outputs and Return Contract

The function returns the final sanitized table name used for the write. It does not return the written dataframe or row count.

#### Side Effects

Confirmed side effects are:

- Creates the target schema when needed through the shared schema helper.
- Copies the source dataframe before modifying column names for SQL output.
- Writes to PostgreSQL through `pandas.DataFrame.to_sql`.
- Uses the caller-selected `if_exists` mode for table creation, replacement, append, or failure behavior.
- Emits an informational logger message when a logger is supplied.

#### Failure Behavior and Guardrails

The function raises `TypeError` if the input is not a pandas dataframe. It raises `ValueError` when the dataframe is empty and `allow_empty` is false, when `if_exists` is invalid, or when no `dataset_name` is provided for derived table naming. The supporting table-name helper raises `ValueError` when a layer prefix is requested without a layer value. Database and `to_sql` failures propagate to the caller.

#### Lineage and Reproducibility Role

The function preserves lineage columns only if they already exist in the dataframe. It does not add `dataset_id`, `run_id`, truth hashes, parent truth hashes, or pipeline metadata on its own. Its reproducibility role is to standardize schema creation, table naming, column sanitization, dtype inference, chunked writes, and write-mode behavior.

#### Why This Function Matters

Layer SQL writes are high-risk places for notebook drift because each notebook could otherwise make slightly different choices about schema creation, table names, column normalization, and write modes. This helper provides a common path for dataframe persistence when a specialized writer is not needed.

#### Verification Method

- Confirm the returned table name matches the expected explicit or derived table name after sanitization.
- Confirm the target schema and table exist after writing.
- Confirm row count and column names in PostgreSQL match the intended dataframe output.
- Confirm empty dataframe writes fail unless `allow_empty=True`.
- Confirm invalid `if_exists` values raise the expected error.

### `read_layer_dataframe`

#### Functional Purpose

`read_layer_dataframe` reads a PostgreSQL layer table into a pandas dataframe using the same schema and table naming pattern as the writer. It provides optional projection, filtering, ordering, and limit behavior for notebook or pipeline reads.

#### Pipeline Context

Project manual and workflow references confirm shared layer reads as part of the SQL-backed Medallion handoff pattern. Bronze-to-Silver and broader layer readback contexts are source-confirmed at the documentation level, while specific notebook usage varies by stage.

#### Inputs and Assumptions

Important inputs include the SQLAlchemy `engine`, target `schema`, and either an explicit `table_name` or a `dataset_name` used for derived table naming. Optional `layer`, `artifact_name`, and `include_layer_prefix_in_table_name` settings mirror the table naming behavior used by `write_layer_dataframe`.

Optional `columns` are sanitized and quoted for the `SELECT` projection. Optional `where_clause`, `order_by`, `params`, and `limit` values are passed into the generated SQL read pattern. Caller-provided `where_clause` and `order_by` text remain caller responsibilities; the function sanitizes schema, table, and projected column names but does not parse arbitrary SQL fragments.

#### Outputs and Return Contract

The function returns the dataframe produced by the shared SQL read helper. The returned dataframe shape and columns depend on the target table, selected columns, filter clause, ordering, and limit.

#### Side Effects

The function performs a SQL read and returns a dataframe. No write side effects are confirmed from available source.

#### Failure Behavior and Guardrails

The function raises `ValueError` when no `dataset_name` is provided for derived table naming. If `require_exists=True` and the resolved table is absent, it raises `FileNotFoundError`. Database read failures propagate from the shared SQL read helper.

#### Lineage and Reproducibility Role

The function does not add or transform lineage fields. It supports reproducibility by using the same table-name resolution as the writer and by allowing callers to specify deterministic projections, filters, ordering, parameter bindings, and limits when reading persisted layer outputs.

#### Why This Function Matters

A shared layer reader reduces the chance that notebooks reconstruct table names or SQL reads inconsistently. It gives downstream stages a repeatable way to retrieve persisted layer data without duplicating schema/table resolution logic.

#### Verification Method

- Confirm the resolved table name matches the corresponding writer naming pattern.
- Confirm `require_exists=True` raises `FileNotFoundError` for a missing table.
- Confirm selected columns are returned when `columns` is provided.
- Confirm parameterized filters return the expected dataset/run subset when the caller supplies an appropriate `where_clause` and `params`.
- Confirm `limit` constrains the returned row count.

## Cross-Function Relationships

`write_layer_dataframe` and `read_layer_dataframe` share the same table-name derivation path when callers provide `dataset_name`, optional layer identity, optional artifact name, and layer-prefix settings. This relationship allows a dataframe written through the shared layer writer to be read back through the shared layer reader without duplicating table-name construction in notebook code.

The project-specific Medallion writers use more specialized SQL contracts for selected tables. The shared layer functions remain useful when the pipeline needs dataframe-oriented persistence rather than a table-specific insert and metadata logging contract.

## Source-Limited Items

- Direct use of `write_layer_dataframe` and `read_layer_dataframe` by every individual notebook was not determined from available source.
- Dataset/run filtering is not built into these functions except through caller-provided columns, `where_clause`, and `params`.
- Transaction behavior beyond the behavior of `pandas.DataFrame.to_sql` and the configured SQLAlchemy engine was not determined from available source.
