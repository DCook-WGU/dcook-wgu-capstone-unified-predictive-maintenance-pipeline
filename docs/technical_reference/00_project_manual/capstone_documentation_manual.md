# Capstone Documentation Manual

## Purpose of This Manual

This manual explains how to use the WGU Data Analytics Capstone technical reference package. Where the [Technical Reference Index](technical_reference_index.md) lists *what* documentation exists and links to it, this manual explains *how* the documentation is layered and *which path* a given reader should take. It is a guidance layer; it does not restate the content of the references and introduces no new technical claims.

The project documents an industrial pump telemetry anomaly-detection pipeline built on a medallion architecture (Bronze → Silver → Gold), implemented notebook-first with supporting `utils/` modules.

## Documentation Architecture

The package is organized into five documentation layers:

1. **Project manual and relationship maps** (`00_project_manual/`) — the pipeline shape, notebook dependencies, and artifact/table handoffs, plus this manual and the index.
2. **Notebook workflow references** (`01_notebook_workflow_references/`) — what each notebook does and how it fits the pipeline.
3. **Notebook deep technical references** (`02_notebook_deep_technical_references/`) — why the important technical decisions, validations, persistence, and lineage behaviors were designed as they were.
4. **Utility module references** (`03_utility_module_references/`) — broad, module-level coverage of the supporting `utils/` code.
5. **Deep utility function references** (`04_deep_utility_function_references/`) — curated deep dives on selected high-impact functions.

## How to Use the Documentation

### For a Pipeline Reviewer

1. [medallion_handoff_map.md](medallion_handoff_map.md) — get the Bronze → Silver → Gold data-flow picture.
2. [notebook_relationship_map.md](notebook_relationship_map.md) — see how notebooks relate and execute.
3. Workflow references in layer order (Bronze 01 → Silver 01/02a/02b → Gold 01–06B).
4. Deep technical references for any notebook that is unclear or high-risk.

### For a Technical Evaluator

1. [notebook_dependency_matrix.md](notebook_dependency_matrix.md) — confirm the dependency structure.
2. The Gold deep technical references (Gold 01–06B) — the modeling, validation, and lineage decisions.
3. Deep utility function references for the Gold, database, and synthetic side-effect areas (truths, medallion SQL writers, cascade modeling, generator/missingness).

### For Future Maintenance

1. [technical_reference_index.md](technical_reference_index.md) — locate the relevant documents.
2. The workflow reference for the notebook being edited.
3. The matching deep technical reference.
4. The relevant utility module and/or deep function references.

## Workflow vs Deep Technical References

- **Workflow references (01)** explain *what* each notebook does and how it fits into the pipeline — sections, inputs, outputs, and the step-by-step workflow.
- **Deep technical references (02)** explain *why* the important technical methods, transformations, validations, persistence behavior, and lineage behavior were designed as they were, with decision tables and failure-mode tables.

The two are complementary and should be read together. A deep reference being more detailed than its workflow counterpart is intended, not a discrepancy.

## Module References vs Deep Function References

- **Utility module references (03)** provide broad, module-level coverage across `utils/` (57 module references spanning core, database, medallion bronze/silver/gold, and synthetic generator/pipeline).
- **Deep utility function references (04)** provide deeper coverage for 18 selected high-impact functions.
- The 04 set is a curated subset of the broader 03 module references and is not expected to cover every function.

## Relationship Maps and Handoff Maps

| Map | Use it to … |
|---|---|
| [medallion_handoff_map.md](medallion_handoff_map.md) | See the high-level Bronze → Silver → Gold data flow |
| [notebook_relationship_map.md](notebook_relationship_map.md) | Understand notebook relationships and execution order |
| [notebook_dependency_matrix.md](notebook_dependency_matrix.md) | Check which notebooks feed or depend on a given notebook |
| [artifact_and_table_handoff_map.md](artifact_and_table_handoff_map.md) | Trace a specific artifact or SQL table across stages |

The handoff and relationship maps are project-level relationship aids built primarily from the workflow references. The deep technical references use stricter source-confirmed phrasing and mark some direct file/table dependencies as not determinable from a single notebook's source alone. When the two differ, the map states the project-level relationship and the deep reference states the conservatively source-verified position; both are correct within their evidentiary scope.

## Coverage Summary

- **13** notebook workflow references (Bronze 01; Silver 01, 02a, 02b; Gold 01, 02, 03a, 03b, 03c, 04, 05, 06A, 06B).
- **13** notebook deep technical references (matching 1:1 by stem).
- **4** project manual relationship/handoff files (plus this manual and the index).
- **57** utility module references.
- **18** curated deep utility function references.

## Maintenance Guidance

- When a notebook changes, update **both** its workflow reference and its deep technical reference if the change affects role, inputs, outputs, validation, persistence, or lineage.
- When a utility function changes, update its module reference and, where the function is covered, its deep function reference.
- When notebook dependencies change, update the relationship maps.
- When artifact or SQL behavior changes, update the handoff maps.
