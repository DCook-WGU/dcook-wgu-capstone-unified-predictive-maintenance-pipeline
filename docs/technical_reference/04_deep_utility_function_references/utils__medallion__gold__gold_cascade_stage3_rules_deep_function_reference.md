# utils/medallion/gold/gold_cascade_stage3_rules.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers the selected high-value function from `gold_cascade_stage3_rules.py` that needs deeper explanation than the module-level utility reference. The focus is Stage 3 rule confirmation: combining primary breach evidence, secondary corroboration, persistence evidence, and drift evidence into a final rule-based flag.

## Source Grounding

Sources used:

- `utils/medallion/gold/gold_cascade_stage3_rules.py`
- `function_inventory.json`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__medallion__gold__gold_cascade_stage3_rules_module_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling_code_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling_code_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling_code_reference.md`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling_deep_technical_reference.md`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling_deep_technical_reference.md`
- `technical_reference/00_project_manual/notebook_relationship_map.md`
- `technical_reference/00_project_manual/notebook_dependency_matrix.md`

The active utility source file is the source of truth for function behavior. Workflow, deep technical, and manual references provide consumer and handoff context only.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
|---|---|---|
| `compose_stage3_decision` | Combines Stage 3 evidence columns into a final confirmation flag. | Rule-based confirmation after statistical cascade candidate generation. |

## Module-Level Technical Context

`gold_cascade_stage3_rules.py` contains rule helpers used after statistical scoring has produced candidate anomaly evidence. Lower-level helpers compute primary breach counts, secondary group breach counts, persistence flags, and drift flags. The assigned function, `compose_stage3_decision`, consumes those evidence columns and produces the final rule-confirmation column. It does not fit models, write artifacts, update SQL, update ledgers, register W&B artifacts, or create truth records directly.

## Deep Function References

### `compose_stage3_decision`

#### Functional Purpose

`compose_stage3_decision` combines Stage 3 rule evidence into a binary confirmation flag. It checks whether primary breach counts and secondary breach counts meet configured minimums, then optionally requires either persistence or drift evidence before marking a row as confirmed.

#### Pipeline Context

The Stage 3 rule layer sits after the broad and narrowed statistical cascade stages. Manual and workflow references confirm that Gold_03a, Gold_03b, and Gold_03c use rule-based Stage 3 concepts in their cascade outputs. Direct notebook invocation of this exact shared utility function is Not determined from available source.

#### Inputs and Assumptions

- `dataframe` must contain the configured evidence columns.
- `primary_breach_column` defaults to `stage3_primary_breach_count`.
- `secondary_breach_column` defaults to `stage3_secondary_breach_count`.
- `persistence_column` defaults to `stage3_persistence_flag`.
- `drift_column` defaults to `stage3_drift_flag`.
- `min_primary_breaches` defaults to `1`; `min_secondary_breaches` defaults to `1`.
- `require_persistence_or_drift` defaults to `True`: the final decision requires both breach conditions plus at least one of persistence or drift.
- All four evidence columns are coerced via `pd.to_numeric(..., errors="coerce").fillna(0)` before comparison. A missing or non-numeric column resolves to zero rather than raising, so it will not satisfy any threshold.
- The decision logic applied to each row: `primary_ok = primary_breach_column >= min_primary_breaches`, `secondary_ok = secondary_breach_column >= min_secondary_breaches`. If `require_persistence_or_drift`: `confirmed = primary_ok & secondary_ok & (persistence_ok | drift_ok)`. Otherwise: `confirmed = primary_ok & secondary_ok`.

#### Outputs and Return Contract

The function returns a two-element tuple:

1. A dataframe copy with `output_column` (default `stage3_confirmed_flag`) added as an integer column: `1` for confirmed anomaly, `0` otherwise.
2. An info dictionary with keys: `output_column`, `min_primary_breaches`, `min_secondary_breaches`, `require_persistence_or_drift`, `positive_count` (integer count of confirmed rows).

The output flag is an integer representation of the boolean confirmation rule.

#### Side Effects

The function creates a dataframe copy and writes the output flag column to that copy. No mutation of the caller's dataframe and no external side effects are confirmed from available source.

#### Failure Behavior and Guardrails

- The active source does not perform explicit missing-column checks before selecting evidence columns; missing evidence columns will fail through pandas column access.
- Evidence values that cannot be converted to numeric are coerced, filled as zero, and therefore do not satisfy rule thresholds.
- If `require_persistence_or_drift` is true, rows must meet both breach-count requirements and at least one of persistence or drift.
- If `require_persistence_or_drift` is false, rows only need to meet the primary and secondary breach requirements.

#### Lineage and Reproducibility Role

The returned info dictionary records the decision thresholds and positive-row count. The output flag gives downstream comparison and validation code a stable Stage 3 confirmation column. The function does not attach dataset IDs, run IDs, truth hashes, artifact paths, or model artifact references directly.

#### Why This Function Matters

Stage 3 is the rule-confirmation layer that keeps final cascade alerts from relying only on statistical outlier scores. This function defines the logical gate between evidence signals and a final confirmed flag, so changes to its rule composition can directly change cascade alert counts and model comparison metrics.

#### Verification Method

- Build a small dataframe with known evidence values and confirm the output flag matches the expected boolean rule.
- Confirm rows below either breach-count threshold are not flagged.
- Confirm `require_persistence_or_drift=True` requires at least one of those temporal signals.
- Confirm `positive_count` equals the sum of the output flag column.
- Confirm the input dataframe is unchanged after the call.

## Cross-Function Relationships

- `compute_primary_breach_count` and `compute_secondary_breach_count` create breach-count evidence columns consumed by `compose_stage3_decision`.
- `compute_persistence_flag` and `compute_drift_flag` create temporal evidence columns consumed by `compose_stage3_decision`.
- `compose_stage3_decision` turns those evidence signals into a final Stage 3 confirmation flag.
- Workflow references confirm the broader Stage 3 rule-confirmation concept in Gold_03a, Gold_03b, and Gold_03c, but direct invocation of this exact helper in notebook source is Not determined from available source.

## Source-Limited Items

- Direct notebook invocation of `compose_stage3_decision` is Not determined from available source.
- Notebook-local Stage 3 formulas and weighted operating-mode searches in Gold_03b and Gold_03c are not implemented by this selected utility function.
- SQL, W&B, ledger, truth-record, artifact-write, model persistence, dataset/run identity, and parent-truth behavior are Not determined from available source for this function.
