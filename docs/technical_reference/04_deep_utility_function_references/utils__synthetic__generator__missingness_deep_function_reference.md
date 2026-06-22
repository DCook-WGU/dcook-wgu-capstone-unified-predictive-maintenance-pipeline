# utils/synthetic/generator/missingness.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `missingness.py` that need deeper explanation than the 071d module-level reference. The selected functions reconstruct missingness replay settings from truth payloads, convert missingness rates into present-count targets, and apply clustered missing-value masks to synthetic telemetry.

## Source Grounding

Sources used:

- `utils/synthetic/generator/missingness.py`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__synthetic__generator__missingness_module_reference.md`
- `function_inventory.json`
- `technical_reference/01_notebook_workflow_references/`
- `technical_reference/00_project_manual/`
- Documentation standards under `docs_agent_pack/00_STANDARDS/`

The active utility source file is the source of truth.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
| -------- | -------------- | ------------------------ |
| `build_missingness_spec_from_truth_payload` | Reconstructs missingness replay settings from a truth payload | Synthetic generator missingness replay |
| `apply_clustered_missingness_mask` | Applies clustered NaN runs while targeting per-sensor present counts | Synthetic telemetry masking |
| `build_present_counts_for_block` | Converts missingness percentages into per-sensor present-row targets | Missingness replay by block or state |

## Module-Level Technical Context

`missingness.py` keeps synthetic missingness replay separate from fault generation. It reads missingness audit fields from truth metadata, normalizes dropped-sensor missingness, and applies missing values to copied dataframes so generated sensor values can retain source-like missingness structure without modifying the original frame in place.

## Deep Function References

### `build_missingness_spec_from_truth_payload`

#### Functional Purpose

`build_missingness_spec_from_truth_payload` converts a missingness payload, such as `truth.runtime_facts.missingness_quarantine`, into a `MissingnessSpec`. The result contains global missingness rates, state-specific missingness rates, state-dependency flags, state names, and the synthetic state column used when a generated frame does not contain `phase`.

#### Pipeline Context

The function supports synthetic generation when missingness replay is driven by a parent truth record. Active notebook source confirms use in the synthetic generation notebook to build a `missingness_spec` before constructing `SyntheticGenerator`.

#### Inputs and Assumptions

The input is a dictionary-like payload. Source-confirmed expected keys include:

- `missingness_pct_all`
- `missingness_pct_by_state`
- `missingness_state_dependent_flag`
- `missingness_state_gate_params`
- `dropped_features`
- `dropped_missing_pct`
- `dropped_missing_pct_by_state`

The function accepts missing optional mappings as empty. Mapping fields with the wrong shape raise typed errors through supporting payload-normalization helpers. If `state_list` is absent in `missingness_state_gate_params`, the fallback state list is `normal`, `abnormal`, and `recovery`. The historical typo `machine_status__synethic` is normalized to `machine_status__synthetic`.

#### Outputs and Return Contract

The function returns a `MissingnessSpec` with normalized string keys and numeric percentage values. Dropped-sensor global missingness is merged into `missingness_pct_all`. Dropped-sensor state missingness is merged into state maps when available and marks those sensors as state-dependent.

#### Side Effects

No file, SQL, artifact, or dataframe mutation side effects are confirmed. The function constructs and returns a dataclass object.

#### Failure Behavior and Guardrails

The function raises `TypeError` when expected mapping or iterable payload fields have incompatible shapes. It defaults missing optional payload sections to empty mappings or the default state list rather than failing on absent fields.

#### Lineage, Idempotency, and Reproducibility Role

The function preserves missingness lineage from truth metadata by carrying global rates, state-specific rates, dropped-sensor rates, dependency flags, and the state column into a structured replay object. For the same payload, the returned `MissingnessSpec` is deterministic.

#### Why This Function Matters

Synthetic evaluation is more credible when generated telemetry can replay missingness patterns observed in upstream data. This function is the bridge from truth metadata to generator-time masking behavior.

#### Verification Method

- Confirm a payload with `dropped_missing_pct` adds dropped sensors to `missingness_pct_all`.
- Confirm `dropped_missing_pct_by_state` adds state-specific sensor rates and sets dependency flags.
- Confirm missing `state_list` falls back to `normal`, `abnormal`, and `recovery`.
- Confirm the typo `machine_status__synethic` is normalized.
- Confirm invalid nested payload shapes raise `TypeError`.

### `apply_clustered_missingness_mask`

#### Functional Purpose

`apply_clustered_missingness_mask` masks eligible rows with clustered NaN runs while matching each sensor's target present count. It creates short gaps and occasional longer gaps so missing values look more like sensor dropout than independent random deletion.

#### Pipeline Context

The synthetic generator calls this function during missingness replay for phase-specific or global masking. It supports generated telemetry that keeps row count and sensor alignment while adding source-like missing values.

#### Inputs and Assumptions

Important inputs include the dataframe, sensor column list, NumPy random generator, per-sensor `present_counts`, eligible row indexes, `mean_gap_len`, and `long_gap_probability`.

The function assumes eligible row indexes can be interpreted as dataframe index labels. It sorts eligible indexes, skips sensor names not present in the dataframe, and clamps each keep count between zero and the eligible-row count.

#### Outputs and Return Contract

The function returns a copied dataframe with selected sensor values replaced by `NaN`. The row count, index, and non-selected columns are preserved.

#### Side Effects

The function uses the supplied random generator and returns a modified copy. It does not mutate the caller's dataframe from available source and performs no file, SQL, Kafka, or artifact operations.

#### Failure Behavior and Guardrails

The function returns an unchanged copy when no eligible rows are provided. If the target drop count is zero, the sensor is unchanged. If the target drop count covers all eligible rows, all eligible values for that sensor are set to `NaN`. The loop continues selecting gap positions until the target drop count is reached.

#### Lineage, Idempotency, and Reproducibility Role

The function does not add lineage columns. Reproducibility depends on the supplied random generator state, eligible row set, present-count targets, and gap parameters. For the same RNG state and inputs, the mask placement is repeatable.

#### Why This Function Matters

Missingness replay should preserve the target missing-rate structure without destroying row alignment. Clustered gaps produce a more realistic sensor dropout pattern while keeping exact per-sensor present-count targets.

#### Verification Method

- Confirm output row count and index match the input dataframe.
- For each sensor, count non-null eligible rows and compare with `present_counts`.
- Confirm rows outside `eligible_row_idx` are unchanged.
- Confirm sensors absent from the dataframe are skipped.
- Confirm repeated calls with the same RNG seed and inputs produce the same mask.

### `build_present_counts_for_block`

#### Functional Purpose

`build_present_counts_for_block` converts missingness percentages into per-sensor present-row counts for a block of rows. It supports either global missingness percentages or state-specific percentages when state-dependent replay is enabled.

#### Pipeline Context

The synthetic generator uses present-count targets before applying exact or clustered missingness masks. This function provides the count contract that the masking step tries to satisfy.

#### Inputs and Assumptions

Important inputs include the sensor list, block row count, global percentage map, optional state percentage map, and `use_by_state`.

For each sensor, the function uses a state-specific percentage only when `use_by_state=True`, a state map is provided, and the sensor exists in that state map. Otherwise it falls back to `pct_all`, defaulting to zero missingness when no sensor-specific rate exists.

#### Outputs and Return Contract

The function returns a dictionary mapping each sensor name to a present-row count. Percentages are converted through the module's percentage helper, which clamps missingness to the 0 to 100 range and rounds target missing rows with Python's `round` behavior before subtracting from `n_rows`.

#### Side Effects

No side effects are confirmed. The function returns a new dictionary.

#### Failure Behavior and Guardrails

Non-finite missingness percentages produce a full-present count. Percentages below zero are clamped to zero and percentages above 100 are clamped to 100.

#### Lineage, Idempotency, and Reproducibility Role

The function is deterministic for a given sensor list, row count, and percentage mappings. It does not create lineage metadata, but it preserves the replay relationship between truth-derived percentages and the mask target used later.

#### Why This Function Matters

Present-count targets make missingness replay measurable. Reviewers can verify that the generated output matches configured missingness rates at the block level rather than relying only on visual inspection.

#### Verification Method

- Confirm each input sensor appears in the returned dictionary.
- Confirm zero missingness returns `n_rows`.
- Confirm 100 percent missingness returns zero present rows.
- Confirm state-specific rates override global rates only when `use_by_state=True`.
- Confirm non-finite percentages return `n_rows`.

## Cross-Function Relationships

The missingness replay flow is:

- Truth payload fields are normalized by `build_missingness_spec_from_truth_payload`.
- The resulting `MissingnessSpec` supplies global and state-specific percentages.
- `build_present_counts_for_block` converts those percentages into per-sensor present counts for a row block.
- `apply_clustered_missingness_mask` uses the present-count targets and RNG state to place clustered NaN runs while preserving dataframe row alignment.

## Source-Limited Items

- The exact downstream model or notebook that consumes every missingness-replayed sensor column is Not determined from available source.
- Artifact or SQL writes are not performed by these selected functions from available source.
