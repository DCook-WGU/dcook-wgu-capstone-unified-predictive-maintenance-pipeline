# utils/medallion/gold/gold_cascade_validation_contracts.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `gold_cascade_validation_contracts.py` that need deeper explanation than the module-level utility reference. The focus is explicit Gold model-output validation contracts: contract payload construction, JSON persistence, and comparison of Gold 04 rows against model-output contract metrics.

## Source Grounding

Sources used:

- `utils/medallion/gold/gold_cascade_validation_contracts.py`
- `function_inventory.json`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__medallion__gold__gold_cascade_validation_contracts_module_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling_code_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling_code_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling_code_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_04_Comparison_code_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation_code_reference.md`
- `technical_reference/00_project_manual/notebook_relationship_map.md`
- `technical_reference/00_project_manual/notebook_dependency_matrix.md`
- `technical_reference/00_project_manual/artifact_and_table_handoff_map.md`
- `technical_reference/00_project_manual/medallion_handoff_map.md`

The active utility source file is the source of truth for function behavior. Workflow and manual references provide consumer and handoff context only.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
|---|---|---|
| `build_gold_model_output_validation_contract` | Builds a JSON-safe validation contract for one final Gold comparison model output. | Gold 03 validation-contract creation for Gold 06 replay validation. |
| `write_gold_model_output_validation_contract` | Writes a model-output validation contract to a configured JSON path. | Validation contract persistence after Gold model output generation. |
| `validate_gold04_against_output_contracts` | Compares Gold 04 final comparison rows against loaded model-output contract metric payloads. | Contract-based validation of comparison rows when caller supplies contracts. |

## Module-Level Technical Context

`gold_cascade_validation_contracts.py` provides JSON-safe contract helpers for model-output validation. The selected functions focus on the newer explicit model-output contract path: build a contract with model identity, metric payload, output dataframe summary, artifact/model references, rule metadata, and lineage payload; write the contract to disk; and compare Gold 04 rows against loaded contracts. The selected functions do not query SQL, run models, replay test data, update ledgers, register W&B artifacts, or create truth records directly.

## Deep Function References

### `build_gold_model_output_validation_contract`

#### Functional Purpose

`build_gold_model_output_validation_contract` builds one explicit validation contract for a final Gold comparison row. It packages model identity, source notebook, validation type, model stage, operating mode, normalized metrics, output dataframe summary, rule metadata, model paths, output artifact path, lineage payload, notes, and a stable contract hash into a JSON-safe dictionary.

#### Pipeline Context

Workflow references confirm direct use in Gold_03a, Gold_03b, and Gold_03c to create validation contracts for Gold 06A consumption. Gold_03c writes separate contracts for default and Stage 3 operating modes. Project manual references confirm that validation contract JSON files produced by Gold_02 and Gold_03 variants support Gold_06A replay validation.

#### Inputs and Assumptions

- `dataset_id` and `run_id` identify the dataset/run represented by the contract.
- `model_id`, `model_label`, `source_notebook`, `validation_type`, `model_stage`, and `operating_mode` define the model-output identity.
- `metrics` is normalized by `normalize_gold_metric_payload`, which extracts canonical alert and metric fields and retains the source metrics payload. The normalization accepts multiple key aliases for `alert_count_test_rows`: any of `alert_count_test_rows`, `final_alert_count_test_rows`, `predicted_positive_count`, `alerts`, or `alert_count` in the input dict is mapped to the canonical `alert_count_test_rows` key in the `metric_payload`. This allows the contract to accept metric dicts from Gold notebooks that use different local naming conventions.
- `output_dataframe` and `output_flag_column` are summarized by `summarize_validation_output_dataframe`.
- `test_mask`, when supplied, must align to the output dataframe length.
- `rule_config`, `rule_source`, `stage3_type`, and `stage3_saved_as_joblib` describe rule-based Stage 3 behavior when applicable.
- `stage1_model_path`, `stage2_model_path`, and `output_artifact_path` are recorded as strings when supplied; the function does not check whether these paths exist.
- `lineage_payload` is carried into the contract after JSON-safe conversion.

#### Outputs and Return Contract

The function returns a JSON-safe dictionary with:

- `contract_type="gold_model_output_validation_contract"`
- `contract_version="v1"`
- `created_at_utc`
- dataset, run, model, stage, operating-mode, and source-notebook fields
- `metric_payload`
- `output_dataframe_summary`
- `output_artifact_path`
- rule and Stage 3 fields
- Stage 1 and Stage 2 model path fields
- `lineage_payload`
- `notes`
- `contract_hash`

The contract hash is computed from the contract payload using stable JSON hashing after JSON-safe conversion.

#### Side Effects

No external side effects are confirmed from available source. The function builds an in-memory dictionary and does not write it to disk.

#### Failure Behavior and Guardrails

- `summarize_validation_output_dataframe` raises `ValueError` if `test_mask` length does not match the output dataframe length.
- Metric values that cannot be converted to expected numeric types become `None` through the normalization helpers.
- Dataframe and path-like objects are converted to JSON-safe summaries or strings.
- The function does not validate required field content beyond converting values to strings and dictionaries.

#### Lineage and Reproducibility Role

The contract preserves model identity, dataset/run identity, model-stage naming, operating mode, normalized metrics, output row/column counts, flag counts, artifact path references, rule configuration, model path references, and caller-provided lineage payload. Workflow references confirm that Gold_03a includes lineage payload fields such as `cascade_truth_hash` and `parent_gold_truth_hash` when building its contract.

#### Why This Function Matters

Gold 06 validation needs durable expectations from the original modeling notebooks. This function turns notebook-local model outputs and metric payloads into a portable validation contract that can be inspected or compared later without relying only on live notebook variables.

#### Verification Method

- Confirm the returned contract contains `contract_type`, `contract_version`, `model_id`, `metric_payload`, `output_dataframe_summary`, and `contract_hash`.
- Confirm `metric_payload` contains canonical fields such as `alert_count_test_rows`, `precision`, `recall`, and `f1`.
- Confirm `output_dataframe_summary["flag_count_test_rows"]` matches the supplied flag column and test mask.
- Confirm path-like inputs are represented as strings.
- Pass a mismatched `test_mask` length and confirm `ValueError`.

### `write_gold_model_output_validation_contract`

#### Functional Purpose

`write_gold_model_output_validation_contract` persists one final-model validation contract to a JSON path. It is a thin wrapper around `write_json_contract`, which creates parent directories, converts the contract to JSON-safe values, ensures a contract hash exists, and writes pretty sorted JSON.

#### Pipeline Context

Workflow references confirm direct use in Gold_03a, Gold_03b, and Gold_03c after contract construction. The written JSON files are part of the validation-contract handoff described by the project manual for Gold 06A replay validation.

#### Inputs and Assumptions

- `contract` is a mapping containing the validation contract payload.
- `output_path` is the target JSON file path.
- The caller is responsible for choosing the canonical validation contract path.
- Values in `contract` should be JSON-safe or convertible by `to_json_safe`.

#### Outputs and Return Contract

The function returns the `Path` produced by `write_json_contract`. That path is the JSON file written to disk.

#### Side Effects

The function writes a JSON file and creates the parent directory if needed. It may add `contract_hash` to the payload written to disk if the supplied contract lacks one. No SQL, model persistence, ledger update, W&B operation, or truth-record write is confirmed from available source.

#### Failure Behavior and Guardrails

- Invalid or unwritable paths can fail through `Path.mkdir` or file open/write operations.
- Values that cannot be converted by JSON serialization can fail through `json.dump`, though common notebook objects are handled by `to_json_safe`.
- The helper does not validate that the output path follows a specific naming convention.

#### Lineage and Reproducibility Role

The written JSON file becomes the durable contract artifact. Its hash and JSON-safe payload allow downstream validation or manual review to compare model-output metrics, stage metadata, rule settings, and lineage payloads against later replay or comparison results.

#### Why This Function Matters

Validation contracts only support downstream replay and audit if they are written to a stable artifact path. This helper centralizes the JSON persistence behavior so Gold model notebooks do not each implement contract serialization differently.

#### Verification Method

- Confirm the returned path exists after the function call.
- Load the JSON and confirm it contains `contract_hash`.
- Confirm the parent directory is created when it did not exist.
- Confirm the written file is valid JSON.
- Confirm key contract fields such as `model_id`, `model_stage`, `operating_mode`, and `metric_payload` are preserved.

### `validate_gold04_against_output_contracts`

#### Functional Purpose

`validate_gold04_against_output_contracts` validates Gold 04 final comparison rows against model-output contracts already loaded by the caller. It checks one target row at a time, compares Gold 04 metric values against the contract metric payload, and returns a validation dataframe with status and comparison columns.

#### Pipeline Context

This function is designed for contract-based validation of Gold 04 comparison rows. Manual references confirm that Gold_04 aggregates seven model comparison rows and that Gold_06A uses validation contracts for replay validation, but direct notebook invocation of this exact helper is Not determined from available source.

#### Inputs and Assumptions

- `gold04_dataframe` must contain a `model_id` column.
- `validation_targets` must contain target rows with `model_id`, source-notebook, validation-type, model-stage, operating-mode, and contract-file fields.
- `contracts_by_model_id` maps model IDs to loaded contract dictionaries.
- Contract dictionaries are expected to contain `metric_payload`.
- The compared metrics are `alert_count_test_rows`, `precision`, `recall`, and `f1`.
- `metric_tolerance` defaults to `1e-9` and is passed to the metric comparison helper.

#### Outputs and Return Contract

The function returns a dataframe with one row per validation target. Each row includes:

- target identity fields
- `gold04_row_available`
- `contract_available`
- `validation_status`
- `stage3_type`
- `stage3_saved_as_joblib`
- `contract_hash`
- Gold 04 and contract values for `alert_count_test_rows`, `precision`, `recall`, and `f1`
- per-metric match flags when both sides are available

Validation status is:

- `fail_missing_gold04_row` when Gold 04 has no row for the model ID.
- `fail_missing_contract` when no contract is loaded for the model ID.
- `warn_metric_mismatch` when comparable metric values exist and at least one differs outside tolerance.
- `pass` otherwise.

#### Side Effects

No side effects are confirmed from available source. The function reads supplied dataframes and mappings and returns a validation dataframe.

#### Failure Behavior and Guardrails

- Raises `KeyError` if `gold04_dataframe` has no `model_id` column.
- Missing contracts and missing Gold 04 rows are represented in the returned validation status rather than raising.
- Metric comparisons return `None` when either side is missing.
- `alert_count_test_rows` is compared as an integer count; precision, recall, and F1 are compared using `math.isclose` with the configured tolerance.

#### Lineage and Reproducibility Role

The validation output ties Gold 04 comparison rows to contract hashes, model-stage metadata, operating modes, Stage 3 type, and `stage3_saved_as_joblib`. This gives reviewers a way to see whether final comparison rows still match the contract payloads produced by upstream model notebooks.

#### Why This Function Matters

Gold 04 is the convergence point for the model comparison narrative. This helper provides a contract-based check that comparison metrics still align with saved model-output contracts, reducing the risk that final reporting drifts away from the artifacts produced by model notebooks.

#### Verification Method

- Confirm the returned dataframe has one row per validation target.
- Confirm every target model ID has the expected `contract_available` and `gold04_row_available` values.
- Confirm metric match fields are true when Gold 04 values equal contract metric payload values within tolerance.
- Remove `model_id` from a Gold 04 dataframe copy and confirm `KeyError`.
- Provide a target without a loaded contract and confirm `validation_status == "fail_missing_contract"`.

## Cross-Function Relationships

- `build_gold_model_output_validation_contract` creates the in-memory contract payload.
- `write_gold_model_output_validation_contract` persists that payload as JSON by delegating to `write_json_contract`.
- The written contracts can later be loaded by separate helpers and supplied to `validate_gold04_against_output_contracts`.
- `validate_gold04_against_output_contracts` compares Gold 04 rows against the loaded contract metric payloads and contract metadata.
- Workflow references confirm contract building and writing in Gold_03a, Gold_03b, and Gold_03c. Direct use of `validate_gold04_against_output_contracts` is Not determined from available source.

## Source-Limited Items

- Direct notebook invocation of `validate_gold04_against_output_contracts` is Not determined from available source.
- Gold_06A replay validation is confirmed in workflow and manual references, but the available notebook source shows replay comparison logic implemented in-notebook rather than through this exact helper.
- `write_gold_model_output_validation_contract` does not choose the canonical path; path selection is handled by callers and other artifact helpers.
- The selected functions do not read or write SQL, update ledgers, register W&B artifacts, persist models, or create truth records directly.
