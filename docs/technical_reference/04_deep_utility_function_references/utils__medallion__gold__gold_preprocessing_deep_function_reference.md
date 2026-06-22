# utils/medallion/gold/gold_preprocessing.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `gold_preprocessing.py` that need deeper explanation than the module-level utility reference. The focus is the shared Gold preprocessing foundation used to create model-ready frames, split metadata, fitted preprocessing objects, normal-only fit rows, Stage 2 feature inputs, and downstream support payloads.

## Source Grounding

Sources used:

- `utils/medallion/gold/gold_preprocessing.py`
- `function_inventory.json`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__medallion__gold__gold_preprocessing_module_reference.md`
- `technical_reference/01_notebook_workflow_references/EDA_Notebook_Pump_Gold_01_PreProcessing_code_reference.md`
- `technical_reference/00_project_manual/medallion_handoff_map.md`
- `technical_reference/00_project_manual/notebook_dependency_matrix.md`
- `technical_reference/00_project_manual/artifact_and_table_handoff_map.md`
- `technical_reference/00_project_manual/notebook_relationship_map.md`
- `technical_reference/02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling_deep_technical_reference.md`

The active utility source file is the source of truth for function behavior. Workflow and manual references provide consumer and handoff context only.

## Functions Covered

| Function | Technical Role | Primary Pipeline Context |
|---|---|---|
| `prepare_gold_model_inputs` | Coordinates split construction, metadata stamping, feature selection, optional encoding, imputation, scaling, and model-input frame assembly. | Gold preprocessing model-input preparation for downstream baseline and cascade modeling. |
| `build_gold_support_artifacts` | Builds in-memory support payloads for reference profile, Stage 2 features, and Stage 3 sensor groups. | Gold preprocessing support artifact preparation for downstream model notebooks. |
| `fit_and_apply_scaler` | Fits a configured scaler on train rows and applies it to selected features across the full frame. | Gold feature scaling before model fit and scoring. |
| `get_training_rows_for_unsupervised_model` | Selects training rows with `anomaly_flag == 0` for unsupervised model fit. | Normal-only Isolation Forest fit input preparation. |
| `choose_stage2_features_from_training_stability` | Filters Stage 2 candidate features using train-row non-null and variance thresholds. | Reduced stable feature set for cascade Stage 2. |
| `build_episode_based_split_mask` | Builds train/test masks by episode order when episode IDs exist, with row-order fallback. | Leakage-controlled Gold train/test split construction. |
| `apply_imputation` | Applies a `SimpleImputer` strategy to selected feature columns and returns imputation metadata. | Missing-value handling before scaling and model input handoff. |

## Module-Level Technical Context

`gold_preprocessing.py` converts a cleaned analytical dataframe and feature registry into stable Gold modeling inputs. Its selected functions establish the split, feature list, missing-value handling, scaling behavior, and fit/test partitions that downstream Gold modeling notebooks rely on. The module returns dataframes, learned preprocessing objects, and metadata dictionaries; the selected shared utility functions do not write files, update SQL tables, update ledgers, or register W&B artifacts directly.

## Deep Function References

### "prepare_gold_model_inputs"

#### Functional Purpose

`prepare_gold_model_inputs` is the shared coordinator for Gold model-input preparation. It takes a source dataframe and feature registry, creates a train/test split, stamps split metadata columns, selects usable features, optionally applies one-hot encoding, imputes missing values, scales the selected features, and returns the canonical frame collection used by later modeling stages.

#### Pipeline Context

This function supports the Gold preprocessing stage that produces the model-input foundation for Gold baseline and cascade modeling. Project manual references confirm that Gold_01 outputs the scaled frame, train/test/fit splits, feature lists, and truth-record path overrides consumed by Gold_02 and Gold_03a through Gold_03c. Direct invocation of `prepare_gold_model_inputs` by a notebook is Not determined from available source.

#### Inputs and Assumptions

- `dataframe` must contain the configured split grouping columns, the fallback order column when row-order fallback is needed, any feature columns listed in `feature_registry["feature_columns"]`, and `anomaly_flag` for the normal-only fit selection.
- `feature_registry` is expected to provide `feature_columns` and may provide `one_hot_encoding_columns`.
- `train_fraction` must satisfy the validation enforced by `build_episode_based_split_mask`: greater than 0 and less than 1.
- `split_episode_column`, `split_group_columns`, and `fallback_order_column` control the split strategy and must match available lineage or ordering fields.
- `exclude_feature_columns` removes features before modeling feature selection.
- `imputation_method` must be one of the methods accepted by `apply_imputation`: `mean`, `median`, or `most_frequent`.
- `scaler_kind` must be accepted by `make_scaler`: `standard`, `robust`, or `minmax`.

#### Outputs and Return Contract

The function returns three objects:

| Return Object | Contents |
|---|---|
| `frames` | `gold_preprocessed`, `gold_preprocessed_scaled`, `gold_fit`, `gold_train`, and `gold_test` dataframe copies. |
| `runtime_info` | `split_info`, `training_info`, `feature_selection_info`, `ohe_info`, `imputation_info`, `scaling_info`, `fit_info`, and `selected_feature_columns`. |
| `learned_objects` | Fitted `imputer` and fitted `scaler`; either may be `None` when no usable feature columns exist. |

`gold_fit` is derived by `get_training_rows_for_unsupervised_model`, so it contains rows where `meta__train_mask` is true and `anomaly_flag` coerces to zero. `gold_train` and `gold_test` are split copies from the scaled dataframe.

#### Side Effects

The function works on dataframe copies and returns in-memory objects. No external file writes, SQL writes, W&B registration, ledger updates, or path creation are confirmed in the active utility source. The returned imputer and scaler are fitted objects when usable feature columns are available.

#### Failure Behavior and Guardrails

- Invalid `train_fraction` raises `ValueError` through `build_episode_based_split_mask`.
- Missing fallback order column raises `ValueError` when episode-based splitting is unavailable.
- Unsupported imputation method raises `ValueError` through `apply_imputation`.
- Unsupported scaler kind raises `ValueError` through `make_scaler`.
- Missing `meta__train_mask` or `anomaly_flag` raises `ValueError` through `get_training_rows_for_unsupervised_model`.
- If no usable feature columns remain, imputation and scaling return unchanged dataframe copies with `applied=False` metadata instead of fitting an object.

#### Lineage and Reproducibility Role

The function preserves reproducibility by returning split metadata, training row counts, selected feature lists, imputation metadata, scaling metadata, and fitted preprocessing objects. It stamps `meta__train_mask`, `meta__is_train_flag`, and `meta__is_train` through the internal training metadata helper. It does not create truth hashes, parent truth hashes, dataset IDs, run IDs, pipeline mode fields, or artifact paths directly.

#### Why This Function Matters

Gold modeling validity depends on consistent preprocessing before Isolation Forest fitting and cascade scoring. This coordinator makes the model-input contract explicit: downstream consumers receive a full scaled frame, train and test partitions, a normal-only fit subset, selected feature metadata, and the learned objects that produced the transformations. If this function changes, downstream model comparability can change even when model code is untouched.

#### Verification Method

- Confirm `runtime_info["selected_feature_columns"]` is non-empty for the intended feature registry.
- Confirm `frames["gold_preprocessed_scaled"]` contains the same selected feature columns as the unscaled frame.
- Check `runtime_info["split_info"]["split_method"]` to verify whether episode splitting or row-order fallback was used.
- Confirm `frames["gold_fit"]["anomaly_flag"]` coerces to zero for all rows.
- Confirm `runtime_info["scaling_info"]["train_rows_used_for_fit"]` matches the count of true values in `meta__train_mask`.
- Confirm no source dataframe columns were changed by comparing the original input frame with a copy captured before the call.

### "build_gold_support_artifacts"

#### Functional Purpose

`build_gold_support_artifacts` builds the in-memory support payload used to carry preprocessing-derived feature and reference information into downstream Gold modeling. It combines a reference profile, stable Stage 2 feature selection, and Stage 3 sensor grouping into one dictionary.

#### Pipeline Context

This function supports the Gold preprocessing handoff to baseline and cascade modeling. Manual references confirm that Gold_01 provides Stage 1 features, Stage 2 features, Stage 3 sensor lists, and reference/profile artifacts to Gold_02 and Gold_03a through Gold_03c. The Gold_01 workflow reference states that its notebook calls equivalent support steps individually; direct use of this integrated shared helper in the notebook path is Not determined from available source.

#### Inputs and Assumptions

- `scaled_dataframe` should already contain scaled model feature columns.
- `selected_feature_columns` identifies the candidate features for profile construction and Stage 2 stability filtering.
- `train_mask` must align with `scaled_dataframe.index`; the function reindexes it before use.
- `baseline_feature_columns` optionally overrides the baseline feature list. If not provided, the function uses `selected_feature_columns`.

#### Outputs and Return Contract

The function returns a dictionary with:

- `baseline_feature_columns`
- `reference_profile`
- `stage2_feature_columns`
- `stage2_info`
- `stage3_sensor_groups`

`reference_profile` is computed from train rows only by passing `train_mask` as the profile subset. `stage2_feature_columns` and `stage2_info` come from `choose_stage2_features_from_training_stability` using a `min_non_null_ratio` of `0.95` and `min_variance` of `1e-12`. `stage3_sensor_groups` is derived from the selected Stage 2 feature names.

#### Side Effects

No external side effects are confirmed from available source. The function returns an in-memory dictionary and does not write JSON, CSV, Parquet, SQL, ledger, or W&B outputs.

#### Failure Behavior and Guardrails

The function relies on the called helpers for guardrails. Missing feature columns are ignored by `build_reference_profile` and `choose_stage2_features_from_training_stability`. A train mask that does not align perfectly is reindexed to the scaled dataframe index. No explicit exception is raised by this function for an empty selected feature list.

#### Lineage and Reproducibility Role

The returned payload records the feature list used by downstream model stages and the training-subset reference profile used to describe normal scaled behavior. It does not stamp truth columns or carry dataset/run identifiers directly. Manual references confirm that Gold_01 writes related feature and sensor artifacts for downstream notebooks, but this shared helper only constructs the payload.

#### Why This Function Matters

The downstream cascade stages depend on consistent feature and sensor sets. This helper keeps the support payload internally consistent by deriving the reference profile, Stage 2 feature list, and Stage 3 sensor groups from the same scaled dataframe and train mask.

#### Verification Method

- Confirm all `stage2_feature_columns` are present in `selected_feature_columns`.
- Confirm `reference_profile["feature_count"]` matches the number of profiled selected features present in `scaled_dataframe`.
- Confirm `stage2_info["min_non_null_ratio"]` and `stage2_info["min_variance"]` match the expected thresholds.
- Confirm `stage3_sensor_groups` contains only columns returned in `stage2_feature_columns`.
- Confirm no files are created by calling this helper alone.

### "fit_and_apply_scaler"

#### Functional Purpose

`fit_and_apply_scaler` fits a configured scikit-learn scaler on the selected feature columns from training rows, then applies the fitted scaler to the selected feature columns across the full dataframe. It returns the scaled dataframe copy, the fitted scaler, and scaling metadata.

#### Pipeline Context

This function supports Gold feature scaling before baseline and cascade model training. The Gold_01 workflow confirms scaling as part of the preprocessing handoff, and the project manual confirms that downstream Gold modeling notebooks consume Gold_01 scaled data. The shared utility source fits on the provided `train_mask`; it does not add normal-only filtering on its own.

#### Inputs and Assumptions

- `dataframe` must contain the selected feature columns that should be scaled.
- `feature_columns` is filtered to columns present in the dataframe.
- `train_mask` must align to the dataframe index or be reindexable to it.
- `scaler_kind` must be one of `standard`, `robust`, or `minmax`.
- The training subset for the remaining feature columns must be suitable for the selected scaler's `fit` call.

#### Outputs and Return Contract

When usable feature columns exist, the function returns:

- A dataframe copy with scaled values written into the selected feature columns.
- The fitted scaler instance.
- Scaling metadata containing `applied=True`, `scaler_kind`, `feature_columns`, `feature_count`, and `train_rows_used_for_fit`.

When no usable feature columns exist, the function returns the copied dataframe, `None` for the scaler, and `applied=False` metadata.

#### Side Effects

The function fits the returned scaler object and creates dataframe copies. No external file writes, model persistence, SQL writes, ledger updates, path creation, or W&B interactions are confirmed in the shared utility source.

#### Failure Behavior and Guardrails

- Unsupported `scaler_kind` raises `ValueError` through `make_scaler`.
- Missing feature columns are filtered out before scaling.
- If no feature columns remain after filtering, no scaler is fitted and no exception is raised.
- If the training subset is empty or otherwise invalid for scikit-learn fitting, the scaler fit can fail through the underlying scaler implementation. The shared utility does not add a custom non-empty training-row check.

#### Lineage and Reproducibility Role

Scaling metadata records the scaler type, selected features, feature count, and number of rows used for fitting. This supports reproducibility because the model-input feature space can be compared against the feature registry and train mask. The function does not persist the scaler or attach artifact paths directly.

#### Why This Function Matters

Scaling affects the numeric space used by Isolation Forest and cascade stages. Fitting only on the provided training mask limits evaluation leakage when the caller supplies a leakage-safe mask. Applying the same fitted scaler to all rows keeps train, test, and scoring frames in one feature space.

#### Verification Method

- Confirm the returned scaler is not `None` when selected features exist.
- Confirm `scaling_info["train_rows_used_for_fit"]` equals `train_mask.astype(bool).reindex(dataframe.index).sum()`.
- Compare selected feature values before and after scaling to verify only selected feature columns changed.
- Confirm test rows were transformed but not included in scaler fitting by checking the train-row count and fitting mask.
- Pass an unsupported `scaler_kind` and confirm `ValueError` is raised.

### "get_training_rows_for_unsupervised_model"

#### Functional Purpose

`get_training_rows_for_unsupervised_model` extracts the fit population for unsupervised anomaly modeling. It keeps rows where the configured train mask column is true and the configured anomaly flag column coerces to zero.

#### Pipeline Context

This function supports the normal-only fit contract used before baseline and cascade Isolation Forest modeling. Manual references confirm that Gold_01 produces a fit-normal-only Parquet consumed by Gold_02 and Gold_03a through Gold_03c. Direct file writing is outside this utility function.

#### Inputs and Assumptions

- The dataframe must contain the configured `train_mask_column`; the default is `meta__train_mask`.
- The dataframe must contain the configured `anomaly_flag_column`; the default is `anomaly_flag`.
- The anomaly flag must be numeric or coercible to numeric. Non-numeric or missing values are coerced and filled as zero by this function.
- The train mask column must be coercible to boolean.

#### Outputs and Return Contract

The function returns:

- A copied dataframe containing rows where the train mask is true and `anomaly_flag == 0`.
- Metadata with `fit_row_count` and `fit_selection_method`.

The fit selection method is recorded as `train_rows_where_anomaly_flag_equals_zero`.

#### Side Effects

The function creates dataframe copies only. No external side effects are confirmed from available source.

#### Failure Behavior and Guardrails

- Missing train mask column raises `ValueError`.
- Missing anomaly flag column raises `ValueError`.
- Non-numeric anomaly flag values are coerced to missing, filled as zero, and treated as normal by the active source.
- The function does not raise a custom error for an empty resulting fit dataframe.

#### Lineage and Reproducibility Role

The function preserves the split contract by using `meta__train_mask` by default, and it preserves the normal-only fit contract by excluding rows whose anomaly flag coerces to a nonzero integer. It does not create truth hashes or artifact paths, but its output is the in-memory equivalent of the Gold fit dataset described in the manual handoff maps.

#### Why This Function Matters

Unsupervised anomaly models are sensitive to contaminated fit populations. This helper centralizes the normal-only training rule so model stages do not accidentally fit on held-out rows or labeled anomaly rows.

#### Verification Method

- Confirm the returned fit dataframe has `meta__train_mask == True` for all rows.
- Confirm the returned fit dataframe has `anomaly_flag` coercing to zero for all rows.
- Confirm `fit_info["fit_row_count"]` equals the returned row count.
- Drop the train mask column in a test copy and confirm `ValueError`.
- Drop the anomaly flag column in a test copy and confirm `ValueError`.

### "choose_stage2_features_from_training_stability"

#### Functional Purpose

`choose_stage2_features_from_training_stability` selects Stage 2 candidate features by applying training-row stability filters. It keeps features that exist in the dataframe, meet the minimum non-null ratio, and have variance above the configured threshold on training rows.

#### Pipeline Context

This function supports the reduced Stage 2 feature set used by cascade modeling. Manual and workflow references confirm that Gold_01 provides Stage 2 feature JSON outputs to Gold_03a through Gold_03c. The active shared utility selects features by threshold filtering; it does not rank by coefficient of variation or enforce a target feature count.

#### Inputs and Assumptions

- `dataframe` must contain the candidate feature columns to evaluate.
- `feature_columns` is filtered to columns present in the dataframe.
- `train_mask` must align with the dataframe index or be reindexable to it.
- `min_non_null_ratio` controls the minimum acceptable fraction of non-missing training values.
- `min_variance` controls the minimum acceptable training variance.

#### Outputs and Return Contract

The function returns:

- `selected_columns`: a list of feature names that passed both stability filters.
- `info`: a dictionary containing `selected_feature_count`, `selected_features`, `rejected_columns`, `min_non_null_ratio`, and `min_variance`.

Rejected features include a reason of `low_non_null_ratio` or `low_variance` with the measured value that caused rejection.

#### Side Effects

No external side effects are confirmed from available source. The function computes feature-selection metadata and returns it in memory.

#### Failure Behavior and Guardrails

Missing feature columns are filtered out before evaluation and are not included in `rejected_columns`. Empty training slices produce a non-null ratio of `0.0` and reject features through the low non-null path. Features with no numeric values receive variance `0.0` and are rejected if they reach that check.

#### Lineage and Reproducibility Role

The returned `info` dictionary records the selected feature list, rejection reasons, and threshold values. This makes Stage 2 feature selection auditable without re-running the entire preprocessing stage. The function does not attach dataset IDs, run IDs, truth hashes, or artifact paths directly.

#### Why This Function Matters

Stage 2 cascade modeling depends on a narrower set of stable inputs. Filtering low-coverage or low-variance features reduces the chance that Stage 2 behavior is driven by missingness artifacts or uninformative columns.

#### Verification Method

- Confirm every selected feature appears in the dataframe and the input candidate list.
- Recompute train-row non-null ratio and variance for selected features and verify they meet thresholds.
- Inspect `stage2_info["rejected_columns"]` for expected rejection reasons.
- Use an all-missing training feature and confirm it is rejected for low non-null ratio.
- Use a constant training feature and confirm it is rejected for low variance.

### "build_episode_based_split_mask"

#### Functional Purpose

`build_episode_based_split_mask` builds a boolean train mask that keeps complete episodes together when episode identifiers exist. When episode IDs are unavailable, it falls back to row-order splitting within the configured group columns.

#### Pipeline Context

This function supports Gold train/test split construction before model-input preparation. The Gold_01 workflow reference confirms episode-aware splitting as part of preprocessing, and downstream Gold modeling notebooks use the resulting split columns rather than re-deriving their own partitions.

#### Inputs and Assumptions

- `dataframe` must include the configured group columns.
- `train_fraction` must be greater than 0 and less than 1.
- If `episode_column` is present and contains non-missing values, the dataframe must also contain the fallback order column because that order column is used to sort episodes by their earliest row.
- If episode splitting is unavailable, `fallback_order_column` must exist for row-order fallback.
- The default grouping context is `meta__asset_id` and `meta__run_id`.

#### Outputs and Return Contract

The function returns:

- A boolean `train_mask` aligned to the copied working dataframe.
- `split_info` metadata.

For episode-based splitting, `split_info` contains `split_method="episode_based"`, `episode_column`, `group_columns`, `train_fraction`, and `episode_count`. For row-order fallback, it contains `split_method="row_order_fallback"`, `fallback_order_column`, `group_columns`, and `train_fraction`.

#### Side Effects

The function copies the input dataframe before adding temporary columns. No mutation of the caller's dataframe and no external side effects are confirmed from available source.

#### Failure Behavior and Guardrails

- Invalid `train_fraction` raises `ValueError`.
- Missing fallback order column raises `ValueError` when row-order fallback is needed.
- In episode mode, rows without an episode match after the merge are treated as test rows because missing `__is_train` values are filled as false.

#### Lineage and Reproducibility Role

The function records split method, grouping columns, and train fraction in `split_info`. Episode mode prevents a single episode from being split across train and test partitions, which protects evaluation from episode-level leakage. The function does not stamp the mask into the dataframe; `prepare_gold_model_inputs` does that through the training metadata helper.

#### Why This Function Matters

Gold model evaluation depends on the train/test split representing future or held-out behavior. Keeping episodes intact prevents a model from training on one part of an anomaly episode and evaluating on another part of the same episode.

#### Verification Method

- For episode mode, group by episode and confirm each episode has only one train/test assignment.
- Confirm earlier episode order values are selected for training before later episode order values within each group.
- Confirm `split_info["split_method"]` matches the expected path.
- Confirm invalid `train_fraction` raises `ValueError`.
- Remove the fallback order column in fallback mode and confirm `ValueError`.

### "apply_imputation"

#### Functional Purpose

`apply_imputation` fills missing values in selected feature columns using a configured `SimpleImputer` strategy. It returns a copied dataframe, the fitted imputer, and metadata describing the imputation operation.

#### Pipeline Context

This function supports Gold model-input preparation before scaling and modeling. The shared utility implementation uses scikit-learn `SimpleImputer` with one of three strategies. Notebook-local Gold_01 workflow references describe a more specialized forward-fill strategy, but that behavior is not present in this shared utility source.

#### Inputs and Assumptions

- `dataframe` must contain the feature columns to impute.
- `feature_columns` is filtered to columns present in the dataframe.
- `method` must be one of `mean`, `median`, or `most_frequent`.
- Selected feature columns must be compatible with the chosen `SimpleImputer` strategy.

#### Outputs and Return Contract

When usable feature columns exist, the function returns:

- A dataframe copy with imputed values written into selected feature columns.
- The fitted `SimpleImputer`.
- Metadata with `applied=True`, `method`, `feature_columns`, and `feature_count`.

When no selected features are present, the function returns the copied dataframe, `None` for the imputer, and `applied=False` metadata.

#### Side Effects

The function fits the returned imputer object and creates a dataframe copy. No file writes, SQL writes, ledger updates, path creation, or W&B interactions are confirmed from the shared utility source.

#### Failure Behavior and Guardrails

- Unsupported imputation method raises `ValueError`.
- Missing selected feature columns are filtered out.
- If no usable feature columns remain, the function does not fit an imputer and returns `applied=False`.
- Strategy-specific failures are delegated to `SimpleImputer`.

#### Lineage and Reproducibility Role

The imputation metadata records the method and feature list used for preprocessing. This supports review of the model-input preparation path, especially when comparing preprocessing settings across runs. The function does not record train/test lineage by itself and does not fit on train rows only; callers must supply an already appropriate dataframe or use a different notebook-local method when train-only imputation is required.

#### Why This Function Matters

Missing values can prevent scaling and model fitting or can create inconsistent feature behavior across train and test data. This helper centralizes the shared utility contract for simple imputation and returns the fitted object needed to understand or repeat the transformation.

#### Verification Method

- Confirm selected feature columns contain no missing values after imputation when the strategy can fill them.
- Confirm non-selected columns are unchanged.
- Confirm `imputation_info["feature_count"]` matches the number of imputed columns.
- Confirm the returned imputer is `None` only when no usable feature columns are present.
- Pass an unsupported method and confirm `ValueError` is raised.

## Cross-Function Relationships

- `prepare_gold_model_inputs` calls `build_episode_based_split_mask` to create the split, then stamps that split into `meta__train_mask`, `meta__is_train_flag`, and `meta__is_train`.
- Feature selection and optional one-hot encoding establish the selected feature list before `apply_imputation`.
- `apply_imputation` runs before `fit_and_apply_scaler`, so scaled features are based on imputed values.
- `fit_and_apply_scaler` uses the stamped train mask to fit the scaler and applies the transformation to the full frame.
- `get_training_rows_for_unsupervised_model` extracts the normal-only fit subset after scaling.
- `build_gold_support_artifacts` uses the scaled dataframe and train mask to construct a reference profile, then calls `choose_stage2_features_from_training_stability` and derives Stage 3 sensor groups from the selected Stage 2 columns.
- Manual references confirm that Gold_01 preprocessing outputs are consumed by Gold_02 and Gold_03a through Gold_03c, but external artifact writes and truth-record persistence are outside this shared utility module.

## Source-Limited Items

- Direct notebook invocation of `prepare_gold_model_inputs` is Not determined from available source.
- Direct notebook invocation of the integrated `build_gold_support_artifacts` helper is Not determined from available source; the Gold_01 workflow reference confirms the notebook calls equivalent steps individually.
- The shared utility does not implement the notebook-local forward-fill imputation strategy described in the Gold_01 workflow reference. That notebook-local behavior is not documented here as shared utility behavior.
- The shared utility does not persist scaler files, feature JSON files, Parquet splits, SQL rows, truth records, ledger events, or W&B artifacts. Those side effects are confirmed for the Gold_01 notebook workflow, not for the selected shared utility functions.
- Direct dataset ID, run ID, parent truth hash, pipeline mode, and artifact path handling inside these selected functions is Not determined from available source.
