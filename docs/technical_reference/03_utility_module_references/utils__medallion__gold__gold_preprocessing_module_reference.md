# Utility Module Reference: `utils/medallion/gold/gold_preprocessing.py`

## Module Purpose

This module prepares Silver outputs for Gold modeling by building clean model-input frames and related metadata.

## Pipeline Role

- Stage support: Gold
- Primary responsibility: This module prepares Silver outputs for Gold modeling by building clean model-input frames and related metadata.

## Primary Consumers

`EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `build_episode_based_split_mask` | Build a train/test mask using episode-aware splitting when episode IDs exist. | deep |
| `stamp_training_metadata` | Stamp train/test metadata into a copied dataframe. | deep |
| `select_numeric_feature_columns` | Select numeric feature columns from the registered feature set. | deep |
| `apply_one_hot_encoding_from_truths` | Apply one-hot encoding to selected categorical columns. | deep |
| `apply_imputation` | Impute missing values for selected feature columns. | deep |
| `make_scaler` | Create a scikit-learn scaler instance by kind. | deep |
| `fit_and_apply_scaler` | Fit a scaler on training rows and apply it to all selected rows. | deep |
| `get_training_rows_for_unsupervised_model` | Return normal-only training rows for unsupervised model fitting. | deep |
| `build_reference_profile` | Build a reference profile from a selected feature subset. | medium |
| `choose_stage2_features_from_training_stability` | Choose Stage 2 features using training-stability heuristics. | deep |
| `build_stage3_sensor_groups` | Build Stage 3 sensor groups from feature-name prefixes. | medium |
| `prepare_gold_model_inputs` | Prepare model-ready Gold dataframes and learned preprocessing objects. | deep |
| `build_gold_support_artifacts` | Build downstream support artifacts for baseline and cascade modeling. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `build_episode_based_split_mask` | `dataframe, *, train_fraction, episode_column, group_columns, fallback_order_column` | Build a train/test mask using episode-aware splitting when episode IDs exist. |
| `stamp_training_metadata` | `dataframe, *, train_mask, train_flag_column, train_label_column, train_mask_column` | Stamp train/test metadata into a copied dataframe. |
| `select_numeric_feature_columns` | `dataframe, *, feature_columns, exclude_columns` | Select numeric feature columns from the registered feature set. |
| `apply_one_hot_encoding_from_truths` | `dataframe, *, one_hot_columns, drop_first, dtype` | Apply one-hot encoding to selected categorical columns. |
| `apply_imputation` | `dataframe, *, feature_columns, method` | Impute missing values for selected feature columns. |
| `make_scaler` | `scaler_kind` | Create a scikit-learn scaler instance by kind. |
| `fit_and_apply_scaler` | `dataframe, *, feature_columns, train_mask, scaler_kind` | Fit a scaler on training rows and apply it to all selected rows. |
| `get_training_rows_for_unsupervised_model` | `dataframe, *, train_mask_column, anomaly_flag_column` | Return normal-only training rows for unsupervised model fitting. |
| `build_reference_profile` | `dataframe, *, feature_columns, subset_mask` | Build a reference profile from a selected feature subset. |
| `choose_stage2_features_from_training_stability` | `dataframe, *, feature_columns, train_mask, min_non_null_ratio, min_variance` | Choose Stage 2 features using training-stability heuristics. |
| `build_stage3_sensor_groups` | `feature_columns, *, separators` | Build Stage 3 sensor groups from feature-name prefixes. |
| `prepare_gold_model_inputs` | `dataframe, *, feature_registry, train_fraction, split_episode_column, split_group_columns, fallback_order_column, select_numeric_only, apply_one_hot_encoding, one_hot_columns, imputation_method, scaler_kind, exclude_feature_columns` | Prepare model-ready Gold dataframes and learned preprocessing objects. |
| `build_gold_support_artifacts` | `scaled_dataframe, *, selected_feature_columns, train_mask, baseline_feature_columns` | Build downstream support artifacts for baseline and cascade modeling. |

## Side Effects

- Source includes SQL execution or SQL helper calls; helpers can read from or write to PostgreSQL when invoked with a live engine/connection.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
- SQL: Source references SQL execution, SQLAlchemy, or PostgreSQL-facing helpers.
- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03c_Cascade_Modeling`

## Module Importance

This module matters because Gold notebooks depend on stable shared helpers for model input preparation, cascade modeling, evaluation, validation contracts, and artifact traceability.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
