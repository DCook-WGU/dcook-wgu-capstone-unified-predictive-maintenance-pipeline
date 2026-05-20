"""
utils/gold_preprocessing.py

Gold preprocessing helpers for the capstone pipeline.

Purpose
-------
Take the cleaned Silver dataframe plus Silver EDA/registry artifacts and prepare
model-ready Gold outputs for baseline and cascade modeling.

Main responsibilities
---------------------
- Build train/test masks
- Stamp training metadata
- Select modeling feature columns
- Optionally apply one-hot encoding
- Apply imputation
- Fit and apply scaler
- Build normal-only fit subset
- Build downstream support artifacts for baseline/cascade
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def build_episode_based_split_mask(
    dataframe: pd.DataFrame,
    *,
    train_fraction: float = 0.7,
    episode_column: str = "meta__episode_id",
    group_columns: Sequence[str] = ("meta__asset_id", "meta__run_id"),
    fallback_order_column: str = "time_index",
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Build a train/test mask using episode-aware splitting when episode IDs exist.

    Strategy
    --------
    - If episode IDs are present and populated, split by episode order within each asset/run.
    - Otherwise split by row order within each asset/run using fallback_order_column.
    """
    working_dataframe = dataframe.copy()

    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")

    if episode_column in working_dataframe.columns and working_dataframe[episode_column].notna().any():
        episode_frame = (
            working_dataframe[list(group_columns) + [episode_column, fallback_order_column]]
            .dropna(subset=[episode_column])
            .groupby(list(group_columns) + [episode_column], dropna=False, as_index=False)[fallback_order_column]
            .min()
            .rename(columns={fallback_order_column: "__episode_min_order"})
            .sort_values(list(group_columns) + ["__episode_min_order", episode_column])
            .reset_index(drop=True)
        )

        episode_frame["__episode_rank"] = (
            episode_frame.groupby(list(group_columns), dropna=False).cumcount()
        )
        episode_frame["__episode_total"] = (
            episode_frame.groupby(list(group_columns), dropna=False)["__episode_rank"]
            .transform("max")
            .fillna(0)
            .astype(int)
            + 1
        )

        episode_frame["__is_train"] = (
            (episode_frame["__episode_rank"] / episode_frame["__episode_total"])
            < float(train_fraction)
        )

        merge_columns = list(group_columns) + [episode_column]
        working_dataframe = working_dataframe.merge(
            episode_frame[merge_columns + ["__is_train"]],
            on=merge_columns,
            how="left",
        )

        train_mask = working_dataframe["__is_train"].fillna(False).astype(bool)
        split_info = {
            "split_method": "episode_based",
            "episode_column": episode_column,
            "group_columns": list(group_columns),
            "train_fraction": float(train_fraction),
            "episode_count": int(episode_frame[episode_column].nunique(dropna=True)),
        }
        return train_mask, split_info

    if fallback_order_column not in working_dataframe.columns:
        raise ValueError(
            f"Could not build split mask. Missing fallback_order_column: {fallback_order_column}"
        )

    working_dataframe["__row_rank"] = (
        working_dataframe
        .sort_values(list(group_columns) + [fallback_order_column])
        .groupby(list(group_columns), dropna=False)
        .cumcount()
    )

    group_sizes = (
        working_dataframe.groupby(list(group_columns), dropna=False)["__row_rank"]
        .transform("max")
        .fillna(0)
        .astype(int)
        + 1
    )

    train_mask = ((working_dataframe["__row_rank"] / group_sizes) < float(train_fraction)).astype(bool)

    split_info = {
        "split_method": "row_order_fallback",
        "fallback_order_column": fallback_order_column,
        "group_columns": list(group_columns),
        "train_fraction": float(train_fraction),
    }

    train_mask = train_mask.reindex(working_dataframe.index)
    return train_mask, split_info


def stamp_training_metadata(
    dataframe: pd.DataFrame,
    *,
    train_mask: pd.Series,
    train_flag_column: str = "meta__is_train_flag",
    train_label_column: str = "meta__is_train",
    train_mask_column: str = "meta__train_mask",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Stamp train/test metadata into the dataframe.
    """
    working_dataframe = dataframe.copy()
    train_mask = train_mask.astype(bool).reindex(working_dataframe.index)

    working_dataframe[train_mask_column] = train_mask.astype(bool)
    working_dataframe[train_flag_column] = train_mask.astype(int)
    working_dataframe[train_label_column] = np.where(train_mask, "train", "test")

    train_count = int(train_mask.sum())
    test_count = int((~train_mask).sum())

    return working_dataframe, {
        "train_mask_column": train_mask_column,
        "train_flag_column": train_flag_column,
        "train_label_column": train_label_column,
        "train_count": train_count,
        "test_count": test_count,
    }


def select_numeric_feature_columns(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    exclude_columns: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Select numeric feature columns from the registered feature set.
    """
    exclude_columns = set(exclude_columns or [])
    numeric_feature_columns: List[str] = []
    rejected_columns: List[Dict[str, Any]] = []

    for column_name in feature_columns:
        if column_name in exclude_columns:
            rejected_columns.append(
                {"column_name": column_name, "reason": "explicit_exclude"}
            )
            continue

        if column_name not in dataframe.columns:
            rejected_columns.append(
                {"column_name": column_name, "reason": "missing_from_dataframe"}
            )
            continue

        if pd.api.types.is_numeric_dtype(dataframe[column_name]):
            numeric_feature_columns.append(column_name)
        else:
            rejected_columns.append(
                {
                    "column_name": column_name,
                    "reason": "not_numeric",
                    "dtype": str(dataframe[column_name].dtype),
                }
            )

    info = {
        "selected_numeric_feature_count": int(len(numeric_feature_columns)),
        "selected_numeric_feature_columns": list(numeric_feature_columns),
        "rejected_columns": rejected_columns,
    }
    return numeric_feature_columns, info


def apply_one_hot_encoding_from_truths(
    dataframe: pd.DataFrame,
    *,
    one_hot_columns: Sequence[str],
    drop_first: bool = False,
    dtype: str = "uint8",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply one-hot encoding to selected categorical columns.
    """
    working_dataframe = dataframe.copy()
    one_hot_columns = [column_name for column_name in one_hot_columns if column_name in working_dataframe.columns]

    if len(one_hot_columns) == 0:
        return working_dataframe, {
            "applied": False,
            "one_hot_columns": [],
            "created_columns": [],
            "drop_first": drop_first,
        }

    encoded_frame = pd.get_dummies(
        working_dataframe[one_hot_columns],
        columns=list(one_hot_columns),
        drop_first=drop_first,
        dtype=dtype,
    )

    created_columns = list(encoded_frame.columns)
    working_dataframe = working_dataframe.drop(columns=list(one_hot_columns))
    working_dataframe = pd.concat([working_dataframe, encoded_frame], axis=1)

    return working_dataframe, {
        "applied": True,
        "one_hot_columns": list(one_hot_columns),
        "created_columns": created_columns,
        "drop_first": drop_first,
    }


def apply_imputation(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    method: str = "median",
) -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
    """
    Impute missing values for selected feature columns.
    """
    working_dataframe = dataframe.copy()
    feature_columns = [column_name for column_name in feature_columns if column_name in working_dataframe.columns]

    if len(feature_columns) == 0:
        return working_dataframe, None, {
            "applied": False,
            "method": method,
            "feature_columns": [],
        }

    valid_methods = {"mean", "median", "most_frequent"}
    if method not in valid_methods:
        raise ValueError(f"Unsupported imputation method: {method}")

    imputer = SimpleImputer(strategy=method)
    imputed_values = imputer.fit_transform(working_dataframe[feature_columns])
    working_dataframe.loc[:, feature_columns] = imputed_values

    return working_dataframe, imputer, {
        "applied": True,
        "method": method,
        "feature_columns": list(feature_columns),
        "feature_count": int(len(feature_columns)),
    }


def make_scaler(
    scaler_kind: str,
):
    """
    Create scaler instance by kind.
    """
    scaler_kind = str(scaler_kind).strip().lower()

    if scaler_kind == "standard":
        return StandardScaler()
    if scaler_kind == "robust":
        return RobustScaler()
    if scaler_kind == "minmax":
        return MinMaxScaler()

    raise ValueError(f"Unsupported scaler_kind: {scaler_kind}")


def fit_and_apply_scaler(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    train_mask: pd.Series,
    scaler_kind: str = "robust",
) -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
    """
    Fit scaler on training rows and apply to all rows for selected features.
    """
    working_dataframe = dataframe.copy()
    feature_columns = [column_name for column_name in feature_columns if column_name in working_dataframe.columns]

    if len(feature_columns) == 0:
        return working_dataframe, None, {
            "applied": False,
            "scaler_kind": scaler_kind,
            "feature_columns": [],
        }

    train_mask = train_mask.astype(bool).reindex(working_dataframe.index)
    scaler = make_scaler(scaler_kind)

    scaler.fit(working_dataframe.loc[train_mask, feature_columns])
    scaled_values = scaler.transform(working_dataframe[feature_columns])

    scaled_dataframe = working_dataframe.copy()
    scaled_dataframe.loc[:, feature_columns] = scaled_values

    return scaled_dataframe, scaler, {
        "applied": True,
        "scaler_kind": scaler_kind,
        "feature_columns": list(feature_columns),
        "feature_count": int(len(feature_columns)),
        "train_rows_used_for_fit": int(train_mask.sum()),
    }


def get_training_rows_for_unsupervised_model(
    dataframe: pd.DataFrame,
    *,
    train_mask_column: str = "meta__train_mask",
    anomaly_flag_column: str = "anomaly_flag",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Return normal-only training rows for unsupervised model fitting.
    """
    working_dataframe = dataframe.copy()

    if train_mask_column not in working_dataframe.columns:
        raise ValueError(f"Missing train mask column: {train_mask_column}")
    if anomaly_flag_column not in working_dataframe.columns:
        raise ValueError(f"Missing anomaly flag column: {anomaly_flag_column}")

    train_mask = working_dataframe[train_mask_column].astype(bool)
    anomaly_flag = pd.to_numeric(working_dataframe[anomaly_flag_column], errors="coerce").fillna(0).astype(int)

    fit_dataframe = working_dataframe.loc[train_mask & (anomaly_flag == 0)].copy()

    return fit_dataframe, {
        "fit_row_count": int(len(fit_dataframe)),
        "fit_selection_method": "train_rows_where_anomaly_flag_equals_zero",
    }


def build_reference_profile(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    subset_mask: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Build a simple reference profile from a selected feature subset.
    """
    feature_columns = [column_name for column_name in feature_columns if column_name in dataframe.columns]

    if subset_mask is None:
        profiled_dataframe = dataframe[feature_columns].copy()
    else:
        subset_mask = subset_mask.astype(bool).reindex(dataframe.index)
        profiled_dataframe = dataframe.loc[subset_mask, feature_columns].copy()

    if len(feature_columns) == 0:
        return {
            "feature_columns": [],
            "feature_count": 0,
            "summary": {},
        }

    summary: Dict[str, Dict[str, Any]] = {}
    for column_name in feature_columns:
        series = pd.to_numeric(profiled_dataframe[column_name], errors="coerce")
        summary[column_name] = {
            "mean": float(series.mean()) if series.notna().any() else None,
            "median": float(series.median()) if series.notna().any() else None,
            "std": float(series.std()) if series.notna().any() else None,
            "min": float(series.min()) if series.notna().any() else None,
            "max": float(series.max()) if series.notna().any() else None,
        }

    return {
        "feature_columns": list(feature_columns),
        "feature_count": int(len(feature_columns)),
        "summary": summary,
    }


def choose_stage2_features_from_training_stability(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    train_mask: pd.Series,
    min_non_null_ratio: float = 0.95,
    min_variance: float = 1e-12,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Choose Stage 2 features using simple training stability heuristics.
    """
    feature_columns = [column_name for column_name in feature_columns if column_name in dataframe.columns]
    train_mask = train_mask.astype(bool).reindex(dataframe.index)

    train_frame = dataframe.loc[train_mask, feature_columns].copy()

    selected_columns: List[str] = []
    rejected_columns: List[Dict[str, Any]] = []

    for column_name in feature_columns:
        series = pd.to_numeric(train_frame[column_name], errors="coerce")
        non_null_ratio = float(series.notna().mean()) if len(series) > 0 else 0.0
        variance = float(series.var()) if series.notna().any() else 0.0

        if non_null_ratio < float(min_non_null_ratio):
            rejected_columns.append(
                {
                    "column_name": column_name,
                    "reason": "low_non_null_ratio",
                    "non_null_ratio": non_null_ratio,
                }
            )
            continue

        if variance <= float(min_variance):
            rejected_columns.append(
                {
                    "column_name": column_name,
                    "reason": "low_variance",
                    "variance": variance,
                }
            )
            continue

        selected_columns.append(column_name)

    info = {
        "selected_feature_count": int(len(selected_columns)),
        "selected_features": list(selected_columns),
        "rejected_columns": rejected_columns,
        "min_non_null_ratio": float(min_non_null_ratio),
        "min_variance": float(min_variance),
    }

    return selected_columns, info


def build_stage3_sensor_groups(
    feature_columns: Sequence[str],
    *,
    separators: Sequence[str] = ("__", "_"),
) -> Dict[str, List[str]]:
    """
    Build simple Stage 3 sensor groups based on feature name prefixes.
    """
    sensor_groups: Dict[str, List[str]] = {}

    for column_name in feature_columns:
        sensor_key = column_name
        for separator in separators:
            if separator in column_name:
                sensor_key = column_name.split(separator)[0]
                break

        sensor_groups.setdefault(sensor_key, []).append(column_name)

    return {
        group_name: sorted(columns)
        for group_name, columns in sorted(sensor_groups.items())
    }


def prepare_gold_model_inputs(
    dataframe: pd.DataFrame,
    *,
    feature_registry: Dict[str, Any],
    train_fraction: float,
    split_episode_column: str = "meta__episode_id",
    split_group_columns: Sequence[str] = ("meta__asset_id", "meta__run_id"),
    fallback_order_column: str = "time_index",
    select_numeric_only: bool = True,
    apply_one_hot_encoding: bool = False,
    one_hot_columns: Optional[Sequence[str]] = None,
    imputation_method: str = "median",
    scaler_kind: str = "robust",
    exclude_feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any], Dict[str, Any]]:
    """
    Main orchestration helper for Gold preprocessing.
    """
    working_dataframe = dataframe.copy()

    train_mask, split_info = build_episode_based_split_mask(
        working_dataframe,
        train_fraction=train_fraction,
        episode_column=split_episode_column,
        group_columns=split_group_columns,
        fallback_order_column=fallback_order_column,
    )

    working_dataframe, training_info = stamp_training_metadata(
        working_dataframe,
        train_mask=train_mask,
        train_flag_column="meta__is_train_flag",
        train_label_column="meta__is_train",
        train_mask_column="meta__train_mask",
    )

    candidate_feature_columns = list(feature_registry.get("feature_columns", []))
    if select_numeric_only:
        selected_feature_columns, feature_selection_info = select_numeric_feature_columns(
            working_dataframe,
            feature_columns=candidate_feature_columns,
            exclude_columns=exclude_feature_columns,
        )
    else:
        selected_feature_columns = [
            column_name
            for column_name in candidate_feature_columns
            if column_name in working_dataframe.columns
            and column_name not in set(exclude_feature_columns or [])
        ]
        feature_selection_info = {
            "selected_numeric_feature_count": int(len(selected_feature_columns)),
            "selected_numeric_feature_columns": list(selected_feature_columns),
            "rejected_columns": [],
        }

    ohe_info = {"applied": False, "one_hot_columns": [], "created_columns": []}
    if apply_one_hot_encoding:
        ohe_candidates = list(one_hot_columns or feature_registry.get("one_hot_encoding_columns", []))
        working_dataframe, ohe_info = apply_one_hot_encoding_from_truths(
            working_dataframe,
            one_hot_columns=ohe_candidates,
            drop_first=False,
            dtype="uint8",
        )

        # Rebuild selected feature list with created OHE columns
        created_columns = list(ohe_info.get("created_columns", []))
        selected_feature_columns = [
            column_name
            for column_name in selected_feature_columns
            if column_name not in set(ohe_info.get("one_hot_columns", []))
        ] + created_columns

    working_dataframe, imputer, imputation_info = apply_imputation(
        working_dataframe,
        feature_columns=selected_feature_columns,
        method=imputation_method,
    )

    scaled_dataframe, scaler, scaling_info = fit_and_apply_scaler(
        working_dataframe,
        feature_columns=selected_feature_columns,
        train_mask=working_dataframe["meta__train_mask"],
        scaler_kind=scaler_kind,
    )

    fit_dataframe, fit_info = get_training_rows_for_unsupervised_model(
        scaled_dataframe,
        train_mask_column="meta__train_mask",
        anomaly_flag_column="anomaly_flag",
    )

    train_dataframe = scaled_dataframe.loc[scaled_dataframe["meta__train_mask"].astype(bool)].copy()
    test_dataframe = scaled_dataframe.loc[~scaled_dataframe["meta__train_mask"].astype(bool)].copy()

    frames = {
        "gold_preprocessed": working_dataframe.copy(),
        "gold_preprocessed_scaled": scaled_dataframe.copy(),
        "gold_fit": fit_dataframe.copy(),
        "gold_train": train_dataframe.copy(),
        "gold_test": test_dataframe.copy(),
    }

    runtime_info = {
        "split_info": split_info,
        "training_info": training_info,
        "feature_selection_info": feature_selection_info,
        "ohe_info": ohe_info,
        "imputation_info": imputation_info,
        "scaling_info": scaling_info,
        "fit_info": fit_info,
        "selected_feature_columns": list(selected_feature_columns),
    }

    learned_objects = {
        "imputer": imputer,
        "scaler": scaler,
    }

    return frames, runtime_info, learned_objects


def build_gold_support_artifacts(
    scaled_dataframe: pd.DataFrame,
    *,
    selected_feature_columns: Sequence[str],
    train_mask: pd.Series,
    baseline_feature_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Build downstream support artifacts for baseline and cascade.
    """
    train_mask = train_mask.astype(bool).reindex(scaled_dataframe.index)

    reference_profile = build_reference_profile(
        scaled_dataframe,
        feature_columns=selected_feature_columns,
        subset_mask=train_mask,
    )

    stage2_feature_columns, stage2_info = choose_stage2_features_from_training_stability(
        scaled_dataframe,
        feature_columns=selected_feature_columns,
        train_mask=train_mask,
        min_non_null_ratio=0.95,
        min_variance=1e-12,
    )

    stage3_sensor_groups = build_stage3_sensor_groups(stage2_feature_columns)

    return {
        "baseline_feature_columns": list(baseline_feature_columns or selected_feature_columns),
        "reference_profile": reference_profile,
        "stage2_feature_columns": stage2_feature_columns,
        "stage2_info": stage2_info,
        "stage3_sensor_groups": stage3_sensor_groups,
    }