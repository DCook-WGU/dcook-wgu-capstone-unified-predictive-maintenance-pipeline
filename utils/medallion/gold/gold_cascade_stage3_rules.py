"""
utils/gold_cascade_stage3_rules.py

Stage 3 rule helpers for the Gold cascade modeling stage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def compute_primary_breach_count(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    reference_profile: Dict[str, Any],
    z_threshold: float = 2.5,
    output_column: str = "stage3_primary_breach_count",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Count strong breaches relative to reference profile using z-style deviation.
    """
    working_dataframe = dataframe.copy()
    profile_summary = reference_profile.get("summary", {})
    breach_counts = np.zeros(len(working_dataframe), dtype=int)

    used_features: List[str] = []

    for column_name in feature_columns:
        if column_name not in working_dataframe.columns:
            continue
        if column_name not in profile_summary:
            continue

        mean_value = profile_summary[column_name].get("mean")
        std_value = profile_summary[column_name].get("std")

        if mean_value is None or std_value is None:
            continue
        if std_value == 0 or np.isnan(std_value):
            continue

        series = pd.to_numeric(working_dataframe[column_name], errors="coerce")
        z_values = np.abs((series - float(mean_value)) / float(std_value))
        breach_counts += (z_values >= float(z_threshold)).fillna(False).astype(int).to_numpy()
        used_features.append(column_name)

    working_dataframe[output_column] = breach_counts

    info = {
        "output_column": output_column,
        "z_threshold": float(z_threshold),
        "used_feature_count": int(len(used_features)),
        "used_features": used_features,
        "max_breach_count": int(np.max(breach_counts)) if len(breach_counts) > 0 else 0,
    }
    return working_dataframe, info


def compute_secondary_breach_count(
    dataframe: pd.DataFrame,
    *,
    sensor_groups: Dict[str, Sequence[str]],
    reference_profile: Dict[str, Any],
    z_threshold: float = 1.75,
    output_column: str = "stage3_secondary_breach_count",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Count weaker corroborating breaches grouped by sensor family.
    """
    working_dataframe = dataframe.copy()
    profile_summary = reference_profile.get("summary", {})
    group_breach_counts = np.zeros(len(working_dataframe), dtype=int)

    group_info: List[Dict[str, Any]] = []

    for group_name, columns in sensor_groups.items():
        group_has_breach = np.zeros(len(working_dataframe), dtype=bool)
        used_columns: List[str] = []

        for column_name in columns:
            if column_name not in working_dataframe.columns:
                continue
            if column_name not in profile_summary:
                continue

            mean_value = profile_summary[column_name].get("mean")
            std_value = profile_summary[column_name].get("std")

            if mean_value is None or std_value is None:
                continue
            if std_value == 0 or np.isnan(std_value):
                continue

            series = pd.to_numeric(working_dataframe[column_name], errors="coerce")
            z_values = np.abs((series - float(mean_value)) / float(std_value))
            group_has_breach |= (z_values >= float(z_threshold)).fillna(False).to_numpy()
            used_columns.append(column_name)

        group_breach_counts += group_has_breach.astype(int)
        group_info.append(
            {
                "group_name": group_name,
                "used_column_count": int(len(used_columns)),
                "used_columns": used_columns,
            }
        )

    working_dataframe[output_column] = group_breach_counts

    info = {
        "output_column": output_column,
        "z_threshold": float(z_threshold),
        "group_count": int(len(sensor_groups)),
        "group_info": group_info,
        "max_breach_count": int(np.max(group_breach_counts)) if len(group_breach_counts) > 0 else 0,
    }
    return working_dataframe, info


def compute_persistence_flag(
    dataframe: pd.DataFrame,
    *,
    candidate_column: str,
    group_columns: Sequence[str] = ("meta__asset_id", "meta__run_id"),
    min_consecutive_rows: int = 3,
    output_column: str = "stage3_persistence_flag",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Flag rows that are part of a persistent run of candidate anomalies.
    """
    working_dataframe = dataframe.copy()

    if candidate_column not in working_dataframe.columns:
        raise ValueError(f"Missing candidate column: {candidate_column}")

    candidate_series = pd.to_numeric(working_dataframe[candidate_column], errors="coerce").fillna(0).astype(int)
    persistence_flag = np.zeros(len(working_dataframe), dtype=int)

    grouped = working_dataframe.groupby(list(group_columns), dropna=False, sort=False)

    for _, index_values in grouped.indices.items():
        index_list = list(index_values)
        run_length = 0

        for idx in index_list:
            if candidate_series.iloc[idx] == 1:
                run_length += 1
            else:
                run_length = 0

            if run_length >= int(min_consecutive_rows):
                for mark_idx in range(idx - run_length + 1, idx + 1):
                    persistence_flag[mark_idx] = 1

    working_dataframe[output_column] = persistence_flag

    info = {
        "output_column": output_column,
        "candidate_column": candidate_column,
        "group_columns": list(group_columns),
        "min_consecutive_rows": int(min_consecutive_rows),
        "positive_count": int(np.sum(persistence_flag)),
    }
    return working_dataframe, info


def compute_drift_flag(
    dataframe: pd.DataFrame,
    *,
    score_column: str,
    group_columns: Sequence[str] = ("meta__asset_id", "meta__run_id"),
    rolling_window: int = 10,
    min_drift_delta: float = 0.10,
    output_column: str = "stage3_drift_flag",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Flag local score drift using rolling mean shift against prior score history.
    """
    working_dataframe = dataframe.copy()

    if score_column not in working_dataframe.columns:
        raise ValueError(f"Missing score column: {score_column}")

    working_dataframe[output_column] = 0
    score_series = pd.to_numeric(working_dataframe[score_column], errors="coerce")

    grouped = working_dataframe.groupby(list(group_columns), dropna=False, sort=False)

    for _, group_frame in grouped:
        group_index = group_frame.index
        group_scores = score_series.loc[group_index].astype(float)

        trailing_mean = group_scores.rolling(window=rolling_window, min_periods=rolling_window).mean()
        prior_mean = trailing_mean.shift(rolling_window)

        drift_condition = (trailing_mean - prior_mean) >= float(min_drift_delta)
        drift_condition = drift_condition.fillna(False).astype(int)

        working_dataframe.loc[group_index, output_column] = drift_condition.to_numpy()

    info = {
        "output_column": output_column,
        "score_column": score_column,
        "group_columns": list(group_columns),
        "rolling_window": int(rolling_window),
        "min_drift_delta": float(min_drift_delta),
        "positive_count": int(working_dataframe[output_column].sum()),
    }
    return working_dataframe, info


def compose_stage3_decision(
    dataframe: pd.DataFrame,
    *,
    primary_breach_column: str = "stage3_primary_breach_count",
    secondary_breach_column: str = "stage3_secondary_breach_count",
    persistence_column: str = "stage3_persistence_flag",
    drift_column: str = "stage3_drift_flag",
    min_primary_breaches: int = 1,
    min_secondary_breaches: int = 1,
    require_persistence_or_drift: bool = True,
    output_column: str = "stage3_confirmed_flag",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compose final Stage 3 confirmation flag from rule evidence.
    """
    working_dataframe = dataframe.copy()

    primary_ok = pd.to_numeric(working_dataframe[primary_breach_column], errors="coerce").fillna(0) >= int(min_primary_breaches)
    secondary_ok = pd.to_numeric(working_dataframe[secondary_breach_column], errors="coerce").fillna(0) >= int(min_secondary_breaches)
    persistence_ok = pd.to_numeric(working_dataframe[persistence_column], errors="coerce").fillna(0).astype(int) == 1
    drift_ok = pd.to_numeric(working_dataframe[drift_column], errors="coerce").fillna(0).astype(int) == 1

    if require_persistence_or_drift:
        confirmed = primary_ok & secondary_ok & (persistence_ok | drift_ok)
    else:
        confirmed = primary_ok & secondary_ok

    working_dataframe[output_column] = confirmed.astype(int)

    info = {
        "output_column": output_column,
        "min_primary_breaches": int(min_primary_breaches),
        "min_secondary_breaches": int(min_secondary_breaches),
        "require_persistence_or_drift": bool(require_persistence_or_drift),
        "positive_count": int(working_dataframe[output_column].sum()),
    }
    return working_dataframe, info