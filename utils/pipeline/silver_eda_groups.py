"""
utils/silver_eda_groups.py

Correlation grouping helpers for Silver EDA.
"""

from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd


def find(parent: dict, x: str) -> str:
    """
    Disjoint-set find with path compression.
    """
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]


def union(parent: dict, a: str, b: str) -> None:
    """
    Disjoint-set union.
    """
    root_a = find(parent, a)
    root_b = find(parent, b)
    if root_a != root_b:
        parent[root_b] = root_a


def build_normal_only_correlation_pairs(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    state_column: str,
    target_state: str = "normal",
) -> dict:
    """
    Build normal-only correlation outputs.
    """
    if state_column not in dataframe.columns:
        raise KeyError(f"Missing state column: {state_column}")

    use_features = [
        column for column in feature_columns
        if column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    normal_df = dataframe.loc[dataframe[state_column] == target_state, use_features].copy()

    if normal_df.empty or len(use_features) == 0:
        return {
            "correlation_matrix": pd.DataFrame(),
            "correlation_pairs": pd.DataFrame(),
        }

    correlation_matrix = normal_df.corr(numeric_only=True)

    pair_rows = []
    for sensor_a in correlation_matrix.columns:
        for sensor_b in correlation_matrix.columns:
            if sensor_a >= sensor_b:
                continue
            corr_value = correlation_matrix.loc[sensor_a, sensor_b]
            pair_rows.append(
                {
                    "sensor_a": sensor_a,
                    "sensor_b": sensor_b,
                    "correlation": float(corr_value) if pd.notna(corr_value) else None,
                    "abs_correlation": abs(float(corr_value)) if pd.notna(corr_value) else None,
                }
            )

    correlation_pairs = pd.DataFrame(pair_rows).sort_values(
        ["abs_correlation", "sensor_a", "sensor_b"],
        ascending=[False, True, True],
    ).reset_index(drop=True) if len(pair_rows) > 0 else pd.DataFrame()

    return {
        "correlation_matrix": correlation_matrix,
        "correlation_pairs": correlation_pairs,
    }


def build_sensor_group_map_from_correlation(
    correlation_matrix: pd.DataFrame,
    *,
    min_abs_corr_for_group: float = 0.60,
) -> pd.DataFrame:
    """
    Build connected-component sensor groups from correlation matrix.
    """
    if correlation_matrix.empty:
        return pd.DataFrame()

    abs_corr = correlation_matrix.abs()
    parent: Dict[str, str] = {}

    for sensor_name in abs_corr.columns:
        find(parent, str(sensor_name))

    for sensor_a in abs_corr.columns:
        for sensor_b in abs_corr.columns:
            if sensor_a >= sensor_b:
                continue
            corr_strength = abs_corr.loc[sensor_a, sensor_b]
            if pd.notna(corr_strength) and float(corr_strength) >= float(min_abs_corr_for_group):
                union(parent, str(sensor_a), str(sensor_b))

    groups: Dict[str, list[str]] = {}
    for sensor_name in abs_corr.columns:
        root = find(parent, str(sensor_name))
        groups.setdefault(root, []).append(str(sensor_name))

    rows = []
    for group_priority, members in enumerate(
        sorted(groups.values(), key=lambda values: (-len(values), values[0])),
        start=1,
    ):
        group_name = f"group_{group_priority:03d}"
        members = sorted(members)
        for position, sensor_name in enumerate(members):
            rows.append(
                {
                    "group_name": group_name,
                    "sensor": sensor_name,
                    "group_size": len(members),
                    "group_priority": group_priority,
                    "group_role": "leader" if position == 0 else "member",
                    "group_method": f"corr_components_abs>={min_abs_corr_for_group}",
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["group_priority", "sensor"]
    ).reset_index(drop=True) if len(rows) > 0 else pd.DataFrame()