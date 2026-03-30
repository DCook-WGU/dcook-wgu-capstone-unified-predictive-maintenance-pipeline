"""
utils/pipeline/silver_eda_dropped.py

Dropped-feature parquet helpers for Silver EDA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from utils.file_io import load_data
from utils.pipeline.silver_eda_profiles import (
    build_state_sensor_profile_table,
    build_feature_behavior_effect_size_table,
)


def load_dropped_sensor_dataframe(
    *,
    dropped_path: Path,
) -> pd.DataFrame:
    """
    Load dropped-feature dataframe/parquet.
    """
    if not dropped_path.exists():
        raise FileNotFoundError(f"Dropped sensor dataframe not found: {dropped_path}")

    return load_data(dropped_path.parent, dropped_path.name)


def attach_state_column_to_dropped_dataframe(
    dropped_dataframe: pd.DataFrame,
    *,
    silver_dataframe: pd.DataFrame,
    state_column: str,
    synthetic_state_column: str,
    join_key: str = "meta__record_id",
) -> pd.DataFrame:
    """
    Join state columns from Silver dataframe onto dropped-feature dataframe.
    """
    if join_key not in dropped_dataframe.columns:
        raise KeyError(f"Dropped dataframe missing join key: {join_key}")
    if join_key not in silver_dataframe.columns:
        raise KeyError(f"Silver dataframe missing join key: {join_key}")
    if state_column not in silver_dataframe.columns:
        raise KeyError(f"Silver dataframe missing state column: {state_column}")
    if synthetic_state_column not in silver_dataframe.columns:
        raise KeyError(f"Silver dataframe missing synthetic state column: {synthetic_state_column}")

    right_columns = [join_key, state_column, synthetic_state_column]
    right_frame = silver_dataframe[right_columns].drop_duplicates(subset=[join_key]).copy()

    merged = dropped_dataframe.merge(
        right_frame,
        on=join_key,
        how="left",
    )
    return merged


def build_dropped_sensor_profiles_from_silver_preeda_truth(
    dropped_dataframe: pd.DataFrame,
    *,
    dropped_feature_columns: Sequence[str],
    state_column: str,
    state_values: Sequence[str],
    baseline_state: str = "normal",
    comparison_states: Sequence[str] = ("abnormal", "recovery"),
) -> dict:
    """
    Build dropped-feature profile and effect-size tables.
    """
    profile_df = build_state_sensor_profile_table(
        dropped_dataframe,
        sensors=list(dropped_feature_columns),
        state_column=state_column,
        state_values=list(state_values),
    )

    effect_size_df = build_feature_behavior_effect_size_table(
        dropped_dataframe,
        sensors=list(dropped_feature_columns),
        state_column=state_column,
        baseline_state=baseline_state,
        comparison_states=list(comparison_states),
    )

    return {
        "profile_df": profile_df,
        "effect_size_df": effect_size_df,
    }