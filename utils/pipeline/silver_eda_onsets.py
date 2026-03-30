"""
utils/pipeline/silver_eda_onsets.py

Onset detection and alignment helpers for Silver EDA.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def find_anomaly_onsets(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Find start rows of anomaly periods.
    """
    if "anomaly_flag" not in dataframe.columns:
        return pd.DataFrame(columns=["meta__asset_id", "meta__run_id", "time_index", "event_step"])

    grouping_columns = []
    if "meta__asset_id" in dataframe.columns:
        grouping_columns.append("meta__asset_id")
    if "meta__run_id" in dataframe.columns:
        grouping_columns.append("meta__run_id")

    working = dataframe.copy()

    if "event_step" not in working.columns and "time_index" in working.columns:
        working["event_step"] = working["time_index"]

    if "time_index" not in working.columns:
        working["time_index"] = np.arange(len(working), dtype=np.int64)

    if len(grouping_columns) > 0:
        working = working.sort_values(grouping_columns + ["event_step"]).reset_index(drop=True)
        shifted = working.groupby(grouping_columns, dropna=False)["anomaly_flag"].shift(1)
    else:
        working = working.sort_values(["event_step"]).reset_index(drop=True)
        shifted = working["anomaly_flag"].shift(1)

    onset_mask = (working["anomaly_flag"] == 1) & (shifted.fillna(0) == 0)
    onsets = working.loc[onset_mask, grouping_columns + ["time_index", "event_step"]].copy()
    return onsets.reset_index(drop=True)


def sample_onsets_evenly(onsets: pd.DataFrame, max_count: int) -> pd.DataFrame:
    """
    Evenly subsample onset rows for aligned plots/tables.
    """
    if len(onsets) <= max_count:
        return onsets.reset_index(drop=True)

    indices = np.linspace(0, len(onsets) - 1, num=max_count)
    indices = [int(round(value)) for value in indices]
    indices = sorted(list(set(indices)))
    return onsets.iloc[indices].reset_index(drop=True)


def build_aligned_onset_windows(
    dataframe: pd.DataFrame,
    *,
    onsets: pd.DataFrame,
    feature_columns: Sequence[str],
    pre_window: int = 20,
    post_window: int = 20,
    join_columns: Sequence[str] = ("meta__asset_id", "meta__run_id"),
) -> pd.DataFrame:
    """
    Build aligned windows around each anomaly onset.
    """
    if len(onsets) == 0:
        return pd.DataFrame()

    required_columns = [column for column in join_columns if column in dataframe.columns]
    required_columns += ["time_index"]

    for column in required_columns:
        if column not in dataframe.columns:
            raise KeyError(f"Missing required column for onset alignment: {column}")

    use_features = [column for column in feature_columns if column in dataframe.columns]
    if len(use_features) == 0:
        return pd.DataFrame()

    work = dataframe.copy()
    rows: list[pd.DataFrame] = []

    for onset_id, onset_row in onsets.reset_index(drop=True).iterrows():
        mask = pd.Series(True, index=work.index)

        for column in join_columns:
            if column in work.columns and column in onset_row.index:
                mask &= (work[column] == onset_row[column])

        onset_time_index = int(onset_row["time_index"])
        local = work.loc[mask].copy()

        local = local[
            (local["time_index"] >= onset_time_index - pre_window)
            & (local["time_index"] <= onset_time_index + post_window)
        ].copy()

        if len(local) == 0:
            continue

        local["relative_step"] = local["time_index"].astype(int) - onset_time_index
        local["onset_id"] = int(onset_id)

        id_columns = [column for column in join_columns if column in local.columns]
        keep_columns = id_columns + ["time_index", "relative_step", "onset_id"] + use_features
        rows.append(local[keep_columns])

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.concat(rows, axis=0, ignore_index=True)


def summarize_onset_windows(
    aligned_windows: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Summarize aligned onset windows by relative step.
    """
    if aligned_windows.empty:
        return pd.DataFrame()

    use_features = [column for column in feature_columns if column in aligned_windows.columns]
    if len(use_features) == 0:
        return pd.DataFrame()

    summary_rows = []
    grouped = aligned_windows.groupby("relative_step", dropna=False)

    for relative_step, group_frame in grouped:
        row = {
            "relative_step": int(relative_step),
            "window_count": int(group_frame["onset_id"].nunique()) if "onset_id" in group_frame.columns else int(len(group_frame)),
        }
        for feature_name in use_features:
            series = pd.to_numeric(group_frame[feature_name], errors="coerce")
            row[f"{feature_name}__mean"] = float(series.mean()) if series.notna().any() else None
            row[f"{feature_name}__median"] = float(series.median()) if series.notna().any() else None
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values("relative_step").reset_index(drop=True)