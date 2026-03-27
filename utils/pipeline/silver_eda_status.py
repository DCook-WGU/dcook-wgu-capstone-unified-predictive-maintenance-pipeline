"""
utils/silver_eda_status.py

Status and episode summary helpers for Silver EDA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import json
import pandas as pd


def resolve_state_column_from_truth(
    truth_record: dict,
    *,
    fallback_status_column: str = "machine_status",
) -> str:
    """
    Resolve the label/status source column from parent Silver truth.
    """
    runtime_facts = (truth_record or {}).get("runtime_facts", {}) or {}
    label_resolution = runtime_facts.get("label_resolution", {}) or {}

    label_source_column = label_resolution.get("label_source_column")
    if label_source_column is not None and str(label_source_column).strip() != "":
        return str(label_source_column).strip()

    return fallback_status_column


def build_state_col_synth(
    dataframe: pd.DataFrame,
    *,
    status_column: str,
    state_map: Optional[dict] = None,
    output_column: Optional[str] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Build standardized synthetic state column from the status column.
    """
    if status_column not in dataframe.columns:
        raise KeyError(f"Missing status column: {status_column}")

    working = dataframe.copy()
    output_column = output_column or f"{status_column}__synthetic"

    series = (
        working[status_column]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    if state_map is not None:
        mapped = series.map(state_map)
        series = mapped.fillna(series)

    working[output_column] = series
    return working, output_column


def get_episode_status_state_stats(
    dataframe: pd.DataFrame,
    status_column: str = "machine_status",
    episode_column: str = "meta__episode_id",
    state_order: list | None = None,
    include_null_episode: bool = False,
    state_map: dict | None = None,
    lowercase_states: bool = True,
    strip_states: bool = True,
    percent_suffix: str = "_percent",
) -> dict:
    """
    Build episode-aware status/state summaries.
    """
    required_columns = [status_column, episode_column]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    work_dataframe = dataframe.copy()

    state_series = work_dataframe[status_column].copy()
    null_mask = state_series.isna()

    state_series = state_series.astype("string")

    if strip_states:
        state_series = state_series.str.strip()

    if lowercase_states:
        state_series = state_series.str.lower()

    if state_map is not None:
        state_series = state_series.map(state_map).fillna(state_series)

    state_series = state_series.where(~null_mask, pd.NA)

    normalized_status_column = f"{status_column}_normalized"
    work_dataframe[normalized_status_column] = state_series

    global_status_stats = (
        work_dataframe[normalized_status_column]
        .value_counts(dropna=False)
        .rename_axis("status_state")
        .reset_index(name="global_count")
    )

    total_rows = global_status_stats["global_count"].sum()
    global_status_stats["global_percent"] = (
        global_status_stats["global_count"] / total_rows * 100
    ).round(2)

    if include_null_episode:
        episode_dataframe = work_dataframe.copy()
    else:
        episode_dataframe = work_dataframe[
            work_dataframe[episode_column].notna()
        ].copy()

    episode_status_counts = (
        episode_dataframe.groupby([episode_column, normalized_status_column], dropna=False)
        .size()
        .unstack(fill_value=0)
    )

    if state_order is None:
        state_order = list(episode_status_counts.columns)

    for state in state_order:
        if state not in episode_status_counts.columns:
            episode_status_counts[state] = 0

    episode_status_counts = episode_status_counts[state_order]
    episode_status_counts = episode_status_counts.reset_index()

    status_columns = [column for column in episode_status_counts.columns if column != episode_column]

    episode_status_counts["episode_total_rows"] = episode_status_counts[status_columns].sum(axis=1)

    for column in status_columns:
        episode_status_counts[f"{column}{percent_suffix}"] = (
            episode_status_counts[column] / episode_status_counts["episode_total_rows"] * 100
        ).fillna(0.0).round(2)

    episode_count = int(episode_status_counts[episode_column].nunique(dropna=True))

    episode_status_means = (
        episode_status_counts[status_columns]
        .mean(axis=0)
        .round(4)
        .rename_axis("status_state")
        .reset_index(name="mean_count_per_episode")
    )

    percent_columns = [f"{column}{percent_suffix}" for column in status_columns]
    episode_status_percent_means = (
        episode_status_counts[percent_columns]
        .mean(axis=0)
        .round(4)
        .rename_axis("status_state_percent")
        .reset_index(name="mean_percent_per_episode")
    )

    episode_totals = (
        episode_status_counts[[episode_column, "episode_total_rows"]]
        .copy()
        .sort_values(episode_column)
        .reset_index(drop=True)
    )

    return {
        "status_column": status_column,
        "episode_column": episode_column,
        "episode_count": episode_count,
        "mean_total_rows_per_episode": float(episode_status_counts["episode_total_rows"].mean() or 0.0),
        "global_status_stats": global_status_stats.to_dict(orient="records"),
        "episode_status_counts": episode_status_counts,
        "episode_status_means": episode_status_means.to_dict(orient="records"),
        "episode_status_percent_means": episode_status_percent_means.to_dict(orient="records"),
        "episode_totals": episode_totals.to_dict(orient="records"),
        "episode_status_mean_lookup": dict(
            zip(episode_status_means["status_state"], episode_status_means["mean_count_per_episode"])
        ),
        "episode_status_percent_mean_lookup": dict(
            zip(
                episode_status_percent_means["status_state_percent"],
                episode_status_percent_means["mean_percent_per_episode"],
            )
        ),
        "global_status_count_lookup": dict(
            zip(global_status_stats["status_state"], global_status_stats["global_count"])
        ),
        "global_status_percent_lookup": dict(
            zip(global_status_stats["status_state"], global_status_stats["global_percent"])
        ),
    }


def build_episode_status_payload_and_tables(
    dataframe: pd.DataFrame,
    *,
    status_column: str,
    episode_column: str = "meta__episode_id",
    state_map: Optional[dict] = None,
) -> dict:
    """
    Build the exact payload and tables used by Silver EDA.
    """
    payload = get_episode_status_state_stats(
        dataframe,
        status_column=status_column,
        episode_column=episode_column,
        state_map=state_map,
        include_null_episode=False,
        lowercase_states=True,
        strip_states=True,
    )

    episode_status_counts_df = payload["episode_status_counts"]

    episode_status_means_df = pd.DataFrame(payload["episode_status_means"])
    episode_status_percent_means_df = pd.DataFrame(payload["episode_status_percent_means"])
    episode_totals_df = pd.DataFrame(payload["episode_totals"])
    global_status_stats_df = pd.DataFrame(payload["global_status_stats"])

    payload_no_df = {key: value for key, value in payload.items() if key != "episode_status_counts"}

    return {
        "payload": payload_no_df,
        "episode_status_counts_df": episode_status_counts_df,
        "episode_status_means_df": episode_status_means_df,
        "episode_status_percent_means_df": episode_status_percent_means_df,
        "episode_totals_df": episode_totals_df,
        "global_status_stats_df": global_status_stats_df,
    }


def build_status_distribution_tables(
    dataframe: pd.DataFrame,
    *,
    status_column: str,
) -> Dict[str, pd.DataFrame]:
    """
    Build lightweight row-level status distribution tables.
    """
    if status_column not in dataframe.columns:
        raise KeyError(f"Missing status column: {status_column}")

    work = dataframe.copy()
    status_counts = (
        work[status_column]
        .astype("string")
        .fillna("<NA>")
        .value_counts(dropna=False)
        .rename_axis("status_state")
        .reset_index(name="row_count")
    )
    status_counts["row_percent"] = (
        status_counts["row_count"] / status_counts["row_count"].sum() * 100
    ).round(2)

    return {
        "status_counts": status_counts,
    }


def pull_episode_status_state_stats_from_truth(truth_record: dict) -> dict:
    """
    Pull saved episode-state summary payload back out of Silver EDA truth.
    """
    runtime_facts = (truth_record or {}).get("runtime_facts", {}) or {}
    artifact_paths = (truth_record or {}).get("artifact_paths", {}) or {}

    payload = runtime_facts.get("episode_status_state_stats", {}) or {}
    episode_status_counts_path = artifact_paths.get("episode_status_counts_path")

    episode_status_counts_records = []
    if episode_status_counts_path:
        path_obj = Path(episode_status_counts_path)
        if path_obj.exists():
            with open(path_obj, "r", encoding="utf-8") as file:
                episode_status_counts_records = json.load(file)

    return {
        "status_column": payload.get("status_column"),
        "episode_column": payload.get("episode_column"),
        "episode_count": int(payload.get("episode_count", 0) or 0),
        "mean_total_rows_per_episode": float(
            payload.get("mean_total_rows_per_episode", 0.0) or 0.0
        ),
        "global_status_stats": payload.get("global_status_stats", []) or [],
        "episode_status_means": payload.get("episode_status_means", []) or [],
        "episode_status_percent_means": payload.get("episode_status_percent_means", []) or [],
        "episode_totals": payload.get("episode_totals", []) or [],
        "episode_status_mean_lookup": payload.get("episode_status_mean_lookup", {}) or {},
        "episode_status_percent_mean_lookup": payload.get("episode_status_percent_mean_lookup", {}) or {},
        "global_status_count_lookup": payload.get("global_status_count_lookup", {}) or {},
        "global_status_percent_lookup": payload.get("global_status_percent_lookup", {}) or {},
        "episode_status_counts_path": episode_status_counts_path,
        "episode_status_counts_records": episode_status_counts_records,
    }