"""
utils/pipeline/silver_eda_plots.py

Plot helpers for Silver EDA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from utils.pipeline.silver_eda_profiles import z_score


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    *,
    output_path: Path | None = None,
):
    """
    Plot normal-only correlation heatmap.
    """
    if correlation_matrix.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(correlation_matrix.values, aspect="auto")
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=90)
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_yticklabels(correlation_matrix.index)
    ax.set_title("Normal-State Correlation Heatmap")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_state_distribution_histograms(
    dataframe: pd.DataFrame,
    *,
    features: Sequence[str],
    state_column: str,
    state_values: Sequence[str],
    output_dir: Path | None = None,
):
    """
    Plot feature distributions by state.
    """
    figures = []

    for feature_name in features:
        if feature_name not in dataframe.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))

        for state_name in state_values:
            state_df = dataframe.loc[dataframe[state_column] == state_name].copy()
            if state_df.empty:
                continue

            series = pd.to_numeric(state_df[feature_name], errors="coerce").dropna()
            if series.empty:
                continue

            ax.hist(series, bins=30, alpha=0.5, label=str(state_name))

        ax.set_title(f"State Distribution: {feature_name}")
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / f"distribution__{feature_name}.png", dpi=150, bbox_inches="tight")

        figures.append(fig)

    return figures


def build_flag_spans(
    dataframe: pd.DataFrame,
    *,
    flag_column: str = "anomaly_flag",
    x_column: str = "time_index",
) -> list[tuple[float, float]]:
    """
    Build contiguous spans for a binary flag column.
    """
    if flag_column not in dataframe.columns or x_column not in dataframe.columns:
        return []

    working = dataframe[[x_column, flag_column]].copy()
    working[flag_column] = pd.to_numeric(working[flag_column], errors="coerce").fillna(0).astype(int)
    working = working.sort_values(x_column).reset_index(drop=True)

    spans: list[tuple[float, float]] = []
    start_value = None

    for idx, row in working.iterrows():
        flag_value = int(row[flag_column])
        x_value = float(row[x_column])

        if flag_value == 1 and start_value is None:
            start_value = x_value

        is_last_row = idx == len(working) - 1
        next_flag = 0 if is_last_row else int(working.loc[idx + 1, flag_column])

        if start_value is not None and (flag_value == 1 and next_flag == 0):
            end_value = x_value
            spans.append((start_value, end_value))
            start_value = None

    return spans


def resolve_time_axis_series(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Resolve a preferred time axis for plotting.
    """
    if "event_time" in dataframe.columns and dataframe["event_time"].notna().any():
        return pd.to_datetime(dataframe["event_time"], errors="coerce")
    if "event_step" in dataframe.columns:
        return pd.to_numeric(dataframe["event_step"], errors="coerce")
    if "time_index" in dataframe.columns:
        return pd.to_numeric(dataframe["time_index"], errors="coerce")

    return pd.Series(range(len(dataframe)), index=dataframe.index)


def plot_top_feature_overlay(
    dataframe: pd.DataFrame,
    *,
    features: Sequence[str],
    output_path: Path | None = None,
):
    """
    Plot z-scored overlay of top features across a shared time axis.
    """
    use_features = [feature for feature in features if feature in dataframe.columns]
    if len(use_features) == 0:
        return None

    x_series = resolve_time_axis_series(dataframe)

    fig, ax = plt.subplots(figsize=(10, 5))
    for feature_name in use_features:
        series = pd.to_numeric(dataframe[feature_name], errors="coerce")
        ax.plot(x_series, z_score(series), label=feature_name)

    for start_value, end_value in build_flag_spans(dataframe):
        ax.axvspan(start_value, end_value, alpha=0.15)

    ax.set_title("Top Feature Overlay (Z-Scored)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Z-Score")
    ax.legend()
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_timeseries_with_flag_spans(
    dataframe: pd.DataFrame,
    *,
    features: Sequence[str],
    output_dir: Path | None = None,
):
    """
    Plot one feature at a time with anomaly spans overlaid.
    """
    figures = []
    use_features = [feature for feature in features if feature in dataframe.columns]
    x_series = resolve_time_axis_series(dataframe)
    spans = build_flag_spans(dataframe)

    for feature_name in use_features:
        fig, ax = plt.subplots(figsize=(10, 4))
        series = pd.to_numeric(dataframe[feature_name], errors="coerce")
        ax.plot(x_series, series, label=feature_name)

        for start_value, end_value in spans:
            ax.axvspan(start_value, end_value, alpha=0.15)

        ax.set_title(f"Feature Timeseries with Anomaly Spans: {feature_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel(feature_name)
        ax.legend()
        plt.tight_layout()

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / f"timeseries__{feature_name}.png", dpi=150, bbox_inches="tight")

        figures.append(fig)

    return figures


def plot_aligned_onset_series(
    onset_summary_df: pd.DataFrame,
    *,
    feature_name: str,
    output_path: Path | None = None,
):
    """
    Plot aligned onset mean series for one feature.
    """
    mean_column = f"{feature_name}__mean"
    if onset_summary_df.empty or mean_column not in onset_summary_df.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(onset_summary_df["relative_step"], onset_summary_df[mean_column])
    ax.axvline(0, linestyle="--")
    ax.set_title(f"Aligned Onset Mean: {feature_name}")
    ax.set_xlabel("Relative Step")
    ax.set_ylabel("Mean Value")
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig