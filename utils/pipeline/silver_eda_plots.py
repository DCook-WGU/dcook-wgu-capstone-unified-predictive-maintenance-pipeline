"""
utils/silver_eda_plots.py

Plot helpers for Silver EDA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from utils.pipeline.silver_eda_profiles import z_score


def plot_state_distribution(
    status_counts_df: pd.DataFrame,
    *,
    output_path: Path | None = None,
):
    """
    Plot row-level state distribution.
    """
    if status_counts_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(status_counts_df["status_state"].astype(str), status_counts_df["row_count"])
    ax.set_title("Status Distribution")
    ax.set_xlabel("State")
    ax.set_ylabel("Row Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_sensor_profile_comparison(
    dataframe: pd.DataFrame,
    *,
    sensors: Sequence[str],
    state_column: str,
    state_values: Sequence[str],
    output_dir: Path | None = None,
    use_z_score: bool = True,
):
    """
    Plot sensor profile series by state.
    """
    figures = []

    for sensor in sensors:
        if sensor not in dataframe.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))

        for state in state_values:
            state_df = dataframe.loc[dataframe[state_column] == state].copy()
            if state_df.empty:
                continue

            series = pd.to_numeric(state_df[sensor], errors="coerce")
            if use_z_score:
                series = z_score(series)

            ax.plot(series.reset_index(drop=True), label=str(state))

        ax.set_title(f"Sensor Profile Comparison: {sensor}")
        ax.set_xlabel("Observation Index")
        ax.set_ylabel("Z-Score" if use_z_score else "Value")
        ax.legend()
        plt.tight_layout()

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / f"profile__{sensor}.png", dpi=150, bbox_inches="tight")

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


def plot_normal_correlation_heatmap(
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