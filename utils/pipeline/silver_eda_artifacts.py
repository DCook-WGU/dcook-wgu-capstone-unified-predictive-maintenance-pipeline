"""
utils/silver_eda_artifacts.py

Artifact save helpers for Silver EDA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from utils.file_io import save_data, save_json


def save_eda_table_artifact(
    dataframe: pd.DataFrame,
    *,
    output_dir: Path,
    file_name: str,
) -> str:
    """
    Save table artifact and return full path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_data(dataframe, output_dir, file_name)
    return str(output_dir / file_name)


def save_eda_json_artifact(
    payload: Dict[str, Any],
    *,
    output_path: Path,
) -> str:
    """
    Save JSON artifact and return full path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, output_path)
    return str(output_path)


def save_episode_status_counts_json(
    episode_status_counts_df: pd.DataFrame,
    *,
    output_path: Path,
) -> str:
    """
    Save episode status counts as JSON records.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(episode_status_counts_df.to_dict(orient="records"), output_path)
    return str(output_path)


def build_silver_eda_artifact_index(
    *,
    artifact_paths: Dict[str, str],
    summary_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a compact artifact index payload.
    """
    return {
        "artifact_count": int(len(artifact_paths)),
        "artifact_paths": dict(artifact_paths),
        "summary_payload": dict(summary_payload),
    }