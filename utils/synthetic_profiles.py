from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SensorRichProfile:
    sensor: str
    state_scope: str

    mean: float
    std: float
    min_value: float
    max_value: float

    median: float
    iqr: float

    p01: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float

    skewness: float
    kurtosis: float
    robust_std: float
    distribution_family: str

    lower_bound: float
    upper_bound: float


def _require_columns(dataframe: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def load_rich_profile_csv(path: str, state_scope: str) -> Dict[str, SensorRichProfile]:
    dataframe = pd.read_csv(path)

    _require_columns(
        dataframe,
        [
            "sensor", "mean", "std", "min", "max",
            "median", "iqr",
            "p01", "p05", "p25", "p50", "p75", "p95", "p99",
            "skewness", "kurtosis", "robust_std",
            "distribution_family", "lower_bound", "upper_bound",
        ],
        f"rich_profile({state_scope})",
    )

    out: Dict[str, SensorRichProfile] = {}
    for _, row in dataframe.iterrows():
        sensor = str(row["sensor"])
        out[sensor] = SensorRichProfile(
            sensor=sensor,
            state_scope=state_scope,
            mean=float(row["mean"]),
            std=float(row["std"]) if pd.notna(row["std"]) else 0.0,
            min_value=float(row["min"]),
            max_value=float(row["max"]),
            median=float(row["median"]),
            iqr=float(row["iqr"]),
            p01=float(row["p01"]),
            p05=float(row["p05"]),
            p25=float(row["p25"]),
            p50=float(row["p50"]),
            p75=float(row["p75"]),
            p95=float(row["p95"]),
            p99=float(row["p99"]),
            skewness=float(row["skewness"]) if pd.notna(row["skewness"]) else float("nan"),
            kurtosis=float(row["kurtosis"]) if pd.notna(row["kurtosis"]) else float("nan"),
            robust_std=float(row["robust_std"]) if pd.notna(row["robust_std"]) else 0.0,
            distribution_family=str(row["distribution_family"]),
            lower_bound=float(row["lower_bound"]),
            upper_bound=float(row["upper_bound"]),
        )
    return out


def load_correlation_pairs_csv(path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    _require_columns(dataframe, ["sensor_a", "sensor_b", "pearson_corr", "spearman_corr"], "correlation_pairs")
    return dataframe


def load_group_map_csv(path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    _require_columns(dataframe, ["group_name", "sensor"], "group_map")
    return dataframe


def load_fault_pairings_csv(path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    _require_columns(dataframe, ["sensor_primary", "sensor_secondary", "fault_coupling_strength", "lag_cycles"], "fault_pairings")
    return dataframe