from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MissingnessSpec:
    """
    Holds missingness audit objects loaded from truth.runtime_facts.missingness_quarantine

    missingness_pct_all: dict[sensor] -> percent missing (0..100)
    missingness_pct_by_state: dict[state] -> dict[sensor] -> percent missing (0..100)
    missingness_state_dependent_flag: dict[sensor] -> bool
    """
    missingness_pct_all: Dict[str, float]
    missingness_pct_by_state: Dict[str, Dict[str, float]]
    missingness_state_dependent_flag: Dict[str, bool]
    state_list: List[str]
    state_col_synth: str


def build_missingness_spec_from_truth_payload(payload: Dict[str, object]) -> MissingnessSpec:
    pct_all = dict(payload.get("missingness_pct_all") or {})
    pct_by_state = dict(payload.get("missingness_pct_by_state") or {})
    dep = dict(payload.get("missingness_state_dependent_flag") or {})
    gate = dict(payload.get("missingness_state_gate_params") or {})

    state_list = list(gate.get("state_list") or ["normal", "abnormal", "recovery"])
    state_col_synth = str(gate.get("state_col_synth") or "machine_status__synthetic")

    # normalize nested dict
    pct_by_state_norm: Dict[str, Dict[str, float]] = {}
    for st in state_list:
        pct_by_state_norm[st] = dict(pct_by_state.get(st) or {})

    return MissingnessSpec(
        missingness_pct_all={str(k): float(v) for k, v in pct_all.items()},
        missingness_pct_by_state={
            str(st): {str(k): float(v) for k, v in (pct_by_state_norm.get(st) or {}).items()}
            for st in state_list
        },
        missingness_state_dependent_flag={str(k): bool(v) for k, v in dep.items()},
        state_list=[str(s) for s in state_list],
        state_col_synth=state_col_synth,
    )


def _pct_to_present_count(n: int, missing_pct: float) -> int:
    """
    Convert missing% to an EXACT present count for n rows.
    Uses rounding to nearest int; clamps to [0, n].
    """
    if not np.isfinite(missing_pct):
        return n
    missing_pct = float(missing_pct)
    missing_pct = max(0.0, min(100.0, missing_pct))
    target_missing = int(round(n * (missing_pct / 100.0)))
    target_missing = max(0, min(n, target_missing))
    return n - target_missing


def apply_exact_missingness_mask(
    df: pd.DataFrame,
    *,
    sensor_cols: Sequence[str],
    rng: np.random.Generator,
    present_counts: Dict[str, int],
    eligible_row_idx: np.ndarray,
) -> pd.DataFrame:
    """
    Forces EXACT present counts per sensor within eligible rows.
    - present_counts[sensor] = number of non-null values to keep inside eligible_row_idx
    - all other eligible rows for that sensor become NaN
    - rows outside eligible_row_idx are not touched
    """
    out = df.copy()

    idx = np.asarray(eligible_row_idx, dtype=int)
    n = int(idx.shape[0])
    if n == 0:
        return out

    for sensor in sensor_cols:
        if sensor not in out.columns:
            continue

        keep_n = int(present_counts.get(sensor, n))
        keep_n = max(0, min(n, keep_n))

        # choose which eligible rows remain present
        if keep_n == n:
            continue
        if keep_n == 0:
            out.loc[idx, sensor] = np.nan
            continue

        keep_idx = rng.choice(idx, size=keep_n, replace=False)
        keep_mask = pd.Series(False, index=idx)
        keep_mask.loc[keep_idx] = True

        drop_idx = idx[~keep_mask.values]
        out.loc[drop_idx, sensor] = np.nan

    return out


def build_present_counts_for_block(
    *,
    sensors: Sequence[str],
    n_rows: int,
    pct_all: Dict[str, float],
    pct_by_state: Optional[Dict[str, float]] = None,
    use_by_state: bool = False,
) -> Dict[str, int]:
    """
    Returns dict[sensor] -> exact present count for a block of n_rows.
    If use_by_state=True, pct_by_state[sensor] is used when available; else pct_all.
    """
    present_counts: Dict[str, int] = {}
    for s in sensors:
        pct = None
        if use_by_state and pct_by_state is not None and s in pct_by_state:
            pct = pct_by_state.get(s)
        if pct is None:
            pct = pct_all.get(s, 0.0)

        present_counts[str(s)] = _pct_to_present_count(n_rows, float(pct) if pct is not None else 0.0)
    return present_counts