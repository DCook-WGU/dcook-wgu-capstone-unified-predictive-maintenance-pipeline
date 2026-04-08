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
    """
    Build MissingnessSpec from truth.runtime_facts.missingness_quarantine.

    Permanent behavior:
    - use missingness_pct_all / missingness_pct_by_state / dependency flags when present
    - merge dropped-sensor global missingness from `dropped_missing_pct`
    - optionally merge dropped-sensor state missingness from `dropped_missing_pct_by_state`
      if that key is added to the payload later
    - ensure dropped sensors get a default dependency flag
    - normalize the historical typo in state_col_synth
    """
    pct_all = dict(payload.get("missingness_pct_all") or {})
    pct_by_state = dict(payload.get("missingness_pct_by_state") or {})
    dep = dict(payload.get("missingness_state_dependent_flag") or {})
    gate = dict(payload.get("missingness_state_gate_params") or {})

    dropped_features = [str(s).strip() for s in (payload.get("dropped_features") or []) if str(s).strip()]
    dropped_missing_pct = dict(payload.get("dropped_missing_pct") or {})
    dropped_missing_pct_by_state = dict(payload.get("dropped_missing_pct_by_state") or {})

    state_list = list(gate.get("state_list") or ["normal", "abnormal", "recovery"])
    state_col_synth = str(gate.get("state_col_synth") or "machine_status__synthetic")

    # normalize the historical typo
    if state_col_synth == "machine_status__synethic":
        state_col_synth = "machine_status__synthetic"

    # normalize nested by-state dict first
    pct_by_state_norm: Dict[str, Dict[str, float]] = {}
    for st in state_list:
        pct_by_state_norm[str(st)] = dict(pct_by_state.get(st) or {})

    # ------------------------------------------------------------
    # Permanent fix 1:
    # merge dropped-sensor overall missingness back into pct_all
    # ------------------------------------------------------------
    for sensor_name in dropped_features:
        if sensor_name in dropped_missing_pct:
            pct_all[sensor_name] = float(dropped_missing_pct[sensor_name])

        if sensor_name not in dep:
            dep[sensor_name] = False

    # ------------------------------------------------------------
    # Permanent fix 2:
    # if dropped-sensor by-state missingness is available in the future,
    # merge it too and mark those sensors as state-dependent
    #
    # expected shape:
    # dropped_missing_pct_by_state = {
    #   "normal": {"sensor_50": 12.3, ...},
    #   "abnormal": {"sensor_50": 0.0, ...},
    #   "recovery": {"sensor_50": 55.0, ...},
    # }
    # ------------------------------------------------------------
    for st in state_list:
        st_key = str(st)
        dropped_state_map = dict(dropped_missing_pct_by_state.get(st_key) or {})

        if not dropped_state_map:
            continue

        for sensor_name, pct_value in dropped_state_map.items():
            sensor_name = str(sensor_name).strip()
            if sensor_name == "":
                continue

            pct_by_state_norm.setdefault(st_key, {})
            pct_by_state_norm[st_key][sensor_name] = float(pct_value)
            dep[sensor_name] = True

    return MissingnessSpec(
        missingness_pct_all={str(k): float(v) for k, v in pct_all.items()},
        missingness_pct_by_state={
            str(st): {
                str(k): float(v)
                for k, v in (pct_by_state_norm.get(str(st)) or {}).items()
            }
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

def apply_clustered_missingness_mask(
    df: pd.DataFrame,
    *,
    sensor_cols: Sequence[str],
    rng: np.random.Generator,
    present_counts: Dict[str, int],
    eligible_row_idx: np.ndarray,
    mean_gap_len: int = 3,
    long_gap_probability: float = 0.15,
) -> pd.DataFrame:
    """
    Forces approximate clustered missingness within eligible rows while still
    respecting the exact present_count target per sensor.
    """
    out = df.copy()

    idx = np.asarray(sorted(eligible_row_idx), dtype=int)
    n = int(idx.shape[0])
    if n == 0:
        return out

    for sensor in sensor_cols:
        if sensor not in out.columns:
            continue

        keep_n = int(present_counts.get(sensor, n))
        keep_n = max(0, min(n, keep_n))
        drop_n = n - keep_n

        if drop_n <= 0:
            continue
        if drop_n >= n:
            out.loc[idx, sensor] = np.nan
            continue

        selected_drop_positions: list[int] = []
        used = np.zeros(n, dtype=bool)

        while len(selected_drop_positions) < drop_n:
            start = int(rng.integers(0, n))
            if used[start]:
                continue

            if rng.random() < long_gap_probability:
                gap_len = int(max(2, round(rng.normal(mean_gap_len * 2.0, 1.5))))
            else:
                gap_len = int(max(1, round(rng.normal(mean_gap_len, 1.0))))

            for pos in range(start, min(start + gap_len, n)):
                if not used[pos]:
                    used[pos] = True
                    selected_drop_positions.append(pos)
                    if len(selected_drop_positions) >= drop_n:
                        break

        drop_idx = idx[np.array(sorted(selected_drop_positions[:drop_n]), dtype=int)]
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