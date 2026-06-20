from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable as IterableABC
from collections.abc import Mapping
from typing import Dict, List, Optional, Sequence, Tuple, Mapping, Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MissingnessSpec:
    """Missingness audit settings reconstructed from a parent truth record.

    missingness_pct_all: dict[sensor] -> percent missing (0..100)
    missingness_pct_by_state: dict[state] -> dict[sensor] -> percent missing (0..100)
    missingness_state_dependent_flag: dict[sensor] -> bool
    state_col_synth: dataframe column used to identify row state when "phase" is absent
    state_list: ordered list of valid state names (e.g. ["normal", "abnormal", "recovery"])
    """
    missingness_pct_all: Dict[str, float]
    missingness_pct_by_state: Dict[str, Dict[str, float]]
    missingness_state_dependent_flag: Dict[str, bool]
    state_list: List[str]
    state_col_synth: str

def _payload_mapping(
    payload: Mapping[str, object],
    key: str,
) -> Mapping[Any, Any]:
    """Return a mapping payload value or raise a typed shape error."""
    value = payload.get(key)

    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise TypeError(
            f"Expected payload[{key!r}] to be a mapping, "
            f"got {type(value).__name__}: {value!r}"
        )

    return value


def _payload_str_float_dict(
    payload: Mapping[str, object],
    key: str,
) -> Dict[str, float]:
    """Normalize a flat payload mapping to dict[str, float]."""
    raw_mapping = _payload_mapping(payload, key)

    return {
        str(raw_key): float(raw_value)
        for raw_key, raw_value in raw_mapping.items()
    }


def _payload_str_bool_dict(
    payload: Mapping[str, object],
    key: str,
) -> Dict[str, bool]:
    """Normalize a flat payload mapping to dict[str, bool]."""
    raw_mapping = _payload_mapping(payload, key)

    return {
        str(raw_key): bool(raw_value)
        for raw_key, raw_value in raw_mapping.items()
    }


def _payload_nested_str_float_dict(
    payload: Mapping[str, object],
    key: str,
) -> Dict[str, Dict[str, float]]:
    """Normalize a nested payload mapping to dict[str, dict[str, float]]."""
    raw_outer_mapping = _payload_mapping(payload, key)

    normalized: Dict[str, Dict[str, float]] = {}

    for raw_state, raw_inner_mapping in raw_outer_mapping.items():
        state_key = str(raw_state)

        if raw_inner_mapping is None:
            normalized[state_key] = {}
            continue

        if not isinstance(raw_inner_mapping, Mapping):
            raise TypeError(
                f"Expected payload[{key!r}][{state_key!r}] to be a mapping, "
                f"got {type(raw_inner_mapping).__name__}: {raw_inner_mapping!r}"
            )

        normalized[state_key] = {
            str(raw_sensor): float(raw_pct)
            for raw_sensor, raw_pct in raw_inner_mapping.items()
        }

    return normalized


def _payload_string_list(
    payload: Mapping[str, object],
    key: str,
) -> list[str]:
    """Normalize a payload scalar or iterable to a cleaned string list."""
    value = payload.get(key)

    if value is None:
        return []

    if isinstance(value, str):
        raw_values: IterableABC[object] = [value]
    elif isinstance(value, bytes):
        raw_values = [value.decode("utf-8")]
    elif isinstance(value, IterableABC):
        raw_values = value
    else:
        raise TypeError(
            f"Expected payload[{key!r}] to be iterable, "
            f"got {type(value).__name__}: {value!r}"
        )

    return [
        str(raw_value).strip()
        for raw_value in raw_values
        if str(raw_value).strip()
    ]

def build_missingness_spec_from_truth_payload(
    payload: Dict[str, object],
) -> MissingnessSpec:
    """Build a MissingnessSpec from truth.runtime_facts.missingness_quarantine.

    Permanent behavior:
    - use missingness_pct_all / missingness_pct_by_state / dependency flags when present
    - merge dropped-sensor global missingness from `dropped_missing_pct`
    - optionally merge dropped-sensor state missingness from `dropped_missing_pct_by_state`
      if that key is added to the payload later
    - ensure dropped sensors get a default dependency flag
    - normalize the historical typo in state_col_synth
    """
    pct_all: Dict[str, float] = _payload_str_float_dict(
        payload,
        "missingness_pct_all",
    )

    pct_by_state: Dict[str, Dict[str, float]] = _payload_nested_str_float_dict(
        payload,
        "missingness_pct_by_state",
    )

    dep: Dict[str, bool] = _payload_str_bool_dict(
        payload,
        "missingness_state_dependent_flag",
    )

    gate: Mapping[Any, Any] = _payload_mapping(
        payload,
        "missingness_state_gate_params",
    )

    dropped_features: list[str] = _payload_string_list(
        payload,
        "dropped_features",
    )

    dropped_missing_pct: Dict[str, float] = _payload_str_float_dict(
        payload,
        "dropped_missing_pct",
    )

    dropped_missing_pct_by_state: Dict[str, Dict[str, float]] = (
        _payload_nested_str_float_dict(
            payload,
            "dropped_missing_pct_by_state",
        )
    )

    raw_state_list = gate.get("state_list")

    if raw_state_list is None:
        state_list: list[str] = ["normal", "abnormal", "recovery"]
    elif isinstance(raw_state_list, str):
        state_list = [raw_state_list]
    elif isinstance(raw_state_list, IterableABC):
        state_list = [
            str(raw_state).strip()
            for raw_state in raw_state_list
            if str(raw_state).strip()
        ]
    else:
        raise TypeError(
            "missingness_state_gate_params['state_list'] must be iterable "
            f"or string, got {type(raw_state_list).__name__}: {raw_state_list!r}"
        )

    state_col_synth = str(
        gate.get("state_col_synth") or "machine_status__synthetic"
    )

    # normalize the historical typo
    if state_col_synth == "machine_status__synethic":
        state_col_synth = "machine_status__synthetic"

    pct_by_state_norm: dict[str, dict[str, float]] = {}

    for state_name in state_list:
        state_key = str(state_name)
        pct_by_state_norm[state_key] = dict(
            pct_by_state.get(state_key, {})
        )

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
    for state_name in state_list:
        state_key = str(state_name)
        dropped_state_map = dropped_missing_pct_by_state.get(state_key, {})

        if not dropped_state_map:
            continue

        state_missingness = pct_by_state_norm.setdefault(state_key, {})

        for raw_sensor_name, pct_value in dropped_state_map.items():
            sensor_name = str(raw_sensor_name).strip()

            if sensor_name == "":
                continue

            state_missingness[sensor_name] = float(pct_value)
            dep[sensor_name] = True

    return MissingnessSpec(
        missingness_pct_all={
            str(sensor_name): float(pct_value)
            for sensor_name, pct_value in pct_all.items()
        },
        missingness_pct_by_state={
            str(state_name): {
                str(sensor_name): float(pct_value)
                for sensor_name, pct_value in (
                    pct_by_state_norm.get(str(state_name)) or {}
                ).items()
            }
            for state_name in state_list
        },
        missingness_state_dependent_flag={
            str(sensor_name): bool(is_state_dependent)
            for sensor_name, is_state_dependent in dep.items()
        },
        state_list=[str(state_name) for state_name in state_list],
        state_col_synth=state_col_synth,
    )


def _pct_to_present_count(n: int, missing_pct: float) -> int:
    """Convert a missing percentage into a clamped present-row count."""
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
    """Mask eligible rows so each sensor keeps exactly its target count.

    Rows outside eligible_row_idx are not touched. This gives the generator a
    strict missingness option when clustered gaps are not needed.
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

        #drop_idx = idx[~keep_mask.values]
        #keep_mask_array = np.asarray(keep_mask, dtype=bool)
        keep_mask_array = keep_mask.to_numpy(dtype=bool)
        drop_idx = idx[~keep_mask_array]
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
    """Apply clustered NaN runs while matching each sensor's present count.

    The mask creates short and occasional longer gaps inside eligible rows so
    synthetic missingness resembles sensor dropout instead of uniform removal.
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
    """Build per-sensor present counts for one phase or state block."""
    present_counts: Dict[str, int] = {}
    for s in sensors:
        pct = None
        if use_by_state and pct_by_state is not None and s in pct_by_state:
            pct = pct_by_state.get(s)
        if pct is None:
            pct = pct_all.get(s, 0.0)

        present_counts[str(s)] = _pct_to_present_count(n_rows, float(pct) if pct is not None else 0.0)
    return present_counts
