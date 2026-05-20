from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SensorNormalityAuditConfig:
    """
    Configuration for Silver EDA normal-operation verification.

    Notes
    -----
    - "suspect" means a sensor has some abnormal evidence.
    - "abnormal" means stronger evidence, either by magnitude, multiple components,
      or persistence.
    - Row-level classification uses sensor-level evidence after aggregation.
    """
    normal_values: Tuple[str, ...] = ("normal",)
    transition_buffer_rows: int = 0
    min_seed_row_non_null_ratio: float = 0.95
    min_seed_rows_per_sensor: int = 25

    rolling_window: int = 11

    value_suspect_z: float = 3.5
    value_abnormal_z: float = 4.5

    delta_suspect_z: float = 3.5
    delta_abnormal_z: float = 4.5

    residual_suspect_z: float = 3.5
    residual_abnormal_z: float = 4.5

    persistent_run_length: int = 3
    severe_sensor_score: float = 8.0

    suspect_sensor_count: int = 2
    suspect_sensor_ratio: float = 0.04
    suspect_group_count: int = 2

    exclude_sensor_count: int = 4
    exclude_sensor_ratio: float = 0.08
    exclude_group_count: int = 3
    exclude_persistent_sensor_count: int = 1


# =============================================================================
# Small helpers
# =============================================================================

def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_status_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .fillna("")
    )


def _normalize_group_key_series(series: Optional[pd.Series], index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series("__ALL_ROWS__", index=index, dtype="object")
    return series.astype("string").fillna("__MISSING_GROUP__").astype("object")


def _robust_center_scale(series: pd.Series, eps: float = 1e-9) -> Dict[str, Any]:
    """
    Robustly estimate center and scale.
    Fallback order:
    1) MAD-based scale
    2) IQR-based scale
    3) std
    4) eps
    """
    values = _coerce_numeric(series).dropna().to_numpy()
    n = int(values.size)

    if n == 0:
        return {
            "center": np.nan,
            "scale": np.nan,
            "fit_count": 0,
            "scale_method": "no_data",
        }

    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    mad_scale = 1.4826 * mad

    if mad_scale > eps:
        return {
            "center": center,
            "scale": mad_scale,
            "fit_count": n,
            "scale_method": "mad",
        }

    q25, q75 = np.percentile(values, [25, 75])
    iqr = float(q75 - q25)
    iqr_scale = iqr / 1.349 if iqr > eps else 0.0
    if iqr_scale > eps:
        return {
            "center": center,
            "scale": iqr_scale,
            "fit_count": n,
            "scale_method": "iqr",
        }

    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if std > eps:
        return {
            "center": center,
            "scale": std,
            "fit_count": n,
            "scale_method": "std",
        }

    return {
        "center": center,
        "scale": eps,
        "fit_count": n,
        "scale_method": "eps",
    }


def _robust_abs_z(series: pd.Series, center: float, scale: float) -> pd.Series:
    series = _coerce_numeric(series)
    if pd.isna(center) or pd.isna(scale) or float(scale) <= 0:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    return ((series - float(center)) / float(scale)).abs()


def _groupwise_diff(series: pd.Series, group_keys: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"value": _coerce_numeric(series), "__group": group_keys}, index=series.index)
    return frame.groupby("__group", sort=False)["value"].diff()


def _groupwise_rolling_median(series: pd.Series, group_keys: pd.Series, window: int) -> pd.Series:
    frame = pd.DataFrame({"value": _coerce_numeric(series), "__group": group_keys}, index=series.index)
    return frame.groupby("__group", sort=False)["value"].transform(
        lambda s: s.rolling(window=window, min_periods=1).median()
    )


def _distance_to_false_per_group(flag: pd.Series, group_keys: pd.Series) -> pd.Series:
    """
    For each row, compute distance in rows to the nearest False within the group.

    If a group has no False rows, distance is inf for all rows in that group.
    """
    result = pd.Series(np.inf, index=flag.index, dtype="float64")

    temp = pd.DataFrame(
        {
            "flag": flag.fillna(False).astype(bool),
            "__group": group_keys,
        },
        index=flag.index,
    )

    for _, group in temp.groupby("__group", sort=False):
        idx = group.index
        arr = group["flag"].to_numpy(dtype=bool)
        positions = np.arange(len(arr), dtype=int)
        false_positions = positions[~arr]

        if false_positions.size == 0:
            result.loc[idx] = np.inf
            continue

        right_ix = np.searchsorted(false_positions, positions, side="left")

        right_pos = np.full(len(arr), np.inf)
        valid_right = right_ix < false_positions.size
        right_pos[valid_right] = false_positions[right_ix[valid_right]]

        left_pos = np.full(len(arr), -np.inf)
        valid_left = right_ix > 0
        left_pos[valid_left] = false_positions[right_ix[valid_left] - 1]

        dist_left = positions - left_pos
        dist_right = right_pos - positions
        distance = np.minimum(dist_left, dist_right)

        result.loc[idx] = distance

    return result


def _run_length_of_true(flag: pd.Series, group_keys: pd.Series) -> pd.Series:
    """
    Length of the contiguous True run that each True row belongs to.
    False rows get 0.
    """
    flag = flag.fillna(False).astype(bool)
    group_change = group_keys.ne(group_keys.shift(fill_value=group_keys.iloc[0]))
    flag_change = flag.ne(flag.shift(fill_value=False))
    run_id = (group_change | flag_change).cumsum()

    run_length = flag.groupby(run_id).transform("size")
    run_length = run_length.where(flag, 0).astype(int)
    return run_length


def _normalize_sensor_group_map(
    sensor_columns: Sequence[str],
    sensor_group_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    if sensor_group_map is None:
        return {sensor: sensor for sensor in sensor_columns}
    return {sensor: sensor_group_map.get(sensor, sensor) for sensor in sensor_columns}


# =============================================================================
# Seed mask
# =============================================================================

def build_seed_normal_mask(
    dataframe: pd.DataFrame,
    *,
    sensor_columns: Sequence[str],
    status_column: str = "machine_status",
    normal_values: Sequence[str] = ("normal",),
    episode_column: Optional[str] = None,
    transition_buffer_rows: int = 0,
    min_row_non_null_ratio: float = 0.95,
) -> pd.DataFrame:
    """
    Create a conservative seed-normal mask for first-pass profile fitting.

    Returns a copy of the dataframe with these added columns:
    - __normal_candidate_mask
    - __transition_distance_to_non_normal
    - __seed_normal_mask
    """
    if status_column not in dataframe.columns:
        raise KeyError(f"status_column '{status_column}' not found in dataframe")

    missing_sensors = [c for c in sensor_columns if c not in dataframe.columns]
    if missing_sensors:
        raise KeyError(f"Missing sensor columns: {missing_sensors}")

    df = dataframe.copy()

    normalized_status = _normalize_status_series(df[status_column])
    normal_value_set = {str(v).strip().lower() for v in normal_values}
    normal_candidate_mask = normalized_status.isin(normal_value_set)

    group_keys = _normalize_group_key_series(
        df[episode_column] if episode_column and episode_column in df.columns else None,
        df.index,
    )

    transition_distance = _distance_to_false_per_group(normal_candidate_mask, group_keys)

    if transition_buffer_rows > 0:
        far_enough_from_non_normal = (transition_distance > transition_buffer_rows) | np.isinf(transition_distance)
    else:
        far_enough_from_non_normal = pd.Series(True, index=df.index)

    row_non_null_ratio = df[list(sensor_columns)].notna().mean(axis=1)
    enough_non_null = row_non_null_ratio >= float(min_row_non_null_ratio)

    seed_normal_mask = normal_candidate_mask & far_enough_from_non_normal & enough_non_null

    df["__normal_candidate_mask"] = normal_candidate_mask.astype(bool)
    df["__transition_distance_to_non_normal"] = transition_distance.astype("float64")
    df["__seed_normal_mask"] = seed_normal_mask.astype(bool)
    df["__seed_row_non_null_ratio"] = row_non_null_ratio.astype("float64")

    return df


# =============================================================================
# Sensor profile + scoring
# =============================================================================

def fit_sensor_profile(
    dataframe: pd.DataFrame,
    *,
    sensor_column: str,
    seed_mask_column: str = "__seed_normal_mask",
    episode_column: Optional[str] = None,
    rolling_window: int = 11,
    min_seed_rows_per_sensor: int = 25,
) -> Dict[str, Any]:
    """
    Fit robust profile statistics for a single sensor.

    Returns a dict with:
    - value_center / scale
    - delta_center / scale
    - residual_center / scale
    - fit counts
    - fit scope used
    """
    if sensor_column not in dataframe.columns:
        raise KeyError(f"sensor_column '{sensor_column}' not found in dataframe")
    if seed_mask_column not in dataframe.columns:
        raise KeyError(f"seed_mask_column '{seed_mask_column}' not found in dataframe")

    sensor_values = _coerce_numeric(dataframe[sensor_column])
    seed_mask = dataframe[seed_mask_column].fillna(False).astype(bool)
    normal_candidate_mask = dataframe.get("__normal_candidate_mask", pd.Series(True, index=dataframe.index)).fillna(False).astype(bool)

    sensor_seed_mask = seed_mask & sensor_values.notna()

    if int(sensor_seed_mask.sum()) >= int(min_seed_rows_per_sensor):
        fit_scope_used = "seed_normal"
        fit_mask = sensor_seed_mask
    else:
        fallback_mask = normal_candidate_mask & sensor_values.notna()
        if int(fallback_mask.sum()) >= int(min_seed_rows_per_sensor):
            fit_scope_used = "normal_candidate_fallback"
            fit_mask = fallback_mask
        else:
            fit_scope_used = "non_null_fallback"
            fit_mask = sensor_values.notna()

    group_keys = _normalize_group_key_series(
        dataframe[episode_column] if episode_column and episode_column in dataframe.columns else None,
        dataframe.index,
    )

    delta_series = _groupwise_diff(sensor_values, group_keys)
    rolling_median = _groupwise_rolling_median(sensor_values, group_keys, window=rolling_window)
    residual_series = sensor_values - rolling_median

    value_profile = _robust_center_scale(sensor_values[fit_mask])
    delta_profile = _robust_center_scale(delta_series[fit_mask])
    residual_profile = _robust_center_scale(residual_series[fit_mask])

    return {
        "sensor_column": sensor_column,
        "fit_scope_used": fit_scope_used,
        "rolling_window": int(rolling_window),
        "value_center": value_profile["center"],
        "value_scale": value_profile["scale"],
        "value_fit_count": int(value_profile["fit_count"]),
        "value_scale_method": value_profile["scale_method"],
        "delta_center": delta_profile["center"],
        "delta_scale": delta_profile["scale"],
        "delta_fit_count": int(delta_profile["fit_count"]),
        "delta_scale_method": delta_profile["scale_method"],
        "residual_center": residual_profile["center"],
        "residual_scale": residual_profile["scale"],
        "residual_fit_count": int(residual_profile["fit_count"]),
        "residual_scale_method": residual_profile["scale_method"],
        "value_q01": float(sensor_values[fit_mask].quantile(0.01)) if fit_mask.any() else np.nan,
        "value_q05": float(sensor_values[fit_mask].quantile(0.05)) if fit_mask.any() else np.nan,
        "value_q50": float(sensor_values[fit_mask].quantile(0.50)) if fit_mask.any() else np.nan,
        "value_q95": float(sensor_values[fit_mask].quantile(0.95)) if fit_mask.any() else np.nan,
        "value_q99": float(sensor_values[fit_mask].quantile(0.99)) if fit_mask.any() else np.nan,
    }


def score_sensor_against_profile(
    dataframe: pd.DataFrame,
    *,
    profile: Mapping[str, Any],
    episode_column: Optional[str] = None,
    rolling_window: Optional[int] = None,
    value_suspect_z: float = 3.5,
    value_abnormal_z: float = 4.5,
    delta_suspect_z: float = 3.5,
    delta_abnormal_z: float = 4.5,
    residual_suspect_z: float = 3.5,
    residual_abnormal_z: float = 4.5,
    persistent_run_length: int = 3,
) -> pd.DataFrame:
    """
    Score one sensor across all rows.

    Returns a dataframe with sensor-level columns only.
    """
    sensor = str(profile["sensor_column"])
    if sensor not in dataframe.columns:
        raise KeyError(f"Sensor '{sensor}' not found in dataframe")

    rw = int(rolling_window if rolling_window is not None else profile.get("rolling_window", 11))
    sensor_values = _coerce_numeric(dataframe[sensor])

    group_keys = _normalize_group_key_series(
        dataframe[episode_column] if episode_column and episode_column in dataframe.columns else None,
        dataframe.index,
    )

    delta_series = _groupwise_diff(sensor_values, group_keys)
    rolling_median = _groupwise_rolling_median(sensor_values, group_keys, window=rw)
    residual_series = sensor_values - rolling_median

    level_abs_z = _robust_abs_z(sensor_values, profile["value_center"], profile["value_scale"])
    delta_abs_z = _robust_abs_z(delta_series, profile["delta_center"], profile["delta_scale"])
    residual_abs_z = _robust_abs_z(residual_series, profile["residual_center"], profile["residual_scale"])

    is_level_suspect = level_abs_z >= float(value_suspect_z)
    is_delta_suspect = delta_abs_z >= float(delta_suspect_z)
    is_residual_suspect = residual_abs_z >= float(residual_suspect_z)

    is_level_abnormal = level_abs_z >= float(value_abnormal_z)
    is_delta_abnormal = delta_abs_z >= float(delta_abnormal_z)
    is_residual_abnormal = residual_abs_z >= float(residual_abnormal_z)

    suspect_component_count = pd.concat(
        [
            is_level_suspect.rename("level"),
            is_delta_suspect.rename("delta"),
            is_residual_suspect.rename("residual"),
        ],
        axis=1,
    ).sum(axis=1).astype(int)

    abnormal_component_count = pd.concat(
        [
            is_level_abnormal.rename("level"),
            is_delta_abnormal.rename("delta"),
            is_residual_abnormal.rename("residual"),
        ],
        axis=1,
    ).sum(axis=1).astype(int)

    sensor_is_suspect = suspect_component_count >= 1
    sensor_is_abnormal_core = (abnormal_component_count >= 1) | (suspect_component_count >= 2)

    suspect_run_length = _run_length_of_true(sensor_is_suspect, group_keys)
    sensor_is_persistent_abnormal = sensor_is_suspect & (suspect_run_length >= int(persistent_run_length))
    sensor_is_abnormal = sensor_is_abnormal_core | sensor_is_persistent_abnormal

    sensor_severity = pd.concat(
        [
            level_abs_z.rename("level"),
            delta_abs_z.rename("delta"),
            residual_abs_z.rename("residual"),
        ],
        axis=1,
    ).max(axis=1, skipna=True).fillna(0.0)

    sensor_quality_class = np.select(
        [
            sensor_is_abnormal,
            sensor_is_suspect,
        ],
        [
            "abnormal",
            "suspect",
        ],
        default="clean",
    )

    return pd.DataFrame(
        {
            f"{sensor}__delta": delta_series,
            f"{sensor}__rolling_median": rolling_median,
            f"{sensor}__residual": residual_series,
            f"{sensor}__level_robust_z": level_abs_z,
            f"{sensor}__delta_robust_z": delta_abs_z,
            f"{sensor}__residual_robust_z": residual_abs_z,
            f"{sensor}__is_level_suspect": is_level_suspect.astype(bool),
            f"{sensor}__is_delta_suspect": is_delta_suspect.astype(bool),
            f"{sensor}__is_residual_suspect": is_residual_suspect.astype(bool),
            f"{sensor}__is_level_abnormal": is_level_abnormal.astype(bool),
            f"{sensor}__is_delta_abnormal": is_delta_abnormal.astype(bool),
            f"{sensor}__is_residual_abnormal": is_residual_abnormal.astype(bool),
            f"{sensor}__suspect_component_count": suspect_component_count.astype(int),
            f"{sensor}__abnormal_component_count": abnormal_component_count.astype(int),
            f"{sensor}__suspect_run_length": suspect_run_length.astype(int),
            f"{sensor}__severity": sensor_severity.astype("float64"),
            f"{sensor}__is_suspect": sensor_is_suspect.astype(bool),
            f"{sensor}__is_persistent_abnormal": sensor_is_persistent_abnormal.astype(bool),
            f"{sensor}__is_abnormal": sensor_is_abnormal.astype(bool),
            f"{sensor}__quality_class": pd.Series(sensor_quality_class, index=dataframe.index, dtype="string"),
        },
        index=dataframe.index,
    )


# =============================================================================
# Row aggregation
# =============================================================================

def aggregate_sensor_flags_to_rows(
    dataframe: pd.DataFrame,
    *,
    sensor_columns: Sequence[str],
    sensor_group_map: Optional[Mapping[str, str]] = None,
    suspect_sensor_count: int = 2,
    suspect_sensor_ratio: float = 0.04,
    suspect_group_count: int = 2,
    exclude_sensor_count: int = 4,
    exclude_sensor_ratio: float = 0.08,
    exclude_group_count: int = 3,
    exclude_persistent_sensor_count: int = 1,
    severe_sensor_score: float = 8.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate sensor-level flags into row-level normal-quality evidence.

    Returns
    -------
    row_summary_df, group_flag_df
    """
    group_map = _normalize_sensor_group_map(sensor_columns, sensor_group_map)

    abnormal_cols = [f"{sensor}__is_abnormal" for sensor in sensor_columns]
    suspect_cols = [f"{sensor}__is_suspect" for sensor in sensor_columns]
    persistent_cols = [f"{sensor}__is_persistent_abnormal" for sensor in sensor_columns]
    severity_cols = [f"{sensor}__severity" for sensor in sensor_columns]

    missing_cols = [c for c in abnormal_cols + suspect_cols + persistent_cols + severity_cols if c not in dataframe.columns]
    if missing_cols:
        raise KeyError(f"Missing sensor score columns required for aggregation: {missing_cols[:15]}")

    abnormal_sensor_count = dataframe[abnormal_cols].fillna(False).sum(axis=1).astype(int)
    suspect_sensor_count_series = dataframe[suspect_cols].fillna(False).sum(axis=1).astype(int)
    persistent_sensor_count = dataframe[persistent_cols].fillna(False).sum(axis=1).astype(int)
    max_sensor_severity = dataframe[severity_cols].max(axis=1, skipna=True).fillna(0.0)
    abnormal_sensor_ratio = abnormal_sensor_count / max(len(sensor_columns), 1)
    suspect_sensor_ratio_series = suspect_sensor_count_series / max(len(sensor_columns), 1)

    group_flag_dict: Dict[str, pd.Series] = {}
    for group_name in sorted(set(group_map.values())):
        member_sensors = [sensor for sensor, grp in group_map.items() if grp == group_name]
        member_abnormal_cols = [f"{sensor}__is_abnormal" for sensor in member_sensors]
        group_flag_dict[str(group_name)] = dataframe[member_abnormal_cols].fillna(False).any(axis=1)

    group_flag_df = pd.DataFrame(group_flag_dict, index=dataframe.index)
    abnormal_group_count = group_flag_df.sum(axis=1).astype(int)

    row_is_exclude_from_normal_training = (
        (abnormal_sensor_count >= int(exclude_sensor_count))
        | (abnormal_sensor_ratio >= float(exclude_sensor_ratio))
        | (abnormal_group_count >= int(exclude_group_count))
        | (
            (persistent_sensor_count >= int(exclude_persistent_sensor_count))
            & (max_sensor_severity >= float(severe_sensor_score))
        )
    )

    row_is_suspect_normal = (
        ~row_is_exclude_from_normal_training
        & (
            (abnormal_sensor_count >= int(suspect_sensor_count))
            | (abnormal_sensor_ratio >= float(suspect_sensor_ratio))
            | (abnormal_group_count >= int(suspect_group_count))
            | (persistent_sensor_count >= 1)
            | (max_sensor_severity >= float(severe_sensor_score))
        )
    )

    row_quality_class = np.select(
        [
            row_is_exclude_from_normal_training,
            row_is_suspect_normal,
        ],
        [
            "exclude",
            "suspect",
        ],
        default="clean",
    )

    row_summary_df = pd.DataFrame(
        {
            "row_suspect_sensor_count": suspect_sensor_count_series.astype(int),
            "row_suspect_sensor_ratio": suspect_sensor_ratio_series.astype("float64"),
            "row_abnormal_sensor_count": abnormal_sensor_count.astype(int),
            "row_abnormal_sensor_ratio": abnormal_sensor_ratio.astype("float64"),
            "row_persistent_abnormal_sensor_count": persistent_sensor_count.astype(int),
            "row_abnormal_group_count": abnormal_group_count.astype(int),
            "row_max_sensor_severity": max_sensor_severity.astype("float64"),
            "row_is_suspect_normal": row_is_suspect_normal.astype(bool),
            "row_is_exclude_from_normal_training": row_is_exclude_from_normal_training.astype(bool),
            "row_normal_quality_class": pd.Series(row_quality_class, index=dataframe.index, dtype="string"),
            "row_is_clean_normal": (pd.Series(row_quality_class, index=dataframe.index) == "clean").astype(bool),
        },
        index=dataframe.index,
    )

    return row_summary_df, group_flag_df


# =============================================================================
# Main orchestration function
# =============================================================================

def run_sensor_normality_audit(
    dataframe: pd.DataFrame,
    *,
    sensor_columns: Sequence[str],
    status_column: str = "machine_status",
    normal_values: Sequence[str] = ("normal",),
    episode_column: Optional[str] = None,
    order_column: Optional[str] = None,
    sensor_group_map: Optional[Mapping[str, str]] = None,
    config: Optional[SensorNormalityAuditConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Full Silver EDA normality audit.

    Workflow
    --------
    1) Sort rows by episode/order if provided.
    2) Build seed-normal mask.
    3) Fit per-sensor profiles.
    4) Score each sensor on level, delta, residual, persistence.
    5) Aggregate sensor flags to row-level normal quality.

    Returns
    -------
    dict with:
    - scored_dataframe
    - sensor_profile_dataframe
    - row_group_flag_dataframe
    - audit_metadata_dataframe
    """
    if config is None:
        config = SensorNormalityAuditConfig()

    missing_sensors = [c for c in sensor_columns if c not in dataframe.columns]
    if missing_sensors:
        raise KeyError(f"Missing sensor columns: {missing_sensors}")

    if status_column not in dataframe.columns:
        raise KeyError(f"status_column '{status_column}' not found in dataframe")

    working_df = dataframe.copy()
    working_df["__original_position"] = np.arange(len(working_df), dtype=int)

    sort_columns: List[str] = []
    if episode_column and episode_column in working_df.columns:
        sort_columns.append(episode_column)
    if order_column and order_column in working_df.columns:
        sort_columns.append(order_column)
    sort_columns.append("__original_position")

    working_df = working_df.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    working_df = build_seed_normal_mask(
        working_df,
        sensor_columns=sensor_columns,
        status_column=status_column,
        normal_values=normal_values,
        episode_column=episode_column,
        transition_buffer_rows=config.transition_buffer_rows,
        min_row_non_null_ratio=config.min_seed_row_non_null_ratio,
    )

    sensor_profiles: List[Dict[str, Any]] = []
    sensor_score_frames: List[pd.DataFrame] = []

    for sensor in sensor_columns:
        profile = fit_sensor_profile(
            working_df,
            sensor_column=sensor,
            seed_mask_column="__seed_normal_mask",
            episode_column=episode_column,
            rolling_window=config.rolling_window,
            min_seed_rows_per_sensor=config.min_seed_rows_per_sensor,
        )
        sensor_profiles.append(profile)

        sensor_scored = score_sensor_against_profile(
            working_df,
            profile=profile,
            episode_column=episode_column,
            rolling_window=config.rolling_window,
            value_suspect_z=config.value_suspect_z,
            value_abnormal_z=config.value_abnormal_z,
            delta_suspect_z=config.delta_suspect_z,
            delta_abnormal_z=config.delta_abnormal_z,
            residual_suspect_z=config.residual_suspect_z,
            residual_abnormal_z=config.residual_abnormal_z,
            persistent_run_length=config.persistent_run_length,
        )
        sensor_score_frames.append(sensor_scored)

    sensor_score_df = pd.concat(sensor_score_frames, axis=1)
    working_df = pd.concat([working_df, sensor_score_df], axis=1)

    row_summary_df, group_flag_df = aggregate_sensor_flags_to_rows(
        working_df,
        sensor_columns=sensor_columns,
        sensor_group_map=sensor_group_map,
        suspect_sensor_count=config.suspect_sensor_count,
        suspect_sensor_ratio=config.suspect_sensor_ratio,
        suspect_group_count=config.suspect_group_count,
        exclude_sensor_count=config.exclude_sensor_count,
        exclude_sensor_ratio=config.exclude_sensor_ratio,
        exclude_group_count=config.exclude_group_count,
        exclude_persistent_sensor_count=config.exclude_persistent_sensor_count,
        severe_sensor_score=config.severe_sensor_score,
    )

    working_df = pd.concat([working_df, row_summary_df], axis=1)

    sensor_profile_df = pd.DataFrame(sensor_profiles)
    audit_metadata_df = pd.DataFrame(
        [{"config_key": k, "config_value": str(v)} for k, v in asdict(config).items()]
    )

    working_df = working_df.sort_values("__original_position", kind="stable").drop(columns=["__original_position"])
    group_flag_df = group_flag_df.loc[working_df.index]
    row_summary_df = row_summary_df.loc[working_df.index]

    return {
        "scored_dataframe": working_df,
        "sensor_profile_dataframe": sensor_profile_df,
        "row_group_flag_dataframe": group_flag_df,
        "row_summary_dataframe": row_summary_df,
        "audit_metadata_dataframe": audit_metadata_df,
    }