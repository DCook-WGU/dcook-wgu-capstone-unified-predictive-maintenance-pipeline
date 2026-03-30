"""
utils/pipeline/silver_eda_profiles.py

Profile/statistics helpers for Silver EDA.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def z_score(series: pd.Series) -> pd.Series:
    """
    Z-score normalize a numeric pandas Series.
    Returns a Series with the same index.
    If std is 0 or NaN, returns centered zeros for non-null rows.
    """
    series = pd.to_numeric(series, errors="coerce").astype(float)

    mean_value = np.nanmean(series.to_numpy())
    std_value = np.nanstd(series.to_numpy())

    if std_value == 0 or np.isnan(std_value):
        return pd.Series(
            np.where(series.notna(), 0.0, np.nan),
            index=series.index,
        )

    return (series - mean_value) / std_value


def build_silver_overview_summary(dataframe: pd.DataFrame) -> dict:
    """
    Build a quick structural overview of the Silver dataframe.
    """
    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "meta_column_count": int(sum(column.startswith("meta__") for column in dataframe.columns)),
        "numeric_column_count": int(len(dataframe.select_dtypes(include=[np.number]).columns)),
        "categorical_column_count": int(
            len([
                column for column in dataframe.columns
                if not pd.api.types.is_numeric_dtype(dataframe[column])
                and not pd.api.types.is_datetime64_any_dtype(dataframe[column])
            ])
        ),
        "columns_are_unique": bool(dataframe.columns.is_unique),
    }


def build_missingness_audit_table(
    dataframe: pd.DataFrame,
    *,
    include_only_nonzero: bool = False,
) -> pd.DataFrame:
    """
    Build missingness summary by column.
    """
    rows = []
    total_rows = len(dataframe)

    for column_name in dataframe.columns:
        null_count = int(dataframe[column_name].isna().sum())
        missing_pct = float((null_count / total_rows) * 100) if total_rows > 0 else 0.0

        if include_only_nonzero and null_count == 0:
            continue

        rows.append(
            {
                "column_name": column_name,
                "null_count": null_count,
                "missing_pct": round(missing_pct, 4),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    return result.sort_values(
        ["missing_pct", "column_name"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_duplicate_summary(dataframe: pd.DataFrame) -> dict:
    """
    Summarize duplicate rows and duplicate key identifiers.
    """
    duplicate_row_count = int(dataframe.duplicated().sum())

    duplicate_meta_record_id_count = None
    if "meta__record_id" in dataframe.columns:
        duplicate_meta_record_id_count = int(
            dataframe["meta__record_id"].duplicated().sum()
        )

    duplicate_event_id_count = None
    if "event_id" in dataframe.columns:
        duplicate_event_id_count = int(
            dataframe["event_id"].duplicated().sum()
        )

    return {
        "duplicate_row_count": duplicate_row_count,
        "duplicate_meta__record_id_count": duplicate_meta_record_id_count,
        "duplicate_event_id_count": duplicate_event_id_count,
    }


def build_numeric_describe_table(
    dataframe: pd.DataFrame,
    *,
    include_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build numeric describe table.
    """
    if include_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_columns = [
            column for column in include_columns
            if column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[column])
        ]

    if len(numeric_columns) == 0:
        return pd.DataFrame()

    describe_df = dataframe[numeric_columns].describe().T.reset_index()
    describe_df = describe_df.rename(columns={"index": "column_name"})
    return describe_df


def build_categorical_cardinality_table(
    dataframe: pd.DataFrame,
    *,
    exclude_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build categorical cardinality table.
    """
    exclude_columns = set(exclude_columns or [])

    rows = []
    for column_name in dataframe.columns:
        if column_name in exclude_columns:
            continue
        if pd.api.types.is_numeric_dtype(dataframe[column_name]):
            continue
        if pd.api.types.is_datetime64_any_dtype(dataframe[column_name]):
            continue

        series = dataframe[column_name].astype("string")
        rows.append(
            {
                "column_name": column_name,
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["unique_count", "column_name"],
        ascending=[False, True],
    ).reset_index(drop=True) if len(rows) > 0 else pd.DataFrame()


def profile_sensor_state_table(
    df: pd.DataFrame,
    *,
    sensors: list[str],
    state_values: list[str],
    state_column: str = "state_col_synth",
) -> pd.DataFrame:
    """
    Build per-sensor descriptive stats by state.
    """
    rows = []

    if state_column not in df.columns:
        raise KeyError(f"Missing state column: {state_column}")

    for state in state_values:
        state_df = df.loc[df[state_column] == state].copy()

        for sensor in sensors:
            if sensor not in state_df.columns:
                continue

            series = pd.to_numeric(state_df[sensor], errors="coerce")
            rows.append(
                {
                    "sensor": sensor,
                    "state": state,
                    "row_count": int(series.notna().sum()),
                    "mean": float(series.mean()) if series.notna().any() else None,
                    "median": float(series.median()) if series.notna().any() else None,
                    "std": float(series.std()) if series.notna().any() else None,
                    "min": float(series.min()) if series.notna().any() else None,
                    "max": float(series.max()) if series.notna().any() else None,
                }
            )

    return pd.DataFrame(rows)


def build_state_sensor_profile_table(
    dataframe: pd.DataFrame,
    *,
    sensors: Sequence[str],
    state_column: str,
    state_values: Sequence[str],
) -> pd.DataFrame:
    """
    Wrapper around profile_sensor_state_table.
    """
    return profile_sensor_state_table(
        dataframe,
        sensors=list(sensors),
        state_values=list(state_values),
        state_column=state_column,
    )


def build_feature_behavior_effect_size_table(
    dataframe: pd.DataFrame,
    *,
    sensors: Sequence[str],
    state_column: str,
    baseline_state: str = "normal",
    comparison_states: Sequence[str] = ("abnormal", "recovery"),
) -> pd.DataFrame:
    """
    Build effect-size-style summary table using standardized mean shift vs baseline state.
    """
    rows = []

    if state_column not in dataframe.columns:
        raise KeyError(f"Missing state column: {state_column}")

    for sensor in sensors:
        if sensor not in dataframe.columns:
            continue

        sensor_series = pd.to_numeric(dataframe[sensor], errors="coerce")

        baseline_values = sensor_series.loc[dataframe[state_column] == baseline_state]
        baseline_mean = float(baseline_values.mean()) if baseline_values.notna().any() else None
        baseline_std = float(baseline_values.std()) if baseline_values.notna().any() else None

        for comparison_state in comparison_states:
            comparison_values = sensor_series.loc[dataframe[state_column] == comparison_state]
            comparison_mean = float(comparison_values.mean()) if comparison_values.notna().any() else None

            standardized_mean_shift = None
            if (
                baseline_mean is not None
                and comparison_mean is not None
                and baseline_std is not None
                and baseline_std != 0
                and not np.isnan(baseline_std)
            ):
                standardized_mean_shift = (comparison_mean - baseline_mean) / baseline_std

            rows.append(
                {
                    "sensor": sensor,
                    "baseline_state": baseline_state,
                    "comparison_state": comparison_state,
                    "baseline_mean": baseline_mean,
                    "comparison_mean": comparison_mean,
                    "baseline_std": baseline_std,
                    "standardized_mean_shift": standardized_mean_shift,
                }
            )

    return pd.DataFrame(rows)


def build_plot_feature_list(
    effect_size_df: pd.DataFrame,
    *,
    max_features: int = 6,
) -> list[str]:
    """
    Build top feature list from effect-size table.
    """
    if effect_size_df.empty:
        return []

    working = effect_size_df.copy()
    working["abs_standardized_mean_shift"] = working["standardized_mean_shift"].abs()
    working = working.sort_values(
        ["abs_standardized_mean_shift", "sensor"],
        ascending=[False, True],
    )

    sensors = working["sensor"].dropna().astype(str).tolist()
    ordered_unique = []
    seen = set()

    for sensor_name in sensors:
        if sensor_name not in seen:
            seen.add(sensor_name)
            ordered_unique.append(sensor_name)
        if len(ordered_unique) >= max_features:
            break

    return ordered_unique