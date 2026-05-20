"""
utils/silver_preeda.py

Silver Pre-EDA helpers for the capstone pipeline.

Purpose
-------
Take the Bronze dataframe and convert it into a standardized Silver dataframe
that is ready for EDA and downstream Gold preprocessing.

Main responsibilities
---------------------
- Clean import artifacts and duplicate columns
- Validate / resolve dataset naming
- Resolve label or status source columns
- Protect canonical output names from collisions
- Build canonical event identity and ordering columns
- Build anomaly flag and episode IDs
- Identify candidate feature columns and OHE columns
- Quarantine high-missingness features
- Reorder final Silver columns
- Compute lightweight quality checks
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple, Union

import numpy as np
import pandas as pd


RegexLike = Union[str, Pattern[str]]


def remove_junk_import_columns(
    dataframe: pd.DataFrame,
    *,
    junk_column_candidates: Optional[Sequence[str]] = None,
    unnamed_column_regex: Optional[RegexLike] = None,
) -> Tuple[pd.DataFrame, List[str], Optional[str]]:
    """
    Remove common import-artifact columns like Unnamed: 0, level_0, blank columns.
    """
    working_dataframe = dataframe.copy()

    junk_column_candidates = list(junk_column_candidates or [])
    unnamed_pattern = _ensure_regex_pattern(unnamed_column_regex) if unnamed_column_regex else None

    columns_to_drop: List[str] = []

    for column_name in working_dataframe.columns:
        column_text = str(column_name)

        if column_text in junk_column_candidates:
            columns_to_drop.append(column_name)
            continue

        if unnamed_pattern is not None and unnamed_pattern.match(column_text):
            columns_to_drop.append(column_name)
            continue

    columns_to_drop = list(dict.fromkeys(columns_to_drop))

    if columns_to_drop:
        working_dataframe = working_dataframe.drop(columns=columns_to_drop)

    pattern_used = unnamed_pattern.pattern if unnamed_pattern is not None else None
    return working_dataframe, columns_to_drop, pattern_used


def _ensure_regex_pattern(pattern: RegexLike) -> Pattern[str]:
    """
    Ensure a string/regex input is returned as a compiled regex.
    """
    if isinstance(pattern, re.Pattern):
        return pattern
    return re.compile(str(pattern), flags=re.IGNORECASE)


def deduplicate_columns(
    dataframe: pd.DataFrame,
    *,
    keep: str = "first",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove duplicate column names while preserving the chosen keep strategy.
    """
    if dataframe.columns.is_unique:
        return dataframe.copy(), []

    column_index = pd.Index(dataframe.columns)

    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'.")

    duplicate_mask = column_index.duplicated(keep=keep)
    duplicate_columns = column_index[duplicate_mask].tolist()

    deduplicated_dataframe = dataframe.loc[:, ~duplicate_mask].copy()
    return deduplicated_dataframe, duplicate_columns


def _clean_values(series: pd.Series) -> pd.Series:
    """
    Normalize a series into stripped string values.
    """
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
    )


def _normalize_dataset_name(dataset_name: str) -> str:
    """
    Normalize dataset names into pipeline-safe identifiers.
    """
    normalized_value = str(dataset_name).strip().lower()
    normalized_value = normalized_value.replace(" ", "_")
    normalized_value = normalized_value.replace("-", "_")

    cleaned_characters: List[str] = []
    for character in normalized_value:
        if character.isalnum() or character == "_":
            cleaned_characters.append(character)

    normalized_value = "".join(cleaned_characters)

    while "__" in normalized_value:
        normalized_value = normalized_value.replace("__", "_")

    normalized_value = normalized_value.strip("_")

    if normalized_value == "":
        raise ValueError("Dataset name normalization produced an empty value.")

    return normalized_value


def validate_dataset_name_for_silver(
    dataframe: pd.DataFrame,
    *,
    dataset_column: str = "meta__dataset",
    dataset_name_config: Optional[str] = None,
    dataset_name_parent_truth: Optional[str] = None,
) -> Tuple[str, Optional[str], str]:
    """
    Validate or resolve dataset name for Silver using Bronze column and upstream truth.
    """
    dataset_name_from_column: Optional[str] = None
    dataset_source_column: Optional[str] = None
    dataset_method: Optional[str] = None

    if dataset_column in dataframe.columns:
        dataset_values = _clean_values(dataframe[dataset_column])
        dataset_values = dataset_values[dataset_values != ""]
        unique_dataset_values = sorted(dataset_values.dropna().unique().tolist())

        if len(unique_dataset_values) == 1:
            dataset_name_from_column = _normalize_dataset_name(unique_dataset_values[0])
            dataset_source_column = dataset_column
            dataset_method = "dataset_column"
        elif len(unique_dataset_values) > 1:
            raise ValueError(
                f"Multiple dataset values found in {dataset_column}: {unique_dataset_values}"
            )

    if dataset_name_from_column is not None and dataset_name_parent_truth is not None:
        if dataset_name_from_column != _normalize_dataset_name(dataset_name_parent_truth):
            raise ValueError(
                "Dataset name mismatch between Bronze dataframe and parent truth:\n"
                f"column={dataset_name_from_column}\n"
                f"parent_truth={_normalize_dataset_name(dataset_name_parent_truth)}"
            )

    if dataset_name_parent_truth is not None and str(dataset_name_parent_truth).strip() != "":
        return (
            _normalize_dataset_name(dataset_name_parent_truth),
            dataset_source_column,
            "parent_truth",
        )

    if dataset_name_from_column is not None:
        return dataset_name_from_column, dataset_source_column, dataset_method or "dataset_column"

    if dataset_name_config is not None and str(dataset_name_config).strip() != "":
        return _normalize_dataset_name(dataset_name_config), None, "config"

    raise ValueError("Could not resolve dataset name for Silver stage.")


def resolve_label_or_status_source(
    dataframe: pd.DataFrame,
    *,
    status_column_candidates: Sequence[str],
    label_column_candidates: Sequence[str],
    label_exclude_columns: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Resolve the best source column for anomaly/status labeling.

    Returns
    -------
    source_column, source_type, source_info

    source_type is one of:
    - "status"
    - "label"
    - None
    """
    label_exclude_columns = set(label_exclude_columns or [])

    def _top_values(column: str) -> Dict[str, int]:
        value_counts = (
            dataframe[column]
            .astype("string")
            .fillna("<NA>")
            .value_counts(dropna=False)
            .head(10)
        )
        return {str(index): int(value) for index, value in value_counts.items()}

    def _column_info(column: str) -> Dict[str, Any]:
        return {
            "column_name": column,
            "dtype": str(dataframe[column].dtype),
            "non_null_count": int(dataframe[column].notna().sum()),
            "null_count": int(dataframe[column].isna().sum()),
            "unique_count": int(dataframe[column].nunique(dropna=True)),
            "top_values": _top_values(column),
        }

    for column_name in status_column_candidates:
        if column_name in dataframe.columns and column_name not in label_exclude_columns:
            return column_name, "status", _column_info(column_name)

    for column_name in label_column_candidates:
        if column_name in dataframe.columns and column_name not in label_exclude_columns:
            return column_name, "label", _column_info(column_name)

    return None, None, {
        "column_name": None,
        "dtype": None,
        "non_null_count": 0,
        "null_count": 0,
        "unique_count": 0,
        "top_values": {},
    }


def protect_canonical_output_names(
    dataframe: pd.DataFrame,
    *,
    canonical_output_columns: Sequence[str],
    raw_prefix: str = "raw__",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    If raw input already contains canonical output names, rename them out of the way.

    Example:
    event_time -> raw__event_time
    """
    working_dataframe = dataframe.copy()
    rename_map: Dict[str, str] = {}

    for column_name in canonical_output_columns:
        if column_name in working_dataframe.columns:
            protected_name = f"{raw_prefix}{column_name}"
            suffix_counter = 1
            while protected_name in working_dataframe.columns:
                protected_name = f"{raw_prefix}{column_name}_{suffix_counter}"
                suffix_counter += 1

            rename_map[column_name] = protected_name

    if rename_map:
        working_dataframe = working_dataframe.rename(columns=rename_map)

    return working_dataframe, rename_map


def _pick_first_existing_candidate_column(
    dataframe: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    """
    Return the first candidate column that exists in the dataframe.
    """
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def _ensure_grouping_columns_exists(
    dataframe: pd.DataFrame,
    *,
    asset_id_default_fallback: str,
    run_id_default_fallback: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Ensure meta__asset_id and meta__run_id exist.
    """
    working_dataframe = dataframe.copy()
    filled_info: Dict[str, str] = {}

    if "meta__asset_id" not in working_dataframe.columns:
        working_dataframe["meta__asset_id"] = asset_id_default_fallback
        filled_info["meta__asset_id"] = "created_from_fallback"
    else:
        empty_mask = working_dataframe["meta__asset_id"].isna() | (
            working_dataframe["meta__asset_id"].astype("string").str.strip() == ""
        )
        if empty_mask.any():
            working_dataframe.loc[empty_mask, "meta__asset_id"] = asset_id_default_fallback
            filled_info["meta__asset_id"] = "filled_from_fallback"

    if "meta__run_id" not in working_dataframe.columns:
        working_dataframe["meta__run_id"] = run_id_default_fallback
        filled_info["meta__run_id"] = "created_from_fallback"
    else:
        empty_mask = working_dataframe["meta__run_id"].isna() | (
            working_dataframe["meta__run_id"].astype("string").str.strip() == ""
        )
        if empty_mask.any():
            working_dataframe.loc[empty_mask, "meta__run_id"] = run_id_default_fallback
            filled_info["meta__run_id"] = "filled_from_fallback"

    return working_dataframe, filled_info


def evaluate_time_candidates(
    dataframe: pd.DataFrame,
    *,
    time_column_candidates: Sequence[str],
) -> Dict[str, Any]:
    """
    Evaluate candidate time columns and choose the best parseable option.
    """
    best_column: Optional[str] = None
    best_success_percent: float = 0.0
    evaluation_rows: List[Dict[str, Any]] = []

    for column_name in time_column_candidates:
        if column_name not in dataframe.columns:
            continue

        parsed_series = pd.to_datetime(dataframe[column_name], errors="coerce", utc=True)
        success_count = int(parsed_series.notna().sum())
        total_count = int(len(parsed_series))
        success_percent = (success_count / total_count * 100.0) if total_count > 0 else 0.0

        row = {
            "column_name": column_name,
            "success_count": success_count,
            "total_count": total_count,
            "success_percent": float(success_percent),
        }
        evaluation_rows.append(row)

        if success_percent > best_success_percent:
            best_success_percent = float(success_percent)
            best_column = column_name

    return {
        "selected_column": best_column,
        "selected_success_percent": best_success_percent,
        "candidates": evaluation_rows,
    }


def evaluate_step_candidates(
    dataframe: pd.DataFrame,
    *,
    step_column_candidates: Sequence[str],
) -> Dict[str, Any]:
    """
    Evaluate candidate step/order columns and choose the best numeric option.
    """
    best_column: Optional[str] = None
    best_success_percent: float = 0.0
    evaluation_rows: List[Dict[str, Any]] = []

    for column_name in step_column_candidates:
        if column_name not in dataframe.columns:
            continue

        parsed_series = pd.to_numeric(dataframe[column_name], errors="coerce")
        success_count = int(parsed_series.notna().sum())
        total_count = int(len(parsed_series))
        success_percent = (success_count / total_count * 100.0) if total_count > 0 else 0.0

        row = {
            "column_name": column_name,
            "success_count": success_count,
            "total_count": total_count,
            "success_percent": float(success_percent),
        }
        evaluation_rows.append(row)

        if success_percent > best_success_percent:
            best_success_percent = float(success_percent)
            best_column = column_name

    return {
        "selected_column": best_column,
        "selected_success_percent": best_success_percent,
        "candidates": evaluation_rows,
    }


def build_canonical_identity_and_order_master(
    dataframe: pd.DataFrame,
    *,
    time_column_candidates: Sequence[str],
    step_column_candidates: Sequence[str],
    tie_breaker_candidates: Sequence[str],
    asset_id_default_fallback: str,
    run_id_default_fallback: str,
    min_time_parse_success_percent: float = 50.0,
    min_step_parse_success_percent: float = 50.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build canonical Silver ordering and identity fields.

    Creates / standardizes:
    - meta__asset_id
    - meta__run_id
    - event_time
    - event_step
    - time_index
    - event_date
    - meta__source_row_id
    - event_id
    """
    working_dataframe = dataframe.copy()

    working_dataframe, grouping_fill_info = _ensure_grouping_columns_exists(
        working_dataframe,
        asset_id_default_fallback=asset_id_default_fallback,
        run_id_default_fallback=run_id_default_fallback,
    )

    if "meta__source_row_id" not in working_dataframe.columns:
        working_dataframe["meta__source_row_id"] = np.arange(len(working_dataframe), dtype=np.int64)

    time_info = evaluate_time_candidates(
        working_dataframe,
        time_column_candidates=time_column_candidates,
    )
    step_info = evaluate_step_candidates(
        working_dataframe,
        step_column_candidates=step_column_candidates,
    )

    time_source_column = None
    if (
        time_info["selected_column"] is not None
        and time_info["selected_success_percent"] >= min_time_parse_success_percent
    ):
        time_source_column = time_info["selected_column"]
        working_dataframe["event_time"] = pd.to_datetime(
            working_dataframe[time_source_column],
            errors="coerce",
            utc=True,
        )
    else:
        working_dataframe["event_time"] = pd.NaT

    step_source_column = None
    if (
        step_info["selected_column"] is not None
        and step_info["selected_success_percent"] >= min_step_parse_success_percent
    ):
        step_source_column = step_info["selected_column"]
        working_dataframe["event_step"] = pd.to_numeric(
            working_dataframe[step_source_column],
            errors="coerce",
        )
    else:
        working_dataframe["event_step"] = np.nan

    working_dataframe["event_date"] = pd.to_datetime(
        working_dataframe["event_time"],
        errors="coerce",
        utc=True,
    ).dt.floor("D")

    sort_columns: List[str] = ["meta__asset_id", "meta__run_id"]

    if working_dataframe["event_time"].notna().any():
        sort_columns.append("event_time")
    if working_dataframe["event_step"].notna().any():
        sort_columns.append("event_step")

    tie_breaker_column = _pick_first_existing_candidate_column(
        working_dataframe,
        list(tie_breaker_candidates),
    )
    if tie_breaker_column is not None and tie_breaker_column not in sort_columns:
        sort_columns.append(tie_breaker_column)

    if "meta__source_row_id" not in sort_columns:
        sort_columns.append("meta__source_row_id")

    working_dataframe = working_dataframe.sort_values(sort_columns).reset_index(drop=True)

    working_dataframe["time_index"] = (
        working_dataframe
        .groupby(["meta__asset_id", "meta__run_id"], dropna=False)
        .cumcount()
        .astype(np.int64)
    )

    event_id_inputs = (
        working_dataframe["meta__dataset"].astype("string").fillna("unknown_dataset")
        + "|"
        + working_dataframe["meta__asset_id"].astype("string").fillna("unknown_asset")
        + "|"
        + working_dataframe["meta__run_id"].astype("string").fillna("unknown_run")
        + "|"
        + working_dataframe["time_index"].astype("string").fillna("0")
        + "|"
        + working_dataframe["meta__source_row_id"].astype("string").fillna("0")
    )

    working_dataframe["event_id"] = event_id_inputs.map(
        lambda value: hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]
    )

    canonical_info = {
        "grouping_fill_info": grouping_fill_info,
        "time_source_column": time_source_column,
        "step_source_column": step_source_column,
        "time_candidates": time_info,
        "step_candidates": step_info,
        "tie_breaker_column": tie_breaker_column,
        "sort_columns": sort_columns,
        "created_columns": [
            "event_time",
            "event_step",
            "time_index",
            "event_date",
            "event_id",
        ],
    }

    return working_dataframe, canonical_info


def normalize_label_to_binary(series: pd.Series) -> pd.Series:
    """
    Convert mixed label representations into binary anomaly labels.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int64")

    if pd.api.types.is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors="coerce")
        return (numeric_series.fillna(0) > 0).astype("int64")

    string_series = (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    positive_values = {
        "1",
        "true",
        "yes",
        "y",
        "anomaly",
        "abnormal",
        "broken",
        "fault",
        "failure",
    }

    return string_series.isin(positive_values).astype("int64")


def build_anomaly_flag_from_status(
    dataframe: pd.DataFrame,
    *,
    label_source_column: Optional[str],
    label_source_type: Optional[str],
    normal_status_values: Sequence[str],
    anomaly_flag_column: str = "anomaly_flag",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build anomaly_flag using either a label column or a status column.
    """
    working_dataframe = dataframe.copy()
    normal_status_set = {str(value).strip().lower() for value in normal_status_values}

    if label_source_column is None or label_source_type is None:
        working_dataframe[anomaly_flag_column] = np.nan
        return working_dataframe, {
            "anomaly_flag_column": anomaly_flag_column,
            "label_source_column": None,
            "label_source_type": None,
            "method": "not_built",
        }

    if label_source_type == "label":
        working_dataframe[anomaly_flag_column] = normalize_label_to_binary(
            working_dataframe[label_source_column]
        ).astype("int64")

        return working_dataframe, {
            "anomaly_flag_column": anomaly_flag_column,
            "label_source_column": label_source_column,
            "label_source_type": label_source_type,
            "method": "normalize_label_to_binary",
        }

    if label_source_type == "status":
        status_series = (
            working_dataframe[label_source_column]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.lower()
        )

        working_dataframe[anomaly_flag_column] = (~status_series.isin(normal_status_set)).astype("int64")

        return working_dataframe, {
            "anomaly_flag_column": anomaly_flag_column,
            "label_source_column": label_source_column,
            "label_source_type": label_source_type,
            "method": "status_not_in_normal_status_values",
        }

    raise ValueError(f"Unsupported label_source_type: {label_source_type}")


def build_episode_ids_from_anomaly_flag(
    dataframe: pd.DataFrame,
    *,
    anomaly_flag_column: str = "anomaly_flag",
    episode_column: str = "meta__episode_id",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build episode IDs from transitions in anomaly_flag within asset/run groups.
    """
    working_dataframe = dataframe.copy()

    if anomaly_flag_column not in working_dataframe.columns:
        raise ValueError(f"{anomaly_flag_column} not found in dataframe.")

    anomaly_series = pd.to_numeric(working_dataframe[anomaly_flag_column], errors="coerce").fillna(0).astype(int)

    group_keys = ["meta__asset_id", "meta__run_id"]
    previous_series = anomaly_series.groupby(
        [working_dataframe[key] for key in group_keys],
        dropna=False,
    ).shift(1)

    is_new_episode = (anomaly_series != previous_series.fillna(anomaly_series.iloc[0])).astype(int)
    is_new_episode.iloc[0] = 1

    working_dataframe[episode_column] = (
        is_new_episode.groupby(
            [working_dataframe[key] for key in group_keys],
            dropna=False,
        ).cumsum().astype(np.int64)
    )

    return working_dataframe, {
        "episode_column": episode_column,
        "group_keys": group_keys,
        "method": "transition_based_episode_counter",
        "episode_count": int(working_dataframe[episode_column].nunique(dropna=True)),
    }


def classify_column_type(series: pd.Series) -> str:
    """
    Coarse column typing for feature registry.
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def should_exclude_by_prefix(
    column_name: str,
    exclude_prefixes: List[str],
) -> bool:
    """
    Return True if a column should be excluded based on prefix rules.
    """
    for prefix in exclude_prefixes:
        if column_name.startswith(prefix):
            return True
    return False


def looks_like_identifier_column(
    column_name: str,
    series: pd.Series,
) -> bool:
    """
    Heuristic check for identifier-like columns that should not be treated as features.
    """
    identifier_name_tokens = {
        "id",
        "event_id",
        "record_id",
        "row_id",
        "asset_id",
        "run_id",
        "episode_id",
    }

    lowered_name = str(column_name).strip().lower()
    if lowered_name in identifier_name_tokens:
        return True
    if lowered_name.endswith("_id"):
        return True

    non_null_series = series.dropna()
    if len(non_null_series) == 0:
        return False

    unique_ratio = float(non_null_series.nunique(dropna=True)) / float(len(non_null_series))
    return unique_ratio >= 0.95 and not pd.api.types.is_numeric_dtype(series)


def identify_feature_set(
    dataframe: pd.DataFrame,
    *,
    exclude_prefixes: Sequence[str],
    exclude_columns: Sequence[str],
    label_columns_order: Sequence[str],
    canonical_exclude_columns: Sequence[str],
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """
    Identify usable Silver feature columns and light feature groups.
    """
    exclude_columns_set = set(exclude_columns) | set(label_columns_order) | set(canonical_exclude_columns)

    feature_columns: List[str] = []
    feature_groups: Dict[str, List[str]] = {
        "numeric": [],
        "categorical": [],
        "boolean": [],
        "datetime": [],
    }
    feature_info: Dict[str, Dict[str, Any]] = {}

    for column_name in dataframe.columns:
        if column_name in exclude_columns_set:
            continue

        if should_exclude_by_prefix(column_name, list(exclude_prefixes)):
            continue

        if looks_like_identifier_column(column_name, dataframe[column_name]):
            continue

        column_type = classify_column_type(dataframe[column_name])

        if column_type == "datetime":
            continue

        feature_columns.append(column_name)
        feature_groups.setdefault(column_type, []).append(column_name)
        feature_info[column_name] = {
            "column_name": column_name,
            "column_type": column_type,
            "dtype": str(dataframe[column_name].dtype),
            "non_null_count": int(dataframe[column_name].notna().sum()),
            "null_count": int(dataframe[column_name].isna().sum()),
            "unique_count": int(dataframe[column_name].nunique(dropna=True)),
        }

    return feature_columns, feature_groups, feature_info


def identify_one_hot_encoding_columns(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    max_unique_values: int = 25,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Identify categorical/boolean feature columns suitable for one-hot encoding.
    """
    one_hot_columns: List[str] = []
    one_hot_truths: Dict[str, Dict[str, Any]] = {}

    for column_name in feature_columns:
        series = dataframe[column_name]
        column_type = classify_column_type(series)

        if column_type not in {"categorical", "boolean"}:
            continue

        unique_count = int(series.nunique(dropna=True))
        if unique_count == 0:
            continue

        if unique_count <= max_unique_values:
            one_hot_columns.append(column_name)
            one_hot_truths[column_name] = {
                "column_name": column_name,
                "column_type": column_type,
                "unique_count": unique_count,
                "eligible_for_one_hot_encoding": True,
            }

    return one_hot_columns, one_hot_truths


def compute_missingness_percentage(
    dataframe: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Compute missingness percent by column.
    """
    selected_columns = list(columns) if columns is not None else list(dataframe.columns)

    missing_pct: Dict[str, float] = {}
    total_rows = len(dataframe)

    for column_name in selected_columns:
        if total_rows == 0:
            missing_pct[column_name] = 0.0
        else:
            missing_pct[column_name] = float(dataframe[column_name].isna().mean() * 100.0)

    return missing_pct


def quarantine_features_by_missingness(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    quarantine_missing_pct: float,
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, float], Dict[str, Any], pd.DataFrame]:
    """
    Drop feature columns above the configured missingness threshold.
    """
    working_dataframe = dataframe.copy()
    feature_columns_list = list(feature_columns)
    missing_pct = compute_missingness_percentage(working_dataframe, columns=feature_columns_list)

    dropped_features = [
        column_name
        for column_name, pct_value in missing_pct.items()
        if pct_value > quarantine_missing_pct
    ]

    kept_features = [column_name for column_name in feature_columns_list if column_name not in dropped_features]

    dropped_dataframe = working_dataframe[dropped_features].copy() if dropped_features else pd.DataFrame(index=working_dataframe.index)

    if dropped_features:
        working_dataframe = working_dataframe.drop(columns=dropped_features)

    missing_audit = {
        "quarantine_missing_pct": float(quarantine_missing_pct),
        "feature_count_before": int(len(feature_columns_list)),
        "feature_count_after": int(len(kept_features)),
        "dropped_feature_count": int(len(dropped_features)),
        "dropped_features": list(dropped_features),
        "kept_features": list(kept_features),
    }

    return (
        working_dataframe,
        kept_features,
        dropped_features,
        missing_pct,
        missing_audit,
        dropped_dataframe,
    )


def compute_global_missingness(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> Dict[str, Any]:
    """
    Compute global missingness summary for selected features.
    """
    feature_columns = list(feature_columns)
    if len(feature_columns) == 0:
        return {
            "feature_count": 0,
            "overall_missing_pct": 0.0,
            "top_missing_columns": [],
        }

    missing_pct = compute_missingness_percentage(dataframe, columns=feature_columns)

    overall_missing_pct = float(
        dataframe[feature_columns].isna().sum().sum()
        / max(dataframe[feature_columns].shape[0] * dataframe[feature_columns].shape[1], 1)
        * 100.0
    )

    top_missing_columns = sorted(
        missing_pct.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]

    return {
        "feature_count": int(len(feature_columns)),
        "overall_missing_pct": overall_missing_pct,
        "top_missing_columns": [
            {"column_name": column_name, "missing_pct": float(pct_value)}
            for column_name, pct_value in top_missing_columns
        ],
    }


def build_feature_set_identifier(feature_columns: List[str]) -> str:
    """
    Build a deterministic feature set identifier based on the kept feature list.
    """
    joined_text = "|".join(sorted(feature_columns))
    return hashlib.sha1(joined_text.encode("utf-8")).hexdigest()[:16]


def safe_list_columns(
    columns: List[str],
    existing_columns: List[str],
) -> List[str]:
    """
    Return only columns that currently exist.
    """
    existing_set = set(existing_columns)
    return [column_name for column_name in columns if column_name in existing_set]


def collect_meta_columns(existing_columns: List[str]) -> List[str]:
    """
    Return all meta__ columns in current order.
    """
    return [column_name for column_name in existing_columns if column_name.startswith("meta__")]


def reorder_silver_columns(
    dataframe: pd.DataFrame,
    *,
    canonical_non_meta_order: Sequence[str],
    label_columns_order: Sequence[str],
) -> pd.DataFrame:
    """
    Reorder final Silver columns.

    Order
    -----
    1. meta__ columns
    2. canonical non-meta order
    3. label columns
    4. remaining columns
    """
    working_dataframe = dataframe.copy()
    existing_columns = list(working_dataframe.columns)

    meta_columns = collect_meta_columns(existing_columns)
    canonical_columns = safe_list_columns(list(canonical_non_meta_order), existing_columns)
    label_columns = safe_list_columns(list(label_columns_order), existing_columns)

    already_used = set(meta_columns) | set(canonical_columns) | set(label_columns)

    remaining_columns = [
        column_name
        for column_name in existing_columns
        if column_name not in already_used
    ]

    final_order = meta_columns + canonical_columns + label_columns + remaining_columns
    final_order = list(dict.fromkeys(final_order))

    return working_dataframe[final_order].copy()


def compute_quick_quality_checks(
    dataframe: pd.DataFrame,
    *,
    anomaly_flag_column: str = "anomaly_flag",
    feature_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compute lightweight quality checks for the Silver dataframe.
    """
    quality_info: Dict[str, Any] = {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "columns_are_unique": bool(dataframe.columns.is_unique),
        "meta_column_count": int(len([c for c in dataframe.columns if c.startswith("meta__")])),
        "feature_column_count": int(len(list(feature_columns or []))),
        "null_cells_total": int(dataframe.isna().sum().sum()),
    }

    if anomaly_flag_column in dataframe.columns:
        anomaly_counts = (
            pd.to_numeric(dataframe[anomaly_flag_column], errors="coerce")
            .fillna(-1)
            .value_counts(dropna=False)
            .to_dict()
        )
        quality_info["anomaly_flag_distribution"] = {
            str(key): int(value) for key, value in anomaly_counts.items()
        }

    required_columns = ["meta__dataset", "meta__asset_id", "meta__run_id", "event_id", "time_index"]
    quality_info["missing_required_columns"] = [
        column_name for column_name in required_columns if column_name not in dataframe.columns
    ]

    return quality_info


def prepare_silver_preeda_dataframe(
    dataframe: pd.DataFrame,
    *,
    dataset_name_config: Optional[str],
    dataset_name_parent_truth: Optional[str],
    junk_column_candidates: Sequence[str],
    unnamed_column_regex: RegexLike,
    status_column_candidates: Sequence[str],
    label_column_candidates: Sequence[str],
    canonical_output_columns: Sequence[str],
    time_column_candidates: Sequence[str],
    step_column_candidates: Sequence[str],
    tie_breaker_candidates: Sequence[str],
    normal_status_values: Sequence[str],
    asset_id_default_fallback: str,
    run_id_default_fallback: str,
    raw_prefix: str,
    label_exclude_columns: Sequence[str],
    min_time_parse_success_percent: float,
    min_step_parse_success_percent: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main orchestration helper for Silver Pre-EDA dataframe preparation.
    """
    working_dataframe = dataframe.copy()

    working_dataframe, junk_columns_found, pattern_used = remove_junk_import_columns(
        working_dataframe,
        junk_column_candidates=junk_column_candidates,
        unnamed_column_regex=unnamed_column_regex,
    )

    working_dataframe, duplicate_columns_found = deduplicate_columns(
        working_dataframe,
        keep="first",
    )

    validated_dataset_name, dataset_source_column, dataset_method = validate_dataset_name_for_silver(
        working_dataframe,
        dataset_column="meta__dataset",
        dataset_name_config=dataset_name_config,
        dataset_name_parent_truth=dataset_name_parent_truth,
    )

    if "meta__dataset" not in working_dataframe.columns:
        working_dataframe["meta__dataset"] = validated_dataset_name
    else:
        empty_mask = working_dataframe["meta__dataset"].isna() | (
            working_dataframe["meta__dataset"].astype("string").str.strip() == ""
        )
        if empty_mask.any():
            working_dataframe.loc[empty_mask, "meta__dataset"] = validated_dataset_name

    label_source_column, label_source_type, label_source_info = resolve_label_or_status_source(
        working_dataframe,
        status_column_candidates=status_column_candidates,
        label_column_candidates=label_column_candidates,
        label_exclude_columns=label_exclude_columns,
    )

    working_dataframe, protected_column_map = protect_canonical_output_names(
        working_dataframe,
        canonical_output_columns=canonical_output_columns,
        raw_prefix=raw_prefix,
    )

    working_dataframe, canonical_info = build_canonical_identity_and_order_master(
        working_dataframe,
        time_column_candidates=time_column_candidates,
        step_column_candidates=step_column_candidates,
        tie_breaker_candidates=tie_breaker_candidates,
        asset_id_default_fallback=asset_id_default_fallback,
        run_id_default_fallback=run_id_default_fallback,
        min_time_parse_success_percent=min_time_parse_success_percent,
        min_step_parse_success_percent=min_step_parse_success_percent,
    )

    working_dataframe, anomaly_build_info = build_anomaly_flag_from_status(
        working_dataframe,
        label_source_column=label_source_column,
        label_source_type=label_source_type,
        normal_status_values=normal_status_values,
        anomaly_flag_column="anomaly_flag",
    )

    working_dataframe, episode_info = build_episode_ids_from_anomaly_flag(
        working_dataframe,
        anomaly_flag_column="anomaly_flag",
        episode_column="meta__episode_id",
    )

    preparation_info = {
        "validated_dataset_name": validated_dataset_name,
        "dataset_source_column": dataset_source_column,
        "dataset_method": dataset_method,
        "junk_columns_found": junk_columns_found,
        "unnamed_pattern_used": pattern_used,
        "duplicate_columns_found": duplicate_columns_found,
        "label_source_column": label_source_column,
        "label_source_type": label_source_type,
        "label_source_info": label_source_info,
        "protected_column_map": protected_column_map,
        "canonical_info": canonical_info,
        "anomaly_build_info": anomaly_build_info,
        "episode_info": episode_info,
    }

    return working_dataframe, preparation_info


def build_silver_feature_registry(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    exclude_prefixes: Sequence[str],
    exclude_columns: Sequence[str],
    label_columns_order: Sequence[str],
    canonical_exclude_columns: Sequence[str],
    quarantine_missing_pct: float,
    pipeline_mode: str,
    process_run_id: str,
    label_source_column: Optional[str],
    label_source_type: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Build Silver feature registry and quarantine excessive-missingness features.
    """
    working_dataframe = dataframe.copy()

    feature_columns, feature_groups, feature_info = identify_feature_set(
        working_dataframe,
        exclude_prefixes=exclude_prefixes,
        exclude_columns=exclude_columns,
        label_columns_order=label_columns_order,
        canonical_exclude_columns=canonical_exclude_columns,
    )

    one_hot_columns, one_hot_truths = identify_one_hot_encoding_columns(
        working_dataframe,
        feature_columns=feature_columns,
        max_unique_values=25,
    )

    (
        working_dataframe,
        feature_columns,
        dropped_features,
        missing_pct,
        missing_audit,
        dropped_dataframe,
    ) = quarantine_features_by_missingness(
        working_dataframe,
        feature_columns=feature_columns,
        quarantine_missing_pct=quarantine_missing_pct,
    )

    # Rebuild feature groups/info against the surviving columns only
    surviving_feature_columns = list(feature_columns)
    feature_groups = {
        group_name: [column_name for column_name in columns if column_name in surviving_feature_columns]
        for group_name, columns in feature_groups.items()
    }
    feature_info = {
        column_name: info
        for column_name, info in feature_info.items()
        if column_name in surviving_feature_columns
    }

    feature_set_identifier = build_feature_set_identifier(surviving_feature_columns)

    global_missingness = compute_global_missingness(
        working_dataframe,
        feature_columns=surviving_feature_columns,
    )

    registry = {
        "dataset_name": dataset_name,
        "row_count": int(len(working_dataframe)),
        "column_count": int(len(working_dataframe.columns)),
        "feature_set_id": feature_set_identifier,
        "feature_count": int(len(surviving_feature_columns)),
        "feature_columns": list(surviving_feature_columns),
        "feature_groups": {
            group_name: list(columns)
            for group_name, columns in feature_groups.items()
        },
        "feature_info": feature_info,
        "exclude_prefixes": list(exclude_prefixes),
        "exclude_columns": list(exclude_columns),
        "label_source_column": label_source_column,
        "label_source_type": label_source_type,
        "one_hot_encoding_columns": list(one_hot_columns),
        "one_hot_encoding_truths": one_hot_truths,
        "quarantine_missing_pct": float(quarantine_missing_pct),
        "pipeline_mode": pipeline_mode,
        "process_run_id": process_run_id,
        "dropped_features": list(dropped_features),
        "missingness_summary": global_missingness,
        "missing_pct_by_feature": missing_pct,
    }

    artifact_info = {
        "dropped_dataframe": dropped_dataframe,
        "missing_audit": missing_audit,
        "global_missingness": global_missingness,
    }

    return working_dataframe, registry, artifact_info