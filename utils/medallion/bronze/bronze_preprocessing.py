"""
utils/bronze_preprocessing.py

Bronze preprocessing helpers for the capstone pipeline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _normalize_dataset_name(dataset_name: str) -> str:
    """
    Normalize a dataset name into a stable pipeline-safe identifier.
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


def _generate_deterministic_dataset_name_from_file_details(
    path_value: Optional[str],
) -> Optional[str]:
    """
    Build a deterministic fallback dataset name from source file details.
    """
    if path_value is None or str(path_value).strip() == "":
        return None

    path_object = Path(path_value)

    file_stem_raw = path_object.stem.strip()
    if file_stem_raw == "":
        file_stem_raw = "dataset"

    file_stem_normalized = _normalize_dataset_name(file_stem_raw)

    file_size_bytes = "na"
    modified_timestamp = "na"
    content_fingerprint = "nohash"

    if path_object.exists() and path_object.is_file():
        stat_result = path_object.stat()
        file_size_bytes = str(int(stat_result.st_size))
        modified_timestamp = str(int(stat_result.st_mtime))

        try:
            sample_hasher = hashlib.sha1()

            with open(path_object, "rb") as file_handle:
                first_chunk = file_handle.read(65536)
                sample_hasher.update(first_chunk)

                if stat_result.st_size > 65536:
                    seek_position = max(stat_result.st_size - 65536, 0)
                    file_handle.seek(seek_position)
                    last_chunk = file_handle.read(65536)
                    sample_hasher.update(last_chunk)

            sample_hasher.update(file_size_bytes.encode("utf-8"))
            sample_hasher.update(modified_timestamp.encode("utf-8"))

            content_fingerprint = sample_hasher.hexdigest()[:8]

        except Exception:
            content_fingerprint = "readfail"

    generated_dataset_name = (
        f"{file_stem_normalized}_{file_size_bytes}_{modified_timestamp}_{content_fingerprint}"
    )

    return _normalize_dataset_name(generated_dataset_name)


def resolve_dataset_name_for_bronze_pre_ingest(
    *,
    argument_value: Optional[str] = None,
    config_value: Optional[str] = None,
    handoff_dataset_name: Optional[str] = None,
    source_table_name: Optional[str] = None,
    source_table_dataset_map: Optional[dict] = None,
    fallback_value: Optional[str] = None,
    source_path: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Resolve dataset name before Bronze ingestion.

    Priority order:
    1. CLI / Argument
    2. Config File
    3. Explicit handoff dataset name
    4. Source table -> dataset mapping
    5. Deterministic file-details-based generated name
    6. Fallback
    """
    source_table_dataset_map = source_table_dataset_map or {}

    if argument_value is not None and str(argument_value).strip() != "":
        return (
            _normalize_dataset_name(str(argument_value)),
            "argument",
            "argument",
        )

    if config_value is not None and str(config_value).strip() != "":
        return (
            _normalize_dataset_name(str(config_value)),
            "config",
            "config",
        )

    if handoff_dataset_name is not None and str(handoff_dataset_name).strip() != "":
        return (
            _normalize_dataset_name(str(handoff_dataset_name)),
            "handoff_dataset_name",
            "handoff",
        )

    if source_table_name is not None and str(source_table_name).strip() != "":
        mapped_dataset_name = source_table_dataset_map.get(str(source_table_name).strip())
        if mapped_dataset_name is not None and str(mapped_dataset_name).strip() != "":
            return (
                _normalize_dataset_name(str(mapped_dataset_name)),
                "source_table_name",
                "source_table_map",
            )

    generated_dataset_name = _generate_deterministic_dataset_name_from_file_details(source_path)

    if generated_dataset_name is not None:
        return (
            generated_dataset_name,
            "source_path",
            "file_details",
        )

    fallback_value_text = (
        fallback_value
        if (fallback_value is not None and str(fallback_value).strip() != "")
        else "unknown_dataset"
    )

    return (
        _normalize_dataset_name(str(fallback_value_text)),
        "fallback",
        "fallback",
    )


def write_dataset_resolution_attrs(
    dataframe: pd.DataFrame,
    *,
    dataset_column: str = "meta__dataset",
    fallback_dataset_name: Optional[str] = None,
    fallback_method: str = "fallback_dataset_name",
) -> pd.DataFrame:
    """
    Write Bronze dataset resolution metadata into dataframe.attrs.
    """
    resolved_dataset_name: Optional[str] = None
    dataset_source_column: Optional[str] = None
    dataset_method: Optional[str] = None

    if dataset_column in dataframe.columns:
        dataset_values = dataframe[dataset_column].dropna().astype(str).str.strip()
        unique_dataset_values = sorted(value for value in dataset_values.unique() if value)

        if len(unique_dataset_values) == 1:
            resolved_dataset_name = unique_dataset_values[0]
            dataset_source_column = dataset_column
            dataset_method = "dataset_column"

        elif len(unique_dataset_values) > 1:
            raise ValueError(
                f"Multiple dataset values found in '{dataset_column}': {unique_dataset_values}"
            )

    if resolved_dataset_name is None:
        if fallback_dataset_name is None or str(fallback_dataset_name).strip() == "":
            raise ValueError(
                "Could not resolve dataset name from dataframe column or fallback_dataset_name."
            )

        resolved_dataset_name = str(fallback_dataset_name).strip()
        dataset_source_column = None
        dataset_method = fallback_method

    dataframe.attrs["dataset_resolution"] = {
        "dataset_name": resolved_dataset_name,
        "dataset_source_column": dataset_source_column,
        "dataset_method": dataset_method,
    }

    return dataframe


def collect_meta_columns(existing_columns: List[str]) -> List[str]:
    """
    Return meta__ columns in current order.
    """
    meta_columns: List[str] = []
    for column in existing_columns:
        if column.startswith("meta__"):
            meta_columns.append(column)
    return meta_columns


def reorder_bronze_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Move meta__ columns to the front while preserving existing order.
    """
    existing_columns = list(dataframe.columns)
    meta_columns = collect_meta_columns(existing_columns)

    bronze_columns: List[str] = []
    for column in existing_columns:
        if column not in meta_columns:
            bronze_columns.append(column)

    final_order: List[str] = []
    final_order.extend(meta_columns)
    final_order.extend(bronze_columns)

    return dataframe[final_order].copy()


def prepare_bronze_dataframe(
    dataframe: pd.DataFrame,
    *,
    argument_dataset_name: Optional[str] = None,
    config_dataset_name: Optional[str] = None,
    handoff_dataset_name: Optional[str] = None,
    source_table_name: Optional[str] = None,
    source_table_dataset_map: Optional[Dict[str, str]] = None,
    fallback_dataset_name: Optional[str] = None,
    source_path: Optional[str] = None,
    dataset_column: str = "meta__dataset",
    reorder_columns: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Prepare a Bronze dataframe for downstream saving and truth stamping.
    """
    working_dataframe = dataframe.copy()

    resolved_dataset_name, dataset_source, dataset_method = (
        resolve_dataset_name_for_bronze_pre_ingest(
            argument_value=argument_dataset_name,
            config_value=config_dataset_name,
            handoff_dataset_name=handoff_dataset_name,
            source_table_name=source_table_name,
            source_table_dataset_map=source_table_dataset_map,
            fallback_value=fallback_dataset_name,
            source_path=source_path,
        )
    )

    if dataset_column not in working_dataframe.columns:
        working_dataframe[dataset_column] = resolved_dataset_name
    else:
        dataset_series = working_dataframe[dataset_column]
        empty_mask = dataset_series.isna() | dataset_series.astype(str).str.strip().eq("")
        if empty_mask.any():
            working_dataframe.loc[empty_mask, dataset_column] = resolved_dataset_name

    working_dataframe = write_dataset_resolution_attrs(
        working_dataframe,
        dataset_column=dataset_column,
        fallback_dataset_name=resolved_dataset_name,
        fallback_method=dataset_method,
    )

    if reorder_columns:
        working_dataframe = reorder_bronze_columns(working_dataframe)
    else:
        working_dataframe = working_dataframe.copy()

    final_columns = list(working_dataframe.columns)
    meta_columns = collect_meta_columns(final_columns)
    non_meta_columns = [column for column in final_columns if column not in meta_columns]

    resolution_payload: Dict[str, Any] = {
        "dataset_name": resolved_dataset_name,
        "dataset_source": dataset_source,
        "dataset_method": dataset_method,
        "dataset_resolution_attrs": working_dataframe.attrs.get("dataset_resolution", {}),
        "meta_columns": meta_columns,
        "non_meta_columns": non_meta_columns,
        "final_column_order": final_columns,
        "row_count": int(len(working_dataframe)),
        "column_count": int(len(final_columns)),
    }

    return working_dataframe, resolution_payload