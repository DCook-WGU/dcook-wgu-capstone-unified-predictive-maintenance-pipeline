from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union, List, Optional, Sequence, Tuple, cast

import math
import numpy as np
import pandas as pd



# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------

def cfg_require_mapping(
        value: object, 
        name: str
    ) -> Mapping[str, Any]:
    
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{name} must be a mapping, got {type(value).__name__}: {value!r}"
        )

    return cast(Mapping[str, Any], value)


# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------


def cfg_optional_mapping(
        value: object | None, 
        name: str
    ) -> Mapping[str, Any]:

    if value is None:
        return {}

    return cfg_require_mapping(value, name)


# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------

def require_dict(
        value: Any | None, 
        name: str
    ) -> Dict[str, Any]:

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(
            f"{name} must be a dictionary, got {type(value).__name__}: {value!r}"
        )

    return cast(Dict[str, Any], value)

# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------

def require_list(
        value: Any | None, 
        name: str
    ) -> List[Any]:

    if value is None:
        return []

    if not isinstance(value, list):
        raise TypeError(
            f"{name} must be a list, got {type(value).__name__}: {value!r}"
        )

    return cast(List[Any], value)


# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------

def scalar_to_float(
        value: object, 
        name: str = "value"
    ) -> float:

    if value is None:
        raise ValueError(f"{name} cannot be None.")

    if value is pd.NA:
        raise ValueError(f"{name} cannot be pandas NA.")

    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"{name} cannot be NaN.")

    return float(cast(Any, value))


# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------

def normalize_feature_columns(
        value: Any, 
        name: str = "feature_columns"
    ) -> list[str]:

    """
    Normalize feature-column results into list[str].

    Handles:
    - list[str]
    - tuple[list[str], dict]
    - single string
    """
    if isinstance(value, tuple):
        if len(value) == 0:
            return []
        value = value[0]

    if isinstance(value, str):
        raw_values = [value]
    elif isinstance(value, list):
        raw_values = value
    else:
        raise TypeError(
            f"{name} must be a list[str], tuple[list[str], dict], or str; "
            f"got {type(value).__name__}: {value!r}"
        )

    return [
        str(column_name).strip()
        for column_name in raw_values
        if str(column_name).strip() != ""
    ]


# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------



def require_mapping(value: Any, name: str) -> dict[str, Any]:
    """
    Validate that a loaded JSON/config object is a dictionary.
    """
    if not isinstance(value, dict):
        raise TypeError(
            f"{name} must be a dictionary. "
            f"Got {type(value).__name__}: {value!r}"
        )

    return value

# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------

def require_str_list(value: Any, name: str) -> list[str]:
    """
    Validate that a loaded JSON/config object is a list of strings.
    """
    if value is None:
        raise ValueError(f"{name} is None.")

    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"{name} must be a list/tuple of column names. "
            f"Got {type(value).__name__}: {value!r}"
        )

    cleaned_values = [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]

    if not cleaned_values:
        raise ValueError(f"{name} resolved to an empty list.")

    return cleaned_values

# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------

def require_float(value: Any, name: str) -> float:
    """
    Convert a scalar or threshold-return tuple into a float.

    Some project helpers may return either:
        threshold
    or:
        (threshold, metadata)
    """
    if isinstance(value, tuple):
        if len(value) == 0:
            raise ValueError(f"{name} tuple is empty.")
        value = value[0]

    return float(value)

# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------

def as_bool_array(value: Any, name: str) -> np.ndarray:
    """
    Convert a Pandas/NumPy boolean mask into a NumPy bool array.
    """
    if isinstance(value, pd.Series):
        return value.to_numpy(dtype=bool)

    return np.asarray(value, dtype=bool)

# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------

def as_int_array(value: Any, name: str) -> np.ndarray:
    """
    Convert labels/flags into a NumPy int array.
    """
    if value is None:
        raise ValueError(f"{name} is None.")

    if isinstance(value, pd.Series):
        return value.fillna(0).astype(int).to_numpy(dtype=int)

    return np.asarray(value, dtype=int)

# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------

def as_float_array(value: Any, name: str) -> np.ndarray:
    """
    Convert scores into a flat NumPy float array.
    """
    if value is None:
        raise ValueError(f"{name} is None.")

    return np.asarray(value, dtype=float).reshape(-1)

# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------
def choose_threshold_by_percentile(
    scores: Sequence[float],
    percentile: float = 95.0,
    *,
    return_info: bool = False,
) -> float | Tuple[float, Dict[str, Any]]:
    """
    Choose anomaly threshold using a score percentile.

    Compatibility behavior
    ----------------------
    - Notebook-style usage:
        threshold = choose_threshold_by_percentile(scores, 95.0)
      returns a float threshold.
    - Pipeline-style usage:
        threshold, info = choose_threshold_by_percentile(scores, 95.0, return_info=True)
      returns both threshold and metadata.
    """
    scores_array = np.asarray(scores, dtype=float)

    if scores_array.size == 0:
        raise ValueError("Cannot choose threshold from empty score array.")

    threshold = float(np.percentile(scores_array, float(percentile)))

    info = {
        "threshold_method": "percentile",
        "percentile": float(percentile),
        "threshold": threshold,
        "score_count": int(scores_array.size),
        "score_min": float(np.min(scores_array)),
        "score_max": float(np.max(scores_array)),
        "score_mean": float(np.mean(scores_array)),
    }

    if return_info:
        return threshold, info
    return threshold

# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------
def choose_threshold_value(scores: Any, percentile: float) -> float:
    """
    Normalize score input and threshold helper output for Pylance.
    """
    score_values = as_float_array(scores, "scores")
    threshold_result = choose_threshold_by_percentile(
        score_values.tolist(),
        percentile,
    )
    return require_float(threshold_result, "threshold")

# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------

def normalize_text_value(value: Any) -> str:
    """
    Convert any scalar value into lowercase stripped text.
    """
    if value is None:
        return ""

    return str(value).strip().lower()

# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------

def get_nested_mapping(
    value: dict[str, Any],
    key: str,
    name: str,
) -> dict[str, Any]:
    """
    Safely extract a nested dictionary from a truth/config record.
    """
    raw_value = value.get(key, {})

    if raw_value is None:
        return {}

    if not isinstance(raw_value, dict):
        raise TypeError(
            f"{name}.{key} must be a dictionary. "
            f"Got {type(raw_value).__name__}: {raw_value!r}"
        )

    return raw_value



# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------

def require_truth_record(value: Any, name: str) -> dict[str, Any]:
    """
    Validate a loaded truth record.
    """
    if not isinstance(value, dict):
        raise TypeError(
            f"{name} must be a dictionary. "
            f"Got {type(value).__name__}: {value!r}"
        )

    return value


# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------

def require_int_value(value: int | None, name: str) -> int:
    """
    Validate that a nullable integer value is present before converting it.
    """
    if value is None:
        raise ValueError(f"{name} could not be resolved and is None.")

    return int(value)

# --------------------------------------------------------------------------------------------------------------------------------
#
# --------------------------------------------------------------------------------------------------------------------------------


