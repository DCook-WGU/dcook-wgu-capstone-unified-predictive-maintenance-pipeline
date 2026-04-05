from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import hashlib
import json
import pandas as pd


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_process_run_id(prefix: str = "process") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}__{timestamp}"

def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, list):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value.as_posix())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def compute_sha256(payload: Dict[str, Any]) -> str:
    payload_str = json.dumps(_normalize_for_json(payload), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def identify_meta_columns(dataframe: pd.DataFrame) -> list[str]:
    return sorted([c for c in dataframe.columns if str(c).startswith("meta__")])


def identify_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    return sorted([c for c in dataframe.columns if not str(c).startswith("meta__")])


def extract_truth_hash(dataframe: pd.DataFrame, column_name: str = "meta__truth_hash") -> Optional[str]:
    if column_name not in dataframe.columns:
        return None

    non_null = dataframe[column_name].dropna()
    if len(non_null) == 0:
        return None

    values = non_null.astype(str).unique().tolist()
    if len(values) == 0:
        return None
    if len(values) > 1:
        raise ValueError(f"Expected 1 unique truth hash in '{column_name}', found {len(values)}: {values[:10]}")
    return values[0]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def build_file_fingerprint(file_path: str | Path) -> Dict[str, Any]:
    file_path = Path(file_path)
    stat = file_path.stat()

    payload = {
        "file_name": file_path.name,
        "file_path": str(file_path.as_posix()),
        "size_bytes": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
    }
    payload["fingerprint_hash"] = compute_sha256(payload)
    return payload


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def initialize_layer_truth(
    *,
    truth_version: str,
    dataset_name: str,
    layer_name: str,
    process_run_id: str,
    pipeline_mode: str,
    parent_truth_hash: Optional[str],
) -> Dict[str, Any]:
    return {
        "truth_version": truth_version,
        "dataset_name": dataset_name,
        "layer_name": layer_name,
        "process_run_id": process_run_id,
        "pipeline_mode": pipeline_mode,
        "parent_truth_hash": parent_truth_hash,
        "source_fingerprint": {},
        "config_snapshot": {},
        "runtime_facts": {},
        "artifact_paths": {},
        "notes": {},
    }


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def update_truth_section(
    truth: Dict[str, Any],
    section: str,
    values: Dict[str, Any],
) -> Dict[str, Any]:
    updated = deepcopy(truth)
    updated.setdefault(section, {})
    updated[section].update(values)
    return updated


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def build_truth_record(
    *,
    truth_base: Dict[str, Any],
    row_count: int,
    column_count: int,
    meta_columns: list[str],
    feature_columns: list[str],
) -> Dict[str, Any]:
    payload = {
        "truth_version": truth_base["truth_version"],
        "dataset_name": truth_base["dataset_name"],
        "layer_name": truth_base["layer_name"],
        "process_run_id": truth_base["process_run_id"],
        "pipeline_mode": truth_base["pipeline_mode"],
        "parent_truth_hash": truth_base["parent_truth_hash"],
        "source_fingerprint": truth_base.get("source_fingerprint", {}),
        "row_count": int(row_count),
        "column_count": int(column_count),
        "meta_columns": sorted(meta_columns),
        "feature_columns": sorted(feature_columns),
        "config_snapshot": truth_base.get("config_snapshot", {}),
        "runtime_facts": truth_base.get("runtime_facts", {}),
        "artifact_paths": truth_base.get("artifact_paths", {}),
    }

    truth_hash = compute_sha256(payload)

    truth_record = {
        "truth_hash": truth_hash,
        "created_at_utc": utc_now_iso(),
        **payload,
        "notes": truth_base.get("notes", {}),
    }

    return truth_record


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def save_truth_record(
    truth_record: Dict[str, Any],
    *,
    truth_dir: str | Path,
    dataset_name: str,
    layer_name: str,
) -> Path:
    truth_dir = Path(truth_dir) / layer_name
    truth_dir.mkdir(parents=True, exist_ok=True)

    truth_hash = truth_record["truth_hash"]
    out_name = f"{dataset_name}__{layer_name}__truth__{truth_hash}.json"
    out_path = truth_dir / out_name

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_normalize_for_json(truth_record), f, indent=2, ensure_ascii=False)

    return out_path


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def append_truth_index(
    truth_record: Dict[str, Any],
    *,
    truth_index_path: str | Path,
) -> None:
    truth_index_path = Path(truth_index_path)
    truth_index_path.parent.mkdir(parents=True, exist_ok=True)

    with truth_index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_normalize_for_json(truth_record), ensure_ascii=False) + "\n")


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def stamp_truth_columns(
    dataframe: pd.DataFrame,
    *,
    truth_hash: str,
    parent_truth_hash: Optional[str] = None,
    pipeline_mode: Optional[str] = None,
) -> pd.DataFrame:
    df = dataframe.copy()
    df["meta__truth_hash"] = truth_hash
    df["meta__parent_truth_hash"] = parent_truth_hash

    if pipeline_mode is not None:
        df["meta__pipeline_mode"] = pipeline_mode

    return df


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def load_truth_record(truth_path: str | Path) -> Dict[str, Any]:
    truth_path = Path(truth_path)

    with truth_path.open("r", encoding="utf-8") as f:
        return json.load(f)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def find_truth_record_by_hash(
    *,
    truth_dir: str | Path,
    layer_name: str,
    dataset_name: str,
    truth_hash: str,
) -> Path:
    truth_dir = Path(truth_dir) / layer_name
    expected_path = truth_dir / f"{dataset_name}__{layer_name}__truth__{truth_hash}.json"

    if not expected_path.exists():
        raise FileNotFoundError(
            f"Truth record not found for dataset='{dataset_name}', "
            f"layer='{layer_name}', truth_hash='{truth_hash}'. "
            f"Expected path: {expected_path}"
        )

    return expected_path

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def load_truth_record_by_hash(
    *,
    truth_dir: str | Path,
    layer_name: str,
    dataset_name: str,
    truth_hash: str,
) -> Dict[str, Any]:
    truth_path = find_truth_record_by_hash(
        truth_dir=truth_dir,
        layer_name=layer_name,
        dataset_name=dataset_name,
        truth_hash=truth_hash,
    )
    return load_truth_record(truth_path)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def load_parent_truth_record_from_dataframe(
    dataframe: pd.DataFrame,
    *,
    truth_dir: str | Path,
    parent_layer_name: str,
    dataset_name: str,
    column_name: str = "meta__truth_hash",
) -> Dict[str, Any]:
    parent_truth_hash = extract_truth_hash(dataframe, column_name=column_name)

    if parent_truth_hash is None:
        raise ValueError(
            f"No parent truth hash found in dataframe column '{column_name}'."
        )

    return load_truth_record_by_hash(
        truth_dir=truth_dir,
        layer_name=parent_layer_name,
        dataset_name=dataset_name,
        truth_hash=parent_truth_hash,
    )


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

'''
def get_dataset_name_from_truth(truth_record: Dict[str, Any]) -> str:
    dataset_name = truth_record.get("dataset_name")

    if dataset_name is None or str(dataset_name).strip() == "":
        raise ValueError("Truth record is missing a usable dataset_name.")

    return str(dataset_name).strip()
'''

#### #### #### #### 

def get_dataset_name_from_truth(truth_record: Dict[str, Any]) -> str:
    return get_truth_value(truth_record, "dataset_name")



#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

'''
def get_truth_hash(truth_record: Dict[str, Any]) -> str:
    truth_hash = truth_record.get("truth_hash")

    if truth_hash is None or str(truth_hash).strip() == "":
        raise ValueError("Truth record is missing a usable truth_hash.")

    return str(truth_hash).strip()
'''

#### #### #### #### 

def get_truth_hash(truth_record: Dict[str, Any]) -> str:
    return get_truth_value(truth_record, "truth_hash")

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

'''
def get_parent_truth_hash(truth_record: Dict[str, Any]) -> Optional[str]:
    parent_truth_hash = truth_record.get("parent_truth_hash")

    if parent_truth_hash is None or str(parent_truth_hash).strip() == "":
        return None

    return str(parent_truth_hash).strip()
'''

#### #### #### #### 

def get_parent_truth_hash(truth_record: Dict[str, Any]) -> Optional[str]:
    return get_truth_value(truth_record, "parent_truth_hash", required=False)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

'''
def get_pipeline_mode_from_truth(truth_record: Dict[str, Any]) -> Optional[str]:
    pipeline_mode = truth_record.get("pipeline_mode")

    if pipeline_mode is None or str(pipeline_mode).strip() == "":
        return None

    return str(pipeline_mode).strip()
'''

#### #### #### #### 

def get_pipeline_mode_from_truth(truth_record: Dict[str, Any]) -> Optional[str]:
    return get_truth_value(truth_record, "pipeline_mode", required=False)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

'''
def get_artifact_path_from_truth(
    truth_record: Dict[str, Any],
    key: str,
) -> str:
    artifact_paths = truth_record.get("artifact_paths", {})
    value = artifact_paths.get(key)

    if value is None or str(value).strip() == "":
        raise ValueError(f"Truth record is missing artifact_paths['{key}'].")

    return str(value).strip()
'''

#### #### #### #### 

def get_artifact_path_from_truth(truth_record: Dict[str, Any], key: str) -> str:
    return get_truth_value(truth_record, "artifact_paths", key)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def get_truth_value(
    truth_record: Dict[str, Any],
    *path: str,
    required: bool = True,
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Retrieve a string value from a truth record using a key path.

    Examples:
        get_truth_value(truth_record, "dataset_name")
        get_truth_value(truth_record, "parent_truth_hash", required=False)
        get_truth_value(truth_record, "artifact_paths", "silver_train_path")
    """
    current_value: Any = truth_record
    traversed_path: list[str] = []

    for key in path:
        traversed_path.append(key)

        if not isinstance(current_value, dict):
            current_value = None
            break

        current_value = current_value.get(key)

    if current_value is None or str(current_value).strip() == "":
        if required:
            formatted_path = " -> ".join(traversed_path)
            raise ValueError(f"Truth record is missing a usable value for: {formatted_path}.")
        return default

    return str(current_value).strip()


#### 

# Usage/Function Calls:
'''
dataset_name = get_truth_value(truth_record, "dataset_name")
truth_hash = get_truth_value(truth_record, "truth_hash")
parent_truth_hash = get_truth_value(truth_record, "parent_truth_hash", required=False)
pipeline_mode = get_truth_value(truth_record, "pipeline_mode", required=False)

artifact_path = get_truth_value(truth_record, "artifact_paths", key)
'''

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


