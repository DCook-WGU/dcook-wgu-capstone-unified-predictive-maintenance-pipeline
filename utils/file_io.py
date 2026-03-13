from pathlib import Path

from typing import List, Optional, Tuple

from datetime import datetime
import pandas as pd
import numpy as np

import json
import hashlib 

import logging

# Initiate Logging
logger = logging.getLogger("capstone.file_io")


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


# Helper Function to check file_path and resolve

def _resolve_path(file_path, file_name=None):

    if isinstance(file_path, tuple):
        if len(file_path) != 2:
            raise ValueError(
                f"Tuple for file_path must have a length of 2 (directory, file name), got {len(file_path)}"
            )
        dir_part, name_part = file_path
        return Path(dir_part) / name_part

    if file_name is not None:
        return Path(file_path) / file_name
    
    return Path(file_path)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



def _clean_values(series: pd.Series) -> pd.Series:
    values = (
        series.dropna()
        .astype("string")
        .str.strip()
    )
    return values[values != ""]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def _normalize_dataset_name(dataset_name: str) -> str:
    normalized_value = str(dataset_name).strip().lower()
    normalized_value = normalized_value.replace(" ", "_")
    normalized_value = normalized_value.replace("-", "_")

    cleaned_characters = []
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


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



def _generate_deterministic_dataset_name_from_file_details(path_value: Optional[str]) -> Optional[str]:
    """
    Build a deterministic dataset name from file details.

    Uses:
    - file stem
    - file size in bytes
    - modified timestamp
    - short hash of those combined details
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

    if path_object.exists() and path_object.is_file():
        stat_result = path_object.stat()
        file_size_bytes = str(int(stat_result.st_size))
        modified_timestamp = str(int(stat_result.st_mtime))

    identity_text = "|".join(
        [
            file_stem_normalized,
            file_size_bytes,
            modified_timestamp,
        ]
    )

    identity_hash = hashlib.sha1(identity_text.encode("utf-8")).hexdigest()[:8]

    generated_dataset_name = (
        f"{file_stem_normalized}_{file_size_bytes}_{modified_timestamp}_{identity_hash}"
    )

    return _normalize_dataset_name(generated_dataset_name)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



# Creating Compact Hash ID - May not end up using 



def _create_record_id(
        dataframe,
        source_file_column = "meta__source_file",
        source_row_column = "meta__source_row_id",
        out_column = "meta__record_id"
    
    ):

    """
    Create a deterministic record identifier from source lineage fields.

    The record ID is derived from (meta__source_file, meta__source_row_id). This provides
    a stable, compact key that remains consistent for a given input artifact, which is
    useful for deduplication and reproducible joins across pipeline stages.

    Note: if `meta__source_file` values are environment-dependent (e.g., absolute paths),
    normalize to a stable identifier (e.g., basename or a source file ID) before hashing.
    """

    dataframe = dataframe.copy() 

    dataframe[source_file_column] = dataframe[source_file_column].astype("string").fillna("")

    dataframe[source_row_column] = pd.to_numeric(dataframe[source_row_column], errors="coerce").fillna(-1).astype("int64")
    
    dataframe[out_column] = pd.util.hash_pandas_object(
        dataframe[[source_file_column, source_row_column]], 
        index=False
    ).astype("uint64")

    return dataframe 


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



# Load Function
def ingest_data(
        file_path, 
        file_name=None, 
        dataset_name=None, 
        dataset_name_config=None,
        dataset_candidates=None,
        split="unsplit", 
        #label_type=pd.NA,
        #label_source=pd.NA,
        run_id="run__001",
        asset_id="asset__001",
        add_record_id: bool = False,
        validate: bool = True,
        **read_kwargs
    ):
    """
    Ingest a raw dataset into the Bronze layer.

    Design intent
    ------------
    Bronze ingestion establishes a consistent schema contract across datasets so that
    downstream stages (EDA, validation, feature selection, windowing, and modeling)
    can be implemented once and reused without dataset-specific branching.

    The function adds a small set of `meta__*` fields that support:
    - Lineage: trace each record back to its source file and row position
    - Reproducibility: preserve when the Bronze artifact was created (UTC)
    - Dataset context: standard hooks for dataset identity, split, and run grouping
    - Optional stable record keys: deterministic IDs for deduplication and joins

    Parameters
    ----------
    file_path, file_name:
        Locate a raw file (CSV/Parquet).
    dataset_name:
        Logical dataset identifier used for multi-source pipelines and reporting.
    split:
        Dataset split label (e.g., train/test/val). Use "unsplit" when the dataset
        is not pre-partitioned or when partitioning is done downstream.
    run_id:
        Group identifier for sequential/related observations (e.g., unit/run/simulation).
        For single-run datasets, a constant value (e.g., "run_000") is sufficient.
    label_type:
        Optional descriptor of label semantics (e.g., "fault_number", "anomaly_flag").
        Labels themselves remain separate columns if present.
    add_record_id:
        If True, compute a deterministic record key from source lineage fields.

    Notes
    -----
    - This function does not perform dataset-specific cleaning; that belongs in Silver.
    - Metadata columns are prefixed with `meta__` to prevent accidental inclusion as features.
    """

    path = _resolve_path(file_path, file_name)
    suffix = path.suffix.lower()
    
    rk = dict(read_kwargs)
    
    logger.info("Loading Data file: %s", path)

    try:
        if suffix == ".csv":
            dataframe = pd.read_csv(path, **rk)
        elif suffix in {".parquet", ".pq"}:
            engine = rk.pop("engine", "pyarrow")
            dataframe = pd.read_parquet(path, engine=engine, **rk)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    except Exception:
        logger.exception("Error reading file: %s", path)
        raise

    logger.info(
        "Loaded Data File: %s | shape=%s | columns=%s",
        path.name,
        dataframe.shape,
        list(dataframe.columns),
    )

    # --- Bronze metadata contract ---
    # `meta__*` fields are reserved for lineage/audit/context. Downstream feature selection
    # can safely exclude these columns without maintaining dataset-specific exclusion lists.

    if dataset_candidates is None:
        dataset_candidates = []

    RESOLVED_DATASET_NAME, DATASET_SOURCE_COLUMN, DATASET_METHOD = resolve_dataset_name_for_bronze(
        dataframe=dataframe,
        dataset_candidates=dataset_candidates,
        argument_value=dataset_name,
        config_value=dataset_name_config,
        fallback_value="unknown_dataset",
        source_path=str(path),
        bronze_source_column="meta__dataset",
        fail_on_multiple_in_dataset=False,
    )

    logger.info(
        "Resolved dataset name during Bronze ingest | dataset_name=%s | source_column=%s | method=%s",
        RESOLVED_DATASET_NAME,
        DATASET_SOURCE_COLUMN,
        DATASET_METHOD,
    )

    # Bronze Metadata Contract
    dataframe["meta__dataset"] = RESOLVED_DATASET_NAME
    dataframe["meta__split"] = split
    dataframe["meta__run_id"] = run_id
    dataframe["meta__asset_id"] = asset_id
    #dataframe["meta__label_type"] = label_type
    #dataframe["meta__label_source"] = label_source
    
    ingested_at_utc = pd.Timestamp.now(tz="UTC")
    dataframe["meta__ingested_at_utc"] = ingested_at_utc

    #dataframe["meta__ingested_at_utc"] = pd.Timestamp.now(tz="UTC")
    dataframe["meta__source_file"] = path.name
    dataframe["meta__source_row_id"] = np.arange(len(dataframe), dtype=np.int64)
    

    # Optional deterministic record key.
    # For multi-file ingestion or reprocessing scenarios, a stable record ID enables:
    # - deduplication checks
    # - reproducible joins across intermediate artifacts
    # - straightforward traceability in diagnostics
    #
    # For single-file/single-run datasets, (meta__source_file, meta__source_row_id) is already unique.
    # We keep `meta__record_id` optional so the same utility can be reused across datasets

    if add_record_id:

        required = {"meta__source_file", "meta__source_row_id"}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"Cannot create record_id; missing columns: {missing}")
        
        dataframe = _create_record_id(
            dataframe,
            source_file_column = "meta__source_file",
            source_row_column = "meta__source_row_id",
            out_column = "meta__record_id"
        )
    

    # Cast Low-Cardinality Meta Columns into category type
    #for column in ["meta__dataset", "meta__split", "meta__label_type", "meta__label_source"]:
    for column in ["meta__dataset", "meta__split"]:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype("category")

    # Manual Casting: 
    # run_id and source_id are not included in the above category casting
    # This is because many unique run_ids or many files, category can cause
    # category casting to become counterproductive.
    
    # dataframe["meta__run_id"] = dataframe["meta__run_id"].astype("category")
    # dataframe["meta__source_file"] = dataframe["meta__source_file"].astype("category")

    '''
    columns_to_front = [
        "meta__dataset", "meta__split", 
        "meta__run_id", "meta__asset_id", 
        "meta__label_type",  "meta__label_source",
        "meta__ingested_at_utc", 
        "meta__source_file", "meta__source_row_id"
        ] + (["meta__record_id"] if add_record_id else [])
    '''

    columns_to_front = [
        "meta__dataset", "meta__split", 
        "meta__run_id", "meta__asset_id", 
        "meta__ingested_at_utc", 
        "meta__source_file", "meta__source_row_id"
        ] + (["meta__record_id"] if add_record_id else [])


    dataframe = dataframe[columns_to_front + [column for column in dataframe.columns if column not in columns_to_front]]

    if validate:

        # --- Integrity checks (Bronze contract) ---
        # These are lightweight guards that help catch accidental reindexing, duplication,
        # or unexpected row expansion during ingestion.
       
        #assert dataframe["meta__source_row_id"].is_unique, "meta__source_row_id must be unique within the ingest."
        
        if not dataframe["meta__source_row_id"].is_unique:
            raise ValueError("meta__source_row_id must be unique within the ingest.")

        #assert dataframe["meta__source_row_id"].iloc[0] == 0, "meta__source_row_id must start at 0."

        if not dataframe["meta__source_row_id"].iloc[0] == 0:
            raise ValueError("meta__source_row_id must start at 0.")


        #assert dataframe["meta__source_row_id"].iloc[-1] == len(dataframe) - 1, "meta__source_row_id must be contiguous 0..n-1."

        if not dataframe["meta__source_row_id"].iloc[-1] == len(dataframe) - 1:
            raise ValueError("meta__source_row_id must be contiguous 0..n-1.")

        # Linage Key Explicity
        #assert not dataframe.duplicated(["meta__source_file", "meta__source_row_id"]).any(), \
        #    "Duplicate (meta__source_file, meta__source_row_id) lineage keys detected."
        
        if dataframe.duplicated(["meta__source_file", "meta__source_row_id"]).any():
                raise ValueError("Duplicate (meta__source_file, meta__source_row_id) lineage keys detected.")

        #if add_record_id:
        #    assert dataframe["meta__record_id"].is_unique, "meta__record_id must be unique for this ingest."

        if add_record_id and (not dataframe["meta__record_id"].is_unique):
                raise ValueError("meta__record_id must be unique within the ingest.")
        
        if len(dataframe) == 0:
            raise ValueError("Ingest produced an empty dataframe")
        
    dataframe.attrs["dataset_resolution"] = {
        "dataset_name": RESOLVED_DATASET_NAME,
        "dataset_source_column": DATASET_SOURCE_COLUMN,
        "dataset_method": DATASET_METHOD,
        "priority_order": [
            "argument",
            "config",
            "inside_dataset",
            "file_details",
            "fallback",
        ],
    }

    return dataframe


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def load_data(
        file_path, 
        file_name=None, 
        engine: str = "pyarrow", 
        **read_kwargs
    ):
    

    path = _resolve_path(file_path, file_name)
    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            logger.info("Loading CSV: %s", path)
            return pd.read_csv(path, **read_kwargs)

        elif suffix in {".parquet", ".pq"}:
            logger.info("Loading Parquet: %s", path)
            return pd.read_parquet(path, engine=engine, **read_kwargs)

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    except Exception:
        logger.exception("Error reading file: %s", path)
        raise


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def save_data(
        dataframe, 
        file_path, 
        file_name=None, 
        create_dirs=True, 
        index=False, 
        **write_kwargs
    ):
    

    path = _resolve_path(file_path, file_name)

    if not path.suffix:
        path = path.with_suffix(".parquet")

    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            logger.info("Saving DataFrame to CSV: %s", path)
            dataframe.to_csv(path, index=index, **write_kwargs)

        elif suffix in {".parquet", ".pq"}:
            logger.info("Saving DataFrame to Parquet: %s", path)
            compression = write_kwargs.pop("compression", "snappy")
            engine = write_kwargs.pop("engine", "pyarrow")
            dataframe.to_parquet(path, index=index, engine=engine, compression=compression, **write_kwargs)

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    except Exception:
        logger.exception("Error writing file: %s", path)
        raise

    logger.info("Saved: %s | shape=%s | columns=%s", path.name, dataframe.shape, list(dataframe.columns))
    return path


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def _json_default(value):

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.ndarray,)):
        return value.tolist()

    return str(value)

def save_json(
        data, 
        file_path, 
        file_name=None, 
        create_dirs=True, 
        indent=2
    ):

    path = _resolve_path(file_path, file_name)

    if not path.suffix:
        path = path.with_suffix(".json")

    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=indent, default=_json_default)

    logger.info("Saved JSON: %s", path)
    return path

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def load_json(
    file_path,
    file_name=None,
    *,
    encoding: str = "utf-8",
    default=None,
    raise_if_missing: bool = True,
):
    """
    Load a JSON file and return the parsed Python object (dict/list/etc.).

    Parameters
    ----------
    file_path, file_name:
        Path to the JSON file (same patterns as load_data/save_data).
    default:
        Value to return if the file is missing and raise_if_missing=False.
    raise_if_missing:
        If True, raise FileNotFoundError when the file does not exist.
        If False, return `default` when missing.
    """

    path = _resolve_path(file_path, file_name)

    # If no suffix, assume it's json
    if not path.suffix:
        path = path.with_suffix(".json")

    if not path.exists():
        if raise_if_missing:
            raise FileNotFoundError(f"JSON file not found: {path}")
        logger.info("JSON file not found (returning default): %s", path)
        return default

    logger.info("Loading JSON: %s", path)

    try:
        with open(path, "r", encoding=encoding) as file_handle:
            return json.load(file_handle)
    except Exception:
        logger.exception("Error reading JSON file: %s", path)
        raise


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def resolve_dataset_name_for_bronze(
        dataframe: pd.DataFrame,
        *,
        dataset_candidates: List[str],
        argument_value: Optional[str] = None,
        config_value: Optional[str] = None,
        fallback_value: Optional[str] = None,
        source_path: Optional[str] = None,
        bronze_source_column: str = "meta__dataset",
        fail_on_multiple_in_dataset: bool = True,
    ) -> Tuple[str, str, str]:
    """
    Resolve dataset name during Bronze ingestion using the following priority order:

    1. CLI / Argument
    2. Config File
    3. Inside Dataset
    4. Deterministic file-details-based generated name
    5. Fallback

    Returns
    -------
    Tuple[str, str, str]
        dataset_name,
        dataset_source_column,
        dataset_method
    """


    #### #### #### #### 
    # 1. CLI / Argument

    if argument_value is not None and str(argument_value).strip() != "":
        return (
            _normalize_dataset_name(str(argument_value)),
            "argument",
            "argument",
        )

    #### #### #### #### 
    # 2. Config File

    if config_value is not None and str(config_value).strip() != "":
        return (
            _normalize_dataset_name(str(config_value)),
            "config",
            "config",
        )

    #### #### #### #### 
    # 3. Inside Dataset
    #    First check meta__dataset, then approved dataset candidate columns

    if bronze_source_column in dataframe.columns:
        values = _clean_values(dataframe[bronze_source_column])

        if len(values) > 0:
            unique_values = values.unique()

            if len(unique_values) == 1:
                return (
                    _normalize_dataset_name(str(unique_values[0])),
                    bronze_source_column,
                    "unique",
                )

            if fail_on_multiple_in_dataset:
                raise ValueError(
                    f"Multiple values are not allowed from bronze source '{bronze_source_column}'. "
                    f"Values discovered: {unique_values[:10]}"
                )

            top_value = values.value_counts().index[0]
            return (
                _normalize_dataset_name(str(top_value)),
                bronze_source_column,
                "mode",
            )

    for column in dataset_candidates:
        if column == bronze_source_column:
            continue
        if column not in dataframe.columns:
            continue

        values = _clean_values(dataframe[column])

        if len(values) == 0:
            continue

        unique_values = values.unique()

        if len(unique_values) == 1:
            return (
                _normalize_dataset_name(str(unique_values[0])),
                column,
                "unique",
            )

        if fail_on_multiple_in_dataset:
            raise ValueError(
                f"Multiple values are not allowed from dataset candidate '{column}'. "
                f"Values discovered: {unique_values[:10]}"
            )

        top_value = values.value_counts().index[0]
        return (
            _normalize_dataset_name(str(top_value)),
            column,
            "mode",
        )

    #### #### #### #### 
    # 4. Deterministic file-details-based generated name

    generated_dataset_name = _generate_deterministic_dataset_name_from_file_details(source_path)

    if generated_dataset_name is not None:
        return (
            generated_dataset_name,
            "source_path",
            "file_details",
        )

    #### #### #### #### 
    # 5. Fallback

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

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


