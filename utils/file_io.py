from pathlib import Path

import pandas as pd
import numpy as np

import logging

# Initiate Logging
logger = logging.getLogger("capstone.file_io")


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


# Helper Function to check file_path and resolve

def _resolve_path(file_path, file_name=None):

    if file_name is not None:
        return Path(file_path) / file_name
    
    if isinstance(file_path, tuple):
        if len(file_path) != 2:
            raise ValueError(
                f"Tuple for file_path must have a length of 2 (directory, file name), got {len(file_path)}"
            )
        dir_part, name_part = file_path
        return Path(dir_part) / name_part
    
    return Path(file_path)


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
        split="unsplit", 
        label_type=pd.NA,
        label_source=pd.NA,
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

    dataframe["meta__dataset"] = dataset_name if dataset_name is not None else pd.NA
    dataframe["meta__split"] = split
    dataframe["meta__run_id"] = run_id
    dataframe["meta__asset_id"] = asset_id
    dataframe["meta__label_type"] = label_type
    dataframe["meta__label_source"] = label_source
    dataframe["meta__ingested_at_utc"] = pd.Timestamp.now(tz="UTC")
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
    for column in ["meta__dataset", "meta__split", "meta__label_type", "meta__label_source"]:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype("category")

    # Manual Casting: 
    # run_id and source_id are not included in the above category casting
    # This is because many unique run_ids or many files, category can cause
    # category casting to become counterproductive.
    
    # dataframe["meta__run_id"] = dataframe["meta__run_id"].astype("category")
    # dataframe["meta__source_file"] = dataframe["meta__source_file"].astype("category")


    columns_to_front = [
        "meta__dataset", "meta__split", 
        "meta__run_id", "meta__asset_id", 
        "meta__label_type",  "meta__label_source",
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

