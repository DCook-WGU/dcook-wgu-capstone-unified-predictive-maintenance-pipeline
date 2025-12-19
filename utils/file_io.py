from pathlib import Path

import pandas as pd
import numpy as np

import logging

# Initiate Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# Helper Function to check file_path and resolve

def _resolve_path(file_path, file_name):

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
def _create_record_id(dataframe):
    
    dataframe["record_id"] = pd.util.hash_pandas_object(
        dataframe[["_source_file", "_source_row_id"]], index=False
    ).astype("uint64")

    return dataframe 


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 



# Load Function
def ingest_data(
        file_path, 
        file_name=None, 
        dataset_name=None, 
        split=pd.NA, 
        is_labeled=pd.NA, 
        label_type=pd.NA,
        run_id=pd.NA,
        add_record_id: bool = False,
        **read_kwargs
    ):
    """
    Load a CSV file from a base data directory and file name.

    Parameters
    ----------
    file_path : str or Path or tuple
        - If `file_name` is None:
            - can be a full path: "/data/raw/pump_sensor/sensor.csv"
            - OR a (dir, filename) tuple: ("/data/raw/pump_sensor", "sensor.csv")
        - If `file_name` is not None:
            - `file_path` is treated as a directory, and combined with `file_name`.
    file_name : str or Path, optional
        Optional file name to join with `file_path` when you pass directory + name.
    **read_csv_kwargs :
        Additional keyword arguments passed to pandas.read_csv().

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """

    path = _resolve_path(file_path, file_name)
    suffix = path.suffix.lower()
    
    rk = dict(read_kwargs)
    
    logger.info(f"Loading CSV file: {path }")

    try:
        if suffix == ".csv":
            dataframe = pd.read_csv(path, **rk)
        elif suffix in {".parquet", ".pq"}:
            engine = rk.pop("engine", "pyarrow")
            dataframe = pd.read_parquet(path, engine=engine, **rk)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    except Exception:
        logger.exception(f"Error reading CSV file: {path}")
        raise

    logger.info(
        "Loaded CSV: %s | shape=%s | columns=%s",
        path.name,
        dataframe.shape,
        list(dataframe.columns),
    )

    dataframe["_source_file"] = path.name
    dataframe["dataset_name"] = dataset_name if dataset_name is not None else pd.NA
    dataframe["bronze_ingested_at"] = pd.Timestamp.utcnow().isoformat()
    dataframe["_source_row_id"] = np.arange(len(dataframe), dtype=np.int64)
    
    dataframe["split"] = split
    dataframe["is_labeled"] = is_labeled
    dataframe["label_type"] = label_type
    dataframe["run_id"] = run_id

    if add_record_id:
        required = {"_source_file", "_source_row_id"}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"Cannot create record_id; missing columns: {missing}")
        dataframe = _create_record_id(dataframe)

    for column in ["_source_file", "dataset_name", "split", "is_labeled", "label_type", "run_id"]:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype("category")

    columns_to_front = [
        "_source_file", "dataset_name", "bronze_ingested_at", "_source_row_id", 
        "run_id", "split", "is_labeled", "label_type" 
    ] + (["record_id"] if add_record_id else [])

    dataframe = dataframe[columns_to_front + [column for column in dataframe.columns if column not in columns_to_front]]

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
            logger.info(f"Loading CSV: {path}")
            return pd.read_csv(path, **read_kwargs)

        elif suffix in {".parquet", ".pq"}:
            logger.info(f"Loading Parquet: {path}")
            return pd.read_parquet(path, engine=engine, **read_kwargs)

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    except Exception:
        logger.exception(f"Error reading file: {path}")
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
            logger.info(f"Saving DataFrame to CSV: {path}")
            dataframe.to_csv(path, index=index, **write_kwargs)

        elif suffix in {".parquet", ".pq"}:
            logger.info(f"Saving DataFrame to Parquet: {path}")
            compression = write_kwargs.pop("compression", "snappy")
            engine = write_kwargs.pop("engine", "pyarrow")
            dataframe.to_parquet(path, index=index, engine=engine, compression=compression, **write_kwargs)

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    except Exception:
        logger.exception(f"Error writing file: {path}")
        raise

    logger.info("Saved: %s | shape=%s | columns=%s", path.name, dataframe.shape, list(dataframe.columns))
    return path


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
