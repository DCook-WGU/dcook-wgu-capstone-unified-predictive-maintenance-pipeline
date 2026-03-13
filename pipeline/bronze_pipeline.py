# %% [markdown]
# ## Bronze Layer (Deliverable 1.1.1 and 1.1.2)
# 
# This notebook completes the Bronze layer of the Medallion Architecture for the pump sensor dataset. It focuses on raw data ingestion, structural validation, and standardization so that the rest of the analytical workflow has a clean and consistent foundation.
# 
# The key goals of this notebook are:
# 
# - To ingest the raw pump sensor data from the source files.
# - To validate the structure, column consistency, data types, and timestamp integrity.
# - To standardize the dataset into the unified Bronze schema, including meta fields, identity fields, and provenance tracking.
# - To export the Bronze dataset in a reproducible format that can be used by the Silver Pre-EDA and Silver EDA notebooks.
# 
# Outputs from this notebook support the project write-up in Section B and Section C by:
# 
# - Establishing the standardized dataset required for all downstream analysis and modeling described in Section C.2 and C.2.A.
# - Providing the meta-data, event identifiers, timestamp fields, and structural consistency checks required for later segmentation of normal vs abnormal periods used in Section C.4.
# - Ensuring that the Silver and Gold layers receive a clean, well-formed dataset, enabling reliable feature behavior analysis, anomaly profiling, and model comparison.
# - Serving as the reproducible starting point for the entire analytical pipeline, consistent with the Medallion Architecture described in B.3.

# %%
print("hello")

# %%
import os
import glob
from pathlib import Path
import yaml
import sys

import json
import logging
import wandb
from datetime import datetime, timezone

from typing import Optional, Tuple, List

import pandas as pd 
import numpy as np

import hashlib

# Custom Utilities Module
from utils.paths import get_paths
from utils.file_io import ingest_data, save_data, load_json
from utils.logging_setup import configure_logging, log_layer_paths
from utils.eda_logging import profile_dataframe

from utils.truths import (
    make_process_run_id,
    build_file_fingerprint,
    extract_truth_hash,
    identify_meta_columns,
    identify_feature_columns,
    initialize_layer_truth,
    update_truth_section,
    build_truth_record,
    save_truth_record,
    append_truth_index,
    stamp_truth_columns,
    load_truth_record,
    find_truth_record_by_hash, 
    load_truth_record_by_hash, 
    load_parent_truth_record_from_dataframe,
)

'''
from utils.wandb_utils import (
    log_metrics,
    log_dataframe_head,
    log_dir_as_artifact,
    log_files_as_artifact,
)
'''
from utils.wandb_utils import finalize_wandb_stage

from utils.pipeline_config_loader import (
    load_pipeline_config,
    build_truth_config_block,
    set_wandb_dir_from_config,
    export_config_snapshot,
)


# Show more columns
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)



# %% [markdown]
# ----

# %%
# Get Path's Object
paths = get_paths()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# Get configs
#CONFIG_ROOT = Path("configs")
CONFIG_ROOT = paths.configs

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


CONFIG_RUN_MODE = "train"
CONFIG_PROFILE = "default"


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


CONFIG = load_pipeline_config(
    config_root=CONFIG_ROOT,
    stage="bronze",
    dataset="pump",
    mode=CONFIG_RUN_MODE,
    profile=CONFIG_PROFILE,
    project_root=paths.root,
).data

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

BRONZE_CFG = CONFIG["bronze"]
PATHS = CONFIG["resolved_paths"]
FILENAMES = CONFIG["filenames"]
PIPELINE = CONFIG.get(
    "pipeline",
    {
        "execution_mode": "batch",
        "orchestration_mode": "notebook",
    },
)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


TRUTH_CONFIG = build_truth_config_block(CONFIG)
TRUTH_CONFIG["pipeline"] = PIPELINE

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


# Paths
RAW_DATA_PATH = Path(PATHS["data_raw_dir"])
BRONZE_DATA_PATH = Path(PATHS["data_bronze_train_dir"])

BRONZE_ARTIFACTS_PATH = Path(PATHS["bronze_artifacts_dir"])
TRUTHS_PATH = Path(PATHS["truths_dir"])
TRUTH_INDEX_PATH = Path(PATHS["truth_index_path"])
LOGS_PATH = Path(PATHS["logs_root"])


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# W&B
WANDB_DIR = set_wandb_dir_from_config(CONFIG)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


# Make sure paths exist
BRONZE_DATA_PATH.mkdir(parents=True, exist_ok=True)
BRONZE_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
TRUTHS_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


# Meta details
LAYER_NAME = BRONZE_CFG["layer_name"]
BRONZE_VERSION = CONFIG["versions"]["bronze"]
TRUTH_VERSION = CONFIG["versions"]["truth"]

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# Keep pipeline execution separate from run mode
PIPELINE_MODE = PIPELINE["execution_mode"]
RUN_MODE = CONFIG["runtime"]["mode"] 
CONFIG_PROFILE = CONFIG["runtime"]["profile"] 

# W&B
WANDB_PROJECT = CONFIG["wandb"]["project"]
WANDB_ENTITY = CONFIG["wandb"]["entity"]
WANDB_RUN_NAME = f"{BRONZE_VERSION}"

# Raw file details
RAW_FILE_PATH = Path(PATHS["raw_file_path"]).parent
RAW_FILE_NAME = CONFIG["dataset"]["raw_file_name"]

# Dataset details

DATASET_NAME_ARGUMENT = None
DATASET_NAME_CONFIG = CONFIG["dataset"]["name"]
DATASET_CANDIDATES = BRONZE_CFG["dataset_candidates"]

SPLIT_STATUS = CONFIG["dataset"]["split_status"]
LABEL_TYPE = CONFIG["dataset"]["label_type"]
LABEL_SOURCE = CONFIG["dataset"]["label_source"]
RUN_ID = CONFIG["dataset"]["run_id"]
ASSET_ID = CONFIG["dataset"]["asset_id"]

# Separate processing lineage id
PROCESS_RUN_ID = make_process_run_id(BRONZE_CFG["process_run_id_prefix"])

ADD_RECORD_ID = bool(BRONZE_CFG["add_record_id"])
RECORD_ID_INPUTS = list(BRONZE_CFG["record_id_inputs"])
RECORD_ID_METHOD = BRONZE_CFG["record_id_method"]

# DataFrame-friendly
LABEL_TYPE_DF = pd.NA if LABEL_TYPE is None else LABEL_TYPE
LABEL_SOURCE_DF = pd.NA if LABEL_SOURCE is None else LABEL_SOURCE

BRONZE_TRAIN_DATA_FILE_NAME = FILENAMES["bronze_train_file_name"]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# %% [markdown]
# ----

# %%

# Create bronze log path 
bronze_log_path = paths.logs / "bronze.log"

# Insure directory exists
bronze_log_path.parent.mkdir(parents=True, exist_ok=True)

# Initial Logger
capstone_logger = configure_logging(
    "capstone",
    bronze_log_path,
    level=logging.DEBUG,
    overwrite_handlers=True,
)

# Initiate Logger and log file
logger = logging.getLogger("capstone.bronze")

# Log load and initiation
logger.info("Bronze stage starting")

log_layer_paths(paths, current_layer="bronze", logger=logger)


# %% [markdown]
# ----

# %%
def resolve_dataset_name_for_bronze_pre_ingest(
        *,
        argument_value: Optional[str] = None,
        config_value: Optional[str] = None,
        fallback_value: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> Tuple[str, str, str]:
    """
    Resolve dataset name before Bronze ingestion.

    Priority order:
    1. CLI / Argument
    2. Config File
    3. Deterministic file-details-based generated name
    4. Fallback

    This resolver is intended for pre-ingest use cases such as:
    - Weights & Biases setup
    - early run naming
    - artifact naming before dataframe load

    It does not inspect the dataset contents because the dataframe does not yet exist.
    """
    #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

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


    #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

    
    '''
    def _generate_deterministic_dataset_name_from_file_details(path_value: Optional[str]) -> Optional[str]:
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
    '''

    def _generate_deterministic_dataset_name_from_file_details(path_value: Optional[str]) -> Optional[str]:
        """
        Build a deterministic dataset name from file details plus a lightweight content fingerprint.

        Uses:
        - file stem
        - file size in bytes
        - modified timestamp
        - short hash from sampled file bytes
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

    #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

    # 1. CLI / Argument
    if argument_value is not None and str(argument_value).strip() != "":
        return (
            _normalize_dataset_name(str(argument_value)),
            "argument",
            "argument",
        )

    # 2. Config File
    if config_value is not None and str(config_value).strip() != "":
        return (
            _normalize_dataset_name(str(config_value)),
            "config",
            "config",
        )

    # 3. Deterministic file-details-based generated name
    generated_dataset_name = _generate_deterministic_dataset_name_from_file_details(source_path)

    if generated_dataset_name is not None:
        return (
            generated_dataset_name,
            "source_path",
            "file_details",
        )

    # 4. Fallback
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

# %%
PROVISIONAL_DATASET_NAME, PROVISIONAL_DATASET_SOURCE_COLUMN, PROVISIONAL_DATASET_METHOD = (
    resolve_dataset_name_for_bronze_pre_ingest(
        argument_value=DATASET_NAME_ARGUMENT,
        config_value=DATASET_NAME_CONFIG,
        fallback_value="unknown_dataset",
        source_path=str(RAW_FILE_PATH / RAW_FILE_NAME),
    )
)

# %%
CONFIG_SNAPSHOT_PATH = (
    BRONZE_ARTIFACTS_PATH / f"{PROVISIONAL_DATASET_NAME}__bronze__resolved_config.yaml"
)

if CONFIG["execution"].get("save_config_snapshot", True):
    export_config_snapshot(CONFIG, CONFIG_SNAPSHOT_PATH)

# %%
wandb_run = wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name=WANDB_RUN_NAME,
    job_type="bronze",
    config={
        "dataset_name_provisional": PROVISIONAL_DATASET_NAME,
        "dataset_resolution_stage": "pre_ingest",
        "bronze_version": BRONZE_VERSION,
        "raw_data_file": RAW_FILE_NAME,
        "raw_path": str(RAW_FILE_PATH / RAW_FILE_NAME),
        "split_status": SPLIT_STATUS,
        #"label_type": LABEL_TYPE,
        #"label_source": LABEL_SOURCE,
        "run_id": RUN_ID,
        "asset_id": ASSET_ID,
        
        "add_record_id": ADD_RECORD_ID,
        "record_id_inputs": RECORD_ID_INPUTS if ADD_RECORD_ID else None,
        "record_id_method": RECORD_ID_METHOD if ADD_RECORD_ID else None,
        
        "bronze_out_path": str(BRONZE_DATA_PATH),
    },
)

logger.info("W&B initialized: %s", wandb_run.name)

# %% [markdown]
# ----

# %%
# Load Data

dataframe = ingest_data(
    RAW_FILE_PATH,
    file_name=RAW_FILE_NAME,
    dataset_name=DATASET_NAME_ARGUMENT,
    dataset_name_config=DATASET_NAME_CONFIG,
    dataset_candidates=DATASET_CANDIDATES,
    split=SPLIT_STATUS,
    #label_type=LABEL_TYPE_DF,
    #label_source=LABEL_SOURCE_DF,
    run_id=RUN_ID,
    asset_id=ASSET_ID,
    add_record_id=True,
    validate=True,
)



# %% [markdown]
# ----

# %%
dataset_resolution = dataframe.attrs.get("dataset_resolution", {})

if not dataset_resolution:
    raise ValueError(
        "Bronze ingest did not write dataset_resolution metadata to dataframe.attrs."
    )


RESOLVED_DATASET_NAME = dataset_resolution.get("dataset_name")
DATASET_SOURCE_COLUMN = dataset_resolution.get("dataset_source_column")
DATASET_METHOD = dataset_resolution.get("dataset_method")

if RESOLVED_DATASET_NAME is None:
    raise ValueError("Bronze ingest did not return dataset_resolution metadata in dataframe.attrs.")

# %%
wandb_run.config.update(
    {
        "dataset_name_final": RESOLVED_DATASET_NAME,
        "dataset_source_column": DATASET_SOURCE_COLUMN,
        "dataset_method": DATASET_METHOD,
        "dataset_resolution_stage": "bronze_ingest_final",
    },
    allow_val_change=True,
)

# %%
if PROVISIONAL_DATASET_NAME != RESOLVED_DATASET_NAME:
    logger.info(
        "Dataset name changed after Bronze ingest when inside-dataset evidence became available | provisional=%s | resolved=%s",
        PROVISIONAL_DATASET_NAME,
        RESOLVED_DATASET_NAME,
    )

# %%
DATASET_NAME = RESOLVED_DATASET_NAME

# %%
bronze_truth = initialize_layer_truth(
    truth_version=TRUTH_VERSION,
    dataset_name=DATASET_NAME,
    layer_name=LAYER_NAME,
    process_run_id=PROCESS_RUN_ID,
    pipeline_mode=PIPELINE_MODE,
    parent_truth_hash=None,
)

bronze_truth = update_truth_section(
    bronze_truth,
    "config_snapshot",
    {
        "bronze_version": BRONZE_VERSION,
        "split_status": SPLIT_STATUS,
        "label_type": LABEL_TYPE,
        "label_source": LABEL_SOURCE,
        "run_id": RUN_ID,
        "asset_id": ASSET_ID,
        "add_record_id": ADD_RECORD_ID,
        "record_id_inputs": RECORD_ID_INPUTS if ADD_RECORD_ID else None,
        "record_id_method": RECORD_ID_METHOD if ADD_RECORD_ID else None,
        "pipeline_mode": PIPELINE_MODE,
    },
)

bronze_truth = update_truth_section(
    bronze_truth,
    "runtime_facts",
    {
        "source_run_id": RUN_ID,
        "raw_file_path": str(RAW_FILE_PATH / RAW_FILE_NAME),
        "raw_data_dir": str(RAW_DATA_PATH),
        "dataset_name_provisional": PROVISIONAL_DATASET_NAME,
        "dataset_name_final": RESOLVED_DATASET_NAME,
        "dataset_source_column": DATASET_SOURCE_COLUMN,
        "dataset_method": DATASET_METHOD,
    },
)

bronze_truth = update_truth_section(
    bronze_truth,
    "artifact_paths",
    {
        "bronze_output_dir": str(BRONZE_DATA_PATH),
        "bronze_output_file_name": "pump__bronze__train.parquet",
    },
)

bronze_truth = update_truth_section(
    bronze_truth,
    "notes",
    {
        "purpose": "Bronze ingestion truth record",
    },
)

# %% [markdown]
# ----

# %%
# Basic Dataframe Information/Summary

# Shape 
print("Shape:", dataframe.shape)
logger.info("Shape: %s", dataframe.shape)

# Dtypes as a compact block
print("\nData types:")
print(dataframe.dtypes)
logger.info("Dtypes:\n%s", dataframe.dtypes.to_string())

# Memory Usages
print("\nMemory usage (MB):")
print(dataframe.memory_usage(deep=True).sum() / (1024 ** 2))
mem_mb = dataframe.memory_usage(deep=True).sum() / (1024 ** 2)
logger.info("Memory usage (MB): %.2f", mem_mb)

# Head(15) as text (truncate columns for readability)
print("\nFirst 15 rows:")
display(dataframe.head(15))
logger.info("Head(15):\n%s", dataframe.head(15).to_string(max_cols=40, max_rows=15))

# Describe Numbers
print("\nBasic numeric summary:")
display(dataframe.describe(include=[np.number]).T)
desc_num = dataframe.describe(include=[np.number]).T
logger.info("Numeric describe (truncated):\n%s", desc_num.to_string(max_rows=60))

# Describe Objects and categorical
print("\nBasic object / categorical summary:")
object_category_columns = dataframe.select_dtypes(include=["object", "category", "string"]).columns
if len(object_category_columns):
    desc_obj = dataframe[object_category_columns].describe().T
    display(desc_obj)
    logger.info("Object/categorical describe (truncated):\n%s", desc_obj.to_string(max_rows=60))
else:
    logger.info("No object/category/string columns to describe.")

# Meta Columns
print("\nMeta Columns Summary:")
meta_columns = [column for column in dataframe.columns if column.startswith("meta__")]
dataframe[meta_columns].head(3)
logger.info("Meta Columns: (%d): %s", len(meta_columns), "\n".join(meta_columns))

# All Other Columns
print("\nAll Other Columns Summary:")
other_columns = [column for column in dataframe.columns if not column.startswith("meta__")]
dataframe[other_columns].head(3)
logger.info("All Other Columns: (%d): %s", len(other_columns), "\n".join(other_columns))

# Missing
missing = (dataframe[other_columns].isna().mean() * 100).sort_values(ascending=False).head(20)
display(missing.to_frame("pct_missing"))
logger.info("Top missingness (%%):\n%s", missing.to_string())


# %% [markdown]
# ----

# %%
bronze_source_fingerprint = build_file_fingerprint(RAW_FILE_PATH / RAW_FILE_NAME)
bronze_truth = update_truth_section(
    bronze_truth,
    "source_fingerprint",
    bronze_source_fingerprint,
)

bronze_meta_columns = sorted(
    set(
        identify_meta_columns(dataframe)
        + [
            "meta__truth_hash",
            "meta__parent_truth_hash",
            "meta__pipeline_mode",
        ]
    )
)

bronze_feature_columns = identify_feature_columns(dataframe)

bronze_truth_record = build_truth_record(
    truth_base=bronze_truth,
    row_count=len(dataframe),
    column_count=dataframe.shape[1] + 3,
    meta_columns=bronze_meta_columns,
    feature_columns=bronze_feature_columns,
)

BRONZE_TRUTH_HASH = bronze_truth_record["truth_hash"]

dataframe = stamp_truth_columns(
    dataframe,
    truth_hash=BRONZE_TRUTH_HASH,
    parent_truth_hash=None,
    pipeline_mode=PIPELINE_MODE,
)

bronze_truth_path = save_truth_record(
    bronze_truth_record,
    truth_dir=TRUTHS_PATH,
    dataset_name=DATASET_NAME,
    layer_name=LAYER_NAME,
)

append_truth_index(
    bronze_truth_record,
    truth_index_path=TRUTH_INDEX_PATH,
)

logger.info("Bronze truth hash: %s", BRONZE_TRUTH_HASH)
logger.info("Bronze truth path: %s", bronze_truth_path)
logger.info("Bronze process_run_id: %s", PROCESS_RUN_ID)

print("Bronze truth hash:", BRONZE_TRUTH_HASH)
print("Bronze truth path:", bronze_truth_path)
print("Bronze process_run_id:", PROCESS_RUN_ID)

# %% [markdown]
# ----

# %%
def collect_meta_columns(existing_columns: List[str]) -> List[str]:
    meta_columns: List[str] = []
    for column in existing_columns:
        if column.startswith("meta__"):
            meta_columns.append(column)
    return meta_columns



def reorder_bronze_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
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

# %%
# Reorder dataframe and put meta columns in the front. 

dataframe = reorder_bronze_columns(dataframe)

meta_columns_after_reorder = collect_meta_columns(list(dataframe.columns))

non_meta_columns_after_reorder = [
    column_name
    for column_name in dataframe.columns
    if not column_name.startswith("meta__")
]

logger.info(
    "Bronze columns reordered successfully. "
    "Meta columns moved to the front while preserving original within-group order. "
    "meta_column_count=%d | non_meta_column_count=%d | total_column_count=%d",
    len(meta_columns_after_reorder),
    len(non_meta_columns_after_reorder),
    dataframe.shape[1],
)


# %%
# Save Data as Parquet
save_data(dataframe, paths.data_bronze_train, BRONZE_TRAIN_DATA_FILE_NAME)

# %% [markdown]
# ----

# %%
metrics, saved = profile_dataframe(dataframe, logger, artifacts_dir=BRONZE_ARTIFACTS_PATH)

# %% [markdown]
# ----

# %%
finalize_wandb_stage(
    wandb_run,
    stage="bronze",
    dataframe=dataframe,
    project_root=paths.root,
    logs_dir=paths.logs,
    dataset_dirs=[paths.data_bronze_train],
    dataset_artifact_name="pump__bronze__train.parquet",
    notebook_path=paths.notebooks / "Preprocessing"/ "EDA_Notebook_Bronze_01_Preprocessing.ipynb",
)

# %% [markdown]
# ----

# %%
wandb_run.finish()

# %% [markdown]
# ----

# %%
'''

# Ensure artifacts subdir exists
bronze_art_dir = paths.artifacts / "bronze"
bronze_art_dir.mkdir(parents=True, exist_ok=True)

# Profile + export describe() CSVs into artifacts/bronze
metrics, saved = profile_dataframe(dataframe, logger, artifacts_dir=bronze_art_dir, head=15)

# Start or reuse a W&B run (you can also init at the top; either is fine)
run = wandb.init(project="capstone", job_type="bronze", config={"dataset": "pump", "stage": "bronze"})

# Log scalars + preview table near the end
log_metrics(run, metrics)
log_dataframe_head(run, dataframe, key="bronze_head15", n=15)

# Upload logs (just bronze.log)
log_files_as_artifact(
    run,
    artifact_name="capstone-logs-bronze",
    artifact_type="logs",
    files=[paths.logs / "bronze.log"],
    aliases=["latest"],
    metadata={"stage": "bronze", "dataset": "pump"},
)

# Upload bronze parquet outputs (train dir)
log_dir_as_artifact(
    run,
    artifact_name="pump-bronze-train-parquet",
    artifact_type="dataset",
    dir_path=paths.data_bronze_train,
    patterns=("*.parquet", "*.pq"),  # explicit
    aliases=["latest"],
    metadata={"stage": "bronze", "split": "train", "dataset": "pump"},
)

# Upload bronze diagnostics exports (artifacts/bronze)
log_dir_as_artifact(
    run,
    artifact_name="pump-bronze-diagnostics",
    artifact_type="eda",
    dir_path=bronze_art_dir,
    patterns=("*.csv", "*.json", "*.log", "*.txt"),
    aliases=["latest"],
    metadata={"stage": "bronze", "dataset": "pump"},
)

# Upload the notebook itself (point this at where you actually save it)
# Best practice: save/copy the notebook into paths.notebooks first.
bronze_nb = paths.notebooks / "Bronze_Preprocessing_Pump.ipynb"

if bronze_nb.exists():
    log_files_as_artifact(
        run,
        artifact_name="capstone-notebooks",
        artifact_type="notebook",
        files=[bronze_nb],
        aliases=["latest"],
        metadata={"stage": "bronze"},
    )
else:
    logger.warning("Notebook not found at %s; skipping upload.", bronze_nb)

run.finish()

'''


# %% [markdown]
# ----

# %%
required_bronze_meta_columns = [
    "meta__truth_hash",
    "meta__parent_truth_hash",
    "meta__pipeline_mode",
]

missing_bronze_meta_columns = [
    column_name
    for column_name in required_bronze_meta_columns
    if column_name not in dataframe.columns
]
if missing_bronze_meta_columns:
    raise ValueError(
        f"Bronze dataframe is missing required lineage columns: {missing_bronze_meta_columns}"
    )

bronze_dataframe_truth_hash = extract_truth_hash(dataframe)
if bronze_dataframe_truth_hash is None:
    raise ValueError("Bronze dataframe does not contain a readable meta__truth_hash value.")

if bronze_dataframe_truth_hash != BRONZE_TRUTH_HASH:
    raise ValueError(
        "Bronze dataframe truth hash does not match BRONZE_TRUTH_HASH:\n"
        f"dataframe={bronze_dataframe_truth_hash}\n"
        f"record={BRONZE_TRUTH_HASH}"
    )

if "meta__parent_truth_hash" in dataframe.columns:
    bronze_parent_values = dataframe["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()
    if bronze_parent_values:
        raise ValueError(
            "Bronze should not have a populated parent truth hash, but found values:\n"
            f"{bronze_parent_values}"
        )

if not Path(bronze_truth_path).exists():
    raise FileNotFoundError(f"Bronze truth file was not created: {bronze_truth_path}")

loaded_bronze_truth = load_json(bronze_truth_path)

if loaded_bronze_truth.get("truth_hash") != BRONZE_TRUTH_HASH:
    raise ValueError(
        "Saved Bronze truth file hash does not match BRONZE_TRUTH_HASH:\n"
        f"file={loaded_bronze_truth.get('truth_hash')}\n"
        f"record={BRONZE_TRUTH_HASH}"
    )

if loaded_bronze_truth.get("parent_truth_hash") is not None:
    raise ValueError(
        "Bronze truth file parent_truth_hash should be None.\n"
        f"Found: {loaded_bronze_truth.get('parent_truth_hash')}"
    )

if loaded_bronze_truth.get("row_count") != len(dataframe):
    raise ValueError(
        "Bronze truth row_count does not match dataframe row count:\n"
        f"truth={loaded_bronze_truth.get('row_count')}\n"
        f"dataframe={len(dataframe)}"
    )

if loaded_bronze_truth.get("column_count") != dataframe.shape[1]:
    raise ValueError(
        "Bronze truth column_count does not match stamped dataframe column count:\n"
        f"truth={loaded_bronze_truth.get('column_count')}\n"
        f"dataframe={dataframe.shape[1]}"
    )

print("Bronze lineage sanity check passed.")

# %% [markdown]
# ----

# %%
print("=== Silver Pre-EDA: Data Overview ===")
print(f"Shape: {dataframe.shape[0]:,} rows × {dataframe.shape[1]} columns\n")

print("=== Column Types ===")
display(dataframe.dtypes.to_frame("dtype").head(20))

print("\n=== df.info() ===")
dataframe.info()

print("\n=== Basic Descriptive Statistics (numeric only) ===")
display(dataframe.describe().T)

print("\n=== Get Top Sample Rows")
display(dataframe.head())


