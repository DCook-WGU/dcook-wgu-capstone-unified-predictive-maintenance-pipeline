# %% [markdown]
# ## Silver EDA (Deliverables 1.2.2 and 1.2.3)
# 
# This notebook performs data profiling, exploratory analysis, and feature behavior review on the Silver pump dataset. It focuses on comparing normal and abnormal operating behavior to understand how the sensors change around failure periods.
# 
# The key goals of this notebook are:
# 
# - To document data profiling and exploratory analysis for the Silver layer.
# - To analyze feature behavior across normal and abnormal operating patterns.
# - To generate supporting Silver-layer artifacts (tables, profiles, and charts) that help justify the later model design in the Gold layer.
# 
# Outputs from this notebook support the project write-up in Section C by:
# 
# - Providing evidence for the anomaly-screening approach described in C.2 and C.2.A.
# - Supplying behavior profiles and feature effect-size information that are used to design the Stage 3 rule/profile/historical confirmation layer described in C.2.
# - Providing visualizations that can be used in C.6 to communicate how sensor behavior differs between normal and abnormal operation.

# %%
print("hello")

# %%
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pathlib import Path
import yaml
import re

import logging
import wandb

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import seaborn as sns 


from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


import pyarrow.parquet as pq
import pyarrow as pa

import json 
import hashlib


# Custom Utilities Module
from utils.paths import get_paths
from utils.file_io import load_data, save_data, save_json, load_json
from utils.eda_logging import profile_dataframe
from utils.logging_setup import configure_logging, log_layer_paths
from utils.wandb_utils import finalize_wandb_stage

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
    get_truth_value,
    get_dataset_name_from_truth,
    get_truth_hash,
    get_parent_truth_hash,
    get_pipeline_mode_from_truth,
    get_artifact_path_from_truth,
)

from utils.pipeline_config_loader import (
    load_pipeline_config,
    build_truth_config_block,
    set_wandb_dir_from_config,
    export_config_snapshot,
)

# Ledger 
from utils.ledger import Ledger

# Show more columns
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)



# %% [markdown]
# ----

# %%
def z_score(series: pd.Series) -> pd.Series:
    """
    Z-score normalize a numeric pandas Series.
    Returns a Series with the same index.
    If std is 0 or NaN (constant series), returns series - mean.
    """
    # Ensure Series is a float type
    series = series.astype(float)

    # Use nan-safe stats in case any NaNs sneak in
    mean_value = np.nanmean(series.to_numpy())
    std_value = np.nanstd(series.to_numpy())

    
    if std_value == 0 or np.isnan(std_value):
        # Make all values (effectively) the same or center only
        return pd.Series(
            np.where(series.notna(), 0.0, np.nan),
            index=series.index
        )
    
    return (series - mean_value) / std_value



# %% [markdown]
# ----

# %%
paths = get_paths()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

CONFIG_ROOT = paths.configs

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

CONFIG_RUN_MODE = "train"
CONFIG_PROFILE = "default"

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

CONFIG = load_pipeline_config(
    config_root=CONFIG_ROOT,
    stage="silver_eda",
    dataset="pump",
    mode=CONFIG_RUN_MODE,
    profile=CONFIG_PROFILE,
    project_root=paths.root,
).data

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

SILVER_EDA_CFG = CONFIG["silver_eda"]
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

# ---- Stage details ----
STAGE = "silver_eda"
LAYER_NAME = SILVER_EDA_CFG["layer_name"]
SILVER_VERSION = CONFIG["versions"]["silver_eda"]
CLEANING_RECIPE_ID = SILVER_EDA_CFG["cleaning_recipe_id"]
TRUTH_VERSION = CONFIG["versions"]["truth"]

PIPELINE_MODE = PIPELINE["execution_mode"]
RUN_MODE = CONFIG["runtime"]["mode"]

DATASET_NAME_CONFIG = CONFIG["dataset"]["name"]
DATASET_NAME = None

SILVER_TRUTH_HASH = None
SILVER_PARENT_TRUTH_HASH = None
SILVER_TRUTH_PATH = None
SILVER_PARENT_TRUTH_PATH = None
SILVER_PARENT_LAYER_NAME = "silver"

LABEL_SOURCE_COLUMN = None
LABEL_SOURCE_TYPE = None
LABEL_SOURCE_INFO = {}
CANONICAL_INFO = {}
FEATURE_SET_INFO = {}
QUALITY_INFO = {}
NEEDS_ONE_HOT_ENCODING = None
ONE_HOT_ENCODING_COLUMNS = []

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

SILVER_PROCESS_RUN_ID = make_process_run_id(SILVER_EDA_CFG["process_run_id_prefix"])

# ---- W&B ----
WANDB_PROJECT = CONFIG["wandb"]["project"]
WANDB_ENTITY = CONFIG["wandb"]["entity"]
WANDB_RUN_NAME = f"{SILVER_VERSION}"

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

# Thresholds
MIN_TIME_PARSE_SUCCESS_PERCENT = float(SILVER_EDA_CFG["min_time_parse_success_percent"])
MIN_STEP_PARSE_SUCCESS_PERCENT = float(SILVER_EDA_CFG["min_step_parse_success_percent"])

QUARANTINE_MISSING_PCT = float(SILVER_EDA_CFG["quarantine_missing_pct"])
CORRELATION_THRESHOLD = float(SILVER_EDA_CFG["correlation_threshold"])

USE_ROBUST_SCALER = bool(SILVER_EDA_CFG["use_robust_scaler"])

TOP_N_SENSORS_FOR_PLOTS = int(SILVER_EDA_CFG["top_n_sensors_for_plots"])
PAIRPLOT_SENSOR_CAP = int(SILVER_EDA_CFG["pairplot_sensor_cap"])
PAIRPLOT_SAMPLE_N = int(SILVER_EDA_CFG["pairplot_sample_n"])
TOP_PLOT_COLS = int(SILVER_EDA_CFG["top_plot_cols"])
TOP_CORR_COLS = int(SILVER_EDA_CFG["top_corr_cols"])

ROLLING_MINUTES = int(SILVER_EDA_CFG["rolling_minutes"])
LOOKBACK_HOURS = int(SILVER_EDA_CFG["lookback_hours"])
BASELINE_DAYS = int(SILVER_EDA_CFG["baseline_days"])
BASELINE_GAP_HOURS = int(SILVER_EDA_CFG["baseline_gap_hours"])
SUSTAIN_MINUTES = int(SILVER_EDA_CFG["sustain_minutes"])
TOP_SENSOR_PRE_HOURS = int(SILVER_EDA_CFG["top_sensor_pre_hours"])

PRE_WINDOW_STEPS = int(SILVER_EDA_CFG["pre_window_steps"])
POST_WINDOW_STEPS = int(SILVER_EDA_CFG["post_window_steps"])
MAX_ONSETS_TO_USE = int(SILVER_EDA_CFG["max_onsets_to_use"])
PCA_SAMPLE_ROW_COUNT = int(SILVER_EDA_CFG["pca_sample_row_count"])
IMPUTE_SAMPLE_ROW_COUNT = int(SILVER_EDA_CFG["impute_sample_row_count"])

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

# ---- File names ----
BRONZE_TRAIN_DATA_FILE_NAME = FILENAMES["bronze_train_file_name"]
SILVER_TRAIN_DATA_FILE_NAME = FILENAMES["silver_train_file_name"]

FEATURE_REGISTRY_FILE_NAME = None
FEATURE_REGISTRY_PATH = None

# ---- Paths setup ----
BRONZE_TRAIN_DATA_PATH = Path(PATHS["data_bronze_train_dir"])
SILVER_TRAIN_DATA_PATH = Path(PATHS["data_silver_train_dir"])

SILVER_ARTIFACTS_PATH = Path(PATHS["artifacts_root"]) / "silver"
SILVER_EDA_ARTIFACTS_ROOT = Path(PATHS["artifacts_root"]) / "silver_eda"
SILVER_EDA_ARTIFACTS_PATH = None

ARTIFACTS_PATH = Path(PATHS["artifacts_root"])

TRUTHS_PATH = Path(PATHS["truths_dir"])
TRUTH_INDEX_PATH = Path(PATHS["truth_index_path"])

LOGS_PATH = Path(PATHS["logs_root"])

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

required_columns = [
    "anomaly_flag",
    "event_step",
    "time_index",
    "meta__asset_id",
    "meta__run_id",
]

set_wandb_dir_from_config(CONFIG)

SILVER_TRAIN_DATA_PATH.mkdir(parents=True, exist_ok=True)
SILVER_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
SILVER_EDA_ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
TRUTHS_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ----

# %% [markdown]
# ----

# %%
# Logging Setup

# Create silver log path 
silver_log_path = paths.logs / "silver_eda.log"

# Initial Logger
configure_logging(
    "capstone",
    silver_log_path,
    level=logging.DEBUG,
    overwrite_handlers=True,
)

# Initiate Logger and log file
logger = logging.getLogger("capstone.silver_eda")

# Log load and initiation
logger.info("Silver EDA stage starting")

# Log paths loads
log_layer_paths(paths, current_layer="silver", logger=logger)


# %% [markdown]
# ----

# %%
# W&B

wandb_run = wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name=WANDB_RUN_NAME,
    job_type="silver_eda",
    config={
        "silver_version": SILVER_VERSION,
        "cleaning_recipe_id": CLEANING_RECIPE_ID,
        "quarantine_missing_pct": QUARANTINE_MISSING_PCT,
        "min_time_parse_success_percent": MIN_TIME_PARSE_SUCCESS_PERCENT,
        "rolling_window": ROLLING_MINUTES,
        "silver_path": str(SILVER_TRAIN_DATA_PATH / SILVER_TRAIN_DATA_FILE_NAME),
        "silver_out_dir": str(SILVER_TRAIN_DATA_PATH),
    },
)
logger.info("W&B initialized: %s", wandb.run.name)


# %% [markdown]
# ----

# %%
# Ledger Setup

ledger = Ledger(stage=STAGE, recipe_id=CLEANING_RECIPE_ID)

ledger.add(
    kind="step",
    step="init",
    message="Initialized ledger",
    logger=logger
)


# %% [markdown]
# ----

# %%

# Load Data

preferred_silver = SILVER_TRAIN_DATA_PATH / SILVER_TRAIN_DATA_FILE_NAME

if preferred_silver.exists():
    silver_data_path = preferred_silver
else:
    parquet_files = sorted(SILVER_TRAIN_DATA_PATH.glob("*.parquet"))
    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No parquet files found in {SILVER_TRAIN_DATA_PATH}")
    if len(parquet_files) > 1: 
        logger.warning("Multiple Parquet Files found; Using First %s", parquet_files[0])
    silver_data_path = parquet_files[0]

if not silver_data_path.exists():
    raise FileNotFoundError(f"Silver parquet not found: {silver_data_path}")
    
dataframe = load_data(silver_data_path.parent, silver_data_path.name)



#### #### #### #### #### #### #### #### 

logger.info("Loaded Silver: %s | shape=%s", silver_data_path, dataframe.shape)
wandb_run.log({"silver_rows": int(dataframe.shape[0]), "silver_cols": int(dataframe.shape[1])})

ledger.add(
    kind="step",
    step="load_silver",
    message="Loaded Silver Parquet",
    why="Silver must be derived from reprodicible Silver Artifact",
    consequence="All silver outputs trace back to this file",
    data={"silver_path": str(silver_data_path), "shape": list(dataframe.shape), "cols": len(dataframe.columns)},
    logger=logger
)


#### #### #### #### #### #### #### #### 

display(dataframe.head(3))

# %% [markdown]
# ----

# %%
print("Silver training shape:", dataframe.shape)
dataframe.info()
display(dataframe.describe().T.head(10))

# %% [markdown]
# ----

# %%
SILVER_TRUTH_HASH = extract_truth_hash(dataframe)

if SILVER_TRUTH_HASH is None:
    raise ValueError("Could not resolve meta__truth_hash from Silver dataframe.")

SILVER_DATASET_NAME = (
    dataframe["meta__dataset"]
    .dropna()
    .astype("string")
    .str.strip()
)
SILVER_DATASET_NAME = SILVER_DATASET_NAME[SILVER_DATASET_NAME != ""]

if len(SILVER_DATASET_NAME) == 0:
    raise ValueError("Silver dataframe is missing usable meta__dataset values.")

SILVER_DATASET_NAME = str(SILVER_DATASET_NAME.iloc[0]).strip()

silver_truth = load_parent_truth_record_from_dataframe(
    dataframe=dataframe,
    truth_dir=TRUTHS_PATH,
    parent_layer_name="silver",
    dataset_name=SILVER_DATASET_NAME,
    column_name="meta__truth_hash",
)

DATASET_NAME = get_dataset_name_from_truth(silver_truth)
SILVER_TRUTH_HASH = get_truth_hash(silver_truth)
SILVER_PARENT_TRUTH_HASH = get_parent_truth_hash(silver_truth)

PIPELINE_MODE_FROM_TRUTH = get_pipeline_mode_from_truth(silver_truth)
if PIPELINE_MODE_FROM_TRUTH is not None:
    PIPELINE_MODE = PIPELINE_MODE_FROM_TRUTH

SILVER_TRUTH_PATH = (
    TRUTHS_PATH
    / "silver"
    / f"{DATASET_NAME}__silver__truth__{SILVER_TRUTH_HASH}.json"
)

LABEL_SOURCE_COLUMN = get_truth_value(
    silver_truth,
    "runtime_facts",
    "label_resolution",
    "label_source_column",
    required=False,
)

LABEL_SOURCE_TYPE = get_truth_value(
    silver_truth,
    "runtime_facts",
    "label_resolution",
    "label_source_type",
    required=False,
)

CANONICAL_INFO = silver_truth.get("runtime_facts", {}).get("canonical_info", {})
FEATURE_SET_INFO = silver_truth.get("runtime_facts", {}).get("feature_set", {})
QUALITY_INFO = silver_truth.get("runtime_facts", {}).get("quality_info", {})

NEEDS_ONE_HOT_ENCODING = bool(
    silver_truth.get("needs_one_hot_encoding", False)
)
ONE_HOT_ENCODING_COLUMNS = list(
    silver_truth.get("one_hot_encoding_columns", [])
)

feature_registry_dir = get_artifact_path_from_truth(
    silver_truth,
    "feature_registry_dir",
)
FEATURE_REGISTRY_FILE_NAME = f"{DATASET_NAME}__silver__feature_registry.json"
FEATURE_REGISTRY_PATH = Path(feature_registry_dir) / FEATURE_REGISTRY_FILE_NAME

SILVER_EDA_ARTIFACTS_PATH = SILVER_EDA_ARTIFACTS_ROOT / DATASET_NAME
SILVER_EDA_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

#### #### #### #### #### #### #### #### 

logger.info("Loaded Silver truth: %s", SILVER_TRUTH_PATH)
logger.info("Resolved Silver EDA dataset name from Silver truth: %s", DATASET_NAME)
logger.info("Resolved label source column from Silver truth: %s", LABEL_SOURCE_COLUMN)

print("Loaded Silver truth:", SILVER_TRUTH_PATH)
print("Silver truth hash:", SILVER_TRUTH_HASH)
print("Resolved Silver EDA dataset name:", DATASET_NAME)
print("Resolved label source column:", LABEL_SOURCE_COLUMN)

#### #### #### #### #### #### #### #### 

# %% [markdown]
# ----

# %% [markdown]
# ----

# %%
if FEATURE_REGISTRY_PATH is None:
    raise ValueError("FEATURE_REGISTRY_PATH was not resolved from Silver truth before loading the feature registry.")

feature_registry = load_json(FEATURE_REGISTRY_PATH.parent, FEATURE_REGISTRY_PATH.name)

FEATURE_COLUMNS = feature_registry.get("feature_columns", [])


#### #### #### #### #### #### #### #### 

logger.info("Loaded Silver Feature Registry: %s", FEATURE_REGISTRY_PATH)
wandb_run.log({"feature_registry_keys": int(len(feature_registry))})

ledger.add(
    kind="step",
    step="load_silver_feature_registry",
    message="Loaded Silver Feature Registry JSON file using the path resolved from Silver truth.",
    why="Silver EDA should inherit resolved feature metadata from Silver Pre-EDA rather than rebuilding it from config.",
    consequence="Silver EDA uses the same resolved feature set and feature registry lineage as Silver Pre-EDA.",
    data={
        "feature_registry_path": str(FEATURE_REGISTRY_PATH),
        "keys": int(len(feature_registry)),
        "feature_count": int(len(FEATURE_COLUMNS)),
    },
    logger=logger,
)

#### #### #### #### #### #### #### #### 

# %% [markdown]
# ----

# %%
present = {}
for column_name in required_columns:
    present[column_name] = bool(column_name in dataframe.columns)

overview = {
    "rows": int(len(dataframe)),
    "cols": int(len(dataframe.columns)),
    "required_present": present,
}

logger.info("Overview: %s", overview)

save_json(
    overview,
    file_path=SILVER_EDA_ARTIFACTS_PATH,
    file_name="silver_eda__overview.json",
)

wandb.save(str(SILVER_EDA_ARTIFACTS_PATH / "silver_eda__overview.json"))

# %% [markdown]
# ----

# %%
total_rows = int(len(dataframe))

null_rows = []
for column_name in dataframe.columns:
    null_count = int(dataframe[column_name].isna().sum())
    null_percent = float((null_count / total_rows) * 100.0) if total_rows > 0 else 0.0
    null_rows.append({
        "column": column_name,
        "null_count": null_count,
        "null_percent": null_percent,
    })

null_table = pd.DataFrame(null_rows).sort_values("null_percent", ascending=False)

null_all_path = SILVER_EDA_ARTIFACTS_PATH / "nulls__all_columns.csv"
null_table.to_csv(null_all_path, index=False)
wandb.save(str(null_all_path))

feature_null_table = null_table[null_table["column"].isin(FEATURE_COLUMNS)].copy()
feature_null_path = SILVER_EDA_ARTIFACTS_PATH / "nulls__feature_columns.csv"
feature_null_table.to_csv(feature_null_path, index=False)
wandb.save(str(feature_null_path))

null_table.head(15), feature_null_table.head(15)

# %% [markdown]
# ----

# %%
duplicate_row_count = int(dataframe.duplicated().sum())

duplicate_event_id_count = None
if "meta__event_id" in dataframe.columns:
    duplicate_event_id_count = int(dataframe["meta__event_id"].duplicated().sum())

dup_info = {
    "duplicate_row_count": duplicate_row_count,
    "duplicate_meta__event_id_count": duplicate_event_id_count,
}

save_json(dup_info, file_path=SILVER_EDA_ARTIFACTS_PATH, file_name="duplicates__summary.json")
wandb.save(str(SILVER_EDA_ARTIFACTS_PATH / "duplicates__summary.json"))

dup_info

# %% [markdown]
# ----

# %%
numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
numeric_describe = dataframe[numeric_columns].describe().T if len(numeric_columns) > 0 else pd.DataFrame()

numeric_stats_path = SILVER_EDA_ARTIFACTS_PATH / "column_stats__numeric_describe.csv"
numeric_describe.to_csv(numeric_stats_path)
wandb.save(str(numeric_stats_path))

categorical_like_columns = dataframe.select_dtypes(include=["object", "category", "string"]).columns.tolist()

cardinality_rows = []
for column_name in categorical_like_columns:
    unique_count = int(dataframe[column_name].nunique(dropna=True))
    non_null_count = int(dataframe[column_name].notna().sum())
    cardinality_rows.append({
        "column": column_name,
        "unique_count": unique_count,
        "non_null_count": non_null_count,
    })

cardinality_table = pd.DataFrame(cardinality_rows).sort_values("unique_count", ascending=False)

cardinality_path = SILVER_EDA_ARTIFACTS_PATH / "column_stats__categorical_cardinality.csv"
cardinality_table.to_csv(cardinality_path, index=False)
wandb.save(str(cardinality_path))

numeric_describe.head(10), cardinality_table.head(15)

# %% [markdown]
# ----

# %%
if "anomaly_flag" in dataframe.columns:
    normal_dataframe = dataframe[dataframe["anomaly_flag"] == 0].copy()
else:
    normal_dataframe = dataframe.copy()

profile_rows = []
for column_name in FEATURE_COLUMNS:
    if column_name not in normal_dataframe.columns:
        continue
    if not pd.api.types.is_numeric_dtype(normal_dataframe[column_name]):
        continue

    series = normal_dataframe[column_name].dropna()
    if len(series) == 0:
        continue

    profile_rows.append({
        "feature": column_name,
        "count": int(series.shape[0]),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if series.shape[0] > 1 else 0.0,
        "min": float(series.min()),
        "p01": float(series.quantile(0.01)),
        "p05": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    })

normal_profile_table = pd.DataFrame(profile_rows)

if normal_profile_table.empty:
    print("No numeric features produced a profile for NORMAL (after filters).")
else:

    normal_profile_table = pd.DataFrame(profile_rows).sort_values("std", ascending=False)

    normal_profile_path = SILVER_EDA_ARTIFACTS_PATH / "feature_profile__normal.csv"
    normal_profile_table.to_csv(normal_profile_path, index=False)
    wandb.save(str(normal_profile_path))

normal_profile_table.head(15)

# %% [markdown]
# ----

# %%
if "anomaly_flag" in dataframe.columns:
    abnormal_dataframe = dataframe[dataframe["anomaly_flag"] == 1].copy()
else:
    abnormal_dataframe = dataframe.iloc[0:0].copy()

profile_rows = []
for column_name in FEATURE_COLUMNS:
    if column_name not in abnormal_dataframe.columns:
        continue
    if not pd.api.types.is_numeric_dtype(abnormal_dataframe[column_name]):
        continue

    series = abnormal_dataframe[column_name].dropna()
    if len(series) == 0:
        continue

    profile_rows.append({
        "feature": column_name,
        "count": int(series.shape[0]),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if series.shape[0] > 1 else 0.0,
        "min": float(series.min()),
        "p01": float(series.quantile(0.01)),
        "p05": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    })

abnormal_profile_table = pd.DataFrame(profile_rows)

if abnormal_profile_table.empty:
    print("No numeric features produced a profile for ABNROMAL (after filters).")
else:

    abnormal_profile_table = pd.DataFrame(profile_rows).sort_values("std", ascending=False)

    abnormal_profile_path = SILVER_EDA_ARTIFACTS_PATH / "feature_profile__abnormal.csv"
    abnormal_profile_table.to_csv(abnormal_profile_path, index=False)
    wandb.save(str(abnormal_profile_path))

abnormal_profile_table.head(15)

# %% [markdown]
# ----

# %%
if "anomaly_flag" in dataframe.columns:
    recovering_dataframe = dataframe[dataframe["anomaly_flag"] == 1].copy()
else:
    recovering_dataframe = dataframe.copy()


recovering_dataframe = recovering_dataframe[recovering_dataframe['machine_status'] == "RECOVERING"].copy()

profile_rows = []
for column_name in FEATURE_COLUMNS:
    if column_name not in recovering_dataframe.columns:
        continue
    if not pd.api.types.is_numeric_dtype(recovering_dataframe[column_name]):
        continue

    series = recovering_dataframe[column_name].dropna()
    if len(series) == 0:
        continue

    profile_rows.append({
        "feature": column_name,
        "count": int(series.shape[0]),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if series.shape[0] > 1 else 0.0,
        "min": float(series.min()),
        "p01": float(series.quantile(0.01)),
        "p05": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    })

recovering_profile_table = pd.DataFrame(profile_rows)

if recovering_profile_table.empty:
    print("No numeric features produced a profile for RECOVERING (after filters).")
else:

    recovering_profile_table = recovering_profile_table.sort_values("std", ascending=False)

    recovering_profile_path = SILVER_EDA_ARTIFACTS_PATH / "feature_profile__recovering.csv"
    recovering_profile_table.to_csv(recovering_profile_path, index=False)
    wandb.save(str(normal_profile_path))

display(recovering_profile_table.head(15))

# %% [markdown]
# ----

# %%
dataframe.head()


# %%
truth_state_col = None
FALLBACK_STATE_COL = "machine_status"

label_resolution = silver_truth.get("runtime_facts", {}).get("label_resolution", {})
truth_state_col = label_resolution.get("label_source_column")

if truth_state_col is not None and str(truth_state_col).strip():
    state_col = str(truth_state_col)
else:
    state_col = FALLBACK_STATE_COL

if state_col not in dataframe.columns:
    raise KeyError(
        f"Resolved state_col='{state_col}' not found in dataframe columns."
    )

logger.info("Resolved state_col from Silver truth: %s", state_col)
print("Resolved state_col:", state_col)


# Build the list of states (values)
state_list = sorted(map(str, dataframe[state_col].dropna().unique()))

# Profile loop
for state in state_list:
    working_dataframe = dataframe.loc[dataframe[state_col].astype(str) == state].copy()

    profile_rows = []
    for column_name in FEATURE_COLUMNS:
        if column_name not in working_dataframe.columns:
            continue
        if not pd.api.types.is_numeric_dtype(working_dataframe[column_name]):
            continue

        series = working_dataframe[column_name].dropna()
        if series.empty:
            continue

        profile_rows.append({
            "feature": column_name,
            "count": int(series.shape[0]),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)) if series.shape[0] > 1 else 0.0,
            "min": float(series.min()),
            "p01": float(series.quantile(0.01)),
            "p05": float(series.quantile(0.05)),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99)),
            "max": float(series.max()),
        })

    working_profile_table = pd.DataFrame(profile_rows)
    
    if working_profile_table.empty:
        print(f"No numeric features produced a profile for {state} (after filters).")
        continue

    working_profile_table = working_profile_table.sort_values("std", ascending=False)

    safe_state = re.sub(r"[^A-Za-z0-9._-]+", "_", str(state)).strip("_")
    out_name = f"{DATASET_NAME}__{LAYER_NAME}__feature_profile__{safe_state}_loop.csv"
    out_path = SILVER_EDA_ARTIFACTS_PATH / out_name

    working_profile_table.to_csv(out_path, index=False)

    wandb.log({f"profile_preview/{safe_state}": wandb.Table(dataframe=working_profile_table.head(50))})

    display(working_profile_table.head(15))

# %% [markdown]
# ----

# %%
comparison_rows = []

normal_lookup = normal_profile_table.set_index("feature") if "feature" in normal_profile_table.columns else pd.DataFrame()
abnormal_lookup = abnormal_profile_table.set_index("feature") if "feature" in abnormal_profile_table.columns else pd.DataFrame()

for feature_name in FEATURE_COLUMNS:
    if feature_name not in normal_lookup.index:
        continue
    if feature_name not in abnormal_lookup.index:
        continue

    normal_mean = float(normal_lookup.loc[feature_name, "mean"])
    normal_std = float(normal_lookup.loc[feature_name, "std"])
    abnormal_mean = float(abnormal_lookup.loc[feature_name, "mean"])

    if normal_std == 0.0:
        effect_size = np.nan
    else:
        effect_size = (abnormal_mean - normal_mean) / normal_std

    comparison_rows.append({
        "feature": feature_name,
        "normal_mean": normal_mean,
        "normal_std": normal_std,
        "abnormal_mean": abnormal_mean,
        "effect_size": float(effect_size) if pd.notna(effect_size) else np.nan,
        "direction": "up" if pd.notna(effect_size) and effect_size > 0 else ("down" if pd.notna(effect_size) and effect_size < 0 else "flat/unknown"),
    })

effect_table = pd.DataFrame(comparison_rows)
effect_table["abs_effect_size"] = effect_table["effect_size"].abs()
effect_table = effect_table.sort_values("abs_effect_size", ascending=False)

effect_path = SILVER_EDA_ARTIFACTS_PATH / "feature_behavior__effect_size.csv"
effect_table.to_csv(effect_path, index=False)
wandb.save(str(effect_path))

TOP_FEATURES = effect_table["feature"].head(25).tolist()
TOP_FEATURES, effect_table.head(25)

# %% [markdown]
# ----

# %%
comparison_rows = []

normal_lookup = normal_profile_table.set_index("feature") if "feature" in normal_profile_table.columns else pd.DataFrame()
abnormal_lookup = abnormal_profile_table.set_index("feature") if "feature" in abnormal_profile_table.columns else pd.DataFrame()

for feature_name in FEATURE_COLUMNS:
    if feature_name not in normal_lookup.index:
        continue
    if feature_name not in abnormal_lookup.index:
        continue

    normal_mean = float(normal_lookup.loc[feature_name, "mean"])
    normal_std = float(normal_lookup.loc[feature_name, "std"])
    abnormal_mean = float(abnormal_lookup.loc[feature_name, "mean"])

    if normal_std == 0.0:
        effect_size = np.nan
    else:
        effect_size = (abnormal_mean - normal_mean) / normal_std

    comparison_rows.append({
        "feature": feature_name,
        "normal_mean": normal_mean,
        "normal_std": normal_std,
        "abnormal_mean": abnormal_mean,
        "effect_size": float(effect_size) if pd.notna(effect_size) else np.nan,
        "direction": "up" if pd.notna(effect_size) and effect_size > 0 else ("down" if pd.notna(effect_size) and effect_size < 0 else "flat/unknown"),
    })

effect_table = pd.DataFrame(comparison_rows)
effect_table["abs_effect_size"] = effect_table["effect_size"].abs()
effect_table = effect_table.sort_values("abs_effect_size", ascending=False)

effect_path = SILVER_EDA_ARTIFACTS_PATH / "feature_behavior__effect_size.csv"
effect_table.to_csv(effect_path, index=False)
wandb.save(str(effect_path))

TOP_FEATURES = effect_table["feature"].head(25).tolist()
TOP_FEATURES, effect_table.head(25)

# %% [markdown]
# ----

# %%
CORRELATION_FEATURES = TOP_FEATURES[:]

correlation_dataframe = normal_dataframe[CORRELATION_FEATURES].copy()
correlation_matrix = correlation_dataframe.corr(method="pearson")

corr_path = SILVER_EDA_ARTIFACTS_PATH / "correlation__normal.csv"
correlation_matrix.to_csv(corr_path)
wandb.save(str(corr_path))

# Heatmap
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix.values, aspect="auto")
plt.title("Correlation heatmap (NORMAL) — selected features")
plt.colorbar()
plt.xticks(range(len(CORRELATION_FEATURES)), CORRELATION_FEATURES, rotation=90, fontsize=7)
plt.yticks(range(len(CORRELATION_FEATURES)), CORRELATION_FEATURES, fontsize=7)
plt.tight_layout()

heatmap_path = SILVER_EDA_ARTIFACTS_PATH / "correlation__normal_heatmap.png"
plt.savefig(heatmap_path, dpi=200)
plt.show()

wandb.save(str(heatmap_path))

correlation_matrix.round(3)

# %% [markdown]
# ----

# %%
plot_features = TOP_FEATURES[:] if len(TOP_FEATURES) > 0 else FEATURE_COLUMNS[:]

for feature_name in plot_features:
    if feature_name not in dataframe.columns:
        continue
    if not pd.api.types.is_numeric_dtype(dataframe[feature_name]):
        continue

    plt.figure(figsize=(10, 4))

    series_normal = normal_dataframe[feature_name].dropna()
    series_abnormal = abnormal_dataframe[feature_name].dropna()

    # Use hist with alpha via default; do not set colors explicitly
    plt.hist(series_normal.values, bins=50, alpha=0.5, label="Normal")
    plt.hist(series_abnormal.values, bins=50, alpha=0.5, label="Abnormal")

    plt.title(f"Distribution: {feature_name} (Normal vs Abnormal)")
    plt.legend()
    plt.tight_layout()

    out_path = SILVER_EDA_ARTIFACTS_PATH / f"distribution__{feature_name}.png"
    plt.savefig(out_path, dpi=200)
    plt.show()

    wandb.save(str(out_path))

# %% [markdown]
# ----

# %%
plot_features = TOP_FEATURES[:] if len(TOP_FEATURES) > 0 else FEATURE_COLUMNS[:]

plt.figure(figsize=(12, 6))

x_axis = dataframe["time_index"] if "time_index" in dataframe.columns else np.arange(len(dataframe))

for feature_name in plot_features:
    if feature_name not in dataframe.columns:
        continue
    if not pd.api.types.is_numeric_dtype(dataframe[feature_name]):
        continue
    series = dataframe[feature_name].copy()
    series = series.fillna(series.median())
    plt.plot(x_axis, z_score(series), label=feature_name)

# Mark anomalies if present
if "anomaly_flag" in dataframe.columns:
    anomaly_positions = dataframe.index[dataframe["anomaly_flag"] == 1].tolist()
    if len(anomaly_positions) > 0:
        # draw sparse markers to avoid heavy rendering
        marker_positions = anomaly_positions[::max(1, len(anomaly_positions)//200)]
        plt.scatter(x_axis.iloc[marker_positions] if hasattr(x_axis, "iloc") else np.array(x_axis)[marker_positions],
                    np.zeros(len(marker_positions)),
                    marker="x",
                    label="anomaly_flag=1 (markers at y=0)")

plt.title("Top feature overlay (z-scored)")
plt.legend(fontsize=7)
plt.tight_layout()

overlay_path = SILVER_EDA_ARTIFACTS_PATH / "timeseries__top_features_overlay.png"
plt.savefig(overlay_path, dpi=200)
plt.show()

wandb.save(str(overlay_path))

# %% [markdown]
# ----

# %%
plot_features = TOP_FEATURES[:] if len(TOP_FEATURES) > 0 else FEATURE_COLUMNS[:]

if "machine_status" in dataframe.columns:
    broken_mask = dataframe["machine_status"].eq("BROKEN")
elif "anomaly_flag" in dataframe.columns:
    broken_mask = dataframe["anomaly_flag"].eq(1)
else:
    broken_mask = pd.Series(False, index=dataframe.index)   

x_axis = dataframe["time_index"] if "time_index" in dataframe.columns else np.arange(len(dataframe))

broken_positions = np.flatnonzero(broken_mask.to_numpy())

for feature_name in plot_features:
    if feature_name not in dataframe.columns:
        continue
    if not pd.api.types.is_numeric_dtype(dataframe[feature_name]):
        continue

    series = dataframe[feature_name].copy()
    median_value = series.median()
    if pd.isna(median_value):
        median_value = 0
    series = series.fillna(median_value)

    # For Normalized Values, Use this:
    # For Raw Values, Comment This and Uncomment Below
    plot_series = z_score(series)

    # For Raw Values, use this:
    # For Normalize, Comment This and Uncomment Above
    # plot_series = series.to_numpy()

    plt.figure(figsize=(18, 3))
    plt.plot(x_axis, plot_series, color="blue", label=feature_name)

    if len(broken_positions) > 0:
        step = max(1, len(broken_positions) // 200)
        marker_positions = broken_positions[::step]

        x_marker_values = (
            x_axis.iloc[marker_positions]
            if hasattr(x_axis, "iloc")
            else np.asarray(x_axis)[marker_positions]
        )
        y_marker_values = np.asarray(plot_series)[marker_positions]

        plt.scatter(
            x_marker_values,
            y_marker_values,
            color="red",
            marker="x",
            s=60,
            label="BROKEN"
        )

    plt.title(f"Time Series - {feature_name} (Broken Marked)")
    plt.legend(fontsize=7)
    plt.tight_layout()

    feature_plot_path = SILVER_EDA_ARTIFACTS_PATH / f"timeseries__{feature_name}__broken_marked.png"
    plt.savefig(feature_plot_path, dpi=200)
    plt.show()
    plt.close()

    wandb.save(str(feature_plot_path))
    

# %% [markdown]
# ----

# %%
# Use NORMAL correlation for clustering

ALL_FEATURES = [column for column in FEATURE_COLUMNS if column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[column])]

cluster_features = ALL_FEATURES[:]  # all numeric features
cluster_dataframe = normal_dataframe[cluster_features].copy()

# Fill missing with median so correlation works
for feature_name in cluster_features:
    cluster_dataframe[feature_name] = cluster_dataframe[feature_name].fillna(cluster_dataframe[feature_name].median())

correlation_matrix = cluster_dataframe.corr(method="pearson").fillna(0.0)

# Convert correlation to distance (0 = identical, 1 = unrelated)
distance_matrix = 1.0 - correlation_matrix.abs()

# Choose a small number of clusters to start (you can tune)
cluster_count = 6 if len(cluster_features) >= 6 else max(2, len(cluster_features))

model = AgglomerativeClustering(
    n_clusters=cluster_count,
    metric="precomputed",
    linkage="average",
)

labels = model.fit_predict(distance_matrix.values)

cluster_rows = []
for feature_name, cluster_label in zip(cluster_features, labels):
    cluster_rows.append({
        "feature": feature_name,
        "cluster_id": int(cluster_label),
    })

cluster_table = pd.DataFrame(cluster_rows).sort_values(["cluster_id", "feature"])

cluster_path = SILVER_EDA_ARTIFACTS_PATH / "clusters__correlation_agglomerative.csv"
cluster_table.to_csv(cluster_path, index=False)
wandb.save(str(cluster_path))

cluster_table.head(30)

# %% [markdown]
# ----

# %%
def find_anomaly_onsets(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "anomaly_flag" not in dataframe.columns:
        return pd.DataFrame(columns=["meta__asset_id", "meta__run_id", "time_index", "event_step"])

    grouping_columns = []
    if "meta__asset_id" in dataframe.columns:
        grouping_columns.append("meta__asset_id")
    if "meta__run_id" in dataframe.columns:
        grouping_columns.append("meta__run_id")

    working = dataframe.copy()

    if "event_step" not in working.columns and "time_index" in working.columns:
        working["event_step"] = working["time_index"]

    if "time_index" not in working.columns:
        working["time_index"] = np.arange(len(working), dtype=np.int64)

    # Ensure sorted within group for consistent onset detection
    if len(grouping_columns) > 0:
        working = working.sort_values(grouping_columns + ["event_step"]).reset_index(drop=True)
        shifted = working.groupby(grouping_columns, dropna=False)["anomaly_flag"].shift(1)
    else:
        working = working.sort_values(["event_step"]).reset_index(drop=True)
        shifted = working["anomaly_flag"].shift(1)

    onset_mask = (working["anomaly_flag"] == 1) & (shifted.fillna(0) == 0)
    onsets = working.loc[onset_mask, grouping_columns + ["time_index", "event_step"]].copy()
    return onsets.reset_index(drop=True)

def sample_onsets_evenly(onsets: pd.DataFrame, max_count: int) -> pd.DataFrame:
    if len(onsets) <= max_count:
        return onsets
    indices = np.linspace(0, len(onsets) - 1, num=max_count)
    indices = [int(round(value)) for value in indices]
    indices = sorted(list(set(indices)))
    return onsets.iloc[indices].reset_index(drop=True)

onsets_table = find_anomaly_onsets(dataframe)
onsets_table = sample_onsets_evenly(onsets_table, MAX_ONSETS_TO_USE)

onsets_path = SILVER_EDA_ARTIFACTS_PATH / "anomaly_onsets__table.csv"
onsets_table.to_csv(onsets_path, index=False)
wandb.save(str(onsets_path))

logger.info("Anomaly onsets found: %d", len(onsets_table))

# Choose features to align (use TOP_FEATURES if you already computed them; otherwise fallback)
aligned_features = []
for feature_name in TOP_FEATURES if "TOP_FEATURES" in globals() else FEATURE_COLUMNS:
    if feature_name in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[feature_name]):
        aligned_features.append(feature_name)
    if len(aligned_features) >= 6:
        break

if len(aligned_features) == 0 or len(onsets_table) == 0:
    logger.info("Skipping anomaly-onset alignment (no onsets or no numeric aligned features).")
else:
    relative_steps = np.arange(-PRE_WINDOW_STEPS, POST_WINDOW_STEPS + 1, dtype=np.int64)

    # Build aligned matrices: one matrix per feature, rows = onsets, cols = relative steps
    aligned_mean_rows = []

    for feature_name in aligned_features:
        aligned_values = []

        for onset_index in range(len(onsets_table)):
            onset_row = onsets_table.iloc[onset_index]

            # Filter to the onset's group if asset/run exist
            subset = dataframe
            if "meta__asset_id" in onsets_table.columns:
                subset = subset[subset["meta__asset_id"] == onset_row["meta__asset_id"]]
            if "meta__run_id" in onsets_table.columns:
                subset = subset[subset["meta__run_id"] == onset_row["meta__run_id"]]

            # Ensure ordering by event_step
            if "event_step" in subset.columns:
                subset = subset.sort_values("event_step")
            else:
                subset = subset.sort_values("time_index")

            onset_step = int(onset_row["event_step"])
            start_step = onset_step - PRE_WINDOW_STEPS
            end_step = onset_step + POST_WINDOW_STEPS

            window = subset[(subset["event_step"] >= start_step) & (subset["event_step"] <= end_step)].copy()
            if len(window) == 0:
                continue

            # Reindex to complete relative step range (so we can average across onsets)
            window["relative_step"] = window["event_step"].astype(int) - onset_step
            window = window.set_index("relative_step")

            # Build full aligned vector with NaNs where missing
            aligned_vector = pd.Series(index=relative_steps, dtype=float)
            feature_series = window[feature_name].astype(float)

            # Fill aligned_vector values
            for step_value in feature_series.index:
                if int(step_value) in aligned_vector.index:
                    aligned_vector.loc[int(step_value)] = float(feature_series.loc[step_value])

            # Normalize within-window (z-score) after filling with median to avoid NaN explosion
            filled = aligned_vector.copy()
            median_value = float(np.nanmedian(filled.values))
            filled = filled.fillna(median_value)
            normalized = z_score(filled)

            aligned_values.append(normalized.values)

        if len(aligned_values) == 0:
            continue

        aligned_matrix = np.vstack(aligned_values)
        mean_curve = aligned_matrix.mean(axis=0)

        aligned_mean_rows.append({
            "feature": feature_name,
            "onsets_used": int(aligned_matrix.shape[0]),
        })

        # Plot mean curve
        plt.figure(figsize=(10, 4))
        plt.plot(relative_steps, mean_curve, label=f"{feature_name} (mean)")
        plt.axvline(0, linestyle="--")  # onset marker
        plt.title(f"Anomaly-onset aligned mean (z-scored): {feature_name}")
        plt.xlabel("Relative step (0 = anomaly onset)")
        plt.ylabel("Z-score")
        plt.legend()
        plt.tight_layout()

        plot_path = SILVER_EDA_ARTIFACTS_PATH / f"aligned_onset__mean__{feature_name}.png"
        plt.savefig(plot_path, dpi=200)
        plt.show()
        wandb.save(str(plot_path))

    aligned_summary = pd.DataFrame(aligned_mean_rows)
    aligned_summary_path = SILVER_EDA_ARTIFACTS_PATH / "aligned_onset__summary.csv"
    aligned_summary.to_csv(aligned_summary_path, index=False)
    wandb.save(str(aligned_summary_path))

aligned_features, len(onsets_table)

# %% [markdown]
# ----

# %%


# Collect numeric feature columns that exist
numeric_feature_columns = []
for feature_name in FEATURE_COLUMNS:
    if feature_name in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[feature_name]):
        numeric_feature_columns.append(feature_name)

if len(numeric_feature_columns) == 0:
    logger.info("No numeric feature columns available for PCA.")
else:
    modeling_frame = dataframe[numeric_feature_columns + (["anomaly_flag"] if "anomaly_flag" in dataframe.columns else [])].copy()

    # Optional sampling (keeps PCA fast)
    if len(modeling_frame) > PCA_SAMPLE_ROW_COUNT:
        modeling_frame = modeling_frame.sample(n=PCA_SAMPLE_ROW_COUNT, random_state=42).reset_index(drop=True)

    feature_matrix = modeling_frame[numeric_feature_columns].copy()

    # Median imputation (simple, stable)
    for feature_name in numeric_feature_columns:
        median_value = float(feature_matrix[feature_name].median(skipna=True))
        feature_matrix[feature_name] = feature_matrix[feature_name].fillna(median_value)

    # Scaling
    scaler = RobustScaler() if USE_ROBUST_SCALER else StandardScaler()
    scaled_matrix = scaler.fit_transform(feature_matrix.values)

    # PCA
    pca_model = PCA(n_components=2, random_state=42)
    pca_values = pca_model.fit_transform(scaled_matrix)

    pca_dataframe = pd.DataFrame({
        "pca_1": pca_values[:, 0],
        "pca_2": pca_values[:, 1],
    })

    if "anomaly_flag" in modeling_frame.columns:
        pca_dataframe["anomaly_flag"] = modeling_frame["anomaly_flag"].astype(int).values
    else:
        pca_dataframe["anomaly_flag"] = 0

    # Save PCA table
    pca_table_path = SILVER_EDA_ARTIFACTS_PATH / "pca_2d__table.csv"
    pca_dataframe.to_csv(pca_table_path, index=False)
    wandb.save(str(pca_table_path))

    # Plot normal and abnormal as separate scatters (matplotlib assigns colors automatically)
    plt.figure(figsize=(8, 6))
    normal_points = pca_dataframe[pca_dataframe["anomaly_flag"] == 0]
    abnormal_points = pca_dataframe[pca_dataframe["anomaly_flag"] == 1]

    plt.scatter(normal_points["pca_1"], normal_points["pca_2"], s=10, label="Normal")
    if len(abnormal_points) > 0:
        plt.scatter(abnormal_points["pca_1"], abnormal_points["pca_2"], s=20, marker="x", label="Abnormal")

    explained = pca_model.explained_variance_ratio_
    plt.title(f"PCA 2D projection (explained var: {explained[0]:.2f}, {explained[1]:.2f})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()

    pca_plot_path = SILVER_EDA_ARTIFACTS_PATH / "pca_2d__scatter.png"
    plt.savefig(pca_plot_path, dpi=200)
    plt.show()
    wandb.save(str(pca_plot_path))

    ledger.add(
        kind="step",
        step="pca_2d_projection",
        message="Computed PCA(2) projection for separation check.",
        data={
            "numeric_feature_count": int(len(numeric_feature_columns)),
            "rows_used": int(len(pca_dataframe)),
            "scaler": "RobustScaler" if USE_ROBUST_SCALER else "StandardScaler",
            "explained_variance_ratio": [float(explained[0]), float(explained[1])],
            "pca_table_path": str(pca_table_path),
            "pca_plot_path": str(pca_plot_path),
        },
        logger=logger,
    )



# %% [markdown]
# ----

# %%

# Pick numeric features only
numeric_feature_columns = []
for feature_name in FEATURE_COLUMNS:
    if feature_name in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[feature_name]):
        numeric_feature_columns.append(feature_name)

if len(numeric_feature_columns) == 0:
    logger.info("No numeric feature columns available for imputation comparison.")
else:
    working = dataframe[numeric_feature_columns + (["meta__asset_id", "meta__run_id", "event_step"] if "event_step" in dataframe.columns else [])].copy()

    if len(working) > IMPUTE_SAMPLE_ROW_COUNT:
        working = working.sample(n=IMPUTE_SAMPLE_ROW_COUNT, random_state=42).reset_index(drop=True)

    # Strategy A: median imputation (global)
    median_imputed = working[numeric_feature_columns].copy()
    for feature_name in numeric_feature_columns:
        median_value = float(median_imputed[feature_name].median(skipna=True))
        median_imputed[feature_name] = median_imputed[feature_name].fillna(median_value)

    # Strategy B: mean imputation (global)
    mean_imputed = working[numeric_feature_columns].copy()
    for feature_name in numeric_feature_columns:
        mean_value = float(mean_imputed[feature_name].mean(skipna=True))
        mean_imputed[feature_name] = mean_imputed[feature_name].fillna(mean_value)

    # Strategy C: forward fill within (asset, run) if available, then median
    ffill_imputed = working.copy()
    grouping_columns = []
    if "meta__asset_id" in ffill_imputed.columns:
        grouping_columns.append("meta__asset_id")
    if "meta__run_id" in ffill_imputed.columns:
        grouping_columns.append("meta__run_id")

    if len(grouping_columns) > 0 and "event_step" in ffill_imputed.columns:
        ffill_imputed = ffill_imputed.sort_values(grouping_columns + ["event_step"]).reset_index(drop=True)
        for feature_name in numeric_feature_columns:
            ffill_imputed[feature_name] = ffill_imputed.groupby(grouping_columns, dropna=False)[feature_name].ffill()
    # finish with median for any remaining missing
    for feature_name in numeric_feature_columns:
        median_value = float(ffill_imputed[feature_name].median(skipna=True))
        ffill_imputed[feature_name] = ffill_imputed[feature_name].fillna(median_value)

    # Compare strategies on: percent filled and mean absolute change on originally-missing positions
    comparison_rows = []
    for feature_name in numeric_feature_columns:
        original_series = working[feature_name]
        missing_mask = original_series.isna()

        missing_count = int(missing_mask.sum())
        if missing_count == 0:
            continue

        # Values that got filled at missing positions
        filled_median = median_imputed.loc[missing_mask, feature_name].astype(float)
        filled_mean = mean_imputed.loc[missing_mask, feature_name].astype(float)
        filled_ffill = ffill_imputed.loc[missing_mask, feature_name].astype(float)

        # Compare filled values to the non-missing distribution center (median)
        non_missing = original_series.dropna().astype(float)
        baseline_median = float(non_missing.median()) if len(non_missing) > 0 else 0.0

        median_abs_shift = float((filled_median - baseline_median).abs().mean())
        mean_abs_shift = float((filled_mean - baseline_median).abs().mean())
        ffill_abs_shift = float((filled_ffill - baseline_median).abs().mean())

        comparison_rows.append({
            "feature": feature_name,
            "missing_count": missing_count,
            "missing_percent": float((missing_count / len(working)) * 100.0),
            "baseline_median": baseline_median,
            "median_impute_mean_abs_shift_from_median": median_abs_shift,
            "mean_impute_mean_abs_shift_from_median": mean_abs_shift,
            "ffill_then_median_mean_abs_shift_from_median": ffill_abs_shift,
        })

    impute_compare_table = pd.DataFrame(comparison_rows).sort_values("missing_percent", ascending=False)

    impute_compare_path = SILVER_EDA_ARTIFACTS_PATH / "imputation__comparison.csv"
    impute_compare_table.to_csv(impute_compare_path, index=False)
    wandb.save(str(impute_compare_path))

    # Recommendation logic (simple, explainable)
    # If time-ordered groups exist, ffill+median is often best for time series continuity.
    # Otherwise, median is the safest global method.
    has_groups = ("meta__asset_id" in dataframe.columns) and ("meta__run_id" in dataframe.columns)
    has_event_step = "event_step" in dataframe.columns

    if has_groups and has_event_step:
        recommendation = "forward_fill_within_group_then_median"
        recommendation_reason = (
            "Dataset has (asset,run) grouping and event_step ordering; forward fill preserves short gaps "
            "in time series while median handles leading gaps and long missing runs."
        )
    else:
        recommendation = "global_median"
        recommendation_reason = (
            "Dataset does not clearly support within-group forward fill; global median is robust to outliers "
            "and stable for Isolation Forest and One-Class SVM."
        )

    recommendation_payload = {
        "recommendation": recommendation,
        "reason": recommendation_reason,
        "has_grouping_columns": bool(has_groups),
        "has_event_step": bool(has_event_step),
        "comparison_csv": str(impute_compare_path),
    }

    save_json(recommendation_payload, file_path=SILVER_EDA_ARTIFACTS_PATH, file_name="imputation__recommendation.json")
    wandb.save(str(SILVER_EDA_ARTIFACTS_PATH / "imputation__recommendation.json"))

    ledger.add(
        kind="decision",
        step="imputation_recommendation",
        message="Compared basic imputation strategies and recorded recommendation for Gold layer.",
        data=recommendation_payload,
        logger=logger,
    )

    impute_compare_table.head(15), recommendation_payload

# %% [markdown]
# ----

# %%
ledger_path = SILVER_EDA_ARTIFACTS_PATH / f"ledger__{DATASET_NAME}__{STAGE}.json"

ledger.add(
    kind="step",
    step="finalize",
    message="Saved Silver EDA ledger and finalized W&B run.",
    data={"ledger_path": str(ledger_path)},
    logger=logger,
)

ledger.write_json(ledger_path)

silver_eda_dataset_name = (
    str(DATASET_NAME).strip().lower()
    if DATASET_NAME is not None
    else str(silver_truth.get("dataset_name", "pump")).strip().lower()
)

silver_eda_process_run_id = (
    SILVER_PROCESS_RUN_ID
    if "SILVER_PROCESS_RUN_ID" in globals()
    else make_process_run_id("silver_eda_process")
)

silver_eda_truth_layer_name = "silver_eda"

truth_config_snapshot = (
    TRUTH_CONFIG
    if "TRUTH_CONFIG" in globals()
    else {
        "runtime": {
            "stage": "silver_eda",
            "dataset": silver_eda_dataset_name,
            "mode": RUN_MODE if "RUN_MODE" in globals() else None,
            "profile": CONFIG_PROFILE if "CONFIG_PROFILE" in globals() else "default",
        }
    }
)

silver_eda_artifact_files = sorted(
    str(path)
    for path in SILVER_EDA_ARTIFACTS_PATH.glob("*")
    if path.is_file()
)

silver_eda_truth = initialize_layer_truth(
    truth_version=TRUTH_VERSION,
    dataset_name=silver_eda_dataset_name,
    layer_name=silver_eda_truth_layer_name,
    process_run_id=silver_eda_process_run_id,
    pipeline_mode=PIPELINE_MODE,
    parent_truth_hash=SILVER_TRUTH_HASH,
)

silver_eda_truth = update_truth_section(
    silver_eda_truth,
    "config_snapshot",
    truth_config_snapshot,
)

silver_eda_truth = update_truth_section(
    silver_eda_truth,
    "runtime_facts",
    {
        "input_row_count": int(len(dataframe)),
        "input_column_count": int(dataframe.shape[1]),
        "feature_column_count": int(len(FEATURE_COLUMNS)) if "FEATURE_COLUMNS" in globals() else 0,
        "top_feature_count": int(len(TOP_FEATURES)) if "TOP_FEATURES" in globals() else 0,
        "numeric_feature_count": int(len(numeric_feature_columns)) if "numeric_feature_columns" in globals() else 0,
        "max_onsets_used": int(MAX_ONSETS_TO_USE) if "MAX_ONSETS_TO_USE" in globals() else None,
        "pca_sample_row_count": int(PCA_SAMPLE_ROW_COUNT) if "PCA_SAMPLE_ROW_COUNT" in globals() else None,
        "impute_sample_row_count": int(IMPUTE_SAMPLE_ROW_COUNT) if "IMPUTE_SAMPLE_ROW_COUNT" in globals() else None,
        "parent_truth_hash": SILVER_TRUTH_HASH,
    },
)

silver_eda_truth = update_truth_section(
    silver_eda_truth,
    "artifact_paths",
    {
        "silver_truth_path": str(SILVER_TRUTH_PATH),
        "silver_eda_artifacts_dir": str(SILVER_EDA_ARTIFACTS_PATH),
        "silver_eda_ledger_path": str(ledger_path),
        "silver_eda_output_files": silver_eda_artifact_files,
        "effect_size_table_path": str(effect_path) if "effect_path" in globals() else None,
        "correlation_table_path": str(corr_path) if "corr_path" in globals() else None,
        "correlation_heatmap_path": str(heatmap_path) if "heatmap_path" in globals() else None,
        "top_feature_overlay_path": str(overlay_path) if "overlay_path" in globals() else None,
        "cluster_table_path": str(cluster_path) if "cluster_path" in globals() else None,
        "anomaly_onsets_path": str(onsets_path) if "onsets_path" in globals() else None,
        "pca_table_path": str(pca_table_path) if "pca_table_path" in globals() else None,
        "pca_plot_path": str(pca_plot_path) if "pca_plot_path" in globals() else None,
    },
)

silver_eda_truth_record = build_truth_record(
    truth_base=silver_eda_truth,
    row_count=len(dataframe),
    column_count=dataframe.shape[1],
    meta_columns=identify_meta_columns(dataframe),
    feature_columns=identify_feature_columns(dataframe),
)

SILVER_EDA_TRUTH_HASH = silver_eda_truth_record["truth_hash"]

silver_eda_truth_path = save_truth_record(
    silver_eda_truth_record,
    truth_dir=TRUTHS_PATH,
    dataset_name=silver_eda_dataset_name,
    layer_name=silver_eda_truth_layer_name,
)

append_truth_index(
    silver_eda_truth_record,
    truth_index_path=TRUTH_INDEX_PATH,
)

wandb.save(str(ledger_path))
wandb.save(str(silver_eda_truth_path))

logger.info("Silver EDA truth hash: %s", SILVER_EDA_TRUTH_HASH)
logger.info("Silver EDA truth path: %s", silver_eda_truth_path)

print("Silver EDA truth hash:", SILVER_EDA_TRUTH_HASH)
print("Silver EDA truth path:", silver_eda_truth_path)

wandb_run.finish()

# %% [markdown]
# ----

# %%
dataframe.columns

# %% [markdown]
# ----

# %%
if not Path(silver_eda_truth_path).exists():
    raise FileNotFoundError(f"Silver EDA truth file was not created: {silver_eda_truth_path}")

loaded_silver_eda_truth = load_json(silver_eda_truth_path)

if loaded_silver_eda_truth.get("truth_hash") != SILVER_EDA_TRUTH_HASH:
    raise ValueError(
        "Saved Silver EDA truth file hash does not match SILVER_EDA_TRUTH_HASH:\n"
        f"file={loaded_silver_eda_truth.get('truth_hash')}\n"
        f"record={SILVER_EDA_TRUTH_HASH}"
    )

if loaded_silver_eda_truth.get("parent_truth_hash") != SILVER_TRUTH_HASH:
    raise ValueError(
        "Saved Silver EDA truth file parent hash does not match SILVER_TRUTH_HASH:\n"
        f"truth={loaded_silver_eda_truth.get('parent_truth_hash')}\n"
        f"silver={SILVER_TRUTH_HASH}"
    )

loaded_runtime_facts = loaded_silver_eda_truth.get("runtime_facts", {})
loaded_artifact_paths = loaded_silver_eda_truth.get("artifact_paths", {})

if loaded_runtime_facts.get("parent_truth_hash") != SILVER_TRUTH_HASH:
    raise ValueError(
        "Silver EDA runtime_facts parent hash does not match SILVER_TRUTH_HASH:\n"
        f"runtime_facts={loaded_runtime_facts.get('parent_truth_hash')}\n"
        f"silver={SILVER_TRUTH_HASH}"
    )

if loaded_artifact_paths.get("silver_truth_path") != str(SILVER_TRUTH_PATH):
    raise ValueError(
        "Silver EDA artifact_paths['silver_truth_path'] does not match SILVER_TRUTH_PATH:\n"
        f"truth={loaded_artifact_paths.get('silver_truth_path')}\n"
        f"expected={SILVER_TRUTH_PATH}"
    )

print("Silver EDA lineage sanity check passed.")

# %% [markdown]
# ----


