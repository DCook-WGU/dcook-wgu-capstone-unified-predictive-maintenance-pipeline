# %% [markdown]
# # Silver Pre-EDA — Deliverable 1.2.1
# 
# This notebook completes the Silver Pre-EDA portion of the Medallion Architecture.
# 
# **Purpose:**  
# To perform the initial cleaning, missing-value review, structural preparation, and readiness checks needed before deeper Silver-layer exploratory analysis.  
# This includes validating the dataset structure, identifying early data quality issues, confirming sensor feature availability, and exporting a clean Silver-layer dataset.
# 
# **Outputs:**  
# - Cleaned `pump__silver__train.parquet` ready for downstream EDA and modeling  
# - Silver feature registry (`pump__silver__feature_registry.json`)  
# - Status flags (`is_anomaly`, `is_normal`) and schema-aligned metadata  
# - Structural consistency checks needed for the statistical comparison described in Section C of the project proposal
# 
# This deliverable ensures that the Silver layer provides a stable, reproducible foundation for both Silver EDA (Deliverable 1.2.2) and Gold modeling.

# %%
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Pattern

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import pyarrow.parquet as pq
import pyarrow as pa

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
    extract_truth_hash,
    load_truth_record_by_hash,
    get_dataset_name_from_truth,
    get_truth_hash,
    get_pipeline_mode_from_truth,
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
paths = get_paths()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# Get configs
#CONFIG_ROOT = Path("configs")
CONFIG_ROOT = paths.configs

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


# Temporary selector until you centralize mode selection
CONFIG_RUN_MODE = "train"
CONFIG_PROFILE = "default"


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

CONFIG = load_pipeline_config(
    config_root=CONFIG_ROOT,
    stage="silver_preeda",
    dataset="pump",
    mode=CONFIG_RUN_MODE,
    profile=CONFIG_PROFILE,
    project_root=paths.root,
).data


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

SILVER_CFG = CONFIG["silver_preeda"]
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
STAGE = "silver"
LAYER_NAME = SILVER_CFG["layer_name"]
SILVER_VERSION = CONFIG["versions"]["silver"]
CLEANING_RECIPE_ID = SILVER_CFG["cleaning_recipe_id"]
TRUTH_VERSION = CONFIG["versions"]["truth"]

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


PIPELINE_MODE = PIPELINE["execution_mode"]
RUN_MODE = CONFIG["runtime"]["mode"]

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

DATASET_NAME_CONFIG = CONFIG["dataset"]["name"]
DATASET_NAME = None

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

SILVER_PROCESS_RUN_ID = make_process_run_id(SILVER_CFG["process_run_id_prefix"])

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# ---- W&B ----
WANDB_PROJECT = CONFIG["wandb"]["project"]
WANDB_ENTITY = CONFIG["wandb"]["entity"]
WANDB_RUN_NAME = f"{SILVER_VERSION}"

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# ---- Canonical outputs ----
CANONICAL_OUTPUT_COLUMNS = list(SILVER_CFG["canonical_output_columns"])
CANONICAL_NON_META_ORDER = list(SILVER_CFG["canonical_non_meta_order"])

META_REQUIRED_COLUMNS = list(SILVER_CFG["meta_required_columns"])

CANONICAL_EXCLUDE_COLUMNS = list(SILVER_CFG["canonical_exclude_columns"])
LABEL_EXCLUDE_COLUMNS = list(SILVER_CFG["label_exclude_columns"])

LABEL_COLUMNS_ORDER = list(SILVER_CFG["label_columns_order"])

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# ---- Defaults / fallbacks ----
ASSET_ID_DEFAULT_FALLBACK = SILVER_CFG["asset_id_default_fallback"]
RUN_ID_DEFAULT_FALLBACK = SILVER_CFG["run_id_default_fallback"]

RAW_PREFIX = SILVER_CFG["raw_prefix"]

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# ---- Candidate lists ----
TIME_COLUMN_CANDIDATES = list(SILVER_CFG["time_column_candidates"])
STEP_COLUMN_CANDIDATES = list(SILVER_CFG["step_column_candidates"])
TIE_BREAKER_CANDIDATES = list(SILVER_CFG["tie_breaker_candidates"])

STATUS_COLUMN_CANDIDATES = list(SILVER_CFG["status_column_candidates"])
LABEL_COLUMN_CANDIDATES = list(SILVER_CFG["label_column_candidates"])

NORMAL_STATUS_VALUES = set(CONFIG["dataset"]["normal_status_values"])

RegexLike = Union[str, Pattern[str]]
UNNAMED_COLUMN_REGEX = re.compile(r"^unnamed:\s*\d+(\.\d+)?$", flags=re.IGNORECASE)

JUNK_COLUMN_CANDIDATES = {
    "Unnamed:",
    "Unnamed",
    "unnamed",
    "...",
    "level_0",
    "",
    " ",
    "\t",
    "\ufeff",
    "\ufeff<name>",
}

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# ---- QA / EDA thresholds ----
MIN_TIME_PARSE_SUCCESS_PERCENT = float(SILVER_CFG["min_time_parse_success_percent"])
MIN_STEP_PARSE_SUCCESS_PERCENT = float(SILVER_CFG["min_step_parse_success_percent"])

QUARANTINE_MISSING_PCT = float(SILVER_CFG["quarantine_missing_pct"])
CORRELATION_THRESHOLD = float(SILVER_CFG["correlation_threshold"])

TOP_N_SENSORS_FOR_PLOTS = int(SILVER_CFG["top_n_sensors_for_plots"])
PAIRPLOT_SENSOR_CAP = int(SILVER_CFG["pairplot_sensor_cap"])
PAIRPLOT_SAMPLE_N = int(SILVER_CFG["pairplot_sample_n"])
TOP_PLOT_COLS = int(SILVER_CFG["top_plot_cols"])
TOP_CORR_COLS = int(SILVER_CFG["top_corr_cols"])

ROLLING_MINUTES = int(SILVER_CFG["rolling_minutes"])
LOOKBACK_HOURS = int(SILVER_CFG["lookback_hours"])
BASELINE_DAYS = int(SILVER_CFG["baseline_days"])
BASELINE_GAP_HOURS = int(SILVER_CFG["baseline_gap_hours"])
SUSTAIN_MINUTES = int(SILVER_CFG["sustain_minutes"])
TOP_SENSOR_PRE_HOURS = int(SILVER_CFG["top_sensor_pre_hours"])

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# ---- File names ----
BRONZE_TRAIN_DATA_FILE_NAME = FILENAMES["bronze_train_file_name"]
SILVER_TRAIN_DATA_FILE_NAME = FILENAMES["silver_train_file_name"]

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# ---- Paths setup ----
BRONZE_TRAIN_DATA_PATH = Path(PATHS["data_bronze_train_dir"])
SILVER_TRAIN_DATA_PATH = Path(PATHS["data_silver_train_dir"])

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Keep this as the root silver artifacts dir because your notebook later appends / DATASET_NAME
SILVER_ARTIFACTS_PATH = Path(PATHS["silver_artifacts_dir"])

TRUTHS_PATH = Path(PATHS["truths_dir"])
TRUTH_INDEX_PATH = Path(PATHS["truth_index_path"])
LOGS_PATH = Path(PATHS["logs_root"])

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# W&B
set_wandb_dir_from_config(CONFIG)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Path failsafes
SILVER_TRAIN_DATA_PATH.mkdir(parents=True, exist_ok=True)
SILVER_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
TRUTHS_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
# Optional resolved-config snapshot
CONFIG_SNAPSHOT_PATH = SILVER_ARTIFACTS_PATH / f"{DATASET_NAME_CONFIG}__silver_preeda__resolved_config.yaml"
if CONFIG["execution"].get("save_config_snapshot", True):
    export_config_snapshot(CONFIG, CONFIG_SNAPSHOT_PATH)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# %% [markdown]
# ----

# %%
# Logging Setup

# Create silver log path 
silver_log_path = paths.logs / "silver.log"

# Initial Logger
configure_logging(
    "capstone",
    silver_log_path,
    level=logging.DEBUG,
    overwrite_handlers=True,
)

# Initiate Logger and log file
logger = logging.getLogger("capstone.silver")

# Log load and initiation
logger.info("Silver stage starting")

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
    job_type="silver",
    config={
        "silver_version": SILVER_VERSION,
        "cleaning_recipe_id": CLEANING_RECIPE_ID,
        "quarantine_missing_pct": QUARANTINE_MISSING_PCT,
        "min_time_parse_success_percent": MIN_TIME_PARSE_SUCCESS_PERCENT,
        "rolling_window": ROLLING_MINUTES,
        "bronze_path": str(BRONZE_TRAIN_DATA_PATH / BRONZE_TRAIN_DATA_FILE_NAME),
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

# %% [markdown]
# ### Load Data

# %%
# Load Data

preferred_bronze = BRONZE_TRAIN_DATA_PATH / BRONZE_TRAIN_DATA_FILE_NAME

if preferred_bronze.exists():
    bronze_data_path = preferred_bronze
else:
    parquet_files = sorted(BRONZE_TRAIN_DATA_PATH.glob("*.parquet"))
    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No parquet files found in {BRONZE_TRAIN_DATA_PATH}")
    if len(parquet_files) > 1: 
        logger.warning("Multiple Parquet Files found; Using First %s", parquet_files[0])
    bronze_data_path = parquet_files[0]

if not bronze_data_path.exists():
    raise FileNotFoundError(f"Bronze parquet not found: {bronze_data_path}")
    
dataframe = load_data(bronze_data_path.parent, bronze_data_path.name)



#### #### #### #### #### #### #### #### 

logger.info("Loaded Bronze: %s | shape=%s", bronze_data_path, dataframe.shape)
wandb_run.log({"bronze_rows": int(dataframe.shape[0]), "bronze_cols": int(dataframe.shape[1])})

ledger.add(
    kind="step",
    step="load_bronze",
    message="Loaded Bronze Parquet",
    why="Silver must be derived from reprodicible Bronze Artifact",
    consequence="All silver outputs trace back to this file",
    data={"bronze_path": str(bronze_data_path), "shape": list(dataframe.shape), "cols": len(dataframe.columns)},
    logger=logger
)


#### #### #### #### #### #### #### #### 

display(dataframe.head(3))

# %% [markdown]
# ----

# %%
SILVER_PARENT_TRUTH_HASH = extract_truth_hash(dataframe)

if SILVER_PARENT_TRUTH_HASH is None:
    raise ValueError("Silver input dataframe does not contain a readable meta__truth_hash value.")

BRONZE_DATASET_NAME = (
    dataframe["meta__dataset"]
    .dropna()
    .astype("string")
    .str.strip()
)

BRONZE_DATASET_NAME = BRONZE_DATASET_NAME[BRONZE_DATASET_NAME != ""]

if len(BRONZE_DATASET_NAME) == 0:
    raise ValueError("Silver input dataframe is missing usable meta__dataset values.")

BRONZE_DATASET_NAME = str(BRONZE_DATASET_NAME.iloc[0]).strip()

parent_truth = load_parent_truth_record_from_dataframe(
    dataframe=dataframe,
    truth_dir=TRUTHS_PATH,
    parent_layer_name="bronze",
    dataset_name=BRONZE_DATASET_NAME,
    column_name="meta__truth_hash",
)

DATASET_NAME = get_dataset_name_from_truth(parent_truth)

SILVER_PARENT_TRUTH_HASH = get_truth_hash(parent_truth)

PARENT_PIPELINE_MODE = get_pipeline_mode_from_truth(parent_truth)
if PARENT_PIPELINE_MODE is not None:
    PIPELINE_MODE = PARENT_PIPELINE_MODE

if "meta__pipeline_mode" not in dataframe.columns:
    dataframe["meta__pipeline_mode"] = PIPELINE_MODE
else:
    dataframe["meta__pipeline_mode"] = dataframe["meta__pipeline_mode"].fillna(PIPELINE_MODE)

silver_truth = initialize_layer_truth(
    truth_version=TRUTH_VERSION,
    dataset_name=DATASET_NAME,
    layer_name=LAYER_NAME,
    process_run_id=SILVER_PROCESS_RUN_ID,
    pipeline_mode=PIPELINE_MODE,
    parent_truth_hash=SILVER_PARENT_TRUTH_HASH,
)

silver_truth = update_truth_section(
    silver_truth,
    "config_snapshot",
    {
        "silver_version": SILVER_VERSION,
        "cleaning_recipe_id": CLEANING_RECIPE_ID,
        "dataset_name_config": DATASET_NAME_CONFIG,
        "dataset_name_parent_truth": DATASET_NAME,
        "run_id_default_fallback": RUN_ID_DEFAULT_FALLBACK,
        "quarantine_missing_pct": QUARANTINE_MISSING_PCT,
        "min_time_parse_success_percent": MIN_TIME_PARSE_SUCCESS_PERCENT,
        "min_step_parse_success_percent": MIN_STEP_PARSE_SUCCESS_PERCENT,
        "pipeline_mode": PIPELINE_MODE,
    },
)

silver_truth = update_truth_section(
    silver_truth,
    "runtime_facts",
    {
        "parent_layer_name": "bronze",
        "parent_truth_hash": SILVER_PARENT_TRUTH_HASH,
        "dataset_name_from_parent_truth": DATASET_NAME,
    },
)

logger.info("Resolved Bronze parent truth hash: %s", SILVER_PARENT_TRUTH_HASH)
logger.info("Resolved Silver dataset name from Bronze truth: %s", DATASET_NAME)

print("Silver parent truth hash:", SILVER_PARENT_TRUTH_HASH)
print("Silver dataset name from parent truth:", DATASET_NAME)

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

# %% [markdown]
# ----

# %%
def remove_junk_import_columns(
        dataframe: pd.DataFrame, 
        *,
        junk_column_candidates: List[str],
        regex_pattern: Optional[RegexLike] = None,
        default_pattern: RegexLike = UNNAMED_COLUMN_REGEX,
    ) -> Tuple[pd.DataFrame, List[str], str]:

    dataframe_copy = dataframe.copy()
    
    def _ensure_regex_pattern(
            regex: Optional[RegexLike],
            *,
            default: RegexLike,
            flags: int = re.IGNORECASE,
        ) -> Pattern[str]:

        if regex is None:
            regex = default
        if isinstance(regex, re.Pattern):
            return regex
        if isinstance(regex, str):
            return re.compile(regex, flags=flags)

        raise TypeError(f"Regex must be a str, re.Pattern, or None; Instead got {type(regex)}")

    
    pattern = _ensure_regex_pattern(regex_pattern, default=default_pattern, flags=re.IGNORECASE)

    normalize_candidates = [
        str(candidate).strip().lower()
        for candidate in junk_column_candidates
        if candidate is not None and str(candidate).strip() != ""
        ]

    junk_prefixes = tuple(normalize_candidates)

    junk_columns_found: list[str] = []

    for column in dataframe_copy.columns:
        column_string = str(column)
        column_normalized = column_string.strip().lower()

        is_junk_prefix = column_normalized.startswith(junk_prefixes) if junk_prefixes else False
        is_regex_match = bool(pattern.search(column_string))

        if is_junk_prefix or is_regex_match:
            junk_columns_found.append(column_string)

    if junk_columns_found:
        dataframe_copy = dataframe_copy.drop(columns=junk_columns_found, errors="ignore")

    return dataframe_copy, junk_columns_found, pattern.pattern


# %%
dataframe, junk_columns_found, pattern_used = remove_junk_import_columns(
    dataframe,
    junk_column_candidates=JUNK_COLUMN_CANDIDATES,
    regex_pattern=UNNAMED_COLUMN_REGEX,
    default_pattern=UNNAMED_COLUMN_REGEX,
)

#### #### #### #### #### #### #### #### 


if junk_columns_found:

    logger.info("Dropped junk columns: %s", junk_columns_found)

    ledger.add(
        kind="step",
        step="remove_junk_import_columns",
        message="Dropped junk columns via candidates and/or regex pattern",
        data={"dropped": junk_columns_found, "pattern": pattern_used},
        logger=logger,
    )
else:
    logger.info("No junk columns found.")

    ledger.add(
        kind="step",
        step="remove_junk_import_columns",
        message="No junk columns matched candidates/regex pattern",
        data={"dropped": [], "pattern": pattern_used},
        logger=logger,
    )




        

#### #### #### #### #### #### #### #### 

# %% [markdown]
# ----

# %%
def deduplicate_columns(
    dataframe: pd.DataFrame,
    *,
    keep: str = "first",
) -> Tuple[pd.DataFrame, List[str]]:
    
    dataframe = dataframe.copy()
    
    if dataframe.columns.is_unique:
        return dataframe, []
    
    duplicates_found = dataframe.columns[dataframe.columns.duplicated()].tolist()
    dataframe = dataframe.loc[:, ~dataframe.columns.duplicated(keep=keep)]
    
    return dataframe, duplicates_found




# %%
dataframe, duplicates_columns_found = deduplicate_columns(dataframe, keep="first")

if duplicates_columns_found:
    logger.warning("Duplicate column names removed (kept first): %s", duplicates_columns_found)
else:
    logger.info("No duplicate column names detected.")

ledger.add(
    kind="step",
    step="deduplicate_columns",
    message="Removed duplicate column names (kept first occurrence).",
    data={"duplicates": duplicates_columns_found, "count": len(duplicates_columns_found)},
    logger=logger,
)

# %% [markdown]
# ----

# %%
def validate_dataset_name_for_silver(
        dataframe: pd.DataFrame,
        *,
        truth_dataset_name: Optional[str] = None,
        config_value: Optional[str] = None,
        bronze_source_column: str = "meta__dataset",
        fail_on_multiple_in_bronze: bool = True,
    ) -> Tuple[str, str, str]:
    """
    Validate dataset name for Silver.

    Silver does not resolve or assign dataset identity.
    Silver verifies that Bronze-stamped dataset identity exists and is consistent.
    """

    def _clean_values(series: pd.Series) -> pd.Series:
        values = (
            series.dropna()
            .astype("string")
            .str.strip()
        )
        return values[values != ""]

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

    if bronze_source_column not in dataframe.columns:
        raise ValueError(
            f"Silver dataset validation failed because required column '{bronze_source_column}' is missing."
        )

    values = _clean_values(dataframe[bronze_source_column])

    if len(values) == 0:
        raise ValueError(
            f"Silver dataset validation failed because '{bronze_source_column}' exists but contains no usable values."
        )

    unique_values = values.unique()

    if len(unique_values) == 1:
        dataset_name = _normalize_dataset_name(str(unique_values[0]))
        dataset_method = "unique"
    else:
        if fail_on_multiple_in_bronze:
            raise ValueError(
                f"Silver dataset validation failed because multiple values were found in "
                f"'{bronze_source_column}'. Values discovered: {unique_values[:10]}"
            )
        top_value = values.value_counts().index[0]
        dataset_name = _normalize_dataset_name(str(top_value))
        dataset_method = "mode"

    if truth_dataset_name is not None and str(truth_dataset_name).strip() != "":
        truth_dataset_name_normalized = _normalize_dataset_name(str(truth_dataset_name))
        if dataset_name != truth_dataset_name_normalized:
            raise ValueError(
                "Silver dataset validation failed because dataframe meta__dataset does not match "
                f"truth dataset name. dataframe={dataset_name}, truth={truth_dataset_name_normalized}"
            )

    if config_value is not None and str(config_value).strip() != "":
        config_dataset_name_normalized = _normalize_dataset_name(str(config_value))
        if dataset_name != config_dataset_name_normalized:
            raise ValueError(
                "Silver dataset validation failed because dataframe meta__dataset does not match "
                f"configured dataset name. dataframe={dataset_name}, config={config_dataset_name_normalized}"
            )

    return dataset_name, bronze_source_column, dataset_method

# %%
VALIDATED_DATASET_NAME, DATASET_SOURCE_COLUMN, DATASET_METHOD = validate_dataset_name_for_silver(
    dataframe=dataframe,
    truth_dataset_name=DATASET_NAME,
    config_value=None,
    bronze_source_column="meta__dataset",
    fail_on_multiple_in_bronze=True,
)

DATASET_NAME = str(VALIDATED_DATASET_NAME).strip().lower()

silver_truth["dataset_name"] = DATASET_NAME
silver_truth = update_truth_section(
    silver_truth,
    "runtime_facts",
    {
        "dataset_validation": {
            "dataset_name": DATASET_NAME,
            "dataset_source_column": DATASET_SOURCE_COLUMN,
            "dataset_method": DATASET_METHOD,
        }
    },
)

ledger.add(
    kind="decision",
    step="validate_dataset_name",
    message="Validated Bronze-stamped dataset name for Silver against parent truth and recorded the validation in Truth Store.",
    data={
        "dataset_name": DATASET_NAME,
        "dataset_source_col": DATASET_SOURCE_COLUMN,
        "dataset_method": DATASET_METHOD,
    },
    logger=logger,
)

# %% [markdown]
# ----

# %% [markdown]
# ### Add Silver Meta Columns

# %%
dataframe = dataframe.copy()

# Silver runtime context goes to truth, not dataframe
SILVER_PROCESSED_AT_UTC = pd.Timestamp.now(tz="UTC")

# Ensure required row-level/source columns exist
for column_name in META_REQUIRED_COLUMNS:
    if column_name not in dataframe.columns:
        if column_name == "meta__dataset":
            dataframe[column_name] = pd.NA
        elif column_name == "meta__split":
            dataframe[column_name] = "unsplit"
        elif column_name == "meta__run_id":
            dataframe[column_name] = RUN_ID_DEFAULT_FALLBACK
        elif column_name == "meta__asset_id":
            dataframe[column_name] = ASSET_ID_DEFAULT_FALLBACK
        elif column_name == "meta__source_file":
            dataframe[column_name] = ""
        elif column_name == "meta__source_row_id":
            dataframe[column_name] = pd.RangeIndex(start=0, stop=len(dataframe), step=1)
        else:
            dataframe[column_name] = pd.NA

silver_truth = update_truth_section(
    silver_truth,
    "runtime_facts",
    {
        "processed_at_utc": SILVER_PROCESSED_AT_UTC,
        "silver_version": SILVER_VERSION,
        "cleaning_recipe_id": CLEANING_RECIPE_ID,
    },
)

ledger.add(
    kind="step",
    step="silver_meta_contract",
    message="Ensured required row-level/source meta exists. Silver runtime metadata is being stored in Truth Store, not row-stamped.",
    data={
        "required_meta_columns": list(META_REQUIRED_COLUMNS),
        "processed_at_utc": SILVER_PROCESSED_AT_UTC.isoformat(),
        "shape": {"rows": int(len(dataframe)), "cols": int(len(dataframe.columns))},
    },
    logger=logger,
)

# %% [markdown]
# ----

# %%
def resolve_label_or_status_source(
    dataframe: pd.DataFrame, 
    *,
    label_candidates: list[str], 
    status_candidates: list[str],
    top_n: int = 10,
) -> tuple[Optional[str], str, Dict[str, Any]]:
    
    # Helper function to get value counts from dataframe column
    # holds the top n amount and convert the the value count information into 
    # keys to string format so ledger serialization works well
    def _top_values(column: str) -> Dict[str, int]:
        value_counts = dataframe[column].value_counts(dropna=False).head(top_n)
        return {str(key): int(value) for key, value in value_counts.items()}


    def _column_info(column: str) -> Dict[str, Any]:
        total_rows = int(len(dataframe))
        non_null = int(dataframe[column].notna().sum())
        unique = int(dataframe[column].nunique(dropna=False))
        return {
            "top_values": _top_values(column),
            "unique_count": unique,
            "non_null_count": non_null,
            "total_row_count": total_rows,
            "coverage_pct": float((non_null / total_rows) * 100.0) if total_rows > 0 else 0.0,
        }

    found_labels: Dict[str, Any] = {}
    found_status: Dict[str, Any] = {}

    
    # Label Candidates
    for column in label_candidates:
        if column in dataframe.columns:
            found_labels[column] = _column_info(column)
        
    # Status Candidates
    for column in status_candidates:
        if column in dataframe.columns:
            found_status[column] = _column_info(column)


    # Choose primary source (policy: label preferred)
    if len(found_labels) > 0:
        chosen_column = next(iter(found_labels.keys()))
        chosen_type = "label"
        chosen_info = found_labels[chosen_column]
        chosen_from = "label_candidates"
    elif len(found_status) > 0:
        chosen_column = next(iter(found_status.keys()))
        chosen_type = "status"
        chosen_info = found_status[chosen_column]
        chosen_from = "status_candidates"
    else:
        return None, "none", {
            "chosen_from": "none",
            "found_labels": {},
            "found_status": {},
            "top_values": {},
            "unique_count": 0,
            "non_null_count": 0,
            "total_row_count": int(len(dataframe)),
            "coverage_pct": 0.0,
        }

    # Pack full context (so we know if both existed)
    info = {
        "chosen_from": chosen_from,
        "found_labels": found_labels,
        "found_status": found_status, 
        **chosen_info,                  
    }

    return chosen_column, chosen_type, info

# %%
dataframe.columns

# %%
LABEL_SOURCE_COLUMN, LABEL_SOURCE_TYPE, LABEL_SOURCE_INFO = resolve_label_or_status_source(
    dataframe,
    label_candidates=LABEL_COLUMN_CANDIDATES,
    status_candidates=STATUS_COLUMN_CANDIDATES,
    top_n=10,
)

found_label_columns = list(LABEL_SOURCE_INFO.get("found_labels", {}).keys())
found_status_columns = list(LABEL_SOURCE_INFO.get("found_status", {}).keys())

HAS_LABEL_CANDIDATES = int(len(found_label_columns) > 0)
HAS_STATUS_CANDIDATES = int(len(found_status_columns) > 0)

chosen_summary = {
    "top_values": LABEL_SOURCE_INFO.get("top_values", {}),
    "unique_count": LABEL_SOURCE_INFO.get("unique_count", 0),
    "non_null_count": LABEL_SOURCE_INFO.get("non_null_count", 0),
    "total_row_count": LABEL_SOURCE_INFO.get("total_row_count", int(len(dataframe))),
    "coverage_pct": LABEL_SOURCE_INFO.get("coverage_pct", 0.0),
}

silver_truth = update_truth_section(
    silver_truth,
    "runtime_facts",
    {
        "label_resolution": {
            "label_source_column": LABEL_SOURCE_COLUMN,
            "label_source_type": LABEL_SOURCE_TYPE,
            "label_source_info": LABEL_SOURCE_INFO,
            "has_label_candidates": HAS_LABEL_CANDIDATES,
            "has_status_candidates": HAS_STATUS_CANDIDATES,
            "chosen_summary": chosen_summary,
        }
    },
)

ledger.add(
    kind="decision",
    step="resolve_label_or_status_source",
    message="Resolved label/status source and wrote the resolution to Truth Store.",
    data={
        "label_source_col": LABEL_SOURCE_COLUMN,
        "label_source_type": LABEL_SOURCE_TYPE,
        "found_label_columns": found_label_columns,
        "found_status_columns": found_status_columns,
        "has_label_candidates": HAS_LABEL_CANDIDATES,
        "has_status_candidates": HAS_STATUS_CANDIDATES,
        "chosen_summary": chosen_summary,
    },
    logger=logger,
)

# %%
dataframe.columns

# %%
dataframe.head()

# %% [markdown]
# ----

# %% [markdown]
# ### Protect the Cananoical Time/Event Column Names
# 

# %%
def protect_canonical_output_names(
        dataframe: pd.DataFrame,
        *,
        canonical_output_columns: List[str],
        raw_prefix: str = "raw__",
    ) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, str]]:

    """
    If the input dataset already contains a canonical output name (event_time/event_step/time_index),
    rename the existing column to raw__<name> (or a unique variant) so we don't overwrite it.

    Returns:
      - updated dataframe
      - rename_map {original_name: new_name}
    """

    dataframe = dataframe.copy()

    rename_map: Dict[str, str] = {}

    for canonical_name in canonical_output_columns:
        if canonical_name not in dataframe.columns:
            continue

        base_new_name = f"{raw_prefix}{canonical_name}"
        new_name = base_new_name

        # This should ensure uniqueness
        counter = 2
        while new_name in dataframe.columns:
            new_name = f"{base_new_name}__{counter}"
            counter += 1

        dataframe = dataframe.rename(columns={canonical_name: new_name})
        rename_map[canonical_name] = new_name

    return dataframe, rename_map

# %%

# Run protection
dataframe, rename_map = protect_canonical_output_names(
    dataframe,
    canonical_output_columns=CANONICAL_OUTPUT_COLUMNS,
    raw_prefix = "raw__"
)

# json my rename map
rename_map_json = {str(key): str(value) for key, value in rename_map.items()}


#### #### #### #### #### #### #### #### 


if rename_map_json:
    ledger.add(
        kind="step",
        step="canonical_name_collision_protection",
        message="Renamed input columns that collide with canonical outputs (Policy A).",
        why="Prevent overwriting raw source columns when creating canonical outputs.",
        consequence="Original values preserved under raw__*; canonical outputs can be created safely.",
        data={
            "policy": "A",
            "canonical_outputs": list(CANONICAL_OUTPUT_COLUMNS),
            "raw_prefix": "raw__",
            "collisions": int(len(rename_map_json)),
            "renames": rename_map_json,
            "shape": {"rows": int(len(dataframe)), "cols": int(len(dataframe.columns))},
        },
        logger=logger,
    )
else:
    ledger.add(
        kind="step",
        step="canonical_name_collision_protection",
        message="No canonical-name collisions found (Policy A).",
        why="Confirm input does not contain columns that conflict with canonical outputs.",
        consequence="Canonical outputs can be created without renaming any source columns.",
        data={
            "policy": "A",
            "canonical_outputs": list(CANONICAL_OUTPUT_COLUMNS),
            "collisions": 0,
            "shape": {"rows": int(len(dataframe)), "cols": int(len(dataframe.columns))},
        },
        logger=logger,
    )

rename_map_json


# %% [markdown]
# ----

# %%
def _pick_first_existing_candidate_column(dataframe: pd.DataFrame, candidates: List[str]) -> Optional[str]: 
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def _ensure_grouping_columns_exists(
    dataframe: pd.DataFrame,
    *,
    asset_column: str = "meta__asset_id",
    run_column: str = "meta__run_id",
    default_asset_value: str = "asset__single",
    default_run_value: str = "run__single",
) -> pd.DataFrame:
    if asset_column not in dataframe.columns:
        dataframe[asset_column] = default_asset_value
    if run_column not in dataframe.columns:
        dataframe[run_column] = default_run_value
    return dataframe


def evaluate_time_candidates(
    dataframe: pd.DataFrame,
    *,
    candidates: List[str],
    minimum_parse_success_percent: float,
) -> Dict[str, Any]:

    chosen_time_column = None
    chosen_parse_time_series = None
    chosen_parse_success_percent = None
    
    for candidate in candidates:
        if candidate not in dataframe.columns:
            continue

        parsed_time_series = pd.to_datetime(dataframe[candidate], errors="coerce", utc=True)

        if len(dataframe) == 0:
            parse_success_percent = 0.0
        else:
            parse_success_percent = float(parsed_time_series.notna().mean() * 100.0)

        if parse_success_percent >= minimum_parse_success_percent:
            chosen_time_column = candidate
            chosen_parse_time_series = parsed_time_series
            chosen_parse_success_percent = parse_success_percent
            break

    return {
        "time_column_used": chosen_time_column,
        "time_parse_success_percent": chosen_parse_success_percent,
        "parsed_time_series": chosen_parse_time_series,
    }


def evaluate_step_candidates(
    dataframe: pd.DataFrame,
    *,
    candidates: List[str],
    minimum_parse_success_percent: float,
) -> Dict[str, Any]:

    chosen_step_column = None
    chosen_parse_success_percent = None
    
    for candidate in candidates:
        if candidate not in dataframe.columns:
            continue

        numeric_series = pd.to_numeric(dataframe[candidate], errors="coerce")

        if len(dataframe) == 0:
            parse_success_percent = 0.0
        else:
            parse_success_percent = float(numeric_series.notna().mean() * 100.0)

        if parse_success_percent >= minimum_parse_success_percent:
            chosen_step_column = candidate
            chosen_parse_success_percent = parse_success_percent
            break

    return {
        "step_column_used": chosen_step_column,
        "step_parse_success_percent": chosen_parse_success_percent,
    }


def build_canonical_identity_and_order_master(
    dataframe: pd.DataFrame,
    *,
    dataset_name: str,
    time_candidates: List[str],
    step_candidates: List[str],
    tie_breaker_candidates: List[str],
    minimum_time_parse_success_percent: float = 95.0,
    minimum_step_parse_success_percent: float = 95.0,
    default_asset_id: str = "asset__single",
    default_run_id: str = "run__single",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    dataframe_copy = dataframe.copy()

    dataframe_copy = _ensure_grouping_columns_exists(
        dataframe_copy,
        default_asset_value=default_asset_id,
        default_run_value=default_run_id,
    )

    group_columns = ["meta__asset_id", "meta__run_id"]

    tie_breaker_column_used = _pick_first_existing_candidate_column(
        dataframe_copy,
        candidates=tie_breaker_candidates,
    )

    time_evaluation = evaluate_time_candidates(
        dataframe_copy,
        candidates=time_candidates,
        minimum_parse_success_percent=minimum_time_parse_success_percent,
    )

    step_evaluation = evaluate_step_candidates(
        dataframe_copy,
        candidates=step_candidates,
        minimum_parse_success_percent=minimum_step_parse_success_percent,
    )

    time_column_used = time_evaluation["time_column_used"]
    time_parse_success_percent = time_evaluation["time_parse_success_percent"]
    parsed_time_series = time_evaluation["parsed_time_series"]

    step_column_used = step_evaluation["step_column_used"]
    step_parse_success_percent = step_evaluation["step_parse_success_percent"]

    if time_column_used is not None:
        ordering_mode = "time"
        dataframe_copy["event_time"] = parsed_time_series
    else:
        dataframe_copy["event_time"] = pd.NaT
        ordering_mode = "step" if step_column_used is not None else "row"

    sort_columns: List[str] = []
    sort_columns.extend(group_columns)

    if ordering_mode == "time":
        sort_columns.append("event_time")
    elif ordering_mode == "step":
        sort_columns.append(step_column_used)

    if tie_breaker_column_used is not None:
        sort_columns.append(tie_breaker_column_used)

    should_sort = bool(ordering_mode != "row" or tie_breaker_column_used is not None)

    if should_sort and len(sort_columns) > 0:
        dataframe_copy = dataframe_copy.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    dataframe_copy["event_step"] = dataframe_copy.groupby(group_columns, dropna=False).cumcount()
    dataframe_copy["time_index"] = dataframe_copy["event_step"].astype("int64")

    dataframe_copy["meta__event_id"] = (
        str(dataset_name)
        + ":" + dataframe_copy["meta__asset_id"].astype(str)
        + ":" + dataframe_copy["meta__run_id"].astype(str)
        + ":" + dataframe_copy["event_step"].astype(str)
    )

    if pd.api.types.is_datetime64_any_dtype(dataframe_copy["event_time"]):
        dataframe_copy["event_date"] = dataframe_copy["event_time"].dt.floor("D")
    else:
        dataframe_copy["event_date"] = pd.NaT

    group_count = int(dataframe_copy[group_columns].drop_duplicates().shape[0])

    info = {
        "ordering_mode": ordering_mode,
        "group_columns": group_columns,
        "group_count": group_count,
        "rows": int(len(dataframe_copy)),
        "time_column_used": time_column_used,
        "time_parse_success_percent": time_parse_success_percent,
        "step_column_used": step_column_used,
        "step_parse_success_percent": step_parse_success_percent,
        "tie_breaker_column_used": tie_breaker_column_used,
        "sorted": bool(should_sort),
        "sort_columns": sort_columns,
    }

    return dataframe_copy, info

# %%
dataframe, canonical_info = build_canonical_identity_and_order_master(
    dataframe,
    dataset_name=DATASET_NAME,
    time_candidates=TIME_COLUMN_CANDIDATES,
    step_candidates=STEP_COLUMN_CANDIDATES,
    tie_breaker_candidates=TIE_BREAKER_CANDIDATES,
    minimum_time_parse_success_percent=MIN_TIME_PARSE_SUCCESS_PERCENT,
    minimum_step_parse_success_percent=MIN_STEP_PARSE_SUCCESS_PERCENT,
    default_asset_id=ASSET_ID_DEFAULT_FALLBACK,
    default_run_id=RUN_ID_DEFAULT_FALLBACK,
)

silver_truth = update_truth_section(
    silver_truth,
    "runtime_facts",
    {
        "canonical_info": canonical_info,
    },
)

ledger.add(
    kind="decision",
    step="build_canonical_identity_and_order_master",
    message="Built canonical identity + ordering master and wrote the resolution to Truth Store.",
    data={
        "dataset_name": DATASET_NAME,
        "time_candidates": list(TIME_COLUMN_CANDIDATES),
        "step_candidates": list(STEP_COLUMN_CANDIDATES),
        "tie_breaker_candidates": list(TIE_BREAKER_CANDIDATES),
        **canonical_info,
        "shape": {"rows": int(len(dataframe)), "cols": int(len(dataframe.columns))},
    },
    logger=logger,
)

preview_columns = [
    "meta__asset_id", "meta__run_id",
    "event_time", "event_step", "time_index",
    "meta__event_id", "event_date",
]

preview_columns = [column for column in preview_columns if column in dataframe.columns]

dataframe[preview_columns].head(10), canonical_info

# %%
dataframe.columns

# %%
ANOMALY_FLAG_COLUMN = "anomaly_flag"

# %%

def normalize_label_to_binary(
    series: pd.Series,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Convert a label-like series to 0/1.
    Handles common patterns:
      - already numeric 0/1
      - booleans
      - strings like "normal"/"anomaly", "ok"/"fail", "true"/"false", "yes"/"no"
    Unknown/unhandled values become NaN then filled as 0 (conservative).
    """

    raw_series = series.copy()

    # Try numeric first
    numeric_series = pd.to_numeric(raw_series, errors="coerce")

    # If numeric has enough coverage, use it
    numeric_non_null_percent = float(numeric_series.notna().mean() * 100.0) if len(raw_series) else 0.0

    if numeric_non_null_percent >= 95.0:
        # Normalize: anything > 0 -> 1, else 0
        binary_series = (numeric_series.fillna(0) > 0).astype("int64")
        info = {
            "method": "numeric_threshold_gt_0",
            "numeric_non_null_percent": numeric_non_null_percent,
        }
        return binary_series, info

    # Fall back to string normalization
    string_series = raw_series.astype("string").str.strip().str.lower()

    mapping = {
        "1": 1, "true": 1, "yes": 1, "y": 1, "anomaly": 1, "fault": 1, "fail": 1, "failed": 1, "broken": 1,
        "0": 0, "false": 0, "no": 0, "n": 0, "normal": 0, "ok": 0, "good": 0, "healthy": 0, "running": 0,
    }

    mapped = string_series.map(mapping)
    mapped_non_null_percent = float(mapped.notna().mean() * 100.0) if len(mapped) else 0.0

    # Conservative fill: unknowns become 0 (normal)
    binary_series = mapped.fillna(0).astype("int64")

    info = {
        "method": "string_mapping_conservative_fill",
        "mapped_non_null_percent": mapped_non_null_percent,
        "numeric_non_null_percent": numeric_non_null_percent,
        "mapping_keys_count": int(len(mapping)),
    }
    return binary_series, info



# %%

def build_anomaly_flag_from_status(
    series: pd.Series,
    *,
    normal_value: Optional[str] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Convert a status-like series into anomaly_flag:
      anomaly_flag = 1 if status != normal_value else 0

    If normal_value is None, choose the mode (most frequent non-null).
    """
    raw_series = series.astype("string").str.strip()

    # Remove empty strings
    cleaned = raw_series.where(raw_series != "", other=pd.NA)

    # Determine normal value
    chosen_normal_value = normal_value
    if chosen_normal_value is None:
        non_null_values = cleaned.dropna()
        if len(non_null_values) == 0:
            chosen_normal_value = None
        else:
            chosen_normal_value = str(non_null_values.value_counts().index[0])

    if chosen_normal_value is None:
        # No usable normal value; default all normal (0)
        anomaly_flag = pd.Series(np.zeros(len(series), dtype=np.int64), index=series.index)
        info = {
            "method": "status_no_non_null_values",
            "normal_value": None,
        }
        return anomaly_flag, info

    anomaly_flag = (cleaned.astype("string") != str(chosen_normal_value)).fillna(False).astype("int64")

    info = {
        "method": "status_compare_to_normal_value",
        "normal_value": str(chosen_normal_value),
        "unique_status_count": int(cleaned.nunique(dropna=True)),
        "non_null_count": int(cleaned.notna().sum()),
    }
    return anomaly_flag, info


# %%

anomaly_build_info: Dict[str, Any] = {
    "source_type": LABEL_SOURCE_TYPE, 
    "source_column": LABEL_SOURCE_COLUMN
    }

if LABEL_SOURCE_TYPE == "label" and LABEL_SOURCE_COLUMN:
    anomaly_series, method_info = normalize_label_to_binary(dataframe[LABEL_SOURCE_COLUMN])
    dataframe[ANOMALY_FLAG_COLUMN] = anomaly_series
    anomaly_build_info.update(method_info)

elif LABEL_SOURCE_TYPE == "status" and LABEL_SOURCE_COLUMN:
    anomaly_series, method_info = build_anomaly_flag_from_status(dataframe[LABEL_SOURCE_COLUMN], normal_value=None)
    dataframe[ANOMALY_FLAG_COLUMN] = anomaly_series
    anomaly_build_info.update(method_info)

    # Optional state flags (handy for EDA)
    normal_value_text = anomaly_build_info.get("normal_value")
    if normal_value_text is not None:
        cleaned_status = dataframe[LABEL_SOURCE_COLUMN].astype("string").str.strip()
        dataframe["status_normal_value"] = str(normal_value_text)
        dataframe["is_normal"] = (cleaned_status == str(normal_value_text)).fillna(False).astype("int64")
        dataframe["is_anomaly"] = (dataframe[ANOMALY_FLAG_COLUMN] == 1).astype("int64")

else:
    # No label/status available: default to all normal
    dataframe[ANOMALY_FLAG_COLUMN] = np.zeros(len(dataframe), dtype=np.int64)
    anomaly_build_info.update({"method": "no_label_or_status_available_default_all_normal"})


# Basic summary for ledger
anomaly_counts = dataframe[ANOMALY_FLAG_COLUMN].value_counts(dropna=False).to_dict()
anomaly_counts_json = {str(key): int(value) for key, value in anomaly_counts.items()}

anomaly_build_info["anomaly_flag_counts"] = anomaly_counts_json
anomaly_build_info["anomaly_rate_percent"] = float((dataframe[ANOMALY_FLAG_COLUMN].mean() * 100.0)) if len(dataframe) else 0.0

#### #### #### #### #### #### #### #### 
ledger.add(
    kind="step",
    step="build_anomaly_flag",
    message="Built anomaly_flag from resolved label/status source.",
    why="Silver requires a consistent binary anomaly flag for segmentation, evaluation, and synthetic anomaly generation.",
    consequence="Downstream steps can rely on anomaly_flag (0/1) being present across datasets.",
    data={
        "label_source_column": LABEL_SOURCE_COLUMN,
        "label_source_type": LABEL_SOURCE_TYPE,
        **anomaly_build_info,
        "shape": {"rows": int(len(dataframe)), "columns": int(len(dataframe.columns))},
    },
    logger=logger,
)

#### #### #### #### #### #### #### #### 

# # Quick preview
preview_columns = [
    column for column in [
        LABEL_SOURCE_COLUMN, 
        "status_normal_value", 
        "is_normal", 
        "is_anomaly", 
        "anomaly_flag"
    ] 
    if column and column in dataframe.columns
]

dataframe[preview_columns].head(12), anomaly_build_info

# %% [markdown]
# ----

# %%
dataframe.columns

# %% [markdown]
# ----

# %%
def build_episode_ids_from_anomaly_flag(
    dataframe: pd.DataFrame,
    *,
    anomaly_flag_column: str = "anomaly_flag",
    group_columns: list[str] | None = None,
) -> pd.Series:
    """
    Build an episode id for each row in a time series dataset.

    An 'episode' is:
      normal -> anomaly_flag==1 (failure+recovery window) -> back to normal

    Each time we transition from anomaly (1) back to normal (0),
    the episode id increments by 1.

    Episodes are computed separately per group (meta__asset_id / meta__run_id if available).
    """

    if anomaly_flag_column not in dataframe.columns:
        raise ValueError(f"anomaly_flag_column '{anomaly_flag_column}' not found in dataframe.")

    working = dataframe.copy()

    # 1) Decide grouping columns (per asset/run)
    if group_columns is None:
        group_columns = []
        if "meta__asset_id" in working.columns:
            group_columns.append("meta__asset_id")
        if "meta__run_id" in working.columns:
            group_columns.append("meta__run_id")

    if not group_columns:
        # Single global group
        working["_tmp_group"] = 0
        group_columns = ["_tmp_group"]
        drop_tmp_group = True
    else:
        drop_tmp_group = False

    # 2) Decide time ordering column
    if "time_index" in working.columns:
        ordering_column = "time_index"
    elif "event_step" in working.columns:
        ordering_column = "event_step"
    else:
        ordering_column = None

    anomaly_series = working[anomaly_flag_column].fillna(0).astype(int)

    # 3) Container for episode ids
    episode_ids = pd.Series(index=working.index, dtype="int64")

    grouped = working.groupby(group_columns, dropna=False)

    for _, group_df in grouped:
        # Sort by time inside each asset/run
        if ordering_column is not None and ordering_column in group_df.columns:
            group_df = group_df.sort_values(by=ordering_column)

        idx = group_df.index

        current_episode = 0
        in_anomaly_window = False  # are we currently inside a failure+recovery window?

        for row_idx in idx:
            flag = anomaly_series.loc[row_idx]

            if flag == 1:
                # We are in an anomaly window
                in_anomaly_window = True
            else:  # flag == 0
                if in_anomaly_window:
                    # We just transitioned from anomaly back to normal -> new episode
                    current_episode += 1
                    in_anomaly_window = False

            episode_ids.loc[row_idx] = current_episode

    if drop_tmp_group:
        working.drop(columns=["_tmp_group"], inplace=True, errors="ignore")

    return episode_ids.astype("int64")

# %%
logger.info("Building episode ids from anomaly_flag for Silver dataframe.")

dataframe["meta__episode_id"] = build_episode_ids_from_anomaly_flag(
    dataframe,
    anomaly_flag_column="anomaly_flag",
)


episode_counts = dataframe["meta__episode_id"].value_counts().sort_index()

logger.info("Episode id summary: %s", episode_counts.to_dict())


#### #### #### #### #### #### #### #### 

ledger.add(
    kind="step",
    step="silver_build_episode_ids",
    message="Built meta__episode_id for Silver dataframe using anomaly windows.",
    data={
        "episode_column": "meta__episode_id",
        "unique_episode_count": int(dataframe["meta__episode_id"].nunique()),
    },
    logger=logger,
)

#### #### #### #### #### #### #### #### 


# %% [markdown]
# ----

# %%
DEFAULT_EXCLUDE_PREFIXES = ["meta__", "raw__"]


# %%

def classify_column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_categorical_dtype(series):
        return "categorical"
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        return "text"
    return "other"


# %%

def should_exclude_by_prefix(column_name: str, exclude_prefixes: List[str]) -> bool:
    for prefix in exclude_prefixes:
        if column_name.startswith(prefix):
            return True
    return False


# %%

def looks_like_identifier_column(
    dataframe: pd.DataFrame,
    *,
    column_name: str,
    unique_ratio_threshold: float = 0.50,
) -> bool:
    """
    Heuristic: exclude obvious identifier-like columns from features.
    Examples: *id*, *uuid*, *serial*, etc. or extremely high-cardinality columns.
    """
    lower_name = column_name.lower()

    identifier_keywords = ["id", "uuid", "guid", "serial", "record"]
    keyword_hit = False
    for keyword in identifier_keywords:
        if keyword in lower_name:
            keyword_hit = True
            break

    series = dataframe[column_name]
    total_rows = int(len(series))

    if total_rows == 0:
        return False

    unique_count = int(series.nunique(dropna=True))
    unique_ratio = float(unique_count / total_rows)

    # If it looks like an ID name AND has lots of unique values, treat it like an identifier
    if keyword_hit and unique_ratio >= unique_ratio_threshold:
        return True

    # If it is extremely high-cardinality text, treat as identifier/text signal rather than numeric feature
    if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and unique_ratio >= 0.80:
        return True

    return False


# %%

def identify_feature_set(
    dataframe: pd.DataFrame,
    *,
    exclude_prefixes: List[str],
    exclude_columns: List[str],
    label_source_column: Optional[str],
    include_categorical_features: bool = False,
    include_text_features: bool = False,
    include_datetime_features: bool = False,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, Any]]:
    """
    Returns:
      - selected_feature_columns: List[str]
      - feature_groups: Dict[group_name, List[str]]
      - info: Dict[str, Any] with counts and decisions
    """
    candidate_columns: List[str] = []
    for column_name in dataframe.columns:
        if should_exclude_by_prefix(column_name, exclude_prefixes):
            continue
        if column_name in exclude_columns:
            continue
        if label_source_column is not None and column_name == label_source_column:
            continue
        if looks_like_identifier_column(dataframe, column_name=column_name):
            continue
        candidate_columns.append(column_name)

    feature_groups: Dict[str, List[str]] = {
        "numeric": [],
        "boolean": [],
        "categorical": [],
        "text": [],
        "datetime": [],
        "other": [],
    }

    for column_name in candidate_columns:
        column_type = classify_column_type(dataframe[column_name])
        feature_groups[column_type].append(column_name)

    selected_feature_columns: List[str] = []
    # Default: numeric + boolean (safest across datasets)
    selected_feature_columns.extend(feature_groups["numeric"])
    selected_feature_columns.extend(feature_groups["boolean"])

    if include_categorical_features:
        selected_feature_columns.extend(feature_groups["categorical"])

    if include_text_features:
        selected_feature_columns.extend(feature_groups["text"])

    if include_datetime_features:
        selected_feature_columns.extend(feature_groups["datetime"])

    # Keep stable ordering
    selected_feature_columns = sorted([str(name) for name in selected_feature_columns])

    info: Dict[str, Any] = {
        "candidate_column_count": int(len(candidate_columns)),
        "selected_feature_count": int(len(selected_feature_columns)),
        "group_counts": {
            "numeric": int(len(feature_groups["numeric"])),
            "boolean": int(len(feature_groups["boolean"])),
            "categorical": int(len(feature_groups["categorical"])),
            "text": int(len(feature_groups["text"])),
            "datetime": int(len(feature_groups["datetime"])),
            "other": int(len(feature_groups["other"])),
        },
        "selection_policy": {
            "include_categorical_features": bool(include_categorical_features),
            "include_text_features": bool(include_text_features),
            "include_datetime_features": bool(include_datetime_features),
        },
    }

    return selected_feature_columns, feature_groups, info


# %%

exclude_columns_combined = []
exclude_columns_combined.extend(CANONICAL_EXCLUDE_COLUMNS)
exclude_columns_combined.extend(LABEL_EXCLUDE_COLUMNS)

FEATURE_COLUMNS, FEATURE_GROUPS, FEATURE_INFO = identify_feature_set(
    dataframe,
    exclude_prefixes=DEFAULT_EXCLUDE_PREFIXES,
    exclude_columns=exclude_columns_combined,
    label_source_column=LABEL_SOURCE_COLUMN,
    include_categorical_features=False,
    include_text_features=False,
    include_datetime_features=False,
)



# %% [markdown]
# ----

# %%
def identify_one_hot_encoding_columns(
    silver_dataframe: pd.DataFrame,
    *,
    excluded_columns: list[str] | None = None,
) -> tuple[list[str], dict]:
    """
    Identify categorical columns that should be one-hot encoded later in Gold.

    Returns
    -------
    one_hot_encoding_columns : list[str]
        Ordered list of categorical columns to encode.
    one_hot_encoding_truths : dict
        Truth-file fields describing whether encoding is needed and which columns
        were selected.
    """
    working_excluded_columns = set(excluded_columns or [])

    categorical_columns = [
        column_name
        for column_name in silver_dataframe.columns
        if column_name not in working_excluded_columns
        and (
            pd.api.types.is_object_dtype(silver_dataframe[column_name])
            or pd.api.types.is_categorical_dtype(silver_dataframe[column_name])
            or pd.api.types.is_bool_dtype(silver_dataframe[column_name])
        )
    ]

    categorical_columns = sorted(categorical_columns)

    one_hot_encoding_truths = {
        "needs_one_hot_encoding": bool(categorical_columns),
        "one_hot_encoding_columns": categorical_columns,
    }

    return categorical_columns, one_hot_encoding_truths

# %%
silver_one_hot_encoding_columns, silver_one_hot_encoding_truths = identify_one_hot_encoding_columns(
    silver_dataframe=dataframe,
    excluded_columns=exclude_columns_combined
)

silver_truth["needs_one_hot_encoding"] = silver_one_hot_encoding_truths["needs_one_hot_encoding"]
silver_truth["one_hot_encoding_columns"] = silver_one_hot_encoding_truths["one_hot_encoding_columns"]

# %% [markdown]
# ----

# %%
def compute_missingness_pct(
    dataframe: pd.DataFrame,
    *,
    columns: List[str],
    sort_desc: bool = True,
) -> pd.Series:
    """
    Returns a Series indexed by column name with percent missing (0..100).
    Only includes columns that exist in the dataframe.
    """
    columns = [column for column in columns if column in dataframe.columns]
    if not columns:
        return pd.Series(dtype="float64")

    missing_pct = dataframe[columns].isna().mean().mul(100.0)
    return missing_pct.sort_values(ascending=not sort_desc)


def quarantine_features_by_missingness(
    dataframe: pd.DataFrame,
    *,
    feature_columns: List[str],
    threshold_pct: float,
    drop_all_null: bool = True,
    numeric_only: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str], pd.Series]:
    """
    Drops features whose missingness >= threshold_pct (and optionally all-null).
    Returns:
      (clean_df, kept_features, dropped_features, missing_pct_series)
    """
    dataframe = dataframe.copy()

    present = [column for column in feature_columns if column in dataframe.columns]
    if numeric_only:
        present = [column for column in present if pd.api.types.is_numeric_dtype(dataframe[column])]

    missing_pct = compute_missingness_pct(dataframe, columns=present, sort_desc=True)

    if missing_pct.empty:
        return dataframe, present, [], missing_pct

    mask = missing_pct >= threshold_pct
    if drop_all_null:
        mask = mask | (missing_pct >= 100.0)

    dropped = missing_pct[mask].index.tolist()
    kept = [column for column in feature_columns if column in dataframe.columns and column not in dropped]

    if dropped:
        dataframe = dataframe.drop(columns=dropped, errors="ignore")

    return dataframe, kept, dropped, missing_pct

# %%
dataframe, FEATURE_COLUMNS, dropped_features, missing_pct = quarantine_features_by_missingness(
    dataframe,
    feature_columns=FEATURE_COLUMNS,
    threshold_pct=QUARANTINE_MISSING_PCT,
    drop_all_null=True,
    numeric_only=True,
)

#### #### #### #### #### #### #### #### 

logger.info("Missingness quarantine dropped %d features: %s", len(dropped_features), dropped_features)

ledger.add(
    kind="decision",
    step="missingness_quarantine",
    message="Dropped features exceeding missingness threshold (or all-null).",
    why="High-missingness features add noise/instability to EDA + modeling.",
    consequence="Feature set shrinks; downstream stats/imputation/modeling operate on remaining features.",
    data={
        "threshold_pct": float(QUARANTINE_MISSING_PCT),
        "dropped": dropped_features,
        "remaining": int(len(FEATURE_COLUMNS)),
        "top_missing_pct": missing_pct.head(10).to_dict(),
    },
    logger=logger,
)

#### #### #### #### #### #### #### #### 

# %% [markdown]
# ----

# %%
if dropped_features:
    dropped_set = set(dropped_features)

    # Filter groups to only remaining features
    FEATURE_GROUPS = {
        group: [column for column in columns if column not in dropped_set and column in FEATURE_COLUMNS]
        for group, columns in FEATURE_GROUPS.items()
    }
    # Optional: prune empty groups
    FEATURE_GROUPS = {group: columns for group, columns in FEATURE_GROUPS.items() if columns}

    # Update info counts to match final schema
    FEATURE_INFO["selected_feature_count"] = int(len(FEATURE_COLUMNS))
    if "group_counts" in FEATURE_INFO:
        FEATURE_INFO["group_counts"] = {
            group: int(len(columns)) for group, columns in FEATURE_GROUPS.items()
        }
    FEATURE_INFO["missingness_quarantine"] = {
        "threshold_pct": float(QUARANTINE_MISSING_PCT),
        "dropped_count": int(len(dropped_features)),
        "dropped": list(dropped_features),
    }

# %% [markdown]
# ----

# %%
def build_feature_set_identifier(feature_columns: List[str]) -> str:
    """
    Deterministic identifier for a feature set based on sorted column names.
    """
    normalized = "|".join(sorted([str(name) for name in feature_columns]))
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
    return f"feature_set__{digest}"

# %%
FEATURE_SET_IDENTIFIER = build_feature_set_identifier(FEATURE_COLUMNS)
FEATURE_COUNT = int(len(FEATURE_COLUMNS))

silver_truth = update_truth_section(
    silver_truth,
    "runtime_facts",
    {
        "feature_set": {
            "feature_set_id": FEATURE_SET_IDENTIFIER,
            "feature_count": FEATURE_COUNT,
        }
    },
)

ledger.add(
    kind="decision",
    step="finalize_feature_set",
    message="Finalized feature set and wrote feature set metadata to Truth Store.",
    data={
        "feature_set_id": FEATURE_SET_IDENTIFIER,
        "label_source_column": LABEL_SOURCE_COLUMN,
        "label_source_type": LABEL_SOURCE_TYPE,
        "exclude_prefixes": list(DEFAULT_EXCLUDE_PREFIXES),
        "exclude_columns": list(exclude_columns_combined),
        "feature_count": FEATURE_COUNT,
        "selected_feature_columns": list(FEATURE_COLUMNS),
        "feature_groups": {g: list(cols) for g, cols in FEATURE_GROUPS.items()},
        "shape": {"rows": int(len(dataframe)), "cols": int(len(dataframe.columns))},
    },
    logger=logger,
)

# %%
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# %%
def safe_list_columns(columns: List[str], existing_columns: List[str]) -> List[str]:
    kept: List[str] = []
    for column in columns:
        if column in existing_columns:
            kept.append(column)
    return kept


# %%

def collect_meta_columns(existing_columns: List[str]) -> List[str]:
    meta_columns: List[str] = []
    for column in existing_columns:
        if column.startswith("meta__"):
            meta_columns.append(column)
    return meta_columns


# %%

def reorder_silver_columns(
    dataframe: pd.DataFrame,
    *,
    canonical_non_meta_order: List[str],
    label_columns_order: List[str],
    feature_columns: List[str],
) -> pd.DataFrame:
    existing_columns = list(dataframe.columns)

    meta_columns = collect_meta_columns(existing_columns)
    meta_columns = sorted(meta_columns)

    canonical_columns = safe_list_columns(canonical_non_meta_order, existing_columns)
    label_columns = safe_list_columns(label_columns_order, existing_columns)

    feature_columns_present = safe_list_columns(feature_columns, existing_columns)

    # Remainder columns (preserve original order for anything not in the primary groups)
    ordered_set = set(meta_columns) | set(canonical_columns) | set(label_columns) | set(feature_columns_present)

    remainder_columns: List[str] = []
    for column in existing_columns:
        if column not in ordered_set:
            remainder_columns.append(column)

    final_order: List[str] = []
    final_order.extend(meta_columns)
    final_order.extend(canonical_columns)
    final_order.extend(label_columns)
    final_order.extend(feature_columns_present)
    final_order.extend(remainder_columns)

    return dataframe[final_order].copy()

# %%
dataframe.columns

# %%
dataframe = reorder_silver_columns(
    dataframe,
    canonical_non_meta_order=CANONICAL_NON_META_ORDER,
    label_columns_order=LABEL_COLUMNS_ORDER,
    feature_columns=FEATURE_COLUMNS,
)

# %% [markdown]
# ----

# %%
dataframe.columns

# %% [markdown]
# ----

# %%
def compute_quick_quality_checks(
    dataframe: pd.DataFrame,
    *,
    feature_columns: List[str],
    event_id_column: str = "meta__event_id",
    anomaly_flag_column: str = "anomaly_flag",
) -> Dict[str, Any]:
    total_rows = int(len(dataframe))

    # Duplicate checks
    duplicate_row_count = int(dataframe.duplicated().sum())

    duplicate_event_id_count = None
    if event_id_column in dataframe.columns:
        duplicate_event_id_count = int(dataframe[event_id_column].duplicated().sum())

    # Missingness for feature columns (limit logging size)
    numeric_missingness: Dict[str, float] = {}
    numeric_feature_count = 0

    for column in feature_columns:
        if column not in dataframe.columns:
            continue
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            numeric_feature_count += 1
            missing_percent = float(dataframe[column].isna().mean() * 100.0) if total_rows > 0 else 0.0
            numeric_missingness[column] = missing_percent

    # Keep only top 25 missingness entries (largest first) to avoid huge ledger payload
    sorted_missingness = sorted(numeric_missingness.items(), key=lambda item: item[1], reverse=True)
    top_missingness = dict(sorted_missingness[:25])

    # Anomaly rate
    anomaly_rate_percent = None
    anomaly_counts = None
    if anomaly_flag_column in dataframe.columns and total_rows > 0:
        anomaly_rate_percent = float(dataframe[anomaly_flag_column].mean() * 100.0)
        value_counts = dataframe[anomaly_flag_column].value_counts(dropna=False).to_dict()
        anomaly_counts = {str(key): int(value) for key, value in value_counts.items()}

    return {
        "total_rows": total_rows,
        "total_columns": int(len(dataframe.columns)),
        "duplicate_row_count": duplicate_row_count,
        "duplicate_event_id_count": duplicate_event_id_count,
        "numeric_feature_count": int(numeric_feature_count),
        "top_numeric_missingness_percent": top_missingness,
        "anomaly_rate_percent": anomaly_rate_percent,
        "anomaly_counts": anomaly_counts,
    }

# %%
quality_info = compute_quick_quality_checks(
    dataframe,
    feature_columns=FEATURE_COLUMNS,
)

# %% [markdown]
# ----

# %%
feature_registry: Dict[str, Any] = {
    "dataset_name": DATASET_NAME,
    "row_count": int(len(dataframe)),
    "column_count": int(len(dataframe.columns)),
    "feature_set_id": FEATURE_SET_IDENTIFIER,
    "feature_count": int(len(FEATURE_COLUMNS)),
    "feature_columns": list(FEATURE_COLUMNS),
    "feature_groups": {
        group_name: list(columns) for group_name, columns in FEATURE_GROUPS.items()
    },
    "feature_info": FEATURE_INFO,
    "exclude_prefixes": list(DEFAULT_EXCLUDE_PREFIXES),
    "exclude_columns": list(exclude_columns_combined),
    "label_source_column": LABEL_SOURCE_COLUMN,
    "label_source_type": LABEL_SOURCE_TYPE,
    "quarantine_missing_pct": float(QUARANTINE_MISSING_PCT),
    "pipeline_mode": PIPELINE_MODE,
    "process_run_id": SILVER_PROCESS_RUN_ID,
}

# %% [markdown]
# ----

# %%
if not dataframe.columns.is_unique:
    duplicates_found = dataframe.columns[dataframe.columns.duplicated()].tolist()

    raise ValueError(f"Duplicate columns detected before save: {duplicates_found}")

# %% [markdown]
# ----

# %%
dataframe.columns

# %%
dataframe.head()

# %% [markdown]
# ----

# %%
#SILVER_ARTIFACTS_PATH = paths.artifacts / "silver" / DATASET_NAME
#SILVER_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

#SILVER_TRAIN_DATA_FILE_NAME = f"{DATASET_NAME}__silver__train.parquet"

silver_truth = update_truth_section(
    silver_truth,
    "source_fingerprint",
    build_file_fingerprint(bronze_data_path),
)

silver_truth = update_truth_section(
    silver_truth,
    "runtime_facts",
    {
        "source_run_ids": (
            dataframe["meta__run_id"].dropna().astype(str).unique().tolist()
            if "meta__run_id" in dataframe.columns
            else []
        ),
        "quality_info": quality_info if "quality_info" in locals() else {},
    },
)

silver_truth = update_truth_section(
    silver_truth,
    "artifact_paths",
    {
        "bronze_source_path": str(bronze_data_path),
        "silver_output_dir": str(SILVER_TRAIN_DATA_PATH),
        "silver_output_file_name": SILVER_TRAIN_DATA_FILE_NAME,
        "feature_registry_dir": str(SILVER_ARTIFACTS_PATH),
    },
)

silver_truth = update_truth_section(
    silver_truth,
    "notes",
    {
        "purpose": "Silver preprocessing / canonicalization truth record",
    },
)

silver_truth_record = build_truth_record(
    truth_base=silver_truth,
    row_count=len(dataframe),
    column_count=dataframe.shape[1],
    meta_columns=identify_meta_columns(dataframe),
    feature_columns=identify_feature_columns(dataframe),
)

SILVER_TRUTH_HASH = silver_truth_record["truth_hash"]

dataframe = stamp_truth_columns(
    dataframe,
    truth_hash=SILVER_TRUTH_HASH,
    parent_truth_hash=SILVER_PARENT_TRUTH_HASH,
    pipeline_mode=PIPELINE_MODE,
)

feature_registry["truth_hash"] = SILVER_TRUTH_HASH
feature_registry["parent_truth_hash"] = SILVER_PARENT_TRUTH_HASH

silver_truth_path = save_truth_record(
    silver_truth_record,
    truth_dir=TRUTHS_PATH,
    dataset_name=DATASET_NAME,
    layer_name=LAYER_NAME,
)

append_truth_index(
    silver_truth_record,
    truth_index_path=TRUTH_INDEX_PATH,
)

ledger.add(
    kind="step",
    step="build_silver_truth_record",
    message="Built and saved Silver truth record and stamped only truth lineage columns to dataframe.",
    data={
        "silver_truth_hash": SILVER_TRUTH_HASH,
        "silver_parent_truth_hash": SILVER_PARENT_TRUTH_HASH,
        "silver_truth_path": str(silver_truth_path),
        "pipeline_mode": PIPELINE_MODE,
        "process_run_id": SILVER_PROCESS_RUN_ID,
    },
    logger=logger,
)

# %%
#SILVER_TRAIN_DATA_FILE_NAME = f"{DATASET_NAME}__silver__train.parquet"

saved_parquet_path = save_data(
    dataframe,
    file_path=SILVER_TRAIN_DATA_PATH,
    file_name=SILVER_TRAIN_DATA_FILE_NAME,
    create_dirs=True,
    index=False,
)

saved_registry_path = save_json(
    feature_registry,
    file_path=SILVER_ARTIFACTS_PATH,
    file_name=f"{DATASET_NAME}__silver__feature_registry.json",
    create_dirs=True,
    indent=2,
)

ledger.add(
    kind="step",
    step="silver_finalize_export",
    message="Finalized Silver export under the strict truth-store meta contract.",
    data={
        "dataset_name": DATASET_NAME,
        "silver_parquet_path": str(saved_parquet_path),
        "feature_registry_path": str(saved_registry_path),
        "quality_info": quality_info,
        "feature_set_id": FEATURE_SET_IDENTIFIER,
        "feature_count": FEATURE_COUNT,
        "silver_truth_hash": SILVER_TRUTH_HASH,
        "silver_parent_truth_hash": SILVER_PARENT_TRUTH_HASH,
        "pipeline_mode": PIPELINE_MODE,
        "process_run_id": SILVER_PROCESS_RUN_ID,
        "shape": {"rows": int(len(dataframe)), "cols": int(len(dataframe.columns))},
    },
    logger=logger,
)

# %% [markdown]
# ----

# %%
# Save the ledger
saved_ledger_path = ledger.write_json(
    SILVER_ARTIFACTS_PATH / f"silver__{DATASET_NAME}__ledger.json"
)


# %% [markdown]
# ----

# %%
finalize_info = finalize_wandb_stage(
    run=wandb_run,
    stage=STAGE,
    dataframe=dataframe,
    project_root=paths.root,
    logs_dir=paths.logs,
    dataset_dirs=[paths.data_silver_train],
    dataset_artifact_name=f"{DATASET_NAME}-{STAGE}-dataset",
    logger=logger,
    notebook_path=None,
    aliases=("latest",),
    table_key=None,
    table_n=15,
    profile=False,
)

# Close the W&B run
wandb_run.finish()

# %% [markdown]
# ----

# %%
required_silver_meta_columns = [
    "meta__truth_hash",
    "meta__parent_truth_hash",
    "meta__pipeline_mode",
]

missing_silver_meta_columns = [
    column_name
    for column_name in required_silver_meta_columns
    if column_name not in dataframe.columns
]
if missing_silver_meta_columns:
    raise ValueError(
        f"Silver dataframe is missing required lineage columns: {missing_silver_meta_columns}"
    )

silver_dataframe_truth_hash = extract_truth_hash(dataframe)
if silver_dataframe_truth_hash is None:
    raise ValueError("Silver dataframe does not contain a readable meta__truth_hash value.")

if silver_dataframe_truth_hash != SILVER_TRUTH_HASH:
    raise ValueError(
        "Silver dataframe truth hash does not match SILVER_TRUTH_HASH:\n"
        f"dataframe={silver_dataframe_truth_hash}\n"
        f"record={SILVER_TRUTH_HASH}"
    )

silver_parent_values = dataframe["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()
if not silver_parent_values:
    raise ValueError("Silver dataframe is missing populated meta__parent_truth_hash values.")

if len(silver_parent_values) != 1:
    raise ValueError(
        "Silver dataframe has multiple parent truth hashes:\n"
        f"{silver_parent_values}"
    )

if silver_parent_values[0] != SILVER_PARENT_TRUTH_HASH:
    raise ValueError(
        "Silver dataframe parent truth hash does not match SILVER_PARENT_TRUTH_HASH:\n"
        f"dataframe_parent={silver_parent_values[0]}\n"
        f"silver_parent_truth={SILVER_PARENT_TRUTH_HASH}"
    )

if not Path(silver_truth_path).exists():
    raise FileNotFoundError(f"Silver truth file was not created: {silver_truth_path}")

loaded_silver_truth = load_json(silver_truth_path)

if loaded_silver_truth.get("truth_hash") != SILVER_TRUTH_HASH:
    raise ValueError(
        "Saved Silver truth file hash does not match SILVER_TRUTH_HASH:\n"
        f"file={loaded_silver_truth.get('truth_hash')}\n"
        f"record={SILVER_TRUTH_HASH}"
    )

if loaded_silver_truth.get("parent_truth_hash") != SILVER_PARENT_TRUTH_HASH:
    raise ValueError(
        "Saved Silver truth file parent hash does not match SILVER_PARENT_TRUTH_HASH:\n"
        f"truth={loaded_silver_truth.get('parent_truth_hash')}\n"
        f"silver_parent={SILVER_PARENT_TRUTH_HASH}"
    )

if loaded_silver_truth.get("row_count") != len(dataframe):
    raise ValueError(
        "Silver truth row_count does not match dataframe row count:\n"
        f"truth={loaded_silver_truth.get('row_count')}\n"
        f"dataframe={len(dataframe)}"
    )

if loaded_silver_truth.get("column_count") != dataframe.shape[1]:
    raise ValueError(
        "Silver truth column_count does not match stamped dataframe column count:\n"
        f"truth={loaded_silver_truth.get('column_count')}\n"
        f"dataframe={dataframe.shape[1]}"
    )

print("Silver PreEDA lineage sanity check passed.")

# %% [markdown]
# ----


