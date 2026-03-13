# %% [markdown]
# ## Gold Preprocessing (Deliverable 1.3.1)
# 
# This notebook completes the Gold-layer preprocessing stage of the Medallion Architecture. It prepares the pump sensor dataset for model training and for the comparative evaluation described in Section C of the project proposal.
# 
# **Purpose:**  
# To transform the cleaned Silver-layer dataset into a fully model-ready Gold dataset using the final feature registry, imputation decisions, and anomaly labels produced during Silver EDA. This ensures that both the baseline model and the three-stage cascade model are trained on a consistent and reproducible feature set.
# 
# **Key Goals:**
# 
# - Load the finalized Silver dataset and feature registry.
# - Apply the Silver EDA imputation strategy (forward-fill followed by median).
# - Standardize and scale feature values as required for model stability.
# - Construct the model-ready Gold dataframe with:
#   - 50 vetted numeric features for Stage 1 (broad) modeling,
#   - A reduced feature subset for Stage 2 (narrow) modeling,
#   - Profile- and rule-based sensor groupings for Stage 3 confirmation logic.
# - Generate and export all Gold-layer preprocessing artifacts, including:
#   - The Gold preprocessed parquet dataset,
#   - Stage 1 and Stage 2 feature sets,
#   - Stage 3 rule/profile sensor lists,
#   - A preprocessing summary and metadata record.
# 
# **Relevance to Section C:**  
# Outputs from this notebook directly support the methods described in Section C by:
# 
# - Providing a stable, aligned feature matrix for the baseline Isolation Forest and the three-stage cascade (C.2, C.2.A).
# - Ensuring consistent preprocessing necessary for the paired model comparison and alert-quality evaluation (C.4).
# - Supplying the structured Gold dataset that underpins the visual communication of alert patterns and model differences (C.6).
# 
# This notebook finalizes the dataset that the Gold Modeling notebook will use to implement, evaluate, and compare both anomaly-detection approaches.

# %%
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


from pathlib import Path
import yaml

import logging
import wandb

import pandas as pd 
import numpy as np 

import joblib 

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


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
    stage="gold_preprocessing",
    dataset="pump",
    mode=CONFIG_RUN_MODE,
    profile=CONFIG_PROFILE,
    project_root=paths.root,
).data


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

GOLD_CFG = CONFIG["gold_preprocessing"]
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
STAGE = "gold"
LAYER_NAME = GOLD_CFG["layer_name"]
GOLD_VERSION = CONFIG["versions"]["gold"]
RECIPE_ID = GOLD_CFG["recipe_id"]
TRUTH_VERSION = CONFIG["versions"]["truth"]

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

PIPELINE_MODE = PIPELINE["execution_mode"]
RUN_MODE = CONFIG["runtime"]["mode"]

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

DATASET_NAME_CONFIG = CONFIG["dataset"]["name"]
DATASET_NAME = str(DATASET_NAME_CONFIG).strip().lower()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

GOLD_PROCESS_RUN_ID = make_process_run_id(GOLD_CFG["process_run_id_prefix"])

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  
# ---- W&B ----
WANDB_PROJECT = CONFIG["wandb"]["project"]
WANDB_ENTITY = CONFIG["wandb"]["entity"]
WANDB_RUN_NAME = f"{GOLD_VERSION}"

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

# ---- File names ----
SILVER_FILE_NAME = FILENAMES["silver_train_file_name"]


GOLD_FILE_NAME = FILENAMES["gold_preprocessed_file_name"]
GOLD_TRAIN_FILE_NAME = FILENAMES["gold_train_file_name"]
GOLD_TEST_FILE_NAME = FILENAMES["gold_test_file_name"]
GOLD_FIT_FILE_NAME = FILENAMES["gold_fit_file_name"]

GOLD_PRESCALED_FILE_NAME = FILENAMES["gold_preprocessed_prescaled_file_name"]
GOLD_SCALED_FILE_NAME = FILENAMES["gold_preprocessed_scaled_file_name"]

FEATURE_REGISTRY_FILE_NAME = FILENAMES["feature_registry_file_name"]
IMPUTE_RECOMMENDATION_FILE_NAME = FILENAMES["impute_recommendation_file_name"]

STAGE1_FEATURES_FILE_NAME = FILENAMES["stage1_features_file_name"]
STAGE2_FEATURES_FILE_NAME = FILENAMES["stage2_features_file_name"]
STAGE3_PRIMARY_FILE_NAME = FILENAMES["stage3_primary_file_name"]
STAGE3_SECONDARY_FILE_NAME = FILENAMES["stage3_secondary_file_name"]
CASCADE_RESULTS_FILE_NAME_CSV = FILENAMES["cascade_results_file_name_csv"]
CASCADE_RESULTS_FILE_NAME_PICKLE = FILENAMES["cascade_results_file_name_pickle"]
COMPARISON_FILE_NAME = FILENAMES["comparison_file_name"]

GOLD_PREPROCESSING_LEDGER_FILE_NAME = FILENAMES["gold_preprocessing_ledger_file_name"]

PREPROCESSING_SUMMARY_FILE_NAME = FILENAMES["preprocessing_summary_file_name"]
PREPROCESSING_METADATA_FILE_NAME = FILENAMES["preprocessing_metadata_file_name"]
REFERENCE_PROFILE_FILE_NAME = FILENAMES["reference_profile_file_name"]

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

# ---- Runtime knobs ----
TRAIN_FRACTION = float(GOLD_CFG["train_fraction"])
RANDOM_SEED = int(GOLD_CFG["random_seed"])

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

SCALER_KIND = GOLD_CFG["scaler_kind"]
SCALER_ARTIFACT_NAME_TEMPLATE = GOLD_CFG["scaler_artifact_name_template"]

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

STAGE2_TARGET_FEATURE_COUNT = int(GOLD_CFG["stage2_target_feature_count"])
STAGE3_PRIMARY_COUNT = int(GOLD_CFG["stage3_primary_count"])
STAGE3_SECONDARY_COUNT = int(GOLD_CFG["stage3_secondary_count"])


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  
# ---- Paths setup ----
SILVER_TRAIN_DATA_PATH = Path(PATHS["silver_train_data_path"])
SILVER_ARTIFACTS_PATH = Path(PATHS["silver_artifacts_dir"])
SILVER_EDA_ARTIFACTS_PATH = Path(PATHS["silver_eda_artifacts_dir"])

GOLD_DATA_PATH = Path(PATHS["gold_preprocessed_data_path"])
GOLD_TRAIN_DATA_PATH = Path(PATHS["gold_train_data_path"])
GOLD_TEST_DATA_PATH = Path(PATHS["gold_test_data_path"])
GOLD_FIT_DATA_PATH = Path(PATHS["gold_fit_data_path"])

GOLD_PRESCALED_DATA_PATH = Path(PATHS["gold_preprocessed_prescaled_data_path"])
GOLD_SCALED_DATA_PATH = Path(PATHS["gold_preprocessed_scaled_data_path"])

GOLD_ARTIFACTS_PATH = Path(PATHS["gold_artifacts_dir"])

TRUTHS_PATH = Path(PATHS["truths_dir"])
TRUTH_INDEX_PATH = Path(PATHS["truth_index_path"])

FEATURE_REGISTRY_PATH = Path(PATHS["feature_registry_path"])
IMPUTE_RECOMMENDATION_PATH = Path(PATHS["impute_recommendation_path"])

STAGE1_FEATURES_PATH = Path(PATHS["stage1_features_path"])
STAGE2_FEATURES_PATH = Path(PATHS["stage2_features_path"])
STAGE3_PRIMARY_PATH = Path(PATHS["stage3_primary_path"])
STAGE3_SECONDARY_PATH = Path(PATHS["stage3_secondary_path"])
CASCADE_RESULTS_PATH_CSV = Path(PATHS["cascade_results_path_csv"])
CASCADE_RESULTS_PATH_PICKLE = Path(PATHS["cascade_results_path_pickle"])
COMPARISON_PATH = Path(PATHS["comparison_path"])

PREPROCESSING_SUMMARY_PATH = Path(PATHS["preprocessing_summary_path"])
PREPROCESSING_METADATA_PATH = Path(PATHS["preprocessing_metadata_path"])
REFERENCE_PROFILE_PATH = Path(PATHS["reference_profile_path"])

LOGS_PATH = Path(PATHS["logs_root"])

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

# W&B
set_wandb_dir_from_config(CONFIG)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

# Path failsafes
GOLD_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
GOLD_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
TRUTHS_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

# Optional resolved-config snapshot
CONFIG_SNAPSHOT_PATH = GOLD_ARTIFACTS_PATH / f"{DATASET_NAME}__gold_preprocessing__resolved_config.yaml"
if CONFIG["execution"].get("save_config_snapshot", True):
    export_config_snapshot(CONFIG, CONFIG_SNAPSHOT_PATH)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  


# %% [markdown]
# ----

# %%
# Logging Setup

# Create gold log path 
gold_log_path = paths.logs / "gold_preprocessing.log"

# Initial Logger
configure_logging(
    "capstone",
    gold_log_path,
    level=logging.DEBUG,
    overwrite_handlers=True,
)

# Initiate Logger and log file
logger = logging.getLogger("capstone.gold")

# Log load and initiation
logger.info("Gold stage starting")

# Log paths loads
log_layer_paths(paths, current_layer="gold", logger=logger)


# %% [markdown]
# ----

# %%
# W&B

wandb_run = wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    name=WANDB_RUN_NAME,
    job_type="gold_preprocessing",
    config={
        "gold_version": GOLD_VERSION,
        "dataset": DATASET_NAME,
        "stage": STAGE,
        "train_fraction": TRAIN_FRACTION,
        "silver_path": str(SILVER_TRAIN_DATA_PATH),
        "feature_registry_path": str(FEATURE_REGISTRY_PATH),
        "impute_recommendation_path": str(IMPUTE_RECOMMENDATION_PATH),
        "gold_output_path": str(GOLD_DATA_PATH),
        "scaler_kind": str(SCALER_KIND),
        "stage2_target_feature_count": int(STAGE2_TARGET_FEATURE_COUNT),
        "stage3_primary_count": int(STAGE3_PRIMARY_COUNT),
        "stage3_secondary_count": int(STAGE3_SECONDARY_COUNT),
    },
)
logger.info("W&B initialized: %s", wandb.run.name)


# %% [markdown]
# ----

# %%
# Ledger Setup

ledger = Ledger(stage=STAGE, recipe_id=RECIPE_ID)

ledger.add(
    kind="step",
    step="init",
    message="Initialized ledger",
    logger=logger
)


# %% [markdown]
# ----

# %%

silver_path = SILVER_TRAIN_DATA_PATH

feature_registry_path = FEATURE_REGISTRY_PATH
imputation_recommendation_path = IMPUTE_RECOMMENDATION_PATH

logger.info("Loading Silver parquet: %s", silver_path)
silver_dataframe = load_data(silver_path)

logger.info("Loading Silver Truth: %s", silver_path)


logger.info("Loading feature registry: %s", feature_registry_path)
feature_registry = load_json(feature_registry_path)
feature_columns = feature_registry.get("feature_columns", [])
feature_set_id = feature_registry.get("feature_set_id", "unknown_feature_set")

logger.info("Loading imputation recommendation: %s", imputation_recommendation_path)
imputation_recommendation = load_json(imputation_recommendation_path, raise_if_missing=False, default={})
recommended_imputation = imputation_recommendation.get("recommendation", "global_median")

logger.info("Silver shape=%s", silver_dataframe.shape)
logger.info("Feature count=%d", len(feature_columns))
logger.info("Recommended imputation=%s", recommended_imputation)

#### #### #### #### #### #### #### #### 

ledger.add(
    kind="step",
    step="load_inputs",
    message="Loaded Silver parquet, feature registry, and imputation recommendation.",
    data={
        "silver_path": str(silver_path),
        "feature_registry_path": str(feature_registry_path),
        "impute_recommendation_path": str(imputation_recommendation_path),
        "feature_count": int(len(feature_columns)),
        "feature_set_id": str(feature_set_id),
        "recommended_imputation": str(recommended_imputation),
        "shape": {"rows": int(len(silver_dataframe)), "columns": int(len(silver_dataframe.columns))},
    },
    logger=logger,
)

#### #### #### #### #### #### #### #### 

silver_dataframe.head(3)

# %%
GOLD_PARENT_TRUTH_HASH = extract_truth_hash(silver_dataframe)

if GOLD_PARENT_TRUTH_HASH is None:
    raise ValueError("Gold preprocessing input dataframe does not contain a readable meta__truth_hash value.")

SILVER_DATASET_NAME = (
    silver_dataframe["meta__dataset"]
    .dropna()
    .astype("string")
    .str.strip()
)

SILVER_DATASET_NAME = SILVER_DATASET_NAME[SILVER_DATASET_NAME != ""]

if len(SILVER_DATASET_NAME) == 0:
    raise ValueError("Gold preprocessing input dataframe is missing usable meta__dataset values.")

SILVER_DATASET_NAME = str(SILVER_DATASET_NAME.iloc[0]).strip()

silver_truth = load_parent_truth_record_from_dataframe(
    dataframe=silver_dataframe,
    truth_dir=TRUTHS_PATH,
    parent_layer_name="silver",
    dataset_name=SILVER_DATASET_NAME,
    column_name="meta__truth_hash",
)

DATASET_NAME = get_dataset_name_from_truth(silver_truth)
GOLD_PARENT_TRUTH_HASH = get_truth_hash(silver_truth)

PARENT_PIPELINE_MODE = get_pipeline_mode_from_truth(silver_truth)
if PARENT_PIPELINE_MODE is not None:
    PIPELINE_MODE = PARENT_PIPELINE_MODE

FEATURE_REGISTRY_PATH = Path(get_artifact_path_from_truth(silver_truth, "feature_registry_dir")) / f"{DATASET_NAME}__silver__feature_registry.json"

silver_eda_artifacts_dir = Path(PATHS["artifacts_root"]) / "silver_eda" / DATASET_NAME
IMPUTE_RECOMMENDATION_PATH = silver_eda_artifacts_dir / FILENAMES["impute_recommendation_file_name"]

if "meta__pipeline_mode" not in silver_dataframe.columns:
    silver_dataframe["meta__pipeline_mode"] = PIPELINE_MODE
else:
    silver_dataframe["meta__pipeline_mode"] = silver_dataframe["meta__pipeline_mode"].fillna(PIPELINE_MODE)

gold_truth = initialize_layer_truth(
    truth_version=TRUTH_VERSION,
    dataset_name=DATASET_NAME,
    layer_name=LAYER_NAME,
    process_run_id=GOLD_PROCESS_RUN_ID,
    pipeline_mode=PIPELINE_MODE,
    parent_truth_hash=GOLD_PARENT_TRUTH_HASH,
)

gold_truth = update_truth_section(
    gold_truth,
    "config_snapshot",
    {
        "gold_version": GOLD_VERSION,
        "recipe_id": RECIPE_ID,
        "dataset_name_config": DATASET_NAME_CONFIG,
        "dataset_name_parent_truth": DATASET_NAME,
        "train_fraction": TRAIN_FRACTION,
        "random_seed": RANDOM_SEED,
        "scaler_kind": SCALER_KIND,
        "stage2_target_feature_count": STAGE2_TARGET_FEATURE_COUNT,
        "stage3_primary_count": STAGE3_PRIMARY_COUNT,
        "stage3_secondary_count": STAGE3_SECONDARY_COUNT,
        "pipeline_mode": PIPELINE_MODE,
    },
)

gold_truth = update_truth_section(
    gold_truth,
    "runtime_facts",
    {
        "parent_layer_name": "silver",
        "parent_truth_hash": GOLD_PARENT_TRUTH_HASH,
        "dataset_name_from_parent_truth": DATASET_NAME,
    },
)

logger.info("Resolved Silver parent truth hash: %s", GOLD_PARENT_TRUTH_HASH)
logger.info("Resolved Gold preprocessing dataset name from Silver truth: %s", DATASET_NAME)

print("Gold preprocessing parent truth hash:", GOLD_PARENT_TRUTH_HASH)
print("Gold preprocessing dataset name from parent truth:", DATASET_NAME)
print("Feature registry path from Silver truth:", FEATURE_REGISTRY_PATH)
print("Impute recommendation path from Silver EDA artifacts:", IMPUTE_RECOMMENDATION_PATH)

# %% [markdown]
# ----

# %%
dataframe = silver_dataframe.copy()

GOLD_PROCESSED_AT_UTC = pd.Timestamp.now(tz="UTC")

gold_truth = update_truth_section(
    gold_truth,
    "runtime_facts",
    {
        "processed_at_utc": GOLD_PROCESSED_AT_UTC,
        "gold_version": GOLD_VERSION,
        "preprocessing_recipe_id": RECIPE_ID,
    },
)

gold_working_dataframe = dataframe.copy()

# %% [markdown]
# ----

# %%
silver_truth = load_parent_truth_record_from_dataframe(
    dataframe=gold_working_dataframe,
    truth_dir=TRUTHS_PATH,
    parent_layer_name="silver",
    dataset_name=DATASET_NAME,
    column_name="meta__truth_hash",
)

# %% [markdown]
# ----

# %%
def build_episode_based_split_mask(
    dataframe: pd.DataFrame,
    *,
    train_fraction: float,
    episode_column: str = "meta__episode_id",
) -> tuple[pd.Series, dict]:
    """
    Build a train/test mask at the episode level.

    - Unique episodes are sorted by their id (which should already follow time order).
    - The earliest `train_fraction` of episodes are assigned to train.
    - All rows in those episodes are train; the rest are test.
    """
    if episode_column not in dataframe.columns:
        raise ValueError(f"Episode column '{episode_column}' not found in dataframe.")

    unique_episodes = np.sort(dataframe[episode_column].dropna().unique())
    n_episodes = len(unique_episodes)

    if n_episodes == 0:
        raise ValueError("No episodes found to build a split mask.")

    train_episode_count = max(1, int(np.floor(n_episodes * train_fraction)))
    train_episodes = set(unique_episodes[:train_episode_count])

    train_mask = dataframe[episode_column].isin(train_episodes)

    split_info = {
        "train_fraction": float(train_fraction),
        "episode_column": episode_column,
        "total_episodes": int(n_episodes),
        "train_episode_count": int(train_episode_count),
        "train_episodes": [int(e) for e in sorted(train_episodes)],
        "train_rows": int(train_mask.sum()),
        "test_rows": int((~train_mask).sum()),
    }

    return train_mask, split_info

# %%
gold_working_dataframe.columns

# %%
train_mask, split_info = build_episode_based_split_mask(
    gold_working_dataframe,
    train_fraction=TRAIN_FRACTION,
    episode_column="meta__episode_id",
)

#### #### #### #### #### #### #### #### 

ledger.add(
    kind="step",
    step="build_episode_based_split_mask",
    message="Created time-ordered, episodic sorted train/test split for Gold modeling.",
    data=split_info,
    logger=logger,
)

#### #### #### #### #### #### #### #### 

split_info

# %%
def stamp_training_metadata(
    dataframe: pd.DataFrame,
    train_mask: pd.Series | np.ndarray | None,
) -> pd.DataFrame:
    """
    If train_mask is provided, stamp only:
      - meta__is_train_flag : bool

    Returns a new dataframe (does not modify in place).
    """
    working_dataframe = dataframe.copy()

    if train_mask is None:
        return working_dataframe

    if isinstance(train_mask, pd.Series):
        train_mask_aligned = working_dataframe.index.to_series().map(train_mask).fillna(False).astype(bool)
    else:
        if len(train_mask) != len(working_dataframe):
            raise ValueError("train_mask length does not match dataframe length")
        train_mask_aligned = pd.Series(train_mask, index=working_dataframe.index, dtype=bool)

    working_dataframe["meta__is_train_flag"] = train_mask_aligned
    return working_dataframe

# %%
gold_working_dataframe = stamp_training_metadata(
    gold_working_dataframe,
    train_mask=train_mask,
)

gold_truth = update_truth_section(
    gold_truth,
    "runtime_facts",
    {
        "split_info": split_info,
    },
)

ledger.add(
    kind="step",
    step="add_train_flag",
    message="Stamped only row-level train flag; split metadata was written to Truth Store.",
    data=split_info,
    logger=logger,
)

# %% [markdown]
# ----

# %%
def select_numeric_feature_columns(
    dataframe: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> list[str]:
    numeric_feature_columns: list[str] = []

    for feature_name in feature_columns:
        if feature_name not in dataframe.columns:
            continue
        if pd.api.types.is_numeric_dtype(dataframe[feature_name]):
            numeric_feature_columns.append(feature_name)

    return numeric_feature_columns


# %%
numeric_feature_columns = select_numeric_feature_columns(
    silver_dataframe,
    feature_columns=feature_columns,
)


#### #### #### #### #### #### #### #### 

# TODO: Need Logger

ledger.add(
    kind="step",
    step="select_numeric_feature_columns",
    message="Capture the a list of the numeric feature columns from the dataframe",
    data={
        "silver_dataframe_path": str(silver_path),
        "silver_dataframe_shape": list(silver_dataframe.shape),
        "numeric_feature_columns": numeric_feature_columns
    },
    logger=logger,
)

#### #### #### #### #### #### #### #### 


# %% [markdown]
# ----

# %%
def apply_one_hot_encoding_from_truths(
    gold_dataframe: pd.DataFrame,
    *,
    upstream_truth_record: dict,
    drop_first: bool = False,
    dummy_na: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply one-hot encoding in Gold using the encoding instructions saved in the
    upstream truth record.

    Parameters
    ----------
    gold_dataframe : pd.DataFrame
        Input dataframe for Gold processing.
    upstream_truth_record : dict
        Truth record containing:
            - needs_one_hot_encoding
            - one_hot_encoding_columns
    drop_first : bool, default=False
        Whether to drop the first category level.
    dummy_na : bool, default=False
        Whether to create a separate indicator for missing values.

    Returns
    -------
    encoded_gold_dataframe : pd.DataFrame
        Dataframe after one-hot encoding.
    encoded_column_names : list[str]
        List of original columns that were encoded.
    """
    encoded_gold_dataframe = gold_dataframe.copy()

    needs_one_hot_encoding = bool(
        upstream_truth_record.get("needs_one_hot_encoding", False)
    )

    one_hot_encoding_columns = upstream_truth_record.get(
        "one_hot_encoding_columns",
        []
    )

    if not needs_one_hot_encoding:
        return encoded_gold_dataframe, []

    available_encoding_columns = [
        column_name
        for column_name in one_hot_encoding_columns
        if column_name in encoded_gold_dataframe.columns
    ]

    if not available_encoding_columns:
        return encoded_gold_dataframe, []

    encoded_gold_dataframe = pd.get_dummies(
        encoded_gold_dataframe,
        columns=available_encoding_columns,
        drop_first=drop_first,
        dummy_na=dummy_na,
        dtype=int,
    )

    return encoded_gold_dataframe, available_encoding_columns

# %%
gold_preprocessed_prescaled_dataframe, applied_one_hot_encoding_columns = apply_one_hot_encoding_from_truths(
    gold_dataframe=gold_working_dataframe,
    upstream_truth_record=silver_truth,
    drop_first=False,
    dummy_na=False,
)

gold_truth["needs_one_hot_encoding"] = bool(
    silver_truth.get("needs_one_hot_encoding", False)
)
gold_truth["one_hot_encoding_columns"] = silver_truth.get(
    "one_hot_encoding_columns",
    []
)
gold_truth["applied_one_hot_encoding_columns"] = applied_one_hot_encoding_columns

# %% [markdown]
# ----

# %%
def apply_imputation(
    dataframe: pd.DataFrame,
    *,
    numeric_feature_columns: list[str],
    method: str,
    train_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    
    # Copy Dataframe
    working_dataframe = dataframe.copy()

    # Decide which rows define the stats ("fit" rows)
    if train_mask is not None:
        stats_dataframe = working_dataframe.loc[train_mask].copy()
    else:
        stats_dataframe = working_dataframe

    # Decide Fill method
    # Fill Method - Foward Fill within group with median
    if method == "forward_fill_within_group_then_median":
        grouping_columns: list[str] = []

        if "meta__asset_id" in working_dataframe.columns:
            grouping_columns.append("meta__asset_id")
        if "meta__run_id" in working_dataframe.columns:
            grouping_columns.append("meta__run_id")

        ordering_column = None
        if "event_step" in working_dataframe.columns:
            ordering_column = "event_step"
        elif "time_index" in working_dataframe.columns:
            ordering_column = "time_index"

        if len(grouping_columns) > 0 and ordering_column is not None:
            working_dataframe = working_dataframe.sort_values(
                grouping_columns + [ordering_column]
            ).reset_index(drop=True)

            for feature_name in numeric_feature_columns:
                working_dataframe[feature_name] = (
                    working_dataframe
                    .groupby(grouping_columns, dropna=False)[feature_name]
                    .ffill()
                )
    
        for feature_name in numeric_feature_columns:
            median_value = float(stats_dataframe[feature_name].median(skipna=True))
            working_dataframe[feature_name] = working_dataframe[feature_name].fillna(median_value)

        return working_dataframe, {
            "imputation_method": method,
            "grouping_columns": grouping_columns,
            "ordering_column": ordering_column,
        }

    # Fill Method - Global Mean
    if method == "global_mean":
        for feature_name in numeric_feature_columns:
            mean_value = float(stats_dataframe[feature_name].mean(skipna=True))
            working_dataframe[feature_name] = working_dataframe[feature_name].fillna(mean_value)

        return working_dataframe, {
            "imputation_method": method,
            "grouping_columns": [],
            "ordering_column": None,
        }

    for feature_name in numeric_feature_columns:
        median_value = float(stats_dataframe[feature_name].median(skipna=True))
        working_dataframe[feature_name] = working_dataframe[feature_name].fillna(median_value)

    return working_dataframe, {
        "imputation_method": "global_median",
        "grouping_columns": [],
        "ordering_column": None,
    }

# %%
train_mask_for_stats = gold_working_dataframe["meta__is_train_flag"].astype(bool)

gold_working_dataframe, imputation_info = apply_imputation(
    gold_working_dataframe,
    numeric_feature_columns=numeric_feature_columns,
    method="forward_fill_within_group_then_median",
    train_mask=train_mask_for_stats,
)

gold_truth = update_truth_section(
    gold_truth,
    "runtime_facts",
    {
        "imputation_info": imputation_info,
        "recommended_imputation": recommended_imputation,
    },
)

# %% [markdown]
# ----

# %%
# Rebuild a fresh Series mask from the stamped column 
# We have to do this because the index may have changed after imputation

train_mask_flag = gold_working_dataframe["meta__is_train_flag"].astype(bool)


#### #### #### #### #### #### #### #### 

# TODO: Need Logger


train_mask_flag_dict = train_mask_flag.to_dict()

ledger.add(
    kind="step",
    step="training_mask_flag",
    message="Creating a mask from the meta__is_train_flag boolean flag",
    data={
        "training_mask_flag_count": train_mask_flag.shape,
        "training_mask_flag_values": train_mask_flag.values,
    },
    logger=logger,
)


#### #### #### #### #### #### #### #### 



# %% [markdown]
# ----

# %%
gold_preprocessed_prescaled_dataframe = gold_working_dataframe.copy()

ledger.add(
    kind="step",
    step="prepare_gold_preprocessed_prescaled_dataframe",
    message="Prepared Gold prescaled dataframe in memory. Save is deferred until truth lineage is finalized.",
    data={
        "gold_preprocessed_prescaled_shape": list(gold_preprocessed_prescaled_dataframe.shape),
    },
    logger=logger,
)

gold_preprocessed_prescaled_dataframe.shape

# %% [markdown]
# ----

# %%
def make_scaler(kind: str = "robust"):
    kind = kind.lower()
    if kind == "standard":
        return StandardScaler()
    elif kind == "minmax":
        return MinMaxScaler()
    elif kind == "robust":
        return RobustScaler()
    else:
        raise ValueError(f"Unknown scaler kind: {kind}. Use 'standard', 'minmax', or 'robust'.")

# %%
def fit_and_apply_scaler(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str],
    train_mask: pd.Series,
    normal_only_mask: Optional[pd.Series],
    scaler_kind: str,
    artifacts_path: Path,
    dataset_name: str,
    ledger,
) -> Tuple[pd.DataFrame, Path]:
    """
    Fit a scaler on normal-only train rows and apply it to all rows.
    """

    if normal_only_mask is not None:
        fit_mask = train_mask & normal_only_mask
        fit_source = "train ∩ normal-only"
    else:
        fit_mask = train_mask
        fit_source = "train (no explicit normal-only mask)"

    fit_rows = dataframe.loc[fit_mask, feature_columns]

    if fit_rows.empty:
        raise ValueError(
            "No rows available to fit the scaler. "
            "Check your train_mask and normal_only_mask."
        )

    scaler = make_scaler(kind=scaler_kind)
    scaler.fit(fit_rows.values)

    scaled_dataframe = dataframe.copy()
    scaled_dataframe.loc[:, feature_columns] = scaler.transform(
        scaled_dataframe[feature_columns].values
    )

    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    scaler_filename = SCALER_ARTIFACT_NAME_TEMPLATE.format(
        dataset=dataset_name,
        scaler_kind=scaler_kind.lower(),
    )

    scaler_path = artifacts_path / scaler_filename
    joblib.dump(scaler, scaler_path)

    ledger.add(
        kind="transform",
        step="gold_scaling",
        message="Scaling Gold",
        data={
            "scaler_kind": scaler_kind.lower(),
            "fit_source": fit_source,
            "fit_row_count": int(fit_rows.shape[0]),
            "feature_count": int(len(feature_columns)),
            "feature_columns": list(feature_columns),
            "scaler_path": str(scaler_path),
        },
        logger=logger,
    )

    if wandb.run is not None:
        wandb.config.update(
            {
                "gold_scaler.kind": scaler_kind.lower(),
                "gold_scaler.feature_count": int(len(feature_columns)),
            },
            allow_val_change=True,
        )
        wandb.log(
            {
                "gold_scaler.fit_rows": int(fit_rows.shape[0]),
            }
        )

    return scaled_dataframe, scaler_path

# %%
normal_only_mask = (
    (gold_working_dataframe["anomaly_flag"] == 0)
    if "anomaly_flag" in gold_working_dataframe.columns
    else None
)

gold_preprocessed_scaled_dataframe, scaler_path = fit_and_apply_scaler(
    dataframe=gold_working_dataframe,
    feature_columns=numeric_feature_columns,
    train_mask=train_mask_flag,
    normal_only_mask=normal_only_mask,
    scaler_kind=SCALER_KIND,
    artifacts_path=GOLD_ARTIFACTS_PATH,
    dataset_name=DATASET_NAME,
    ledger=ledger,
)

gold_truth = update_truth_section(
    gold_truth,
    "runtime_facts",
    {
        "feature_set_id": feature_set_id,
        "numeric_feature_count": int(len(numeric_feature_columns)),
        "scaler_path": str(scaler_path),
        "scaler_kind_runtime": SCALER_KIND,
        "recommended_imputation": recommended_imputation,
    },
)

print(f"Scaler saved to: {scaler_path}")
print(f"Scaled dataframe shape: {gold_preprocessed_scaled_dataframe.shape}")

gold_build_info = {
    "numeric_feature_count": int(len(numeric_feature_columns)),
    "feature_set_id": str(feature_set_id),
    "recommended_imputation": str(recommended_imputation),
    "scaler": str(SCALER_KIND),
}

ledger.add(
    kind="step",
    step="build_gold_model_ready_dataframe",
    message="Built Gold model-ready dataframe. Runtime build details were written to Truth Store.",
    data=gold_build_info,
    logger=logger,
)

gold_build_info

# %% [markdown]
# ----

# %% [markdown]
# ----

# %%
def get_training_rows_for_unsupervised_model(
    dataframe: pd.DataFrame,
    *,
    train_mask: pd.Series,
) -> pd.DataFrame:
    training_subset = dataframe.loc[train_mask].copy()

    if "anomaly_flag" in training_subset.columns:
        training_subset = training_subset[training_subset["anomaly_flag"] == 0].copy()

    return training_subset

# %%
# Normal Only
training_rows_for_fit = get_training_rows_for_unsupervised_model(
    gold_preprocessed_scaled_dataframe,
    train_mask=train_mask_flag,
)


#### #### #### #### #### #### #### #### 

# TODO: Need Logger

ledger.add(
    kind="step",
    step="get_training_rows_for_unsupervised_model",
    message="Filter the dataset by anomaly flag to generate normal data only for isolation forest training.",
    data={
        "gold_preprocessed_scaled_shape": list(gold_preprocessed_scaled_dataframe.shape),
        "gold_normal_only_split_shape": list(training_rows_for_fit.shape)
    },
    logger=logger,
)

#### #### #### #### #### #### #### #### 


# %% [markdown]
# ----

# %%
def build_reference_profile(
    dataframe: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> pd.DataFrame:
    reference_rows: list[dict] = []

    for feature_name in feature_columns:
        feature_series = dataframe[feature_name]

        reference_rows.append({
            "feature_name": feature_name,
            "median_value": float(feature_series.median()),
            "mean_value": float(feature_series.mean()),
            "standard_deviation": float(feature_series.std()) if not pd.isna(feature_series.std()) else 0.0,
            "lower_bound": float(feature_series.quantile(0.05)),
            "upper_bound": float(feature_series.quantile(0.95)),
        })

    reference_profile = pd.DataFrame(reference_rows)
    
    return reference_profile

# %%
reference_profile = build_reference_profile(
    training_rows_for_fit,
    feature_columns=numeric_feature_columns,
)


#### #### #### #### #### #### #### #### 

# TODO: Need Logger


ledger.add(
    kind="step",
    step="build_reference_profile",
    message="Built reference profile table from selected feature columns, capturing center, spread, and percentile bounds for downstream profile-based comparison.",
    data={
        "refernce_profile": reference_profile.to_dict('dict')
    },
    logger=logger,
)

#### #### #### #### #### #### #### #### 


# %% [markdown]
# ----

# %%
def choose_stage2_features_from_training_stability(
    training_dataframe: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_count: int,
) -> list[str]:
    ranking_rows: list[dict] = []

    for feature_name in feature_columns:
        feature_series = training_dataframe[feature_name]
        median_value = float(feature_series.median())
        standard_deviation = float(feature_series.std()) if not pd.isna(feature_series.std()) else 0.0

        coefficient_of_variation = standard_deviation / max(abs(median_value), 1e-6)

        ranking_rows.append({
            "feature_name": feature_name,
            "median_value": median_value,
            "standard_deviation": standard_deviation,
            "coefficient_of_variation": coefficient_of_variation,
        })

    ranking_dataframe = pd.DataFrame(ranking_rows)

    ranking_dataframe = ranking_dataframe.sort_values(
        by=["coefficient_of_variation", "standard_deviation", "feature_name"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    chosen_features = ranking_dataframe["feature_name"].head(target_count).tolist()
    
    return chosen_features

# %%
stage1_feature_columns = list(numeric_feature_columns)

stage2_feature_columns = choose_stage2_features_from_training_stability(
    training_rows_for_fit,
    feature_columns=stage1_feature_columns,
    target_count=STAGE2_TARGET_FEATURE_COUNT,
)


#### #### #### #### #### #### #### #### 

# TODO: Need Logger

ledger.add(
    kind="step",
    step="choose_stage2_features_from_training_stability",
    message="Ranked training features by stability and selected the top features with the lowest relative variability for Stage 2 modeling..",
    data={
        "dataframe_original_column_count": int(len(list(training_rows_for_fit.columns))),
        "dataframe_original_columns": list(training_rows_for_fit.columns),
        "stage1_feature_column_count": int(len(stage1_feature_columns)),
        "stage1_feature_columns": list(stage1_feature_columns),
        "stage2_feature_column_count": int(len(stage2_feature_columns)),
        "stage2_feature_columns": list(stage2_feature_columns),
    },
    logger=logger,
)

#### #### #### #### #### #### #### #### 


# %% [markdown]
# ----

# %%
def build_stage3_sensor_groups(
    reference_profile: pd.DataFrame,
    *,
    stage2_feature_columns: list[str],
    primary_count: int,
    secondary_count: int,
) -> tuple[list[str], list[str]]:
    ranked_reference = reference_profile[reference_profile["feature_name"].isin(stage2_feature_columns)].copy()

    ranked_reference = ranked_reference.sort_values(
        by=["standard_deviation", "feature_name"],
        ascending=[True, True]
    ).reset_index(drop=True)

    primary_rule_sensors = ranked_reference["feature_name"].head(primary_count).tolist()

    remaining_features = [
        feature_name
        for feature_name in ranked_reference["feature_name"].tolist()
        if feature_name not in primary_rule_sensors
    ]

    secondary_rule_sensors = remaining_features[:secondary_count]

    return primary_rule_sensors, secondary_rule_sensors

# %%
stage3_primary_rule_sensors, stage3_secondary_rule_sensors = build_stage3_sensor_groups(
    reference_profile,
    stage2_feature_columns=stage2_feature_columns,
    primary_count=STAGE3_PRIMARY_COUNT,
    secondary_count=STAGE3_SECONDARY_COUNT,
)


#### #### #### #### #### #### #### #### 

# TODO: Need Logger

ledger.add(
    kind="step",
    step="build_stage3_sensor_groups",
    message="Built Stage 3 sensor groups by ranking Stage 2 features on reference-profile stability and assigning the most stable features to primary and secondary rule sets.",
    data={
        "refernce_profile":  reference_profile.to_dict('dict'),
        "stage2_feature_column_count": int(len(list(stage2_feature_columns))),
        "stage2_feature_columns": list(stage2_feature_columns),
        "stage3_primary_rul_sensors": list(stage3_primary_rule_sensors),
        "stage3_primary_rul_sensors_count": int(len(list(stage3_primary_rule_sensors))),
        "stage3_secondary_rule_sensors": list(stage3_secondary_rule_sensors),
        "stage3_secondary_rule_sensors_count": int(len(list(stage3_secondary_rule_sensors))),
    },
    logger=logger,
)

#### #### #### #### #### #### #### #### 


# %% [markdown]
# ----

# %%
save_json(stage1_feature_columns, STAGE1_FEATURES_PATH)
save_json(stage2_feature_columns, STAGE2_FEATURES_PATH)
save_json(stage3_primary_rule_sensors, STAGE3_PRIMARY_PATH)
save_json(stage3_secondary_rule_sensors, STAGE3_SECONDARY_PATH)

reference_profile.to_csv(REFERENCE_PROFILE_PATH, index=False)

wandb.save(str(STAGE1_FEATURES_PATH))
wandb.save(str(STAGE2_FEATURES_PATH))
wandb.save(str(STAGE3_PRIMARY_PATH))
wandb.save(str(STAGE3_SECONDARY_PATH))
wandb.save(str(REFERENCE_PROFILE_PATH))

feature_set_summary = {
    "stage1_feature_count": int(len(stage1_feature_columns)),
    "stage2_feature_count": int(len(stage2_feature_columns)),
    "stage3_primary_rule_count": int(len(stage3_primary_rule_sensors)),
    "stage3_secondary_rule_count": int(len(stage3_secondary_rule_sensors)),
}

gold_truth = update_truth_section(
    gold_truth,
    "runtime_facts",
    {
        "stage_feature_summary": feature_set_summary,
    },
)

gold_truth = update_truth_section(
    gold_truth,
    "artifact_paths",
    {
        "reference_profile_path": str(REFERENCE_PROFILE_PATH),
        "stage1_features_path": str(STAGE1_FEATURES_PATH),
        "stage2_features_path": str(STAGE2_FEATURES_PATH),
        "stage3_primary_path": str(STAGE3_PRIMARY_PATH),
        "stage3_secondary_path": str(STAGE3_SECONDARY_PATH),
    },
)

ledger.add(
    kind="step",
    step="build_stage_feature_sets",
    message="Built feature sets for Stage 1, Stage 2, and Stage 3 and wrote the results to Truth Store.",
    data=feature_set_summary,
    logger=logger,
)

feature_set_summary

# %% [markdown]
# ----

# %% [markdown]
# ## Save Gold Split Artifacts
# 
# This step saves the Gold preprocessing outputs as separate parquet artifacts so downstream modeling notebooks can use a fixed and reproducible split.
# 
# The notebook already created a time-ordered split earlier in the workflow:
# 
# - The **train split** contains the earlier portion of the time-ordered Gold dataset.
# - The **test split** contains the later holdout portion of the time-ordered Gold dataset.
# - The **fit subset** contains only normal rows from the training split and is used for unsupervised model fitting.
# 
# Saving these artifacts ensures that the baseline, cascade, and comparison notebooks all use the same consistent partitions.

# %%
if "meta__is_train_flag" in gold_preprocessed_scaled_dataframe.columns:
    gold_preprocessed_scaled_dataframe["meta__split"] = np.where(
        gold_preprocessed_scaled_dataframe["meta__is_train_flag"].astype(bool),
        "train",
        "test",
    )
else:
    gold_preprocessed_scaled_dataframe["meta__split"] = pd.NA


gold_train_dataframe = gold_preprocessed_scaled_dataframe.loc[train_mask_flag].copy()
gold_test_dataframe  = gold_preprocessed_scaled_dataframe.loc[~train_mask_flag].copy()
gold_fit_dataframe   = training_rows_for_fit.copy()

gold_train_dataframe["meta__split"] = "train"
gold_test_dataframe["meta__split"] = "test"
gold_fit_dataframe["meta__split"] = "fit_normal_only"

gold_split_summary = {
    "train_rows": int(len(gold_train_dataframe)),
    "test_rows": int(len(gold_test_dataframe)),
    "fit_rows_normal_only": int(len(gold_fit_dataframe)),
}

gold_split_summary

# %%

gold_truth = update_truth_section(
    gold_truth,
    "runtime_facts",
    {
        "source_run_ids": (
            gold_preprocessed_scaled_dataframe["meta__run_id"].dropna().astype(str).unique().tolist()
            if "meta__run_id" in gold_preprocessed_scaled_dataframe.columns
            else []
        ),
        "split_summary": gold_split_summary,
    },
)

gold_truth = update_truth_section(
    gold_truth,
    "source_fingerprint",
    build_file_fingerprint(silver_path),
)

gold_truth = update_truth_section(
    gold_truth,
    "artifact_paths",
    {
        "silver_source_path": str(silver_path),
        "feature_registry_path": str(FEATURE_REGISTRY_PATH),
        "imputation_recommendation_path": str(IMPUTE_RECOMMENDATION_PATH),
        "gold_preprocessed_path": str(GOLD_PRESCALED_DATA_PATH),
        "gold_prescaled_path": str(GOLD_PRESCALED_DATA_PATH),
        "gold_scaled_path": str(GOLD_SCALED_DATA_PATH),
        "gold_train_path": str(GOLD_TRAIN_DATA_PATH),
        "gold_test_path": str(GOLD_TEST_DATA_PATH),
        "gold_fit_path": str(GOLD_FIT_DATA_PATH),
        "preprocessing_summary_path": str(PREPROCESSING_SUMMARY_PATH),
        "preprocessing_metadata_path": str(PREPROCESSING_METADATA_PATH),
        "reference_profile_path": str(REFERENCE_PROFILE_PATH),
        "stage1_features_path": str(STAGE1_FEATURES_PATH),
        "stage2_features_path": str(STAGE2_FEATURES_PATH),
        "stage3_primary_path": str(STAGE3_PRIMARY_PATH),
        "stage3_secondary_path": str(STAGE3_SECONDARY_PATH),
        "scaler_path": str(scaler_path),
    },
)

gold_truth = update_truth_section(
    gold_truth,
    "notes",
    {
        "purpose": "Gold preprocessing truth record",
    },
)

gold_truth["runtime_facts"]

# %%

gold_meta_columns = identify_meta_columns(gold_preprocessed_scaled_dataframe)
gold_meta_columns = sorted(set(gold_meta_columns + [
    "meta__truth_hash",
    "meta__parent_truth_hash",
    "meta__pipeline_mode",
]))

gold_feature_columns = identify_feature_columns(gold_preprocessed_scaled_dataframe)

gold_truth_record = build_truth_record(
    truth_base=gold_truth,
    row_count=len(gold_preprocessed_scaled_dataframe),
    column_count=gold_preprocessed_scaled_dataframe.shape[1] + 3,
    meta_columns=gold_meta_columns,
    feature_columns=gold_feature_columns,
)

GOLD_TRUTH_HASH = gold_truth_record["truth_hash"]

gold_preprocessed_prescaled_dataframe = stamp_truth_columns(
    gold_preprocessed_prescaled_dataframe,
    truth_hash=GOLD_TRUTH_HASH,
    parent_truth_hash=GOLD_PARENT_TRUTH_HASH,
    pipeline_mode=PIPELINE_MODE,
)

gold_preprocessed_scaled_dataframe = stamp_truth_columns(
    gold_preprocessed_scaled_dataframe,
    truth_hash=GOLD_TRUTH_HASH,
    parent_truth_hash=GOLD_PARENT_TRUTH_HASH,
    pipeline_mode=PIPELINE_MODE,
)

gold_train_dataframe = stamp_truth_columns(
    gold_train_dataframe,
    truth_hash=GOLD_TRUTH_HASH,
    parent_truth_hash=GOLD_PARENT_TRUTH_HASH,
    pipeline_mode=PIPELINE_MODE,
)

gold_test_dataframe = stamp_truth_columns(
    gold_test_dataframe,
    truth_hash=GOLD_TRUTH_HASH,
    parent_truth_hash=GOLD_PARENT_TRUTH_HASH,
    pipeline_mode=PIPELINE_MODE,
)

gold_fit_dataframe = stamp_truth_columns(
    gold_fit_dataframe,
    truth_hash=GOLD_TRUTH_HASH,
    parent_truth_hash=GOLD_PARENT_TRUTH_HASH,
    pipeline_mode=PIPELINE_MODE,
)

gold_truth_path = save_truth_record(
    gold_truth_record,
    truth_dir=TRUTHS_PATH,
    dataset_name=DATASET_NAME,
    layer_name=LAYER_NAME,
)

append_truth_index(
    gold_truth_record,
    truth_index_path=TRUTH_INDEX_PATH,
)

print("Gold truth hash:", GOLD_TRUTH_HASH)
print("Gold truth path:", gold_truth_path)

# %%

gold_preprocessed_prescaled_path = save_data(
    gold_preprocessed_prescaled_dataframe,
    GOLD_PRESCALED_DATA_PATH.parent,
    GOLD_PRESCALED_DATA_PATH.name,
)

gold_preprocessed_scaled_path = save_data(
    gold_preprocessed_scaled_dataframe,
    GOLD_SCALED_DATA_PATH.parent,
    GOLD_SCALED_DATA_PATH.name,
)

gold_train_path = save_data(
    gold_train_dataframe,
    GOLD_TRAIN_DATA_PATH.parent,
    GOLD_TRAIN_DATA_PATH.name,
)

gold_test_path = save_data(
    gold_test_dataframe,
    GOLD_TEST_DATA_PATH.parent,
    GOLD_TEST_DATA_PATH.name,
)

gold_fit_path = save_data(
    gold_fit_dataframe,
    GOLD_FIT_DATA_PATH.parent,
    GOLD_FIT_DATA_PATH.name,
)

wandb.save(str(gold_preprocessed_prescaled_path))
wandb.save(str(gold_preprocessed_scaled_path))
wandb.save(str(gold_train_path))
wandb.save(str(gold_test_path))
wandb.save(str(gold_fit_path))

ledger.add(
    kind="step",
    step="finalize_gold_truth_and_save_artifacts",
    message="Finalized Gold truth, stamped lineage columns, and saved prescaled/scaled/train/test/fit outputs.",
    data={
        "gold_truth_hash": GOLD_TRUTH_HASH,
        "gold_parent_truth_hash": GOLD_PARENT_TRUTH_HASH,
        "gold_truth_path": str(gold_truth_path),
        "gold_prescaled_path": str(gold_preprocessed_prescaled_path),
        "gold_scaled_path": str(gold_preprocessed_scaled_path),
        "gold_train_path": str(gold_train_path),
        "gold_test_path": str(gold_test_path),
        "gold_fit_path": str(gold_fit_path),
        "pipeline_mode": PIPELINE_MODE,
        "process_run_id": GOLD_PROCESS_RUN_ID,
        **gold_split_summary,
    },
    logger=logger,
)

pd.DataFrame(
    [
        {
            "split": "train",
            "rows": int(len(gold_train_dataframe)),
            "abnormal_rows": int(gold_train_dataframe["anomaly_flag"].sum()) if "anomaly_flag" in gold_train_dataframe.columns else None,
            "path": str(gold_train_path),
        },
        {
            "split": "test",
            "rows": int(len(gold_test_dataframe)),
            "abnormal_rows": int(gold_test_dataframe["anomaly_flag"].sum()) if "anomaly_flag" in gold_test_dataframe.columns else None,
            "path": str(gold_test_path),
        },
        {
            "split": "fit_normal_only",
            "rows": int(len(gold_fit_dataframe)),
            "abnormal_rows": int(gold_fit_dataframe["anomaly_flag"].sum()) if "anomaly_flag" in gold_fit_dataframe.columns else None,
            "path": str(gold_fit_path),
        },
    ]
)

# %%
preprocessing_summary = {
    "gold_prescaled_path": str(GOLD_PRESCALED_DATA_PATH),
    "gold_scaled_path": str(GOLD_SCALED_DATA_PATH),
    "gold_scaled_shape": list(gold_preprocessed_scaled_dataframe.shape),
    "feature_count": int(len(numeric_feature_columns)),
    "stage1_feature_count": int(len(stage1_feature_columns)),
    "stage2_feature_count": int(len(stage2_feature_columns)),
    "stage3_primary_rule_count": int(len(stage3_primary_rule_sensors)),
    "stage3_secondary_rule_count": int(len(stage3_secondary_rule_sensors)),
    "imputation_method": str(imputation_info.get("imputation_method")),
    "grouping_columns": list(imputation_info.get("grouping_columns", [])),
    "ordering_column": imputation_info.get("ordering_column"),
    "scaler_kind": str(SCALER_KIND),
    "train_fraction": float(TRAIN_FRACTION),
    "train_rows": int(len(gold_train_dataframe)),
    "test_rows": int(len(gold_test_dataframe)),
    "fit_rows_normal_only": int(len(gold_fit_dataframe)),
    "gold_train_path": str(gold_train_path),
    "gold_test_path": str(gold_test_path),
    "gold_fit_path": str(gold_fit_path),
    "gold_truth_hash": GOLD_TRUTH_HASH,
    "gold_parent_truth_hash": GOLD_PARENT_TRUTH_HASH,
    "gold_truth_path": str(gold_truth_path),
    "process_run_id": GOLD_PROCESS_RUN_ID,
}

save_json(preprocessing_summary, PREPROCESSING_SUMMARY_PATH)

preprocessing_metadata = {
    "recipe_id": RECIPE_ID,
    "gold_version": GOLD_VERSION,
    "dataset_name": DATASET_NAME,
    "feature_set_source": str(FEATURE_REGISTRY_PATH),
    "imputation_recommendation_source": str(IMPUTE_RECOMMENDATION_PATH),
    "gold_prescaled_output_path": str(GOLD_PRESCALED_DATA_PATH),
    "gold_scaled_output_path": str(GOLD_SCALED_DATA_PATH),
    "reference_profile_path": str(REFERENCE_PROFILE_PATH),
    "stage1_features_path": str(STAGE1_FEATURES_PATH),
    "stage2_features_path": str(STAGE2_FEATURES_PATH),
    "stage3_primary_path": str(STAGE3_PRIMARY_PATH),
    "stage3_secondary_path": str(STAGE3_SECONDARY_PATH),
    "preprocessor_scaler_path": str(scaler_path) if SCALER_KIND else None,
    "gold_train_path": str(gold_train_path),
    "gold_test_path": str(gold_test_path),
    "gold_fit_path": str(gold_fit_path),
    "gold_truth_hash": GOLD_TRUTH_HASH,
    "gold_parent_truth_hash": GOLD_PARENT_TRUTH_HASH,
    "gold_truth_path": str(gold_truth_path),
    "process_run_id": GOLD_PROCESS_RUN_ID,
}

save_json(preprocessing_metadata, PREPROCESSING_METADATA_PATH)

wandb.save(str(PREPROCESSING_SUMMARY_PATH))
wandb.save(str(PREPROCESSING_METADATA_PATH))

ledger.add(
    kind="step",
    step="save_preprocessing_outputs",
    message="Saved Gold preprocessing summary and metadata artifacts.",
    data={
        "preprocessing_summary_path": str(PREPROCESSING_SUMMARY_PATH),
        "preprocessing_metadata_path": str(PREPROCESSING_METADATA_PATH),
        "gold_truth_hash": GOLD_TRUTH_HASH,
        "gold_truth_path": str(gold_truth_path),
    },
    logger=logger,
)

{
    "preprocessing_summary_path": str(PREPROCESSING_SUMMARY_PATH),
    "preprocessing_metadata_path": str(PREPROCESSING_METADATA_PATH),
    "gold_truth_hash": GOLD_TRUTH_HASH,
    "gold_truth_path": str(gold_truth_path),
}

# %% [markdown]
# ----

# %%
gold_preprocesssing_ledger_path = GOLD_ARTIFACTS_PATH / GOLD_PREPROCESSING_LEDGER_FILE_NAME
ledger.write_json(gold_preprocesssing_ledger_path)

wandb.save(str(gold_preprocesssing_ledger_path))
wandb_run.finish()

# %% [markdown]
# ----

# %%
# Gold Preprocessing Sanity Checks


print("Running Gold preprocessing sanity checks...\n")

assert "gold_preprocessed_scaled_dataframe" in locals(), "gold_preprocessed_scaled_dataframe is missing."
assert "gold_preprocessed_prescaled_dataframe" in locals(), "gold_preprocessed_prescaled_dataframe is missing."
assert "train_mask_flag" in locals(), "train_mask_flag is missing."
assert "numeric_feature_columns" in locals(), "numeric_feature_columns is missing."
assert "gold_train_dataframe" in locals(), "gold_train_dataframe is missing."
assert "gold_test_dataframe" in locals(), "gold_test_dataframe is missing."
assert "gold_fit_dataframe" in locals(), "gold_fit_dataframe is missing."
assert "gold_truth_record" in locals(), "gold_truth_record is missing."
assert "GOLD_TRUTH_HASH" in locals(), "GOLD_TRUTH_HASH is missing."

dataframe_scaled = gold_preprocessed_scaled_dataframe
dataframe_prescaled = gold_preprocessed_prescaled_dataframe
n_rows, n_cols = dataframe_scaled.shape

print(f"- Scaled Gold shape: {n_rows} rows x {n_cols} columns")

train_mask_flag = train_mask_flag.astype(bool)

assert len(train_mask_flag) == n_rows, "train_mask_flag length does not match Gold dataframe."
assert len(gold_train_dataframe) == train_mask_flag.sum(), "Train split row count mismatch."
assert len(gold_test_dataframe) == (~train_mask_flag).sum(), "Test split row count mismatch."

print(f"- Train rows: {len(gold_train_dataframe)}, Test rows: {len(gold_test_dataframe)}")

if "anomaly_flag" in dataframe_scaled.columns:
    fit_mask_expected = train_mask_flag & (dataframe_scaled["anomaly_flag"] == 0)
    assert len(gold_fit_dataframe) == fit_mask_expected.sum(), "Fit (normal-only) row count mismatch."
    assert gold_fit_dataframe["anomaly_flag"].sum() == 0, "Fit dataframe contains anomalous rows."
    print(f"- Fit (normal-only) rows: {len(gold_fit_dataframe)} (anomaly_flag == 0 confirmed)")
else:
    print("- anomaly_flag not present, skipping normal-only checks for fit dataframe.")

for column in ["meta__is_train_flag", "meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode"]:
    assert column in dataframe_scaled.columns, f"Expected meta column '{column}' is missing."

print("- Meta columns present: meta__is_train_flag, meta__truth_hash, meta__parent_truth_hash, meta__pipeline_mode")

assert dataframe_prescaled.shape[0] == dataframe_scaled.shape[0], "Prescaled and scaled row counts differ."

diffs = []
for column in numeric_feature_columns:
    if column not in dataframe_prescaled.columns or column not in dataframe_scaled.columns:
        continue
    sample_scaled = dataframe_scaled[column].head(100).to_numpy()
    sample_prescaled = dataframe_prescaled[column].head(100).to_numpy()
    if not np.allclose(sample_scaled, sample_prescaled):
        diffs.append(column)

assert len(diffs) > 0, "No numeric features appear to have changed after scaling."
print(f"- Scaling changed {len(diffs)} numeric feature(s) (good).")

assert Path(gold_truth_path).exists(), "Gold truth file was not saved."
assert gold_truth_record["truth_hash"] == GOLD_TRUTH_HASH, "Truth hash mismatch between record and runtime variable."
assert gold_truth_record["parent_truth_hash"] == GOLD_PARENT_TRUTH_HASH, "Parent truth hash mismatch."

truth_runtime_facts = gold_truth_record.get("runtime_facts", {})
truth_artifact_paths = gold_truth_record.get("artifact_paths", {})

assert "scaler_path" in truth_artifact_paths or "scaler_path" in truth_runtime_facts, "Scaler path missing from truth record."
assert "split_summary" in truth_runtime_facts, "Split summary missing from truth record."
assert "stage_feature_summary" in truth_runtime_facts, "Stage feature summary missing from truth record."

print("- Gold truth record exists and contains scaler/split/stage metadata.")
print("\nAll Gold preprocessing sanity checks passed.")

# %% [markdown]
# ----

# %%
gold_preprocessed_prescaled_dataframe.shape

# %% [markdown]
# ----

# %%
gold_preprocessed_scaled_dataframe.shape

# %% [markdown]
# ----

# %%

# Get column lists
prescaled_columns = gold_preprocessed_prescaled_dataframe.columns.tolist()
scaled_columns = gold_preprocessed_scaled_dataframe.columns.tolist()

print(f"Prescaled Columns columns: {prescaled_columns}")
print(f"Scaled columns: {scaled_columns}\n")

# Find columns unique to each
unique_to_prescaled = set(prescaled_columns) - set(scaled_columns)
unique_to_scaled = set(scaled_columns) - set(prescaled_columns)

print(f"Columns in dataframe1 but not in dataframe2: {unique_to_prescaled}")
print(f"Columns in dataframe2 but not in dataframe1: {unique_to_scaled}\n")

# Find all columns that are not in both (symmetric difference)
columns_not_in_both = set(prescaled_columns).symmetric_difference(set(scaled_columns))
print(f"All columns not present in both dataframes: {columns_not_in_both}")

# %% [markdown]
# ----

# %%
required_gold_meta_columns = [
    "meta__truth_hash",
    "meta__parent_truth_hash",
    "meta__pipeline_mode",
]

gold_frames_to_check = {
    "gold_preprocessed_prescaled": gold_preprocessed_prescaled_dataframe,
    "gold_preprocessed_scaled": gold_preprocessed_scaled_dataframe,
    "gold_fit": gold_fit_dataframe,
    "gold_train": gold_train_dataframe,
    "gold_test": gold_test_dataframe,
}

for frame_name, frame_value in gold_frames_to_check.items():
    missing_gold_meta_columns = [
        column_name
        for column_name in required_gold_meta_columns
        if column_name not in frame_value.columns
    ]
    if missing_gold_meta_columns:
        raise ValueError(
            f"{frame_name} is missing required lineage columns: {missing_gold_meta_columns}"
        )

    frame_truth_hash = extract_truth_hash(frame_value)
    if frame_truth_hash is None:
        raise ValueError(f"{frame_name} does not contain a readable meta__truth_hash value.")

    if frame_truth_hash != GOLD_TRUTH_HASH:
        raise ValueError(
            f"{frame_name} truth hash does not match GOLD_TRUTH_HASH:\n"
            f"dataframe={frame_truth_hash}\n"
            f"record={GOLD_TRUTH_HASH}"
        )

    frame_parent_values = frame_value["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()
    if not frame_parent_values:
        raise ValueError(f"{frame_name} is missing populated meta__parent_truth_hash values.")

    if len(frame_parent_values) != 1:
        raise ValueError(f"{frame_name} has multiple parent truth hashes: {frame_parent_values}")

    if frame_parent_values[0] != GOLD_PARENT_TRUTH_HASH:
        raise ValueError(
            f"{frame_name} parent truth hash does not match GOLD_PARENT_TRUTH_HASH:\n"
            f"dataframe_parent={frame_parent_values[0]}\n"
            f"gold_parent_truth={GOLD_PARENT_TRUTH_HASH}"
        )

if not Path(gold_truth_path).exists():
    raise FileNotFoundError(f"Gold truth file was not created: {gold_truth_path}")

loaded_gold_truth = load_json(gold_truth_path)

if loaded_gold_truth.get("truth_hash") != GOLD_TRUTH_HASH:
    raise ValueError(
        "Saved Gold truth file hash does not match GOLD_TRUTH_HASH:\n"
        f"file={loaded_gold_truth.get('truth_hash')}\n"
        f"record={GOLD_TRUTH_HASH}"
    )

if loaded_gold_truth.get("parent_truth_hash") != GOLD_PARENT_TRUTH_HASH:
    raise ValueError(
        "Saved Gold truth file parent hash does not match GOLD_PARENT_TRUTH_HASH:\n"
        f"truth={loaded_gold_truth.get('parent_truth_hash')}\n"
        f"gold_parent={GOLD_PARENT_TRUTH_HASH}"
    )

print("Gold PreProcessing lineage sanity check passed.")

# %% [markdown]
# ----


