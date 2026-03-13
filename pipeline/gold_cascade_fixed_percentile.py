# %% [markdown]
# ## Gold Cascade Modeling (Deliverable 1.3.3)
# 
# This notebook implements the full three-stage cascade anomaly detection pipeline. The objective is to determine whether a layered approach improves alert quality when compared to the single-model baseline.
# 
# **Purpose:**  
# To operationalize the cascade architecture defined in Section C, generating alert outputs for each stage of the cascade: broad detection (Stage 1), refined detection (Stage 2), and rule/profile/historical confirmation (Stage 3).
# 
# **Stages Implemented:**
# 
# 1. **Stage 1 — Broad Isolation Forest**  
#    High-sensitivity anomaly screen using all 50 Gold numeric features.
# 
# 2. **Stage 2 — Narrow Isolation Forest**  
#    Secondary detector trained on the reduced feature subset identified during Silver EDA.
# 
# 3. **Stage 3 — Rule / Profile / Historical Confirmation**  
#    Final confirmation based on behavior profiles, persistence checks, drift features, and cross-sensor consistency.
# 
# **Key Goals:**
# 
# - Load the Gold preprocessed dataset and Gold feature artifacts.
# - Train Stage 1 and Stage 2 Isolation Forest models.
# - Apply Stage 3 rule/profile confirmation logic based on Silver EDA outputs.
# - Generate and store alert outputs for all three cascade stages.
# - Export all cascade artifacts for comparative evaluation.
# 
# **Relevance to Section C:**  
# This notebook produces the layered alert outputs required for evaluating the cascade’s effect on false positives, noisy alerts, and anomaly sensitivity. These outputs are necessary for the statistical tests, alert-volume comparisons, and visual communication described in Section C.

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

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

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

CONFIG_ROOT = paths.configs
CONFIG_RUN_MODE = "train"
CONFIG_PROFILE = "default"

CONFIG = load_pipeline_config(
    config_root=CONFIG_ROOT,
    stage="gold_cascade",
    dataset="pump",
    mode=CONFIG_RUN_MODE,
    profile=CONFIG_PROFILE,
    project_root=paths.root,
).data

GOLD_CFG = CONFIG["gold_cascade"]
PATHS = CONFIG["resolved_paths"]
FILENAMES = CONFIG["filenames"]
PIPELINE = CONFIG.get(
    "pipeline",
    {
        "execution_mode": "batch",
        "orchestration_mode": "notebook",
    },
)

TRUTH_CONFIG = build_truth_config_block(CONFIG)
TRUTH_CONFIG["pipeline"] = PIPELINE

# Stage details
STAGE = "gold"
LAYER_NAME = GOLD_CFG["layer_name"]
GOLD_VERSION = CONFIG["versions"]["gold"]
RECIPE_ID = GOLD_CFG["recipe_id"]
TRUTH_VERSION = CONFIG["versions"]["truth"]
PIPELINE_MODE = PIPELINE["execution_mode"]
RUN_MODE = CONFIG["runtime"]["mode"]

DATASET_NAME_CONFIG = CONFIG["dataset"]["name"]
DATASET_NAME = str(DATASET_NAME_CONFIG).strip().lower()

GOLD_PROCESS_RUN_ID = make_process_run_id(GOLD_CFG["process_run_id_prefix"])

# Weights and Biases
WANDB_PROJECT = CONFIG["wandb"]["project"]
WANDB_ENTITY = CONFIG["wandb"]["entity"]
WANDB_RUN_NAME = f"{GOLD_VERSION}"

# File names
GOLD_PREPROCESSED_FILE_NAME = FILENAMES["gold_preprocessed_file_name"]
GOLD_PREPROCESSED_SCALED_FILE_NAME = FILENAMES["gold_preprocessed_scaled_file_name"]

GOLD_FIT_FILE_NAME = FILENAMES["gold_fit_file_name"]
GOLD_TRAIN_FILE_NAME = FILENAMES["gold_train_file_name"]
GOLD_TEST_FILE_NAME = FILENAMES["gold_test_file_name"]

STAGE1_FEATURES_FILE_NAME = FILENAMES["stage1_features_file_name"]
STAGE2_FEATURES_FILE_NAME = FILENAMES["stage2_features_file_name"]
STAGE3_PRIMARY_FILE_NAME = FILENAMES["stage3_primary_file_name"]
STAGE3_SECONDARY_FILE_NAME = FILENAMES["stage3_secondary_file_name"]

STAGE1_MODEL_FILE_NAME = FILENAMES["stage1_model_file_name"]
STAGE2_MODEL_FILE_NAME = FILENAMES["stage2_model_file_name"]

CASCADE_THRESHOLDS_FILE_NAME = FILENAMES["cascade_thresholds_file_name"]
CASCADE_SUMMARY_FILE_NAME = FILENAMES["cascade_summary_file_name"]
CASCADE_METADATA_FILE_NAME = FILENAMES["cascade_metadata_file_name"]

CASCADE_RESULTS_FILE_NAME_CSV = FILENAMES["cascade_results_file_name_csv"]
CASCADE_RESULTS_FILE_NAME_PICKLE = FILENAMES["cascade_results_file_name_pickle"]

CASCADE_REFERENCE_PROFILE_FILE_NAME = FILENAMES["cascade_reference_profile_file_name"]

STAGE1_CFG = GOLD_CFG["stage1"]
STAGE2_CFG = GOLD_CFG["stage2"]
STAGE3_CFG = GOLD_CFG["stage3"]

RANDOM_SEED = int(GOLD_CFG["random_seed"])

STAGE1_ESTIMATOR_COUNT = int(STAGE1_CFG["estimator_count"])
STAGE1_THRESHOLD_PERCENTILE = float(STAGE1_CFG["threshold_percentile"])

# Fixed-only notebook:
STAGE2_SELECTION_MODE = "fixed"
STAGE2_RANDOM_STATE = int(STAGE2_CFG.get("random_state", RANDOM_SEED))
STAGE2_FIXED_PARAMS = dict(STAGE2_CFG["fixed"]["params"])
STAGE2_FIXED_THRESHOLD_PERCENTILE = float(STAGE2_CFG["fixed"]["threshold_percentile"])

STAGE3_MIN_PRIMARY_SENSOR_HITS = int(STAGE3_CFG["min_primary_sensor_hits"])
STAGE3_MIN_SECONDARY_SENSOR_HITS = int(STAGE3_CFG["min_secondary_sensor_hits"])
STAGE3_ROLLING_WINDOW_SIZE = int(STAGE3_CFG["rolling_window_size"])
STAGE3_MINIMUM_FLAGS_IN_WINDOW = int(STAGE3_CFG["minimum_flags_in_window"])

GOLD_CASCADE_LEDGER_FILE_NAME = FILENAMES["gold_cascade_ledger_file_name"]

set_wandb_dir_from_config(CONFIG)


GOLD_PREPROCESSED_DATA_PATH = Path(PATHS["gold_preprocessed_data_path"])
GOLD_PREPROCESSED_SCALED_DATA_PATH = Path(PATHS["gold_preprocessed_scaled_data_path"])

GOLD_TRAIN_DATA_PATH = Path(PATHS["gold_train_data_path"])
GOLD_TEST_DATA_PATH = Path(PATHS["gold_test_data_path"])
GOLD_FIT_DATA_PATH = Path(PATHS["gold_fit_data_path"])
GOLD_ARTIFACTS_PATH = Path(PATHS["gold_artifacts_dir"])

STAGE1_FEATURES_PATH = Path(PATHS["stage1_features_path"])
STAGE2_FEATURES_PATH = Path(PATHS["stage2_features_path"])
STAGE3_PRIMARY_PATH = Path(PATHS["stage3_primary_path"])
STAGE3_SECONDARY_PATH = Path(PATHS["stage3_secondary_path"])


MODELS_PATH = Path(PATHS["models_root"])

STAGE1_MODELS_PATH = Path(PATHS["stage1_models_path"])
STAGE1_MODEL_ARTIFACT_PATH = Path(PATHS["stage1_model_artifact_path"])

STAGE2_MODELS_PATH = Path(PATHS["stage2_models_path"])
STAGE2_MODEL_ARTIFACT_PATH = Path(PATHS["stage2_model_artifact_path"])

CASCADE_RESULTS_PATH_CSV = Path(PATHS["cascade_results_path_csv"])
CASCADE_RESULTS_PATH_PICKLE = Path(PATHS["cascade_results_path_pickle"])

CASCADE_THRESHOLDS_PATH = Path(PATHS["cascade_thresholds_path"])
CASCADE_SUMMARY_PATH = Path(PATHS["cascade_summary_path"])
CASCADE_METADATA_PATH = Path(PATHS["cascade_metadata_path"])

CASCADE_REFERENCE_PROFILE_PATH = Path(PATHS["cascade_reference_profile_path"])

# Logs
LOGS_PATH = Path(PATHS["logs_root"])

# Truths
TRUTHS_PATH = Path(PATHS["truths_dir"])
TRUTH_INDEX_PATH = Path(PATHS["truth_index_path"])

# Path Failsafes

GOLD_PREPROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
GOLD_PREPROCESSED_SCALED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
GOLD_FIT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
GOLD_TEST_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
GOLD_TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
GOLD_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)
TRUTHS_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)



# %% [markdown]
# ----

# %%
# Logging Setup

# Create gold log path 
gold_log_path = paths.logs / "gold_modeling_cascade.log"

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
logger.info("Gold Modeling stage starting")

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
    job_type="gold_modeling_cascade",
    config={
        "gold_version": GOLD_VERSION,
        "dataset": DATASET_NAME,
        "stage": STAGE,
        "stage1_threshold_percentile": STAGE1_THRESHOLD_PERCENTILE,
        "stage2_selection_mode": STAGE2_SELECTION_MODE,
        "stage2_fixed_threshold_percentile": STAGE2_FIXED_THRESHOLD_PERCENTILE,
        "stage3_min_primary_sensor_hits": STAGE3_MIN_PRIMARY_SENSOR_HITS,
        "stage3_min_secondary_sensor_hits": STAGE3_MIN_SECONDARY_SENSOR_HITS,
        "stage3_rolling_window_size": STAGE3_ROLLING_WINDOW_SIZE,
        "stage3_minimum_flags_in_window": STAGE3_MINIMUM_FLAGS_IN_WINDOW,
        "gold_fit_path": str(GOLD_FIT_DATA_PATH),
        "gold_scored_path": str(GOLD_PREPROCESSED_SCALED_DATA_PATH),
        "stage1_features_path": str(STAGE1_FEATURES_PATH),
        "stage2_features_path": str(STAGE2_FEATURES_PATH),
        "stage3_primary_path": str(STAGE3_PRIMARY_PATH),
        "stage3_secondary_path": str(STAGE3_SECONDARY_PATH),
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
logger.info("Loading Gold Preprocessed parquet: %s", GOLD_PREPROCESSED_SCALED_DATA_PATH)
gold_preprocessed_scaled_dataframe = load_data(GOLD_PREPROCESSED_SCALED_DATA_PATH)

GOLD_DATASET_NAME = (
    gold_preprocessed_scaled_dataframe["meta__dataset"]
    .dropna()
    .astype("string")
    .str.strip()
)
GOLD_DATASET_NAME = GOLD_DATASET_NAME[GOLD_DATASET_NAME != ""]

if len(GOLD_DATASET_NAME) == 0:
    raise ValueError("Gold cascade input dataframe is missing usable meta__dataset values.")

GOLD_DATASET_NAME = str(GOLD_DATASET_NAME.iloc[0]).strip()

gold_truth = load_parent_truth_record_from_dataframe(
    dataframe=gold_preprocessed_scaled_dataframe,
    truth_dir=TRUTHS_PATH,
    parent_layer_name="gold",
    dataset_name=GOLD_DATASET_NAME,
    column_name="meta__truth_hash",
)

DATASET_NAME = get_dataset_name_from_truth(gold_truth)
GOLD_PARENT_TRUTH_HASH = get_truth_hash(gold_truth)

PARENT_PIPELINE_MODE = get_pipeline_mode_from_truth(gold_truth)
if PARENT_PIPELINE_MODE is not None:
    PIPELINE_MODE = PARENT_PIPELINE_MODE

GOLD_TRUTH_PATH = (
    TRUTHS_PATH
    / "gold"
    / f"{DATASET_NAME}__gold__truth__{GOLD_PARENT_TRUTH_HASH}.json"
)

gold_truth_runtime_facts = gold_truth.get("runtime_facts", {})
gold_truth_artifact_paths = gold_truth.get("artifact_paths", {})

GOLD_PREPROCESSED_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_preprocessed_path", str(GOLD_PREPROCESSED_DATA_PATH)))
GOLD_FIT_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_fit_path", str(GOLD_FIT_DATA_PATH)))
GOLD_TEST_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_test_path", str(GOLD_TEST_DATA_PATH)))
GOLD_TRAIN_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_train_path", str(GOLD_TRAIN_DATA_PATH)))
STAGE1_FEATURES_PATH = Path(gold_truth_artifact_paths.get("stage1_features_path", str(STAGE1_FEATURES_PATH)))
STAGE2_FEATURES_PATH = Path(gold_truth_artifact_paths.get("stage2_features_path", str(STAGE2_FEATURES_PATH)))
STAGE3_PRIMARY_PATH = Path(gold_truth_artifact_paths.get("stage3_primary_path", str(STAGE3_PRIMARY_PATH)))
STAGE3_SECONDARY_PATH = Path(gold_truth_artifact_paths.get("stage3_secondary_path", str(STAGE3_SECONDARY_PATH)))

logger.info("Resolved Gold cascade dataset name from Gold truth: %s", DATASET_NAME)
logger.info("Resolved Gold truth path: %s", GOLD_TRUTH_PATH)

print("Gold cascade dataset name from parent truth:", DATASET_NAME)
print("Gold cascade parent truth hash:", GOLD_PARENT_TRUTH_HASH)

logger.info("Loading Gold Preprocessed parquet: %s", GOLD_PREPROCESSED_DATA_PATH)
gold_preprocessed_dataframe = load_data(GOLD_PREPROCESSED_DATA_PATH)

logger.info("Loading Gold fit parquet: %s", GOLD_FIT_DATA_PATH)
gold_fit_dataframe = load_data(GOLD_FIT_DATA_PATH)

logger.info("Loading Gold test parquet: %s", GOLD_TEST_DATA_PATH)
gold_test_dataframe = load_data(GOLD_TEST_DATA_PATH)

logger.info("Loading Gold train parquet: %s", GOLD_TRAIN_DATA_PATH)
gold_train_dataframe = load_data(GOLD_TRAIN_DATA_PATH)

logger.info("Loading Stage 1 features: %s", STAGE1_FEATURES_PATH)
stage1_feature_columns = load_json(STAGE1_FEATURES_PATH)

logger.info("Loading Stage 2 features: %s", STAGE2_FEATURES_PATH)
stage2_feature_columns = load_json(STAGE2_FEATURES_PATH)

logger.info("Loading Stage 3 primary rule sensors: %s", STAGE3_PRIMARY_PATH)
stage3_primary_rule_sensors = load_json(STAGE3_PRIMARY_PATH)

logger.info("Loading Stage 3 secondary rule sensors: %s", STAGE3_SECONDARY_PATH)
stage3_secondary_rule_sensors = load_json(STAGE3_SECONDARY_PATH)

ledger.add(
    kind="step",
    step="load_modeling_inputs",
    message="Loaded Gold scaled parquet, loaded Gold truth, substituted truth-linked artifact paths, then loaded cascade inputs.",
    data={
        "gold_scaled_path": str(GOLD_PREPROCESSED_SCALED_DATA_PATH),
        "gold_truth_hash": GOLD_PARENT_TRUTH_HASH,
        "gold_truth_path": str(GOLD_TRUTH_PATH),
        "gold_preprocessed_path": str(GOLD_PREPROCESSED_DATA_PATH),
        "gold_fit_path": str(GOLD_FIT_DATA_PATH),
        "gold_test_path": str(GOLD_TEST_DATA_PATH),
        "gold_train_path": str(GOLD_TRAIN_DATA_PATH),
        "stage1_features_path": str(STAGE1_FEATURES_PATH),
        "stage2_features_path": str(STAGE2_FEATURES_PATH),
        "stage3_primary_path": str(STAGE3_PRIMARY_PATH),
        "stage3_secondary_path": str(STAGE3_SECONDARY_PATH),
        "gold_preprocessed_shape": list(gold_preprocessed_dataframe.shape),
        "gold_scaled_shape": list(gold_preprocessed_scaled_dataframe.shape),
        "gold_fit_shape": list(gold_fit_dataframe.shape),
        "gold_test_shape": list(gold_test_dataframe.shape),
        "gold_train_shape": list(gold_train_dataframe.shape),
        "stage1_feature_count": int(len(stage1_feature_columns)),
        "stage2_feature_count": int(len(stage2_feature_columns)),
        "stage3_primary_count": int(len(stage3_primary_rule_sensors)),
        "stage3_secondary_count": int(len(stage3_secondary_rule_sensors)),
    },
    logger=logger,
)

gold_test_dataframe.head(3)

# %% [markdown]
# ----

# %%
# Masks
if "meta__is_train_flag" not in gold_preprocessed_scaled_dataframe.columns:
    raise ValueError("meta__is_train_flag missing from gold_preprocessed_scaled_dataframe. "
                     "Gold preprocessing must stamp it before saving.")

train_mask = gold_preprocessed_scaled_dataframe["meta__is_train_flag"].astype(bool)
test_mask = ~train_mask

logger.info("Split counts: all=%d train=%d test=%d", len(train_mask), int(train_mask.sum()), int(test_mask.sum()))

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
reference_profile_features = list(dict.fromkeys(
    stage1_feature_columns + stage3_primary_rule_sensors + stage3_secondary_rule_sensors
))

reference_profile = build_reference_profile(
    gold_fit_dataframe,
    feature_columns=reference_profile_features,
)

ledger.add(
    kind="step",
    step="build_reference_profile",
    message="Built fit-period reference profile for Stage 3 confirmation logic.",
    data={
        "training_rows": int(len(gold_fit_dataframe)),
        "reference_feature_count": int(len(reference_profile_features)),
    },
    logger=logger,
)

reference_profile.head(10)

# %% [markdown]
# ----

# %%
# Fit features from normal-only fit parquet
stage1_train_fit_features = gold_fit_dataframe[stage1_feature_columns].values
stage2_train_fit_features = gold_fit_dataframe[stage2_feature_columns].values

# Score features from the full scaled dataset (ALL rows)
stage1_all_features = gold_preprocessed_scaled_dataframe[stage1_feature_columns].values
stage2_all_features = gold_preprocessed_scaled_dataframe[stage2_feature_columns].values


test_labels = None

if "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns:
    test_labels = (
        gold_preprocessed_scaled_dataframe
        .loc[test_mask, "anomaly_flag"]
        .fillna(0)
        .astype(int)
        .values
    )

# %% [markdown]
# ----

# %%
def compute_anomaly_scores_isolation_forest(
    model: IsolationForest,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    scores = model.score_samples(feature_matrix)
    anomaly_scores = -scores
    return anomaly_scores

# %%
def choose_threshold_by_percentile(
    anomaly_scores: np.ndarray,
    percentile: float,
) -> float:
    return float(np.percentile(anomaly_scores, percentile))

# %%

def evaluate_against_labels(
    true_labels: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
) -> dict:
    predicted_labels = (anomaly_scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="binary",
        zero_division=0,
    )

    results = {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if len(np.unique(true_labels)) == 2:
        results["roc_auc"] = float(roc_auc_score(true_labels, anomaly_scores))
        results["pr_auc"] = float(average_precision_score(true_labels, anomaly_scores))
    else:
        results["roc_auc"] = None
        results["pr_auc"] = None

    return results

# %% [markdown]
# ----

# %%
stage1_model = IsolationForest(
    n_estimators=STAGE1_ESTIMATOR_COUNT,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

stage1_model.fit(stage1_train_fit_features)

stage1_train_scores = compute_anomaly_scores_isolation_forest(
    stage1_model, 
    stage1_train_fit_features
)

stage1_all_scores = compute_anomaly_scores_isolation_forest(
    stage1_model, 
    stage1_all_features
)

stage1_threshold = choose_threshold_by_percentile(
    stage1_train_scores, 
    STAGE1_THRESHOLD_PERCENTILE
)

stage1_flags = (stage1_all_scores >= stage1_threshold).astype(int)

stage1_summary = {
    "threshold_percentile": float(STAGE1_THRESHOLD_PERCENTILE),
    "threshold": float(stage1_threshold),
    "alert_count_all_rows": int(stage1_flags.sum()),
    "alert_count_test_rows": int(stage1_flags[test_mask.values].sum()),
}

ledger.add(
    kind="step",
    step="run_cascade_stage1",
    message="Ran Stage 1 broad Isolation Forest using saved Gold fit data and scored all rows of the scaled dataset.",
    data={
        "estimator_count": int(STAGE1_ESTIMATOR_COUNT),
        "threshold_percentile": float(STAGE1_THRESHOLD_PERCENTILE),
        "threshold": float(stage1_threshold),
        "feature_count": int(len(stage1_feature_columns)),
        "alert_count_all_rows": int(stage1_summary["alert_count_all_rows"]),
        "alert_count_test_rows": int(stage1_summary["alert_count_test_rows"]),
    },
    logger=logger,
)

# %% [markdown]
# ----

# %%
stage2_model = IsolationForest(
    random_state=STAGE2_RANDOM_STATE,
    n_jobs=-1,
    **STAGE2_FIXED_PARAMS,
)

stage2_model.fit(stage2_train_fit_features)

stage2_train_scores = compute_anomaly_scores_isolation_forest(
    stage2_model,
    stage2_train_fit_features,
)

stage2_all_scores = compute_anomaly_scores_isolation_forest(
    stage2_model,
    stage2_all_features,
)

stage2_threshold = choose_threshold_by_percentile(
    stage2_train_scores,
    STAGE2_FIXED_THRESHOLD_PERCENTILE,
)

stage2_raw_flags = (stage2_all_scores >= stage2_threshold).astype(int)
stage2_flags = ((stage1_flags == 1) & (stage2_raw_flags == 1)).astype(int)

stage2_selected_threshold_percentile = float(STAGE2_FIXED_THRESHOLD_PERCENTILE)
stage2_best_params = dict(STAGE2_FIXED_PARAMS)

stage2_summary = {
    "selection_mode": "fixed",
    "selected_threshold_percentile": float(stage2_selected_threshold_percentile),
    "threshold": float(stage2_threshold),
    "best_params": stage2_best_params,
    "raw_alert_count_all_rows": int(stage2_raw_flags.sum()),
    "raw_alert_count_test_rows": int(stage2_raw_flags[test_mask.values].sum()),
    "stage2_confirmed_count_all_rows": int(stage2_flags.sum()),
    "stage2_confirmed_count_test_rows": int(stage2_flags[test_mask.values].sum()),
}

ledger.add(
    kind="step",
    step="run_cascade_stage2_fixed",
    message="Ran Stage 2 fixed narrow Isolation Forest using the fixed branch from the cascade config.",
    data={
        "selection_mode": "fixed",
        "best_params": stage2_best_params,
        "selected_threshold_percentile": float(stage2_selected_threshold_percentile),
        "threshold": float(stage2_threshold),
        "feature_count": int(len(stage2_feature_columns)),
        "raw_alert_count_all_rows": int(stage2_summary["raw_alert_count_all_rows"]),
        "raw_alert_count_test_rows": int(stage2_summary["raw_alert_count_test_rows"]),
        "stage2_confirmed_count_all_rows": int(stage2_summary["stage2_confirmed_count_all_rows"]),
        "stage2_confirmed_count_test_rows": int(stage2_summary["stage2_confirmed_count_test_rows"]),
    },
    logger=logger,
)

# %% [markdown]
# ----

# %%
cascade_results = gold_preprocessed_scaled_dataframe.copy()

cascade_results["stage1_score"] = stage1_all_scores
cascade_results["stage1_flag"] = stage1_flags

cascade_results["stage2_score"] = stage2_all_scores
cascade_results["stage2_raw_flag"] = stage2_raw_flags
cascade_results["stage2_flag"] = stage2_flags

# %% [markdown]
# ----

# %%
# --- Stage 3 sanity check: ensure rule sensors exist in scored dataframe
missing_primary = [c for c in stage3_primary_rule_sensors if c not in cascade_results.columns]
missing_secondary = [c for c in stage3_secondary_rule_sensors if c not in cascade_results.columns]

logger.info("Stage3 missing sensors: primary=%d secondary=%d", len(missing_primary), len(missing_secondary))

if missing_primary:
    logger.warning("Missing Stage3 PRIMARY sensors (showing up to 20): %s", missing_primary[:20])
if missing_secondary:
    logger.warning("Missing Stage3 SECONDARY sensors (showing up to 20): %s", missing_secondary[:20])


# %% [markdown]
# ----

# %%

def compute_primary_breach_count(
    dataframe: pd.DataFrame,
    *,
    reference_profile: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    reference_lookup = reference_profile.set_index("feature_name")[["lower_bound", "upper_bound"]]

    breach_counts = pd.Series(0, index=dataframe.index, dtype=int)

    for feature_name in feature_columns:
        if feature_name not in dataframe.columns or feature_name not in reference_lookup.index:
            continue

        lower = reference_lookup.loc[feature_name, "lower_bound"]
        upper = reference_lookup.loc[feature_name, "upper_bound"]

        breach_flag = ((dataframe[feature_name] < lower) | (dataframe[feature_name] > upper)).astype(int)
        breach_counts = breach_counts + breach_flag

    breach_counts.name = "stage3_profile_breach_count"
    return breach_counts





# %%
cascade_results["stage3_profile_breach_count"] = compute_primary_breach_count(
    cascade_results,
    reference_profile=reference_profile,
    feature_columns=stage3_primary_rule_sensors,
)

# %% [markdown]
# ----

# %%
def compute_secondary_breach_count(
    dataframe: pd.DataFrame,
    *,
    reference_profile: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    reference_lookup = reference_profile.set_index("feature_name").to_dict("index")
    breach_counts = pd.Series(0, index=dataframe.index, dtype=int)

    for feature_name in feature_columns:
        if feature_name not in reference_lookup:
            continue

        lower_bound = reference_lookup[feature_name]["lower_bound"]
        upper_bound = reference_lookup[feature_name]["upper_bound"]

        feature_breach_flag = (
            (dataframe[feature_name] < lower_bound) |
            (dataframe[feature_name] > upper_bound)
        ).astype(int)

        breach_counts = breach_counts + feature_breach_flag

    breach_counts.name = "stage3_secondary_breach_count"
    return breach_counts

# %%
cascade_results["stage3_secondary_breach_count"] = compute_secondary_breach_count(
    cascade_results,
    reference_profile=reference_profile,
    feature_columns=stage3_secondary_rule_sensors,
)

# %% [markdown]
# ----

# %%
def compute_persistence_flag(
    source_flags: pd.Series,
    *,
    rolling_window_size: int = 3,
    minimum_flags_in_window: int = 2,
) -> pd.Series:
    persistence_flag = (
        source_flags
        .rolling(window=rolling_window_size, min_periods=1)
        .sum()
        .ge(minimum_flags_in_window)
        .astype(int)
    )

    persistence_flag.name = "stage3_persistence_flag"
    return persistence_flag

# %%
cascade_results["stage3_persistence_flag"] = compute_persistence_flag(
    cascade_results["stage2_flag"],
    rolling_window_size=STAGE3_ROLLING_WINDOW_SIZE,
    minimum_flags_in_window=STAGE3_MINIMUM_FLAGS_IN_WINDOW,
)


# %% [markdown]
# ----

# %%
def compute_drift_flag(
    dataframe: pd.DataFrame,
    *,
    feature_columns: list[str],
    rolling_window_size: int = 5,
    drift_threshold_multiplier: float = 1.0,
) -> pd.Series:
    drift_trigger_counts = pd.Series(0, index=dataframe.index, dtype=int)

    for feature_name in feature_columns:
        feature_series = dataframe[feature_name]
        feature_standard_deviation = feature_series.std()

        if pd.isna(feature_standard_deviation) or feature_standard_deviation == 0:
            continue

        rolling_median = feature_series.rolling(window=rolling_window_size, min_periods=1).median()
        rolling_delta = (feature_series - rolling_median).abs()

        feature_drift_flag = (
            rolling_delta > (feature_standard_deviation * drift_threshold_multiplier)
        ).astype(int)

        drift_trigger_counts = drift_trigger_counts + feature_drift_flag

    drift_flag = (drift_trigger_counts >= 1).astype(int)
    drift_flag.name = "stage3_drift_flag"
    return drift_flag

# %% [markdown]
# ----

# %%
stage3_rule_watch_features = list(dict.fromkeys(
    stage3_primary_rule_sensors + stage3_secondary_rule_sensors
))

cascade_results["stage3_drift_flag"] = compute_drift_flag(
    cascade_results,
    feature_columns=stage3_rule_watch_features,
    rolling_window_size=5,
    drift_threshold_multiplier=1.0,
)

# %% [markdown]
# ----

# %%
cascade_results["stage3_profile_breach_flag"] = (
    cascade_results["stage3_profile_breach_count"] >= STAGE3_MIN_PRIMARY_SENSOR_HITS
).astype(int)

cascade_results["stage3_corroboration_flag"] = (
    cascade_results["stage3_secondary_breach_count"] >= STAGE3_MIN_SECONDARY_SENSOR_HITS
).astype(int)

cascade_results["stage3_rule_evidence_count"] = (
    cascade_results["stage3_profile_breach_flag"] +
    cascade_results["stage3_persistence_flag"] +
    cascade_results["stage3_drift_flag"] +
    cascade_results["stage3_corroboration_flag"]
)

cascade_results["cascade_final_flag"] = (
    (cascade_results["stage1_flag"] == 1) &
    (cascade_results["stage2_flag"] == 1) &
    (
        (cascade_results["stage3_profile_breach_count"] >= STAGE3_MIN_PRIMARY_SENSOR_HITS) |
        (cascade_results["stage3_rule_evidence_count"] >= 2)
    )
).astype(int)

stage3_summary = {
    "primary_rule_sensor_count": int(len(stage3_primary_rule_sensors)),
    "secondary_rule_sensor_count": int(len(stage3_secondary_rule_sensors)),
    "profile_breach_rows_all": int((cascade_results["stage3_profile_breach_flag"] == 1).sum()),
    "profile_breach_rows_test": int(cascade_results.loc[test_mask, "stage3_profile_breach_flag"].sum()),
    "corroboration_rows_all": int((cascade_results["stage3_corroboration_flag"] == 1).sum()),
    "corroboration_rows_test": int(cascade_results.loc[test_mask, "stage3_corroboration_flag"].sum()),
    "persistence_rows_all": int((cascade_results["stage3_persistence_flag"] == 1).sum()),
    "persistence_rows_test": int(cascade_results.loc[test_mask, "stage3_persistence_flag"].sum()),
    "drift_rows_all": int((cascade_results["stage3_drift_flag"] == 1).sum()),
    "drift_rows_test": int(cascade_results.loc[test_mask, "stage3_drift_flag"].sum()),
    "final_alert_count_all_rows": int(cascade_results["cascade_final_flag"].sum()),
    "final_alert_count_test_rows": int(cascade_results.loc[test_mask, "cascade_final_flag"].sum()),
}

ledger.add(
    kind="step",
    step="run_cascade_stage3_confirmation",
    message="Applied Stage 3 confirmation checks to all rows of the scaled dataset.",
    data=stage3_summary,
    logger=logger,
)

cascade_results.head(5)

# %% [markdown]
# ----

# %%
cascade_metrics = {
    "model": "3-Stage Cascade",
    "stage1_alert_count_all_rows": int(cascade_results["stage1_flag"].sum()),
    "stage2_alert_count_all_rows": int(cascade_results["stage2_flag"].sum()),
    "final_alert_count_all_rows": int(cascade_results["cascade_final_flag"].sum()),
    "stage1_alert_count_test_rows": int(cascade_results.loc[test_mask, "stage1_flag"].sum()),
    "stage2_alert_count_test_rows": int(cascade_results.loc[test_mask, "stage2_flag"].sum()),
    "final_alert_count_test_rows": int(cascade_results.loc[test_mask, "cascade_final_flag"].sum()),
}



if test_labels is not None:
    cascade_test_flags = (
        cascade_results
        .loc[test_mask, "cascade_final_flag"]
        .astype(int)
        .values
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels,
        cascade_test_flags,
        average="binary",
        zero_division=0,
    )

    cascade_metrics["precision"] = float(precision)
    cascade_metrics["recall"] = float(recall)
    cascade_metrics["f1"] = float(f1)

cascade_metrics

# %% [markdown]
# ----

# %%
stage1_alert_count_all_rows = int(cascade_results["stage1_flag"].sum())
stage2_alert_count_all_rows = int(cascade_results["stage2_flag"].sum())
final_cascade_alert_count_all_rows = int(cascade_results["cascade_final_flag"].sum())

stage1_alert_count_test_rows = int(cascade_results.loc[test_mask, "stage1_flag"].sum())
stage2_alert_count_test_rows = int(cascade_results.loc[test_mask, "stage2_flag"].sum())
final_cascade_alert_count_test_rows = int(cascade_results.loc[test_mask, "cascade_final_flag"].sum())

cascade_thresholds = {
    "stage1_threshold_percentile": float(STAGE1_THRESHOLD_PERCENTILE),
    "stage1_threshold": float(stage1_threshold),
    "stage2_selection_mode": STAGE2_SELECTION_MODE,
    "stage2_selected_threshold_percentile": float(stage2_selected_threshold_percentile),
    "stage2_threshold": float(stage2_threshold),
    "stage2_best_params": stage2_best_params,
}

cascade_summary = {
    "dataset_name": DATASET_NAME,
    "cascade_metrics": cascade_metrics,
    "stage1_alert_count_all_rows": stage1_alert_count_all_rows,
    "stage2_alert_count_all_rows": stage2_alert_count_all_rows,
    "final_cascade_alert_count_all_rows": final_cascade_alert_count_all_rows,
    "stage1_alert_count_test_rows": stage1_alert_count_test_rows,
    "stage2_alert_count_test_rows": stage2_alert_count_test_rows,
    "final_cascade_alert_count_test_rows": final_cascade_alert_count_test_rows,
    "result_row_count": int(len(cascade_results)),
    "stage1_feature_count": int(len(stage1_feature_columns)),
    "stage2_feature_count": int(len(stage2_feature_columns)),
    "stage3_primary_rule_count": int(len(stage3_primary_rule_sensors)),
    "stage3_secondary_rule_count": int(len(stage3_secondary_rule_sensors)),
    "stage2_selection_mode": STAGE2_SELECTION_MODE,
    "stage2_best_params": stage2_best_params,
}

truth_config_snapshot = (
    TRUTH_CONFIG
    if "TRUTH_CONFIG" in globals()
    else {
        "runtime": {
            "stage": "gold_cascade",
            "dataset": DATASET_NAME,
            "mode": RUN_MODE if "RUN_MODE" in globals() else None,
            "profile": CONFIG_PROFILE if "CONFIG_PROFILE" in globals() else "default",
        }
    }
)

cascade_truth_layer_name = "gold_cascade"
cascade_process_run_id = (
    GOLD_PROCESS_RUN_ID
    if "GOLD_PROCESS_RUN_ID" in globals()
    else make_process_run_id("gold_cascade_process")
)

cascade_truth = initialize_layer_truth(
    truth_version=TRUTH_VERSION,
    dataset_name=DATASET_NAME,
    layer_name=cascade_truth_layer_name,
    process_run_id=cascade_process_run_id,
    pipeline_mode=PIPELINE_MODE,
    parent_truth_hash=GOLD_PARENT_TRUTH_HASH,
)

cascade_truth = update_truth_section(
    cascade_truth,
    "config_snapshot",
    truth_config_snapshot,
)

cascade_truth = update_truth_section(
    cascade_truth,
    "runtime_facts",
    {
        "stage1_threshold_percentile": float(STAGE1_THRESHOLD_PERCENTILE),
        "stage1_threshold": float(stage1_threshold),
        "stage2_selection_mode": STAGE2_SELECTION_MODE,
        "stage2_selected_threshold_percentile": float(stage2_selected_threshold_percentile),
        "stage2_threshold": float(stage2_threshold),
        "stage1_estimator_count": int(STAGE1_ESTIMATOR_COUNT),
        "stage2_estimator_count": int(stage2_model.get_params()["n_estimators"]),
        "stage2_best_params": stage2_best_params,
        "stage1_feature_count": int(len(stage1_feature_columns)),
        "stage2_feature_count": int(len(stage2_feature_columns)),
        "stage3_primary_rule_count": int(len(stage3_primary_rule_sensors)),
        "stage3_secondary_rule_count": int(len(stage3_secondary_rule_sensors)),
        "result_row_count": int(len(cascade_results)),
        "parent_truth_hash": GOLD_PARENT_TRUTH_HASH,
        "gold_process_run_id": gold_truth.get("process_run_id"),
        "gold_feature_set_id": gold_truth_runtime_facts.get("feature_set_id"),
    },
)

cascade_truth = update_truth_section(
    cascade_truth,
    "artifact_paths",
    {
        "gold_truth_path": str(GOLD_TRUTH_PATH),
        "gold_fit_path": str(GOLD_FIT_DATA_PATH),
        "gold_scored_path": str(GOLD_PREPROCESSED_SCALED_DATA_PATH),
        "stage1_features_path": str(STAGE1_FEATURES_PATH),
        "stage2_features_path": str(STAGE2_FEATURES_PATH),
        "stage3_primary_path": str(STAGE3_PRIMARY_PATH),
        "stage3_secondary_path": str(STAGE3_SECONDARY_PATH),
        "cascade_results_path_csv": str(CASCADE_RESULTS_PATH_CSV),
        "cascade_results_path_pickle": str(CASCADE_RESULTS_PATH_PICKLE),
        "stage1_model_artifact_path": str(STAGE1_MODEL_ARTIFACT_PATH),
        "stage1_models_path": str(STAGE1_MODELS_PATH),
        "stage2_model_artifact_path": str(STAGE2_MODEL_ARTIFACT_PATH),
        "stage2_models_path": str(STAGE2_MODELS_PATH),
        "cascade_thresholds_path": str(CASCADE_THRESHOLDS_PATH),
        "cascade_summary_path": str(CASCADE_SUMMARY_PATH),
        "cascade_metadata_path": str(CASCADE_METADATA_PATH),
        "cascade_reference_profile_path": str(CASCADE_REFERENCE_PROFILE_PATH),
    },
)

cascade_meta_columns = sorted(
    set(
        identify_meta_columns(cascade_results)
        + [
            "meta__truth_hash",
            "meta__parent_truth_hash",
            "meta__pipeline_mode",
        ]
    )
)

cascade_feature_columns = identify_feature_columns(cascade_results)

cascade_truth_record = build_truth_record(
    truth_base=cascade_truth,
    row_count=len(cascade_results),
    column_count=cascade_results.shape[1] + 3,
    meta_columns=cascade_meta_columns,
    feature_columns=cascade_feature_columns,
)

CASCADE_TRUTH_HASH = cascade_truth_record["truth_hash"]

cascade_results = stamp_truth_columns(
    cascade_results,
    truth_hash=CASCADE_TRUTH_HASH,
    parent_truth_hash=GOLD_PARENT_TRUTH_HASH,
    pipeline_mode=PIPELINE_MODE,
)

cascade_truth_path = save_truth_record(
    cascade_truth_record,
    truth_dir=TRUTHS_PATH,
    dataset_name=DATASET_NAME,
    layer_name=cascade_truth_layer_name,
)

append_truth_index(
    cascade_truth_record,
    truth_index_path=TRUTH_INDEX_PATH,
)

cascade_summary["cascade_truth_hash"] = CASCADE_TRUTH_HASH
cascade_summary["cascade_truth_path"] = str(cascade_truth_path)
cascade_summary["cascade_process_run_id"] = cascade_process_run_id
cascade_summary["gold_truth_hash"] = GOLD_PARENT_TRUTH_HASH
cascade_summary["gold_truth_path"] = str(GOLD_TRUTH_PATH)
cascade_summary["gold_process_run_id"] = gold_truth.get("process_run_id")
cascade_summary["gold_feature_set_id"] = gold_truth_runtime_facts.get("feature_set_id")

cascade_metadata = {
    "cascade_results_path_csv": str(CASCADE_RESULTS_PATH_CSV),
    "cascade_results_path_pickle": str(CASCADE_RESULTS_PATH_PICKLE),
    "stage1_model_artifact_path": str(STAGE1_MODEL_ARTIFACT_PATH),
    "stage1_models_path": str(STAGE1_MODELS_PATH),
    "stage2_model_artifact_path": str(STAGE2_MODEL_ARTIFACT_PATH),
    "stage2_models_path": str(STAGE2_MODELS_PATH),
    "cascade_thresholds_path": str(CASCADE_THRESHOLDS_PATH),
    "cascade_summary_path": str(CASCADE_SUMMARY_PATH),
    "cascade_metadata_path": str(CASCADE_METADATA_PATH),
    "cascade_reference_profile_path": str(CASCADE_REFERENCE_PROFILE_PATH),
    "gold_fit_path": str(GOLD_FIT_DATA_PATH),
    "gold_scored_path": str(GOLD_PREPROCESSED_SCALED_DATA_PATH),

    # upstream truth linkage
    "gold_truth_hash": GOLD_PARENT_TRUTH_HASH,
    "gold_truth_path": str(GOLD_TRUTH_PATH),
    "gold_process_run_id": gold_truth.get("process_run_id"),
    "gold_feature_set_id": gold_truth_runtime_facts.get("feature_set_id"),
    "gold_scaler_path": gold_truth_artifact_paths.get("scaler_path"),
    "gold_scaler_kind": gold_truth_runtime_facts.get("scaler_kind_runtime"),
    "gold_recommended_imputation": gold_truth_runtime_facts.get("recommended_imputation"),

    # stage truth linkage
    "cascade_truth_hash": CASCADE_TRUTH_HASH,
    "cascade_truth_path": str(cascade_truth_path),
    "cascade_process_run_id": cascade_process_run_id,
}

cascade_results.to_csv(CASCADE_RESULTS_PATH_CSV, index=False)
cascade_results.to_pickle(CASCADE_RESULTS_PATH_PICKLE)

reference_profile.to_csv(CASCADE_REFERENCE_PROFILE_PATH, index=False)


joblib.dump(stage1_model, STAGE1_MODEL_ARTIFACT_PATH)
joblib.dump(stage1_model, STAGE1_MODELS_PATH)

joblib.dump(stage2_model, STAGE2_MODEL_ARTIFACT_PATH)
joblib.dump(stage2_model, STAGE2_MODELS_PATH)

save_json(cascade_thresholds, CASCADE_THRESHOLDS_PATH)
save_json(cascade_summary, CASCADE_SUMMARY_PATH)
save_json(cascade_metadata, CASCADE_METADATA_PATH)

wandb.save(str(CASCADE_RESULTS_PATH_CSV))
wandb.save(str(CASCADE_RESULTS_PATH_PICKLE))
wandb.save(str(CASCADE_REFERENCE_PROFILE_PATH))
wandb.save(str(STAGE1_MODEL_ARTIFACT_PATH))
wandb.save(str(STAGE1_MODELS_PATH))
wandb.save(str(STAGE2_MODEL_ARTIFACT_PATH))
wandb.save(str(STAGE2_MODELS_PATH))
wandb.save(str(CASCADE_THRESHOLDS_PATH))
wandb.save(str(CASCADE_SUMMARY_PATH))
wandb.save(str(CASCADE_METADATA_PATH))
wandb.save(str(cascade_truth_path))

ledger.add(
    kind="step",
    step="save_cascade_outputs",
    message="Saved cascade results, trained Stage 1 and Stage 2 models, thresholds, summary, metadata, reference profile, and cascade stage truth record.",
    data={
        "cascade_results_path_csv": str(CASCADE_RESULTS_PATH_CSV),
        "cascade_results_path_pickle": str(CASCADE_RESULTS_PATH_PICKLE),
        "cascade_reference_profile_path": str(CASCADE_REFERENCE_PROFILE_PATH),
        "stage1_model_artifact_path": str(STAGE1_MODEL_ARTIFACT_PATH),
        "stage1_models_path": str(STAGE1_MODELS_PATH),
        "stage2_model_artifact_path": str(STAGE2_MODEL_ARTIFACT_PATH),
        "stage2_models_path": str(STAGE2_MODELS_PATH),
        "cascade_thresholds_path": str(CASCADE_THRESHOLDS_PATH),
        "cascade_summary_path": str(CASCADE_SUMMARY_PATH),
        "cascade_metadata_path": str(CASCADE_METADATA_PATH),
        "cascade_truth_hash": CASCADE_TRUTH_HASH,
        "cascade_truth_path": str(cascade_truth_path),
        "result_row_count": int(len(cascade_results)),
        "final_alert_count_all_rows": final_cascade_alert_count_all_rows,
        "final_alert_count_test_rows": final_cascade_alert_count_test_rows,
    },
    logger=logger,
)

# %% [markdown]
# ----

# %%
ledger.add(
    kind="step",
    step="finalize_cascade_modeling",
    message="Gold cascade modeling notebook complete.",
    data={
        "cascade_metrics": cascade_metrics,
        "cascade_results_path_csv": str(CASCADE_RESULTS_PATH_CSV),
        "cascade_results_path_pickle": str(CASCADE_RESULTS_PATH_PICKLE),
        "stage1_model_artifact_path": str(STAGE1_MODEL_ARTIFACT_PATH),
        "stage1_models_path": str(STAGE1_MODELS_PATH),
        "stage2_model_artifact_path": str(STAGE2_MODEL_ARTIFACT_PATH),
        "stage2_models_path": str(STAGE2_MODELS_PATH),
    },
    logger=logger,
)

cascade_ledger_path = GOLD_ARTIFACTS_PATH / GOLD_CASCADE_LEDGER_FILE_NAME
ledger.write_json(cascade_ledger_path)

wandb.save(str(cascade_ledger_path))
wandb_run.finish()

# %% [markdown]
# ----

# %%
required_cascade_meta_columns = [
    "meta__truth_hash",
    "meta__parent_truth_hash",
    "meta__pipeline_mode",
]

missing_cascade_meta_columns = [
    column_name
    for column_name in required_cascade_meta_columns
    if column_name not in cascade_results.columns
]
if missing_cascade_meta_columns:
    raise ValueError(
        f"cascade_results is missing required lineage columns: {missing_cascade_meta_columns}"
    )

cascade_results_truth_hash_check = extract_truth_hash(cascade_results)
if cascade_results_truth_hash_check is None:
    raise ValueError("cascade_results does not contain a readable meta__truth_hash value.")

if cascade_results_truth_hash_check != CASCADE_TRUTH_HASH:
    raise ValueError(
        "cascade_results truth hash does not match CASCADE_TRUTH_HASH:\n"
        f"dataframe={cascade_results_truth_hash_check}\n"
        f"record={CASCADE_TRUTH_HASH}"
    )

cascade_parent_values = cascade_results["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()
if not cascade_parent_values:
    raise ValueError("cascade_results is missing populated meta__parent_truth_hash values.")

if len(cascade_parent_values) != 1:
    raise ValueError(f"cascade_results has multiple parent truth hashes: {cascade_parent_values}")

if cascade_parent_values[0] != GOLD_PARENT_TRUTH_HASH:
    raise ValueError(
        "cascade_results parent truth hash does not match GOLD_PARENT_TRUTH_HASH:\n"
        f"dataframe_parent={cascade_parent_values[0]}\n"
        f"gold_parent={GOLD_PARENT_TRUTH_HASH}"
    )

if not Path(cascade_truth_path).exists():
    raise FileNotFoundError(f"Cascade truth file was not created: {cascade_truth_path}")

loaded_cascade_truth = load_json(cascade_truth_path)

if loaded_cascade_truth.get("truth_hash") != CASCADE_TRUTH_HASH:
    raise ValueError(
        "Saved Cascade truth file hash does not match CASCADE_TRUTH_HASH:\n"
        f"file={loaded_cascade_truth.get('truth_hash')}\n"
        f"record={CASCADE_TRUTH_HASH}"
    )

if loaded_cascade_truth.get("parent_truth_hash") != GOLD_PARENT_TRUTH_HASH:
    raise ValueError(
        "Saved Cascade truth file parent hash does not match GOLD_PARENT_TRUTH_HASH:\n"
        f"truth={loaded_cascade_truth.get('parent_truth_hash')}\n"
        f"gold_parent={GOLD_PARENT_TRUTH_HASH}"
    )

saved_cascade_metadata = load_json(CASCADE_METADATA_PATH)
if saved_cascade_metadata.get("cascade_truth_hash") != CASCADE_TRUTH_HASH:
    raise ValueError(
        "cascade_metadata cascade_truth_hash does not match CASCADE_TRUTH_HASH:\n"
        f"metadata={saved_cascade_metadata.get('cascade_truth_hash')}\n"
        f"record={CASCADE_TRUTH_HASH}"
    )

print("Gold Cascade lineage sanity check passed.")

# %% [markdown]
# ----


