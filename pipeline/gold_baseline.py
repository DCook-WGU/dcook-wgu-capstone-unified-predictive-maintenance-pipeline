# %% [markdown]
# ## Gold Baseline Modeling (Deliverable 1.3.2)
# 
# This notebook implements the baseline anomaly detection model using a single, broad Isolation Forest trained on the complete Gold feature set. The baseline serves as the primary comparison point for evaluating whether the three-stage cascade model improves alert quality.
# 
# **Purpose:**  
# To produce the baseline model’s anomaly scores and alert outputs using the fully preprocessed Gold dataset. These outputs represent the simplest form of unsupervised anomaly detection and form the quantitative reference for the comparative evaluation in the Gold Comparison notebook.
# 
# **Key Goals:**
# 
# - Load the Gold preprocessed dataset and Stage 1 feature set.
# - Train a single Isolation Forest using all vetted numeric features.
# - Generate anomaly scores and binary alert flags.
# - Produce baseline alert frequency counts, false-positive counts, and normal-period alert rates.
# - Save all baseline model artifacts, metrics, and alert outputs for use in the Gold Comparison notebook.
# 
# **Relevance to Section C:**  
# This notebook provides the baseline alert patterns used in Section C to evaluate whether the cascade reduces false positives, reduces noisy alerts, and preserves meaningful anomaly sensitivity. The outputs here form the “reference condition” for the paired statistical tests and practical significance analysis.

# %%
print("hello"   )

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

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

CONFIG_ROOT = paths.configs
CONFIG_RUN_MODE = "train"
CONFIG_PROFILE = "default"

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

CONFIG = load_pipeline_config(
    config_root=CONFIG_ROOT,
    stage="gold_baseline",
    dataset="pump",
    mode=CONFIG_RUN_MODE,
    profile=CONFIG_PROFILE,
    project_root=paths.root,
).data

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

GOLD_CFG = CONFIG["gold_baseline"]
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

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

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

# ---- W&B ----
WANDB_PROJECT = CONFIG["wandb"]["project"]
WANDB_ENTITY = CONFIG["wandb"]["entity"]
WANDB_RUN_NAME = f"{GOLD_VERSION}"

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####  

GOLD_PREPROCESSED_FILE_NAME = FILENAMES["gold_preprocessed_file_name"]
GOLD_PREPROCESSED_SCALED_FILE_NAME = FILENAMES["gold_preprocessed_scaled_file_name"]
GOLD_FIT_FILE_NAME = FILENAMES["gold_fit_file_name"]
GOLD_TEST_FILE_NAME = FILENAMES["gold_test_file_name"]
GOLD_TRAIN_FILE_NAME = FILENAMES["gold_train_file_name"]

STAGE1_FEATURES_FILE_NAME = FILENAMES["stage1_features_file_name"]

BASELINE_RESULTS_FILE_NAME_CSV = FILENAMES["baseline_results_file_name_csv"]
BASELINE_RESULTS_FILE_NAME_PICKLE = FILENAMES["baseline_results_file_name_pickle"]
BASELINE_MODEL_FILE_NAME = FILENAMES["baseline_model_file_name"]
BASELINE_THRESHOLDS_FILE_NAME = FILENAMES["baseline_thresholds_file_name"]
BASELINE_SUMMARY_FILE_NAME = FILENAMES["baseline_summary_file_name"]
BASELINE_METADATA_FILE_NAME = FILENAMES["baseline_metadata_file_name"]

TRUTH_INDEX_FILE_NAME = "truth_index.jsonl"

TRAIN_FRACTION = float(GOLD_CFG["train_fraction"])
RANDOM_SEED = int(GOLD_CFG["random_seed"])
BASELINE_THRESHOLD_PERCENTILE = float(GOLD_CFG["baseline_threshold_percentile"])
STAGE1_THRESHOLD_PERCENTILE = float(GOLD_CFG["stage1_threshold_percentile"])
STAGE2_THRESHOLD_PERCENTILE = float(GOLD_CFG["stage2_threshold_percentile"])
BASELINE_ESTIMATOR_COUNT = int(GOLD_CFG["baseline_estimator_count"])

GOLD_BASELINE_LEDGER_FILE_NAME = FILENAMES["gold_baseline_ledger_file_name"]



# ---- Paths setup ----
GOLD_PREPROCESSED_DATA_PATH = Path(PATHS["gold_preprocessed_data_path"])
GOLD_PREPROCESSED_SCALED_DATA_PATH = Path(PATHS["gold_preprocessed_scaled_data_path"])

GOLD_TRAIN_DATA_PATH = Path(PATHS["gold_train_data_path"])
GOLD_TEST_DATA_PATH = Path(PATHS["gold_test_data_path"])
GOLD_FIT_DATA_PATH = Path(PATHS["gold_fit_data_path"])
GOLD_ARTIFACTS_PATH = Path(PATHS["gold_artifacts_dir"])

MODELS_PATH = Path(PATHS["models_root"])
BASELINE_MODELS_PATH = Path(PATHS["baseline_models_path"])
BASELINE_MODEL_ARTIFACT_PATH = Path(PATHS["baseline_model_artifact_path"])

BASELINE_RESULTS_PATH_CSV = Path(PATHS["baseline_results_path_csv"])
BASELINE_RESULTS_PATH_PICKLE = Path(PATHS["baseline_results_path_pickle"])
BASELINE_THRESHOLDS_PATH = Path(PATHS["baseline_thresholds_path"])
BASELINE_SUMMARY_PATH = Path(PATHS["baseline_summary_path"])
BASELINE_METADATA_PATH = Path(PATHS["baseline_metadata_path"])

TRUTHS_PATH = Path(PATHS["truths_dir"])
TRUTH_INDEX_PATH = Path(PATHS["truth_index_path"])

STAGE1_FEATURES_PATH = Path(PATHS["stage1_features_path"])

LOGS_PATH = Path(PATHS["logs_root"])

set_wandb_dir_from_config(CONFIG)

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

# %% [markdown]
# ----

# %%
# Logging Setup

# Create gold log path 
gold_log_path = paths.logs / "gold_modeling_baseline.log"

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
    job_type="gold_modeling_baseline",
    config={
        "gold_version": GOLD_VERSION,
        "dataset": DATASET_NAME,
        "stage": STAGE,
        "train_fraction": TRAIN_FRACTION,
        "baseline_threshold_percentile": BASELINE_THRESHOLD_PERCENTILE,
        "gold_input_path": str(GOLD_FIT_DATA_PATH),
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
    raise ValueError("Gold baseline input dataframe is missing usable meta__dataset values.")

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

GOLD_FIT_DATA_PATH = Path(gold_truth_artifact_paths.get("gold_fit_path", str(GOLD_FIT_DATA_PATH)))
STAGE1_FEATURES_PATH = Path(gold_truth_artifact_paths.get("stage1_features_path", str(STAGE1_FEATURES_PATH)))

logger.info("Resolved Gold baseline dataset name from Gold truth: %s", DATASET_NAME)
logger.info("Resolved Gold truth path: %s", GOLD_TRUTH_PATH)
logger.info("Resolved Gold fit parquet from Gold truth: %s", GOLD_FIT_DATA_PATH)
logger.info("Resolved Stage 1 features path from Gold truth: %s", STAGE1_FEATURES_PATH)

logger.info("Loading Gold fit parquet: %s", GOLD_FIT_DATA_PATH)
gold_fit_dataframe = load_data(GOLD_FIT_DATA_PATH)

logger.info("Loading Stage 1 features JSON: %s", STAGE1_FEATURES_PATH)
stage1_feature_columns = load_json(STAGE1_FEATURES_PATH)

print("Gold baseline dataset name from parent truth:", DATASET_NAME)
print("Gold baseline parent truth hash:", GOLD_PARENT_TRUTH_HASH)

ledger.add(
    kind="step",
    step="load_modeling_inputs",
    message="Loaded Gold scaled parquet, loaded Gold truth, substituted truth-linked artifact paths, then loaded baseline inputs.",
    data={
        "gold_scaled_path": str(GOLD_PREPROCESSED_SCALED_DATA_PATH),
        "gold_truth_hash": GOLD_PARENT_TRUTH_HASH,
        "gold_truth_path": str(GOLD_TRUTH_PATH),
        "gold_fit_path": str(GOLD_FIT_DATA_PATH),
        "stage1_features_path": str(STAGE1_FEATURES_PATH),
        "gold_scaled_shape": list(gold_preprocessed_scaled_dataframe.shape),
        "gold_fit_shape": list(gold_fit_dataframe.shape),
        "stage1_feature_count": int(len(stage1_feature_columns)),
    },
    logger=logger,
)

gold_fit_dataframe.head(3)

# %% [markdown]
# ----

# %% [markdown]
# ----

# %%
# Masks (must exist in scaled parquet)
if "meta__is_train_flag" not in gold_preprocessed_scaled_dataframe.columns:
    raise ValueError("meta__is_train_flag missing from gold_preprocessed_scaled_dataframe. "
                     "Gold preprocessing must stamp it before saving.")


train_mask = gold_preprocessed_scaled_dataframe["meta__is_train_flag"].astype(bool)

test_mask = ~train_mask


logger.info("Split counts: all=%d train=%d test=%d", len(train_mask), int(train_mask.sum()), int(test_mask.sum()))
if "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns:
    logger.info("Anomalies: all=%d test=%d", 
                int(gold_preprocessed_scaled_dataframe["anomaly_flag"].fillna(0).astype(int).sum()),
                int(gold_preprocessed_scaled_dataframe.loc[test_mask, "anomaly_flag"].fillna(0).astype(int).sum()))

# %% [markdown]
# ----

# %%
all_labels = None
test_labels = None

if "anomaly_flag" in gold_preprocessed_scaled_dataframe.columns:
    all_labels = gold_preprocessed_scaled_dataframe["anomaly_flag"].fillna(0).astype(int).values
    test_labels = gold_preprocessed_scaled_dataframe.loc[test_mask, "anomaly_flag"].fillna(0).astype(int).values
else:
    all_labels = None
    test_labels = None


baseline_train_fit_features = gold_fit_dataframe[stage1_feature_columns].values

baseline_all_features = gold_preprocessed_scaled_dataframe[stage1_feature_columns].values

#baseline_test_features = gold_test_dataframe[stage1_feature_columns].values

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

# %%
baseline_model = IsolationForest(
    n_estimators=BASELINE_ESTIMATOR_COUNT,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)


baseline_model.fit(baseline_train_fit_features)


# %%
# Score on Normal Only 
baseline_train_scores = compute_anomaly_scores_isolation_forest(
    baseline_model,
    baseline_train_fit_features,
)

# Score on All Only 
baseline_all_scores = compute_anomaly_scores_isolation_forest(
    baseline_model,
    baseline_all_features,
)


'''
# Score on Test
baseline_test_scores = compute_anomaly_scores_isolation_forest(
    baseline_model,
    baseline_test_features,
)
'''

if len(baseline_all_scores) != len(gold_preprocessed_scaled_dataframe):
    raise ValueError("Score length mismatch vs all-rows dataframe. Check feature matrix source.")

baseline_threshold = choose_threshold_by_percentile(
    baseline_train_scores,
    BASELINE_THRESHOLD_PERCENTILE,
)

#baseline_flags = (baseline_test_scores >= baseline_threshold).astype(int)
baseline_flags = (baseline_all_scores >= baseline_threshold).astype(int)



baseline_results = gold_preprocessed_scaled_dataframe.copy()



#baseline_results["baseline_score"] = baseline_test_scores

baseline_results["baseline_score"] = baseline_all_scores

baseline_results["baseline_flag"] = baseline_flags

baseline_metrics = {
    "model": "Baseline IsolationForest",
    "threshold_percentile": BASELINE_THRESHOLD_PERCENTILE,
    "threshold": float(baseline_threshold),
    "alert_count_all_rows": int(baseline_flags.sum()),
    "alert_count_test_rows": int(baseline_flags[test_mask.values].sum()),
}

if test_labels is not None:
    baseline_metrics.update(
        evaluate_against_labels(
            test_labels,
            #baseline_test_scores,
            baseline_all_scores[test_mask.values],
            baseline_threshold,
        )
    )

ledger.add(
    kind="step",
    step="run_baseline_isolation_forest",
    message="Ran baseline Isolation Forest fit on normal-only rows and scored the full scaled Gold dataset; evaluated on test rows.",
    data={
        "estimator_count": int(BASELINE_ESTIMATOR_COUNT),
        "threshold_percentile": float(BASELINE_THRESHOLD_PERCENTILE),
        "threshold": float(baseline_threshold),
        "training_rows": int(len(gold_fit_dataframe)),
        "test_rows": int(test_mask.sum()),
        "all_rows": int(len(gold_preprocessed_scaled_dataframe)),
        "train_rows": int(train_mask.sum()),
        "feature_count": int(len(stage1_feature_columns)),
        "alert_count_test_rows": int(baseline_flags[test_mask.values].sum()),
        "precision": baseline_metrics.get("precision"),
        "recall": baseline_metrics.get("recall"),
        "f1": baseline_metrics.get("f1"),
        "roc_auc": baseline_metrics.get("roc_auc"),
        "pr_auc": baseline_metrics.get("pr_auc"),
    },
    logger=logger,
)

baseline_metrics

# %%
baseline_alert_count_all_rows = int(baseline_results["baseline_flag"].sum())
baseline_alert_count_test_rows = int(baseline_results.loc[test_mask, "baseline_flag"].sum())

baseline_thresholds = {
    "baseline_threshold_percentile": float(BASELINE_THRESHOLD_PERCENTILE),
    "baseline_threshold": float(baseline_threshold),
}

baseline_summary = {
    "dataset_name": DATASET_NAME,
    "baseline_metrics": baseline_metrics,
    "alert_count_all_rows": baseline_alert_count_all_rows,
    "alert_count_test_rows": baseline_alert_count_test_rows,
    "result_row_count": int(len(baseline_results)),
}

truth_config_snapshot = (
    TRUTH_CONFIG
    if "TRUTH_CONFIG" in globals()
    else {
        "runtime": {
            "stage": "gold_baseline",
            "dataset": DATASET_NAME,
            "mode": RUN_MODE if "RUN_MODE" in globals() else None,
            "profile": CONFIG_PROFILE if "CONFIG_PROFILE" in globals() else "default",
        }
    }
)

baseline_truth_layer_name = "gold_baseline"
baseline_process_run_id = (
    GOLD_PROCESS_RUN_ID
    if "GOLD_PROCESS_RUN_ID" in globals()
    else make_process_run_id("gold_baseline_process")
)

baseline_truth = initialize_layer_truth(
    truth_version=TRUTH_VERSION,
    dataset_name=DATASET_NAME,
    layer_name=baseline_truth_layer_name,
    process_run_id=baseline_process_run_id,
    pipeline_mode=PIPELINE_MODE,
    parent_truth_hash=GOLD_PARENT_TRUTH_HASH,
)

baseline_truth = update_truth_section(
    baseline_truth,
    "config_snapshot",
    truth_config_snapshot,
)

baseline_truth = update_truth_section(
    baseline_truth,
    "runtime_facts",
    {
        "baseline_threshold_percentile": float(BASELINE_THRESHOLD_PERCENTILE),
        "baseline_threshold": float(baseline_threshold),
        "baseline_estimator_count": int(BASELINE_ESTIMATOR_COUNT),
        "train_fraction": float(TRAIN_FRACTION),
        "random_seed": int(RANDOM_SEED),
        "alert_count_all_rows": baseline_alert_count_all_rows,
        "alert_count_test_rows": baseline_alert_count_test_rows,
        "result_row_count": int(len(baseline_results)),
        "parent_truth_hash": GOLD_PARENT_TRUTH_HASH,
        "gold_process_run_id": gold_truth.get("process_run_id"),
        "gold_feature_set_id": gold_truth_runtime_facts.get("feature_set_id"),
    },
)

baseline_truth = update_truth_section(
    baseline_truth,
    "artifact_paths",
    {
        "gold_truth_path": str(GOLD_TRUTH_PATH),
        "gold_preprocessed_scaled_path": str(GOLD_PREPROCESSED_SCALED_DATA_PATH),
        "gold_fit_path": str(GOLD_FIT_DATA_PATH),
        "baseline_results_path_csv": str(BASELINE_RESULTS_PATH_CSV),
        "baseline_results_path_pickle": str(BASELINE_RESULTS_PATH_PICKLE),
        "baseline_model_artifact_path": str(BASELINE_MODEL_ARTIFACT_PATH),
        "baseline_models_path": str(BASELINE_MODELS_PATH),
        "baseline_thresholds_path": str(BASELINE_THRESHOLDS_PATH),
        "baseline_summary_path": str(BASELINE_SUMMARY_PATH),
        "baseline_metadata_path": str(BASELINE_METADATA_PATH),
    },
)

baseline_meta_columns = sorted(
    set(
        identify_meta_columns(baseline_results)
        + [
            "meta__truth_hash",
            "meta__parent_truth_hash",
            "meta__pipeline_mode",
        ]
    )
)

baseline_feature_columns = identify_feature_columns(baseline_results)

baseline_truth_record = build_truth_record(
    truth_base=baseline_truth,
    row_count=len(baseline_results),
    column_count=baseline_results.shape[1] + 3,
    meta_columns=baseline_meta_columns,
    feature_columns=baseline_feature_columns,
)

BASELINE_TRUTH_HASH = baseline_truth_record["truth_hash"]

baseline_results = stamp_truth_columns(
    baseline_results,
    truth_hash=BASELINE_TRUTH_HASH,
    parent_truth_hash=GOLD_PARENT_TRUTH_HASH,
    pipeline_mode=PIPELINE_MODE,
)

baseline_truth_path = save_truth_record(
    baseline_truth_record,
    truth_dir=TRUTHS_PATH,
    dataset_name=DATASET_NAME,
    layer_name=baseline_truth_layer_name,
)

append_truth_index(
    baseline_truth_record,
    truth_index_path=TRUTH_INDEX_PATH,
)

baseline_summary["baseline_truth_hash"] = BASELINE_TRUTH_HASH
baseline_summary["baseline_truth_path"] = str(baseline_truth_path)
baseline_summary["baseline_process_run_id"] = baseline_process_run_id
baseline_summary["gold_truth_hash"] = GOLD_PARENT_TRUTH_HASH
baseline_summary["gold_truth_path"] = str(GOLD_TRUTH_PATH)
baseline_summary["gold_process_run_id"] = gold_truth.get("process_run_id")
baseline_summary["gold_feature_set_id"] = gold_truth_runtime_facts.get("feature_set_id")

baseline_metadata = {
    "gold_preprocessed_scaled_path": str(GOLD_PREPROCESSED_SCALED_DATA_PATH),
    "gold_fit_path": str(GOLD_FIT_DATA_PATH),
    "baseline_results_path_csv": str(BASELINE_RESULTS_PATH_CSV),
    "baseline_results_path_pickle": str(BASELINE_RESULTS_PATH_PICKLE),
    "baseline_model_artifact_path": str(BASELINE_MODEL_ARTIFACT_PATH),
    "baseline_models_path": str(BASELINE_MODELS_PATH),
    "baseline_thresholds_path": str(BASELINE_THRESHOLDS_PATH),
    "baseline_summary_path": str(BASELINE_SUMMARY_PATH),
    "baseline_metadata_path": str(BASELINE_METADATA_PATH),

    # upstream truth linkage
    "gold_truth_hash": GOLD_PARENT_TRUTH_HASH,
    "gold_truth_path": str(GOLD_TRUTH_PATH),
    "gold_process_run_id": gold_truth.get("process_run_id"),
    "gold_feature_set_id": gold_truth_runtime_facts.get("feature_set_id"),
    "gold_scaler_path": gold_truth_artifact_paths.get("scaler_path"),
    "gold_scaler_kind": gold_truth_runtime_facts.get("scaler_kind_runtime"),
    "gold_recommended_imputation": gold_truth_runtime_facts.get("recommended_imputation"),

    # stage truth linkage
    "baseline_truth_hash": BASELINE_TRUTH_HASH,
    "baseline_truth_path": str(baseline_truth_path),
    "baseline_process_run_id": baseline_process_run_id,
}

baseline_results.to_csv(BASELINE_RESULTS_PATH_CSV, index=False)
baseline_results.to_pickle(BASELINE_RESULTS_PATH_PICKLE)


joblib.dump(baseline_model, BASELINE_MODEL_ARTIFACT_PATH)
joblib.dump(baseline_model, BASELINE_MODELS_PATH)

save_json(baseline_thresholds, BASELINE_THRESHOLDS_PATH)
save_json(baseline_summary, BASELINE_SUMMARY_PATH)
save_json(baseline_metadata, BASELINE_METADATA_PATH)

wandb.save(str(BASELINE_RESULTS_PATH_CSV))
wandb.save(str(BASELINE_RESULTS_PATH_PICKLE))
wandb.save(str(BASELINE_MODEL_ARTIFACT_PATH))
wandb.save(str(BASELINE_MODELS_PATH))
wandb.save(str(BASELINE_THRESHOLDS_PATH))
wandb.save(str(BASELINE_SUMMARY_PATH))
wandb.save(str(BASELINE_METADATA_PATH))
wandb.save(str(baseline_truth_path))

ledger.add(
    kind="step",
    step="save_baseline_outputs",
    message="Saved baseline results, trained Isolation Forest model, thresholds, summary, metadata, and baseline stage truth record.",
    data={
        "baseline_results_path_csv": str(BASELINE_RESULTS_PATH_CSV),
        "baseline_results_path_pickle": str(BASELINE_RESULTS_PATH_PICKLE),
        "baseline_model_artifact_path": str(BASELINE_MODEL_ARTIFACT_PATH),
        "baseline_models_path": str(BASELINE_MODELS_PATH),
        "baseline_thresholds_path": str(BASELINE_THRESHOLDS_PATH),
        "baseline_summary_path": str(BASELINE_SUMMARY_PATH),
        "baseline_metadata_path": str(BASELINE_METADATA_PATH),
        "baseline_truth_hash": BASELINE_TRUTH_HASH,
        "baseline_truth_path": str(baseline_truth_path),
        "result_row_count": int(len(baseline_results)),
        "alert_count_all_rows": baseline_alert_count_all_rows,
        "alert_count_test_rows": baseline_alert_count_test_rows,
    },
    logger=logger,
)

# %% [markdown]
# ----

# %%
ledger.add(
    kind="step",
    step="finalize_baseline_modeling",
    message="Gold baseline modeling notebook complete.",
    data={
        "baseline_metrics": baseline_metrics,
        "baseline_results_path_csv": str(BASELINE_RESULTS_PATH_CSV),
        "baseline_results_path_pickle": str(BASELINE_RESULTS_PATH_PICKLE),
        "baseline_model_artifact_path": str(BASELINE_MODEL_ARTIFACT_PATH),
        "baseline_model_path": str(BASELINE_MODELS_PATH),
    },
    logger=logger,
)

baseline_ledger_path = GOLD_ARTIFACTS_PATH / GOLD_BASELINE_LEDGER_FILE_NAME
ledger.write_json(baseline_ledger_path)

wandb.save(str(baseline_ledger_path))
wandb_run.finish()

# %% [markdown]
# ----

# %%
required_baseline_meta_columns = [
    "meta__truth_hash",
    "meta__parent_truth_hash",
    "meta__pipeline_mode",
]

missing_baseline_meta_columns = [
    column_name
    for column_name in required_baseline_meta_columns
    if column_name not in baseline_results.columns
]
if missing_baseline_meta_columns:
    raise ValueError(
        f"baseline_results is missing required lineage columns: {missing_baseline_meta_columns}"
    )

baseline_results_truth_hash_check = extract_truth_hash(baseline_results)
if baseline_results_truth_hash_check is None:
    raise ValueError("baseline_results does not contain a readable meta__truth_hash value.")

if baseline_results_truth_hash_check != BASELINE_TRUTH_HASH:
    raise ValueError(
        "baseline_results truth hash does not match BASELINE_TRUTH_HASH:\n"
        f"dataframe={baseline_results_truth_hash_check}\n"
        f"record={BASELINE_TRUTH_HASH}"
    )

baseline_parent_values = baseline_results["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()
if not baseline_parent_values:
    raise ValueError("baseline_results is missing populated meta__parent_truth_hash values.")

if len(baseline_parent_values) != 1:
    raise ValueError(f"baseline_results has multiple parent truth hashes: {baseline_parent_values}")

if baseline_parent_values[0] != GOLD_PARENT_TRUTH_HASH:
    raise ValueError(
        "baseline_results parent truth hash does not match GOLD_PARENT_TRUTH_HASH:\n"
        f"dataframe_parent={baseline_parent_values[0]}\n"
        f"gold_parent={GOLD_PARENT_TRUTH_HASH}"
    )

if not Path(baseline_truth_path).exists():
    raise FileNotFoundError(f"Baseline truth file was not created: {baseline_truth_path}")

loaded_baseline_truth = load_json(baseline_truth_path)

if loaded_baseline_truth.get("truth_hash") != BASELINE_TRUTH_HASH:
    raise ValueError(
        "Saved Baseline truth file hash does not match BASELINE_TRUTH_HASH:\n"
        f"file={loaded_baseline_truth.get('truth_hash')}\n"
        f"record={BASELINE_TRUTH_HASH}"
    )

if loaded_baseline_truth.get("parent_truth_hash") != GOLD_PARENT_TRUTH_HASH:
    raise ValueError(
        "Saved Baseline truth file parent hash does not match GOLD_PARENT_TRUTH_HASH:\n"
        f"truth={loaded_baseline_truth.get('parent_truth_hash')}\n"
        f"gold_parent={GOLD_PARENT_TRUTH_HASH}"
    )

saved_baseline_metadata = load_json(BASELINE_METADATA_PATH)
if saved_baseline_metadata.get("baseline_truth_hash") != BASELINE_TRUTH_HASH:
    raise ValueError(
        "baseline_metadata baseline_truth_hash does not match BASELINE_TRUTH_HASH:\n"
        f"metadata={saved_baseline_metadata.get('baseline_truth_hash')}\n"
        f"record={BASELINE_TRUTH_HASH}"
    )

print("Gold Baseline lineage sanity check passed.")

# %% [markdown]
# ----


