"""
pipelines/gold/run_gold_preprocessing.py

Gold preprocessing pipeline runner for the capstone project.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import wandb

from utils.paths import get_paths
from utils.file_io import load_data, save_data, save_json, load_json
from utils.logging_setup import configure_logging, log_layer_paths
from utils.wandb_utils import finalize_wandb_stage
from utils.truths import (
    extract_truth_hash,
    identify_meta_columns,
    identify_feature_columns,
    initialize_layer_truth,
    update_truth_section,
    build_truth_record,
    save_truth_record,
    append_truth_index,
    stamp_truth_columns,
    load_parent_truth_record_from_dataframe,
    get_dataset_name_from_truth,
    get_truth_hash,
    get_pipeline_mode_from_truth,
    make_process_run_id,
)
from utils.pipeline_config_loader import (
    load_pipeline_config,
    build_truth_config_block,
    set_wandb_dir_from_config,
    export_config_snapshot,
)
from utils.postgres_util import get_engine_from_env
from utils.layer_postgres_writer import write_layer_dataframe, prepare_layer_dataframe
from utils.ledger import Ledger
from utils.pipeline.gold_preprocessing import (
    prepare_gold_model_inputs,
    build_gold_support_artifacts,
)
from utils.cascade_row_tracking import ensure_stable_row_id

def _build_default_runtime_inputs(
    *,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pull commonly used Gold preprocessing runtime inputs from config.
    """
    gold_cfg = config["gold_preprocessing"]
    paths_cfg = config["resolved_paths"]
    filenames = config["filenames"]
    pipeline_cfg = config.get(
        "pipeline",
        {
            "execution_mode": "batch",
            "orchestration_mode": "script",
        },
    )

    runtime_inputs = {
        "gold_cfg": gold_cfg,
        "paths_cfg": paths_cfg,
        "filenames": filenames,
        "pipeline_cfg": pipeline_cfg,
        "stage": "gold",
        "layer_name": gold_cfg["layer_name"],
        "gold_version": config["versions"]["gold"],
        "truth_version": config["versions"]["truth"],
        "pipeline_mode": pipeline_cfg["execution_mode"],
        "run_mode": config["runtime"]["mode"],
        "profile": config["runtime"]["profile"],
        "dataset_name_config": config["dataset"]["name"],
        "process_run_id": make_process_run_id(gold_cfg["process_run_id_prefix"]),
        "wandb_project": config["wandb"]["project"],
        "wandb_entity": config["wandb"]["entity"],
        "wandb_run_name": f"{config['versions']['gold']}",
        "silver_train_data_path": Path(paths_cfg["data_silver_train_dir"]),
        "gold_train_data_path": Path(paths_cfg["data_gold_train_dir"]),
        "gold_artifacts_path": Path(paths_cfg["gold_artifacts_dir"]),
        "truths_path": Path(paths_cfg["truths_dir"]),
        "truth_index_path": Path(paths_cfg["truth_index_path"]),
        "logs_path": Path(paths_cfg["logs_root"]),
        "silver_train_data_file_name": filenames["silver_train_file_name"],
        "feature_registry_file_name": filenames["silver_feature_registry_file_name"],
        "gold_preprocessed_file_name": filenames["gold_preprocessed_file_name"],
        "gold_preprocessed_scaled_file_name": filenames["gold_preprocessed_scaled_file_name"],
        "gold_fit_file_name": filenames["gold_fit_file_name"],
        "gold_train_file_name": filenames["gold_train_file_name"],
        "gold_test_file_name": filenames["gold_test_file_name"],
        "reference_profile_file_name": filenames["gold_reference_profile_file_name"],
        "stage2_features_file_name": filenames["gold_stage2_features_file_name"],
        "stage3_sensor_groups_file_name": filenames["gold_stage3_sensor_groups_file_name"],
        "scaler_file_name": filenames["gold_scaler_file_name"],
        "imputer_file_name": filenames["gold_imputer_file_name"],
        "train_fraction": float(gold_cfg["train_fraction"]),
        "split_episode_column": gold_cfg["split_episode_column"],
        "fallback_order_column": gold_cfg["fallback_order_column"],
        "select_numeric_only": bool(gold_cfg["select_numeric_only"]),
        "apply_one_hot_encoding": bool(gold_cfg["apply_one_hot_encoding"]),
        "imputation_method": gold_cfg["imputation_method"],
        "scaler_kind": gold_cfg["scaler_kind"],
        "exclude_feature_columns": list(gold_cfg["exclude_feature_columns"]),
        "write_sql_output": bool(gold_cfg.get("write_sql_output", False)),
        "gold_schema": gold_cfg.get("gold_schema", "gold"),
    }

    return runtime_inputs


def _apply_runtime_overrides(
    runtime_inputs: Dict[str, Any],
    *,
    silver_train_data_file_name: Optional[str] = None,
    feature_registry_file_name: Optional[str] = None,
    gold_preprocessed_file_name: Optional[str] = None,
    gold_preprocessed_scaled_file_name: Optional[str] = None,
    gold_fit_file_name: Optional[str] = None,
    gold_train_file_name: Optional[str] = None,
    gold_test_file_name: Optional[str] = None,
    train_fraction: Optional[float] = None,
    imputation_method: Optional[str] = None,
    scaler_kind: Optional[str] = None,
    write_sql_output: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Apply explicit runtime overrides on top of config-derived defaults.
    """
    updated_inputs = dict(runtime_inputs)

    string_overrides = {
        "silver_train_data_file_name": silver_train_data_file_name,
        "feature_registry_file_name": feature_registry_file_name,
        "gold_preprocessed_file_name": gold_preprocessed_file_name,
        "gold_preprocessed_scaled_file_name": gold_preprocessed_scaled_file_name,
        "gold_fit_file_name": gold_fit_file_name,
        "gold_train_file_name": gold_train_file_name,
        "gold_test_file_name": gold_test_file_name,
        "imputation_method": imputation_method,
        "scaler_kind": scaler_kind,
    }

    for key, value in string_overrides.items():
        if value is not None and str(value).strip() != "":
            updated_inputs[key] = str(value).strip()

    if train_fraction is not None:
        updated_inputs["train_fraction"] = float(train_fraction)

    if write_sql_output is not None:
        updated_inputs["write_sql_output"] = bool(write_sql_output)

    return updated_inputs


def _ensure_stage_directories(runtime_inputs: Dict[str, Any]) -> None:
    """
    Ensure required Gold directories exist.
    """
    runtime_inputs["gold_train_data_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["gold_artifacts_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["truths_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["logs_path"].mkdir(parents=True, exist_ok=True)


def _initialize_gold_logger(paths) -> logging.Logger:
    """
    Create and configure the Gold logger.
    """
    gold_log_path = paths.logs / "gold_preprocessing.log"
    gold_log_path.parent.mkdir(parents=True, exist_ok=True)

    configure_logging(
        "capstone",
        gold_log_path,
        level=logging.DEBUG,
        overwrite_handlers=True,
    )

    logger = logging.getLogger("capstone.gold_preprocessing")
    logger.info("Gold preprocessing stage starting")
    log_layer_paths(paths, current_layer="gold", logger=logger)

    return logger


def _initialize_wandb_run(
    *,
    runtime_inputs: Dict[str, Any],
    silver_data_path: Path,
    logger: logging.Logger,
):
    """
    Start W&B run for Gold preprocessing.
    """
    wandb_run = wandb.init(
        project=runtime_inputs["wandb_project"],
        entity=runtime_inputs["wandb_entity"],
        name=runtime_inputs["wandb_run_name"],
        job_type="gold_preprocessing",
        config={
            "gold_version": runtime_inputs["gold_version"],
            "train_fraction": runtime_inputs["train_fraction"],
            "select_numeric_only": runtime_inputs["select_numeric_only"],
            "apply_one_hot_encoding": runtime_inputs["apply_one_hot_encoding"],
            "imputation_method": runtime_inputs["imputation_method"],
            "scaler_kind": runtime_inputs["scaler_kind"],
            "silver_path": str(silver_data_path),
            "gold_out_dir": str(runtime_inputs["gold_train_data_path"]),
        },
    )

    logger.info("W&B initialized: %s", wandb.run.name)
    return wandb_run


def _load_silver_inputs(
    *,
    runtime_inputs: Dict[str, Any],
    logger: logging.Logger,
):
    """
    Load Silver cleaned dataframe and feature registry.
    """
    silver_data_path = runtime_inputs["silver_train_data_path"] / runtime_inputs["silver_train_data_file_name"]
    if not silver_data_path.exists():
        raise FileNotFoundError(f"Silver parquet not found: {silver_data_path}")

    feature_registry_path = runtime_inputs["gold_artifacts_path"].parent / "silver" / runtime_inputs["feature_registry_file_name"]
    # Fallback to silver artifacts sibling if project structure differs
    if not feature_registry_path.exists():
        feature_registry_path = runtime_inputs["silver_train_data_path"].parent / "silver_artifacts" / runtime_inputs["feature_registry_file_name"]
    # Last fallback: resolved file directly in configured artifacts dir if user placed it there
    if not feature_registry_path.exists():
        candidate = runtime_inputs["gold_artifacts_path"] / runtime_inputs["feature_registry_file_name"]
        if candidate.exists():
            feature_registry_path = candidate

    if not feature_registry_path.exists():
        raise FileNotFoundError(f"Feature registry not found: {feature_registry_path}")

    dataframe = load_data(silver_data_path.parent, silver_data_path.name)
    feature_registry = load_json(feature_registry_path)

    logger.info("Loaded Silver cleaned dataframe: %s | shape=%s", silver_data_path, dataframe.shape)
    logger.info("Loaded Silver feature registry: %s", feature_registry_path)

    return dataframe, feature_registry, silver_data_path, feature_registry_path


def _resolve_parent_truth_and_initialize_gold_truth(
    *,
    dataframe,
    runtime_inputs: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Resolve Silver parent truth and initialize Gold truth object.
    """
    gold_parent_truth_hash = extract_truth_hash(dataframe)

    if gold_parent_truth_hash is None:
        raise ValueError("Gold input dataframe does not contain a readable meta__truth_hash value.")

    silver_dataset_name_series = dataframe["meta__dataset"].dropna().astype("string").str.strip()
    silver_dataset_name_series = silver_dataset_name_series[silver_dataset_name_series != ""]
    if len(silver_dataset_name_series) == 0:
        raise ValueError("Gold input dataframe is missing usable meta__dataset values.")

    silver_dataset_name = str(silver_dataset_name_series.iloc[0]).strip()

    parent_truth = load_parent_truth_record_from_dataframe(
        dataframe=dataframe,
        truth_dir=runtime_inputs["truths_path"],
        parent_layer_name="silver",
        dataset_name=silver_dataset_name,
        column_name="meta__truth_hash",
    )

    dataset_name = get_dataset_name_from_truth(parent_truth)
    gold_parent_truth_hash = get_truth_hash(parent_truth)
    parent_pipeline_mode = get_pipeline_mode_from_truth(parent_truth)
    pipeline_mode = parent_pipeline_mode or runtime_inputs["pipeline_mode"]

    if "meta__pipeline_mode" not in dataframe.columns:
        dataframe["meta__pipeline_mode"] = pipeline_mode
    else:
        dataframe["meta__pipeline_mode"] = dataframe["meta__pipeline_mode"].fillna(pipeline_mode)

    gold_truth = initialize_layer_truth(
        truth_version=runtime_inputs["truth_version"],
        dataset_name=dataset_name,
        layer_name=runtime_inputs["layer_name"],
        process_run_id=runtime_inputs["process_run_id"],
        pipeline_mode=pipeline_mode,
        parent_truth_hash=gold_parent_truth_hash,
    )

    gold_truth = update_truth_section(
        gold_truth,
        "config_snapshot",
        {
            "gold_version": runtime_inputs["gold_version"],
            "train_fraction": runtime_inputs["train_fraction"],
            "select_numeric_only": runtime_inputs["select_numeric_only"],
            "apply_one_hot_encoding": runtime_inputs["apply_one_hot_encoding"],
            "imputation_method": runtime_inputs["imputation_method"],
            "scaler_kind": runtime_inputs["scaler_kind"],
            "pipeline_mode": pipeline_mode,
        },
    )

    gold_truth = update_truth_section(
        gold_truth,
        "runtime_facts",
        {
            "parent_layer_name": "silver",
            "parent_truth_hash": gold_parent_truth_hash,
            "dataset_name_from_parent_truth": dataset_name,
        },
    )

    logger.info("Resolved Silver parent truth hash: %s", gold_parent_truth_hash)
    logger.info("Resolved Gold dataset name from Silver truth: %s", dataset_name)

    return {
        "dataframe": dataframe,
        "parent_truth": parent_truth,
        "dataset_name": dataset_name,
        "gold_parent_truth_hash": gold_parent_truth_hash,
        "pipeline_mode": pipeline_mode,
        "gold_truth": gold_truth,
    }


def _save_pickle_object(obj, output_path: Path) -> str:
    """
    Save Python object with pickle.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file_handle:
        pickle.dump(obj, file_handle)
    return str(output_path)


def _build_and_save_gold_truth(
    *,
    stamped_dataframe,
    gold_truth: Dict[str, Any],
    dataset_name: str,
    gold_parent_truth_hash: str,
    pipeline_mode: str,
    runtime_inputs: Dict[str, Any],
    preprocessing_info: Dict[str, Any],
    support_artifacts: Dict[str, Any],
    silver_data_path: Path,
    feature_registry_path: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build Gold truth record, stamp truth columns, and save truth artifact.
    """
    gold_truth = update_truth_section(
        gold_truth,
        "runtime_facts",
        {
            "silver_input_path": str(silver_data_path),
            "feature_registry_path": str(feature_registry_path),
            "dataset_name_final": dataset_name,
            "selected_feature_columns": preprocessing_info["selected_feature_columns"],
        },
    )

    gold_truth = update_truth_section(
        gold_truth,
        "artifact_paths",
        {
            "gold_output_dir": str(runtime_inputs["gold_train_data_path"]),
            "gold_preprocessed_file_name": runtime_inputs["gold_preprocessed_file_name"],
            "gold_preprocessed_scaled_file_name": runtime_inputs["gold_preprocessed_scaled_file_name"],
            "gold_fit_file_name": runtime_inputs["gold_fit_file_name"],
            "gold_train_file_name": runtime_inputs["gold_train_file_name"],
            "gold_test_file_name": runtime_inputs["gold_test_file_name"],
            "reference_profile_file_name": runtime_inputs["reference_profile_file_name"],
            "stage2_features_file_name": runtime_inputs["stage2_features_file_name"],
            "stage3_sensor_groups_file_name": runtime_inputs["stage3_sensor_groups_file_name"],
            "scaler_file_name": runtime_inputs["scaler_file_name"],
            "imputer_file_name": runtime_inputs["imputer_file_name"],
        },
    )

    gold_truth = update_truth_section(
        gold_truth,
        "gold_preprocessing",
        {
            "preprocessing_info": preprocessing_info,
            "support_artifacts_summary": {
                "baseline_feature_count": len(support_artifacts["baseline_feature_columns"]),
                "stage2_feature_count": len(support_artifacts["stage2_feature_columns"]),
                "stage3_sensor_group_count": len(support_artifacts["stage3_sensor_groups"]),
            },
        },
    )

    gold_meta_columns = sorted(
        set(
            identify_meta_columns(stamped_dataframe)
            + ["meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode"]
        )
    )
    gold_feature_columns = identify_feature_columns(stamped_dataframe)

    gold_truth_record = build_truth_record(
        truth_base=gold_truth,
        row_count=len(stamped_dataframe),
        column_count=stamped_dataframe.shape[1] + 3,
        meta_columns=gold_meta_columns,
        feature_columns=gold_feature_columns,
    )

    gold_truth_hash = gold_truth_record["truth_hash"]

    stamped_dataframe = stamp_truth_columns(
        stamped_dataframe,
        truth_hash=gold_truth_hash,
        parent_truth_hash=gold_parent_truth_hash,
        pipeline_mode=pipeline_mode,
    )

    gold_truth_path = save_truth_record(
        gold_truth_record,
        truth_dir=runtime_inputs["truths_path"],
        dataset_name=dataset_name,
        layer_name=runtime_inputs["layer_name"],
    )

    append_truth_index(
        gold_truth_record,
        truth_index_path=runtime_inputs["truth_index_path"],
    )

    logger.info("Gold truth hash: %s", gold_truth_hash)
    logger.info("Gold truth path: %s", gold_truth_path)

    return {
        "dataframe": stamped_dataframe,
        "gold_truth_record": gold_truth_record,
        "gold_truth_hash": gold_truth_hash,
        "gold_truth_path": str(gold_truth_path),
    }


def _run_lineage_sanity_checks(
    *,
    dataframe,
    gold_truth_hash: str,
    gold_parent_truth_hash: str,
    gold_truth_path: str,
):
    """
    Final lineage sanity checks before stage completion.
    """
    required_gold_meta_columns = [
        "meta__truth_hash",
        "meta__parent_truth_hash",
        "meta__pipeline_mode",
    ]

    missing_gold_meta_columns = [
        column_name
        for column_name in required_gold_meta_columns
        if column_name not in dataframe.columns
    ]
    if missing_gold_meta_columns:
        raise ValueError(
            f"Gold dataframe is missing required lineage columns: {missing_gold_meta_columns}"
        )

    gold_dataframe_truth_hash = extract_truth_hash(dataframe)
    if gold_dataframe_truth_hash is None:
        raise ValueError("Gold dataframe does not contain a readable meta__truth_hash value.")

    if gold_dataframe_truth_hash != gold_truth_hash:
        raise ValueError(
            "Gold dataframe truth hash does not match gold truth record:\n"
            f"dataframe={gold_dataframe_truth_hash}\n"
            f"record={gold_truth_hash}"
        )

    gold_parent_values = dataframe["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()
    if not gold_parent_values:
        raise ValueError("Gold dataframe is missing populated meta__parent_truth_hash values.")

    if len(gold_parent_values) != 1:
        raise ValueError(
            "Gold dataframe has multiple parent truth hashes:\n"
            f"{gold_parent_values}"
        )

    if gold_parent_values[0] != gold_parent_truth_hash:
        raise ValueError(
            "Gold dataframe parent truth hash does not match Gold parent truth hash:\n"
            f"dataframe_parent={gold_parent_values[0]}\n"
            f"gold_parent_truth={gold_parent_truth_hash}"
        )

    if not Path(gold_truth_path).exists():
        raise FileNotFoundError(f"Gold truth file was not created: {gold_truth_path}")

    loaded_gold_truth = load_json(gold_truth_path)

    if loaded_gold_truth.get("truth_hash") != gold_truth_hash:
        raise ValueError(
            "Saved Gold truth file hash does not match truth record:\n"
            f"file={loaded_gold_truth.get('truth_hash')}\n"
            f"record={gold_truth_hash}"
        )

    if loaded_gold_truth.get("parent_truth_hash") != gold_parent_truth_hash:
        raise ValueError(
            "Saved Gold truth file parent hash does not match expected parent:\n"
            f"truth={loaded_gold_truth.get('parent_truth_hash')}\n"
            f"gold_parent={gold_parent_truth_hash}"
        )


def _optionally_write_sql_output(
    *,
    dataframe,
    runtime_inputs: Dict[str, Any],
    gold_truth_hash: str,
    gold_parent_truth_hash: str,
    pipeline_mode: str,
    logger: logging.Logger,
) -> Optional[str]:
    """
    Optional PostgreSQL persistence for Gold output.
    """
    if not runtime_inputs["write_sql_output"]:
        return None

    engine = get_engine_from_env()

    gold_sql_dataframe = prepare_layer_dataframe(
        dataframe,
        truth_hash=gold_truth_hash,
        parent_truth_hash=gold_parent_truth_hash,
        pipeline_mode=pipeline_mode,
        process_run_id=runtime_inputs["process_run_id"],
        add_loaded_at_column=True,
    )

    gold_table_name = write_layer_dataframe(
        engine=engine,
        dataframe=gold_sql_dataframe,
        schema=runtime_inputs["gold_schema"],
        dataset_name=dataframe["meta__dataset"].dropna().astype(str).iloc[0],
        if_exists="replace",
        index=False,
    )

    logger.info(
        "Wrote Gold preprocessing SQL table: %s.%s",
        runtime_inputs["gold_schema"],
        gold_table_name,
    )

    return gold_table_name


def run_gold_preprocessing(
    *,
    config_root: Optional[Path] = None,
    dataset: str = "pump",
    mode: str = "train",
    profile: str = "default",
    project_root: Optional[Path] = None,
    silver_train_data_file_name: Optional[str] = None,
    feature_registry_file_name: Optional[str] = None,
    gold_preprocessed_file_name: Optional[str] = None,
    gold_preprocessed_scaled_file_name: Optional[str] = None,
    gold_fit_file_name: Optional[str] = None,
    gold_train_file_name: Optional[str] = None,
    gold_test_file_name: Optional[str] = None,
    train_fraction: Optional[float] = None,
    imputation_method: Optional[str] = None,
    scaler_kind: Optional[str] = None,
    write_sql_output: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run the Gold preprocessing stage.
    """
    paths = get_paths()

    config_root = config_root or paths.configs
    project_root = project_root or paths.root

    config = load_pipeline_config(
        config_root=config_root,
        stage="gold_preprocessing",
        dataset=dataset,
        mode=mode,
        profile=profile,
        project_root=project_root,
    ).data

    runtime_inputs = _build_default_runtime_inputs(config=config)
    runtime_inputs = _apply_runtime_overrides(
        runtime_inputs,
        silver_train_data_file_name=silver_train_data_file_name,
        feature_registry_file_name=feature_registry_file_name,
        gold_preprocessed_file_name=gold_preprocessed_file_name,
        gold_preprocessed_scaled_file_name=gold_preprocessed_scaled_file_name,
        gold_fit_file_name=gold_fit_file_name,
        gold_train_file_name=gold_train_file_name,
        gold_test_file_name=gold_test_file_name,
        train_fraction=train_fraction,
        imputation_method=imputation_method,
        scaler_kind=scaler_kind,
        write_sql_output=write_sql_output,
    )

    _ensure_stage_directories(runtime_inputs)

    logger = _initialize_gold_logger(paths)

    truthed_config = build_truth_config_block(config)
    truthed_config["pipeline"] = runtime_inputs["pipeline_cfg"]

    set_wandb_dir_from_config(config)
    export_config_snapshot(
        config,
        output_path=runtime_inputs["gold_artifacts_path"] / f"{dataset}__gold_preprocessing__resolved_config.yaml",
    )

    dataframe, feature_registry, silver_data_path, feature_registry_path = _load_silver_inputs(
        runtime_inputs=runtime_inputs,
        logger=logger,
    )

    wandb_run = _initialize_wandb_run(
        runtime_inputs=runtime_inputs,
        silver_data_path=silver_data_path,
        logger=logger,
    )

    ledger = Ledger(stage=runtime_inputs["stage"], recipe_id="gold_preprocessing")
    ledger.add(
        kind="step",
        step="load_silver",
        message="Loaded Silver cleaned dataset and feature registry",
        data={
            "silver_path": str(silver_data_path),
            "feature_registry_path": str(feature_registry_path),
            "shape": list(dataframe.shape),
            "feature_count": int(feature_registry.get("feature_count", 0)),
        },
        logger=logger,
    )

    parent_context = _resolve_parent_truth_and_initialize_gold_truth(
        dataframe=dataframe,
        runtime_inputs=runtime_inputs,
        logger=logger,
    )

    dataframe = parent_context["dataframe"]
    dataset_name = parent_context["dataset_name"]
    gold_parent_truth_hash = parent_context["gold_parent_truth_hash"]
    pipeline_mode = parent_context["pipeline_mode"]
    gold_truth = parent_context["gold_truth"]

    frames, preprocessing_info, learned_objects = prepare_gold_model_inputs(
        dataframe,
        feature_registry=feature_registry,
        train_fraction=runtime_inputs["train_fraction"],
        split_episode_column=runtime_inputs["split_episode_column"],
        split_group_columns=("meta__asset_id", "meta__run_id"),
        fallback_order_column=runtime_inputs["fallback_order_column"],
        select_numeric_only=runtime_inputs["select_numeric_only"],
        apply_one_hot_encoding=runtime_inputs["apply_one_hot_encoding"],
        one_hot_columns=feature_registry.get("one_hot_encoding_columns", []),
        imputation_method=runtime_inputs["imputation_method"],
        scaler_kind=runtime_inputs["scaler_kind"],
        exclude_feature_columns=runtime_inputs["exclude_feature_columns"],
    )

    support_artifacts = build_gold_support_artifacts(
        frames["gold_preprocessed_scaled"],
        selected_feature_columns=preprocessing_info["selected_feature_columns"],
        train_mask=frames["gold_preprocessed_scaled"]["meta__train_mask"],
        baseline_feature_columns=preprocessing_info["selected_feature_columns"],
    )

    stamped_truth_payload = _build_and_save_gold_truth(
        stamped_dataframe=frames["gold_preprocessed_scaled"],
        gold_truth=gold_truth,
        dataset_name=dataset_name,
        gold_parent_truth_hash=gold_parent_truth_hash,
        pipeline_mode=pipeline_mode,
        runtime_inputs=runtime_inputs,
        preprocessing_info=preprocessing_info,
        support_artifacts=support_artifacts,
        silver_data_path=silver_data_path,
        feature_registry_path=feature_registry_path,
        logger=logger,
    )

    stamped_scaled_dataframe = stamped_truth_payload["dataframe"]
    gold_truth_hash = stamped_truth_payload["gold_truth_hash"]
    gold_truth_path = stamped_truth_payload["gold_truth_path"]

    # Stamp same lineage columns into all output frames for consistency
    for frame_key in frames:
        frames[frame_key] = frames[frame_key].copy()
        frames[frame_key]["meta__truth_hash"] = gold_truth_hash
        frames[frame_key]["meta__parent_truth_hash"] = gold_parent_truth_hash
        frames[frame_key]["meta__pipeline_mode"] = pipeline_mode

    frames["gold_preprocessed_scaled"] = stamped_scaled_dataframe

    save_data(frames["gold_preprocessed"], runtime_inputs["gold_train_data_path"], runtime_inputs["gold_preprocessed_file_name"])
    save_data(frames["gold_preprocessed_scaled"], runtime_inputs["gold_train_data_path"], runtime_inputs["gold_preprocessed_scaled_file_name"])
    save_data(frames["gold_fit"], runtime_inputs["gold_train_data_path"], runtime_inputs["gold_fit_file_name"])
    save_data(frames["gold_train"], runtime_inputs["gold_train_data_path"], runtime_inputs["gold_train_file_name"])
    save_data(frames["gold_test"], runtime_inputs["gold_train_data_path"], runtime_inputs["gold_test_file_name"])

    reference_profile_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["reference_profile_file_name"]
    stage2_features_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["stage2_features_file_name"]
    stage3_sensor_groups_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["stage3_sensor_groups_file_name"]
    scaler_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["scaler_file_name"]
    imputer_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["imputer_file_name"]

    save_json(support_artifacts["reference_profile"], reference_profile_path)
    save_json(
        {
            "stage2_feature_columns": support_artifacts["stage2_feature_columns"],
            "stage2_info": support_artifacts["stage2_info"],
        },
        stage2_features_path,
    )
    save_json(support_artifacts["stage3_sensor_groups"], stage3_sensor_groups_path)

    _save_pickle_object(learned_objects["scaler"], scaler_path)
    _save_pickle_object(learned_objects["imputer"], imputer_path)

    saved_ledger_path = ledger.write_json(
        runtime_inputs["gold_artifacts_path"] / f"gold_preprocessing__{dataset_name}__ledger.json"
    )

    _run_lineage_sanity_checks(
        dataframe=frames["gold_preprocessed_scaled"],
        gold_truth_hash=gold_truth_hash,
        gold_parent_truth_hash=gold_parent_truth_hash,
        gold_truth_path=gold_truth_path,
    )

    sql_table_name = _optionally_write_sql_output(
        dataframe=frames["gold_preprocessed_scaled"],
        runtime_inputs=runtime_inputs,
        gold_truth_hash=gold_truth_hash,
        gold_parent_truth_hash=gold_parent_truth_hash,
        pipeline_mode=pipeline_mode,
        logger=logger,
    )

    finalize_wandb_stage(
        run=wandb_run,
        stage=runtime_inputs["stage"],
        dataframe=frames["gold_preprocessed_scaled"],
        project_root=paths.root,
        logs_dir=paths.logs,
        dataset_dirs=[runtime_inputs["gold_train_data_path"]],
        dataset_artifact_name=f"{dataset_name}-{runtime_inputs['stage']}-dataset",
        logger=logger,
        notebook_path=None,
        aliases=("latest",),
        table_key=None,
        table_n=15,
        profile=False,
    )

    wandb_run.finish()

    summary = {
        "status": "success",
        "layer_name": runtime_inputs["layer_name"],
        "dataset_name": dataset_name,
        "row_count": int(frames["gold_preprocessed_scaled"].shape[0]),
        "column_count": int(frames["gold_preprocessed_scaled"].shape[1]),
        "gold_preprocessed_path": str(runtime_inputs["gold_train_data_path"] / runtime_inputs["gold_preprocessed_file_name"]),
        "gold_preprocessed_scaled_path": str(runtime_inputs["gold_train_data_path"] / runtime_inputs["gold_preprocessed_scaled_file_name"]),
        "gold_fit_path": str(runtime_inputs["gold_train_data_path"] / runtime_inputs["gold_fit_file_name"]),
        "gold_train_path": str(runtime_inputs["gold_train_data_path"] / runtime_inputs["gold_train_file_name"]),
        "gold_test_path": str(runtime_inputs["gold_train_data_path"] / runtime_inputs["gold_test_file_name"]),
        "reference_profile_path": str(reference_profile_path),
        "stage2_features_path": str(stage2_features_path),
        "stage3_sensor_groups_path": str(stage3_sensor_groups_path),
        "scaler_path": str(scaler_path),
        "imputer_path": str(imputer_path),
        "truth_hash": gold_truth_hash,
        "truth_path": gold_truth_path,
        "parent_truth_hash": gold_parent_truth_hash,
        "process_run_id": runtime_inputs["process_run_id"],
        "selected_feature_count": int(len(preprocessing_info["selected_feature_columns"])),
        "stage2_feature_count": int(len(support_artifacts["stage2_feature_columns"])),
        "sql_table_name": sql_table_name,
        "ledger_path": str(saved_ledger_path),
    }

    logger.info("Gold preprocessing stage completed successfully.")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for Gold preprocessing stage execution.
    """
    parser = argparse.ArgumentParser(description="Run Gold preprocessing stage.")

    parser.add_argument("--dataset", default="pump")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--project-root", default=None)

    parser.add_argument("--silver-train-data-file-name", default=None)
    parser.add_argument("--feature-registry-file-name", default=None)
    parser.add_argument("--gold-preprocessed-file-name", default=None)
    parser.add_argument("--gold-preprocessed-scaled-file-name", default=None)
    parser.add_argument("--gold-fit-file-name", default=None)
    parser.add_argument("--gold-train-file-name", default=None)
    parser.add_argument("--gold-test-file-name", default=None)

    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--imputation-method", default=None)
    parser.add_argument("--scaler-kind", default=None)

    parser.add_argument(
        "--write-sql-output",
        default=None,
        help="true/false",
    )

    return parser


def _parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    """
    Parse optional CLI boolean.
    """
    if value is None:
        return None

    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Could not parse boolean value: {value}")


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    result = run_gold_preprocessing(
        config_root=Path(args.config_root) if args.config_root else None,
        dataset=args.dataset,
        mode=args.mode,
        profile=args.profile,
        project_root=Path(args.project_root) if args.project_root else None,
        silver_train_data_file_name=args.silver_train_data_file_name,
        feature_registry_file_name=args.feature_registry_file_name,
        gold_preprocessed_file_name=args.gold_preprocessed_file_name,
        gold_preprocessed_scaled_file_name=args.gold_preprocessed_scaled_file_name,
        gold_fit_file_name=args.gold_fit_file_name,
        gold_train_file_name=args.gold_train_file_name,
        gold_test_file_name=args.gold_test_file_name,
        train_fraction=args.train_fraction,
        imputation_method=args.imputation_method,
        scaler_kind=args.scaler_kind,
        write_sql_output=_parse_optional_bool(args.write_sql_output),
    )

    print(json.dumps(result, indent=2))


'''
# Sample Runs 

# Default Run
python -m pipelines.gold.run_gold_preprocessing \
  --dataset pump \
  --mode train \
  --profile default

# Override Split and Scaler
python -m pipelines.gold.run_gold_preprocessing \
  --dataset pump \
  --mode train \
  --profile default \
  --train-fraction 0.70 \
  --imputation-method medium \
  --scaler-kind robust

# Override output file names
python -m pipelines.gold.run_gold_preprocessing \
  --dataset pump \
  --mode train \
  --profile default \
  --gold-preprocessed-file-name pump__gold__preprocessed.parquet \
  --gold-preprocessed-scaled-file-name pump__gold__preprocessed_scaled.parquet \
  --gold-fit-file-name pump__gold__fit.parquet \
  --gold-train-file-name pump__gold__train.parquet \
  --gold-test-file-name pump__gold__test.parquet

# SQL Write Output
python -m pipelines.gold.run_gold_preprocessing \
  --dataset pump \
  --mode train \
  --profile default \
  --write-sql-output true


'''