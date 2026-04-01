"""
pipelines/gold/run_gold_cascade_modeling.py

Gold cascade modeling pipeline runner for the capstone project.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
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
from utils.pipeline.gold_cascade_modeling import run_cascade_pipeline, build_cascade_summary
from utils.cascade_row_tracking import ensure_stable_row_id

def _build_default_runtime_inputs(
    *,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pull commonly used Gold cascade runtime inputs from config.
    """
    cascade_cfg = config["gold_cascade_modeling"]
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
        "cascade_cfg": cascade_cfg,
        "paths_cfg": paths_cfg,
        "filenames": filenames,
        "pipeline_cfg": pipeline_cfg,
        "stage": "gold_cascade",
        "layer_name": cascade_cfg["layer_name"],
        "gold_version": config["versions"]["gold"],
        "truth_version": config["versions"]["truth"],
        "pipeline_mode": pipeline_cfg["execution_mode"],
        "run_mode": config["runtime"]["mode"],
        "profile": config["runtime"]["profile"],
        "dataset_name_config": config["dataset"]["name"],
        "process_run_id": make_process_run_id(cascade_cfg["process_run_id_prefix"]),
        "wandb_project": config["wandb"]["project"],
        "wandb_entity": config["wandb"]["entity"],
        "wandb_run_name": f"{config['versions']['gold']}_cascade",
        "gold_train_data_path": Path(paths_cfg["data_gold_train_dir"]),
        "gold_artifacts_path": Path(paths_cfg["gold_artifacts_dir"]),
        "truths_path": Path(paths_cfg["truths_dir"]),
        "truth_index_path": Path(paths_cfg["truth_index_path"]),
        "logs_path": Path(paths_cfg["logs_root"]),
        "gold_preprocessed_scaled_file_name": filenames["gold_preprocessed_scaled_file_name"],
        "gold_fit_file_name": filenames["gold_fit_file_name"],
        "gold_train_file_name": filenames["gold_train_file_name"],
        "gold_test_file_name": filenames["gold_test_file_name"],
        "reference_profile_file_name": filenames["gold_reference_profile_file_name"],
        "stage2_features_file_name": filenames["gold_stage2_features_file_name"],
        "stage3_sensor_groups_file_name": filenames["gold_stage3_sensor_groups_file_name"],
        "cascade_stage1_model_file_name": filenames["gold_cascade_stage1_model_file_name"],
        "cascade_stage2_model_file_name": filenames["gold_cascade_stage2_model_file_name"],
        "cascade_summary_file_name": filenames["gold_cascade_summary_file_name"],
        "cascade_scored_fit_file_name": filenames["gold_cascade_scored_fit_file_name"],
        "cascade_scored_train_file_name": filenames["gold_cascade_scored_train_file_name"],
        "cascade_scored_test_file_name": filenames["gold_cascade_scored_test_file_name"],
        "cascade_scored_all_file_name": filenames["gold_cascade_scored_all_file_name"],
        "variant": cascade_cfg.get("variant", "default"),
        "label_column": cascade_cfg.get("label_column", "anomaly_flag"),
        "stage1_model_params": {
            "n_estimators": int(cascade_cfg["stage1_n_estimators"]),
            "contamination": cascade_cfg["stage1_contamination"],
            "max_samples": cascade_cfg["stage1_max_samples"],
            "max_features": float(cascade_cfg["stage1_max_features"]),
            "bootstrap": bool(cascade_cfg["stage1_bootstrap"]),
            "random_state": int(cascade_cfg["stage1_random_state"]),
            "n_jobs": int(cascade_cfg["stage1_n_jobs"]),
        },
        "stage2_model_params": {
            "n_estimators": int(cascade_cfg["stage2_n_estimators"]),
            "contamination": cascade_cfg["stage2_contamination"],
            "max_samples": cascade_cfg["stage2_max_samples"],
            "max_features": float(cascade_cfg["stage2_max_features"]),
            "bootstrap": bool(cascade_cfg["stage2_bootstrap"]),
            "random_state": int(cascade_cfg["stage2_random_state"]),
            "n_jobs": int(cascade_cfg["stage2_n_jobs"]),
        },
        "write_sql_output": bool(cascade_cfg.get("write_sql_output", False)),
        "gold_schema": cascade_cfg.get("gold_schema", "gold"),
    }

    return runtime_inputs


def _apply_runtime_overrides(
    runtime_inputs: Dict[str, Any],
    *,
    variant: Optional[str] = None,
    write_sql_output: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Apply explicit runtime overrides.
    """
    updated_inputs = dict(runtime_inputs)

    if variant is not None and str(variant).strip() != "":
        updated_inputs["variant"] = str(variant).strip()

    if write_sql_output is not None:
        updated_inputs["write_sql_output"] = bool(write_sql_output)

    return updated_inputs


def _ensure_stage_directories(runtime_inputs: Dict[str, Any]) -> None:
    runtime_inputs["gold_train_data_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["gold_artifacts_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["truths_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["logs_path"].mkdir(parents=True, exist_ok=True)


def _initialize_cascade_logger(paths) -> logging.Logger:
    cascade_log_path = paths.logs / "gold_modeling_cascade.log"
    cascade_log_path.parent.mkdir(parents=True, exist_ok=True)

    configure_logging(
        "capstone",
        cascade_log_path,
        level=logging.DEBUG,
        overwrite_handlers=True,
    )

    logger = logging.getLogger("capstone.gold_cascade")
    logger.info("Gold cascade stage starting")
    log_layer_paths(paths, current_layer="gold", logger=logger)
    return logger


def _initialize_wandb_run(
    *,
    runtime_inputs: Dict[str, Any],
    scaled_gold_path: Path,
    logger: logging.Logger,
):
    wandb_run = wandb.init(
        project=runtime_inputs["wandb_project"],
        entity=runtime_inputs["wandb_entity"],
        name=f"{runtime_inputs['wandb_run_name']}_{runtime_inputs['variant']}",
        job_type="gold_cascade_modeling",
        config={
            "gold_version": runtime_inputs["gold_version"],
            "variant": runtime_inputs["variant"],
            "stage1_model_params": runtime_inputs["stage1_model_params"],
            "stage2_model_params": runtime_inputs["stage2_model_params"],
            "scaled_gold_path": str(scaled_gold_path),
            "gold_artifacts_dir": str(runtime_inputs["gold_artifacts_path"]),
        },
    )
    logger.info("W&B initialized: %s", wandb.run.name)
    return wandb_run


def _load_gold_inputs(
    *,
    runtime_inputs: Dict[str, Any],
    logger: logging.Logger,
):
    scaled_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["gold_preprocessed_scaled_file_name"]
    fit_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["gold_fit_file_name"]
    train_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["gold_train_file_name"]
    test_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["gold_test_file_name"]

    reference_profile_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["reference_profile_file_name"]
    stage2_features_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["stage2_features_file_name"]
    stage3_sensor_groups_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["stage3_sensor_groups_file_name"]

    for required_path in [
        scaled_path, fit_path, train_path, test_path,
        reference_profile_path, stage2_features_path, stage3_sensor_groups_path
    ]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required cascade input not found: {required_path}")

    all_dataframe = load_data(scaled_path.parent, scaled_path.name)
    fit_dataframe = load_data(fit_path.parent, fit_path.name)
    train_dataframe = load_data(train_path.parent, train_path.name)
    test_dataframe = load_data(test_path.parent, test_path.name)

    reference_profile = load_json(reference_profile_path)
    stage2_features_payload = load_json(stage2_features_path)
    stage3_sensor_groups = load_json(stage3_sensor_groups_path)

    logger.info("Loaded Gold scaled dataframe: %s | shape=%s", scaled_path, all_dataframe.shape)
    logger.info("Loaded Gold fit dataframe: %s | shape=%s", fit_path, fit_dataframe.shape)
    logger.info("Loaded Gold train dataframe: %s | shape=%s", train_path, train_dataframe.shape)
    logger.info("Loaded Gold test dataframe: %s | shape=%s", test_path, test_dataframe.shape)

    return {
        "scaled_path": scaled_path,
        "fit_path": fit_path,
        "train_path": train_path,
        "test_path": test_path,
        "all_dataframe": all_dataframe,
        "fit_dataframe": fit_dataframe,
        "train_dataframe": train_dataframe,
        "test_dataframe": test_dataframe,
        "reference_profile": reference_profile,
        "stage2_features_payload": stage2_features_payload,
        "stage3_sensor_groups": stage3_sensor_groups,
    }


def _resolve_parent_truth_and_initialize_cascade_truth(
    *,
    dataframe,
    runtime_inputs: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    cascade_parent_truth_hash = extract_truth_hash(dataframe)
    if cascade_parent_truth_hash is None:
        raise ValueError("Cascade input dataframe does not contain a readable meta__truth_hash value.")

    dataset_name_series = dataframe["meta__dataset"].dropna().astype("string").str.strip()
    dataset_name_series = dataset_name_series[dataset_name_series != ""]
    if len(dataset_name_series) == 0:
        raise ValueError("Cascade input dataframe is missing usable meta__dataset values.")

    gold_dataset_name = str(dataset_name_series.iloc[0]).strip()

    parent_truth = load_parent_truth_record_from_dataframe(
        dataframe=dataframe,
        truth_dir=runtime_inputs["truths_path"],
        parent_layer_name="gold_preprocessing",
        dataset_name=gold_dataset_name,
        column_name="meta__truth_hash",
    )

    dataset_name = get_dataset_name_from_truth(parent_truth)
    cascade_parent_truth_hash = get_truth_hash(parent_truth)
    parent_pipeline_mode = get_pipeline_mode_from_truth(parent_truth)
    pipeline_mode = parent_pipeline_mode or runtime_inputs["pipeline_mode"]

    cascade_truth = initialize_layer_truth(
        truth_version=runtime_inputs["truth_version"],
        dataset_name=dataset_name,
        layer_name=runtime_inputs["layer_name"],
        process_run_id=runtime_inputs["process_run_id"],
        pipeline_mode=pipeline_mode,
        parent_truth_hash=cascade_parent_truth_hash,
    )

    cascade_truth = update_truth_section(
        cascade_truth,
        "config_snapshot",
        {
            "gold_version": runtime_inputs["gold_version"],
            "variant": runtime_inputs["variant"],
            "stage1_model_params": runtime_inputs["stage1_model_params"],
            "stage2_model_params": runtime_inputs["stage2_model_params"],
            "pipeline_mode": pipeline_mode,
        },
    )

    cascade_truth = update_truth_section(
        cascade_truth,
        "runtime_facts",
        {
            "parent_layer_name": "gold_preprocessing",
            "parent_truth_hash": cascade_parent_truth_hash,
            "dataset_name_from_parent_truth": dataset_name,
        },
    )

    logger.info("Resolved Gold preprocessing parent truth hash: %s", cascade_parent_truth_hash)
    logger.info("Resolved cascade dataset name from parent truth: %s", dataset_name)

    return {
        "dataset_name": dataset_name,
        "cascade_parent_truth_hash": cascade_parent_truth_hash,
        "pipeline_mode": pipeline_mode,
        "cascade_truth": cascade_truth,
    }


def _resolve_stage1_feature_columns(all_dataframe: pd.DataFrame) -> list[str]:
    """
    First-pass Stage 1 features: numeric non-meta columns excluding target/helper fields.
    """
    excluded = {
        "anomaly_flag",
        "baseline_anomaly_score",
        "baseline_predicted_anomaly",
    }

    stage1_features: list[str] = []
    for column_name in all_dataframe.columns:
        if column_name.startswith("meta__"):
            continue
        if column_name in excluded:
            continue
        if pd.api.types.is_numeric_dtype(all_dataframe[column_name]):
            stage1_features.append(column_name)

    if len(stage1_features) == 0:
        raise ValueError("No Stage 1 feature columns resolved from Gold dataframe.")

    return stage1_features


def _save_pickle_object(obj, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file_handle:
        pickle.dump(obj, file_handle)
    return str(output_path)


def _build_and_save_cascade_truth(
    *,
    scored_all_dataframe,
    cascade_truth: Dict[str, Any],
    dataset_name: str,
    cascade_parent_truth_hash: str,
    pipeline_mode: str,
    runtime_inputs: Dict[str, Any],
    cascade_results: Dict[str, Any],
    stage1_feature_columns: list[str],
    stage2_feature_columns: list[str],
    scaled_gold_path: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    cascade_truth = update_truth_section(
        cascade_truth,
        "runtime_facts",
        {
            "scaled_gold_input_path": str(scaled_gold_path),
            "dataset_name_final": dataset_name,
            "stage1_feature_columns": stage1_feature_columns,
            "stage2_feature_columns": stage2_feature_columns,
        },
    )

    cascade_truth = update_truth_section(
        cascade_truth,
        "artifact_paths",
        {
            "cascade_stage1_model_file_name": runtime_inputs["cascade_stage1_model_file_name"],
            "cascade_stage2_model_file_name": runtime_inputs["cascade_stage2_model_file_name"],
            "cascade_summary_file_name": runtime_inputs["cascade_summary_file_name"],
            "cascade_scored_fit_file_name": runtime_inputs["cascade_scored_fit_file_name"],
            "cascade_scored_train_file_name": runtime_inputs["cascade_scored_train_file_name"],
            "cascade_scored_test_file_name": runtime_inputs["cascade_scored_test_file_name"],
            "cascade_scored_all_file_name": runtime_inputs["cascade_scored_all_file_name"],
        },
    )

    cascade_truth = update_truth_section(
        cascade_truth,
        "cascade_modeling",
        {
            "summary": cascade_results["summary"],
        },
    )

    cascade_meta_columns = sorted(
        set(
            identify_meta_columns(scored_all_dataframe)
            + ["meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode"]
        )
    )
    cascade_feature_columns = identify_feature_columns(scored_all_dataframe)

    cascade_truth_record = build_truth_record(
        truth_base=cascade_truth,
        row_count=len(scored_all_dataframe),
        column_count=scored_all_dataframe.shape[1] + 3,
        meta_columns=cascade_meta_columns,
        feature_columns=cascade_feature_columns,
    )

    cascade_truth_hash = cascade_truth_record["truth_hash"]

    scored_all_dataframe = stamp_truth_columns(
        scored_all_dataframe,
        truth_hash=cascade_truth_hash,
        parent_truth_hash=cascade_parent_truth_hash,
        pipeline_mode=pipeline_mode,
    )

    cascade_truth_path = save_truth_record(
        cascade_truth_record,
        truth_dir=runtime_inputs["truths_path"],
        dataset_name=dataset_name,
        layer_name=runtime_inputs["layer_name"],
    )

    append_truth_index(
        cascade_truth_record,
        truth_index_path=runtime_inputs["truth_index_path"],
    )

    logger.info("Cascade truth hash: %s", cascade_truth_hash)
    logger.info("Cascade truth path: %s", cascade_truth_path)

    return {
        "dataframe": scored_all_dataframe,
        "cascade_truth_hash": cascade_truth_hash,
        "cascade_truth_path": str(cascade_truth_path),
    }


def _optionally_write_sql_output(
    *,
    dataframe,
    runtime_inputs: Dict[str, Any],
    cascade_truth_hash: str,
    cascade_parent_truth_hash: str,
    pipeline_mode: str,
    logger: logging.Logger,
) -> Optional[str]:
    if not runtime_inputs["write_sql_output"]:
        return None

    engine = get_engine_from_env()

    cascade_sql_dataframe = prepare_layer_dataframe(
        dataframe,
        truth_hash=cascade_truth_hash,
        parent_truth_hash=cascade_parent_truth_hash,
        pipeline_mode=pipeline_mode,
        process_run_id=runtime_inputs["process_run_id"],
        add_loaded_at_column=True,
    )

    cascade_table_name = write_layer_dataframe(
        engine=engine,
        dataframe=cascade_sql_dataframe,
        schema=runtime_inputs["gold_schema"],
        dataset_name=dataframe["meta__dataset"].dropna().astype(str).iloc[0],
        if_exists="replace",
        index=False,
    )

    logger.info(
        "Wrote Gold cascade SQL table: %s.%s",
        runtime_inputs["gold_schema"],
        cascade_table_name,
    )

    return cascade_table_name


def run_gold_cascade_modeling(
    *,
    config_root: Optional[Path] = None,
    dataset: str = "pump",
    mode: str = "train",
    profile: str = "default",
    project_root: Optional[Path] = None,
    variant: Optional[str] = None,
    write_sql_output: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run Gold cascade modeling stage.
    """
    paths = get_paths()

    config_root = config_root or paths.configs
    project_root = project_root or paths.root

    config = load_pipeline_config(
        config_root=config_root,
        stage="gold_cascade_modeling",
        dataset=dataset,
        mode=mode,
        profile=profile,
        project_root=project_root,
    ).data

    runtime_inputs = _build_default_runtime_inputs(config=config)
    runtime_inputs = _apply_runtime_overrides(
        runtime_inputs,
        variant=variant,
        write_sql_output=write_sql_output,
    )

    _ensure_stage_directories(runtime_inputs)
    logger = _initialize_cascade_logger(paths)

    truthed_config = build_truth_config_block(config)
    truthed_config["pipeline"] = runtime_inputs["pipeline_cfg"]

    set_wandb_dir_from_config(config)
    export_config_snapshot(
        config,
        output_path=runtime_inputs["gold_artifacts_path"] / f"{dataset}__gold_cascade__resolved_config.yaml",
    )

    gold_inputs = _load_gold_inputs(
        runtime_inputs=runtime_inputs,
        logger=logger,
    )

    wandb_run = _initialize_wandb_run(
        runtime_inputs=runtime_inputs,
        scaled_gold_path=gold_inputs["scaled_path"],
        logger=logger,
    )

    ledger = Ledger(stage=runtime_inputs["stage"], recipe_id="gold_cascade_modeling")
    ledger.add(
        kind="step",
        step="load_gold_inputs",
        message="Loaded Gold preprocessing outputs for cascade modeling",
        data={
            "variant": runtime_inputs["variant"],
            "scaled_path": str(gold_inputs["scaled_path"]),
            "fit_path": str(gold_inputs["fit_path"]),
            "train_path": str(gold_inputs["train_path"]),
            "test_path": str(gold_inputs["test_path"]),
        },
        logger=logger,
    )

    parent_context = _resolve_parent_truth_and_initialize_cascade_truth(
        dataframe=gold_inputs["all_dataframe"],
        runtime_inputs=runtime_inputs,
        logger=logger,
    )

    dataset_name = parent_context["dataset_name"]
    cascade_parent_truth_hash = parent_context["cascade_parent_truth_hash"]
    pipeline_mode = parent_context["pipeline_mode"]
    cascade_truth = parent_context["cascade_truth"]

    stage1_feature_columns = _resolve_stage1_feature_columns(gold_inputs["all_dataframe"])
    stage2_feature_columns = list(gold_inputs["stage2_features_payload"].get("stage2_feature_columns", []))
    if len(stage2_feature_columns) == 0:
        raise ValueError("Stage 2 feature list is empty.")

    cascade_results = run_cascade_pipeline(
        fit_dataframe=gold_inputs["fit_dataframe"],
        train_dataframe=gold_inputs["train_dataframe"],
        test_dataframe=gold_inputs["test_dataframe"],
        all_dataframe=gold_inputs["all_dataframe"],
        stage1_feature_columns=stage1_feature_columns,
        stage2_feature_columns=stage2_feature_columns,
        reference_profile=gold_inputs["reference_profile"],
        stage3_sensor_groups=gold_inputs["stage3_sensor_groups"],
        label_column=runtime_inputs["label_column"],
        stage1_model_params=runtime_inputs["stage1_model_params"],
        stage2_model_params=runtime_inputs["stage2_model_params"],
        variant=runtime_inputs["variant"],
    )

    cascade_results["summary"] = build_cascade_summary(
        variant=runtime_inputs["variant"],
        summary=cascade_results["summary"],
        stage1_feature_columns=stage1_feature_columns,
        stage2_feature_columns=stage2_feature_columns,
    )

    cascade_truth_payload = _build_and_save_cascade_truth(
        scored_all_dataframe=cascade_results["scored_all"],
        cascade_truth=cascade_truth,
        dataset_name=dataset_name,
        cascade_parent_truth_hash=cascade_parent_truth_hash,
        pipeline_mode=pipeline_mode,
        runtime_inputs=runtime_inputs,
        cascade_results=cascade_results,
        stage1_feature_columns=stage1_feature_columns,
        stage2_feature_columns=stage2_feature_columns,
        scaled_gold_path=gold_inputs["scaled_path"],
        logger=logger,
    )

    scored_all_with_truth = cascade_truth_payload["dataframe"]
    cascade_truth_hash = cascade_truth_payload["cascade_truth_hash"]
    cascade_truth_path = cascade_truth_payload["cascade_truth_path"]

    scored_outputs = {
        "scored_fit": cascade_results["scored_fit"].copy(),
        "scored_train": cascade_results["scored_train"].copy(),
        "scored_test": cascade_results["scored_test"].copy(),
        "scored_all": scored_all_with_truth.copy(),
    }
    for key in ["scored_fit", "scored_train", "scored_test"]:
        scored_outputs[key]["meta__truth_hash"] = cascade_truth_hash
        scored_outputs[key]["meta__parent_truth_hash"] = cascade_parent_truth_hash
        scored_outputs[key]["meta__pipeline_mode"] = pipeline_mode

    stage1_model_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["cascade_stage1_model_file_name"]
    stage2_model_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["cascade_stage2_model_file_name"]
    cascade_summary_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["cascade_summary_file_name"]
    scored_fit_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["cascade_scored_fit_file_name"]
    scored_train_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["cascade_scored_train_file_name"]
    scored_test_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["cascade_scored_test_file_name"]
    scored_all_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["cascade_scored_all_file_name"]

    _save_pickle_object(cascade_results["stage1_model"], stage1_model_path)
    _save_pickle_object(cascade_results["stage2_model"], stage2_model_path)
    save_json(cascade_results["summary"], cascade_summary_path)
    save_data(scored_outputs["scored_fit"], scored_fit_path.parent, scored_fit_path.name)
    save_data(scored_outputs["scored_train"], scored_train_path.parent, scored_train_path.name)
    save_data(scored_outputs["scored_test"], scored_test_path.parent, scored_test_path.name)
    save_data(scored_outputs["scored_all"], scored_all_path.parent, scored_all_path.name)

    saved_ledger_path = ledger.write_json(
        runtime_inputs["gold_artifacts_path"] / f"gold_cascade__{dataset_name}__{runtime_inputs['variant']}__ledger.json"
    )

    sql_table_name = _optionally_write_sql_output(
        dataframe=scored_outputs["scored_all"],
        runtime_inputs=runtime_inputs,
        cascade_truth_hash=cascade_truth_hash,
        cascade_parent_truth_hash=cascade_parent_truth_hash,
        pipeline_mode=pipeline_mode,
        logger=logger,
    )

    finalize_wandb_stage(
        run=wandb_run,
        stage=runtime_inputs["stage"],
        dataframe=scored_outputs["scored_all"],
        project_root=paths.root,
        logs_dir=paths.logs,
        dataset_dirs=[runtime_inputs["gold_train_data_path"]],
        dataset_artifact_name=f"{dataset_name}-{runtime_inputs['stage']}-{runtime_inputs['variant']}-dataset",
        logger=logger,
        notebook_path=None,
        aliases=("latest", runtime_inputs["variant"]),
        table_key=None,
        table_n=15,
        profile=False,
    )

    wandb_run.finish()

    summary = {
        "status": "success",
        "layer_name": runtime_inputs["layer_name"],
        "dataset_name": dataset_name,
        "variant": runtime_inputs["variant"],
        "row_count": int(scored_outputs["scored_all"].shape[0]),
        "column_count": int(scored_outputs["scored_all"].shape[1]),
        "cascade_stage1_model_path": str(stage1_model_path),
        "cascade_stage2_model_path": str(stage2_model_path),
        "cascade_summary_path": str(cascade_summary_path),
        "cascade_scored_fit_path": str(scored_fit_path),
        "cascade_scored_train_path": str(scored_train_path),
        "cascade_scored_test_path": str(scored_test_path),
        "cascade_scored_all_path": str(scored_all_path),
        "truth_hash": cascade_truth_hash,
        "truth_path": cascade_truth_path,
        "parent_truth_hash": cascade_parent_truth_hash,
        "process_run_id": runtime_inputs["process_run_id"],
        "cascade_metrics": cascade_results["summary"]["cascade_metrics"],
        "sql_table_name": sql_table_name,
        "ledger_path": str(saved_ledger_path),
    }

    logger.info("Gold cascade modeling stage completed successfully.")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gold cascade modeling stage.")
    parser.add_argument("--dataset", default="pump")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--variant", default=None, choices=["default", "tuned", "improved"])
    parser.add_argument("--write-sql-output", default=None, help="true/false")
    return parser


def _parse_optional_bool(value: Optional[str]) -> Optional[bool]:
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

    result = run_gold_cascade_modeling(
        config_root=Path(args.config_root) if args.config_root else None,
        dataset=args.dataset,
        mode=args.mode,
        profile=args.profile,
        project_root=Path(args.project_root) if args.project_root else None,
        variant=args.variant,
        write_sql_output=_parse_optional_bool(args.write_sql_output),
    )

    print(json.dumps(result, indent=2))



'''
#Example usage

#Default cascade:

python -m pipelines.gold.run_gold_cascade_modeling \
  --dataset pump \
  --mode train \
  --profile default \
  --variant default

#Tuned cascade:

python -m pipelines.gold.run_gold_cascade_modeling \
  --dataset pump \
  --mode train \
  --profile default \
  --variant tuned

#Improved cascade:

python -m pipelines.gold.run_gold_cascade_modeling \
  --dataset pump \
  --mode train \
  --profile default \
  --variant improved

#Enable SQL write:

python -m pipelines.gold.run_gold_cascade_modeling \
  --dataset pump \
  --mode train \
  --profile default \
  --variant improved \
  --write-sql-output true

'''