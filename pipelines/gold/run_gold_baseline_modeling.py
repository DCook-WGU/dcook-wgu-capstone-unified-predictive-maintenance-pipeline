"""
pipelines/gold/run_gold_baseline_modeling.py

Gold baseline modeling pipeline runner for the capstone project.
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
from utils.pipeline.gold_baseline_modeling import run_baseline_pipeline


def _build_default_runtime_inputs(
    *,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pull commonly used Gold baseline runtime inputs from config.
    """
    baseline_cfg = config["gold_baseline_modeling"]
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
        "baseline_cfg": baseline_cfg,
        "paths_cfg": paths_cfg,
        "filenames": filenames,
        "pipeline_cfg": pipeline_cfg,
        "stage": "gold_baseline",
        "layer_name": baseline_cfg["layer_name"],
        "gold_version": config["versions"]["gold"],
        "truth_version": config["versions"]["truth"],
        "pipeline_mode": pipeline_cfg["execution_mode"],
        "run_mode": config["runtime"]["mode"],
        "profile": config["runtime"]["profile"],
        "dataset_name_config": config["dataset"]["name"],
        "process_run_id": make_process_run_id(baseline_cfg["process_run_id_prefix"]),
        "wandb_project": config["wandb"]["project"],
        "wandb_entity": config["wandb"]["entity"],
        "wandb_run_name": f"{config['versions']['gold']}_baseline",
        "gold_train_data_path": Path(paths_cfg["data_gold_train_dir"]),
        "gold_artifacts_path": Path(paths_cfg["gold_artifacts_dir"]),
        "truths_path": Path(paths_cfg["truths_dir"]),
        "truth_index_path": Path(paths_cfg["truth_index_path"]),
        "logs_path": Path(paths_cfg["logs_root"]),
        "gold_preprocessed_scaled_file_name": filenames["gold_preprocessed_scaled_file_name"],
        "gold_fit_file_name": filenames["gold_fit_file_name"],
        "gold_train_file_name": filenames["gold_train_file_name"],
        "gold_test_file_name": filenames["gold_test_file_name"],
        "baseline_feature_file_name": filenames.get("gold_baseline_feature_file_name", filenames["gold_stage2_features_file_name"]),
        "baseline_model_file_name": filenames["gold_baseline_model_file_name"],
        "baseline_summary_file_name": filenames["gold_baseline_summary_file_name"],
        "baseline_scored_fit_file_name": filenames["gold_baseline_scored_fit_file_name"],
        "baseline_scored_train_file_name": filenames["gold_baseline_scored_train_file_name"],
        "baseline_scored_test_file_name": filenames["gold_baseline_scored_test_file_name"],
        "baseline_scored_all_file_name": filenames["gold_baseline_scored_all_file_name"],
        "threshold_percentile": float(baseline_cfg["threshold_percentile"]),
        "score_column_name": baseline_cfg.get("score_column_name", "baseline_anomaly_score"),
        "prediction_column_name": baseline_cfg.get("prediction_column_name", "baseline_predicted_anomaly"),
        "label_column": baseline_cfg.get("label_column", "anomaly_flag"),
        "model_params": {
            "n_estimators": int(baseline_cfg["n_estimators"]),
            "contamination": baseline_cfg["contamination"],
            "max_samples": baseline_cfg["max_samples"],
            "max_features": float(baseline_cfg["max_features"]),
            "bootstrap": bool(baseline_cfg["bootstrap"]),
            "random_state": int(baseline_cfg["random_state"]),
            "n_jobs": int(baseline_cfg["n_jobs"]),
        },
        "write_sql_output": bool(baseline_cfg.get("write_sql_output", False)),
        "gold_schema": baseline_cfg.get("gold_schema", "gold"),
    }

    return runtime_inputs


def _apply_runtime_overrides(
    runtime_inputs: Dict[str, Any],
    *,
    gold_preprocessed_scaled_file_name: Optional[str] = None,
    gold_fit_file_name: Optional[str] = None,
    gold_train_file_name: Optional[str] = None,
    gold_test_file_name: Optional[str] = None,
    baseline_model_file_name: Optional[str] = None,
    baseline_summary_file_name: Optional[str] = None,
    threshold_percentile: Optional[float] = None,
    n_estimators: Optional[int] = None,
    contamination: Optional[str] = None,
    max_samples: Optional[str] = None,
    max_features: Optional[float] = None,
    bootstrap: Optional[bool] = None,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
    write_sql_output: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Apply explicit runtime overrides on top of config-derived defaults.
    """
    updated_inputs = dict(runtime_inputs)

    string_overrides = {
        "gold_preprocessed_scaled_file_name": gold_preprocessed_scaled_file_name,
        "gold_fit_file_name": gold_fit_file_name,
        "gold_train_file_name": gold_train_file_name,
        "gold_test_file_name": gold_test_file_name,
        "baseline_model_file_name": baseline_model_file_name,
        "baseline_summary_file_name": baseline_summary_file_name,
    }

    for key, value in string_overrides.items():
        if value is not None and str(value).strip() != "":
            updated_inputs[key] = str(value).strip()

    if threshold_percentile is not None:
        updated_inputs["threshold_percentile"] = float(threshold_percentile)

    if n_estimators is not None:
        updated_inputs["model_params"]["n_estimators"] = int(n_estimators)
    if contamination is not None and str(contamination).strip() != "":
        contamination_text = str(contamination).strip()
        try:
            updated_inputs["model_params"]["contamination"] = float(contamination_text)
        except ValueError:
            updated_inputs["model_params"]["contamination"] = contamination_text
    if max_samples is not None and str(max_samples).strip() != "":
        max_samples_text = str(max_samples).strip()
        try:
            updated_inputs["model_params"]["max_samples"] = float(max_samples_text)
        except ValueError:
            updated_inputs["model_params"]["max_samples"] = max_samples_text
    if max_features is not None:
        updated_inputs["model_params"]["max_features"] = float(max_features)
    if bootstrap is not None:
        updated_inputs["model_params"]["bootstrap"] = bool(bootstrap)
    if random_state is not None:
        updated_inputs["model_params"]["random_state"] = int(random_state)
    if n_jobs is not None:
        updated_inputs["model_params"]["n_jobs"] = int(n_jobs)

    if write_sql_output is not None:
        updated_inputs["write_sql_output"] = bool(write_sql_output)

    return updated_inputs


def _ensure_stage_directories(runtime_inputs: Dict[str, Any]) -> None:
    """
    Ensure required baseline directories exist.
    """
    runtime_inputs["gold_train_data_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["gold_artifacts_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["truths_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["logs_path"].mkdir(parents=True, exist_ok=True)


def _initialize_baseline_logger(paths) -> logging.Logger:
    """
    Create and configure baseline logger.
    """
    baseline_log_path = paths.logs / "gold_modeling_baseline.log"
    baseline_log_path.parent.mkdir(parents=True, exist_ok=True)

    configure_logging(
        "capstone",
        baseline_log_path,
        level=logging.DEBUG,
        overwrite_handlers=True,
    )

    logger = logging.getLogger("capstone.gold_baseline")
    logger.info("Gold baseline stage starting")
    log_layer_paths(paths, current_layer="gold", logger=logger)

    return logger


def _initialize_wandb_run(
    *,
    runtime_inputs: Dict[str, Any],
    scaled_gold_path: Path,
    logger: logging.Logger,
):
    """
    Start W&B run for Gold baseline modeling.
    """
    wandb_run = wandb.init(
        project=runtime_inputs["wandb_project"],
        entity=runtime_inputs["wandb_entity"],
        name=runtime_inputs["wandb_run_name"],
        job_type="gold_baseline_modeling",
        config={
            "gold_version": runtime_inputs["gold_version"],
            "threshold_percentile": runtime_inputs["threshold_percentile"],
            "model_params": runtime_inputs["model_params"],
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
    """
    Load Gold preprocessing outputs needed for baseline modeling.
    """
    scaled_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["gold_preprocessed_scaled_file_name"]
    fit_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["gold_fit_file_name"]
    train_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["gold_train_file_name"]
    test_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["gold_test_file_name"]

    for required_path in [scaled_path, fit_path, train_path, test_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required Gold input not found: {required_path}")

    all_dataframe = load_data(scaled_path.parent, scaled_path.name)
    fit_dataframe = load_data(fit_path.parent, fit_path.name)
    train_dataframe = load_data(train_path.parent, train_path.name)
    test_dataframe = load_data(test_path.parent, test_path.name)

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
    }


def _resolve_parent_truth_and_initialize_baseline_truth(
    *,
    dataframe,
    runtime_inputs: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Resolve Gold preprocessing parent truth and initialize baseline truth object.
    """
    baseline_parent_truth_hash = extract_truth_hash(dataframe)

    if baseline_parent_truth_hash is None:
        raise ValueError("Baseline input dataframe does not contain a readable meta__truth_hash value.")

    dataset_name_series = dataframe["meta__dataset"].dropna().astype("string").str.strip()
    dataset_name_series = dataset_name_series[dataset_name_series != ""]
    if len(dataset_name_series) == 0:
        raise ValueError("Baseline input dataframe is missing usable meta__dataset values.")

    gold_dataset_name = str(dataset_name_series.iloc[0]).strip()

    parent_truth = load_parent_truth_record_from_dataframe(
        dataframe=dataframe,
        truth_dir=runtime_inputs["truths_path"],
        parent_layer_name="gold_preprocessing",
        dataset_name=gold_dataset_name,
        column_name="meta__truth_hash",
    )

    dataset_name = get_dataset_name_from_truth(parent_truth)
    baseline_parent_truth_hash = get_truth_hash(parent_truth)
    parent_pipeline_mode = get_pipeline_mode_from_truth(parent_truth)
    pipeline_mode = parent_pipeline_mode or runtime_inputs["pipeline_mode"]

    baseline_truth = initialize_layer_truth(
        truth_version=runtime_inputs["truth_version"],
        dataset_name=dataset_name,
        layer_name=runtime_inputs["layer_name"],
        process_run_id=runtime_inputs["process_run_id"],
        pipeline_mode=pipeline_mode,
        parent_truth_hash=baseline_parent_truth_hash,
    )

    baseline_truth = update_truth_section(
        baseline_truth,
        "config_snapshot",
        {
            "gold_version": runtime_inputs["gold_version"],
            "threshold_percentile": runtime_inputs["threshold_percentile"],
            "model_params": runtime_inputs["model_params"],
            "pipeline_mode": pipeline_mode,
        },
    )

    baseline_truth = update_truth_section(
        baseline_truth,
        "runtime_facts",
        {
            "parent_layer_name": "gold_preprocessing",
            "parent_truth_hash": baseline_parent_truth_hash,
            "dataset_name_from_parent_truth": dataset_name,
        },
    )

    logger.info("Resolved Gold preprocessing parent truth hash: %s", baseline_parent_truth_hash)
    logger.info("Resolved baseline dataset name from parent truth: %s", dataset_name)

    return {
        "dataset_name": dataset_name,
        "baseline_parent_truth_hash": baseline_parent_truth_hash,
        "pipeline_mode": pipeline_mode,
        "baseline_truth": baseline_truth,
    }


def _load_baseline_feature_columns(
    *,
    runtime_inputs: Dict[str, Any],
    all_dataframe,
    logger: logging.Logger,
) -> list[str]:
    """
    Resolve baseline feature columns.

    Current fallback strategy:
    - Use numeric non-meta columns excluding score/prediction/label helper columns.
    This can be tightened later to load a dedicated baseline feature artifact.
    """
    excluded = {
        "anomaly_flag",
        "baseline_anomaly_score",
        "baseline_predicted_anomaly",
    }

    feature_columns: list[str] = []
    for column_name in all_dataframe.columns:
        if column_name.startswith("meta__"):
            continue
        if column_name in excluded:
            continue
        if pd.api.types.is_numeric_dtype(all_dataframe[column_name]):
            feature_columns.append(column_name)

    if len(feature_columns) == 0:
        raise ValueError("No usable baseline feature columns resolved from Gold dataframe.")

    logger.info("Resolved %d baseline feature columns.", len(feature_columns))
    return feature_columns


def _save_pickle_object(obj, output_path: Path) -> str:
    """
    Save Python object with pickle.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file_handle:
        pickle.dump(obj, file_handle)
    return str(output_path)


def _build_and_save_baseline_truth(
    *,
    scored_all_dataframe,
    baseline_truth: Dict[str, Any],
    dataset_name: str,
    baseline_parent_truth_hash: str,
    pipeline_mode: str,
    runtime_inputs: Dict[str, Any],
    baseline_results: Dict[str, Any],
    feature_columns: list[str],
    scaled_gold_path: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build baseline truth record, stamp truth columns, and save truth artifact.
    """
    baseline_truth = update_truth_section(
        baseline_truth,
        "runtime_facts",
        {
            "scaled_gold_input_path": str(scaled_gold_path),
            "dataset_name_final": dataset_name,
            "baseline_feature_columns": feature_columns,
        },
    )

    baseline_truth = update_truth_section(
        baseline_truth,
        "artifact_paths",
        {
            "baseline_model_file_name": runtime_inputs["baseline_model_file_name"],
            "baseline_summary_file_name": runtime_inputs["baseline_summary_file_name"],
            "baseline_scored_fit_file_name": runtime_inputs["baseline_scored_fit_file_name"],
            "baseline_scored_train_file_name": runtime_inputs["baseline_scored_train_file_name"],
            "baseline_scored_test_file_name": runtime_inputs["baseline_scored_test_file_name"],
            "baseline_scored_all_file_name": runtime_inputs["baseline_scored_all_file_name"],
        },
    )

    baseline_truth = update_truth_section(
        baseline_truth,
        "baseline_modeling",
        {
            "summary": baseline_results["summary"],
        },
    )

    baseline_meta_columns = sorted(
        set(
            identify_meta_columns(scored_all_dataframe)
            + ["meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode"]
        )
    )
    baseline_feature_columns = identify_feature_columns(scored_all_dataframe)

    baseline_truth_record = build_truth_record(
        truth_base=baseline_truth,
        row_count=len(scored_all_dataframe),
        column_count=scored_all_dataframe.shape[1] + 3,
        meta_columns=baseline_meta_columns,
        feature_columns=baseline_feature_columns,
    )

    baseline_truth_hash = baseline_truth_record["truth_hash"]

    scored_all_dataframe = stamp_truth_columns(
        scored_all_dataframe,
        truth_hash=baseline_truth_hash,
        parent_truth_hash=baseline_parent_truth_hash,
        pipeline_mode=pipeline_mode,
    )

    baseline_truth_path = save_truth_record(
        baseline_truth_record,
        truth_dir=runtime_inputs["truths_path"],
        dataset_name=dataset_name,
        layer_name=runtime_inputs["layer_name"],
    )

    append_truth_index(
        baseline_truth_record,
        truth_index_path=runtime_inputs["truth_index_path"],
    )

    logger.info("Baseline truth hash: %s", baseline_truth_hash)
    logger.info("Baseline truth path: %s", baseline_truth_path)

    return {
        "dataframe": scored_all_dataframe,
        "baseline_truth_record": baseline_truth_record,
        "baseline_truth_hash": baseline_truth_hash,
        "baseline_truth_path": str(baseline_truth_path),
    }


def _optionally_write_sql_output(
    *,
    dataframe,
    runtime_inputs: Dict[str, Any],
    baseline_truth_hash: str,
    baseline_parent_truth_hash: str,
    pipeline_mode: str,
    logger: logging.Logger,
) -> Optional[str]:
    """
    Optional PostgreSQL persistence for baseline scored output.
    """
    if not runtime_inputs["write_sql_output"]:
        return None

    engine = get_engine_from_env()

    baseline_sql_dataframe = prepare_layer_dataframe(
        dataframe,
        truth_hash=baseline_truth_hash,
        parent_truth_hash=baseline_parent_truth_hash,
        pipeline_mode=pipeline_mode,
        process_run_id=runtime_inputs["process_run_id"],
        add_loaded_at_column=True,
    )

    baseline_table_name = write_layer_dataframe(
        engine=engine,
        dataframe=baseline_sql_dataframe,
        schema=runtime_inputs["gold_schema"],
        dataset_name=dataframe["meta__dataset"].dropna().astype(str).iloc[0],
        if_exists="replace",
        index=False,
    )

    logger.info(
        "Wrote Gold baseline SQL table: %s.%s",
        runtime_inputs["gold_schema"],
        baseline_table_name,
    )

    return baseline_table_name


def run_gold_baseline_modeling(
    *,
    config_root: Optional[Path] = None,
    dataset: str = "pump",
    mode: str = "train",
    profile: str = "default",
    project_root: Optional[Path] = None,
    gold_preprocessed_scaled_file_name: Optional[str] = None,
    gold_fit_file_name: Optional[str] = None,
    gold_train_file_name: Optional[str] = None,
    gold_test_file_name: Optional[str] = None,
    baseline_model_file_name: Optional[str] = None,
    baseline_summary_file_name: Optional[str] = None,
    threshold_percentile: Optional[float] = None,
    n_estimators: Optional[int] = None,
    contamination: Optional[str] = None,
    max_samples: Optional[str] = None,
    max_features: Optional[float] = None,
    bootstrap: Optional[bool] = None,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
    write_sql_output: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run Gold baseline modeling stage.
    """
    import pandas as pd

    paths = get_paths()

    config_root = config_root or paths.configs
    project_root = project_root or paths.root

    config = load_pipeline_config(
        config_root=config_root,
        stage="gold_baseline_modeling",
        dataset=dataset,
        mode=mode,
        profile=profile,
        project_root=project_root,
    ).data

    runtime_inputs = _build_default_runtime_inputs(config=config)
    runtime_inputs = _apply_runtime_overrides(
        runtime_inputs,
        gold_preprocessed_scaled_file_name=gold_preprocessed_scaled_file_name,
        gold_fit_file_name=gold_fit_file_name,
        gold_train_file_name=gold_train_file_name,
        gold_test_file_name=gold_test_file_name,
        baseline_model_file_name=baseline_model_file_name,
        baseline_summary_file_name=baseline_summary_file_name,
        threshold_percentile=threshold_percentile,
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state,
        n_jobs=n_jobs,
        write_sql_output=write_sql_output,
    )

    _ensure_stage_directories(runtime_inputs)

    logger = _initialize_baseline_logger(paths)

    truthed_config = build_truth_config_block(config)
    truthed_config["pipeline"] = runtime_inputs["pipeline_cfg"]

    set_wandb_dir_from_config(config)
    export_config_snapshot(
        config,
        output_path=runtime_inputs["gold_artifacts_path"] / f"{dataset}__gold_baseline__resolved_config.yaml",
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

    ledger = Ledger(stage=runtime_inputs["stage"], recipe_id="gold_baseline_modeling")
    ledger.add(
        kind="step",
        step="load_gold_inputs",
        message="Loaded Gold preprocessing outputs for baseline modeling",
        data={
            "scaled_path": str(gold_inputs["scaled_path"]),
            "fit_path": str(gold_inputs["fit_path"]),
            "train_path": str(gold_inputs["train_path"]),
            "test_path": str(gold_inputs["test_path"]),
        },
        logger=logger,
    )

    parent_context = _resolve_parent_truth_and_initialize_baseline_truth(
        dataframe=gold_inputs["all_dataframe"],
        runtime_inputs=runtime_inputs,
        logger=logger,
    )

    dataset_name = parent_context["dataset_name"]
    baseline_parent_truth_hash = parent_context["baseline_parent_truth_hash"]
    pipeline_mode = parent_context["pipeline_mode"]
    baseline_truth = parent_context["baseline_truth"]

    feature_columns = _load_baseline_feature_columns(
        runtime_inputs=runtime_inputs,
        all_dataframe=gold_inputs["all_dataframe"],
        logger=logger,
    )

    baseline_results = run_baseline_pipeline(
        fit_dataframe=gold_inputs["fit_dataframe"],
        train_dataframe=gold_inputs["train_dataframe"],
        test_dataframe=gold_inputs["test_dataframe"],
        all_dataframe=gold_inputs["all_dataframe"],
        feature_columns=feature_columns,
        threshold_percentile=runtime_inputs["threshold_percentile"],
        model_params=runtime_inputs["model_params"],
        score_column_name=runtime_inputs["score_column_name"],
        prediction_column_name=runtime_inputs["prediction_column_name"],
        label_column=runtime_inputs["label_column"],
    )

    baseline_truth_payload = _build_and_save_baseline_truth(
        scored_all_dataframe=baseline_results["scored_all"],
        baseline_truth=baseline_truth,
        dataset_name=dataset_name,
        baseline_parent_truth_hash=baseline_parent_truth_hash,
        pipeline_mode=pipeline_mode,
        runtime_inputs=runtime_inputs,
        baseline_results=baseline_results,
        feature_columns=feature_columns,
        scaled_gold_path=gold_inputs["scaled_path"],
        logger=logger,
    )

    scored_all_with_truth = baseline_truth_payload["dataframe"]
    baseline_truth_hash = baseline_truth_payload["baseline_truth_hash"]
    baseline_truth_path = baseline_truth_payload["baseline_truth_path"]

    # Stamp lineage into all scored outputs
    scored_outputs = {
        "scored_fit": baseline_results["scored_fit"].copy(),
        "scored_train": baseline_results["scored_train"].copy(),
        "scored_test": baseline_results["scored_test"].copy(),
        "scored_all": scored_all_with_truth.copy(),
    }
    for key in ["scored_fit", "scored_train", "scored_test"]:
        scored_outputs[key]["meta__truth_hash"] = baseline_truth_hash
        scored_outputs[key]["meta__parent_truth_hash"] = baseline_parent_truth_hash
        scored_outputs[key]["meta__pipeline_mode"] = pipeline_mode

    baseline_model_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["baseline_model_file_name"]
    baseline_summary_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["baseline_summary_file_name"]
    scored_fit_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["baseline_scored_fit_file_name"]
    scored_train_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["baseline_scored_train_file_name"]
    scored_test_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["baseline_scored_test_file_name"]
    scored_all_path = runtime_inputs["gold_train_data_path"] / runtime_inputs["baseline_scored_all_file_name"]

    _save_pickle_object(baseline_results["model"], baseline_model_path)
    save_json(baseline_results["summary"], baseline_summary_path)
    save_data(scored_outputs["scored_fit"], scored_fit_path.parent, scored_fit_path.name)
    save_data(scored_outputs["scored_train"], scored_train_path.parent, scored_train_path.name)
    save_data(scored_outputs["scored_test"], scored_test_path.parent, scored_test_path.name)
    save_data(scored_outputs["scored_all"], scored_all_path.parent, scored_all_path.name)

    saved_ledger_path = ledger.write_json(
        runtime_inputs["gold_artifacts_path"] / f"gold_baseline__{dataset_name}__ledger.json"
    )

    sql_table_name = _optionally_write_sql_output(
        dataframe=scored_outputs["scored_all"],
        runtime_inputs=runtime_inputs,
        baseline_truth_hash=baseline_truth_hash,
        baseline_parent_truth_hash=baseline_parent_truth_hash,
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
        "row_count": int(scored_outputs["scored_all"].shape[0]),
        "column_count": int(scored_outputs["scored_all"].shape[1]),
        "baseline_model_path": str(baseline_model_path),
        "baseline_summary_path": str(baseline_summary_path),
        "baseline_scored_fit_path": str(scored_fit_path),
        "baseline_scored_train_path": str(scored_train_path),
        "baseline_scored_test_path": str(scored_test_path),
        "baseline_scored_all_path": str(scored_all_path),
        "truth_hash": baseline_truth_hash,
        "truth_path": baseline_truth_path,
        "parent_truth_hash": baseline_parent_truth_hash,
        "process_run_id": runtime_inputs["process_run_id"],
        "threshold": float(baseline_results["threshold"]),
        "test_metrics": baseline_results["summary"]["test_metrics"],
        "sql_table_name": sql_table_name,
        "ledger_path": str(saved_ledger_path),
    }

    logger.info("Gold baseline modeling stage completed successfully.")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for Gold baseline stage execution.
    """
    parser = argparse.ArgumentParser(description="Run Gold baseline modeling stage.")

    parser.add_argument("--dataset", default="pump")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--project-root", default=None)

    parser.add_argument("--gold-preprocessed-scaled-file-name", default=None)
    parser.add_argument("--gold-fit-file-name", default=None)
    parser.add_argument("--gold-train-file-name", default=None)
    parser.add_argument("--gold-test-file-name", default=None)
    parser.add_argument("--baseline-model-file-name", default=None)
    parser.add_argument("--baseline-summary-file-name", default=None)

    parser.add_argument("--threshold-percentile", type=float, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--contamination", default=None)
    parser.add_argument("--max-samples", default=None)
    parser.add_argument("--max-features", type=float, default=None)
    parser.add_argument("--bootstrap", default=None, help="true/false")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--write-sql-output", default=None, help="true/false")

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

    result = run_gold_baseline_modeling(
        config_root=Path(args.config_root) if args.config_root else None,
        dataset=args.dataset,
        mode=args.mode,
        profile=args.profile,
        project_root=Path(args.project_root) if args.project_root else None,
        gold_preprocessed_scaled_file_name=args.gold_preprocessed_scaled_file_name,
        gold_fit_file_name=args.gold_fit_file_name,
        gold_train_file_name=args.gold_train_file_name,
        gold_test_file_name=args.gold_test_file_name,
        baseline_model_file_name=args.baseline_model_file_name,
        baseline_summary_file_name=args.baseline_summary_file_name,
        threshold_percentile=args.threshold_percentile,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        max_samples=args.max_samples,
        max_features=args.max_features,
        bootstrap=_parse_optional_bool(args.bootstrap),
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        write_sql_output=_parse_optional_bool(args.write_sql_output),
    )

    print(json.dumps(result, indent=2))


'''
# Sample Usages:


# Default

python -m pipelines.gold.run_gold_baseline_modeling \
  --dataset pump \
  --mode train \
  --profile default


# Override Threshold and Model Parameters

python -m pipelines.gold.run_gold_baseline_modeling \
  --dataset pump \
  --mode train \
  --profile default \
  --threshold-percentile 95 \
  --n-estimators 300 \
  --contamination auto \
  --max-samples auto \
  --max-features 1.0 \
  --bootstrap false \
  --random-state 42 \
  --n-jobs -1

# Enable SQL Write Output
    
python -m pipelines.gold.run_gold_baseline_modeling \
  --dataset pump \
  --mode train \
  --profile default \
  --write-sql-output true
    
    
'''