"""
pipelines/silver/run_silver_preeda.py

Silver Pre-EDA pipeline runner for the capstone project.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import re
import wandb

from utils.paths import get_paths
from utils.file_io import load_data, save_data, save_json, load_json
from utils.logging_setup import configure_logging, log_layer_paths
from utils.wandb_utils import finalize_wandb_stage
from utils.truths import (
    make_process_run_id,
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
from utils.pipeline.silver_preeda import (
    prepare_silver_preeda_dataframe,
    build_silver_feature_registry,
    reorder_silver_columns,
    compute_quick_quality_checks,
)


def _build_default_runtime_inputs(
    *,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pull commonly used Silver Pre-EDA runtime inputs from config.
    """
    silver_cfg = config["silver_preeda"]
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
        "silver_cfg": silver_cfg,
        "paths_cfg": paths_cfg,
        "filenames": filenames,
        "pipeline_cfg": pipeline_cfg,
        "stage": "silver",
        "layer_name": silver_cfg["layer_name"],
        "silver_version": config["versions"]["silver"],
        "truth_version": config["versions"]["truth"],
        "cleaning_recipe_id": silver_cfg["cleaning_recipe_id"],
        "pipeline_mode": pipeline_cfg["execution_mode"],
        "run_mode": config["runtime"]["mode"],
        "profile": config["runtime"]["profile"],
        "dataset_name_config": config["dataset"]["name"],
        "process_run_id": make_process_run_id(silver_cfg["process_run_id_prefix"]),
        "wandb_project": config["wandb"]["project"],
        "wandb_entity": config["wandb"]["entity"],
        "wandb_run_name": f"{config['versions']['silver']}",
        "bronze_train_data_path": Path(paths_cfg["data_bronze_train_dir"]),
        "silver_train_data_path": Path(paths_cfg["data_silver_train_dir"]),
        "silver_artifacts_path": Path(paths_cfg["silver_artifacts_dir"]),
        "truths_path": Path(paths_cfg["truths_dir"]),
        "truth_index_path": Path(paths_cfg["truth_index_path"]),
        "logs_path": Path(paths_cfg["logs_root"]),
        "bronze_train_data_file_name": filenames["bronze_train_file_name"],
        "silver_train_data_file_name": filenames["silver_train_file_name"],
        "feature_registry_file_name": filenames["silver_feature_registry_file_name"],
        "asset_id_default_fallback": silver_cfg["asset_id_default_fallback"],
        "run_id_default_fallback": silver_cfg["run_id_default_fallback"],
        "raw_prefix": silver_cfg["raw_prefix"],
        "canonical_output_columns": list(silver_cfg["canonical_output_columns"]),
        "canonical_non_meta_order": list(silver_cfg["canonical_non_meta_order"]),
        "meta_required_columns": list(silver_cfg["meta_required_columns"]),
        "canonical_exclude_columns": list(silver_cfg["canonical_exclude_columns"]),
        "label_exclude_columns": list(silver_cfg["label_exclude_columns"]),
        "label_columns_order": list(silver_cfg["label_columns_order"]),
        "time_column_candidates": list(silver_cfg["time_column_candidates"]),
        "step_column_candidates": list(silver_cfg["step_column_candidates"]),
        "tie_breaker_candidates": list(silver_cfg["tie_breaker_candidates"]),
        "status_column_candidates": list(silver_cfg["status_column_candidates"]),
        "label_column_candidates": list(silver_cfg["label_column_candidates"]),
        "normal_status_values": list(config["dataset"]["normal_status_values"]),
        "quarantine_missing_pct": float(silver_cfg["quarantine_missing_pct"]),
        "min_time_parse_success_percent": float(silver_cfg["min_time_parse_success_percent"]),
        "min_step_parse_success_percent": float(silver_cfg["min_step_parse_success_percent"]),
        "default_exclude_prefixes": list(silver_cfg["default_exclude_prefixes"]),
        "unnamed_column_regex": re.compile(
            silver_cfg["unnamed_column_regex"],
            flags=re.IGNORECASE,
        ),
        "junk_column_candidates": list(silver_cfg["junk_column_candidates"]),
        "write_sql_output": bool(silver_cfg.get("write_sql_output", False)),
        "silver_schema": silver_cfg.get("silver_schema", "silver"),
    }

    return runtime_inputs


def _apply_runtime_overrides(
    runtime_inputs: Dict[str, Any],
    *,
    bronze_train_data_file_name: Optional[str] = None,
    silver_train_data_file_name: Optional[str] = None,
    feature_registry_file_name: Optional[str] = None,
    write_sql_output: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Apply explicit runtime overrides on top of config-derived defaults.
    """
    updated_inputs = dict(runtime_inputs)

    if bronze_train_data_file_name is not None and str(bronze_train_data_file_name).strip() != "":
        updated_inputs["bronze_train_data_file_name"] = str(bronze_train_data_file_name).strip()

    if silver_train_data_file_name is not None and str(silver_train_data_file_name).strip() != "":
        updated_inputs["silver_train_data_file_name"] = str(silver_train_data_file_name).strip()

    if feature_registry_file_name is not None and str(feature_registry_file_name).strip() != "":
        updated_inputs["feature_registry_file_name"] = str(feature_registry_file_name).strip()

    if write_sql_output is not None:
        updated_inputs["write_sql_output"] = bool(write_sql_output)

    return updated_inputs


def _ensure_stage_directories(runtime_inputs: Dict[str, Any]) -> None:
    """
    Ensure required Silver directories exist.
    """
    runtime_inputs["silver_train_data_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["silver_artifacts_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["truths_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["logs_path"].mkdir(parents=True, exist_ok=True)


def _initialize_silver_logger(paths) -> logging.Logger:
    """
    Create and configure the Silver logger.
    """
    silver_log_path = paths.logs / "silver.log"
    silver_log_path.parent.mkdir(parents=True, exist_ok=True)

    configure_logging(
        "capstone",
        silver_log_path,
        level=logging.DEBUG,
        overwrite_handlers=True,
    )

    logger = logging.getLogger("capstone.silver")
    logger.info("Silver stage starting")
    log_layer_paths(paths, current_layer="silver", logger=logger)

    return logger


def _initialize_wandb_run(
    *,
    runtime_inputs: Dict[str, Any],
    bronze_data_path: Path,
    logger: logging.Logger,
):
    """
    Start W&B run for Silver Pre-EDA.
    """
    wandb_run = wandb.init(
        project=runtime_inputs["wandb_project"],
        entity=runtime_inputs["wandb_entity"],
        name=runtime_inputs["wandb_run_name"],
        job_type="silver",
        config={
            "silver_version": runtime_inputs["silver_version"],
            "cleaning_recipe_id": runtime_inputs["cleaning_recipe_id"],
            "quarantine_missing_pct": runtime_inputs["quarantine_missing_pct"],
            "min_time_parse_success_percent": runtime_inputs["min_time_parse_success_percent"],
            "min_step_parse_success_percent": runtime_inputs["min_step_parse_success_percent"],
            "bronze_path": str(bronze_data_path),
            "silver_out_dir": str(runtime_inputs["silver_train_data_path"]),
        },
    )

    logger.info("W&B initialized: %s", wandb.run.name)
    return wandb_run


def _load_bronze_dataframe(
    *,
    runtime_inputs: Dict[str, Any],
    logger: logging.Logger,
):
    """
    Load the Bronze parquet input for Silver Pre-EDA.
    """
    preferred_bronze = runtime_inputs["bronze_train_data_path"] / runtime_inputs["bronze_train_data_file_name"]

    if preferred_bronze.exists():
        bronze_data_path = preferred_bronze
    else:
        parquet_files = sorted(runtime_inputs["bronze_train_data_path"].glob("*.parquet"))
        if len(parquet_files) == 0:
            raise FileNotFoundError(
                f"No parquet files found in {runtime_inputs['bronze_train_data_path']}"
            )
        if len(parquet_files) > 1:
            logger.warning("Multiple parquet files found; using first: %s", parquet_files[0])
        bronze_data_path = parquet_files[0]

    if not bronze_data_path.exists():
        raise FileNotFoundError(f"Bronze parquet not found: {bronze_data_path}")

    dataframe = load_data(bronze_data_path.parent, bronze_data_path.name)
    logger.info("Loaded Bronze: %s | shape=%s", bronze_data_path, dataframe.shape)

    return dataframe, bronze_data_path


def _resolve_parent_truth_and_initialize_silver_truth(
    *,
    dataframe,
    runtime_inputs: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Resolve Bronze parent truth and initialize Silver truth object.
    """
    silver_parent_truth_hash = extract_truth_hash(dataframe)

    if silver_parent_truth_hash is None:
        raise ValueError("Silver input dataframe does not contain a readable meta__truth_hash value.")

    bronze_dataset_name_series = (
        dataframe["meta__dataset"]
        .dropna()
        .astype("string")
        .str.strip()
    )
    bronze_dataset_name_series = bronze_dataset_name_series[bronze_dataset_name_series != ""]

    if len(bronze_dataset_name_series) == 0:
        raise ValueError("Silver input dataframe is missing usable meta__dataset values.")

    bronze_dataset_name = str(bronze_dataset_name_series.iloc[0]).strip()

    parent_truth = load_parent_truth_record_from_dataframe(
        dataframe=dataframe,
        truth_dir=runtime_inputs["truths_path"],
        parent_layer_name="bronze",
        dataset_name=bronze_dataset_name,
        column_name="meta__truth_hash",
    )

    dataset_name = get_dataset_name_from_truth(parent_truth)
    silver_parent_truth_hash = get_truth_hash(parent_truth)

    parent_pipeline_mode = get_pipeline_mode_from_truth(parent_truth)
    pipeline_mode = parent_pipeline_mode or runtime_inputs["pipeline_mode"]

    if "meta__pipeline_mode" not in dataframe.columns:
        dataframe["meta__pipeline_mode"] = pipeline_mode
    else:
        dataframe["meta__pipeline_mode"] = dataframe["meta__pipeline_mode"].fillna(pipeline_mode)

    silver_truth = initialize_layer_truth(
        truth_version=runtime_inputs["truth_version"],
        dataset_name=dataset_name,
        layer_name=runtime_inputs["layer_name"],
        process_run_id=runtime_inputs["process_run_id"],
        pipeline_mode=pipeline_mode,
        parent_truth_hash=silver_parent_truth_hash,
    )

    silver_truth = update_truth_section(
        silver_truth,
        "config_snapshot",
        {
            "silver_version": runtime_inputs["silver_version"],
            "cleaning_recipe_id": runtime_inputs["cleaning_recipe_id"],
            "dataset_name_config": runtime_inputs["dataset_name_config"],
            "dataset_name_parent_truth": dataset_name,
            "run_id_default_fallback": runtime_inputs["run_id_default_fallback"],
            "quarantine_missing_pct": runtime_inputs["quarantine_missing_pct"],
            "min_time_parse_success_percent": runtime_inputs["min_time_parse_success_percent"],
            "min_step_parse_success_percent": runtime_inputs["min_step_parse_success_percent"],
            "pipeline_mode": pipeline_mode,
        },
    )

    silver_truth = update_truth_section(
        silver_truth,
        "runtime_facts",
        {
            "parent_layer_name": "bronze",
            "parent_truth_hash": silver_parent_truth_hash,
            "dataset_name_from_parent_truth": dataset_name,
        },
    )

    logger.info("Resolved Bronze parent truth hash: %s", silver_parent_truth_hash)
    logger.info("Resolved Silver dataset name from Bronze truth: %s", dataset_name)

    return {
        "dataframe": dataframe,
        "parent_truth": parent_truth,
        "dataset_name": dataset_name,
        "silver_parent_truth_hash": silver_parent_truth_hash,
        "pipeline_mode": pipeline_mode,
        "silver_truth": silver_truth,
    }


def _build_and_save_silver_truth(
    *,
    dataframe,
    silver_truth: Dict[str, Any],
    dataset_name: str,
    silver_parent_truth_hash: str,
    pipeline_mode: str,
    runtime_inputs: Dict[str, Any],
    feature_registry: Dict[str, Any],
    preparation_info: Dict[str, Any],
    quality_info: Dict[str, Any],
    bronze_data_path: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build Silver truth record, stamp truth columns, and save truth artifact.
    """
    silver_truth = update_truth_section(
        silver_truth,
        "runtime_facts",
        {
            "bronze_input_path": str(bronze_data_path),
            "dataset_name_final": dataset_name,
            "label_source_column": preparation_info["label_source_column"],
            "label_source_type": preparation_info["label_source_type"],
        },
    )

    silver_truth = update_truth_section(
        silver_truth,
        "artifact_paths",
        {
            "silver_output_dir": str(runtime_inputs["silver_train_data_path"]),
            "silver_output_file_name": runtime_inputs["silver_train_data_file_name"],
            "feature_registry_path": str(
                runtime_inputs["silver_artifacts_path"] / runtime_inputs["feature_registry_file_name"]
            ),
        },
    )

    silver_truth = update_truth_section(
        silver_truth,
        "preeda",
        {
            "preparation_info": preparation_info,
            "quality_info": quality_info,
            "feature_set_id": feature_registry["feature_set_id"],
            "feature_count": feature_registry["feature_count"],
            "one_hot_encoding_columns": feature_registry["one_hot_encoding_columns"],
            "dropped_features": feature_registry["dropped_features"],
        },
    )

    silver_meta_columns = sorted(
        set(
            identify_meta_columns(dataframe)
            + ["meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode"]
        )
    )
    silver_feature_columns = identify_feature_columns(dataframe)

    silver_truth_record = build_truth_record(
        truth_base=silver_truth,
        row_count=len(dataframe),
        column_count=dataframe.shape[1] + 3,
        meta_columns=silver_meta_columns,
        feature_columns=silver_feature_columns,
    )

    silver_truth_hash = silver_truth_record["truth_hash"]

    dataframe = stamp_truth_columns(
        dataframe,
        truth_hash=silver_truth_hash,
        parent_truth_hash=silver_parent_truth_hash,
        pipeline_mode=pipeline_mode,
    )

    silver_truth_path = save_truth_record(
        silver_truth_record,
        truth_dir=runtime_inputs["truths_path"],
        dataset_name=dataset_name,
        layer_name=runtime_inputs["layer_name"],
    )

    append_truth_index(
        silver_truth_record,
        truth_index_path=runtime_inputs["truth_index_path"],
    )

    logger.info("Silver truth hash: %s", silver_truth_hash)
    logger.info("Silver truth path: %s", silver_truth_path)

    return {
        "dataframe": dataframe,
        "silver_truth_record": silver_truth_record,
        "silver_truth_hash": silver_truth_hash,
        "silver_truth_path": str(silver_truth_path),
    }


def _run_lineage_sanity_checks(
    *,
    dataframe,
    silver_truth_hash: str,
    silver_parent_truth_hash: str,
    silver_truth_path: str,
):
    """
    Final lineage sanity checks before stage completion.
    """
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

    if silver_dataframe_truth_hash != silver_truth_hash:
        raise ValueError(
            "Silver dataframe truth hash does not match silver truth record:\n"
            f"dataframe={silver_dataframe_truth_hash}\n"
            f"record={silver_truth_hash}"
        )

    silver_parent_values = dataframe["meta__parent_truth_hash"].dropna().astype(str).unique().tolist()
    if not silver_parent_values:
        raise ValueError("Silver dataframe is missing populated meta__parent_truth_hash values.")

    if len(silver_parent_values) != 1:
        raise ValueError(
            "Silver dataframe has multiple parent truth hashes:\n"
            f"{silver_parent_values}"
        )

    if silver_parent_values[0] != silver_parent_truth_hash:
        raise ValueError(
            "Silver dataframe parent truth hash does not match Silver parent truth hash:\n"
            f"dataframe_parent={silver_parent_values[0]}\n"
            f"silver_parent_truth={silver_parent_truth_hash}"
        )

    if not Path(silver_truth_path).exists():
        raise FileNotFoundError(f"Silver truth file was not created: {silver_truth_path}")

    loaded_silver_truth = load_json(silver_truth_path)

    if loaded_silver_truth.get("truth_hash") != silver_truth_hash:
        raise ValueError(
            "Saved Silver truth file hash does not match truth record:\n"
            f"file={loaded_silver_truth.get('truth_hash')}\n"
            f"record={silver_truth_hash}"
        )

    if loaded_silver_truth.get("parent_truth_hash") != silver_parent_truth_hash:
        raise ValueError(
            "Saved Silver truth file parent hash does not match expected parent:\n"
            f"truth={loaded_silver_truth.get('parent_truth_hash')}\n"
            f"silver_parent={silver_parent_truth_hash}"
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


def _optionally_write_sql_output(
    *,
    dataframe,
    runtime_inputs: Dict[str, Any],
    silver_truth_hash: str,
    silver_parent_truth_hash: str,
    pipeline_mode: str,
    logger: logging.Logger,
) -> Optional[str]:
    """
    Optional PostgreSQL persistence for Silver Pre-EDA output.
    """
    if not runtime_inputs["write_sql_output"]:
        return None

    engine = get_engine_from_env()

    silver_sql_dataframe = prepare_layer_dataframe(
        dataframe,
        truth_hash=silver_truth_hash,
        parent_truth_hash=silver_parent_truth_hash,
        pipeline_mode=pipeline_mode,
        process_run_id=runtime_inputs["process_run_id"],
        add_loaded_at_column=True,
    )

    silver_table_name = write_layer_dataframe(
        engine=engine,
        dataframe=silver_sql_dataframe,
        schema=runtime_inputs["silver_schema"],
        dataset_name=dataframe["meta__dataset"].dropna().astype(str).iloc[0],
        if_exists="replace",
        index=False,
    )

    logger.info(
        "Wrote Silver Pre-EDA SQL table: %s.%s",
        runtime_inputs["silver_schema"],
        silver_table_name,
    )

    return silver_table_name


def run_silver_preeda(
    *,
    config_root: Optional[Path] = None,
    dataset: str = "pump",
    mode: str = "train",
    profile: str = "default",
    project_root: Optional[Path] = None,
    bronze_train_data_file_name: Optional[str] = None,
    silver_train_data_file_name: Optional[str] = None,
    feature_registry_file_name: Optional[str] = None,
    write_sql_output: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run the Silver Pre-EDA stage.
    """
    paths = get_paths()

    config_root = config_root or paths.configs
    project_root = project_root or paths.root

    config = load_pipeline_config(
        config_root=config_root,
        stage="silver_preeda",
        dataset=dataset,
        mode=mode,
        profile=profile,
        project_root=project_root,
    ).data

    runtime_inputs = _build_default_runtime_inputs(config=config)
    runtime_inputs = _apply_runtime_overrides(
        runtime_inputs,
        bronze_train_data_file_name=bronze_train_data_file_name,
        silver_train_data_file_name=silver_train_data_file_name,
        feature_registry_file_name=feature_registry_file_name,
        write_sql_output=write_sql_output,
    )

    _ensure_stage_directories(runtime_inputs)

    logger = _initialize_silver_logger(paths)

    truthed_config = build_truth_config_block(config)
    truthed_config["pipeline"] = runtime_inputs["pipeline_cfg"]

    set_wandb_dir_from_config(config)
    export_config_snapshot(
        config,
        output_path=runtime_inputs["silver_artifacts_path"] / f"{dataset}__silver_preeda__resolved_config.yaml",
    )

    dataframe, bronze_data_path = _load_bronze_dataframe(
        runtime_inputs=runtime_inputs,
        logger=logger,
    )

    wandb_run = _initialize_wandb_run(
        runtime_inputs=runtime_inputs,
        bronze_data_path=bronze_data_path,
        logger=logger,
    )

    wandb_run.log({
        "bronze_rows": int(dataframe.shape[0]),
        "bronze_cols": int(dataframe.shape[1]),
    })

    ledger = Ledger(stage=runtime_inputs["stage"], recipe_id=runtime_inputs["cleaning_recipe_id"])
    ledger.add(
        kind="step",
        step="init",
        message="Initialized ledger",
        logger=logger,
    )
    ledger.add(
        kind="step",
        step="load_bronze",
        message="Loaded Bronze Parquet",
        why="Silver must be derived from reproducible Bronze artifact",
        consequence="All Silver outputs trace back to this file",
        data={"bronze_path": str(bronze_data_path), "shape": list(dataframe.shape), "cols": len(dataframe.columns)},
        logger=logger,
    )

    parent_context = _resolve_parent_truth_and_initialize_silver_truth(
        dataframe=dataframe,
        runtime_inputs=runtime_inputs,
        logger=logger,
    )

    dataframe = parent_context["dataframe"]
    dataset_name = parent_context["dataset_name"]
    silver_parent_truth_hash = parent_context["silver_parent_truth_hash"]
    pipeline_mode = parent_context["pipeline_mode"]
    silver_truth = parent_context["silver_truth"]

    dataframe, preparation_info = prepare_silver_preeda_dataframe(
        dataframe,
        dataset_name_config=runtime_inputs["dataset_name_config"],
        dataset_name_parent_truth=dataset_name,
        junk_column_candidates=runtime_inputs["junk_column_candidates"],
        unnamed_column_regex=runtime_inputs["unnamed_column_regex"],
        status_column_candidates=runtime_inputs["status_column_candidates"],
        label_column_candidates=runtime_inputs["label_column_candidates"],
        canonical_output_columns=runtime_inputs["canonical_output_columns"],
        time_column_candidates=runtime_inputs["time_column_candidates"],
        step_column_candidates=runtime_inputs["step_column_candidates"],
        tie_breaker_candidates=runtime_inputs["tie_breaker_candidates"],
        normal_status_values=runtime_inputs["normal_status_values"],
        asset_id_default_fallback=runtime_inputs["asset_id_default_fallback"],
        run_id_default_fallback=runtime_inputs["run_id_default_fallback"],
        raw_prefix=runtime_inputs["raw_prefix"],
        label_exclude_columns=runtime_inputs["label_exclude_columns"],
        min_time_parse_success_percent=runtime_inputs["min_time_parse_success_percent"],
        min_step_parse_success_percent=runtime_inputs["min_step_parse_success_percent"],
    )

    exclude_columns_combined = (
        list(runtime_inputs["meta_required_columns"])
        + list(runtime_inputs["canonical_output_columns"])
        + list(runtime_inputs["label_columns_order"])
    )

    dataframe, feature_registry, artifact_info = build_silver_feature_registry(
        dataframe,
        dataset_name=dataset_name,
        exclude_prefixes=runtime_inputs["default_exclude_prefixes"],
        exclude_columns=exclude_columns_combined,
        label_columns_order=runtime_inputs["label_columns_order"],
        canonical_exclude_columns=runtime_inputs["canonical_exclude_columns"],
        quarantine_missing_pct=runtime_inputs["quarantine_missing_pct"],
        pipeline_mode=pipeline_mode,
        process_run_id=runtime_inputs["process_run_id"],
        label_source_column=preparation_info["label_source_column"],
        label_source_type=preparation_info["label_source_type"],
    )

    dataframe = reorder_silver_columns(
        dataframe,
        canonical_non_meta_order=runtime_inputs["canonical_non_meta_order"],
        label_columns_order=runtime_inputs["label_columns_order"],
    )

    quality_info = compute_quick_quality_checks(
        dataframe,
        anomaly_flag_column="anomaly_flag",
        feature_columns=feature_registry["feature_columns"],
    )

    if not dataframe.columns.is_unique:
        raise ValueError("Silver dataframe columns are not unique after preprocessing.")

    silver_truth_payload = _build_and_save_silver_truth(
        dataframe=dataframe,
        silver_truth=silver_truth,
        dataset_name=dataset_name,
        silver_parent_truth_hash=silver_parent_truth_hash,
        pipeline_mode=pipeline_mode,
        runtime_inputs=runtime_inputs,
        feature_registry=feature_registry,
        preparation_info=preparation_info,
        quality_info=quality_info,
        bronze_data_path=bronze_data_path,
        logger=logger,
    )

    dataframe = silver_truth_payload["dataframe"]
    silver_truth_hash = silver_truth_payload["silver_truth_hash"]
    silver_truth_path = silver_truth_payload["silver_truth_path"]

    save_data(
        dataframe,
        runtime_inputs["silver_train_data_path"],
        runtime_inputs["silver_train_data_file_name"],
    )

    feature_registry_path = runtime_inputs["silver_artifacts_path"] / runtime_inputs["feature_registry_file_name"]
    save_json(feature_registry, feature_registry_path)

    dropped_features_path = runtime_inputs["silver_artifacts_path"] / f"{dataset_name}__silver__dropped_features.json"
    save_json(
        {
            "dropped_features": feature_registry["dropped_features"],
            "missing_audit": artifact_info["missing_audit"],
            "global_missingness": artifact_info["global_missingness"],
        },
        dropped_features_path,
    )

    saved_ledger_path = ledger.write_json(
        runtime_inputs["silver_artifacts_path"] / f"silver__{dataset_name}__ledger.json"
    )

    _run_lineage_sanity_checks(
        dataframe=dataframe,
        silver_truth_hash=silver_truth_hash,
        silver_parent_truth_hash=silver_parent_truth_hash,
        silver_truth_path=silver_truth_path,
    )

    sql_table_name = _optionally_write_sql_output(
        dataframe=dataframe,
        runtime_inputs=runtime_inputs,
        silver_truth_hash=silver_truth_hash,
        silver_parent_truth_hash=silver_parent_truth_hash,
        pipeline_mode=pipeline_mode,
        logger=logger,
    )

    finalize_wandb_stage(
        run=wandb_run,
        stage=runtime_inputs["stage"],
        dataframe=dataframe,
        project_root=paths.root,
        logs_dir=paths.logs,
        dataset_dirs=[runtime_inputs["silver_train_data_path"]],
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
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "silver_output_path": str(
            runtime_inputs["silver_train_data_path"] / runtime_inputs["silver_train_data_file_name"]
        ),
        "feature_registry_path": str(feature_registry_path),
        "dropped_features_path": str(dropped_features_path),
        "truth_hash": silver_truth_hash,
        "truth_path": silver_truth_path,
        "parent_truth_hash": silver_parent_truth_hash,
        "process_run_id": runtime_inputs["process_run_id"],
        "feature_count": int(feature_registry["feature_count"]),
        "feature_set_id": feature_registry["feature_set_id"],
        "one_hot_encoding_columns": feature_registry["one_hot_encoding_columns"],
        "dropped_features": feature_registry["dropped_features"],
        "ledger_path": str(saved_ledger_path),
        "sql_table_name": sql_table_name,
    }

    logger.info("Silver Pre-EDA stage completed successfully.")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for Silver Pre-EDA stage execution.
    """
    parser = argparse.ArgumentParser(description="Run Silver Pre-EDA stage.")

    parser.add_argument("--dataset", default="pump")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--project-root", default=None)

    parser.add_argument("--bronze-train-data-file-name", default=None)
    parser.add_argument("--silver-train-data-file-name", default=None)
    parser.add_argument("--feature-registry-file-name", default=None)

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

    result = run_silver_preeda(
        config_root=Path(args.config_root) if args.config_root else None,
        dataset=args.dataset,
        mode=args.mode,
        profile=args.profile,
        project_root=Path(args.project_root) if args.project_root else None,
        bronze_train_data_file_name=args.bronze_train_data_file_name,
        silver_train_data_file_name=args.silver_train_data_file_name,
        feature_registry_file_name=args.feature_registry_file_name,
        write_sql_output=_parse_optional_bool(args.write_sql_output),
    )

    print(json.dumps(result, indent=2))


'''
# Sample Usage:

# Run Default Bronze Artifact:

python -m pipelines.silver.run_silver_preeda \
  --dataset pump \
  --mode train \
  --profile default

# Override Bronze input and Silver Output file names:

python -m pipelines.silver.run_silver_preeda \
  --dataset pump \
  --mode train \
  --profile default \
  --bronze-train-data-file-name pump__bronze__train.parquet \
  --silver-train-data-file-name pump__silver__train.parquet \
  --feature-registry-file-name pump__silver__feature_registry.json


# Enable SQL Write:

python -m pipelines.silver.run_silver_preeda \
  --dataset pump \
  --mode train \
  --profile default \
  --write-sql-output true

'''