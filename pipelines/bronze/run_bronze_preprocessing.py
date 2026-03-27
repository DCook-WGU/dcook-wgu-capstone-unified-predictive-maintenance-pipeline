"""
pipelines/bronze/run_bronze_preprocessing.py

Bronze pipeline runner for the capstone project.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from utils.paths import get_paths
from utils.file_io import ingest_data, save_data
from utils.logging_setup import configure_logging, log_layer_paths
from utils.truths import (
    make_process_run_id,
    build_file_fingerprint,
    identify_meta_columns,
    identify_feature_columns,
    initialize_layer_truth,
    update_truth_section,
    build_truth_record,
    save_truth_record,
    append_truth_index,
    stamp_truth_columns,
)
from utils.wandb_utils import finalize_wandb_stage
from utils.pipeline_config_loader import (
    load_pipeline_config,
    build_truth_config_block,
    set_wandb_dir_from_config,
    export_config_snapshot,
)
from utils.postgres_util import get_engine_from_env
from utils.layer_postgres_writer import read_layer_dataframe
from utils.pipeline.bronze_preprocessing import (
    prepare_bronze_dataframe,
    collect_meta_columns,
)


def _build_default_runtime_inputs(
    *,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pull commonly used Bronze runtime inputs out of config.
    """
    bronze_cfg = config["bronze"]
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
        "bronze_cfg": bronze_cfg,
        "paths_cfg": paths_cfg,
        "filenames": filenames,
        "pipeline_cfg": pipeline_cfg,
        "layer_name": bronze_cfg["layer_name"],
        "bronze_version": config["versions"]["bronze"],
        "truth_version": config["versions"]["truth"],
        "pipeline_mode": pipeline_cfg["execution_mode"],
        "run_mode": config["runtime"]["mode"],
        "profile": config["runtime"]["profile"],
        "wandb_project": config["wandb"]["project"],
        "wandb_entity": config["wandb"]["entity"],
        "wandb_run_name": f"{config['versions']['bronze']}",
        "raw_data_path": Path(paths_cfg["data_raw_dir"]),
        "bronze_data_path": Path(paths_cfg["data_bronze_train_dir"]),
        "bronze_artifacts_path": Path(paths_cfg["bronze_artifacts_dir"]),
        "truths_path": Path(paths_cfg["truths_dir"]),
        "truth_index_path": Path(paths_cfg["truth_index_path"]),
        "logs_path": Path(paths_cfg["logs_root"]),
        "raw_file_path": Path(paths_cfg["raw_file_path"]).parent,
        "raw_file_name": config["dataset"]["raw_file_name"],
        "dataset_name_argument": None,
        "dataset_name_config": config["dataset"]["name"],
        "dataset_candidates": bronze_cfg["dataset_candidates"],
        "split_status": config["dataset"]["split_status"],
        "label_type": config["dataset"]["label_type"],
        "label_source": config["dataset"]["label_source"],
        "run_id": config["dataset"]["run_id"],
        "asset_id": config["dataset"]["asset_id"],
        "add_record_id": bool(bronze_cfg["add_record_id"]),
        "record_id_inputs": list(bronze_cfg["record_id_inputs"]),
        "record_id_method": bronze_cfg["record_id_method"],
        "bronze_train_data_file_name": filenames["bronze_train_file_name"],
        "bronze_source_mode": "file",   # "file" | "postgres_handoff"
        "dataset_name_postgres": "pump",
        "postgres_source_table_name": "bronze_observations_input_stage",
        "postgres_source_table_dataset_map": {
            "bronze_observations_input_stage": "pump",
        },
        "process_run_id": make_process_run_id(bronze_cfg["process_run_id_prefix"]),
    }

    return runtime_inputs


def _apply_runtime_overrides(
    runtime_inputs: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
    asset_id: Optional[str] = None,
    raw_file_name: Optional[str] = None,
    dataset_name_argument: Optional[str] = None,
    bronze_source_mode: Optional[str] = None,
    dataset_name_postgres: Optional[str] = None,
    postgres_source_table_name: Optional[str] = None,
    postgres_source_table_dataset_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Apply explicit runtime overrides on top of config-derived defaults.
    """
    updated_inputs = dict(runtime_inputs)

    if run_id is not None and str(run_id).strip() != "":
        updated_inputs["run_id"] = str(run_id).strip()

    if asset_id is not None and str(asset_id).strip() != "":
        updated_inputs["asset_id"] = str(asset_id).strip()

    if raw_file_name is not None and str(raw_file_name).strip() != "":
        updated_inputs["raw_file_name"] = str(raw_file_name).strip()

    if dataset_name_argument is not None and str(dataset_name_argument).strip() != "":
        updated_inputs["dataset_name_argument"] = str(dataset_name_argument).strip()

    if bronze_source_mode is not None and str(bronze_source_mode).strip() != "":
        updated_inputs["bronze_source_mode"] = str(bronze_source_mode).strip()

    if dataset_name_postgres is not None and str(dataset_name_postgres).strip() != "":
        updated_inputs["dataset_name_postgres"] = str(dataset_name_postgres).strip()

    if postgres_source_table_name is not None and str(postgres_source_table_name).strip() != "":
        updated_inputs["postgres_source_table_name"] = str(postgres_source_table_name).strip()

    if postgres_source_table_dataset_map is not None:
        updated_inputs["postgres_source_table_dataset_map"] = dict(postgres_source_table_dataset_map)

    valid_source_modes = {"file", "postgres_handoff"}
    if updated_inputs["bronze_source_mode"] not in valid_source_modes:
        raise ValueError(
            f"Unsupported bronze_source_mode: {updated_inputs['bronze_source_mode']}. "
            f"Expected one of: {sorted(valid_source_modes)}"
        )

    return updated_inputs


def _ensure_stage_directories(runtime_inputs: Dict[str, Any]) -> None:
    """
    Ensure required Bronze directories exist.
    """
    runtime_inputs["bronze_data_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["bronze_artifacts_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["truths_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["logs_path"].mkdir(parents=True, exist_ok=True)


def _initialize_bronze_logger(paths) -> logging.Logger:
    """
    Create and configure the Bronze logger.
    """
    bronze_log_path = paths.logs / "bronze.log"
    bronze_log_path.parent.mkdir(parents=True, exist_ok=True)

    configure_logging(
        "capstone",
        bronze_log_path,
        level=logging.DEBUG,
        overwrite_handlers=True,
    )

    logger = logging.getLogger("capstone.bronze")
    logger.info("Bronze stage starting")
    log_layer_paths(paths, current_layer="bronze", logger=logger)

    return logger


def _load_bronze_source_dataframe(
    *,
    runtime_inputs: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Load Bronze source dataframe from file or postgres handoff.
    """
    bronze_source_mode = runtime_inputs["bronze_source_mode"]

    if bronze_source_mode == "file":
        logger.info("Loading Bronze source dataframe from file.")

        dataframe = ingest_data(
            runtime_inputs["raw_file_path"],
            file_name=runtime_inputs["raw_file_name"],
            dataset_name=runtime_inputs["dataset_name_argument"],
            dataset_name_config=runtime_inputs["dataset_name_config"],
            dataset_candidates=runtime_inputs["dataset_candidates"],
            split=runtime_inputs["split_status"],
            run_id=runtime_inputs["run_id"],
            asset_id=runtime_inputs["asset_id"],
            add_record_id=runtime_inputs["add_record_id"],
            validate=True,
        )

        return dataframe

    if bronze_source_mode == "postgres_handoff":
        logger.info("Loading Bronze source dataframe from postgres handoff.")

        engine = get_engine_from_env()

        dataframe = read_layer_dataframe(
            engine=engine,
            schema="public",
            table_name=runtime_inputs["postgres_source_table_name"],
            where_clause="dataset_id = :dataset_id AND run_id = :run_id",
            params={
                "dataset_id": runtime_inputs["dataset_name_postgres"],
                "run_id": runtime_inputs["run_id"],
            },
            order_by="batch_id, row_in_batch",
            require_exists=True,
        )

        return dataframe

    raise ValueError(f"Unsupported bronze_source_mode: {bronze_source_mode}")


def _build_and_save_bronze_truth(
    *,
    dataframe: pd.DataFrame,
    dataset_name: str,
    resolution_payload: Dict[str, Any],
    runtime_inputs: Dict[str, Any],
    raw_source_path_for_truth: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build Bronze truth record, stamp truth columns, and save truth artifact.
    """
    bronze_truth = initialize_layer_truth(
        truth_version=runtime_inputs["truth_version"],
        dataset_name=dataset_name,
        layer_name=runtime_inputs["layer_name"],
        process_run_id=runtime_inputs["process_run_id"],
        pipeline_mode=runtime_inputs["pipeline_mode"],
        parent_truth_hash=None,
    )

    bronze_truth = update_truth_section(
        bronze_truth,
        "config_snapshot",
        {
            "bronze_version": runtime_inputs["bronze_version"],
            "split_status": runtime_inputs["split_status"],
            "label_type": runtime_inputs["label_type"],
            "label_source": runtime_inputs["label_source"],
            "run_id": runtime_inputs["run_id"],
            "asset_id": runtime_inputs["asset_id"],
            "add_record_id": runtime_inputs["add_record_id"],
            "record_id_inputs": (
                runtime_inputs["record_id_inputs"]
                if runtime_inputs["add_record_id"]
                else None
            ),
            "record_id_method": (
                runtime_inputs["record_id_method"]
                if runtime_inputs["add_record_id"]
                else None
            ),
            "pipeline_mode": runtime_inputs["pipeline_mode"],
            "bronze_source_mode": runtime_inputs["bronze_source_mode"],
        },
    )

    bronze_truth = update_truth_section(
        bronze_truth,
        "runtime_facts",
        {
            "source_run_id": runtime_inputs["run_id"],
            "raw_file_path": raw_source_path_for_truth,
            "raw_data_dir": str(runtime_inputs["raw_data_path"]),
            "dataset_name_final": dataset_name,
            "dataset_source_column": resolution_payload["dataset_resolution_attrs"].get(
                "dataset_source_column"
            ),
            "dataset_method": resolution_payload["dataset_resolution_attrs"].get(
                "dataset_method"
            ),
        },
    )

    bronze_truth = update_truth_section(
        bronze_truth,
        "artifact_paths",
        {
            "bronze_output_dir": str(runtime_inputs["bronze_data_path"]),
            "bronze_output_file_name": runtime_inputs["bronze_train_data_file_name"],
        },
    )

    bronze_truth = update_truth_section(
        bronze_truth,
        "notes",
        {
            "purpose": "Bronze ingestion truth record",
        },
    )

    raw_source_path_obj = Path(raw_source_path_for_truth)
    if raw_source_path_obj.exists():
        bronze_source_fingerprint = build_file_fingerprint(raw_source_path_obj)
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

    bronze_truth_hash = bronze_truth_record["truth_hash"]

    dataframe = stamp_truth_columns(
        dataframe,
        truth_hash=bronze_truth_hash,
        parent_truth_hash=None,
        pipeline_mode=runtime_inputs["pipeline_mode"],
    )

    bronze_truth_path = save_truth_record(
        bronze_truth_record,
        truth_dir=runtime_inputs["truths_path"],
        dataset_name=dataset_name,
        layer_name=runtime_inputs["layer_name"],
    )

    append_truth_index(
        bronze_truth_record,
        truth_index_path=runtime_inputs["truth_index_path"],
    )

    logger.info("Bronze truth hash: %s", bronze_truth_hash)
    logger.info("Bronze truth path: %s", bronze_truth_path)
    logger.info("Bronze process_run_id: %s", runtime_inputs["process_run_id"])

    return {
        "dataframe": dataframe,
        "bronze_truth_record": bronze_truth_record,
        "bronze_truth_hash": bronze_truth_hash,
        "bronze_truth_path": str(bronze_truth_path),
    }


def run_bronze_preprocessing(
    *,
    config_root: Optional[Path] = None,
    dataset: str = "pump",
    mode: str = "train",
    profile: str = "default",
    project_root: Optional[Path] = None,
    run_id: Optional[str] = None,
    asset_id: Optional[str] = None,
    raw_file_name: Optional[str] = None,
    dataset_name_argument: Optional[str] = None,
    bronze_source_mode: Optional[str] = None,
    dataset_name_postgres: Optional[str] = None,
    postgres_source_table_name: Optional[str] = None,
    postgres_source_table_dataset_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run the Bronze preprocessing stage.
    """
    paths = get_paths()

    config_root = config_root or paths.configs
    project_root = project_root or paths.root

    config = load_pipeline_config(
        config_root=config_root,
        stage="bronze",
        dataset=dataset,
        mode=mode,
        profile=profile,
        project_root=project_root,
    ).data

    runtime_inputs = _build_default_runtime_inputs(config=config)

    runtime_inputs = _apply_runtime_overrides(
        runtime_inputs,
        run_id=run_id,
        asset_id=asset_id,
        raw_file_name=raw_file_name,
        dataset_name_argument=dataset_name_argument,
        bronze_source_mode=bronze_source_mode,
        dataset_name_postgres=dataset_name_postgres,
        postgres_source_table_name=postgres_source_table_name,
        postgres_source_table_dataset_map=postgres_source_table_dataset_map,
    )

    _ensure_stage_directories(runtime_inputs)

    logger = _initialize_bronze_logger(paths)

    truthed_config = build_truth_config_block(config)
    truthed_config["pipeline"] = runtime_inputs["pipeline_cfg"]

    set_wandb_dir_from_config(config)
    export_config_snapshot(
        config,
        output_path=runtime_inputs["bronze_artifacts_path"] / f"{dataset}__bronze__resolved_config.yaml",
    )

    logger.info(
        "Bronze runtime initialized | dataset=%s | mode=%s | profile=%s | source_mode=%s | run_id=%s | asset_id=%s | raw_file_name=%s",
        dataset,
        mode,
        profile,
        runtime_inputs["bronze_source_mode"],
        runtime_inputs["run_id"],
        runtime_inputs["asset_id"],
        runtime_inputs["raw_file_name"],
    )

    dataframe = _load_bronze_source_dataframe(
        runtime_inputs=runtime_inputs,
        logger=logger,
    )

    source_path_for_resolution = (
        str(runtime_inputs["raw_file_path"] / runtime_inputs["raw_file_name"])
        if runtime_inputs["bronze_source_mode"] == "file"
        else None
    )

    handoff_dataset_name = (
        runtime_inputs["dataset_name_postgres"]
        if runtime_inputs["bronze_source_mode"] == "postgres_handoff"
        else None
    )

    source_table_name = (
        runtime_inputs["postgres_source_table_name"]
        if runtime_inputs["bronze_source_mode"] == "postgres_handoff"
        else None
    )

    dataframe, resolution_payload = prepare_bronze_dataframe(
        dataframe,
        argument_dataset_name=runtime_inputs["dataset_name_argument"],
        config_dataset_name=runtime_inputs["dataset_name_config"],
        handoff_dataset_name=handoff_dataset_name,
        source_table_name=source_table_name,
        source_table_dataset_map=runtime_inputs["postgres_source_table_dataset_map"],
        fallback_dataset_name="unknown_dataset",
        source_path=source_path_for_resolution,
        dataset_column="meta__dataset",
        reorder_columns=True,
    )

    resolved_dataset_name = resolution_payload["dataset_resolution_attrs"]["dataset_name"]

    logger.info(
        "Bronze dataframe prepared successfully | rows=%d | columns=%d | dataset_name=%s",
        dataframe.shape[0],
        dataframe.shape[1],
        resolved_dataset_name,
    )

    raw_source_path_for_truth = (
        str(runtime_inputs["raw_file_path"] / runtime_inputs["raw_file_name"])
        if runtime_inputs["bronze_source_mode"] == "file"
        else f"postgres://public/{runtime_inputs['postgres_source_table_name']}"
    )

    truth_payload = _build_and_save_bronze_truth(
        dataframe=dataframe,
        dataset_name=resolved_dataset_name,
        resolution_payload=resolution_payload,
        runtime_inputs=runtime_inputs,
        raw_source_path_for_truth=raw_source_path_for_truth,
        logger=logger,
    )

    dataframe = truth_payload["dataframe"]

    dataframe = dataframe.copy()
    dataframe = dataframe[
        collect_meta_columns(list(dataframe.columns))
        + [column for column in dataframe.columns if not column.startswith("meta__")]
    ]

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

    save_data(
        dataframe,
        runtime_inputs["bronze_data_path"],
        runtime_inputs["bronze_train_data_file_name"],
    )

    logger.info(
        "Bronze dataframe saved successfully to %s / %s",
        runtime_inputs["bronze_data_path"],
        runtime_inputs["bronze_train_data_file_name"],
    )

    finalize_wandb_stage()

    summary = {
        "status": "success",
        "layer_name": runtime_inputs["layer_name"],
        "dataset_name": resolved_dataset_name,
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "bronze_output_path": str(
            runtime_inputs["bronze_data_path"] / runtime_inputs["bronze_train_data_file_name"]
        ),
        "truth_hash": truth_payload["bronze_truth_hash"],
        "truth_path": truth_payload["bronze_truth_path"],
        "process_run_id": runtime_inputs["process_run_id"],
        "source_mode": runtime_inputs["bronze_source_mode"],
        "run_id": runtime_inputs["run_id"],
        "asset_id": runtime_inputs["asset_id"],
        "raw_file_name": runtime_inputs["raw_file_name"],
        "dataset_name_argument": runtime_inputs["dataset_name_argument"],
        "dataset_name_postgres": runtime_inputs["dataset_name_postgres"],
        "postgres_source_table_name": runtime_inputs["postgres_source_table_name"],
    }

    logger.info("Bronze stage completed successfully.")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for Bronze stage execution.
    """
    parser = argparse.ArgumentParser(description="Run Bronze preprocessing stage.")

    parser.add_argument("--dataset", default="pump")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--project-root", default=None)

    parser.add_argument("--run-id", default=None)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--raw-file-name", default=None)
    parser.add_argument("--dataset-name-argument", default=None)

    parser.add_argument(
        "--bronze-source-mode",
        default=None,
        choices=["file", "postgres_handoff"],
    )
    parser.add_argument("--dataset-name-postgres", default=None)
    parser.add_argument("--postgres-source-table-name", default=None)
    parser.add_argument(
        "--postgres-source-table-dataset-map",
        default=None,
        help='JSON string like \'{"bronze_observations_input_stage": "pump"}\'',
    )

    return parser


def _parse_dataset_map_arg(dataset_map_text: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Parse optional JSON dataset map from CLI.
    """
    if dataset_map_text is None or str(dataset_map_text).strip() == "":
        return None

    parsed_value = json.loads(dataset_map_text)

    if not isinstance(parsed_value, dict):
        raise ValueError("postgres_source_table_dataset_map must parse to a dict.")

    return {str(key): str(value) for key, value in parsed_value.items()}


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    result = run_bronze_preprocessing(
        config_root=Path(args.config_root) if args.config_root else None,
        dataset=args.dataset,
        mode=args.mode,
        profile=args.profile,
        project_root=Path(args.project_root) if args.project_root else None,
        run_id=args.run_id,
        asset_id=args.asset_id,
        raw_file_name=args.raw_file_name,
        dataset_name_argument=args.dataset_name_argument,
        bronze_source_mode=args.bronze_source_mode,
        dataset_name_postgres=args.dataset_name_postgres,
        postgres_source_table_name=args.postgres_source_table_name,
        postgres_source_table_dataset_map=_parse_dataset_map_arg(
            args.postgres_source_table_dataset_map
        ),
    )

    print(result)





'''
Sample Commands

# File Mode Override:
python -m pipelines.bronze.run_bronze_preprocessing \
  --dataset pump \
  --mode train \
  --profile default \
  --bronze-source-mode postgres_handoff \
  --run-id synthetic_run_001 \
  --dataset-name-postgres pump \
  --postgres-source-table-name bronze_observations_input_stage \
  --postgres-source-table-dataset-map '{"bronze_observations_input_stage":"pump"}'

# Postgres Mode Override

python -m pipelines.bronze.run_bronze_preprocessing \
  --dataset pump \
  --mode train \
  --profile default \
  --bronze-source-mode postgres_handoff \
  --run-id synthetic_run_001 \
  --dataset-name-postgres pump \
  --postgres-source-table-name bronze_observations_input_stage \
  --postgres-source-table-dataset-map '{"bronze_observations_input_stage":"pump"}'

'''