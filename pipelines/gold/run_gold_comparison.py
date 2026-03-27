"""
pipelines/gold/run_gold_comparison.py

Gold comparison pipeline runner for the capstone project.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import wandb

from utils.paths import get_paths
from utils.file_io import save_json, load_json, save_data
from utils.logging_setup import configure_logging, log_layer_paths
from utils.wandb_utils import finalize_wandb_stage
from utils.truths import (
    initialize_layer_truth,
    update_truth_section,
    build_truth_record,
    save_truth_record,
    append_truth_index,
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
from utils.pipeline.gold_comparison import (
    load_model_result_artifacts,
    validate_comparison_inputs,
    build_model_comparison_dataframe,
    build_alert_count_comparison,
    build_metric_comparison,
    build_comparison_summary,
    build_baseline_vs_best_cascade_delta,
)


def _build_default_runtime_inputs(
    *,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pull commonly used Gold comparison runtime inputs from config.
    """
    comparison_cfg = config["gold_comparison"]
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
        "comparison_cfg": comparison_cfg,
        "paths_cfg": paths_cfg,
        "filenames": filenames,
        "pipeline_cfg": pipeline_cfg,
        "stage": "gold_comparison",
        "layer_name": comparison_cfg["layer_name"],
        "gold_version": config["versions"]["gold"],
        "truth_version": config["versions"]["truth"],
        "pipeline_mode": pipeline_cfg["execution_mode"],
        "run_mode": config["runtime"]["mode"],
        "profile": config["runtime"]["profile"],
        "dataset_name_config": config["dataset"]["name"],
        "process_run_id": make_process_run_id(comparison_cfg["process_run_id_prefix"]),
        "wandb_project": config["wandb"]["project"],
        "wandb_entity": config["wandb"]["entity"],
        "wandb_run_name": f"{config['versions']['gold']}_comparison",
        "gold_train_data_path": Path(paths_cfg["data_gold_train_dir"]),
        "gold_artifacts_path": Path(paths_cfg["gold_artifacts_dir"]),
        "truths_path": Path(paths_cfg["truths_dir"]),
        "truth_index_path": Path(paths_cfg["truth_index_path"]),
        "logs_path": Path(paths_cfg["logs_root"]),
        "baseline_summary_file_name": filenames["gold_baseline_summary_file_name"],
        "cascade_summary_file_name": filenames["gold_cascade_summary_file_name"],
        "comparison_summary_file_name": filenames["gold_comparison_summary_file_name"],
        "comparison_table_file_name": filenames["gold_comparison_table_file_name"],
        "comparison_alert_table_file_name": filenames["gold_comparison_alert_table_file_name"],
        "comparison_metric_table_file_name": filenames["gold_comparison_metric_table_file_name"],
        "cascade_variants": list(comparison_cfg.get("cascade_variants", ["default", "tuned", "improved"])),
        "write_sql_output": bool(comparison_cfg.get("write_sql_output", False)),
        "gold_schema": comparison_cfg.get("gold_schema", "gold"),
    }

    return runtime_inputs


def _apply_runtime_overrides(
    runtime_inputs: Dict[str, Any],
    *,
    cascade_variants: Optional[List[str]] = None,
    write_sql_output: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Apply explicit runtime overrides.
    """
    updated_inputs = dict(runtime_inputs)

    if cascade_variants is not None and len(cascade_variants) > 0:
        updated_inputs["cascade_variants"] = list(cascade_variants)

    if write_sql_output is not None:
        updated_inputs["write_sql_output"] = bool(write_sql_output)

    return updated_inputs


def _ensure_stage_directories(runtime_inputs: Dict[str, Any]) -> None:
    runtime_inputs["gold_artifacts_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["truths_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["logs_path"].mkdir(parents=True, exist_ok=True)


def _initialize_comparison_logger(paths) -> logging.Logger:
    comparison_log_path = paths.logs / "gold_model_comparison.log"
    comparison_log_path.parent.mkdir(parents=True, exist_ok=True)

    configure_logging(
        "capstone",
        comparison_log_path,
        level=logging.DEBUG,
        overwrite_handlers=True,
    )

    logger = logging.getLogger("capstone.gold_comparison")
    logger.info("Gold comparison stage starting")
    log_layer_paths(paths, current_layer="gold", logger=logger)
    return logger


def _initialize_wandb_run(
    *,
    runtime_inputs: Dict[str, Any],
    logger: logging.Logger,
):
    wandb_run = wandb.init(
        project=runtime_inputs["wandb_project"],
        entity=runtime_inputs["wandb_entity"],
        name=runtime_inputs["wandb_run_name"],
        job_type="gold_model_comparison",
        config={
            "gold_version": runtime_inputs["gold_version"],
            "cascade_variants": runtime_inputs["cascade_variants"],
            "gold_artifacts_dir": str(runtime_inputs["gold_artifacts_path"]),
        },
    )
    logger.info("W&B initialized: %s", wandb.run.name)
    return wandb_run


def _resolve_variant_summary_path(
    gold_artifacts_path: Path,
    base_file_name: str,
    variant: str,
) -> Path:
    """
    Resolve cascade summary path for a variant.

    First-pass convention:
    - if file exists directly, use it
    - else insert variant before suffix
    """
    direct_path = gold_artifacts_path / base_file_name
    if direct_path.exists() and variant == "default":
        return direct_path

    file_stem = Path(base_file_name).stem
    suffix = Path(base_file_name).suffix
    variant_name = f"{file_stem}__{variant}{suffix}"
    return gold_artifacts_path / variant_name


def _load_comparison_inputs(
    *,
    runtime_inputs: Dict[str, Any],
    logger: logging.Logger,
):
    baseline_summary_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["baseline_summary_file_name"]
    if not baseline_summary_path.exists():
        raise FileNotFoundError(f"Baseline summary not found: {baseline_summary_path}")

    baseline_summary = load_json(baseline_summary_path)

    cascade_summaries: List[Dict[str, Any]] = []
    cascade_paths: List[str] = []

    for variant in runtime_inputs["cascade_variants"]:
        variant_path = _resolve_variant_summary_path(
            runtime_inputs["gold_artifacts_path"],
            runtime_inputs["cascade_summary_file_name"],
            variant,
        )
        if not variant_path.exists():
            logger.warning("Cascade summary for variant '%s' not found at %s", variant, variant_path)
            continue

        summary = load_json(variant_path)
        if "variant" not in summary:
            summary["variant"] = variant
        cascade_summaries.append(summary)
        cascade_paths.append(str(variant_path))

    if len(cascade_summaries) == 0:
        raise FileNotFoundError("No cascade summaries were found for comparison.")

    logger.info("Loaded baseline summary: %s", baseline_summary_path)
    logger.info("Loaded %d cascade summaries.", len(cascade_summaries))

    return {
        "baseline_summary": baseline_summary,
        "baseline_summary_path": baseline_summary_path,
        "cascade_summaries": cascade_summaries,
        "cascade_paths": cascade_paths,
    }


def _build_and_save_comparison_truth(
    *,
    dataset_name: str,
    runtime_inputs: Dict[str, Any],
    validation_info: Dict[str, Any],
    comparison_summary: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Build and save Gold comparison truth.
    """
    comparison_truth = initialize_layer_truth(
        truth_version=runtime_inputs["truth_version"],
        dataset_name=dataset_name,
        layer_name=runtime_inputs["layer_name"],
        process_run_id=runtime_inputs["process_run_id"],
        pipeline_mode=runtime_inputs["pipeline_mode"],
        parent_truth_hash=None,
    )

    comparison_truth = update_truth_section(
        comparison_truth,
        "config_snapshot",
        {
            "gold_version": runtime_inputs["gold_version"],
            "cascade_variants": runtime_inputs["cascade_variants"],
            "pipeline_mode": runtime_inputs["pipeline_mode"],
        },
    )

    comparison_truth = update_truth_section(
        comparison_truth,
        "comparison_validation",
        validation_info,
    )

    comparison_truth = update_truth_section(
        comparison_truth,
        "comparison_summary",
        comparison_summary,
    )

    comparison_truth_record = build_truth_record(
        truth_base=comparison_truth,
        row_count=0,
        column_count=0,
        meta_columns=[],
        feature_columns=[],
    )

    comparison_truth_hash = comparison_truth_record["truth_hash"]

    comparison_truth_path = save_truth_record(
        comparison_truth_record,
        truth_dir=runtime_inputs["truths_path"],
        dataset_name=dataset_name,
        layer_name=runtime_inputs["layer_name"],
    )

    append_truth_index(
        comparison_truth_record,
        truth_index_path=runtime_inputs["truth_index_path"],
    )

    logger.info("Comparison truth hash: %s", comparison_truth_hash)
    logger.info("Comparison truth path: %s", comparison_truth_path)

    return {
        "comparison_truth_hash": comparison_truth_hash,
        "comparison_truth_path": str(comparison_truth_path),
    }


def _optionally_write_sql_output(
    *,
    comparison_df: pd.DataFrame,
    runtime_inputs: Dict[str, Any],
    dataset_name: str,
    logger: logging.Logger,
) -> Optional[str]:
    if not runtime_inputs["write_sql_output"]:
        return None

    engine = get_engine_from_env()

    sql_dataframe = comparison_df.copy()
    sql_dataframe["meta__dataset"] = dataset_name
    sql_dataframe["meta__process_run_id"] = runtime_inputs["process_run_id"]
    sql_dataframe["meta__pipeline_mode"] = runtime_inputs["pipeline_mode"]

    sql_dataframe = prepare_layer_dataframe(
        sql_dataframe,
        truth_hash=None,
        parent_truth_hash=None,
        pipeline_mode=runtime_inputs["pipeline_mode"],
        process_run_id=runtime_inputs["process_run_id"],
        add_loaded_at_column=True,
    )

    table_name = write_layer_dataframe(
        engine=engine,
        dataframe=sql_dataframe,
        schema=runtime_inputs["gold_schema"],
        dataset_name=dataset_name,
        if_exists="replace",
        index=False,
    )

    logger.info(
        "Wrote Gold comparison SQL table: %s.%s",
        runtime_inputs["gold_schema"],
        table_name,
    )

    return table_name


def run_gold_comparison(
    *,
    config_root: Optional[Path] = None,
    dataset: str = "pump",
    mode: str = "train",
    profile: str = "default",
    project_root: Optional[Path] = None,
    cascade_variants: Optional[List[str]] = None,
    write_sql_output: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run Gold comparison stage.
    """
    paths = get_paths()

    config_root = config_root or paths.configs
    project_root = project_root or paths.root

    config = load_pipeline_config(
        config_root=config_root,
        stage="gold_comparison",
        dataset=dataset,
        mode=mode,
        profile=profile,
        project_root=project_root,
    ).data

    runtime_inputs = _build_default_runtime_inputs(config=config)
    runtime_inputs = _apply_runtime_overrides(
        runtime_inputs,
        cascade_variants=cascade_variants,
        write_sql_output=write_sql_output,
    )

    _ensure_stage_directories(runtime_inputs)
    logger = _initialize_comparison_logger(paths)

    truthed_config = build_truth_config_block(config)
    truthed_config["pipeline"] = runtime_inputs["pipeline_cfg"]

    set_wandb_dir_from_config(config)
    export_config_snapshot(
        config,
        output_path=runtime_inputs["gold_artifacts_path"] / f"{dataset}__gold_comparison__resolved_config.yaml",
    )

    comparison_inputs = _load_comparison_inputs(
        runtime_inputs=runtime_inputs,
        logger=logger,
    )

    wandb_run = _initialize_wandb_run(
        runtime_inputs=runtime_inputs,
        logger=logger,
    )

    ledger = Ledger(stage=runtime_inputs["stage"], recipe_id="gold_comparison")
    ledger.add(
        kind="step",
        step="load_model_summaries",
        message="Loaded baseline and cascade summaries",
        data={
            "baseline_summary_path": str(comparison_inputs["baseline_summary_path"]),
            "cascade_summary_paths": comparison_inputs["cascade_paths"],
        },
        logger=logger,
    )

    comparison_payload = load_model_result_artifacts(
        baseline_summary=comparison_inputs["baseline_summary"],
        cascade_summaries=comparison_inputs["cascade_summaries"],
    )

    validation_info = validate_comparison_inputs(comparison_payload)
    if not validation_info["is_valid"]:
        raise ValueError(
            "Comparison input validation failed:\n"
            + "\n".join(validation_info["issues"])
        )

    comparison_df = build_model_comparison_dataframe(comparison_payload)
    alert_df = build_alert_count_comparison(comparison_payload)
    metric_df = build_metric_comparison(comparison_payload)

    comparison_summary = build_comparison_summary(comparison_df)
    baseline_vs_best_delta = build_baseline_vs_best_cascade_delta(comparison_df)

    dataset_name = validation_info["baseline_dataset_name"] or runtime_inputs["dataset_name_config"]

    comparison_summary_payload = {
        "dataset_name": dataset_name,
        "validation_info": validation_info,
        "comparison_summary": comparison_summary,
        "baseline_vs_best_cascade_delta": baseline_vs_best_delta,
    }

    comparison_truth_payload = _build_and_save_comparison_truth(
        dataset_name=dataset_name,
        runtime_inputs=runtime_inputs,
        validation_info=validation_info,
        comparison_summary=comparison_summary_payload,
        logger=logger,
    )

    comparison_summary_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["comparison_summary_file_name"]
    comparison_table_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["comparison_table_file_name"]
    comparison_alert_table_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["comparison_alert_table_file_name"]
    comparison_metric_table_path = runtime_inputs["gold_artifacts_path"] / runtime_inputs["comparison_metric_table_file_name"]

    save_json(comparison_summary_payload, comparison_summary_path)
    save_data(comparison_df, comparison_table_path.parent, comparison_table_path.name)
    save_data(alert_df, comparison_alert_table_path.parent, comparison_alert_table_path.name)
    save_data(metric_df, comparison_metric_table_path.parent, comparison_metric_table_path.name)

    saved_ledger_path = ledger.write_json(
        runtime_inputs["gold_artifacts_path"] / f"gold_comparison__{dataset_name}__ledger.json"
    )

    sql_table_name = _optionally_write_sql_output(
        comparison_df=comparison_df,
        runtime_inputs=runtime_inputs,
        dataset_name=dataset_name,
        logger=logger,
    )

    finalize_wandb_stage(
        run=wandb_run,
        stage=runtime_inputs["stage"],
        dataframe=comparison_df,
        project_root=paths.root,
        logs_dir=paths.logs,
        dataset_dirs=[runtime_inputs["gold_artifacts_path"]],
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
        "model_count": int(len(comparison_df)),
        "comparison_summary_path": str(comparison_summary_path),
        "comparison_table_path": str(comparison_table_path),
        "comparison_alert_table_path": str(comparison_alert_table_path),
        "comparison_metric_table_path": str(comparison_metric_table_path),
        "truth_hash": comparison_truth_payload["comparison_truth_hash"],
        "truth_path": comparison_truth_payload["comparison_truth_path"],
        "process_run_id": runtime_inputs["process_run_id"],
        "best_f1_model": comparison_summary.get("best_f1_model"),
        "lowest_alert_model": comparison_summary.get("lowest_alert_model"),
        "baseline_vs_best_cascade_delta": baseline_vs_best_delta,
        "sql_table_name": sql_table_name,
        "ledger_path": str(saved_ledger_path),
    }

    logger.info("Gold comparison stage completed successfully.")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gold comparison stage.")
    parser.add_argument("--dataset", default="pump")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--cascade-variants",
        nargs="*",
        default=None,
        help="Example: --cascade-variants default tuned improved",
    )
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

    result = run_gold_comparison(
        config_root=Path(args.config_root) if args.config_root else None,
        dataset=args.dataset,
        mode=args.mode,
        profile=args.profile,
        project_root=Path(args.project_root) if args.project_root else None,
        cascade_variants=args.cascade_variants,
        write_sql_output=_parse_optional_bool(args.write_sql_output),
    )

    print(json.dumps(result, indent=2))

'''

#Example usage

# Compare baseline against default, tuned, and improved cascade variants:

python -m pipelines.gold.run_gold_comparison \
  --dataset pump \
  --mode train \
  --profile default \
  --cascade-variants default tuned improved

# ---- ---- ---- ---- #
# Compare only baseline vs improved:

python -m pipelines.gold.run_gold_comparison \
  --dataset pump \
  --mode train \
  --profile default \
  --cascade-variants improved

# ---- ---- ---- ---- #
#Enable SQL write:

python -m pipelines.gold.run_gold_comparison \
  --dataset pump \
  --mode train \
  --profile default \
  --cascade-variants default tuned improved \
  --write-sql-output true

'''