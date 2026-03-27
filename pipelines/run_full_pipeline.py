"""
pipelines/run_full_pipeline.py

Top-level pipeline orchestrator for the capstone project.

Purpose
-------
Run the pipeline end-to-end, or run a bounded subset of stages, using the
converted stage runners.

Supported stages
----------------
1. bronze
2. silver_preeda
3. silver_eda
4. gold_preprocessing
5. gold_baseline
6. gold_cascade_default
7. gold_cascade_tuned
8. gold_cascade_improved
9. gold_comparison

Examples
--------
Full run:
    python -m pipelines.run_full_pipeline --dataset pump --mode train --profile default

Run from Silver onward:
    python -m pipelines.run_full_pipeline \
        --dataset pump \
        --mode train \
        --profile default \
        --start-stage silver_preeda

Run only Gold branch:
    python -m pipelines.run_full_pipeline \
        --dataset pump \
        --mode train \
        --profile default \
        --start-stage gold_preprocessing \
        --end-stage gold_comparison

Skip tuned cascade:
    python -m pipelines.run_full_pipeline \
        --dataset pump \
        --mode train \
        --profile default \
        --cascade-variants default improved

Run Bronze with overrides:
    python -m pipelines.run_full_pipeline \
        --dataset pump \
        --mode train \
        --profile default \
        --raw-file-name pump_synthetic_batch.parquet \
        --run-id synthetic_run_001 \
        --asset-id pump_01 \
        --dataset-name-argument pump
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.paths import get_paths
from utils.logging_setup import configure_logging

from pipelines.bronze.run_bronze_preprocessing import run_bronze_preprocessing
from pipelines.silver.run_silver_preeda import run_silver_preeda
from pipelines.silver.run_silver_eda import run_silver_eda
from pipelines.gold.run_gold_preprocessing import run_gold_preprocessing
from pipelines.gold.run_gold_baseline_modeling import run_gold_baseline_modeling
from pipelines.gold.run_gold_cascade_modeling import run_gold_cascade_modeling
from pipelines.gold.run_gold_comparison import run_gold_comparison


STAGE_ORDER: List[str] = [
    "bronze",
    "silver_preeda",
    "silver_eda",
    "gold_preprocessing",
    "gold_baseline",
    "gold_cascade_default",
    "gold_cascade_tuned",
    "gold_cascade_improved",
    "gold_comparison",
]


def _initialize_orchestrator_logger(logs_dir: Path) -> logging.Logger:
    """
    Configure top-level orchestrator logging.
    """
    log_path = logs_dir / "pipeline_orchestrator.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    configure_logging(
        "capstone",
        log_path,
        level=logging.DEBUG,
        overwrite_handlers=True,
    )

    logger = logging.getLogger("capstone.orchestrator")
    logger.info("Pipeline orchestrator starting.")
    logger.info("Orchestrator log path: %s", log_path)

    return logger


def _validate_stage_name(stage_name: Optional[str], *, arg_name: str) -> Optional[str]:
    """
    Validate stage name against known stages.
    """
    if stage_name is None:
        return None

    stage_name = str(stage_name).strip()
    if stage_name not in STAGE_ORDER:
        raise ValueError(
            f"Invalid {arg_name}: {stage_name}. "
            f"Expected one of: {STAGE_ORDER}"
        )
    return stage_name


def _resolve_stages_to_run(
    *,
    start_stage: Optional[str],
    end_stage: Optional[str],
    cascade_variants: List[str],
) -> List[str]:
    """
    Resolve the exact ordered stage list to execute.
    """
    start_stage = _validate_stage_name(start_stage, arg_name="start_stage")
    end_stage = _validate_stage_name(end_stage, arg_name="end_stage")

    selected_order = list(STAGE_ORDER)

    if start_stage is not None:
        selected_order = selected_order[STAGE_ORDER.index(start_stage):]

    if end_stage is not None:
        selected_order = selected_order[: selected_order.index(end_stage) + 1]

    allowed_variant_stage_names = {
        "default": "gold_cascade_default",
        "tuned": "gold_cascade_tuned",
        "improved": "gold_cascade_improved",
    }

    cascade_variants_clean = [str(value).strip() for value in cascade_variants]
    invalid_variants = [value for value in cascade_variants_clean if value not in allowed_variant_stage_names]
    if invalid_variants:
        raise ValueError(
            f"Invalid cascade variant(s): {invalid_variants}. "
            f"Expected any of: {sorted(allowed_variant_stage_names)}"
        )

    cascade_stages_to_keep = {
        allowed_variant_stage_names[variant]
        for variant in cascade_variants_clean
    }

    filtered_order: List[str] = []
    for stage_name in selected_order:
        if stage_name.startswith("gold_cascade_"):
            if stage_name in cascade_stages_to_keep:
                filtered_order.append(stage_name)
        else:
            filtered_order.append(stage_name)

    return filtered_order


def _call_stage(
    *,
    stage_name: str,
    common_kwargs: Dict[str, Any],
    runtime_overrides: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Call one stage runner with the appropriate arguments.
    """
    logger.info("Starting stage: %s", stage_name)

    if stage_name == "bronze":
        return run_bronze_preprocessing(
            config_root=common_kwargs["config_root"],
            dataset=common_kwargs["dataset"],
            mode=common_kwargs["mode"],
            profile=common_kwargs["profile"],
            project_root=common_kwargs["project_root"],
            run_id=runtime_overrides.get("run_id"),
            asset_id=runtime_overrides.get("asset_id"),
            raw_file_name=runtime_overrides.get("raw_file_name"),
            dataset_name_argument=runtime_overrides.get("dataset_name_argument"),
            bronze_source_mode=runtime_overrides.get("bronze_source_mode"),
            dataset_name_postgres=runtime_overrides.get("dataset_name_postgres"),
            postgres_source_table_name=runtime_overrides.get("postgres_source_table_name"),
            postgres_source_table_dataset_map=runtime_overrides.get("postgres_source_table_dataset_map"),
        )

    if stage_name == "silver_preeda":
        return run_silver_preeda(
            config_root=common_kwargs["config_root"],
            dataset=common_kwargs["dataset"],
            mode=common_kwargs["mode"],
            profile=common_kwargs["profile"],
            project_root=common_kwargs["project_root"],
            bronze_train_data_file_name=runtime_overrides.get("bronze_train_data_file_name"),
            silver_train_data_file_name=runtime_overrides.get("silver_train_data_file_name"),
            feature_registry_file_name=runtime_overrides.get("feature_registry_file_name"),
            write_sql_output=runtime_overrides.get("silver_preeda_write_sql_output"),
        )

    if stage_name == "silver_eda":
        return run_silver_eda(
            config_root=common_kwargs["config_root"],
            dataset=common_kwargs["dataset"],
            mode=common_kwargs["mode"],
            profile=common_kwargs["profile"],
            project_root=common_kwargs["project_root"],
            silver_train_data_file_name=runtime_overrides.get("silver_train_data_file_name"),
            feature_registry_file_name=runtime_overrides.get("feature_registry_file_name"),
            max_onsets_to_use=runtime_overrides.get("max_onsets_to_use"),
            onset_pre_window=runtime_overrides.get("onset_pre_window"),
            onset_post_window=runtime_overrides.get("onset_post_window"),
            top_feature_count=runtime_overrides.get("top_feature_count"),
            dropped_parquet_file_name=runtime_overrides.get("dropped_parquet_file_name"),
            join_key=runtime_overrides.get("join_key"),
        )

    if stage_name == "gold_preprocessing":
        return run_gold_preprocessing(
            config_root=common_kwargs["config_root"],
            dataset=common_kwargs["dataset"],
            mode=common_kwargs["mode"],
            profile=common_kwargs["profile"],
            project_root=common_kwargs["project_root"],
            silver_train_data_file_name=runtime_overrides.get("silver_train_data_file_name"),
            feature_registry_file_name=runtime_overrides.get("feature_registry_file_name"),
            gold_preprocessed_file_name=runtime_overrides.get("gold_preprocessed_file_name"),
            gold_preprocessed_scaled_file_name=runtime_overrides.get("gold_preprocessed_scaled_file_name"),
            gold_fit_file_name=runtime_overrides.get("gold_fit_file_name"),
            gold_train_file_name=runtime_overrides.get("gold_train_file_name"),
            gold_test_file_name=runtime_overrides.get("gold_test_file_name"),
            train_fraction=runtime_overrides.get("train_fraction"),
            imputation_method=runtime_overrides.get("imputation_method"),
            scaler_kind=runtime_overrides.get("scaler_kind"),
            write_sql_output=runtime_overrides.get("gold_preprocessing_write_sql_output"),
        )

    if stage_name == "gold_baseline":
        return run_gold_baseline_modeling(
            config_root=common_kwargs["config_root"],
            dataset=common_kwargs["dataset"],
            mode=common_kwargs["mode"],
            profile=common_kwargs["profile"],
            project_root=common_kwargs["project_root"],
            gold_preprocessed_scaled_file_name=runtime_overrides.get("gold_preprocessed_scaled_file_name"),
            gold_fit_file_name=runtime_overrides.get("gold_fit_file_name"),
            gold_train_file_name=runtime_overrides.get("gold_train_file_name"),
            gold_test_file_name=runtime_overrides.get("gold_test_file_name"),
            baseline_model_file_name=runtime_overrides.get("baseline_model_file_name"),
            baseline_summary_file_name=runtime_overrides.get("baseline_summary_file_name"),
            threshold_percentile=runtime_overrides.get("baseline_threshold_percentile"),
            n_estimators=runtime_overrides.get("baseline_n_estimators"),
            contamination=runtime_overrides.get("baseline_contamination"),
            max_samples=runtime_overrides.get("baseline_max_samples"),
            max_features=runtime_overrides.get("baseline_max_features"),
            bootstrap=runtime_overrides.get("baseline_bootstrap"),
            random_state=runtime_overrides.get("baseline_random_state"),
            n_jobs=runtime_overrides.get("baseline_n_jobs"),
            write_sql_output=runtime_overrides.get("gold_baseline_write_sql_output"),
        )

    if stage_name == "gold_cascade_default":
        return run_gold_cascade_modeling(
            config_root=common_kwargs["config_root"],
            dataset=common_kwargs["dataset"],
            mode=common_kwargs["mode"],
            profile=common_kwargs["profile"],
            project_root=common_kwargs["project_root"],
            variant="default",
            write_sql_output=runtime_overrides.get("gold_cascade_write_sql_output"),
        )

    if stage_name == "gold_cascade_tuned":
        return run_gold_cascade_modeling(
            config_root=common_kwargs["config_root"],
            dataset=common_kwargs["dataset"],
            mode=common_kwargs["mode"],
            profile=common_kwargs["profile"],
            project_root=common_kwargs["project_root"],
            variant="tuned",
            write_sql_output=runtime_overrides.get("gold_cascade_write_sql_output"),
        )

    if stage_name == "gold_cascade_improved":
        return run_gold_cascade_modeling(
            config_root=common_kwargs["config_root"],
            dataset=common_kwargs["dataset"],
            mode=common_kwargs["mode"],
            profile=common_kwargs["profile"],
            project_root=common_kwargs["project_root"],
            variant="improved",
            write_sql_output=runtime_overrides.get("gold_cascade_write_sql_output"),
        )

    if stage_name == "gold_comparison":
        return run_gold_comparison(
            config_root=common_kwargs["config_root"],
            dataset=common_kwargs["dataset"],
            mode=common_kwargs["mode"],
            profile=common_kwargs["profile"],
            project_root=common_kwargs["project_root"],
            cascade_variants=runtime_overrides.get("cascade_variants_for_comparison"),
            write_sql_output=runtime_overrides.get("gold_comparison_write_sql_output"),
        )

    raise ValueError(f"Unsupported stage_name: {stage_name}")


def run_full_pipeline(
    *,
    config_root: Optional[Path] = None,
    dataset: str = "pump",
    mode: str = "train",
    profile: str = "default",
    project_root: Optional[Path] = None,
    start_stage: Optional[str] = None,
    end_stage: Optional[str] = None,
    cascade_variants: Optional[List[str]] = None,
    stop_on_error: bool = True,
    # Bronze overrides
    run_id: Optional[str] = None,
    asset_id: Optional[str] = None,
    raw_file_name: Optional[str] = None,
    dataset_name_argument: Optional[str] = None,
    bronze_source_mode: Optional[str] = None,
    dataset_name_postgres: Optional[str] = None,
    postgres_source_table_name: Optional[str] = None,
    postgres_source_table_dataset_map: Optional[Dict[str, str]] = None,
    # Silver / Gold shared artifact overrides
    bronze_train_data_file_name: Optional[str] = None,
    silver_train_data_file_name: Optional[str] = None,
    feature_registry_file_name: Optional[str] = None,
    dropped_parquet_file_name: Optional[str] = None,
    join_key: Optional[str] = None,
    # Silver EDA overrides
    max_onsets_to_use: Optional[int] = None,
    onset_pre_window: Optional[int] = None,
    onset_post_window: Optional[int] = None,
    top_feature_count: Optional[int] = None,
    # Gold preprocessing overrides
    gold_preprocessed_file_name: Optional[str] = None,
    gold_preprocessed_scaled_file_name: Optional[str] = None,
    gold_fit_file_name: Optional[str] = None,
    gold_train_file_name: Optional[str] = None,
    gold_test_file_name: Optional[str] = None,
    train_fraction: Optional[float] = None,
    imputation_method: Optional[str] = None,
    scaler_kind: Optional[str] = None,
    # Baseline overrides
    baseline_model_file_name: Optional[str] = None,
    baseline_summary_file_name: Optional[str] = None,
    baseline_threshold_percentile: Optional[float] = None,
    baseline_n_estimators: Optional[int] = None,
    baseline_contamination: Optional[str] = None,
    baseline_max_samples: Optional[str] = None,
    baseline_max_features: Optional[float] = None,
    baseline_bootstrap: Optional[bool] = None,
    baseline_random_state: Optional[int] = None,
    baseline_n_jobs: Optional[int] = None,
    # SQL toggles
    silver_preeda_write_sql_output: Optional[bool] = None,
    gold_preprocessing_write_sql_output: Optional[bool] = None,
    gold_baseline_write_sql_output: Optional[bool] = None,
    gold_cascade_write_sql_output: Optional[bool] = None,
    gold_comparison_write_sql_output: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run the full pipeline or a bounded subset of stages.
    """
    paths = get_paths()
    config_root = config_root or paths.configs
    project_root = project_root or paths.root

    logger = _initialize_orchestrator_logger(paths.logs)

    cascade_variants = cascade_variants or ["default", "tuned", "improved"]

    common_kwargs = {
        "config_root": config_root,
        "dataset": dataset,
        "mode": mode,
        "profile": profile,
        "project_root": project_root,
    }

    runtime_overrides: Dict[str, Any] = {
        "run_id": run_id,
        "asset_id": asset_id,
        "raw_file_name": raw_file_name,
        "dataset_name_argument": dataset_name_argument,
        "bronze_source_mode": bronze_source_mode,
        "dataset_name_postgres": dataset_name_postgres,
        "postgres_source_table_name": postgres_source_table_name,
        "postgres_source_table_dataset_map": postgres_source_table_dataset_map,
        "bronze_train_data_file_name": bronze_train_data_file_name,
        "silver_train_data_file_name": silver_train_data_file_name,
        "feature_registry_file_name": feature_registry_file_name,
        "dropped_parquet_file_name": dropped_parquet_file_name,
        "join_key": join_key,
        "max_onsets_to_use": max_onsets_to_use,
        "onset_pre_window": onset_pre_window,
        "onset_post_window": onset_post_window,
        "top_feature_count": top_feature_count,
        "gold_preprocessed_file_name": gold_preprocessed_file_name,
        "gold_preprocessed_scaled_file_name": gold_preprocessed_scaled_file_name,
        "gold_fit_file_name": gold_fit_file_name,
        "gold_train_file_name": gold_train_file_name,
        "gold_test_file_name": gold_test_file_name,
        "train_fraction": train_fraction,
        "imputation_method": imputation_method,
        "scaler_kind": scaler_kind,
        "baseline_model_file_name": baseline_model_file_name,
        "baseline_summary_file_name": baseline_summary_file_name,
        "baseline_threshold_percentile": baseline_threshold_percentile,
        "baseline_n_estimators": baseline_n_estimators,
        "baseline_contamination": baseline_contamination,
        "baseline_max_samples": baseline_max_samples,
        "baseline_max_features": baseline_max_features,
        "baseline_bootstrap": baseline_bootstrap,
        "baseline_random_state": baseline_random_state,
        "baseline_n_jobs": baseline_n_jobs,
        "silver_preeda_write_sql_output": silver_preeda_write_sql_output,
        "gold_preprocessing_write_sql_output": gold_preprocessing_write_sql_output,
        "gold_baseline_write_sql_output": gold_baseline_write_sql_output,
        "gold_cascade_write_sql_output": gold_cascade_write_sql_output,
        "gold_comparison_write_sql_output": gold_comparison_write_sql_output,
        "cascade_variants_for_comparison": cascade_variants,
    }

    stages_to_run = _resolve_stages_to_run(
        start_stage=start_stage,
        end_stage=end_stage,
        cascade_variants=cascade_variants,
    )

    logger.info("Dataset=%s | Mode=%s | Profile=%s", dataset, mode, profile)
    logger.info("Stages to run: %s", stages_to_run)

    run_started_at = time.time()
    stage_results: Dict[str, Any] = {}
    stage_timings: Dict[str, float] = {}
    failed_stage: Optional[str] = None
    failed_error: Optional[str] = None

    for stage_name in stages_to_run:
        stage_started_at = time.time()
        try:
            result = _call_stage(
                stage_name=stage_name,
                common_kwargs=common_kwargs,
                runtime_overrides=runtime_overrides,
                logger=logger,
            )
            elapsed = time.time() - stage_started_at

            stage_results[stage_name] = result
            stage_timings[stage_name] = round(elapsed, 3)

            logger.info(
                "Completed stage: %s | elapsed_seconds=%.3f | status=%s",
                stage_name,
                elapsed,
                result.get("status"),
            )

        except Exception as exc:
            elapsed = time.time() - stage_started_at
            stage_timings[stage_name] = round(elapsed, 3)
            failed_stage = stage_name
            failed_error = f"{type(exc).__name__}: {exc}"

            logger.exception("Stage failed: %s", stage_name)
            stage_results[stage_name] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }

            if stop_on_error:
                break

    total_elapsed = time.time() - run_started_at

    summary: Dict[str, Any] = {
        "status": "failed" if failed_stage is not None else "success",
        "dataset": dataset,
        "mode": mode,
        "profile": profile,
        "stages_requested": stages_to_run,
        "stages_completed": [
            stage_name
            for stage_name in stages_to_run
            if stage_results.get(stage_name, {}).get("status") == "success"
        ],
        "stage_timings_seconds": stage_timings,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "failed_stage": failed_stage,
        "failed_error": failed_error,
        "stage_results": stage_results,
    }

    summary_output_path = paths.logs / "pipeline_orchestrator_summary.json"
    with open(summary_output_path, "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2, default=str)

    logger.info("Pipeline orchestrator finished with status=%s", summary["status"])
    logger.info("Summary written to: %s", summary_output_path)

    return summary


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


def _parse_optional_json_dict(value: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Parse optional CLI JSON dict.
    """
    if value is None or str(value).strip() == "":
        return None

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object/dict.")

    return {str(key): str(val) for key, val in parsed.items()}


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for full pipeline orchestrator.
    """
    parser = argparse.ArgumentParser(description="Run the full capstone pipeline orchestrator.")

    parser.add_argument("--dataset", default="pump")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--project-root", default=None)

    parser.add_argument("--start-stage", default=None, choices=STAGE_ORDER)
    parser.add_argument("--end-stage", default=None, choices=STAGE_ORDER)
    parser.add_argument(
        "--cascade-variants",
        nargs="*",
        default=["default", "tuned", "improved"],
        choices=["default", "tuned", "improved"],
    )
    parser.add_argument("--stop-on-error", default="true")

    # Bronze
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--raw-file-name", default=None)
    parser.add_argument("--dataset-name-argument", default=None)
    parser.add_argument("--bronze-source-mode", default=None, choices=["file", "postgres_handoff"])
    parser.add_argument("--dataset-name-postgres", default=None)
    parser.add_argument("--postgres-source-table-name", default=None)
    parser.add_argument("--postgres-source-table-dataset-map", default=None)

    # Shared artifact names
    parser.add_argument("--bronze-train-data-file-name", default=None)
    parser.add_argument("--silver-train-data-file-name", default=None)
    parser.add_argument("--feature-registry-file-name", default=None)
    parser.add_argument("--dropped-parquet-file-name", default=None)
    parser.add_argument("--join-key", default=None)

    # Silver EDA
    parser.add_argument("--max-onsets-to-use", type=int, default=None)
    parser.add_argument("--onset-pre-window", type=int, default=None)
    parser.add_argument("--onset-post-window", type=int, default=None)
    parser.add_argument("--top-feature-count", type=int, default=None)

    # Gold preprocessing
    parser.add_argument("--gold-preprocessed-file-name", default=None)
    parser.add_argument("--gold-preprocessed-scaled-file-name", default=None)
    parser.add_argument("--gold-fit-file-name", default=None)
    parser.add_argument("--gold-train-file-name", default=None)
    parser.add_argument("--gold-test-file-name", default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--imputation-method", default=None)
    parser.add_argument("--scaler-kind", default=None)

    # Baseline
    parser.add_argument("--baseline-model-file-name", default=None)
    parser.add_argument("--baseline-summary-file-name", default=None)
    parser.add_argument("--baseline-threshold-percentile", type=float, default=None)
    parser.add_argument("--baseline-n-estimators", type=int, default=None)
    parser.add_argument("--baseline-contamination", default=None)
    parser.add_argument("--baseline-max-samples", default=None)
    parser.add_argument("--baseline-max-features", type=float, default=None)
    parser.add_argument("--baseline-bootstrap", default=None)
    parser.add_argument("--baseline-random-state", type=int, default=None)
    parser.add_argument("--baseline-n-jobs", type=int, default=None)

    # SQL flags
    parser.add_argument("--silver-preeda-write-sql-output", default=None)
    parser.add_argument("--gold-preprocessing-write-sql-output", default=None)
    parser.add_argument("--gold-baseline-write-sql-output", default=None)
    parser.add_argument("--gold-cascade-write-sql-output", default=None)
    parser.add_argument("--gold-comparison-write-sql-output", default=None)

    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    result = run_full_pipeline(
        config_root=Path(args.config_root) if args.config_root else None,
        dataset=args.dataset,
        mode=args.mode,
        profile=args.profile,
        project_root=Path(args.project_root) if args.project_root else None,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        cascade_variants=args.cascade_variants,
        stop_on_error=_parse_optional_bool(args.stop_on_error) if args.stop_on_error is not None else True,
        run_id=args.run_id,
        asset_id=args.asset_id,
        raw_file_name=args.raw_file_name,
        dataset_name_argument=args.dataset_name_argument,
        bronze_source_mode=args.bronze_source_mode,
        dataset_name_postgres=args.dataset_name_postgres,
        postgres_source_table_name=args.postgres_source_table_name,
        postgres_source_table_dataset_map=_parse_optional_json_dict(args.postgres_source_table_dataset_map),
        bronze_train_data_file_name=args.bronze_train_data_file_name,
        silver_train_data_file_name=args.silver_train_data_file_name,
        feature_registry_file_name=args.feature_registry_file_name,
        dropped_parquet_file_name=args.dropped_parquet_file_name,
        join_key=args.join_key,
        max_onsets_to_use=args.max_onsets_to_use,
        onset_pre_window=args.onset_pre_window,
        onset_post_window=args.onset_post_window,
        top_feature_count=args.top_feature_count,
        gold_preprocessed_file_name=args.gold_preprocessed_file_name,
        gold_preprocessed_scaled_file_name=args.gold_preprocessed_scaled_file_name,
        gold_fit_file_name=args.gold_fit_file_name,
        gold_train_file_name=args.gold_train_file_name,
        gold_test_file_name=args.gold_test_file_name,
        train_fraction=args.train_fraction,
        imputation_method=args.imputation_method,
        scaler_kind=args.scaler_kind,
        baseline_model_file_name=args.baseline_model_file_name,
        baseline_summary_file_name=args.baseline_summary_file_name,
        baseline_threshold_percentile=args.baseline_threshold_percentile,
        baseline_n_estimators=args.baseline_n_estimators,
        baseline_contamination=args.baseline_contamination,
        baseline_max_samples=args.baseline_max_samples,
        baseline_max_features=args.baseline_max_features,
        baseline_bootstrap=_parse_optional_bool(args.baseline_bootstrap),
        baseline_random_state=args.baseline_random_state,
        baseline_n_jobs=args.baseline_n_jobs,
        silver_preeda_write_sql_output=_parse_optional_bool(args.silver_preeda_write_sql_output),
        gold_preprocessing_write_sql_output=_parse_optional_bool(args.gold_preprocessing_write_sql_output),
        gold_baseline_write_sql_output=_parse_optional_bool(args.gold_baseline_write_sql_output),
        gold_cascade_write_sql_output=_parse_optional_bool(args.gold_cascade_write_sql_output),
        gold_comparison_write_sql_output=_parse_optional_bool(args.gold_comparison_write_sql_output),
    )

    print(json.dumps(result, indent=2, default=str))

'''
#Example commands

#Full run:

python -m pipelines.run_full_pipeline \
  --dataset pump \
  --mode train \
  --profile default

Start at Silver:

python -m pipelines.run_full_pipeline \
  --dataset pump \
  --mode train \
  --profile default \
  --start-stage silver_preeda

#Gold-only branch:

python -m pipelines.run_full_pipeline \
  --dataset pump \
  --mode train \
  --profile default \
  --start-stage gold_preprocessing \
  --end-stage gold_comparison

#Only baseline + improved cascade + comparison:

python -m pipelines.run_full_pipeline \
  --dataset pump \
  --mode train \
  --profile default \
  --start-stage gold_baseline \
  --cascade-variants improved

#Bronze from file with overrides:

python -m pipelines.run_full_pipeline \
  --dataset pump \
  --mode train \
  --profile default \
  --start-stage bronze \
  --end-stage bronze \
  --run-id synthetic_run_001 \
  --asset-id pump_01 \
  --raw-file-name pump_synthetic_batch.parquet \
  --dataset-name-argument pump

#Bronze from postgres handoff:

python -m pipelines.run_full_pipeline \
  --dataset pump \
  --mode train \
  --profile default \
  --start-stage bronze \
  --end-stage bronze \
  --bronze-source-mode postgres_handoff \
  --run-id synthetic_run_001 \
  --dataset-name-postgres pump \
  --postgres-source-table-name bronze_observations_input_stage \
  --postgres-source-table-dataset-map '{"bronze_observations_input_stage":"pump"}'

#Silver EDA with dropped parquet:

python -m pipelines.run_full_pipeline \
  --dataset pump \
  --mode train \
  --profile default \
  --start-stage silver_eda \
  --end-stage silver_eda \
  --dropped-parquet-file-name pump__silver__dropped_features.parquet \
  --join-key meta__record_id \
  --max-onsets-to-use 15 \
  --top-feature-count 10

#Full run with SQL writes enabled on selected stages:

python -m pipelines.run_full_pipeline \
  --dataset pump \
  --mode train \
  --profile default \
  --silver-preeda-write-sql-output true \
  --gold-preprocessing-write-sql-output true \
  --gold-baseline-write-sql-output true \
  --gold-cascade-write-sql-output true \
  --gold-comparison-write-sql-output true

'''