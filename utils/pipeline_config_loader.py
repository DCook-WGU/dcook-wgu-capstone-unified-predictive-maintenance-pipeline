from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import os
import re

import yaml


class ConfigError(Exception):
    pass

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


@dataclass(frozen=True)
class LoadedConfig:
    """Container for the resolved pipeline config."""

    data: dict[str, Any]
    config_hash: str
    source_files: list[str]

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Config file must contain a mapping at top level: {path}")
    return payload

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _flatten_for_templates(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        compound = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(_flatten_for_templates(value, compound))
        else:
            flat[compound] = value
            flat[key] = value
    return flat

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


_TEMPLATE_PATTERN = re.compile(r"\{([A-Za-z0-9_.-]+)\}")

def _render_template_string(template: str, context: Mapping[str, Any]) -> str:
    """
    Render template placeholders using literal key matching.

    This supports flattened keys like:
        {paths.gold}
        {filenames.truth}
        {runtime.mode}

    Missing keys are left unchanged.
    """
    def replace(match: re.Match) -> str:
        key = match.group(1)

        if key in context:
            value = context[key]
            return str(value)

        return match.group(0)

    return _TEMPLATE_PATTERN.sub(replace, template)


def _render_templates(obj: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(obj, str):
        return _render_template_string(obj, context)

    if isinstance(obj, list):
        return [_render_templates(item, context) for item in obj]

    if isinstance(obj, dict):
        return {
            key: _render_templates(value, context)
            for key, value in obj.items()
        }

    return obj

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def _build_filename_map(cfg: dict[str, Any]) -> dict[str, str]:
    dataset_name = cfg["dataset"]["name"]

    bronze_version = cfg["versions"]["bronze"]
    silver_version = cfg["versions"]["silver"]
    silver_eda_version = cfg["versions"]["silver_eda"]
    gold_version = cfg["versions"]["gold"]

    filenames = {
        # Dataset Files
        "raw_file_name": cfg["dataset"]["raw_file_name"],
        "bronze_train_file_name": f"{dataset_name}__bronze__train.parquet",
        "silver_train_file_name": f"{dataset_name}__silver__train.parquet",
        "gold_preprocessed_file_name": f"{dataset_name}__gold__preprocessed.parquet",
        "gold_preprocessed_prescaled_file_name": f"{dataset_name}__gold__preprocessed_prescaled.parquet",
        "gold_preprocessed_scaled_file_name": f"{dataset_name}__gold__preprocessed_scaled.parquet",
        "gold_fit_file_name": f"{dataset_name}__gold__fit_normal_only.parquet",
        "gold_train_file_name": f"{dataset_name}__gold__train.parquet",
        "gold_test_file_name": f"{dataset_name}__gold__test.parquet",
        "silver_preeda_dropped_sensors_file_name": f"{dataset_name}__silver_preeda__dropped_sensors.parquet",

        # EDA Files
        "feature_registry_file_name": f"{dataset_name}__silver__feature_registry.json",
        "impute_recommendation_file_name": "imputation__recommendation.json",
        "stage1_features_file_name": f"{dataset_name}__gold__stage1_features.json",
        "stage2_features_file_name": f"{dataset_name}__gold__stage2_features.json",
        "stage3_primary_file_name": f"{dataset_name}__gold__stage3_primary_rule_sensors.json",
        "stage3_secondary_file_name": f"{dataset_name}__gold__stage3_secondary_rule_sensors.json",


        # Plots
        
        # Baseline
        "baseline_results_file_name_csv": f"{dataset_name}__gold__baseline_results.csv",
        "baseline_results_file_name_pickle": f"{dataset_name}__gold__baseline_results.pkl",
        "baseline_model_file_name": f"{dataset_name}__gold__baseline_isolation_forest.joblib",
        "baseline_thresholds_file_name": f"{dataset_name}__gold__baseline_thresholds.json",
        "baseline_summary_file_name": f"{dataset_name}__gold__baseline_summary.json",
        "baseline_metadata_file_name": f"{dataset_name}__gold__baseline_metadata.json",

        # Cascade - Defaults
        "cascade_defaults_results_file_name_csv": f"{dataset_name}__gold__cascade_defaults_results.csv",
        "cascade_defaults_results_file_name_pickle": f"{dataset_name}__gold__cascade_defaults_results.pkl",
        "cascade_defaults_thresholds_file_name": f"{dataset_name}__gold__cascade_defaults_thresholds.json",
        "cascade_defaults_summary_file_name": f"{dataset_name}__gold__cascade_defaults_summary.json",
        "cascade_defaults_metadata_file_name": f"{dataset_name}__gold__cascade_defaults_metadata.json",
        "cascade_defaults_reference_profile_file_name": f"{dataset_name}__gold__cascade_defaults_reference_profile.csv",
        "cascade_defaults_stage1_model_file_name": f"{dataset_name}__gold__cascade_defaults_stage1_isolation_forest.joblib",
        "cascade_defaults_stage2_model_file_name": f"{dataset_name}__gold__cascade_defaults_stage2_isolation_forest.joblib",

        # Cascade - Tuned
        "cascade_tuned_results_file_name_csv": f"{dataset_name}__gold__cascade_tuned_results.csv",
        "cascade_tuned_results_file_name_pickle": f"{dataset_name}__gold__cascade_tuned_results.pkl",
        "cascade_tuned_thresholds_file_name": f"{dataset_name}__gold__cascade_tuned_thresholds.json",
        "cascade_tuned_summary_file_name": f"{dataset_name}__gold__cascade_tuned_summary.json",
        "cascade_tuned_metadata_file_name": f"{dataset_name}__gold__cascade_tuned_metadata.json",
        "cascade_tuned_reference_profile_file_name": f"{dataset_name}__gold__cascade_tuned_reference_profile.csv",
        "cascade_tuned_stage1_model_file_name": f"{dataset_name}__gold__cascade_tuned_stage1_isolation_forest.joblib",
        "cascade_tuned_stage2_model_file_name": f"{dataset_name}__gold__cascade_tuned_stage2_isolation_forest.joblib",

        # Cascade - Stage 3 Improved
        "cascade_stage3_improved_results_file_name_csv": f"{dataset_name}__gold__cascade_stage3_improved_results.csv",
        "cascade_stage3_improved_results_file_name_pickle": f"{dataset_name}__gold__cascade_stage3_improved_results.pkl",
        "cascade_stage3_improved_thresholds_file_name": f"{dataset_name}__gold__cascade_stage3_improved_thresholds.json",
        "cascade_stage3_improved_summary_file_name": f"{dataset_name}__gold__cascade_stage3_improved_summary.json",
        "cascade_stage3_improved_metadata_file_name": f"{dataset_name}__gold__cascade_stage3_improved_metadata.json",
        "cascade_stage3_improved_reference_profile_file_name": f"{dataset_name}__gold__cascade_stage3_improved_reference_profile.csv",
        "cascade_stage3_improved_stage1_model_file_name": f"{dataset_name}__gold__cascade_stage3_improved_stage1_isolation_forest.joblib",
        "cascade_stage3_improved_stage2_model_file_name": f"{dataset_name}__gold__cascade_stage3_improved_stage2_isolation_forest.joblib",


        # Comparisons 
        "comparison_file_name": f"{dataset_name}__gold__comparison__model_comparison.csv",
        "model_comparison_file_name": f"{dataset_name}__gold__model_comparison.csv",
        "model_comparison_summary_file_name": f"{dataset_name}__gold__model_comparison_summary.json",
        "preprocessing_summary_file_name": f"{dataset_name}__gold__preprocessing_summary.json",
        "preprocessing_metadata_file_name": f"{dataset_name}__gold__preprocessing_metadata.json",
        "reference_profile_file_name": f"{dataset_name}__gold__reference_profile.csv",
        "comparison_plot_with_test_alerts_file_name": f"{dataset_name}__gold__comparison__2panel_test_alerts_and_metrics.png",
        
        # Ledgers
        "bronze_ledger_file_name": f"ledger__{dataset_name}__bronze_preprocessing.json",
        "silver_ledger_file_name": f"ledger__{dataset_name}__silver_preeda.json",
        "silver_eda_ledger_file_name": f"ledger__{dataset_name}__silver_eda.json",
        "gold_preprocessing_ledger_file_name": f"ledger__{dataset_name}__gold_preprocessing.json",
        "gold_baseline_ledger_file_name": f"ledger__{dataset_name}__gold_baseline_modeling.json",
        "gold_cascade_defaults_ledger_file_name": f"ledger__{dataset_name}__gold_cascade_defaults_modeling.json",
        "gold_cascade_tuned_ledger_file_name": f"ledger__{dataset_name}__gold_cascade_tuned_modeling.json",
        "gold_cascade_stage3_imporved_ledger_file_name": f"ledger__{dataset_name}__gold_cascade_stage3_improved_modeling.json",
        "gold_comparison_ledger_file_name": f"ledger__{dataset_name}__gold_comparison.json",
        
        # Versions 
        "bronze_version": bronze_version,
        "silver_version": silver_version,
        "silver_eda_version": silver_eda_version,
        "gold_version": gold_version,
    }
    return filenames

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def _build_path_map(project_root: Path, cfg: dict[str, Any], filenames: dict[str, str]) -> dict[str, str]:
    roots = cfg["paths"]
    data_root = project_root / roots["data_dir"]
    artifacts_root = project_root / roots["artifacts_dir"]
    models_root = project_root / roots["models_dir"]
    logs_root = project_root / roots["logs_dir"]
    wandb_root = project_root / roots["wandb_dir"]
    pipelines_root = project_root / roots["pipelines_dir"]

    dataset_name = cfg["dataset"]["name"]
    raw_dataset_subdir = cfg["dataset"]["raw_dataset_subdir"]

    path_map = {

        # Core Folders
        "project_root": str(project_root),
        "data_root": str(data_root),
        "artifacts_root": str(artifacts_root),
        "models_root": str(models_root),
        "logs_root": str(logs_root),
        "wandb_root": str(wandb_root),
        "pipelines_root": str(pipelines_root), 


        # Data Folders
        "data_raw_dir": str(data_root / roots["raw_subdir"]),
        "data_bronze_dir": str(data_root / roots["bronze_subdir"]),
        "data_bronze_train_dir": str(data_root / roots["bronze_train_subdir"]),
        "data_bronze_test_dir": str(data_root / roots["bronze_test_subdir"]),
        "data_silver_dir": str(data_root / roots["silver_subdir"]),
        "data_silver_train_dir": str(data_root / roots["silver_train_subdir"]),
        "data_silver_test_dir": str(data_root / roots["silver_test_subdir"]),
        "data_gold_dir": str(data_root / roots["gold_subdir"]),

        # Pipeline Folders:
        "piplines_bronze_dir": str(pipelines_root / roots["bronze_dir"]),
        "piplines_silver_dir": str(pipelines_root / roots["silver_dir"]), 
        "piplines_gold_dir": str(pipelines_root / roots["gold_dir"]), 

        # Truths
        "truths_dir": str(artifacts_root / "truths"),
        "truth_index_path": str(artifacts_root / "truths" / "truth_index.jsonl"),

        # Artifacts
        "bronze_artifacts_dir": str(artifacts_root / "bronze" / dataset_name),
        "silver_artifacts_dir": str(artifacts_root / "silver" / dataset_name),
        "silver_eda_artifacts_dir": str(artifacts_root / "silver_eda" / dataset_name),
        "gold_artifacts_dir": str(artifacts_root / "gold" / dataset_name),

        # Models 
        "dataset_models_dir": str(models_root / dataset_name),

        # Datasets 
        "raw_file_path": str(data_root / roots["raw_subdir"] / raw_dataset_subdir / filenames["raw_file_name"]),
        "bronze_train_data_path": str(data_root / roots["bronze_train_subdir"] / filenames["bronze_train_file_name"]),
        "silver_train_data_path": str(data_root / roots["silver_train_subdir"] / filenames["silver_train_file_name"]),
        "gold_preprocessed_data_path": str(data_root / roots["gold_subdir"] / filenames["gold_preprocessed_file_name"]),
        "gold_preprocessed_prescaled_data_path": str(data_root / roots["gold_subdir"] / filenames["gold_preprocessed_prescaled_file_name"]),
        "gold_preprocessed_scaled_data_path": str(data_root / roots["gold_subdir"] / filenames["gold_preprocessed_scaled_file_name"]),
        "gold_fit_data_path": str(data_root / roots["gold_subdir"] / filenames["gold_fit_file_name"]),
        "gold_train_data_path": str(data_root / roots["gold_subdir"] / filenames["gold_train_file_name"]),
        "gold_test_data_path": str(data_root / roots["gold_subdir"] / filenames["gold_test_file_name"]),
        "silver_preeda_dropped_sensors_data_path": str(artifacts_root / "silver" / dataset_name / filenames["silver_preeda_dropped_sensors_file_name"]),

        # EDA
        "feature_registry_path": str(artifacts_root / "silver" / dataset_name / filenames["feature_registry_file_name"]),
        "impute_recommendation_path": str(artifacts_root / "silver_eda" / dataset_name / filenames["impute_recommendation_file_name"]),
        "stage1_features_path": str(artifacts_root / "gold" / dataset_name / filenames["stage1_features_file_name"]),
        "stage2_features_path": str(artifacts_root / "gold" / dataset_name / filenames["stage2_features_file_name"]),
        "stage3_primary_path": str(artifacts_root / "gold" / dataset_name / filenames["stage3_primary_file_name"]),
        "stage3_secondary_path": str(artifacts_root / "gold" / dataset_name / filenames["stage3_secondary_file_name"]),

        # Baseline
        "baseline_results_path_csv": str(artifacts_root / "gold" / dataset_name / filenames["baseline_results_file_name_csv"]),
        "baseline_results_path_pickle": str(artifacts_root / "gold" / dataset_name / filenames["baseline_results_file_name_pickle"]),
        "baseline_model_artifact_path": str(artifacts_root / "gold" / dataset_name / filenames["baseline_model_file_name"]),
        "baseline_models_path": str(models_root / dataset_name / filenames["baseline_model_file_name"]),
        "baseline_thresholds_path": str(artifacts_root / "gold" / dataset_name / filenames["baseline_thresholds_file_name"]),
        "baseline_summary_path": str(artifacts_root / "gold" / dataset_name / filenames["baseline_summary_file_name"]),
        "baseline_metadata_path": str(artifacts_root / "gold" / dataset_name / filenames["baseline_metadata_file_name"]),

        # Cascade - Defaults
        "cascade_defaults_results_path_csv": str(artifacts_root / "gold" / dataset_name / filenames["cascade_defaults_results_file_name_csv"]),
        "cascade_defaults_results_path_pickle": str(artifacts_root / "gold" / dataset_name / filenames["cascade_defaults_results_file_name_pickle"]),
        "cascade_defaults_thresholds_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_defaults_thresholds_file_name"]),
        "cascade_defaults_summary_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_defaults_summary_file_name"]),
        "cascade_defaults_metadata_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_defaults_metadata_file_name"]),
        "cascade_defaults_reference_profile_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_defaults_reference_profile_file_name"]),
        "cascade_defaults_stage1_model_artifact_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_defaults_stage1_model_file_name"]),
        "cascade_defaults_stage2_model_artifact_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_defaults_stage2_model_file_name"]),
        "cascade_defaults_stage1_models_path": str(models_root / dataset_name / filenames["cascade_defaults_stage1_model_file_name"]),
        "cascade_defaults_stage2_models_path": str(models_root / dataset_name / filenames["cascade_defaults_stage2_model_file_name"]),

        # Cascade - Tuned 
        "cascade_tuned_results_path_csv": str(artifacts_root / "gold" / dataset_name / filenames["cascade_tuned_results_file_name_csv"]),
        "cascade_tuned_results_path_pickle": str(artifacts_root / "gold" / dataset_name / filenames["cascade_tuned_results_file_name_pickle"]),
        "cascade_tuned_thresholds_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_tuned_thresholds_file_name"]),
        "cascade_tuned_summary_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_tuned_summary_file_name"]),
        "cascade_tuned_metadata_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_tuned_metadata_file_name"]),
        "cascade_tuned_reference_profile_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_tuned_reference_profile_file_name"]),
        "cascade_tuned_stage1_model_artifact_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_tuned_stage1_model_file_name"]),
        "cascade_tuned_stage2_model_artifact_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_tuned_stage2_model_file_name"]),
        "cascade_tuned_stage1_models_path": str(models_root / dataset_name / filenames["cascade_tuned_stage1_model_file_name"]),
        "cascade_tuned_stage2_models_path": str(models_root / dataset_name / filenames["cascade_tuned_stage2_model_file_name"]),

        # Cascade - Stage3 Improved 
        "cascade_stage3_improved_results_path_csv": str(artifacts_root / "gold" / dataset_name / filenames["cascade_stage3_improved_results_file_name_csv"]),
        "cascade_stage3_improved_results_path_pickle": str(artifacts_root / "gold" / dataset_name / filenames["cascade_stage3_improved_results_file_name_pickle"]),
        "cascade_stage3_improved_thresholds_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_stage3_improved_thresholds_file_name"]),
        "cascade_stage3_improved_summary_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_stage3_improved_summary_file_name"]),
        "cascade_stage3_improved_metadata_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_stage3_improved_metadata_file_name"]),
        "cascade_stage3_improved_reference_profile_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_stage3_improved_reference_profile_file_name"]),
        "cascade_stage3_improved_stage1_model_artifact_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_stage3_improved_stage1_model_file_name"]),
        "cascade_stage3_improved_stage2_model_artifact_path": str(artifacts_root / "gold" / dataset_name / filenames["cascade_stage3_improved_stage2_model_file_name"]),
        "cascade_stage3_improved_stage1_models_path": str(models_root / dataset_name / filenames["cascade_stage3_improved_stage1_model_file_name"]),
        "cascade_stage3_improved_stage2_models_path": str(models_root / dataset_name / filenames["cascade_stage3_improved_stage2_model_file_name"]),


        # Comparision
        "comparison_path": str(artifacts_root / "gold" / dataset_name / filenames["comparison_file_name"]),
        "model_comparison_path": str(artifacts_root / "gold" / dataset_name / filenames["model_comparison_file_name"]),
        "model_comparison_summary_path": str(artifacts_root / "gold" / dataset_name / filenames["model_comparison_summary_file_name"]),
        "preprocessing_summary_path": str(artifacts_root / "gold" / dataset_name / filenames["preprocessing_summary_file_name"]),
        "preprocessing_metadata_path": str(artifacts_root / "gold" / dataset_name / filenames["preprocessing_metadata_file_name"]),
        "reference_profile_path": str(artifacts_root / "gold" / dataset_name / filenames["reference_profile_file_name"]),
        "comparison_plot_with_test_alerts_path": str(artifacts_root / "gold" / dataset_name / filenames["comparison_plot_with_test_alerts_file_name"]),
        
        # Loggs
        "bronze_log_path": str(logs_root / "bronze.log"),
        "silver_log_path": str(logs_root / "silver.log"),
        "silver_eda_log_path": str(logs_root / "silver_eda.log"),
        "gold_preprocessing_log_path": str(logs_root / "gold_preprocessing.log"),
        "gold_baseline_log_path": str(logs_root / "gold_modeling_baseline.log"),
        "gold_cascade_log_path": str(logs_root / "gold_modeling_cascade.log"),
        "gold_cascade_log_path": str(logs_root / "gold_modeling_cascade_defaulfts.log"),
        "gold_cascade_log_path": str(logs_root / "gold_modeling_cascade_tuned.log"),
        "gold_cascade_log_path": str(logs_root / "gold_modeling_cascade_stage3_improved.log"),
        "gold_comparison_log_path": str(logs_root / "gold_model_comparison.log"),
    }
    return path_map

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def _normalize_mode_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    mode = cfg["runtime"]["mode"]
    overrides = deepcopy(cfg.get("mode_behavior", {}).get(mode, {}))
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def _resolve_config_file(path: Path) -> Path:
    candidates: list[Path] = []

    if path.suffix:
        candidates.append(path)
    else:
        candidates.extend(
            [
                path.with_suffix(".yaml"),
                path.with_suffix(".yml"),
                path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise ConfigError(f"Config file not found. Tried: {tried}")

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def load_pipeline_config(
    *,
    config_root: str | Path,
    stage: str,
    dataset: str,
    mode: str,
    profile: str = "default",
    project_root: str | Path | None = None,
) -> LoadedConfig:
    """
    Merge config fragments in this order:
    1. base.yaml
    2. datasets/<dataset>.yaml
    3. modes/<mode>.yaml | modes/<mode>
    4. stages/<stage>.yaml | stages/<stage>

    The merged config is then template-rendered and enriched with derived filenames and paths.
    """
    config_root = Path(config_root).resolve()
    project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()

    sources = [
        _resolve_config_file(config_root / "base.yaml"),
        _resolve_config_file(config_root / "datasets" / dataset),
        _resolve_config_file(config_root / "modes" / mode),
        _resolve_config_file(config_root / "stages" / stage),
    ]

    merged: dict[str, Any] = {}
    source_files: list[str] = []

    for source in sources:
        fragment = _read_yaml(source)
        merged = _deep_merge(merged, fragment)
        source_files.append(str(source))

    merged.setdefault("runtime", {})
    merged["runtime"]["stage"] = stage
    merged["runtime"]["dataset"] = dataset
    merged["runtime"]["mode"] = mode
    merged["runtime"]["profile"] = profile

    merged = _normalize_mode_overrides(merged)

    flat_before = _flatten_for_templates(merged)
    merged = _render_templates(merged, flat_before)

    filenames = _build_filename_map(merged)
    merged["filenames"] = filenames

    path_map = _build_path_map(project_root, merged, filenames)
    merged["resolved_paths"] = path_map

    flat_after = _flatten_for_templates(merged)
    merged = _render_templates(merged, flat_after)

    merged["config_meta"] = {
        "source_files": source_files,
        "project_root": str(project_root),
    }

    cfg_bytes = json.dumps(merged, sort_keys=True, default=str).encode("utf-8")
    config_hash = sha256(cfg_bytes).hexdigest()
    merged["config_meta"]["config_hash"] = config_hash

    return LoadedConfig(data=merged, config_hash=config_hash, source_files=source_files)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def export_config_snapshot(config: Mapping[str, Any], destination: str | Path) -> Path:
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False)
    return destination


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def build_truth_config_block(config: Mapping[str, Any]) -> dict[str, Any]:
    """Small config payload to embed in your truth record."""
    return {
        "config_hash": config["config_meta"]["config_hash"],
        "source_files": config["config_meta"]["source_files"],
        "runtime": config["runtime"],
        "versions": config["versions"],
        "wandb": config.get("wandb", {}),
        "execution": config.get("execution", {}),
        "lineage": config.get("lineage", {}),
        "stage_params": config.get(config["runtime"]["stage"], {}),
    }

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def set_wandb_dir_from_config(config: Mapping[str, Any]) -> Path:
    wandb_root = Path(config["resolved_paths"]["wandb_root"]).resolve()
    wandb_root.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_root)
    return wandb_root

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
