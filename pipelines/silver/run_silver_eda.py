"""
pipelines/silver/run_silver_eda.py

Silver EDA pipeline runner for the capstone project.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import wandb

from utils.paths import get_paths
from utils.file_io import load_data, load_json, save_json
from utils.logging_setup import configure_logging, log_layer_paths
from utils.wandb_utils import finalize_wandb_stage
from utils.truths import (
    extract_truth_hash,
    initialize_layer_truth,
    update_truth_section,
    build_truth_record,
    save_truth_record,
    append_truth_index,
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
from utils.ledger import Ledger

from utils.pipeline.silver_eda_status import (
    resolve_state_column_from_truth,
    build_state_col_synth,
    build_episode_status_payload_and_tables,
    build_status_distribution_tables,
    pull_episode_status_state_stats_from_truth,
)
from utils.pipeline.silver_eda_profiles import (
    build_silver_overview_summary,
    build_missingness_audit_table,
    build_duplicate_summary,
    build_numeric_describe_table,
    build_categorical_cardinality_table,
    build_state_sensor_profile_table,
    build_feature_behavior_effect_size_table,
    build_plot_feature_list,
)
from utils.pipeline.silver_eda_onsets import (
    find_anomaly_onsets,
    sample_onsets_evenly,
    build_aligned_onset_windows,
    summarize_onset_windows,
)
from utils.pipeline.silver_eda_groups import (
    build_normal_only_correlation_pairs,
    build_sensor_group_map_from_correlation,
    build_fault_propagation_pairings_from_strong_relationships,
)
from utils.pipeline.silver_eda_dropped import (
    load_dropped_sensor_dataframe,
    attach_state_column_to_dropped_dataframe,
    build_dropped_sensor_profiles_from_silver_preeda_truth,
)
from utils.pipeline.silver_eda_plots import (
    plot_correlation_heatmap,
    plot_state_distribution_histograms,
    plot_top_feature_overlay,
    plot_feature_timeseries_with_flag_spans,
    plot_aligned_onset_series,
)
from utils.pipeline.silver_eda_artifacts import (
    save_eda_table_artifact,
    save_eda_json_artifact,
    save_episode_status_counts_json,
    build_silver_eda_artifact_index,
)
from utils.pipeline.silver_eda_addons import (
    build_missingness_by_group_table,
    build_missingness_group_artifacts,
    build_state_transition_artifacts,
    build_robust_feature_comparison_artifacts,
    build_pca_diagnostics_artifacts,
    build_outlier_audit_artifacts,
)


def _build_default_runtime_inputs(
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    silver_cfg = config["silver_eda"]
    paths_cfg = config["resolved_paths"]
    filenames = config["filenames"]
    pipeline_cfg = config.get(
        "pipeline",
        {"execution_mode": "batch", "orchestration_mode": "script"},
    )

    return {
        "silver_cfg": silver_cfg,
        "filenames": filenames,
        "pipeline_cfg": pipeline_cfg,
        "stage": "silver_eda",
        "layer_name": silver_cfg["layer_name"],
        "silver_version": config["versions"]["silver"],
        "truth_version": config["versions"]["truth"],
        "pipeline_mode": pipeline_cfg["execution_mode"],
        "run_mode": config["runtime"]["mode"],
        "profile": config["runtime"]["profile"],
        "dataset_name_config": config["dataset"]["name"],
        "process_run_id": make_process_run_id(silver_cfg["process_run_id_prefix"]),
        "wandb_project": config["wandb"]["project"],
        "wandb_entity": config["wandb"]["entity"],
        "wandb_run_name": f"{config['versions']['silver']}_eda",
        "silver_train_data_path": Path(paths_cfg["data_silver_train_dir"]),
        "silver_artifacts_path": Path(paths_cfg["silver_artifacts_dir"]),
        "truths_path": Path(paths_cfg["truths_dir"]),
        "truth_index_path": Path(paths_cfg["truth_index_path"]),
        "logs_path": Path(paths_cfg["logs_root"]),
        "silver_train_data_file_name": filenames["silver_train_file_name"],
        "feature_registry_file_name": filenames["silver_feature_registry_file_name"],
        "status_column_fallback": silver_cfg.get("status_column_fallback", "machine_status"),
        "episode_column": silver_cfg.get("episode_column", "meta__episode_id"),
        "state_map": silver_cfg.get(
            "state_map",
            {"normal": "normal", "broken": "abnormal", "recovering": "recovery"},
        ),
        "max_onsets_to_use": int(silver_cfg.get("max_onsets_to_use", 25)),
        "onset_pre_window": int(silver_cfg.get("onset_pre_window", 20)),
        "onset_post_window": int(silver_cfg.get("onset_post_window", 20)),
        "top_feature_count": int(silver_cfg.get("top_feature_count", 6)),
        "min_abs_corr_for_group": float(silver_cfg.get("min_abs_corr_for_group", 0.60)),
        "strong_relationship_min_abs_corr": float(silver_cfg.get("strong_relationship_min_abs_corr", 0.70)),
        "dropped_parquet_file_name": silver_cfg.get("dropped_parquet_file_name"),
        "join_key": silver_cfg.get("join_key", "meta__record_id"),
    }


def _apply_runtime_overrides(
    runtime_inputs: dict[str, Any],
    *,
    silver_train_data_file_name: Optional[str] = None,
    feature_registry_file_name: Optional[str] = None,
    max_onsets_to_use: Optional[int] = None,
    onset_pre_window: Optional[int] = None,
    onset_post_window: Optional[int] = None,
    top_feature_count: Optional[int] = None,
    dropped_parquet_file_name: Optional[str] = None,
    join_key: Optional[str] = None,
) -> dict[str, Any]:
    updated = dict(runtime_inputs)

    if silver_train_data_file_name:
        updated["silver_train_data_file_name"] = silver_train_data_file_name
    if feature_registry_file_name:
        updated["feature_registry_file_name"] = feature_registry_file_name
    if max_onsets_to_use is not None:
        updated["max_onsets_to_use"] = int(max_onsets_to_use)
    if onset_pre_window is not None:
        updated["onset_pre_window"] = int(onset_pre_window)
    if onset_post_window is not None:
        updated["onset_post_window"] = int(onset_post_window)
    if top_feature_count is not None:
        updated["top_feature_count"] = int(top_feature_count)
    if dropped_parquet_file_name:
        updated["dropped_parquet_file_name"] = dropped_parquet_file_name
    if join_key:
        updated["join_key"] = join_key

    return updated


def _ensure_stage_directories(runtime_inputs: dict[str, Any]) -> None:
    runtime_inputs["silver_artifacts_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["truths_path"].mkdir(parents=True, exist_ok=True)
    runtime_inputs["logs_path"].mkdir(parents=True, exist_ok=True)


def _initialize_logger(paths) -> logging.Logger:
    log_path = paths.logs / "silver_eda.log"
    configure_logging("capstone", log_path, level=logging.DEBUG, overwrite_handlers=True)
    logger = logging.getLogger("capstone.silver_eda")
    logger.info("Silver EDA stage starting")
    log_layer_paths(paths, current_layer="silver", logger=logger)
    return logger


def _initialize_wandb_run(runtime_inputs: dict[str, Any]):
    return wandb.init(
        project=runtime_inputs["wandb_project"],
        entity=runtime_inputs["wandb_entity"],
        name=runtime_inputs["wandb_run_name"],
        job_type="silver_eda",
    )


def _load_inputs(runtime_inputs: dict[str, Any], logger: logging.Logger):
    silver_path = runtime_inputs["silver_train_data_path"] / runtime_inputs["silver_train_data_file_name"]
    registry_path = runtime_inputs["silver_artifacts_path"] / runtime_inputs["feature_registry_file_name"]

    if not silver_path.exists():
        raise FileNotFoundError(f"Silver dataset not found: {silver_path}")
    if not registry_path.exists():
        raise FileNotFoundError(f"Feature registry not found: {registry_path}")

    dataframe = load_data(silver_path.parent, silver_path.name)
    feature_registry = load_json(registry_path)

    logger.info("Loaded Silver dataframe: %s | shape=%s", silver_path, dataframe.shape)
    logger.info("Loaded feature registry: %s", registry_path)

    return dataframe, feature_registry, silver_path, registry_path


def _resolve_parent_truth(dataframe, runtime_inputs: dict[str, Any]):
    parent_truth_hash = extract_truth_hash(dataframe)
    if parent_truth_hash is None:
        raise ValueError("Silver EDA input dataframe missing readable meta__truth_hash.")

    dataset_name_series = dataframe["meta__dataset"].dropna().astype("string").str.strip()
    dataset_name_series = dataset_name_series[dataset_name_series != ""]
    if len(dataset_name_series) == 0:
        raise ValueError("Silver EDA input dataframe missing usable meta__dataset values.")

    dataset_name = str(dataset_name_series.iloc[0]).strip()

    parent_truth = load_parent_truth_record_from_dataframe(
        dataframe=dataframe,
        truth_dir=runtime_inputs["truths_path"],
        parent_layer_name="silver",
        dataset_name=dataset_name,
        column_name="meta__truth_hash",
    )

    resolved_dataset_name = get_dataset_name_from_truth(parent_truth)
    resolved_parent_truth_hash = get_truth_hash(parent_truth)
    pipeline_mode = get_pipeline_mode_from_truth(parent_truth) or runtime_inputs["pipeline_mode"]

    return parent_truth, resolved_dataset_name, resolved_parent_truth_hash, pipeline_mode


def _pick_top_features(
    dataframe: pd.DataFrame,
    feature_registry: dict[str, Any],
    *,
    top_feature_count: int,
) -> list[str]:
    feature_columns = list(feature_registry.get("feature_columns", []))
    top_features = []

    for feature_name in feature_columns:
        if feature_name in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[feature_name]):
            top_features.append(feature_name)
        if len(top_features) >= top_feature_count:
            break

    return top_features


def run_silver_eda(
    *,
    config_root: Optional[Path] = None,
    dataset: str = "pump",
    mode: str = "train",
    profile: str = "default",
    project_root: Optional[Path] = None,
    silver_train_data_file_name: Optional[str] = None,
    feature_registry_file_name: Optional[str] = None,
    max_onsets_to_use: Optional[int] = None,
    onset_pre_window: Optional[int] = None,
    onset_post_window: Optional[int] = None,
    top_feature_count: Optional[int] = None,
    dropped_parquet_file_name: Optional[str] = None,
    join_key: Optional[str] = None,
) -> dict[str, Any]:
    paths = get_paths()

    config_root = config_root or paths.configs
    project_root = project_root or paths.root

    config = load_pipeline_config(
        config_root=config_root,
        stage="silver_eda",
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
        max_onsets_to_use=max_onsets_to_use,
        onset_pre_window=onset_pre_window,
        onset_post_window=onset_post_window,
        top_feature_count=top_feature_count,
        dropped_parquet_file_name=dropped_parquet_file_name,
        join_key=join_key,
    )

    _ensure_stage_directories(runtime_inputs)

    logger = _initialize_logger(paths)

    truthed_config = build_truth_config_block(config)
    truthed_config["pipeline"] = runtime_inputs["pipeline_cfg"]

    set_wandb_dir_from_config(config)
    export_config_snapshot(
        config,
        output_path=runtime_inputs["silver_artifacts_path"] / f"{dataset}__silver_eda__resolved_config.yaml",
    )

    dataframe, feature_registry, silver_path, registry_path = _load_inputs(runtime_inputs, logger)
    wandb_run = _initialize_wandb_run(runtime_inputs)

    ledger = Ledger(stage=runtime_inputs["stage"], recipe_id="silver_eda")
    ledger.add(
        kind="step",
        step="load_inputs",
        message="Loaded Silver dataset and feature registry",
        data={
            "silver_path": str(silver_path),
            "feature_registry_path": str(registry_path),
            "shape": list(dataframe.shape),
        },
        logger=logger,
    )

    parent_truth, dataset_name, parent_truth_hash, pipeline_mode = _resolve_parent_truth(
        dataframe,
        runtime_inputs,
    )

    status_column = resolve_state_column_from_truth(
        parent_truth,
        fallback_status_column=runtime_inputs["status_column_fallback"],
    )

    dataframe, state_col_synth = build_state_col_synth(
        dataframe,
        status_column=status_column,
        state_map=runtime_inputs["state_map"],
    )

    state_values = sorted(
        dataframe[state_col_synth]
        .dropna()
        .astype("string")
        .str.strip()
        .unique()
        .tolist()
    )

    top_features = _pick_top_features(
        dataframe,
        feature_registry,
        top_feature_count=runtime_inputs["top_feature_count"],
    )

    overview_summary = build_silver_overview_summary(dataframe)
    missingness_df = build_missingness_audit_table(dataframe)
    duplicates_summary = build_duplicate_summary(dataframe)
    numeric_describe_df = build_numeric_describe_table(dataframe)
    categorical_cardinality_df = build_categorical_cardinality_table(dataframe, exclude_columns=[])

    episode_payload_bundle = build_episode_status_payload_and_tables(
        dataframe,
        status_column=status_column,
        episode_column=runtime_inputs["episode_column"],
        state_map=runtime_inputs["state_map"],
    )
    status_tables = build_status_distribution_tables(
        dataframe,
        status_column=status_column,
    )

    profile_df = build_state_sensor_profile_table(
        dataframe,
        sensors=top_features,
        state_column=state_col_synth,
        state_values=state_values,
    )

    comparison_states = [state for state in state_values if state != "normal"]
    effect_size_df = build_feature_behavior_effect_size_table(
        dataframe,
        sensors=top_features,
        state_column=state_col_synth,
        baseline_state="normal",
        comparison_states=comparison_states,
    )

    corr_payload = build_normal_only_correlation_pairs(
        dataframe,
        feature_columns=top_features,
        state_column=state_col_synth,
        target_state="normal",
    )
    correlation_matrix_df = corr_payload["correlation_matrix"]
    correlation_pairs_df = corr_payload["correlation_pairs"]

    sensor_group_map_normal_df = build_sensor_group_map_from_correlation(
        correlation_matrix_df,
        min_abs_corr_for_group=runtime_inputs["min_abs_corr_for_group"],
    )

    strong_relationships_df = build_fault_propagation_pairings_from_strong_relationships(
        correlation_pairs_df,
        min_abs_corr=runtime_inputs["strong_relationship_min_abs_corr"],
    )

    plot_features = build_plot_feature_list(
        effect_size_df,
        max_features=runtime_inputs["top_feature_count"],
    )

    onsets_table = find_anomaly_onsets(dataframe)
    onsets_table = sample_onsets_evenly(onsets_table, runtime_inputs["max_onsets_to_use"])

    aligned_windows_df = build_aligned_onset_windows(
        dataframe,
        onsets=onsets_table,
        feature_columns=plot_features,
        pre_window=runtime_inputs["onset_pre_window"],
        post_window=runtime_inputs["onset_post_window"],
    )
    onset_summary_df = summarize_onset_windows(
        aligned_windows_df,
        feature_columns=plot_features,
    )

    dropped_profile_df = pd.DataFrame()
    dropped_effect_size_df = pd.DataFrame()
    dropped_artifact_present = False

    if runtime_inputs["dropped_parquet_file_name"]:
        dropped_path = runtime_inputs["silver_artifacts_path"] / runtime_inputs["dropped_parquet_file_name"]
        if dropped_path.exists():
            dropped_df = load_dropped_sensor_dataframe(dropped_path=dropped_path)
            dropped_df = attach_state_column_to_dropped_dataframe(
                dropped_df,
                silver_dataframe=dataframe,
                state_column=status_column,
                synthetic_state_column=state_col_synth,
                join_key=runtime_inputs["join_key"],
            )

            dropped_feature_columns = [
                column for column in dropped_df.columns
                if column not in {
                    runtime_inputs["join_key"],
                    status_column,
                    state_col_synth,
                }
                and pd.api.types.is_numeric_dtype(dropped_df[column])
            ]

            if len(dropped_feature_columns) > 0:
                dropped_outputs = build_dropped_sensor_profiles_from_silver_preeda_truth(
                    dropped_df,
                    dropped_feature_columns=dropped_feature_columns,
                    state_column=state_col_synth,
                    state_values=state_values,
                    baseline_state="normal",
                    comparison_states=comparison_states,
                )
                dropped_profile_df = dropped_outputs["profile_df"]
                dropped_effect_size_df = dropped_outputs["effect_size_df"]
                dropped_artifact_present = True

    artifact_paths: dict[str, str] = {}

    artifact_paths["overview_summary_path"] = save_eda_json_artifact(
        overview_summary,
        output_path=runtime_inputs["silver_artifacts_path"] / "silver_overview__summary.json",
    )

    artifact_paths["missingness_path"] = save_eda_table_artifact(
        missingness_df,
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="missingness__audit.csv",
    )

    artifact_paths["duplicates_summary_path"] = save_eda_json_artifact(
        duplicates_summary,
        output_path=runtime_inputs["silver_artifacts_path"] / "duplicates__summary.json",
    )

    artifact_paths["numeric_describe_path"] = save_eda_table_artifact(
        numeric_describe_df,
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="column_stats__numeric_describe.csv",
    )

    artifact_paths["categorical_cardinality_path"] = save_eda_table_artifact(
        categorical_cardinality_df,
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="column_stats__categorical_cardinality.csv",
    )

    artifact_paths["episode_status_counts_path"] = save_episode_status_counts_json(
        episode_payload_bundle["episode_status_counts_df"],
        output_path=runtime_inputs["silver_artifacts_path"] / f"{dataset_name}__silver_eda__episode_status_counts.json",
    )

    artifact_paths["status_counts_path"] = save_eda_table_artifact(
        status_tables["status_counts"],
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="status_counts.csv",
    )

    artifact_paths["global_status_stats_path"] = save_eda_table_artifact(
        episode_payload_bundle["global_status_stats_df"],
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="global_status_stats.csv",
    )

    artifact_paths["episode_status_means_path"] = save_eda_table_artifact(
        episode_payload_bundle["episode_status_means_df"],
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="episode_status_means.csv",
    )

    artifact_paths["episode_status_percent_means_path"] = save_eda_table_artifact(
        episode_payload_bundle["episode_status_percent_means_df"],
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="episode_status_percent_means.csv",
    )

    artifact_paths["episode_totals_path"] = save_eda_table_artifact(
        episode_payload_bundle["episode_totals_df"],
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="episode_totals.csv",
    )

    artifact_paths["profile_table_path"] = save_eda_table_artifact(
        profile_df,
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="feature_profile_by_state.csv",
    )

    artifact_paths["effect_size_path"] = save_eda_table_artifact(
        effect_size_df,
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="feature_behavior__effect_size.csv",
    )

    if not correlation_matrix_df.empty:
        artifact_paths["correlation_matrix_normal_path"] = save_eda_table_artifact(
            correlation_matrix_df.reset_index(),
            output_dir=runtime_inputs["silver_artifacts_path"],
            file_name="correlation__normal.csv",
        )

    if not correlation_pairs_df.empty:
        artifact_paths["correlation_pairs_normal_path"] = save_eda_table_artifact(
            correlation_pairs_df,
            output_dir=runtime_inputs["silver_artifacts_path"],
            file_name="sensor_correlation_pairs_normal.csv",
        )

    artifact_paths["sensor_group_map_normal_path"] = save_eda_table_artifact(
        sensor_group_map_normal_df,
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="sensor_group_map_normal.csv",
    )

    artifact_paths["fault_propagation_pairings_path"] = save_eda_table_artifact(
        strong_relationships_df,
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="fault_propagation_pairings.csv",
    )

    artifact_paths["anomaly_onsets_path"] = save_eda_table_artifact(
        onsets_table,
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="anomaly_onsets__table.csv",
    )

    artifact_paths["aligned_onset_windows_path"] = save_eda_table_artifact(
        aligned_windows_df,
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="aligned_onset_windows.csv",
    )

    artifact_paths["aligned_onset_summary_path"] = save_eda_table_artifact(
        onset_summary_df,
        output_dir=runtime_inputs["silver_artifacts_path"],
        file_name="aligned_onset__summary.csv",
    )

    if dropped_artifact_present:
        artifact_paths["dropped_profile_table_path"] = save_eda_table_artifact(
            dropped_profile_df,
            output_dir=runtime_inputs["silver_artifacts_path"],
            file_name="dropped_feature_profile_by_state.csv",
        )
        artifact_paths["dropped_effect_size_path"] = save_eda_table_artifact(
            dropped_effect_size_df,
            output_dir=runtime_inputs["silver_artifacts_path"],
            file_name="dropped_feature_behavior__effect_size.csv",
        )

    if not correlation_matrix_df.empty:
        heatmap_path = runtime_inputs["silver_artifacts_path"] / "correlation__normal__heatmap.png"
        figure = plot_correlation_heatmap(correlation_matrix_df, output_path=heatmap_path)
        if figure is not None:
            artifact_paths["correlation_heatmap_normal_path"] = str(heatmap_path)

    distribution_dir = runtime_inputs["silver_artifacts_path"] / "distributions"
    plot_state_distribution_histograms(
        dataframe,
        features=plot_features,
        state_column=state_col_synth,
        state_values=state_values,
        output_dir=distribution_dir,
    )
    artifact_paths["distribution_plot_dir"] = str(distribution_dir)

    overlay_path = runtime_inputs["silver_artifacts_path"] / "top_feature_overlay.png"
    figure = plot_top_feature_overlay(dataframe, features=plot_features, output_path=overlay_path)
    if figure is not None:
        artifact_paths["top_feature_overlay_path"] = str(overlay_path)

    timeseries_dir = runtime_inputs["silver_artifacts_path"] / "timeseries"
    plot_feature_timeseries_with_flag_spans(
        dataframe,
        features=plot_features,
        output_dir=timeseries_dir,
    )
    artifact_paths["timeseries_plot_dir"] = str(timeseries_dir)

    for feature_name in plot_features:
        onset_plot_path = runtime_inputs["silver_artifacts_path"] / f"aligned_onset__mean__{feature_name}.png"
        figure = plot_aligned_onset_series(
            onset_summary_df,
            feature_name=feature_name,
            output_path=onset_plot_path,
        )
        if figure is not None:
            artifact_paths[f"aligned_onset_plot__{feature_name}"] = str(onset_plot_path)

    summary_payload = {
        "dataset_name": dataset_name,
        "state_column": status_column,
        "synthetic_state_column": state_col_synth,
        "top_features": top_features,
        "plot_features": plot_features,
        "state_values": state_values,
        "sensor_group_count": int(sensor_group_map_normal_df["group_name"].nunique()) if not sensor_group_map_normal_df.empty else 0,
        "strong_relationship_count": int(len(strong_relationships_df)),
        "anomaly_onset_count": int(len(onsets_table)),
        "episode_count": int(episode_payload_bundle["payload"]["episode_count"]),
        "duplicates_summary": duplicates_summary,
    }

    artifact_index_payload = build_silver_eda_artifact_index(
        artifact_paths=artifact_paths,
        summary_payload=summary_payload,
    )
    artifact_paths["artifact_index_path"] = save_eda_json_artifact(
        artifact_index_payload,
        output_path=runtime_inputs["silver_artifacts_path"] / f"{dataset_name}__silver_eda__artifact_index.json",
    )

    silver_truth = initialize_layer_truth(
        truth_version=runtime_inputs["truth_version"],
        dataset_name=dataset_name,
        layer_name=runtime_inputs["layer_name"],
        process_run_id=runtime_inputs["process_run_id"],
        pipeline_mode=pipeline_mode,
        parent_truth_hash=parent_truth_hash,
    )

    silver_truth = update_truth_section(
        silver_truth,
        "config_snapshot",
        {
            "silver_version": runtime_inputs["silver_version"],
            "status_column": status_column,
            "episode_column": runtime_inputs["episode_column"],
            "max_onsets_to_use": runtime_inputs["max_onsets_to_use"],
            "onset_pre_window": runtime_inputs["onset_pre_window"],
            "onset_post_window": runtime_inputs["onset_post_window"],
            "top_features": top_features,
            "plot_features": plot_features,
            "min_abs_corr_for_group": runtime_inputs["min_abs_corr_for_group"],
            "strong_relationship_min_abs_corr": runtime_inputs["strong_relationship_min_abs_corr"],
            "pipeline_mode": pipeline_mode,
        },
    )

    silver_truth = update_truth_section(
        silver_truth,
        "runtime_facts",
        {
            "input_row_count": int(len(dataframe)),
            "input_column_count": int(len(dataframe.columns)),
            "feature_column_count": int(len(feature_registry.get("feature_columns", []))),
            "state_column": status_column,
            "state_col_synth": state_col_synth,
            "top_features": top_features,
            "plot_features": plot_features,
            "state_values": state_values,
            "sensor_group_count": int(sensor_group_map_normal_df["group_name"].nunique()) if not sensor_group_map_normal_df.empty else 0,
            "strong_relationship_count": int(len(strong_relationships_df)),
            "anomaly_onset_count": int(len(onsets_table)),
            "episode_status_state_stats": episode_payload_bundle["payload"],
        },
    )

    silver_truth = update_truth_section(
        silver_truth,
        "artifact_paths",
        artifact_paths,
    )

    truth_record = build_truth_record(
        truth_base=silver_truth,
        row_count=0,
        column_count=0,
        meta_columns=[],
        feature_columns=[],
    )

    truth_hash = truth_record["truth_hash"]
    truth_path = save_truth_record(
        truth_record,
        truth_dir=runtime_inputs["truths_path"],
        dataset_name=dataset_name,
        layer_name=runtime_inputs["layer_name"],
    )
    append_truth_index(
        truth_record,
        truth_index_path=runtime_inputs["truth_index_path"],
    )

    saved_episode_status_payload = pull_episode_status_state_stats_from_truth(truth_record)

    final_summary_payload = {
        "dataset_name": dataset_name,
        "state_column": status_column,
        "synthetic_state_column": state_col_synth,
        "top_features": top_features,
        "plot_features": plot_features,
        "state_values": state_values,
        "sensor_group_count": int(sensor_group_map_normal_df["group_name"].nunique()) if not sensor_group_map_normal_df.empty else 0,
        "strong_relationship_count": int(len(strong_relationships_df)),
        "anomaly_onset_count": int(len(onsets_table)),
        "episode_count": int(episode_payload_bundle["payload"]["episode_count"]),
        "artifact_paths": artifact_paths,
        "saved_episode_status_payload": saved_episode_status_payload,
        "truth_hash": truth_hash,
        "truth_path": str(truth_path),
    }

    save_json(
        final_summary_payload,
        runtime_inputs["silver_artifacts_path"] / f"{dataset_name}__silver_eda__summary.json",
    )

    ledger.add(
        kind="step",
        step="save_artifacts",
        message="Saved Silver EDA artifacts and truth",
        data=final_summary_payload,
        logger=logger,
    )
    saved_ledger_path = ledger.write_json(
        runtime_inputs["silver_artifacts_path"] / f"silver_eda__{dataset_name}__ledger.json"
    )

    finalize_wandb_stage(
        run=wandb_run,
        stage=runtime_inputs["stage"],
        dataframe=dataframe,
        project_root=paths.root,
        logs_dir=paths.logs,
        dataset_dirs=[runtime_inputs["silver_artifacts_path"]],
        dataset_artifact_name=f"{dataset_name}-{runtime_inputs['stage']}-dataset",
        logger=logger,
        notebook_path=None,
        aliases=("latest",),
        table_key=None,
        table_n=15,
        profile=False,
    )
    wandb_run.finish()

    logger.info("Silver EDA stage completed successfully.")

    return {
        "status": "success",
        "layer_name": runtime_inputs["layer_name"],
        "dataset_name": dataset_name,
        "truth_hash": truth_hash,
        "truth_path": str(truth_path),
        "artifact_paths": artifact_paths,
        "ledger_path": str(saved_ledger_path),
        "top_features": top_features,
        "plot_features": plot_features,
        "sensor_group_count": int(sensor_group_map_normal_df["group_name"].nunique()) if not sensor_group_map_normal_df.empty else 0,
        "strong_relationship_count": int(len(strong_relationships_df)),
        "anomaly_onset_count": int(len(onsets_table)),
        "episode_count": int(episode_payload_bundle["payload"]["episode_count"]),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Silver EDA stage.")
    parser.add_argument("--dataset", default="pump")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--silver-train-data-file-name", default=None)
    parser.add_argument("--feature-registry-file-name", default=None)
    parser.add_argument("--max-onsets-to-use", type=int, default=None)
    parser.add_argument("--onset-pre-window", type=int, default=None)
    parser.add_argument("--onset-post-window", type=int, default=None)
    parser.add_argument("--top-feature-count", type=int, default=None)
    parser.add_argument("--dropped-parquet-file-name", default=None)
    parser.add_argument("--join-key", default=None)
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    result = run_silver_eda(
        config_root=Path(args.config_root) if args.config_root else None,
        dataset=args.dataset,
        mode=args.mode,
        profile=args.profile,
        project_root=Path(args.project_root) if args.project_root else None,
        silver_train_data_file_name=args.silver_train_data_file_name,
        feature_registry_file_name=args.feature_registry_file_name,
        max_onsets_to_use=args.max_onsets_to_use,
        onset_pre_window=args.onset_pre_window,
        onset_post_window=args.onset_post_window,
        top_feature_count=args.top_feature_count,
        dropped_parquet_file_name=args.dropped_parquet_file_name,
        join_key=args.join_key,
    )

    print(result)



'''
# Default Run
python -m pipelines.silver.run_silver_eda \
  --dataset pump \
  --mode train \
  --profile default

#With dropped parquet review:

python -m pipelines.silver.run_silver_eda \
  --dataset pump \
  --mode train \
  --profile default \
  --dropped-parquet-file-name pump__silver__dropped_features.parquet \
  --join-key meta__record_id




'''