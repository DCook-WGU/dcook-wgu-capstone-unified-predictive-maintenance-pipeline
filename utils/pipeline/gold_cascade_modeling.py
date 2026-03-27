"""
utils/gold_cascade_modeling.py

Cascade modeling helpers for Gold cascade stage.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from utils.pipeline.gold_modeling_common import (
    compute_anomaly_scores_isolation_forest,
    choose_threshold_by_percentile,
    build_prediction_flags_from_scores,
    evaluate_against_labels,
)
from utils.pipeline.gold_cascade_stage3_rules import (
    compute_primary_breach_count,
    compute_secondary_breach_count,
    compute_persistence_flag,
    compute_drift_flag,
    compose_stage3_decision,
)


def fit_stage1_model(
    fit_dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    model_params: Dict[str, Any],
) -> Tuple[IsolationForest, Dict[str, Any]]:
    """
    Fit broad Stage 1 Isolation Forest.
    """
    stage1_features = [column_name for column_name in feature_columns if column_name in fit_dataframe.columns]
    if len(stage1_features) == 0:
        raise ValueError("No Stage 1 feature columns available for fitting.")

    fit_matrix = fit_dataframe[stage1_features].copy()

    model = IsolationForest(**model_params)
    model.fit(fit_matrix)

    return model, {
        "stage_name": "stage1",
        "fit_row_count": int(len(fit_matrix)),
        "feature_count": int(len(stage1_features)),
        "feature_columns": list(stage1_features),
        "params": dict(model_params),
    }


def fit_stage2_model(
    fit_dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    model_params: Dict[str, Any],
) -> Tuple[IsolationForest, Dict[str, Any]]:
    """
    Fit narrower Stage 2 Isolation Forest.
    """
    stage2_features = [column_name for column_name in feature_columns if column_name in fit_dataframe.columns]
    if len(stage2_features) == 0:
        raise ValueError("No Stage 2 feature columns available for fitting.")

    fit_matrix = fit_dataframe[stage2_features].copy()

    model = IsolationForest(**model_params)
    model.fit(fit_matrix)

    return model, {
        "stage_name": "stage2",
        "fit_row_count": int(len(fit_matrix)),
        "feature_count": int(len(stage2_features)),
        "feature_columns": list(stage2_features),
        "params": dict(model_params),
    }


def _score_stage_dataframe(
    fitted_model,
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    score_column: str,
    threshold: float | None = None,
    prediction_column: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Score dataframe with fitted IF model and optionally add predictions.
    """
    working_dataframe = dataframe.copy()
    use_features = [column_name for column_name in feature_columns if column_name in working_dataframe.columns]

    score_values = compute_anomaly_scores_isolation_forest(
        fitted_model,
        working_dataframe[use_features],
    )
    working_dataframe[score_column] = score_values

    info: Dict[str, Any] = {
        "score_column": score_column,
        "row_count": int(len(working_dataframe)),
        "feature_count": int(len(use_features)),
        "score_min": float(np.min(score_values)) if len(score_values) > 0 else None,
        "score_max": float(np.max(score_values)) if len(score_values) > 0 else None,
        "score_mean": float(np.mean(score_values)) if len(score_values) > 0 else None,
    }

    if threshold is not None and prediction_column is not None:
        predictions = build_prediction_flags_from_scores(score_values, threshold=threshold)
        working_dataframe[prediction_column] = predictions
        info["prediction_column"] = prediction_column
        info["predicted_positive_count"] = int(np.sum(predictions))

    return working_dataframe, info


def evaluate_stage2_model_with_thresholds(
    fitted_model,
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    label_column: str,
    threshold_percentiles: Sequence[float],
    score_column: str = "stage2_anomaly_score",
    prediction_column: str = "stage2_predicted_anomaly",
) -> pd.DataFrame:
    """
    Evaluate multiple Stage 2 thresholds against labels.
    """
    scored_frame, _ = _score_stage_dataframe(
        fitted_model,
        dataframe,
        feature_columns=feature_columns,
        score_column=score_column,
        threshold=None,
        prediction_column=None,
    )

    results = []
    scores = scored_frame[score_column].to_numpy()

    for percentile in threshold_percentiles:
        threshold, threshold_info = choose_threshold_by_percentile(scores, percentile=float(percentile))
        predictions = build_prediction_flags_from_scores(scores, threshold=threshold)
        metrics = evaluate_against_labels(
            scored_frame[label_column],
            predictions,
            scores=scores,
        )
        results.append(
            {
                "threshold_percentile": float(percentile),
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "predicted_positive_count": metrics["predicted_positive_count"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
            }
        )

    return pd.DataFrame(results)


def run_stage2_selection(
    fitted_model,
    validation_dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    label_column: str,
    threshold_percentiles: Sequence[float],
    min_recall: float = 0.0,
    optimization_metric: str = "f1",
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    """
    Select Stage 2 threshold from a threshold grid.
    """
    threshold_table = evaluate_stage2_model_with_thresholds(
        fitted_model,
        validation_dataframe,
        feature_columns=feature_columns,
        label_column=label_column,
        threshold_percentiles=threshold_percentiles,
        score_column="stage2_anomaly_score",
        prediction_column="stage2_predicted_anomaly",
    )

    filtered = threshold_table.loc[threshold_table["recall"] >= float(min_recall)].copy()
    if len(filtered) == 0:
        filtered = threshold_table.copy()

    if optimization_metric not in filtered.columns:
        raise ValueError(f"Unsupported optimization_metric: {optimization_metric}")

    best_row = filtered.sort_values(
        by=[optimization_metric, "recall", "precision", "threshold_percentile"],
        ascending=[False, False, False, True],
    ).iloc[0]

    selected_threshold = float(best_row["threshold"])
    selection_info = {
        "selection_method": "threshold_grid_search",
        "optimization_metric": optimization_metric,
        "min_recall": float(min_recall),
        "selected_threshold_percentile": float(best_row["threshold_percentile"]),
        "selected_threshold": selected_threshold,
        "selected_metrics": {
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "f1": float(best_row["f1"]),
            "predicted_positive_count": int(best_row["predicted_positive_count"]),
        },
    }

    return selected_threshold, selection_info, threshold_table


def _variant_defaults(variant: str) -> Dict[str, Any]:
    """
    Variant-specific defaults for cascade behavior.
    """
    variant = str(variant).strip().lower()

    presets = {
        "default": {
            "stage1_percentile": 92.0,
            "stage2_percentile": 95.0,
            "use_stage2_search": False,
            "stage2_threshold_grid": [90.0, 92.5, 95.0, 97.0],
            "min_stage2_recall": 0.0,
            "optimization_metric": "f1",
            "primary_z_threshold": 2.5,
            "secondary_z_threshold": 1.75,
            "min_primary_breaches": 1,
            "min_secondary_breaches": 1,
            "min_consecutive_rows": 3,
            "drift_rolling_window": 10,
            "min_drift_delta": 0.10,
            "require_persistence_or_drift": True,
        },
        "tuned": {
            "stage1_percentile": 90.0,
            "stage2_percentile": 96.0,
            "use_stage2_search": True,
            "stage2_threshold_grid": [90.0, 92.0, 94.0, 95.0, 96.0, 97.0, 98.0],
            "min_stage2_recall": 0.35,
            "optimization_metric": "f1",
            "primary_z_threshold": 2.25,
            "secondary_z_threshold": 1.60,
            "min_primary_breaches": 1,
            "min_secondary_breaches": 1,
            "min_consecutive_rows": 2,
            "drift_rolling_window": 8,
            "min_drift_delta": 0.08,
            "require_persistence_or_drift": True,
        },
        "improved": {
            "stage1_percentile": 89.0,
            "stage2_percentile": 96.5,
            "use_stage2_search": True,
            "stage2_threshold_grid": [91.0, 93.0, 95.0, 96.0, 96.5, 97.0, 98.0],
            "min_stage2_recall": 0.40,
            "optimization_metric": "f1",
            "primary_z_threshold": 2.10,
            "secondary_z_threshold": 1.50,
            "min_primary_breaches": 1,
            "min_secondary_breaches": 1,
            "min_consecutive_rows": 2,
            "drift_rolling_window": 6,
            "min_drift_delta": 0.06,
            "require_persistence_or_drift": False,
        },
    }

    if variant not in presets:
        raise ValueError(f"Unsupported cascade variant: {variant}")

    return presets[variant]


def run_cascade_pipeline(
    *,
    fit_dataframe: pd.DataFrame,
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    all_dataframe: pd.DataFrame,
    stage1_feature_columns: Sequence[str],
    stage2_feature_columns: Sequence[str],
    reference_profile: Dict[str, Any],
    stage3_sensor_groups: Dict[str, Sequence[str]],
    label_column: str = "anomaly_flag",
    stage1_model_params: Dict[str, Any],
    stage2_model_params: Dict[str, Any],
    variant: str = "default",
) -> Dict[str, Any]:
    """
    End-to-end 3-stage cascade modeling pipeline.
    """
    variant_config = _variant_defaults(variant)

    stage1_model, stage1_fit_info = fit_stage1_model(
        fit_dataframe,
        feature_columns=stage1_feature_columns,
        model_params=stage1_model_params,
    )

    stage2_model, stage2_fit_info = fit_stage2_model(
        fit_dataframe,
        feature_columns=stage2_feature_columns,
        model_params=stage2_model_params,
    )

    # Threshold selection on training rows
    train_stage1_unscored, _ = _score_stage_dataframe(
        stage1_model,
        train_dataframe,
        feature_columns=stage1_feature_columns,
        score_column="stage1_anomaly_score",
        threshold=None,
        prediction_column=None,
    )
    stage1_threshold, stage1_threshold_info = choose_threshold_by_percentile(
        train_stage1_unscored["stage1_anomaly_score"].to_numpy(),
        percentile=float(variant_config["stage1_percentile"]),
    )

    if variant_config["use_stage2_search"]:
        stage2_threshold, stage2_selection_info, stage2_threshold_table = run_stage2_selection(
            stage2_model,
            train_dataframe,
            feature_columns=stage2_feature_columns,
            label_column=label_column,
            threshold_percentiles=variant_config["stage2_threshold_grid"],
            min_recall=float(variant_config["min_stage2_recall"]),
            optimization_metric=str(variant_config["optimization_metric"]),
        )
        stage2_threshold_info = dict(stage2_selection_info)
    else:
        train_stage2_unscored, _ = _score_stage_dataframe(
            stage2_model,
            train_dataframe,
            feature_columns=stage2_feature_columns,
            score_column="stage2_anomaly_score",
            threshold=None,
            prediction_column=None,
        )
        stage2_threshold, stage2_threshold_info = choose_threshold_by_percentile(
            train_stage2_unscored["stage2_anomaly_score"].to_numpy(),
            percentile=float(variant_config["stage2_percentile"]),
        )
        stage2_selection_info = {
            "selection_method": "fixed_percentile",
            "selected_threshold": float(stage2_threshold),
            "selected_threshold_percentile": float(variant_config["stage2_percentile"]),
        }
        stage2_threshold_table = pd.DataFrame()

    def _run_full_scoring(frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()

        working, _ = _score_stage_dataframe(
            stage1_model,
            working,
            feature_columns=stage1_feature_columns,
            score_column="stage1_anomaly_score",
            threshold=stage1_threshold,
            prediction_column="stage1_predicted_anomaly",
        )

        working, _ = _score_stage_dataframe(
            stage2_model,
            working,
            feature_columns=stage2_feature_columns,
            score_column="stage2_anomaly_score",
            threshold=stage2_threshold,
            prediction_column="stage2_predicted_anomaly",
        )

        # Stage 2 candidate gate
        working["stage2_candidate_flag"] = (
            (pd.to_numeric(working["stage1_predicted_anomaly"], errors="coerce").fillna(0).astype(int) == 1)
            & (pd.to_numeric(working["stage2_predicted_anomaly"], errors="coerce").fillna(0).astype(int) == 1)
        ).astype(int)

        working, primary_info = compute_primary_breach_count(
            working,
            feature_columns=stage2_feature_columns,
            reference_profile=reference_profile,
            z_threshold=float(variant_config["primary_z_threshold"]),
            output_column="stage3_primary_breach_count",
        )

        working, secondary_info = compute_secondary_breach_count(
            working,
            sensor_groups=stage3_sensor_groups,
            reference_profile=reference_profile,
            z_threshold=float(variant_config["secondary_z_threshold"]),
            output_column="stage3_secondary_breach_count",
        )

        working, persistence_info = compute_persistence_flag(
            working,
            candidate_column="stage2_candidate_flag",
            group_columns=("meta__asset_id", "meta__run_id"),
            min_consecutive_rows=int(variant_config["min_consecutive_rows"]),
            output_column="stage3_persistence_flag",
        )

        working, drift_info = compute_drift_flag(
            working,
            score_column="stage2_anomaly_score",
            group_columns=("meta__asset_id", "meta__run_id"),
            rolling_window=int(variant_config["drift_rolling_window"]),
            min_drift_delta=float(variant_config["min_drift_delta"]),
            output_column="stage3_drift_flag",
        )

        working, stage3_info = compose_stage3_decision(
            working,
            primary_breach_column="stage3_primary_breach_count",
            secondary_breach_column="stage3_secondary_breach_count",
            persistence_column="stage3_persistence_flag",
            drift_column="stage3_drift_flag",
            min_primary_breaches=int(variant_config["min_primary_breaches"]),
            min_secondary_breaches=int(variant_config["min_secondary_breaches"]),
            require_persistence_or_drift=bool(variant_config["require_persistence_or_drift"]),
            output_column="stage3_confirmed_flag",
        )

        working["cascade_predicted_anomaly"] = (
            (pd.to_numeric(working["stage2_candidate_flag"], errors="coerce").fillna(0).astype(int) == 1)
            & (pd.to_numeric(working["stage3_confirmed_flag"], errors="coerce").fillna(0).astype(int) == 1)
        ).astype(int)

        working.attrs["stage3_debug"] = {
            "primary_info": primary_info,
            "secondary_info": secondary_info,
            "persistence_info": persistence_info,
            "drift_info": drift_info,
            "stage3_info": stage3_info,
        }

        return working

    scored_fit = _run_full_scoring(fit_dataframe)
    scored_train = _run_full_scoring(train_dataframe)
    scored_test = _run_full_scoring(test_dataframe)
    scored_all = _run_full_scoring(all_dataframe)

    fit_metrics = evaluate_against_labels(
        scored_fit[label_column],
        scored_fit["cascade_predicted_anomaly"],
        scores=scored_fit["stage2_anomaly_score"],
    )
    train_metrics = evaluate_against_labels(
        scored_train[label_column],
        scored_train["cascade_predicted_anomaly"],
        scores=scored_train["stage2_anomaly_score"],
    )
    test_metrics = evaluate_against_labels(
        scored_test[label_column],
        scored_test["cascade_predicted_anomaly"],
        scores=scored_test["stage2_anomaly_score"],
    )
    all_metrics = evaluate_against_labels(
        scored_all[label_column],
        scored_all["cascade_predicted_anomaly"],
        scores=scored_all["stage2_anomaly_score"],
    )

    cascade_metrics = {
        "model": f"3-Stage Cascade ({variant})",
        "stage1_alert_count_all_rows": int(scored_all["stage1_predicted_anomaly"].sum()),
        "stage2_alert_count_all_rows": int(scored_all["stage2_candidate_flag"].sum()),
        "final_alert_count_all_rows": int(scored_all["cascade_predicted_anomaly"].sum()),
        "stage1_alert_count_test_rows": int(scored_test["stage1_predicted_anomaly"].sum()),
        "stage2_alert_count_test_rows": int(scored_test["stage2_candidate_flag"].sum()),
        "final_alert_count_test_rows": int(scored_test["cascade_predicted_anomaly"].sum()),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "f1": float(test_metrics["f1"]),
        "roc_auc": test_metrics["roc_auc"],
        "average_precision": test_metrics["average_precision"],
    }

    summary = {
        "variant": variant,
        "variant_config": variant_config,
        "stage1_fit_info": stage1_fit_info,
        "stage2_fit_info": stage2_fit_info,
        "stage1_threshold": float(stage1_threshold),
        "stage1_threshold_info": stage1_threshold_info,
        "stage2_threshold": float(stage2_threshold),
        "stage2_threshold_info": stage2_threshold_info,
        "stage2_selection_info": stage2_selection_info,
        "stage2_threshold_table": stage2_threshold_table.to_dict(orient="records"),
        "fit_metrics": fit_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "all_metrics": all_metrics,
        "cascade_metrics": cascade_metrics,
        "stage3_debug_test": scored_test.attrs.get("stage3_debug", {}),
    }

    return {
        "stage1_model": stage1_model,
        "stage2_model": stage2_model,
        "summary": summary,
        "scored_fit": scored_fit,
        "scored_train": scored_train,
        "scored_test": scored_test,
        "scored_all": scored_all,
    }


def build_cascade_summary(
    *,
    variant: str,
    summary: Dict[str, Any],
    stage1_feature_columns: Sequence[str],
    stage2_feature_columns: Sequence[str],
) -> Dict[str, Any]:
    """
    Add compact comparison-friendly summary fields.
    """
    compact_summary = dict(summary)
    compact_summary["variant"] = variant
    compact_summary["stage1_feature_count"] = int(len(list(stage1_feature_columns)))
    compact_summary["stage2_feature_count"] = int(len(list(stage2_feature_columns)))
    return compact_summary