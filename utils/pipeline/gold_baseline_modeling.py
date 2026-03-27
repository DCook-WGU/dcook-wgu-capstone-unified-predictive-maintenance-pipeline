"""
utils/gold_baseline_modeling.py

Baseline Isolation Forest helpers for Gold baseline stage.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from utils.gold_modeling_common import (
    compute_anomaly_scores_isolation_forest,
    choose_threshold_by_percentile,
    build_prediction_flags_from_scores,
    evaluate_against_labels,
    build_model_metric_summary,
)


def fit_baseline_isolation_forest(
    fit_dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    n_estimators: int = 200,
    contamination: str | float = "auto",
    max_samples: str | int | float = "auto",
    max_features: float = 1.0,
    bootstrap: bool = False,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[IsolationForest, Dict[str, Any]]:
    """
    Fit baseline Isolation Forest on normal-only fit rows.
    """
    feature_columns = [column_name for column_name in feature_columns if column_name in fit_dataframe.columns]
    if len(feature_columns) == 0:
        raise ValueError("No baseline feature columns available for model fitting.")

    fit_matrix = fit_dataframe[feature_columns].copy()
    if len(fit_matrix) == 0:
        raise ValueError("Fit dataframe has zero rows; cannot fit baseline model.")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(fit_matrix)

    fit_info = {
        "model_type": "IsolationForest",
        "fit_row_count": int(len(fit_matrix)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": list(feature_columns),
        "params": {
            "n_estimators": n_estimators,
            "contamination": contamination,
            "max_samples": max_samples,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "random_state": random_state,
            "n_jobs": n_jobs,
        },
    }
    return model, fit_info


def score_baseline_model(
    fitted_model,
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    score_column_name: str = "baseline_anomaly_score",
    threshold: float | None = None,
    prediction_column_name: str = "baseline_predicted_anomaly",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Score dataframe with fitted baseline model and optionally create predictions.
    """
    working_dataframe = dataframe.copy()
    feature_columns = [column_name for column_name in feature_columns if column_name in working_dataframe.columns]

    score_values = compute_anomaly_scores_isolation_forest(
        fitted_model,
        working_dataframe[feature_columns],
    )
    working_dataframe[score_column_name] = score_values

    scoring_info: Dict[str, Any] = {
        "row_count": int(len(working_dataframe)),
        "feature_count": int(len(feature_columns)),
        "score_column_name": score_column_name,
        "score_min": float(np.min(score_values)) if len(score_values) > 0 else None,
        "score_max": float(np.max(score_values)) if len(score_values) > 0 else None,
        "score_mean": float(np.mean(score_values)) if len(score_values) > 0 else None,
    }

    if threshold is not None:
        predictions = build_prediction_flags_from_scores(score_values, threshold=float(threshold))
        working_dataframe[prediction_column_name] = predictions
        scoring_info["prediction_column_name"] = prediction_column_name
        scoring_info["predicted_positive_count"] = int(np.sum(predictions))

    return working_dataframe, scoring_info


def evaluate_baseline_model(
    scored_dataframe: pd.DataFrame,
    *,
    label_column: str = "anomaly_flag",
    score_column: str = "baseline_anomaly_score",
    prediction_column: str = "baseline_predicted_anomaly",
) -> Dict[str, Any]:
    """
    Evaluate scored baseline dataframe against anomaly labels.
    """
    if label_column not in scored_dataframe.columns:
        raise ValueError(f"Missing label column: {label_column}")
    if score_column not in scored_dataframe.columns:
        raise ValueError(f"Missing score column: {score_column}")
    if prediction_column not in scored_dataframe.columns:
        raise ValueError(f"Missing prediction column: {prediction_column}")

    metrics = evaluate_against_labels(
        scored_dataframe[label_column],
        scored_dataframe[prediction_column],
        scores=scored_dataframe[score_column],
    )
    return metrics


def run_baseline_pipeline(
    *,
    fit_dataframe: pd.DataFrame,
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    all_dataframe: pd.DataFrame,
    feature_columns: Sequence[str],
    threshold_percentile: float = 95.0,
    model_params: Dict[str, Any] | None = None,
    score_column_name: str = "baseline_anomaly_score",
    prediction_column_name: str = "baseline_predicted_anomaly",
    label_column: str = "anomaly_flag",
) -> Dict[str, Any]:
    """
    End-to-end baseline modeling pipeline.
    """
    model_params = dict(model_params or {})

    model, fit_info = fit_baseline_isolation_forest(
        fit_dataframe,
        feature_columns=feature_columns,
        **model_params,
    )

    # Use training rows for threshold selection
    scored_train_no_threshold, train_scoring_info_no_threshold = score_baseline_model(
        model,
        train_dataframe,
        feature_columns=feature_columns,
        score_column_name=score_column_name,
        threshold=None,
        prediction_column_name=prediction_column_name,
    )

    threshold, threshold_info = choose_threshold_by_percentile(
        scored_train_no_threshold[score_column_name].to_numpy(),
        percentile=threshold_percentile,
    )

    scored_fit, fit_scoring_info = score_baseline_model(
        model,
        fit_dataframe,
        feature_columns=feature_columns,
        score_column_name=score_column_name,
        threshold=threshold,
        prediction_column_name=prediction_column_name,
    )

    scored_train, train_scoring_info = score_baseline_model(
        model,
        train_dataframe,
        feature_columns=feature_columns,
        score_column_name=score_column_name,
        threshold=threshold,
        prediction_column_name=prediction_column_name,
    )

    scored_test, test_scoring_info = score_baseline_model(
        model,
        test_dataframe,
        feature_columns=feature_columns,
        score_column_name=score_column_name,
        threshold=threshold,
        prediction_column_name=prediction_column_name,
    )

    scored_all, all_scoring_info = score_baseline_model(
        model,
        all_dataframe,
        feature_columns=feature_columns,
        score_column_name=score_column_name,
        threshold=threshold,
        prediction_column_name=prediction_column_name,
    )

    fit_metrics = evaluate_baseline_model(
        scored_fit,
        label_column=label_column,
        score_column=score_column_name,
        prediction_column=prediction_column_name,
    )
    train_metrics = evaluate_baseline_model(
        scored_train,
        label_column=label_column,
        score_column=score_column_name,
        prediction_column=prediction_column_name,
    )
    test_metrics = evaluate_baseline_model(
        scored_test,
        label_column=label_column,
        score_column=score_column_name,
        prediction_column=prediction_column_name,
    )
    all_metrics = evaluate_baseline_model(
        scored_all,
        label_column=label_column,
        score_column=score_column_name,
        prediction_column=prediction_column_name,
    )

    baseline_summary = {
        "model_name": "Baseline Isolation Forest",
        "fit_info": fit_info,
        "threshold": float(threshold),
        "threshold_info": threshold_info,
        "fit_metrics": fit_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "all_metrics": all_metrics,
        "fit_scoring_info": fit_scoring_info,
        "train_scoring_info": train_scoring_info,
        "test_scoring_info": test_scoring_info,
        "all_scoring_info": all_scoring_info,
        "baseline_metrics": build_model_metric_summary(
            model_name="Baseline Isolation Forest",
            threshold=threshold,
            threshold_info=threshold_info,
            evaluation_metrics=test_metrics,
            feature_columns=feature_columns,
        ),
    }

    return {
        "model": model,
        "threshold": threshold,
        "summary": baseline_summary,
        "scored_fit": scored_fit,
        "scored_train": scored_train,
        "scored_test": scored_test,
        "scored_all": scored_all,
        "train_scores_for_threshold": scored_train_no_threshold[[score_column_name]].copy(),
        "train_scoring_info_no_threshold": train_scoring_info_no_threshold,
    }