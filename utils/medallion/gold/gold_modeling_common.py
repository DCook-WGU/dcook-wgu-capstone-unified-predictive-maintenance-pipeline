"""
utils/gold_modeling_common.py

Shared modeling helpers for Gold baseline and cascade stages.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_anomaly_scores_isolation_forest(
    fitted_model,
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    """
    Compute anomaly scores from a fitted Isolation Forest.

    sklearn IsolationForest.score_samples() returns higher values for more normal rows.
    For anomaly scoring, we negate the values so larger score = more anomalous.
    """
    raw_scores = fitted_model.score_samples(feature_frame)
    anomaly_scores = -1.0 * np.asarray(raw_scores, dtype=float)
    return anomaly_scores


def choose_threshold_by_percentile(
    scores: Sequence[float],
    *,
    percentile: float = 95.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Choose anomaly threshold using a score percentile.
    """
    scores_array = np.asarray(scores, dtype=float)

    if scores_array.size == 0:
        raise ValueError("Cannot choose threshold from empty score array.")

    threshold = float(np.percentile(scores_array, percentile))

    info = {
        "threshold_method": "percentile",
        "percentile": float(percentile),
        "threshold": threshold,
        "score_count": int(scores_array.size),
        "score_min": float(np.min(scores_array)),
        "score_max": float(np.max(scores_array)),
        "score_mean": float(np.mean(scores_array)),
    }
    return threshold, info


def build_prediction_flags_from_scores(
    scores: Sequence[float],
    *,
    threshold: float,
) -> np.ndarray:
    """
    Convert anomaly scores to binary predictions using threshold.
    """
    scores_array = np.asarray(scores, dtype=float)
    return (scores_array >= float(threshold)).astype(int)


def evaluate_against_labels(
    true_labels: Sequence[Any],
    predicted_labels: Sequence[Any],
    *,
    scores: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Evaluate binary predictions against labels.
    """
    y_true = pd.to_numeric(pd.Series(true_labels), errors="coerce").fillna(0).astype(int).to_numpy()
    y_pred = pd.to_numeric(pd.Series(predicted_labels), errors="coerce").fillna(0).astype(int).to_numpy()

    metrics: Dict[str, Any] = {
        "row_count": int(len(y_true)),
        "positive_label_count": int(np.sum(y_true)),
        "predicted_positive_count": int(np.sum(y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if scores is not None:
        score_array = np.asarray(scores, dtype=float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, score_array))
        except Exception:
            metrics["roc_auc"] = None

        try:
            metrics["average_precision"] = float(average_precision_score(y_true, score_array))
        except Exception:
            metrics["average_precision"] = None
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


def build_model_metric_summary(
    *,
    model_name: str,
    threshold: float,
    threshold_info: Dict[str, Any],
    evaluation_metrics: Dict[str, Any],
    feature_columns: Sequence[str],
) -> Dict[str, Any]:
    """
    Build compact model summary payload.
    """
    return {
        "model_name": model_name,
        "threshold": float(threshold),
        "threshold_info": threshold_info,
        "evaluation_metrics": evaluation_metrics,
        "feature_count": int(len(list(feature_columns))),
        "feature_columns": list(feature_columns),
    }