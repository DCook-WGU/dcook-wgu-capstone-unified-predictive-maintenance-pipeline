"""
utils/gold_comparison.py

Comparison helpers for Gold model evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd


def load_model_result_artifacts(
    *,
    baseline_summary: Dict[str, Any],
    cascade_summaries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Normalize baseline and cascade summaries into one comparison payload.
    """
    return {
        "baseline_summary": dict(baseline_summary),
        "cascade_summaries": [dict(summary) for summary in cascade_summaries],
    }


def validate_comparison_inputs(
    comparison_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate that baseline and cascade summaries can be compared together.
    """
    baseline_summary = comparison_payload["baseline_summary"]
    cascade_summaries = comparison_payload["cascade_summaries"]

    issues: List[str] = []

    baseline_model_name = baseline_summary.get("model_name", "Baseline Isolation Forest")
    baseline_test_metrics = baseline_summary.get("test_metrics", {})
    baseline_dataset_name = baseline_summary.get("dataset_name")

    variants_seen = set()
    cascade_dataset_names = set()

    for summary in cascade_summaries:
        variant = summary.get("variant")
        if variant in variants_seen:
            issues.append(f"Duplicate cascade variant found: {variant}")
        variants_seen.add(variant)

        dataset_name = summary.get("dataset_name")
        if dataset_name is not None:
            cascade_dataset_names.add(dataset_name)

        if "cascade_metrics" not in summary:
            issues.append(f"Cascade summary missing cascade_metrics for variant: {variant}")

    if baseline_dataset_name is not None and len(cascade_dataset_names) > 0:
        if any(dataset_name != baseline_dataset_name for dataset_name in cascade_dataset_names):
            issues.append(
                "Dataset name mismatch between baseline and cascade summaries."
            )

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "baseline_model_name": baseline_model_name,
        "baseline_dataset_name": baseline_dataset_name,
        "baseline_test_metrics": baseline_test_metrics,
        "cascade_variant_count": len(cascade_summaries),
    }


def build_model_comparison_dataframe(
    comparison_payload: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build one comparison dataframe containing baseline plus cascade variants.
    """
    baseline_summary = comparison_payload["baseline_summary"]
    cascade_summaries = comparison_payload["cascade_summaries"]

    rows: List[Dict[str, Any]] = []

    baseline_test_metrics = baseline_summary.get("test_metrics", {})
    baseline_metrics_block = baseline_summary.get("baseline_metrics", {})
    baseline_fit_info = baseline_summary.get("fit_info", {})

    rows.append(
        {
            "model_label": "Baseline",
            "model_name": baseline_summary.get("model_name", "Baseline Isolation Forest"),
            "variant": "baseline",
            "feature_count": baseline_metrics_block.get("feature_count", baseline_fit_info.get("feature_count")),
            "threshold": baseline_summary.get("threshold"),
            "alert_count_test_rows": baseline_test_metrics.get("predicted_positive_count"),
            "precision": baseline_test_metrics.get("precision"),
            "recall": baseline_test_metrics.get("recall"),
            "f1": baseline_test_metrics.get("f1"),
            "roc_auc": baseline_test_metrics.get("roc_auc"),
            "average_precision": baseline_test_metrics.get("average_precision"),
        }
    )

    for cascade_summary in cascade_summaries:
        cascade_metrics = cascade_summary.get("cascade_metrics", {})
        rows.append(
            {
                "model_label": f"Cascade ({cascade_summary.get('variant', 'unknown')})",
                "model_name": cascade_metrics.get("model", "3-Stage Cascade"),
                "variant": cascade_summary.get("variant"),
                "feature_count": cascade_summary.get("stage2_feature_count"),
                "threshold": cascade_summary.get("stage2_threshold"),
                "alert_count_test_rows": cascade_metrics.get("final_alert_count_test_rows"),
                "precision": cascade_metrics.get("precision"),
                "recall": cascade_metrics.get("recall"),
                "f1": cascade_metrics.get("f1"),
                "roc_auc": cascade_metrics.get("roc_auc"),
                "average_precision": cascade_metrics.get("average_precision"),
            }
        )

    return pd.DataFrame(rows)


def build_alert_count_comparison(
    comparison_payload: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build focused alert-count comparison table.
    """
    baseline_summary = comparison_payload["baseline_summary"]
    cascade_summaries = comparison_payload["cascade_summaries"]

    rows: List[Dict[str, Any]] = []

    baseline_test_metrics = baseline_summary.get("test_metrics", {})
    rows.append(
        {
            "model_label": "Baseline",
            "variant": "baseline",
            "alert_count_test_rows": baseline_test_metrics.get("predicted_positive_count"),
            "stage1_alert_count_test_rows": None,
            "stage2_alert_count_test_rows": None,
            "final_alert_count_test_rows": baseline_test_metrics.get("predicted_positive_count"),
        }
    )

    for cascade_summary in cascade_summaries:
        cascade_metrics = cascade_summary.get("cascade_metrics", {})
        rows.append(
            {
                "model_label": f"Cascade ({cascade_summary.get('variant', 'unknown')})",
                "variant": cascade_summary.get("variant"),
                "alert_count_test_rows": cascade_metrics.get("final_alert_count_test_rows"),
                "stage1_alert_count_test_rows": cascade_metrics.get("stage1_alert_count_test_rows"),
                "stage2_alert_count_test_rows": cascade_metrics.get("stage2_alert_count_test_rows"),
                "final_alert_count_test_rows": cascade_metrics.get("final_alert_count_test_rows"),
            }
        )

    return pd.DataFrame(rows)


def build_metric_comparison(
    comparison_payload: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build focused metric comparison table.
    """
    comparison_df = build_model_comparison_dataframe(comparison_payload)

    metric_columns = [
        "model_label",
        "variant",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    available_columns = [column for column in metric_columns if column in comparison_df.columns]
    return comparison_df[available_columns].copy()


def build_comparison_summary(
    comparison_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build compact summary useful for reporting and downstream display.
    """
    if comparison_df.empty:
        return {
            "model_count": 0,
            "best_f1_model": None,
            "lowest_alert_model": None,
            "summary_rows": [],
        }

    sorted_f1 = comparison_df.sort_values(by=["f1", "precision", "recall"], ascending=[False, False, False])
    best_f1_row = sorted_f1.iloc[0]

    sorted_alerts = comparison_df.sort_values(by=["alert_count_test_rows", "f1"], ascending=[True, False])
    lowest_alert_row = sorted_alerts.iloc[0]

    return {
        "model_count": int(len(comparison_df)),
        "best_f1_model": {
            "model_label": best_f1_row["model_label"],
            "variant": best_f1_row["variant"],
            "f1": best_f1_row["f1"],
            "precision": best_f1_row["precision"],
            "recall": best_f1_row["recall"],
        },
        "lowest_alert_model": {
            "model_label": lowest_alert_row["model_label"],
            "variant": lowest_alert_row["variant"],
            "alert_count_test_rows": lowest_alert_row["alert_count_test_rows"],
            "f1": lowest_alert_row["f1"],
        },
        "summary_rows": comparison_df.to_dict(orient="records"),
    }


def build_baseline_vs_best_cascade_delta(
    comparison_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compare baseline against the best cascade row by F1.
    """
    if comparison_df.empty:
        return {}

    baseline_rows = comparison_df.loc[comparison_df["variant"] == "baseline"].copy()
    cascade_rows = comparison_df.loc[comparison_df["variant"] != "baseline"].copy()

    if baseline_rows.empty or cascade_rows.empty:
        return {}

    baseline_row = baseline_rows.iloc[0]
    best_cascade_row = cascade_rows.sort_values(
        by=["f1", "precision", "recall"],
        ascending=[False, False, False],
    ).iloc[0]

    return {
        "baseline_model_label": baseline_row["model_label"],
        "best_cascade_model_label": best_cascade_row["model_label"],
        "best_cascade_variant": best_cascade_row["variant"],
        "delta_alert_count_test_rows": (
            None
            if pd.isna(baseline_row["alert_count_test_rows"]) or pd.isna(best_cascade_row["alert_count_test_rows"])
            else int(best_cascade_row["alert_count_test_rows"] - baseline_row["alert_count_test_rows"])
        ),
        "delta_precision": (
            None
            if pd.isna(baseline_row["precision"]) or pd.isna(best_cascade_row["precision"])
            else float(best_cascade_row["precision"] - baseline_row["precision"])
        ),
        "delta_recall": (
            None
            if pd.isna(baseline_row["recall"]) or pd.isna(best_cascade_row["recall"])
            else float(best_cascade_row["recall"] - baseline_row["recall"])
        ),
        "delta_f1": (
            None
            if pd.isna(baseline_row["f1"]) or pd.isna(best_cascade_row["f1"])
            else float(best_cascade_row["f1"] - baseline_row["f1"])
        ),
    }