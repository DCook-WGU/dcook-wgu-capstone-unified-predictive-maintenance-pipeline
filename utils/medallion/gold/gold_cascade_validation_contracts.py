from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import math

def to_json_safe(value: Any) -> Any:
    """Convert common notebook objects into JSON-serializable values."""
    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe_summary",
            "rows": int(len(value)),
            "columns": [str(column) for column in value.columns],
        }

    if isinstance(value, pd.Series):
        return value.tolist()

    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]

    return value


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    """Create a stable SHA-256 hash for a JSON-like payload."""
    normalized = json.dumps(to_json_safe(payload), sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_json_contract(contract: Mapping[str, Any], output_path: Path) -> Path:
    """Write a validation contract to disk as pretty JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(to_json_safe(contract))
    payload.setdefault("contract_hash", stable_json_hash(payload))

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)

    return output_path


def load_json_contract(path: Path) -> dict[str, Any]:
    """Load a JSON validation contract from disk."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload


def _path_exists(path_value: Any) -> bool:
    if not path_value:
        return False
    return Path(str(path_value)).exists()


def _first_existing_path(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    return None


def _metric_value(metrics: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def build_cascade_variant_contract(
    *,
    dataset_id: str,
    run_id: str,
    model_id: str,
    source_notebook: str,
    cascade_variant: str,
    model_stage: str,
    operating_mode: str,
    stage3_type: str,
    rule_source: str,
    cascade_results: pd.DataFrame | None,
    cascade_metrics: Mapping[str, Any] | None,
    threshold_payload: Mapping[str, Any] | None,
    artifact_paths: Mapping[str, Any],
    stage1_model_path: Path | str | None = None,
    stage2_model_path: Path | str | None = None,
    stage3_rule_payload: Mapping[str, Any] | None = None,
    gold04_targets: Sequence[Mapping[str, Any]] | None = None,
    lineage_payload: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """
    Build a validation contract for one cascade notebook output.

    The contract records the durable evidence Gold 06 needs to validate a final
    model row without assuming Stage 3 is a joblib model. Stage 1 and Stage 2 may
    have saved estimator artifacts; Stage 3 can be represented as rule/config
    payload plus scored outputs.
    """
    cascade_metrics = dict(cascade_metrics or {})
    threshold_payload = dict(threshold_payload or {})
    stage3_rule_payload = dict(stage3_rule_payload or {})
    lineage_payload = dict(lineage_payload or {})

    result_columns: list[str] = []
    result_row_count: int | None = None
    if isinstance(cascade_results, pd.DataFrame):
        result_columns = [str(column) for column in cascade_results.columns]
        result_row_count = int(len(cascade_results))

    if gold04_targets is None:
        gold04_targets = [
            {
                "model_id": model_id,
                "operating_mode": operating_mode,
                "flag_column": "cascade_final_flag",
                "score_column": "cascade_final_score",
                "direct_gold04_match": True,
            }
        ]

    contract = {
        "contract_type": "cascade_validation_contract",
        "contract_version": "cascade_validation_contract_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "run_id": run_id,
        "model_id": model_id,
        "source_notebook": source_notebook,
        "cascade_variant": cascade_variant,
        "model_stage": model_stage,
        "operating_mode": operating_mode,
        "stage1_type": "joblib_model",
        "stage2_type": "joblib_model",
        "stage3_type": stage3_type,
        "stage3_saved_as_joblib": False,
        "rule_source": rule_source,
        "stage1_model_path": None if stage1_model_path is None else str(stage1_model_path),
        "stage2_model_path": None if stage2_model_path is None else str(stage2_model_path),
        "stage1_model_available": _path_exists(stage1_model_path),
        "stage2_model_available": _path_exists(stage2_model_path),
        "artifact_paths": dict(artifact_paths),
        "artifact_availability": {
            key: _path_exists(value)
            for key, value in dict(artifact_paths).items()
        },
        "result_row_count": result_row_count,
        "result_columns": result_columns,
        "metrics": cascade_metrics,
        "thresholds": threshold_payload,
        "stage3_rule_payload": stage3_rule_payload,
        "gold04_targets": [dict(target) for target in gold04_targets],
        "lineage": lineage_payload,
        "notes": notes or "",
    }
    contract["contract_hash"] = stable_json_hash(contract)
    return to_json_safe(contract)


def build_stage3_rule_payload_from_globals(
    *,
    notebook_globals: Mapping[str, Any],
    selected_mode: str = "selected_improved",
) -> dict[str, Any]:
    """
    Build a Stage 3 rule payload from common Gold 03 notebook variables.

    Missing values are recorded as None instead of raising. This lets notebooks
    write a partial contract now and improve it as rule variables become more
    standardized.
    """
    names = [
        "STAGE3_MIN_PRIMARY_SENSOR_HITS",
        "STAGE3_MIN_SECONDARY_SENSOR_HITS",
        "STAGE3_ROLLING_WINDOW_SIZE",
        "STAGE3_MINIMUM_FLAGS_IN_WINDOW",
        "STAGE3_MIN_WEIGHTED_SCORE",
        "STAGE3_STRONG_PRIMARY_HITS",
        "STAGE3_MIN_SELECTION_RECALL",
        "STAGE3_PROFILE_BREACH_WEIGHT",
        "STAGE3_CORROBORATION_WEIGHT",
        "STAGE3_PERSISTENCE_WEIGHT",
        "STAGE3_DRIFT_WEIGHT",
        "STAGE3_DRIFT_ROLLING_WINDOW_SIZE",
        "STAGE3_DRIFT_THRESHOLD_MULTIPLIER",
        "RELAXED_STAGE3_WEIGHTED_EVIDENCE_SCORE_COMPARISON",
        "MEDIUM_STAGE3_WEIGHTED_EVIDENCE_SCORE_COMPARISON",
        "STRICT_STAGE3_WEIGHTED_EVIDENCE_SCORE_COMPARISON",
    ]

    rule_parameters = {name: to_json_safe(notebook_globals.get(name)) for name in names}

    payload = {
        "stage3_is_rule_based": True,
        "stage3_saved_as_joblib": False,
        "selected_mode": selected_mode,
        "rule_parameters": rule_parameters,
        "primary_rule_sensors": to_json_safe(notebook_globals.get("stage3_primary_rule_sensors", [])),
        "secondary_rule_sensors": to_json_safe(notebook_globals.get("stage3_secondary_rule_sensors", [])),
        "watch_features": to_json_safe(notebook_globals.get("stage3_rule_watch_features", [])),
        "selected_params": to_json_safe(notebook_globals.get("stage3_selected_params", {})),
        "search_candidate_count": (
            int(len(notebook_globals["stage3_search_results"]))
            if isinstance(notebook_globals.get("stage3_search_results"), pd.DataFrame)
            else 0
        ),
        "operating_mode_metrics": to_json_safe(notebook_globals.get("stage3_operating_mode_metrics", {})),
    }
    return payload


def build_gold06_validation_targets() -> pd.DataFrame:
    """Return the expected final Gold model outputs Gold 06 should validate."""
    return pd.DataFrame(
        [
            {
                "model_id": "baseline",
                "source_notebook": "gold_02",
                "validation_type": "saved_model_and_score_output",
                "model_stage": "baseline",
                "operating_mode": "baseline",
                "contract_file": "pump__gold02_baseline_validation_contract.json",
                "direct_gold04_match": True,
            },
            {
                "model_id": "cascade_default",
                "source_notebook": "gold_03a",
                "validation_type": "cascade_rule_artifact",
                "model_stage": "cascade_default_final",
                "operating_mode": "default",
                "contract_file": "pump__gold03a_cascade_default_validation_contract.json",
                "direct_gold04_match": True,
            },
            {
                "model_id": "cascade_tuned",
                "source_notebook": "gold_03b",
                "validation_type": "cascade_rule_artifact",
                "model_stage": "cascade_tuned_final",
                "operating_mode": "tuned",
                "contract_file": "pump__gold03b_cascade_tuned_validation_contract.json",
                "direct_gold04_match": True,
            },
            {
                "model_id": "stage3_improved",
                "source_notebook": "gold_03c",
                "validation_type": "stage3_rule_artifact",
                "model_stage": "cascade_stage3_improved_final",
                "operating_mode": "selected_improved",
                "contract_file": "pump__gold03c_stage3_operating_modes_validation_contract.json",
                "direct_gold04_match": True,
            },
            {
                "model_id": "stage3_relaxed",
                "source_notebook": "gold_03c",
                "validation_type": "stage3_operating_mode",
                "model_stage": "cascade_stage3_improved_final",
                "operating_mode": "relaxed",
                "contract_file": "pump__gold03c_stage3_operating_modes_validation_contract.json",
                "direct_gold04_match": True,
            },
            {
                "model_id": "stage3_medium",
                "source_notebook": "gold_03c",
                "validation_type": "stage3_operating_mode",
                "model_stage": "cascade_stage3_improved_final",
                "operating_mode": "medium",
                "contract_file": "pump__gold03c_stage3_operating_modes_validation_contract.json",
                "direct_gold04_match": True,
            },
            {
                "model_id": "stage3_strict",
                "source_notebook": "gold_03c",
                "validation_type": "stage3_operating_mode",
                "model_stage": "cascade_stage3_improved_final",
                "operating_mode": "strict",
                "contract_file": "pump__gold03c_stage3_operating_modes_validation_contract.json",
                "direct_gold04_match": True,
            },
        ]
    )


def load_validation_contracts(contract_dir: Path, targets: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Load contracts referenced by the Gold 06 validation target table."""
    contract_dir = Path(contract_dir)
    contracts: dict[str, dict[str, Any]] = {}

    for contract_file in sorted(set(targets["contract_file"].dropna().astype(str))):
        path = contract_dir / contract_file
        if path.exists():
            contracts[contract_file] = load_json_contract(path)

    return contracts


def validate_gold04_against_contracts(
    *,
    gold04_dataframe: pd.DataFrame,
    validation_targets: pd.DataFrame,
    contracts: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """
    Validate that every Gold 04 final model row has a supporting contract.

    This function does not try to replay Stage 3. It checks whether the final
    seven Gold 04 rows can be linked to 03a/03b/03c contracts and whether core
    metrics are present in both places.
    """
    records: list[dict[str, Any]] = []

    if "model_id" not in gold04_dataframe.columns:
        raise KeyError("Gold 04 dataframe must include a model_id column.")

    gold04_by_model = {
        str(row["model_id"]): row.to_dict()
        for _, row in gold04_dataframe.iterrows()
    }

    for _, target in validation_targets.iterrows():
        model_id = str(target["model_id"])
        contract_file = str(target["contract_file"])
        contract = contracts.get(contract_file)
        gold04_row = gold04_by_model.get(model_id)

        record = {
            "model_id": model_id,
            "source_notebook": target.get("source_notebook"),
            "validation_type": target.get("validation_type"),
            "model_stage": target.get("model_stage"),
            "operating_mode": target.get("operating_mode"),
            "contract_file": contract_file,
            "contract_available": contract is not None,
            "gold04_row_available": gold04_row is not None,
            "stage3_type": None if contract is None else contract.get("stage3_type"),
            "stage3_saved_as_joblib": None if contract is None else contract.get("stage3_saved_as_joblib"),
            "stage1_model_available": None if contract is None else contract.get("stage1_model_available"),
            "stage2_model_available": None if contract is None else contract.get("stage2_model_available"),
            "result_row_count": None if contract is None else contract.get("result_row_count"),
        }

        if gold04_row is not None:
            record.update(
                {
                    "gold04_alert_count_test_rows": gold04_row.get("alert_count_test_rows"),
                    "gold04_precision": gold04_row.get("precision"),
                    "gold04_recall": gold04_row.get("recall"),
                    "gold04_f1": gold04_row.get("f1"),
                }
            )

        record["validation_status"] = (
            "pass"
            if record["contract_available"] and record["gold04_row_available"]
            else "missing_contract"
            if not record["contract_available"]
            else "missing_gold04_row"
        )
        records.append(record)

    return pd.DataFrame(records)


# =========================================================
# Explicit-input contract helpers for Gold 06 validation
# =========================================================


def _first_present_value(mapping: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    """Return the first non-null value found for a list of candidate keys."""
    for key in candidates:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _as_float_or_none(value: Any) -> float | None:
    """Convert a value to float when possible; otherwise return None."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int_or_none(value: Any) -> int | None:
    """Convert a value to int when possible; otherwise return None."""
    float_value = _as_float_or_none(value)
    if float_value is None:
        return None
    return int(float_value)


def normalize_gold_metric_payload(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize model metrics into the same fields used by Gold 04."""
    metrics_map = dict(metrics or {})

    return {
        "alert_count_test_rows": _as_int_or_none(
            _first_present_value(
                metrics_map,
                [
                    "alert_count_test_rows",
                    "final_alert_count_test_rows",
                    "predicted_positive_count",
                    "alerts",
                    "alert_count",
                ],
            )
        ),
        "alert_count_all_rows": _as_int_or_none(
            _first_present_value(
                metrics_map,
                [
                    "alert_count_all_rows",
                    "final_alert_count_all_rows",
                    "final_cascade_alert_count_all_rows",
                ],
            )
        ),
        "precision": _as_float_or_none(metrics_map.get("precision")),
        "recall": _as_float_or_none(metrics_map.get("recall")),
        "f1": _as_float_or_none(metrics_map.get("f1")),
        "roc_auc": _as_float_or_none(metrics_map.get("roc_auc")),
        "average_precision": _as_float_or_none(metrics_map.get("average_precision")),
        "source_metrics": to_json_safe(metrics_map),
    }


def summarize_validation_output_dataframe(
    *,
    dataframe: pd.DataFrame | None,
    flag_column: str | None = None,
    test_mask: pd.Series | None = None,
) -> dict[str, Any]:
    """Summarize the dataframe that supports one validation contract."""
    if dataframe is None:
        return {
            "dataframe_available": False,
            "row_count": None,
            "column_count": None,
            "flag_column": flag_column,
            "flag_column_available": False,
            "flag_count_all_rows": None,
            "flag_count_test_rows": None,
            "columns": [],
        }

    flag_column_available = bool(flag_column and flag_column in dataframe.columns)
    flag_count_all_rows = None
    flag_count_test_rows = None

    if flag_column_available and flag_column is not None:
        flag_values = pd.to_numeric(dataframe[flag_column], errors="coerce").fillna(0).astype(int)
        flag_count_all_rows = int(flag_values.sum())

        if test_mask is not None:
            if len(test_mask) != len(dataframe):
                raise ValueError("test_mask length does not match validation dataframe length.")
            aligned_mask = test_mask.reindex(dataframe.index).fillna(False).astype(bool)
            flag_count_test_rows = int(flag_values.loc[aligned_mask].sum())

    return {
        "dataframe_available": True,
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "flag_column": flag_column,
        "flag_column_available": flag_column_available,
        "flag_count_all_rows": flag_count_all_rows,
        "flag_count_test_rows": flag_count_test_rows,
        "columns": [str(column) for column in dataframe.columns],
    }


def build_gold_model_output_validation_contract(
    *,
    dataset_id: str,
    run_id: str,
    model_id: str,
    model_label: str,
    source_notebook: str,
    validation_type: str,
    model_stage: str,
    operating_mode: str,
    metrics: Mapping[str, Any] | None,
    output_dataframe: pd.DataFrame | None,
    output_flag_column: str | None,
    test_mask: pd.Series | None = None,
    rule_config: Mapping[str, Any] | None = None,
    rule_source: str | None = None,
    stage3_type: str = "rule_based",
    stage3_saved_as_joblib: bool = False,
    stage1_model_path: str | Path | None = None,
    stage2_model_path: str | Path | None = None,
    output_artifact_path: str | Path | None = None,
    lineage_payload: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Build one explicit validation contract for a final Gold model-comparison row."""
    contract = {
        "contract_type": "gold_model_output_validation_contract",
        "contract_version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": str(dataset_id),
        "run_id": str(run_id),
        "model_id": str(model_id),
        "model_label": str(model_label),
        "source_notebook": str(source_notebook),
        "validation_type": str(validation_type),
        "model_stage": str(model_stage),
        "operating_mode": str(operating_mode),
        "metric_payload": normalize_gold_metric_payload(metrics),
        "output_dataframe_summary": summarize_validation_output_dataframe(
            dataframe=output_dataframe,
            flag_column=output_flag_column,
            test_mask=test_mask,
        ),
        "output_artifact_path": None if output_artifact_path is None else str(output_artifact_path),
        "rule_source": rule_source,
        "rule_config": to_json_safe(dict(rule_config or {})),
        "stage1_type": "joblib_model",
        "stage2_type": "joblib_model",
        "stage3_type": stage3_type,
        "stage3_saved_as_joblib": bool(stage3_saved_as_joblib),
        "stage1_model_path": None if stage1_model_path is None else str(stage1_model_path),
        "stage2_model_path": None if stage2_model_path is None else str(stage2_model_path),
        "lineage_payload": to_json_safe(dict(lineage_payload or {})),
        "notes": notes or "",
    }
    contract["contract_hash"] = stable_json_hash(contract)
    return to_json_safe(contract)


def write_gold_model_output_validation_contract(
    *,
    contract: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """Write one final-model validation contract to its canonical artifact path."""
    return write_json_contract(contract, output_path)


def build_gold06_contract_validation_targets(
    *,
    dataset_id: str,
    include_baseline: bool = False,
) -> pd.DataFrame:
    """Build the Gold 06 contract-validation target table."""
    rows: list[dict[str, Any]] = []

    if include_baseline:
        rows.append(
            {
                "model_id": "baseline",
                "source_notebook": "gold_02",
                "validation_type": "saved_model_and_score_output",
                "model_stage": "baseline",
                "operating_mode": "baseline",
            }
        )

    rows.extend(
        [
            {
                "model_id": "cascade_default",
                "source_notebook": "gold_03a",
                "validation_type": "cascade_rule_artifact",
                "model_stage": "cascade_default_final",
                "operating_mode": "default",
            },
            {
                "model_id": "cascade_tuned",
                "source_notebook": "gold_03b",
                "validation_type": "cascade_rule_artifact",
                "model_stage": "cascade_tuned_final",
                "operating_mode": "tuned",
            },
            {
                "model_id": "stage3_improved",
                "source_notebook": "gold_03c",
                "validation_type": "stage3_rule_artifact",
                "model_stage": "cascade_stage3_improved_final",
                "operating_mode": "selected_improved",
            },
            {
                "model_id": "stage3_relaxed",
                "source_notebook": "gold_03c",
                "validation_type": "stage3_operating_mode",
                "model_stage": "cascade_stage3_improved_final",
                "operating_mode": "relaxed",
            },
            {
                "model_id": "stage3_medium",
                "source_notebook": "gold_03c",
                "validation_type": "stage3_operating_mode",
                "model_stage": "cascade_stage3_improved_final",
                "operating_mode": "medium",
            },
            {
                "model_id": "stage3_strict",
                "source_notebook": "gold_03c",
                "validation_type": "stage3_operating_mode",
                "model_stage": "cascade_stage3_improved_final",
                "operating_mode": "strict",
            },
        ]
    )

    targets = pd.DataFrame(rows)
    targets["contract_file"] = targets["model_id"].map(
        lambda model_id: f"{dataset_id}__gold__{model_id}_validation_contract.json"
    )
    return targets


def load_gold_model_output_validation_contracts(
    *,
    contract_dir: Path,
    validation_targets: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Load one contract per Gold 06 validation target."""
    contract_dir = Path(contract_dir)
    contracts_by_model_id: dict[str, dict[str, Any]] = {}

    if not contract_dir.exists():
        return contracts_by_model_id

    for _, target in validation_targets.iterrows():
        model_id = str(target["model_id"])
        contract_file = str(target["contract_file"])
        contract_path = contract_dir / contract_file

        if contract_path.exists():
            contracts_by_model_id[model_id] = load_json_contract(contract_path)

    return contracts_by_model_id


def _metrics_match(
    left: Any,
    right: Any,
    *,
    metric_name: str,
    tolerance: float = 1e-9,
) -> bool | None:
    """Return whether two metric values match, or None if either side is missing."""
    if left is None or right is None:
        return None

    if metric_name == "alert_count_test_rows":
        return int(float(left)) == int(float(right))

    left_float = _as_float_or_none(left)
    right_float = _as_float_or_none(right)

    if left_float is None or right_float is None:
        return None

    return math.isclose(left_float, right_float, rel_tol=tolerance, abs_tol=tolerance)


def validate_gold04_against_output_contracts(
    *,
    gold04_dataframe: pd.DataFrame,
    validation_targets: pd.DataFrame,
    contracts_by_model_id: Mapping[str, Mapping[str, Any]],
    metric_tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Validate Gold 04 final comparison rows against model-output contracts."""
    if "model_id" not in gold04_dataframe.columns:
        raise KeyError("Gold 04 dataframe must include a model_id column.")

    gold04_by_model_id = {
        str(row["model_id"]): row.to_dict()
        for _, row in gold04_dataframe.iterrows()
    }

    records: list[dict[str, Any]] = []

    for _, target in validation_targets.iterrows():
        model_id = str(target["model_id"])
        gold04_row = gold04_by_model_id.get(model_id)
        contract = contracts_by_model_id.get(model_id)

        contract_metrics = (
            dict(contract.get("metric_payload", {}))
            if contract is not None
            else {}
        )

        comparisons: dict[str, Any] = {}
        for metric_name in ["alert_count_test_rows", "precision", "recall", "f1"]:
            gold04_value = None if gold04_row is None else gold04_row.get(metric_name)
            contract_value = contract_metrics.get(metric_name)
            comparisons[f"{metric_name}_gold04"] = gold04_value
            comparisons[f"{metric_name}_contract"] = contract_value
            comparisons[f"{metric_name}_matches"] = _metrics_match(
                gold04_value,
                contract_value,
                metric_name=metric_name,
                tolerance=metric_tolerance,
            )

        match_values = [
            value
            for key, value in comparisons.items()
            if key.endswith("_matches") and value is not None
        ]

        if gold04_row is None:
            validation_status = "fail_missing_gold04_row"
        elif contract is None:
            validation_status = "fail_missing_contract"
        elif match_values and not all(match_values):
            validation_status = "warn_metric_mismatch"
        else:
            validation_status = "pass"

        records.append(
            {
                "model_id": model_id,
                "source_notebook": target.get("source_notebook"),
                "validation_type": target.get("validation_type"),
                "model_stage": target.get("model_stage"),
                "operating_mode": target.get("operating_mode"),
                "contract_file": target.get("contract_file"),
                "gold04_row_available": gold04_row is not None,
                "contract_available": contract is not None,
                "validation_status": validation_status,
                "stage3_type": None if contract is None else contract.get("stage3_type"),
                "stage3_saved_as_joblib": None if contract is None else contract.get("stage3_saved_as_joblib"),
                "contract_hash": None if contract is None else contract.get("contract_hash"),
                **comparisons,
            }
        )

    return pd.DataFrame(records)