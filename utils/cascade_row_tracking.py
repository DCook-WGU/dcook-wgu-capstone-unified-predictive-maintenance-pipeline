import numpy as np
import pandas as pd


DEFAULT_ROW_ID_COLUMN = "meta__row_id"


def ensure_stable_row_id(
    dataframe: pd.DataFrame,
    row_id_column: str = DEFAULT_ROW_ID_COLUMN
) -> pd.DataFrame:
    """
    Ensure the dataframe has a stable unique row id for downstream stage scoring
    and merge-back operations.
    """
    out = dataframe.copy()

    if row_id_column not in out.columns:
        out[row_id_column] = np.arange(len(out), dtype=np.int64)

    if out[row_id_column].isna().any():
        raise ValueError(f"{row_id_column} contains null values.")

    if not out[row_id_column].is_unique:
        raise ValueError(f"{row_id_column} must be unique.")

    return out


def get_identity_columns(
    dataframe: pd.DataFrame,
    row_id_column: str = DEFAULT_ROW_ID_COLUMN
) -> list[str]:
    """
    Return stable identity and ordering columns that should be carried through
    all stage scoring outputs.
    """
    preferred_columns = [
        row_id_column,
        "meta__record_id",
        "event_id",
        "time_index",
        "event_step",
        "event_time",
        "meta__asset_id",
        "meta__run_id",
        "machine_status",
    ]

    return [column for column in preferred_columns if column in dataframe.columns]


def build_stage_scoring_frame(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    mask: pd.Series | None = None,
    row_id_column: str = DEFAULT_ROW_ID_COLUMN,
) -> pd.DataFrame:
    """
    Build the exact dataframe that will be scored by a cascade stage while
    preserving row identity and ordering columns.
    """
    working = ensure_stable_row_id(dataframe, row_id_column=row_id_column)
    identity_columns = get_identity_columns(working, row_id_column=row_id_column)

    selected_columns = identity_columns + [column for column in feature_columns if column not in identity_columns]

    if mask is None:
        stage_dataframe = working[selected_columns].copy()
    else:
        stage_dataframe = working.loc[mask, selected_columns].copy()

    if stage_dataframe.empty:
        raise ValueError("Stage scoring frame is empty.")

    if stage_dataframe[row_id_column].isna().any():
        raise ValueError(f"{row_id_column} contains null values in stage scoring frame.")

    if not stage_dataframe[row_id_column].is_unique:
        raise ValueError(f"{row_id_column} must remain unique in stage scoring frame.")

    return stage_dataframe


def score_isolation_forest_stage(
    stage_dataframe: pd.DataFrame,
    model,
    feature_columns: list[str],
    stage_name: str,
    row_id_column: str = DEFAULT_ROW_ID_COLUMN,
) -> pd.DataFrame:
    """
    Score one Isolation Forest stage and return a row-level stage result dataframe.
    """
    out = stage_dataframe.copy().set_index(row_id_column, drop=False)

    X = out[feature_columns].copy()

    if X.empty:
        raise ValueError(f"{stage_name}: feature matrix is empty.")

    if not X.index.equals(out.index):
        raise ValueError(f"{stage_name}: feature matrix index does not match stage dataframe index.")

    score_column = f"{stage_name}_score"
    decision_column = f"{stage_name}_decision"
    pred_column = f"{stage_name}_pred"
    flag_column = f"{stage_name}_flag"

    out[score_column] = pd.Series(model.score_samples(X), index=out.index, name=score_column)
    out[decision_column] = pd.Series(model.decision_function(X), index=out.index, name=decision_column)
    out[pred_column] = pd.Series(model.predict(X), index=out.index, name=pred_column)
    out[flag_column] = (out[pred_column] == -1).astype(int)

    return out.reset_index(drop=True)


def merge_stage_results_back(
    master_dataframe: pd.DataFrame,
    stage_results_dataframe: pd.DataFrame,
    stage_name: str,
    row_id_column: str = DEFAULT_ROW_ID_COLUMN,
) -> pd.DataFrame:
    """
    Merge row-level stage results back onto the full master dataframe.
    """
    out = ensure_stable_row_id(master_dataframe, row_id_col=row_id_column)

    stage_columns = [
        row_id_column,
        f"{stage_name}_score",
        f"{stage_name}_decision",
        f"{stage_name}_pred",
        f"{stage_name}_flag",
    ]

    available_stage_columns = [column for column in stage_columns if column in stage_results_dataframe.columns]

    if row_id_column not in available_stage_columns:
        raise ValueError(f"{stage_name}: missing {row_id_column} in stage results.")

    if not out[row_id_column].is_unique:
        raise ValueError("Master dataframe row ids must be unique.")

    if not stage_results_dataframe[row_id_column].is_unique:
        raise ValueError(f"{stage_name}: stage result row ids must be unique.")

    out = out.merge(
        stage_results_dataframe[available_stage_columns],
        on=row_id_column,
        how="left",
    )

    return out


def finalize_stage_flag_columns(
    dataframe: pd.DataFrame,
    stage_names: list[str],
) -> pd.DataFrame:
    """
    Fill missing stage flag columns after merge-back so non-candidate rows are
    represented as 0 rather than NaN.
    """
    out = dataframe.copy()

    for stage_name in stage_names:
        flag_column = f"{stage_name}_flag"
        if flag_column in out.columns:
            out[flag_column] = out[flag_column].fillna(0).astype(int)

    return out


from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd


DEFAULT_ROW_ID_COL = "meta__row_id"


def get_detected_rows_dataframe(
    dataframe: pd.DataFrame,
    *,
    target_flag_column: str,
    row_id_col: str = DEFAULT_ROW_ID_COL,
    score_column: Optional[str] = None,
    decision_column: Optional[str] = None,
    pred_column: Optional[str] = None,
    include_columns: Optional[Sequence[str]] = None,
    preferred_identity_columns: Optional[Sequence[str]] = None,
    sort_by: Optional[str] = "time_index",
    ascending: bool = True,
    require_flag_column: bool = True,
) -> pd.DataFrame:
    """
    Return a dataframe containing only rows where the requested flag column is 1.

    This is intended for row-level anomaly review after baseline or cascade scoring.
    It preserves stable row identity and returns a clean subset that can be used in
    notebooks, exports, plots, and diagnostics.

    Parameters
    ----------
    dataframe:
        Full scored dataframe.
    target_flag_column:
        Column used to filter detected rows (for example 'baseline_flag',
        'stage1_flag', 'stage2_flag', or 'cascade_final_flag').
    row_id_col:
        Stable row identity column.
    score_column:
        Optional score column to include.
    decision_column:
        Optional decision column to include.
    pred_column:
        Optional prediction column to include.
    include_columns:
        Optional extra columns to include if present.
    preferred_identity_columns:
        Optional ordered list of identity/order columns to try to include.
    sort_by:
        Optional sort column after filtering. Defaults to 'time_index'.
    ascending:
        Sort direction.
    require_flag_column:
        If True, raise an error when the flag column is missing.
        If False, return an empty dataframe when missing.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe of detected rows.
    """
    if target_flag_column not in dataframe.columns:
        if require_flag_column:
            raise ValueError(
                f"Target flag column '{target_flag_column}' not found in dataframe."
            )
        return pd.DataFrame()

    if preferred_identity_columns is None:
        preferred_identity_columns = [
            row_id_col,
            "meta__record_id",
            "event_id",
            "time_index",
            "event_step",
            "event_time",
            "meta__asset_id",
            "meta__run_id",
            "machine_status",
            "anomaly_flag",
        ]

    candidate_columns: list[str] = []

    for col in preferred_identity_columns:
        if col in dataframe.columns and col not in candidate_columns:
            candidate_columns.append(col)

    if target_flag_column in dataframe.columns and target_flag_column not in candidate_columns:
        candidate_columns.append(target_flag_column)

    for optional_col in [score_column, decision_column, pred_column]:
        if optional_col and optional_col in dataframe.columns and optional_col not in candidate_columns:
            candidate_columns.append(optional_col)

    if include_columns is not None:
        for col in include_columns:
            if col in dataframe.columns and col not in candidate_columns:
                candidate_columns.append(col)

    detected_rows_dataframe = (
        dataframe.loc[
            dataframe[target_flag_column].fillna(0).astype(int) == 1,
            candidate_columns,
        ]
        .copy()
    )

    if sort_by is not None and sort_by in detected_rows_dataframe.columns:
        detected_rows_dataframe = detected_rows_dataframe.sort_values(
            by=sort_by,
            ascending=ascending,
        ).reset_index(drop=True)

    return detected_rows_dataframe


def get_stage_detected_rows_dataframe(
    dataframe: pd.DataFrame,
    *,
    stage_name: str,
    row_id_col: str = DEFAULT_ROW_ID_COL,
    include_columns: Optional[Sequence[str]] = None,
    sort_by: Optional[str] = "time_index",
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Convenience wrapper for standard stage naming patterns.

    Examples
    --------
    stage_name='baseline' -> baseline_flag / baseline_score / baseline_decision / baseline_pred
    stage_name='stage1'   -> stage1_flag / stage1_score / stage1_decision / stage1_pred
    stage_name='stage2'   -> stage2_flag / stage2_score / stage2_model_decision / stage2_model_pred
    """
    if stage_name == "stage2":
        score_column = "stage2_score"
        decision_column = (
            "stage2_model_decision"
            if "stage2_model_decision" in dataframe.columns
            else "stage2_decision"
        )
        pred_column = (
            "stage2_model_pred"
            if "stage2_model_pred" in dataframe.columns
            else "stage2_pred"
        )
        target_flag_column = "stage2_flag"
    elif stage_name == "cascade_final":
        target_flag_column = "cascade_final_flag"
        score_column = (
            "stage2_score"
            if "stage2_score" in dataframe.columns
            else None
        )
        decision_column = (
            "stage2_model_decision"
            if "stage2_model_decision" in dataframe.columns
            else None
        )
        pred_column = (
            "stage2_model_pred"
            if "stage2_model_pred" in dataframe.columns
            else None
        )
    else:
        target_flag_column = f"{stage_name}_flag"
        score_column = f"{stage_name}_score" if f"{stage_name}_score" in dataframe.columns else None
        decision_column = f"{stage_name}_decision" if f"{stage_name}_decision" in dataframe.columns else None
        pred_column = f"{stage_name}_pred" if f"{stage_name}_pred" in dataframe.columns else None

    return get_detected_rows_dataframe(
        dataframe,
        target_flag_column=target_flag_column,
        row_id_col=row_id_col,
        score_column=score_column,
        decision_column=decision_column,
        pred_column=pred_column,
        include_columns=include_columns,
        sort_by=sort_by,
        ascending=ascending,
    )


'''
# Baseline

baseline_detected_rows_dataframe = get_stage_detected_rows_dataframe(
    baseline_results,
    stage_name="baseline",
    include_columns=["baseline_flag"],
    sort_by="time_index",
)

print(f"Baseline detected rows: {len(baseline_detected_rows_dataframe):,}")
display(baseline_detected_rows_dataframe.head(20))

# Stage 1

stage1_detected_rows_dataframe = get_stage_detected_rows_dataframe(
    cascade_results,
    stage_name="stage1",
    include_columns=[
        "stage1_flag",
        "stage2_flag",
        "cascade_final_flag",
    ],
    sort_by="time_index",
)

print(f"Stage 1 detected rows: {len(stage1_detected_rows_dataframe):,}")
display(stage1_detected_rows_dataframe.head(20))

# Stage 2

stage2_detected_rows_dataframe = get_stage_detected_rows_dataframe(
    cascade_results,
    stage_name="stage2",
    include_columns=[
        "stage2_raw_flag",
        "stage2_flag",
        "cascade_final_flag",
    ],
    sort_by="time_index",
)

print(f"Stage 2 detected rows: {len(stage2_detected_rows_dataframe):,}")
display(stage2_detected_rows_dataframe.head(20))

# Final Cascade Alerts:

final_detected_rows_dataframe = get_stage_detected_rows_dataframe(
    cascade_results,
    stage_name="cascade_final",
    include_columns=[
        "stage1_flag",
        "stage2_raw_flag",
        "stage2_flag",
        "cascade_final_flag",
        "stage3_profile_breach_count",
        "stage3_rule_evidence_count",
    ],
    sort_by="time_index",
)

print(f"Final cascade detected rows: {len(final_detected_rows_dataframe):,}")
display(final_detected_rows_dataframe.head(20))

# For custom Calling

final_detected_rows_dataframe = get_detected_rows_dataframe(
    dataframe=cascade_results,
    target_flag_column="cascade_final_flag",
    score_column="stage2_score",
    decision_column="stage2_model_decision",
    pred_column="stage2_model_pred",
    include_columns=[
        "stage1_flag",
        "stage2_raw_flag",
        "stage2_flag",
        "stage3_profile_breach_count",
        "stage3_secondary_breach_count",
        "stage3_persistence_flag",
        "stage3_drift_flag",
        "stage3_rule_evidence_count",
        "cascade_final_flag",
    ],
    sort_by="time_index",
)


'''