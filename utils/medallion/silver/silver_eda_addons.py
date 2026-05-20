from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _choose_order_column(dataframe: pd.DataFrame) -> str:
    for column_name in ["event_step", "time_index", "event_time"]:
        if column_name in dataframe.columns:
            return column_name
    raise KeyError("Could not resolve an order column. Expected one of: event_step, time_index, event_time.")


def _iqr_bounds(series: pd.Series, iqr_multiplier: float = 1.5) -> tuple[float | None, float | None, float | None]:
    clean = _safe_numeric_series(series).dropna()
    if clean.empty:
        return None, None, None

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1

    if np.isnan(iqr):
        return None, None, None

    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    return float(lower), float(upper), float(iqr)


def _mad(series: pd.Series) -> float | None:
    clean = _safe_numeric_series(series).dropna()
    if clean.empty:
        return None
    median_value = float(clean.median())
    mad_value = float(np.median(np.abs(clean - median_value)))
    if np.isnan(mad_value):
        return None
    return mad_value


def _plot_heatmap_from_pivot(
    plot_matrix: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    x_label: str,
    y_label: str,
    figsize: tuple[int, int] = (12, 6),
) -> Optional[str]:
    if plot_matrix.empty:
        return None

    plt.figure(figsize=figsize)
    plt.imshow(plot_matrix.values, aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(plot_matrix.columns)), plot_matrix.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(plot_matrix.index)), plot_matrix.index, fontsize=7)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()
    plt.close()

    return str(out_path)


def build_missingness_by_group_table(
    dataframe: pd.DataFrame,
    *,
    group_column: str,
    feature_columns: Sequence[str],
    include_only_features_with_missingness: bool = True,
) -> pd.DataFrame:
    """
    Build a long missingness table by group and feature.
    """
    if group_column not in dataframe.columns:
        return pd.DataFrame(
            columns=[
                "group_column",
                "group_value",
                "feature",
                "row_count",
                "missing_count",
                "missing_pct",
            ]
        )

    available_features = [column for column in feature_columns if column in dataframe.columns]
    if len(available_features) == 0:
        return pd.DataFrame(
            columns=[
                "group_column",
                "group_value",
                "feature",
                "row_count",
                "missing_count",
                "missing_pct",
            ]
        )

    rows = []

    grouped = dataframe.groupby(group_column, dropna=False)
    for group_value, group_frame in grouped:
        row_count = int(len(group_frame))
        if row_count == 0:
            continue

        for feature_name in available_features:
            missing_count = int(group_frame[feature_name].isna().sum())
            missing_pct = float(missing_count / row_count) if row_count > 0 else None

            if include_only_features_with_missingness and missing_count == 0:
                continue

            rows.append(
                {
                    "group_column": str(group_column),
                    "group_value": str(group_value),
                    "feature": str(feature_name),
                    "row_count": row_count,
                    "missing_count": missing_count,
                    "missing_pct": missing_pct,
                }
            )

    return pd.DataFrame(rows)


def build_missingness_group_artifacts(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    artifacts_dir: Path,
    state_column: Optional[str] = None,
    episode_column: Optional[str] = None,
    top_feature_count_for_heatmap: int = 25,
    top_episode_count_for_heatmap: int = 25,
) -> dict:
    """
    Build missingness-by-state and missingness-by-episode tables and heatmaps.
    """
    artifacts = {
        "missingness_by_state_df": pd.DataFrame(),
        "missingness_by_state_path": None,
        "missingness_by_state_heatmap_path": None,
        "missingness_by_episode_df": pd.DataFrame(),
        "missingness_by_episode_path": None,
        "missingness_by_episode_heatmap_path": None,
    }

    if state_column is not None and state_column in dataframe.columns:
        missingness_by_state_df = build_missingness_by_group_table(
            dataframe,
            group_column=state_column,
            feature_columns=feature_columns,
            include_only_features_with_missingness=True,
        )

        artifacts["missingness_by_state_df"] = missingness_by_state_df

        if not missingness_by_state_df.empty:
            state_path = artifacts_dir / "missingness__by_state.csv"
            missingness_by_state_df.to_csv(state_path, index=False)
            artifacts["missingness_by_state_path"] = str(state_path)

            state_feature_order = (
                missingness_by_state_df.groupby("feature")["missing_pct"]
                .max()
                .sort_values(ascending=False)
                .head(top_feature_count_for_heatmap)
                .index
                .tolist()
            )

            state_pivot = (
                missingness_by_state_df.loc[
                    missingness_by_state_df["feature"].isin(state_feature_order)
                ]
                .pivot(index="group_value", columns="feature", values="missing_pct")
                .fillna(0.0)
            )

            heatmap_path = artifacts_dir / "missingness__state_heatmap.png"
            artifacts["missingness_by_state_heatmap_path"] = _plot_heatmap_from_pivot(
                state_pivot,
                title="Missingness by state",
                out_path=heatmap_path,
                x_label="Feature",
                y_label="State",
                figsize=(12, 5),
            )

    if episode_column is not None and episode_column in dataframe.columns:
        missingness_by_episode_df = build_missingness_by_group_table(
            dataframe,
            group_column=episode_column,
            feature_columns=feature_columns,
            include_only_features_with_missingness=True,
        )

        artifacts["missingness_by_episode_df"] = missingness_by_episode_df

        if not missingness_by_episode_df.empty:
            episode_path = artifacts_dir / "missingness__by_episode.csv"
            missingness_by_episode_df.to_csv(episode_path, index=False)
            artifacts["missingness_by_episode_path"] = str(episode_path)

            episode_score = (
                missingness_by_episode_df.groupby("group_value")["missing_pct"]
                .mean()
                .sort_values(ascending=False)
            )

            top_episodes = episode_score.head(top_episode_count_for_heatmap).index.tolist()

            episode_feature_order = (
                missingness_by_episode_df.groupby("feature")["missing_pct"]
                .max()
                .sort_values(ascending=False)
                .head(top_feature_count_for_heatmap)
                .index
                .tolist()
            )

            episode_pivot = (
                missingness_by_episode_df.loc[
                    missingness_by_episode_df["group_value"].isin(top_episodes)
                    & missingness_by_episode_df["feature"].isin(episode_feature_order)
                ]
                .pivot(index="group_value", columns="feature", values="missing_pct")
                .fillna(0.0)
            )

            heatmap_path = artifacts_dir / "missingness__episode_heatmap.png"
            artifacts["missingness_by_episode_heatmap_path"] = _plot_heatmap_from_pivot(
                episode_pivot,
                title="Missingness by episode (top missing episodes)",
                out_path=heatmap_path,
                x_label="Feature",
                y_label="Episode",
                figsize=(12, 7),
            )

    return artifacts


def build_state_transition_artifacts(
    dataframe: pd.DataFrame,
    *,
    state_column: str,
    artifacts_dir: Path,
    episode_column: Optional[str] = None,
    order_column: Optional[str] = None,
    state_order: Optional[Sequence[str]] = None,
) -> dict:
    """
    Build row-collapsed state transitions and dwell summaries.
    """
    if state_column not in dataframe.columns:
        return {
            "transition_counts_df": pd.DataFrame(),
            "transition_probability_df": pd.DataFrame(),
            "dwell_events_df": pd.DataFrame(),
            "dwell_summary_df": pd.DataFrame(),
            "transition_counts_path": None,
            "transition_probability_path": None,
            "dwell_events_path": None,
            "dwell_summary_path": None,
            "transition_heatmap_path": None,
            "dwell_plot_path": None,
        }

    if order_column is None:
        order_column = _choose_order_column(dataframe)

    group_columns = []
    if episode_column is not None and episode_column in dataframe.columns:
        group_columns.append(episode_column)
    else:
        for candidate in ["meta__asset_id", "meta__run_id"]:
            if candidate in dataframe.columns:
                group_columns.append(candidate)

    working_columns = [state_column, order_column] + group_columns
    working = dataframe[working_columns].copy()
    working = working.dropna(subset=[state_column]).copy()

    if working.empty:
        return {
            "transition_counts_df": pd.DataFrame(),
            "transition_probability_df": pd.DataFrame(),
            "dwell_events_df": pd.DataFrame(),
            "dwell_summary_df": pd.DataFrame(),
            "transition_counts_path": None,
            "transition_probability_path": None,
            "dwell_events_path": None,
            "dwell_summary_path": None,
            "transition_heatmap_path": None,
            "dwell_plot_path": None,
        }

    if len(group_columns) > 0:
        working = working.sort_values(group_columns + [order_column]).reset_index(drop=True)
    else:
        working = working.sort_values([order_column]).reset_index(drop=True)

    dwell_rows = []
    transition_rows = []

    if len(group_columns) > 0:
        grouped_items = working.groupby(group_columns, dropna=False)
    else:
        grouped_items = [(("all_rows",), working)]

    for group_key, group_frame in grouped_items:
        group_frame = group_frame.reset_index(drop=True)

        if isinstance(group_key, tuple):
            group_key_values = list(group_key)
        else:
            group_key_values = [group_key]

        previous_state = None
        run_start_position = None
        collapsed_states = []

        state_values = group_frame[state_column].astype(str).tolist()
        order_values = group_frame[order_column].tolist()

        for idx, current_state in enumerate(state_values):
            if previous_state is None:
                previous_state = current_state
                run_start_position = idx
                continue

            if current_state != previous_state:
                start_order_value = order_values[run_start_position]
                end_order_value = order_values[idx - 1]
                dwell_rows.append(
                    {
                        "group_key": "|".join(map(str, group_key_values)),
                        "state": str(previous_state),
                        "start_position": int(run_start_position),
                        "end_position": int(idx - 1),
                        "dwell_length_rows": int(idx - run_start_position),
                        "start_order_value": start_order_value,
                        "end_order_value": end_order_value,
                    }
                )
                collapsed_states.append(str(previous_state))
                previous_state = current_state
                run_start_position = idx

        if previous_state is not None and run_start_position is not None:
            start_order_value = order_values[run_start_position]
            end_order_value = order_values[-1]
            dwell_rows.append(
                {
                    "group_key": "|".join(map(str, group_key_values)),
                    "state": str(previous_state),
                    "start_position": int(run_start_position),
                    "end_position": int(len(state_values) - 1),
                    "dwell_length_rows": int(len(state_values) - run_start_position),
                    "start_order_value": start_order_value,
                    "end_order_value": end_order_value,
                }
            )
            collapsed_states.append(str(previous_state))

        for from_state, to_state in zip(collapsed_states[:-1], collapsed_states[1:]):
            transition_rows.append(
                {
                    "from_state": str(from_state),
                    "to_state": str(to_state),
                    "transition_count": 1,
                }
            )

    dwell_events_df = pd.DataFrame(dwell_rows)

    if transition_rows:
        transition_counts_df = (
            pd.DataFrame(transition_rows)
            .groupby(["from_state", "to_state"], as_index=False)["transition_count"]
            .sum()
        )
    else:
        transition_counts_df = pd.DataFrame(columns=["from_state", "to_state", "transition_count"])

    if state_order is None:
        state_order = sorted(
            pd.unique(
                pd.Series(
                    list(transition_counts_df.get("from_state", pd.Series(dtype="object")).dropna())
                    + list(transition_counts_df.get("to_state", pd.Series(dtype="object")).dropna())
                    + list(dwell_events_df.get("state", pd.Series(dtype="object")).dropna())
                )
            ).tolist()
        )

    transition_matrix = (
        transition_counts_df.pivot(index="from_state", columns="to_state", values="transition_count")
        .reindex(index=list(state_order), columns=list(state_order))
        .fillna(0.0)
    )

    transition_probability_df = transition_matrix.div(
        transition_matrix.sum(axis=1).replace(0.0, np.nan),
        axis=0,
    ).fillna(0.0)

    if dwell_events_df.empty:
        dwell_summary_df = pd.DataFrame(
            columns=[
                "state",
                "dwell_event_count",
                "mean_dwell_length_rows",
                "median_dwell_length_rows",
                "max_dwell_length_rows",
            ]
        )
    else:
        dwell_summary_df = (
            dwell_events_df.groupby("state", as_index=False)["dwell_length_rows"]
            .agg(
                dwell_event_count="count",
                mean_dwell_length_rows="mean",
                median_dwell_length_rows="median",
                max_dwell_length_rows="max",
            )
            .sort_values("mean_dwell_length_rows", ascending=False)
            .reset_index(drop=True)
        )

    transition_counts_path = artifacts_dir / "state_transition__counts.csv"
    transition_probability_path = artifacts_dir / "state_transition__probabilities.csv"
    dwell_events_path = artifacts_dir / "state_dwell__events.csv"
    dwell_summary_path = artifacts_dir / "state_dwell__summary.csv"

    transition_matrix.to_csv(transition_counts_path)
    transition_probability_df.to_csv(transition_probability_path)
    dwell_events_df.to_csv(dwell_events_path, index=False)
    dwell_summary_df.to_csv(dwell_summary_path, index=False)

    transition_heatmap_path = artifacts_dir / "state_transition__heatmap.png"
    transition_heatmap_path = _plot_heatmap_from_pivot(
        transition_probability_df,
        title="State transition probabilities",
        out_path=transition_heatmap_path,
        x_label="To state",
        y_label="From state",
        figsize=(6, 5),
    )

    dwell_plot_path = None
    if not dwell_events_df.empty:
        plt.figure(figsize=(10, 5))
        for state_value in state_order:
            state_durations = dwell_events_df.loc[
                dwell_events_df["state"].eq(state_value),
                "dwell_length_rows",
            ].dropna()

            if len(state_durations) == 0:
                continue

            bins = min(30, max(5, int(np.sqrt(len(state_durations)))))
            plt.hist(
                state_durations,
                bins=bins,
                alpha=0.5,
                label=str(state_value),
            )

        plt.title("State dwell lengths")
        plt.xlabel("Rows in contiguous run")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()

        out_path = artifacts_dir / "state_dwell__histogram.png"
        plt.savefig(out_path, dpi=200)
        plt.show()
        plt.close()
        dwell_plot_path = str(out_path)

    return {
        "transition_counts_df": transition_matrix.reset_index().rename(columns={"from_state": "from_state"}),
        "transition_probability_df": transition_probability_df.reset_index().rename(columns={"from_state": "from_state"}),
        "dwell_events_df": dwell_events_df,
        "dwell_summary_df": dwell_summary_df,
        "transition_counts_path": str(transition_counts_path),
        "transition_probability_path": str(transition_probability_path),
        "dwell_events_path": str(dwell_events_path),
        "dwell_summary_path": str(dwell_summary_path),
        "transition_heatmap_path": transition_heatmap_path,
        "dwell_plot_path": dwell_plot_path,
    }


def build_robust_feature_comparison_artifacts(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    state_column: str,
    artifacts_dir: Path,
    baseline_state: str = "normal",
    comparison_states: Sequence[str] = ("abnormal", "recovery"),
    plot_features: Optional[Sequence[str]] = None,
    state_plot_order: Optional[Sequence[str]] = None,
    max_plot_features: int = 12,
) -> dict:
    """
    Build a more robust state-comparison table and a few state comparison plots.
    """
    from scipy.stats import ks_2samp

    if state_column not in dataframe.columns:
        return {
            "comparison_table": pd.DataFrame(),
            "comparison_path": None,
            "boxplot_paths": [],
            "ecdf_paths": [],
            "top_features": [],
        }

    available_features = [
        feature_name
        for feature_name in feature_columns
        if feature_name in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[feature_name])
    ]

    rows = []

    for feature_name in available_features:
        numeric_series = _safe_numeric_series(dataframe[feature_name])

        baseline_values = numeric_series.loc[dataframe[state_column].eq(baseline_state)].dropna()
        if baseline_values.empty:
            continue

        baseline_median = float(baseline_values.median())
        baseline_q1 = float(baseline_values.quantile(0.25))
        baseline_q3 = float(baseline_values.quantile(0.75))
        baseline_iqr = float(baseline_q3 - baseline_q1)

        for comparison_state in comparison_states:
            comparison_values = numeric_series.loc[dataframe[state_column].eq(comparison_state)].dropna()

            if comparison_values.empty:
                continue

            comparison_median = float(comparison_values.median())
            comparison_q1 = float(comparison_values.quantile(0.25))
            comparison_q3 = float(comparison_values.quantile(0.75))
            comparison_iqr = float(comparison_q3 - comparison_q1)

            median_shift = comparison_median - baseline_median
            standardized_median_shift = None
            if baseline_iqr not in (0.0, np.nan) and not np.isnan(baseline_iqr) and baseline_iqr != 0:
                standardized_median_shift = median_shift / baseline_iqr

            iqr_ratio = None
            if baseline_iqr not in (0.0, np.nan) and not np.isnan(baseline_iqr) and baseline_iqr != 0:
                iqr_ratio = comparison_iqr / baseline_iqr

            ks_result = ks_2samp(baseline_values.values, comparison_values.values, method="auto")

            rows.append(
                {
                    "sensor": str(feature_name),
                    "baseline_state": str(baseline_state),
                    "comparison_state": str(comparison_state),
                    "baseline_n": int(len(baseline_values)),
                    "comparison_n": int(len(comparison_values)),
                    "baseline_median": baseline_median,
                    "comparison_median": comparison_median,
                    "baseline_iqr": baseline_iqr,
                    "comparison_iqr": comparison_iqr,
                    "median_shift": float(median_shift),
                    "standardized_median_shift": float(standardized_median_shift) if standardized_median_shift is not None else None,
                    "iqr_ratio": float(iqr_ratio) if iqr_ratio is not None else None,
                    "ks_statistic": float(ks_result.statistic),
                    "ks_pvalue": float(ks_result.pvalue),
                }
            )

    comparison_table = pd.DataFrame(rows)

    if comparison_table.empty:
        return {
            "comparison_table": comparison_table,
            "comparison_path": None,
            "boxplot_paths": [],
            "ecdf_paths": [],
            "top_features": [],
        }

    comparison_table["abs_standardized_median_shift"] = comparison_table["standardized_median_shift"].abs()
    comparison_table = comparison_table.sort_values(
        ["comparison_state", "abs_standardized_median_shift", "ks_statistic", "sensor"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    comparison_path = artifacts_dir / "feature_behavior__robust_comparison.csv"
    comparison_table.to_csv(comparison_path, index=False)

    if plot_features is None:
        plot_source = comparison_table.loc[
            comparison_table["comparison_state"].eq(comparison_states[0])
        ].copy()
        if plot_source.empty:
            plot_source = comparison_table.copy()

        plot_features = (
            plot_source["sensor"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(max_plot_features)
            .tolist()
        )
    else:
        plot_features = [feature_name for feature_name in plot_features if feature_name in available_features][:max_plot_features]

    if state_plot_order is None:
        state_plot_order = [baseline_state] + [state for state in comparison_states if state != baseline_state]

    boxplot_paths = []
    ecdf_paths = []

    for feature_name in plot_features:
        plot_rows = []
        for state_value in state_plot_order:
            feature_values = _safe_numeric_series(
                dataframe.loc[dataframe[state_column].eq(state_value), feature_name]
            ).dropna()

            if feature_values.empty:
                continue

            for value in feature_values.values:
                plot_rows.append({"state": str(state_value), "value": float(value)})

        if len(plot_rows) == 0:
            continue

        plot_df = pd.DataFrame(plot_rows)

        plt.figure(figsize=(9, 4))
        ordered_values = [
            plot_df.loc[plot_df["state"].eq(state_value), "value"].values
            for state_value in state_plot_order
            if state_value in plot_df["state"].unique()
        ]
        ordered_labels = [
            state_value
            for state_value in state_plot_order
            if state_value in plot_df["state"].unique()
        ]
        plt.boxplot(ordered_values, labels=ordered_labels, showfliers=False)
        plt.title(f"State boxplot: {feature_name}")
        plt.ylabel(feature_name)
        plt.tight_layout()

        boxplot_path = artifacts_dir / f"state_boxplot__{feature_name}.png"
        plt.savefig(boxplot_path, dpi=200)
        plt.show()
        plt.close()
        boxplot_paths.append(str(boxplot_path))

        plt.figure(figsize=(9, 4))
        any_line = False
        for state_value in state_plot_order:
            feature_values = _safe_numeric_series(
                dataframe.loc[dataframe[state_column].eq(state_value), feature_name]
            ).dropna().sort_values()

            if feature_values.empty:
                continue

            y_values = np.arange(1, len(feature_values) + 1) / len(feature_values)
            plt.plot(feature_values.values, y_values, label=str(state_value))
            any_line = True

        if any_line:
            plt.title(f"State ECDF: {feature_name}")
            plt.xlabel(feature_name)
            plt.ylabel("ECDF")
            plt.legend()
            plt.tight_layout()

            ecdf_path = artifacts_dir / f"state_ecdf__{feature_name}.png"
            plt.savefig(ecdf_path, dpi=200)
            plt.show()
            plt.close()
            ecdf_paths.append(str(ecdf_path))
        else:
            plt.close()

    return {
        "comparison_table": comparison_table,
        "comparison_path": str(comparison_path),
        "boxplot_paths": boxplot_paths,
        "ecdf_paths": ecdf_paths,
        "top_features": list(plot_features),
    }


def build_pca_diagnostics_artifacts(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    artifacts_dir: Path,
    sample_row_count: int = 20000,
    use_robust_scaler: bool = True,
    top_loading_count: int = 15,
) -> dict:
    """
    Build PCA explained variance and loading diagnostics.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler, StandardScaler

    numeric_feature_columns = [
        feature_name
        for feature_name in feature_columns
        if feature_name in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[feature_name])
    ]

    if len(numeric_feature_columns) < 2:
        return {
            "explained_variance_df": pd.DataFrame(),
            "loadings_df": pd.DataFrame(),
            "explained_variance_path": None,
            "loadings_path": None,
            "scree_plot_path": None,
            "pc1_loading_plot_path": None,
            "pc2_loading_plot_path": None,
        }

    modeling_frame = dataframe[numeric_feature_columns].copy()
    if len(modeling_frame) > sample_row_count:
        modeling_frame = modeling_frame.sample(n=sample_row_count, random_state=42).reset_index(drop=True)

    for feature_name in numeric_feature_columns:
        median_value = float(modeling_frame[feature_name].median(skipna=True))
        modeling_frame[feature_name] = _safe_numeric_series(modeling_frame[feature_name]).fillna(median_value)

    scaler = RobustScaler() if use_robust_scaler else StandardScaler()
    scaled_matrix = scaler.fit_transform(modeling_frame.values)

    component_count = min(10, scaled_matrix.shape[1], scaled_matrix.shape[0])
    if component_count < 2:
        component_count = 2

    pca_model = PCA(n_components=component_count, random_state=42)
    pca_model.fit(scaled_matrix)

    explained_variance_df = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(component_count)],
            "explained_variance_ratio": pca_model.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca_model.explained_variance_ratio_),
        }
    )

    loadings_df = pd.DataFrame(
        pca_model.components_.T,
        index=numeric_feature_columns,
        columns=[f"PC{i+1}" for i in range(component_count)],
    ).reset_index().rename(columns={"index": "feature"})

    explained_variance_path = artifacts_dir / "pca__explained_variance.csv"
    loadings_path = artifacts_dir / "pca__loadings.csv"
    explained_variance_df.to_csv(explained_variance_path, index=False)
    loadings_df.to_csv(loadings_path, index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, component_count + 1), explained_variance_df["explained_variance_ratio"].values, marker="o")
    plt.plot(range(1, component_count + 1), explained_variance_df["cumulative_explained_variance_ratio"].values, marker="o")
    plt.xticks(range(1, component_count + 1))
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA scree and cumulative variance")
    plt.legend(["Explained variance", "Cumulative variance"])
    plt.tight_layout()

    scree_plot_path = artifacts_dir / "pca__scree.png"
    plt.savefig(scree_plot_path, dpi=200)
    plt.show()
    plt.close()

    pc1_loading_plot_path = None
    pc2_loading_plot_path = None

    for component_name in ["PC1", "PC2"]:
        if component_name not in loadings_df.columns:
            continue

        top_loadings = (
            loadings_df[["feature", component_name]]
            .assign(abs_loading=lambda x: x[component_name].abs())
            .sort_values(["abs_loading", "feature"], ascending=[False, True])
            .head(top_loading_count)
            .sort_values(component_name, ascending=True)
        )

        plt.figure(figsize=(8, max(4, int(len(top_loadings) * 0.35) + 1)))
        plt.barh(top_loadings["feature"], top_loadings[component_name])
        plt.xlabel("Loading")
        plt.ylabel("Feature")
        plt.title(f"PCA loadings: {component_name}")
        plt.tight_layout()

        out_path = artifacts_dir / f"pca__{component_name.lower()}_loadings.png"
        plt.savefig(out_path, dpi=200)
        plt.show()
        plt.close()

        if component_name == "PC1":
            pc1_loading_plot_path = str(out_path)
        elif component_name == "PC2":
            pc2_loading_plot_path = str(out_path)

    return {
        "explained_variance_df": explained_variance_df,
        "loadings_df": loadings_df,
        "explained_variance_path": str(explained_variance_path),
        "loadings_path": str(loadings_path),
        "scree_plot_path": str(scree_plot_path),
        "pc1_loading_plot_path": pc1_loading_plot_path,
        "pc2_loading_plot_path": pc2_loading_plot_path,
    }


def build_outlier_audit_artifacts(
    dataframe: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    artifacts_dir: Path,
    state_column: Optional[str] = None,
    iqr_multiplier: float = 1.5,
    robust_z_threshold: float = 3.5,
    max_plot_features: int = 20,
) -> dict:
    """
    Build overall and optional state-level outlier summary artifacts.
    """
    available_features = [
        feature_name
        for feature_name in feature_columns
        if feature_name in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[feature_name])
    ]

    overall_rows = []
    state_rows = []

    for feature_name in available_features:
        feature_series = _safe_numeric_series(dataframe[feature_name])
        clean = feature_series.dropna()

        if clean.empty:
            continue

        lower, upper, iqr_value = _iqr_bounds(clean, iqr_multiplier=iqr_multiplier)
        median_value = float(clean.median())
        mad_value = _mad(clean)

        if lower is not None and upper is not None:
            iqr_mask = (clean < lower) | (clean > upper)
            iqr_outlier_count = int(iqr_mask.sum())
            iqr_outlier_pct = float(iqr_outlier_count / len(clean))
        else:
            iqr_outlier_count = 0
            iqr_outlier_pct = 0.0

        robust_z_outlier_count = 0
        robust_z_outlier_pct = 0.0
        if mad_value is not None and mad_value > 0:
            robust_scores = 0.6745 * (clean - median_value) / mad_value
            robust_mask = robust_scores.abs() > robust_z_threshold
            robust_z_outlier_count = int(robust_mask.sum())
            robust_z_outlier_pct = float(robust_z_outlier_count / len(clean))

        overall_rows.append(
            {
                "feature": str(feature_name),
                "non_null_count": int(len(clean)),
                "median": median_value,
                "iqr": float(iqr_value) if iqr_value is not None else None,
                "iqr_outlier_count": iqr_outlier_count,
                "iqr_outlier_pct": iqr_outlier_pct,
                "robust_z_outlier_count": robust_z_outlier_count,
                "robust_z_outlier_pct": robust_z_outlier_pct,
                "zero_pct": float((clean == 0).mean()),
            }
        )

        if state_column is not None and state_column in dataframe.columns:
            state_values = dataframe[state_column].dropna().astype(str).unique().tolist()
            for state_value in state_values:
                state_clean = _safe_numeric_series(
                    dataframe.loc[dataframe[state_column].astype(str).eq(str(state_value)), feature_name]
                ).dropna()

                if state_clean.empty:
                    continue

                state_lower, state_upper, state_iqr = _iqr_bounds(state_clean, iqr_multiplier=iqr_multiplier)
                if state_lower is not None and state_upper is not None:
                    state_iqr_mask = (state_clean < state_lower) | (state_clean > state_upper)
                    state_iqr_count = int(state_iqr_mask.sum())
                    state_iqr_pct = float(state_iqr_count / len(state_clean))
                else:
                    state_iqr_count = 0
                    state_iqr_pct = 0.0

                state_rows.append(
                    {
                        "state": str(state_value),
                        "feature": str(feature_name),
                        "non_null_count": int(len(state_clean)),
                        "iqr": float(state_iqr) if state_iqr is not None else None,
                        "iqr_outlier_count": state_iqr_count,
                        "iqr_outlier_pct": state_iqr_pct,
                    }
                )

    overall_df = pd.DataFrame(overall_rows)
    state_df = pd.DataFrame(state_rows)

    overall_path = artifacts_dir / "outliers__feature_summary.csv"
    overall_df.to_csv(overall_path, index=False)

    state_path = None
    if not state_df.empty:
        state_path = artifacts_dir / "outliers__by_state.csv"
        state_df.to_csv(state_path, index=False)
        state_path = str(state_path)

    outlier_plot_path = None
    if not overall_df.empty:
        plot_df = (
            overall_df.sort_values(["iqr_outlier_pct", "feature"], ascending=[False, True])
            .head(max_plot_features)
            .sort_values("iqr_outlier_pct", ascending=True)
        )

        plt.figure(figsize=(9, max(4, int(len(plot_df) * 0.35) + 1)))
        plt.barh(plot_df["feature"], plot_df["iqr_outlier_pct"])
        plt.xlabel("IQR outlier rate")
        plt.ylabel("Feature")
        plt.title("Top features by IQR outlier rate")
        plt.tight_layout()

        out_path = artifacts_dir / "outliers__top_iqr_rate.png"
        plt.savefig(out_path, dpi=200)
        plt.show()
        plt.close()
        outlier_plot_path = str(out_path)

    return {
        "overall_df": overall_df,
        "state_df": state_df,
        "overall_path": str(overall_path),
        "state_path": state_path,
        "outlier_plot_path": outlier_plot_path,
    }