from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from scipy import stats
except ImportError:
    stats = None

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
except ImportError:
    PCA = None
    StandardScaler = None
    KMeans = None
    IsolationForest = None

warnings.filterwarnings("ignore")


# ======================================================================================
# CONFIG / RESULTS OBJECTS
# ======================================================================================

@dataclass
class EDAConfig:
    dataset_name: str = "dataset"
    target_column: Optional[str] = None
    datetime_columns: Optional[List[str]] = None
    id_columns: Optional[List[str]] = None
    exclude_columns: Optional[List[str]] = None
    sample_size_for_pairplot: int = 1000
    max_categories_to_plot: int = 20
    top_n_correlations: int = 25
    run_outlier_detection: bool = True
    run_pca: bool = True
    run_clustering: bool = False
    clustering_k: int = 3
    save_outputs: bool = False
    output_dir: str = "eda_outputs"
    figure_dpi: int = 120
    random_state: int = 42


@dataclass
class EDAResults:
    overview: Dict[str, Any] = field(default_factory=dict)
    column_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    missing_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    numeric_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    categorical_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    datetime_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    outlier_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_correlations: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    pca_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    clustering_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: List[str] = field(default_factory=list)


# ======================================================================================
# CORE HELPERS
# ======================================================================================

def ensure_output_dir(config: EDAConfig) -> Path:
    output_path = Path(config.output_dir)
    if config.save_outputs:
        output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def infer_datetime_columns(dataframe: pd.DataFrame, config: EDAConfig) -> List[str]:
    if config.datetime_columns is not None:
        return [column for column in config.datetime_columns if column in dataframe.columns]

    detected = []
    for column in dataframe.columns:
        if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
            detected.append(column)
    return detected


def get_excluded_columns(config: EDAConfig) -> List[str]:
    excluded = []
    if config.exclude_columns:
        excluded.extend(config.exclude_columns)
    if config.id_columns:
        excluded.extend(config.id_columns)
    return list(dict.fromkeys(excluded))


def get_numeric_columns(dataframe: pd.DataFrame, excluded_columns: Optional[List[str]] = None) -> List[str]:
    excluded_columns = excluded_columns or []
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if column not in excluded_columns]


def get_categorical_columns(
    dataframe: pd.DataFrame,
    datetime_columns: Optional[List[str]] = None,
    excluded_columns: Optional[List[str]] = None,
) -> List[str]:
    datetime_columns = datetime_columns or []
    excluded_columns = excluded_columns or []

    categorical_columns = dataframe.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    final_columns = [
        column
        for column in categorical_columns
        if column not in datetime_columns and column not in excluded_columns
    ]
    return final_columns


def safe_mode(series: pd.Series) -> Any:
    mode_values = series.mode(dropna=True)
    if len(mode_values) == 0:
        return np.nan
    return mode_values.iloc[0]


def coefficient_of_variation(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) == 0:
        return np.nan
    mean_value = clean.mean()
    if mean_value == 0:
        return np.nan
    return clean.std() / mean_value


def mad_value(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) == 0:
        return np.nan
    median_value = clean.median()
    return np.median(np.abs(clean - median_value))


def iqr_bounds(series: pd.Series) -> Tuple[float, float]:
    clean = series.dropna()
    if len(clean) == 0:
        return (np.nan, np.nan)
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def save_figure(config: EDAConfig, filename: str) -> None:
    if config.save_outputs:
        output_path = ensure_output_dir(config)
        plt.savefig(output_path / filename, dpi=config.figure_dpi, bbox_inches="tight")
    plt.show()
    plt.close()


# ======================================================================================
# STRUCTURAL OVERVIEW
# ======================================================================================

def build_overview(dataframe: pd.DataFrame, config: EDAConfig) -> Dict[str, Any]:
    overview = {
        "dataset_name": config.dataset_name,
        "n_rows": int(dataframe.shape[0]),
        "n_columns": int(dataframe.shape[1]),
        "memory_usage_mb": float(dataframe.memory_usage(deep=True).sum() / (1024 ** 2)),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "total_missing_cells": int(dataframe.isna().sum().sum()),
        "total_missing_percent": float(
            (dataframe.isna().sum().sum() / (dataframe.shape[0] * dataframe.shape[1]) * 100)
            if dataframe.shape[0] > 0 and dataframe.shape[1] > 0
            else 0.0
        ),
    }
    return overview


def build_column_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    records = []

    for column in dataframe.columns:
        series = dataframe[column]
        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "null_percent": float(series.isna().mean() * 100),
                "unique_count": int(series.nunique(dropna=True)),
                "is_constant": bool(series.nunique(dropna=True) <= 1),
                "sample_value_1": series.dropna().iloc[0] if series.notna().any() else np.nan,
            }
        )

    summary = pd.DataFrame(records).sort_values(["null_percent", "unique_count"], ascending=[False, False])
    return summary.reset_index(drop=True)


# ======================================================================================
# MISSINGNESS
# ======================================================================================

def build_missing_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing_df = pd.DataFrame({
        "column": dataframe.columns,
        "null_count": dataframe.isna().sum().values,
        "null_percent": (dataframe.isna().mean() * 100).values,
    })
    missing_df = missing_df.sort_values(["null_percent", "null_count"], ascending=[False, False]).reset_index(drop=True)
    return missing_df


def plot_missingness_bar(dataframe: pd.DataFrame, config: EDAConfig) -> None:
    missing_summary = build_missing_summary(dataframe)
    missing_summary = missing_summary[missing_summary["null_count"] > 0]

    if missing_summary.empty:
        print("No missing values found.")
        return

    plt.figure(figsize=(12, 6))
    plt.bar(missing_summary["column"], missing_summary["null_percent"])
    plt.xticks(rotation=90)
    plt.ylabel("Missing %")
    plt.title(f"{config.dataset_name} - Missingness by Column")
    plt.tight_layout()
    save_figure(config, "missingness_bar.png")


# ======================================================================================
# DESCRIPTIVE STATISTICS
# ======================================================================================

def build_numeric_summary(dataframe: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    records = []

    for column in numeric_columns:
        series = dataframe[column].dropna()
        if len(series) == 0:
            continue

        lower_bound, upper_bound = iqr_bounds(series)

        record = {
            "column": column,
            "count": int(series.count()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "mode": safe_mode(series),
            "std": float(series.std()),
            "variance": float(series.var()),
            "min": float(series.min()),
            "q1": float(series.quantile(0.25)),
            "q3": float(series.quantile(0.75)),
            "max": float(series.max()),
            "range": float(series.max() - series.min()),
            "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurt()),
            "cv": float(coefficient_of_variation(series)),
            "mad": float(mad_value(series)),
            "zero_count": int((series == 0).sum()),
            "negative_count": int((series < 0).sum()),
            "iqr_lower_bound": float(lower_bound),
            "iqr_upper_bound": float(upper_bound),
            "iqr_outlier_count": int(((series < lower_bound) | (series > upper_bound)).sum()),
        }
        records.append(record)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("iqr_outlier_count", ascending=False).reset_index(drop=True)


def build_categorical_summary(dataframe: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    records = []

    for column in categorical_columns:
        series = dataframe[column]
        clean = series.dropna()

        if len(clean) == 0:
            continue

        value_counts = clean.value_counts(dropna=True)
        top_category = value_counts.index[0] if len(value_counts) > 0 else np.nan
        top_frequency = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
        top_percent = float((top_frequency / len(clean)) * 100) if len(clean) > 0 else np.nan

        records.append(
            {
                "column": column,
                "count": int(clean.count()),
                "unique_count": int(clean.nunique(dropna=True)),
                "mode": safe_mode(clean),
                "top_category": top_category,
                "top_frequency": top_frequency,
                "top_percent": top_percent,
                "rare_category_count_lt_1pct": int((value_counts / len(clean) < 0.01).sum()),
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values(["unique_count", "top_percent"], ascending=[False, False]).reset_index(drop=True)


def build_datetime_summary(dataframe: pd.DataFrame, datetime_columns: List[str]) -> pd.DataFrame:
    records = []

    for column in datetime_columns:
        series = pd.to_datetime(dataframe[column], errors="coerce").dropna()
        if len(series) == 0:
            continue

        records.append(
            {
                "column": column,
                "count": int(series.count()),
                "min_datetime": series.min(),
                "max_datetime": series.max(),
                "date_span_days": int((series.max() - series.min()).days),
                "unique_timestamps": int(series.nunique()),
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("date_span_days", ascending=False).reset_index(drop=True)


# ======================================================================================
# UNIVARIATE PLOTS
# ======================================================================================

def plot_numeric_distributions(
    dataframe: pd.DataFrame,
    numeric_columns: List[str],
    config: EDAConfig,
    max_plots: Optional[int] = None,
) -> None:
    plot_columns = numeric_columns[:max_plots] if max_plots is not None else numeric_columns

    for column in plot_columns:
        plt.figure(figsize=(10, 5))
        plt.hist(dataframe[column].dropna(), bins=30)
        plt.title(f"{config.dataset_name} - Histogram: {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        save_figure(config, f"hist_{column}.png")

        plt.figure(figsize=(10, 4))
        plt.boxplot(dataframe[column].dropna(), vert=False)
        plt.title(f"{config.dataset_name} - Boxplot: {column}")
        plt.xlabel(column)
        save_figure(config, f"box_{column}.png")


def plot_categorical_counts(
    dataframe: pd.DataFrame,
    categorical_columns: List[str],
    config: EDAConfig,
    max_plots: Optional[int] = None,
) -> None:
    plot_columns = categorical_columns[:max_plots] if max_plots is not None else categorical_columns

    for column in plot_columns:
        counts = dataframe[column].astype(str).value_counts(dropna=False).head(config.max_categories_to_plot)

        plt.figure(figsize=(12, 5))
        plt.bar(counts.index.astype(str), counts.values)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{config.dataset_name} - Count Plot: {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.tight_layout()
        save_figure(config, f"count_{column}.png")


# ======================================================================================
# CORRELATION / ASSOCIATION
# ======================================================================================

def build_correlation_matrix(dataframe: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    if len(numeric_columns) < 2:
        return pd.DataFrame()
    return dataframe[numeric_columns].corr(numeric_only=True)


def build_top_correlations(correlation_matrix: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if correlation_matrix.empty:
        return pd.DataFrame()

    corr_pairs = (
        correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["feature_1", "feature_2", "correlation"]
    corr_pairs["abs_correlation"] = corr_pairs["correlation"].abs()

    corr_pairs = corr_pairs.sort_values("abs_correlation", ascending=False).head(top_n).reset_index(drop=True)
    return corr_pairs


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, config: EDAConfig) -> None:
    if correlation_matrix.empty:
        print("Correlation heatmap skipped: not enough numeric columns.")
        return

    plt.figure(figsize=(12, 10))
    if sns is not None:
        sns.heatmap(correlation_matrix, cmap="coolwarm", center=0)
    else:
        plt.imshow(correlation_matrix, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
        plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    plt.title(f"{config.dataset_name} - Correlation Heatmap")
    plt.tight_layout()
    save_figure(config, "correlation_heatmap.png")


# ======================================================================================
# BIVARIATE / TARGET-AWARE
# ======================================================================================

def build_target_summary(
    dataframe: pd.DataFrame,
    config: EDAConfig,
    numeric_columns: List[str],
) -> pd.DataFrame:
    if config.target_column is None or config.target_column not in dataframe.columns:
        return pd.DataFrame()

    target_series = dataframe[config.target_column]
    records = []

    if pd.api.types.is_numeric_dtype(target_series):
        for column in numeric_columns:
            if column == config.target_column:
                continue

            pair_df = dataframe[[column, config.target_column]].dropna()
            if len(pair_df) < 2:
                continue

            corr_value = pair_df[column].corr(pair_df[config.target_column])
            records.append(
                {
                    "feature": column,
                    "relationship_type": "numeric_vs_numeric",
                    "pearson_correlation_to_target": corr_value,
                }
            )
    else:
        target_values = target_series.dropna().astype(str).value_counts()
        for target_label in target_values.index:
            records.append(
                {
                    "feature": config.target_column,
                    "relationship_type": "target_class_distribution",
                    "target_class": target_label,
                    "count": int(target_values[target_label]),
                    "percent": float(target_values[target_label] / len(target_series.dropna()) * 100),
                }
            )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def plot_target_distributions(
    dataframe: pd.DataFrame,
    config: EDAConfig,
    numeric_columns: List[str],
    max_plots: int = 10,
) -> None:
    if config.target_column is None or config.target_column not in dataframe.columns:
        return

    target_series = dataframe[config.target_column]

    if not pd.api.types.is_numeric_dtype(target_series):
        counts = target_series.astype(str).value_counts(dropna=False)

        plt.figure(figsize=(10, 5))
        plt.bar(counts.index.astype(str), counts.values)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{config.dataset_name} - Target Distribution: {config.target_column}")
        plt.ylabel("Count")
        plt.tight_layout()
        save_figure(config, "target_distribution.png")

        if sns is not None:
            usable_numeric = [col for col in numeric_columns if col != config.target_column][:max_plots]
            for column in usable_numeric:
                plt.figure(figsize=(10, 5))
                sns.boxplot(data=dataframe, x=config.target_column, y=column)
                plt.xticks(rotation=45, ha="right")
                plt.title(f"{config.dataset_name} - {column} by {config.target_column}")
                plt.tight_layout()
                save_figure(config, f"target_box_{column}.png")


# ======================================================================================
# OUTLIER DETECTION
# ======================================================================================

def build_outlier_summary(dataframe: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    records = []

    for column in numeric_columns:
        series = dataframe[column].dropna()
        if len(series) == 0:
            continue

        lower_bound, upper_bound = iqr_bounds(series)
        outlier_mask = (series < lower_bound) | (series > upper_bound)

        z_outlier_count = np.nan
        if stats is not None and len(series) > 2 and series.std() != 0:
            z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
            z_outlier_count = int((z_scores > 3).sum())

        records.append(
            {
                "column": column,
                "iqr_lower_bound": float(lower_bound),
                "iqr_upper_bound": float(upper_bound),
                "iqr_outlier_count": int(outlier_mask.sum()),
                "iqr_outlier_percent": float(outlier_mask.mean() * 100),
                "zscore_outlier_count_gt3": z_outlier_count,
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("iqr_outlier_count", ascending=False).reset_index(drop=True)


def run_isolation_forest_outliers(
    dataframe: pd.DataFrame,
    numeric_columns: List[str],
    config: EDAConfig,
) -> pd.DataFrame:
    if IsolationForest is None or not numeric_columns:
        return pd.DataFrame()

    model_df = dataframe[numeric_columns].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan)

    if model_df.dropna().empty:
        return pd.DataFrame()

    model_df = model_df.fillna(model_df.median(numeric_only=True))

    model = IsolationForest(
        contamination="auto",
        random_state=config.random_state,
    )
    predictions = model.fit_predict(model_df)
    scores = model.decision_function(model_df)

    result_df = pd.DataFrame({
        "row_index": dataframe.index,
        "isolation_forest_label": predictions,
        "isolation_forest_score": scores,
    })
    result_df["is_anomaly"] = result_df["isolation_forest_label"] == -1
    return result_df.sort_values("isolation_forest_score").reset_index(drop=True)


# ======================================================================================
# PCA / CLUSTERING
# ======================================================================================

def run_pca_summary(
    dataframe: pd.DataFrame,
    numeric_columns: List[str],
    config: EDAConfig,
) -> pd.DataFrame:
    if PCA is None or StandardScaler is None or len(numeric_columns) < 2:
        return pd.DataFrame()

    model_df = dataframe[numeric_columns].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.fillna(model_df.median(numeric_only=True))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(model_df)

    pca = PCA()
    transformed = pca.fit_transform(scaled)

    explained = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })

    if transformed.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(transformed[:, 0], transformed[:, 1], alpha=0.6)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{config.dataset_name} - PCA Projection")
        save_figure(config, "pca_scatter.png")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"{config.dataset_name} - PCA Explained Variance")
    plt.grid(True, alpha=0.3)
    save_figure(config, "pca_explained_variance.png")

    return explained


def run_kmeans_summary(
    dataframe: pd.DataFrame,
    numeric_columns: List[str],
    config: EDAConfig,
) -> pd.DataFrame:
    if KMeans is None or StandardScaler is None or len(numeric_columns) < 2:
        return pd.DataFrame()

    model_df = dataframe[numeric_columns].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.fillna(model_df.median(numeric_only=True))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(model_df)

    model = KMeans(n_clusters=config.clustering_k, random_state=config.random_state, n_init=10)
    cluster_labels = model.fit_predict(scaled)

    cluster_summary = pd.DataFrame({
        "row_index": dataframe.index,
        "cluster_label": cluster_labels,
    })

    size_summary = cluster_summary["cluster_label"].value_counts().sort_index().reset_index()
    size_summary.columns = ["cluster_label", "count"]

    if PCA is not None:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(scaled)

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, alpha=0.6)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"{config.dataset_name} - KMeans Cluster Projection")
        save_figure(config, "kmeans_clusters.png")

    return size_summary


# ======================================================================================
# TIME SERIES EDA
# ======================================================================================

def plot_datetime_counts(dataframe: pd.DataFrame, datetime_columns: List[str], config: EDAConfig) -> None:
    for column in datetime_columns:
        series = pd.to_datetime(dataframe[column], errors="coerce").dropna()
        if len(series) == 0:
            continue

        counts = series.dt.date.value_counts().sort_index()

        plt.figure(figsize=(12, 5))
        plt.plot(counts.index, counts.values)
        plt.xticks(rotation=45)
        plt.title(f"{config.dataset_name} - Observation Counts Over Time: {column}")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.tight_layout()
        save_figure(config, f"datetime_counts_{column}.png")


# ======================================================================================
# REPORTING
# ======================================================================================

def print_eda_summary(results: EDAResults) -> None:
    print("=" * 80)
    print("EDA SUMMARY")
    print("=" * 80)

    if results.overview:
        for key, value in results.overview.items():
            print(f"{key}: {value}")

    print("\nTop missing columns:")
    if not results.missing_summary.empty:
        print(results.missing_summary.head(10).to_string(index=False))
    else:
        print("No missing summary available.")

    print("\nTop numeric outlier columns:")
    if not results.outlier_summary.empty:
        print(results.outlier_summary.head(10).to_string(index=False))
    else:
        print("No outlier summary available.")

    print("\nTop correlations:")
    if not results.top_correlations.empty:
        print(results.top_correlations.head(10).to_string(index=False))
    else:
        print("No top correlations available.")

    if results.notes:
        print("\nNotes:")
        for note in results.notes:
            print(f"- {note}")


def save_tables(results: EDAResults, config: EDAConfig) -> None:
    if not config.save_outputs:
        return

    output_path = ensure_output_dir(config)

    table_map = {
        "column_summary.csv": results.column_summary,
        "missing_summary.csv": results.missing_summary,
        "numeric_summary.csv": results.numeric_summary,
        "categorical_summary.csv": results.categorical_summary,
        "datetime_summary.csv": results.datetime_summary,
        "outlier_summary.csv": results.outlier_summary,
        "correlation_matrix.csv": results.correlation_matrix,
        "top_correlations.csv": results.top_correlations,
        "target_summary.csv": results.target_summary,
        "pca_summary.csv": results.pca_summary,
        "clustering_summary.csv": results.clustering_summary,
    }

    for filename, dataframe in table_map.items():
        if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
            dataframe.to_csv(output_path / filename, index=False)


# ======================================================================================
# ORCHESTRATOR
# ======================================================================================

def run_universal_eda(dataframe: pd.DataFrame, config: Optional[EDAConfig] = None) -> EDAResults:
    if config is None:
        config = EDAConfig()

    results = EDAResults()

    excluded_columns = get_excluded_columns(config)
    datetime_columns = infer_datetime_columns(dataframe, config)
    numeric_columns = get_numeric_columns(dataframe, excluded_columns=excluded_columns)
    categorical_columns = get_categorical_columns(
        dataframe,
        datetime_columns=datetime_columns,
        excluded_columns=excluded_columns,
    )

    results.overview = build_overview(dataframe, config)
    results.column_summary = build_column_summary(dataframe)
    results.missing_summary = build_missing_summary(dataframe)
    results.numeric_summary = build_numeric_summary(dataframe, numeric_columns)
    results.categorical_summary = build_categorical_summary(dataframe, categorical_columns)
    results.datetime_summary = build_datetime_summary(dataframe, datetime_columns)
    results.correlation_matrix = build_correlation_matrix(dataframe, numeric_columns)
    results.top_correlations = build_top_correlations(
        results.correlation_matrix,
        top_n=config.top_n_correlations,
    )
    results.target_summary = build_target_summary(dataframe, config, numeric_columns)

    if config.run_outlier_detection:
        results.outlier_summary = build_outlier_summary(dataframe, numeric_columns)

    if config.run_pca:
        results.pca_summary = run_pca_summary(dataframe, numeric_columns, config)

    if config.run_clustering:
        results.clustering_summary = run_kmeans_summary(dataframe, numeric_columns, config)

    # Plots
    plot_missingness_bar(dataframe, config)
    plot_numeric_distributions(dataframe, numeric_columns, config, max_plots=10)
    plot_categorical_counts(dataframe, categorical_columns, config, max_plots=10)
    plot_correlation_heatmap(results.correlation_matrix, config)
    plot_target_distributions(dataframe, config, numeric_columns, max_plots=10)
    plot_datetime_counts(dataframe, datetime_columns, config)

    save_tables(results, config)
    print_eda_summary(results)

    return results