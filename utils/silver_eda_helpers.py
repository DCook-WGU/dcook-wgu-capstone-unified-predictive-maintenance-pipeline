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

