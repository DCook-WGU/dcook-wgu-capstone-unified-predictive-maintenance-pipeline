from __future__ import annotations

import logging

from pathlib import Path
import pandas as pd


def profile_dataframe(
        df: pd.DataFrame, 
        logger: logging.Logger, 
        artifacts_dir: Path | None = None, 
        head: int = 15
    ) -> tuple[dict, dict]:


    metrics = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
    }

    logger.info("Shape: %s", df.shape)
    logger.info("Memory (MB): %.2f", metrics["memory_mb"])
    logger.info("Dtypes:\n%s", df.dtypes.to_string())
    logger.info("Head(%d):\n%s", head, df.head(head).to_string(max_cols=40, max_rows=head))

    saved = {}

    if artifacts_dir is not None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        num_path = artifacts_dir / "describe_numeric.csv"
        obj_path = artifacts_dir / "describe_object.csv"

        df.describe().T.to_csv(num_path)
        df.describe(include=["object", "category"]).T.to_csv(obj_path)

        saved["describe_numeric_csv"] = num_path
        saved["describe_object_csv"] = obj_path

        logger.info("Saved describe artifacts to: %s", artifacts_dir)

    return metrics, saved
