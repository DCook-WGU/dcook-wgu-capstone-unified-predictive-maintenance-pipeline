from __future__ import annotations

import logging

from pathlib import Path
import pandas as pd


def profile_dataframe(
        dataframe: pd.DataFrame, 
        logger: logging.Logger, 
        artifacts_dir: Path | None = None, 
        head: int = 15
    ) -> tuple[dict, dict]:


    metrics = {
        "rows": int(dataframe.shape[0]),
        "cols": int(dataframe.shape[1]),
        "memory_mb": float(dataframe.memory_usage(deep=True).sum() / (1024**2)),
    }

    logger.info("Shape: %s", dataframe.shape)
    logger.info("Memory (MB): %.2f", metrics["memory_mb"])
    logger.info("Dtypes:\n%s", dataframe.dtypes.to_string())
    logger.info("Head(%d):\n%s", head, dataframe.head(head).to_string(max_cols=40, max_rows=head))

    saved = {}

    if artifacts_dir is not None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        num_path = artifacts_dir / "describe_numeric.csv"
        obj_path = artifacts_dir / "describe_object.csv"
        bool_path = artifacts_dir / "describe_boolean.csv"

        dataframe.describe().T.to_csv(num_path)
        dataframe.describe(include=["object", "category", "string"]).T.to_csv(obj_path)
        dataframe.describe(include=["bool", "boolean"]).T.to_csv(bool_path)

        saved["describe_numeric_csv"] = num_path
        saved["describe_object_csv"] = obj_path
        saved["describe_boolean_csv"] = bool_path

        logger.info("Saved describe artifacts to: %s", artifacts_dir)

    return metrics, saved




#bool_df = df.select_dtypes(include=["bool", "boolean"])
#if bool_df.shape[1] > 0:
#    bool_path = artifacts_dir / "describe_bool.csv"
#    bool_df.describe().T.to_csv(bool_path)
#    logger.info("Saved boolean describe to %s", bool_path)
#else:
#    logger.info("No boolean columns; skipping boolean describe.")