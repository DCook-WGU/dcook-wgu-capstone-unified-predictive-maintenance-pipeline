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

        numeric_path = artifacts_dir / "eda_logging_dataframe_profile_describe_numeric.csv"
        object_path = artifacts_dir / "eda_logging_dataframe_profile_describe_object.csv"
        boolean_path = artifacts_dir / "eda_logging_dataframe_profile_describe_boolean.csv"

        numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_columns) > 0:
            numeric_path = artifacts_dir / "describe_numeric.csv"
            dataframe[numeric_columns].describe().T.to_csv(numeric_path)
            saved["describe_numeric_csv"] = str(numeric_path)
        else:
            logger.info("No numeric columns to describe; skipping numeric describe.")


        object_category_columns = dataframe.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if len(object_category_columns) > 0:
            object_path = artifacts_dir / "describe_object.csv"
            dataframe[object_category_columns].describe().T.to_csv(object_path)
            saved["describe_object_csv"] = str(object_path)
        else:
            logger.info("No object/category/string columns to describe; skipping.")

        boolean_columns = dataframe.select_dtypes(include=["bool", "boolean"]).columns.tolist()
        if len(boolean_columns) > 0:
            boolean_path = artifacts_dir / "describe_boolean.csv"
            dataframe[boolean_columns].describe().T.to_csv(boolean_path)
            saved["describe_boolean_csv"] = str(boolean_path)
        else:
            logger.info("No boolean columns to describe; skipping.")

        logger.info("Saved describe artifacts to: %s", artifacts_dir)

    return metrics, saved




#bool_df = df.select_dtypes(include=["bool", "boolean"])
#if bool_df.shape[1] > 0:
#    bool_path = artifacts_dir / "describe_bool.csv"
#    bool_df.describe().T.to_csv(bool_path)
#    logger.info("Saved boolean describe to %s", bool_path)
#else:
#    logger.info("No boolean columns; skipping boolean describe.")