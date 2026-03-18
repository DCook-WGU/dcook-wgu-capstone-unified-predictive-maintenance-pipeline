from __future__ import annotations

from pathlib import Path
import pandas as pd

from utils.postgres_util import get_engine_from_env, read_sql_dataframe
from utils.file_io import save_data


def export_synthetic_batch_to_parquet(
    *,
    dataset_name: str,
    batch_id: int,
    out_dir: Path,
    schema: str = "public",
    artifact_name: str = "stream",
) -> Path:
    engine = get_engine_from_env()

    table = f'{schema}."synthetic_{dataset_name}_{artifact_name}"'

    sql = f"""
    SELECT *
    FROM {table}
    WHERE batch_id = {int(batch_id)}
    ORDER BY row_in_batch
    """

    df = read_sql_dataframe(engine, sql)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_name}__synthetic__batch_{batch_id}.parquet"

    # save_data(dataframe, file_path, file_name=None)
    save_data(df, out_dir, out_path.name)

    return out_path