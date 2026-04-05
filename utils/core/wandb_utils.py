from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd
import logging

from utils.eda_logging import profile_dataframe


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def _require_wandb():
    """
    Import wandb with a clear error message if not installed.
    """
    try:
        import wandb  # type: ignore
        return wandb
    except Exception as e:
        raise ImportError(
            "wandb is not available in this environment. "
            "Install it (pip install wandb) and ensure your kernel/container uses that environment."
        ) from e


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def _sanitize_dataframe_for_wandb_table(frame: pd.DataFrame) -> pd.DataFrame:
    sanitized = frame.copy()

    # Convert datetime/tz-aware datetime columns to strings 
    for column in sanitized.columns:
        if pd.api.types.is_datetime64_any_dtype(sanitized[column]) or pd.api.types.is_datetime64tz_dtype(sanitized[column]):
            sanitized[column] = sanitized[column].astype("string")

    # Convert pd.NA / NaN -> None (JSON serializable)
    sanitized = sanitized.astype(object).where(pd.notna(sanitized), None)

    return sanitized

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def log_metrics(run: Any, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
    """
    Log scalar metrics (or small JSON-serializable values) to an active W&B run.
    """
    wandb = _require_wandb()
    if run is None:
        raise ValueError("run is None. Call wandb.init(...) in your notebook/entry point first.")
    wandb.log(metrics, step=step, commit=commit)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def log_dataframe_head(
    run: Any,
    dataframe: pd.DataFrame,
    key: str = "head",
    n: int = 15,
    *,
    max_rows: int = 200,
) -> None:
    """
    Log a small DataFrame sample as a W&B Table.
    Defaults to head(n), and hard-caps rows to avoid huge uploads.
    """
    wandb = _require_wandb()
    if run is None:
        raise ValueError("run is None. Call wandb.init(...) in your notebook/entry point first.")

    sample = dataframe.head(n)

    if len(sample) > max_rows:
        sample = sample.head(max_rows)

    sample = _sanitize_dataframe_for_wandb_table(sample)

    table = wandb.Table(dataframe=sample)
    wandb.log({key: table})

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def log_text(
    run: Any,
    key: str,
    text: str,
) -> None:
    """
    Log a text blob. Useful for small notes, dataset provenance strings, etc.
    """
    wandb = _require_wandb()
    if run is None:
        raise ValueError("run is None. Call wandb.init(...) in your notebook/entry point first.")
    wandb.log({key: text})

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def log_files_as_artifact(
    run: Any,
    *,
    artifact_name: str,
    artifact_type: str,
    files: Sequence[Union[str, Path]],
    aliases: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a W&B artifact and attach specific files.

    Returns:
        The created artifact object.
    """
    wandb = _require_wandb()
    if run is None:
        raise ValueError("run is None. Call wandb.init(...) in your notebook/entry point first.")

    artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=metadata or {})

    added_any = False
    for file in files:
        file_path = Path(file)
        if file_path.exists() and file_path.is_file():
            artifact.add_file(str(file_path))
            added_any = True

    if not added_any:
        raise FileNotFoundError(
            f"No valid files found to add to artifact '{artifact_name}'. "
            f"Checked: {[str(Path(file)) for file in files]}"
        )

    run.log_artifact(artifact, aliases=list(aliases) if aliases else None)
    return artifact

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

def log_dir_as_artifact(
    run: Any,
    *,
    artifact_name: str,
    artifact_type: str,
    dir_path: Union[str, Path],
    patterns: Sequence[str] = ("*.parquet", "*.pq", "*.csv", "*.json", "*.log", "*.txt"),
    aliases: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    recursive: bool = True,
) -> Any:
    """
    Create a W&B artifact and attach files from a directory matching glob patterns.

    Example:
        log_dir_as_artifact(run, artifact_name="silver-artifacts", artifact_type="eda",
                            dir_path=paths.root/"artifacts"/"silver",
                            patterns=("*.csv","*.log"), recursive=True)
    """
    wandb = _require_wandb()
    if run is None:
        raise ValueError("run is None. Call wandb.init(...) in your notebook/entry point first.")

    directory_path = Path(dir_path)
    if not directory_path.exists() or not directory_path.is_dir():
        raise NotADirectoryError(f"dir_path is not a directory: {directory_path}")

    artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=metadata or {})

    matches: list[Path] = []
    for pattern in patterns:
        if recursive:
            matches.extend(directory_path.rglob(pattern))
        else:
            matches.extend(directory_path.glob(pattern))

    matches = [file_path for file_path in matches if file_path.is_file()]

    if not matches:
        raise FileNotFoundError(f"No files matched patterns {patterns} in {directory_path}")

    for file_path in matches:
        artifact.add_file(str(file_path))

    run.log_artifact(artifact, aliases=list(aliases) if aliases else None)
    return artifact


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 


def finalize_wandb_stage(
    run: Any,
    *,
    stage: str,
    dataframe: pd.DataFrame,
    project_root: Union[str, Path],
    logs_dir: Union[str, Path],
    dataset_dirs: Sequence[Union[str, Path]],
    dataset_artifact_name: str,
    logger: Optional[logging.Logger] = None,
    notebook_path: Optional[Union[str, Path]] = None,
    aliases: Sequence[str] = ("latest",),
    table_key: Optional[str] = None,
    table_n: int = 15,
    profile: bool = True,
    diagnostics_patterns: Sequence[str] = ("*.csv", "*.json", "*.txt", "*.log"),
    parquet_patterns: Sequence[str] = ("*.parquet", "*.pq"),
) -> Dict[str, Any]:
    """
    End-of-stage W&B finalizer:
      - optionally profiles df and saves CSVs into project_root/artifacts/<stage>/
      - logs metrics + a head table
      - uploads:
          * logs/<stage>.log
          * parquet outputs from dataset_dirs
          * diagnostics from artifacts/<stage>/
          * optional notebook .ipynb

    Returns a dict with paths used + computed metrics.
    """
    # Local import to avoid hard dependency if you call this without profiling.
    if logger is None:
        logger = logging.getLogger(f"capstone.{stage}")

    project_root = Path(project_root)
    logs_dir = Path(logs_dir)

    artifacts_stage_dir = project_root / "artifacts" / stage
    artifacts_stage_dir.mkdir(parents=True, exist_ok=True)

    # --- compute + (optionally) export profiling artifacts ---
    metrics: Dict[str, Any] = {
        "rows": int(dataframe.shape[0]),
        "cols": int(dataframe.shape[1]),
        "memory_mb": float(dataframe.memory_usage(deep=True).sum() / (1024**2)),
    }
    saved: Dict[str, Path] = {}

    if profile:
        try:
            
            prof_metrics, prof_saved = profile_dataframe(
                dataframe=dataframe,
                logger=logger,
                artifacts_dir=artifacts_stage_dir,
                head=table_n,
            )
            # merge returned values
            if isinstance(prof_metrics, dict):
                metrics.update(prof_metrics)
            if isinstance(prof_saved, dict):
                saved.update({k: Path(v) for k, v in prof_saved.items()})
        except Exception:
            logger.exception("Profiling/export failed; continuing finalize_wandb_stage without profiling exports.")

    log_metrics(run, metrics)

    if table_key is None:
        table_key = f"{stage}_head{table_n}"
    log_dataframe_head(run, dataframe, key=table_key, n=table_n)

    # --- upload stage log file ---
    stage_log = logs_dir / f"{stage}.log"
    if stage_log.exists():
        log_files_as_artifact(
            run,
            artifact_name=f"capstone-logs-{stage}",
            artifact_type="logs",
            files=[stage_log],
            aliases=aliases,
            metadata={"stage": stage},
        )
    else:
        logger.warning("Stage log not found at %s; skipping log upload.", stage_log)

    # --- upload parquet outputs (one artifact that can include multiple dirs) ---
    # To avoid over-uploading, we upload only parquet patterns from each dataset_dir.
    # We attach each dir via log_dir_as_artifact separately, or combine by calling multiple times.
    for directory_path in dataset_dirs:
        directory_path = Path(directory_path)
        if directory_path.exists():
            log_dir_as_artifact(
                run,
                artifact_name=dataset_artifact_name,
                artifact_type="dataset",
                dir_path=directory_path,
                patterns=parquet_patterns,
                aliases=aliases,
                metadata={"stage": stage, "dir": str(directory_path)},
                recursive=True,
            )
        else:
            logger.warning("Dataset dir not found at %s; skipping dataset upload.", directory_path)

    # --- upload diagnostics exports from artifacts/<stage>/ ---
    # (This will include your describe CSVs if profile=True)
    if artifacts_stage_dir.exists():
        log_dir_as_artifact(
            run,
            artifact_name=f"{stage}-diagnostics",
            artifact_type="eda",
            dir_path=artifacts_stage_dir,
            patterns=diagnostics_patterns,
            aliases=aliases,
            metadata={"stage": stage},
            recursive=True,
        )

    # --- optional notebook upload ---
    if notebook_path is not None:
        notebook_file_path = Path(notebook_path)
        if notebook_file_path.exists():
            log_files_as_artifact(
                run,
                artifact_name="capstone-notebooks",
                artifact_type="notebook",
                files=[notebook_file_path],
                aliases=aliases,
                metadata={"stage": stage},
            )
        else:
            logger.warning("Notebook not found at %s; skipping notebook upload.", notebook_file_path)

    return {
        "metrics": metrics,
        "saved": saved,
        "artifacts_stage_dir": artifacts_stage_dir,
        "stage_log": stage_log,
        "dataset_dirs": [Path(directory_path) for directory_path in dataset_dirs],
    }


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 