from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import logging

# Initiate Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized project paths."""
    root: Path
    data: Path
    #Optional extras you can uncomment/add as your tree solidifies:
    data_raw: Path
    data_bronze: Path
    data_bronze_train: Path
    data_bronze_test: Path
    data_silver: Path
    data_silver_train: Path
    data_silver_test: Path
    data_gold: Path
    notebooks: Path
    models: Path
    utils: Path
    # logs: Path


@lru_cache(maxsize=1)

def get_paths() -> ProjectPaths:
    """
    Resolve project root and main directories.

    Priority:
    1. PROJECT_ROOT environment variable
    2. Python script location (__file__)
    3. Jupyter / other: current working directory's parent
    """
     # Get Current Working Directory
    cwd = Path().resolve()

    # Priority: ENV → script → Jupyter fallback
    env_root = os.getenv("PROJECT_ROOT")

    if env_root:
        project_root = Path(env_root).resolve()
    else:
        try:
            # Script Path
            project_root = Path(__file__).resolve().parents[1]
        except NameError:
            # Jupyter notebook path
            # If you normally open notebooks in <root>/notebooks,
            # then cwd.parent should be the project root.
            project_root = cwd.parent

    # Define data directory
    data_dir = project_root / "data"

    logger.debug(f"CWD:          {cwd}")
    logger.debug(f"PROJECT_ROOT: {project_root}")
    logger.debug(f"DATA DIR:     {data_dir}")

    return ProjectPaths(
        root=project_root,
        data=data_dir,
        data_raw=data_dir / "raw",
        data_bronze=data_dir / "bronze",
        data_bronze_train=data_dir / "bronze/train",
        data_bronze_test=data_dir / "bronze/test",
        data_silver=data_dir / "silver",
        data_silver_train=data_dir / "silver/train",
        data_silver_test=data_dir / "silver/test",
        data_gold=data_dir / "gold",
        notebooks=project_root / "notebooks",
        models=project_root / "models",
        utils=project_root / "utils",
        #logs=project_root / "logs",
    )
   
