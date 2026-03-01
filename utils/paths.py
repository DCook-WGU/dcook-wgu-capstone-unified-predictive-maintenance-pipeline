from __future__ import annotations

import os
import logging


from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


# Initiate Logging
logger = logging.getLogger("capstone.paths")


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
    artifacts: Path
    models: Path
    utils: Path
    logs: Path


@lru_cache(maxsize=1)
def get_paths() -> ProjectPaths:
    """
    Resolve project root and main directories.

    Priority:
    1) PROJECT_ROOT environment variable
    2) Python module location (__file__) if it looks like project root
    3) Jupyter/interactive fallback:
       - if cwd is <root>/notebooks, use cwd.parent
       - otherwise use cwd
    """
    # Get Current Working Directory
    cwd = Path.cwd().resolve()

    # Priority: ENV → script → Jupyter fallback
    env_root = os.getenv("PROJECT_ROOT")

    if env_root:
        project_root = Path(env_root).resolve()
        if not project_root.exists():
            raise ValueError(f"PROJECT_ROOT does not exist: {project_root}")
        if not (project_root / "data").exists():
            raise ValueError("PROJECT_ROOT does not look like project root (missing data/)")
        source = "env:PROJECT_ROOT"

    else:
        try:
            # Script Pathing
            # Get Project Root From the file location

            project_root_from_file  = Path(__file__).resolve().parents[1]

            if (project_root_from_file  / "data").exists():
                project_root = project_root_from_file 
                source = "__file__"

            else:
                if cwd.name == "notebooks":
                    project_root = cwd.parent
                    source = "cwd.parent (jupyter fallback)"
                else:
                    project_root = cwd
                    source = "cwd (jupyter fallback)"

        except NameError:
            # Jupyter notebook path
            # Notebooks should always be run from <root>/notebooks,
            # Thus cwd.parent should be the project root, else we will get the cwd
            if cwd.name == "notebooks":
                project_root = cwd.parent
                source = "cwd.parent (jupyter fallback)"
            else:
                project_root = cwd
                source = "cwd (jupyter fallback)"


    # Define data directory
    data_dir = project_root / "data"

    logger.debug("Resolved paths source=%s", source)
    logger.debug("CWD=%s", cwd)
    logger.debug("PROJECT_ROOT=%s", project_root)
    logger.debug("DATA_DIR=%s", data_dir)

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
        artifacts=project_root / "artifacts",
        models=project_root / "models",
        utils=project_root / "utils",
        logs=project_root / "logs",
    )
   
