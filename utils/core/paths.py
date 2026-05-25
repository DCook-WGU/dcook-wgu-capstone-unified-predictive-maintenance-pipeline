from __future__ import annotations

import logging
import os

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


logger = logging.getLogger("capstone.paths")


def find_project_root(start_path: str | Path | None = None) -> Path:
    """
    Resolve the capstone project root from a notebook, script, or container path.

    This prevents notebooks from accidentally treating their own folder as the
    project root. For example, a notebook launched from:

        /workspace/notebooks/synthetic

    should still resolve the project root as:

        /workspace

    Resolution order:
    1. CAPSTONE_PROJECT_ROOT environment variable, if set.
    2. PROJECT_ROOT environment variable, if set.
    3. Walk upward from start_path or the current working directory.
    4. Fall back to the current working directory if no marker is found.

    A directory is treated as the project root when it contains strong project
    markers such as configs/base.yaml plus utils/, docker-compose.yaml,
    environment.yml, or notebooks/.
    """
    env_root = os.getenv("CAPSTONE_PROJECT_ROOT") or os.getenv("PROJECT_ROOT")

    if env_root:
        root = Path(env_root).expanduser().resolve()

        if not root.exists():
            raise ValueError(f"Configured project root does not exist: {root}")

        return root

    current = Path(start_path or Path.cwd()).expanduser().resolve()

    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        has_configs = (candidate / "configs" / "base.yaml").exists()
        has_utils = (candidate / "utils").exists()
        has_compose = (candidate / "docker-compose.yaml").exists()
        has_environment = (candidate / "environment.yml").exists()
        has_notebooks = (candidate / "notebooks").exists()

        if has_configs and has_utils:
            return candidate

        if has_configs and (has_compose or has_environment or has_notebooks):
            return candidate

    return Path.cwd().resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """
    Centralized project path map used by notebooks, scripts, and utilities.

    Keeping these paths in one dataclass helps avoid hardcoded notebook-specific
    paths and keeps the medallion pipeline easier to run after a Docker or WSL
    reset.
    """

    root: Path
    data: Path
    data_raw: Path
    data_bronze: Path
    data_bronze_train: Path
    data_bronze_test: Path
    data_silver: Path
    data_silver_train: Path
    data_silver_test: Path
    data_gold: Path
    data_synthetic: Path
    notebooks: Path
    artifacts: Path
    models: Path
    utils: Path
    logs: Path
    configs: Path
    truths: Path
    pipelines: Path


@lru_cache(maxsize=8)
def get_paths(project_root: str | Path | None = None) -> ProjectPaths:
    """
    Resolve the project root and return standardized project directories.

    Parameters
    ----------
    project_root:
        Optional explicit project root. When omitted, the root is discovered
        by environment variable or by walking upward from the current working
        directory until project markers are found.

    Returns
    -------
    ProjectPaths
        Dataclass containing common project directories.
    """
    root = find_project_root(project_root)
    data_dir = root / "data"

    logger.debug("Resolved project root: %s", root)
    logger.debug("Resolved data directory: %s", data_dir)
    logger.debug("Resolved config directory: %s", root / "configs")

    return ProjectPaths(
        root=root,
        data=data_dir,
        data_raw=data_dir / "raw",
        data_bronze=data_dir / "bronze",
        data_bronze_train=data_dir / "bronze" / "train",
        data_bronze_test=data_dir / "bronze" / "test",
        data_silver=data_dir / "silver",
        data_silver_train=data_dir / "silver" / "train",
        data_silver_test=data_dir / "silver" / "test",
        data_gold=data_dir / "gold",
        data_synthetic=data_dir / "synthetic",
        notebooks=root / "notebooks",
        artifacts=root / "artifacts",
        models=root / "models",
        utils=root / "utils",
        logs=root / "logs",
        configs=root / "configs",
        truths=root / "artifacts" / "truths",
        pipelines=root / "pipelines",
    )