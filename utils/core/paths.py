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

    Parameters
    ----------
    start_path:
        Optional file or directory used as the starting point for upward project
        marker discovery. When omitted, discovery starts from the current
        working directory.

    Returns
    -------
    Path
        Resolved project root path.

    Raises
    ------
    ValueError
        If an explicit environment-configured root does not exist.

    Notes
    -----
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
    # CAPSTONE_PROJECT_ROOT is preferred; PROJECT_ROOT is a generic fallback for
    # environments that set only the shorter name (e.g. devcontainer presets).
    env_root = os.getenv("CAPSTONE_PROJECT_ROOT") or os.getenv("PROJECT_ROOT")

    if env_root:
        # expanduser + resolve makes the path absolute regardless of the shell's
        # working directory at import time, preventing CWD-relative surprises.
        root = Path(env_root).expanduser().resolve()

        if not root.exists():
            raise ValueError(f"Configured project root does not exist: {root}")

        return root

    # resolve() here ensures that symlinks and ".." segments in the cwd or
    # start_path are fully expanded before we begin the upward walk.
    current = Path(start_path or Path.cwd()).expanduser().resolve()

    if current.is_file():
        # If a notebook passes __file__, step up to its containing directory
        # so that the parent-walk starts from a directory, not a file node.
        current = current.parent

    for candidate in [current, *current.parents]:
        has_configs = (candidate / "configs" / "base.yaml").exists()
        has_utils = (candidate / "utils").exists()
        has_compose = (candidate / "docker-compose.yaml").exists()
        has_environment = (candidate / "environment.yml").exists()
        has_notebooks = (candidate / "notebooks").exists()

        # configs/base.yaml + utils/ is the strongest signal — both must be
        # present to avoid false positives in nested virtual-environment trees.
        if has_configs and has_utils:
            return candidate

        # Accept docker-compose.yaml, environment.yml, or notebooks/ as weaker
        # corroborating signals when utils/ is absent (e.g. stripped containers).
        if has_configs and (has_compose or has_environment or has_notebooks):
            return candidate

    # No markers found anywhere in the ancestry; treat the current working
    # directory as the root so callers always receive a valid Path.
    return Path.cwd().resolve()


@dataclass(frozen=True)
class ProjectPaths:
    """
    Centralized project path map used by notebooks, scripts, and utilities.

    Attributes map the resolved project root to commonly used project
    directories, including data layers, notebooks, artifacts, models, logs,
    configuration files, truth records, and pipeline scripts.

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


# maxsize=8 allows a small number of distinct root overrides (e.g. in tests)
# without unbounded growth; typical notebooks only ever call get_paths() once.
@lru_cache(maxsize=8)
def get_paths(project_root: str | Path | None = None) -> ProjectPaths:
    """
    Resolve the project root and return standardized project directory paths.

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

    Notes
    -----
    The result is cached by project root argument. Debug log messages record the
    resolved root, data directory, and configuration directory.
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
        # truths/ lives under artifacts/ so truth records and their parent
        # artifact outputs share the same root and can be cross-referenced by
        # relative path without hard-coding a second top-level directory.
        truths=root / "artifacts" / "truths",
        pipelines=root / "pipelines",
    )
