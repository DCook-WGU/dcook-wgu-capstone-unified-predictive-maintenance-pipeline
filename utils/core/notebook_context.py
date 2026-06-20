# utils/core/notebook_context.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from utils.core.paths import get_paths
from utils.core.config_loader import load_pipeline_config
from utils.core.logging_setup import configure_logging
from utils.core.ledger import Ledger


@dataclass(frozen=True)
class NotebookContext:
    """Container for resolved notebook configuration, logging, and ledger state."""

    stage: str
    recipe_id: str
    dataset: str
    mode: str
    profile: str
    paths: Any
    config: Dict[str, Any]
    stage_config: Dict[str, Any]
    resolved_paths: Dict[str, Any]
    filenames: Dict[str, Any]
    versions: Dict[str, Any]
    runtime: Dict[str, Any]
    dataset_config: Dict[str, Any]
    wandb: Dict[str, Any]
    execution: Dict[str, Any]
    pipeline: Dict[str, Any]
    default_fallbacks: Dict[str, Any]
    logger: logging.Logger
    ledger: Ledger
    log_path: Path


def _require_mapping(value: Any, name: str) -> Dict[str, Any]:
    """Return a copied mapping or raise TypeError for required config sections."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _optional_mapping(value: Any, name: str) -> Dict[str, Any]:
    """Return a copied optional mapping, using an empty dict when omitted."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def load_notebook_context(
    *,
    stage: str,
    recipe_id: str | None = None,
    dataset: str = "pump",
    mode: str = "train",
    profile: str = "default",
    logger_name: str = "capstone",
    logger_child_name: str | None = None,
    log_filename: str | None = None,
    log_level: int = logging.DEBUG,
    overwrite_handlers: bool = True,
) -> NotebookContext:
    """
    Load the shared runtime context used by capstone notebooks.

    This function centralizes the repeated notebook setup pattern: project paths,
    stage-aware configuration, common config sections, logging, and ledger
    initialization. Individual notebooks should use this function once near the
    top of the notebook, then create small compatibility aliases such as
    `GOLD_CFG = CTX.stage_config` or `DATASET_CFG = CTX.dataset_config`.

    Parameters
    ----------
    stage:
        Stage configuration key to load and expose as `stage_config`.
    recipe_id:
        Optional recipe identifier. Falls back to stage config values or a
        deterministic stage-based default.
    dataset, mode, profile:
        Configuration overlays passed to the project config loader.
    logger_name, logger_child_name, log_filename, log_level, overwrite_handlers:
        Logging options forwarded to the shared logging setup.

    Returns
    -------
    NotebookContext
        Frozen context containing paths, resolved config sections, logger,
        ledger, and log path for notebook use.

    Side Effects
    ------------
    Configures logging handlers and records an initialization entry in the
    notebook ledger.
    """
    paths = get_paths()

    config = load_pipeline_config(
        config_root=paths.configs,
        stage=stage,
        dataset=dataset,
        mode=mode,
        profile=profile,
        project_root=paths.root,
    ).data

    config_map = _require_mapping(config, "CONFIG")

    stage_config = _require_mapping(config_map.get(stage, {}), stage)

    # recipe_id resolution order:
    # 1. explicit caller value (notebook override)
    # 2. stage config "recipe_id" key (normal YAML location)
    # 3. stage config "cleaning_recipe_id" (Bronze/Silver legacy name)
    # 4. deterministic default so recipe_id is never None or empty
    resolved_recipe_id = str(
        recipe_id
        or stage_config.get("recipe_id")
        or stage_config.get("cleaning_recipe_id")
        or f"{stage}__v001"
    )

    resolved_paths = _require_mapping(config_map.get("resolved_paths", {}), "resolved_paths")
    filenames = _require_mapping(config_map.get("filenames", {}), "filenames")
    versions = _require_mapping(config_map.get("versions", {}), "versions")

    runtime = _optional_mapping(config_map.get("runtime"), "runtime")
    if not runtime:
        runtime = {
            "mode": mode,
            "profile": profile,
        }

    dataset_config = _require_mapping(config_map.get("dataset", {}), "dataset")
    wandb = _optional_mapping(config_map.get("wandb"), "wandb")
    execution = _optional_mapping(config_map.get("execution"), "execution")

    pipeline = _optional_mapping(config_map.get("pipeline"), "pipeline")
    if not pipeline:
        # Older config files do not include a pipeline block; default to batch/notebook so
        # the context is always usable without requiring a config update for each stage.
        pipeline = {
            "execution_mode": "batch",
            "orchestration_mode": "notebook",
        }

    # default_fallbacks is read from stage_config (not top-level config) so each stage
    # can define its own column/value fallbacks without sharing a global fallback namespace.
    default_fallbacks = _optional_mapping(
        stage_config.get("default_fallbacks"),
        f"{stage}.default_fallbacks",
    )

    safe_stage_name = stage.replace(" ", "_").replace("/", "_")
    log_path = paths.logs / (log_filename or f"{safe_stage_name}.log")

    configure_logging(
        logger_name,
        log_path,
        level=log_level,
        overwrite_handlers=overwrite_handlers,
    )

    logger = logging.getLogger(logger_child_name or f"{logger_name}.{stage}")
    logger.info(
        "%s stage starting",
        stage,
        extra={
            "stage": stage,
            "recipe_id": resolved_recipe_id,
            "dataset": dataset,
            "mode": mode,
            "profile": profile,
        },
    )

    ledger = Ledger(stage=stage, recipe_id=resolved_recipe_id)
    ledger.add(
        kind="step",
        step="init",
        message="Initialized ledger from shared notebook context",
        data={
            "stage": stage,
            "recipe_id": resolved_recipe_id,
            "dataset": dataset,
            "mode": mode,
            "profile": profile,
            "log_path": str(log_path),
        },
        logger=logger,
    )

    return NotebookContext(
        stage=stage,
        recipe_id=resolved_recipe_id,
        dataset=dataset,
        mode=mode,
        profile=profile,
        paths=paths,
        config=config_map,
        stage_config=stage_config,
        resolved_paths=resolved_paths,
        filenames=filenames,
        versions=versions,
        runtime=runtime,
        dataset_config=dataset_config,
        wandb=wandb,
        execution=execution,
        pipeline=pipeline,
        default_fallbacks=default_fallbacks,
        logger=logger,
        ledger=ledger,
        log_path=log_path,
    )
