# =========================================================
# Artifact directory utilities
# =========================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


def _clean_path_part(value: str | None) -> str | None:
    """Return a safe, trimmed directory-name component."""
    if value is None:
        return None

    clean_value = str(value).strip()

    if clean_value == "":
        return None

    return clean_value


def build_artifact_dirs(
    *,
    artifacts_root: str | Path,
    stage: str,
    dataset_name: str,
    family: str | None = None,
    subdirs: Iterable[str] | None = None,
    create: bool = True,
) -> dict[str, Path]:
    """
    Build standardized artifact directories for a pipeline stage.

    The directory pattern is:
        artifacts_root / stage / dataset_name / family / subdir

    If family is None or blank, the family level is skipped:
        artifacts_root / stage / dataset_name / subdir

    Returns a dictionary with:
        stage_dataset_root -> artifacts_root / stage / dataset_name
        root               -> family root or stage-dataset root
        <subdir>           -> root / <subdir>
    """

    stage_clean = _clean_path_part(stage)
    dataset_clean = _clean_path_part(dataset_name)
    family_clean = _clean_path_part(family)

    if stage_clean is None:
        raise ValueError("stage must be a non-empty string.")

    if dataset_clean is None:
        raise ValueError("dataset_name must be a non-empty string.")

    artifacts_root = Path(artifacts_root)
    stage_dataset_root = artifacts_root / stage_clean / dataset_clean

    if family_clean is None:
        artifact_root = stage_dataset_root
    else:
        artifact_root = stage_dataset_root / family_clean

    artifact_dirs: dict[str, Path] = {
        "stage_dataset_root": stage_dataset_root,
        "root": artifact_root,
    }

    if subdirs is not None:
        for subdir in subdirs:
            subdir_clean = _clean_path_part(subdir)
            if subdir_clean is None:
                continue

            if subdir_clean in artifact_dirs:
                raise ValueError(
                    f"Artifact subdir key conflicts with reserved key: {subdir_clean}"
                )

            artifact_dirs[subdir_clean] = artifact_root / subdir_clean

    if create:
        for path in artifact_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    return artifact_dirs


def build_artifact_dirs_from_config(
    *,
    config: Mapping[str, Any],
    stage_key: str,
    family_override: str | None = None,
    variant: str | None = None,
    subdirs_override: list[str] | None = None,
    create: bool = True,
) -> dict[str, Path]:
    """
    Build standardized artifact directories from the resolved pipeline config.

    Expected YAML pattern:
        bronze:
          artifact_layout:
            stage: bronze
            family: null
            subdirs: [profiles, summaries, metadata, config, lineage]

    Optional variant pattern:
        gold_cascade:
          artifact_layout:
            stage: gold
            subdirs: [...]
            variants:
              default:
                family: cascade_defaults
              tuned:
                family: cascade_tuned
    """

    if stage_key not in config:
        raise KeyError(f"stage_key not found in config: {stage_key}")

    stage_config = config[stage_key]
    layout = dict(stage_config.get("artifact_layout", {}))

    if variant is not None:
        variant_layout = layout.get("variants", {}).get(variant, {})

        if not variant_layout:
            raise KeyError(
                f"Variant '{variant}' was requested, but no artifact_layout "
                f"variant exists under config['{stage_key}']['artifact_layout']['variants']."
            )

        merged_layout = dict(layout)
        merged_layout.update(variant_layout)
        merged_layout.pop("variants", None)
        layout = merged_layout

    artifact_stage = layout.get(
        "stage",
        stage_config.get("layer_name", config.get("runtime", {}).get("stage")),
    )

    dataset_name = config["dataset"]["name"]

    family = family_override if family_override is not None else layout.get("family")

    subdirs = (
        subdirs_override
        if subdirs_override is not None
        else list(layout.get("subdirs", []))
    )

    return build_artifact_dirs(
        artifacts_root=Path(config["resolved_paths"]["artifacts_root"]),
        stage=artifact_stage,
        dataset_name=dataset_name,
        family=family,
        subdirs=subdirs,
        create=create,
    )


def artifact_file_path(
    artifact_dirs: Mapping[str, Path],
    subdir_key: str,
    file_name: str,
) -> Path:
    """Build a file path inside one standardized artifact subdirectory."""

    if subdir_key not in artifact_dirs:
        raise KeyError(
            f"Artifact directory key not found: {subdir_key}. "
            f"Available keys: {list(artifact_dirs)}"
        )

    return Path(artifact_dirs[subdir_key]) / file_name