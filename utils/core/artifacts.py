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


def _copy_artifact_mapping(value: Mapping[Any, Any] | None = None) -> dict[str, Any]:
    """
    Copy an artifact configuration mapping into a plain string-keyed dictionary.

    This avoids Pylance incorrectly inferring config dictionaries as byte-keyed
    mappings after calls such as dict(config or {}).
    """
    if value is None:
        return {}

    return {
        str(key): item
        for key, item in value.items()
    }


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    """
    Validate that a value is a mapping and return it as a string-keyed dictionary.
    """
    if not isinstance(value, Mapping):
        raise TypeError(
            f"Expected {label} to be a mapping, "
            f"got {type(value).__name__}: {value!r}"
        )

    return _copy_artifact_mapping(value)


def _get_artifact_mapping(
    mapping: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    """
    Return a nested artifact mapping as a plain string-keyed dictionary.
    """
    raw_value = mapping.get(key)

    if raw_value is None:
        return {}

    return _require_mapping(raw_value, f"artifact config key '{key}'")


def _get_optional_artifact_str(
    mapping: Mapping[str, Any],
    key: str,
    default: str | None = None,
) -> str | None:
    """
    Return an optional artifact config value as str | None.
    """
    raw_value = mapping.get(key, default)

    if raw_value is None:
        return None

    return str(raw_value)


def _get_required_artifact_str(
    mapping: Mapping[str, Any],
    key: str,
    label: str,
) -> str:
    """
    Return a required artifact config value as a non-empty string.
    """
    raw_value = mapping.get(key)

    if raw_value is None:
        raise KeyError(f"Required artifact config value missing: {label}")

    clean_value = str(raw_value).strip()

    if clean_value == "":
        raise ValueError(f"Required artifact config value is blank: {label}")

    return clean_value


def _get_artifact_subdirs(
    mapping: Mapping[str, Any],
    key: str = "subdirs",
    default: list[str] | None = None,
) -> list[str]:
    """
    Return artifact subdirectories as a list of strings.
    """
    if default is None:
        default = []

    raw_value = mapping.get(key, default)

    if raw_value is None:
        return list(default)

    if isinstance(raw_value, bytes):
        return [raw_value.decode("utf-8")]

    if isinstance(raw_value, str):
        return [raw_value]

    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]

    if isinstance(raw_value, tuple):
        return [str(item) for item in raw_value]

    raise TypeError(
        f"Expected artifact config key '{key}' to be a list/tuple/string, "
        f"got {type(raw_value).__name__}: {raw_value!r}"
    )


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

    artifacts_root_path = Path(artifacts_root)
    stage_dataset_root = artifacts_root_path / stage_clean / dataset_clean

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
            subdir_clean = _clean_path_part(str(subdir))

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
    config_map: dict[str, Any] = _copy_artifact_mapping(config)

    if stage_key not in config_map:
        raise KeyError(f"stage_key not found in config: {stage_key}")

    stage_config: dict[str, Any] = _require_mapping(
        config_map[stage_key],
        f"config['{stage_key}']",
    )

    layout: dict[str, Any] = _get_artifact_mapping(
        stage_config,
        "artifact_layout",
    )

    if variant is not None:
        variants: dict[str, Any] = _get_artifact_mapping(
            layout,
            "variants",
        )

        variant_key = str(variant)

        if variant_key not in variants:
            raise KeyError(
                f"Variant '{variant}' was requested, but no artifact_layout "
                f"variant exists under config['{stage_key}']['artifact_layout']['variants']."
            )

        variant_layout: dict[str, Any] = _require_mapping(
            variants[variant_key],
            (
                f"config['{stage_key}']['artifact_layout']"
                f"['variants']['{variant_key}']"
            ),
        )

        merged_layout: dict[str, Any] = {
            str(key): value
            for key, value in layout.items()
            if key != "variants"
        }

        merged_layout.update(variant_layout)
        layout = merged_layout

    runtime_config: dict[str, Any] = _get_artifact_mapping(
        config_map,
        "runtime",
    )

    default_stage = _get_optional_artifact_str(
        stage_config,
        "layer_name",
        _get_optional_artifact_str(runtime_config, "stage"),
    )

    artifact_stage = _get_optional_artifact_str(
        layout,
        "stage",
        default_stage,
    )

    if artifact_stage is None:
        raise KeyError(
            "Could not resolve artifact stage. Expected one of: "
            f"config['{stage_key}']['artifact_layout']['stage'], "
            f"config['{stage_key}']['layer_name'], or config['runtime']['stage']."
        )

    dataset_config: dict[str, Any] = _get_artifact_mapping(
        config_map,
        "dataset",
    )

    dataset_name = _get_required_artifact_str(
        dataset_config,
        "name",
        "config['dataset']['name']",
    )

    if family_override is not None:
        family = str(family_override)
    else:
        family = _get_optional_artifact_str(layout, "family")

    if subdirs_override is not None:
        subdirs = [str(subdir) for subdir in subdirs_override]
    else:
        subdirs = _get_artifact_subdirs(
            layout,
            "subdirs",
            default=[],
        )

    resolved_paths: dict[str, Any] = _get_artifact_mapping(
        config_map,
        "resolved_paths",
    )

    artifacts_root = _get_required_artifact_str(
        resolved_paths,
        "artifacts_root",
        "config['resolved_paths']['artifacts_root']",
    )

    return build_artifact_dirs(
        artifacts_root=artifacts_root,
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