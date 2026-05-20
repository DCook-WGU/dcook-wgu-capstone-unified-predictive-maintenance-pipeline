# =========================================================
# Config-driven artifact directory utilities
# =========================================================

from typing import Any, Mapping


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

    Parameters
    ----------
    config:
        Loaded pipeline config dictionary.

    stage_key:
        Top-level stage config key, such as:
            "bronze"
            "silver_preeda"
            "silver_eda"
            "gold_preprocessing"
            "gold_baseline"
            "gold_cascade"
            "gold_comparison"
            "gold_anomaly_detection"

    family_override:
        Optional family folder override. Use this when the notebook needs
        to force a family name.

    variant:
        Optional variant key for stages with multiple variants, such as
        gold_cascade default/tuned/stage3_improved.

    subdirs_override:
        Optional explicit subdirectory list. Mostly useful during testing.

    create:
        If True, create all directories.

    Returns
    -------
    dict[str, Path]
        Named artifact directory paths.
    """

    if stage_key not in config:
        raise KeyError(f"stage_key not found in config: {stage_key}")

    stage_config = config[stage_key]
    layout = dict(stage_config.get("artifact_layout", {}))

    if variant is not None:
        variant_layout = (
            layout
            .get("variants", {})
            .get(variant, {})
        )

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
        stage_config.get("layer_name", config["runtime"]["stage"]),
    )

    dataset_name = config["dataset"]["name"]

    family = (
        family_override
        if family_override is not None
        else layout.get("family")
    )

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
    """
    Build a file path inside one standardized artifact subdirectory.
    """

    if subdir_key not in artifact_dirs:
        raise KeyError(
            f"Artifact directory key not found: {subdir_key}. "
            f"Available keys: {list(artifact_dirs)}"
        )

    return Path(artifact_dirs[subdir_key]) / file_name