from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from utils.paths import get_paths
from utils.file_io import save_data
from utils.logging_setup import configure_logging
from utils.pipeline_config_loader import (
    load_pipeline_config,
    build_truth_config_block,
    set_wandb_dir_from_config,
    export_config_snapshot,
)

from utils.truths import (
    make_process_run_id,
    initialize_layer_truth,
    update_truth_section,
    build_truth_record,
    save_truth_record,
    append_truth_index,
    stamp_truth_columns,
    load_truth_record_by_hash,
    get_truth_hash,
    get_pipeline_mode_from_truth,
    get_artifact_path_from_truth,
)

from utils.synthetic_profiles import (
    load_rich_profile_csv,
    load_correlation_pairs_csv,
    load_group_map_csv,
    load_fault_pairings_csv,
)
from utils.synthetic_generator import SyntheticGenerator, EpisodeSpec


logger = logging.getLogger("capstone.synthetic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic data generation stage.")
    parser.add_argument("--stage", default="synthetic")
    parser.add_argument("--dataset", default="pump")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--profile", default="default")

    # This is the one thing you MUST provide if config doesn't already include it
    parser.add_argument(
        "--silver-eda-truth-hash",
        default=None,
        help="Truth hash of the Silver EDA run to pull artifacts from (overrides config.runtime.silver_eda_truth_hash).",
    )

    # Optional overrides
    parser.add_argument("--primary-sensor", default=None, help="Primary sensor name (default: first sensor).")
    parser.add_argument("--primary-fault-type", default="drift_up")
    parser.add_argument("--magnitude", type=float, default=1.5)

    parser.add_argument("--normal-before", type=int, default=300)
    parser.add_argument("--buildup", type=int, default=100)
    parser.add_argument("--failure", type=int, default=80)
    parser.add_argument("--recovery", type=int, default=120)
    parser.add_argument("--normal-after", type=int, default=300)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---------------------------
    # 0) Paths + config
    # ---------------------------
    paths = get_paths()

    config_obj = load_pipeline_config(
        config_root=paths.configs,
        stage=args.stage,
        dataset=args.dataset,
        mode=args.mode,
        profile=args.profile,
        project_root=paths.root,
    )
    CONFIG = config_obj.data

    SYN_CFG = CONFIG["synthetic"]
    PATHS = CONFIG["resolved_paths"]
    PIPELINE = CONFIG.get("pipeline", {"execution_mode": "batch", "orchestration_mode": "script"})

    PIPELINE_MODE = PIPELINE["execution_mode"]
    DATASET_NAME = str(CONFIG["dataset"]["name"]).strip().lower()
    TRUTH_VERSION = CONFIG["versions"]["truth"]

    TRUTHS_PATH = Path(PATHS["truths_dir"])
    TRUTH_INDEX_PATH = Path(PATHS["truth_index_path"])
    LOGS_PATH = Path(PATHS["logs_root"])
    ARTIFACTS_ROOT = Path(PATHS["artifacts_root"])

    TRUTHS_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

    synthetic_log_path = Path(paths.logs) / "synthetic.log"
    configure_logging("capstone", synthetic_log_path, level=logging.DEBUG, overwrite_handlers=True)
    set_wandb_dir_from_config(CONFIG)

    # ---------------------------
    # 1) Load parent truth (Silver EDA)
    # ---------------------------
    silver_hash = args.silver_eda_truth_hash or CONFIG.get("runtime", {}).get("silver_eda_truth_hash")
    if silver_hash is None or str(silver_hash).strip() == "":
        raise ValueError(
            "Silver EDA truth hash is required. Provide --silver-eda-truth-hash or set config.runtime.silver_eda_truth_hash"
        )

    silver_eda_truth = load_truth_record_by_hash(
        truth_dir=TRUTHS_PATH,
        layer_name="silver_eda",
        dataset_name=DATASET_NAME,
        truth_hash=str(silver_hash).strip(),
    )

    PARENT_TRUTH_HASH = get_truth_hash(silver_eda_truth)

    parent_mode = get_pipeline_mode_from_truth(silver_eda_truth)
    if parent_mode:
        PIPELINE_MODE = parent_mode

    # ---------------------------
    # 2) Resolve artifact paths from parent truth
    # ---------------------------
    keys = SYN_CFG["silver_eda_artifact_keys"]

    profile_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_normal"])
    profile_abnormal_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_abnormal"])
    profile_recovery_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_recovery"])

    corr_pairs_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["corr_pairs_normal"])
    group_map_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["group_map_normal"])
    fault_pairings_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["fault_pairings_normal"])

    # ---------------------------
    # 3) Load profiles + build generator
    # ---------------------------
    normal_profiles = load_rich_profile_csv(profile_normal_path, state_scope="normal")
    abnormal_profiles = load_rich_profile_csv(profile_abnormal_path, state_scope="abnormal")
    recovery_profiles = load_rich_profile_csv(profile_recovery_path, state_scope="recovery")

    corr_pairs_df = load_correlation_pairs_csv(corr_pairs_normal_path)
    group_map_df = load_group_map_csv(group_map_normal_path)
    fault_pairings_df = load_fault_pairings_csv(fault_pairings_normal_path)

    generator = SyntheticGenerator(
        normal_profiles=normal_profiles,
        abnormal_profiles=abnormal_profiles,
        recovery_profiles=recovery_profiles,
        correlation_pairs_df=corr_pairs_df,
        group_map_df=group_map_df,
        fault_pairings_df=fault_pairings_df,
        random_seed=int(SYN_CFG["random_seed"]),
    )

    # ---------------------------
    # 4) Build episode spec (manual overrides supported)
    # ---------------------------
    primary_sensor = args.primary_sensor or generator.sensors[0]
    primary_fault_type = args.primary_fault_type

    episode = EpisodeSpec(
        primary_sensor=primary_sensor,
        primary_fault_type=primary_fault_type,
        magnitude=float(args.magnitude),
        normal_before=int(args.normal_before),
        buildup=int(args.buildup),
        failure=int(args.failure),
        recovery=int(args.recovery),
        normal_after=int(args.normal_after),
    )

    synthetic_df = generator.generate_episode(episode)

    # ---------------------------
    # 5) Truth record (synthetic layer)
    # ---------------------------
    process_run_id = make_process_run_id(str(SYN_CFG.get("process_run_id_prefix", "synthetic")))

    synthetic_truth = initialize_layer_truth(
        truth_version=str(TRUTH_VERSION),
        dataset_name=DATASET_NAME,
        layer_name="synthetic",
        process_run_id=process_run_id,
        pipeline_mode=PIPELINE_MODE,
        parent_truth_hash=PARENT_TRUTH_HASH,
    )

    synthetic_truth = update_truth_section(
        synthetic_truth,
        "config_snapshot",
        {
            "pipeline_config": export_config_snapshot(CONFIG),
            "synthetic_cfg": SYN_CFG,
        },
    )

    synthetic_truth = update_truth_section(
        synthetic_truth,
        "runtime_facts",
        {
            "primary_sensor": primary_sensor,
            "primary_fault_type": primary_fault_type,
            "episode": episode.__dict__,
            "row_count": int(len(synthetic_df)),
            "parent_truth_hash": PARENT_TRUTH_HASH,
            "silver_eda_truth_hash": PARENT_TRUTH_HASH,
        },
    )

    synth_dir = ARTIFACTS_ROOT / "synthetic" / DATASET_NAME
    synth_dir.mkdir(parents=True, exist_ok=True)

    out_path = synth_dir / f"{DATASET_NAME}__synthetic__episode.parquet"

    # IMPORTANT: correct save_data call order for your file_io.py
    # save_data(dataframe, file_path, file_name=None, ...)
    save_data(synthetic_df, synth_dir, out_path.name)

    synthetic_truth = update_truth_section(
        synthetic_truth,
        "artifact_paths",
        {
            "silver_eda_truth_hash": PARENT_TRUTH_HASH,
            "profile_normal_path": profile_normal_path,
            "profile_abnormal_path": profile_abnormal_path,
            "profile_recovery_path": profile_recovery_path,
            "corr_pairs_normal_path": corr_pairs_normal_path,
            "group_map_normal_path": group_map_normal_path,
            "fault_pairings_normal_path": fault_pairings_normal_path,
            "synthetic_episode_path": str(out_path),
        },
    )

    meta_columns = sorted(["meta__truth_hash", "meta__parent_truth_hash", "meta__pipeline_mode"])
    feature_columns = sorted([c for c in synthetic_df.columns if not str(c).startswith("meta__")])

    truth_record = build_truth_record(
        truth_base=synthetic_truth,
        row_count=int(len(synthetic_df)),
        column_count=int(synthetic_df.shape[1] + 3),
        meta_columns=meta_columns,
        feature_columns=feature_columns,
    )

    synth_truth_hash = truth_record["truth_hash"]

    synthetic_df = stamp_truth_columns(
        synthetic_df,
        truth_hash=synth_truth_hash,
        parent_truth_hash=PARENT_TRUTH_HASH,
        pipeline_mode=PIPELINE_MODE,
    )

    truth_path = save_truth_record(
        truth_record,
        truth_dir=TRUTHS_PATH,
        dataset_name=DATASET_NAME,
        layer_name="synthetic",
    )

    append_truth_index(truth_record, truth_index_path=TRUTH_INDEX_PATH)

    print("Synthetic truth hash:", synth_truth_hash)
    print("Synthetic truth path:", truth_path)
    print("Synthetic episode path:", out_path)


if __name__ == "__main__":
    main()