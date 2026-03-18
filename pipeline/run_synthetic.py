from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from utils.paths import get_paths
from utils.file_io import save_data
from utils.logging_setup import configure_logging
from utils.pipeline_config_loader import (
    load_pipeline_config,
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
    get_parent_truth_hash,
    get_artifact_path_from_truth,
)

from utils.synthetic_profiles import (
    load_and_merge_rich_profiles,
    load_correlation_pairs_csv,
    load_group_map_csv,
    load_fault_pairings_csv,
)

from utils.synthetic_missingness import build_missingness_spec_from_truth_payload
from utils.synthetic_generator import SyntheticGenerator, EpisodeSpec
from utils.synthetic_postgres_writer import (
    ensure_sequence,
    reserve_next_batch_id,
    reserve_cycle_range,
    write_stream_batch,
)
from utils.synthetic_export import export_synthetic_batch_to_parquet
from utils.postgres_util import get_engine_from_env

logger = logging.getLogger("capstone.synthetic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic data generation stage.")
    parser.add_argument("--stage", default="synthetic")
    parser.add_argument("--dataset", default="pump")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--profile", default="default")

    parser.add_argument(
        "--silver-eda-truth-hash",
        default=None,
        help="Truth hash of the Silver EDA run to pull artifacts from.",
    )

    parser.add_argument("--primary-sensor", default=None)
    parser.add_argument("--primary-fault-type", default="drift_up")  # or "random"
    parser.add_argument("--seed", type=int, default=None)

    # Optional scalar overrides (if omitted, ranges from config are sampled)
    parser.add_argument("--normal-before", type=int, default=None)
    parser.add_argument("--buildup", type=int, default=None)
    parser.add_argument("--failure", type=int, default=None)
    parser.add_argument("--recovery", type=int, default=None)
    parser.add_argument("--normal-after", type=int, default=None)
    parser.add_argument("--magnitude", type=float, default=None)

    return parser.parse_args()


def _sample_range(rng: np.random.Generator, value: Any, *, cast_type: type) -> Any:
    """
    Accepts either:
      - scalar (already usable)
      - 2-list / 2-tuple [low, high] -> uniform int/float sample
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = value
        if cast_type is int:
            return int(rng.integers(int(lo), int(hi) + 1))
        return float(rng.uniform(float(lo), float(hi)))

    # scalar
    return cast_type(value)


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

    # RNG seed
    seed = int(args.seed) if args.seed is not None else int(SYN_CFG.get("random_seed", 42))
    rng = np.random.default_rng(seed)

    # ---------------------------
    # 1) Load parent truth (Silver EDA) + parent (Silver PreEDA)
    # ---------------------------
    silver_hash = args.silver_eda_truth_hash or (CONFIG.get("runtime", {}) or {}).get("silver_eda_truth_hash")
    if not silver_hash:
        raise ValueError("Silver EDA truth hash required: --silver-eda-truth-hash or config.runtime.silver_eda_truth_hash")

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

    # PreEDA truth (needed for missingness)
    silver_preeda_hash = get_parent_truth_hash(silver_eda_truth)
    if not silver_preeda_hash:
        raise ValueError("Silver EDA truth is missing parent_truth_hash (Silver PreEDA).")

    silver_preeda_truth = load_truth_record_by_hash(
        truth_dir=TRUTHS_PATH,
        layer_name="silver_preeda",
        dataset_name=DATASET_NAME,
        truth_hash=str(silver_preeda_hash).strip(),
    )

    # ---------------------------
    # 2) Resolve artifact paths from Silver EDA truth
    # ---------------------------
    keys = SYN_CFG["silver_eda_artifact_keys"]

    # Base profiles
    profile_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_normal"])
    profile_abnormal_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_abnormal"])
    profile_recovery_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_recovery"])

    # Dropped profiles (optional, new)
    dropped_profile_normal_path = None
    dropped_profile_abnormal_path = None
    dropped_profile_recovery_path = None
    if "profile_dropped_normal" in keys:
        dropped_profile_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_dropped_normal"])
    if "profile_dropped_abnormal" in keys:
        dropped_profile_abnormal_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_dropped_abnormal"])
    if "profile_dropped_recovery" in keys:
        dropped_profile_recovery_path = get_artifact_path_from_truth(silver_eda_truth, keys["profile_dropped_recovery"])

    # Normal-only correlation/group/pairings
    corr_pairs_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["corr_pairs_normal"])
    group_map_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["group_map_normal"])
    fault_pairings_normal_path = get_artifact_path_from_truth(silver_eda_truth, keys["fault_pairings_normal"])

    # ---------------------------
    # 3) MissingnessSpec from Silver PreEDA truth (runtime_facts.missingness_quarantine)
    # ---------------------------
    preeda_runtime = (silver_preeda_truth.get("runtime_facts", {}) or {})
    missingness_payload = (preeda_runtime.get("missingness_quarantine", {}) or {})
    if not missingness_payload:
        raise ValueError("Silver PreEDA truth is missing runtime_facts.missingness_quarantine (required for synthetic).")

    missingness_spec = build_missingness_spec_from_truth_payload(missingness_payload)

    # ---------------------------
    # 4) Load profiles + build generator
    # ---------------------------
    normal_profiles = load_and_merge_rich_profiles(
        base_profile_csv_path=str(profile_normal_path),
        dropped_profile_csv_path=str(dropped_profile_normal_path) if dropped_profile_normal_path else None,
        state_scope="normal",
    )
    abnormal_profiles = load_and_merge_rich_profiles(
        base_profile_csv_path=str(profile_abnormal_path),
        dropped_profile_csv_path=str(dropped_profile_abnormal_path) if dropped_profile_abnormal_path else None,
        state_scope="abnormal",
    )
    recovery_profiles = load_and_merge_rich_profiles(
        base_profile_csv_path=str(profile_recovery_path),
        dropped_profile_csv_path=str(dropped_profile_recovery_path) if dropped_profile_recovery_path else None,
        state_scope="recovery",
    )

    corr_pairs_df = load_correlation_pairs_csv(str(corr_pairs_normal_path))
    group_map_df = load_group_map_csv(str(group_map_normal_path))
    fault_pairings_df = load_fault_pairings_csv(str(fault_pairings_normal_path))

    generator = SyntheticGenerator(
        normal_profiles=normal_profiles,
        abnormal_profiles=abnormal_profiles,
        recovery_profiles=recovery_profiles,
        correlation_pairs_dataframe=corr_pairs_df,
        group_map_dataframe=group_map_df,
        fault_pairings_dataframe=fault_pairings_df,
        random_seed=seed,
        missingness_spec=missingness_spec,
    )

    # ---------------------------
    # 5) Build episode spec (scalar overrides OR sample from config ranges)
    # ---------------------------
    ep_cfg = (SYN_CFG.get("episode", {}) or {})

    PRIMARY_SENSOR = args.primary_sensor
    PRIMARY_FAULT_TYPE = args.primary_fault_type

    # sample ranges unless user provided scalar override
    NORMAL_BEFORE = args.normal_before if args.normal_before is not None else _sample_range(rng, ep_cfg.get("normal_before_range", 300), cast_type=int)
    BUILDUP = args.buildup if args.buildup is not None else _sample_range(rng, ep_cfg.get("buildup_range", 100), cast_type=int)
    FAILURE = args.failure if args.failure is not None else _sample_range(rng, ep_cfg.get("failure_range", 80), cast_type=int)
    RECOVERY = args.recovery if args.recovery is not None else _sample_range(rng, ep_cfg.get("recovery_range", 120), cast_type=int)
    NORMAL_AFTER = args.normal_after if args.normal_after is not None else _sample_range(rng, ep_cfg.get("normal_after_range", 300), cast_type=int)
    MAGNITUDE = args.magnitude if args.magnitude is not None else _sample_range(rng, ep_cfg.get("magnitude_range", 1.5), cast_type=float)

    primary_sensor = PRIMARY_SENSOR or generator.sensors[0]

    if str(PRIMARY_FAULT_TYPE).strip().lower() == "random":
        allowed = (SYN_CFG.get("fault_selection", {}) or {}).get("allowed_primary_faults", []) or []
        if not allowed:
            raise ValueError("primary-fault-type=random requested but synthetic.fault_selection.allowed_primary_faults is empty.")
        primary_fault_type = str(rng.choice(np.array(allowed, dtype=object)))
    else:
        primary_fault_type = str(PRIMARY_FAULT_TYPE)

    episode = EpisodeSpec(
        primary_sensor=primary_sensor,
        primary_fault_type=primary_fault_type,
        magnitude=float(MAGNITUDE),
        normal_before=int(NORMAL_BEFORE),
        buildup=int(BUILDUP),
        failure=int(FAILURE),
        recovery=int(RECOVERY),
        normal_after=int(NORMAL_AFTER),
    )

    synthetic_df = generator.generate_episode(episode)

    # ---------------------------
    # 6) Write to Postgres (batch)
    # ---------------------------
    engine = get_engine_from_env()
    PG_SCHEMA = str(SYN_CFG.get("postgres_schema", "capstone"))

    batch_seq = f"seq_synthetic_{DATASET_NAME}_batch_id"
    cycle_seq = f"seq_synthetic_{DATASET_NAME}_cycle_id"

    ensure_sequence(engine, schema=PG_SCHEMA, sequence_name=batch_seq)
    ensure_sequence(engine, schema=PG_SCHEMA, sequence_name=cycle_seq)

    batch_id = reserve_next_batch_id(engine, schema=PG_SCHEMA, sequence_name=batch_seq)
    cycle_start = reserve_cycle_range(engine, schema=PG_SCHEMA, sequence_name=cycle_seq, n_rows=len(synthetic_df))

    table_name = write_stream_batch(
        engine,
        synthetic_df,
        dataset_name=DATASET_NAME,
        schema=PG_SCHEMA,
        artifact_name="stream",
        batch_id=batch_id,
        cycle_start=cycle_start,
    )

    logger.info("Wrote synthetic batch to %s (batch_id=%s cycle_start=%s)", table_name, batch_id, cycle_start)
    print("Wrote synthetic batch to table:", table_name)

    # ---------------------------
    # 7) Export batch parquet (optional but useful)
    # ---------------------------
    EXPORT_ENABLED = bool(SYN_CFG.get("export_batch_parquet", True))
    export_path = None
    if EXPORT_ENABLED:
        export_dir = ARTIFACTS_ROOT / "synthetic_exports" / DATASET_NAME
        export_path = export_synthetic_batch_to_parquet(
            dataset_name=DATASET_NAME,
            batch_id=batch_id,
            out_dir=export_dir,
            schema=PG_SCHEMA,
            artifact_name="stream",
        )
        print("Exported batch parquet:", export_path)

    # ---------------------------
    # 8) Local episode artifact + truth record
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

    # resolved config snapshot (write to file, then store path)
    resolved_dir = ARTIFACTS_ROOT / "synthetic" / DATASET_NAME
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = resolved_dir / f"{DATASET_NAME}__synthetic__resolved_config.yaml"
    export_config_snapshot(CONFIG, destination=resolved_config_path)

    synthetic_truth = update_truth_section(
        synthetic_truth,
        "config_snapshot",
        {
            "resolved_config_path": str(resolved_config_path),
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
            "sampled_ranges": {
                "normal_before": NORMAL_BEFORE,
                "buildup": BUILDUP,
                "failure": FAILURE,
                "recovery": RECOVERY,
                "normal_after": NORMAL_AFTER,
                "magnitude": MAGNITUDE,
            },
            "row_count": int(len(synthetic_df)),
            "parent_truth_hash": PARENT_TRUTH_HASH,
            "silver_eda_truth_hash": PARENT_TRUTH_HASH,
            "silver_preeda_truth_hash": str(silver_preeda_hash),
        },
    )

    # local parquet of the episode
    out_path = resolved_dir / f"{DATASET_NAME}__synthetic__episode.parquet"
    save_data(synthetic_df, resolved_dir, out_path.name)

    artifact_paths_payload: Dict[str, Any] = {
        "profile_normal_path": str(profile_normal_path),
        "profile_abnormal_path": str(profile_abnormal_path),
        "profile_recovery_path": str(profile_recovery_path),
        "dropped_profile_normal_path": str(dropped_profile_normal_path) if dropped_profile_normal_path else None,
        "dropped_profile_abnormal_path": str(dropped_profile_abnormal_path) if dropped_profile_abnormal_path else None,
        "dropped_profile_recovery_path": str(dropped_profile_recovery_path) if dropped_profile_recovery_path else None,
        "corr_pairs_normal_path": str(corr_pairs_normal_path),
        "group_map_normal_path": str(group_map_normal_path),
        "fault_pairings_normal_path": str(fault_pairings_normal_path),
        "synthetic_episode_path": str(out_path),
        "postgres_schema": PG_SCHEMA,
        "postgres_table": table_name,
        "postgres_batch_id": int(batch_id),
        "postgres_cycle_start": int(cycle_start),
    }
    if export_path is not None:
        artifact_paths_payload["export_batch_parquet_path"] = str(export_path)

    synthetic_truth = update_truth_section(synthetic_truth, "artifact_paths", artifact_paths_payload)

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
    print("Local episode parquet:", out_path)


if __name__ == "__main__":
    main()