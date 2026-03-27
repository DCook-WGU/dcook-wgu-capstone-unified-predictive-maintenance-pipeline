from __future__ import annotations

from typing import Optional

from utils.row_rebuilder import rebuild_consumed_messages_to_observations
from utils.final_aligned_incremental import run_final_align_loop
from utils.bronze_handoff import run_bronze_handoff_loop


def run_synthetic_to_bronze_once(
    engine,
    *,
    schema: str = "capstone",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    observation_batch_size: int = 1000,
    n_sensors: int = 52,
    complete_only: bool = True,
) -> dict:
    rebuild_result = rebuild_consumed_messages_to_observations(
        engine=engine,
        schema=schema,
        source_table="synthetic_sensor_messages_consumed_stage",
        target_table="synthetic_sensor_observations_rebuilt_stage",
        dataset_id=dataset_id,
        run_id=run_id,
        rebuild_status="pending",
        n_sensors=n_sensors,
        complete_only=complete_only,
        mark_source_rebuilt=True,
        observation_window_size=observation_batch_size,
    )

    final_align_results = run_final_align_loop(
        engine=engine,
        batch_size=observation_batch_size,
        schema=schema,
        premelt_table="synthetic_observations_premelt_stage",
        rebuilt_table="synthetic_sensor_observations_rebuilt_stage",
        target_table="synthetic_sensor_observations_final_aligned_stage",
        dataset_id=dataset_id,
        run_id=run_id,
        n_sensors=n_sensors,
        complete_only=complete_only,
        prefer_rebuilt_sensor_values=True,
        max_iterations=None,
        stop_on_failure=True,
    )

    bronze_results = run_bronze_handoff_loop(
        engine=engine,
        mode="row_batch",
        batch_size=observation_batch_size,
        schema=schema,
        source_table="synthetic_sensor_observations_final_aligned_stage",
        target_table="bronze_observations_input_stage",
        dataset_id=dataset_id,
        run_id=run_id,
        complete_only=complete_only,
        max_iterations=None,
        stop_on_failure=True,
    )

    return {
        "rebuild": rebuild_result,
        "final_align": final_align_results,
        "bronze": bronze_results,
    }


def run_synthetic_to_bronze_loop(
    engine,
    *,
    schema: str = "capstone",
    dataset_id: Optional[str] = None,
    run_id: Optional[str] = None,
    observation_batch_size: int = 1000,
    n_sensors: int = 52,
    complete_only: bool = True,
    max_iterations: Optional[int] = None,
) -> list[dict]:
    results: list[dict] = []
    iteration = 0

    while True:
        if max_iterations is not None and iteration >= int(max_iterations):
            break

        result = run_synthetic_to_bronze_once(
            engine=engine,
            schema=schema,
            dataset_id=dataset_id,
            run_id=run_id,
            observation_batch_size=observation_batch_size,
            n_sensors=n_sensors,
            complete_only=complete_only,
        )
        results.append(result)
        iteration += 1

        rebuild_rows = int(result["rebuild"].get("rebuilt_rows", 0))
        final_written = sum(int(x.get("written_count", 0)) for x in result["final_align"])
        bronze_written = sum(int(x.get("written_count", 0)) for x in result["bronze"])

        if rebuild_rows == 0 and final_written == 0 and bronze_written == 0:
            break

    return results


__all__ = [
    "run_synthetic_to_bronze_once",
    "run_synthetic_to_bronze_loop",
]