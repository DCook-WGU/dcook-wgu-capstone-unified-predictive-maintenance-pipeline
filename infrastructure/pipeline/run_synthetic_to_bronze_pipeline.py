from __future__ import annotations

import os

from utils.postgres_util import get_engine_from_env
from utils.synthetic_to_bronze_runner import run_synthetic_to_bronze_loop


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return str(value).strip()


def main() -> None:
    engine = get_engine_from_env()

    dataset_id = _require_env("DATASET_ID")
    run_id = _require_env("RUN_ID")

    schema = os.getenv("PIPELINE_SCHEMA", "capstone")
    observation_batch_size = int(os.getenv("OBSERVATION_BATCH_SIZE", "1000"))
    n_sensors = int(os.getenv("N_SENSORS", "52"))
    max_iterations_value = os.getenv("PIPELINE_MAX_ITERATIONS")
    max_iterations = None if not max_iterations_value else int(max_iterations_value)

    results = run_synthetic_to_bronze_loop(
        engine=engine,
        schema=schema,
        dataset_id=dataset_id,
        run_id=run_id,
        observation_batch_size=observation_batch_size,
        n_sensors=n_sensors,
        complete_only=True,
        max_iterations=max_iterations,
    )

    print(results)


if __name__ == "__main__":
    main()