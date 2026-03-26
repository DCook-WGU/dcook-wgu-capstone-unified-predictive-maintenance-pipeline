import os

from utils.postgres_util import get_engine_from_env
from utils.kafka_producer_adapter import run_send_queue_producer_loop


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return str(value).strip()


def _optional_int(name: str):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def main() -> None:
    engine = get_engine_from_env()

    dataset_id = _require_env("DATASET_ID")
    run_id = _require_env("RUN_ID")

    results = run_send_queue_producer_loop(
        engine=engine,
        dataset_id=dataset_id,
        run_id=run_id,
        schema=os.getenv("PRODUCER_SCHEMA", "capstone"),
        queue_table=os.getenv("PRODUCER_QUEUE_TABLE", "synthetic_sensor_messages_send_queue"),
        control_table=os.getenv("PRODUCER_CONTROL_TABLE", "simulation_state_control"),
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
        producer_worker_id=os.getenv("PRODUCER_WORKER_ID", "producer_worker_001"),
        client_id=os.getenv("KAFKA_CLIENT_ID", "synthetic-telemetry-producer"),
        max_batches=_optional_int("PRODUCER_MAX_BATCHES"),
        stop_on_failure=True,
        flush_timeout_seconds=float(os.getenv("PRODUCER_FLUSH_TIMEOUT_SECONDS", "30.0")),
    )

    print(results)


if __name__ == "__main__":
    main()