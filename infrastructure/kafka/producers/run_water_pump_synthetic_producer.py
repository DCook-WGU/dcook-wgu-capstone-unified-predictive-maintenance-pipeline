from __future__ import annotations

from utils.core.env_helpers import (
    env_float,
    env_optional_int,
    env_required_str,
    env_str,
)
from utils.database.postgres import get_engine_from_env
from utils.synthetic.pipeline.kafka_producer_adapter import run_send_queue_producer_loop


def main() -> None:
    engine = get_engine_from_env()

    dataset_id = env_required_str(
        "SYNTHETIC_DATASET_ID",
        aliases=("DATASET_ID",),
    )

    run_id = env_required_str(
        "SYNTHETIC_RUN_ID",
        aliases=("RUN_ID",),
    )

    results = run_send_queue_producer_loop(
        engine=engine,
        dataset_id=dataset_id,
        run_id=run_id,
        schema=env_str(
            "CAPSTONE_SCHEMA",
            "capstone",
            aliases=("PRODUCER_SCHEMA",),
        ),
        queue_table=env_str(
            "SYNTHETIC_SEND_QUEUE_TABLE",
            "synthetic_sensor_messages_send_queue",
            aliases=("PRODUCER_QUEUE_TABLE",),
        ),
        control_table=env_str(
            "SYNTHETIC_CONTROL_TABLE",
            "simulation_state_control",
            aliases=("PRODUCER_CONTROL_TABLE",),
        ),
        bootstrap_servers=env_str(
            "KAFKA_BOOTSTRAP_SERVERS",
            "kafka:9092",
        ),
        producer_worker_id=env_str(
            "SYNTHETIC_PRODUCER_WORKER_ID",
            "producer_worker_test_001",
        ),
        client_id=env_str(
            "SYNTHETIC_PRODUCER_CLIENT_ID",
            "synthetic-telemetry-producer",
            aliases=("KAFKA_CLIENT_ID", "SYNTHETIC_PRODUCER_GROUP_ID"),
        ),
        max_batches=env_optional_int(
            "PRODUCER_MAX_BATCHES_LIMIT",
            default=None,
            aliases=("SYNTHETIC_PRODUCER_MAX_BATCHES_LIMIT",),
        ),
        stop_on_failure=True,
        flush_timeout_seconds=env_float(
            "PRODUCER_FLUSH_TIMEOUT_SECONDS",
            30.0,
        ),
    )

    print(results)


if __name__ == "__main__":
    main()