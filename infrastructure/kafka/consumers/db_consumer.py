from __future__ import annotations

import json
import socket

from utils.core.env_helpers import (
    env_bool,
    env_float,
    env_int,
    env_optional_int,
    env_raw,
    env_str,
)
from utils.database.postgres import get_engine_from_env
from utils.synthetic.pipeline.kafka_consumer_adapter import run_kafka_consumer_to_postgres_loop


def resolve_consumer_worker_id() -> str:
    explicit = env_raw(
        "SYNTHETIC_CONSUMER_WORKER_ID",
        aliases=("CONSUMER_WORKER_ID",),
    )

    if explicit is not None:
        return explicit

    hostname = socket.gethostname().strip()

    if hostname:
        return hostname

    return "consumer_worker_001"


def main() -> None:
    engine = get_engine_from_env()

    schema = env_str(
        "CAPSTONE_SCHEMA",
        "capstone",
        aliases=("CONSUMER_SCHEMA",),
    )

    topic = env_str(
        "SYNTHETIC_KAFKA_TOPIC",
        "pump.telemetry.synthetic",
        aliases=("KAFKA_TOPIC",),
    )

    target_table = env_str(
        "SYNTHETIC_CONSUMED_MESSAGES_TABLE",
        "synthetic_sensor_messages_consumed_stage",
        aliases=("CONSUMER_TARGET_TABLE",),
    )

    consumer_group_id = env_str(
        "SYNTHETIC_CONSUMER_GROUP_ID",
        "synthetic-telemetry-consumer-group",
        aliases=("KAFKA_CONSUMER_GROUP_ID", "CONSUMER_GROUP_ID"),
    )

    summary = run_kafka_consumer_to_postgres_loop(
        engine=engine,
        topic=topic,
        schema=schema,
        table_name=target_table,
        max_messages=env_int("CONSUMER_BATCH_SIZE", 5200),
        poll_timeout_seconds=env_float("CONSUMER_POLL_TIMEOUT_SECONDS", 1.0),
        consumer_group_id=consumer_group_id,
        consumer_worker_id=resolve_consumer_worker_id(),
        auto_offset_reset=env_str(
            "CONSUMER_AUTO_OFFSET_RESET",
            "earliest",
            aliases=("SYNTHETIC_AUTO_OFFSET_RESET",),
        ),
        max_batches=env_optional_int(
            "CONSUMER_MAX_BATCHES_LIMIT",
            default=None,
        ),
        idle_sleep_seconds=env_float(
            "CONSUMER_IDLE_SLEEP_SECONDS",
            0.0,
        ),
        stop_on_empty=env_bool(
            "CONSUMER_STOP_ON_EMPTY",
            False,
        ),
        progress_log_every_batches=env_int(
            "CONSUMER_PROGRESS_LOG_EVERY_BATCHES",
            25,
        ),
        return_result_history=False,
    )

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()