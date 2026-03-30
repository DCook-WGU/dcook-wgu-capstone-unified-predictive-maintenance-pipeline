from __future__ import annotations

import json
import os
import socket

from utils.postgres_util import get_engine_from_env
from utils.kafka_consumer_adapter import run_kafka_consumer_to_postgres_loop

SCHEMA = "capstone"
TOPIC = "pump.telemetry.synthetic"
TABLE_NAME = "synthetic_sensor_messages_consumed_stage"
CONSUMER_GROUP_ID = "synthetic-telemetry-consumer-group"


def _get_env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return str(default)
    return str(value).strip()


def _get_env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return int(str(value).strip())


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return float(default)
    return float(str(value).strip())


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return bool(default)

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False

    raise ValueError(f"Invalid boolean value for {name}: {value!r}")


def _resolve_consumer_worker_id() -> str:
    explicit = os.getenv("CONSUMER_WORKER_ID")
    if explicit is not None and str(explicit).strip() != "":
        return str(explicit).strip()

    hostname = socket.gethostname().strip()
    if hostname:
        return hostname

    return "consumer_worker_001"


CONSUMER_WORKER_ID = _resolve_consumer_worker_id()
MAX_MESSAGES = _get_env_int("KAFKA_CONSUMER_MAX_MESSAGES", 5200)
POLL_TIMEOUT_SECONDS = _get_env_float("KAFKA_CONSUMER_POLL_TIMEOUT_SECONDS", 1.0)
AUTO_OFFSET_RESET = _get_env_str("KAFKA_CONSUMER_AUTO_OFFSET_RESET", "earliest")
MAX_BATCHES = _get_env_int("KAFKA_CONSUMER_MAX_BATCHES", None)
IDLE_SLEEP_SECONDS = _get_env_float("KAFKA_CONSUMER_IDLE_SLEEP_SECONDS", 0.0)
STOP_ON_EMPTY = _get_env_bool("KAFKA_CONSUMER_STOP_ON_EMPTY", False)
PROGRESS_LOG_EVERY_BATCHES = _get_env_int("KAFKA_CONSUMER_PROGRESS_LOG_EVERY_BATCHES", 25)


def main() -> None:
    engine = get_engine_from_env()

    summary = run_kafka_consumer_to_postgres_loop(
        engine=engine,
        topic=TOPIC,
        schema=SCHEMA,
        table_name=TABLE_NAME,
        max_messages=MAX_MESSAGES,
        poll_timeout_seconds=POLL_TIMEOUT_SECONDS,
        consumer_group_id=CONSUMER_GROUP_ID,
        consumer_worker_id=CONSUMER_WORKER_ID,
        auto_offset_reset=AUTO_OFFSET_RESET,
        max_batches=MAX_BATCHES,
        idle_sleep_seconds=IDLE_SLEEP_SECONDS,
        stop_on_empty=STOP_ON_EMPTY,
        progress_log_every_batches=PROGRESS_LOG_EVERY_BATCHES,
        return_result_history=False,
    )

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
