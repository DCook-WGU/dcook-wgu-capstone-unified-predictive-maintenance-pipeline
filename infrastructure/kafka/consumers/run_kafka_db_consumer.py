from utils.postgres_util import get_engine_from_env
from utils.kafka_consumer_adapter import run_kafka_consumer_to_postgres_loop

SCHEMA = "capstone"
TOPIC = "pump.telemetry.synthetic"
TABLE_NAME = "synthetic_sensor_messages_consumed_stage"
CONSUMER_GROUP_ID = "synthetic-telemetry-consumer-group"
CONSUMER_WORKER_ID = "consumer_worker_001"

# Keep this small for now so each poll/land/commit cycle finishes faster.
MAX_MESSAGES = 25
POLL_TIMEOUT_SECONDS = 1.0
AUTO_OFFSET_RESET = "earliest"

MAX_BATCHES = None
IDLE_SLEEP_SECONDS = 0.10
STOP_ON_EMPTY = False


def main() -> None:
    engine = get_engine_from_env()

    results = run_kafka_consumer_to_postgres_loop(
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
    )

    print(results)


if __name__ == "__main__":
    main()