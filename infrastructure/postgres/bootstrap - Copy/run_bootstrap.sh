#!/usr/bin/env bash
set -Eeuo pipefail

# =============================================================================
# Capstone Postgres Bootstrap Runner
# =============================================================================
# This wrapper validates environment variables, then executes the split SQL
# bootstrap files in order. It is safe to rerun against an existing local/dev
# database.
# =============================================================================

echo "[db-bootstrap] Starting split capstone database bootstrap..."

BOOTSTRAP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${POSTGRES_HOST:=postgres}"
: "${POSTGRES_PORT:=5432}"
: "${POSTGRES_DB:?POSTGRES_DB is required}"
: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"

: "${CAPSTONE_SCHEMA:=capstone}"
: "${SYNTHETIC_DATASET_ID:=pump_synthetic_v1}"
: "${SYNTHETIC_RUN_ID:=synthetic_run_001}"
: "${KAFKA_TOPIC:=pump.telemetry.synthetic}"

: "${KAFKA_PRODUCER_DB_USER:=kafka_producer}"
: "${KAFKA_PRODUCER_DB_PASSWORD:?KAFKA_PRODUCER_DB_PASSWORD is required}"
: "${KAFKA_INGEST_DB_USER:=kafka_ingest}"
: "${KAFKA_INGEST_DB_PASSWORD:?KAFKA_INGEST_DB_PASSWORD is required}"

: "${PRODUCER_QUEUE_TABLE:=synthetic_sensor_messages_send_queue}"
: "${PRODUCER_CONTROL_TABLE:=simulation_state_control}"
: "${CONSUMER_TARGET_TABLE:=synthetic_sensor_messages_consumed_stage}"

: "${PRODUCER_BATCH_SIZE:=5200}"
: "${PRODUCER_POLL_SECONDS:=0.0}"
: "${MAX_SEND_ATTEMPTS:=3}"

validate_identifier() {
  local name="$1"
  local value="$2"

  if [[ ! "$value" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "[db-bootstrap] Invalid SQL identifier for ${name}: ${value}" >&2
    exit 1
  fi
}

validate_integer() {
  local name="$1"
  local value="$2"

  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "[db-bootstrap] Invalid integer for ${name}: ${value}" >&2
    exit 1
  fi
}

validate_float() {
  local name="$1"
  local value="$2"

  if [[ ! "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[db-bootstrap] Invalid numeric value for ${name}: ${value}" >&2
    exit 1
  fi
}

validate_identifier "CAPSTONE_SCHEMA" "$CAPSTONE_SCHEMA"
validate_identifier "KAFKA_PRODUCER_DB_USER" "$KAFKA_PRODUCER_DB_USER"
validate_identifier "KAFKA_INGEST_DB_USER" "$KAFKA_INGEST_DB_USER"
validate_identifier "PRODUCER_QUEUE_TABLE" "$PRODUCER_QUEUE_TABLE"
validate_identifier "PRODUCER_CONTROL_TABLE" "$PRODUCER_CONTROL_TABLE"
validate_identifier "CONSUMER_TARGET_TABLE" "$CONSUMER_TARGET_TABLE"

validate_integer "PRODUCER_BATCH_SIZE" "$PRODUCER_BATCH_SIZE"
validate_integer "MAX_SEND_ATTEMPTS" "$MAX_SEND_ATTEMPTS"
validate_float "PRODUCER_POLL_SECONDS" "$PRODUCER_POLL_SECONDS"

export PGPASSWORD="$POSTGRES_PASSWORD"

PSQL_BASE=(
  psql
  -v ON_ERROR_STOP=1
  -h "$POSTGRES_HOST"
  -p "$POSTGRES_PORT"
  -U "$POSTGRES_USER"
  -d "$POSTGRES_DB"
  -v "capstone_schema=${CAPSTONE_SCHEMA}"
  -v "synthetic_dataset_id=${SYNTHETIC_DATASET_ID}"
  -v "synthetic_run_id=${SYNTHETIC_RUN_ID}"
  -v "kafka_topic=${KAFKA_TOPIC}"
  -v "producer_role=${KAFKA_PRODUCER_DB_USER}"
  -v "producer_password=${KAFKA_PRODUCER_DB_PASSWORD}"
  -v "ingest_role=${KAFKA_INGEST_DB_USER}"
  -v "ingest_password=${KAFKA_INGEST_DB_PASSWORD}"
  -v "producer_queue_table=${PRODUCER_QUEUE_TABLE}"
  -v "producer_control_table=${PRODUCER_CONTROL_TABLE}"
  -v "consumer_target_table=${CONSUMER_TARGET_TABLE}"
  -v "producer_batch_size=${PRODUCER_BATCH_SIZE}"
  -v "producer_poll_seconds=${PRODUCER_POLL_SECONDS}"
  -v "max_send_attempts=${MAX_SEND_ATTEMPTS}"
)

run_sql_file() {
  local file_name="$1"
  echo "[db-bootstrap] Running ${file_name}..."
  "${PSQL_BASE[@]}" -f "${BOOTSTRAP_DIR}/${file_name}"
}

run_sql_file "001_roles_schemas.sql"
run_sql_file "002_core_metadata_tables.sql"
run_sql_file "003_streaming_runtime_tables.sql"
run_sql_file "004_medallion_schema_shells.sql"
run_sql_file "005_synthetic_notebook_stage_tables.sql"
run_sql_file "006_silver_eda_summary_tables.sql"
run_sql_file "007_grants_ownership.sql"

echo "[db-bootstrap] Split bootstrap complete."
