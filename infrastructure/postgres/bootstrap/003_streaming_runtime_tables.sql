-- =============================================================================
-- 003_streaming_runtime_tables.sql
-- Purpose: Create producer/consumer runtime tables, indexes, and seed records.
-- =============================================================================

\echo '[db-bootstrap] 003 streaming runtime tables'

CREATE TABLE IF NOT EXISTS :"capstone_schema".:"producer_control_table" (
    control_id BIGSERIAL PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    producer_topic TEXT,
    producer_batch_size INTEGER NOT NULL DEFAULT 500,
    producer_poll_seconds DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    max_send_attempts INTEGER NOT NULL DEFAULT 3,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, run_id)
);

CREATE INDEX IF NOT EXISTS "idx_simulation_state_control_dataset_run"
ON :"capstone_schema".:"producer_control_table" (dataset_id, run_id);

CREATE TABLE IF NOT EXISTS :"capstone_schema"."simulation_timing_config" (
    config_id BIGSERIAL PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    simulation_start_datetime TIMESTAMPTZ NOT NULL,
    sampling_interval_seconds DOUBLE PRECISION NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT "uq_simulation_timing_config_dataset_run"
        UNIQUE (dataset_id, run_id)
);

CREATE INDEX IF NOT EXISTS "idx_simulation_timing_config_dataset_run"
ON :"capstone_schema"."simulation_timing_config" (dataset_id, run_id);

CREATE TABLE IF NOT EXISTS :"capstone_schema".:"producer_queue_table" (
    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    message_key TEXT NOT NULL,
    generated_row_id TEXT,
    observation_index BIGINT NOT NULL,
    observation_timestamp TIMESTAMPTZ,
    message_sequence_index BIGINT NOT NULL,
    batch_id BIGINT,
    row_in_batch BIGINT,
    global_cycle_id BIGINT,
    stream_state TEXT,
    phase TEXT,
    created_at TIMESTAMPTZ,
    meta_episode_id TEXT,
    meta_primary_fault_type TEXT,
    meta_magnitude DOUBLE PRECISION,
    sensor_name TEXT NOT NULL,
    sensor_index BIGINT NOT NULL,
    sensor_value DOUBLE PRECISION,
    is_telemetry_event BOOLEAN,
    telemetry_event_type TEXT,
    producer_send_attempt INTEGER NOT NULL DEFAULT 1,
    queue_status TEXT NOT NULL DEFAULT 'pending',
    queued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    claim_token TEXT,
    claimed_at TIMESTAMPTZ,
    producer_topic TEXT,
    producer_worker_id TEXT,
    producer_sent_at TIMESTAMPTZ,
    producer_ack_at TIMESTAMPTZ,
    producer_delivery_status TEXT,
    producer_delivery_error TEXT,
    CONSTRAINT "chk_send_queue_status"
        CHECK (queue_status IN ('pending', 'claimed', 'sent', 'failed')),
    CONSTRAINT "uq_send_queue_message_key"
        UNIQUE (message_key)
);

CREATE INDEX IF NOT EXISTS "idx_send_queue_dataset_run_claim"
ON :"capstone_schema".:"producer_queue_table"
(dataset_id, run_id, queue_status, observation_index, message_sequence_index, sensor_index);

CREATE INDEX IF NOT EXISTS "idx_send_queue_status"
ON :"capstone_schema".:"producer_queue_table" (queue_status);

CREATE INDEX IF NOT EXISTS "idx_send_queue_topic_status"
ON :"capstone_schema".:"producer_queue_table" (producer_topic, queue_status);

CREATE INDEX IF NOT EXISTS "idx_send_queue_sent_at"
ON :"capstone_schema".:"producer_queue_table" (producer_sent_at);

CREATE INDEX IF NOT EXISTS "idx_send_queue_claim_token"
ON :"capstone_schema".:"producer_queue_table" (claim_token);

CREATE INDEX IF NOT EXISTS "idx_send_queue_claimed_at"
ON :"capstone_schema".:"producer_queue_table" (claimed_at);

CREATE INDEX IF NOT EXISTS "idx_send_queue_message_key"
ON :"capstone_schema".:"producer_queue_table" (message_key);

CREATE TABLE IF NOT EXISTS :"capstone_schema".:"consumer_target_table" (
    consumed_id BIGSERIAL PRIMARY KEY,
    kafka_topic TEXT,
    kafka_partition INTEGER,
    kafka_offset BIGINT,
    kafka_key TEXT,
    kafka_timestamp TIMESTAMPTZ,
    message_key TEXT,
    dataset_id TEXT,
    run_id TEXT,
    asset_id TEXT,
    generated_row_id TEXT,
    observation_index BIGINT,
    observation_timestamp TIMESTAMPTZ,
    message_sequence_index BIGINT,
    batch_id BIGINT,
    row_in_batch BIGINT,
    global_cycle_id BIGINT,
    stream_state TEXT,
    phase TEXT,
    created_at TIMESTAMPTZ,
    meta_episode_id TEXT,
    meta_primary_fault_type TEXT,
    meta_magnitude DOUBLE PRECISION,
    sensor_name TEXT,
    sensor_index BIGINT,
    sensor_value DOUBLE PRECISION,
    is_telemetry_event BOOLEAN,
    telemetry_event_type TEXT,
    producer_topic TEXT,
    producer_worker_id TEXT,
    producer_sent_at TIMESTAMPTZ,
    producer_ack_at TIMESTAMPTZ,
    producer_delivery_status TEXT,
    producer_delivery_error TEXT,
    consumer_group_id TEXT,
    consumer_worker_id TEXT,
    consumed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    raw_payload JSONB,
    consumer_received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload_json TEXT,
    is_duplicate BOOLEAN NOT NULL DEFAULT FALSE,
    rebuild_status TEXT,
    CONSTRAINT "uq_consumed_topic_partition_offset"
        UNIQUE (kafka_topic, kafka_partition, kafka_offset)
);

CREATE INDEX IF NOT EXISTS "idx_consumed_dataset_run"
ON :"capstone_schema".:"consumer_target_table" (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS "idx_consumed_message_key"
ON :"capstone_schema".:"consumer_target_table" (message_key);

CREATE INDEX IF NOT EXISTS "idx_consumed_observation_sensor"
ON :"capstone_schema".:"consumer_target_table" (dataset_id, run_id, observation_index, sensor_index);

CREATE INDEX IF NOT EXISTS "idx_consumed_topic_partition_offset"
ON :"capstone_schema".:"consumer_target_table" (kafka_topic, kafka_partition, kafka_offset);

CREATE INDEX IF NOT EXISTS "idx_consumed_raw_payload_gin"
ON :"capstone_schema".:"consumer_target_table" USING GIN (raw_payload);

CREATE INDEX IF NOT EXISTS "idx_consumed_rebuild_status"
ON :"capstone_schema".:"consumer_target_table" (rebuild_status);

INSERT INTO :"capstone_schema"."simulation_timing_config" (
    dataset_id,
    run_id,
    simulation_start_datetime,
    sampling_interval_seconds,
    is_active
)
VALUES (
    :'synthetic_dataset_id',
    :'synthetic_run_id',
    now(),
    60.0,
    TRUE
)
ON CONFLICT (dataset_id, run_id)
DO UPDATE SET
    is_active = EXCLUDED.is_active;

INSERT INTO :"capstone_schema".:"producer_control_table" (
    dataset_id,
    run_id,
    is_enabled,
    producer_topic,
    producer_batch_size,
    producer_poll_seconds,
    max_send_attempts,
    updated_at,
    created_at
)
VALUES (
    :'synthetic_dataset_id',
    :'synthetic_run_id',
    TRUE,
    :'kafka_topic',
    :producer_batch_size,
    :producer_poll_seconds,
    :max_send_attempts,
    now(),
    now()
)
ON CONFLICT (dataset_id, run_id)
DO UPDATE SET
    is_enabled = EXCLUDED.is_enabled,
    producer_topic = EXCLUDED.producer_topic,
    producer_batch_size = EXCLUDED.producer_batch_size,
    producer_poll_seconds = EXCLUDED.producer_poll_seconds,
    max_send_attempts = EXCLUDED.max_send_attempts,
    updated_at = now();
