\echo '[db-bootstrap] 006 synthetic notebook stage tables'

-- =============================================================================
-- Synthetic Notebook Stage Tables
-- =============================================================================
-- Purpose:
--   Create empty table shells for the synthetic notebook pipeline so a fresh
--   Docker/Postgres reset has the expected staging relations available.
--
-- Important:
--   These tables are intentionally lightweight compatibility shells.
--   The synthetic notebooks still own data generation, transformation, and
--   final dataframe shape. Some notebooks may replace these tables with richer
--   dataframe-derived schemas when they run.
--
-- Why this exists:
--   After a Docker volume reset, runtime checks and dry-run validation can fail
--   with "relation does not exist" before the notebooks are rerun. These shells
--   make the database recovery path more predictable.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Wide synthetic source stream
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS capstone.synthetic_pump_stream (
    synthetic_row_id BIGSERIAL PRIMARY KEY,

    dataset_id TEXT NOT NULL DEFAULT current_setting('app.dataset_id', true),
    run_id TEXT NOT NULL DEFAULT current_setting('app.run_id', true),
    asset_id TEXT,

    event_time TIMESTAMPTZ,
    event_step BIGINT,
    time_index BIGINT,

    machine_status TEXT,
    operational_state TEXT,
    anomaly_flag INTEGER,

    -- Flexible payload allows this shell to survive changes in generated sensor
    -- columns before the notebook writes/replaces the final dataframe schema.
    payload JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_synthetic_pump_stream_dataset_run
    ON capstone.synthetic_pump_stream (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS idx_synthetic_pump_stream_event_time
    ON capstone.synthetic_pump_stream (event_time);


-- -----------------------------------------------------------------------------
-- Premelt observation stage
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS capstone.synthetic_observations_premelt_stage (
    observation_id BIGSERIAL PRIMARY KEY,

    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    asset_id TEXT,

    source_row_id BIGINT,
    event_time TIMESTAMPTZ,
    event_step BIGINT,
    time_index BIGINT,

    machine_status TEXT,
    operational_state TEXT,
    anomaly_flag INTEGER,

    payload JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_synthetic_premelt_dataset_run
    ON capstone.synthetic_observations_premelt_stage (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS idx_synthetic_premelt_time
    ON capstone.synthetic_observations_premelt_stage (event_time, time_index);


-- -----------------------------------------------------------------------------
-- Timestamped observation stage
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS capstone.synthetic_observations_timestamped_stage (
    observation_id BIGSERIAL PRIMARY KEY,

    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    asset_id TEXT,

    source_row_id BIGINT,
    event_time TIMESTAMPTZ,
    event_step BIGINT,
    time_index BIGINT,

    machine_status TEXT,
    operational_state TEXT,
    anomaly_flag INTEGER,

    payload JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_synthetic_timestamped_dataset_run
    ON capstone.synthetic_observations_timestamped_stage (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS idx_synthetic_timestamped_time
    ON capstone.synthetic_observations_timestamped_stage (event_time, time_index);


-- -----------------------------------------------------------------------------
-- Long-format synthetic sensor messages stage
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS capstone.synthetic_sensor_messages_stage (
    message_stage_id BIGSERIAL PRIMARY KEY,

    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    asset_id TEXT,

    event_time TIMESTAMPTZ,
    event_step BIGINT,
    time_index BIGINT,

    sensor_name TEXT NOT NULL,
    sensor_value DOUBLE PRECISION,
    sensor_unit TEXT,

    machine_status TEXT,
    operational_state TEXT,
    anomaly_flag INTEGER,

    kafka_topic TEXT,
    message_key TEXT,
    message_payload JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_synthetic_messages_stage_dataset_run
    ON capstone.synthetic_sensor_messages_stage (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS idx_synthetic_messages_stage_sensor
    ON capstone.synthetic_sensor_messages_stage (sensor_name);

CREATE INDEX IF NOT EXISTS idx_synthetic_messages_stage_time
    ON capstone.synthetic_sensor_messages_stage (event_time, time_index);


-- -----------------------------------------------------------------------------
-- Final aligned/rebuilt synthetic observations
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS capstone.synthetic_sensor_observations_final_aligned_stage (
    aligned_observation_id BIGSERIAL PRIMARY KEY,

    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    asset_id TEXT,

    event_time TIMESTAMPTZ,
    event_step BIGINT,
    time_index BIGINT,

    machine_status TEXT,
    operational_state TEXT,
    anomaly_flag INTEGER,

    payload JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_synthetic_final_aligned_dataset_run
    ON capstone.synthetic_sensor_observations_final_aligned_stage (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS idx_synthetic_final_aligned_time
    ON capstone.synthetic_sensor_observations_final_aligned_stage (event_time, time_index);


-- -----------------------------------------------------------------------------
-- Bronze handoff input stage
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS capstone.bronze_observations_input_stage (
    bronze_input_id BIGSERIAL PRIMARY KEY,

    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    asset_id TEXT,

    event_time TIMESTAMPTZ,
    event_step BIGINT,
    time_index BIGINT,

    machine_status TEXT,
    operational_state TEXT,
    anomaly_flag INTEGER,

    payload JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bronze_input_dataset_run
    ON capstone.bronze_observations_input_stage (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS idx_bronze_input_time
    ON capstone.bronze_observations_input_stage (event_time, time_index);


-- -----------------------------------------------------------------------------
-- Runtime grants
-- -----------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA capstone TO kafka_producer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA capstone TO kafka_ingest;

GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA capstone TO kafka_producer;
GRANT USAGE, SELECT, UPDATE ON ALL SEQUENCES IN SCHEMA capstone TO kafka_ingest;