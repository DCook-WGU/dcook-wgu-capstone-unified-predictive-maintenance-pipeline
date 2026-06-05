\echo '[db-bootstrap] 005 synthetic notebook stage tables'

-- =============================================================================
-- Reset synthetic notebook stage tables
--
-- These are generated/intermediate notebook-stage tables, not durable runtime
-- producer/consumer tables. Runtime queue and consumer tables are created in:
--
--   003_streaming_runtime_tables.sql
--
-- Do NOT drop these here:
--   capstone.synthetic_sensor_messages_send_queue
--   capstone.synthetic_sensor_messages_consumed_stage
--
-- Dropping those runtime tables here removes the tables created by 003 and
-- causes the final grant/ownership file to fail.
-- =============================================================================

DROP TABLE IF EXISTS capstone.synthetic_pump_stream CASCADE;
DROP TABLE IF EXISTS capstone.synthetic_observations_premelt_stage CASCADE;
DROP TABLE IF EXISTS capstone.synthetic_observations_timestamped_stage CASCADE;
DROP TABLE IF EXISTS capstone.synthetic_sensor_messages_stage CASCADE;
DROP TABLE IF EXISTS capstone.synthetic_sensor_observations_rebuilt_stage CASCADE;
DROP TABLE IF EXISTS capstone.synthetic_sensor_rebuild_comparison_stage CASCADE;
DROP TABLE IF EXISTS capstone.synthetic_sensor_observations_final_aligned_stage CASCADE;
DROP TABLE IF EXISTS capstone.synthetic_sensor_observations_final_output CASCADE;
DROP TABLE IF EXISTS capstone.bronze_observations_input_stage CASCADE;


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
-- Rebuilt synthetic observations stage
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS capstone.synthetic_sensor_observations_rebuilt_stage (
    rebuilt_observation_id BIGSERIAL PRIMARY KEY,

    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    asset_id TEXT,

    generated_row_id TEXT,
    observation_index BIGINT,
    observation_timestamp TIMESTAMPTZ,

    event_time TIMESTAMPTZ,
    event_step BIGINT,
    time_index BIGINT,

    machine_status TEXT,
    operational_state TEXT,
    anomaly_flag INTEGER,

    rebuild_status TEXT,
    rebuild_is_complete BOOLEAN DEFAULT FALSE,

    payload JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_synthetic_rebuilt_dataset_run
    ON capstone.synthetic_sensor_observations_rebuilt_stage (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS idx_synthetic_rebuilt_observation_index
    ON capstone.synthetic_sensor_observations_rebuilt_stage (dataset_id, run_id, observation_index);

CREATE INDEX IF NOT EXISTS idx_synthetic_rebuilt_generated_row
    ON capstone.synthetic_sensor_observations_rebuilt_stage (generated_row_id);


-- -----------------------------------------------------------------------------
-- Rebuild comparison stage
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS capstone.synthetic_sensor_rebuild_comparison_stage (
    comparison_id BIGSERIAL PRIMARY KEY,

    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    asset_id TEXT,

    generated_row_id TEXT,
    observation_index BIGINT,
    observation_timestamp TIMESTAMPTZ,

    all_fields_match BOOLEAN,
    mismatch_count INTEGER,
    mismatch_columns JSONB,

    original_payload JSONB,
    rebuilt_payload JSONB,
    comparison_payload JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_synthetic_rebuild_comparison_dataset_run
    ON capstone.synthetic_sensor_rebuild_comparison_stage (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS idx_synthetic_rebuild_comparison_observation_index
    ON capstone.synthetic_sensor_rebuild_comparison_stage (dataset_id, run_id, observation_index);

CREATE INDEX IF NOT EXISTS idx_synthetic_rebuild_comparison_match
    ON capstone.synthetic_sensor_rebuild_comparison_stage (all_fields_match);


-- -----------------------------------------------------------------------------
-- Compact final synthetic output
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS capstone.synthetic_sensor_observations_final_output (
    final_output_id BIGSERIAL PRIMARY KEY,

    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    asset_id TEXT,

    generated_row_id TEXT,
    observation_index BIGINT,
    observation_timestamp TIMESTAMPTZ,

    machine_status TEXT,
    operational_state TEXT,
    anomaly_flag INTEGER,

    payload JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_synthetic_final_output_dataset_run
    ON capstone.synthetic_sensor_observations_final_output (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS idx_synthetic_final_output_observation_index
    ON capstone.synthetic_sensor_observations_final_output (dataset_id, run_id, observation_index);

CREATE INDEX IF NOT EXISTS idx_synthetic_final_output_generated_row
    ON capstone.synthetic_sensor_observations_final_output (generated_row_id);

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


