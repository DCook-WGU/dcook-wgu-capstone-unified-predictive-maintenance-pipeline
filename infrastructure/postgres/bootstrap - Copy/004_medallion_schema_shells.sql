-- =============================================================================
-- 004_medallion_schema_shells.sql
-- Purpose: Create lightweight Bronze/Silver/Gold SQL-facing shell tables.
-- =============================================================================

\echo '[db-bootstrap] 004 medallion schema shells'

CREATE TABLE IF NOT EXISTS bronze."sensor_observations" (
    bronze_id BIGSERIAL PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    asset_id TEXT,
    event_time TIMESTAMPTZ,
    event_step BIGINT,
    time_index BIGINT,
    source_table TEXT,
    source_row_id TEXT,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    meta_truth_hash TEXT,
    meta_parent_truth_hash TEXT,
    meta_ingested_at_utc TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS "idx_bronze_sensor_observations_dataset_run"
ON bronze."sensor_observations" (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS "idx_bronze_sensor_observations_time_index"
ON bronze."sensor_observations" (dataset_id, run_id, time_index);

CREATE TABLE IF NOT EXISTS silver."sensor_observation_features" (
    silver_id BIGSERIAL PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    asset_id TEXT,
    event_time TIMESTAMPTZ,
    event_step BIGINT,
    time_index BIGINT,
    feature_set_id TEXT,
    features_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    quality_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    meta_truth_hash TEXT,
    meta_parent_truth_hash TEXT,
    meta_processed_at_utc TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS "idx_silver_features_dataset_run"
ON silver."sensor_observation_features" (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS "idx_silver_features_feature_set"
ON silver."sensor_observation_features" (dataset_id, run_id, feature_set_id);

CREATE TABLE IF NOT EXISTS gold."preprocessed_features" (
    gold_preprocessed_id BIGSERIAL PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    asset_id TEXT,
    event_time TIMESTAMPTZ,
    event_step BIGINT,
    time_index BIGINT,
    split_name TEXT,
    feature_set_id TEXT,
    features_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    label_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    meta_truth_hash TEXT,
    meta_parent_truth_hash TEXT,
    meta_processed_at_utc TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS "idx_gold_preprocessed_dataset_run"
ON gold."preprocessed_features" (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS "idx_gold_preprocessed_split"
ON gold."preprocessed_features" (dataset_id, run_id, split_name);

CREATE TABLE IF NOT EXISTS gold."anomaly_detection_scores" (
    gold_id BIGSERIAL PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    asset_id TEXT,
    event_time TIMESTAMPTZ,
    event_step BIGINT,
    time_index BIGINT,
    model_name TEXT NOT NULL,
    model_stage TEXT,
    anomaly_score DOUBLE PRECISION,
    anomaly_flag BOOLEAN,
    alert_severity TEXT,
    evidence_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    meta_truth_hash TEXT,
    meta_parent_truth_hash TEXT,
    meta_scored_at_utc TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS "idx_gold_scores_dataset_run"
ON gold."anomaly_detection_scores" (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS "idx_gold_scores_model_stage"
ON gold."anomaly_detection_scores" (dataset_id, run_id, model_name, model_stage);

CREATE INDEX IF NOT EXISTS "idx_gold_scores_flag"
ON gold."anomaly_detection_scores" (dataset_id, run_id, anomaly_flag);

CREATE TABLE IF NOT EXISTS gold."model_comparison_results" (
    comparison_id BIGSERIAL PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    baseline_model TEXT,
    comparison_model TEXT,
    alert_count_baseline BIGINT,
    alert_count_comparison BIGINT,
    precision_baseline DOUBLE PRECISION,
    precision_comparison DOUBLE PRECISION,
    recall_baseline DOUBLE PRECISION,
    recall_comparison DOUBLE PRECISION,
    f1_baseline DOUBLE PRECISION,
    f1_comparison DOUBLE PRECISION,
    comparison_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS "idx_gold_model_comparison_dataset_run"
ON gold."model_comparison_results" (dataset_id, run_id);
