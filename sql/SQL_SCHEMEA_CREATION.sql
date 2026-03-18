BEGIN;

-- =========================================================
-- 1) SCHEMAS
-- =========================================================
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS audit;

-- =========================================================
-- 2) CORE: shared run / dataset metadata
-- =========================================================
CREATE TABLE IF NOT EXISTS core.pipeline_runs (
    run_id                TEXT PRIMARY KEY,
    pipeline_mode         TEXT,
    dataset_name          TEXT NOT NULL,
    layer_name            TEXT,
    notebook_name         TEXT,
    step_name             TEXT,
    started_at_utc        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at_utc      TIMESTAMPTZ,
    status                TEXT,
    created_by            TEXT,
    notes                 TEXT
);

CREATE TABLE IF NOT EXISTS core.dataset_registry (
    dataset_name          TEXT PRIMARY KEY,
    dataset_display_name  TEXT,
    source_name           TEXT,
    source_type           TEXT,
    grain_description     TEXT,
    target_column         TEXT,
    created_at_utc        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS core.dataset_versions (
    dataset_version_id    BIGSERIAL PRIMARY KEY,
    dataset_name          TEXT NOT NULL REFERENCES core.dataset_registry(dataset_name),
    layer_name            TEXT NOT NULL,
    version_label         TEXT NOT NULL,
    truth_hash            TEXT,
    parent_truth_hash     TEXT,
    file_path             TEXT,
    file_format           TEXT,
    row_count             BIGINT,
    column_count          BIGINT,
    created_at_utc        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (dataset_name, layer_name, version_label)
);

-- =========================================================
-- 3) AUDIT: truth / logs / lineage
-- =========================================================
CREATE TABLE IF NOT EXISTS audit.truth_records (
    truth_id              BIGSERIAL PRIMARY KEY,
    truth_hash            TEXT NOT NULL UNIQUE,
    parent_truth_hash     TEXT,
    dataset_name          TEXT NOT NULL,
    layer_name            TEXT NOT NULL,
    notebook_name         TEXT,
    step_name             TEXT,
    run_id                TEXT,
    truth_payload         JSONB NOT NULL,
    created_at_utc        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit.pipeline_logs (
    log_id                BIGSERIAL PRIMARY KEY,
    run_id                TEXT,
    dataset_name          TEXT,
    layer_name            TEXT,
    step_name             TEXT,
    log_level             TEXT,
    message               TEXT,
    details               JSONB,
    logged_at_utc         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit.lineage_events (
    lineage_event_id      BIGSERIAL PRIMARY KEY,
    run_id                TEXT,
    dataset_name          TEXT NOT NULL,
    from_layer            TEXT,
    to_layer              TEXT,
    input_artifact        TEXT,
    output_artifact       TEXT,
    parent_truth_hash     TEXT,
    output_truth_hash     TEXT,
    event_payload         JSONB,
    created_at_utc        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =========================================================
-- 4) BRONZE: raw landed data
--    Flexible enough for your canonical/raw stage
-- =========================================================
CREATE TABLE IF NOT EXISTS bronze.sensor_readings (
    bronze_id                     BIGSERIAL PRIMARY KEY,

    -- business / observation grain
    observation_id                BIGINT,
    event_timestamp               TIMESTAMPTZ,
    machine_status                TEXT,

    -- raw sensor payload
    sensor_payload                JSONB NOT NULL,

    -- medallion metadata
    meta__dataset_name            TEXT NOT NULL,
    meta__layer_name              TEXT NOT NULL DEFAULT 'bronze',
    meta__run_id                  TEXT,
    meta__truth_hash              TEXT,
    meta__parent_truth_hash       TEXT,
    meta__source_file_name        TEXT,
    meta__source_file_path        TEXT,
    meta__source_row_index        BIGINT,
    meta__ingested_at_utc         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    meta__record_hash             TEXT,
    meta__is_quarantined          BOOLEAN NOT NULL DEFAULT FALSE,
    meta__quarantine_reason       TEXT
);

CREATE TABLE IF NOT EXISTS bronze.file_ingestions (
    ingestion_id                  BIGSERIAL PRIMARY KEY,
    dataset_name                  TEXT NOT NULL,
    source_file_name              TEXT NOT NULL,
    source_file_path              TEXT,
    file_hash                     TEXT,
    row_count                     BIGINT,
    ingested_at_utc               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id                        TEXT,
    truth_hash                    TEXT,
    status                        TEXT,
    notes                         TEXT
);

-- =========================================================
-- 5) SILVER: cleaned / standardized / feature-aware records
-- =========================================================
CREATE TABLE IF NOT EXISTS silver.sensor_readings (
    silver_id                     BIGSERIAL PRIMARY KEY,

    -- business / observation grain
    observation_id                BIGINT NOT NULL,
    event_timestamp               TIMESTAMPTZ,
    machine_status                TEXT,

    -- standardized wide-form features
    feature_payload               JSONB NOT NULL,

    -- optional flags / quality
    missing_feature_count         INTEGER,
    missing_feature_pct           NUMERIC(8,4),
    outlier_feature_count         INTEGER,
    row_quality_status            TEXT,

    -- medallion metadata
    meta__dataset_name            TEXT NOT NULL,
    meta__layer_name              TEXT NOT NULL DEFAULT 'silver',
    meta__run_id                  TEXT,
    meta__truth_hash              TEXT,
    meta__parent_truth_hash       TEXT,
    meta__source_bronze_id        BIGINT,
    meta__created_at_utc          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    meta__record_hash             TEXT,

    UNIQUE (observation_id, meta__truth_hash)
);

CREATE TABLE IF NOT EXISTS silver.feature_registry (
    feature_registry_id           BIGSERIAL PRIMARY KEY,
    dataset_name                  TEXT NOT NULL,
    feature_name                  TEXT NOT NULL,
    feature_group                 TEXT,
    data_type                     TEXT,
    is_sensor_feature             BOOLEAN NOT NULL DEFAULT TRUE,
    is_target                     BOOLEAN NOT NULL DEFAULT FALSE,
    is_meta                       BOOLEAN NOT NULL DEFAULT FALSE,
    include_in_model              BOOLEAN NOT NULL DEFAULT TRUE,
    notes                         TEXT,
    created_at_utc                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at_utc                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (dataset_name, feature_name)
);

CREATE TABLE IF NOT EXISTS silver.sensor_profiles (
    sensor_profile_id             BIGSERIAL PRIMARY KEY,
    dataset_name                  TEXT NOT NULL,
    feature_name                  TEXT NOT NULL,
    profile_scope                 TEXT NOT NULL,   -- e.g. global / normal / abnormal / recovery
    row_count                     BIGINT,
    mean_value                    DOUBLE PRECISION,
    median_value                  DOUBLE PRECISION,
    std_value                     DOUBLE PRECISION,
    min_value                     DOUBLE PRECISION,
    max_value                     DOUBLE PRECISION,
    q01_value                     DOUBLE PRECISION,
    q05_value                     DOUBLE PRECISION,
    q25_value                     DOUBLE PRECISION,
    q75_value                     DOUBLE PRECISION,
    q95_value                     DOUBLE PRECISION,
    q99_value                     DOUBLE PRECISION,
    missing_pct                   NUMERIC(8,4),
    created_at_utc                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    truth_hash                    TEXT,
    UNIQUE (dataset_name, feature_name, profile_scope, truth_hash)
);

CREATE TABLE IF NOT EXISTS silver.data_quality_results (
    dq_result_id                  BIGSERIAL PRIMARY KEY,
    dataset_name                  TEXT NOT NULL,
    feature_name                  TEXT,
    check_name                    TEXT NOT NULL,
    check_scope                   TEXT,
    check_status                  TEXT,
    severity                      TEXT,
    observed_value                TEXT,
    expected_value                TEXT,
    details                       JSONB,
    run_id                        TEXT,
    truth_hash                    TEXT,
    created_at_utc                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =========================================================
-- 6) GOLD: model outputs / alerts / summaries
-- =========================================================
CREATE TABLE IF NOT EXISTS gold.model_runs (
    model_run_id                  BIGSERIAL PRIMARY KEY,
    run_id                        TEXT NOT NULL,
    dataset_name                  TEXT NOT NULL,
    model_family                  TEXT NOT NULL,   -- baseline_iforest / cascade_iforest / etc
    model_variant                 TEXT,            -- stage1 / stage2 / final / comparison
    config_payload                JSONB,
    metrics_payload               JSONB,
    threshold_payload             JSONB,
    artifact_path                 TEXT,
    truth_hash                    TEXT,
    parent_truth_hash             TEXT,
    created_at_utc                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, dataset_name, model_family, model_variant)
);

CREATE TABLE IF NOT EXISTS gold.baseline_results (
    baseline_result_id            BIGSERIAL PRIMARY KEY,
    run_id                        TEXT NOT NULL,
    dataset_name                  TEXT NOT NULL,
    observation_id                BIGINT NOT NULL,
    event_timestamp               TIMESTAMPTZ,
    anomaly_score                 DOUBLE PRECISION,
    anomaly_threshold             DOUBLE PRECISION,
    is_anomaly                    BOOLEAN NOT NULL,
    label_actual                  TEXT,
    evaluation_bucket             TEXT,
    truth_hash                    TEXT,
    created_at_utc                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gold.cascade_results (
    cascade_result_id             BIGSERIAL PRIMARY KEY,
    run_id                        TEXT NOT NULL,
    dataset_name                  TEXT NOT NULL,
    observation_id                BIGINT NOT NULL,
    event_timestamp               TIMESTAMPTZ,

    stage1_score                  DOUBLE PRECISION,
    stage1_flag                   BOOLEAN,
    stage2_score                  DOUBLE PRECISION,
    stage2_flag                   BOOLEAN,
    stage3_flag                   BOOLEAN,
    final_alert_flag              BOOLEAN NOT NULL,

    label_actual                  TEXT,
    evaluation_bucket             TEXT,
    truth_hash                    TEXT,
    created_at_utc                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gold.alerts (
    alert_id                      BIGSERIAL PRIMARY KEY,
    run_id                        TEXT NOT NULL,
    dataset_name                  TEXT NOT NULL,
    observation_id                BIGINT NOT NULL,
    event_timestamp               TIMESTAMPTZ,
    alert_type                    TEXT NOT NULL,
    alert_severity                TEXT,
    alert_message                 TEXT,
    supporting_features           JSONB,
    source_model                  TEXT,
    truth_hash                    TEXT,
    created_at_utc                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gold.model_comparisons (
    comparison_id                 BIGSERIAL PRIMARY KEY,
    run_id                        TEXT NOT NULL,
    dataset_name                  TEXT NOT NULL,
    baseline_run_id               TEXT,
    cascade_run_id                TEXT,
    metric_name                   TEXT NOT NULL,
    baseline_value                DOUBLE PRECISION,
    cascade_value                 DOUBLE PRECISION,
    delta_value                   DOUBLE PRECISION,
    notes                         TEXT,
    created_at_utc                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =========================================================
-- 7) INDEXES
-- =========================================================

-- bronze
CREATE INDEX IF NOT EXISTS idx_bronze_sensor_readings_obs_id
    ON bronze.sensor_readings (observation_id);

CREATE INDEX IF NOT EXISTS idx_bronze_sensor_readings_event_ts
    ON bronze.sensor_readings (event_timestamp);

CREATE INDEX IF NOT EXISTS idx_bronze_sensor_readings_run_id
    ON bronze.sensor_readings (meta__run_id);

CREATE INDEX IF NOT EXISTS idx_bronze_sensor_readings_truth_hash
    ON bronze.sensor_readings (meta__truth_hash);

CREATE INDEX IF NOT EXISTS idx_bronze_sensor_payload_gin
    ON bronze.sensor_readings
    USING GIN (sensor_payload);

-- silver
CREATE INDEX IF NOT EXISTS idx_silver_sensor_readings_obs_id
    ON silver.sensor_readings (observation_id);

CREATE INDEX IF NOT EXISTS idx_silver_sensor_readings_event_ts
    ON silver.sensor_readings (event_timestamp);

CREATE INDEX IF NOT EXISTS idx_silver_sensor_readings_truth_hash
    ON silver.sensor_readings (meta__truth_hash);

CREATE INDEX IF NOT EXISTS idx_silver_feature_payload_gin
    ON silver.sensor_readings
    USING GIN (feature_payload);

-- gold
CREATE INDEX IF NOT EXISTS idx_gold_baseline_results_obs_id
    ON gold.baseline_results (observation_id);

CREATE INDEX IF NOT EXISTS idx_gold_baseline_results_run_id
    ON gold.baseline_results (run_id);

CREATE INDEX IF NOT EXISTS idx_gold_cascade_results_obs_id
    ON gold.cascade_results (observation_id);

CREATE INDEX IF NOT EXISTS idx_gold_cascade_results_run_id
    ON gold.cascade_results (run_id);

CREATE INDEX IF NOT EXISTS idx_gold_alerts_obs_id
    ON gold.alerts (observation_id);

CREATE INDEX IF NOT EXISTS idx_gold_alerts_run_id
    ON gold.alerts (run_id);

-- audit
CREATE INDEX IF NOT EXISTS idx_audit_truth_records_dataset_layer
    ON audit.truth_records (dataset_name, layer_name);

CREATE INDEX IF NOT EXISTS idx_audit_pipeline_logs_run_id
    ON audit.pipeline_logs (run_id);

CREATE INDEX IF NOT EXISTS idx_audit_lineage_events_run_id
    ON audit.lineage_events (run_id);

COMMIT;