-- =============================================================================
-- 008_gold_sql_contract_extensions.sql
-- Purpose:
--   Optional extension script for stronger Gold SQL output contracts.
--   This is designed to prevent Gold 03a/03b/03c cascade variants from
--   overwriting each other and to make Gold 04/Gold 05 outputs easier to query.
-- =============================================================================

\echo '[db-bootstrap] 008 gold SQL contract extensions'

CREATE SCHEMA IF NOT EXISTS gold;

-- Add identity columns to the existing score table.
ALTER TABLE gold.anomaly_detection_scores
ADD COLUMN IF NOT EXISTS model_variant TEXT;

ALTER TABLE gold.anomaly_detection_scores
ADD COLUMN IF NOT EXISTS stage_gate TEXT;

ALTER TABLE gold.anomaly_detection_scores
ADD COLUMN IF NOT EXISTS operating_mode TEXT;

ALTER TABLE gold.anomaly_detection_scores
ADD COLUMN IF NOT EXISTS split_name TEXT;

ALTER TABLE gold.anomaly_detection_scores
ADD COLUMN IF NOT EXISTS source_dataframe_name TEXT;

CREATE INDEX IF NOT EXISTS idx_gold_scores_variant_gate
ON gold.anomaly_detection_scores (
    dataset_id,
    run_id,
    model_name,
    model_variant,
    stage_gate,
    operating_mode
);

-- Row-per-model comparison metrics from Gold 04.
CREATE TABLE IF NOT EXISTS gold.model_comparison_metrics (
    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_label TEXT,
    variant_family TEXT,
    stage3_mode TEXT,
    alert_count_test_rows BIGINT,
    precision DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    f1 DOUBLE PRECISION,
    stage_truth_hash TEXT,
    parent_gold_truth_hash TEXT,
    meta_truth_hash TEXT,
    meta_parent_truth_hash TEXT,
    meta_pipeline_mode TEXT,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (dataset_id, run_id, model_id)
);

CREATE INDEX IF NOT EXISTS idx_gold_model_comparison_metrics_dataset_run
ON gold.model_comparison_metrics (dataset_id, run_id);

-- Gold 05 output manifest. This records which dashboard/report-ready outputs
-- were produced and where their file artifacts live.
CREATE TABLE IF NOT EXISTS gold.anomaly_detection_output_manifest (
    dataset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    output_name TEXT NOT NULL,
    output_type TEXT,
    row_count BIGINT,
    column_count INTEGER,
    artifact_path TEXT,
    selected_run_key TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (dataset_id, run_id, output_name)
);

CREATE INDEX IF NOT EXISTS idx_gold_anomaly_output_manifest_dataset_run
ON gold.anomaly_detection_output_manifest (dataset_id, run_id);
