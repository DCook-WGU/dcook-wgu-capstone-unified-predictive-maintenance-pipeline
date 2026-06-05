-- =============================================================================
-- 002_core_metadata_tables.sql
-- Purpose: Create core project metadata tables and metadata indexes.
-- =============================================================================

\echo '[db-bootstrap] 002 core metadata tables'

CREATE TABLE IF NOT EXISTS :"capstone_schema"."pipeline_runs" (
    run_id TEXT PRIMARY KEY,
    dataset_id TEXT,
    dataset_name TEXT,
    pipeline_stage TEXT,
    pipeline_mode TEXT,
    run_status TEXT NOT NULL DEFAULT 'started',
    started_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at_utc TIMESTAMPTZ,
    source_system TEXT,
    notes TEXT,
    runtime_facts JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS :"capstone_schema"."pipeline_artifacts" (
    artifact_id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    dataset_id TEXT,
    layer_name TEXT,
    stage_name TEXT,
    artifact_name TEXT NOT NULL,
    artifact_type TEXT,
    artifact_path TEXT,
    truth_hash TEXT,
    parent_truth_hash TEXT,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS :"capstone_schema"."truth_records" (
    truth_id BIGSERIAL PRIMARY KEY,
    dataset_id TEXT,
    layer_name TEXT,
    truth_hash TEXT NOT NULL UNIQUE,
    parent_truth_hash TEXT,
    truth_path TEXT,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT now(),
    truth_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS :"capstone_schema"."data_quality_events" (
    event_id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    dataset_id TEXT,
    layer_name TEXT,
    table_name TEXT,
    severity TEXT NOT NULL DEFAULT 'info',
    check_name TEXT NOT NULL,
    check_status TEXT NOT NULL,
    row_count BIGINT,
    details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS "idx_pipeline_artifacts_run_dataset"
ON :"capstone_schema"."pipeline_artifacts" (run_id, dataset_id);

CREATE INDEX IF NOT EXISTS "idx_truth_records_dataset_layer"
ON :"capstone_schema"."truth_records" (dataset_id, layer_name);

CREATE INDEX IF NOT EXISTS "idx_data_quality_events_run_dataset"
ON :"capstone_schema"."data_quality_events" (run_id, dataset_id);
