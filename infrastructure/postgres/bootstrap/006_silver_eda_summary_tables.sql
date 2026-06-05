\echo '[db-bootstrap] 006 silver eda summary tables'

-- =============================================================================
-- 006_silver_eda_summary_tables.sql
-- Purpose:
--   Create SQL-facing summary tables for Silver 02b exploratory data analysis.
--
-- Design:
--   These tables store durable EDA summaries, not every temporary notebook
--   dataframe. Large charts and reports remain file artifacts and are referenced
--   through silver.eda_artifact_index and capstone.pipeline_artifacts.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS silver;

-- -----------------------------------------------------------------------------
-- Dataset/profile-level EDA summary.
-- One row per dataset/run/notebook/dataframe/profile scope.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.eda_dataset_profile (
    dataset_id              TEXT NOT NULL,
    run_id                  TEXT NOT NULL,
    notebook_name           TEXT NOT NULL,
    dataframe_name          TEXT NOT NULL,
    profile_scope           TEXT NOT NULL DEFAULT 'full_dataset',

    row_count               BIGINT,
    column_count            INTEGER,
    duplicate_row_count     BIGINT,
    memory_usage_bytes      BIGINT,

    numeric_column_count    INTEGER,
    categorical_column_count INTEGER,
    datetime_column_count   INTEGER,
    boolean_column_count    INTEGER,

    profile_notes           TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (
        dataset_id,
        run_id,
        notebook_name,
        dataframe_name,
        profile_scope
    )
);

-- -----------------------------------------------------------------------------
-- Numeric feature statistics.
-- One row per dataset/run/dataframe/feature.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.eda_feature_statistics (
    dataset_id              TEXT NOT NULL,
    run_id                  TEXT NOT NULL,
    notebook_name           TEXT NOT NULL,
    dataframe_name          TEXT NOT NULL,
    feature_name            TEXT NOT NULL,

    non_null_count          BIGINT,
    null_count              BIGINT,
    null_pct                DOUBLE PRECISION,

    mean_value              DOUBLE PRECISION,
    std_value               DOUBLE PRECISION,
    min_value               DOUBLE PRECISION,
    p01_value               DOUBLE PRECISION,
    p05_value               DOUBLE PRECISION,
    p25_value               DOUBLE PRECISION,
    p50_value               DOUBLE PRECISION,
    p75_value               DOUBLE PRECISION,
    p95_value               DOUBLE PRECISION,
    p99_value               DOUBLE PRECISION,
    max_value               DOUBLE PRECISION,

    skew_value              DOUBLE PRECISION,
    kurtosis_value          DOUBLE PRECISION,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (
        dataset_id,
        run_id,
        notebook_name,
        dataframe_name,
        feature_name
    )
);

-- -----------------------------------------------------------------------------
-- Missingness summary by feature.
-- Useful for data quality reporting and Silver-stage validation.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.eda_missingness_summary (
    dataset_id              TEXT NOT NULL,
    run_id                  TEXT NOT NULL,
    notebook_name           TEXT NOT NULL,
    dataframe_name          TEXT NOT NULL,
    feature_name            TEXT NOT NULL,

    row_count               BIGINT,
    null_count              BIGINT,
    non_null_count          BIGINT,
    null_pct                DOUBLE PRECISION,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (
        dataset_id,
        run_id,
        notebook_name,
        dataframe_name,
        feature_name
    )
);

-- -----------------------------------------------------------------------------
-- Correlation pairs.
-- Store the long-form correlation matrix instead of a wide matrix.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.eda_correlation_pairs (
    dataset_id              TEXT NOT NULL,
    run_id                  TEXT NOT NULL,
    notebook_name           TEXT NOT NULL,
    dataframe_name          TEXT NOT NULL,
    correlation_method      TEXT NOT NULL DEFAULT 'pearson',

    feature_a               TEXT NOT NULL,
    feature_b               TEXT NOT NULL,
    correlation_value       DOUBLE PRECISION,

    abs_correlation_value   DOUBLE PRECISION,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (
        dataset_id,
        run_id,
        notebook_name,
        dataframe_name,
        correlation_method,
        feature_a,
        feature_b
    )
);

-- -----------------------------------------------------------------------------
-- Outlier summary by feature.
-- Supports IQR, z-score, robust z-score, or other EDA outlier methods.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.eda_outlier_summary (
    dataset_id              TEXT NOT NULL,
    run_id                  TEXT NOT NULL,
    notebook_name           TEXT NOT NULL,
    dataframe_name          TEXT NOT NULL,
    feature_name            TEXT NOT NULL,
    outlier_method          TEXT NOT NULL,

    lower_threshold         DOUBLE PRECISION,
    upper_threshold         DOUBLE PRECISION,
    outlier_count           BIGINT,
    row_count               BIGINT,
    outlier_pct             DOUBLE PRECISION,

    method_notes            TEXT,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (
        dataset_id,
        run_id,
        notebook_name,
        dataframe_name,
        feature_name,
        outlier_method
    )
);

-- -----------------------------------------------------------------------------
-- Categorical/status distribution.
-- Useful for machine_status, status buckets, labels, and other categorical fields.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.eda_categorical_distribution (
    dataset_id              TEXT NOT NULL,
    run_id                  TEXT NOT NULL,
    notebook_name           TEXT NOT NULL,
    dataframe_name          TEXT NOT NULL,
    feature_name            TEXT NOT NULL,

    category_value          TEXT NOT NULL,
    category_count          BIGINT,
    category_pct            DOUBLE PRECISION,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (
        dataset_id,
        run_id,
        notebook_name,
        dataframe_name,
        feature_name,
        category_value
    )
);

-- -----------------------------------------------------------------------------
-- EDA artifact index.
-- Stores paths and metadata for charts, saved tables, and EDA reports.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS silver.eda_artifact_index (
    dataset_id              TEXT NOT NULL,
    run_id                  TEXT NOT NULL,
    notebook_name           TEXT NOT NULL,
    artifact_name           TEXT NOT NULL,

    artifact_type           TEXT NOT NULL,
    artifact_path           TEXT NOT NULL,
    artifact_format         TEXT,
    artifact_description    TEXT,
    wandb_artifact_name     TEXT,
    wandb_artifact_type     TEXT,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (
        dataset_id,
        run_id,
        notebook_name,
        artifact_name
    )
);

-- -----------------------------------------------------------------------------
-- Helpful indexes for inspection queries.
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_eda_feature_statistics_dataset_run
    ON silver.eda_feature_statistics (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS idx_eda_missingness_dataset_run
    ON silver.eda_missingness_summary (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS idx_eda_correlation_dataset_run
    ON silver.eda_correlation_pairs (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS idx_eda_outlier_dataset_run
    ON silver.eda_outlier_summary (dataset_id, run_id);

CREATE INDEX IF NOT EXISTS idx_eda_artifact_index_dataset_run
    ON silver.eda_artifact_index (dataset_id, run_id);