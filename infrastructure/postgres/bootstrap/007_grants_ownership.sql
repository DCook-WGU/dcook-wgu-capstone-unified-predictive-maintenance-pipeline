-- =============================================================================
-- 007_grants_ownership.sql
-- Purpose:
--   Apply final runtime ownership, schema grants, table grants, sequence grants,
--   default privileges, and bootstrap verification.
--
-- Important:
--   This file must run after all schemas and tables have been created.
--
-- Expected order:
--   001_roles_schemas.sql
--   002_core_metadata_tables.sql
--   003_streaming_runtime_tables.sql
--   004_medallion_schema_shells.sql
--   005_synthetic_notebook_stage_tables.sql
--   006_silver_eda_summary_tables.sql
--   007_grants_ownership.sql
-- =============================================================================

\echo '[db-bootstrap] 007 grants and ownership'

-- -----------------------------------------------------------------------------
-- Runtime table ownership
--
-- These tables are created in 003_streaming_runtime_tables.sql.
-- ALTER TABLE IF EXISTS prevents this file from crashing if a table name changes,
-- but the verification query at the end should still be checked.
-- -----------------------------------------------------------------------------

ALTER TABLE IF EXISTS :"capstone_schema".:"producer_queue_table"
    OWNER TO :"producer_role";

ALTER TABLE IF EXISTS :"capstone_schema".:"producer_control_table"
    OWNER TO :"producer_role";

ALTER TABLE IF EXISTS :"capstone_schema".:"consumer_target_table"
    OWNER TO :"ingest_role";


-- -----------------------------------------------------------------------------
-- Runtime sequence ownership
-- -----------------------------------------------------------------------------

SELECT format(
    'ALTER SEQUENCE %s OWNER TO %I',
    pg_get_serial_sequence(format('%I.%I', :'capstone_schema', :'producer_control_table'), 'control_id'),
    :'producer_role'
)
WHERE pg_get_serial_sequence(format('%I.%I', :'capstone_schema', :'producer_control_table'), 'control_id') IS NOT NULL
\gexec

SELECT format(
    'ALTER SEQUENCE %s OWNER TO %I',
    pg_get_serial_sequence(format('%I.%I', :'capstone_schema', :'consumer_target_table'), 'consumed_id'),
    :'ingest_role'
)
WHERE pg_get_serial_sequence(format('%I.%I', :'capstone_schema', :'consumer_target_table'), 'consumed_id') IS NOT NULL
\gexec


-- -----------------------------------------------------------------------------
-- Schema usage grants
-- -----------------------------------------------------------------------------

GRANT USAGE ON SCHEMA :"capstone_schema" TO :"producer_role", :"ingest_role";
GRANT USAGE ON SCHEMA bronze TO :"producer_role", :"ingest_role";
GRANT USAGE ON SCHEMA silver TO :"producer_role", :"ingest_role";
GRANT USAGE ON SCHEMA gold TO :"producer_role", :"ingest_role";
GRANT USAGE ON SCHEMA metadata TO :"producer_role", :"ingest_role";


-- -----------------------------------------------------------------------------
-- Temporary runtime DDL permissions
--
-- Needed while producer/consumer utilities still perform limited runtime DDL
-- checks such as CREATE INDEX IF NOT EXISTS or ALTER TABLE ADD COLUMN IF NOT
-- EXISTS. Final target state is to move all DDL into bootstrap/migrations and
-- remove CREATE privileges from runtime roles.
-- -----------------------------------------------------------------------------

GRANT USAGE, CREATE ON SCHEMA :"capstone_schema" TO :"producer_role", :"ingest_role";


-- -----------------------------------------------------------------------------
-- Broad table grants for runtime roles
--
-- This is intentionally broad for the capstone runtime environment. It keeps
-- notebook, producer, consumer, and SQL inspection flows from failing because of
-- missing permissions while the project is still notebook-driven.
-- -----------------------------------------------------------------------------

GRANT SELECT, INSERT, UPDATE, DELETE
ON ALL TABLES IN SCHEMA :"capstone_schema"
TO :"producer_role", :"ingest_role";

GRANT SELECT, INSERT, UPDATE, DELETE
ON ALL TABLES IN SCHEMA bronze
TO :"producer_role", :"ingest_role";

GRANT SELECT, INSERT, UPDATE, DELETE
ON ALL TABLES IN SCHEMA silver
TO :"producer_role", :"ingest_role";

GRANT SELECT, INSERT, UPDATE, DELETE
ON ALL TABLES IN SCHEMA gold
TO :"producer_role", :"ingest_role";

GRANT SELECT, INSERT, UPDATE, DELETE
ON ALL TABLES IN SCHEMA metadata
TO :"producer_role", :"ingest_role";


-- -----------------------------------------------------------------------------
-- Sequence grants
-- -----------------------------------------------------------------------------

GRANT USAGE, SELECT, UPDATE
ON ALL SEQUENCES IN SCHEMA :"capstone_schema"
TO :"producer_role", :"ingest_role";

GRANT USAGE, SELECT, UPDATE
ON ALL SEQUENCES IN SCHEMA bronze
TO :"producer_role", :"ingest_role";

GRANT USAGE, SELECT, UPDATE
ON ALL SEQUENCES IN SCHEMA silver
TO :"producer_role", :"ingest_role";

GRANT USAGE, SELECT, UPDATE
ON ALL SEQUENCES IN SCHEMA gold
TO :"producer_role", :"ingest_role";

GRANT USAGE, SELECT, UPDATE
ON ALL SEQUENCES IN SCHEMA metadata
TO :"producer_role", :"ingest_role";


-- -----------------------------------------------------------------------------
-- Default privileges for future tables
-- -----------------------------------------------------------------------------

ALTER DEFAULT PRIVILEGES IN SCHEMA :"capstone_schema"
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO :"producer_role", :"ingest_role";

ALTER DEFAULT PRIVILEGES IN SCHEMA bronze
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO :"producer_role", :"ingest_role";

ALTER DEFAULT PRIVILEGES IN SCHEMA silver
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO :"producer_role", :"ingest_role";

ALTER DEFAULT PRIVILEGES IN SCHEMA gold
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO :"producer_role", :"ingest_role";

ALTER DEFAULT PRIVILEGES IN SCHEMA metadata
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO :"producer_role", :"ingest_role";


-- -----------------------------------------------------------------------------
-- Verification output
-- -----------------------------------------------------------------------------

SELECT 'schemas ready' AS bootstrap_step;
SELECT 'roles ready' AS bootstrap_step;
SELECT 'core metadata tables ready' AS bootstrap_step;
SELECT 'streaming runtime tables ready' AS bootstrap_step;
SELECT 'medallion shell tables ready' AS bootstrap_step;
SELECT 'synthetic notebook stage tables ready' AS bootstrap_step;
SELECT 'silver eda summary tables ready' AS bootstrap_step;
SELECT 'grants and ownership ready' AS bootstrap_step;

SELECT
    table_schema,
    table_name
FROM information_schema.tables
WHERE table_schema IN (:'capstone_schema', 'bronze', 'silver', 'gold', 'metadata')
ORDER BY table_schema, table_name;

SELECT
    to_regclass(format('%I.%I', :'capstone_schema', :'producer_queue_table')) AS producer_queue_table,
    to_regclass(format('%I.%I', :'capstone_schema', :'producer_control_table')) AS producer_control_table,
    to_regclass(format('%I.%I', :'capstone_schema', :'consumer_target_table')) AS consumer_target_table;