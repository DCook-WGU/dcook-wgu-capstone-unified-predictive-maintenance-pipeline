-- =============================================================================
-- 005_grants_ownership.sql
-- Purpose: Apply temporary runtime ownership and grants, then print verification.
-- =============================================================================

\echo '[db-bootstrap] 005 grants and temporary ownership'

ALTER TABLE :"capstone_schema".:"producer_queue_table" OWNER TO :"producer_role";
ALTER TABLE :"capstone_schema".:"producer_control_table" OWNER TO :"producer_role";
ALTER TABLE :"capstone_schema".:"consumer_target_table" OWNER TO :"ingest_role";

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

GRANT USAGE ON SCHEMA :"capstone_schema" TO :"producer_role", :"ingest_role";
GRANT USAGE ON SCHEMA bronze TO :"producer_role", :"ingest_role";
GRANT USAGE ON SCHEMA silver TO :"producer_role", :"ingest_role";
GRANT USAGE ON SCHEMA gold TO :"producer_role", :"ingest_role";
GRANT USAGE ON SCHEMA metadata TO :"producer_role", :"ingest_role";

GRANT SELECT, INSERT, UPDATE, DELETE
ON :"capstone_schema".:"producer_queue_table"
TO :"producer_role";

GRANT SELECT, INSERT, UPDATE
ON :"capstone_schema".:"producer_control_table"
TO :"producer_role";

GRANT SELECT, INSERT, UPDATE
ON :"capstone_schema".:"consumer_target_table"
TO :"ingest_role";

GRANT SELECT
ON :"capstone_schema"."simulation_timing_config"
TO :"producer_role", :"ingest_role";

GRANT USAGE, SELECT, UPDATE
ON ALL SEQUENCES IN SCHEMA :"capstone_schema"
TO :"producer_role", :"ingest_role";

SELECT 'schemas ready' AS bootstrap_step;
SELECT 'roles ready' AS bootstrap_step;
SELECT 'core metadata tables ready' AS bootstrap_step;
SELECT 'streaming runtime tables ready' AS bootstrap_step;
SELECT 'medallion shell tables ready' AS bootstrap_step;

SELECT
    table_schema,
    table_name
FROM information_schema.tables
WHERE table_schema IN (:'capstone_schema', 'bronze', 'silver', 'gold', 'metadata')
ORDER BY table_schema, table_name;


-- Temporary runtime DDL permissions.
-- Needed while producer/consumer utilities still perform limited CREATE INDEX
-- or ALTER TABLE IF NOT EXISTS checks at runtime.
-- Final target state: move all DDL into bootstrap/migrations and remove
-- CREATE privileges from runtime roles.
GRANT USAGE, CREATE ON SCHEMA capstone TO kafka_producer;
GRANT USAGE, CREATE ON SCHEMA capstone TO kafka_ingest;


GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA silver TO kafka_ingest;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA silver TO kafka_producer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA silver TO dcook_admin;


GRANT USAGE ON SCHEMA silver TO kafka_ingest;
GRANT USAGE ON SCHEMA silver TO kafka_producer;

GRANT SELECT, INSERT, UPDATE, DELETE
ON ALL TABLES IN SCHEMA silver
TO kafka_ingest;

GRANT SELECT, INSERT, UPDATE, DELETE
ON ALL TABLES IN SCHEMA silver
TO kafka_producer;

ALTER DEFAULT PRIVILEGES IN SCHEMA silver
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO kafka_ingest;

ALTER DEFAULT PRIVILEGES IN SCHEMA silver
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO kafka_producer;