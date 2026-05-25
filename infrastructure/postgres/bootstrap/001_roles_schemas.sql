-- =============================================================================
-- 001_roles_schemas.sql
-- Purpose: Create/update runtime roles and project schemas.
-- =============================================================================

\echo '[db-bootstrap] 001 roles and schemas'

SELECT format(
    'CREATE ROLE %I LOGIN PASSWORD %L',
    :'producer_role',
    :'producer_password'
)
WHERE NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_roles
    WHERE rolname = :'producer_role'
)
\gexec

ALTER ROLE :"producer_role" WITH LOGIN PASSWORD :'producer_password';

SELECT format(
    'CREATE ROLE %I LOGIN PASSWORD %L',
    :'ingest_role',
    :'ingest_password'
)
WHERE NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_roles
    WHERE rolname = :'ingest_role'
)
\gexec

ALTER ROLE :"ingest_role" WITH LOGIN PASSWORD :'ingest_password';

CREATE SCHEMA IF NOT EXISTS :"capstone_schema";
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS metadata;

GRANT USAGE ON SCHEMA :"capstone_schema" TO :"producer_role", :"ingest_role";
GRANT USAGE ON SCHEMA bronze TO :"producer_role", :"ingest_role";
GRANT USAGE ON SCHEMA silver TO :"producer_role", :"ingest_role";
GRANT USAGE ON SCHEMA gold TO :"producer_role", :"ingest_role";
GRANT USAGE ON SCHEMA metadata TO :"producer_role", :"ingest_role";
