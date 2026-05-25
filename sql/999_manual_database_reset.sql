-- ============================================================================
-- Manual database reset for capstone project schemas
-- WARNING:
--   This drops all project schemas and all objects inside them.
--   Run only when intentionally resetting a local/dev database.
-- ============================================================================

DROP SCHEMA IF EXISTS capstone CASCADE;
DROP SCHEMA IF EXISTS bronze CASCADE;
DROP SCHEMA IF EXISTS silver CASCADE;
DROP SCHEMA IF EXISTS gold CASCADE;
DROP SCHEMA IF EXISTS metadata CASCADE;