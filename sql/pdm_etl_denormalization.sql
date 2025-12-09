

---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

-- Create a schema
CREATE SCHEMA IF NOT EXISTS pdm;

---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

-- Create Tables

-- =========================================
-- Telemetry (core time-series)
-- =========================================
CREATE TABLE IF NOT EXISTS pdm.pdm_telemetry ( 
    datetime   TIMESTAMP        NOT NULL,
    machineid  INT              NOT NULL,
    volt       DOUBLE PRECISION NOT NULL,
    rotate     DOUBLE PRECISION NOT NULL,
    pressure   DOUBLE PRECISION NOT NULL,
    vibration  DOUBLE PRECISION NOT NULL,

    -- One reading per machine per timestamp
    CONSTRAINT pk_pdm_telemetry PRIMARY KEY (datetime, machineid)
);

-- =========================================
-- Machines (static dimension)
-- =========================================
CREATE TABLE IF NOT EXISTS pdm.pdm_machines ( 
    machineid  INT   NOT NULL,
    model      TEXT  NOT NULL,
    age        INT   NOT NULL,
    
    CONSTRAINT pk_pdm_machines PRIMARY KEY (machineid),
    CONSTRAINT ck_pdm_machines_age_nonneg CHECK (age >= 0)
);

-- =========================================
-- Errors (events, can be multiple per ts)
-- =========================================
CREATE TABLE IF NOT EXISTS pdm.pdm_errors ( 
    datetime   TIMESTAMP NOT NULL, 
    machineid  INT       NOT NULL, 
    errorid    TEXT      NOT NULL,

    CONSTRAINT pk_pdm_errors PRIMARY KEY (datetime, machineid, errorid),
    CONSTRAINT fk_pdm_errors_machine
        FOREIGN KEY (machineid)
        REFERENCES pdm.pdm_machines (machineid)
        ON DELETE CASCADE
);

-- =========================================
-- Failures (events, can be multiple per ts)
-- =========================================
CREATE TABLE IF NOT EXISTS pdm.pdm_failures ( 
    datetime   TIMESTAMP NOT NULL, 
    machineid  INT       NOT NULL, 
    failure    TEXT      NOT NULL,

    CONSTRAINT pk_pdm_failures PRIMARY KEY (datetime, machineid, failure),
    CONSTRAINT fk_pdm_failures_machine
        FOREIGN KEY (machineid)
        REFERENCES pdm.pdm_machines (machineid)
        ON DELETE CASCADE
);

-- =========================================
-- Maintenance (events, can be multiple per ts)
-- =========================================
CREATE TABLE IF NOT EXISTS pdm.pdm_maint ( 
    datetime   TIMESTAMP NOT NULL, 
    machineid  INT       NOT NULL, 
    comp       TEXT      NOT NULL,

    CONSTRAINT pk_pdm_maint PRIMARY KEY (datetime, machineid, comp),
    CONSTRAINT fk_pdm_maint_machine
        FOREIGN KEY (machineid)
        REFERENCES pdm.pdm_machines (machineid)
        ON DELETE CASCADE
);

---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

-- Import Datasets to new tables

-- Telemetry 
COPY pdm.pdm_telemetry 
FROM '/data/raw/Microsoft_Azure_Predictive_Maintenance/PdM_telemetry.csv' 
DELIMITER ',' CSV HEADER; 

-- Machines 
COPY pdm.pdm_machines 
FROM '/data/raw/Microsoft_Azure_Predictive_Maintenance/PdM_machines.csv' 
DELIMITER ',' CSV HEADER; 

-- Errors
COPY pdm.pdm_errors 
FROM '/data/raw/Microsoft_Azure_Predictive_Maintenance/PdM_errors.csv' 
DELIMITER ',' CSV HEADER; 

-- Failures 
COPY pdm.pdm_failures 
FROM '/data/raw/Microsoft_Azure_Predictive_Maintenance/PdM_failures.csv' 
DELIMITER ',' CSV HEADER; 

-- Maintenance 
COPY pdm.pdm_maint 
FROM '/data/raw/Microsoft_Azure_Predictive_Maintenance/PdM_maint.csv' 
DELIMITER ',' CSV HEADER;



---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

-- Using Materialized Views to create Aggregate tables
-- Materialized Views are required to do a "text" aggregation for some of the
-- data's features, such as: failuers, errors, maintenance. Multiple of these 
-- events can occur at the same date and time and on the same machine. 

-- Aggregate errors per (datetime, machine) 
CREATE OR REPLACE MATERIALIZED VIEW pdm.pdm_errors_agg AS 
    SELECT 
        datetime, 
        machineid, 
        STRING_AGG(errorid, ',' ORDER BY errorid) AS error_ids 
    FROM pdm.pdm_errors 
    GROUP BY datetime, machineid;

-- Aggregate maintenance per (datetime, machine) 
CREATE OR REPLACE MATERIALIZED VIEW pdm.pdm_maint_agg AS 
    SELECT 
        datetime, 
        machineid, 
        STRING_AGG(comp, ',' ORDER BY comp) AS maint_comps 
    FROM pdm.pdm_maint 
    GROUP BY datetime, machineid;

-- Aggregate failures per (datetime, machine)
CREATE OR REPLACE MATERIALIZED VIEW pdm.pdm_failures_agg AS 
    SELECT 
        datetime, 
        machineid, 
        STRING_AGG(failure, ',' ORDER BY failure) AS failure_comps 
    FROM pdm.pdm_failures 
    GROUP BY datetime, machineid;

---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


-- Run Drop Table command first to confirm clean slate run. 
DROP TABLE IF EXISTS pdm.pdm_denormalized;


---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

-- Creating denormalized table 

CREATE TABLE pdm.pdm_denormalized AS 
    SELECT
        -- Creating the dataset name column 
        'AZURE_PDM'::text AS dataset_name, 
        
        -- Creating a time index for each machine individually with respect 
        -- time. 

        ROW_NUMBER() OVER (
            PARTITION BY t.machineid
            ORDER BY t.datetime
        ) AS time_index, 

        -- Creating the event_time column, which is just the original date
        -- time column
        t.datetime AS event_time, 
        
        -- Creating a machine_id column to match formatting
        t.machineid AS machine_id, 
        
        -- keeping the old columns just in case for now
        t.datetime,
        t.machineid, 

        -- Remainder of the columns 
        m.model, 
        m.age, 
        t.volt, 
        t.rotate, 
        t.pressure, 
        t.vibration, 

        -- Aggregated columns made with the material views 
        e.error_ids, 
        f.failure_comps, 
        mt.maint_comps, 
        
        -- Simple Anomaly Flag :: 
        -- This simple flag is being included to make filtering for normal vs
        -- abnormal events easier and to be more uniform with the other datasets. 
        -- This checks each of three aggregated lists to see if they are null.
        -- If values are found in any of the lists, anomaly_flag == TRUE
        -- If no values are found in any of the lists, anomaly_flag == FALSE
        (
            e.error_ids IS NOT NULL
            OR f.failure_comps IS NOT NULL
            OR mt.maint_comps IS NOT NULL
        ) AS anomaly_flag


    FROM pdm.pdm_telemetry t 
        LEFT JOIN pdm.pdm_machines m 
            ON t.machineid = m.machineid 
        LEFT JOIN pdm.pdm_errors_agg e 
            ON t.machineid = e.machineid AND t.datetime = e.datetime 
        LEFT JOIN pdm.pdm_failures_agg f
            ON t.machineid = f.machineid AND t.datetime = f.datetime  
        LEFT JOIN pdm.pdm_maint_agg mt 
            ON t.machineid = mt.machineid AND t.datetime = mt.datetime;

---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

-- Saving Copy of Denormlized data to bronze data directory
COPY pdm.pdm_denormalized 
TO '/data/bronze/PdM_denormalized.csv' 
DELIMITER ',' CSV HEADER;

---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

-- Dropping Raw Ingestion Tables
DROP TABLE IF EXISTS pdm.pdm_telemetry;
DROP TABLE IF EXISTS pdm.pdm_machines;
DROP TABLE IF EXISTS pdm.pdm_errors;
DROP TABLE IF EXISTS pdm.pdm_failures;
DROP TABLE IF EXISTS pdm.pdm_maint;

---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

-- Dropping Materialized Views created for Aggregation
DROP MATERIALIZED VIEW IF EXISTS pdm.pdm_errors_agg;
DROP MATERIALIZED VIEW IF EXISTS pdm.pdm_maint_agg;
DROP MATERIALIZED VIEW IF EXISTS pdm.pdm_failures_agg;

---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

-- Dropping Completed Postgres File
DROP TABLE IF EXISTS pdm.pdm_denormalized;
