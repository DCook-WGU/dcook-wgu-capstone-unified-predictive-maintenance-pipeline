CREATE SCHEMA IF NOT EXISTS capstone;

CREATE TABLE IF NOT EXISTS capstone.synthetic_pump_stream (
    -- ordering / batching
    batch_id BIGINT NOT NULL,
    row_in_batch INTEGER NOT NULL,
    global_cycle_id BIGINT,

    -- state labels
    stream_state TEXT NOT NULL,      -- normal|buildup|abnormal|recovery
    phase TEXT,                      -- optional (if you keep both)

    -- lineage (matches your meta pattern)
    meta__truth_hash TEXT,
    meta__parent_truth_hash TEXT,
    meta__pipeline_mode TEXT,

    -- optional timestamps for debugging/replay
    created_at TIMESTAMPTZ DEFAULT now(),

    -- NOTE: sensor columns are written by the generator dataframe writer.
    -- Example:
    -- sensor_01 DOUBLE PRECISION,
    -- sensor_02 DOUBLE PRECISION,
    -- ...
    PRIMARY KEY (batch_id, row_in_batch)
);

-- Helpful indexes for pulling batches and streaming in order
CREATE INDEX IF NOT EXISTS idx_synth_pump_stream_batch
    ON capstone.synthetic_pump_stream (batch_id, row_in_batch);

CREATE INDEX IF NOT EXISTS idx_synth_pump_stream_cycle
    ON capstone.synthetic_pump_stream (global_cycle_id);