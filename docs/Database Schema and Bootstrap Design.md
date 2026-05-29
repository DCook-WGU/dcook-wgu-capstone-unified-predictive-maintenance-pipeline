# Database Schema and Bootstrap Design

## Purpose

This project uses PostgreSQL as an operational and SQL-facing support layer for the capstone pipeline. The primary analytical workflow still runs through notebooks, utilities, and file-based artifacts, but PostgreSQL provides a stable place to store runtime state, streaming records, metadata, and selected medallion outputs.

The database layer supports four major goals:

1. Recover from a fresh Docker/Postgres reset without manually recreating schemas or tables.
2. Support Kafka-based synthetic telemetry producer and consumer workflows.
3. Preserve traceability through pipeline run records, artifact records, truth hashes, and data quality events.
4. Provide SQL-facing inspection tables for Bronze, Silver, and Gold outputs.

The database bootstrap does not replace the notebook pipeline. It provides infrastructure and selected durable tables that support the notebook-driven medallion architecture.

---

## Project Database Architecture

The project follows a medallion architecture:

```text
Bronze → Silver → Gold
```

The database mirrors that design using separate schemas:

```text
capstone  → operational/runtime metadata and streaming support
bronze    → Bronze-level SQL-facing observation records
silver    → Silver-level SQL-facing feature/profile records
gold      → Gold-level SQL-facing model/scoring/comparison records
metadata  → reserved for future metadata expansion
```

The notebook pipeline remains the main source of transformation logic and artifact generation. The SQL layer supplements it with durable operational tables and selected analytical outputs.

---

## Capstone Pipeline Context

The main analytical problem is industrial pump sensor anomaly detection.

The modeling comparison is:

```text
Baseline:
  Single Isolation Forest model

Cascade:
  Stage 1: broad/high-sensitivity Isolation Forest
  Stage 2: narrow/reduced-feature Isolation Forest
  Stage 3: rule/profile/historical confirmation
```

The database layer supports this comparison by recording:

- pipeline metadata
- data quality events
- file/artifact locations
- selected Bronze/Silver/Gold outputs
- baseline anomaly scores
- cascade anomaly scores
- baseline-vs-cascade comparison summaries

---

## Bootstrap Location

The current bootstrap entry point is:

```text
infrastructure/postgres/bootstrap/001_capstone_database_bootstrap.sh
```

This script is run by the Docker Compose `db_bootstrap` service after PostgreSQL becomes healthy.

The bootstrap is designed to be rerunnable. It uses `CREATE ... IF NOT EXISTS`, `INSERT ... ON CONFLICT`, and role update logic so that a fresh database can be initialized consistently.

---

## Future Bootstrap Split

The current one-file bootstrap is intentionally kept simple while the end-to-end pipeline is still being stabilized.

The planned split structure is:

```text
infrastructure/postgres/bootstrap/
├── run_bootstrap.sh
├── 001_roles_schemas.sql
├── 002_core_metadata_tables.sql
├── 003_streaming_runtime_tables.sql
├── 004_medallion_schema_shells.sql
└── 005_grants_ownership.sql
```

The planned responsibility of each file is:

```text
run_bootstrap.sh
  Reads environment variables, validates required values, and runs SQL files in order.

001_roles_schemas.sql
  Creates runtime database roles and project schemas.

002_core_metadata_tables.sql
  Creates pipeline metadata, artifact, truth, and data quality tables.

003_streaming_runtime_tables.sql
  Creates Kafka producer/consumer runtime tables.

004_medallion_schema_shells.sql
  Creates lightweight Bronze, Silver, and Gold SQL-facing tables.

005_grants_ownership.sql
  Applies runtime ownership and grants.
```

The split should preserve the current one-file behavior.

---

## Runtime Environment Variables

The bootstrap and database utilities use environment variables from `.env` and Docker Compose.

Important variables include:

```env
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=dcook_capstone_postgres_db
POSTGRES_USER=dcook_admin
POSTGRES_PASSWORD=<admin_password>

DB_HOST=postgres
DB_PORT=5432
DB_NAME=dcook_capstone_postgres_db
DB_DATABASE=dcook_capstone_postgres_db
DB_USER=dcook_admin
DB_PASSWORD=<admin_password>

CAPSTONE_SCHEMA=capstone
SYNTHETIC_DATASET_ID=pump_synthetic_v1
SYNTHETIC_RUN_ID=synthetic_run_001

KAFKA_TOPIC=pump.telemetry.synthetic
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_CONSUMER_GROUP_ID=synthetic-telemetry-consumer-group

PRODUCER_QUEUE_TABLE=synthetic_sensor_messages_send_queue
PRODUCER_CONTROL_TABLE=simulation_state_control
CONSUMER_TARGET_TABLE=synthetic_sensor_messages_consumed_stage

PRODUCER_BATCH_SIZE=5200
PRODUCER_POLL_SECONDS=0.0
MAX_SEND_ATTEMPTS=3

KAFKA_PRODUCER_DB_USER=kafka_producer
KAFKA_PRODUCER_DB_PASSWORD=<producer_password>

KAFKA_INGEST_DB_USER=kafka_ingest
KAFKA_INGEST_DB_PASSWORD=<consumer_password>

ENV_NAME=capstone
PYTHONPATH=/workspace:/workspace/app
```

The same dataset and run identifiers are used across notebooks, streaming utilities, and database tables to keep outputs aligned.

---

## Docker Services Supported by the Database Bootstrap

The bootstrap supports the following Docker Compose services:

```text
postgres
db_bootstrap
pgadmin
kafka
kafka_topic_init
app
producer_water_pump_synthetic
db_consumer
```

The expected service dependency pattern is:

```text
postgres:
  must become healthy before db_bootstrap runs

db_bootstrap:
  initializes schemas, tables, roles, control rows, and grants

kafka:
  must become healthy before kafka_topic_init runs

kafka_topic_init:
  creates the pump.telemetry.synthetic Kafka topic

producer_water_pump_synthetic:
  depends on healthy Kafka, initialized topic, healthy Postgres, and completed database bootstrap

db_consumer:
  depends on healthy Kafka, initialized topic, healthy Postgres, and completed database bootstrap
```

---

## Docker Volumes and Network

The project uses named Docker volumes and a named network:

```yaml
volumes:
  postgres_data:
    name: dcook_capstone_postgres_data
  pgadmin_data:
    name: dcook_capstone_pgadmin_data
  kafka_data:
    name: dcook_capstone_kafka_data

networks:
  capstone_net:
    name: dcook_capstone_net
    driver: bridge
```

These named resources make it easier to intentionally reset or inspect project state.

---

## Schemas

### `capstone`

The `capstone` schema stores operational project tables, runtime control state, Kafka staging tables, and project metadata.

It is the central schema for:

- pipeline run records
- artifact records
- truth records
- data quality events
- producer control state
- producer send queue
- consumer landed messages
- simulation timing configuration

### `bronze`

The `bronze` schema stores SQL-facing Bronze observation records.

Bronze data represents raw or minimally standardized observations with lineage fields and raw payload preservation.

### `silver`

The `silver` schema stores SQL-facing cleaned and feature-ready records.

Silver data represents records that have been prepared for exploratory analysis, profiling, data quality review, and downstream modeling.

### `gold`

The `gold` schema stores SQL-facing model-ready features, anomaly scores, alert outputs, and model comparison results.

Gold data supports the anomaly detection comparison between:

- single Isolation Forest baseline
- three-stage cascade model

### `metadata`

The `metadata` schema is reserved for future metadata expansion. The current project primarily uses `capstone` for active metadata tables.

---

# Core Metadata Tables

## `capstone.pipeline_runs`

Tracks notebook or pipeline-stage execution records.

Important fields include:

```text
run_id
dataset_id
dataset_name
pipeline_stage
pipeline_mode
run_status
started_at_utc
completed_at_utc
source_system
notes
runtime_facts
```

Because the same dataset/run can pass through multiple notebook stages, utility writers may create stage-specific pipeline run IDs internally while preserving the original run ID inside `runtime_facts`.

Example pipeline stages:

```text
bronze_preprocessing
silver_preprocessing
silver_eda
gold_preprocessing
gold_baseline_modeling
gold_cascade_modeling
gold_model_comparison
gold_anomaly_detection_summary
```

## `capstone.pipeline_artifacts`

Tracks generated files and artifacts.

Examples include:

```text
Parquet outputs
CSV summaries
JSON summaries
truth files
ledger files
plots
model artifacts
W&B-related artifacts
```

Important fields include:

```text
artifact_id
run_id
dataset_id
layer_name
stage_name
artifact_name
artifact_type
artifact_path
truth_hash
parent_truth_hash
created_at_utc
metadata_json
```

## `capstone.truth_records`

Tracks truth hash records and associated truth JSON content.

This supports the project’s traceability design where dataframe-level and layer-level facts are recorded outside of row-level data.

Important fields include:

```text
truth_id
dataset_id
layer_name
truth_hash
parent_truth_hash
truth_path
created_at_utc
truth_json
```

## `capstone.data_quality_events`

Tracks data quality checks and runtime validations.

Examples include:

```text
SQL write completed
row count validation
missingness checks
schema checks
EDA profile logging
model output validation
```

Important fields include:

```text
event_id
run_id
dataset_id
layer_name
table_name
severity
check_name
check_status
row_count
details_json
created_at_utc
```

---

# Streaming Runtime Tables

## `capstone.simulation_state_control`

Controls whether a producer should run for a dataset/run and which Kafka topic it should write to.

Important fields include:

```text
control_id
dataset_id
run_id
is_enabled
producer_topic
producer_batch_size
producer_poll_seconds
max_send_attempts
updated_at
created_at
```

The bootstrap seeds a row for:

```text
dataset_id = pump_synthetic_v1
run_id = synthetic_run_001
topic = pump.telemetry.synthetic
```

This table allows the producer service to resolve runtime settings without hardcoding all values in application code.

## `capstone.simulation_timing_config`

Stores simulation timing information for synthetic telemetry.

Important fields include:

```text
config_id
dataset_id
run_id
simulation_start_datetime
sampling_interval_seconds
is_active
created_at
```

This table supports consistent timestamp interpretation across generated and streamed records.

## `capstone.synthetic_sensor_messages_send_queue`

Stores sensor-level messages that are ready to be sent to Kafka.

This table is populated by the synthetic send queue notebook/stage and consumed by the producer service.

Important fields include:

```text
dataset_id
run_id
asset_id
message_key
generated_row_id
observation_index
observation_timestamp
message_sequence_index
batch_id
row_in_batch
global_cycle_id
stream_state
phase
created_at
meta_episode_id
meta_primary_fault_type
meta_magnitude
sensor_name
sensor_index
sensor_value
is_telemetry_event
telemetry_event_type
producer_send_attempt
queue_status
queued_at
claim_token
claimed_at
producer_topic
producer_worker_id
producer_sent_at
producer_ack_at
producer_delivery_status
producer_delivery_error
```

The producer claims rows by:

```text
dataset_id
run_id
queue_status
observation_index
message_sequence_index
sensor_index
```

This prevents one producer run from accidentally claiming rows from another dataset/run.

Expected queue states:

```text
pending
claimed
sent
failed
```

## `capstone.synthetic_sensor_messages_consumed_stage`

Stores landed Kafka messages consumed from the telemetry topic.

This table preserves Kafka metadata and raw payloads.

Important fields include:

```text
consumed_id
kafka_topic
kafka_partition
kafka_offset
kafka_key
kafka_timestamp
message_key
dataset_id
run_id
asset_id
generated_row_id
observation_index
observation_timestamp
message_sequence_index
batch_id
row_in_batch
global_cycle_id
stream_state
phase
created_at
meta_episode_id
meta_primary_fault_type
meta_magnitude
sensor_name
sensor_index
sensor_value
is_telemetry_event
telemetry_event_type
producer_topic
producer_worker_id
producer_sent_at
producer_ack_at
producer_delivery_status
producer_delivery_error
consumer_group_id
consumer_worker_id
consumed_at
raw_payload
```

This table is the input to downstream row rebuild and final alignment stages.

---

# Medallion SQL Shell Tables

The bootstrap creates lightweight SQL-facing medallion tables. These tables are not intended to replace the richer notebook artifacts. They provide durable summaries and selected records for inspection, traceability, and reporting.

## `bronze.sensor_observations`

Stores Bronze-level observation records.

The full row is preserved in `raw_payload` as JSONB while key lineage fields are promoted to SQL columns.

Important fields include:

```text
bronze_id
dataset_id
run_id
asset_id
event_time
event_step
time_index
source_table
source_row_id
raw_payload
meta_truth_hash
meta_parent_truth_hash
meta_ingested_at_utc
```

## `silver.sensor_observation_features`

Stores Silver-level feature records.

Feature values are stored in `features_json`, while data quality and metadata fields can be stored in `quality_json`.

Important fields include:

```text
silver_id
dataset_id
run_id
asset_id
event_time
event_step
time_index
feature_set_id
features_json
quality_json
meta_truth_hash
meta_parent_truth_hash
meta_processed_at_utc
```

## `gold.preprocessed_features`

Stores Gold-level preprocessed/model-ready feature records.

This table may be created by the bootstrap or by the medallion SQL writer utility if it does not already exist.

Important fields include:

```text
preprocessed_id
dataset_id
run_id
asset_id
event_time
event_step
time_index
feature_set_id
split_name
is_train
features_json
meta_truth_hash
meta_parent_truth_hash
created_at_utc
```

## `gold.anomaly_detection_scores`

Stores model score and anomaly flag records.

This table supports both baseline and cascade scoring outputs.

Important fields include:

```text
gold_id
dataset_id
run_id
asset_id
event_time
event_step
time_index
model_name
model_stage
anomaly_score
anomaly_flag
alert_severity
evidence_json
meta_truth_hash
meta_parent_truth_hash
meta_scored_at_utc
```

Expected model values include:

```text
model_name = baseline_isolation_forest
model_stage = baseline

model_name = cascade_isolation_forest_rule_confirmation
model_stage = cascade_final
```

## `gold.model_comparison_results`

Stores final baseline-vs-cascade comparison summaries.

Important fields include:

```text
comparison_id
dataset_id
run_id
baseline_model
comparison_model
alert_count_baseline
alert_count_comparison
precision_baseline
precision_comparison
recall_baseline
recall_comparison
f1_baseline
f1_comparison
comparison_json
created_at_utc
```

---

# Database Utility Path Map

## Bootstrap

```text
infrastructure/postgres/bootstrap/001_capstone_database_bootstrap.sh
```

Future split:

```text
infrastructure/postgres/bootstrap/run_bootstrap.sh
infrastructure/postgres/bootstrap/001_roles_schemas.sql
infrastructure/postgres/bootstrap/002_core_metadata_tables.sql
infrastructure/postgres/bootstrap/003_streaming_runtime_tables.sql
infrastructure/postgres/bootstrap/004_medallion_schema_shells.sql
infrastructure/postgres/bootstrap/005_grants_ownership.sql
```

## Manual Reset

```text
sql/999_manual_database_reset.sql
```

## Generic Database Helpers

```text
utils/database/postgres.py
utils/database/layer_postgres.py
utils/database/sql_notebook_helpers.py
```

## Medallion SQL Writers

```text
utils/database/medallion_sql_writers.py
```

## Streaming Producer/Consumer Database Utilities

```text
utils/synthetic/pipeline/producer_queue_manager.py
utils/synthetic/pipeline/kafka_producer_adapter.py
utils/synthetic/pipeline/kafka_consumer_adapter.py
```

## Synthetic Pipeline Utilities

```text
utils/synthetic/pipeline/send_queue_stage_writer.py
utils/synthetic/pipeline/row_rebuilder.py
utils/synthetic/pipeline/final_aligned_incremental.py
utils/synthetic/pipeline/final_aligned_observation_writer.py
```

## Core Utility Files

```text
utils/core/paths.py
utils/core/config_loader.py
utils/core/file_io.py
utils/core/logging_setup.py
utils/core/logging_profiler.py
utils/core/ledger.py
utils/core/truths.py
utils/core/artifacts.py
utils/core/wandb_utils.py
```

---

# Runtime Roles and Permissions

The bootstrap creates two runtime database users:

```text
kafka_producer
kafka_ingest
```

## `kafka_producer`

The producer role is used by the Kafka producer service.

It needs access to:

```text
capstone.simulation_state_control
capstone.synthetic_sensor_messages_send_queue
capstone.simulation_timing_config
```

Required actions include:

```text
read producer control settings
claim pending queue rows
update queue rows as sent or failed
read simulation timing configuration
```

## `kafka_ingest`

The ingest role is used by the Kafka consumer service.

It needs access to:

```text
capstone.synthetic_sensor_messages_consumed_stage
capstone.simulation_timing_config
```

Required actions include:

```text
insert landed Kafka messages
read landed messages for verification/debugging
read simulation timing configuration
```

---

# Temporary Ownership Design

The producer and consumer utilities may still perform limited runtime DDL checks, such as:

```sql
ALTER TABLE ... ADD COLUMN IF NOT EXISTS
```

PostgreSQL requires table ownership for `ALTER TABLE`, so the bootstrap temporarily assigns ownership of runtime tables to the runtime roles.

Current temporary design:

```text
kafka_producer owns producer runtime tables
kafka_ingest owns consumer landed-message table
```

Final target design:

```text
bootstrap/migrations own all DDL
producer/consumer code performs no DDL
runtime users receive only minimal SELECT/INSERT/UPDATE permissions
table ownership returns to admin/capstone owner
```

This is an intentional transitional design while the runtime code is being standardized.

---

# Manual Reset Procedure

A manual schema reset file is stored at:

```text
sql/999_manual_database_reset.sql
```

It should be used only when intentionally resetting the local development database.

Recommended reset scope:

```sql
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
```

The reset file should not drop the physical database or runtime users unless a full credential reset is required.

---

# Fresh Docker/Postgres Rebuild Procedure

A fresh database rebuild can be performed with:

```bash
docker compose down
docker volume rm dcook_capstone_postgres_data
docker compose up -d postgres
docker compose up db_bootstrap
```

Expected bootstrap ending:

```text
[db-bootstrap] Complete.
```

After bootstrap, verify tables with:

```bash
docker compose exec postgres psql \
  -U dcook_admin \
  -d dcook_capstone_postgres_db \
  -c "
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema IN ('capstone', 'bronze', 'silver', 'gold', 'metadata')
ORDER BY table_schema, table_name;
"
```

Verify the seeded producer control row:

```bash
docker compose exec postgres psql \
  -U dcook_admin \
  -d dcook_capstone_postgres_db \
  -c "
SELECT *
FROM capstone.simulation_state_control;
"
```

Verify the producer send queue exists:

```bash
docker compose exec postgres psql \
  -U dcook_admin \
  -d dcook_capstone_postgres_db \
  -c "
SELECT queue_status, COUNT(*)
FROM capstone.synthetic_sensor_messages_send_queue
GROUP BY queue_status;
"
```

Before the synthetic send queue notebook is rerun, this table may be empty. That is expected.

---

# Kafka Topic Setup

The Kafka topic initialization service creates:

```text
pump.telemetry.synthetic
```

Expected topic configuration:

```env
KAFKA_TOPIC=pump.telemetry.synthetic
KAFKA_TOPIC_PARTITIONS=3
KAFKA_TOPIC_REPLICATION_FACTOR=1
```

For a local single-broker Docker setup, a replication factor of `1` is expected.

---

# Producer/Consumer Startup Expectations

## Before send queue is populated

If the producer is started before the send queue is populated, the expected result is:

```text
status: empty
claimed_rows: 0
sent_rows: 0
failed_rows: 0
topic: pump.telemetry.synthetic
```

This confirms that infrastructure exists and the producer can start cleanly.

## After send queue is populated

After running the synthetic queue-building notebook/stage, the producer should claim and send rows.

Expected queue check:

```sql
SELECT
    dataset_id,
    run_id,
    queue_status,
    COUNT(*) AS row_count
FROM capstone.synthetic_sensor_messages_send_queue
GROUP BY dataset_id, run_id, queue_status
ORDER BY dataset_id, run_id, queue_status;
```

Expected result should include:

```text
pump_synthetic_v1 | synthetic_run_001 | pending | <row_count>
```

After producer execution, rows should move toward:

```text
sent
failed
```

depending on delivery results.

---

# Notebook SQL Usage

The medallion notebooks use a SQL runtime context cell near the top:

```python
engine = get_engine_from_env()

CAPSTONE_SCHEMA = os.getenv("CAPSTONE_SCHEMA", "capstone")

DATASET_ID = os.getenv(
    "SYNTHETIC_DATASET_ID",
    globals().get("DATASET_NAME", "pump_synthetic_v1"),
)

RUN_ID = os.getenv(
    "SYNTHETIC_RUN_ID",
    globals().get("RUN_ID", "synthetic_run_001"),
)

print(f"SQL schema: {CAPSTONE_SCHEMA}")
print(f"Dataset ID: {DATASET_ID}")
print(f"Run ID: {RUN_ID}")
```

A SQL smoke check cell should follow:

```python
sql_smoke_check_df = read_sql_dataframe(
    engine,
    """
    SELECT
        table_schema,
        table_name
    FROM information_schema.tables
    WHERE table_schema IN (:capstone_schema, 'bronze', 'silver', 'gold', 'metadata')
    ORDER BY table_schema, table_name
    """,
    params={"capstone_schema": CAPSTONE_SCHEMA},
)

display(sql_smoke_check_df)
```

Near the bottom of the notebooks, short SQL writer calls are used instead of long inline SQL blocks.

Example:

```python
bronze_sql_summary_df = write_bronze_sensor_observations_sql(
    engine=engine,
    capstone_schema=CAPSTONE_SCHEMA,
    dataset_id=DATASET_ID,
    run_id=RUN_ID,
    notebook_globals=globals(),
    dataset_name=globals().get("DATASET_NAME", DATASET_ID),
)

display(bronze_sql_summary_df)
```

These utility writers keep notebooks clean while preserving explicit SQL-facing outputs.

---

# Notebook SQL Writer Mapping

## Bronze Preprocessing Notebook

Notebook:

```text
notebooks/preprocessing/EDA_Notebook_Pump_Bronze_01_Preprocessing.ipynb
```

Writer:

```python
write_bronze_sensor_observations_sql(...)
```

Target table:

```text
bronze.sensor_observations
```

## Silver Pre-EDA Notebook

Notebook:

```text
notebooks/eda/EDA_Notebook_Pump_Silver_01_PreEDA.ipynb
```

Writer:

```python
write_silver_sensor_observation_features_sql(...)
```

Target table:

```text
silver.sensor_observation_features
```

## Silver EDA Notebooks

Notebooks:

```text
notebooks/eda/EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3.ipynb
notebooks/eda/EDA_Notebook_Pump_Silver_02b_EDA_v2.ipynb
```

Writer/logger:

```python
log_silver_eda_sql(...)
```

Target tables:

```text
capstone.pipeline_runs
capstone.data_quality_events
capstone.pipeline_artifacts
```

## Gold Preprocessing Notebook

Notebook:

```text
notebooks/experiments/EDA_Notebook_Pump_Gold_01_PreProcessing.ipynb
```

Writer:

```python
write_gold_preprocessed_features_sql(...)
```

Target table:

```text
gold.preprocessed_features
```

## Gold Baseline Notebook

Notebook:

```text
notebooks/experiments/EDA_Notebook_Pump_Gold_02_Baseline_Modeling.ipynb
```

Writer:

```python
write_gold_baseline_scores_sql(...)
```

Target table:

```text
gold.anomaly_detection_scores
```

## Gold Cascade Notebooks

Notebooks:

```text
notebooks/experiments/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling.ipynb
notebooks/experiments/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling.ipynb
notebooks/experiments/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling.ipynb
```

Writer:

```python
write_gold_cascade_scores_sql(...)
```

Target table:

```text
gold.anomaly_detection_scores
```

## Gold Comparison Notebook

Notebook:

```text
notebooks/experiments/EDA_Notebook_Pump_Gold_04_Comparision.ipynb
```

Writer:

```python
write_gold_model_comparison_results_sql(...)
```

Target table:

```text
gold.model_comparison_results
```

## Gold 05 Anomaly Detection Summary Notebook

Notebook:

```text
notebooks/experiments/EDA_Notebook_Pump_Gold_05_Anomaly_Detection.ipynb
```

Writer/logger:

```python
log_gold_05_anomaly_detection_summary_sql(...)
```

Target tables:

```text
capstone.pipeline_runs
capstone.data_quality_events
capstone.pipeline_artifacts
```

---

# Recommended Notebook Markdown Additions

## SQL Runtime Context Markdown

Add this before the SQL runtime context cell:

```markdown
## SQL Runtime Context

This notebook writes selected outputs to the local PostgreSQL database for traceability, recovery, and operational inspection. The database write does not replace the Parquet, CSV, JSON, or W&B artifact outputs. Instead, it provides a SQL-facing layer for pipeline metadata, medallion summaries, and selected row-level outputs.

The SQL context below resolves the active schema, dataset ID, and run ID from environment variables so that notebook outputs remain aligned with Docker, Kafka, and Postgres runtime configuration.
```

## SQL Smoke Check Markdown

Add this before the SQL smoke check cell:

```markdown
## SQL Smoke Check

This check confirms that the expected database schemas and tables are available before the notebook attempts to write SQL-facing outputs. If this cell fails, the database bootstrap should be rerun before continuing.
```

## Bottom SQL Write Markdown

Add this before each bottom SQL write/logging cell:

```markdown
## SQL Write / Metadata Logging

This final step writes the notebook’s selected final output or summary metadata to PostgreSQL. The write is designed to be rerunnable for the current dataset/run and supports SQL-based inspection of the medallion pipeline.

These SQL records supplement the project’s file-based artifacts and do not replace the notebook’s primary saved outputs.
```

---

# Configurables Cell Standard

Each medallion notebook should use a consistent configurables cell near the top.

Recommended pattern:

```python
# =============================================================================
# Notebook Configurables
# =============================================================================

import os

STAGE_NAME = "replace_with_stage_name"
LAYER_NAME = "replace_with_layer_name"

DATASET_NAME = "pump_sensor"
DATASET_ID = os.getenv("SYNTHETIC_DATASET_ID", "pump_synthetic_v1")
RUN_ID = os.getenv("SYNTHETIC_RUN_ID", "synthetic_run_001")
CAPSTONE_SCHEMA = os.getenv("CAPSTONE_SCHEMA", "capstone")

EXECUTION_MODE = os.getenv("EXECUTION_MODE", "notebook")
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "batch")

print(f"Stage: {STAGE_NAME}")
print(f"Layer: {LAYER_NAME}")
print(f"Dataset name: {DATASET_NAME}")
print(f"Dataset ID: {DATASET_ID}")
print(f"Run ID: {RUN_ID}")
print(f"Capstone schema: {CAPSTONE_SCHEMA}")
```

Stage-specific options should be grouped below the common fields.

Example:

```python
# =============================================================================
# Stage-Specific Options
# =============================================================================

WRITE_SQL_OUTPUTS = True
LOG_TO_WANDB = True
SAVE_ARTIFACTS = True
```

---

# Gold 05 Layout Recommendation

Gold 05 should be treated as the final anomaly detection summary/reporting notebook, not as another modeling notebook.

Recommended purpose:

```text
Gold 05 consolidates outputs from baseline modeling, cascade modeling, and comparison notebooks. It prepares the final anomaly detection summary, validates the selected run outputs, and logs final reporting metadata to SQL.
```

Recommended layout:

```text
1. Title and notebook purpose
2. Ask & Answer summary
3. Imports
4. Notebook configurables
5. SQL runtime context
6. SQL smoke check
7. Load selected run artifacts
8. Validate required inputs
9. Summarize baseline results
10. Summarize cascade results
11. Summarize comparison results
12. Final interpretation table
13. Save final report artifacts
14. SQL summary logging
15. Notebook conclusion / deliverable notes
```

Recommended intro markdown:

```markdown
# Gold 05 — Final Anomaly Detection Summary

This notebook consolidates the Gold modeling outputs into a final anomaly detection summary. It is not intended to retrain the baseline or cascade models. Instead, it loads the selected run outputs, validates that required artifacts are present, summarizes baseline and cascade behavior, and prepares final comparison outputs for the capstone report.

The key comparison is whether the cascade approach reduces alert volume and improves alert quality compared with a single Isolation Forest baseline while preserving useful anomaly sensitivity.
```

---

# Pipeline Script Wrapper Direction

Pipeline scripts should become thin wrappers around utility functions.

The goal is to avoid duplicated logic between notebooks and scripts.

Recommended wrapper principles:

```text
Scripts should:
  - parse environment variables or CLI arguments
  - call tested utility functions
  - log concise execution results
  - exit with clear status codes

Scripts should not:
  - duplicate notebook transformation logic
  - define large data processing workflows inline
  - maintain separate versions of pipeline behavior
```

Likely scripts to review:

```text
infrastructure/kafka/producers/run_water_pump_synthetic_producer.py
infrastructure/kafka/consumers/run_kafka_db_consumer.py
infrastructure/kafka/consumers/db_consumer.py
```

Potential future structure:

```text
app/scripts/
  run_synthetic_generation.py
  run_send_queue_build.py
  run_kafka_producer.py
  run_kafka_consumer.py
  run_row_rebuild.py
  run_final_alignment.py
```

Each script should import from:

```text
utils/synthetic/pipeline/
utils/database/
utils/core/
```

---

# Optional GPU Docker Override

The base Docker Compose file should not require GPU support.

GPU support should move into:

```text
docker-compose.gpu.yaml
```

Example override:

```yaml
services:
  app:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: ["gpu"]

  producer_water_pump_synthetic:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: ["gpu"]

  db_consumer:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: ["gpu"]
```

Normal startup:

```bash
docker compose up -d
```

GPU startup:

```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d
```

This keeps the project runnable on systems without NVIDIA support while preserving optional GPU capability.

---

# Current Limitations

The database layer is currently an operational support layer, not a complete enterprise data warehouse.

Current limitations include:

1. Some richer analytical outputs are still stored primarily as files.
2. Runtime producer/consumer utilities may still perform limited DDL safety checks.
3. Medallion SQL shell tables are intentionally lightweight.
4. The notebook pipeline remains the authoritative transformation flow.
5. The bootstrap is currently one file and should be split into smaller SQL files for maintainability.
6. Some dataframe/table contracts may continue to evolve as the capstone stabilizes.
7. Some SQL writer functions may need dataframe name adjustments depending on final notebook variable names.
8. Gold 05 still needs a layout pass to clarify its role as a final summary/reporting notebook.

---

# Future Improvements

Planned improvements include:

1. Split the bootstrap into smaller SQL files.
2. Remove runtime DDL from producer and consumer code.
3. Move table ownership back to an admin/capstone owner role.
4. Formalize richer Bronze, Silver, and Gold relational table contracts.
5. Add more database-level validation queries.
6. Add optional database views for reporting.
7. Add a clearer migration/versioning process for schema changes.
8. Add optional GPU Docker Compose override for systems that support NVIDIA runtime.
9. Standardize configurables cells across all notebooks.
10. Improve notebook markdown flow and deliverable context.
11. Convert pipeline scripts into thin wrappers around utilities.
12. Complete docstring cleanup across utility modules.