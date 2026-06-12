# dcook-wgu-capstone-unified-predictive-maintenance-pipeline

## Project purpose

This project is my WGU Data Analytics capstone for industrial pump anomaly detection. The goal is to evaluate whether a staged anomaly-detection pipeline can produce a more useful maintenance alert stream than a single broad Isolation Forest baseline.

The main research question is:

> Does a multi-stage anomaly detection cascade improve alert quality compared with a single Isolation Forest baseline for industrial pump sensor data?

This project is not framed as a fully autonomous production failure predictor. I am treating the final model as a maintenance triage aid: it reduces alert burden, preserves useful early-warning behavior, and gives reviewers a structured way to inspect abnormal pump behavior before a recorded broken state.

---

## Project architecture

The project follows a Medallion-style analytical structure:

```text
Bronze -> Silver -> Gold
```

### Bronze

Bronze handles raw dataset ingestion and standardization. It creates the first clean project layer, adds metadata fields, writes artifacts, and prepares the dataset for downstream profiling.

### Silver

Silver handles exploratory data analysis, sensor profiling, clean-normal review, feature inspection, and dataset-quality checks. The Silver layer creates the analytical context needed before modeling.

### Gold

Gold handles model-ready preprocessing, baseline modeling, cascade modeling, model comparison, and final anomaly timeline review.

The main Gold model sequence is:

```text
Gold 01 -> Gold 02 -> Gold 03a -> Gold 03b -> Gold 03c -> Gold 04 -> Gold 05
```

Where:

- **Gold 01** prepares model-ready features.
- **Gold 02** trains and evaluates the baseline Isolation Forest.
- **Gold 03a** builds the default cascade.
- **Gold 03b** builds the tuned cascade.
- **Gold 03c** builds the Stage 3 improved cascade and operating modes.
- **Gold 04** compares the baseline, cascade variants, and Stage 3 operating modes.
- **Gold 05** builds the final anomaly timeline and early-warning review artifacts.

---

## Technology stack

The project uses:

- Python
- Jupyter notebooks
- pandas and NumPy
- scikit-learn
- matplotlib and seaborn
- PostgreSQL
- Kafka
- Docker / Docker Compose
- Weights & Biases tracking
- YAML-based configuration files
- Custom project utilities under `utils/`

The current project was developed inside a containerized workspace.

```text
Container project root: /workspace
Windows project root:   D:\wgu\capstone
```

---

## Local setup and startup: Windows + WSL + Docker

This project is meant to be started from WSL, not directly from a Windows PowerShell prompt. The Windows project folder is still the same folder, but WSL sees it through `/mnt/d`.

```text
Windows path: D:\wgu\capstone
WSL path:     /mnt/d/wgu/capstone
Container:    /workspace
```

### 1. Install or verify WSL 2

From Windows PowerShell:

```powershell
wsl --install -d Ubuntu
wsl --update
wsl -l -v
```

The Ubuntu distribution should show `VERSION 2`.

If it does not, set WSL 2 as the default and convert the distro:

```powershell
wsl --set-default-version 2
wsl --set-version Ubuntu 2
```

If the machine already has WSL installed, the important check is still:

```powershell
wsl -l -v
```

### 2. Install Docker Desktop and enable WSL integration

Docker Desktop should be installed on Windows and configured to use the WSL 2 backend.

In Docker Desktop:

```text
Settings -> Resources -> WSL Integration
```

Enable integration for the Ubuntu/WSL distribution used for this project.

Then open WSL and verify that Docker is visible from inside the Linux shell:

```bash
docker --version
docker compose version
docker info
```

If WSL prints `The command 'docker' could not be found in this WSL 2 distro`, Docker Desktop is either not running or WSL integration is not enabled for the distro.

### 3. Open the project from WSL

From PowerShell:

```powershell
wsl
```

Then from the WSL shell:

```bash
cd /mnt/d/wgu/capstone
pwd
ls
```

The project root should contain files such as:

```text
start.sh
docker-compose.yaml
.env.example
app/
configs/
notebooks/
utils/
```

### 4. Create the local `.env` file

The project expects a local `.env` file in the project root.

```bash
cp .env.example .env
nano .env
```

Update the values for local passwords, W&B settings, Kafka settings, and Postgres settings as needed.

Do not commit a real `.env` file. Keep real API keys and passwords out of Git and out of shared review bundles.

### 5. Start the project

The normal project startup command is:

```bash
chmod +x start.sh
./start.sh
```

The `start.sh` script performs the project startup wrapper and then runs:

```bash
docker compose up -d
```

This brings up the main project services, including Postgres, pgAdmin, Kafka, bootstrap steps, and the long-running app container.

### 6. Check container status

After startup, check the services:

```bash
docker compose ps
```

Useful log commands:

```bash
docker compose logs -f app
docker compose logs -f postgres
docker compose logs -f db_bootstrap
docker compose logs -f kafka
docker compose logs -f kafka_topic_init
```

The expected healthy path is:

```text
postgres healthy
kafka healthy
db_bootstrap completed successfully
kafka_topic_init completed successfully
app running
pgadmin running
```

### 7. Work inside the app container

The app container mounts the project root into `/workspace` and uses `/workspace` as the working directory.

To open a shell inside the app container:

```bash
docker exec -it dcook_capstone_app bash
cd /workspace
```

From there, notebooks and Python commands should resolve project imports using the container-side paths.

### 8. Stop the project

To stop the containers without deleting volumes:

```bash
docker compose down
```

Only use volume removal when intentionally resetting local database state:

```bash
docker compose down -v
```

That command deletes Docker-managed volumes, including the local Postgres data volume.

---

## Main folder structure

```text
.
├── app/
├── artifacts/
├── bootstrap/
├── configs/
├── data/
├── docs/
├── infrastructure/
├── logs/
├── models/
├── notebooks/
│   ├── preprocessing/
│   ├── eda/
│   ├── experiments/
│   └── synthetic/
├── pipelines/
├── project_tools/
├── scripts/
├── sql/
└── utils/
```

Important output roots:

```text
/workspace/data
/workspace/artifacts
/workspace/logs
/workspace/models
```

Some large data files, generated logs, parquet outputs, and model artifacts may be intentionally excluded from shared review bundles or compressed submissions because of file size and file-lock constraints.

---

## Configuration-driven execution

The notebooks use project configuration files and shared utilities instead of hard-coded one-off paths wherever possible. The configuration layer resolves:

- dataset identity
- run identity
- artifact directories
- truth and ledger locations
- model parameters
- runtime mode
- pipeline/profile settings

This matters because the project is intended to move from notebook execution into CLI pipeline execution after the notebooks are stable.

---

## Notebook execution order

Run the notebooks in this order for the main analytical pipeline:

```text
1. notebooks/preprocessing/EDA_Notebook_Pump_Bronze_01_Preprocessing.ipynb
2. notebooks/eda/EDA_Notebook_Pump_Silver_01_PreEDA.ipynb
3. notebooks/eda/EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3.ipynb
4. notebooks/eda/EDA_Notebook_Pump_Silver_02b_EDA_v2.ipynb
5. notebooks/experiments/EDA_Notebook_Pump_Gold_01_PreProcessing.ipynb
6. notebooks/experiments/EDA_Notebook_Pump_Gold_02_Baseline_Modeling.ipynb
7. notebooks/experiments/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling.ipynb
8. notebooks/experiments/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling.ipynb
9. notebooks/experiments/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling.ipynb
10. notebooks/experiments/EDA_Notebook_Pump_Gold_04_Comparision.ipynb
11. notebooks/experiments/EDA_Notebook_Pump_Gold_05_Anomaly_Detection.ipynb
```

Synthetic notebooks are used to support the Kafka/PostgreSQL simulation path, but they are not part of the first CLI conversion build.

---

## Docker, PostgreSQL, and Kafka

The project includes Docker-based infrastructure for PostgreSQL, pgAdmin, Kafka, and supporting bootstrap steps.

The preferred startup path is the project wrapper script:

```bash
./start.sh
```

The script checks for the local `.env` file, warns if W&B authentication is not configured, and then starts the Docker Compose stack. Direct `docker compose` commands are still useful for debugging, but the standard project startup command should stay as `./start.sh`.

The database layer includes schemas such as:

```text
bronze
silver
gold
capstone
metadata
```

The SQL infrastructure supports pipeline run logging, data-quality events, artifact tracking, synthetic message staging, and medallion-layer output tables.

PostgreSQL and Kafka are included because the project is designed to be more than a single notebook analysis. The current capstone deliverable focuses on the analytical result, while the infrastructure supports future pipeline, streaming, and dashboard extensions.

---

## Artifact, truth, and ledger design

The project writes artifacts into structured layer folders. These artifacts include CSV outputs, JSON summaries, plots, model files, ledgers, and truth records.

The truth-record design is used to track lineage between stages. Outputs are stamped with metadata such as:

```text
meta__truth_hash
meta__parent_truth_hash
meta__pipeline_mode
```

This makes it easier to verify that model comparisons are using outputs from the same upstream data lineage.

Example truth layout:

```text
artifacts/truths/
├── truth_index.jsonl
├── silver/
├── gold_baseline/
├── gold_cascade/
├── gold_comparison/
└── gold_anomaly_detection/
```

Ledger files are written as run-level records so that each notebook has a basic audit trail of major steps, decisions, outputs, and validation checks.

---

## Modeling approach

The baseline model is a single Isolation Forest. It is used as a broad high-sensitivity reference model.

The cascade approach uses staged filtering:

```text
Stage 1: broad anomaly detection
Stage 2: narrower anomaly scoring
Stage 3: confirmation / operating-mode logic
```

The purpose of the cascade is not just to maximize recall. The purpose is to test whether alert volume can be reduced while still preserving useful detection behavior.

---

## Main current results

Current Gold comparison results from the latest reviewed run:

| Model | Test alerts | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline IsolationForest | 31,200 | 0.003782 | 1.000000 | 0.007536 |
| Cascade Default | 24,895 | 0.003093 | 0.652542 | 0.006157 |
| Cascade Tuned | 15,153 | 0.005345 | 0.686441 | 0.010608 |
| Stage 3 Improved | 6,594 | 0.010616 | 0.593220 | 0.020858 |
| Stage 3 Relaxed | 13,713 | 0.005834 | 0.677966 | 0.011568 |
| Stage 3 Medium | 7,286 | 0.010705 | 0.661017 | 0.021070 |
| Stage 3 Strict | 61 | 0.442623 | 0.228814 | 0.301676 |

The Stage 3 Improved cascade reduced test alerts from:

```text
31,200 -> 6,594
```

That is:

```text
24,606 fewer alerts
about 78.9% fewer alerts than the baseline
```

Stage 3 Strict has the highest precision and F1, but it is much more conservative and has lower recall. For the final project narrative, Stage 3 Improved is treated as the main practical model, while Strict is better described as a conservative audit mode.

---

## Gold 05 early-warning review

Gold 05 builds the final anomaly timeline and review artifacts. The current Stage 3 Improved early-warning summary is:

| Field | Value |
|---|---:|
| Row count | 220,320 |
| First alert plot order index | 5,201 |
| First broken plot order index | 17,155 |
| Lead time minutes to failure | 11,954 |
| Total final alert rows | 37,024 |

Detection class counts:

| Detection class | Count |
|---|---:|
| no_alert | 183,296 |
| false_positive | 37,021 |
| early_warning | 3 |

The final interpretation is cautious: the cascade materially reduced alert burden and preserved useful early-warning behavior, but it still produced many false positives. It should be treated as a triage aid, not an autonomous production failure predictor.

---

## Dashboard-ready outputs

The project already writes several outputs that can support a future dashboard:

```text
artifacts/gold/pump/comparison/results/pump__gold__model_comparison.csv
artifacts/gold/pump/comparison/summaries/pump__gold__model_comparison_summary.json
artifacts/gold/pump/anomaly_detection/summaries/stage3_improved__detection_summary.json
artifacts/gold/pump/anomaly_detection/summaries/stage3_improved__failure_lead_time_summary.csv
artifacts/gold/pump/anomaly_detection/summaries/multi_run_lead_time_comparison.csv
artifacts/gold/pump/anomaly_detection/exports/stage3_improved__detected_rows_review.csv
artifacts/gold/pump/anomaly_detection/packets/stage3_improved__alert_packet_summary.csv
artifacts/gold/pump/anomaly_detection/packets/stage3_improved__top_alert_packets.csv
```

A future dashboard could use these artifacts to show:

- model comparison metrics
- alert burden by model
- operating-mode tradeoffs
- first alert timing
- lead-time summary
- alert packet summaries
- detected-row review tables
- timeline plots and sensor review plots

---

## Current limitations

The project has several important limitations:

- The dataset contains limited failure examples.
- The final alert stream still contains many false positives.
- Row-level precision is low for the main practical model.
- Stage 3 Strict is precise but too conservative to be the only operating mode.
- The current project is notebook-first and still needs a clean CLI pipeline rebuild.
- PostgreSQL writes need a final validation pass across all medallion layers.
- Event-level alert evaluation would be more realistic than only row-level scoring.
- A production implementation would need more real equipment history, retraining rules, monitoring, and business-defined alert thresholds.

---

## Planned next work

Immediate finish-line tasks:

1. Finish final metrics and document alignment.
2. Regenerate stale Task 3 figures from current Gold artifacts.
3. Polish notebook markdown and section flow.
4. Verify PostgreSQL layer writes.
5. Rebuild the pipeline scripts from scratch instead of using Jupyter exports.
6. Add Gold 05 into the pipeline path.
7. Build a dashboard-ready artifact manifest.
8. Run one final clean test.
9. Package the final capstone submission.

The next pipeline rebuild should be CLI-first, config-driven, and stage-oriented. It should reuse the same project utilities as the notebooks, but it should not depend on notebook globals or exported notebook code.
