# Technical Reference Index

## Purpose

This file is the entry point for the WGU Data Analytics Capstone technical reference documentation. It indexes the documentation in `technical_reference/`, explains how the layers relate, and points a reviewer or maintainer to the right document for a given need. It is a navigation layer only — it does not restate the content of the references it links.

## Recommended Reading Order

1. **Project manual relationship maps** (`00_project_manual/`) — understand the pipeline shape, notebook dependencies, and artifact/table handoffs first.
2. **Notebook workflow references** (`01_notebook_workflow_references/`) — read what each notebook does, in medallion order (Bronze → Silver → Gold).
3. **Notebook deep technical references** (`02_notebook_deep_technical_references/`) — read why the important technical decisions were made for any notebook needing deeper understanding.
4. **Utility module references** (`03_utility_module_references/`) — broad module-level coverage of the supporting `utils/` code.
5. **Deep utility function references** (`04_deep_utility_function_references/`) — curated deep dives on selected high-impact functions.

## Project Manual Files

| File | Purpose | When to Use |
|---|---|---|
| [artifact_and_table_handoff_map.md](artifact_and_table_handoff_map.md) | Maps which artifacts and SQL tables each stage produces and consumes | When tracing a specific Parquet, JSON, model, or table across stages |
| [medallion_handoff_map.md](medallion_handoff_map.md) | Bronze → Silver → Gold layer handoff overview | When you need the high-level data-flow picture across the medallion architecture |
| [notebook_dependency_matrix.md](notebook_dependency_matrix.md) | Notebook-to-notebook dependency grid | When checking which notebooks feed or depend on a given notebook |
| [notebook_relationship_map.md](notebook_relationship_map.md) | Narrative + diagram of notebook relationships and execution order | When you want the overall pipeline relationship story |

## Notebook Workflow References

| Layer | Notebook | Workflow Reference |
|---|---|---|
| Bronze | Bronze 01 Preprocessing | [Bronze 01](../01_notebook_workflow_references/EDA_Notebook_Pump_Bronze_01_Preprocessing_code_reference.md) |
| Silver | Silver 01 Pre-EDA | [Silver 01](../01_notebook_workflow_references/EDA_Notebook_Pump_Silver_01_PreEDA_code_reference.md) |
| Silver | Silver 02a Subset Builder | [Silver 02a](../01_notebook_workflow_references/EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3_code_reference.md) |
| Silver | Silver 02b Profiled-State EDA | [Silver 02b](../01_notebook_workflow_references/EDA_Notebook_Pump_Silver_02b_EDA_v2_code_reference.md) |
| Gold | Gold 01 Preprocessing | [Gold 01](../01_notebook_workflow_references/EDA_Notebook_Pump_Gold_01_PreProcessing_code_reference.md) |
| Gold | Gold 02 Baseline Modeling | [Gold 02](../01_notebook_workflow_references/EDA_Notebook_Pump_Gold_02_Baseline_Modeling_code_reference.md) |
| Gold | Gold 03a Cascade Modeling | [Gold 03a](../01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling_code_reference.md) |
| Gold | Gold 03b Cascade Modeling | [Gold 03b](../01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling_code_reference.md) |
| Gold | Gold 03c Cascade Modeling | [Gold 03c](../01_notebook_workflow_references/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling_code_reference.md) |
| Gold | Gold 04 Comparison | [Gold 04](../01_notebook_workflow_references/EDA_Notebook_Pump_Gold_04_Comparison_code_reference.md) |
| Gold | Gold 05 Anomaly Detection | [Gold 05](../01_notebook_workflow_references/EDA_Notebook_Pump_Gold_05_Anomaly_Detection_code_reference.md) |
| Gold | Gold 06A Test Replay Validation | [Gold 06A](../01_notebook_workflow_references/EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation_code_reference.md) |
| Gold | Gold 06B Test Early-Warning Validation | [Gold 06B](../01_notebook_workflow_references/EDA_Notebook_Pump_Gold_06B_Test_Early_Warning_Validation_code_reference.md) |

## Notebook Deep Technical References

| Layer | Notebook | Deep Technical Reference | Primary Focus |
|---|---|---|---|
| Bronze | Bronze 01 | [Bronze 01 deep](../02_notebook_deep_technical_references/EDA_Notebook_Pump_Bronze_01_Preprocessing_deep_technical_reference.md) | Raw ingestion, identity resolution, truth-chain origin |
| Silver | Silver 01 | [Silver 01 deep](../02_notebook_deep_technical_references/EDA_Notebook_Pump_Silver_01_PreEDA_deep_technical_reference.md) | Pre-EDA, feature selection, missingness quarantine, truth continuation |
| Silver | Silver 02a | [Silver 02a deep](../02_notebook_deep_technical_references/EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3_deep_technical_reference.md) | Clean-normal profiling pipeline, profiled-state labeling |
| Silver | Silver 02b | [Silver 02b deep](../02_notebook_deep_technical_references/EDA_Notebook_Pump_Silver_02b_EDA_v2_deep_technical_reference.md) | Profiled-state EDA, generator input bundle, EDA profile truth |
| Gold | Gold 01 | [Gold 01 deep](../02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_01_PreProcessing_deep_technical_reference.md) | Preprocessing foundation, split, scaling, stage feature sets, truth stamping |
| Gold | Gold 02 | [Gold 02 deep](../02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_02_Baseline_Modeling_deep_technical_reference.md) | Baseline Isolation Forest, threshold calibration, scoring, baseline truth |
| Gold | Gold 03a | [Gold 03a deep](../02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03a_Cascade_Modeling_deep_technical_reference.md) | Cascade modeling (variant a) |
| Gold | Gold 03b | [Gold 03b deep](../02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03b_Cascade_Modeling_deep_technical_reference.md) | Cascade modeling (variant b) |
| Gold | Gold 03c | [Gold 03c deep](../02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_03c_Cascade_Modeling_deep_technical_reference.md) | Cascade modeling (variant c) |
| Gold | Gold 04 | [Gold 04 deep](../02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_04_Comparison_deep_technical_reference.md) | Multi-model comparison and selection |
| Gold | Gold 05 | [Gold 05 deep](../02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_05_Anomaly_Detection_deep_technical_reference.md) | Anomaly timeline and early-warning analysis |
| Gold | Gold 06A | [Gold 06A deep](../02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_06A_Test_Replay_Validation_deep_technical_reference.md) | Held-out test replay validation |
| Gold | Gold 06B | [Gold 06B deep](../02_notebook_deep_technical_references/EDA_Notebook_Pump_Gold_06B_Test_Early_Warning_Validation_deep_technical_reference.md) | Early-warning validation, lead-time metrics |

## Utility Module References

The `03_utility_module_references/` set provides broad, module-level coverage of the supporting `utils/` code (57 module references), organized by area:

| Area | Module References | Scope |
|---|---|---|
| core | 12 | Config loading, notebook context, paths, file I/O, ledger, logging, truths, artifacts, W&B helpers |
| database | 5 | PostgreSQL engine, layer read/write, medallion SQL writers, SQL notebook helpers, chunk-stage utilities |
| medallion / bronze | 1 | Bronze preprocessing utilities |
| medallion / silver | 9 | Silver Pre-EDA and EDA add-ons (profiles, groups, onsets, plots, status, dropped, artifacts) |
| medallion / gold | 8 | Gold preprocessing, baseline/cascade modeling, cascade Stage 3 rules, validation contracts, comparison, row tracking, modeling common |
| synthetic / generator | 6 | Synthetic generator, profiles, missingness, export, run, postgres writer |
| synthetic / pipeline | 16 | Kafka adapters, stage writers, row rebuilder, bronze handoff, postgres-to-bronze, queue manager, comparison |

## Deep Utility Function References

The `04_deep_utility_function_references/` set provides deeper coverage for 18 selected high-impact functions and modules. It is a curated subset of the broader module reference set in `03`.

| Area | Deep Function Reference | Purpose |
|---|---|---|
| core | [artifacts](../04_deep_utility_function_references/utils__core__artifacts_deep_function_reference.md) | Artifact directory/snapshot handling |
| core | [config_loader](../04_deep_utility_function_references/utils__core__config_loader_deep_function_reference.md) | Notebook context/config resolution |
| core | [truths](../04_deep_utility_function_references/utils__core__truths_deep_function_reference.md) | Truth record build, stamp, index, lineage |
| database | [layer_postgres](../04_deep_utility_function_references/utils__database__layer_postgres_deep_function_reference.md) | Layer-level PostgreSQL read/write |
| database | [medallion_sql_writers](../04_deep_utility_function_references/utils__database__medallion_sql_writers_deep_function_reference.md) | Stage-specific SQL write functions |
| gold | [cascade_row_tracking](../04_deep_utility_function_references/utils__medallion__gold__cascade_row_tracking_deep_function_reference.md) | Per-row scoring lineage for cascade stages |
| gold | [gold_baseline_modeling](../04_deep_utility_function_references/utils__medallion__gold__gold_baseline_modeling_deep_function_reference.md) | Baseline modeling helpers |
| gold | [gold_cascade_modeling](../04_deep_utility_function_references/utils__medallion__gold__gold_cascade_modeling_deep_function_reference.md) | Cascade stage modeling helpers |
| gold | [gold_cascade_stage3_rules](../04_deep_utility_function_references/utils__medallion__gold__gold_cascade_stage3_rules_deep_function_reference.md) | Stage 3 rule/profile confirmation |
| gold | [gold_cascade_validation_contracts](../04_deep_utility_function_references/utils__medallion__gold__gold_cascade_validation_contracts_deep_function_reference.md) | Validation contract construction |
| gold | [gold_preprocessing](../04_deep_utility_function_references/utils__medallion__gold__gold_preprocessing_deep_function_reference.md) | Gold preprocessing helpers |
| generator | [generator](../04_deep_utility_function_references/utils__synthetic__generator__generator_deep_function_reference.md) | Synthetic signal generation |
| generator | [missingness](../04_deep_utility_function_references/utils__synthetic__generator__missingness_deep_function_reference.md) | Synthetic missingness modeling |
| pipeline | [bronze_handoff](../04_deep_utility_function_references/utils__synthetic__pipeline__bronze_handoff_deep_function_reference.md) | Synthetic-to-Bronze handoff |
| pipeline | [kafka_producer_adapter](../04_deep_utility_function_references/utils__synthetic__pipeline__kafka_producer_adapter_deep_function_reference.md) | Kafka producer adapter |
| pipeline | [postgres_to_bronze](../04_deep_utility_function_references/utils__synthetic__pipeline__postgres_to_bronze_deep_function_reference.md) | PostgreSQL-to-Bronze staging |
| pipeline | [row_rebuilder](../04_deep_utility_function_references/utils__synthetic__pipeline__row_rebuilder_deep_function_reference.md) | Streamed-row reconstruction |
| pipeline | [send_queue_stage_writer](../04_deep_utility_function_references/utils__synthetic__pipeline__send_queue_stage_writer_deep_function_reference.md) | Send-queue stage writer |

## Maintenance Notes

- Update a notebook's workflow reference and its deep technical reference **together** when the notebook materially changes (role, inputs, outputs, validation, persistence, or lineage).
- Update the relationship and handoff maps when notebook dependencies or artifact/SQL handoffs change.
- Update the utility module and deep function references when the corresponding `utils/` code changes.
