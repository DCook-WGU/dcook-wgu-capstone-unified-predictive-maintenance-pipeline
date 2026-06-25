# Codebase Reference Archive — WGU D502 Capstone

*Project documentation artifact for traceability and technical explanation. Not the formal Task 3 reference list.*

**Intended repository path:** `technical_reference/codebase_reference_archive.md`
*(The repository uses the singular `technical_reference/` folder, confirmed by project evidence referencing `technical_reference/00_project_manual/medallion_handoff_map.md` and `technical_reference/05_audit_outputs/function_inventory.json`. If your working copy uses a different folder, move this file accordingly.)*

---

## 1. Purpose of This Reference Archive

This file is a **technical reference archive** that records sources which **support, explain, resemble, or provide implementation context** for the methods used in this project's codebase. Its purpose is traceability, technical explanation, and future review — so that a reader can understand the project's methods and the related literature/documentation without having to re-derive them.

**What this file is *not*:**

- It is **not** the formal Task 3 report reference list. Only a subset of these sources belongs in the paper (see §3).
- It is **not** proof that the project was built from, copied from, derived from, or inspired by any source listed here. The project's own files do **not** document where each design idea originated. Every entry is therefore phrased as a *related reference / comparable implementation / official documentation / methodological precedent / background resource / found-after-implementation comparison* — never as the origin of the design.
- It does **not** imply that every source was consulted during development. Several were located **after** implementation purely to provide comparison or context.

GitHub repositories appear here as **implementation references only** and are never treated as academic or methodological validation.

---

## 2. Evidence Sources Reviewed

**Project folders checked (in the provided project evidence):**

| Folder | Present? | Notes |
|---|---|---|
| `technical_reference/` (singular) | **Referenced/exists in repo** | Project files cite `technical_reference/00_project_manual/medallion_handoff_map.md` and `technical_reference/05_audit_outputs/function_inventory.json`. Used as the destination folder for this archive. |
| `technical_references/` (plural) | Not found | Not present; no files attributed to it. |
| `technical_resource/` | Not found | Not present. |
| `research_context/` (as a literal folder) | Not present as a folder | **However, the research-context *package contents* were provided** as the six files below and were fully used. |

**Research-context files inspected (the research_context package):**

- `00_project_methodology_context.md` — methodology dossier: pipeline architecture, function/module inventory, data-prep and modeling methods, statistical concepts, assumptions/risks, and per-method source needs. *(Primary evidence for this archive.)*
- `01_function_method_map.csv` — function/method map (overlaps the dossier's function inventory; used for confirmation).
- `02_research_seed_table.csv` — method→claim→suggested-source seed table with citation priorities and overclaiming warnings.
- `03_task3_claim_to_method_map.md` — maps project methods to D502 Task 3 rubric sections (A–J) with evidence paths and cautions.
- `04_deep_research_prompt.md` — prior research prompt (context only).
- `05_context_extraction_summary.md` — extraction summary (context only).

**Source-validation authority inspected:**

- `D502_Source_Validation_Map.md` (prior research pass) — used as the **authority** for source tiering: strong academic sources, official documentation, implementation references, background-only sources, and do-not-cite sources, plus the claims-to-soften list. This archive does **not** restart research; it reuses that map's classifications.
- `D502_Task3_Citation_Patch_Plan.md` (prior pass) — used to confirm which sources are actually cited in the report.
- `Danty_Cook-Task_3-v5_clean.docx` (final report) — used to confirm the **as-cited** reference list and the report's data framing.

**Files skipped and why:** `04_deep_research_prompt.md` and `05_context_extraction_summary.md` were read for context but contributed no new sources beyond the dossier, seed table, and validation map; they are not separately mapped below. The repository's *runtime code* itself was not mounted in this session, so file paths below are taken from the dossier/seed table/claim map (which are evidence-grounded extractions of that code).

**Important data-framing note (carried from the report and the project guardrail):** the methodology dossier (`00`) and seed table (`02`) describe the analytical data as *synthetic*. The **final Task 3 report uses the real, public Kaggle "Pump Sensor Data" dataset (nphantawee), 220,320 rows with 7 broken observations.** This archive follows the report: the dataset is treated as a **single public/historical dataset (not production-validated)**, and is **never** described as synthetic. The codebase additionally contains a **synthetic data generator** (`utils/synthetic/`) as *separate supporting infrastructure*; it is documented as such and is not the report's analytical dataset.

---

## 3. Report-Cited Sources

Sources appropriate for the **formal Task 3 paper** because they support an actual report claim. All are present in the final `v5_clean` reference list. (Full citation strings live in the paper; abbreviated here.)

**Academic / methodological (support specific report claims):**

| Source | Supports report claim | Cited in paper | Safe to cite | Caution |
|---|---|---|---|---|
| Liu, Ting & Zhou (2008), *Isolation Forest*, ICDM — doi:10.1109/ICDM.2008.17 | Isolation Forest as the unlabeled anomaly-detection method (A.3, E.1) | ✓ | ✓ Strong (primary method source) | Don't call IF "best/optimal." |
| Dietterich (1998), *Approximate statistical tests…*, Neural Computation 10(7) — doi:10.1162/089976698300017197 | McNemar appropriate for comparing two classifiers on the same test set (E.2, F.2) | ✓ | ✓ Strong | Pair with large-N caveat (significance ≠ adequacy). |
| Kaufman, Rosset, Perlich & Stitelman (2012), *Leakage in data mining*, ACM TKDD 6(4) — doi:10.1145/2382577.2382579 | Episode-aware split + train-only stats prevent common leakage paths (D.2, E.3) | ✓ | ✓ Strong | Say "helped prevent common leakage paths," not "eliminated." |
| Saito & Rehmsmeier (2015), *Precision-recall plot…*, PLOS ONE 10(3) — doi:10.1371/journal.pone.0118432 | Under severe imbalance, accuracy/ROC mislead; PR-based evaluation is more informative (F.1) | ✓ | ✓ Strong | Report computes P/R/F1, not PR curves — don't imply curve analysis. |
| Montgomery (2013), *Introduction to Statistical Quality Control* (7th ed.), Wiley | Stage 3 reference-bound confirmation framed as analogous to control-limit monitoring (E.2) | ✓ | ✓ Acceptable (analogy) | "Analogous to control-limit monitoring," not "used SPC control charts." |
| Lei, Li, Guo, Li, Yan & Lin (2018), *Machinery health prognostics*, MSSP 104 — doi:10.1016/j.ymssp.2017.11.016 | Early detection provides maintenance lead time (prognostics framing) (F.3) | ✓ | ✓ Strong (background framing) | Lead time observed only in the evaluated sequence; not guaranteed prediction. |
| Bilal & Hanif (2025), *Fast anomaly detection… cascades of null-subspace PCA detectors*, Sensors 25(15):4853 — doi:10.3390/s25154853 | Coarse-to-fine cascade precedent (A.3, E.1) | ✓ | ✓ Acceptable (verified; analogy) | Vision-domain; use "informed by / consistent with the cascade principle," not "implemented a proven cascade detector." |
| Ahmad, Purdy, Lavin & Agha (2019), *Evaluating real-time anomaly detection — NAB*, arXiv | Evaluating anomaly detectors by *when* alerts appear (E.3 timeline) | ✓ | ✓ Acceptable | On-point for timeline evaluation. |
| Antonini, Pincheira, Vecchio & Antonelli (2023), *TinyML anomaly detection for industrial environments*, Sensors 23(4):2344 | Lightweight unsupervised models support industrial anomaly detection (A.3, E.1) | ✓ | ✓ Acceptable | Edge/TinyML context; supporting, not core. |

**Dataset and first-party tool docs cited in the report** are listed in §4 (they double as report citations): nphantawee (Kaggle dataset), scikit-learn (IsolationForest, Outlier detection, precision_recall_fscore_support, RobustScaler), statsmodels (McNemar), pandas, Confluent (Kafka client), Docker, Weights & Biases.

> **Removed from the report and not to be reintroduced:** *Tam & Nguyen (1995)* — could not be verified in the prior pass and was replaced by Montgomery (2013). Do not cite it.

---

## 4. Official Tool and Library Documentation

Documentation for tools **actually used** in the codebase. "Cited in paper" marks whether the report's reference list already includes it.

**Core analytical libraries:**

| Tool / page | Org | URL | Related code area | Cited in paper | Recommended use |
|---|---|---|---|---|---|
| scikit-learn — `IsolationForest` | scikit-learn devs | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html | `gold_baseline_modeling.py`, `gold_cascade_modeling.py` | ✓ | Cite in report + technical docs |
| scikit-learn — Outlier detection (user guide) | scikit-learn devs | https://scikit-learn.org/stable/modules/outlier_detection.html | baseline + cascade IF | ✓ | Cite in report + technical docs |
| scikit-learn — `precision_recall_fscore_support` | scikit-learn devs | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html | `gold_modeling_common.py`, Gold 02/04 | ✓ | Cite in report + technical docs |
| scikit-learn — `RobustScaler` | scikit-learn devs | https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html | `gold_preprocessing.py` | ✓ | Cite in report + technical docs |
| scikit-learn — `PCA` | scikit-learn devs | https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html | Silver 02b EDA (`EDA_Notebook_Pump_Silver_02b_EDA_v2.ipynb`) | ✗ | Technical docs only (diagnostic EDA) |
| scikit-learn — `AgglomerativeClustering` | scikit-learn devs | https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html | Silver 02b sensor grouping | ✗ | Technical docs only (diagnostic EDA) |
| scikit-learn — Model persistence | scikit-learn devs | https://scikit-learn.org/stable/model_persistence.html | `.joblib` model save/replay (Gold 02/03/06A) | ✗ | Technical docs only |
| statsmodels — `contingency_tables.mcnemar` | statsmodels devs | https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html | McNemar inline in `EDA_Notebook_Pump_Gold_04_Comparison.ipynb` | ✓ (stable URL) | Cite in report + technical docs |
| joblib — Persistence | joblib devs | https://joblib.readthedocs.io/en/stable/persistence.html | model serialization / replay (Gold 06A) | ✗ | Technical docs only |
| pandas — User guide | pandas dev team | https://pandas.pydata.org/docs/user_guide/index.html | all stages (dataframes) | ✓ | Cite in report + technical docs |
| NumPy — Documentation | NumPy devs | https://numpy.org/doc/stable/ | `np.percentile` thresholds, array math | ✗ | Technical docs only |

**Supporting infrastructure / tooling (not the analytical core):**

| Tool / page | Org | URL | Related code area | Cited in paper | Recommended use |
|---|---|---|---|---|---|
| PostgreSQL — Documentation | PostgreSQL Global Dev Group | https://www.postgresql.org/docs/current/ | `medallion_sql_writers.py`, `sql/SQL_SCHEMA_CREATION.sql`, `infrastructure/postgres/bootstrap/` | ✗ | Technical docs / I–J tooling listing only |
| Apache Kafka — Documentation | Apache Software Foundation | https://kafka.apache.org/documentation/ | `utils/synthetic/pipeline/` (telemetry simulation — supporting infra) | ✗ (report cites Confluent client) | Technical docs only |
| Confluent — Python client for Kafka | Confluent | https://docs.confluent.io/platform/current/clients/python.html | Kafka producer/consumer client | ✓ | Cite in report (tooling) |
| Apache Parquet — Documentation | Apache Software Foundation | https://parquet.apache.org/docs/ | staged `data/**/*.parquet` outputs | ✗ | Technical docs / I–J tooling listing only |
| Docker — Develop best practices | Docker, Inc. | https://docs.docker.com/develop/ | Postgres/infra containerization | ✓ | Cite in report (tooling) |
| Weights & Biases — Artifacts | Weights & Biases | https://docs.wandb.ai/guides/artifacts | experiment/artifact tracking | ✓ | Cite in report (tooling) |

*Guidance:* list a tool in the **paper's** I/J section only if the project actually depends on it. Items marked "technical docs only" (NumPy, joblib, scikit-learn PCA/clustering/persistence, raw Kafka/Parquet/PostgreSQL docs) support code that is either diagnostic or infrastructural and need not appear in the formal reference list unless the report explicitly discusses them.

---

## 5. Academic and Methodological Support

Papers/textbooks that support major analytical, statistical, modeling, or data-engineering methods. Tier follows the source-validation map.

| Source | Method supported | Code area | Tier | Cited in paper | Recommended use | Caution / overclaiming note |
|---|---|---|---|---|---|---|
| Liu, Ting & Zhou (2008) — *Isolation Forest* (ICDM); doi:10.1109/ICDM.2008.17 | Isolation Forest (baseline + cascade stages) | `gold_baseline_modeling.py`, `gold_cascade_modeling.py` | **Strong** (primary) | ✓ | Cite in report | Established standard method — not "optimal/SOTA." |
| Liu, Ting & Zhou (2012) — *Isolation-based anomaly detection* (ACM TKDD 6(1)); doi:10.1145/2133360.2133363 | IF (journal/extended treatment) | same | **Strong** (redundant w/ 2008) | ✗ | Technical docs only | Redundant with the 2008 paper for the report; keep here for completeness. |
| Dietterich (1998) — *Approximate statistical tests…* (Neural Computation 10(7)); doi:10.1162/089976698300017197 | McNemar for classifier comparison | `EDA_Notebook_Pump_Gold_04_Comparison.ipynb` | **Strong** | ✓ | Cite in report | Significance ≠ operational adequacy (large N ≈ 83,889). |
| Kaufman, Rosset, Perlich & Stitelman (2012) — *Leakage in data mining* (ACM TKDD 6(4)); doi:10.1145/2382577.2382579 | Leakage prevention (episode split, train-only imputation/threshold) | `gold_preprocessing.py` | **Strong** | ✓ | Cite in report | "Prevents common leakage paths," not "eliminates." |
| Saito & Rehmsmeier (2015) — *PR plot more informative than ROC…* (PLOS ONE 10(3)); doi:10.1371/journal.pone.0118432 | Metric choice under imbalance (PR vs ROC) | `gold_modeling_common.py` (`roc_auc`, `average_precision`) | **Strong** | ✓ | Cite in report | Don't present ROC-AUC (0.941) alone as success; pair with PR-AUC (0.122)/precision (0.0038). |
| Davis & Goadrich (2006) — *The relationship between PR and ROC curves* (ICML); doi:10.1145/1143844.1143874 | PR/ROC theory (companion to Saito & Rehmsmeier) | same | **Acceptable** | ✗ | Background / technical docs only | Not needed in the report — it computes no PR/ROC curves. One imbalance anchor (Saito & Rehmsmeier) suffices. |
| Viola & Jones (2001) — *Rapid object detection… boosted cascade* (CVPR); doi:10.1109/CVPR.2001.990517 | Coarse-to-fine cascade precedent | `gold_cascade_modeling.py` | **Acceptable** (analogy) | ✗ (report uses Bilal & Hanif instead) | Background / technical docs only | Canonical cascade analogy; not added to the report because Bilal & Hanif (2025) already anchors the cascade. "Informed by," not "implemented." |
| Montgomery (2013) — *Introduction to Statistical Quality Control* (7th ed.), Wiley | Stage 3 reference-bound confirmation as control-limit analogy | `gold_cascade_stage3_rules.py` | **Acceptable** (analogy) | ✓ | Cite in report | "Analogous to control-limit monitoring," not "used SPC control charts." |
| Lei et al. (2018) — *Machinery health prognostics* (MSSP 104); doi:10.1016/j.ymssp.2017.11.016 | Predictive-maintenance lead-time / early-warning framing | `stage3_improved__detection_summary.json` (Gold 05/06B) | **Strong** (framing) | ✓ | Cite in report | Hedge: lead time observed only in the evaluated historical sequence. |
| Rubin (1976) — *Inference and missing data* (Biometrika 63(3)); doi:10.1093/biomet/63.3.581 | Missing-data mechanisms (MCAR/MAR/MNAR) behind feature quarantine | `silver_preeda.py` (`quarantine_features_by_missingness`) | **Background** | ✗ | Background only | Conceptual support for the *governance choice* only; the report makes no mechanism claim, so not cited. |

---

## 6. Comparable Implementation Examples

**Implementation comparison only — not academic or methodological validation.** These open-source repositories demonstrate that comparable patterns exist; quality and maintenance vary; several are tutorial-grade. None is documented as a source the project was built from.

| Repository | URL | What it demonstrates | Resembles (project area) | How to use it | Caution |
|---|---|---|---|---|---|
| nagdevAmruthnath/Predictive-Maintenance | https://github.com/nagdevAmruthnath/Predictive-Maintenance | PdM tutorial incl. an Isolation Forest notebook: train on normal-condition data only, score the rest, threshold on anomaly score | **High** — the baseline IF pattern (normal-only fit + percentile threshold) in `gold_baseline_modeling.py` | Implementation reference that the normal-only-fit + threshold pattern is used elsewhere | Tutorial-grade; partly R; not a published benchmark |
| Divya-Bhargavi/isolation-forest | https://github.com/Divya-Bhargavi/isolation-forest | From-scratch implementation of the original Liu/Ting/Zhou IF algorithm | **Moderate** — the algorithm itself, not a pump pipeline | Reference that the algorithm's mechanics are reproduced independently of scikit-learn | Educational; small repo |
| Sa1f27/predictive-maintenance-mlops | https://github.com/Sa1f27/predictive-maintenance-mlops | End-to-end PdM pipeline: training, drift detection, deployment, monitoring | **Moderate** — overall staged/reproducible pipeline structure | Reference that staged, monitored PdM pipelines are a common pattern | MLOps/deployment focus exceeds this project's notebook scope; its "92% accuracy" is its claim, not this project's |
| NotAndex/kafka_iot_sim | https://github.com/NotAndex/kafka_iot_sim | Docker-composed IoT sensor generator → Kafka topic (keyed by sensor) → consumers, with a layered (Delta) sink | **High** for the Kafka telemetry-simulation pattern in `utils/synthetic/pipeline/`; layered sink is medallion-adjacent | Reference for the producer/consumer sensor-simulation design (supporting infra) | Demonstration project; not a pump domain |
| vinodtkn/MedallionArchitecture | https://github.com/vinodtkn/MedallionArchitecture | Notebook-first medallion demo: one notebook per Bronze/Silver/Gold layer, intermediate data as Parquet | **High** for the notebook-first, Parquet-per-layer structure | Reference that a notebook-first Bronze/Silver/Gold layout is an established pattern | Runs in Databricks cloud; this project is local — note the difference |
| chayansraj/Microsoft-Azure-Medallion-Data-pipeline | https://github.com/chayansraj/Microsoft-Azure-Medallion-Data-pipeline | End-to-end Azure medallion (ADF + Databricks + Parquet/Delta) | **Moderate** (cloud-heavy) | Optional secondary medallion reference | Platform-specific; further from this project's local notebook implementation |

*Recommended use for all of §6:* **cite in technical docs only as implementation comparisons.** Do not place any GitHub repository in the Task 3 reference list, and never present one as evidence that a method is valid.

---

## 7. Project-Defined Heuristics and Engineering Practices

Project-specific choices. Classified as **externally supported**, **loosely analogous** to an external method, **project-defined**, **engineering practice**, or **heuristic**. Per academic-integrity guidance, these are **not over-cited**.

| Practice | Code area | Classification | External relation (if any) | How to treat it |
|---|---|---|---|---|
| **Truth-hash lineage** (`meta__truth_hash` / `meta__parent_truth_hash`, truth index) | `utils/core/truths.py` (`compute_sha256`, `build_truth_record`, `stamp_truth_columns`, `extract_truth_hash`) | **Engineering practice** | Content-addressed lineage underpins tools like Git/DVC (implementation references), but no single method paper matches the design | Describe as "an engineering practice for auditability and reproducibility (content-addressed lineage)." **Do not** cite an academic method paper for it. |
| **Coefficient-of-variation feature stability** (Stage 2 feature reduction by `std/|median|`) | `gold_preprocessing.py` (`choose_stage2_features_from_training_stability`) | **Project-defined heuristic** | CV is a standard statistic, but its use here as a Stage-2 selector is project-defined | Treat as a project-defined heuristic. Optionally note CV is a standard dispersion measure; **do not** claim a feature-selection method paper unless one is genuinely used. |
| **Stage 3 profile confirmation** (breach counts vs per-sensor 5th/95th-pct bounds, persistence, drift) | `gold_cascade_stage3_rules.py` (`compute_primary_breach_count`, `compute_persistence_flag`, `compute_drift_flag`, `compose_stage3_decision`) | **Loosely analogous** to an external method | Analogous to control-limit monitoring in SPC (Montgomery, 2013) | Frame as "analogous to control-limit monitoring." Bounds are project-defined; **not** formal SPC control charts. |
| **Cascade composition** (Stage 1 broad IF → Stage 2 narrow IF → Stage 3 rules) | `gold_cascade_modeling.py`, `gold_cascade_stage3_rules.py` | **Loosely analogous** (partly bespoke) | Coarse-to-fine cascade principle (Bilal & Hanif, 2025; Viola & Jones, 2001) | "Informed by the coarse-to-fine cascade principle." **Do not** claim a proven/standard cascade detector was implemented. |
| **Feature quarantine** (exclude high-missingness features) | `silver_preeda.py` (`quarantine_features_by_missingness`, `compute_missingness_percentage`) | **Engineering practice / heuristic** | Background support: missing-data mechanisms (Rubin, 1976) | Treat as a documented data-governance choice; threshold is configurable/heuristic. Rubin is background only. |
| **Notebook-first orchestration** (one notebook per medallion stage; shared bootstrap) | `notebooks/**`, `utils/core/` (config_loader, notebook_context, paths) | **Engineering practice** | Medallion pattern (Databricks docs) + comparable repos in §6 | Describe as an implemented pattern; cite Databricks medallion docs for the pattern, GitHub repos as implementation comparisons. |
| **Configuration-driven stage behavior** (`configs/stages/*.yaml`) | `configs/stages/`, `utils/core/config_loader` | **Engineering practice** | None required | Describe as a software-engineering choice; no academic citation needed. |
| **Percentile threshold calibration** (k-th percentile of *training* scores) | Gold 02/03 (`np.percentile`) | **Heuristic** (leakage-safe) | Quantile/percentile thresholding (general) | Note it is a training-only, operating-point-sensitive heuristic. |
| **Metric-contract / replay validation** (JSON metric contracts; reload-and-rescore within tolerance) | `gold_cascade_validation_contracts.py`, Gold 06A | **Engineering practice** | Reproducible-evaluation/serialization (joblib, scikit-learn persistence — implementation references) | Describe as a reproducibility/auditability practice; cite tool docs, not a method paper. |

---

## 8. Method-to-Reference Map

Condensed cross-reference. "In paper?" = belongs in the Task 3 reference list. "Tech-docs only?" = keep for documentation, not the paper.

| Method / codebase area | Project file / technical reference | Supporting source | Source category | How the source should be used | In paper? | Tech-docs only? | Caution / overclaiming risk |
|---|---|---|---|---|---|---|---|
| Isolation Forest (baseline) | `gold_baseline_modeling.py`; `pump__gold__baseline_summary.json` | Liu, Ting & Zhou (2008); scikit-learn IsolationForest | Academic + official docs | Method source + API doc | ✓ | — | Not "optimal/SOTA." |
| Two-stage cascade IF | `gold_cascade_modeling.py`; `pump__gold__model_comparison.csv` | Bilal & Hanif (2025); Viola & Jones (2001) | Academic (analogy) | "Informed by" cascade principle | Bilal&Hanif ✓ / Viola&Jones ✗ | Viola&Jones | Partly bespoke — analogy only. |
| Stage 3 rule/profile confirmation | `gold_cascade_stage3_rules.py` | Montgomery (2013) | Academic (analogy) | Control-limit analogy | ✓ | — | "Analogous to," not "SPC control charts." |
| Robust scaling | `gold_preprocessing.py` (`make_scaler`) | scikit-learn RobustScaler | Official docs | API + robust-stats rationale | ✓ | — | — |
| Episode-aware chronological split + leakage control | `gold_preprocessing.py` (`build_episode_based_split_mask`, `apply_imputation`) | Kaufman et al. (2012); scikit-learn cross-validation | Academic + official docs | Leakage rationale | Kaufman ✓ | scikit-learn CV | "Prevents common leakage paths," not "eliminates." |
| CV feature selection (Stage 2) | `gold_preprocessing.py` (`choose_stage2_features_from_training_stability`) | *(project-defined heuristic)* | Heuristic | Describe as project-defined | ✗ | ✓ | Don't over-cite a heuristic. |
| Precision / recall / F1 | `gold_modeling_common.py`; Gold 02/04 | scikit-learn `precision_recall_fscore_support` | Official docs | Metric definitions | ✓ | — | — |
| ROC-AUC vs PR-AUC under imbalance | `gold_modeling_common.py` (`roc_auc`, `average_precision`); `pump__gold__baseline_summary.json` | Saito & Rehmsmeier (2015); Davis & Goadrich (2006) | Academic | Imbalance metric choice | Saito ✓ / Davis ✗ | Davis&Goadrich | Never ROC-AUC alone as success. |
| McNemar paired significance | `EDA_Notebook_Pump_Gold_04_Comparison.ipynb`; `pump__gold__statistical_test_summary.json` | Dietterich (1998); statsmodels McNemar | Academic + official docs | Test appropriateness | ✓ | — | Large-N caveat. |
| Missingness quarantine | `silver_preeda.py` (`quarantine_features_by_missingness`) | Rubin (1976) | Academic (background) | Governance-choice support | ✗ | ✓ (background) | Not a likelihood-theory application. |
| Medallion (Bronze/Silver/Gold) | `technical_reference/00_project_manual/medallion_handoff_map.md`; `configs/stages/*.yaml` | Databricks medallion docs | Industry/vendor docs | Pattern reference | ✓ | — | Not an enterprise lakehouse platform. |
| Truth-hash lineage | `utils/core/truths.py` | *(engineering practice; Git/DVC as impl. refs)* | Engineering practice | Auditability/reproducibility framing | ✗ | ✓ | No academic method claim. |
| Saved-model replay / persistence | `gold_cascade_validation_contracts.py` (Gold 06A); `.joblib` models | joblib; scikit-learn model persistence | Official docs | Serialization/replay | ✗ | ✓ | — |
| Diagnostic EDA (PCA, correlation clustering) | `EDA_Notebook_Pump_Silver_02b_EDA_v2.ipynb` | scikit-learn PCA; AgglomerativeClustering | Official docs | Diagnostic only | ✗ | ✓ | Diagnostic — no predictive claim. |
| Lead-time / early-warning | `stage3_improved__detection_summary.json` (Gold 05/06B) | Lei et al. (2018) | Academic/industry | PdM framing | ✓ | — | Observed in evaluated sequence only. |
| Persistence to PostgreSQL | `medallion_sql_writers.py`; `sql/` | PostgreSQL docs | Official docs | Tooling listing | ✗ | ✓ | Supporting infra. |
| Synthetic generator + Kafka simulation | `utils/synthetic/generator/`, `utils/synthetic/pipeline/` | Apache Kafka docs; Confluent client; NotAndex/kafka_iot_sim | Official docs + impl. ref | Supporting infra only | Confluent ✓ / Kafka ✗ | Kafka, repo | **Not** the report's analytical dataset. |
| Parquet staged outputs | `data/**/*.parquet` | Apache Parquet docs | Official docs | Tooling listing | ✗ | ✓ | Supporting infra. |
| Dataset | `pump__silver_subsets__summary.json` (abnormal source = 7) | nphantawee, *Pump Sensor Data* (Kaggle) | Dataset | Data source | ✓ | — | Real public/historical dataset — **never** "synthetic." |

---

## 9. Sources Not Used as Academic Support

Per the source-validation map, the following should **not** appear in the formal Task 3 report.

**Background-only (read for understanding; cite sparingly or not at all):**

- **Rubin (1976)** — cite only if the report explicitly discusses missingness mechanisms (MCAR/MAR/MNAR); otherwise background. The current report does not, so it stays out of the paper.
- **Davis & Goadrich (2006)** and **Liu, Ting & Zhou (2012)** — legitimate academic sources kept for technical completeness, but redundant for the report (one imbalance anchor and one IF anchor already suffice). Background/technical-docs only.

**Do not cite academically (weak / non-authoritative / off-target):**

- Blog and tutorial posts that surfaced during research — e.g., Medium, Towards Data Science, DEV.to, GeeksforGeeks, Baeldung — on medallion architecture, McNemar's test, ACID, or Parquet. These are non-authoritative and risk an academic-integrity flag. **Use the first-party docs or peer-reviewed papers they themselves cite**, not the posts.
- "How-to" pages that merely restate library syntax (e.g., a GeeksforGeeks McNemar tutorial) — replace with the **statsmodels** documentation.
- Any **GitHub repository** as evidence that a method is *valid*. GitHub belongs only in §6 as an **implementation comparison**.
- **Tam & Nguyen (1995)** — unverifiable in the prior research pass; removed from the report and replaced by Montgomery (2013). Do not reintroduce.

---

## 10. Final Notes and Usage Guidance

**Dataset (synthetic vs. real).** The report's analytical dataset is the **real, public Kaggle "Pump Sensor Data" set (220,320 rows; 7 broken observations)**. The codebase also contains a **synthetic generator** (`utils/synthetic/`) and a **Kafka simulation** (`utils/synthetic/pipeline/`) as *separate supporting infrastructure*. **Do not conflate the two**, and **never describe the report dataset as synthetic.** The correct limitation language is *"a single public/historical dataset, not production-validated."*

**Production-readiness.** The system is framed as a **maintenance triage aid / review aid**, explicitly **not** a production-ready predictive-maintenance system. Production use would require validation on additional real telemetry, cost-based threshold tuning, and live monitoring/drift checks (future work).

**Real-world industrial validation.** Absolute metrics (e.g., baseline precision ≈ 0.0038; Stage 3 improved ≈ 6,594 alerts, ~78.9% reduction; strict precision ≈ 0.443) come from a **single historical dataset with one chronological split** and **no cross-asset or k-fold validation**. Do not generalize them to real pumps in production.

**Cascade claims.** The cascade is **partly bespoke**. Describe it as *"informed by the coarse-to-fine cascade principle"* / *"analogous to coarse-to-fine detection"* (Bilal & Hanif, 2025; Viola & Jones, 2001) — **not** as a proven or replicated production cascade detector.

**Stage 3 bounds.** Stage 3 reference-bound confirmation is *"analogous to control-limit monitoring in statistical process control"* (Montgomery, 2013) — **not** formal SPC control charts.

**Leakage claims.** Say *"helped prevent common leakage paths"* or *"reduced the risk of leakage"* (Kaufman et al., 2012) — **never** *"eliminated leakage."*

**Statistical vs. practical significance.** McNemar significance (stat ≈ 22,609.57, p ≈ 0 across ≈ 83,889 paired test rows) reflects a very large N and **does not by itself prove operational adequacy**; pair it with the still-low absolute precision and the FPR reduction (0.371 → 0.078). Report practical significance (alert-volume reduction; precision/recall operating dial) separately and honestly.

**GitHub examples.** Every repository in §6 is an **implementation reference only**. It demonstrates that a comparable pattern exists in open source; it is **not** academic support and must not be cited as evidence a method is valid, nor presented as a source the project was built from.

**Academic-integrity reminder.** Throughout this archive and any technical write-up, use precedent/comparison language — *related reference, comparable implementation, official documentation, methodological precedent, background resource, found-after-implementation comparison, informed by, analogous to*. The project's own files do not document the origin of its design ideas, so never state that the project derived, copied, or was inspired by any source listed here.
