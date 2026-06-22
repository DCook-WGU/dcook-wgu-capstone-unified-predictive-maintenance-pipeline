# Utility Module Reference: `utils/medallion/silver/silver_eda_artifacts.py`

## Module Purpose

This module writes or organizes Silver EDA artifacts such as summaries, profiles, and export paths.

## Pipeline Role

- Stage support: Silver
- Primary responsibility: This module writes or organizes Silver EDA artifacts such as summaries, profiles, and export paths.

## Primary Consumers

`EDA_Notebook_Pump_Gold_01_PreProcessing`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `save_eda_table_artifact` | Save a dataframe artifact and return its full path as text. | deep |
| `save_eda_json_artifact` | Save a JSON artifact payload and return its full path as text. | deep |
| `save_episode_status_counts_json` | Save episode status counts as JSON records and return the output path. | deep |
| `build_silver_eda_artifact_index` | Build a compact artifact index payload for Silver EDA outputs. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `save_eda_table_artifact` | `dataframe, *, output_dir, file_name` | Save a dataframe artifact and return its full path as text. |
| `save_eda_json_artifact` | `payload, *, output_path` | Save a JSON artifact payload and return its full path as text. |
| `save_episode_status_counts_json` | `episode_status_counts_df, *, output_path` | Save episode status counts as JSON records and return the output path. |
| `build_silver_eda_artifact_index` | `*, artifact_paths, summary_payload` | Build a compact artifact index payload for Silver EDA outputs. |

## Side Effects

- Source includes directory creation; helpers can create configured output directories.

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.

## Failure Behavior

- Not determined from available source

## Downstream Usage

`EDA_Notebook_Pump_Gold_01_PreProcessing`

## Module Importance

This module matters because Silver notebooks depend on repeatable profiling and EDA helpers before the modeling-ready Gold layer is built.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
