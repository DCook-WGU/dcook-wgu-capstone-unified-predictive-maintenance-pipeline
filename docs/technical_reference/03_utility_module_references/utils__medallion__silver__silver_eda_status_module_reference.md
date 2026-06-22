# Utility Module Reference: `utils/medallion/silver/silver_eda_status.py`

## Module Purpose

This module normalizes and summarizes status/state fields for Silver EDA checks.

## Pipeline Role

- Stage support: Silver
- Primary responsibility: This module normalizes and summarizes status/state fields for Silver EDA checks.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `resolve_state_column_from_truth` | Resolve the label/status source column from parent Silver truth metadata. | deep |
| `build_state_col_synth` | Build a normalized synthetic state column from the status column. | deep |
| `get_episode_status_state_stats` | Build episode-aware status and state summary payloads. | deep |
| `build_episode_status_payload_and_tables` | Build the episode status payload and dataframe tables used by Silver EDA. | deep |
| `build_status_distribution_tables` | Build a row-level status distribution table. | deep |
| `pull_episode_status_state_stats_from_truth` | Load saved episode-state summary fields from a Silver EDA truth record. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.
- PostgreSQL connection settings, schema names, and table names provided by config or notebook context.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `resolve_state_column_from_truth` | `truth_record, *, fallback_status_column` | Resolve the label/status source column from parent Silver truth metadata. |
| `build_state_col_synth` | `dataframe, *, status_column, state_map, output_column` | Build a normalized synthetic state column from the status column. |
| `get_episode_status_state_stats` | `dataframe, status_column, episode_column, state_order, include_null_episode, state_map, lowercase_states, strip_states, percent_suffix` | Build episode-aware status and state summary payloads. |
| `build_episode_status_payload_and_tables` | `dataframe, *, status_column, episode_column, state_map` | Build the episode status payload and dataframe tables used by Silver EDA. |
| `build_status_distribution_tables` | `dataframe, *, status_column` | Build a row-level status distribution table. |
| `pull_episode_status_state_stats_from_truth` | `truth_record` | Load saved episode-state summary fields from a Silver EDA truth record. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `KeyError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because Silver notebooks depend on repeatable profiling and EDA helpers before the modeling-ready Gold layer is built.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
