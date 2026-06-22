# Utility Module Reference: `utils/medallion/silver/silver_eda_groups.py`

## Module Purpose

This module builds sensor grouping and grouped summary structures for Silver exploratory analysis.

## Pipeline Role

- Stage support: Silver
- Primary responsibility: This module builds sensor grouping and grouped summary structures for Silver exploratory analysis.

## Primary Consumers

`EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`, `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `find` | Return the representative element for ``x`` using path compression. | short |
| `union` | Merge the disjoint-set components containing ``a`` and ``b``. | short |
| `build_normal_only_correlation_pairs` | Build correlation matrix and pair table for rows in the target state. | deep |
| `build_sensor_group_map_from_correlation` | Build connected-component sensor groups from absolute correlations. | deep |
| `build_fault_propagation_pairings_from_strong_relationships` | Filter correlation pairs into a strong-relationship pairing table. | deep |

## Configuration Dependencies

- Project root, resolved path mappings, and artifact directory configuration.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `find` | `parent, x` | Return the representative element for ``x`` using path compression. |
| `union` | `parent, a, b` | Merge the disjoint-set components containing ``a`` and ``b``. |
| `build_normal_only_correlation_pairs` | `dataframe, *, feature_columns, state_column, target_state` | Build correlation matrix and pair table for rows in the target state. |
| `build_sensor_group_map_from_correlation` | `correlation_matrix, *, min_abs_corr_for_group` | Build connected-component sensor groups from absolute correlations. |
| `build_fault_propagation_pairings_from_strong_relationships` | `correlation_pairs_df, *, min_abs_corr` | Filter correlation pairs into a strong-relationship pairing table. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Not determined from available source

## Failure Behavior

- Source raises `KeyError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Gold_01_PreProcessing`, `EDA_Notebook_Pump_Gold_03a_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_03b_Cascade_Modeling`, `EDA_Notebook_Pump_Gold_05_Anomaly_Detection`, `EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Module Importance

This module matters because Silver notebooks depend on repeatable profiling and EDA helpers before the modeling-ready Gold layer is built.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
