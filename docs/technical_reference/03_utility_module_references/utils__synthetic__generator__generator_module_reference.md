# Utility Module Reference: `utils/synthetic/generator/generator.py`

## Module Purpose

This module generates synthetic pump telemetry with configurable profiles, correlations, missingness, and fault truth metadata.

## Pipeline Role

- Stage support: Synthetic generator
- Primary responsibility: This module generates synthetic pump telemetry with configurable profiles, correlations, missingness, and fault truth metadata.

## Primary Consumers

`EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `EpisodeSpec` | Row-count and fault settings for one generated synthetic episode. | short |
| `SyntheticGenerator` | Generate synthetic pump telemetry from Silver EDA profile artifacts. | short |
| `_as_float` | Coerce optional tuning values to float with a default fallback. | short |
| `_as_int` | Coerce optional tuning values to int with a default fallback. | short |
| `_as_object_dict` | Return a string-keyed dict when a tuning value is mapping-like. | short |
| `_as_object_list` | Return a list for list/tuple tuning values and [] otherwise. | short |
| `_pair_key` | Build a stable unordered key for a sensor pair. | short |
| `SyntheticGenerator.__init__` | Build lookup tables and random state used across generation calls. | deep |
| `SyntheticGenerator._sample_series` | Sample one bounded sensor series using its profile distribution family. | deep |
| `SyntheticGenerator._apply_group_driver` | Add a shared latent movement to sensors in one group in place. | deep |
| `SyntheticGenerator._build_group_correlation_matrix` | Build a repaired correlation matrix for the sensors in one group. | deep |
| `SyntheticGenerator._apply_group_correlated_residuals` | Overlay multivariate residual movement using a group correlation matrix. | deep |
| `SyntheticGenerator._apply_top_pairwise_overlay` | Reinforce the strongest configured pairwise correlations in place. | deep |
| `SyntheticGenerator._apply_named_cluster_overlay` | Apply a shared latent signal across configured hotspot clusters. | deep |
| `SyntheticGenerator._normalize_cluster_list` | Clean configured clusters to known, de-duplicated sensor names. | short |
| `SyntheticGenerator._derive_hotspot_clusters_from_corr` | Derive hotspot clusters from high-correlation sensor pairs. | deep |
| `SyntheticGenerator._resolve_hotspot_clusters` | Prefer configured hotspot clusters, otherwise derive them from correlations. | short |
| `SyntheticGenerator._get_corr_tuning_section` | Return one named correlation tuning section as a plain dict. | deep |
| `SyntheticGenerator._get_corr_tuning_block` | Merge a correlation tuning block with its default values. | short |
| `SyntheticGenerator._get_chain_family_split_threshold` | Threshold used to split 3-sensor clusters into strong vs weak chain families. | deep |
| `SyntheticGenerator._cluster_avg_abs_corr` | Average absolute pairwise correlation inside a cluster, using self.corr. | deep |
| `SyntheticGenerator._classify_cluster_family` | Classify hotspot clusters into family types. | short |

## Configuration Dependencies

- Resolved pipeline configuration dictionaries and YAML-derived stage settings.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `_as_float` | `value, default` | Coerce optional tuning values to float with a default fallback. |
| `_as_int` | `value, default` | Coerce optional tuning values to int with a default fallback. |
| `_as_object_dict` | `value` | Return a string-keyed dict when a tuning value is mapping-like. |
| `_as_object_list` | `value` | Return a list for list/tuple tuning values and [] otherwise. |
| `_pair_key` | `left, right` | Build a stable unordered key for a sensor pair. |
| `SyntheticGenerator.__init__` | `self, *, normal_profiles, abnormal_profiles, recovery_profiles, correlation_pairs_dataframe, group_map_dataframe, fault_pairings_dataframe, correlation_hotspot_clusters, correlation_cluster_derivation, fault_excluded_sensors, correlation_tuning, random_seed, missingness_spec, state_calibration_targets, mean_within_k_std, std_ratio_bounds` | Build lookup tables and random state used across generation calls. |
| `SyntheticGenerator._sample_series` | `self, profile, n, smoothing` | Sample one bounded sensor series using its profile distribution family. |
| `SyntheticGenerator._apply_group_driver` | `self, dataframe, group_name, profiles, strength` | Add a shared latent movement to sensors in one group in place. |
| `SyntheticGenerator._build_group_correlation_matrix` | `self, group_name, profiles, min_corr, shrinkage` | Build a repaired correlation matrix for the sensors in one group. |
| `SyntheticGenerator._apply_group_correlated_residuals` | `self, dataframe, group_name, profiles, strength, smooth_alpha` | Overlay multivariate residual movement using a group correlation matrix. |
| `SyntheticGenerator._apply_top_pairwise_overlay` | `self, dataframe, profiles, *, min_abs_corr, top_n, strength, smooth_alpha` | Reinforce the strongest configured pairwise correlations in place. |
| `SyntheticGenerator._apply_named_cluster_overlay` | `self, dataframe, profiles, *, clusters, strength, smooth_alpha` | Apply a shared latent signal across configured hotspot clusters. |
| `SyntheticGenerator._normalize_cluster_list` | `self, clusters` | Clean configured clusters to known, de-duplicated sensor names. |
| `SyntheticGenerator._derive_hotspot_clusters_from_corr` | `self` | Derive hotspot clusters from high-correlation sensor pairs. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Artifact/file-system: Source references file paths, artifact paths, or directories.
- Lineage/truth metadata: Source references truth metadata or parent/child truth handling.

## Failure Behavior

- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

`EDA_Notebook_Pump_Silver_02a_EDA_Building_Subsets_v3`, `EDA_Notebook_Pump_Silver_02b_EDA_v2`

## Module Importance

This module matters because the synthetic subsystem depends on repeatable staging, truth handling, streaming handoff, or final alignment behavior to make controlled pump telemetry tests explainable.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
