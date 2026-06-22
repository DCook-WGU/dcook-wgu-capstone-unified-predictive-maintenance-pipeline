# utils/synthetic/generator/generator.py Deep Utility Function Reference

## Purpose of This Deep Reference

This document covers selected high-value functions from `generator.py` that need deeper explanation than the 071d module-level reference. The selected methods define the synthetic generator's episode orchestration, row-level truth stamping, fault injection, correlation repair, cluster behavior, variance control, and buildup transition behavior.

## Source Grounding

Sources used:

- `utils/synthetic/generator/generator.py`
- `technical_reference/03_utility_module_references/071e_scope_plan.md`
- `technical_reference/03_utility_module_references/utils__synthetic__generator__generator_module_reference.md`
- `function_inventory.json`
- `technical_reference/01_notebook_workflow_references/`
- `technical_reference/00_project_manual/`
- Documentation standards under `docs_agent_pack/00_STANDARDS/`

The active utility source file is the source of truth.

## Functions and Classes Covered

| Function / Class | Technical Role | Primary Pipeline Context |
| ---------------- | -------------- | ------------------------ |
| `SyntheticGenerator.generate_episode` | Orchestrates one complete synthetic telemetry episode | Synthetic generator stage |
| `SyntheticGenerator._build_episode_truth_columns` | Stamps row-level fault, phase, onset, and lead-time metadata | Synthetic truth construction |
| `SyntheticGenerator._build_group_correlation_matrix` | Builds repaired sensor-group correlation matrices | Correlation repair |
| `SyntheticGenerator._apply_group_correlated_residuals` | Adds group-level correlated residual movement to sensor values | Correlation repair |
| `SyntheticGenerator._apply_anchor_cluster_generation` | Regenerates cluster members from an anchor sensor path | Hotspot cluster correlation behavior |
| `SyntheticGenerator._apply_sensor_variance_floor` | Adds bounded noise when a generated block has too little variance | Variance control |
| `SyntheticGenerator._apply_buildup_transition_noise` | Shapes buildup-window transition behavior before failure | Phase-specific fault buildup |
| `SyntheticGenerator._inject_fault` | Applies configured fault shapes to sensor vectors | Fault injection |

## Module-Level Technical Context

`generator.py` converts Silver EDA profile artifacts and correlation inputs into synthetic pump telemetry. The generator samples normal sensor behavior, applies group and cluster correlation structure, injects primary and propagated fault behavior, restores phase-level calibration, stamps truth metadata, and optionally replays missingness.

The assigned methods are central because they decide whether synthetic rows are plausible enough for downstream pipeline testing and whether generated truth fields can support evaluation. They are not standalone reporting helpers; they are the internal mechanics that make the synthetic episode output interpretable.

## Deep Function References

### `SyntheticGenerator.generate_episode`

#### Functional Purpose

`generate_episode` creates one synthetic telemetry episode from an `EpisodeSpec`. It starts from a normal generated batch, assigns phase and stream-state windows, applies primary and secondary fault behavior, repairs correlation and variance structure by phase, stamps truth metadata, and optionally applies configured missingness replay.

#### Pipeline Context

This method is the public episode-generation entry point for the synthetic generator. It supports the synthetic generator stage and downstream synthetic pipeline handoffs that depend on a row-aligned dataframe containing sensor columns, phase labels, stream state, and truth metadata.

Direct final-facing notebook consumer context is Not determined from available source.

#### Inputs and Assumptions

Important inputs include:

- `spec`, an `EpisodeSpec` containing the primary sensor, primary fault type, magnitude, and phase row counts.
- `episode_id`, which is stamped into truth metadata when provided.
- `observable_zscore_threshold` and `observable_min_consecutive`, which control observable onset detection in the truth-stamping step.

The method assumes the generator has already been constructed with normal, abnormal, and recovery profiles; sensor groups; correlation pairs; fault-pair propagation links; optional missingness settings; optional state calibration targets; and an RNG initialized from the configured random seed.

The primary sensor must exist in `self.sensors` and must not be in `fault_excluded_sensors`. Negative phase counts are clamped to zero. A fault episode is determined by `spec.failure > 0`; when total generated length would otherwise be zero, the method creates a one-row normal episode.

#### Outputs and Return Contract

The method returns a pandas dataframe with one column per generated sensor plus generated state/truth columns. Confirmed output columns include `phase`, `stream_state`, and the `meta__...` truth columns added by `_build_episode_truth_columns`. If missingness replay is configured, the returned dataframe may contain missing sensor values after truth stamping.

#### Side Effects

The method mutates local dataframe windows during generation and uses the generator RNG for stochastic sampling, fault injection, residual overlays, cluster generation, and optional missingness replay. It does not write files, SQL rows, artifacts, or ledgers from available source.

#### Failure Behavior and Guardrails

Confirmed guardrails include:

- Raises `ValueError` when `spec.primary_sensor` is not known to the generator.
- Raises `ValueError` when the primary sensor is configured as fault-excluded.
- Skips fault-specific behavior when `spec.failure` is not positive.
- Falls back to a one-row normal episode when all requested phase lengths sum to zero or less after clamping.
- Falls back to `variance_burst` for unsupported propagated secondary fault recommendations.

#### Truth and Reproducibility Role

The method defines the generated episode's truth windows by computing normal-before, buildup, failure, recovery, and normal-after boundaries. It uses `phase` values of `normal`, `buildup`, `abnormal`, and `recovery`, while `stream_state` exposes `normal`, `broken`, and `recovering` states. Buildup rows keep `stream_state="normal"` even when `phase="buildup"`, which separates hidden fault buildup from the surfaced broken state.

Reproducibility depends on the generator's `np.random.default_rng(random_seed)` state. Calling the method advances that RNG through normal sampling, residual generation, fault injection, transition noise, and optional missingness.

#### Why This Function / Class Matters

This method is the synthetic generator's episode contract. Downstream testing depends on the generated dataframe remaining row-aligned across sensor values, phase labels, fault metadata, and optional missingness. If this orchestration changes, the downstream Bronze/Silver/Gold pipeline may still run, but the meaning of synthetic anomaly labels and phase windows would change.

#### Verification Method

- Confirm returned row count equals the sum of clamped phase lengths, or one row for an all-zero request.
- Confirm `phase` counts match `normal_before`, `buildup`, `failure`, `recovery`, and `normal_after`.
- Confirm `stream_state="broken"` appears only in the failure window.
- Confirm fault episodes contain truth metadata for fault onset and failure rows.
- Confirm fault-excluded primary sensors raise `ValueError`.
- Confirm repeated runs are reproducible only when generator construction and call order use the same random seed.

### `SyntheticGenerator._build_episode_truth_columns`

#### Functional Purpose

`_build_episode_truth_columns` stamps row-level truth metadata onto a generated episode dataframe. It records episode identity, fault-episode status, phase truth, primary fault metadata, observable primary-sensor z-score behavior, generator truth onset, failure row, observable onset row, buildup progress, and lead-time helper fields.

#### Pipeline Context

This method supports synthetic truth construction after generated sensor values and phase labels have been finalized for an episode. It is called by `generate_episode` before optional missingness replay.

#### Inputs and Assumptions

Important inputs include:

- A generated dataframe that already contains the primary sensor column and a `phase` column.
- `spec`, which supplies primary sensor, fault type, and magnitude metadata.
- `episode_id`, which may be null.
- `buildup_start_idx` and `failure_start_idx`, which define generator truth onset and failure start rows.
- `is_fault_episode`, which controls whether fault-onset and failure rows are stamped.
- Observable onset parameters based on primary-sensor z-score threshold and consecutive-row count.

The method assumes `self.normal[spec.primary_sensor]` exists and provides the normal profile mean and standard deviation used for z-score calculations.

#### Outputs and Return Contract

The method returns a copy of the input dataframe with additional truth columns. Confirmed columns include:

- `meta__episode_id`
- `meta__is_fault_episode`
- `meta__phase_truth`
- `meta__primary_sensor`
- `meta__primary_fault_type`
- `meta__primary_magnitude`
- `meta__primary_sensor_abs_zscore`
- `meta__primary_sensor_threshold_crossed`
- `meta__fault_onset_truth_flag`
- `meta__fault_onset_truth_row`
- `meta__failure_truth_flag`
- `meta__failure_truth_row`
- `meta__observable_onset_flag`
- `meta__observable_onset_row`
- `meta__buildup_progress`
- `meta__rows_until_failure`
- `meta__rows_since_truth_onset`
- `meta__rows_since_observable_onset`

#### Side Effects

The method returns a copied dataframe and does not mutate the caller's dataframe object from available source. No file, SQL, artifact, or ledger side effects are confirmed.

#### Failure Behavior and Guardrails

No explicit validation exceptions are defined inside this method. Missing required columns or missing primary sensor profiles would fail through normal pandas or dictionary access behavior. Observable onset is only searched for fault episodes and only from `buildup_start_idx` onward.

#### Truth and Reproducibility Role

This method is the row-level truth anchor for synthetic episodes. It distinguishes generator truth onset from the first observable threshold crossing, which allows later analysis to compare true fault buildup timing against model-detectable signal timing. Lead-time helper columns preserve rows-until-failure and rows-since-onset relationships directly in the generated dataframe.

#### Why This Function / Class Matters

Synthetic anomaly evaluation is only meaningful if the generated labels and timing metadata align with the generated sensor values. This method keeps fault type, phase truth, onset, failure, observable onset, and lead-time metadata in the same row-aligned output frame used by downstream stages.

#### Verification Method

- Confirm all expected `meta__` truth columns are present.
- Confirm `meta__fault_onset_truth_flag` is true at `buildup_start_idx` for fault episodes.
- Confirm `meta__failure_truth_flag` is true at `failure_start_idx` for fault episodes.
- Confirm observable onset is null when no consecutive threshold crossing is found.
- Confirm `meta__buildup_progress` is nonzero only in the buildup window.
- Confirm non-fault episodes keep onset and failure row fields null.

### `SyntheticGenerator._build_group_correlation_matrix`

#### Functional Purpose

`_build_group_correlation_matrix` builds a repaired correlation matrix for sensors in one configured group. It uses pairwise correlations from `self.corr`, applies a minimum-correlation cutoff, shrinks the matrix toward identity, repairs non-positive-semidefinite behavior by clipping eigenvalues, renormalizes the diagonal, clips correlations to a stable range, and returns the sensor ordering with the repaired matrix.

#### Pipeline Context

This method supports group-level correlation repair for generated sensor values. It is called by `_apply_group_correlated_residuals`.

#### Inputs and Assumptions

Important inputs include the `group_name`, the profile dictionary for the current phase or state, `min_corr`, and `shrinkage`.

The method assumes the generator was initialized with `group_to_sensors`, `self.sensors`, and pairwise correlations in `self.corr`. It only includes sensors that are in the requested group, present in the profile dictionary, and known to the generator.

#### Outputs and Return Contract

The method returns a tuple containing:

- The ordered list of sensors used in the group matrix.
- A NumPy correlation matrix whose shape is `(len(sensors), len(sensors))`.

For groups with fewer than two eligible sensors, it returns the sensor list and an identity matrix of matching size.

#### Side Effects

No mutation, file, SQL, artifact, or ledger side effects are confirmed. The method constructs and returns new matrix objects.

#### Failure Behavior and Guardrails

The source does not define custom exceptions. Guardrails include returning an identity matrix for undersized groups, zeroing pairwise correlations below `min_corr`, shrinking toward identity, eigenvalue clipping, and diagonal renormalization. NumPy linear algebra failures are not explicitly caught.

#### Truth and Reproducibility Role

This method does not stamp truth metadata. Its reproducibility role is structural: it provides a deterministic repaired matrix for a given group, profile set, correlation lookup, cutoff, and shrinkage value.

#### Why This Function / Class Matters

Synthetic telemetry should preserve important multivariate relationships from the Silver EDA inputs. This method provides the matrix used to generate correlated residual overlays while protecting the generator from unstable correlation matrices.

#### Verification Method

- Confirm returned matrix shape matches the returned sensor list length.
- Confirm diagonal values are `1.0`.
- Confirm all matrix values are within the clipped range.
- Confirm the matrix is symmetric.
- Confirm eigenvalues are non-negative within numerical tolerance.
- Confirm groups with fewer than two eligible sensors return identity matrices.

### `SyntheticGenerator._apply_group_correlated_residuals`

#### Functional Purpose

`_apply_group_correlated_residuals` overlays correlated residual movement onto sensors in one group. It samples multivariate normal residuals using the repaired group correlation matrix, smooths each residual series through time, scales residuals by each sensor's profile standard deviation and the requested strength, clips values to profile bounds, and writes the adjusted values back to the dataframe.

#### Pipeline Context

This method supports correlation repair during normal batch generation and phase-specific episode adjustments. It is used by `generate_normal_batch` and by `generate_episode` for abnormal-window group boosting.

#### Inputs and Assumptions

Important inputs include the dataframe to mutate, `group_name`, phase/state profiles, residual `strength`, and `smooth_alpha`.

The method assumes the dataframe contains group sensor columns and that matching profiles exist for the sensors returned by `_build_group_correlation_matrix`. It exits without changes for groups with fewer than two eligible sensors or empty dataframes.

#### Outputs and Return Contract

The method returns `None`. Its output is the in-place modification of the dataframe columns for eligible group sensors.

#### Side Effects

Confirmed side effects are in-place dataframe column updates and stochastic residual generation through the generator RNG. No file, SQL, artifact, or ledger side effects are confirmed.

#### Failure Behavior and Guardrails

The method returns without action for empty dataframes or insufficient eligible sensors. Sensor values are coerced to numeric, adjusted, and clipped to profile percentile and absolute bounds. NumPy random or numeric failures are not explicitly caught.

#### Truth and Reproducibility Role

This method does not create truth labels. It affects reproducibility through RNG state and through deterministic use of the repaired correlation matrix for a given call order. Because it mutates sensor values, it indirectly affects observable onset calculations that happen later in the episode.

#### Why This Function / Class Matters

Without group-level residual correlation, synthetic sensors can look independent even when source profiles indicate correlated pump behavior. This method helps generated telemetry preserve group relationships that downstream anomaly detection models may rely on.

#### Verification Method

- Compare group sensor correlations before and after the residual overlay.
- Confirm row count and index alignment are unchanged.
- Confirm adjusted values remain within profile bounds.
- Confirm no changes occur for empty dataframes or groups with fewer than two eligible sensors.
- Confirm repeated runs with the same seed and call order produce the same adjusted values.

### `SyntheticGenerator._apply_anchor_cluster_generation`

#### Functional Purpose

`_apply_anchor_cluster_generation` strengthens configured hotspot clusters by selecting an anchor sensor and generating other cluster members from the anchor's standardized path plus residual noise. It blends generated values with existing member values and clips the result to each member sensor's profile bounds.

#### Pipeline Context

This method supports hotspot cluster correlation behavior in normal, buildup, and recovery windows. It is called from normal batch generation and from phase-specific windows inside `generate_episode` when correlation hotspot clusters are configured or derived.

#### Inputs and Assumptions

Important inputs include the dataframe to mutate, phase/state profiles, configured clusters, `blend`, `min_abs_corr`, `residual_floor`, and `smooth_alpha`.

The method assumes clusters contain sensor names, that at least two cluster sensors are present in both the dataframe and profiles, and that the selected anchor has at least three finite values. Pairwise correlations are read from `self.corr`. Members whose anchor correlation magnitude is below `min_abs_corr` are skipped.

#### Outputs and Return Contract

The method returns `None`. Its output is the in-place modification of eligible non-anchor cluster member columns.

#### Side Effects

Confirmed side effects are in-place dataframe updates and stochastic residual generation through the generator RNG. The method does not write files, SQL rows, artifacts, or ledgers.

#### Failure Behavior and Guardrails

The method returns without changes for empty dataframes. It skips undersized clusters, clusters without a valid anchor, anchors with fewer than three finite values, and member sensors below the minimum correlation threshold. Generated member values are clipped to profile bounds.

#### Truth and Reproducibility Role

This method does not create truth metadata. It affects reproducibility through the RNG stream and through the configured or derived hotspot cluster list. Its value transformations can influence later observable truth fields because `_build_episode_truth_columns` runs after phase value generation.

#### Why This Function / Class Matters

Some sensor relationships are cluster-shaped rather than only group-wide or pairwise. Anchor-based cluster generation helps synthetic telemetry carry realistic multivariate structure, especially for groups of sensors that should move together around a dominant anchor path.

#### Verification Method

- Confirm only eligible non-anchor cluster members are modified.
- Confirm member values remain row-aligned with the original dataframe.
- Confirm adjusted member correlations move toward the anchor relationship when the configured correlation is above threshold.
- Confirm values remain within profile bounds.
- Confirm no changes occur for clusters with too few valid sensors.

### `SyntheticGenerator._apply_sensor_variance_floor`

#### Functional Purpose

`_apply_sensor_variance_floor` adds bounded noise to selected rows when a generated sensor block has lower variance than a profile-derived floor. It prevents generated windows from becoming too flat relative to the source profile.

#### Pipeline Context

This method supports variance control in generated normal batches and phase-specific windows inside `generate_episode`. It is applied to normal, buildup, and recovery windows with different floor and maximum-noise settings.

#### Inputs and Assumptions

Important inputs include the dataframe to mutate, row index array `idx`, phase/state profiles, `std_floor_ratio`, and `max_extra_noise_ratio`.

The method assumes the dataframe has sensor columns from `self.sensors` and that the profile dictionary contains matching profile statistics. It only operates on finite values and requires at least three finite values in the selected block for a sensor.

#### Outputs and Return Contract

The method returns `None`. Its output is in-place updates to selected dataframe cells for sensors whose current selected-window standard deviation is below the computed floor.

#### Side Effects

Confirmed side effects are in-place dataframe updates and stochastic noise generation through the generator RNG. No file, SQL, artifact, or ledger side effects are confirmed.

#### Failure Behavior and Guardrails

The method exits without changes when the selected index array is empty. It skips missing sensors, missing profiles, sensors with fewer than three finite selected values, and sensors whose current standard deviation already meets the floor. Added noise is capped by `target_std * max_extra_noise_ratio`, and values are clipped to profile bounds.

#### Truth and Reproducibility Role

This method does not stamp truth metadata. It supports reproducible simulation quality by enforcing a source-profile variance floor for a given RNG seed and call order. Because truth stamping happens after generation, variance-floor adjustments can affect observable z-score threshold behavior.

#### Why This Function / Class Matters

Flatlined synthetic sensors can make model testing easier or less realistic than intended. This method helps preserve sensor movement inside generated windows so downstream anomaly detection sees telemetry with plausible variation.

#### Verification Method

- Compute selected-window standard deviation before and after the method.
- Confirm adjusted sensors had pre-adjustment standard deviation below the configured floor.
- Confirm sensors above the floor remain unchanged.
- Confirm values remain within profile bounds.
- Confirm empty `idx` produces no changes.

### `SyntheticGenerator._apply_buildup_transition_noise`

#### Functional Purpose

`_apply_buildup_transition_noise` creates a length-aware transition for the buildup window before failure. It blends the original sensor path toward a lower-severity fault component, adds shaped transition noise, smooths the result, and clips values to profile bounds.

#### Pipeline Context

This method is called by `generate_episode` for the primary sensor during the buildup phase of fault episodes. It models a pre-failure transition that can be subtle early and stronger later, rather than making buildup rows immediately look like full failure rows.

#### Inputs and Assumptions

Important inputs include the primary sensor value array, normal profile, primary fault type, and magnitude. The method derives behavior from the buildup length, profile standard deviation, configured fault shape, and generator RNG.

The method assumes the input values correspond to the buildup segment for one sensor. It copies the input array before modifying it.

#### Outputs and Return Contract

The method returns a NumPy array of the same length as the input. For zero-length inputs, it returns an empty copied array. Returned values are clipped to the profile's percentile and absolute bounds.

#### Side Effects

The method uses the generator RNG and calls `_inject_fault` on a copied array to create the transition component. It does not mutate the caller's input array from available source and does not perform file, SQL, artifact, or ledger writes.

#### Failure Behavior and Guardrails

No custom exceptions are defined. Guardrails include zero-length return behavior, minimum standard deviation protection, minimum effective magnitude, smoothing, and bounds clipping.

#### Truth and Reproducibility Role

This method does not stamp truth fields, but it shapes the rows between `meta__fault_onset_truth_row` and `meta__failure_truth_row`. Its output affects whether observable onset is detected before the failure window and therefore affects lead-time interpretation.

#### Why This Function / Class Matters

Buildup behavior is the bridge between normal operation and explicit failure. If the buildup transition is too abrupt, the synthetic episode becomes a simple step change; if it is too weak, the truth onset may not produce detectable signal. This method controls that balance.

#### Verification Method

- Confirm output length matches input length.
- Confirm values remain within profile bounds.
- Compare early and late buildup windows to verify the late window has stronger transition influence.
- Confirm zero-length buildup returns without error.
- Confirm repeated runs with the same seed and call order produce the same transition values.

### `SyntheticGenerator._inject_fault`

#### Functional Purpose

`_inject_fault` applies one configured fault shape to a sensor value vector. Supported source-confirmed fault types are `drift_up`, `drift_down`, `spike`, `stuck_constant`, `variance_burst`, `step_shift`, `intermittent_dropout`, and `sawtooth`.

#### Pipeline Context

This method is used by `generate_episode` for primary failure-window faults, propagated secondary faults, and buildup transition construction through `_apply_buildup_transition_noise`.

#### Inputs and Assumptions

Important inputs include the sensor value array, `fault_type`, magnitude, and a `SensorRichProfile` used for standard deviation and bounds. The method assumes the fault type is one of the supported literal values, although unsupported values are not explicitly rejected inside this method.

Fault magnitude is scaled by the profile standard deviation. Several fault types use the generator RNG for spike direction/index selection, step-shift direction, variance burst noise, and intermittent dropout masking.

#### Outputs and Return Contract

The method returns the fault-adjusted NumPy array clipped to the profile's percentile and absolute bounds.

Confirmed fault behavior:

- `drift_up`: linearly increases values over the vector.
- `drift_down`: linearly decreases values over the vector.
- `spike`: changes a small random set of rows by a positive or negative spike.
- `stuck_constant`: sets all values to the local median of the input vector.
- `variance_burst`: adds random normal noise.
- `step_shift`: shifts the full vector up or down.
- `intermittent_dropout`: lowers a random subset of rows toward a profile-derived low value.
- `sawtooth`: adds a repeating sawtooth wave.

#### Side Effects

The method can mutate the passed NumPy array object for some branches because several fault types assign into `values` directly. In `generate_episode`, callers pass copied segment arrays before writing the returned values back to the dataframe. The method also advances the generator RNG for stochastic fault types.

No file, SQL, artifact, or ledger side effects are confirmed.

#### Failure Behavior and Guardrails

No explicit exception is raised for unsupported fault types inside `_inject_fault`; unsupported values pass through to final clipping without a fault-specific transformation. In `generate_episode`, propagated secondary fault recommendations are checked against the supported set and default to `variance_burst` when unsupported.

All outputs are clipped to profile percentile bounds and absolute min/max bounds.

#### Truth and Reproducibility Role

This method does not stamp truth columns directly. It is the value-transformation source behind primary and secondary fault behavior, so it must stay aligned with truth metadata created later by `_build_episode_truth_columns`. Reproducibility depends on the RNG state for stochastic fault types.

#### Why This Function / Class Matters

Controlled anomaly simulation depends on fault shapes being deliberate, bounded, and repeatable. This method centralizes the transformation logic that makes a configured fault type visible in sensor values while respecting profile-derived limits.

#### Verification Method

- Confirm each supported fault type changes values in the expected direction or pattern.
- Confirm returned values remain within profile bounds.
- Confirm `stuck_constant` produces a constant vector based on the input vector's local median.
- Confirm stochastic fault types are reproducible with the same RNG seed and call order.
- Confirm propagated unsupported secondary fault recommendations are normalized before reaching this method.

## Cross-Function Relationships

`generate_episode` is the orchestration path that brings the selected methods together:

- It uses `_apply_buildup_transition_noise` and `_inject_fault` to create primary fault behavior.
- It uses `_inject_fault` again for propagated secondary sensor effects.
- It uses group and cluster correlation helpers to restore multivariate structure after phase and fault edits.
- `_build_group_correlation_matrix` supplies the repaired matrix used by `_apply_group_correlated_residuals`.
- `_apply_anchor_cluster_generation` reinforces hotspot clusters where a group-wide residual is not specific enough.
- `_apply_sensor_variance_floor` protects normal, buildup, and recovery windows from variance collapse.
- `_build_episode_truth_columns` runs after value generation so row-level truth metadata reflects the final phase windows and generated primary-sensor behavior.
- Optional missingness replay runs after truth stamping, so missing sensor values do not remove the row-level truth columns.

## Source-Limited Items

- Direct final-facing notebook consumer context for each assigned private method is Not determined from available source.
- Artifact, SQL, W&B, ledger, and truth-record file interactions are not performed by these assigned methods from available source.
- Unsupported primary fault-type behavior before `_inject_fault` is Not determined from available source beyond the `FaultType` annotation and the implemented branches.
- The exact downstream notebook or pipeline stage that consumes every generated truth column is Not determined from available source.
