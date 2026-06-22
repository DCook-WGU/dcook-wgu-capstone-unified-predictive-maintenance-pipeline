# Utility Module Reference: `utils/synthetic/pipeline/output_to_bronze_runner.py`

## Module Purpose

This module runs the synthetic final-output to Bronze handoff workflow.

## Pipeline Role

- Stage support: Synthetic pipeline
- Primary responsibility: This module runs the synthetic final-output to Bronze handoff workflow.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `run_synthetic_to_bronze_once` | Run one rebuild, final-align, and Bronze handoff pass. | deep |
| `run_synthetic_to_bronze_loop` | Repeat synthetic-to-Bronze passes until no stage writes new rows. | deep |

## Configuration Dependencies

- No explicit configuration dependency was determined from available source.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `run_synthetic_to_bronze_once` | `engine, *, schema, dataset_id, run_id, observation_batch_size, n_sensors, complete_only` | Run one rebuild, final-align, and Bronze handoff pass. |
| `run_synthetic_to_bronze_loop` | `engine, *, schema, dataset_id, run_id, observation_batch_size, n_sensors, complete_only, max_iterations` | Repeat synthetic-to-Bronze passes until no stage writes new rows. |

## Side Effects

- Not determined from available source

## Artifact / SQL / File-System Interactions

- Not determined from available source

## Failure Behavior

- Not determined from available source

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because the synthetic subsystem depends on repeatable staging, truth handling, streaming handoff, or final alignment behavior to make controlled pump telemetry tests explainable.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
