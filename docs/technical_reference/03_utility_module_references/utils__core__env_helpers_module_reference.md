# Utility Module Reference: `utils/core/env_helpers.py`

## Module Purpose

This module provides typed environment-variable readers used by runtime and optional integration helpers.

## Pipeline Role

- Stage support: Core / cross-stage
- Primary responsibility: This module provides typed environment-variable readers used by runtime and optional integration helpers.

## Primary Consumers

Not determined from available source

## Main Objects and Functions

| Object / Function | Purpose | Reference Depth |
|---|---|---|
| `env_raw` | Return the first non-empty environment value for a name or alias. | short |
| `env_required_str` | Return a required environment string from a name or alias. | deep |
| `env_str` | Return an environment string, falling back to a default value. | short |
| `env_int` | Return an environment integer, falling back to a default value. | short |
| `env_optional_int` | Return an optional environment integer. | short |
| `env_float` | Return an environment float, falling back to a default value. | short |
| `env_bool` | Return an environment boolean, falling back to a default value. | deep |
| `get_first_env_value` | Return the first non-empty value from a sequence of environment names. | short |
| `get_kafka_bootstrap_servers_from_env` | Resolve Kafka bootstrap servers from configured environment names. | deep |
| `get_kafka_consumer_group_from_env` | Resolve the Kafka consumer group from the environment or default. | deep |

## Configuration Dependencies

- Environment variables where runtime mode or optional integration behavior is configured.
- Kafka topic and producer/consumer settings for synthetic streaming handoff.

## Inputs and Outputs

| Function / Object | Inputs | Outputs / Returns |
|---|---|---|
| `env_raw` | `name, aliases` | Return the first non-empty environment value for a name or alias. |
| `env_required_str` | `name, *, aliases` | Return a required environment string from a name or alias. |
| `env_str` | `name, default, *, aliases` | Return an environment string, falling back to a default value. |
| `env_int` | `name, default, *, aliases` | Return an environment integer, falling back to a default value. |
| `env_optional_int` | `name, default, *, aliases` | Return an optional environment integer. |
| `env_float` | `name, default, *, aliases` | Return an environment float, falling back to a default value. |
| `env_bool` | `name, default, *, aliases` | Return an environment boolean, falling back to a default value. |
| `get_first_env_value` | `names` | Return the first non-empty value from a sequence of environment names. |
| `get_kafka_bootstrap_servers_from_env` | `env_names` | Resolve Kafka bootstrap servers from configured environment names. |
| `get_kafka_consumer_group_from_env` | `env_names, default` | Resolve the Kafka consumer group from the environment or default. |

## Side Effects

- Source includes Kafka producer/consumer terminology or calls; helpers participate in synthetic streaming handoff when used by the synthetic pipeline.

## Artifact / SQL / File-System Interactions

- Kafka/PostgreSQL handoff: Source references producer, consumer, topic, or streaming-stage behavior.

## Failure Behavior

- Source raises `RuntimeError` for invalid input, missing context, or failed validation paths.
- Source raises `ValueError` for invalid input, missing context, or failed validation paths.

## Downstream Usage

Not determined from available source

## Module Importance

This module matters because shared configuration, paths, logging, ledger, truth, artifact, and optional W&B behavior reduce duplicated notebook setup.

## Notes / Risks / Deferred Cleanup

- This is a module-level reference. Detailed function-level design belongs in a separate deep utility reference pass.
- Primary notebook or script consumers were not determined from the available workflow references.
