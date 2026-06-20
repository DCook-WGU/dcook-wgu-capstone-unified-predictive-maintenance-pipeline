from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Dict, Optional, Sequence

TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}
# Strings that represent "nothing configured" in an env var. "all" is included
# because some tools set it as a "no specific selection / apply globally" sentinel.
NONE_VALUES = {"", "none", "null", "nil", "na", "n/a", "all"}


def env_raw(name: str, aliases: Sequence[str] = ()) -> str | None:
    """
    Return the first non-empty environment value for a name or alias.

    Args:
        name: Primary environment variable name to read.
        aliases: Fallback environment variable names checked in order.

    Returns:
        The stripped environment value, or None when no name is set.

    Side Effects:
        Reads from the process environment without mutating it.
    """
    for candidate_name in (name, *aliases):
        value = os.getenv(candidate_name)

        if value is None:
            continue

        cleaned_value = str(value).strip()

        if cleaned_value != "":
            return cleaned_value

    return None


def env_required_str(name: str, *, aliases: Sequence[str] = ()) -> str:
    """
    Return a required environment string from a name or alias.

    Args:
        name: Primary environment variable name to read.
        aliases: Fallback environment variable names checked in order.

    Returns:
        The stripped environment value.

    Raises:
        RuntimeError: If the variable and all aliases are missing or empty.

    Side Effects:
        Reads from the process environment without mutating it.
    """
    value = env_raw(name, aliases=aliases)

    if value is None:
        alias_text = f" or aliases {tuple(aliases)}" if aliases else ""
        raise RuntimeError(f"Missing required environment variable: {name}{alias_text}")

    return value


def env_str(name: str, default: str, *, aliases: Sequence[str] = ()) -> str:
    """
    Return an environment string, falling back to a default value.

    Args:
        name: Primary environment variable name to read.
        default: Value returned when no name or alias is set.
        aliases: Fallback environment variable names checked in order.

    Returns:
        The stripped environment value, or default converted to str.

    Side Effects:
        Reads from the process environment without mutating it.
    """
    value = env_raw(name, aliases=aliases)

    if value is None:
        return str(default)

    return value


def env_int(name: str, default: int, *, aliases: Sequence[str] = ()) -> int:
    """
    Return an environment integer, falling back to a default value.

    Args:
        name: Primary environment variable name to read.
        default: Value returned when no name or alias is set.
        aliases: Fallback environment variable names checked in order.

    Returns:
        The environment value parsed as int, or default converted to int.

    Side Effects:
        Reads from the process environment without mutating it.
    """
    value = env_raw(name, aliases=aliases)

    if value is None:
        return int(default)

    return int(value)


def env_optional_int(
    name: str,
    default: int | None = None,
    *,
    aliases: Sequence[str] = (),
) -> int | None:
    """
    Return an optional environment integer.

    Args:
        name: Primary environment variable name to read.
        default: Value returned when no name or alias is set.
        aliases: Fallback environment variable names checked in order.

    Returns:
        The environment value parsed as int, None for configured none-like
        strings, or the default when the variable is unset.

    Side Effects:
        Reads from the process environment without mutating it.
    """
    value = env_raw(name, aliases=aliases)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in NONE_VALUES:
        return None

    return int(value)


def env_float(name: str, default: float, *, aliases: Sequence[str] = ()) -> float:
    """
    Return an environment float, falling back to a default value.

    Args:
        name: Primary environment variable name to read.
        default: Value returned when no name or alias is set.
        aliases: Fallback environment variable names checked in order.

    Returns:
        The environment value parsed as float, or default converted to float.

    Side Effects:
        Reads from the process environment without mutating it.
    """
    value = env_raw(name, aliases=aliases)

    if value is None:
        return float(default)

    return float(value)


def env_bool(name: str, default: bool, *, aliases: Sequence[str] = ()) -> bool:
    """
    Return an environment boolean, falling back to a default value.

    Args:
        name: Primary environment variable name to read.
        default: Value returned when no name or alias is set.
        aliases: Fallback environment variable names checked in order.

    Returns:
        The parsed boolean value, or default converted to bool.

    Raises:
        ValueError: If the environment value is not a recognized boolean.

    Side Effects:
        Reads from the process environment without mutating it.
    """
    value = env_raw(name, aliases=aliases)

    if value is None:
        return bool(default)

    normalized = value.strip().lower()

    if normalized in TRUE_VALUES:
        return True

    if normalized in FALSE_VALUES:
        return False

    # Raising is intentional: bool(value) would silently coerce any non-empty string to
    # True (e.g., bool("maybe") == True), masking misconfigured environment variables.
    raise ValueError(
        f"Invalid boolean value for {name}: {value!r}. "
        f"Use one of {sorted(TRUE_VALUES | FALSE_VALUES)}."
    )


def get_first_env_value(names: Sequence[str]) -> Optional[str]:
    """
    Return the first non-empty value from a sequence of environment names.

    Args:
        names: Environment variable names checked in order.

    Returns:
        The stripped first matching value, or None when all names are unset.

    Side Effects:
        Reads from the process environment without mutating it.
    """
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def get_kafka_bootstrap_servers_from_env(
    env_names: Sequence[str] = (
        "KAFKA_BOOTSTRAP_SERVERS",
        "BOOTSTRAP_SERVERS",
        "KAFKA_BROKERS",
    ),
) -> str:
    """
    Resolve Kafka bootstrap servers from configured environment names.

    Args:
        env_names: Environment variable names checked in order.

    Returns:
        The stripped Kafka bootstrap server string.

    Raises:
        RuntimeError: If none of the configured variables are set.

    Side Effects:
        Reads from the process environment without mutating it.
    """
    value = get_first_env_value(env_names)
    if value is None:
        raise RuntimeError(
            "Missing Kafka bootstrap servers. Checked: "
            + ", ".join(env_names)
        )
    return value


def get_kafka_consumer_group_from_env(
    env_names: Sequence[str] = (
        "KAFKA_CONSUMER_GROUP_ID",
        "CONSUMER_GROUP_ID",
    ),
    default: str = "synthetic-telemetry-consumer-group",
) -> str:
    """
    Resolve the Kafka consumer group from the environment or default.

    Args:
        env_names: Environment variable names checked in order.
        default: Consumer group returned when no configured variable is set.

    Returns:
        The stripped environment value, or default converted to str and stripped.

    Side Effects:
        Reads from the process environment without mutating it.
    """
    return get_first_env_value(env_names) or str(default).strip()

__all__ = [
    "env_raw",
    "env_required_str",
    "env_str",
    "env_int",
    "env_optional_int",
    "env_float",
    "env_bool",
    "get_first_env_value",
    "get_kafka_bootstrap_servers_from_env",
    "get_kafka_consumer_group_from_env",
]
