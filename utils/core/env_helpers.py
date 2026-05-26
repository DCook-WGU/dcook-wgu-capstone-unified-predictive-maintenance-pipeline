from __future__ import annotations

import os
from collections.abc import Sequence


TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}
NONE_VALUES = {"", "none", "null", "nil", "na", "n/a", "all"}


def env_raw(name: str, aliases: Sequence[str] = ()) -> str | None:
    """
    Return the first non-empty environment value for a variable name or aliases.

    This lets newer standardized names work while still supporting older names
    during the transition period.
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
    Return a required environment string.

    Raises:
        RuntimeError: If the variable and all aliases are missing or empty.
    """
    value = env_raw(name, aliases=aliases)

    if value is None:
        alias_text = f" or aliases {tuple(aliases)}" if aliases else ""
        raise RuntimeError(f"Missing required environment variable: {name}{alias_text}")

    return value


def env_str(name: str, default: str, *, aliases: Sequence[str] = ()) -> str:
    """
    Return an environment string, falling back to a default value.
    """
    value = env_raw(name, aliases=aliases)

    if value is None:
        return str(default)

    return value


def env_int(name: str, default: int, *, aliases: Sequence[str] = ()) -> int:
    """
    Return an environment integer, falling back to a default value.
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
    Return an optional integer.

    Strings like None, null, all, n/a, or an empty value are treated as None.
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
    """
    value = env_raw(name, aliases=aliases)

    if value is None:
        return float(default)

    return float(value)


def env_bool(name: str, default: bool, *, aliases: Sequence[str] = ()) -> bool:
    """
    Return an environment boolean, falling back to a default value.

    Accepted true values:
        1, true, t, yes, y, on

    Accepted false values:
        0, false, f, no, n, off
    """
    value = env_raw(name, aliases=aliases)

    if value is None:
        return bool(default)

    normalized = value.strip().lower()

    if normalized in TRUE_VALUES:
        return True

    if normalized in FALSE_VALUES:
        return False

    raise ValueError(
        f"Invalid boolean value for {name}: {value!r}. "
        f"Use one of {sorted(TRUE_VALUES | FALSE_VALUES)}."
    )