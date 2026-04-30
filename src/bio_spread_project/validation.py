"""Validation utilities for configuration and runtime policy checks."""

from __future__ import annotations

from typing import Any


def to_float(name: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def to_int(name: str, value: int | float | str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    return parsed


def require_range(name: str, value: float, *, minimum: float, maximum: float) -> float:
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


def require_non_negative(name: str, value: float) -> float:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value
