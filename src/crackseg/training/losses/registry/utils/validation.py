"""Validation utilities for loss registry.

Extracted from enhanced_registry to reduce module size and improve modularity.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def extract_parameter_schema(
    factory_func: Callable[..., Any],
) -> dict[str, Any]:
    """Extract parameter schema from a factory function signature.

    Returns a dict with keys: required (set), types (dict), defaults (dict).
    """
    try:
        sig = inspect.signature(factory_func)
        schema: dict[str, Any] = {
            "required": set(),
            "types": {},
            "defaults": {},
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "args", "kwargs"):
                continue

            if param.default == inspect.Parameter.empty:
                schema["required"].add(param_name)
            else:
                schema["defaults"][param_name] = param.default

            if param.annotation != inspect.Parameter.empty:
                schema["types"][param_name] = param.annotation

        return schema
    except Exception:
        return {}


def check_constraint(value: Any, constraint: dict[str, Any]) -> bool:
    """Check value against min/max/choices constraints."""
    if "min" in constraint and value < constraint["min"]:
        return False
    if "max" in constraint and value > constraint["max"]:
        return False
    if "choices" in constraint and value not in constraint["choices"]:
        return False
    return True
