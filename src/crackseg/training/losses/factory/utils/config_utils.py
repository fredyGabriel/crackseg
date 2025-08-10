"""Utility helpers for loss config parsing.

Extracted from factory/config_parser.py to reduce size and improve modularity.
"""

from __future__ import annotations

from typing import Any, cast


def is_combinator_node(config: dict[str, Any]) -> bool:
    return "type" in config and config["type"] in {"sum", "product"}


def is_leaf_node(config: dict[str, Any]) -> bool:
    return "name" in config


def normalize_weights(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total == 0:
        raise ValueError("Cannot normalize weights: sum is zero")
    return [w / total for w in weights]


def extract_and_validate_weights(
    config: dict[str, Any], children_count: int
) -> list[float] | None:
    combinator_type = config.get("type")
    weights = config.get("weights")
    if combinator_type != "sum":
        return None
    if weights is None:
        equal = 1.0 / children_count
        return [equal] * children_count
    if not isinstance(weights, list):
        return None
    if len(cast(list[float], weights)) != children_count:
        return None
    if any(w <= 0 for w in weights):
        return None
    return normalize_weights(weights)
