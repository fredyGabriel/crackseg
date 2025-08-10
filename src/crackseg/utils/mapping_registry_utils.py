"""Utilities extracted from mapping registry to reduce size and duplication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PathMappingLite:
    old_path: str
    new_path: str
    mapping_type: str
    description: str
    deprecated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def serialize_mappings(mappings: list[PathMappingLite]) -> dict[str, Any]:
    return {
        "version": "1.0",
        "mappings": [
            {
                "old_path": m.old_path,
                "new_path": m.new_path,
                "mapping_type": m.mapping_type,
                "description": m.description,
                "deprecated": m.deprecated,
                "metadata": m.metadata,
            }
            for m in mappings
        ],
    }


def apply_simple_replacements(content: str, old: str, new: str) -> str:
    return content.replace(old, new)
