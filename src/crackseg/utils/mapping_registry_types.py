"""Types for mapping registry to keep main module small and focused."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PathMapping:
    """Represents a mapping between old and new paths."""

    old_path: str
    new_path: str
    mapping_type: str  # 'import', 'config', 'docs', 'artifact', 'checkpoint'
    description: str
    deprecated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
