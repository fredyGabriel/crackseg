from __future__ import annotations

import json
from pathlib import Path

from .mapping_registry_types import PathMapping
from .mapping_registry_utils import PathMappingLite, serialize_mappings


def save_registry_to_file(
    file_path: Path, mappings: list[PathMapping]
) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lite = [
        PathMappingLite(
            old_path=m.old_path,
            new_path=m.new_path,
            mapping_type=m.mapping_type,
            description=m.description,
            deprecated=m.deprecated,
            metadata=m.metadata,
        )
        for m in mappings
    ]
    registry_data = serialize_mappings(lite)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2, ensure_ascii=False)


def load_registry_from_file(file_path: Path) -> list[PathMapping]:
    with open(file_path, encoding="utf-8") as f:
        registry_data = json.load(f)
    mappings: list[PathMapping] = []
    for mapping_data in registry_data.get("mappings", []):
        mappings.append(
            PathMapping(
                old_path=mapping_data["old_path"],
                new_path=mapping_data["new_path"],
                mapping_type=mapping_data["mapping_type"],
                description=mapping_data["description"],
                deprecated=mapping_data.get("deprecated", False),
                metadata=mapping_data.get("metadata", {}),
            )
        )
    return mappings


def initialize_default_mappings() -> list[tuple[str, str, str, str]]:
    """Return default mapping tuples (old, new, type, description)."""
    return [
        (
            "from src.crackseg",
            "from crackseg",
            "import",
            "Standardize import paths to use package name",
        ),
        (
            "from src.training_pipeline",
            "from crackseg.training",
            "import",
            "Consolidate training pipeline imports",
        ),
        (
            "model.encoder",
            "model.architectures",
            "config",
            "Reorganize model configuration structure",
        ),
        (
            "outputs/",
            "artifacts/",
            "artifact",
            "Standardize artifact output directory",
        ),
        (
            "docs/guides/",
            "docs/user-guides/",
            "docs",
            "Reorganize documentation structure",
        ),
    ]
