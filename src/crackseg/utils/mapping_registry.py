"""Mapping registry for managing path changes across the project.

This module provides a centralized registry to track and manage all path changes
across documentation, tickets, analysis reports, and code components. It supports
automated updates during refactors and migrations.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import CrackSegError


@dataclass
class PathMapping:
    """Represents a mapping between old and new paths."""

    old_path: str
    new_path: str
    mapping_type: str  # 'import', 'config', 'docs', 'artifact', 'checkpoint'
    description: str
    deprecated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MappingRegistry:
    """Centralized registry for managing path mappings across the project."""

    mappings: list[PathMapping] = field(default_factory=list)
    registry_file: Path | None = None
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(__name__)
    )

    def __post_init__(self) -> None:
        """Initialize the registry."""
        if self.registry_file is None:
            self.registry_file = Path("artifacts/mapping_registry.json")

    def add_mapping(
        self,
        old_path: str,
        new_path: str,
        mapping_type: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a new path mapping to the registry.

        Args:
            old_path: The old path to be replaced
            new_path: The new path to replace with
            mapping_type: Type of mapping ('import', 'config', 'docs', 'artifact', 'checkpoint')
            description: Human-readable description of the mapping
            metadata: Additional metadata for the mapping
        """
        mapping = PathMapping(
            old_path=old_path,
            new_path=new_path,
            mapping_type=mapping_type,
            description=description,
            metadata=metadata or {},
        )

        # Check for conflicts
        for existing in self.mappings:
            if (
                existing.old_path == old_path
                and existing.mapping_type == mapping_type
            ):
                self.logger.warning(
                    f"Mapping conflict detected for {old_path} ({mapping_type}). "
                    f"Existing: {existing.new_path}, New: {new_path}"
                )
                return

        self.mappings.append(mapping)
        self.logger.info(
            f"Added mapping: {old_path} -> {new_path} ({mapping_type})"
        )

    def get_mapping(
        self, old_path: str, mapping_type: str
    ) -> PathMapping | None:
        """Get a mapping for a specific old path and type.

        Args:
            old_path: The old path to look up
            mapping_type: Type of mapping to search for

        Returns:
            PathMapping if found, None otherwise
        """
        for mapping in self.mappings:
            if (
                mapping.old_path == old_path
                and mapping.mapping_type == mapping_type
            ):
                return mapping
        return None

    def get_mappings_by_type(self, mapping_type: str) -> list[PathMapping]:
        """Get all mappings of a specific type.

        Args:
            mapping_type: Type of mappings to retrieve

        Returns:
            List of PathMapping objects of the specified type
        """
        return [m for m in self.mappings if m.mapping_type == mapping_type]

    def apply_mapping(self, content: str, mapping_type: str) -> str:
        """Apply mappings to content based on type.

        Args:
            content: The content to apply mappings to
            mapping_type: Type of mappings to apply

        Returns:
            Content with mappings applied
        """
        result = content
        type_mappings = self.get_mappings_by_type(mapping_type)

        for mapping in type_mappings:
            if mapping.deprecated:
                continue

            # Apply different replacement strategies based on type
            if mapping_type == "import":
                result = self._apply_import_mapping(result, mapping)
            elif mapping_type == "config":
                result = self._apply_config_mapping(result, mapping)
            elif mapping_type == "docs":
                result = self._apply_docs_mapping(result, mapping)
            elif mapping_type == "artifact":
                result = self._apply_artifact_mapping(result, mapping)
            else:
                # Generic replacement
                result = result.replace(mapping.old_path, mapping.new_path)

        return result

    def _apply_import_mapping(self, content: str, mapping: PathMapping) -> str:
        """Apply import-specific mapping rules."""
        # Handle Python import statements
        import_patterns = [
            f"from {mapping.old_path}",
            f"import {mapping.old_path}",
            f"from {mapping.old_path}.",
            f"import {mapping.old_path}.",
        ]

        result = content
        for pattern in import_patterns:
            new_pattern = pattern.replace(mapping.old_path, mapping.new_path)
            result = result.replace(pattern, new_pattern)

        return result

    def _apply_config_mapping(self, content: str, mapping: PathMapping) -> str:
        """Apply configuration-specific mapping rules."""
        # Handle Hydra config paths
        config_patterns = [
            f"{mapping.old_path}:",
            f"  {mapping.old_path}:",
            f"    {mapping.old_path}:",
            f"defaults:\n  - {mapping.old_path}",
            f"defaults:\n    - {mapping.old_path}",
        ]

        result = content
        for pattern in config_patterns:
            new_pattern = pattern.replace(mapping.old_path, mapping.new_path)
            result = result.replace(pattern, new_pattern)

        return result

    def _apply_docs_mapping(self, content: str, mapping: PathMapping) -> str:
        """Apply documentation-specific mapping rules."""
        # Handle markdown links and references
        docs_patterns = [
            f"({mapping.old_path})",
            f"[{mapping.old_path}]",
            f"`{mapping.old_path}`",
            f"```{mapping.old_path}```",
        ]

        result = content
        for pattern in docs_patterns:
            new_pattern = pattern.replace(mapping.old_path, mapping.new_path)
            result = result.replace(pattern, new_pattern)

        return result

    def _apply_artifact_mapping(
        self, content: str, mapping: PathMapping
    ) -> str:
        """Apply artifact-specific mapping rules."""
        # Handle file paths and directories
        artifact_patterns = [
            f'"{mapping.old_path}"',
            f"'{mapping.old_path}'",
            f"Path('{mapping.old_path}')",
            f"pathlib.Path('{mapping.old_path}')",
        ]

        result = content
        for pattern in artifact_patterns:
            new_pattern = pattern.replace(mapping.old_path, mapping.new_path)
            result = result.replace(pattern, new_pattern)

        return result

    def save_registry(self, file_path: Path | None = None) -> None:
        """Save the registry to a JSON file.

        Args:
            file_path: Optional path to save the registry. Uses default if None.
        """
        if file_path is None:
            file_path = self.registry_file
        if file_path is None:
            print("No registry file path specified")
            return

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert mappings to serializable format
        registry_data = {
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
                for m in self.mappings
            ],
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Registry saved to {file_path}")

    def load_registry(self, file_path: Path | None = None) -> None:
        """Load the registry from a JSON file.

        Args:
            file_path: Optional path to load the registry from. Uses default if None.
        """
        if file_path is None:
            file_path = self.registry_file
        if file_path is None:
            print("No registry file path specified")
            return

        if not file_path.exists():
            self.logger.warning(f"Registry file not found: {file_path}")
            return

        try:
            with open(file_path, encoding="utf-8") as f:
                registry_data = json.load(f)

            # Clear existing mappings
            self.mappings.clear()

            # Load mappings
            for mapping_data in registry_data.get("mappings", []):
                mapping = PathMapping(
                    old_path=mapping_data["old_path"],
                    new_path=mapping_data["new_path"],
                    mapping_type=mapping_data["mapping_type"],
                    description=mapping_data["description"],
                    deprecated=mapping_data.get("deprecated", False),
                    metadata=mapping_data.get("metadata", {}),
                )
                self.mappings.append(mapping)

            self.logger.info(
                f"Registry loaded from {file_path} with {len(self.mappings)} mappings"
            )

        except Exception as e:
            raise CrackSegError(
                f"Failed to load registry from {file_path}: {e}"
            ) from e

    def validate_mappings(self) -> list[str]:
        """Validate all mappings for consistency and correctness.

        Returns:
            List of validation error messages
        """
        errors = []

        for mapping in self.mappings:
            # Check for empty paths
            if not mapping.old_path or not mapping.new_path:
                errors.append(f"Empty path in mapping: {mapping}")

            # Check for self-references
            if mapping.old_path == mapping.new_path:
                errors.append(f"Self-reference in mapping: {mapping.old_path}")

            # Check for circular references
            for other in self.mappings:
                if (
                    other != mapping
                    and other.old_path == mapping.new_path
                    and other.new_path == mapping.old_path
                ):
                    errors.append(
                        f"Circular reference: {mapping.old_path} <-> {mapping.new_path}"
                    )

        return errors

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the registry.

        Returns:
            Dictionary with registry statistics
        """
        stats = {
            "total_mappings": len(self.mappings),
            "by_type": {},
            "deprecated_count": 0,
        }

        for mapping in self.mappings:
            mapping_type = mapping.mapping_type
            if mapping_type not in stats["by_type"]:
                stats["by_type"][mapping_type] = 0
            stats["by_type"][mapping_type] += 1

            if mapping.deprecated:
                stats["deprecated_count"] += 1

        return stats


# Global registry instance
_registry: MappingRegistry | None = None


def get_registry() -> MappingRegistry:
    """Get the global registry instance.

    Returns:
        Global MappingRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = MappingRegistry()
        # Try to load existing registry
        try:
            _registry.load_registry()
        except Exception:
            # Initialize with default mappings if loading fails
            _initialize_default_mappings(_registry)
    return _registry


def _initialize_default_mappings(registry: MappingRegistry) -> None:
    """Initialize the registry with default mappings for the project.

    Args:
        registry: The registry to initialize
    """
    # Import mappings
    registry.add_mapping(
        "from src.crackseg",
        "from crackseg",
        "import",
        "Standardize import paths to use package name",
    )

    registry.add_mapping(
        "from src.training_pipeline",
        "from crackseg.training",
        "import",
        "Consolidate training pipeline imports",
    )

    # Config mappings
    registry.add_mapping(
        "model.encoder",
        "model.architectures",
        "config",
        "Reorganize model configuration structure",
    )

    registry.add_mapping(
        "outputs/",
        "artifacts/",
        "artifact",
        "Standardize artifact output directory",
    )

    # Documentation mappings
    registry.add_mapping(
        "docs/guides/",
        "docs/user-guides/",
        "docs",
        "Reorganize documentation structure",
    )

    # Save the default registry
    registry.save_registry()
