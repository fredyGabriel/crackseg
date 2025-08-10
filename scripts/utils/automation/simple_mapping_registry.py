"""Simple mapping registry for testing and automation.

This is a standalone version of the mapping registry that doesn't depend
on the project's logging system to avoid circular imports.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass
class SimpleMappingRegistry:
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
        """Add a new path mapping to the registry."""
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
                print(
                    f"Warning: Mapping conflict detected for {old_path} ({mapping_type})"
                )
                return

        self.mappings.append(mapping)
        print(f"Added mapping: {old_path} -> {new_path} ({mapping_type})")

    def get_mapping(
        self, old_path: str, mapping_type: str
    ) -> PathMapping | None:
        """Get a mapping for a specific old path and type."""
        for mapping in self.mappings:
            if (
                mapping.old_path == old_path
                and mapping.mapping_type == mapping_type
            ):
                return mapping
        return None

    def get_mappings_by_type(self, mapping_type: str) -> list[PathMapping]:
        """Get all mappings of a specific type."""
        return [m for m in self.mappings if m.mapping_type == mapping_type]

    def apply_mapping(self, content: str, mapping_type: str) -> str:
        """Apply mappings to content based on type."""
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
        """Save the registry to a JSON file."""
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

        print(f"Registry saved to {file_path}")

    def load_registry(self, file_path: Path | None = None) -> None:
        """Load the registry from a JSON file."""
        if file_path is None:
            file_path = self.registry_file

        if file_path is None:
            print("No registry file path specified")
            return

        if not file_path.exists():
            print(f"Registry file not found: {file_path}")
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

            print(
                f"Registry loaded from {file_path} with {len(self.mappings)} mappings"
            )

        except Exception as e:
            print(f"Failed to load registry from {file_path}: {e}")

    def validate_mappings(self) -> list[str]:
        """Validate all mappings for consistency and correctness."""
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
        """Get statistics about the registry."""
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


def create_default_registry() -> SimpleMappingRegistry:
    """Create a registry with default mappings for the project."""
    registry = SimpleMappingRegistry()

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

    return registry


def test_registry() -> None:
    """Test the simple mapping registry."""
    print("🧪 Testing Simple Mapping Registry")
    print("=" * 50)

    # Create registry
    registry = create_default_registry()

    # Test basic functionality
    print(f"Registry has {len(registry.mappings)} mappings")

    # Test applying mappings
    test_content = "from src.crackseg.model import UNet"
    result = registry.apply_mapping(test_content, "import")
    print(f"Test: '{test_content}' -> '{result}'")

    # Test statistics
    stats = registry.get_statistics()
    print(f"Statistics: {stats}")

    # Test validation
    errors = registry.validate_mappings()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("✅ No validation errors")

    # Test persistence
    test_file = Path("test_simple_registry.json")
    registry.save_registry(test_file)

    new_registry = SimpleMappingRegistry()
    new_registry.load_registry(test_file)
    print(f"✅ Loaded registry with {len(new_registry.mappings)} mappings")

    # Clean up
    test_file.unlink(missing_ok=True)
    print("✅ Test completed successfully!")


if __name__ == "__main__":
    test_registry()
