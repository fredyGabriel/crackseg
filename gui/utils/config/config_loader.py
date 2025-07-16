"""
Unified Configuration Loading System with Advanced YAML Validation.

This module provides a comprehensive configuration loading mechanism that
combines file I/O operations with extensive YAML validation, schema checking,
and detailed error reporting for the CrackSeg crack segmentation project.

Key Features:
- Unified loading interface combining I/O + validation
- Advanced schema validation for crack segmentation configs
- Comprehensive error detection and reporting
- Support for nested configuration structures
- Caching integration for performance optimization
- Type-safe configuration loading with validation
"""

import logging
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from .cache import _config_cache
from .exceptions import ConfigError, ValidationError
from .validation import validate_with_hydra, validate_yaml_advanced

logger = logging.getLogger(__name__)


class ConfigLoadResult(NamedTuple):
    """Result container for configuration loading operations."""

    config: dict[str, Any]
    is_valid: bool
    errors: list[ValidationError]
    warnings: list[str]
    metadata: dict[str, Any]


class UnifiedConfigLoader:
    """
    Advanced configuration loader with integrated validation.

    Provides a single interface for loading, validating, and parsing
    YAML configuration files with comprehensive error detection and
    detailed reporting specifically tailored for crack segmentation
    model configurations.
    """

    def __init__(self, enable_caching: bool = True) -> None:
        """
        Initialize the unified configuration loader.

        Args:
            enable_caching: Whether to enable configuration caching for
                performance optimization.
        """
        self.enable_caching = enable_caching
        self._cache = _config_cache if enable_caching else None

        # CrackSeg-specific schema definitions
        self._schema_definitions = self._initialize_crackseg_schemas()

        logger.debug(
            "UnifiedConfigLoader initialized with caching=%s", enable_caching
        )

    def load_config(
        self,
        path: str | Path,
        validate_schema: bool = True,
        validate_hydra: bool = False,
        strict_mode: bool = False,
    ) -> ConfigLoadResult:
        """
        Load and validate a configuration file with comprehensive checks.

        Args:
            path: Path to the YAML configuration file
            validate_schema: Whether to perform schema validation
            validate_hydra: Whether to validate with Hydra composition
            strict_mode: Whether to treat warnings as errors

        Returns:
            ConfigLoadResult containing config data, validation status,
            errors, warnings, and metadata

        Raises:
            ConfigError: If critical loading errors occur
        """
        path = Path(path)
        cache_key = str(path) if self.enable_caching else None

        # Check cache first
        if cache_key and self._cache:
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                logger.debug("Loaded config from cache: %s", path)
                return cached_result  # type: ignore[return-value]

        # Load and validate configuration
        try:
            config_data = self._load_yaml_file(path)
            errors: list[ValidationError] = []
            warnings: list[str] = []

            # Perform syntax validation
            content = self._read_file_content(path)
            syntax_valid, syntax_errors = validate_yaml_advanced(content)
            if syntax_errors:
                errors.extend(syntax_errors)

            # Perform schema validation if requested
            if validate_schema and config_data:
                schema_errors, schema_warnings = self._validate_schema(
                    config_data
                )
                errors.extend(schema_errors)
                warnings.extend(schema_warnings)

            # Perform Hydra validation if requested
            if validate_hydra and path.exists():
                hydra_valid, hydra_error = validate_with_hydra(
                    str(path.parent), path.stem
                )
                if not hydra_valid and hydra_error:
                    errors.append(
                        ValidationError(
                            message=f"Hydra validation failed: {hydra_error}",
                            field="hydra_composition",
                        )
                    )

            # Generate metadata
            metadata = self._generate_metadata(path, config_data)

            # Determine overall validity
            is_valid = len(errors) == 0 or (
                not strict_mode and all(not e.is_critical for e in errors)
            )

            result = ConfigLoadResult(
                config=config_data,
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                metadata=metadata,
            )

            # Cache result if caching is enabled
            if cache_key and self._cache and is_valid:
                self._cache.set(cache_key, result)  # type: ignore[arg-type]
                logger.debug("Cached loaded config: %s", path)

            return result

        except Exception as e:
            logger.error("Failed to load config %s: %s", path, str(e))
            raise ConfigError(
                f"Critical error loading {path}: {str(e)}"
            ) from e

    def load_multiple_configs(
        self, paths: list[str | Path], fail_on_first_error: bool = False
    ) -> dict[str, ConfigLoadResult]:
        """
        Load multiple configuration files efficiently.

        Args:
            paths: List of configuration file paths
            fail_on_first_error: Whether to stop on first loading error

        Returns:
            Dictionary mapping file paths to ConfigLoadResult objects
        """
        results: dict[str, ConfigLoadResult] = {}

        for path in paths:
            try:
                result = self.load_config(path)
                results[str(path)] = result

                if fail_on_first_error and not result.is_valid:
                    break

            except ConfigError as e:
                logger.warning("Failed to load config %s: %s", path, str(e))
                if fail_on_first_error:
                    raise

                # Create error result
                results[str(path)] = ConfigLoadResult(
                    config={},
                    is_valid=False,
                    errors=[ValidationError(message=str(e))],
                    warnings=[],
                    metadata={"error": True, "path": str(path)},
                )

        return results

    def _load_yaml_file(self, path: Path) -> dict[str, Any]:
        """Load YAML file with proper error handling."""
        try:
            with open(path, encoding="utf-8") as f:
                content = yaml.safe_load(f)
                return content if content is not None else {}
        except FileNotFoundError as e:
            raise ConfigError(f"Configuration file not found: {path}") from e
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML syntax in {path}: {e}") from e

    def _read_file_content(self, path: Path) -> str:
        """Read file content as string for validation."""
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning("Could not read file content for validation: %s", e)
            return ""

    def _validate_schema(
        self, config: dict[str, Any]
    ) -> tuple[list[ValidationError], list[str]]:
        """Validate configuration against CrackSeg schemas."""
        errors: list[ValidationError] = []
        warnings: list[str] = []

        # Validate required sections
        required_sections = self._schema_definitions["required_sections"]
        for section in required_sections:
            if section not in config:
                errors.append(
                    ValidationError(
                        message=f"Missing required section: '{section}'",
                        field=section,
                        suggestions=[
                            f"Add '{section}:' section to your configuration",
                            f"See example configs in configs/{section}/",
                        ],
                    )
                )

        # Validate section-specific schemas
        for section_name, section_config in config.items():
            if section_name in self._schema_definitions["section_schemas"]:
                section_errors = self._validate_section_schema(
                    section_name, section_config
                )
                errors.extend(section_errors)

        return errors, warnings

    def _validate_section_schema(
        self, section_name: str, section_config: Any
    ) -> list[ValidationError]:
        """Validate a specific configuration section."""
        errors: list[ValidationError] = []
        schema = self._schema_definitions["section_schemas"].get(
            section_name, {}
        )

        if not isinstance(section_config, dict):
            errors.append(
                ValidationError(
                    message=f"Section '{section_name}' must be a dictionary",
                    field=section_name,
                )
            )
            return errors

        # Validate required fields in section
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in section_config:
                errors.append(
                    ValidationError(
                        message=(
                            f"Missing required field '{field}' "
                            f"in section '{section_name}'"
                        ),
                        field=f"{section_name}.{field}",
                        suggestions=[
                            f"Add '{field}:' to your {section_name} config"
                        ],
                    )
                )

        return errors

    def _generate_metadata(
        self, path: Path, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate metadata about the loaded configuration."""
        metadata = {
            "path": str(path),
            "name": path.name,
            "exists": path.exists(),
            "config_sections": list(config.keys()) if config else [],
            "config_size": len(str(config)) if config else 0,
        }

        if path.exists():
            stat = path.stat()
            file_metadata: dict[str, Any] = {
                "file_size": int(stat.st_size),
                "modified_time": float(stat.st_mtime),
                "is_readable": True,
            }
            metadata.update(file_metadata)

        return metadata

    def _initialize_crackseg_schemas(self) -> dict[str, Any]:
        """Initialize CrackSeg-specific configuration schemas."""
        return {
            "required_sections": ["model", "training", "data"],
            "section_schemas": {
                "model": {
                    "required": ["architecture", "encoder", "decoder"],
                    "optional": [
                        "bottleneck",
                        "num_classes",
                        "input_channels",
                    ],
                },
                "training": {
                    "required": ["epochs", "learning_rate", "optimizer"],
                    "optional": [
                        "scheduler",
                        "loss",
                        "metrics",
                        "early_stopping",
                    ],
                },
                "data": {
                    "required": ["data_root", "image_size", "batch_size"],
                    "optional": ["augmentations", "normalization", "splits"],
                },
            },
        }


# Global instance for backward compatibility
_unified_loader = UnifiedConfigLoader()


def load_config_with_validation(
    path: str | Path, validate_schema: bool = True, strict_mode: bool = False
) -> ConfigLoadResult:
    """
    Convenience function for loading and validating configuration files.

    Args:
        path: Path to configuration file
        validate_schema: Whether to perform schema validation
        strict_mode: Whether to treat warnings as errors

    Returns:
        ConfigLoadResult with config data and validation results
    """
    return _unified_loader.load_config(
        path=path, validate_schema=validate_schema, strict_mode=strict_mode
    )
