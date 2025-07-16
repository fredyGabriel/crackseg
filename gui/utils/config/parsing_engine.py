"""
Advanced YAML Parsing Engine for Nested Configuration Structures.

This module provides sophisticated parsing capabilities for complex YAML
configurations with deep nesting, type conversion, data extraction, and
intelligent handling of crack segmentation specific configuration patterns.

Key Features:
- Deep nested structure parsing and navigation
- Intelligent type conversion with validation
- Configuration path resolution and interpolation
- Schema-aware data extraction
- Error-tolerant parsing with detailed reporting
- Support for Hydra-style configuration composition
"""

import logging
from typing import Any

import yaml

from .exceptions import ValidationError

logger = logging.getLogger(__name__)

PathType = str | list[str]
ConfigValue = str | int | float | bool | list[Any] | dict[str, Any] | None


class ConfigPath:
    """Utility for navigating nested configuration paths."""

    def __init__(self, path: PathType) -> None:
        """
        Initialize a configuration path.

        Args:
            path: Dot-separated string path or list of path components
        """
        if isinstance(path, str):
            self.components = path.split(".") if path else []
        else:
            self.components = list(path)

    def __str__(self) -> str:
        return ".".join(self.components)

    def __repr__(self) -> str:
        return f"ConfigPath('{self}')"

    def extend(self, component: str) -> "ConfigPath":
        """Create a new path by extending with a component."""
        return ConfigPath(self.components + [component])

    def parent(self) -> "ConfigPath":
        """Get parent path."""
        return ConfigPath(self.components[:-1])

    def leaf(self) -> str:
        """Get the leaf component."""
        return self.components[-1] if self.components else ""


class AdvancedYAMLParser:
    """
    Advanced YAML parser with support for nested structures and intelligent
    type conversion for crack segmentation configurations.
    """

    def __init__(self) -> None:
        """Initialize the advanced YAML parser."""
        self.type_converters = self._initialize_type_converters()
        self.interpolation_patterns = self._initialize_interpolation_patterns()

    def parse_config(
        self,
        content: str,
        enable_interpolation: bool = True,
        strict_types: bool = False,
    ) -> tuple[dict[str, Any], list[ValidationError]]:
        """
        Parse YAML content with advanced features.

        Args:
            content: YAML content as string
            enable_interpolation: Whether to resolve interpolation patterns
            strict_types: Whether to enforce strict type validation

        Returns:
            Tuple of (parsed_config, parsing_errors)
        """
        errors: list[ValidationError] = []

        try:
            # Parse basic YAML structure
            config = yaml.safe_load(content)
            if config is None:
                config = {}

            # Process nested structures
            processed_config = self._process_nested_structure(config, errors)

            # Resolve interpolations if enabled
            if enable_interpolation:
                processed_config = self._resolve_interpolations(
                    processed_config, processed_config, errors
                )

            # Apply type conversions
            if strict_types:
                processed_config = self._apply_type_conversions(
                    processed_config, errors
                )

            return processed_config, errors

        except yaml.YAMLError as e:
            errors.append(
                ValidationError(
                    message=f"YAML parsing error: {str(e)}",
                    line=getattr(
                        getattr(e, "problem_mark", None), "line", None
                    ),
                    column=getattr(
                        getattr(e, "problem_mark", None), "column", None
                    ),
                )
            )
            return {}, errors
        except Exception as e:
            errors.append(
                ValidationError(message=f"Unexpected parsing error: {str(e)}")
            )
            return {}, errors

    def extract_value(
        self,
        config: dict[str, Any],
        path: PathType,
        default: ConfigValue = None,
        required: bool = False,
    ) -> tuple[ConfigValue, ValidationError | None]:
        """
        Extract a value from nested configuration using path navigation.

        Args:
            config: Configuration dictionary
            path: Path to the desired value
            default: Default value if path not found
            required: Whether the path is required to exist

        Returns:
            Tuple of (extracted_value, error)
        """
        config_path = (
            ConfigPath(path) if not isinstance(path, ConfigPath) else path
        )

        try:
            current = config
            for component in config_path.components:
                if not isinstance(current, dict):
                    if required:
                        return None, ValidationError(
                            message=(
                                f"Path '{config_path}' not found: "
                                f"'{component}' is not a dictionary"
                            ),
                            field=str(config_path),
                        )
                    return default, None

                if component not in current:
                    if required:
                        return None, ValidationError(
                            message=f"Required path '{config_path}' not found",
                            field=str(config_path),
                            suggestions=[
                                f"Add '{component}:' to your configuration"
                            ],
                        )
                    return default, None

                current = current[component]

            return current, None

        except Exception as e:
            return None, ValidationError(
                message=f"Error extracting path '{config_path}': {str(e)}",
                field=str(config_path),
            )

    def extract_multiple_values(
        self,
        config: dict[str, Any],
        paths: dict[str, PathType],
        defaults: dict[str, ConfigValue] | None = None,
    ) -> tuple[dict[str, ConfigValue], list[ValidationError]]:
        """
        Extract multiple values from configuration efficiently.

        Args:
            config: Configuration dictionary
            paths: Dictionary mapping result keys to configuration paths
            defaults: Default values for each key

        Returns:
            Tuple of (extracted_values, errors)
        """
        results: dict[str, ConfigValue] = {}
        errors: list[ValidationError] = []
        defaults = defaults or {}

        for key, path in paths.items():
            default_value = defaults.get(key)
            value, error = self.extract_value(config, path, default_value)

            if error:
                errors.append(error)
            else:
                results[key] = value

        return results, errors

    def flatten_nested_config(
        self, config: dict[str, Any], separator: str = ".", max_depth: int = 10
    ) -> dict[str, ConfigValue]:
        """
        Flatten nested configuration into dot-notation keys.

        Args:
            config: Nested configuration dictionary
            separator: Separator for flattened keys
            max_depth: Maximum depth to prevent infinite recursion

        Returns:
            Flattened configuration dictionary
        """

        def _flatten_recursive(
            obj: Any, parent_key: str = "", depth: int = 0
        ) -> dict[str, ConfigValue]:
            if depth > max_depth:
                logger.warning(
                    "Maximum flattening depth reached for key: %s", parent_key
                )
                return {
                    parent_key: str(obj)
                }  # Convert to string to prevent recursion

            items: dict[str, ConfigValue] = {}

            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = (
                        f"{parent_key}{separator}{key}" if parent_key else key
                    )
                    items.update(_flatten_recursive(value, new_key, depth + 1))
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    new_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
                    items.update(_flatten_recursive(value, new_key, depth + 1))
            else:
                items[parent_key] = obj

            return items

        return _flatten_recursive(config)

    def unflatten_config(
        self, flat_config: dict[str, ConfigValue], separator: str = "."
    ) -> dict[str, Any]:
        """
        Convert flattened configuration back to nested structure.

        Args:
            flat_config: Flattened configuration dictionary
            separator: Separator used in flattened keys

        Returns:
            Nested configuration dictionary
        """
        result: dict[str, Any] = {}

        for key, value in flat_config.items():
            current = result
            parts = key.split(separator)

            for part in parts[:-1]:
                # Handle array indices
                if "[" in part and "]" in part:
                    base_key = part.split("[")[0]
                    if base_key not in current:
                        current[base_key] = []
                    current = current[base_key]
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

            # Set the final value
            final_key = parts[-1]
            if "[" in final_key and "]" in final_key:
                # Handle array assignment
                base_key = final_key.split("[")[0]
                index = int(final_key.split("[")[1].split("]")[0])
                if base_key not in current:
                    current[base_key] = []

                # Extend list if necessary
                while len(current[base_key]) <= index:
                    current[base_key].append(None)

                current[base_key][index] = value
            else:
                current[final_key] = value

        return result

    def validate_nested_types(
        self,
        config: dict[str, Any],
        type_schema: dict[str, type | tuple[type, ...]],
    ) -> list[ValidationError]:
        """
        Validate types in nested configuration against schema.

        Args:
            config: Configuration to validate
            type_schema: Schema defining expected types for paths

        Returns:
            List of validation errors
        """
        errors: list[ValidationError] = []

        for path, expected_type in type_schema.items():
            value, error = self.extract_value(config, path)

            if error:
                errors.append(error)
                continue

            if value is not None:
                if not isinstance(value, expected_type):
                    type_name = (
                        expected_type.__name__
                        if isinstance(expected_type, type)
                        else " or ".join(t.__name__ for t in expected_type)
                    )
                    errors.append(
                        ValidationError(
                            message=(
                                f"Type mismatch for '{path}': expected "
                                f"{type_name}, got {type(value).__name__}"
                            ),
                            field=path,
                            suggestions=[
                                f"Convert '{path}' to {type_name} type"
                            ],
                        )
                    )

        return errors

    def _process_nested_structure(
        self, config: dict[str, Any], errors: list[ValidationError]
    ) -> dict[str, Any]:
        """Process and validate nested configuration structure."""
        processed = {}

        for key, value in config.items():
            try:
                if isinstance(value, dict):
                    processed[key] = self._process_nested_structure(
                        value, errors
                    )
                elif isinstance(value, list):
                    processed[key] = self._process_list_structure(
                        value, errors
                    )
                else:
                    processed[key] = value
            except Exception as e:
                errors.append(
                    ValidationError(
                        message=f"Error processing key '{key}': {str(e)}",
                        field=key,
                    )
                )
                processed[key] = value

        return processed

    def _process_list_structure(
        self, config_list: list[Any], errors: list[ValidationError]
    ) -> list[Any]:
        """Process list structures in configuration."""
        processed = []

        for i, item in enumerate(config_list):
            try:
                if isinstance(item, dict):
                    processed.append(
                        self._process_nested_structure(item, errors)
                    )
                elif isinstance(item, list):
                    processed.append(
                        self._process_list_structure(item, errors)
                    )
                else:
                    processed.append(item)
            except Exception as e:
                errors.append(
                    ValidationError(
                        message=f"Error processing list item {i}: {str(e)}",
                        field=f"[{i}]",
                    )
                )
                processed.append(item)

        return processed

    def _resolve_interpolations(
        self,
        config: dict[str, Any],
        root_config: dict[str, Any],
        errors: list[ValidationError],
    ) -> dict[str, Any]:
        """Resolve interpolation patterns in configuration."""
        resolved = {}

        for key, value in config.items():
            try:
                if isinstance(value, str):
                    resolved[key] = self._resolve_string_interpolation(
                        value, root_config, errors
                    )
                elif isinstance(value, dict):
                    resolved[key] = self._resolve_interpolations(
                        value, root_config, errors
                    )
                elif isinstance(value, list):
                    resolved[key] = [
                        (
                            self._resolve_string_interpolation(
                                item, root_config, errors
                            )
                            if isinstance(item, str)
                            else item
                        )
                        for item in value
                    ]
                else:
                    resolved[key] = value
            except Exception as e:
                errors.append(
                    ValidationError(
                        message=(
                            f"Error resolving interpolation for '{key}': "
                            f"{str(e)}"
                        ),
                        field=key,
                    )
                )
                resolved[key] = value

        return resolved

    def _resolve_string_interpolation(
        self,
        value: str,
        root_config: dict[str, Any],
        errors: list[ValidationError],
    ) -> ConfigValue:
        """Resolve interpolation patterns in string values."""
        # Simple ${path} interpolation pattern
        import re

        pattern = r"\$\{([^}]+)\}"

        def replace_interpolation(match: re.Match[str]) -> str:
            path = match.group(1)
            resolved_value, error = self.extract_value(root_config, path)

            if error:
                errors.append(
                    ValidationError(
                        message=(
                            f"Failed to resolve interpolation '${{{path}}}': "
                            f"{error.message}"
                        ),
                        field=path,
                    )
                )
                return match.group(0)  # Return original if resolution fails

            return str(resolved_value) if resolved_value is not None else ""

        resolved = re.sub(pattern, replace_interpolation, value)

        # Try to convert to appropriate type if the entire string was an
        # interpolation
        if (
            resolved != value
            and resolved.replace(".", "").replace("-", "").isdigit()
        ):
            try:
                return float(resolved) if "." in resolved else int(resolved)
            except ValueError:
                pass

        return resolved

    def _apply_type_conversions(
        self, config: dict[str, Any], errors: list[ValidationError]
    ) -> dict[str, Any]:
        """Apply intelligent type conversions to configuration values."""
        converted = {}

        for key, value in config.items():
            try:
                if isinstance(value, dict):
                    converted[key] = self._apply_type_conversions(
                        value, errors
                    )
                elif isinstance(value, str) and key in self.type_converters:
                    converter = self.type_converters[key]
                    converted[key] = converter(value)
                else:
                    converted[key] = value
            except Exception as e:
                errors.append(
                    ValidationError(
                        message=f"Type conversion error for '{key}': {str(e)}",
                        field=key,
                    )
                )
                converted[key] = value

        return converted

    def _initialize_type_converters(self) -> dict[str, Any]:
        """Initialize type converters for common configuration fields."""

        def str_to_bool(value: str) -> bool:
            return value.lower() in ("true", "yes", "1", "on")

        def str_to_int_list(value: str) -> list[int]:
            return [int(x.strip()) for x in value.split(",")]

        def str_to_float_list(value: str) -> list[float]:
            return [float(x.strip()) for x in value.split(",")]

        return {
            "batch_size": int,
            "epochs": int,
            "num_workers": int,
            "learning_rate": float,
            "weight_decay": float,
            "dropout": float,
            "image_size": str_to_int_list,
            "mean": str_to_float_list,
            "std": str_to_float_list,
            "use_amp": str_to_bool,
            "shuffle": str_to_bool,
            "pin_memory": str_to_bool,
        }

    def _initialize_interpolation_patterns(self) -> dict[str, str]:
        """Initialize common interpolation patterns for crack segmentation."""
        return {
            "data_root": "${oc.env:DATA_ROOT,./data}",
            "output_dir": "${oc.env:OUTPUT_DIR,./outputs}",
            "model_name": "${model.architecture}_${model.encoder}",
            "experiment_name": "${model_name}_${now:%Y%m%d_%H%M%S}",
        }


# Global parser instance
_yaml_parser = AdvancedYAMLParser()


def parse_nested_config(
    content: str, enable_interpolation: bool = True
) -> tuple[dict[str, Any], list[ValidationError]]:
    """
    Convenience function for parsing nested YAML configurations.

    Args:
        content: YAML content as string
        enable_interpolation: Whether to resolve interpolation patterns

    Returns:
        Tuple of (parsed_config, parsing_errors)
    """
    return _yaml_parser.parse_config(content, enable_interpolation)


def extract_config_value(
    config: dict[str, Any], path: str, default: ConfigValue = None
) -> ConfigValue:
    """
    Convenience function for extracting values from nested configurations.

    Args:
        config: Configuration dictionary
        path: Dot-separated path to value
        default: Default value if path not found

    Returns:
        Extracted value or default
    """
    value, error = _yaml_parser.extract_value(config, path, default)
    return value if error is None else default
