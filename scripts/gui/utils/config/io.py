"""Core I/O operations for configuration files.

This module provides the fundamental file I/O operations for loading, scanning,
and managing YAML configuration files with proper error handling and caching.
"""

import logging
from datetime import datetime
from pathlib import Path

import yaml

from .cache import _config_cache  # pyright: ignore[reportPrivateUsage]
from .exceptions import ConfigError, ValidationError

logger = logging.getLogger(__name__)


def load_config_file(path: str | Path) -> dict[str, object]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the parsed configuration.

    Raises:
        ConfigError: If the file cannot be loaded or parsed.
    """
    path_str = str(path)

    # Check cache first
    cached_config = _config_cache.get(path_str)
    if cached_config is not None:
        logger.debug(f"Loaded config from cache: {path_str}")
        return cached_config

    try:
        with open(path_str, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        # Cache the loaded configuration
        _config_cache.set(path_str, config)
        logger.debug(f"Loaded and cached config: {path_str}")

        return config

    except FileNotFoundError as e:
        raise ConfigError(f"Configuration file not found: {path_str}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path_str}: {e}") from e
    except Exception as e:
        raise ConfigError(f"Error loading {path_str}: {e}") from e


def scan_config_directories() -> dict[str, list[str]]:
    """Scan configuration directories for available YAML files.

    Scans both the configs/ directory and generated_configs/ directory
    (if it exists) for YAML configuration files.

    Returns:
        Dictionary mapping category names to lists of config file paths.
    """
    config_dirs = {
        "configs": Path("configs"),
        "generated_configs": Path("generated_configs"),
    }

    categorized_configs: dict[str, list[str]] = {}

    for base_name, base_dir in config_dirs.items():
        if not base_dir.exists():
            continue

        # Scan for YAML files
        for yaml_file in base_dir.rglob("*.yaml"):
            # Skip __pycache__ directories
            if "__pycache__" in str(yaml_file):
                continue

            # Determine category based on path
            relative_path = yaml_file.relative_to(base_dir)
            parts = relative_path.parts

            if len(parts) > 1:
                # File is in a subdirectory, use that as category
                category = f"{base_name}/{parts[0]}"
            else:
                # File is in root directory
                category = base_name

            if category not in categorized_configs:
                categorized_configs[category] = []

            categorized_configs[category].append(str(yaml_file))

    # Sort file lists for consistent ordering
    for category in categorized_configs:
        categorized_configs[category].sort()

    return categorized_configs


def get_config_metadata(
    path: str | Path,
) -> dict[str, str | bool | list[str] | int | None]:
    """Get metadata about a configuration file.

    Args:
        path: Path to the configuration file.

    Returns:
        Dictionary containing file metadata.
    """
    path = Path(path)
    metadata: dict[str, str | bool | list[str] | int | None] = {
        "path": str(path),
        "name": path.name,
        "exists": path.exists(),
    }

    if path.exists():
        stat = path.stat()
        metadata["size"] = stat.st_size
        metadata["modified"] = datetime.fromtimestamp(
            stat.st_mtime
        ).isoformat()
        metadata["size_human"] = _format_file_size(stat.st_size)

        # Try to get first few lines for preview
        try:
            lines: list[str] = []
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 5:  # First 5 lines
                        break
                    lines.append(line.rstrip())
            metadata["preview"] = lines
        except Exception:
            metadata["preview"] = []

    return metadata


def load_and_validate_config(
    path: str | Path,
) -> tuple[dict[str, object], list[ValidationError]]:
    """Load and validate a configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Tuple of (config_dict, validation_errors).

    Raises:
        ConfigError: If the file cannot be loaded.
    """
    from .validation import validate_yaml_advanced

    # Load the configuration
    config = load_config_file(path)

    # Convert to string for validation
    try:
        config_str = yaml.dump(config, default_flow_style=False)
        is_valid, errors = validate_yaml_advanced(config_str)
        return config, errors
    except Exception:
        # If we can't serialize back to YAML, just return basic validation
        from .validation import (
            validate_config_structure,
            validate_config_types,
            validate_config_values,
        )

        structure_valid, structure_errors = validate_config_structure(config)
        types_valid, type_errors = validate_config_types(config)
        values_valid, value_errors = validate_config_values(config)

        all_errors = structure_errors + type_errors + value_errors
        return config, all_errors


def _format_file_size(size: int) -> str:
    """Format file size in human-readable format.

    Args:
        size: File size in bytes.

    Returns:
        Human-readable file size string.
    """
    size_f = float(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_f < 1024.0:
            return f"{size_f:.1f} {unit}"
        size_f /= 1024.0
    return f"{size_f:.1f} TB"
