"""Core I/O operations for configuration files.

This module provides the fundamental file I/O operations for loading, scanning,
and managing YAML configuration files with proper error handling and caching.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from .cache import _config_cache
from .exceptions import ConfigError, ValidationError

logger = logging.getLogger(__name__)

# File upload constraints
MAX_UPLOAD_SIZE_MB = 10  # Maximum upload size in MB
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".yaml", ".yml"}


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


def upload_config_file(
    uploaded_file: Any,
    target_directory: str | Path = "generated_configs",
    validate_on_upload: bool = True,
) -> tuple[str, dict[str, Any], list[ValidationError]]:
    """Upload and process a YAML configuration file from user's local system.

    Args:
        uploaded_file: Streamlit uploaded file object.
        target_directory: Directory where the file should be saved.
        validate_on_upload: Whether to validate the file during upload.

    Returns:
        Tuple of (saved_file_path, config_dict, validation_errors).

    Raises:
        ConfigError: If the file cannot be processed or saved.
    """
    # Validate file size
    if uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
        size_mb = uploaded_file.size / (1024 * 1024)
        raise ConfigError(
            f"File size ({size_mb:.1f} MB) exceeds maximum allowed "
            f"size of {MAX_UPLOAD_SIZE_MB} MB"
        )

    # Validate file extension
    file_path = Path(uploaded_file.name)
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ConfigError(
            f"Invalid file extension '{file_path.suffix}'. "
            f"Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Create target directory if it doesn't exist
    target_dir = Path(target_directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename with timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = file_path.stem
    extension = file_path.suffix
    unique_filename = f"{timestamp}_{base_name}{extension}"
    target_path = target_dir / unique_filename

    try:
        # Read and parse YAML content
        content = uploaded_file.read()

        # Try to decode as UTF-8
        try:
            content_str = content.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ConfigError(
                "File encoding error. Please ensure the file is UTF-8 "
                f"encoded: {e}"
            ) from e

        # Parse YAML to validate syntax
        try:
            config_dict = yaml.safe_load(content_str)
            if config_dict is None:
                config_dict = {}
        except yaml.YAMLError as e:
            # Get line and column from problem_mark if it exists
            problem_mark = getattr(e, "problem_mark", None)
            line = (
                getattr(problem_mark, "line", 0)
                if problem_mark is not None
                else 0
            )
            column = (
                getattr(problem_mark, "column", 0)
                if problem_mark is not None
                else 0
            )
            config_dict = {}
            validation_errors = [
                ValidationError(
                    f"YAML syntax error: {e}",
                    line=line,
                    column=column,
                )
            ]
            return str(target_path), config_dict, validation_errors

        # Validate configuration if requested
        validation_errors: list[ValidationError] = []
        if validate_on_upload:
            try:
                from .validation import validate_yaml_advanced

                is_valid, errors = validate_yaml_advanced(content_str)
                validation_errors = errors
                if not is_valid:
                    logger.warning(
                        f"Uploaded file {uploaded_file.name} has validation "
                        f"issues: {len(errors)} errors found"
                    )
                else:
                    logger.info(
                        f"Uploaded file {uploaded_file.name} passed validation"
                    )
            except Exception as e:
                logger.warning(f"Validation failed for uploaded file: {e}")
                validation_errors = [
                    ValidationError(
                        f"Validation check failed: {e}", line=0, column=0
                    )
                ]

        # Save file to target location
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content_str)

        # Cache the configuration
        _config_cache.set(str(target_path), config_dict)

        logger.info(
            f"Successfully uploaded and saved config file: {target_path}"
        )

        return str(target_path), config_dict, validation_errors

    except Exception as e:
        # Clean up partial file if it was created
        if target_path.exists():
            try:
                target_path.unlink()
            except Exception:
                pass  # Ignore cleanup errors

        if isinstance(e, ConfigError):
            raise
        else:
            raise ConfigError(f"Error processing uploaded file: {e}") from e


def create_upload_progress_placeholder() -> Any:
    """Create a placeholder for upload progress indication.

    Returns:
        Streamlit placeholder object for progress updates.
    """
    return st.empty()


def update_upload_progress(
    placeholder: Any,
    stage: str,
    progress: float = 0.0,
    message: str = "",
) -> None:
    """Update upload progress indication.

    Args:
        placeholder: Streamlit placeholder object.
        stage: Current stage of upload (reading, validating, saving).
        progress: Progress value between 0.0 and 1.0.
        message: Additional message to display.
    """
    stage_icons = {
        "reading": "📖",
        "validating": "✅",
        "saving": "💾",
        "complete": "🎉",
        "error": "❌",
    }

    icon = stage_icons.get(stage, "⏳")

    if stage == "complete":
        placeholder.success(f"{icon} Upload complete! {message}")
    elif stage == "error":
        placeholder.error(f"{icon} Upload failed: {message}")
    else:
        if progress > 0:
            placeholder.info(
                f"{icon} {stage.title()}... {progress:.0%} {message}"
            )
        else:
            placeholder.info(f"{icon} {stage.title()}... {message}")


def validate_uploaded_content(
    content: str,
) -> tuple[bool, list[ValidationError]]:
    """Validate uploaded YAML content with comprehensive checks.

    Args:
        content: YAML content as string.

    Returns:
        Tuple of (is_valid, validation_errors).
    """
    validation_errors: list[ValidationError] = []

    try:
        # Basic YAML syntax check
        yaml.safe_load(content)
    except yaml.YAMLError as e:
        # Extraigo line y column de problem_mark si existen
        problem_mark = getattr(e, "problem_mark", None)
        line = (
            getattr(problem_mark, "line", 0) if problem_mark is not None else 0
        )
        column = (
            getattr(problem_mark, "column", 0)
            if problem_mark is not None
            else 0
        )
        validation_errors.append(
            ValidationError(
                f"YAML syntax error: {e}",
                line=line,
                column=column,
            )
        )
        return False, validation_errors

    # Additional validation using existing validation system
    try:
        from .validation import validate_yaml_advanced

        is_valid, errors = validate_yaml_advanced(content)
        validation_errors.extend(errors)
        return is_valid, validation_errors
    except Exception as e:
        validation_errors.append(
            ValidationError(
                f"Advanced validation failed: {e}",
                line=0,
                column=0,
            )
        )
        return len(validation_errors) == 0, validation_errors


def get_upload_file_info(uploaded_file: Any) -> dict[str, Any]:
    """Get information about an uploaded file.

    Args:
        uploaded_file: Streamlit uploaded file object.

    Returns:
        Dictionary containing file information.
    """
    return {
        "name": uploaded_file.name,
        "size": uploaded_file.size,
        "size_human": _format_file_size(uploaded_file.size),
        "type": getattr(uploaded_file, "type", None),
        "extension": Path(uploaded_file.name).suffix.lower(),
        "is_valid_extension": Path(uploaded_file.name).suffix.lower()
        in ALLOWED_EXTENSIONS,
        "is_valid_size": uploaded_file.size <= MAX_UPLOAD_SIZE_BYTES,
        "max_size_mb": MAX_UPLOAD_SIZE_MB,
    }
