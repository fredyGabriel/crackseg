"""
File I/O utilities for E2E testing. This module provides utilities for
file operations, configuration management, and test artifact handling
for the CrackSeg testing framework.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml


def ensure_artifacts_dir(artifacts_path: Path | str) -> Path:
    """Ensure artifacts directory exists and return Path object.

    Args:
        artifacts_path: Path to artifacts directory

    Returns:
        Path object for artifacts directory

    Example:
        >>> artifacts_dir = ensure_artifacts_dir("test-artifacts")
        >>> artifacts_dir.exists()
        True
    """
    path = Path(artifacts_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_exists(file_path: Path | str) -> bool:
    """Validate that a file exists and is readable.

    Args:
        file_path: Path to file to validate

    Returns:
        True if file exists and is readable, False otherwise

    Example:
        >>> valid = validate_file_exists("configs/base.yaml")
        >>> isinstance(valid, bool)
        True
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except (OSError, PermissionError):
        return False


def read_test_config(config_path: Path | str) -> dict[str, Any]:
    """Read and parse test configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If configuration file not found
        ValueError: If configuration file format is invalid

    Example:
        >>> config = read_test_config("configs/test.yaml")
        >>> isinstance(config, dict)
        True
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            if path.suffix.lower() in [".yaml", ".yml"]:
                return yaml.safe_load(file) or {}
            elif path.suffix.lower() == ".json":
                return json.load(file) or {}
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid configuration format in {path}: {e}") from e


def create_temp_config_file(
    config_data: dict[str, Any],
    file_format: str = "yaml",
    prefix: str = "test_config_",
) -> Path:
    """Create temporary configuration file for testing.

    Args:
        config_data: Configuration data to write
        file_format: Format for config file ("yaml" or "json")
        prefix: Prefix for temporary filename

    Returns:
        Path to created temporary config file

    Example:
        >>> config = {"model": {"type": "unet"}}
        >>> temp_path = create_temp_config_file(config)
        >>> temp_path.exists()
        True
    """
    suffix = f".{file_format.lower()}"

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        prefix=prefix,
        delete=False,
        encoding="utf-8",
    ) as tmp_file:
        if file_format.lower() in ["yaml", "yml"]:
            yaml.dump(config_data, tmp_file, default_flow_style=False)
        elif file_format.lower() == "json":
            json.dump(config_data, tmp_file, indent=2)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        return Path(tmp_file.name)


def get_test_config_path(config_name: str = "base.yaml") -> Path:
    """Get path to test configuration file.

    Args:
        config_name: Name of configuration file

    Returns:
        Path to configuration file

    Example:
        >>> config_path = get_test_config_path("test.yaml")
        >>> config_path.name == "test.yaml"
        True
    """
    # Look for config in standard locations
    possible_paths = [
        Path("configs") / config_name,
        Path("tests/configs") / config_name,
        Path("test-configs") / config_name,
        Path(config_name),  # Current directory
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Return first path as default (may not exist)
    return possible_paths[0]


def save_test_artifacts(
    artifacts: dict[str, Any],
    artifacts_dir: Path | str,
    test_name: str,
) -> dict[str, Path]:
    """Save test artifacts to specified directory.

    Args:
        artifacts: Dictionary of artifacts to save
        artifacts_dir: Directory to save artifacts
        test_name: Name of test for organizing artifacts

    Returns:
        Dictionary mapping artifact names to saved file paths

    Example:
        >>> artifacts = {"config": {"model": "unet"}, "log": "Test completed"}
        >>> saved = save_test_artifacts(artifacts, "test-artifacts", "test_1")
        >>> len(saved) > 0
        True
    """
    base_dir = ensure_artifacts_dir(artifacts_dir)
    test_dir = base_dir / test_name
    test_dir.mkdir(exist_ok=True)

    saved_paths: dict[str, Path] = {}

    for artifact_name, artifact_data in artifacts.items():
        # Determine file extension based on data type
        if isinstance(artifact_data, dict):
            file_path = test_dir / f"{artifact_name}.json"
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(artifact_data, f, indent=2)
        elif isinstance(artifact_data, str):
            file_path = test_dir / f"{artifact_name}.txt"
            with file_path.open("w", encoding="utf-8") as f:
                f.write(artifact_data)
        elif isinstance(artifact_data, bytes):
            file_path = test_dir / f"{artifact_name}.bin"
            with file_path.open("wb") as f:
                f.write(artifact_data)
        else:
            # Try to convert to string
            file_path = test_dir / f"{artifact_name}.txt"
            with file_path.open("w", encoding="utf-8") as f:
                f.write(str(artifact_data))

        saved_paths[artifact_name] = file_path

    return saved_paths


def cleanup_test_files(file_paths: list[Path | str]) -> None:
    """Clean up temporary test files and directories.

    Args:
        file_paths: List of file or directory paths to clean up

    Example:
        >>> temp_file = Path("temp_test.txt")
        >>> temp_file.write_text("test")
        >>> cleanup_test_files([temp_file])
        >>> temp_file.exists()
        False
    """
    for path_item in file_paths:
        path = Path(path_item)
        try:
            if path.exists():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
        except (OSError, PermissionError) as e:
            # Log warning but don't fail test
            print(f"Warning: Could not clean up {path}: {e}")


def copy_test_data_files(
    source_dir: Path | str,
    dest_dir: Path | str,
    file_patterns: list[str] | None = None,
) -> list[Path]:
    """Copy test data files from source to destination.

    Args:
        source_dir: Source directory containing test files
        dest_dir: Destination directory for copied files
        file_patterns: List of file patterns to match
            (e.g., ["*.jpg", "*.png"])

    Returns:
        List of copied file paths

    Example:
        >>> copied = copy_test_data_files("data/test", "temp/test", ["*.jpg"])
        >>> all(path.suffix == ".jpg" for path in copied)
        True
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    dest_path.mkdir(parents=True, exist_ok=True)
    copied_files: list[Path] = []

    patterns = file_patterns or ["*"]

    for pattern in patterns:
        for source_file in source_path.glob(pattern):
            if source_file.is_file():
                dest_file = dest_path / source_file.name
                shutil.copy2(source_file, dest_file)
                copied_files.append(dest_file)

    return copied_files


def create_test_directory_structure(base_dir: Path | str) -> dict[str, Path]:
    """Create standard test directory structure.

    Args:
        base_dir: Base directory for creating structure

    Returns:
        Dictionary mapping directory names to paths

    Example:
        >>> dirs = create_test_directory_structure("test-env")
        >>> "artifacts" in dirs
        True
    """
    base_path = Path(base_dir)

    directories = {
        "artifacts": base_path / "artifacts",
        "configs": base_path / "configs",
        "data": base_path / "data",
        "screenshots": base_path / "screenshots",
        "logs": base_path / "logs",
        "temp": base_path / "temp",
    }

    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories


def get_file_size_mb(file_path: Path | str) -> float:
    """Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in megabytes

    Example:
        >>> size = get_file_size_mb("configs/base.yaml")
        >>> size >= 0
        True
    """
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            return path.stat().st_size / (1024 * 1024)
        return 0.0
    except (OSError, PermissionError):
        return 0.0


def find_project_root(marker_files: list[str] | None = None) -> Path:
    """Find project root directory by looking for marker files.

    Args:
        marker_files: List of files that indicate project root

    Returns:
        Path to project root directory

    Example:
        >>> root = find_project_root()
        >>> root.name == "crackseg"
        True
    """
    markers = marker_files or [
        "pyproject.toml",
        "setup.py",
        "environment.yml",
        ".git",
        "src",
    ]

    current = Path.cwd()

    # Walk up directory tree looking for markers
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent

    # Default to current directory if no markers found
    return current
