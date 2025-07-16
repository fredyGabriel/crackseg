"""Unit tests for the configuration I/O operations."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from gui.utils.config.cache import _config_cache
from gui.utils.config.exceptions import ConfigError
from gui.utils.config.io import (
    get_config_metadata,
    load_config_file,
    scan_config_directories,
)


@pytest.fixture(autouse=True)
def clear_cache_before_test():
    """Fixture to clear the config cache before each test."""
    _config_cache.clear()
    yield


def test_load_config_file_success(tmp_path: Path) -> None:
    """Test successful loading of a real YAML configuration file."""
    config_path = tmp_path / "config.yaml"
    expected_config: dict[str, object] = {"key": "value", "number": 123}
    config_path.write_text(yaml.dump(expected_config))

    config = load_config_file(config_path)
    assert config == expected_config


def test_load_config_file_not_found() -> None:
    """Test that loading a non-existent file raises ConfigError."""
    mock_path = "/non/existent/path.yaml"
    # No need to patch open, as it will raise FileNotFoundError naturally
    # if the file doesn't exist. This makes the test more realistic.
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_config_file(mock_path)


def test_load_config_file_invalid_yaml(tmp_path: Path) -> None:
    """Test that invalid YAML raises ConfigError."""
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("key: value: [invalid")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config_file(config_path)


def test_load_config_file_uses_cache(tmp_path: Path) -> None:
    """Test that a configuration is loaded from cache on the second call."""
    config_path = tmp_path / "cache_config.yaml"
    expected_config: dict[str, object] = {"key": "cached_value"}
    config_path.write_text(yaml.dump(expected_config))

    # First call - should load from disk and populate the cache
    config1 = load_config_file(config_path)
    assert config1 == expected_config

    # To verify the cache is used, we patch `open` *after* the first call
    # and assert it's not called again.
    with patch("builtins.open") as m_open:
        config2 = load_config_file(config_path)
        assert config2 == expected_config
        m_open.assert_not_called()


@pytest.mark.skip(
    reason="Needs refactoring of scan_config_directories to be testable"
)
def test_scan_config_directories(tmp_path: Path) -> None:
    """Test scanning of configuration directories."""
    # Create dummy directory structure
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "base.yaml").touch()
    (configs_dir / "models").mkdir()
    (configs_dir / "models" / "model1.yaml").touch()
    (configs_dir / "__pycache__").mkdir()
    (configs_dir / "__pycache__" / "cache.yaml").touch()

    gen_configs_dir = tmp_path / "generated_configs"
    gen_configs_dir.mkdir()
    (gen_configs_dir / "generated1.yaml").touch()

    # To test this properly, scan_config_directories should be refactored
    # to accept base directories as arguments.
    # For now, this test is skipped.
    result = scan_config_directories()  # This will scan the real project dirs
    # Assertions would go here if the function was testable
    assert result is not None


def test_get_config_metadata_exists(tmp_path: Path) -> None:
    """Test metadata retrieval for an existing file."""
    file_path = tmp_path / "test.yaml"
    file_content = "key: value"
    file_path.write_text(file_content)

    # Allow some time for file system to register the write
    time.sleep(0.1)

    metadata = get_config_metadata(file_path)

    assert metadata["path"] == str(file_path)
    assert metadata["name"] == "test.yaml"
    assert metadata["exists"] is True
    assert metadata["size"] == len(file_content.encode("utf-8"))
    assert "modified" in metadata
    assert isinstance(metadata["modified"], str)


def test_get_config_metadata_not_exists() -> None:
    """Test metadata retrieval for a non-existent file."""
    file_path = Path("/non/existent/file.yaml")

    metadata = get_config_metadata(file_path)

    assert metadata["path"] == str(file_path)
    assert metadata["exists"] is False
    assert "size" not in metadata
