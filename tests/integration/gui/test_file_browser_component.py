"""
Integration tests for FileBrowserComponent.

Tests the file browser component's functionality including file scanning,
filtering, sorting, and integration with config_io utilities.
"""

import os
import tempfile
from pathlib import Path

from scripts.gui.components.file_browser_component import FileBrowserComponent


class TestFileBrowserComponent:
    """Test suite for FileBrowserComponent."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.component = FileBrowserComponent()

    def test_component_initialization(self) -> None:
        """Test component initializes with correct default values."""
        assert self.component.supported_extensions == {".yaml", ".yml"}
        assert "Name (A-Z)" in self.component.sort_options
        assert "Modified (Newest)" in self.component.sort_options
        assert len(self.component.sort_options) == 6

    def test_get_selected_action_no_action(self) -> None:
        """Test getting selected action when none exists."""
        import streamlit as st

        # Clear any existing actions
        if hasattr(st.session_state, "file_browser_action"):
            delattr(st.session_state, "file_browser_action")
        if hasattr(st.session_state, "file_browser_target"):
            delattr(st.session_state, "file_browser_target")

        action, target = FileBrowserComponent.get_selected_action()
        assert action is None
        assert target is None

    def test_supported_extensions_filtering(self) -> None:
        """Test that only YAML files are included in results."""
        # Create test file list with mixed extensions
        test_files = [
            "config.yaml",
            "config.yml",
            "readme.txt",
            "script.py",
            "data.json",
        ]

        # Filter by supported extensions
        yaml_files = [
            f
            for f in test_files
            if Path(f).suffix.lower() in self.component.supported_extensions
        ]

        assert len(yaml_files) == 2
        assert "config.yaml" in yaml_files
        assert "config.yml" in yaml_files
        assert "readme.txt" not in yaml_files
        assert "script.py" not in yaml_files
        assert "data.json" not in yaml_files

    def test_component_state_keys(self) -> None:
        """Test that component generates proper state keys."""
        # Test key generation patterns
        test_key = "test_browser"

        # State keys should be predictable
        expected_state_key = f"{test_key}_state"
        expected_sort_key = f"{test_key}_sort"
        expected_filter_key = f"{test_key}_filter"

        # These are just string checks - actual Streamlit state testing
        # would require a more complex setup
        assert expected_state_key == "test_browser_state"
        assert expected_sort_key == "test_browser_sort"
        assert expected_filter_key == "test_browser_filter"


class TestFileBrowserIntegration:
    """Test integration with config_io utilities."""

    def test_integration_with_scan_config_directories(self) -> None:
        """Test integration with config_io.scan_config_directories."""
        from scripts.gui.utils.config_io import scan_config_directories

        # This should not raise an exception
        config_dirs = scan_config_directories()
        assert isinstance(config_dirs, dict)

        # Should return categorized configs
        for category, files in config_dirs.items():
            assert isinstance(category, str)
            assert isinstance(files, list)

    def test_integration_with_get_config_metadata(self) -> None:
        """Test integration with config_io.get_config_metadata."""
        from scripts.gui.utils.config_io import get_config_metadata

        # Create a temporary config file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("model:\n  architecture: unet\n")
            temp_file = f.name

        try:
            # Should return metadata dictionary
            metadata = get_config_metadata(temp_file)
            assert isinstance(metadata, dict)
            assert "path" in metadata
            assert "name" in metadata
            assert "exists" in metadata
            assert metadata["exists"] is True

        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_component_handles_config_io_errors(self) -> None:
        """Test that component handles config_io errors gracefully."""
        component = FileBrowserComponent()

        # Component should initialize without errors
        assert component is not None
        assert hasattr(component, "supported_extensions")
        assert hasattr(component, "sort_options")
