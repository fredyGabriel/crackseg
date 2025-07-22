"""
Pytest configuration for GUI component tests. Provides common fixtures
and utilities for testing GUI components.
"""

from pathlib import Path

import pytest


@pytest.fixture
def sample_project_root(tmp_path: Path) -> Path:
    """Create temporary project root for testing."""
    project_root = tmp_path / "project"
    project_root.mkdir()

    # Create necessary subdirectories for realistic testing
    (project_root / "configs").mkdir()
    (project_root / "generated_configs").mkdir()
    (project_root / "outputs").mkdir()
    (project_root / "assets").mkdir()
    (project_root / "assets" / "images").mkdir()

    # Create a sample config file
    sample_config = project_root / "configs" / "base.yaml"
    sample_config.write_text("model:\n  name: test_model\n")

    # Create a sample logo file
    sample_logo = project_root / "assets" / "images" / "logo.png"
    sample_logo.write_bytes(b"fake_png_data")

    return project_root
