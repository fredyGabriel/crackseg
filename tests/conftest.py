"""Configuration file for pytest."""

import sys
from pathlib import Path
import pytest

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA capable device"
    )
    # Register hydra marker
    config.addinivalue_line(
        "markers", "hydra: mark test as using hydra for configuration"
    )


# Added fixture
@pytest.fixture(scope='session')
def hydra_config_dir():
    """Provides the absolute path to the Hydra configuration directory."""
    # Path(__file__).parent is tests/
    # Path(__file__).parent.parent is the project root (crackseg/)
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs"
    return str(config_path.resolve())
