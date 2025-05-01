"""Configuration file for pytest."""

import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA capable device"
    )
