"""Analysis utilities for CrackSeg project.

This package contains utilities for analyzing project components:
- Import inventory
- Code analysis
- Training imports analysis
"""

from .inventory_training_imports import main as inventory_imports

__all__ = [
    "inventory_imports",
]
