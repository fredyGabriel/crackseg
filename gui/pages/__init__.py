"""
GUI Pages for CrackSeg Application

This package contains all GUI pages organized in a modular structure:

- core/: Core application pages (home, navigation)
- ml/: ML-specific pages (training, config, architecture)
- data/: Data-related pages (results, analysis)
- deprecated/: Obsolete pages (for removal)

All pages follow ML project best practices with type safety,
error handling, and user experience optimization.
"""

# Core pages
from .core import (
    page_home,
)

# Data pages
from .data import (
    page_results,
    page_results_legacy,
)

# ML pages
from .ml import (
    page_advanced_config,
    page_architecture,
    page_config,
    page_train,
    page_train_legacy,
)

__all__ = [
    # Core
    "page_home",
    # ML
    "page_train",
    "page_train_legacy",
    "page_config",
    "page_advanced_config",
    "page_architecture",
    # Data
    "page_results",
    "page_results_legacy",
]
