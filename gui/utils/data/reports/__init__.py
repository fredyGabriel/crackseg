"""
Results scanning package for crack segmentation prediction triplets.
This package provides async scanning capabilities for handling large
crack segmentation datasets with triplet validation
(image|mask|prediction). Note: This module import s classes from the
correct locations where they are actually defined. Public API:
ResultTriplet: Triplet data structure (from results.core)
ScanProgress: Progress tracking (from results.core) TripletType:
Enumeration for triplet types (from results.core)
"""

# Import from the actual location where classes are defined
from gui.utils.results.core import (
    ResultTriplet,
    ScanProgress,
    TripletType,
)
from gui.utils.results.scanner import (
    AsyncResultsScanner,
    create_results_scanner,
)

# Note: ResultsScanner and ResultsScannerConfig are not available in the core
# module

__all__ = [
    "ResultTriplet",
    "ScanProgress",
    "TripletType",
    "AsyncResultsScanner",
    "create_results_scanner",
]
