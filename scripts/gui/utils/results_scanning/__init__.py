"""Results scanning package for crack segmentation prediction triplets.

This package provides async scanning capabilities for handling large crack
segmentation datasets with triplet validation (image|mask|prediction).

Refactored into focused modules following coding standards:
- core.py: Data structures and progress tracking
- scanner.py: Main AsyncResultsScanner class
- validation.py: Triplet validation logic
- utils.py: Factory functions and utilities

Public API:
    AsyncResultsScanner: Main scanner class
    ResultTriplet: Triplet data structure
    ScanProgress: Progress tracking
    create_results_scanner: Factory function
"""

from .core import ResultTriplet, ScanProgress, TripletType
from .scanner import AsyncResultsScanner
from .utils import create_results_scanner

__all__ = [
    "AsyncResultsScanner",
    "ResultTriplet",
    "ScanProgress",
    "TripletType",
    "create_results_scanner",
]
