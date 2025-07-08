"""Results scanning package for crack segmentation prediction triplets.

This package provides async scanning capabilities for handling large crack
segmentation datasets with triplet validation (image|mask|prediction).

Note: This module imports classes from the correct locations where they
are actually defined.

Public API:
    ResultTriplet: Triplet data structure (from results.core)
    ScanProgress: Progress tracking (from results.core)
    TripletType: Enumeration for triplet types (from results.core)
"""

# Import from the actual location where classes are defined
from scripts.gui.utils.results.core import (
    ResultTriplet,
    ScanProgress,
    TripletType,
)

# Note: AsyncResultsScanner and create_results_scanner are not yet implemented
# These would need to be created if this functionality is required

__all__ = [
    "ResultTriplet",
    "ScanProgress",
    "TripletType",
]
