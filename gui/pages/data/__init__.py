"""
Data-related pages for the CrackSeg application.

This module contains pages for results visualization, data analysis,
and training outcomes display.
"""

from .results.legacy import page_results as page_results_legacy
from .results.main import page_results

__all__ = [
    "page_results",
    "page_results_legacy",
]
