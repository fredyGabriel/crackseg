"""Utility functions for evaluation module.

This module provides utility functions for evaluation operations including
data loading, results processing, and core evaluation functionality.
"""

from .core import EvaluationConfig, EvaluationCore
from .data import load_evaluation_data, process_evaluation_data
from .loading import load_evaluation_results, save_evaluation_results
from .results import EvaluationResults, process_results

__all__ = [
    # Core evaluation utilities
    "EvaluationCore",
    "EvaluationConfig",
    # Data utilities
    "load_evaluation_data",
    "process_evaluation_data",
    # Loading utilities
    "load_evaluation_results",
    "save_evaluation_results",
    # Results utilities
    "EvaluationResults",
    "process_results",
]
