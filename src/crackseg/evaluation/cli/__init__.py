"""Command-line interface for evaluation module.

This module provides CLI functionality for evaluation operations including
environment setup, configuration handling, and evaluation execution.
"""

from .components import get_evaluation_components, load_models_for_evaluation
from .config import load_and_prepare_config, validate_evaluation_config
from .environment import setup_evaluation_environment, setup_output_directory
from .runner import (
    EvaluationRunParameters,
    run_evaluation_and_log,
    run_single_evaluation,
)

__all__ = [
    # Environment setup
    "setup_evaluation_environment",
    "setup_output_directory",
    # Configuration handling
    "load_and_prepare_config",
    "validate_evaluation_config",
    # Component preparation
    "get_evaluation_components",
    "load_models_for_evaluation",
    # Evaluation execution
    "EvaluationRunParameters",
    "run_evaluation_and_log",
    "run_single_evaluation",
]
