"""
E2E Testing Modules Package.

This package contains modular components for end-to-end testing of the
CrackSeg pipeline, organized by functionality.

Available Modules:
- checkpointing: Checkpoint management and validation
- config: Configuration generation and management
- dataclasses: Data structures for E2E testing
- data: Data generation and loading utilities
- evaluation: Model evaluation and metrics
- setup: Initialization and setup utilities
- training: Training loop and component initialization
- utils: General utility functions

Usage:
    from scripts.experiments.e2e.modules import config, training, evaluation
"""

from . import (
    checkpointing,
    config,
    data,
    dataclasses,
    evaluation,
    setup,
    training,
    utils,
)

__all__ = [
    "checkpointing",
    "config",
    "dataclasses",
    "data",
    "evaluation",
    "setup",
    "training",
    "utils",
]
