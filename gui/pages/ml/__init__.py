"""
ML-specific pages for the CrackSeg application.

This module contains pages for training workflows, configuration management,
and model architecture visualization.
"""

from .architecture.main import page_architecture
from .config.advanced import page_advanced_config
from .config.basic import page_config
from .training.legacy import page_train as page_train_legacy
from .training.main import page_train

__all__ = [
    "page_train",
    "page_train_legacy",
    "page_config",
    "page_advanced_config",
    "page_architecture",
]
