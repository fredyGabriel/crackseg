"""Ensemble evaluation functionality.

This module provides ensemble evaluation capabilities for crack segmentation
models, including model combination strategies and ensemble prediction.
"""

from .ensemble import (
    EnsembleConfig,
    EnsembleEvaluator,
    ModelEnsemble,
    combine_predictions,
    ensemble_predict,
)

__all__ = [
    # Ensemble evaluation
    "EnsembleEvaluator",
    "ModelEnsemble",
    "EnsembleConfig",
    "ensemble_predict",
    "combine_predictions",
]
