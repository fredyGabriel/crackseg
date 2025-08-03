"""Legacy visualization modules.

This module contains legacy visualization files that have been
refactored into specialized modules. These files are kept for
backward compatibility but should not be used in new code.
"""

# Legacy imports for backward compatibility
from .experiment_viz import ExperimentVisualizer as LegacyExperimentVisualizer
from .learning_rate_analysis import (
    LearningRateAnalyzer as LegacyLearningRateAnalyzer,
)
from .parameter_analysis import ParameterAnalyzer as LegacyParameterAnalyzer
from .prediction_viz import PredictionVisualizer as LegacyPredictionVisualizer

__all__ = [
    "LegacyExperimentVisualizer",
    "LegacyParameterAnalyzer",
    "LegacyPredictionVisualizer",
    "LegacyLearningRateAnalyzer",
]
