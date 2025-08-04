"""Visualization module for crack segmentation evaluation.

This module provides various visualization capabilities for training
curves, prediction results, and model analysis.
"""

# Core visualization components
from .advanced_prediction_viz import AdvancedPredictionVisualizer

# Specialized modules
from .analysis import ParameterAnalyzer, PredictionAnalyzer
from .experiment import ExperimentPlotter, ExperimentVisualizer
from .interactive_plotly import InteractivePlotlyVisualizer

# Legacy imports for backward compatibility
from .legacy import (
    LegacyExperimentVisualizer,
    LegacyLearningRateAnalyzer,
    LegacyParameterAnalyzer,
    LegacyPredictionVisualizer,
)
from .prediction import (
    ConfidenceMapVisualizer,
    PredictionGridVisualizer,
    SegmentationOverlayVisualizer,
)
from .training import (
    AdvancedTrainingVisualizer,
    ParameterAnalysisVisualizer,
    TrainingCurvesVisualizer,
    TrainingReportGenerator,
)

# Backward compatibility aliases
PredictionConfidenceVisualizer = ConfidenceMapVisualizer
PredictionOverlayVisualizer = SegmentationOverlayVisualizer

__all__ = [
    # Core components
    "AdvancedPredictionVisualizer",
    # Analysis modules
    "ParameterAnalyzer",
    "PredictionAnalyzer",
    # Experiment modules
    "ExperimentVisualizer",
    "ExperimentPlotter",
    # Interactive modules
    "InteractivePlotlyVisualizer",
    # Prediction modules
    "PredictionGridVisualizer",
    "ConfidenceMapVisualizer",
    "SegmentationOverlayVisualizer",
    # Backward compatibility aliases
    "PredictionConfidenceVisualizer",
    "PredictionOverlayVisualizer",
    # Training modules
    "AdvancedTrainingVisualizer",
    "TrainingCurvesVisualizer",
    "ParameterAnalysisVisualizer",
    "TrainingReportGenerator",
    # Legacy modules (for backward compatibility)
    "LegacyExperimentVisualizer",
    "LegacyParameterAnalyzer",
    "LegacyPredictionVisualizer",
    "LegacyLearningRateAnalyzer",
]
