"""Publication-ready figure generation module.

This module provides automated generation of high-quality, publication-ready
figures for experiment results and analyses. Supports multiple formats and
academic/industry publication styles.
"""

from .publication_figure_generator import (
    BaseFigureGenerator,
    FigureDataProvider,
    PerformanceComparisonFigureGenerator,
    PublicationFigureGenerator,
    PublicationStyle,
    TrainingCurvesFigureGenerator,
)

__all__ = [
    "PublicationFigureGenerator",
    "PublicationStyle",
    "BaseFigureGenerator",
    "TrainingCurvesFigureGenerator",
    "PerformanceComparisonFigureGenerator",
    "FigureDataProvider",
]
