"""Experimental reporting system for CrackSeg.

This module provides comprehensive reporting capabilities for experiment
analysis, performance evaluation, and publication-ready figure generation.
"""

from .config import OutputFormat, ReportConfig, TemplateType
from .core import ExperimentReporter
from .data_loader import ExperimentDataLoader
from .interfaces import (
    ComparisonEngine,
    FigureGenerator,
    PerformanceAnalyzer,
    RecommendationEngine,
    SummaryGenerator,
    TemplateManager,
)
from .performance import ExperimentPerformanceAnalyzer
from .recommendations import AutomatedRecommendationEngine

__all__ = [
    "ExperimentReporter",
    "ExperimentDataLoader",
    "ExperimentPerformanceAnalyzer",
    "AutomatedRecommendationEngine",
    "ReportConfig",
    "OutputFormat",
    "TemplateType",
    "SummaryGenerator",
    "PerformanceAnalyzer",
    "ComparisonEngine",
    "FigureGenerator",
    "TemplateManager",
    "RecommendationEngine",
]
