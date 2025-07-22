"""
Analysis components for test reporting system. This package provides
failure analysis and trend analysis capabilities for comprehensive
test result analysis and insights generation.
"""

from tests.e2e.reporting.analysis.failure_analyzer import FailureAnalyzer
from tests.e2e.reporting.analysis.trend_analyzer import TestTrendAnalyzer

__all__ = ["FailureAnalyzer", "TestTrendAnalyzer"]
