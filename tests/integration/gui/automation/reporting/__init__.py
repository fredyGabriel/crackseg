"""
Comprehensive integration test reporting module. This module provides
stakeholder-specific reporting capabilities for the CrackSeg
integration testing framework, extending the automation infrastructure
from subtasks 9.1-9.7.
"""

from .analysis_engine import RegressionDetectionEngine, TrendAnalysisEngine
from .data_aggregation import TestDataAggregator
from .export_manager import MultiFormatExportManager
from .integration_test_reporting import IntegrationTestReportingComponent
from .stakeholder_reporting import (
    StakeholderReportConfig,
    StakeholderReportGenerator,
)

__all__ = [
    "IntegrationTestReportingComponent",
    "StakeholderReportConfig",
    "StakeholderReportGenerator",
    "TestDataAggregator",
    "TrendAnalysisEngine",
    "RegressionDetectionEngine",
    "MultiFormatExportManager",
]
