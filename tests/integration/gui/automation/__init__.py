"""
Automated workflow scripting framework for CrackSeg GUI testing. This
module provides automated execution of workflow scenarios, including
positive and negative test cases, with comprehensive orchestration and
reporting capabilities. Extended with performance benchmarking and
analysis for systematic workflow performance measurement and
optimization.
"""

from .automation_orchestrator import AutomationOrchestrator
from .automation_protocols import (
    AutomatableWorkflow,
    AutomationReporter,
    AutomationResult,
)
from .ci_integration import CIIntegrationAutomator
from .performance_benchmarking import (
    BottleneckAnalysis,
    PerformanceBenchmarkingComponent,
    PerformanceInstrumentationMixin,
    PerformanceMetrics,
)
from .test_data_automation import TestDataAutomator
from .workflow_automation import WorkflowAutomationComponent

__all__ = [
    "AutomationOrchestrator",
    "AutomatableWorkflow",
    "AutomationReporter",
    "AutomationResult",
    "CIIntegrationAutomator",
    "PerformanceBenchmarkingComponent",
    "PerformanceInstrumentationMixin",
    "PerformanceMetrics",
    "BottleneckAnalysis",
    "TestDataAutomator",
    "WorkflowAutomationComponent",
]
