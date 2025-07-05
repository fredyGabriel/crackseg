"""Protocols and interfaces for automated workflow execution.

This module defines the core protocols and data structures used throughout
the automation framework for consistency and type safety.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..workflow_components import (
    ConfigurationWorkflowComponent,
    TrainingWorkflowComponent,
)


@dataclass
class AutomationResult:
    """Result of an automated workflow execution."""

    workflow_name: str
    success: bool
    start_time: datetime
    end_time: datetime
    execution_time_seconds: float
    test_count: int
    passed_count: int
    failed_count: int
    error_details: list[str] = field(default_factory=list)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    artifacts_generated: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.test_count == 0:
            return 0.0
        return (self.passed_count / self.test_count) * 100.0


@runtime_checkable
class AutomatableWorkflow(Protocol):
    """Protocol for workflows that can be automated."""

    def get_workflow_name(self) -> str:
        """Get the name of this workflow for automation tracking."""
        ...

    def execute_automated_workflow(
        self, automation_config: dict[str, Any]
    ) -> AutomationResult:
        """Execute the workflow automatically with given configuration."""
        ...

    def validate_automation_preconditions(self) -> bool:
        """Check if automation preconditions are met."""
        ...

    def get_automation_metrics(self) -> dict[str, float]:
        """Get automation-specific performance metrics."""
        ...


@runtime_checkable
class AutomationReporter(Protocol):
    """Protocol for automation reporting capabilities."""

    def generate_automation_report(
        self, results: Sequence[AutomationResult]
    ) -> Path:
        """Generate comprehensive automation report."""
        ...

    def generate_summary_report(
        self, results: Sequence[AutomationResult]
    ) -> dict[str, Any]:
        """Generate summary statistics for automation results."""
        ...

    def export_metrics_data(
        self, results: Sequence[AutomationResult], output_path: Path
    ) -> None:
        """Export automation metrics for external analysis."""
        ...


class AutomationStrategy(ABC):
    """Abstract base class for automation execution strategies."""

    @abstractmethod
    def execute_strategy(
        self,
        workflows: Sequence[AutomatableWorkflow],
        automation_config: dict[str, Any],
    ) -> list[AutomationResult]:
        """Execute automation strategy on provided workflows."""

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this automation strategy."""

    @abstractmethod
    def validate_strategy_requirements(
        self, workflows: Sequence[AutomatableWorkflow]
    ) -> bool:
        """Validate that workflows are compatible with this strategy."""


@dataclass
class AutomationConfiguration:
    """Configuration for automated workflow execution."""

    execution_mode: str = "sequential"  # sequential, parallel, matrix
    timeout_seconds: int = 300
    retry_count: int = 1
    continue_on_failure: bool = True
    generate_reports: bool = True
    capture_artifacts: bool = True
    performance_monitoring: bool = True
    output_directory: Path = field(
        default_factory=lambda: Path("automation_results")
    )
    included_test_patterns: list[str] = field(default_factory=list)
    excluded_test_patterns: list[str] = field(default_factory=list)
    environment_variables: dict[str, str] = field(default_factory=dict)
    custom_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.execution_mode not in ["sequential", "parallel", "matrix"]:
            raise ValueError(
                f"Invalid execution_mode: {self.execution_mode}. "
                "Must be 'sequential', 'parallel', or 'matrix'"
            )

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if self.retry_count < 0:
            raise ValueError("retry_count cannot be negative")


@runtime_checkable
class WorkflowComponentAdapter(Protocol):
    """Protocol for adapting workflow components to automation framework."""

    def adapt_configuration_workflow(
        self, component: ConfigurationWorkflowComponent
    ) -> AutomatableWorkflow:
        """Adapt configuration workflow component for automation."""
        ...

    def adapt_training_workflow(
        self, component: TrainingWorkflowComponent
    ) -> AutomatableWorkflow:
        """Adapt training workflow component for automation."""
        ...

    def get_supported_components(self) -> list[str]:
        """Get list of supported workflow component types."""
        ...
