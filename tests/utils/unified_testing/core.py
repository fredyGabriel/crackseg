"""
Core configuration and base classes for unified testing framework.
This module provides the foundational configuration and data
structures for the unified testing framework, consolidating common
settings and types.
"""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UnifiedTestConfig:
    """Unified configuration for all testing capabilities."""

    # Core testing settings
    enable_session_state: bool = True
    enable_widget_callbacks: bool = True
    enable_file_uploads: bool = True
    enable_download_buttons: bool = True
    mock_external_apis: bool = True

    # Performance testing
    performance_tracking: bool = True
    ui_interaction_timeout: float = 5.0
    widget_interaction_delay: float = 0.1

    # Visual regression settings
    enable_visual_regression: bool = True
    snapshots_dir: str = "test-artifacts/visual-snapshots"
    visual_tolerance_percent: float = 5.0

    # Error testing settings
    enable_error_testing: bool = True
    capture_error_details: bool = True


@dataclass
class TestInteractionResult:
    """Result of an automated UI interaction test."""

    success: bool
    interaction_type: str
    element_id: str | None
    execution_time: float
    error_message: str | None = None
    captured_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance profile for a GUI component."""

    component_name: str
    render_time_ms: float
    memory_usage_mb: float
    interaction_latency_ms: float
    widget_count: int
    complexity_score: float


class BaseTestTracker:
    """Base class for tracking test interactions and history."""

    def __init__(self) -> None:
        self._interaction_history: list[dict[str, Any]] = []

    def track_interaction(
        self, interaction_type: str, details: dict[str, Any]
    ) -> None:
        """Track interactions for debugging and analysis."""
        self._interaction_history.append(
            {
                "type": interaction_type,
                "timestamp": time.time(),
                "details": details,
            }
        )

    def get_interaction_history(self) -> list[dict[str, Any]]:
        """Get the history of tracked interactions."""
        return self._interaction_history.copy()

    def clear_history(self) -> None:
        """Clear the interaction history."""
        self._interaction_history.clear()


class TestValidationMixin:
    """Mixin providing common validation methods for testing."""

    @staticmethod
    def validate_mock_call_count(
        mock_method: Any, expected_calls: int
    ) -> bool:
        """Validate that a mock method was called expected number of times."""
        return mock_method.call_count >= expected_calls

    @staticmethod
    def validate_expected_state(
        expected: dict[str, Any], actual: dict[str, Any]
    ) -> bool:
        """Validate expected state against actual state."""
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            if actual[key] != expected_value:
                return False
        return True

    @staticmethod
    def extract_mock_call_args(mock_method: Any) -> list[tuple[Any, ...]]:
        """Extract arguments from all calls to a mock method."""
        return [call.args for call in mock_method.call_args_list]

    @staticmethod
    def extract_mock_call_kwargs(mock_method: Any) -> list[dict[str, Any]]:
        """Extract keyword arguments from all calls to a mock method."""
        return [call.kwargs for call in mock_method.call_args_list]


# Configuration factories for common test scenarios
def create_minimal_test_config() -> UnifiedTestConfig:
    """Create minimal configuration for lightweight testing."""
    return UnifiedTestConfig(
        enable_visual_regression=False,
        performance_tracking=False,
        enable_error_testing=False,
        mock_external_apis=False,
    )


def create_performance_test_config() -> UnifiedTestConfig:
    """Create configuration optimized for performance testing."""
    return UnifiedTestConfig(
        enable_visual_regression=False,
        performance_tracking=True,
        enable_error_testing=False,
        mock_external_apis=False,
    )


def create_visual_test_config(
    snapshots_dir: str = "test-artifacts/visual-snapshots",
) -> UnifiedTestConfig:
    """Create configuration optimized for visual regression testing."""
    return UnifiedTestConfig(
        enable_visual_regression=True,
        snapshots_dir=snapshots_dir,
        performance_tracking=False,
        enable_error_testing=False,
        visual_tolerance_percent=2.0,
    )


def create_comprehensive_test_config() -> UnifiedTestConfig:
    """Create comprehensive configuration with all features enabled."""
    return UnifiedTestConfig()  # Uses all defaults which enable everything
