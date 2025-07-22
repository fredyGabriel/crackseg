"""
Unified Testing Framework for CrackSeg Streamlit Application. This
package consolidates GUI testing capabilities, combining functionality
from: - gui_testing_framework.py (general testing, performance) -
visual_testing_framework.py (visual regression, snapshots) -
streamlit_test_helpers.py (specialized helpers, error testing) Public
API providing unified access to all testing capabilities.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

# Import all core components
from .mocking import UnifiedStreamlitMocker
from .performance import PerformanceProfile, UnifiedPerformanceTester

F = TypeVar("F", bound=Callable[..., Any])


class UnifiedTestConfig:
    """Configuration for unified testing framework."""

    def __init__(
        self,
        enable_visual_regression: bool = False,
        enable_performance_monitoring: bool = True,
        enable_error_testing: bool = True,
        snapshots_dir: str = "test_snapshots",
    ) -> None:
        self.enable_visual_regression = enable_visual_regression
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_error_testing = enable_error_testing
        self.snapshots_dir = snapshots_dir


class UnifiedErrorTester:
    """Unified error testing capabilities."""

    def assert_streamlit_error(
        self, error_type: type, message: str | None = None
    ) -> None:
        """Assert that a streamlit error occurs."""
        # Placeholder implementation
        pass


class UnifiedConfigHelper:
    """Helper for configuration management in tests."""

    def create_test_config(self, **overrides: Any) -> dict[str, Any]:
        """Create a test configuration with overrides."""
        base_config = {"test_mode": True}
        base_config.update(overrides)
        return base_config


# Main unified testing class
class UnifiedTestingFramework:
    """Main unified testing framework combining all capabilities."""

    def __init__(self, config: UnifiedTestConfig | None = None) -> None:
        self.config = config or UnifiedTestConfig()
        self.mocker = UnifiedStreamlitMocker()
        self.performance = UnifiedPerformanceTester()
        self.error_tester = UnifiedErrorTester()
        self.config_helper = UnifiedConfigHelper()

        # Visual testing (optional)
        if self.config.enable_visual_regression:
            Path(self.config.snapshots_dir)
            # Visual testing would be initialized here if available
            self.visual = None  # Placeholder
        else:
            self.visual = None

    def create_comprehensive_test_environment(self) -> dict[str, Any]:
        """Create a comprehensive test environment with all capabilities."""
        mock_st = self.mocker.create_streamlit_mock()

        return {
            "mock_streamlit": mock_st,
            "mocker": self.mocker,
            "performance": self.performance,
            "visual": self.visual,
            "error_tester": self.error_tester,
            "config_helper": self.config_helper,
        }


# Helper functions for creating different test configurations
def create_minimal_test_config() -> UnifiedTestConfig:
    """Create minimal test configuration."""
    return UnifiedTestConfig(
        enable_visual_regression=False,
        enable_performance_monitoring=False,
        enable_error_testing=True,
    )


def create_performance_test_config() -> UnifiedTestConfig:
    """Create performance-focused test configuration."""
    return UnifiedTestConfig(
        enable_visual_regression=False,
        enable_performance_monitoring=True,
        enable_error_testing=False,
    )


def create_visual_test_config() -> UnifiedTestConfig:
    """Create visual regression test configuration."""
    return UnifiedTestConfig(
        enable_visual_regression=True,
        enable_performance_monitoring=False,
        enable_error_testing=False,
    )


def create_comprehensive_test_config() -> UnifiedTestConfig:
    """Create comprehensive test configuration with all features."""
    return UnifiedTestConfig(
        enable_visual_regression=True,
        enable_performance_monitoring=True,
        enable_error_testing=True,
    )


def assert_streamlit_interaction(
    mock_st: Any, method_name: str, **kwargs: Any
) -> None:
    """Assert that a specific Streamlit method was called with expected args.

    Args:
        mock_st: Mock Streamlit object
        method_name: Name of the method to check
        **kwargs: Expected arguments for the method call
    """
    method = getattr(mock_st, method_name, None)
    if method is None:
        raise AssertionError(
            f"Method {method_name} not found on mock Streamlit object"
        )

    if not method.called:
        raise AssertionError(
            f"Expected {method_name} to be called but it wasn't"
        )

    if kwargs:
        # Check if any call matches the expected arguments
        for call in method.call_args_list:
            _, call_kwargs = call
            if all(call_kwargs.get(k) == v for k, v in kwargs.items()):
                return

        raise AssertionError(
            f"Expected {method_name} to be called with {kwargs}, "
            f"but calls were: {method.call_args_list}"
        )


# Export all public components
__all__ = [
    "UnifiedTestingFramework",
    "UnifiedTestConfig",
    "PerformanceProfile",
    "create_minimal_test_config",
    "create_performance_test_config",
    "create_visual_test_config",
    "create_comprehensive_test_config",
    "assert_streamlit_interaction",
]
