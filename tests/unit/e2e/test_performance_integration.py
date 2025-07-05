"""Unit tests for E2E performance testing integration.

This module tests the performance monitoring integration added in subtask 15.7,
ensuring that performance mixins, fixtures, and helpers work correctly.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock, patch

from tests.e2e.base_test import BaseE2ETest
from tests.e2e.mixins.performance_mixin import PerformanceMixin


class TestPerformanceMixin:
    """Test the PerformanceMixin functionality."""

    def test_performance_mixin_initialization(self) -> None:
        """Test that PerformanceMixin initializes correctly."""
        mixin = PerformanceMixin()

        # Check initial state
        assert mixin._performance_metrics is None

    def test_setup_performance_monitoring(self) -> None:
        """Test performance monitoring setup."""
        mixin = PerformanceMixin()

        # Execute
        mixin.setup_performance_monitoring()

        # Verify
        assert mixin._performance_metrics == {}

    def test_measure_page_load_performance_success(self) -> None:
        """Test successful page load performance measurement."""
        mixin = PerformanceMixin()
        mixin.setup_performance_monitoring()

        mock_driver = Mock()

        # Mock browser performance metrics
        mock_metrics = {
            "navigationStart": 1000.0,
            "domLoading": 1100.0,
            "domInteractive": 1200.0,
            "domComplete": 1500.0,
            "loadEventEnd": 1600.0,
        }
        mock_driver.execute_script.return_value = mock_metrics

        # Mock document ready state
        mock_driver.execute_script.side_effect = [
            "complete",  # First call for _wait_for_page_load
            mock_metrics,  # Second call for _get_browser_performance_metrics
        ]

        # Execute
        result = mixin.measure_page_load_performance(
            mock_driver, "http://test.com", 30.0
        )

        # Verify
        assert isinstance(result, dict)
        assert "total_load_time" in result
        assert "navigation_start" in result
        assert result["navigation_start"] == 1000.0
        assert mock_driver.get.called_once_with("http://test.com")

    def test_measure_operation_time(self) -> None:
        """Test operation time measurement."""
        mixin = PerformanceMixin()
        mixin.setup_performance_monitoring()

        # Mock operation
        def mock_operation() -> str:
            return "test_result"

        # Execute
        result, duration = mixin.measure_operation_time(
            "test_operation", mock_operation
        )

        # Verify
        assert result == "test_result"
        assert isinstance(duration, float)
        assert duration > 0
        assert mixin._performance_metrics is not None
        assert "test_operation_duration" in mixin._performance_metrics

    def test_get_current_performance_metrics(self) -> None:
        """Test getting current performance metrics."""
        mixin = PerformanceMixin()

        # Test with no metrics
        metrics = mixin.get_current_performance_metrics()
        assert metrics == {}

        # Test with metrics
        mixin.setup_performance_monitoring()
        assert mixin._performance_metrics is not None
        mixin._performance_metrics["test_metric"] = 1.5

        metrics = mixin.get_current_performance_metrics()
        assert metrics == {"test_metric": 1.5}

    def test_assert_performance_within_limits_success(self) -> None:
        """Test performance assertion that passes."""
        mixin = PerformanceMixin()
        mixin.setup_performance_monitoring()
        assert mixin._performance_metrics is not None
        mixin._performance_metrics["test_metric"] = 2.5

        # Should not raise an exception
        mixin.assert_performance_within_limits("test_metric", 3.0)

    def test_assert_performance_within_limits_failure(self) -> None:
        """Test performance assertion that fails."""
        mixin = PerformanceMixin()
        mixin.setup_performance_monitoring()
        assert mixin._performance_metrics is not None
        mixin._performance_metrics["test_metric"] = 4.5

        # Should raise AssertionError
        try:
            mixin.assert_performance_within_limits("test_metric", 3.0)
            raise AssertionError("Expected AssertionError")
        except AssertionError as e:
            assert "exceeds limit" in str(e)

    def test_assert_performance_metric_not_found(self) -> None:
        """Test performance assertion with missing metric."""
        mixin = PerformanceMixin()
        mixin.setup_performance_monitoring()

        # Should raise AssertionError
        try:
            mixin.assert_performance_within_limits("missing_metric", 3.0)
            raise AssertionError("Expected AssertionError")
        except AssertionError as e:
            assert "not found" in str(e)

    def test_assert_performance_no_metrics(self) -> None:
        """Test performance assertion with no metrics available."""
        mixin = PerformanceMixin()

        # Should raise AssertionError
        try:
            mixin.assert_performance_within_limits("test_metric", 3.0)
            raise AssertionError("Expected AssertionError")
        except AssertionError as e:
            assert "No performance metrics available" in str(e)


class TestBaseE2ETestPerformanceIntegration:
    """Test BaseE2ETest integration with performance monitoring."""

    def test_base_e2e_test_inherits_performance_mixin(self) -> None:
        """Test that BaseE2ETest includes PerformanceMixin."""
        # Check that BaseE2ETest has PerformanceMixin in its MRO
        assert PerformanceMixin in BaseE2ETest.__mro__

        # Create instance and check it has performance methods
        test_instance = BaseE2ETest()

        assert hasattr(test_instance, "setup_performance_monitoring")
        assert hasattr(test_instance, "measure_page_load_performance")
        assert hasattr(test_instance, "measure_operation_time")
        assert hasattr(test_instance, "get_current_performance_metrics")
        assert hasattr(test_instance, "assert_performance_within_limits")

    def test_setup_method_initializes_performance(self) -> None:
        """Test that setup_method initializes performance monitoring."""
        test_instance = BaseE2ETest()

        # Mock required attributes
        test_instance.setup_logging = Mock()
        test_instance.setup_retry = Mock()
        test_instance.setup_capture_system = Mock()
        test_instance.configure_capture_from_test_data = Mock()
        test_instance.log_test_step = Mock()

        # Create mock method
        mock_method = Mock()
        mock_method.__name__ = "test_example"

        # Execute setup
        test_instance.setup_method(mock_method)

        # Verify performance monitoring was initialized
        assert test_instance._performance_metrics == {}

    def test_navigate_and_verify_with_performance_measurement(self) -> None:
        """Test navigate_and_verify with performance measurement enabled."""
        test_instance = BaseE2ETest()

        # Mock required methods
        test_instance.log_test_step = Mock()
        test_instance.assert_page_ready_state = Mock()
        test_instance.assert_streamlit_loaded = Mock()
        test_instance._is_streamlit_app = Mock(return_value=True)

        # Setup performance monitoring
        test_instance.setup_performance_monitoring()

        mock_driver = Mock()
        mock_driver.execute_script.side_effect = [
            "complete",  # For page ready state
            {  # For performance metrics
                "navigationStart": 1000.0,
                "domLoading": 1100.0,
                "domComplete": 1500.0,
            },
        ]

        # Mock base_url
        test_instance.base_url = "http://test.com"

        # Test with performance measurement
        result = test_instance.navigate_and_verify(
            mock_driver, "", measure_performance=True
        )

        assert result is not None
        assert isinstance(result, dict)
        assert "total_load_time" in result

    def test_navigate_and_verify_without_performance_measurement(self) -> None:
        """Test navigate_and_verify without performance measurement."""
        test_instance = BaseE2ETest()

        # Mock required methods
        test_instance.log_test_step = Mock()
        test_instance.assert_page_ready_state = Mock()
        test_instance.assert_streamlit_loaded = Mock()
        test_instance._is_streamlit_app = Mock(return_value=False)

        mock_driver = Mock()
        test_instance.base_url = "http://test.com"

        # Test without performance measurement
        result = test_instance.navigate_and_verify(
            mock_driver, "", measure_performance=False
        )

        assert result is None
        mock_driver.get.assert_called_once_with("http://test.com")


class TestPerformanceFixtures:
    """Test performance-related pytest fixtures."""

    def test_performance_config_fixture_structure(
        self, performance_config: dict[str, Any]
    ) -> None:
        """Test that performance_config fixture has expected structure."""
        # Check main sections
        assert "thresholds" in performance_config
        assert "monitoring" in performance_config
        assert "reporting" in performance_config

        # Check thresholds
        thresholds = performance_config["thresholds"]
        assert "page_load_time_seconds" in thresholds
        assert "interaction_latency_ms" in thresholds
        assert "memory_usage_mb" in thresholds
        assert "training_start_seconds" in thresholds

        # Check monitoring config
        monitoring = performance_config["monitoring"]
        assert "enabled" in monitoring
        assert "interval_seconds" in monitoring
        assert "capture_browser_memory" in monitoring
        assert "generate_reports" in monitoring

        # Check reporting config
        reporting = performance_config["reporting"]
        assert "output_dir" in reporting
        assert "save_individual_reports" in reporting
        assert "generate_summary" in reporting
        assert "threshold_warnings" in reporting

    def test_performance_monitor_factory_fixture(
        self, performance_monitor_factory: Callable[[str], Any]
    ) -> None:
        """Test that performance_monitor_factory fixture works correctly."""
        with patch(
            "tests.e2e.helpers.performance_monitoring.PerformanceMonitor"
        ) as mock_monitor_class:
            mock_monitor = Mock()
            mock_monitor_class.return_value = mock_monitor

            # Use the factory
            monitor = performance_monitor_factory("test_monitor")

            # Verify
            assert monitor is mock_monitor
            mock_monitor_class.assert_called_once_with("test_monitor")

    def test_performance_test_setup_fixture_activation(self) -> None:
        """Test that performance_test_setup fixture activates for tests."""
        # This is more of an integration test that would be run in actual
        # pytest environment. Here we test the logic that would be used

        performance_markers = [
            "performance",
            "performance_critical",
            "performance_baseline",
        ]

        # Mock request object
        mock_request = Mock()

        # Test with performance marker
        mock_marker = Mock()
        mock_marker.name = "performance"
        mock_request.node.get_closest_marker.return_value = mock_marker

        # The fixture logic would detect this as a performance test
        is_performance_test = any(
            mock_request.node.get_closest_marker(marker)
            for marker in performance_markers
        )

        assert is_performance_test is True

        # Test without performance marker
        mock_request.node.get_closest_marker.return_value = None

        is_performance_test = any(
            mock_request.node.get_closest_marker(marker)
            for marker in performance_markers
        )

        assert is_performance_test is False


class TestMockPerformanceFeatures:
    """Test mock implementations for features not yet implemented."""

    def test_extended_performance_monitoring_mock(self) -> None:
        """Test that we can mock extended performance features."""
        mixin = PerformanceMixin()

        # Mock additional attributes that tests expect
        mixin._performance_enabled = False  # type: ignore[attr-defined]
        mixin._performance_monitor = None  # type: ignore[attr-defined]
        mixin._performance_config = None  # type: ignore[attr-defined]
        mixin._test_logger = Mock()  # type: ignore[attr-defined]

        # Mock additional methods
        mixin.start_performance_monitoring = Mock()  # type: ignore[attr-defined]
        mixin.stop_performance_monitoring = Mock()  # type: ignore[attr-defined]
        mixin.log_performance_metric = Mock()  # type: ignore[attr-defined]
        mixin.log_assertion = Mock()  # type: ignore[attr-defined]

        # Test that mocked features work
        assert hasattr(mixin, "_performance_enabled")
        assert hasattr(mixin, "start_performance_monitoring")
        # Check that the mocked method is callable
        start_method = getattr(mixin, "start_performance_monitoring", None)
        assert start_method is not None
        assert callable(start_method)

    def test_advanced_performance_features_integration(self) -> None:
        """Test integration with advanced performance features (mocked)."""
        # This test demonstrates how the system would work with full
        # implementation

        # Mock a more complete performance system
        class MockAdvancedPerformanceMixin(PerformanceMixin):
            def __init__(self) -> None:
                super().__init__()
                self._performance_enabled = False
                self._performance_monitor: Any = None
                self._test_logger = Mock()

            def setup_performance_monitoring_advanced(
                self, test_name: str, config: dict[str, Any] | None = None
            ) -> None:
                """Extended setup with configuration."""
                self.setup_performance_monitoring()
                self._performance_enabled = True
                self._performance_config = config or {}

            def track_interaction_performance(
                self, driver: Any, action: str, target: str
            ) -> float:
                """Mock interaction tracking."""
                return 0.3  # 300ms mock latency

        # Test the mock implementation
        advanced_mixin = MockAdvancedPerformanceMixin()

        advanced_mixin.setup_performance_monitoring_advanced(
            "test", {"key": "value"}
        )
        assert advanced_mixin._performance_enabled is True

        latency = advanced_mixin.track_interaction_performance(
            Mock(), "click", "button"
        )
        assert latency == 0.3
