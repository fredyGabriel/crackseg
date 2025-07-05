"""Performance mixin for E2E testing.

This module provides performance monitoring and measurement capabilities
for E2E tests, including page load times and navigation metrics.
"""

import time
from collections.abc import Callable
from typing import Any

from selenium.webdriver.remote.webdriver import WebDriver


class PerformanceMixin:
    """Mixin providing performance monitoring capabilities for E2E tests."""

    _performance_metrics: dict[str, float] | None = None

    def setup_performance_monitoring(self) -> None:
        """Initialize performance monitoring."""
        self._performance_metrics = {}

    def measure_page_load_performance(
        self, driver: WebDriver, url: str, timeout: float = 30.0
    ) -> dict[str, float]:
        """Measure page load performance metrics.

        Args:
            driver: WebDriver instance
            url: URL to navigate to and measure
            timeout: Timeout for page load

        Returns:
            Dictionary containing performance metrics
        """
        # Record start time
        start_time = time.time()

        # Navigate to URL
        driver.get(url)

        # Wait for page to load and measure timing
        self._wait_for_page_load(driver, timeout)

        # Calculate total load time
        total_load_time = time.time() - start_time

        # Get browser performance metrics
        browser_metrics = self._get_browser_performance_metrics(driver)

        # Combine metrics
        metrics = {
            "total_load_time": total_load_time,
            "navigation_start": browser_metrics.get("navigationStart", 0.0),
            "dom_loading": browser_metrics.get("domLoading", 0.0),
            "dom_interactive": browser_metrics.get("domInteractive", 0.0),
            "dom_complete": browser_metrics.get("domComplete", 0.0),
            "load_event_end": browser_metrics.get("loadEventEnd", 0.0),
        }

        # Calculate derived metrics
        if metrics["dom_interactive"] > 0 and metrics["navigation_start"] > 0:
            metrics["time_to_interactive"] = (
                metrics["dom_interactive"] - metrics["navigation_start"]
            ) / 1000.0  # Convert to seconds

        if metrics["dom_complete"] > 0 and metrics["navigation_start"] > 0:
            metrics["dom_ready_time"] = (
                metrics["dom_complete"] - metrics["navigation_start"]
            ) / 1000.0  # Convert to seconds

        # Store metrics for reporting
        if self._performance_metrics is not None:
            self._performance_metrics.update(metrics)

        # Log metrics if logging is available
        if hasattr(self, "log_performance_metric"):
            for metric_name, value in metrics.items():
                if value > 0:
                    self.log_performance_metric(  # type: ignore[attr-defined]
                        metric_name, float(value)
                    )

        return metrics

    def measure_operation_time(
        self, operation_name: str, operation: Callable[[], Any]
    ) -> tuple[Any, float]:
        """Measure time taken for a specific operation.

        Args:
            operation_name: Name of the operation for logging
            operation: Callable to execute and measure

        Returns:
            Tuple of (operation_result, time_taken_seconds)
        """
        start_time = time.time()
        result = operation()
        duration = time.time() - start_time

        # Store metric
        if self._performance_metrics is not None:
            self._performance_metrics[f"{operation_name}_duration"] = duration

        # Log metric if logging is available
        if hasattr(self, "log_performance_metric"):
            self.log_performance_metric(  # type: ignore[attr-defined]
                f"{operation_name}_duration", duration
            )

        return result, duration

    def get_current_performance_metrics(self) -> dict[str, float]:
        """Get all collected performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        return (
            self._performance_metrics.copy()
            if self._performance_metrics
            else {}
        )

    def assert_performance_within_limits(
        self, metric_name: str, max_value: float, unit: str = "seconds"
    ) -> None:
        """Assert that a performance metric is within acceptable limits.

        Args:
            metric_name: Name of the metric to check
            max_value: Maximum acceptable value
            unit: Unit of measurement for logging

        Raises:
            AssertionError: If metric exceeds limit
        """
        if not self._performance_metrics:
            raise AssertionError("No performance metrics available")

        if metric_name not in self._performance_metrics:
            raise AssertionError(
                f"Performance metric '{metric_name}' not found"
            )

        actual_value = self._performance_metrics[metric_name]
        passed = actual_value <= max_value

        # Log assertion
        if hasattr(self, "log_assertion"):
            self.log_assertion(  # type: ignore[attr-defined]
                f"{metric_name} <= {max_value} {unit}",
                passed,
                f"Actual: {actual_value:.3f} {unit}",
            )

        if not passed:
            raise AssertionError(
                f"Performance metric '{metric_name}' "
                f"({actual_value:.3f} {unit}) exceeds limit "
                f"({max_value} {unit})"
            )

    def _wait_for_page_load(
        self, driver: WebDriver, timeout: float = 30.0
    ) -> None:
        """Wait for page to complete loading.

        Args:
            driver: WebDriver instance
            timeout: Maximum time to wait
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                ready_state = driver.execute_script(
                    "return document.readyState"
                )
                if ready_state == "complete":
                    return
                time.sleep(0.1)
            except Exception:
                # Continue waiting on any script execution errors
                time.sleep(0.1)

        # Log timeout if logging is available
        if hasattr(self, "log_test_step"):
            self.log_test_step(  # type: ignore[attr-defined]
                f"Page load timeout after {timeout}s"
            )

    def _get_browser_performance_metrics(
        self, driver: WebDriver
    ) -> dict[str, float]:
        """Get performance metrics from browser.

        Args:
            driver: WebDriver instance

        Returns:
            Dictionary of browser performance metrics
        """
        try:
            # Get navigation timing API metrics
            script = """
                var timing = window.performance.timing;
                return {
                    navigationStart: timing.navigationStart,
                    domLoading: timing.domLoading,
                    domInteractive: timing.domInteractive,
                    domComplete: timing.domComplete,
                    loadEventStart: timing.loadEventStart,
                    loadEventEnd: timing.loadEventEnd
                };
            """
            return driver.execute_script(script) or {}
        except Exception:
            # Return empty dict if performance API is not available
            return {}
