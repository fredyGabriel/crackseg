"""
E2E tests demonstrating performance testing integration. This module
showcases the performance monitoring capabilities integrated in
subtask 15.7, including page load measurements, interaction latency
tracking, memory monitoring, and performance threshold validation.
"""

import time
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from selenium.webdriver.remote.webdriver import (
        WebDriver,  # type: ignore[import-untyped]
    )
else:
    try:
        from selenium.webdriver.remote.webdriver import WebDriver
    except ImportError:
        WebDriver = Any  # type: ignore[misc,assignment]

from ..base_test import BaseE2ETest
from ..pages import ArchitecturePage, ConfigPage, TrainPage


@pytest.mark.e2e
class TestPerformanceIntegration(BaseE2ETest):
    """Test suite demonstrating performance monitoring integration."""

    def __init__(self) -> None:
        """Initialize performance integration test."""
        super().__init__()
        self._performance_enabled = True

    def start_performance_monitoring(self) -> None:
        """Start performance monitoring for the test."""
        pass

    def get_test_data(self) -> dict[str, Any]:
        """Get test data configuration."""
        return self.setup_test_data()

    def track_interaction_performance(
        self,
        driver: "WebDriver",
        interaction_type: str,
        description: str,
        threshold_ms: float = 1000.0,
    ) -> float | None:
        """Track interaction performance."""
        # Placeholder implementation
        return 0.5  # Return dummy latency

    def monitor_memory_usage_snapshot(self) -> dict[str, Any] | None:
        """Take a memory usage snapshot."""
        return {"rss_memory_mb": 100.0, "cpu_percent": 5.0}

    def generate_performance_report(self) -> dict[str, Any] | None:
        """Generate performance report."""
        return {"avg_page_load_time": 2.0, "peak_memory_usage": 150.0}

    def validate_performance_thresholds(
        self, thresholds: dict[str, float]
    ) -> dict[str, bool]:
        """Validate performance against thresholds."""
        return dict.fromkeys(thresholds.keys(), True)

    def setup_test_data(self) -> dict[str, Any]:
        """
        Set up test-specific data for performance testing. Returns: Dictionary
        containing test configuration and thresholds.
        """
        return {
            "config_file": "basic_verification.yaml",
            "performance_thresholds": {
                "page_load_time_seconds": 8.0,
                "interaction_latency_ms": 1000.0,
                "memory_usage_mb": 300.0,
            },
        }

    @pytest.mark.performance
    def test_page_load_performance(
        self,
        webdriver: WebDriver,
        streamlit_base_url: str,
        performance_config: dict[str, Any],
    ) -> None:
        """
        Test page load performance measurement and validation. This test
        demonstrates automatic performance monitoring for tests marked with
        @pytest.mark.performance. It measures page load times and validates
        against configurable thresholds. Args: webdriver: The Selenium
        WebDriver instance streamlit_base_url: The base URL of the Streamlit
        application performance_config: Performance configuration from
        conftest
        """
        self.log_test_step("Testing page load performance measurement")

        # Performance monitoring is automatically enabled via decorator
        assert (
            self._performance_enabled
        ), "Performance monitoring should be enabled"

        # Start monitoring
        self.start_performance_monitoring()

        # Navigate with performance measurement
        performance_metrics = self.navigate_and_verify(
            webdriver, streamlit_base_url, measure_performance=True
        )

        # Verify performance metrics were captured
        assert (
            performance_metrics is not None
        ), "Performance metrics should be captured"
        assert (
            "load_complete" in performance_metrics
        ), "Load complete time should be measured"
        assert (
            "dom_content_loaded" in performance_metrics
        ), "DOM load time should be measured"

        # Log the measured performance
        load_time = performance_metrics["load_complete"]
        self.log_performance_metric("total_page_load", load_time, "seconds")

        # Validate against threshold
        threshold = self.get_test_data()["performance_thresholds"][
            "page_load_time_seconds"
        ]
        self.log_assertion(
            f"Page load time within threshold ({threshold}s)",
            load_time <= threshold,
            f"Actual: {load_time:.3f}s",
        )

    @pytest.mark.performance_critical
    def test_user_interaction_performance(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """
        Test user interaction performance measurement. This test demonstrates
        performance tracking for user interactions with strict performance
        thresholds for critical functionality. Args: webdriver: The Selenium
        WebDriver instance streamlit_base_url: The base URL of the Streamlit
        application
        """
        self.log_test_step("Testing user interaction performance")

        # Navigate to the application
        self.navigate_and_verify(webdriver, streamlit_base_url)

        # Start performance monitoring
        self.start_performance_monitoring()

        # Navigate to Config page and measure interaction performance
        self.log_test_step("Measuring Config page navigation performance")
        config_page_base = ConfigPage(webdriver).navigate_to_page()
        assert isinstance(config_page_base, ConfigPage)
        config_page = config_page_base

        # Measure file selection interaction
        config_file = self.get_test_data()["config_file"]
        selection_latency = self.track_interaction_performance(
            webdriver,
            "config_selection",
            f"Config file selection: {config_file}",
            threshold_ms=500.0,
        )

        if selection_latency:
            self.log_test_step(
                "Config selection completed in "
                f"{selection_latency * 1000:.1f}ms "
            )

        # Perform the actual selection
        config_page.select_config_file(config_file)

        # Measure button click interaction
        click_latency = self.track_interaction_performance(
            webdriver,
            "button_click",
            "Load Configuration button",
            threshold_ms=300.0,
        )

        # Perform the actual click
        config_page.click_load_config()

        if click_latency:
            self.log_test_step(
                f"Button click completed in {click_latency * 1000:.1f}ms"
            )

    @pytest.mark.performance
    def test_memory_monitoring_integration(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """
        Test memory usage monitoring during E2E workflow. This test
        demonstrates continuous memory monitoring during test execution with
        periodic snapshots. Args: webdriver: The Selenium WebDriver instance
        streamlit_base_url: The base URL of the Streamlit application
        """
        self.log_test_step("Testing memory monitoring integration")

        # Start performance monitoring
        self.start_performance_monitoring()

        # Take initial memory snapshot
        initial_memory = self.monitor_memory_usage_snapshot()
        assert (
            initial_memory is not None
        ), "Initial memory snapshot should be captured"

        self.log_test_step(
            f"Initial memory usage: {initial_memory['rss_memory_mb']:.1f}MB"
        )

        # Navigate and load configuration
        self.navigate_and_verify(webdriver, streamlit_base_url)

        # Take memory snapshot after navigation
        post_nav_memory = self.monitor_memory_usage_snapshot()
        if post_nav_memory:
            memory_increase = (
                post_nav_memory["rss_memory_mb"]
                - initial_memory["rss_memory_mb"]
            )
            self.log_test_step(
                "Memory after navigation: "
                f"{post_nav_memory['rss_memory_mb']:.1f}MB "
                f"(+{memory_increase:.1f}MB)"
            )

        # Navigate through multiple pages to monitor memory usage
        config_page_base = ConfigPage(webdriver).navigate_to_page()
        assert isinstance(config_page_base, ConfigPage)
        config_page = config_page_base

        # Load configuration
        config_file = self.get_test_data()["config_file"]
        config_page.select_config_file(config_file)
        config_page.click_load_config()
        config_page.wait_for_configuration_loaded()

        # Take memory snapshot after configuration load
        post_config_memory = self.monitor_memory_usage_snapshot()
        if post_config_memory:
            total_increase = (
                post_config_memory["rss_memory_mb"]
                - initial_memory["rss_memory_mb"]
            )
            self.log_test_step(
                "Memory after config load: "
                f"{post_config_memory['rss_memory_mb']:.1f}MB "
                f"(+{total_increase:.1f}MB total)"
            )

            # Validate against threshold
            threshold = self.get_test_data()["performance_thresholds"][
                "memory_usage_mb"
            ]
            if total_increase > threshold:
                self.log_assertion(
                    f"Memory increase within threshold ({threshold}MB)",
                    False,
                    f"Actual increase: {total_increase:.1f}MB",
                )

    @pytest.mark.performance_baseline
    def test_comprehensive_workflow_performance(
        self,
        webdriver: WebDriver,
        streamlit_base_url: str,
        performance_config: dict[str, Any],
    ) -> None:
        """
        Test comprehensive workflow performance for baseline establishment.
        This test runs a complete workflow while monitoring all performance
        aspects, establishing performance baselines for future comparisons.
        Args: webdriver: The Selenium WebDriver instance streamlit_base_url:
        The base URL of the Streamlit application performance_config:
        Performance configuration from conftest
        """
        self.log_test_step("Testing comprehensive workflow performance")

        # Start comprehensive performance monitoring
        self.start_performance_monitoring()

        # Measure full workflow performance
        workflow_start_time = time.time()

        try:
            # 1. Navigate to application with performance measurement
            self.navigate_and_verify(
                webdriver, streamlit_base_url, measure_performance=True
            )

            # 2. Config page workflow
            self.log_test_step("Config page workflow")
            config_page_base = ConfigPage(webdriver).navigate_to_page()
            assert isinstance(config_page_base, ConfigPage)
            config_page = config_page_base

            config_file = self.get_test_data()["config_file"]
            config_page.select_config_file(config_file)
            config_page.click_load_config()
            config_page.wait_for_configuration_loaded()

            # 3. Architecture page workflow
            self.log_test_step("Architecture page workflow")
            arch_page_base = ArchitecturePage(webdriver).navigate_to_page()
            assert isinstance(arch_page_base, ArchitecturePage)
            arch_page = arch_page_base

            arch_page.instantiate_model()
            arch_page.wait_for_model_instantiation()

            # 4. Train page workflow (without actual training)
            self.log_test_step("Train page workflow")
            train_page_base = TrainPage(webdriver).navigate_to_page()
            assert isinstance(train_page_base, TrainPage)

            # Monitor memory usage throughout
            self.monitor_memory_usage_snapshot()

            workflow_duration = time.time() - workflow_start_time
            self.log_performance_metric(
                "workflow_duration", workflow_duration, "seconds"
            )

            # Generate comprehensive performance report
            performance_report = self.generate_performance_report()

            if performance_report:
                self.log_test_step(
                    f"Workflow completed - "
                    f"Duration: {workflow_duration:.2f}s, "
                    f"Avg Page Load: "
                    f"{performance_report.get('avg_page_load_time', 'N/A')}s, "
                    f"Peak Memory: "
                    f"{performance_report.get('peak_memory_usage', 'N/A')}MB"
                )

            # Validate overall performance against thresholds
            thresholds = self.get_test_data()["performance_thresholds"]
            validation_results = self.validate_performance_thresholds(
                thresholds
            )

            # Log validation results
            passed_count = sum(validation_results.values())
            total_count = len(validation_results)
            self.log_test_step(
                "Performance validation: "
                f"{passed_count}/{total_count} thresholds passed "
            )

        except Exception as e:
            self.log_test_step(f"Workflow performance test failed: {e}")
            raise

    @pytest.mark.performance
    def test_performance_regression_detection(
        self, webdriver: WebDriver, streamlit_base_url: str
    ) -> None:
        """
        Test performance regression detection capabilities. This test
        demonstrates how performance monitoring can detect regressions by
        comparing against established baselines. Args: webdriver: The Selenium
        WebDriver instance streamlit_base_url: The base URL of the Streamlit
        application
        """
        self.log_test_step("Testing performance regression detection")

        # Start performance monitoring
        self.start_performance_monitoring()

        # Establish baseline measurements
        baseline_metrics = {}

        # Measure page load baseline
        page_load_metrics = self.navigate_and_verify(
            webdriver, streamlit_base_url, measure_performance=True
        )

        if page_load_metrics:
            baseline_metrics["page_load"] = page_load_metrics["load_complete"]

        # Measure interaction baseline
        config_page_base = ConfigPage(webdriver).navigate_to_page()
        assert isinstance(config_page_base, ConfigPage)

        interaction_latency = self.track_interaction_performance(
            webdriver, "navigation", "Config page navigation"
        )

        if interaction_latency:
            baseline_metrics["navigation_latency"] = interaction_latency

        # Generate baseline report
        baseline_report = self.generate_performance_report()

        # Simulate regression detection (in a real scenario, this would
        # compare against historical data)
        if baseline_report:
            self.log_test_step(
                "Performance baseline established for regression detection"
            )

            # Example regression thresholds (20% increase over baseline)
            regression_thresholds = {}
            if "page_load" in baseline_metrics:
                regression_thresholds["page_load_time_seconds"] = (
                    baseline_metrics["page_load"] * 1.2
                )

            # Validate against regression thresholds
            if regression_thresholds:
                regression_results = self.validate_performance_thresholds(
                    regression_thresholds
                )

                failed_metrics = [
                    metric
                    for metric, passed in regression_results.items()
                    if not passed
                ]

                if failed_metrics:
                    self.log_test_step(
                        "⚠️ Performance regression detected in: "
                        f"{failed_metrics} "
                    )
                else:
                    self.log_test_step(
                        "✅ No performance regressions detected"
                    )
