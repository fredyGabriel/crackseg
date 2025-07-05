"""Performance monitoring helpers for E2E testing with metrics collection.

This module provides utilities for measuring and monitoring performance metrics
during E2E tests, including page load times, memory usage, interaction latency,
and comprehensive performance reporting. These helpers are essential for
performance testing integration (subtask 15.7).
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import psutil
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)


class PerformanceMetric(TypedDict):
    """Type definition for performance metric data."""

    metric_name: str
    value: float
    unit: str
    timestamp: float
    context: dict[str, Any]


class PageLoadMetrics(TypedDict):
    """Type definition for page load performance metrics."""

    dom_content_loaded: float
    load_complete: float
    first_paint: float | None
    first_contentful_paint: float | None
    largest_contentful_paint: float | None


class MemoryMetrics(TypedDict):
    """Type definition for memory usage metrics."""

    rss_memory_mb: float
    vms_memory_mb: float
    cpu_percent: float
    memory_percent: float
    timestamp: float


class ExtendedMemoryMetrics(MemoryMetrics, total=False):
    """Extended memory metrics with optional browser memory."""

    browser_memory_mb: float


@dataclass
class PerformanceReport:
    """Data class for comprehensive performance reports."""

    test_name: str
    start_time: float
    end_time: float
    metrics: list[PerformanceMetric] = field(default_factory=list)
    page_loads: list[PageLoadMetrics] = field(default_factory=list)
    memory_snapshots: list[MemoryMetrics] = field(default_factory=list)
    interactions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        """Calculate total test duration."""
        return self.end_time - self.start_time

    @property
    def average_page_load_time(self) -> float:
        """Calculate average page load time."""
        if not self.page_loads:
            return 0.0

        total_load_time = sum(pl["load_complete"] for pl in self.page_loads)
        return total_load_time / len(self.page_loads)

    @property
    def peak_memory_usage(self) -> float:
        """Get peak memory usage in MB."""
        if not self.memory_snapshots:
            return 0.0

        return max(
            snapshot["rss_memory_mb"] for snapshot in self.memory_snapshots
        )


class PerformanceMonitor:
    """Comprehensive performance monitor for E2E testing."""

    def __init__(self, test_name: str) -> None:
        """Initialize performance monitor.

        Args:
            test_name: Name of the test for monitoring context
        """
        self.test_name = test_name
        self.report = PerformanceReport(
            test_name=test_name, start_time=time.time(), end_time=0.0
        )
        self.monitoring_active = False
        self.logger = logging.getLogger(f"{__name__}.{test_name}")

    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.report.start_time = time.time()
        self.monitoring_active = True
        self.logger.info(
            f"Performance monitoring started for {self.test_name}"
        )

    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.report.end_time = time.time()
        self.monitoring_active = False
        self.logger.info(
            f"Performance monitoring stopped for {self.test_name}"
        )

    def measure_page_load(
        self, driver: WebDriver, url: str | None = None
    ) -> PageLoadMetrics:
        """Measure comprehensive page load performance metrics.

        Args:
            driver: WebDriver instance
            url: Optional URL to navigate to

        Returns:
            PageLoadMetrics with timing information
        """
        if url:
            start_time = time.time()
            driver.get(url)
        else:
            start_time = time.time()

        try:
            # Wait for page to be ready
            WebDriverWait(driver, 30).until(
                lambda d: d.execute_script("return document.readyState")
                == "complete"
            )

            # Get navigation timing API data
            navigation_timing = driver.execute_script(
                """
                const perfData = performance.getEntriesByType('navigation')[0];
                const paintEntries = performance.getEntriesByType('paint');

                return {
                    domContentLoadedEventEnd:
                        perfData.domContentLoadedEventEnd,
                    loadEventEnd: perfData.loadEventEnd,
                    domContentLoadedEventStart:
                        perfData.domContentLoadedEventStart,
                    navigationStart: perfData.navigationStart,
                    firstPaint: paintEntries.find(
                        entry => entry.name === 'first-paint'
                    )?.startTime || null,
                    firstContentfulPaint: paintEntries.find(
                        entry => entry.name === 'first-contentful-paint'
                    )?.startTime || null
                };
            """
            )

            # Calculate metrics
            dom_loaded_time = (
                navigation_timing["domContentLoadedEventEnd"]
                - navigation_timing["navigationStart"]
            ) / 1000.0  # Convert to seconds

            load_complete_time = (
                navigation_timing["loadEventEnd"]
                - navigation_timing["navigationStart"]
            ) / 1000.0

            page_metrics: PageLoadMetrics = {
                "dom_content_loaded": dom_loaded_time,
                "load_complete": load_complete_time,
                "first_paint": (
                    navigation_timing["firstPaint"] / 1000.0
                    if navigation_timing["firstPaint"]
                    else None
                ),
                "first_contentful_paint": (
                    navigation_timing["firstContentfulPaint"] / 1000.0
                    if navigation_timing["firstContentfulPaint"]
                    else None
                ),
                "largest_contentful_paint": None,  # Need additional JS for LCP
            }

            self.report.page_loads.append(page_metrics)

            self.logger.info(
                f"Page load measured: DOM {dom_loaded_time:.3f}s, "
                f"Complete {load_complete_time:.3f}s"
            )
            return page_metrics

        except Exception as e:
            self.logger.error(f"Failed to measure page load performance: {e}")

            # Fallback measurement
            fallback_time = time.time() - start_time
            fallback_metrics: PageLoadMetrics = {
                "dom_content_loaded": fallback_time,
                "load_complete": fallback_time,
                "first_paint": None,
                "first_contentful_paint": None,
                "largest_contentful_paint": None,
            }

            self.report.page_loads.append(fallback_metrics)
            return fallback_metrics

    def monitor_memory_usage(
        self, include_browser_process: bool = True
    ) -> ExtendedMemoryMetrics:
        """Monitor current memory usage.

        Args:
            include_browser_process: Whether to include browser process

        Returns:
            MemoryMetrics with current usage information
        """
        try:
            # Get current process memory
            current_process = psutil.Process()
            memory_info = current_process.memory_info()

            # Get browser memory if requested
            browser_memory = 0.0
            if include_browser_process:
                for proc in psutil.process_iter(
                    ["pid", "name", "memory_info"]
                ):
                    try:
                        if (
                            "chrome" in proc.info["name"].lower()
                            or "firefox" in proc.info["name"].lower()
                        ):
                            browser_memory += proc.info["memory_info"].rss / (
                                1024 * 1024
                            )
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

            metrics: ExtendedMemoryMetrics = {
                "rss_memory_mb": memory_info.rss
                / (1024 * 1024),  # Convert to MB
                "vms_memory_mb": memory_info.vms / (1024 * 1024),
                "cpu_percent": current_process.cpu_percent(),
                "memory_percent": current_process.memory_percent(),
                "timestamp": time.time(),
                "browser_memory_mb": browser_memory,
            }

            self.report.memory_snapshots.append(metrics)

            self.logger.debug(
                f"Memory snapshot: RSS {metrics['rss_memory_mb']:.1f}MB, "
                f"CPU {metrics['cpu_percent']:.1f}%"
            )
            return metrics

        except Exception as e:
            self.logger.error(f"Failed to monitor memory usage: {e}")

            # Return empty metrics on failure
            return {
                "rss_memory_mb": 0.0,
                "vms_memory_mb": 0.0,
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "timestamp": time.time(),
            }

    def track_user_interaction(
        self, driver: WebDriver, interaction_type: str, target_element: str
    ) -> float:
        """Track latency of user interactions.

        Args:
            driver: WebDriver instance
            interaction_type: Type of interaction (click, type, etc.)
            target_element: CSS selector or description of target

        Returns:
            Interaction latency in seconds
        """
        start_time = time.time()

        try:
            if interaction_type == "click":
                element = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, target_element)
                    )
                )
                element.click()
            elif interaction_type == "page_ready":
                WebDriverWait(driver, 30).until(
                    lambda d: d.execute_script("return document.readyState")
                    == "complete"
                )

            latency = time.time() - start_time

            interaction_data = {
                "type": interaction_type,
                "target": target_element,
                "latency": latency,
                "timestamp": time.time(),
                "success": True,
            }

            self.report.interactions.append(interaction_data)

            self.logger.debug(
                f"Interaction tracked: {interaction_type} on "
                f"{target_element} = {latency:.3f}s"
            )
            return latency

        except Exception as e:
            latency = time.time() - start_time

            interaction_data = {
                "type": interaction_type,
                "target": target_element,
                "latency": latency,
                "timestamp": time.time(),
                "success": False,
                "error": str(e),
            }

            self.report.interactions.append(interaction_data)

            self.logger.error(f"Interaction tracking failed: {e}")
            return latency

    def add_custom_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "ms",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Add a custom performance metric.

        Args:
            metric_name: Name of the metric
            value: Measured value
            unit: Unit of measurement
            context: Additional context information
        """
        metric: PerformanceMetric = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "timestamp": time.time(),
            "context": context or {},
        }

        self.report.metrics.append(metric)
        self.logger.info(
            f"Custom metric added: {metric_name} = {value} {unit}"
        )

    def generate_report(
        self, output_file: Path | str | None = None
    ) -> dict[str, Any]:
        """Generate comprehensive performance report.

        Args:
            output_file: Optional file path to save report

        Returns:
            Performance report as dictionary
        """
        if self.monitoring_active:
            self.stop_monitoring()

        report_data = {
            "test_name": self.report.test_name,
            "duration": self.report.total_duration,
            "summary": {
                "total_page_loads": len(self.report.page_loads),
                "average_page_load_time": self.report.average_page_load_time,
                "peak_memory_usage_mb": self.report.peak_memory_usage,
                "total_interactions": len(self.report.interactions),
                "successful_interactions": sum(
                    1
                    for i in self.report.interactions
                    if i.get("success", False)
                ),
            },
            "detailed_metrics": {
                "page_loads": self.report.page_loads,
                "memory_snapshots": self.report.memory_snapshots,
                "interactions": self.report.interactions,
                "custom_metrics": self.report.metrics,
            },
            "performance_analysis": self._analyze_performance(),
        }

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, default=str)

            self.logger.info(f"Performance report saved to {output_path}")

        return report_data

    def _analyze_performance(self) -> dict[str, Any]:
        """Analyze performance data and provide insights."""
        analysis: dict[str, Any] = {
            "warnings": [],
            "recommendations": [],
            "performance_score": 100,  # Start with perfect score
        }

        # Analyze page load times
        if self.report.page_loads:
            avg_load_time = self.report.average_page_load_time

            if avg_load_time > 5.0:
                analysis["warnings"].append(
                    f"High average page load time: {avg_load_time:.2f}s"
                )
                analysis["recommendations"].append(
                    "Consider optimizing page load performance"
                )
                analysis["performance_score"] -= 20
            elif avg_load_time > 3.0:
                analysis["warnings"].append(
                    f"Moderate page load time: {avg_load_time:.2f}s"
                )
                analysis["performance_score"] -= 10

        # Analyze memory usage
        if self.report.memory_snapshots:
            peak_memory = self.report.peak_memory_usage

            if peak_memory > 500:  # 500MB threshold
                analysis["warnings"].append(
                    f"High memory usage: {peak_memory:.1f}MB"
                )
                analysis["recommendations"].append(
                    "Monitor memory usage for potential leaks"
                )
                analysis["performance_score"] -= 15

        # Analyze interaction latency
        if self.report.interactions:
            failed_interactions = [
                i
                for i in self.report.interactions
                if not i.get("success", True)
            ]

            if failed_interactions:
                analysis["warnings"].append(
                    f"{len(failed_interactions)} interactions failed"
                )
                analysis["performance_score"] -= 10 * len(failed_interactions)

        return analysis


# Utility functions for common performance measurement scenarios


def measure_page_load_time(driver: WebDriver, url: str) -> float:
    """Measure page load time for a specific URL.

    Args:
        driver: WebDriver instance
        url: URL to measure

    Returns:
        Page load time in seconds
    """
    monitor = PerformanceMonitor("page_load_measurement")
    monitor.start_monitoring()

    metrics = monitor.measure_page_load(driver, url)

    monitor.stop_monitoring()

    return metrics["load_complete"]


def monitor_memory_usage(
    duration_seconds: float = 60.0, interval_seconds: float = 5.0
) -> list[MemoryMetrics]:
    """Monitor memory usage over a specified duration.

    Args:
        duration_seconds: How long to monitor
        interval_seconds: Interval between measurements

    Returns:
        List of memory metrics snapshots
    """
    monitor = PerformanceMonitor("memory_monitoring")
    snapshots: list[MemoryMetrics] = []

    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        snapshot = monitor.monitor_memory_usage()
        snapshots.append(snapshot)

        time.sleep(interval_seconds)

    return snapshots


def track_user_interaction_latency(
    driver: WebDriver, interactions: list[tuple[str, str]]
) -> dict[str, float]:
    """Track latency for multiple user interactions.

    Args:
        driver: WebDriver instance
        interactions: List of (interaction_type, target_element) tuples

    Returns:
        Dictionary of interaction latencies
    """
    monitor = PerformanceMonitor("interaction_latency")
    latencies: dict[str, float] = {}

    for interaction_type, target_element in interactions:
        latency = monitor.track_user_interaction(
            driver, interaction_type, target_element
        )
        key = f"{interaction_type}_{target_element}"
        latencies[key] = latency

    return latencies


def generate_performance_report(
    test_name: str,
    page_loads: list[PageLoadMetrics],
    memory_snapshots: list[MemoryMetrics],
    interactions: list[dict[str, Any]],
    output_file: Path | str | None = None,
) -> dict[str, Any]:
    """Generate a performance report from collected metrics.

    Args:
        test_name: Name of the test
        page_loads: Page load metrics
        memory_snapshots: Memory usage snapshots
        interactions: User interaction data
        output_file: Optional output file path

    Returns:
        Generated performance report
    """
    monitor = PerformanceMonitor(test_name)

    # Populate report with provided data
    monitor.report.page_loads = page_loads
    monitor.report.memory_snapshots = memory_snapshots
    monitor.report.interactions = interactions

    return monitor.generate_report(output_file)
