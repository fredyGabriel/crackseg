"""
Test report generator implementation. This module provides the main
TestReportGenerator class that orchestrates the collection of data
from performance monitoring, capture systems, and test execution
results to generate comprehensive reports.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.e2e.capture.storage import CaptureStorage, StorageConfig
from tests.e2e.helpers.performance_monitoring import PerformanceMonitor
from tests.e2e.reporting.config import ReportConfig
from tests.e2e.reporting.models import (
    ExecutionSummary,
    ReportFormat,
    TestResult,
    TestStatus,
)

logger = logging.getLogger(__name__)


class TestReportGenerator:
    """
    Main class for generating comprehensive test reports. Integrates with
    existing performance monitoring and capture systems to provide
    detailed reporting with metrics, trends, and insights.
    """

    def __init__(
        self,
        config: ReportConfig | None = None,
        capture_storage: CaptureStorage | None = None,
    ) -> None:
        """
        Initialize the test report generator. Args: config: Report generation
        configuration capture_storage: Existing capture storage instance
        """
        self.config = config or ReportConfig()
        self.capture_storage = capture_storage or CaptureStorage(
            StorageConfig(base_dir=self.config.output_dir / "artifacts")
        )
        self._setup_directories()
        self._test_results: list[TestResult] = []
        self._execution_start_time = time.time()
        self._performance_monitors: dict[str, PerformanceMonitor] = {}

        logger.info(f"Test report generator initialized: {self.config.mode}")

    def start_test_execution(self, session_id: str | None = None) -> None:
        """
        Mark the start of test execution session. Args: session_id: Optional
        session identifier
        """
        self._execution_start_time = time.time()
        self._test_results.clear()
        self._performance_monitors.clear()

        logger.info(
            f"Test execution session started: {session_id or 'default'}"
        )

    def register_test_start(
        self, test_name: str, enable_performance_monitoring: bool = True
    ) -> str:
        """
        Register the start of an individual test. Args: test_name: Name of the
        test enable_performance_monitoring: Enable performance monitoring
        Returns: Test ID for tracking
        """
        test_id = f"{test_name}_{int(time.time())}"

        if enable_performance_monitoring and self.config.include_performance:
            monitor = PerformanceMonitor(test_name)
            monitor.start_monitoring()
            self._performance_monitors[test_id] = monitor

        logger.debug(f"Test registered: {test_name} (ID: {test_id})")
        return test_id

    def register_test_completion(
        self,
        test_id: str,
        test_name: str,
        status: TestStatus,
        duration: float,
        error_message: str | None = None,
        failure_reason: str | None = None,
        artifact_paths: list[str] | None = None,
    ) -> None:
        """
        Register the completion of a test. Args: test_id: Test identifier
        test_name: Name of the test status: Test execution status duration:
        Test duration in seconds error_message: Error message if test failed
        failure_reason: Reason for failure artifact_paths: Paths to artifacts
        (screenshots, videos)
        """
        end_time = time.time()
        start_time = end_time - duration

        # Get performance data if monitoring was enabled
        performance_data = None
        if test_id in self._performance_monitors:
            monitor = self._performance_monitors[test_id]
            monitor.stop_monitoring()
            performance_data = monitor.generate_report()

        # Create test result
        test_result: TestResult = {
            "test_id": test_id,
            "test_name": test_name,
            "status": status.value,
            "duration": duration,
            "start_time": start_time,
            "end_time": end_time,
            "error_message": error_message,
            "failure_reason": failure_reason,
            "performance_data": performance_data,
            "artifacts": artifact_paths or [],
        }

        self._test_results.append(test_result)
        logger.debug(f"Test completed: {test_name} ({status.value})")

    def generate_execution_summary(self) -> ExecutionSummary:
        """
        Generate summary statistics for the test execution. Returns:
        ExecutionSummary with key metrics
        """
        if not self._test_results:
            return ExecutionSummary(
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                error=0,
                success_rate=0.0,
                total_duration=0.0,
                start_time=datetime.fromtimestamp(
                    self._execution_start_time
                ).isoformat(),
                end_time=datetime.now().isoformat(),
            )

        status_counts = {}
        for status in TestStatus:
            status_counts[status.value] = sum(
                1
                for result in self._test_results
                if result["status"] == status.value
            )

        total_tests = len(self._test_results)
        passed_tests = status_counts.get("passed", 0)
        success_rate = (
            (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
        )

        total_duration = sum(
            result["duration"] for result in self._test_results
        )

        return ExecutionSummary(
            total_tests=total_tests,
            passed=status_counts.get("passed", 0),
            failed=status_counts.get("failed", 0),
            skipped=status_counts.get("skipped", 0),
            error=status_counts.get("error", 0),
            success_rate=success_rate,
            total_duration=total_duration,
            start_time=datetime.fromtimestamp(
                self._execution_start_time
            ).isoformat(),
            end_time=datetime.now().isoformat(),
        )

    def generate_report(
        self, output_filename: str | None = None
    ) -> dict[str, Path]:
        """
        Generate comprehensive test report in configured formats. Args:
        output_filename: Base filename for reports Returns: Dictionary mapping
        format to generated file path
        """
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"test_report_{timestamp}"

        execution_summary = self.generate_execution_summary()
        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "config_mode": self.config.mode.value,
                "report_version": "1.0",
            },
            "execution_summary": execution_summary,
            "test_results": self._test_results,
            "performance_summary": self._generate_performance_summary(),
            "artifact_summary": self._generate_artifact_summary(),
        }

        generated_files = {}

        # Generate reports in requested formats
        for report_format in self.config.formats:
            output_path = self._generate_report_by_format(
                report_data, output_filename, report_format
            )
            generated_files[report_format.value] = output_path

        logger.info(f"Generated {len(generated_files)} report files")
        return generated_files

    def _setup_directories(self) -> None:
        """Setup required directories for report generation."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "assets").mkdir(exist_ok=True)
        (self.config.output_dir / "data").mkdir(exist_ok=True)

    def _generate_performance_summary(self) -> dict[str, Any]:
        """Generate performance metrics summary."""
        if (
            not self.config.include_performance
            or not self._performance_monitors
        ):
            return {}

        performance_stats = {
            "tests_with_performance_data": len(self._performance_monitors),
            "average_page_load_time": 0.0,
            "peak_memory_usage": 0.0,
            "performance_violations": [],
        }

        page_load_times = []
        memory_peaks = []

        for monitor in self._performance_monitors.values():
            report = monitor.report
            if report.page_loads:
                page_load_times.append(report.average_page_load_time)
            if report.memory_snapshots:
                memory_peaks.append(report.peak_memory_usage)

            # Check performance thresholds
            if (
                report.average_page_load_time
                > self.config.performance_thresholds.get("page_load_max", 3.0)
            ):
                performance_stats["performance_violations"].append(
                    {
                        "test": monitor.test_name,
                        "metric": "page_load_time",
                        "value": report.average_page_load_time,
                        "threshold": self.config.performance_thresholds[
                            "page_load_max"
                        ],
                    }
                )

        if page_load_times:
            performance_stats["average_page_load_time"] = sum(
                page_load_times
            ) / len(page_load_times)
        if memory_peaks:
            performance_stats["peak_memory_usage"] = max(memory_peaks)

        return performance_stats

    def _generate_artifact_summary(self) -> dict[str, Any]:
        """Generate artifacts summary."""
        if not self.config.include_artifacts:
            return {}

        total_artifacts = sum(
            len(result["artifacts"]) for result in self._test_results
        )
        failure_artifacts = sum(
            len(result["artifacts"])
            for result in self._test_results
            if result["status"] in ["failed", "error"]
        )

        return {
            "total_artifacts": total_artifacts,
            "failure_artifacts": failure_artifacts,
            "artifact_retention_days": self.config.retention_days,
        }

    def _generate_report_by_format(
        self,
        report_data: dict[str, Any],
        filename: str,
        report_format: ReportFormat,
    ) -> Path:
        """Generate report in specific format."""
        output_path = (
            self.config.output_dir / f"{filename}.{report_format.value}"
        )

        if report_format == ReportFormat.JSON:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

        elif report_format == ReportFormat.HTML:
            # Basic HTML report - will be enhanced by HTMLReportExporter
            html_content = self._generate_basic_html_report(report_data)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

        logger.debug(f"Generated {report_format.value} report: {output_path}")
        return output_path

    def _generate_basic_html_report(self, report_data: dict[str, Any]) -> str:
        """Generate basic HTML report."""
        summary = report_data["execution_summary"]
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CrackSeg E2E Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px;
                          border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .metrics {{ display: grid;
                          grid-template-columns: repeat(auto-fit,
                          minmax(200px, 1fr)); gap: 10px; }}
                .metric {{ background: white; padding: 10px;
                         border: 1px solid #ddd; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h1>CrackSeg E2E Test Report</h1>
            <div class="summary">
                <h2>Execution Summary</h2>
                <div class="metrics">
                    <div class="metric">
                        <strong>Total Tests:</strong> {summary["total_tests"]}
                    </div>
                    <div class="metric">
                        <strong class="passed">Passed:</strong>
                        {summary["passed"]}
                    </div>
                    <div class="metric">
                        <strong class="failed">Failed:</strong>
                        {summary["failed"]}
                    </div>
                    <div class="metric">
                        <strong>Success Rate:</strong>
                        {summary["success_rate"]:.1f}%
                    </div>
                    <div class="metric">
                        <strong>Duration:</strong>
                        {summary["total_duration"]:.2f}s
                    </div>
                </div>
            </div>
            <p><em>Generated at:
            {report_data["metadata"]["generated_at"]}</em></p>
        </body>
        </html>
        """
