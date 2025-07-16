"""Visual Regression and Performance Benchmarking System.

This module extends the existing visual testing framework and benchmark tools
to provide comprehensive visual regression testing and performance monitoring
for GUI components. Part of subtask 7.7 - Visual Regression and Performance
Benchmarks.
"""

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from .test_benchmark import BenchmarkConfig, TestBenchmark
from .visual_testing_framework import (
    ComponentPerformanceProfiler,
    PerformanceProfile,
    VisualRegressionTester,
    VisualTestSnapshot,
)

# Type aliases for better type checking
MockTestFunction = Callable[[Mock], None]
ComponentTestSuite = dict[str, MockTestFunction]


@dataclass
class VisualRegressionResult:
    """Result from visual regression comparison."""

    test_name: str
    passed: bool
    similarity_score: float
    baseline_checksum: str
    current_checksum: str
    deviation_percentage: float = 0.0
    regression_detected: bool = False
    error_message: str = ""


@dataclass
class PerformanceRegressionResult:
    """Result from performance regression analysis."""

    component_name: str
    current_profile: PerformanceProfile
    baseline_profile: PerformanceProfile | None
    performance_regression: bool = False
    degradation_percentage: float = 0.0
    memory_regression: bool = False
    memory_increase_mb: float = 0.0
    alert_triggered: bool = False


@dataclass
class ComprehensiveTestReport:
    """Complete test report combining visual and performance results."""

    timestamp: float
    test_session_id: str
    visual_results: list[VisualRegressionResult] = field(default_factory=list)
    performance_results: list[PerformanceRegressionResult] = field(
        default_factory=list
    )
    overall_status: str = "pending"
    total_regressions: int = 0


class VisualRegressionDetector:
    """Advanced visual regression detection with screenshot comparison."""

    def __init__(
        self,
        baselines_dir: Path,
        tolerance: float = 0.02,
        regression_threshold: float = 0.05,
    ) -> None:
        """Initialize visual regression detector.

        Args:
            baselines_dir: Directory containing baseline snapshots
            tolerance: Tolerance for visual differences (2% default)
            regression_threshold: Threshold for regression detection
            (5% default)
        """
        self.baselines_dir = baselines_dir
        self.tolerance = tolerance
        self.regression_threshold = regression_threshold
        self.baselines_dir.mkdir(parents=True, exist_ok=True)

    def compare_with_baseline(
        self, current_snapshot: VisualTestSnapshot
    ) -> VisualRegressionResult:
        """Compare current snapshot with baseline.

        Args:
            current_snapshot: Current test snapshot

        Returns:
            Visual regression comparison result
        """
        baseline_file = (
            self.baselines_dir / f"{current_snapshot.test_name}.json"
        )

        if not baseline_file.exists():
            # First run - save as baseline
            self._save_baseline(current_snapshot)
            return VisualRegressionResult(
                test_name=current_snapshot.test_name,
                passed=True,
                similarity_score=1.0,
                baseline_checksum=current_snapshot.checksum,
                current_checksum=current_snapshot.checksum,
            )

        # Load baseline
        baseline_snapshot = self._load_baseline(baseline_file)

        # Calculate similarity
        similarity = self._calculate_similarity(
            baseline_snapshot, current_snapshot
        )

        # Determine regression
        deviation = 1.0 - similarity
        regression_detected = deviation > self.regression_threshold

        return VisualRegressionResult(
            test_name=current_snapshot.test_name,
            passed=deviation <= self.tolerance,
            similarity_score=similarity,
            baseline_checksum=baseline_snapshot.checksum,
            current_checksum=current_snapshot.checksum,
            deviation_percentage=deviation * 100,
            regression_detected=regression_detected,
        )

    def _calculate_similarity(
        self, baseline: VisualTestSnapshot, current: VisualTestSnapshot
    ) -> float:
        """Calculate similarity between snapshots."""
        if baseline.checksum == current.checksum:
            return 1.0

        # Simple text-based similarity for mock render output
        baseline_content = baseline.render_output
        current_content = current.render_output

        if baseline_content == current_content:
            return 1.0

        # Calculate character-level similarity
        max_len = max(len(baseline_content), len(current_content))
        if max_len == 0:
            return 1.0

        common_chars = sum(
            1
            for a, b in zip(baseline_content, current_content, strict=False)
            if a == b
        )
        return common_chars / max_len

    def _save_baseline(self, snapshot: VisualTestSnapshot) -> None:
        """Save snapshot as baseline."""
        baseline_file = self.baselines_dir / f"{snapshot.test_name}.json"
        snapshot_data = {
            "test_name": snapshot.test_name,
            "component_type": snapshot.component_type,
            "render_output": snapshot.render_output,
            "timestamp": snapshot.timestamp,
            "checksum": snapshot.checksum,
        }
        with baseline_file.open("w") as f:
            json.dump(snapshot_data, f, indent=2)

    def _load_baseline(self, baseline_file: Path) -> VisualTestSnapshot:
        """Load baseline snapshot from file."""
        with baseline_file.open("r") as f:
            data = json.load(f)
        return VisualTestSnapshot(**data)


class PerformanceRegressionDetector:
    """Advanced performance regression detection and alerting."""

    def __init__(
        self,
        baselines_dir: Path,
        performance_threshold: float = 0.20,
        memory_threshold_mb: float = 10.0,
    ) -> None:
        """Initialize performance regression detector.

        Args:
            baselines_dir: Directory for baseline performance data
            performance_threshold: Performance degradation threshold
                (20% default)
            memory_threshold_mb: Memory increase threshold (10MB default)
        """
        self.baselines_dir = baselines_dir
        self.performance_threshold = performance_threshold
        self.memory_threshold_mb = memory_threshold_mb
        self.baselines_dir.mkdir(parents=True, exist_ok=True)

    def analyze_performance_regression(
        self, current_profile: PerformanceProfile
    ) -> PerformanceRegressionResult:
        """Analyze performance regression against baseline.

        Args:
            current_profile: Current performance profile

        Returns:
            Performance regression analysis result
        """
        baseline_file = (
            self.baselines_dir / f"{current_profile.component_name}_perf.json"
        )

        if not baseline_file.exists():
            # First run - save as baseline
            self._save_performance_baseline(current_profile)
            return PerformanceRegressionResult(
                component_name=current_profile.component_name,
                current_profile=current_profile,
                baseline_profile=None,
            )

        # Load baseline
        baseline_profile = self._load_performance_baseline(baseline_file)

        # Calculate regression metrics
        render_time_ratio = (
            current_profile.render_time_ms / baseline_profile.render_time_ms
        )
        memory_increase = (
            current_profile.memory_usage_mb - baseline_profile.memory_usage_mb
        )

        performance_regression = render_time_ratio > (
            1.0 + self.performance_threshold
        )
        memory_regression = memory_increase > self.memory_threshold_mb

        degradation_percentage = max(0, (render_time_ratio - 1.0) * 100)
        alert_triggered = performance_regression or memory_regression

        return PerformanceRegressionResult(
            component_name=current_profile.component_name,
            current_profile=current_profile,
            baseline_profile=baseline_profile,
            performance_regression=performance_regression,
            degradation_percentage=degradation_percentage,
            memory_regression=memory_regression,
            memory_increase_mb=memory_increase,
            alert_triggered=alert_triggered,
        )

    def _save_performance_baseline(self, profile: PerformanceProfile) -> None:
        """Save performance profile as baseline."""
        baseline_file = (
            self.baselines_dir / f"{profile.component_name}_perf.json"
        )
        profile_data = {
            "component_name": profile.component_name,
            "render_time_ms": profile.render_time_ms,
            "memory_usage_mb": profile.memory_usage_mb,
            "interaction_latency_ms": profile.interaction_latency_ms,
            "widget_count": profile.widget_count,
            "complexity_score": profile.complexity_score,
            "timestamp": profile.timestamp,
        }
        with baseline_file.open("w") as f:
            json.dump(profile_data, f, indent=2)

    def _load_performance_baseline(
        self, baseline_file: Path
    ) -> PerformanceProfile:
        """Load performance baseline from file."""
        with baseline_file.open("r") as f:
            data = json.load(f)
        return PerformanceProfile(**data)


class ComprehensiveRegressionTester:
    """
    Comprehensive testing system combining visual and performance regression.
    """

    def __init__(
        self,
        test_artifacts_dir: Path,
        visual_tolerance: float = 0.02,
        performance_threshold: float = 0.20,
    ) -> None:
        """Initialize comprehensive regression tester.

        Args:
            test_artifacts_dir: Directory for test artifacts
            visual_tolerance: Visual difference tolerance
            performance_threshold: Performance degradation threshold
        """
        self.test_artifacts_dir = test_artifacts_dir
        self.test_artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.visual_tester = VisualRegressionTester(
            test_artifacts_dir / "visual_snapshots"
        )
        self.performance_profiler = ComponentPerformanceProfiler()

        # Initialize regression detectors
        self.visual_detector = VisualRegressionDetector(
            test_artifacts_dir / "visual_baselines", tolerance=visual_tolerance
        )
        self.performance_detector = PerformanceRegressionDetector(
            test_artifacts_dir / "performance_baselines",
            performance_threshold=performance_threshold,
        )

        # Initialize benchmark system
        benchmark_config = BenchmarkConfig(
            test_patterns=["tests/gui/**/*.py"],
            output_dir=test_artifacts_dir / "benchmarks",
        )
        self.benchmark_system = TestBenchmark(benchmark_config)

    def run_comprehensive_test(
        self, component_tests: ComponentTestSuite
    ) -> ComprehensiveTestReport:
        """Run comprehensive visual and performance testing.

        Args:
            component_tests: Dictionary of test name to test function

        Returns:
            Complete test report
        """
        session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        report = ComprehensiveTestReport(
            timestamp=time.time(), test_session_id=session_id
        )

        for test_name, test_func in component_tests.items():
            try:
                # Run visual regression test
                visual_result = self._run_visual_test(test_name, test_func)
                report.visual_results.append(visual_result)

                # Run performance regression test
                perf_result = self._run_performance_test(test_name, test_func)
                report.performance_results.append(perf_result)

            except Exception as e:
                # Log error and continue
                error_result = VisualRegressionResult(
                    test_name=test_name,
                    passed=False,
                    similarity_score=0.0,
                    baseline_checksum="",
                    current_checksum="",
                    error_message=str(e),
                )
                report.visual_results.append(error_result)

        # Calculate summary metrics
        self._finalize_report(report)
        self._save_report(report)

        return report

    def _run_visual_test(
        self, test_name: str, test_func: MockTestFunction
    ) -> VisualRegressionResult:
        """Run visual regression test for component."""
        mock_st = Mock()

        # Execute test function with mock
        test_func(mock_st)

        # Capture snapshot
        snapshot = self.visual_tester.capture_component_snapshot(
            test_name, "gui_component", mock_st
        )

        # Compare with baseline
        return self.visual_detector.compare_with_baseline(snapshot)

    def _run_performance_test(
        self, test_name: str, test_func: MockTestFunction
    ) -> PerformanceRegressionResult:
        """Run performance regression test for component."""

        # Create wrapper function for profiling
        def wrapper() -> None:
            mock_st = Mock()
            test_func(mock_st)

        # Profile component performance
        profile = self.performance_profiler.profile_component_render(  # type: ignore
            test_name,
            wrapper,
        )

        # Analyze regression
        return self.performance_detector.analyze_performance_regression(
            profile
        )

    def _finalize_report(self, report: ComprehensiveTestReport) -> None:
        """Finalize report with summary metrics."""
        visual_regressions = sum(
            1 for r in report.visual_results if r.regression_detected
        )
        performance_regressions = sum(
            1 for r in report.performance_results if r.alert_triggered
        )

        report.total_regressions = visual_regressions + performance_regressions

        if report.total_regressions == 0:
            report.overall_status = "passed"
        elif report.total_regressions < 3:
            report.overall_status = "warning"
        else:
            report.overall_status = "failed"

    def _save_report(self, report: ComprehensiveTestReport) -> None:
        """Save comprehensive test report."""
        report_file = (
            self.test_artifacts_dir
            / f"regression_report_{report.test_session_id}.json"
        )

        # Convert to serializable format
        report_data = {
            "timestamp": report.timestamp,
            "test_session_id": report.test_session_id,
            "overall_status": report.overall_status,
            "total_regressions": report.total_regressions,
            "visual_results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "similarity_score": r.similarity_score,
                    "regression_detected": r.regression_detected,
                    "deviation_percentage": r.deviation_percentage,
                }
                for r in report.visual_results
            ],
            "performance_results": [
                {
                    "component_name": r.component_name,
                    "performance_regression": r.performance_regression,
                    "degradation_percentage": r.degradation_percentage,
                    "memory_regression": r.memory_regression,
                    "alert_triggered": r.alert_triggered,
                }
                for r in report.performance_results
            ],
        }

        with report_file.open("w") as f:
            json.dump(report_data, f, indent=2)


# Pytest fixtures
@pytest.fixture
def regression_tester(tmp_path: Path) -> ComprehensiveRegressionTester:
    """Pytest fixture for comprehensive regression testing."""
    return ComprehensiveRegressionTester(tmp_path / "regression_tests")


# Decorators for automated regression testing
def comprehensive_regression_test(
    visual_tolerance: float = 0.02, performance_threshold: float = 0.20
) -> Callable[[MockTestFunction], Callable[..., ComprehensiveTestReport]]:
    """Decorator for comprehensive regression testing."""

    def decorator(
        test_func: MockTestFunction,
    ) -> Callable[..., ComprehensiveTestReport]:
        def wrapper(*args: Any, **kwargs: Any) -> ComprehensiveTestReport:
            # Get or create regression tester
            tester = ComprehensiveRegressionTester(
                Path("test-artifacts") / "automated_regression"
            )

            # Run comprehensive test
            component_tests: ComponentTestSuite = {
                test_func.__name__: test_func
            }
            return tester.run_comprehensive_test(component_tests)

        return wrapper

    return decorator
