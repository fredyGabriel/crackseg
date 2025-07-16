"""Visual testing framework with regression testing capabilities.

This module provides visual regression testing and performance monitoring
for complex GUI components, completing the enhanced GUI testing framework.
Part of subtask 7.6 - GUI Testing Framework Enhancement.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest


@dataclass
class VisualTestSnapshot:
    """Snapshot of a visual test state."""

    test_name: str
    component_type: str
    render_output: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self) -> None:
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate checksum of render output."""
        content = (
            f"{self.test_name}:{self.component_type}:{self.render_output}"
        )
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class PerformanceProfile:
    """Performance profile for a GUI component."""

    component_name: str
    render_time_ms: float
    memory_usage_mb: float
    interaction_latency_ms: float
    widget_count: int
    complexity_score: float
    timestamp: float = field(default_factory=time.time)


class VisualRegressionTester:
    """Visual regression testing system for GUI components."""

    def __init__(self, snapshots_dir: Path) -> None:
        self.snapshots_dir = snapshots_dir
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._current_snapshots: dict[str, VisualTestSnapshot] = {}

    def capture_component_snapshot(
        self,
        test_name: str,
        component_type: str,
        mock_st: Mock,
        metadata: dict[str, Any] | None = None,
    ) -> VisualTestSnapshot:
        """Capture a snapshot of component render state."""
        # Extract render information from mock calls
        render_output = self._extract_render_output(mock_st)

        snapshot = VisualTestSnapshot(
            test_name=test_name,
            component_type=component_type,
            render_output=render_output,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self._current_snapshots[test_name] = snapshot
        return snapshot

    def save_snapshot(self, snapshot: VisualTestSnapshot) -> Path:
        """Save snapshot to disk."""
        snapshot_file = self.snapshots_dir / f"{snapshot.test_name}.json"

        snapshot_data = {
            "test_name": snapshot.test_name,
            "component_type": snapshot.component_type,
            "render_output": snapshot.render_output,
            "timestamp": snapshot.timestamp,
            "metadata": snapshot.metadata,
            "checksum": snapshot.checksum,
        }

        with open(snapshot_file, "w") as f:
            json.dump(snapshot_data, f, indent=2)

        return snapshot_file

    def load_snapshot(self, test_name: str) -> VisualTestSnapshot | None:
        """Load snapshot from disk."""
        snapshot_file = self.snapshots_dir / f"{test_name}.json"

        if not snapshot_file.exists():
            return None

        with open(snapshot_file) as f:
            data = json.load(f)

        return VisualTestSnapshot(
            test_name=data["test_name"],
            component_type=data["component_type"],
            render_output=data["render_output"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
            checksum=data.get("checksum", ""),
        )

    def compare_snapshots(
        self, current: VisualTestSnapshot, baseline: VisualTestSnapshot
    ) -> dict[str, Any]:
        """Compare two snapshots and return differences."""
        differences = {
            "checksum_match": current.checksum == baseline.checksum,
            "content_changes": [],
            "metadata_changes": {},
        }

        # Detailed content comparison
        if not differences["checksum_match"]:
            differences["content_changes"] = self._analyze_content_differences(
                current.render_output, baseline.render_output
            )

        # Metadata comparison
        for key in set(current.metadata.keys()) | set(
            baseline.metadata.keys()
        ):
            current_val = current.metadata.get(key)
            baseline_val = baseline.metadata.get(key)
            if current_val != baseline_val:
                differences["metadata_changes"][key] = {
                    "current": current_val,
                    "baseline": baseline_val,
                }

        return differences

    def assert_visual_regression(
        self, test_name: str, tolerance: float = 0.0
    ) -> bool:
        """Assert that visual output matches baseline snapshot."""
        current = self._current_snapshots.get(test_name)
        if not current:
            raise ValueError(
                f"No current snapshot found for test: {test_name}"
            )

        baseline = self.load_snapshot(test_name)
        if not baseline:
            # First run - save as baseline
            self.save_snapshot(current)
            return True

        comparison = self.compare_snapshots(current, baseline)

        if not comparison["checksum_match"]:
            if tolerance == 0.0:
                raise AssertionError(
                    f"Visual regression detected in {test_name}:\n"
                    f"Content changes: {comparison['content_changes']}\n"
                    f"Metadata changes: {comparison['metadata_changes']}"
                )
            else:
                # TODO: Implement tolerance-based comparison
                # For now, strict comparison only
                raise AssertionError(
                    f"Visual regression detected in {test_name}"
                )

        return True

    def _extract_render_output(self, mock_st: Mock) -> str:
        """Extract render output from Streamlit mock calls."""
        render_calls = []

        # Extract common rendering calls
        for method_name in [
            "write",
            "markdown",
            "text",
            "header",
            "subheader",
            "title",
            "info",
            "success",
            "warning",
            "error",
        ]:
            if hasattr(mock_st, method_name):
                method = getattr(mock_st, method_name)
                for call in method.call_args_list:
                    render_calls.append(f"{method_name}: {call}")

        return "\n".join(render_calls)

    def _analyze_content_differences(
        self, current: str, baseline: str
    ) -> list[dict[str, Any]]:
        """Analyze differences between content strings."""
        current_lines = current.split("\n")
        baseline_lines = baseline.split("\n")

        differences = []

        # Simple line-by-line comparison
        max_lines = max(len(current_lines), len(baseline_lines))
        for i in range(max_lines):
            current_line = current_lines[i] if i < len(current_lines) else ""
            baseline_line = (
                baseline_lines[i] if i < len(baseline_lines) else ""
            )

            if current_line != baseline_line:
                differences.append(
                    {
                        "line": i + 1,
                        "current": current_line,
                        "baseline": baseline_line,
                        "type": "content_change",
                    }
                )

        return differences


class ComponentPerformanceProfiler:
    """Performance profiler for GUI components."""

    def __init__(self) -> None:
        self._profiles: dict[str, list[PerformanceProfile]] = {}

    def profile_component_render(
        self,
        component_name: str,
        render_func: callable,
        *args: Any,
        **kwargs: Any,
    ) -> PerformanceProfile:
        """Profile a component's rendering performance."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()
        result = render_func(*args, **kwargs)
        render_time = (time.time() - start_time) * 1000  # Convert to ms

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before

        # Estimate component complexity
        widget_count = self._estimate_widget_count(result)
        complexity_score = self._calculate_complexity_score(
            render_time, memory_usage, widget_count
        )

        profile = PerformanceProfile(
            component_name=component_name,
            render_time_ms=render_time,
            memory_usage_mb=memory_usage,
            interaction_latency_ms=0.0,  # Will be measured separately
            widget_count=widget_count,
            complexity_score=complexity_score,
        )

        if component_name not in self._profiles:
            self._profiles[component_name] = []
        self._profiles[component_name].append(profile)

        return profile

    def get_performance_baseline(
        self, component_name: str
    ) -> dict[str, float] | None:
        """Get performance baseline for a component."""
        if component_name not in self._profiles:
            return None

        profiles = self._profiles[component_name]
        if not profiles:
            return None

        render_times = [p.render_time_ms for p in profiles]
        memory_usages = [p.memory_usage_mb for p in profiles]

        return {
            "mean_render_time_ms": sum(render_times) / len(render_times),
            "max_render_time_ms": max(render_times),
            "mean_memory_usage_mb": sum(memory_usages) / len(memory_usages),
            "max_memory_usage_mb": max(memory_usages),
        }

    def assert_performance_regression(
        self,
        component_name: str,
        current_profile: PerformanceProfile,
        tolerance_percent: float = 20.0,
    ) -> bool:
        """Assert that performance hasn't regressed beyond tolerance."""
        baseline = self.get_performance_baseline(component_name)
        if not baseline:
            # First run - establish baseline
            return True

        render_regression = (
            (current_profile.render_time_ms - baseline["mean_render_time_ms"])
            / baseline["mean_render_time_ms"]
            * 100
        )

        memory_regression = (
            (
                current_profile.memory_usage_mb
                - baseline["mean_memory_usage_mb"]
            )
            / max(
                baseline["mean_memory_usage_mb"], 0.1
            )  # Avoid division by zero
            * 100
        )

        if render_regression > tolerance_percent:
            raise AssertionError(
                f"Render time regression detected for {component_name}: "
                f"{render_regression:.1f}% increase "
                f"({current_profile.render_time_ms:.2f}ms vs "
                f"{baseline['mean_render_time_ms']:.2f}ms baseline)"
            )

        if memory_regression > tolerance_percent:
            raise AssertionError(
                f"Memory usage regression detected for {component_name}: "
                f"{memory_regression:.1f}% increase "
                f"({current_profile.memory_usage_mb:.2f}MB vs "
                f"{baseline['mean_memory_usage_mb']:.2f}MB baseline)"
            )

        return True

    def _estimate_widget_count(self, render_result: Any) -> int:
        """Estimate number of widgets in render result."""
        # Simple heuristic - in real implementation, this would
        # analyze the mock calls to count widgets
        return 1

    def _calculate_complexity_score(
        self, render_time: float, memory_usage: float, widget_count: int
    ) -> float:
        """Calculate a complexity score for the component."""
        # Simple scoring algorithm
        time_score = min(render_time / 10.0, 10.0)  # Normalize to 0-10
        memory_score = min(memory_usage / 5.0, 10.0)  # Normalize to 0-10
        widget_score = min(widget_count / 5.0, 10.0)  # Normalize to 0-10

        return (time_score + memory_score + widget_score) / 3.0


class ComponentTestOrchestrator:
    """Orchestrates comprehensive testing of GUI components."""

    def __init__(self, snapshots_dir: Path) -> None:
        self.visual_tester = VisualRegressionTester(snapshots_dir)
        self.performance_profiler = ComponentPerformanceProfiler()

    def comprehensive_component_test(
        self,
        test_name: str,
        component_name: str,
        component_func: callable,
        mock_st: Mock,
        test_scenarios: list[dict[str, Any]],
        performance_tolerance: float = 20.0,
    ) -> dict[str, Any]:
        """Run comprehensive testing including visual and performance
        testing."""
        results = {
            "test_name": test_name,
            "component_name": component_name,
            "scenarios_tested": len(test_scenarios),
            "visual_tests": [],
            "performance_tests": [],
            "overall_success": True,
        }

        # Test each scenario
        for i, scenario in enumerate(test_scenarios):
            scenario_name = f"{test_name}_scenario_{i}"

            try:
                # Run component with scenario parameters
                scenario_params = scenario.get("params", {})
                component_func(**scenario_params)

                # Visual regression test
                snapshot = self.visual_tester.capture_component_snapshot(
                    scenario_name, component_name, mock_st, scenario
                )
                visual_passed = self.visual_tester.assert_visual_regression(
                    scenario_name
                )

                results["visual_tests"].append(
                    {
                        "scenario": scenario_name,
                        "passed": visual_passed,
                        "snapshot_checksum": snapshot.checksum,
                    }
                )

                # Performance test
                profile = self.performance_profiler.profile_component_render(
                    component_name, component_func, **scenario_params
                )
                performance_passed = (
                    self.performance_profiler.assert_performance_regression(
                        component_name, profile, performance_tolerance
                    )
                )

                results["performance_tests"].append(
                    {
                        "scenario": scenario_name,
                        "passed": performance_passed,
                        "render_time_ms": profile.render_time_ms,
                        "memory_usage_mb": profile.memory_usage_mb,
                        "complexity_score": profile.complexity_score,
                    }
                )

            except Exception as e:
                results["overall_success"] = False
                results["visual_tests"].append(
                    {
                        "scenario": scenario_name,
                        "passed": False,
                        "error": str(e),
                    }
                )
                results["performance_tests"].append(
                    {
                        "scenario": scenario_name,
                        "passed": False,
                        "error": str(e),
                    }
                )

        return results


# Pytest fixtures and decorators
@pytest.fixture
def visual_tester(tmp_path: Path) -> VisualRegressionTester:
    """Pytest fixture providing visual regression tester."""
    return VisualRegressionTester(tmp_path / "visual_snapshots")


@pytest.fixture
def performance_profiler() -> ComponentPerformanceProfiler:
    """Pytest fixture providing performance profiler."""
    return ComponentPerformanceProfiler()


@pytest.fixture
def test_orchestrator(tmp_path: Path) -> ComponentTestOrchestrator:
    """Pytest fixture providing test orchestrator."""
    return ComponentTestOrchestrator(tmp_path / "test_snapshots")


def visual_regression_test(tolerance: float = 0.0) -> callable:
    """Decorator for visual regression testing."""

    def decorator(test_func: callable) -> callable:
        def wrapper(
            visual_tester: VisualRegressionTester, *args: Any, **kwargs: Any
        ) -> Any:
            # Run the test
            result = test_func(*args, **kwargs)

            # Capture and assert visual regression
            test_name = test_func.__name__
            if (
                hasattr(visual_tester, "_current_snapshots")
                and test_name in visual_tester._current_snapshots
            ):
                visual_tester.assert_visual_regression(test_name, tolerance)

            return result

        return wrapper

    return decorator


def performance_test(tolerance_percent: float = 20.0) -> callable:
    """Decorator for performance regression testing."""

    def decorator(test_func: callable) -> callable:
        def wrapper(
            performance_profiler: ComponentPerformanceProfiler,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            # Profile the test execution
            component_name = test_func.__name__
            profile = performance_profiler.profile_component_render(
                component_name, test_func, *args, **kwargs
            )

            # Assert performance regression
            performance_profiler.assert_performance_regression(
                component_name, profile, tolerance_percent
            )

            return profile

        return wrapper

    return decorator
