"""
Unified performance testing framework. This module provides
comprehensive performance monitoring and profiling capabilities for GUI
component testing, consolidating functionality from various testing
frameworks.
"""

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import psutil


@dataclass
class PerformanceProfile:
    """Performance profile data for component testing."""

    component_name: str
    execution_time_ms: float
    memory_usage_mb: float
    timestamp: float
    metadata: dict[str, Any]


class UnifiedPerformanceTester:
    """Unified performance testing consolidating all performance functionality.

    This class provides comprehensive performance testing capabilities.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, list[PerformanceProfile]] = {}

    def profile_component_render(
        self,
        component_name: str,
        render_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> PerformanceProfile:
        """Profile a component's rendering performance."""
        # Get process for memory monitoring
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Time the rendering
        start_time = time.time()
        result = render_func(*args, **kwargs)
        render_time = (time.time() - start_time) * 1000  # Convert to ms

        # Measure memory after rendering
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before

        # Estimate component complexity
        widget_count = self._estimate_widget_count(result)
        complexity_score = self._calculate_complexity_score(
            render_time, memory_usage, widget_count
        )

        profile = PerformanceProfile(
            component_name=component_name,
            execution_time_ms=render_time,
            memory_usage_mb=memory_usage,
            timestamp=time.time(),
            metadata={
                "widget_count": widget_count,
                "complexity_score": complexity_score,
                "args_count": len(args),
                "kwargs_count": len(kwargs),
            },
        )

        # Store profile
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

        # Calculate baseline metrics from all profiles
        render_times = [p.execution_time_ms for p in profiles]
        memory_usages = [p.memory_usage_mb for p in profiles]

        return {
            "avg_render_time_ms": sum(render_times) / len(render_times),
            "max_render_time_ms": max(render_times),
            "min_render_time_ms": min(render_times),
            "avg_memory_usage_mb": sum(memory_usages) / len(memory_usages),
            "max_memory_usage_mb": max(memory_usages),
            "min_memory_usage_mb": min(memory_usages),
        }

    def assert_performance_regression(
        self,
        component_name: str,
        current_profile: PerformanceProfile,
        threshold_multiplier: float = 1.5,
    ) -> bool:
        """Assert that performance hasn't regressed beyond threshold."""
        baseline = self.get_performance_baseline(component_name)
        if not baseline:
            # No baseline to compare against
            return True

        current_time = current_profile.execution_time_ms
        baseline_time = baseline["avg_render_time_ms"]

        if current_time > baseline_time * threshold_multiplier:
            raise AssertionError(
                f"Performance regression detected for {component_name}: "
                f"Current: {current_time:.2f}ms, "
                f"Baseline: {baseline_time:.2f}ms "
                f"(threshold: {threshold_multiplier}x)"
            )

        current_memory = current_profile.memory_usage_mb
        baseline_memory = baseline["avg_memory_usage_mb"]

        if current_memory > baseline_memory * threshold_multiplier:
            raise AssertionError(
                f"Memory regression detected for {component_name}: "
                f"Current: {current_memory:.2f}MB, "
                f"Baseline: {baseline_memory:.2f}MB "
                f"(threshold: {threshold_multiplier}x)"
            )

        return True

    def benchmark_component_render(
        self, component_func: Callable[..., Any], iterations: int = 100
    ) -> dict[str, float]:
        """Benchmark component rendering performance (legacy compatibility)."""
        render_times = []
        memory_usages = []

        for _ in range(iterations):
            profile = self.profile_component_render(
                "benchmark_test", component_func
            )
            render_times.append(profile.execution_time_ms)
            memory_usages.append(profile.memory_usage_mb)

        return {
            "avg_render_time_ms": sum(render_times) / len(render_times),
            "max_render_time_ms": max(render_times),
            "min_render_time_ms": min(render_times),
            "std_dev_render_time_ms": self._calculate_std_dev(render_times),
            "avg_memory_usage_mb": sum(memory_usages) / len(memory_usages),
            "max_memory_usage_mb": max(memory_usages),
            "min_memory_usage_mb": min(memory_usages),
        }

    def measure_memory_usage(self, func: Callable[..., Any]) -> dict[str, Any]:
        """Measure memory usage during function execution."""
        process = psutil.Process(os.getpid())

        # Baseline memory
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Execute function
        start_time = time.time()
        result = func()
        execution_time = time.time() - start_time

        # Final memory
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before

        return {
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_delta_mb": memory_delta,
            "execution_time_ms": execution_time * 1000,
            "result": result,
        }

    def get_all_profiles(self) -> dict[str, list[PerformanceProfile]]:
        """Get all performance profiles."""
        return self._profiles.copy()

    def clear_profiles(self, component_name: str | None = None) -> None:
        """Clear performance profiles for a component or all components."""
        if component_name:
            if component_name in self._profiles:
                self._profiles[component_name].clear()
        else:
            self._profiles.clear()

    def _estimate_widget_count(self, render_result: Any) -> int:
        """Estimate number of widgets in render result."""
        # Simple heuristic - count common widget patterns
        # In real implementation, would analyze the mock calls to count widgets
        return 1

    def _calculate_complexity_score(
        self, render_time: float, memory_usage: float, widget_count: int
    ) -> float:
        """Calculate a complexity score for the component."""
        time_score = min(render_time / 100.0, 10.0)  # Normalize to 0-10
        memory_score = min(memory_usage / 10.0, 10.0)  # Normalize to 0-10
        widget_score = min(widget_count / 5.0, 10.0)  # Normalize to 0-10

        return (time_score + memory_score + widget_score) / 3.0

    def _calculate_std_dev(self, values: list[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5
