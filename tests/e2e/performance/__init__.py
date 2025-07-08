"""Performance benchmarking suite for E2E testing pipeline.

This module provides comprehensive performance benchmarking capabilities
for the crack segmentation E2E testing pipeline, integrating with the
ResourceMonitor system for real-time metrics collection.
"""

from .benchmark_runner import BenchmarkRunner
from .benchmark_suite import BenchmarkSuite
from .metrics_collector import MetricsCollector

__all__ = [
    "BenchmarkSuite",
    "BenchmarkRunner",
    "MetricsCollector",
]
