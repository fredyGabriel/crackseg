"""Benchmark runner with ResourceMonitor integration.

This module provides the BenchmarkRunner class that orchestrates
benchmark execution with real-time resource monitoring integration,
collecting comprehensive performance metrics during test execution.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from crackseg.utils.monitoring import ResourceMonitor, ResourceSnapshot
from tests.e2e.config.performance_thresholds import PerformanceThresholds

from .benchmark_suite import BenchmarkConfig, BenchmarkResult, BenchmarkSuite

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Executes benchmarks with integrated resource monitoring."""

    def __init__(
        self,
        thresholds: PerformanceThresholds,
        enable_resource_monitoring: bool = True,
        monitoring_interval: float = 1.0,
    ) -> None:
        """Initialize benchmark runner."""
        self.thresholds = thresholds
        self.enable_resource_monitoring = enable_resource_monitoring
        self.monitoring_interval = monitoring_interval

        self.benchmark_suite = BenchmarkSuite(thresholds)
        self.resource_monitor: ResourceMonitor | None = None
        self.resource_snapshots: list[ResourceSnapshot] = []

        self.logger = logging.getLogger(__name__)

    async def run_with_monitoring(
        self,
        benchmark_name: str,
        config: BenchmarkConfig,
    ) -> dict[str, Any]:
        """Run benchmark with resource monitoring integration."""
        self.logger.info(
            f"Starting benchmark '{benchmark_name}' with resource monitoring"
        )

        # Initialize resource monitoring if enabled
        if self.enable_resource_monitoring:
            self.resource_monitor = ResourceMonitor(
                enable_gpu_monitoring=True,
                enable_network_monitoring=True,
                enable_file_monitoring=True,
            )

            # Add callback to collect snapshots
            self.resource_monitor.add_callback(self._collect_resource_snapshot)

            # Start monitoring
            self.resource_monitor.start_real_time_monitoring(
                interval=self.monitoring_interval
            )

        try:
            # Execute benchmark
            benchmark_result = await self.benchmark_suite.run_benchmark(
                benchmark_name, config
            )

            # Collect final resource snapshot
            final_snapshot = None
            if self.resource_monitor:
                final_snapshot = self.resource_monitor.get_current_snapshot()

            # Combine results
            return self._combine_results(benchmark_result, final_snapshot)

        finally:
            # Stop monitoring
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()

    async def run_benchmark_suite(
        self,
        configs: dict[str, BenchmarkConfig],
    ) -> dict[str, dict[str, Any]]:
        """Run multiple benchmarks with monitoring."""
        results = {}

        for benchmark_name, config in configs.items():
            self.logger.info(f"Running benchmark suite: {benchmark_name}")

            try:
                result = await self.run_with_monitoring(benchmark_name, config)
                results[benchmark_name] = result

                # Brief pause between benchmarks
                await asyncio.sleep(2.0)

            except Exception as e:
                self.logger.error(f"Benchmark {benchmark_name} failed: {e}")
                results[benchmark_name] = {
                    "error": str(e),
                    "benchmark_result": None,
                    "resource_metrics": None,
                }

        return results

    def _collect_resource_snapshot(
        self, resource_data: dict[str, Any]
    ) -> None:
        """Callback to collect resource snapshots during monitoring."""
        # Extract data from ResourceMonitor's snapshot dictionary format
        snapshot = ResourceSnapshot(
            timestamp=resource_data.get("timestamp", time.time()),
            cpu_percent=resource_data.get("resource_monitor/cpu_percent", 0.0),
            memory_used_mb=resource_data.get(
                "resource_monitor/memory_used_mb", 0.0
            ),
            memory_available_mb=resource_data.get(
                "resource_monitor/memory_available_mb", 0.0
            ),
            memory_percent=resource_data.get(
                "resource_monitor/memory_percent", 0.0
            ),
            gpu_memory_used_mb=resource_data.get(
                "resource_monitor/gpu_memory_used_mb", 0.0
            ),
            gpu_memory_total_mb=resource_data.get(
                "resource_monitor/gpu_memory_total_mb", 8192.0
            ),
            gpu_memory_percent=resource_data.get(
                "resource_monitor/gpu_memory_percent", 0.0
            ),
            gpu_utilization_percent=resource_data.get(
                "resource_monitor/gpu_utilization_percent", 0.0
            ),
            gpu_temperature_celsius=resource_data.get(
                "resource_monitor/gpu_temperature_celsius", 0.0
            ),
            process_count=resource_data.get(
                "resource_monitor/process_count", 0
            ),
            thread_count=resource_data.get("resource_monitor/thread_count", 0),
            file_handles=resource_data.get("resource_monitor/file_handles", 0),
            network_connections=resource_data.get(
                "resource_monitor/network_connections", 0
            ),
            open_ports=resource_data.get("resource_monitor/open_ports", []),
            disk_read_mb=resource_data.get(
                "resource_monitor/disk_read_mb", 0.0
            ),
            disk_write_mb=resource_data.get(
                "resource_monitor/disk_write_mb", 0.0
            ),
            temp_files_count=resource_data.get(
                "resource_monitor/temp_files_count", 0
            ),
            temp_files_size_mb=resource_data.get(
                "resource_monitor/temp_files_size_mb", 0.0
            ),
        )

        self.resource_snapshots.append(snapshot)

    def _combine_results(
        self,
        benchmark_result: BenchmarkResult,
        final_snapshot: ResourceSnapshot | None,
    ) -> dict[str, Any]:
        """Combine benchmark results with resource monitoring data."""
        combined = {
            "benchmark_result": {
                "name": benchmark_result.benchmark_name,
                "success_rate": benchmark_result.success_rate,
                "throughput": benchmark_result.throughput,
                "duration": benchmark_result.duration,
                "threshold_violations": benchmark_result.threshold_violations,
                "metrics": benchmark_result.metrics,
            },
            "resource_metrics": None,
        }

        if final_snapshot and self.resource_snapshots:
            combined["resource_metrics"] = self._analyze_resource_usage()

        return combined

    def _analyze_resource_usage(self) -> dict[str, Any]:
        """Analyze collected resource snapshots."""
        if not self.resource_snapshots:
            return {}

        # Calculate statistics
        cpu_values = [s.cpu_percent for s in self.resource_snapshots]
        memory_values = [s.memory_used_mb for s in self.resource_snapshots]
        gpu_memory_values = [
            s.gpu_memory_used_mb for s in self.resource_snapshots
        ]

        return {
            "cpu_usage": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values),
            },
            "memory_usage_mb": {
                "avg": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values),
            },
            "gpu_memory_mb": {
                "avg": sum(gpu_memory_values) / len(gpu_memory_values),
                "max": max(gpu_memory_values),
                "min": min(gpu_memory_values),
            },
            "sample_count": len(self.resource_snapshots),
            "monitoring_duration": (
                self.resource_snapshots[-1].timestamp
                - self.resource_snapshots[0].timestamp
                if len(self.resource_snapshots) > 1
                else 0.0
            ),
        }

    def export_results(
        self,
        results: dict[str, dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Export benchmark results to file."""
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert results to JSON-serializable format
        serializable_results = {}
        for benchmark_name, result in results.items():
            serializable_results[benchmark_name] = {
                "benchmark_result": result.get("benchmark_result"),
                "resource_metrics": result.get("resource_metrics"),
                "error": result.get("error"),
            }

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Benchmark results exported to {output_path}")

    def validate_system_health(self) -> list[str]:
        """Validate system health before running benchmarks."""
        violations = []

        if self.resource_monitor:
            try:
                snapshot = self.resource_monitor.get_current_snapshot()

                # Check current resource usage
                if (
                    snapshot.cpu_percent
                    > self.thresholds.system_resources.cpu_warning_percent
                ):
                    violations.append(
                        f"High CPU usage before benchmark: "
                        f"{snapshot.cpu_percent:.1f}%"
                    )

                if (
                    snapshot.memory_used_mb
                    > self.thresholds.system_resources.memory_warning_mb
                ):
                    violations.append(
                        f"High memory usage before benchmark: "
                        f"{snapshot.memory_used_mb:.1f}MB"
                    )

                if (
                    snapshot.gpu_memory_used_mb
                    > self.thresholds.model_processing.memory_warning_mb
                ):
                    violations.append(
                        f"High GPU memory usage before benchmark: "
                        f"{snapshot.gpu_memory_used_mb:.1f}MB"
                    )

            except Exception as e:
                violations.append(f"Failed to check system health: {e}")

        return violations
