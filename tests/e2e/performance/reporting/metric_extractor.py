"""Metric extraction utilities for performance data.

This module provides utilities for extracting and normalizing performance
metrics from various data structures.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MetricExtractor:
    """Handles extraction of performance metrics from crackseg.data
    structures."""

    def __init__(self) -> None:
        """Initialize metric extractor."""
        self.logger = logging.getLogger(__name__)
        self._metric_paths = {
            "average_success_rate": [
                "overall_summary",
                "average_success_rate",
            ],
            "average_throughput": ["overall_summary", "average_throughput"],
            "total_violations": ["overall_summary", "total_violations"],
            "peak_memory_mb": ["resource_summary", "peak_memory_mb"],
            "avg_cpu_usage": ["resource_summary", "avg_cpu_usage"],
        }

    def extract_metric_value(
        self, data: dict[str, Any], metric_name: str
    ) -> float | None:
        """Extract metric value from crackseg.data structure."""
        path = self._metric_paths.get(metric_name)
        if not path:
            self.logger.warning(f"Unknown metric: {metric_name}")
            return None

        try:
            current: Any = data
            for key in path:
                current = current[key]

            # Ensure current is a number before converting
            if current is None:
                return None
            if isinstance(current, int | float):
                return float(current)
            if isinstance(current, str):
                return float(current)

            self.logger.warning(
                f"Invalid metric type for {metric_name}: {type(current)}"
            )
            return None
        except (KeyError, TypeError, ValueError) as e:
            self.logger.debug(f"Failed to extract {metric_name}: {e}")
            return None

    def extract_all_metrics(self, data: dict[str, Any]) -> dict[str, float]:
        """Extract all known metrics from crackseg.data structure."""
        metrics = {}
        for metric_name in self._metric_paths:
            value = self.extract_metric_value(data, metric_name)
            if value is not None:
                metrics[metric_name] = value
        return metrics

    def add_metric_path(self, metric_name: str, path: list[str]) -> None:
        """Add a new metric extraction path."""
        self._metric_paths[metric_name] = path

    def get_available_metrics(self) -> list[str]:
        """Get list of all available metric names."""
        return list(self._metric_paths.keys())

    def validate_data_structure(self, data: dict[str, Any]) -> dict[str, bool]:
        """Validate that data structure contains expected metric paths."""
        validation_results = {}
        for metric_name, path in self._metric_paths.items():
            try:
                current: Any = data
                for key in path:
                    current = current[key]
                validation_results[metric_name] = current is not None
            except (KeyError, TypeError):
                validation_results[metric_name] = False
        return validation_results
