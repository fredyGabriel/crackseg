"""Performance trend analysis for experiment comparison."""

import logging
from typing import Any

import numpy as np

from ...config import ExperimentData


class TrendAnalyzer:
    """Performance trend analysis utilities for experiment comparison."""

    def __init__(self) -> None:
        """Initialize the trend analyzer."""
        self.logger = logging.getLogger(__name__)

        # Define metrics to analyze
        self.metrics = ["iou", "dice", "f1", "precision", "recall"]

    def analyze_performance_trends(
        self, experiments_data: list[ExperimentData]
    ) -> dict[str, Any]:
        """Analyze performance trends across experiments.

        Args:
            experiments_data: List of experiment data to analyze

        Returns:
            Dictionary containing trend analysis results
        """
        trends = {
            "metric_trends": {},
            "improvement_patterns": {},
            "consistency_analysis": {},
        }

        # Analyze trends for each metric
        for metric_name in self.metrics:
            values = []
            for exp_data in experiments_data:
                if "complete_summary" in exp_data.metrics:
                    summary = exp_data.metrics["complete_summary"]
                    best_metrics = summary.get("best_metrics", {})
                    if metric_name in best_metrics:
                        values.append(
                            best_metrics[metric_name].get("value", 0.0)
                        )
                    else:
                        values.append(0.0)

            if len(values) > 1:
                trends["metric_trends"][metric_name] = {
                    "values": values,
                    "trend_direction": (
                        "increasing"
                        if values[-1] > values[0]
                        else "decreasing"
                    ),
                    "improvement_rate": (
                        (values[-1] - values[0]) / len(values)
                        if len(values) > 1
                        else 0.0
                    ),
                    "consistency": np.std(values),
                    "total_change": values[-1] - values[0],
                    "average_change_per_experiment": (
                        (values[-1] - values[0]) / (len(values) - 1)
                        if len(values) > 1
                        else 0.0
                    ),
                }

        # Analyze improvement patterns
        trends["improvement_patterns"] = self._analyze_improvement_patterns(
            experiments_data
        )

        # Analyze consistency
        trends["consistency_analysis"] = self._analyze_consistency(
            experiments_data
        )

        return trends

    def _analyze_improvement_patterns(
        self, experiments_data: list[ExperimentData]
    ) -> dict[str, Any]:
        """Analyze improvement patterns across experiments.

        Args:
            experiments_data: List of experiment data to analyze

        Returns:
            Dictionary containing improvement pattern analysis
        """
        patterns = {
            "consistent_improvement": [],
            "fluctuating_performance": [],
            "plateau_detected": [],
        }

        for metric_name in self.metrics:
            values = []
            for exp_data in experiments_data:
                if "complete_summary" in exp_data.metrics:
                    summary = exp_data.metrics["complete_summary"]
                    best_metrics = summary.get("best_metrics", {})
                    if metric_name in best_metrics:
                        values.append(
                            best_metrics[metric_name].get("value", 0.0)
                        )
                    else:
                        values.append(0.0)

            if len(values) > 2:
                # Check for consistent improvement
                if all(
                    values[i] <= values[i + 1] for i in range(len(values) - 1)
                ):
                    patterns["consistent_improvement"].append(metric_name)

                # Check for fluctuations
                changes = [
                    values[i + 1] - values[i] for i in range(len(values) - 1)
                ]
                if any(c < 0 for c in changes) and any(c > 0 for c in changes):
                    patterns["fluctuating_performance"].append(metric_name)

                # Check for plateau (no significant change in last 3 experiments)
                if len(values) >= 3:
                    recent_values = values[-3:]
                    recent_std = np.std(recent_values)
                    if recent_std < 0.01:  # Small variation threshold
                        patterns["plateau_detected"].append(metric_name)

        return patterns

    def _analyze_consistency(
        self, experiments_data: list[ExperimentData]
    ) -> dict[str, Any]:
        """Analyze consistency of performance across experiments.

        Args:
            experiments_data: List of experiment data to analyze

        Returns:
            Dictionary containing consistency analysis
        """
        consistency = {
            "high_consistency": [],
            "medium_consistency": [],
            "low_consistency": [],
        }

        for metric_name in self.metrics:
            values = []
            for exp_data in experiments_data:
                if "complete_summary" in exp_data.metrics:
                    summary = exp_data.metrics["complete_summary"]
                    best_metrics = summary.get("best_metrics", {})
                    if metric_name in best_metrics:
                        values.append(
                            best_metrics[metric_name].get("value", 0.0)
                        )
                    else:
                        values.append(0.0)

            if len(values) > 1:
                std_dev = np.std(values)
                mean_val = np.mean(values)
                coefficient_of_variation = (
                    std_dev / mean_val if mean_val > 0 else 0
                )

                if coefficient_of_variation < 0.05:
                    consistency["high_consistency"].append(metric_name)
                elif coefficient_of_variation < 0.15:
                    consistency["medium_consistency"].append(metric_name)
                else:
                    consistency["low_consistency"].append(metric_name)

        return consistency

    def get_trend_summary(
        self, experiments_data: list[ExperimentData]
    ) -> dict[str, Any]:
        """Get a summary of trends across all experiments.

        Args:
            experiments_data: List of experiment data to analyze

        Returns:
            Dictionary containing trend summary
        """
        trends = self.analyze_performance_trends(experiments_data)

        summary = {
            "total_experiments": len(experiments_data),
            "trending_metrics": [],
            "stable_metrics": [],
            "improving_metrics": [],
            "declining_metrics": [],
        }

        for metric_name, trend_data in trends["metric_trends"].items():
            if trend_data["trend_direction"] == "increasing":
                summary["improving_metrics"].append(metric_name)
            else:
                summary["declining_metrics"].append(metric_name)

            if trend_data["consistency"] < 0.01:
                summary["stable_metrics"].append(metric_name)
            else:
                summary["trending_metrics"].append(metric_name)

        return summary
