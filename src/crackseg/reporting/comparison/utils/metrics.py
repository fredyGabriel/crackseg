"""Metrics extraction and processing utilities for experiment comparison."""

import logging
from typing import Any

from ...config import ExperimentData


class MetricsExtractor:
    """Metrics extraction and processing utilities for experiment comparison."""

    def __init__(self) -> None:
        """Initialize the metrics extractor."""
        self.logger = logging.getLogger(__name__)

        # Define metrics to extract
        self.metrics = ["iou", "dice", "f1", "precision", "recall", "loss"]

    def extract_comparison_metrics(
        self, experiments_data: list[ExperimentData]
    ) -> dict[str, dict[str, float]]:
        """Extract comparison metrics from experiment data.

        Args:
            experiments_data: List of experiment data to process

        Returns:
            Dictionary mapping experiment IDs to metric dictionaries
        """
        metrics_data = {}

        for exp_data in experiments_data:
            exp_id = exp_data.experiment_id
            metrics = {}

            # Extract final metrics from complete summary
            if "complete_summary" in exp_data.metrics:
                summary = exp_data.metrics["complete_summary"]
                best_metrics = summary.get("best_metrics", {})

                for metric_name in self.metrics:
                    if metric_name in best_metrics:
                        metrics[metric_name] = best_metrics[metric_name].get(
                            "value", 0.0
                        )
                    else:
                        metrics[metric_name] = 0.0

            metrics_data[exp_id] = metrics

        return metrics_data

    def extract_model_config(
        self, exp_id: str, experiments_data: list[ExperimentData]
    ) -> dict[str, Any]:
        """Extract model configuration for a specific experiment.

        Args:
            exp_id: Experiment ID to extract config for
            experiments_data: List of experiment data

        Returns:
            Dictionary containing model configuration
        """
        for exp_data in experiments_data:
            if exp_data.experiment_id == exp_id:
                if "model_config" in exp_data.config:
                    return exp_data.config["model_config"]
                else:
                    return {}

        return {}

    def extract_training_config(
        self, exp_id: str, experiments_data: list[ExperimentData]
    ) -> dict[str, Any]:
        """Extract training configuration for a specific experiment.

        Args:
            exp_id: Experiment ID to extract config for
            experiments_data: List of experiment data

        Returns:
            Dictionary containing training configuration
        """
        for exp_data in experiments_data:
            if exp_data.experiment_id == exp_id:
                if "training_config" in exp_data.config:
                    return exp_data.config["training_config"]
                else:
                    return {}

        return {}

    def validate_metrics_data(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Validate extracted metrics data for completeness and consistency.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary containing validation results
        """
        validation = {
            "is_valid": True,
            "missing_metrics": {},
            "invalid_values": {},
            "warnings": [],
        }

        for exp_id, metrics in metrics_data.items():
            # Check for missing metrics
            missing = [
                metric for metric in self.metrics if metric not in metrics
            ]
            if missing:
                validation["missing_metrics"][exp_id] = missing
                validation["is_valid"] = False

            # Check for invalid values
            invalid = []
            for metric_name, value in metrics.items():
                if not isinstance(value, int | float):
                    invalid.append(f"{metric_name}: {type(value).__name__}")
                elif value < 0 or value > 1:
                    if metric_name != "loss":  # Loss can be > 1
                        invalid.append(f"{metric_name}: {value}")

            if invalid:
                validation["invalid_values"][exp_id] = invalid
                validation["is_valid"] = False

        # Generate warnings
        if len(metrics_data) < 2:
            validation["warnings"].append(
                "Only one experiment found - limited comparison possible"
            )

        return validation

    def normalize_metrics(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Normalize metrics to 0-1 range for comparison.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary with normalized metrics
        """
        normalized_data = {}

        for exp_id, metrics in metrics_data.items():
            normalized_metrics = {}

            for metric_name, value in metrics.items():
                if metric_name == "loss":
                    # For loss, lower is better, so we invert
                    normalized_metrics[metric_name] = max(0.0, 1.0 - value)
                else:
                    # For other metrics, higher is better, so we clamp to 0-1
                    normalized_metrics[metric_name] = max(0.0, min(1.0, value))

            normalized_data[exp_id] = normalized_metrics

        return normalized_data

    def get_metrics_summary(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Get a summary of all metrics across experiments.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary containing metrics summary
        """
        summary = {
            "total_experiments": len(metrics_data),
            "metrics_coverage": {},
            "value_ranges": {},
        }

        # Calculate metrics coverage
        for metric_name in self.metrics:
            coverage = sum(
                1
                for metrics in metrics_data.values()
                if metric_name in metrics and metrics[metric_name] > 0
            )
            summary["metrics_coverage"][metric_name] = {
                "count": coverage,
                "percentage": (
                    (coverage / len(metrics_data)) * 100 if metrics_data else 0
                ),
            }

        # Calculate value ranges
        for metric_name in self.metrics:
            values = [
                metrics.get(metric_name, 0.0)
                for metrics in metrics_data.values()
            ]

            if values:
                summary["value_ranges"][metric_name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "count": len([v for v in values if v > 0]),
                }

        return summary
