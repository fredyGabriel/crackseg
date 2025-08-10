"""Anomaly detection for experiment comparison."""

import logging
from typing import Any

import numpy as np


class AnomalyDetector:
    """Anomaly detection utilities for experiment comparison."""

    def __init__(self) -> None:
        """Initialize the anomaly detector."""
        self.logger = logging.getLogger(__name__)

        # Define metrics to analyze
        self.metrics = ["iou", "dice", "f1", "precision", "recall"]

    def detect_anomalies(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Detect anomalies in experiment results.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary containing anomaly detection results
        """
        anomalies = {
            "outliers": {},
            "performance_gaps": {},
            "unusual_patterns": [],
            "suspicious_experiments": [],
        }

        # Detect outliers for each metric
        for metric_name in self.metrics:
            values = [
                metrics.get(metric_name, 0.0)
                for metrics in metrics_data.values()
            ]

            if len(values) > 2:
                outliers = self._detect_outliers(
                    list(metrics_data.keys()), values, metric_name
                )
                if outliers:
                    anomalies["outliers"][metric_name] = outliers

        # Detect performance gaps
        anomalies["performance_gaps"] = self._detect_performance_gaps(
            metrics_data
        )

        # Detect unusual patterns
        anomalies["unusual_patterns"] = self._detect_unusual_patterns(
            metrics_data
        )

        # Identify suspicious experiments
        anomalies["suspicious_experiments"] = (
            self._identify_suspicious_experiments(metrics_data)
        )

        return anomalies

    def _detect_outliers(
        self, experiment_ids: list[str], values: list[float], metric_name: str
    ) -> list[tuple[str, float]]:
        """Detect outliers using IQR method.

        Args:
            experiment_ids: List of experiment IDs
            values: List of metric values
            metric_name: Name of the metric being analyzed

        Returns:
            List of (experiment_id, value) tuples for outliers
        """
        outliers = []

        # Calculate quartiles and IQR
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        # Define bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Find outliers
        for exp_id, value in zip(experiment_ids, values, strict=False):
            if value < lower_bound or value > upper_bound:
                outliers.append((exp_id, value))

        return outliers

    def _detect_performance_gaps(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Detect significant performance gaps between experiments.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary containing performance gap analysis
        """
        gaps = {}

        for metric_name in self.metrics:
            values = [
                metrics.get(metric_name, 0.0)
                for metrics in metrics_data.values()
            ]

            if len(values) > 1:
                sorted_values = sorted(values, reverse=True)
                max_gap = 0.0
                gap_location = None

                for i in range(len(sorted_values) - 1):
                    gap = sorted_values[i] - sorted_values[i + 1]
                    if gap > max_gap:
                        max_gap = gap
                        gap_location = i

                if max_gap > 0.1:  # Significant gap threshold
                    gaps[metric_name] = {
                        "max_gap": max_gap,
                        "gap_location": gap_location,
                        "gap_percentage": (max_gap / sorted_values[0]) * 100,
                    }

        return gaps

    def _detect_unusual_patterns(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> list[dict[str, Any]]:
        """Detect unusual patterns in experiment results.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            List of unusual patterns detected
        """
        patterns = []

        # Check for experiments with very high variance across metrics
        for exp_id, metrics in metrics_data.items():
            values = [metrics.get(metric, 0.0) for metric in self.metrics]
            variance = np.var(values)

            if variance > 0.1:  # High variance threshold
                patterns.append(
                    {
                        "type": "high_variance",
                        "experiment_id": exp_id,
                        "variance": variance,
                        "description": f"Experiment {exp_id} shows high variance across metrics",
                    }
                )

        # Check for experiments with all metrics below average
        for metric_name in self.metrics:
            values = [
                metrics.get(metric_name, 0.0)
                for metrics in metrics_data.values()
            ]
            mean_val = np.mean(values)

            for exp_id, metrics in metrics_data.items():
                if (
                    metrics.get(metric_name, 0.0) < mean_val * 0.5
                ):  # 50% below mean
                    patterns.append(
                        {
                            "type": "below_average",
                            "experiment_id": exp_id,
                            "metric": metric_name,
                            "value": metrics.get(metric_name, 0.0),
                            "mean": mean_val,
                            "description": f"Experiment {exp_id} significantly below average in {metric_name}",
                        }
                    )

        return patterns

    def _identify_suspicious_experiments(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> list[dict[str, Any]]:
        """Identify experiments that might be suspicious or problematic.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            List of suspicious experiments with reasons
        """
        suspicious = []

        for exp_id, metrics in metrics_data.items():
            reasons = []

            # Check for zero or near-zero values
            zero_metrics = [
                metric
                for metric in self.metrics
                if metrics.get(metric, 0.0) < 0.001
            ]
            if zero_metrics:
                reasons.append(
                    f"Zero/near-zero values in: {', '.join(zero_metrics)}"
                )

            # Check for extremely high values (potential errors)
            high_metrics = [
                metric
                for metric in self.metrics
                if metrics.get(metric, 0.0) > 0.99
            ]
            if high_metrics:
                reasons.append(
                    f"Suspiciously high values in: {', '.join(high_metrics)}"
                )

            # Check for missing metrics
            missing_metrics = [
                metric for metric in self.metrics if metric not in metrics
            ]
            if missing_metrics:
                reasons.append(
                    f"Missing metrics: {', '.join(missing_metrics)}"
                )

            if reasons:
                suspicious.append(
                    {
                        "experiment_id": exp_id,
                        "reasons": reasons,
                        "severity": "high" if len(reasons) > 2 else "medium",
                    }
                )

        return suspicious

    def get_anomaly_summary(
        self, metrics_data: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Get a summary of anomalies detected.

        Args:
            metrics_data: Dictionary mapping experiment IDs to metric dictionaries

        Returns:
            Dictionary containing anomaly summary
        """
        anomalies = self.detect_anomalies(metrics_data)

        summary = {
            "total_experiments": len(metrics_data),
            "total_outliers": sum(
                len(outliers) for outliers in anomalies["outliers"].values()
            ),
            "metrics_with_outliers": list(anomalies["outliers"].keys()),
            "suspicious_experiments": len(anomalies["suspicious_experiments"]),
            "unusual_patterns": len(anomalies["unusual_patterns"]),
            "performance_gaps": len(anomalies["performance_gaps"]),
        }

        return summary
