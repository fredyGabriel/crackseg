"""Anomaly detection for performance monitoring.

This module provides anomaly detection functionality for identifying
performance outliers across multiple experiments.
"""

import logging
import statistics
from typing import Any

from ..config import ExperimentData, ReportConfig


class AnomalyDetector:
    """Handles anomaly detection across experiments."""

    def __init__(self) -> None:
        """Initialize the AnomalyDetector."""
        self.logger = logging.getLogger(__name__)

    def detect_anomalies(
        self,
        experiments_data: list[ExperimentData],
        config: ReportConfig,
    ) -> dict[str, Any]:
        """
        Detect performance anomalies across experiments.

        Args:
            experiments_data: List of experiment data
            config: Reporting configuration

        Returns:
            Dictionary with anomaly detection results
        """
        self.logger.info(
            f"Detecting anomalies across {len(experiments_data)} experiments"
        )

        anomalies = {
            "total_experiments": len(experiments_data),
            "anomalies_detected": 0,
            "anomaly_details": [],
            "statistical_summary": {},
            "outlier_experiments": [],
        }

        if len(experiments_data) < 2:
            self.logger.warning(
                "Need at least 2 experiments for anomaly detection"
            )
            return anomalies

        # Extract key metrics for analysis
        metrics_data = self._extract_metrics_data(experiments_data)

        if not metrics_data:
            return anomalies

        # Calculate statistical measures
        anomalies["statistical_summary"] = self._calculate_statistical_summary(
            metrics_data
        )

        # Detect outliers using z-score method
        anomalies["anomaly_details"] = self._detect_outliers(
            metrics_data, anomalies["statistical_summary"]
        )

        anomalies["outlier_experiments"] = [
            detail["experiment_id"] for detail in anomalies["anomaly_details"]
        ]
        anomalies["anomalies_detected"] = len(anomalies["anomaly_details"])

        self.logger.info(
            f"Anomaly detection completed: {anomalies['anomalies_detected']} "
            "anomalies found"
        )
        return anomalies

    def _extract_metrics_data(
        self, experiments_data: list[ExperimentData]
    ) -> list[dict[str, Any]]:
        """Extract key metrics from experiment data."""
        metrics_data = []
        for exp_data in experiments_data:
            if "complete_summary" in exp_data.metrics:
                summary = exp_data.metrics["complete_summary"]
                metrics_data.append(
                    {
                        "experiment_id": exp_data.experiment_id,
                        "iou": summary.get("best_iou", 0.0),
                        "f1": summary.get("best_f1", 0.0),
                        "precision": summary.get("best_precision", 0.0),
                        "recall": summary.get("best_recall", 0.0),
                        "loss": summary.get("final_loss", float("inf")),
                        "training_time": summary.get("training_time", 0.0),
                    }
                )
        return metrics_data

    def _calculate_statistical_summary(
        self, metrics_data: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """Calculate statistical summary for metrics."""
        iou_values = [m["iou"] for m in metrics_data]
        f1_values = [m["f1"] for m in metrics_data]
        loss_values = [m["loss"] for m in metrics_data]

        return {
            "iou": {
                "mean": statistics.mean(iou_values),
                "std": (
                    statistics.stdev(iou_values)
                    if len(iou_values) > 1
                    else 0.0
                ),
                "min": min(iou_values),
                "max": max(iou_values),
            },
            "f1": {
                "mean": statistics.mean(f1_values),
                "std": (
                    statistics.stdev(f1_values) if len(f1_values) > 1 else 0.0
                ),
                "min": min(f1_values),
                "max": max(f1_values),
            },
            "loss": {
                "mean": statistics.mean(loss_values),
                "std": (
                    statistics.stdev(loss_values)
                    if len(loss_values) > 1
                    else 0.0
                ),
                "min": min(loss_values),
                "max": max(loss_values),
            },
        }

    def _detect_outliers(
        self,
        metrics_data: list[dict[str, Any]],
        statistical_summary: dict[str, dict[str, float]],
    ) -> list[dict[str, Any]]:
        """Detect outliers using z-score method."""
        anomaly_details = []

        for metric_data in metrics_data:
            anomaly_flags = []

            # Check IoU outliers
            iou_values = [m["iou"] for m in metrics_data]
            if len(iou_values) > 1:
                iou_zscore = abs(
                    (metric_data["iou"] - statistical_summary["iou"]["mean"])
                    / statistical_summary["iou"]["std"]
                )
                if iou_zscore > 2.0:  # 2 standard deviations
                    anomaly_flags.append(
                        f"High IoU variance (z-score: {iou_zscore:.2f})"
                    )

            # Check F1 outliers
            f1_values = [m["f1"] for m in metrics_data]
            if len(f1_values) > 1:
                f1_zscore = abs(
                    (metric_data["f1"] - statistical_summary["f1"]["mean"])
                    / statistical_summary["f1"]["std"]
                )
                if f1_zscore > 2.0:
                    anomaly_flags.append(
                        f"High F1 variance (z-score: {f1_zscore:.2f})"
                    )

            # Check loss outliers
            loss_values = [m["loss"] for m in metrics_data]
            if len(loss_values) > 1:
                loss_zscore = abs(
                    (metric_data["loss"] - statistical_summary["loss"]["mean"])
                    / statistical_summary["loss"]["std"]
                )
                if loss_zscore > 2.0:
                    anomaly_flags.append(
                        f"High loss variance (z-score: {loss_zscore:.2f})"
                    )

            if anomaly_flags:
                anomaly_details.append(
                    {
                        "experiment_id": metric_data["experiment_id"],
                        "anomalies": anomaly_flags,
                        "metrics": metric_data,
                    }
                )

        return anomaly_details
