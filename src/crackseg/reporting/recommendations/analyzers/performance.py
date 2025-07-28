"""Performance analysis for recommendation engine.

This module provides analysis of final performance metrics to generate
actionable recommendations for improvement.
"""

import logging

from ...config import ExperimentData
from ..thresholds import PerformanceThresholds


class PerformanceAnalyzer:
    """Analyze final performance and generate recommendations."""

    def __init__(self) -> None:
        """Initialize the performance analyzer."""
        self.logger = logging.getLogger(__name__)
        self.thresholds = PerformanceThresholds()

    def analyze_final_performance(
        self, experiment_data: ExperimentData
    ) -> list[str]:
        """Analyze final performance and generate recommendations."""
        recommendations = []

        if "final_metrics" not in experiment_data.metrics:
            return recommendations

        metrics = experiment_data.metrics["final_metrics"]

        # Check IoU performance
        if "iou" in metrics:
            iou = metrics["iou"]
            if iou < self.thresholds.get_threshold("iou", "poor"):
                recommendations.append(
                    "📉 **Poor IoU Performance**: IoU below 60%. Consider: "
                    "1) Data augmentation improvements, 2) Model architecture "
                    "changes, 3) Loss function optimization"
                )
            elif iou < self.thresholds.get_threshold("iou", "good"):
                recommendations.append(
                    "📊 **Moderate IoU Performance**: IoU between 60-75%. "
                    "Consider fine-tuning hyperparameters or trying different "
                    "architectures."
                )

        # Check Dice performance
        if "dice" in metrics:
            dice = metrics["dice"]
            if dice < self.thresholds.get_threshold("dice", "poor"):
                recommendations.append(
                    "📉 **Poor Dice Performance**: Dice below 65%. Focus on: "
                    "1) Class imbalance handling, 2) Boundary-aware losses, "
                    "3) Multi-scale feature extraction"
                )

        # Check F1 performance
        if "f1" in metrics:
            f1 = metrics["f1"]
            if f1 < self.thresholds.get_threshold("f1", "poor"):
                recommendations.append(
                    "📉 **Poor F1 Performance**: F1 below 65%. Consider: "
                    "1) Threshold tuning, 2) Precision-recall balance, "
                    "3) Data quality improvements"
                )

        return recommendations
