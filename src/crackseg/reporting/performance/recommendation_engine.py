"""Recommendation engine for performance optimization.

This module provides recommendation generation functionality based on
performance analysis results.
"""

import logging

from ..config import ExperimentData, ReportConfig


class RecommendationEngine:
    """Handles generation of actionable recommendations."""

    def __init__(self) -> None:
        """Initialize the RecommendationEngine."""
        self.logger = logging.getLogger(__name__)

    def generate_recommendations(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
    ) -> list[str]:
        """
        Generate actionable recommendations based on analysis.

        Args:
            experiment_data: Loaded experiment data
            config: Reporting configuration

        Returns:
            List of actionable recommendations
        """
        self.logger.info(
            f"Generating recommendations for experiment: "
            f"{experiment_data.experiment_id}"
        )

        recommendations = []

        if "complete_summary" not in experiment_data.metrics:
            recommendations.append(
                "No complete metrics available for analysis"
            )
            return recommendations

        complete_summary = experiment_data.metrics["complete_summary"]
        thresholds = config.performance_thresholds

        # Check IoU performance
        best_iou = complete_summary.get("best_iou", 0.0)
        iou_threshold = thresholds.get("iou_min", 0.7)
        if best_iou < iou_threshold:
            recommendations.append(
                f"IoU ({best_iou:.3f}) below threshold ({iou_threshold:.3f}). "
                "Consider: data augmentation, model architecture changes, "
                "or hyperparameter tuning."
            )

        # Check F1 performance
        best_f1 = complete_summary.get("best_f1", 0.0)
        f1_threshold = thresholds.get("f1_min", 0.75)
        if best_f1 < f1_threshold:
            recommendations.append(
                f"F1 score ({best_f1:.3f}) below threshold "
                f"({f1_threshold:.3f}). "
                "Consider: class balancing, loss function adjustments, "
                "or training strategy improvements."
            )

        # Check precision performance
        best_precision = complete_summary.get("best_precision", 0.0)
        precision_threshold = thresholds.get("precision_min", 0.8)
        if best_precision < precision_threshold:
            recommendations.append(
                f"Precision ({best_precision:.3f}) below threshold "
                f"({precision_threshold:.3f}). "
                "Consider: threshold tuning, data quality improvements, "
                "or model confidence calibration."
            )

        # Check recall performance
        best_recall = complete_summary.get("best_recall", 0.0)
        recall_threshold = thresholds.get("recall_min", 0.7)
        if best_recall < recall_threshold:
            recommendations.append(
                f"Recall ({best_recall:.3f}) below threshold "
                f"({recall_threshold:.3f}). "
                "Consider: data augmentation, model capacity increase, "
                "or training duration extension."
            )

        # Training time analysis
        training_time = complete_summary.get("training_time", 0.0)
        if training_time > 7200:  # 2 hours
            recommendations.append(
                f"Training time ({training_time / 3600:.1f}h) is high. "
                "Consider: batch size optimization, model simplification, "
                "or hardware acceleration."
            )

        # Loss analysis
        final_loss = complete_summary.get("final_loss", float("inf"))
        if final_loss > 1.0:
            recommendations.append(
                f"Final loss ({final_loss:.3f}) is high. "
                "Consider: learning rate adjustment, loss function selection, "
                "or training stability improvements."
            )

        # If all metrics are good, provide positive feedback
        if (
            best_iou >= iou_threshold
            and best_f1 >= f1_threshold
            and best_precision >= precision_threshold
            and best_recall >= recall_threshold
        ):
            recommendations.append(
                "All key metrics meet or exceed thresholds. "
                "Consider: model deployment, further optimization, "
                "or additional validation testing."
            )

        self.logger.info(
            f"Generated {len(recommendations)} recommendations for "
            f"{experiment_data.experiment_id}"
        )
        return recommendations
