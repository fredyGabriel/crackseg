"""Metric evaluation and threshold compliance checking.

This module provides metric evaluation functionality including threshold
compliance checking and performance score calculation.
"""

import statistics
from typing import Any

from ..config import ReportConfig


class MetricEvaluator:
    """Handles metric evaluation and threshold compliance checking."""

    def evaluate_metrics(
        self, complete_summary: dict[str, Any], config: ReportConfig
    ) -> dict[str, Any]:
        """Evaluate metrics against thresholds and best practices."""
        evaluation = {
            "iou_score": 0.0,
            "f1_score": 0.0,
            "precision_score": 0.0,
            "recall_score": 0.0,
            "overall_score": 0.0,
            "metric_quality": {},
        }

        thresholds = config.performance_thresholds

        # Evaluate IoU
        best_iou = complete_summary.get("best_iou", 0.0)
        iou_threshold = thresholds.get("iou_min", 0.7)
        evaluation["iou_score"] = min(best_iou / iou_threshold, 1.0)
        evaluation["metric_quality"]["iou"] = (
            "excellent"
            if best_iou >= 0.8
            else "good" if best_iou >= 0.7 else "poor"
        )

        # Evaluate F1
        best_f1 = complete_summary.get("best_f1", 0.0)
        f1_threshold = thresholds.get("f1_min", 0.75)
        evaluation["f1_score"] = min(best_f1 / f1_threshold, 1.0)
        evaluation["metric_quality"]["f1"] = (
            "excellent"
            if best_f1 >= 0.85
            else "good" if best_f1 >= 0.75 else "poor"
        )

        # Evaluate Precision
        best_precision = complete_summary.get("best_precision", 0.0)
        precision_threshold = thresholds.get("precision_min", 0.8)
        evaluation["precision_score"] = min(
            best_precision / precision_threshold, 1.0
        )
        evaluation["metric_quality"]["precision"] = (
            "excellent"
            if best_precision >= 0.9
            else "good" if best_precision >= 0.8 else "poor"
        )

        # Evaluate Recall
        best_recall = complete_summary.get("best_recall", 0.0)
        recall_threshold = thresholds.get("recall_min", 0.7)
        evaluation["recall_score"] = min(best_recall / recall_threshold, 1.0)
        evaluation["metric_quality"]["recall"] = (
            "excellent"
            if best_recall >= 0.8
            else "good" if best_recall >= 0.7 else "poor"
        )

        # Calculate overall score
        scores = [
            evaluation["iou_score"],
            evaluation["f1_score"],
            evaluation["precision_score"],
            evaluation["recall_score"],
        ]
        evaluation["overall_score"] = statistics.mean(scores)

        return evaluation

    def calculate_performance_score(
        self, complete_summary: dict[str, Any], config: ReportConfig
    ) -> float:
        """Calculate overall performance score."""
        thresholds = config.performance_thresholds

        best_iou = complete_summary.get("best_iou", 0.0)
        best_f1 = complete_summary.get("best_f1", 0.0)
        best_precision = complete_summary.get("best_precision", 0.0)
        best_recall = complete_summary.get("best_recall", 0.0)

        # Normalize scores to 0-1 range
        iou_score = min(best_iou / thresholds.get("iou_min", 0.7), 1.0)
        f1_score = min(best_f1 / thresholds.get("f1_min", 0.75), 1.0)
        precision_score = min(
            best_precision / thresholds.get("precision_min", 0.8), 1.0
        )
        recall_score = min(
            best_recall / thresholds.get("recall_min", 0.7), 1.0
        )

        # Weighted average (IoU and F1 are more important for segmentation)
        weighted_score = (
            iou_score * 0.35
            + f1_score * 0.35
            + precision_score * 0.15
            + recall_score * 0.15
        )

        return weighted_score

    def check_threshold_compliance(
        self, complete_summary: dict[str, Any], config: ReportConfig
    ) -> dict[str, bool]:
        """Check if metrics meet minimum thresholds."""
        thresholds = config.performance_thresholds

        return {
            "iou_compliant": complete_summary.get("best_iou", 0.0)
            >= thresholds.get("iou_min", 0.7),
            "f1_compliant": complete_summary.get("best_f1", 0.0)
            >= thresholds.get("f1_min", 0.75),
            "precision_compliant": complete_summary.get("best_precision", 0.0)
            >= thresholds.get("precision_min", 0.8),
            "recall_compliant": complete_summary.get("best_recall", 0.0)
            >= thresholds.get("recall_min", 0.7),
        }
