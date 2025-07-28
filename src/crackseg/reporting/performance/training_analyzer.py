"""Training pattern analysis and convergence detection.

This module provides training pattern analysis functionality including
convergence detection, stability assessment, and learning pattern analysis.
"""

import statistics
from typing import Any

import numpy as np


class TrainingAnalyzer:
    """Handles training pattern analysis and convergence detection."""

    def analyze_training_patterns(
        self, epoch_metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze training patterns and convergence."""
        if not epoch_metrics:
            return {"error": "No epoch metrics available"}

        analysis = {
            "total_epochs": len(epoch_metrics),
            "convergence_analysis": {},
            "training_stability": {},
            "learning_patterns": {},
        }

        # Extract metrics
        losses = [m.get("loss", float("inf")) for m in epoch_metrics]
        ious = [m.get("iou", 0.0) for m in epoch_metrics]

        # Convergence analysis
        analysis["convergence_analysis"] = self._analyze_convergence(losses)

        # Training stability
        analysis["training_stability"] = self._analyze_stability(ious)

        # Learning patterns
        analysis["learning_patterns"] = self._analyze_learning_patterns(losses)

        return analysis

    def _analyze_convergence(self, losses: list[float]) -> dict[str, Any]:
        """Analyze convergence patterns."""
        convergence = {}

        if len(losses) > 1:
            loss_trend = np.polyfit(range(len(losses)), losses, 1)[0]
            convergence["loss_trend"] = (
                "decreasing"
                if loss_trend < -0.01
                else "stable" if abs(loss_trend) < 0.01 else "increasing"
            )

            # Check for convergence
            final_losses = losses[-min(5, len(losses)) :]
            loss_variance = (
                statistics.variance(final_losses)
                if len(final_losses) > 1
                else 0.0
            )
            convergence["converged"] = loss_variance < 0.01

        return convergence

    def _analyze_stability(self, ious: list[float]) -> dict[str, Any]:
        """Analyze training stability."""
        stability = {}

        if len(ious) > 1:
            iou_std = statistics.stdev(ious)
            stability["iou_stability"] = (
                "stable" if iou_std < 0.05 else "unstable"
            )

            # Check for overfitting
            if len(ious) > 10:
                early_ious = ious[: len(ious) // 2]
                late_ious = ious[len(ious) // 2 :]
                early_avg = statistics.mean(early_ious)
                late_avg = statistics.mean(late_ious)
                stability["overfitting_risk"] = (
                    "high" if late_avg < early_avg - 0.05 else "low"
                )

        return stability

    def _analyze_learning_patterns(
        self, losses: list[float]
    ) -> dict[str, Any]:
        """Analyze learning patterns and efficiency."""
        patterns = {}

        if len(losses) > 5:
            # Detect learning rate issues
            loss_changes = [
                losses[i] - losses[i - 1] for i in range(1, len(losses))
            ]
            negative_changes = sum(1 for change in loss_changes if change < 0)
            improvement_rate = negative_changes / len(loss_changes)

            patterns["improvement_rate"] = improvement_rate
            patterns["learning_efficiency"] = (
                "good" if improvement_rate > 0.6 else "poor"
            )

        return patterns
