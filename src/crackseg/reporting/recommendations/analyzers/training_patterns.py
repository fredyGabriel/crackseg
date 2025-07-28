"""Training pattern analysis for recommendation engine.

This module provides analysis of training curves, convergence patterns,
and fitting behavior to generate actionable recommendations.
"""

import logging

import pandas as pd

from ...config import ExperimentData
from ..thresholds import PerformanceThresholds


class TrainingPatternAnalyzer:
    """Analyze training patterns and generate recommendations."""

    def __init__(self) -> None:
        """Initialize the training pattern analyzer."""
        self.logger = logging.getLogger(__name__)
        self.thresholds = PerformanceThresholds()

    def analyze_training_curves(
        self, experiment_data: ExperimentData
    ) -> list[str]:
        """Analyze training curves and generate recommendations."""
        recommendations = []

        if "training_metrics" not in experiment_data.metrics:
            return recommendations

        # Convert to DataFrame for analysis
        df = pd.DataFrame(experiment_data.metrics["training_metrics"])

        # Check for training instability
        if "loss" in df.columns:
            loss_std = df["loss"].std()
            if loss_std > 0.1:
                recommendations.append(
                    "🔧 **Training Instability Detected**: High loss variance "
                    "suggests learning rate may be too high. Consider "
                    "reducing learning rate by 50%."
                )

        # Check for slow convergence
        if "loss" in df.columns and len(df) > 20:
            early_loss = df["loss"].iloc[:10].mean()
            late_loss = df["loss"].iloc[-10:].mean()
            if late_loss > early_loss * 0.8:
                recommendations.append(
                    "🐌 **Slow Convergence**: Loss not decreasing "
                    "significantly. Consider increasing learning rate or "
                    "changing optimizer."
                )

        return recommendations

    def analyze_convergence_patterns(
        self, experiment_data: ExperimentData
    ) -> list[str]:
        """Analyze convergence patterns and generate recommendations."""
        recommendations = []

        if "training_metrics" not in experiment_data.metrics:
            return recommendations

        df = pd.DataFrame(experiment_data.metrics["training_metrics"])

        # Check for early convergence
        if "loss" in df.columns and len(df) > 10:
            early_epochs = df["loss"].iloc[:5]
            later_epochs = df["loss"].iloc[-5:]

            early_improvement = early_epochs.iloc[0] - early_epochs.iloc[-1]
            later_improvement = later_epochs.iloc[0] - later_epochs.iloc[-1]

            if later_improvement < early_improvement * 0.1:
                recommendations.append(
                    "⚡ **Early Convergence**: Model converged quickly. "
                    "Consider: 1) Higher learning rate, 2) More aggressive "
                    "augmentation, 3) Deeper architecture"
                )

        # Check for late convergence
        if "loss" in df.columns and len(df) > 30:
            mid_loss = df["loss"].iloc[10:20].mean()
            final_loss = df["loss"].iloc[-10:].mean()

            if final_loss > mid_loss * 0.9:
                recommendations.append(
                    "⏰ **Late Convergence**: Model still improving at end. "
                    "Consider: 1) More training epochs, "
                    "2) Lower learning rate, "
                    "3) Better initialization"
                )

        return recommendations

    def analyze_fitting_patterns(
        self, experiment_data: ExperimentData
    ) -> list[str]:
        """Analyze overfitting/underfitting patterns."""
        recommendations = []

        if not (
            "training_metrics" in experiment_data.metrics
            and "validation_metrics" in experiment_data.metrics
        ):
            return recommendations

        train_df = pd.DataFrame(experiment_data.metrics["training_metrics"])
        val_df = pd.DataFrame(experiment_data.metrics["validation_metrics"])

        # Check for overfitting
        if "loss" in train_df.columns and "loss" in val_df.columns:
            train_loss = train_df["loss"].iloc[-1]
            val_loss = val_df["loss"].iloc[-1]

            if val_loss > train_loss * 1.5:
                recommendations.append(
                    "📈 **Overfitting Detected**: Validation loss "
                    "significantly higher than training. Consider: "
                    "1) Dropout/regularization, 2) Data augmentation, "
                    "3) Early stopping, 4) Model simplification"
                )

        # Check for underfitting
        if "loss" in train_df.columns:
            final_train_loss = train_df["loss"].iloc[-1]
            if final_train_loss > 0.3:
                recommendations.append(
                    "📉 **Underfitting Detected**: High training loss suggests"
                    " model capacity issues. Consider: 1) Deeper architecture,"
                    " 2) Higher learning rate, 3) Better initialization, "
                    "4) Feature engineering"
                )

        return recommendations

    def analyze_learning_rate_patterns(
        self, experiment_data: ExperimentData
    ) -> list[str]:
        """Analyze learning rate patterns and generate recommendations."""
        recommendations = []

        if "training_metrics" not in experiment_data.metrics:
            return recommendations

        df = pd.DataFrame(experiment_data.metrics["training_metrics"])

        # Check for learning rate issues
        if "loss" in df.columns and len(df) > 5:
            loss_changes = df["loss"].diff().dropna()

            # Check for oscillating loss
            oscillations = (loss_changes > 0).sum()
            if oscillations > len(loss_changes) * 0.3:
                recommendations.append(
                    "🔄 **Oscillating Training**: Loss oscillating frequently."
                    " Consider: 1) Lower learning rate, 2) Gradient clipping, "
                    "3) Better momentum settings"
                )

            # Check for very slow improvement
            avg_improvement = -loss_changes.mean()
            if avg_improvement < 0.001:
                recommendations.append(
                    "🐌 **Very Slow Improvement**: Minimal loss reduction. "
                    "Consider: 1) Higher learning rate, "
                    "2) Different optimizer, "
                    "3) Better initialization"
                )

        return recommendations
