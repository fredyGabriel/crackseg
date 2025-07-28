"""Performance thresholds and training pattern indicators for recommendations.

This module defines the thresholds and indicators used by the recommendation
engine to analyze experiment performance and generate suggestions.
"""


class PerformanceThresholds:
    """Performance thresholds for recommendation analysis."""

    def __init__(self) -> None:
        """Initialize performance thresholds."""
        self.thresholds = {
            "iou": {"excellent": 0.85, "good": 0.75, "poor": 0.60},
            "dice": {"excellent": 0.90, "good": 0.80, "poor": 0.65},
            "f1": {"excellent": 0.90, "good": 0.80, "poor": 0.65},
            "loss": {"excellent": 0.1, "good": 0.2, "poor": 0.4},
        }

    def get_threshold(self, metric: str, level: str) -> float:
        """Get threshold value for a metric and performance level."""
        return self.thresholds.get(metric, {}).get(level, 0.0)


class TrainingIndicators:
    """Training pattern indicators for analysis."""

    def __init__(self) -> None:
        """Initialize training indicators."""
        self.indicators = {
            "overfitting": {
                "val_loss_increasing": (
                    "Validation loss increasing while training loss decreases"
                ),
                "large_gap": (
                    "Large gap between training and validation metrics"
                ),
                "late_plateau": "Metrics plateau late in training",
            },
            "underfitting": {
                "low_metrics": "All metrics below good thresholds",
                "early_plateau": "Metrics plateau early in training",
                "high_loss": "High loss values throughout training",
            },
            "convergence": {
                "stable_metrics": "Stable metrics with good performance",
                "consistent_improvement": (
                    "Consistent improvement throughout training"
                ),
                "optimal_epochs": (
                    "Reached optimal performance in reasonable epochs"
                ),
            },
        }

    def get_indicator(self, pattern: str, indicator: str) -> str:
        """Get indicator description for a training pattern."""
        return self.indicators.get(pattern, {}).get(indicator, "")


class RecommendationCategories:
    """Categories for organizing recommendations."""

    def __init__(self) -> None:
        """Initialize recommendation categories."""
        self.categories = {
            "hyperparameters": "Learning rate, batch size, optimizer settings",
            "architecture": "Model architecture modifications",
            "data": "Data augmentation and preprocessing",
            "training": "Training strategy improvements",
            "loss": "Loss function optimization",
            "regularization": "Regularization techniques",
        }

    def get_category_description(self, category: str) -> str:
        """Get description for a recommendation category."""
        return self.categories.get(category, "")
