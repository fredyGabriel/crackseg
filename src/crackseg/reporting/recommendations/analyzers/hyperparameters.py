"""Hyperparameter analysis for recommendation engine.

This module provides analysis of hyperparameters and generates suggestions
for optimization based on training patterns and performance.
"""

import logging
from typing import Any

import pandas as pd

from ...config import ExperimentData


class HyperparameterAnalyzer:
    """Analyze hyperparameters and generate suggestions."""

    def __init__(self) -> None:
        """Initialize the hyperparameter analyzer."""
        self.logger = logging.getLogger(__name__)

    def analyze_learning_rate(
        self, experiment_data: ExperimentData, config: Any
    ) -> dict[str, Any]:
        """Analyze learning rate and suggest improvements."""
        suggestions = {
            "current": config.get("learning_rate", "Unknown"),
            "recommendations": [],
            "reasoning": "",
        }

        if "learning_rate" in config:
            lr = config["learning_rate"]

            # Analyze based on training patterns
            if "training_metrics" in experiment_data.metrics:
                df = pd.DataFrame(experiment_data.metrics["training_metrics"])
                if "loss" in df.columns:
                    loss_std = df["loss"].std()

                    if loss_std > 0.1:
                        suggestions["recommendations"].append(
                            f"Reduce learning rate to {lr * 0.5}"
                        )
                        suggestions["reasoning"] = (
                            "High loss variance indicates unstable training"
                        )
                    elif loss_std < 0.01:
                        suggestions["recommendations"].append(
                            f"Increase learning rate to {lr * 2.0}"
                        )
                        suggestions["reasoning"] = (
                            "Very stable training suggests room for faster "
                            "learning"
                        )

        return suggestions

    def analyze_batch_size(
        self, experiment_data: ExperimentData, config: Any
    ) -> dict[str, Any]:
        """Analyze batch size and suggest improvements."""
        suggestions = {
            "current": config.get("batch_size", "Unknown"),
            "recommendations": [],
            "reasoning": "",
        }

        if "batch_size" in config:
            batch_size = config["batch_size"]

            # Check for memory efficiency
            if batch_size < 8:
                suggestions["recommendations"].append(
                    f"Increase batch size to {batch_size * 2}"
                )
                suggestions["reasoning"] = (
                    "Small batch size may limit training efficiency"
                )
            elif batch_size > 32:
                suggestions["recommendations"].append(
                    f"Consider reducing batch size to {batch_size // 2}"
                )
                suggestions["reasoning"] = (
                    "Large batch size may reduce generalization"
                )

        return suggestions

    def analyze_optimizer(
        self, experiment_data: ExperimentData, config: Any
    ) -> dict[str, Any]:
        """Analyze optimizer settings and suggest improvements."""
        suggestions = {
            "current": config.get("optimizer", "Unknown"),
            "recommendations": [],
            "reasoning": "",
        }

        # Suggest optimizer based on training patterns
        if "training_metrics" in experiment_data.metrics:
            df = pd.DataFrame(experiment_data.metrics["training_metrics"])
            if "loss" in df.columns:
                loss_std = df["loss"].std()

                if loss_std > 0.1:
                    suggestions["recommendations"].append(
                        "Switch to AdamW with lower learning rate"
                    )
                    suggestions["reasoning"] = (
                        "High variance suggests need for adaptive optimizer"
                    )
                elif loss_std < 0.01:
                    suggestions["recommendations"].append(
                        "Try SGD with momentum for better convergence"
                    )
                    suggestions["reasoning"] = (
                        "Very stable training may benefit from momentum"
                    )

        return suggestions

    def analyze_scheduler(
        self, experiment_data: ExperimentData, config: Any
    ) -> dict[str, Any]:
        """Analyze scheduler settings and suggest improvements."""
        suggestions = {
            "current": config.get("scheduler", "Unknown"),
            "recommendations": [],
            "reasoning": "",
        }

        # Suggest scheduler based on convergence patterns
        if "training_metrics" in experiment_data.metrics:
            df = pd.DataFrame(experiment_data.metrics["training_metrics"])
            if "loss" in df.columns and len(df) > 10:
                early_loss = df["loss"].iloc[:5].mean()
                late_loss = df["loss"].iloc[-5:].mean()

                if late_loss > early_loss * 0.8:
                    suggestions["recommendations"].append(
                        "Add CosineAnnealingLR scheduler"
                    )
                    suggestions["reasoning"] = (
                        "Slow convergence suggests need for learning rate "
                        "scheduling"
                    )
                elif late_loss < early_loss * 0.3:
                    suggestions["recommendations"].append(
                        "Try StepLR with longer intervals"
                    )
                    suggestions["reasoning"] = (
                        "Good convergence may benefit from step-based "
                        "scheduling"
                    )

        return suggestions

    def analyze_regularization(
        self, experiment_data: ExperimentData, config: Any
    ) -> dict[str, Any]:
        """Analyze regularization settings and suggest improvements."""
        suggestions = {
            "current": "Unknown",
            "recommendations": [],
            "reasoning": "",
        }

        # Check for overfitting
        if (
            "training_metrics" in experiment_data.metrics
            and "validation_metrics" in experiment_data.metrics
        ):
            train_df = pd.DataFrame(
                experiment_data.metrics["training_metrics"]
            )
            val_df = pd.DataFrame(
                experiment_data.metrics["validation_metrics"]
            )

            if "loss" in train_df.columns and "loss" in val_df.columns:
                train_loss = train_df["loss"].iloc[-1]
                val_loss = val_df["loss"].iloc[-1]

                if val_loss > train_loss * 1.5:
                    suggestions["recommendations"].extend(
                        [
                            "Add dropout layers (rate: 0.1-0.3)",
                            "Implement weight decay (1e-4)",
                            "Use data augmentation",
                            "Consider early stopping",
                        ]
                    )
                    suggestions["reasoning"] = (
                        "Overfitting detected - regularization needed"
                    )

        return suggestions

    def analyze_data_augmentation(
        self, experiment_data: ExperimentData, config: Any
    ) -> dict[str, Any]:
        """Analyze data augmentation settings and suggest improvements."""
        suggestions = {
            "current": "Unknown",
            "recommendations": [],
            "reasoning": "",
        }

        # Suggest augmentation based on performance
        if "final_metrics" in experiment_data.metrics:
            metrics = experiment_data.metrics["final_metrics"]
            if "iou" in metrics and metrics["iou"] < 0.75:
                suggestions["recommendations"].extend(
                    [
                        "Add rotation augmentation (±15°)",
                        "Implement elastic transformations",
                        "Add brightness/contrast variations",
                        "Consider mixup or cutmix techniques",
                    ]
                )
                suggestions["reasoning"] = (
                    "Low IoU suggests need for better data augmentation"
                )

        return suggestions
