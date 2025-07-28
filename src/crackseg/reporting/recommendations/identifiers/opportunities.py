"""Opportunity identification for recommendation engine.

This module provides identification of various optimization opportunities
based on experiment performance and configuration.
"""

import logging

import pandas as pd

from ...config import ExperimentData


class OpportunityIdentifier:
    """Identify optimization opportunities in experiments."""

    def __init__(self) -> None:
        """Initialize the opportunity identifier."""
        self.logger = logging.getLogger(__name__)

    def identify_performance_opportunities(
        self, experiment_data: ExperimentData
    ) -> list[str]:
        """Identify performance-based optimization opportunities."""
        opportunities = []

        if "final_metrics" not in experiment_data.metrics:
            return opportunities

        metrics = experiment_data.metrics["final_metrics"]

        # IoU opportunities
        if "iou" in metrics:
            iou = metrics["iou"]
            if iou < 0.75:
                opportunities.append(
                    "🎯 **IoU Optimization**: Current IoU below 75%. "
                    "Focus on: 1) Boundary-aware losses, "
                    "2) Multi-scale features, "
                    "3) Attention mechanisms"
                )

        # Dice opportunities
        if "dice" in metrics:
            dice = metrics["dice"]
            if dice < 0.80:
                opportunities.append(
                    "🎯 **Dice Optimization**: Current Dice below 80%. "
                    "Focus on: 1) Class imbalance handling, 2) Focal losses, "
                    "3) Post-processing improvements"
                )

        return opportunities

    def identify_training_efficiency_opportunities(
        self, experiment_data: ExperimentData
    ) -> list[str]:
        """Identify training efficiency optimization opportunities."""
        opportunities = []

        if "training_metrics" not in experiment_data.metrics:
            return opportunities

        df = pd.DataFrame(experiment_data.metrics["training_metrics"])

        # Check training time efficiency
        if len(df) > 50:
            opportunities.append(
                "⏱️ **Training Efficiency**: Long training time detected. "
                "Consider: 1) Higher learning rate, 2) Better initialization, "
                "3) Gradient accumulation, 4) Mixed precision training"
            )

        # Check convergence efficiency
        if "loss" in df.columns and len(df) > 20:
            early_loss = df["loss"].iloc[:10].mean()
            final_loss = df["loss"].iloc[-10:].mean()

            if final_loss > early_loss * 0.7:
                opportunities.append(
                    "🔄 **Convergence Efficiency**: Poor convergence pattern. "
                    "Consider: 1) Learning rate scheduling, "
                    "2) Better optimizer, "
                    "3) Architecture modifications"
                )

        return opportunities

    def identify_data_opportunities(
        self, experiment_data: ExperimentData
    ) -> list[str]:
        """Identify data-related optimization opportunities."""
        opportunities = []

        # Suggest data quality improvements
        opportunities.append(
            "📊 **Data Quality**: Consider: 1) Data cleaning and validation, "
            "2) Annotation quality review, 3) Class balance analysis, "
            "4) Data augmentation strategies"
        )

        # Suggest dataset expansion
        opportunities.append(
            "📈 **Dataset Expansion**: Consider: "
            "1) Collecting more diverse data, "
            "2) Synthetic data generation, "
            "3) Transfer learning from larger datasets"
        )

        return opportunities

    def identify_loss_opportunities(
        self, experiment_data: ExperimentData
    ) -> list[str]:
        """Identify loss function optimization opportunities."""
        opportunities = []

        if not experiment_data.config:
            return opportunities

        config = experiment_data.config

        # Check current loss function
        if "loss" in config:
            loss = config["loss"]
            if "bce" in loss.lower() or "cross_entropy" in loss.lower():
                opportunities.append(
                    "🎯 **Loss Function**: Consider advanced losses like "
                    "Dice Loss, Focal Loss, or Boundary Loss for crack "
                    "segmentation"
                )

        # Suggest loss combinations
        opportunities.append(
            "🎯 **Loss Combination**: Consider combining multiple losses: "
            "1) BCE + Dice Loss, 2) Focal + Boundary Loss, "
            "3) IoU Loss + Cross-Entropy"
        )

        return opportunities
