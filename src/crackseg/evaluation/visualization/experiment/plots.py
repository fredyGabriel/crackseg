"""Experiment plotting functionality.

This module provides functionality for creating various plots
for experiment analysis and comparison.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentPlotter:
    """Create various plots for experiment analysis."""

    def __init__(self) -> None:
        """Initialize the experiment plotter."""

    def plot_training_curves(
        self,
        experiments_data: dict[str, dict],
        title: str = "Training Curves Comparison",
        save_path: str | Path | None = None,
    ) -> None:
        """Plot training curves for multiple experiments.

        Args:
            experiments_data: Dictionary mapping experiment names to data
            title: Plot title
            save_path: Optional path to save the plot
        """
        _fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        metrics = ["loss", "iou", "precision", "recall"]
        metric_names = ["Loss", "IoU", "Precision", "Recall"]

        for i, (metric, metric_name) in enumerate(
            zip(metrics, metric_names, strict=False)
        ):
            ax = axes[i]

            for exp_name, data in experiments_data.items():
                if "metrics" in data:
                    epochs = []
                    values = []

                    for epoch_data in data["metrics"]:
                        if metric in epoch_data:
                            epochs.append(epoch_data.get("epoch", len(epochs)))
                            values.append(epoch_data[metric])

                    if epochs and values:
                        ax.plot(
                            epochs,
                            values,
                            label=exp_name,
                            marker="o",
                            markersize=3,
                        )

            ax.set_title(f"{metric_name} vs Epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Training curves saved to: {save_path}")

        plt.show()

    def create_performance_radar(
        self,
        experiments_data: dict[str, dict],
        title: str = "Performance Comparison (Radar Chart)",
        save_path: str | Path | None = None,
    ) -> None:
        """Create radar chart for performance comparison.

        Args:
            experiments_data: Dictionary mapping experiment names to data
            title: Plot title
            save_path: Optional path to save the plot
        """
        # Extract final metrics for each experiment
        metrics_data = {}

        for exp_name, data in experiments_data.items():
            if "summary" in data:
                summary = data["summary"]
                best_metrics = summary.get("best_metrics", {})

                metrics_data[exp_name] = {
                    "IoU": best_metrics.get("iou", {}).get("value", 0),
                    "F1": best_metrics.get("f1", {}).get("value", 0),
                    "Precision": best_metrics.get("precision", {}).get(
                        "value", 0
                    ),
                    "Recall": best_metrics.get("recall", {}).get("value", 0),
                }

        if not metrics_data:
            logger.warning("No metrics data available for radar chart")
            return

        # Create radar chart
        categories = list(metrics_data[list(metrics_data.keys())[0]].keys())
        N = len(categories)

        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle

        _fig, ax = plt.subplots(
            figsize=(10, 10), subplot_kw={"projection": "polar"}
        )

        for exp_name, metrics in metrics_data.items():
            values = [metrics[cat] for cat in categories]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, "o-", linewidth=2, label=exp_name)
            ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=16, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Radar chart saved to: {save_path}")

        plt.show()

    def print_detailed_analysis(
        self, comparison_df: pd.DataFrame, title: str = "EXPERIMENT ANALYSIS"
    ) -> None:
        """Print detailed analysis of experiment results.

        Args:
            comparison_df: DataFrame with experiment comparisons
            title: Analysis title
        """
        print("=" * 80)
        print(f"{title:^80}")
        print("=" * 80)

        # Display comparison table
        print("\nEXPERIMENT COMPARISON TABLE:")
        print("-" * 80)
        print(comparison_df.to_string(index=False))

        # Statistical summary
        print("\nSTATISTICAL SUMMARY:")
        print("-" * 80)

        numeric_columns = comparison_df.select_dtypes(
            include=[np.number]
        ).columns
        for col in numeric_columns:
            if col != "Total Epochs" and col != "Best Epoch":
                print(f"{col}:")
                print(f"  Mean: {comparison_df[col].mean():.4f}")
                print(f"  Std:  {comparison_df[col].std():.4f}")
                print(f"  Min:  {comparison_df[col].min():.4f}")
                print(f"  Max:  {comparison_df[col].max():.4f}")
                print()

        # Best performing experiment
        if "Final IoU" in comparison_df.columns:
            best_iou_idx = comparison_df["Final IoU"].idxmax()
            best_experiment = comparison_df.loc[best_iou_idx, "Experiment"]
            best_iou = comparison_df.loc[best_iou_idx, "Final IoU"]
            print("BEST PERFORMING EXPERIMENT:")
            print(f"  Experiment: {best_experiment}")
            print(f"  IoU: {best_iou:.4f}")

        print("=" * 80)
