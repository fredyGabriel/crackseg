"""Experiment visualization utilities for crack segmentation."""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentVisualizer:
    """Create visualizations for experiment results and comparisons."""

    def __init__(self) -> None:
        """Initialize the experiment visualizer."""

    def load_experiment_data(self, experiment_dir: Path) -> dict[str, Any]:
        """
        Load all data for an experiment including metrics and logs.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            Dictionary containing experiment data
        """
        data = {}

        # Load complete summary
        summary_file = experiment_dir / "metrics" / "complete_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                data["summary"] = json.load(f)

        # Load detailed metrics
        metrics_file = experiment_dir / "metrics" / "metrics.jsonl"
        if metrics_file.exists():
            metrics_data = []
            with open(metrics_file) as f:
                for line in f:
                    metrics_data.append(json.loads(line.strip()))
            data["metrics"] = metrics_data

        return data

    def create_comparison_table(
        self, experiments_data: dict[str, dict]
    ) -> pd.DataFrame:
        """
        Create a comparison table of all experiments.

        Args:
            experiments_data: Dictionary mapping experiment names to data

        Returns:
            DataFrame with experiment comparisons
        """
        rows = []

        for exp_name, data in experiments_data.items():
            if "summary" in data:
                summary = data["summary"]
                best_metrics = summary.get("best_metrics", {})

                row = {
                    "Experiment": exp_name,
                    "Final Loss": best_metrics.get("loss", {}).get(
                        "value", np.nan
                    ),
                    "Final IoU": best_metrics.get("iou", {}).get(
                        "value", np.nan
                    ),
                    "Final F1": best_metrics.get("f1", {}).get(
                        "value", np.nan
                    ),
                    "Final Precision": best_metrics.get("precision", {}).get(
                        "value", np.nan
                    ),
                    "Final Recall": best_metrics.get("recall", {}).get(
                        "value", np.nan
                    ),
                    "Total Epochs": summary.get("experiment_info", {}).get(
                        "total_epochs", np.nan
                    ),
                    "Best Epoch": summary.get("experiment_info", {}).get(
                        "best_epoch", np.nan
                    ),
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def plot_training_curves(
        self,
        experiments_data: dict[str, dict],
        title: str = "Training Curves Comparison",
        save_path: str | Path | None = None,
    ) -> None:
        """
        Plot training curves for all experiments.

        Args:
            experiments_data: Dictionary mapping experiment names to data
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        metrics_to_plot = ["loss", "iou", "f1", "precision"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i // 2, i % 2]

            for j, (exp_name, data) in enumerate(experiments_data.items()):
                if "metrics" in data and data["metrics"]:
                    epochs = []
                    values = []

                    for entry in data["metrics"]:
                        if metric in entry:
                            epochs.append(entry.get("epoch", len(epochs) + 1))
                            values.append(entry[metric])

                    if epochs and values:
                        ax.plot(
                            epochs,
                            values,
                            label=exp_name,
                            color=colors[j % len(colors)],
                            linewidth=2,
                            alpha=0.8,
                        )

            ax.set_title(f"{metric.upper()} Over Time", fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.upper())
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Training curves saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def create_performance_radar(
        self,
        experiments_data: dict[str, dict],
        title: str = "Performance Comparison (Radar Chart)",
        save_path: str | Path | None = None,
    ) -> None:
        """
        Create a radar chart comparing final performance metrics.

        Args:
            experiments_data: Dictionary mapping experiment names to data
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        # Prepare data for radar chart
        metrics = ["Final IoU", "Final F1", "Final Precision", "Final Recall"]
        comparison_df = self.create_comparison_table(experiments_data)

        if comparison_df.empty:
            logger.warning("No data available for radar chart")
            return

        # Number of variables
        num_vars = len(metrics)

        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the circle

        # Initialize the plot
        _, ax = plt.subplots(
            figsize=(10, 10), subplot_kw={"projection": "polar"}
        )

        # Plot data for each experiment
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, (_, row) in enumerate(comparison_df.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=row["Experiment"],
                color=colors[i % len(colors)],
            )
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)  # type: ignore
        ax.set_theta_direction(-1)  # type: ignore

        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)

        # Add legend
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        plt.title(title, size=20, color="black", y=1.1)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Radar chart saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def print_detailed_analysis(
        self, comparison_df: pd.DataFrame, title: str = "EXPERIMENT ANALYSIS"
    ) -> None:
        """
        Print detailed analysis of experiment results.

        Args:
            comparison_df: DataFrame with experiment comparisons
            title: Analysis title
        """
        print("\n" + "=" * 60)
        print(f"📊 {title}")
        print("=" * 60)

        if comparison_df.empty:
            print("❌ No data available for analysis")
            return

        # Print comparison table
        print("\n📋 COMPARISON TABLE:")
        print("-" * 60)
        print(comparison_df.to_string(index=False, float_format="%.4f"))

        # Print summary statistics
        print("\n📈 SUMMARY STATISTICS:")
        print("-" * 60)

        # Best performers
        if not comparison_df.empty:
            best_loss_exp = comparison_df.loc[
                comparison_df["Final Loss"].idxmin()
            ]
            print(
                f"• Best Loss: {best_loss_exp['Experiment']} "
                f"({best_loss_exp['Final Loss']:.4f})"
            )

            best_iou_exp = comparison_df.loc[
                comparison_df["Final IoU"].idxmax()
            ]
            print(
                f"• Best IoU: {best_iou_exp['Experiment']} "
                f"({best_iou_exp['Final IoU']:.4f})"
            )

            best_f1_exp = comparison_df.loc[comparison_df["Final F1"].idxmax()]
            print(
                f"• Best F1: {best_f1_exp['Experiment']} "
                f"({best_f1_exp['Final F1']:.4f})"
            )
            print(
                f"• Most stable: {best_f1_exp['Experiment']} "
                "(highest F1 score)"
            )

        # Training efficiency
        print("\n⏱️ TRAINING EFFICIENCY:")
        print("-" * 60)
        if not comparison_df.empty:
            fastest_exp = comparison_df.loc[
                comparison_df["Total Epochs"].idxmin()
            ]
            print(
                f"• Fastest convergence: {fastest_exp['Experiment']} "
                f"({fastest_exp['Total Epochs']:.0f} epochs)"
            )

            most_epochs_exp = comparison_df.loc[
                comparison_df["Total Epochs"].idxmax()
            ]
            print(
                f"• Longest training: {most_epochs_exp['Experiment']} "
                f"({most_epochs_exp['Total Epochs']:.0f} epochs)"
            )

        print("\n" + "=" * 60)

    def find_experiment_directories(
        self, base_path: str = "src/crackseg/outputs/experiments"
    ) -> list[Path]:
        """
        Find all experiment directories in the base path.

        Args:
            base_path: Base path to search for experiments

        Returns:
            List of experiment directory paths
        """
        base_dir = Path(base_path)
        if not base_dir.exists():
            return []

        # Find directories that contain metrics/complete_summary.json
        experiment_dirs = []
        for item in base_dir.iterdir():
            if item.is_dir():
                metrics_file = item / "metrics" / "complete_summary.json"
                if metrics_file.exists():
                    experiment_dirs.append(item)

        # Sort by modification time (newest first)
        experiment_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return experiment_dirs
