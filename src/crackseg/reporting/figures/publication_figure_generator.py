"""Publication-ready figure generator for experiment reports.

This module provides automated generation of high-quality, publication-ready
figures for experiment results and analyses. Supports multiple formats and
academic/industry publication styles.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# DictConfig is used in type hints but not directly imported
from crackseg.reporting.config import ExperimentData, ReportConfig

from .base import BaseFigureGenerator
from .base import PublicationStyle as _BasePublicationStyle

logger = logging.getLogger(__name__)

PublicationStyle = _BasePublicationStyle


class TrainingCurvesFigureGenerator(BaseFigureGenerator):
    """Generate publication-ready training curves figures."""

    def generate_figure(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
        save_path: Path | None = None,
    ) -> dict[str, Path]:
        """Generate training curves figure."""
        # Extract training metrics
        metrics_data = experiment_data.metrics.get("training_metrics", {})

        if not metrics_data:
            logger.warning("No training metrics found for figure generation")
            return {}

        # Create figure
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(self.style.figure_width, self.style.figure_height),
            constrained_layout=True,
        )
        axes = axes.flatten()

        # Plot training loss
        if "train_loss" in metrics_data:
            epochs = list(range(len(metrics_data["train_loss"])))
            axes[0].plot(
                epochs, metrics_data["train_loss"], label="Training Loss"
            )
            if "val_loss" in metrics_data:
                axes[0].plot(
                    epochs, metrics_data["val_loss"], label="Validation Loss"
                )
            axes[0].set_title("Training and Validation Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(True, alpha=self.style.grid_alpha)

        # Plot accuracy metrics
        if "train_iou" in metrics_data:
            axes[1].plot(
                epochs, metrics_data["train_iou"], label="Training IoU"
            )
            if "val_iou" in metrics_data:
                axes[1].plot(
                    epochs, metrics_data["val_iou"], label="Validation IoU"
                )
            axes[1].set_title("IoU Metrics")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("IoU")
            axes[1].legend()
            axes[1].grid(True, alpha=self.style.grid_alpha)

        # Plot F1 score
        if "train_f1" in metrics_data:
            axes[2].plot(epochs, metrics_data["train_f1"], label="Training F1")
            if "val_f1" in metrics_data:
                axes[2].plot(
                    epochs, metrics_data["val_f1"], label="Validation F1"
                )
            axes[2].set_title("F1 Score")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("F1 Score")
            axes[2].legend()
            axes[2].grid(True, alpha=self.style.grid_alpha)

        # Plot learning rate
        if "learning_rate" in metrics_data:
            axes[3].plot(epochs, metrics_data["learning_rate"])
            axes[3].set_title("Learning Rate Schedule")
            axes[3].set_xlabel("Epoch")
            axes[3].set_ylabel("Learning Rate")
            axes[3].set_yscale("log")
            axes[3].grid(True, alpha=self.style.grid_alpha)

        # Hide unused subplots
        for i in range(4):
            if not axes[i].get_children():
                axes[i].set_visible(False)

        # Save figure
        if save_path:
            return self.save_figure(fig, save_path)

        return {}


class PerformanceComparisonFigureGenerator(BaseFigureGenerator):
    """Generate publication-ready performance comparison figures."""

    def generate_figure(
        self,
        experiments_data: list[ExperimentData],
        config: ReportConfig,
        save_path: Path | None = None,
    ) -> dict[str, Path]:
        """Generate performance comparison figure."""
        if len(experiments_data) < 2:
            logger.warning("Need at least 2 experiments for comparison")
            return {}

        # Extract performance metrics
        metrics_data = []
        experiment_ids = []

        for exp_data in experiments_data:
            performance = exp_data.metrics.get("performance_metrics", {})
            if performance:
                metrics_data.append(performance)
                experiment_ids.append(exp_data.experiment_id)

        if not metrics_data:
            logger.warning("No performance metrics found")
            return {}

        # Create comparison figure
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(self.style.figure_width, self.style.figure_height),
            constrained_layout=True,
        )
        axes = axes.flatten()

        # Prepare data for plotting
        df = pd.DataFrame(metrics_data)
        df.index = experiment_ids

        # Plot IoU comparison
        if "iou" in df.columns:
            axes[0].bar(range(len(df)), df["iou"])
            axes[0].set_title("IoU Comparison")
            axes[0].set_xlabel("Experiment")
            axes[0].set_ylabel("IoU")
            axes[0].set_xticks(range(len(df)))
            axes[0].set_xticklabels(experiment_ids, rotation=45, ha="right")
            axes[0].grid(True, alpha=self.style.grid_alpha)

        # Plot F1 comparison
        if "f1_score" in df.columns:
            axes[1].bar(range(len(df)), df["f1_score"])
            axes[1].set_title("F1 Score Comparison")
            axes[1].set_xlabel("Experiment")
            axes[1].set_ylabel("F1 Score")
            axes[1].set_xticks(range(len(df)))
            axes[1].set_xticklabels(experiment_ids, rotation=45, ha="right")
            axes[1].grid(True, alpha=self.style.grid_alpha)

        # Plot precision/recall
        if "precision" in df.columns and "recall" in df.columns:
            x = np.arange(len(df))
            width = 0.35

            axes[2].bar(
                x - width / 2, df["precision"], width, label="Precision"
            )
            axes[2].bar(x + width / 2, df["recall"], width, label="Recall")
            axes[2].set_title("Precision vs Recall")
            axes[2].set_xlabel("Experiment")
            axes[2].set_ylabel("Score")
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(experiment_ids, rotation=45, ha="right")
            axes[2].legend()
            axes[2].grid(True, alpha=self.style.grid_alpha)

        # Plot training time comparison
        if "training_time_hours" in df.columns:
            axes[3].bar(range(len(df)), df["training_time_hours"])
            axes[3].set_title("Training Time Comparison")
            axes[3].set_xlabel("Experiment")
            axes[3].set_ylabel("Training Time (hours)")
            axes[3].set_xticks(range(len(df)))
            axes[3].set_xticklabels(experiment_ids, rotation=45, ha="right")
            axes[3].grid(True, alpha=self.style.grid_alpha)

        # Hide unused subplots
        for i in range(4):
            if not axes[i].get_children():
                axes[i].set_visible(False)

        # Save figure
        if save_path:
            return self.save_figure(fig, save_path)

        return {}


class ModelArchitectureFigureGenerator(BaseFigureGenerator):
    """Generate publication-ready model architecture figures."""

    def generate_figure(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
        save_path: Path | None = None,
    ) -> dict[str, Path]:
        """Generate model architecture visualization."""
        # Extract model configuration
        model_config = experiment_data.config.get("model", {})

        if not model_config:
            logger.warning("No model configuration found")
            return {}

        # Create architecture diagram
        fig, ax = plt.subplots(
            figsize=(self.style.figure_width, self.style.figure_height),
            constrained_layout=True,
        )

        # Create simple architecture visualization
        layers = []
        if "encoder" in model_config:
            layers.append(f"Encoder: {model_config['encoder']}")
        if "decoder" in model_config:
            layers.append(f"Decoder: {model_config['decoder']}")
        if "bottleneck" in model_config:
            layers.append(f"Bottleneck: {model_config['bottleneck']}")

        # Create horizontal architecture diagram
        y_pos = np.arange(len(layers))
        ax.barh(y_pos, [1] * len(layers))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layers)
        ax.set_xlim(0, 1.2)
        ax.set_title("Model Architecture")
        ax.set_xlabel("Component")

        # Add arrows between components
        for i in range(len(layers) - 1):
            ax.arrow(
                1.1,
                i,
                0,
                1,
                head_width=0.05,
                head_length=0.1,
                fc="black",
                ec="black",
            )

        # Save figure
        if save_path:
            return self.save_figure(fig, save_path)

        return {}


class PublicationFigureGenerator:
    """Main publication-ready figure generator."""

    def __init__(self, style: PublicationStyle | None = None) -> None:
        """Initialize publication figure generator."""
        self.style = style or PublicationStyle()
        self.generators = {
            "training_curves": TrainingCurvesFigureGenerator(self.style),
            "performance_comparison": PerformanceComparisonFigureGenerator(
                self.style
            ),
            "model_architecture": ModelArchitectureFigureGenerator(self.style),
        }

    def generate_publication_figures(
        self,
        experiment_data: ExperimentData,
        config: ReportConfig,
        save_dir: Path | None = None,
    ) -> dict[str, dict[str, Path]]:
        """Generate all publication-ready figures for an experiment."""
        if save_dir is None:
            save_dir = Path("artifacts/visualizations/publication_figures")

        save_dir.mkdir(parents=True, exist_ok=True)

        generated_figures = {}

        # Generate training curves
        training_save_path = (
            save_dir / f"{experiment_data.experiment_id}_training_curves"
        )
        training_figures = self.generators["training_curves"].generate_figure(
            experiment_data, config, training_save_path
        )
        if training_figures:
            generated_figures["training_curves"] = training_figures

        # Generate model architecture
        arch_save_path = (
            save_dir / f"{experiment_data.experiment_id}_architecture"
        )
        arch_figures = self.generators["model_architecture"].generate_figure(
            experiment_data, config, arch_save_path
        )
        if arch_figures:
            generated_figures["model_architecture"] = arch_figures

        return generated_figures

    def generate_comparison_figures(
        self,
        experiments_data: list[ExperimentData],
        config: ReportConfig,
        save_dir: Path | None = None,
    ) -> dict[str, dict[str, Path]]:
        """Generate comparison figures for multiple experiments."""
        if save_dir is None:
            save_dir = Path("artifacts/visualizations/publication_figures")

        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate performance comparison
        comparison_save_path = save_dir / "performance_comparison"
        comparison_figures = self.generators[
            "performance_comparison"
        ].generate_figure(experiments_data, config, comparison_save_path)

        return (
            {"performance_comparison": comparison_figures}
            if comparison_figures
            else {}
        )

    def get_supported_formats(self) -> list[str]:
        """Get list of supported export formats."""
        return self.style.supported_formats

    def get_style_config(self) -> PublicationStyle:
        """Get current publication style configuration."""
        return self.style
