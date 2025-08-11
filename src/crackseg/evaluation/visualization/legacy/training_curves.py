"""Training curves visualization module.

This module provides functionality for creating training curves
visualizations including loss, accuracy, and other metrics over epochs.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
from matplotlib.figure import Figure
from plotly.graph_objects import Figure as PlotlyFigure

logger = logging.getLogger(__name__)


class TrainingCurvesVisualizer:
    """Visualizer for training curves and metrics over time."""

    def __init__(self, style_config: dict[str, Any]) -> None:
        """Initialize the training curves visualizer.

        Args:
            style_config: Configuration for plot styling
        """
        self.style_config = style_config

    def create_training_curves(
        self,
        training_data: dict[str, Any],
        metrics: list[str] | None = None,
        save_path: Path | None = None,
        interactive: bool = True,
    ) -> Figure | PlotlyFigure:
        """Create training curves visualization.

        Args:
            training_data: Training data containing metrics
            metrics: List of metrics to plot (auto-detected if None)
            save_path: Path to save the visualization
            interactive: Whether to create interactive plot

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        if "metrics" not in training_data or not training_data["metrics"]:
            logger.warning("No metrics data available, creating empty plot")
            return self._create_empty_plot("No Training Data Available")

        metrics_data = training_data["metrics"]

        # Auto-detect metrics if not specified
        if metrics is None:
            sample_metric = metrics_data[0]
            metrics = [
                key
                for key in sample_metric.keys()
                if key != "epoch"
                and isinstance(sample_metric[key], int | float)
            ]

        if not metrics:
            logger.warning("No valid metrics found")
            return self._create_empty_plot("No Valid Metrics")

        # Determine subplot layout
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        if interactive:
            return self._create_interactive_training_curves(
                metrics_data, metrics, n_rows, n_cols, save_path
            )
        else:
            return self._create_static_training_curves(
                metrics_data, metrics, n_rows, n_cols, save_path
            )

    def _create_static_training_curves(
        self,
        metrics_data: list[dict[str, Any]],
        metrics: list[str],
        n_rows: int,
        n_cols: int,
        save_path: Path | None,
    ) -> Figure:
        """Create static training curves using matplotlib.

        Args:
            metrics_data: List of metric dictionaries
            metrics: List of metric names to plot
            n_rows: Number of subplot rows
            n_cols: Number of subplot columns
            save_path: Path to save the figure

        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=self.style_config["figure_size"]
        )
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        epochs = [entry["epoch"] for entry in metrics_data]

        for i, metric in enumerate(metrics):
            if i < len(axes):
                ax = axes[i]
                values = [entry.get(metric, 0) for entry in metrics_data]

                ax.plot(
                    epochs, values, linewidth=self.style_config["line_width"]
                )
                ax.set_title(f"{metric.replace('_', ' ').title()}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.grid(True, alpha=self.style_config["grid_alpha"])

        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            fig.savefig(
                save_path.with_suffix(".png"), dpi=self.style_config["dpi"]
            )
            logger.info(f"Training curves saved to: {save_path}")

        return fig

    def _create_interactive_training_curves(
        self,
        metrics_data: list[dict[str, Any]],
        metrics: list[str],
        n_rows: int,
        n_cols: int,
        save_path: Path | None,
    ) -> PlotlyFigure:
        """Create interactive training curves using Plotly.

        Args:
            metrics_data: List of metric dictionaries
            metrics: List of metric names to plot
            n_rows: Number of subplot rows
            n_cols: Number of subplot columns
            save_path: Path to save the figure

        Returns:
            Plotly Figure
        """
        fig = sp.make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[m.replace("_", " ").title() for m in metrics],
        )

        epochs = [entry["epoch"] for entry in metrics_data]

        for i, metric in enumerate(metrics):
            row = i // n_cols + 1
            col = i % n_cols + 1

            values = [entry.get(metric, 0) for entry in metrics_data]

            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    mode="lines",
                    name=metric.replace("_", " ").title(),
                    line={"width": 2},
                ),
                row=row,
                col=col,
            )

        fig.update_layout(
            title="Training Curves",
            height=300 * n_rows,
            width=400 * n_cols,
            showlegend=False,
        )

        if save_path:
            fig.write_html(str(save_path.with_suffix(".html")))
            logger.info(f"Interactive training curves saved to: {save_path}")

        return fig

    def _create_empty_plot(self, title: str) -> Figure:
        from crackseg.evaluation.visualization.utils.plot_utils import (
            create_empty_plot,
        )

        fig, _ = create_empty_plot(
            f"No data available for {title}", figsize=(8, 6)
        )
        return fig
