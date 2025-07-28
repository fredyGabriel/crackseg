"""Learning rate analysis visualization module.

This module provides functionality for analyzing and visualizing
learning rate schedules during training.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.figure import Figure
from plotly.graph_objects import Figure as PlotlyFigure

logger = logging.getLogger(__name__)


class LearningRateAnalyzer:
    """Analyzer for learning rate schedules and visualization."""

    def __init__(self, style_config: dict[str, Any]) -> None:
        """Initialize the learning rate analyzer.

        Args:
            style_config: Configuration for plot styling
        """
        self.style_config = style_config

    def analyze_learning_rate_schedule(
        self, training_data: dict[str, Any], save_path: Path | None = None
    ) -> Figure | PlotlyFigure:
        """Analyze and visualize learning rate schedule.

        Args:
            training_data: Training data containing learning rate information
            save_path: Path to save the visualization

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        if "metrics" not in training_data:
            logger.warning("No metrics data available, creating empty plot")
            return self._create_empty_plot("Learning Rate Schedule")

        metrics_data = training_data["metrics"]
        lr_values = [entry.get("lr", 0) for entry in metrics_data]

        if not lr_values or all(lr == 0 for lr in lr_values):
            logger.warning("No learning rate data available")
            return self._create_empty_plot("Learning Rate Schedule")

        epochs = [entry["epoch"] for entry in metrics_data]

        # Determine if we should use interactive mode
        # For now, use static plots for consistency
        return self._create_static_lr_analysis(epochs, lr_values, save_path)

    def _create_static_lr_analysis(
        self,
        epochs: list[int],
        lr_values: list[float],
        save_path: Path | None,
    ) -> Figure:
        """Create static learning rate analysis using matplotlib.

        Args:
            epochs: List of epoch numbers
            lr_values: List of learning rate values
            save_path: Path to save the figure

        Returns:
            Matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=self.style_config["figure_size"]
        )

        # Linear scale plot
        ax1.plot(epochs, lr_values, linewidth=self.style_config["line_width"])
        ax1.set_title("Learning Rate Schedule (Linear Scale)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Learning Rate")
        ax1.grid(True, alpha=self.style_config["grid_alpha"])

        # Log scale plot
        ax2.semilogy(
            epochs, lr_values, linewidth=self.style_config["line_width"]
        )
        ax2.set_title("Learning Rate Schedule (Log Scale)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.grid(True, alpha=self.style_config["grid_alpha"])

        plt.tight_layout()

        if save_path:
            fig.savefig(
                save_path.with_suffix(".png"), dpi=self.style_config["dpi"]
            )
            logger.info(f"Learning rate analysis saved to: {save_path}")

        return fig

    def _create_interactive_lr_analysis(
        self,
        epochs: list[int],
        lr_values: list[float],
        save_path: Path | None,
    ) -> PlotlyFigure:
        """Create interactive learning rate analysis using Plotly.

        Args:
            epochs: List of epoch numbers
            lr_values: List of learning rate values
            save_path: Path to save the figure

        Returns:
            Plotly Figure
        """
        fig = go.Figure()

        # Add linear scale trace
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=lr_values,
                mode="lines",
                name="Learning Rate",
                line={"width": 2},
            )
        )

        # Add log scale trace
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=lr_values,
                mode="lines",
                name="Learning Rate (Log)",
                line={"width": 2},
                yaxis="y2",
            )
        )

        fig.update_layout(
            title="Learning Rate Schedule Analysis",
            xaxis={"title": "Epoch"},
            yaxis={"title": "Learning Rate", "type": "linear"},
            yaxis2={
                "title": "Learning Rate (Log)",
                "type": "log",
                "overlaying": "y",
            },
            height=600,
            width=800,
        )

        if save_path:
            fig.write_html(str(save_path.with_suffix(".html")))
            logger.info(
                f"Interactive learning rate analysis saved to: {save_path}"
            )

        return fig

    def _create_empty_plot(self, title: str) -> Figure:
        """Create an empty plot with a message.

        Args:
            title: Title for the empty plot

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            f"No data available for {title}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
        )
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
