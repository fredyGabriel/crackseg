"""Grid visualization for prediction comparisons.

This module provides functionality for creating comparison grids
of prediction results with configurable layouts and styling.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from plotly.graph_objs import Figure as PlotlyFigure

from ..templates.prediction_template import PredictionVisualizationTemplate

logger = logging.getLogger(__name__)


class PredictionGridVisualizer:
    """Grid visualizer for prediction comparisons."""

    def __init__(
        self,
        template: PredictionVisualizationTemplate | None = None,
    ) -> None:
        """Initialize the grid visualizer.

        Args:
            template: Optional prediction visualization template.
        """
        self.template = template or PredictionVisualizationTemplate()

    def create_comparison_grid(
        self,
        results: list[dict[str, Any]],
        save_path: str | Path | None = None,
        max_images: int = 9,
        show_metrics: bool = True,
        show_confidence: bool = True,
        grid_layout: tuple[int, int] | None = None,
    ) -> Figure | PlotlyFigure:
        """Create a configurable comparison grid of prediction results.

        Args:
            results: List of prediction analysis results.
            save_path: Optional path to save the visualization.
            max_images: Maximum number of images to display.
            show_metrics: Whether to show metrics on each image.
            show_confidence: Whether to show confidence maps.
            grid_layout: Optional custom grid layout (rows, cols).

        Returns:
            Matplotlib or Plotly figure with comparison grid.

        Raises:
            ValueError: If no results provided or invalid grid layout.
        """
        if not results:
            raise ValueError("No results provided for comparison")

        # Limit number of images
        results = results[:max_images]
        n_results = len(results)

        # Determine grid layout
        if grid_layout:
            rows, cols = grid_layout
            if rows * cols < n_results:
                raise ValueError(
                    f"Grid layout {grid_layout} too small for "
                    f"{n_results} results"
                )
        else:
            cols = min(3, n_results)
            rows = (n_results + cols - 1) // cols

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each result
        for idx, result in enumerate(results):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            self._plot_single_result(ax, result, show_metrics, show_confidence)

        # Hide empty subplots
        for idx in range(n_results, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)

        plt.tight_layout()

        if save_path:
            self._save_visualization(fig, save_path)

        return fig

    def _plot_single_result(
        self,
        ax: Any,
        result: dict[str, Any],
        show_metrics: bool,
        show_confidence: bool,
    ) -> None:
        """Plot a single result in the grid.

        Args:
            ax: Matplotlib axes to plot on.
            result: Result dictionary.
            show_metrics: Whether to show metrics.
            show_confidence: Whether to show confidence.
        """
        # Implementation would go here
        # This is a placeholder for the actual plotting logic
        pass

    def _save_visualization(
        self, fig: Figure | PlotlyFigure, save_path: str | Path
    ) -> None:
        """Save visualization to file.

        Args:
            fig: Figure to save.
            save_path: Path to save the figure.
        """
        if isinstance(fig, Figure):
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            fig.write_image(str(save_path))
        logger.info(f"Grid visualization saved to: {save_path}")
