"""Segmentation overlay visualization.

This module provides functionality for creating segmentation overlays
and tabular comparisons for prediction results.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from plotly.graph_objs import Figure as PlotlyFigure

from ..templates.prediction_template import PredictionVisualizationTemplate

logger = logging.getLogger(__name__)


class SegmentationOverlayVisualizer:
    """Visualizer for segmentation overlays and tabular comparisons."""

    def __init__(
        self,
        template: PredictionVisualizationTemplate | None = None,
    ) -> None:
        """Initialize the segmentation overlay visualizer.

        Args:
            template: Optional prediction visualization template.
        """
        self.template = template or PredictionVisualizationTemplate()

    def create_segmentation_overlay(
        self,
        result: dict[str, Any],
        save_path: str | Path | None = None,
        show_confidence: bool = True,
    ) -> Figure | PlotlyFigure:
        """Create a segmentation overlay visualization.

        Args:
            result: Prediction result dictionary.
            save_path: Optional path to save the visualization.
            show_confidence: Whether to show confidence overlay.

        Returns:
            Matplotlib or Plotly figure with segmentation overlay.
        """
        # Implementation would go here
        # This is a placeholder for the actual overlay logic
        fig, _ax = plt.subplots(1, 1, figsize=(10, 8))

        if save_path:
            self._save_visualization(fig, save_path)

        return fig

    def create_tabular_comparison(
        self,
        results: list[dict[str, Any]],
        save_path: str | Path | None = None,
    ) -> Figure | PlotlyFigure:
        """Create a tabular comparison of prediction results.

        Args:
            results: List of prediction results.
            save_path: Optional path to save the visualization.

        Returns:
            Matplotlib or Plotly figure with tabular comparison.
        """
        # Implementation would go here
        # This is a placeholder for the actual tabular comparison logic
        fig, _ax = plt.subplots(1, 1, figsize=(12, 8))

        if save_path:
            self._save_visualization(fig, save_path)

        return fig

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
        logger.info(f"Overlay visualization saved to: {save_path}")
