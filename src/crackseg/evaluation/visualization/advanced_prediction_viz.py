"""Advanced prediction visualization system.

This module provides comprehensive prediction visualization capabilities
for crack segmentation including comparison grids, confidence maps,
error analysis, and segmentation overlays with configurable templates.
"""

import logging
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from plotly.graph_objs import Figure as PlotlyFigure

from .advanced_ops import (
    create_comparison_grid as _op_create_comparison_grid,
)
from .advanced_ops import (
    create_confidence_map as _op_create_confidence_map,
)
from .advanced_ops import (
    create_error_analysis as _op_create_error_analysis,
)
from .advanced_ops import (
    create_segmentation_overlay as _op_create_segmentation_overlay,
)
from .advanced_ops import (
    create_tabular_comparison as _op_create_tabular_comparison,
)
from .templates.prediction_template import PredictionVisualizationTemplate

logger = logging.getLogger(__name__)


class AdvancedPredictionVisualizer:
    """Advanced prediction visualizer with template system integration.

    This class provides comprehensive prediction visualization capabilities
    including comparison grids, confidence maps, error analysis, and
    segmentation overlays with configurable styling and templates.
    """

    def __init__(
        self,
        style_config: dict[str, Any] | None = None,
        template: PredictionVisualizationTemplate | None = None,
    ) -> None:
        """Initialize the advanced prediction visualizer.

        Args:
            style_config: Optional style configuration dictionary.
            template: Optional prediction visualization template.
        """
        self.template = template or PredictionVisualizationTemplate(
            style_config or {}
        )
        self.config = self.template.get_config()

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

        return _op_create_comparison_grid(
            self.template,
            results,
            save_path=save_path,
            max_images=max_images,
            show_metrics=show_metrics,
            show_confidence=show_confidence,
            grid_layout=grid_layout,
        )

    def create_confidence_map(
        self,
        result: dict[str, Any],
        save_path: str | Path | None = None,
        show_original: bool = True,
        show_contours: bool = True,
    ) -> Figure | PlotlyFigure:
        """Create a confidence map visualization.

        Args:
            result: Prediction analysis result.
            save_path: Optional path to save the visualization.
            show_original: Whether to overlay original image.
            show_contours: Whether to show confidence contours.

        Returns:
            Matplotlib or Plotly figure with confidence map.
        """
        if "probability_mask" not in result:
            raise ValueError(
                "Result must contain 'probability_mask' for confidence map"
            )

        return _op_create_confidence_map(
            self.template,
            result,
            save_path=save_path,
            show_original=show_original,
            show_contours=show_contours,
        )

    def create_error_analysis(
        self,
        result: dict[str, Any],
        save_path: str | Path | None = None,
    ) -> Figure | PlotlyFigure:
        """Create error analysis visualization.

        Args:
            result: Prediction analysis result with ground truth.
            save_path: Optional path to save the visualization.

        Returns:
            Matplotlib or Plotly figure with error analysis.
        """
        if (
            "ground_truth_mask" not in result
            or "prediction_mask" not in result
        ):
            raise ValueError(
                "Result must contain both 'ground_truth_mask' and "
                "'prediction_mask'"
            )

        return _op_create_error_analysis(
            self.template,
            result,
            save_path=save_path,
        )

    def create_segmentation_overlay(
        self,
        result: dict[str, Any],
        save_path: str | Path | None = None,
        show_confidence: bool = True,
    ) -> Figure | PlotlyFigure:
        """Create segmentation overlay visualization.

        Args:
            result: Prediction analysis result.
            save_path: Optional path to save the visualization.
            show_confidence: Whether to show confidence overlay.

        Returns:
            Matplotlib or Plotly figure with segmentation overlay.
        """
        return _op_create_segmentation_overlay(
            self.template,
            result,
            save_path=save_path,
            show_confidence=show_confidence,
        )

    def create_tabular_comparison(
        self,
        results: list[dict[str, Any]],
        save_path: str | Path | None = None,
    ) -> Figure | PlotlyFigure:
        """Create tabular comparison of prediction metrics.

        Args:
            results: List of prediction analysis results.
            save_path: Optional path to save the visualization.

        Returns:
            Matplotlib or Plotly figure with tabular comparison.
        """
        if not results:
            raise ValueError("No results provided for tabular comparison")

        return _op_create_tabular_comparison(
            self.template,
            results,
            save_path=save_path,
        )

    def get_template(self) -> PredictionVisualizationTemplate:
        """Get the prediction visualization template.

        Returns:
            The prediction visualization template.
        """
        return self.template

    def update_template_config(self, updates: dict[str, Any]) -> None:
        """Update template configuration.

        Args:
            updates: Dictionary with configuration updates.
        """
        self.template.update_config(updates)
        self.config = self.template.get_config()
