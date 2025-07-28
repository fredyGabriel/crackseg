"""Prediction visualization template.

This module provides specialized templates for prediction-related
visualizations including comparison grids, confidence maps, and
error analysis plots.
"""

from typing import Any

from .base_template import BaseVisualizationTemplate


class PredictionVisualizationTemplate(BaseVisualizationTemplate):
    """Template for prediction-related visualizations.

    This template provides optimized styling for prediction
    comparison grids, confidence maps, and error analysis
    visualizations.
    """

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for prediction visualizations.

        Returns:
            Dictionary containing default prediction template configuration.
        """
        return {
            "figure_size": [16, 12],
            "dpi": 300,
            "color_palette": "plasma",
            "grid_alpha": 0.2,
            "line_width": 1.5,
            "font_size": 11,
            "title_font_size": 16,
            "legend_font_size": 12,
            "comparison_grid": {
                "grid_layout": [3, 3],
                "image_size": [256, 256],
                "show_titles": True,
                "show_metrics": True,
                "border_width": 2,
                "border_color": "black",
            },
            "confidence_map": {
                "colormap": "viridis",
                "show_colorbar": True,
                "colorbar_position": "right",
                "threshold_contours": True,
                "contour_levels": 10,
                "show_uncertainty": True,
            },
            "error_analysis": {
                "error_types": [
                    "false_positive",
                    "false_negative",
                    "boundary_error",
                ],
                "show_distributions": True,
                "show_correlation": True,
                "highlight_errors": True,
                "error_colors": {
                    "false_positive": "#ff0000",
                    "false_negative": "#00ff00",
                    "boundary_error": "#0000ff",
                },
            },
            "segmentation_overlay": {
                "overlay_alpha": 0.6,
                "ground_truth_color": "#00ff00",
                "prediction_color": "#ff0000",
                "show_legend": True,
                "blend_mode": "alpha",
            },
        }

    def get_comparison_grid_config(self) -> dict[str, Any]:
        """Get configuration specific to comparison grids.

        Returns:
            Comparison grid configuration dictionary.
        """
        return self.config.get("comparison_grid", {})

    def get_confidence_map_config(self) -> dict[str, Any]:
        """Get configuration specific to confidence maps.

        Returns:
            Confidence map configuration dictionary.
        """
        return self.config.get("confidence_map", {})

    def get_error_analysis_config(self) -> dict[str, Any]:
        """Get configuration specific to error analysis.

        Returns:
            Error analysis configuration dictionary.
        """
        return self.config.get("error_analysis", {})

    def get_segmentation_overlay_config(self) -> dict[str, Any]:
        """Get configuration specific to segmentation overlays.

        Returns:
            Segmentation overlay configuration dictionary.
        """
        return self.config.get("segmentation_overlay", {})

    def update_comparison_grid_config(self, updates: dict[str, Any]) -> None:
        """Update comparison grid specific configuration.

        Args:
            updates: Dictionary with comparison grid configuration updates.
        """
        if "comparison_grid" not in self.config:
            self.config["comparison_grid"] = {}
        self.config["comparison_grid"].update(updates)

    def update_confidence_map_config(self, updates: dict[str, Any]) -> None:
        """Update confidence map specific configuration.

        Args:
            updates: Dictionary with confidence map configuration updates.
        """
        if "confidence_map" not in self.config:
            self.config["confidence_map"] = {}
        self.config["confidence_map"].update(updates)

    def update_error_analysis_config(self, updates: dict[str, Any]) -> None:
        """Update error analysis specific configuration.

        Args:
            updates: Dictionary with error analysis configuration updates.
        """
        if "error_analysis" not in self.config:
            self.config["error_analysis"] = {}
        self.config["error_analysis"].update(updates)

    def update_segmentation_overlay_config(
        self, updates: dict[str, Any]
    ) -> None:
        """Update segmentation overlay specific configuration.

        Args:
            updates: Dictionary with segmentation overlay configuration
                updates.
        """
        if "segmentation_overlay" not in self.config:
            self.config["segmentation_overlay"] = {}
        self.config["segmentation_overlay"].update(updates)
