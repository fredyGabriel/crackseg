"""Training visualization template.

This module provides specialized templates for training-related
visualizations including curves, learning rate analysis, and
parameter distributions.
"""

from typing import Any

from .base_template import BaseVisualizationTemplate


class TrainingVisualizationTemplate(BaseVisualizationTemplate):
    """Template for training-related visualizations.

    This template provides optimized styling for training curves,
    learning rate schedules, gradient flow, and parameter
    distribution plots.
    """

    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for training visualizations.

        Returns:
            Dictionary containing default training template configuration.
        """
        return {
            "figure_size": [12, 8],
            "dpi": 300,
            "color_palette": "viridis",
            "grid_alpha": 0.3,
            "line_width": 2,
            "font_size": 12,
            "title_font_size": 14,
            "legend_font_size": 10,
            "training_curves": {
                "subplot_layout": [2, 2],
                "metric_colors": {
                    "loss": "#1f77b4",
                    "accuracy": "#ff7f0e",
                    "precision": "#2ca02c",
                    "recall": "#d62728",
                    "f1": "#9467bd",
                    "iou": "#8c564b",
                },
                "show_grid": True,
                "show_legend": True,
                "legend_position": "upper right",
            },
            "learning_rate": {
                "show_schedule_type": True,
                "show_statistics": True,
                "zoom_enabled": True,
                "highlight_peaks": True,
            },
            "gradient_flow": {
                "layer_colors": "plasma",
                "show_norm_stats": True,
                "normalize_by_layer": True,
                "log_scale": True,
            },
            "parameter_distributions": {
                "bins": 50,
                "show_outliers": True,
                "show_statistics": True,
                "color_by_layer": True,
                "log_scale": False,
            },
        }

    def get_training_curves_config(self) -> dict[str, Any]:
        """Get configuration specific to training curves.

        Returns:
            Training curves configuration dictionary.
        """
        return self.config.get("training_curves", {})

    def get_learning_rate_config(self) -> dict[str, Any]:
        """Get configuration specific to learning rate analysis.

        Returns:
            Learning rate configuration dictionary.
        """
        return self.config.get("learning_rate", {})

    def get_gradient_flow_config(self) -> dict[str, Any]:
        """Get configuration specific to gradient flow visualization.

        Returns:
            Gradient flow configuration dictionary.
        """
        return self.config.get("gradient_flow", {})

    def get_parameter_distributions_config(self) -> dict[str, Any]:
        """Get configuration specific to parameter distributions.

        Returns:
            Parameter distributions configuration dictionary.
        """
        return self.config.get("parameter_distributions", {})

    def update_training_curves_config(self, updates: dict[str, Any]) -> None:
        """Update training curves specific configuration.

        Args:
            updates: Dictionary with training curves configuration updates.
        """
        if "training_curves" not in self.config:
            self.config["training_curves"] = {}
        self.config["training_curves"].update(updates)

    def update_learning_rate_config(self, updates: dict[str, Any]) -> None:
        """Update learning rate specific configuration.

        Args:
            updates: Dictionary with learning rate configuration updates.
        """
        if "learning_rate" not in self.config:
            self.config["learning_rate"] = {}
        self.config["learning_rate"].update(updates)

    def update_gradient_flow_config(self, updates: dict[str, Any]) -> None:
        """Update gradient flow specific configuration.

        Args:
            updates: Dictionary with gradient flow configuration updates.
        """
        if "gradient_flow" not in self.config:
            self.config["gradient_flow"] = {}
        self.config["gradient_flow"].update(updates)

    def update_parameter_distributions_config(
        self, updates: dict[str, Any]
    ) -> None:
        """Update parameter distributions specific configuration.

        Args:
            updates: Dictionary with parameter distributions configuration
                updates.
        """
        if "parameter_distributions" not in self.config:
            self.config["parameter_distributions"] = {}
        self.config["parameter_distributions"].update(updates)
