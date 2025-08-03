"""Parameter distribution analysis visualization.

This module provides functionality for analyzing and visualizing
model parameter distributions from checkpoints.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from plotly.graph_objects import Figure as PlotlyFigure

logger = logging.getLogger(__name__)


class ParameterAnalyzer:
    """Analyzer for model parameter distributions and visualization."""

    def __init__(self, style_config: dict[str, Any]) -> None:
        """Initialize the parameter analyzer.

        Args:
            style_config: Configuration for plot styling
        """
        self.style_config = style_config

    def visualize_parameter_distributions(
        self, model_path: Path, save_path: Path | None = None
    ) -> Figure | PlotlyFigure:
        """Visualize parameter distributions from model checkpoint.

        Args:
            model_path: Path to model checkpoint
            save_path: Path to save the visualization

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {model_path}: {e}")
            return self._create_empty_plot("Parameter Distributions")

        if "model_state_dict" not in checkpoint:
            logger.warning("No model_state_dict found in checkpoint")
            return self._create_empty_plot("Parameter Distributions")

        model_state = checkpoint["model_state_dict"]
        param_stats = self._extract_parameter_statistics(model_state)

        if not param_stats:
            logger.warning("No valid parameters found in checkpoint")
            return self._create_empty_plot("Parameter Distributions")

        # Use static plots for consistency
        return self._create_static_param_distributions(param_stats, save_path)

    def _extract_parameter_statistics(
        self, model_state: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, float]]:
        """Extract statistical information from model parameters.

        Args:
            model_state: Model state dictionary

        Returns:
            Dictionary mapping parameter names to statistics
        """
        param_stats = {}

        for param_name, param_tensor in model_state.items():
            if param_tensor.dim() == 0:  # Skip scalars
                continue

            # Calculate statistics
            param_stats[param_name] = {
                "mean": float(param_tensor.mean()),
                "std": float(param_tensor.std()),
                "min": float(param_tensor.min()),
                "max": float(param_tensor.max()),
                "norm": float(param_tensor.norm()),
            }

        return param_stats

    def _create_static_param_distributions(
        self,
        param_stats: dict[str, dict[str, float]],
        save_path: Path | None,
    ) -> Figure:
        """Create static parameter distribution visualization.

        Args:
            param_stats: Parameter statistics dictionary
            save_path: Path to save the figure

        Returns:
            Matplotlib figure with parameter distributions
        """
        # Implementation would go here
        # This is a placeholder for the actual parameter analysis logic
        fig, ax = plt.subplots(
            figsize=self.style_config.get("figure_size", (12, 8))
        )
        ax.set_title("Parameter Distributions")

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _create_empty_plot(self, title: str) -> Figure:
        """Create an empty plot with title.

        Args:
            title: Title for the plot

        Returns:
            Empty matplotlib figure
        """
        fig, ax = plt.subplots(
            figsize=self.style_config.get("figure_size", (12, 8))
        )
        ax.set_title(title)
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig
