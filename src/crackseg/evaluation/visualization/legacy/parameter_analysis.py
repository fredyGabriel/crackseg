"""Parameter distribution analysis visualization module.

This module provides functionality for analyzing and visualizing
model parameter distributions from checkpoints.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
            Matplotlib Figure
        """
        # Select top parameters by norm for visualization
        sorted_params = sorted(
            param_stats.items(), key=lambda x: x[1]["norm"], reverse=True
        )[:10]

        param_names = [name.replace(".", "\n") for name, _ in sorted_params]
        means = [stats["mean"] for _, stats in sorted_params]
        stds = [stats["std"] for _, stats in sorted_params]
        norms = [stats["norm"] for _, stats in sorted_params]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=self.style_config["figure_size"]
        )

        # Mean values
        ax1.bar(range(len(param_names)), means, alpha=0.7)
        ax1.set_title("Parameter Means")
        ax1.set_xticks(range(len(param_names)))
        ax1.set_xticklabels(param_names, rotation=45, ha="right")
        ax1.grid(True, alpha=self.style_config["grid_alpha"])

        # Standard deviations
        ax2.bar(range(len(param_names)), stds, alpha=0.7, color="orange")
        ax2.set_title("Parameter Standard Deviations")
        ax2.set_xticks(range(len(param_names)))
        ax2.set_xticklabels(param_names, rotation=45, ha="right")
        ax2.grid(True, alpha=self.style_config["grid_alpha"])

        # Parameter norms
        ax3.bar(range(len(param_names)), norms, alpha=0.7, color="green")
        ax3.set_title("Parameter Norms")
        ax3.set_xticks(range(len(param_names)))
        ax3.set_xticklabels(param_names, rotation=45, ha="right")
        ax3.grid(True, alpha=self.style_config["grid_alpha"])

        # Statistics summary
        ax4.text(
            0.1,
            0.9,
            f"Total Parameters: {len(param_stats)}",
            transform=ax4.transAxes,
            fontsize=12,
        )
        ax4.text(
            0.1,
            0.8,
            f"Max Norm: {max(norms):.4f}",
            transform=ax4.transAxes,
            fontsize=12,
        )
        ax4.text(
            0.1,
            0.7,
            f"Min Norm: {min(norms):.4f}",
            transform=ax4.transAxes,
            fontsize=12,
        )
        ax4.set_title("Summary Statistics")
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(
                save_path.with_suffix(".png"), dpi=self.style_config["dpi"]
            )
            logger.info(f"Parameter distributions saved to: {save_path}")

        return fig

    def _create_interactive_param_distributions(
        self,
        param_stats: dict[str, dict[str, float]],
        save_path: Path | None,
    ) -> PlotlyFigure:
        """Create interactive parameter distribution visualization.

        Args:
            param_stats: Parameter statistics dictionary
            save_path: Path to save the figure

        Returns:
            Plotly Figure
        """
        # Select top parameters by norm
        sorted_params = sorted(
            param_stats.items(), key=lambda x: x[1]["norm"], reverse=True
        )[:10]

        param_names = [name for name, _ in sorted_params]
        means = [stats["mean"] for _, stats in sorted_params]
        stds = [stats["std"] for _, stats in sorted_params]
        norms = [stats["norm"] for _, stats in sorted_params]

        fig = go.Figure()

        # Add bar chart for means
        fig.add_trace(
            go.Bar(
                x=param_names,
                y=means,
                name="Mean",
                marker_color="blue",
            )
        )

        # Add bar chart for stds
        fig.add_trace(
            go.Bar(
                x=param_names,
                y=stds,
                name="Std",
                marker_color="orange",
            )
        )

        # Add bar chart for norms
        fig.add_trace(
            go.Bar(
                x=param_names,
                y=norms,
                name="Norm",
                marker_color="green",
            )
        )

        fig.update_layout(
            title="Parameter Distributions",
            xaxis_title="Parameter",
            yaxis_title="Value",
            barmode="group",
            height=500,
            width=800,
        )

        if save_path:
            fig.write_html(str(save_path.with_suffix(".html")))
            logger.info(
                f"Interactive parameter distributions saved to: {save_path}"
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
