"""Base visualization template system.

This module provides the foundational template interface for
consistent visualization styling across the CrackSeg system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from plotly.graph_objects import Figure as PlotlyFigure

logger = logging.getLogger(__name__)


class BaseVisualizationTemplate(ABC):
    """Base class for visualization templates.

    This abstract base class defines the interface that all
    visualization templates must implement for consistent
    styling and behavior across the system.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the template with configuration.

        Args:
            config: Template configuration dictionary. If None,
                uses default configuration.
        """
        self.config = config or self.get_default_config()
        self._validate_config()
        self._setup_style()

    @abstractmethod
    def get_default_config(self) -> dict[str, Any]:
        """Get default configuration for this template.

        Returns:
            Dictionary containing default configuration values.
        """
        pass

    def _validate_config(self) -> None:
        """Validate template configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        required_keys = [
            "figure_size",
            "dpi",
            "color_palette",
            "grid_alpha",
            "line_width",
            "font_size",
            "title_font_size",
            "legend_font_size",
        ]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def _setup_style(self) -> None:
        """Setup matplotlib and seaborn styling based on configuration."""
        try:
            plt.style.use("seaborn-v0_8")
            sns.set_palette(self.config["color_palette"])

            # Configure matplotlib defaults
            plt.rcParams.update(
                {
                    "figure.dpi": self.config["dpi"],
                    "font.size": self.config["font_size"],
                    "axes.titlesize": self.config["title_font_size"],
                    "axes.labelsize": self.config["font_size"],
                    "xtick.labelsize": self.config["font_size"] - 2,
                    "ytick.labelsize": self.config["font_size"] - 2,
                    "legend.fontsize": self.config["legend_font_size"],
                    "lines.linewidth": self.config["line_width"],
                }
            )

            logger.debug(
                f"Style configured for template: {self.__class__.__name__}"
            )

        except Exception as e:
            logger.warning(f"Failed to setup style: {e}")

    def apply_template(
        self, fig: Figure | PlotlyFigure
    ) -> Figure | PlotlyFigure:
        """Apply template styling to a figure.

        Args:
            fig: Figure to apply styling to.

        Returns:
            Styled figure.
        """
        if isinstance(fig, Figure):
            return self._apply_matplotlib_template(fig)
        else:
            return self._apply_plotly_template(fig)

    def _apply_matplotlib_template(self, fig: Figure) -> Figure:
        """Apply template styling to matplotlib figure.

        Args:
            fig: Matplotlib figure to style.

        Returns:
            Styled matplotlib figure.
        """
        # Set figure size
        fig.set_size_inches(*self.config["figure_size"])
        fig.set_dpi(self.config["dpi"])

        # Apply styling to all axes
        for ax in fig.axes:
            ax.grid(alpha=self.config["grid_alpha"])

            # Style title
            if ax.get_title():
                ax.set_title(
                    ax.get_title(),
                    fontsize=self.config["title_font_size"],
                    fontweight="bold",
                )

            # Style labels
            if ax.get_xlabel():
                ax.set_xlabel(
                    ax.get_xlabel(), fontsize=self.config["font_size"]
                )
            if ax.get_ylabel():
                ax.set_ylabel(
                    ax.get_ylabel(), fontsize=self.config["font_size"]
                )

        return fig

    def _apply_plotly_template(self, fig: PlotlyFigure) -> PlotlyFigure:
        """Apply template styling to plotly figure.

        Args:
            fig: Plotly figure to style.

        Returns:
            Styled plotly figure.
        """
        # Update layout with template styling
        fig.update_layout(
            width=self.config["figure_size"][0] * self.config["dpi"],
            height=self.config["figure_size"][1] * self.config["dpi"],
            font={
                "size": self.config["font_size"],
                "family": "Arial, sans-serif",
            },
            title={
                "font": {
                    "size": self.config["title_font_size"],
                    "color": "black",
                }
            },
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        # Update all traces - simplified to avoid type issues
        # Note: Plotly trace styling is handled automatically by the layout

        return fig

    def get_config(self) -> dict[str, Any]:
        """Get current template configuration.

        Returns:
            Current configuration dictionary.
        """
        return self.config.copy()

    def update_config(self, updates: dict[str, Any]) -> None:
        """Update template configuration.

        Args:
            updates: Dictionary with configuration updates.
        """
        self.config.update(updates)
        self._validate_config()
        self._setup_style()
        logger.info(
            f"Updated configuration for template: {self.__class__.__name__}"
        )

    def create_empty_plot(
        self, title: str = "No Data Available"
    ) -> Figure | PlotlyFigure:
        """Create an empty plot with informative message.

        Args:
            title: Title for the empty plot.

        Returns:
            Empty matplotlib figure with message.
        """
        fig, ax = plt.subplots(
            figsize=self.config["figure_size"], dpi=self.config["dpi"]
        )

        ax.text(
            0.5,
            0.5,
            title,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=self.config["font_size"],
            style="italic",
            alpha=0.7,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        return self.apply_template(fig)
