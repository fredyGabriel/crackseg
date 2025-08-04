"""Core training visualization functionality.

This module provides the core functionality for advanced training
visualization including data loading and basic visualization setup.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from plotly.graph_objects import Figure as PlotlyFigure

from crackseg.utils.artifact_manager import ArtifactManager

from ..legacy.learning_rate_analysis import LearningRateAnalyzer
from ..legacy.parameter_analysis import ParameterAnalyzer
from ..legacy.training_curves import TrainingCurvesVisualizer
from ..templates import TrainingVisualizationTemplate

logger = logging.getLogger(__name__)


class AdvancedTrainingVisualizer:
    """Advanced training visualization with comprehensive analysis capabilities."""

    def __init__(
        self,
        style_config: dict[str, Any] | None = None,
        interactive: bool = True,
        artifact_manager: ArtifactManager | None = None,
    ) -> None:
        """Initialize the advanced training visualizer.

        Args:
            style_config: Configuration for plot styling
            interactive: Whether to use interactive Plotly plots
            artifact_manager: Optional ArtifactManager for saving visualizations
        """
        self.interactive = interactive
        self.artifact_manager = artifact_manager

        # Maintain compatibility with existing tests
        self.style_config = style_config or self._get_default_style()

        # Use template system for consistent styling
        self.template = TrainingVisualizationTemplate(style_config)

        # Initialize component visualizers with style config
        self.training_curves_viz = TrainingCurvesVisualizer(self.style_config)
        self.lr_analyzer = LearningRateAnalyzer(self.style_config)
        self.param_analyzer = ParameterAnalyzer(self.style_config)

    def _get_default_style(self) -> dict[str, Any]:
        """Get default styling configuration."""
        return {
            "figure_size": (12, 8),
            "dpi": 300,
            "color_palette": "viridis",
            "grid_alpha": 0.3,
            "line_width": 2,
            "font_size": 12,
            "title_font_size": 14,
            "legend_font_size": 10,
        }

    def connect_artifact_manager(
        self, artifact_manager: ArtifactManager
    ) -> None:
        """Connect with ArtifactManager for visualization storage.

        Args:
            artifact_manager: ArtifactManager instance to use for saving
        """
        self.artifact_manager = artifact_manager
        logger.info(
            "AdvancedTrainingVisualizer connected with ArtifactManager"
        )

    def _save_visualization_with_artifacts(
        self,
        fig: Figure | PlotlyFigure,
        filename: str,
        description: str = "",
    ) -> tuple[str, Any] | None:
        """Save visualization using ArtifactManager if available.

        Args:
            fig: Figure to save
            filename: Name for the saved file
            description: Description for artifact metadata

        Returns:
            Tuple of (artifact_id, metadata) if saved, None otherwise
        """
        if self.artifact_manager is None:
            logger.warning(
                "No ArtifactManager connected, skipping artifact save"
            )
            return None

        try:
            # Save to temporary file first
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as tmp_file:
                if isinstance(fig, Figure):
                    fig.savefig(tmp_file.name, dpi=300, bbox_inches="tight")
                else:
                    fig.write_image(tmp_file.name)

                # Register with ArtifactManager
                artifact_id = self.artifact_manager.register_artifact(
                    file_path=Path(tmp_file.name),
                    artifact_type="visualization",
                    description=description,
                    metadata={"filename": filename},
                )

                logger.info(f"Visualization saved as artifact: {artifact_id}")
                return artifact_id, {"filename": filename}

        except Exception as e:
            logger.error(f"Failed to save visualization as artifact: {e}")
            return None

    def load_training_data(
        self, experiment_dir: Path, include_gradients: bool = False
    ) -> dict[str, Any]:
        """Load training data from experiment directory.

        Args:
            experiment_dir: Path to experiment directory
            include_gradients: Whether to include gradient data

        Returns:
            Dictionary containing training data
        """
        training_data = {}

        # Load metrics
        metrics_file = experiment_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                training_data["metrics"] = json.load(f)

        # Load learning rate data
        lr_file = experiment_dir / "learning_rate.json"
        if lr_file.exists():
            with open(lr_file) as f:
                training_data["learning_rate"] = json.load(f)

        # Load gradient data if requested
        if include_gradients:
            gradient_file = experiment_dir / "gradients.json"
            if gradient_file.exists():
                with open(gradient_file) as f:
                    training_data["gradients"] = json.load(f)

        logger.info(f"Loaded training data from: {experiment_dir}")
        return training_data

    def _create_empty_plot(self, title: str) -> Figure:
        """Create an empty plot with title.

        Args:
            title: Title for the plot

        Returns:
            Empty matplotlib figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.style_config["figure_size"])
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
