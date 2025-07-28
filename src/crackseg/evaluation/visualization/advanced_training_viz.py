"""Advanced training visualization system for crack segmentation.

This module provides comprehensive training visualization capabilities
including training curves with multiple metrics, learning rate schedule
analysis, gradient flow visualization, and parameter distributions.

This visualizer integrates with the ArtifactManager system for proper
artifact tracking and storage.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from plotly.graph_objects import Figure as PlotlyFigure

from crackseg.utils.artifact_manager import ArtifactManager

from .learning_rate_analysis import LearningRateAnalyzer
from .parameter_analysis import ParameterAnalyzer
from .templates import TrainingVisualizationTemplate
from .training_curves import TrainingCurvesVisualizer

logger = logging.getLogger(__name__)


class AdvancedTrainingVisualizer:
    """Advanced training visualization with comprehensive analysis
    capabilities."""

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
            artifact_manager: Optional ArtifactManager for saving
                visualizations
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
            # Apply template styling before saving
            fig = self.template.apply_template(fig)

            # Save figure to temporary file first
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as tmp_file:
                if isinstance(fig, Figure):
                    fig.savefig(tmp_file.name, dpi=self.template.config["dpi"])
                else:  # PlotlyFigure
                    fig.write_image(tmp_file.name)

                # Save using ArtifactManager
                artifact_id, metadata = (
                    self.artifact_manager.storage.save_visualization(
                        tmp_file.name, filename, description
                    )
                )

                # Cleanup temporary file
                import os

                os.unlink(tmp_file.name)

            logger.info(f"Visualization saved as artifact: {artifact_id}")
            return artifact_id, metadata
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

        # Load metrics data
        metrics_file = experiment_dir / "metrics" / "metrics.jsonl"
        if metrics_file.exists():
            metrics_data = []
            with open(metrics_file) as f:
                for line in f:
                    metrics_data.append(json.loads(line.strip()))
            training_data["metrics"] = metrics_data

        # Load summary data
        summary_file = experiment_dir / "metrics" / "complete_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                training_data["summary"] = json.load(f)

        # Load configuration
        config_file = experiment_dir / "config.yaml"
        if config_file.exists():
            with open(config_file) as f:
                training_data["config"] = json.load(f)

        # Load gradient data if requested
        if include_gradients:
            gradient_file = experiment_dir / "metrics" / "gradients.jsonl"
            if gradient_file.exists():
                gradient_data = []
                with open(gradient_file) as f:
                    for line in f:
                        gradient_data.append(json.loads(line.strip()))
                training_data["gradients"] = gradient_data

        return training_data

    def create_training_curves(
        self,
        training_data: dict[str, Any],
        metrics: list[str] | None = None,
        save_path: Path | None = None,
        interactive: bool | None = None,
    ) -> Figure | PlotlyFigure:
        """Create training curves visualization.

        Args:
            training_data: Training data containing metrics
            metrics: List of metrics to plot (auto-detected if None)
            save_path: Path to save the visualization
            interactive: Whether to create interactive plot

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        interactive = (
            interactive if interactive is not None else self.interactive
        )
        fig = self.training_curves_viz.create_training_curves(
            training_data, metrics, save_path, interactive
        )

        if save_path and self.artifact_manager:
            self._save_visualization_with_artifacts(
                fig,
                f"training_curves_{save_path.stem}",
                "Training curves visualization",
            )

        return fig

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
        fig = self.lr_analyzer.analyze_learning_rate_schedule(
            training_data, save_path
        )

        if save_path and self.artifact_manager:
            self._save_visualization_with_artifacts(
                fig,
                f"learning_rate_analysis_{save_path.stem}",
                "Learning rate schedule analysis",
            )

        return fig

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
        fig = self.param_analyzer.visualize_parameter_distributions(
            model_path, save_path
        )

        if save_path and self.artifact_manager:
            self._save_visualization_with_artifacts(
                fig,
                f"parameter_distributions_{save_path.stem}",
                "Parameter distribution analysis",
            )

        return fig

    def visualize_gradient_flow(
        self, gradient_data: dict[str, Any], save_path: Path | None = None
    ) -> Figure | PlotlyFigure:
        """Visualize gradient flow during training.

        Args:
            gradient_data: Dictionary containing gradient information
            save_path: Path to save the visualization

        Returns:
            Matplotlib Figure or Plotly Figure
        """
        if not gradient_data or not gradient_data.get("gradients"):
            logger.warning("No gradient data available")
            return self.template.create_empty_plot("Gradient Flow")

        # Create gradient flow visualization using template
        fig = self.template.create_empty_plot("Gradient Flow")
        ax = fig.axes[0]  # Get the single axis

        gradients = gradient_data["gradients"]
        epochs = list(range(len(gradients)))

        # Plot gradient norms over epochs
        gradient_norms = []
        for epoch_grads in gradients:
            if isinstance(epoch_grads, dict):
                # Handle case where epoch_grads is a single gradient dict
                epoch_norm = sum(
                    float(val)
                    for val in epoch_grads.values()
                    if isinstance(val, int | float)
                    and val != epoch_grads.get("epoch", 0)
                )
                gradient_norms.append(epoch_norm)
            else:
                # Handle case where epoch_grads is a list of gradient dicts
                epoch_norm = sum(grad.get("norm", 0.0) for grad in epoch_grads)
                gradient_norms.append(epoch_norm)

        # Clear the empty plot and create actual visualization
        ax.clear()
        ax.plot(epochs, gradient_norms)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Flow During Training")

        # Apply template styling
        fig = self.template.apply_template(fig)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    def _extract_parameter_statistics(
        self, model_state: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """Extract statistical information from model parameters.

        Args:
            model_state: Model state dictionary

        Returns:
            Dictionary mapping parameter names to statistics
        """
        return self.param_analyzer._extract_parameter_statistics(model_state)

    def _create_empty_plot(self, title: str) -> Figure:
        """Create an empty plot with informative message.

        Args:
            title: Title for the empty plot

        Returns:
            Matplotlib Figure with empty plot
        """
        return self.template.create_empty_plot(title)

    def create_comprehensive_report(
        self,
        experiment_dir: Path,
        output_dir: Path,
        include_gradients: bool = False,
    ) -> dict[str, Path]:
        """Create a comprehensive training visualization report.

        Args:
            experiment_dir: Path to experiment directory
            output_dir: Path to save visualizations
            include_gradients: Whether to include gradient analysis

        Returns:
            Dictionary mapping visualization names to file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load training data
        training_data = self.load_training_data(
            experiment_dir, include_gradients=include_gradients
        )

        report_files = {}

        try:
            # Training curves
            curves_path = output_dir / "training_curves.png"
            _ = self.create_training_curves(
                training_data, save_path=curves_path
            )
            report_files["training_curves"] = curves_path

            # Learning rate analysis
            lr_path = output_dir / "learning_rate_analysis.png"
            _ = self.analyze_learning_rate_schedule(
                training_data, save_path=lr_path
            )
            report_files["learning_rate_analysis"] = lr_path

            # Parameter distributions (if model checkpoint available)
            checkpoint_path = experiment_dir / "model_best.pth"
            if checkpoint_path.exists():
                param_path = output_dir / "parameter_distributions.png"
                self.visualize_parameter_distributions(
                    checkpoint_path, save_path=param_path
                )
                report_files["parameter_distributions"] = param_path

            logger.info(f"Comprehensive report created in: {output_dir}")
            return report_files

        except Exception as e:
            logger.error(f"Error creating comprehensive report: {e}")
            return report_files
