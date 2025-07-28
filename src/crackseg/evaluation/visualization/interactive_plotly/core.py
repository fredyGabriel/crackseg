"""Core interactive Plotly visualizer implementation."""

from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..templates.base_template import BaseVisualizationTemplate
from .export_handlers import ExportHandler
from .metadata_handlers import MetadataHandler


class InteractivePlotlyVisualizer:
    """Interactive visualization using Plotly with multi-format export
    capabilities."""

    def __init__(
        self,
        template: BaseVisualizationTemplate | None = None,
        responsive: bool = True,
        export_formats: list[str] | None = None,
    ) -> None:
        """Initialize the interactive Plotly visualizer.

        Args:
            template: Optional template for consistent styling.
            responsive: Whether to make plots responsive.
            export_formats: List of export formats (html, png, pdf, svg, jpg,
                json).
        """
        self.template = template
        self.responsive = responsive
        self.export_handler = ExportHandler(
            export_formats or ["html", "png", "pdf", "svg"]
        )
        self.metadata_handler = MetadataHandler()

    def create_interactive_training_curves(
        self,
        metrics_data: dict[str, list[float]],
        epochs: list[int],
        title: str = "Training Curves",
        save_path: Path | None = None,
    ) -> go.Figure:
        """Create interactive training curves.

        Args:
            metrics_data: Dictionary of metric names to values.
            epochs: List of epoch numbers.
            title: Plot title.
            save_path: Optional path to save the plot.

        Returns:
            Interactive Plotly figure.
        """
        fig = go.Figure()

        for metric_name, values in metrics_data.items():
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    mode="lines+markers",
                    name=metric_name,
                    line={"width": 2},
                    marker={"size": 6},
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="Value",
            hovermode="x unified",
            template="plotly_white",
        )

        if self.responsive:
            fig.update_layout(
                autosize=True,
                margin={"l": 50, "r": 50, "t": 50, "b": 50},
            )

        # Apply template styling if available
        if self.template:
            fig = self.template.apply_template(fig)

        # Save if path provided
        if save_path:
            self.export_handler.save_plot(fig, save_path)

        return fig

    def create_interactive_prediction_grid(
        self,
        results: list[dict[str, Any]],
        max_images: int = 9,
        show_metrics: bool = True,
        show_confidence: bool = True,
        save_path: Path | None = None,
    ) -> go.Figure:
        """Create interactive prediction comparison grid.

        Args:
            results: List of prediction results.
            max_images: Maximum number of images to display.
            show_metrics: Whether to show metrics.
            show_confidence: Whether to show confidence maps.
            save_path: Optional path to save the plot.

        Returns:
            Interactive Plotly figure.
        """
        # Limit number of images
        results = results[:max_images]
        n_images = len(results)

        if n_images == 0:
            return go.Figure()

        # Calculate grid dimensions
        n_cols = min(3, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"Image {i + 1}" for i in range(n_images)],
            specs=[
                [{"secondary_y": False} for _ in range(n_cols)]
                for _ in range(n_rows)
            ],
        )

        for idx, result in enumerate(results):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            # Add image
            if "image" in result:
                fig.add_trace(
                    go.Image(z=result["image"], name=f"Image {idx + 1}"),
                    row=row,
                    col=col,
                )

            # Add prediction if available
            if "prediction" in result:
                fig.add_trace(
                    go.Image(
                        z=result["prediction"], name=f"Prediction {idx + 1}"
                    ),
                    row=row,
                    col=col,
                )

        fig.update_layout(
            title="Prediction Comparison Grid",
            template="plotly_white",
            height=300 * n_rows,
            width=400 * n_cols,
        )

        if self.responsive:
            fig.update_layout(autosize=True)

        # Apply template styling if available
        if self.template:
            fig = self.template.apply_template(fig)

        # Save if path provided
        if save_path:
            self.export_handler.save_plot(fig, save_path)

        return fig

    def create_interactive_confidence_map(
        self,
        confidence_data: dict[str, Any],
        save_path: Path | None = None,
    ) -> go.Figure:
        """Create interactive confidence map visualization.

        Args:
            confidence_data: Confidence map data.
            save_path: Optional path to save the plot.

        Returns:
            Interactive Plotly figure.
        """
        fig = go.Figure()

        # Add confidence heatmap
        if "confidence_map" in confidence_data:
            fig.add_trace(
                go.Heatmap(
                    z=confidence_data["confidence_map"],
                    colorscale="Viridis",
                    name="Confidence",
                    showscale=True,
                )
            )

        fig.update_layout(
            title="Confidence Map",
            template="plotly_white",
        )

        if self.responsive:
            fig.update_layout(autosize=True)

        # Apply template styling if available
        if self.template:
            fig = self.template.apply_template(fig)

        # Save if path provided
        if save_path:
            self.export_handler.save_plot(fig, save_path)

        return fig
