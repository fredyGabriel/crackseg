"""Advanced prediction visualization system.

This module provides comprehensive prediction visualization capabilities
for crack segmentation including comparison grids, confidence maps,
error analysis, and segmentation overlays with configurable templates.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from plotly.graph_objs import Figure as PlotlyFigure

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
        n_results = len(results)

        # Determine grid layout
        if grid_layout:
            rows, cols = grid_layout
            if rows * cols < n_results:
                raise ValueError(
                    f"Grid layout {grid_layout} too small for "
                    f"{n_results} results"
                )
        else:
            cols = min(3, n_results)
            rows = (n_results + cols - 1) // cols

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each result
        for i, result in enumerate(results):
            row = i // cols
            col = i % cols
            ax = axes[row, col]

            # Load and display original image
            original_image = self._load_original_image(result["image_path"])
            ax.imshow(original_image)

            # Add prediction overlay
            if "prediction_mask" in result:
                pred_mask = result["prediction_mask"]
                ax.imshow(pred_mask, cmap="Reds", alpha=0.7)

            # Add ground truth overlay if available
            if "ground_truth_mask" in result:
                gt_mask = result["ground_truth_mask"]
                ax.imshow(gt_mask, cmap="Blues", alpha=0.5)

            # Add title with metrics
            title = Path(result["image_path"]).stem
            if show_metrics and "iou" in result:
                title += f"\nIoU: {result['iou']:.3f}"
            if show_metrics and "dice" in result:
                title += f" Dice: {result['dice']:.3f}"
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

        # Hide empty subplots
        for i in range(n_results, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis("off")

        # Apply template styling
        fig = self.template.apply_template(fig)
        plt.tight_layout()

        # Save if requested
        if save_path:
            self._save_visualization(fig, save_path)

        return fig

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

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Original image
        original_image = self._load_original_image(result["image_path"])
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontweight="bold")
        axes[0].axis("off")

        # Confidence map
        prob_mask = result["probability_mask"]
        im = axes[1].imshow(prob_mask, cmap="viridis", alpha=0.8)
        if show_original:
            axes[1].imshow(original_image, alpha=0.3)
        axes[1].set_title("Confidence Map", fontweight="bold")
        axes[1].axis("off")

        # Add colorbar
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Add confidence contours
        if show_contours:
            contours = axes[1].contour(
                prob_mask, levels=10, colors="white", alpha=0.5, linewidths=0.5
            )
            axes[1].clabel(contours, inline=True, fontsize=8)

        # Apply template styling
        fig = self.template.apply_template(fig)
        plt.tight_layout()

        # Save if requested
        if save_path:
            self._save_visualization(fig, save_path)

        return fig

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

        gt_mask = result["ground_truth_mask"]
        pred_mask = result["prediction_mask"]

        # Calculate error types
        false_positives = pred_mask & ~gt_mask
        false_negatives = ~pred_mask & gt_mask
        true_positives = pred_mask & gt_mask

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Original image
        original_image = self._load_original_image(result["image_path"])
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original Image", fontweight="bold")
        axes[0, 0].axis("off")

        # Ground truth
        axes[0, 1].imshow(gt_mask, cmap="Blues", alpha=0.7)
        axes[0, 1].imshow(original_image, alpha=0.3)
        axes[0, 1].set_title("Ground Truth", fontweight="bold")
        axes[0, 1].axis("off")

        # Prediction
        axes[1, 0].imshow(pred_mask, cmap="Reds", alpha=0.7)
        axes[1, 0].imshow(original_image, alpha=0.3)
        axes[1, 0].set_title("Prediction", fontweight="bold")
        axes[1, 0].axis("off")

        # Error analysis
        error_map = np.zeros_like(gt_mask, dtype=np.uint8)
        error_map[false_positives] = 1  # Red
        error_map[false_negatives] = 2  # Blue
        error_map[true_positives] = 3  # Green

        colors = ["black", "red", "blue", "green"]
        cmap = plt.cm.colors.ListedColormap(colors)
        axes[1, 1].imshow(error_map, cmap=cmap, alpha=0.7)
        axes[1, 1].imshow(original_image, alpha=0.3)
        axes[1, 1].set_title("Error Analysis", fontweight="bold")
        axes[1, 1].axis("off")

        # Add legend
        legend_elements = [
            patches.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="red",
                alpha=0.7,
                label="False Positive",
            ),
            patches.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="blue",
                alpha=0.7,
                label="False Negative",
            ),
            patches.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="green",
                alpha=0.7,
                label="True Positive",
            ),
        ]
        axes[1, 1].legend(handles=legend_elements, loc="upper right")

        # Apply template styling
        fig = self.template.apply_template(fig)
        plt.tight_layout()

        # Save if requested
        if save_path:
            self._save_visualization(fig, save_path)

        return fig

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
        original_image = self._load_original_image(result["image_path"])

        # Determine number of subplots
        has_pred = "prediction_mask" in result
        has_gt = "ground_truth_mask" in result
        has_confidence = show_confidence and "probability_mask" in result
        num_plots = (
            1
            + (1 if has_pred else 0)
            + (1 if has_gt else 0)
            + (1 if has_confidence else 0)
        )

        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Original image
        axes[plot_idx].imshow(original_image)
        axes[plot_idx].set_title("Original Image", fontweight="bold")
        axes[plot_idx].axis("off")
        plot_idx += 1

        # Prediction overlay
        if "prediction_mask" in result:
            pred_mask = result["prediction_mask"]
            axes[plot_idx].imshow(pred_mask, cmap="Reds", alpha=0.7)
            axes[plot_idx].imshow(original_image, alpha=0.3)
            axes[plot_idx].set_title("Prediction Overlay", fontweight="bold")
            axes[plot_idx].axis("off")
            plot_idx += 1

        # Ground truth overlay
        if has_gt:
            gt_mask = result["ground_truth_mask"]
            axes[plot_idx].imshow(gt_mask, cmap="Blues", alpha=0.7)
            axes[plot_idx].imshow(original_image, alpha=0.3)
            axes[plot_idx].set_title("Ground Truth Overlay", fontweight="bold")
            axes[plot_idx].axis("off")
            plot_idx += 1

        # Confidence overlay
        if has_confidence:
            prob_mask = result["probability_mask"]
            im = axes[plot_idx].imshow(prob_mask, cmap="viridis", alpha=0.8)
            axes[plot_idx].imshow(original_image, alpha=0.2)
            axes[plot_idx].set_title("Confidence Overlay", fontweight="bold")
            axes[plot_idx].axis("off")
            plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)

        # Apply template styling
        fig = self.template.apply_template(fig)
        plt.tight_layout()

        # Save if requested
        if save_path:
            self._save_visualization(fig, save_path)

        return fig

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

        # Extract metrics
        metrics_data = []
        for result in results:
            row = {"Image": Path(result["image_path"]).stem}
            for metric in ["iou", "dice", "precision", "recall", "f1"]:
                if metric in result:
                    row[metric.upper()] = f"{result[metric]:.3f}"
            metrics_data.append(row)

        # Create table
        fig, ax = plt.subplots(figsize=(12, len(metrics_data) * 0.5 + 2))
        ax.axis("tight")
        ax.axis("off")

        # Create table
        table_data = []
        headers = list(metrics_data[0].keys())
        for row in metrics_data:
            table_data.append([row[header] for header in headers])

        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            bbox=Bbox.from_bounds(0, 0, 1, 1),
        )

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Color header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # Color alternating rows
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor("#f0f0f0")

        ax.set_title(
            "Prediction Metrics Comparison",
            fontweight="bold",
            fontsize=14,
            pad=20,
        )

        # Apply template styling
        fig = self.template.apply_template(fig)

        # Save if requested
        if save_path:
            self._save_visualization(fig, save_path)

        return fig

    def _load_original_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess original image.

        Args:
            image_path: Path to the original image.

        Returns:
            Loaded image as numpy array.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _save_visualization(
        self, fig: Figure | PlotlyFigure, save_path: str | Path
    ) -> None:
        """Save visualization to file.

        Args:
            fig: Matplotlib or Plotly figure to save.
            save_path: Path where to save the figure.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(fig, Figure):
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            fig.write_image(str(save_path))

        logger.info(f"Visualization saved to: {save_path}")

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
