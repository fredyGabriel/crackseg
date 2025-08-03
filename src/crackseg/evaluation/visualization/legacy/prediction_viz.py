"""Prediction visualization utilities for crack segmentation."""

import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class PredictionVisualizer:
    """Create professional visualizations of prediction results."""

    def __init__(self, config: Any) -> None:
        """
        Initialize the prediction visualizer.

        Args:
            config: Model configuration for image size
        """
        self.config = config
        self.target_size = self._get_target_size()

    def _get_target_size(self) -> tuple[int, int]:
        """Get target image size from config."""
        target_size = self.config.data.image_size
        if isinstance(target_size, list):
            return tuple(target_size)
        return target_size

    def create_visualization(
        self,
        analysis_result: dict[str, Any],
        save_path: str | Path | None = None,
        show_confidence: bool = True,
        show_metrics: bool = True,
    ) -> np.ndarray:
        """
        Create professional visualization of prediction results.

        Args:
            analysis_result: Result from prediction analysis
            save_path: Path to save the visualization (optional)
            show_confidence: Whether to show confidence heatmap
            show_metrics: Whether to show metrics (if available)

        Returns:
            Visualization image as numpy array
        """
        # Load original image
        image_path = analysis_result["image_path"]
        original_image = self._load_original_image(image_path)

        # Create subplots
        has_gt = "ground_truth_mask" in analysis_result
        num_cols = 3 if has_gt else 2
        if show_confidence:
            num_cols += 1

        fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
        if num_cols == 1:
            axes = [axes]

        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontweight="bold", fontsize=12)
        axes[0].axis("off")

        # Prediction mask
        pred_mask = analysis_result["prediction_mask"]
        axes[1].imshow(pred_mask, cmap="Reds", alpha=0.7)
        axes[1].imshow(original_image, alpha=0.3)
        axes[1].set_title("Prediction", fontweight="bold", fontsize=12)
        axes[1].axis("off")

        col_idx = 2

        # Ground truth (if available)
        if has_gt:
            gt_mask = analysis_result["ground_truth_mask"]
            axes[col_idx].imshow(gt_mask, cmap="Blues", alpha=0.7)
            axes[col_idx].imshow(original_image, alpha=0.3)
            axes[col_idx].set_title(
                "Ground Truth", fontweight="bold", fontsize=12
            )
            axes[col_idx].axis("off")
            col_idx += 1

        # Confidence heatmap
        if show_confidence:
            prob_mask = analysis_result["probability_mask"]
            im = axes[col_idx].imshow(prob_mask, cmap="viridis", alpha=0.8)
            axes[col_idx].imshow(original_image, alpha=0.2)
            axes[col_idx].set_title(
                "Confidence", fontweight="bold", fontsize=12
            )
            axes[col_idx].axis("off")

            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[col_idx], fraction=0.046, pad=0.04)
            cbar.set_label("Probability", fontsize=10)
            col_idx += 1

        # Add metrics text
        if show_metrics and "metrics" in analysis_result:
            self._add_metrics_text(fig, analysis_result)

        plt.tight_layout()

        # Save if requested
        if save_path:
            self._save_visualization(fig, save_path)

        # Convert to numpy array
        visualization = self._fig_to_array(fig)
        plt.close()

        return visualization

    def _load_original_image(self, image_path: str) -> np.ndarray:
        """Load and resize original image for visualization."""
        original_image = cv2.imread(image_path)
        if original_image is not None:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            # Create a placeholder if image can't be loaded
            original_image = np.zeros((*self.target_size, 3), dtype=np.uint8)

        # Resize to match prediction size
        original_image = cv2.resize(original_image, self.target_size)
        return original_image

    def _add_metrics_text(
        self, fig: Figure, analysis_result: dict[str, Any]
    ) -> None:
        """Add metrics text to the visualization."""
        metrics = analysis_result["metrics"]
        metrics_text = (
            f"IoU: {analysis_result.get('iou', 'N/A'):.3f}\n"
            f"F1: {metrics['f1']:.3f}\n"
            f"Precision: {metrics['precision']:.3f}\n"
            f"Recall: {metrics['recall']:.3f}"
        )

        # Add text box
        fig.text(
            0.02,
            0.98,
            metrics_text,
            transform=fig.transFigure,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    def _save_visualization(self, fig: Figure, save_path: str | Path) -> None:
        """Save visualization to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Visualization saved to: {save_path}")

    def _fig_to_array(self, fig: Figure) -> np.ndarray:
        """Convert matplotlib figure to numpy array."""
        fig.canvas.draw()

        # Use a more compatible approach
        try:
            buf = np.asarray(fig.canvas.buffer_rgba())  # type: ignore
            return buf[:, :, :3]  # Return RGB channels only
        except AttributeError:
            # Fallback for older matplotlib versions
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
            buf.shape = (h, w, 3)
            return buf

    def create_comparison_grid(
        self,
        results: list[dict[str, Any]],
        save_path: str | Path | None = None,
        max_images: int = 9,
    ) -> np.ndarray:
        """
        Create a grid comparison of multiple prediction results.

        Args:
            results: List of analysis results
            save_path: Path to save the visualization (optional)
            max_images: Maximum number of images to display

        Returns:
            Grid visualization as numpy array
        """
        # Limit number of images
        results = results[:max_images]
        n_results = len(results)

        if n_results == 0:
            raise ValueError("No results provided for comparison")

        # Calculate grid dimensions
        cols = min(3, n_results)
        rows = (n_results + cols - 1) // cols

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

            # Load original image
            original_image = self._load_original_image(result["image_path"])

            # Show prediction overlay
            pred_mask = result["prediction_mask"]
            ax.imshow(pred_mask, cmap="Reds", alpha=0.7)
            ax.imshow(original_image, alpha=0.3)

            # Add title with IoU if available
            title = Path(result["image_path"]).stem
            if "iou" in result:
                title += f" (IoU: {result['iou']:.3f})"
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        # Hide empty subplots
        for i in range(n_results, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis("off")

        plt.tight_layout()

        # Save if requested
        if save_path:
            self._save_visualization(fig, save_path)

        # Convert to numpy array
        visualization = self._fig_to_array(fig)
        plt.close()

        return visualization
