"""Prediction analysis visualization.

This module provides functionality for creating professional
visualizations of prediction results.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class PredictionAnalyzer:
    """Create professional visualizations of prediction results."""

    def __init__(self, config: Any) -> None:
        """Initialize the prediction analyzer.

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
        """Create professional visualization of prediction results.

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
            plt.colorbar(im, ax=axes[col_idx], fraction=0.046, pad=0.04)

        # Add metrics text if requested
        if show_metrics:
            self._add_metrics_text(fig, analysis_result)

        plt.tight_layout()

        # Save if requested
        if save_path:
            self._save_visualization(fig, save_path)

        # Convert to numpy array
        return self._fig_to_array(fig)

    def _load_original_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess original image.

        Args:
            image_path: Path to the original image

        Returns:
            Loaded image as numpy array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _add_metrics_text(
        self, fig: Figure, analysis_result: dict[str, Any]
    ) -> None:
        """Add metrics text to the figure.

        Args:
            fig: Figure to add text to
            analysis_result: Analysis result with metrics
        """
        # Implementation would go here
        # This is a placeholder for the actual metrics text logic
        pass

    def _save_visualization(self, fig: Figure, save_path: str | Path) -> None:
        """Save visualization to file.

        Args:
            fig: Figure to save
            save_path: Path to save the figure
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Visualization saved to: {save_path}")

    def _fig_to_array(self, fig: Figure) -> np.ndarray:
        """Convert matplotlib figure to numpy array.

        Args:
            fig: Matplotlib figure

        Returns:
            Figure as numpy array
        """
        # Implementation would go here
        # This is a placeholder for the actual conversion logic
        return np.array([])
