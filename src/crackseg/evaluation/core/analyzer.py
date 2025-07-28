"""Main prediction analyzer for crack segmentation evaluation."""

import logging
from pathlib import Path
from typing import Any

import torch

from ..metrics.calculator import MetricsCalculator
from .image_processor import ImageProcessor
from .model_loader import ModelLoader

logger = logging.getLogger(__name__)


class PredictionAnalyzer:
    """
    Main prediction analyzer for crack segmentation.

    Integrates model loading, image processing, and metrics calculation
    for comprehensive evaluation of crack segmentation models.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        device: str | None = None,
        mask_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize the prediction analyzer.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            config_path: Path to the model configuration (optional)
            device: Device to use for inference
                    ('cuda', 'cpu', or None for auto)
            mask_dir: Directory containing ground truth masks (optional)
        """
        self.device = self._setup_device(device)
        self.mask_dir = Path(mask_dir) if mask_dir else None

        # Initialize components
        self.model_loader = ModelLoader(checkpoint_path)
        self.config = self.model_loader.load_config(config_path)
        self.model = self.model_loader.create_model(self.config)  # type: ignore
        self.image_processor = ImageProcessor(self.config)  # type: ignore
        self.metrics_calculator = MetricsCalculator()

        # Setup model for inference
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"PredictionAnalyzer initialized with model: {checkpoint_path}"
        )
        logger.info(f"Device: {self.device}")
        if self.mask_dir:
            logger.info(f"Mask directory: {self.mask_dir}")

    def _setup_device(self, device: str | None) -> torch.device:
        """Setup the device for inference."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available, falling back to CPU"
            )
            device = "cpu"

        return torch.device(device)

    def predict_single_image(
        self,
        image_path: str | Path,
        threshold: float = 0.5,
        return_confidence: bool = False,
    ) -> dict[str, Any]:
        """
        Predict segmentation for a single image.

        Args:
            image_path: Path to the input image
            threshold: Threshold for binary segmentation
            return_confidence: Whether to return confidence scores

        Returns:
            Dictionary containing prediction results
        """
        # Load and preprocess image
        image_tensor = self.image_processor.load_and_preprocess_image(
            image_path
        )

        # Make prediction
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()

        # Convert to numpy for visualization
        pred_mask = predictions.squeeze().cpu().numpy()
        prob_mask = probabilities.squeeze().cpu().numpy()

        result = {
            "image_path": str(image_path),
            "prediction_mask": pred_mask,
            "probability_mask": prob_mask,
            "threshold": threshold,
            "prediction_shape": pred_mask.shape,
        }

        if return_confidence:
            result["confidence"] = float(prob_mask.max())

        return result

    def analyze_with_ground_truth(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Analyze prediction against ground truth.

        Args:
            image_path: Path to the input image
            mask_path: Path to the ground truth mask
            threshold: Threshold for binary segmentation

        Returns:
            Dictionary containing analysis results
        """
        # Get prediction
        prediction_result = self.predict_single_image(image_path, threshold)

        # Load ground truth
        gt_mask = self.image_processor.load_mask(mask_path)

        # Calculate metrics
        pred_tensor = (
            torch.from_numpy(prediction_result["prediction_mask"])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0)

        metrics = self.metrics_calculator.calculate_single_batch(
            pred_tensor, gt_tensor
        )

        # Calculate IoU
        iou = self.metrics_calculator.calculate_iou(pred_tensor, gt_tensor)

        result = {
            **prediction_result,
            "ground_truth_mask": gt_mask,
            "metrics": metrics,
            "iou": iou,
        }

        return result

    def analyze_image(
        self,
        image_path: str | Path,
        mask_path: str | Path | None = None,
        threshold: float = 0.5,
        auto_find_mask: bool = True,
    ) -> dict[str, Any]:
        """
        Analyze an image with optional ground truth comparison.

        Args:
            image_path: Path to the input image
            mask_path: Path to the ground truth mask (optional)
            threshold: Threshold for binary segmentation
            auto_find_mask: Whether to automatically find mask if not provided

        Returns:
            Dictionary containing analysis results
        """
        # Auto-infer mask path if not provided
        if mask_path is None and auto_find_mask and self.mask_dir:
            inferred_mask_path = self.image_processor.infer_mask_path(
                image_path, self.mask_dir
            )
            if inferred_mask_path:
                mask_path = inferred_mask_path
                logger.info(f"Using auto-inferred mask: {mask_path}")
            else:
                logger.info(
                    "No mask found, performing prediction-only analysis"
                )

        # Perform analysis based on whether we have a mask
        if mask_path:
            return self.analyze_with_ground_truth(
                image_path, mask_path, threshold
            )
        else:
            return self.predict_single_image(image_path, threshold)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return self.model_loader.get_model_info(self.config)  # type: ignore
