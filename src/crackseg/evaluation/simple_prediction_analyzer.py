#!/usr/bin/env python3
"""
Simple Prediction Analyzer for Crack Segmentation.

A simplified version of the prediction analyzer that focuses on core
functionality
and avoids complex dependencies.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class SimplePredictionAnalyzer:
    """
    Simplified analyzer for crack segmentation predictions.

    Provides core functionality for:
    - Single image prediction and visualization
    - Basic metrics computation
    - Professional visualization generation
    - Automatic mask path inference
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
            device: Device to use for inference ('cuda', 'cpu', or None for
                auto)
            mask_dir: Directory containing ground truth masks (optional)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path) if config_path else None
        self.device = self._setup_device(device)
        self.mask_dir = Path(mask_dir) if mask_dir else None

        # Load model and configuration
        self.model, self.config = self._load_model_and_config()
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"SimplePredictionAnalyzer initialized with model: "
            f"{self.checkpoint_path}"
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

    def _load_model_and_config(self) -> tuple[torch.nn.Module, DictConfig]:
        """Load the trained model and configuration."""
        # Load checkpoint with weights_only=False for compatibility with
        # PyTorch 2.6+
        try:
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
        except Exception as e:
            logger.warning(
                f"Failed to load checkpoint with weights_only=False: {e}"
            )
            # Fallback to weights_only=True
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )

        # Load configuration
        if self.config_path and self.config_path.exists():
            config = OmegaConf.load(self.config_path)
        elif "config" in checkpoint:
            config = OmegaConf.create(checkpoint["config"])
        else:
            raise ValueError(
                "No configuration found in checkpoint or config_path"
            )

        # Create model using Hydra
        model_config = config.model
        if hasattr(model_config, "_target_"):
            import importlib

            module_path, class_name = model_config._target_.rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)

            # Filter out _target_ from config
            model_kwargs = {
                k: v for k, v in model_config.items() if k != "_target_"
            }
            model = model_class(**model_kwargs)
        else:
            raise ValueError("Model configuration must have _target_ field")

        # Load model weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(
                f"Loaded model weights from epoch "
                f"{checkpoint.get('epoch', 'unknown')}"
            )
        else:
            raise ValueError("No model_state_dict found in checkpoint")

        return model, config

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
        image = self._load_and_preprocess_image(image_path)

        # Make prediction
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(
                self.device
            )  # Add batch dimension
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

    def _load_and_preprocess_image(
        self, image_path: str | Path
    ) -> torch.Tensor:
        """Load and preprocess image for model input."""
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        target_size = self.config.data.image_size
        if isinstance(target_size, list):
            target_size = tuple(target_size)

        image = cv2.resize(image, target_size)

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Convert to tensor and normalize with ImageNet stats
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        return image_tensor

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
        gt_mask = self._load_mask(mask_path)

        # Calculate metrics
        pred_tensor = (
            torch.from_numpy(prediction_result["prediction_mask"])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0)

        metrics = self._calculate_metrics(pred_tensor, gt_tensor)

        # Calculate IoU
        intersection = (pred_tensor * gt_tensor).sum()
        union = pred_tensor.sum() + gt_tensor.sum() - intersection
        iou = intersection / (union + 1e-8)

        result = {
            **prediction_result,
            "ground_truth_mask": gt_mask,
            "metrics": metrics,
            "iou": float(iou.item()),
        }

        return result

    def _load_mask(self, mask_path: str | Path) -> np.ndarray:
        """Load and preprocess ground truth mask."""
        mask_path = Path(mask_path)

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")

        # Resize to match prediction size
        target_size = self.config.data.image_size
        if isinstance(target_size, list):
            target_size = tuple(target_size)

        mask = cv2.resize(mask, target_size)

        # Normalize to [0, 1]
        mask = mask.astype(np.float32) / 255.0

        # Binarize
        mask = (mask > 0.5).astype(np.float32)

        return mask

    def _calculate_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, float]:
        """Calculate basic segmentation metrics."""
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Calculate confusion matrix components
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()

        # Calculate metrics
        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        return {
            "precision": float(precision.item()),
            "recall": float(recall.item()),
            "f1": float(f1.item()),
            "accuracy": float(accuracy.item()),
        }

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
            analysis_result: Result from predict_single_image or
                analyze_with_ground_truth
            save_path: Path to save the visualization (optional)
            show_confidence: Whether to show confidence heatmap
            show_metrics: Whether to show metrics (if available)

        Returns:
            Visualization image as numpy array
        """
        # Load original image
        image_path = analysis_result["image_path"]
        original_image = cv2.imread(image_path)
        if original_image is not None:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            # Create a placeholder if image can't be loaded
            original_image = np.zeros((256, 256, 3), dtype=np.uint8)

        # Resize to match prediction size
        target_size = self.config.data.image_size
        if isinstance(target_size, list):
            target_size = tuple(target_size)

        original_image = cv2.resize(original_image, target_size)

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

        plt.tight_layout()

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Visualization saved to: {save_path}")

        # Convert to numpy array
        fig.canvas.draw()
        visualization = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype=np.uint8
        )
        visualization = visualization.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )

        plt.close()

        return visualization

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "model_type": self.config.model._target_,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "input_size": self.config.data.image_size,
            "num_classes": self.config.model.num_classes,
        }

    def _infer_mask_path(self, image_path: str | Path) -> Path | None:
        """
        Automatically infer the mask path based on the image path.

        This method looks for a mask file with the same name as the image
        but potentially different extension (e.g., .jpg -> .png) in the
        configured mask directory.

        Args:
            image_path: Path to the input image

        Returns:
            Path to the inferred mask file, or None if not found
        """
        if not self.mask_dir:
            return None

        image_path = Path(image_path)
        image_name = image_path.stem  # Get filename without extension

        # Common mask extensions to try
        mask_extensions = [".png", ".jpg", ".jpeg", ".tiff", ".tif"]

        for ext in mask_extensions:
            mask_path = self.mask_dir / f"{image_name}{ext}"
            if mask_path.exists():
                logger.info(f"Inferred mask path: {mask_path}")
                return mask_path

        logger.warning(
            f"No mask found for image {image_path.name} in {self.mask_dir}"
        )
        return None

    def analyze_image(
        self,
        image_path: str | Path,
        mask_path: str | Path | None = None,
        threshold: float = 0.5,
        auto_find_mask: bool = True,
    ) -> dict[str, Any]:
        """
        Analyze an image with optional ground truth comparison.

        This is the main method that automatically handles mask inference.
        If no mask_path is provided and auto_find_mask is True, it will
        attempt to find the corresponding mask automatically.

        Args:
            image_path: Path to the input image
            mask_path: Path to the ground truth mask (optional)
            threshold: Threshold for binary segmentation
            auto_find_mask: Whether to automatically find mask if not provided

        Returns:
            Dictionary containing analysis results
        """
        # Auto-infer mask path if not provided
        if mask_path is None and auto_find_mask:
            inferred_mask_path = self._infer_mask_path(image_path)
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


def main():
    """Command-line interface for the simple prediction analyzer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple Crack Segmentation Prediction Analyzer"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--config", help="Path to model configuration")
    parser.add_argument(
        "--image", required=True, help="Path to image for analysis"
    )
    parser.add_argument("--mask", help="Path to ground truth mask (optional)")
    parser.add_argument(
        "--mask-dir",
        help="Directory containing ground truth masks (for auto-inference)",
    )
    parser.add_argument("--output", help="Output path for visualization")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Segmentation threshold"
    )
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    parser.add_argument(
        "--no-auto-mask",
        action="store_true",
        help="Disable automatic mask inference",
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = SimplePredictionAnalyzer(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        mask_dir=args.mask_dir,
    )

    # Print model info
    model_info = analyzer.get_model_info()
    print("Model Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print()

    # Analyze image with automatic mask inference
    result = analyzer.analyze_image(
        image_path=args.image,
        mask_path=args.mask,
        threshold=args.threshold,
        auto_find_mask=not args.no_auto_mask,
    )

    # Print results
    if "metrics" in result:
        print("Analysis Results:")
        if "iou" in result:
            print(f"  IoU: {result['iou']:.3f}")
        for metric, value in result["metrics"].items():
            print(f"  {metric}: {value:.3f}")
    else:
        print("Prediction completed (no ground truth available)")

    # Create visualization
    if args.output:
        analyzer.create_visualization(result, args.output)
        print(f"Visualization saved to: {args.output}")


if __name__ == "__main__":
    main()
