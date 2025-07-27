#!/usr/bin/env python3
"""
Prediction Analyzer for Crack Segmentation.

This module provides comprehensive tools for analyzing model predictions
on crack segmentation tasks, including visualization, metrics computation,
and batch processing capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from crackseg.training.metrics import CrackMetrics
from crackseg.utils.checkpointing.core import load_checkpoint

logger = logging.getLogger(__name__)


class PredictionAnalyzer:
    """
    Comprehensive analyzer for crack segmentation predictions.

    Provides functionality for:
    - Single image prediction and visualization
    - Batch processing with metrics
    - Professional visualization generation
    - Comparison with ground truth
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initialize the prediction analyzer.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            config_path: Path to the model configuration (optional)
            device: Device to use for inference ('cuda', 'cpu', or None for
                auto)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path) if config_path else None
        self.device = self._setup_device(device)

        # Load model and configuration
        self.model, self.config = self._load_model_and_config()
        self.model.to(self.device)
        self.model.eval()

        # Initialize metrics
        self.metrics = CrackMetrics(num_classes=1)

        logger.info(
            f"PredictionAnalyzer initialized with model: "
            f"{self.checkpoint_path}"
        )
        logger.info(f"Device: {self.device}")
        logger.info(f"Model type: {self.config.model._target_}")

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
        # Load checkpoint
        checkpoint = load_checkpoint(self.checkpoint_path)

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
            model = model_class(
                **{k: v for k, v in model_config.items() if k != "_target_"}
            )
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
            result["confidence"] = prob_mask.max()

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

        self.metrics.reset()
        self.metrics.update(pred_tensor, gt_tensor)
        metrics = self.metrics.compute()

        # Calculate additional metrics
        intersection = (pred_tensor * gt_tensor).sum()
        union = pred_tensor.sum() + gt_tensor.sum() - intersection
        iou = intersection / (union + 1e-8)

        result = {
            **prediction_result,
            "ground_truth_mask": gt_mask,
            "metrics": metrics,
            "iou": iou.item(),
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
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

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
                f"IoU: {metrics['iou']:.3f}\nF1: {metrics['f1']:.3f}\n"
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

        # Convert to numpy array
        fig.canvas.draw()
        visualization = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype=np.uint8
        )
        visualization = visualization.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )

        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Visualization saved to: {save_path}")

        plt.close()

        return visualization

    def batch_analysis(
        self,
        image_dir: str | Path,
        mask_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        threshold: float = 0.5,
        save_visualizations: bool = True,
    ) -> dict[str, Any]:
        """
        Perform batch analysis on multiple images.

        Args:
            image_dir: Directory containing input images
            mask_dir: Directory containing ground truth masks (optional)
            output_dir: Directory to save results (optional)
            threshold: Threshold for binary segmentation
            save_visualizations: Whether to save individual visualizations

        Returns:
            Dictionary containing batch analysis results
        """
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir) if mask_dir else None
        output_dir = Path(output_dir) if output_dir else None

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        # Find image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [
            f
            for f in image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")

        logger.info(f"Found {len(image_files)} images for batch analysis")

        # Process each image
        results = []
        all_metrics = []

        for i, image_file in enumerate(image_files):
            logger.info(
                f"Processing {i + 1}/{len(image_files)}: {image_file.name}"
            )

            try:
                # Find corresponding mask if available
                mask_file = None
                if mask_dir and mask_dir.exists():
                    # Try different mask extensions
                    for ext in [".png", ".jpg", ".jpeg"]:
                        potential_mask = mask_dir / f"{image_file.stem}{ext}"
                        if potential_mask.exists():
                            mask_file = potential_mask
                            break

                # Analyze image
                if mask_file:
                    analysis_result = self.analyze_with_ground_truth(
                        image_file, mask_file, threshold
                    )
                    all_metrics.append(analysis_result["metrics"])
                else:
                    analysis_result = self.predict_single_image(
                        image_file, threshold
                    )

                results.append(analysis_result)

                # Save visualization if requested
                if save_visualizations and output_dir:
                    vis_path = output_dir / f"{image_file.stem}_analysis.png"
                    self.create_visualization(analysis_result, vis_path)

            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                continue

        # Calculate aggregate metrics
        batch_summary = {
            "total_images": len(image_files),
            "processed_images": len(results),
            "failed_images": len(image_files) - len(results),
        }

        if all_metrics:
            # Calculate average metrics
            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])

            batch_summary["average_metrics"] = avg_metrics
            batch_summary["all_metrics"] = all_metrics

        # Save batch results
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save summary
            summary_path = output_dir / "batch_analysis_summary.json"
            with open(summary_path, "w") as f:
                json.dump(batch_summary, f, indent=2, default=str)

            # Save detailed results
            results_path = output_dir / "detailed_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

        return batch_summary

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


def main():
    """Command-line interface for the prediction analyzer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Crack Segmentation Prediction Analyzer"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--config", help="Path to model configuration")
    parser.add_argument("--image", help="Path to single image for analysis")
    parser.add_argument("--mask", help="Path to ground truth mask")
    parser.add_argument(
        "--image-dir", help="Directory of images for batch analysis"
    )
    parser.add_argument("--mask-dir", help="Directory of ground truth masks")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Segmentation threshold"
    )
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization generation"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = PredictionAnalyzer(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )

    # Print model info
    model_info = analyzer.get_model_info()
    print("Model Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print()

    # Single image analysis
    if args.image:
        if args.mask:
            result = analyzer.analyze_with_ground_truth(
                args.image, args.mask, args.threshold
            )
            print("Analysis Results:")
            print(f"  IoU: {result['iou']:.3f}")
            for metric, value in result["metrics"].items():
                print(f"  {metric}: {value:.3f}")
        else:
            result = analyzer.predict_single_image(args.image, args.threshold)
            print("Prediction completed")

        # Create visualization
        if not args.no_viz:
            output_path = (
                args.output_dir / f"{Path(args.image).stem}_analysis.png"
                if args.output_dir
                else None
            )
            analyzer.create_visualization(result, output_path)

    # Batch analysis
    elif args.image_dir:
        if not args.output_dir:
            args.output_dir = Path("prediction_analysis_results")

        summary = analyzer.batch_analysis(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            save_visualizations=not args.no_viz,
        )

        print("Batch Analysis Summary:")
        for key, value in summary.items():
            if key != "all_metrics":
                print(f"  {key}: {value}")

        if "average_metrics" in summary:
            print("\nAverage Metrics:")
            for metric, value in summary["average_metrics"].items():
                print(f"  {metric}: {value:.3f}")


if __name__ == "__main__":
    main()
