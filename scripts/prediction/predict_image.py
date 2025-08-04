#!/usr/bin/env python3
"""
Simple image prediction script.

Usage:
    python scripts/predict_image.py --image path/to/image.jpg \
        --mask-dir path/to/masks
    python scripts/predict_image.py --image data/unified/images/98.jpg \
        --mask-dir data/unified/masks
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf


def load_model(checkpoint_path, config_path):
    """Load trained model."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = OmegaConf.load(config_path)

    from crackseg.model.factory.config import create_model_from_config

    model = create_model_from_config(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, target_size)

    # Convert to float32 and normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0

    # Apply ImageNet normalization (same as training)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_normalized = (image_normalized - mean) / std

    # Convert to tensor and add batch dimension
    image_tensor = (
        torch.from_numpy(image_normalized)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )

    return image_tensor, image_resized


def find_mask(image_path, mask_dir):
    """Find corresponding mask file."""
    if not mask_dir:
        return None

    image_name = Path(image_path).stem
    for ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
        mask_path = Path(mask_dir) / f"{image_name}{ext}"
        if mask_path.exists():
            return mask_path
    return None


def load_mask(mask_path, target_size=(256, 256)):
    """Load and preprocess mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")

    mask_resized = cv2.resize(mask, target_size)
    return (mask_resized > 127).astype(np.uint8)


def predict_and_visualize(
    image_path, mask_dir=None, output_path="prediction_result.png"
):
    """Make prediction and create visualization."""
    # Configuration
    checkpoint_path = "artifacts/experiments/checkpoints/model_best.pth.tar"
    config_path = "artifacts/experiments/configurations/default_experiment/config_epoch_0100.yaml"

    print("Loading model...")
    model = load_model(checkpoint_path, config_path)

    print(f"Processing image: {image_path}")
    image_tensor, image_resized = preprocess_image(image_path)

    # Make prediction
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)

        # Debug information
        print(
            f"Logits range: [{logits.min().item():.3f}, "
            f"{logits.max().item():.3f}]"
        )
        print(
            f"Probabilities range: [{probabilities.min().item():.3f}, "
            f"{probabilities.max().item():.3f}]"
        )

        # Try different thresholds
        prediction_05 = (probabilities > 0.5).float()
        prediction_01 = (probabilities > 0.1).float()
        prediction_001 = (probabilities > 0.01).float()
        prediction_06 = (probabilities > 0.6).float()  # More strict threshold
        prediction_07 = (probabilities > 0.7).float()  # Even more strict

        print(f"Pixels > 0.5: {prediction_05.sum().item()}")
        print(f"Pixels > 0.1: {prediction_01.sum().item()}")
        print(f"Pixels > 0.01: {prediction_001.sum().item()}")
        print(f"Pixels > 0.6: {prediction_06.sum().item()}")
        print(f"Pixels > 0.7: {prediction_07.sum().item()}")

    prob_mask = probabilities.squeeze().cpu().numpy()
    pred_mask = (
        prediction_06.squeeze().cpu().numpy()
    )  # Use 0.6 threshold for more realistic predictions

    # Find and load ground truth mask
    mask_path = find_mask(image_path, mask_dir)
    gt_mask = None
    if mask_path:
        print(f"Found ground truth mask: {mask_path}")
        gt_mask = load_mask(mask_path)
    else:
        print("No ground truth mask found")

    # Create visualization
    num_plots = 4 if gt_mask is not None else 3
    _, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    # Original image
    axes[0].imshow(image_resized)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Probabilities
    im1 = axes[1].imshow(prob_mask, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Probabilities")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Binary prediction
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Binary Prediction (threshold=0.6)")
    axes[2].axis("off")

    # Ground truth (if available)
    if gt_mask is not None:
        axes[3].imshow(gt_mask, cmap="gray")
        axes[3].set_title("Ground Truth")
        axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Result saved as: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Make predictions on images")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--mask-dir", help="Directory containing ground truth masks"
    )
    parser.add_argument(
        "--output", default="prediction_result.png", help="Output image path"
    )

    args = parser.parse_args()

    predict_and_visualize(args.image, args.mask_dir, args.output)


if __name__ == "__main__":
    main()
