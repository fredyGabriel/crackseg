#!/usr/bin/env python3
"""
Advanced Prediction Visualization Demo

This script demonstrates advanced prediction visualization techniques
including confidence maps, error analysis, and interactive plots.
"""

import argparse
import random

# Ensure the project root is in the path
import sys
from pathlib import Path
from typing import list, tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crackseg.utils.visualization import (
    create_comparison_grid,
    create_confidence_map,
    create_custom_styled_grid,
    create_error_analysis,
    create_segmentation_overlay,
    create_tabular_comparison,
)


def generate_sample_predictions(
    num_samples: int = 5,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate sample prediction data for demonstration."""
    samples = []

    for _ in range(num_samples):
        # Generate random image (256x256x3)
        image = np.random.rand(256, 256, 3)

        # Generate ground truth mask
        gt_mask = np.zeros((256, 256), dtype=np.uint8)

        # Add some random "cracks" (thin lines)
        for _ in range(random.randint(3, 8)):
            start_x = random.randint(0, 255)
            start_y = random.randint(0, 255)
            length = random.randint(20, 80)
            angle = random.uniform(0, 2 * np.pi)

            for j in range(length):
                x = int(start_x + j * np.cos(angle))
                y = int(start_y + j * np.sin(angle))
                if 0 <= x < 256 and 0 <= y < 256:
                    gt_mask[y, x] = 1

        # Generate prediction mask with some noise
        pred_mask = gt_mask.copy()
        # Add some false positives
        noise_points = random.randint(10, 30)
        for _ in range(noise_points):
            x, y = random.randint(0, 255), random.randint(0, 255)
            pred_mask[y, x] = 1

        # Add some false negatives
        missing_points = random.randint(5, 15)
        for _ in range(missing_points):
            x, y = random.randint(0, 255), random.randint(0, 255)
            if gt_mask[y, x] == 1:
                pred_mask[y, x] = 0

        # Generate confidence map
        confidence = np.random.rand(256, 256) * 0.3 + 0.7  # 0.7-1.0 range
        confidence[gt_mask == 1] = (
            np.random.rand(int(np.sum(gt_mask))) * 0.4 + 0.6
        )  # 0.6-1.0 for GT

        samples.append((image, pred_mask, confidence))

    return samples


def create_demo_predictions():
    """Create and save demo prediction visualizations."""
    print("🎨 Creating Advanced Prediction Visualizations")
    print("=" * 50)

    # Generate sample data
    samples = generate_sample_predictions(5)

    # Create output directory
    output_dir = Path("artifacts/global/visualizations/demo_prediction")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Comparison Grid
    print("📊 Creating comparison grid...")
    images = [sample[0] for sample in samples]
    masks = [sample[1] for sample in samples]

    save_path = output_dir / "comparison_grid.png"
    create_comparison_grid(
        images=images,
        masks=masks,
        save_path=str(save_path),
        title="Demo Prediction Comparison",
    )
    print(f"✅ Saved: {save_path}")

    # 2. Confidence Map
    print("🎯 Creating confidence map...")
    image = samples[0][0]
    confidence = samples[0][2]

    save_path = output_dir / "confidence_map.png"
    create_confidence_map(
        image=image,
        confidence=confidence,
        save_path=str(save_path),
        title="Prediction Confidence Map",
    )
    print(f"✅ Saved: {save_path}")

    # 3. Error Analysis
    print("🔍 Creating error analysis...")
    image = samples[1][0]
    gt_mask = np.zeros_like(samples[1][1])  # Create dummy GT
    pred_mask = samples[1][1]

    # Add some ground truth for error analysis
    gt_mask[50:100, 50:150] = 1  # Some GT regions

    save_path = output_dir / "error_analysis.png"
    create_error_analysis(
        image=image,
        ground_truth=gt_mask,
        prediction=pred_mask,
        save_path=str(save_path),
        title="Error Analysis",
    )
    print(f"✅ Saved: {save_path}")

    # 4. Segmentation Overlay
    print("🎨 Creating segmentation overlay...")
    image = samples[2][0]
    mask = samples[2][1]

    save_path = output_dir / "segmentation_overlay.png"
    create_segmentation_overlay(
        image=image,
        mask=mask,
        save_path=str(save_path),
        title="Segmentation Overlay",
    )
    print(f"✅ Saved: {save_path}")

    # 5. Tabular Comparison
    print("📋 Creating tabular comparison...")
    metrics_data = {
        "Sample 1": {
            "IoU": 0.85,
            "Dice": 0.92,
            "Precision": 0.88,
            "Recall": 0.91,
        },
        "Sample 2": {
            "IoU": 0.78,
            "Dice": 0.87,
            "Precision": 0.82,
            "Recall": 0.89,
        },
        "Sample 3": {
            "IoU": 0.92,
            "Dice": 0.96,
            "Precision": 0.94,
            "Recall": 0.93,
        },
        "Sample 4": {
            "IoU": 0.81,
            "Dice": 0.89,
            "Precision": 0.85,
            "Recall": 0.87,
        },
        "Sample 5": {
            "IoU": 0.88,
            "Dice": 0.93,
            "Precision": 0.90,
            "Recall": 0.92,
        },
    }

    save_path = output_dir / "tabular_comparison.png"
    create_tabular_comparison(
        metrics_data=metrics_data,
        save_path=str(save_path),
        title="Performance Metrics Comparison",
    )
    print(f"✅ Saved: {save_path}")

    # 6. Custom Styled Grid
    print("🎨 Creating custom styled grid...")
    images = [sample[0] for sample in samples[:3]]
    masks = [sample[1] for sample in samples[:3]]

    save_path = output_dir / "custom_styled_grid.png"
    create_custom_styled_grid(
        images=images,
        masks=masks,
        save_path=str(save_path),
        title="Custom Styled Prediction Grid",
        style="dark_background",
    )
    print(f"✅ Saved: {save_path}")

    print(f"\n🎉 All visualizations saved to: {output_dir}")
    print("=" * 50)


def create_interactive_demo():
    """Create interactive visualization demo."""
    print("\n🔄 Creating Interactive Demo")
    print("=" * 30)

    # Generate sample data
    samples = generate_sample_predictions(3)

    # Create interactive plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Interactive Prediction Demo", fontsize=16)

    for i, (image, _mask, _confidence) in enumerate(samples):
        row = i // 3
        col = i % 3

        # Original image
        axes[row, col].imshow(image)
        axes[row, col].set_title(f"Sample {i + 1}")
        axes[row, col].axis("off")

        # Add click handler
        def on_click(event, sample_idx=i, row_idx=row, col_idx=col):
            if event.inaxes == axes[row_idx, col_idx]:
                print(f"Clicked on Sample {sample_idx + 1}")
                # Could add more interactive features here

        axes[row, col].figure.canvas.mpl_connect(
            "button_press_event", on_click
        )

    # Add colorbar
    im = axes[0, 0].imshow(np.random.rand(10, 10), alpha=0)
    cbar = plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.remove()  # Remove the dummy colorbar

    plt.tight_layout()

    # Save interactive version
    output_dir = Path("artifacts/global/visualizations/demo_prediction")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "interactive_demo.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Interactive demo saved: {save_path}")

    plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Advanced prediction visualization demo"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Create interactive demo"
    )
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of sample predictions"
    )

    args = parser.parse_args()

    if args.interactive:
        create_interactive_demo()
    else:
        create_demo_predictions()


if __name__ == "__main__":
    main()
