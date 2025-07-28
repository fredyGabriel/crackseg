"""Advanced prediction visualization demo.

This script demonstrates the capabilities of the AdvancedPredictionVisualizer
including comparison grids, confidence maps, error analysis, and
segmentation overlays with configurable templates.
"""

import logging
from pathlib import Path

import numpy as np

from crackseg.evaluation.visualization import AdvancedPredictionVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_prediction_data() -> list[dict]:
    """Create sample prediction data for demonstration.

    Returns:
        List of sample prediction results.
    """
    # Create sample images and masks
    sample_data = []

    for i in range(6):
        # Create synthetic image (simulating pavement)
        image = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)

        # Create synthetic ground truth mask (crack pattern)
        gt_mask = np.zeros((256, 256), dtype=bool)
        # Add some crack-like patterns
        gt_mask[100:120, 50:200] = True  # Horizontal crack
        gt_mask[150:170, 100:150] = True  # Another crack
        gt_mask[80:100, 80:120] = True  # Small crack

        # Create prediction mask (with some errors)
        pred_mask = gt_mask.copy()
        # Add some false positives
        pred_mask[200:220, 50:100] = True
        # Remove some true positives (false negatives)
        pred_mask[80:100, 80:120] = False

        # Create probability mask
        prob_mask = np.random.random((256, 256))
        prob_mask[gt_mask] = np.random.uniform(0.7, 1.0, gt_mask.sum())
        prob_mask[pred_mask] = np.random.uniform(0.6, 1.0, pred_mask.sum())

        # Calculate metrics
        tp = (pred_mask & gt_mask).sum()
        fp = (pred_mask & ~gt_mask).sum()
        fn = (~pred_mask & gt_mask).sum()
        # tn = (~pred_mask & ~gt_mask).sum()  # Not used in metrics calculation

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Save sample image
        image_path = f"outputs/demo_prediction/sample_image_{i}.jpg"
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert to BGR for OpenCV
        image_bgr = image[:, :, ::-1]
        import cv2

        cv2.imwrite(image_path, image_bgr)

        sample_data.append(
            {
                "image_path": image_path,
                "prediction_mask": pred_mask,
                "ground_truth_mask": gt_mask,
                "probability_mask": prob_mask,
                "iou": iou,
                "dice": dice,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return sample_data


def demonstrate_comparison_grid(
    visualizer: AdvancedPredictionVisualizer,
) -> None:
    """Demonstrate comparison grid functionality.

    Args:
        visualizer: Advanced prediction visualizer instance.
    """
    logger.info("Creating comparison grid...")

    sample_data = create_sample_prediction_data()

    # Create comparison grid
    _ = visualizer.create_comparison_grid(
        results=sample_data,
        save_path="outputs/demo_prediction/comparison_grid.png",
        max_images=6,
        show_metrics=True,
        grid_layout=(2, 3),
    )

    logger.info("✅ Comparison grid created successfully")


def demonstrate_confidence_map(
    visualizer: AdvancedPredictionVisualizer,
) -> None:
    """Demonstrate confidence map functionality.

    Args:
        visualizer: Advanced prediction visualizer instance.
    """
    logger.info("Creating confidence map...")

    sample_data = create_sample_prediction_data()

    # Create confidence map for first sample
    _ = visualizer.create_confidence_map(
        result=sample_data[0],
        save_path="outputs/demo_prediction/confidence_map.png",
        show_original=True,
        show_contours=True,
    )

    logger.info("✅ Confidence map created successfully")


def demonstrate_error_analysis(
    visualizer: AdvancedPredictionVisualizer,
) -> None:
    """Demonstrate error analysis functionality.

    Args:
        visualizer: Advanced prediction visualizer instance.
    """
    logger.info("Creating error analysis...")

    sample_data = create_sample_prediction_data()

    # Create error analysis for first sample
    _ = visualizer.create_error_analysis(
        result=sample_data[0],
        save_path="outputs/demo_prediction/error_analysis.png",
    )

    logger.info("✅ Error analysis created successfully")


def demonstrate_segmentation_overlay(
    visualizer: AdvancedPredictionVisualizer,
) -> None:
    """Demonstrate segmentation overlay functionality.

    Args:
        visualizer: Advanced prediction visualizer instance.
    """
    logger.info("Creating segmentation overlay...")

    sample_data = create_sample_prediction_data()

    # Create segmentation overlay for first sample
    _ = visualizer.create_segmentation_overlay(
        result=sample_data[0],
        save_path="outputs/demo_prediction/segmentation_overlay.png",
        show_confidence=True,
    )

    logger.info("✅ Segmentation overlay created successfully")


def demonstrate_tabular_comparison(
    visualizer: AdvancedPredictionVisualizer,
) -> None:
    """Demonstrate tabular comparison functionality.

    Args:
        visualizer: Advanced prediction visualizer instance.
    """
    logger.info("Creating tabular comparison...")

    sample_data = create_sample_prediction_data()

    # Create tabular comparison
    _ = visualizer.create_tabular_comparison(
        results=sample_data,
        save_path="outputs/demo_prediction/tabular_comparison.png",
    )

    logger.info("✅ Tabular comparison created successfully")


def demonstrate_template_customization(
    visualizer: AdvancedPredictionVisualizer,
) -> None:
    """Demonstrate template customization functionality.

    Args:
        visualizer: Advanced prediction visualizer instance.
    """
    logger.info("Demonstrating template customization...")

    # Customize template configuration
    custom_config = {
        "figure_size": [14, 10],
        "dpi": 300,
        "color_palette": "viridis",
        "grid_alpha": 0.3,
        "line_width": 2.0,
        "font_size": 12,
        "title_font_size": 18,
        "legend_font_size": 14,
        "comparison_grid": {
            "grid_layout": [2, 2],
            "image_size": [300, 300],
            "show_titles": True,
            "show_metrics": True,
            "border_width": 3,
            "border_color": "darkblue",
        },
        "confidence_map": {
            "colormap": "plasma",
            "show_colorbar": True,
            "colorbar_position": "right",
            "threshold_contours": True,
            "contour_levels": 15,
            "show_uncertainty": True,
        },
    }

    # Update template configuration
    visualizer.update_template_config(custom_config)

    sample_data = create_sample_prediction_data()

    # Create comparison grid with custom styling
    _ = visualizer.create_comparison_grid(
        results=sample_data[:4],
        save_path="outputs/demo_prediction/custom_styled_grid.png",
        max_images=4,
        grid_layout=(2, 2),
    )

    logger.info("✅ Custom styled comparison grid created successfully")


def main() -> None:
    """Main demonstration function."""
    logger.info("🚀 Starting Advanced Prediction Visualizer Demo")

    # Initialize visualizer with default template
    visualizer = AdvancedPredictionVisualizer()

    # Create output directory
    output_dir = Path("outputs/demo_prediction")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Demonstrate all functionalities
        demonstrate_comparison_grid(visualizer)
        demonstrate_confidence_map(visualizer)
        demonstrate_error_analysis(visualizer)
        demonstrate_segmentation_overlay(visualizer)
        demonstrate_tabular_comparison(visualizer)
        demonstrate_template_customization(visualizer)

        logger.info("🎉 All demonstrations completed successfully!")
        logger.info(f"📁 Output files saved to: {output_dir.absolute()}")

        # List generated files
        generated_files = list(output_dir.glob("*.png"))
        logger.info(
            f"📊 Generated {len(generated_files)} visualization files:"
        )
        for file in generated_files:
            logger.info(f"   - {file.name}")

    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
