"""Interactive Plotly visualization demonstration script.

This script demonstrates the interactive Plotly visualization capabilities
for crack segmentation including training curves, prediction grids,
3D confidence maps, error analysis, and real-time dashboards.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from crackseg.evaluation.visualization import InteractivePlotlyVisualizer
from crackseg.evaluation.visualization.templates import (
    PredictionVisualizationTemplate,
    TrainingVisualizationTemplate,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_training_data() -> dict[str, Any]:
    """Create sample training data for demonstration.

    Returns:
        Dictionary with sample training metrics.
    """
    epochs = 50
    metrics = []

    for epoch in range(1, epochs + 1):
        # Simulate realistic training curves
        train_loss = 2.0 * np.exp(-epoch / 20) + 0.1 * np.random.random()
        val_loss = 1.8 * np.exp(-epoch / 25) + 0.15 * np.random.random()
        train_iou = (
            0.3 + 0.6 * (1 - np.exp(-epoch / 15)) + 0.02 * np.random.random()
        )
        val_iou = (
            0.25 + 0.55 * (1 - np.exp(-epoch / 18)) + 0.03 * np.random.random()
        )

        # Learning rate schedule
        lr = 1e-3 * (0.9 ** (epoch // 10))

        # Gradient norm
        grad_norm = 1.0 * np.exp(-epoch / 30) + 0.1 * np.random.random()

        metrics.append(
            {
                "loss": train_loss,
                "val_loss": val_loss,
                "iou": train_iou,
                "val_iou": val_iou,
                "learning_rate": lr,
                "gradient_norm": grad_norm,
            }
        )

    return {"metrics": metrics}


def create_sample_prediction_data() -> list[dict[str, Any]]:
    """Create sample prediction data for demonstration.

    Returns:
        List of prediction result dictionaries.
    """
    results = []

    for _ in range(6):
        # Create sample images and masks
        img_size = 256
        original_img = np.random.randint(
            0, 255, (img_size, img_size, 3), dtype=np.uint8
        )

        # Create synthetic prediction and ground truth masks
        pred_mask = np.zeros((img_size, img_size), dtype=bool)
        gt_mask = np.zeros((img_size, img_size), dtype=bool)

        # Add some random crack-like structures
        for _ in range(3):
            start_x = np.random.randint(0, img_size)
            start_y = np.random.randint(0, img_size)
            length = np.random.randint(20, 80)
            angle = np.random.uniform(0, 2 * np.pi)

            for t in range(length):
                x = int(start_x + t * np.cos(angle))
                y = int(start_y + t * np.sin(angle))
                if 0 <= x < img_size and 0 <= y < img_size:
                    pred_mask[x, y] = True
                    if np.random.random() > 0.3:  # 70% overlap with GT
                        gt_mask[x, y] = True

        # Calculate metrics
        tp = (pred_mask & gt_mask).sum()
        fp = (pred_mask & ~gt_mask).sum()
        fn = (~pred_mask & gt_mask).sum()

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        # Create confidence map
        confidence_map = np.random.random((img_size, img_size))
        confidence_map[pred_mask] = np.random.uniform(
            0.6, 1.0, pred_mask.sum()
        )

        results.append(
            {
                "original_image": original_img,
                "prediction_mask": pred_mask,
                "ground_truth_mask": gt_mask,
                "confidence_map": confidence_map,
                "metrics": {
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "dice": dice,
                },
            }
        )

    return results


def demonstrate_interactive_training_curves(
    visualizer: InteractivePlotlyVisualizer,
) -> None:
    """Demonstrate interactive training curves."""
    logger.info("Creating interactive training curves...")

    # Create interactive training curves
    training_data = create_sample_training_data()
    metrics_data = {
        "loss": [m["loss"] for m in training_data["metrics"]],
        "val_loss": [m["val_loss"] for m in training_data["metrics"]],
        "iou": [m["iou"] for m in training_data["metrics"]],
        "val_iou": [m["val_iou"] for m in training_data["metrics"]],
    }
    epochs = list(range(1, len(training_data["metrics"]) + 1))

    _ = visualizer.create_interactive_training_curves(
        metrics_data=metrics_data,
        epochs=epochs,
        save_path=Path("outputs/demo_interactive/training_curves"),
    )

    logger.info("✅ Interactive training curves created successfully")


def demonstrate_interactive_prediction_grid(
    visualizer: InteractivePlotlyVisualizer,
) -> None:
    """Demonstrate interactive prediction grid."""
    logger.info("Creating interactive prediction grid...")

    # Create interactive prediction grid
    prediction_data = create_sample_prediction_data()

    # Create interactive prediction grid
    _ = visualizer.create_interactive_prediction_grid(
        results=prediction_data,
        max_images=6,
        show_metrics=True,
        save_path=Path("outputs/demo_interactive/prediction_grid"),
    )

    logger.info("✅ Interactive prediction grid created successfully")


def demonstrate_3d_confidence_map(
    visualizer: InteractivePlotlyVisualizer,
) -> None:
    """Demonstrate 3D confidence map."""
    logger.info("Creating 3D confidence map...")

    prediction_data = create_sample_prediction_data()

    # Create 3D confidence map for first sample
    _ = visualizer.create_3d_confidence_map(
        result=prediction_data[0],
        save_path=Path("outputs/demo_interactive/3d_confidence_map"),
    )

    logger.info("✅ 3D confidence map created successfully")


def demonstrate_dynamic_error_analysis(
    visualizer: InteractivePlotlyVisualizer,
) -> None:
    """Demonstrate dynamic error analysis."""
    logger.info("Creating dynamic error analysis...")

    prediction_data = create_sample_prediction_data()

    # Create dynamic error analysis for first sample
    _ = visualizer.create_dynamic_error_analysis(
        result=prediction_data[0],
        save_path=Path("outputs/demo_interactive/error_analysis"),
    )

    logger.info("✅ Dynamic error analysis created successfully")


def demonstrate_real_time_dashboard(
    visualizer: InteractivePlotlyVisualizer,
) -> None:
    """Demonstrate real-time training dashboard."""
    logger.info("Creating real-time training dashboard...")

    training_data = create_sample_training_data()

    # Create real-time training dashboard
    _ = visualizer.create_real_time_training_dashboard(
        training_data=training_data,
        save_path=Path("outputs/demo_interactive/real_time_dashboard"),
    )


def demonstrate_template_integration() -> None:
    """Demonstrate template integration with interactive visualizations."""
    logger.info("Demonstrating template integration...")

    # Create visualizer with training template and extended export formats
    training_template = TrainingVisualizationTemplate(
        {
            "figure_size": (12, 8),
            "color_palette": "viridis",
            "line_width": 3,
            "font_size": 14,
            "dpi": 100,
            "grid_alpha": 0.3,
            "title_font_size": 16,
            "legend_font_size": 12,
        }
    )

    training_visualizer = InteractivePlotlyVisualizer(
        template=training_template,
        export_formats=["html", "png", "pdf", "svg", "jpg", "json"],
    )

    # Create visualizer with prediction template
    prediction_template = PredictionVisualizationTemplate(
        {
            "figure_size": (10, 8),
            "color_palette": "plasma",
            "line_width": 2,
            "font_size": 12,
            "dpi": 100,
            "grid_alpha": 0.2,
            "title_font_size": 14,
            "legend_font_size": 10,
        }
    )

    prediction_visualizer = InteractivePlotlyVisualizer(
        template=prediction_template,
        export_formats=["html", "png", "svg", "json"],
    )

    # Test multi-format export with training data
    training_data = create_sample_training_data()
    metrics_data = {
        "loss": [m["loss"] for m in training_data["metrics"]],
        "val_loss": [m["val_loss"] for m in training_data["metrics"]],
        "iou": [m["iou"] for m in training_data["metrics"]],
        "val_iou": [m["val_iou"] for m in training_data["metrics"]],
    }
    epochs = list(range(1, len(training_data["metrics"]) + 1))

    _ = training_visualizer.create_interactive_training_curves(
        metrics_data=metrics_data,
        epochs=epochs,
        save_path=Path("outputs/demo_interactive/multi_format_training"),
    )

    # Test multi-format export with prediction data
    prediction_data = create_sample_prediction_data()

    _ = prediction_visualizer.create_interactive_prediction_grid(
        results=prediction_data,
        max_images=4,
        show_metrics=True,
        save_path=Path("outputs/demo_interactive/multi_format_prediction"),
    )

    logger.info("✅ Template integration with multi-format export completed")


def main() -> None:
    """Main demonstration function."""
    logger.info("🚀 Starting Interactive Plotly Visualization Demo")

    # Create output directory
    output_dir = Path("outputs/demo_interactive")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer with default settings
    visualizer = InteractivePlotlyVisualizer(
        export_formats=["html", "png", "pdf"]
    )

    try:
        # Demonstrate all interactive features
        demonstrate_interactive_training_curves(visualizer)
        demonstrate_interactive_prediction_grid(visualizer)
        demonstrate_3d_confidence_map(visualizer)
        demonstrate_dynamic_error_analysis(visualizer)
        demonstrate_real_time_dashboard(visualizer)
        demonstrate_template_integration()

        logger.info(
            "🎉 All interactive demonstrations completed successfully!"
        )
        logger.info(f"📁 Output files saved to: {output_dir.absolute()}")

        # List generated files
        generated_files = list(output_dir.rglob("*"))
        logger.info(f"📊 Generated {len(generated_files)} interactive files:")
        for file in generated_files:
            if file.is_file():
                logger.info(f"   - {file.name}")

    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
