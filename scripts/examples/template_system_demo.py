"""Template system demonstration script.

This script demonstrates the visualization template system
including base templates, training templates, and prediction templates.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crackseg.evaluation.visualization.templates import (
    PredictionVisualizationTemplate,
    TrainingVisualizationTemplate,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data() -> dict[str, np.ndarray]:
    """Create sample data for demonstration.

    Returns:
        Dictionary containing sample training data.
    """
    epochs = np.arange(1, 101)

    # Create sample training curves
    loss = 2.0 * np.exp(-epochs / 30) + 0.1 * np.random.randn(100)
    accuracy = 0.95 * (1 - np.exp(-epochs / 25)) + 0.02 * np.random.randn(100)
    precision = 0.92 * (1 - np.exp(-epochs / 20)) + 0.03 * np.random.randn(100)
    recall = 0.88 * (1 - np.exp(-epochs / 35)) + 0.04 * np.random.randn(100)

    return {
        "epochs": epochs,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def demonstrate_base_template() -> None:
    """Demonstrate base template functionality."""
    logger.info("🔧 Demonstrating Base Template System")

    # Create training template with custom config
    custom_config = {
        "figure_size": [10, 6],
        "dpi": 150,
        "color_palette": "Set2",
        "grid_alpha": 0.4,
        "line_width": 3,
        "font_size": 14,
        "title_font_size": 16,
        "legend_font_size": 12,
    }

    template = TrainingVisualizationTemplate(custom_config)

    # Create sample plot
    data = create_sample_data()
    fig, ax = plt.subplots(figsize=template.config["figure_size"])

    ax.plot(data["epochs"], data["loss"], label="Training Loss", linewidth=3)
    ax.plot(data["epochs"], data["accuracy"], label="Accuracy", linewidth=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Sample Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.4)

    # Apply template styling
    styled_fig = template.apply_template(fig)

    # Save demonstration
    output_dir = Path("artifacts") / "template_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    styled_fig.savefig(
        output_dir / "base_template_demo.png", dpi=150, bbox_inches="tight"
    )
    logger.info(
        "✅ Base template demo saved to: "
        f"{output_dir / 'base_template_demo.png'}"
    )

    plt.close()


def demonstrate_training_template() -> None:
    """Demonstrate training template functionality."""
    logger.info("📊 Demonstrating Training Template System")

    # Create training template
    template = TrainingVisualizationTemplate()

    # Show configuration structure
    logger.info("📋 Training Template Configuration:")
    logger.info(f"  - Figure size: {template.config['figure_size']}")
    logger.info(f"  - DPI: {template.config['dpi']}")
    logger.info(f"  - Color palette: {template.config['color_palette']}")

    # Show specialized configs
    training_config = template.get_training_curves_config()
    logger.info(
        "  - Training curves subplot layout: "
        f"{training_config['subplot_layout']}"
    )
    logger.info(
        "  - Available metric colors: "
        f"{list(training_config['metric_colors'].keys())}"
    )

    # Create multi-subplot demonstration
    data = create_sample_data()
    fig, axes = plt.subplots(2, 2, figsize=template.config["figure_size"])

    # Loss subplot
    axes[0, 0].plot(
        data["epochs"],
        data["loss"],
        color=training_config["metric_colors"]["loss"],
    )
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].grid(alpha=template.config["grid_alpha"])

    # Accuracy subplot
    axes[0, 1].plot(
        data["epochs"],
        data["accuracy"],
        color=training_config["metric_colors"]["accuracy"],
    )
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].grid(alpha=template.config["grid_alpha"])

    # Precision subplot
    axes[1, 0].plot(
        data["epochs"],
        data["precision"],
        color=training_config["metric_colors"]["precision"],
    )
    axes[1, 0].set_title("Precision")
    axes[1, 0].grid(alpha=template.config["grid_alpha"])

    # Recall subplot
    axes[1, 1].plot(
        data["epochs"],
        data["recall"],
        color=training_config["metric_colors"]["recall"],
    )
    axes[1, 1].set_title("Recall")
    axes[1, 1].grid(alpha=template.config["grid_alpha"])

    fig.suptitle(
        "Training Metrics Overview",
        fontsize=template.config["title_font_size"],
    )
    fig.tight_layout()

    # Apply template styling
    styled_fig = template.apply_template(fig)

    # Save demonstration
    output_dir = Path("artifacts") / "template_demo"
    styled_fig.savefig(
        output_dir / "training_template_demo.png", dpi=300, bbox_inches="tight"
    )
    logger.info(
        "✅ Training template demo saved to: "
        f"{output_dir / 'training_template_demo.png'}"
    )

    plt.close()


def demonstrate_prediction_template() -> None:
    """Demonstrate prediction template functionality."""
    logger.info("🎯 Demonstrating Prediction Template System")

    # Create prediction template
    template = PredictionVisualizationTemplate()

    # Show configuration structure
    logger.info("📋 Prediction Template Configuration:")
    logger.info(f"  - Figure size: {template.config['figure_size']}")
    logger.info(f"  - DPI: {template.config['dpi']}")
    logger.info(f"  - Color palette: {template.config['color_palette']}")

    # Show specialized configs
    comparison_config = template.get_comparison_grid_config()
    logger.info(
        f"  - Comparison grid layout: {comparison_config['grid_layout']}"
    )
    logger.info(f"  - Image size: {comparison_config['image_size']}")

    confidence_config = template.get_confidence_map_config()
    logger.info(
        f"  - Confidence map colormap: {confidence_config['colormap']}"
    )
    logger.info(f"  - Contour levels: {confidence_config['contour_levels']}")

    # Create sample confidence map
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)

    # Create sample confidence values
    confidence = np.exp(
        -((X - 5) ** 2 + (Y - 5) ** 2) / 8
    ) + 0.1 * np.random.randn(100, 100)
    confidence = np.clip(confidence, 0, 1)

    fig, ax = plt.subplots(figsize=template.config["figure_size"])

    im = ax.imshow(
        confidence,
        cmap=confidence_config["colormap"],
        extent=(0, 10, 0, 10),
        origin="lower",
    )
    ax.set_title("Sample Confidence Map")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    if confidence_config["show_colorbar"]:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Confidence")

    # Apply template styling
    styled_fig = template.apply_template(fig)

    # Save demonstration
    output_dir = Path("artifacts") / "template_demo"
    styled_fig.savefig(
        output_dir / "prediction_template_demo.png",
        dpi=300,
        bbox_inches="tight",
    )
    logger.info(
        "✅ Prediction template demo saved to: "
        f"{output_dir / 'prediction_template_demo.png'}"
    )

    plt.close()


def demonstrate_template_customization() -> None:
    """Demonstrate template customization capabilities."""
    logger.info("🎨 Demonstrating Template Customization")

    # Create base template
    template = TrainingVisualizationTemplate()

    # Show original config
    logger.info("📋 Original Configuration:")
    logger.info(f"  - Figure size: {template.config['figure_size']}")
    logger.info(f"  - Color palette: {template.config['color_palette']}")

    # Customize template
    template.update_config(
        {
            "figure_size": [14, 10],
            "color_palette": "Set3",
            "line_width": 3,
            "title_font_size": 18,
        }
    )

    # Update specific training curves config
    template.update_training_curves_config(
        {
            "subplot_layout": [1, 3],
            "show_grid": False,
        }
    )

    logger.info("📋 Updated Configuration:")
    logger.info(f"  - Figure size: {template.config['figure_size']}")
    logger.info(f"  - Color palette: {template.config['color_palette']}")
    logger.info(f"  - Line width: {template.config['line_width']}")

    # Create demonstration plot
    data = create_sample_data()
    fig, axes = plt.subplots(1, 3, figsize=template.config["figure_size"])

    metrics = ["loss", "accuracy", "precision"]
    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1"]

    for i, (metric, color) in enumerate(zip(metrics, colors, strict=False)):
        axes[i].plot(data["epochs"], data[metric], color=color, linewidth=3)
        axes[i].set_title(metric.title())
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Value")
        if not template.get_training_curves_config()["show_grid"]:
            axes[i].grid(False)

    fig.suptitle(
        "Customized Training Template",
        fontsize=template.config["title_font_size"],
    )
    fig.tight_layout()

    # Apply template styling
    styled_fig = template.apply_template(fig)

    # Save demonstration
    output_dir = Path("artifacts") / "template_demo"
    styled_fig.savefig(
        output_dir / "customized_template_demo.png",
        dpi=300,
        bbox_inches="tight",
    )
    logger.info(
        "✅ Customized template demo saved to: "
        f"{output_dir / 'customized_template_demo.png'}"
    )

    plt.close()


def main() -> None:
    """Run template system demonstration."""
    logger.info("🚀 Starting Template System Demonstration")

    # Create output directory
    output_dir = Path("artifacts") / "template_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Demonstrate each template type
        demonstrate_base_template()
        demonstrate_training_template()
        demonstrate_prediction_template()
        demonstrate_template_customization()

        logger.info("✅ Template system demonstration completed successfully!")
        logger.info(f"📁 All outputs saved to: {output_dir.absolute()}")

    except Exception as e:
        logger.error(f"❌ Error during template demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
